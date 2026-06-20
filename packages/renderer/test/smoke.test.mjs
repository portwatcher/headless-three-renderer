import test from 'node:test'
import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { createRequire } from 'node:module'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { promisify } from 'node:util'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { assertValidPng, parsePngDimensions } from './helpers.mjs'

const execFileAsync = promisify(execFile)
const REPO_ROOT = fileURLToPath(new URL('../../../', import.meta.url))
const require = createRequire(import.meta.url)
const native = require('../native.js')

const {
  Renderer,
  applyVrmAnimation,
  createNodeGltfLoader,
  createEncodedImageTextureLoader,
  installLocalFileFetch,
  loadGltfFromFile,
  loadVrmAnimationFromFile,
  loadVrmFromFile,
  render,
  resolveLocalAssetPath,
} = pkg

test('module exports Renderer class and render function', () => {
  assert.equal(typeof Renderer, 'function')
  assert.equal(typeof render, 'function')
  assert.equal(typeof applyVrmAnimation, 'function')
  assert.equal(typeof createEncodedImageTextureLoader, 'function')
  assert.equal(typeof installLocalFileFetch, 'function')
  assert.equal(typeof loadGltfFromFile, 'function')
  assert.equal(typeof loadVrmAnimationFromFile, 'function')
  assert.equal(typeof loadVrmFromFile, 'function')
  assert.equal(typeof resolveLocalAssetPath, 'function')
})

test('native color-space diagnostics list accepted linear aliases', () => {
  const camera = { width: 1, height: 1 }
  assert.throws(
    () => native.renderNative({ width: 1, height: 1, outputColorSpace: 'display-p3' }, camera),
    /unsupported scene\.outputColorSpace `display-p3`; expected `srgb`, `srgb-linear`, `linear-srgb`, `linearsrgb`, or `linear`/i,
  )
  assert.throws(
    () => native.renderNative({
      width: 1,
      height: 1,
      environmentMap: Buffer.from([255, 255, 255, 255]),
      environmentMapWidth: 1,
      environmentMapHeight: 1,
      environmentMapColorSpace: 'display-p3',
    }, camera),
    /unsupported scene\.environmentMapColorSpace `display-p3`; expected `srgb`, `srgb-linear`, `linear-srgb`, `linearsrgb`, or `linear`/i,
  )
})

test('Node loader helpers expose encoded image buffers and local file fetch', async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'headless-three-loader-'))
  try {
    const imagePath = path.join(dir, 'tex.png')
    const textureDir = path.join(dir, 'textures')
    const nestedImagePath = path.join(textureDir, 'tex.png')
    const imageBytes = Buffer.from(
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR42mP8z8BQDwAFgwJ/l6g+WQAAAABJRU5ErkJggg==',
      'base64',
    )
    await mkdir(textureDir)
    await writeFile(imagePath, imageBytes)
    await writeFile(nestedImagePath, imageBytes)

    const loader = createEncodedImageTextureLoader(dir)
    const texture = await new Promise((resolve, reject) => {
      loader.load('tex.png', resolve, undefined, reject)
    })

    assert.equal(texture.isTexture, true)
    assert.deepEqual(Buffer.from(texture.image), imageBytes)
    assert.equal(texture.source.data, texture.image)

    const asyncTexture = await loader.loadAsync('tex.png')
    assert.equal(asyncTexture.isTexture, true)
    assert.deepEqual(Buffer.from(asyncTexture.image), imageBytes)
    assert.equal(asyncTexture.source.data, asyncTexture.image)

    const fileUrlRootLoader = createEncodedImageTextureLoader(pathToFileURL(dir).href)
    const fileUrlRootTexture = await fileUrlRootLoader.loadAsync('tex.png')
    assert.deepEqual(Buffer.from(fileUrlRootTexture.image), imageBytes)

    const pathLoader = createEncodedImageTextureLoader(dir)
    pathLoader.setPath('textures')
    const pathTexture = await pathLoader.loadAsync('tex.png')
    assert.deepEqual(Buffer.from(pathTexture.image), imageBytes)

    const fileUrlPathLoader = createEncodedImageTextureLoader(dir)
    fileUrlPathLoader.setPath(pathToFileURL(textureDir).href)
    const fileUrlPathTexture = await fileUrlPathLoader.loadAsync('tex.png')
    assert.deepEqual(Buffer.from(fileUrlPathTexture.image), imageBytes)

    const managerEvents = []
    const manager = {
      addHandler() {},
      itemStart(url) {
        managerEvents.push(['start', url])
      },
      itemEnd(url) {
        managerEvents.push(['end', url])
      },
      itemError(url) {
        managerEvents.push(['error', url])
      },
    }
    const managedLoader = createEncodedImageTextureLoader(dir, manager)
    const managedTexture = await managedLoader.loadAsync('tex.png')
    assert.deepEqual(Buffer.from(managedTexture.image), imageBytes)
    assert.deepEqual(managerEvents, [
      ['start', 'tex.png'],
      ['end', 'tex.png'],
    ])

    managerEvents.length = 0
    await assert.rejects(
      () => managedLoader.loadAsync('missing-manager.png'),
      /ENOENT|no such file/i,
    )
    assert.deepEqual(managerEvents, [
      ['start', 'missing-manager.png'],
      ['error', 'missing-manager.png'],
      ['end', 'missing-manager.png'],
    ])

    assert.equal(resolveLocalAssetPath('tex.png', dir), imagePath)
    assert.equal(resolveLocalAssetPath('tex.png', pathToFileURL(dir).href), imagePath)
    assert.equal(resolveLocalAssetPath(imagePath, dir), imagePath)
    assert.equal(resolveLocalAssetPath(pathToFileURL(imagePath).href, dir), imagePath)

    const dataUriTexture = await new Promise((resolve, reject) => {
      loader.load(`data:image/png;base64,${imageBytes.toString('base64')}`, resolve, undefined, reject)
    })
    assert.deepEqual(Buffer.from(dataUriTexture.image), imageBytes)
    assert.equal(dataUriTexture.source.data, dataUriTexture.image)
    assert.throws(
      () => loader.load('data:text/plain,not-an-image'),
      /Data URI texture is not a supported encoded image/i,
    )
    assert.throws(
      () => loader.load('data:image/png;base64AAAA'),
      /Data URI texture is missing a comma separator/i,
    )

    const blobUrl = URL.createObjectURL(new Blob([imageBytes], { type: 'image/png' }))
    try {
      const blobTexture = await new Promise((resolve, reject) => {
        pathLoader.load(blobUrl, resolve, undefined, reject)
      })
      assert.deepEqual(Buffer.from(blobTexture.image), imageBytes)
      assert.equal(blobTexture.source.data, blobTexture.image)
    } finally {
      URL.revokeObjectURL(blobUrl)
    }

    const unsupportedBlobUrl = URL.createObjectURL(new Blob(['not an image'], { type: 'text/plain' }))
    try {
      await new Promise((resolve, reject) => {
        loader.load(unsupportedBlobUrl, () => reject(new Error('unsupported Blob URL should not load')), undefined, (error) => {
          assert.match(String(error?.message ?? error), /Blob URL texture has unsupported content type "text\/plain"/i)
          resolve()
        })
      })
    } finally {
      URL.revokeObjectURL(unsupportedBlobUrl)
    }

    await new Promise((resolve, reject) => {
      loader.load('missing.png', () => reject(new Error('missing image should not load')), undefined, (error) => {
        assert.match(String(error?.message ?? error), /ENOENT|no such file/i)
        resolve()
      })
    })

    const unhandledRejections = []
    const onUnhandledRejection = (reason) => {
      unhandledRejections.push(reason)
    }
    process.on('unhandledRejection', onUnhandledRejection)
    try {
      loader.load('missing-without-error-callback.png')
      await new Promise((resolve) => setTimeout(resolve, 20))
      assert.deepEqual(unhandledRejections, [])
    } finally {
      process.off('unhandledRejection', onUnhandledRejection)
    }

    installLocalFileFetch()
    const response = await fetch(pathToFileURL(imagePath).href)
    assert.deepEqual(Buffer.from(await response.arrayBuffer()), imageBytes)

    const fileUrlRootBundle = await createNodeGltfLoader(pathToFileURL(dir).href, {
      installFetch: false,
      registerTextureHandlers: false,
    })
    assert.equal(fileUrlRootBundle.rootDir, dir)

    const fileUrlGltf = await loadGltfFromFile(
      new URL('./fixtures/simple-triangle.gltf', import.meta.url).href,
      { installFetch: false },
    )
    assert.equal(fileUrlGltf.scene.name, 'SimpleTriangleScene')
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
})

test('Node glTF loader option booleans fail clearly', async () => {
  await assert.rejects(
    () => createNodeGltfLoader(process.cwd(), { installFetch: 'yes' }),
    /options\.installFetch must be a boolean/i,
  )

  await assert.rejects(
    () => createNodeGltfLoader(process.cwd(), {
      installFetch: false,
      registerTextureHandlers: 'yes',
    }),
    /options\.registerTextureHandlers must be a boolean/i,
  )
})

test('local glTF example renders a committed fixture from the repo root', async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'headless-three-example-'))
  try {
    const outputPath = path.join(dir, 'render.png')
    const { stdout } = await execFileAsync(process.execPath, [
      path.join(REPO_ROOT, 'examples', 'render-gltf.mjs'),
      path.join(REPO_ROOT, 'packages', 'renderer', 'test', 'fixtures', 'simple-triangle.gltf'),
      outputPath,
    ], {
      cwd: REPO_ROOT,
      env: {
        ...process.env,
        WIDTH: '64',
        HEIGHT: '48',
      },
    })
    assert.match(stdout, /Rendered .*simple-triangle\.gltf.* \(64x48\)/)

    const image = await readFile(outputPath)
    assertValidPng(image, { width: 64, height: 48 })
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
})

test('local VRM example resolves optional Pixiv packages from the caller project', async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'headless-three-vrm-example-'))
  try {
    const projectDir = path.join(dir, 'project')
    const pixivDir = path.join(projectDir, 'node_modules', '@pixiv', 'three-vrm')
    const pixivAnimationDir = path.join(projectDir, 'node_modules', '@pixiv', 'three-vrm-animation')
    await mkdir(pixivDir, { recursive: true })
    await mkdir(pixivAnimationDir, { recursive: true })
    await writeFile(path.join(projectDir, 'package.json'), '{"type":"commonjs"}\n')
    await writeFile(path.join(pixivDir, 'package.json'), '{"main":"index.cjs"}\n')
    await writeFile(path.join(pixivDir, 'index.cjs'), `
class VRMLoaderPlugin {
  constructor(parser) {
    this.parser = parser
    this.name = 'FakeVRMLoaderPlugin'
  }

  afterRoot(gltf) {
    gltf.userData = gltf.userData || {}
    gltf.userData.vrm = {
      scene: gltf.scene,
      update() {},
    }
  }
}

const VRMUtils = {
  removeUnnecessaryVertices() {},
  removeUnnecessaryJoints() {},
}

module.exports = { VRMLoaderPlugin, VRMUtils }
`)
    await writeFile(path.join(pixivAnimationDir, 'package.json'), '{"main":"index.cjs"}\n')
    await writeFile(path.join(pixivAnimationDir, 'index.cjs'), `
class VRMAnimationLoaderPlugin {
  constructor(parser) {
    this.parser = parser
    this.name = 'FakeVRMAnimationLoaderPlugin'
  }

  afterRoot(gltf) {
    gltf.userData = gltf.userData || {}
    gltf.userData.vrmAnimations = [
      { name: 'idle' },
      { name: 'dance' },
    ]
  }
}

function createVRMAnimationClip(animation) {
  if (!animation || animation.name !== 'dance') {
    throw new Error('Expected render-vrm.mjs to select animation index 1.')
  }
  return { name: 'selected-dance', tracks: [], duration: 0 }
}

module.exports = { VRMAnimationLoaderPlugin, createVRMAnimationClip }
`)

    const outputPath = path.join(dir, 'vrm-render.png')
    const animationPath = path.join(dir, 'fake-animation.vrma')
    await writeFile(
      animationPath,
      await readFile(path.join(REPO_ROOT, 'packages', 'renderer', 'test', 'fixtures', 'simple-triangle.gltf')),
    )
    const { stdout } = await execFileAsync(process.execPath, [
      path.join(REPO_ROOT, 'examples', 'render-vrm.mjs'),
      path.join(REPO_ROOT, 'packages', 'renderer', 'test', 'fixtures', 'simple-triangle.gltf'),
      animationPath,
      outputPath,
    ], {
      cwd: projectDir,
      env: {
        ...process.env,
        ANIMATION_INDEX: '1',
        TIME: '0.25',
        WIDTH: '64',
        HEIGHT: '48',
      },
    })
    assert.match(stdout, /Rendered .*simple-triangle\.gltf.*fake-animation\.vrma animation #1 at 0\.25s.* \(64x48\)/)

    const image = await readFile(outputPath)
    assertValidPng(image, { width: 64, height: 48 })
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
})

test('VRM animation helper creates a clip, seeks the mixer, and updates the avatar', async () => {
  const calls = []
  const scene = { name: 'avatar-scene' }
  const vrm = {
    scene,
    update(delta) {
      calls.push(['vrm.update', delta])
    },
  }
  const vrmAnimation = { name: 'wave' }

  class FakeAnimationMixer {
    constructor(root) {
      calls.push(['mixer.constructor', root])
    }

    clipAction(clip) {
      calls.push(['mixer.clipAction', clip])
      return {
        play() {
          calls.push(['action.play'])
        },
      }
    }

    setTime(time) {
      calls.push(['mixer.setTime', time])
    }
  }

  const result = await applyVrmAnimation(vrm, vrmAnimation, {
    AnimationMixer: FakeAnimationMixer,
    createVRMAnimationClip(animation, model) {
      calls.push(['createVRMAnimationClip', animation, model])
      return { animation, model }
    },
    time: 1.25,
    updateDelta: 0.1,
  })

  assert.equal(result.clip.animation, vrmAnimation)
  assert.equal(result.clip.model, vrm)
  assert.deepEqual(calls, [
    ['createVRMAnimationClip', vrmAnimation, vrm],
    ['mixer.constructor', scene],
    ['mixer.clipAction', result.clip],
    ['action.play'],
    ['mixer.setTime', 1.25],
    ['vrm.update', 0.1],
  ])
})

test('VRM animation helper supports mixer update fallback and VRM update opt-out', async () => {
  const calls = []
  const scene = { name: 'avatar-scene' }
  const vrm = {
    scene,
    update(delta) {
      calls.push(['vrm.update', delta])
    },
  }
  const vrmAnimation = { name: 'idle' }

  class UpdateOnlyAnimationMixer {
    constructor(root) {
      calls.push(['mixer.constructor', root])
    }

    clipAction(clip) {
      calls.push(['mixer.clipAction', clip])
      return {
        play() {
          calls.push(['action.play'])
        },
      }
    }

    update(delta) {
      calls.push(['mixer.update', delta])
    }
  }

  const result = await applyVrmAnimation(vrm, vrmAnimation, {
    AnimationMixer: UpdateOnlyAnimationMixer,
    createVRMAnimationClip(animation, model) {
      calls.push(['createVRMAnimationClip', animation, model])
      return { animation, model }
    },
    time: 0.75,
    updateDelta: 0.2,
    updateVrm: false,
  })

  assert.equal(result.clip.animation, vrmAnimation)
  assert.equal(result.clip.model, vrm)
  assert.deepEqual(calls, [
    ['createVRMAnimationClip', vrmAnimation, vrm],
    ['mixer.constructor', scene],
    ['mixer.clipAction', result.clip],
    ['action.play'],
    ['mixer.update', 0.75],
  ])
})

test('VRM animation helper accepts loaded VRM and VRMA glTF wrappers', async () => {
  const calls = []
  const scene = { name: 'wrapped-avatar-scene' }
  const vrm = { scene }
  const vrmAnimation = { name: 'wrapped-wave' }
  const vrmGltf = { userData: { vrm } }
  const animationGltf = { userData: { vrmAnimations: [vrmAnimation] } }

  class FakeAnimationMixer {
    constructor(root) {
      calls.push(['mixer.constructor', root])
    }

    clipAction(clip) {
      calls.push(['mixer.clipAction', clip])
      return {
        play() {
          calls.push(['action.play'])
        },
      }
    }

    setTime(time) {
      calls.push(['mixer.setTime', time])
    }
  }

  const result = await applyVrmAnimation(vrmGltf, animationGltf, {
    AnimationMixer: FakeAnimationMixer,
    createVRMAnimationClip(animation, model) {
      calls.push(['createVRMAnimationClip', animation, model])
      return { animation, model }
    },
    time: 2,
  })

  assert.equal(result.clip.animation, vrmAnimation)
  assert.equal(result.clip.model, vrm)
  assert.deepEqual(calls, [
    ['createVRMAnimationClip', vrmAnimation, vrm],
    ['mixer.constructor', scene],
    ['mixer.clipAction', result.clip],
    ['action.play'],
    ['mixer.setTime', 2],
  ])
})

test('VRM animation helper selects VRMA wrapper animations by index', async () => {
  const calls = []
  const scene = { name: 'indexed-avatar-scene' }
  const vrm = { scene }
  const idleAnimation = { name: 'idle' }
  const danceAnimation = { name: 'dance' }
  const vrmGltf = { userData: { vrm } }
  const animationGltf = { userData: { vrmAnimations: [idleAnimation, danceAnimation] } }

  class FakeAnimationMixer {
    constructor(root) {
      calls.push(['mixer.constructor', root])
    }

    clipAction(clip) {
      calls.push(['mixer.clipAction', clip])
      return {
        play() {
          calls.push(['action.play'])
        },
      }
    }

    setTime(time) {
      calls.push(['mixer.setTime', time])
    }
  }

  const result = await applyVrmAnimation(vrmGltf, animationGltf, {
    AnimationMixer: FakeAnimationMixer,
    animationIndex: 1,
    createVRMAnimationClip(animation, model) {
      calls.push(['createVRMAnimationClip', animation, model])
      return { animation, model }
    },
    time: 0.5,
  })

  assert.equal(result.clip.animation, danceAnimation)
  assert.equal(result.clip.model, vrm)
  assert.deepEqual(calls, [
    ['createVRMAnimationClip', danceAnimation, vrm],
    ['mixer.constructor', scene],
    ['mixer.clipAction', result.clip],
    ['action.play'],
    ['mixer.setTime', 0.5],
  ])
})

test('Node loader helper path and option containers fail clearly', async () => {
  assert.throws(
    () => createEncodedImageTextureLoader(123),
    /rootDir must be a string/i,
  )
  assert.throws(
    () => createEncodedImageTextureLoader(process.cwd(), {}),
    /manager must provide an addHandler\(\) function/i,
  )
  assert.throws(
    () => createEncodedImageTextureLoader('https://example.com/assets'),
    /rootDir is not a local directory path/i,
  )

  const imageLoader = createEncodedImageTextureLoader(process.cwd())
  assert.throws(
    () => imageLoader.setPath(123),
    /loaderPath must be a string/i,
  )
  assert.throws(
    () => imageLoader.load(123),
    /url must be a string/i,
  )
  await assert.rejects(
    () => imageLoader.loadAsync(123),
    /url must be a string/i,
  )
  assert.throws(
    () => imageLoader.load('tex.png', 'yes'),
    /onLoad must be a function/i,
  )
  assert.throws(
    () => imageLoader.load('tex.png', undefined, undefined, 'yes'),
    /onError must be a function/i,
  )
  assert.throws(
    () => resolveLocalAssetPath(123),
    /url must be a string/i,
  )
  assert.throws(
    () => resolveLocalAssetPath('tex.png', 123),
    /rootDir must be a string/i,
  )
  assert.throws(
    () => resolveLocalAssetPath('data:image/png;base64,AAAA'),
    /Data URI textures should be decoded/i,
  )
  assert.throws(
    () => resolveLocalAssetPath('https://example.com/tex.png'),
    /Remote texture URL is not a local file/i,
  )
  assert.throws(
    () => resolveLocalAssetPath('tex.png', 'https://example.com/assets'),
    /rootDir is not a local directory path/i,
  )
  const windowsPath = String.raw`C:\assets\tex.png`
  assert.equal(resolveLocalAssetPath(windowsPath, process.cwd()), path.normalize(windowsPath))

  await assert.rejects(
    () => createNodeGltfLoader(123),
    /rootDir must be a string/i,
  )
  await assert.rejects(
    () => createNodeGltfLoader('https://example.com/assets'),
    /rootDir is not a local directory path/i,
  )
  await assert.rejects(
    () => createNodeGltfLoader(process.cwd(), null),
    /options must be an object/i,
  )
  await assert.rejects(
    () => createNodeGltfLoader(process.cwd(), { configureLoader: 'yes' }),
    /options\.configureLoader must be a function/i,
  )
  await assert.rejects(
    () => createNodeGltfLoader(process.cwd(), { manager: {} }),
    /options\.manager must provide an addHandler\(\) function/i,
  )
  await assert.rejects(
    () => loadGltfFromFile(123),
    /filePath must be a string/i,
  )
  await assert.rejects(
    () => loadGltfFromFile('scene.gltf', null),
    /options must be an object/i,
  )
  await assert.rejects(
    () => loadGltfFromFile('scene.gltf', { rootDir: 123 }),
    /options\.rootDir must be a string/i,
  )
  await assert.rejects(
    () => loadGltfFromFile('scene.gltf', { rootDir: 'https://example.com/assets' }),
    /options\.rootDir is not a local directory path/i,
  )
  await assert.rejects(
    () => loadGltfFromFile('scene.gltf', { baseUrl: 123, installFetch: false }),
    /options\.baseUrl must be a string/i,
  )
  await assert.rejects(
    () => loadVrmFromFile(123, { VRMLoaderPlugin: class VRMLoaderPlugin {} }),
    /filePath must be a string/i,
  )
  await assert.rejects(
    () => loadVrmFromFile('avatar.vrm', { rootDir: 123 }),
    /options\.rootDir must be a string/i,
  )
  await assert.rejects(
    () => loadVrmFromFile('avatar.vrm', { VRMLoaderPlugin: 'yes' }),
    /options\.VRMLoaderPlugin must be a function/i,
  )
  await assert.rejects(
    () => loadVrmAnimationFromFile(123, { VRMAnimationLoaderPlugin: class VRMAnimationLoaderPlugin {} }),
    /filePath must be a string/i,
  )
  await assert.rejects(
    () => loadVrmAnimationFromFile('avatar.vrma', { VRMAnimationLoaderPlugin: 'yes' }),
    /options\.VRMAnimationLoaderPlugin must be a function/i,
  )

  class ValidAnimationMixer {
    clipAction() {
      return { play() {} }
    }

    setTime() {}
  }

  await assert.rejects(
    () => applyVrmAnimation({ scene: {} }, {}, {
      AnimationMixer: 'yes',
      createVRMAnimationClip() {},
    }),
    /options\.AnimationMixer must be a function/i,
  )
  await assert.rejects(
    () => applyVrmAnimation({ scene: {} }, {}, {
      AnimationMixer: ValidAnimationMixer,
      createVRMAnimationClip: 'yes',
    }),
    /options\.createVRMAnimationClip must be a function/i,
  )
  await assert.rejects(
    () => applyVrmAnimation(null, {}, {
      AnimationMixer: class FakeAnimationMixer {},
      createVRMAnimationClip() {},
    }),
    /vrm must be an object/i,
  )
  await assert.rejects(
    () => applyVrmAnimation({ scene: [] }, {}, {
      AnimationMixer: class FakeAnimationMixer {},
      createVRMAnimationClip() {},
    }),
    /vrm\.scene must be an object/i,
  )
  await assert.rejects(
    () => applyVrmAnimation({ userData: { vrm: { scene: [] } } }, {}, {
      AnimationMixer: class FakeAnimationMixer {},
      createVRMAnimationClip() {},
    }),
    /vrm\.userData\.vrm\.scene must be an object/i,
  )
  await assert.rejects(
    () => applyVrmAnimation({ scene: {} }, { userData: { vrmAnimations: {} } }, {
      AnimationMixer: ValidAnimationMixer,
      createVRMAnimationClip() {},
    }),
    /vrmAnimation\.userData\.vrmAnimations must be an array/i,
  )
  await assert.rejects(
    () => applyVrmAnimation({ scene: {} }, { userData: { vrmAnimations: [] } }, {
      AnimationMixer: ValidAnimationMixer,
      createVRMAnimationClip() {},
    }),
    /vrmAnimation\.userData\.vrmAnimations\[0\] must be an object/i,
  )
  await assert.rejects(
    () => applyVrmAnimation({ scene: {} }, { userData: { vrmAnimations: [{}] } }, {
      AnimationMixer: ValidAnimationMixer,
      animationIndex: 1,
      createVRMAnimationClip() {},
    }),
    /vrmAnimation\.userData\.vrmAnimations\[1\] must be an object/i,
  )
  await assert.rejects(
    () => applyVrmAnimation({ scene: {} }, {}, {
      AnimationMixer: ValidAnimationMixer,
      animationIndex: 1.5,
      createVRMAnimationClip() {},
    }),
    /options\.animationIndex must be a non-negative integer/i,
  )
  await assert.rejects(
    () => applyVrmAnimation({ scene: {} }, {}, {
      AnimationMixer: class FakeAnimationMixer {},
      createVRMAnimationClip() {},
      time: 'soon',
    }),
    /options\.time must be a finite non-negative number/i,
  )
  await assert.rejects(
    () => applyVrmAnimation({ scene: {} }, {}, {
      AnimationMixer: ValidAnimationMixer,
      createVRMAnimationClip() {},
      updateDelta: -1,
    }),
    /options\.updateDelta must be a finite non-negative number/i,
  )
  await assert.rejects(
    () => applyVrmAnimation({ scene: {} }, {}, {
      AnimationMixer: ValidAnimationMixer,
      createVRMAnimationClip() {},
      updateVrm: 'yes',
    }),
    /options\.updateVrm must be a boolean/i,
  )
  await assert.rejects(
    () => applyVrmAnimation({ scene: {}, update: 'yes' }, {}, {
      AnimationMixer: class FakeAnimationMixer {
        clipAction() {
          return { play() {} }
        }

        setTime() {}
      },
      createVRMAnimationClip() {},
    }),
    /vrm\.update must be a function/i,
  )
  await assert.rejects(
    () => applyVrmAnimation({ scene: {} }, {}, {
      AnimationMixer: class FakeAnimationMixer {},
      createVRMAnimationClip() {},
    }),
    /AnimationMixer must provide a clipAction\(\) function/i,
  )
  await assert.rejects(
    () => applyVrmAnimation({ scene: {} }, {}, {
      AnimationMixer: class FakeAnimationMixer {
        clipAction() {
          return {}
        }
      },
      createVRMAnimationClip() {},
    }),
    /AnimationMixer\.clipAction\(\) must return an action with play\(\)/i,
  )
  await assert.rejects(
    () => applyVrmAnimation({ scene: {} }, {}, {
      AnimationMixer: class FakeAnimationMixer {
        clipAction() {
          return { play() {} }
        }
      },
      createVRMAnimationClip() {},
    }),
    /AnimationMixer must provide setTime\(\) or update\(\)/i,
  )
})

test('renders a simple scene and returns a PNG buffer of the requested size', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.1, 0.1, 0.1)
  scene.add(
    new THREE.Mesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshBasicMaterial({ color: 0xff0000 }),
    ),
  )

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(2.5, 1.8, 3.2)
  camera.lookAt(0, 0, 0)

  const r = new Renderer()
  const buf = r.render(scene, camera, { width: 256, height: 256 })

  assert.ok(Buffer.isBuffer(buf), 'output should be a Buffer')
  assert.ok(buf.length > 0, 'output should be non-empty')
  assertValidPng(buf, { width: 256, height: 256 })
})

test('renderer is reusable across multiple calls', () => {
  const scene = new THREE.Scene()
  scene.add(
    new THREE.Mesh(new THREE.SphereGeometry(1, 16, 16), new THREE.MeshBasicMaterial({ color: 0x00ff00 })),
  )
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const r = new Renderer()
  for (let i = 0; i < 3; i++) {
    const buf = r.render(scene, camera, { width: 128, height: 128 })
    const { width, height } = parsePngDimensions(buf)
    assert.equal(width, 128)
    assert.equal(height, 128)
  }
})

test('top-level render() function works without a Renderer instance', () => {
  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial({ color: 0x0000ff })))
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const buf = render(scene, camera, { width: 64, height: 64 })
  assertValidPng(buf, { width: 64, height: 64 })
})

test('Object3D roots render without a wrapping Scene', () => {
  const root = new THREE.Group()
  root.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial({ color: 0x00ffaa })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const buf = render(root, camera, { width: 64, height: 64 })
  assertValidPng(buf, { width: 64, height: 64 })
})

test('different sizes produce correctly sized outputs', () => {
  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial()))
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const r = new Renderer()
  for (const [w, h] of [
    [100, 100],
    [320, 240],
    [512, 256],
  ]) {
    const buf = r.render(scene, camera, { width: w, height: h })
    assertValidPng(buf, { width: w, height: h })
  }
})
