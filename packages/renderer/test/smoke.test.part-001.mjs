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
export const execFileAsync = promisify(execFile)
export const REPO_ROOT = fileURLToPath(new URL('../../../', import.meta.url))
export const require = createRequire(import.meta.url)
export const native = require('../native.js')

export const {
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

    const modifierEvents = []
    const modifierManager = {
      addHandler() {},
      resolveURL(url) {
        modifierEvents.push(['resolve', url])
        return 'textures/tex.png'
      },
      itemStart(url) {
        modifierEvents.push(['start', url])
      },
      itemEnd(url) {
        modifierEvents.push(['end', url])
      },
    }
    const modifierLoader = createEncodedImageTextureLoader(dir, modifierManager)
    const modifiedTexture = await modifierLoader.loadAsync('virtual.png')
    assert.deepEqual(Buffer.from(modifiedTexture.image), imageBytes)
    assert.deepEqual(modifierEvents, [
      ['resolve', 'virtual.png'],
      ['start', 'textures/tex.png'],
      ['end', 'textures/tex.png'],
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

    const fixtureRoot = fileURLToPath(new URL('./fixtures/', import.meta.url))
    const directRootCases = [
      ['absolute', fixtureRoot],
      ['relative', path.relative(process.cwd(), fixtureRoot) || '.'],
      ['file URL', pathToFileURL(fixtureRoot).href],
    ]
    for (const [label, rootDir] of directRootCases) {
      const rootedLoaderBundle = await createNodeGltfLoader(rootDir)
      const directLoadedGltf = await new Promise((resolve, reject) => {
        rootedLoaderBundle.loader.load('simple-triangle.gltf', resolve, undefined, reject)
      })
      assert.equal(directLoadedGltf.scene.name, 'SimpleTriangleScene', `${label} root should support direct loader.load()`)
    }
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
