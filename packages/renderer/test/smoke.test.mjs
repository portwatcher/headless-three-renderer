import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { createRequire } from 'node:module'
import os from 'node:os'
import path from 'node:path'
import { pathToFileURL } from 'node:url'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { assertValidPng, parsePngDimensions } from './helpers.mjs'

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
    const imageBytes = Buffer.from(
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR42mP8z8BQDwAFgwJ/l6g+WQAAAABJRU5ErkJggg==',
      'base64',
    )
    await writeFile(imagePath, imageBytes)

    const loader = createEncodedImageTextureLoader(dir)
    const texture = await new Promise((resolve, reject) => {
      loader.load('tex.png', resolve, undefined, reject)
    })

    assert.equal(texture.isTexture, true)
    assert.deepEqual(Buffer.from(texture.image), imageBytes)
    assert.equal(texture.source.data, texture.image)
    assert.equal(resolveLocalAssetPath('tex.png', dir), imagePath)
    assert.equal(resolveLocalAssetPath(imagePath, dir), imagePath)
    assert.equal(resolveLocalAssetPath(pathToFileURL(imagePath).href, dir), imagePath)

    const dataUriTexture = await new Promise((resolve, reject) => {
      loader.load(`data:image/png;base64,${imageBytes.toString('base64')}`, resolve, undefined, reject)
    })
    assert.deepEqual(Buffer.from(dataUriTexture.image), imageBytes)
    assert.equal(dataUriTexture.source.data, dataUriTexture.image)

    const blobUrl = URL.createObjectURL(new Blob([imageBytes], { type: 'image/png' }))
    try {
      const blobTexture = await new Promise((resolve, reject) => {
        loader.load(blobUrl, resolve, undefined, reject)
      })
      assert.deepEqual(Buffer.from(blobTexture.image), imageBytes)
      assert.equal(blobTexture.source.data, blobTexture.image)
    } finally {
      URL.revokeObjectURL(blobUrl)
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

test('Node loader helper path and option containers fail clearly', async () => {
  assert.throws(
    () => createEncodedImageTextureLoader(123),
    /rootDir must be a string/i,
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
  const windowsPath = String.raw`C:\assets\tex.png`
  assert.equal(resolveLocalAssetPath(windowsPath, process.cwd()), path.normalize(windowsPath))

  await assert.rejects(
    () => createNodeGltfLoader(123),
    /rootDir must be a string/i,
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
