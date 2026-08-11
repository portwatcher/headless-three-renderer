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
import { Renderer, applyVrmAnimation, createEncodedImageTextureLoader, createNodeGltfLoader, loadGltfFromFile, loadVrmAnimationFromFile, loadVrmFromFile, render, resolveLocalAssetPath } from './smoke.test.part-001.mjs'
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
    () => createEncodedImageTextureLoader(process.cwd(), { addHandler() {}, resolveURL: 'url' }),
    /manager\.resolveURL must be a function when provided/i,
  )
  for (const method of ['itemStart', 'itemEnd', 'itemError']) {
    assert.throws(
      () => createEncodedImageTextureLoader(process.cwd(), { addHandler() {}, [method]: 'hook' }),
      new RegExp(`manager\\.${method} must be a function when provided`, 'i'),
    )
  }
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
    () => imageLoader.setPath('https://example.com/assets/'),
    /loaderPath is not a local directory path/i,
  )
  assert.throws(
    () => imageLoader.load(123),
    /url must be a string/i,
  )
  await assert.rejects(
    () => imageLoader.loadAsync(123),
    /url must be a string/i,
  )
  await assert.rejects(
    () => createEncodedImageTextureLoader(process.cwd(), {
      addHandler() {},
      resolveURL() {
        return 1
      },
    }).loadAsync('tex.png'),
    /manager\.resolveURL return value must be a string/i,
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
    () => createNodeGltfLoader(process.cwd(), { manager: { addHandler() {}, itemStart: 'hook' } }),
    /options\.manager\.itemStart must be a function when provided/i,
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

test('reusable renderer omits repeated native mesh arrays after cache seed render', () => {
  const scene = new THREE.Scene()
  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(1, 1, 1),
    new THREE.MeshBasicMaterial({ color: 0xff3355 }),
  )
  scene.add(mesh)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const r = new Renderer()
  const capturedMeshes = []
  const originalRender = r.native.render.bind(r.native)
  r.native.render = (nativeScene, nativeCamera) => {
    capturedMeshes.push(nativeScene.meshes.map((nativeMesh) => ({
      nativeMeshKey: nativeMesh.nativeMeshKey,
      nativeVertexCount: nativeMesh.nativeVertexCount,
      nativeIndexCount: nativeMesh.nativeIndexCount,
      positionsLength: nativeMesh.positions?.length ?? 0,
      indicesLength: nativeMesh.indices?.length ?? 0,
    })))
    return originalRender(nativeScene, nativeCamera)
  }

  const first = r.render(scene, camera, { width: 64, height: 64, format: 'rgba' })
  mesh.position.x = 0.15
  const second = r.render(scene, camera, { width: 64, height: 64, format: 'rgba' })

  assert.equal(first.length, 64 * 64 * 4)
  assert.equal(second.length, 64 * 64 * 4)
  assert.equal(capturedMeshes.length, 2)
  assert.ok(capturedMeshes[0][0].nativeMeshKey > 0)
  assert.ok(capturedMeshes[0][0].positionsLength > 0)
  assert.ok(capturedMeshes[0][0].indicesLength > 0)
  assert.equal(capturedMeshes[1][0].nativeMeshKey, capturedMeshes[0][0].nativeMeshKey)
  assert.equal(capturedMeshes[1][0].nativeVertexCount, capturedMeshes[0][0].nativeVertexCount)
  assert.equal(capturedMeshes[1][0].nativeIndexCount, capturedMeshes[0][0].nativeIndexCount)
  assert.equal(capturedMeshes[1][0].positionsLength, 0)
  assert.equal(capturedMeshes[1][0].indicesLength, 0)
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
