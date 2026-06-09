import test from 'node:test'
import assert from 'node:assert/strict'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { assertValidPng, meanRgba, nonBackgroundRatio } from './helpers.mjs'

const { Renderer, renderToTarget } = pkg

const SIZE = 128
const BG = [26, 26, 26] // 0.1 * 255

function makeCamera() {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(2.5, 2, 3)
  camera.lookAt(0, 0, 0)
  return camera
}

function addLights(scene) {
  scene.add(new THREE.AmbientLight(0xffffff, 0.3))
  const dir = new THREE.DirectionalLight(0xffffff, 1.2)
  dir.position.set(3, 4, 2)
  scene.add(dir)
}

function renderRgba(scene, camera, options = {}) {
  const r = new Renderer()
  return r.render(scene, camera, { width: SIZE, height: SIZE, format: 'rgba', ...options })
}

function makeEnvironmentTexture() {
  const data = new Uint8Array([
    255, 255, 255, 255,
    64, 128, 255, 255,
    255, 180, 96, 255,
    18, 24, 36, 255,
  ])
  const texture = new THREE.DataTexture(data, 2, 2, THREE.RGBAFormat)
  texture.needsUpdate = true
  return texture
}

function solidTexture(r, g, b, a = 255) {
  const texture = new THREE.DataTexture(new Uint8Array([r, g, b, a]), 1, 1, THREE.RGBAFormat)
  texture.needsUpdate = true
  return texture
}

function meanAbsDiff(a, b) {
  assert.equal(a.length, b.length)
  let total = 0
  for (let i = 0; i < a.length; i += 4) {
    total += Math.abs(a[i] - b[i])
    total += Math.abs(a[i + 1] - b[i + 1])
    total += Math.abs(a[i + 2] - b[i + 2])
  }
  return total / ((a.length / 4) * 3)
}

test('rgba format returns raw pixel buffer of the expected byte length', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.1, 0.1, 0.1)
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial({ color: 0xff00ff })))

  const buf = renderRgba(scene, makeCamera())
  assert.equal(buf.length, SIZE * SIZE * 4, 'rgba buffer must be width*height*4 bytes')
})

test('MeshBasicMaterial renders foreground pixels distinct from background', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.1, 0.1, 0.1)
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial({ color: 0xffaa00 })))

  const rgba = renderRgba(scene, makeCamera())
  const ratio = nonBackgroundRatio(rgba, BG)
  assert.ok(ratio > 0.05, `expected mesh to cover >5% of frame, got ${(ratio * 100).toFixed(1)}%`)
  assert.ok(ratio < 0.95, `expected background to be visible, got ${(ratio * 100).toFixed(1)}% non-bg`)
})

test('different materials produce visibly different outputs', async () => {
  const camera = makeCamera()

  const sceneA = new THREE.Scene()
  sceneA.background = new THREE.Color(0.1, 0.1, 0.1)
  sceneA.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial({ color: 0xff0000 })))

  const sceneB = new THREE.Scene()
  sceneB.background = new THREE.Color(0.1, 0.1, 0.1)
  sceneB.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial({ color: 0x00ff00 })))

  const a = renderRgba(sceneA, camera)
  const b = renderRgba(sceneB, camera)
  const meanA = meanRgba(a)
  const meanB = meanRgba(b)

  assert.ok(meanA.r > meanB.r + 5, `red scene should have higher red channel mean (${meanA.r} vs ${meanB.r})`)
  assert.ok(meanB.g > meanA.g + 5, `green scene should have higher green channel mean (${meanB.g} vs ${meanA.g})`)
})

test('PBR scene with lights renders and shows lighting variation', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.05, 0.05, 0.05)
  addLights(scene)
  scene.add(
    new THREE.Mesh(
      new THREE.SphereGeometry(1, 32, 32),
      new THREE.MeshStandardMaterial({ color: 0xdddddd, metalness: 0.1, roughness: 0.4 }),
    ),
  )

  const rgba = renderRgba(scene, makeCamera())
  const ratio = nonBackgroundRatio(rgba, [13, 13, 13])
  assert.ok(ratio > 0.05, 'sphere should be visible')

  // Sample both sides of the sphere — the lit side should be brighter than the shadowed side.
  // Top-right quadrant vs bottom-left quadrant of the image.
  let litSum = 0
  let litCount = 0
  let darkSum = 0
  let darkCount = 0
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const i = (y * SIZE + x) * 4
      const lum = rgba[i] + rgba[i + 1] + rgba[i + 2]
      if (lum < 50) continue // skip background
      if (x > SIZE * 0.6 && y < SIZE * 0.4) {
        litSum += lum
        litCount++
      } else if (x < SIZE * 0.4 && y > SIZE * 0.6) {
        darkSum += lum
        darkCount++
      }
    }
  }
  if (litCount > 0 && darkCount > 0) {
    const litAvg = litSum / litCount
    const darkAvg = darkSum / darkCount
    assert.ok(litAvg > darkAvg, `lit side (${litAvg.toFixed(1)}) should be brighter than shadowed side (${darkAvg.toFixed(1)})`)
  }
})

test('MeshPhysicalMaterial extensions and maps affect rendered output', () => {
  const camera = makeCamera()

  function makeScene(material) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0.04, 0.04, 0.045)
    scene.environment = makeEnvironmentTexture()
    scene.environmentIntensity = 0.8
    addLights(scene)
    scene.add(new THREE.Mesh(new THREE.SphereGeometry(1, 32, 32), material))
    return scene
  }

  const base = renderRgba(
    makeScene(new THREE.MeshPhysicalMaterial({
      color: 0x7aa7ff,
      roughness: 0.35,
      metalness: 0.0,
    })),
    camera,
  )
  const physical = renderRgba(
    makeScene(new THREE.MeshPhysicalMaterial({
      color: 0x7aa7ff,
      roughness: 0.35,
      metalness: 0.0,
      clearcoat: 1.0,
      clearcoatMap: solidTexture(255, 0, 0),
      clearcoatRoughness: 0.04,
      clearcoatRoughnessMap: solidTexture(0, 96, 0),
      clearcoatNormalMap: solidTexture(128, 180, 240),
      clearcoatNormalScale: new THREE.Vector2(0.6, 0.4),
      sheen: 0.8,
      sheenColor: new THREE.Color(1.0, 0.25, 0.12),
      sheenColorMap: solidTexture(255, 128, 96),
      sheenRoughness: 0.35,
      sheenRoughnessMap: solidTexture(0, 0, 0, 160),
      anisotropy: 0.85,
      anisotropyRotation: Math.PI / 4,
      anisotropyMap: solidTexture(255, 128, 255),
      transmission: 0.25,
      transmissionMap: solidTexture(180, 0, 0),
      ior: 1.45,
      thickness: 0.4,
      thicknessMap: solidTexture(0, 255, 0),
      attenuationColor: new THREE.Color(0.8, 0.95, 1.0),
      attenuationDistance: 1.5,
    })),
    camera,
  )

  const ratio = nonBackgroundRatio(physical, [10, 10, 11])
  assert.ok(ratio > 0.05, 'physical material sphere should be visible')
  const diff = meanAbsDiff(base, physical)
  assert.ok(diff > 0.5, `expected physical extensions to change output, mean abs diff=${diff.toFixed(3)}`)
})

test('custom WGSL fragment material affects rendered output', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.02)
  const material = new THREE.MeshBasicMaterial({ color: 0xffffff })
  material.userData.headlessThreeRenderer = {
    fragmentWgsl: 'return vec4<f32>(0.0, 1.0, 1.0, alpha);',
  }
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(), material))

  const rgba = renderRgba(scene, makeCamera())
  const mean = meanRgba(rgba)
  assert.ok(mean.g > mean.r + 5, `custom shader should raise green over red (${mean.g} vs ${mean.r})`)
  assert.ok(mean.b > mean.r + 5, `custom shader should raise blue over red (${mean.b} vs ${mean.r})`)
})

test('renderToTarget populates a target-like object with raw RGBA', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.1, 0.1, 0.1)
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial({ color: 0x00ffaa })))

  const target = { texture: {} }
  const out = renderToTarget(scene, makeCamera(), target, { width: 64, height: 32 })
  assert.equal(out, target)
  assert.equal(target.width, 64)
  assert.equal(target.height, 32)
  assert.equal(target.data.length, 64 * 32 * 4)
  assert.equal(target.texture.image.data, target.data)
})

test('post-processing options modify the final image', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(1, 0, 0)
  const camera = makeCamera()

  const base = renderRgba(scene, camera, { width: 64, height: 64 })
  const processed = renderRgba(scene, camera, {
    width: 64,
    height: 64,
    postProcessing: { invert: 1, saturation: 1.5, vignette: 0.25 },
  })
  const diff = meanAbsDiff(base, processed)
  const mean = meanRgba(processed)
  assert.ok(diff > 20, `expected post processing to change image, diff=${diff.toFixed(2)}`)
  assert.ok(mean.g > mean.r, `inverted red background should have stronger green than red (${mean.g} vs ${mean.r})`)
})

test('scene-level reflection probe feeds physical IBL when scene.environment is absent', () => {
  const camera = makeCamera()

  function makeScene(withProbe) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0.04, 0.04, 0.045)
    addLights(scene)
    if (withProbe) {
      scene.userData.headlessThreeRenderer = {
        reflectionProbe: {
          texture: makeEnvironmentTexture(),
          intensity: 1.0,
        },
      }
    }
    scene.add(new THREE.Mesh(
      new THREE.SphereGeometry(1, 32, 32),
      new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1.0, roughness: 0.2 }),
    ))
    return scene
  }

  const withoutProbe = renderRgba(makeScene(false), camera)
  const withProbe = renderRgba(makeScene(true), camera)
  const diff = meanAbsDiff(withoutProbe, withProbe)
  assert.ok(diff > 0.5, `expected reflection probe to affect metallic IBL, diff=${diff.toFixed(3)}`)
})

test('physical transmission samples the already-rendered scene color', () => {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  function makeScene(material) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 0, 0)
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(3, 3), material))
    return scene
  }

  const opaque = renderRgba(
    makeScene(new THREE.MeshPhysicalMaterial({ color: 0xffffff, roughness: 0.2 })),
    camera,
    { width: 64, height: 64 },
  )
  const transmissive = renderRgba(
    makeScene(new THREE.MeshPhysicalMaterial({
      color: 0xffffff,
      roughness: 0.05,
      transmission: 1.0,
      thickness: 0.2,
      ior: 1.5,
    })),
    camera,
    { width: 64, height: 64 },
  )

  const diff = meanAbsDiff(opaque, transmissive)
  const mean = meanRgba(transmissive)
  assert.ok(diff > 5, `expected transmission to differ from opaque material, diff=${diff.toFixed(2)}`)
  assert.ok(mean.r > mean.g + 30, `transmission should reveal red scene color (${mean.r} vs ${mean.g})`)
})

test('directional cascaded shadow hints render successfully', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.05, 0.05, 0.05)
  scene.add(new THREE.AmbientLight(0xffffff, 0.2))

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(8, 8),
    new THREE.MeshStandardMaterial({ color: 0x888888, roughness: 0.8 }),
  )
  ground.rotation.x = -Math.PI / 2
  ground.receiveShadow = true
  scene.add(ground)

  const box = new THREE.Mesh(
    new THREE.BoxGeometry(1, 1, 1),
    new THREE.MeshStandardMaterial({ color: 0xff5533 }),
  )
  box.position.y = 0.5
  box.castShadow = true
  scene.add(box)

  const light = new THREE.DirectionalLight(0xffffff, 1.5)
  light.position.set(4, 6, 3)
  light.target.position.set(0, 0, 0)
  light.castShadow = true
  light.shadow.mapSize.set(256, 256)
  light.userData.headlessThreeRenderer = {
    shadowCascades: [
      { left: -3, right: 3, top: 3, bottom: -3, near: 0.1, far: 16, split: 4 },
      { left: -7, right: 7, top: 7, bottom: -7, near: 0.1, far: 32, split: 12 },
    ],
  }
  scene.add(light)
  scene.add(light.target)

  const rgba = renderRgba(scene, makeCamera(), { width: 64, height: 64 })
  assert.equal(rgba.length, 64 * 64 * 4)
})

test('lines topology renders successfully', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.1, 0.1, 0.1)
  const geom = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-1, 0, 0),
    new THREE.Vector3(1, 0, 0),
    new THREE.Vector3(0, 1, 0),
    new THREE.Vector3(0, -1, 0),
  ])
  scene.add(new THREE.LineSegments(geom, new THREE.LineBasicMaterial({ color: 0xffffff })))

  const camera = makeCamera()
  const r = new Renderer()
  const buf = r.render(scene, camera, { width: SIZE, height: SIZE })
  assertValidPng(buf, { width: SIZE, height: SIZE })
})

test('points topology renders successfully', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.1, 0.1, 0.1)
  const positions = new Float32Array(30)
  for (let i = 0; i < 10; i++) {
    positions[i * 3 + 0] = Math.cos(i) * 0.8
    positions[i * 3 + 1] = Math.sin(i) * 0.8
    positions[i * 3 + 2] = 0
  }
  const geom = new THREE.BufferGeometry()
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  scene.add(new THREE.Points(geom, new THREE.PointsMaterial({ color: 0xffffff, size: 0.1 })))

  const camera = makeCamera()
  const r = new Renderer()
  const buf = r.render(scene, camera, { width: SIZE, height: SIZE })
  assertValidPng(buf, { width: SIZE, height: SIZE })
})

test('empty scene renders the background color', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(1, 0, 0)
  const camera = makeCamera()

  const r = new Renderer()
  const rgba = r.render(scene, camera, { width: 64, height: 64, format: 'rgba' })
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 200, `expected red background, got r=${mean.r}`)
  assert.ok(mean.g < 20, `expected red background, got g=${mean.g}`)
  assert.ok(mean.b < 20, `expected red background, got b=${mean.b}`)
})
