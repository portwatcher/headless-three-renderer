import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { deflateSync } from 'node:zlib'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'
import { BACKGROUND, NODE_PERFORMANCE_IMAGE_COUNT, NODE_PERFORMANCE_NODE_COUNT, SIZE, loadGltfFromFile, makeCamera, makeEncodedTexture, makeLargeTexture, makeNodePerformanceGltfSource, makeTexture, renderer } from './scale.test.part-001.mjs'
test('line object budget renders 2,048 separate transformed lines', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.02)

  const columns = 64
  const rows = 32
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.034, 0, 0,
    0.034, 0, 0,
  ]), 3))
  const materials = [
    new THREE.LineBasicMaterial({ color: 0xf25f5c, linewidth: 2.1 }),
    new THREE.LineBasicMaterial({ color: 0x247ba0, linewidth: 2.1 }),
    new THREE.LineBasicMaterial({ color: 0x70c1b3, linewidth: 2.1 }),
    new THREE.LineBasicMaterial({ color: 0xffe066, linewidth: 2.1 }),
    new THREE.LineBasicMaterial({ color: 0xc77dff, linewidth: 2.1 }),
  ]

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < columns; col += 1) {
      const line = new THREE.LineSegments(geometry, materials[(row + col) % materials.length])
      line.position.set(
        (col / (columns - 1) - 0.5) * 1.94,
        (row / (rows - 1) - 0.5) * 1.88,
        Math.sin(col * 0.19 + row * 0.17) * 0.02,
      )
      line.rotation.z = ((row * columns + col) % 13) * 0.11
      scene.add(line)
    }
  }

  assert.equal(scene.children.length, columns * rows)

  const camera = new THREE.OrthographicCamera(-1.08, 1.08, 1.08, -1.08, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)

  const rgba = renderer().render(scene, camera, { width: SIZE, height: SIZE, format: 'rgba' })
  assert.equal(rgba.length, SIZE * SIZE * 4)
  const ratio = nonBackgroundRatio(rgba, BACKGROUND, 6)
  assert.ok(ratio > 0.3, `separate line object scale scene should cover much of the frame (${ratio})`)
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 30 && mean.g > 30 && mean.b > 30, `separate line object colors should survive traversal (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('texture-heavy scene budget renders 225 unique maps', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.02)

  const columns = 15
  const rows = 15
  const geometry = new THREE.PlaneGeometry(0.12, 0.12)
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < columns; col += 1) {
      const index = row * columns + col
      const material = new THREE.MeshBasicMaterial({ map: makeTexture(index) })
      const mesh = new THREE.Mesh(geometry, material)
      mesh.position.set((col - (columns - 1) / 2) * 0.14, (row - (rows - 1) / 2) * 0.14, 0)
      scene.add(mesh)
    }
  }

  const camera = new THREE.OrthographicCamera(-1.1, 1.1, 1.1, -1.1, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)

  const rgba = renderer().render(scene, camera, { width: SIZE, height: SIZE, format: 'rgba' })
  assert.equal(rgba.length, SIZE * SIZE * 4)
  const ratio = nonBackgroundRatio(rgba, BACKGROUND, 6)
  assert.ok(ratio > 0.25, `texture-heavy scene should render many mapped pixels (${ratio})`)
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 15 && mean.g > 15 && mean.b > 15, `texture-heavy scene should retain textured color (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('encoded texture budget renders 169 unique PNG buffer maps', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.02)

  const columns = 13
  const rows = 13
  const geometry = new THREE.PlaneGeometry(0.12, 0.12)
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < columns; col += 1) {
      const index = row * columns + col
      const material = new THREE.MeshBasicMaterial({ map: makeEncodedTexture(index) })
      const mesh = new THREE.Mesh(geometry, material)
      mesh.position.set((col - (columns - 1) / 2) * 0.15, (row - (rows - 1) / 2) * 0.15, 0)
      scene.add(mesh)
    }
  }

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)

  const rgba = renderer().render(scene, camera, { width: SIZE, height: SIZE, format: 'rgba' })
  assert.equal(rgba.length, SIZE * SIZE * 4)
  const ratio = nonBackgroundRatio(rgba, BACKGROUND, 6)
  assert.ok(ratio > 0.25, `encoded texture scene should render many mapped pixels (${ratio})`)
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 15 && mean.g > 15 && mean.b > 15, `encoded texture scene should retain decoded color (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('large raw texture budget renders a 512 x 512 material map', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ map: makeLargeTexture() }),
  ))

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)

  const rgba = renderer().render(scene, camera, { width: 128, height: 128, format: 'rgba' })
  assert.equal(rgba.length, 128 * 128 * 4)
  const ratio = nonBackgroundRatio(rgba, [0, 0, 0], 3)
  assert.ok(ratio > 0.9, `large texture plane should fill the frame (${ratio})`)
  const left = meanRgba(rgba.filter((_, index) => Math.floor((index / 4) % 128) < 48))
  const right = meanRgba(rgba.filter((_, index) => Math.floor((index / 4) % 128) >= 80))
  const leftIsRed = left.r > left.b + 40
  const leftIsBlue = left.b > left.r + 40
  const rightIsRed = right.r > right.b + 40
  const rightIsBlue = right.b > right.r + 40
  assert.ok(
    (leftIsRed && rightIsBlue) || (leftIsBlue && rightIsRed),
    `large texture plane should retain red/blue horizontal detail (left ${left.r}, ${left.b}; right ${right.r}, ${right.b})`,
  )
})

test('output-size budget renders a 512 x 512 RGBA frame', () => {
  const width = 512
  const height = 512
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(1.8, 1.2),
    new THREE.MeshBasicMaterial({ color: 0xff8844 }),
  ))
  const accent = new THREE.Mesh(
    new THREE.PlaneGeometry(0.9, 0.9),
    new THREE.MeshBasicMaterial({ color: 0x2288ff }),
  )
  accent.position.set(0.18, -0.08, 0.01)
  accent.rotation.z = 0.18
  scene.add(accent)

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)

  const rgba = renderer().render(scene, camera, { width, height, format: 'rgba' })
  assert.equal(rgba.length, width * height * 4)
  const ratio = nonBackgroundRatio(rgba, [0, 0, 0], 3)
  assert.ok(ratio > 0.45, `large output-size scene should cover much of the frame (${ratio})`)
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 35 && mean.g > 35 && mean.b > 25, `large output-size readback should retain color (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('NodePerformanceTest-shaped glTF graph loads many nodes, meshes, materials, and texture definitions', async () => {
  const source = makeNodePerformanceGltfSource()
  assert.equal(source.nodes.length, NODE_PERFORMANCE_NODE_COUNT + 2)
  assert.equal(source.meshes.length, NODE_PERFORMANCE_NODE_COUNT)
  assert.equal(source.materials.length, NODE_PERFORMANCE_NODE_COUNT)
  assert.equal(source.textures.length, NODE_PERFORMANCE_NODE_COUNT)
  assert.equal(source.images.length, NODE_PERFORMANCE_IMAGE_COUNT)
  assert.equal(source.bufferViews.length, NODE_PERFORMANCE_NODE_COUNT * 4)
  assert.equal(source.accessors.length, NODE_PERFORMANCE_NODE_COUNT * 4)

  const tmp = await mkdtemp(path.join(os.tmpdir(), 'headless-three-node-performance-gltf-'))
  try {
    const modelPath = path.join(tmp, 'NodePerformanceShape.gltf')
    await writeFile(modelPath, JSON.stringify(source))

    const gltf = await loadGltfFromFile(modelPath)
    let meshCount = 0
    let firstMesh = null
    let lastMesh = null
    const materials = new Set()
    const textures = new Set()
    const imageBuffers = new Set()
    gltf.scene.traverse((object) => {
      if (object.isMesh !== true) return
      meshCount += 1
      firstMesh ??= object
      lastMesh = object
      materials.add(object.material)
      textures.add(object.material.map)
      imageBuffers.add(object.material.map.image)
    })

    assert.equal(meshCount, NODE_PERFORMANCE_NODE_COUNT)
    assert.equal(materials.size, NODE_PERFORMANCE_NODE_COUNT)
    assert.equal(textures.has(undefined), false)
    assert.equal(textures.size, NODE_PERFORMANCE_IMAGE_COUNT)
    assert.equal(imageBuffers.size, NODE_PERFORMANCE_IMAGE_COUNT)
    assert.equal(gltf.cameras.length, 1)
    assert.equal(firstMesh?.geometry.getAttribute('position')?.count, 3)
    assert.equal(lastMesh?.material?.name, 'material_9999')
  } finally {
    await rm(tmp, { recursive: true, force: true })
  }
})

test('more than 64 visible non-ambient lights fail clearly', () => {
  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshStandardMaterial({ color: 0xffffff })))
  for (let i = 0; i < 65; i += 1) {
    const light = new THREE.PointLight(0xffffff, 0.2)
    light.position.set((i % 5) - 2, 2, Math.floor(i / 5) - 1)
    scene.add(light)
  }

  assert.throws(
    () => renderer().render(scene, makeCamera(), { width: 32, height: 32, format: 'rgba' }),
    /More than 64 visible non-ambient lights/i,
  )
})
