import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { deflateSync } from 'node:zlib'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'

const { Renderer, loadGltfFromFile } = pkg

const SIZE = 96
const BACKGROUND = [5, 5, 5]
const NODE_PERFORMANCE_NODE_COUNT = 10000
const NODE_PERFORMANCE_IMAGE_COUNT = 100
const LARGE_TEXTURE_SIZE = 512

let sharedRenderer

function renderer() {
  sharedRenderer ??= new Renderer()
  return sharedRenderer
}

function makeCamera() {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(2.8, 2.2, 4.2)
  camera.lookAt(0, 0, 0)
  return camera
}

function makeTexture(index) {
  const size = 4
  const data = new Uint8Array(size * size * 4)
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const i = (y * size + x) * 4
      data[i] = (48 + index * 29 + x * 37) % 256
      data[i + 1] = (96 + index * 17 + y * 41) % 256
      data[i + 2] = (144 + index * 23 + (x + y) * 19) % 256
      data[i + 3] = 255
    }
  }
  const texture = new THREE.DataTexture(data, size, size, THREE.RGBAFormat)
  texture.colorSpace = THREE.SRGBColorSpace
  texture.needsUpdate = true
  return texture
}

function makeEncodedTexture(index) {
  const raw = makeTexture(index)
  const image = raw.image
  const data = Buffer.from(image.data.buffer, image.data.byteOffset, image.data.byteLength)
  const encoded = encodePng(data, image.width, image.height)
  const texture = new THREE.Texture()
  texture.image = encoded
  texture.source.data = encoded
  texture.colorSpace = THREE.SRGBColorSpace
  texture.needsUpdate = true
  return texture
}

function makeLargeTexture() {
  const size = LARGE_TEXTURE_SIZE
  const data = new Uint8Array(size * size * 4)
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const i = (y * size + x) * 4
      const horizontal = x / (size - 1)
      const vertical = y / (size - 1)
      data[i] = x < size / 2 ? 235 : 20 + Math.round(horizontal * 55)
      data[i + 1] = 35 + Math.round(vertical * 155)
      data[i + 2] = x < size / 2 ? 25 + Math.round(vertical * 45) : 230
      data[i + 3] = 255
    }
  }
  const texture = new THREE.DataTexture(data, size, size, THREE.RGBAFormat)
  texture.colorSpace = THREE.SRGBColorSpace
  texture.needsUpdate = true
  return texture
}

function makePngDataUri(index) {
  const raw = makeTexture(index)
  const image = raw.image
  const data = Buffer.from(image.data.buffer, image.data.byteOffset, image.data.byteLength)
  return `data:image/png;base64,${encodePng(data, image.width, image.height).toString('base64')}`
}

const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
const CRC32_TABLE = Array.from({ length: 256 }, (_, index) => {
  let value = index
  for (let bit = 0; bit < 8; bit += 1) {
    value = value & 1 ? 0xedb88320 ^ (value >>> 1) : value >>> 1
  }
  return value >>> 0
})

function crc32(buffer) {
  let crc = 0xffffffff
  for (const byte of buffer) {
    crc = CRC32_TABLE[(crc ^ byte) & 0xff] ^ (crc >>> 8)
  }
  return (crc ^ 0xffffffff) >>> 0
}

function pngChunk(type, data) {
  const typeBuffer = Buffer.from(type)
  const chunk = Buffer.alloc(12 + data.length)
  chunk.writeUInt32BE(data.length, 0)
  typeBuffer.copy(chunk, 4)
  data.copy(chunk, 8)
  chunk.writeUInt32BE(crc32(Buffer.concat([typeBuffer, data])), 8 + data.length)
  return chunk
}

function encodePng(rgba, width, height) {
  assert.equal(rgba.length, width * height * 4)
  const ihdr = Buffer.alloc(13)
  ihdr.writeUInt32BE(width, 0)
  ihdr.writeUInt32BE(height, 4)
  ihdr[8] = 8
  ihdr[9] = 6

  const stride = width * 4
  const scanlines = Buffer.alloc((stride + 1) * height)
  for (let y = 0; y < height; y += 1) {
    rgba.copy(scanlines, y * (stride + 1) + 1, y * stride, (y + 1) * stride)
  }

  return Buffer.concat([
    PNG_SIGNATURE,
    pngChunk('IHDR', ihdr),
    pngChunk('IDAT', deflateSync(scanlines)),
    pngChunk('IEND', Buffer.alloc(0)),
  ])
}

function alignedLength(length) {
  return (length + 3) & ~3
}

function makeSharedTriangleBuffer() {
  const arrays = [
    new Float32Array([-0.04, -0.04, 0, 0.04, -0.04, 0, 0, 0.04, 0]),
    new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    new Float32Array([0, 0, 1, 0, 0.5, 1]),
    new Uint16Array([0, 1, 2]),
  ]
  const parts = arrays.map((array) => Buffer.from(array.buffer))
  const offsets = []
  let totalLength = 0
  for (const part of parts) {
    totalLength = alignedLength(totalLength)
    offsets.push(totalLength)
    totalLength += part.length
  }

  const buffer = Buffer.alloc(alignedLength(totalLength))
  for (let i = 0; i < parts.length; i += 1) {
    parts[i].copy(buffer, offsets[i])
  }
  return { buffer, offsets, lengths: parts.map((part) => part.length) }
}

function makeNodePerformanceGltfSource() {
  const { buffer, offsets, lengths } = makeSharedTriangleBuffer()
  const bufferViews = []
  const accessors = []
  const meshes = []
  const materials = []
  const textures = []
  const nodes = []
  const images = Array.from({ length: NODE_PERFORMANCE_IMAGE_COUNT }, (_, index) => ({
    name: `NodePerformanceTest_img${String(index).padStart(2, '0')}`,
    uri: makePngDataUri(index),
  }))

  for (let index = 0; index < NODE_PERFORMANCE_NODE_COUNT; index += 1) {
    const baseBufferView = bufferViews.length
    bufferViews.push(
      { buffer: 0, byteOffset: offsets[0], byteLength: lengths[0], target: 34962 },
      { buffer: 0, byteOffset: offsets[1], byteLength: lengths[1], target: 34962 },
      { buffer: 0, byteOffset: offsets[2], byteLength: lengths[2], target: 34962 },
      { buffer: 0, byteOffset: offsets[3], byteLength: lengths[3], target: 34963 },
    )

    const baseAccessor = accessors.length
    accessors.push(
      { bufferView: baseBufferView, componentType: 5126, count: 3, type: 'VEC3', min: [-0.04, -0.04, 0], max: [0.04, 0.04, 0] },
      { bufferView: baseBufferView + 1, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: baseBufferView + 2, componentType: 5126, count: 3, type: 'VEC2' },
      { bufferView: baseBufferView + 3, componentType: 5123, count: 3, type: 'SCALAR' },
    )

    textures.push({ sampler: 0, source: index % NODE_PERFORMANCE_IMAGE_COUNT })
    materials.push({
      doubleSided: true,
      name: `material_${index}`,
      pbrMetallicRoughness: {
        baseColorTexture: { index },
        metallicFactor: 0,
        roughnessFactor: 0.65,
      },
    })
    meshes.push({
      name: `Cube.${String(index).padStart(4, '0')}`,
      primitives: [{
        attributes: {
          POSITION: baseAccessor,
          NORMAL: baseAccessor + 1,
          TEXCOORD_0: baseAccessor + 2,
        },
        indices: baseAccessor + 3,
        material: index,
      }],
    })
    nodes.push({
      mesh: index,
      name: `rock.${String(index).padStart(4, '0')}`,
      translation: [index % 100, Math.floor(index / 100), 0],
    })
  }

  nodes.push({ camera: 0, name: 'Camera', translation: [50, 50, 120] })
  nodes.push({
    extensions: { KHR_lights_punctual: { light: 0 } },
    name: 'Light',
    translation: [50, 50, 20],
  })

  return {
    accessors,
    asset: { generator: 'headless-three-renderer scale test', version: '2.0' },
    buffers: [{
      byteLength: buffer.length,
      uri: `data:application/octet-stream;base64,${buffer.toString('base64')}`,
    }],
    bufferViews,
    cameras: [{
      type: 'perspective',
      perspective: { aspectRatio: 1, yfov: 0.4, zfar: 1000, znear: 0.1 },
    }],
    extensions: {
      KHR_lights_punctual: {
        lights: [{ type: 'point', intensity: 1 }],
      },
    },
    extensionsRequired: ['KHR_lights_punctual'],
    extensionsUsed: ['KHR_lights_punctual'],
    images,
    materials,
    meshes,
    nodes,
    samplers: [{ magFilter: 9729, minFilter: 9729, wrapS: 10497, wrapT: 10497 }],
    scene: 0,
    scenes: [{ nodes: nodes.map((_, index) => index) }],
    textures,
  }
}

function addSupportedLightBudget(scene, count = 8) {
  scene.add(new THREE.AmbientLight(0xffffff, 0.08))
  for (let i = 0; i < count; i += 1) {
    const angle = (i / count) * Math.PI * 2
    const light = new THREE.PointLight(new THREE.Color().setHSL(i / count, 0.55, 0.65), 0.12, 6, 1.6)
    light.position.set(Math.cos(angle) * 2.2, 1.2 + (i % 4) * 0.28, Math.sin(angle) * 2.2)
    scene.add(light)
  }
}

test('large scene budget renders 144 meshes, textures, and supported lights', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.02)

  const columns = 12
  const rows = 12
  const geometry = new THREE.BoxGeometry(0.115, 0.115, 0.115)
  const textures = Array.from({ length: 10 }, (_, i) => makeTexture(i))
  const materials = textures.map((map, i) => new THREE.MeshStandardMaterial({
    map,
    roughness: 0.48 + (i % 3) * 0.12,
    metalness: i % 2 === 0 ? 0.05 : 0.18,
  }))

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < columns; col += 1) {
      const mesh = new THREE.Mesh(geometry, materials[(row * columns + col) % materials.length])
      mesh.position.set((col - (columns - 1) / 2) * 0.2, (row - (rows - 1) / 2) * 0.18, Math.sin(row * 0.8 + col * 0.45) * 0.18)
      mesh.rotation.set(row * 0.07, col * 0.05, (row + col) * 0.03)
      scene.add(mesh)
    }
  }
  addSupportedLightBudget(scene)

  const rgba = renderer().render(scene, makeCamera(), { width: SIZE, height: SIZE, format: 'rgba' })
  assert.equal(rgba.length, SIZE * SIZE * 4)
  const ratio = nonBackgroundRatio(rgba, BACKGROUND, 6)
  assert.ok(ratio > 0.04, `scale scene should render visible non-background pixels (${ratio})`)
  const mean = meanRgba(rgba)
  assert.ok(mean.a > 240, `scale scene should remain opaque on average (${mean.a})`)
})

test('mesh render budget handles 1,936 separate mesh objects', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.02)

  const columns = 44
  const rows = 44
  const geometry = new THREE.PlaneGeometry(0.034, 0.034)
  const materials = [
    new THREE.MeshBasicMaterial({ color: 0xf25f5c }),
    new THREE.MeshBasicMaterial({ color: 0x247ba0 }),
    new THREE.MeshBasicMaterial({ color: 0x70c1b3 }),
    new THREE.MeshBasicMaterial({ color: 0xffe066 }),
    new THREE.MeshBasicMaterial({ color: 0xc77dff }),
  ]

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < columns; col += 1) {
      const mesh = new THREE.Mesh(geometry, materials[(row + col) % materials.length])
      mesh.position.set((col - (columns - 1) / 2) * 0.049, (row - (rows - 1) / 2) * 0.049, 0)
      mesh.rotation.z = ((row * columns + col) % 7) * 0.04
      scene.add(mesh)
    }
  }

  assert.equal(scene.children.length, rows * columns)

  const camera = new THREE.OrthographicCamera(-1.08, 1.08, 1.08, -1.08, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)

  const rgba = renderer().render(scene, camera, { width: SIZE, height: SIZE, format: 'rgba' })
  assert.equal(rgba.length, SIZE * SIZE * 4)
  const ratio = nonBackgroundRatio(rgba, BACKGROUND, 6)
  assert.ok(ratio > 0.25, `separate mesh budget scene should render broad coverage (${ratio})`)
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 25 && mean.g > 25 && mean.b > 25, `separate mesh colors should survive rendering (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('instanced mesh budget renders 7,056 transformed colored instances', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.02)

  const columns = 84
  const rows = 84
  const count = columns * rows
  const mesh = new THREE.InstancedMesh(
    new THREE.PlaneGeometry(0.022, 0.022),
    new THREE.MeshBasicMaterial({ color: 0xffffff }),
    count,
  )

  const matrix = new THREE.Matrix4()
  const color = new THREE.Color()
  for (let i = 0; i < count; i += 1) {
    const col = i % columns
    const row = Math.floor(i / columns)
    matrix.makeTranslation((col - (columns - 1) / 2) * 0.027, (row - (rows - 1) / 2) * 0.027, 0)
    mesh.setMatrixAt(i, matrix)
    color.setRGB(
      0.25 + 0.75 * (col / (columns - 1)),
      0.25 + 0.75 * (row / (rows - 1)),
      0.25 + 0.75 * ((col + row) / (columns + rows - 2)),
    )
    mesh.setColorAt(i, color)
  }
  scene.add(mesh)

  const camera = new THREE.OrthographicCamera(-1.1, 1.1, 1.1, -1.1, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)

  const rgba = renderer().render(scene, camera, { width: SIZE, height: SIZE, format: 'rgba' })
  assert.equal(rgba.length, SIZE * SIZE * 4)
  const ratio = nonBackgroundRatio(rgba, BACKGROUND, 6)
  assert.ok(ratio > 0.4, `instanced scale scene should fill much of the frame (${ratio})`)
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 40 && mean.g > 40 && mean.b > 40, `instanced colors should survive expansion (${mean.r}, ${mean.g}, ${mean.b})`)
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
  assert.ok(left.r > left.b + 40, `left half should retain red texture detail (${left.r}, ${left.b})`)
  assert.ok(right.b > right.r + 40, `right half should retain blue texture detail (${right.b}, ${right.r})`)
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
