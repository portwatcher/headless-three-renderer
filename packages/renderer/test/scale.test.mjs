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
const NESTED_GRAPH_COLUMNS = 16
const NESTED_GRAPH_ROWS = 16
const NESTED_GRAPH_DEPTH = 8

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

test('material group budget renders 512 grouped spans in one mesh', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.02)

  const columns = 32
  const rows = 16
  const groupCount = columns * rows
  const quadWidth = 0.052
  const quadHeight = 0.096
  const positions = new Float32Array(groupCount * 4 * 3)
  const indices = []
  const geometry = new THREE.BufferGeometry()

  for (let index = 0; index < groupCount; index += 1) {
    const col = index % columns
    const row = Math.floor(index / columns)
    const centerX = (col / (columns - 1) - 0.5) * 1.95
    const centerY = (row / (rows - 1) - 0.5) * 1.85
    const z = Math.sin(col * 0.23 + row * 0.31) * 0.01
    const vertexOffset = index * 4
    const positionOffset = vertexOffset * 3

    positions.set([
      centerX - quadWidth / 2, centerY - quadHeight / 2, z,
      centerX + quadWidth / 2, centerY - quadHeight / 2, z,
      centerX + quadWidth / 2, centerY + quadHeight / 2, z,
      centerX - quadWidth / 2, centerY + quadHeight / 2, z,
    ], positionOffset)
    indices.push(
      vertexOffset, vertexOffset + 1, vertexOffset + 2,
      vertexOffset, vertexOffset + 2, vertexOffset + 3,
    )
    geometry.addGroup(index * 6, 6, index % 16)
  }

  const materials = Array.from({ length: 16 }, (_, index) => new THREE.MeshBasicMaterial({
    color: new THREE.Color().setHSL(index / 16, 0.72, 0.55),
  }))
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geometry.setIndex(indices)
  scene.add(new THREE.Mesh(geometry, materials))

  assert.equal(geometry.groups.length, groupCount)
  assert.equal(materials.length, 16)

  const camera = new THREE.OrthographicCamera(-1.08, 1.08, 1.08, -1.08, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)

  const rgba = renderer().render(scene, camera, { width: SIZE, height: SIZE, format: 'rgba' })
  assert.equal(rgba.length, SIZE * SIZE * 4)
  const ratio = nonBackgroundRatio(rgba, BACKGROUND, 6)
  assert.ok(ratio > 0.3, `grouped material-array mesh should render broad coverage (${ratio})`)
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 30 && mean.g > 30 && mean.b > 25, `group material colors should survive batching (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('transparent sorting budget renders 1,024 layered meshes', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.02)

  const columns = 32
  const rows = 16
  const geometry = new THREE.PlaneGeometry(0.07, 0.09)
  const redMaterial = new THREE.MeshBasicMaterial({
    color: 0xff3344,
    opacity: 0.48,
    transparent: true,
    depthWrite: false,
  })
  const blueMaterial = new THREE.MeshBasicMaterial({
    color: 0x2266ff,
    opacity: 0.48,
    transparent: true,
    depthWrite: false,
  })

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < columns; col += 1) {
      const x = (col / (columns - 1) - 0.5) * 2.0
      const y = (row / (rows - 1) - 0.5) * 1.8
      const z = ((row + col) % 7) * 0.001

      const red = new THREE.Mesh(geometry, redMaterial)
      red.position.set(x, y, z)
      red.rotation.z = ((row * columns + col) % 11) * 0.03
      red.renderOrder = 1
      scene.add(red)

      const blue = new THREE.Mesh(geometry, blueMaterial)
      blue.position.set(x, y, z + 0.0005)
      blue.rotation.z = red.rotation.z + 0.02
      blue.renderOrder = 0
      scene.add(blue)
    }
  }

  assert.equal(scene.children.length, columns * rows * 2)

  const camera = new THREE.OrthographicCamera(-1.08, 1.08, 1.08, -1.08, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)

  const rgba = renderer().render(scene, camera, { width: SIZE, height: SIZE, format: 'rgba' })
  assert.equal(rgba.length, SIZE * SIZE * 4)
  const ratio = nonBackgroundRatio(rgba, BACKGROUND, 6)
  assert.ok(ratio > 0.25, `transparent sorting budget should render broad visible coverage (${ratio})`)
  const mean = meanRgba(rgba)
  assert.ok(mean.r > mean.b + 8, `renderOrder should draw red transparent layers after blue at scale (${mean.r} vs ${mean.b})`)
})

test('nested scene graph budget renders 2,048 transform groups with 256 meshes', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.02)

  const geometry = new THREE.PlaneGeometry(0.078, 0.078)
  const materials = [
    new THREE.MeshBasicMaterial({ color: 0xf25f5c }),
    new THREE.MeshBasicMaterial({ color: 0x247ba0 }),
    new THREE.MeshBasicMaterial({ color: 0xffe066 }),
    new THREE.MeshBasicMaterial({ color: 0x70c1b3 }),
  ]
  let groupCount = 0
  let meshCount = 0

  for (let row = 0; row < NESTED_GRAPH_ROWS; row += 1) {
    for (let col = 0; col < NESTED_GRAPH_COLUMNS; col += 1) {
      const root = new THREE.Object3D()
      root.position.set(
        (col - (NESTED_GRAPH_COLUMNS - 1) / 2) * 0.13,
        (row - (NESTED_GRAPH_ROWS - 1) / 2) * 0.13,
        0,
      )
      scene.add(root)

      let parent = root
      groupCount += 1
      for (let depth = 1; depth < NESTED_GRAPH_DEPTH; depth += 1) {
        const group = new THREE.Object3D()
        group.rotation.z = ((row + col + depth) % 5 - 2) * 0.006
        parent.add(group)
        parent = group
        groupCount += 1
      }

      const mesh = new THREE.Mesh(geometry, materials[(row + col) % materials.length])
      mesh.rotation.z = ((row * NESTED_GRAPH_COLUMNS + col) % 9) * 0.035
      parent.add(mesh)
      meshCount += 1
    }
  }

  assert.equal(groupCount, NESTED_GRAPH_ROWS * NESTED_GRAPH_COLUMNS * NESTED_GRAPH_DEPTH)
  assert.equal(meshCount, NESTED_GRAPH_ROWS * NESTED_GRAPH_COLUMNS)

  const camera = new THREE.OrthographicCamera(-1.08, 1.08, 1.08, -1.08, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)

  const rgba = renderer().render(scene, camera, { width: SIZE, height: SIZE, format: 'rgba' })
  assert.equal(rgba.length, SIZE * SIZE * 4)
  const ratio = nonBackgroundRatio(rgba, BACKGROUND, 6)
  assert.ok(ratio > 0.15, `nested scene graph should render broad visible coverage (${ratio})`)
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 25 && mean.g > 25 && mean.b > 20, `nested scene graph colors should survive traversal (${mean.r}, ${mean.g}, ${mean.b})`)
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

test('InstancedBufferGeometry budget renders 4,096 mapped colored mesh instances', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.02)

  const columns = 64
  const rows = 64
  const count = columns * rows
  const base = new THREE.PlaneGeometry(0.026, 0.026)
  const geometry = new THREE.InstancedBufferGeometry()
  geometry.setAttribute('position', base.getAttribute('position'))
  geometry.setIndex(base.index)

  const offsets = new Float32Array(count * 3)
  const scales = new Float32Array(count)
  const colors = new Float32Array(count * 3)
  const normals = new Float32Array(count * 3)
  const uvs = new Float32Array(count * 2)
  for (let index = 0; index < count; index += 1) {
    const col = index % columns
    const row = Math.floor(index / columns)
    offsets[index * 3] = (col / (columns - 1) - 0.5) * 1.9
    offsets[index * 3 + 1] = (row / (rows - 1) - 0.5) * 1.9
    offsets[index * 3 + 2] = Math.sin(col * 0.19 + row * 0.13) * 0.01
    scales[index] = 0.75 + 0.5 * ((col + row) % 5) / 4
    colors[index * 3] = 0.25 + 0.75 * (col / (columns - 1))
    colors[index * 3 + 1] = 0.25 + 0.75 * (row / (rows - 1))
    colors[index * 3 + 2] = 0.35 + 0.65 * ((col + row) / (columns + rows - 2))
    normals[index * 3] = 0
    normals[index * 3 + 1] = 0
    normals[index * 3 + 2] = 1
    uvs[index * 2] = col < columns / 2 ? 0.25 : 0.75
    uvs[index * 2 + 1] = row < rows / 2 ? 0.25 : 0.75
  }

  geometry.setAttribute('instanceOffset', new THREE.InstancedBufferAttribute(offsets, 3))
  geometry.setAttribute('instanceScale', new THREE.InstancedBufferAttribute(scales, 1))
  geometry.setAttribute('color', new THREE.InstancedBufferAttribute(colors, 3))
  geometry.setAttribute('normal', new THREE.InstancedBufferAttribute(normals, 3))
  geometry.setAttribute('uv', new THREE.InstancedBufferAttribute(uvs, 2))

  const textureData = new Uint8Array([
    255, 255, 255, 255,
    96, 180, 255, 255,
    255, 160, 96, 255,
    180, 255, 160, 255,
  ])
  const texture = new THREE.DataTexture(textureData, 2, 2, THREE.RGBAFormat)
  texture.colorSpace = THREE.SRGBColorSpace
  texture.needsUpdate = true

  scene.add(new THREE.Mesh(
    geometry,
    new THREE.MeshBasicMaterial({ color: 0xffffff, vertexColors: true, map: texture }),
  ))

  const camera = new THREE.OrthographicCamera(-1.1, 1.1, 1.1, -1.1, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)

  const rgba = renderer().render(scene, camera, { width: SIZE, height: SIZE, format: 'rgba' })
  assert.equal(rgba.length, SIZE * SIZE * 4)
  const ratio = nonBackgroundRatio(rgba, BACKGROUND, 6)
  assert.ok(ratio > 0.35, `InstancedBufferGeometry scale scene should cover much of the frame (${ratio})`)
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 35 && mean.g > 35 && mean.b > 35, `InstancedBufferGeometry mapped colors should survive expansion (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('BatchedMesh budget renders 2,048 packed colored instances', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.02)

  const columns = 64
  const rows = 32
  const count = columns * rows
  const source = new THREE.PlaneGeometry(0.04, 0.04)
  const material = new THREE.MeshBasicMaterial({ color: 0xffffff })
  const batched = new THREE.BatchedMesh(
    count,
    source.getAttribute('position').count,
    source.index.count,
    material,
  )
  const geometryId = batched.addGeometry(source)
  const matrix = new THREE.Matrix4()
  const color = new THREE.Color()
  for (let index = 0; index < count; index += 1) {
    const col = index % columns
    const row = Math.floor(index / columns)
    const instanceId = batched.addInstance(geometryId)
    matrix.makeTranslation(
      (col / (columns - 1) - 0.5) * 1.9,
      (row / (rows - 1) - 0.5) * 1.9,
      Math.sin(col * 0.11 + row * 0.29) * 0.01,
    )
    batched.setMatrixAt(instanceId, matrix)
    color.setRGB(
      0.2 + 0.8 * (col / (columns - 1)),
      0.2 + 0.8 * (row / (rows - 1)),
      0.35 + 0.65 * ((col + row) / (columns + rows - 2)),
    )
    batched.setColorAt(instanceId, color)
  }
  batched.frustumCulled = false
  batched.perObjectFrustumCulled = false
  batched.sortObjects = false
  scene.add(batched)

  const camera = new THREE.OrthographicCamera(-1.08, 1.08, 1.08, -1.08, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)

  const rgba = renderer().render(scene, camera, { width: SIZE, height: SIZE, format: 'rgba' })
  assert.equal(rgba.length, SIZE * SIZE * 4)
  const ratio = nonBackgroundRatio(rgba, BACKGROUND, 6)
  assert.ok(ratio > 0.3, `BatchedMesh scale scene should cover much of the frame (${ratio})`)
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 35 && mean.g > 35 && mean.b > 35, `BatchedMesh instance colors should survive expansion (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('CPU deformation budget renders a 4,096-vertex morphed skinned mesh', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.02)

  const panelColumns = 32
  const rows = 64
  const panelCount = 2
  const vertexCount = panelColumns * rows * panelCount
  const positions = new Float32Array(vertexCount * 3)
  const normals = new Float32Array(vertexCount * 3)
  const colors = new Float32Array(vertexCount * 3)
  const skinIndices = new Uint16Array(vertexCount * 4)
  const skinWeights = new Float32Array(vertexCount * 4)
  const morphPositions = new Float32Array(vertexCount * 3)
  const morphNormals = new Float32Array(vertexCount * 3)
  const indices = []

  let vertex = 0
  for (let panel = 0; panel < panelCount; panel += 1) {
    const boneIndex = panel
    for (let row = 0; row < rows; row += 1) {
      const v = row / (rows - 1)
      for (let column = 0; column < panelColumns; column += 1) {
        const u = column / (panelColumns - 1)
        const baseX = panel === 0 ? -2.85 + u : 1.85 + u
        const baseY = -3.0 + v * 1.2
        const wave = Math.sin((u * 5 + v * 7 + panel) * Math.PI) * 0.06
        const offset = vertex * 3

        positions[offset] = baseX
        positions[offset + 1] = baseY
        positions[offset + 2] = 0
        normals[offset] = 0
        normals[offset + 1] = 0
        normals[offset + 2] = 1
        colors[offset] = panel === 0 ? 1 - v * 0.4 : 0.25 + u * 0.5
        colors[offset + 1] = 0.25 + v * 0.7
        colors[offset + 2] = panel === 0 ? 0.35 + u * 0.55 : 0.95 - v * 0.35
        skinIndices[vertex * 4] = boneIndex
        skinWeights[vertex * 4] = 1
        morphPositions[offset + 1] = 2.25 + wave
        morphPositions[offset + 2] = wave
        morphNormals[offset + 2] = 0.02
        vertex += 1
      }
    }
  }

  for (let panel = 0; panel < panelCount; panel += 1) {
    const panelStart = panel * panelColumns * rows
    for (let row = 0; row < rows - 1; row += 1) {
      for (let column = 0; column < panelColumns - 1; column += 1) {
        const a = panelStart + row * panelColumns + column
        const b = a + 1
        const c = a + panelColumns
        const d = c + 1
        indices.push(a, c, b, b, c, d)
      }
    }
  }

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3))
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  geometry.setAttribute('skinIndex', new THREE.BufferAttribute(skinIndices, 4))
  geometry.setAttribute('skinWeight', new THREE.BufferAttribute(skinWeights, 4))
  geometry.setIndex(indices)
  geometry.morphTargetsRelative = true
  geometry.morphAttributes.position = [new THREE.BufferAttribute(morphPositions, 3)]
  geometry.morphAttributes.normal = [new THREE.BufferAttribute(morphNormals, 3)]

  const mesh = new THREE.SkinnedMesh(
    geometry,
    new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }),
  )
  mesh.frustumCulled = false
  mesh.morphTargetInfluences = [1]

  const leftBone = new THREE.Bone()
  const rightBone = new THREE.Bone()
  mesh.add(leftBone)
  mesh.add(rightBone)
  mesh.bind(new THREE.Skeleton([leftBone, rightBone]))
  leftBone.position.x = 2.35
  rightBone.position.x = -2.35
  scene.add(mesh)

  const camera = new THREE.OrthographicCamera(-1.08, 1.08, 1.08, -1.08, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)

  assert.equal(vertexCount, 4096)
  assert.equal(indices.length, (panelColumns - 1) * (rows - 1) * panelCount * 6)

  const rgba = renderer().render(scene, camera, { width: SIZE, height: SIZE, format: 'rgba' })
  assert.equal(rgba.length, SIZE * SIZE * 4)
  const ratio = nonBackgroundRatio(rgba, BACKGROUND, 6)
  assert.ok(ratio > 0.22, `morphed skinned scale scene should render after CPU deformation (${ratio})`)
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 35 && mean.g > 35 && mean.b > 35, `deformed vertex colors should survive CPU baking (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('points billboard budget renders 4,096 colored points', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.02)

  const columns = 64
  const rows = 64
  const count = columns * rows
  const positions = new Float32Array(count * 3)
  const colors = new Float32Array(count * 3)
  for (let index = 0; index < count; index += 1) {
    const col = index % columns
    const row = Math.floor(index / columns)
    positions[index * 3] = (col / (columns - 1) - 0.5) * 1.9
    positions[index * 3 + 1] = (row / (rows - 1) - 0.5) * 1.9
    positions[index * 3 + 2] = Math.sin(col * 0.23 + row * 0.17) * 0.02
    colors[index * 3] = 0.2 + 0.8 * (col / (columns - 1))
    colors[index * 3 + 1] = 0.2 + 0.8 * (row / (rows - 1))
    colors[index * 3 + 2] = 0.35 + 0.65 * ((col + row) / (columns + rows - 2))
  }

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  scene.add(new THREE.Points(
    geometry,
    new THREE.PointsMaterial({ size: 2.2, sizeAttenuation: false, vertexColors: true }),
  ))

  const camera = new THREE.OrthographicCamera(-1.08, 1.08, 1.08, -1.08, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)

  const rgba = renderer().render(scene, camera, { width: SIZE, height: SIZE, format: 'rgba' })
  assert.equal(rgba.length, SIZE * SIZE * 4)
  const ratio = nonBackgroundRatio(rgba, BACKGROUND, 6)
  assert.ok(ratio > 0.55, `point billboard scale scene should cover much of the frame (${ratio})`)
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 35 && mean.g > 35 && mean.b > 35, `point colors should survive billboard expansion (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('point object budget renders 2,048 separate transformed Points objects', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.02)

  const columns = 64
  const rows = 32
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))
  const materials = [
    new THREE.PointsMaterial({ color: 0xf25f5c, size: 2.4, sizeAttenuation: false }),
    new THREE.PointsMaterial({ color: 0x247ba0, size: 2.4, sizeAttenuation: false }),
    new THREE.PointsMaterial({ color: 0x70c1b3, size: 2.4, sizeAttenuation: false }),
    new THREE.PointsMaterial({ color: 0xffe066, size: 2.4, sizeAttenuation: false }),
    new THREE.PointsMaterial({ color: 0xc77dff, size: 2.4, sizeAttenuation: false }),
  ]

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < columns; col += 1) {
      const points = new THREE.Points(geometry, materials[(row + col) % materials.length])
      points.position.set(
        (col / (columns - 1) - 0.5) * 1.9,
        (row / (rows - 1) - 0.5) * 1.9,
        Math.sin(col * 0.21 + row * 0.13) * 0.02,
      )
      scene.add(points)
    }
  }

  assert.equal(scene.children.length, columns * rows)

  const camera = new THREE.OrthographicCamera(-1.08, 1.08, 1.08, -1.08, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)

  const rgba = renderer().render(scene, camera, { width: SIZE, height: SIZE, format: 'rgba' })
  assert.equal(rgba.length, SIZE * SIZE * 4)
  const ratio = nonBackgroundRatio(rgba, BACKGROUND, 6)
  assert.ok(ratio > 0.4, `separate Points object scale scene should cover much of the frame (${ratio})`)
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 30 && mean.g > 30 && mean.b > 30, `separate Points object colors should survive traversal (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('sprite billboard budget renders 2,048 colored sprites', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.02)

  const columns = 64
  const rows = 32
  const materialCount = 64
  const materials = Array.from({ length: materialCount }, (_, index) => {
    const t = index / (materialCount - 1)
    return new THREE.SpriteMaterial({
      color: new THREE.Color(0.25 + 0.75 * t, 0.25 + 0.65 * (1 - t), 0.45 + 0.45 * Math.sin(t * Math.PI)),
      transparent: false,
    })
  })

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < columns; col += 1) {
      const sprite = new THREE.Sprite(materials[(row * columns + col) % materialCount])
      sprite.position.set(
        (col / (columns - 1) - 0.5) * 1.9,
        (row / (rows - 1) - 0.5) * 1.9,
        Math.sin(col * 0.17 + row * 0.31) * 0.01,
      )
      sprite.scale.setScalar(0.045)
      scene.add(sprite)
    }
  }

  const camera = new THREE.OrthographicCamera(-1.08, 1.08, 1.08, -1.08, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)

  const rgba = renderer().render(scene, camera, { width: SIZE, height: SIZE, format: 'rgba' })
  assert.equal(rgba.length, SIZE * SIZE * 4)
  const ratio = nonBackgroundRatio(rgba, BACKGROUND, 6)
  assert.ok(ratio > 0.35, `sprite billboard scale scene should cover much of the frame (${ratio})`)
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 35 && mean.g > 35 && mean.b > 35, `sprite colors should survive billboard expansion (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('wide line budget renders 4,032 colored segments', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.02)

  const columns = 64
  const rows = 64
  const segments = rows * (columns - 1)
  const positions = new Float32Array(segments * 2 * 3)
  const colors = new Float32Array(segments * 2 * 3)
  for (let index = 0; index < segments; index += 1) {
    const row = Math.floor(index / (columns - 1))
    const col = index % (columns - 1)
    const x0 = (col / (columns - 1) - 0.5) * 1.9
    const x1 = ((col + 1) / (columns - 1) - 0.5) * 1.9
    const y = (row / (rows - 1) - 0.5) * 1.9
    const z = Math.sin(col * 0.19 + row * 0.13) * 0.01
    const offset = index * 6
    positions[offset] = x0
    positions[offset + 1] = y
    positions[offset + 2] = z
    positions[offset + 3] = x1
    positions[offset + 4] = y
    positions[offset + 5] = z

    const r0 = 0.2 + 0.8 * (col / (columns - 1))
    const r1 = 0.2 + 0.8 * ((col + 1) / (columns - 1))
    const g = 0.2 + 0.8 * (row / (rows - 1))
    const b0 = 0.35 + 0.65 * ((col + row) / (columns + rows - 2))
    const b1 = 0.35 + 0.65 * ((col + 1 + row) / (columns + rows - 2))
    colors[offset] = r0
    colors[offset + 1] = g
    colors[offset + 2] = b0
    colors[offset + 3] = r1
    colors[offset + 4] = g
    colors[offset + 5] = b1
  }

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  scene.add(new THREE.LineSegments(
    geometry,
    new THREE.LineBasicMaterial({ linewidth: 2.2, vertexColors: true }),
  ))

  const camera = new THREE.OrthographicCamera(-1.08, 1.08, 1.08, -1.08, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)

  const rgba = renderer().render(scene, camera, { width: SIZE, height: SIZE, format: 'rgba' })
  assert.equal(rgba.length, SIZE * SIZE * 4)
  const ratio = nonBackgroundRatio(rgba, BACKGROUND, 6)
  assert.ok(ratio > 0.5, `wide line scale scene should cover much of the frame (${ratio})`)
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 35 && mean.g > 35 && mean.b > 35, `line colors should survive wide-line expansion (${mean.r}, ${mean.g}, ${mean.b})`)
})

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
