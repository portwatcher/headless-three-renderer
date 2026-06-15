import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'

const {
  Renderer,
  loadGltfFromFile,
  loadVrmAnimationFromFile,
  loadVrmFromFile,
} = pkg

const FIXTURE_DIR = fileURLToPath(new URL('./fixtures/', import.meta.url))
const SIMPLE_TRIANGLE = path.join(FIXTURE_DIR, 'simple-triangle.gltf')
const TEXTURED_QUAD = path.join(FIXTURE_DIR, 'textured-quad.gltf')
const VERTEX_COLOR_QUAD = path.join(FIXTURE_DIR, 'vertex-color-quad.gltf')
const MORPHED_TRIANGLE = path.join(FIXTURE_DIR, 'morphed-triangle.gltf')
const SKINNED_QUAD = path.join(FIXTURE_DIR, 'skinned-quad.gltf')
const SYNTHETIC_VRM = path.join(FIXTURE_DIR, 'synthetic-avatar.vrm')
const SYNTHETIC_VRMA = path.join(FIXTURE_DIR, 'synthetic-animation.vrma')

test('committed glTF fixture loads through GLTFLoader and renders', async () => {
  let configured = false
  const gltf = await loadGltfFixture(SIMPLE_TRIANGLE, {
    configureLoader(loader) {
      configured = typeof loader.parse === 'function'
    },
  })
  assert.equal(configured, true, 'loadGltfFromFile should expose the loader before parsing')

  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'fixture should load a mesh')
  assert.equal(mesh.geometry.getAttribute('position').count, 3)
  assert.equal(mesh.material.isMeshStandardMaterial, true)

  const camera = gltf.cameras[0]
  assert.ok(camera, 'fixture should load a camera')
  camera.aspect = 1
  camera.updateProjectionMatrix()

  const scene = gltf.scene
  scene.add(new THREE.AmbientLight(0xffffff, 0.6))
  const light = new THREE.DirectionalLight(0xffffff, 1.4)
  light.position.set(2, 3, 4)
  scene.add(light)
  scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [0.02, 0.02, 0.03],
  })
  assert.equal(rgba.length, 96 * 96 * 4)
  assert.ok(nonBackgroundRatio(rgba, [5, 5, 8], 3) > 0.04, 'glTF triangle should render visible pixels')

  const mean = meanRgba(rgba)
  assert.ok(mean.b > mean.r, `loaded blue PBR material should contribute blue output (${mean.b} vs ${mean.r})`)
  assert.ok(mean.a > 240, `loaded glTF output should be opaque (${mean.a})`)
})

test('committed textured glTF fixture loads data URI image and renders texture', async () => {
  const gltf = await loadGltfFixture(TEXTURED_QUAD)

  assertTexturedQuadLoadsEncodedMap(gltf, 'textured fixture')
  assertTexturedQuadRendersTexture(gltf, 'textured quad')
})

test('loadGltfFromFile loads helper-normalized GLB bufferView images', async () => {
  const source = JSON.parse(await readFile(TEXTURED_QUAD, 'utf8'))
  const glbBytes = buildTexturedQuadGlb(source)
  const tmp = await mkdtemp(path.join(os.tmpdir(), 'headless-three-glb-image-'))
  try {
    const modelPath = path.join(tmp, 'buffer-view-image.glb')
    await writeFile(modelPath, glbBytes)

    const gltf = await loadGltfFixture(modelPath)
    assertTexturedQuadLoadsEncodedMap(gltf, 'GLB bufferView-image fixture')
    assertTexturedQuadRendersTexture(gltf, 'GLB bufferView-image quad')
  } finally {
    await rm(tmp, { recursive: true, force: true })
  }
})

test('loadGltfFromFile rejects compressed GLB bufferView images with pre-decode guidance', async () => {
  const source = JSON.parse(await readFile(TEXTURED_QUAD, 'utf8'))
  source.images[0].mimeType = 'image/ktx2'
  const glbBytes = buildTexturedQuadGlb(source)
  const tmp = await mkdtemp(path.join(os.tmpdir(), 'headless-three-glb-compressed-image-'))
  try {
    const modelPath = path.join(tmp, 'compressed-buffer-view-image.glb')
    await writeFile(modelPath, glbBytes)

    await assert.rejects(
      () => loadGltfFixture(modelPath),
      /GLB bufferView image.*compressed texture.*KTX2.*Basis.*pre-decode/i,
    )
  } finally {
    await rm(tmp, { recursive: true, force: true })
  }
})

test('loadGltfFromFile rejects external compressed glTF image references with pre-decode guidance', async () => {
  const source = JSON.parse(await readFile(TEXTURED_QUAD, 'utf8'))
  source.images[0].uri = 'textures/albedo.ktx2'
  const tmp = await mkdtemp(path.join(os.tmpdir(), 'headless-three-gltf-compressed-image-'))
  try {
    const modelPath = path.join(tmp, 'compressed-image-reference.gltf')
    await writeFile(modelPath, JSON.stringify(source))

    await assert.rejects(
      () => loadGltfFixture(modelPath),
      /glTF image URI.*compressed texture.*KTX2.*Basis.*pre-decode/i,
    )
  } finally {
    await rm(tmp, { recursive: true, force: true })
  }
})

test('committed vertex-color glTF fixture renders COLOR_0 attributes', async () => {
  const gltf = await loadGltfFixture(VERTEX_COLOR_QUAD)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'vertex-color fixture should load a mesh')
  assert.equal(mesh.geometry.getAttribute('color')?.count, 4)
  assert.equal(mesh.material.vertexColors, true)

  const camera = gltf.cameras[0]
  assert.ok(camera, 'vertex-color fixture should load a camera')
  camera.aspect = 1
  camera.updateProjectionMatrix()
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })
  assert.equal(rgba.length, 96 * 96 * 4)
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.25, 'vertex-color quad should render visible pixels')

  const left = meanRegion(rgba, 96, 96, 24, 36, 42, 60)
  const right = meanRegion(rgba, 96, 96, 54, 36, 72, 60)
  assert.ok(left.r > left.g + 60, `left half should be dominated by COLOR_0 red (${left.r} vs ${left.g})`)
  assert.ok(right.g > right.r + 60, `right half should be dominated by COLOR_0 green (${right.g} vs ${right.r})`)
})

test('committed morph-target glTF fixture applies POSITION targets', async () => {
  const gltf = await loadGltfFixture(MORPHED_TRIANGLE)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'morph fixture should load a mesh')
  assert.equal(mesh.geometry.morphAttributes.position?.length, 1)
  assert.equal(mesh.morphTargetInfluences?.length, 1)

  const camera = gltf.cameras[0]
  assert.ok(camera, 'morph fixture should load a camera')
  camera.aspect = 1
  camera.updateProjectionMatrix()
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  function renderBounds(influence) {
    mesh.morphTargetInfluences[0] = influence
    const rgba = new Renderer().render(gltf.scene, camera, {
      width: 96,
      height: 96,
      format: 'rgba',
      background: [0, 0, 0],
      outputColorSpace: THREE.LinearSRGBColorSpace,
    })
    return nonBackgroundBounds(rgba, 96, 96, [0, 0, 0], 3)
  }

  const flat = renderBounds(0)
  const morphed = renderBounds(1)
  assert.ok(flat.height > 10, `flat triangle should render visible bounds (${flat.height})`)
  assert.ok(morphed.minY < flat.minY - 12, `morph target should move the triangle top upward (${morphed.minY} vs ${flat.minY})`)
  assert.ok(morphed.height > flat.height + 10, `morph target should expand rendered height (${morphed.height} vs ${flat.height})`)
})

test('committed skinned glTF fixture applies JOINTS_0 and WEIGHTS_0 attributes', async () => {
  const gltf = await loadGltfFixture(SKINNED_QUAD)
  const mesh = findFirst(gltf.scene, (object) => object.isSkinnedMesh === true)
  assert.ok(mesh, 'skinned fixture should load a SkinnedMesh')
  assert.equal(mesh.geometry.getAttribute('skinIndex')?.count, 4)
  assert.equal(mesh.geometry.getAttribute('skinWeight')?.count, 4)
  assert.equal(mesh.skeleton.bones.length, 1)

  const camera = gltf.cameras[0]
  assert.ok(camera, 'skinned fixture should load a camera')
  camera.aspect = 1
  camera.updateProjectionMatrix()
  camera.updateMatrixWorld(true)

  function renderBounds(jointY) {
    mesh.skeleton.bones[0].position.y = jointY
    gltf.scene.updateMatrixWorld(true)
    const rgba = new Renderer().render(gltf.scene, camera, {
      width: 96,
      height: 96,
      format: 'rgba',
      background: [0, 0, 0],
      outputColorSpace: THREE.LinearSRGBColorSpace,
    })
    return nonBackgroundBounds(rgba, 96, 96, [0, 0, 0], 3)
  }

  const base = renderBounds(0)
  const moved = renderBounds(0.55)
  assert.ok(base.height > 20, `base skinned quad should render visible bounds (${base.height})`)
  assert.ok(moved.minY < base.minY - 12, `joint translation should move the skinned quad upward (${moved.minY} vs ${base.minY})`)
  assert.ok(Math.abs(moved.height - base.height) <= 4, `single-joint translation should preserve quad height (${moved.height} vs ${base.height})`)
})

test('VRM loader helpers register supplied Pixiv-style plugins', async () => {
  let vrmPluginParser = null
  let animationPluginParser = null
  let modelPluginParser = null

  class FakeVRMLoaderPlugin {
    constructor(parser) {
      this.name = 'FakeVRMLoaderPlugin'
      vrmPluginParser = parser
    }
  }

  class FakeModelLoaderPlugin {
    constructor(parser) {
      this.name = 'FakeModelLoaderPlugin'
      modelPluginParser = parser
    }
  }

  class FakeVRMAnimationLoaderPlugin {
    constructor(parser) {
      this.name = 'FakeVRMAnimationLoaderPlugin'
      animationPluginParser = parser
    }
  }

  const vrmGltf = await loadVrmFromFile(SYNTHETIC_VRM, {
    VRMLoaderPlugin: FakeVRMLoaderPlugin,
  })
  assert.ok(findFirst(vrmGltf.scene, (object) => object.isMesh === true), 'VRM helper should still parse glTF scenes')
  assert.ok(vrmPluginParser, 'VRM helper should install the supplied VRMLoaderPlugin')
  assert.ok(vrmPluginParser.json?.extensionsUsed?.includes('VRMC_vrm'), 'VRM fixture should expose VRMC_vrm metadata to the plugin')

  const animationGltf = await loadVrmAnimationFromFile(SYNTHETIC_VRMA, {
    VRMLoaderPlugin: FakeModelLoaderPlugin,
    VRMAnimationLoaderPlugin: FakeVRMAnimationLoaderPlugin,
  })
  assert.ok(findFirst(animationGltf.scene, (object) => object.isMesh === true), 'VRMA helper should still parse glTF scenes')
  assert.ok(modelPluginParser, 'VRMA helper should install the supplied VRMLoaderPlugin when provided')
  assert.ok(animationPluginParser, 'VRMA helper should install the supplied VRMAnimationLoaderPlugin')
  assert.ok(
    animationPluginParser.json?.extensionsUsed?.includes('VRMC_vrm_animation'),
    'VRMA fixture should expose VRMC_vrm_animation metadata to the plugin',
  )
})

test('loadGltfFromFile resolves external glTF image files from the model directory', async () => {
  const source = JSON.parse(await readFile(TEXTURED_QUAD, 'utf8'))
  const imageUri = source.images[0].uri
  const encodedImage = imageUri.slice(imageUri.indexOf(',') + 1)
  const tmp = await mkdtemp(path.join(os.tmpdir(), 'headless-three-gltf-image-'))
  try {
    const textureDir = path.join(tmp, 'textures')
    await mkdir(textureDir)
    await writeFile(path.join(textureDir, 'quad.png'), Buffer.from(encodedImage, 'base64'))
    source.images[0].uri = 'textures/quad.png'
    const modelPath = path.join(tmp, 'external-image.gltf')
    await writeFile(modelPath, JSON.stringify(source))

    const gltf = await loadGltfFixture(modelPath)
    assertTexturedQuadLoadsEncodedMap(gltf, 'external-image fixture')
    assertTexturedQuadRendersTexture(gltf, 'external-image quad')
  } finally {
    await rm(tmp, { recursive: true, force: true })
  }
})

test('loadGltfFromFile resolves external glTF buffers from the model directory', async () => {
  const source = JSON.parse(await readFile(SIMPLE_TRIANGLE, 'utf8'))
  const bufferUri = source.buffers[0].uri
  const encodedBuffer = bufferUri.slice(bufferUri.indexOf(',') + 1)
  const tmp = await mkdtemp(path.join(os.tmpdir(), 'headless-three-gltf-'))
  try {
    await writeFile(path.join(tmp, 'triangle.bin'), Buffer.from(encodedBuffer, 'base64'))
    source.buffers[0].uri = 'triangle.bin'
    const modelPath = path.join(tmp, 'external-buffer.gltf')
    await writeFile(modelPath, JSON.stringify(source))

    const gltf = await loadGltfFixture(modelPath)
    const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
    assert.ok(mesh, 'external-buffer fixture should load a mesh')

    const camera = gltf.cameras[0]
    assert.ok(camera, 'external-buffer fixture should load a camera')
    camera.aspect = 1
    camera.updateProjectionMatrix()
    gltf.scene.add(new THREE.AmbientLight(0xffffff, 1))
    gltf.scene.updateMatrixWorld(true)
    camera.updateMatrixWorld(true)

    const rgba = new Renderer().render(gltf.scene, camera, {
      width: 64,
      height: 64,
      format: 'rgba',
      background: [0, 0, 0],
    })
    assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.04, 'external buffer glTF should render visible pixels')
  } finally {
    await rm(tmp, { recursive: true, force: true })
  }
})

function assertTexturedQuadLoadsEncodedMap(gltf, label) {
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, `${label} should load a mesh`)
  assert.ok(mesh.material.map?.isTexture, `${label} should load a base color texture`)
  assert.ok(Buffer.isBuffer(mesh.material.map.image), 'encoded image helper should expose the PNG as a Buffer')
}

function assertTexturedQuadRendersTexture(gltf, label) {
  const camera = gltf.cameras[0]
  assert.ok(camera, `${label} should load a camera`)
  camera.aspect = 1
  camera.updateProjectionMatrix()

  const scene = gltf.scene
  scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })
  assert.equal(rgba.length, 96 * 96 * 4)
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.25, `${label} should render visible pixels`)

  const left = meanRegion(rgba, 96, 96, 24, 36, 42, 60)
  const right = meanRegion(rgba, 96, 96, 54, 36, 72, 60)
  assert.ok(left.r > left.g + 80, `left half should sample the red texture texel (${left.r} vs ${left.g})`)
  assert.ok(right.g > right.r + 80, `right half should sample the green texture texel (${right.g} vs ${right.r})`)
}

async function loadGltfFixture(filePath, options) {
  return await loadGltfFromFile(filePath, options)
}

function buildTexturedQuadGlb(source) {
  const geometryBytes = decodeDataUriBuffer(source.buffers[0].uri, 'textured fixture geometry buffer')
  const imageBytes = decodeDataUriBuffer(source.images[0].uri, 'textured fixture image')
  const imageOffset = alignedLength(geometryBytes.length)
  const binLength = imageOffset + imageBytes.length
  const bin = Buffer.alloc(alignedLength(binLength))
  geometryBytes.copy(bin, 0)
  imageBytes.copy(bin, imageOffset)

  const glb = structuredClone(source)
  delete glb.buffers[0].uri
  glb.buffers[0].byteLength = binLength
  glb.bufferViews.push({
    buffer: 0,
    byteOffset: imageOffset,
    byteLength: imageBytes.length,
  })
  glb.images[0] = {
    name: source.images[0].name,
    mimeType: source.images[0].mimeType,
    bufferView: glb.bufferViews.length - 1,
  }

  return encodeGlb(glb, bin)
}

function decodeDataUriBuffer(uri, label) {
  assert.equal(typeof uri, 'string', `${label} should be a data URI`)
  const comma = uri.indexOf(',')
  assert.notEqual(comma, -1, `${label} should contain a comma separator`)
  const metadata = uri.slice(5, comma)
  const payload = uri.slice(comma + 1)
  return /(?:^|;)base64(?:;|$)/i.test(metadata)
    ? Buffer.from(payload, 'base64')
    : Buffer.from(decodeURIComponent(payload), 'utf8')
}

function encodeGlb(json, bin) {
  const jsonChunk = paddedBuffer(Buffer.from(JSON.stringify(json), 'utf8'), 0x20)
  const binChunk = paddedBuffer(bin, 0x00)
  const totalLength = 12 + 8 + jsonChunk.length + 8 + binChunk.length
  const glb = Buffer.alloc(totalLength)
  let offset = 0
  offset = writeUint32(glb, offset, 0x46546c67)
  offset = writeUint32(glb, offset, 2)
  offset = writeUint32(glb, offset, totalLength)
  offset = writeUint32(glb, offset, jsonChunk.length)
  offset = writeUint32(glb, offset, 0x4e4f534a)
  jsonChunk.copy(glb, offset)
  offset += jsonChunk.length
  offset = writeUint32(glb, offset, binChunk.length)
  offset = writeUint32(glb, offset, 0x004e4942)
  binChunk.copy(glb, offset)
  return glb
}

function paddedBuffer(buffer, fill) {
  const padded = Buffer.alloc(alignedLength(buffer.length), fill)
  buffer.copy(padded)
  return padded
}

function alignedLength(length) {
  return (length + 3) & ~3
}

function writeUint32(buffer, offset, value) {
  buffer.writeUInt32LE(value, offset)
  return offset + 4
}

function findFirst(root, predicate) {
  let match = null
  root.traverse((object) => {
    if (!match && predicate(object)) match = object
  })
  return match
}

function meanRegion(rgba, width, _height, x0, y0, x1, y1) {
  let r = 0
  let g = 0
  let b = 0
  let a = 0
  let count = 0
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const i = (y * width + x) * 4
      r += rgba[i]
      g += rgba[i + 1]
      b += rgba[i + 2]
      a += rgba[i + 3]
      count++
    }
  }
  return { r: r / count, g: g / count, b: b / count, a: a / count }
}

function nonBackgroundBounds(rgba, width, height, bg, tolerance = 2) {
  let minX = width
  let minY = height
  let maxX = -1
  let maxY = -1
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const i = (y * width + x) * 4
      if (
        Math.abs(rgba[i] - bg[0]) > tolerance ||
        Math.abs(rgba[i + 1] - bg[1]) > tolerance ||
        Math.abs(rgba[i + 2] - bg[2]) > tolerance
      ) {
        minX = Math.min(minX, x)
        minY = Math.min(minY, y)
        maxX = Math.max(maxX, x)
        maxY = Math.max(maxY, y)
      }
    }
  }
  return {
    minX: maxX >= minX ? minX : 0,
    minY: maxY >= minY ? minY : 0,
    maxX,
    maxY,
    width: maxX >= minX ? maxX - minX + 1 : 0,
    height: maxY >= minY ? maxY - minY + 1 : 0,
  }
}
