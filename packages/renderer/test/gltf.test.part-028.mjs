import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'
import { REAL_VRMA_ANIMATION_SAMPLE, REAL_VRM_EXPRESSION_OVERRIDES_SAMPLE, REAL_VRM_EXPRESSION_SAMPLE, Renderer, SIMPLE_TRIANGLE, TEXTURED_QUAD, loadGltfFromFile, loadVrmAnimationFromFile, loadVrmFromFile } from './gltf.test.part-001.mjs'
test('real external VRM and VRMA fixtures expose extension metadata through loader helpers', async () => {
  const vrmPluginParsers = []
  let animationPluginParser = null

  class CaptureVRMLoaderPlugin {
    constructor(parser) {
      this.name = 'CaptureVRMLoaderPlugin'
      vrmPluginParsers.push(parser)
    }
  }

  class CaptureVRMAnimationLoaderPlugin {
    constructor(parser) {
      this.name = 'CaptureVRMAnimationLoaderPlugin'
      animationPluginParser = parser
    }
  }

  const vrmGltf = await loadVrmFromFile(REAL_VRM_EXPRESSION_SAMPLE, {
    VRMLoaderPlugin: CaptureVRMLoaderPlugin,
  })
  const vrmPluginParser = vrmPluginParsers.at(-1)
  assert.ok(findFirst(vrmGltf.scene, (object) => object.isMesh === true), 'real VRM fixture should parse renderable meshes')
  assert.ok(vrmPluginParser, 'real VRM fixture should initialize the supplied VRM loader plugin')
  assert.ok(vrmPluginParser.json?.extensionsUsed?.includes('VRMC_vrm'), 'real VRM fixture should expose VRMC_vrm metadata')
  assert.ok(
    vrmPluginParser.json?.extensionsUsed?.includes('KHR_texture_transform'),
    'real VRM fixture should expose its texture transform extension metadata',
  )
  assert.equal(vrmPluginParser.json?.meshes?.length, 4)

  const vrmExtension = vrmPluginParser.json?.extensions?.VRMC_vrm
  assert.equal(vrmExtension?.specVersion, '1.0')
  assert.equal(vrmExtension?.meta?.name, 'isBinary overridden')
  assert.equal(vrmExtension?.meta?.licenseUrl, 'https://vrm.dev/licenses/1.0/')
  assert.equal(vrmExtension?.meta?.allowRedistribution, true)
  assert.equal(vrmExtension?.expressions?.preset?.happy?.overrideBlink, 'blend')
  assert.equal(vrmExtension?.expressions?.preset?.blink?.isBinary, true)

  const overridesVrmGltf = await loadVrmFromFile(REAL_VRM_EXPRESSION_OVERRIDES_SAMPLE, {
    VRMLoaderPlugin: CaptureVRMLoaderPlugin,
  })
  const overridesPluginParser = vrmPluginParsers.at(-1)
  assert.ok(findFirst(overridesVrmGltf.scene, (object) => object.isMesh === true), 'real VRM overrides fixture should parse renderable meshes')
  assert.ok(overridesPluginParser, 'real VRM overrides fixture should initialize the supplied VRM loader plugin')
  assert.ok(
    overridesPluginParser.json?.extensionsUsed?.includes('VRMC_vrm'),
    'real VRM overrides fixture should expose VRMC_vrm metadata',
  )
  assert.equal(overridesPluginParser.json?.meshes?.length, 4)

  const overridesVrmExtension = overridesPluginParser.json?.extensions?.VRMC_vrm
  assert.equal(overridesVrmExtension?.specVersion, '1.0')
  assert.equal(overridesVrmExtension?.meta?.name, 'isBinary overrides')
  assert.equal(overridesVrmExtension?.meta?.licenseUrl, 'https://vrm.dev/licenses/1.0/')
  assert.equal(overridesVrmExtension?.meta?.allowRedistribution, true)
  assert.equal(overridesVrmExtension?.expressions?.preset?.happy?.isBinary, true)
  assert.equal(overridesVrmExtension?.expressions?.preset?.happy?.overrideBlink, 'blend')
  assert.equal(overridesVrmExtension?.expressions?.preset?.blink?.isBinary, undefined)

  const animationGltf = await loadVrmAnimationFromFile(REAL_VRMA_ANIMATION_SAMPLE, {
    VRMAnimationLoaderPlugin: CaptureVRMAnimationLoaderPlugin,
  })
  assert.equal(animationGltf.animations.length, 1)
  assert.ok(animationPluginParser, 'real VRMA fixture should initialize the supplied VRM animation loader plugin')
  assert.ok(
    animationPluginParser.json?.extensionsUsed?.includes('VRMC_vrm_animation'),
    'real VRMA fixture should expose VRMC_vrm_animation metadata',
  )
  assert.equal(animationPluginParser.json?.nodes?.length, 53)
  assert.equal(animationPluginParser.json?.animations?.[0]?.channels?.length, 3)
  assert.equal(animationPluginParser.json?.animations?.[0]?.samplers?.length, 3)

  const vrmaExtension = animationPluginParser.json?.extensions?.VRMC_vrm_animation
  const humanBones = vrmaExtension?.humanoid?.humanBones ?? {}
  assert.equal(vrmaExtension?.specVersion, '1.0')
  assert.ok('hips' in humanBones, 'real VRMA fixture should map humanoid hips')
  assert.ok('leftUpperArm' in humanBones, 'real VRMA fixture should map humanoid upper-body bones')
  assert.ok('rightFoot' in humanBones, 'real VRMA fixture should map humanoid lower-body bones')
  assert.equal(vrmaExtension?.lookAt?.node, 52)
  assert.equal(vrmaExtension?.expressions?.preset?.happy?.node, 51)
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

export function assertTexturedQuadLoadsEncodedMap(gltf, label) {
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, `${label} should load a mesh`)
  assert.ok(mesh.material.map?.isTexture, `${label} should load a base color texture`)
  assert.ok(Buffer.isBuffer(mesh.material.map.image), 'encoded image helper should expose the PNG as a Buffer')
}

export function assertTexturedQuadRendersTexture(gltf, label) {
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

export function assertTextureSampler(mesh, wrapS, wrapT) {
  assert.ok(mesh?.isMesh, 'texture sampler assertion requires a mesh')
  const texture = mesh.material.map
  assert.ok(texture?.isTexture, `${mesh.name} should load a base color texture`)
  assert.equal(Buffer.isBuffer(texture.image), true, `${mesh.name} texture should load as an encoded Buffer`)
  assert.equal(texture.wrapS, wrapS, `${mesh.name} should preserve sampler wrapS`)
  assert.equal(texture.wrapT, wrapT, `${mesh.name} should preserve sampler wrapT`)
  assert.equal(texture.magFilter, THREE.LinearFilter)
  assert.equal(texture.minFilter, THREE.NearestMipmapLinearFilter)
  assert.equal(texture.flipY, false)
}

export async function loadGltfFixture(filePath, options) {
  return await loadGltfFromFile(filePath, options)
}

export async function captureConsoleWarn(callback) {
  const originalWarn = console.warn
  const warnings = []
  console.warn = (...args) => {
    warnings.push(args.map((arg) => String(arg)).join(' '))
  }
  try {
    const result = await callback()
    return { result, warnings }
  } finally {
    console.warn = originalWarn
  }
}

export function vectorFromAttribute(attribute, index) {
  return [attribute.getX(index), attribute.getY(index), attribute.getZ(index)]
}

export function pngDimensions(buffer) {
  assert.equal(Buffer.isBuffer(buffer), true, 'PNG source should be an encoded Buffer')
  assert.equal(buffer.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])), true, 'PNG source should start with a PNG signature')
  return [buffer.readUInt32BE(16), buffer.readUInt32BE(20)]
}

export function assertWebpBuffer(buffer, label) {
  assert.equal(Buffer.isBuffer(buffer), true, `${label} should be an encoded Buffer`)
  assert.equal(buffer.subarray(0, 4).toString('ascii'), 'RIFF', `${label} should start with a RIFF header`)
  assert.equal(buffer.subarray(8, 12).toString('ascii'), 'WEBP', `${label} should be a WebP payload`)
}

export function assertVectorClose(actual, expected, label, tolerance = 1e-6) {
  assert.equal(actual.length, expected.length, `${label} should have ${expected.length} components`)
  for (let i = 0; i < expected.length; i++) {
    assert.ok(Math.abs(actual[i] - expected[i]) <= tolerance, `${label}[${i}] should be close to ${expected[i]} (${actual[i]})`)
  }
}

export function isEffectivelyVisible(object) {
  let current = object
  while (current) {
    if (current.visible === false) return false
    current = current.parent
  }
  return true
}

export function worldDeterminant(object) {
  object.updateWorldMatrix(true, false)
  return object.matrixWorld.determinant()
}

export function renderSingleObjectRatio(renderer, object, padding = 0.2) {
  object.updateWorldMatrix(true, true)
  const bounds = new THREE.Box3().setFromObject(object)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())

  const scene = new THREE.Scene()
  scene.add(object.clone(true))
  scene.add(new THREE.AmbientLight(0xffffff, 1.0))

  const camera = new THREE.OrthographicCamera(
    -size.x / 2 - padding,
    size.x / 2 + padding,
    size.y / 2 + padding,
    -size.y / 2 - padding,
    0.01,
    20,
  )
  camera.position.set(center.x, center.y, center.z + 8)
  camera.lookAt(center)
  scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = renderer.render(scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })
  return nonBackgroundRatio(rgba, [0, 0, 0], 3)
}

export async function assertRejectsMutatedGltfSource(mutator, pattern) {
  const source = JSON.parse(await readFile(TEXTURED_QUAD, 'utf8'))
  mutator(source)
  const tmp = await mkdtemp(path.join(os.tmpdir(), 'headless-three-gltf-malformed-image-'))
  try {
    const modelPath = path.join(tmp, 'malformed-image-metadata.gltf')
    await writeFile(modelPath, JSON.stringify(source))
    await assert.rejects(
      () => loadGltfFixture(modelPath),
      pattern,
    )
  } finally {
    await rm(tmp, { recursive: true, force: true })
  }
}

export function buildTexturedQuadGlb(source) {
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

export function decodeDataUriBuffer(uri, label) {
  assert.equal(typeof uri, 'string', `${label} should be a data URI`)
  const comma = uri.indexOf(',')
  assert.notEqual(comma, -1, `${label} should contain a comma separator`)
  const metadata = uri.slice(5, comma)
  const payload = uri.slice(comma + 1)
  return /(?:^|;)base64(?:;|$)/i.test(metadata)
    ? Buffer.from(payload, 'base64')
    : Buffer.from(decodeURIComponent(payload), 'utf8')
}

export function encodeGlb(json, bin) {
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

export function paddedBuffer(buffer, fill) {
  const padded = Buffer.alloc(alignedLength(buffer.length), fill)
  buffer.copy(padded)
  return padded
}

export function alignedLength(length) {
  return (length + 3) & ~3
}

export function writeUint32(buffer, offset, value) {
  buffer.writeUInt32LE(value, offset)
  return offset + 4
}

export function findFirst(root, predicate) {
  let match = null
  root.traverse((object) => {
    if (!match && predicate(object)) match = object
  })
  return match
}

export function uniqueMaterials(root) {
  const materials = []
  const seen = new Set()
  root.traverse((object) => {
    const objectMaterials = Array.isArray(object.material) ? object.material : object.material ? [object.material] : []
    for (const material of objectMaterials) {
      if (seen.has(material.uuid)) continue
      seen.add(material.uuid)
      materials.push(material)
    }
  })
  return materials
}

export function frameSceneCamera(scene, { fov = 35, xOffset = 0.8, yOffset = 0.35, distance = 2.4 } = {}) {
  scene.updateMatrixWorld(true)
  const bounds = new THREE.Box3().setFromObject(scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const radius = Math.max(size.length() * 0.55, 0.001)
  const camera = new THREE.PerspectiveCamera(fov, 1, radius / 100, radius * 25)
  camera.position.set(center.x + radius * xOffset, center.y + radius * yOffset, center.z + radius * distance)
  camera.lookAt(center)
  camera.updateMatrixWorld(true)
  camera.updateProjectionMatrix()
  return camera
}

export function meanRegion(rgba, width, _height, x0, y0, x1, y1) {
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

export function nonBackgroundBounds(rgba, width, height, bg, tolerance = 2) {
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
