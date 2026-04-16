'use strict'

const native = require('./native.js')

const DEFAULT_WIDTH = 512
const DEFAULT_HEIGHT = 512

// Three.js projection matrices use WebGL/OpenGL clip-space depth (-1..1).
// wgpu/WebGPU expects 0..1, so the public adapter converts clip-space here.
const OPENGL_TO_WGPU_CLIP = [
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 0.5, 0,
  0, 0, 0.5, 1,
]

class Renderer {
  constructor() {
    this.native = new native.NativeRenderer()
  }

  render(scene, camera, options = {}) {
    const { nativeScene, nativeCamera } = toNativeInput(scene, camera, options)
    return this.native.render(nativeScene, nativeCamera)
  }
}

function render(scene, camera, options = {}) {
  const { nativeScene, nativeCamera } = toNativeInput(scene, camera, options)
  return native.renderNative(nativeScene, nativeCamera)
}

function toNativeInput(scene, camera, options) {
  validateThreeScene(scene)
  validateThreeCamera(camera)

  if (typeof scene.updateMatrixWorld === 'function') {
    scene.updateMatrixWorld(true)
  }
  if (typeof camera.updateMatrixWorld === 'function') {
    camera.updateMatrixWorld(true)
  }

  const size = resolveSize(camera, options)
  const nativeScene = {
    width: size.width,
    height: size.height,
    background: resolveBackground(scene, options),
    format: options.format ?? 'png',
    meshes: flattenScene(scene),
    lights: extractLights(scene),
    ambientLight: extractAmbientLight(scene) ?? undefined,
    ambientIntensity: extractAmbientIntensity(scene) ?? undefined,
  }
  const nativeCamera = {
    width: size.width,
    height: size.height,
    viewProjection: cameraViewProjection(camera),
    cameraPosition: cameraWorldPosition(camera),
  }

  return { nativeScene, nativeCamera }
}

function validateThreeScene(scene) {
  if (!scene || scene.isScene !== true) {
    throw new TypeError('render(scene, camera) expects scene to be a THREE.Scene')
  }
}

function validateThreeCamera(camera) {
  if (!camera || camera.isCamera !== true) {
    throw new TypeError('render(scene, camera) expects camera to be a THREE.Camera')
  }
  if (!camera.projectionMatrix || !camera.matrixWorldInverse) {
    throw new TypeError('THREE.Camera must have projectionMatrix and matrixWorldInverse')
  }
}

function resolveSize(camera, options) {
  let width = numberOrUndefined(options.width)
  let height = numberOrUndefined(options.height)

  if (width == null && height == null) {
    width = numberOrUndefined(camera.userData?.width) ?? DEFAULT_WIDTH
    height = numberOrUndefined(camera.userData?.height)
  }
  if (height == null && width != null && isFinitePositive(camera.aspect)) {
    height = Math.round(width / camera.aspect)
  }
  if (width == null && height != null && isFinitePositive(camera.aspect)) {
    width = Math.round(height * camera.aspect)
  }

  width ??= DEFAULT_WIDTH
  height ??= DEFAULT_HEIGHT

  if (!Number.isInteger(width) || width <= 0) {
    throw new TypeError('render options width must be a positive integer')
  }
  if (!Number.isInteger(height) || height <= 0) {
    throw new TypeError('render options height must be a positive integer')
  }

  return { width, height }
}

function resolveBackground(scene, options) {
  if (Array.isArray(options.background)) {
    return normalizeColorArray(options.background)
  }

  const color = colorLikeToArray(options.background) ?? colorLikeToArray(scene.background)
  return color ?? [0.04, 0.045, 0.05, 1]
}

function flattenScene(scene) {
  const meshes = []
  visitObject(scene, true, meshes)
  return meshes
}

function visitObject(object, parentVisible, meshes) {
  if (!object || !parentVisible || object.visible === false) return

  if (object.isMesh === true && object.geometry) {
    appendMesh(object, meshes)
  }

  const children = Array.isArray(object.children) ? object.children : []
  for (const child of children) {
    visitObject(child, true, meshes)
  }
}

function appendMesh(object, meshes) {
  const geometry = object.geometry
  const position = getAttribute(geometry, 'position')
  if (!position) return

  const positions = readVec3Attribute(position)
  const uvAttribute = getAttribute(geometry, 'uv')
  const uvs = uvAttribute ? readVec2Attribute(uvAttribute) : null
  const normalAttribute = getAttribute(geometry, 'normal')
  const normals = normalAttribute ? readVec3Attribute(normalAttribute) : null
  const vertexColors = getAttribute(geometry, 'color')
  const index = geometry.index ? readIndexAttribute(geometry.index) : null
  const groups = effectiveGroups(geometry, index, position.count)

  for (const group of groups) {
    const material = materialForGroup(object.material, group.materialIndex)
    if (material?.visible === false) continue

    const color = materialColor(material)
    const useVertexColors = vertexColors && material?.vertexColors !== false
    const pbrProps = extractPbrProperties(material)

    const textureInfo = extractTextureData(material)

    if (index) {
      const indices = index.slice(group.start, group.start + group.count)
      if (indices.length % 3 !== 0) {
        throw new Error(`THREE.Mesh "${object.name || object.uuid || '<unnamed>'}" has a non-triangle index range`)
      }

      meshes.push({
        positions,
        indices,
        normals: normals ?? undefined,
        color,
        colors: useVertexColors ? readColorAttribute(vertexColors, color) : undefined,
        uvs: uvs ?? undefined,
        texture: textureInfo?.data,
        textureWidth: textureInfo?.width ?? undefined,
        textureHeight: textureInfo?.height ?? undefined,
        transform: matrixElements(object.matrixWorld, 'mesh.matrixWorld'),
        ...pbrProps,
      })
    } else {
      if (group.count % 3 !== 0) {
        throw new Error(`THREE.Mesh "${object.name || object.uuid || '<unnamed>'}" has a non-triangle vertex range`)
      }

      meshes.push({
        positions: positions.slice(group.start * 3, (group.start + group.count) * 3),
        normals: normals ? normals.slice(group.start * 3, (group.start + group.count) * 3) : undefined,
        color,
        colors: useVertexColors
          ? readColorAttribute(vertexColors, color).slice(group.start * 4, (group.start + group.count) * 4)
          : undefined,
        uvs: uvs ? uvs.slice(group.start * 2, (group.start + group.count) * 2) : undefined,
        texture: textureInfo?.data,
        textureWidth: textureInfo?.width ?? undefined,
        textureHeight: textureInfo?.height ?? undefined,
        transform: matrixElements(object.matrixWorld, 'mesh.matrixWorld'),
        ...pbrProps,
      })
    }
  }
}

function getAttribute(geometry, name) {
  if (typeof geometry.getAttribute === 'function') {
    return geometry.getAttribute(name)
  }
  return geometry.attributes?.[name]
}

function effectiveGroups(geometry, index, vertexCount) {
  const range = geometry.drawRange ?? {}
  const maxCount = index ? index.length : vertexCount
  const drawStart = clampInteger(range.start ?? 0, 0, maxCount)
  const requestedCount = range.count == null || range.count === Infinity ? maxCount : range.count
  const drawEnd = clampInteger(drawStart + requestedCount, drawStart, maxCount)
  const sourceGroups = Array.isArray(geometry.groups) && geometry.groups.length
    ? geometry.groups
    : [{ start: drawStart, count: drawEnd - drawStart, materialIndex: 0 }]

  const groups = []
  for (const group of sourceGroups) {
    const start = Math.max(drawStart, clampInteger(group.start ?? 0, 0, maxCount))
    const end = Math.min(drawEnd, clampInteger((group.start ?? 0) + (group.count ?? 0), 0, maxCount))
    if (end > start) {
      groups.push({
        start,
        count: end - start,
        materialIndex: group.materialIndex ?? 0,
      })
    }
  }
  return groups
}

function readVec3Attribute(attribute) {
  if (attribute.count == null) {
    throw new TypeError('THREE.BufferGeometry position attribute must have count')
  }
  const values = new Array(attribute.count * 3)
  for (let i = 0; i < attribute.count; i += 1) {
    values[i * 3] = attributeComponent(attribute, i, 0)
    values[i * 3 + 1] = attributeComponent(attribute, i, 1)
    values[i * 3 + 2] = attributeComponent(attribute, i, 2)
  }
  return values
}

function readVec2Attribute(attribute) {
  if (attribute.count == null) {
    throw new TypeError('THREE.BufferGeometry UV attribute must have count')
  }
  const values = new Array(attribute.count * 2)
  for (let i = 0; i < attribute.count; i += 1) {
    values[i * 2] = attributeComponent(attribute, i, 0)
    values[i * 2 + 1] = attributeComponent(attribute, i, 1)
  }
  return values
}

function readColorAttribute(attribute, materialColor) {
  const itemSize = attribute.itemSize ?? 3
  const values = new Array(attribute.count * 4)
  for (let i = 0; i < attribute.count; i += 1) {
    values[i * 4] = clamp01(attributeComponent(attribute, i, 0) * materialColor[0])
    values[i * 4 + 1] = clamp01(attributeComponent(attribute, i, 1) * materialColor[1])
    values[i * 4 + 2] = clamp01(attributeComponent(attribute, i, 2) * materialColor[2])
    values[i * 4 + 3] = clamp01((itemSize >= 4 ? attributeComponent(attribute, i, 3) : 1) * materialColor[3])
  }
  return values
}

function readIndexAttribute(attribute) {
  const values = new Array(attribute.count)
  for (let i = 0; i < attribute.count; i += 1) {
    values[i] = attributeComponent(attribute, i, 0)
  }
  return values
}

function attributeComponent(attribute, index, component) {
  if (component >= (attribute.itemSize ?? 1)) return 0

  let value
  if (component === 0 && typeof attribute.getX === 'function') value = attribute.getX(index)
  else if (component === 1 && typeof attribute.getY === 'function') value = attribute.getY(index)
  else if (component === 2 && typeof attribute.getZ === 'function') value = attribute.getZ(index)
  else if (component === 3 && typeof attribute.getW === 'function') value = attribute.getW(index)
  else {
    const array = attribute.array ?? attribute.data?.array
    const stride = attribute.data?.stride ?? attribute.itemSize ?? 1
    const offset = attribute.offset ?? 0
    value = array[index * stride + offset + component]
  }

  if (!Number.isFinite(value)) {
    throw new TypeError('THREE.BufferAttribute contains a non-finite value')
  }

  return attribute.normalized ? normalizeAttributeValue(value, attribute.array ?? attribute.data?.array) : value
}

function normalizeAttributeValue(value, array) {
  if (array instanceof Uint8Array || array instanceof Uint8ClampedArray) return value / 255
  if (array instanceof Uint16Array) return value / 65535
  if (array instanceof Int8Array) return Math.max(value / 127, -1)
  if (array instanceof Int16Array) return Math.max(value / 32767, -1)
  return value
}

function materialForGroup(material, materialIndex) {
  if (Array.isArray(material)) {
    return material[materialIndex] ?? material[0]
  }
  return material
}

function extractTextureData(material) {
  const map = material?.map
  if (!map) return null

  const image = map.image ?? map.source?.data
  if (!image) return null

  // DataTexture style: { data: TypedArray, width, height }
  if (image.data && image.width > 0 && image.height > 0) {
    const rgba = toRgba8(image.data, image.width, image.height)
    if (rgba) {
      return { data: Buffer.from(rgba.buffer, rgba.byteOffset, rgba.byteLength), width: image.width, height: image.height }
    }
  }

  // Encoded image (PNG/JPEG/WebP Buffer from file loaders)
  if (Buffer.isBuffer(image)) {
    return { data: image, width: 0, height: 0 }
  }
  if (image instanceof Uint8Array && !(image.width > 0)) {
    return { data: Buffer.from(image.buffer, image.byteOffset, image.byteLength), width: 0, height: 0 }
  }

  // ImageData (canvas-based polyfill): { data: Uint8ClampedArray, width, height }
  if (image.data instanceof Uint8ClampedArray && image.width > 0 && image.height > 0) {
    return {
      data: Buffer.from(image.data.buffer, image.data.byteOffset, image.data.byteLength),
      width: image.width,
      height: image.height,
    }
  }

  return null
}

function toRgba8(data, width, height) {
  const pixels = width * height

  if (data instanceof Uint8Array || data instanceof Uint8ClampedArray) {
    if (data.length === pixels * 4) return new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
    if (data.length === pixels * 3) {
      const out = new Uint8Array(pixels * 4)
      for (let i = 0; i < pixels; i++) {
        out[i * 4] = data[i * 3]
        out[i * 4 + 1] = data[i * 3 + 1]
        out[i * 4 + 2] = data[i * 3 + 2]
        out[i * 4 + 3] = 255
      }
      return out
    }
    return null
  }

  if (data instanceof Float32Array || data instanceof Float64Array) {
    if (data.length === pixels * 4) {
      const out = new Uint8Array(pixels * 4)
      for (let i = 0; i < pixels * 4; i++) {
        out[i] = Math.max(0, Math.min(255, Math.round(data[i] * 255)))
      }
      return out
    }
    if (data.length === pixels * 3) {
      const out = new Uint8Array(pixels * 4)
      for (let i = 0; i < pixels; i++) {
        out[i * 4] = Math.max(0, Math.min(255, Math.round(data[i * 3] * 255)))
        out[i * 4 + 1] = Math.max(0, Math.min(255, Math.round(data[i * 3 + 1] * 255)))
        out[i * 4 + 2] = Math.max(0, Math.min(255, Math.round(data[i * 3 + 2] * 255)))
        out[i * 4 + 3] = 255
      }
      return out
    }
    return null
  }

  // Uint16Array or other numeric typed arrays — treat as 8-bit range after clamping
  if (ArrayBuffer.isView(data) && data.length === pixels * 4) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels * 4; i++) {
      out[i] = Math.max(0, Math.min(255, data[i]))
    }
    return out
  }

  return null
}

function materialColor(material) {
  const color = colorLikeToArray(material?.color) ?? [1, 1, 1, 1]
  color[3] = clamp01(material?.opacity ?? color[3] ?? 1)
  return color
}

function colorLikeToArray(value) {
  if (!value) return null
  if (Array.isArray(value)) return normalizeColorArray(value)
  if (value.isColor === true || areFiniteNumbers(value.r, value.g, value.b)) {
    return [clamp01(value.r), clamp01(value.g), clamp01(value.b), 1]
  }
  return null
}

function normalizeColorArray(values) {
  if (values.length !== 3 && values.length !== 4) {
    throw new TypeError('Color arrays must be [r, g, b] or [r, g, b, a]')
  }
  return [
    clamp01(assertFinite(values[0], 'color r')),
    clamp01(assertFinite(values[1], 'color g')),
    clamp01(assertFinite(values[2], 'color b')),
    clamp01(values.length === 4 ? assertFinite(values[3], 'color a') : 1),
  ]
}

function cameraViewProjection(camera) {
  const projection = matrixElements(camera.projectionMatrix, 'camera.projectionMatrix')
  const view = matrixElements(camera.matrixWorldInverse, 'camera.matrixWorldInverse')
  return multiplyMatrices(OPENGL_TO_WGPU_CLIP, multiplyMatrices(projection, view))
}

function cameraWorldPosition(camera) {
  if (camera.matrixWorld?.elements) {
    const e = camera.matrixWorld.elements
    return [e[12], e[13], e[14]]
  }
  // Fallback: invert matrixWorldInverse
  if (camera.matrixWorldInverse?.elements) {
    const e = camera.matrixWorldInverse.elements
    // Extract translation from inverse of view matrix (column 3 of the inverse)
    // For a view matrix V, camera position = -(V^T * t) where t is the translation column
    const tx = e[12], ty = e[13], tz = e[14]
    return [
      -(e[0] * tx + e[1] * ty + e[2] * tz),
      -(e[4] * tx + e[5] * ty + e[6] * tz),
      -(e[8] * tx + e[9] * ty + e[10] * tz),
    ]
  }
  return [0, 0, 0]
}

function extractPbrProperties(material) {
  if (!material) return {}
  const props = {}

  // MeshStandardMaterial / MeshPhysicalMaterial
  if (Number.isFinite(material.metalness)) {
    props.metallic = clamp01(material.metalness)
  }
  if (Number.isFinite(material.roughness)) {
    props.roughness = clamp01(material.roughness)
  }

  // Emissive
  const emissive = colorLikeToArray(material.emissive)
  if (emissive) {
    props.emissive = [emissive[0], emissive[1], emissive[2]]
    props.emissiveIntensity = Number.isFinite(material.emissiveIntensity) ? material.emissiveIntensity : 1
  }

  // MeshBasicMaterial has no lighting
  if (material.isMeshBasicMaterial === true) {
    // Force metallic=0, roughness=1, and we'll handle no-light path
    // The shader handles "no lights" as a fallback ambient mode
  }

  return props
}

function extractLights(scene) {
  const lights = []
  visitLights(scene, lights)
  return lights.length > 0 ? lights : undefined
}

function visitLights(object, lights) {
  if (!object) return
  if (object.visible === false) return

  if (object.isLight === true) {
    const light = extractLight(object)
    if (light) lights.push(light)
  }

  const children = Array.isArray(object.children) ? object.children : []
  for (const child of children) {
    visitLights(child, lights)
  }
}

function extractLight(light) {
  const color = colorLikeToArray(light.color) ?? [1, 1, 1, 1]
  const intensity = Number.isFinite(light.intensity) ? light.intensity : 1

  if (light.isDirectionalLight === true) {
    // Three.js directional light: shines from position to target
    const pos = light.matrixWorld ? [light.matrixWorld.elements[12], light.matrixWorld.elements[13], light.matrixWorld.elements[14]] : [0, 10, 0]
    let targetPos = [0, 0, 0]
    if (light.target?.matrixWorld) {
      const te = light.target.matrixWorld.elements
      targetPos = [te[12], te[13], te[14]]
    }
    const direction = [
      targetPos[0] - pos[0],
      targetPos[1] - pos[1],
      targetPos[2] - pos[2],
    ]
    const len = Math.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
    if (len > 0) {
      direction[0] /= len
      direction[1] /= len
      direction[2] /= len
    }
    return {
      lightType: 'directional',
      color: [color[0], color[1], color[2]],
      intensity,
      direction,
    }
  }

  if (light.isPointLight === true) {
    const pos = light.matrixWorld ? [light.matrixWorld.elements[12], light.matrixWorld.elements[13], light.matrixWorld.elements[14]] : [0, 0, 0]
    return {
      lightType: 'point',
      color: [color[0], color[1], color[2]],
      intensity,
      position: pos,
    }
  }

  if (light.isSpotLight === true) {
    const pos = light.matrixWorld ? [light.matrixWorld.elements[12], light.matrixWorld.elements[13], light.matrixWorld.elements[14]] : [0, 0, 0]
    let targetPos = [0, 0, 0]
    if (light.target?.matrixWorld) {
      const te = light.target.matrixWorld.elements
      targetPos = [te[12], te[13], te[14]]
    }
    const direction = [
      targetPos[0] - pos[0],
      targetPos[1] - pos[1],
      targetPos[2] - pos[2],
    ]
    const len = Math.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
    if (len > 0) {
      direction[0] /= len
      direction[1] /= len
      direction[2] /= len
    }
    return {
      lightType: 'spot',
      color: [color[0], color[1], color[2]],
      intensity,
      position: pos,
      direction,
    }
  }

  // AmbientLight is handled separately
  return null
}

function extractAmbientLight(scene) {
  let color = null
  visitForAmbient(scene, (light) => {
    const c = colorLikeToArray(light.color) ?? [1, 1, 1, 1]
    if (!color) {
      color = [c[0], c[1], c[2]]
    } else {
      color[0] = Math.min(1, color[0] + c[0])
      color[1] = Math.min(1, color[1] + c[1])
      color[2] = Math.min(1, color[2] + c[2])
    }
  })
  return color
}

function extractAmbientIntensity(scene) {
  let intensity = 0
  visitForAmbient(scene, (light) => {
    intensity += Number.isFinite(light.intensity) ? light.intensity : 1
  })
  return intensity > 0 ? intensity : undefined
}

function visitForAmbient(object, callback) {
  if (!object) return
  if (object.visible === false) return
  if (object.isAmbientLight === true) callback(object)
  const children = Array.isArray(object.children) ? object.children : []
  for (const child of children) {
    visitForAmbient(child, callback)
  }
}

function matrixElements(matrix, label) {
  const elements = matrix?.elements
  if (!elements || elements.length !== 16) {
    throw new TypeError(`${label} must be a THREE.Matrix4`)
  }
  return Array.from(elements, (value) => assertFinite(value, label))
}

function multiplyMatrices(a, b) {
  const out = new Array(16)
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      out[column * 4 + row] =
        a[row] * b[column * 4] +
        a[4 + row] * b[column * 4 + 1] +
        a[8 + row] * b[column * 4 + 2] +
        a[12 + row] * b[column * 4 + 3]
    }
  }
  return out
}

function clampInteger(value, min, max) {
  if (!Number.isFinite(value)) return max
  return Math.max(min, Math.min(max, Math.trunc(value)))
}

function numberOrUndefined(value) {
  return Number.isFinite(value) ? value : undefined
}

function isFinitePositive(value) {
  return Number.isFinite(value) && value > 0
}

function assertFinite(value, label) {
  if (!Number.isFinite(value)) {
    throw new TypeError(`${label} must be finite`)
  }
  return value
}

function areFiniteNumbers(...values) {
  return values.every(Number.isFinite)
}

function clamp01(value) {
  return Math.min(1, Math.max(0, value))
}

module.exports = {
  Renderer,
  render,
}
