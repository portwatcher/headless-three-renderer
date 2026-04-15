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
  }
  const nativeCamera = {
    width: size.width,
    height: size.height,
    viewProjection: cameraViewProjection(camera),
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
  const vertexColors = getAttribute(geometry, 'color')
  const index = geometry.index ? readIndexAttribute(geometry.index) : null
  const groups = effectiveGroups(geometry, index, position.count)

  for (const group of groups) {
    const material = materialForGroup(object.material, group.materialIndex)
    if (material?.visible === false) continue

    const color = materialColor(material)
    const useVertexColors = vertexColors && material?.vertexColors !== false

    if (index) {
      const indices = index.slice(group.start, group.start + group.count)
      if (indices.length % 3 !== 0) {
        throw new Error(`THREE.Mesh "${object.name || object.uuid || '<unnamed>'}" has a non-triangle index range`)
      }

      meshes.push({
        positions,
        indices,
        color,
        colors: useVertexColors ? readColorAttribute(vertexColors, color) : undefined,
        transform: matrixElements(object.matrixWorld, 'mesh.matrixWorld'),
      })
    } else {
      if (group.count % 3 !== 0) {
        throw new Error(`THREE.Mesh "${object.name || object.uuid || '<unnamed>'}" has a non-triangle vertex range`)
      }

      meshes.push({
        positions: positions.slice(group.start * 3, (group.start + group.count) * 3),
        color,
        colors: useVertexColors
          ? readColorAttribute(vertexColors, color).slice(group.start * 4, (group.start + group.count) * 4)
          : undefined,
        transform: matrixElements(object.matrixWorld, 'mesh.matrixWorld'),
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
