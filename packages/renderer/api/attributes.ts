import type { ThreeBufferAttributeLike, ThreeBufferGeometryLike, Color4 } from './types'
import { clamp01 } from './math'

const COMPONENT_LABELS = ['x', 'y', 'z', 'w']

export function getAttribute(geometry: ThreeBufferGeometryLike, name: string): ThreeBufferAttributeLike | undefined {
  const attributes = geometryAttributes(geometry)
  if (typeof geometry.getAttribute === 'function') {
    return geometry.getAttribute(name)
  }
  return attributes[name]
}

export function geometryAttributes(geometry: ThreeBufferGeometryLike): Record<string, ThreeBufferAttributeLike | undefined> {
  const attributes = geometry.attributes
  if (attributes == null) return {}
  if (typeof attributes !== 'object' || Array.isArray(attributes)) {
    throw new TypeError('geometry.attributes must be an object.')
  }
  return attributes
}

export function readVec3Attribute(attribute: ThreeBufferAttributeLike, label = 'THREE.BufferAttribute'): number[] {
  const count = attributeCount(attribute, label)
  const values = new Array<number>(count * 3)
  for (let i = 0; i < count; i += 1) {
    values[i * 3] = attributeComponent(attribute, i, 0, label)
    values[i * 3 + 1] = attributeComponent(attribute, i, 1, label)
    values[i * 3 + 2] = attributeComponent(attribute, i, 2, label)
  }
  return values
}

export function readVec2Attribute(attribute: ThreeBufferAttributeLike, label = 'THREE.BufferAttribute'): number[] {
  const count = attributeCount(attribute, label)
  const values = new Array<number>(count * 2)
  for (let i = 0; i < count; i += 1) {
    values[i * 2] = attributeComponent(attribute, i, 0, label)
    values[i * 2 + 1] = attributeComponent(attribute, i, 1, label)
  }
  return values
}

export function readColorAttribute(attribute: ThreeBufferAttributeLike, materialColor: Color4, label = 'THREE.BufferAttribute'): number[] {
  const count = attributeCount(attribute, label)
  const itemSize = attributeItemSize(attribute, label) ?? 3
  const values = new Array<number>(count * 4)
  for (let i = 0; i < count; i += 1) {
    values[i * 4] = clamp01(attributeComponent(attribute, i, 0, label) * materialColor[0])
    values[i * 4 + 1] = clamp01(attributeComponent(attribute, i, 1, label) * materialColor[1])
    values[i * 4 + 2] = clamp01(attributeComponent(attribute, i, 2, label) * materialColor[2])
    values[i * 4 + 3] = clamp01((itemSize >= 4 ? attributeComponent(attribute, i, 3, label) : 1) * materialColor[3])
  }
  return values
}

export function readIndexAttribute(
  attribute: ThreeBufferAttributeLike,
  label = 'THREE.BufferAttribute',
  vertexCount?: number,
): number[] {
  const count = attributeCount(attribute, label)
  const values = new Array<number>(count)
  for (let i = 0; i < count; i += 1) {
    const value = attributeComponent(attribute, i, 0, label)
    if (!Number.isInteger(value) || value < 0) {
      throw new TypeError(`${label}[${i}].x must be a non-negative integer.`)
    }
    if (vertexCount !== undefined && value >= vertexCount) {
      throw new RangeError(`${label}[${i}].x must reference a vertex below geometry.attributes.position.count (${vertexCount}).`)
    }
    values[i] = value
  }
  return values
}

export function attributeCount(attribute: ThreeBufferAttributeLike, label: string): number {
  if (Number.isInteger(attribute.count) && attribute.count >= 0) return attribute.count
  throw new TypeError(`${label}.count must be a non-negative integer.`)
}

export function attributeComponent(
  attribute: ThreeBufferAttributeLike,
  index: number,
  component: number,
  label = 'THREE.BufferAttribute',
): number {
  const itemSize = attributeItemSize(attribute, label) ?? 1
  if (component >= itemSize) return 0

  let value: number | undefined
  let readFromRawArray = false
  if (component === 0 && typeof attribute.getX === 'function') value = attribute.getX(index)
  else if (component === 1 && typeof attribute.getY === 'function') value = attribute.getY(index)
  else if (component === 2 && typeof attribute.getZ === 'function') value = attribute.getZ(index)
  else if (component === 3 && typeof attribute.getW === 'function') value = attribute.getW(index)
  else {
    const data = attributeData(attribute, label)
    const array = attribute.array ?? data?.array
    const stride = attributeStride(attribute, label) ?? itemSize
    const offset = attributeOffset(attribute, label)
    value = array?.[index * stride + offset + component]
    readFromRawArray = true
  }

  if (!Number.isFinite(value)) {
    throw new TypeError(`${label}[${index}].${COMPONENT_LABELS[component] ?? component} must be a finite number.`)
  }

  return attributeNormalized(attribute, label) && readFromRawArray
    ? normalizeAttributeValue(value!, attribute.array ?? attributeData(attribute, label)?.array)
    : value!
}

function attributeItemSize(attribute: ThreeBufferAttributeLike, label: string): number | undefined {
  if (attribute.itemSize == null) return undefined
  if (Number.isInteger(attribute.itemSize) && attribute.itemSize > 0) return attribute.itemSize
  throw new TypeError(`${label}.itemSize must be a positive integer.`)
}

function attributeStride(attribute: ThreeBufferAttributeLike, label: string): number | undefined {
  const stride = attributeData(attribute, label)?.stride
  if (stride == null) return undefined
  if (Number.isInteger(stride) && stride > 0) return stride
  throw new TypeError(`${label}.data.stride must be a positive integer.`)
}

function attributeData(attribute: ThreeBufferAttributeLike, label: string): ThreeBufferAttributeLike['data'] | undefined {
  const data = attribute.data
  if (data == null) return undefined
  if (typeof data !== 'object' || Array.isArray(data)) {
    throw new TypeError(`${label}.data must be an object.`)
  }
  return data
}

function attributeOffset(attribute: ThreeBufferAttributeLike, label: string): number {
  if (attribute.offset == null) return 0
  if (Number.isInteger(attribute.offset) && attribute.offset >= 0) return attribute.offset
  throw new TypeError(`${label}.offset must be a non-negative integer.`)
}

function attributeNormalized(attribute: ThreeBufferAttributeLike, label: string): boolean {
  if (attribute.normalized == null) return false
  if (typeof attribute.normalized === 'boolean') return attribute.normalized
  throw new TypeError(`${label}.normalized must be a boolean.`)
}

function normalizeAttributeValue(value: number, array: ArrayLike<number> | undefined): number {
  if (array instanceof Uint8Array || array instanceof Uint8ClampedArray) return value / 255
  if (array instanceof Uint16Array) return value / 65535
  if (array instanceof Int8Array) return Math.max(value / 127, -1)
  if (array instanceof Int16Array) return Math.max(value / 32767, -1)
  return value
}
