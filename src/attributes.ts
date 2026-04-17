import type { ThreeBufferAttributeLike, ThreeBufferGeometryLike, Color4 } from './types'
import { clamp01 } from './math'

export function getAttribute(geometry: ThreeBufferGeometryLike, name: string): ThreeBufferAttributeLike | undefined {
  if (typeof geometry.getAttribute === 'function') {
    return geometry.getAttribute(name)
  }
  return geometry.attributes?.[name]
}

export function readVec3Attribute(attribute: ThreeBufferAttributeLike): number[] {
  if (attribute.count == null) {
    throw new TypeError('THREE.BufferGeometry position attribute must have count')
  }
  const values = new Array<number>(attribute.count * 3)
  for (let i = 0; i < attribute.count; i += 1) {
    values[i * 3] = attributeComponent(attribute, i, 0)
    values[i * 3 + 1] = attributeComponent(attribute, i, 1)
    values[i * 3 + 2] = attributeComponent(attribute, i, 2)
  }
  return values
}

export function readVec2Attribute(attribute: ThreeBufferAttributeLike): number[] {
  if (attribute.count == null) {
    throw new TypeError('THREE.BufferGeometry UV attribute must have count')
  }
  const values = new Array<number>(attribute.count * 2)
  for (let i = 0; i < attribute.count; i += 1) {
    values[i * 2] = attributeComponent(attribute, i, 0)
    values[i * 2 + 1] = attributeComponent(attribute, i, 1)
  }
  return values
}

export function readColorAttribute(attribute: ThreeBufferAttributeLike, materialColor: Color4): number[] {
  const itemSize = attribute.itemSize ?? 3
  const values = new Array<number>(attribute.count * 4)
  for (let i = 0; i < attribute.count; i += 1) {
    values[i * 4] = clamp01(attributeComponent(attribute, i, 0) * materialColor[0])
    values[i * 4 + 1] = clamp01(attributeComponent(attribute, i, 1) * materialColor[1])
    values[i * 4 + 2] = clamp01(attributeComponent(attribute, i, 2) * materialColor[2])
    values[i * 4 + 3] = clamp01((itemSize >= 4 ? attributeComponent(attribute, i, 3) : 1) * materialColor[3])
  }
  return values
}

export function readIndexAttribute(attribute: ThreeBufferAttributeLike): number[] {
  const values = new Array<number>(attribute.count)
  for (let i = 0; i < attribute.count; i += 1) {
    values[i] = attributeComponent(attribute, i, 0)
  }
  return values
}

export function attributeComponent(attribute: ThreeBufferAttributeLike, index: number, component: number): number {
  if (component >= (attribute.itemSize ?? 1)) return 0

  let value: number | undefined
  if (component === 0 && typeof attribute.getX === 'function') value = attribute.getX(index)
  else if (component === 1 && typeof attribute.getY === 'function') value = attribute.getY(index)
  else if (component === 2 && typeof attribute.getZ === 'function') value = attribute.getZ(index)
  else if (component === 3 && typeof attribute.getW === 'function') value = attribute.getW(index)
  else {
    const array = attribute.array ?? attribute.data?.array
    const stride = attribute.data?.stride ?? attribute.itemSize ?? 1
    const offset = attribute.offset ?? 0
    value = array?.[index * stride + offset + component]
  }

  if (!Number.isFinite(value)) {
    throw new TypeError('THREE.BufferAttribute contains a non-finite value')
  }

  return attribute.normalized ? normalizeAttributeValue(value!, attribute.array ?? attribute.data?.array) : value!
}

function normalizeAttributeValue(value: number, array: ArrayLike<number> | undefined): number {
  if (array instanceof Uint8Array || array instanceof Uint8ClampedArray) return value / 255
  if (array instanceof Uint16Array) return value / 65535
  if (array instanceof Int8Array) return Math.max(value / 127, -1)
  if (array instanceof Int16Array) return Math.max(value / 32767, -1)
  return value
}
