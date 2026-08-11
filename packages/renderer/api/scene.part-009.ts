import type {
  ThreeObject3DLike,
  ThreeBufferAttributeLike,
  ThreeBufferGeometryLike,
  ThreeCameraLike,
  ThreeMaterialLike,
  ThreeSphereLike,
  NativeSceneMesh,
  GeometryGroup,
  Color4,
  PbrProperties,
  RenderSortFunction,
  RenderSortItem,
} from './types'
import { IDENTITY_4X4, matrixElements, clampInteger, clamp01 } from './math'
import {
  attributeComponent,
  attributeCount,
  getAttribute,
  readVec3Attribute,
  readVec2Attribute,
  readColorAttribute,
  readIndexAttribute,
  geometryAttributes,
} from './attributes'
import {
  materialForGroup,
  materialColor,
  extractPbrProperties,
  extractTextureData,
  materialShadowSide,
  textureUvChannel,
  assertMaterialLike,
  type MaterialExtractionContext,
  type TextureExtractionCache,
  type MaterialColorExtractionCache,
  type TextureStateExtractionCache,
  type MaterialRenderStateExtractionCache,
  type MaterialScalarFeatureExtractionCache,
} from './materials'
import { applyCpuSkinning } from './skinning'
import { applyMorphTargets } from './morphs'
import { objectLayersMatchCamera } from './layers'
import { objectChildren } from './objects'
import {
  MAX_CLIPPING_PLANES,
  type NativeClippingPlane,
  extractClippingPlanes,
  flattenClippingPlanes,
} from './clipping'
import { InstancedAttributeRef, InstancedNormalExpansionSignature, InstancedPositionExpansionSignature, InstancedUvExpansionSignature, SceneExtractionCache, SupportedInstancedBufferGeometryAttributes, UvChannel } from './scene.part-001'
import { attributeSignature, attributeSignatureCacheable, sameAttributeSignature } from './scene.part-003'
import { sameUvChannelSignature, uvChannelSignature } from './scene.part-005'
import { integerCountOrDefault } from './scene.part-008'
import { labelForMeshPerAttribute } from './scene.part-010'
export function instancedBufferGeometryCount(geometry: ThreeBufferGeometryLike): number {
  const attributes = Object.entries(geometryAttributes(geometry))
  const instancedAttributes = attributes.filter((entry): entry is [string, ThreeBufferAttributeLike] => isInstancedAttribute(entry[1]))
  if (geometry.isInstancedBufferGeometry !== true && instancedAttributes.length === 0) return 1

  let maxCount = Infinity
  for (const [name, attribute] of instancedAttributes) {
    const label = `geometry.attributes.${name}`
    maxCount = Math.min(maxCount, attributeCount(attribute, label) * meshPerAttribute(attribute, `${label}.meshPerAttribute`))
  }

  const requested = integerCountOrDefault(geometry.instanceCount, 'geometry.instanceCount', Infinity, true)
  const effectiveCount = Math.min(requested, maxCount)
  if (effectiveCount === Infinity) return 1
  return clampInteger(effectiveCount, 0, Math.max(0, Math.floor(maxCount)))
}

export function assertSupportedCustomFragmentInstancedAttributes(
  geometry: ThreeBufferGeometryLike,
  pbrProps: Pick<PbrProperties, 'customFragmentShader'>,
): void {
  if (!pbrProps.customFragmentShader) return

  const unsupported: string[] = []
  for (const [name, attribute] of Object.entries(geometryAttributes(geometry))) {
    if (!isInstancedAttribute(attribute)) continue
    if (!SupportedInstancedBufferGeometryAttributes.has(name)) unsupported.push(`geometry.attributes.${name}`)
  }
  if (unsupported.length === 0) return

  throw new Error(
    `Custom WGSL fragment materials cannot bind arbitrary InstancedBufferGeometry vertex attributes (${unsupported.join(', ')}) in @headless-three/renderer yet. Use supported instanced offset, color, normal, or UV attributes, expand the geometry on the CPU, or render separate draws until custom vertex-attribute shader integration is implemented.`,
  )
}

export function isInstancedAttribute(attribute: ThreeBufferAttributeLike | undefined | null): attribute is ThreeBufferAttributeLike {
  return attribute?.isInstancedBufferAttribute === true
}

export function meshPerAttribute(attribute: ThreeBufferAttributeLike, label = 'InstancedBufferAttribute.meshPerAttribute'): number {
  const value = attribute.meshPerAttribute
  if (value == null) return 1
  if (typeof value === 'number' && Number.isFinite(value)) {
    if (value <= 0) {
      throw new TypeError(`${label} must be a positive finite number.`)
    }
    if (!Number.isInteger(value)) {
      throw new TypeError(`${label} must be a positive integer.`)
    }
    return value
  }
  throw new TypeError(`${label} must be a positive finite number.`)
}

export function instancedAttributeIndex(
  attribute: ThreeBufferAttributeLike,
  instanceIndex: number,
  label = 'InstancedBufferAttribute',
): number {
  return Math.min(
    attributeCount(attribute, label) - 1,
    Math.floor(instanceIndex / meshPerAttribute(attribute, `${label}.meshPerAttribute`)),
  )
}

export function instancedOffsetAttribute(geometry: ThreeBufferGeometryLike): InstancedAttributeRef | null {
  const names = ['instanceOffset', 'instancePosition', 'offset', 'translate', 'translation']
  for (const name of names) {
    const attribute = getAttribute(geometry, name)
    if (isInstancedAttribute(attribute)) return { attribute, label: `geometry.attributes.${name}` }
  }
  return null
}

export function instancedScaleAttribute(geometry: ThreeBufferGeometryLike): InstancedAttributeRef | null {
  const names = ['instanceScale', 'scale']
  for (const name of names) {
    const attribute = getAttribute(geometry, name)
    if (isInstancedAttribute(attribute)) return { attribute, label: `geometry.attributes.${name}` }
  }
  return null
}

export function expandVec3ValuesForInstances(
  values: number[],
  start: number,
  count: number,
  instanceCount: number,
  offsetAttribute?: InstancedAttributeRef | null,
  scaleAttribute?: InstancedAttributeRef | null,
): number[] {
  if (instanceCount <= 1 && !offsetAttribute && !scaleAttribute) {
    return values.slice(start * 3, (start + count) * 3)
  }
  const out = new Array<number>(count * instanceCount * 3)
  let dst = 0
  for (let instance = 0; instance < instanceCount; instance += 1) {
    const offsetIndex = offsetAttribute
      ? instancedAttributeIndex(offsetAttribute.attribute, instance, offsetAttribute.label)
      : 0
    const ox = offsetAttribute ? attributeComponent(offsetAttribute.attribute, offsetIndex, 0, offsetAttribute.label) : 0
    const oy = offsetAttribute ? attributeComponent(offsetAttribute.attribute, offsetIndex, 1, offsetAttribute.label) : 0
    const oz = offsetAttribute ? attributeComponent(offsetAttribute.attribute, offsetIndex, 2, offsetAttribute.label) : 0
    const scale = instanceScaleComponents(scaleAttribute, instance)
    for (let vertex = start; vertex < start + count; vertex += 1) {
      out[dst++] = values[vertex * 3] * scale[0] + ox
      out[dst++] = values[vertex * 3 + 1] * scale[1] + oy
      out[dst++] = values[vertex * 3 + 2] * scale[2] + oz
    }
  }
  return out
}

export function expandVec3ValuesForInstancesWithCache(
  cache: SceneExtractionCache | undefined,
  geometry: ThreeBufferGeometryLike,
  position: ThreeBufferAttributeLike,
  values: number[],
  start: number,
  count: number,
  instanceCount: number,
  offsetAttribute?: InstancedAttributeRef | null,
  scaleAttribute?: InstancedAttributeRef | null,
): number[] {
  if (!cache) {
    return expandVec3ValuesForInstances(values, start, count, instanceCount, offsetAttribute, scaleAttribute)
  }

  const signature = instancedPositionExpansionSignature(
    geometry,
    position,
    values,
    start,
    count,
    instanceCount,
    offsetAttribute,
    scaleAttribute,
  )
  if (!signature.cacheable) {
    return expandVec3ValuesForInstances(values, start, count, instanceCount, offsetAttribute, scaleAttribute)
  }

  const key = `${start}:${count}:${instanceCount}`
  let geometryCache = cache.instancedPositionExpansions.get(geometry)
  const cached = geometryCache?.get(key)
  if (cached && sameInstancedPositionExpansionSignature(cached.signature, signature)) {
    return cached.positions
  }

  const positions = expandVec3ValuesForInstances(values, start, count, instanceCount, offsetAttribute, scaleAttribute)
  if (!geometryCache) {
    geometryCache = new Map()
    cache.instancedPositionExpansions.set(geometry, geometryCache)
  }
  geometryCache.set(key, { signature, positions })
  return positions
}

export function instancedPositionExpansionSignature(
  geometry: ThreeBufferGeometryLike,
  position: ThreeBufferAttributeLike,
  values: number[],
  start: number,
  count: number,
  instanceCount: number,
  offsetAttribute?: InstancedAttributeRef | null,
  scaleAttribute?: InstancedAttributeRef | null,
): InstancedPositionExpansionSignature {
  const signature: InstancedPositionExpansionSignature = {
    cacheable: true,
    geometryVersion: geometry.version,
    sourcePositions: values,
    start,
    count,
    instanceCount,
    position: attributeSignature(position),
    instancedPositionOffset: attributeSignature(offsetAttribute?.attribute),
    instancedPositionScale: attributeSignature(scaleAttribute?.attribute),
  }
  signature.cacheable = attributeSignatureCacheable(signature.position)
    && attributeSignatureCacheable(signature.instancedPositionOffset)
    && attributeSignatureCacheable(signature.instancedPositionScale)
  return signature
}

export function sameInstancedPositionExpansionSignature(
  a: InstancedPositionExpansionSignature,
  b: InstancedPositionExpansionSignature,
): boolean {
  return a.cacheable === b.cacheable
    && a.geometryVersion === b.geometryVersion
    && a.sourcePositions === b.sourcePositions
    && a.start === b.start
    && a.count === b.count
    && a.instanceCount === b.instanceCount
    && sameAttributeSignature(a.position, b.position)
    && sameAttributeSignature(a.instancedPositionOffset, b.instancedPositionOffset)
    && sameAttributeSignature(a.instancedPositionScale, b.instancedPositionScale)
}

export function instanceScaleComponents(
  scaleAttribute: InstancedAttributeRef | null | undefined,
  instance: number,
): [number, number, number] {
  if (!scaleAttribute) return [1, 1, 1]
  const sourceIndex = instancedAttributeIndex(scaleAttribute.attribute, instance, scaleAttribute.label)
  const itemSize = scaleAttribute.attribute.itemSize ?? 1
  const sx = attributeComponent(scaleAttribute.attribute, sourceIndex, 0, scaleAttribute.label)
  const sy = itemSize >= 2
    ? attributeComponent(scaleAttribute.attribute, sourceIndex, 1, scaleAttribute.label)
    : sx
  const sz = itemSize >= 3
    ? attributeComponent(scaleAttribute.attribute, sourceIndex, 2, scaleAttribute.label)
    : sx
  return [sx, sy, sz]
}

export function expandVec2ValuesForInstances(values: number[], start: number, count: number, instanceCount: number): number[] {
  if (instanceCount <= 1) return values.slice(start * 2, (start + count) * 2)
  const out = new Array<number>(count * instanceCount * 2)
  let dst = 0
  for (let instance = 0; instance < instanceCount; instance += 1) {
    for (let vertex = start; vertex < start + count; vertex += 1) {
      out[dst++] = values[vertex * 2]
      out[dst++] = values[vertex * 2 + 1]
    }
  }
  return out
}

export function expandNormalValuesForInstances(
  attribute: ThreeBufferAttributeLike,
  values: number[],
  start: number,
  count: number,
  instanceCount: number,
  label = 'geometry.attributes.normal',
): number[] {
  if (!isInstancedAttribute(attribute)) {
    return expandVec3ValuesForInstances(values, start, count, instanceCount)
  }

  const out = new Array<number>(count * instanceCount * 3)
  let dst = 0
  for (let instance = 0; instance < instanceCount; instance += 1) {
    const sourceIndex = instancedAttributeIndex(attribute, instance, label)
    const nx = attributeComponent(attribute, sourceIndex, 0, label)
    const ny = attributeComponent(attribute, sourceIndex, 1, label)
    const nz = attributeComponent(attribute, sourceIndex, 2, label)
    for (let vertex = 0; vertex < count; vertex += 1) {
      out[dst++] = nx
      out[dst++] = ny
      out[dst++] = nz
    }
  }
  return out
}

export function expandNormalValuesForInstancesWithCache(
  cache: SceneExtractionCache | undefined,
  geometry: ThreeBufferGeometryLike,
  attribute: ThreeBufferAttributeLike,
  values: number[],
  start: number,
  count: number,
  instanceCount: number,
  label = 'geometry.attributes.normal',
): number[] {
  if (!cache) {
    return expandNormalValuesForInstances(attribute, values, start, count, instanceCount, label)
  }

  const signature = instancedNormalExpansionSignature(
    geometry,
    attribute,
    values,
    start,
    count,
    instanceCount,
    label,
  )
  if (!signature.cacheable) {
    return expandNormalValuesForInstances(attribute, values, start, count, instanceCount, label)
  }

  const key = `${label}:${start}:${count}:${instanceCount}`
  let geometryCache = cache.instancedNormalExpansions.get(geometry)
  const cached = geometryCache?.get(key)
  if (cached && sameInstancedNormalExpansionSignature(cached.signature, signature)) {
    return cached.normals
  }

  const normals = expandNormalValuesForInstances(attribute, values, start, count, instanceCount, label)
  if (!geometryCache) {
    geometryCache = new Map()
    cache.instancedNormalExpansions.set(geometry, geometryCache)
  }
  geometryCache.set(key, { signature, normals })
  return normals
}

export function instancedNormalExpansionSignature(
  geometry: ThreeBufferGeometryLike,
  attribute: ThreeBufferAttributeLike,
  values: number[],
  start: number,
  count: number,
  instanceCount: number,
  label: string,
): InstancedNormalExpansionSignature {
  const signature: InstancedNormalExpansionSignature = {
    cacheable: true,
    geometryVersion: geometry.version,
    sourceNormals: values,
    start,
    count,
    instanceCount,
    normal: attributeSignature(attribute),
    label,
  }
  signature.cacheable = attributeSignatureCacheable(signature.normal)
  return signature
}

export function sameInstancedNormalExpansionSignature(
  a: InstancedNormalExpansionSignature,
  b: InstancedNormalExpansionSignature,
): boolean {
  return a.cacheable === b.cacheable
    && a.geometryVersion === b.geometryVersion
    && a.sourceNormals === b.sourceNormals
    && a.start === b.start
    && a.count === b.count
    && a.instanceCount === b.instanceCount
    && sameAttributeSignature(a.normal, b.normal)
    && a.label === b.label
}

export function expandUvChannelForInstances(channel: UvChannel, start: number, count: number, instanceCount: number): number[] {
  if (!isInstancedAttribute(channel.attribute)) {
    return expandVec2ValuesForInstances(channel.values, start, count, instanceCount)
  }

  const out = new Array<number>(count * instanceCount * 2)
  let dst = 0
  for (let instance = 0; instance < instanceCount; instance += 1) {
    const sourceIndex = instancedAttributeIndex(channel.attribute, instance, labelForMeshPerAttribute(channel))
    const u = attributeComponent(channel.attribute, sourceIndex, 0, channel.label)
    const v = attributeComponent(channel.attribute, sourceIndex, 1, channel.label)
    for (let vertex = 0; vertex < count; vertex += 1) {
      out[dst++] = u
      out[dst++] = v
    }
  }
  return out
}

export function expandUvChannelForInstancesWithCache(
  cache: SceneExtractionCache | undefined,
  geometry: ThreeBufferGeometryLike,
  channel: UvChannel,
  start: number,
  count: number,
  instanceCount: number,
): number[] {
  if (!cache) {
    return expandUvChannelForInstances(channel, start, count, instanceCount)
  }

  const signature = instancedUvExpansionSignature(geometry, channel, start, count, instanceCount)
  if (!signature.cacheable) {
    return expandUvChannelForInstances(channel, start, count, instanceCount)
  }

  const key = `${channel.label}:${start}:${count}:${instanceCount}`
  let geometryCache = cache.instancedUvExpansions.get(geometry)
  const cached = geometryCache?.get(key)
  if (cached && sameInstancedUvExpansionSignature(cached.signature, signature)) {
    return cached.uvs
  }

  const uvs = expandUvChannelForInstances(channel, start, count, instanceCount)
  if (!geometryCache) {
    geometryCache = new Map()
    cache.instancedUvExpansions.set(geometry, geometryCache)
  }
  geometryCache.set(key, { signature, uvs })
  return uvs
}

export function instancedUvExpansionSignature(
  geometry: ThreeBufferGeometryLike,
  channel: UvChannel,
  start: number,
  count: number,
  instanceCount: number,
): InstancedUvExpansionSignature {
  const signature: InstancedUvExpansionSignature = {
    cacheable: true,
    geometryVersion: geometry.version,
    channel: uvChannelSignature(channel),
    start,
    count,
    instanceCount,
  }
  signature.cacheable = attributeSignatureCacheable(signature.channel.attribute)
  return signature
}

export function sameInstancedUvExpansionSignature(
  a: InstancedUvExpansionSignature,
  b: InstancedUvExpansionSignature,
): boolean {
  return a.cacheable === b.cacheable
    && a.geometryVersion === b.geometryVersion
    && sameUvChannelSignature(a.channel, b.channel)
    && a.start === b.start
    && a.count === b.count
    && a.instanceCount === b.instanceCount
}

export function appendUvForVertex(
  out: number[],
  channel: UvChannel,
  vertexIndex: number,
  instanceIndex: number,
): void {
  const sourceIndex = isInstancedAttribute(channel.attribute)
    ? instancedAttributeIndex(channel.attribute, instanceIndex, labelForMeshPerAttribute(channel))
    : vertexIndex
  out.push(
    attributeComponent(channel.attribute, sourceIndex, 0, channel.label),
    attributeComponent(channel.attribute, sourceIndex, 1, channel.label),
  )
}
