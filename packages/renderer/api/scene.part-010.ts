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
import { IndexRangeExpansionSignature, InstancedColorExpansionSignature, MaterialUvSlot, PbrUvFlag, SceneExtractionCache, TextureUvStreams, UvChannel } from './scene.part-001'
import { attributeSignature, attributeSignatureCacheable, sameAttributeSignature } from './scene.part-003'
import { sameNumberArray } from './scene.part-005'
import { instancedAttributeIndex, isInstancedAttribute } from './scene.part-009'
export function uvValuesForInstance(channel: UvChannel | null, vertexCount: number, instanceIndex: number): number[] | null {
  if (!channel) return null
  if (!isInstancedAttribute(channel.attribute)) return channel.values

  const out = new Array<number>(vertexCount * 2)
  let dst = 0
  const sourceIndex = instancedAttributeIndex(channel.attribute, instanceIndex, labelForMeshPerAttribute(channel))
  const u = attributeComponent(channel.attribute, sourceIndex, 0, channel.label)
  const v = attributeComponent(channel.attribute, sourceIndex, 1, channel.label)
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    out[dst++] = u
    out[dst++] = v
  }
  return out
}

export function labelForMeshPerAttribute(channel: UvChannel): string {
  return `${channel.label}.meshPerAttribute`
}

export function expandColorAttributeForInstances(
  attribute: ThreeBufferAttributeLike,
  materialColor: Color4,
  start: number,
  count: number,
  instanceCount: number,
  label = 'geometry.attributes.color',
): number[] {
  if (!isInstancedAttribute(attribute)) {
    const colors = readColorAttribute(attribute, materialColor, label)
    if (instanceCount <= 1) return colors.slice(start * 4, (start + count) * 4)
    const out = new Array<number>(count * instanceCount * 4)
    let dst = 0
    for (let instance = 0; instance < instanceCount; instance += 1) {
      for (let vertex = start; vertex < start + count; vertex += 1) {
        out[dst++] = colors[vertex * 4]
        out[dst++] = colors[vertex * 4 + 1]
        out[dst++] = colors[vertex * 4 + 2]
        out[dst++] = colors[vertex * 4 + 3]
      }
    }
    return out
  }

  const itemSize = attribute.itemSize ?? 3
  const out = new Array<number>(count * instanceCount * 4)
  let dst = 0
  for (let instance = 0; instance < instanceCount; instance += 1) {
    const sourceIndex = instancedAttributeIndex(attribute, instance, label)
    const r = clamp01(attributeComponent(attribute, sourceIndex, 0, label) * materialColor[0])
    const g = clamp01(attributeComponent(attribute, sourceIndex, 1, label) * materialColor[1])
    const b = clamp01(attributeComponent(attribute, sourceIndex, 2, label) * materialColor[2])
    const a = clamp01((itemSize >= 4 ? attributeComponent(attribute, sourceIndex, 3, label) : 1) * materialColor[3])
    for (let vertex = 0; vertex < count; vertex += 1) {
      out[dst++] = r
      out[dst++] = g
      out[dst++] = b
      out[dst++] = a
    }
  }
  return out
}

export function expandColorAttributeForInstancesWithCache(
  cache: SceneExtractionCache | undefined,
  geometry: ThreeBufferGeometryLike,
  attribute: ThreeBufferAttributeLike,
  materialColor: Color4,
  start: number,
  count: number,
  instanceCount: number,
  label = 'geometry.attributes.color',
): number[] {
  if (!cache) {
    return expandColorAttributeForInstances(attribute, materialColor, start, count, instanceCount, label)
  }

  const signature = instancedColorExpansionSignature(
    geometry,
    attribute,
    materialColor,
    start,
    count,
    instanceCount,
    label,
  )
  if (!signature.cacheable) {
    return expandColorAttributeForInstances(attribute, materialColor, start, count, instanceCount, label)
  }

  const key = `${label}:${start}:${count}:${instanceCount}`
  let geometryCache = cache.instancedColorExpansions.get(geometry)
  const cached = geometryCache?.get(key)
  if (cached && sameInstancedColorExpansionSignature(cached.signature, signature)) {
    return cached.colors
  }

  const colors = expandColorAttributeForInstances(attribute, materialColor, start, count, instanceCount, label)
  if (!geometryCache) {
    geometryCache = new Map()
    cache.instancedColorExpansions.set(geometry, geometryCache)
  }
  geometryCache.set(key, { signature, colors })
  return colors
}

export function instancedColorExpansionSignature(
  geometry: ThreeBufferGeometryLike,
  attribute: ThreeBufferAttributeLike,
  materialColor: Color4,
  start: number,
  count: number,
  instanceCount: number,
  label: string,
): InstancedColorExpansionSignature {
  const signature: InstancedColorExpansionSignature = {
    cacheable: true,
    geometryVersion: geometry.version,
    materialColor: materialColor.slice() as Color4,
    start,
    count,
    instanceCount,
    color: attributeSignature(attribute),
    label,
  }
  signature.cacheable = attributeSignatureCacheable(signature.color)
  return signature
}

export function sameInstancedColorExpansionSignature(
  a: InstancedColorExpansionSignature,
  b: InstancedColorExpansionSignature,
): boolean {
  return a.cacheable === b.cacheable
    && a.geometryVersion === b.geometryVersion
    && sameNumberArray(a.materialColor, b.materialColor)
    && a.start === b.start
    && a.count === b.count
    && a.instanceCount === b.instanceCount
    && sameAttributeSignature(a.color, b.color)
    && a.label === b.label
}

export function indexRangeWithCache(
  cache: SceneExtractionCache | undefined,
  geometry: ThreeBufferGeometryLike,
  indices: number[],
  start: number,
  count: number,
): number[] {
  if (!cache) return indices.slice(start, start + count)

  const signature = indexRangeExpansionSignature(geometry, indices, start, count)
  if (!signature.cacheable) return indices.slice(start, start + count)

  const key = `${start}:${count}`
  let geometryCache = cache.indexRanges.get(geometry)
  const cached = geometryCache?.get(key)
  if (cached && sameIndexRangeExpansionSignature(cached.signature, signature)) {
    return cached.indices
  }

  const range = indices.slice(start, start + count)
  if (!geometryCache) {
    geometryCache = new Map()
    cache.indexRanges.set(geometry, geometryCache)
  }
  geometryCache.set(key, { signature, indices: range })
  return range
}

export function indexRangeExpansionSignature(
  geometry: ThreeBufferGeometryLike,
  indices: number[],
  start: number,
  count: number,
): IndexRangeExpansionSignature {
  const signature: IndexRangeExpansionSignature = {
    cacheable: true,
    geometryVersion: geometry.version,
    sourceIndices: indices,
    start,
    count,
    index: attributeSignature(geometry.index),
  }
  signature.cacheable = attributeSignatureCacheable(signature.index)
  return signature
}

export function sameIndexRangeExpansionSignature(
  a: IndexRangeExpansionSignature,
  b: IndexRangeExpansionSignature,
): boolean {
  return a.cacheable === b.cacheable
    && a.geometryVersion === b.geometryVersion
    && a.sourceIndices === b.sourceIndices
    && a.start === b.start
    && a.count === b.count
    && sameAttributeSignature(a.index, b.index)
}

export function expandIndicesForInstances(indices: number[], vertexCount: number, instanceCount: number): number[] {
  if (instanceCount <= 1) return indices
  const out = new Array<number>(indices.length * instanceCount)
  let dst = 0
  for (let instance = 0; instance < instanceCount; instance += 1) {
    const offset = instance * vertexCount
    for (const index of indices) {
      out[dst++] = index + offset
    }
  }
  return out
}

export function wireframeIndicesForTriangles(indices: number[]): number[] {
  const out = new Array<number>(indices.length * 2)
  let dst = 0
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i]
    const b = indices[i + 1]
    const c = indices[i + 2]
    out[dst++] = a
    out[dst++] = b
    out[dst++] = b
    out[dst++] = c
    out[dst++] = c
    out[dst++] = a
  }
  return out
}

export function wireframeIndicesForUnindexedTriangles(vertexCount: number): number[] {
  const out = new Array<number>(vertexCount * 2)
  let dst = 0
  for (let i = 0; i < vertexCount; i += 3) {
    out[dst++] = i
    out[dst++] = i + 1
    out[dst++] = i + 1
    out[dst++] = i + 2
    out[dst++] = i + 2
    out[dst++] = i
  }
  return out
}

export function isDepthDistanceWireframeMaterial(material: ThreeMaterialLike | undefined): boolean {
  return material?.wireframe === true
    && (material.isMeshDepthMaterial === true || material.isMeshDistanceMaterial === true)
}

export function isMeshWireframeMaterial(material: ThreeMaterialLike | undefined): boolean {
  return material?.wireframe === true
}

export function readUvChannels(geometry: ThreeBufferGeometryLike): Array<UvChannel | null> {
  const primaryUvs = readOptionalUvAttribute(geometry, 'uv')
  return [
    primaryUvs,
    readOptionalUvAttribute(geometry, 'uv1') ?? readOptionalUvAttribute(geometry, 'uv2') ?? primaryUvs,
    readOptionalUvAttribute(geometry, 'uv2') ?? readOptionalUvAttribute(geometry, 'uv1') ?? primaryUvs,
    readOptionalUvAttribute(geometry, 'uv3') ?? primaryUvs,
  ]
}

export function readOptionalUvAttribute(geometry: ThreeBufferGeometryLike, name: string): UvChannel | null {
  const attribute = getAttribute(geometry, name)
  if (!attribute) return null
  const label = `geometry.attributes.${name}`
  return {
    attribute,
    label,
    values: readVec2Attribute(attribute, label),
  }
}

export function textureUvStreamsForMapAlphaMaterial(
  channels: Array<UvChannel | null>,
  material: {
    map?: { channel?: number } | null
    alphaMap?: { channel?: number } | null
  } | undefined,
): TextureUvStreams {
  const mapChannel = material?.map ? textureUvChannel(material.map) : undefined
  const alphaChannel = material?.alphaMap ? textureUvChannel(material.alphaMap) : undefined
  const requestedChannels = [mapChannel, alphaChannel]
    .filter((channel): channel is number => channel !== undefined)
  const distinctChannels = [...new Set(requestedChannels)]

  let primaryChannel = 0
  let secondaryChannel: number | undefined
  if (mapChannel !== undefined
    && alphaChannel !== undefined
    && mapChannel !== alphaChannel
    && mapChannel > 0
    && alphaChannel > 0) {
    primaryChannel = mapChannel
    secondaryChannel = alphaChannel
  } else {
    secondaryChannel = distinctChannels.find((channel) => channel > 0)
  }

  return {
    uvs: channels[primaryChannel] ?? channels[0],
    uvs2: secondaryChannel !== undefined
      ? channels[secondaryChannel] ?? channels[0]
      : null,
    textureUsesUv2: mapChannel !== undefined ? mapChannel !== primaryChannel : undefined,
    alphaMapUsesUv2: alphaChannel !== undefined ? alphaChannel !== primaryChannel : undefined,
  }
}

export function textureUvStreamsForMeshMaterial(
  channels: Array<UvChannel | null>,
  material: ThreeMaterialLike | undefined,
): TextureUvStreams {
  return textureUvStreamsForMaterialSlots(channels, meshTextureUvSlots(material))
}

export function meshTextureUvSlots(material: ThreeMaterialLike | undefined): MaterialUvSlot[] {
  if (!material) return []

  const slots: MaterialUvSlot[] = material.isMeshMatcapMaterial === true
    ? [{ texture: material.map, pbrFlag: 'matcapMapUsesUv2' }]
    : [{ texture: material.map, textureFlag: 'textureUsesUv2' }]

  slots.push(
    { texture: material.clearcoatMap, pbrFlag: 'clearcoatMapUsesUv2' },
    { texture: material.clearcoatRoughnessMap, pbrFlag: 'clearcoatRoughnessMapUsesUv2' },
    { texture: material.clearcoatNormalMap, pbrFlag: 'clearcoatNormalMapUsesUv2' },
    { texture: material.sheenColorMap, pbrFlag: 'sheenColorMapUsesUv2' },
    { texture: material.sheenRoughnessMap, pbrFlag: 'sheenRoughnessMapUsesUv2' },
    { texture: material.anisotropyMap, pbrFlag: 'anisotropyMapUsesUv2' },
    { texture: material.iridescenceMap, pbrFlag: 'iridescenceMapUsesUv2' },
    { texture: material.iridescenceThicknessMap, pbrFlag: 'iridescenceThicknessMapUsesUv2' },
    { texture: material.normalMap, pbrFlag: 'normalMapUsesUv2' },
    { texture: material.bumpMap, pbrFlag: 'bumpMapUsesUv2' },
    { texture: material.transmissionMap, pbrFlag: 'transmissionMapUsesUv2' },
    { texture: material.thicknessMap, pbrFlag: 'thicknessMapUsesUv2' },
    { texture: material.specularColorMap, pbrFlag: 'specularColorMapUsesUv2' },
    { texture: material.specularIntensityMap, pbrFlag: 'specularIntensityMapUsesUv2' },
    { texture: material.displacementMap, pbrFlag: 'displacementMapUsesUv2' },
    { texture: material.metalnessMap ?? material.roughnessMap, pbrFlag: 'metallicRoughnessTextureUsesUv2' },
    { texture: material.emissiveMap, pbrFlag: 'emissiveMapUsesUv2' },
    { texture: material.lightMap, pbrFlag: 'lightMapUsesUv2' },
    { texture: material.aoMap, pbrFlag: 'aoMapUsesUv2' },
    { texture: material.specularMap, pbrFlag: 'specularMapUsesUv2' },
    { texture: material.alphaMap, textureFlag: 'alphaMapUsesUv2' },
  )

  return slots
}

export function textureUvStreamsForMaterialSlots(
  channels: Array<UvChannel | null>,
  slots: MaterialUvSlot[],
): TextureUvStreams {
  const activeSlots = slots.filter((slot) => slot.texture != null)
  const requestedChannels: number[] = []
  for (const slot of activeSlots) {
    const channel = textureUvChannel(slot.texture)
    if (!requestedChannels.includes(channel)) requestedChannels.push(channel)
  }

  if (requestedChannels.length > 2) {
    const channelList = [...requestedChannels].sort((a, b) => a - b).join(', ')
    throw new Error(
      `Material uses texture.channel values ${channelList}, but @headless-three/renderer can bind only two UV attributes per draw. Use at most two texture channels or render separate passes.`,
    )
  }

  const preferredPrimary = preferredPrimaryTextureChannel(activeSlots)
  const primaryChannel = requestedChannels.includes(0)
    ? 0
    : preferredPrimary !== undefined && requestedChannels.includes(preferredPrimary)
      ? preferredPrimary
      : requestedChannels[0] ?? 0
  const secondaryChannel = requestedChannels.find((channel) => channel !== primaryChannel)
  const out: TextureUvStreams = {
    uvs: channels[primaryChannel] ?? channels[0],
    uvs2: secondaryChannel !== undefined
      ? channels[secondaryChannel] ?? channels[0]
      : null,
  }

  for (const slot of activeSlots) {
    const usesUv2 = secondaryChannel !== undefined
      && textureUvChannel(slot.texture) === secondaryChannel
    if (slot.textureFlag === 'textureUsesUv2') {
      out.textureUsesUv2 = usesUv2
    } else if (slot.textureFlag === 'alphaMapUsesUv2') {
      out.alphaMapUsesUv2 = usesUv2
    } else if (slot.pbrFlag) {
      out.pbrUsesUv2 ??= {}
      out.pbrUsesUv2[slot.pbrFlag] = usesUv2
    }
  }

  return out
}

export function preferredPrimaryTextureChannel(slots: MaterialUvSlot[]): number | undefined {
  for (const flag of ['textureUsesUv2', 'alphaMapUsesUv2'] as const) {
    const slot = slots.find((candidate) => candidate.textureFlag === flag)
    if (slot?.texture != null) return textureUvChannel(slot.texture)
  }
  return slots[0]?.texture ? textureUvChannel(slots[0].texture) : undefined
}

export function applyPbrUvStreamFlags(props: PbrProperties, uvStreams: TextureUvStreams): void {
  if (!uvStreams.pbrUsesUv2) return
  for (const [flag, usesUv2] of Object.entries(uvStreams.pbrUsesUv2)) {
    props[flag as PbrUvFlag] = usesUv2
  }
}
