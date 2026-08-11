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
import { BatchedMeshDraw, DashedLineExpansion, InstancedAttributeRef, InstancedDashedLineSignature, UvChannel } from './scene.part-001'
import { attributeSignature, attributeSignatureCacheable, sameAttributeSignature } from './scene.part-003'
import { sameOptionalNumberArray, sameUvChannelSignature, uvChannelSignature } from './scene.part-005'
import { finitePositive } from './scene.part-007'
import { transformPoint, transformedSphereOutsideFrustum, viewSpaceZ } from './scene.part-008'
import { instanceScaleComponents, instancedAttributeIndex, isInstancedAttribute } from './scene.part-009'
import { uvValuesForInstance } from './scene.part-010'
import { dashedLineAttributes } from './scene.part-011'
import { appendNumberArray, batchedGeometryRangeBoundingSphere, batchedInstanceColor, batchedInstanceMatrix, batchedNonNegativeInteger, batchedOptionalBoolean, sphereLike } from './scene.part-013'
import { multiplyMat4 } from './scene.part-014'
export function instancedDashedLineSignature(
  geometry: ThreeBufferGeometryLike,
  position: ThreeBufferAttributeLike,
  uvChannel: UvChannel | null,
  uvChannel2: UvChannel | null,
  vertexColors: ThreeBufferAttributeLike | undefined,
  materialColor: Color4,
  lineDistance: ThreeBufferAttributeLike | undefined,
  start: number,
  end: number,
  sourceLength: number,
  object: ThreeObject3DLike,
  dashSize: number,
  gapSize: number,
  scale: number,
  instanceCount: number,
  offsetAttribute: InstancedAttributeRef | null,
  scaleAttribute: InstancedAttributeRef | null,
): InstancedDashedLineSignature {
  const signature: InstancedDashedLineSignature = {
    cacheable: true,
    geometryVersion: geometry.version,
    position: attributeSignature(position),
    index: attributeSignature(geometry.index),
    uv: uvChannelSignature(uvChannel),
    uv2: uvChannelSignature(uvChannel2),
    lineDistance: attributeSignature(lineDistance),
    vertexColors: attributeSignature(vertexColors),
    instancedPositionOffset: attributeSignature(offsetAttribute?.attribute),
    instancedPositionScale: attributeSignature(scaleAttribute?.attribute),
    materialColor: vertexColors ? materialColor.slice() as Color4 : undefined,
    start,
    end,
    sourceLength,
    instanceCount,
    isLineSegments: object.isLineSegments,
    isLineLoop: object.isLineLoop,
    isLine: object.isLine,
    dashSize,
    gapSize,
    scale,
  }
  signature.cacheable = instancedDashedLineSignatureCacheable(signature)
  return signature
}

export function instancedDashedLineSignatureCacheable(signature: InstancedDashedLineSignature): boolean {
  return attributeSignatureCacheable(signature.position)
    && attributeSignatureCacheable(signature.index)
    && attributeSignatureCacheable(signature.uv.attribute)
    && attributeSignatureCacheable(signature.uv2.attribute)
    && attributeSignatureCacheable(signature.lineDistance)
    && attributeSignatureCacheable(signature.vertexColors)
    && attributeSignatureCacheable(signature.instancedPositionOffset)
    && attributeSignatureCacheable(signature.instancedPositionScale)
}

export function instancedDashedLineCacheKey(signature: InstancedDashedLineSignature): string {
  return [
    'instanced',
    signature.start,
    signature.end,
    signature.sourceLength,
    signature.instanceCount,
    signature.isLineSegments ? 1 : 0,
    signature.isLineLoop ? 1 : 0,
    signature.isLine ? 1 : 0,
    signature.uv.attribute.ref ? 1 : 0,
    signature.uv2.attribute.ref ? 1 : 0,
    signature.vertexColors.ref ? 1 : 0,
    signature.instancedPositionOffset.ref ? 1 : 0,
    signature.instancedPositionScale.ref ? 1 : 0,
    signature.dashSize,
    signature.gapSize,
    signature.scale,
  ].join(':')
}

export function sameInstancedDashedLineSignature(
  a: InstancedDashedLineSignature,
  b: InstancedDashedLineSignature,
): boolean {
  return a.cacheable === b.cacheable
    && a.geometryVersion === b.geometryVersion
    && sameAttributeSignature(a.position, b.position)
    && sameAttributeSignature(a.index, b.index)
    && sameUvChannelSignature(a.uv, b.uv)
    && sameUvChannelSignature(a.uv2, b.uv2)
    && sameAttributeSignature(a.lineDistance, b.lineDistance)
    && sameAttributeSignature(a.vertexColors, b.vertexColors)
    && sameAttributeSignature(a.instancedPositionOffset, b.instancedPositionOffset)
    && sameAttributeSignature(a.instancedPositionScale, b.instancedPositionScale)
    && sameOptionalNumberArray(a.materialColor, b.materialColor)
    && a.start === b.start
    && a.end === b.end
    && a.sourceLength === b.sourceLength
    && a.instanceCount === b.instanceCount
    && a.isLineSegments === b.isLineSegments
    && a.isLineLoop === b.isLineLoop
    && a.isLine === b.isLine
    && Object.is(a.dashSize, b.dashSize)
    && Object.is(a.gapSize, b.gapSize)
    && Object.is(a.scale, b.scale)
}

export function dashedLineAttributesForInstances(
  positions: number[],
  uvChannel: UvChannel | null,
  uvChannel2: UvChannel | null,
  vertexColors: ThreeBufferAttributeLike | undefined,
  materialColor: Color4,
  source: number[],
  start: number,
  end: number,
  object: ThreeObject3DLike,
  lineDistance: ThreeBufferAttributeLike | undefined,
  material: { dashSize?: number; gapSize?: number; scale?: number },
  instanceCount: number,
  offsetAttribute: InstancedAttributeRef | null,
  scaleAttribute: InstancedAttributeRef | null,
): DashedLineExpansion {
  const out: DashedLineExpansion = {
    positions: [],
    uvs: uvChannel ? [] : undefined,
    uvs2: uvChannel2 ? [] : undefined,
    colors: vertexColors ? [] : undefined,
  }
  const baseColors = vertexColors && !isInstancedAttribute(vertexColors)
    ? readColorAttribute(vertexColors, materialColor, 'geometry.attributes.color')
    : undefined
  const vertexCount = positions.length / 3

  for (let instance = 0; instance < instanceCount; instance += 1) {
    const instancePositions = offsetAttribute
      ? transformVec3ValuesForInstance(positions, offsetAttribute, scaleAttribute, instance)
      : scaleAttribute
        ? transformVec3ValuesForInstance(positions, null, scaleAttribute, instance)
        : positions
    const instanceUvs = uvValuesForInstance(uvChannel, vertexCount, instance)
    const instanceUvs2 = uvValuesForInstance(uvChannel2, vertexCount, instance)
    const instanceColors = vertexColors
      ? baseColors ?? repeatedInstancedColorValues(vertexColors, materialColor, positions.length / 3, instance)
      : undefined
    const dashed = dashedLineAttributes(
      instancePositions,
      instanceUvs,
      instanceUvs2,
      instanceColors,
      source,
      start,
      end,
      object,
      lineDistance,
      material,
    )
    appendDashedLineExpansion(out, dashed)
  }
  return out
}

export function transformVec3ValuesForInstance(
  values: number[],
  offsetAttribute: InstancedAttributeRef | null,
  scaleAttribute: InstancedAttributeRef | null,
  instance: number,
): number[] {
  const offsetIndex = offsetAttribute
    ? instancedAttributeIndex(offsetAttribute.attribute, instance, offsetAttribute.label)
    : 0
  const ox = offsetAttribute ? attributeComponent(offsetAttribute.attribute, offsetIndex, 0, offsetAttribute.label) : 0
  const oy = offsetAttribute ? attributeComponent(offsetAttribute.attribute, offsetIndex, 1, offsetAttribute.label) : 0
  const oz = offsetAttribute ? attributeComponent(offsetAttribute.attribute, offsetIndex, 2, offsetAttribute.label) : 0
  const scale = instanceScaleComponents(scaleAttribute, instance)
  const out = new Array<number>(values.length)
  for (let i = 0; i < values.length; i += 3) {
    out[i] = values[i] * scale[0] + ox
    out[i + 1] = values[i + 1] * scale[1] + oy
    out[i + 2] = values[i + 2] * scale[2] + oz
  }
  return out
}

export function repeatedInstancedColorValues(
  attribute: ThreeBufferAttributeLike,
  materialColor: Color4,
  vertexCount: number,
  instance: number,
  label = 'geometry.attributes.color',
): number[] {
  const sourceIndex = instancedAttributeIndex(attribute, instance, label)
  const itemSize = attribute.itemSize ?? 3
  const color = [
    clamp01(attributeComponent(attribute, sourceIndex, 0, label) * materialColor[0]),
    clamp01(attributeComponent(attribute, sourceIndex, 1, label) * materialColor[1]),
    clamp01(attributeComponent(attribute, sourceIndex, 2, label) * materialColor[2]),
    clamp01((itemSize >= 4 ? attributeComponent(attribute, sourceIndex, 3, label) : 1) * materialColor[3]),
  ]
  const out = new Array<number>(vertexCount * 4)
  let dst = 0
  for (let i = 0; i < vertexCount; i += 1) {
    out[dst++] = color[0]
    out[dst++] = color[1]
    out[dst++] = color[2]
    out[dst++] = color[3]
  }
  return out
}

export function appendDashedLineExpansion(out: DashedLineExpansion, value: DashedLineExpansion): void {
  appendNumberArray(out.positions, value.positions)
  if (out.uvs && value.uvs) appendNumberArray(out.uvs, value.uvs)
  if (out.uvs2 && value.uvs2) appendNumberArray(out.uvs2, value.uvs2)
  if (out.colors && value.colors) appendNumberArray(out.colors, value.colors)
}

export function batchedMeshDraws(
  object: ThreeObject3DLike,
  camera: ThreeCameraLike | undefined,
  geometry: ThreeBufferGeometryLike,
): BatchedMeshDraw[] {
  const instanceInfo = object._instanceInfo
  if (!Array.isArray(instanceInfo)) {
    throw new Error(
      'THREE.BatchedMesh instance table is not readable. Use a real THREE.BatchedMesh or expand the batch to ordinary Mesh or InstancedMesh objects before rendering.',
    )
  }

  const baseTransform = matrixElements(object.matrixWorld!, 'batchedMesh.matrixWorld')
  const cullPerObject = batchedPerObjectFrustumCulled(object, camera)
  const sortObjects = batchedSortObjects(object)
  const customSort = batchedCustomSort(object)
  const draws: BatchedMeshDraw[] = []
  for (let instanceId = 0; instanceId < instanceInfo.length; instanceId += 1) {
    const info = instanceInfo[instanceId]
    if (!info || typeof info !== 'object') {
      throw new TypeError(`THREE.BatchedMesh._instanceInfo[${instanceId}] must be an object.`)
    }

    if (!batchedOptionalBoolean(info.active, `THREE.BatchedMesh._instanceInfo[${instanceId}].active`, true)) continue
    if (!batchedOptionalBoolean(info.visible, `THREE.BatchedMesh._instanceInfo[${instanceId}].visible`, true)) continue

    const geometryId = batchedNonNegativeInteger(
      info.geometryIndex,
      `THREE.BatchedMesh._instanceInfo[${instanceId}].geometryIndex`,
    )
    const range = batchedGeometryRange(object, geometry, geometryId)
    if (range.count === 0) continue
    const transform = multiplyMat4(baseTransform, batchedInstanceMatrix(object, instanceId))
    if (cullPerObject && batchedDrawOutsideFrustum(object, geometry, geometryId, range, transform, camera!)) {
      continue
    }

    draws.push({
      range,
      instanceId,
      instance: {
        transform,
        color: batchedInstanceColor(object, instanceId),
      },
      z: sortObjects ? batchedDrawDistanceZ(object, geometry, geometryId, range, transform, camera) : 0,
    })
  }

  if (!sortObjects || draws.length < 2) return draws
  if (customSort) {
    return customSortedBatchedMeshDraws(object, draws, customSort, camera)
  }

  return draws
    .slice()
    .sort(batchedMeshUsesTransparentSort(object) ? compareBatchedDrawsTransparent : compareBatchedDrawsOpaque)
}

export function batchedPerObjectFrustumCulled(
  object: ThreeObject3DLike,
  camera: ThreeCameraLike | undefined,
): boolean {
  if (!camera || camera.isArrayCamera === true) return false
  return batchedOptionalBoolean(object.perObjectFrustumCulled, 'THREE.BatchedMesh.perObjectFrustumCulled', true)
}

export function batchedSortObjects(object: ThreeObject3DLike): boolean {
  return batchedOptionalBoolean(object.sortObjects, 'THREE.BatchedMesh.sortObjects', true)
}

export function batchedCustomSort(object: ThreeObject3DLike): ThreeObject3DLike['customSort'] {
  if (object.customSort == null) return null
  if (typeof object.customSort === 'function') return object.customSort
  throw new TypeError('THREE.BatchedMesh.customSort must be a function or null.')
}

export function customSortedBatchedMeshDraws(
  object: ThreeObject3DLike,
  draws: BatchedMeshDraw[],
  customSort: NonNullable<ThreeObject3DLike['customSort']>,
  camera: ThreeCameraLike | undefined,
): BatchedMeshDraw[] {
  const drawByInstance = new Map(draws.map((draw) => [draw.instanceId, draw]))
  const list = draws.map((draw) => ({
    start: draw.range.start,
    count: draw.range.count,
    z: draw.z,
    index: draw.instanceId,
  }))

  customSort.call(object, list, camera)

  if (list.length !== draws.length) {
    throw new Error(`THREE.BatchedMesh.customSort must keep ${draws.length} draw items; received ${list.length}.`)
  }

  const seenInstances = new Set<number>()
  return list.map((item, itemIndex) => {
    if (!item || typeof item !== 'object') {
      throw new TypeError(`THREE.BatchedMesh.customSort list[${itemIndex}] must be an object.`)
    }
    const instanceId = batchedNonNegativeInteger(item.index, `THREE.BatchedMesh.customSort list[${itemIndex}].index`)
    if (seenInstances.has(instanceId)) {
      throw new Error(`THREE.BatchedMesh.customSort returned duplicate instance index ${instanceId}.`)
    }
    seenInstances.add(instanceId)
    const draw = drawByInstance.get(instanceId)
    if (!draw) {
      throw new Error(`THREE.BatchedMesh.customSort returned unknown instance index ${instanceId}.`)
    }
    return draw
  })
}

export function compareBatchedDrawsOpaque(a: BatchedMeshDraw, b: BatchedMeshDraw): number {
  return a.z - b.z || a.instanceId - b.instanceId
}

export function compareBatchedDrawsTransparent(a: BatchedMeshDraw, b: BatchedMeshDraw): number {
  return b.z - a.z || a.instanceId - b.instanceId
}

export function batchedMeshUsesTransparentSort(object: ThreeObject3DLike): boolean {
  const materials = Array.isArray(object.material)
    ? object.material
    : [object.material]
  return materials.some((material) => material?.transparent === true || finitePositive(material?.transmission))
}

export function batchedDrawDistanceZ(
  object: ThreeObject3DLike,
  geometry: ThreeBufferGeometryLike,
  geometryId: number,
  range: { start: number; count: number },
  transform: ArrayLike<number>,
  camera: ThreeCameraLike | undefined,
): number {
  if (!camera) return 0
  const sphere = batchedGeometryBoundingSphere(object, geometry, geometryId, range)
  if (!sphere) return 0
  const center = transformPoint(transform, sphere.center)
  const viewZ = viewSpaceZ(center, camera)
  return Number.isFinite(viewZ) ? -viewZ : 0
}

export function batchedDrawSortIndex(base: number, drawOrder: number): number {
  return Math.max(0, Math.min(0xffffffff, base + drawOrder))
}

export function batchedDrawOutsideFrustum(
  object: ThreeObject3DLike,
  geometry: ThreeBufferGeometryLike,
  geometryId: number,
  range: { start: number; count: number },
  transform: ArrayLike<number>,
  camera: ThreeCameraLike,
): boolean {
  const sphere = batchedGeometryBoundingSphere(object, geometry, geometryId, range)
  if (!sphere) return false
  return transformedSphereOutsideFrustum(camera, transform, sphere)
}

export function batchedGeometryRange(
  object: ThreeObject3DLike,
  geometry: ThreeBufferGeometryLike,
  geometryId: number,
): { start: number; count: number } {
  const cachedRange = object._geometryInfo?.[geometryId]
  const cachedActive = cachedRange && typeof cachedRange === 'object' && !Array.isArray(cachedRange)
    ? (cachedRange as { active?: unknown }).active
    : null
  if (cachedActive != null && !batchedOptionalBoolean(cachedActive, `THREE.BatchedMesh._geometryInfo[${geometryId}].active`, true)) {
    return { start: 0, count: 0 }
  }

  const range = typeof object.getGeometryRangeAt === 'function'
    ? object.getGeometryRangeAt(geometryId, {})
    : cachedRange ?? null

  if (!range || typeof range !== 'object') {
    throw new Error(
      `THREE.BatchedMesh geometry range ${geometryId} is not readable. Use a real THREE.BatchedMesh or expand the batch before rendering.`,
    )
  }

  const rawActive = (range as { active?: unknown }).active
  if (rawActive != null && !batchedOptionalBoolean(rawActive, `THREE.BatchedMesh._geometryInfo[${geometryId}].active`, true)) {
    return { start: 0, count: 0 }
  }

  const startLabel = `THREE.BatchedMesh._geometryInfo[${geometryId}].start`
  const countLabel = `THREE.BatchedMesh._geometryInfo[${geometryId}].count`
  const start = batchedNonNegativeInteger((range as { start?: unknown }).start, startLabel)
  const count = batchedNonNegativeInteger((range as { count?: unknown }).count, countLabel)
  const limit = batchedGeometryRangeLimit(geometry)
  if (start > limit) {
    throw new RangeError(`${startLabel} must be less than or equal to packed geometry count (${limit}).`)
  }
  if (count > limit - start) {
    throw new RangeError(`${countLabel} must fit within packed geometry count (${limit}) from start ${start}.`)
  }
  return { start, count }
}

export function batchedGeometryRangeLimit(geometry: ThreeBufferGeometryLike): number {
  if (geometry.index) return attributeCount(geometry.index, 'geometry.index')
  const position = getAttribute(geometry, 'position')
  return position ? attributeCount(position, 'geometry.attributes.position') : 0
}

export function batchedGeometryBoundingSphere(
  object: ThreeObject3DLike,
  geometry: ThreeBufferGeometryLike,
  geometryId: number,
  range: { start: number; count: number },
): { center: [number, number, number]; radius: number } | null {
  const cached = object._geometryInfo?.[geometryId]?.boundingSphere
  if (cached != null) {
    return batchedSphereLike(cached, `THREE.BatchedMesh._geometryInfo[${geometryId}].boundingSphere`)
  }

  const computed = batchedGeometryRangeBoundingSphere(geometry, range)
  return computed ? batchedSphereLike(computed, `THREE.BatchedMesh._geometryInfo[${geometryId}].computedBoundingSphere`) : null
}

export function batchedSphereLike(
  sphere: ThreeSphereLike,
  label: string,
): { center: [number, number, number]; radius: number } {
  return sphereLike(sphere, label)
}
