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
import { AttributeSignature, CachedMeshGeometryExtraction, ClippingContext, FlattenedMesh, InstancedAttributeRef, MeshGeometryExtraction, MeshGeometrySignature, MeshInstance, SceneExtractionCache, ShadowMaterialMode, UvChannel } from './scene.part-001'
import { shadowOnlyMainPassState } from './scene.part-004'
import { shadowMaterialWithSourceShadowState, shadowPbrProperties } from './scene.part-005'
import { effectiveGroups, optionalSceneBoolean } from './scene.part-006'
import { clipShadowsForMaterial, clippingState, pushMesh, sortInfoForObject } from './scene.part-007'
import { assertSupportedCustomFragmentInstancedAttributes, expandNormalValuesForInstancesWithCache, expandUvChannelForInstancesWithCache, expandVec3ValuesForInstancesWithCache, instancedBufferGeometryCount, instancedOffsetAttribute, instancedScaleAttribute, isInstancedAttribute } from './scene.part-009'
import { applyPbrUvStreamFlags, expandColorAttributeForInstancesWithCache, expandIndicesForInstances, indexRangeWithCache, isDepthDistanceWireframeMaterial, readUvChannels, textureUvStreamsForMeshMaterial, wireframeIndicesForTriangles, wireframeIndicesForUnindexedTriangles } from './scene.part-010'
import { instanceColor, rangeIndices } from './scene.part-014'
export function meshGeometryExtraction(
  geometry: ThreeBufferGeometryLike,
  cache?: SceneExtractionCache,
): MeshGeometryExtraction | null {
  const position = getAttribute(geometry, 'position')
  if (!position) return null

  const signature = meshGeometrySignature(geometry, position)
  if (cache && signature.cacheable) {
    const cached = cache.meshGeometry.get(geometry) as CachedMeshGeometryExtraction | undefined
    if (cached && sameMeshGeometrySignature(cached.signature, signature)) {
      return cached.extraction
    }
  }

  const extraction = readMeshGeometryExtraction(geometry, position)
  if (cache && signature.cacheable) {
    cache.meshGeometry.set(geometry, { signature, extraction })
  }
  return extraction
}

export function readMeshGeometryExtraction(
  geometry: ThreeBufferGeometryLike,
  position: ThreeBufferAttributeLike,
): MeshGeometryExtraction {
  const positions = readVec3Attribute(position, 'geometry.attributes.position')
  const uvChannels = readUvChannels(geometry)
  const normalAttribute = getAttribute(geometry, 'normal')
  const normals = normalAttribute ? readVec3Attribute(normalAttribute, 'geometry.attributes.normal') : null
  const vertexColors = getAttribute(geometry, 'color')
  const index = geometry.index ? readIndexAttribute(geometry.index, 'geometry.index', position.count) : null
  const sourceIndex = index ?? rangeIndices(position.count)
  const groups = effectiveGroups(geometry, index, position.count)
  const instancedGeometryCount = instancedBufferGeometryCount(geometry)
  const instancedPositionOffset = instancedOffsetAttribute(geometry)
  const instancedPositionScale = instancedScaleAttribute(geometry)

  return {
    position,
    positions,
    uvChannels,
    uvs: uvChannels[0]?.values ?? null,
    normalAttribute,
    normals,
    vertexColors,
    index,
    sourceIndex,
    groups,
    instancedGeometryCount,
    instancedPositionOffset,
    instancedPositionScale,
  }
}

export function meshGeometrySignature(
  geometry: ThreeBufferGeometryLike,
  position: ThreeBufferAttributeLike,
): MeshGeometrySignature {
  const attributes = geometryAttributes(geometry)
  const instancedAttributes = Object.entries(attributes)
    .filter((entry): entry is [string, ThreeBufferAttributeLike] => isInstancedAttribute(entry[1]))
    .map(([name, attribute]) => ({ name, signature: attributeSignature(attribute) }))
  const instancedPositionOffset = namedInstancedOffsetAttribute(geometry)
  const instancedPositionScale = namedInstancedScaleAttribute(geometry)
  const signature: MeshGeometrySignature = {
    cacheable: true,
    geometryVersion: geometry.version,
    isInstancedBufferGeometry: geometry.isInstancedBufferGeometry,
    instanceCount: geometry.instanceCount,
    drawRange: geometryDrawRangeSignature(geometry.drawRange),
    drawRangeStart: geometry.drawRange?.start,
    drawRangeCount: geometry.drawRange?.count,
    groups: geometryGroupsSignature(geometry.groups),
    position: attributeSignature(position),
    normal: attributeSignature(getAttribute(geometry, 'normal')),
    color: attributeSignature(getAttribute(geometry, 'color')),
    index: attributeSignature(geometry.index),
    uv: attributeSignature(getAttribute(geometry, 'uv')),
    uv1: attributeSignature(getAttribute(geometry, 'uv1')),
    uv2: attributeSignature(getAttribute(geometry, 'uv2')),
    uv3: attributeSignature(getAttribute(geometry, 'uv3')),
    instancedPositionOffsetName: instancedPositionOffset?.name,
    instancedPositionOffset: attributeSignature(instancedPositionOffset?.attribute),
    instancedPositionScaleName: instancedPositionScale?.name,
    instancedPositionScale: attributeSignature(instancedPositionScale?.attribute),
    instancedAttributes,
  }
  signature.cacheable = meshGeometrySignatureCacheable(signature)
  return signature
}

export function meshGeometrySignatureCacheable(signature: MeshGeometrySignature): boolean {
  return [
    signature.position,
    signature.normal,
    signature.color,
    signature.index,
    signature.uv,
    signature.uv1,
    signature.uv2,
    signature.uv3,
    signature.instancedPositionOffset,
    signature.instancedPositionScale,
    ...signature.instancedAttributes.map(({ signature }) => signature),
  ].every(attributeSignatureCacheable)
}

export function attributeSignature(attribute: ThreeBufferAttributeLike | null | undefined): AttributeSignature {
  if (!attribute) return {}
  const data = attribute.data && typeof attribute.data === 'object' && !Array.isArray(attribute.data)
    ? attribute.data
    : undefined
  return {
    ref: attribute,
    version: attribute.version,
    count: attribute.count,
    itemSize: attribute.itemSize,
    normalized: attribute.normalized,
    array: attribute.array,
    dataArray: data?.array,
    dataStride: data?.stride,
    offset: attribute.offset,
    isInstancedBufferAttribute: attribute.isInstancedBufferAttribute,
    meshPerAttribute: attribute.meshPerAttribute,
  }
}

export function attributeSignatureCacheable(signature: AttributeSignature): boolean {
  return signature.ref == null || typeof signature.version === 'number'
}

export function namedInstancedOffsetAttribute(
  geometry: ThreeBufferGeometryLike,
): { name: string; attribute: ThreeBufferAttributeLike } | null {
  const names = ['instanceOffset', 'instancePosition', 'offset', 'translate', 'translation']
  for (const name of names) {
    const attribute = getAttribute(geometry, name)
    if (isInstancedAttribute(attribute)) return { name, attribute }
  }
  return null
}

export function namedInstancedScaleAttribute(
  geometry: ThreeBufferGeometryLike,
): { name: string; attribute: ThreeBufferAttributeLike } | null {
  const names = ['instanceScale', 'scale']
  for (const name of names) {
    const attribute = getAttribute(geometry, name)
    if (isInstancedAttribute(attribute)) return { name, attribute }
  }
  return null
}

export function geometryGroupsSignature(groups: ThreeBufferGeometryLike['groups']): string {
  if (groups == null) return 'none'
  if (!Array.isArray(groups)) return `invalid:${typeof groups}`
  return groups.map((group, index) => {
    if (!group || typeof group !== 'object' || Array.isArray(group)) {
      return `${index}:invalid:${typeof group}`
    }
    return [
      index,
      typeof group.start,
      String(group.start),
      typeof group.count,
      String(group.count),
      typeof group.materialIndex,
      String(group.materialIndex),
    ].join(':')
  }).join('|')
}

export function geometryDrawRangeSignature(drawRange: ThreeBufferGeometryLike['drawRange']): string {
  if (drawRange == null) return 'none'
  if (typeof drawRange !== 'object' || Array.isArray(drawRange)) return `invalid:${typeof drawRange}`
  return [
    typeof drawRange.start,
    String(drawRange.start),
    typeof drawRange.count,
    String(drawRange.count),
  ].join(':')
}

export function sameMeshGeometrySignature(a: MeshGeometrySignature, b: MeshGeometrySignature): boolean {
  return a.cacheable === b.cacheable
    && a.geometryVersion === b.geometryVersion
    && a.isInstancedBufferGeometry === b.isInstancedBufferGeometry
    && a.instanceCount === b.instanceCount
    && a.drawRange === b.drawRange
    && Object.is(a.drawRangeStart, b.drawRangeStart)
    && Object.is(a.drawRangeCount, b.drawRangeCount)
    && a.groups === b.groups
    && a.instancedPositionOffsetName === b.instancedPositionOffsetName
    && a.instancedPositionScaleName === b.instancedPositionScaleName
    && sameAttributeSignature(a.position, b.position)
    && sameAttributeSignature(a.normal, b.normal)
    && sameAttributeSignature(a.color, b.color)
    && sameAttributeSignature(a.index, b.index)
    && sameAttributeSignature(a.uv, b.uv)
    && sameAttributeSignature(a.uv1, b.uv1)
    && sameAttributeSignature(a.uv2, b.uv2)
    && sameAttributeSignature(a.uv3, b.uv3)
    && sameAttributeSignature(a.instancedPositionOffset, b.instancedPositionOffset)
    && sameAttributeSignature(a.instancedPositionScale, b.instancedPositionScale)
    && sameInstancedAttributeSignatures(a.instancedAttributes, b.instancedAttributes)
}

export function sameInstancedAttributeSignatures(
  a: MeshGeometrySignature['instancedAttributes'],
  b: MeshGeometrySignature['instancedAttributes'],
): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i += 1) {
    if (a[i].name !== b[i].name) return false
    if (!sameAttributeSignature(a[i].signature, b[i].signature)) return false
  }
  return true
}

export function sameAttributeSignature(a: AttributeSignature, b: AttributeSignature): boolean {
  return a.ref === b.ref
    && a.version === b.version
    && a.count === b.count
    && a.itemSize === b.itemSize
    && a.normalized === b.normalized
    && a.array === b.array
    && a.dataArray === b.dataArray
    && a.dataStride === b.dataStride
    && a.offset === b.offset
    && a.isInstancedBufferAttribute === b.isInstancedBufferAttribute
    && a.meshPerAttribute === b.meshPerAttribute
}

export function appendShadowOnlyMeshGroup(
  object: ThreeObject3DLike,
  camera: ThreeCameraLike | undefined,
  meshes: FlattenedMesh[],
  group: GeometryGroup,
  groupOrder: number,
  clippingContext: ClippingContext,
  localClippingEnabled: boolean,
  material: ThreeMaterialLike,
  sourceMaterial: ThreeMaterialLike | undefined,
  materialContext: MaterialExtractionContext,
  positions: number[],
  normals: number[] | null,
  normalAttribute: ThreeBufferAttributeLike | undefined,
  uvs: number[] | null,
  uvChannels: Array<UvChannel | null>,
  vertexColors: ThreeBufferAttributeLike | undefined,
  vertexCount: number,
  index: number[] | null,
  instancedGeometryCount: number,
  instancedPositionOffset: InstancedAttributeRef | null,
  instancedPositionScale: InstancedAttributeRef | null,
  instances: MeshInstance[],
  cache?: SceneExtractionCache,
): void {
  const shadowMaterial = shadowMaterialWithSourceShadowState(material, sourceMaterial)
  const baseColor = materialColor(shadowMaterial, materialContext)
  const useVertexColors = vertexColors && material.vertexColors !== false
  const pbrProps = shadowPbrProperties(shadowMaterial, sourceMaterial, materialContext)
  assertSupportedCustomFragmentInstancedAttributes(object.geometry!, pbrProps)
  const uvStreams = textureUvStreamsForMeshMaterial(uvChannels, shadowMaterial)
  if (uvStreams.alphaMapUsesUv2 !== undefined) {
    pbrProps.alphaMapUsesUv2 = uvStreams.alphaMapUsesUv2
  }
  applyPbrUvStreamFlags(pbrProps, uvStreams)
  const textureInfo = extractTextureData(shadowMaterial, materialContext)
  const clipping = clippingState(clippingContext, shadowMaterial, localClippingEnabled)
  const wireframe = isDepthDistanceWireframeMaterial(shadowMaterial)
  const hiddenMainPass = shadowOnlyMainPassState()

  if (index) {
    const indices = indexRangeWithCache(cache, object.geometry!, index, group.start, group.count)
    if (indices.length % 3 !== 0) {
      throw new Error(`THREE.Mesh "${object.name || object.uuid || '<unnamed>'}" has a non-triangle index range`)
    }
    const renderIndices = wireframe ? wireframeIndicesForTriangles(indices) : indices
    const expandedIndices = expandIndicesForInstances(renderIndices, vertexCount, instancedGeometryCount)
    const geometry = object.geometry!
    const expandedPositions = expandVec3ValuesForInstancesWithCache(
      cache,
      geometry,
      getAttribute(geometry, 'position')!,
      positions,
      0,
      vertexCount,
      instancedGeometryCount,
      instancedPositionOffset,
      instancedPositionScale,
    )
    const expandedNormals = normalAttribute && normals
      ? expandNormalValuesForInstancesWithCache(cache, geometry, normalAttribute, normals, 0, vertexCount, instancedGeometryCount)
      : undefined
    const expandedUvs = uvStreams.uvs
      ? expandUvChannelForInstancesWithCache(cache, geometry, uvStreams.uvs, 0, vertexCount, instancedGeometryCount)
      : undefined
    const expandedSecondaryUvs = uvStreams.uvs2
      ? expandUvChannelForInstancesWithCache(cache, geometry, uvStreams.uvs2, 0, vertexCount, instancedGeometryCount)
      : undefined

    for (const instance of instances) {
      const color = instanceColor(baseColor, instance)
      const sortInfo = sortInfoForObject(object, material, camera, meshes.length, groupOrder, instance.transform, object.geometry, group)
      pushMesh(meshes, {
        positions: expandedPositions,
        indices: expandedIndices,
        normals: expandedNormals,
        color,
        colors: useVertexColors
          ? expandColorAttributeForInstancesWithCache(cache, geometry, vertexColors!, color, 0, vertexCount, instancedGeometryCount)
          : undefined,
        uvs: expandedUvs,
        uvs2: expandedSecondaryUvs,
        texture: textureInfo?.data,
        textureWidth: textureInfo?.width ?? undefined,
        textureHeight: textureInfo?.height ?? undefined,
        textureWrapS: textureInfo?.wrapS,
        textureWrapT: textureInfo?.wrapT,
        textureMagFilter: textureInfo?.magFilter,
        textureMinFilter: textureInfo?.minFilter,
        textureAnisotropy: textureInfo?.anisotropy,
        textureTransform: textureInfo?.transform,
        textureColorSpace: textureInfo?.colorSpace,
        textureUsesUv2: uvStreams.textureUsesUv2 ?? textureInfo?.usesUv2,
        transform: instance.transform,
        topology: wireframe ? 'lines' : undefined,
        castShadow: true,
        receiveShadow: false,
        clipShadows: clipShadowsForMaterial(shadowMaterial, clippingContext),
        ...clipping,
        ...sortInfo.keys,
        ...pbrProps,
        ...hiddenMainPass,
      }, sortInfo.item)
    }
    return
  }

  if (group.count % 3 !== 0) {
    throw new Error(`THREE.Mesh "${object.name || object.uuid || '<unnamed>'}" has a non-triangle vertex range`)
  }

  const geometry = object.geometry!
  const expandedGroupPositions = expandVec3ValuesForInstancesWithCache(
    cache,
    geometry,
    getAttribute(geometry, 'position')!,
    positions,
    group.start,
    group.count,
    instancedGeometryCount,
    instancedPositionOffset,
    instancedPositionScale,
  )
  const expandedGroupNormals = normalAttribute && normals
    ? expandNormalValuesForInstancesWithCache(cache, geometry, normalAttribute, normals, group.start, group.count, instancedGeometryCount)
    : undefined
  const expandedGroupUvs = uvStreams.uvs
    ? expandUvChannelForInstancesWithCache(cache, geometry, uvStreams.uvs, group.start, group.count, instancedGeometryCount)
    : undefined
  const expandedGroupSecondaryUvs = uvStreams.uvs2
    ? expandUvChannelForInstancesWithCache(cache, geometry, uvStreams.uvs2, group.start, group.count, instancedGeometryCount)
    : undefined
  const expandedGroupIndices = wireframe
    ? expandIndicesForInstances(wireframeIndicesForUnindexedTriangles(group.count), group.count, instancedGeometryCount)
    : undefined

  for (const instance of instances) {
    const color = instanceColor(baseColor, instance)
    const sortInfo = sortInfoForObject(object, material, camera, meshes.length, groupOrder, instance.transform, object.geometry, group)
    pushMesh(meshes, {
      positions: expandedGroupPositions,
      indices: expandedGroupIndices,
      normals: expandedGroupNormals,
      color,
      colors: useVertexColors
        ? expandColorAttributeForInstancesWithCache(cache, geometry, vertexColors!, color, group.start, group.count, instancedGeometryCount)
        : undefined,
      uvs: expandedGroupUvs,
      uvs2: expandedGroupSecondaryUvs,
      texture: textureInfo?.data,
      textureWidth: textureInfo?.width ?? undefined,
      textureHeight: textureInfo?.height ?? undefined,
      textureWrapS: textureInfo?.wrapS,
      textureWrapT: textureInfo?.wrapT,
      textureMagFilter: textureInfo?.magFilter,
      textureMinFilter: textureInfo?.minFilter,
      textureAnisotropy: textureInfo?.anisotropy,
      textureTransform: textureInfo?.transform,
      textureColorSpace: textureInfo?.colorSpace,
      textureUsesUv2: uvStreams.textureUsesUv2 ?? textureInfo?.usesUv2,
      transform: instance.transform,
      topology: wireframe ? 'lines' : undefined,
      castShadow: true,
      receiveShadow: false,
      clipShadows: clipShadowsForMaterial(shadowMaterial, clippingContext),
      ...clipping,
      ...sortInfo.keys,
      ...pbrProps,
      ...hiddenMainPass,
    }, sortInfo.item)
  }
}

export function customShadowMaterialForMode(
  object: ThreeObject3DLike,
  mode: ShadowMaterialMode | undefined,
): ThreeMaterialLike | undefined {
  if (mode === 'depth') {
    assertMaterialLike(object.customDepthMaterial, 'Object3D.customDepthMaterial')
    optionalSceneBoolean(object.customDepthMaterial?.visible, 'Object3D.customDepthMaterial.visible')
    return object.customDepthMaterial
  }
  if (mode === 'distance') {
    assertMaterialLike(object.customDistanceMaterial, 'Object3D.customDistanceMaterial')
    optionalSceneBoolean(object.customDistanceMaterial?.visible, 'Object3D.customDistanceMaterial.visible')
    return object.customDistanceMaterial
  }
  return undefined
}

export function materialForObjectGroup(
  object: ThreeObject3DLike,
  materialIndex: number,
  overrideMaterial: ThreeMaterialLike | undefined,
): ThreeMaterialLike | undefined {
  if (overrideMaterial !== undefined) {
    assertMaterialLike(overrideMaterial, 'scene.overrideMaterial')
    return overrideMaterial
  }
  return materialForGroup(object.material, materialIndex)
}
