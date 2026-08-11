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
import { ClippingContext, FlattenedMesh, MeshInstance, SceneExtractionCache, ShadowMaterialMode } from './scene.part-001'
import { appendShadowOnlyMeshGroup, customShadowMaterialForMode, materialForObjectGroup, meshGeometryExtraction } from './scene.part-003'
import { appendPoints, appendSprite, invokeObjectRenderCallback } from './scene.part-004'
import { optionalObjectBoolean } from './scene.part-005'
import { appendLineOrPoints } from './scene.part-006'
import { applyNativeMeshPayloadCache, clipShadowsForMaterial, clippingContextForObject, clippingState, mergeSortKeys, nativeMeshesFromSortedFlattened, needsTransparentDoubleSidePass, projectedObjectZ, pushMesh, renderableObjectOutsideFrustum, sortFlattenedMeshes, sortInfoForObject } from './scene.part-007'
import { renderOrderNumber, unsignedSortKey } from './scene.part-008'
import { assertSupportedCustomFragmentInstancedAttributes, expandNormalValuesForInstancesWithCache, expandUvChannelForInstancesWithCache, expandVec3ValuesForInstancesWithCache } from './scene.part-009'
import { applyPbrUvStreamFlags, expandColorAttributeForInstancesWithCache, expandIndicesForInstances, indexRangeWithCache, isMeshWireframeMaterial, textureUvStreamsForMeshMaterial, wireframeIndicesForTriangles, wireframeIndicesForUnindexedTriangles } from './scene.part-010'
import { updateLodObject } from './scene.part-011'
import { batchedDrawSortIndex, batchedMeshDraws } from './scene.part-012'
import { batchedGeometryView, meshInstances } from './scene.part-013'
import { instanceColor } from './scene.part-014'
export interface SceneSortOptions {
  sortObjects?: boolean
  opaqueSort?: RenderSortFunction | null
  transparentSort?: RenderSortFunction | null
  opaque?: boolean
  transparent?: boolean
}

export interface RenderCallbackContext {
  renderer: unknown
  scene: ThreeObject3DLike
}

export interface MeshSortInfo {
  keys: Pick<NativeSceneMesh, 'groupOrder' | 'renderOrder' | 'sortZ' | 'sortIndex' | 'materialSortKey' | 'materialVariant'>
  item: RenderSortItem
}

export type SortKeyOverride = Partial<MeshSortInfo['keys']>

export const MAX_POINT_SPRITE_SIZE = 64

export function createSceneExtractionCache(): SceneExtractionCache {
  return {
    meshGeometry: new WeakMap(),
    instancedMeshes: new WeakMap(),
    instancedPositionExpansions: new WeakMap(),
    instancedNormalExpansions: new WeakMap(),
    instancedUvExpansions: new WeakMap(),
    instancedColorExpansions: new WeakMap(),
    indexRanges: new WeakMap(),
    batchedGeometryViews: new WeakMap(),
    dashedLines: new WeakMap(),
    instancedDashedLines: new WeakMap(),
    texturePayloads: new WeakMap(),
    materialColors: new WeakMap(),
    textureStates: new WeakMap(),
    materialRenderStates: new WeakMap(),
    materialScalarFeatures: new WeakMap(),
    pointBillboards: new WeakMap(),
    spriteBillboards: new WeakMap(),
    nativeMeshPayloads: {
      objectIds: new WeakMap(),
      payloads: new Map(),
      pending: new Set(),
      nextObjectId: 1,
      nextPayloadId: 1,
    },
  }
}

export function flattenScene(
  scene: ThreeObject3DLike,
  camera?: ThreeCameraLike,
  viewportHeight = 512,
  globalClippingPlanes: readonly NativeClippingPlane[] = [],
  localClippingEnabled = true,
  shadowMaterialMode?: ShadowMaterialMode,
  materialContext: MaterialExtractionContext = {},
  sortOptions: SceneSortOptions = {},
  overrideMaterial?: ThreeMaterialLike,
  cache?: SceneExtractionCache,
  callbackContext?: RenderCallbackContext,
): NativeSceneMesh[] {
  const meshes: FlattenedMesh[] = []
  const clippingContext: ClippingContext = {
    unionPlanes: globalClippingPlanes,
    intersectionPlanes: [],
    clipShadows: false,
  }
  visitObject(scene, camera, meshes, 0, viewportHeight, clippingContext, localClippingEnabled, shadowMaterialMode, materialContext, overrideMaterial, cache, callbackContext)
  const nativeMeshes = nativeMeshesFromSortedFlattened(sortFlattenedMeshes(meshes, sortOptions))
  if (cache) {
    applyNativeMeshPayloadCache(nativeMeshes, cache.nativeMeshPayloads)
  }
  return nativeMeshes
}

export function commitNativeMeshPayloadCache(cache: SceneExtractionCache): void {
  const payloads = cache.nativeMeshPayloads
  for (const signature of payloads.pending) {
    const cached = payloads.payloads.get(signature)
    if (cached) cached.ready = true
  }
  payloads.pending.clear()
}

export function visitObject(
  object: ThreeObject3DLike,
  camera: ThreeCameraLike | undefined,
  meshes: FlattenedMesh[],
  groupOrder: number,
  viewportHeight: number,
  clippingContext: ClippingContext,
  localClippingEnabled: boolean,
  shadowMaterialMode: ShadowMaterialMode | undefined,
  materialContext: MaterialExtractionContext,
  overrideMaterial: ThreeMaterialLike | undefined,
  cache: SceneExtractionCache | undefined,
  callbackContext: RenderCallbackContext | undefined,
): void {
  if (!object) return
  if (optionalObjectBoolean(object.visible, 'object.visible') === false) return

  const nextGroupOrder = object.isGroup === true
    ? renderOrderNumber(object.renderOrder, 'object.renderOrder', 0)
    : groupOrder
  const visibleToCamera = objectLayersMatchCamera(object, camera)
  const nextClippingContext = visibleToCamera
    ? clippingContextForObject(clippingContext, object)
    : clippingContext
  if (visibleToCamera) {
    updateLodObject(object, camera)

    if (object.isBatchedMesh === true && object.geometry) {
      if (!renderableObjectOutsideFrustum(object, camera)) {
        appendBatchedMesh(object, camera, meshes, nextGroupOrder, nextClippingContext, localClippingEnabled, shadowMaterialMode, materialContext, overrideMaterial, cache, callbackContext)
      }
    } else if (object.isMesh === true && object.geometry) {
      if (!renderableObjectOutsideFrustum(object, camera)) {
        appendMesh(object, camera, meshes, nextGroupOrder, nextClippingContext, localClippingEnabled, shadowMaterialMode, materialContext, overrideMaterial, cache, undefined, undefined, undefined, callbackContext)
      }
    } else if ((object.isLineSegments === true || object.isLineLoop === true || object.isLine === true) && object.geometry) {
      if (!renderableObjectOutsideFrustum(object, camera)) {
        appendLineOrPoints(object, camera, meshes, 'lines', nextGroupOrder, viewportHeight, nextClippingContext, localClippingEnabled, materialContext, overrideMaterial, cache, callbackContext)
      }
    } else if (object.isPoints === true && object.geometry) {
      if (!renderableObjectOutsideFrustum(object, camera, viewportHeight, overrideMaterial)) {
        appendPoints(object, camera, meshes, nextGroupOrder, viewportHeight, nextClippingContext, localClippingEnabled, shadowMaterialMode, materialContext, overrideMaterial, cache, callbackContext)
      }
    } else if (object.isSprite === true) {
      appendSprite(object, camera, meshes, nextGroupOrder, nextClippingContext, localClippingEnabled, shadowMaterialMode, materialContext, overrideMaterial, cache, callbackContext)
    }
  }

  for (const child of objectChildren(object)) {
    visitObject(child, camera, meshes, nextGroupOrder, viewportHeight, nextClippingContext, localClippingEnabled, shadowMaterialMode, materialContext, overrideMaterial, cache, callbackContext)
  }
}

export function appendBatchedMesh(
  object: ThreeObject3DLike,
  camera: ThreeCameraLike | undefined,
  meshes: FlattenedMesh[],
  groupOrder: number,
  clippingContext: ClippingContext,
  localClippingEnabled: boolean,
  shadowMaterialMode: ShadowMaterialMode | undefined,
  materialContext: MaterialExtractionContext,
  overrideMaterial: ThreeMaterialLike | undefined,
  cache: SceneExtractionCache | undefined,
  callbackContext: RenderCallbackContext | undefined,
): void {
  const geometry = object.geometry!
  const draws = batchedMeshDraws(object, camera, geometry)
  if (draws.length === 0) return
  const objectSortZ = camera ? projectedObjectZ(object, camera) : 0
  const sortIndexBase = unsignedSortKey(object.id, meshes.length)

  for (let drawOrder = 0; drawOrder < draws.length; drawOrder += 1) {
    const draw = draws[drawOrder]
    const geometryView = batchedGeometryView(geometry, draw.range, cache)
    const objectView = Object.create(object) as ThreeObject3DLike
    objectView.geometry = geometryView
    objectView.isInstancedMesh = false
    objectView.instanceMatrix = undefined
    objectView.instanceColor = undefined
    objectView.count = undefined
    const sortKeyOverride: SortKeyOverride = {
      sortZ: objectSortZ,
      sortIndex: batchedDrawSortIndex(sortIndexBase, drawOrder),
    }
    appendMesh(
      objectView,
      camera,
      meshes,
      groupOrder,
      clippingContext,
      localClippingEnabled,
      shadowMaterialMode,
      materialContext,
      overrideMaterial,
      cache,
      [draw.instance],
      sortKeyOverride,
      object,
      callbackContext,
      draw.z,
    )
  }
}

export function appendMesh(
  object: ThreeObject3DLike,
  camera: ThreeCameraLike | undefined,
  meshes: FlattenedMesh[],
  groupOrder: number,
  clippingContext: ClippingContext,
  localClippingEnabled: boolean,
  shadowMaterialMode: ShadowMaterialMode | undefined,
  materialContext: MaterialExtractionContext,
  overrideMaterial?: ThreeMaterialLike,
  cache?: SceneExtractionCache,
  instanceOverride?: MeshInstance[],
  sortKeyOverride?: SortKeyOverride,
  sortItemObject?: ThreeObject3DLike,
  callbackContext?: RenderCallbackContext,
  sortItemZOverride?: number,
): void {
  const geometry = object.geometry!
  const geometryExtraction = meshGeometryExtraction(geometry, cache)
  if (!geometryExtraction) return

  const {
    position,
    uvChannels,
    uvs,
    normalAttribute,
    vertexColors,
    index,
    groups,
    instancedGeometryCount,
    instancedPositionOffset,
    instancedPositionScale,
  } = geometryExtraction
  let positions = geometryExtraction.positions
  let normals = geometryExtraction.normals

  // CPU-side morph targets (blend shapes / shape keys / VRM blendshapes)
  if (object.morphTargetInfluences && object.morphTargetInfluences.length > 0) {
    const morphed = applyMorphTargets(object, positions, normals)
    positions = morphed.positions
    normals = morphed.normals
  }

  // CPU-side skinning for SkinnedMesh (Three.js, VRM, VRMA)
  if (object.isSkinnedMesh === true && object.skeleton) {
    const skinned = applyCpuSkinning(object, positions, normals)
    positions = skinned.positions
    normals = skinned.normals
  }

  // For skinned meshes, positions are already in world space after CPU skinning.
  const isSkinned = object.isSkinnedMesh === true && object.skeleton
  const meshTransform = instanceOverride
    ? IDENTITY_4X4.slice()
    : isSkinned
      ? IDENTITY_4X4.slice()
      : matrixElements(object.matrixWorld!, 'mesh.matrixWorld')
  const instances = instanceOverride ?? meshInstances(object, meshTransform, cache)
  if (instances.length === 0) return
  const objectCastsShadow = optionalObjectBoolean(object.castShadow, 'object.castShadow') === true
  const objectReceivesShadow = optionalObjectBoolean(object.receiveShadow, 'object.receiveShadow') === true
  const callbackObject = sortItemObject ?? object

  for (const group of groups) {
    const material = materialForObjectGroup(object, group.materialIndex, overrideMaterial)
    if (material?.visible === false) continue

    invokeObjectRenderCallback(callbackObject.onBeforeRender, 'onBeforeRender', callbackContext, callbackObject, camera, geometry, material, group)

    const customShadowMaterial = customShadowMaterialForMode(object, shadowMaterialMode)
    const usesCustomShadowMaterial = objectCastsShadow && customShadowMaterial != null
    const baseColor = materialColor(material, materialContext)
    const useVertexColors = vertexColors && material?.vertexColors !== false
    const pbrProps = extractPbrProperties(material, materialContext)
    assertSupportedCustomFragmentInstancedAttributes(geometry, pbrProps)
    const uvStreams = textureUvStreamsForMeshMaterial(uvChannels, material)
    if (uvStreams.alphaMapUsesUv2 !== undefined) {
      pbrProps.alphaMapUsesUv2 = uvStreams.alphaMapUsesUv2
    }
    applyPbrUvStreamFlags(pbrProps, uvStreams)
    const textureInfo = extractTextureData(material, materialContext)
    const castShadow = objectCastsShadow && !usesCustomShadowMaterial ? true : undefined
    const receiveShadow = objectReceivesShadow ? true : undefined
    const clipping = clippingState(clippingContext, material, localClippingEnabled)
    const wireframe = isMeshWireframeMaterial(material)

    if (index) {
      const indices = indexRangeWithCache(cache, geometry, index, group.start, group.count)
      if (indices.length % 3 !== 0) {
        throw new Error(`THREE.Mesh "${object.name || object.uuid || '<unnamed>'}" has a non-triangle index range`)
      }
      const renderIndices = wireframe ? wireframeIndicesForTriangles(indices) : indices

      const expandedIndices = expandIndicesForInstances(renderIndices, position.count, instancedGeometryCount)
      const expandedPositions = expandVec3ValuesForInstancesWithCache(
        cache,
        geometry,
        position,
        positions,
        0,
        position.count,
        instancedGeometryCount,
        instancedPositionOffset,
        instancedPositionScale,
      )
      const expandedNormals = normalAttribute && normals
        ? expandNormalValuesForInstancesWithCache(cache, geometry, normalAttribute, normals, 0, position.count, instancedGeometryCount)
        : undefined
      const expandedUvs = uvStreams.uvs
        ? expandUvChannelForInstancesWithCache(cache, geometry, uvStreams.uvs, 0, position.count, instancedGeometryCount)
        : undefined
      const expandedSecondaryUvs = uvStreams.uvs2
        ? expandUvChannelForInstancesWithCache(cache, geometry, uvStreams.uvs2, 0, position.count, instancedGeometryCount)
        : undefined

      for (const instance of instances) {
        const color = instanceColor(baseColor, instance)
        const sortInfo = sortInfoForObject(object, material, camera, meshes.length, groupOrder, instance.transform, geometry, group, sortItemObject, sortItemZOverride)
        const sortKeys = mergeSortKeys(sortInfo.keys, sortKeyOverride)
        pushMesh(meshes, {
          positions: expandedPositions,
          indices: expandedIndices,
          normals: expandedNormals,
          color,
          colors: useVertexColors
            ? expandColorAttributeForInstancesWithCache(cache, geometry, vertexColors!, color, 0, position.count, instancedGeometryCount)
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
          castShadow,
          receiveShadow,
          clipShadows: clipShadowsForMaterial(material, clippingContext),
          ...clipping,
          ...sortKeys,
          ...pbrProps,
        }, sortInfo.item, needsTransparentDoubleSidePass(material, pbrProps, wireframe))
      }
    } else {
      if (group.count % 3 !== 0) {
        throw new Error(`THREE.Mesh "${object.name || object.uuid || '<unnamed>'}" has a non-triangle vertex range`)
      }

      const expandedGroupPositions = expandVec3ValuesForInstancesWithCache(
        cache,
        geometry,
        position,
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
        const sortInfo = sortInfoForObject(object, material, camera, meshes.length, groupOrder, instance.transform, geometry, group, sortItemObject, sortItemZOverride)
        const sortKeys = mergeSortKeys(sortInfo.keys, sortKeyOverride)
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
          castShadow,
          receiveShadow,
          clipShadows: clipShadowsForMaterial(material, clippingContext),
          ...clipping,
          ...sortKeys,
          ...pbrProps,
        }, sortInfo.item, needsTransparentDoubleSidePass(material, pbrProps, wireframe))
      }
    }

    if (usesCustomShadowMaterial && customShadowMaterial?.visible !== false) {
      appendShadowOnlyMeshGroup(
        object,
        camera,
        meshes,
        group,
        groupOrder,
        clippingContext,
        localClippingEnabled,
        customShadowMaterial,
        material,
        materialContext,
        positions,
        normals,
        normalAttribute,
        uvs,
        uvChannels,
        vertexColors,
        position.count,
        index,
        instancedGeometryCount,
        instancedPositionOffset,
        instancedPositionScale,
        instances,
        cache,
      )
    }

    invokeObjectRenderCallback(callbackObject.onAfterRender, 'onAfterRender', callbackContext, callbackObject, camera, geometry, material, group)
  }
}
