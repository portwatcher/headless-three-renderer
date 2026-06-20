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

interface FlattenedMesh {
  mesh: NativeSceneMesh
  sortItem: RenderSortItem
  groupOrder: number
  renderOrder: number
  sortZ: number
  materialSortKey: number
  materialVariant: number
  sortIndex: number
}

interface MeshInstance {
  transform: number[]
  color?: Color4
}

interface BatchedMeshDraw {
  range: { start: number; count: number }
  instanceId: number
  instance: MeshInstance
  z: number
}

export type ShadowMaterialMode = 'depth' | 'distance'

interface LineSegmentDistance {
  a: number
  b: number
  d0: number
  d1: number
}

interface DashedLineExpansion {
  positions: number[]
  uvs?: number[]
  uvs2?: number[]
  colors?: number[]
}

interface InstancedAttributeRef {
  attribute: ThreeBufferAttributeLike
  label: string
}

const SupportedInstancedBufferGeometryAttributes = new Set([
  'color',
  'normal',
  'uv',
  'uv1',
  'uv2',
  'uv3',
  'instanceOffset',
  'instancePosition',
  'offset',
  'translate',
  'translation',
])

interface ThickLineExpansion {
  center: [number, number, number]
  colors?: number[]
  indices: number[]
  positions: number[]
  uvs?: number[]
  uvs2?: number[]
}

interface TextureUvStreams {
  uvs: UvChannel | null
  uvs2: UvChannel | null
  textureUsesUv2?: boolean
  alphaMapUsesUv2?: boolean
  pbrUsesUv2?: Partial<Record<PbrUvFlag, boolean>>
}

export interface SceneExtractionCache {
  meshGeometry: WeakMap<ThreeBufferGeometryLike, unknown>
  batchedGeometryViews: WeakMap<ThreeBufferGeometryLike, Map<string, CachedBatchedGeometryView>>
  dashedLines: WeakMap<ThreeBufferGeometryLike, Map<string, CachedDashedLineExpansion>>
  texturePayloads: TextureExtractionCache
  materialColors: MaterialColorExtractionCache
  textureStates: TextureStateExtractionCache
  materialRenderStates: MaterialRenderStateExtractionCache
  materialScalarFeatures: MaterialScalarFeatureExtractionCache
  pointBillboards: WeakMap<ThreeObject3DLike, Map<string, CachedPointBillboardExpansion>>
  spriteBillboards: WeakMap<ThreeObject3DLike, CachedSpriteBillboardExpansion>
}

interface CachedMeshGeometryExtraction {
  signature: MeshGeometrySignature
  extraction: MeshGeometryExtraction
}

interface CachedBatchedGeometryView {
  signature: BatchedGeometryViewSignature
  view: ThreeBufferGeometryLike
}

interface CachedDashedLineExpansion {
  signature: DashedLineSignature
  expansion: DashedLineExpansion
}

interface CachedPointBillboardExpansion {
  signature: PointBillboardSignature
  expansion: PointBillboardExpansion
}

interface CachedSpriteBillboardExpansion {
  signature: SpriteBillboardSignature
  expansion: SpriteBillboardExpansion
}

interface SpriteBillboardExpansion {
  positions: number[]
  indices: number[]
  uvs: number[]
}

interface SpriteBillboardSignature {
  matrix: number[]
  center: [number, number]
  scaleX: number
  scaleY: number
  rotation: number
  sizeAttenuation?: boolean
  cameraRight: [number, number, number]
  cameraUp: [number, number, number]
  cameraView: number[] | null
  cameraIsPerspective?: boolean
}

interface PointBillboardExpansion {
  positions: number[]
  indices: number[]
  uvs: number[]
  uvs2?: number[]
  colors?: number[]
  pointRefs: Array<{ pointIndex: number, instance: number }>
}

interface PointBillboardSignature {
  cacheable: boolean
  positions: number[]
  positionCount: number
  index: number[] | null
  groupStart: number
  groupCount: number
  instancedGeometryCount: number
  instancedPositionOffset: AttributeSignature
  transform: number[]
  cameraRight: [number, number, number]
  cameraUp: [number, number, number]
  cameraProjection: number[] | null
  cameraView: number[] | null
  cameraIsPerspective?: boolean
  viewportHeight: number
  pointSize: number
  sizeAttenuation?: boolean
  uvs: UvChannelSignature
  uvs2: UvChannelSignature
  useVertexColors: boolean
  vertexColors: AttributeSignature
  baseColor?: Color4
}

interface DashedLineSignature {
  cacheable: boolean
  geometryVersion?: number
  position: AttributeSignature
  index: AttributeSignature
  uv: UvChannelSignature
  uv2: UvChannelSignature
  lineDistance: AttributeSignature
  start: number
  end: number
  sourceLength: number
  isLineSegments?: boolean
  isLineLoop?: boolean
  isLine?: boolean
  dashSize: number
  gapSize: number
  scale: number
}

interface UvChannelSignature {
  attribute: AttributeSignature
  values?: number[]
  label?: string
}

interface BatchedGeometryViewSignature {
  cacheable: boolean
  geometryVersion?: number
  rangeStart: number
  rangeCount: number
  position: AttributeSignature
  index: AttributeSignature
}

interface MeshGeometryExtraction {
  position: ThreeBufferAttributeLike
  positions: number[]
  uvChannels: Array<UvChannel | null>
  uvs: number[] | null
  normalAttribute?: ThreeBufferAttributeLike
  normals: number[] | null
  vertexColors?: ThreeBufferAttributeLike
  index: number[] | null
  sourceIndex: number[]
  groups: GeometryGroup[]
  instancedGeometryCount: number
  instancedPositionOffset: InstancedAttributeRef | null
}

interface MeshGeometrySignature {
  cacheable: boolean
  geometryVersion?: number
  isInstancedBufferGeometry?: boolean
  instanceCount?: number
  drawRange: string
  drawRangeStart?: unknown
  drawRangeCount?: unknown
  groups: string
  position: AttributeSignature
  normal: AttributeSignature
  color: AttributeSignature
  index: AttributeSignature
  uv: AttributeSignature
  uv1: AttributeSignature
  uv2: AttributeSignature
  uv3: AttributeSignature
  instancedPositionOffsetName?: string
  instancedPositionOffset: AttributeSignature
  instancedAttributes: Array<{ name: string; signature: AttributeSignature }>
}

interface AttributeSignature {
  ref?: ThreeBufferAttributeLike
  version?: number
  count?: number
  itemSize?: number
  normalized?: boolean
  array?: ArrayLike<number>
  dataArray?: ArrayLike<number>
  dataStride?: number
  offset?: number
  isInstancedBufferAttribute?: boolean
  meshPerAttribute?: number
}

type PbrUvFlag = Extract<keyof PbrProperties, `${string}UsesUv2`>

interface MaterialUvSlot {
  texture?: { channel?: number } | null
  textureFlag?: 'textureUsesUv2' | 'alphaMapUsesUv2'
  pbrFlag?: PbrUvFlag
}

interface UvChannel {
  attribute: ThreeBufferAttributeLike
  label: string
  values: number[]
}

interface ClippingContext {
  unionPlanes: readonly NativeClippingPlane[]
  intersectionPlanes: readonly NativeClippingPlane[]
  clipShadows: boolean
}

interface SceneSortOptions {
  sortObjects?: boolean
  opaqueSort?: RenderSortFunction | null
  transparentSort?: RenderSortFunction | null
  opaque?: boolean
  transparent?: boolean
}

interface RenderCallbackContext {
  renderer: unknown
  scene: ThreeObject3DLike
}

interface MeshSortInfo {
  keys: Pick<NativeSceneMesh, 'groupOrder' | 'renderOrder' | 'sortZ' | 'sortIndex' | 'materialSortKey' | 'materialVariant'>
  item: RenderSortItem
}

type SortKeyOverride = Partial<MeshSortInfo['keys']>

const MAX_POINT_SPRITE_SIZE = 64

export function createSceneExtractionCache(): SceneExtractionCache {
  return {
    meshGeometry: new WeakMap(),
    batchedGeometryViews: new WeakMap(),
    dashedLines: new WeakMap(),
    texturePayloads: new WeakMap(),
    materialColors: new WeakMap(),
    textureStates: new WeakMap(),
    materialRenderStates: new WeakMap(),
    materialScalarFeatures: new WeakMap(),
    pointBillboards: new WeakMap(),
    spriteBillboards: new WeakMap(),
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
  return sortFlattenedMeshes(meshes, sortOptions)
    .map(({ mesh }) => mesh)
}

function visitObject(
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
    ? finiteMaterialOrObjectNumber(object.renderOrder, 'object.renderOrder', 0)
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

function appendBatchedMesh(
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

function appendMesh(
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
  const instances = instanceOverride ?? meshInstances(object, meshTransform)
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
      const indices = index.slice(group.start, group.start + group.count)
      if (indices.length % 3 !== 0) {
        throw new Error(`THREE.Mesh "${object.name || object.uuid || '<unnamed>'}" has a non-triangle index range`)
      }
      const renderIndices = wireframe ? wireframeIndicesForTriangles(indices) : indices

      const expandedIndices = expandIndicesForInstances(renderIndices, position.count, instancedGeometryCount)
      const expandedPositions = expandVec3ValuesForInstances(
        positions,
        0,
        position.count,
        instancedGeometryCount,
        instancedPositionOffset,
      )
      const expandedNormals = normalAttribute && normals
        ? expandNormalValuesForInstances(normalAttribute, normals, 0, position.count, instancedGeometryCount)
        : undefined
      const expandedUvs = uvStreams.uvs
        ? expandUvChannelForInstances(uvStreams.uvs, 0, position.count, instancedGeometryCount)
        : undefined
      const expandedSecondaryUvs = uvStreams.uvs2
        ? expandUvChannelForInstances(uvStreams.uvs2, 0, position.count, instancedGeometryCount)
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
            ? expandColorAttributeForInstances(vertexColors!, color, 0, position.count, instancedGeometryCount)
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
        }, sortInfo.item)
      }
    } else {
      if (group.count % 3 !== 0) {
        throw new Error(`THREE.Mesh "${object.name || object.uuid || '<unnamed>'}" has a non-triangle vertex range`)
      }

      const expandedGroupPositions = expandVec3ValuesForInstances(
        positions,
        group.start,
        group.count,
        instancedGeometryCount,
        instancedPositionOffset,
      )
      const expandedGroupNormals = normalAttribute && normals
        ? expandNormalValuesForInstances(normalAttribute, normals, group.start, group.count, instancedGeometryCount)
        : undefined
      const expandedGroupUvs = uvStreams.uvs
        ? expandUvChannelForInstances(uvStreams.uvs, group.start, group.count, instancedGeometryCount)
        : undefined
      const expandedGroupSecondaryUvs = uvStreams.uvs2
        ? expandUvChannelForInstances(uvStreams.uvs2, group.start, group.count, instancedGeometryCount)
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
            ? expandColorAttributeForInstances(vertexColors!, color, group.start, group.count, instancedGeometryCount)
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
        }, sortInfo.item)
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
        instances,
      )
    }

    invokeObjectRenderCallback(callbackObject.onAfterRender, 'onAfterRender', callbackContext, callbackObject, camera, geometry, material, group)
  }
}

function meshGeometryExtraction(
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

function readMeshGeometryExtraction(
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
  }
}

function meshGeometrySignature(
  geometry: ThreeBufferGeometryLike,
  position: ThreeBufferAttributeLike,
): MeshGeometrySignature {
  const attributes = geometryAttributes(geometry)
  const instancedAttributes = Object.entries(attributes)
    .filter((entry): entry is [string, ThreeBufferAttributeLike] => isInstancedAttribute(entry[1]))
    .map(([name, attribute]) => ({ name, signature: attributeSignature(attribute) }))
  const instancedPositionOffset = namedInstancedOffsetAttribute(geometry)
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
    instancedAttributes,
  }
  signature.cacheable = meshGeometrySignatureCacheable(signature)
  return signature
}

function meshGeometrySignatureCacheable(signature: MeshGeometrySignature): boolean {
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
    ...signature.instancedAttributes.map(({ signature }) => signature),
  ].every(attributeSignatureCacheable)
}

function attributeSignature(attribute: ThreeBufferAttributeLike | null | undefined): AttributeSignature {
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

function attributeSignatureCacheable(signature: AttributeSignature): boolean {
  return signature.ref == null || typeof signature.version === 'number'
}

function namedInstancedOffsetAttribute(
  geometry: ThreeBufferGeometryLike,
): { name: string; attribute: ThreeBufferAttributeLike } | null {
  const names = ['instanceOffset', 'instancePosition', 'offset', 'translate', 'translation']
  for (const name of names) {
    const attribute = getAttribute(geometry, name)
    if (isInstancedAttribute(attribute)) return { name, attribute }
  }
  return null
}

function geometryGroupsSignature(groups: ThreeBufferGeometryLike['groups']): string {
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

function geometryDrawRangeSignature(drawRange: ThreeBufferGeometryLike['drawRange']): string {
  if (drawRange == null) return 'none'
  if (typeof drawRange !== 'object' || Array.isArray(drawRange)) return `invalid:${typeof drawRange}`
  return [
    typeof drawRange.start,
    String(drawRange.start),
    typeof drawRange.count,
    String(drawRange.count),
  ].join(':')
}

function sameMeshGeometrySignature(a: MeshGeometrySignature, b: MeshGeometrySignature): boolean {
  return a.cacheable === b.cacheable
    && a.geometryVersion === b.geometryVersion
    && a.isInstancedBufferGeometry === b.isInstancedBufferGeometry
    && a.instanceCount === b.instanceCount
    && a.drawRange === b.drawRange
    && Object.is(a.drawRangeStart, b.drawRangeStart)
    && Object.is(a.drawRangeCount, b.drawRangeCount)
    && a.groups === b.groups
    && a.instancedPositionOffsetName === b.instancedPositionOffsetName
    && sameAttributeSignature(a.position, b.position)
    && sameAttributeSignature(a.normal, b.normal)
    && sameAttributeSignature(a.color, b.color)
    && sameAttributeSignature(a.index, b.index)
    && sameAttributeSignature(a.uv, b.uv)
    && sameAttributeSignature(a.uv1, b.uv1)
    && sameAttributeSignature(a.uv2, b.uv2)
    && sameAttributeSignature(a.uv3, b.uv3)
    && sameAttributeSignature(a.instancedPositionOffset, b.instancedPositionOffset)
    && sameInstancedAttributeSignatures(a.instancedAttributes, b.instancedAttributes)
}

function sameInstancedAttributeSignatures(
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

function sameAttributeSignature(a: AttributeSignature, b: AttributeSignature): boolean {
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

function appendShadowOnlyMeshGroup(
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
  instances: MeshInstance[],
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
    const indices = index.slice(group.start, group.start + group.count)
    if (indices.length % 3 !== 0) {
      throw new Error(`THREE.Mesh "${object.name || object.uuid || '<unnamed>'}" has a non-triangle index range`)
    }
    const renderIndices = wireframe ? wireframeIndicesForTriangles(indices) : indices
    const expandedIndices = expandIndicesForInstances(renderIndices, vertexCount, instancedGeometryCount)
    const expandedPositions = expandVec3ValuesForInstances(
      positions,
      0,
      vertexCount,
      instancedGeometryCount,
      instancedPositionOffset,
    )
    const expandedNormals = normalAttribute && normals
      ? expandNormalValuesForInstances(normalAttribute, normals, 0, vertexCount, instancedGeometryCount)
      : undefined
    const expandedUvs = uvStreams.uvs
      ? expandUvChannelForInstances(uvStreams.uvs, 0, vertexCount, instancedGeometryCount)
      : undefined
    const expandedSecondaryUvs = uvStreams.uvs2
      ? expandUvChannelForInstances(uvStreams.uvs2, 0, vertexCount, instancedGeometryCount)
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
          ? expandColorAttributeForInstances(vertexColors!, color, 0, vertexCount, instancedGeometryCount)
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

  const expandedGroupPositions = expandVec3ValuesForInstances(
    positions,
    group.start,
    group.count,
    instancedGeometryCount,
    instancedPositionOffset,
  )
  const expandedGroupNormals = normalAttribute && normals
    ? expandNormalValuesForInstances(normalAttribute, normals, group.start, group.count, instancedGeometryCount)
    : undefined
  const expandedGroupUvs = uvStreams.uvs
    ? expandUvChannelForInstances(uvStreams.uvs, group.start, group.count, instancedGeometryCount)
    : undefined
  const expandedGroupSecondaryUvs = uvStreams.uvs2
    ? expandUvChannelForInstances(uvStreams.uvs2, group.start, group.count, instancedGeometryCount)
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
        ? expandColorAttributeForInstances(vertexColors!, color, group.start, group.count, instancedGeometryCount)
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

function customShadowMaterialForMode(
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

function materialForObjectGroup(
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

function invokeObjectRenderCallback(
  callback: unknown,
  name: 'onBeforeRender' | 'onAfterRender',
  context: RenderCallbackContext | undefined,
  object: ThreeObject3DLike,
  camera: ThreeCameraLike | undefined,
  geometry: ThreeBufferGeometryLike | undefined,
  material: ThreeMaterialLike | undefined,
  group?: GeometryGroup,
): void {
  if (callback == null) return
  if (typeof callback !== 'function') {
    throw new TypeError(`THREE.Object3D.${name} must be a function when provided.`)
  }
  if (isInternalBatchedMeshRenderCallback(object, callback, name)) return
  if (!context) return
  callback.call(object, context.renderer, context.scene, camera, geometry, material, group)
}

function isInternalBatchedMeshRenderCallback(
  object: ThreeObject3DLike,
  callback: Function,
  name: 'onBeforeRender' | 'onAfterRender',
): boolean {
  if (name !== 'onBeforeRender' || object.isBatchedMesh !== true) return false
  if (Object.prototype.hasOwnProperty.call(object, name)) return false

  let prototype = Object.getPrototypeOf(object)
  while (prototype && prototype !== Object.prototype) {
    if (prototype.constructor?.name === 'BatchedMesh' && typeof prototype[name] === 'function') {
      return callback === prototype[name]
    }
    prototype = Object.getPrototypeOf(prototype)
  }
  return false
}

function shadowOnlyMainPassState(): Pick<
  NativeSceneMesh,
  'blending' | 'colorWrite' | 'depthTest' | 'depthWrite' | 'stencilWrite' | 'transparent'
> {
  return {
    blending: 'none',
    colorWrite: false,
    depthTest: false,
    depthWrite: false,
    stencilWrite: false,
    transparent: false,
  }
}

function appendSprite(
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
  const objectCastsShadow = optionalObjectBoolean(object.castShadow, 'object.castShadow') === true
  optionalObjectBoolean(object.receiveShadow, 'object.receiveShadow')

  const material = materialForObjectGroup(object, 0, overrideMaterial)
  if (material?.visible === false) return

  validateSpriteScale(object)
  const matrix = matrixElements(object.matrixWorld!, 'sprite.matrixWorld')
  const center: [number, number] = [
    finiteMaterialOrObjectNumber(object.center?.x, 'Sprite.center.x', 0.5),
    finiteMaterialOrObjectNumber(object.center?.y, 'Sprite.center.y', 0.5),
  ]
  if (spriteOutsideFrustum(object, camera, matrix, center)) return

  invokeObjectRenderCallback(object.onBeforeRender, 'onBeforeRender', callbackContext, object, camera, object.geometry, material)

  const worldPosition = [matrix[12], matrix[13], matrix[14]]
  let scaleX = columnLength3(matrix, 0)
  let scaleY = columnLength3(matrix, 4)

  const sizeAttenuation = optionalSceneBoolean(material?.sizeAttenuation, 'material.sizeAttenuation')
  if (sizeAttenuation === false && camera?.isPerspectiveCamera === true) {
    const viewZ = viewSpaceZ(worldPosition, camera)
    if (Number.isFinite(viewZ)) {
      scaleX *= -viewZ
      scaleY *= -viewZ
    }
  }

  if (scaleX <= 0 || scaleY <= 0) return

  const axes = cameraBillboardAxes(camera)
  const rotation = finiteMaterialOrObjectNumber(material?.rotation, 'material.rotation', 0)
  const billboard = spriteBillboardExpansion(
    object,
    matrix,
    worldPosition,
    center,
    scaleX,
    scaleY,
    rotation,
    sizeAttenuation,
    axes,
    camera,
    cache,
  )
  const positions = billboard.positions
  const indices = billboard.indices
  const uvs = billboard.uvs

  const textureInfo = extractTextureData(material, materialContext)
  const sortInfo = sortInfoForObject(object, material, camera, meshes.length, groupOrder)
  const pbrProps = extractPbrProperties(material, materialContext)
  pbrProps.alphaMapUsesUv2 = false
  const clipping = clippingState(clippingContext, material, localClippingEnabled)
  const customShadowMaterial = customShadowMaterialForMode(object, shadowMaterialMode)
  const usesCustomShadowMaterial = objectCastsShadow && customShadowMaterial != null

  pushMesh(meshes, {
    positions,
    indices,
    uvs,
    color: materialColor(material, materialContext),
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
    textureUsesUv2: false,
    transform: IDENTITY_4X4.slice(),
    transparent: material?.transparent !== false,
    castShadow: objectCastsShadow && !usesCustomShadowMaterial ? true : undefined,
    receiveShadow: undefined,
    clipShadows: clipShadowsForMaterial(material, clippingContext),
    ...clipping,
    ...sortInfo.keys,
    ...pbrProps,
  }, sortInfo.item)

  if (usesCustomShadowMaterial && customShadowMaterial?.visible !== false) {
    appendShadowOnlyBillboardMesh(
      object,
      camera,
      meshes,
      groupOrder,
      clippingContext,
      localClippingEnabled,
      customShadowMaterial,
      material,
      materialContext,
      positions,
      indices,
      uvs,
    )
  }

  invokeObjectRenderCallback(object.onAfterRender, 'onAfterRender', callbackContext, object, camera, object.geometry, material)
}

function spriteBillboardExpansion(
  object: ThreeObject3DLike,
  matrix: number[],
  worldPosition: number[],
  center: [number, number],
  scaleX: number,
  scaleY: number,
  rotation: number,
  sizeAttenuation: boolean | undefined,
  axes: { right: [number, number, number]; up: [number, number, number] },
  camera: ThreeCameraLike | undefined,
  cache: SceneExtractionCache | undefined,
): SpriteBillboardExpansion {
  const signature = spriteBillboardSignature(
    matrix,
    center,
    scaleX,
    scaleY,
    rotation,
    sizeAttenuation,
    axes,
    camera,
  )
  if (cache) {
    const cached = cache.spriteBillboards.get(object)
    if (cached && sameSpriteBillboardSignature(cached.signature, signature)) {
      return cached.expansion
    }
  }

  const expansion = readSpriteBillboardExpansion(worldPosition, center, scaleX, scaleY, rotation, axes)
  if (cache) {
    cache.spriteBillboards.set(object, { signature, expansion })
  }
  return expansion
}

function readSpriteBillboardExpansion(
  worldPosition: number[],
  center: [number, number],
  scaleX: number,
  scaleY: number,
  rotation: number,
  axes: { right: [number, number, number]; up: [number, number, number] },
): SpriteBillboardExpansion {
  const cos = Math.cos(rotation)
  const sin = Math.sin(rotation)
  const corners = [
    [-0.5, -0.5, 0, 0],
    [0.5, -0.5, 1, 0],
    [0.5, 0.5, 1, 1],
    [-0.5, 0.5, 0, 1],
  ]
  const positions: number[] = []
  const uvs: number[] = []
  for (const [x, y, u, v] of corners) {
    const alignedX = (x - (center[0] - 0.5)) * scaleX
    const alignedY = (y - (center[1] - 0.5)) * scaleY
    const rotatedX = cos * alignedX - sin * alignedY
    const rotatedY = sin * alignedX + cos * alignedY
    positions.push(
      worldPosition[0] + axes.right[0] * rotatedX + axes.up[0] * rotatedY,
      worldPosition[1] + axes.right[1] * rotatedX + axes.up[1] * rotatedY,
      worldPosition[2] + axes.right[2] * rotatedX + axes.up[2] * rotatedY,
    )
    uvs.push(u, v)
  }
  return {
    positions,
    indices: [0, 1, 2, 0, 2, 3],
    uvs,
  }
}

function spriteBillboardSignature(
  matrix: number[],
  center: [number, number],
  scaleX: number,
  scaleY: number,
  rotation: number,
  sizeAttenuation: boolean | undefined,
  axes: { right: [number, number, number]; up: [number, number, number] },
  camera: ThreeCameraLike | undefined,
): SpriteBillboardSignature {
  return {
    matrix: matrix.slice(0, 16),
    center: center.slice() as [number, number],
    scaleX,
    scaleY,
    rotation,
    sizeAttenuation,
    cameraRight: axes.right.slice() as [number, number, number],
    cameraUp: axes.up.slice() as [number, number, number],
    cameraView: matrixValues(camera?.matrixWorldInverse?.elements),
    cameraIsPerspective: camera?.isPerspectiveCamera,
  }
}

function sameSpriteBillboardSignature(a: SpriteBillboardSignature, b: SpriteBillboardSignature): boolean {
  return sameNumberArray(a.matrix, b.matrix)
    && sameNumberArray(a.center, b.center)
    && a.scaleX === b.scaleX
    && a.scaleY === b.scaleY
    && a.rotation === b.rotation
    && a.sizeAttenuation === b.sizeAttenuation
    && sameNumberArray(a.cameraRight, b.cameraRight)
    && sameNumberArray(a.cameraUp, b.cameraUp)
    && sameOptionalNumberArray(a.cameraView, b.cameraView)
    && a.cameraIsPerspective === b.cameraIsPerspective
}

function validateSpriteScale(object: ThreeObject3DLike): void {
  finiteMaterialOrObjectNumber(object.scale?.x, 'Sprite.scale.x', 1)
  finiteMaterialOrObjectNumber(object.scale?.y, 'Sprite.scale.y', 1)
  finiteMaterialOrObjectNumber(object.scale?.z, 'Sprite.scale.z', 1)
}

function appendPoints(
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
  const objectCastsShadow = optionalObjectBoolean(object.castShadow, 'object.castShadow') === true
  optionalObjectBoolean(object.receiveShadow, 'object.receiveShadow')

  const geometry = object.geometry!
  const geometryExtraction = meshGeometryExtraction(geometry, cache)
  if (!geometryExtraction) return

  const {
    position,
    positions,
    vertexColors,
    index,
    sourceIndex,
    groups,
    instancedGeometryCount,
    instancedPositionOffset,
  } = geometryExtraction
  const pointUvChannels = geometryExtraction.uvChannels
  const primaryPointUvs = geometryExtraction.uvs
  const transform = matrixElements(object.matrixWorld!, 'points.matrixWorld')
  const axes = cameraBillboardAxes(camera)

  for (const group of groups) {
    const material = materialForObjectGroup(object, group.materialIndex, overrideMaterial)
    if (material?.visible === false) continue
    invokeObjectRenderCallback(object.onBeforeRender, 'onBeforeRender', callbackContext, object, camera, geometry, material, group)
    const pointUvStreams = primaryPointUvs && (material?.map || material?.alphaMap)
      ? textureUvStreamsForMapAlphaMaterial(pointUvChannels, {
        map: material.map,
        alphaMap: material.alphaMap,
      })
      : null

    const baseColor = materialColor(material, materialContext)
    const useVertexColors = vertexColors && material?.vertexColors !== false
    const pointSize = positiveMaterialOrObjectNumber(material?.size, 'material.size', 1)
    const sizeAttenuation = optionalSceneBoolean(material?.sizeAttenuation, 'material.sizeAttenuation')
    const billboard = pointBillboardExpansion(
      object,
      group,
      position,
      positions,
      index ?? sourceIndex,
      instancedGeometryCount,
      instancedPositionOffset,
      transform,
      axes,
      camera,
      viewportHeight,
      pointSize,
      sizeAttenuation,
      pointUvStreams,
      vertexColors,
      useVertexColors ? baseColor : undefined,
      cache,
    )

    if (billboard.positions.length === 0) continue
    const outputPositions = billboard.positions
    const outputUvs = billboard.uvs
    const outputUvs2 = billboard.uvs2
    const outputColors = billboard.colors
    const outputIndices = billboard.indices
    const outputPointRefs = billboard.pointRefs

    const textureInfo = extractTextureData(material, materialContext)
    const sortInfo = sortInfoForObject(object, material, camera, meshes.length, groupOrder, undefined, geometry, group)
    const pbrProps = extractPbrProperties(material, materialContext)
    assertSupportedCustomFragmentInstancedAttributes(geometry, pbrProps)
    pbrProps.alphaMapUsesUv2 = pointUvStreams?.alphaMapUsesUv2 ?? false
    const clipping = clippingState(clippingContext, material, localClippingEnabled)
    const customShadowMaterial = customShadowMaterialForMode(object, shadowMaterialMode)
    const usesCustomShadowMaterial = objectCastsShadow && customShadowMaterial != null
    const effectiveCustomShadowMaterial = usesCustomShadowMaterial
      ? shadowMaterialWithSourceShadowState(customShadowMaterial, material)
      : null
    const pointShadowUvStreams = primaryPointUvs && effectiveCustomShadowMaterial && (
      effectiveCustomShadowMaterial.map ||
      effectiveCustomShadowMaterial.alphaMap
    )
      ? textureUvStreamsForMapAlphaMaterial(pointUvChannels, {
        map: effectiveCustomShadowMaterial.map,
        alphaMap: effectiveCustomShadowMaterial.alphaMap,
      })
      : null

    pushMesh(meshes, {
      positions: outputPositions,
      indices: outputIndices,
      uvs: outputUvs,
      uvs2: outputUvs2,
      color: baseColor,
      colors: outputColors,
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
      textureUsesUv2: pointUvStreams?.textureUsesUv2 ?? false,
      transform: IDENTITY_4X4.slice(),
      transparent: material?.transparent === true || (material?.opacity != null && material.opacity < 1),
      topology: 'triangles',
      castShadow: objectCastsShadow && !usesCustomShadowMaterial ? true : undefined,
      receiveShadow: false,
      clipShadows: clipShadowsForMaterial(material, clippingContext),
      ...clipping,
      ...sortInfo.keys,
      ...pbrProps,
      shadingModel: 'basic',
    }, sortInfo.item)

    if (usesCustomShadowMaterial && customShadowMaterial?.visible !== false) {
      appendShadowOnlyBillboardMesh(
        object,
        camera,
        meshes,
        groupOrder,
        clippingContext,
        localClippingEnabled,
        customShadowMaterial,
        material,
        materialContext,
        outputPositions,
        outputIndices,
        pointShadowUvStreams?.uvs ? expandPointBillboardUvStream(pointShadowUvStreams.uvs, outputPointRefs) : outputUvs,
        pointShadowUvStreams?.uvs2 ? expandPointBillboardUvStream(pointShadowUvStreams.uvs2, outputPointRefs) : undefined,
        pointShadowUvStreams?.textureUsesUv2 ?? false,
        pointShadowUvStreams?.alphaMapUsesUv2 ?? false,
      )
    }
    invokeObjectRenderCallback(object.onAfterRender, 'onAfterRender', callbackContext, object, camera, geometry, material, group)
  }
}

function pointBillboardExpansion(
  object: ThreeObject3DLike,
  group: GeometryGroup,
  position: ThreeBufferAttributeLike,
  positions: number[],
  index: number[] | null,
  instancedGeometryCount: number,
  instancedPositionOffset: InstancedAttributeRef | null,
  transform: number[],
  axes: { right: [number, number, number]; up: [number, number, number] },
  camera: ThreeCameraLike | undefined,
  viewportHeight: number,
  pointSize: number,
  sizeAttenuation: boolean | undefined,
  pointUvStreams: TextureUvStreams | null,
  vertexColors: ThreeBufferAttributeLike | undefined,
  vertexColorBase: Color4 | undefined,
  cache: SceneExtractionCache | undefined,
): PointBillboardExpansion {
  const signature = pointBillboardSignature(
    group,
    position,
    positions,
    index,
    instancedGeometryCount,
    instancedPositionOffset,
    transform,
    axes,
    camera,
    viewportHeight,
    pointSize,
    sizeAttenuation,
    pointUvStreams,
    vertexColors,
    vertexColorBase,
  )
  const cacheKey = `${group.start}:${group.count}:${group.materialIndex ?? 0}`
  if (cache && signature.cacheable) {
    const objectCache = cache.pointBillboards.get(object)
    const cached = objectCache?.get(cacheKey)
    if (cached && samePointBillboardSignature(cached.signature, signature)) {
      return cached.expansion
    }
  }

  const expansion = readPointBillboardExpansion(
    group,
    position,
    positions,
    index,
    instancedGeometryCount,
    instancedPositionOffset,
    transform,
    axes,
    camera,
    viewportHeight,
    pointSize,
    sizeAttenuation,
    pointUvStreams,
    vertexColors,
    vertexColorBase,
  )
  if (cache && signature.cacheable) {
    let objectCache = cache.pointBillboards.get(object)
    if (!objectCache) {
      objectCache = new Map()
      cache.pointBillboards.set(object, objectCache)
    }
    objectCache.set(cacheKey, { signature, expansion })
  }
  return expansion
}

function readPointBillboardExpansion(
  group: GeometryGroup,
  position: ThreeBufferAttributeLike,
  positions: number[],
  index: number[] | null,
  instancedGeometryCount: number,
  instancedPositionOffset: InstancedAttributeRef | null,
  transform: number[],
  axes: { right: [number, number, number]; up: [number, number, number] },
  camera: ThreeCameraLike | undefined,
  viewportHeight: number,
  pointSize: number,
  sizeAttenuation: boolean | undefined,
  pointUvStreams: TextureUvStreams | null,
  vertexColors: ThreeBufferAttributeLike | undefined,
  vertexColorBase: Color4 | undefined,
): PointBillboardExpansion {
  const source = index ?? rangeIndices(position.count)
  const points = source.slice(group.start, group.start + group.count)
  const outputPositions: number[] = []
  const outputUvs: number[] = []
  const outputUvs2: number[] | undefined = pointUvStreams?.uvs2 ? [] : undefined
  const outputColors: number[] | undefined = vertexColorBase ? [] : undefined
  const outputIndices: number[] = []
  const outputPointRefs: Array<{ pointIndex: number, instance: number }> = []
  const corners = [
    [-0.5, -0.5, 0, 0],
    [0.5, -0.5, 1, 0],
    [0.5, 0.5, 1, 1],
    [-0.5, 0.5, 0, 1],
  ]

  for (let instance = 0; instance < instancedGeometryCount; instance += 1) {
    const offsetIndex = instancedPositionOffset
      ? instancedAttributeIndex(instancedPositionOffset.attribute, instance, instancedPositionOffset.label)
      : 0
    const offset: [number, number, number] = instancedPositionOffset
      ? [
        attributeComponent(instancedPositionOffset.attribute, offsetIndex, 0, instancedPositionOffset.label),
        attributeComponent(instancedPositionOffset.attribute, offsetIndex, 1, instancedPositionOffset.label),
        attributeComponent(instancedPositionOffset.attribute, offsetIndex, 2, instancedPositionOffset.label),
      ]
      : [0, 0, 0]

    for (let pointOffset = 0; pointOffset < points.length; pointOffset += 1) {
      const pointIndex = points[pointOffset]
      if (!Number.isInteger(pointIndex) || pointIndex < 0 || pointIndex >= position.count) continue

      const center = transformPoint(transform, [
        positions[pointIndex * 3] + offset[0],
        positions[pointIndex * 3 + 1] + offset[1],
        positions[pointIndex * 3 + 2] + offset[2],
      ])
      const worldSize = pointWorldSize(pointSize, center, sizeAttenuation, camera, viewportHeight)
      if (worldSize <= 0) continue

      const vertexBase = outputPositions.length / 3
      outputPointRefs.push({ pointIndex, instance })
      const pointColor = outputColors ? pointVertexColor(vertexColors!, vertexColorBase!, pointIndex, instance) : null
      for (const [x, y, u, v] of corners) {
        outputPositions.push(
          center[0] + axes.right[0] * x * worldSize + axes.up[0] * y * worldSize,
          center[1] + axes.right[1] * x * worldSize + axes.up[1] * y * worldSize,
          center[2] + axes.right[2] * x * worldSize + axes.up[2] * y * worldSize,
        )
        if (pointUvStreams?.uvs) {
          appendUvForVertex(outputUvs, pointUvStreams.uvs, pointIndex, instance)
          if (outputUvs2 && pointUvStreams.uvs2) {
            appendUvForVertex(outputUvs2, pointUvStreams.uvs2, pointIndex, instance)
          }
        } else {
          outputUvs.push(u, v)
        }
        if (pointColor) {
          outputColors!.push(pointColor[0], pointColor[1], pointColor[2], pointColor[3])
        }
      }
      outputIndices.push(vertexBase, vertexBase + 1, vertexBase + 2, vertexBase, vertexBase + 2, vertexBase + 3)
    }
  }

  return {
    positions: outputPositions,
    indices: outputIndices,
    uvs: outputUvs,
    uvs2: outputUvs2,
    colors: outputColors,
    pointRefs: outputPointRefs,
  }
}

function pointBillboardSignature(
  group: GeometryGroup,
  position: ThreeBufferAttributeLike,
  positions: number[],
  index: number[] | null,
  instancedGeometryCount: number,
  instancedPositionOffset: InstancedAttributeRef | null,
  transform: number[],
  axes: { right: [number, number, number]; up: [number, number, number] },
  camera: ThreeCameraLike | undefined,
  viewportHeight: number,
  pointSize: number,
  sizeAttenuation: boolean | undefined,
  pointUvStreams: TextureUvStreams | null,
  vertexColors: ThreeBufferAttributeLike | undefined,
  vertexColorBase: Color4 | undefined,
): PointBillboardSignature {
  const signature: PointBillboardSignature = {
    cacheable: true,
    positions,
    positionCount: position.count,
    index,
    groupStart: group.start,
    groupCount: group.count,
    instancedGeometryCount,
    instancedPositionOffset: attributeSignature(instancedPositionOffset?.attribute),
    transform: transform.slice(0, 16),
    cameraRight: axes.right.slice() as [number, number, number],
    cameraUp: axes.up.slice() as [number, number, number],
    cameraProjection: matrixValues(camera?.projectionMatrix?.elements),
    cameraView: matrixValues(camera?.matrixWorldInverse?.elements),
    cameraIsPerspective: camera?.isPerspectiveCamera,
    viewportHeight,
    pointSize,
    sizeAttenuation,
    uvs: uvChannelSignature(pointUvStreams?.uvs),
    uvs2: uvChannelSignature(pointUvStreams?.uvs2),
    useVertexColors: !!vertexColorBase,
    vertexColors: attributeSignature(vertexColorBase ? vertexColors : undefined),
    baseColor: vertexColorBase ? vertexColorBase.slice() as Color4 : undefined,
  }
  signature.cacheable = pointBillboardSignatureCacheable(signature)
  return signature
}

function pointBillboardSignatureCacheable(signature: PointBillboardSignature): boolean {
  return attributeSignatureCacheable(signature.instancedPositionOffset)
    && attributeSignatureCacheable(signature.uvs.attribute)
    && attributeSignatureCacheable(signature.uvs2.attribute)
    && attributeSignatureCacheable(signature.vertexColors)
}

function uvChannelSignature(channel: UvChannel | null | undefined): UvChannelSignature {
  if (!channel) return { attribute: {} }
  return {
    attribute: attributeSignature(channel.attribute),
    values: channel.values,
    label: channel.label,
  }
}

function matrixValues(matrix: ArrayLike<number> | undefined): number[] | null {
  if (!matrix || matrix.length < 16) return null
  const out = new Array<number>(16)
  for (let i = 0; i < 16; i += 1) out[i] = matrix[i]
  return out
}

function samePointBillboardSignature(a: PointBillboardSignature, b: PointBillboardSignature): boolean {
  return a.cacheable === b.cacheable
    && a.positions === b.positions
    && a.positionCount === b.positionCount
    && a.index === b.index
    && a.groupStart === b.groupStart
    && a.groupCount === b.groupCount
    && a.instancedGeometryCount === b.instancedGeometryCount
    && sameAttributeSignature(a.instancedPositionOffset, b.instancedPositionOffset)
    && sameNumberArray(a.transform, b.transform)
    && sameNumberArray(a.cameraRight, b.cameraRight)
    && sameNumberArray(a.cameraUp, b.cameraUp)
    && sameOptionalNumberArray(a.cameraProjection, b.cameraProjection)
    && sameOptionalNumberArray(a.cameraView, b.cameraView)
    && a.cameraIsPerspective === b.cameraIsPerspective
    && a.viewportHeight === b.viewportHeight
    && a.pointSize === b.pointSize
    && a.sizeAttenuation === b.sizeAttenuation
    && sameUvChannelSignature(a.uvs, b.uvs)
    && sameUvChannelSignature(a.uvs2, b.uvs2)
    && a.useVertexColors === b.useVertexColors
    && sameAttributeSignature(a.vertexColors, b.vertexColors)
    && sameOptionalNumberArray(a.baseColor, b.baseColor)
}

function sameUvChannelSignature(a: UvChannelSignature, b: UvChannelSignature): boolean {
  return sameAttributeSignature(a.attribute, b.attribute)
    && a.values === b.values
    && a.label === b.label
}

function sameOptionalNumberArray(a: ArrayLike<number> | null | undefined, b: ArrayLike<number> | null | undefined): boolean {
  if (a == null || b == null) return a == null && b == null
  return sameNumberArray(a, b)
}

function sameNumberArray(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i += 1) {
    if (!Object.is(a[i], b[i])) return false
  }
  return true
}

function expandPointBillboardUvStream(
  channel: UvChannel,
  pointRefs: Array<{ pointIndex: number, instance: number }>,
): number[] {
  const outputUvs: number[] = []
  for (const { pointIndex, instance } of pointRefs) {
    for (let corner = 0; corner < 4; corner += 1) {
      appendUvForVertex(outputUvs, channel, pointIndex, instance)
    }
  }
  return outputUvs
}

function appendShadowOnlyBillboardMesh(
  object: ThreeObject3DLike,
  camera: ThreeCameraLike | undefined,
  meshes: FlattenedMesh[],
  groupOrder: number,
  clippingContext: ClippingContext,
  localClippingEnabled: boolean,
  material: ThreeMaterialLike,
  sourceMaterial: ThreeMaterialLike | undefined,
  materialContext: MaterialExtractionContext,
  positions: number[],
  indices: number[],
  uvs: number[],
  uvs2: number[] | undefined = undefined,
  textureUsesUv2 = false,
  alphaMapUsesUv2 = false,
): void {
  const shadowMaterial = shadowMaterialWithSourceShadowState(material, sourceMaterial)
  const textureInfo = extractTextureData(shadowMaterial, materialContext)
  const sortInfo = sortInfoForObject(object, material, camera, meshes.length, groupOrder)
  const clipping = clippingState(clippingContext, shadowMaterial, localClippingEnabled)
  const hiddenMainPass = shadowOnlyMainPassState()
  const shadowProps = shadowPbrProperties(shadowMaterial, sourceMaterial, materialContext)
  shadowProps.alphaMapUsesUv2 = alphaMapUsesUv2

  pushMesh(meshes, {
    positions,
    indices,
    uvs,
    uvs2,
    color: materialColor(shadowMaterial, materialContext),
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
    textureUsesUv2,
    transform: IDENTITY_4X4.slice(),
    topology: 'triangles',
    castShadow: true,
    receiveShadow: false,
    clipShadows: clipShadowsForMaterial(shadowMaterial, clippingContext),
    ...clipping,
    ...sortInfo.keys,
    ...shadowProps,
    ...hiddenMainPass,
  }, sortInfo.item)
}

function shadowMaterialWithSourceShadowState(
  material: ThreeMaterialLike,
  sourceMaterial: ThreeMaterialLike | undefined,
): ThreeMaterialLike {
  if (!sourceMaterialHasShadowState(sourceMaterial)) return material
  const shadowMaterial = Object.create(material) as ThreeMaterialLike
  if (sourceMaterialHasShadowAlphaState(sourceMaterial)) {
    shadowMaterial.map = sourceMaterial.map ?? null
    shadowMaterial.alphaMap = sourceMaterial.alphaMap ?? null
    shadowMaterial.alphaTest = sourceMaterial.alphaToCoverage === true ? 0.5 : sourceMaterial.alphaTest
    shadowMaterial.alphaToCoverage = sourceMaterial.alphaToCoverage
    if (sourceMaterial.alphaHash === true) shadowMaterial.alphaHash = true
    if (sourceMaterial.alphaHash === true || sourceMaterial.alphaToCoverage === true) {
      if (sourceMaterial.opacity != null) {
        shadowMaterial.opacity = sourceMaterial.opacity
      }
    }
  }
  if (sourceMaterialHasShadowDisplacementState(sourceMaterial)) {
    shadowMaterial.displacementMap = sourceMaterial.displacementMap ?? null
    shadowMaterial.displacementScale = sourceMaterial.displacementScale
    shadowMaterial.displacementBias = sourceMaterial.displacementBias
  }
  if (sourceMaterialHasShadowClippingState(sourceMaterial)) {
    shadowMaterial.clipShadows = sourceMaterial.clipShadows
    shadowMaterial.clippingPlanes = sourceMaterial.clippingPlanes
    shadowMaterial.clipIntersection = sourceMaterial.clipIntersection
  }
  if (sourceMaterialHasShadowWireframeState(sourceMaterial)) {
    shadowMaterial.wireframe = sourceMaterial.wireframe
  }
  return shadowMaterial
}

function sourceMaterialHasShadowState(material: ThreeMaterialLike | undefined): material is ThreeMaterialLike {
  return sourceMaterialHasShadowAlphaState(material) ||
    sourceMaterialHasShadowDisplacementState(material) ||
    sourceMaterialHasShadowClippingState(material) ||
    sourceMaterialHasShadowWireframeState(material)
}

function sourceMaterialHasShadowAlphaState(material: ThreeMaterialLike | undefined): material is ThreeMaterialLike {
  if (!material) return false
  const hasAlphaTexture = !!(material.map || material.alphaMap)
  const hasOpacityAlpha = typeof material.opacity === 'number' && Number.isFinite(material.opacity) && material.opacity < 1
  if (material.alphaHash === true && (hasAlphaTexture || hasOpacityAlpha)) return true
  if (material.alphaToCoverage === true && (hasAlphaTexture || hasOpacityAlpha)) return true
  if (!hasAlphaTexture) return false
  if (material.alphaToCoverage === true) return true
  return typeof material.alphaTest === 'number' && Number.isFinite(material.alphaTest) && material.alphaTest > 0
}

function sourceMaterialHasShadowDisplacementState(material: ThreeMaterialLike | undefined): material is ThreeMaterialLike {
  return !!material?.displacementMap
}

function sourceMaterialHasShadowClippingState(material: ThreeMaterialLike | undefined): material is ThreeMaterialLike {
  return !!material && (
    'clipShadows' in material ||
    material.clippingPlanes != null ||
    'clipIntersection' in material
  )
}

function sourceMaterialHasShadowWireframeState(material: ThreeMaterialLike | undefined): material is ThreeMaterialLike {
  return material?.wireframe === true
}

function shadowPbrProperties(
  material: ThreeMaterialLike,
  sourceMaterial: ThreeMaterialLike | undefined,
  materialContext: MaterialExtractionContext,
): ReturnType<typeof extractPbrProperties> {
  const props = extractPbrProperties(material, materialContext)
  const sourceShadowSide = materialShadowSide(sourceMaterial)
  if (sourceShadowSide) {
    props.shadowSide = sourceShadowSide
  }
  return props
}

function validateObjectShadowFlags(object: ThreeObject3DLike): void {
  optionalObjectBoolean(object.castShadow, 'object.castShadow')
  optionalObjectBoolean(object.receiveShadow, 'object.receiveShadow')
}

function optionalObjectBoolean(value: unknown, label: string): boolean | undefined {
  return optionalSceneBoolean(value, label)
}

function optionalSceneBoolean(value: unknown, label: string): boolean | undefined {
  if (value == null) return undefined
  if (typeof value === 'boolean') return value
  throw new TypeError(`${label} must be a boolean.`)
}

function pointVertexColor(
  attribute: ThreeBufferAttributeLike,
  materialColor: Color4,
  pointIndex: number,
  instanceIndex: number,
  label = 'geometry.attributes.color',
): Color4 {
  const sourceIndex = isInstancedAttribute(attribute)
    ? instancedAttributeIndex(attribute, instanceIndex, label)
    : pointIndex
  return [
    clamp01(attributeComponent(attribute, sourceIndex, 0, label) * materialColor[0]),
    clamp01(attributeComponent(attribute, sourceIndex, 1, label) * materialColor[1]),
    clamp01(attributeComponent(attribute, sourceIndex, 2, label) * materialColor[2]),
    clamp01((attribute.itemSize && attribute.itemSize >= 4 ? attributeComponent(attribute, sourceIndex, 3, label) : 1) * materialColor[3]),
  ]
}

function effectiveGroups(
  geometry: ThreeBufferGeometryLike,
  index: number[] | null,
  vertexCount: number,
): GeometryGroup[] {
  const range = geometry.drawRange ?? {}
  if (range != null && (typeof range !== 'object' || Array.isArray(range))) {
    throw new TypeError('geometry.drawRange must be an object.')
  }
  const maxCount = index ? index.length : vertexCount
  const drawStart = clampInteger(geometryDrawRangeStart(range.start), 0, maxCount)
  const requestedCount = geometryDrawRangeCount(range.count, maxCount)
  const drawEnd = clampInteger(drawStart + requestedCount, drawStart, maxCount)
  if (geometry.groups != null && !Array.isArray(geometry.groups)) {
    throw new TypeError('geometry.groups must be an array.')
  }

  const sourceGroups = Array.isArray(geometry.groups) && geometry.groups.length
    ? geometry.groups
    : null
  if (!sourceGroups) {
    return [{ start: drawStart, count: drawEnd - drawStart, materialIndex: 0 }]
  }

  const groups: GeometryGroup[] = []
  for (let groupIndex = 0; groupIndex < sourceGroups.length; groupIndex += 1) {
    const group = sourceGroups[groupIndex]
    if (!group || typeof group !== 'object' || Array.isArray(group)) {
      throw new TypeError(`geometry.groups[${groupIndex}] must be an object.`)
    }
    const groupStart = geometryGroupNonNegativeInteger(group.start, `geometry.groups[${groupIndex}].start`)
    const groupCount = geometryGroupNonNegativeInteger(group.count, `geometry.groups[${groupIndex}].count`)
    const groupMaterialIndex = group.materialIndex == null
      ? 0
      : geometryGroupNonNegativeInteger(group.materialIndex, `geometry.groups[${groupIndex}].materialIndex`)
    const start = Math.max(drawStart, clampInteger(groupStart, 0, maxCount))
    const end = Math.min(drawEnd, clampInteger(groupStart + groupCount, 0, maxCount))
    if (end > start) {
      groups.push({
        start,
        count: end - start,
        materialIndex: groupMaterialIndex,
      })
    }
  }
  return groups
}

function geometryDrawRangeStart(value: unknown): number {
  if (value == null) return 0
  return geometryGroupNonNegativeInteger(value, 'geometry.drawRange.start')
}

function geometryDrawRangeCount(value: unknown, fallback: number): number {
  if (value == null || value === Infinity) return fallback
  return geometryGroupNonNegativeInteger(value, 'geometry.drawRange.count')
}

function geometryGroupNonNegativeInteger(value: unknown, label: string): number {
  if (typeof value === 'number' && Number.isFinite(value) && Number.isInteger(value) && value >= 0) {
    return value
  }
  throw new TypeError(`${label} must be a non-negative integer.`)
}

/**
 * Emit a `NativeSceneMesh` with `topology: 'lines'` or `'points'` for
 * `THREE.Line` / `THREE.LineSegments` / `THREE.LineLoop` / `THREE.Points`.
 * Lines are always expanded to a LineList (pairs of vertex indices) so the
 * Rust side only has to deal with one line topology.
 */
function appendLineOrPoints(
  object: ThreeObject3DLike,
  camera: ThreeCameraLike | undefined,
  meshes: FlattenedMesh[],
  topology: 'lines' | 'points',
  groupOrder: number,
  viewportHeight: number,
  clippingContext: ClippingContext,
  localClippingEnabled: boolean,
  materialContext: MaterialExtractionContext,
  overrideMaterial?: ThreeMaterialLike,
  cache?: SceneExtractionCache,
  callbackContext?: RenderCallbackContext,
): void {
  validateObjectShadowFlags(object)
  const objectCastsShadow = topology === 'lines' && optionalObjectBoolean(object.castShadow, 'object.castShadow') === true
  const geometry = object.geometry!
  const geometryExtraction = meshGeometryExtraction(geometry, cache)
  if (!geometryExtraction) return

  const {
    position,
    positions,
    uvChannels,
    uvs,
    vertexColors,
    index,
    sourceIndex,
    groups,
    instancedGeometryCount,
    instancedPositionOffset,
  } = geometryExtraction
  const vertexCount = position.count
  const indexAttr = index

  for (const group of groups) {
    const material = materialForObjectGroup(object, group.materialIndex, overrideMaterial)
    if (material?.visible === false) continue

    invokeObjectRenderCallback(object.onBeforeRender, 'onBeforeRender', callbackContext, object, camera, geometry, material, group)

    const uvStreams: TextureUvStreams = topology === 'lines'
      ? textureUvStreamsForMapAlphaMaterial(uvChannels, material)
      : { uvs: uvChannels[0], uvs2: secondaryUvsForMaterial(uvChannels, material) }
    let indices: number[] | null = null
    let outputPositions = positions
    let outputUvs: number[] | undefined = topology === 'lines' ? uvStreams.uvs?.values : undefined
    let outputSecondaryUvs: number[] | undefined = topology === 'lines' ? uvStreams.uvs2?.values : undefined
    let outputColors: number[] | undefined
    let thickCenter: [number, number, number] | undefined
    const color = materialColor(material, materialContext)
    const useVertexColors = vertexColors && material?.vertexColors !== false
    const pbrProps = extractPbrProperties(material, materialContext)
    assertSupportedCustomFragmentInstancedAttributes(geometry, pbrProps)
    if (topology === 'lines') {
      pbrProps.alphaMapUsesUv2 = uvStreams.alphaMapUsesUv2
    }
    const textureInfo = extractTextureData(material, materialContext)
    const drawStart = group.start
    const drawEnd = group.start + group.count
    if (topology === 'lines') {
      validateLineMaterialCompatibilityHints(material)
    }
    const lineWidth = positiveMaterialOrObjectNumber(material?.linewidth, 'material.linewidth', 1)
    const thickLine = topology === 'lines' && lineWidth > 1

    if (topology === 'lines') {
      const source = sourceIndex
      if (material?.isLineDashedMaterial === true) {
        const lineDistance = getAttribute(geometry, 'lineDistance')
        const dashed = instancedGeometryCount > 1 || instancedPositionOffset
          ? dashedLineAttributesForInstances(
            positions,
            uvStreams.uvs,
            uvStreams.uvs2,
            useVertexColors ? vertexColors! : undefined,
            color,
            source,
            drawStart,
            drawEnd,
            object,
            lineDistance,
            material,
            instancedGeometryCount,
            instancedPositionOffset,
          )
          : dashedLineAttributesWithCache(
            cache,
            geometry,
            position,
            positions,
            uvStreams.uvs,
            uvStreams.uvs2,
            useVertexColors ? readColorAttribute(vertexColors!, color, 'geometry.attributes.color') : undefined,
            source,
            drawStart,
            drawEnd,
            object,
            lineDistance,
            material,
          )
        if (dashed.positions.length < 6) continue
        outputPositions = dashed.positions
        outputUvs = dashed.uvs
        outputSecondaryUvs = dashed.uvs2
        outputColors = dashed.colors
        indices = rangeIndices(dashed.positions.length / 3)
        if (thickLine) {
          const transform = matrixElements(object.matrixWorld!, 'object.matrixWorld')
          const thick = thickLineAttributes(
            outputPositions,
            outputUvs,
            outputSecondaryUvs,
            outputColors,
            indices,
            transform,
            camera,
            viewportHeight,
            lineWidth,
          )
          if (thick.positions.length < 12) continue
          outputPositions = thick.positions
          outputUvs = thick.uvs
          outputSecondaryUvs = thick.uvs2
          outputColors = thick.colors
          indices = thick.indices
          thickCenter = thick.center
        } else {
          indices = null
        }
      } else {
        indices = expandLineIndices(source, drawStart, drawEnd, object)
        if (indices.length < 2) continue
        if (instancedGeometryCount > 1 || instancedPositionOffset) {
          outputPositions = expandVec3ValuesForInstances(positions, 0, vertexCount, instancedGeometryCount, instancedPositionOffset)
          outputUvs = uvStreams.uvs ? expandUvChannelForInstances(uvStreams.uvs, 0, vertexCount, instancedGeometryCount) : undefined
          outputSecondaryUvs = uvStreams.uvs2 ? expandUvChannelForInstances(uvStreams.uvs2, 0, vertexCount, instancedGeometryCount) : undefined
          indices = expandIndicesForInstances(indices, vertexCount, instancedGeometryCount)
        }
        if (thickLine) {
          const transform = matrixElements(object.matrixWorld!, 'object.matrixWorld')
          const thick = thickLineAttributes(
            outputPositions,
            outputUvs,
            outputSecondaryUvs,
            useVertexColors ? outputColors ?? expandColorAttributeForInstances(vertexColors!, color, 0, vertexCount, instancedGeometryCount) : undefined,
            indices,
            transform,
            camera,
            viewportHeight,
            lineWidth,
          )
          if (thick.positions.length < 12) continue
          outputPositions = thick.positions
          outputUvs = thick.uvs
          outputSecondaryUvs = thick.uvs2
          outputColors = thick.colors
          indices = thick.indices
          thickCenter = thick.center
        }
      }
    } else if (indexAttr) {
      indices = indexAttr.slice(drawStart, drawEnd)
      if (indices.length === 0) continue
    }

    if (useVertexColors && material?.isLineDashedMaterial !== true && !thickLine) {
      outputColors = expandColorAttributeForInstances(vertexColors!, color, 0, vertexCount, instancedGeometryCount)
    }
    const sortInfo = sortInfoForObject(object, material, camera, meshes.length, groupOrder, undefined, geometry, group)
    if (thickCenter && camera) {
      sortInfo.keys.sortZ = projectedWorldPointZ(thickCenter, camera)
      sortInfo.item.z = sortInfo.keys.sortZ
    }
    const clipping = clippingState(clippingContext, material, localClippingEnabled)

    pushMesh(meshes, {
      positions: outputPositions,
      indices: indices ?? undefined,
      uvs: topology === 'lines' ? outputUvs : undefined,
      uvs2: topology === 'lines' ? outputSecondaryUvs : undefined,
      color,
      colors: outputColors,
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
      textureUsesUv2: topology === 'lines' ? uvStreams.textureUsesUv2 : textureInfo?.usesUv2,
      transform: thickLine ? IDENTITY_4X4.slice() : matrixElements(object.matrixWorld!, 'object.matrixWorld'),
      transparent: material?.transparent === true || (material?.opacity != null && material.opacity < 1),
      alphaTest: material && Number.isFinite(material.alphaTest) && material.alphaTest! > 0 ? material.alphaTest : undefined,
      clipShadows: clipShadowsForMaterial(material, clippingContext),
      ...pbrProps,
      ...(thickLine ? { side: 'double' } : {}),
      shadingModel: 'basic',
      topology: thickLine ? 'triangles' : topology,
      castShadow: objectCastsShadow ? true : undefined,
      ...clipping,
      ...sortInfo.keys,
    }, sortInfo.item)
    invokeObjectRenderCallback(object.onAfterRender, 'onAfterRender', callbackContext, object, camera, geometry, material, group)
  }
}

function thickLineAttributes(
  positions: number[],
  uvs: number[] | undefined,
  uvs2: number[] | undefined,
  colors: number[] | undefined,
  lineIndices: number[],
  transform: ArrayLike<number>,
  camera: ThreeCameraLike | undefined,
  viewportHeight: number,
  lineWidth: number,
): ThickLineExpansion {
  const axes = cameraBillboardAxes(camera)
  const outputPositions: number[] = []
  const outputUvs: number[] | undefined = uvs ? [] : undefined
  const outputUvs2: number[] | undefined = uvs2 ? [] : undefined
  const outputColors: number[] | undefined = colors ? [] : undefined
  const outputIndices: number[] = []

  for (let i = 0; i + 1 < lineIndices.length; i += 2) {
    const aIndex = lineIndices[i]
    const bIndex = lineIndices[i + 1]
    if (!validPositionIndex(positions, aIndex) || !validPositionIndex(positions, bIndex)) continue

    const a = transformPoint(transform, [
      positions[aIndex * 3],
      positions[aIndex * 3 + 1],
      positions[aIndex * 3 + 2],
    ])
    const b = transformPoint(transform, [
      positions[bIndex * 3],
      positions[bIndex * 3 + 1],
      positions[bIndex * 3 + 2],
    ])
    const dx = b[0] - a[0]
    const dy = b[1] - a[1]
    const dz = b[2] - a[2]
    if (Math.hypot(dx, dy, dz) <= 1e-8) continue

    const screenDx = dx * axes.right[0] + dy * axes.right[1] + dz * axes.right[2]
    const screenDy = dx * axes.up[0] + dy * axes.up[1] + dz * axes.up[2]
    const side = normalizeVec3([
      -screenDy * axes.right[0] + screenDx * axes.up[0],
      -screenDy * axes.right[1] + screenDx * axes.up[1],
      -screenDy * axes.right[2] + screenDx * axes.up[2],
    ], axes.up)
    const midpoint: [number, number, number] = [
      (a[0] + b[0]) * 0.5,
      (a[1] + b[1]) * 0.5,
      (a[2] + b[2]) * 0.5,
    ]
    const halfWidth = linePixelWorldSize(lineWidth, midpoint, camera, viewportHeight) * 0.5
    if (halfWidth <= 0) continue

    const vertexBase = outputPositions.length / 3
    pushThickLineVertex(outputPositions, a, side, -halfWidth)
    pushThickLineVertex(outputPositions, a, side, halfWidth)
    pushThickLineVertex(outputPositions, b, side, halfWidth)
    pushThickLineVertex(outputPositions, b, side, -halfWidth)
    outputIndices.push(vertexBase, vertexBase + 1, vertexBase + 2, vertexBase, vertexBase + 2, vertexBase + 3)

    pushRepeatedVec2(outputUvs, uvs, aIndex)
    pushRepeatedVec2(outputUvs, uvs, aIndex)
    pushRepeatedVec2(outputUvs, uvs, bIndex)
    pushRepeatedVec2(outputUvs, uvs, bIndex)
    pushRepeatedVec2(outputUvs2, uvs2, aIndex)
    pushRepeatedVec2(outputUvs2, uvs2, aIndex)
    pushRepeatedVec2(outputUvs2, uvs2, bIndex)
    pushRepeatedVec2(outputUvs2, uvs2, bIndex)
    pushRepeatedColor(outputColors, colors, aIndex)
    pushRepeatedColor(outputColors, colors, aIndex)
    pushRepeatedColor(outputColors, colors, bIndex)
    pushRepeatedColor(outputColors, colors, bIndex)
  }

  return {
    center: thickLineCenter(outputPositions),
    colors: outputColors,
    indices: outputIndices,
    positions: outputPositions,
    uvs: outputUvs,
    uvs2: outputUvs2,
  }
}

function validPositionIndex(positions: number[], index: number): boolean {
  return Number.isInteger(index) && index >= 0 && index * 3 + 2 < positions.length
}

function pushThickLineVertex(
  positions: number[],
  point: [number, number, number],
  side: [number, number, number],
  offset: number,
): void {
  positions.push(
    point[0] + side[0] * offset,
    point[1] + side[1] * offset,
    point[2] + side[2] * offset,
  )
}

function pushRepeatedVec2(target: number[] | undefined, source: number[] | undefined, index: number): void {
  if (!target || !source || index * 2 + 1 >= source.length) return
  target.push(source[index * 2], source[index * 2 + 1])
}

function pushRepeatedColor(target: number[] | undefined, source: number[] | undefined, index: number): void {
  if (!target || !source || index * 4 + 3 >= source.length) return
  target.push(source[index * 4], source[index * 4 + 1], source[index * 4 + 2], source[index * 4 + 3])
}

function linePixelWorldSize(
  lineWidth: number,
  worldPosition: [number, number, number],
  camera: ThreeCameraLike | undefined,
  viewportHeight: number,
): number {
  const projectionY = Math.abs(finiteOrDefault(camera?.projectionMatrix?.elements?.[5], 1))
  if (projectionY <= 0) return 0

  if (camera?.isPerspectiveCamera === true) {
    const viewZ = viewSpaceZ(worldPosition, camera)
    const depth = Number.isFinite(viewZ) ? Math.max(0.0001, Math.abs(viewZ)) : 1
    return lineWidth * 2 * depth / Math.max(1, viewportHeight) / projectionY
  }

  return lineWidth * 2 / Math.max(1, viewportHeight) / projectionY
}

function thickLineCenter(positions: number[]): [number, number, number] {
  if (positions.length < 3) return [0, 0, 0]
  let x = 0
  let y = 0
  let z = 0
  let count = 0
  for (let i = 0; i + 2 < positions.length; i += 3) {
    x += positions[i]
    y += positions[i + 1]
    z += positions[i + 2]
    count += 1
  }
  return count > 0 ? [x / count, y / count, z / count] : [0, 0, 0]
}

function clippingState(
  clippingContext: ClippingContext,
  material: ThreeMaterialLike | undefined,
  localClippingEnabled: boolean,
): Pick<NativeSceneMesh, 'clippingPlanes' | 'clippingUnionCount'> {
  const inheritedUnionPlanes = clippingContext.unionPlanes.slice(0, MAX_CLIPPING_PLANES)
  const inheritedIntersectionPlanes = clippingContext.intersectionPlanes.slice(
    0,
    Math.max(0, MAX_CLIPPING_PLANES - inheritedUnionPlanes.length),
  )
  const localBudget = Math.max(0, MAX_CLIPPING_PLANES - inheritedUnionPlanes.length - inheritedIntersectionPlanes.length)
  const localPlanes = localClippingEnabled
    ? extractClippingPlanes(material?.clippingPlanes, 'material.clippingPlanes', localBudget)
    : []
  const clipIntersection = optionalObjectBoolean(material?.clipIntersection, 'material.clipIntersection') === true
  const localUnionPlanes = clipIntersection ? [] : localPlanes
  const localIntersectionPlanes = clipIntersection ? localPlanes : []
  const unionPlanes = [...inheritedUnionPlanes, ...localUnionPlanes]
  const intersectionPlanes = [...inheritedIntersectionPlanes, ...localIntersectionPlanes]
  const planes = [...unionPlanes, ...intersectionPlanes]
  if (planes.length === 0) return {}

  return {
    clippingPlanes: flattenClippingPlanes(planes),
    clippingUnionCount: unionPlanes.length,
  }
}

function clippingContextForObject(parent: ClippingContext, object: ThreeObject3DLike): ClippingContext {
  if (object.isClippingGroup !== true) return parent
  if (optionalObjectBoolean(object.enabled, 'ClippingGroup.enabled') === false) return parent
  const clipIntersection = optionalObjectBoolean(object.clipIntersection, 'ClippingGroup.clipIntersection') === true
  const clipShadows = parent.clipShadows || optionalObjectBoolean(object.clipShadows, 'ClippingGroup.clipShadows') === true
  const currentCount = parent.unionPlanes.length + parent.intersectionPlanes.length
  const remainingBudget = Math.max(0, MAX_CLIPPING_PLANES - currentCount)
  const planes = extractClippingPlanes(object.clippingPlanes, 'ClippingGroup.clippingPlanes', remainingBudget)
  if (planes.length === 0) return parent

  return clipIntersection
    ? {
      unionPlanes: parent.unionPlanes,
      intersectionPlanes: [...parent.intersectionPlanes, ...planes],
      clipShadows,
    }
    : {
      unionPlanes: [...parent.unionPlanes, ...planes],
      intersectionPlanes: parent.intersectionPlanes,
      clipShadows,
    }
}

function clipShadowsForMaterial(material: ThreeMaterialLike | undefined, clippingContext: ClippingContext): boolean | undefined {
  return optionalObjectBoolean(material?.clipShadows, 'material.clipShadows') === true || clippingContext.clipShadows ? true : undefined
}

function pushMesh(meshes: FlattenedMesh[], mesh: NativeSceneMesh, sortItem: RenderSortItem): void {
  meshes.push({
    mesh,
    sortItem,
    groupOrder: mesh.groupOrder ?? 0,
    renderOrder: mesh.renderOrder ?? 0,
    sortZ: mesh.sortZ ?? 0,
    materialSortKey: mesh.materialSortKey ?? 0,
    materialVariant: mesh.materialVariant ?? 0,
    sortIndex: mesh.sortIndex ?? meshes.length,
  })
}

function sortInfoForObject(
  object: ThreeObject3DLike,
  material: ThreeMaterialLike | undefined,
  camera: ThreeCameraLike | undefined,
  sortIndex: number,
  groupOrder: number,
  transform?: number[],
  geometry?: ThreeBufferGeometryLike,
  group?: GeometryGroup,
  sortItemObject?: ThreeObject3DLike,
  sortItemZOverride?: number,
): MeshSortInfo {
  const renderOrder = finiteMaterialOrObjectNumber(object.renderOrder, 'object.renderOrder', 0)
  const z = camera ? projectedObjectZ(object, camera, transform) : 0
  const itemZ = sortItemZOverride ?? z
  const id = unsignedSortKey(object.id, sortIndex)
  const materialSortKey = finiteOrDefault(material?.id, 0)
  const materialVariant = materialVariantForObject(object)
  return {
    keys: {
      groupOrder,
      renderOrder,
      sortZ: z,
      sortIndex: id,
      materialSortKey,
      materialVariant,
    },
    item: {
      id,
      object: sortItemObject ?? object,
      geometry,
      material,
      group,
      groupOrder,
      renderOrder,
      z: itemZ,
      materialVariant,
    },
  }
}

function mergeSortKeys(keys: MeshSortInfo['keys'], override: SortKeyOverride | undefined): MeshSortInfo['keys'] {
  return override ? { ...keys, ...override } : keys
}

function sortFlattenedMeshes(meshes: FlattenedMesh[], options: SceneSortOptions): FlattenedMesh[] {
  const sortObjects = options.sortObjects !== false
  const buckets = partitionFlattenedMeshes(meshes)
  if (options.opaque === false) buckets.opaque.length = 0
  if (options.transparent === false) {
    buckets.transmissive.length = 0
    buckets.transparent.length = 0
  }

  if (!sortObjects) {
    normalizeSortKeys(buckets.opaque)
    normalizeSortKeys(buckets.transmissive)
    normalizeSortKeys(buckets.transparent)
    return [...buckets.opaque, ...buckets.transmissive, ...buckets.transparent]
  }

  if (options.opaqueSort) {
    buckets.opaque.sort(compareWithSort(options.opaqueSort))
    normalizeSortKeys(buckets.opaque)
  } else {
    buckets.opaque.sort(compareFlattenedMeshes)
  }

  if (options.transparentSort) {
    const transparentSort = compareWithSort(options.transparentSort)
    buckets.transmissive.sort(transparentSort)
    buckets.transparent.sort(transparentSort)
    normalizeSortKeys(buckets.transmissive)
    normalizeSortKeys(buckets.transparent)
  } else {
    buckets.transmissive.sort(compareTransparentFlattenedMeshes)
    buckets.transparent.sort(compareTransparentFlattenedMeshes)
  }

  return [...buckets.opaque, ...buckets.transmissive, ...buckets.transparent]
}

function partitionFlattenedMeshes(meshes: FlattenedMesh[]): {
  opaque: FlattenedMesh[]
  transmissive: FlattenedMesh[]
  transparent: FlattenedMesh[]
} {
  const opaque: FlattenedMesh[] = []
  const transmissive: FlattenedMesh[] = []
  const transparent: FlattenedMesh[] = []

  for (const mesh of meshes) {
    if (finitePositive(mesh.mesh.transmission)) {
      transmissive.push(mesh)
    } else if (meshDefaultsTransparent(mesh.mesh)) {
      transparent.push(mesh)
    } else {
      opaque.push(mesh)
    }
  }

  return { opaque, transmissive, transparent }
}

function compareWithSort(sort: RenderSortFunction): (a: FlattenedMesh, b: FlattenedMesh) => number {
  return (a, b) => {
    const result = Number(sort(a.sortItem, b.sortItem))
    return Number.isFinite(result) ? result : 0
  }
}

function normalizeSortKeys(meshes: FlattenedMesh[]): void {
  meshes.forEach((entry, index) => {
    entry.mesh.groupOrder = 0
    entry.mesh.renderOrder = 0
    entry.mesh.materialSortKey = 0
    entry.mesh.materialVariant = 0
    entry.mesh.sortZ = 0
    entry.mesh.sortIndex = index
  })
}

function compareFlattenedMeshes(a: FlattenedMesh, b: FlattenedMesh): number {
  return a.groupOrder - b.groupOrder
    || a.renderOrder - b.renderOrder
    || a.materialSortKey - b.materialSortKey
    || a.materialVariant - b.materialVariant
    || a.sortZ - b.sortZ
    || a.sortIndex - b.sortIndex
}

function compareTransparentFlattenedMeshes(a: FlattenedMesh, b: FlattenedMesh): number {
  return a.groupOrder - b.groupOrder
    || a.renderOrder - b.renderOrder
    || b.sortZ - a.sortZ
    || a.sortIndex - b.sortIndex
}

function meshDefaultsTransparent(mesh: NativeSceneMesh): boolean {
  if (mesh.alphaHash === true) return false
  if (mesh.transparent === true) return true
  if (mesh.transparent === false) return false
  return materialAlpha(mesh) < 0.999
}

function materialAlpha(mesh: NativeSceneMesh): number {
  return mesh.color && mesh.color.length >= 4 ? finiteOrDefault(mesh.color[3], 1) : 1
}

function finitePositive(value: unknown): boolean {
  return typeof value === 'number' && Number.isFinite(value) && value > 0.0001
}

function materialVariantForObject(object: ThreeObject3DLike): number {
  return (object.isInstancedMesh === true ? 2 : 0) + (object.isSkinnedMesh === true ? 1 : 0)
}

function projectedObjectZ(object: ThreeObject3DLike, camera: ThreeCameraLike, transform?: ArrayLike<number>): number {
  const world = transform ?? object.matrixWorld?.elements
  if (!world || world.length < 16) return 0
  const view = camera.matrixWorldInverse?.elements
  const projection = camera.projectionMatrix?.elements
  if (!view || view.length < 16 || !projection || projection.length < 16) return 0

  const center = objectSortCenter(object)
  const x = world[0] * center[0] + world[4] * center[1] + world[8] * center[2] + world[12]
  const y = world[1] * center[0] + world[5] * center[1] + world[9] * center[2] + world[13]
  const z = world[2] * center[0] + world[6] * center[1] + world[10] * center[2] + world[14]
  const vx = view[0] * x + view[4] * y + view[8] * z + view[12]
  const vy = view[1] * x + view[5] * y + view[9] * z + view[13]
  const vz = view[2] * x + view[6] * y + view[10] * z + view[14]
  const vw = view[3] * x + view[7] * y + view[11] * z + view[15]
  const clipZ = projection[2] * vx + projection[6] * vy + projection[10] * vz + projection[14] * vw
  const clipW = projection[3] * vx + projection[7] * vy + projection[11] * vz + projection[15] * vw
  return clipW === 0 ? clipZ : clipZ / clipW
}

function projectedWorldPointZ(worldPoint: [number, number, number], camera: ThreeCameraLike): number {
  const view = camera.matrixWorldInverse?.elements
  const projection = camera.projectionMatrix?.elements
  if (!view || view.length < 16 || !projection || projection.length < 16) return 0

  const x = worldPoint[0]
  const y = worldPoint[1]
  const z = worldPoint[2]
  const vx = view[0] * x + view[4] * y + view[8] * z + view[12]
  const vy = view[1] * x + view[5] * y + view[9] * z + view[13]
  const vz = view[2] * x + view[6] * y + view[10] * z + view[14]
  const vw = view[3] * x + view[7] * y + view[11] * z + view[15]
  const clipZ = projection[2] * vx + projection[6] * vy + projection[10] * vz + projection[14] * vw
  const clipW = projection[3] * vx + projection[7] * vy + projection[11] * vz + projection[15] * vw
  return clipW === 0 ? clipZ : clipZ / clipW
}

function objectSortCenter(object: ThreeObject3DLike): [number, number, number] {
  if (object.boundingSphere !== undefined) {
    if (object.boundingSphere == null && typeof object.computeBoundingSphere === 'function') {
      try {
        object.computeBoundingSphere()
      } catch {
        return [0, 0, 0]
      }
    }

    if (object.boundingSphere == null) return [0, 0, 0]
    if (!object.boundingSphere || typeof object.boundingSphere !== 'object') {
      throw new TypeError('object.boundingSphere must be a THREE.Sphere-like object.')
    }
    return requiredVec3Like(object.boundingSphere.center, 'object.boundingSphere.center')
  }

  const geometry = object.geometry
  if (!geometry) return [0, 0, 0]

  if (geometry.boundingSphere == null && typeof geometry.computeBoundingSphere === 'function') {
    try {
      geometry.computeBoundingSphere()
    } catch {
      return [0, 0, 0]
    }
  }

  const sphere = geometry.boundingSphere
  if (sphere == null) return [0, 0, 0]
  if (!sphere || typeof sphere !== 'object') {
    throw new TypeError('geometry.boundingSphere must be a THREE.Sphere-like object.')
  }
  return requiredVec3Like((sphere as { center?: { x?: number; y?: number; z?: number } | ArrayLike<number> }).center, 'geometry.boundingSphere.center')
}

function renderableObjectOutsideFrustum(
  object: ThreeObject3DLike,
  camera: ThreeCameraLike | undefined,
  viewportHeight = 512,
  overrideMaterial?: ThreeMaterialLike,
): boolean {
  if (!camera) return false
  const frustumCulled = optionalObjectBoolean(object.frustumCulled, 'object.frustumCulled')
  if (frustumCulled === false) return false

  const sphere = objectBoundingSphere(object)
  if (!sphere) return false
  const transform = matrixElements(object.matrixWorld!, renderableMatrixWorldLabel(object))
  const extraRadius = object.isPoints === true
    ? pointBillboardCullRadius(object, camera, transform, sphere, viewportHeight, overrideMaterial)
    : 0
  return transformedSphereOutsideFrustum(camera, transform, sphere, extraRadius)
}

function renderableMatrixWorldLabel(object: ThreeObject3DLike): string {
  if (object.isBatchedMesh === true) return 'batchedMesh.matrixWorld'
  if (object.isMesh === true) return 'mesh.matrixWorld'
  if (object.isPoints === true) return 'points.matrixWorld'
  return 'object.matrixWorld'
}

function objectBoundingSphere(object: ThreeObject3DLike): { center: [number, number, number]; radius: number } | null {
  if (object.isBatchedMesh === true && object.boundingSphere == null) {
    return null
  }

  if (object.boundingSphere !== undefined) {
    if (object.boundingSphere == null && typeof object.computeBoundingSphere === 'function') {
      object.computeBoundingSphere()
    }
    return object.boundingSphere == null
      ? null
      : sphereLike(object.boundingSphere, 'object.boundingSphere')
  }

  const geometry = object.geometry
  if (!geometry) return null
  if (geometry.boundingSphere == null && typeof geometry.computeBoundingSphere === 'function') {
    geometry.computeBoundingSphere()
  }
  return geometry.boundingSphere == null
    ? null
    : sphereLike(geometry.boundingSphere, 'geometry.boundingSphere')
}

function pointBillboardCullRadius(
  object: ThreeObject3DLike,
  camera: ThreeCameraLike,
  transform: ArrayLike<number>,
  sphere: { center: [number, number, number]; radius: number },
  viewportHeight: number,
  overrideMaterial: ThreeMaterialLike | undefined,
): number {
  const center = transformPoint(transform, sphere.center)
  let radius = 0
  for (const material of pointCullMaterials(object, overrideMaterial)) {
    const pointSize = safePositiveNumber(material?.size, 1)
    const sizeAttenuation = typeof material?.sizeAttenuation === 'boolean'
      ? material.sizeAttenuation
      : undefined
    const worldSize = pointWorldSize(pointSize, center, sizeAttenuation, camera, viewportHeight)
    radius = Math.max(radius, worldSize * Math.SQRT1_2)
  }
  return radius
}

function pointCullMaterials(
  object: ThreeObject3DLike,
  overrideMaterial: ThreeMaterialLike | undefined,
): Array<ThreeMaterialLike | undefined> {
  if (overrideMaterial !== undefined) return [overrideMaterial]
  if (Array.isArray(object.material)) {
    return object.material.filter((material): material is ThreeMaterialLike => material != null && typeof material === 'object' && !Array.isArray(material))
  }
  return object.material != null && typeof object.material === 'object' && !Array.isArray(object.material)
    ? [object.material]
    : [undefined]
}

function spriteOutsideFrustum(
  object: ThreeObject3DLike,
  camera: ThreeCameraLike | undefined,
  transform: ArrayLike<number>,
  center: [number, number],
): boolean {
  if (!camera) return false
  const frustumCulled = optionalObjectBoolean(object.frustumCulled, 'object.frustumCulled')
  if (frustumCulled === false) return false

  const offset = Math.hypot(center[0] - 0.5, center[1] - 0.5)
  return transformedSphereOutsideFrustum(camera, transform, {
    center: [0, 0, 0],
    radius: 0.7071067811865476 + offset,
  })
}

function transformedSphereOutsideFrustum(
  camera: ThreeCameraLike,
  transform: ArrayLike<number>,
  sphere: { center: [number, number, number]; radius: number },
  extraRadius = 0,
): boolean {
  const center = transformPoint(transform, sphere.center)
  const scale = Math.max(columnLength3(transform, 0), columnLength3(transform, 4), columnLength3(transform, 8))
  const radius = sphere.radius * scale + extraRadius
  if (!Number.isFinite(radius) || radius < 0) return false
  return !cameraFrustumIntersectsSphere(camera, center, radius)
}

function vec3Like(value: { x?: number; y?: number; z?: number } | ArrayLike<number> | undefined): [number, number, number] | null {
  if (!value) return null
  const objectValue = value as { x?: unknown; y?: unknown; z?: unknown }
  const x = typeof objectValue.x === 'number' ? objectValue.x : (value as ArrayLike<unknown>)[0]
  const y = typeof objectValue.y === 'number' ? objectValue.y : (value as ArrayLike<unknown>)[1]
  const z = typeof objectValue.z === 'number' ? objectValue.z : (value as ArrayLike<unknown>)[2]
  return typeof x === 'number' && Number.isFinite(x)
    && typeof y === 'number' && Number.isFinite(y)
    && typeof z === 'number' && Number.isFinite(z)
    ? [x, y, z]
    : null
}

function requiredVec3Like(value: { x?: number; y?: number; z?: number } | ArrayLike<number> | undefined, label: string): [number, number, number] {
  const vector = vec3Like(value)
  if (vector) return vector
  throw new TypeError(`${label} must be a finite Vector3-like value.`)
}

function columnLength3(matrix: ArrayLike<number>, start: number): number {
  const x = matrix[start]
  const y = matrix[start + 1]
  const z = matrix[start + 2]
  return Math.hypot(x, y, z)
}

function cameraBillboardAxes(camera: ThreeCameraLike | undefined): { right: [number, number, number]; up: [number, number, number] } {
  const matrix = camera?.matrixWorld?.elements
  if (!matrix || matrix.length < 16) {
    return { right: [1, 0, 0], up: [0, 1, 0] }
  }
  return {
    right: normalizeVec3([matrix[0], matrix[1], matrix[2]], [1, 0, 0]),
    up: normalizeVec3([matrix[4], matrix[5], matrix[6]], [0, 1, 0]),
  }
}

function normalizeVec3(value: [number, number, number], fallback: [number, number, number]): [number, number, number] {
  const length = Math.hypot(value[0], value[1], value[2])
  if (!Number.isFinite(length) || length <= 1e-8) return fallback
  return [value[0] / length, value[1] / length, value[2] / length]
}

function viewSpaceZ(worldPosition: number[], camera: ThreeCameraLike): number {
  const view = camera.matrixWorldInverse?.elements
  if (!view || view.length < 16) return Number.NaN
  return view[2] * worldPosition[0] + view[6] * worldPosition[1] + view[10] * worldPosition[2] + view[14]
}

function transformPoint(matrix: ArrayLike<number>, point: [number, number, number]): [number, number, number] {
  const x = point[0]
  const y = point[1]
  const z = point[2]
  return [
    matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12],
    matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13],
    matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14],
  ]
}

function pointWorldSize(
  pointSize: number,
  worldPosition: [number, number, number],
  sizeAttenuation: boolean | undefined,
  camera: ThreeCameraLike | undefined,
  viewportHeight: number,
): number {
  const projectionY = Math.abs(finiteOrDefault(camera?.projectionMatrix?.elements?.[5], 1))
  if (projectionY <= 0) return 0

  if (camera?.isPerspectiveCamera === true && sizeAttenuation !== false) {
    const viewZ = viewSpaceZ(worldPosition, camera)
    const depth = Number.isFinite(viewZ) ? Math.max(0.0001, Math.abs(viewZ)) : 1
    const pixelSize = clampPointSpriteSize(pointSize * Math.max(1, viewportHeight) / (2 * depth))
    return pixelSize * 2 * depth / Math.max(1, viewportHeight) / projectionY
  }

  const cappedPointSize = clampPointSpriteSize(pointSize)
  if (camera?.isPerspectiveCamera !== true) {
    return cappedPointSize * 2 / Math.max(1, viewportHeight) / projectionY
  }

  const viewZ = camera ? viewSpaceZ(worldPosition, camera) : -1
  const depth = Number.isFinite(viewZ) ? Math.max(0.0001, Math.abs(viewZ)) : 1
  return cappedPointSize * 2 * depth / Math.max(1, viewportHeight) / projectionY
}

function clampPointSpriteSize(pixelSize: number): number {
  return Math.min(pixelSize, MAX_POINT_SPRITE_SIZE)
}

function finiteOrDefault(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function safePositiveNumber(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : fallback
}

function finiteMaterialOrObjectNumber(value: unknown, label: string, fallback: number): number {
  if (value == null) return fallback
  if (typeof value === 'number' && Number.isFinite(value)) return value
  throw new TypeError(`${label} must be a finite number.`)
}

function nonNegativeMaterialOrObjectNumber(value: unknown, label: string, fallback: number): number {
  const number = finiteMaterialOrObjectNumber(value, label, fallback)
  if (number < 0) {
    throw new TypeError(`${label} must be non-negative.`)
  }
  return number
}

function positiveMaterialOrObjectNumber(value: unknown, label: string, fallback: number): number {
  const number = finiteMaterialOrObjectNumber(value, label, fallback)
  if (number <= 0) {
    throw new TypeError(`${label} must be positive.`)
  }
  return number
}

function validateLineMaterialCompatibilityHints(material: ThreeMaterialLike | undefined): void {
  if (!material) return
  optionalLineCap(material.linecap)
  optionalLineJoin(material.linejoin)
}

function optionalLineCap(value: unknown): void {
  if (value == null) return
  if (typeof value !== 'string') {
    throw new TypeError('material.linecap must be a string.')
  }
  if (value !== 'butt' && value !== 'round' && value !== 'square') {
    throw new Error(
      `material.linecap ${JSON.stringify(value)} is not supported by @headless-three/renderer. Use "butt", "round", "square", null, or undefined.`,
    )
  }
}

function optionalLineJoin(value: unknown): void {
  if (value == null) return
  if (typeof value !== 'string') {
    throw new TypeError('material.linejoin must be a string.')
  }
  if (value !== 'round' && value !== 'bevel' && value !== 'miter') {
    throw new Error(
      `material.linejoin ${JSON.stringify(value)} is not supported by @headless-three/renderer. Use "round", "bevel", "miter", null, or undefined.`,
    )
  }
}

function normalizedMaterialOrObjectNumber(value: unknown, label: string, fallback: number): number {
  const number = finiteMaterialOrObjectNumber(value, label, fallback)
  if (number < 0 || number > 1) {
    throw new TypeError(`${label} must be between 0 and 1.`)
  }
  return number
}

function cameraZoomOrDefault(value: unknown): number {
  if (value == null) return 1
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError('camera.zoom must be a finite number.')
  }
  if (value <= 0) {
    throw new TypeError('camera.zoom must be positive.')
  }
  return value
}

function finiteCountOrDefault(value: unknown, label: string, fallback: number, allowInfinity = false): number {
  if (value == null) return fallback
  if (allowInfinity && value === Infinity) return value
  if (typeof value === 'number' && Number.isFinite(value)) return value
  throw new TypeError(`${label} must be a finite number.`)
}

function integerCountOrDefault(value: unknown, label: string, fallback: number, allowInfinity = false): number {
  const number = finiteCountOrDefault(value, label, fallback, allowInfinity)
  if (number === Infinity) return number
  if (!Number.isInteger(number)) {
    throw new TypeError(`${label} must be an integer.`)
  }
  if (number < 0) {
    throw new TypeError(`${label} must be non-negative.`)
  }
  return number
}

function unsignedSortKey(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? value : fallback
}

function instancedBufferGeometryCount(geometry: ThreeBufferGeometryLike): number {
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

function assertSupportedCustomFragmentInstancedAttributes(
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

function isInstancedAttribute(attribute: ThreeBufferAttributeLike | undefined | null): attribute is ThreeBufferAttributeLike {
  return attribute?.isInstancedBufferAttribute === true
}

function meshPerAttribute(attribute: ThreeBufferAttributeLike, label = 'InstancedBufferAttribute.meshPerAttribute'): number {
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

function instancedAttributeIndex(
  attribute: ThreeBufferAttributeLike,
  instanceIndex: number,
  label = 'InstancedBufferAttribute',
): number {
  return Math.min(
    attributeCount(attribute, label) - 1,
    Math.floor(instanceIndex / meshPerAttribute(attribute, `${label}.meshPerAttribute`)),
  )
}

function instancedOffsetAttribute(geometry: ThreeBufferGeometryLike): InstancedAttributeRef | null {
  const names = ['instanceOffset', 'instancePosition', 'offset', 'translate', 'translation']
  for (const name of names) {
    const attribute = getAttribute(geometry, name)
    if (isInstancedAttribute(attribute)) return { attribute, label: `geometry.attributes.${name}` }
  }
  return null
}

function expandVec3ValuesForInstances(
  values: number[],
  start: number,
  count: number,
  instanceCount: number,
  offsetAttribute?: InstancedAttributeRef | null,
): number[] {
  if (instanceCount <= 1 && !offsetAttribute) {
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
    for (let vertex = start; vertex < start + count; vertex += 1) {
      out[dst++] = values[vertex * 3] + ox
      out[dst++] = values[vertex * 3 + 1] + oy
      out[dst++] = values[vertex * 3 + 2] + oz
    }
  }
  return out
}

function expandVec2ValuesForInstances(values: number[], start: number, count: number, instanceCount: number): number[] {
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

function expandNormalValuesForInstances(
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

function expandUvChannelForInstances(channel: UvChannel, start: number, count: number, instanceCount: number): number[] {
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

function appendUvForVertex(
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

function uvValuesForInstance(channel: UvChannel | null, vertexCount: number, instanceIndex: number): number[] | null {
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

function labelForMeshPerAttribute(channel: UvChannel): string {
  return `${channel.label}.meshPerAttribute`
}

function expandColorAttributeForInstances(
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

function expandIndicesForInstances(indices: number[], vertexCount: number, instanceCount: number): number[] {
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

function wireframeIndicesForTriangles(indices: number[]): number[] {
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

function wireframeIndicesForUnindexedTriangles(vertexCount: number): number[] {
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

function isDepthDistanceWireframeMaterial(material: ThreeMaterialLike | undefined): boolean {
  return material?.wireframe === true
    && (material.isMeshDepthMaterial === true || material.isMeshDistanceMaterial === true)
}

function isMeshWireframeMaterial(material: ThreeMaterialLike | undefined): boolean {
  return material?.wireframe === true
}

function readUvChannels(geometry: ThreeBufferGeometryLike): Array<UvChannel | null> {
  const primaryUvs = readOptionalUvAttribute(geometry, 'uv')
  return [
    primaryUvs,
    readOptionalUvAttribute(geometry, 'uv1') ?? readOptionalUvAttribute(geometry, 'uv2') ?? primaryUvs,
    readOptionalUvAttribute(geometry, 'uv2') ?? readOptionalUvAttribute(geometry, 'uv1') ?? primaryUvs,
    readOptionalUvAttribute(geometry, 'uv3') ?? primaryUvs,
  ]
}

function readOptionalUvAttribute(geometry: ThreeBufferGeometryLike, name: string): UvChannel | null {
  const attribute = getAttribute(geometry, name)
  if (!attribute) return null
  const label = `geometry.attributes.${name}`
  return {
    attribute,
    label,
    values: readVec2Attribute(attribute, label),
  }
}

function textureUvStreamsForMapAlphaMaterial(
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

function textureUvStreamsForMeshMaterial(
  channels: Array<UvChannel | null>,
  material: ThreeMaterialLike | undefined,
): TextureUvStreams {
  return textureUvStreamsForMaterialSlots(channels, meshTextureUvSlots(material))
}

function meshTextureUvSlots(material: ThreeMaterialLike | undefined): MaterialUvSlot[] {
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

function textureUvStreamsForMaterialSlots(
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

function preferredPrimaryTextureChannel(slots: MaterialUvSlot[]): number | undefined {
  for (const flag of ['textureUsesUv2', 'alphaMapUsesUv2'] as const) {
    const slot = slots.find((candidate) => candidate.textureFlag === flag)
    if (slot?.texture != null) return textureUvChannel(slot.texture)
  }
  return slots[0]?.texture ? textureUvChannel(slots[0].texture) : undefined
}

function applyPbrUvStreamFlags(props: PbrProperties, uvStreams: TextureUvStreams): void {
  if (!uvStreams.pbrUsesUv2) return
  for (const [flag, usesUv2] of Object.entries(uvStreams.pbrUsesUv2)) {
    props[flag as PbrUvFlag] = usesUv2
  }
}

function secondaryUvsForMaterial(
  channels: Array<UvChannel | null>,
  material: {
    map?: { channel?: number } | null
    clearcoatMap?: { channel?: number } | null
    clearcoatRoughnessMap?: { channel?: number } | null
    clearcoatNormalMap?: { channel?: number } | null
    sheenColorMap?: { channel?: number } | null
    sheenRoughnessMap?: { channel?: number } | null
    anisotropyMap?: { channel?: number } | null
    iridescenceMap?: { channel?: number } | null
    iridescenceThicknessMap?: { channel?: number } | null
    displacementMap?: { channel?: number } | null
    normalMap?: { channel?: number } | null
    bumpMap?: { channel?: number } | null
    transmissionMap?: { channel?: number } | null
    thicknessMap?: { channel?: number } | null
    specularColorMap?: { channel?: number } | null
    specularIntensityMap?: { channel?: number } | null
    metalnessMap?: { channel?: number } | null
    roughnessMap?: { channel?: number } | null
    emissiveMap?: { channel?: number } | null
    aoMap?: { channel?: number } | null
    lightMap?: { channel?: number } | null
    specularMap?: { channel?: number } | null
    alphaMap?: { channel?: number } | null
  } | undefined,
): UvChannel | null {
  const textures = [
    material?.clearcoatMap,
    material?.clearcoatRoughnessMap,
    material?.clearcoatNormalMap,
    material?.sheenColorMap,
    material?.sheenRoughnessMap,
    material?.anisotropyMap,
    material?.iridescenceMap,
    material?.iridescenceThicknessMap,
    material?.normalMap,
    material?.bumpMap,
    material?.transmissionMap,
    material?.thicknessMap,
    material?.specularColorMap,
    material?.specularIntensityMap,
    material?.displacementMap,
    material?.map,
    material?.metalnessMap,
    material?.roughnessMap,
    material?.emissiveMap,
    material?.lightMap,
    material?.aoMap,
    material?.specularMap,
    material?.alphaMap,
  ]
  let channel = 0
  for (const texture of textures) {
    const textureChannel = textureUvChannel(texture)
    if (textureChannel === 0) continue
    if (channel === 0) {
      channel = textureChannel
      continue
    }
    if (channel !== textureChannel) {
      throw new Error(
        `Material uses multiple non-primary texture.channel values (${channel} and ${textureChannel}), but @headless-three/renderer can bind one secondary UV attribute per draw. Use one shared non-primary channel or render separate passes.`,
      )
    }
  }
  return channels[channel] ?? channels[0]
}

function updateLodObject(object: ThreeObject3DLike, camera: ThreeCameraLike | undefined): void {
  if (object.isLOD !== true || !camera) return
  if (optionalObjectBoolean(object.autoUpdate, 'LOD.autoUpdate') === false) return

  const levels = object.levels
  const normalizedLevels = normalizeLodLevels(levels)
  const cameraZoom = cameraZoomOrDefault(camera.zoom)

  if (typeof object.update === 'function') {
    object.update(camera)
    return
  }

  if (normalizedLevels.length <= 1) return

  const distance = distanceBetweenMatrices(camera.matrixWorld, object.matrixWorld) / cameraZoom
  normalizedLevels[0].object.visible = true

  let i = 1
  for (; i < normalizedLevels.length; i += 1) {
    const level = normalizedLevels[i]
    let levelDistance = level.distance
    if (level.object.visible) {
      levelDistance -= levelDistance * level.hysteresis
    }
    if (distance >= levelDistance) {
      normalizedLevels[i - 1].object.visible = false
      level.object.visible = true
    } else {
      break
    }
  }

  ;(object as { _currentLevel?: number })._currentLevel = i - 1

  for (; i < normalizedLevels.length; i += 1) {
    normalizedLevels[i].object.visible = false
  }
}

function normalizeLodLevels(levels: unknown): Array<{ object: ThreeObject3DLike; distance: number; hysteresis: number }> {
  if (levels == null) return []
  if (!Array.isArray(levels)) {
    throw new TypeError('LOD.levels must be an array.')
  }
  return levels.map((level, index) => {
    if (!level || typeof level !== 'object' || Array.isArray(level)) {
      throw new TypeError(`LOD.levels[${index}] must be an object.`)
    }
    const object = (level as { object?: unknown }).object
    if (!object || typeof object !== 'object' || Array.isArray(object)) {
      throw new TypeError(`LOD.levels[${index}].object must be a THREE.Object3D-like object.`)
    }
    return {
      object: object as ThreeObject3DLike,
      distance: nonNegativeMaterialOrObjectNumber((level as { distance?: unknown }).distance, `LOD.levels[${index}].distance`, 0),
      hysteresis: normalizedMaterialOrObjectNumber((level as { hysteresis?: unknown }).hysteresis, `LOD.levels[${index}].hysteresis`, 0),
    }
  })
}

function distanceBetweenMatrices(a: ThreeCameraLike['matrixWorld'], b: ThreeObject3DLike['matrixWorld']): number {
  const ae = a?.elements
  const be = b?.elements
  if (!ae || ae.length < 16 || !be || be.length < 16) return 0
  const dx = ae[12] - be[12]
  const dy = ae[13] - be[13]
  const dz = ae[14] - be[14]
  return Math.sqrt(dx * dx + dy * dy + dz * dz)
}

function dashedLineAttributes(
  positions: number[],
  uvs: number[] | null,
  uvs2: number[] | null,
  colors: number[] | undefined,
  source: number[],
  start: number,
  end: number,
  object: ThreeObject3DLike,
  lineDistance: ThreeBufferAttributeLike | undefined,
  material: { dashSize?: number; gapSize?: number; scale?: number },
): DashedLineExpansion {
  const dashSize = positiveMaterialOrObjectNumber(material.dashSize, 'material.dashSize', 3)
  const gapSize = nonNegativeMaterialOrObjectNumber(material.gapSize, 'material.gapSize', 1)
  const scale = nonNegativeMaterialOrObjectNumber(material.scale, 'material.scale', 1)

  const segments = lineSegmentsWithDistances(positions, source, start, end, object, lineDistance)
  const out = createDashedLineExpansion(uvs, uvs2, colors)
  if (!lineDistance || gapSize <= 0 || scale === 0) {
    for (const segment of segments) {
      appendInterpolatedLine(out, positions, uvs, uvs2, colors, segment.a, segment.b, 0, 1)
    }
    return out
  }

  const totalSize = dashSize + gapSize
  for (const segment of segments) {
    appendDashedSegment(out, positions, uvs, uvs2, colors, segment, scale, dashSize, totalSize)
  }
  return out
}

function dashedLineAttributesWithCache(
  cache: SceneExtractionCache | undefined,
  geometry: ThreeBufferGeometryLike,
  position: ThreeBufferAttributeLike,
  positions: number[],
  uvChannel: UvChannel | null,
  uvChannel2: UvChannel | null,
  colors: number[] | undefined,
  source: number[],
  start: number,
  end: number,
  object: ThreeObject3DLike,
  lineDistance: ThreeBufferAttributeLike | undefined,
  material: { dashSize?: number; gapSize?: number; scale?: number },
): DashedLineExpansion {
  const dashSize = positiveMaterialOrObjectNumber(material.dashSize, 'material.dashSize', 3)
  const gapSize = nonNegativeMaterialOrObjectNumber(material.gapSize, 'material.gapSize', 1)
  const scale = nonNegativeMaterialOrObjectNumber(material.scale, 'material.scale', 1)
  const uvs = uvChannel?.values ?? null
  const uvs2 = uvChannel2?.values ?? null
  if (!cache || colors) {
    return dashedLineAttributes(positions, uvs, uvs2, colors, source, start, end, object, lineDistance, material)
  }

  const signature = dashedLineSignature(
    geometry,
    position,
    uvChannel,
    uvChannel2,
    lineDistance,
    start,
    end,
    source.length,
    object,
    dashSize,
    gapSize,
    scale,
  )
  if (!signature.cacheable) {
    return dashedLineAttributes(positions, uvs, uvs2, colors, source, start, end, object, lineDistance, material)
  }

  const cacheKey = dashedLineCacheKey(signature)
  const geometryCache = cache.dashedLines.get(geometry)
  const cached = geometryCache?.get(cacheKey)
  if (cached && sameDashedLineSignature(cached.signature, signature)) {
    return cached.expansion
  }

  const expansion = dashedLineAttributes(positions, uvs, uvs2, colors, source, start, end, object, lineDistance, material)
  let writableGeometryCache = geometryCache
  if (!writableGeometryCache) {
    writableGeometryCache = new Map()
    cache.dashedLines.set(geometry, writableGeometryCache)
  }
  writableGeometryCache.set(cacheKey, { signature, expansion })
  return expansion
}

function dashedLineSignature(
  geometry: ThreeBufferGeometryLike,
  position: ThreeBufferAttributeLike,
  uvChannel: UvChannel | null,
  uvChannel2: UvChannel | null,
  lineDistance: ThreeBufferAttributeLike | undefined,
  start: number,
  end: number,
  sourceLength: number,
  object: ThreeObject3DLike,
  dashSize: number,
  gapSize: number,
  scale: number,
): DashedLineSignature {
  const signature: DashedLineSignature = {
    cacheable: true,
    geometryVersion: geometry.version,
    position: attributeSignature(position),
    index: attributeSignature(geometry.index),
    uv: uvChannelSignature(uvChannel),
    uv2: uvChannelSignature(uvChannel2),
    lineDistance: attributeSignature(lineDistance),
    start,
    end,
    sourceLength,
    isLineSegments: object.isLineSegments,
    isLineLoop: object.isLineLoop,
    isLine: object.isLine,
    dashSize,
    gapSize,
    scale,
  }
  signature.cacheable = dashedLineSignatureCacheable(signature)
  return signature
}

function dashedLineSignatureCacheable(signature: DashedLineSignature): boolean {
  return attributeSignatureCacheable(signature.position)
    && attributeSignatureCacheable(signature.index)
    && attributeSignatureCacheable(signature.uv.attribute)
    && attributeSignatureCacheable(signature.uv2.attribute)
    && attributeSignatureCacheable(signature.lineDistance)
}

function dashedLineCacheKey(signature: DashedLineSignature): string {
  return [
    signature.start,
    signature.end,
    signature.sourceLength,
    signature.isLineSegments ? 1 : 0,
    signature.isLineLoop ? 1 : 0,
    signature.isLine ? 1 : 0,
    signature.uv.attribute.ref ? 1 : 0,
    signature.uv2.attribute.ref ? 1 : 0,
    signature.dashSize,
    signature.gapSize,
    signature.scale,
  ].join(':')
}

function sameDashedLineSignature(a: DashedLineSignature, b: DashedLineSignature): boolean {
  return a.cacheable === b.cacheable
    && a.geometryVersion === b.geometryVersion
    && sameAttributeSignature(a.position, b.position)
    && sameAttributeSignature(a.index, b.index)
    && sameUvChannelSignature(a.uv, b.uv)
    && sameUvChannelSignature(a.uv2, b.uv2)
    && sameAttributeSignature(a.lineDistance, b.lineDistance)
    && a.start === b.start
    && a.end === b.end
    && a.sourceLength === b.sourceLength
    && a.isLineSegments === b.isLineSegments
    && a.isLineLoop === b.isLineLoop
    && a.isLine === b.isLine
    && Object.is(a.dashSize, b.dashSize)
    && Object.is(a.gapSize, b.gapSize)
    && Object.is(a.scale, b.scale)
}

function dashedLineAttributesForInstances(
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
      ? offsetVec3ValuesForInstance(positions, offsetAttribute, instance)
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

function offsetVec3ValuesForInstance(
  values: number[],
  offsetAttribute: InstancedAttributeRef,
  instance: number,
): number[] {
  const offsetIndex = instancedAttributeIndex(offsetAttribute.attribute, instance, offsetAttribute.label)
  const ox = attributeComponent(offsetAttribute.attribute, offsetIndex, 0, offsetAttribute.label)
  const oy = attributeComponent(offsetAttribute.attribute, offsetIndex, 1, offsetAttribute.label)
  const oz = attributeComponent(offsetAttribute.attribute, offsetIndex, 2, offsetAttribute.label)
  const out = new Array<number>(values.length)
  for (let i = 0; i < values.length; i += 3) {
    out[i] = values[i] + ox
    out[i + 1] = values[i + 1] + oy
    out[i + 2] = values[i + 2] + oz
  }
  return out
}

function repeatedInstancedColorValues(
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

function appendDashedLineExpansion(out: DashedLineExpansion, value: DashedLineExpansion): void {
  appendNumberArray(out.positions, value.positions)
  if (out.uvs && value.uvs) appendNumberArray(out.uvs, value.uvs)
  if (out.uvs2 && value.uvs2) appendNumberArray(out.uvs2, value.uvs2)
  if (out.colors && value.colors) appendNumberArray(out.colors, value.colors)
}

function batchedMeshDraws(
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

function batchedPerObjectFrustumCulled(
  object: ThreeObject3DLike,
  camera: ThreeCameraLike | undefined,
): boolean {
  if (!camera || camera.isArrayCamera === true) return false
  return batchedOptionalBoolean(object.perObjectFrustumCulled, 'THREE.BatchedMesh.perObjectFrustumCulled', true)
}

function batchedSortObjects(object: ThreeObject3DLike): boolean {
  return batchedOptionalBoolean(object.sortObjects, 'THREE.BatchedMesh.sortObjects', true)
}

function batchedCustomSort(object: ThreeObject3DLike): ThreeObject3DLike['customSort'] {
  if (object.customSort == null) return null
  if (typeof object.customSort === 'function') return object.customSort
  throw new TypeError('THREE.BatchedMesh.customSort must be a function or null.')
}

function customSortedBatchedMeshDraws(
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

function compareBatchedDrawsOpaque(a: BatchedMeshDraw, b: BatchedMeshDraw): number {
  return a.z - b.z || a.instanceId - b.instanceId
}

function compareBatchedDrawsTransparent(a: BatchedMeshDraw, b: BatchedMeshDraw): number {
  return b.z - a.z || a.instanceId - b.instanceId
}

function batchedMeshUsesTransparentSort(object: ThreeObject3DLike): boolean {
  const materials = Array.isArray(object.material)
    ? object.material
    : [object.material]
  return materials.some((material) => material?.transparent === true || finitePositive(material?.transmission))
}

function batchedDrawDistanceZ(
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

function batchedDrawSortIndex(base: number, drawOrder: number): number {
  return Math.max(0, Math.min(0xffffffff, base + drawOrder))
}

function batchedDrawOutsideFrustum(
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

function batchedGeometryRange(
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

function batchedGeometryRangeLimit(geometry: ThreeBufferGeometryLike): number {
  if (geometry.index) return attributeCount(geometry.index, 'geometry.index')
  const position = getAttribute(geometry, 'position')
  return position ? attributeCount(position, 'geometry.attributes.position') : 0
}

function batchedGeometryBoundingSphere(
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

function batchedSphereLike(
  sphere: ThreeSphereLike,
  label: string,
): { center: [number, number, number]; radius: number } {
  return sphereLike(sphere, label)
}

function sphereLike(
  sphere: ThreeSphereLike,
  label: string,
): { center: [number, number, number]; radius: number } {
  if (!sphere || typeof sphere !== 'object') {
    throw new TypeError(`${label} must be a THREE.Sphere-like object.`)
  }
  const center = vec3Like(sphere.center)
  if (!center) {
    throw new TypeError(`${label}.center must be a finite Vector3-like value.`)
  }
  const radius = nonNegativeMaterialOrObjectNumber(sphere.radius, `${label}.radius`, 0)
  return { center, radius }
}

function batchedGeometryView(
  geometry: ThreeBufferGeometryLike,
  range: { start: number; count: number },
  cache?: SceneExtractionCache,
): ThreeBufferGeometryLike {
  const signature = batchedGeometryViewSignature(geometry, range)
  if (cache && signature.cacheable) {
    let views = cache.batchedGeometryViews.get(geometry)
    if (!views) {
      views = new Map()
      cache.batchedGeometryViews.set(geometry, views)
    }
    const key = batchedGeometryViewKey(range)
    const cached = views.get(key)
    if (cached && sameBatchedGeometryViewSignature(cached.signature, signature)) {
      return cached.view
    }
    const view = cached?.view ?? createBatchedGeometryView(geometry)
    updateBatchedGeometryView(view, geometry, range)
    views.set(key, { signature, view })
    return view
  }

  const view = createBatchedGeometryView(geometry)
  updateBatchedGeometryView(view, geometry, range)
  return view
}

function createBatchedGeometryView(geometry: ThreeBufferGeometryLike): ThreeBufferGeometryLike {
  return Object.create(geometry) as ThreeBufferGeometryLike
}

function updateBatchedGeometryView(
  view: ThreeBufferGeometryLike,
  geometry: ThreeBufferGeometryLike,
  range: { start: number; count: number },
): void {
  view.drawRange = { start: range.start, count: range.count }
  view.boundingSphere = batchedGeometryRangeBoundingSphere(geometry, range)
}

function batchedGeometryViewSignature(
  geometry: ThreeBufferGeometryLike,
  range: { start: number; count: number },
): BatchedGeometryViewSignature {
  const signature: BatchedGeometryViewSignature = {
    cacheable: true,
    geometryVersion: geometry.version,
    rangeStart: range.start,
    rangeCount: range.count,
    position: attributeSignature(getAttribute(geometry, 'position')),
    index: attributeSignature(geometry.index),
  }
  signature.cacheable = attributeSignatureCacheable(signature.position)
    && attributeSignatureCacheable(signature.index)
  return signature
}

function sameBatchedGeometryViewSignature(
  a: BatchedGeometryViewSignature,
  b: BatchedGeometryViewSignature,
): boolean {
  return a.cacheable === b.cacheable
    && a.geometryVersion === b.geometryVersion
    && a.rangeStart === b.rangeStart
    && a.rangeCount === b.rangeCount
    && sameAttributeSignature(a.position, b.position)
    && sameAttributeSignature(a.index, b.index)
}

function batchedGeometryViewKey(range: { start: number; count: number }): string {
  return `${range.start}:${range.count}`
}

function batchedGeometryRangeBoundingSphere(
  geometry: ThreeBufferGeometryLike,
  range: { start: number; count: number },
): { center: { x: number; y: number; z: number }; radius: number } | undefined {
  const position = getAttribute(geometry, 'position')
  if (!position || range.count <= 0) return undefined

  const index = geometry.index ? readIndexAttribute(geometry.index, 'geometry.index', position.count) : null
  const start = Math.max(0, range.start)
  const end = Math.max(start, range.start + range.count)
  let minX = Number.POSITIVE_INFINITY
  let minY = Number.POSITIVE_INFINITY
  let minZ = Number.POSITIVE_INFINITY
  let maxX = Number.NEGATIVE_INFINITY
  let maxY = Number.NEGATIVE_INFINITY
  let maxZ = Number.NEGATIVE_INFINITY
  const vertexIndices: number[] = []

  for (let offset = start; offset < end; offset += 1) {
    const vertexIndex = index ? index[offset] : offset
    if (!Number.isInteger(vertexIndex) || vertexIndex < 0 || vertexIndex >= position.count) continue
    vertexIndices.push(vertexIndex)
    const x = attributeComponent(position, vertexIndex, 0, 'geometry.attributes.position')
    const y = attributeComponent(position, vertexIndex, 1, 'geometry.attributes.position')
    const z = attributeComponent(position, vertexIndex, 2, 'geometry.attributes.position')
    minX = Math.min(minX, x)
    minY = Math.min(minY, y)
    minZ = Math.min(minZ, z)
    maxX = Math.max(maxX, x)
    maxY = Math.max(maxY, y)
    maxZ = Math.max(maxZ, z)
  }

  if (vertexIndices.length === 0) return undefined

  const center = {
    x: (minX + maxX) * 0.5,
    y: (minY + maxY) * 0.5,
    z: (minZ + maxZ) * 0.5,
  }
  let radiusSq = 0
  for (const vertexIndex of vertexIndices) {
    const dx = attributeComponent(position, vertexIndex, 0, 'geometry.attributes.position') - center.x
    const dy = attributeComponent(position, vertexIndex, 1, 'geometry.attributes.position') - center.y
    const dz = attributeComponent(position, vertexIndex, 2, 'geometry.attributes.position') - center.z
    radiusSq = Math.max(radiusSq, dx * dx + dy * dy + dz * dz)
  }

  return { center, radius: Math.sqrt(radiusSq) }
}

function batchedInstanceMatrix(object: ThreeObject3DLike, instanceId: number): number[] {
  const data = batchedTextureImageData(
    object._matricesTexture,
    'THREE.BatchedMesh._matricesTexture',
    instanceId,
  )
  const offset = instanceId * 16
  if (typeof data.length !== 'number' || data.length < offset + 16) {
    throw new Error(
      `THREE.BatchedMesh matrix texture is not readable for instance ${instanceId}. Use a real THREE.BatchedMesh or expand the batch before rendering.`,
    )
  }

  const matrix = new Array<number>(16)
  for (let component = 0; component < 16; component += 1) {
    matrix[component] = finiteArrayValue(data, offset + component, 'THREE.BatchedMesh._matricesTexture.image.data')
  }
  return matrix
}

function batchedInstanceColor(object: ThreeObject3DLike, instanceId: number): Color4 | undefined {
  const texture = object._colorsTexture
  if (texture == null) return undefined
  const data = batchedTextureImageData(texture, 'THREE.BatchedMesh._colorsTexture', instanceId)

  const offset = instanceId * 4
  if (typeof data.length !== 'number' || data.length < offset + 4) {
    throw new Error(
      `THREE.BatchedMesh color texture is not readable for instance ${instanceId}. Use a real THREE.BatchedMesh or expand the batch before rendering.`,
    )
  }

  return [
    finiteArrayValue(data, offset, 'THREE.BatchedMesh._colorsTexture.image.data'),
    finiteArrayValue(data, offset + 1, 'THREE.BatchedMesh._colorsTexture.image.data'),
    finiteArrayValue(data, offset + 2, 'THREE.BatchedMesh._colorsTexture.image.data'),
    finiteArrayValue(data, offset + 3, 'THREE.BatchedMesh._colorsTexture.image.data'),
  ]
}

function batchedTextureImageData(texture: unknown, label: string, instanceId: number): ArrayLike<number> {
  if (!texture || typeof texture !== 'object') {
    throw new TypeError(`${label} must be a texture-like object for instance ${instanceId}.`)
  }
  const image = (texture as { image?: unknown }).image
  if (!image || typeof image !== 'object') {
    throw new TypeError(`${label}.image must be an image-like object for instance ${instanceId}.`)
  }
  const data = (image as { data?: unknown }).data
  if (!data || typeof data !== 'object') {
    throw new TypeError(`${label}.image.data must be an array-like object for instance ${instanceId}.`)
  }
  return data as ArrayLike<number>
}

function batchedOptionalBoolean(value: unknown, label: string, fallback: boolean): boolean {
  if (value == null) return fallback
  if (typeof value === 'boolean') return value
  throw new TypeError(`${label} must be a boolean.`)
}

function batchedNonNegativeInteger(value: unknown, label: string): number {
  if (typeof value === 'number' && Number.isFinite(value) && Number.isInteger(value) && value >= 0) {
    return value
  }
  throw new TypeError(`${label} must be a non-negative integer.`)
}

function finiteArrayValue(values: ArrayLike<number>, index: number, label: string): number {
  const value = values[index]
  if (typeof value === 'number' && Number.isFinite(value)) return value
  throw new TypeError(`${label}[${index}] must be a finite number.`)
}

function appendNumberArray(out: number[], values: number[]): void {
  for (const value of values) out.push(value)
}

function createDashedLineExpansion(
  uvs: number[] | null,
  uvs2: number[] | null,
  colors: number[] | undefined,
): DashedLineExpansion {
  return {
    positions: [],
    uvs: uvs ? [] : undefined,
    uvs2: uvs2 ? [] : undefined,
    colors: colors ? [] : undefined,
  }
}

function cameraFrustumIntersectsSphere(
  camera: ThreeCameraLike,
  center: [number, number, number],
  radius: number,
): boolean {
  const view = camera.matrixWorldInverse?.elements
  const projection = camera.projectionMatrix?.elements
  if (!view || view.length < 16 || !projection || projection.length < 16) return true
  const viewProjection = multiplyMat4(projection, view)

  const planes: Array<[number, number, number, number]> = [
    [viewProjection[3] + viewProjection[0], viewProjection[7] + viewProjection[4], viewProjection[11] + viewProjection[8], viewProjection[15] + viewProjection[12]],
    [viewProjection[3] - viewProjection[0], viewProjection[7] - viewProjection[4], viewProjection[11] - viewProjection[8], viewProjection[15] - viewProjection[12]],
    [viewProjection[3] + viewProjection[1], viewProjection[7] + viewProjection[5], viewProjection[11] + viewProjection[9], viewProjection[15] + viewProjection[13]],
    [viewProjection[3] - viewProjection[1], viewProjection[7] - viewProjection[5], viewProjection[11] - viewProjection[9], viewProjection[15] - viewProjection[13]],
    [viewProjection[3] + viewProjection[2], viewProjection[7] + viewProjection[6], viewProjection[11] + viewProjection[10], viewProjection[15] + viewProjection[14]],
    [viewProjection[3] - viewProjection[2], viewProjection[7] - viewProjection[6], viewProjection[11] - viewProjection[10], viewProjection[15] - viewProjection[14]],
  ]

  for (const [a, b, c, d] of planes) {
    const length = Math.hypot(a, b, c)
    if (!Number.isFinite(length) || length <= 1e-8) continue
    const distance = (a * center[0] + b * center[1] + c * center[2] + d) / length
    if (distance < -radius) return false
  }

  return true
}

function lineSegmentsWithDistances(
  positions: number[],
  source: number[],
  start: number,
  end: number,
  object: ThreeObject3DLike,
  lineDistance: ThreeBufferAttributeLike | undefined,
): LineSegmentDistance[] {
  const count = end - start
  const segments: LineSegmentDistance[] = []
  if (count < 2) return segments

  if (object.isLineSegments === true) {
    let cumulative = 0
    const aligned = count - (count % 2)
    for (let i = 0; i < aligned; i += 2) {
      const a = source[start + i]
      const b = source[start + i + 1]
      const length = vertexDistance(positions, a, b)
      const d0 = lineDistance ? attributeComponent(lineDistance, a, 0, 'geometry.attributes.lineDistance') : cumulative
      const d1 = lineDistance ? attributeComponent(lineDistance, b, 0, 'geometry.attributes.lineDistance') : d0 + length
      segments.push({ a, b, d0, d1 })
      cumulative = d1
    }
    return segments
  }

  let previous = source[start]
  let previousDistance = lineDistance ? attributeComponent(lineDistance, previous, 0, 'geometry.attributes.lineDistance') : 0
  for (let i = 1; i < count; i += 1) {
    const current = source[start + i]
    const length = vertexDistance(positions, previous, current)
    const currentDistance = lineDistance ? attributeComponent(lineDistance, current, 0, 'geometry.attributes.lineDistance') : previousDistance + length
    segments.push({ a: previous, b: current, d0: previousDistance, d1: currentDistance })
    previous = current
    previousDistance = currentDistance
  }
  if (object.isLineLoop === true && count >= 2) {
    const first = source[start]
    segments.push({
      a: previous,
      b: first,
      d0: previousDistance,
      d1: lineDistance
        ? attributeComponent(lineDistance, first, 0, 'geometry.attributes.lineDistance')
        : previousDistance + vertexDistance(positions, previous, first),
    })
  }
  return segments
}

function appendDashedSegment(
  out: DashedLineExpansion,
  positions: number[],
  uvs: number[] | null,
  uvs2: number[] | null,
  colors: number[] | undefined,
  segment: LineSegmentDistance,
  scale: number,
  dashSize: number,
  totalSize: number,
): void {
  const s0 = segment.d0 * scale
  const s1 = segment.d1 * scale
  const span = s1 - s0
  const direction = Math.sign(span)
  if (Math.abs(span) <= 1e-6 || direction === 0) return

  let cursor = s0
  let guard = 0
  while ((direction > 0 ? cursor < s1 - 1e-6 : cursor > s1 + 1e-6) && guard < 10000) {
    guard += 1
    const cycle = Math.floor(cursor / totalSize)
    const cycleStart = cycle * totalSize
    const inCycle = cursor - cycleStart
    const visible = inCycle <= dashSize
    const boundary = direction > 0
      ? cycleStart + (visible ? dashSize : totalSize)
      : cycleStart + (visible ? 0 : dashSize)
    const next = direction > 0
      ? Math.min(s1, boundary <= cursor + 1e-6 ? cursor + 1e-6 : boundary)
      : Math.max(s1, boundary >= cursor - 1e-6 ? cursor - 1e-6 : boundary)
    const hasVisibleSpan = direction > 0 ? next > cursor + 1e-6 : next < cursor - 1e-6
    if (visible && hasVisibleSpan) {
      const t0 = (cursor - s0) / span
      const t1 = (next - s0) / span
      appendInterpolatedLine(out, positions, uvs, uvs2, colors, segment.a, segment.b, t0, t1)
    }
    cursor = next
  }
}

function appendInterpolatedLine(
  out: DashedLineExpansion,
  positions: number[],
  uvs: number[] | null,
  uvs2: number[] | null,
  colors: number[] | undefined,
  a: number,
  b: number,
  t0: number,
  t1: number,
): void {
  appendInterpolatedAttribute(out.positions, positions, 3, a, b, t0)
  appendInterpolatedAttribute(out.positions, positions, 3, a, b, t1)
  if (out.uvs && uvs) {
    appendInterpolatedAttribute(out.uvs, uvs, 2, a, b, t0)
    appendInterpolatedAttribute(out.uvs, uvs, 2, a, b, t1)
  }
  if (out.uvs2 && uvs2) {
    appendInterpolatedAttribute(out.uvs2, uvs2, 2, a, b, t0)
    appendInterpolatedAttribute(out.uvs2, uvs2, 2, a, b, t1)
  }
  if (out.colors && colors) {
    appendInterpolatedAttribute(out.colors, colors, 4, a, b, t0)
    appendInterpolatedAttribute(out.colors, colors, 4, a, b, t1)
  }
}

function appendInterpolatedAttribute(
  out: number[],
  values: number[],
  itemSize: number,
  a: number,
  b: number,
  t: number,
): void {
  const aBase = a * itemSize
  const bBase = b * itemSize
  for (let component = 0; component < itemSize; component += 1) {
    const av = values[aBase + component]
    const bv = values[bBase + component]
    out.push(av + (bv - av) * t)
  }
}

function vertexDistance(positions: number[], a: number, b: number): number {
  const dx = positions[a * 3] - positions[b * 3]
  const dy = positions[a * 3 + 1] - positions[b * 3 + 1]
  const dz = positions[a * 3 + 2] - positions[b * 3 + 2]
  return Math.sqrt(dx * dx + dy * dy + dz * dz)
}

function meshInstances(object: ThreeObject3DLike, baseTransform: number[]): MeshInstance[] {
  if (object.isInstancedMesh !== true) {
    return [{ transform: baseTransform }]
  }

  const instanceMatrix = object.instanceMatrix
  if (!instanceMatrix || instanceMatrix.count == null) return []
  const instanceMatrixCount = attributeCount(instanceMatrix, 'InstancedMesh.instanceMatrix')

  const count = clampInteger(
    integerCountOrDefault(object.count, 'InstancedMesh.count', instanceMatrixCount),
    0,
    instanceMatrixCount,
  )
  const instances = new Array<MeshInstance>(count)
  for (let i = 0; i < count; i += 1) {
    instances[i] = {
      transform: multiplyMat4(baseTransform, readMat4Attribute(instanceMatrix, i)),
      color: readInstanceColor(object.instanceColor, i),
    }
  }
  return instances
}

function readMat4Attribute(attribute: ThreeObject3DLike['instanceMatrix'], index: number): number[] {
  if (!attribute) return IDENTITY_4X4.slice()
  const matrix = new Array<number>(16)
  for (let component = 0; component < 16; component += 1) {
    matrix[component] = attributeComponent(attribute, index, component, 'InstancedMesh.instanceMatrix')
  }
  return matrix
}

function readInstanceColor(attribute: ThreeObject3DLike['instanceColor'], index: number): Color4 | undefined {
  if (!attribute || index >= attributeCount(attribute, 'InstancedMesh.instanceColor')) return undefined
  return [
    attributeComponent(attribute, index, 0, 'InstancedMesh.instanceColor'),
    attributeComponent(attribute, index, 1, 'InstancedMesh.instanceColor'),
    attributeComponent(attribute, index, 2, 'InstancedMesh.instanceColor'),
    attribute.itemSize && attribute.itemSize >= 4 ? attributeComponent(attribute, index, 3, 'InstancedMesh.instanceColor') : 1,
  ]
}

function instanceColor(baseColor: Color4, instance: MeshInstance): Color4 {
  if (!instance.color) return baseColor
  return [
    baseColor[0] * instance.color[0],
    baseColor[1] * instance.color[1],
    baseColor[2] * instance.color[2],
    baseColor[3] * instance.color[3],
  ]
}

function multiplyMat4(a: ArrayLike<number>, b: ArrayLike<number>): number[] {
  const out = new Array<number>(16)
  for (let col = 0; col < 4; col += 1) {
    for (let row = 0; row < 4; row += 1) {
      out[col * 4 + row] =
        a[row] * b[col * 4]
        + a[4 + row] * b[col * 4 + 1]
        + a[8 + row] * b[col * 4 + 2]
        + a[12 + row] * b[col * 4 + 3]
    }
  }
  return out
}

function rangeIndices(count: number): number[] {
  const out = new Array<number>(count)
  for (let i = 0; i < count; i++) out[i] = i
  return out
}

/**
 * Convert a LineStrip / LineSegments / LineLoop index stream into a flat
 * LineList `[a, b, b, c, ...]` array.
 */
function expandLineIndices(
  source: number[],
  start: number,
  end: number,
  object: ThreeObject3DLike,
): number[] {
  const count = end - start
  if (count < 2) return []

  if (object.isLineSegments === true) {
    // already pairs; just validate alignment
    const aligned = count - (count % 2)
    return source.slice(start, start + aligned)
  }

  const out: number[] = []
  for (let i = 0; i < count - 1; i++) {
    out.push(source[start + i], source[start + i + 1])
  }
  if (object.isLineLoop === true && count >= 2) {
    out.push(source[start + count - 1], source[start])
  }
  return out
}
