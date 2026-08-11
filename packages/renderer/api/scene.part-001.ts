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
export interface FlattenedMesh {
  mesh: NativeSceneMesh
  sortItem: RenderSortItem
  transparentDoubleSidePass?: boolean
  groupOrder: number
  renderOrder: number
  sortZ: number
  materialSortKey: number
  materialVariant: number
  sortIndex: number
}

export interface MeshInstance {
  transform: number[]
  color?: Color4
}

export interface BatchedMeshDraw {
  range: { start: number; count: number }
  instanceId: number
  instance: MeshInstance
  z: number
}

export type ShadowMaterialMode = 'depth' | 'distance'

export interface LineSegmentDistance {
  a: number
  b: number
  d0: number
  d1: number
}

export interface DashedLineExpansion {
  positions: number[]
  uvs?: number[]
  uvs2?: number[]
  colors?: number[]
}

export interface InstancedAttributeRef {
  attribute: ThreeBufferAttributeLike
  label: string
}

export const SupportedInstancedBufferGeometryAttributes = new Set([
  'color',
  'normal',
  'uv',
  'uv1',
  'uv2',
  'uv3',
  'instanceOffset',
  'instancePosition',
  'instanceScale',
  'offset',
  'scale',
  'translate',
  'translation',
])

export const NativeRenderOrderLimit = 1_000_000_000

export interface ThickLineExpansion {
  center: [number, number, number]
  colors?: number[]
  indices: number[]
  positions: number[]
  uvs?: number[]
  uvs2?: number[]
}

export interface TextureUvStreams {
  uvs: UvChannel | null
  uvs2: UvChannel | null
  textureUsesUv2?: boolean
  alphaMapUsesUv2?: boolean
  pbrUsesUv2?: Partial<Record<PbrUvFlag, boolean>>
}

export interface SceneExtractionCache {
  meshGeometry: WeakMap<ThreeBufferGeometryLike, unknown>
  instancedMeshes: WeakMap<ThreeObject3DLike, CachedInstancedMeshInstances>
  instancedPositionExpansions: WeakMap<ThreeBufferGeometryLike, Map<string, CachedInstancedPositionExpansion>>
  instancedNormalExpansions: WeakMap<ThreeBufferGeometryLike, Map<string, CachedInstancedNormalExpansion>>
  instancedUvExpansions: WeakMap<ThreeBufferGeometryLike, Map<string, CachedInstancedUvExpansion>>
  instancedColorExpansions: WeakMap<ThreeBufferGeometryLike, Map<string, CachedInstancedColorExpansion>>
  indexRanges: WeakMap<ThreeBufferGeometryLike, Map<string, CachedIndexRangeExpansion>>
  batchedGeometryViews: WeakMap<ThreeBufferGeometryLike, Map<string, CachedBatchedGeometryView>>
  dashedLines: WeakMap<ThreeBufferGeometryLike, Map<string, CachedDashedLineExpansion>>
  instancedDashedLines: WeakMap<ThreeBufferGeometryLike, Map<string, CachedInstancedDashedLineExpansion>>
  texturePayloads: TextureExtractionCache
  materialColors: MaterialColorExtractionCache
  textureStates: TextureStateExtractionCache
  materialRenderStates: MaterialRenderStateExtractionCache
  materialScalarFeatures: MaterialScalarFeatureExtractionCache
  pointBillboards: WeakMap<ThreeObject3DLike, Map<string, CachedPointBillboardExpansion>>
  spriteBillboards: WeakMap<ThreeObject3DLike, CachedSpriteBillboardExpansion>
  nativeMeshPayloads: NativeMeshPayloadCache
}

export interface NativeMeshPayloadCache {
  objectIds: WeakMap<object, number>
  payloads: Map<string, CachedNativeMeshPayload>
  pending: Set<string>
  nextObjectId: number
  nextPayloadId: number
}

export interface CachedNativeMeshPayload {
  key: number
  vertexCount: number
  indexCount?: number
  ready: boolean
}

export interface CachedMeshGeometryExtraction {
  signature: MeshGeometrySignature
  extraction: MeshGeometryExtraction
}

export interface CachedInstancedMeshInstances {
  signature: InstancedMeshSignature
  localInstances: MeshInstance[]
}

export interface CachedInstancedPositionExpansion {
  signature: InstancedPositionExpansionSignature
  positions: number[]
}

export interface CachedInstancedNormalExpansion {
  signature: InstancedNormalExpansionSignature
  normals: number[]
}

export interface CachedInstancedUvExpansion {
  signature: InstancedUvExpansionSignature
  uvs: number[]
}

export interface CachedInstancedColorExpansion {
  signature: InstancedColorExpansionSignature
  colors: number[]
}

export interface CachedIndexRangeExpansion {
  signature: IndexRangeExpansionSignature
  indices: number[]
}

export interface CachedBatchedGeometryView {
  signature: BatchedGeometryViewSignature
  view: ThreeBufferGeometryLike
}

export interface CachedDashedLineExpansion {
  signature: DashedLineSignature
  expansion: DashedLineExpansion
}

export interface CachedInstancedDashedLineExpansion {
  signature: InstancedDashedLineSignature
  expansion: DashedLineExpansion
}

export interface CachedPointBillboardExpansion {
  signature: PointBillboardSignature
  expansion: PointBillboardExpansion
}

export interface CachedSpriteBillboardExpansion {
  signature: SpriteBillboardSignature
  expansion: SpriteBillboardExpansion
}

export interface SpriteBillboardExpansion {
  positions: number[]
  indices: number[]
  uvs: number[]
}

export interface SpriteBillboardSignature {
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

export interface PointBillboardExpansion {
  positions: number[]
  indices: number[]
  uvs: number[]
  uvs2?: number[]
  colors?: number[]
  pointRefs: Array<{ pointIndex: number, instance: number }>
}

export interface PointBillboardSignature {
  cacheable: boolean
  positions: number[]
  positionCount: number
  index: number[] | null
  groupStart: number
  groupCount: number
  instancedGeometryCount: number
  instancedPositionOffset: AttributeSignature
  instancedPositionScale: AttributeSignature
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

export interface DashedLineSignature {
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

export interface InstancedDashedLineSignature {
  cacheable: boolean
  geometryVersion?: number
  position: AttributeSignature
  index: AttributeSignature
  uv: UvChannelSignature
  uv2: UvChannelSignature
  lineDistance: AttributeSignature
  vertexColors: AttributeSignature
  instancedPositionOffset: AttributeSignature
  instancedPositionScale: AttributeSignature
  materialColor?: Color4
  start: number
  end: number
  sourceLength: number
  instanceCount: number
  isLineSegments?: boolean
  isLineLoop?: boolean
  isLine?: boolean
  dashSize: number
  gapSize: number
  scale: number
}

export interface UvChannelSignature {
  attribute: AttributeSignature
  values?: number[]
  label?: string
}

export interface BatchedGeometryViewSignature {
  cacheable: boolean
  geometryVersion?: number
  rangeStart: number
  rangeCount: number
  position: AttributeSignature
  index: AttributeSignature
}

export interface MeshGeometryExtraction {
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
  instancedPositionScale: InstancedAttributeRef | null
}

export interface MeshGeometrySignature {
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
  instancedPositionScaleName?: string
  instancedPositionScale: AttributeSignature
  instancedAttributes: Array<{ name: string; signature: AttributeSignature }>
}

export interface InstancedMeshSignature {
  cacheable: boolean
  count: number
  instanceMatrix: AttributeSignature
  instanceColor: AttributeSignature
}

export interface InstancedPositionExpansionSignature {
  cacheable: boolean
  geometryVersion?: number
  sourcePositions: number[]
  start: number
  count: number
  instanceCount: number
  position: AttributeSignature
  instancedPositionOffset: AttributeSignature
  instancedPositionScale: AttributeSignature
}

export interface InstancedNormalExpansionSignature {
  cacheable: boolean
  geometryVersion?: number
  sourceNormals: number[]
  start: number
  count: number
  instanceCount: number
  normal: AttributeSignature
  label: string
}

export interface InstancedUvExpansionSignature {
  cacheable: boolean
  geometryVersion?: number
  channel: UvChannelSignature
  start: number
  count: number
  instanceCount: number
}

export interface InstancedColorExpansionSignature {
  cacheable: boolean
  geometryVersion?: number
  materialColor: Color4
  start: number
  count: number
  instanceCount: number
  color: AttributeSignature
  label: string
}

export interface IndexRangeExpansionSignature {
  cacheable: boolean
  geometryVersion?: number
  sourceIndices: number[]
  start: number
  count: number
  index: AttributeSignature
}

export interface AttributeSignature {
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

export type PbrUvFlag = Extract<keyof PbrProperties, `${string}UsesUv2`>

export interface MaterialUvSlot {
  texture?: { channel?: number } | null
  textureFlag?: 'textureUsesUv2' | 'alphaMapUsesUv2'
  pbrFlag?: PbrUvFlag
}

export interface UvChannel {
  attribute: ThreeBufferAttributeLike
  label: string
  values: number[]
}

export interface ClippingContext {
  unionPlanes: readonly NativeClippingPlane[]
  intersectionPlanes: readonly NativeClippingPlane[]
  clipShadows: boolean
}
