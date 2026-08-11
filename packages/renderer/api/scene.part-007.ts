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
import { ClippingContext, FlattenedMesh, NativeMeshPayloadCache } from './scene.part-001'
import { MeshSortInfo, SceneSortOptions, SortKeyOverride } from './scene.part-002'
import { optionalObjectBoolean } from './scene.part-005'
import { finiteOrDefault, nativeRenderOrderKey, objectBoundingSphere, pointBillboardCullRadius, renderOrderNumber, renderableMatrixWorldLabel, requiredVec3Like, transformedSphereOutsideFrustum, unsignedSortKey } from './scene.part-008'
export function clippingState(
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

export function clippingContextForObject(parent: ClippingContext, object: ThreeObject3DLike): ClippingContext {
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

export function clipShadowsForMaterial(material: ThreeMaterialLike | undefined, clippingContext: ClippingContext): boolean | undefined {
  return optionalObjectBoolean(material?.clipShadows, 'material.clipShadows') === true || clippingContext.clipShadows ? true : undefined
}

export function pushMesh(
  meshes: FlattenedMesh[],
  mesh: NativeSceneMesh,
  sortItem: RenderSortItem,
  transparentDoubleSidePass = false,
): void {
  const groupOrder = mesh.groupOrder ?? 0
  const renderOrder = mesh.renderOrder ?? 0
  meshes.push({
    mesh: {
      ...mesh,
      groupOrder: nativeRenderOrderKey(groupOrder),
      renderOrder: nativeRenderOrderKey(renderOrder),
    },
    sortItem,
    groupOrder,
    renderOrder,
    sortZ: mesh.sortZ ?? 0,
    materialSortKey: mesh.materialSortKey ?? 0,
    materialVariant: mesh.materialVariant ?? 0,
    sortIndex: mesh.sortIndex ?? meshes.length,
    transparentDoubleSidePass,
  })
}

export function needsTransparentDoubleSidePass(material: ThreeMaterialLike | undefined, props: PbrProperties, wireframe: boolean): boolean {
  return !wireframe
    && material?.transparent === true
    && material.forceSinglePass === false
    && props.side === 'double'
}

export function nativeMeshesFromSortedFlattened(meshes: FlattenedMesh[]): NativeSceneMesh[] {
  const nativeMeshes: NativeSceneMesh[] = []
  for (const { mesh, transparentDoubleSidePass } of meshes) {
    if (transparentDoubleSidePass) {
      nativeMeshes.push({ ...mesh, side: 'back' })
      nativeMeshes.push({ ...mesh, side: 'front' })
    } else {
      nativeMeshes.push(mesh)
    }
  }
  return nativeMeshes
}

export const NativeMeshPayloadCacheLimit = 2048

export function applyNativeMeshPayloadCache(meshes: NativeSceneMesh[], cache: NativeMeshPayloadCache): void {
  for (const mesh of meshes) {
    const signature = nativeMeshPayloadSignature(mesh, cache)
    if (!signature) continue

    let cached = cache.payloads.get(signature)
    if (!cached) {
      const positions = mesh.positions!
      cached = {
        key: cache.nextPayloadId++,
        vertexCount: positions.length / 3,
        indexCount: mesh.indices?.length,
        ready: false,
      }
      evictNativeMeshPayloads(cache)
      cache.payloads.set(signature, cached)
    }

    mesh.nativeMeshKey = cached.key
    mesh.nativeVertexCount = cached.vertexCount
    mesh.nativeIndexCount = cached.indexCount

    if (cached.ready) {
      mesh.positions = []
      delete mesh.indices
      delete mesh.normals
      delete mesh.colors
      delete mesh.uvs
      delete mesh.uvs2
    } else {
      cache.pending.add(signature)
    }
  }
}

export function nativeMeshPayloadSignature(mesh: NativeSceneMesh, cache: NativeMeshPayloadCache): string | null {
  if (!mesh.positions || mesh.positions.length === 0) return null
  if (mesh.positions.length % 3 !== 0) return null
  if (!nativeMeshPayloadCacheable(mesh)) return null

  return [
    'p', nativeMeshPayloadObjectId(cache, mesh.positions),
    'i', nativeMeshPayloadObjectId(cache, mesh.indices),
    'n', nativeMeshPayloadObjectId(cache, mesh.normals),
    'c', nativeMeshPayloadObjectId(cache, mesh.colors),
    'u', nativeMeshPayloadObjectId(cache, mesh.uvs),
    'v', nativeMeshPayloadObjectId(cache, mesh.uvs2),
    'flat', mesh.flatShading === true ? 1 : 0,
    'topology', mesh.topology ?? 'triangles',
  ].join(':')
}

export function nativeMeshPayloadCacheable(mesh: NativeSceneMesh): boolean {
  return mesh.displacementMap == null
    && mesh.normalMap == null
    && mesh.bumpMap == null
    && mesh.clearcoatNormalMap == null
    && (mesh.anisotropy == null || mesh.anisotropy <= 0)
}

export function nativeMeshPayloadObjectId(cache: NativeMeshPayloadCache, value: object | null | undefined): string {
  if (value == null) return 'none'
  let id = cache.objectIds.get(value)
  if (id === undefined) {
    id = cache.nextObjectId++
    cache.objectIds.set(value, id)
  }
  return String(id)
}

export function evictNativeMeshPayloads(cache: NativeMeshPayloadCache): void {
  while (cache.payloads.size >= NativeMeshPayloadCacheLimit) {
    const first = cache.payloads.keys().next()
    if (first.done) return
    cache.payloads.delete(first.value)
    cache.pending.delete(first.value)
  }
}

export function sortInfoForObject(
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
  const renderOrder = renderOrderNumber(object.renderOrder, 'object.renderOrder', 0)
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

export function mergeSortKeys(keys: MeshSortInfo['keys'], override: SortKeyOverride | undefined): MeshSortInfo['keys'] {
  return override ? { ...keys, ...override } : keys
}

export function sortFlattenedMeshes(meshes: FlattenedMesh[], options: SceneSortOptions): FlattenedMesh[] {
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

export function partitionFlattenedMeshes(meshes: FlattenedMesh[]): {
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

export function compareWithSort(sort: RenderSortFunction): (a: FlattenedMesh, b: FlattenedMesh) => number {
  return (a, b) => {
    const result = Number(sort(a.sortItem, b.sortItem))
    return Number.isFinite(result) ? result : 0
  }
}

export function normalizeSortKeys(meshes: FlattenedMesh[]): void {
  meshes.forEach((entry, index) => {
    entry.mesh.groupOrder = 0
    entry.mesh.renderOrder = 0
    entry.mesh.materialSortKey = 0
    entry.mesh.materialVariant = 0
    entry.mesh.sortZ = 0
    entry.mesh.sortIndex = index
  })
}

export function compareFlattenedMeshes(a: FlattenedMesh, b: FlattenedMesh): number {
  return a.groupOrder - b.groupOrder
    || a.renderOrder - b.renderOrder
    || a.materialSortKey - b.materialSortKey
    || a.materialVariant - b.materialVariant
    || a.sortZ - b.sortZ
    || a.sortIndex - b.sortIndex
}

export function compareTransparentFlattenedMeshes(a: FlattenedMesh, b: FlattenedMesh): number {
  return a.groupOrder - b.groupOrder
    || a.renderOrder - b.renderOrder
    || b.sortZ - a.sortZ
    || a.sortIndex - b.sortIndex
}

export function meshDefaultsTransparent(mesh: NativeSceneMesh): boolean {
  if (mesh.alphaHash === true) return false
  if (mesh.transparent === true) return true
  if (mesh.transparent === false) return false
  return materialAlpha(mesh) < 0.999
}

export function materialAlpha(mesh: NativeSceneMesh): number {
  return mesh.color && mesh.color.length >= 4 ? finiteOrDefault(mesh.color[3], 1) : 1
}

export function finitePositive(value: unknown): boolean {
  return typeof value === 'number' && Number.isFinite(value) && value > 0.0001
}

export function materialVariantForObject(object: ThreeObject3DLike): number {
  return (object.isInstancedMesh === true ? 2 : 0) + (object.isSkinnedMesh === true ? 1 : 0)
}

export function projectedObjectZ(object: ThreeObject3DLike, camera: ThreeCameraLike, transform?: ArrayLike<number>): number {
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

export function projectedWorldPointZ(worldPoint: [number, number, number], camera: ThreeCameraLike): number {
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

export function objectSortCenter(object: ThreeObject3DLike): [number, number, number] {
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

export function renderableObjectOutsideFrustum(
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
