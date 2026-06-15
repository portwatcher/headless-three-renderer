import type {
  ThreeObject3DLike,
  ThreeBufferAttributeLike,
  ThreeBufferGeometryLike,
  ThreeCameraLike,
  ThreeMaterialLike,
  NativeSceneMesh,
  GeometryGroup,
  Color4,
  RenderSortFunction,
  RenderSortItem,
} from './types'
import { IDENTITY_4X4, matrixElements, clampInteger, clamp01 } from './math'
import {
  attributeComponent,
  getAttribute,
  readVec3Attribute,
  readVec2Attribute,
  readColorAttribute,
  readIndexAttribute,
} from './attributes'
import {
  materialForGroup,
  materialColor,
  extractPbrProperties,
  extractTextureData,
  materialShadowSide,
  textureUvChannel,
  type MaterialExtractionContext,
} from './materials'
import { applyCpuSkinning } from './skinning'
import { applyMorphTargets } from './morphs'
import { objectLayersMatchCamera } from './layers'
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

interface ThickLineExpansion {
  center: [number, number, number]
  colors?: number[]
  indices: number[]
  positions: number[]
  uvs?: number[]
  uvs2?: number[]
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
}

interface MeshSortInfo {
  keys: Pick<NativeSceneMesh, 'groupOrder' | 'renderOrder' | 'sortZ' | 'sortIndex' | 'materialSortKey' | 'materialVariant'>
  item: RenderSortItem
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
): NativeSceneMesh[] {
  const meshes: FlattenedMesh[] = []
  const clippingContext: ClippingContext = {
    unionPlanes: globalClippingPlanes,
    intersectionPlanes: [],
    clipShadows: false,
  }
  visitObject(scene, camera, meshes, 0, viewportHeight, clippingContext, localClippingEnabled, shadowMaterialMode, materialContext)
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
): void {
  if (!object || object.visible === false) return

  const nextGroupOrder = object.isGroup === true
    ? finiteMaterialOrObjectNumber(object.renderOrder, 'object.renderOrder', 0)
    : groupOrder
  const visibleToCamera = objectLayersMatchCamera(object, camera)
  const nextClippingContext = visibleToCamera
    ? clippingContextForObject(clippingContext, object)
    : clippingContext
  if (visibleToCamera) {
    updateLodObject(object, camera)

    if (object.isMesh === true && object.geometry) {
      appendMesh(object, camera, meshes, nextGroupOrder, nextClippingContext, localClippingEnabled, shadowMaterialMode, materialContext)
    } else if ((object.isLineSegments === true || object.isLineLoop === true || object.isLine === true) && object.geometry) {
      appendLineOrPoints(object, camera, meshes, 'lines', nextGroupOrder, viewportHeight, nextClippingContext, localClippingEnabled, materialContext)
    } else if (object.isPoints === true && object.geometry) {
      appendPoints(object, camera, meshes, nextGroupOrder, viewportHeight, nextClippingContext, localClippingEnabled, shadowMaterialMode, materialContext)
    } else if (object.isSprite === true) {
      appendSprite(object, camera, meshes, nextGroupOrder, nextClippingContext, localClippingEnabled, shadowMaterialMode, materialContext)
    }
  }

  const children = Array.isArray(object.children) ? object.children : []
  for (const child of children) {
    visitObject(child, camera, meshes, nextGroupOrder, viewportHeight, nextClippingContext, localClippingEnabled, shadowMaterialMode, materialContext)
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
): void {
  const geometry = object.geometry!
  const position = getAttribute(geometry, 'position')
  if (!position) return

  let positions = readVec3Attribute(position, 'geometry.attributes.position')
  const uvAttribute = getAttribute(geometry, 'uv')
  const uvs = uvAttribute ? readVec2Attribute(uvAttribute, 'geometry.attributes.uv') : null
  const uvChannels = readUvChannels(geometry, uvs)
  const normalAttribute = getAttribute(geometry, 'normal')
  let normals = normalAttribute ? readVec3Attribute(normalAttribute, 'geometry.attributes.normal') : null
  const vertexColors = getAttribute(geometry, 'color')
  const index = geometry.index ? readIndexAttribute(geometry.index, 'geometry.index') : null
  const groups = effectiveGroups(geometry, index, position.count)
  const instancedGeometryCount = instancedBufferGeometryCount(geometry)
  const instancedPositionOffset = instancedOffsetAttribute(geometry)

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
  const meshTransform = isSkinned
    ? IDENTITY_4X4.slice()
    : matrixElements(object.matrixWorld!, 'mesh.matrixWorld')
  const instances = meshInstances(object, meshTransform)
  if (instances.length === 0) return
  const objectCastsShadow = optionalObjectBoolean(object.castShadow, 'object.castShadow') === true
  const objectReceivesShadow = optionalObjectBoolean(object.receiveShadow, 'object.receiveShadow') === true

  for (const group of groups) {
    const material = materialForGroup(object.material, group.materialIndex)
    if (material?.visible === false) continue

    const customShadowMaterial = customShadowMaterialForMode(object, shadowMaterialMode)
    const usesCustomShadowMaterial = objectCastsShadow && customShadowMaterial != null
    const baseColor = materialColor(material)
    const useVertexColors = vertexColors && material?.vertexColors !== false
    const pbrProps = extractPbrProperties(material, materialContext)
    const secondaryUvs = secondaryUvsForMaterial(uvChannels, material)
    const textureInfo = extractTextureData(material)
    const castShadow = objectCastsShadow && !usesCustomShadowMaterial ? true : undefined
    const receiveShadow = objectReceivesShadow ? true : undefined
    const clipping = clippingState(clippingContext, material, localClippingEnabled)
    const wireframe = isDepthDistanceWireframeMaterial(material)

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
      const expandedNormals = normals
        ? expandVec3ValuesForInstances(normals, 0, position.count, instancedGeometryCount)
        : undefined
      const expandedUvs = uvs
        ? expandVec2ValuesForInstances(uvs, 0, position.count, instancedGeometryCount)
        : undefined
      const expandedSecondaryUvs = secondaryUvs
        ? expandVec2ValuesForInstances(secondaryUvs, 0, position.count, instancedGeometryCount)
        : undefined

      for (const instance of instances) {
        const color = instanceColor(baseColor, instance)
        const sortInfo = sortInfoForObject(object, material, camera, meshes.length, groupOrder, instance.transform)
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
          textureUsesUv2: textureInfo?.usesUv2,
          transform: instance.transform,
          topology: wireframe ? 'lines' : undefined,
          castShadow,
          receiveShadow,
          clipShadows: clipShadowsForMaterial(material, clippingContext),
          ...clipping,
          ...sortInfo.keys,
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
      const expandedGroupNormals = normals
        ? expandVec3ValuesForInstances(normals, group.start, group.count, instancedGeometryCount)
        : undefined
      const expandedGroupUvs = uvs
        ? expandVec2ValuesForInstances(uvs, group.start, group.count, instancedGeometryCount)
        : undefined
      const expandedGroupSecondaryUvs = secondaryUvs
        ? expandVec2ValuesForInstances(secondaryUvs, group.start, group.count, instancedGeometryCount)
        : undefined
      const expandedGroupIndices = wireframe
        ? expandIndicesForInstances(wireframeIndicesForUnindexedTriangles(group.count), group.count, instancedGeometryCount)
        : undefined
      for (const instance of instances) {
        const color = instanceColor(baseColor, instance)
        const sortInfo = sortInfoForObject(object, material, camera, meshes.length, groupOrder, instance.transform)
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
          textureUsesUv2: textureInfo?.usesUv2,
          transform: instance.transform,
          topology: wireframe ? 'lines' : undefined,
          castShadow,
          receiveShadow,
          clipShadows: clipShadowsForMaterial(material, clippingContext),
          ...clipping,
          ...sortInfo.keys,
          ...pbrProps,
        }, sortInfo.item)
      }
    }

    if (usesCustomShadowMaterial && customShadowMaterial?.visible !== false) {
      assertSupportedCustomShadowMaterial(customShadowMaterial, shadowMaterialMode)
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
  }
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
  uvs: number[] | null,
  uvChannels: Array<number[] | null>,
  vertexColors: ThreeBufferAttributeLike | undefined,
  vertexCount: number,
  index: number[] | null,
  instancedGeometryCount: number,
  instancedPositionOffset: ThreeBufferAttributeLike | null,
  instances: MeshInstance[],
): void {
  const baseColor = materialColor(material)
  const useVertexColors = vertexColors && material.vertexColors !== false
  const pbrProps = shadowPbrProperties(material, sourceMaterial, materialContext)
  const secondaryUvs = secondaryUvsForMaterial(uvChannels, material)
  const textureInfo = extractTextureData(material)
  const clipping = clippingState(clippingContext, material, localClippingEnabled)
  const wireframe = isDepthDistanceWireframeMaterial(material)
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
    const expandedNormals = normals
      ? expandVec3ValuesForInstances(normals, 0, vertexCount, instancedGeometryCount)
      : undefined
    const expandedUvs = uvs
      ? expandVec2ValuesForInstances(uvs, 0, vertexCount, instancedGeometryCount)
      : undefined
    const expandedSecondaryUvs = secondaryUvs
      ? expandVec2ValuesForInstances(secondaryUvs, 0, vertexCount, instancedGeometryCount)
      : undefined

    for (const instance of instances) {
      const color = instanceColor(baseColor, instance)
      const sortInfo = sortInfoForObject(object, material, camera, meshes.length, groupOrder, instance.transform)
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
        textureUsesUv2: textureInfo?.usesUv2,
        transform: instance.transform,
        topology: wireframe ? 'lines' : undefined,
        castShadow: true,
        receiveShadow: false,
        clipShadows: clipShadowsForMaterial(material, clippingContext),
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
  const expandedGroupNormals = normals
    ? expandVec3ValuesForInstances(normals, group.start, group.count, instancedGeometryCount)
    : undefined
  const expandedGroupUvs = uvs
    ? expandVec2ValuesForInstances(uvs, group.start, group.count, instancedGeometryCount)
    : undefined
  const expandedGroupSecondaryUvs = secondaryUvs
    ? expandVec2ValuesForInstances(secondaryUvs, group.start, group.count, instancedGeometryCount)
    : undefined
  const expandedGroupIndices = wireframe
    ? expandIndicesForInstances(wireframeIndicesForUnindexedTriangles(group.count), group.count, instancedGeometryCount)
    : undefined

  for (const instance of instances) {
    const color = instanceColor(baseColor, instance)
    const sortInfo = sortInfoForObject(object, material, camera, meshes.length, groupOrder, instance.transform)
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
      textureUsesUv2: textureInfo?.usesUv2,
      transform: instance.transform,
      topology: wireframe ? 'lines' : undefined,
      castShadow: true,
      receiveShadow: false,
      clipShadows: clipShadowsForMaterial(material, clippingContext),
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
  if (mode === 'depth') return object.customDepthMaterial
  if (mode === 'distance') return object.customDistanceMaterial
  return undefined
}

function assertSupportedCustomShadowMaterial(
  material: ThreeMaterialLike,
  mode: ShadowMaterialMode | undefined,
): void {
  if (!isDepthDistanceWireframeMaterial(material)) return
  const property = mode === 'distance' ? 'customDistanceMaterial' : 'customDepthMaterial'
  throw new Error(
    `Object3D.${property} wireframe shadow casters are not supported by @headless-three/renderer yet. Disable wireframe on the custom shadow material or expand the intended shadow shape to mesh geometry before rendering.`,
  )
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
): void {
  const objectCastsShadow = optionalObjectBoolean(object.castShadow, 'object.castShadow') === true
  const objectReceivesShadow = optionalObjectBoolean(object.receiveShadow, 'object.receiveShadow') === true
  assertUnsupportedSpriteShadows(objectReceivesShadow)

  const material = materialForGroup(object.material, 0)
  if (material?.visible === false) return

  const matrix = matrixElements(object.matrixWorld!, 'sprite.matrixWorld')
  const center = [
    finiteMaterialOrObjectNumber(object.center?.x, 'Sprite.center.x', 0.5),
    finiteMaterialOrObjectNumber(object.center?.y, 'Sprite.center.y', 0.5),
  ]
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

  const textureInfo = extractTextureData(material)
  const sortInfo = sortInfoForObject(object, material, camera, meshes.length, groupOrder)
  const clipping = clippingState(clippingContext, material, localClippingEnabled)
  const customShadowMaterial = customShadowMaterialForMode(object, shadowMaterialMode)
  const usesCustomShadowMaterial = objectCastsShadow && customShadowMaterial != null

  pushMesh(meshes, {
    positions,
    indices: [0, 1, 2, 0, 2, 3],
    uvs,
    color: materialColor(material),
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
    textureUsesUv2: textureInfo?.usesUv2,
    transform: IDENTITY_4X4.slice(),
    transparent: material?.transparent !== false,
    castShadow: objectCastsShadow && !usesCustomShadowMaterial ? true : undefined,
    receiveShadow: undefined,
    clipShadows: clipShadowsForMaterial(material, clippingContext),
    ...clipping,
    ...sortInfo.keys,
    ...extractPbrProperties(material, materialContext),
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
      [0, 1, 2, 0, 2, 3],
      uvs,
    )
  }
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
): void {
  const objectCastsShadow = optionalObjectBoolean(object.castShadow, 'object.castShadow') === true
  const objectReceivesShadow = optionalObjectBoolean(object.receiveShadow, 'object.receiveShadow') === true
  assertUnsupportedPointShadows(objectReceivesShadow)

  const geometry = object.geometry!
  const position = getAttribute(geometry, 'position')
  if (!position) return

  const positions = readVec3Attribute(position, 'geometry.attributes.position')
  const vertexColors = getAttribute(geometry, 'color')
  const index = geometry.index ? readIndexAttribute(geometry.index, 'geometry.index') : null
  const groups = effectiveGroups(geometry, index, position.count)
  const instancedGeometryCount = instancedBufferGeometryCount(geometry)
  const instancedPositionOffset = instancedOffsetAttribute(geometry)
  const transform = matrixElements(object.matrixWorld!, 'points.matrixWorld')
  const axes = cameraBillboardAxes(camera)

  for (const group of groups) {
    const material = materialForGroup(object.material, group.materialIndex)
    if (material?.visible === false) continue

    const baseColor = materialColor(material)
    const useVertexColors = vertexColors && material?.vertexColors !== false
    const source = index ?? rangeIndices(position.count)
    const points = source.slice(group.start, group.start + group.count)
    if (points.length === 0) continue

    const outputPositions: number[] = []
    const outputUvs: number[] = []
    const outputColors: number[] | undefined = useVertexColors ? [] : undefined
    const outputIndices: number[] = []
    const pointSize = Math.max(0, finiteMaterialOrObjectNumber(material?.size, 'material.size', 1))
    if (pointSize <= 0) continue

    for (let instance = 0; instance < instancedGeometryCount; instance += 1) {
      const offsetIndex = instancedPositionOffset ? instancedAttributeIndex(instancedPositionOffset, instance) : 0
      const offset = instancedPositionOffset
        ? [
          attributeComponent(instancedPositionOffset, offsetIndex, 0),
          attributeComponent(instancedPositionOffset, offsetIndex, 1),
          attributeComponent(instancedPositionOffset, offsetIndex, 2),
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
        const worldSize = pointWorldSize(pointSize, center, material, camera, viewportHeight)
        if (worldSize <= 0) continue

        const vertexBase = outputPositions.length / 3
        const corners = [
          [-0.5, -0.5, 0, 0],
          [0.5, -0.5, 1, 0],
          [0.5, 0.5, 1, 1],
          [-0.5, 0.5, 0, 1],
        ]
        const pointColor = outputColors ? pointVertexColor(vertexColors!, baseColor, pointIndex, instance) : null
        for (const [x, y, u, v] of corners) {
          outputPositions.push(
            center[0] + axes.right[0] * x * worldSize + axes.up[0] * y * worldSize,
            center[1] + axes.right[1] * x * worldSize + axes.up[1] * y * worldSize,
            center[2] + axes.right[2] * x * worldSize + axes.up[2] * y * worldSize,
          )
          outputUvs.push(u, v)
          if (pointColor) {
            outputColors!.push(pointColor[0], pointColor[1], pointColor[2], pointColor[3])
          }
        }
        outputIndices.push(vertexBase, vertexBase + 1, vertexBase + 2, vertexBase, vertexBase + 2, vertexBase + 3)
      }
    }

    if (outputPositions.length === 0) continue

    const textureInfo = extractTextureData(material)
    const sortInfo = sortInfoForObject(object, material, camera, meshes.length, groupOrder)
    const pbrProps = extractPbrProperties(material, materialContext)
    const clipping = clippingState(clippingContext, material, localClippingEnabled)
    const customShadowMaterial = customShadowMaterialForMode(object, shadowMaterialMode)
    const usesCustomShadowMaterial = objectCastsShadow && customShadowMaterial != null

    pushMesh(meshes, {
      positions: outputPositions,
      indices: outputIndices,
      uvs: outputUvs,
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
      textureUsesUv2: textureInfo?.usesUv2,
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
        outputUvs,
      )
    }
  }
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
): void {
  const textureInfo = extractTextureData(material)
  const sortInfo = sortInfoForObject(object, material, camera, meshes.length, groupOrder)
  const clipping = clippingState(clippingContext, material, localClippingEnabled)
  const hiddenMainPass = shadowOnlyMainPassState()

  pushMesh(meshes, {
    positions,
    indices,
    uvs,
    color: materialColor(material),
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
    textureUsesUv2: textureInfo?.usesUv2,
    transform: IDENTITY_4X4.slice(),
    topology: 'triangles',
    castShadow: true,
    receiveShadow: false,
    clipShadows: clipShadowsForMaterial(material, clippingContext),
    ...clipping,
    ...sortInfo.keys,
    ...shadowPbrProperties(material, sourceMaterial, materialContext),
    ...hiddenMainPass,
  }, sortInfo.item)
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

function assertUnsupportedSpriteShadows(receiveShadow: boolean): void {
  if (receiveShadow) {
    throw new Error('THREE.Sprite receiveShadow is not supported by @headless-three/renderer yet. Disable receiveShadow or expand the sprite to mesh geometry before receiving shadows.')
  }
}

function assertUnsupportedPointShadows(receiveShadow: boolean): void {
  if (receiveShadow) {
    throw new Error('THREE.Points receiveShadow is not supported by @headless-three/renderer yet. Disable receiveShadow or expand the points to mesh geometry before receiving shadows.')
  }
}

function pointVertexColor(
  attribute: ThreeBufferAttributeLike,
  materialColor: Color4,
  pointIndex: number,
  instanceIndex: number,
): Color4 {
  const sourceIndex = isInstancedAttribute(attribute)
    ? instancedAttributeIndex(attribute, instanceIndex)
    : pointIndex
  return [
    clamp01(attributeComponent(attribute, sourceIndex, 0) * materialColor[0]),
    clamp01(attributeComponent(attribute, sourceIndex, 1) * materialColor[1]),
    clamp01(attributeComponent(attribute, sourceIndex, 2) * materialColor[2]),
    clamp01((attribute.itemSize && attribute.itemSize >= 4 ? attributeComponent(attribute, sourceIndex, 3) : 1) * materialColor[3]),
  ]
}

function effectiveGroups(
  geometry: ThreeBufferGeometryLike,
  index: number[] | null,
  vertexCount: number,
): GeometryGroup[] {
  const range = geometry.drawRange ?? {}
  const maxCount = index ? index.length : vertexCount
  const drawStart = clampInteger(range.start ?? 0, 0, maxCount)
  const requestedCount = range.count == null || range.count === Infinity ? maxCount : range.count
  const drawEnd = clampInteger(drawStart + requestedCount, drawStart, maxCount)
  const sourceGroups = Array.isArray(geometry.groups) && geometry.groups.length
    ? geometry.groups
    : [{ start: drawStart, count: drawEnd - drawStart, materialIndex: 0 }]

  const groups: GeometryGroup[] = []
  for (const group of sourceGroups) {
    const start = Math.max(drawStart, clampInteger(group.start ?? 0, 0, maxCount))
    const end = Math.min(drawEnd, clampInteger((group.start ?? 0) + (group.count ?? 0), 0, maxCount))
    if (end > start) {
      groups.push({
        start,
        count: end - start,
        materialIndex: group.materialIndex ?? 0,
      })
    }
  }
  return groups
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
): void {
  validateObjectShadowFlags(object)
  const geometry = object.geometry!
  const position = getAttribute(geometry, 'position')
  if (!position) return

  const positions = readVec3Attribute(position, 'geometry.attributes.position')
  const uvAttribute = getAttribute(geometry, 'uv')
  const uvs = uvAttribute ? readVec2Attribute(uvAttribute, 'geometry.attributes.uv') : null
  const uvChannels = readUvChannels(geometry, uvs)
  const vertexColors = getAttribute(geometry, 'color')
  const indexAttr = geometry.index ? readIndexAttribute(geometry.index, 'geometry.index') : null
  const vertexCount = position.count
  const instancedGeometryCount = instancedBufferGeometryCount(geometry)
  const instancedPositionOffset = instancedOffsetAttribute(geometry)
  const groups = effectiveGroups(geometry, indexAttr, vertexCount)

  for (const group of groups) {
    const material = materialForGroup(object.material, group.materialIndex)
    if (material?.visible === false) continue

    const secondaryUvs = secondaryUvsForMaterial(uvChannels, material)
    let indices: number[] | null = null
    let outputPositions = positions
    let outputUvs: number[] | undefined = topology === 'lines' ? uvs ?? undefined : undefined
    let outputSecondaryUvs: number[] | undefined = topology === 'lines' ? secondaryUvs ?? undefined : undefined
    let outputColors: number[] | undefined
    let thickCenter: [number, number, number] | undefined
    const color = materialColor(material)
    const useVertexColors = vertexColors && material?.vertexColors !== false
    const pbrProps = extractPbrProperties(material, materialContext)
    const textureInfo = extractTextureData(material)
    const drawStart = group.start
    const drawEnd = group.start + group.count
    const lineWidth = finiteMaterialOrObjectNumber(material?.linewidth, 'material.linewidth', 1)
    const thickLine = topology === 'lines' && lineWidth > 1

    if (topology === 'lines') {
      const source = indexAttr ?? rangeIndices(vertexCount)
      if (material?.isLineDashedMaterial === true) {
        const dashed = instancedGeometryCount > 1 || instancedPositionOffset
          ? dashedLineAttributesForInstances(
            positions,
            uvs,
            secondaryUvs,
            useVertexColors ? vertexColors! : undefined,
            color,
            source,
            drawStart,
            drawEnd,
            object,
            getAttribute(geometry, 'lineDistance'),
            material,
            instancedGeometryCount,
            instancedPositionOffset,
          )
          : dashedLineAttributes(
            positions,
            uvs,
            secondaryUvs,
            useVertexColors ? readColorAttribute(vertexColors!, color, 'geometry.attributes.color') : undefined,
            source,
            drawStart,
            drawEnd,
            object,
            getAttribute(geometry, 'lineDistance'),
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
          outputUvs = uvs ? expandVec2ValuesForInstances(uvs, 0, vertexCount, instancedGeometryCount) : undefined
          outputSecondaryUvs = secondaryUvs ? expandVec2ValuesForInstances(secondaryUvs, 0, vertexCount, instancedGeometryCount) : undefined
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
    const sortInfo = sortInfoForObject(object, material, camera, meshes.length, groupOrder)
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
      textureUsesUv2: textureInfo?.usesUv2,
      transform: thickLine ? IDENTITY_4X4.slice() : matrixElements(object.matrixWorld!, 'object.matrixWorld'),
      transparent: material?.transparent === true || (material?.opacity != null && material.opacity < 1),
      alphaTest: material && Number.isFinite(material.alphaTest) && material.alphaTest! > 0 ? material.alphaTest : undefined,
      clipShadows: clipShadowsForMaterial(material, clippingContext),
      ...pbrProps,
      ...(thickLine ? { side: 'double' } : {}),
      shadingModel: 'basic',
      topology: thickLine ? 'triangles' : topology,
      ...clipping,
      ...sortInfo.keys,
    }, sortInfo.item)
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
  const localUnionPlanes = material?.clipIntersection === true ? [] : localPlanes
  const localIntersectionPlanes = material?.clipIntersection === true ? localPlanes : []
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
  if (object.isClippingGroup !== true || object.enabled === false) return parent
  const currentCount = parent.unionPlanes.length + parent.intersectionPlanes.length
  const remainingBudget = Math.max(0, MAX_CLIPPING_PLANES - currentCount)
  const planes = extractClippingPlanes(object.clippingPlanes, 'ClippingGroup.clippingPlanes', remainingBudget)
  if (planes.length === 0) return parent

  return object.clipIntersection === true
    ? {
      unionPlanes: parent.unionPlanes,
      intersectionPlanes: [...parent.intersectionPlanes, ...planes],
      clipShadows: parent.clipShadows || object.clipShadows === true,
    }
    : {
      unionPlanes: [...parent.unionPlanes, ...planes],
      intersectionPlanes: parent.intersectionPlanes,
      clipShadows: parent.clipShadows || object.clipShadows === true,
    }
}

function clipShadowsForMaterial(material: ThreeMaterialLike | undefined, clippingContext: ClippingContext): boolean | undefined {
  return material?.clipShadows === true || clippingContext.clipShadows ? true : undefined
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
): MeshSortInfo {
  const renderOrder = finiteMaterialOrObjectNumber(object.renderOrder, 'object.renderOrder', 0)
  const z = camera ? projectedObjectZ(object, camera, transform) : 0
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
      object,
      material,
      groupOrder,
      renderOrder,
      z,
      materialVariant,
    },
  }
}

function sortFlattenedMeshes(meshes: FlattenedMesh[], options: SceneSortOptions): FlattenedMesh[] {
  const sortObjects = options.sortObjects !== false
  const buckets = partitionFlattenedMeshes(meshes)

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
  const geometry = object.geometry
  if (!geometry) return [0, 0, 0]

  if (geometry.boundingSphere == null && typeof geometry.computeBoundingSphere === 'function') {
    try {
      geometry.computeBoundingSphere()
    } catch {
      return [0, 0, 0]
    }
  }

  return vec3Like(geometry.boundingSphere?.center) ?? [0, 0, 0]
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
  material: { sizeAttenuation?: boolean } | undefined,
  camera: ThreeCameraLike | undefined,
  viewportHeight: number,
): number {
  const projectionY = Math.abs(finiteOrDefault(camera?.projectionMatrix?.elements?.[5], 1))
  if (projectionY <= 0) return 0

  const sizeAttenuation = optionalSceneBoolean(material?.sizeAttenuation, 'material.sizeAttenuation')
  if (camera?.isPerspectiveCamera === true && sizeAttenuation !== false) {
    return pointSize / projectionY
  }

  if (camera?.isPerspectiveCamera !== true) {
    return pointSize * 2 / Math.max(1, viewportHeight) / projectionY
  }

  const viewZ = camera ? viewSpaceZ(worldPosition, camera) : -1
  const depth = Number.isFinite(viewZ) ? Math.max(0.0001, Math.abs(viewZ)) : 1
  return pointSize * 2 * depth / Math.max(1, viewportHeight) / projectionY
}

function finiteOrDefault(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function finiteMaterialOrObjectNumber(value: unknown, label: string, fallback: number): number {
  if (value == null) return fallback
  if (typeof value === 'number' && Number.isFinite(value)) return value
  throw new TypeError(`${label} must be a finite number.`)
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

function finiteCountOrDefault(value: unknown, label: string, fallback: number): number {
  if (value == null) return fallback
  if (typeof value === 'number' && Number.isFinite(value)) return value
  throw new TypeError(`${label} must be a finite number.`)
}

function unsignedSortKey(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? value : fallback
}

function instancedBufferGeometryCount(geometry: ThreeBufferGeometryLike): number {
  const attributes = Object.entries(geometry.attributes ?? {})
  const instancedAttributes = attributes.filter((entry): entry is [string, ThreeBufferAttributeLike] => isInstancedAttribute(entry[1]))
  if (geometry.isInstancedBufferGeometry !== true && instancedAttributes.length === 0) return 1

  let maxCount = Infinity
  for (const [name, attribute] of instancedAttributes) {
    maxCount = Math.min(maxCount, attribute.count * meshPerAttribute(attribute, `geometry.attributes.${name}.meshPerAttribute`))
  }

  const requested = finiteCountOrDefault(geometry.instanceCount, 'geometry.instanceCount', Infinity)
  const effectiveCount = Math.min(requested, maxCount)
  if (effectiveCount === Infinity) return 1
  return clampInteger(Math.floor(effectiveCount), 0, Math.max(0, Math.floor(maxCount)))
}

function isInstancedAttribute(attribute: ThreeBufferAttributeLike | undefined | null): attribute is ThreeBufferAttributeLike {
  return attribute?.isInstancedBufferAttribute === true
}

function meshPerAttribute(attribute: ThreeBufferAttributeLike, label = 'InstancedBufferAttribute.meshPerAttribute'): number {
  const value = attribute.meshPerAttribute
  if (value == null) return 1
  if (typeof value === 'number' && Number.isFinite(value) && value > 0) {
    return Math.max(1, Math.floor(value))
  }
  throw new TypeError(`${label} must be a positive finite number.`)
}

function instancedAttributeIndex(attribute: ThreeBufferAttributeLike, instanceIndex: number): number {
  return Math.min(attribute.count - 1, Math.floor(instanceIndex / meshPerAttribute(attribute)))
}

function instancedOffsetAttribute(geometry: ThreeBufferGeometryLike): ThreeBufferAttributeLike | null {
  const names = ['instanceOffset', 'instancePosition', 'offset', 'translate', 'translation']
  for (const name of names) {
    const attribute = getAttribute(geometry, name)
    if (isInstancedAttribute(attribute)) return attribute
  }
  return null
}

function expandVec3ValuesForInstances(
  values: number[],
  start: number,
  count: number,
  instanceCount: number,
  offsetAttribute?: ThreeBufferAttributeLike | null,
): number[] {
  if (instanceCount <= 1 && !offsetAttribute) {
    return values.slice(start * 3, (start + count) * 3)
  }
  const out = new Array<number>(count * instanceCount * 3)
  let dst = 0
  for (let instance = 0; instance < instanceCount; instance += 1) {
    const offsetIndex = offsetAttribute ? instancedAttributeIndex(offsetAttribute, instance) : 0
    const ox = offsetAttribute ? attributeComponent(offsetAttribute, offsetIndex, 0) : 0
    const oy = offsetAttribute ? attributeComponent(offsetAttribute, offsetIndex, 1) : 0
    const oz = offsetAttribute ? attributeComponent(offsetAttribute, offsetIndex, 2) : 0
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

function expandColorAttributeForInstances(
  attribute: ThreeBufferAttributeLike,
  materialColor: Color4,
  start: number,
  count: number,
  instanceCount: number,
): number[] {
  if (!isInstancedAttribute(attribute)) {
    const colors = readColorAttribute(attribute, materialColor, 'geometry.attributes.color')
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
    const sourceIndex = instancedAttributeIndex(attribute, instance)
    const r = clamp01(attributeComponent(attribute, sourceIndex, 0) * materialColor[0])
    const g = clamp01(attributeComponent(attribute, sourceIndex, 1) * materialColor[1])
    const b = clamp01(attributeComponent(attribute, sourceIndex, 2) * materialColor[2])
    const a = clamp01((itemSize >= 4 ? attributeComponent(attribute, sourceIndex, 3) : 1) * materialColor[3])
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

function readUvChannels(geometry: ThreeBufferGeometryLike, primaryUvs: number[] | null): Array<number[] | null> {
  return [
    primaryUvs,
    readOptionalUvAttribute(geometry, 'uv1') ?? readOptionalUvAttribute(geometry, 'uv2') ?? primaryUvs,
    readOptionalUvAttribute(geometry, 'uv2') ?? readOptionalUvAttribute(geometry, 'uv1') ?? primaryUvs,
    readOptionalUvAttribute(geometry, 'uv3') ?? primaryUvs,
  ]
}

function readOptionalUvAttribute(geometry: ThreeBufferGeometryLike, name: string): number[] | null {
  const attribute = getAttribute(geometry, name)
  return attribute ? readVec2Attribute(attribute, `geometry.attributes.${name}`) : null
}

function secondaryUvsForMaterial(
  channels: Array<number[] | null>,
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
): number[] | null {
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
  if (object.isLOD !== true || !camera || object.autoUpdate === false) return

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
  if (!Array.isArray(levels)) return []
  return levels.map((level, index) => ({
    object: (level as { object?: ThreeObject3DLike }).object!,
    distance: finiteMaterialOrObjectNumber((level as { distance?: unknown }).distance, `LOD.levels[${index}].distance`, 0),
    hysteresis: finiteMaterialOrObjectNumber((level as { hysteresis?: unknown }).hysteresis, `LOD.levels[${index}].hysteresis`, 0),
  }))
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
  const dashSize = Math.max(0, finiteMaterialOrObjectNumber(material.dashSize, 'material.dashSize', 3))
  const gapSize = Math.max(0, finiteMaterialOrObjectNumber(material.gapSize, 'material.gapSize', 1))
  const scale = finiteMaterialOrObjectNumber(material.scale, 'material.scale', 1)
  if (dashSize <= 0) return { positions: [] }

  const segments = lineSegmentsWithDistances(positions, source, start, end, object, lineDistance)
  const out = createDashedLineExpansion(uvs, uvs2, colors)
  if (scale <= 0 || gapSize <= 0) {
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

function dashedLineAttributesForInstances(
  positions: number[],
  uvs: number[] | null,
  uvs2: number[] | null,
  vertexColors: ThreeBufferAttributeLike | undefined,
  materialColor: Color4,
  source: number[],
  start: number,
  end: number,
  object: ThreeObject3DLike,
  lineDistance: ThreeBufferAttributeLike | undefined,
  material: { dashSize?: number; gapSize?: number; scale?: number },
  instanceCount: number,
  offsetAttribute: ThreeBufferAttributeLike | null,
): DashedLineExpansion {
  const out: DashedLineExpansion = {
    positions: [],
    uvs: uvs ? [] : undefined,
    uvs2: uvs2 ? [] : undefined,
    colors: vertexColors ? [] : undefined,
  }
  const baseColors = vertexColors && !isInstancedAttribute(vertexColors)
    ? readColorAttribute(vertexColors, materialColor, 'geometry.attributes.color')
    : undefined

  for (let instance = 0; instance < instanceCount; instance += 1) {
    const instancePositions = offsetAttribute
      ? offsetVec3ValuesForInstance(positions, offsetAttribute, instance)
      : positions
    const instanceColors = vertexColors
      ? baseColors ?? repeatedInstancedColorValues(vertexColors, materialColor, positions.length / 3, instance)
      : undefined
    const dashed = dashedLineAttributes(
      instancePositions,
      uvs,
      uvs2,
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
  offsetAttribute: ThreeBufferAttributeLike,
  instance: number,
): number[] {
  const offsetIndex = instancedAttributeIndex(offsetAttribute, instance)
  const ox = attributeComponent(offsetAttribute, offsetIndex, 0)
  const oy = attributeComponent(offsetAttribute, offsetIndex, 1)
  const oz = attributeComponent(offsetAttribute, offsetIndex, 2)
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
): number[] {
  const sourceIndex = instancedAttributeIndex(attribute, instance)
  const itemSize = attribute.itemSize ?? 3
  const color = [
    clamp01(attributeComponent(attribute, sourceIndex, 0) * materialColor[0]),
    clamp01(attributeComponent(attribute, sourceIndex, 1) * materialColor[1]),
    clamp01(attributeComponent(attribute, sourceIndex, 2) * materialColor[2]),
    clamp01((itemSize >= 4 ? attributeComponent(attribute, sourceIndex, 3) : 1) * materialColor[3]),
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
      const d0 = lineDistance ? attributeComponent(lineDistance, a, 0) : cumulative
      const d1 = lineDistance ? attributeComponent(lineDistance, b, 0) : d0 + length
      segments.push({ a, b, d0, d1 })
      cumulative = d1
    }
    return segments
  }

  let previous = source[start]
  let previousDistance = lineDistance ? attributeComponent(lineDistance, previous, 0) : 0
  for (let i = 1; i < count; i += 1) {
    const current = source[start + i]
    const length = vertexDistance(positions, previous, current)
    const currentDistance = lineDistance ? attributeComponent(lineDistance, current, 0) : previousDistance + length
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
      d1: previousDistance + vertexDistance(positions, previous, first),
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
  if (span <= 1e-6) return

  let cursor = s0
  let guard = 0
  while (cursor < s1 - 1e-6 && guard < 10000) {
    guard += 1
    const cycle = Math.floor(cursor / totalSize)
    const cycleStart = cycle * totalSize
    const inCycle = cursor - cycleStart
    const visible = inCycle <= dashSize
    const boundary = cycleStart + (visible ? dashSize : totalSize)
    const next = Math.min(s1, boundary <= cursor + 1e-6 ? cursor + 1e-6 : boundary)
    if (visible && next > cursor + 1e-6) {
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

  const count = clampInteger(
    finiteCountOrDefault(object.count, 'InstancedMesh.count', instanceMatrix.count),
    0,
    instanceMatrix.count,
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
    matrix[component] = attributeComponent(attribute, index, component)
  }
  return matrix
}

function readInstanceColor(attribute: ThreeObject3DLike['instanceColor'], index: number): Color4 | undefined {
  if (!attribute || index >= attribute.count) return undefined
  return [
    attributeComponent(attribute, index, 0),
    attributeComponent(attribute, index, 1),
    attributeComponent(attribute, index, 2),
    attribute.itemSize && attribute.itemSize >= 4 ? attributeComponent(attribute, index, 3) : 1,
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
