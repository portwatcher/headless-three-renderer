import type {
  ThreeObject3DLike,
  ThreeBufferAttributeLike,
  ThreeBufferGeometryLike,
  ThreeCameraLike,
  ThreeMaterialLike,
  NativeSceneMesh,
  GeometryGroup,
  Color4,
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
import { materialForGroup, materialColor, extractPbrProperties, extractTextureData } from './materials'
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
  groupOrder: number
  renderOrder: number
  sortZ: number
  materialSortKey: number
  sortIndex: number
}

interface MeshInstance {
  transform: number[]
  color?: Color4
}

interface LineSegmentDistance {
  a: number
  b: number
  d0: number
  d1: number
}

interface DashedLineExpansion {
  positions: number[]
  uvs?: number[]
  colors?: number[]
}

export function flattenScene(
  scene: ThreeObject3DLike,
  camera?: ThreeCameraLike,
  viewportHeight = 512,
  globalClippingPlanes: readonly NativeClippingPlane[] = [],
): NativeSceneMesh[] {
  const meshes: FlattenedMesh[] = []
  visitObject(scene, camera, meshes, 0, viewportHeight, globalClippingPlanes)
  return meshes
    .sort(compareFlattenedMeshes)
    .map(({ mesh }) => mesh)
}

function visitObject(
  object: ThreeObject3DLike,
  camera: ThreeCameraLike | undefined,
  meshes: FlattenedMesh[],
  groupOrder: number,
  viewportHeight: number,
  globalClippingPlanes: readonly NativeClippingPlane[],
): void {
  if (!object || object.visible === false) return

  const nextGroupOrder = object.isGroup === true
    ? finiteOrDefault(object.renderOrder, 0)
    : groupOrder
  const visibleToCamera = objectLayersMatchCamera(object, camera)
  if (visibleToCamera) {
    updateLodObject(object, camera)

    if (object.isMesh === true && object.geometry) {
      appendMesh(object, camera, meshes, nextGroupOrder, globalClippingPlanes)
    } else if ((object.isLineSegments === true || object.isLineLoop === true || object.isLine === true) && object.geometry) {
      appendLineOrPoints(object, camera, meshes, 'lines', nextGroupOrder, globalClippingPlanes)
    } else if (object.isPoints === true && object.geometry) {
      appendPoints(object, camera, meshes, nextGroupOrder, viewportHeight, globalClippingPlanes)
    } else if (object.isSprite === true) {
      appendSprite(object, camera, meshes, nextGroupOrder, globalClippingPlanes)
    }
  }

  const children = Array.isArray(object.children) ? object.children : []
  for (const child of children) {
    visitObject(child, camera, meshes, nextGroupOrder, viewportHeight, globalClippingPlanes)
  }
}

function appendMesh(
  object: ThreeObject3DLike,
  camera: ThreeCameraLike | undefined,
  meshes: FlattenedMesh[],
  groupOrder: number,
  globalClippingPlanes: readonly NativeClippingPlane[],
): void {
  const geometry = object.geometry!
  const position = getAttribute(geometry, 'position')
  if (!position) return

  let positions = readVec3Attribute(position)
  const uvAttribute = getAttribute(geometry, 'uv')
  const uvs = uvAttribute ? readVec2Attribute(uvAttribute) : null
  const uvChannels = readUvChannels(geometry, uvs)
  const normalAttribute = getAttribute(geometry, 'normal')
  let normals = normalAttribute ? readVec3Attribute(normalAttribute) : null
  const vertexColors = getAttribute(geometry, 'color')
  const index = geometry.index ? readIndexAttribute(geometry.index) : null
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

  for (const group of groups) {
    const material = materialForGroup(object.material, group.materialIndex)
    if (material?.visible === false) continue

    const baseColor = materialColor(material)
    const useVertexColors = vertexColors && material?.vertexColors !== false
    const pbrProps = extractPbrProperties(material)
    const secondaryUvs = secondaryUvsForMaterial(uvChannels, material)
    const textureInfo = extractTextureData(material)
    const castShadow = object.castShadow === true ? true : undefined
    const receiveShadow = object.receiveShadow === true ? true : undefined
    const clipping = clippingState(globalClippingPlanes, material)
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
          textureTransform: textureInfo?.transform,
          textureColorSpace: textureInfo?.colorSpace,
          textureUsesUv2: textureInfo?.usesUv2,
          transform: instance.transform,
          topology: wireframe ? 'lines' : undefined,
          castShadow,
          receiveShadow,
          ...clipping,
          ...sortInfo,
          ...pbrProps,
        })
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
          textureTransform: textureInfo?.transform,
          textureColorSpace: textureInfo?.colorSpace,
          textureUsesUv2: textureInfo?.usesUv2,
          transform: instance.transform,
          topology: wireframe ? 'lines' : undefined,
          castShadow,
          receiveShadow,
          ...clipping,
          ...sortInfo,
          ...pbrProps,
        })
      }
    }
  }
}

function appendSprite(
  object: ThreeObject3DLike,
  camera: ThreeCameraLike | undefined,
  meshes: FlattenedMesh[],
  groupOrder: number,
  globalClippingPlanes: readonly NativeClippingPlane[],
): void {
  const material = materialForGroup(object.material, 0)
  if (material?.visible === false) return

  const matrix = matrixElements(object.matrixWorld!, 'sprite.matrixWorld')
  const center = [
    finiteOrDefault(object.center?.x, 0.5),
    finiteOrDefault(object.center?.y, 0.5),
  ]
  const worldPosition = [matrix[12], matrix[13], matrix[14]]
  let scaleX = columnLength3(matrix, 0)
  let scaleY = columnLength3(matrix, 4)

  if (material?.sizeAttenuation === false && camera?.isPerspectiveCamera === true) {
    const viewZ = viewSpaceZ(worldPosition, camera)
    if (Number.isFinite(viewZ)) {
      scaleX *= -viewZ
      scaleY *= -viewZ
    }
  }

  if (scaleX <= 0 || scaleY <= 0) return

  const axes = cameraBillboardAxes(camera)
  const rotation = finiteOrDefault(material?.rotation, 0)
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
  const clipping = clippingState(globalClippingPlanes, material)

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
    textureTransform: textureInfo?.transform,
    textureColorSpace: textureInfo?.colorSpace,
    textureUsesUv2: textureInfo?.usesUv2,
    transform: IDENTITY_4X4.slice(),
    transparent: material?.transparent !== false,
    castShadow: undefined,
    receiveShadow: undefined,
    ...clipping,
    ...sortInfo,
    ...extractPbrProperties(material),
  })
}

function appendPoints(
  object: ThreeObject3DLike,
  camera: ThreeCameraLike | undefined,
  meshes: FlattenedMesh[],
  groupOrder: number,
  viewportHeight: number,
  globalClippingPlanes: readonly NativeClippingPlane[],
): void {
  const geometry = object.geometry!
  const position = getAttribute(geometry, 'position')
  if (!position) return

  const positions = readVec3Attribute(position)
  const uvs = getAttribute(geometry, 'uv') ? readVec2Attribute(getAttribute(geometry, 'uv')!) : null
  const vertexColors = getAttribute(geometry, 'color')
  const index = geometry.index ? readIndexAttribute(geometry.index) : null
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
    const pointSize = Math.max(0, finiteOrDefault(material?.size, 1))
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
        const pointUv = uvs ? [uvs[pointIndex * 2], uvs[pointIndex * 2 + 1]] : null
        const pointColor = outputColors ? pointVertexColor(vertexColors!, baseColor, pointIndex, instance) : null
        for (const [x, y, u, v] of corners) {
          outputPositions.push(
            center[0] + axes.right[0] * x * worldSize + axes.up[0] * y * worldSize,
            center[1] + axes.right[1] * x * worldSize + axes.up[1] * y * worldSize,
            center[2] + axes.right[2] * x * worldSize + axes.up[2] * y * worldSize,
          )
          if (pointUv) {
            outputUvs.push(pointUv[0], pointUv[1])
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

    if (outputPositions.length === 0) continue

    const textureInfo = extractTextureData(material)
    const sortInfo = sortInfoForObject(object, material, camera, meshes.length, groupOrder)
    const pbrProps = extractPbrProperties(material)
    const clipping = clippingState(globalClippingPlanes, material)

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
      textureTransform: textureInfo?.transform,
      textureColorSpace: textureInfo?.colorSpace,
      textureUsesUv2: textureInfo?.usesUv2,
      transform: IDENTITY_4X4.slice(),
      transparent: material?.transparent === true || (material?.opacity != null && material.opacity < 1),
      topology: 'triangles',
      ...clipping,
      ...sortInfo,
      ...pbrProps,
      shadingModel: 'basic',
    })
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
  globalClippingPlanes: readonly NativeClippingPlane[],
): void {
  const geometry = object.geometry!
  const position = getAttribute(geometry, 'position')
  if (!position) return

  const material = materialForGroup(object.material, 0)
  if (material?.visible === false) return
  if (topology === 'lines') {
    assertSupportedLineMaterial(material)
  }

  const positions = readVec3Attribute(position)
  const uvAttribute = getAttribute(geometry, 'uv')
  const uvs = uvAttribute ? readVec2Attribute(uvAttribute) : null
  const vertexColors = getAttribute(geometry, 'color')
  const indexAttr = geometry.index ? readIndexAttribute(geometry.index) : null
  const vertexCount = position.count

  const range = geometry.drawRange ?? {}
  const sourceCount = indexAttr ? indexAttr.length : vertexCount
  const drawStart = clampInteger(range.start ?? 0, 0, sourceCount)
  const requestedCount = range.count == null || range.count === Infinity ? sourceCount : range.count
  const drawEnd = clampInteger(drawStart + requestedCount, drawStart, sourceCount)

  let indices: number[] | null = null
  let outputPositions = positions
  let outputUvs: number[] | undefined = topology === 'lines' ? uvs ?? undefined : undefined
  let outputColors: number[] | undefined
  const color = materialColor(material)
  const useVertexColors = vertexColors && material?.vertexColors !== false
  if (topology === 'lines') {
    const source = indexAttr ?? rangeIndices(vertexCount)
    if (material?.isLineDashedMaterial === true) {
      const dashed = dashedLineAttributes(
        positions,
        uvs,
        useVertexColors ? readColorAttribute(vertexColors!, color) : undefined,
        source,
        drawStart,
        drawEnd,
        object,
        getAttribute(geometry, 'lineDistance'),
        material,
      )
      if (dashed.positions.length < 6) return
      outputPositions = dashed.positions
      outputUvs = dashed.uvs
      outputColors = dashed.colors
      indices = null
    } else {
      indices = expandLineIndices(source, drawStart, drawEnd, object)
      if (indices.length < 2) return
    }
  } else if (indexAttr) {
    indices = indexAttr.slice(drawStart, drawEnd)
    if (indices.length === 0) return
  }

  if (useVertexColors && material?.isLineDashedMaterial !== true) {
    outputColors = readColorAttribute(vertexColors!, color)
  }
  const sortInfo = sortInfoForObject(object, material, camera, meshes.length, groupOrder)
  const clipping = clippingState(globalClippingPlanes, material)
  const pbrProps = extractPbrProperties(material)
  const textureInfo = extractTextureData(material)

  pushMesh(meshes, {
    positions: outputPositions,
    indices: indices ?? undefined,
    uvs: topology === 'lines' ? outputUvs : undefined,
    color,
    colors: outputColors,
    texture: textureInfo?.data,
    textureWidth: textureInfo?.width ?? undefined,
    textureHeight: textureInfo?.height ?? undefined,
    textureWrapS: textureInfo?.wrapS,
    textureWrapT: textureInfo?.wrapT,
    textureMagFilter: textureInfo?.magFilter,
    textureMinFilter: textureInfo?.minFilter,
    textureTransform: textureInfo?.transform,
    textureColorSpace: textureInfo?.colorSpace,
    textureUsesUv2: textureInfo?.usesUv2,
    transform: matrixElements(object.matrixWorld!, 'object.matrixWorld'),
    transparent: material?.transparent === true || (material?.opacity != null && material.opacity < 1),
    alphaTest: material && Number.isFinite(material.alphaTest) && material.alphaTest! > 0 ? material.alphaTest : undefined,
    ...pbrProps,
    shadingModel: 'basic',
    topology,
    ...clipping,
    ...sortInfo,
  })
}

function assertSupportedLineMaterial(material: ThreeMaterialLike | undefined): void {
  if (!material || !Number.isFinite(material.linewidth) || material.linewidth === 1) return
  throw new Error(
    'Line material linewidth values other than 1 are not supported by @headless-three/renderer yet. Use the default 1-pixel line width or expand thick lines to mesh geometry before rendering.',
  )
}

function clippingState(
  globalClippingPlanes: readonly NativeClippingPlane[],
  material: ThreeMaterialLike | undefined,
): Pick<NativeSceneMesh, 'clippingPlanes' | 'clippingUnionCount'> {
  const globalPlanes = globalClippingPlanes.slice(0, MAX_CLIPPING_PLANES)
  const localPlanes = extractClippingPlanes(material?.clippingPlanes).slice(
    0,
    Math.max(0, MAX_CLIPPING_PLANES - globalPlanes.length),
  )
  const planes = [...globalPlanes, ...localPlanes]
  if (planes.length === 0) return {}

  return {
    clippingPlanes: flattenClippingPlanes(planes),
    clippingUnionCount: globalPlanes.length + (material?.clipIntersection === true ? 0 : localPlanes.length),
  }
}

function pushMesh(meshes: FlattenedMesh[], mesh: NativeSceneMesh): void {
  meshes.push({
    mesh,
    groupOrder: mesh.groupOrder ?? 0,
    renderOrder: mesh.renderOrder ?? 0,
    sortZ: mesh.sortZ ?? 0,
    materialSortKey: mesh.materialSortKey ?? 0,
    sortIndex: mesh.sortIndex ?? meshes.length,
  })
}

function sortInfoForObject(
  object: ThreeObject3DLike,
  material: { id?: number } | undefined,
  camera: ThreeCameraLike | undefined,
  sortIndex: number,
  groupOrder: number,
  transform?: number[],
): Pick<NativeSceneMesh, 'groupOrder' | 'renderOrder' | 'sortZ' | 'sortIndex' | 'materialSortKey'> {
  return {
    groupOrder,
    renderOrder: finiteOrDefault(object.renderOrder, 0),
    sortZ: camera ? projectedObjectZ(object, camera, transform) : 0,
    sortIndex: unsignedSortKey(object.id, sortIndex),
    materialSortKey: finiteOrDefault(material?.id, 0),
  }
}

function compareFlattenedMeshes(a: FlattenedMesh, b: FlattenedMesh): number {
  return a.groupOrder - b.groupOrder
    || a.renderOrder - b.renderOrder
    || a.materialSortKey - b.materialSortKey
    || a.sortZ - b.sortZ
    || a.sortIndex - b.sortIndex
}

function projectedObjectZ(object: ThreeObject3DLike, camera: ThreeCameraLike, transform?: ArrayLike<number>): number {
  const world = transform ?? object.matrixWorld?.elements
  if (!world || world.length < 16) return 0
  const view = camera.matrixWorldInverse?.elements
  const projection = camera.projectionMatrix?.elements
  if (!view || view.length < 16 || !projection || projection.length < 16) return 0

  const x = world[12]
  const y = world[13]
  const z = world[14]
  const vx = view[0] * x + view[4] * y + view[8] * z + view[12]
  const vy = view[1] * x + view[5] * y + view[9] * z + view[13]
  const vz = view[2] * x + view[6] * y + view[10] * z + view[14]
  const vw = view[3] * x + view[7] * y + view[11] * z + view[15]
  const clipZ = projection[2] * vx + projection[6] * vy + projection[10] * vz + projection[14] * vw
  const clipW = projection[3] * vx + projection[7] * vy + projection[11] * vz + projection[15] * vw
  return clipW === 0 ? clipZ : clipZ / clipW
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

  if (camera?.isPerspectiveCamera === true && material?.sizeAttenuation !== false) {
    return pointSize / projectionY
  }

  const viewZ = camera ? viewSpaceZ(worldPosition, camera) : -1
  const depth = Number.isFinite(viewZ) ? Math.max(0.0001, Math.abs(viewZ)) : 1
  return pointSize * 2 * depth / Math.max(1, viewportHeight) / projectionY
}

function finiteOrDefault(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function unsignedSortKey(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? value : fallback
}

function instancedBufferGeometryCount(geometry: ThreeBufferGeometryLike): number {
  const attributes = Object.values(geometry.attributes ?? {})
  const instancedAttributes = attributes.filter((attribute): attribute is ThreeBufferAttributeLike => isInstancedAttribute(attribute))
  if (geometry.isInstancedBufferGeometry !== true && instancedAttributes.length === 0) return 1

  let maxCount = Infinity
  for (const attribute of instancedAttributes) {
    maxCount = Math.min(maxCount, attribute.count * meshPerAttribute(attribute))
  }

  const requested = Number.isFinite(geometry.instanceCount) ? geometry.instanceCount! : Infinity
  const effectiveCount = Math.min(requested, maxCount)
  if (effectiveCount === Infinity) return 1
  return clampInteger(Math.floor(effectiveCount), 0, Math.max(0, Math.floor(maxCount)))
}

function isInstancedAttribute(attribute: ThreeBufferAttributeLike | undefined | null): attribute is ThreeBufferAttributeLike {
  return attribute?.isInstancedBufferAttribute === true
}

function meshPerAttribute(attribute: ThreeBufferAttributeLike): number {
  return Math.max(1, Math.floor(finiteOrDefault(attribute.meshPerAttribute, 1)))
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
    const colors = readColorAttribute(attribute, materialColor)
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
  return attribute ? readVec2Attribute(attribute) : null
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
  const channelTexture = [
    material?.clearcoatMap,
    material?.clearcoatRoughnessMap,
    material?.clearcoatNormalMap,
    material?.sheenColorMap,
    material?.sheenRoughnessMap,
    material?.anisotropyMap,
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
  ].find((texture) => texture && Number.isInteger(texture.channel) && texture.channel! > 0)
  const texture = channelTexture
    ?? material?.clearcoatMap
    ?? material?.clearcoatRoughnessMap
    ?? material?.clearcoatNormalMap
    ?? material?.sheenColorMap
    ?? material?.sheenRoughnessMap
    ?? material?.anisotropyMap
    ?? material?.normalMap
    ?? material?.bumpMap
    ?? material?.transmissionMap
    ?? material?.thicknessMap
    ?? material?.specularColorMap
    ?? material?.specularIntensityMap
    ?? material?.displacementMap
    ?? material?.map
    ?? material?.metalnessMap
    ?? material?.roughnessMap
    ?? material?.emissiveMap
    ?? material?.lightMap
    ?? material?.aoMap
    ?? material?.specularMap
    ?? material?.alphaMap
  const channel = texture && Number.isInteger(texture.channel) ? texture.channel! : 0
  return channels[Math.max(0, Math.min(channel, channels.length - 1))] ?? channels[0]
}

function updateLodObject(object: ThreeObject3DLike, camera: ThreeCameraLike | undefined): void {
  if (object.isLOD !== true || !camera || object.autoUpdate === false) return

  if (typeof object.update === 'function') {
    object.update(camera)
    return
  }

  const levels = object.levels
  if (!Array.isArray(levels) || levels.length <= 1) return

  const distance = distanceBetweenMatrices(camera.matrixWorld, object.matrixWorld) / finiteOrDefault(camera.zoom, 1)
  levels[0].object.visible = true

  let i = 1
  for (; i < levels.length; i += 1) {
    const level = levels[i]
    let levelDistance = finiteOrDefault(level.distance, 0)
    if (level.object.visible) {
      levelDistance -= levelDistance * finiteOrDefault(level.hysteresis, 0)
    }
    if (distance >= levelDistance) {
      levels[i - 1].object.visible = false
      level.object.visible = true
    } else {
      break
    }
  }

  ;(object as { _currentLevel?: number })._currentLevel = i - 1

  for (; i < levels.length; i += 1) {
    levels[i].object.visible = false
  }
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
  colors: number[] | undefined,
  source: number[],
  start: number,
  end: number,
  object: ThreeObject3DLike,
  lineDistance: ThreeBufferAttributeLike | undefined,
  material: { dashSize?: number; gapSize?: number; scale?: number },
): DashedLineExpansion {
  const dashSize = Math.max(0, finiteOrDefault(material.dashSize, 3))
  const gapSize = Math.max(0, finiteOrDefault(material.gapSize, 1))
  const scale = finiteOrDefault(material.scale, 1)
  if (dashSize <= 0) return { positions: [] }

  const segments = lineSegmentsWithDistances(positions, source, start, end, object, lineDistance)
  const out = createDashedLineExpansion(uvs, colors)
  if (scale <= 0 || gapSize <= 0) {
    for (const segment of segments) {
      appendInterpolatedLine(out, positions, uvs, colors, segment.a, segment.b, 0, 1)
    }
    return out
  }

  const totalSize = dashSize + gapSize
  for (const segment of segments) {
    appendDashedSegment(out, positions, uvs, colors, segment, scale, dashSize, totalSize)
  }
  return out
}

function createDashedLineExpansion(uvs: number[] | null, colors: number[] | undefined): DashedLineExpansion {
  return {
    positions: [],
    uvs: uvs ? [] : undefined,
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
      appendInterpolatedLine(out, positions, uvs, colors, segment.a, segment.b, t0, t1)
    }
    cursor = next
  }
}

function appendInterpolatedLine(
  out: DashedLineExpansion,
  positions: number[],
  uvs: number[] | null,
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
    Number.isFinite(object.count) ? object.count! : instanceMatrix.count,
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
