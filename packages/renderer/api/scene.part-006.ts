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
import { ClippingContext, FlattenedMesh, SceneExtractionCache, TextureUvStreams, ThickLineExpansion } from './scene.part-001'
import { RenderCallbackContext } from './scene.part-002'
import { materialForObjectGroup, meshGeometryExtraction } from './scene.part-003'
import { invokeObjectRenderCallback } from './scene.part-004'
import { optionalObjectBoolean, validateObjectShadowFlags } from './scene.part-005'
import { clipShadowsForMaterial, clippingState, projectedWorldPointZ, pushMesh, sortInfoForObject } from './scene.part-007'
import { cameraBillboardAxes, finiteOrDefault, normalizeVec3, positiveMaterialOrObjectNumber, transformPoint, validateLineMaterialCompatibilityHints, viewSpaceZ } from './scene.part-008'
import { assertSupportedCustomFragmentInstancedAttributes, expandUvChannelForInstancesWithCache, expandVec3ValuesForInstancesWithCache, instancedAttributeIndex, isInstancedAttribute } from './scene.part-009'
import { expandColorAttributeForInstancesWithCache, expandIndicesForInstances, textureUvStreamsForMapAlphaMaterial } from './scene.part-010'
import { dashedLineAttributesForInstancesWithCache, dashedLineAttributesWithCache, secondaryUvsForMaterial } from './scene.part-011'
import { expandLineIndices, rangeIndices } from './scene.part-014'
export function optionalSceneBoolean(value: unknown, label: string): boolean | undefined {
  if (value == null) return undefined
  if (typeof value === 'boolean') return value
  throw new TypeError(`${label} must be a boolean.`)
}

export function pointVertexColor(
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

export function effectiveGroups(
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

export function geometryDrawRangeStart(value: unknown): number {
  if (value == null) return 0
  return geometryGroupNonNegativeInteger(value, 'geometry.drawRange.start')
}

export function geometryDrawRangeCount(value: unknown, fallback: number): number {
  if (value == null || value === Infinity) return fallback
  return geometryGroupNonNegativeInteger(value, 'geometry.drawRange.count')
}

export function geometryGroupNonNegativeInteger(value: unknown, label: string): number {
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
export function appendLineOrPoints(
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
    instancedPositionScale,
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
        const dashed = instancedGeometryCount > 1 || instancedPositionOffset || instancedPositionScale
          ? dashedLineAttributesForInstancesWithCache(
            cache,
            geometry,
            position,
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
            instancedPositionScale,
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
        if (instancedGeometryCount > 1 || instancedPositionOffset || instancedPositionScale) {
          outputPositions = expandVec3ValuesForInstancesWithCache(cache, geometry, position, positions, 0, vertexCount, instancedGeometryCount, instancedPositionOffset, instancedPositionScale)
          outputUvs = uvStreams.uvs ? expandUvChannelForInstancesWithCache(cache, geometry, uvStreams.uvs, 0, vertexCount, instancedGeometryCount) : undefined
          outputSecondaryUvs = uvStreams.uvs2 ? expandUvChannelForInstancesWithCache(cache, geometry, uvStreams.uvs2, 0, vertexCount, instancedGeometryCount) : undefined
          indices = expandIndicesForInstances(indices, vertexCount, instancedGeometryCount)
        }
        if (thickLine) {
          const transform = matrixElements(object.matrixWorld!, 'object.matrixWorld')
          const thick = thickLineAttributes(
            outputPositions,
            outputUvs,
            outputSecondaryUvs,
            useVertexColors ? outputColors ?? expandColorAttributeForInstancesWithCache(cache, geometry, vertexColors!, color, 0, vertexCount, instancedGeometryCount) : undefined,
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
      outputColors = expandColorAttributeForInstancesWithCache(cache, geometry, vertexColors!, color, 0, vertexCount, instancedGeometryCount)
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

export function thickLineAttributes(
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

export function validPositionIndex(positions: number[], index: number): boolean {
  return Number.isInteger(index) && index >= 0 && index * 3 + 2 < positions.length
}

export function pushThickLineVertex(
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

export function pushRepeatedVec2(target: number[] | undefined, source: number[] | undefined, index: number): void {
  if (!target || !source || index * 2 + 1 >= source.length) return
  target.push(source[index * 2], source[index * 2 + 1])
}

export function pushRepeatedColor(target: number[] | undefined, source: number[] | undefined, index: number): void {
  if (!target || !source || index * 4 + 3 >= source.length) return
  target.push(source[index * 4], source[index * 4 + 1], source[index * 4 + 2], source[index * 4 + 3])
}

export function linePixelWorldSize(
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

export function thickLineCenter(positions: number[]): [number, number, number] {
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
