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
import { ClippingContext, FlattenedMesh, InstancedAttributeRef, PointBillboardExpansion, PointBillboardSignature, SceneExtractionCache, TextureUvStreams, UvChannel, UvChannelSignature } from './scene.part-001'
import { attributeSignature, attributeSignatureCacheable, sameAttributeSignature } from './scene.part-003'
import { shadowOnlyMainPassState } from './scene.part-004'
import { optionalSceneBoolean, pointVertexColor } from './scene.part-006'
import { clipShadowsForMaterial, clippingState, pushMesh, sortInfoForObject } from './scene.part-007'
import { pointWorldSize, transformPoint } from './scene.part-008'
import { appendUvForVertex, instanceScaleComponents, instancedAttributeIndex } from './scene.part-009'
import { rangeIndices } from './scene.part-014'
export function pointBillboardExpansion(
  object: ThreeObject3DLike,
  group: GeometryGroup,
  position: ThreeBufferAttributeLike,
  positions: number[],
  index: number[] | null,
  instancedGeometryCount: number,
  instancedPositionOffset: InstancedAttributeRef | null,
  instancedPositionScale: InstancedAttributeRef | null,
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
    instancedPositionScale,
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
    instancedPositionScale,
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

export function readPointBillboardExpansion(
  group: GeometryGroup,
  position: ThreeBufferAttributeLike,
  positions: number[],
  index: number[] | null,
  instancedGeometryCount: number,
  instancedPositionOffset: InstancedAttributeRef | null,
  instancedPositionScale: InstancedAttributeRef | null,
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
    const scale = instanceScaleComponents(instancedPositionScale, instance)

    for (let pointOffset = 0; pointOffset < points.length; pointOffset += 1) {
      const pointIndex = points[pointOffset]
      if (!Number.isInteger(pointIndex) || pointIndex < 0 || pointIndex >= position.count) continue

      const center = transformPoint(transform, [
        positions[pointIndex * 3] * scale[0] + offset[0],
        positions[pointIndex * 3 + 1] * scale[1] + offset[1],
        positions[pointIndex * 3 + 2] * scale[2] + offset[2],
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
        if (pointUvStreams) {
          if (pointUvStreams.uvs) {
            appendUvForVertex(outputUvs, pointUvStreams.uvs, pointIndex, instance)
          } else {
            outputUvs.push(u, v)
          }
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

export function pointBillboardSignature(
  group: GeometryGroup,
  position: ThreeBufferAttributeLike,
  positions: number[],
  index: number[] | null,
  instancedGeometryCount: number,
  instancedPositionOffset: InstancedAttributeRef | null,
  instancedPositionScale: InstancedAttributeRef | null,
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
    instancedPositionScale: attributeSignature(instancedPositionScale?.attribute),
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

export function pointBillboardSignatureCacheable(signature: PointBillboardSignature): boolean {
  return attributeSignatureCacheable(signature.instancedPositionOffset)
    && attributeSignatureCacheable(signature.instancedPositionScale)
    && attributeSignatureCacheable(signature.uvs.attribute)
    && attributeSignatureCacheable(signature.uvs2.attribute)
    && attributeSignatureCacheable(signature.vertexColors)
}

export function uvChannelSignature(channel: UvChannel | null | undefined): UvChannelSignature {
  if (!channel) return { attribute: {} }
  return {
    attribute: attributeSignature(channel.attribute),
    values: channel.values,
    label: channel.label,
  }
}

export function matrixValues(matrix: ArrayLike<number> | undefined): number[] | null {
  if (!matrix || matrix.length < 16) return null
  const out = new Array<number>(16)
  for (let i = 0; i < 16; i += 1) out[i] = matrix[i]
  return out
}

export function samePointBillboardSignature(a: PointBillboardSignature, b: PointBillboardSignature): boolean {
  return a.cacheable === b.cacheable
    && a.positions === b.positions
    && a.positionCount === b.positionCount
    && a.index === b.index
    && a.groupStart === b.groupStart
    && a.groupCount === b.groupCount
    && a.instancedGeometryCount === b.instancedGeometryCount
    && sameAttributeSignature(a.instancedPositionOffset, b.instancedPositionOffset)
    && sameAttributeSignature(a.instancedPositionScale, b.instancedPositionScale)
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

export function sameUvChannelSignature(a: UvChannelSignature, b: UvChannelSignature): boolean {
  return sameAttributeSignature(a.attribute, b.attribute)
    && a.values === b.values
    && a.label === b.label
}

export function sameOptionalNumberArray(a: ArrayLike<number> | null | undefined, b: ArrayLike<number> | null | undefined): boolean {
  if (a == null || b == null) return a == null && b == null
  return sameNumberArray(a, b)
}

export function sameNumberArray(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i += 1) {
    if (!Object.is(a[i], b[i])) return false
  }
  return true
}

export function expandPointBillboardUvStream(
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

export function appendShadowOnlyBillboardMesh(
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

export function shadowMaterialWithSourceShadowState(
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

export function sourceMaterialHasShadowState(material: ThreeMaterialLike | undefined): material is ThreeMaterialLike {
  return sourceMaterialHasShadowAlphaState(material) ||
    sourceMaterialHasShadowDisplacementState(material) ||
    sourceMaterialHasShadowClippingState(material) ||
    sourceMaterialHasShadowWireframeState(material)
}

export function sourceMaterialHasShadowAlphaState(material: ThreeMaterialLike | undefined): material is ThreeMaterialLike {
  if (!material) return false
  const hasAlphaTexture = !!(material.map || material.alphaMap)
  const hasOpacityAlpha = typeof material.opacity === 'number' && Number.isFinite(material.opacity) && material.opacity < 1
  if (material.alphaHash === true && (hasAlphaTexture || hasOpacityAlpha)) return true
  if (material.alphaToCoverage === true && (hasAlphaTexture || hasOpacityAlpha)) return true
  if (!hasAlphaTexture) return false
  if (material.alphaToCoverage === true) return true
  return typeof material.alphaTest === 'number' && Number.isFinite(material.alphaTest) && material.alphaTest > 0
}

export function sourceMaterialHasShadowDisplacementState(material: ThreeMaterialLike | undefined): material is ThreeMaterialLike {
  return !!material?.displacementMap
}

export function sourceMaterialHasShadowClippingState(material: ThreeMaterialLike | undefined): material is ThreeMaterialLike {
  return !!material && (
    'clipShadows' in material ||
    material.clippingPlanes != null ||
    'clipIntersection' in material
  )
}

export function sourceMaterialHasShadowWireframeState(material: ThreeMaterialLike | undefined): material is ThreeMaterialLike {
  return material?.wireframe === true
}

export function shadowPbrProperties(
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

export function validateObjectShadowFlags(object: ThreeObject3DLike): void {
  optionalObjectBoolean(object.castShadow, 'object.castShadow')
  optionalObjectBoolean(object.receiveShadow, 'object.receiveShadow')
}

export function optionalObjectBoolean(value: unknown, label: string): boolean | undefined {
  return optionalSceneBoolean(value, label)
}
