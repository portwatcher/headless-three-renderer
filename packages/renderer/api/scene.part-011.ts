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
import { DashedLineExpansion, DashedLineSignature, InstancedAttributeRef, SceneExtractionCache, UvChannel } from './scene.part-001'
import { attributeSignature, attributeSignatureCacheable, sameAttributeSignature } from './scene.part-003'
import { optionalObjectBoolean, sameUvChannelSignature, uvChannelSignature } from './scene.part-005'
import { cameraZoomOrDefault, nonNegativeMaterialOrObjectNumber, normalizedMaterialOrObjectNumber, positiveMaterialOrObjectNumber } from './scene.part-008'
import { dashedLineAttributesForInstances, instancedDashedLineCacheKey, instancedDashedLineSignature, sameInstancedDashedLineSignature } from './scene.part-012'
import { appendDashedSegment, appendInterpolatedLine, createDashedLineExpansion, lineSegmentsWithDistances } from './scene.part-013'
export function secondaryUvsForMaterial(
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

export function updateLodObject(object: ThreeObject3DLike, camera: ThreeCameraLike | undefined): void {
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

export function normalizeLodLevels(levels: unknown): Array<{ object: ThreeObject3DLike; distance: number; hysteresis: number }> {
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

export function distanceBetweenMatrices(a: ThreeCameraLike['matrixWorld'], b: ThreeObject3DLike['matrixWorld']): number {
  const ae = a?.elements
  const be = b?.elements
  if (!ae || ae.length < 16 || !be || be.length < 16) return 0
  const dx = ae[12] - be[12]
  const dy = ae[13] - be[13]
  const dz = ae[14] - be[14]
  return Math.sqrt(dx * dx + dy * dy + dz * dz)
}

export function dashedLineAttributes(
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

export function dashedLineAttributesWithCache(
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

export function dashedLineSignature(
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

export function dashedLineSignatureCacheable(signature: DashedLineSignature): boolean {
  return attributeSignatureCacheable(signature.position)
    && attributeSignatureCacheable(signature.index)
    && attributeSignatureCacheable(signature.uv.attribute)
    && attributeSignatureCacheable(signature.uv2.attribute)
    && attributeSignatureCacheable(signature.lineDistance)
}

export function dashedLineCacheKey(signature: DashedLineSignature): string {
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

export function sameDashedLineSignature(a: DashedLineSignature, b: DashedLineSignature): boolean {
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

export function dashedLineAttributesForInstancesWithCache(
  cache: SceneExtractionCache | undefined,
  geometry: ThreeBufferGeometryLike,
  position: ThreeBufferAttributeLike,
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
  const dashSize = positiveMaterialOrObjectNumber(material.dashSize, 'material.dashSize', 3)
  const gapSize = nonNegativeMaterialOrObjectNumber(material.gapSize, 'material.gapSize', 1)
  const scale = nonNegativeMaterialOrObjectNumber(material.scale, 'material.scale', 1)
  if (!cache) {
    return dashedLineAttributesForInstances(
      positions,
      uvChannel,
      uvChannel2,
      vertexColors,
      materialColor,
      source,
      start,
      end,
      object,
      lineDistance,
      material,
      instanceCount,
      offsetAttribute,
      scaleAttribute,
    )
  }

  const signature = instancedDashedLineSignature(
    geometry,
    position,
    uvChannel,
    uvChannel2,
    vertexColors,
    materialColor,
    lineDistance,
    start,
    end,
    source.length,
    object,
    dashSize,
    gapSize,
    scale,
    instanceCount,
    offsetAttribute,
    scaleAttribute,
  )
  if (!signature.cacheable) {
    return dashedLineAttributesForInstances(
      positions,
      uvChannel,
      uvChannel2,
      vertexColors,
      materialColor,
      source,
      start,
      end,
      object,
      lineDistance,
      material,
      instanceCount,
      offsetAttribute,
      scaleAttribute,
    )
  }

  const cacheKey = instancedDashedLineCacheKey(signature)
  let geometryCache = cache.instancedDashedLines.get(geometry)
  const cached = geometryCache?.get(cacheKey)
  if (cached && sameInstancedDashedLineSignature(cached.signature, signature)) {
    return cached.expansion
  }

  const expansion = dashedLineAttributesForInstances(
    positions,
    uvChannel,
    uvChannel2,
    vertexColors,
    materialColor,
    source,
    start,
    end,
    object,
    lineDistance,
    material,
    instanceCount,
    offsetAttribute,
    scaleAttribute,
  )
  if (!geometryCache) {
    geometryCache = new Map()
    cache.instancedDashedLines.set(geometry, geometryCache)
  }
  geometryCache.set(cacheKey, { signature, expansion })
  return expansion
}
