import type { Color4, ThreeMaterialLike, PbrProperties, TextureInfo, ThreeTextureLike, ThreeSceneRootLike, ThreeObject3DLike } from './types'
import { clamp01 } from './math'
import { strictColorLikeToArray, validatedColorLikeToArray } from './color'
import { objectChildren } from './objects'
import { BasicDepthPacking, MaterialExtractionContext, MultiplyOperation } from './materials.part-001'
import { materialSlotColor } from './materials.part-002'
import { copyMaterialScalarFeatureProperties, materialRenderStateProperties, materialScalarFeatureSignature, sameMaterialScalarFeatureSignature } from './materials.part-004'
import { extractCustomFragmentShader, finiteNumberOrDefault, firstOptionalFiniteNumber, firstOptionalVector3LikeToArray, materialDepthPacking, materialNormalMapType, materialRangePair, materialRendererHints, optionalBoolean, optionalFiniteNumber, optionalFiniteNumberOrInfinityDefault } from './materials.part-005'
import { assertSupportedShaderMaterial } from './materials.part-006'
import { assertSupportedMaterialNodeHooks, assertSupportedOnBeforeCompile } from './materials.part-007'
import { TextureStateOptions, assertSupportedMaterialClass, assertSupportedMaterialState, textureSamplerState } from './materials.part-008'
import { extractTextureFromSlot } from './materials.part-009'
import { PhysicalMapFeatureGates, isRefractionEnvironmentMapping } from './materials.part-010'
import { assertCompatiblePackedPhysicalMapSamplers } from './materials.part-011'
export function extractPbrProperties(
  material: ThreeMaterialLike | undefined,
  context: MaterialExtractionContext = {},
): PbrProperties {
  if (!material) return {}
  const customFragmentShader = extractCustomFragmentShader(material)
  assertSupportedShaderMaterial(material, customFragmentShader)
  assertSupportedOnBeforeCompile(material, customFragmentShader)
  assertSupportedMaterialNodeHooks(material)
  assertSupportedMaterialClass(material, customFragmentShader)
  assertSupportedMaterialState(material, context)
  optionalBoolean(material.visible, 'material.visible')
  optionalBoolean(material.vertexColors, 'material.vertexColors')
  const props: PbrProperties = materialScalarFeatureProperties(material, context)
  const sheen = clamp01(optionalFiniteNumber(material.sheen, 'material.sheen') ?? 0)
  const physicalMapFeatures: PhysicalMapFeatureGates = {
    clearcoat: (props.clearcoat ?? 0) > 0,
    sheen: sheen > 0,
    anisotropy: (props.anisotropy ?? 0) > 0,
    iridescence: (props.iridescence ?? 0) > 0,
    transmission: (props.transmission ?? 0) > 0,
  }
  assertCompatiblePackedPhysicalMapSamplers(material, physicalMapFeatures)
  const textureFromSlot = (slot: ThreeMaterialLike['map'], label: string) => extractTextureFromSlot(
    slot,
    label,
    context.textureCache,
  )
  const colorFromSlot = (slot: string, value: unknown, label: string) => materialSlotColor(
    material,
    slot,
    value,
    label,
    context,
  )
  const textureStateFromSlot = (
    prefix: string,
    slot: ThreeMaterialLike['map'],
    label: string,
    options?: TextureStateOptions,
  ) => assignPbrTextureSamplerState(props, prefix, slot, label, context, options)

  const clearcoatMapInfo = physicalMapFeatures.clearcoat
    ? textureFromSlot(material.clearcoatMap, 'material.clearcoatMap')
    : null
  if (clearcoatMapInfo) {
    props.clearcoatMap = clearcoatMapInfo.data
    props.clearcoatMapWidth = clearcoatMapInfo.width
    props.clearcoatMapHeight = clearcoatMapInfo.height
    textureStateFromSlot('clearcoatMap', material.clearcoatMap, 'material.clearcoatMap')
  }
  const clearcoatRoughnessMapInfo = physicalMapFeatures.clearcoat
    ? textureFromSlot(material.clearcoatRoughnessMap, 'material.clearcoatRoughnessMap')
    : null
  if (clearcoatRoughnessMapInfo) {
    props.clearcoatRoughnessMap = clearcoatRoughnessMapInfo.data
    props.clearcoatRoughnessMapWidth = clearcoatRoughnessMapInfo.width
    props.clearcoatRoughnessMapHeight = clearcoatRoughnessMapInfo.height
    textureStateFromSlot('clearcoatRoughnessMap', material.clearcoatRoughnessMap, 'material.clearcoatRoughnessMap')
  }
  const clearcoatNormalMapInfo = physicalMapFeatures.clearcoat
    ? textureFromSlot(material.clearcoatNormalMap, 'material.clearcoatNormalMap')
    : null
  if (clearcoatNormalMapInfo) {
    props.clearcoatNormalMap = clearcoatNormalMapInfo.data
    props.clearcoatNormalMapWidth = clearcoatNormalMapInfo.width
    props.clearcoatNormalMapHeight = clearcoatNormalMapInfo.height
    textureStateFromSlot('clearcoatNormalMap', material.clearcoatNormalMap, 'material.clearcoatNormalMap')
  }
  const sheenColor = colorFromSlot('sheenColor', material.sheenColor, 'material.sheenColor')
  if (sheenColor && sheen > 0) {
    props.sheenColor = [
      sheenColor[0] * sheen,
      sheenColor[1] * sheen,
      sheenColor[2] * sheen,
    ]
  }
  const sheenColorMapInfo = physicalMapFeatures.sheen
    ? textureFromSlot(material.sheenColorMap, 'material.sheenColorMap')
    : null
  if (sheenColorMapInfo) {
    props.sheenColorMap = sheenColorMapInfo.data
    props.sheenColorMapWidth = sheenColorMapInfo.width
    props.sheenColorMapHeight = sheenColorMapInfo.height
    textureStateFromSlot('sheenColorMap', material.sheenColorMap, 'material.sheenColorMap')
  }
  const sheenRoughnessMapInfo = physicalMapFeatures.sheen
    ? textureFromSlot(material.sheenRoughnessMap, 'material.sheenRoughnessMap')
    : null
  if (sheenRoughnessMapInfo) {
    props.sheenRoughnessMap = sheenRoughnessMapInfo.data
    props.sheenRoughnessMapWidth = sheenRoughnessMapInfo.width
    props.sheenRoughnessMapHeight = sheenRoughnessMapInfo.height
    textureStateFromSlot('sheenRoughnessMap', material.sheenRoughnessMap, 'material.sheenRoughnessMap')
  }
  const anisotropyMapInfo = physicalMapFeatures.anisotropy
    ? textureFromSlot(material.anisotropyMap, 'material.anisotropyMap')
    : null
  if (anisotropyMapInfo) {
    props.anisotropyMap = anisotropyMapInfo.data
    props.anisotropyMapWidth = anisotropyMapInfo.width
    props.anisotropyMapHeight = anisotropyMapInfo.height
    textureStateFromSlot('anisotropyMap', material.anisotropyMap, 'material.anisotropyMap')
  }
  const iridescenceMapInfo = physicalMapFeatures.iridescence
    ? textureFromSlot(material.iridescenceMap, 'material.iridescenceMap')
    : null
  if (iridescenceMapInfo) {
    props.iridescenceMap = iridescenceMapInfo.data
    props.iridescenceMapWidth = iridescenceMapInfo.width
    props.iridescenceMapHeight = iridescenceMapInfo.height
    textureStateFromSlot('iridescenceMap', material.iridescenceMap, 'material.iridescenceMap')
  }
  const iridescenceThicknessMapInfo = physicalMapFeatures.iridescence
    ? textureFromSlot(material.iridescenceThicknessMap, 'material.iridescenceThicknessMap')
    : null
  if (iridescenceThicknessMapInfo) {
    props.iridescenceThicknessMap = iridescenceThicknessMapInfo.data
    props.iridescenceThicknessMapWidth = iridescenceThicknessMapInfo.width
    props.iridescenceThicknessMapHeight = iridescenceThicknessMapInfo.height
    textureStateFromSlot('iridescenceThicknessMap', material.iridescenceThicknessMap, 'material.iridescenceThicknessMap')
  }
  const transmissionMapInfo = physicalMapFeatures.transmission
    ? textureFromSlot(material.transmissionMap, 'material.transmissionMap')
    : null
  if (transmissionMapInfo) {
    props.transmissionMap = transmissionMapInfo.data
    props.transmissionMapWidth = transmissionMapInfo.width
    props.transmissionMapHeight = transmissionMapInfo.height
    textureStateFromSlot('transmissionMap', material.transmissionMap, 'material.transmissionMap')
  }
  const thicknessMapInfo = physicalMapFeatures.transmission
    ? textureFromSlot(material.thicknessMap, 'material.thicknessMap')
    : null
  if (thicknessMapInfo) {
    props.thicknessMap = thicknessMapInfo.data
    props.thicknessMapWidth = thicknessMapInfo.width
    props.thicknessMapHeight = thicknessMapInfo.height
    textureStateFromSlot('thicknessMap', material.thicknessMap, 'material.thicknessMap')
  }
  const attenuationColor = colorFromSlot('attenuationColor', material.attenuationColor, 'material.attenuationColor')
  if (attenuationColor) {
    props.attenuationColor = [attenuationColor[0], attenuationColor[1], attenuationColor[2]]
  }
  const physicalSpecularColor = colorFromSlot('specularColor', material.specularColor, 'material.specularColor')
  if (physicalSpecularColor) {
    props.physicalSpecularColor = [
      physicalSpecularColor[0],
      physicalSpecularColor[1],
      physicalSpecularColor[2],
    ]
  }
  const specularColorMapInfo = textureFromSlot(material.specularColorMap, 'material.specularColorMap')
  if (specularColorMapInfo) {
    props.specularColorMap = specularColorMapInfo.data
    props.specularColorMapWidth = specularColorMapInfo.width
    props.specularColorMapHeight = specularColorMapInfo.height
    textureStateFromSlot('specularColorMap', material.specularColorMap, 'material.specularColorMap')
  }
  const specularIntensityMapInfo = textureFromSlot(material.specularIntensityMap, 'material.specularIntensityMap')
  if (specularIntensityMapInfo) {
    props.specularIntensityMap = specularIntensityMapInfo.data
    props.specularIntensityMapWidth = specularIntensityMapInfo.width
    props.specularIntensityMapHeight = specularIntensityMapInfo.height
    textureStateFromSlot('specularIntensityMap', material.specularIntensityMap, 'material.specularIntensityMap')
  }

  const specularColor = colorFromSlot('specular', material.specular, 'material.specular')
  if (specularColor || material.isMeshPhongMaterial) {
    const color = specularColor ?? [17 / 255, 17 / 255, 17 / 255, 1]
    props.specularColor = [color[0], color[1], color[2]]
  }

  const emissive = colorFromSlot('emissive', material.emissive, 'material.emissive')
  if (emissive) {
    props.emissive = [emissive[0], emissive[1], emissive[2]]
    props.emissiveIntensity = finiteNumberOrDefault(material.emissiveIntensity, 'material.emissiveIntensity', 1)
  }

  const normalMapInfo = textureFromSlot(material.normalMap, 'material.normalMap')
  if (normalMapInfo) {
    props.normalMap = normalMapInfo.data
    props.normalMapWidth = normalMapInfo.width
    props.normalMapHeight = normalMapInfo.height
    textureStateFromSlot('normalMap', material.normalMap, 'material.normalMap')
    props.normalMapType = materialNormalMapType(material)
  }
  const bumpMapInfo = textureFromSlot(material.bumpMap, 'material.bumpMap')
  if (bumpMapInfo) {
    props.bumpMap = bumpMapInfo.data
    props.bumpMapWidth = bumpMapInfo.width
    props.bumpMapHeight = bumpMapInfo.height
    textureStateFromSlot('bumpMap', material.bumpMap, 'material.bumpMap')
    props.bumpScale = finiteNumberOrDefault(material.bumpScale, 'material.bumpScale', 1)
  }
  if (material.isMeshMatcapMaterial) {
    const matcapMapInfo = textureFromSlot(material.map, 'material.map')
    if (matcapMapInfo) {
      props.matcapMap = matcapMapInfo.data
      props.matcapMapWidth = matcapMapInfo.width
      props.matcapMapHeight = matcapMapInfo.height
      textureStateFromSlot('matcapMap', material.map, 'material.map')
    }
  }
  const gradientMapInfo = textureFromSlot(material.gradientMap, 'material.gradientMap')
  if (gradientMapInfo) {
    props.gradientMap = gradientMapInfo.data
    props.gradientMapWidth = gradientMapInfo.width
    props.gradientMapHeight = gradientMapInfo.height
    textureStateFromSlot('gradientMap', material.gradientMap, 'material.gradientMap', {
      includeTransform: false,
      includeUvChannel: false,
    })
  }

  const displacementMapInfo = textureFromSlot(material.displacementMap, 'material.displacementMap')
  if (displacementMapInfo) {
    props.displacementMap = displacementMapInfo.data
    props.displacementMapWidth = displacementMapInfo.width
    props.displacementMapHeight = displacementMapInfo.height
    textureStateFromSlot('displacementMap', material.displacementMap, 'material.displacementMap')
    props.displacementScale = finiteNumberOrDefault(material.displacementScale, 'material.displacementScale', 1)
    props.displacementBias = finiteNumberOrDefault(material.displacementBias, 'material.displacementBias', 0)
  }

  const mrMap = material.metalnessMap ?? material.roughnessMap
  const mrMapLabel = material.metalnessMap ? 'material.metalnessMap' : 'material.roughnessMap'
  const mrMapInfo = textureFromSlot(mrMap, mrMapLabel)
  if (mrMapInfo) {
    props.metallicRoughnessTexture = mrMapInfo.data
    props.metallicRoughnessTextureWidth = mrMapInfo.width
    props.metallicRoughnessTextureHeight = mrMapInfo.height
    textureStateFromSlot('metallicRoughnessTexture', mrMap, mrMapLabel)
  }

  const specularMapInfo = textureFromSlot(material.specularMap, 'material.specularMap')
  if (specularMapInfo) {
    props.specularMap = specularMapInfo.data
    props.specularMapWidth = specularMapInfo.width
    props.specularMapHeight = specularMapInfo.height
    textureStateFromSlot('specularMap', material.specularMap, 'material.specularMap')
  }

  const emissiveMapInfo = textureFromSlot(material.emissiveMap, 'material.emissiveMap')
  if (emissiveMapInfo) {
    props.emissiveMap = emissiveMapInfo.data
    props.emissiveMapWidth = emissiveMapInfo.width
    props.emissiveMapHeight = emissiveMapInfo.height
    textureStateFromSlot('emissiveMap', material.emissiveMap, 'material.emissiveMap')
  }

  const aoMapInfo = textureFromSlot(material.aoMap, 'material.aoMap')
  if (aoMapInfo) {
    props.aoMap = aoMapInfo.data
    props.aoMapWidth = aoMapInfo.width
    props.aoMapHeight = aoMapInfo.height
    textureStateFromSlot('aoMap', material.aoMap, 'material.aoMap')
    props.aoMapIntensity = finiteNumberOrDefault(material.aoMapIntensity, 'material.aoMapIntensity', 1)
  }

  const lightMapInfo = textureFromSlot(material.lightMap, 'material.lightMap')
  if (lightMapInfo) {
    props.lightMap = lightMapInfo.data
    props.lightMapWidth = lightMapInfo.width
    props.lightMapHeight = lightMapInfo.height
    textureStateFromSlot('lightMap', material.lightMap, 'material.lightMap')
    props.lightMapIntensity = finiteNumberOrDefault(material.lightMapIntensity, 'material.lightMapIntensity', 1)
  }

  const alphaMapInfo = textureFromSlot(material.alphaMap, 'material.alphaMap')
  if (alphaMapInfo) {
    props.alphaMap = alphaMapInfo.data
    props.alphaMapWidth = alphaMapInfo.width
    props.alphaMapHeight = alphaMapInfo.height
    textureStateFromSlot('alphaMap', material.alphaMap, 'material.alphaMap')
  }

  Object.assign(props, materialRenderStateProperties(material, customFragmentShader, context))

  return props
}

export function assignPbrTextureSamplerState(
  props: PbrProperties,
  prefix: string,
  map: ThreeTextureLike | null | undefined,
  label: string,
  context: MaterialExtractionContext,
  options: TextureStateOptions = {},
): void {
  const state = textureSamplerState(map, label, context, options)
  const target = props as Record<string, unknown>
  target[`${prefix}WrapS`] = state.wrapS
  target[`${prefix}WrapT`] = state.wrapT
  target[`${prefix}MagFilter`] = state.magFilter
  target[`${prefix}MinFilter`] = state.minFilter
  target[`${prefix}Anisotropy`] = state.anisotropy
  if (options.includeTransform !== false) {
    target[`${prefix}Transform`] = state.transform
  }
  target[`${prefix}ColorSpace`] = state.colorSpace
  if (options.includeUvChannel !== false) {
    target[`${prefix}UsesUv2`] = state.usesUv2
  }
}

export interface CachedMaterialScalarFeatureExtraction {
  signature: MaterialScalarFeatureSignature
  props: PbrProperties
}

export interface MaterialScalarFeatureSignature {
  values: unknown[]
}

export function materialScalarFeatureProperties(
  material: ThreeMaterialLike,
  context: MaterialExtractionContext,
): PbrProperties {
  const signature = context.materialScalarFeatureCache
    ? materialScalarFeatureSignature(material, context)
    : null
  if (signature) {
    const cached = context.materialScalarFeatureCache?.get(material) as CachedMaterialScalarFeatureExtraction | undefined
    if (cached && sameMaterialScalarFeatureSignature(cached.signature, signature)) {
      return copyMaterialScalarFeatureProperties(cached.props)
    }
  }

  const props: PbrProperties = {}
  const usesMaterialEnvironmentMap = material.envMap != null
    && context.materialEnvironmentMaps?.has(material) === true
  if (usesMaterialEnvironmentMap) {
    props.useEnvironmentMap = true
    props.environmentMapIntensity = finiteNumberOrDefault(material.envMapIntensity, 'material.envMapIntensity', 1)
    props.environmentMapCombine = material.combine ?? MultiplyOperation
    props.environmentMapReflectivity = finiteNumberOrDefault(material.reflectivity, 'material.reflectivity', 1)
    props.environmentMapRefraction = isRefractionEnvironmentMapping(material.envMap?.mapping)
    props.environmentMapRefractionRatio = finiteNumberOrDefault(material.refractionRatio, 'material.refractionRatio', 0.98)
  } else if (context.materialEnvironmentSource === 'material') {
    props.useEnvironmentMap = false
  }

  const metalness = optionalFiniteNumber(material.metalness, 'material.metalness')
  if (metalness !== undefined) {
    props.metallic = clamp01(metalness)
  }
  const roughness = optionalFiniteNumber(material.roughness, 'material.roughness')
  if (roughness !== undefined) {
    props.roughness = clamp01(roughness)
  }
  const clearcoat = optionalFiniteNumber(material.clearcoat, 'material.clearcoat')
  if (clearcoat !== undefined) {
    props.clearcoat = clamp01(clearcoat)
  }
  const clearcoatRoughness = optionalFiniteNumber(material.clearcoatRoughness, 'material.clearcoatRoughness')
  if (clearcoatRoughness !== undefined) {
    props.clearcoatRoughness = clamp01(clearcoatRoughness)
  }
  if (material.clearcoatNormalScale != null) {
    if (typeof material.clearcoatNormalScale !== 'object') {
      throw new TypeError('material.clearcoatNormalScale must be a Vector2-like object.')
    }
    props.clearcoatNormalScale = [
      finiteNumberOrDefault(material.clearcoatNormalScale.x, 'material.clearcoatNormalScale.x', 1),
      finiteNumberOrDefault(material.clearcoatNormalScale.y, 'material.clearcoatNormalScale.y', 1),
    ]
  }

  const sheenRoughness = optionalFiniteNumber(material.sheenRoughness, 'material.sheenRoughness')
  if (sheenRoughness !== undefined) {
    props.sheenRoughness = clamp01(sheenRoughness)
  }
  const anisotropy = optionalFiniteNumber(material.anisotropy, 'material.anisotropy')
  if (anisotropy !== undefined) {
    props.anisotropy = clamp01(anisotropy)
  }
  const anisotropyRotation = optionalFiniteNumber(material.anisotropyRotation, 'material.anisotropyRotation')
  if (anisotropyRotation !== undefined) {
    props.anisotropyRotation = anisotropyRotation
  }
  const iridescence = optionalFiniteNumber(material.iridescence, 'material.iridescence')
  if (iridescence !== undefined) {
    props.iridescence = clamp01(iridescence)
  }
  const iridescenceIor = optionalFiniteNumber(material.iridescenceIOR, 'material.iridescenceIOR')
  if (iridescenceIor !== undefined) {
    props.iridescenceIor = Math.max(1, Math.min(2.333, iridescenceIor))
  }
  const iridescenceThicknessRange = materialRangePair(
    material.iridescenceThicknessRange,
    'material.iridescenceThicknessRange',
  )
  if (iridescenceThicknessRange) {
    const [min, max] = iridescenceThicknessRange
    props.iridescenceThicknessMin = Math.max(0, min)
    props.iridescenceThicknessMax = Math.max(props.iridescenceThicknessMin, max)
  }
  const transmission = optionalFiniteNumber(material.transmission, 'material.transmission')
  if (transmission !== undefined) {
    props.transmission = clamp01(transmission)
  }
  const dispersion = optionalFiniteNumber(material.dispersion, 'material.dispersion')
  if (dispersion !== undefined) {
    props.dispersion = Math.max(0, dispersion)
  }
  const ior = optionalFiniteNumber(material.ior, 'material.ior')
  if (ior !== undefined) {
    props.ior = Math.max(1, Math.min(2.333, ior))
  }
  const thickness = optionalFiniteNumber(material.thickness, 'material.thickness')
  if (thickness !== undefined) {
    props.thickness = Math.max(0, thickness)
  }
  const attenuationDistance = optionalFiniteNumberOrInfinityDefault(
    material.attenuationDistance,
    'material.attenuationDistance',
  )
  if (attenuationDistance !== undefined) {
    props.attenuationDistance = Math.max(0, attenuationDistance)
  }
  const specularIntensity = optionalFiniteNumber(material.specularIntensity, 'material.specularIntensity')
  if (specularIntensity !== undefined) {
    props.physicalSpecularIntensity = clamp01(specularIntensity)
  }

  const shininess = material.isMeshPhongMaterial
    ? finiteNumberOrDefault(material.shininess, 'material.shininess', 30)
    : optionalFiniteNumber(material.shininess, 'material.shininess')
  if (shininess !== undefined) {
    props.shininess = Math.max(0.0001, shininess)
  }
  if (material.normalScale != null) {
    if (typeof material.normalScale !== 'object') {
      throw new TypeError('material.normalScale must be a Vector2-like object.')
    }
    props.normalScale = [
      finiteNumberOrDefault(material.normalScale.x, 'material.normalScale.x', 1),
      finiteNumberOrDefault(material.normalScale.y, 'material.normalScale.y', 1),
    ]
  }
  if (material.isMeshDepthMaterial) {
    props.depthPacking = materialDepthPacking(material) ?? BasicDepthPacking
  }
  if (material.isMeshDistanceMaterial) {
    const hintBag = materialRendererHints(material.userData)
    const hints = hintBag?.value ?? {}
    const hintsLabel = hintBag?.label ?? 'material.userData.headlessThreeRenderer'
    const referencePosition = firstOptionalVector3LikeToArray([
      [material.referencePosition, 'material.referencePosition'],
      [hints.referencePosition, `${hintsLabel}.referencePosition`],
      [hints.distanceReferencePosition, `${hintsLabel}.distanceReferencePosition`],
    ])
    if (referencePosition) {
      props.distanceReferencePosition = referencePosition
    }
    const nearDistance = firstOptionalFiniteNumber([
      [material.nearDistance, 'material.nearDistance'],
      [hints.nearDistance, `${hintsLabel}.nearDistance`],
      [hints.distanceNear, `${hintsLabel}.distanceNear`],
    ])
    if (nearDistance !== undefined) {
      props.distanceNear = nearDistance
    }
    const farDistance = firstOptionalFiniteNumber([
      [material.farDistance, 'material.farDistance'],
      [hints.farDistance, `${hintsLabel}.farDistance`],
      [hints.distanceFar, `${hintsLabel}.distanceFar`],
    ])
    if (farDistance !== undefined) {
      props.distanceFar = farDistance
    }
  }

  if (signature) {
    context.materialScalarFeatureCache?.set(material, {
      signature,
      props: copyMaterialScalarFeatureProperties(props),
    })
  }
  return props
}
