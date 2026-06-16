import type { Color4, ThreeMaterialLike, PbrProperties, TextureInfo, ThreeTextureLike, ThreeSceneRootLike, ThreeObject3DLike } from './types'
import { clamp01 } from './math'
import { strictColorLikeToArray } from './color'

// eslint-disable-next-line @typescript-eslint/no-var-requires
const native = require('../native.js') as {
  decodeImage?(data: Buffer): { data?: Buffer | Uint8Array; width?: number; height?: number }
}

// Three.js wrapping constants
const RepeatWrapping = 1000
const ClampToEdgeWrapping = 1001
const MirroredRepeatWrapping = 1002
const NearestFilter = 1003
const NearestMipmapNearestFilter = 1004
const NearestMipmapLinearFilter = 1005
const LinearFilter = 1006
const LinearMipmapNearestFilter = 1007
const LinearMipmapLinearFilter = 1008

// Three.js texture mapping constants
const CubeReflectionMapping = 301
const CubeRefractionMapping = 302
const EquirectangularReflectionMapping = 303
const EquirectangularRefractionMapping = 304
const CubeUVReflectionMapping = 306

// Three.js normal map type constants
const TangentSpaceNormalMap = 0
const ObjectSpaceNormalMap = 1

// Three.js environment combine constants
const MultiplyOperation = 0
const MixOperation = 1
const AddOperation = 2

// Three.js side constants
const FrontSide = 0
const BackSide = 1
const DoubleSide = 2

// Three.js depth comparison constants
const NeverDepth = 0
const AlwaysDepth = 1
const LessDepth = 2
const LessEqualDepth = 3
const EqualDepth = 4
const GreaterEqualDepth = 5
const GreaterDepth = 6
const NotEqualDepth = 7

// Three.js blending constants
const NoBlending = 0
const NormalBlending = 1
const AdditiveBlending = 2
const SubtractiveBlending = 3
const MultiplyBlending = 4
const CustomBlending = 5
const AddEquation = 100
const SubtractEquation = 101
const ReverseSubtractEquation = 102
const MinEquation = 103
const MaxEquation = 104
const ZeroFactor = 200
const OneFactor = 201
const SrcColorFactor = 202
const OneMinusSrcColorFactor = 203
const SrcAlphaFactor = 204
const OneMinusSrcAlphaFactor = 205
const DstAlphaFactor = 206
const OneMinusDstAlphaFactor = 207
const DstColorFactor = 208
const OneMinusDstColorFactor = 209
const SrcAlphaSaturateFactor = 210
const ConstantColorFactor = 211
const OneMinusConstantColorFactor = 212
const ConstantAlphaFactor = 213
const OneMinusConstantAlphaFactor = 214
const NeverStencilFunc = 512
const LessStencilFunc = 513
const EqualStencilFunc = 514
const LessEqualStencilFunc = 515
const GreaterStencilFunc = 516
const NotEqualStencilFunc = 517
const GreaterEqualStencilFunc = 518
const AlwaysStencilFunc = 519
const ZeroStencilOp = 0
const KeepStencilOp = 7680
const ReplaceStencilOp = 7681
const IncrementStencilOp = 7682
const DecrementStencilOp = 7683
const IncrementWrapStencilOp = 34055
const DecrementWrapStencilOp = 34056
const InvertStencilOp = 5386

// Three.js depth-packing constants
const BasicDepthPacking = 3200
const RGBADepthPacking = 3201
const RGBDepthPacking = 3202
const RGDepthPacking = 3203

// Three.js texture type constants
const UnsignedByteType = 1009
const ByteType = 1010
const ShortType = 1011
const UnsignedShortType = 1012
const IntType = 1013
const UnsignedIntType = 1014
const HalfFloatType = 1016
const FloatType = 1015
const UnsignedShort4444Type = 1017
const UnsignedShort5551Type = 1018
const UnsignedInt248Type = 1020
const UnsignedInt5999Type = 35902
const LinearEncoding = 3000
const sRGBEncoding = 3001

const DefaultOnBeforeCompileSource = 'onBeforeCompile( /* shaderobject, renderer */ ) {}'

export interface EnvironmentMapInfo {
  data: Buffer
  width: number
  height: number
  intensity: number
  colorSpace?: string
}

export interface MaterialExtractionContext {
  materialEnvironmentSource?: 'material'
  materialEnvironmentMaps?: WeakSet<ThreeMaterialLike>
}

export interface EnvironmentMapResolution {
  envMap: EnvironmentMapInfo | null
  materialContext?: MaterialExtractionContext
  rotation?: ThreeMaterialLike['envMapRotation']
}

type TextureImageInput = {
  data?: ArrayLike<number>
  width?: number
  height?: number
} | Buffer | Uint8Array

/**
 * Extract environment map data from scene.environment.
 * Supports DataTexture (equirectangular) with Uint8, Float16, Float32 pixel data.
 * Normalizes 3-channel inputs to RGBA before handing bytes to the native IBL path.
 */
export function extractEnvironmentMap(scene: ThreeSceneRootLike, intensityOverride?: number): EnvironmentMapInfo | null {
  if (scene.environment != null) {
    const envTex = requiredEnvironmentTexture(scene.environment, 'scene.environment')
    const intensity = intensityOverride !== undefined
      ? intensityOverride
      : optionalFiniteNumber((scene as any).environmentIntensity, 'scene.environmentIntensity') ?? 1.0
    return extractEnvironmentMapFromTexture(envTex, 'scene.environment', intensity)
  }

  const probe = extractReflectionProbe(scene)
  if (!probe) return null

  const intensity = intensityOverride !== undefined
    ? intensityOverride
    : optionalFiniteNumber(probe.intensity, 'reflectionProbe.intensity') ?? 1.0
  return extractEnvironmentMapFromTexture(probe.texture, probe.label, intensity)
}

export function resolveEnvironmentMap(scene: ThreeSceneRootLike, intensityOverride?: number): EnvironmentMapResolution {
  const sceneEnvMap = extractEnvironmentMap(scene, intensityOverride)
  if (sceneEnvMap) {
    return { envMap: sceneEnvMap }
  }

  const materialEnvMap = extractMaterialEnvironmentMap(scene)
  if (!materialEnvMap) {
    return { envMap: null }
  }

  return {
    envMap: materialEnvMap.envMap,
    rotation: materialEnvMap.rotation,
    materialContext: {
      materialEnvironmentSource: 'material',
      materialEnvironmentMaps: materialEnvMap.materials,
    },
  }
}

function extractMaterialEnvironmentMap(
  scene: ThreeSceneRootLike,
): { envMap: EnvironmentMapInfo; materials: WeakSet<ThreeMaterialLike>; rotation?: ThreeMaterialLike['envMapRotation'] } | null {
  let envTex: ThreeTextureLike | null = null
  let envRotation: ThreeMaterialLike['envMapRotation'] | undefined
  let allowRefractionMapping = false
  const materials = new WeakSet<ThreeMaterialLike>()

  const visit = (object: ThreeObject3DLike): void => {
    if (!object) return
    if (optionalBoolean(object.visible, 'object.visible') === false) return

    for (const material of objectMaterials(object.material)) {
      const materialEnvMap = material?.envMap
      if (!materialEnvMap) continue
      if (optionalBoolean(material.visible, 'material.visible') === false) continue
      if (!supportsNativeMaterialEnvironmentMap(material)) continue
      assertSupportedMaterialEnvironmentMap(material)
      if (hasNonZeroVector3Like(material.envMapRotation)) {
        if (envRotation && !sameVector3Like(envRotation, material.envMapRotation)) {
          throw new Error(
            'Multiple material.envMapRotation values are not supported by @headless-three/renderer yet. Use one shared material envMapRotation, scene.environmentRotation, or render separate passes.',
          )
        }
        envRotation = material.envMapRotation
      }
      if (envTex && envTex !== materialEnvMap) {
        throw new Error(
          'Multiple distinct material.envMap textures are not supported by @headless-three/renderer yet. Use one shared material envMap, scene.environment, or render separate passes until per-material IBL maps are supported.',
        )
      }
      envTex = materialEnvMap
      if (isRefractionEnvironmentMapping(materialEnvMap.mapping)) {
        allowRefractionMapping = true
      }
      materials.add(material)
    }

    for (const child of object.children ?? []) {
      visit(child)
    }
  }

  visit(scene as unknown as ThreeObject3DLike)
  if (!envTex) return null

  const envMap = extractEnvironmentMapFromTexture(envTex, 'material.envMap', 1, { allowRefraction: allowRefractionMapping })
  return envMap ? { envMap, materials, rotation: envRotation } : null
}

function objectMaterials(
  material: ThreeMaterialLike | ThreeMaterialLike[] | undefined,
): ThreeMaterialLike[] {
  if (!material) return []
  return Array.isArray(material) ? material.filter(Boolean) : [material]
}

function supportsNativeMaterialEnvironmentMap(material: ThreeMaterialLike): boolean {
  return material.isMeshStandardMaterial === true
    || material.isMeshPhysicalMaterial === true
    || material.isMeshBasicMaterial === true
    || material.isMeshPhongMaterial === true
    || material.isMeshLambertMaterial === true
}

function assertSupportedMaterialEnvironmentMap(material: ThreeMaterialLike): void {
  const usesRefraction = isRefractionEnvironmentMapping(material.envMap!.mapping)
  if (usesRefraction && material.isMeshBasicMaterial !== true) {
    throw new Error(
      'material.envMap refraction mappings are only supported for MeshBasicMaterial by @headless-three/renderer yet. Use a reflection mapping, remove material.envMap, or render this material separately.',
    )
  }
  assertSupportedEnvironmentTexture(material.envMap!, 'material.envMap', { allowRefraction: usesRefraction })
  const combine = material.combine ?? MultiplyOperation
  if (![MultiplyOperation, MixOperation, AddOperation].includes(combine)) {
    throw new Error(
      'material.envMap combine must be MultiplyOperation, MixOperation, or AddOperation for @headless-three/renderer.',
    )
  }
}

function extractEnvironmentMapFromTexture(
  envTex: ThreeTextureLike,
  label: string,
  intensity: number,
  options: { allowRefraction?: boolean } = {},
): EnvironmentMapInfo | null {
  assertSupportedEnvironmentTexture(envTex, label, options)
  if (isCubeEnvironmentTexture(envTex)) {
    const cube = cubeTextureToEquirectangular(envTex, label)
    return { data: cube.data, width: cube.width, height: cube.height, intensity, colorSpace: textureColorSpace(envTex) }
  }

  const image = (envTex as any).image ?? (envTex as any).source?.data
  if (!image) throw unsupportedTextureImageError(label, 'environment map rendering')

  // DataTexture: { data, width, height }
  if (image.data && image.width > 0 && image.height > 0) {
    const texType = (envTex as any).type ?? UnsignedByteType
    assertSupportedRawTextureType(texType, label, 'environment map rendering')
    const rawData = image.data as ArrayBufferView & { buffer: ArrayBuffer; byteOffset: number; byteLength: number }

    if (texType === HalfFloatType) {
      if (!(rawData instanceof Uint16Array)) {
        throw new Error(
          `${label} HalfFloatType environment maps must provide Uint16Array RGB or RGBA pixel data.`,
        )
      }
      const channels = rawTextureChannelCount(rawData, image.width, image.height, label, 'environment map rendering')
      let buf: Buffer
      if (channels === 3) {
        // Expand RGB16F → RGBA16F (half=0x3C00 is 1.0 for alpha)
        const pixels = image.width * image.height
        const out = new Uint16Array(pixels * 4)
        for (let i = 0; i < pixels; i++) {
          out[i * 4] = rawData[i * 3]
          out[i * 4 + 1] = rawData[i * 3 + 1]
          out[i * 4 + 2] = rawData[i * 3 + 2]
          out[i * 4 + 3] = 0x3C00 // 1.0 in half-float
        }
        buf = Buffer.from(out.buffer, out.byteOffset, out.byteLength)
      } else {
        buf = Buffer.from(rawData.buffer, rawData.byteOffset, rawData.byteLength)
      }
      return { data: buf, width: image.width, height: image.height, intensity, colorSpace: textureColorSpace(envTex) }
    }

    if (texType === FloatType) {
      if (!(rawData instanceof Float32Array)) {
        throw new Error(
          `${label} FloatType environment maps must provide Float32Array RGB or RGBA pixel data.`,
        )
      }
      const channels = rawTextureChannelCount(rawData, image.width, image.height, label, 'environment map rendering')
      let buf: Buffer
      if (channels === 3) {
        const pixels = image.width * image.height
        const out = new Float32Array(pixels * 4)
        for (let i = 0; i < pixels; i++) {
          out[i * 4] = rawData[i * 3]
          out[i * 4 + 1] = rawData[i * 3 + 1]
          out[i * 4 + 2] = rawData[i * 3 + 2]
          out[i * 4 + 3] = 1.0
        }
        buf = Buffer.from(out.buffer, out.byteOffset, out.byteLength)
      } else {
        buf = Buffer.from(rawData.buffer, rawData.byteOffset, rawData.byteLength)
      }
      return { data: buf, width: image.width, height: image.height, intensity, colorSpace: textureColorSpace(envTex) }
    }

    // UnsignedByteType / default: convert to RGBA8
    const rgba = toRgba8(rawData as any, image.width, image.height, { narrowChannels: false, type: texType })
    if (rgba) {
      return {
        data: Buffer.from(rgba.buffer, rgba.byteOffset, rgba.byteLength),
        width: image.width,
        height: image.height,
        intensity,
        colorSpace: textureColorSpace(envTex),
      }
    }
    throw unsupportedRawTextureDataError(label, 'environment map rendering')
  }

  // Encoded image buffer (e.g. loaded HDR encoded as PNG/EXR)
  if (Buffer.isBuffer(image)) {
    return { data: image, width: 0, height: 0, intensity, colorSpace: textureColorSpace(envTex) }
  }
  if (image instanceof Uint8Array && !((image as any).width > 0)) {
    return {
      data: Buffer.from(image.buffer, image.byteOffset, image.byteLength),
      width: 0,
      height: 0,
      intensity,
      colorSpace: textureColorSpace(envTex),
    }
  }

  throw unsupportedTextureImageError(label, 'environment map rendering')
}

function extractReflectionProbe(scene: ThreeSceneRootLike): { texture: ThreeTextureLike; intensity?: unknown; label: string } | null {
  const hints = scene.userData?.headlessThreeRenderer ?? scene.userData?.headlessRenderer ?? {}
  const probes = hints.reflectionProbes ?? hints.probes
  const probe = hints.reflectionProbe ?? (Array.isArray(probes) ? probes[0] : undefined)
  if (probe == null) return null

  const directTexture = textureLike(probe)
  if (directTexture) {
    return {
      texture: directTexture,
      intensity: undefined,
      label: 'reflectionProbe',
    }
  }

  const probeObject = probe as { texture?: unknown; map?: unknown; intensity?: unknown }
  if (probeObject.texture != null) {
    return {
      texture: requiredEnvironmentTexture(probeObject.texture, 'reflectionProbe.texture'),
      intensity: probeObject.intensity,
      label: 'reflectionProbe.texture',
    }
  }
  if (probeObject.map != null) {
    return {
      texture: requiredEnvironmentTexture(probeObject.map, 'reflectionProbe.map'),
      intensity: probeObject.intensity,
      label: 'reflectionProbe.map',
    }
  }

  const texture = requiredEnvironmentTexture(probe, 'reflectionProbe')
  return {
    texture,
    intensity: probeObject.intensity,
    label: 'reflectionProbe',
  }
}

export function materialForGroup(
  material: ThreeMaterialLike | ThreeMaterialLike[] | undefined,
  materialIndex: number,
): ThreeMaterialLike | undefined {
  if (Array.isArray(material)) {
    return material[materialIndex] ?? material[0]
  }
  return material
}

export function materialColor(material: ThreeMaterialLike | undefined): Color4 {
  const color = strictColorLikeToArray(material?.color, 'material.color') ?? [1, 1, 1, 1] as Color4
  color[3] = clamp01(optionalFiniteNumber(material?.opacity, 'material.opacity') ?? color[3] ?? 1)
  return color
}

export function extractPbrProperties(
  material: ThreeMaterialLike | undefined,
  context: MaterialExtractionContext = {},
): PbrProperties {
  if (!material) return {}
  const customFragmentShader = extractCustomFragmentShader(material)
  assertSupportedShaderMaterial(material, customFragmentShader)
  assertSupportedOnBeforeCompile(material, customFragmentShader)
  assertSupportedMaterialClass(material, customFragmentShader)
  assertSupportedMaterialState(material, context)
  assertCompatiblePackedPhysicalMapSamplers(material)
  optionalBoolean(material.visible, 'material.visible')
  optionalBoolean(material.vertexColors, 'material.vertexColors')
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
  const clearcoatMapInfo = extractTextureFromSlot(material.clearcoatMap)
  if (clearcoatMapInfo) {
    props.clearcoatMap = clearcoatMapInfo.data
    props.clearcoatMapWidth = clearcoatMapInfo.width
    props.clearcoatMapHeight = clearcoatMapInfo.height
    props.clearcoatMapWrapS = wrapModeToString(material.clearcoatMap?.wrapS)
    props.clearcoatMapWrapT = wrapModeToString(material.clearcoatMap?.wrapT)
    props.clearcoatMapMagFilter = filterModeToString(material.clearcoatMap?.magFilter)
    props.clearcoatMapMinFilter = minFilterModeToString(material.clearcoatMap)
    props.clearcoatMapAnisotropy = textureAnisotropy(material.clearcoatMap, 'material.clearcoatMap')
    props.clearcoatMapTransform = textureTransform(material.clearcoatMap, 'material.clearcoatMap')
    props.clearcoatMapUsesUv2 = textureUvChannel(material.clearcoatMap) > 0
  }
  const clearcoatRoughness = optionalFiniteNumber(material.clearcoatRoughness, 'material.clearcoatRoughness')
  if (clearcoatRoughness !== undefined) {
    props.clearcoatRoughness = clamp01(clearcoatRoughness)
  }
  const clearcoatRoughnessMapInfo = extractTextureFromSlot(material.clearcoatRoughnessMap)
  if (clearcoatRoughnessMapInfo) {
    props.clearcoatRoughnessMap = clearcoatRoughnessMapInfo.data
    props.clearcoatRoughnessMapWidth = clearcoatRoughnessMapInfo.width
    props.clearcoatRoughnessMapHeight = clearcoatRoughnessMapInfo.height
    props.clearcoatRoughnessMapWrapS = wrapModeToString(material.clearcoatRoughnessMap?.wrapS)
    props.clearcoatRoughnessMapWrapT = wrapModeToString(material.clearcoatRoughnessMap?.wrapT)
    props.clearcoatRoughnessMapMagFilter = filterModeToString(material.clearcoatRoughnessMap?.magFilter)
    props.clearcoatRoughnessMapMinFilter = minFilterModeToString(material.clearcoatRoughnessMap)
    props.clearcoatRoughnessMapAnisotropy = textureAnisotropy(material.clearcoatRoughnessMap, 'material.clearcoatRoughnessMap')
    props.clearcoatRoughnessMapTransform = textureTransform(material.clearcoatRoughnessMap, 'material.clearcoatRoughnessMap')
    props.clearcoatRoughnessMapUsesUv2 = textureUvChannel(material.clearcoatRoughnessMap) > 0
  }
  const clearcoatNormalMapInfo = extractTextureFromSlot(material.clearcoatNormalMap)
  if (clearcoatNormalMapInfo) {
    props.clearcoatNormalMap = clearcoatNormalMapInfo.data
    props.clearcoatNormalMapWidth = clearcoatNormalMapInfo.width
    props.clearcoatNormalMapHeight = clearcoatNormalMapInfo.height
    props.clearcoatNormalMapWrapS = wrapModeToString(material.clearcoatNormalMap?.wrapS)
    props.clearcoatNormalMapWrapT = wrapModeToString(material.clearcoatNormalMap?.wrapT)
    props.clearcoatNormalMapMagFilter = filterModeToString(material.clearcoatNormalMap?.magFilter)
    props.clearcoatNormalMapMinFilter = minFilterModeToString(material.clearcoatNormalMap)
    props.clearcoatNormalMapAnisotropy = textureAnisotropy(material.clearcoatNormalMap, 'material.clearcoatNormalMap')
    props.clearcoatNormalMapTransform = textureTransform(material.clearcoatNormalMap, 'material.clearcoatNormalMap')
    props.clearcoatNormalMapUsesUv2 = textureUvChannel(material.clearcoatNormalMap) > 0
  }
  if (material.clearcoatNormalScale) {
    props.clearcoatNormalScale = [
      finiteNumberOrDefault(material.clearcoatNormalScale.x, 'material.clearcoatNormalScale.x', 1),
      finiteNumberOrDefault(material.clearcoatNormalScale.y, 'material.clearcoatNormalScale.y', 1),
    ]
  }

  const sheenColor = strictColorLikeToArray(material.sheenColor, 'material.sheenColor')
  const sheen = clamp01(optionalFiniteNumber(material.sheen, 'material.sheen') ?? 0)
  if (sheenColor && sheen > 0) {
    props.sheenColor = [
      sheenColor[0] * sheen,
      sheenColor[1] * sheen,
      sheenColor[2] * sheen,
    ]
  }
  const sheenRoughness = optionalFiniteNumber(material.sheenRoughness, 'material.sheenRoughness')
  if (sheenRoughness !== undefined) {
    props.sheenRoughness = clamp01(sheenRoughness)
  }
  const sheenColorMapInfo = extractTextureFromSlot(material.sheenColorMap)
  if (sheenColorMapInfo) {
    props.sheenColorMap = sheenColorMapInfo.data
    props.sheenColorMapWidth = sheenColorMapInfo.width
    props.sheenColorMapHeight = sheenColorMapInfo.height
    props.sheenColorMapWrapS = wrapModeToString(material.sheenColorMap?.wrapS)
    props.sheenColorMapWrapT = wrapModeToString(material.sheenColorMap?.wrapT)
    props.sheenColorMapMagFilter = filterModeToString(material.sheenColorMap?.magFilter)
    props.sheenColorMapMinFilter = minFilterModeToString(material.sheenColorMap)
    props.sheenColorMapAnisotropy = textureAnisotropy(material.sheenColorMap, 'material.sheenColorMap')
    props.sheenColorMapTransform = textureTransform(material.sheenColorMap, 'material.sheenColorMap')
    props.sheenColorMapColorSpace = textureColorSpace(material.sheenColorMap)
    props.sheenColorMapUsesUv2 = textureUvChannel(material.sheenColorMap) > 0
  }
  const sheenRoughnessMapInfo = extractTextureFromSlot(material.sheenRoughnessMap)
  if (sheenRoughnessMapInfo) {
    props.sheenRoughnessMap = sheenRoughnessMapInfo.data
    props.sheenRoughnessMapWidth = sheenRoughnessMapInfo.width
    props.sheenRoughnessMapHeight = sheenRoughnessMapInfo.height
    props.sheenRoughnessMapWrapS = wrapModeToString(material.sheenRoughnessMap?.wrapS)
    props.sheenRoughnessMapWrapT = wrapModeToString(material.sheenRoughnessMap?.wrapT)
    props.sheenRoughnessMapMagFilter = filterModeToString(material.sheenRoughnessMap?.magFilter)
    props.sheenRoughnessMapMinFilter = minFilterModeToString(material.sheenRoughnessMap)
    props.sheenRoughnessMapAnisotropy = textureAnisotropy(material.sheenRoughnessMap, 'material.sheenRoughnessMap')
    props.sheenRoughnessMapTransform = textureTransform(material.sheenRoughnessMap, 'material.sheenRoughnessMap')
    props.sheenRoughnessMapUsesUv2 = textureUvChannel(material.sheenRoughnessMap) > 0
  }

  const anisotropy = optionalFiniteNumber(material.anisotropy, 'material.anisotropy')
  if (anisotropy !== undefined) {
    props.anisotropy = clamp01(Math.abs(anisotropy))
  }
  const anisotropyRotation = optionalFiniteNumber(material.anisotropyRotation, 'material.anisotropyRotation')
  if (anisotropyRotation !== undefined) {
    props.anisotropyRotation = anisotropyRotation
  }
  const anisotropyMapInfo = extractTextureFromSlot(material.anisotropyMap)
  if (anisotropyMapInfo) {
    props.anisotropyMap = anisotropyMapInfo.data
    props.anisotropyMapWidth = anisotropyMapInfo.width
    props.anisotropyMapHeight = anisotropyMapInfo.height
    props.anisotropyMapWrapS = wrapModeToString(material.anisotropyMap?.wrapS)
    props.anisotropyMapWrapT = wrapModeToString(material.anisotropyMap?.wrapT)
    props.anisotropyMapMagFilter = filterModeToString(material.anisotropyMap?.magFilter)
    props.anisotropyMapMinFilter = minFilterModeToString(material.anisotropyMap)
    props.anisotropyMapAnisotropy = textureAnisotropy(material.anisotropyMap, 'material.anisotropyMap')
    props.anisotropyMapTransform = textureTransform(material.anisotropyMap, 'material.anisotropyMap')
    props.anisotropyMapUsesUv2 = textureUvChannel(material.anisotropyMap) > 0
  }

  const iridescence = optionalFiniteNumber(material.iridescence, 'material.iridescence')
  if (iridescence !== undefined) {
    props.iridescence = clamp01(iridescence)
  }
  const iridescenceMapInfo = extractTextureFromSlot(material.iridescenceMap)
  if (iridescenceMapInfo) {
    props.iridescenceMap = iridescenceMapInfo.data
    props.iridescenceMapWidth = iridescenceMapInfo.width
    props.iridescenceMapHeight = iridescenceMapInfo.height
    props.iridescenceMapWrapS = wrapModeToString(material.iridescenceMap?.wrapS)
    props.iridescenceMapWrapT = wrapModeToString(material.iridescenceMap?.wrapT)
    props.iridescenceMapMagFilter = filterModeToString(material.iridescenceMap?.magFilter)
    props.iridescenceMapMinFilter = minFilterModeToString(material.iridescenceMap)
    props.iridescenceMapAnisotropy = textureAnisotropy(material.iridescenceMap, 'material.iridescenceMap')
    props.iridescenceMapTransform = textureTransform(material.iridescenceMap, 'material.iridescenceMap')
    props.iridescenceMapUsesUv2 = textureUvChannel(material.iridescenceMap) > 0
  }
  const iridescenceIor = optionalFiniteNumber(material.iridescenceIOR, 'material.iridescenceIOR')
  if (iridescenceIor !== undefined) {
    props.iridescenceIor = Math.max(1, Math.min(2.333, iridescenceIor))
  }
  const iridescenceThicknessRange = material.iridescenceThicknessRange
  if (iridescenceThicknessRange && iridescenceThicknessRange.length >= 2) {
    const min = requiredFiniteNumber(iridescenceThicknessRange[0], 'material.iridescenceThicknessRange[0]')
    const max = requiredFiniteNumber(iridescenceThicknessRange[1], 'material.iridescenceThicknessRange[1]')
    props.iridescenceThicknessMin = Math.max(0, min)
    props.iridescenceThicknessMax = Math.max(props.iridescenceThicknessMin, max)
  }
  const iridescenceThicknessMapInfo = extractTextureFromSlot(material.iridescenceThicknessMap)
  if (iridescenceThicknessMapInfo) {
    props.iridescenceThicknessMap = iridescenceThicknessMapInfo.data
    props.iridescenceThicknessMapWidth = iridescenceThicknessMapInfo.width
    props.iridescenceThicknessMapHeight = iridescenceThicknessMapInfo.height
    props.iridescenceThicknessMapWrapS = wrapModeToString(material.iridescenceThicknessMap?.wrapS)
    props.iridescenceThicknessMapWrapT = wrapModeToString(material.iridescenceThicknessMap?.wrapT)
    props.iridescenceThicknessMapMagFilter = filterModeToString(material.iridescenceThicknessMap?.magFilter)
    props.iridescenceThicknessMapMinFilter = minFilterModeToString(material.iridescenceThicknessMap)
    props.iridescenceThicknessMapAnisotropy = textureAnisotropy(material.iridescenceThicknessMap, 'material.iridescenceThicknessMap')
    props.iridescenceThicknessMapTransform = textureTransform(material.iridescenceThicknessMap, 'material.iridescenceThicknessMap')
    props.iridescenceThicknessMapUsesUv2 = textureUvChannel(material.iridescenceThicknessMap) > 0
  }

  const transmission = optionalFiniteNumber(material.transmission, 'material.transmission')
  if (transmission !== undefined) {
    props.transmission = clamp01(transmission)
  }
  const dispersion = optionalFiniteNumber(material.dispersion, 'material.dispersion')
  if (dispersion !== undefined) {
    props.dispersion = Math.max(0, dispersion)
  }
  const transmissionMapInfo = extractTextureFromSlot(material.transmissionMap)
  if (transmissionMapInfo) {
    props.transmissionMap = transmissionMapInfo.data
    props.transmissionMapWidth = transmissionMapInfo.width
    props.transmissionMapHeight = transmissionMapInfo.height
    props.transmissionMapWrapS = wrapModeToString(material.transmissionMap?.wrapS)
    props.transmissionMapWrapT = wrapModeToString(material.transmissionMap?.wrapT)
    props.transmissionMapMagFilter = filterModeToString(material.transmissionMap?.magFilter)
    props.transmissionMapMinFilter = minFilterModeToString(material.transmissionMap)
    props.transmissionMapAnisotropy = textureAnisotropy(material.transmissionMap, 'material.transmissionMap')
    props.transmissionMapTransform = textureTransform(material.transmissionMap, 'material.transmissionMap')
    props.transmissionMapUsesUv2 = textureUvChannel(material.transmissionMap) > 0
  }
  const ior = optionalFiniteNumber(material.ior, 'material.ior')
  if (ior !== undefined) {
    props.ior = Math.max(1, Math.min(2.333, ior))
  }
  const thickness = optionalFiniteNumber(material.thickness, 'material.thickness')
  if (thickness !== undefined) {
    props.thickness = Math.max(0, thickness)
  }
  const thicknessMapInfo = extractTextureFromSlot(material.thicknessMap)
  if (thicknessMapInfo) {
    props.thicknessMap = thicknessMapInfo.data
    props.thicknessMapWidth = thicknessMapInfo.width
    props.thicknessMapHeight = thicknessMapInfo.height
    props.thicknessMapWrapS = wrapModeToString(material.thicknessMap?.wrapS)
    props.thicknessMapWrapT = wrapModeToString(material.thicknessMap?.wrapT)
    props.thicknessMapMagFilter = filterModeToString(material.thicknessMap?.magFilter)
    props.thicknessMapMinFilter = minFilterModeToString(material.thicknessMap)
    props.thicknessMapAnisotropy = textureAnisotropy(material.thicknessMap, 'material.thicknessMap')
    props.thicknessMapTransform = textureTransform(material.thicknessMap, 'material.thicknessMap')
    props.thicknessMapUsesUv2 = textureUvChannel(material.thicknessMap) > 0
  }
  const attenuationDistance = optionalFiniteNumberOrInfinityDefault(
    material.attenuationDistance,
    'material.attenuationDistance',
  )
  if (attenuationDistance !== undefined) {
    props.attenuationDistance = Math.max(0, attenuationDistance)
  }
  const attenuationColor = strictColorLikeToArray(material.attenuationColor, 'material.attenuationColor')
  if (attenuationColor) {
    props.attenuationColor = [attenuationColor[0], attenuationColor[1], attenuationColor[2]]
  }
  const physicalSpecularColor = strictColorLikeToArray(material.specularColor, 'material.specularColor')
  if (physicalSpecularColor) {
    props.physicalSpecularColor = [
      physicalSpecularColor[0],
      physicalSpecularColor[1],
      physicalSpecularColor[2],
    ]
  }
  const specularIntensity = optionalFiniteNumber(material.specularIntensity, 'material.specularIntensity')
  if (specularIntensity !== undefined) {
    props.physicalSpecularIntensity = clamp01(specularIntensity)
  }
  const specularColorMapInfo = extractTextureFromSlot(material.specularColorMap)
  if (specularColorMapInfo) {
    props.specularColorMap = specularColorMapInfo.data
    props.specularColorMapWidth = specularColorMapInfo.width
    props.specularColorMapHeight = specularColorMapInfo.height
    props.specularColorMapWrapS = wrapModeToString(material.specularColorMap?.wrapS)
    props.specularColorMapWrapT = wrapModeToString(material.specularColorMap?.wrapT)
    props.specularColorMapMagFilter = filterModeToString(material.specularColorMap?.magFilter)
    props.specularColorMapMinFilter = minFilterModeToString(material.specularColorMap)
    props.specularColorMapAnisotropy = textureAnisotropy(material.specularColorMap, 'material.specularColorMap')
    props.specularColorMapTransform = textureTransform(material.specularColorMap, 'material.specularColorMap')
    props.specularColorMapColorSpace = textureColorSpace(material.specularColorMap)
    props.specularColorMapUsesUv2 = textureUvChannel(material.specularColorMap) > 0
  }
  const specularIntensityMapInfo = extractTextureFromSlot(material.specularIntensityMap)
  if (specularIntensityMapInfo) {
    props.specularIntensityMap = specularIntensityMapInfo.data
    props.specularIntensityMapWidth = specularIntensityMapInfo.width
    props.specularIntensityMapHeight = specularIntensityMapInfo.height
    props.specularIntensityMapWrapS = wrapModeToString(material.specularIntensityMap?.wrapS)
    props.specularIntensityMapWrapT = wrapModeToString(material.specularIntensityMap?.wrapT)
    props.specularIntensityMapMagFilter = filterModeToString(material.specularIntensityMap?.magFilter)
    props.specularIntensityMapMinFilter = minFilterModeToString(material.specularIntensityMap)
    props.specularIntensityMapAnisotropy = textureAnisotropy(material.specularIntensityMap, 'material.specularIntensityMap')
    props.specularIntensityMapTransform = textureTransform(material.specularIntensityMap, 'material.specularIntensityMap')
    props.specularIntensityMapUsesUv2 = textureUvChannel(material.specularIntensityMap) > 0
  }

  const specularColor = strictColorLikeToArray(material.specular, 'material.specular')
  if (specularColor || material.isMeshPhongMaterial) {
    const color = specularColor ?? [17 / 255, 17 / 255, 17 / 255, 1]
    props.specularColor = [color[0], color[1], color[2]]
  }
  const shininess = material.isMeshPhongMaterial
    ? finiteNumberOrDefault(material.shininess, 'material.shininess', 30)
    : optionalFiniteNumber(material.shininess, 'material.shininess')
  if (shininess !== undefined) {
    props.shininess = Math.max(0.0001, shininess)
  }

  const emissive = strictColorLikeToArray(material.emissive, 'material.emissive')
  if (emissive) {
    props.emissive = [emissive[0], emissive[1], emissive[2]]
    props.emissiveIntensity = finiteNumberOrDefault(material.emissiveIntensity, 'material.emissiveIntensity', 1)
  }

  const normalMapInfo = extractTextureFromSlot(material.normalMap)
  if (normalMapInfo) {
    props.normalMap = normalMapInfo.data
    props.normalMapWidth = normalMapInfo.width
    props.normalMapHeight = normalMapInfo.height
    props.normalMapWrapS = wrapModeToString(material.normalMap?.wrapS)
    props.normalMapWrapT = wrapModeToString(material.normalMap?.wrapT)
    props.normalMapMagFilter = filterModeToString(material.normalMap?.magFilter)
    props.normalMapMinFilter = minFilterModeToString(material.normalMap)
    props.normalMapAnisotropy = textureAnisotropy(material.normalMap, 'material.normalMap')
    props.normalMapTransform = textureTransform(material.normalMap, 'material.normalMap')
    props.normalMapUsesUv2 = textureUvChannel(material.normalMap) > 0
    props.normalMapType = materialNormalMapType(material)
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
  const bumpMapInfo = extractTextureFromSlot(material.bumpMap)
  if (bumpMapInfo) {
    props.bumpMap = bumpMapInfo.data
    props.bumpMapWidth = bumpMapInfo.width
    props.bumpMapHeight = bumpMapInfo.height
    props.bumpMapWrapS = wrapModeToString(material.bumpMap?.wrapS)
    props.bumpMapWrapT = wrapModeToString(material.bumpMap?.wrapT)
    props.bumpMapMagFilter = filterModeToString(material.bumpMap?.magFilter)
    props.bumpMapMinFilter = minFilterModeToString(material.bumpMap)
    props.bumpMapAnisotropy = textureAnisotropy(material.bumpMap, 'material.bumpMap')
    props.bumpMapTransform = textureTransform(material.bumpMap, 'material.bumpMap')
    props.bumpMapUsesUv2 = textureUvChannel(material.bumpMap) > 0
    props.bumpScale = finiteNumberOrDefault(material.bumpScale, 'material.bumpScale', 1)
  }
  if (material.isMeshMatcapMaterial) {
    const matcapMapInfo = extractTextureFromSlot(material.map)
    if (matcapMapInfo) {
      props.matcapMap = matcapMapInfo.data
      props.matcapMapWidth = matcapMapInfo.width
      props.matcapMapHeight = matcapMapInfo.height
      props.matcapMapWrapS = wrapModeToString(material.map?.wrapS)
      props.matcapMapWrapT = wrapModeToString(material.map?.wrapT)
      props.matcapMapMagFilter = filterModeToString(material.map?.magFilter)
      props.matcapMapMinFilter = minFilterModeToString(material.map)
      props.matcapMapAnisotropy = textureAnisotropy(material.map, 'material.map')
      props.matcapMapTransform = textureTransform(material.map, 'material.map')
      props.matcapMapColorSpace = textureColorSpace(material.map)
      props.matcapMapUsesUv2 = textureUvChannel(material.map) > 0
    }
  }
  if (material.isMeshDepthMaterial) {
    props.depthPacking = materialDepthPacking(material) ?? BasicDepthPacking
  }
  if (material.isMeshDistanceMaterial) {
    const hints = material.userData?.headlessThreeRenderer ?? material.userData?.headlessRenderer ?? {}
    const hintsLabel = material.userData?.headlessThreeRenderer != null
      ? 'material.userData.headlessThreeRenderer'
      : 'material.userData.headlessRenderer'
    const referencePosition = vector3LikeToArray(
      material.referencePosition ?? hints.referencePosition ?? hints.distanceReferencePosition,
    )
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

  const gradientMapInfo = extractTextureFromSlot(material.gradientMap)
  if (gradientMapInfo) {
    props.gradientMap = gradientMapInfo.data
    props.gradientMapWidth = gradientMapInfo.width
    props.gradientMapHeight = gradientMapInfo.height
    props.gradientMapWrapS = wrapModeToString(material.gradientMap?.wrapS)
    props.gradientMapWrapT = wrapModeToString(material.gradientMap?.wrapT)
    props.gradientMapMagFilter = filterModeToString(material.gradientMap?.magFilter)
    props.gradientMapMinFilter = minFilterModeToString(material.gradientMap)
    props.gradientMapAnisotropy = textureAnisotropy(material.gradientMap, 'material.gradientMap')
    props.gradientMapColorSpace = textureColorSpace(material.gradientMap)
  }

  const displacementMapInfo = extractTextureFromSlot(material.displacementMap)
  if (displacementMapInfo) {
    props.displacementMap = displacementMapInfo.data
    props.displacementMapWidth = displacementMapInfo.width
    props.displacementMapHeight = displacementMapInfo.height
    props.displacementMapWrapS = wrapModeToString(material.displacementMap?.wrapS)
    props.displacementMapWrapT = wrapModeToString(material.displacementMap?.wrapT)
    props.displacementMapMagFilter = filterModeToString(material.displacementMap?.magFilter)
    props.displacementMapMinFilter = minFilterModeToString(material.displacementMap)
    props.displacementMapAnisotropy = textureAnisotropy(material.displacementMap, 'material.displacementMap')
    props.displacementMapTransform = textureTransform(material.displacementMap, 'material.displacementMap')
    props.displacementMapUsesUv2 = textureUvChannel(material.displacementMap) > 0
    props.displacementScale = finiteNumberOrDefault(material.displacementScale, 'material.displacementScale', 1)
    props.displacementBias = finiteNumberOrDefault(material.displacementBias, 'material.displacementBias', 0)
  }

  const mrMapInfo = extractTextureFromSlot(material.metalnessMap ?? material.roughnessMap)
  if (mrMapInfo) {
    const mrMap = material.metalnessMap ?? material.roughnessMap
    props.metallicRoughnessTexture = mrMapInfo.data
    props.metallicRoughnessTextureWidth = mrMapInfo.width
    props.metallicRoughnessTextureHeight = mrMapInfo.height
    props.metallicRoughnessTextureWrapS = wrapModeToString(mrMap?.wrapS)
    props.metallicRoughnessTextureWrapT = wrapModeToString(mrMap?.wrapT)
    props.metallicRoughnessTextureMagFilter = filterModeToString(mrMap?.magFilter)
    props.metallicRoughnessTextureMinFilter = minFilterModeToString(mrMap)
    props.metallicRoughnessTextureAnisotropy = textureAnisotropy(
      mrMap,
      material.metalnessMap ? 'material.metalnessMap' : 'material.roughnessMap',
    )
    props.metallicRoughnessTextureTransform = textureTransform(
      mrMap,
      material.metalnessMap ? 'material.metalnessMap' : 'material.roughnessMap',
    )
    props.metallicRoughnessTextureUsesUv2 = textureUvChannel(mrMap) > 0
  }

  const specularMapInfo = extractTextureFromSlot(material.specularMap)
  if (specularMapInfo) {
    props.specularMap = specularMapInfo.data
    props.specularMapWidth = specularMapInfo.width
    props.specularMapHeight = specularMapInfo.height
    props.specularMapWrapS = wrapModeToString(material.specularMap?.wrapS)
    props.specularMapWrapT = wrapModeToString(material.specularMap?.wrapT)
    props.specularMapMagFilter = filterModeToString(material.specularMap?.magFilter)
    props.specularMapMinFilter = minFilterModeToString(material.specularMap)
    props.specularMapAnisotropy = textureAnisotropy(material.specularMap, 'material.specularMap')
    props.specularMapTransform = textureTransform(material.specularMap, 'material.specularMap')
    props.specularMapUsesUv2 = textureUvChannel(material.specularMap) > 0
  }

  const emissiveMapInfo = extractTextureFromSlot(material.emissiveMap)
  if (emissiveMapInfo) {
    props.emissiveMap = emissiveMapInfo.data
    props.emissiveMapWidth = emissiveMapInfo.width
    props.emissiveMapHeight = emissiveMapInfo.height
    props.emissiveMapWrapS = wrapModeToString(material.emissiveMap?.wrapS)
    props.emissiveMapWrapT = wrapModeToString(material.emissiveMap?.wrapT)
    props.emissiveMapMagFilter = filterModeToString(material.emissiveMap?.magFilter)
    props.emissiveMapMinFilter = minFilterModeToString(material.emissiveMap)
    props.emissiveMapAnisotropy = textureAnisotropy(material.emissiveMap, 'material.emissiveMap')
    props.emissiveMapTransform = textureTransform(material.emissiveMap, 'material.emissiveMap')
    props.emissiveMapColorSpace = textureColorSpace(material.emissiveMap)
    props.emissiveMapUsesUv2 = textureUvChannel(material.emissiveMap) > 0
  }

  const aoMapInfo = extractTextureFromSlot(material.aoMap)
  if (aoMapInfo) {
    props.aoMap = aoMapInfo.data
    props.aoMapWidth = aoMapInfo.width
    props.aoMapHeight = aoMapInfo.height
    props.aoMapWrapS = wrapModeToString(material.aoMap?.wrapS)
    props.aoMapWrapT = wrapModeToString(material.aoMap?.wrapT)
    props.aoMapMagFilter = filterModeToString(material.aoMap?.magFilter)
    props.aoMapMinFilter = minFilterModeToString(material.aoMap)
    props.aoMapAnisotropy = textureAnisotropy(material.aoMap, 'material.aoMap')
    props.aoMapTransform = textureTransform(material.aoMap, 'material.aoMap')
    props.aoMapUsesUv2 = textureUvChannel(material.aoMap) > 0
    props.aoMapIntensity = finiteNumberOrDefault(material.aoMapIntensity, 'material.aoMapIntensity', 1)
  }

  const lightMapInfo = extractTextureFromSlot(material.lightMap)
  if (lightMapInfo) {
    props.lightMap = lightMapInfo.data
    props.lightMapWidth = lightMapInfo.width
    props.lightMapHeight = lightMapInfo.height
    props.lightMapWrapS = wrapModeToString(material.lightMap?.wrapS)
    props.lightMapWrapT = wrapModeToString(material.lightMap?.wrapT)
    props.lightMapMagFilter = filterModeToString(material.lightMap?.magFilter)
    props.lightMapMinFilter = minFilterModeToString(material.lightMap)
    props.lightMapAnisotropy = textureAnisotropy(material.lightMap, 'material.lightMap')
    props.lightMapTransform = textureTransform(material.lightMap, 'material.lightMap')
    props.lightMapColorSpace = textureColorSpace(material.lightMap)
    props.lightMapUsesUv2 = textureUvChannel(material.lightMap) > 0
    props.lightMapIntensity = finiteNumberOrDefault(material.lightMapIntensity, 'material.lightMapIntensity', 1)
  }

  const alphaMapInfo = extractTextureFromSlot(material.alphaMap)
  if (alphaMapInfo) {
    props.alphaMap = alphaMapInfo.data
    props.alphaMapWidth = alphaMapInfo.width
    props.alphaMapHeight = alphaMapInfo.height
    props.alphaMapWrapS = wrapModeToString(material.alphaMap?.wrapS)
    props.alphaMapWrapT = wrapModeToString(material.alphaMap?.wrapT)
    props.alphaMapMagFilter = filterModeToString(material.alphaMap?.magFilter)
    props.alphaMapMinFilter = minFilterModeToString(material.alphaMap)
    props.alphaMapAnisotropy = textureAnisotropy(material.alphaMap, 'material.alphaMap')
    props.alphaMapTransform = textureTransform(material.alphaMap, 'material.alphaMap')
    props.alphaMapUsesUv2 = textureUvChannel(material.alphaMap) > 0
  }

  const alphaTest = optionalFiniteNumber(material.alphaTest, 'material.alphaTest')
  if (alphaTest !== undefined && alphaTest > 0) {
    props.alphaTest = clamp01(alphaTest)
  }
  if (optionalBoolean(material.alphaHash, 'material.alphaHash') === true) {
    props.alphaHash = true
  }
  if (optionalBoolean(material.alphaToCoverage, 'material.alphaToCoverage') === true) {
    props.alphaToCoverage = true
  }
  if (optionalBoolean(material.premultipliedAlpha, 'material.premultipliedAlpha') === true) {
    props.premultipliedAlpha = true
  }
  const transparent = optionalBoolean(material.transparent, 'material.transparent')
  if (transparent !== undefined) {
    props.transparent = transparent
  }
  const blending = materialBlending(material)
  if (blending) {
    props.blending = blending
    if (blending === 'custom') {
      props.blendEquation = materialBlendEquationOrDefault(material.blendEquation, 'material.blendEquation', AddEquation)
      props.blendSrc = materialBlendFactorOrDefault(material.blendSrc, 'material.blendSrc', SrcAlphaFactor)
      props.blendDst = materialBlendFactorOrDefault(material.blendDst, 'material.blendDst', OneMinusSrcAlphaFactor)
      if (material.blendEquationAlpha != null) {
        props.blendEquationAlpha = materialBlendEquation(material.blendEquationAlpha, 'material.blendEquationAlpha')
      }
      if (material.blendSrcAlpha != null) {
        props.blendSrcAlpha = materialBlendFactor(material.blendSrcAlpha, 'material.blendSrcAlpha')
      }
      if (material.blendDstAlpha != null) {
        props.blendDstAlpha = materialBlendFactor(material.blendDstAlpha, 'material.blendDstAlpha')
      }
      const blendColor = strictColorLikeToArray(material.blendColor, 'material.blendColor')
      if (blendColor) {
        props.blendColor = [blendColor[0], blendColor[1], blendColor[2]]
      }
      const blendAlpha = optionalFiniteNumber(material.blendAlpha, 'material.blendAlpha')
      if (blendAlpha !== undefined) {
        props.blendAlpha = clamp01(blendAlpha)
      }
    }
  }
  const depthTest = optionalBoolean(material.depthTest, 'material.depthTest')
  if (depthTest !== undefined) {
    props.depthTest = depthTest
  }
  const depthFunc = materialDepthFunc(material)
  if (depthFunc) {
    props.depthFunc = depthFunc
  }
  const depthWrite = optionalBoolean(material.depthWrite, 'material.depthWrite')
  if (depthWrite !== undefined) {
    props.depthWrite = depthWrite
  }
  const colorWrite = optionalBoolean(material.colorWrite, 'material.colorWrite')
  if (colorWrite !== undefined) {
    props.colorWrite = colorWrite
  }
  const polygonOffset = optionalBoolean(material.polygonOffset, 'material.polygonOffset')
  if (polygonOffset !== undefined) {
    props.polygonOffset = polygonOffset
    const polygonOffsetFactor = optionalFiniteNumber(material.polygonOffsetFactor, 'material.polygonOffsetFactor')
    if (polygonOffsetFactor !== undefined) {
      props.polygonOffsetFactor = polygonOffsetFactor
    }
    const polygonOffsetUnits = optionalFiniteNumber(material.polygonOffsetUnits, 'material.polygonOffsetUnits')
    if (polygonOffsetUnits !== undefined) {
      props.polygonOffsetUnits = polygonOffsetUnits
    }
  }
  const stencilWrite = optionalBoolean(material.stencilWrite, 'material.stencilWrite')
  if (stencilWrite !== undefined) {
    props.stencilWrite = stencilWrite
  }
  const stencilWriteMask = optionalFiniteNumber(material.stencilWriteMask, 'material.stencilWriteMask')
  if (stencilWriteMask !== undefined) {
    props.stencilWriteMask = Math.trunc(stencilWriteMask)
  }
  if (material.stencilFunc != null) {
    props.stencilFunc = materialStencilFunc(material.stencilFunc, 'material.stencilFunc')
  }
  const stencilRef = optionalFiniteNumber(material.stencilRef, 'material.stencilRef')
  if (stencilRef !== undefined) {
    props.stencilRef = Math.trunc(stencilRef)
  }
  const stencilFuncMask = optionalFiniteNumber(material.stencilFuncMask, 'material.stencilFuncMask')
  if (stencilFuncMask !== undefined) {
    props.stencilFuncMask = Math.trunc(stencilFuncMask)
  }
  if (material.stencilFail != null) {
    props.stencilFail = materialStencilOperation(material.stencilFail, 'material.stencilFail')
  }
  if (material.stencilZFail != null) {
    props.stencilZFail = materialStencilOperation(material.stencilZFail, 'material.stencilZFail')
  }
  if (material.stencilZPass != null) {
    props.stencilZPass = materialStencilOperation(material.stencilZPass, 'material.stencilZPass')
  }
  const side = materialSide(material)
  if (side) {
    props.side = side
  }
  const shadowSide = materialShadowSide(material)
  if (shadowSide) {
    props.shadowSide = shadowSide
  }
  if (optionalBoolean(material.flatShading, 'material.flatShading') === true) {
    props.flatShading = true
  }
  if (optionalBoolean(material.fog, 'material.fog') === false) {
    props.fog = false
  }

  // Shading model: MeshBasicMaterial is unlit, MeshDepthMaterial outputs
  // normalized depth, MeshLambertMaterial is diffuse-only, MeshNormalMaterial
  // outputs view-space normals, and MeshMatcapMaterial samples a baked lighting
  // texture from view-space normals. Everything else
  // (MeshStandardMaterial / MeshPhysicalMaterial / unknown) uses the default PBR path.
  if (customFragmentShader && shaderMaterialKind(material)) {
    props.shadingModel = 'basic'
  } else if (material.isMeshBasicMaterial || material.isSpriteMaterial) {
    props.shadingModel = 'basic'
  } else if (material.isMeshDepthMaterial) {
    props.shadingModel = 'depth'
  } else if (material.isMeshDistanceMaterial) {
    props.shadingModel = 'distance'
  } else if (material.isMeshLambertMaterial) {
    props.shadingModel = 'lambert'
  } else if (material.isMeshNormalMaterial) {
    props.shadingModel = 'normal'
  } else if (material.isMeshMatcapMaterial) {
    props.shadingModel = 'matcap'
  } else if (material.isMeshPhongMaterial) {
    props.shadingModel = 'phong'
  } else if (material.isMeshToonMaterial) {
    props.shadingModel = 'toon'
  } else if (material.isShadowMaterial) {
    props.shadingModel = 'shadow'
  }

  if (customFragmentShader) {
    props.customFragmentShader = customFragmentShader
  }

  return props
}

function materialBlending(material: ThreeMaterialLike): string | undefined {
  switch (material.blending) {
    case NoBlending:
      return 'none'
    case NormalBlending:
      return 'normal'
    case AdditiveBlending:
      return 'additive'
    case SubtractiveBlending:
      return 'subtractive'
    case MultiplyBlending:
      return 'multiply'
    case CustomBlending:
      return 'custom'
    default:
      if (material.blending != null) {
        throw new Error(
          `material.blending ${String(material.blending)} is not supported by @headless-three/renderer. Use a Three.js blending constant such as NormalBlending, AdditiveBlending, or CustomBlending.`,
        )
      }
      return undefined
  }
}

function materialBlendEquationOrDefault(value: unknown, label: string, fallback: number): number {
  if (value == null) return fallback
  return materialBlendEquation(value, label)
}

function materialBlendEquation(value: unknown, label: string): number {
  if (
    value === AddEquation ||
    value === SubtractEquation ||
    value === ReverseSubtractEquation ||
    value === MinEquation ||
    value === MaxEquation
  ) {
    return value
  }
  throw new Error(
    `${label} ${String(value)} is not supported by @headless-three/renderer. Use a Three.js blend equation constant such as AddEquation, SubtractEquation, ReverseSubtractEquation, MinEquation, or MaxEquation.`,
  )
}

function materialBlendFactorOrDefault(value: unknown, label: string, fallback: number): number {
  if (value == null) return fallback
  return materialBlendFactor(value, label)
}

function materialBlendFactor(value: unknown, label: string): number {
  if (
    value === ZeroFactor ||
    value === OneFactor ||
    value === SrcColorFactor ||
    value === OneMinusSrcColorFactor ||
    value === SrcAlphaFactor ||
    value === OneMinusSrcAlphaFactor ||
    value === DstAlphaFactor ||
    value === OneMinusDstAlphaFactor ||
    value === DstColorFactor ||
    value === OneMinusDstColorFactor ||
    value === SrcAlphaSaturateFactor ||
    value === ConstantColorFactor ||
    value === OneMinusConstantColorFactor ||
    value === ConstantAlphaFactor ||
    value === OneMinusConstantAlphaFactor
  ) {
    return value
  }
  throw new Error(
    `${label} ${String(value)} is not supported by @headless-three/renderer. Use a Three.js blend factor constant such as SrcAlphaFactor, OneMinusSrcAlphaFactor, OneFactor, or ZeroFactor.`,
  )
}

function materialDepthFunc(material: ThreeMaterialLike): string | undefined {
  if (material.depthFunc == null) return undefined
  switch (material.depthFunc) {
    case NeverDepth:
      return 'never'
    case AlwaysDepth:
      return 'always'
    case LessDepth:
      return 'less'
    case LessEqualDepth:
      return 'less-equal'
    case EqualDepth:
      return 'equal'
    case GreaterEqualDepth:
      return 'greater-equal'
    case GreaterDepth:
      return 'greater'
    case NotEqualDepth:
      return 'not-equal'
    default:
      throw new Error(
        `material.depthFunc ${String(material.depthFunc)} is not supported by @headless-three/renderer. Use a Three.js depth comparison constant such as LessEqualDepth, GreaterDepth, or AlwaysDepth.`,
      )
  }
}

function materialStencilFunc(value: unknown, label: string): number {
  if (
    value === NeverStencilFunc ||
    value === LessStencilFunc ||
    value === EqualStencilFunc ||
    value === LessEqualStencilFunc ||
    value === GreaterStencilFunc ||
    value === NotEqualStencilFunc ||
    value === GreaterEqualStencilFunc ||
    value === AlwaysStencilFunc
  ) {
    return value
  }
  throw new Error(
    `${label} ${String(value)} is not supported by @headless-three/renderer. Use a Three.js stencil function constant such as AlwaysStencilFunc, EqualStencilFunc, or NotEqualStencilFunc.`,
  )
}

function materialStencilOperation(value: unknown, label: string): number {
  if (
    value === ZeroStencilOp ||
    value === KeepStencilOp ||
    value === ReplaceStencilOp ||
    value === IncrementStencilOp ||
    value === DecrementStencilOp ||
    value === IncrementWrapStencilOp ||
    value === DecrementWrapStencilOp ||
    value === InvertStencilOp
  ) {
    return value
  }
  throw new Error(
    `${label} ${String(value)} is not supported by @headless-three/renderer. Use a Three.js stencil operation constant such as KeepStencilOp, ReplaceStencilOp, or InvertStencilOp.`,
  )
}

function materialSide(material: ThreeMaterialLike): string | undefined {
  if (material.side == null) return undefined
  switch (material.side) {
    case FrontSide:
      return 'front'
    case BackSide:
      return 'back'
    case DoubleSide:
      return 'double'
    default:
      throw new Error(
        `material.side ${String(material.side)} is not supported by @headless-three/renderer. Use FrontSide, BackSide, or DoubleSide.`,
      )
  }
}

function materialDepthPacking(material: ThreeMaterialLike): number | undefined {
  if (material.depthPacking == null) return undefined
  switch (material.depthPacking) {
    case BasicDepthPacking:
    case RGBADepthPacking:
    case RGBDepthPacking:
    case RGDepthPacking:
      return material.depthPacking
    default:
      throw new Error(
        `material.depthPacking ${String(material.depthPacking)} is not supported by @headless-three/renderer. Use BasicDepthPacking, RGBADepthPacking, RGBDepthPacking, or RGDepthPacking.`,
      )
  }
}

function materialNormalMapType(material: ThreeMaterialLike): 'tangent' | 'object' {
  if (material.normalMapType == null) return 'tangent'
  switch (material.normalMapType) {
    case TangentSpaceNormalMap:
      return 'tangent'
    case ObjectSpaceNormalMap:
      return 'object'
    default:
      throw new Error(
        `material.normalMapType ${String(material.normalMapType)} is not supported by @headless-three/renderer. Use TangentSpaceNormalMap or ObjectSpaceNormalMap.`,
      )
  }
}

export function materialShadowSide(material: ThreeMaterialLike | undefined): string | undefined {
  if (!material || material.shadowSide == null) return undefined
  switch (material.shadowSide) {
    case FrontSide:
      return 'front'
    case BackSide:
      return 'back'
    case DoubleSide:
      return 'double'
    default:
      throw new Error(
        `material.shadowSide ${String(material.shadowSide)} is not supported by @headless-three/renderer. Use FrontSide, BackSide, DoubleSide, null, or undefined.`,
      )
  }
}

function finiteIntegerOrDefault(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? Math.trunc(value) : fallback
}

export function textureUvChannel(texture: ThreeTextureLike | null | undefined): number {
  if (!Number.isInteger(texture?.channel)) return 0
  const channel = texture!.channel!
  if (channel >= 0 && channel <= 3) return channel
  throw new Error(
    `texture.channel ${channel} is not supported by @headless-three/renderer yet. Use channel 0, 1, 2, or 3 for Three.js UV attributes.`,
  )
}

function firstOptionalFiniteNumber(entries: Array<[unknown, string]>): number | undefined {
  for (const [value, label] of entries) {
    if (value != null) return optionalFiniteNumber(value, label)
  }
  return undefined
}

function optionalFiniteNumber(value: unknown, label: string): number | undefined {
  if (value == null) return undefined
  if (typeof value === 'number' && Number.isFinite(value)) return value
  throw new TypeError(`${label} must be a finite number.`)
}

function optionalBoolean(value: unknown, label: string): boolean | undefined {
  if (value == null) return undefined
  if (typeof value === 'boolean') return value
  throw new TypeError(`${label} must be a boolean.`)
}

function optionalFiniteNumberOrInfinityDefault(value: unknown, label: string): number | undefined {
  if (value === Number.POSITIVE_INFINITY) return undefined
  return optionalFiniteNumber(value, label)
}

function requiredFiniteNumber(value: unknown, label: string): number {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  throw new TypeError(`${label} must be a finite number.`)
}

function finiteNumberOrDefault(value: unknown, label: string, fallback: number): number {
  return optionalFiniteNumber(value, label) ?? fallback
}

function vector3LikeToArray(value: unknown): number[] | undefined {
  if (!value || typeof value !== 'object') return undefined

  const arrayLike = value as ArrayLike<unknown>
  if (typeof arrayLike.length === 'number' && arrayLike.length >= 3) {
    const x = arrayLike[0]
    const y = arrayLike[1]
    const z = arrayLike[2]
    if (typeof x === 'number' && typeof y === 'number' && typeof z === 'number'
      && Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
      return [x, y, z]
    }
  }

  const vector = value as { x?: unknown; y?: unknown; z?: unknown }
  const { x, y, z } = vector
  if (typeof x === 'number' && typeof y === 'number' && typeof z === 'number'
    && Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
    return [x, y, z]
  }

  return undefined
}

function hasNonZeroVector3Like(value: unknown): boolean {
  const components = vector3LikeToArray(value)
  return components ? components.some((component) => Math.abs(component) > 1e-12) : false
}

function sameVector3Like(left: unknown, right: unknown): boolean {
  const leftComponents = vector3LikeToArray(left)
  const rightComponents = vector3LikeToArray(right)
  if (!leftComponents || !rightComponents) return false
  return leftComponents.every((component, index) => Math.abs(component - rightComponents[index]) <= 1e-12)
    && eulerLikeOrder(left) === eulerLikeOrder(right)
}

function eulerLikeOrder(value: unknown): string {
  if (!value || typeof value !== 'object') return 'XYZ'
  const arrayLike = value as ArrayLike<unknown>
  if (typeof arrayLike.length === 'number' && arrayLike.length >= 4 && typeof arrayLike[3] === 'string') {
    return arrayLike[3]
  }
  const order = (value as { order?: unknown }).order
  return typeof order === 'string' ? order : 'XYZ'
}

function extractCustomFragmentShader(material: ThreeMaterialLike | undefined): string | undefined {
  if (!material) return undefined

  const userData = material.userData ?? {}
  const hints = userData.headlessThreeRenderer ?? userData.headlessRenderer ?? {}
  const candidates = [
    material.customFragmentWgsl,
    material.customFragmentShader,
    material.headlessFragmentWgsl,
    material.headlessFragmentShader,
    hints.fragmentWgsl,
    hints.fragmentShader,
    hints.customFragmentWgsl,
    hints.customFragmentShader,
  ]

  for (const candidate of candidates) {
    if (typeof candidate === 'string' && candidate.trim().length > 0) {
      return candidate.trim()
    }
  }

  return undefined
}

function assertSupportedShaderMaterial(
  material: ThreeMaterialLike,
  customFragmentShader: string | undefined,
): void {
  const kind = shaderMaterialKind(material)
  if (!kind || customFragmentShader) return

  throw new Error(
    `${kind} is not supported directly by @headless-three/renderer. Use a built-in Three.js material, or provide material.userData.headlessThreeRenderer.fragmentWgsl with a WGSL fragment body for the renderer's custom material path.`,
  )
}

function assertSupportedOnBeforeCompile(
  material: ThreeMaterialLike,
  customFragmentShader: string | undefined,
): void {
  if (customFragmentShader || !hasCustomOnBeforeCompile(material)) return

  throw new Error(
    'material.onBeforeCompile customizations are not translated by @headless-three/renderer yet. Provide material.userData.headlessThreeRenderer.fragmentWgsl with a WGSL fragment body for the renderer custom material path.',
  )
}

function assertSupportedMaterialState(
  material: ThreeMaterialLike,
  context: MaterialExtractionContext,
): void {
  optionalBoolean(material.wireframe, 'material.wireframe')
  if (
    material.envMap != null &&
    supportsNativeMaterialEnvironmentMap(material) &&
    context.materialEnvironmentMaps?.has(material) !== true
  ) {
    throw new Error(
      'material.envMap on MeshBasicMaterial, MeshStandardMaterial, MeshPhysicalMaterial, MeshPhongMaterial, and MeshLambertMaterial requires one shared material envMap represented by the native IBL path. Use scene.environment, remove material.envMap when a scene environment or reflection probe is active, or render separate passes.',
    )
  }
}

function assertSupportedMaterialClass(
  material: ThreeMaterialLike,
  customFragmentShader: string | undefined,
): void {
  if (customFragmentShader || supportedMaterialClass(material)) return

  const type = typeof material.type === 'string' && material.type.trim()
    ? material.type
    : 'Material'
  throw new Error(
    `${type} is not supported directly by @headless-three/renderer. Use a supported built-in Three.js material, or provide material.userData.headlessThreeRenderer.fragmentWgsl with a WGSL fragment body for the renderer's custom material path.`,
  )
}

function supportedMaterialClass(material: ThreeMaterialLike): boolean {
  return material.isMeshBasicMaterial === true
    || material.isMeshDepthMaterial === true
    || material.isMeshDistanceMaterial === true
    || material.isMeshLambertMaterial === true
    || material.isMeshMatcapMaterial === true
    || material.isMeshNormalMaterial === true
    || material.isMeshPhongMaterial === true
    || material.isMeshStandardMaterial === true
    || material.isMeshPhysicalMaterial === true
    || material.isMeshToonMaterial === true
    || material.isShadowMaterial === true
    || material.isLineBasicMaterial === true
    || material.isLineDashedMaterial === true
    || material.isPointsMaterial === true
    || material.isSpriteMaterial === true
}

function hasCustomOnBeforeCompile(material: ThreeMaterialLike): boolean {
  if (typeof material.onBeforeCompile !== 'function') return false
  return normalizeFunctionSource(material.onBeforeCompile) !== DefaultOnBeforeCompileSource
}

function normalizeFunctionSource(fn: (...args: any[]) => unknown): string {
  return Function.prototype.toString.call(fn).replace(/\s+/g, ' ').trim()
}

function shaderMaterialKind(material: ThreeMaterialLike): string | undefined {
  if (material.isRawShaderMaterial === true || material.type === 'RawShaderMaterial') {
    return 'RawShaderMaterial'
  }
  if (
    material.isNodeMaterial === true ||
    (typeof material.type === 'string' && material.type.includes('NodeMaterial'))
  ) {
    return 'NodeMaterial'
  }
  if (material.isShaderMaterial === true || material.type === 'ShaderMaterial') {
    return 'ShaderMaterial'
  }
  return undefined
}

export function extractTextureData(material: ThreeMaterialLike | undefined): TextureInfo | null {
  const slot = material?.isMeshMatcapMaterial ? material.matcap : material?.map
  const label = material?.isMeshMatcapMaterial ? 'material.matcap' : 'material.map'
  const base = extractTextureFromSlot(slot, label)
  if (!base) return null

  const map = slot as ThreeTextureLike | null | undefined
  return {
    ...base,
    wrapS: material?.isMeshMatcapMaterial ? undefined : wrapModeToString(map?.wrapS),
    wrapT: material?.isMeshMatcapMaterial ? undefined : wrapModeToString(map?.wrapT),
    magFilter: filterModeToString(map?.magFilter),
    minFilter: minFilterModeToString(map),
    anisotropy: textureAnisotropy(map, label),
    transform: material?.isMeshMatcapMaterial ? undefined : textureTransform(map, label),
    colorSpace: textureColorSpace(map),
    usesUv2: material?.isMeshMatcapMaterial ? false : textureUvChannel(map) > 0,
  }
}

export function extractBackgroundTexture(
  background: ThreeSceneRootLike['background'] | ThreeTextureLike | number[] | undefined,
  label = 'background',
): TextureInfo | null {
  const map = textureLike(background)
  if (!map) return null
  if (isCubeBackgroundTexture(map)) {
    assertSupportedTextureInput(map, label)
    return extractCubeBackgroundTexture(map, label)
  }
  assertSupportedBackgroundTexture(map, label)

  const base = extractTextureFromSlot(map, label)
  if (!base) return null

  return {
    ...base,
    wrapS: wrapModeToString(map?.wrapS),
    wrapT: wrapModeToString(map?.wrapT),
    magFilter: filterModeToString(map?.magFilter),
    minFilter: minFilterModeToString(map),
    anisotropy: textureAnisotropy(map, label),
    transform: textureTransform(map, label),
    colorSpace: textureColorSpace(map),
    mapping: backgroundTextureMapping(map),
  }
}

function backgroundTextureMapping(map: ThreeTextureLike): 'uv' | 'equirectangular' {
  return map.mapping === EquirectangularReflectionMapping || map.mapping === EquirectangularRefractionMapping
    ? 'equirectangular'
    : 'uv'
}

function isCubeBackgroundTexture(map: ThreeTextureLike): boolean {
  return map.isCubeTexture === true ||
    map.mapping === CubeReflectionMapping ||
    map.mapping === CubeRefractionMapping ||
    map.mapping === CubeUVReflectionMapping
}

function isCubeEnvironmentTexture(map: ThreeTextureLike): boolean {
  return map.isCubeTexture === true || map.mapping === CubeReflectionMapping || map.mapping === CubeRefractionMapping
}

function extractCubeBackgroundTexture(map: ThreeTextureLike, label: string): TextureInfo {
  if (map.mapping === CubeUVReflectionMapping) {
    throw new Error(
      `${label} uses PMREM/CubeUV texture mapping, which is not supported as a background yet. Use a six-face CubeTexture, a 2D/equirectangular texture, or pre-render the background to a 2D image before rendering.`,
    )
  }

  const cube = cubeTextureToEquirectangular(map, label)
  return {
    ...cube,
    wrapS: 'repeat',
    wrapT: 'clamp',
    magFilter: filterModeToString(map.magFilter),
    minFilter: minFilterModeToString(map),
    anisotropy: textureAnisotropy(map, label),
    colorSpace: textureColorSpace(map),
    mapping: 'equirectangular',
  }
}

function cubeTextureToEquirectangular(map: ThreeTextureLike, label: string): { data: Buffer; width: number; height: number } {
  const faces = cubeFaceImages(map)
  if (!faces) {
    throw new Error(
      `${label} uses a cube texture without six raw or encoded face images. Provide a CubeTexture with six DataTexture-style or encoded PNG/JPEG/WebP face images, use a 2D/equirectangular texture, or pre-render the background to a 2D image before rendering.`,
    )
  }

  const faceTextures = faces.map((face, index) => imageToRgbaTexture(face, `${label}.image[${index}]`, map.type))
  const faceWidth = faceTextures[0].width
  const faceHeight = faceTextures[0].height
  if (faceWidth !== faceHeight) {
    throw new Error(`${label} cube background faces must be square raw RGBA images.`)
  }
  for (let i = 1; i < faceTextures.length; i += 1) {
    if (faceTextures[i].width !== faceWidth || faceTextures[i].height !== faceHeight) {
      throw new Error(`${label} cube background faces must all use the same dimensions.`)
    }
  }

  const width = Math.max(64, faceWidth * 4)
  const height = Math.max(32, faceHeight * 2)
  const out = new Uint8Array(width * height * 4)
  for (let y = 0; y < height; y += 1) {
    const v = (y + 0.5) / height
    const pitch = (v - 0.5) * Math.PI
    const dirY = Math.sin(pitch)
    const ring = Math.cos(pitch)
    for (let x = 0; x < width; x += 1) {
      const u = (x + 0.5) / width
      const yaw = (u - 0.5) * Math.PI * 2
      const dir = [
        Math.cos(yaw) * ring,
        dirY,
        Math.sin(yaw) * ring,
      ] as const
      const sample = sampleCubeFace(faceTextures, dir)
      out.set(sample, (y * width + x) * 4)
    }
  }

  return {
    data: Buffer.from(out.buffer, out.byteOffset, out.byteLength),
    width,
    height,
  }
}

function cubeFaceImages(map: ThreeTextureLike): TextureImageInput[] | null {
  const image = (map as any).image ?? (map as any).source?.data
  if (Array.isArray(image) && image.length >= 6) return image.slice(0, 6) as TextureImageInput[]
  return null
}

function imageToRgbaTexture(image: TextureImageInput, label: string, textureType?: number): { rgba: Uint8Array; width: number; height: number } {
  if (Buffer.isBuffer(image) || image instanceof Uint8Array) {
    const buffer = Buffer.isBuffer(image)
      ? image
      : Buffer.from(image.buffer, image.byteOffset, image.byteLength)
    const decoded = native.decodeImage?.(buffer)
    if (!decoded?.data || !(decoded.width! > 0) || !(decoded.height! > 0)) {
      throw new Error(`${label} encoded cube face image could not be decoded to RGBA pixels.`)
    }
    const rgba = decoded.data instanceof Uint8Array
      ? decoded.data
      : new Uint8Array(decoded.data)
    if (rgba.byteLength !== decoded.width! * decoded.height! * 4) {
      throw new Error(`${label} encoded cube face image decoded to an unexpected RGBA byte length.`)
    }
    return { rgba, width: decoded.width!, height: decoded.height! }
  }
  if (!image || !image.data || !(image.width! > 0) || !(image.height! > 0)) {
    throw new Error(`${label} must provide raw face data, width, and height for cube background rendering.`)
  }
  const rgba = toRgba8(image.data, image.width!, image.height!, { type: textureType })
  if (!rgba) {
    throw new Error(`${label} must contain RGB or RGBA numeric pixel data for cube background rendering.`)
  }
  return { rgba, width: image.width!, height: image.height! }
}

function sampleCubeFace(
  faces: Array<{ rgba: Uint8Array; width: number; height: number }>,
  dir: readonly [number, number, number],
): Uint8Array {
  const [x, y, z] = dir
  const ax = Math.abs(x)
  const ay = Math.abs(y)
  const az = Math.abs(z)
  let faceIndex = 0
  let sc = 0
  let tc = 0

  if (ax >= ay && ax >= az) {
    if (x >= 0) {
      faceIndex = 0
      sc = -z / ax
    } else {
      faceIndex = 1
      sc = z / ax
    }
    tc = -y / ax
  } else if (ay >= ax && ay >= az) {
    if (y >= 0) {
      faceIndex = 2
      sc = x / ay
      tc = z / ay
    } else {
      faceIndex = 3
      sc = x / ay
      tc = -z / ay
    }
  } else {
    if (z >= 0) {
      faceIndex = 4
      sc = x / az
    } else {
      faceIndex = 5
      sc = -x / az
    }
    tc = -y / az
  }

  const face = faces[faceIndex]
  const u = Math.max(0, Math.min(1, (sc + 1) * 0.5))
  const v = Math.max(0, Math.min(1, (tc + 1) * 0.5))
  const px = Math.min(face.width - 1, Math.floor(u * face.width))
  const py = Math.min(face.height - 1, Math.floor(v * face.height))
  const offset = (py * face.width + px) * 4
  return face.rgba.subarray(offset, offset + 4)
}

function textureLike(value: unknown): ThreeTextureLike | null {
  if (!value || Array.isArray(value)) return null
  const candidate = value as ThreeTextureLike & { isTexture?: boolean }
  if (candidate.isTexture === true || candidate.image || candidate.source?.data) {
    return candidate
  }
  return null
}

function requiredEnvironmentTexture(value: unknown, label: string): ThreeTextureLike {
  const texture = textureLike(value)
  if (texture) return texture
  throw new TypeError(
    `${label} must be a Three.js texture or null for environment map rendering.`,
  )
}

function wrapModeToString(mode: unknown): string | undefined {
  if (mode == null || mode === ClampToEdgeWrapping) return undefined // default = clamp
  if (mode === RepeatWrapping) return 'repeat'
  if (mode === MirroredRepeatWrapping) return 'mirror'
  throw new Error(
    `texture wrap mode ${String(mode)} is not supported by @headless-three/renderer. Use ClampToEdgeWrapping, RepeatWrapping, or MirroredRepeatWrapping.`,
  )
}

function filterModeToString(mode: unknown): string | undefined {
  if (mode == null) return undefined // default = linear
  if (
    mode === NearestFilter ||
    mode === NearestMipmapNearestFilter ||
    mode === NearestMipmapLinearFilter
  ) {
    return 'nearest'
  }
  if (
    mode === LinearFilter ||
    mode === LinearMipmapNearestFilter ||
    mode === LinearMipmapLinearFilter
  ) {
    return 'linear'
  }
  throw new Error(
    `texture.magFilter ${String(mode)} is not supported by @headless-three/renderer. Use NearestFilter or LinearFilter.`,
  )
}

function minFilterModeToString(texture: ThreeTextureLike | null | undefined): string | undefined {
  const mode = texture?.minFilter
  if (mode == null) return undefined
  const generateMipmaps = optionalTextureBoolean(texture?.generateMipmaps, 'texture.generateMipmaps')
  const allowMipmaps = generateMipmaps !== false || hasExplicitMipmaps(texture)
  if (mode === NearestFilter) return 'nearest'
  if (mode === LinearFilter) return 'linear'
  if (mode === NearestMipmapNearestFilter) return allowMipmaps ? 'nearest-mipmap-nearest' : 'nearest'
  if (mode === NearestMipmapLinearFilter) return allowMipmaps ? 'nearest-mipmap-linear' : 'nearest'
  if (mode === LinearMipmapNearestFilter) return allowMipmaps ? 'linear-mipmap-nearest' : 'linear'
  if (mode === LinearMipmapLinearFilter) return allowMipmaps ? 'linear-mipmap-linear' : 'linear'
  throw new Error(
    `texture.minFilter ${String(mode)} is not supported by @headless-three/renderer. Use NearestFilter, LinearFilter, or a Three.js mipmap minFilter constant.`,
  )
}

function extractTextureFromSlot(map: ThreeMaterialLike['map'], label = 'texture'): TextureInfo | null {
  if (!map) return null
  assertSupportedTextureInput(map, label, { allowMipmaps: true })
  assertSupportedTwoDimensionalTextureSlot(map, label)

  const image = (map as any).image ?? (map as any).source?.data
  if (!image) return null

  // DataTexture style: { data: TypedArray, width, height }
  if (image.data && image.width > 0 && image.height > 0) {
    assertSupportedRawTextureType((map as any).type, label, 'texture rendering')
    const rgba = toRgba8(image.data, image.width, image.height, { type: map.type })
    if (rgba) {
      const data = textureBytesWithExplicitMipmaps(map, label, rgba, image.width, image.height)
      return { data: Buffer.from(data.buffer, data.byteOffset, data.byteLength), width: image.width, height: image.height }
    }
    throw unsupportedRawTextureDataError(label, 'texture rendering')
  }

  // Encoded image (PNG/JPEG/WebP Buffer from file loaders)
  if (Buffer.isBuffer(image)) {
    assertNoEncodedExplicitMipmaps(map, label)
    return { data: image, width: 0, height: 0 }
  }
  if (image instanceof Uint8Array && !((image as any).width > 0)) {
    assertNoEncodedExplicitMipmaps(map, label)
    return { data: Buffer.from(image.buffer, image.byteOffset, image.byteLength), width: 0, height: 0 }
  }

  // ImageData (canvas-based polyfill): { data: Uint8ClampedArray, width, height }
  if (image.data instanceof Uint8ClampedArray && image.width > 0 && image.height > 0) {
    const data = textureBytesWithExplicitMipmaps(map, label, image.data, image.width, image.height)
    return {
      data: Buffer.from(data.buffer, data.byteOffset, data.byteLength),
      width: image.width,
      height: image.height,
    }
  }

  throw unsupportedTextureImageError(label, 'texture rendering')
}

function hasExplicitMipmaps(texture: ThreeTextureLike | null | undefined): boolean {
  return Array.isArray(texture?.mipmaps) && texture.mipmaps.length > 0
}

function assertNoEncodedExplicitMipmaps(map: ThreeTextureLike, label: string): void {
  if (!hasExplicitMipmaps(map)) return
  throw new Error(
    `${label} provides explicit texture mipmaps with an encoded base image. Explicit mipmap upload requires raw DataTexture-style base image data with raw mipmap levels.`,
  )
}

function textureBytesWithExplicitMipmaps(
  map: ThreeTextureLike,
  label: string,
  baseRgba: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
): Uint8Array | Uint8ClampedArray {
  if (!hasExplicitMipmaps(map)) return baseRgba
  if (width <= 1 && height <= 1) {
    throw new Error(
      `${label} provides explicit texture mipmaps for a ${width}x${height} base image, but no additional mip levels are valid after the 1x1 level.`,
    )
  }

  const levels: Uint8Array[] = [
    baseRgba instanceof Uint8Array
      ? new Uint8Array(baseRgba.buffer, baseRgba.byteOffset, baseRgba.byteLength)
      : new Uint8Array(baseRgba),
  ]
  let expectedWidth = width
  let expectedHeight = height
  const mipmaps = map.mipmaps!

  for (let i = 0; i < mipmaps.length; i += 1) {
    expectedWidth = Math.max(1, Math.floor(expectedWidth / 2))
    expectedHeight = Math.max(1, Math.floor(expectedHeight / 2))

    const mip = mipmaps[i]
    if (!mip || !mip.data || mip.width !== expectedWidth || mip.height !== expectedHeight) {
      throw new Error(
        `${label}.mipmaps[${i}] must provide raw pixel data with size ${expectedWidth}x${expectedHeight} for explicit mipmap upload.`,
      )
    }
    const rgba = toRgba8(mip.data, expectedWidth, expectedHeight, { type: map.type })
    if (!rgba) {
      throw unsupportedRawTextureDataError(`${label}.mipmaps[${i}]`, 'texture rendering')
    }
    levels.push(rgba)

    if (expectedWidth === 1 && expectedHeight === 1 && i < mipmaps.length - 1) {
      throw new Error(
        `${label} provides extra explicit mipmap levels after the 1x1 level.`,
      )
    }
  }

  if (expectedWidth !== 1 || expectedHeight !== 1) {
    throw new Error(
      `${label} explicit texture mipmaps must include the complete mip chain down to 1x1.`,
    )
  }

  const byteLength = levels.reduce((total, level) => total + level.byteLength, 0)
  const out = new Uint8Array(byteLength)
  let offset = 0
  for (const level of levels) {
    out.set(level, offset)
    offset += level.byteLength
  }
  return out
}

function unsupportedRawTextureDataError(label: string, usage: string): Error {
  const supported = usage === 'texture rendering'
    ? 'one-channel, two-channel, RGB, or RGBA numeric pixel data'
    : 'RGB or RGBA numeric pixel data'
  const unsupported = usage === 'texture rendering'
    ? 'mismatched data lengths are not supported yet'
    : 'one-channel, two-channel, and mismatched data lengths are not supported yet'
  return new Error(
    `${label} raw texture data must contain ${supported} for ${usage}; ${unsupported}.`,
  )
}

function unsupportedTextureImageError(label: string, usage: string): Error {
  return new Error(
    `${label} uses a texture image object that is not readable by @headless-three/renderer for ${usage}. Provide encoded PNG/JPEG/WebP bytes directly as texture.image or texture.source.data, or raw one-channel, two-channel, RGB, or RGBA numeric pixel data as { data, width, height } before rendering.`,
  )
}

function assertSupportedRawTextureType(type: unknown, label: string, usage: string): void {
  if (
    type == null ||
    type === UnsignedByteType ||
    type === ByteType ||
    type === ShortType ||
    type === UnsignedShortType ||
    type === IntType ||
    type === UnsignedIntType ||
    type === HalfFloatType ||
    type === FloatType ||
    type === UnsignedShort4444Type ||
    type === UnsignedShort5551Type
  ) {
    return
  }
  throw new Error(
    `${label} raw texture type ${textureTypeName(type)} is not supported by @headless-three/renderer for ${usage}. Use UnsignedByteType, ByteType, ShortType, UnsignedShortType, IntType, UnsignedIntType, HalfFloatType, FloatType, UnsignedShort4444Type, or UnsignedShort5551Type raw data, or pre-convert the texture to RGBA8 before rendering.`,
  )
}

function textureTypeName(type: unknown): string {
  switch (type) {
    case ByteType:
      return 'ByteType'
    case ShortType:
      return 'ShortType'
    case IntType:
      return 'IntType'
    case UnsignedInt248Type:
      return 'UnsignedInt248Type'
    case UnsignedInt5999Type:
      return 'UnsignedInt5999Type'
    default:
      return String(type)
  }
}

function rawTextureChannelCount(
  data: ArrayLike<number>,
  width: number,
  height: number,
  label: string,
  usage: string,
): 3 | 4 {
  const pixels = width * height
  const length = typeof data.length === 'number' ? data.length : Number.NaN
  const channels = length / pixels
  if (channels === 3 || channels === 4) return channels
  throw unsupportedRawTextureDataError(label, usage)
}

function assertSupportedBackgroundTexture(map: ThreeTextureLike, label: string): void {
  assertSupportedTextureInput(map, label, { allowMipmaps: true })
  if (
    map.isCubeTexture === true ||
    map.mapping === CubeReflectionMapping ||
    map.mapping === CubeRefractionMapping ||
    map.mapping === CubeUVReflectionMapping
  ) {
    throw new Error(
      `${label} uses a cube or PMREM/CubeUV texture mapping, which is not supported as a background yet. Use a 2D/equirectangular texture or pre-render the background to a 2D image before rendering.`,
    )
  }
}

function assertSupportedTwoDimensionalTextureSlot(map: ThreeTextureLike, label: string): void {
  if (
    map.isCubeTexture === true ||
    map.mapping === CubeReflectionMapping ||
    map.mapping === CubeRefractionMapping ||
    map.mapping === CubeUVReflectionMapping
  ) {
    throw new Error(
      `${label} uses a cube or PMREM/CubeUV texture mapping, which is not supported for 2D material texture slots. Use a 2D texture for material maps or move cube textures to scene.environment, scene.background, or material.envMap where supported.`,
    )
  }
}

function isRefractionEnvironmentMapping(mapping: number | undefined): boolean {
  return mapping === CubeRefractionMapping || mapping === EquirectangularRefractionMapping
}

function assertSupportedEnvironmentTexture(
  map: ThreeTextureLike,
  label: string,
  options: { allowRefraction?: boolean } = {},
): void {
  assertSupportedTextureInput(map, label)
  const usesRefraction = isRefractionEnvironmentMapping(map.mapping)
  if ((usesRefraction && options.allowRefraction !== true) || map.mapping === CubeUVReflectionMapping) {
    throw new Error(
      `${label} uses refraction or PMREM/CubeUV environment mapping, which is not supported by @headless-three/renderer yet. Provide an equirectangular or six-face cube reflection texture and let the renderer precompute IBL, or pre-convert the source before rendering.`,
    )
  }
}

function assertSupportedTextureInput(
  map: ThreeTextureLike,
  label: string,
  options: { allowMipmaps?: boolean } = {},
): void {
  if (
    map.isCompressedTexture === true ||
    map.isCompressedArrayTexture === true ||
    map.isCompressedCubeTexture === true
  ) {
    throw new Error(
      `${label} uses a compressed texture. KTX2, Basis, and THREE.CompressedTexture inputs are not decoded by @headless-three/renderer yet; pre-decode the texture to RGBA data or an encoded PNG/JPEG/WebP image before rendering.`,
    )
  }
  if (
    (map as any).isDataArrayTexture === true ||
    (map as any).isData3DTexture === true ||
    (map as any).isArrayTexture === true ||
    (map as any).is3DTexture === true
  ) {
    throw new Error(
      `${label} uses an array or 3D texture, which is not supported by @headless-three/renderer yet. Provide a 2D texture image for this slot or render each layer separately.`,
    )
  }
  if (!options.allowMipmaps && hasExplicitMipmaps(map)) {
    throw new Error(
      `${label} provides explicit texture mipmaps, which are not uploaded by @headless-three/renderer yet. Provide only the base image level or prefilter/downsample the texture before rendering.`,
    )
  }
}

function assertCompatiblePackedPhysicalMapSamplers(material: ThreeMaterialLike): void {
  assertNoPackedPhysicalMapMipmaps('physical extension scalar maps', [
    ['clearcoatMap', material.clearcoatMap],
    ['clearcoatRoughnessMap', material.clearcoatRoughnessMap],
    ['transmissionMap', material.transmissionMap],
    ['thicknessMap', material.thicknessMap],
    ['anisotropyMap', material.anisotropyMap],
    ['iridescenceMap', material.iridescenceMap],
    ['iridescenceThicknessMap', material.iridescenceThicknessMap],
  ])
  assertNoPackedPhysicalMapMipmaps('physical extension sheen maps', [
    ['sheenColorMap', material.sheenColorMap],
    ['sheenRoughnessMap', material.sheenRoughnessMap],
  ])
  assertNoPackedPhysicalMapMipmaps('physical extension specular maps', [
    ['specularColorMap', material.specularColorMap],
    ['specularIntensityMap', material.specularIntensityMap],
  ])
  assertMatchingSamplerSettings('physical extension scalar maps', [
    ['clearcoatMap', material.clearcoatMap],
    ['clearcoatRoughnessMap', material.clearcoatRoughnessMap],
    ['transmissionMap', material.transmissionMap],
    ['thicknessMap', material.thicknessMap],
    ['anisotropyMap', material.anisotropyMap],
    ['iridescenceMap', material.iridescenceMap],
    ['iridescenceThicknessMap', material.iridescenceThicknessMap],
  ])
  assertMatchingSamplerSettings('physical extension sheen maps', [
    ['sheenColorMap', material.sheenColorMap],
    ['sheenRoughnessMap', material.sheenRoughnessMap],
  ])
  assertMatchingSamplerSettings('physical extension specular maps', [
    ['specularColorMap', material.specularColorMap],
    ['specularIntensityMap', material.specularIntensityMap],
  ])
}

function assertNoPackedPhysicalMapMipmaps(groupLabel: string, slots: Array<[string, ThreeTextureLike | null | undefined]>): void {
  for (const [label, texture] of slots) {
    if (!texture || !hasExplicitMipmaps(texture)) continue
    throw new Error(
      `${groupLabel} are packed into one native texture, and explicit mipmaps for ${label} are not supported by @headless-three/renderer yet. Remove texture.mipmaps from packed physical-extension maps or rely on generated mipmaps from the packed base level.`,
    )
  }
}

function assertMatchingSamplerSettings(groupLabel: string, slots: Array<[string, ThreeTextureLike | null | undefined]>): void {
  let first: { label: string; signature: string } | null = null
  for (const [label, texture] of slots) {
    if (!texture) continue
    const signature = samplerSignature(texture, `material.${label}`)
    if (!first) {
      first = { label, signature }
      continue
    }
    if (signature !== first.signature) {
      throw new Error(
        `${groupLabel} are packed into one native texture and must use matching wrap/filter/anisotropy sampler settings. ${label} differs from ${first.label}; use matching wrapS/wrapT/magFilter/minFilter/anisotropy values or render separate passes until independent packed-channel samplers are supported.`,
      )
    }
  }
}

function samplerSignature(texture: ThreeTextureLike, label: string): string {
  return [
    wrapModeToString(texture.wrapS) ?? 'clamp',
    wrapModeToString(texture.wrapT) ?? 'clamp',
    filterModeToString(texture.magFilter) ?? 'linear',
    minFilterModeToString(texture) ?? 'linear',
    String(textureAnisotropy(texture, label) ?? 1),
  ].join('|')
}

function textureAnisotropy(map: ThreeTextureLike | null | undefined, label: string): number | undefined {
  const value = map?.anisotropy
  if (value == null) return undefined
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${label}.anisotropy must be a finite number.`)
  }
  if (value <= 1) return undefined
  return Math.max(1, Math.min(16, Math.floor(value)))
}

function textureTransform(map: ThreeTextureLike | null | undefined, label: string): number[] | undefined {
  const flipY = optionalTextureBoolean(map?.flipY, `${label}.flipY`) !== false
  const flipTransform = flipY ? undefined : [1, 0, 0, 0, -1, 1]
  if (!map) return flipTransform

  const matrixAutoUpdate = optionalTextureBoolean(map.matrixAutoUpdate, `${label}.matrixAutoUpdate`)
  if (matrixAutoUpdate === false) {
    const e = map.matrix?.elements
    if (!e || e.length !== 9) {
      throw new TypeError(`${label}.matrix.elements must contain nine finite numbers.`)
    }
    validateFiniteMatrix3(e, `${label}.matrix.elements`)
    return composeTextureTransformWithFlipY([e[0], e[3], e[6], e[1], e[4], e[7]], flipY)
  }

  const tx = finiteTextureTransformNumber(map.offset?.x, `${label}.offset.x`, 0)
  const ty = finiteTextureTransformNumber(map.offset?.y, `${label}.offset.y`, 0)
  const sx = finiteTextureTransformNumber(map.repeat?.x, `${label}.repeat.x`, 1)
  const sy = finiteTextureTransformNumber(map.repeat?.y, `${label}.repeat.y`, 1)
  const rotation = finiteTextureTransformNumber(map.rotation, `${label}.rotation`, 0)
  const cx = finiteTextureTransformNumber(map.center?.x, `${label}.center.x`, 0)
  const cy = finiteTextureTransformNumber(map.center?.y, `${label}.center.y`, 0)
  if (tx === 0 && ty === 0 && sx === 1 && sy === 1 && rotation === 0 && cx === 0 && cy === 0) {
    return flipTransform
  }

  const c = Math.cos(rotation)
  const s = Math.sin(rotation)
  return composeTextureTransformWithFlipY([
    sx * c,
    sx * s,
    -sx * (c * cx + s * cy) + cx + tx,
    -sy * s,
    sy * c,
    -sy * (-s * cx + c * cy) + cy + ty,
  ], flipY)
}

function composeTextureTransformWithFlipY(transform: number[], flipY: boolean): number[] {
  if (flipY) return transform
  const [a, c, tx, b, d, ty] = transform
  return [a, -c, c + tx, b, -d, d + ty]
}

function optionalTextureBoolean(value: unknown, label: string): boolean | undefined {
  if (value == null) return undefined
  if (typeof value === 'boolean') return value
  throw new TypeError(`${label} must be a boolean.`)
}

function textureColorSpace(map: ThreeTextureLike | null | undefined): string | undefined {
  if (!map) return undefined
  if (map.colorSpace === 'srgb') return 'srgb'
  if (
    map.colorSpace === 'srgb-linear' ||
    map.colorSpace === 'linear-srgb' ||
    map.colorSpace === 'linearsrgb' ||
    map.colorSpace === 'linear'
  ) {
    return 'linear'
  }
  if (map.colorSpace != null && map.colorSpace !== '') {
    throw new Error(
      `texture.colorSpace ${String(map.colorSpace)} is not supported by @headless-three/renderer. Use THREE.SRGBColorSpace, THREE.LinearSRGBColorSpace, or THREE.NoColorSpace.`,
    )
  }
  if (map.encoding === sRGBEncoding) return 'srgb'
  if (map.encoding != null && map.encoding !== LinearEncoding) {
    throw new Error(
      `texture.encoding ${String(map.encoding)} is not supported by @headless-three/renderer. Use sRGBEncoding, LinearEncoding, or texture.colorSpace with THREE.SRGBColorSpace/THREE.LinearSRGBColorSpace.`,
    )
  }
  return undefined
}

function finiteOrDefault(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function finiteTextureTransformNumber(value: unknown, label: string, fallback: number): number {
  if (value == null) return fallback
  if (typeof value === 'number' && Number.isFinite(value)) return value
  throw new TypeError(`${label} must be a finite number.`)
}

function validateFiniteMatrix3(values: ArrayLike<unknown>, label: string): void {
  for (let i = 0; i < 9; i += 1) {
    if (typeof values[i] !== 'number' || !Number.isFinite(values[i])) {
      throw new TypeError(`${label}[${i}] must be a finite number.`)
    }
  }
}

function toRgba8(
  data: ArrayLike<number>,
  width: number,
  height: number,
  options: { narrowChannels?: boolean; type?: number } = {},
): Uint8Array | null {
  const pixels = width * height
  const allowNarrowChannels = options.narrowChannels !== false
  const textureType = options.type ?? UnsignedByteType

  if (textureType === HalfFloatType) {
    if (!(data instanceof Uint16Array)) return null
    return halfFloatDataToRgba8(data, pixels, allowNarrowChannels)
  }
  if (textureType === ByteType) {
    if (!(data instanceof Int8Array)) return null
    return normalizedSignedIntegerDataToRgba8(data, pixels, allowNarrowChannels, 0x7f)
  }
  if (textureType === ShortType) {
    if (!(data instanceof Int16Array)) return null
    return normalizedSignedIntegerDataToRgba8(data, pixels, allowNarrowChannels, 0x7fff)
  }
  if (textureType === UnsignedShortType) {
    if (!(data instanceof Uint16Array)) return null
    return normalizedUnsignedIntegerDataToRgba8(data, pixels, allowNarrowChannels, 0xffff)
  }
  if (textureType === IntType) {
    if (!(data instanceof Int32Array)) return null
    return normalizedSignedIntegerDataToRgba8(data, pixels, allowNarrowChannels, 0x7fffffff)
  }
  if (textureType === UnsignedIntType) {
    if (!(data instanceof Uint32Array)) return null
    return normalizedUnsignedIntegerDataToRgba8(data, pixels, allowNarrowChannels, 0xffffffff)
  }
  if (textureType === UnsignedShort4444Type) {
    if (!(data instanceof Uint16Array)) return null
    return packedUnsignedShort4444ToRgba8(data, pixels)
  }
  if (textureType === UnsignedShort5551Type) {
    if (!(data instanceof Uint16Array)) return null
    return packedUnsignedShort5551ToRgba8(data, pixels)
  }

  if (data instanceof Uint8Array || data instanceof Uint8ClampedArray) {
    if (data.length === pixels * 4) return new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
    if (data.length === pixels * 3) {
      const out = new Uint8Array(pixels * 4)
      for (let i = 0; i < pixels; i++) {
        out[i * 4] = data[i * 3]
        out[i * 4 + 1] = data[i * 3 + 1]
        out[i * 4 + 2] = data[i * 3 + 2]
        out[i * 4 + 3] = 255
      }
      return out
    }
    if (allowNarrowChannels && data.length === pixels * 2) {
      const out = new Uint8Array(pixels * 4)
      for (let i = 0; i < pixels; i++) {
        out[i * 4] = data[i * 2]
        out[i * 4 + 1] = data[i * 2 + 1]
        out[i * 4 + 2] = 0
        out[i * 4 + 3] = 255
      }
      return out
    }
    if (allowNarrowChannels && data.length === pixels) {
      const out = new Uint8Array(pixels * 4)
      for (let i = 0; i < pixels; i++) {
        out[i * 4] = data[i]
        out[i * 4 + 1] = data[i]
        out[i * 4 + 2] = data[i]
        out[i * 4 + 3] = 255
      }
      return out
    }
    return null
  }

  if (data instanceof Float32Array || data instanceof Float64Array) {
    if (data.length === pixels * 4) {
      const out = new Uint8Array(pixels * 4)
      for (let i = 0; i < pixels * 4; i++) {
        out[i] = Math.max(0, Math.min(255, Math.round(data[i] * 255)))
      }
      return out
    }
    if (data.length === pixels * 3) {
      const out = new Uint8Array(pixels * 4)
      for (let i = 0; i < pixels; i++) {
        out[i * 4] = Math.max(0, Math.min(255, Math.round(data[i * 3] * 255)))
        out[i * 4 + 1] = Math.max(0, Math.min(255, Math.round(data[i * 3 + 1] * 255)))
        out[i * 4 + 2] = Math.max(0, Math.min(255, Math.round(data[i * 3 + 2] * 255)))
        out[i * 4 + 3] = 255
      }
      return out
    }
    if (allowNarrowChannels && data.length === pixels * 2) {
      const out = new Uint8Array(pixels * 4)
      for (let i = 0; i < pixels; i++) {
        out[i * 4] = Math.max(0, Math.min(255, Math.round(data[i * 2] * 255)))
        out[i * 4 + 1] = Math.max(0, Math.min(255, Math.round(data[i * 2 + 1] * 255)))
        out[i * 4 + 2] = 0
        out[i * 4 + 3] = 255
      }
      return out
    }
    if (allowNarrowChannels && data.length === pixels) {
      const out = new Uint8Array(pixels * 4)
      for (let i = 0; i < pixels; i++) {
        const value = Math.max(0, Math.min(255, Math.round(data[i] * 255)))
        out[i * 4] = value
        out[i * 4 + 1] = value
        out[i * 4 + 2] = value
        out[i * 4 + 3] = 255
      }
      return out
    }
    return null
  }

  // Uint16Array or other numeric typed arrays — treat as 8-bit range after clamping
  if (ArrayBuffer.isView(data)) {
    if (data.length === pixels * 4) {
      const out = new Uint8Array(pixels * 4)
      for (let i = 0; i < pixels * 4; i++) {
        out[i] = Math.max(0, Math.min(255, (data as any)[i]))
      }
      return out
    }
    if (data.length === pixels * 3) {
      const out = new Uint8Array(pixels * 4)
      for (let i = 0; i < pixels; i++) {
        out[i * 4] = Math.max(0, Math.min(255, (data as any)[i * 3]))
        out[i * 4 + 1] = Math.max(0, Math.min(255, (data as any)[i * 3 + 1]))
        out[i * 4 + 2] = Math.max(0, Math.min(255, (data as any)[i * 3 + 2]))
        out[i * 4 + 3] = 255
      }
      return out
    }
    if (allowNarrowChannels && data.length === pixels * 2) {
      const out = new Uint8Array(pixels * 4)
      for (let i = 0; i < pixels; i++) {
        out[i * 4] = Math.max(0, Math.min(255, (data as any)[i * 2]))
        out[i * 4 + 1] = Math.max(0, Math.min(255, (data as any)[i * 2 + 1]))
        out[i * 4 + 2] = 0
        out[i * 4 + 3] = 255
      }
      return out
    }
    if (allowNarrowChannels && data.length === pixels) {
      const out = new Uint8Array(pixels * 4)
      for (let i = 0; i < pixels; i++) {
        const value = Math.max(0, Math.min(255, (data as any)[i]))
        out[i * 4] = value
        out[i * 4 + 1] = value
        out[i * 4 + 2] = value
        out[i * 4 + 3] = 255
      }
      return out
    }
  }

  if (data.length === pixels * 4) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels * 4; i++) {
      out[i] = Math.max(0, Math.min(255, data[i]))
    }
    return out
  }
  if (data.length === pixels * 3) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels; i++) {
      out[i * 4] = Math.max(0, Math.min(255, data[i * 3]))
      out[i * 4 + 1] = Math.max(0, Math.min(255, data[i * 3 + 1]))
      out[i * 4 + 2] = Math.max(0, Math.min(255, data[i * 3 + 2]))
      out[i * 4 + 3] = 255
    }
    return out
  }
  if (allowNarrowChannels && data.length === pixels * 2) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels; i++) {
      out[i * 4] = Math.max(0, Math.min(255, data[i * 2]))
      out[i * 4 + 1] = Math.max(0, Math.min(255, data[i * 2 + 1]))
      out[i * 4 + 2] = 0
      out[i * 4 + 3] = 255
    }
    return out
  }
  if (allowNarrowChannels && data.length === pixels) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels; i++) {
      const value = Math.max(0, Math.min(255, data[i]))
      out[i * 4] = value
      out[i * 4 + 1] = value
      out[i * 4 + 2] = value
      out[i * 4 + 3] = 255
    }
    return out
  }

  return null
}

function normalizedUnsignedIntegerDataToRgba8(
  data: Uint16Array | Uint32Array,
  pixels: number,
  allowNarrowChannels: boolean,
  maxValue: number,
): Uint8Array | null {
  if (data.length === pixels * 4) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels * 4; i++) {
      out[i] = normalizedUnsignedIntegerToByte(data[i], maxValue)
    }
    return out
  }
  if (data.length === pixels * 3) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels; i++) {
      out[i * 4] = normalizedUnsignedIntegerToByte(data[i * 3], maxValue)
      out[i * 4 + 1] = normalizedUnsignedIntegerToByte(data[i * 3 + 1], maxValue)
      out[i * 4 + 2] = normalizedUnsignedIntegerToByte(data[i * 3 + 2], maxValue)
      out[i * 4 + 3] = 255
    }
    return out
  }
  if (allowNarrowChannels && data.length === pixels * 2) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels; i++) {
      out[i * 4] = normalizedUnsignedIntegerToByte(data[i * 2], maxValue)
      out[i * 4 + 1] = normalizedUnsignedIntegerToByte(data[i * 2 + 1], maxValue)
      out[i * 4 + 2] = 0
      out[i * 4 + 3] = 255
    }
    return out
  }
  if (allowNarrowChannels && data.length === pixels) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels; i++) {
      const value = normalizedUnsignedIntegerToByte(data[i], maxValue)
      out[i * 4] = value
      out[i * 4 + 1] = value
      out[i * 4 + 2] = value
      out[i * 4 + 3] = 255
    }
    return out
  }
  return null
}

function normalizedUnsignedIntegerToByte(value: number, maxValue: number): number {
  return Math.max(0, Math.min(255, Math.round((value / maxValue) * 255)))
}

function normalizedSignedIntegerDataToRgba8(
  data: Int8Array | Int16Array | Int32Array,
  pixels: number,
  allowNarrowChannels: boolean,
  maxValue: number,
): Uint8Array | null {
  if (data.length === pixels * 4) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels * 4; i++) {
      out[i] = normalizedSignedIntegerToByte(data[i], maxValue)
    }
    return out
  }
  if (data.length === pixels * 3) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels; i++) {
      out[i * 4] = normalizedSignedIntegerToByte(data[i * 3], maxValue)
      out[i * 4 + 1] = normalizedSignedIntegerToByte(data[i * 3 + 1], maxValue)
      out[i * 4 + 2] = normalizedSignedIntegerToByte(data[i * 3 + 2], maxValue)
      out[i * 4 + 3] = 255
    }
    return out
  }
  if (allowNarrowChannels && data.length === pixels * 2) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels; i++) {
      out[i * 4] = normalizedSignedIntegerToByte(data[i * 2], maxValue)
      out[i * 4 + 1] = normalizedSignedIntegerToByte(data[i * 2 + 1], maxValue)
      out[i * 4 + 2] = 0
      out[i * 4 + 3] = 255
    }
    return out
  }
  if (allowNarrowChannels && data.length === pixels) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels; i++) {
      const value = normalizedSignedIntegerToByte(data[i], maxValue)
      out[i * 4] = value
      out[i * 4 + 1] = value
      out[i * 4 + 2] = value
      out[i * 4 + 3] = 255
    }
    return out
  }
  return null
}

function normalizedSignedIntegerToByte(value: number, maxValue: number): number {
  return Math.max(0, Math.min(255, Math.round((Math.max(value, 0) / maxValue) * 255)))
}

function packedUnsignedShort4444ToRgba8(data: Uint16Array, pixels: number): Uint8Array | null {
  if (data.length !== pixels) return null
  const out = new Uint8Array(pixels * 4)
  for (let i = 0; i < pixels; i++) {
    const value = data[i]
    out[i * 4] = ((value >> 12) & 0xf) * 17
    out[i * 4 + 1] = ((value >> 8) & 0xf) * 17
    out[i * 4 + 2] = ((value >> 4) & 0xf) * 17
    out[i * 4 + 3] = (value & 0xf) * 17
  }
  return out
}

function packedUnsignedShort5551ToRgba8(data: Uint16Array, pixels: number): Uint8Array | null {
  if (data.length !== pixels) return null
  const out = new Uint8Array(pixels * 4)
  for (let i = 0; i < pixels; i++) {
    const value = data[i]
    out[i * 4] = normalizedUnsignedIntegerToByte((value >> 11) & 0x1f, 0x1f)
    out[i * 4 + 1] = normalizedUnsignedIntegerToByte((value >> 6) & 0x1f, 0x1f)
    out[i * 4 + 2] = normalizedUnsignedIntegerToByte((value >> 1) & 0x1f, 0x1f)
    out[i * 4 + 3] = (value & 0x1) === 1 ? 255 : 0
  }
  return out
}

function halfFloatDataToRgba8(
  data: Uint16Array,
  pixels: number,
  allowNarrowChannels: boolean,
): Uint8Array | null {
  if (data.length === pixels * 4) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels * 4; i++) {
      out[i] = halfFloatToByte(data[i])
    }
    return out
  }
  if (data.length === pixels * 3) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels; i++) {
      out[i * 4] = halfFloatToByte(data[i * 3])
      out[i * 4 + 1] = halfFloatToByte(data[i * 3 + 1])
      out[i * 4 + 2] = halfFloatToByte(data[i * 3 + 2])
      out[i * 4 + 3] = 255
    }
    return out
  }
  if (allowNarrowChannels && data.length === pixels * 2) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels; i++) {
      out[i * 4] = halfFloatToByte(data[i * 2])
      out[i * 4 + 1] = halfFloatToByte(data[i * 2 + 1])
      out[i * 4 + 2] = 0
      out[i * 4 + 3] = 255
    }
    return out
  }
  if (allowNarrowChannels && data.length === pixels) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels; i++) {
      const value = halfFloatToByte(data[i])
      out[i * 4] = value
      out[i * 4 + 1] = value
      out[i * 4 + 2] = value
      out[i * 4 + 3] = 255
    }
    return out
  }
  return null
}

function halfFloatToByte(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1
  const exponent = (bits >> 10) & 0x1f
  const mantissa = bits & 0x03ff
  let value: number
  if (exponent === 0) {
    value = sign * (mantissa / 0x400) * (2 ** -14)
  } else if (exponent === 0x1f) {
    value = mantissa === 0 ? sign * Infinity : Number.NaN
  } else {
    value = sign * (1 + mantissa / 0x400) * (2 ** (exponent - 15))
  }
  if (!Number.isFinite(value)) return value > 0 ? 255 : 0
  return Math.max(0, Math.min(255, Math.round(value * 255)))
}
