import type { Color4, ThreeMaterialLike, PbrProperties, TextureInfo, ThreeTextureLike, ThreeSceneRootLike, ThreeObject3DLike } from './types'
import { clamp01 } from './math'
import { strictColorLikeToArray, validatedColorLikeToArray } from './color'
import { objectChildren } from './objects'

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
const UnsignedInt101111Type = 35899
const LinearEncoding = 3000
const sRGBEncoding = 3001

// Three.js texture format constants
const AlphaFormat = 1021
const LuminanceAlphaFormat = 1025

const CompressedTextureFormats = new Set([
  33776, // RGB_S3TC_DXT1_Format
  33777, // RGBA_S3TC_DXT1_Format
  33778, // RGBA_S3TC_DXT3_Format
  33779, // RGBA_S3TC_DXT5_Format
  35840, // RGB_PVRTC_4BPPV1_Format
  35841, // RGB_PVRTC_2BPPV1_Format
  35842, // RGBA_PVRTC_4BPPV1_Format
  35843, // RGBA_PVRTC_2BPPV1_Format
  36196, // RGB_ETC1_Format
  36283, // RED_RGTC1_Format
  36284, // SIGNED_RED_RGTC1_Format
  36285, // RED_GREEN_RGTC2_Format
  36286, // SIGNED_RED_GREEN_RGTC2_Format
  36492, // RGBA_BPTC_Format
  36494, // RGB_BPTC_SIGNED_Format
  36495, // RGB_BPTC_UNSIGNED_Format
  37488, // R11_EAC_Format
  37489, // SIGNED_R11_EAC_Format
  37490, // RG11_EAC_Format
  37491, // SIGNED_RG11_EAC_Format
  37492, // RGB_ETC2_Format
  37496, // RGBA_ETC2_EAC_Format
  37808, // RGBA_ASTC_4x4_Format
  37809, // RGBA_ASTC_5x4_Format
  37810, // RGBA_ASTC_5x5_Format
  37811, // RGBA_ASTC_6x5_Format
  37812, // RGBA_ASTC_6x6_Format
  37813, // RGBA_ASTC_8x5_Format
  37814, // RGBA_ASTC_8x6_Format
  37815, // RGBA_ASTC_8x8_Format
  37816, // RGBA_ASTC_10x5_Format
  37817, // RGBA_ASTC_10x6_Format
  37818, // RGBA_ASTC_10x8_Format
  37819, // RGBA_ASTC_10x10_Format
  37820, // RGBA_ASTC_12x10_Format
  37821, // RGBA_ASTC_12x12_Format
])

export function isCompressedTextureFormat(format: unknown): boolean {
  return typeof format === 'number' && CompressedTextureFormats.has(format)
}

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
  textureCache?: TextureExtractionCache
  materialColorCache?: MaterialColorExtractionCache
  textureStateCache?: TextureStateExtractionCache
  materialRenderStateCache?: MaterialRenderStateExtractionCache
  materialScalarFeatureCache?: MaterialScalarFeatureExtractionCache
}

export type TextureExtractionCache = WeakMap<ThreeTextureLike, unknown>
export type MaterialColorExtractionCache = WeakMap<ThreeMaterialLike, unknown>
export type TextureStateExtractionCache = WeakMap<ThreeTextureLike, unknown>
export type MaterialRenderStateExtractionCache = WeakMap<ThreeMaterialLike, unknown>
export type MaterialScalarFeatureExtractionCache = WeakMap<ThreeMaterialLike, unknown>

export interface EnvironmentMapResolution {
  envMap: EnvironmentMapInfo | null
  materialContext?: MaterialExtractionContext
  rotation?: ThreeMaterialLike['envMapRotation']
}

type TextureImageInput = {
  data?: ArrayLike<number>
  width?: number
  height?: number
  getContext?: (contextId: string, options?: unknown) => unknown
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

export function resolveSceneOverrideMaterial(scene: ThreeSceneRootLike): ThreeMaterialLike | undefined {
  const overrideMaterial = scene.overrideMaterial
  if (overrideMaterial == null) return undefined
  assertMaterialLike(overrideMaterial, 'scene.overrideMaterial')
  return overrideMaterial
}

export function resolveEnvironmentMap(
  scene: ThreeSceneRootLike,
  intensityOverride?: number,
  overrideMaterial?: ThreeMaterialLike,
): EnvironmentMapResolution {
  const sceneEnvMap = extractEnvironmentMap(scene, intensityOverride)
  if (sceneEnvMap) {
    return { envMap: sceneEnvMap }
  }

  if (overrideMaterial) {
    const overrideEnvMap = extractOverrideMaterialEnvironmentMap(overrideMaterial)
    if (overrideEnvMap) {
      return overrideEnvMap
    }
    return { envMap: null }
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

function extractOverrideMaterialEnvironmentMap(material: ThreeMaterialLike): EnvironmentMapResolution | null {
  if (optionalBoolean(material.visible, 'scene.overrideMaterial.visible') === false) return null
  const materialEnvMap = material.envMap
  if (!materialEnvMap) return null
  if (!supportsNativeMaterialEnvironmentMap(material)) return null

  assertSupportedMaterialEnvironmentMap(material)
  const usesRefraction = isRefractionEnvironmentMapping(materialEnvMap.mapping)
  const envMap = extractEnvironmentMapFromTexture(materialEnvMap, 'material.envMap', 1, { allowRefraction: usesRefraction })
  const materials = new WeakSet<ThreeMaterialLike>()
  materials.add(material)

  return {
    envMap,
    rotation: materialEnvMapRotation(material),
    materialContext: {
      materialEnvironmentSource: 'material',
      materialEnvironmentMaps: materials,
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
      const materialEnvRotation = materialEnvMapRotation(material)
      if (materialEnvRotation) {
        if (envRotation && !sameVector3Like(envRotation, materialEnvRotation)) {
          throw new Error(
            'Multiple material.envMapRotation values are not supported by @headless-three/renderer yet. Use one shared material envMapRotation, scene.environmentRotation, or render separate passes.',
          )
        }
        envRotation = materialEnvRotation
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

    for (const child of objectChildren(object)) {
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
  if (!Array.isArray(material)) {
    assertMaterialLike(material, 'material')
    return [material]
  }
  const materials: ThreeMaterialLike[] = []
  for (let index = 0; index < material.length; index += 1) {
    const entry = material[index]
    if (!entry) continue
    assertMaterialLike(entry, `material[${index}]`)
    materials.push(entry)
  }
  return materials
}

function supportsNativeMaterialEnvironmentMap(material: ThreeMaterialLike): boolean {
  return material.isMeshStandardMaterial === true
    || material.isMeshPhysicalMaterial === true
    || material.isMeshBasicMaterial === true
    || material.isMeshPhongMaterial === true
    || material.isMeshLambertMaterial === true
}

function supportsLegacyMaterialEnvironmentRefraction(material: ThreeMaterialLike): boolean {
  return material.isMeshBasicMaterial === true
    || material.isMeshPhongMaterial === true
    || material.isMeshLambertMaterial === true
}

function assertSupportedMaterialEnvironmentMap(material: ThreeMaterialLike): void {
  const usesRefraction = isRefractionEnvironmentMapping(material.envMap!.mapping)
  if (usesRefraction && !supportsLegacyMaterialEnvironmentRefraction(material)) {
    throw new Error(
      'material.envMap refraction mappings are only supported for MeshBasicMaterial, MeshLambertMaterial, and MeshPhongMaterial by @headless-three/renderer yet. Use a reflection mapping, remove material.envMap, or render this material separately.',
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
  textureUnpackAlignment(envTex, label)
  const premultiplyAlpha = optionalTextureBoolean(envTex.premultiplyAlpha, `${label}.premultiplyAlpha`) === true
  if (isCubeEnvironmentTexture(envTex, label)) {
    const cube = cubeTextureToEquirectangular(envTex, label)
    return { data: cube.data, width: cube.width, height: cube.height, intensity, colorSpace: textureColorSpace(envTex) }
  }

  const sourceData = textureSourceData(envTex, label)
  const image = (envTex as any).image ?? sourceData
  if (!image) throw unsupportedTextureImageError(label, 'environment map rendering')

  // DataTexture: { data, width, height }
  if (image.data && image.width > 0 && image.height > 0) {
    const texType = (envTex as any).type ?? UnsignedByteType
    assertSupportedRawTextureType(texType, label, 'environment map rendering')
    const rawData = image.data as ArrayBufferView & { buffer: ArrayBuffer; byteOffset: number; byteLength: number }
    const texFormat = (envTex as any).format

    if (texType === HalfFloatType) {
      if (!(rawData instanceof Uint16Array)) {
        throw new Error(
          `${label} HalfFloatType environment maps must provide Uint16Array one-channel, two-channel, RGB, or RGBA pixel data.`,
        )
      }
      const buf = rawHalfFloatTextureDataToRgba(rawData, image.width, image.height, label, 'environment map rendering', { premultiplyAlpha, format: texFormat })
      return { data: buf, width: image.width, height: image.height, intensity, colorSpace: textureColorSpace(envTex) }
    }

    if (texType === FloatType) {
      if (!(rawData instanceof Float32Array)) {
        throw new Error(
          `${label} FloatType environment maps must provide Float32Array one-channel, two-channel, RGB, or RGBA pixel data.`,
        )
      }
      const buf = rawFloatTextureDataToRgba(rawData, image.width, image.height, label, 'environment map rendering', { premultiplyAlpha, format: texFormat })
      return { data: buf, width: image.width, height: image.height, intensity, colorSpace: textureColorSpace(envTex) }
    }

    // UnsignedByteType / default: convert to RGBA8
    const rgba = toRgba8(rawData as any, image.width, image.height, { type: texType, format: texFormat })
    if (rgba) {
      const data = premultiplyAlpha ? premultiplyRgbaAlpha(rgba) : rgba
      return {
        data: Buffer.from(data.buffer, data.byteOffset, data.byteLength),
        width: image.width,
        height: image.height,
        intensity,
        colorSpace: textureColorSpace(envTex),
      }
    }
    throw unsupportedRawTextureDataError(label, 'environment map rendering')
  }

  const canvasImage = canvasLikeImageToRgba(image, label)
  if (canvasImage) {
    const data = premultiplyAlpha ? premultiplyRgbaAlpha(canvasImage.rgba) : canvasImage.rgba
    return {
      data: Buffer.from(data.buffer, data.byteOffset, data.byteLength),
      width: canvasImage.width,
      height: canvasImage.height,
      intensity,
      colorSpace: textureColorSpace(envTex),
    }
  }

  // Encoded image buffer (e.g. loaded HDR encoded as PNG/EXR)
  if (Buffer.isBuffer(image)) {
    assertNoEncodedPremultiplyAlpha(envTex, label)
    return { data: image, width: 0, height: 0, intensity, colorSpace: textureColorSpace(envTex) }
  }
  if (image instanceof Uint8Array && !((image as any).width > 0)) {
    assertNoEncodedPremultiplyAlpha(envTex, label)
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
  const hintBag = sceneRendererHints(scene)
  const hints = hintBag?.value ?? {}
  const probesKey = hints.reflectionProbes != null ? 'reflectionProbes' : 'probes'
  const probes = hints.reflectionProbes ?? hints.probes
  if (probes != null && !Array.isArray(probes)) {
    const label = hintBag ? `${hintBag.label}.${probesKey}` : `scene.userData.${probesKey}`
    throw new TypeError(`${label} must be an array.`)
  }
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

function sceneRendererHints(scene: ThreeSceneRootLike): { value: Record<string, unknown>; label: string } | undefined {
  const userData = scene.userData
  if (userData == null) return undefined
  if (typeof userData !== 'object' || Array.isArray(userData)) {
    throw new TypeError('scene.userData must be an object.')
  }
  const value = userData.headlessThreeRenderer ?? userData.headlessRenderer
  if (value == null) return undefined
  const label = userData.headlessThreeRenderer != null
    ? 'scene.userData.headlessThreeRenderer'
    : 'scene.userData.headlessRenderer'
  if (typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be an object.`)
  }
  return { value: value as Record<string, unknown>, label }
}

export function materialForGroup(
  material: ThreeMaterialLike | ThreeMaterialLike[] | undefined,
  materialIndex: number,
): ThreeMaterialLike | undefined {
  if (Array.isArray(material)) {
    const index = material[materialIndex] != null ? materialIndex : 0
    const resolved = material[index]
    assertMaterialLike(resolved, `material[${index}]`)
    return resolved
  }
  assertMaterialLike(material, 'material')
  return material
}

export function assertMaterialLike(value: unknown, label: string): asserts value is ThreeMaterialLike | undefined {
  if (value == null) return
  if (typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be a material-like object.`)
  }
}

interface CachedMaterialColorExtraction {
  signature: MaterialColorSignature
  color: Color4
}

interface MaterialColorCacheEntry {
  base?: CachedMaterialColorExtraction
  slots?: Map<string, CachedMaterialColorExtraction>
}

interface MaterialColorSignature {
  color: unknown
  colorLength?: unknown
  r?: unknown
  g?: unknown
  b?: unknown
  a?: unknown
  opacity?: unknown
  values?: unknown[]
}

export function materialColor(
  material: ThreeMaterialLike | undefined,
  context: MaterialExtractionContext = {},
): Color4 {
  const signature = material && context.materialColorCache
    ? materialColorSignature(material)
    : null
  if (material && signature) {
    const cached = materialColorCacheEntry(context, material)?.base
    if (cached && sameMaterialColorSignature(cached.signature, signature)) {
      return copyColor(cached.color)
    }
  }

  const copyShader = copyShaderMaterialInfo(material)
  const color = validatedColorLikeToArray(material?.color, 'material.color') ?? [1, 1, 1, 1] as Color4
  const opacity = copyShader
    ? optionalFiniteNumber(copyShader.opacity, 'material.uniforms.opacity.value')
    : optionalFiniteNumber(material?.opacity, 'material.opacity')
  color[3] = clamp01(opacity ?? color[3] ?? 1)
  if (material && signature) {
    materialColorCacheEntry(context, material, true)!.base = { signature, color: copyColor(color) }
  }
  return color
}

function materialColorSignature(material: ThreeMaterialLike): MaterialColorSignature {
  const copyShader = copyShaderMaterialInfo(material)
  return materialSlotColorSignature(material.color, copyShader ? copyShader.opacity : material.opacity)
}

function materialSlotColor(
  material: ThreeMaterialLike,
  slot: string,
  value: unknown,
  label: string,
  context: MaterialExtractionContext,
): Color4 | null {
  const signature = context.materialColorCache
    ? materialSlotColorSignature(value)
    : null
  if (signature) {
    const cached = materialColorCacheEntry(context, material)?.slots?.get(slot)
    if (cached && sameMaterialColorSignature(cached.signature, signature)) {
      return copyColor(cached.color)
    }
  }

  const color = validatedColorLikeToArray(value, label)
  if (signature && color) {
    const entry = materialColorCacheEntry(context, material, true)!
    entry.slots ??= new Map()
    entry.slots.set(slot, { signature, color: copyColor(color) })
  }
  return color
}

function materialColorCacheEntry(
  context: MaterialExtractionContext,
  material: ThreeMaterialLike,
  create = false,
): MaterialColorCacheEntry | undefined {
  const cache = context.materialColorCache
  if (!cache) return undefined
  let entry = cache.get(material) as MaterialColorCacheEntry | undefined
  if (!entry && create) {
    entry = {}
    cache.set(material, entry)
  }
  return entry
}

function materialSlotColorSignature(color: unknown, opacity?: unknown): MaterialColorSignature {
  const signature: MaterialColorSignature = {
    color,
    opacity,
  }
  if (Array.isArray(color)) {
    signature.colorLength = color.length
    signature.values = color.slice()
  } else if (color && typeof color === 'object') {
    const shaped = color as { r?: unknown; g?: unknown; b?: unknown; a?: unknown }
    signature.r = shaped.r
    signature.g = shaped.g
    signature.b = shaped.b
    signature.a = shaped.a
  }
  return signature
}

function copyColor(color: Color4): Color4 {
  return color.slice() as Color4
}

function sameMaterialColorSignature(a: MaterialColorSignature, b: MaterialColorSignature): boolean {
  return a.color === b.color
    && a.colorLength === b.colorLength
    && a.r === b.r
    && a.g === b.g
    && a.b === b.b
    && a.a === b.a
    && a.opacity === b.opacity
    && sameUnknownArray(a.values, b.values)
}

function sameUnknownArray(a: unknown[] | undefined, b: unknown[] | undefined): boolean {
  if (a === b) return true
  if (!a || !b) return false
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i += 1) {
    if (!Object.is(a[i], b[i])) return false
  }
  return true
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

function assignPbrTextureSamplerState(
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

interface CachedMaterialScalarFeatureExtraction {
  signature: MaterialScalarFeatureSignature
  props: PbrProperties
}

interface MaterialScalarFeatureSignature {
  values: unknown[]
}

function materialScalarFeatureProperties(
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

function materialScalarFeatureSignature(
  material: ThreeMaterialLike,
  context: MaterialExtractionContext,
): MaterialScalarFeatureSignature {
  const iridescenceThicknessRange = material.iridescenceThicknessRange as ArrayLike<unknown> | undefined
  const hintInfo = material.isMeshDistanceMaterial ? materialHintSignatureInfo(material.userData) : []
  return {
    values: [
      context.materialEnvironmentSource,
      context.materialEnvironmentMaps?.has(material),
      material.envMap,
      material.envMap?.mapping,
      material.envMapIntensity,
      material.combine,
      material.reflectivity,
      material.refractionRatio,
      material.metalness,
      material.roughness,
      material.clearcoat,
      material.clearcoatRoughness,
      ...vector2SignatureValues(material.clearcoatNormalScale),
      material.sheenRoughness,
      material.anisotropy,
      material.anisotropyRotation,
      material.iridescence,
      material.iridescenceIOR,
      iridescenceThicknessRange,
      iridescenceThicknessRange?.length,
      iridescenceThicknessRange?.[0],
      iridescenceThicknessRange?.[1],
      material.transmission,
      material.dispersion,
      material.ior,
      material.thickness,
      material.attenuationDistance,
      material.specularIntensity,
      material.isMeshPhongMaterial,
      material.shininess,
      ...vector2SignatureValues(material.normalScale),
      material.isMeshDepthMaterial,
      material.depthPacking,
      material.isMeshDistanceMaterial,
      material.referencePosition,
      ...vector3SignatureValues(material.referencePosition),
      material.nearDistance,
      material.farDistance,
      ...hintInfo,
    ],
  }
}

function materialHintSignatureInfo(userData: Record<string, any> | undefined): unknown[] {
  const values: unknown[] = [userData]
  if (userData && typeof userData === 'object' && !Array.isArray(userData)) {
    const hints = userData.headlessThreeRenderer ?? userData.headlessRenderer
    values.push(hints)
    if (hints && typeof hints === 'object' && !Array.isArray(hints)) {
      const hintRecord = hints as Record<string, unknown>
      values.push(
        hintRecord.referencePosition,
        ...vector3SignatureValues(hintRecord.referencePosition),
        hintRecord.distanceReferencePosition,
        ...vector3SignatureValues(hintRecord.distanceReferencePosition),
        hintRecord.nearDistance,
        hintRecord.distanceNear,
        hintRecord.farDistance,
        hintRecord.distanceFar,
      )
    }
  }
  return values
}

function vector2SignatureValues(value: unknown): unknown[] {
  if (!value || typeof value !== 'object') return [value]
  const arrayLike = value as ArrayLike<unknown>
  return [
    value,
    arrayLike.length,
    arrayLike[0],
    arrayLike[1],
    (value as { x?: unknown }).x,
    (value as { y?: unknown }).y,
  ]
}

function vector3SignatureValues(value: unknown): unknown[] {
  if (!value || typeof value !== 'object') return [value]
  const arrayLike = value as ArrayLike<unknown>
  return [
    value,
    arrayLike.length,
    arrayLike[0],
    arrayLike[1],
    arrayLike[2],
    (value as { x?: unknown }).x,
    (value as { y?: unknown }).y,
    (value as { z?: unknown }).z,
  ]
}

function copyMaterialScalarFeatureProperties(props: PbrProperties): PbrProperties {
  return {
    ...props,
    clearcoatNormalScale: props.clearcoatNormalScale ? props.clearcoatNormalScale.slice() : undefined,
    normalScale: props.normalScale ? props.normalScale.slice() : undefined,
    distanceReferencePosition: props.distanceReferencePosition ? props.distanceReferencePosition.slice() : undefined,
  }
}

function sameMaterialScalarFeatureSignature(
  a: MaterialScalarFeatureSignature,
  b: MaterialScalarFeatureSignature,
): boolean {
  return sameUnknownArray(a.values, b.values)
}

interface CachedMaterialRenderStateExtraction {
  signature: MaterialRenderStateSignature
  props: PbrProperties
}

interface MaterialRenderStateSignature {
  values: unknown[]
  blendColor: MaterialColorSignature
}

function materialRenderStateProperties(
  material: ThreeMaterialLike,
  customFragmentShader: string | undefined,
  context: MaterialExtractionContext,
): PbrProperties {
  const signature = context.materialRenderStateCache
    ? materialRenderStateSignature(material, customFragmentShader)
    : null
  if (signature) {
    const cached = context.materialRenderStateCache?.get(material) as CachedMaterialRenderStateExtraction | undefined
    if (cached && sameMaterialRenderStateSignature(cached.signature, signature)) {
      return copyMaterialRenderStateProperties(cached.props)
    }
  }

  const props: PbrProperties = {}
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
  if (optionalBoolean(material.toneMapped, 'material.toneMapped') === false) {
    props.toneMapped = false
  }
  optionalBoolean(material.dithering, 'material.dithering')
  optionalMaterialPrecision(material.precision)
  const transparent = optionalBoolean(material.transparent, 'material.transparent')
  if (transparent !== undefined) {
    props.transparent = transparent
  }
  optionalBoolean(material.forceSinglePass, 'material.forceSinglePass')
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
      const blendColor = materialSlotColor(material, 'blendColor', material.blendColor, 'material.blendColor', context)
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

  if (customFragmentShader && shaderMaterialKind(material)) {
    props.shadingModel = 'basic'
  } else if (copyShaderMaterialInfo(material)) {
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

  if (signature) {
    context.materialRenderStateCache?.set(material, {
      signature,
      props: copyMaterialRenderStateProperties(props),
    })
  }
  return props
}

function materialRenderStateSignature(
  material: ThreeMaterialLike,
  customFragmentShader: string | undefined,
): MaterialRenderStateSignature {
  return {
    values: [
      customFragmentShader,
      material.alphaTest,
      material.alphaHash,
      material.alphaToCoverage,
      material.premultipliedAlpha,
      material.toneMapped,
      material.dithering,
      material.precision,
      material.transparent,
      material.forceSinglePass,
      material.blending,
      material.blendEquation,
      material.blendSrc,
      material.blendDst,
      material.blendEquationAlpha,
      material.blendSrcAlpha,
      material.blendDstAlpha,
      material.blendAlpha,
      material.depthTest,
      material.depthFunc,
      material.depthWrite,
      material.colorWrite,
      material.polygonOffset,
      material.polygonOffsetFactor,
      material.polygonOffsetUnits,
      material.stencilWrite,
      material.stencilWriteMask,
      material.stencilFunc,
      material.stencilRef,
      material.stencilFuncMask,
      material.stencilFail,
      material.stencilZFail,
      material.stencilZPass,
      material.side,
      material.shadowSide,
      material.flatShading,
      material.fog,
      material.type,
      copyShaderMaterialInfo(material) != null,
      material.isRawShaderMaterial,
      material.isNodeMaterial,
      material.isShaderMaterial,
      material.isMeshBasicMaterial,
      material.isSpriteMaterial,
      material.isMeshDepthMaterial,
      material.isMeshDistanceMaterial,
      material.isMeshLambertMaterial,
      material.isMeshNormalMaterial,
      material.isMeshMatcapMaterial,
      material.isMeshPhongMaterial,
      material.isMeshToonMaterial,
      material.isShadowMaterial,
    ],
    blendColor: materialSlotColorSignature(material.blendColor),
  }
}

function copyMaterialRenderStateProperties(props: PbrProperties): PbrProperties {
  return {
    ...props,
    blendColor: props.blendColor ? props.blendColor.slice() : undefined,
  }
}

function sameMaterialRenderStateSignature(
  a: MaterialRenderStateSignature,
  b: MaterialRenderStateSignature,
): boolean {
  return sameUnknownArray(a.values, b.values)
    && sameMaterialColorSignature(a.blendColor, b.blendColor)
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
  if (texture?.channel == null) return 0
  if (!Number.isInteger(texture.channel)) {
    throw new TypeError('texture.channel must be an integer.')
  }
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

function firstOptionalVector3LikeToArray(entries: Array<[unknown, string]>): number[] | undefined {
  for (const [value, label] of entries) {
    if (value != null) return requiredFiniteVector3LikeToArray(value, label)
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

function optionalPositiveFiniteNumber(value: unknown, label: string): void {
  if (value == null) return
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a finite number.`)
  }
  if (value <= 0) {
    throw new TypeError(`${label} must be positive.`)
  }
}

function optionalWireframeLinecap(value: unknown): void {
  if (value == null) return
  if (typeof value !== 'string') {
    throw new TypeError('material.wireframeLinecap must be a string.')
  }
  if (value !== 'butt' && value !== 'round' && value !== 'square') {
    throw new Error(
      `material.wireframeLinecap ${JSON.stringify(value)} is not supported by @headless-three/renderer. Use "butt", "round", "square", null, or undefined.`,
    )
  }
}

function optionalWireframeLinejoin(value: unknown): void {
  if (value == null) return
  if (typeof value !== 'string') {
    throw new TypeError('material.wireframeLinejoin must be a string.')
  }
  if (value !== 'round' && value !== 'bevel' && value !== 'miter') {
    throw new Error(
      `material.wireframeLinejoin ${JSON.stringify(value)} is not supported by @headless-three/renderer. Use "round", "bevel", "miter", null, or undefined.`,
    )
  }
}

function optionalMaterialPrecision(value: unknown): void {
  if (value == null) return
  if (typeof value !== 'string') {
    throw new TypeError('material.precision must be "highp", "mediump", "lowp", null, or undefined.')
  }
  if (value !== 'highp' && value !== 'mediump' && value !== 'lowp') {
    throw new Error(
      `material.precision ${JSON.stringify(value)} is not supported by @headless-three/renderer. Use "highp", "mediump", "lowp", null, or undefined.`,
    )
  }
}

function optionalFiniteNumberOrInfinityDefault(value: unknown, label: string): number | undefined {
  if (value === Number.POSITIVE_INFINITY) return undefined
  return optionalFiniteNumber(value, label)
}

function requiredFiniteNumber(value: unknown, label: string): number {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  throw new TypeError(`${label} must be a finite number.`)
}

function materialRangePair(value: unknown, label: string): [number, number] | undefined {
  if (value == null) return undefined
  if (typeof value !== 'object') {
    throw new TypeError(`${label} must be an array-like pair.`)
  }
  const range = value as ArrayLike<unknown>
  if (typeof range.length !== 'number' || range.length < 2) {
    throw new TypeError(`${label} must contain at least two values.`)
  }
  return [
    requiredFiniteNumber(range[0], `${label}[0]`),
    requiredFiniteNumber(range[1], `${label}[1]`),
  ]
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

function requiredFiniteVector3LikeToArray(value: unknown, label: string): number[] {
  if (!value || typeof value !== 'object') {
    throw new TypeError(`${label} must be a finite Vector3-like value.`)
  }

  const arrayLike = value as ArrayLike<unknown>
  if (typeof arrayLike.length === 'number' && arrayLike.length >= 3) {
    return [
      requiredFiniteNumber(arrayLike[0], `${label}[0]`),
      requiredFiniteNumber(arrayLike[1], `${label}[1]`),
      requiredFiniteNumber(arrayLike[2], `${label}[2]`),
    ]
  }

  const vector = value as { x?: unknown; y?: unknown; z?: unknown }
  if ('x' in vector || 'y' in vector || 'z' in vector) {
    return [
      requiredFiniteNumber(vector.x, `${label}.x`),
      requiredFiniteNumber(vector.y, `${label}.y`),
      requiredFiniteNumber(vector.z, `${label}.z`),
    ]
  }

  throw new TypeError(`${label} must be a finite Vector3-like value.`)
}

function materialEnvMapRotation(material: ThreeMaterialLike): ThreeMaterialLike['envMapRotation'] | undefined {
  const value = material.envMapRotation
  if (value == null) return undefined
  const components = requiredVector3LikeToArray(value, 'material.envMapRotation')
  return components.some((component) => Math.abs(component) > 1e-12)
    ? value
    : undefined
}

function requiredVector3LikeToArray(value: unknown, label: string): number[] {
  if (!value || typeof value !== 'object') {
    throw new TypeError(`${label} must be a finite Vector3-like value.`)
  }

  const arrayLike = value as ArrayLike<unknown>
  if (typeof arrayLike.length === 'number' && arrayLike.length >= 3) {
    validateEulerLikeOrder(arrayLike[3], `${label}[3]`)
    return [
      requiredFiniteNumber(arrayLike[0], `${label}[0]`),
      requiredFiniteNumber(arrayLike[1], `${label}[1]`),
      requiredFiniteNumber(arrayLike[2], `${label}[2]`),
    ]
  }

  const vector = value as { x?: unknown; y?: unknown; z?: unknown }
  if ('x' in vector || 'y' in vector || 'z' in vector) {
    validateEulerLikeOrder((vector as { order?: unknown }).order, `${label}.order`)
    return [
      requiredFiniteNumber(vector.x, `${label}.x`),
      requiredFiniteNumber(vector.y, `${label}.y`),
      requiredFiniteNumber(vector.z, `${label}.z`),
    ]
  }

  throw new TypeError(`${label} must be a finite Vector3-like value.`)
}

function validateEulerLikeOrder(value: unknown, label: string): void {
  if (value == null) return
  if (
    value === 'XYZ' ||
    value === 'YXZ' ||
    value === 'ZXY' ||
    value === 'ZYX' ||
    value === 'YZX' ||
    value === 'XZY'
  ) {
    return
  }
  throw new TypeError(`${label} must be one of XYZ, YXZ, ZXY, ZYX, YZX, or XZY.`)
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

  const candidates: Array<[unknown, string]> = [
    [material.customFragmentWgsl, 'material.customFragmentWgsl'],
    [material.customFragmentShader, 'material.customFragmentShader'],
    [material.headlessFragmentWgsl, 'material.headlessFragmentWgsl'],
    [material.headlessFragmentShader, 'material.headlessFragmentShader'],
  ]

  const hints = customFragmentHints(material.userData)
  if (hints) {
    candidates.push(
      [hints.value.fragmentWgsl, `${hints.label}.fragmentWgsl`],
      [hints.value.fragmentShader, `${hints.label}.fragmentShader`],
      [hints.value.customFragmentWgsl, `${hints.label}.customFragmentWgsl`],
      [hints.value.customFragmentShader, `${hints.label}.customFragmentShader`],
    )
  }

  for (const [value, label] of candidates) {
    const candidate = customFragmentCandidate(value, label)
    if (candidate) return candidate
  }

  return undefined
}

function materialRendererHints(userData: Record<string, any> | undefined): { value: Record<string, unknown>; label: string } | undefined {
  if (userData == null) return undefined
  if (typeof userData !== 'object' || Array.isArray(userData)) {
    throw new TypeError('material.userData must be an object.')
  }
  const value = userData.headlessThreeRenderer ?? userData.headlessRenderer
  if (value == null) return undefined
  const label = userData.headlessThreeRenderer != null
    ? 'material.userData.headlessThreeRenderer'
    : 'material.userData.headlessRenderer'
  if (typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be an object.`)
  }
  return { value: value as Record<string, unknown>, label }
}

function customFragmentHints(userData: Record<string, any> | undefined): { value: Record<string, unknown>; label: string } | undefined {
  return materialRendererHints(userData)
}

function customFragmentCandidate(value: unknown, label: string): string | undefined {
  if (value == null) return undefined
  if (typeof value !== 'string') {
    throw new TypeError(`${label} must be a string.`)
  }
  const candidate = value.trim()
  return candidate.length > 0 ? candidate : undefined
}

function assertSupportedShaderMaterial(
  material: ThreeMaterialLike,
  customFragmentShader: string | undefined,
): void {
  const kind = shaderMaterialKind(material)
  if (!kind || customFragmentShader || copyShaderMaterialInfo(material)) return

  if (isThreePmremShaderMaterial(material)) {
    throw new Error(
      `${material.name} is a Three.js PMREMGenerator internal ShaderMaterial and is not translated by @headless-three/renderer yet. Use readable scene.environment, scene.background, material.envMap, or reflection-probe textures directly so the renderer can run its native CPU IBL precompute path, or precompute PMREM/CubeUV assets before rendering.`,
    )
  }

  const label = namedShaderMaterialLabel(kind, material)
  throw new Error(
    `${label} is not supported directly by @headless-three/renderer. Use a built-in Three.js material, or provide material.userData.headlessThreeRenderer.fragmentWgsl with a WGSL fragment body for the renderer's custom material path.`,
  )
}

function isThreePmremShaderMaterial(material: ThreeMaterialLike): material is ThreeMaterialLike & { name: string } {
  return material.name === 'EquirectangularToCubeUV' ||
    material.name === 'CubemapToCubeUV' ||
    material.name === 'SphericalGaussianBlur'
}

function namedShaderMaterialLabel(kind: string, material: ThreeMaterialLike): string {
  return typeof material.name === 'string' && material.name.trim().length > 0
    ? `${kind} "${material.name}"`
    : kind
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
  optionalPositiveFiniteNumber(material.wireframeLinewidth, 'material.wireframeLinewidth')
  optionalWireframeLinecap(material.wireframeLinecap)
  optionalWireframeLinejoin(material.wireframeLinejoin)
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
  if (customFragmentShader || supportedMaterialClass(material) || copyShaderMaterialInfo(material)) return

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

interface CopyShaderMaterialInfo {
  texture: unknown
  opacity: unknown
}

function copyShaderMaterialInfo(material: ThreeMaterialLike | undefined): CopyShaderMaterialInfo | null {
  if (!material) return null
  const kind = shaderMaterialKind(material)
  if (kind !== 'ShaderMaterial' && kind !== 'RawShaderMaterial') return null
  if (kind === 'ShaderMaterial' && !isCopyShaderFragment(material.fragmentShader)) return null
  if (kind === 'RawShaderMaterial' && !isOutputShaderFragment(material.fragmentShader)) return null
  const uniforms = material.uniforms
  if (!uniforms || typeof uniforms !== 'object' || Array.isArray(uniforms)) return null
  return {
    texture: uniformValue((uniforms as Record<string, unknown>).tDiffuse),
    opacity: kind === 'ShaderMaterial'
      ? uniformValue((uniforms as Record<string, unknown>).opacity) ?? 1
      : 1,
  }
}

function uniformValue(uniform: unknown): unknown {
  if (!uniform || typeof uniform !== 'object' || Array.isArray(uniform)) return undefined
  return (uniform as { value?: unknown }).value
}

function isCopyShaderFragment(fragmentShader: unknown): boolean {
  if (typeof fragmentShader !== 'string') return false
  const compact = fragmentShader.replace(/\s+/g, '')
  return compact.includes('uniformfloatopacity;') &&
    compact.includes('uniformsampler2DtDiffuse;') &&
    compact.includes('texture2D(tDiffuse,vUv)') &&
    compact.includes('gl_FragColor=opacity*texel;')
}

function isOutputShaderFragment(fragmentShader: unknown): boolean {
  if (typeof fragmentShader !== 'string') return false
  const compact = fragmentShader.replace(/\s+/g, '')
  return compact.includes('uniformsampler2DtDiffuse;') &&
    compact.includes('gl_FragColor=texture2D(tDiffuse,vUv);') &&
    compact.includes('tonemapping_pars_fragment') &&
    compact.includes('colorspace_pars_fragment')
}

export function extractTextureData(
  material: ThreeMaterialLike | undefined,
  context: MaterialExtractionContext = {},
): TextureInfo | null {
  const copyShader = copyShaderMaterialInfo(material)
  const slot = copyShader
    ? copyShader.texture as ThreeMaterialLike['map']
    : material?.isMeshMatcapMaterial
      ? material.matcap
      : material?.map
  const label = copyShader
    ? 'material.uniforms.tDiffuse.value'
    : material?.isMeshMatcapMaterial
      ? 'material.matcap'
      : 'material.map'
  const base = extractTextureFromSlot(slot, label, context.textureCache)
  if (!base) return null

  const map = slot as ThreeTextureLike | null | undefined
  const state = textureSamplerState(map, label, context, {
    includeWrap: material?.isMeshMatcapMaterial !== true,
    includeTransform: material?.isMeshMatcapMaterial !== true,
    includeUvChannel: material?.isMeshMatcapMaterial !== true,
  })
  return { ...base, ...state }
}

interface CachedTextureStateExtraction {
  signature: TextureStateSignature
  state: TextureSamplerState
}

interface TextureSamplerState {
  wrapS?: string
  wrapT?: string
  magFilter?: string
  minFilter?: string
  anisotropy?: number
  transform?: number[]
  colorSpace?: string
  usesUv2?: boolean
}

interface TextureStateOptions {
  includeWrap?: boolean
  includeTransform?: boolean
  includeUvChannel?: boolean
}

interface TextureStateSignature {
  includeWrap?: boolean
  includeTransform?: boolean
  includeUvChannel?: boolean
  wrapS?: unknown
  wrapT?: unknown
  magFilter?: unknown
  minFilter?: unknown
  generateMipmaps?: unknown
  mipmaps?: unknown
  mipmapCount?: number
  anisotropy?: unknown
  colorSpace?: unknown
  encoding?: unknown
  channel?: unknown
  flipY?: unknown
  matrixAutoUpdate?: unknown
  matrix?: unknown
  matrixElements?: unknown
  matrixValues?: unknown[]
  offset?: unknown
  offsetX?: unknown
  offsetY?: unknown
  repeat?: unknown
  repeatX?: unknown
  repeatY?: unknown
  rotation?: unknown
  center?: unknown
  centerX?: unknown
  centerY?: unknown
}

function textureSamplerState(
  map: ThreeTextureLike | null | undefined,
  label: string,
  context: MaterialExtractionContext,
  options: TextureStateOptions = {},
): TextureSamplerState {
  const signature = map && context.textureStateCache
    ? textureStateSignature(map, options)
    : null
  if (map && signature) {
    const cached = context.textureStateCache?.get(map) as CachedTextureStateExtraction | undefined
    if (cached && sameTextureStateSignature(cached.signature, signature)) {
      return copyTextureSamplerState(cached.state)
    }
  }

  const state: TextureSamplerState = {
    wrapS: options.includeWrap === false ? undefined : wrapModeToString(map?.wrapS),
    wrapT: options.includeWrap === false ? undefined : wrapModeToString(map?.wrapT),
    magFilter: filterModeToString(map?.magFilter),
    minFilter: minFilterModeToString(map),
    anisotropy: textureAnisotropy(map, label),
    transform: options.includeTransform === false ? undefined : textureTransform(map, label),
    colorSpace: textureColorSpace(map),
    usesUv2: options.includeUvChannel === false ? false : textureUvChannel(map) > 0,
  }
  if (map && signature) {
    context.textureStateCache?.set(map, { signature, state: copyTextureSamplerState(state) })
  }
  return state
}

function textureStateSignature(
  texture: ThreeTextureLike,
  options: TextureStateOptions,
): TextureStateSignature {
  const mipmaps = texture.mipmaps
  const matrix = texture.matrix
  const matrixElements = matrix?.elements
  return {
    includeWrap: options.includeWrap,
    includeTransform: options.includeTransform,
    includeUvChannel: options.includeUvChannel,
    wrapS: texture.wrapS,
    wrapT: texture.wrapT,
    magFilter: texture.magFilter,
    minFilter: texture.minFilter,
    generateMipmaps: texture.generateMipmaps,
    mipmaps,
    mipmapCount: Array.isArray(mipmaps) ? mipmaps.length : undefined,
    anisotropy: texture.anisotropy,
    colorSpace: texture.colorSpace,
    encoding: texture.encoding,
    channel: texture.channel,
    flipY: texture.flipY,
    matrixAutoUpdate: texture.matrixAutoUpdate,
    matrix,
    matrixElements,
    matrixValues: matrixElements ? Array.from(matrixElements as ArrayLike<unknown>) : undefined,
    offset: texture.offset,
    offsetX: texture.offset?.x,
    offsetY: texture.offset?.y,
    repeat: texture.repeat,
    repeatX: texture.repeat?.x,
    repeatY: texture.repeat?.y,
    rotation: texture.rotation,
    center: texture.center,
    centerX: texture.center?.x,
    centerY: texture.center?.y,
  }
}

function copyTextureSamplerState(state: TextureSamplerState): TextureSamplerState {
  return {
    ...state,
    transform: state.transform ? state.transform.slice() : undefined,
  }
}

function sameTextureStateSignature(a: TextureStateSignature, b: TextureStateSignature): boolean {
  return a.includeWrap === b.includeWrap
    && a.includeTransform === b.includeTransform
    && a.includeUvChannel === b.includeUvChannel
    && a.wrapS === b.wrapS
    && a.wrapT === b.wrapT
    && a.magFilter === b.magFilter
    && a.minFilter === b.minFilter
    && a.generateMipmaps === b.generateMipmaps
    && a.mipmaps === b.mipmaps
    && a.mipmapCount === b.mipmapCount
    && a.anisotropy === b.anisotropy
    && a.colorSpace === b.colorSpace
    && a.encoding === b.encoding
    && a.channel === b.channel
    && a.flipY === b.flipY
    && a.matrixAutoUpdate === b.matrixAutoUpdate
    && a.matrix === b.matrix
    && a.matrixElements === b.matrixElements
    && sameUnknownArray(a.matrixValues, b.matrixValues)
    && a.offset === b.offset
    && a.offsetX === b.offsetX
    && a.offsetY === b.offsetY
    && a.repeat === b.repeat
    && a.repeatX === b.repeatX
    && a.repeatY === b.repeatY
    && a.rotation === b.rotation
    && a.center === b.center
    && a.centerX === b.centerX
    && a.centerY === b.centerY
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

function isCubeEnvironmentTexture(map: ThreeTextureLike, label = 'texture'): boolean {
  return map.isCubeTexture === true ||
    map.mapping === CubeReflectionMapping ||
    map.mapping === CubeRefractionMapping ||
    (
      map.mapping === CubeUVReflectionMapping &&
      (cubeFaceImages(map, label) !== null || cubeUvPackedImage(map, label) !== null)
    )
}

function extractCubeBackgroundTexture(map: ThreeTextureLike, label: string): TextureInfo {
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
  textureUnpackAlignment(map, label)
  const faces = cubeFaceImages(map, label)
  if (!faces) {
    if (map.mapping === CubeUVReflectionMapping) {
      const packedFaceTextures = packedCubeUvTextureToFaceTextures(map, label)
      if (packedFaceTextures) return cubeFaceTexturesToEquirectangular(packedFaceTextures, label)
      throw new Error(`${label} uses PMREM/CubeUV mapping without readable six-face cube images or a readable packed CubeUV image.`)
    }
    throw new Error(
      `${label} uses a cube texture without six raw or encoded face images. Provide a CubeTexture with six DataTexture-style or encoded PNG/JPEG/WebP face images, use a 2D/equirectangular texture, or pre-render the background to a 2D image before rendering.`,
    )
  }

  const premultiplyAlpha = optionalTextureBoolean(map.premultiplyAlpha, `${label}.premultiplyAlpha`) === true
  const faceTextures = faces.map((face, index) => imageToRgbaTexture(face, `${label}.image[${index}]`, map.type, map.format, { premultiplyAlpha }))
  return cubeFaceTexturesToEquirectangular(faceTextures, label)
}

function cubeFaceTexturesToEquirectangular(
  faceTextures: Array<{ rgba: Uint8Array; width: number; height: number }>,
  label: string,
): { data: Buffer; width: number; height: number } {
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

function cubeFaceImages(map: ThreeTextureLike, label = 'texture'): TextureImageInput[] | null {
  const sourceData = textureSourceData(map, label)
  const image = (map as any).image ?? sourceData
  if (Array.isArray(image) && image.length >= 6) return image.slice(0, 6) as TextureImageInput[]
  return null
}

function cubeUvPackedImage(map: ThreeTextureLike, label = 'texture'): TextureImageInput | null {
  if (map.mapping !== CubeUVReflectionMapping) return null
  const sourceData = textureSourceData(map, label)
  const image = (map as any).image ?? sourceData
  if (!image || Array.isArray(image)) return null
  if (Buffer.isBuffer(image) || image instanceof Uint8Array) return image
  if (typeof image === 'object') return image as TextureImageInput
  return null
}

function packedCubeUvTextureToFaceTextures(
  map: ThreeTextureLike,
  label: string,
): Array<{ rgba: Uint8Array; width: number; height: number }> | null {
  const packedImage = cubeUvPackedImage(map, label)
  if (!packedImage) return null

  const premultiplyAlpha = optionalTextureBoolean(map.premultiplyAlpha, `${label}.premultiplyAlpha`) === true
  const atlas = imageToRgbaTexture(packedImage, `${label}.image`, map.type, map.format, { premultiplyAlpha })
  if (atlas.height % 4 !== 0) {
    throw new Error(`${label} packed PMREM/CubeUV image height must be divisible by 4.`)
  }

  const faceSize = atlas.height / 4
  if (!Number.isInteger(faceSize) || faceSize < 16 || atlas.width < faceSize * 3) {
    throw new Error(
      `${label} packed PMREM/CubeUV image must use Three.js' 3-column by 4-row layout with at least 16x16 face tiles.`,
    )
  }

  const atlasFaceToCubeFace = [0, 2, 4, 1, 3, 5]
  const cubeFaces: Array<{ rgba: Uint8Array; width: number; height: number } | undefined> = []
  for (let atlasFace = 0; atlasFace < 6; atlasFace += 1) {
    const col = atlasFace % 3
    const row = atlasFace > 2 ? 1 : 0
    cubeFaces[atlasFaceToCubeFace[atlasFace]] = extractRgbaTile(atlas, col * faceSize, row * faceSize, faceSize)
  }

  return cubeFaces as Array<{ rgba: Uint8Array; width: number; height: number }>
}

function extractRgbaTile(
  source: { rgba: Uint8Array; width: number; height: number },
  x: number,
  y: number,
  size: number,
): { rgba: Uint8Array; width: number; height: number } {
  const out = new Uint8Array(size * size * 4)
  for (let row = 0; row < size; row += 1) {
    const sourceStart = ((y + row) * source.width + x) * 4
    out.set(source.rgba.subarray(sourceStart, sourceStart + size * 4), row * size * 4)
  }
  return { rgba: out, width: size, height: size }
}

function imageToRgbaTexture(
  image: TextureImageInput,
  label: string,
  textureType?: number,
  textureFormat?: unknown,
  options: { premultiplyAlpha?: boolean } = {},
): { rgba: Uint8Array; width: number; height: number } {
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
    return {
      rgba: options.premultiplyAlpha === true ? premultiplyRgbaAlpha(rgba) : rgba,
      width: decoded.width!,
      height: decoded.height!,
    }
  }
  if (!image || !image.data || !(image.width! > 0) || !(image.height! > 0)) {
    const canvasImage = canvasLikeImageToRgba(image, label)
    if (canvasImage) {
      return {
        rgba: options.premultiplyAlpha === true ? premultiplyRgbaAlpha(canvasImage.rgba) : canvasImage.rgba,
        width: canvasImage.width,
        height: canvasImage.height,
      }
    }
    throw new Error(`${label} must provide raw face data, width, and height for cube background rendering.`)
  }
  const rgba = toRgba8(image.data, image.width!, image.height!, { type: textureType, format: textureFormat })
  if (!rgba) {
    throw new Error(`${label} must contain RGB or RGBA numeric pixel data for cube background rendering.`)
  }
  return {
    rgba: options.premultiplyAlpha === true ? premultiplyRgbaAlpha(rgba) : rgba,
    width: image.width!,
    height: image.height!,
  }
}

export function canvasLikeImageToRgba(
  image: unknown,
  label: string,
): { rgba: Uint8Array; width: number; height: number } | null {
  if (!image || typeof image !== 'object') {
    return null
  }

  if (typeof (image as { getContext?: unknown }).getContext === 'function') {
    const candidate = image as {
      width?: unknown
      height?: unknown
      getContext: (contextId: string, options?: unknown) => unknown
    }
    const width = canvasLikeImageDimension(candidate.width, `${label}.width`)
    const height = canvasLikeImageDimension(candidate.height, `${label}.height`)
    const context = canvasLike2dContext(candidate, label)
    return canvasLikeContextToRgba(context, width, height, label)
  }

  return drawImageLikeToRgba(image, label)
}

function drawImageLikeToRgba(
  image: object,
  label: string,
): { rgba: Uint8Array; width: number; height: number } | null {
  const offscreenCanvas = (globalThis as unknown as {
    OffscreenCanvas?: new (width: number, height: number) => { getContext?: (contextId: string, options?: unknown) => unknown }
  }).OffscreenCanvas
  if (typeof offscreenCanvas !== 'function') return null

  const candidate = image as {
    width?: unknown
    height?: unknown
    naturalWidth?: unknown
    naturalHeight?: unknown
  }
  const width = canvasLikeImageDimension(candidate.width ?? candidate.naturalWidth, `${label}.width`)
  const height = canvasLikeImageDimension(candidate.height ?? candidate.naturalHeight, `${label}.height`)
  const canvas = new offscreenCanvas(width, height)
  const context = canvasLike2dContext(canvas, label)
  if (typeof (context as { drawImage?: unknown }).drawImage !== 'function') {
    throw new Error(`${label} OffscreenCanvas 2D context must provide drawImage() to read image-like texture pixels.`)
  }
  try {
    (context as { drawImage: (source: object, x: number, y: number, width: number, height: number) => unknown })
      .drawImage(image, 0, 0, width, height)
  } catch {
    throw new Error(`${label} OffscreenCanvas 2D context drawImage() failed while reading image-like texture pixels.`)
  }
  return canvasLikeContextToRgba(context, width, height, label)
}

function canvasLike2dContext(
  canvas: { getContext?: (contextId: string, options?: unknown) => unknown },
  label: string,
): unknown {
  let context: unknown
  try {
    context = canvas.getContext?.('2d', { willReadFrequently: true })
      ?? canvas.getContext?.('2d')
  } catch {
    throw new Error(`${label}.getContext("2d") failed while reading canvas texture pixels.`)
  }
  if (!context || typeof context !== 'object' || typeof (context as { getImageData?: unknown }).getImageData !== 'function') {
    throw new Error(`${label} canvas-like texture images must provide getContext("2d").getImageData().`)
  }
  return context
}

function canvasLikeContextToRgba(
  context: unknown,
  width: number,
  height: number,
  label: string,
): { rgba: Uint8Array; width: number; height: number } {
  let imageData: unknown
  try {
    imageData = (context as { getImageData: (x: number, y: number, width: number, height: number) => unknown })
      .getImageData(0, 0, width, height)
  } catch {
    throw new Error(`${label}.getContext("2d").getImageData() failed while reading canvas texture pixels.`)
  }

  if (!imageData || typeof imageData !== 'object') {
    throw new Error(`${label}.getContext("2d").getImageData() must return an ImageData-like object.`)
  }
  const data = (imageData as { data?: unknown }).data
  if (!(data instanceof Uint8Array) && !(data instanceof Uint8ClampedArray)) {
    throw new Error(`${label}.getContext("2d").getImageData().data must be a Uint8Array or Uint8ClampedArray.`)
  }
  if (data.length !== width * height * 4) {
    throw new Error(`${label}.getContext("2d").getImageData().data length must equal width * height * 4.`)
  }

  return {
    rgba: new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
    width,
    height,
  }
}

function canvasLikeImageDimension(value: unknown, label: string): number {
  if (!Number.isInteger(value) || (value as number) <= 0) {
    throw new TypeError(`${label} must be a positive integer for canvas-like texture image reads.`)
  }
  return value as number
}

function premultiplyRgbaAlpha(data: Uint8Array | Uint8ClampedArray): Uint8Array {
  const out = new Uint8Array(data.byteLength)
  for (let i = 0; i < data.byteLength; i += 4) {
    const alpha = data[i + 3]
    out[i] = Math.round((data[i] * alpha) / 255)
    out[i + 1] = Math.round((data[i + 1] * alpha) / 255)
    out[i + 2] = Math.round((data[i + 2] * alpha) / 255)
    out[i + 3] = alpha
  }
  return out
}

function premultiplyFloatRgba(data: Float32Array): Float32Array {
  const out = new Float32Array(data.length)
  for (let i = 0; i < data.length; i += 4) {
    const alpha = data[i + 3]
    out[i] = data[i] * alpha
    out[i + 1] = data[i + 1] * alpha
    out[i + 2] = data[i + 2] * alpha
    out[i + 3] = alpha
  }
  return out
}

function premultiplyHalfFloatRgba(data: Uint16Array): Uint16Array {
  const out = new Uint16Array(data.length)
  for (let i = 0; i < data.length; i += 4) {
    const alpha = halfFloatToNumber(data[i + 3])
    out[i] = numberToHalfFloat(halfFloatToNumber(data[i]) * alpha)
    out[i + 1] = numberToHalfFloat(halfFloatToNumber(data[i + 1]) * alpha)
    out[i + 2] = numberToHalfFloat(halfFloatToNumber(data[i + 2]) * alpha)
    out[i + 3] = data[i + 3]
  }
  return out
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

function textureSourceData(texture: ThreeTextureLike, label: string): unknown {
  const source = (texture as { source?: unknown }).source
  if (source == null) return undefined
  if (typeof source !== 'object' || Array.isArray(source)) {
    throw new TypeError(`${label}.source must be a source-like object.`)
  }
  const data = (source as { data?: unknown }).data
  if (data == null) return undefined
  if (typeof data !== 'object') {
    throw new TypeError(`${label}.source.data must be an image-like object.`)
  }
  return data
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

interface CachedTextureExtraction {
  signature: TexturePayloadSignature
  info: TextureInfo
}

interface TexturePayloadSignature {
  version: number
  image: unknown
  sourceData: unknown
  imageData?: TextureDataSignature
  imageWidth?: unknown
  imageHeight?: unknown
  type?: unknown
  format?: unknown
  premultiplyAlpha?: unknown
  generateMipmaps?: unknown
  mipmaps: TextureMipmapSignature[]
}

interface TextureMipmapSignature {
  image: unknown
  data?: TextureDataSignature
  width?: unknown
  height?: unknown
}

interface TextureDataSignature {
  data: unknown
  length?: unknown
  buffer?: ArrayBufferLike
  byteOffset?: number
  byteLength?: number
}

function extractTextureFromSlot(
  map: ThreeMaterialLike['map'],
  label = 'texture',
  cache?: TextureExtractionCache,
): TextureInfo | null {
  if (!map) return null
  assertSupportedTextureInput(map, label, { allowMipmaps: true })
  assertSupportedTwoDimensionalTextureSlot(map, label)
  textureUnpackAlignment(map, label)

  const sourceData = textureSourceData(map, label)
  const image = (map as any).image ?? sourceData
  if (!image) return null
  const signature = cache ? texturePayloadSignature(map, image, sourceData, label) : null
  if (signature) {
    const cached = cache?.get(map) as CachedTextureExtraction | undefined
    if (cached && texturePayloadSignaturesEqual(cached.signature, signature)) {
      return cached.info
    }
  }
  const cacheInfo = (info: TextureInfo): TextureInfo => {
    if (signature) {
      cache?.set(map, { signature, info })
    }
    return info
  }

  // DataTexture style: { data: TypedArray, width, height }
  if (image.data && image.width > 0 && image.height > 0) {
    assertSupportedRawTextureType((map as any).type, label, 'texture rendering')
    const rgba = toRgba8(image.data, image.width, image.height, { type: map.type, format: map.format })
    if (rgba) {
      const data = textureBytesWithExplicitMipmaps(map, label, rgba, image.width, image.height)
      return cacheInfo({ data: Buffer.from(data.buffer, data.byteOffset, data.byteLength), width: image.width, height: image.height })
    }
    throw unsupportedRawTextureDataError(label, 'texture rendering')
  }

  // Encoded image (PNG/JPEG/WebP Buffer from file loaders)
  if (Buffer.isBuffer(image)) {
    assertNoEncodedExplicitMipmaps(map, label)
    assertNoEncodedPremultiplyAlpha(map, label)
    return cacheInfo({ data: image, width: 0, height: 0 })
  }
  if (image instanceof Uint8Array && !((image as any).width > 0)) {
    assertNoEncodedExplicitMipmaps(map, label)
    assertNoEncodedPremultiplyAlpha(map, label)
    return cacheInfo({ data: Buffer.from(image.buffer, image.byteOffset, image.byteLength), width: 0, height: 0 })
  }

  // ImageData (canvas-based polyfill): { data: Uint8ClampedArray, width, height }
  if (image.data instanceof Uint8ClampedArray && image.width > 0 && image.height > 0) {
    const data = textureBytesWithExplicitMipmaps(map, label, image.data, image.width, image.height)
    return cacheInfo({
      data: Buffer.from(data.buffer, data.byteOffset, data.byteLength),
      width: image.width,
      height: image.height,
    })
  }

  const canvasImage = canvasLikeImageToRgba(image, label)
  if (canvasImage) {
    const data = textureBytesWithExplicitMipmaps(map, label, canvasImage.rgba, canvasImage.width, canvasImage.height)
    return cacheInfo({
      data: Buffer.from(data.buffer, data.byteOffset, data.byteLength),
      width: canvasImage.width,
      height: canvasImage.height,
    })
  }

  throw unsupportedTextureImageError(label, 'texture rendering')
}

function texturePayloadSignature(
  map: ThreeTextureLike,
  image: unknown,
  sourceData: unknown,
  label: string,
): TexturePayloadSignature | null {
  const version = map.version
  if (typeof version !== 'number' || !Number.isFinite(version)) return null

  return {
    version,
    image,
    sourceData,
    imageData: textureImageDataSignature(image),
    imageWidth: textureImageDimension(image, 'width'),
    imageHeight: textureImageDimension(image, 'height'),
    type: map.type,
    format: map.format,
    premultiplyAlpha: map.premultiplyAlpha,
    generateMipmaps: map.generateMipmaps,
    mipmaps: textureMipmapSignatures(map, label),
  }
}

function textureImageDataSignature(image: unknown): TextureDataSignature | undefined {
  if (Buffer.isBuffer(image) || image instanceof Uint8Array) {
    return textureDataSignature(image)
  }
  if (!image || typeof image !== 'object') return undefined
  return textureDataSignature((image as { data?: unknown }).data)
}

function textureDataSignature(data: unknown): TextureDataSignature | undefined {
  if (data == null) return undefined
  const arrayLike = data as { length?: unknown }
  const view = ArrayBuffer.isView(data) ? data as ArrayBufferView : undefined
  return {
    data,
    length: arrayLike.length,
    buffer: view?.buffer,
    byteOffset: view?.byteOffset,
    byteLength: view?.byteLength,
  }
}

function textureImageDimension(image: unknown, key: 'width' | 'height'): unknown {
  if (!image || typeof image !== 'object') return undefined
  return (image as Record<'width' | 'height', unknown>)[key]
}

function textureMipmapSignatures(map: ThreeTextureLike, label: string): TextureMipmapSignature[] {
  const mipmaps = map.mipmaps
  if (mipmaps == null) return []
  if (!Array.isArray(mipmaps)) {
    throw new TypeError(`${label}.mipmaps must be an array of image-like mip levels.`)
  }
  return mipmaps.map((image) => ({
    image,
    data: image && typeof image === 'object' ? textureDataSignature((image as { data?: unknown }).data) : undefined,
    width: image && typeof image === 'object' ? (image as { width?: unknown }).width : undefined,
    height: image && typeof image === 'object' ? (image as { height?: unknown }).height : undefined,
  }))
}

function texturePayloadSignaturesEqual(a: TexturePayloadSignature, b: TexturePayloadSignature): boolean {
  return a.version === b.version
    && a.image === b.image
    && a.sourceData === b.sourceData
    && textureDataSignaturesEqual(a.imageData, b.imageData)
    && a.imageWidth === b.imageWidth
    && a.imageHeight === b.imageHeight
    && a.type === b.type
    && a.format === b.format
    && a.premultiplyAlpha === b.premultiplyAlpha
    && a.generateMipmaps === b.generateMipmaps
    && textureMipmapSignaturesEqual(a.mipmaps, b.mipmaps)
}

function textureMipmapSignaturesEqual(a: TextureMipmapSignature[], b: TextureMipmapSignature[]): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i += 1) {
    if (
      a[i].image !== b[i].image ||
      !textureDataSignaturesEqual(a[i].data, b[i].data) ||
      a[i].width !== b[i].width ||
      a[i].height !== b[i].height
    ) {
      return false
    }
  }
  return true
}

function textureDataSignaturesEqual(a: TextureDataSignature | undefined, b: TextureDataSignature | undefined): boolean {
  if (a === b) return true
  if (!a || !b) return false
  return a.data === b.data
    && a.length === b.length
    && a.buffer === b.buffer
    && a.byteOffset === b.byteOffset
    && a.byteLength === b.byteLength
}

function hasExplicitMipmaps(texture: ThreeTextureLike | null | undefined, label = 'texture'): boolean {
  const mipmaps = texture?.mipmaps
  if (mipmaps == null) return false
  if (!Array.isArray(mipmaps)) {
    throw new TypeError(`${label}.mipmaps must be an array of image-like mip levels.`)
  }
  return mipmaps.length > 0
}

function assertNoEncodedExplicitMipmaps(map: ThreeTextureLike, label: string): void {
  if (!hasExplicitMipmaps(map, label)) return
  throw new Error(
    `${label} provides explicit texture mipmaps with an encoded base image. Explicit mipmap upload requires raw DataTexture-style base image data with raw mipmap levels.`,
  )
}

function assertNoEncodedPremultiplyAlpha(map: ThreeTextureLike, label: string): void {
  if (optionalTextureBoolean(map.premultiplyAlpha, `${label}.premultiplyAlpha`) !== true) return
  throw new Error(
    `${label}.premultiplyAlpha is only supported for readable raw texture image data. Decode the encoded image to raw RGBA DataTexture-style data before rendering.`,
  )
}

function textureBytesWithExplicitMipmaps(
  map: ThreeTextureLike,
  label: string,
  baseRgba: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
): Uint8Array | Uint8ClampedArray {
  const premultiplyAlpha = optionalTextureBoolean(map.premultiplyAlpha, `${label}.premultiplyAlpha`) === true
  const baseLevel = premultiplyAlpha ? premultiplyRgbaAlpha(baseRgba) : baseRgba
  if (!hasExplicitMipmaps(map, label)) return baseLevel
  if (width <= 1 && height <= 1) {
    throw new Error(
      `${label} provides explicit texture mipmaps for a ${width}x${height} base image, but no additional mip levels are valid after the 1x1 level.`,
    )
  }

  const levels: Uint8Array[] = [
    baseLevel instanceof Uint8Array
      ? new Uint8Array(baseLevel.buffer, baseLevel.byteOffset, baseLevel.byteLength)
      : new Uint8Array(baseLevel),
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
    const rgba = toRgba8(mip.data, expectedWidth, expectedHeight, { type: map.type, format: map.format })
    if (!rgba) {
      throw unsupportedRawTextureDataError(`${label}.mipmaps[${i}]`, 'texture rendering')
    }
    levels.push(premultiplyAlpha ? premultiplyRgbaAlpha(rgba) : rgba)

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
  const supported = 'one-channel, two-channel, RGB, or RGBA numeric pixel data'
  const expected = 'mismatched data lengths must match width * height texels, width * height * 2 values, width * height * 3 values, or width * height * 4 values; packed color types use one value per texel'
  return new Error(
    `${label} raw texture data must contain ${supported} for ${usage}; ${expected}.`,
  )
}

function unsupportedTextureImageError(label: string, usage: string): Error {
  return new Error(
    `${label} uses a texture image object that is not readable or drawable by @headless-three/renderer for ${usage}. Provide encoded PNG/JPEG/WebP bytes directly as texture.image or texture.source.data, a canvas-like object with getContext("2d").getImageData(), an image-like object drawable through an available OffscreenCanvas/2D canvas polyfill, or raw one-channel, two-channel, RGB, or RGBA numeric pixel data as { data, width, height } before rendering.`,
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
    type === UnsignedShort5551Type ||
    type === UnsignedInt5999Type ||
    type === UnsignedInt101111Type
  ) {
    return
  }
  throw new Error(
    `${label} raw texture type ${textureTypeName(type)} is not supported by @headless-three/renderer for ${usage}. Use UnsignedByteType, ByteType, ShortType, UnsignedShortType, IntType, UnsignedIntType, HalfFloatType, FloatType, UnsignedShort4444Type, UnsignedShort5551Type, UnsignedInt5999Type, or UnsignedInt101111Type raw data, or pre-convert the texture to RGBA8 before rendering.`,
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
    case UnsignedInt101111Type:
      return 'UnsignedInt101111Type'
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
): 1 | 2 | 3 | 4 {
  const pixels = width * height
  const length = typeof data.length === 'number' ? data.length : Number.NaN
  const channels = length / pixels
  if (channels === 1 || channels === 2 || channels === 3 || channels === 4) return channels
  throw unsupportedRawTextureDataError(label, usage)
}

function rawHalfFloatTextureDataToRgba(
  rawData: Uint16Array,
  width: number,
  height: number,
  label: string,
  usage: string,
  options: { premultiplyAlpha?: boolean; format?: unknown } = {},
): Buffer {
  const channels = rawTextureChannelCount(rawData, width, height, label, usage)
  if (channels === 4) {
    const data = options.premultiplyAlpha === true ? premultiplyHalfFloatRgba(rawData) : rawData
    return Buffer.from(data.buffer, data.byteOffset, data.byteLength)
  }
  const pixels = width * height
  const out = new Uint16Array(pixels * 4)
  for (let i = 0; i < pixels; i += 1) {
    if (channels === 1) {
      writeOneChannelRawRgba(out, i, rawData[i], 0x3C00, options.format)
    } else if (channels === 2) {
      writeTwoChannelRawRgba(out, i, rawData[i * channels], rawData[i * channels + 1], 0x3C00, options.format)
    } else {
      out[i * 4] = rawData[i * channels]
      out[i * 4 + 1] = rawData[i * channels + 1]
      out[i * 4 + 2] = rawData[i * channels + 2]
      out[i * 4 + 3] = 0x3C00
    }
  }
  const data = options.premultiplyAlpha === true ? premultiplyHalfFloatRgba(out) : out
  return Buffer.from(data.buffer, data.byteOffset, data.byteLength)
}

function rawFloatTextureDataToRgba(
  rawData: Float32Array,
  width: number,
  height: number,
  label: string,
  usage: string,
  options: { premultiplyAlpha?: boolean; format?: unknown } = {},
): Buffer {
  const channels = rawTextureChannelCount(rawData, width, height, label, usage)
  if (channels === 4) {
    const data = options.premultiplyAlpha === true ? premultiplyFloatRgba(rawData) : rawData
    return Buffer.from(data.buffer, data.byteOffset, data.byteLength)
  }
  const pixels = width * height
  const out = new Float32Array(pixels * 4)
  for (let i = 0; i < pixels; i += 1) {
    if (channels === 1) {
      writeOneChannelRawRgba(out, i, rawData[i], 1.0, options.format)
    } else if (channels === 2) {
      writeTwoChannelRawRgba(out, i, rawData[i * channels], rawData[i * channels + 1], 1.0, options.format)
    } else {
      out[i * 4] = rawData[i * channels]
      out[i * 4 + 1] = rawData[i * channels + 1]
      out[i * 4 + 2] = rawData[i * channels + 2]
      out[i * 4 + 3] = 1.0
    }
  }
  const data = options.premultiplyAlpha === true ? premultiplyFloatRgba(out) : out
  return Buffer.from(data.buffer, data.byteOffset, data.byteLength)
}

function writeOneChannelRawRgba(
  out: Uint16Array | Float32Array,
  pixelIndex: number,
  value: number,
  opaqueAlpha: number,
  format: unknown,
): void {
  const offset = pixelIndex * 4
  out[offset] = value
  out[offset + 1] = value
  out[offset + 2] = value
  out[offset + 3] = format === AlphaFormat ? value : opaqueAlpha
}

function writeTwoChannelRawRgba(
  out: Uint16Array | Float32Array,
  pixelIndex: number,
  first: number,
  second: number,
  opaqueAlpha: number,
  format: unknown,
): void {
  const offset = pixelIndex * 4
  if (format === LuminanceAlphaFormat) {
    out[offset] = first
    out[offset + 1] = first
    out[offset + 2] = first
    out[offset + 3] = second
    return
  }
  out[offset] = first
  out[offset + 1] = second
  out[offset + 2] = 0
  out[offset + 3] = opaqueAlpha
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
      `${label} uses a cube or PMREM/CubeUV texture mapping in a 2D background texture path. Use a readable six-face CubeTexture, a 2D/equirectangular texture, or pre-render the background to a 2D image before rendering.`,
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
  if (usesRefraction && options.allowRefraction !== true) {
    throw new Error(
      `${label} uses refraction environment mapping, which is not supported by @headless-three/renderer yet. Provide an equirectangular or six-face cube reflection texture and let the renderer precompute IBL, or pre-convert the source before rendering.`,
    )
  }
  if (map.mapping === CubeUVReflectionMapping && !isCubeEnvironmentTexture(map, label)) {
    throw new Error(
      `${label} uses PMREM/CubeUV environment mapping without readable six-face cube images, which is not supported by @headless-three/renderer yet. Provide a CubeUV-mapped CubeTexture, an equirectangular texture, or a six-face cube reflection texture and let the renderer precompute IBL.`,
    )
  }
}

function assertSupportedTextureInput(
  map: ThreeTextureLike,
  label: string,
  options: { allowMipmaps?: boolean } = {},
): void {
  if (map.isFramebufferTexture === true) {
    throw new Error(
      `${label} uses a FramebufferTexture, which is not supported by @headless-three/renderer texture slots. Copy framebuffer output into a readable raw texture with Renderer.copyFramebufferToTexture(), or render into a target-like object and use its color texture data.`,
    )
  }
  if (map.isDepthTexture === true) {
    throw new Error(
      `${label} uses a DepthTexture, which is only supported as target.depthTexture for render-target depth readback. Use a readable color texture for material, background, or environment slots.`,
    )
  }
  if (map.isVideoTexture === true) {
    throw new Error(
      `${label} uses a VideoTexture, which is not supported by @headless-three/renderer in Node because live video frames are not directly readable. Provide a canvas-like image exposing getContext("2d").getImageData(), an encoded image, or raw DataTexture pixels before rendering.`,
    )
  }
  if (map.isStorageTexture === true) {
    throw new Error(
      `${label} uses a StorageTexture, which is not supported by @headless-three/renderer texture slots because WebGPU storage texture backing data is not directly readable. Provide a readable raw, encoded, or canvas-like texture before rendering.`,
    )
  }
  if (
    map.isCompressedTexture === true ||
    map.isCompressedArrayTexture === true ||
    map.isCompressedCubeTexture === true
  ) {
    throw new Error(
      `${label} uses a compressed texture. KTX2, Basis, and THREE.CompressedTexture inputs are not decoded by @headless-three/renderer yet; pre-decode the texture to RGBA data or an encoded PNG/JPEG/WebP image before rendering.`,
    )
  }
  if (isCompressedTextureFormat(map.format)) {
    throw new Error(
      `${label} uses a compressed texture format. KTX2, Basis, and compressed texture formats are not decoded by @headless-three/renderer yet; pre-decode the texture to RGBA data or an encoded PNG/JPEG/WebP image before rendering.`,
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
  if (!options.allowMipmaps && hasExplicitMipmaps(map, label)) {
    throw new Error(
      `${label} provides explicit texture mipmaps, which are not uploaded by @headless-three/renderer yet. Provide only the base image level or prefilter/downsample the texture before rendering.`,
    )
  }
}

interface PhysicalMapFeatureGates {
  clearcoat: boolean
  sheen: boolean
  anisotropy: boolean
  iridescence: boolean
  transmission: boolean
}

function assertCompatiblePackedPhysicalMapSamplers(
  material: ThreeMaterialLike,
  features: PhysicalMapFeatureGates,
): void {
  const scalarSlots = activePackedPhysicalScalarMapSlots(material, features)
  const sheenSlots = features.sheen
    ? [
      ['sheenColorMap', material.sheenColorMap],
      ['sheenRoughnessMap', material.sheenRoughnessMap],
    ] satisfies Array<[string, ThreeTextureLike | null | undefined]>
    : []
  assertNoPackedPhysicalMapMipmaps('physical extension scalar maps', scalarSlots)
  assertNoPackedPhysicalMapMipmaps('physical extension sheen maps', sheenSlots)
  assertNoPackedPhysicalMapMipmaps('physical extension specular maps', [
    ['specularColorMap', material.specularColorMap],
    ['specularIntensityMap', material.specularIntensityMap],
  ])
  assertMatchingSamplerSettings('physical extension scalar maps', scalarSlots)
  assertMatchingSamplerSettings('physical extension sheen maps', sheenSlots)
  assertMatchingSamplerSettings('physical extension specular maps', [
    ['specularColorMap', material.specularColorMap],
    ['specularIntensityMap', material.specularIntensityMap],
  ])
}

function activePackedPhysicalScalarMapSlots(
  material: ThreeMaterialLike,
  features: PhysicalMapFeatureGates,
): Array<[string, ThreeTextureLike | null | undefined]> {
  const slots: Array<[string, ThreeTextureLike | null | undefined]> = []
  if (features.clearcoat) {
    slots.push(
      ['clearcoatMap', material.clearcoatMap],
      ['clearcoatRoughnessMap', material.clearcoatRoughnessMap],
    )
  }
  if (features.transmission) {
    slots.push(
      ['transmissionMap', material.transmissionMap],
      ['thicknessMap', material.thicknessMap],
    )
  }
  if (features.anisotropy) {
    slots.push(['anisotropyMap', material.anisotropyMap])
  }
  if (features.iridescence) {
    slots.push(
      ['iridescenceMap', material.iridescenceMap],
      ['iridescenceThicknessMap', material.iridescenceThicknessMap],
    )
  }
  return slots
}

function assertNoPackedPhysicalMapMipmaps(groupLabel: string, slots: Array<[string, ThreeTextureLike | null | undefined]>): void {
  for (const [label, texture] of slots) {
    if (!texture || !hasExplicitMipmaps(texture, `material.${label}`)) continue
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

function textureUnpackAlignment(map: ThreeTextureLike | null | undefined, label: string): number | undefined {
  const value = map?.unpackAlignment
  if (value == null) return undefined
  if (!Number.isInteger(value)) {
    throw new TypeError(`${label}.unpackAlignment must be an integer.`)
  }
  if (value === 1 || value === 2 || value === 4 || value === 8) return value
  throw new Error(
    `${label}.unpackAlignment ${value} is not supported by @headless-three/renderer. Use 1, 2, 4, or 8.`,
  )
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

  const tx = textureVector2Component(map.offset, `${label}.offset`, 'x', 0)
  const ty = textureVector2Component(map.offset, `${label}.offset`, 'y', 0)
  const sx = textureVector2Component(map.repeat, `${label}.repeat`, 'x', 1)
  const sy = textureVector2Component(map.repeat, `${label}.repeat`, 'y', 1)
  const rotation = finiteTextureTransformNumber(map.rotation, `${label}.rotation`, 0)
  const cx = textureVector2Component(map.center, `${label}.center`, 'x', 0)
  const cy = textureVector2Component(map.center, `${label}.center`, 'y', 0)
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

function textureVector2Component(value: unknown, label: string, component: 'x' | 'y', fallback: number): number {
  if (value == null) return fallback
  if (typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be a vector-like object.`)
  }
  return finiteTextureTransformNumber((value as { x?: unknown; y?: unknown })[component], `${label}.${component}`, fallback)
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
  options: { narrowChannels?: boolean; type?: number; format?: unknown } = {},
): Uint8Array | null {
  const pixels = width * height
  const allowNarrowChannels = options.narrowChannels !== false
  const textureType = options.type ?? UnsignedByteType
  const textureFormat = options.format

  if (textureType === HalfFloatType) {
    if (!(data instanceof Uint16Array)) return null
    return halfFloatDataToRgba8(data, pixels, allowNarrowChannels, textureFormat)
  }
  if (textureType === ByteType) {
    if (!(data instanceof Int8Array)) return null
    return normalizedSignedIntegerDataToRgba8(data, pixels, allowNarrowChannels, 0x7f, textureFormat)
  }
  if (textureType === ShortType) {
    if (!(data instanceof Int16Array)) return null
    return normalizedSignedIntegerDataToRgba8(data, pixels, allowNarrowChannels, 0x7fff, textureFormat)
  }
  if (textureType === UnsignedShortType) {
    if (!(data instanceof Uint16Array)) return null
    return normalizedUnsignedIntegerDataToRgba8(data, pixels, allowNarrowChannels, 0xffff, textureFormat)
  }
  if (textureType === IntType) {
    if (!(data instanceof Int32Array)) return null
    return normalizedSignedIntegerDataToRgba8(data, pixels, allowNarrowChannels, 0x7fffffff, textureFormat)
  }
  if (textureType === UnsignedIntType) {
    if (!(data instanceof Uint32Array)) return null
    return normalizedUnsignedIntegerDataToRgba8(data, pixels, allowNarrowChannels, 0xffffffff, textureFormat)
  }
  if (textureType === UnsignedInt5999Type) {
    if (!(data instanceof Uint32Array)) return null
    return packedUnsignedInt5999ToRgba8(data, pixels)
  }
  if (textureType === UnsignedInt101111Type) {
    if (!(data instanceof Uint32Array)) return null
    return packedUnsignedInt101111ToRgba8(data, pixels)
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
        writeTwoChannelRgba8(out, i, data[i * 2], data[i * 2 + 1], textureFormat)
      }
      return out
    }
    if (allowNarrowChannels && data.length === pixels) {
      const out = new Uint8Array(pixels * 4)
      for (let i = 0; i < pixels; i++) {
        writeOneChannelRgba8(out, i, data[i], textureFormat)
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
        writeTwoChannelRgba8(
          out,
          i,
          Math.max(0, Math.min(255, Math.round(data[i * 2] * 255))),
          Math.max(0, Math.min(255, Math.round(data[i * 2 + 1] * 255))),
          textureFormat,
        )
      }
      return out
    }
    if (allowNarrowChannels && data.length === pixels) {
      const out = new Uint8Array(pixels * 4)
      for (let i = 0; i < pixels; i++) {
        const value = Math.max(0, Math.min(255, Math.round(data[i] * 255)))
        writeOneChannelRgba8(out, i, value, textureFormat)
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
        writeTwoChannelRgba8(
          out,
          i,
          Math.max(0, Math.min(255, (data as any)[i * 2])),
          Math.max(0, Math.min(255, (data as any)[i * 2 + 1])),
          textureFormat,
        )
      }
      return out
    }
    if (allowNarrowChannels && data.length === pixels) {
      const out = new Uint8Array(pixels * 4)
      for (let i = 0; i < pixels; i++) {
        const value = Math.max(0, Math.min(255, (data as any)[i]))
        writeOneChannelRgba8(out, i, value, textureFormat)
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
      writeTwoChannelRgba8(
        out,
        i,
        Math.max(0, Math.min(255, data[i * 2])),
        Math.max(0, Math.min(255, data[i * 2 + 1])),
        textureFormat,
      )
    }
    return out
  }
  if (allowNarrowChannels && data.length === pixels) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels; i++) {
      const value = Math.max(0, Math.min(255, data[i]))
      writeOneChannelRgba8(out, i, value, textureFormat)
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
  format: unknown,
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
      writeTwoChannelRgba8(
        out,
        i,
        normalizedUnsignedIntegerToByte(data[i * 2], maxValue),
        normalizedUnsignedIntegerToByte(data[i * 2 + 1], maxValue),
        format,
      )
    }
    return out
  }
  if (allowNarrowChannels && data.length === pixels) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels; i++) {
      const value = normalizedUnsignedIntegerToByte(data[i], maxValue)
      writeOneChannelRgba8(out, i, value, format)
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
  format: unknown,
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
      writeTwoChannelRgba8(
        out,
        i,
        normalizedSignedIntegerToByte(data[i * 2], maxValue),
        normalizedSignedIntegerToByte(data[i * 2 + 1], maxValue),
        format,
      )
    }
    return out
  }
  if (allowNarrowChannels && data.length === pixels) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels; i++) {
      const value = normalizedSignedIntegerToByte(data[i], maxValue)
      writeOneChannelRgba8(out, i, value, format)
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

function packedUnsignedInt5999ToRgba8(data: Uint32Array, pixels: number): Uint8Array | null {
  if (data.length !== pixels) return null
  const out = new Uint8Array(pixels * 4)
  for (let i = 0; i < pixels; i++) {
    const value = data[i]
    const scale = 2 ** (((value >>> 27) & 0x1f) - 24)
    out[i * 4] = normalizedPackedRgb9E5ToByte(value & 0x1ff, scale)
    out[i * 4 + 1] = normalizedPackedRgb9E5ToByte((value >>> 9) & 0x1ff, scale)
    out[i * 4 + 2] = normalizedPackedRgb9E5ToByte((value >>> 18) & 0x1ff, scale)
    out[i * 4 + 3] = 255
  }
  return out
}

function normalizedPackedRgb9E5ToByte(mantissa: number, scale: number): number {
  return Math.max(0, Math.min(255, Math.round(mantissa * scale * 255)))
}

function packedUnsignedInt101111ToRgba8(data: Uint32Array, pixels: number): Uint8Array | null {
  if (data.length !== pixels) return null
  const out = new Uint8Array(pixels * 4)
  for (let i = 0; i < pixels; i++) {
    const value = data[i]
    out[i * 4] = unsignedPackedFloatToByte(value & 0x7ff, 6)
    out[i * 4 + 1] = unsignedPackedFloatToByte((value >>> 11) & 0x7ff, 6)
    out[i * 4 + 2] = unsignedPackedFloatToByte((value >>> 22) & 0x3ff, 5)
    out[i * 4 + 3] = 255
  }
  return out
}

function unsignedPackedFloatToByte(bits: number, mantissaBits: 5 | 6): number {
  const exponent = bits >>> mantissaBits
  const mantissa = bits & ((1 << mantissaBits) - 1)
  let value: number
  if (exponent === 0) {
    value = (mantissa / (2 ** mantissaBits)) * (2 ** -14)
  } else if (exponent === 0x1f) {
    value = mantissa === 0 ? Infinity : Number.NaN
  } else {
    value = (1 + mantissa / (2 ** mantissaBits)) * (2 ** (exponent - 15))
  }
  if (!Number.isFinite(value)) return value > 0 ? 255 : 0
  return Math.max(0, Math.min(255, Math.round(value * 255)))
}

function halfFloatDataToRgba8(
  data: Uint16Array,
  pixels: number,
  allowNarrowChannels: boolean,
  format: unknown,
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
      writeTwoChannelRgba8(out, i, halfFloatToByte(data[i * 2]), halfFloatToByte(data[i * 2 + 1]), format)
    }
    return out
  }
  if (allowNarrowChannels && data.length === pixels) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels; i++) {
      const value = halfFloatToByte(data[i])
      writeOneChannelRgba8(out, i, value, format)
    }
    return out
  }
  return null
}

function writeOneChannelRgba8(
  out: Uint8Array,
  pixelIndex: number,
  value: number,
  format: unknown,
): void {
  const offset = pixelIndex * 4
  out[offset] = value
  out[offset + 1] = value
  out[offset + 2] = value
  out[offset + 3] = format === AlphaFormat ? value : 255
}

function writeTwoChannelRgba8(
  out: Uint8Array,
  pixelIndex: number,
  first: number,
  second: number,
  format: unknown,
): void {
  const offset = pixelIndex * 4
  if (format === LuminanceAlphaFormat) {
    out[offset] = first
    out[offset + 1] = first
    out[offset + 2] = first
    out[offset + 3] = second
    return
  }
  out[offset] = first
  out[offset + 1] = second
  out[offset + 2] = 0
  out[offset + 3] = 255
}

function halfFloatToByte(bits: number): number {
  const value = halfFloatToNumber(bits)
  if (!Number.isFinite(value)) return value > 0 ? 255 : 0
  return Math.max(0, Math.min(255, Math.round(value * 255)))
}

function halfFloatToNumber(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1
  const exponent = (bits >> 10) & 0x1f
  const mantissa = bits & 0x03ff
  if (exponent === 0) {
    return sign * (mantissa / 0x400) * (2 ** -14)
  }
  if (exponent === 0x1f) {
    return mantissa === 0 ? sign * Infinity : Number.NaN
  }
  return sign * (1 + mantissa / 0x400) * (2 ** (exponent - 15))
}

function numberToHalfFloat(value: number): number {
  if (Number.isNaN(value)) return 0x7e00
  const sign = value < 0 || Object.is(value, -0) ? 0x8000 : 0
  const abs = Math.abs(value)
  if (abs === 0) return sign
  if (!Number.isFinite(abs)) return sign | 0x7c00
  if (abs >= 65504) return sign | 0x7bff
  if (abs < 2 ** -14) {
    return sign | Math.round(abs / (2 ** -24))
  }

  const exponent = Math.floor(Math.log2(abs))
  let mantissa = Math.round((abs / (2 ** exponent) - 1) * 0x400)
  let biasedExponent = exponent + 15
  if (mantissa === 0x400) {
    mantissa = 0
    biasedExponent += 1
  }
  if (biasedExponent >= 31) return sign | 0x7bff
  return sign | (biasedExponent << 10) | mantissa
}
