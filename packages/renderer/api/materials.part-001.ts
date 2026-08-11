import type { Color4, ThreeMaterialLike, PbrProperties, TextureInfo, ThreeTextureLike, ThreeSceneRootLike, ThreeObject3DLike } from './types'
import { clamp01 } from './math'
import { strictColorLikeToArray, validatedColorLikeToArray } from './color'
import { objectChildren } from './objects'
import { assertMaterialLike, assertSupportedMaterialEnvironmentMap, extractEnvironmentMapFromTexture, extractReflectionProbe, supportsNativeMaterialEnvironmentMap } from './materials.part-002'
import { materialEnvMapRotation, optionalBoolean, optionalFiniteNumber, sameVector3Like } from './materials.part-005'
import { requiredEnvironmentTexture } from './materials.part-009'
import { isRefractionEnvironmentMapping } from './materials.part-010'
// eslint-disable-next-line @typescript-eslint/no-var-requires
export const native = require('../native.js') as {
  decodeImage?(data: Buffer): { data?: Buffer | Uint8Array; width?: number; height?: number }
}

// Three.js wrapping constants
export const RepeatWrapping = 1000
export const ClampToEdgeWrapping = 1001
export const MirroredRepeatWrapping = 1002
export const NearestFilter = 1003
export const NearestMipmapNearestFilter = 1004
export const NearestMipmapLinearFilter = 1005
export const LinearFilter = 1006
export const LinearMipmapNearestFilter = 1007
export const LinearMipmapLinearFilter = 1008

// Three.js texture mapping constants
export const CubeReflectionMapping = 301
export const CubeRefractionMapping = 302
export const EquirectangularReflectionMapping = 303
export const EquirectangularRefractionMapping = 304
export const CubeUVReflectionMapping = 306

// Three.js normal map type constants
export const TangentSpaceNormalMap = 0
export const ObjectSpaceNormalMap = 1

// Three.js environment combine constants
export const MultiplyOperation = 0
export const MixOperation = 1
export const AddOperation = 2

// Three.js side constants
export const FrontSide = 0
export const BackSide = 1
export const DoubleSide = 2

// Three.js depth comparison constants
export const NeverDepth = 0
export const AlwaysDepth = 1
export const LessDepth = 2
export const LessEqualDepth = 3
export const EqualDepth = 4
export const GreaterEqualDepth = 5
export const GreaterDepth = 6
export const NotEqualDepth = 7

// Three.js blending constants
export const NoBlending = 0
export const NormalBlending = 1
export const AdditiveBlending = 2
export const SubtractiveBlending = 3
export const MultiplyBlending = 4
export const CustomBlending = 5
export const AddEquation = 100
export const SubtractEquation = 101
export const ReverseSubtractEquation = 102
export const MinEquation = 103
export const MaxEquation = 104
export const ZeroFactor = 200
export const OneFactor = 201
export const SrcColorFactor = 202
export const OneMinusSrcColorFactor = 203
export const SrcAlphaFactor = 204
export const OneMinusSrcAlphaFactor = 205
export const DstAlphaFactor = 206
export const OneMinusDstAlphaFactor = 207
export const DstColorFactor = 208
export const OneMinusDstColorFactor = 209
export const SrcAlphaSaturateFactor = 210
export const ConstantColorFactor = 211
export const OneMinusConstantColorFactor = 212
export const ConstantAlphaFactor = 213
export const OneMinusConstantAlphaFactor = 214
export const NeverStencilFunc = 512
export const LessStencilFunc = 513
export const EqualStencilFunc = 514
export const LessEqualStencilFunc = 515
export const GreaterStencilFunc = 516
export const NotEqualStencilFunc = 517
export const GreaterEqualStencilFunc = 518
export const AlwaysStencilFunc = 519
export const ZeroStencilOp = 0
export const KeepStencilOp = 7680
export const ReplaceStencilOp = 7681
export const IncrementStencilOp = 7682
export const DecrementStencilOp = 7683
export const IncrementWrapStencilOp = 34055
export const DecrementWrapStencilOp = 34056
export const InvertStencilOp = 5386

// Three.js depth-packing constants
export const BasicDepthPacking = 3200
export const RGBADepthPacking = 3201
export const RGBDepthPacking = 3202
export const RGDepthPacking = 3203

// Three.js texture type constants
export const UnsignedByteType = 1009
export const ByteType = 1010
export const ShortType = 1011
export const UnsignedShortType = 1012
export const IntType = 1013
export const UnsignedIntType = 1014
export const HalfFloatType = 1016
export const FloatType = 1015
export const UnsignedShort4444Type = 1017
export const UnsignedShort5551Type = 1018
export const UnsignedInt248Type = 1020
export const UnsignedInt5999Type = 35902
export const UnsignedInt101111Type = 35899
export const LinearEncoding = 3000
export const sRGBEncoding = 3001

// Three.js texture format constants
export const AlphaFormat = 1021
export const LuminanceAlphaFormat = 1025

export const CompressedTextureFormats = new Set([
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

export const DefaultOnBeforeCompileSource = 'onBeforeCompile( /* shaderobject, renderer */ ) {}'

export const MaterialNodeHookProperties = new Set<string>([
  'alphaTestNode',
  'aoNode',
  'attenuationColorNode',
  'attenuationDistanceNode',
  'backdropAlphaNode',
  'backdropNode',
  'castShadowNode',
  'clearcoatNode',
  'clearcoatRoughnessNode',
  'colorNode',
  'dashScaleNode',
  'dashSizeNode',
  'depthNode',
  'dispersionNode',
  'emissiveNode',
  'envNode',
  'fragmentNode',
  'gapSizeNode',
  'geometryNode',
  'iorNode',
  'iridescenceIORNode',
  'iridescenceNode',
  'iridescenceThicknessNode',
  'lightMapNode',
  'lightsNode',
  'metalnessNode',
  'mrtNode',
  'normalNode',
  'offsetNode',
  'opacityNode',
  'outputNode',
  'positionNode',
  'receivedShadowNode',
  'rotationNode',
  'roughnessNode',
  'scaleNode',
  'shadowPositionNode',
  'sheenNode',
  'sheenRoughnessNode',
  'shininessNode',
  'specularNode',
  'thicknessNode',
  'transmissionNode',
  'vertexNode',
])

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

export type TextureImageInput = {
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

export function extractOverrideMaterialEnvironmentMap(material: ThreeMaterialLike): EnvironmentMapResolution | null {
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

export function extractMaterialEnvironmentMap(
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

export function objectMaterials(
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
