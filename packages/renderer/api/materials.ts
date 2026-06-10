import type { Color4, ThreeMaterialLike, PbrProperties, TextureInfo, ThreeTextureLike, ThreeSceneLike } from './types'
import { clamp01 } from './math'
import { colorLikeToArray } from './color'

// Three.js wrapping constants
const RepeatWrapping = 1000
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

// Three.js side constants
const FrontSide = 0
const BackSide = 1
const DoubleSide = 2

// Three.js blending constants
const NoBlending = 0
const NormalBlending = 1
const AdditiveBlending = 2
const SubtractiveBlending = 3
const MultiplyBlending = 4
const CustomBlending = 5
const AddEquation = 100
const SrcAlphaFactor = 204
const OneMinusSrcAlphaFactor = 205
const AlwaysStencilFunc = 519
const KeepStencilOp = 7680

// Three.js depth-packing constants
const BasicDepthPacking = 3200
const RGBADepthPacking = 3201
const RGBDepthPacking = 3202
const RGDepthPacking = 3203

// Three.js texture type constants
const UnsignedByteType = 1009
const HalfFloatType = 1016
const FloatType = 1015
const sRGBEncoding = 3001

const DefaultOnBeforeCompileSource = 'onBeforeCompile( /* shaderobject, renderer */ ) {}'

export interface EnvironmentMapInfo {
  data: Buffer
  width: number
  height: number
  intensity: number
}

/**
 * Extract environment map data from scene.environment.
 * Supports DataTexture (equirectangular) with Uint8, Float16, Float32 pixel data.
 * Passes raw typed-array bytes to Rust which handles format detection.
 */
export function extractEnvironmentMap(scene: ThreeSceneLike): EnvironmentMapInfo | null {
  const probe = extractReflectionProbe(scene)
  const envTex = scene.environment ?? probe?.texture
  if (!envTex) return null
  assertUncompressedTexture(envTex, 'scene.environment')

  const image = (envTex as any).image ?? (envTex as any).source?.data
  if (!image) return null

  const intensity = probe?.intensity ?? (scene as any).environmentIntensity ?? 1.0

  // DataTexture: { data, width, height }
  if (image.data && image.width > 0 && image.height > 0) {
    const texType = (envTex as any).type ?? UnsignedByteType
    const rawData = image.data as ArrayBufferView & { buffer: ArrayBuffer; byteOffset: number; byteLength: number }

    if (texType === HalfFloatType && rawData instanceof Uint16Array) {
      // Pass raw half-float bytes — Rust ibl.rs decodes them
      const channels = rawData.length / (image.width * image.height)
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
      return { data: buf, width: image.width, height: image.height, intensity }
    }

    if (texType === FloatType && rawData instanceof Float32Array) {
      const channels = rawData.length / (image.width * image.height)
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
      return { data: buf, width: image.width, height: image.height, intensity }
    }

    // UnsignedByteType / default: convert to RGBA8
    const rgba = toRgba8(rawData as any, image.width, image.height)
    if (rgba) {
      return { data: Buffer.from(rgba.buffer, rgba.byteOffset, rgba.byteLength), width: image.width, height: image.height, intensity }
    }
  }

  // Encoded image buffer (e.g. loaded HDR encoded as PNG/EXR)
  if (Buffer.isBuffer(image)) {
    return { data: image, width: 0, height: 0, intensity }
  }
  if (image instanceof Uint8Array && !((image as any).width > 0)) {
    return { data: Buffer.from(image.buffer, image.byteOffset, image.byteLength), width: 0, height: 0, intensity }
  }

  return null
}

function extractReflectionProbe(scene: ThreeSceneLike): { texture: ThreeTextureLike; intensity?: number } | null {
  const hints = scene.userData?.headlessThreeRenderer ?? scene.userData?.headlessRenderer ?? {}
  const probes = hints.reflectionProbes ?? hints.probes
  const probe = hints.reflectionProbe ?? (Array.isArray(probes) ? probes[0] : undefined)
  const candidate = probe?.texture ?? probe?.map ?? probe
  if (!candidate) return null
  return {
    texture: candidate as ThreeTextureLike,
    intensity: Number.isFinite(probe?.intensity) ? probe.intensity : undefined,
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
  const color = colorLikeToArray(material?.color) ?? [1, 1, 1, 1] as Color4
  color[3] = clamp01(material?.opacity ?? color[3] ?? 1)
  return color
}

export function extractPbrProperties(material: ThreeMaterialLike | undefined): PbrProperties {
  if (!material) return {}
  const customFragmentShader = extractCustomFragmentShader(material)
  assertSupportedShaderMaterial(material, customFragmentShader)
  assertSupportedOnBeforeCompile(material, customFragmentShader)
  assertSupportedMaterialState(material)
  const props: PbrProperties = {}

  if (Number.isFinite(material.metalness)) {
    props.metallic = clamp01(material.metalness!)
  }
  if (Number.isFinite(material.roughness)) {
    props.roughness = clamp01(material.roughness!)
  }
  if (Number.isFinite(material.clearcoat)) {
    props.clearcoat = clamp01(material.clearcoat!)
  }
  const clearcoatMapInfo = extractTextureFromSlot(material.clearcoatMap)
  if (clearcoatMapInfo) {
    props.clearcoatMap = clearcoatMapInfo.data
    props.clearcoatMapWidth = clearcoatMapInfo.width
    props.clearcoatMapHeight = clearcoatMapInfo.height
    props.clearcoatMapWrapS = wrapModeToString(material.clearcoatMap?.wrapS)
    props.clearcoatMapWrapT = wrapModeToString(material.clearcoatMap?.wrapT)
    props.clearcoatMapMagFilter = filterModeToString(material.clearcoatMap?.magFilter)
    props.clearcoatMapMinFilter = filterModeToString(material.clearcoatMap?.minFilter)
    props.clearcoatMapTransform = textureTransform(material.clearcoatMap)
    props.clearcoatMapUsesUv2 = textureUvChannel(material.clearcoatMap) > 0
  }
  if (Number.isFinite(material.clearcoatRoughness)) {
    props.clearcoatRoughness = clamp01(material.clearcoatRoughness!)
  }
  const clearcoatRoughnessMapInfo = extractTextureFromSlot(material.clearcoatRoughnessMap)
  if (clearcoatRoughnessMapInfo) {
    props.clearcoatRoughnessMap = clearcoatRoughnessMapInfo.data
    props.clearcoatRoughnessMapWidth = clearcoatRoughnessMapInfo.width
    props.clearcoatRoughnessMapHeight = clearcoatRoughnessMapInfo.height
    props.clearcoatRoughnessMapWrapS = wrapModeToString(material.clearcoatRoughnessMap?.wrapS)
    props.clearcoatRoughnessMapWrapT = wrapModeToString(material.clearcoatRoughnessMap?.wrapT)
    props.clearcoatRoughnessMapMagFilter = filterModeToString(material.clearcoatRoughnessMap?.magFilter)
    props.clearcoatRoughnessMapMinFilter = filterModeToString(material.clearcoatRoughnessMap?.minFilter)
    props.clearcoatRoughnessMapTransform = textureTransform(material.clearcoatRoughnessMap)
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
    props.clearcoatNormalMapMinFilter = filterModeToString(material.clearcoatNormalMap?.minFilter)
    props.clearcoatNormalMapTransform = textureTransform(material.clearcoatNormalMap)
    props.clearcoatNormalMapUsesUv2 = textureUvChannel(material.clearcoatNormalMap) > 0
  }
  if (material.clearcoatNormalScale) {
    props.clearcoatNormalScale = [material.clearcoatNormalScale.x ?? 1, material.clearcoatNormalScale.y ?? 1]
  }

  const sheenColor = colorLikeToArray(material.sheenColor)
  const sheen = Number.isFinite(material.sheen) ? clamp01(material.sheen!) : 0
  if (sheenColor && sheen > 0) {
    props.sheenColor = [
      sheenColor[0] * sheen,
      sheenColor[1] * sheen,
      sheenColor[2] * sheen,
    ]
  }
  if (Number.isFinite(material.sheenRoughness)) {
    props.sheenRoughness = clamp01(material.sheenRoughness!)
  }
  const sheenColorMapInfo = extractTextureFromSlot(material.sheenColorMap)
  if (sheenColorMapInfo) {
    props.sheenColorMap = sheenColorMapInfo.data
    props.sheenColorMapWidth = sheenColorMapInfo.width
    props.sheenColorMapHeight = sheenColorMapInfo.height
    props.sheenColorMapWrapS = wrapModeToString(material.sheenColorMap?.wrapS)
    props.sheenColorMapWrapT = wrapModeToString(material.sheenColorMap?.wrapT)
    props.sheenColorMapMagFilter = filterModeToString(material.sheenColorMap?.magFilter)
    props.sheenColorMapMinFilter = filterModeToString(material.sheenColorMap?.minFilter)
    props.sheenColorMapTransform = textureTransform(material.sheenColorMap)
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
    props.sheenRoughnessMapMinFilter = filterModeToString(material.sheenRoughnessMap?.minFilter)
    props.sheenRoughnessMapTransform = textureTransform(material.sheenRoughnessMap)
    props.sheenRoughnessMapUsesUv2 = textureUvChannel(material.sheenRoughnessMap) > 0
  }

  if (Number.isFinite(material.anisotropy)) {
    props.anisotropy = clamp01(Math.abs(material.anisotropy!))
  }
  if (Number.isFinite(material.anisotropyRotation)) {
    props.anisotropyRotation = material.anisotropyRotation!
  }
  const anisotropyMapInfo = extractTextureFromSlot(material.anisotropyMap)
  if (anisotropyMapInfo) {
    props.anisotropyMap = anisotropyMapInfo.data
    props.anisotropyMapWidth = anisotropyMapInfo.width
    props.anisotropyMapHeight = anisotropyMapInfo.height
    props.anisotropyMapWrapS = wrapModeToString(material.anisotropyMap?.wrapS)
    props.anisotropyMapWrapT = wrapModeToString(material.anisotropyMap?.wrapT)
    props.anisotropyMapMagFilter = filterModeToString(material.anisotropyMap?.magFilter)
    props.anisotropyMapMinFilter = filterModeToString(material.anisotropyMap?.minFilter)
    props.anisotropyMapTransform = textureTransform(material.anisotropyMap)
    props.anisotropyMapUsesUv2 = textureUvChannel(material.anisotropyMap) > 0
  }

  if (Number.isFinite(material.transmission)) {
    props.transmission = clamp01(material.transmission!)
  }
  const transmissionMapInfo = extractTextureFromSlot(material.transmissionMap)
  if (transmissionMapInfo) {
    props.transmissionMap = transmissionMapInfo.data
    props.transmissionMapWidth = transmissionMapInfo.width
    props.transmissionMapHeight = transmissionMapInfo.height
    props.transmissionMapWrapS = wrapModeToString(material.transmissionMap?.wrapS)
    props.transmissionMapWrapT = wrapModeToString(material.transmissionMap?.wrapT)
    props.transmissionMapMagFilter = filterModeToString(material.transmissionMap?.magFilter)
    props.transmissionMapMinFilter = filterModeToString(material.transmissionMap?.minFilter)
    props.transmissionMapTransform = textureTransform(material.transmissionMap)
    props.transmissionMapUsesUv2 = textureUvChannel(material.transmissionMap) > 0
  }
  if (Number.isFinite(material.ior)) {
    props.ior = Math.max(1, Math.min(2.333, material.ior!))
  }
  if (Number.isFinite(material.thickness)) {
    props.thickness = Math.max(0, material.thickness!)
  }
  const thicknessMapInfo = extractTextureFromSlot(material.thicknessMap)
  if (thicknessMapInfo) {
    props.thicknessMap = thicknessMapInfo.data
    props.thicknessMapWidth = thicknessMapInfo.width
    props.thicknessMapHeight = thicknessMapInfo.height
    props.thicknessMapWrapS = wrapModeToString(material.thicknessMap?.wrapS)
    props.thicknessMapWrapT = wrapModeToString(material.thicknessMap?.wrapT)
    props.thicknessMapMagFilter = filterModeToString(material.thicknessMap?.magFilter)
    props.thicknessMapMinFilter = filterModeToString(material.thicknessMap?.minFilter)
    props.thicknessMapTransform = textureTransform(material.thicknessMap)
    props.thicknessMapUsesUv2 = textureUvChannel(material.thicknessMap) > 0
  }
  if (Number.isFinite(material.attenuationDistance)) {
    props.attenuationDistance = Math.max(0, material.attenuationDistance!)
  }
  const attenuationColor = colorLikeToArray(material.attenuationColor)
  if (attenuationColor) {
    props.attenuationColor = [attenuationColor[0], attenuationColor[1], attenuationColor[2]]
  }
  const physicalSpecularColor = colorLikeToArray(material.specularColor)
  if (physicalSpecularColor) {
    props.physicalSpecularColor = [
      physicalSpecularColor[0],
      physicalSpecularColor[1],
      physicalSpecularColor[2],
    ]
  }
  if (Number.isFinite(material.specularIntensity)) {
    props.physicalSpecularIntensity = clamp01(material.specularIntensity!)
  }
  const specularColorMapInfo = extractTextureFromSlot(material.specularColorMap)
  if (specularColorMapInfo) {
    props.specularColorMap = specularColorMapInfo.data
    props.specularColorMapWidth = specularColorMapInfo.width
    props.specularColorMapHeight = specularColorMapInfo.height
    props.specularColorMapWrapS = wrapModeToString(material.specularColorMap?.wrapS)
    props.specularColorMapWrapT = wrapModeToString(material.specularColorMap?.wrapT)
    props.specularColorMapMagFilter = filterModeToString(material.specularColorMap?.magFilter)
    props.specularColorMapMinFilter = filterModeToString(material.specularColorMap?.minFilter)
    props.specularColorMapTransform = textureTransform(material.specularColorMap)
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
    props.specularIntensityMapMinFilter = filterModeToString(material.specularIntensityMap?.minFilter)
    props.specularIntensityMapTransform = textureTransform(material.specularIntensityMap)
    props.specularIntensityMapUsesUv2 = textureUvChannel(material.specularIntensityMap) > 0
  }

  const specularColor = colorLikeToArray(material.specular)
  if (specularColor || material.isMeshPhongMaterial) {
    const color = specularColor ?? [17 / 255, 17 / 255, 17 / 255, 1]
    props.specularColor = [color[0], color[1], color[2]]
  }
  if (Number.isFinite(material.shininess) || material.isMeshPhongMaterial) {
    props.shininess = Math.max(0.0001, material.shininess ?? 30)
  }

  const emissive = colorLikeToArray(material.emissive)
  if (emissive) {
    props.emissive = [emissive[0], emissive[1], emissive[2]]
    props.emissiveIntensity = Number.isFinite(material.emissiveIntensity) ? material.emissiveIntensity! : 1
  }

  const normalMapInfo = extractTextureFromSlot(material.normalMap)
  if (normalMapInfo) {
    props.normalMap = normalMapInfo.data
    props.normalMapWidth = normalMapInfo.width
    props.normalMapHeight = normalMapInfo.height
    props.normalMapWrapS = wrapModeToString(material.normalMap?.wrapS)
    props.normalMapWrapT = wrapModeToString(material.normalMap?.wrapT)
    props.normalMapMagFilter = filterModeToString(material.normalMap?.magFilter)
    props.normalMapMinFilter = filterModeToString(material.normalMap?.minFilter)
    props.normalMapTransform = textureTransform(material.normalMap)
    props.normalMapUsesUv2 = textureUvChannel(material.normalMap) > 0
  }
  if (material.normalScale) {
    props.normalScale = [material.normalScale.x ?? 1, material.normalScale.y ?? 1]
  }
  const bumpMapInfo = extractTextureFromSlot(material.bumpMap)
  if (bumpMapInfo) {
    props.bumpMap = bumpMapInfo.data
    props.bumpMapWidth = bumpMapInfo.width
    props.bumpMapHeight = bumpMapInfo.height
    props.bumpMapWrapS = wrapModeToString(material.bumpMap?.wrapS)
    props.bumpMapWrapT = wrapModeToString(material.bumpMap?.wrapT)
    props.bumpMapMagFilter = filterModeToString(material.bumpMap?.magFilter)
    props.bumpMapMinFilter = filterModeToString(material.bumpMap?.minFilter)
    props.bumpMapTransform = textureTransform(material.bumpMap)
    props.bumpMapUsesUv2 = textureUvChannel(material.bumpMap) > 0
    props.bumpScale = Number.isFinite(material.bumpScale) ? material.bumpScale! : 1
  }
  if (material.isMeshMatcapMaterial) {
    const matcapMapInfo = extractTextureFromSlot(material.map)
    if (matcapMapInfo) {
      props.matcapMap = matcapMapInfo.data
      props.matcapMapWidth = matcapMapInfo.width
      props.matcapMapHeight = matcapMapInfo.height
      props.matcapMapTransform = textureTransform(material.map)
      props.matcapMapColorSpace = textureColorSpace(material.map)
      props.matcapMapUsesUv2 = textureUvChannel(material.map) > 0
    }
  }
  if (material.isMeshDepthMaterial) {
    const depthPacking = finiteIntegerOrDefault(material.depthPacking, BasicDepthPacking)
    props.depthPacking = [
      BasicDepthPacking,
      RGBADepthPacking,
      RGBDepthPacking,
      RGDepthPacking,
    ].includes(depthPacking) ? depthPacking : BasicDepthPacking
  }
  if (material.isMeshDistanceMaterial) {
    const hints = material.userData?.headlessThreeRenderer ?? material.userData?.headlessRenderer ?? {}
    const referencePosition = vector3LikeToArray(
      material.referencePosition ?? hints.referencePosition ?? hints.distanceReferencePosition,
    )
    if (referencePosition) {
      props.distanceReferencePosition = referencePosition
    }
    const nearDistance = finiteNumberOrUndefined(material.nearDistance ?? hints.nearDistance ?? hints.distanceNear)
    if (nearDistance !== undefined) {
      props.distanceNear = nearDistance
    }
    const farDistance = finiteNumberOrUndefined(material.farDistance ?? hints.farDistance ?? hints.distanceFar)
    if (farDistance !== undefined) {
      props.distanceFar = farDistance
    }
  }

  const gradientMapInfo = extractTextureFromSlot(material.gradientMap)
  if (gradientMapInfo) {
    props.gradientMap = gradientMapInfo.data
    props.gradientMapWidth = gradientMapInfo.width
    props.gradientMapHeight = gradientMapInfo.height
  }

  const displacementMapInfo = extractTextureFromSlot(material.displacementMap)
  if (displacementMapInfo) {
    props.displacementMap = displacementMapInfo.data
    props.displacementMapWidth = displacementMapInfo.width
    props.displacementMapHeight = displacementMapInfo.height
    props.displacementMapTransform = textureTransform(material.displacementMap)
    props.displacementMapUsesUv2 = textureUvChannel(material.displacementMap) > 0
    props.displacementScale = Number.isFinite(material.displacementScale) ? material.displacementScale! : 1
    props.displacementBias = Number.isFinite(material.displacementBias) ? material.displacementBias! : 0
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
    props.metallicRoughnessTextureMinFilter = filterModeToString(mrMap?.minFilter)
    props.metallicRoughnessTextureTransform = textureTransform(mrMap)
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
    props.specularMapMinFilter = filterModeToString(material.specularMap?.minFilter)
    props.specularMapTransform = textureTransform(material.specularMap)
  }

  const emissiveMapInfo = extractTextureFromSlot(material.emissiveMap)
  if (emissiveMapInfo) {
    props.emissiveMap = emissiveMapInfo.data
    props.emissiveMapWidth = emissiveMapInfo.width
    props.emissiveMapHeight = emissiveMapInfo.height
    props.emissiveMapWrapS = wrapModeToString(material.emissiveMap?.wrapS)
    props.emissiveMapWrapT = wrapModeToString(material.emissiveMap?.wrapT)
    props.emissiveMapMagFilter = filterModeToString(material.emissiveMap?.magFilter)
    props.emissiveMapMinFilter = filterModeToString(material.emissiveMap?.minFilter)
    props.emissiveMapTransform = textureTransform(material.emissiveMap)
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
    props.aoMapMinFilter = filterModeToString(material.aoMap?.minFilter)
    props.aoMapTransform = textureTransform(material.aoMap)
    props.aoMapIntensity = Number.isFinite(material.aoMapIntensity) ? material.aoMapIntensity! : 1
  }

  const lightMapInfo = extractTextureFromSlot(material.lightMap)
  if (lightMapInfo) {
    props.lightMap = lightMapInfo.data
    props.lightMapWidth = lightMapInfo.width
    props.lightMapHeight = lightMapInfo.height
    props.lightMapWrapS = wrapModeToString(material.lightMap?.wrapS)
    props.lightMapWrapT = wrapModeToString(material.lightMap?.wrapT)
    props.lightMapMagFilter = filterModeToString(material.lightMap?.magFilter)
    props.lightMapMinFilter = filterModeToString(material.lightMap?.minFilter)
    props.lightMapTransform = textureTransform(material.lightMap)
    props.lightMapColorSpace = textureColorSpace(material.lightMap)
    props.lightMapIntensity = Number.isFinite(material.lightMapIntensity) ? material.lightMapIntensity! : 1
  }

  const alphaMapInfo = extractTextureFromSlot(material.alphaMap)
  if (alphaMapInfo) {
    props.alphaMap = alphaMapInfo.data
    props.alphaMapWidth = alphaMapInfo.width
    props.alphaMapHeight = alphaMapInfo.height
    props.alphaMapWrapS = wrapModeToString(material.alphaMap?.wrapS)
    props.alphaMapWrapT = wrapModeToString(material.alphaMap?.wrapT)
    props.alphaMapMagFilter = filterModeToString(material.alphaMap?.magFilter)
    props.alphaMapMinFilter = filterModeToString(material.alphaMap?.minFilter)
    props.alphaMapTransform = textureTransform(material.alphaMap)
    props.alphaMapUsesUv2 = textureUvChannel(material.alphaMap) > 0
  }

  if (Number.isFinite(material.alphaTest) && material.alphaTest! > 0) {
    props.alphaTest = clamp01(material.alphaTest!)
  }
  if (material.alphaHash === true) {
    props.alphaHash = true
  }
  if (material.premultipliedAlpha === true) {
    props.premultipliedAlpha = true
  }
  if (typeof material.transparent === 'boolean') {
    props.transparent = material.transparent
  }
  const blending = materialBlending(material)
  if (blending) {
    props.blending = blending
    if (blending === 'custom') {
      props.blendEquation = finiteIntegerOrDefault(material.blendEquation, AddEquation)
      props.blendSrc = finiteIntegerOrDefault(material.blendSrc, SrcAlphaFactor)
      props.blendDst = finiteIntegerOrDefault(material.blendDst, OneMinusSrcAlphaFactor)
      if (Number.isFinite(material.blendEquationAlpha)) {
        props.blendEquationAlpha = material.blendEquationAlpha!
      }
      if (Number.isFinite(material.blendSrcAlpha)) {
        props.blendSrcAlpha = material.blendSrcAlpha!
      }
      if (Number.isFinite(material.blendDstAlpha)) {
        props.blendDstAlpha = material.blendDstAlpha!
      }
      const blendColor = colorLikeToArray(material.blendColor)
      if (blendColor) {
        props.blendColor = [blendColor[0], blendColor[1], blendColor[2]]
      }
      if (Number.isFinite(material.blendAlpha)) {
        props.blendAlpha = clamp01(material.blendAlpha!)
      }
    }
  }
  if (typeof material.depthTest === 'boolean') {
    props.depthTest = material.depthTest
  }
  if (typeof material.depthWrite === 'boolean') {
    props.depthWrite = material.depthWrite
  }
  if (typeof material.colorWrite === 'boolean') {
    props.colorWrite = material.colorWrite
  }
  if (typeof material.polygonOffset === 'boolean') {
    props.polygonOffset = material.polygonOffset
    if (Number.isFinite(material.polygonOffsetFactor)) {
      props.polygonOffsetFactor = material.polygonOffsetFactor!
    }
    if (Number.isFinite(material.polygonOffsetUnits)) {
      props.polygonOffsetUnits = material.polygonOffsetUnits!
    }
  }
  if (typeof material.stencilWrite === 'boolean') {
    props.stencilWrite = material.stencilWrite
  }
  if (Number.isFinite(material.stencilWriteMask)) {
    props.stencilWriteMask = finiteIntegerOrDefault(material.stencilWriteMask, 0xff)
  }
  if (Number.isFinite(material.stencilFunc)) {
    props.stencilFunc = finiteIntegerOrDefault(material.stencilFunc, AlwaysStencilFunc)
  }
  if (Number.isFinite(material.stencilRef)) {
    props.stencilRef = finiteIntegerOrDefault(material.stencilRef, 0)
  }
  if (Number.isFinite(material.stencilFuncMask)) {
    props.stencilFuncMask = finiteIntegerOrDefault(material.stencilFuncMask, 0xff)
  }
  if (Number.isFinite(material.stencilFail)) {
    props.stencilFail = finiteIntegerOrDefault(material.stencilFail, KeepStencilOp)
  }
  if (Number.isFinite(material.stencilZFail)) {
    props.stencilZFail = finiteIntegerOrDefault(material.stencilZFail, KeepStencilOp)
  }
  if (Number.isFinite(material.stencilZPass)) {
    props.stencilZPass = finiteIntegerOrDefault(material.stencilZPass, KeepStencilOp)
  }
  if (material.side === BackSide) {
    props.side = 'back'
  } else if (material.side === DoubleSide) {
    props.side = 'double'
  } else if (material.side === FrontSide) {
    props.side = 'front'
  }
  if (material.flatShading === true) {
    props.flatShading = true
  }
  if (material.fog === false) {
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
      return undefined
  }
}

function finiteIntegerOrDefault(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? Math.trunc(value) : fallback
}

function textureUvChannel(texture: ThreeTextureLike | null | undefined): number {
  return Number.isInteger(texture?.channel) ? Math.max(0, texture!.channel!) : 0
}

function finiteNumberOrUndefined(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
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

function assertSupportedMaterialState(material: ThreeMaterialLike): void {
  if (material.alphaToCoverage === true) {
    throw new Error(
      'material.alphaToCoverage is not supported by @headless-three/renderer yet. Disable alphaToCoverage or use alphaTest/alphaHash for explicit coverage behavior before rendering.',
    )
  }
  if (material.clipShadows === true) {
    throw new Error(
      'material.clipShadows is not supported by @headless-three/renderer yet. Shadow-pass clipping is not translated; disable clipShadows or pre-bake the clipped shadow caster before rendering.',
    )
  }
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
  const base = extractTextureFromSlot(slot)
  if (!base) return null

  const map = slot as ThreeTextureLike | null | undefined
  return {
    ...base,
    wrapS: material?.isMeshMatcapMaterial ? undefined : wrapModeToString(map?.wrapS),
    wrapT: material?.isMeshMatcapMaterial ? undefined : wrapModeToString(map?.wrapT),
    magFilter: filterModeToString(map?.magFilter),
    minFilter: filterModeToString(map?.minFilter),
    transform: material?.isMeshMatcapMaterial ? undefined : textureTransform(map),
    colorSpace: textureColorSpace(map),
    usesUv2: material?.isMeshMatcapMaterial ? false : textureUvChannel(map) > 0,
  }
}

export function extractBackgroundTexture(
  background: ThreeSceneLike['background'] | ThreeTextureLike | number[] | undefined,
  label = 'background',
): TextureInfo | null {
  const map = textureLike(background)
  if (!map) return null
  assertSupportedBackgroundTexture(map, label)

  const base = extractTextureFromSlot(map)
  if (!base) return null

  return {
    ...base,
    wrapS: wrapModeToString(map?.wrapS),
    wrapT: wrapModeToString(map?.wrapT),
    magFilter: filterModeToString(map?.magFilter),
    minFilter: filterModeToString(map?.minFilter),
    transform: textureTransform(map),
    colorSpace: textureColorSpace(map),
  }
}

function textureLike(value: unknown): ThreeTextureLike | null {
  if (!value || Array.isArray(value)) return null
  const candidate = value as ThreeTextureLike & { isTexture?: boolean }
  if (candidate.isTexture === true || candidate.image || candidate.source?.data) {
    return candidate
  }
  return null
}

function wrapModeToString(mode: number | undefined): string | undefined {
  if (mode === RepeatWrapping) return 'repeat'
  if (mode === MirroredRepeatWrapping) return 'mirror'
  return undefined // default = clamp
}

function filterModeToString(mode: number | undefined): string | undefined {
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
  return undefined // default = linear
}

function extractTextureFromSlot(map: ThreeMaterialLike['map']): TextureInfo | null {
  if (!map) return null
  assertUncompressedTexture(map, 'texture')

  const image = (map as any).image ?? (map as any).source?.data
  if (!image) return null

  // DataTexture style: { data: TypedArray, width, height }
  if (image.data && image.width > 0 && image.height > 0) {
    const rgba = toRgba8(image.data, image.width, image.height)
    if (rgba) {
      return { data: Buffer.from(rgba.buffer, rgba.byteOffset, rgba.byteLength), width: image.width, height: image.height }
    }
  }

  // Encoded image (PNG/JPEG/WebP Buffer from file loaders)
  if (Buffer.isBuffer(image)) {
    return { data: image, width: 0, height: 0 }
  }
  if (image instanceof Uint8Array && !((image as any).width > 0)) {
    return { data: Buffer.from(image.buffer, image.byteOffset, image.byteLength), width: 0, height: 0 }
  }

  // ImageData (canvas-based polyfill): { data: Uint8ClampedArray, width, height }
  if (image.data instanceof Uint8ClampedArray && image.width > 0 && image.height > 0) {
    return {
      data: Buffer.from(image.data.buffer, image.data.byteOffset, image.data.byteLength),
      width: image.width,
      height: image.height,
    }
  }

  return null
}

function assertSupportedBackgroundTexture(map: ThreeTextureLike, label: string): void {
  assertUncompressedTexture(map, label)
  if (
    map.isCubeTexture === true ||
    map.mapping === CubeReflectionMapping ||
    map.mapping === CubeRefractionMapping ||
    map.mapping === EquirectangularReflectionMapping ||
    map.mapping === EquirectangularRefractionMapping ||
    map.mapping === CubeUVReflectionMapping
  ) {
    throw new Error(
      `${label} uses a cube/equirectangular texture mapping, which is not supported as a background yet. Use a 2D UV-mapped texture or pre-render the background to a 2D image before rendering.`,
    )
  }
}

function assertUncompressedTexture(map: ThreeTextureLike, label: string): void {
  if (
    map.isCompressedTexture === true ||
    map.isCompressedArrayTexture === true ||
    map.isCompressedCubeTexture === true
  ) {
    throw new Error(
      `${label} uses a compressed texture. KTX2, Basis, and THREE.CompressedTexture inputs are not decoded by @headless-three/renderer yet; pre-decode the texture to RGBA data or an encoded PNG/JPEG/WebP image before rendering.`,
    )
  }
}

function textureTransform(map: ThreeTextureLike | null | undefined): number[] | undefined {
  const flipY = map?.flipY !== false
  const flipTransform = flipY ? undefined : [1, 0, 0, 0, -1, 1]
  if (!map) return flipTransform

  if (map.matrixAutoUpdate === false && map.matrix?.elements?.length === 9) {
    const e = map.matrix.elements
    if (areFiniteNumbers(e[0], e[1], e[3], e[4], e[6], e[7])) {
      return composeTextureTransformWithFlipY([e[0], e[3], e[6], e[1], e[4], e[7]], flipY)
    }
  }

  const tx = finiteOrDefault(map.offset?.x, 0)
  const ty = finiteOrDefault(map.offset?.y, 0)
  const sx = finiteOrDefault(map.repeat?.x, 1)
  const sy = finiteOrDefault(map.repeat?.y, 1)
  const rotation = finiteOrDefault(map.rotation, 0)
  const cx = finiteOrDefault(map.center?.x, 0)
  const cy = finiteOrDefault(map.center?.y, 0)
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

function textureColorSpace(map: ThreeTextureLike | null | undefined): string | undefined {
  if (!map) return undefined
  if (map.colorSpace === 'srgb' || map.encoding === sRGBEncoding) return 'srgb'
  return undefined
}

function finiteOrDefault(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function areFiniteNumbers(...values: number[]): boolean {
  for (let i = 0; i < values.length; i += 1) {
    if (!Number.isFinite(values[i])) return false
  }
  return true
}

function toRgba8(data: ArrayBufferView & { length: number }, width: number, height: number): Uint8Array | null {
  const pixels = width * height

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
    return null
  }

  // Uint16Array or other numeric typed arrays — treat as 8-bit range after clamping
  if (ArrayBuffer.isView(data) && data.length === pixels * 4) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels * 4; i++) {
      out[i] = Math.max(0, Math.min(255, (data as any)[i]))
    }
    return out
  }

  return null
}
