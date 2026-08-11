import type { Color4, ThreeMaterialLike, PbrProperties, TextureInfo, ThreeTextureLike, ThreeSceneRootLike, ThreeObject3DLike } from './types'
import { clamp01 } from './math'
import { strictColorLikeToArray, validatedColorLikeToArray } from './color'
import { objectChildren } from './objects'
import { AddOperation, EnvironmentMapInfo, FloatType, HalfFloatType, MaterialExtractionContext, MixOperation, MultiplyOperation, UnsignedByteType } from './materials.part-001'
import { optionalFiniteNumber } from './materials.part-005'
import { copyShaderMaterialInfo, cubeTextureToEquirectangular, isCubeEnvironmentTexture } from './materials.part-008'
import { canvasLikeImageToRgba, premultiplyRgbaAlpha, requiredEnvironmentTexture, textureLike, textureSourceData } from './materials.part-009'
import { assertNoEncodedPremultiplyAlpha, assertSupportedEnvironmentTexture, assertSupportedRawTextureType, isRefractionEnvironmentMapping, rawFloatTextureDataToRgba, rawHalfFloatTextureDataToRgba, unsupportedRawTextureDataError, unsupportedTextureImageError } from './materials.part-010'
import { optionalTextureBoolean, textureColorSpace, textureUnpackAlignment, toRgba8 } from './materials.part-011'
export function supportsNativeMaterialEnvironmentMap(material: ThreeMaterialLike): boolean {
  return material.isMeshStandardMaterial === true
    || material.isMeshPhysicalMaterial === true
    || material.isMeshBasicMaterial === true
    || material.isMeshPhongMaterial === true
    || material.isMeshLambertMaterial === true
}

export function supportsLegacyMaterialEnvironmentRefraction(material: ThreeMaterialLike): boolean {
  return material.isMeshBasicMaterial === true
    || material.isMeshPhongMaterial === true
    || material.isMeshLambertMaterial === true
}

export function assertSupportedMaterialEnvironmentMap(material: ThreeMaterialLike): void {
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

export function extractEnvironmentMapFromTexture(
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

export function extractReflectionProbe(scene: ThreeSceneRootLike): { texture: ThreeTextureLike; intensity?: unknown; label: string } | null {
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

export function sceneRendererHints(scene: ThreeSceneRootLike): { value: Record<string, unknown>; label: string } | undefined {
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

export interface CachedMaterialColorExtraction {
  signature: MaterialColorSignature
  color: Color4
}

export interface MaterialColorCacheEntry {
  base?: CachedMaterialColorExtraction
  slots?: Map<string, CachedMaterialColorExtraction>
}

export interface MaterialColorSignature {
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

export function materialColorSignature(material: ThreeMaterialLike): MaterialColorSignature {
  const copyShader = copyShaderMaterialInfo(material)
  return materialSlotColorSignature(material.color, copyShader ? copyShader.opacity : material.opacity)
}

export function materialSlotColor(
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

export function materialColorCacheEntry(
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

export function materialSlotColorSignature(color: unknown, opacity?: unknown): MaterialColorSignature {
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

export function copyColor(color: Color4): Color4 {
  return color.slice() as Color4
}

export function sameMaterialColorSignature(a: MaterialColorSignature, b: MaterialColorSignature): boolean {
  return a.color === b.color
    && a.colorLength === b.colorLength
    && a.r === b.r
    && a.g === b.g
    && a.b === b.b
    && a.a === b.a
    && a.opacity === b.opacity
    && sameUnknownArray(a.values, b.values)
}

export function sameUnknownArray(a: unknown[] | undefined, b: unknown[] | undefined): boolean {
  if (a === b) return true
  if (!a || !b) return false
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i += 1) {
    if (!Object.is(a[i], b[i])) return false
  }
  return true
}
