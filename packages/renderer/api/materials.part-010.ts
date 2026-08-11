import type { Color4, ThreeMaterialLike, PbrProperties, TextureInfo, ThreeTextureLike, ThreeSceneRootLike, ThreeObject3DLike } from './types'
import { clamp01 } from './math'
import { strictColorLikeToArray, validatedColorLikeToArray } from './color'
import { objectChildren } from './objects'
import { AlphaFormat, ByteType, CubeReflectionMapping, CubeRefractionMapping, CubeUVReflectionMapping, EquirectangularRefractionMapping, FloatType, HalfFloatType, IntType, LuminanceAlphaFormat, ShortType, UnsignedByteType, UnsignedInt101111Type, UnsignedInt248Type, UnsignedInt5999Type, UnsignedIntType, UnsignedShort4444Type, UnsignedShort5551Type, UnsignedShortType, isCompressedTextureFormat } from './materials.part-001'
import { isCubeEnvironmentTexture } from './materials.part-008'
import { TextureDataSignature, TextureMipmapSignature, TexturePayloadSignature, premultiplyFloatRgba, premultiplyHalfFloatRgba, premultiplyRgbaAlpha } from './materials.part-009'
import { optionalTextureBoolean, toRgba8 } from './materials.part-011'
export function texturePayloadSignature(
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

export function textureImageDataSignature(image: unknown): TextureDataSignature | undefined {
  if (Buffer.isBuffer(image) || image instanceof Uint8Array) {
    return textureDataSignature(image)
  }
  if (!image || typeof image !== 'object') return undefined
  return textureDataSignature((image as { data?: unknown }).data)
}

export function textureDataSignature(data: unknown): TextureDataSignature | undefined {
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

export function textureImageDimension(image: unknown, key: 'width' | 'height'): unknown {
  if (!image || typeof image !== 'object') return undefined
  return (image as Record<'width' | 'height', unknown>)[key]
}

export function textureMipmapSignatures(map: ThreeTextureLike, label: string): TextureMipmapSignature[] {
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

export function texturePayloadSignaturesEqual(a: TexturePayloadSignature, b: TexturePayloadSignature): boolean {
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

export function textureMipmapSignaturesEqual(a: TextureMipmapSignature[], b: TextureMipmapSignature[]): boolean {
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

export function textureDataSignaturesEqual(a: TextureDataSignature | undefined, b: TextureDataSignature | undefined): boolean {
  if (a === b) return true
  if (!a || !b) return false
  return a.data === b.data
    && a.length === b.length
    && a.buffer === b.buffer
    && a.byteOffset === b.byteOffset
    && a.byteLength === b.byteLength
}

export function hasExplicitMipmaps(texture: ThreeTextureLike | null | undefined, label = 'texture'): boolean {
  const mipmaps = texture?.mipmaps
  if (mipmaps == null) return false
  if (!Array.isArray(mipmaps)) {
    throw new TypeError(`${label}.mipmaps must be an array of image-like mip levels.`)
  }
  return mipmaps.length > 0
}

export function assertNoEncodedExplicitMipmaps(map: ThreeTextureLike, label: string): void {
  if (!hasExplicitMipmaps(map, label)) return
  throw new Error(
    `${label} provides explicit texture mipmaps with an encoded base image. Explicit mipmap upload requires raw DataTexture-style base image data with raw mipmap levels.`,
  )
}

export function assertNoEncodedPremultiplyAlpha(map: ThreeTextureLike, label: string): void {
  if (optionalTextureBoolean(map.premultiplyAlpha, `${label}.premultiplyAlpha`) !== true) return
  throw new Error(
    `${label}.premultiplyAlpha is only supported for readable raw texture image data. Decode the encoded image to raw RGBA DataTexture-style data before rendering.`,
  )
}

export function textureBytesWithExplicitMipmaps(
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

export function unsupportedRawTextureDataError(label: string, usage: string): Error {
  const supported = 'one-channel, two-channel, RGB, or RGBA numeric pixel data'
  const expected = 'mismatched data lengths must match width * height texels, width * height * 2 values, width * height * 3 values, or width * height * 4 values; packed color types use one value per texel'
  return new Error(
    `${label} raw texture data must contain ${supported} for ${usage}; ${expected}.`,
  )
}

export function unsupportedTextureImageError(label: string, usage: string): Error {
  return new Error(
    `${label} uses a texture image object that is not readable or drawable by @headless-three/renderer for ${usage}. Provide encoded PNG/JPEG/WebP bytes directly as texture.image or texture.source.data, a canvas-like object with getContext("2d").getImageData(), an image-like object drawable through an available OffscreenCanvas/2D canvas polyfill, or raw one-channel, two-channel, RGB, or RGBA numeric pixel data as { data, width, height } before rendering.`,
  )
}

export function assertSupportedRawTextureType(type: unknown, label: string, usage: string): void {
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

export function textureTypeName(type: unknown): string {
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

export function rawTextureChannelCount(
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

export function rawHalfFloatTextureDataToRgba(
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

export function rawFloatTextureDataToRgba(
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

export function writeOneChannelRawRgba(
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

export function writeTwoChannelRawRgba(
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

export function assertSupportedBackgroundTexture(map: ThreeTextureLike, label: string): void {
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

export function assertSupportedTwoDimensionalTextureSlot(map: ThreeTextureLike, label: string): void {
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

export function isRefractionEnvironmentMapping(mapping: number | undefined): boolean {
  return mapping === CubeRefractionMapping || mapping === EquirectangularRefractionMapping
}

export function assertSupportedEnvironmentTexture(
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

export function assertSupportedTextureInput(
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

export interface PhysicalMapFeatureGates {
  clearcoat: boolean
  sheen: boolean
  anisotropy: boolean
  iridescence: boolean
  transmission: boolean
}
