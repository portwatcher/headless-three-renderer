import type {
  ThreeSceneRootLike,
  ThreeCameraLike,
  ThreeCubeCameraLike,
  ThreeRenderCameraLike,
  RenderOptions,
  RenderTargetLike,
  RenderTargetTextureLike,
  RenderTargetImageLike,
  RenderPixelRectLike,
  RenderSizeLike,
  ThreeColorLike,
  NativeRenderScene,
  NativeCamera,
  NativeSceneMesh,
  NativeSceneLight,
  RenderMode,
  RenderOutputColorSpace,
  Color4,
  RenderObjectIdEntry,
  ThreeEulerLike,
  ThreePlaneLike,
  ThreeTextureLike,
  ThreeMaterialLike,
  ThreeObject3DLike,
  RenderSortFunction,
  RenderAnimationLoopCallback,
  RendererParametersLike,
  RendererContextAttributesLike,
  RendererInspectorLike,
} from './types'
import { resolveSize, cameraViewProjection, cameraViewMatrix, cameraWorldPosition } from './camera'
import { DEFAULT_BACKGROUND_COLOR, cssColorStringToArray, resolveBackground, validatedColorLikeToArray } from './color'
import { commitNativeMeshPayloadCache, createSceneExtractionCache, flattenScene, type SceneExtractionCache, type ShadowMaterialMode } from './scene'
import { extractLights, extractAmbientLight, extractAmbientIntensity, extractLightProbe } from './lights'
import { canvasLikeImageToRgba, extractBackgroundTexture, extractTextureData, isCompressedTextureFormat, resolveEnvironmentMap, resolveSceneOverrideMaterial, type MaterialExtractionContext } from './materials'
import { extractClippingPlanes } from './clipping'
import { validateObjectChildrenTree } from './objects'
import { clamp01, matrixElements } from './math'
import { AlphaFormat, ByteType, FloatType, HalfFloatType, IntType, LuminanceAlphaFormat, LuminanceFormat, PixelRect, RGBFormat, RGBIntegerFormat, RGFormat, RGIntegerFormat, RedFormat, RedIntegerFormat, ShortType, UnsignedInt101111Type, UnsignedInt5999Type, UnsignedIntType, UnsignedShort4444Type, UnsignedShort5551Type, UnsignedShortType } from './index.part-012'
import { assertActiveCubeFace } from './index.part-013'
import { assertRenderTargetLike } from './index.part-016'
import { validateUnsupportedRenderTargetOptions } from './index.part-017'
import { renderTargetColorTextures } from './index.part-019'
import { normalizedFloatToHalf } from './index.part-021'
export function renderTargetReadbackSource(
  target: RenderTargetLike,
  activeCubeFaceIndex: number | undefined,
  textureIndex: number,
  label: string,
  activeMipmapLevel = 0,
): { data: NonNullable<RenderTargetImageLike['data']>; width: number; height: number; channels: number } {
  assertRenderTargetLike(target, `${label} target`)
  validateUnsupportedRenderTargetOptions(target)
  if (!Number.isInteger(textureIndex) || textureIndex < 0) {
    throw new TypeError(`${label} textureIndex must be a non-negative integer.`)
  }
  if (activeCubeFaceIndex !== undefined) {
    assertActiveCubeFace(activeCubeFaceIndex, `${label} activeCubeFaceIndex`)
  }

  const texture = renderTargetColorTextures(target)[textureIndex]
  const image = renderTargetReadbackImage(target, texture, activeCubeFaceIndex, textureIndex, activeMipmapLevel)
  if (!image?.data) {
    throw new Error(
      `${label} target has no readable color data. Render into the target before reading pixels.`,
    )
  }

  const rawWidth = image.width ?? target.width
  const rawHeight = image.height ?? target.height
  if (
    typeof rawWidth !== 'number' ||
    !Number.isInteger(rawWidth) ||
    rawWidth <= 0 ||
    typeof rawHeight !== 'number' ||
    !Number.isInteger(rawHeight) ||
    rawHeight <= 0
  ) {
    throw new Error(`${label} target readable color data is missing valid width and height.`)
  }
  const width = rawWidth
  const height = rawHeight
  const channels = renderTargetReadbackChannelCount(image.data, width, height, label)
  return { data: image.data, width, height, channels }
}

export function renderTargetReadbackImage(
  target: RenderTargetLike,
  texture: RenderTargetTextureLike | undefined,
  activeCubeFaceIndex: number | undefined,
  textureIndex: number,
  activeMipmapLevel = 0,
): RenderTargetImageLike | undefined {
  if (activeCubeFaceIndex !== undefined) {
    return renderTargetTextureFaceImage(texture, activeCubeFaceIndex, activeMipmapLevel)
  }
  if (texture) {
    const image = Array.isArray(texture.image) ? texture.image[0] : texture.image
    if (image?.data) return image
    const sourceData = texture.source?.data
    const sourceImage = Array.isArray(sourceData) ? sourceData[0] : sourceData
    if (sourceImage?.data) return sourceImage
  }
  if (textureIndex === 0) {
    if (target.image?.data) return target.image
    if (target.data) return {
      data: target.data,
      width: target.width,
      height: target.height,
    }
  }
  return undefined
}

export function renderTargetTextureFaceImage(
  texture: RenderTargetTextureLike | undefined,
  activeCubeFaceIndex: number,
  activeMipmapLevel = 0,
): RenderTargetImageLike | undefined {
  if (!texture) return undefined
  if (activeMipmapLevel > 0) {
    const mipmapImage = texture.mipmaps?.[activeMipmapLevel]?.image
    if (Array.isArray(mipmapImage) && mipmapImage[activeCubeFaceIndex]?.data) {
      return mipmapImage[activeCubeFaceIndex]
    }
    return undefined
  }
  if (Array.isArray(texture.image) && texture.image[activeCubeFaceIndex]?.data) {
    return texture.image[activeCubeFaceIndex]
  }
  const sourceData = texture.source?.data
  if (Array.isArray(sourceData) && sourceData[activeCubeFaceIndex]?.data) {
    return sourceData[activeCubeFaceIndex]
  }
  return undefined
}

export function renderTargetReadbackChannelCount(
  data: NonNullable<RenderTargetImageLike['data']>,
  width: number,
  height: number,
  label: string,
): number {
  const pixelCount = width * height
  if (data.length === 0 || data.length % pixelCount !== 0) {
    throw new Error(`${label} target readable color data length does not match its width and height.`)
  }
  return data.length / pixelCount
}

export function copyRenderTargetReadbackPixels(
  readback: { data: NonNullable<RenderTargetImageLike['data']>; width: number; height: number; channels: number },
  x: number,
  y: number,
  width: number,
  height: number,
  buffer: NonNullable<RenderTargetImageLike['data']>,
  label: string,
): void {
  const rect = readbackRect(x, y, width, height, label)
  if (rect.x + rect.width > readback.width || rect.y + rect.height > readback.height) {
    throw new Error(`${label} requested read bounds are out of range.`)
  }
  assertRenderTargetReadbackBuffer(buffer, rect.width * rect.height * readback.channels, label)
  for (let row = 0; row < rect.height; row += 1) {
    const sourceStart = ((rect.y + row) * readback.width + rect.x) * readback.channels
    const sourceEnd = sourceStart + rect.width * readback.channels
    const targetStart = row * rect.width * readback.channels
    buffer.set(readback.data.subarray(sourceStart, sourceEnd) as any, targetStart)
  }
}

export function readbackRect(x: unknown, y: unknown, width: unknown, height: unknown, label: string): PixelRect {
  const values = [x, y, width, height]
  if (!values.every((value) => typeof value === 'number' && Number.isFinite(value))) {
    throw new TypeError(`${label} x, y, width, and height must be finite numbers.`)
  }
  if (!values.every((value) => Number.isInteger(value))) {
    throw new TypeError(`${label} x, y, width, and height must be integers.`)
  }
  if ((x as number) < 0 || (y as number) < 0) {
    throw new TypeError(`${label} x and y must be greater than or equal to 0.`)
  }
  if ((width as number) <= 0 || (height as number) <= 0) {
    throw new TypeError(`${label} width and height must be greater than 0.`)
  }
  return { x: x as number, y: y as number, width: width as number, height: height as number }
}

export function assertRenderTargetReadbackBuffer(
  buffer: unknown,
  minimumLength: number,
  label: string,
): asserts buffer is NonNullable<RenderTargetImageLike['data']> {
  const candidate = buffer as Partial<NonNullable<RenderTargetImageLike['data']>>
  if (!candidate || typeof candidate.length !== 'number' || typeof candidate.set !== 'function') {
    throw new TypeError(`${label} buffer must be a mutable typed array or Buffer.`)
  }
  if (candidate.length < minimumLength) {
    throw new Error(`${label} buffer length is too small for the requested read.`)
  }
}

export function createRenderTargetReadbackBuffer(
  source: NonNullable<RenderTargetImageLike['data']>,
  length: number,
): NonNullable<RenderTargetImageLike['data']> {
  if (Buffer.isBuffer(source)) return Buffer.alloc(length)
  const TypedArrayConstructor = source.constructor as new (length: number) => Exclude<NonNullable<RenderTargetImageLike['data']>, Buffer>
  return new TypedArrayConstructor(length)
}

export function writeRenderTargetTexture(
  texture: RenderTargetTextureLike,
  data: NonNullable<RenderTargetImageLike['data']>,
  width: number,
  height: number,
): void {
  const imageWasArray = Array.isArray(texture.image)
  const textureImage = Array.isArray(texture.image)
    ? texture.image[0] ?? (texture.image[0] = {})
    : texture.image ?? (texture.image = {})
  textureImage.data = data
  textureImage.width = width
  textureImage.height = height

  const source = texture.source ?? (texture.source = {})
  source.data ??= imageWasArray ? texture.image as RenderTargetImageLike[] : textureImage
  const sourceData = Array.isArray(source.data)
    ? source.data[0] ?? (source.data[0] = {})
    : source.data
  sourceData.data = data
  sourceData.width = width
  sourceData.height = height
  texture.needsUpdate = true
}

export function colorTextureData(texture: RenderTargetTextureLike, rgba: Buffer): NonNullable<RenderTargetImageLike['data']> {
  const channels = colorTextureChannelCount(texture.format)
  const values = colorTextureBytes(rgba, texture.format, channels)
  if (texture.type === FloatType) {
    const color = new Float32Array(values.length)
    for (let i = 0; i < values.length; i += 1) {
      color[i] = values[i] / 255
    }
    return color
  }
  if (texture.type === ByteType) {
    const color = new Int8Array(values.length)
    for (let i = 0; i < values.length; i += 1) {
      color[i] = normalizedByteToSignedInteger(values[i], 0x7f)
    }
    return color
  }
  if (texture.type === ShortType) {
    const color = new Int16Array(values.length)
    for (let i = 0; i < values.length; i += 1) {
      color[i] = normalizedByteToSignedInteger(values[i], 0x7fff)
    }
    return color
  }
  if (texture.type === UnsignedShortType) {
    const color = new Uint16Array(values.length)
    for (let i = 0; i < values.length; i += 1) {
      color[i] = Math.round((values[i] / 255) * 0xffff)
    }
    return color
  }
  if (texture.type === IntType) {
    const color = new Int32Array(values.length)
    for (let i = 0; i < values.length; i += 1) {
      color[i] = normalizedByteToSignedInteger(values[i], 0x7fffffff)
    }
    return color
  }
  if (texture.type === UnsignedIntType) {
    const color = new Uint32Array(values.length)
    for (let i = 0; i < values.length; i += 1) {
      color[i] = Math.round((values[i] / 255) * 0xffffffff)
    }
    return color
  }
  if (texture.type === HalfFloatType) {
    const color = new Uint16Array(values.length)
    for (let i = 0; i < values.length; i += 1) {
      color[i] = normalizedFloatToHalf(values[i] / 255)
    }
    return color
  }
  if (texture.type === UnsignedShort4444Type) {
    return packedUnsignedShort4444ColorTextureData(values, channels)
  }
  if (texture.type === UnsignedShort5551Type) {
    return packedUnsignedShort5551ColorTextureData(values, channels)
  }
  if (texture.type === UnsignedInt5999Type) {
    return packedUnsignedInt5999ColorTextureData(values, channels)
  }
  if (texture.type === UnsignedInt101111Type) {
    return packedUnsignedInt101111ColorTextureData(values, channels)
  }
  return values
}

export function normalizedByteToSignedInteger(value: number, max: number): number {
  return Math.round((value / 255) * max)
}

export function normalizedByteToUnsignedInteger(value: number, max: number): number {
  return Math.round((value / 255) * max)
}

export function packedUnsignedShort4444ColorTextureData(values: Uint8Array, channels: 1 | 2 | 3 | 4): Uint16Array {
  const out = new Uint16Array(values.length / channels)
  for (let src = 0, pixel = 0; src < values.length; src += channels, pixel += 1) {
    const r = normalizedByteToUnsignedInteger(values[src], 0xf)
    const g = normalizedByteToUnsignedInteger(channels > 1 ? values[src + 1] : 0, 0xf)
    const b = normalizedByteToUnsignedInteger(channels > 2 ? values[src + 2] : 0, 0xf)
    const a = normalizedByteToUnsignedInteger(channels > 3 ? values[src + 3] : 255, 0xf)
    out[pixel] = (r << 12) | (g << 8) | (b << 4) | a
  }
  return out
}

export function packedUnsignedShort5551ColorTextureData(values: Uint8Array, channels: 1 | 2 | 3 | 4): Uint16Array {
  const out = new Uint16Array(values.length / channels)
  for (let src = 0, pixel = 0; src < values.length; src += channels, pixel += 1) {
    const r = normalizedByteToUnsignedInteger(values[src], 0x1f)
    const g = normalizedByteToUnsignedInteger(channels > 1 ? values[src + 1] : 0, 0x1f)
    const b = normalizedByteToUnsignedInteger(channels > 2 ? values[src + 2] : 0, 0x1f)
    const a = (channels > 3 ? values[src + 3] : 255) >= 128 ? 1 : 0
    out[pixel] = (r << 11) | (g << 6) | (b << 1) | a
  }
  return out
}

export function packedUnsignedInt5999ColorTextureData(values: Uint8Array, channels: 1 | 2 | 3 | 4): Uint32Array {
  const out = new Uint32Array(values.length / channels)
  for (let src = 0, pixel = 0; src < values.length; src += channels, pixel += 1) {
    const r = values[src] / 255
    const g = channels > 1 ? values[src + 1] / 255 : 0
    const b = channels > 2 ? values[src + 2] / 255 : 0
    out[pixel] = packRgb9E5(r, g, b)
  }
  return out
}

export function packedUnsignedInt101111ColorTextureData(values: Uint8Array, channels: 1 | 2 | 3 | 4): Uint32Array {
  const out = new Uint32Array(values.length / channels)
  for (let src = 0, pixel = 0; src < values.length; src += channels, pixel += 1) {
    const r = values[src] / 255
    const g = channels > 1 ? values[src + 1] / 255 : 0
    const b = channels > 2 ? values[src + 2] / 255 : 0
    out[pixel] = packR11G11B10F(r, g, b)
  }
  return out
}

export function packRgb9E5(r: number, g: number, b: number): number {
  const maxChannel = Math.max(r, g, b)
  if (maxChannel <= 0) return 0

  let exponent = Math.max(0, Math.min(31, Math.floor(Math.log2(maxChannel)) + 16))
  let scale = 2 ** (24 - exponent)
  let rm = Math.round(r * scale)
  let gm = Math.round(g * scale)
  let bm = Math.round(b * scale)

  if (rm > 0x1ff || gm > 0x1ff || bm > 0x1ff) {
    exponent = Math.min(31, exponent + 1)
    scale = 2 ** (24 - exponent)
    rm = Math.round(r * scale)
    gm = Math.round(g * scale)
    bm = Math.round(b * scale)
  }

  return (
    (exponent << 27) |
    ((Math.min(0x1ff, bm) & 0x1ff) << 18) |
    ((Math.min(0x1ff, gm) & 0x1ff) << 9) |
    (Math.min(0x1ff, rm) & 0x1ff)
  ) >>> 0
}

export function packR11G11B10F(r: number, g: number, b: number): number {
  return (
    (packUnsignedFloat(b, 5) << 22) |
    (packUnsignedFloat(g, 6) << 11) |
    packUnsignedFloat(r, 6)
  ) >>> 0
}

export function packUnsignedFloat(value: number, mantissaBits: 5 | 6): number {
  const clamped = Math.max(0, value)
  if (clamped <= 0) return 0

  const mantissaScale = 2 ** mantissaBits
  const minNormal = 2 ** -14
  if (clamped < minNormal) {
    return Math.min((1 << mantissaBits) - 1, Math.round((clamped / minNormal) * mantissaScale))
  }

  let exponent = Math.max(0, Math.min(31, Math.floor(Math.log2(clamped)) + 15))
  let mantissa = Math.round(((clamped / (2 ** (exponent - 15))) - 1) * mantissaScale)
  if (mantissa >= mantissaScale) {
    exponent = Math.min(31, exponent + 1)
    mantissa = 0
  }
  if (exponent >= 31) return 0x1f << mantissaBits
  return (exponent << mantissaBits) | (mantissa & ((1 << mantissaBits) - 1))
}

export function colorTextureChannelCount(format: number | undefined): 1 | 2 | 3 | 4 {
  switch (format) {
    case AlphaFormat:
    case LuminanceFormat:
    case RedFormat:
    case RedIntegerFormat:
      return 1
    case LuminanceAlphaFormat:
    case RGFormat:
    case RGIntegerFormat:
      return 2
    case RGBFormat:
    case RGBIntegerFormat:
      return 3
    default:
      return 4
  }
}

export function colorTextureBytes(rgba: Buffer, format: number | undefined, channels: 1 | 2 | 3 | 4): Uint8Array {
  if (format === AlphaFormat) return alphaColorTextureBytes(rgba)
  if (format === LuminanceAlphaFormat) return luminanceAlphaColorTextureBytes(rgba)
  return channels === 4 ? rgba : narrowedColorTextureBytes(rgba, channels)
}

export function alphaColorTextureBytes(rgba: Buffer): Uint8Array {
  const pixels = rgba.length / 4
  const out = new Uint8Array(pixels)
  for (let i = 0, p = 0; i < rgba.length; i += 4, p += 1) {
    out[p] = rgba[i + 3]
  }
  return out
}

export function narrowedColorTextureBytes(rgba: Buffer, channels: 1 | 2 | 3): Uint8Array {
  const pixels = rgba.length / 4
  const out = new Uint8Array(pixels * channels)
  for (let i = 0, p = 0; i < rgba.length; i += 4, p += channels) {
    out[p] = rgba[i]
    if (channels > 1) out[p + 1] = rgba[i + 1]
    if (channels > 2) out[p + 2] = rgba[i + 2]
  }
  return out
}

export function luminanceAlphaColorTextureBytes(rgba: Buffer): Uint8Array {
  const pixels = rgba.length / 4
  const out = new Uint8Array(pixels * 2)
  for (let i = 0, p = 0; i < rgba.length; i += 4, p += 2) {
    out[p] = rgba[i]
    out[p + 1] = rgba[i + 3]
  }
  return out
}
