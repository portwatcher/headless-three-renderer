import type { Color4, ThreeMaterialLike, PbrProperties, TextureInfo, ThreeTextureLike, ThreeSceneRootLike, ThreeObject3DLike } from './types'
import { clamp01 } from './math'
import { strictColorLikeToArray, validatedColorLikeToArray } from './color'
import { objectChildren } from './objects'
import { ClampToEdgeWrapping, CubeUVReflectionMapping, LinearFilter, LinearMipmapLinearFilter, LinearMipmapNearestFilter, MirroredRepeatWrapping, NearestFilter, NearestMipmapLinearFilter, NearestMipmapNearestFilter, RepeatWrapping, TextureExtractionCache, TextureImageInput, native } from './materials.part-001'
import { assertNoEncodedExplicitMipmaps, assertNoEncodedPremultiplyAlpha, assertSupportedRawTextureType, assertSupportedTextureInput, assertSupportedTwoDimensionalTextureSlot, hasExplicitMipmaps, textureBytesWithExplicitMipmaps, texturePayloadSignature, texturePayloadSignaturesEqual, unsupportedRawTextureDataError, unsupportedTextureImageError } from './materials.part-010'
import { optionalTextureBoolean, textureUnpackAlignment, toRgba8 } from './materials.part-011'
import { halfFloatToNumber, numberToHalfFloat } from './materials.part-012'
export function cubeUvPackedImage(map: ThreeTextureLike, label = 'texture'): TextureImageInput | null {
  if (map.mapping !== CubeUVReflectionMapping) return null
  const sourceData = textureSourceData(map, label)
  const image = (map as any).image ?? sourceData
  if (!image || Array.isArray(image)) return null
  if (Buffer.isBuffer(image) || image instanceof Uint8Array) return image
  if (typeof image === 'object') return image as TextureImageInput
  return null
}

export function packedCubeUvTextureToFaceTextures(
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

export function extractRgbaTile(
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

export function imageToRgbaTexture(
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

export function drawImageLikeToRgba(
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

export function canvasLike2dContext(
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

export function canvasLikeContextToRgba(
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

export function canvasLikeImageDimension(value: unknown, label: string): number {
  if (!Number.isInteger(value) || (value as number) <= 0) {
    throw new TypeError(`${label} must be a positive integer for canvas-like texture image reads.`)
  }
  return value as number
}

export function premultiplyRgbaAlpha(data: Uint8Array | Uint8ClampedArray): Uint8Array {
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

export function premultiplyFloatRgba(data: Float32Array): Float32Array {
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

export function premultiplyHalfFloatRgba(data: Uint16Array): Uint16Array {
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

export function sampleCubeFace(
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

export function textureLike(value: unknown): ThreeTextureLike | null {
  if (!value || Array.isArray(value)) return null
  const candidate = value as ThreeTextureLike & { isTexture?: boolean }
  if (candidate.isTexture === true || candidate.image || candidate.source?.data) {
    return candidate
  }
  return null
}

export function textureSourceData(texture: ThreeTextureLike, label: string): unknown {
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

export function requiredEnvironmentTexture(value: unknown, label: string): ThreeTextureLike {
  const texture = textureLike(value)
  if (texture) return texture
  throw new TypeError(
    `${label} must be a Three.js texture or null for environment map rendering.`,
  )
}

export function wrapModeToString(mode: unknown): string | undefined {
  if (mode == null || mode === ClampToEdgeWrapping) return undefined // default = clamp
  if (mode === RepeatWrapping) return 'repeat'
  if (mode === MirroredRepeatWrapping) return 'mirror'
  throw new Error(
    `texture wrap mode ${String(mode)} is not supported by @headless-three/renderer. Use ClampToEdgeWrapping, RepeatWrapping, or MirroredRepeatWrapping.`,
  )
}

export function filterModeToString(mode: unknown): string | undefined {
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

export function minFilterModeToString(texture: ThreeTextureLike | null | undefined): string | undefined {
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

export interface CachedTextureExtraction {
  signature: TexturePayloadSignature
  info: TextureInfo
}

export interface TexturePayloadSignature {
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

export interface TextureMipmapSignature {
  image: unknown
  data?: TextureDataSignature
  width?: unknown
  height?: unknown
}

export interface TextureDataSignature {
  data: unknown
  length?: unknown
  buffer?: ArrayBufferLike
  byteOffset?: number
  byteLength?: number
}

export function extractTextureFromSlot(
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
