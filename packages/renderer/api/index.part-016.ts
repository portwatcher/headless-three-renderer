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
import { RendererInspectorOptionalMethods, SupportedTimestampQueryTypes } from './index.part-001'
import { textureCopyFlooredInteger, textureCopyInteger, textureCopyPositiveFlooredInteger, textureCopyPositiveInteger } from './index.part-017'
export function assertRenderTargetLike(value: unknown, label: string): asserts value is RenderTargetLike {
  if (value == null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be a target-like object.`)
  }
}

export function assertThreeTextureLike(value: unknown, label: string): asserts value is ThreeTextureLike {
  if (value == null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be a texture-like object.`)
  }
}

export function hasThreeTextureMarker(value: unknown): value is ThreeTextureLike {
  return (
    value !== null
    && typeof value === 'object'
    && !Array.isArray(value)
    && (value as { isTexture?: unknown }).isTexture === true
  )
}

export function isThreeTextureArgument(value: unknown): value is ThreeTextureLike {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return false
  const texture = value as Record<string, unknown>
  return (
    texture.isTexture === true
    || 'image' in texture
    || 'source' in texture
    || 'mipmaps' in texture
    || 'format' in texture
    || 'type' in texture
    || 'needsUpdate' in texture
    || 'colorSpace' in texture
    || texture.isFramebufferTexture === true
    || texture.isDepthTexture === true
    || texture.isVideoTexture === true
    || texture.isStorageTexture === true
    || texture.isCompressedTexture === true
    || texture.isDataArrayTexture === true
    || texture.isData3DTexture === true
    || texture.isArrayTexture === true
    || texture.is3DTexture === true
  )
}

export function assertCanvasTargetLike(value: unknown, label: string): void {
  if (value == null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be a canvas-target-like object.`)
  }
}

export function assertRendererInspectorLike(value: unknown, label: string): asserts value is RendererInspectorLike {
  if (value == null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be an inspector-like object.`)
  }
  const inspector = value as Record<string, unknown>
  if (typeof inspector.setRenderer !== 'function') {
    throw new TypeError(`${label}.setRenderer must be a function.`)
  }
  for (const method of RendererInspectorOptionalMethods) {
    if (inspector[method] !== undefined && typeof inspector[method] !== 'function') {
      throw new TypeError(`${label}.${method} must be a function when provided.`)
    }
  }
}

export function assertTextureBindingSlot(value: unknown, label: string): asserts value is number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
    throw new TypeError(`${label} must be a non-negative integer.`)
  }
}

export function unsupportedTextureBindingError(method: string): Error {
  return new Error(
    `${method}() is not supported by @headless-three/renderer because it does not expose browser WebGL texture units or direct texture binding. Use material, background, environment, or render-target texture inputs instead.`,
  )
}

export function unsupportedInternalRenderDispatchError(method: string): Error {
  return new Error(
    `${method}() is not supported by @headless-three/renderer because CommonRenderer internal render pipeline dispatch depends on backend render contexts, render lists, nodes, pipelines, and bindings that are outside the scene-oriented API. Render normal Three.js scene graphs with Renderer.render() or renderToTarget().`,
  )
}

export function unsupportedNodeOperationError(method: string, operation: string): Error {
  return new Error(
    `${method}() is not supported by @headless-three/renderer because ${operation} requires Three.js shader-node graph translation and backend shader builder state that are outside the scene-oriented API. Use material.userData.headlessThreeRenderer.fragmentWgsl or material.customFragmentWgsl for the supported custom WGSL fragment path.`,
  )
}

export function unsupportedBackendOperationError(method: string, operation: string): Error {
  return new Error(
    `${method}() is not supported by @headless-three/renderer because ${operation} would require backend WebGL/WebGPU resource state that is outside the scene-oriented API. Render normal Three.js scene graphs with Renderer.render() or renderToTarget().`,
  )
}

export function assertComputeNodesLike(value: unknown, label: string): void {
  if (Array.isArray(value)) {
    if (value.length === 0) {
      throw new TypeError(`${label} must contain at least one ComputeNode-like object.`)
    }
    for (const [index, node] of value.entries()) {
      assertComputeNodeLike(node, `${label}[${index}]`)
    }
    return
  }
  assertComputeNodeLike(value, label)
}

export function assertComputeNodeLike(value: unknown, label: string): void {
  if (value == null || typeof value !== 'object' || Array.isArray(value) || (value as { isComputeNode?: unknown }).isComputeNode !== true) {
    throw new TypeError(`${label} must be a ComputeNode-like object.`)
  }
}

export function assertComputeDispatchSize(value: unknown, label: string): void {
  if (value == null) return
  if (typeof value === 'number') {
    assertPositiveInteger(value, label)
    return
  }
  if (Array.isArray(value)) {
    if (value.length < 1 || value.length > 3) {
      throw new TypeError(`${label} array must contain 1, 2, or 3 positive integer dimensions.`)
    }
    for (const [index, dimension] of value.entries()) {
      assertPositiveInteger(dimension, `${label}[${index}]`)
    }
    return
  }
  if (typeof value === 'object' && (value as { isIndirectStorageBufferAttribute?: unknown }).isIndirectStorageBufferAttribute === true) {
    return
  }
  throw new TypeError(`${label} must be a positive integer, [x, y, z] positive integer array, indirect storage buffer attribute, or null.`)
}

export function assertPositiveInteger(value: unknown, label: string): void {
  if (typeof value !== 'number' || !Number.isFinite(value) || !Number.isInteger(value) || value <= 0) {
    throw new TypeError(`${label} must be a positive integer.`)
  }
}

export function assertStorageBufferAttributeLike(value: unknown, label: string): void {
  if (
    value == null
    || typeof value !== 'object'
    || Array.isArray(value)
    || (
      (value as { isStorageBufferAttribute?: unknown }).isStorageBufferAttribute !== true
      && (value as { isStorageInstancedBufferAttribute?: unknown }).isStorageInstancedBufferAttribute !== true
    )
  ) {
    throw new TypeError(`${label} must be a storage buffer attribute-like object.`)
  }
}

export function assertTimestampQueryType(value: unknown, label: string): void {
  if (typeof value !== 'string') {
    throw new TypeError(`${label} must be "render" or "compute".`)
  }
  if (!SupportedTimestampQueryTypes.has(value)) {
    throw new TypeError(`${label} must be "render" or "compute"; received "${value}".`)
  }
}

export function assertTimestampUid(value: unknown, label: string): void {
  if (typeof value !== 'string' || value.length === 0) {
    throw new TypeError(`${label} must be a non-empty string.`)
  }
}

export function unsupportedComputeError(method: string): Error {
  return new Error(
    `${method}() is not supported by @headless-three/renderer because it does not expose WebGPU compute pipelines, storage buffers, or GPU dispatch. Render normal Three.js scene graphs with Renderer.render() or renderToTarget().`,
  )
}

export function assertExternalWebGlObjectLike(value: unknown, label: string): void {
  if (value == null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be an external WebGL object-like handle.`)
  }
}

export function assertOptionalExternalWebGlObjectLike(value: unknown, label: string): void {
  if (value == null) return
  assertExternalWebGlObjectLike(value, label)
}

export interface RawTextureCopyImage {
  data: { length: number; [index: number]: number }
  width: number
  height: number
  channels: number
}

export interface TextureCopyRegion {
  x: number
  y: number
  width: number
  height: number
}

export interface TextureCopyPosition {
  x: number
  y: number
}

export function rawTextureCopyImage(
  texture: ThreeTextureLike,
  label: string,
  options: { allowCanvasRead?: boolean; level?: number } = {},
): RawTextureCopyImage {
  if (texture.isFramebufferTexture === true) {
    throw new Error(
      `${label} uses a FramebufferTexture, which is not supported by @headless-three/renderer texture copy yet. Use a readable raw DataTexture-style source or destination, or render into a target-like object and copy its readable color data.`,
    )
  }
  if (texture.isDepthTexture === true) {
    throw new Error(
      `${label} uses a DepthTexture, which is not supported by @headless-three/renderer texture copy yet. Use Renderer.readRenderTargetPixels() or target.depthTexture readback for depth data.`,
    )
  }
  if (texture.isVideoTexture === true) {
    throw new Error(
      `${label} uses a VideoTexture, which is not supported by @headless-three/renderer texture copy because live video frames are not directly readable in Node. Copy from a readable raw texture or canvas-like image instead.`,
    )
  }
  if (texture.isStorageTexture === true) {
    throw new Error(
      `${label} uses a StorageTexture, which is not supported by @headless-three/renderer texture copy because WebGPU storage texture backing data is not directly readable in Node. Copy from a readable raw texture or canvas-like image instead.`,
    )
  }
  if (
    texture.isCompressedTexture === true ||
    texture.isCompressedArrayTexture === true ||
    texture.isCompressedCubeTexture === true
  ) {
    throw new Error(
      `${label} uses a compressed texture, which is not supported by @headless-three/renderer texture copy because compressed GPU payloads are not decoded in this path. Pre-decode the texture to readable raw data before copying.`,
    )
  }
  if (isCompressedTextureFormat(texture.format)) {
    throw new Error(
      `${label} uses a compressed texture format, which is not supported by @headless-three/renderer texture copy because compressed GPU payloads are not decoded in this path. Pre-decode the texture to readable raw data before copying.`,
    )
  }
  if (
    texture.isDataArrayTexture === true ||
    texture.isData3DTexture === true ||
    texture.isArrayTexture === true ||
    texture.is3DTexture === true
  ) {
    throw new Error(
      `${label} uses an array or 3D texture, which is not supported by @headless-three/renderer texture copy yet. Use a readable 2D texture or copy layers separately.`,
    )
  }
  const level = options.level ?? 0
  let image = texture.image ?? texture.source?.data
  if (level > 0) {
    if (!Array.isArray(texture.mipmaps)) {
      throw new TypeError(`${label}.mipmaps must be an array of image-like mip levels for level ${level}.`)
    }
    image = texture.mipmaps[level - 1]
    if (!image) {
      throw new TypeError(`${label}.mipmaps[${level - 1}] must provide a readable raw image object with data, width, and height.`)
    }
  }
  if (!image || Array.isArray(image) || Buffer.isBuffer(image) || image instanceof Uint8Array) {
    throw new TypeError(textureCopyReadableImageError(label, options.allowCanvasRead === true))
  }
  if (typeof image !== 'object') {
    throw new TypeError(textureCopyReadableImageError(label, options.allowCanvasRead === true))
  }
  const candidate = image as { data?: unknown; width?: unknown; height?: unknown }
  if (candidate.data == null) {
    if (options.allowCanvasRead === true) {
      const canvasImage = canvasLikeImageToRgba(image, label)
      if (canvasImage) {
        return { data: canvasImage.rgba, width: canvasImage.width, height: canvasImage.height, channels: 4 }
      }
    }
    throw new TypeError(textureCopyReadableImageError(label, options.allowCanvasRead === true))
  }
  const width = textureCopyPositiveInteger(candidate.width, `${label}.width`)
  const height = textureCopyPositiveInteger(candidate.height, `${label}.height`)
  const data = candidate.data
  if (!isMutableTextureCopyData(data)) {
    throw new TypeError(`${label}.data must be a mutable numeric array or typed array.`)
  }
  const pixels = width * height
  if (data.length === 0 || data.length % pixels !== 0) {
    throw new RangeError(`${label}.data length must be a positive multiple of width * height.`)
  }
  const channels = data.length / pixels
  if (!Number.isInteger(channels) || channels < 1 || channels > 4) {
    throw new RangeError(`${label}.data must use 1, 2, 3, or 4 channels per pixel.`)
  }
  return { data, width, height, channels }
}

export function textureCopyReadableImageError(label: string, allowCanvasRead: boolean): string {
  if (allowCanvasRead) {
    return `${label} must provide a readable image object with raw data, width, and height, or canvas-like pixel access, including OffscreenCanvas-backed image reads.`
  }
  return `${label} must provide a readable raw image object with data, width, and height.`
}

export function isMutableTextureCopyData(value: unknown): value is { length: number; [index: number]: number } {
  return (
    (Array.isArray(value) || ArrayBuffer.isView(value)) &&
    typeof (value as { length?: unknown }).length === 'number'
  )
}

export function assertTextureCopyLevel(value: unknown, label: string): void {
  const level = value == null ? 0 : value
  if (!Number.isInteger(level) || (level as number) < 0) {
    throw new TypeError(`${label} must be a non-negative integer.`)
  }
}

export function textureCopySourceRegion(value: unknown, sourceWidth: number, sourceHeight: number, label: string): TextureCopyRegion {
  if (value == null) {
    return { x: 0, y: 0, width: sourceWidth, height: sourceHeight }
  }
  let region: TextureCopyRegion
  if (Array.isArray(value)) {
    region = {
      x: textureCopyInteger(value[0], `${label}.x`),
      y: textureCopyInteger(value[1], `${label}.y`),
      width: textureCopyPositiveInteger(value[2], `${label}.width`),
      height: textureCopyPositiveInteger(value[3], `${label}.height`),
    }
  } else if (typeof value === 'object') {
    const candidate = value as {
      x?: unknown
      y?: unknown
      width?: unknown
      height?: unknown
      min?: { x?: unknown; y?: unknown }
      max?: { x?: unknown; y?: unknown }
    }
    if (candidate.min && candidate.max) {
      const x = textureCopyInteger(candidate.min.x, `${label}.min.x`)
      const y = textureCopyInteger(candidate.min.y, `${label}.min.y`)
      const maxX = textureCopyInteger(candidate.max.x, `${label}.max.x`)
      const maxY = textureCopyInteger(candidate.max.y, `${label}.max.y`)
      region = { x, y, width: maxX - x, height: maxY - y }
      if (region.width <= 0 || region.height <= 0) {
        throw new RangeError(`${label} box must have positive width and height.`)
      }
    } else {
      region = {
        x: textureCopyInteger(candidate.x, `${label}.x`),
        y: textureCopyInteger(candidate.y, `${label}.y`),
        width: textureCopyPositiveInteger(candidate.width, `${label}.width`),
        height: textureCopyPositiveInteger(candidate.height, `${label}.height`),
      }
    }
  } else {
    throw new TypeError(`${label} must be a rectangle object, Box2-like object, array, or null.`)
  }
  if (region.x < 0 || region.y < 0 || region.x + region.width > sourceWidth || region.y + region.height > sourceHeight) {
    throw new RangeError(`${label} must fit inside the source texture bounds.`)
  }
  return region
}

export function textureCopyFramebufferSourceRegion(
  value: unknown,
  defaultWidth: number,
  defaultHeight: number,
  sourceWidth: number,
  sourceHeight: number,
  label: string,
): TextureCopyRegion {
  let region: TextureCopyRegion
  if (value == null) {
    region = { x: 0, y: 0, width: defaultWidth, height: defaultHeight }
  } else if (Array.isArray(value)) {
    region = value.length >= 4
      ? {
          x: textureCopyFlooredInteger(value[0], `${label}.x`),
          y: textureCopyFlooredInteger(value[1], `${label}.y`),
          width: textureCopyPositiveFlooredInteger(value[2], `${label}.width`),
          height: textureCopyPositiveFlooredInteger(value[3], `${label}.height`),
        }
      : {
          x: textureCopyFlooredInteger(value[0], `${label}.x`),
          y: textureCopyFlooredInteger(value[1], `${label}.y`),
          width: defaultWidth,
          height: defaultHeight,
        }
  } else if (typeof value === 'object') {
    const candidate = value as {
      isVector2?: unknown
      isVector4?: unknown
      x?: unknown
      y?: unknown
      z?: unknown
      w?: unknown
      width?: unknown
      height?: unknown
    }
    const x = textureCopyFlooredInteger(candidate.x, `${label}.x`)
    const y = textureCopyFlooredInteger(candidate.y, `${label}.y`)
    const isVector2 = candidate.isVector2 === true
    const isVector4 = candidate.isVector4 === true
    const width = isVector2 ? undefined : (isVector4 ? candidate.z : candidate.width ?? candidate.z)
    const height = isVector2 ? undefined : (isVector4 ? candidate.w : candidate.height ?? candidate.w)
    region = width === undefined && height === undefined
      ? { x, y, width: defaultWidth, height: defaultHeight }
      : {
          x,
          y,
          width: textureCopyPositiveFlooredInteger(width, `${label}.width`),
          height: textureCopyPositiveFlooredInteger(height, `${label}.height`),
        }
  } else {
    throw new TypeError(`${label} must be a vector, rectangle object, array, or null.`)
  }
  if (region.x < 0 || region.y < 0) {
    throw new RangeError(`${label} x and y must be non-negative.`)
  }
  if (region.x + region.width > sourceWidth || region.y + region.height > sourceHeight) {
    throw new RangeError(`${label} must fit inside the active framebuffer bounds.`)
  }
  return region
}

export function textureCopyDestinationPosition(value: unknown, label: string): TextureCopyPosition {
  if (value == null) return { x: 0, y: 0 }
  if (Array.isArray(value)) {
    const x = textureCopyInteger(value[0], `${label}.x`)
    const y = textureCopyInteger(value[1], `${label}.y`)
    if (x < 0 || y < 0) throw new RangeError(`${label} must be non-negative.`)
    return { x, y }
  }
  if (typeof value === 'object') {
    const candidate = value as { x?: unknown; y?: unknown }
    const x = textureCopyInteger(candidate.x, `${label}.x`)
    const y = textureCopyInteger(candidate.y, `${label}.y`)
    if (x < 0 || y < 0) throw new RangeError(`${label} must be non-negative.`)
    return { x, y }
  }
  throw new TypeError(`${label} must be a vector object, array, or null.`)
}
