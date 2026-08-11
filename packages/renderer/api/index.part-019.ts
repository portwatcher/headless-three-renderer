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
import { FloatType, HalfFloatType, PixelRect, PixelSize, UnsignedByteType, UnsignedInt248Type, UnsignedIntType, UnsignedShortType } from './index.part-012'
import { cubeMipmapRect, cubeMipmapSize, cubeTargetTexture, isCubeRenderTarget, normalizeOptionalPixelRect, normalizePixelRect, resolveActiveMipmapLevel, writeCubeRenderTargetFace, writeCubeTextureFace } from './index.part-013'
import { writeRenderTarget } from './index.part-018'
import { renderTargetTextureFaceImage, writeRenderTargetTexture } from './index.part-020'
import { normalizedFloatToHalf } from './index.part-021'
export function clearRenderTargetColor(
  target: RenderTargetLike,
  color: Color4,
  fallbackSize: PixelSize | null,
  rendererScissor: PixelRect | null,
  rendererScissorTest: boolean,
  activeCubeFace: number,
  activeMipmapLevel: number,
): void {
  const size = renderTargetClearSize(target, fallbackSize)
  if (!size) return

  const colorTextures = renderTargetColorTextures(target)
  if (isCubeRenderTarget(target)) {
    const resolvedMipmapLevel = resolveActiveMipmapLevel(activeMipmapLevel, size.width, 'Renderer activeMipmapLevel')
    const mipSize = cubeMipmapSize(size.width, size.height, resolvedMipmapLevel)
    const scissor = renderTargetClearScissor(
      target,
      rendererScissor,
      rendererScissorTest,
      mipSize.width,
      mipSize.height,
      resolvedMipmapLevel,
    )
    const face = clearColorBuffer(
      color,
      mipSize.width,
      mipSize.height,
      scissor,
      renderTargetExistingColorBuffer(
        renderTargetTextureFaceImage(cubeTargetTexture(target), activeCubeFace, resolvedMipmapLevel)?.data,
        mipSize.width,
        mipSize.height,
      ),
    )
    const attachments = colorTextures.slice(1).map((texture) => ({ texture, data: face }))
    writeCubeRenderTargetFace(
      target,
      face,
      size.width,
      size.height,
      mipSize.width,
      mipSize.height,
      activeCubeFace,
      resolvedMipmapLevel,
      undefined,
      undefined,
      attachments,
    )
    return
  }

  const scissor = renderTargetClearScissor(
    target,
    rendererScissor,
    rendererScissorTest,
    size.width,
    size.height,
  )
  const data = clearColorBuffer(
    color,
    size.width,
    size.height,
    scissor,
    renderTargetExistingColorBuffer(target.data, size.width, size.height),
  )
  const attachments = colorTextures.slice(1).map((texture) => ({ texture, data }))
  writeRenderTarget(target, data, size.width, size.height, undefined, undefined, attachments)
}

export function clearRenderTargetDepth(
  target: RenderTargetLike,
  depth: number,
  fallbackSize: PixelSize | null,
  rendererScissor: PixelRect | null,
  rendererScissorTest: boolean,
  activeCubeFace: number,
  activeMipmapLevel: number,
): void {
  if (!target.depthTexture) return
  const size = renderTargetClearSize(target, fallbackSize)
  if (!size) return

  if (isCubeRenderTarget(target)) {
    const resolvedMipmapLevel = resolveActiveMipmapLevel(activeMipmapLevel, size.width, 'Renderer activeMipmapLevel')
    const mipSize = cubeMipmapSize(size.width, size.height, resolvedMipmapLevel)
    const scissor = renderTargetClearScissor(
      target,
      rendererScissor,
      rendererScissorTest,
      mipSize.width,
      mipSize.height,
      resolvedMipmapLevel,
    )
    writeCubeTextureFace(
      target.depthTexture,
      clearDepthTextureData(
        target.depthTexture,
        depth,
        mipSize.width,
        mipSize.height,
        scissor,
        renderTargetTextureFaceImage(target.depthTexture, activeCubeFace, resolvedMipmapLevel)?.data,
      ),
      mipSize.width,
      mipSize.height,
      activeCubeFace,
      resolvedMipmapLevel,
      'target.depthTexture',
    )
    return
  }

  const scissor = renderTargetClearScissor(
    target,
    rendererScissor,
    rendererScissorTest,
    size.width,
    size.height,
  )
  writeRenderTargetTexture(
    target.depthTexture,
    clearDepthTextureData(
      target.depthTexture,
      depth,
      size.width,
      size.height,
      scissor,
      renderTargetDepthImage(target.depthTexture)?.data,
    ),
    size.width,
    size.height,
  )
}

export function clearRenderTargetStencil(
  target: RenderTargetLike,
  stencil: number,
  fallbackDepth: number,
  fallbackSize: PixelSize | null,
  rendererScissor: PixelRect | null,
  rendererScissorTest: boolean,
  activeCubeFace: number,
  activeMipmapLevel: number,
): void {
  if (!target.depthTexture || target.depthTexture.type !== UnsignedInt248Type) return
  const size = renderTargetClearSize(target, fallbackSize)
  if (!size) return

  if (isCubeRenderTarget(target)) {
    const resolvedMipmapLevel = resolveActiveMipmapLevel(activeMipmapLevel, size.width, 'Renderer activeMipmapLevel')
    const mipSize = cubeMipmapSize(size.width, size.height, resolvedMipmapLevel)
    const scissor = renderTargetClearScissor(
      target,
      rendererScissor,
      rendererScissorTest,
      mipSize.width,
      mipSize.height,
      resolvedMipmapLevel,
    )
    writeCubeTextureFace(
      target.depthTexture,
      clearPackedDepthStencilData(
        fallbackDepth,
        stencil,
        mipSize.width,
        mipSize.height,
        scissor,
        renderTargetTextureFaceImage(target.depthTexture, activeCubeFace, resolvedMipmapLevel)?.data,
      ),
      mipSize.width,
      mipSize.height,
      activeCubeFace,
      resolvedMipmapLevel,
      'target.depthTexture',
    )
    return
  }

  const scissor = renderTargetClearScissor(
    target,
    rendererScissor,
    rendererScissorTest,
    size.width,
    size.height,
  )
  writeRenderTargetTexture(
    target.depthTexture,
    clearPackedDepthStencilData(
      fallbackDepth,
      stencil,
      size.width,
      size.height,
      scissor,
      renderTargetDepthImage(target.depthTexture)?.data,
    ),
    size.width,
    size.height,
  )
}

export function renderTargetClearScissor(
  target: RenderTargetLike,
  rendererScissor: PixelRect | null,
  rendererScissorTest: boolean,
  width: number,
  height: number,
  activeMipmapLevel = 0,
): PixelRect | undefined {
  if (target.scissorTest === true) {
    const scissor = activeMipmapLevel > 0
      ? cubeMipmapRect(target.scissor, activeMipmapLevel)
      : target.scissor
    return normalizeOptionalPixelRect(scissor, width, height, 'target.scissor')
  }
  if (rendererScissorTest && rendererScissor) {
    return normalizePixelRect(rendererScissor, width, height, 'Renderer.scissor')
  }
  return undefined
}

export function renderTargetClearSize(
  target: RenderTargetLike,
  fallbackSize: PixelSize | null,
): { width: number; height: number } | null {
  const image = renderTargetClearImage(target)
  const rawWidth = target.width ?? image?.width ?? fallbackSize?.width
  const width = renderTargetClearDimension(rawWidth)
  const height = renderTargetClearDimension(
    target.height ?? image?.height ?? fallbackSize?.height ?? (isCubeRenderTarget(target) ? rawWidth : undefined),
  )
  if (width === null || height === null) return null
  return { width, height }
}

export function renderTargetClearImage(target: RenderTargetLike): RenderTargetImageLike | undefined {
  const texture = renderTargetColorTexture(target)
  if (texture) {
    const image = Array.isArray(texture.image) ? texture.image[0] : texture.image
    if (image && !Buffer.isBuffer(image) && !(image instanceof Uint8Array)) return image
    const sourceData = texture.source?.data
    if (sourceData && !Buffer.isBuffer(sourceData) && !(sourceData instanceof Uint8Array) && !Array.isArray(sourceData)) {
      return sourceData
    }
  }
  return target.image
}

export function renderTargetDepthImage(texture: RenderTargetTextureLike): RenderTargetImageLike | undefined {
  if (Array.isArray(texture.image)) return texture.image[0]
  if (texture.image?.data) return texture.image
  const sourceData = texture.source?.data
  if (Array.isArray(sourceData)) return sourceData[0]
  return sourceData
}

export function renderTargetClearDimension(value: unknown): number | null {
  return typeof value === 'number' && Number.isInteger(value) && value > 0 ? value : null
}

export function renderTargetExistingColorBuffer(
  data: RenderTargetImageLike['data'] | undefined,
  width: number,
  height: number,
): Buffer | undefined {
  if (
    (
      Buffer.isBuffer(data) ||
      data instanceof Uint8Array ||
      data instanceof Uint8ClampedArray
    ) &&
    data.length === width * height * 4
  ) {
    return Buffer.from(data)
  }
  return undefined
}

export function clearColorBuffer(
  color: Color4,
  width: number,
  height: number,
  rect?: PixelRect,
  existing?: Buffer,
): Buffer {
  const data = existing ? Buffer.from(existing) : Buffer.alloc(width * height * 4)
  const r = Math.round(clamp01(color[0]) * 255)
  const g = Math.round(clamp01(color[1]) * 255)
  const b = Math.round(clamp01(color[2]) * 255)
  const a = Math.round(clamp01(color[3]) * 255)
  const clearRect = rect ?? { x: 0, y: 0, width, height }
  for (let row = 0; row < clearRect.height; row += 1) {
    const rowStart = ((clearRect.y + row) * width + clearRect.x) * 4
    const rowEnd = rowStart + clearRect.width * 4
    for (let offset = rowStart; offset < rowEnd; offset += 4) {
      data[offset] = r
      data[offset + 1] = g
      data[offset + 2] = b
      data[offset + 3] = a
    }
  }
  return data
}

export type ScalarDepthArray = Uint8Array | Uint16Array | Uint32Array | Float32Array
export type ScalarDepthArrayConstructor<T extends ScalarDepthArray> = {
  new(length: number): T
  new(array: ArrayLike<number>): T
}

export function clearDepthTextureData(
  texture: RenderTargetTextureLike,
  depth: number,
  width: number,
  height: number,
  rect?: PixelRect,
  existing?: NonNullable<RenderTargetImageLike['data']>,
): NonNullable<RenderTargetImageLike['data']> {
  const clampedDepth = clamp01(depth)
  if (texture.type === UnsignedByteType) {
    return clearScalarDepthData(Uint8Array, existing, width, height, rect, Math.round(clampedDepth * 0xff))
  }
  if (texture.type === UnsignedShortType) {
    return clearScalarDepthData(Uint16Array, existing, width, height, rect, Math.round(clampedDepth * 0xffff))
  }
  if (texture.type === UnsignedIntType) {
    return clearScalarDepthData(Uint32Array, existing, width, height, rect, Math.round(clampedDepth * 0xffffffff))
  }
  if (texture.type === UnsignedInt248Type) {
    return clearPackedDepthData(clampedDepth, width, height, rect, existing)
  }
  if (texture.type === FloatType) {
    return clearScalarDepthData(Float32Array, existing, width, height, rect, clampedDepth)
  }
  if (texture.type === HalfFloatType) {
    return clearScalarDepthData(Uint16Array, existing, width, height, rect, normalizedFloatToHalf(clampedDepth))
  }
  return clearDepthRgbaBuffer(clampedDepth, width, height, rect, existing)
}

export function clearPackedDepthData(
  depth: number,
  width: number,
  height: number,
  rect: PixelRect | undefined,
  existing: NonNullable<RenderTargetImageLike['data']> | undefined,
): Uint32Array {
  const pixelCount = width * height
  const data = existing instanceof Uint32Array && existing.length === pixelCount
    ? new Uint32Array(existing)
    : new Uint32Array(pixelCount)
  const value = Math.round(clamp01(depth) * 0xffffff) * 0x100
  const clearRect = rect ?? { x: 0, y: 0, width, height }
  for (let row = 0; row < clearRect.height; row += 1) {
    const rowStart = (clearRect.y + row) * width + clearRect.x
    const rowEnd = rowStart + clearRect.width
    for (let offset = rowStart; offset < rowEnd; offset += 1) {
      data[offset] = value + (data[offset] & 0xff)
    }
  }
  return data
}

export function clearPackedDepthStencilData(
  fallbackDepth: number,
  stencil: number,
  width: number,
  height: number,
  rect: PixelRect | undefined,
  existing: NonNullable<RenderTargetImageLike['data']> | undefined,
): Uint32Array {
  const pixelCount = width * height
  const fallbackValue = Math.round(clamp01(fallbackDepth) * 0xffffff) * 0x100
  const data = existing instanceof Uint32Array && existing.length === pixelCount
    ? new Uint32Array(existing)
    : new Uint32Array(pixelCount).fill(fallbackValue)
  const value = stencil & 0xff
  const clearRect = rect ?? { x: 0, y: 0, width, height }
  for (let row = 0; row < clearRect.height; row += 1) {
    const rowStart = (clearRect.y + row) * width + clearRect.x
    const rowEnd = rowStart + clearRect.width
    for (let offset = rowStart; offset < rowEnd; offset += 1) {
      data[offset] = Math.floor(data[offset] / 0x100) * 0x100 + value
    }
  }
  return data
}

export function clearScalarDepthData<T extends ScalarDepthArray>(
  ctor: ScalarDepthArrayConstructor<T>,
  existing: NonNullable<RenderTargetImageLike['data']> | undefined,
  width: number,
  height: number,
  rect: PixelRect | undefined,
  value: number,
): T {
  const pixelCount = width * height
  const data = existing instanceof ctor && existing.length === pixelCount
    ? new ctor(existing)
    : new ctor(pixelCount)
  const clearRect = rect ?? { x: 0, y: 0, width, height }
  for (let row = 0; row < clearRect.height; row += 1) {
    const rowStart = (clearRect.y + row) * width + clearRect.x
    const rowEnd = rowStart + clearRect.width
    for (let offset = rowStart; offset < rowEnd; offset += 1) {
      data[offset] = value
    }
  }
  return data
}

export function clearDepthRgbaBuffer(
  depth: number,
  width: number,
  height: number,
  rect?: PixelRect,
  existing?: NonNullable<RenderTargetImageLike['data']>,
): Buffer {
  const data = Buffer.isBuffer(existing) && existing.length === width * height * 4
    ? Buffer.from(existing)
    : Buffer.alloc(width * height * 4)
  const value = Math.round(depth * 255)
  const clearRect = rect ?? { x: 0, y: 0, width, height }
  for (let row = 0; row < clearRect.height; row += 1) {
    const rowStart = ((clearRect.y + row) * width + clearRect.x) * 4
    const rowEnd = rowStart + clearRect.width * 4
    for (let offset = rowStart; offset < rowEnd; offset += 4) {
      data[offset] = value
      data[offset + 1] = value
      data[offset + 2] = value
      data[offset + 3] = 255
    }
  }
  return data
}

export function writeObjectIdMetadata(target: RenderTargetLike, objectIdEntries?: RenderObjectIdEntry[]): void {
  if (objectIdEntries) {
    target.objectIdEntries = objectIdEntries
    target.objectIdMap = Object.fromEntries(objectIdEntries.map((entry) => [String(entry.encodedId), entry]))
  } else {
    delete target.objectIdEntries
    delete target.objectIdMap
  }
}

export function renderTargetColorTexture(target: RenderTargetLike): RenderTargetTextureLike | undefined {
  return renderTargetColorTextures(target)[0]
}

export function renderTargetColorTextures(target: RenderTargetLike): RenderTargetTextureLike[] {
  if (Array.isArray(target.texture)) return target.texture
  if (target.textures) return target.textures
  return target.texture ? [target.texture] : []
}
