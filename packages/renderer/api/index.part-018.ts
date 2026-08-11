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
import { AdditiveBlending } from './index.part-001'
import { AlphaFormat, ByteType, DepthFormat, DepthStencilFormat, FloatType, HalfFloatType, IntType, LuminanceAlphaFormat, LuminanceFormat, PixelRect, RGBAFormat, RGBAIntegerFormat, RGBFormat, RGBIntegerFormat, RGFormat, RGIntegerFormat, RedFormat, RedIntegerFormat, RenderTargetAttachmentData, ShortType, UnsignedByteType, UnsignedInt101111Type, UnsignedInt248Type, UnsignedInt5999Type, UnsignedIntType, UnsignedShort4444Type, UnsignedShort5551Type, UnsignedShortType } from './index.part-012'
import { effectiveScissor, effectiveScissorLabel, effectiveViewport, effectiveViewportLabel, rendererStatePixelRect } from './index.part-014'
import { targetColorTextureLabel } from './index.part-017'
import { renderTargetColorTexture, renderTargetColorTextures, renderTargetExistingColorBuffer, writeObjectIdMetadata } from './index.part-019'
import { colorTextureData, writeRenderTargetTexture } from './index.part-020'
import { depthTextureData } from './index.part-021'
export function validatePostProcessingOptions(value: unknown): void {
  if (value == null || value === false) return
  if (typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError('options.postProcessing must be an object.')
  }
  const post = value as RenderOptions['postProcessing']
  if (post?.enabled != null && typeof post.enabled !== 'boolean') {
    throw new TypeError('options.postProcessing.enabled must be a boolean.')
  }
  if (post?.enabled === false) return
  assertFinitePostProcessingNumber(post?.exposure, 'options.postProcessing.exposure')
  assertFinitePostProcessingNumber(post?.contrast, 'options.postProcessing.contrast')
  assertFinitePostProcessingNumber(post?.saturation, 'options.postProcessing.saturation')
  assertNormalizedPostProcessingNumber(post?.vignette, 'options.postProcessing.vignette')
  assertFinitePostProcessingBlend(post?.grayscale, 'options.postProcessing.grayscale')
  assertFinitePostProcessingBlend(post?.invert, 'options.postProcessing.invert')
}

export function assertFinitePostProcessingNumber(value: unknown, label: string): void {
  assertFiniteNumberOption(value, label)
}

export function assertNormalizedPostProcessingNumber(value: unknown, label: string): void {
  assertFiniteNumberOption(value, label)
  if (typeof value === 'number' && (value < 0 || value > 1)) {
    throw new TypeError(`${label} must be between 0 and 1.`)
  }
}

export function assertFinitePostProcessingBlend(value: unknown, label: string): void {
  if (value == null || typeof value === 'boolean') return
  if (typeof value === 'number' && Number.isFinite(value)) {
    if (value < 0 || value > 1) {
      throw new TypeError(`${label} must be between 0 and 1.`)
    }
    return
  }
  throw new TypeError(`${label} must be a finite number or boolean.`)
}

export function assertFiniteNumberOption(value: unknown, label: string): void {
  if (value == null) return
  if (typeof value === 'number' && Number.isFinite(value)) return
  throw new TypeError(`${label} must be a finite number.`)
}

export function assertNonNegativeNumberOption(value: unknown, label: string): void {
  assertFiniteNumberOption(value, label)
  if (typeof value === 'number' && value < 0) {
    throw new TypeError(`${label} must be non-negative.`)
  }
}

export function assertNormalizedNumberOption(value: unknown, label: string): void {
  assertFiniteNumberOption(value, label)
  if (typeof value === 'number' && (value < 0 || value > 1)) {
    throw new TypeError(`${label} must be between 0 and 1.`)
  }
}

export function assertSupportedRenderTargetColorTexture(texture: RenderTargetTextureLike | undefined, label = 'target color texture'): void {
  if (!texture) return
  if (texture.isDepthTexture === true) {
    throw new Error(
      `${label} uses a DepthTexture as a color attachment, which is not supported by @headless-three/renderer render targets. Use target.depthTexture for depth readback and a regular color texture for color output.`,
    )
  }
  const format = texture.format
  if (isCompressedTextureFormat(format)) {
    throw new Error(
      `${label} format uses a compressed texture format, which is not supported by @headless-three/renderer render targets. Use a regular 2D target texture and compress output after readback if needed.`,
    )
  }
  if (
    format != null &&
    format !== AlphaFormat &&
    format !== LuminanceFormat &&
    format !== LuminanceAlphaFormat &&
    format !== RedFormat &&
    format !== RedIntegerFormat &&
    format !== RGFormat &&
    format !== RGIntegerFormat &&
    format !== RGBFormat &&
    format !== RGBIntegerFormat &&
    format !== RGBAFormat &&
    format !== RGBAIntegerFormat
  ) {
    throw new Error(
      `${label} format ${String(format)} is not supported by @headless-three/renderer yet. Use AlphaFormat, LuminanceFormat, LuminanceAlphaFormat, RedFormat, RedIntegerFormat, RGFormat, RGIntegerFormat, RGBFormat, RGBIntegerFormat, RGBAFormat, RGBAIntegerFormat, or omit format for RGBA8 readback.`,
    )
  }
  const type = texture.type
  if (
    type != null &&
    type !== UnsignedByteType &&
    type !== ByteType &&
    type !== ShortType &&
    type !== UnsignedShortType &&
    type !== IntType &&
    type !== UnsignedIntType &&
    type !== FloatType &&
    type !== HalfFloatType &&
    type !== UnsignedShort4444Type &&
    type !== UnsignedShort5551Type &&
    type !== UnsignedInt101111Type &&
    type !== UnsignedInt5999Type
  ) {
    throw new Error(
      `${label} type ${String(type)} is not supported by @headless-three/renderer yet. Use UnsignedByteType, ByteType, ShortType, UnsignedShortType, IntType, UnsignedIntType, HalfFloatType, FloatType, UnsignedShort4444Type, UnsignedShort5551Type, UnsignedInt101111Type, UnsignedInt5999Type, or omit type for RGBA8 readback.`,
    )
  }
}

export function isReadableRenderTargetColorFormat(format: number): boolean {
  return (
    format === AlphaFormat ||
    format === LuminanceFormat ||
    format === LuminanceAlphaFormat ||
    format === RedFormat ||
    format === RedIntegerFormat ||
    format === RGFormat ||
    format === RGIntegerFormat ||
    format === RGBFormat ||
    format === RGBIntegerFormat ||
    format === RGBAFormat ||
    format === RGBAIntegerFormat
  )
}

export function isReadableRenderTargetColorType(type: number): boolean {
  return (
    type === UnsignedByteType ||
    type === ByteType ||
    type === ShortType ||
    type === UnsignedShortType ||
    type === IntType ||
    type === UnsignedIntType ||
    type === FloatType ||
    type === HalfFloatType ||
    type === UnsignedShort4444Type ||
    type === UnsignedShort5551Type ||
    type === UnsignedInt101111Type ||
    type === UnsignedInt5999Type
  )
}

export function assertSupportedRenderTargetTextureDimensionality(texture: RenderTargetTextureLike | undefined, label: string): void {
  if (!texture) return
  if (
    texture.isDataArrayTexture === true ||
    texture.isData3DTexture === true ||
    texture.isArrayTexture === true ||
    texture.is3DTexture === true
  ) {
    throw new Error(
      `${label} uses an array or 3D texture, which is not supported by @headless-three/renderer render targets yet. Use a single 2D texture target or render layers separately.`,
    )
  }
}

export function assertSupportedRenderTargetTextureClass(texture: RenderTargetTextureLike | undefined, label: string): void {
  if (!texture) return
  if (texture.isFramebufferTexture === true) {
    throw new Error(
      `${label} uses a FramebufferTexture, which is not supported by @headless-three/renderer render targets. Use a regular target texture or target-like texture object for renderer-owned readback data.`,
    )
  }
  if (texture.isStorageTexture === true) {
    throw new Error(
      `${label} uses a StorageTexture, which is not supported by @headless-three/renderer render targets because WebGPU storage texture attachments are outside the scene-oriented output contract. Use a regular target texture or target-like texture object for renderer-owned readback data.`,
    )
  }
  if (texture.isCompressedTexture === true) {
    throw new Error(
      `${label} uses a compressed texture, which is not supported by @headless-three/renderer render targets. Use a regular 2D target texture and compress output after readback if needed.`,
    )
  }
}

export function assertNonCubeCameraRenderTargetTextures(target: RenderTargetLike): void {
  const colorTextures = renderTargetColorTextures(target)
  for (let i = 0; i < colorTextures.length; i += 1) {
    if (colorTextures[i]?.isCubeTexture === true) {
      throw new Error(
        `${targetColorTextureLabel(i)} uses a cube texture, which is only supported when rendering with THREE.CubeCamera. Use a 2D texture target for regular cameras.`,
      )
    }
  }
  if (target.depthTexture?.isCubeTexture === true) {
    throw new Error(
      'target.depthTexture uses a cube texture, which is only supported when rendering with THREE.CubeCamera. Use a 2D depth texture target for regular cameras.',
    )
  }
}

export function assertSupportedDepthTextureType(depthTexture: RenderTargetTextureLike | undefined): void {
  const type = depthTexture?.type
  if (type == null) return
  if (
    type === UnsignedByteType ||
    type === UnsignedShortType ||
    type === UnsignedIntType ||
    type === FloatType ||
    type === HalfFloatType ||
    type === UnsignedInt248Type
  ) return
  throw new Error(
    `target.depthTexture.type ${String(type)} is not supported by @headless-three/renderer yet. Use FloatType, HalfFloatType, UnsignedByteType, UnsignedShortType, UnsignedIntType, UnsignedInt248Type, or omit type for RGBA8 normalized depth readback.`,
  )
}

export function assertSupportedDepthTextureFormat(depthTexture: RenderTargetTextureLike | undefined): void {
  const format = depthTexture?.format
  if (format == null) return
  if (isCompressedTextureFormat(format)) {
    throw new Error(
      'target.depthTexture.format uses a compressed texture format, which is not supported by @headless-three/renderer render targets. Use DepthFormat or DepthStencilFormat for depth readback and compress output after readback if needed.',
    )
  }
  if (format === DepthFormat) {
    if (depthTexture?.type === UnsignedInt248Type) {
      throw new Error(
        'target.depthTexture.format DepthFormat is not supported with UnsignedInt248Type by @headless-three/renderer. Use DepthStencilFormat with UnsignedInt248Type, or use DepthFormat with a scalar depth texture type.',
      )
    }
    return
  }
  if (format === DepthStencilFormat) {
    if (depthTexture?.type === UnsignedInt248Type) return
    throw new Error(
      'target.depthTexture.format DepthStencilFormat is only supported with UnsignedInt248Type by @headless-three/renderer. Use DepthFormat for scalar depth readback, or set type to UnsignedInt248Type for packed depth24-stencil8 readback.',
    )
  }
  throw new Error(
    `target.depthTexture.format ${String(format)} is not supported by @headless-three/renderer yet. Use DepthFormat, or DepthStencilFormat with UnsignedInt248Type.`,
  )
}

export function resolveSampleCount(options: RenderOptions): number {
  const requested = options.target?.sampleCount
    ?? options.target?.samples
    ?? options.sampleCount
    ?? options.samples
    ?? 1
  return requested > 1 ? requested : 1
}

export function writeRenderTarget(
  target: RenderTargetLike,
  data: Buffer,
  width: number,
  height: number,
  objectIdEntries?: RenderObjectIdEntry[],
  depthData?: Buffer,
  colorAttachments?: RenderTargetAttachmentData[],
): RenderTargetLike {
  target.width = width
  target.height = height
  target.data = data

  const image = target.image ?? (target.image = {})
  image.data = data
  image.width = width
  image.height = height

  const texture = renderTargetColorTexture(target)
  if (texture) {
    writeRenderTargetTexture(texture, colorTextureData(texture, data), width, height)
  }

  for (const attachment of colorAttachments ?? []) {
    writeRenderTargetTexture(attachment.texture, colorTextureData(attachment.texture, attachment.data), width, height)
  }

  if (target.depthTexture != null && depthData) {
    writeRenderTargetTexture(target.depthTexture, depthTextureData(target.depthTexture, depthData), width, height)
  }

  writeObjectIdMetadata(target, objectIdEntries)

  return target
}

export function compositeActiveTargetColorBuffer(
  target: RenderTargetLike,
  data: Buffer,
  width: number,
  height: number,
  options: RenderOptions,
  autoClear: boolean,
  scene?: ThreeSceneRootLike,
): Buffer {
  if (autoClear) return data
  const existing = renderTargetExistingColorBuffer(target.data, width, height)
  if (!existing) return data

  const rect = activeTargetRenderRect(options, width, height)
  if (rect.width <= 0 || rect.height <= 0) return existing

  const copyShaderBlend = activeTargetCopyShaderAdditiveBlend(scene, width, height)
  if (copyShaderBlend) {
    return additiveCompositeColorBuffer(
      existing,
      copyShaderBlend.sourceData ?? data,
      width,
      rect,
      copyShaderBlend.sourceScale,
    )
  }

  if (rect.x === 0 && rect.y === 0 && rect.width === width && rect.height === height) return data

  const output = Buffer.from(existing)
  for (let row = 0; row < rect.height; row += 1) {
    const rowStart = ((rect.y + row) * width + rect.x) * 4
    const rowEnd = rowStart + rect.width * 4
    data.copy(output, rowStart, rowStart, rowEnd)
  }
  return output
}

export function activeTargetCopyShaderAdditiveBlend(
  scene: ThreeSceneRootLike | undefined,
  width: number,
  height: number,
): { sourceScale: number; sourceData?: Buffer } | null {
  if (!scene || scene.isMesh !== true || Array.isArray(scene.material)) return null
  if (Array.isArray(scene.children) && scene.children.length > 0) return null

  const material = scene.material as ThreeMaterialLike | undefined
  const copyShader = activeTargetCopyShaderMaterialInfo(material)
  if (!copyShader || material?.blending !== AdditiveBlending) return null

  const opacity = typeof copyShader.opacity === 'number' && Number.isFinite(copyShader.opacity)
    ? Math.max(0, copyShader.opacity)
    : 1
  if (material?.premultipliedAlpha === true) {
    const source = extractTextureData(material)
    if (source && source.width === width && source.height === height) {
      return { sourceScale: opacity, sourceData: source.data }
    }
  }

  const sourceScale = material?.premultipliedAlpha === true && opacity > 0
    ? 1 / opacity
    : 1
  return { sourceScale }
}

export function activeTargetCopyShaderMaterialInfo(material: ThreeMaterialLike | undefined): { opacity: unknown } | null {
  if (!material || !activeTargetShaderMaterialKind(material)) return null
  if (!activeTargetCopyShaderFragment(material.fragmentShader)) return null
  const uniforms = material.uniforms
  if (!uniforms || typeof uniforms !== 'object' || Array.isArray(uniforms)) return null
  return {
    opacity: activeTargetUniformValue((uniforms as Record<string, unknown>).opacity) ?? 1,
  }
}

export function activeTargetShaderMaterialKind(material: ThreeMaterialLike): boolean {
  return material.isShaderMaterial === true || material.type === 'ShaderMaterial'
}

export function activeTargetUniformValue(uniform: unknown): unknown {
  if (!uniform || typeof uniform !== 'object' || Array.isArray(uniform)) return undefined
  return (uniform as { value?: unknown }).value
}

export function activeTargetCopyShaderFragment(fragmentShader: unknown): boolean {
  if (typeof fragmentShader !== 'string') return false
  const compact = fragmentShader.replace(/\s+/g, '')
  return compact.includes('uniformfloatopacity;') &&
    compact.includes('uniformsampler2DtDiffuse;') &&
    compact.includes('texture2D(tDiffuse,vUv)') &&
    compact.includes('gl_FragColor=opacity*texel;')
}

export function additiveCompositeColorBuffer(
  existing: Buffer,
  data: Buffer,
  width: number,
  rect: PixelRect,
  sourceScale: number,
): Buffer {
  const output = Buffer.from(existing)
  for (let row = 0; row < rect.height; row += 1) {
    const rowStart = ((rect.y + row) * width + rect.x) * 4
    const rowEnd = rowStart + rect.width * 4
    for (let offset = rowStart; offset < rowEnd; offset += 1) {
      output[offset] = Math.min(255, Math.round(output[offset] + data[offset] * sourceScale))
    }
  }
  return output
}

export function activeTargetRenderRect(options: RenderOptions, width: number, height: number): PixelRect {
  const bounds = { x: 0, y: 0, width, height }
  const viewport = effectiveViewport(options)
  const viewportRect = viewport
    ? intersectPixelRects(bounds, rendererStatePixelRect(viewport, undefined, undefined, undefined, effectiveViewportLabel(options))!)
    : bounds
  const scissor = effectiveScissor(options)
  if (!scissor) return viewportRect ?? bounds
  return intersectPixelRects(
    viewportRect ?? bounds,
    rendererStatePixelRect(scissor, undefined, undefined, undefined, effectiveScissorLabel(options))!,
  ) ?? { x: 0, y: 0, width: 0, height: 0 }
}

export function intersectPixelRects(a: PixelRect, b: PixelRect): PixelRect | null {
  const x = Math.max(a.x, b.x)
  const y = Math.max(a.y, b.y)
  const right = Math.min(a.x + a.width, b.x + b.width)
  const bottom = Math.min(a.y + a.height, b.y + b.height)
  if (right <= x || bottom <= y) return null
  return { x, y, width: right - x, height: bottom - y }
}
