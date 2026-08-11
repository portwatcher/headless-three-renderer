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
import { RendererColorBufferState, RendererDepthBufferState, RendererStateBuffersState, RendererStencilBufferState } from './index.part-005'
import { assertRendererStateBlendingMode, assertRendererStateCullFace, rendererStateBoolean, rendererStateClearAlpha, rendererStateFiniteNumber, rendererStateOptionalFiniteInteger, rendererStatePixelRect, rendererStatePositiveFiniteNumber, throwUnsupportedRendererStateWebGl } from './index.part-014'
import { assertFiniteInteger, assertPropertyKey, assertWeakMapKey, assertWebGlExtensionName } from './index.part-017'
import { isReadableRenderTargetColorFormat, isReadableRenderTargetColorType } from './index.part-018'
export class RendererState {
  readonly buffers = new RendererStateBuffersState()

  get color(): RendererColorBufferState {
    return this.buffers.color
  }

  get depth(): RendererDepthBufferState {
    return this.buffers.depth
  }

  get stencil(): RendererStencilBufferState {
    return this.buffers.stencil
  }

  setBlending(
    blending: unknown,
    blendEquation?: unknown,
    blendSrc?: unknown,
    blendDst?: unknown,
    blendEquationAlpha?: unknown,
    blendSrcAlpha?: unknown,
    blendDstAlpha?: unknown,
    blendColor?: unknown,
    blendAlpha?: unknown,
    premultipliedAlpha?: unknown,
  ): void {
    assertRendererStateBlendingMode(blending, 'Renderer.state.setBlending blending')
    rendererStateOptionalFiniteInteger(blendEquation, 'Renderer.state.setBlending blendEquation')
    rendererStateOptionalFiniteInteger(blendSrc, 'Renderer.state.setBlending blendSrc')
    rendererStateOptionalFiniteInteger(blendDst, 'Renderer.state.setBlending blendDst')
    rendererStateOptionalFiniteInteger(blendEquationAlpha, 'Renderer.state.setBlending blendEquationAlpha')
    rendererStateOptionalFiniteInteger(blendSrcAlpha, 'Renderer.state.setBlending blendSrcAlpha')
    rendererStateOptionalFiniteInteger(blendDstAlpha, 'Renderer.state.setBlending blendDstAlpha')
    if (blendColor !== undefined && (blendColor === null || typeof blendColor !== 'object')) {
      throw new TypeError('Renderer.state.setBlending blendColor must be a color-like object when provided.')
    }
    if (blendAlpha !== undefined) {
      rendererStateClearAlpha(blendAlpha, 'Renderer.state.setBlending blendAlpha')
    }
    if (premultipliedAlpha !== undefined) {
      rendererStateBoolean(premultipliedAlpha, 'Renderer.state.setBlending premultipliedAlpha')
    }
  }

  setMaterial(material: unknown, frontFaceCW?: unknown): void {
    if (material === null || typeof material !== 'object' || Array.isArray(material)) {
      throw new TypeError('Renderer.state.setMaterial material must be a material-like object.')
    }
    if (frontFaceCW !== undefined) {
      rendererStateBoolean(frontFaceCW, 'Renderer.state.setMaterial frontFaceCW')
    }
  }

  setFlipSided(flipSided: unknown): void {
    rendererStateBoolean(flipSided, 'Renderer.state.setFlipSided flipSided')
  }

  setCullFace(cullFace: unknown): void {
    assertRendererStateCullFace(cullFace, 'Renderer.state.setCullFace cullFace')
  }

  setLineWidth(width: unknown): void {
    rendererStatePositiveFiniteNumber(width, 'Renderer.state.setLineWidth width')
  }

  setPolygonOffset(polygonOffset: unknown, factor = 0, units = 0): void {
    rendererStateBoolean(polygonOffset, 'Renderer.state.setPolygonOffset polygonOffset')
    rendererStateFiniteNumber(factor, 'Renderer.state.setPolygonOffset factor')
    rendererStateFiniteNumber(units, 'Renderer.state.setPolygonOffset units')
  }

  setScissorTest(scissorTest: unknown): void {
    rendererStateBoolean(scissorTest, 'Renderer.state.setScissorTest scissorTest')
  }

  setMRTBlending(): never {
    throwUnsupportedRendererStateWebGl('setMRTBlending', 'WebGL MRT indexed blending')
  }

  setVertexState(): never {
    throwUnsupportedRendererStateWebGl('setVertexState', 'WebGL vertex-array binding')
  }

  resetVertexState(): void {
    // Native vertex state is rebuilt while preparing each render pass.
  }

  setColorMask(colorMask: unknown): void {
    this.buffers.color.setMask(colorMask)
  }

  setDepthTest(depthTest: unknown): void {
    this.buffers.depth.setTest(depthTest)
  }

  setDepthMask(depthMask: unknown): void {
    this.buffers.depth.setMask(depthMask)
  }

  setDepthFunc(depthFunc: unknown): void {
    this.buffers.depth.setFunc(depthFunc)
  }

  setReversedDepth(reversed: unknown): void {
    this.buffers.depth.setReversed(reversed)
  }

  setStencilTest(stencilTest: unknown): void {
    this.buffers.stencil.setTest(stencilTest)
  }

  setStencilMask(stencilMask: unknown): void {
    this.buffers.stencil.setMask(stencilMask)
  }

  setStencilFunc(stencilFunc: unknown, stencilRef: unknown, stencilMask: unknown): void {
    this.buffers.stencil.setFunc(stencilFunc, stencilRef, stencilMask)
  }

  setStencilOp(stencilFail: unknown, stencilZFail: unknown, stencilZPass: unknown): void {
    this.buffers.stencil.setOp(stencilFail, stencilZFail, stencilZPass)
  }

  scissor(rectOrX: RenderPixelRectLike | null | number, y?: number, width?: number, height?: number): void {
    rendererStatePixelRect(rectOrX, y, width, height, 'Renderer.state.scissor')
  }

  viewport(rectOrX: RenderPixelRectLike | null | number, y?: number, width?: number, height?: number): void {
    rendererStatePixelRect(rectOrX, y, width, height, 'Renderer.state.viewport')
  }

  reset(): void {
    // Native render state is rebuilt for each pass.
  }

  unbindTexture(): void {
    // Texture binding is not exposed by the wgpu-backed adapter.
  }

  enable(): never {
    throwUnsupportedRendererStateWebGl('enable', 'WebGL capability flags')
  }

  disable(): never {
    throwUnsupportedRendererStateWebGl('disable', 'WebGL capability flags')
  }

  bindFramebuffer(): never {
    throwUnsupportedRendererStateWebGl('bindFramebuffer', 'WebGL framebuffer binding')
  }

  drawBuffers(): never {
    throwUnsupportedRendererStateWebGl('drawBuffers', 'WebGL draw-buffer binding')
  }

  useProgram(): never {
    throwUnsupportedRendererStateWebGl('useProgram', 'WebGL program binding')
  }

  activeTexture(): never {
    throwUnsupportedRendererStateWebGl('activeTexture', 'WebGL texture-unit binding')
  }

  bindTexture(): never {
    throwUnsupportedRendererStateWebGl('bindTexture', 'WebGL texture binding')
  }

  compressedTexImage2D(): never {
    throwUnsupportedRendererStateWebGl('compressedTexImage2D', 'WebGL texture uploads')
  }

  compressedTexImage3D(): never {
    throwUnsupportedRendererStateWebGl('compressedTexImage3D', 'WebGL texture uploads')
  }

  texImage2D(): never {
    throwUnsupportedRendererStateWebGl('texImage2D', 'WebGL texture uploads')
  }

  texImage3D(): never {
    throwUnsupportedRendererStateWebGl('texImage3D', 'WebGL texture uploads')
  }

  texStorage2D(): never {
    throwUnsupportedRendererStateWebGl('texStorage2D', 'WebGL texture storage')
  }

  texStorage3D(): never {
    throwUnsupportedRendererStateWebGl('texStorage3D', 'WebGL texture storage')
  }

  texSubImage2D(): never {
    throwUnsupportedRendererStateWebGl('texSubImage2D', 'WebGL texture uploads')
  }

  texSubImage3D(): never {
    throwUnsupportedRendererStateWebGl('texSubImage3D', 'WebGL texture uploads')
  }

  compressedTexSubImage2D(): never {
    throwUnsupportedRendererStateWebGl('compressedTexSubImage2D', 'WebGL texture uploads')
  }

  compressedTexSubImage3D(): never {
    throwUnsupportedRendererStateWebGl('compressedTexSubImage3D', 'WebGL texture uploads')
  }

  updateUBOMapping(): never {
    throwUnsupportedRendererStateWebGl('updateUBOMapping', 'WebGL uniform-buffer binding')
  }

  uniformBlockBinding(): never {
    throwUnsupportedRendererStateWebGl('uniformBlockBinding', 'WebGL uniform-buffer binding')
  }

  bindBufferBase(): never {
    throwUnsupportedRendererStateWebGl('bindBufferBase', 'WebGL uniform-buffer binding')
  }
}

export class RendererExtensionsState {
  has(name: string): boolean {
    assertWebGlExtensionName(name, 'Renderer.extensions.has name')
    return false
  }

  init(): void {
    // There are no browser WebGL extensions to preload in the wgpu-backed adapter.
  }

  get(name: string): null {
    assertWebGlExtensionName(name, 'Renderer.extensions.get name')
    return null
  }
}

export class RendererCapabilitiesState {
  readonly isWebGL2 = false
  readonly drawBuffers = false
  readonly precision = 'highp'
  readonly logarithmicDepthBuffer = false
  readonly reversedDepthBuffer = false
  readonly reverseDepthBuffer = false
  readonly vertexTextures = false
  readonly floatFragmentTextures = false
  readonly floatVertexTextures = false
  readonly maxTextures = 0
  readonly maxVertexTextures = 0
  readonly maxTextureSize = 0
  readonly maxCubemapSize = 0
  readonly maxAttributes = 0
  readonly maxVertexUniforms = 0
  readonly maxVaryings = 0
  readonly maxFragmentUniforms = 0
  readonly maxDrawBuffers = 1
  readonly maxColorAttachments = 1
  readonly maxSamples = 4
  readonly samples = 0

  getMaxAnisotropy(): number {
    return 0
  }

  getMaxPrecision(precision: string): string {
    if (precision === 'highp' || precision === 'mediump' || precision === 'lowp') {
      return precision
    }
    throw new Error(
      `Renderer.capabilities.getMaxPrecision precision ${String(precision)} is not supported. Use "highp", "mediump", or "lowp".`,
    )
  }

  textureFormatReadable(textureFormat: number): boolean {
    assertFiniteInteger(textureFormat, 'Renderer.capabilities.textureFormatReadable format')
    return isReadableRenderTargetColorFormat(textureFormat)
  }

  textureTypeReadable(textureType: number): boolean {
    assertFiniteInteger(textureType, 'Renderer.capabilities.textureTypeReadable type')
    return isReadableRenderTargetColorType(textureType)
  }
}

export class RendererPropertiesState {
  private properties = new WeakMap<object, Record<string, unknown>>()

  has(object: object): boolean {
    assertWeakMapKey(object, 'Renderer.properties.has object')
    return this.properties.has(object)
  }

  get(object: object): Record<string, unknown> {
    assertWeakMapKey(object, 'Renderer.properties.get object')
    let map = this.properties.get(object)
    if (map === undefined) {
      map = {}
      this.properties.set(object, map)
    }
    return map
  }

  remove(object: object): void {
    assertWeakMapKey(object, 'Renderer.properties.remove object')
    this.properties.delete(object)
  }

  update(object: object, key: string, value: unknown): void {
    assertWeakMapKey(object, 'Renderer.properties.update object')
    assertPropertyKey(key, 'Renderer.properties.update key')
    this.get(object)[key] = value
  }

  dispose(): void {
    this.properties = new WeakMap()
  }
}

export type RendererRenderListSort = (a: RendererRenderListItem, b: RendererRenderListItem) => number

export type RendererRenderListItem = {
  id: unknown
  object: unknown
  geometry: unknown
  material: unknown
  materialVariant: number | null
  groupOrder: unknown
  renderOrder: unknown
  z: unknown
  group: unknown
  clippingContext: unknown
}
