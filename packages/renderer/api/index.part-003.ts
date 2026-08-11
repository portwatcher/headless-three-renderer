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
import { WEBGL_COORDINATE_SYSTEM } from './index.part-001'
import { RendererDomElementState } from './index.part-005'
import { Renderer } from './index.part-008'
import { rendererStateBoolean } from './index.part-014'
import { assertStorageBufferAttributeLike, assertTimestampQueryType, assertTimestampUid, unsupportedBackendOperationError } from './index.part-016'
import { assertRendererProbeName, assertWeakMapKey } from './index.part-017'
export class RendererBackendState {
  readonly isWebGLBackend = false
  readonly isWebGPUBackend = false
  readonly coordinateSystem = WEBGL_COORDINATE_SYSTEM
  readonly parameters: Record<string, never> = Object.freeze({})
  private data = new WeakMap<object, Record<string, unknown>>()
  private timestampUid = 0

  constructor(readonly renderer: Renderer) {}

  get domElement(): RendererDomElementState {
    return this.renderer.domElement
  }

  async init(_renderer: unknown = this.renderer): Promise<void> {}

  beginRender(_renderContext?: unknown): void {}

  finishRender(_renderContext?: unknown): void {}

  beginCompute(_computeGroup?: unknown): void {}

  finishCompute(_computeGroup?: unknown): void {}

  clear(): never {
    throw unsupportedBackendOperationError('Renderer.backend.clear', 'backend render-context clearing')
  }

  _getDefaultRenderPassDescriptor(): never {
    throw unsupportedBackendOperationError(
      'Renderer.backend._getDefaultRenderPassDescriptor',
      'backend default render-pass descriptor creation',
    )
  }

  _getRenderPassDescriptor(): never {
    throw unsupportedBackendOperationError(
      'Renderer.backend._getRenderPassDescriptor',
      'backend render-target pass descriptor creation',
    )
  }

  async resolveOccludedAsync(_renderContext?: unknown): Promise<void> {}

  initTimestampQuery(_renderContext?: unknown, _descriptor?: unknown): void {}

  prepareTimestampBuffer(_renderContext?: unknown, _encoder?: unknown): void {}

  beginBundle(): never {
    throw unsupportedBackendOperationError('Renderer.backend.beginBundle', 'backend render-bundle encoding')
  }

  finishBundle(): never {
    throw unsupportedBackendOperationError('Renderer.backend.finishBundle', 'backend render-bundle encoding')
  }

  addBundle(): never {
    throw unsupportedBackendOperationError('Renderer.backend.addBundle', 'backend render-bundle encoding')
  }

  draw(): never {
    throw unsupportedBackendOperationError('Renderer.backend.draw', 'backend draw commands')
  }

  compute(): never {
    throw unsupportedBackendOperationError('Renderer.backend.compute', 'backend compute dispatch')
  }

  createProgram(): never {
    throw unsupportedBackendOperationError('Renderer.backend.createProgram', 'backend shader program creation')
  }

  destroyProgram(): void {}

  _handleSource(): never {
    throw unsupportedBackendOperationError('Renderer.backend._handleSource', 'backend shader source diagnostics')
  }

  _getShaderErrors(): never {
    throw unsupportedBackendOperationError('Renderer.backend._getShaderErrors', 'backend shader error diagnostics')
  }

  _logProgramError(): never {
    throw unsupportedBackendOperationError('Renderer.backend._logProgramError', 'backend shader program diagnostics')
  }

  _completeCompile(): never {
    throw unsupportedBackendOperationError('Renderer.backend._completeCompile', 'backend shader program compilation')
  }

  createBindings(): never {
    throw unsupportedBackendOperationError('Renderer.backend.createBindings', 'backend bind-group creation')
  }

  updateBindings(): never {
    throw unsupportedBackendOperationError('Renderer.backend.updateBindings', 'backend bind-group updates')
  }

  updateBinding(): never {
    throw unsupportedBackendOperationError('Renderer.backend.updateBinding', 'backend buffer binding updates')
  }

  _setupBindings(): never {
    throw unsupportedBackendOperationError('Renderer.backend._setupBindings', 'backend program binding setup')
  }

  _bindUniforms(): never {
    throw unsupportedBackendOperationError('Renderer.backend._bindUniforms', 'WebGL uniform binding')
  }

  createRenderPipeline(): never {
    throw unsupportedBackendOperationError('Renderer.backend.createRenderPipeline', 'backend render pipeline creation')
  }

  createComputePipeline(): never {
    throw unsupportedBackendOperationError('Renderer.backend.createComputePipeline', 'backend compute pipeline creation')
  }

  needsRenderUpdate(_renderObject?: unknown): boolean {
    return false
  }

  getRenderCacheKey(_renderObject?: unknown): string {
    return 'headless-three-renderer'
  }

  createNodeBuilder(): never {
    throw unsupportedBackendOperationError('Renderer.backend.createNodeBuilder', 'Three.js shader node builders')
  }

  createSampler(): never {
    throw unsupportedBackendOperationError('Renderer.backend.createSampler', 'backend sampler creation')
  }

  updateSampler(_texture?: unknown): string {
    return ''
  }

  destroySampler(): void {}

  createDefaultTexture(): never {
    throw unsupportedBackendOperationError('Renderer.backend.createDefaultTexture', 'backend default texture creation')
  }

  createTexture(): never {
    throw unsupportedBackendOperationError('Renderer.backend.createTexture', 'backend texture creation')
  }

  updateTexture(): never {
    throw unsupportedBackendOperationError('Renderer.backend.updateTexture', 'backend texture uploads')
  }

  generateMipmaps(): never {
    throw unsupportedBackendOperationError('Renderer.backend.generateMipmaps', 'backend mipmap generation')
  }

  destroyTexture(): void {}

  copyTextureToBuffer(): never {
    throw unsupportedBackendOperationError('Renderer.backend.copyTextureToBuffer', 'backend texture-to-buffer readback')
  }

  copyTextureToTexture(): never {
    throw unsupportedBackendOperationError('Renderer.backend.copyTextureToTexture', 'backend texture copies')
  }

  copyFramebufferToTexture(): never {
    throw unsupportedBackendOperationError('Renderer.backend.copyFramebufferToTexture', 'backend framebuffer-to-texture copies')
  }

  createAttribute(): never {
    throw unsupportedBackendOperationError('Renderer.backend.createAttribute', 'backend vertex attribute buffer creation')
  }

  _getVaoKey(): never {
    throw unsupportedBackendOperationError('Renderer.backend._getVaoKey', 'WebGL vertex-array cache keys')
  }

  _createVao(): never {
    throw unsupportedBackendOperationError('Renderer.backend._createVao', 'WebGL vertex-array binding')
  }

  createIndexAttribute(): never {
    throw unsupportedBackendOperationError('Renderer.backend.createIndexAttribute', 'backend index buffer creation')
  }

  createStorageAttribute(): never {
    throw unsupportedBackendOperationError('Renderer.backend.createStorageAttribute', 'backend storage buffer creation')
  }

  createIndirectStorageAttribute(): never {
    throw unsupportedBackendOperationError('Renderer.backend.createIndirectStorageAttribute', 'backend indirect storage buffer creation')
  }

  _getTransformFeedback(): never {
    throw unsupportedBackendOperationError('Renderer.backend._getTransformFeedback', 'WebGL transform-feedback state')
  }

  updateAttribute(): never {
    throw unsupportedBackendOperationError('Renderer.backend.updateAttribute', 'backend attribute buffer updates')
  }

  destroyAttribute(): void {}

  _setFramebuffer(): never {
    throw unsupportedBackendOperationError('Renderer.backend._setFramebuffer', 'WebGL framebuffer configuration')
  }

  getDomElement(): RendererDomElementState {
    return this.renderer.domElement
  }

  getDrawingBufferSize(): RenderSizeLike | null
  getDrawingBufferSize<T extends RenderSizeLike>(target: T): T | null
  getDrawingBufferSize(target?: RenderSizeLike): RenderSizeLike | null {
    return this.renderer.getDrawingBufferSize(target as RenderSizeLike)
  }

  hasFeature(name: unknown): boolean {
    assertRendererProbeName(name, 'Renderer.backend.hasFeature name')
    return false
  }

  async hasFeatureAsync(name: unknown): Promise<boolean> {
    return this.hasFeature(name)
  }

  hasCompatibility(name: unknown): boolean {
    assertRendererProbeName(name, 'Renderer.backend.hasCompatibility name')
    return false
  }

  updateTimeStampUID(abstractRenderContext: unknown): void {
    assertWeakMapKey(abstractRenderContext, 'Renderer.backend.updateTimeStampUID abstractRenderContext')
    this.get(abstractRenderContext).timestampUID = `r:${this.timestampUid += 1}`
  }

  getTimestampUID(abstractRenderContext: unknown): string {
    assertWeakMapKey(abstractRenderContext, 'Renderer.backend.getTimestampUID abstractRenderContext')
    const data = this.get(abstractRenderContext)
    if (typeof data.timestampUID !== 'string') {
      data.timestampUID = `r:${this.timestampUid += 1}`
    }
    return data.timestampUID as string
  }

  getTimestampFrames(type: unknown): number[] {
    assertTimestampQueryType(type, 'Renderer.backend.getTimestampFrames type')
    return []
  }

  _getQueryPool(uid: unknown): null {
    assertTimestampUid(uid, 'Renderer.backend._getQueryPool uid')
    return null
  }

  getTimestamp(uid: unknown): number {
    assertTimestampUid(uid, 'Renderer.backend.getTimestamp uid')
    return 0
  }

  hasTimestamp(uid: unknown): boolean {
    assertTimestampUid(uid, 'Renderer.backend.hasTimestamp uid')
    return false
  }

  async resolveTimestampsAsync(type: unknown = 'render'): Promise<void> {
    assertTimestampQueryType(type, 'Renderer.backend.resolveTimestampsAsync type')
  }

  getMaxAnisotropy(): number {
    return 0
  }

  setScissorTest(value: unknown): void {
    this.renderer.setScissorTest(rendererStateBoolean(value, 'Renderer.backend.setScissorTest value'))
  }

  updateSize(): void {}

  updateViewport(_renderContext?: unknown): void {}

  isOccluded(renderContextOrObject: unknown, object?: unknown): boolean {
    return this.renderer.isOccluded(object === undefined ? renderContextOrObject : object)
  }

  getClearColor(): ThreeColorLike & {
    a: number
    getRGB(target?: ThreeColorLike & { a?: number }, colorSpace?: unknown): ThreeColorLike & { a?: number }
  } {
    const color = this.renderer.getClearColor() as ThreeColorLike & {
      a: number
      getRGB(target?: ThreeColorLike & { a?: number }, colorSpace?: unknown): ThreeColorLike & { a?: number }
    }
    color.a = this.renderer.getClearAlpha()
    color.getRGB = (target = color) => {
      this.renderer.getClearColor(target)
      target.a = this.renderer.getClearAlpha()
      return target
    }
    return color
  }

  getContext(): never {
    throw new Error(
      'Renderer.backend.getContext() is not supported by @headless-three/renderer because it does not expose a browser WebGL or WebGPU context. Render normal Three.js scene graphs with Renderer.render() or renderToTarget().',
    )
  }

  async getArrayBufferAsync(attribute: unknown): Promise<ArrayBuffer> {
    assertStorageBufferAttributeLike(attribute, 'Renderer.backend.getArrayBufferAsync attribute')
    throw new Error(
      'Renderer.backend.getArrayBufferAsync() is not supported by @headless-three/renderer because storage-buffer GPU readback requires backend WebGPU state that this package does not expose. Use Renderer.readRenderTargetPixels() for render-target CPU readback.',
    )
  }

  async resolveTimestampAsync(_renderContext: unknown, type: unknown = 'render'): Promise<number> {
    assertTimestampQueryType(type, 'Renderer.backend.resolveTimestampAsync type')
    throw new Error(
      'Renderer.backend.resolveTimestampAsync() is not supported by @headless-three/renderer because timestamp queries require backend GPU query pools that are outside the scene-oriented API.',
    )
  }

  initRenderTarget(_renderContext?: unknown): void {}

  deleteBindGroupData(_bindGroup?: unknown): void {}

  async waitForGPU(): Promise<void> {
    throw new Error(
      'Renderer.backend.waitForGPU() is not supported by @headless-three/renderer because it does not expose direct GPU task synchronization. Renderer.render() and renderToTarget() return after native scene output readback or target writeback has completed.',
    )
  }

  set(object: unknown, value: unknown): void {
    assertWeakMapKey(object, 'Renderer.backend.set object')
    if (value === null || typeof value !== 'object' || Array.isArray(value)) {
      throw new TypeError('Renderer.backend.set value must be an object.')
    }
    this.data.set(object, value as Record<string, unknown>)
  }

  get(object: unknown): Record<string, unknown> {
    assertWeakMapKey(object, 'Renderer.backend.get object')
    let map = this.data.get(object)
    if (map === undefined) {
      map = {}
      this.data.set(object, map)
    }
    return map
  }

  has(object: unknown): boolean {
    assertWeakMapKey(object, 'Renderer.backend.has object')
    return this.data.has(object)
  }

  delete(object: unknown): void {
    assertWeakMapKey(object, 'Renderer.backend.delete object')
    this.data.delete(object)
  }

  dispose(): void {
    this.data = new WeakMap()
  }
}
