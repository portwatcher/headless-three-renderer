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
import { ACESFilmicToneMapping, RendererInfoState, RendererShadowMapState, WEBGL_COORDINATE_SYSTEM, native } from './index.part-001'
import { RendererDebugState, RendererInspectorState, RendererXrState } from './index.part-002'
import { RendererBackendState } from './index.part-003'
import { RendererLightingState, RendererNodeLibraryState, RendererNodesState } from './index.part-004'
import { RendererDomElementState } from './index.part-005'
import { RendererCapabilitiesState, RendererExtensionsState, RendererPropertiesState, RendererState } from './index.part-006'
import { RendererRenderListsState, RendererRenderStatesState } from './index.part-007'
import { renderer__onDeviceLost_1, renderer_compile_7, renderer_computeAsync_13, renderer_compute_12, renderer_copyFramebufferToTexture_27, renderer_copyTextureToTexture3D_29, renderer_copyTextureToTexture_28, renderer_currentSamples_6, renderer_getArrayBufferAsync_14, renderer_getClearColor_37, renderer_getContext_31, renderer_getCurrentViewport_40, renderer_getDrawingBufferSize_36, renderer_getScissor_41, renderer_getSize_34, renderer_getViewport_39, renderer_highPrecision_5, renderer_inspector_2, renderer_isOccluded_3, renderer_onDeviceLost_0, renderer_renderBufferDirect_9, renderer_renderBufferImmediate_10, renderer_renderObject_11, renderer_resolveTimestampsAsync_15, renderer_setAnimationLoop_30, renderer_setCanvasTarget_19, renderer_setClearAlpha_38, renderer_setDrawingBufferSize_35, renderer_setMRT_17, renderer_setOutputRenderTarget_18, renderer_setRenderObjectFunction_8, renderer_setRenderTargetFramebuffer_26, renderer_setRenderTargetTextures_25, renderer_setRenderTarget_32, renderer_setScissorTest_42, renderer_setSize_33, renderer_setTexture2DArray_24, renderer_setTexture2D_20, renderer_setTexture3D_23, renderer_setTextureCubeDynamic_22, renderer_setTextureCube_21, renderer_sortObjects_4, renderer_waitForGPU_16 } from './index.part-009'
import { renderer_clearTarget_44, renderer_clear_43, renderer_dispose_45, renderer_optionsWithRendererSizeFallback_54, renderer_readRenderTargetPixelsAsync_49, renderer_readRenderTargetPixels_48, renderer_renderCurrentCubeFace_51, renderer_renderCurrentRenderTarget_50, renderer_renderNative_52, renderer_renderToTarget_47, renderer_render_46, renderer_resolveRenderOptions_53 } from './index.part-010'
import { InternalRenderOptions, PixelRect, PixelSize, UnsignedByteType } from './index.part-012'
import { assertDefaultViewportDepthRange, assertRendererParametersLike, rendererStateBoolean, rendererStateClearColor, rendererStateClearDepth, rendererStateClearStencil, rendererStatePixelRatio, rendererStatePixelRect, rendererStatePositiveFiniteNumber, rendererStateSize } from './index.part-014'
import { assertEffectsArrayOrNull, finiteNonNegativeNumber, rendererContextAttributes, rendererStateToneMapping } from './index.part-015'
import { assertRenderTargetLike, assertThreeTextureLike, unsupportedInternalRenderDispatchError } from './index.part-016'
import { assertRendererProbeName, assertSortFunctionOrNull, checkedOutputColorSpace, validateUnsupportedRenderTargetOptions } from './index.part-017'
export class Renderer {
  private native: InstanceType<typeof native.NativeRenderer>
  private readonly sceneExtractionCache: SceneExtractionCache = createSceneExtractionCache()
  private opaqueSort: RenderSortFunction | null = null
  private opaqueValue = true
  private sortObjectsValue = true
  private transparentSort: RenderSortFunction | null = null
  private transparentValue = true
  private currentRenderTarget: RenderTargetLike | null = null
  private currentActiveCubeFace = 0
  private currentActiveMipmapLevel = 0
  private currentSize: PixelSize | null = null
  private currentClearColor: Color4 = [...DEFAULT_BACKGROUND_COLOR] as Color4
  private currentClearDepth = 1
  private currentClearStencil = 0
  private currentViewport: PixelRect | null = null
  private currentScissor: PixelRect | null = null
  private currentScissorTest = false
  private pixelRatioValue = 1
  private autoClearValue = true
  private autoClearColorValue = true
  private autoClearDepthValue = true
  private autoClearStencilValue = true
  private outputColorSpaceValue: RenderOutputColorSpace = 'srgb'
  private localClippingEnabledValue = true
  private toneMappingValue = ACESFilmicToneMapping
  private toneMappingExposureValue = 1
  private transmissionResolutionScaleValue = 1
  private animationLoop: RenderAnimationLoopCallback | null = null
  private inspectorValue: RendererInspectorLike = new RendererInspectorState()
  private isDeviceLostValue = false
  private readonly defaultOnDeviceLost: (info?: unknown) => void = () => {
    this.isDeviceLostValue = true
  }
  private onDeviceLostValue: (info?: unknown) => void = this.defaultOnDeviceLost
  private readonly contextAttributes: RendererContextAttributesLike
  private readonly domElementValue = new RendererDomElementState()

  readonly isRenderer = true
  readonly isWebGLRenderer = true
  readonly isWebGPURenderer = false
  readonly alpha: boolean
  readonly depth: boolean
  readonly stencil: boolean
  readonly logarithmicDepthBuffer = false
  readonly backend: RendererBackendState
  readonly capabilities = new RendererCapabilitiesState()
  clippingPlanes: ThreePlaneLike[] = []
  readonly debug = new RendererDebugState()
  readonly extensions = new RendererExtensionsState()
  readonly info = new RendererInfoState()
  readonly library = new RendererNodeLibraryState()
  readonly lighting = new RendererLightingState()
  readonly nodes: RendererNodesState
  readonly properties = new RendererPropertiesState()
  readonly renderLists = new RendererRenderListsState(this.lighting)
  readonly renderStates = new RendererRenderStatesState()
  readonly reversedDepthBuffer = false
  readonly shadowMap = new RendererShadowMapState()
  readonly state = new RendererState()
  readonly xr = new RendererXrState()

  constructor(parameters?: RendererParametersLike) {
    assertRendererParametersLike(parameters, 'Renderer parameters')
    this.contextAttributes = rendererContextAttributes(parameters)
    this.alpha = this.contextAttributes.alpha
    this.depth = this.contextAttributes.depth
    this.stencil = this.contextAttributes.stencil
    this.backend = new RendererBackendState(this)
    this.nodes = new RendererNodesState(this, this.backend)
    this.native = new native.NativeRenderer()
    this.inspectorValue.setRenderer(this)
  }

  get domElement(): RendererDomElementState {
    return this.domElementValue
  }

  async init(): Promise<this> {
    return this
  }

  get initialized(): boolean {
    return true
  }

  get isDeviceLost(): boolean {
    return this.isDeviceLostValue
  }

  get onDeviceLost(): (info?: unknown) => void {
    return this.onDeviceLostValue
  }

  set onDeviceLost(value: (info?: unknown) => void) { renderer_onDeviceLost_0.call(this, value) }

  _onDeviceLost(info?: unknown): void { return renderer__onDeviceLost_1.call(this, info) }

  get inspector(): RendererInspectorLike {
    return this.inspectorValue
  }

  set inspector(value: RendererInspectorLike) { renderer_inspector_2.call(this, value) }

  get coordinateSystem(): number {
    return WEBGL_COORDINATE_SYSTEM
  }

  getMaxAnisotropy(): number {
    return this.capabilities.getMaxAnisotropy()
  }

  hasFeature(name: unknown): boolean {
    assertRendererProbeName(name, 'Renderer.hasFeature name')
    return false
  }

  async hasFeatureAsync(name: unknown): Promise<boolean> {
    return this.hasFeature(name)
  }

  hasCompatibility(name: unknown): boolean {
    assertRendererProbeName(name, 'Renderer.hasCompatibility name')
    return false
  }

  isOccluded(object: unknown): boolean { return renderer_isOccluded_3.call(this, object) }

  getOutputBufferType(): number {
    return UnsignedByteType
  }

  getColorBufferType(): number {
    return this.getOutputBufferType()
  }

  get autoClear(): boolean {
    return this.autoClearValue
  }

  set autoClear(value: boolean) {
    this.autoClearValue = rendererStateBoolean(value, 'Renderer.autoClear')
  }

  get autoClearColor(): boolean {
    return this.autoClearColorValue
  }

  set autoClearColor(value: boolean) {
    this.autoClearColorValue = rendererStateBoolean(value, 'Renderer.autoClearColor')
  }

  get autoClearDepth(): boolean {
    return this.autoClearDepthValue
  }

  set autoClearDepth(value: boolean) {
    this.autoClearDepthValue = rendererStateBoolean(value, 'Renderer.autoClearDepth')
  }

  get autoClearStencil(): boolean {
    return this.autoClearStencilValue
  }

  set autoClearStencil(value: boolean) {
    this.autoClearStencilValue = rendererStateBoolean(value, 'Renderer.autoClearStencil')
  }

  get sortObjects(): boolean {
    return this.sortObjectsValue
  }

  set sortObjects(value: boolean) { renderer_sortObjects_4.call(this, value) }

  get opaque(): boolean {
    return this.opaqueValue
  }

  set opaque(value: boolean) {
    this.opaqueValue = rendererStateBoolean(value, 'Renderer.opaque')
  }

  get transparent(): boolean {
    return this.transparentValue
  }

  set transparent(value: boolean) {
    this.transparentValue = rendererStateBoolean(value, 'Renderer.transparent')
  }

  get outputColorSpace(): RenderOutputColorSpace {
    return this.outputColorSpaceValue
  }

  set outputColorSpace(value: RenderOutputColorSpace) {
    this.outputColorSpaceValue = checkedOutputColorSpace(value, 'Renderer.outputColorSpace')
  }

  get currentColorSpace(): RenderOutputColorSpace {
    return this.outputColorSpaceValue
  }

  set currentColorSpace(value: RenderOutputColorSpace) {
    this.outputColorSpaceValue = checkedOutputColorSpace(value, 'Renderer.currentColorSpace')
  }

  get _outputColorSpace(): RenderOutputColorSpace {
    return this.outputColorSpaceValue
  }

  set _outputColorSpace(value: RenderOutputColorSpace) {
    this.outputColorSpaceValue = checkedOutputColorSpace(value, 'Renderer._outputColorSpace')
  }

  get toneMapping(): number {
    return this.toneMappingValue
  }

  set toneMapping(value: number) {
    this.toneMappingValue = rendererStateToneMapping(value)
  }

  get currentToneMapping(): number {
    return this.toneMappingValue
  }

  set currentToneMapping(value: number) {
    this.toneMappingValue = rendererStateToneMapping(value)
  }

  get toneMappingExposure(): number {
    return this.toneMappingExposureValue
  }

  set toneMappingExposure(value: number) {
    this.toneMappingExposureValue = finiteNonNegativeNumber(value, 'Renderer.toneMappingExposure')
  }

  get transmissionResolutionScale(): number {
    return this.transmissionResolutionScaleValue
  }

  set transmissionResolutionScale(value: number) {
    this.transmissionResolutionScaleValue = rendererStatePositiveFiniteNumber(value, 'Renderer.transmissionResolutionScale')
  }

  get localClippingEnabled(): boolean {
    return this.localClippingEnabledValue
  }

  set localClippingEnabled(value: boolean) {
    this.localClippingEnabledValue = rendererStateBoolean(value, 'Renderer.localClippingEnabled')
  }

  get highPrecision(): boolean {
    return false
  }

  set highPrecision(value: boolean) { renderer_highPrecision_5.call(this, value) }

  get samples(): number {
    return 0
  }

  get needsFrameBufferTarget(): boolean {
    return false
  }

  get currentSamples(): number { return renderer_currentSamples_6.call(this) }

  get isOutputTarget(): boolean {
    return this.currentRenderTarget === null
  }

  setOpaqueSort(method: RenderSortFunction | null): void {
    assertSortFunctionOrNull(method, 'Renderer.setOpaqueSort')
    this.opaqueSort = method
  }

  setTransparentSort(method: RenderSortFunction | null): void {
    assertSortFunctionOrNull(method, 'Renderer.setTransparentSort')
    this.transparentSort = method
  }

  compile(
    scene: ThreeSceneRootLike,
    camera: ThreeRenderCameraLike,
    targetScene: ThreeSceneRootLike | null = null,
  ): Set<ThreeMaterialLike> { return renderer_compile_7.call(this, scene, camera, targetScene) }

  async compileAsync(
    scene: ThreeSceneRootLike,
    camera: ThreeRenderCameraLike,
    targetScene: ThreeSceneRootLike | null = null,
  ): Promise<Set<ThreeMaterialLike>> {
    return this.compile(scene, camera, targetScene)
  }

  setEffects(effects: readonly unknown[] | null = null): void {
    assertEffectsArrayOrNull(effects, 'Renderer.setEffects effects')
  }

  setRenderObjectFunction(renderObjectFunction: ((...args: unknown[]) => unknown) | null): void { return renderer_setRenderObjectFunction_8.call(this, renderObjectFunction) }

  getRenderObjectFunction(): null {
    return null
  }

  renderBufferDirect(): never { return renderer_renderBufferDirect_9.call(this) }

  renderBufferImmediate(): never { return renderer_renderBufferImmediate_10.call(this) }

  renderObject(): never { return renderer_renderObject_11.call(this) }

  _getFrameBufferTarget(): never {
    throw unsupportedInternalRenderDispatchError('Renderer._getFrameBufferTarget')
  }

  _renderScene(): never {
    throw unsupportedInternalRenderDispatchError('Renderer._renderScene')
  }

  _projectObject(): never {
    throw unsupportedInternalRenderDispatchError('Renderer._projectObject')
  }

  _renderBundles(): never {
    throw unsupportedInternalRenderDispatchError('Renderer._renderBundles')
  }

  _renderBundle(): never {
    throw unsupportedInternalRenderDispatchError('Renderer._renderBundle')
  }

  _renderTransparents(): never {
    throw unsupportedInternalRenderDispatchError('Renderer._renderTransparents')
  }

  _renderObjects(): never {
    throw unsupportedInternalRenderDispatchError('Renderer._renderObjects')
  }

  _renderObjectDirect(): never {
    throw unsupportedInternalRenderDispatchError('Renderer._renderObjectDirect')
  }

  _renderOutput(): never {
    throw unsupportedInternalRenderDispatchError('Renderer._renderOutput')
  }

  _createObjectPipeline(): never {
    throw unsupportedInternalRenderDispatchError('Renderer._createObjectPipeline')
  }

  _getShadowNodes(): never {
    throw unsupportedInternalRenderDispatchError('Renderer._getShadowNodes')
  }

  _onCanvasTargetResize(): void {}

  _setXRLayerSize(width: unknown, height: unknown): void {
    const size = rendererStateSize(width, height, 'Renderer._setXRLayerSize')
    this.setViewport(0, 0, size.width, size.height)
  }

  _resetXRState(): void {
    this.setRenderTarget(null)
  }

  compute(computeNodes: unknown, dispatchSize: unknown = null): never { return renderer_compute_12.call(this, computeNodes, dispatchSize) }

  async computeAsync(computeNodes: unknown, dispatchSize: unknown = null): Promise<never> { return renderer_computeAsync_13.call(this, computeNodes, dispatchSize) }

  async getArrayBufferAsync(attribute: unknown): Promise<ArrayBuffer> { return renderer_getArrayBufferAsync_14.call(this, attribute) }

  async resolveTimestampsAsync(type: unknown = 'render'): Promise<number> { return renderer_resolveTimestampsAsync_15.call(this, type) }

  async waitForGPU(): Promise<void> { return renderer_waitForGPU_16.call(this) }

  setMRT(mrt: unknown = null): this { return renderer_setMRT_17.call(this, mrt) as this }

  getMRT(): null {
    return null
  }

  setOutputRenderTarget(renderTarget: unknown = null): void { return renderer_setOutputRenderTarget_18.call(this, renderTarget) }

  getOutputRenderTarget(): null {
    return null
  }

  setCanvasTarget(canvasTarget: unknown = null): void { return renderer_setCanvasTarget_19.call(this, canvasTarget) }

  getCanvasTarget(): null {
    return null
  }

  setTexture2D(texture: unknown, slot: unknown): never { return renderer_setTexture2D_20.call(this, texture, slot) }

  setTextureCube(texture: unknown, slot: unknown): never { return renderer_setTextureCube_21.call(this, texture, slot) }

  setTextureCubeDynamic(texture: unknown, slot: unknown): never { return renderer_setTextureCubeDynamic_22.call(this, texture, slot) }

  setTexture3D(texture: unknown, slot: unknown): never { return renderer_setTexture3D_23.call(this, texture, slot) }

  setTexture2DArray(texture: unknown, slot: unknown): never { return renderer_setTexture2DArray_24.call(this, texture, slot) }

  initRenderTarget(target: RenderTargetLike): void {
    assertRenderTargetLike(target, 'Renderer.initRenderTarget target')
    validateUnsupportedRenderTargetOptions(target)
  }

  initTexture(texture: ThreeTextureLike): void {
    assertThreeTextureLike(texture, 'Renderer.initTexture texture')
  }

  async initTextureAsync(texture: ThreeTextureLike): Promise<void> {
    this.initTexture(texture)
  }

  hasInitialized(): boolean {
    return true
  }

  setRenderTargetTextures(renderTarget: RenderTargetLike, colorTexture: unknown, depthTexture: unknown = null): never { return renderer_setRenderTargetTextures_25.call(this, renderTarget, colorTexture, depthTexture) }

  setRenderTargetFramebuffer(renderTarget: RenderTargetLike, defaultFramebuffer?: unknown): never { return renderer_setRenderTargetFramebuffer_26.call(this, renderTarget, defaultFramebuffer) }

  copyFramebufferToTexture(texture: ThreeTextureLike, position?: unknown, level?: number): void
  copyFramebufferToTexture(position: unknown, texture: ThreeTextureLike, level?: number): void
  copyFramebufferToTexture(textureOrPosition: unknown, positionOrTexture: unknown = null, level = 0): void { return renderer_copyFramebufferToTexture_27.call(this, textureOrPosition, positionOrTexture, level) }

  copyTextureToTexture(
    srcTexture: ThreeTextureLike,
    dstTexture: ThreeTextureLike,
    srcRegion?: unknown,
    dstPosition?: unknown,
    srcLevel?: number,
    dstLevel?: number | null,
  ): void
  copyTextureToTexture(
    dstPosition: unknown,
    srcTexture: ThreeTextureLike,
    dstTexture: ThreeTextureLike,
    dstLevel?: number,
  ): void
  copyTextureToTexture(
    srcTextureOrDstPosition: unknown,
    dstTextureOrSrcTexture: unknown,
    srcRegionOrDstTexture: unknown = null,
    dstPositionOrDstLevel: unknown = null,
    srcLevel = 0,
    dstLevel: number | null = null,
  ): void { return renderer_copyTextureToTexture_28.call(this, srcTextureOrDstPosition, dstTextureOrSrcTexture, srcRegionOrDstTexture, dstPositionOrDstLevel, srcLevel, dstLevel) }

  copyTextureToTexture3D(
    srcTexture: ThreeTextureLike,
    dstTexture: ThreeTextureLike,
    srcRegion?: unknown,
    dstPosition?: unknown,
    level?: number,
  ): never
  copyTextureToTexture3D(
    srcRegion: unknown,
    dstPosition: unknown,
    srcTexture: ThreeTextureLike,
    dstTexture: ThreeTextureLike,
    level?: number,
  ): never
  copyTextureToTexture3D(
    srcTextureOrSrcRegion: unknown,
    dstTextureOrDstPosition: unknown,
    _srcRegionOrSrcTexture: unknown = null,
    _dstPositionOrDstTexture: unknown = null,
    level = 0,
  ): never { return renderer_copyTextureToTexture3D_29.call(this, srcTextureOrSrcRegion, dstTextureOrDstPosition, _srcRegionOrSrcTexture, _dstPositionOrDstTexture, level) }

  setAnimationLoop(callback: RenderAnimationLoopCallback | null): void { return renderer_setAnimationLoop_30.call(this, callback) }

  getAnimationLoop(): RenderAnimationLoopCallback | null {
    return this.animationLoop
  }

  getContext(): never { return renderer_getContext_31.call(this) }

  getContextAttributes(): RendererContextAttributesLike {
    return { ...this.contextAttributes }
  }

  forceContextLoss(): void {
    // There is no browser WebGL context; native resources follow renderer object lifetime.
  }

  forceContextRestore(): void {
    // Native render state is recreated per pass, so there is no persistent WebGL context to restore.
    this.isDeviceLostValue = false
  }

  getRenderTarget(): RenderTargetLike | null {
    return this.currentRenderTarget
  }

  getActiveCubeFace(): number {
    return this.currentActiveCubeFace
  }

  getActiveMipmapLevel(): number {
    return this.currentActiveMipmapLevel
  }

  setRenderTarget(target: RenderTargetLike | null = null, activeCubeFace = 0, activeMipmapLevel = 0): void { return renderer_setRenderTarget_32.call(this, target, activeCubeFace, activeMipmapLevel) }

  setSize(width: number, height: number, updateStyle = true): void { return renderer_setSize_33.call(this, width, height, updateStyle) }

  setPixelRatio(value?: number): void {
    if (value === undefined) return
    this.pixelRatioValue = rendererStatePixelRatio(value, 'Renderer.setPixelRatio')
  }

  getPixelRatio(): number {
    return this.pixelRatioValue
  }

  getSize(): RenderSizeLike | null
  getSize<T extends RenderSizeLike>(target: T): T | null
  getSize(target?: RenderSizeLike): RenderSizeLike | null { return renderer_getSize_34.call(this, target) }

  setDrawingBufferSize(width: number, height: number, pixelRatio: number): void { return renderer_setDrawingBufferSize_35.call(this, width, height, pixelRatio) }

  getDrawingBufferSize(): RenderSizeLike | null
  getDrawingBufferSize<T extends RenderSizeLike>(target: T): T | null
  getDrawingBufferSize(target?: RenderSizeLike): RenderSizeLike | null { return renderer_getDrawingBufferSize_36.call(this, target) }

  setClearColor(color: number | string | ThreeColorLike | number[], alpha?: number): void {
    this.currentClearColor = rendererStateClearColor(color, alpha)
  }

  getClearColor(): ThreeColorLike
  getClearColor<T extends ThreeColorLike>(target: T): T
  getClearColor(target?: ThreeColorLike): ThreeColorLike { return renderer_getClearColor_37.call(this, target) }

  setClearAlpha(alpha: number): void { return renderer_setClearAlpha_38.call(this, alpha) }

  getClearAlpha(): number {
    return this.currentClearColor[3]
  }

  setClearDepth(depth: number): void {
    this.currentClearDepth = rendererStateClearDepth(depth, 'Renderer.setClearDepth depth')
  }

  getClearDepth(): number {
    return this.currentClearDepth
  }

  setClearStencil(stencil: number): void {
    this.currentClearStencil = rendererStateClearStencil(stencil, 'Renderer.setClearStencil stencil')
  }

  getClearStencil(): number {
    return this.currentClearStencil
  }

  setViewport(rect: RenderPixelRectLike | null): void
  setViewport(x: number, y: number, width: number, height: number, minDepth?: number, maxDepth?: number): void
  setViewport(rectOrX: RenderPixelRectLike | null | number, y?: number, width?: number, height?: number, minDepth = 0, maxDepth = 1): void {
    assertDefaultViewportDepthRange(minDepth, maxDepth, 'Renderer.setViewport')
    this.currentViewport = rendererStatePixelRect(rectOrX, y, width, height, 'Renderer.setViewport')
  }

  getViewport(): RenderPixelRectLike | null
  getViewport<T extends RenderPixelRectLike>(target: T): T | null
  getViewport(target?: RenderPixelRectLike): RenderPixelRectLike | null { return renderer_getViewport_39.call(this, target) }

  getCurrentViewport(): RenderPixelRectLike | null
  getCurrentViewport<T extends RenderPixelRectLike>(target: T): T | null
  getCurrentViewport(target?: RenderPixelRectLike): RenderPixelRectLike | null { return renderer_getCurrentViewport_40.call(this, target) }

  setScissor(rect: RenderPixelRectLike | null): void
  setScissor(x: number, y: number, width: number, height: number): void
  setScissor(rectOrX: RenderPixelRectLike | null | number, y?: number, width?: number, height?: number): void {
    this.currentScissor = rendererStatePixelRect(rectOrX, y, width, height, 'Renderer.setScissor')
  }

  getScissor(): RenderPixelRectLike | null
  getScissor<T extends RenderPixelRectLike>(target: T): T | null
  getScissor(target?: RenderPixelRectLike): RenderPixelRectLike | null { return renderer_getScissor_41.call(this, target) }

  setScissorTest(enabled: boolean): void { return renderer_setScissorTest_42.call(this, enabled) }

  getScissorTest(): boolean {
    return this.currentScissorTest
  }

  clear(color = true, depth = true, stencil = true): void { return renderer_clear_43.call(this, color, depth, stencil) }

  async clearAsync(color = true, depth = true, stencil = true): Promise<void> {
    this.clear(color, depth, stencil)
  }

  clearTarget(target: RenderTargetLike | null, color = true, depth = true, stencil = true): void { return renderer_clearTarget_44.call(this, target, color, depth, stencil) }

  clearColor(): void {
    this.clear(true, false, false)
  }

  async clearColorAsync(): Promise<void> {
    this.clearColor()
  }

  clearDepth(): void {
    this.clear(false, true, false)
  }

  async clearDepthAsync(): Promise<void> {
    this.clearDepth()
  }

  clearStencil(): void {
    this.clear(false, false, true)
  }

  async clearStencilAsync(): Promise<void> {
    this.clearStencil()
  }

  dispose(): void { return renderer_dispose_45.call(this) }

  resetState(): void {
    // Native render state is rebuilt for each pass, so there is no persistent GL state to reset.
  }

  resetGLState(): void {
    this.resetState()
  }

  render(scene: ThreeSceneRootLike, camera: ThreeRenderCameraLike, options: RenderOptions = {}): Buffer { return renderer_render_46.call(this, scene, camera, options) }

  async renderAsync(scene: ThreeSceneRootLike, camera: ThreeRenderCameraLike, options: RenderOptions = {}): Promise<Buffer> {
    return this.render(scene, camera, options)
  }

  renderToTarget(
    scene: ThreeSceneRootLike,
    camera: ThreeRenderCameraLike,
    target: RenderTargetLike = {},
    options: RenderOptions = {},
  ): RenderTargetLike { return renderer_renderToTarget_47.call(this, scene, camera, target, options) }

  readRenderTargetPixels(
    target: RenderTargetLike,
    x: number,
    y: number,
    width: number,
    height: number,
    buffer: NonNullable<RenderTargetImageLike['data']>,
    activeCubeFaceIndex?: number,
    textureIndex = 0,
  ): void { return renderer_readRenderTargetPixels_48.call(this, target, x, y, width, height, buffer, activeCubeFaceIndex, textureIndex) }

  async readRenderTargetPixelsAsync(
    target: RenderTargetLike,
    x: number,
    y: number,
    width: number,
    height: number,
    bufferOrTextureIndex?: NonNullable<RenderTargetImageLike['data']> | number,
    activeCubeFaceIndexOrFaceIndex?: number,
    textureIndex = 0,
  ): Promise<NonNullable<RenderTargetImageLike['data']>> { return renderer_readRenderTargetPixelsAsync_49.call(this, target, x, y, width, height, bufferOrTextureIndex, activeCubeFaceIndexOrFaceIndex, textureIndex) }

  private renderCurrentRenderTarget(
    scene: ThreeSceneRootLike,
    camera: ThreeCameraLike,
    options: RenderOptions,
  ): Buffer { return renderer_renderCurrentRenderTarget_50.call(this, scene, camera, options) }

  private renderCurrentCubeFace(
    scene: ThreeSceneRootLike,
    camera: ThreeCameraLike,
    target: RenderTargetLike,
    options: RenderOptions,
  ): Buffer { return renderer_renderCurrentCubeFace_51.call(this, scene, camera, target, options) }

  private renderNative(
    scene: ThreeSceneRootLike,
    camera: ThreeCameraLike,
    options: RenderOptions,
  ): { buffer: Buffer; nativeScene: NativeRenderScene; nativeCamera: NativeCamera; objectIdEntries?: RenderObjectIdEntry[] } { return renderer_renderNative_52.call(this, scene, camera, options) }

  private resolveRenderOptions(options: RenderOptions, fallbackTarget: RenderTargetLike | null | undefined = options.target): InternalRenderOptions { return renderer_resolveRenderOptions_53.call(this, options, fallbackTarget) }

  private optionsWithRendererSizeFallback(
    options: RenderOptions,
    fallbackTarget: RenderTargetLike | null | undefined,
  ): RenderOptions { return renderer_optionsWithRendererSizeFallback_54.call(this, options, fallbackTarget) }
}
