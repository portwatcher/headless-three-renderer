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

// eslint-disable-next-line @typescript-eslint/no-var-requires
const native = require('../native.js')

import { resolveSize, cameraViewProjection, cameraViewMatrix, cameraWorldPosition } from './camera'
import { DEFAULT_BACKGROUND_COLOR, cssColorStringToArray, resolveBackground, validatedColorLikeToArray } from './color'
import { createSceneExtractionCache, flattenScene, type SceneExtractionCache, type ShadowMaterialMode } from './scene'
import { extractLights, extractAmbientLight, extractAmbientIntensity, extractLightProbe } from './lights'
import { canvasLikeImageToRgba, extractBackgroundTexture, extractTextureData, isCompressedTextureFormat, resolveEnvironmentMap, resolveSceneOverrideMaterial, type MaterialExtractionContext } from './materials'
import { extractClippingPlanes } from './clipping'
import { validateObjectChildrenTree } from './objects'
import { clamp01, matrixElements } from './math'

const WEBGL_COORDINATE_SYSTEM = 2000
const BasicShadowMap = 0
const PCFShadowMap = 1
const PCFSoftShadowMap = 2
const VSMShadowMap = 3
const SupportedRendererShadowMapTypes = new Set([BasicShadowMap, PCFShadowMap, PCFSoftShadowMap, VSMShadowMap])
const CullFaceNone = 0
const CullFaceBack = 1
const CullFaceFront = 2
const CullFaceFrontBack = 3
const SupportedRendererStateCullFaces = new Set([CullFaceNone, CullFaceBack, CullFaceFront, CullFaceFrontBack])
const NoBlending = 0
const NormalBlending = 1
const AdditiveBlending = 2
const SubtractiveBlending = 3
const MultiplyBlending = 4
const CustomBlending = 5
const SupportedRendererStateBlendingModes = new Set([
  NoBlending,
  NormalBlending,
  AdditiveBlending,
  SubtractiveBlending,
  MultiplyBlending,
  CustomBlending,
])
const WebGLDrawModePoints = 0x0000
const WebGLDrawModeLines = 0x0001
const WebGLDrawModeLineLoop = 0x0002
const WebGLDrawModeLineStrip = 0x0003
const WebGLDrawModeTriangles = 0x0004
const SupportedRendererInfoDrawModes = new Set([
  WebGLDrawModePoints,
  WebGLDrawModeLines,
  WebGLDrawModeLineLoop,
  WebGLDrawModeLineStrip,
  WebGLDrawModeTriangles,
])
const NoToneMapping = 0
const LinearToneMapping = 1
const ReinhardToneMapping = 2
const CineonToneMapping = 3
const ACESFilmicToneMapping = 4
const CustomToneMapping = 5
const AgXToneMapping = 6
const NeutralToneMapping = 7
const SupportedRendererToneMappings = new Set([
  NoToneMapping,
  LinearToneMapping,
  ReinhardToneMapping,
  CineonToneMapping,
  ACESFilmicToneMapping,
  CustomToneMapping,
  AgXToneMapping,
  NeutralToneMapping,
])
const SupportedRendererPowerPreferences = new Set(['default', 'high-performance', 'low-power'])
const SupportedTimestampQueryTypes = new Set(['render', 'compute'])
const RendererBooleanParameters = [
  'alpha',
  'depth',
  'stencil',
  'antialias',
  'premultipliedAlpha',
  'preserveDrawingBuffer',
  'failIfMajorPerformanceCaveat',
] as const
const DefaultRendererContextAttributes: RendererContextAttributesLike = {
  alpha: false,
  depth: true,
  stencil: false,
  antialias: false,
  premultipliedAlpha: true,
  preserveDrawingBuffer: false,
  powerPreference: 'default',
  failIfMajorPerformanceCaveat: false,
}
const RendererInspectorOptionalMethods = [
  'getRenderer',
  'init',
  'begin',
  'finish',
  'inspect',
  'computeAsync',
  'beginCompute',
  'finishCompute',
  'beginRender',
  'finishRender',
  'copyTextureToTexture',
  'copyFramebufferToTexture',
] as const

export {
  applyVrmAnimation,
  EncodedImageTextureLoader,
  createEncodedImageTextureLoader,
  createNodeGltfLoader,
  installLocalFileFetch,
  loadGltfFromFile,
  loadVrmAnimationFromFile,
  loadVrmFromFile,
  resolveLocalAssetPath,
} from './loaders'

export type {
  AppliedVrmAnimation,
  AnimationMixerConstructor,
  ApplyVrmAnimationOptions,
  ConfigureGltfLoader,
  LoadGltfFromFileOptions,
  LoadVrmAnimationFromFileOptions,
  LoadVrmFromFileOptions,
  NodeGltfLoaderBundle,
  NodeGltfLoaderOptions,
  ThreeGltfLoaderLike,
  ThreeLoadingManagerLike,
  VrmAnimationActionLike,
  VrmAnimationClipFactory,
  VrmAnimationMixerLike,
  VrmLoaderPluginConstructor,
} from './loaders'

export type {
  RenderOutputFormat,
  RenderOutputColorSpace,
  RenderMode,
  ThreeColorLike,
  ThreeMatrix4Like,
  ThreeBufferAttributeLike,
  ThreeBufferGeometryLike,
  ThreeTextureLike,
  ThreeVector3Like,
  ThreeEulerLike,
  ThreePlaneLike,
  RenderPixelRectLike,
  RenderSizeLike,
  ThreeLayersLike,
  ThreeMaterialLike,
  ThreeBoneLike,
  ThreeSkeletonLike,
  ThreeObject3DLike,
  ThreeSceneRootLike,
  ThreeSceneLike,
  ThreeCameraLike,
  ThreeCubeCameraLike,
  ThreeRenderCameraLike,
  RenderOptions,
  RenderTargetLike,
  RenderObjectIdEntry,
  RenderAnimationLoopCallback,
  RendererParametersLike,
  RendererContextAttributesLike,
  RendererPowerPreferenceLike,
  RendererInspectorLike,
  RenderSortFunction,
  RenderSortItem,
  PostProcessingOptions,
} from './types'

class RendererShadowMapState {
  private enabledValue = true
  private autoUpdateValue = true
  private needsUpdateValue = false
  private transmittedValue = false
  private typeValue = PCFShadowMap

  get enabled(): boolean {
    return this.enabledValue
  }

  set enabled(value: boolean) {
    this.enabledValue = rendererStateBoolean(value, 'Renderer.shadowMap.enabled')
  }

  get autoUpdate(): boolean {
    return this.autoUpdateValue
  }

  set autoUpdate(value: boolean) {
    this.autoUpdateValue = rendererStateBoolean(value, 'Renderer.shadowMap.autoUpdate')
  }

  get needsUpdate(): boolean {
    return this.needsUpdateValue
  }

  set needsUpdate(value: boolean) {
    this.needsUpdateValue = rendererStateBoolean(value, 'Renderer.shadowMap.needsUpdate')
  }

  get transmitted(): boolean {
    return this.transmittedValue
  }

  set transmitted(value: boolean) {
    this.transmittedValue = rendererStateBoolean(value, 'Renderer.shadowMap.transmitted')
  }

  get type(): number {
    return this.typeValue
  }

  set type(value: number) {
    this.typeValue = rendererStateShadowMapType(value)
  }
}

class RendererInfoState {
  private autoResetValue = true

  calls = 0
  frame = 0

  readonly memory = {
    geometries: 0,
    textures: 0,
  }

  readonly render = {
    calls: 0,
    frameCalls: 0,
    drawCalls: 0,
    triangles: 0,
    points: 0,
    lines: 0,
    timestamp: 0,
    previousFrameCalls: 0,
    timestampCalls: 0,
    frame: 0,
  }

  readonly compute = {
    calls: 0,
    frameCalls: 0,
    timestamp: 0,
    previousFrameCalls: 0,
    timestampCalls: 0,
  }

  programs: unknown[] | null = null

  get autoReset(): boolean {
    return this.autoResetValue
  }

  set autoReset(value: boolean) {
    this.autoResetValue = rendererStateBoolean(value, 'Renderer.info.autoReset')
  }

  update(objectOrCount: unknown, modeOrCount: unknown, instanceCount: unknown = 1): void {
    if (objectOrCount !== null && typeof objectOrCount === 'object' && !Array.isArray(objectOrCount)) {
      this.updateCommonRendererObject(objectOrCount as ThreeObject3DLike, modeOrCount, instanceCount)
      return
    }

    const drawCount = rendererInfoDrawCount(objectOrCount, 'Renderer.info.update count')
    const drawMode = rendererInfoDrawMode(modeOrCount, 'Renderer.info.update mode')
    const instances = rendererInfoInstanceCount(instanceCount, 'Renderer.info.update instanceCount')

    this.render.calls += 1
    this.render.drawCalls += 1

    switch (drawMode) {
      case WebGLDrawModeTriangles:
        this.render.triangles += instances * (drawCount / 3)
        break
      case WebGLDrawModeLines:
        this.render.lines += instances * (drawCount / 2)
        break
      case WebGLDrawModeLineStrip:
        this.render.lines += instances * (drawCount - 1)
        break
      case WebGLDrawModeLineLoop:
        this.render.lines += instances * drawCount
        break
      case WebGLDrawModePoints:
        this.render.points += instances * drawCount
        break
    }
  }

  reset(): void {
    this.render.previousFrameCalls = this.render.frameCalls
    this.compute.previousFrameCalls = this.compute.frameCalls
    this.render.calls = 0
    this.render.frameCalls = 0
    this.render.drawCalls = 0
    this.render.triangles = 0
    this.render.points = 0
    this.render.lines = 0
    this.compute.frameCalls = 0
  }

  dispose(): void {
    this.reset()
    this.calls = 0
    this.compute.calls = 0
    this.render.timestamp = 0
    this.render.previousFrameCalls = 0
    this.render.timestampCalls = 0
    this.compute.timestamp = 0
    this.compute.previousFrameCalls = 0
    this.compute.timestampCalls = 0
    this.memory.geometries = 0
    this.memory.textures = 0
  }

  updateTimestamp(type: unknown, time: unknown): void {
    assertTimestampQueryType(type, 'Renderer.info.updateTimestamp type')
    const elapsed = rendererInfoTimestampTime(time, 'Renderer.info.updateTimestamp time')
    const target = type === 'render' ? this.render : this.compute

    if (target.timestampCalls === 0) {
      target.timestamp = 0
    }

    target.timestamp += elapsed
    target.timestampCalls += 1

    if (target.timestampCalls >= target.previousFrameCalls) {
      target.timestampCalls = 0
    }
  }

  private updateCommonRendererObject(object: ThreeObject3DLike, count: unknown, instanceCount: unknown): void {
    const drawCount = rendererInfoDrawCount(count, 'Renderer.info.update count')
    const instances = rendererInfoInstanceCount(instanceCount, 'Renderer.info.update instanceCount')

    this.render.drawCalls += 1

    if (object.isMesh === true || object.isSprite === true) {
      this.render.triangles += instances * (drawCount / 3)
    } else if (object.isPoints === true) {
      this.render.points += instances * drawCount
    } else if (object.isLineSegments === true) {
      this.render.lines += instances * (drawCount / 2)
    } else if (object.isLineLoop === true) {
      this.render.lines += instances * drawCount
    } else if (object.isLine === true) {
      this.render.lines += instances * (drawCount - 1)
    } else {
      throw new Error(
        'Renderer.info.update object type is not supported. Use a mesh, sprite, points, line, line segments, or line loop object.',
      )
    }
  }
}

class RendererXrState {
  private enabledValue = false
  private cameraAutoUpdateValue = true
  private referenceSpaceTypeValue = 'local-floor'
  private framebufferScaleFactorValue = 1
  private foveationValue: number | undefined
  private readonly listeners = new Map<string, Set<(event: unknown) => void>>()

  readonly isPresenting = false

  get enabled(): boolean {
    return this.enabledValue
  }

  set enabled(value: boolean) {
    this.enabledValue = rendererStateBoolean(value, 'Renderer.xr.enabled')
  }

  get cameraAutoUpdate(): boolean {
    return this.cameraAutoUpdateValue
  }

  set cameraAutoUpdate(value: boolean) {
    this.cameraAutoUpdateValue = rendererStateBoolean(value, 'Renderer.xr.cameraAutoUpdate')
  }

  setFramebufferScaleFactor(value: unknown): void {
    this.framebufferScaleFactorValue = rendererStatePositiveFiniteNumber(value, 'Renderer.xr.setFramebufferScaleFactor value')
  }

  getController(index: unknown): never {
    assertXrInputIndex(index, 'Renderer.xr.getController index')
    throw new Error(
      'Renderer.xr.getController() is not supported by @headless-three/renderer because it does not provide a browser WebXR runtime.',
    )
  }

  getControllerGrip(index: unknown): never {
    assertXrInputIndex(index, 'Renderer.xr.getControllerGrip index')
    throw new Error(
      'Renderer.xr.getControllerGrip() is not supported by @headless-three/renderer because it does not provide a browser WebXR runtime.',
    )
  }

  getHand(index: unknown): never {
    assertXrInputIndex(index, 'Renderer.xr.getHand index')
    throw new Error(
      'Renderer.xr.getHand() is not supported by @headless-three/renderer because it does not provide a browser WebXR runtime.',
    )
  }

  getSession(): null {
    return null
  }

  getReferenceSpace(): null {
    return null
  }

  getReferenceSpaceType(): string {
    return this.referenceSpaceTypeValue
  }

  setReferenceSpaceType(type: unknown): void {
    if (typeof type !== 'string' || type.length === 0) {
      throw new TypeError('Renderer.xr.setReferenceSpaceType type must be a non-empty string.')
    }
    this.referenceSpaceTypeValue = type
  }

  setReferenceSpace(space: unknown): never {
    if (space === null || typeof space !== 'object' || Array.isArray(space)) {
      throw new TypeError('Renderer.xr.setReferenceSpace space must be a WebXR reference-space-like object.')
    }
    throw new Error(
      'Renderer.xr.setReferenceSpace() is not supported by @headless-three/renderer because it does not provide a browser WebXR runtime.',
    )
  }

  getBaseLayer(): null {
    return null
  }

  getBinding(): null {
    return null
  }

  getFrame(): null {
    return null
  }

  getEnvironmentBlendMode(): 'opaque' {
    return 'opaque'
  }

  getDepthTexture(): null {
    return null
  }

  hasDepthSensing(): boolean {
    return false
  }

  getDepthSensingMesh(): null {
    return null
  }

  getCamera(): null {
    return null
  }

  getCameraTexture(camera: unknown): null {
    if (camera === null || typeof camera !== 'object' || Array.isArray(camera)) {
      throw new TypeError('Renderer.xr.getCameraTexture camera must be an XR camera-like object.')
    }
    return null
  }

  getFoveation(): number | undefined {
    return this.foveationValue
  }

  setFoveation(value: unknown): void {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      throw new TypeError('Renderer.xr.setFoveation value must be a finite number.')
    }
    if (value < 0 || value > 1) {
      throw new RangeError('Renderer.xr.setFoveation value must be between 0 and 1.')
    }
    this.foveationValue = value
  }

  updateCamera(camera: ThreeRenderCameraLike): void {
    validateTopLevelRenderCamera(camera)
  }

  setAnimationLoop(callback: RenderAnimationLoopCallback | null): void {
    if (callback !== null && typeof callback !== 'function') {
      throw new TypeError('Renderer.xr.setAnimationLoop callback must be a function or null.')
    }
  }

  addEventListener(type: unknown, listener: unknown): void {
    assertEventListener(type, listener, 'Renderer.xr.addEventListener')
    const eventType = type as string
    const eventListener = listener as (event: unknown) => void
    let typeListeners = this.listeners.get(eventType)
    if (!typeListeners) {
      typeListeners = new Set()
      this.listeners.set(eventType, typeListeners)
    }
    typeListeners.add(eventListener)
  }

  removeEventListener(type: unknown, listener: unknown): void {
    assertEventListener(type, listener, 'Renderer.xr.removeEventListener')
    this.listeners.get(type as string)?.delete(listener as (event: unknown) => void)
  }

  hasEventListener(type: unknown, listener: unknown): boolean {
    assertEventListener(type, listener, 'Renderer.xr.hasEventListener')
    return this.listeners.get(type as string)?.has(listener as (event: unknown) => void) ?? false
  }

  dispatchEvent(event: unknown): void {
    if (event == null || typeof event !== 'object' || Array.isArray(event)) {
      throw new TypeError('Renderer.xr.dispatchEvent event must be an event-like object.')
    }
    const type = (event as { type?: unknown }).type
    if (typeof type !== 'string' || type.length === 0) {
      throw new TypeError('Renderer.xr.dispatchEvent event.type must be a non-empty string.')
    }
    for (const listener of [...(this.listeners.get(type) ?? [])]) {
      listener.call(this, event)
    }
  }

  async setSession(session: unknown): Promise<never> {
    if (session === null || typeof session !== 'object') {
      throw new TypeError('Renderer.xr.setSession session must be a WebXR session-like object.')
    }
    throw new Error(
      'Renderer.xr.setSession() is not supported by @headless-three/renderer because it does not provide a browser WebXR runtime.',
    )
  }

  dispose(): void {
    // XR resources are not allocated by the headless renderer.
  }
}

class RendererDebugState {
  private checkShaderErrorsValue = true
  private onShaderErrorValue: ((...args: unknown[]) => void) | null = null

  get checkShaderErrors(): boolean {
    return this.checkShaderErrorsValue
  }

  set checkShaderErrors(value: boolean) {
    this.checkShaderErrorsValue = rendererStateBoolean(value, 'Renderer.debug.checkShaderErrors')
  }

  get onShaderError(): ((...args: unknown[]) => void) | null {
    return this.onShaderErrorValue
  }

  set onShaderError(value: ((...args: unknown[]) => void) | null) {
    if (value !== null && typeof value !== 'function') {
      throw new TypeError('Renderer.debug.onShaderError must be a function or null.')
    }
    this.onShaderErrorValue = value
  }

  async getShaderAsync(scene: unknown, camera: unknown, object: unknown): Promise<never> {
    validateThreeSceneRoot(scene)
    validateTopLevelRenderCamera(camera)
    if (object === null || typeof object !== 'object' || Array.isArray(object)) {
      throw new TypeError('Renderer.debug.getShaderAsync object must be an object-like value.')
    }
    throw new Error(
      'Renderer.debug.getShaderAsync() is not supported by @headless-three/renderer because generated backend shader source is not exposed by the scene-oriented native renderer. Use material.userData.headlessThreeRenderer.fragmentWgsl for explicit custom WGSL fragments.',
    )
  }
}

class RendererInspectorState implements RendererInspectorLike {
  currentFrame: unknown = null
  private renderer: unknown = null

  setRenderer(renderer: unknown): this {
    this.renderer = renderer
    return this
  }

  getRenderer(): unknown {
    return this.renderer
  }

  init(): void {}

  begin(): void {}

  finish(): void {}

  inspect(_node: unknown): void {}

  computeAsync(_computeNode: unknown, _dispatchSizeOrCount?: unknown): void {}

  beginCompute(_uid: unknown, _computeNode?: unknown): void {}

  finishCompute(_uid?: unknown): void {}

  beginRender(_uid: unknown, _scene?: unknown, _camera?: unknown, _renderTarget?: unknown): void {}

  finishRender(_uid?: unknown): void {}

  copyTextureToTexture(_srcTexture: unknown, _dstTexture: unknown): void {}

  copyFramebufferToTexture(_framebufferTexture: unknown): void {}
}

class RendererBackendState {
  readonly isWebGLBackend = false
  readonly isWebGPUBackend = false
  readonly coordinateSystem = WEBGL_COORDINATE_SYSTEM
  readonly parameters: Record<string, never> = Object.freeze({})
  private data = new WeakMap<object, Record<string, unknown>>()

  constructor(readonly renderer: Renderer) {}

  async init(_renderer: unknown = this.renderer): Promise<void> {}

  beginRender(_renderContext?: unknown): void {}

  finishRender(_renderContext?: unknown): void {}

  beginCompute(_computeGroup?: unknown): void {}

  finishCompute(_computeGroup?: unknown): void {}

  clear(): never {
    throw unsupportedBackendOperationError('Renderer.backend.clear', 'backend render-context clearing')
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

  createBindings(): never {
    throw unsupportedBackendOperationError('Renderer.backend.createBindings', 'backend bind-group creation')
  }

  updateBindings(): never {
    throw unsupportedBackendOperationError('Renderer.backend.updateBindings', 'backend bind-group updates')
  }

  updateBinding(): never {
    throw unsupportedBackendOperationError('Renderer.backend.updateBinding', 'backend buffer binding updates')
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

  createIndexAttribute(): never {
    throw unsupportedBackendOperationError('Renderer.backend.createIndexAttribute', 'backend index buffer creation')
  }

  createStorageAttribute(): never {
    throw unsupportedBackendOperationError('Renderer.backend.createStorageAttribute', 'backend storage buffer creation')
  }

  createIndirectStorageAttribute(): never {
    throw unsupportedBackendOperationError('Renderer.backend.createIndirectStorageAttribute', 'backend indirect storage buffer creation')
  }

  updateAttribute(): never {
    throw unsupportedBackendOperationError('Renderer.backend.updateAttribute', 'backend attribute buffer updates')
  }

  destroyAttribute(): void {}

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

class RendererNodesState {
  modelViewMatrix: unknown = null
  modelNormalViewMatrix: unknown = null
}

class RendererNodeLibraryState {
  readonly lightNodes = new WeakMap<object, (...args: unknown[]) => unknown>()
  readonly materialNodes = new Map<string, new (...args: unknown[]) => Record<string, unknown>>()
  readonly toneMappingNodes = new Map<number, (...args: unknown[]) => unknown>()

  fromMaterial(material: unknown): unknown {
    if (material == null || typeof material !== 'object' || Array.isArray(material)) {
      throw new TypeError('Renderer.library.fromMaterial material must be a material-like object.')
    }
    if ((material as { isNodeMaterial?: unknown }).isNodeMaterial === true) return material
    const materialType = (material as { type?: unknown }).type
    if (typeof materialType !== 'string' || materialType.length === 0) return null
    const NodeMaterialClass = this.getMaterialNodeClass(materialType)
    if (NodeMaterialClass === null) return null
    const nodeMaterial = new NodeMaterialClass()
    Object.assign(nodeMaterial, material)
    return nodeMaterial
  }

  addToneMapping(toneMappingNode: unknown, toneMapping: unknown): void {
    assertFunction(toneMappingNode, 'Renderer.library.addToneMapping toneMappingNode')
    assertFiniteInteger(toneMapping, 'Renderer.library.addToneMapping toneMapping')
    this.addType(toneMappingNode, toneMapping, this.toneMappingNodes)
  }

  getToneMappingFunction(toneMapping: unknown): ((...args: unknown[]) => unknown) | null {
    assertFiniteInteger(toneMapping, 'Renderer.library.getToneMappingFunction toneMapping')
    return this.toneMappingNodes.get(toneMapping as number) ?? null
  }

  getMaterialNodeClass(materialType: unknown): (new (...args: unknown[]) => Record<string, unknown>) | null {
    assertNonEmptyString(materialType, 'Renderer.library.getMaterialNodeClass materialType')
    return this.materialNodes.get(materialType) ?? null
  }

  addMaterial(materialNodeClass: unknown, materialClassType: unknown): void {
    assertConstructorFunction(materialNodeClass, 'Renderer.library.addMaterial materialNodeClass')
    assertNonEmptyString(materialClassType, 'Renderer.library.addMaterial materialClassType')
    this.addType(materialNodeClass, materialClassType, this.materialNodes)
  }

  getLightNodeClass(light: unknown): ((...args: unknown[]) => unknown) | null {
    assertWeakMapKey(light, 'Renderer.library.getLightNodeClass light')
    return this.lightNodes.get(light) ?? null
  }

  addLight(lightNodeClass: unknown, lightClass: unknown): void {
    assertFunction(lightNodeClass, 'Renderer.library.addLight lightNodeClass')
    assertConstructorFunction(lightClass, 'Renderer.library.addLight lightClass')
    this.addClass(lightNodeClass, lightClass, this.lightNodes)
  }

  addType<T>(nodeClass: unknown, type: unknown, library: Map<any, T>): void {
    assertFunction(nodeClass, 'Renderer.library.addType nodeClass')
    if ((typeof type !== 'string' && typeof type !== 'number') || (typeof type === 'string' && type.length === 0)) {
      throw new TypeError('Renderer.library.addType type must be a non-empty string or integer.')
    }
    if (typeof type === 'number') assertFiniteInteger(type, 'Renderer.library.addType type')
    if (!library.has(type)) {
      library.set(type, nodeClass as T)
    }
  }

  addClass<T>(nodeClass: unknown, baseClass: unknown, library: WeakMap<object, T>): void {
    assertFunction(nodeClass, 'Renderer.library.addClass nodeClass')
    assertConstructorFunction(baseClass, 'Renderer.library.addClass baseClass')
    if (!library.has(baseClass)) {
      library.set(baseClass, nodeClass as T)
    }
  }
}

class RendererLightingNodeState {
  readonly isLightsNode = true
  private lightsValue: unknown[] = []

  constructor(lights: unknown[] = []) {
    this.setLights(lights)
  }

  setLights(lights: unknown[] = []): this {
    if (!Array.isArray(lights)) {
      throw new TypeError('Renderer.lighting lights must be an array.')
    }
    this.lightsValue = [...lights]
    return this
  }

  getLights(): unknown[] {
    return [...this.lightsValue]
  }
}

class RendererLightingState {
  private readonly nodes = new WeakMap<object, WeakMap<object, RendererLightingNodeState>>()
  private readonly defaultLightsNode = new RendererLightingNodeState()

  createNode(lights: unknown[] = []): RendererLightingNodeState {
    return new RendererLightingNodeState(lights)
  }

  getNode(scene: unknown, camera: unknown): RendererLightingNodeState {
    assertWeakMapKey(scene, 'Renderer.lighting.getNode scene')
    if ((scene as { isQuadMesh?: unknown }).isQuadMesh === true) return this.defaultLightsNode
    assertWeakMapKey(camera, 'Renderer.lighting.getNode camera')
    let cameraMap = this.nodes.get(scene)
    if (cameraMap === undefined) {
      cameraMap = new WeakMap()
      this.nodes.set(scene, cameraMap)
    }
    let node = cameraMap.get(camera)
    if (node === undefined) {
      node = this.createNode()
      cameraMap.set(camera, node)
    }
    return node
  }
}

class RendererDomElementState {
  width = 0
  height = 0
  private readonly attributes = new Map<string, string>()
  private readonly listeners = new Map<string, Set<(event: unknown) => void>>()

  readonly style = createRendererDomElementStyle()

  get clientWidth(): number {
    return this.stylePixelSize(this.style.width, this.width)
  }

  get clientHeight(): number {
    return this.stylePixelSize(this.style.height, this.height)
  }

  get offsetWidth(): number {
    return this.clientWidth
  }

  get offsetHeight(): number {
    return this.clientHeight
  }

  setSize(width: number, height: number, updateStyle = true): void {
    this.width = width
    this.height = height
    if (updateStyle) {
      this.style.width = `${width}px`
      this.style.height = `${height}px`
    }
  }

  setAttribute(name: unknown, value: unknown): void {
    assertDomElementAttributeName(name, 'Renderer.domElement.setAttribute name')
    this.attributes.set(name, String(value))
  }

  getAttribute(name: unknown): string | null {
    assertDomElementAttributeName(name, 'Renderer.domElement.getAttribute name')
    return this.attributes.get(name) ?? null
  }

  hasAttribute(name: unknown): boolean {
    assertDomElementAttributeName(name, 'Renderer.domElement.hasAttribute name')
    return this.attributes.has(name)
  }

  removeAttribute(name: unknown): void {
    assertDomElementAttributeName(name, 'Renderer.domElement.removeAttribute name')
    this.attributes.delete(name)
  }

  getBoundingClientRect(): {
    x: number
    y: number
    width: number
    height: number
    top: number
    right: number
    bottom: number
    left: number
  } {
    const width = this.clientWidth
    const height = this.clientHeight
    return {
      x: 0,
      y: 0,
      width,
      height,
      top: 0,
      right: width,
      bottom: height,
      left: 0,
    }
  }

  getContext(): never {
    throw new Error(
      'Renderer.domElement.getContext() is not supported by @headless-three/renderer because the domElement is an inert offscreen compatibility object, not a browser canvas.',
    )
  }

  toDataURL(_type?: unknown, _quality?: unknown): never {
    throw new Error(
      'Renderer.domElement.toDataURL() is not supported by @headless-three/renderer because the domElement is an inert offscreen compatibility object, not a browser canvas. Use Renderer.render() without format: "rgba" to receive a PNG Buffer.',
    )
  }

  toBlob(callback: unknown, _type?: unknown, _quality?: unknown): never {
    if (typeof callback !== 'function') {
      throw new TypeError('Renderer.domElement.toBlob callback must be a function.')
    }
    throw new Error(
      'Renderer.domElement.toBlob() is not supported by @headless-three/renderer because the domElement is an inert offscreen compatibility object, not a browser canvas. Use Renderer.render() without format: "rgba" to receive a PNG Buffer.',
    )
  }

  captureStream(_frameRate?: unknown): never {
    throw new Error(
      'Renderer.domElement.captureStream() is not supported by @headless-three/renderer because the domElement is an inert offscreen compatibility object, not a browser canvas.',
    )
  }

  transferToImageBitmap(): never {
    throw new Error(
      'Renderer.domElement.transferToImageBitmap() is not supported by @headless-three/renderer because the domElement is an inert offscreen compatibility object, not an OffscreenCanvas.',
    )
  }

  addEventListener(type: unknown, listener: unknown, _options?: unknown): void {
    assertEventListener(type, listener, 'Renderer.domElement.addEventListener')
    const eventType = type as string
    const eventListener = listener as (event: unknown) => void
    let typeListeners = this.listeners.get(eventType)
    if (!typeListeners) {
      typeListeners = new Set()
      this.listeners.set(eventType, typeListeners)
    }
    typeListeners.add(eventListener)
  }

  removeEventListener(type: unknown, listener: unknown, _options?: unknown): void {
    assertEventListener(type, listener, 'Renderer.domElement.removeEventListener')
    this.listeners.get(type as string)?.delete(listener as (event: unknown) => void)
  }

  dispatchEvent(event: unknown): boolean {
    if (event == null || typeof event !== 'object' || Array.isArray(event)) {
      throw new TypeError('Renderer.domElement.dispatchEvent event must be an event-like object.')
    }
    const type = (event as { type?: unknown }).type
    if (typeof type !== 'string' || type.length === 0) {
      throw new TypeError('Renderer.domElement.dispatchEvent event.type must be a non-empty string.')
    }
    for (const listener of [...(this.listeners.get(type) ?? [])]) {
      listener.call(this, event)
    }
    return true
  }

  private stylePixelSize(value: unknown, fallback: number): number {
    const size = typeof value === 'number' ? value : typeof value === 'string' ? Number.parseFloat(value) : Number.NaN
    return Number.isFinite(size) && size >= 0 ? Math.round(size) : fallback
  }
}

type RendererDomElementStyle = {
  width: string
  height: string
  setProperty(propertyName: unknown, value?: unknown): void
  getPropertyValue(propertyName: unknown): string
  removeProperty(propertyName: unknown): string
  [key: string]: unknown
}

function createRendererDomElementStyle(): RendererDomElementStyle {
  const style = { width: '0px', height: '0px' } as RendererDomElementStyle
  Object.defineProperties(style, {
    setProperty: {
      value: function setProperty(this: RendererDomElementStyle, propertyName: unknown, value: unknown = ''): void {
        const key = domElementStyleWritablePropertyKey(propertyName, 'Renderer.domElement.style.setProperty propertyName')
        this[key] = String(value)
      },
    },
    getPropertyValue: {
      value: function getPropertyValue(this: RendererDomElementStyle, propertyName: unknown): string {
        const key = domElementStylePropertyKey(propertyName, 'Renderer.domElement.style.getPropertyValue propertyName')
        const value = this[key]
        return value === undefined || typeof value === 'function' ? '' : String(value)
      },
    },
    removeProperty: {
      value: function removeProperty(this: RendererDomElementStyle, propertyName: unknown): string {
        const key = domElementStyleWritablePropertyKey(propertyName, 'Renderer.domElement.style.removeProperty propertyName')
        const previous = this.getPropertyValue(propertyName)
        if (key === 'width' || key === 'height') {
          this[key] = ''
        } else {
          delete this[key]
        }
        return previous
      },
    },
  })
  return style
}

class RendererColorBufferState {
  setMask(colorMask: unknown): void {
    rendererStateBoolean(colorMask, 'Renderer.state.buffers.color.setMask mask')
  }

  setLocked(lock: unknown): void {
    rendererStateBoolean(lock, 'Renderer.state.buffers.color.setLocked lock')
  }

  setClear(r: unknown, g: unknown, b: unknown, a: unknown, premultipliedAlpha?: unknown): void {
    rendererStateClearAlpha(r, 'Renderer.state.buffers.color.setClear r')
    rendererStateClearAlpha(g, 'Renderer.state.buffers.color.setClear g')
    rendererStateClearAlpha(b, 'Renderer.state.buffers.color.setClear b')
    rendererStateClearAlpha(a, 'Renderer.state.buffers.color.setClear a')
    if (premultipliedAlpha !== undefined) {
      rendererStateBoolean(premultipliedAlpha, 'Renderer.state.buffers.color.setClear premultipliedAlpha')
    }
  }

  reset(): void {
    // Native color-buffer state is rebuilt for each render pass.
  }
}

class RendererDepthBufferState {
  getReversed(): boolean {
    return false
  }

  setReversed(reversed: unknown): void {
    const enabled = rendererStateBoolean(reversed, 'Renderer.state.buffers.depth.setReversed reversed')
    if (enabled) {
      throw new Error(
        'Renderer.state.buffers.depth.setReversed(true) is not supported by @headless-three/renderer because reversed depth buffers are not implemented. Keep reversedDepthBuffer disabled.',
      )
    }
  }

  setTest(depthTest: unknown): void {
    rendererStateBoolean(depthTest, 'Renderer.state.buffers.depth.setTest test')
  }

  setMask(depthMask: unknown): void {
    rendererStateBoolean(depthMask, 'Renderer.state.buffers.depth.setMask mask')
  }

  setFunc(depthFunc: unknown): void {
    rendererStateClearStencil(depthFunc, 'Renderer.state.buffers.depth.setFunc func')
  }

  setLocked(lock: unknown): void {
    rendererStateBoolean(lock, 'Renderer.state.buffers.depth.setLocked lock')
  }

  setClear(depth: unknown): void {
    rendererStateClearDepth(depth, 'Renderer.state.buffers.depth.setClear depth')
  }

  reset(): void {
    // Native depth-buffer state is rebuilt for each render pass.
  }
}

class RendererStencilBufferState {
  setTest(stencilTest: unknown): void {
    rendererStateBoolean(stencilTest, 'Renderer.state.buffers.stencil.setTest test')
  }

  setMask(stencilMask: unknown): void {
    rendererStateClearStencil(stencilMask, 'Renderer.state.buffers.stencil.setMask mask')
  }

  setFunc(stencilFunc: unknown, stencilRef: unknown, stencilMask: unknown): void {
    rendererStateClearStencil(stencilFunc, 'Renderer.state.buffers.stencil.setFunc func')
    rendererStateClearStencil(stencilRef, 'Renderer.state.buffers.stencil.setFunc ref')
    rendererStateClearStencil(stencilMask, 'Renderer.state.buffers.stencil.setFunc mask')
  }

  setOp(stencilFail: unknown, stencilZFail: unknown, stencilZPass: unknown): void {
    rendererStateClearStencil(stencilFail, 'Renderer.state.buffers.stencil.setOp fail')
    rendererStateClearStencil(stencilZFail, 'Renderer.state.buffers.stencil.setOp zFail')
    rendererStateClearStencil(stencilZPass, 'Renderer.state.buffers.stencil.setOp zPass')
  }

  setLocked(lock: unknown): void {
    rendererStateBoolean(lock, 'Renderer.state.buffers.stencil.setLocked lock')
  }

  setClear(stencil: unknown): void {
    rendererStateClearStencil(stencil, 'Renderer.state.buffers.stencil.setClear stencil')
  }

  reset(): void {
    // Native stencil-buffer state is rebuilt for each render pass.
  }
}

class RendererStateBuffersState {
  readonly color = new RendererColorBufferState()
  readonly depth = new RendererDepthBufferState()
  readonly stencil = new RendererStencilBufferState()
}

class RendererState {
  readonly buffers = new RendererStateBuffersState()

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

class RendererExtensionsState {
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

class RendererCapabilitiesState {
  readonly isWebGL2 = false
  readonly drawBuffers = false
  readonly precision = 'highp'
  readonly logarithmicDepthBuffer = false
  readonly reversedDepthBuffer = false
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

class RendererPropertiesState {
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

type RendererRenderListSort = (a: RendererRenderListItem, b: RendererRenderListItem) => number

type RendererRenderListItem = {
  id: unknown
  object: unknown
  geometry: unknown
  material: unknown
  materialVariant: number
  groupOrder: number
  renderOrder: unknown
  z: number
  group: unknown
}

class RendererRenderList {
  private readonly renderItems: RendererRenderListItem[] = []
  private renderItemsIndex = 0

  readonly opaque: RendererRenderListItem[] = []
  readonly transmissive: RendererRenderListItem[] = []
  readonly transparent: RendererRenderListItem[] = []

  init(): void {
    this.renderItemsIndex = 0
    this.opaque.length = 0
    this.transmissive.length = 0
    this.transparent.length = 0
  }

  push(
    object: unknown,
    geometry: unknown,
    material: unknown,
    groupOrder = 0,
    z = 0,
    group: unknown = null,
  ): void {
    const renderItem = this.getNextRenderItem(object, geometry, material, groupOrder, z, group)
    this.bucketForMaterial(material).push(renderItem)
  }

  unshift(
    object: unknown,
    geometry: unknown,
    material: unknown,
    groupOrder = 0,
    z = 0,
    group: unknown = null,
  ): void {
    const renderItem = this.getNextRenderItem(object, geometry, material, groupOrder, z, group)
    this.bucketForMaterial(material).unshift(renderItem)
  }

  sort(customOpaqueSort?: RendererRenderListSort | null, customTransparentSort?: RendererRenderListSort | null): void {
    if (customOpaqueSort !== undefined && customOpaqueSort !== null && typeof customOpaqueSort !== 'function') {
      throw new TypeError('Renderer.renderLists list opaque sort must be a function or null.')
    }
    if (customTransparentSort !== undefined && customTransparentSort !== null && typeof customTransparentSort !== 'function') {
      throw new TypeError('Renderer.renderLists list transparent sort must be a function or null.')
    }
    if (customOpaqueSort) this.opaque.sort(customOpaqueSort)
    if (customTransparentSort) {
      this.transmissive.sort(customTransparentSort)
      this.transparent.sort(customTransparentSort)
    }
  }

  finish(): void {
    for (let i = this.renderItemsIndex; i < this.renderItems.length; i += 1) {
      this.renderItems[i].id = null
      this.renderItems[i].object = null
      this.renderItems[i].geometry = null
      this.renderItems[i].material = null
      this.renderItems[i].group = null
    }
  }

  private getNextRenderItem(
    object: unknown,
    geometry: unknown,
    material: unknown,
    groupOrder: number,
    z: number,
    group: unknown,
  ): RendererRenderListItem {
    assertFiniteNumberOption(groupOrder, 'Renderer.renderLists list groupOrder')
    assertFiniteNumberOption(z, 'Renderer.renderLists list z')
    let renderItem = this.renderItems[this.renderItemsIndex]
    if (renderItem === undefined) {
      renderItem = {
        id: rendererRenderListId(object),
        object,
        geometry,
        material,
        materialVariant: rendererRenderListMaterialVariant(object),
        groupOrder,
        renderOrder: rendererRenderListRenderOrder(object),
        z,
        group,
      }
      this.renderItems[this.renderItemsIndex] = renderItem
    } else {
      renderItem.id = rendererRenderListId(object)
      renderItem.object = object
      renderItem.geometry = geometry
      renderItem.material = material
      renderItem.materialVariant = rendererRenderListMaterialVariant(object)
      renderItem.groupOrder = groupOrder
      renderItem.renderOrder = rendererRenderListRenderOrder(object)
      renderItem.z = z
      renderItem.group = group
    }
    this.renderItemsIndex += 1
    return renderItem
  }

  private bucketForMaterial(material: unknown): RendererRenderListItem[] {
    const record = material && typeof material === 'object' ? material as Record<string, unknown> : undefined
    if (typeof record?.transmission === 'number' && record.transmission > 0) return this.transmissive
    if (record?.transparent === true) return this.transparent
    return this.opaque
  }
}

class RendererRenderListsState {
  private lists = new WeakMap<object, RendererRenderList[]>()

  get(scene: object, renderCallDepth = 0): RendererRenderList {
    assertWeakMapKey(scene, 'Renderer.renderLists.get scene')
    if (!Number.isInteger(renderCallDepth) || renderCallDepth < 0) {
      throw new TypeError(`Renderer.renderLists.get renderCallDepth must be a non-negative integer; received ${String(renderCallDepth)}.`)
    }
    let listArray = this.lists.get(scene)
    if (listArray === undefined) {
      listArray = []
      this.lists.set(scene, listArray)
    }
    let list = listArray[renderCallDepth]
    if (list === undefined) {
      list = new RendererRenderList()
      listArray[renderCallDepth] = list
    }
    return list
  }

  dispose(): void {
    this.lists = new WeakMap()
  }
}

class RendererRenderLightsState {
  readonly state = {
    version: 0,
    hash: {
      directionalLength: -1,
      pointLength: -1,
      spotLength: -1,
      rectAreaLength: -1,
      hemiLength: -1,
      numDirectionalShadows: -1,
      numPointShadows: -1,
      numSpotShadows: -1,
      numSpotMaps: -1,
      numLightProbes: -1,
    },
    ambient: [0, 0, 0],
    probe: Array.from({ length: 9 }, () => ({ x: 0, y: 0, z: 0 })),
    directional: [],
    directionalShadow: [],
    directionalShadowMap: [],
    directionalShadowMatrix: [],
    spot: [],
    spotLightMap: [],
    spotShadow: [],
    spotShadowMap: [],
    spotLightMatrix: [],
    rectArea: [],
    rectAreaLTC1: null,
    rectAreaLTC2: null,
    point: [],
    pointShadow: [],
    pointShadowMap: [],
    pointShadowMatrix: [],
    hemi: [],
    numSpotLightShadowsWithMaps: 0,
    numLightProbes: 0,
  }

  setup(_lights: unknown[] = []): void {}

  setupView(_lights: unknown[] = [], _camera?: unknown): void {}
}

class RendererRenderState {
  readonly state = {
    lightsArray: [] as unknown[],
    shadowsArray: [] as unknown[],
    camera: null as unknown,
    lights: new RendererRenderLightsState(),
    transmissionRenderTarget: {} as Record<PropertyKey, unknown>,
  }

  init(camera: unknown): void {
    this.state.camera = camera
    this.state.lightsArray.length = 0
    this.state.shadowsArray.length = 0
  }

  pushLight(light: unknown): void {
    this.state.lightsArray.push(light)
  }

  pushShadow(shadowLight: unknown): void {
    this.state.shadowsArray.push(shadowLight)
  }

  setupLights(): void {
    this.state.lights.setup(this.state.lightsArray)
  }

  setupLightsView(camera: unknown): void {
    this.state.lights.setupView(this.state.lightsArray, camera)
  }
}

class RendererRenderStatesState {
  private states = new WeakMap<object, RendererRenderState[]>()

  get(scene: object, renderCallDepth = 0): RendererRenderState {
    assertWeakMapKey(scene, 'Renderer.renderStates.get scene')
    if (!Number.isInteger(renderCallDepth) || renderCallDepth < 0) {
      throw new TypeError(`Renderer.renderStates.get renderCallDepth must be a non-negative integer; received ${String(renderCallDepth)}.`)
    }
    let stateArray = this.states.get(scene)
    if (stateArray === undefined) {
      stateArray = []
      this.states.set(scene, stateArray)
    }
    let renderState = stateArray[renderCallDepth]
    if (renderState === undefined) {
      renderState = new RendererRenderState()
      stateArray[renderCallDepth] = renderState
    }
    return renderState
  }

  dispose(): void {
    this.states = new WeakMap()
  }
}

function collectCompileMaterials(scene: ThreeSceneRootLike): Set<ThreeMaterialLike> {
  const materials = new Set<ThreeMaterialLike>()
  collectObjectCompileMaterials(scene, materials, 'Renderer.compile scene')
  return materials
}

function collectObjectCompileMaterials(
  object: ThreeObject3DLike,
  materials: Set<ThreeMaterialLike>,
  label: string,
): void {
  if (isCompileRenderableObject(object) && object.material != null) {
    if (Array.isArray(object.material)) {
      for (let i = 0; i < object.material.length; i += 1) {
        addCompileMaterial(object.material[i], materials, `${label}.material[${i}]`)
      }
    } else {
      addCompileMaterial(object.material, materials, `${label}.material`)
    }
  }

  const children = object.children ?? []
  for (let i = 0; i < children.length; i += 1) {
    collectObjectCompileMaterials(children[i], materials, `${label}.children[${i}]`)
  }
}

function isCompileRenderableObject(object: ThreeObject3DLike): boolean {
  return object.isMesh === true
    || object.isPoints === true
    || object.isLine === true
    || object.isLineSegments === true
    || object.isLineLoop === true
    || object.isSprite === true
}

function addCompileMaterial(material: unknown, materials: Set<ThreeMaterialLike>, label: string): void {
  if (material === null || typeof material !== 'object' || Array.isArray(material)) {
    throw new TypeError(`${label} must be a material-like object.`)
  }
  materials.add(material as ThreeMaterialLike)
}

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
  readonly domElement = new RendererDomElementState()
  readonly extensions = new RendererExtensionsState()
  readonly info = new RendererInfoState()
  readonly library = new RendererNodeLibraryState()
  readonly lighting = new RendererLightingState()
  readonly nodes = new RendererNodesState()
  readonly properties = new RendererPropertiesState()
  readonly renderLists = new RendererRenderListsState()
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
    this.native = new native.NativeRenderer()
    this.inspectorValue.setRenderer(this)
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

  set onDeviceLost(value: (info?: unknown) => void) {
    if (typeof value !== 'function') {
      throw new TypeError('Renderer.onDeviceLost must be a function.')
    }
    this.onDeviceLostValue = value
  }

  _onDeviceLost(info?: unknown): void {
    this.defaultOnDeviceLost(info)
    if (this.onDeviceLostValue !== this.defaultOnDeviceLost) {
      this.onDeviceLostValue(info)
    }
  }

  get inspector(): RendererInspectorLike {
    return this.inspectorValue
  }

  set inspector(value: RendererInspectorLike) {
    assertRendererInspectorLike(value, 'Renderer.inspector')
    this.inspectorValue.setRenderer(null)
    this.inspectorValue = value
    this.inspectorValue.setRenderer(this)
  }

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

  isOccluded(object: unknown): boolean {
    if (object === null || typeof object !== 'object' || Array.isArray(object)) {
      throw new TypeError('Renderer.isOccluded object must be an object-like value.')
    }
    return false
  }

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

  set sortObjects(value: boolean) {
    if (typeof value !== 'boolean') {
      throw new TypeError(`Renderer.sortObjects must be a boolean; received ${String(value)}.`)
    }
    this.sortObjectsValue = value
  }

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

  set highPrecision(value: boolean) {
    const enabled = rendererStateBoolean(value, 'Renderer.highPrecision')
    if (!enabled) return
    throw new Error(
      'Renderer.highPrecision = true is not supported by @headless-three/renderer because Three.js CommonRenderer high-precision matrix nodes require backend shader-node state that is outside the scene-oriented API.',
    )
  }

  get samples(): number {
    return 0
  }

  get needsFrameBufferTarget(): boolean {
    return false
  }

  get currentSamples(): number {
    if (!this.currentRenderTarget) return this.samples
    const targetSamples = this.currentRenderTarget.sampleCount ?? this.currentRenderTarget.samples ?? 1
    return targetSamples > 1 ? targetSamples : 1
  }

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
  ): Set<ThreeMaterialLike> {
    validateThreeSceneRoot(scene)
    validateTopLevelRenderCamera(camera)
    if (targetScene !== null) validateThreeSceneRoot(targetScene)
    validateObjectChildrenTree(scene)
    if (targetScene !== null) validateObjectChildrenTree(targetScene)
    return collectCompileMaterials(scene)
  }

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

  setRenderObjectFunction(renderObjectFunction: ((...args: unknown[]) => unknown) | null): void {
    if (renderObjectFunction === null) return
    if (typeof renderObjectFunction !== 'function') {
      throw new TypeError('Renderer.setRenderObjectFunction renderObjectFunction must be a function or null.')
    }
    throw new Error(
      'Renderer.setRenderObjectFunction() is not supported by @headless-three/renderer because it does not expose renderer-internal render-object dispatch. Render normal Three.js scene graphs with Renderer.render() or renderToTarget().',
    )
  }

  getRenderObjectFunction(): null {
    return null
  }

  renderBufferDirect(): never {
    throw new Error(
      'Renderer.renderBufferDirect() is not supported by @headless-three/renderer because it does not expose WebGL buffer binding or direct material program dispatch. Render normal Three.js scene graphs with Renderer.render() or renderToTarget().',
    )
  }

  renderObject(): never {
    throw new Error(
      'Renderer.renderObject() is not supported by @headless-three/renderer because it does not expose renderer-internal render-object dispatch or direct material program dispatch. Render normal Three.js scene graphs with Renderer.render() or renderToTarget().',
    )
  }

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

  _createObjectPipeline(): never {
    throw unsupportedInternalRenderDispatchError('Renderer._createObjectPipeline')
  }

  compute(computeNodes: unknown, dispatchSize: unknown = null): never {
    assertComputeNodesLike(computeNodes, 'Renderer.compute computeNodes')
    assertComputeDispatchSize(dispatchSize, 'Renderer.compute dispatchSize')
    throw unsupportedComputeError('Renderer.compute')
  }

  async computeAsync(computeNodes: unknown, dispatchSize: unknown = null): Promise<never> {
    assertComputeNodesLike(computeNodes, 'Renderer.computeAsync computeNodes')
    assertComputeDispatchSize(dispatchSize, 'Renderer.computeAsync dispatchSize')
    throw unsupportedComputeError('Renderer.computeAsync')
  }

  async getArrayBufferAsync(attribute: unknown): Promise<ArrayBuffer> {
    assertStorageBufferAttributeLike(attribute, 'Renderer.getArrayBufferAsync attribute')
    throw new Error(
      'Renderer.getArrayBufferAsync() is not supported by @headless-three/renderer because storage-buffer GPU readback requires WebGPU backend state that this package does not expose. Use Renderer.readRenderTargetPixels() for render-target CPU readback.',
    )
  }

  async resolveTimestampsAsync(type: unknown = 'render'): Promise<number> {
    assertTimestampQueryType(type, 'Renderer.resolveTimestampsAsync type')
    throw new Error(
      'Renderer.resolveTimestampsAsync() is not supported by @headless-three/renderer because timestamp queries require backend GPU query pools that are outside the scene-oriented API.',
    )
  }

  async waitForGPU(): Promise<void> {
    throw new Error(
      'Renderer.waitForGPU() is not supported by @headless-three/renderer because it does not expose direct GPU task synchronization. Renderer.render() and renderToTarget() return after native scene output readback or target writeback has completed.',
    )
  }

  setMRT(mrt: unknown = null): this {
    if (mrt !== null) {
      throw new Error(
        'Renderer.setMRT() is not supported by @headless-three/renderer because arbitrary native MRT shader outputs are outside the scene-oriented API. Use target texture userData.headlessThreeRenderer.renderMode for the supported color, mask, object-id, normal, and depth auxiliary outputs.',
      )
    }
    return this
  }

  getMRT(): null {
    return null
  }

  setOutputRenderTarget(renderTarget: unknown = null): void {
    if (renderTarget === null) return
    assertRenderTargetLike(renderTarget, 'Renderer.setOutputRenderTarget renderTarget')
    validateUnsupportedRenderTargetOptions(renderTarget)
    throw new Error(
      'Renderer.setOutputRenderTarget() is not supported by @headless-three/renderer because common-renderer output targets are backend-owned canvas/WebGPU state. Use Renderer.setRenderTarget() or renderToTarget() with a target-like object for offscreen output.',
    )
  }

  getOutputRenderTarget(): null {
    return null
  }

  setCanvasTarget(canvasTarget: unknown = null): void {
    if (canvasTarget === null) return
    assertCanvasTargetLike(canvasTarget, 'Renderer.setCanvasTarget canvasTarget')
    throw new Error(
      'Renderer.setCanvasTarget() is not supported by @headless-three/renderer because it does not own a browser canvas or WebGPU canvas target. Use Renderer.domElement for inert canvas compatibility metadata and Renderer.render() for headless output.',
    )
  }

  getCanvasTarget(): null {
    return null
  }

  setTexture2D(texture: unknown, slot: unknown): never {
    assertThreeTextureLike(texture, 'Renderer.setTexture2D texture')
    assertTextureBindingSlot(slot, 'Renderer.setTexture2D slot')
    throw unsupportedTextureBindingError('Renderer.setTexture2D')
  }

  setTextureCube(texture: unknown, slot: unknown): never {
    assertThreeTextureLike(texture, 'Renderer.setTextureCube texture')
    assertTextureBindingSlot(slot, 'Renderer.setTextureCube slot')
    throw unsupportedTextureBindingError('Renderer.setTextureCube')
  }

  setTextureCubeDynamic(texture: unknown, slot: unknown): never {
    assertThreeTextureLike(texture, 'Renderer.setTextureCubeDynamic texture')
    assertTextureBindingSlot(slot, 'Renderer.setTextureCubeDynamic slot')
    throw unsupportedTextureBindingError('Renderer.setTextureCubeDynamic')
  }

  setTexture3D(texture: unknown, slot: unknown): never {
    assertThreeTextureLike(texture, 'Renderer.setTexture3D texture')
    assertTextureBindingSlot(slot, 'Renderer.setTexture3D slot')
    throw unsupportedTextureBindingError('Renderer.setTexture3D')
  }

  setTexture2DArray(texture: unknown, slot: unknown): never {
    assertThreeTextureLike(texture, 'Renderer.setTexture2DArray texture')
    assertTextureBindingSlot(slot, 'Renderer.setTexture2DArray slot')
    throw unsupportedTextureBindingError('Renderer.setTexture2DArray')
  }

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

  setRenderTargetTextures(renderTarget: RenderTargetLike, colorTexture: unknown, depthTexture: unknown = null): never {
    assertRenderTargetLike(renderTarget, 'Renderer.setRenderTargetTextures renderTarget')
    validateUnsupportedRenderTargetOptions(renderTarget)
    assertExternalWebGlObjectLike(colorTexture, 'Renderer.setRenderTargetTextures colorTexture')
    assertOptionalExternalWebGlObjectLike(depthTexture, 'Renderer.setRenderTargetTextures depthTexture')
    throw new Error(
      'Renderer.setRenderTargetTextures() is not supported by @headless-three/renderer because WebGLTexture handles cannot be attached to wgpu-backed render targets. Render into a target-like object and use Renderer.readRenderTargetPixels() for CPU readback.',
    )
  }

  setRenderTargetFramebuffer(renderTarget: RenderTargetLike, defaultFramebuffer?: unknown): never {
    assertRenderTargetLike(renderTarget, 'Renderer.setRenderTargetFramebuffer renderTarget')
    validateUnsupportedRenderTargetOptions(renderTarget)
    assertOptionalExternalWebGlObjectLike(defaultFramebuffer, 'Renderer.setRenderTargetFramebuffer defaultFramebuffer')
    throw new Error(
      'Renderer.setRenderTargetFramebuffer() is not supported by @headless-three/renderer because it does not expose a browser WebGL framebuffer. Use Renderer.setRenderTarget() with a target-like object for offscreen output.',
    )
  }

  copyFramebufferToTexture(texture: ThreeTextureLike, position?: unknown, level?: number): void
  copyFramebufferToTexture(position: unknown, texture: ThreeTextureLike, level?: number): void
  copyFramebufferToTexture(textureOrPosition: unknown, positionOrTexture: unknown = null, level = 0): void {
    let texture = textureOrPosition
    let position = positionOrTexture
    if (!hasThreeTextureMarker(textureOrPosition) && isThreeTextureArgument(positionOrTexture)) {
      position = textureOrPosition ?? null
      texture = positionOrTexture
    }
    assertThreeTextureLike(texture, 'Renderer.copyFramebufferToTexture texture')
    assertTextureCopyLevel(level, 'Renderer.copyFramebufferToTexture level')
    if (!this.currentRenderTarget) {
      throw new Error(
        'Renderer.copyFramebufferToTexture() requires an active render target with readable color data. Call Renderer.setRenderTarget(target), render into that target, then copy into a readable raw texture; use Renderer.readRenderTargetPixels() for explicit CPU readback.',
      )
    }

    const source = renderTargetReadbackSource(
      this.currentRenderTarget,
      isCubeRenderTarget(this.currentRenderTarget) ? this.currentActiveCubeFace : undefined,
      0,
      'Renderer.copyFramebufferToTexture',
      isCubeRenderTarget(this.currentRenderTarget) ? this.currentActiveMipmapLevel : 0,
    )
    const destination = rawTextureCopyImage(texture, 'Renderer.copyFramebufferToTexture texture', { level })
    if (source.channels !== destination.channels) {
      throw new Error(
        `Renderer.copyFramebufferToTexture framebuffer and destination texture must use the same raw channel count (${source.channels} framebuffer channels, ${destination.channels} destination channels).`,
      )
    }

    const region = textureCopyFramebufferSourceRegion(
      position,
      destination.width,
      destination.height,
      source.width,
      source.height,
      'Renderer.copyFramebufferToTexture source rectangle',
    )
    if (region.width > destination.width || region.height > destination.height) {
      throw new RangeError('Renderer.copyFramebufferToTexture source rectangle size exceeds destination texture bounds.')
    }

    const channels = source.channels
    for (let row = 0; row < region.height; row += 1) {
      const sourceStart = (((region.y + row) * source.width) + region.x) * channels
      const destinationStart = row * destination.width * channels
      for (let i = 0; i < region.width * channels; i += 1) {
        destination.data[destinationStart + i] = source.data[sourceStart + i]
      }
    }
    texture.needsUpdate = true
    this.inspector.copyFramebufferToTexture?.(texture)
  }

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
  ): void {
    let srcTexture = srcTextureOrDstPosition
    let dstTexture = dstTextureOrSrcTexture
    let srcRegion = srcRegionOrDstTexture
    let dstPosition = dstPositionOrDstLevel
    let resolvedSrcLevel = srcLevel
    let resolvedDstLevel: unknown = dstLevel
    if (
      !hasThreeTextureMarker(srcTextureOrDstPosition)
      && isThreeTextureArgument(dstTextureOrSrcTexture)
      && isThreeTextureArgument(srcRegionOrDstTexture)
    ) {
      dstPosition = srcTextureOrDstPosition ?? null
      srcTexture = dstTextureOrSrcTexture
      dstTexture = srcRegionOrDstTexture
      srcRegion = null
      resolvedSrcLevel = 0
      resolvedDstLevel = dstPositionOrDstLevel || 0
    }

    assertThreeTextureLike(srcTexture, 'Renderer.copyTextureToTexture source texture')
    assertThreeTextureLike(dstTexture, 'Renderer.copyTextureToTexture destination texture')

    if (resolvedDstLevel === null) {
      if (resolvedSrcLevel !== 0) {
        assertTextureCopyLevel(resolvedSrcLevel, 'Renderer.copyTextureToTexture legacy destination level')
        resolvedDstLevel = resolvedSrcLevel
        resolvedSrcLevel = 0
      } else {
        resolvedDstLevel = 0
      }
    }
    assertTextureCopyLevel(resolvedSrcLevel, 'Renderer.copyTextureToTexture source level')
    assertTextureCopyLevel(resolvedDstLevel, 'Renderer.copyTextureToTexture destination level')
    const resolvedDestinationLevel = resolvedDstLevel as number

    const source = rawTextureCopyImage(srcTexture, 'Renderer.copyTextureToTexture source texture', {
      allowCanvasRead: true,
      level: resolvedSrcLevel,
    })
    const destination = rawTextureCopyImage(dstTexture, 'Renderer.copyTextureToTexture destination texture', {
      level: resolvedDestinationLevel,
    })
    if (source.channels !== destination.channels) {
      throw new Error(
        `Renderer.copyTextureToTexture textures must use the same raw channel count (${source.channels} source channels, ${destination.channels} destination channels).`,
      )
    }

    const region = textureCopySourceRegion(srcRegion, source.width, source.height, 'Renderer.copyTextureToTexture source region')
    const position = textureCopyDestinationPosition(dstPosition, 'Renderer.copyTextureToTexture destination position')
    if (position.x + region.width > destination.width || position.y + region.height > destination.height) {
      throw new RangeError('Renderer.copyTextureToTexture destination position and source region exceed destination texture bounds.')
    }

    const channels = source.channels
    for (let row = 0; row < region.height; row += 1) {
      const sourceStart = (((region.y + row) * source.width) + region.x) * channels
      const destinationStart = (((position.y + row) * destination.width) + position.x) * channels
      for (let i = 0; i < region.width * channels; i += 1) {
        destination.data[destinationStart + i] = source.data[sourceStart + i]
      }
    }
    dstTexture.needsUpdate = true
    this.inspector.copyTextureToTexture?.(srcTexture, dstTexture)
  }

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
  ): never {
    let srcTexture = srcTextureOrSrcRegion
    let dstTexture = dstTextureOrDstPosition
    if (
      !hasThreeTextureMarker(srcTextureOrSrcRegion)
      && isThreeTextureArgument(_srcRegionOrSrcTexture)
      && isThreeTextureArgument(_dstPositionOrDstTexture)
    ) {
      srcTexture = _srcRegionOrSrcTexture
      dstTexture = _dstPositionOrDstTexture
    }

    assertThreeTextureLike(srcTexture, 'Renderer.copyTextureToTexture3D source texture')
    assertThreeTextureLike(dstTexture, 'Renderer.copyTextureToTexture3D destination texture')
    assertTextureCopyLevel(level, 'Renderer.copyTextureToTexture3D level')
    throw new Error(
      'Renderer.copyTextureToTexture3D() is not supported by @headless-three/renderer because 3D and array texture GPU copies require backend texture-layer state that this package does not expose. Use Renderer.copyTextureToTexture() for supported readable 2D raw texture copies.',
    )
  }

  setAnimationLoop(callback: RenderAnimationLoopCallback | null): void {
    if (callback !== null && typeof callback !== 'function') {
      throw new TypeError('Renderer.setAnimationLoop callback must be a function or null.')
    }
    this.animationLoop = callback
  }

  getAnimationLoop(): RenderAnimationLoopCallback | null {
    return this.animationLoop
  }

  getContext(): never {
    throw new Error(
      'Renderer.getContext() is not supported by @headless-three/renderer because it renders offscreen through wgpu instead of a browser WebGL context.',
    )
  }

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

  setRenderTarget(target: RenderTargetLike | null = null, activeCubeFace = 0, activeMipmapLevel = 0): void {
    if (target !== null) {
      assertRenderTargetLike(target, 'Renderer.setRenderTarget target')
      validateUnsupportedRenderTargetOptions(target)
    }
    assertActiveCubeFace(activeCubeFace, 'Renderer.setRenderTarget activeCubeFace')
    assertActiveMipmapLevel(activeMipmapLevel, 'Renderer.setRenderTarget activeMipmapLevel')
    this.currentRenderTarget = target
    this.currentActiveCubeFace = activeCubeFace
    this.currentActiveMipmapLevel = activeMipmapLevel
  }

  setSize(width: number, height: number, updateStyle = true): void {
    this.currentSize = rendererStateSize(width, height, 'Renderer.setSize')
    this.domElement.setSize(
      this.currentSize.width,
      this.currentSize.height,
      rendererStateBoolean(updateStyle, 'Renderer.setSize updateStyle'),
    )
  }

  setPixelRatio(value: number): void {
    this.pixelRatioValue = rendererStatePixelRatio(value, 'Renderer.setPixelRatio')
  }

  getPixelRatio(): number {
    return this.pixelRatioValue
  }

  getSize(): RenderSizeLike | null
  getSize<T extends RenderSizeLike>(target: T): T | null
  getSize(target?: RenderSizeLike): RenderSizeLike | null {
    return target === undefined
      ? clonePixelSize(this.currentSize)
      : clonePixelSize(this.currentSize, target)
  }

  setDrawingBufferSize(width: number, height: number, pixelRatio: number): void {
    this.currentSize = rendererStateSize(width, height, 'Renderer.setDrawingBufferSize')
    this.pixelRatioValue = rendererStatePixelRatio(pixelRatio, 'Renderer.setDrawingBufferSize pixelRatio')
    this.domElement.setSize(this.currentSize.width, this.currentSize.height)
  }

  getDrawingBufferSize(): RenderSizeLike | null
  getDrawingBufferSize<T extends RenderSizeLike>(target: T): T | null
  getDrawingBufferSize(target?: RenderSizeLike): RenderSizeLike | null {
    return target === undefined
      ? clonePixelSize(this.currentSize)
      : clonePixelSize(this.currentSize, target)
  }

  setClearColor(color: number | string | ThreeColorLike | number[], alpha?: number): void {
    this.currentClearColor = rendererStateClearColor(color, alpha)
  }

  getClearColor(): ThreeColorLike
  getClearColor<T extends ThreeColorLike>(target: T): T
  getClearColor(target?: ThreeColorLike): ThreeColorLike {
    return target === undefined
      ? cloneColor3(this.currentClearColor)
      : cloneColor3(this.currentClearColor, target)
  }

  setClearAlpha(alpha: number): void {
    this.currentClearColor = [
      this.currentClearColor[0],
      this.currentClearColor[1],
      this.currentClearColor[2],
      rendererStateClearAlpha(alpha, 'Renderer.setClearAlpha alpha'),
    ]
  }

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
  getViewport(target?: RenderPixelRectLike): RenderPixelRectLike | null {
    return target === undefined
      ? clonePixelRect(this.currentViewport)
      : clonePixelRect(this.currentViewport, target)
  }

  getCurrentViewport(): RenderPixelRectLike | null
  getCurrentViewport<T extends RenderPixelRectLike>(target: T): T | null
  getCurrentViewport(target?: RenderPixelRectLike): RenderPixelRectLike | null {
    return target === undefined
      ? clonePixelRect(this.currentViewport)
      : clonePixelRect(this.currentViewport, target)
  }

  setScissor(rect: RenderPixelRectLike | null): void
  setScissor(x: number, y: number, width: number, height: number): void
  setScissor(rectOrX: RenderPixelRectLike | null | number, y?: number, width?: number, height?: number): void {
    this.currentScissor = rendererStatePixelRect(rectOrX, y, width, height, 'Renderer.setScissor')
  }

  getScissor(): RenderPixelRectLike | null
  getScissor<T extends RenderPixelRectLike>(target: T): T | null
  getScissor(target?: RenderPixelRectLike): RenderPixelRectLike | null {
    return target === undefined
      ? clonePixelRect(this.currentScissor)
      : clonePixelRect(this.currentScissor, target)
  }

  setScissorTest(enabled: boolean): void {
    if (typeof enabled !== 'boolean') {
      throw new TypeError(`Renderer.setScissorTest enabled must be a boolean; received ${String(enabled)}.`)
    }
    this.currentScissorTest = enabled
  }

  getScissorTest(): boolean {
    return this.currentScissorTest
  }

  clear(color = true, depth = true, stencil = true): void {
    assertOptionalBoolean(color, 'Renderer.clear color')
    assertOptionalBoolean(depth, 'Renderer.clear depth')
    assertOptionalBoolean(stencil, 'Renderer.clear stencil')
    if (this.currentRenderTarget) {
      if (color) {
        clearRenderTargetColor(
          this.currentRenderTarget,
          this.currentClearColor,
          this.currentSize,
          this.currentScissor,
          this.currentScissorTest,
          this.currentActiveCubeFace,
          this.currentActiveMipmapLevel,
        )
      }
      if (depth) {
        clearRenderTargetDepth(
          this.currentRenderTarget,
          this.currentClearDepth,
          this.currentSize,
          this.currentScissor,
          this.currentScissorTest,
          this.currentActiveCubeFace,
          this.currentActiveMipmapLevel,
        )
      }
      if (stencil) {
        clearRenderTargetStencil(
          this.currentRenderTarget,
          this.currentClearStencil,
          this.currentClearDepth,
          this.currentSize,
          this.currentScissor,
          this.currentScissorTest,
          this.currentActiveCubeFace,
          this.currentActiveMipmapLevel,
        )
      }
    }
  }

  async clearAsync(color = true, depth = true, stencil = true): Promise<void> {
    this.clear(color, depth, stencil)
  }

  clearTarget(target: RenderTargetLike | null, color = true, depth = true, stencil = true): void {
    if (target !== null) {
      assertRenderTargetLike(target, 'Renderer.clearTarget target')
    }
    assertOptionalBoolean(color, 'Renderer.clearTarget color')
    assertOptionalBoolean(depth, 'Renderer.clearTarget depth')
    assertOptionalBoolean(stencil, 'Renderer.clearTarget stencil')

    const previousTarget = this.currentRenderTarget
    const previousActiveCubeFace = this.currentActiveCubeFace
    const previousActiveMipmapLevel = this.currentActiveMipmapLevel
    this.setRenderTarget(target)
    this.clear(color, depth, stencil)
    this.setRenderTarget(previousTarget, previousActiveCubeFace, previousActiveMipmapLevel)
  }

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

  dispose(): void {
    this.info.dispose()
    this.properties.dispose()
    this.renderLists.dispose()
    this.renderStates.dispose()
    // Native resources are owned by the renderer instance and released with normal object lifetime.
  }

  resetState(): void {
    // Native render state is rebuilt for each pass, so there is no persistent GL state to reset.
  }

  resetGLState(): void {
    this.resetState()
  }

  render(scene: ThreeSceneRootLike, camera: ThreeRenderCameraLike, options: RenderOptions = {}): Buffer {
    validateThreeSceneRoot(scene)
    validateTopLevelRenderCamera(camera)
    assertRenderOptionsLike(options, 'options')
    const renderOptions = this.resolveRenderOptions(
      options,
      isCubeCamera(camera) ? options.target ?? camera.renderTarget : options.target ?? this.currentRenderTarget,
    )
    if (isCubeCamera(camera)) {
      const { buffer } = renderCubeCamera(
        scene,
        camera,
        renderOptions,
        (targetScene, targetCamera) => this.native.render(targetScene, targetCamera),
      )
      return buffer
    }

    if (renderOptions.target === undefined && this.currentRenderTarget !== null) {
      return this.renderCurrentRenderTarget(scene, camera, renderOptions)
    }

    if (renderOptions.target) assertNonCubeCameraRenderTargetTextures(renderOptions.target)

    if (isArrayCamera(camera)) {
      const { buffer, width, height, objectIdEntries, depthData } = renderArrayCamera(
        scene,
        camera,
        renderOptions,
        (targetScene, targetCamera) => this.native.render(targetScene, targetCamera),
      )
      if (renderOptions.target) {
        const auxiliary = renderArrayCameraAuxiliaryTargetAttachments(
          scene,
          camera,
          renderOptions.target,
          renderOptions,
          buffer,
          objectIdEntries,
          (targetScene, targetCamera) => this.native.render(targetScene, targetCamera),
        )
        writeRenderTarget(
          renderOptions.target,
          buffer,
          width,
          height,
          auxiliary.objectIdEntries,
          depthData,
          auxiliary.attachments,
        )
      }
      return buffer
    }

    const { buffer, nativeScene, nativeCamera, objectIdEntries } = this.renderNative(scene, camera, renderOptions)
    if (renderOptions.target) {
      const depthData = renderTargetDepthBuffer(
        renderOptions.target,
        nativeScene,
        nativeCamera,
        (targetScene, targetCamera) => this.native.render(targetScene, targetCamera),
      )
      const auxiliary = renderRegularCameraAuxiliaryTargetAttachments(
        scene,
        camera,
        renderOptions.target,
        renderOptions,
        buffer,
        objectIdEntries,
        (targetScene, targetCamera) => this.native.render(targetScene, targetCamera),
      )
      writeRenderTarget(
        renderOptions.target,
        buffer,
        nativeScene.width!,
        nativeScene.height!,
        auxiliary.objectIdEntries,
        depthData,
        auxiliary.attachments,
      )
    }
    return buffer
  }

  async renderAsync(scene: ThreeSceneRootLike, camera: ThreeRenderCameraLike, options: RenderOptions = {}): Promise<Buffer> {
    return this.render(scene, camera, options)
  }

  renderToTarget(
    scene: ThreeSceneRootLike,
    camera: ThreeRenderCameraLike,
    target: RenderTargetLike = {},
    options: RenderOptions = {},
  ): RenderTargetLike {
    validateThreeSceneRoot(scene)
    validateTopLevelRenderCamera(camera)
    assertRenderTargetLike(target, 'target')
    assertRenderOptionsLike(options, 'options')
    const targetOptions: RenderOptions = this.resolveRenderOptions({ ...options, target, format: options.format ?? 'rgba' }, target)
    if (isCubeCamera(camera)) {
      const { target: cubeTarget } = renderCubeCamera(
        scene,
        camera,
        targetOptions,
        (targetScene, targetCamera) => this.native.render(targetScene, targetCamera),
      )
      return cubeTarget
    }

    assertNonCubeCameraRenderTargetTextures(target)

    if (isArrayCamera(camera)) {
      const { buffer, width, height, objectIdEntries, depthData } = renderArrayCamera(
        scene,
        camera,
        targetOptions,
        (targetScene, targetCamera) => this.native.render(targetScene, targetCamera),
      )
      const auxiliary = renderArrayCameraAuxiliaryTargetAttachments(
        scene,
        camera,
        target,
        targetOptions,
        buffer,
        objectIdEntries,
        (targetScene, targetCamera) => this.native.render(targetScene, targetCamera),
      )
      return writeRenderTarget(
        target,
        buffer,
        width,
        height,
        auxiliary.objectIdEntries,
        depthData,
        auxiliary.attachments,
      )
    }

    const { buffer, nativeScene, nativeCamera, objectIdEntries } = this.renderNative(scene, camera, targetOptions)
    const depthData = renderTargetDepthBuffer(
      target,
      nativeScene,
      nativeCamera,
      (targetScene, targetCamera) => this.native.render(targetScene, targetCamera),
    )
    const auxiliary = renderRegularCameraAuxiliaryTargetAttachments(
      scene,
      camera,
      target,
      targetOptions,
      buffer,
      objectIdEntries,
      (targetScene, targetCamera) => this.native.render(targetScene, targetCamera),
    )
    return writeRenderTarget(
      target,
      buffer,
      nativeScene.width!,
      nativeScene.height!,
      auxiliary.objectIdEntries,
      depthData,
      auxiliary.attachments,
    )
  }

  readRenderTargetPixels(
    target: RenderTargetLike,
    x: number,
    y: number,
    width: number,
    height: number,
    buffer: NonNullable<RenderTargetImageLike['data']>,
    activeCubeFaceIndex?: number,
    textureIndex = 0,
  ): void {
    const readback = renderTargetReadbackSource(
      target,
      activeCubeFaceIndex,
      textureIndex,
      'Renderer.readRenderTargetPixels',
    )
    copyRenderTargetReadbackPixels(readback, x, y, width, height, buffer, 'Renderer.readRenderTargetPixels')
  }

  async readRenderTargetPixelsAsync(
    target: RenderTargetLike,
    x: number,
    y: number,
    width: number,
    height: number,
    bufferOrTextureIndex?: NonNullable<RenderTargetImageLike['data']> | number,
    activeCubeFaceIndexOrFaceIndex?: number,
    textureIndex = 0,
  ): Promise<NonNullable<RenderTargetImageLike['data']>> {
    let buffer: NonNullable<RenderTargetImageLike['data']> | undefined
    let targetTextureIndex = textureIndex
    if (typeof bufferOrTextureIndex === 'number') {
      targetTextureIndex = bufferOrTextureIndex
    } else {
      buffer = bufferOrTextureIndex
    }
    const activeCubeFaceIndex = activeCubeFaceIndexOrFaceIndex
    const readback = renderTargetReadbackSource(
      target,
      activeCubeFaceIndex,
      targetTextureIndex,
      'Renderer.readRenderTargetPixelsAsync',
    )
    const rect = readbackRect(x, y, width, height, 'Renderer.readRenderTargetPixelsAsync')
    if (rect.x + rect.width > readback.width || rect.y + rect.height > readback.height) {
      throw new Error('Renderer.readRenderTargetPixelsAsync requested read bounds are out of range.')
    }
    const output = buffer === undefined
      ? createRenderTargetReadbackBuffer(readback.data, rect.width * rect.height * readback.channels)
      : buffer
    copyRenderTargetReadbackPixels(readback, x, y, width, height, output, 'Renderer.readRenderTargetPixelsAsync')
    return output
  }

  private renderCurrentRenderTarget(
    scene: ThreeSceneRootLike,
    camera: ThreeCameraLike,
    options: RenderOptions,
  ): Buffer {
    const target = this.currentRenderTarget!
    const targetOptions: RenderOptions = { ...options, target, format: options.format ?? 'rgba' }

    if (isCubeRenderTarget(target)) {
      if (isArrayCamera(camera)) {
        throw new Error(
          'THREE.ArrayCamera cannot render into an active cube render target. Render each cube face with a regular THREE.Camera or pass a THREE.CubeCamera as the top-level camera.',
        )
      }
      return this.renderCurrentCubeFace(scene, camera, target, targetOptions)
    }

    assertNonCubeCameraRenderTargetTextures(target)

    if (isArrayCamera(camera)) {
      const { buffer, width, height, objectIdEntries, depthData } = renderArrayCamera(
        scene,
        camera,
        targetOptions,
        (targetScene, targetCamera) => this.native.render(targetScene, targetCamera),
      )
      const auxiliary = renderArrayCameraAuxiliaryTargetAttachments(
        scene,
        camera,
        target,
        targetOptions,
        buffer,
        objectIdEntries,
        (targetScene, targetCamera) => this.native.render(targetScene, targetCamera),
      )
      const outputBuffer = compositeActiveTargetColorBuffer(target, buffer, width, height, targetOptions, this.autoClear, scene)
      writeRenderTarget(
        target,
        outputBuffer,
        width,
        height,
        auxiliary.objectIdEntries,
        depthData,
        auxiliary.attachments,
      )
      return buffer
    }

    const { buffer, nativeScene, nativeCamera, objectIdEntries } = this.renderNative(scene, camera, targetOptions)
    const depthData = renderTargetDepthBuffer(
      target,
      nativeScene,
      nativeCamera,
      (targetScene, targetCamera) => this.native.render(targetScene, targetCamera),
    )
    const auxiliary = renderRegularCameraAuxiliaryTargetAttachments(
      scene,
      camera,
      target,
      targetOptions,
      buffer,
      objectIdEntries,
      (targetScene, targetCamera) => this.native.render(targetScene, targetCamera),
    )
    const outputBuffer = compositeActiveTargetColorBuffer(
      target,
      buffer,
      nativeScene.width!,
      nativeScene.height!,
      targetOptions,
      this.autoClear,
      scene,
    )
    writeRenderTarget(
      target,
      outputBuffer,
      nativeScene.width!,
      nativeScene.height!,
      auxiliary.objectIdEntries,
      depthData,
      auxiliary.attachments,
    )
    return buffer
  }

  private renderCurrentCubeFace(
    scene: ThreeSceneRootLike,
    camera: ThreeCameraLike,
    target: RenderTargetLike,
    options: RenderOptions,
  ): Buffer {
    const { width: targetWidth, height: targetHeight } = resolveCubeTargetSize(target, options)
    const activeMipmapLevel = resolveActiveMipmapLevel(
      this.currentActiveMipmapLevel,
      targetWidth,
      'Renderer activeMipmapLevel',
    )
    const { width, height } = cubeMipmapSize(targetWidth, targetHeight, activeMipmapLevel)
    const faceOptions: InternalRenderOptions = {
      ...options,
      target,
      width,
      height,
      format: 'rgba',
      viewport: cubeMipmapViewport(options, target, activeMipmapLevel),
      scissor: cubeMipmapScissor(options, target, activeMipmapLevel),
      __headlessThreeViewportLabel: cubeMipmapViewportLabel(options),
      __headlessThreeScissorLabel: cubeMipmapScissorLabel(options, target),
    }
    const { buffer, nativeScene, nativeCamera, objectIdEntries } = this.renderNative(scene, camera, faceOptions)
    const depthData = renderTargetDepthBuffer(
      target,
      nativeScene,
      nativeCamera,
      (targetScene, targetCamera) => this.native.render(targetScene, targetCamera),
    )
    const auxiliary = renderRegularCameraAuxiliaryTargetAttachments(
      scene,
      camera,
      target,
      faceOptions,
      buffer,
      objectIdEntries,
      (targetScene, targetCamera) => this.native.render(targetScene, targetCamera),
    )
    writeCubeRenderTargetFace(
      target,
      buffer,
      targetWidth,
      targetHeight,
      width,
      height,
      this.currentActiveCubeFace,
      activeMipmapLevel,
      depthData ? cloneTargetData(depthTextureData(target.depthTexture!, depthData)) : undefined,
      auxiliary.objectIdEntries,
      auxiliary.attachments,
    )
    return buffer
  }

  private renderNative(
    scene: ThreeSceneRootLike,
    camera: ThreeCameraLike,
    options: RenderOptions,
  ): { buffer: Buffer; nativeScene: NativeRenderScene; nativeCamera: NativeCamera; objectIdEntries?: RenderObjectIdEntry[] } {
    const { nativeScene, nativeCamera, objectIdEntries } = toNativeInput(scene, camera, options, this.sceneExtractionCache)
    return { buffer: this.native.render(nativeScene, nativeCamera), nativeScene, nativeCamera, objectIdEntries }
  }

  private resolveRenderOptions(options: RenderOptions, fallbackTarget: RenderTargetLike | null | undefined = options.target): InternalRenderOptions {
    const sizeOptions = this.optionsWithRendererSizeFallback(options, fallbackTarget)
    const hasExplicitClippingPlanes = sizeOptions.clippingPlanes !== undefined
    return {
      ...sizeOptions,
      clippingPlanes: hasExplicitClippingPlanes ? sizeOptions.clippingPlanes : this.clippingPlanes,
      outputColorSpace: sizeOptions.outputColorSpace ?? this.outputColorSpace,
      localClippingEnabled: sizeOptions.localClippingEnabled ?? this.localClippingEnabled,
      sortObjects: sizeOptions.sortObjects ?? this.sortObjects,
      opaqueSort: sizeOptions.opaqueSort === undefined ? this.opaqueSort : sizeOptions.opaqueSort,
      transparentSort: sizeOptions.transparentSort === undefined ? this.transparentSort : sizeOptions.transparentSort,
      __headlessThreeRendererOpaque: sizeOptions.opaque ?? this.opaque,
      __headlessThreeRendererTransparent: sizeOptions.transparent ?? this.transparent,
      __headlessThreeClippingPlanesLabel: hasExplicitClippingPlanes ? undefined : 'Renderer.clippingPlanes',
      __headlessThreeRendererClearColor: cloneColor4(this.currentClearColor),
      __headlessThreeRendererViewport: clonePixelRect(this.currentViewport),
      __headlessThreeRendererScissor: clonePixelRect(this.currentScissor),
      __headlessThreeRendererScissorTest: this.currentScissorTest,
      __headlessThreeRendererShadowMapEnabled: this.shadowMap.enabled,
      __headlessThreeRendererShadowMapType: this.shadowMap.type,
      __headlessThreeRendererToneMapping: sizeOptions.toneMapping ?? this.toneMapping,
      __headlessThreeRendererToneMappingExposure: sizeOptions.toneMappingExposure ?? this.toneMappingExposure,
      __headlessThreeRendererTransmissionResolutionScale: sizeOptions.transmissionResolutionScale ?? this.transmissionResolutionScale,
      __headlessThreeRenderer: this,
    }
  }

  private optionsWithRendererSizeFallback(
    options: RenderOptions,
    fallbackTarget: RenderTargetLike | null | undefined,
  ): RenderOptions {
    if (
      this.currentSize === null ||
      options.width != null ||
      options.height != null ||
      renderTargetHasExplicitSize(fallbackTarget)
    ) {
      return options
    }
    return { ...options, width: this.currentSize.width, height: this.currentSize.height }
  }
}

export function render(scene: ThreeSceneRootLike, camera: ThreeRenderCameraLike, options: RenderOptions = {}): Buffer {
  validateThreeSceneRoot(scene)
  validateTopLevelRenderCamera(camera)
  assertRenderOptionsLike(options, 'options')
  if (isCubeCamera(camera)) {
    const { buffer } = renderCubeCamera(scene, camera, options, native.renderNative)
    return buffer
  }

  if (options.target) assertNonCubeCameraRenderTargetTextures(options.target)

  if (isArrayCamera(camera)) {
    const { buffer, width, height, objectIdEntries, depthData } = renderArrayCamera(scene, camera, options, native.renderNative)
    if (options.target) {
      const auxiliary = renderArrayCameraAuxiliaryTargetAttachments(
        scene,
        camera,
        options.target,
        options,
        buffer,
        objectIdEntries,
        native.renderNative,
      )
      writeRenderTarget(
        options.target,
        buffer,
        width,
        height,
        auxiliary.objectIdEntries,
        depthData,
        auxiliary.attachments,
      )
    }
    return buffer
  }

  const { nativeScene, nativeCamera, objectIdEntries } = toNativeInput(scene, camera, options)
  const buffer = native.renderNative(nativeScene, nativeCamera)
  if (options.target) {
    const depthData = renderTargetDepthBuffer(options.target, nativeScene, nativeCamera, native.renderNative)
    const auxiliary = renderRegularCameraAuxiliaryTargetAttachments(
      scene,
      camera,
      options.target,
      options,
      buffer,
      objectIdEntries,
      native.renderNative,
    )
    writeRenderTarget(
      options.target,
      buffer,
      nativeScene.width!,
      nativeScene.height!,
      auxiliary.objectIdEntries,
      depthData,
      auxiliary.attachments,
    )
  }
  return buffer
}

export function renderToTarget(
  scene: ThreeSceneRootLike,
  camera: ThreeRenderCameraLike,
  target: RenderTargetLike = {},
  options: RenderOptions = {},
): RenderTargetLike {
  validateThreeSceneRoot(scene)
  validateTopLevelRenderCamera(camera)
  assertRenderTargetLike(target, 'target')
  assertRenderOptionsLike(options, 'options')
  const targetOptions: RenderOptions = { ...options, target, format: options.format ?? 'rgba' }
  if (isCubeCamera(camera)) {
    const { target: cubeTarget } = renderCubeCamera(scene, camera, targetOptions, native.renderNative)
    return cubeTarget
  }

  assertNonCubeCameraRenderTargetTextures(target)

  if (isArrayCamera(camera)) {
    const { buffer, width, height, objectIdEntries, depthData } = renderArrayCamera(scene, camera, targetOptions, native.renderNative)
    const auxiliary = renderArrayCameraAuxiliaryTargetAttachments(
      scene,
      camera,
      target,
      targetOptions,
      buffer,
      objectIdEntries,
      native.renderNative,
    )
    return writeRenderTarget(
      target,
      buffer,
      width,
      height,
      auxiliary.objectIdEntries,
      depthData,
      auxiliary.attachments,
    )
  }

  const { nativeScene, nativeCamera, objectIdEntries } = toNativeInput(scene, camera, targetOptions)
  const buffer = native.renderNative(nativeScene, nativeCamera)
  const depthData = renderTargetDepthBuffer(target, nativeScene, nativeCamera, native.renderNative)
  const auxiliary = renderRegularCameraAuxiliaryTargetAttachments(
    scene,
    camera,
    target,
    targetOptions,
    buffer,
    objectIdEntries,
    native.renderNative,
  )
  return writeRenderTarget(
    target,
    buffer,
    nativeScene.width!,
    nativeScene.height!,
    auxiliary.objectIdEntries,
    depthData,
    auxiliary.attachments,
  )
}

function toNativeInput(
  scene: ThreeSceneRootLike,
  camera: ThreeCameraLike,
  options: RenderOptions,
  sceneExtractionCache?: SceneExtractionCache,
): { nativeScene: NativeRenderScene; nativeCamera: NativeCamera; objectIdEntries?: RenderObjectIdEntry[] } {
  validateThreeSceneRoot(scene)
  validateThreeCamera(camera)
  validateUnsupportedRenderOptions(options)
  validateObjectChildrenTree(scene)
  const renderMode = normalizedRenderMode(options.renderMode)
  const colorMode = renderMode === 'color'

  if (typeof scene.updateMatrixWorld === 'function') {
    scene.updateMatrixWorld(true)
  }
  if (typeof camera.updateMatrixWorld === 'function') {
    camera.updateMatrixWorld()
  }

  const size = resolveSize(camera, options)
  const overrideMaterial = colorMode ? resolveSceneOverrideMaterial(scene) : undefined
  const environment = colorMode ? resolveEnvironmentMap(scene, options.environmentIntensity, overrideMaterial) : { envMap: null }
  const envMap = environment.envMap
  const hasEnvironmentRotationOverride = options.environmentRotation !== undefined
  const environmentRotation = environment.rotation ?? (
    hasEnvironmentRotationOverride ? options.environmentRotation : scene.environmentRotation
  )
  const environmentRotationLabel = environment.rotation
    ? 'material.envMapRotation'
    : hasEnvironmentRotationOverride ? 'options.environmentRotation' : 'scene.environmentRotation'
  const environmentMapRotation = colorMode
    ? environmentRotationToNative(environmentRotation, envMap, environmentRotationLabel)
    : undefined
  const hasBackgroundOverride = options.background !== undefined
  const optionBackgroundTexture = colorMode && hasBackgroundOverride
    ? extractBackgroundTexture(options.background, 'options.background')
    : null
  const backgroundTexture = colorMode
    ? optionBackgroundTexture ?? (hasBackgroundOverride ? null : extractBackgroundTexture(scene.background, 'scene.background'))
    : null
  const hasBackgroundRotationOverride = options.backgroundRotation !== undefined
  const backgroundRotation = hasBackgroundOverride
    ? options.backgroundRotation
    : hasBackgroundRotationOverride ? options.backgroundRotation : scene.backgroundRotation
  const backgroundTextureRotation = colorMode
    ? backgroundRotationToNative(
      backgroundRotation,
      backgroundTexture,
      hasBackgroundOverride || hasBackgroundRotationOverride ? 'options.backgroundRotation' : 'scene.backgroundRotation',
    )
    : undefined
  const backgroundTextureBlurriness = colorMode && backgroundTexture
    ? optionalNormalizedFiniteNumber(
      hasBackgroundOverride ? options.backgroundBlurriness : options.backgroundBlurriness ?? scene.backgroundBlurriness,
      hasBackgroundOverride || options.backgroundBlurriness !== undefined ? 'options.backgroundBlurriness' : 'scene.backgroundBlurriness',
    )
    : undefined
  const backgroundIntensity = colorMode
    ? optionalNonNegativeFiniteNumber(
      hasBackgroundOverride ? options.backgroundIntensity : options.backgroundIntensity ?? scene.backgroundIntensity,
      hasBackgroundOverride || options.backgroundIntensity !== undefined ? 'options.backgroundIntensity' : 'scene.backgroundIntensity',
    )
    : undefined
  const clippingPlanes = extractClippingPlanes(
    options.clippingPlanes,
    (options as InternalRenderOptions).__headlessThreeClippingPlanesLabel ?? 'options.clippingPlanes',
  )
  const rendererShadowMapEnabled = (options as InternalRenderOptions).__headlessThreeRendererShadowMapEnabled !== false
  const rendererShadowMapType = (options as InternalRenderOptions).__headlessThreeRendererShadowMapType ?? PCFShadowMap
  const rendererToneMapping = (options as InternalRenderOptions).__headlessThreeRendererToneMapping ?? ACESFilmicToneMapping
  const toneMappingExposure = (options as InternalRenderOptions).__headlessThreeRendererToneMappingExposure ?? 1
  const rendererCallbackContext = colorMode && (options as InternalRenderOptions).__headlessThreeRenderer !== undefined
    ? { renderer: (options as InternalRenderOptions).__headlessThreeRenderer, scene }
    : undefined
  const extractedLights: NativeSceneLight[] | undefined = colorMode ? extractLights(scene, camera) : []
  const lights = rendererShadowMapEnabled ? extractedLights : nativeLightsWithoutShadows(extractedLights)
  const shadowMaterialMode = colorMode ? shadowMaterialModeForLights(lights) : undefined
  const materialContext: MaterialExtractionContext = {
    ...(environment.materialContext ?? {}),
    textureCache: sceneExtractionCache?.texturePayloads,
    materialColorCache: sceneExtractionCache?.materialColors,
    textureStateCache: sceneExtractionCache?.textureStates,
    materialRenderStateCache: sceneExtractionCache?.materialRenderStates,
    materialScalarFeatureCache: sceneExtractionCache?.materialScalarFeatures,
  }
  const flattenedMeshes = flattenScene(
    scene,
    camera,
    size.height,
    clippingPlanes,
    options.localClippingEnabled !== false,
    shadowMaterialMode,
    materialContext,
    {
      sortObjects: options.sortObjects,
      opaqueSort: options.opaqueSort,
      transparentSort: options.transparentSort,
      opaque: (options as InternalRenderOptions).__headlessThreeRendererOpaque,
      transparent: (options as InternalRenderOptions).__headlessThreeRendererTransparent,
    },
    overrideMaterial,
    sceneExtractionCache,
    rendererCallbackContext,
  )
  const objectIdEntries = renderMode === 'object-id' ? objectIdEntriesForMeshes(flattenedMeshes) : undefined
  const meshes = renderMode === 'depth'
    ? flattenedMeshes.map(depthReadbackMesh)
    : applyRendererToneMapping(applyRenderMode(flattenedMeshes, renderMode), rendererToneMapping)
  const viewport = normalizeOptionalPixelRect(
    effectiveViewport(options),
    size.width,
    size.height,
    effectiveViewportLabel(options),
  )
  const scissor = normalizeOptionalPixelRect(
    effectiveScissor(options),
    size.width,
    size.height,
    effectiveScissorLabel(options),
  )
  const nativeScene: NativeRenderScene = {
    width: size.width,
    height: size.height,
    background: colorMode
      ? resolveBackground(
        scene,
        options,
        backgroundTexture != null,
        (options as InternalRenderOptions).__headlessThreeRendererClearColor,
        options.outputColorSpace,
      )
      : [0, 0, 0, 1],
    backgroundIntensity,
    viewport: pixelRectToArray(viewport),
    scissor: pixelRectToArray(scissor),
    backgroundTexture: backgroundTexture?.data,
    backgroundTextureWidth: backgroundTexture?.width,
    backgroundTextureHeight: backgroundTexture?.height,
    backgroundTextureWrapS: backgroundTexture?.wrapS,
    backgroundTextureWrapT: backgroundTexture?.wrapT,
    backgroundTextureMagFilter: backgroundTexture?.magFilter,
    backgroundTextureMinFilter: backgroundTexture?.minFilter,
    backgroundTextureAnisotropy: backgroundTexture?.anisotropy,
    backgroundTextureTransform: backgroundTexture?.transform,
    backgroundTextureColorSpace: backgroundTexture?.colorSpace,
    backgroundTextureMapping: backgroundTexture?.mapping,
    backgroundTextureRotation,
    backgroundTextureBlurriness,
    format: options.format ?? (options.target ? 'rgba' : 'png'),
    outputColorSpace: renderMode === 'depth' ? 'srgb-linear' : options.outputColorSpace,
    toneMapping: renderMode === 'depth' ? undefined : rendererToneMapping,
    toneMappingExposure: renderMode === 'depth' ? undefined : toneMappingExposure,
    transmissionResolutionScale: (options as InternalRenderOptions).__headlessThreeRendererTransmissionResolutionScale,
    sampleCount: renderMode === 'depth' ? 1 : resolveSampleCount(options),
    shadowMapType: rendererShadowMapType,
    meshes,
    lights,
    ambientLight: colorMode ? extractAmbientLight(scene, camera) ?? undefined : undefined,
    ambientIntensity: colorMode ? extractAmbientIntensity(scene, camera) ?? undefined : undefined,
    lightProbe: colorMode ? extractLightProbe(scene, camera) ?? undefined : undefined,
    environmentMap: envMap?.data,
    environmentMapWidth: envMap?.width,
    environmentMapHeight: envMap?.height,
    environmentMapIntensity: envMap?.intensity,
    environmentMapColorSpace: envMap?.colorSpace,
    environmentMapRotation,
    ...(colorMode ? fogToNative(scene.fog) : {}),
    ...(colorMode ? postProcessingToNative(options.postProcessing) : {}),
  }
  const clipDistances = cameraClipDistances(camera)
  const nativeCamera: NativeCamera = {
    width: size.width,
    height: size.height,
    near: clipDistances.near,
    far: clipDistances.far,
    viewProjection: cameraViewProjection(camera),
    viewMatrix: cameraViewMatrix(camera),
    cameraPosition: cameraWorldPosition(camera),
  }

  return { nativeScene, nativeCamera, objectIdEntries }
}

function normalizedRenderMode(mode: RenderOptions['renderMode']): RenderMode {
  if (mode == null) return 'color'
  return checkedRenderMode(mode, 'options.renderMode')
}

function checkedRenderMode(mode: unknown, label: string): RenderMode {
  if (mode === 'color' || mode === 'mask' || mode === 'object-id' || mode === 'normal' || mode === 'depth') return mode
  throw new TypeError(
    `${label} must be "color", "mask", "object-id", "normal", or "depth"; received ${String(mode)}`,
  )
}

function shadowMaterialModeForLights(lights: NativeSceneLight[] | undefined): ShadowMaterialMode | undefined {
  const shadowLight = lights?.find((light) => light.castShadow === true)
  if (!shadowLight) return undefined
  return shadowLight.lightType === 'point' ? 'distance' : 'depth'
}

function nativeLightsWithoutShadows(lights: NativeSceneLight[] | undefined): NativeSceneLight[] | undefined {
  if (!lights) return undefined
  return lights.map((light) => {
    const withoutShadow = { ...light }
    delete withoutShadow.castShadow
    return withoutShadow
  })
}

function applyRenderMode(meshes: NativeSceneMesh[], mode: RenderMode): NativeSceneMesh[] {
  if (mode === 'color') return meshes
  return meshes.map((mesh, index) => renderModeMesh(mesh, mode, index))
}

function applyRendererToneMapping(meshes: NativeSceneMesh[], toneMapping: number): NativeSceneMesh[] {
  if (toneMapping !== NoToneMapping) return meshes
  return meshes.map((mesh) => (mesh.toneMapped === false ? mesh : { ...mesh, toneMapped: false }))
}

function renderModeMesh(mesh: NativeSceneMesh, mode: Exclude<RenderMode, 'color'>, index: number): NativeSceneMesh {
  const color = mode === 'mask'
    ? [1, 1, 1, materialAlpha(mesh)] as Color4
    : mode === 'object-id'
      ? objectIdColor(mesh, index)
      : [1, 1, 1, materialAlpha(mesh)] as Color4
  return {
    positions: mesh.positions,
    indices: mesh.indices,
    normals: mesh.normals,
    colors: mesh.colors,
    color,
    transform: mesh.transform,
    uvs: mesh.uvs,
    uvs2: mesh.uvs2,
    texture: mesh.texture,
    textureWidth: mesh.textureWidth,
    textureHeight: mesh.textureHeight,
    textureWrapS: mesh.textureWrapS,
    textureWrapT: mesh.textureWrapT,
    textureMagFilter: mesh.textureMagFilter,
    textureMinFilter: mesh.textureMinFilter,
    textureTransform: mesh.textureTransform,
    textureColorSpace: mesh.textureColorSpace,
    textureUsesUv2: mesh.textureUsesUv2,
    alphaMap: mesh.alphaMap,
    alphaMapWidth: mesh.alphaMapWidth,
    alphaMapHeight: mesh.alphaMapHeight,
    alphaMapWrapS: mesh.alphaMapWrapS,
    alphaMapWrapT: mesh.alphaMapWrapT,
    alphaMapMagFilter: mesh.alphaMapMagFilter,
    alphaMapMinFilter: mesh.alphaMapMinFilter,
    alphaMapTransform: mesh.alphaMapTransform,
    alphaMapColorSpace: mesh.alphaMapColorSpace,
    alphaMapUsesUv2: mesh.alphaMapUsesUv2,
    alphaTest: mesh.alphaTest,
    alphaHash: mesh.alphaHash,
    alphaToCoverage: mesh.alphaToCoverage,
    premultipliedAlpha: mesh.premultipliedAlpha,
    toneMapped: false,
    clippingPlanes: mesh.clippingPlanes,
    clippingUnionCount: mesh.clippingUnionCount,
    blending: 'none',
    depthTest: mesh.depthTest,
    depthFunc: mesh.depthFunc,
    depthWrite: true,
    colorWrite: true,
    polygonOffset: mesh.polygonOffset,
    polygonOffsetFactor: mesh.polygonOffsetFactor,
    polygonOffsetUnits: mesh.polygonOffsetUnits,
    stencilWrite: mesh.stencilWrite,
    stencilWriteMask: mesh.stencilWriteMask,
    stencilFunc: mesh.stencilFunc,
    stencilRef: mesh.stencilRef,
    stencilFuncMask: mesh.stencilFuncMask,
    stencilFail: mesh.stencilFail,
    stencilZFail: mesh.stencilZFail,
    stencilZPass: mesh.stencilZPass,
    transparent: false,
    side: mesh.side,
    shadingModel: 'basic',
    topology: mesh.topology,
    customFragmentShader: renderModeFragment(mode, color),
    castShadow: false,
    receiveShadow: false,
    groupOrder: mesh.groupOrder,
    renderOrder: mesh.renderOrder,
    sortZ: mesh.sortZ,
    sortIndex: mesh.sortIndex,
    materialVariant: mesh.materialVariant,
    materialSortKey: mesh.materialSortKey,
  }
}

function materialAlpha(mesh: NativeSceneMesh): number {
  const alpha = mesh.color?.[3]
  return typeof alpha === 'number' && Number.isFinite(alpha) ? Math.min(1, Math.max(0, alpha)) : 1
}

function objectIdColor(mesh: NativeSceneMesh, index: number): Color4 {
  const value = encodedObjectId(mesh, index)
  return [
    ((value >> 16) & 0xff) / 255,
    ((value >> 8) & 0xff) / 255,
    (value & 0xff) / 255,
    materialAlpha(mesh),
  ]
}

function objectIdEntriesForMeshes(meshes: NativeSceneMesh[]): RenderObjectIdEntry[] {
  const entries = new Map<number, RenderObjectIdEntry>()
  meshes.forEach((mesh, index) => {
    const id = objectSortId(mesh, index)
    const encodedId = encodedObjectId(mesh, index)
    if (entries.has(encodedId)) return
    entries.set(encodedId, {
      id,
      encodedId,
      rgb: [
        (encodedId >> 16) & 0xff,
        (encodedId >> 8) & 0xff,
        encodedId & 0xff,
      ],
      hex: `#${encodedId.toString(16).padStart(6, '0')}`,
    })
  })
  return [...entries.values()].sort((a, b) => a.encodedId - b.encodedId)
}

function encodedObjectId(mesh: NativeSceneMesh, index: number): number {
  const encoded = (objectSortId(mesh, index) + 1) & 0xffffff
  return encoded === 0 ? 1 : encoded
}

function objectSortId(mesh: NativeSceneMesh, index: number): number {
  return typeof mesh.sortIndex === 'number' && Number.isSafeInteger(mesh.sortIndex) && mesh.sortIndex >= 0
    ? mesh.sortIndex
    : index
}

function renderModeFragment(mode: Exclude<RenderMode, 'color'>, color: Color4): string {
  if (mode === 'normal') {
    return [
      'let view_normal = normalize((uniforms.view * vec4<f32>(normal, 0.0)).xyz);',
      'return vec4<f32>(view_normal * 0.5 + vec3<f32>(0.5), 1.0);',
    ].join('\n')
  }
  if (mode === 'depth') return DEPTH_READBACK_FRAGMENT
  return `return vec4<f32>(${formatWgslFloat(color[0])}, ${formatWgslFloat(color[1])}, ${formatWgslFloat(color[2])}, 1.0);`
}

const DEPTH_READBACK_FRAGMENT = [
  'let frag_depth = clamp(input.position.z, 0.0, 1.0);',
  'let depth = 1.0 - frag_depth;',
  'return vec4<f32>(depth, depth, depth, 1.0);',
].join('\n')

function renderTargetDepthBuffer(
  target: RenderTargetLike | undefined,
  nativeScene: NativeRenderScene,
  nativeCamera: NativeCamera,
  renderNativeScene: (scene: NativeRenderScene, camera: NativeCamera) => Buffer,
): Buffer | undefined {
  if (target?.depthTexture == null) return undefined
  return renderNativeScene(depthReadbackScene(nativeScene), nativeCamera)
}

function renderTargetHasExplicitSize(target: RenderTargetLike | null | undefined): boolean {
  if (!target) return false
  if (target.width != null || target.height != null) return true
  const texture = cubeTargetTexture(target)
  const firstImage = Array.isArray(texture?.image) ? texture.image[0] : undefined
  return firstImage?.width != null || firstImage?.height != null
}

function renderAuxiliaryTargetAttachments(
  target: RenderTargetLike | undefined,
  options: RenderOptions,
  primaryData: Buffer,
  primaryObjectIdEntries: RenderObjectIdEntry[] | undefined,
  renderAttachment: (mode: RenderMode) => { data: Buffer; objectIdEntries?: RenderObjectIdEntry[] },
): { attachments?: RenderTargetAttachmentData[]; objectIdEntries?: RenderObjectIdEntry[] } {
  if (!target) return { objectIdEntries: primaryObjectIdEntries }
  const colorTextures = renderTargetColorTextures(target)
  if (colorTextures.length <= 1) return { objectIdEntries: primaryObjectIdEntries }

  const primaryMode = normalizedRenderMode(options.renderMode)
  let targetObjectIdEntries = primaryObjectIdEntries
  const attachments: RenderTargetAttachmentData[] = []

  for (let i = 1; i < colorTextures.length; i += 1) {
    const texture = colorTextures[i]
    const mode = renderTargetTextureRenderMode(texture, targetColorTextureLabel(i))!
    if (mode === primaryMode) {
      attachments.push({ texture, data: primaryData })
      if (mode === 'object-id') targetObjectIdEntries = primaryObjectIdEntries
      continue
    }

    const rendered = renderAttachment(mode)
    attachments.push({ texture, data: rendered.data })
    if (mode === 'object-id') targetObjectIdEntries = rendered.objectIdEntries
  }

  return { attachments, objectIdEntries: targetObjectIdEntries }
}

function renderRegularCameraAuxiliaryTargetAttachments(
  scene: ThreeSceneRootLike,
  camera: ThreeCameraLike,
  target: RenderTargetLike | undefined,
  options: RenderOptions,
  primaryData: Buffer,
  primaryObjectIdEntries: RenderObjectIdEntry[] | undefined,
  renderNativeScene: (scene: NativeRenderScene, camera: NativeCamera) => Buffer,
): { attachments?: RenderTargetAttachmentData[]; objectIdEntries?: RenderObjectIdEntry[] } {
  return renderAuxiliaryTargetAttachments(
    target,
    options,
    primaryData,
    primaryObjectIdEntries,
    (mode) => {
      const { nativeScene, nativeCamera, objectIdEntries } = toNativeInput(scene, camera, {
        ...options,
        renderMode: mode,
        format: 'rgba',
      })
      return {
        data: renderNativeScene(nativeScene, nativeCamera),
        objectIdEntries,
      }
    },
  )
}

function renderArrayCameraAuxiliaryTargetAttachments(
  scene: ThreeSceneRootLike,
  camera: ThreeCameraLike,
  target: RenderTargetLike | undefined,
  options: RenderOptions,
  primaryData: Buffer,
  primaryObjectIdEntries: RenderObjectIdEntry[] | undefined,
  renderNativeScene: RenderNativeScene,
): { attachments?: RenderTargetAttachmentData[]; objectIdEntries?: RenderObjectIdEntry[] } {
  return renderAuxiliaryTargetAttachments(
    target,
    options,
    primaryData,
    primaryObjectIdEntries,
    (mode) => {
      const rendered = renderArrayCamera(scene, camera, {
        ...options,
        renderMode: mode,
        format: 'rgba',
      }, renderNativeScene)
      return {
        data: rendered.buffer,
        objectIdEntries: rendered.objectIdEntries,
      }
    },
  )
}

function renderCubeCameraAuxiliaryTargetAttachments(
  target: RenderTargetLike,
  options: RenderOptions,
  primaryFaces: Buffer[],
  primaryObjectIdEntries: RenderObjectIdEntry[] | undefined,
  renderAttachment: (mode: RenderMode) => { faces: Buffer[]; objectIdEntries?: RenderObjectIdEntry[] },
): { attachments?: RenderCubeTargetAttachmentData[]; objectIdEntries?: RenderObjectIdEntry[] } {
  const colorTextures = renderTargetColorTextures(target)
  if (colorTextures.length <= 1) return { objectIdEntries: primaryObjectIdEntries }

  const primaryMode = normalizedRenderMode(options.renderMode)
  let targetObjectIdEntries = primaryObjectIdEntries
  const attachments: RenderCubeTargetAttachmentData[] = []

  for (let i = 1; i < colorTextures.length; i += 1) {
    const texture = colorTextures[i]
    const mode = renderTargetTextureRenderMode(texture, targetColorTextureLabel(i))!
    if (mode === primaryMode) {
      attachments.push({ texture, faces: primaryFaces })
      if (mode === 'object-id') targetObjectIdEntries = primaryObjectIdEntries
      continue
    }

    const rendered = renderAttachment(mode)
    attachments.push({ texture, faces: rendered.faces })
    if (mode === 'object-id') targetObjectIdEntries = rendered.objectIdEntries
  }

  return { attachments, objectIdEntries: targetObjectIdEntries }
}

function sortedObjectIdEntries(objectIdEntryMap: Map<number, RenderObjectIdEntry>): RenderObjectIdEntry[] | undefined {
  return objectIdEntryMap.size > 0
    ? [...objectIdEntryMap.values()].sort((a, b) => a.encodedId - b.encodedId)
    : undefined
}

function renderCubeCameraFaces(
  scene: ThreeSceneRootLike,
  subCameras: ThreeCameraLike[],
  target: RenderTargetLike,
  faceOptions: InternalRenderOptions,
  renderNativeScene: RenderNativeScene,
  includeDepth: boolean,
): {
  faces: Buffer[]
  depthFaces?: NonNullable<RenderTargetImageLike['data']>[]
  objectIdEntries?: RenderObjectIdEntry[]
} {
  const objectIdEntryMap = new Map<number, RenderObjectIdEntry>()
  const faces: Buffer[] = []
  const depthFaces: NonNullable<RenderTargetImageLike['data']>[] = []

  for (const subCamera of subCameras) {
    const { nativeScene, nativeCamera, objectIdEntries } = toNativeInput(scene, subCamera, faceOptions)
    faces.push(Buffer.from(renderNativeScene(nativeScene, nativeCamera)))
    if (objectIdEntries) {
      for (const entry of objectIdEntries) {
        objectIdEntryMap.set(entry.encodedId, entry)
      }
    }
    if (includeDepth) {
      const depthFace = renderTargetDepthBuffer(target, nativeScene, nativeCamera, renderNativeScene)
      if (depthFace) {
        depthFaces.push(cloneTargetData(depthTextureData(target.depthTexture!, depthFace)))
      }
    }
  }

  return {
    faces,
    depthFaces: depthFaces.length > 0 ? depthFaces : undefined,
    objectIdEntries: sortedObjectIdEntries(objectIdEntryMap),
  }
}

function assertCubeFaceCount(faces: unknown[], label: string): void {
  if (faces.length !== CUBE_FACE_COUNT) {
    throw new Error(
      `THREE.CubeCamera expected ${CUBE_FACE_COUNT} rendered ${label} faces, received ${faces.length}.`,
    )
  }
}

function ensureCubeTargetAttachmentTexture(texture: RenderTargetTextureLike): void {
  texture.isCubeTexture = true
  texture.needsPMREMUpdate = true
  texture.pmremVersion = (texture.pmremVersion ?? 0) + 1
}

function writeCubeTargetAttachmentFaces(
  attachment: RenderCubeTargetAttachmentData,
  faceWidth: number,
  faceHeight: number,
  activeMipmapLevel: number,
  label: string,
): void {
  assertCubeFaceCount(attachment.faces, label)
  ensureCubeTargetAttachmentTexture(attachment.texture)
  writeCubeTextureFaces(
    attachment.texture,
    attachment.faces.map((face) => colorTextureData(attachment.texture, face)),
    faceWidth,
    faceHeight,
    activeMipmapLevel,
    label,
  )
}

type RenderNativeScene = (scene: NativeRenderScene, camera: NativeCamera) => Buffer

type PixelRect = {
  x: number
  y: number
  width: number
  height: number
}
type PixelSize = {
  width: number
  height: number
}
type RenderTargetAttachmentData = {
  texture: RenderTargetTextureLike
  data: Buffer
}
type RenderCubeTargetAttachmentData = {
  texture: RenderTargetTextureLike
  faces: Buffer[]
}
type InternalRenderOptions = RenderOptions & {
  __headlessThreeViewportLabel?: string
  __headlessThreeScissorLabel?: string
  __headlessThreeRendererClearColor?: Color4
  __headlessThreeRendererViewport?: PixelRect | null
  __headlessThreeRendererScissor?: PixelRect | null
  __headlessThreeRendererScissorTest?: boolean
  __headlessThreeRendererShadowMapEnabled?: boolean
  __headlessThreeRendererShadowMapType?: number
  __headlessThreeClippingPlanesLabel?: string
  __headlessThreeRendererToneMapping?: number
  __headlessThreeRendererToneMappingExposure?: number
  __headlessThreeRendererTransmissionResolutionScale?: number
  __headlessThreeRendererOpaque?: boolean
  __headlessThreeRendererTransparent?: boolean
  __headlessThreeRenderer?: unknown
}

const CUBE_FACE_COUNT = 6
const UnsignedByteType = 1009
const ByteType = 1010
const ShortType = 1011
const UnsignedShortType = 1012
const IntType = 1013
const UnsignedIntType = 1014
const FloatType = 1015
const HalfFloatType = 1016
const UnsignedShort4444Type = 1017
const UnsignedShort5551Type = 1018
const UnsignedInt248Type = 1020
const AlphaFormat = 1021
const RGBFormat = 1022
const RGBAFormat = 1023
const LuminanceFormat = 1024
const LuminanceAlphaFormat = 1025
const DepthFormat = 1026
const DepthStencilFormat = 1027
const RedFormat = 1028
const RedIntegerFormat = 1029
const RGFormat = 1030
const RGIntegerFormat = 1031
const RGBIntegerFormat = 1032
const RGBAIntegerFormat = 1033
const UnsignedInt101111Type = 35899
const UnsignedInt5999Type = 35902

function renderCubeCamera(
  scene: ThreeSceneRootLike,
  camera: ThreeCubeCameraLike,
  options: RenderOptions,
  renderNativeScene: RenderNativeScene,
): { buffer: Buffer; target: RenderTargetLike; width: number; height: number; faces: Buffer[] } {
  validateThreeSceneRoot(scene)
  validateCubeCamera(camera, options)
  const target = options.target ?? camera.renderTarget
  if (!target) {
    throw new Error('THREE.CubeCamera rendering requires a WebGLCubeRenderTarget via camera.renderTarget or options.target.')
  }
  assertRenderTargetLike(target, options.target !== undefined ? 'options.target' : 'THREE.CubeCamera renderTarget')
  validateUnsupportedRenderTargetOptions(target)

  const { width: targetWidth, height: targetHeight } = resolveCubeTargetSize(target, options)
  const activeMipmapLevel = resolveCubeMipmapLevel(camera, targetWidth)
  const { width, height } = cubeMipmapSize(targetWidth, targetHeight, activeMipmapLevel)
  const outputFormat = options.format ?? (options.target ? 'rgba' : 'png')
  const subCameras = cubeSubCameras(camera)
  const faceOptions: InternalRenderOptions = {
    ...options,
    target,
    width,
    height,
    format: 'rgba',
    viewport: cubeMipmapViewport(options, target, activeMipmapLevel),
    scissor: cubeMipmapScissor(options, target, activeMipmapLevel),
    __headlessThreeViewportLabel: cubeMipmapViewportLabel(options),
    __headlessThreeScissorLabel: cubeMipmapScissorLabel(options, target),
  }
  const primary = renderCubeCameraFaces(scene, subCameras, target, faceOptions, renderNativeScene, true)
  const auxiliary = renderCubeCameraAuxiliaryTargetAttachments(
    target,
    options,
    primary.faces,
    primary.objectIdEntries,
    (mode) => renderCubeCameraFaces(
      scene,
      subCameras,
      target,
      {
        ...faceOptions,
        renderMode: mode,
        format: 'rgba',
      },
      renderNativeScene,
      false,
    ),
  )

  writeCubeRenderTarget(
    target,
    primary.faces,
    targetWidth,
    targetHeight,
    width,
    height,
    activeMipmapLevel,
    primary.depthFaces,
    auxiliary.objectIdEntries,
    auxiliary.attachments,
  )

  const buffer = outputFormat === 'png' ? native.encodePng(primary.faces[0], width, height) : primary.faces[0]
  return { buffer, target, width, height, faces: primary.faces }
}

function validateCubeCamera(camera: ThreeCubeCameraLike, options: RenderOptions): void {
  if (!isCubeCamera(camera)) {
    throw new TypeError('render(scene, camera) expected a THREE.CubeCamera-compatible object.')
  }
  assertSupportedOutputFormat(options.format, 'options.format')
}

function cubeSubCameras(camera: ThreeCubeCameraLike): ThreeCameraLike[] {
  const children = camera.children
  if (!Array.isArray(children)) {
    throw new TypeError('THREE.CubeCamera.children must be an array of internal perspective cameras.')
  }
  if (children.length < CUBE_FACE_COUNT) {
    throw new Error('THREE.CubeCamera requires six internal perspective cameras.')
  }
  const subCameras = children.slice(0, CUBE_FACE_COUNT)
  for (let index = 0; index < subCameras.length; index += 1) {
    validateThreeCamera(subCameras[index], `THREE.CubeCamera.children[${index}]`)
  }

  if (typeof camera.updateCoordinateSystem === 'function' && camera.coordinateSystem !== WEBGL_COORDINATE_SYSTEM) {
    camera.coordinateSystem = WEBGL_COORDINATE_SYSTEM
    camera.updateCoordinateSystem()
  }
  if (typeof camera.updateMatrixWorld === 'function') {
    camera.updateMatrixWorld(true)
  }
  for (const subCamera of subCameras) {
    if (typeof subCamera.updateMatrixWorld === 'function') {
      subCamera.updateMatrixWorld(true)
    }
  }
  return subCameras
}

function resolveCubeTargetSize(target: RenderTargetLike, options: RenderOptions): { width: number; height: number } {
  const texture = cubeTargetTexture(target)
  const firstImage = Array.isArray(texture?.image) ? texture.image[0] : undefined
  const width = options.width ?? target.width ?? firstImage?.width
  const height = options.height ?? target.height ?? firstImage?.height ?? width
  if (!Number.isInteger(width) || width! <= 0) {
    throw new TypeError('THREE.CubeCamera target width must be a positive integer.')
  }
  if (!Number.isInteger(height) || height! <= 0) {
    throw new TypeError('THREE.CubeCamera target height must be a positive integer.')
  }
  if (width !== height) {
    throw new TypeError('THREE.CubeCamera target faces must be square.')
  }
  return { width: width!, height: height! }
}

function resolveCubeMipmapLevel(camera: ThreeCubeCameraLike, targetSize: number): number {
  return resolveActiveMipmapLevel(camera.activeMipmapLevel ?? 0, targetSize, 'THREE.CubeCamera activeMipmapLevel')
}

function resolveActiveMipmapLevel(level: number, targetSize: number, label: string): number {
  if (!Number.isInteger(level) || level < 0) {
    throw new TypeError(`${label} must be a non-negative integer; received ${String(level)}.`)
  }
  const maxLevel = Math.floor(Math.log2(targetSize))
  if (level > maxLevel) {
    throw new Error(
      `${label} ${level} exceeds the maximum mip level ${maxLevel} for a ${targetSize}x${targetSize} cube target.`,
    )
  }
  return level
}

function assertActiveCubeFace(value: number, label: string): void {
  if (!Number.isInteger(value) || value < 0 || value >= CUBE_FACE_COUNT) {
    throw new TypeError(`${label} must be an integer from 0 to ${CUBE_FACE_COUNT - 1}; received ${String(value)}.`)
  }
}

function assertActiveMipmapLevel(value: number, label: string): void {
  if (!Number.isInteger(value) || value < 0) {
    throw new TypeError(`${label} must be a non-negative integer; received ${String(value)}.`)
  }
}

function cubeMipmapSize(width: number, height: number, activeMipmapLevel: number): { width: number; height: number } {
  if (activeMipmapLevel === 0) return { width, height }
  return {
    width: Math.max(1, width >> activeMipmapLevel),
    height: Math.max(1, height >> activeMipmapLevel),
  }
}

function cubeMipmapViewport(
  options: RenderOptions,
  target: RenderTargetLike,
  activeMipmapLevel: number,
): RenderPixelRectLike | null | undefined {
  if (options.viewport !== undefined) return options.viewport
  return cubeMipmapRect(target.viewport, activeMipmapLevel)
}

function cubeMipmapViewportLabel(options: RenderOptions): string | undefined {
  if (options.viewport !== undefined) return 'options.viewport'
  return 'target.viewport'
}

function cubeMipmapScissor(
  options: RenderOptions,
  target: RenderTargetLike,
  activeMipmapLevel: number,
): RenderPixelRectLike | null | undefined {
  if (options.scissor !== undefined) return options.scissor
  return target.scissorTest === true ? cubeMipmapRect(target.scissor, activeMipmapLevel) : undefined
}

function cubeMipmapScissorLabel(options: RenderOptions, target: RenderTargetLike): string | undefined {
  if (options.scissor !== undefined) return 'options.scissor'
  return target.scissorTest === true ? 'target.scissor' : undefined
}

function cubeMipmapRect(rect: RenderPixelRectLike | null | undefined, activeMipmapLevel: number): RenderPixelRectLike | null | undefined {
  if (!rect || activeMipmapLevel === 0) return rect
  const [x, y, width, height] = pixelRectComponents(rect)
  if (![x, y, width, height].every((value) => typeof value === 'number' && Number.isFinite(value))) {
    return { x, y, width, height }
  }
  return {
    x,
    y,
    width: Math.max(1, Math.floor(width / 2 ** activeMipmapLevel)),
    height: Math.max(1, Math.floor(height / 2 ** activeMipmapLevel)),
  }
}

function writeCubeRenderTarget(
  target: RenderTargetLike,
  faces: Buffer[],
  targetWidth: number,
  targetHeight: number,
  faceWidth: number,
  faceHeight: number,
  activeMipmapLevel: number,
  depthFaces?: NonNullable<RenderTargetImageLike['data']>[],
  objectIdEntries?: RenderObjectIdEntry[],
  colorAttachments?: RenderCubeTargetAttachmentData[],
): RenderTargetLike {
  assertCubeFaceCount(faces, 'color')
  target.width = targetWidth
  target.height = targetHeight
  target.data = faces[0]

  const texture = ensureCubeTargetTexture(target)
  ensureCubeTargetAttachmentTexture(texture)
  writeCubeTextureFaces(texture, faces.map((face) => colorTextureData(texture, face)), faceWidth, faceHeight, activeMipmapLevel, 'target.texture')
  if (target.depthTexture && depthFaces) {
    assertCubeFaceCount(depthFaces, 'depth')
    writeCubeTextureFaces(target.depthTexture, depthFaces, faceWidth, faceHeight, activeMipmapLevel, 'target.depthTexture')
  }
  const attachments = colorAttachments ?? []
  for (let i = 0; i < attachments.length; i += 1) {
    writeCubeTargetAttachmentFaces(
      attachments[i],
      faceWidth,
      faceHeight,
      activeMipmapLevel,
      targetColorTextureLabel(i + 1),
    )
  }
  writeObjectIdMetadata(target, objectIdEntries)
  return target
}

function writeCubeRenderTargetFace(
  target: RenderTargetLike,
  face: Buffer,
  targetWidth: number,
  targetHeight: number,
  faceWidth: number,
  faceHeight: number,
  activeCubeFace: number,
  activeMipmapLevel: number,
  depthFace?: NonNullable<RenderTargetImageLike['data']>,
  objectIdEntries?: RenderObjectIdEntry[],
  colorAttachments?: RenderTargetAttachmentData[],
): RenderTargetLike {
  assertActiveCubeFace(activeCubeFace, 'Renderer activeCubeFace')
  target.width = targetWidth
  target.height = targetHeight
  target.data = face

  const texture = ensureCubeTargetTexture(target)
  ensureCubeTargetAttachmentTexture(texture)
  writeCubeTextureFace(
    texture,
    colorTextureData(texture, face),
    faceWidth,
    faceHeight,
    activeCubeFace,
    activeMipmapLevel,
    'target.texture',
  )
  if (target.depthTexture && depthFace) {
    writeCubeTextureFace(target.depthTexture, depthFace, faceWidth, faceHeight, activeCubeFace, activeMipmapLevel, 'target.depthTexture')
  }
  const attachments = colorAttachments ?? []
  for (let i = 0; i < attachments.length; i += 1) {
    const attachment = attachments[i]
    ensureCubeTargetAttachmentTexture(attachment.texture)
    writeCubeTextureFace(
      attachment.texture,
      colorTextureData(attachment.texture, attachment.data),
      faceWidth,
      faceHeight,
      activeCubeFace,
      activeMipmapLevel,
      targetColorTextureLabel(i + 1),
    )
  }
  writeObjectIdMetadata(target, objectIdEntries)
  return target
}

function writeCubeTextureFaces(
  texture: RenderTargetTextureLike,
  faces: NonNullable<RenderTargetImageLike['data']>[],
  width: number,
  height: number,
  activeMipmapLevel: number,
  label: string,
): void {
  const images = faces.map((data) => ({ data, width, height, depth: 1 }))
  if (activeMipmapLevel === 0) {
    texture.image = images
    texture.source ??= {}
    texture.source.data = images
  } else {
    if (texture.mipmaps != null && !Array.isArray(texture.mipmaps)) {
      throw new TypeError(`${label}.mipmaps must be an array of image-like objects.`)
    }
    const mipmaps = texture.mipmaps ?? (texture.mipmaps = [])
    for (let level = 0; level <= activeMipmapLevel; level += 1) {
      mipmaps[level] ??= {}
    }
    const mipmap = mipmaps[activeMipmapLevel]
    mipmap.image = images
    mipmap.width = width
    mipmap.height = height
    mipmap.depth = 1
  }
  texture.needsUpdate = true
}

function writeCubeTextureFace(
  texture: RenderTargetTextureLike,
  data: NonNullable<RenderTargetImageLike['data']>,
  width: number,
  height: number,
  activeCubeFace: number,
  activeMipmapLevel: number,
  label: string,
): void {
  const image = { data, width, height, depth: 1 }
  if (activeMipmapLevel === 0) {
    const images = cubeTextureImages(texture.image)
    images[activeCubeFace] = image
    texture.image = images
    texture.source ??= {}
    texture.source.data = images
  } else {
    if (texture.mipmaps != null && !Array.isArray(texture.mipmaps)) {
      throw new TypeError(`${label}.mipmaps must be an array of image-like objects.`)
    }
    const mipmaps = texture.mipmaps ?? (texture.mipmaps = [])
    for (let level = 0; level <= activeMipmapLevel; level += 1) {
      mipmaps[level] ??= {}
    }
    const mipmap = mipmaps[activeMipmapLevel]
    const images = cubeTextureImages(mipmap.image)
    images[activeCubeFace] = image
    mipmap.image = images
    mipmap.width = width
    mipmap.height = height
    mipmap.depth = 1
  }
  texture.needsUpdate = true
}

function cubeTextureImages(value: RenderTargetImageLike | RenderTargetImageLike[] | undefined): RenderTargetImageLike[] {
  const images = Array.isArray(value) ? value.slice() : Array.from({ length: CUBE_FACE_COUNT }, () => ({}))
  while (images.length < CUBE_FACE_COUNT) {
    images.push({})
  }
  return images
}

function cubeTargetTexture(target: RenderTargetLike): RenderTargetTextureLike | undefined {
  return Array.isArray(target.texture)
    ? target.texture[0]
    : target.textures?.[0] ?? target.texture
}

function ensureCubeTargetTexture(target: RenderTargetLike): RenderTargetTextureLike {
  const texture = cubeTargetTexture(target)
  if (texture) return texture
  const images = Array.from({ length: CUBE_FACE_COUNT }, () => ({}))
  const created: RenderTargetTextureLike = { image: images, source: { data: images }, isCubeTexture: true }
  target.texture = created
  return created
}

function isCubeRenderTarget(target: RenderTargetLike): boolean {
  return target.isWebGLCubeRenderTarget === true ||
    cubeTargetTexture(target)?.isCubeTexture === true ||
    target.depthTexture?.isCubeTexture === true
}

function renderArrayCamera(
  scene: ThreeSceneRootLike,
  camera: ThreeCameraLike,
  options: RenderOptions,
  renderNativeScene: RenderNativeScene,
): { buffer: Buffer; width: number; height: number; objectIdEntries?: RenderObjectIdEntry[]; depthData?: Buffer } {
  validateThreeSceneRoot(scene)
  validateArrayCameraOutput(camera, options)
  validateUnsupportedRenderOptions(options)

  const size = resolveSize(camera, options)
  const subCameras = arraySubCameras(camera)
  const outputFormat = options.format ?? (options.target ? 'rgba' : 'png')
  const objectIdEntryMap = new Map<number, RenderObjectIdEntry>()
  let colorBuffer: Buffer | undefined
  let depthBuffer: Buffer | undefined

  for (const subCamera of subCameras) {
    const viewport = resolveSubCameraViewport(subCamera, options.viewport, size.width, size.height)
    const copyRect = viewport ?? { x: 0, y: 0, width: size.width, height: size.height }
    const subOptions: RenderOptions = {
      ...options,
      width: size.width,
      height: size.height,
      format: 'rgba',
      viewport: viewport ?? undefined,
    }
    const { nativeScene, nativeCamera, objectIdEntries } = toNativeInput(scene, subCamera, subOptions)
    const subBuffer = renderNativeScene(nativeScene, nativeCamera)
    if (colorBuffer == null) {
      colorBuffer = Buffer.from(subBuffer)
    } else {
      copyPixelRect(subBuffer, colorBuffer, size.width, copyRect)
    }

    if (objectIdEntries) {
      for (const entry of objectIdEntries) {
        objectIdEntryMap.set(entry.encodedId, entry)
      }
    }

    const subDepth = renderTargetDepthBuffer(options.target, nativeScene, nativeCamera, renderNativeScene)
    if (subDepth) {
      if (depthBuffer == null) {
        depthBuffer = Buffer.from(subDepth)
      } else {
        copyPixelRect(subDepth, depthBuffer, size.width, copyRect)
      }
    }
  }

  return {
    buffer: outputFormat === 'png' ? native.encodePng(colorBuffer!, size.width, size.height) : colorBuffer!,
    width: size.width,
    height: size.height,
    objectIdEntries: objectIdEntryMap.size > 0
      ? [...objectIdEntryMap.values()].sort((a, b) => a.encodedId - b.encodedId)
      : undefined,
    depthData: depthBuffer,
  }
}

function validateArrayCameraOutput(camera: ThreeCameraLike, options: RenderOptions): void {
  const cameraLike = camera as any
  if (cameraLike?.isCubeCamera === true || cameraLike?.type === 'CubeCamera') {
    throw new Error(
      'THREE.CubeCamera cannot be used as an ArrayCamera sub-camera. Pass the CubeCamera as the top-level camera with a cube render target.',
    )
  }
  if (!camera || cameraLike.isCamera !== true) {
    throw new TypeError('render(scene, camera) expects camera to be a THREE.Camera')
  }
  assertSupportedOutputFormat(options.format, 'options.format')
}

function arraySubCameras(camera: ThreeCameraLike): ThreeCameraLike[] {
  const cameras = (camera as any).cameras
  if (!Array.isArray(cameras)) {
    throw new TypeError('THREE.ArrayCamera.cameras must be an array.')
  }
  if (cameras.length === 0) {
    throw new Error('THREE.ArrayCamera requires at least one sub-camera in camera.cameras.')
  }
  for (let index = 0; index < cameras.length; index += 1) {
    validateThreeCamera(cameras[index], `THREE.ArrayCamera.cameras[${index}]`)
  }
  return cameras
}

function resolveSubCameraViewport(
  camera: ThreeCameraLike,
  fallback: RenderPixelRectLike | null | undefined,
  width: number,
  height: number,
): PixelRect | undefined {
  const viewport = cameraViewport(camera) ?? fallback
  return viewport ? normalizePixelRect(viewport, width, height, 'THREE.ArrayCamera sub-camera viewport') : undefined
}

function cameraViewport(camera: ThreeCameraLike): RenderPixelRectLike | undefined {
  const viewport = camera.viewport as any
  if (viewport == null) return undefined
  if (typeof viewport.length === 'number') {
    return [viewport[0], viewport[1], viewport[2], viewport[3]]
  }
  return {
    x: viewport.x,
    y: viewport.y,
    width: viewport.width ?? viewport.z,
    height: viewport.height ?? viewport.w,
  }
}

function normalizePixelRect(rect: RenderPixelRectLike, targetWidth: number, targetHeight: number, label: string): PixelRect {
  const [rawX, rawY, rawWidth, rawHeight] = pixelRectComponents(rect)
  if (![rawX, rawY, rawWidth, rawHeight].every((value) => typeof value === 'number' && Number.isFinite(value))) {
    throw new TypeError(`${label} must contain finite x, y, width, and height values.`)
  }
  const x = Math.round(rawX)
  const y = Math.round(rawY)
  const width = Math.round(rawWidth)
  const height = Math.round(rawHeight)
  if (x < 0 || y < 0) {
    throw new TypeError(`${label} x and y must be greater than or equal to 0.`)
  }
  if (width <= 0 || height <= 0) {
    throw new TypeError(`${label} width and height must be greater than 0.`)
  }
  if (x + width > targetWidth || y + height > targetHeight) {
    throw new TypeError(`${label} must fit inside the render target.`)
  }
  return { x, y, width, height }
}

function normalizeOptionalPixelRect(
  rect: RenderPixelRectLike | null | undefined,
  targetWidth: number,
  targetHeight: number,
  label: string,
): PixelRect | undefined {
  if (rect == null) return undefined
  return normalizePixelRect(rect, targetWidth, targetHeight, label)
}

function copyPixelRect(source: Buffer, destination: Buffer, imageWidth: number, rect: PixelRect): void {
  const rowBytes = rect.width * 4
  for (let row = 0; row < rect.height; row += 1) {
    const offset = ((rect.y + row) * imageWidth + rect.x) * 4
    source.copy(destination, offset, offset, offset + rowBytes)
  }
}

function depthReadbackScene(scene: NativeRenderScene): NativeRenderScene {
  return {
    ...scene,
    background: [0, 0, 0, 1],
    backgroundIntensity: 1,
    backgroundTexture: undefined,
    backgroundTextureWidth: undefined,
    backgroundTextureHeight: undefined,
    backgroundTextureWrapS: undefined,
    backgroundTextureWrapT: undefined,
    backgroundTextureMagFilter: undefined,
    backgroundTextureMinFilter: undefined,
    backgroundTextureAnisotropy: undefined,
    backgroundTextureTransform: undefined,
    backgroundTextureColorSpace: undefined,
    backgroundTextureMapping: undefined,
    backgroundTextureRotation: undefined,
    backgroundTextureBlurriness: undefined,
    format: 'rgba',
    outputColorSpace: 'srgb-linear',
    toneMapping: undefined,
    toneMappingExposure: undefined,
    sampleCount: 1,
    meshes: scene.meshes?.map(depthReadbackMesh),
    lights: [],
    ambientLight: undefined,
    ambientIntensity: undefined,
    lightProbe: undefined,
    environmentMap: undefined,
    environmentMapWidth: undefined,
    environmentMapHeight: undefined,
    environmentMapIntensity: undefined,
    environmentMapColorSpace: undefined,
    environmentMapRotation: undefined,
    fogType: undefined,
    fogColor: undefined,
    fogNear: undefined,
    fogFar: undefined,
    fogDensity: undefined,
    postExposure: undefined,
    postContrast: undefined,
    postSaturation: undefined,
    postVignette: undefined,
    postGrayscale: undefined,
    postInvert: undefined,
  }
}

function depthReadbackMesh(mesh: NativeSceneMesh): NativeSceneMesh {
  const writesDepth = meshWritesDepth(mesh)
  return {
    ...mesh,
    blending: 'none',
    depthWrite: writesDepth,
    colorWrite: writesDepth,
    transparent: false,
    shadingModel: 'basic',
    toneMapped: false,
    alphaToCoverage: false,
    customFragmentShader: DEPTH_READBACK_FRAGMENT,
    castShadow: false,
    receiveShadow: false,
  }
}

function meshWritesDepth(mesh: NativeSceneMesh): boolean {
  if (mesh.depthTest === false) return false
  if (typeof mesh.depthWrite === 'boolean') return mesh.depthWrite
  return true
}

function formatWgslFloat(value: number): string {
  if (value <= 0) return '0.0'
  if (value >= 1) return '1.0'
  return value.toFixed(10)
}

function fogToNative(fog: ThreeSceneRootLike['fog']): Partial<NativeRenderScene> {
  if (!fog) return {}
  if (typeof fog !== 'object') {
    throw new TypeError('scene.fog must be a THREE.Fog or THREE.FogExp2 object.')
  }
  const color = validatedColorLikeToArray(fog.color, 'scene.fog.color')
  if (fog.isFogExp2) {
    return {
      fogType: 'exp2',
      fogColor: color ?? undefined,
      fogDensity: optionalNonNegativeFiniteNumber(fog.density, 'scene.fog.density'),
    }
  }
  if (fog.isFog) {
    const clipDistances = fogClipDistances(fog)
    return {
      fogType: 'linear',
      fogColor: color ?? undefined,
      fogNear: clipDistances.fogNear,
      fogFar: clipDistances.fogFar,
    }
  }
  throw new TypeError('scene.fog must be a THREE.Fog or THREE.FogExp2 object.')
}

function postProcessingToNative(post: RenderOptions['postProcessing']): Partial<NativeRenderScene> {
  if (!post || post.enabled === false) return {}
  return {
    postExposure: finiteOrUndefined(post.exposure),
    postContrast: finiteOrUndefined(post.contrast),
    postSaturation: finiteOrUndefined(post.saturation),
    postVignette: finiteOrUndefined(post.vignette),
    postGrayscale: booleanOrNumber(post.grayscale),
    postInvert: booleanOrNumber(post.invert),
  }
}

function pixelRectToArray(rect: RenderPixelRectLike | null | undefined): number[] | undefined {
  if (!rect) return undefined
  return pixelRectComponents(rect)
}

function effectiveViewport(options: RenderOptions): RenderPixelRectLike | null | undefined {
  if (options.viewport !== undefined) return options.viewport
  if (options.target?.viewport !== undefined) return options.target.viewport
  return (options as InternalRenderOptions).__headlessThreeRendererViewport
}

function effectiveScissor(options: RenderOptions): RenderPixelRectLike | null | undefined {
  if (options.scissor !== undefined) return options.scissor
  if (options.target?.scissorTest === true) return options.target.scissor
  const internal = options as InternalRenderOptions
  return internal.__headlessThreeRendererScissorTest === true ? internal.__headlessThreeRendererScissor : undefined
}

function effectiveViewportLabel(options: RenderOptions): string {
  const internalLabel = (options as InternalRenderOptions).__headlessThreeViewportLabel
  if (internalLabel) return internalLabel
  if (options.viewport !== undefined) return 'options.viewport'
  if (options.target?.viewport !== undefined) return 'target.viewport'
  return (options as InternalRenderOptions).__headlessThreeRendererViewport !== undefined
    ? 'Renderer.viewport'
    : 'target.viewport'
}

function effectiveScissorLabel(options: RenderOptions): string {
  const internalLabel = (options as InternalRenderOptions).__headlessThreeScissorLabel
  if (internalLabel) return internalLabel
  if (options.scissor !== undefined) return 'options.scissor'
  if (options.target?.scissorTest === true) return 'target.scissor'
  return (options as InternalRenderOptions).__headlessThreeRendererScissorTest === true
    ? 'Renderer.scissor'
    : 'target.scissor'
}

function pixelRectComponents(rect: RenderPixelRectLike): number[] {
  if (typeof (rect as ArrayLike<number>).length === 'number') {
    const values = rect as ArrayLike<number>
    return [values[0], values[1], values[2], values[3]]
  }
  const values = rect as { x?: number; y?: number; width?: number; height?: number; z?: number; w?: number }
  return [values.x!, values.y!, values.width ?? values.z!, values.height ?? values.w!]
}

function rendererStatePixelRect(
  rectOrX: RenderPixelRectLike | null | number,
  y: number | undefined,
  width: number | undefined,
  height: number | undefined,
  label: string,
): PixelRect | null {
  if (rectOrX == null) return null
  if (typeof rectOrX === 'number') {
    return rendererStatePixelRectFromComponents([rectOrX, y, width, height], label)
  }
  if (typeof rectOrX !== 'object') {
    throw new TypeError(`${label} expects a rectangle object, array, or x/y/width/height numbers.`)
  }
  return rendererStatePixelRectFromComponents(pixelRectComponents(rectOrX), label)
}

function rendererStatePixelRectFromComponents(values: unknown[], label: string): PixelRect {
  const [rawX, rawY, rawWidth, rawHeight] = values
  if (![rawX, rawY, rawWidth, rawHeight].every((value) => typeof value === 'number' && Number.isFinite(value))) {
    throw new TypeError(`${label} must contain finite x, y, width, and height values.`)
  }
  const x = Math.round(rawX as number)
  const y = Math.round(rawY as number)
  const width = Math.round(rawWidth as number)
  const height = Math.round(rawHeight as number)
  if (x < 0 || y < 0) {
    throw new TypeError(`${label} x and y must be greater than or equal to 0.`)
  }
  if (width <= 0 || height <= 0) {
    throw new TypeError(`${label} width and height must be greater than 0.`)
  }
  return { x, y, width, height }
}

function assertDefaultViewportDepthRange(minDepth: unknown, maxDepth: unknown, label: string): void {
  const min = rendererViewportDepthValue(minDepth, `${label} minDepth`)
  const max = rendererViewportDepthValue(maxDepth, `${label} maxDepth`)
  if (min !== 0 || max !== 1) {
    throw new Error(`${label} depth ranges other than 0..1 are not supported by @headless-three/renderer.`)
  }
}

function rendererViewportDepthValue(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a finite number.`)
  }
  if (value < 0 || value > 1) {
    throw new TypeError(`${label} must be between 0 and 1.`)
  }
  return value
}

function rendererStateClearColor(color: number | string | ThreeColorLike | number[], alpha?: number): Color4 {
  const colorArray = typeof color === 'number'
    ? rendererStateHexColor(color, 'Renderer.setClearColor color')
    : typeof color === 'string'
      ? cssColorStringToArray(color, 'Renderer.setClearColor color')
      : validatedColorLikeToArray(color, 'Renderer.setClearColor color')
  if (!colorArray) {
    throw new TypeError('Renderer.setClearColor color must be a hex number, CSS color string, color-like object, or [r, g, b].')
  }
  return [
    colorArray[0],
    colorArray[1],
    colorArray[2],
    alpha === undefined ? colorArray[3] : rendererStateClearAlpha(alpha, 'Renderer.setClearColor alpha'),
  ]
}

function rendererStateHexColor(value: number, label: string): Color4 {
  if (!Number.isFinite(value) || !Number.isInteger(value)) {
    throw new TypeError(`${label} must be a finite integer hex color.`)
  }
  if (value < 0 || value > 0xffffff) {
    throw new TypeError(`${label} must be between 0x000000 and 0xffffff.`)
  }
  return [
    ((value >> 16) & 0xff) / 255,
    ((value >> 8) & 0xff) / 255,
    (value & 0xff) / 255,
    1,
  ]
}

function rendererStateClearAlpha(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a finite number.`)
  }
  return clamp01(value)
}

function rendererStateClearDepth(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a finite number.`)
  }
  return clamp01(value)
}

function rendererStateClearStencil(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || !Number.isInteger(value)) {
    throw new TypeError(`${label} must be a finite integer.`)
  }
  return value
}

function cloneColor4(color: Color4): Color4 {
  return [color[0], color[1], color[2], color[3]]
}

function cloneColor3(color: Color4): ThreeColorLike
function cloneColor3<T extends ThreeColorLike>(color: Color4, target: T): T
function cloneColor3<T extends ThreeColorLike>(color: Color4, target?: T): ThreeColorLike | T {
  if (target) {
    const mutable = target as any
    if (typeof mutable.setRGB === 'function') {
      mutable.setRGB(color[0], color[1], color[2])
    } else {
      mutable.r = color[0]
      mutable.g = color[1]
      mutable.b = color[2]
    }
    return target
  }
  return { isColor: true, r: color[0], g: color[1], b: color[2] }
}

function rendererStateSize(width: unknown, height: unknown, label: string): PixelSize {
  return {
    width: rendererStateSizeDimension(width, `${label} width`),
    height: rendererStateSizeDimension(height, `${label} height`),
  }
}

function rendererStateSizeDimension(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a finite number.`)
  }
  if (!Number.isInteger(value) || value <= 0) {
    throw new TypeError(`${label} must be a positive integer.`)
  }
  return value
}

function rendererStatePixelRatio(value: unknown, label: string): number {
  return rendererStatePositiveFiniteNumber(value, `${label} value`)
}

function rendererStatePositiveFiniteNumber(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a finite number.`)
  }
  if (value <= 0) {
    throw new TypeError(`${label} must be greater than 0.`)
  }
  return value
}

function rendererStateFiniteNumber(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a finite number.`)
  }
  return value
}

function rendererStateOptionalFiniteInteger(value: unknown, label: string): void {
  if (value !== undefined) {
    rendererStateClearStencil(value, label)
  }
}

function rendererStateBoolean(value: unknown, label: string): boolean {
  if (typeof value !== 'boolean') {
    throw new TypeError(`${label} must be a boolean.`)
  }
  return value
}

function rendererInfoDrawCount(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || Math.floor(value) !== value || value < 0) {
    throw new TypeError(`${label} must be a non-negative integer.`)
  }
  return value
}

function rendererInfoInstanceCount(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || Math.floor(value) !== value || value < 0) {
    throw new TypeError(`${label} must be a non-negative integer.`)
  }
  return value
}

function rendererInfoDrawMode(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || Math.floor(value) !== value) {
    throw new TypeError(`${label} must be an integer WebGL draw mode.`)
  }
  if (!SupportedRendererInfoDrawModes.has(value)) {
    throw new Error(
      `${label} ${String(value)} is not supported. Use POINTS, LINES, LINE_STRIP, LINE_LOOP, or TRIANGLES WebGL draw mode constants.`,
    )
  }
  return value
}

function rendererInfoTimestampTime(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a finite number.`)
  }
  if (value < 0) {
    throw new TypeError(`${label} must be non-negative.`)
  }
  return value
}

function assertRendererStateCullFace(value: unknown, label: string): asserts value is number {
  if (!SupportedRendererStateCullFaces.has(value as number)) {
    throw new Error(
      `${label} ${String(value)} is not supported. Use THREE.CullFaceNone, CullFaceBack, CullFaceFront, or CullFaceFrontBack.`,
    )
  }
}

function assertRendererStateBlendingMode(value: unknown, label: string): asserts value is number {
  if (!SupportedRendererStateBlendingModes.has(value as number)) {
    throw new Error(
      `${label} ${String(value)} is not supported. Use a Three.js blending constant such as NormalBlending, AdditiveBlending, or CustomBlending.`,
    )
  }
}

function throwUnsupportedRendererStateWebGl(method: string, operation: string): never {
  throw new Error(
    `Renderer.state.${method}() is not supported by @headless-three/renderer because it does not expose ${operation}. Render normal Three.js scene graphs with Renderer.render() or renderToTarget().`,
  )
}

function assertRendererParametersLike(value: RendererParametersLike | undefined, label: string): void {
  if (value === undefined) return
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be an object when provided.`)
  }
  const parameters = value as Record<string, unknown>
  for (const name of RendererBooleanParameters) {
    if (parameters[name] !== undefined) {
      rendererStateBoolean(parameters[name], `${label}.${name}`)
    }
  }
  if (parameters.powerPreference !== undefined) {
    rendererStatePowerPreference(parameters.powerPreference, `${label}.powerPreference`)
  }
  assertRendererConstructorSamples(parameters.samples, `${label}.samples`)
  assertRendererOutputBufferType(parameters.outputBufferType, `${label}.outputBufferType`)
  assertRendererContextParameterAbsent(parameters, 'canvas', label)
  assertRendererContextParameterAbsent(parameters, 'context', label)
  assertRendererUnsupportedDepthParameterFalse(parameters, 'logarithmicDepthBuffer', label)
  assertRendererUnsupportedDepthParameterFalse(parameters, 'reversedDepthBuffer', label)
  assertRendererUnsupportedDepthParameterFalse(parameters, 'reverseDepthBuffer', label)
}

function rendererStatePowerPreference(value: unknown, label: string): void {
  if (typeof value !== 'string') {
    throw new TypeError(`${label} must be a WebGL powerPreference string.`)
  }
  if (!SupportedRendererPowerPreferences.has(value)) {
    throw new TypeError(`${label} "${value}" is not supported. Use "default", "high-performance", or "low-power".`)
  }
}

function assertRendererOutputBufferType(value: unknown, label: string): void {
  if (value === undefined) return
  if (typeof value !== 'number' || !Number.isInteger(value)) {
    throw new TypeError(`${label} must be a Three.js texture type integer.`)
  }
  if (value !== UnsignedByteType) {
    throw new Error(
      `${label} ${String(value)} is not supported by @headless-three/renderer because it has no browser drawing buffer. Omit outputBufferType for RGBA8 output, or use a target texture with FloatType or HalfFloatType for typed offscreen readback.`,
    )
  }
}

function assertRendererConstructorSamples(value: unknown, label: string): void {
  if (value == null) return
  if (typeof value !== 'number' || !Number.isFinite(value) || !Number.isInteger(value) || value < 0) {
    throw new TypeError(`${label} must be a non-negative integer sample count.`)
  }
  if (value > 1) {
    throw new Error(
      `${label} ${String(value)} is not supported as constructor-level MSAA state by @headless-three/renderer. Use render options samples/sampleCount or target samples/sampleCount for 4x MSAA.`,
    )
  }
}

function rendererContextAttributes(parameters?: RendererParametersLike): RendererContextAttributesLike {
  return {
    alpha: parameters?.alpha ?? DefaultRendererContextAttributes.alpha,
    depth: parameters?.depth ?? DefaultRendererContextAttributes.depth,
    stencil: parameters?.stencil ?? DefaultRendererContextAttributes.stencil,
    antialias: parameters?.antialias ?? DefaultRendererContextAttributes.antialias,
    premultipliedAlpha: parameters?.premultipliedAlpha ?? DefaultRendererContextAttributes.premultipliedAlpha,
    preserveDrawingBuffer: parameters?.preserveDrawingBuffer ?? DefaultRendererContextAttributes.preserveDrawingBuffer,
    powerPreference: parameters?.powerPreference ?? DefaultRendererContextAttributes.powerPreference,
    failIfMajorPerformanceCaveat: parameters?.failIfMajorPerformanceCaveat
      ?? DefaultRendererContextAttributes.failIfMajorPerformanceCaveat,
  }
}

function assertRendererContextParameterAbsent(
  parameters: Record<string, unknown>,
  name: 'canvas' | 'context',
  label: string,
): void {
  if (parameters[name] === undefined || parameters[name] === null) return
  throw new Error(
    `${label}.${name} is not supported by @headless-three/renderer because it renders offscreen through wgpu instead of a browser WebGL context.`,
  )
}

function assertRendererUnsupportedDepthParameterFalse(
  parameters: Record<string, unknown>,
  name: 'logarithmicDepthBuffer' | 'reversedDepthBuffer' | 'reverseDepthBuffer',
  label: string,
): void {
  if (parameters[name] === undefined) return
  const enabled = rendererStateBoolean(parameters[name], `${label}.${name}`)
  if (!enabled) return
  throw new Error(`${label}.${name} true is not supported by @headless-three/renderer yet.`)
}

function rendererStateShadowMapType(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`Renderer.shadowMap.type must be a Three.js shadow map type constant; received ${String(value)}.`)
  }
  if (!Number.isInteger(value) || !SupportedRendererShadowMapTypes.has(value)) {
    throw new TypeError(
      `Renderer.shadowMap.type ${String(value)} is not supported by @headless-three/renderer. Use THREE.BasicShadowMap, THREE.PCFShadowMap, THREE.PCFSoftShadowMap, or THREE.VSMShadowMap.`,
    )
  }
  return value
}

function rendererStateToneMapping(value: unknown, label = 'Renderer.toneMapping'): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a Three.js tone mapping constant; received ${String(value)}.`)
  }
  if (!Number.isInteger(value) || !SupportedRendererToneMappings.has(value)) {
    throw new TypeError(
      `${label} ${String(value)} is not supported by @headless-three/renderer yet. Use THREE.NoToneMapping, THREE.LinearToneMapping, THREE.ReinhardToneMapping, THREE.CineonToneMapping, THREE.ACESFilmicToneMapping, THREE.CustomToneMapping, THREE.AgXToneMapping, or THREE.NeutralToneMapping.`,
    )
  }
  return value
}

function assertOptionalBoolean(value: unknown, label: string): void {
  rendererStateBoolean(value, label)
}

function assertEffectsArrayOrNull(value: unknown, label: string): asserts value is readonly unknown[] | null {
  if (value !== null && !Array.isArray(value)) {
    throw new TypeError(`${label} must be an array or null.`)
  }
}

function clonePixelSize(size: PixelSize | null | undefined): PixelSize | null
function clonePixelSize<T extends RenderSizeLike>(size: PixelSize | null | undefined, target: T): T | null
function clonePixelSize<T extends RenderSizeLike>(
  size: PixelSize | null | undefined,
  target?: T,
): PixelSize | T | null {
  if (!size) return null
  if (target) {
    const mutable = target as any
    if (typeof mutable.length === 'number') {
      mutable[0] = size.width
      mutable[1] = size.height
    } else {
      if (typeof mutable.set === 'function') mutable.set(size.width, size.height)
      if ('width' in mutable || 'height' in mutable || typeof mutable.set !== 'function') {
        mutable.width = size.width
        mutable.height = size.height
      }
      if ('x' in mutable || 'y' in mutable || typeof mutable.set === 'function') {
        mutable.x = size.width
        mutable.y = size.height
      }
    }
    return target
  }
  return { width: size.width, height: size.height }
}

function clonePixelRect(rect: PixelRect | null | undefined): PixelRect | null
function clonePixelRect<T extends RenderPixelRectLike>(rect: PixelRect | null | undefined, target: T): T | null
function clonePixelRect<T extends RenderPixelRectLike>(
  rect: PixelRect | null | undefined,
  target?: T,
): PixelRect | T | null {
  if (!rect) return null
  if (target) {
    const mutable = target as any
    if (typeof mutable.length === 'number') {
      mutable[0] = rect.x
      mutable[1] = rect.y
      mutable[2] = rect.width
      mutable[3] = rect.height
    } else {
      if (typeof mutable.set === 'function') mutable.set(rect.x, rect.y, rect.width, rect.height)
      mutable.x = rect.x
      mutable.y = rect.y
      mutable.width = rect.width
      mutable.height = rect.height
      mutable.z = rect.width
      mutable.w = rect.height
    }
    return target
  }
  return { x: rect.x, y: rect.y, width: rect.width, height: rect.height }
}

function finiteOrUndefined(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function optionalFiniteNumber(value: unknown, label: string): number | undefined {
  if (value == null) return undefined
  if (typeof value === 'number' && Number.isFinite(value)) return value
  throw new TypeError(`${label} must be a finite number.`)
}

function optionalNonNegativeFiniteNumber(value: unknown, label: string): number | undefined {
  const number = optionalFiniteNumber(value, label)
  if (number === undefined) return undefined
  if (number < 0) {
    throw new TypeError(`${label} must be non-negative.`)
  }
  return number
}

function finiteNonNegativeNumber(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a finite number.`)
  }
  if (value < 0) {
    throw new TypeError(`${label} must be non-negative.`)
  }
  return value
}

function optionalNormalizedFiniteNumber(value: unknown, label: string): number | undefined {
  const number = optionalFiniteNumber(value, label)
  if (number === undefined) return undefined
  if (number < 0 || number > 1) {
    throw new TypeError(`${label} must be between 0 and 1.`)
  }
  return number
}

function fogClipDistances(fog: NonNullable<ThreeSceneRootLike['fog']>): Pick<NativeRenderScene, 'fogNear' | 'fogFar'> {
  const near = optionalFiniteNumber(fog.near, 'scene.fog.near')
  const far = optionalFiniteNumber(fog.far, 'scene.fog.far')
  const effectiveNear = near ?? 1
  const effectiveFar = far ?? 1000

  if (effectiveFar <= effectiveNear) {
    if (far !== undefined) {
      throw new TypeError('scene.fog.far must be greater than scene.fog.near.')
    }
    throw new TypeError('scene.fog.near must be less than the effective scene.fog.far.')
  }

  return { fogNear: near, fogFar: far }
}

function cameraClipDistances(camera: ThreeCameraLike): Pick<NativeCamera, 'near' | 'far'> {
  const near = optionalFiniteNumber(camera.near, 'camera.near')
  const far = optionalFiniteNumber(camera.far, 'camera.far')

  if (near != null && camera.isOrthographicCamera === true && near < 0) {
    throw new TypeError('camera.near must be non-negative for orthographic cameras.')
  }
  if (near != null && camera.isOrthographicCamera !== true && near <= 0) {
    throw new TypeError('camera.near must be positive.')
  }
  if (far != null && far <= 0) {
    throw new TypeError('camera.far must be positive.')
  }
  if (near != null && far != null && far <= near) {
    throw new TypeError('camera.far must be greater than camera.near.')
  }

  return { near, far }
}

function booleanOrNumber(value: unknown): number | undefined {
  if (typeof value === 'boolean') return value ? 1 : 0
  return finiteOrUndefined(value)
}

type EulerOrder = 'XYZ' | 'YXZ' | 'ZXY' | 'ZYX' | 'YZX' | 'XZY'
type EulerComponents = { x: number; y: number; z: number; order: EulerOrder }

function backgroundRotationToNative(
  rotation: ThreeSceneRootLike['backgroundRotation'],
  backgroundTexture: { mapping?: string } | null,
  label = 'scene.backgroundRotation',
): number[] | undefined {
  const euler = optionalEulerComponents(rotation, label)
  if (!euler || !hasNonZeroEulerRotation(euler)) return undefined
  if (backgroundTexture?.mapping !== 'equirectangular') {
    throw new Error(
      `${label} is only supported for equirectangular or cube texture backgrounds by @headless-three/renderer. Leave backgroundRotation at its default for color/2D backgrounds or pre-rotate the background texture before rendering.`,
    )
  }
  const { x, y, z, order } = euler
  // Three.js negates background Euler angles before producing the rotation matrix
  // to account for the background shader's left-handed frame.
  return eulerRotationMatrix3Columns(-x, -y, -z, order)
}

function environmentRotationToNative(
  rotation: ThreeSceneRootLike['environmentRotation'],
  envMap: { data?: Buffer } | null,
  label = 'scene.environmentRotation',
): number[] | undefined {
  if (!envMap) return undefined
  const euler = optionalEulerComponents(rotation, label)
  if (!euler || !hasNonZeroEulerRotation(euler)) return undefined
  const { x, y, z, order } = euler
  return eulerRotationMatrix3Columns(-x, -y, -z, order)
}

function optionalEulerComponents(value: ThreeEulerLike | ArrayLike<number> | null | undefined, label: string): EulerComponents | null {
  if (!value) return null
  return eulerComponents(value, label)
}

function eulerComponents(value: ThreeEulerLike | ArrayLike<number>, label: string): EulerComponents {
  const rotation = value as ThreeEulerLike & { length?: number }
  if (typeof rotation.length === 'number') {
    const values = value as ArrayLike<number | string | undefined>
    return {
      x: finiteRotationComponent(values[0], `${label}[0]`),
      y: finiteRotationComponent(values[1], `${label}[1]`),
      z: finiteRotationComponent(values[2], `${label}[2]`),
      order: eulerOrder(values[3], `${label}[3]`),
    }
  }
  return {
    x: finiteRotationComponent(rotation.x, `${label}.x`),
    y: finiteRotationComponent(rotation.y, `${label}.y`),
    z: finiteRotationComponent(rotation.z, `${label}.z`),
    order: eulerOrder(rotation.order, `${label}.order`),
  }
}

function finiteRotationComponent(value: unknown, label: string): number {
  if (value == null) return 0
  if (typeof value === 'number' && Number.isFinite(value)) return value
  throw new TypeError(`${label} must be a finite number`)
}

function eulerOrder(value: unknown, label: string): EulerOrder {
  if (value == null) return 'XYZ'
  if (
    value === 'XYZ' ||
    value === 'YXZ' ||
    value === 'ZXY' ||
    value === 'ZYX' ||
    value === 'YZX' ||
    value === 'XZY'
  ) {
    return value
  }
  throw new TypeError(`${label} must be one of XYZ, YXZ, ZXY, ZYX, YZX, or XZY`)
}

function eulerRotationMatrix3Columns(x: number, y: number, z: number, order: EulerOrder): number[] {
  const a = Math.cos(x)
  const b = Math.sin(x)
  const c = Math.cos(y)
  const d = Math.sin(y)
  const e = Math.cos(z)
  const f = Math.sin(z)
  const te = new Array<number>(9).fill(0)

  if (order === 'XYZ') {
    const ae = a * e
    const af = a * f
    const be = b * e
    const bf = b * f
    te[0] = c * e
    te[3] = -c * f
    te[6] = d
    te[1] = af + be * d
    te[4] = ae - bf * d
    te[7] = -b * c
    te[2] = bf - ae * d
    te[5] = be + af * d
    te[8] = a * c
  } else if (order === 'YXZ') {
    const ce = c * e
    const cf = c * f
    const de = d * e
    const df = d * f
    te[0] = ce + df * b
    te[3] = de * b - cf
    te[6] = a * d
    te[1] = a * f
    te[4] = a * e
    te[7] = -b
    te[2] = cf * b - de
    te[5] = df + ce * b
    te[8] = a * c
  } else if (order === 'ZXY') {
    const ce = c * e
    const cf = c * f
    const de = d * e
    const df = d * f
    te[0] = ce - df * b
    te[3] = -a * f
    te[6] = de + cf * b
    te[1] = cf + de * b
    te[4] = a * e
    te[7] = df - ce * b
    te[2] = -a * d
    te[5] = b
    te[8] = a * c
  } else if (order === 'ZYX') {
    const ae = a * e
    const af = a * f
    const be = b * e
    const bf = b * f
    te[0] = c * e
    te[3] = be * d - af
    te[6] = ae * d + bf
    te[1] = c * f
    te[4] = bf * d + ae
    te[7] = af * d - be
    te[2] = -d
    te[5] = b * c
    te[8] = a * c
  } else if (order === 'YZX') {
    const ac = a * c
    const ad = a * d
    const bc = b * c
    const bd = b * d
    te[0] = c * e
    te[3] = bd - ac * f
    te[6] = bc * f + ad
    te[1] = f
    te[4] = a * e
    te[7] = -b * e
    te[2] = -d * e
    te[5] = ad * f + bc
    te[8] = ac - bd * f
  } else {
    const ac = a * c
    const ad = a * d
    const bc = b * c
    const bd = b * d
    te[0] = c * e
    te[3] = -f
    te[6] = d * e
    te[1] = ac * f + bd
    te[4] = a * e
    te[7] = ad * f - bc
    te[2] = bc * f - ad
    te[5] = b * e
    te[8] = bd * f + ac
  }

  return te
}

function hasNonZeroEulerRotation(rotation: EulerComponents): boolean {
  return Math.abs(rotation.x) > 1e-12 || Math.abs(rotation.y) > 1e-12 || Math.abs(rotation.z) > 1e-12
}

function validateUnsupportedRenderOptions(options: RenderOptions): void {
  assertSupportedOutputFormat(options.format, 'options.format')
  assertSupportedOutputColorSpace(options.outputColorSpace)
  if (options.toneMapping != null) rendererStateToneMapping(options.toneMapping, 'options.toneMapping')
  if (options.toneMappingExposure != null) finiteNonNegativeNumber(options.toneMappingExposure, 'options.toneMappingExposure')
  assertNonNegativeNumberOption(options.backgroundIntensity, 'options.backgroundIntensity')
  assertNormalizedNumberOption(options.backgroundBlurriness, 'options.backgroundBlurriness')
  assertFiniteNumberOption(options.environmentIntensity, 'options.environmentIntensity')
  assertEulerOption(options.backgroundRotation, 'options.backgroundRotation')
  assertEulerOption(options.environmentRotation, 'options.environmentRotation')
  if (options.localClippingEnabled != null && typeof options.localClippingEnabled !== 'boolean') {
    throw new TypeError('options.localClippingEnabled must be a boolean.')
  }
  validateSortControls(options)
  validatePostProcessingOptions(options.postProcessing)
  assertSupportedSampleCount(options.samples, 'options.samples')
  assertSupportedSampleCount(options.sampleCount, 'options.sampleCount')
  if (options.transmissionResolutionScale != null) {
    rendererStatePositiveFiniteNumber(options.transmissionResolutionScale, 'options.transmissionResolutionScale')
  }
  if (Object.prototype.hasOwnProperty.call(options, 'target') && options.target !== undefined) {
    assertRenderTargetLike(options.target, 'options.target')
  }
  if (options.target) validateUnsupportedRenderTargetOptions(options.target)
}

function assertRenderOptionsLike(value: unknown, label: string): asserts value is RenderOptions {
  if (value == null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be an options object.`)
  }
}

function assertRenderTargetLike(value: unknown, label: string): asserts value is RenderTargetLike {
  if (value == null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be a target-like object.`)
  }
}

function assertThreeTextureLike(value: unknown, label: string): asserts value is ThreeTextureLike {
  if (value == null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be a texture-like object.`)
  }
}

function hasThreeTextureMarker(value: unknown): value is ThreeTextureLike {
  return (
    value !== null
    && typeof value === 'object'
    && !Array.isArray(value)
    && (value as { isTexture?: unknown }).isTexture === true
  )
}

function isThreeTextureArgument(value: unknown): value is ThreeTextureLike {
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

function assertCanvasTargetLike(value: unknown, label: string): void {
  if (value == null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be a canvas-target-like object.`)
  }
}

function assertRendererInspectorLike(value: unknown, label: string): asserts value is RendererInspectorLike {
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

function assertTextureBindingSlot(value: unknown, label: string): asserts value is number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
    throw new TypeError(`${label} must be a non-negative integer.`)
  }
}

function unsupportedTextureBindingError(method: string): Error {
  return new Error(
    `${method}() is not supported by @headless-three/renderer because it does not expose browser WebGL texture units or direct texture binding. Use material, background, environment, or render-target texture inputs instead.`,
  )
}

function unsupportedInternalRenderDispatchError(method: string): Error {
  return new Error(
    `${method}() is not supported by @headless-three/renderer because CommonRenderer internal render pipeline dispatch depends on backend render contexts, render lists, nodes, pipelines, and bindings that are outside the scene-oriented API. Render normal Three.js scene graphs with Renderer.render() or renderToTarget().`,
  )
}

function unsupportedBackendOperationError(method: string, operation: string): Error {
  return new Error(
    `${method}() is not supported by @headless-three/renderer because ${operation} would require backend WebGL/WebGPU resource state that is outside the scene-oriented API. Render normal Three.js scene graphs with Renderer.render() or renderToTarget().`,
  )
}

function assertComputeNodesLike(value: unknown, label: string): void {
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

function assertComputeNodeLike(value: unknown, label: string): void {
  if (value == null || typeof value !== 'object' || Array.isArray(value) || (value as { isComputeNode?: unknown }).isComputeNode !== true) {
    throw new TypeError(`${label} must be a ComputeNode-like object.`)
  }
}

function assertComputeDispatchSize(value: unknown, label: string): void {
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

function assertPositiveInteger(value: unknown, label: string): void {
  if (typeof value !== 'number' || !Number.isFinite(value) || !Number.isInteger(value) || value <= 0) {
    throw new TypeError(`${label} must be a positive integer.`)
  }
}

function assertStorageBufferAttributeLike(value: unknown, label: string): void {
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

function assertTimestampQueryType(value: unknown, label: string): void {
  if (typeof value !== 'string') {
    throw new TypeError(`${label} must be "render" or "compute".`)
  }
  if (!SupportedTimestampQueryTypes.has(value)) {
    throw new TypeError(`${label} must be "render" or "compute"; received "${value}".`)
  }
}

function unsupportedComputeError(method: string): Error {
  return new Error(
    `${method}() is not supported by @headless-three/renderer because it does not expose WebGPU compute pipelines, storage buffers, or GPU dispatch. Render normal Three.js scene graphs with Renderer.render() or renderToTarget().`,
  )
}

function assertExternalWebGlObjectLike(value: unknown, label: string): void {
  if (value == null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be an external WebGL object-like handle.`)
  }
}

function assertOptionalExternalWebGlObjectLike(value: unknown, label: string): void {
  if (value == null) return
  assertExternalWebGlObjectLike(value, label)
}

interface RawTextureCopyImage {
  data: { length: number; [index: number]: number }
  width: number
  height: number
  channels: number
}

interface TextureCopyRegion {
  x: number
  y: number
  width: number
  height: number
}

interface TextureCopyPosition {
  x: number
  y: number
}

function rawTextureCopyImage(
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

function textureCopyReadableImageError(label: string, allowCanvasRead: boolean): string {
  if (allowCanvasRead) {
    return `${label} must provide a readable image object with raw data, width, and height, or canvas-like pixel access, including OffscreenCanvas-backed image reads.`
  }
  return `${label} must provide a readable raw image object with data, width, and height.`
}

function isMutableTextureCopyData(value: unknown): value is { length: number; [index: number]: number } {
  return (
    (Array.isArray(value) || ArrayBuffer.isView(value)) &&
    typeof (value as { length?: unknown }).length === 'number'
  )
}

function assertTextureCopyLevel(value: unknown, label: string): void {
  const level = value == null ? 0 : value
  if (!Number.isInteger(level) || (level as number) < 0) {
    throw new TypeError(`${label} must be a non-negative integer.`)
  }
}

function textureCopySourceRegion(value: unknown, sourceWidth: number, sourceHeight: number, label: string): TextureCopyRegion {
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

function textureCopyFramebufferSourceRegion(
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

function textureCopyDestinationPosition(value: unknown, label: string): TextureCopyPosition {
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

function textureCopyInteger(value: unknown, label: string): number {
  if (!Number.isInteger(value)) {
    throw new TypeError(`${label} must be an integer.`)
  }
  return value as number
}

function textureCopyFlooredInteger(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a finite number.`)
  }
  return Math.floor(value)
}

function textureCopyPositiveInteger(value: unknown, label: string): number {
  const integer = textureCopyInteger(value, label)
  if (integer <= 0) {
    throw new RangeError(`${label} must be a positive integer.`)
  }
  return integer
}

function textureCopyPositiveFlooredInteger(value: unknown, label: string): number {
  const integer = textureCopyFlooredInteger(value, label)
  if (integer <= 0) {
    throw new RangeError(`${label} must be a positive integer.`)
  }
  return integer
}

function assertEulerOption(value: unknown, label: string): void {
  if (value == null) return
  if (typeof value !== 'object') {
    throw new TypeError(`${label} must be a rotation object or array.`)
  }
  eulerComponents(value as ThreeEulerLike | ArrayLike<number>, label)
}

function validateSortControls(options: RenderOptions): void {
  if (options.sortObjects != null && typeof options.sortObjects !== 'boolean') {
    throw new TypeError(`options.sortObjects must be a boolean; received ${String(options.sortObjects)}.`)
  }
  if (options.opaque != null && typeof options.opaque !== 'boolean') {
    throw new TypeError(`options.opaque must be a boolean; received ${String(options.opaque)}.`)
  }
  if (options.transparent != null && typeof options.transparent !== 'boolean') {
    throw new TypeError(`options.transparent must be a boolean; received ${String(options.transparent)}.`)
  }
  if (options.opaqueSort != null && typeof options.opaqueSort !== 'function') {
    throw new TypeError('options.opaqueSort must be a function or null.')
  }
  if (options.transparentSort != null && typeof options.transparentSort !== 'function') {
    throw new TypeError('options.transparentSort must be a function or null.')
  }
}

function assertSortFunctionOrNull(value: unknown, label: string): asserts value is RenderSortFunction | null {
  if (value != null && typeof value !== 'function') {
    throw new TypeError(`${label} expects a function or null.`)
  }
}

function validateUnsupportedRenderTargetOptions(target: RenderTargetLike): void {
  if (target.scissorTest != null && typeof target.scissorTest !== 'boolean') {
    throw new TypeError('target.scissorTest must be a boolean.')
  }
  if (target.image != null) assertRenderTargetImageLike(target.image, 'target.image')
  assertRenderTargetTextureSlot(target.texture, 'target.texture')
  assertRenderTargetTexturesSlot(target.textures, 'target.textures')
  if (target.depthTexture != null) assertRenderTargetTextureLike(target.depthTexture, 'target.depthTexture')
  assertSupportedSampleCount(target.samples, 'target.samples')
  assertSupportedSampleCount(target.sampleCount, 'target.sampleCount')
  const colorTextures = renderTargetColorTextures(target)
  assertAuxiliaryRenderTargetAttachments(colorTextures)
  for (let i = 0; i < colorTextures.length; i += 1) {
    const colorTexture = colorTextures[i]
    const label = targetColorTextureLabel(i)
    assertSupportedRenderTargetTextureDimensionality(colorTexture, label)
    assertSupportedRenderTargetTextureClass(colorTexture, label)
    assertSupportedRenderTargetColorTexture(colorTexture, label)
  }
  assertSupportedRenderTargetTextureDimensionality(target.depthTexture, 'target.depthTexture')
  assertSupportedRenderTargetTextureClass(target.depthTexture, 'target.depthTexture')
  assertSupportedDepthTextureType(target.depthTexture)
  assertSupportedDepthTextureFormat(target.depthTexture)
}

function assertAuxiliaryRenderTargetAttachments(colorTextures: RenderTargetTextureLike[]): void {
  if (colorTextures.length <= 1) {
    for (let i = 0; i < colorTextures.length; i += 1) {
      renderTargetTextureRenderMode(colorTextures[i], targetColorTextureLabel(i))
    }
    return
  }

  for (let i = 0; i < colorTextures.length; i += 1) {
    const mode = renderTargetTextureRenderMode(colorTextures[i], targetColorTextureLabel(i))
    if (i > 0 && mode == null) {
      throw new Error(
        `${targetColorTextureLabel(i)} is a secondary color attachment and must declare userData.headlessThreeRenderer.renderMode as "color", "mask", "object-id", "normal", or "depth". Arbitrary native MRT shader outputs are not supported yet.`,
      )
    }
  }
}

function renderTargetTextureRenderMode(texture: RenderTargetTextureLike, label: string): RenderMode | undefined {
  const hints = renderTargetTextureRendererHints(texture, label)
  if (!hints || hints.value.renderMode == null) return undefined
  return checkedRenderMode(hints.value.renderMode, `${hints.label}.renderMode`)
}

function renderTargetTextureRendererHints(
  texture: RenderTargetTextureLike,
  label: string,
): { value: Record<string, unknown>; label: string } | undefined {
  const userData = texture.userData
  if (userData == null) return undefined
  assertPlainObject(userData, `${label}.userData`)

  const modernHints = userData.headlessThreeRenderer
  if (modernHints != null) {
    assertPlainObject(modernHints, `${label}.userData.headlessThreeRenderer`)
    return { value: modernHints, label: `${label}.userData.headlessThreeRenderer` }
  }

  const legacyHints = userData.headlessRenderer
  if (legacyHints != null) {
    assertPlainObject(legacyHints, `${label}.userData.headlessRenderer`)
    return { value: legacyHints, label: `${label}.userData.headlessRenderer` }
  }

  return undefined
}

function assertPlainObject(value: unknown, label: string): asserts value is Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be an object.`)
  }
}

function targetColorTextureLabel(index: number): string {
  return index === 0 ? 'target color texture' : `target color texture[${index}]`
}

function assertSupportedSampleCount(value: unknown, label: string): void {
  if (value == null) return
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || Math.floor(value) !== value) {
    throw new Error(
      `${label} must be a non-negative integer sample count; received ${String(value)}.`,
    )
  }
  if (value > 1 && value !== 4) {
    throw new Error(
      `MSAA sample count ${value} is not supported by @headless-three/renderer yet (${label}=${value}). Use 4 for 4x MSAA or the default single-sample render path.`,
    )
  }
}

function assertSupportedOutputFormat(value: unknown, label: string): void {
  if (value == null) return
  if (value === 'png' || value === 'rgba') return
  throw new Error(
    `${label} ${String(value)} is not supported by @headless-three/renderer. Use "png" or "rgba".`,
  )
}

function assertSupportedOutputColorSpace(value: unknown): void {
  if (value == null) return
  checkedOutputColorSpace(value, 'options.outputColorSpace')
}

function checkedOutputColorSpace(value: unknown, label: string): RenderOutputColorSpace {
  if (
    value === 'srgb' ||
    value === 'srgb-linear' ||
    value === 'linear-srgb' ||
    value === 'linearsrgb' ||
    value === 'linear'
  ) return value
  throw new Error(
    `${label} ${String(value)} is not supported by @headless-three/renderer. Use THREE.SRGBColorSpace or THREE.LinearSRGBColorSpace.`,
  )
}

function assertRenderTargetTextureSlot(value: unknown, label: string): void {
  if (value == null) return
  if (Array.isArray(value)) {
    if (value.length === 0) {
      throw new TypeError(`${label} must contain one texture-like object when provided as an array.`)
    }
    for (let i = 0; i < value.length; i += 1) {
      assertRenderTargetTextureLike(value[i], `${label}[${i}]`)
    }
    return
  }
  assertRenderTargetTextureLike(value, label)
}

function assertRenderTargetTexturesSlot(value: unknown, label: string): void {
  if (value == null) return
  if (!Array.isArray(value)) {
    throw new TypeError(`${label} must be an array of texture-like objects.`)
  }
  if (value.length === 0) {
    throw new TypeError(`${label} must contain one texture-like object when provided.`)
  }
  for (let i = 0; i < value.length; i += 1) {
    assertRenderTargetTextureLike(value[i], `${label}[${i}]`)
  }
}

function assertRenderTargetTextureLike(value: unknown, label: string): asserts value is RenderTargetTextureLike {
  if (value == null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be a texture-like object.`)
  }
  const texture = value as RenderTargetTextureLike
  assertRenderTargetImageSlot(texture.image, `${label}.image`)
  assertRenderTargetMipmaps(texture.mipmaps, `${label}.mipmaps`)
  assertRenderTargetSource(texture.source, `${label}.source`)
}

function assertRenderTargetSource(value: unknown, label: string): void {
  if (value == null) return
  if (typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be a source-like object.`)
  }
  assertRenderTargetImageSlot((value as { data?: unknown }).data, `${label}.data`)
}

function assertRenderTargetMipmaps(value: unknown, label: string): void {
  if (value == null) return
  if (!Array.isArray(value)) {
    throw new TypeError(`${label} must be an array of image-like objects.`)
  }
  for (let index = 0; index < value.length; index += 1) {
    assertRenderTargetImageLike(value[index], `${label}[${index}]`)
  }
}

function assertRenderTargetImageSlot(value: unknown, label: string): void {
  if (value == null) return
  if (Array.isArray(value)) {
    value.forEach((image, index) => {
      assertRenderTargetImageLike(image, `${label}[${index}]`)
    })
    return
  }
  assertRenderTargetImageLike(value, label)
}

function assertRenderTargetImageLike(value: unknown, label: string): asserts value is RenderTargetImageLike {
  if (value == null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be an image-like object.`)
  }
}

function assertWebGlExtensionName(value: unknown, label: string): asserts value is string {
  if (typeof value !== 'string' || value.length === 0) {
    throw new TypeError(`${label} must be a non-empty string.`)
  }
}

function assertRendererProbeName(value: unknown, label: string): asserts value is string {
  if (typeof value !== 'string' || value.length === 0) {
    throw new TypeError(`${label} must be a non-empty string.`)
  }
}

function assertEventListener(type: unknown, listener: unknown, label: string): void {
  if (typeof type !== 'string' || type.length === 0) {
    throw new TypeError(`${label} type must be a non-empty string.`)
  }
  if (typeof listener !== 'function') {
    throw new TypeError(`${label} listener must be a function.`)
  }
}

function assertDomElementAttributeName(value: unknown, label: string): asserts value is string {
  if (typeof value !== 'string' || value.length === 0) {
    throw new TypeError(`${label} must be a non-empty string.`)
  }
}

function domElementStylePropertyKey(value: unknown, label: string): string {
  if (typeof value !== 'string' || value.length === 0) {
    throw new TypeError(`${label} must be a non-empty string.`)
  }
  if (value.startsWith('--')) return value
  return value.replace(/-([a-z])/g, (_match, letter: string) => letter.toUpperCase())
}

function domElementStyleWritablePropertyKey(value: unknown, label: string): string {
  const key = domElementStylePropertyKey(value, label)
  if (key === 'setProperty' || key === 'getPropertyValue' || key === 'removeProperty') {
    throw new TypeError(`${label} must not name a reserved style method.`)
  }
  return key
}

function assertXrInputIndex(value: unknown, label: string): asserts value is number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
    throw new TypeError(`${label} must be a non-negative integer.`)
  }
}

function assertWeakMapKey(value: unknown, label: string): asserts value is object {
  if (value == null || (typeof value !== 'object' && typeof value !== 'function')) {
    throw new TypeError(`${label} must be an object.`)
  }
}

function assertPropertyKey(value: unknown, label: string): asserts value is string {
  if (typeof value !== 'string' || value.length === 0) {
    throw new TypeError(`${label} must be a non-empty string.`)
  }
}

function assertNonEmptyString(value: unknown, label: string): asserts value is string {
  if (typeof value !== 'string' || value.length === 0) {
    throw new TypeError(`${label} must be a non-empty string.`)
  }
}

function assertFunction(value: unknown, label: string): asserts value is (...args: unknown[]) => unknown {
  if (typeof value !== 'function') {
    throw new TypeError(`${label} must be a function.`)
  }
}

function assertConstructorFunction(value: unknown, label: string): asserts value is new (...args: unknown[]) => Record<string, unknown> {
  if (typeof value !== 'function') {
    throw new TypeError(`${label} must be a constructor function.`)
  }
}

function assertFiniteInteger(value: unknown, label: string): asserts value is number {
  if (typeof value !== 'number' || !Number.isFinite(value) || Math.floor(value) !== value) {
    throw new TypeError(`${label} must be an integer.`)
  }
}

function rendererRenderListId(object: unknown): unknown {
  return object && typeof object === 'object'
    ? (object as Record<string, unknown>).id
    : undefined
}

function rendererRenderListRenderOrder(object: unknown): unknown {
  return object && typeof object === 'object'
    ? (object as Record<string, unknown>).renderOrder
    : undefined
}

function rendererRenderListMaterialVariant(object: unknown): number {
  if (!object || typeof object !== 'object') return 0
  const record = object as Record<string, unknown>
  return (record.isInstancedMesh === true ? 2 : 0) + (record.isSkinnedMesh === true ? 1 : 0)
}

function validatePostProcessingOptions(value: unknown): void {
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

function assertFinitePostProcessingNumber(value: unknown, label: string): void {
  assertFiniteNumberOption(value, label)
}

function assertNormalizedPostProcessingNumber(value: unknown, label: string): void {
  assertFiniteNumberOption(value, label)
  if (typeof value === 'number' && (value < 0 || value > 1)) {
    throw new TypeError(`${label} must be between 0 and 1.`)
  }
}

function assertFinitePostProcessingBlend(value: unknown, label: string): void {
  if (value == null || typeof value === 'boolean') return
  if (typeof value === 'number' && Number.isFinite(value)) {
    if (value < 0 || value > 1) {
      throw new TypeError(`${label} must be between 0 and 1.`)
    }
    return
  }
  throw new TypeError(`${label} must be a finite number or boolean.`)
}

function assertFiniteNumberOption(value: unknown, label: string): void {
  if (value == null) return
  if (typeof value === 'number' && Number.isFinite(value)) return
  throw new TypeError(`${label} must be a finite number.`)
}

function assertNonNegativeNumberOption(value: unknown, label: string): void {
  assertFiniteNumberOption(value, label)
  if (typeof value === 'number' && value < 0) {
    throw new TypeError(`${label} must be non-negative.`)
  }
}

function assertNormalizedNumberOption(value: unknown, label: string): void {
  assertFiniteNumberOption(value, label)
  if (typeof value === 'number' && (value < 0 || value > 1)) {
    throw new TypeError(`${label} must be between 0 and 1.`)
  }
}

function assertSupportedRenderTargetColorTexture(texture: RenderTargetTextureLike | undefined, label = 'target color texture'): void {
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

function isReadableRenderTargetColorFormat(format: number): boolean {
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

function isReadableRenderTargetColorType(type: number): boolean {
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

function assertSupportedRenderTargetTextureDimensionality(texture: RenderTargetTextureLike | undefined, label: string): void {
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

function assertSupportedRenderTargetTextureClass(texture: RenderTargetTextureLike | undefined, label: string): void {
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

function assertNonCubeCameraRenderTargetTextures(target: RenderTargetLike): void {
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

function assertSupportedDepthTextureType(depthTexture: RenderTargetTextureLike | undefined): void {
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

function assertSupportedDepthTextureFormat(depthTexture: RenderTargetTextureLike | undefined): void {
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

function resolveSampleCount(options: RenderOptions): number {
  const requested = options.target?.sampleCount
    ?? options.target?.samples
    ?? options.sampleCount
    ?? options.samples
    ?? 1
  return requested > 1 ? requested : 1
}

function writeRenderTarget(
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

function compositeActiveTargetColorBuffer(
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

function activeTargetCopyShaderAdditiveBlend(
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

function activeTargetCopyShaderMaterialInfo(material: ThreeMaterialLike | undefined): { opacity: unknown } | null {
  if (!material || !activeTargetShaderMaterialKind(material)) return null
  if (!activeTargetCopyShaderFragment(material.fragmentShader)) return null
  const uniforms = material.uniforms
  if (!uniforms || typeof uniforms !== 'object' || Array.isArray(uniforms)) return null
  return {
    opacity: activeTargetUniformValue((uniforms as Record<string, unknown>).opacity) ?? 1,
  }
}

function activeTargetShaderMaterialKind(material: ThreeMaterialLike): boolean {
  return material.isShaderMaterial === true || material.type === 'ShaderMaterial'
}

function activeTargetUniformValue(uniform: unknown): unknown {
  if (!uniform || typeof uniform !== 'object' || Array.isArray(uniform)) return undefined
  return (uniform as { value?: unknown }).value
}

function activeTargetCopyShaderFragment(fragmentShader: unknown): boolean {
  if (typeof fragmentShader !== 'string') return false
  const compact = fragmentShader.replace(/\s+/g, '')
  return compact.includes('uniformfloatopacity;') &&
    compact.includes('uniformsampler2DtDiffuse;') &&
    compact.includes('texture2D(tDiffuse,vUv)') &&
    compact.includes('gl_FragColor=opacity*texel;')
}

function additiveCompositeColorBuffer(
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

function activeTargetRenderRect(options: RenderOptions, width: number, height: number): PixelRect {
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

function intersectPixelRects(a: PixelRect, b: PixelRect): PixelRect | null {
  const x = Math.max(a.x, b.x)
  const y = Math.max(a.y, b.y)
  const right = Math.min(a.x + a.width, b.x + b.width)
  const bottom = Math.min(a.y + a.height, b.y + b.height)
  if (right <= x || bottom <= y) return null
  return { x, y, width: right - x, height: bottom - y }
}

function clearRenderTargetColor(
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

function clearRenderTargetDepth(
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

function clearRenderTargetStencil(
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

function renderTargetClearScissor(
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

function renderTargetClearSize(
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

function renderTargetClearImage(target: RenderTargetLike): RenderTargetImageLike | undefined {
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

function renderTargetDepthImage(texture: RenderTargetTextureLike): RenderTargetImageLike | undefined {
  if (Array.isArray(texture.image)) return texture.image[0]
  if (texture.image?.data) return texture.image
  const sourceData = texture.source?.data
  if (Array.isArray(sourceData)) return sourceData[0]
  return sourceData
}

function renderTargetClearDimension(value: unknown): number | null {
  return typeof value === 'number' && Number.isInteger(value) && value > 0 ? value : null
}

function renderTargetExistingColorBuffer(
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

function clearColorBuffer(
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

type ScalarDepthArray = Uint8Array | Uint16Array | Uint32Array | Float32Array
type ScalarDepthArrayConstructor<T extends ScalarDepthArray> = {
  new(length: number): T
  new(array: ArrayLike<number>): T
}

function clearDepthTextureData(
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

function clearPackedDepthData(
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

function clearPackedDepthStencilData(
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

function clearScalarDepthData<T extends ScalarDepthArray>(
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

function clearDepthRgbaBuffer(
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

function writeObjectIdMetadata(target: RenderTargetLike, objectIdEntries?: RenderObjectIdEntry[]): void {
  if (objectIdEntries) {
    target.objectIdEntries = objectIdEntries
    target.objectIdMap = Object.fromEntries(objectIdEntries.map((entry) => [String(entry.encodedId), entry]))
  } else {
    delete target.objectIdEntries
    delete target.objectIdMap
  }
}

function renderTargetColorTexture(target: RenderTargetLike): RenderTargetTextureLike | undefined {
  return renderTargetColorTextures(target)[0]
}

function renderTargetColorTextures(target: RenderTargetLike): RenderTargetTextureLike[] {
  if (Array.isArray(target.texture)) return target.texture
  if (target.textures) return target.textures
  return target.texture ? [target.texture] : []
}

function renderTargetReadbackSource(
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

function renderTargetReadbackImage(
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

function renderTargetTextureFaceImage(
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

function renderTargetReadbackChannelCount(
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

function copyRenderTargetReadbackPixels(
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

function readbackRect(x: unknown, y: unknown, width: unknown, height: unknown, label: string): PixelRect {
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

function assertRenderTargetReadbackBuffer(
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

function createRenderTargetReadbackBuffer(
  source: NonNullable<RenderTargetImageLike['data']>,
  length: number,
): NonNullable<RenderTargetImageLike['data']> {
  if (Buffer.isBuffer(source)) return Buffer.alloc(length)
  const TypedArrayConstructor = source.constructor as new (length: number) => Exclude<NonNullable<RenderTargetImageLike['data']>, Buffer>
  return new TypedArrayConstructor(length)
}

function writeRenderTargetTexture(
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

function colorTextureData(texture: RenderTargetTextureLike, rgba: Buffer): NonNullable<RenderTargetImageLike['data']> {
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

function normalizedByteToSignedInteger(value: number, max: number): number {
  return Math.round((value / 255) * max)
}

function normalizedByteToUnsignedInteger(value: number, max: number): number {
  return Math.round((value / 255) * max)
}

function packedUnsignedShort4444ColorTextureData(values: Uint8Array, channels: 1 | 2 | 3 | 4): Uint16Array {
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

function packedUnsignedShort5551ColorTextureData(values: Uint8Array, channels: 1 | 2 | 3 | 4): Uint16Array {
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

function packedUnsignedInt5999ColorTextureData(values: Uint8Array, channels: 1 | 2 | 3 | 4): Uint32Array {
  const out = new Uint32Array(values.length / channels)
  for (let src = 0, pixel = 0; src < values.length; src += channels, pixel += 1) {
    const r = values[src] / 255
    const g = channels > 1 ? values[src + 1] / 255 : 0
    const b = channels > 2 ? values[src + 2] / 255 : 0
    out[pixel] = packRgb9E5(r, g, b)
  }
  return out
}

function packedUnsignedInt101111ColorTextureData(values: Uint8Array, channels: 1 | 2 | 3 | 4): Uint32Array {
  const out = new Uint32Array(values.length / channels)
  for (let src = 0, pixel = 0; src < values.length; src += channels, pixel += 1) {
    const r = values[src] / 255
    const g = channels > 1 ? values[src + 1] / 255 : 0
    const b = channels > 2 ? values[src + 2] / 255 : 0
    out[pixel] = packR11G11B10F(r, g, b)
  }
  return out
}

function packRgb9E5(r: number, g: number, b: number): number {
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

function packR11G11B10F(r: number, g: number, b: number): number {
  return (
    (packUnsignedFloat(b, 5) << 22) |
    (packUnsignedFloat(g, 6) << 11) |
    packUnsignedFloat(r, 6)
  ) >>> 0
}

function packUnsignedFloat(value: number, mantissaBits: 5 | 6): number {
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

function colorTextureChannelCount(format: number | undefined): 1 | 2 | 3 | 4 {
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

function colorTextureBytes(rgba: Buffer, format: number | undefined, channels: 1 | 2 | 3 | 4): Uint8Array {
  if (format === AlphaFormat) return alphaColorTextureBytes(rgba)
  if (format === LuminanceAlphaFormat) return luminanceAlphaColorTextureBytes(rgba)
  return channels === 4 ? rgba : narrowedColorTextureBytes(rgba, channels)
}

function alphaColorTextureBytes(rgba: Buffer): Uint8Array {
  const pixels = rgba.length / 4
  const out = new Uint8Array(pixels)
  for (let i = 0, p = 0; i < rgba.length; i += 4, p += 1) {
    out[p] = rgba[i + 3]
  }
  return out
}

function narrowedColorTextureBytes(rgba: Buffer, channels: 1 | 2 | 3): Uint8Array {
  const pixels = rgba.length / 4
  const out = new Uint8Array(pixels * channels)
  for (let i = 0, p = 0; i < rgba.length; i += 4, p += channels) {
    out[p] = rgba[i]
    if (channels > 1) out[p + 1] = rgba[i + 1]
    if (channels > 2) out[p + 2] = rgba[i + 2]
  }
  return out
}

function luminanceAlphaColorTextureBytes(rgba: Buffer): Uint8Array {
  const pixels = rgba.length / 4
  const out = new Uint8Array(pixels * 2)
  for (let i = 0, p = 0; i < rgba.length; i += 4, p += 2) {
    out[p] = rgba[i]
    out[p + 1] = rgba[i + 3]
  }
  return out
}

function depthTextureData(texture: RenderTargetTextureLike, rgbaDepth: Buffer): NonNullable<RenderTargetImageLike['data']> {
  if (texture.type === UnsignedByteType) {
    const depth = new Uint8Array(rgbaDepth.length / 4)
    for (let i = 0, p = 0; i < rgbaDepth.length; i += 4, p += 1) {
      depth[p] = rgbaDepth[i]
    }
    return depth
  }
  if (texture.type === UnsignedShortType) {
    const depth = new Uint16Array(rgbaDepth.length / 4)
    for (let i = 0, p = 0; i < rgbaDepth.length; i += 4, p += 1) {
      depth[p] = Math.round((rgbaDepth[i] / 255) * 0xffff)
    }
    return depth
  }
  if (texture.type === UnsignedIntType) {
    const depth = new Uint32Array(rgbaDepth.length / 4)
    for (let i = 0, p = 0; i < rgbaDepth.length; i += 4, p += 1) {
      depth[p] = Math.round((rgbaDepth[i] / 255) * 0xffffffff)
    }
    return depth
  }
  if (texture.type === UnsignedInt248Type) {
    const depth = new Uint32Array(rgbaDepth.length / 4)
    for (let i = 0, p = 0; i < rgbaDepth.length; i += 4, p += 1) {
      depth[p] = Math.round((rgbaDepth[i] / 255) * 0xffffff) * 0x100
    }
    return depth
  }
  if (texture.type === FloatType) {
    const depth = new Float32Array(rgbaDepth.length / 4)
    for (let i = 0, p = 0; i < rgbaDepth.length; i += 4, p += 1) {
      depth[p] = rgbaDepth[i] / 255
    }
    return depth
  }
  if (texture.type === HalfFloatType) {
    const depth = new Uint16Array(rgbaDepth.length / 4)
    for (let i = 0, p = 0; i < rgbaDepth.length; i += 4, p += 1) {
      depth[p] = normalizedFloatToHalf(rgbaDepth[i] / 255)
    }
    return depth
  }
  return rgbaDepth
}

function normalizedFloatToHalf(value: number): number {
  const clamped = Math.min(1, Math.max(0, value))
  if (clamped === 0) return 0
  if (clamped === 1) return 0x3c00

  const exponent = Math.floor(Math.log2(clamped))
  if (exponent < -14) {
    return Math.round(clamped * 0x1000000)
  }

  let mantissa = Math.round((clamped / (2 ** exponent) - 1) * 0x400)
  let biasedExponent = exponent + 15
  if (mantissa === 0x400) {
    mantissa = 0
    biasedExponent += 1
  }
  return (biasedExponent << 10) | mantissa
}

function cloneTargetData(data: NonNullable<RenderTargetImageLike['data']>): NonNullable<RenderTargetImageLike['data']> {
  if (Buffer.isBuffer(data)) return Buffer.from(data)
  if (data instanceof Float32Array) return new Float32Array(data)
  if (data instanceof Int32Array) return new Int32Array(data)
  if (data instanceof Uint32Array) return new Uint32Array(data)
  if (data instanceof Int16Array) return new Int16Array(data)
  if (data instanceof Uint16Array) return new Uint16Array(data)
  if (data instanceof Int8Array) return new Int8Array(data)
  if (data instanceof Uint8ClampedArray) return new Uint8ClampedArray(data)
  return new Uint8Array(data)
}

function validateThreeSceneRoot(scene: unknown): asserts scene is ThreeSceneRootLike {
  const root = scene as any
  if (!root || (root.isScene !== true && root.isObject3D !== true)) {
    throw new TypeError('render(scene, camera) expects scene to be a THREE.Scene or THREE.Object3D root')
  }
}

function validateTopLevelRenderCamera(camera: unknown): asserts camera is ThreeRenderCameraLike {
  const cameraLike = camera as any
  if (!cameraLike || typeof cameraLike !== 'object' || Array.isArray(cameraLike)) {
    throw new TypeError('render(scene, camera) expects camera to be a THREE.Camera, THREE.ArrayCamera, or THREE.CubeCamera')
  }
  if (isCubeCamera(cameraLike)) return
  if (cameraLike.isCamera !== true) {
    throw new TypeError('render(scene, camera) expects camera to be a THREE.Camera, THREE.ArrayCamera, or THREE.CubeCamera')
  }
}

function isArrayCamera(camera: unknown): camera is ThreeCameraLike {
  const cameraLike = camera as any
  return cameraLike?.isArrayCamera === true || Array.isArray(cameraLike?.cameras)
}

function isCubeCamera(camera: unknown): camera is ThreeCubeCameraLike {
  const cameraLike = camera as any
  return cameraLike?.isCubeCamera === true || cameraLike?.type === 'CubeCamera'
}

function validateThreeCamera(camera: unknown, label = 'render(scene, camera)'): asserts camera is ThreeCameraLike {
  const defaultLabel = label === 'render(scene, camera)'
  const cameraLike = camera as any
  if (cameraLike?.isCubeCamera === true || cameraLike?.type === 'CubeCamera') {
    throw new Error(
      defaultLabel
        ? 'THREE.CubeCamera cannot be used where a regular THREE.Camera is required. Pass the CubeCamera as the top-level camera with a cube render target.'
        : `${label} cannot be a THREE.CubeCamera. Pass the CubeCamera as the top-level camera with a cube render target.`,
    )
  }
  if (!camera || typeof camera !== 'object' || Array.isArray(camera) || cameraLike.isCamera !== true) {
    throw new TypeError(defaultLabel ? 'render(scene, camera) expects camera to be a THREE.Camera' : `${label} must be a THREE.Camera.`)
  }
  if (cameraLike.isArrayCamera === true || Array.isArray(cameraLike.cameras)) {
    throw new Error(
      defaultLabel
        ? 'THREE.ArrayCamera cannot be used where a regular THREE.Camera is required. Pass the ArrayCamera as the top-level camera.'
        : `${label} cannot be a THREE.ArrayCamera. Pass the ArrayCamera as the top-level camera.`,
    )
  }
  if (!cameraLike.projectionMatrix || !cameraLike.matrixWorldInverse) {
    throw new TypeError(defaultLabel ? 'THREE.Camera must have projectionMatrix and matrixWorldInverse' : `${label} must have projectionMatrix and matrixWorldInverse.`)
  }
  matrixElements(cameraLike.projectionMatrix, defaultLabel ? 'camera.projectionMatrix' : `${label}.projectionMatrix`)
  matrixElements(cameraLike.matrixWorldInverse, defaultLabel ? 'camera.matrixWorldInverse' : `${label}.matrixWorldInverse`)
}
