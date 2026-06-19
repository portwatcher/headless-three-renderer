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
  RenderSortFunction,
  RenderAnimationLoopCallback,
} from './types'

// eslint-disable-next-line @typescript-eslint/no-var-requires
const native = require('../native.js')

import { resolveSize, cameraViewProjection, cameraViewMatrix, cameraWorldPosition } from './camera'
import { DEFAULT_BACKGROUND_COLOR, cssColorStringToArray, resolveBackground, validatedColorLikeToArray } from './color'
import { flattenScene, type ShadowMaterialMode } from './scene'
import { extractLights, extractAmbientLight, extractAmbientIntensity, extractLightProbe } from './lights'
import { extractBackgroundTexture, isCompressedTextureFormat, resolveEnvironmentMap, resolveSceneOverrideMaterial } from './materials'
import { extractClippingPlanes } from './clipping'
import { validateObjectChildrenTree } from './objects'
import { clamp01, matrixElements } from './math'

const WEBGL_COORDINATE_SYSTEM = 2000
const BasicShadowMap = 0
const PCFShadowMap = 1
const PCFSoftShadowMap = 2
const VSMShadowMap = 3
const SupportedRendererShadowMapTypes = new Set([BasicShadowMap, PCFShadowMap, PCFSoftShadowMap, VSMShadowMap])
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
  RenderSortFunction,
  RenderSortItem,
  PostProcessingOptions,
} from './types'

class RendererShadowMapState {
  private enabledValue = true
  private autoUpdateValue = true
  private needsUpdateValue = false
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

  get type(): number {
    return this.typeValue
  }

  set type(value: number) {
    this.typeValue = rendererStateShadowMapType(value)
  }
}

class RendererInfoState {
  private autoResetValue = true

  readonly memory = {
    geometries: 0,
    textures: 0,
  }

  readonly render = {
    calls: 0,
    triangles: 0,
    points: 0,
    lines: 0,
    frame: 0,
  }

  programs: unknown[] | null = null

  get autoReset(): boolean {
    return this.autoResetValue
  }

  set autoReset(value: boolean) {
    this.autoResetValue = rendererStateBoolean(value, 'Renderer.info.autoReset')
  }

  reset(): void {
    this.render.calls = 0
    this.render.triangles = 0
    this.render.points = 0
    this.render.lines = 0
    this.render.frame = 0
  }
}

class RendererXrState {
  private enabledValue = false

  get enabled(): boolean {
    return this.enabledValue
  }

  set enabled(value: boolean) {
    this.enabledValue = rendererStateBoolean(value, 'Renderer.xr.enabled')
  }
}

export class Renderer {
  private native: InstanceType<typeof native.NativeRenderer>
  private opaqueSort: RenderSortFunction | null = null
  private sortObjectsValue = true
  private transparentSort: RenderSortFunction | null = null
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
  private animationLoop: RenderAnimationLoopCallback | null = null

  readonly coordinateSystem = WEBGL_COORDINATE_SYSTEM
  readonly info = new RendererInfoState()
  readonly reversedDepthBuffer = false
  readonly shadowMap = new RendererShadowMapState()
  readonly xr = new RendererXrState()

  constructor() {
    this.native = new native.NativeRenderer()
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

  get outputColorSpace(): RenderOutputColorSpace {
    return this.outputColorSpaceValue
  }

  set outputColorSpace(value: RenderOutputColorSpace) {
    this.outputColorSpaceValue = checkedOutputColorSpace(value, 'Renderer.outputColorSpace')
  }

  get toneMapping(): number {
    return this.toneMappingValue
  }

  set toneMapping(value: number) {
    this.toneMappingValue = rendererStateToneMapping(value)
  }

  get toneMappingExposure(): number {
    return this.toneMappingExposureValue
  }

  set toneMappingExposure(value: number) {
    this.toneMappingExposureValue = finiteNonNegativeNumber(value, 'Renderer.toneMappingExposure')
  }

  get localClippingEnabled(): boolean {
    return this.localClippingEnabledValue
  }

  set localClippingEnabled(value: boolean) {
    this.localClippingEnabledValue = rendererStateBoolean(value, 'Renderer.localClippingEnabled')
  }

  setOpaqueSort(method: RenderSortFunction | null): void {
    assertSortFunctionOrNull(method, 'Renderer.setOpaqueSort')
    this.opaqueSort = method
  }

  setTransparentSort(method: RenderSortFunction | null): void {
    assertSortFunctionOrNull(method, 'Renderer.setTransparentSort')
    this.transparentSort = method
  }

  setAnimationLoop(callback: RenderAnimationLoopCallback | null): void {
    if (callback !== null && typeof callback !== 'function') {
      throw new TypeError('Renderer.setAnimationLoop callback must be a function or null.')
    }
    this.animationLoop = callback
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

  setSize(width: number, height: number, _updateStyle = true): void {
    this.currentSize = rendererStateSize(width, height, 'Renderer.setSize')
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
  setViewport(x: number, y: number, width: number, height: number): void
  setViewport(rectOrX: RenderPixelRectLike | null | number, y?: number, width?: number, height?: number): void {
    this.currentViewport = rendererStatePixelRect(rectOrX, y, width, height, 'Renderer.setViewport')
  }

  getViewport(): RenderPixelRectLike | null
  getViewport<T extends RenderPixelRectLike>(target: T): T | null
  getViewport(target?: RenderPixelRectLike): RenderPixelRectLike | null {
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
    // Each render call owns its native pass, so there is no persistent framebuffer to clear.
  }

  clearColor(): void {
    // Color buffers are created and cleared inside each native render pass.
  }

  clearDepth(): void {
    // Depth is owned by each native render pass, so there is no persistent buffer to clear.
  }

  clearStencil(): void {
    // Stencil state is scoped to each native render pass.
  }

  dispose(): void {
    // Native resources are owned by the renderer instance and released with normal object lifetime.
  }

  resetState(): void {
    // Native render state is rebuilt for each pass, so there is no persistent GL state to reset.
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
      writeRenderTarget(
        target,
        buffer,
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
    writeRenderTarget(
      target,
      buffer,
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
    const { nativeScene, nativeCamera, objectIdEntries } = toNativeInput(scene, camera, options)
    return { buffer: this.native.render(nativeScene, nativeCamera), nativeScene, nativeCamera, objectIdEntries }
  }

  private resolveRenderOptions(options: RenderOptions, fallbackTarget: RenderTargetLike | null | undefined = options.target): InternalRenderOptions {
    const sizeOptions = this.optionsWithRendererSizeFallback(options, fallbackTarget)
    return {
      ...sizeOptions,
      outputColorSpace: sizeOptions.outputColorSpace ?? this.outputColorSpace,
      localClippingEnabled: sizeOptions.localClippingEnabled ?? this.localClippingEnabled,
      sortObjects: sizeOptions.sortObjects ?? this.sortObjects,
      opaqueSort: sizeOptions.opaqueSort === undefined ? this.opaqueSort : sizeOptions.opaqueSort,
      transparentSort: sizeOptions.transparentSort === undefined ? this.transparentSort : sizeOptions.transparentSort,
      __headlessThreeRendererClearColor: cloneColor4(this.currentClearColor),
      __headlessThreeRendererViewport: clonePixelRect(this.currentViewport),
      __headlessThreeRendererScissor: clonePixelRect(this.currentScissor),
      __headlessThreeRendererScissorTest: this.currentScissorTest,
      __headlessThreeRendererShadowMapEnabled: this.shadowMap.enabled,
      __headlessThreeRendererShadowMapType: this.shadowMap.type,
      __headlessThreeRendererToneMapping: this.toneMapping,
      __headlessThreeRendererToneMappingExposure: this.toneMappingExposure,
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
    camera.updateMatrixWorld(true)
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
  const clippingPlanes = extractClippingPlanes(options.clippingPlanes, 'options.clippingPlanes')
  const rendererShadowMapEnabled = (options as InternalRenderOptions).__headlessThreeRendererShadowMapEnabled !== false
  const rendererShadowMapType = (options as InternalRenderOptions).__headlessThreeRendererShadowMapType ?? PCFShadowMap
  const rendererToneMapping = (options as InternalRenderOptions).__headlessThreeRendererToneMapping ?? ACESFilmicToneMapping
  const toneMappingExposure = (options as InternalRenderOptions).__headlessThreeRendererToneMappingExposure ?? 1
  const extractedLights: NativeSceneLight[] | undefined = colorMode ? extractLights(scene, camera) : []
  const lights = rendererShadowMapEnabled ? extractedLights : nativeLightsWithoutShadows(extractedLights)
  const shadowMaterialMode = colorMode ? shadowMaterialModeForLights(lights) : undefined
  const flattenedMeshes = flattenScene(
    scene,
    camera,
    size.height,
    clippingPlanes,
    options.localClippingEnabled !== false,
    shadowMaterialMode,
    environment.materialContext,
    {
      sortObjects: options.sortObjects,
      opaqueSort: options.opaqueSort,
      transparentSort: options.transparentSort,
    },
    overrideMaterial,
  )
  const objectIdEntries = renderMode === 'object-id' ? objectIdEntriesForMeshes(flattenedMeshes) : undefined
  const meshes = applyRendererToneMapping(applyRenderMode(flattenedMeshes, renderMode), rendererToneMapping)
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
    outputColorSpace: options.outputColorSpace,
    toneMapping: rendererToneMapping,
    toneMappingExposure,
    sampleCount: resolveSampleCount(options),
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
  if (mode === 'color' || mode === 'mask' || mode === 'object-id' || mode === 'normal') return mode
  throw new TypeError(
    `${label} must be "color", "mask", "object-id", or "normal"; received ${String(mode)}`,
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
  __headlessThreeRendererToneMapping?: number
  __headlessThreeRendererToneMappingExposure?: number
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
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${label} value must be a finite number.`)
  }
  if (value <= 0) {
    throw new TypeError(`${label} value must be greater than 0.`)
  }
  return value
}

function rendererStateBoolean(value: unknown, label: string): boolean {
  if (typeof value !== 'boolean') {
    throw new TypeError(`${label} must be a boolean.`)
  }
  return value
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

function rendererStateToneMapping(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`Renderer.toneMapping must be a Three.js tone mapping constant; received ${String(value)}.`)
  }
  if (!Number.isInteger(value) || !SupportedRendererToneMappings.has(value)) {
    throw new TypeError(
      `Renderer.toneMapping ${String(value)} is not supported by @headless-three/renderer yet. Use THREE.NoToneMapping, THREE.LinearToneMapping, THREE.ReinhardToneMapping, THREE.CineonToneMapping, THREE.ACESFilmicToneMapping, THREE.CustomToneMapping, THREE.AgXToneMapping, or THREE.NeutralToneMapping.`,
    )
  }
  return value
}

function assertOptionalBoolean(value: unknown, label: string): void {
  rendererStateBoolean(value, label)
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

  if (near != null && near <= 0) {
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
        `${targetColorTextureLabel(i)} is a secondary color attachment and must declare userData.headlessThreeRenderer.renderMode as "color", "mask", "object-id", or "normal". Arbitrary native MRT shader outputs are not supported yet.`,
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
  const format = texture.format
  if (isCompressedTextureFormat(format)) {
    throw new Error(
      `${label} format uses a compressed texture format, which is not supported by @headless-three/renderer render targets. Use a regular 2D target texture and compress output after readback if needed.`,
    )
  }
  if (
    format != null &&
    format !== AlphaFormat &&
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
      `${label} format ${String(format)} is not supported by @headless-three/renderer yet. Use AlphaFormat, RedFormat, RedIntegerFormat, RGFormat, RGIntegerFormat, RGBFormat, RGBIntegerFormat, RGBAFormat, RGBAIntegerFormat, or omit format for RGBA8 readback.`,
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
    case RedFormat:
    case RedIntegerFormat:
      return 1
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
