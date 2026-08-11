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
import { rendererStateBoolean, rendererStatePositiveFiniteNumber } from './index.part-014'
import { assertEventListener, assertXrInputIndex } from './index.part-017'
import { validateThreeSceneRoot, validateTopLevelRenderCamera } from './index.part-021'
export class RendererXrState {
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

export class RendererDebugState {
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

export class RendererInspectorState implements RendererInspectorLike {
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
