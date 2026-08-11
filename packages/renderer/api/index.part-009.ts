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
import { collectCompileMaterials } from './index.part-007'
import { Renderer } from './index.part-008'
import { assertActiveCubeFace, assertActiveMipmapLevel, isCubeRenderTarget } from './index.part-013'
import { cloneColor3, rendererStateBoolean, rendererStateClearAlpha, rendererStatePixelRatio, rendererStateSize } from './index.part-014'
import { clonePixelRect, clonePixelSize } from './index.part-015'
import { assertCanvasTargetLike, assertComputeDispatchSize, assertComputeNodesLike, assertExternalWebGlObjectLike, assertOptionalExternalWebGlObjectLike, assertRenderTargetLike, assertRendererInspectorLike, assertStorageBufferAttributeLike, assertTextureBindingSlot, assertTextureCopyLevel, assertThreeTextureLike, assertTimestampQueryType, hasThreeTextureMarker, isThreeTextureArgument, rawTextureCopyImage, textureCopyDestinationPosition, textureCopyFramebufferSourceRegion, textureCopySourceRegion, unsupportedComputeError, unsupportedTextureBindingError } from './index.part-016'
import { validateUnsupportedRenderTargetOptions } from './index.part-017'
import { renderTargetReadbackSource } from './index.part-020'
import { validateThreeSceneRoot, validateTopLevelRenderCamera } from './index.part-021'
export function renderer_onDeviceLost_0(this: any, value: (info?: unknown) => void) {
    if (typeof value !== 'function') {
      throw new TypeError('Renderer.onDeviceLost must be a function.')
    }
    this.onDeviceLostValue = value
  }

export function renderer__onDeviceLost_1(this: any, info?: unknown): void {
    this.defaultOnDeviceLost(info)
    if (this.onDeviceLostValue !== this.defaultOnDeviceLost) {
      this.onDeviceLostValue(info)
    }
  }

export function renderer_inspector_2(this: any, value: RendererInspectorLike) {
    assertRendererInspectorLike(value, 'Renderer.inspector')
    this.inspectorValue.setRenderer(null)
    this.inspectorValue = value
    this.inspectorValue.setRenderer(this)
  }

export function renderer_isOccluded_3(this: any, object: unknown): boolean {
    if (object === null || typeof object !== 'object' || Array.isArray(object)) {
      throw new TypeError('Renderer.isOccluded object must be an object-like value.')
    }
    return false
  }

export function renderer_sortObjects_4(this: any, value: boolean) {
    if (typeof value !== 'boolean') {
      throw new TypeError(`Renderer.sortObjects must be a boolean; received ${String(value)}.`)
    }
    this.sortObjectsValue = value
  }

export function renderer_highPrecision_5(this: any, value: boolean) {
    const enabled = rendererStateBoolean(value, 'Renderer.highPrecision')
    if (!enabled) return
    throw new Error(
      'Renderer.highPrecision = true is not supported by @headless-three/renderer because Three.js CommonRenderer high-precision matrix nodes require backend shader-node state that is outside the scene-oriented API.',
    )
  }

export function renderer_currentSamples_6(this: any): number {
    if (!this.currentRenderTarget) return this.samples
    const targetSamples = this.currentRenderTarget.sampleCount ?? this.currentRenderTarget.samples ?? 1
    return targetSamples > 1 ? targetSamples : 1
  }

export function renderer_compile_7(this: any, scene: ThreeSceneRootLike, camera: ThreeRenderCameraLike, targetScene: ThreeSceneRootLike | null = null): Set<ThreeMaterialLike> {
    validateThreeSceneRoot(scene)
    validateTopLevelRenderCamera(camera)
    if (targetScene !== null) validateThreeSceneRoot(targetScene)
    validateObjectChildrenTree(scene)
    if (targetScene !== null) validateObjectChildrenTree(targetScene)
    return collectCompileMaterials(scene)
  }

export function renderer_setRenderObjectFunction_8(this: any, renderObjectFunction: ((...args: unknown[]) => unknown) | null): void {
    if (renderObjectFunction === null) return
    if (typeof renderObjectFunction !== 'function') {
      throw new TypeError('Renderer.setRenderObjectFunction renderObjectFunction must be a function or null.')
    }
    throw new Error(
      'Renderer.setRenderObjectFunction() is not supported by @headless-three/renderer because it does not expose renderer-internal render-object dispatch. Render normal Three.js scene graphs with Renderer.render() or renderToTarget().',
    )
  }

export function renderer_renderBufferDirect_9(this: any): never {
    throw new Error(
      'Renderer.renderBufferDirect() is not supported by @headless-three/renderer because it does not expose WebGL buffer binding or direct material program dispatch. Render normal Three.js scene graphs with Renderer.render() or renderToTarget().',
    )
  }

export function renderer_renderBufferImmediate_10(this: any): never {
    throw new Error(
      'Renderer.renderBufferImmediate() is not supported by @headless-three/renderer because it does not expose legacy WebGL immediate buffer binding or direct material program dispatch. Render normal Three.js scene graphs with Renderer.render() or renderToTarget().',
    )
  }

export function renderer_renderObject_11(this: any): never {
    throw new Error(
      'Renderer.renderObject() is not supported by @headless-three/renderer because it does not expose renderer-internal render-object dispatch or direct material program dispatch. Render normal Three.js scene graphs with Renderer.render() or renderToTarget().',
    )
  }

export function renderer_compute_12(this: any, computeNodes: unknown, dispatchSize: unknown = null): never {
    assertComputeNodesLike(computeNodes, 'Renderer.compute computeNodes')
    assertComputeDispatchSize(dispatchSize, 'Renderer.compute dispatchSize')
    throw unsupportedComputeError('Renderer.compute')
  }

export async function renderer_computeAsync_13(this: any, computeNodes: unknown, dispatchSize: unknown = null): Promise<never> {
    assertComputeNodesLike(computeNodes, 'Renderer.computeAsync computeNodes')
    assertComputeDispatchSize(dispatchSize, 'Renderer.computeAsync dispatchSize')
    throw unsupportedComputeError('Renderer.computeAsync')
  }

export async function renderer_getArrayBufferAsync_14(this: any, attribute: unknown): Promise<ArrayBuffer> {
    assertStorageBufferAttributeLike(attribute, 'Renderer.getArrayBufferAsync attribute')
    throw new Error(
      'Renderer.getArrayBufferAsync() is not supported by @headless-three/renderer because storage-buffer GPU readback requires WebGPU backend state that this package does not expose. Use Renderer.readRenderTargetPixels() for render-target CPU readback.',
    )
  }

export async function renderer_resolveTimestampsAsync_15(this: any, type: unknown = 'render'): Promise<number> {
    assertTimestampQueryType(type, 'Renderer.resolveTimestampsAsync type')
    throw new Error(
      'Renderer.resolveTimestampsAsync() is not supported by @headless-three/renderer because timestamp queries require backend GPU query pools that are outside the scene-oriented API.',
    )
  }

export async function renderer_waitForGPU_16(this: any): Promise<void> {
    throw new Error(
      'Renderer.waitForGPU() is not supported by @headless-three/renderer because it does not expose direct GPU task synchronization. Renderer.render() and renderToTarget() return after native scene output readback or target writeback has completed.',
    )
  }

export function renderer_setMRT_17(this: any, mrt: unknown = null): Renderer {
    if (mrt !== null) {
      throw new Error(
        'Renderer.setMRT() is not supported by @headless-three/renderer because arbitrary native MRT shader outputs are outside the scene-oriented API. Use target texture userData.headlessThreeRenderer.renderMode for the supported color, mask, object-id, normal, and depth auxiliary outputs.',
      )
    }
    return this
  }

export function renderer_setOutputRenderTarget_18(this: any, renderTarget: unknown = null): void {
    if (renderTarget === null) return
    assertRenderTargetLike(renderTarget, 'Renderer.setOutputRenderTarget renderTarget')
    validateUnsupportedRenderTargetOptions(renderTarget)
    throw new Error(
      'Renderer.setOutputRenderTarget() is not supported by @headless-three/renderer because common-renderer output targets are backend-owned canvas/WebGPU state. Use Renderer.setRenderTarget() or renderToTarget() with a target-like object for offscreen output.',
    )
  }

export function renderer_setCanvasTarget_19(this: any, canvasTarget: unknown = null): void {
    if (canvasTarget === null) return
    assertCanvasTargetLike(canvasTarget, 'Renderer.setCanvasTarget canvasTarget')
    throw new Error(
      'Renderer.setCanvasTarget() is not supported by @headless-three/renderer because it does not own a browser canvas or WebGPU canvas target. Use Renderer.domElement for inert canvas compatibility metadata and Renderer.render() for headless output.',
    )
  }

export function renderer_setTexture2D_20(this: any, texture: unknown, slot: unknown): never {
    assertThreeTextureLike(texture, 'Renderer.setTexture2D texture')
    assertTextureBindingSlot(slot, 'Renderer.setTexture2D slot')
    throw unsupportedTextureBindingError('Renderer.setTexture2D')
  }

export function renderer_setTextureCube_21(this: any, texture: unknown, slot: unknown): never {
    assertThreeTextureLike(texture, 'Renderer.setTextureCube texture')
    assertTextureBindingSlot(slot, 'Renderer.setTextureCube slot')
    throw unsupportedTextureBindingError('Renderer.setTextureCube')
  }

export function renderer_setTextureCubeDynamic_22(this: any, texture: unknown, slot: unknown): never {
    assertThreeTextureLike(texture, 'Renderer.setTextureCubeDynamic texture')
    assertTextureBindingSlot(slot, 'Renderer.setTextureCubeDynamic slot')
    throw unsupportedTextureBindingError('Renderer.setTextureCubeDynamic')
  }

export function renderer_setTexture3D_23(this: any, texture: unknown, slot: unknown): never {
    assertThreeTextureLike(texture, 'Renderer.setTexture3D texture')
    assertTextureBindingSlot(slot, 'Renderer.setTexture3D slot')
    throw unsupportedTextureBindingError('Renderer.setTexture3D')
  }

export function renderer_setTexture2DArray_24(this: any, texture: unknown, slot: unknown): never {
    assertThreeTextureLike(texture, 'Renderer.setTexture2DArray texture')
    assertTextureBindingSlot(slot, 'Renderer.setTexture2DArray slot')
    throw unsupportedTextureBindingError('Renderer.setTexture2DArray')
  }

export function renderer_setRenderTargetTextures_25(this: any, renderTarget: RenderTargetLike, colorTexture: unknown, depthTexture: unknown = null): never {
    assertRenderTargetLike(renderTarget, 'Renderer.setRenderTargetTextures renderTarget')
    validateUnsupportedRenderTargetOptions(renderTarget)
    assertExternalWebGlObjectLike(colorTexture, 'Renderer.setRenderTargetTextures colorTexture')
    assertOptionalExternalWebGlObjectLike(depthTexture, 'Renderer.setRenderTargetTextures depthTexture')
    throw new Error(
      'Renderer.setRenderTargetTextures() is not supported by @headless-three/renderer because WebGLTexture handles cannot be attached to wgpu-backed render targets. Render into a target-like object and use Renderer.readRenderTargetPixels() for CPU readback.',
    )
  }

export function renderer_setRenderTargetFramebuffer_26(this: any, renderTarget: RenderTargetLike, defaultFramebuffer?: unknown): never {
    assertRenderTargetLike(renderTarget, 'Renderer.setRenderTargetFramebuffer renderTarget')
    validateUnsupportedRenderTargetOptions(renderTarget)
    assertOptionalExternalWebGlObjectLike(defaultFramebuffer, 'Renderer.setRenderTargetFramebuffer defaultFramebuffer')
    throw new Error(
      'Renderer.setRenderTargetFramebuffer() is not supported by @headless-three/renderer because it does not expose a browser WebGL framebuffer. Use Renderer.setRenderTarget() with a target-like object for offscreen output.',
    )
  }

export function renderer_copyFramebufferToTexture_27(this: any, textureOrPosition: unknown, positionOrTexture: unknown = null, level = 0): void {
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

export function renderer_copyTextureToTexture_28(this: any, srcTextureOrDstPosition: unknown, dstTextureOrSrcTexture: unknown, srcRegionOrDstTexture: unknown = null, dstPositionOrDstLevel: unknown = null, srcLevel = 0, dstLevel: number | null = null): void {
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

export function renderer_copyTextureToTexture3D_29(this: any, srcTextureOrSrcRegion: unknown, dstTextureOrDstPosition: unknown, _srcRegionOrSrcTexture: unknown = null, _dstPositionOrDstTexture: unknown = null, level = 0): never {
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

export function renderer_setAnimationLoop_30(this: any, callback: RenderAnimationLoopCallback | null): void {
    if (callback !== null && typeof callback !== 'function') {
      throw new TypeError('Renderer.setAnimationLoop callback must be a function or null.')
    }
    this.animationLoop = callback
  }

export function renderer_getContext_31(this: any): never {
    throw new Error(
      'Renderer.getContext() is not supported by @headless-three/renderer because it renders offscreen through wgpu instead of a browser WebGL context.',
    )
  }

export function renderer_setRenderTarget_32(this: any, target: RenderTargetLike | null = null, activeCubeFace = 0, activeMipmapLevel = 0): void {
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

export function renderer_setSize_33(this: any, width: number, height: number, updateStyle = true): void {
    this.currentSize = rendererStateSize(width, height, 'Renderer.setSize')
    this.domElement.setSize(
      this.currentSize.width,
      this.currentSize.height,
      rendererStateBoolean(updateStyle, 'Renderer.setSize updateStyle'),
    )
  }

export function renderer_getSize_34(this: any, target?: RenderSizeLike): RenderSizeLike | null {
    return target === undefined
      ? clonePixelSize(this.currentSize)
      : clonePixelSize(this.currentSize, target)
  }

export function renderer_setDrawingBufferSize_35(this: any, width: number, height: number, pixelRatio: number): void {
    this.currentSize = rendererStateSize(width, height, 'Renderer.setDrawingBufferSize')
    this.pixelRatioValue = rendererStatePixelRatio(pixelRatio, 'Renderer.setDrawingBufferSize pixelRatio')
    this.domElement.setSize(this.currentSize.width, this.currentSize.height)
  }

export function renderer_getDrawingBufferSize_36(this: any, target?: RenderSizeLike): RenderSizeLike | null {
    return target === undefined
      ? clonePixelSize(this.currentSize)
      : clonePixelSize(this.currentSize, target)
  }

export function renderer_getClearColor_37(this: any, target?: ThreeColorLike): ThreeColorLike {
    return target === undefined
      ? cloneColor3(this.currentClearColor)
      : cloneColor3(this.currentClearColor, target)
  }

export function renderer_setClearAlpha_38(this: any, alpha: number): void {
    this.currentClearColor = [
      this.currentClearColor[0],
      this.currentClearColor[1],
      this.currentClearColor[2],
      rendererStateClearAlpha(alpha, 'Renderer.setClearAlpha alpha'),
    ]
  }

export function renderer_getViewport_39(this: any, target?: RenderPixelRectLike): RenderPixelRectLike | null {
    return target === undefined
      ? clonePixelRect(this.currentViewport)
      : clonePixelRect(this.currentViewport, target)
  }

export function renderer_getCurrentViewport_40(this: any, target?: RenderPixelRectLike): RenderPixelRectLike | null {
    return target === undefined
      ? clonePixelRect(this.currentViewport)
      : clonePixelRect(this.currentViewport, target)
  }

export function renderer_getScissor_41(this: any, target?: RenderPixelRectLike): RenderPixelRectLike | null {
    return target === undefined
      ? clonePixelRect(this.currentScissor)
      : clonePixelRect(this.currentScissor, target)
  }

export function renderer_setScissorTest_42(this: any, enabled: boolean): void {
    if (typeof enabled !== 'boolean') {
      throw new TypeError(`Renderer.setScissorTest enabled must be a boolean; received ${String(enabled)}.`)
    }
    this.currentScissorTest = enabled
  }
