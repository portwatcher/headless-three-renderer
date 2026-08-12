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
import { toNativeInput } from './index.part-011'
import { InternalRenderOptions, renderArrayCameraAuxiliaryTargetAttachments, renderCubeCamera, renderRegularCameraAuxiliaryTargetAttachments, renderTargetDepthBuffer, renderTargetHasExplicitSize, resolveCubeTargetSize } from './index.part-012'
import { cubeMipmapScissor, cubeMipmapScissorLabel, cubeMipmapSize, cubeMipmapViewport, cubeMipmapViewportLabel, isCubeRenderTarget, renderArrayCamera, resolveActiveMipmapLevel, writeCubeRenderTargetFace } from './index.part-013'
import { cloneColor4 } from './index.part-014'
import { assertOptionalBoolean, assertRenderOptionsLike, clonePixelRect } from './index.part-015'
import { assertRenderTargetLike } from './index.part-016'
import { assertNonCubeCameraRenderTargetTextures, compositeActiveTargetColorBuffer, writeRenderTarget } from './index.part-018'
import { clearRenderTargetColor, clearRenderTargetDepth, clearRenderTargetStencil } from './index.part-019'
import { copyRenderTargetReadbackPixels, createRenderTargetReadbackBuffer, readbackRect, renderTargetReadbackSource } from './index.part-020'
import { cloneTargetData, depthTextureData, isArrayCamera, isCubeCamera, validateThreeSceneRoot, validateTopLevelRenderCamera } from './index.part-021'
import { GpuFrameLease, GpuFramePool, type GpuFramePoolOptions, type GpuOutputCapabilities, wrapGpuOutputCapabilities } from './gpu-output'
export function renderer_clear_43(this: any, color = true, depth = true, stencil = true): void {
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

export function renderer_clearTarget_44(this: any, target: RenderTargetLike | null, color = true, depth = true, stencil = true): void {
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

export function renderer_dispose_45(this: any): void {
    this.info.dispose()
    this.nodes.dispose()
    this.properties.dispose()
    this.renderLists.dispose()
    this.renderStates.dispose()
    // Native resources are owned by the renderer instance and released with normal object lifetime.
  }

export function renderer_render_46(this: any, scene: ThreeSceneRootLike, camera: ThreeRenderCameraLike, options: RenderOptions = {}): Buffer {
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

export function renderer_renderToTarget_47(this: any, scene: ThreeSceneRootLike, camera: ThreeRenderCameraLike, target: RenderTargetLike = {}, options: RenderOptions = {}): RenderTargetLike {
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

export function renderer_readRenderTargetPixels_48(this: any, target: RenderTargetLike, x: number, y: number, width: number, height: number, buffer: NonNullable<RenderTargetImageLike['data']>, activeCubeFaceIndex?: number, textureIndex = 0): void {
    const readback = renderTargetReadbackSource(
      target,
      activeCubeFaceIndex,
      textureIndex,
      'Renderer.readRenderTargetPixels',
    )
    copyRenderTargetReadbackPixels(readback, x, y, width, height, buffer, 'Renderer.readRenderTargetPixels')
  }

export async function renderer_readRenderTargetPixelsAsync_49(this: any, target: RenderTargetLike, x: number, y: number, width: number, height: number, bufferOrTextureIndex?: NonNullable<RenderTargetImageLike['data']> | number, activeCubeFaceIndexOrFaceIndex?: number, textureIndex = 0): Promise<NonNullable<RenderTargetImageLike['data']>> {
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

export function renderer_renderCurrentRenderTarget_50(this: any, scene: ThreeSceneRootLike, camera: ThreeCameraLike, options: RenderOptions): Buffer {
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

export function renderer_renderCurrentCubeFace_51(this: any, scene: ThreeSceneRootLike, camera: ThreeCameraLike, target: RenderTargetLike, options: RenderOptions): Buffer {
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

export function renderer_renderNative_52(this: any, scene: ThreeSceneRootLike, camera: ThreeCameraLike, options: RenderOptions): { buffer: Buffer; nativeScene: NativeRenderScene; nativeCamera: NativeCamera; objectIdEntries?: RenderObjectIdEntry[] } {
    const { nativeScene, nativeCamera, objectIdEntries } = toNativeInput(scene, camera, options, this.sceneExtractionCache)
    const buffer = this.native.render(nativeScene, nativeCamera)
    commitNativeMeshPayloadCache(this.sceneExtractionCache)
    return { buffer, nativeScene, nativeCamera, objectIdEntries }
  }

export function renderer_getGpuOutputCapabilities_55(this: any): GpuOutputCapabilities {
  return wrapGpuOutputCapabilities(this.native.getGpuOutputCapabilities())
}

export function renderer_renderGpuFrame_56(this: any, scene: ThreeSceneRootLike, camera: ThreeCameraLike, options: RenderOptions = {}): GpuFrameLease {
  validateThreeSceneRoot(scene)
  validateTopLevelRenderCamera(camera)
  assertRenderOptionsLike(options, 'options')
  const renderOptions = this.resolveRenderOptions(options, undefined)
  if (renderOptions.target) {
    throw new Error('Renderer.renderGpuFrame does not support CPU-backed render targets')
  }
  const { nativeScene, nativeCamera } = toNativeInput(scene, camera, { ...renderOptions, format: 'rgba' }, this.sceneExtractionCache)
  const lease = this.native.renderGpuFrame(nativeScene, nativeCamera)
  commitNativeMeshPayloadCache(this.sceneExtractionCache)
  return new GpuFrameLease(lease)
}

export function renderer_createGpuFramePool_57(this: any, options: GpuFramePoolOptions): GpuFramePool {
  if (!options || typeof options !== 'object') throw new TypeError('GPU frame pool options must be an object')
  const normalized: Required<GpuFramePoolOptions> = {
    width: options.width,
    height: options.height,
    capacity: options.capacity ?? 3,
    format: options.format ?? 'rgba8unorm',
    overflow: options.overflow ?? 'error',
  }
  const nativePool = this.native.createGpuFramePool(normalized)
  let pool: GpuFramePool
  pool = new GpuFramePool(nativePool, async (scene, camera, renderOptions) => {
    validateThreeSceneRoot(scene as ThreeSceneRootLike)
    validateTopLevelRenderCamera(camera as ThreeCameraLike)
    assertRenderOptionsLike(renderOptions, 'options')
    const resolved = this.resolveRenderOptions({
      ...(renderOptions as RenderOptions),
      width: normalized.width,
      height: normalized.height,
      format: 'rgba',
    }, undefined)
    if (resolved.target) throw new Error('GpuFramePool.render does not support CPU-backed render targets')
    const input = toNativeInput(scene as ThreeSceneRootLike, camera as ThreeCameraLike, resolved, this.sceneExtractionCache)
    const result = pool.renderNative(input.nativeScene, input.nativeCamera)
    commitNativeMeshPayloadCache(this.sceneExtractionCache)
    return result
  }, normalized)
  return pool
}

export function renderer_resolveRenderOptions_53(this: any, options: RenderOptions, fallbackTarget: RenderTargetLike | null | undefined = options.target): InternalRenderOptions {
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

export function renderer_optionsWithRendererSizeFallback_54(this: any, options: RenderOptions, fallbackTarget: RenderTargetLike | null | undefined): RenderOptions {
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
