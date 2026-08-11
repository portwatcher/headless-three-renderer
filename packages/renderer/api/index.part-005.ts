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
import { rendererStateBoolean, rendererStateClearAlpha, rendererStateClearDepth, rendererStateClearStencil } from './index.part-014'
import { assertDomElementAttributeName, assertEventListener, domElementStylePropertyKey, domElementStyleWritablePropertyKey } from './index.part-017'
export class RendererDomElementState {
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

export type RendererDomElementStyle = {
  width: string
  height: string
  setProperty(propertyName: unknown, value?: unknown): void
  getPropertyValue(propertyName: unknown): string
  removeProperty(propertyName: unknown): string
  [key: string]: unknown
}

export function createRendererDomElementStyle(): RendererDomElementStyle {
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

export class RendererColorBufferState {
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

export class RendererDepthBufferState {
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

export class RendererStencilBufferState {
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

export class RendererStateBuffersState {
  readonly color = new RendererColorBufferState()
  readonly depth = new RendererDepthBufferState()
  readonly stencil = new RendererStencilBufferState()
}
