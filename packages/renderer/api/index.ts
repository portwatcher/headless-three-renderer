import type {
  ThreeSceneLike,
  ThreeCameraLike,
  RenderOptions,
  RenderTargetLike,
  RenderPixelRectLike,
  NativeRenderScene,
  NativeCamera,
} from './types'

// eslint-disable-next-line @typescript-eslint/no-var-requires
const native = require('../native.js')

import { resolveSize, cameraViewProjection, cameraViewMatrix, cameraWorldPosition } from './camera'
import { colorLikeToArray, resolveBackground } from './color'
import { flattenScene } from './scene'
import { extractLights, extractAmbientLight, extractAmbientIntensity, extractLightProbe } from './lights'
import { extractBackgroundTexture, extractEnvironmentMap } from './materials'
import { extractClippingPlanes } from './clipping'

export type {
  RenderOutputFormat,
  RenderOutputColorSpace,
  ThreeColorLike,
  ThreeMatrix4Like,
  ThreeBufferAttributeLike,
  ThreeBufferGeometryLike,
  ThreeTextureLike,
  ThreeVector3Like,
  ThreePlaneLike,
  RenderPixelRectLike,
  ThreeLayersLike,
  ThreeMaterialLike,
  ThreeBoneLike,
  ThreeSkeletonLike,
  ThreeObject3DLike,
  ThreeSceneLike,
  ThreeCameraLike,
  RenderOptions,
  RenderTargetLike,
  PostProcessingOptions,
} from './types'

export class Renderer {
  private native: InstanceType<typeof native.NativeRenderer>

  constructor() {
    this.native = new native.NativeRenderer()
  }

  render(scene: ThreeSceneLike, camera: ThreeCameraLike, options: RenderOptions = {}): Buffer {
    const { buffer, nativeScene } = this.renderNative(scene, camera, options)
    if (options.target) {
      writeRenderTarget(options.target, buffer, nativeScene.width!, nativeScene.height!)
    }
    return buffer
  }

  renderToTarget(
    scene: ThreeSceneLike,
    camera: ThreeCameraLike,
    target: RenderTargetLike = {},
    options: RenderOptions = {},
  ): RenderTargetLike {
    const targetOptions: RenderOptions = { ...options, target, format: options.format ?? 'rgba' }
    const { buffer, nativeScene } = this.renderNative(scene, camera, targetOptions)
    return writeRenderTarget(target, buffer, nativeScene.width!, nativeScene.height!)
  }

  private renderNative(
    scene: ThreeSceneLike,
    camera: ThreeCameraLike,
    options: RenderOptions,
  ): { buffer: Buffer; nativeScene: NativeRenderScene } {
    const { nativeScene, nativeCamera } = toNativeInput(scene, camera, options)
    return { buffer: this.native.render(nativeScene, nativeCamera), nativeScene }
  }
}

export function render(scene: ThreeSceneLike, camera: ThreeCameraLike, options: RenderOptions = {}): Buffer {
  const { nativeScene, nativeCamera } = toNativeInput(scene, camera, options)
  const buffer = native.renderNative(nativeScene, nativeCamera)
  if (options.target) {
    writeRenderTarget(options.target, buffer, nativeScene.width!, nativeScene.height!)
  }
  return buffer
}

export function renderToTarget(
  scene: ThreeSceneLike,
  camera: ThreeCameraLike,
  target: RenderTargetLike = {},
  options: RenderOptions = {},
): RenderTargetLike {
  const targetOptions: RenderOptions = { ...options, target, format: options.format ?? 'rgba' }
  const { nativeScene, nativeCamera } = toNativeInput(scene, camera, targetOptions)
  const buffer = native.renderNative(nativeScene, nativeCamera)
  return writeRenderTarget(target, buffer, nativeScene.width!, nativeScene.height!)
}

function toNativeInput(
  scene: ThreeSceneLike,
  camera: ThreeCameraLike,
  options: RenderOptions,
): { nativeScene: NativeRenderScene; nativeCamera: NativeCamera } {
  validateThreeScene(scene)
  validateThreeCamera(camera)
  validateUnsupportedRenderOptions(options)

  if (typeof scene.updateMatrixWorld === 'function') {
    scene.updateMatrixWorld(true)
  }
  if (typeof camera.updateMatrixWorld === 'function') {
    camera.updateMatrixWorld(true)
  }

  const size = resolveSize(camera, options)
  const envMap = extractEnvironmentMap(scene)
  const hasBackgroundOverride = options.background !== undefined
  const optionBackgroundTexture = hasBackgroundOverride
    ? extractBackgroundTexture(options.background, 'options.background')
    : null
  const backgroundTexture = optionBackgroundTexture ?? (
    hasBackgroundOverride ? null : extractBackgroundTexture(scene.background, 'scene.background')
  )
  const clippingPlanes = extractClippingPlanes(options.clippingPlanes)
  const nativeScene: NativeRenderScene = {
    width: size.width,
    height: size.height,
    background: resolveBackground(scene, options),
    backgroundIntensity: options.backgroundIntensity ?? scene.backgroundIntensity,
    viewport: pixelRectToArray(options.viewport),
    scissor: pixelRectToArray(options.scissor),
    backgroundTexture: backgroundTexture?.data,
    backgroundTextureWidth: backgroundTexture?.width,
    backgroundTextureHeight: backgroundTexture?.height,
    backgroundTextureWrapS: backgroundTexture?.wrapS,
    backgroundTextureWrapT: backgroundTexture?.wrapT,
    backgroundTextureMagFilter: backgroundTexture?.magFilter,
    backgroundTextureMinFilter: backgroundTexture?.minFilter,
    backgroundTextureTransform: backgroundTexture?.transform,
    backgroundTextureColorSpace: backgroundTexture?.colorSpace,
    backgroundTextureBlurriness: finiteOrUndefined(options.backgroundBlurriness ?? scene.backgroundBlurriness),
    format: options.format ?? (options.target ? 'rgba' : 'png'),
    outputColorSpace: options.outputColorSpace,
    meshes: flattenScene(scene, camera, size.height, clippingPlanes),
    lights: extractLights(scene, camera),
    ambientLight: extractAmbientLight(scene, camera) ?? undefined,
    ambientIntensity: extractAmbientIntensity(scene, camera) ?? undefined,
    lightProbe: extractLightProbe(scene, camera) ?? undefined,
    environmentMap: envMap?.data,
    environmentMapWidth: envMap?.width,
    environmentMapHeight: envMap?.height,
    environmentMapIntensity: envMap?.intensity,
    ...fogToNative(scene.fog),
    ...postProcessingToNative(options.postProcessing),
  }
  const nativeCamera: NativeCamera = {
    width: size.width,
    height: size.height,
    near: finiteOrUndefined(camera.near),
    far: finiteOrUndefined(camera.far),
    viewProjection: cameraViewProjection(camera),
    viewMatrix: cameraViewMatrix(camera),
    cameraPosition: cameraWorldPosition(camera),
  }

  return { nativeScene, nativeCamera }
}

function fogToNative(fog: ThreeSceneLike['fog']): Partial<NativeRenderScene> {
  if (!fog) return {}
  const color = colorLikeToArray(fog.color)
  if (fog.isFogExp2) {
    return {
      fogType: 'exp2',
      fogColor: color ?? undefined,
      fogDensity: finiteOrUndefined(fog.density),
    }
  }
  if (fog.isFog) {
    return {
      fogType: 'linear',
      fogColor: color ?? undefined,
      fogNear: finiteOrUndefined(fog.near),
      fogFar: finiteOrUndefined(fog.far),
    }
  }
  return {}
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
  if (typeof (rect as ArrayLike<number>).length === 'number') {
    const values = rect as ArrayLike<number>
    return [values[0], values[1], values[2], values[3]]
  }
  const values = rect as { x?: number; y?: number; width?: number; height?: number }
  return [values.x!, values.y!, values.width!, values.height!]
}

function finiteOrUndefined(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function booleanOrNumber(value: unknown): number | undefined {
  if (typeof value === 'boolean') return value ? 1 : 0
  return finiteOrUndefined(value)
}

function validateUnsupportedRenderOptions(options: RenderOptions): void {
  assertSupportedSampleCount(options.samples, 'options.samples')
  assertSupportedSampleCount(options.sampleCount, 'options.sampleCount')
  if (options.target) validateUnsupportedRenderTargetOptions(options.target)
}

function validateUnsupportedRenderTargetOptions(target: RenderTargetLike): void {
  if (target.depthTexture != null) {
    throw new Error(
      'Render target depthTexture output is not supported by @headless-three/renderer yet. Render depth with MeshDepthMaterial or omit target.depthTexture until depth readback support lands.',
    )
  }
  if (target.isWebGLMultipleRenderTargets === true || Array.isArray(target.texture)) {
    throw new Error(
      'Multiple render target color attachments are not supported by @headless-three/renderer yet. Render separate passes or use a single color target until MRT support lands.',
    )
  }
  if (Array.isArray(target.textures) && target.textures.length > 1) {
    throw new Error(
      'Multiple render target color attachments are not supported by @headless-three/renderer yet. Render separate passes or use a single color target until MRT support lands.',
    )
  }
  assertSupportedSampleCount(target.samples, 'target.samples')
  assertSupportedSampleCount(target.sampleCount, 'target.sampleCount')
}

function assertSupportedSampleCount(value: unknown, label: string): void {
  if (typeof value === 'number' && Number.isFinite(value) && value > 1) {
    throw new Error(
      `MSAA sample counts greater than 1 are not supported by @headless-three/renderer yet (${label}=${value}). Use the default single-sample render path until MSAA support lands.`,
    )
  }
}

function writeRenderTarget(
  target: RenderTargetLike,
  data: Buffer,
  width: number,
  height: number,
): RenderTargetLike {
  target.width = width
  target.height = height
  target.data = data

  const image = target.image ?? (target.image = {})
  image.data = data
  image.width = width
  image.height = height

  const texture = target.texture ?? target.textures?.[0]
  if (texture && !Array.isArray(texture)) {
    const textureImage = texture.image ?? (texture.image = {})
    textureImage.data = data
    textureImage.width = width
    textureImage.height = height

    if (texture.source?.data) {
      texture.source.data.data = data
      texture.source.data.width = width
      texture.source.data.height = height
    }
  }

  return target
}

function validateThreeScene(scene: unknown): asserts scene is ThreeSceneLike {
  if (!scene || (scene as any).isScene !== true) {
    throw new TypeError('render(scene, camera) expects scene to be a THREE.Scene')
  }
}

function validateThreeCamera(camera: unknown): asserts camera is ThreeCameraLike {
  if (!camera || (camera as any).isCamera !== true) {
    throw new TypeError('render(scene, camera) expects camera to be a THREE.Camera')
  }
  if (!(camera as any).projectionMatrix || !(camera as any).matrixWorldInverse) {
    throw new TypeError('THREE.Camera must have projectionMatrix and matrixWorldInverse')
  }
}
