import type {
  ThreeSceneLike,
  ThreeCameraLike,
  RenderOptions,
  RenderTargetLike,
  NativeRenderScene,
  NativeCamera,
} from './types'

// eslint-disable-next-line @typescript-eslint/no-var-requires
const native = require('../native.js')

import { resolveSize, cameraViewProjection, cameraWorldPosition } from './camera'
import { resolveBackground } from './color'
import { flattenScene } from './scene'
import { extractLights, extractAmbientLight, extractAmbientIntensity } from './lights'
import { extractEnvironmentMap } from './materials'

export type {
  RenderOutputFormat,
  ThreeColorLike,
  ThreeMatrix4Like,
  ThreeBufferAttributeLike,
  ThreeBufferGeometryLike,
  ThreeTextureLike,
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

  if (typeof scene.updateMatrixWorld === 'function') {
    scene.updateMatrixWorld(true)
  }
  if (typeof camera.updateMatrixWorld === 'function') {
    camera.updateMatrixWorld(true)
  }

  const size = resolveSize(camera, options)
  const envMap = extractEnvironmentMap(scene)
  const nativeScene: NativeRenderScene = {
    width: size.width,
    height: size.height,
    background: resolveBackground(scene, options),
    format: options.format ?? (options.target ? 'rgba' : 'png'),
    meshes: flattenScene(scene),
    lights: extractLights(scene),
    ambientLight: extractAmbientLight(scene) ?? undefined,
    ambientIntensity: extractAmbientIntensity(scene) ?? undefined,
    environmentMap: envMap?.data,
    environmentMapWidth: envMap?.width,
    environmentMapHeight: envMap?.height,
    environmentMapIntensity: envMap?.intensity,
    ...postProcessingToNative(options.postProcessing),
  }
  const nativeCamera: NativeCamera = {
    width: size.width,
    height: size.height,
    viewProjection: cameraViewProjection(camera),
    cameraPosition: cameraWorldPosition(camera),
  }

  return { nativeScene, nativeCamera }
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

function finiteOrUndefined(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function booleanOrNumber(value: unknown): number | undefined {
  if (typeof value === 'boolean') return value ? 1 : 0
  return finiteOrUndefined(value)
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

  if (target.texture) {
    const textureImage = target.texture.image ?? (target.texture.image = {})
    textureImage.data = data
    textureImage.width = width
    textureImage.height = height

    if (target.texture.source?.data) {
      target.texture.source.data.data = data
      target.texture.source.data.width = width
      target.texture.source.data.height = height
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
