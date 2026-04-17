import type {
  ThreeSceneLike,
  ThreeCameraLike,
  RenderOptions,
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
} from './types'

export class Renderer {
  private native: InstanceType<typeof native.NativeRenderer>

  constructor() {
    this.native = new native.NativeRenderer()
  }

  render(scene: ThreeSceneLike, camera: ThreeCameraLike, options: RenderOptions = {}): Buffer {
    const { nativeScene, nativeCamera } = toNativeInput(scene, camera, options)
    return this.native.render(nativeScene, nativeCamera)
  }
}

export function render(scene: ThreeSceneLike, camera: ThreeCameraLike, options: RenderOptions = {}): Buffer {
  const { nativeScene, nativeCamera } = toNativeInput(scene, camera, options)
  return native.renderNative(nativeScene, nativeCamera)
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
    format: options.format ?? 'png',
    meshes: flattenScene(scene),
    lights: extractLights(scene),
    ambientLight: extractAmbientLight(scene) ?? undefined,
    ambientIntensity: extractAmbientIntensity(scene) ?? undefined,
    environmentMap: envMap?.data,
    environmentMapWidth: envMap?.width,
    environmentMapHeight: envMap?.height,
    environmentMapIntensity: envMap?.intensity,
  }
  const nativeCamera: NativeCamera = {
    width: size.width,
    height: size.height,
    viewProjection: cameraViewProjection(camera),
    cameraPosition: cameraWorldPosition(camera),
  }

  return { nativeScene, nativeCamera }
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
