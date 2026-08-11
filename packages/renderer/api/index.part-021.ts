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
import { FloatType, HalfFloatType, UnsignedByteType, UnsignedInt248Type, UnsignedIntType, UnsignedShortType } from './index.part-012'
export function depthTextureData(texture: RenderTargetTextureLike, rgbaDepth: Buffer): NonNullable<RenderTargetImageLike['data']> {
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

export function normalizedFloatToHalf(value: number): number {
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

export function cloneTargetData(data: NonNullable<RenderTargetImageLike['data']>): NonNullable<RenderTargetImageLike['data']> {
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

export function validateThreeSceneRoot(scene: unknown): asserts scene is ThreeSceneRootLike {
  const root = scene as any
  if (!root || (root.isScene !== true && root.isObject3D !== true)) {
    throw new TypeError('render(scene, camera) expects scene to be a THREE.Scene or THREE.Object3D root')
  }
}

export function validateTopLevelRenderCamera(camera: unknown): asserts camera is ThreeRenderCameraLike {
  const cameraLike = camera as any
  if (!cameraLike || typeof cameraLike !== 'object' || Array.isArray(cameraLike)) {
    throw new TypeError('render(scene, camera) expects camera to be a THREE.Camera, THREE.ArrayCamera, or THREE.CubeCamera')
  }
  if (isCubeCamera(cameraLike)) return
  if (cameraLike.isCamera !== true) {
    throw new TypeError('render(scene, camera) expects camera to be a THREE.Camera, THREE.ArrayCamera, or THREE.CubeCamera')
  }
}

export function isArrayCamera(camera: unknown): camera is ThreeCameraLike {
  const cameraLike = camera as any
  return cameraLike?.isArrayCamera === true || Array.isArray(cameraLike?.cameras)
}

export function isCubeCamera(camera: unknown): camera is ThreeCubeCameraLike {
  const cameraLike = camera as any
  return cameraLike?.isCubeCamera === true || cameraLike?.type === 'CubeCamera'
}

export function validateThreeCamera(camera: unknown, label = 'render(scene, camera)'): asserts camera is ThreeCameraLike {
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
