import type { ThreeCameraLike, RenderOptions, Mat4 } from './types'
import { OPENGL_TO_WGPU_CLIP, multiplyMatrices, matrixElements, numberOrUndefined, isFinitePositive } from './math'

const DEFAULT_WIDTH = 512
const DEFAULT_HEIGHT = 512

export function resolveSize(camera: ThreeCameraLike, options: RenderOptions): { width: number; height: number } {
  let width = numberOrUndefined(options.width)
  let height = numberOrUndefined(options.height)

  if (width == null && height == null) {
    width = numberOrUndefined(camera.userData?.width) ?? DEFAULT_WIDTH
    height = numberOrUndefined(camera.userData?.height)
  }
  if (height == null && width != null && isFinitePositive(camera.aspect)) {
    height = Math.round(width / camera.aspect)
  }
  if (width == null && height != null && isFinitePositive(camera.aspect)) {
    width = Math.round(height * camera.aspect!)
  }

  width ??= DEFAULT_WIDTH
  height ??= DEFAULT_HEIGHT

  if (!Number.isInteger(width) || width <= 0) {
    throw new TypeError('render options width must be a positive integer')
  }
  if (!Number.isInteger(height) || height <= 0) {
    throw new TypeError('render options height must be a positive integer')
  }

  return { width, height }
}

export function cameraViewProjection(camera: ThreeCameraLike): Mat4 {
  const projection = matrixElements(camera.projectionMatrix, 'camera.projectionMatrix')
  const view = matrixElements(camera.matrixWorldInverse, 'camera.matrixWorldInverse')
  return multiplyMatrices(OPENGL_TO_WGPU_CLIP, multiplyMatrices(projection, view))
}

export function cameraWorldPosition(camera: ThreeCameraLike): number[] {
  if (camera.matrixWorld?.elements) {
    const e = camera.matrixWorld.elements
    return [e[12], e[13], e[14]]
  }
  if (camera.matrixWorldInverse?.elements) {
    const e = camera.matrixWorldInverse.elements
    const tx = e[12], ty = e[13], tz = e[14]
    return [
      -(e[0] * tx + e[1] * ty + e[2] * tz),
      -(e[4] * tx + e[5] * ty + e[6] * tz),
      -(e[8] * tx + e[9] * ty + e[10] * tz),
    ]
  }
  return [0, 0, 0]
}
