import type { ThreeCameraLike, RenderOptions, Mat4 } from './types'
import { OPENGL_TO_WGPU_CLIP, multiplyMatrices, matrixElements } from './math'

const DEFAULT_WIDTH = 512
const DEFAULT_HEIGHT = 512

export function resolveSize(camera: ThreeCameraLike, options: RenderOptions): { width: number; height: number } {
  let width = optionalDimension(options.width, 'options.width') ?? optionalDimension(options.target?.width, 'target.width')
  let height = optionalDimension(options.height, 'options.height') ?? optionalDimension(options.target?.height, 'target.height')
  let widthLabel = options.width != null ? 'options.width' : options.target?.width != null ? 'target.width' : 'output width'
  let heightLabel = options.height != null ? 'options.height' : options.target?.height != null ? 'target.height' : 'output height'

  if (width == null && height == null) {
    width = optionalDimension(camera.userData?.width, 'camera.userData.width') ?? DEFAULT_WIDTH
    height = optionalDimension(camera.userData?.height, 'camera.userData.height')
    widthLabel = camera.userData?.width != null ? 'camera.userData.width' : 'output width'
    heightLabel = camera.userData?.height != null ? 'camera.userData.height' : 'output height'
  }
  if (height == null && width != null && camera.aspect != null) {
    const aspect = requiredPositiveFiniteNumber(camera.aspect, 'camera.aspect')
    height = Math.round(width / aspect)
    heightLabel = 'camera.aspect-derived height'
  }
  if (width == null && height != null && camera.aspect != null) {
    const aspect = requiredPositiveFiniteNumber(camera.aspect, 'camera.aspect')
    width = Math.round(height * aspect)
    widthLabel = 'camera.aspect-derived width'
  }

  width ??= DEFAULT_WIDTH
  height ??= DEFAULT_HEIGHT

  if (!Number.isInteger(width) || width <= 0) {
    throw new TypeError(`${widthLabel} must be a positive integer.`)
  }
  if (!Number.isInteger(height) || height <= 0) {
    throw new TypeError(`${heightLabel} must be a positive integer.`)
  }

  return { width, height }
}

function optionalDimension(value: unknown, label: string): number | undefined {
  if (value == null) return undefined
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a finite number.`)
  }
  return value
}

function requiredPositiveFiniteNumber(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a finite number.`)
  }
  if (value <= 0) {
    throw new TypeError(`${label} must be positive.`)
  }
  return value
}

export function cameraViewProjection(camera: ThreeCameraLike): Mat4 {
  const projection = matrixElements(camera.projectionMatrix, 'camera.projectionMatrix')
  const view = matrixElements(camera.matrixWorldInverse, 'camera.matrixWorldInverse')
  return multiplyMatrices(OPENGL_TO_WGPU_CLIP, multiplyMatrices(projection, view))
}

export function cameraViewMatrix(camera: ThreeCameraLike): Mat4 {
  return matrixElements(camera.matrixWorldInverse, 'camera.matrixWorldInverse')
}

export function cameraWorldPosition(camera: ThreeCameraLike): number[] {
  if (camera.matrixWorld?.elements) {
    const e = matrixElements(camera.matrixWorld, 'camera.matrixWorld')
    return [e[12], e[13], e[14]]
  }
  if (camera.matrixWorldInverse?.elements) {
    const e = matrixElements(camera.matrixWorldInverse, 'camera.matrixWorldInverse')
    const tx = e[12], ty = e[13], tz = e[14]
    return [
      -(e[0] * tx + e[1] * ty + e[2] * tz),
      -(e[4] * tx + e[5] * ty + e[6] * tz),
      -(e[8] * tx + e[9] * ty + e[10] * tz),
    ]
  }
  return [0, 0, 0]
}
