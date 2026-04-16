/// <reference types="node" />

export type RenderOutputFormat = 'png' | 'rgba' | 'raw' | 'raw-rgba'

export interface ThreeColorLike {
  r: number
  g: number
  b: number
  isColor?: boolean
}

export interface ThreeMatrix4Like {
  elements: ArrayLike<number>
}

export interface ThreeBufferAttributeLike {
  count: number
  itemSize?: number
  normalized?: boolean
  array?: ArrayLike<number>
  data?: {
    array: ArrayLike<number>
    stride: number
  }
  offset?: number
  getX?(index: number): number
  getY?(index: number): number
  getZ?(index: number): number
  getW?(index: number): number
}

export interface ThreeBufferGeometryLike {
  attributes?: Record<string, ThreeBufferAttributeLike | undefined>
  index?: ThreeBufferAttributeLike | null
  groups?: Array<{ start: number; count: number; materialIndex?: number }>
  drawRange?: { start?: number; count?: number }
  getAttribute?(name: string): ThreeBufferAttributeLike | undefined
}

export interface ThreeTextureLike {
  image?: {
    data?: ArrayLike<number>
    width?: number
    height?: number
  } | Buffer | Uint8Array
  source?: {
    data?: {
      data?: ArrayLike<number>
      width?: number
      height?: number
    } | Buffer | Uint8Array
  }
}

export interface ThreeMaterialLike {
  color?: ThreeColorLike
  opacity?: number
  visible?: boolean
  vertexColors?: boolean
  map?: ThreeTextureLike | null
}

export interface ThreeObject3DLike {
  visible?: boolean
  children?: ThreeObject3DLike[]
  isMesh?: boolean
  geometry?: ThreeBufferGeometryLike
  material?: ThreeMaterialLike | ThreeMaterialLike[]
  matrixWorld?: ThreeMatrix4Like
  name?: string
  uuid?: string
}

export interface ThreeSceneLike extends ThreeObject3DLike {
  isScene: true
  background?: ThreeColorLike | null
  updateMatrixWorld?(force?: boolean): void
}

export interface ThreeCameraLike {
  isCamera: true
  projectionMatrix: ThreeMatrix4Like
  matrixWorldInverse: ThreeMatrix4Like
  aspect?: number
  userData?: {
    width?: number
    height?: number
  }
  updateMatrixWorld?(force?: boolean): void
}

export interface RenderOptions {
  width?: number
  height?: number
  background?: number[] | ThreeColorLike
  format?: RenderOutputFormat
}

export class Renderer {
  constructor()
  render(scene: ThreeSceneLike, camera: ThreeCameraLike, options?: RenderOptions): Buffer
}

export function render(scene: ThreeSceneLike, camera: ThreeCameraLike, options?: RenderOptions): Buffer
