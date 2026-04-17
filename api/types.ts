/// <reference types="node" />

// ── Three.js duck-typed interfaces ──────────────────────────────────

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
  morphAttributes?: Record<string, ThreeBufferAttributeLike[] | undefined>
  morphTargetsRelative?: boolean
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
  wrapS?: number
  wrapT?: number
}

export interface ThreeMaterialLike {
  color?: ThreeColorLike
  opacity?: number
  visible?: boolean
  vertexColors?: boolean
  map?: ThreeTextureLike | null
  metalness?: number
  roughness?: number
  emissive?: ThreeColorLike
  emissiveIntensity?: number
  emissiveMap?: ThreeTextureLike | null
  normalMap?: ThreeTextureLike | null
  normalScale?: { x: number; y: number }
  metalnessMap?: ThreeTextureLike | null
  roughnessMap?: ThreeTextureLike | null
  alphaTest?: number
  transparent?: boolean
  isMeshBasicMaterial?: boolean
  isMeshStandardMaterial?: boolean
  isMeshPhysicalMaterial?: boolean
}

export interface ThreeBoneLike {
  matrixWorld?: ThreeMatrix4Like
}

export interface ThreeSkeletonLike {
  bones?: ThreeBoneLike[]
  boneInverses?: ThreeMatrix4Like[]
  update?(): void
}

export interface ThreeObject3DLike {
  visible?: boolean
  children?: ThreeObject3DLike[]
  isMesh?: boolean
  isSkinnedMesh?: boolean
  isLight?: boolean
  isDirectionalLight?: boolean
  isPointLight?: boolean
  isSpotLight?: boolean
  isHemisphereLight?: boolean
  isAmbientLight?: boolean
  geometry?: ThreeBufferGeometryLike
  material?: ThreeMaterialLike | ThreeMaterialLike[]
  matrixWorld?: ThreeMatrix4Like
  skeleton?: ThreeSkeletonLike
  bindMatrix?: ThreeMatrix4Like
  bindMatrixInverse?: ThreeMatrix4Like
  morphTargetInfluences?: number[]
  morphTargetDictionary?: Record<string, number>
  color?: ThreeColorLike
  groundColor?: ThreeColorLike
  intensity?: number
  distance?: number
  decay?: number
  angle?: number
  penumbra?: number
  target?: ThreeObject3DLike & { matrixWorld?: ThreeMatrix4Like }
  name?: string
  uuid?: string
}

export interface ThreeSceneLike extends ThreeObject3DLike {
  isScene: true
  background?: ThreeColorLike | null
  environment?: ThreeTextureLike | null
  environmentIntensity?: number
  updateMatrixWorld?(force?: boolean): void
}

export interface ThreeCameraLike {
  isCamera: true
  projectionMatrix: ThreeMatrix4Like
  matrixWorldInverse: ThreeMatrix4Like
  matrixWorld?: ThreeMatrix4Like
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

// ── Native (Rust NAPI) types ────────────────────────────────────────

export interface NativeCamera {
  width?: number
  height?: number
  eye?: number[]
  target?: number[]
  up?: number[]
  fovYDegrees?: number
  near?: number
  far?: number
  viewProjection?: number[]
  cameraPosition?: number[]
}

export interface NativeSceneLight {
  lightType: string
  color?: number[]
  intensity?: number
  position?: number[]
  direction?: number[]
  distance?: number
  decay?: number
  angle?: number
  penumbra?: number
  groundColor?: number[]
}

export interface NativeSceneMesh {
  positions: number[]
  indices?: number[]
  normals?: number[]
  colors?: number[]
  color?: number[]
  transform?: number[]
  uvs?: number[]
  texture?: Buffer
  textureWidth?: number
  textureHeight?: number
  textureWrapS?: string
  textureWrapT?: string
  metallic?: number
  roughness?: number
  emissive?: number[]
  emissiveIntensity?: number
  normalMap?: Buffer
  normalMapWidth?: number
  normalMapHeight?: number
  normalScale?: number[]
  metallicRoughnessTexture?: Buffer
  metallicRoughnessTextureWidth?: number
  metallicRoughnessTextureHeight?: number
  emissiveMap?: Buffer
  emissiveMapWidth?: number
  emissiveMapHeight?: number
  alphaTest?: number
  transparent?: boolean
}

export interface NativeRenderScene {
  width?: number
  height?: number
  background?: number[]
  format?: string
  meshes?: NativeSceneMesh[]
  lights?: NativeSceneLight[]
  ambientLight?: number[]
  ambientIntensity?: number
  environmentMap?: Buffer
  environmentMapWidth?: number
  environmentMapHeight?: number
  environmentMapIntensity?: number
}

// ── Internal helper types ───────────────────────────────────────────

export type Color4 = [number, number, number, number]
export type Mat4 = number[]
export type Vec3 = [number, number, number]

export interface PbrProperties {
  metallic?: number
  roughness?: number
  emissive?: number[]
  emissiveIntensity?: number
  normalMap?: Buffer
  normalMapWidth?: number
  normalMapHeight?: number
  normalScale?: number[]
  metallicRoughnessTexture?: Buffer
  metallicRoughnessTextureWidth?: number
  metallicRoughnessTextureHeight?: number
  emissiveMap?: Buffer
  emissiveMapWidth?: number
  emissiveMapHeight?: number
  alphaTest?: number
  transparent?: boolean
}

export interface TextureInfo {
  data: Buffer
  width: number
  height: number
  wrapS?: string
  wrapT?: string
}

export interface GeometryGroup {
  start: number
  count: number
  materialIndex: number
}
