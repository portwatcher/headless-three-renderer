/// <reference types="node" />

// ── Three.js duck-typed interfaces ──────────────────────────────────

export type RenderOutputFormat = 'png' | 'rgba'
export type RenderOutputColorSpace = 'srgb' | 'srgb-linear' | 'linear-srgb' | 'linearsrgb' | 'linear'
export type RenderMode = 'color' | 'mask' | 'object-id' | 'normal' | 'depth'
export type RenderAnimationLoopCallback = (time: number, frame?: unknown) => void
export type RendererPowerPreferenceLike = 'default' | 'high-performance' | 'low-power'

export interface RendererInspectorLike {
  currentFrame?: unknown
  setRenderer(renderer: unknown): unknown
  getRenderer?(): unknown
  init?(): unknown
  begin?(): unknown
  finish?(): unknown
  inspect?(node: unknown): unknown
  computeAsync?(computeNode: unknown, dispatchSizeOrCount?: unknown): unknown
  beginCompute?(uid: unknown, computeNode?: unknown): unknown
  finishCompute?(uid?: unknown): unknown
  beginRender?(uid: unknown, scene?: unknown, camera?: unknown, renderTarget?: unknown): unknown
  finishRender?(uid?: unknown): unknown
  copyTextureToTexture?(srcTexture: unknown, dstTexture: unknown): unknown
  copyFramebufferToTexture?(framebufferTexture: unknown): unknown
}

export interface RendererParametersLike {
  canvas?: unknown
  context?: unknown
  alpha?: boolean
  depth?: boolean
  stencil?: boolean
  antialias?: boolean
  premultipliedAlpha?: boolean
  preserveDrawingBuffer?: boolean
  powerPreference?: RendererPowerPreferenceLike
  failIfMajorPerformanceCaveat?: boolean
  samples?: number
  outputBufferType?: number
  logarithmicDepthBuffer?: boolean
  reversedDepthBuffer?: boolean
  reverseDepthBuffer?: boolean
}

export interface RendererContextAttributesLike {
  alpha: boolean
  depth: boolean
  stencil: boolean
  antialias: boolean
  premultipliedAlpha: boolean
  preserveDrawingBuffer: boolean
  powerPreference: RendererPowerPreferenceLike
  failIfMajorPerformanceCaveat: boolean
}

export interface ThreeColorLike {
  r: number
  g: number
  b: number
  isColor?: boolean
}

export type ThreeColorInput = ThreeColorLike | string | number[]

export interface ThreeMatrix4Like {
  elements: ArrayLike<number>
}

export interface ThreeMatrix3Like {
  elements: ArrayLike<number>
}

export interface ThreeSphereLike {
  center?: { x?: number; y?: number; z?: number } | ArrayLike<number>
  radius?: number
}

export interface ThreeBufferAttributeLike {
  count: number
  version?: number
  itemSize?: number
  normalized?: boolean
  isPacked?: boolean
  packingMethod?: unknown
  isInstancedBufferAttribute?: boolean
  meshPerAttribute?: number
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
  version?: number
  isInstancedBufferGeometry?: boolean
  instanceCount?: number
  attributes?: Record<string, ThreeBufferAttributeLike | undefined>
  index?: ThreeBufferAttributeLike | null
  groups?: Array<{ start: number; count: number; materialIndex?: number }>
  drawRange?: { start?: number; count?: number }
  getAttribute?(name: string): ThreeBufferAttributeLike | undefined
  boundingSphere?: ThreeSphereLike | null
  computeBoundingSphere?(): void
  morphAttributes?: Record<string, ThreeBufferAttributeLike[] | undefined>
  morphTargetsRelative?: boolean
}

export interface ThreeTextureLike {
  version?: number
  isTexture?: boolean
  isCubeTexture?: boolean
  isFramebufferTexture?: boolean
  isDepthTexture?: boolean
  isVideoTexture?: boolean
  isStorageTexture?: boolean
  isCompressedTexture?: boolean
  isCompressedArrayTexture?: boolean
  isCompressedCubeTexture?: boolean
  isDataArrayTexture?: boolean
  isData3DTexture?: boolean
  isArrayTexture?: boolean
  is3DTexture?: boolean
  image?: {
    data?: ArrayLike<number>
    width?: number
    height?: number
  } | Buffer | Uint8Array | Array<{
    data?: ArrayLike<number>
    width?: number
    height?: number
  } | Buffer | Uint8Array>
  source?: {
    data?: {
      data?: ArrayLike<number>
      width?: number
      height?: number
    } | Buffer | Uint8Array
  }
  wrapS?: number
  wrapT?: number
  magFilter?: number
  minFilter?: number
  format?: number
  type?: number
  generateMipmaps?: boolean
  needsUpdate?: boolean
  mipmaps?: Array<{
    data?: ArrayLike<number>
    width?: number
    height?: number
  }>
  anisotropy?: number
  mapping?: number
  channel?: number
  offset?: { x?: number; y?: number }
  repeat?: { x?: number; y?: number }
  rotation?: number
  center?: { x?: number; y?: number }
  matrix?: ThreeMatrix3Like
  matrixAutoUpdate?: boolean
  flipY?: boolean
  premultiplyAlpha?: boolean
  unpackAlignment?: number
  colorSpace?: string
  encoding?: number
}

export interface ThreeVector3Like {
  x?: number
  y?: number
  z?: number
}

export interface ThreeEulerLike {
  x?: number
  y?: number
  z?: number
  order?: string
  isEuler?: boolean
}

export type ThreePlaneLike = {
  normal?: ThreeVector3Like | ArrayLike<number>
  constant?: number
} | ArrayLike<number>

export type RenderPixelRectLike = {
  x?: number
  y?: number
  width?: number
  height?: number
  z?: number
  w?: number
} | ArrayLike<number>

export type RenderSizeLike = {
  width?: number
  height?: number
  x?: number
  y?: number
  set?(width: number, height: number): unknown
} | ArrayLike<number>

export interface ThreeLayersLike {
  mask?: number
  test?(layers: ThreeLayersLike): boolean
}

export interface ThreeMaterialLike {
  type?: string
  color?: ThreeColorInput
  opacity?: number
  visible?: boolean
  vertexColors?: boolean
  userData?: Record<string, any>
  onBeforeCompile?: (...args: any[]) => void
  blending?: number
  blendEquation?: number
  blendSrc?: number
  blendDst?: number
  blendEquationAlpha?: number | null
  blendSrcAlpha?: number | null
  blendDstAlpha?: number | null
  blendColor?: ThreeColorInput
  blendAlpha?: number
  premultipliedAlpha?: boolean
  toneMapped?: boolean
  dithering?: boolean
  precision?: 'highp' | 'mediump' | 'lowp' | null
  map?: ThreeTextureLike | null
  envMap?: ThreeTextureLike | null
  envMapIntensity?: number
  envMapRotation?: ThreeEulerLike | ArrayLike<number> | null
  combine?: number
  reflectivity?: number
  refractionRatio?: number
  metalness?: number
  roughness?: number
  clearcoat?: number
  clearcoatMap?: ThreeTextureLike | null
  clearcoatRoughness?: number
  clearcoatRoughnessMap?: ThreeTextureLike | null
  clearcoatNormalMap?: ThreeTextureLike | null
  clearcoatNormalScale?: { x: number; y: number }
  sheen?: number
  sheenColor?: ThreeColorInput
  sheenColorMap?: ThreeTextureLike | null
  sheenRoughness?: number
  sheenRoughnessMap?: ThreeTextureLike | null
  anisotropy?: number
  anisotropyRotation?: number
  anisotropyMap?: ThreeTextureLike | null
  iridescence?: number
  iridescenceMap?: ThreeTextureLike | null
  iridescenceIOR?: number
  iridescenceThicknessRange?: ArrayLike<number>
  iridescenceThicknessMap?: ThreeTextureLike | null
  dispersion?: number
  transmission?: number
  transmissionMap?: ThreeTextureLike | null
  ior?: number
  thickness?: number
  thicknessMap?: ThreeTextureLike | null
  attenuationDistance?: number
  attenuationColor?: ThreeColorInput
  specularColor?: ThreeColorInput
  specularColorMap?: ThreeTextureLike | null
  specularIntensity?: number
  specularIntensityMap?: ThreeTextureLike | null
  specular?: ThreeColorInput
  shininess?: number
  emissive?: ThreeColorInput
  emissiveIntensity?: number
  emissiveMap?: ThreeTextureLike | null
  normalMap?: ThreeTextureLike | null
  normalMapType?: number
  normalScale?: { x: number; y: number }
  bumpMap?: ThreeTextureLike | null
  bumpScale?: number
  displacementMap?: ThreeTextureLike | null
  displacementScale?: number
  displacementBias?: number
  matcap?: ThreeTextureLike | null
  gradientMap?: ThreeTextureLike | null
  metalnessMap?: ThreeTextureLike | null
  roughnessMap?: ThreeTextureLike | null
  specularMap?: ThreeTextureLike | null
  lightMap?: ThreeTextureLike | null
  lightMapIntensity?: number
  aoMap?: ThreeTextureLike | null
  aoMapIntensity?: number
  alphaMap?: ThreeTextureLike | null
  alphaTest?: number
  alphaHash?: boolean
  alphaToCoverage?: boolean
  clippingPlanes?: ThreePlaneLike[] | null
  clipIntersection?: boolean
  clipShadows?: boolean
  depthTest?: boolean
  depthFunc?: number
  depthWrite?: boolean
  colorWrite?: boolean
  polygonOffset?: boolean
  polygonOffsetFactor?: number
  polygonOffsetUnits?: number
  stencilWrite?: boolean
  stencilWriteMask?: number
  stencilFunc?: number
  stencilRef?: number
  stencilFuncMask?: number
  stencilFail?: number
  stencilZFail?: number
  stencilZPass?: number
  transparent?: boolean
  forceSinglePass?: boolean
  side?: number
  shadowSide?: number | null
  flatShading?: boolean
  wireframe?: boolean
  wireframeLinewidth?: number | null
  wireframeLinecap?: 'butt' | 'round' | 'square' | null
  wireframeLinejoin?: 'round' | 'bevel' | 'miter' | null
  fog?: boolean
  isLineDashedMaterial?: boolean
  isMeshBasicMaterial?: boolean
  isMeshDepthMaterial?: boolean
  isMeshDistanceMaterial?: boolean
  isMeshLambertMaterial?: boolean
  isMeshMatcapMaterial?: boolean
  isMeshNormalMaterial?: boolean
  isMeshPhongMaterial?: boolean
  isMeshStandardMaterial?: boolean
  isMeshPhysicalMaterial?: boolean
  isMeshToonMaterial?: boolean
  isShadowMaterial?: boolean
  isLineBasicMaterial?: boolean
  isPointsMaterial?: boolean
  isSpriteMaterial?: boolean
  isShaderMaterial?: boolean
  isRawShaderMaterial?: boolean
  isNodeMaterial?: boolean
  name?: string
  vertexShader?: string
  fragmentShader?: string
  uniforms?: Record<string, { value?: unknown } | unknown>
  depthPacking?: number
  referencePosition?: ThreeVector3Like | ArrayLike<number>
  nearDistance?: number
  farDistance?: number
  dashSize?: number
  gapSize?: number
  scale?: number
  linewidth?: number
  linecap?: 'butt' | 'round' | 'square' | null
  linejoin?: 'round' | 'bevel' | 'miter' | null
  rotation?: number
  customFragmentShader?: string
  customFragmentWgsl?: string
  headlessFragmentShader?: string
  headlessFragmentWgsl?: string
  size?: number
  sizeAttenuation?: boolean
  id?: number
}

export interface ThreeBoneLike {
  matrixWorld?: ThreeMatrix4Like
}

export interface ThreeSkeletonLike {
  bones?: ThreeBoneLike[]
  boneInverses?: ThreeMatrix4Like[]
  update?(): void
}
