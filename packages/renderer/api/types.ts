/// <reference types="node" />

// ── Three.js duck-typed interfaces ──────────────────────────────────

export type RenderOutputFormat = 'png' | 'rgba'
export type RenderOutputColorSpace = 'srgb' | 'srgb-linear' | 'linear-srgb' | 'linear'
export type RenderMode = 'color' | 'mask' | 'object-id'

export interface ThreeColorLike {
  r: number
  g: number
  b: number
  isColor?: boolean
}

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
  itemSize?: number
  normalized?: boolean
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
  isTexture?: boolean
  isCubeTexture?: boolean
  isCompressedTexture?: boolean
  isCompressedArrayTexture?: boolean
  isCompressedCubeTexture?: boolean
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
  generateMipmaps?: boolean
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

export interface ThreeLayersLike {
  mask?: number
  test?(layers: ThreeLayersLike): boolean
}

export interface ThreeMaterialLike {
  type?: string
  color?: ThreeColorLike
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
  blendColor?: ThreeColorLike
  blendAlpha?: number
  premultipliedAlpha?: boolean
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
  sheenColor?: ThreeColorLike
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
  attenuationColor?: ThreeColorLike
  specularColor?: ThreeColorLike
  specularColorMap?: ThreeTextureLike | null
  specularIntensity?: number
  specularIntensityMap?: ThreeTextureLike | null
  specular?: ThreeColorLike
  shininess?: number
  emissive?: ThreeColorLike
  emissiveIntensity?: number
  emissiveMap?: ThreeTextureLike | null
  normalMap?: ThreeTextureLike | null
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
  side?: number
  shadowSide?: number | null
  flatShading?: boolean
  wireframe?: boolean
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
  depthPacking?: number
  referencePosition?: ThreeVector3Like | ArrayLike<number>
  nearDistance?: number
  farDistance?: number
  dashSize?: number
  gapSize?: number
  scale?: number
  linewidth?: number
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

export interface ThreeObject3DLike {
  isObject3D?: boolean
  visible?: boolean
  layers?: ThreeLayersLike
  renderOrder?: number
  id?: number
  children?: ThreeObject3DLike[]
  userData?: Record<string, any>
  isMesh?: boolean
  isInstancedMesh?: boolean
  isSkinnedMesh?: boolean
  isLOD?: boolean
  isGroup?: boolean
  isClippingGroup?: boolean
  isLine?: boolean
  isLineSegments?: boolean
  isLineLoop?: boolean
  isPoints?: boolean
  isSprite?: boolean
  isLight?: boolean
  isDirectionalLight?: boolean
  isPointLight?: boolean
  isSpotLight?: boolean
  isRectAreaLight?: boolean
  isHemisphereLight?: boolean
  isAmbientLight?: boolean
  isLightProbe?: boolean
  geometry?: ThreeBufferGeometryLike
  material?: ThreeMaterialLike | ThreeMaterialLike[]
  clippingPlanes?: ThreePlaneLike[] | null
  enabled?: boolean
  clipIntersection?: boolean
  clipShadows?: boolean
  customDepthMaterial?: ThreeMaterialLike
  customDistanceMaterial?: ThreeMaterialLike
  center?: { x?: number; y?: number }
  count?: number
  instanceMatrix?: ThreeBufferAttributeLike
  instanceColor?: ThreeBufferAttributeLike | null
  matrixWorld?: ThreeMatrix4Like
  autoUpdate?: boolean
  levels?: Array<{ object: ThreeObject3DLike; distance?: number; hysteresis?: number }>
  update?(camera: ThreeCameraLike): void
  skeleton?: ThreeSkeletonLike
  bindMatrix?: ThreeMatrix4Like
  bindMatrixInverse?: ThreeMatrix4Like
  morphTargetInfluences?: number[]
  morphTargetDictionary?: Record<string, number>
  color?: ThreeColorLike
  groundColor?: ThreeColorLike
  sh?: {
    coefficients?: Array<ThreeColorLike | { x?: number; y?: number; z?: number } | ArrayLike<number>>
  }
  intensity?: number
  distance?: number
  decay?: number
  angle?: number
  penumbra?: number
  width?: number
  height?: number
  target?: ThreeObject3DLike & { matrixWorld?: ThreeMatrix4Like }
  name?: string
  uuid?: string
  castShadow?: boolean
  receiveShadow?: boolean
  /** DirectionalLight.shadow (three.js). */
  shadow?: {
    mapSize?: { x?: number; y?: number; width?: number; height?: number }
    bias?: number
    normalBias?: number
    radius?: number
    blurSamples?: number
    camera?: {
      left?: number
      right?: number
      top?: number
      bottom?: number
      near?: number
      far?: number
    }
  }
}

export interface ThreeSceneRootLike extends ThreeObject3DLike {
  isScene?: boolean
  background?: ThreeColorLike | ThreeTextureLike | null
  backgroundIntensity?: number
  backgroundBlurriness?: number
  backgroundRotation?: ThreeEulerLike | ArrayLike<number> | null
  fog?: {
    isFog?: boolean
    isFogExp2?: boolean
    color?: ThreeColorLike
    near?: number
    far?: number
    density?: number
  } | null
  environment?: ThreeTextureLike | null
  environmentIntensity?: number
  environmentRotation?: ThreeEulerLike | ArrayLike<number> | null
  updateMatrixWorld?(force?: boolean): void
}

export interface ThreeSceneLike extends ThreeSceneRootLike {
  isScene: true
}

export interface ThreeCameraLike {
  isCamera: true
  layers?: ThreeLayersLike
  projectionMatrix: ThreeMatrix4Like
  matrixWorldInverse: ThreeMatrix4Like
  matrixWorld?: ThreeMatrix4Like
  isPerspectiveCamera?: boolean
  isArrayCamera?: boolean
  isCubeCamera?: boolean
  cameras?: ThreeCameraLike[]
  viewport?: RenderPixelRectLike | ({ x?: number; y?: number; z?: number; w?: number } & Record<string, unknown>) | null
  aspect?: number
  zoom?: number
  near?: number
  far?: number
  userData?: {
    width?: number
    height?: number
  }
  updateMatrixWorld?(force?: boolean): void
}

export interface ThreeCubeCameraLike {
  isObject3D?: boolean
  isCubeCamera?: boolean
  type?: string
  renderTarget?: RenderTargetLike
  children?: ThreeCameraLike[]
  coordinateSystem?: number | null
  activeMipmapLevel?: number
  updateCoordinateSystem?(): void
  updateMatrixWorld?(force?: boolean): void
}

export type ThreeRenderCameraLike = ThreeCameraLike | ThreeCubeCameraLike

export interface RenderSortItem {
  id: number
  object?: ThreeObject3DLike
  material?: ThreeMaterialLike
  groupOrder: number
  renderOrder: number
  z: number
  materialVariant: number
}

export type RenderSortFunction = (a: RenderSortItem, b: RenderSortItem) => number

export interface RenderOptions {
  width?: number
  height?: number
  background?: number[] | ThreeColorLike | ThreeTextureLike
  backgroundIntensity?: number
  backgroundBlurriness?: number
  backgroundRotation?: ThreeEulerLike | ArrayLike<number> | null
  environmentIntensity?: number
  environmentRotation?: ThreeEulerLike | ArrayLike<number> | null
  viewport?: RenderPixelRectLike | null
  scissor?: RenderPixelRectLike | null
  clippingPlanes?: ThreePlaneLike[] | null
  /**
   * Controls whether material-local clippingPlanes are included. Defaults to true
   * for the scene-oriented API; set false to mimic disabled WebGL local clipping.
   */
  localClippingEnabled?: boolean
  format?: RenderOutputFormat
  outputColorSpace?: RenderOutputColorSpace
  /** Alternate flat render passes. Defaults to normal color rendering. */
  renderMode?: RenderMode
  /** Whether render-list sorting is enabled. Defaults to true. */
  sortObjects?: boolean
  /** Custom opaque-bucket sort callback. `null` uses the default sort. */
  opaqueSort?: RenderSortFunction | null
  /** Custom transmissive/transparent-bucket sort callback. `null` uses the default sort. */
  transparentSort?: RenderSortFunction | null
  /** MSAA sample count. Supports 4x MSAA; omitted, 0, or 1 use the single-sample path. */
  samples?: number
  /** MSAA sample count. Supports 4x MSAA; omitted, 0, or 1 use the single-sample path. */
  sampleCount?: number
  target?: RenderTargetLike
  postProcessing?: PostProcessingOptions
}

export interface RenderTargetTextureLike {
  image?: RenderTargetImageLike | RenderTargetImageLike[]
  mipmaps?: Array<RenderTargetImageLike & { image?: RenderTargetImageLike | RenderTargetImageLike[] }>
  source?: {
    data?: RenderTargetImageLike | RenderTargetImageLike[]
  }
  isCubeTexture?: boolean
  format?: number
  type?: number
  needsUpdate?: boolean
  needsPMREMUpdate?: boolean
}

export interface RenderTargetImageLike {
  data?: Buffer | Uint8Array | Uint8ClampedArray | Uint16Array | Uint32Array | Float32Array
  width?: number
  height?: number
  depth?: number
}

export interface RenderTargetLike {
  width?: number
  height?: number
  viewport?: RenderPixelRectLike | null
  scissor?: RenderPixelRectLike | null
  scissorTest?: boolean
  texture?: RenderTargetTextureLike | RenderTargetTextureLike[]
  textures?: RenderTargetTextureLike[]
  objectIdEntries?: RenderObjectIdEntry[]
  objectIdMap?: Record<string, RenderObjectIdEntry>
  /** Optional normalized depth readback texture-like target. FloatType receives Float32Array data; HalfFloatType receives Uint16Array half-float data; unsigned depth types receive scalar Uint8Array/Uint16Array/Uint32Array data; UnsignedInt248Type packs depth24 in the high bits with zero stencil; plain objects receive RGBA8 bytes. */
  depthTexture?: RenderTargetTextureLike
  /** MSAA sample count. Supports 4x MSAA; omitted, 0, or 1 use the single-sample path. */
  samples?: number
  /** MSAA sample count. Supports 4x MSAA; omitted, 0, or 1 use the single-sample path. */
  sampleCount?: number
  isWebGLMultipleRenderTargets?: boolean
  image?: {
    data?: Buffer
    width?: number
    height?: number
  }
  data?: Buffer
}

export interface RenderObjectIdEntry {
  /** Adapter object sort id, usually `THREE.Object3D.id`. */
  id: number
  /** 24-bit integer encoded in RGB, with 0 reserved for background. */
  encodedId: number
  rgb: [number, number, number]
  hex: string
}

export interface PostProcessingOptions {
  enabled?: boolean
  exposure?: number
  contrast?: number
  saturation?: number
  vignette?: number
  grayscale?: number | boolean
  invert?: number | boolean
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
  viewMatrix?: number[]
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
  width?: number
  height?: number
  groundColor?: number[]
  /** Whether this light casts shadows (directional, spot, and point lights). */
  castShadow?: boolean
  /** Legacy square shadow map resolution. Defaults to 512 when width/height are absent. */
  shadowMapSize?: number
  /** Shadow map width in pixels. Defaults to `shadowMapSize` or 512. */
  shadowMapWidth?: number
  /** Shadow map height in pixels. Defaults to `shadowMapSize` or 512. */
  shadowMapHeight?: number
  /** Depth bias applied when sampling the shadow map. */
  shadowBias?: number
  /** Normal-offset bias (world space units) applied at receivers. */
  shadowNormalBias?: number
  /** PCF shadow radius multiplier. Defaults to 1. */
  shadowRadius?: number
  /** Orthographic shadow-camera frustum bounds. */
  shadowCameraLeft?: number
  shadowCameraRight?: number
  shadowCameraTop?: number
  shadowCameraBottom?: number
  shadowCameraNear?: number
  shadowCameraFar?: number
  /** Directional cascaded shadow maps: split distances from camera and flattened [left,right,top,bottom,near,far] bounds. */
  shadowCascadeSplits?: number[]
  shadowCascadeBounds?: number[]
}

export interface NativeSceneMesh {
  positions: number[]
  indices?: number[]
  normals?: number[]
  colors?: number[]
  color?: number[]
  transform?: number[]
  uvs?: number[]
  uvs2?: number[]
  texture?: Buffer
  textureWidth?: number
  textureHeight?: number
  textureWrapS?: string
  textureWrapT?: string
  textureMagFilter?: string
  textureMinFilter?: string
  textureAnisotropy?: number
  textureTransform?: number[]
  textureColorSpace?: string
  textureUsesUv2?: boolean
  specularColor?: number[]
  shininess?: number
  metallic?: number
  roughness?: number
  clearcoat?: number
  clearcoatMap?: Buffer
  clearcoatMapWidth?: number
  clearcoatMapHeight?: number
  clearcoatMapWrapS?: string
  clearcoatMapWrapT?: string
  clearcoatMapMagFilter?: string
  clearcoatMapMinFilter?: string
  clearcoatMapAnisotropy?: number
  clearcoatMapTransform?: number[]
  clearcoatMapUsesUv2?: boolean
  clearcoatRoughness?: number
  clearcoatRoughnessMap?: Buffer
  clearcoatRoughnessMapWidth?: number
  clearcoatRoughnessMapHeight?: number
  clearcoatRoughnessMapWrapS?: string
  clearcoatRoughnessMapWrapT?: string
  clearcoatRoughnessMapMagFilter?: string
  clearcoatRoughnessMapMinFilter?: string
  clearcoatRoughnessMapAnisotropy?: number
  clearcoatRoughnessMapTransform?: number[]
  clearcoatRoughnessMapUsesUv2?: boolean
  clearcoatNormalMap?: Buffer
  clearcoatNormalMapWidth?: number
  clearcoatNormalMapHeight?: number
  clearcoatNormalMapWrapS?: string
  clearcoatNormalMapWrapT?: string
  clearcoatNormalMapMagFilter?: string
  clearcoatNormalMapMinFilter?: string
  clearcoatNormalMapAnisotropy?: number
  clearcoatNormalMapTransform?: number[]
  clearcoatNormalMapUsesUv2?: boolean
  clearcoatNormalScale?: number[]
  sheenColor?: number[]
  sheenColorMap?: Buffer
  sheenColorMapWidth?: number
  sheenColorMapHeight?: number
  sheenColorMapWrapS?: string
  sheenColorMapWrapT?: string
  sheenColorMapMagFilter?: string
  sheenColorMapMinFilter?: string
  sheenColorMapAnisotropy?: number
  sheenColorMapTransform?: number[]
  sheenColorMapColorSpace?: string
  sheenColorMapUsesUv2?: boolean
  sheenRoughness?: number
  sheenRoughnessMap?: Buffer
  sheenRoughnessMapWidth?: number
  sheenRoughnessMapHeight?: number
  sheenRoughnessMapWrapS?: string
  sheenRoughnessMapWrapT?: string
  sheenRoughnessMapMagFilter?: string
  sheenRoughnessMapMinFilter?: string
  sheenRoughnessMapAnisotropy?: number
  sheenRoughnessMapTransform?: number[]
  sheenRoughnessMapUsesUv2?: boolean
  anisotropy?: number
  anisotropyRotation?: number
  anisotropyMap?: Buffer
  anisotropyMapWidth?: number
  anisotropyMapHeight?: number
  anisotropyMapWrapS?: string
  anisotropyMapWrapT?: string
  anisotropyMapMagFilter?: string
  anisotropyMapMinFilter?: string
  anisotropyMapAnisotropy?: number
  anisotropyMapTransform?: number[]
  anisotropyMapUsesUv2?: boolean
  iridescence?: number
  iridescenceMap?: Buffer
  iridescenceMapWidth?: number
  iridescenceMapHeight?: number
  iridescenceMapWrapS?: string
  iridescenceMapWrapT?: string
  iridescenceMapMagFilter?: string
  iridescenceMapMinFilter?: string
  iridescenceMapAnisotropy?: number
  iridescenceMapTransform?: number[]
  iridescenceMapUsesUv2?: boolean
  iridescenceIor?: number
  iridescenceThicknessMin?: number
  iridescenceThicknessMax?: number
  iridescenceThicknessMap?: Buffer
  iridescenceThicknessMapWidth?: number
  iridescenceThicknessMapHeight?: number
  iridescenceThicknessMapWrapS?: string
  iridescenceThicknessMapWrapT?: string
  iridescenceThicknessMapMagFilter?: string
  iridescenceThicknessMapMinFilter?: string
  iridescenceThicknessMapAnisotropy?: number
  iridescenceThicknessMapTransform?: number[]
  iridescenceThicknessMapUsesUv2?: boolean
  transmission?: number
  transmissionMap?: Buffer
  transmissionMapWidth?: number
  transmissionMapHeight?: number
  transmissionMapWrapS?: string
  transmissionMapWrapT?: string
  transmissionMapMagFilter?: string
  transmissionMapMinFilter?: string
  transmissionMapAnisotropy?: number
  transmissionMapTransform?: number[]
  transmissionMapUsesUv2?: boolean
  dispersion?: number
  ior?: number
  thickness?: number
  thicknessMap?: Buffer
  thicknessMapWidth?: number
  thicknessMapHeight?: number
  thicknessMapWrapS?: string
  thicknessMapWrapT?: string
  thicknessMapMagFilter?: string
  thicknessMapMinFilter?: string
  thicknessMapAnisotropy?: number
  thicknessMapTransform?: number[]
  thicknessMapUsesUv2?: boolean
  attenuationDistance?: number
  attenuationColor?: number[]
  physicalSpecularColor?: number[]
  physicalSpecularIntensity?: number
  specularColorMap?: Buffer
  specularColorMapWidth?: number
  specularColorMapHeight?: number
  specularColorMapWrapS?: string
  specularColorMapWrapT?: string
  specularColorMapMagFilter?: string
  specularColorMapMinFilter?: string
  specularColorMapAnisotropy?: number
  specularColorMapTransform?: number[]
  specularColorMapColorSpace?: string
  specularColorMapUsesUv2?: boolean
  specularIntensityMap?: Buffer
  specularIntensityMapWidth?: number
  specularIntensityMapHeight?: number
  specularIntensityMapWrapS?: string
  specularIntensityMapWrapT?: string
  specularIntensityMapMagFilter?: string
  specularIntensityMapMinFilter?: string
  specularIntensityMapAnisotropy?: number
  specularIntensityMapTransform?: number[]
  specularIntensityMapUsesUv2?: boolean
  emissive?: number[]
  emissiveIntensity?: number
  normalMap?: Buffer
  normalMapWidth?: number
  normalMapHeight?: number
  normalMapWrapS?: string
  normalMapWrapT?: string
  normalMapMagFilter?: string
  normalMapMinFilter?: string
  normalMapAnisotropy?: number
  normalMapTransform?: number[]
  normalMapUsesUv2?: boolean
  normalScale?: number[]
  bumpMap?: Buffer
  bumpMapWidth?: number
  bumpMapHeight?: number
  bumpMapWrapS?: string
  bumpMapWrapT?: string
  bumpMapMagFilter?: string
  bumpMapMinFilter?: string
  bumpMapAnisotropy?: number
  bumpMapTransform?: number[]
  bumpMapUsesUv2?: boolean
  bumpScale?: number
  matcapMap?: Buffer
  matcapMapWidth?: number
  matcapMapHeight?: number
  matcapMapWrapS?: string
  matcapMapWrapT?: string
  matcapMapMagFilter?: string
  matcapMapMinFilter?: string
  matcapMapAnisotropy?: number
  matcapMapTransform?: number[]
  matcapMapColorSpace?: string
  matcapMapUsesUv2?: boolean
  depthPacking?: number
  distanceReferencePosition?: number[]
  distanceNear?: number
  distanceFar?: number
  gradientMap?: Buffer
  gradientMapWidth?: number
  gradientMapHeight?: number
  gradientMapWrapS?: string
  gradientMapWrapT?: string
  gradientMapMagFilter?: string
  gradientMapMinFilter?: string
  gradientMapAnisotropy?: number
  gradientMapColorSpace?: string
  displacementMap?: Buffer
  displacementMapWidth?: number
  displacementMapHeight?: number
  displacementMapTransform?: number[]
  displacementMapUsesUv2?: boolean
  displacementScale?: number
  displacementBias?: number
  metallicRoughnessTexture?: Buffer
  metallicRoughnessTextureWidth?: number
  metallicRoughnessTextureHeight?: number
  metallicRoughnessTextureWrapS?: string
  metallicRoughnessTextureWrapT?: string
  metallicRoughnessTextureMagFilter?: string
  metallicRoughnessTextureMinFilter?: string
  metallicRoughnessTextureAnisotropy?: number
  metallicRoughnessTextureTransform?: number[]
  metallicRoughnessTextureUsesUv2?: boolean
  emissiveMap?: Buffer
  emissiveMapWidth?: number
  emissiveMapHeight?: number
  emissiveMapWrapS?: string
  emissiveMapWrapT?: string
  emissiveMapMagFilter?: string
  emissiveMapMinFilter?: string
  emissiveMapAnisotropy?: number
  emissiveMapTransform?: number[]
  emissiveMapColorSpace?: string
  emissiveMapUsesUv2?: boolean
  aoMap?: Buffer
  aoMapWidth?: number
  aoMapHeight?: number
  aoMapWrapS?: string
  aoMapWrapT?: string
  aoMapMagFilter?: string
  aoMapMinFilter?: string
  aoMapAnisotropy?: number
  aoMapTransform?: number[]
  aoMapUsesUv2?: boolean
  aoMapIntensity?: number
  lightMap?: Buffer
  lightMapWidth?: number
  lightMapHeight?: number
  lightMapWrapS?: string
  lightMapWrapT?: string
  lightMapMagFilter?: string
  lightMapMinFilter?: string
  lightMapAnisotropy?: number
  lightMapTransform?: number[]
  lightMapColorSpace?: string
  lightMapUsesUv2?: boolean
  lightMapIntensity?: number
  specularMap?: Buffer
  specularMapWidth?: number
  specularMapHeight?: number
  specularMapWrapS?: string
  specularMapWrapT?: string
  specularMapMagFilter?: string
  specularMapMinFilter?: string
  specularMapAnisotropy?: number
  specularMapTransform?: number[]
  specularMapUsesUv2?: boolean
  alphaMap?: Buffer
  alphaMapWidth?: number
  alphaMapHeight?: number
  alphaMapWrapS?: string
  alphaMapWrapT?: string
  alphaMapMagFilter?: string
  alphaMapMinFilter?: string
  alphaMapAnisotropy?: number
  alphaMapTransform?: number[]
  alphaMapUsesUv2?: boolean
  alphaTest?: number
  alphaHash?: boolean
  alphaToCoverage?: boolean
  premultipliedAlpha?: boolean
  flatShading?: boolean
  fog?: boolean
  /** Flattened world-space clipping planes `[nx, ny, nz, constant, ...]`, up to 8 planes. */
  clippingPlanes?: number[]
  /** Number of leading clipping planes evaluated as union planes; remaining planes use intersection semantics. */
  clippingUnionCount?: number
  /** Whether material clipping planes affect this mesh in the shadow pass. */
  clipShadows?: boolean
  blending?: string
  blendEquation?: number
  blendSrc?: number
  blendDst?: number
  blendEquationAlpha?: number
  blendSrcAlpha?: number
  blendDstAlpha?: number
  blendColor?: number[]
  blendAlpha?: number
  depthTest?: boolean
  depthFunc?: string
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
  side?: string
  /** Optional shadow-pass face filter from material.shadowSide. Unset preserves the renderer's default no-cull caster behavior. */
  shadowSide?: string
  shadingModel?: string
  useEnvironmentMap?: boolean
  environmentMapIntensity?: number
  environmentMapCombine?: number
  environmentMapReflectivity?: number
  environmentMapRefraction?: boolean
  environmentMapRefractionRatio?: number
  topology?: string
  /** WGSL fragment body used by the custom material path. */
  customFragmentShader?: string
  /** Whether this mesh casts shadows in the shadow pass. Defaults to false. */
  castShadow?: boolean
  /** Whether this mesh receives shadows in the main pass. Defaults to false. */
  receiveShadow?: boolean
  groupOrder?: number
  renderOrder?: number
  sortZ?: number
  sortIndex?: number
  materialVariant?: number
  materialSortKey?: number
}

export interface NativeRenderScene {
  width?: number
  height?: number
  background?: number[]
  backgroundIntensity?: number
  viewport?: number[]
  scissor?: number[]
  backgroundTexture?: Buffer
  backgroundTextureWidth?: number
  backgroundTextureHeight?: number
  backgroundTextureWrapS?: string
  backgroundTextureWrapT?: string
  backgroundTextureMagFilter?: string
  backgroundTextureMinFilter?: string
  backgroundTextureAnisotropy?: number
  backgroundTextureTransform?: number[]
  backgroundTextureColorSpace?: string
  backgroundTextureMapping?: string
  backgroundTextureRotation?: number[]
  backgroundTextureBlurriness?: number
  format?: string
  outputColorSpace?: string
  sampleCount?: number
  meshes?: NativeSceneMesh[]
  lights?: NativeSceneLight[]
  ambientLight?: number[]
  ambientIntensity?: number
  lightProbe?: number[]
  environmentMap?: Buffer
  environmentMapWidth?: number
  environmentMapHeight?: number
  environmentMapIntensity?: number
  environmentMapColorSpace?: string
  environmentMapRotation?: number[]
  fogType?: string
  fogColor?: number[]
  fogNear?: number
  fogFar?: number
  fogDensity?: number
  postExposure?: number
  postContrast?: number
  postSaturation?: number
  postVignette?: number
  postGrayscale?: number
  postInvert?: number
}

// ── Internal helper types ───────────────────────────────────────────

export type Color4 = [number, number, number, number]
export type Mat4 = number[]
export type Vec3 = [number, number, number]

export interface PbrProperties {
  metallic?: number
  roughness?: number
  clearcoat?: number
  clearcoatMap?: Buffer
  clearcoatMapWidth?: number
  clearcoatMapHeight?: number
  clearcoatMapWrapS?: string
  clearcoatMapWrapT?: string
  clearcoatMapMagFilter?: string
  clearcoatMapMinFilter?: string
  clearcoatMapAnisotropy?: number
  clearcoatMapTransform?: number[]
  clearcoatMapUsesUv2?: boolean
  clearcoatRoughness?: number
  clearcoatRoughnessMap?: Buffer
  clearcoatRoughnessMapWidth?: number
  clearcoatRoughnessMapHeight?: number
  clearcoatRoughnessMapWrapS?: string
  clearcoatRoughnessMapWrapT?: string
  clearcoatRoughnessMapMagFilter?: string
  clearcoatRoughnessMapMinFilter?: string
  clearcoatRoughnessMapAnisotropy?: number
  clearcoatRoughnessMapTransform?: number[]
  clearcoatRoughnessMapUsesUv2?: boolean
  clearcoatNormalMap?: Buffer
  clearcoatNormalMapWidth?: number
  clearcoatNormalMapHeight?: number
  clearcoatNormalMapWrapS?: string
  clearcoatNormalMapWrapT?: string
  clearcoatNormalMapMagFilter?: string
  clearcoatNormalMapMinFilter?: string
  clearcoatNormalMapAnisotropy?: number
  clearcoatNormalMapTransform?: number[]
  clearcoatNormalMapUsesUv2?: boolean
  clearcoatNormalScale?: number[]
  sheenColor?: number[]
  sheenColorMap?: Buffer
  sheenColorMapWidth?: number
  sheenColorMapHeight?: number
  sheenColorMapWrapS?: string
  sheenColorMapWrapT?: string
  sheenColorMapMagFilter?: string
  sheenColorMapMinFilter?: string
  sheenColorMapAnisotropy?: number
  sheenColorMapTransform?: number[]
  sheenColorMapColorSpace?: string
  sheenColorMapUsesUv2?: boolean
  sheenRoughness?: number
  sheenRoughnessMap?: Buffer
  sheenRoughnessMapWidth?: number
  sheenRoughnessMapHeight?: number
  sheenRoughnessMapWrapS?: string
  sheenRoughnessMapWrapT?: string
  sheenRoughnessMapMagFilter?: string
  sheenRoughnessMapMinFilter?: string
  sheenRoughnessMapAnisotropy?: number
  sheenRoughnessMapTransform?: number[]
  sheenRoughnessMapUsesUv2?: boolean
  anisotropy?: number
  anisotropyRotation?: number
  anisotropyMap?: Buffer
  anisotropyMapWidth?: number
  anisotropyMapHeight?: number
  anisotropyMapWrapS?: string
  anisotropyMapWrapT?: string
  anisotropyMapMagFilter?: string
  anisotropyMapMinFilter?: string
  anisotropyMapAnisotropy?: number
  anisotropyMapTransform?: number[]
  anisotropyMapUsesUv2?: boolean
  iridescence?: number
  iridescenceMap?: Buffer
  iridescenceMapWidth?: number
  iridescenceMapHeight?: number
  iridescenceMapWrapS?: string
  iridescenceMapWrapT?: string
  iridescenceMapMagFilter?: string
  iridescenceMapMinFilter?: string
  iridescenceMapAnisotropy?: number
  iridescenceMapTransform?: number[]
  iridescenceMapUsesUv2?: boolean
  iridescenceIor?: number
  iridescenceThicknessMin?: number
  iridescenceThicknessMax?: number
  iridescenceThicknessMap?: Buffer
  iridescenceThicknessMapWidth?: number
  iridescenceThicknessMapHeight?: number
  iridescenceThicknessMapWrapS?: string
  iridescenceThicknessMapWrapT?: string
  iridescenceThicknessMapMagFilter?: string
  iridescenceThicknessMapMinFilter?: string
  iridescenceThicknessMapAnisotropy?: number
  iridescenceThicknessMapTransform?: number[]
  iridescenceThicknessMapUsesUv2?: boolean
  transmission?: number
  transmissionMap?: Buffer
  transmissionMapWidth?: number
  transmissionMapHeight?: number
  transmissionMapWrapS?: string
  transmissionMapWrapT?: string
  transmissionMapMagFilter?: string
  transmissionMapMinFilter?: string
  transmissionMapAnisotropy?: number
  transmissionMapTransform?: number[]
  transmissionMapUsesUv2?: boolean
  dispersion?: number
  ior?: number
  thickness?: number
  thicknessMap?: Buffer
  thicknessMapWidth?: number
  thicknessMapHeight?: number
  thicknessMapWrapS?: string
  thicknessMapWrapT?: string
  thicknessMapMagFilter?: string
  thicknessMapMinFilter?: string
  thicknessMapAnisotropy?: number
  thicknessMapTransform?: number[]
  thicknessMapUsesUv2?: boolean
  attenuationDistance?: number
  attenuationColor?: number[]
  physicalSpecularColor?: number[]
  physicalSpecularIntensity?: number
  useEnvironmentMap?: boolean
  environmentMapIntensity?: number
  environmentMapCombine?: number
  environmentMapReflectivity?: number
  environmentMapRefraction?: boolean
  environmentMapRefractionRatio?: number
  specularColorMap?: Buffer
  specularColorMapWidth?: number
  specularColorMapHeight?: number
  specularColorMapWrapS?: string
  specularColorMapWrapT?: string
  specularColorMapMagFilter?: string
  specularColorMapMinFilter?: string
  specularColorMapAnisotropy?: number
  specularColorMapTransform?: number[]
  specularColorMapColorSpace?: string
  specularColorMapUsesUv2?: boolean
  specularIntensityMap?: Buffer
  specularIntensityMapWidth?: number
  specularIntensityMapHeight?: number
  specularIntensityMapWrapS?: string
  specularIntensityMapWrapT?: string
  specularIntensityMapMagFilter?: string
  specularIntensityMapMinFilter?: string
  specularIntensityMapAnisotropy?: number
  specularIntensityMapTransform?: number[]
  specularIntensityMapUsesUv2?: boolean
  emissive?: number[]
  emissiveIntensity?: number
  specularColor?: number[]
  shininess?: number
  normalMap?: Buffer
  normalMapWidth?: number
  normalMapHeight?: number
  normalMapWrapS?: string
  normalMapWrapT?: string
  normalMapMagFilter?: string
  normalMapMinFilter?: string
  normalMapAnisotropy?: number
  normalMapTransform?: number[]
  normalMapUsesUv2?: boolean
  normalScale?: number[]
  bumpMap?: Buffer
  bumpMapWidth?: number
  bumpMapHeight?: number
  bumpMapWrapS?: string
  bumpMapWrapT?: string
  bumpMapMagFilter?: string
  bumpMapMinFilter?: string
  bumpMapAnisotropy?: number
  bumpMapTransform?: number[]
  bumpMapUsesUv2?: boolean
  bumpScale?: number
  matcapMap?: Buffer
  matcapMapWidth?: number
  matcapMapHeight?: number
  matcapMapWrapS?: string
  matcapMapWrapT?: string
  matcapMapMagFilter?: string
  matcapMapMinFilter?: string
  matcapMapAnisotropy?: number
  matcapMapTransform?: number[]
  matcapMapColorSpace?: string
  matcapMapUsesUv2?: boolean
  depthPacking?: number
  distanceReferencePosition?: number[]
  distanceNear?: number
  distanceFar?: number
  gradientMap?: Buffer
  gradientMapWidth?: number
  gradientMapHeight?: number
  gradientMapWrapS?: string
  gradientMapWrapT?: string
  gradientMapMagFilter?: string
  gradientMapMinFilter?: string
  gradientMapAnisotropy?: number
  gradientMapColorSpace?: string
  displacementMap?: Buffer
  displacementMapWidth?: number
  displacementMapHeight?: number
  displacementMapTransform?: number[]
  displacementMapUsesUv2?: boolean
  displacementScale?: number
  displacementBias?: number
  metallicRoughnessTexture?: Buffer
  metallicRoughnessTextureWidth?: number
  metallicRoughnessTextureHeight?: number
  metallicRoughnessTextureWrapS?: string
  metallicRoughnessTextureWrapT?: string
  metallicRoughnessTextureMagFilter?: string
  metallicRoughnessTextureMinFilter?: string
  metallicRoughnessTextureAnisotropy?: number
  metallicRoughnessTextureTransform?: number[]
  metallicRoughnessTextureUsesUv2?: boolean
  emissiveMap?: Buffer
  emissiveMapWidth?: number
  emissiveMapHeight?: number
  emissiveMapWrapS?: string
  emissiveMapWrapT?: string
  emissiveMapMagFilter?: string
  emissiveMapMinFilter?: string
  emissiveMapAnisotropy?: number
  emissiveMapTransform?: number[]
  emissiveMapColorSpace?: string
  emissiveMapUsesUv2?: boolean
  aoMap?: Buffer
  aoMapWidth?: number
  aoMapHeight?: number
  aoMapWrapS?: string
  aoMapWrapT?: string
  aoMapMagFilter?: string
  aoMapMinFilter?: string
  aoMapAnisotropy?: number
  aoMapTransform?: number[]
  aoMapUsesUv2?: boolean
  aoMapIntensity?: number
  lightMap?: Buffer
  lightMapWidth?: number
  lightMapHeight?: number
  lightMapWrapS?: string
  lightMapWrapT?: string
  lightMapMagFilter?: string
  lightMapMinFilter?: string
  lightMapAnisotropy?: number
  lightMapTransform?: number[]
  lightMapColorSpace?: string
  lightMapUsesUv2?: boolean
  lightMapIntensity?: number
  specularMap?: Buffer
  specularMapWidth?: number
  specularMapHeight?: number
  specularMapWrapS?: string
  specularMapWrapT?: string
  specularMapMagFilter?: string
  specularMapMinFilter?: string
  specularMapAnisotropy?: number
  specularMapTransform?: number[]
  specularMapUsesUv2?: boolean
  alphaMap?: Buffer
  alphaMapWidth?: number
  alphaMapHeight?: number
  alphaMapWrapS?: string
  alphaMapWrapT?: string
  alphaMapMagFilter?: string
  alphaMapMinFilter?: string
  alphaMapAnisotropy?: number
  alphaMapTransform?: number[]
  alphaMapUsesUv2?: boolean
  alphaTest?: number
  alphaHash?: boolean
  alphaToCoverage?: boolean
  premultipliedAlpha?: boolean
  flatShading?: boolean
  fog?: boolean
  transparent?: boolean
  blending?: string
  blendEquation?: number
  blendSrc?: number
  blendDst?: number
  blendEquationAlpha?: number
  blendSrcAlpha?: number
  blendDstAlpha?: number
  blendColor?: number[]
  blendAlpha?: number
  depthTest?: boolean
  depthFunc?: string
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
  side?: string
  shadowSide?: string
  shadingModel?: string
  customFragmentShader?: string
}

export interface TextureInfo {
  data: Buffer
  width: number
  height: number
  wrapS?: string
  wrapT?: string
  magFilter?: string
  minFilter?: string
  anisotropy?: number
  transform?: number[]
  colorSpace?: string
  mapping?: 'uv' | 'equirectangular'
  usesUv2?: boolean
}

export interface GeometryGroup {
  start: number
  count: number
  materialIndex: number
}
