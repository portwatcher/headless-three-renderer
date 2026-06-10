/// <reference types="node" />

// ── Three.js duck-typed interfaces ──────────────────────────────────

export type RenderOutputFormat = 'png' | 'rgba' | 'raw' | 'raw-rgba'
export type RenderOutputColorSpace = 'srgb' | 'srgb-linear' | 'linear-srgb' | 'linear'

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
  magFilter?: number
  minFilter?: number
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

export type ThreePlaneLike = {
  normal?: ThreeVector3Like | ArrayLike<number>
  constant?: number
} | ArrayLike<number>

export type RenderPixelRectLike = {
  x?: number
  y?: number
  width?: number
  height?: number
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

export interface ThreeSceneLike extends ThreeObject3DLike {
  isScene: true
  background?: ThreeColorLike | ThreeTextureLike | null
  backgroundIntensity?: number
  backgroundBlurriness?: number
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
  updateMatrixWorld?(force?: boolean): void
}

export interface ThreeCameraLike {
  isCamera: true
  layers?: ThreeLayersLike
  projectionMatrix: ThreeMatrix4Like
  matrixWorldInverse: ThreeMatrix4Like
  matrixWorld?: ThreeMatrix4Like
  isPerspectiveCamera?: boolean
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

export interface RenderOptions {
  width?: number
  height?: number
  background?: number[] | ThreeColorLike | ThreeTextureLike
  backgroundIntensity?: number
  backgroundBlurriness?: number
  viewport?: RenderPixelRectLike | null
  scissor?: RenderPixelRectLike | null
  clippingPlanes?: ThreePlaneLike[] | null
  format?: RenderOutputFormat
  outputColorSpace?: RenderOutputColorSpace
  target?: RenderTargetLike
  postProcessing?: PostProcessingOptions
}

export interface RenderTargetLike {
  width?: number
  height?: number
  texture?: {
    image?: {
      data?: Buffer
      width?: number
      height?: number
    }
    source?: {
      data?: {
        data?: Buffer
        width?: number
        height?: number
      }
    }
  }
  image?: {
    data?: Buffer
    width?: number
    height?: number
  }
  data?: Buffer
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
  /** Shadow map resolution (square). Defaults to 512. */
  shadowMapSize?: number
  /** Depth bias applied when sampling the shadow map. */
  shadowBias?: number
  /** Normal-offset bias (world space units) applied at receivers. */
  shadowNormalBias?: number
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
  clearcoatRoughnessMapTransform?: number[]
  clearcoatRoughnessMapUsesUv2?: boolean
  clearcoatNormalMap?: Buffer
  clearcoatNormalMapWidth?: number
  clearcoatNormalMapHeight?: number
  clearcoatNormalMapWrapS?: string
  clearcoatNormalMapWrapT?: string
  clearcoatNormalMapMagFilter?: string
  clearcoatNormalMapMinFilter?: string
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
  anisotropyMapTransform?: number[]
  anisotropyMapUsesUv2?: boolean
  transmission?: number
  transmissionMap?: Buffer
  transmissionMapWidth?: number
  transmissionMapHeight?: number
  transmissionMapWrapS?: string
  transmissionMapWrapT?: string
  transmissionMapMagFilter?: string
  transmissionMapMinFilter?: string
  transmissionMapTransform?: number[]
  transmissionMapUsesUv2?: boolean
  ior?: number
  thickness?: number
  thicknessMap?: Buffer
  thicknessMapWidth?: number
  thicknessMapHeight?: number
  thicknessMapWrapS?: string
  thicknessMapWrapT?: string
  thicknessMapMagFilter?: string
  thicknessMapMinFilter?: string
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
  bumpMapTransform?: number[]
  bumpMapUsesUv2?: boolean
  bumpScale?: number
  matcapMap?: Buffer
  matcapMapWidth?: number
  matcapMapHeight?: number
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
  metallicRoughnessTextureTransform?: number[]
  metallicRoughnessTextureUsesUv2?: boolean
  emissiveMap?: Buffer
  emissiveMapWidth?: number
  emissiveMapHeight?: number
  emissiveMapWrapS?: string
  emissiveMapWrapT?: string
  emissiveMapMagFilter?: string
  emissiveMapMinFilter?: string
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
  aoMapTransform?: number[]
  aoMapIntensity?: number
  lightMap?: Buffer
  lightMapWidth?: number
  lightMapHeight?: number
  lightMapWrapS?: string
  lightMapWrapT?: string
  lightMapMagFilter?: string
  lightMapMinFilter?: string
  lightMapTransform?: number[]
  lightMapColorSpace?: string
  lightMapIntensity?: number
  specularMap?: Buffer
  specularMapWidth?: number
  specularMapHeight?: number
  specularMapWrapS?: string
  specularMapWrapT?: string
  specularMapMagFilter?: string
  specularMapMinFilter?: string
  specularMapTransform?: number[]
  alphaMap?: Buffer
  alphaMapWidth?: number
  alphaMapHeight?: number
  alphaMapWrapS?: string
  alphaMapWrapT?: string
  alphaMapMagFilter?: string
  alphaMapMinFilter?: string
  alphaMapTransform?: number[]
  alphaMapUsesUv2?: boolean
  alphaTest?: number
  alphaHash?: boolean
  premultipliedAlpha?: boolean
  flatShading?: boolean
  fog?: boolean
  /** Flattened world-space clipping planes `[nx, ny, nz, constant, ...]`, up to 8 planes. */
  clippingPlanes?: number[]
  /** Number of leading clipping planes evaluated as union planes; remaining planes use intersection semantics. */
  clippingUnionCount?: number
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
  shadingModel?: string
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
  backgroundTextureTransform?: number[]
  backgroundTextureColorSpace?: string
  backgroundTextureBlurriness?: number
  format?: string
  outputColorSpace?: string
  meshes?: NativeSceneMesh[]
  lights?: NativeSceneLight[]
  ambientLight?: number[]
  ambientIntensity?: number
  lightProbe?: number[]
  environmentMap?: Buffer
  environmentMapWidth?: number
  environmentMapHeight?: number
  environmentMapIntensity?: number
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
  clearcoatRoughnessMapTransform?: number[]
  clearcoatRoughnessMapUsesUv2?: boolean
  clearcoatNormalMap?: Buffer
  clearcoatNormalMapWidth?: number
  clearcoatNormalMapHeight?: number
  clearcoatNormalMapWrapS?: string
  clearcoatNormalMapWrapT?: string
  clearcoatNormalMapMagFilter?: string
  clearcoatNormalMapMinFilter?: string
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
  anisotropyMapTransform?: number[]
  anisotropyMapUsesUv2?: boolean
  transmission?: number
  transmissionMap?: Buffer
  transmissionMapWidth?: number
  transmissionMapHeight?: number
  transmissionMapWrapS?: string
  transmissionMapWrapT?: string
  transmissionMapMagFilter?: string
  transmissionMapMinFilter?: string
  transmissionMapTransform?: number[]
  transmissionMapUsesUv2?: boolean
  ior?: number
  thickness?: number
  thicknessMap?: Buffer
  thicknessMapWidth?: number
  thicknessMapHeight?: number
  thicknessMapWrapS?: string
  thicknessMapWrapT?: string
  thicknessMapMagFilter?: string
  thicknessMapMinFilter?: string
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
  bumpMapTransform?: number[]
  bumpMapUsesUv2?: boolean
  bumpScale?: number
  matcapMap?: Buffer
  matcapMapWidth?: number
  matcapMapHeight?: number
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
  metallicRoughnessTextureTransform?: number[]
  metallicRoughnessTextureUsesUv2?: boolean
  emissiveMap?: Buffer
  emissiveMapWidth?: number
  emissiveMapHeight?: number
  emissiveMapWrapS?: string
  emissiveMapWrapT?: string
  emissiveMapMagFilter?: string
  emissiveMapMinFilter?: string
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
  aoMapTransform?: number[]
  aoMapIntensity?: number
  lightMap?: Buffer
  lightMapWidth?: number
  lightMapHeight?: number
  lightMapWrapS?: string
  lightMapWrapT?: string
  lightMapMagFilter?: string
  lightMapMinFilter?: string
  lightMapTransform?: number[]
  lightMapColorSpace?: string
  lightMapIntensity?: number
  specularMap?: Buffer
  specularMapWidth?: number
  specularMapHeight?: number
  specularMapWrapS?: string
  specularMapWrapT?: string
  specularMapMagFilter?: string
  specularMapMinFilter?: string
  specularMapTransform?: number[]
  alphaMap?: Buffer
  alphaMapWidth?: number
  alphaMapHeight?: number
  alphaMapWrapS?: string
  alphaMapWrapT?: string
  alphaMapMagFilter?: string
  alphaMapMinFilter?: string
  alphaMapTransform?: number[]
  alphaMapUsesUv2?: boolean
  alphaTest?: number
  alphaHash?: boolean
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
  transform?: number[]
  colorSpace?: string
  usesUv2?: boolean
}

export interface GeometryGroup {
  start: number
  count: number
  materialIndex: number
}
