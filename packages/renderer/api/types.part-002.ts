import { RenderMode, RenderOutputColorSpace, RenderOutputFormat, RenderPixelRectLike, ThreeBufferAttributeLike, ThreeBufferGeometryLike, ThreeColorInput, ThreeColorLike, ThreeEulerLike, ThreeLayersLike, ThreeMaterialLike, ThreeMatrix4Like, ThreePlaneLike, ThreeSkeletonLike, ThreeSphereLike, ThreeTextureLike } from './types.part-001'
import { GeometryGroup } from './types.part-004'
export interface ThreeObject3DLike {
  isObject3D?: boolean
  visible?: boolean
  layers?: ThreeLayersLike
  renderOrder?: number
  id?: number
  children?: ThreeObject3DLike[]
  userData?: Record<string, any>
  isMesh?: boolean
  isBatchedMesh?: boolean
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
  frustumCulled?: boolean
  boundingSphere?: ThreeSphereLike | null
  computeBoundingSphere?(): void
  geometry?: ThreeBufferGeometryLike
  material?: ThreeMaterialLike | ThreeMaterialLike[]
  onBeforeRender?: (...args: unknown[]) => unknown
  onAfterRender?: (...args: unknown[]) => unknown
  clippingPlanes?: ThreePlaneLike[] | null
  enabled?: boolean
  clipIntersection?: boolean
  clipShadows?: boolean
  customDepthMaterial?: ThreeMaterialLike
  customDistanceMaterial?: ThreeMaterialLike
  center?: { x?: number; y?: number }
  scale?: { x?: number; y?: number; z?: number }
  count?: number
  instanceCount?: number
  perObjectFrustumCulled?: boolean
  sortObjects?: boolean
  customSort?: ((items: Array<{ start: number; count: number; z: number; index: number }>, camera?: ThreeCameraLike) => void) | null
  maxInstanceCount?: number
  instanceMatrix?: ThreeBufferAttributeLike
  instanceColor?: ThreeBufferAttributeLike | null
  _instanceInfo?: Array<{
    visible?: boolean
    active?: boolean
    geometryIndex?: number
  }>
  _geometryInfo?: Array<{
    active?: boolean
    start?: number
    count?: number
    vertexStart?: number
    vertexCount?: number
    reservedVertexCount?: number
    indexStart?: number
    indexCount?: number
    reservedIndexCount?: number
    boundingSphere?: ThreeSphereLike | null
  }>
  _matricesTexture?: {
    image?: {
      data?: ArrayLike<number>
    }
  } | null
  _colorsTexture?: {
    image?: {
      data?: ArrayLike<number>
    }
  } | null
  getGeometryRangeAt?(geometryId: number, target?: Record<string, unknown>): Record<string, unknown> | null
  matrixWorld?: ThreeMatrix4Like
  autoUpdate?: boolean
  levels?: Array<{ object: ThreeObject3DLike; distance?: number; hysteresis?: number }>
  update?(camera: ThreeCameraLike): void
  skeleton?: ThreeSkeletonLike
  bindMatrix?: ThreeMatrix4Like
  bindMatrixInverse?: ThreeMatrix4Like
  morphTargetInfluences?: number[]
  morphTargetDictionary?: Record<string, number>
  color?: ThreeColorInput
  groundColor?: ThreeColorInput
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
    map?: object | null
    mapPass?: object | null
    matrix?: ThreeMatrix4Like
    autoUpdate?: boolean
    needsUpdate?: boolean
    bias?: number
    normalBias?: number
    radius?: number
    intensity?: number
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
  overrideMaterial?: ThreeMaterialLike | null
  background?: ThreeColorInput | ThreeTextureLike | null
  backgroundIntensity?: number
  backgroundBlurriness?: number
  backgroundRotation?: ThreeEulerLike | ArrayLike<number> | null
  fog?: {
    isFog?: boolean
    isFogExp2?: boolean
    color?: ThreeColorInput
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
  isOrthographicCamera?: boolean
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
  geometry?: ThreeBufferGeometryLike
  material?: ThreeMaterialLike
  group?: GeometryGroup
  groupOrder: number
  renderOrder: number
  z: number
  materialVariant: number
}

export type RenderSortFunction = (a: RenderSortItem, b: RenderSortItem) => number

export interface RenderOptions {
  width?: number
  height?: number
  background?: ThreeColorInput | ThreeTextureLike | null
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
  /** Per-render tone-mapping mode. Defaults to Renderer.toneMapping. */
  toneMapping?: number
  /** Per-render tone-mapping exposure. Defaults to Renderer.toneMappingExposure. */
  toneMappingExposure?: number
  /** Alternate flat render passes. Defaults to normal color rendering. */
  renderMode?: RenderMode
  /** Whether render-list sorting is enabled. Defaults to true. */
  sortObjects?: boolean
  /** Whether opaque render-list buckets are rendered. Defaults to Renderer.opaque. */
  opaque?: boolean
  /** Whether transmissive/transparent render-list buckets are rendered. Defaults to Renderer.transparent. */
  transparent?: boolean
  /** Per-render transmission scene-color resolution scale. Defaults to Renderer.transmissionResolutionScale. */
  transmissionResolutionScale?: number
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
  userData?: Record<string, any>
  isCubeTexture?: boolean
  isFramebufferTexture?: boolean
  isDepthTexture?: boolean
  isStorageTexture?: boolean
  isCompressedTexture?: boolean
  isDataArrayTexture?: boolean
  isData3DTexture?: boolean
  isArrayTexture?: boolean
  is3DTexture?: boolean
  format?: number
  /** Color targets support Alpha/Red/RG/RGB/RGBA and integer channel formats with byte, signed/unsigned integer, packed color, FloatType, and HalfFloatType arrays. */
  type?: number
  generateMipmaps?: boolean
  needsUpdate?: boolean
  needsPMREMUpdate?: boolean
  pmremVersion?: number
}

export interface RenderTargetImageLike {
  data?: Buffer | Int8Array | Uint8Array | Uint8ClampedArray | Int16Array | Uint16Array | Int32Array | Uint32Array | Float32Array
  width?: number
  height?: number
  depth?: number
}

export interface RenderTargetLike {
  isRenderTarget?: boolean
  isWebGLRenderTarget?: boolean
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
  isWebGLCubeRenderTarget?: boolean
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
  /** Shadow darkness multiplier. Defaults to 1; 0 disables received shadow darkening. */
  shadowIntensity?: number
  /** VSM blur sample count. Defaults to Three.js' LightShadow default of 8. */
  shadowBlurSamples?: number
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
