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
import { DefaultRendererContextAttributes, SupportedRendererShadowMapTypes, SupportedRendererToneMappings } from './index.part-001'
import { PixelRect, PixelSize, UnsignedByteType } from './index.part-012'
import { rendererStateBoolean, rendererStatePositiveFiniteNumber } from './index.part-014'
import { assertRenderTargetLike } from './index.part-016'
import { assertEulerOption, assertSupportedOutputColorSpace, assertSupportedOutputFormat, assertSupportedSampleCount, validateSortControls, validateUnsupportedRenderTargetOptions } from './index.part-017'
import { assertFiniteNumberOption, assertNonNegativeNumberOption, assertNormalizedNumberOption, validatePostProcessingOptions } from './index.part-018'
export function assertRendererOutputBufferType(value: unknown, label: string): void {
  if (value === undefined) return
  if (typeof value !== 'number' || !Number.isInteger(value)) {
    throw new TypeError(`${label} must be a Three.js texture type integer.`)
  }
  if (value !== UnsignedByteType) {
    throw new Error(
      `${label} ${String(value)} is not supported by @headless-three/renderer because it has no browser drawing buffer. Omit outputBufferType for RGBA8 output, or use a target texture with FloatType or HalfFloatType for typed offscreen readback.`,
    )
  }
}

export function assertRendererConstructorSamples(value: unknown, label: string): void {
  if (value == null) return
  if (typeof value !== 'number' || !Number.isFinite(value) || !Number.isInteger(value) || value < 0) {
    throw new TypeError(`${label} must be a non-negative integer sample count.`)
  }
  if (value > 1) {
    throw new Error(
      `${label} ${String(value)} is not supported as constructor-level MSAA state by @headless-three/renderer. Use render options samples/sampleCount or target samples/sampleCount for 4x MSAA.`,
    )
  }
}

export function rendererContextAttributes(parameters?: RendererParametersLike): RendererContextAttributesLike {
  return {
    alpha: parameters?.alpha ?? DefaultRendererContextAttributes.alpha,
    depth: parameters?.depth ?? DefaultRendererContextAttributes.depth,
    stencil: parameters?.stencil ?? DefaultRendererContextAttributes.stencil,
    antialias: parameters?.antialias ?? DefaultRendererContextAttributes.antialias,
    premultipliedAlpha: parameters?.premultipliedAlpha ?? DefaultRendererContextAttributes.premultipliedAlpha,
    preserveDrawingBuffer: parameters?.preserveDrawingBuffer ?? DefaultRendererContextAttributes.preserveDrawingBuffer,
    powerPreference: parameters?.powerPreference ?? DefaultRendererContextAttributes.powerPreference,
    failIfMajorPerformanceCaveat: parameters?.failIfMajorPerformanceCaveat
      ?? DefaultRendererContextAttributes.failIfMajorPerformanceCaveat,
  }
}

export function assertRendererContextParameterAbsent(
  parameters: Record<string, unknown>,
  name: 'canvas' | 'context',
  label: string,
): void {
  if (parameters[name] === undefined || parameters[name] === null) return
  throw new Error(
    `${label}.${name} is not supported by @headless-three/renderer because it renders offscreen through wgpu instead of a browser WebGL context.`,
  )
}

export function assertRendererUnsupportedDepthParameterFalse(
  parameters: Record<string, unknown>,
  name: 'logarithmicDepthBuffer' | 'reversedDepthBuffer' | 'reverseDepthBuffer',
  label: string,
): void {
  if (parameters[name] === undefined) return
  const enabled = rendererStateBoolean(parameters[name], `${label}.${name}`)
  if (!enabled) return
  throw new Error(`${label}.${name} true is not supported by @headless-three/renderer yet.`)
}

export function rendererStateShadowMapType(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`Renderer.shadowMap.type must be a Three.js shadow map type constant; received ${String(value)}.`)
  }
  if (!Number.isInteger(value) || !SupportedRendererShadowMapTypes.has(value)) {
    throw new TypeError(
      `Renderer.shadowMap.type ${String(value)} is not supported by @headless-three/renderer. Use THREE.BasicShadowMap, THREE.PCFShadowMap, THREE.PCFSoftShadowMap, or THREE.VSMShadowMap.`,
    )
  }
  return value
}

export function rendererStateToneMapping(value: unknown, label = 'Renderer.toneMapping'): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a Three.js tone mapping constant; received ${String(value)}.`)
  }
  if (!Number.isInteger(value) || !SupportedRendererToneMappings.has(value)) {
    throw new TypeError(
      `${label} ${String(value)} is not supported by @headless-three/renderer yet. Use THREE.NoToneMapping, THREE.LinearToneMapping, THREE.ReinhardToneMapping, THREE.CineonToneMapping, THREE.ACESFilmicToneMapping, THREE.CustomToneMapping, THREE.AgXToneMapping, or THREE.NeutralToneMapping.`,
    )
  }
  return value
}

export function assertOptionalBoolean(value: unknown, label: string): void {
  rendererStateBoolean(value, label)
}

export function assertEffectsArrayOrNull(value: unknown, label: string): asserts value is readonly unknown[] | null {
  if (value !== null && !Array.isArray(value)) {
    throw new TypeError(`${label} must be an array or null.`)
  }
}

export function clonePixelSize(size: PixelSize | null | undefined): PixelSize | null
export function clonePixelSize<T extends RenderSizeLike>(size: PixelSize | null | undefined, target: T): T | null
export function clonePixelSize<T extends RenderSizeLike>(
  size: PixelSize | null | undefined,
  target?: T,
): PixelSize | T | null {
  if (!size) return null
  if (target) {
    const mutable = target as any
    if (typeof mutable.length === 'number') {
      mutable[0] = size.width
      mutable[1] = size.height
    } else {
      if (typeof mutable.set === 'function') mutable.set(size.width, size.height)
      if ('width' in mutable || 'height' in mutable || typeof mutable.set !== 'function') {
        mutable.width = size.width
        mutable.height = size.height
      }
      if ('x' in mutable || 'y' in mutable || typeof mutable.set === 'function') {
        mutable.x = size.width
        mutable.y = size.height
      }
    }
    return target
  }
  return { width: size.width, height: size.height }
}

export function clonePixelRect(rect: PixelRect | null | undefined): PixelRect | null
export function clonePixelRect<T extends RenderPixelRectLike>(rect: PixelRect | null | undefined, target: T): T | null
export function clonePixelRect<T extends RenderPixelRectLike>(
  rect: PixelRect | null | undefined,
  target?: T,
): PixelRect | T | null {
  if (!rect) return null
  if (target) {
    const mutable = target as any
    if (typeof mutable.length === 'number') {
      mutable[0] = rect.x
      mutable[1] = rect.y
      mutable[2] = rect.width
      mutable[3] = rect.height
    } else {
      if (typeof mutable.set === 'function') mutable.set(rect.x, rect.y, rect.width, rect.height)
      mutable.x = rect.x
      mutable.y = rect.y
      mutable.width = rect.width
      mutable.height = rect.height
      mutable.z = rect.width
      mutable.w = rect.height
    }
    return target
  }
  return { x: rect.x, y: rect.y, width: rect.width, height: rect.height }
}

export function finiteOrUndefined(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

export function optionalFiniteNumber(value: unknown, label: string): number | undefined {
  if (value == null) return undefined
  if (typeof value === 'number' && Number.isFinite(value)) return value
  throw new TypeError(`${label} must be a finite number.`)
}

export function optionalNonNegativeFiniteNumber(value: unknown, label: string): number | undefined {
  const number = optionalFiniteNumber(value, label)
  if (number === undefined) return undefined
  if (number < 0) {
    throw new TypeError(`${label} must be non-negative.`)
  }
  return number
}

export function finiteNonNegativeNumber(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a finite number.`)
  }
  if (value < 0) {
    throw new TypeError(`${label} must be non-negative.`)
  }
  return value
}

export function optionalNormalizedFiniteNumber(value: unknown, label: string): number | undefined {
  const number = optionalFiniteNumber(value, label)
  if (number === undefined) return undefined
  if (number < 0 || number > 1) {
    throw new TypeError(`${label} must be between 0 and 1.`)
  }
  return number
}

export function fogClipDistances(fog: NonNullable<ThreeSceneRootLike['fog']>): Pick<NativeRenderScene, 'fogNear' | 'fogFar'> {
  const near = optionalFiniteNumber(fog.near, 'scene.fog.near')
  const far = optionalFiniteNumber(fog.far, 'scene.fog.far')
  const effectiveNear = near ?? 1
  const effectiveFar = far ?? 1000

  if (effectiveFar <= effectiveNear) {
    if (far !== undefined) {
      throw new TypeError('scene.fog.far must be greater than scene.fog.near.')
    }
    throw new TypeError('scene.fog.near must be less than the effective scene.fog.far.')
  }

  return { fogNear: near, fogFar: far }
}

export function cameraClipDistances(camera: ThreeCameraLike): Pick<NativeCamera, 'near' | 'far'> {
  const near = optionalFiniteNumber(camera.near, 'camera.near')
  const far = optionalFiniteNumber(camera.far, 'camera.far')

  if (near != null && camera.isOrthographicCamera === true && near < 0) {
    throw new TypeError('camera.near must be non-negative for orthographic cameras.')
  }
  if (near != null && camera.isOrthographicCamera !== true && near <= 0) {
    throw new TypeError('camera.near must be positive.')
  }
  if (far != null && far <= 0) {
    throw new TypeError('camera.far must be positive.')
  }
  if (near != null && far != null && far <= near) {
    throw new TypeError('camera.far must be greater than camera.near.')
  }

  return { near, far }
}

export function booleanOrNumber(value: unknown): number | undefined {
  if (typeof value === 'boolean') return value ? 1 : 0
  return finiteOrUndefined(value)
}

export type EulerOrder = 'XYZ' | 'YXZ' | 'ZXY' | 'ZYX' | 'YZX' | 'XZY'
export type EulerComponents = { x: number; y: number; z: number; order: EulerOrder }

export function backgroundRotationToNative(
  rotation: ThreeSceneRootLike['backgroundRotation'],
  backgroundTexture: { mapping?: string } | null,
  label = 'scene.backgroundRotation',
): number[] | undefined {
  const euler = optionalEulerComponents(rotation, label)
  if (!euler || !hasNonZeroEulerRotation(euler)) return undefined
  if (backgroundTexture?.mapping !== 'equirectangular') {
    throw new Error(
      `${label} is only supported for equirectangular or cube texture backgrounds by @headless-three/renderer. Leave backgroundRotation at its default for color/2D backgrounds or pre-rotate the background texture before rendering.`,
    )
  }
  const { x, y, z, order } = euler
  // Three.js negates background Euler angles before producing the rotation matrix
  // to account for the background shader's left-handed frame.
  return eulerRotationMatrix3Columns(-x, -y, -z, order)
}

export function environmentRotationToNative(
  rotation: ThreeSceneRootLike['environmentRotation'],
  envMap: { data?: Buffer } | null,
  label = 'scene.environmentRotation',
): number[] | undefined {
  if (!envMap) return undefined
  const euler = optionalEulerComponents(rotation, label)
  if (!euler || !hasNonZeroEulerRotation(euler)) return undefined
  const { x, y, z, order } = euler
  return eulerRotationMatrix3Columns(-x, -y, -z, order)
}

export function optionalEulerComponents(value: ThreeEulerLike | ArrayLike<number> | null | undefined, label: string): EulerComponents | null {
  if (!value) return null
  return eulerComponents(value, label)
}

export function eulerComponents(value: ThreeEulerLike | ArrayLike<number>, label: string): EulerComponents {
  const rotation = value as ThreeEulerLike & { length?: number }
  if (typeof rotation.length === 'number') {
    const values = value as ArrayLike<number | string | undefined>
    return {
      x: finiteRotationComponent(values[0], `${label}[0]`),
      y: finiteRotationComponent(values[1], `${label}[1]`),
      z: finiteRotationComponent(values[2], `${label}[2]`),
      order: eulerOrder(values[3], `${label}[3]`),
    }
  }
  return {
    x: finiteRotationComponent(rotation.x, `${label}.x`),
    y: finiteRotationComponent(rotation.y, `${label}.y`),
    z: finiteRotationComponent(rotation.z, `${label}.z`),
    order: eulerOrder(rotation.order, `${label}.order`),
  }
}

export function finiteRotationComponent(value: unknown, label: string): number {
  if (value == null) return 0
  if (typeof value === 'number' && Number.isFinite(value)) return value
  throw new TypeError(`${label} must be a finite number`)
}

export function eulerOrder(value: unknown, label: string): EulerOrder {
  if (value == null) return 'XYZ'
  if (
    value === 'XYZ' ||
    value === 'YXZ' ||
    value === 'ZXY' ||
    value === 'ZYX' ||
    value === 'YZX' ||
    value === 'XZY'
  ) {
    return value
  }
  throw new TypeError(`${label} must be one of XYZ, YXZ, ZXY, ZYX, YZX, or XZY`)
}

export function eulerRotationMatrix3Columns(x: number, y: number, z: number, order: EulerOrder): number[] {
  const a = Math.cos(x)
  const b = Math.sin(x)
  const c = Math.cos(y)
  const d = Math.sin(y)
  const e = Math.cos(z)
  const f = Math.sin(z)
  const te = new Array<number>(9).fill(0)

  if (order === 'XYZ') {
    const ae = a * e
    const af = a * f
    const be = b * e
    const bf = b * f
    te[0] = c * e
    te[3] = -c * f
    te[6] = d
    te[1] = af + be * d
    te[4] = ae - bf * d
    te[7] = -b * c
    te[2] = bf - ae * d
    te[5] = be + af * d
    te[8] = a * c
  } else if (order === 'YXZ') {
    const ce = c * e
    const cf = c * f
    const de = d * e
    const df = d * f
    te[0] = ce + df * b
    te[3] = de * b - cf
    te[6] = a * d
    te[1] = a * f
    te[4] = a * e
    te[7] = -b
    te[2] = cf * b - de
    te[5] = df + ce * b
    te[8] = a * c
  } else if (order === 'ZXY') {
    const ce = c * e
    const cf = c * f
    const de = d * e
    const df = d * f
    te[0] = ce - df * b
    te[3] = -a * f
    te[6] = de + cf * b
    te[1] = cf + de * b
    te[4] = a * e
    te[7] = df - ce * b
    te[2] = -a * d
    te[5] = b
    te[8] = a * c
  } else if (order === 'ZYX') {
    const ae = a * e
    const af = a * f
    const be = b * e
    const bf = b * f
    te[0] = c * e
    te[3] = be * d - af
    te[6] = ae * d + bf
    te[1] = c * f
    te[4] = bf * d + ae
    te[7] = af * d - be
    te[2] = -d
    te[5] = b * c
    te[8] = a * c
  } else if (order === 'YZX') {
    const ac = a * c
    const ad = a * d
    const bc = b * c
    const bd = b * d
    te[0] = c * e
    te[3] = bd - ac * f
    te[6] = bc * f + ad
    te[1] = f
    te[4] = a * e
    te[7] = -b * e
    te[2] = -d * e
    te[5] = ad * f + bc
    te[8] = ac - bd * f
  } else {
    const ac = a * c
    const ad = a * d
    const bc = b * c
    const bd = b * d
    te[0] = c * e
    te[3] = -f
    te[6] = d * e
    te[1] = ac * f + bd
    te[4] = a * e
    te[7] = ad * f - bc
    te[2] = bc * f - ad
    te[5] = b * e
    te[8] = bd * f + ac
  }

  return te
}

export function hasNonZeroEulerRotation(rotation: EulerComponents): boolean {
  return Math.abs(rotation.x) > 1e-12 || Math.abs(rotation.y) > 1e-12 || Math.abs(rotation.z) > 1e-12
}

export function validateUnsupportedRenderOptions(options: RenderOptions): void {
  assertSupportedOutputFormat(options.format, 'options.format')
  assertSupportedOutputColorSpace(options.outputColorSpace)
  if (options.toneMapping != null) rendererStateToneMapping(options.toneMapping, 'options.toneMapping')
  if (options.toneMappingExposure != null) finiteNonNegativeNumber(options.toneMappingExposure, 'options.toneMappingExposure')
  assertNonNegativeNumberOption(options.backgroundIntensity, 'options.backgroundIntensity')
  assertNormalizedNumberOption(options.backgroundBlurriness, 'options.backgroundBlurriness')
  assertFiniteNumberOption(options.environmentIntensity, 'options.environmentIntensity')
  assertEulerOption(options.backgroundRotation, 'options.backgroundRotation')
  assertEulerOption(options.environmentRotation, 'options.environmentRotation')
  if (options.localClippingEnabled != null && typeof options.localClippingEnabled !== 'boolean') {
    throw new TypeError('options.localClippingEnabled must be a boolean.')
  }
  validateSortControls(options)
  validatePostProcessingOptions(options.postProcessing)
  assertSupportedSampleCount(options.samples, 'options.samples')
  assertSupportedSampleCount(options.sampleCount, 'options.sampleCount')
  if (options.transmissionResolutionScale != null) {
    rendererStatePositiveFiniteNumber(options.transmissionResolutionScale, 'options.transmissionResolutionScale')
  }
  if (Object.prototype.hasOwnProperty.call(options, 'target') && options.target !== undefined) {
    assertRenderTargetLike(options.target, 'options.target')
  }
  if (options.target) validateUnsupportedRenderTargetOptions(options.target)
}

export function assertRenderOptionsLike(value: unknown, label: string): asserts value is RenderOptions {
  if (value == null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be an options object.`)
  }
}
