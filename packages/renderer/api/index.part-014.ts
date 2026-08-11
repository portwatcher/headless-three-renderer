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
import { RendererBooleanParameters, SupportedRendererInfoDrawModes, SupportedRendererPowerPreferences, SupportedRendererStateBlendingModes, SupportedRendererStateCullFaces } from './index.part-001'
import { DEPTH_READBACK_FRAGMENT, InternalRenderOptions, PixelRect, PixelSize } from './index.part-012'
import { assertRendererConstructorSamples, assertRendererContextParameterAbsent, assertRendererOutputBufferType, assertRendererUnsupportedDepthParameterFalse, booleanOrNumber, finiteOrUndefined, fogClipDistances, optionalNonNegativeFiniteNumber } from './index.part-015'
export function depthReadbackScene(scene: NativeRenderScene): NativeRenderScene {
  return {
    ...scene,
    background: [0, 0, 0, 1],
    backgroundIntensity: 1,
    backgroundTexture: undefined,
    backgroundTextureWidth: undefined,
    backgroundTextureHeight: undefined,
    backgroundTextureWrapS: undefined,
    backgroundTextureWrapT: undefined,
    backgroundTextureMagFilter: undefined,
    backgroundTextureMinFilter: undefined,
    backgroundTextureAnisotropy: undefined,
    backgroundTextureTransform: undefined,
    backgroundTextureColorSpace: undefined,
    backgroundTextureMapping: undefined,
    backgroundTextureRotation: undefined,
    backgroundTextureBlurriness: undefined,
    format: 'rgba',
    outputColorSpace: 'srgb-linear',
    toneMapping: undefined,
    toneMappingExposure: undefined,
    sampleCount: 1,
    meshes: scene.meshes?.map(depthReadbackMesh),
    lights: [],
    ambientLight: undefined,
    ambientIntensity: undefined,
    lightProbe: undefined,
    environmentMap: undefined,
    environmentMapWidth: undefined,
    environmentMapHeight: undefined,
    environmentMapIntensity: undefined,
    environmentMapColorSpace: undefined,
    environmentMapRotation: undefined,
    fogType: undefined,
    fogColor: undefined,
    fogNear: undefined,
    fogFar: undefined,
    fogDensity: undefined,
    postExposure: undefined,
    postContrast: undefined,
    postSaturation: undefined,
    postVignette: undefined,
    postGrayscale: undefined,
    postInvert: undefined,
  }
}

export function depthReadbackMesh(mesh: NativeSceneMesh): NativeSceneMesh {
  const writesDepth = meshWritesDepth(mesh)
  return {
    ...mesh,
    blending: 'none',
    depthWrite: writesDepth,
    colorWrite: writesDepth,
    transparent: false,
    shadingModel: 'basic',
    toneMapped: false,
    alphaToCoverage: false,
    customFragmentShader: DEPTH_READBACK_FRAGMENT,
    castShadow: false,
    receiveShadow: false,
  }
}

export function meshWritesDepth(mesh: NativeSceneMesh): boolean {
  if (mesh.depthTest === false) return false
  if (typeof mesh.depthWrite === 'boolean') return mesh.depthWrite
  return true
}

export function formatWgslFloat(value: number): string {
  if (value <= 0) return '0.0'
  if (value >= 1) return '1.0'
  return value.toFixed(10)
}

export function fogToNative(fog: ThreeSceneRootLike['fog']): Partial<NativeRenderScene> {
  if (!fog) return {}
  if (typeof fog !== 'object') {
    throw new TypeError('scene.fog must be a THREE.Fog or THREE.FogExp2 object.')
  }
  const color = validatedColorLikeToArray(fog.color, 'scene.fog.color')
  if (fog.isFogExp2) {
    return {
      fogType: 'exp2',
      fogColor: color ?? undefined,
      fogDensity: optionalNonNegativeFiniteNumber(fog.density, 'scene.fog.density'),
    }
  }
  if (fog.isFog) {
    const clipDistances = fogClipDistances(fog)
    return {
      fogType: 'linear',
      fogColor: color ?? undefined,
      fogNear: clipDistances.fogNear,
      fogFar: clipDistances.fogFar,
    }
  }
  throw new TypeError('scene.fog must be a THREE.Fog or THREE.FogExp2 object.')
}

export function postProcessingToNative(post: RenderOptions['postProcessing']): Partial<NativeRenderScene> {
  if (!post || post.enabled === false) return {}
  return {
    postExposure: finiteOrUndefined(post.exposure),
    postContrast: finiteOrUndefined(post.contrast),
    postSaturation: finiteOrUndefined(post.saturation),
    postVignette: finiteOrUndefined(post.vignette),
    postGrayscale: booleanOrNumber(post.grayscale),
    postInvert: booleanOrNumber(post.invert),
  }
}

export function pixelRectToArray(rect: RenderPixelRectLike | null | undefined): number[] | undefined {
  if (!rect) return undefined
  return pixelRectComponents(rect)
}

export function effectiveViewport(options: RenderOptions): RenderPixelRectLike | null | undefined {
  if (options.viewport !== undefined) return options.viewport
  if (options.target?.viewport !== undefined) return options.target.viewport
  return (options as InternalRenderOptions).__headlessThreeRendererViewport
}

export function effectiveScissor(options: RenderOptions): RenderPixelRectLike | null | undefined {
  if (options.scissor !== undefined) return options.scissor
  if (options.target?.scissorTest === true) return options.target.scissor
  const internal = options as InternalRenderOptions
  return internal.__headlessThreeRendererScissorTest === true ? internal.__headlessThreeRendererScissor : undefined
}

export function effectiveViewportLabel(options: RenderOptions): string {
  const internalLabel = (options as InternalRenderOptions).__headlessThreeViewportLabel
  if (internalLabel) return internalLabel
  if (options.viewport !== undefined) return 'options.viewport'
  if (options.target?.viewport !== undefined) return 'target.viewport'
  return (options as InternalRenderOptions).__headlessThreeRendererViewport !== undefined
    ? 'Renderer.viewport'
    : 'target.viewport'
}

export function effectiveScissorLabel(options: RenderOptions): string {
  const internalLabel = (options as InternalRenderOptions).__headlessThreeScissorLabel
  if (internalLabel) return internalLabel
  if (options.scissor !== undefined) return 'options.scissor'
  if (options.target?.scissorTest === true) return 'target.scissor'
  return (options as InternalRenderOptions).__headlessThreeRendererScissorTest === true
    ? 'Renderer.scissor'
    : 'target.scissor'
}

export function pixelRectComponents(rect: RenderPixelRectLike): number[] {
  if (typeof (rect as ArrayLike<number>).length === 'number') {
    const values = rect as ArrayLike<number>
    return [values[0], values[1], values[2], values[3]]
  }
  const values = rect as { x?: number; y?: number; width?: number; height?: number; z?: number; w?: number }
  return [values.x!, values.y!, values.width ?? values.z!, values.height ?? values.w!]
}

export function rendererStatePixelRect(
  rectOrX: RenderPixelRectLike | null | number,
  y: number | undefined,
  width: number | undefined,
  height: number | undefined,
  label: string,
): PixelRect | null {
  if (rectOrX == null) return null
  if (typeof rectOrX === 'number') {
    return rendererStatePixelRectFromComponents([rectOrX, y, width, height], label)
  }
  if (typeof rectOrX !== 'object') {
    throw new TypeError(`${label} expects a rectangle object, array, or x/y/width/height numbers.`)
  }
  return rendererStatePixelRectFromComponents(pixelRectComponents(rectOrX), label)
}

export function rendererStatePixelRectFromComponents(values: unknown[], label: string): PixelRect {
  const [rawX, rawY, rawWidth, rawHeight] = values
  if (![rawX, rawY, rawWidth, rawHeight].every((value) => typeof value === 'number' && Number.isFinite(value))) {
    throw new TypeError(`${label} must contain finite x, y, width, and height values.`)
  }
  const x = Math.round(rawX as number)
  const y = Math.round(rawY as number)
  const width = Math.round(rawWidth as number)
  const height = Math.round(rawHeight as number)
  if (x < 0 || y < 0) {
    throw new TypeError(`${label} x and y must be greater than or equal to 0.`)
  }
  if (width <= 0 || height <= 0) {
    throw new TypeError(`${label} width and height must be greater than 0.`)
  }
  return { x, y, width, height }
}

export function assertDefaultViewportDepthRange(minDepth: unknown, maxDepth: unknown, label: string): void {
  const min = rendererViewportDepthValue(minDepth, `${label} minDepth`)
  const max = rendererViewportDepthValue(maxDepth, `${label} maxDepth`)
  if (min !== 0 || max !== 1) {
    throw new Error(`${label} depth ranges other than 0..1 are not supported by @headless-three/renderer.`)
  }
}

export function rendererViewportDepthValue(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a finite number.`)
  }
  if (value < 0 || value > 1) {
    throw new TypeError(`${label} must be between 0 and 1.`)
  }
  return value
}

export function rendererStateClearColor(color: number | string | ThreeColorLike | number[], alpha?: number): Color4 {
  const colorArray = typeof color === 'number'
    ? rendererStateHexColor(color, 'Renderer.setClearColor color')
    : typeof color === 'string'
      ? cssColorStringToArray(color, 'Renderer.setClearColor color')
      : validatedColorLikeToArray(color, 'Renderer.setClearColor color')
  if (!colorArray) {
    throw new TypeError('Renderer.setClearColor color must be a hex number, CSS color string, color-like object, or [r, g, b].')
  }
  return [
    colorArray[0],
    colorArray[1],
    colorArray[2],
    alpha === undefined ? colorArray[3] : rendererStateClearAlpha(alpha, 'Renderer.setClearColor alpha'),
  ]
}

export function rendererStateHexColor(value: number, label: string): Color4 {
  if (!Number.isFinite(value) || !Number.isInteger(value)) {
    throw new TypeError(`${label} must be a finite integer hex color.`)
  }
  if (value < 0 || value > 0xffffff) {
    throw new TypeError(`${label} must be between 0x000000 and 0xffffff.`)
  }
  return [
    ((value >> 16) & 0xff) / 255,
    ((value >> 8) & 0xff) / 255,
    (value & 0xff) / 255,
    1,
  ]
}

export function rendererStateClearAlpha(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a finite number.`)
  }
  return clamp01(value)
}

export function rendererStateClearDepth(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a finite number.`)
  }
  return clamp01(value)
}

export function rendererStateClearStencil(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || !Number.isInteger(value)) {
    throw new TypeError(`${label} must be a finite integer.`)
  }
  return value
}

export function cloneColor4(color: Color4): Color4 {
  return [color[0], color[1], color[2], color[3]]
}

export function cloneColor3(color: Color4): ThreeColorLike
export function cloneColor3<T extends ThreeColorLike>(color: Color4, target: T): T
export function cloneColor3<T extends ThreeColorLike>(color: Color4, target?: T): ThreeColorLike | T {
  if (target) {
    const mutable = target as any
    if (typeof mutable.setRGB === 'function') {
      mutable.setRGB(color[0], color[1], color[2])
    } else {
      mutable.r = color[0]
      mutable.g = color[1]
      mutable.b = color[2]
    }
    return target
  }
  return { isColor: true, r: color[0], g: color[1], b: color[2] }
}

export function rendererStateSize(width: unknown, height: unknown, label: string): PixelSize {
  return {
    width: rendererStateSizeDimension(width, `${label} width`),
    height: rendererStateSizeDimension(height, `${label} height`),
  }
}

export function rendererStateSizeDimension(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a finite number.`)
  }
  if (!Number.isInteger(value) || value <= 0) {
    throw new TypeError(`${label} must be a positive integer.`)
  }
  return value
}

export function rendererStatePixelRatio(value: unknown, label: string): number {
  return rendererStatePositiveFiniteNumber(value, `${label} value`)
}

export function rendererStatePositiveFiniteNumber(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a finite number.`)
  }
  if (value <= 0) {
    throw new TypeError(`${label} must be greater than 0.`)
  }
  return value
}

export function rendererStateFiniteNumber(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a finite number.`)
  }
  return value
}

export function rendererStateOptionalFiniteInteger(value: unknown, label: string): void {
  if (value !== undefined) {
    rendererStateClearStencil(value, label)
  }
}

export function rendererStateBoolean(value: unknown, label: string): boolean {
  if (typeof value !== 'boolean') {
    throw new TypeError(`${label} must be a boolean.`)
  }
  return value
}

export function rendererInfoDrawCount(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || Math.floor(value) !== value || value < 0) {
    throw new TypeError(`${label} must be a non-negative integer.`)
  }
  return value
}

export function rendererInfoInstanceCount(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || Math.floor(value) !== value || value < 0) {
    throw new TypeError(`${label} must be a non-negative integer.`)
  }
  return value
}

export function rendererInfoDrawMode(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || Math.floor(value) !== value) {
    throw new TypeError(`${label} must be an integer WebGL draw mode.`)
  }
  if (!SupportedRendererInfoDrawModes.has(value)) {
    throw new Error(
      `${label} ${String(value)} is not supported. Use POINTS, LINES, LINE_STRIP, LINE_LOOP, or TRIANGLES WebGL draw mode constants.`,
    )
  }
  return value
}

export function rendererInfoTimestampTime(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a finite number.`)
  }
  if (value < 0) {
    throw new TypeError(`${label} must be non-negative.`)
  }
  return value
}

export function assertRendererStateCullFace(value: unknown, label: string): asserts value is number {
  if (!SupportedRendererStateCullFaces.has(value as number)) {
    throw new Error(
      `${label} ${String(value)} is not supported. Use THREE.CullFaceNone, CullFaceBack, CullFaceFront, or CullFaceFrontBack.`,
    )
  }
}

export function assertRendererStateBlendingMode(value: unknown, label: string): asserts value is number {
  if (!SupportedRendererStateBlendingModes.has(value as number)) {
    throw new Error(
      `${label} ${String(value)} is not supported. Use a Three.js blending constant such as NormalBlending, AdditiveBlending, or CustomBlending.`,
    )
  }
}

export function throwUnsupportedRendererStateWebGl(method: string, operation: string): never {
  throw new Error(
    `Renderer.state.${method}() is not supported by @headless-three/renderer because it does not expose ${operation}. Render normal Three.js scene graphs with Renderer.render() or renderToTarget().`,
  )
}

export function assertRendererParametersLike(value: RendererParametersLike | undefined, label: string): void {
  if (value === undefined) return
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be an object when provided.`)
  }
  const parameters = value as Record<string, unknown>
  for (const name of RendererBooleanParameters) {
    if (parameters[name] !== undefined) {
      rendererStateBoolean(parameters[name], `${label}.${name}`)
    }
  }
  if (parameters.powerPreference !== undefined) {
    rendererStatePowerPreference(parameters.powerPreference, `${label}.powerPreference`)
  }
  assertRendererConstructorSamples(parameters.samples, `${label}.samples`)
  assertRendererOutputBufferType(parameters.outputBufferType, `${label}.outputBufferType`)
  assertRendererContextParameterAbsent(parameters, 'canvas', label)
  assertRendererContextParameterAbsent(parameters, 'context', label)
  assertRendererUnsupportedDepthParameterFalse(parameters, 'logarithmicDepthBuffer', label)
  assertRendererUnsupportedDepthParameterFalse(parameters, 'reversedDepthBuffer', label)
  assertRendererUnsupportedDepthParameterFalse(parameters, 'reverseDepthBuffer', label)
}

export function rendererStatePowerPreference(value: unknown, label: string): void {
  if (typeof value !== 'string') {
    throw new TypeError(`${label} must be a WebGL powerPreference string.`)
  }
  if (!SupportedRendererPowerPreferences.has(value)) {
    throw new TypeError(`${label} "${value}" is not supported. Use "default", "high-performance", or "low-power".`)
  }
}
