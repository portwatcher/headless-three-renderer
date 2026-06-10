import type {
  ThreeSceneRootLike,
  ThreeCameraLike,
  RenderOptions,
  RenderTargetLike,
  RenderPixelRectLike,
  NativeRenderScene,
  NativeCamera,
  NativeSceneMesh,
  RenderMode,
  Color4,
} from './types'

// eslint-disable-next-line @typescript-eslint/no-var-requires
const native = require('../native.js')

import { resolveSize, cameraViewProjection, cameraViewMatrix, cameraWorldPosition } from './camera'
import { colorLikeToArray, resolveBackground } from './color'
import { flattenScene } from './scene'
import { extractLights, extractAmbientLight, extractAmbientIntensity, extractLightProbe } from './lights'
import { extractBackgroundTexture, extractEnvironmentMap } from './materials'
import { extractClippingPlanes } from './clipping'

export {
  EncodedImageTextureLoader,
  createEncodedImageTextureLoader,
  installLocalFileFetch,
  resolveLocalAssetPath,
} from './loaders'

export type {
  RenderOutputFormat,
  RenderOutputColorSpace,
  RenderMode,
  ThreeColorLike,
  ThreeMatrix4Like,
  ThreeBufferAttributeLike,
  ThreeBufferGeometryLike,
  ThreeTextureLike,
  ThreeVector3Like,
  ThreeEulerLike,
  ThreePlaneLike,
  RenderPixelRectLike,
  ThreeLayersLike,
  ThreeMaterialLike,
  ThreeBoneLike,
  ThreeSkeletonLike,
  ThreeObject3DLike,
  ThreeSceneRootLike,
  ThreeSceneLike,
  ThreeCameraLike,
  RenderOptions,
  RenderTargetLike,
  PostProcessingOptions,
} from './types'

export class Renderer {
  private native: InstanceType<typeof native.NativeRenderer>

  constructor() {
    this.native = new native.NativeRenderer()
  }

  render(scene: ThreeSceneRootLike, camera: ThreeCameraLike, options: RenderOptions = {}): Buffer {
    const { buffer, nativeScene } = this.renderNative(scene, camera, options)
    if (options.target) {
      writeRenderTarget(options.target, buffer, nativeScene.width!, nativeScene.height!)
    }
    return buffer
  }

  renderToTarget(
    scene: ThreeSceneRootLike,
    camera: ThreeCameraLike,
    target: RenderTargetLike = {},
    options: RenderOptions = {},
  ): RenderTargetLike {
    const targetOptions: RenderOptions = { ...options, target, format: options.format ?? 'rgba' }
    const { buffer, nativeScene } = this.renderNative(scene, camera, targetOptions)
    return writeRenderTarget(target, buffer, nativeScene.width!, nativeScene.height!)
  }

  private renderNative(
    scene: ThreeSceneRootLike,
    camera: ThreeCameraLike,
    options: RenderOptions,
  ): { buffer: Buffer; nativeScene: NativeRenderScene } {
    const { nativeScene, nativeCamera } = toNativeInput(scene, camera, options)
    return { buffer: this.native.render(nativeScene, nativeCamera), nativeScene }
  }
}

export function render(scene: ThreeSceneRootLike, camera: ThreeCameraLike, options: RenderOptions = {}): Buffer {
  const { nativeScene, nativeCamera } = toNativeInput(scene, camera, options)
  const buffer = native.renderNative(nativeScene, nativeCamera)
  if (options.target) {
    writeRenderTarget(options.target, buffer, nativeScene.width!, nativeScene.height!)
  }
  return buffer
}

export function renderToTarget(
  scene: ThreeSceneRootLike,
  camera: ThreeCameraLike,
  target: RenderTargetLike = {},
  options: RenderOptions = {},
): RenderTargetLike {
  const targetOptions: RenderOptions = { ...options, target, format: options.format ?? 'rgba' }
  const { nativeScene, nativeCamera } = toNativeInput(scene, camera, targetOptions)
  const buffer = native.renderNative(nativeScene, nativeCamera)
  return writeRenderTarget(target, buffer, nativeScene.width!, nativeScene.height!)
}

function toNativeInput(
  scene: ThreeSceneRootLike,
  camera: ThreeCameraLike,
  options: RenderOptions,
): { nativeScene: NativeRenderScene; nativeCamera: NativeCamera } {
  validateThreeSceneRoot(scene)
  validateThreeCamera(camera)
  validateUnsupportedSceneState(scene)
  validateUnsupportedRenderOptions(options)
  const renderMode = normalizedRenderMode(options.renderMode)
  const colorMode = renderMode === 'color'

  if (typeof scene.updateMatrixWorld === 'function') {
    scene.updateMatrixWorld(true)
  }
  if (typeof camera.updateMatrixWorld === 'function') {
    camera.updateMatrixWorld(true)
  }

  const size = resolveSize(camera, options)
  const envMap = colorMode ? extractEnvironmentMap(scene) : null
  const hasBackgroundOverride = options.background !== undefined
  const optionBackgroundTexture = colorMode && hasBackgroundOverride
    ? extractBackgroundTexture(options.background, 'options.background')
    : null
  const backgroundTexture = colorMode
    ? optionBackgroundTexture ?? (hasBackgroundOverride ? null : extractBackgroundTexture(scene.background, 'scene.background'))
    : null
  const clippingPlanes = extractClippingPlanes(options.clippingPlanes)
  const meshes = applyRenderMode(flattenScene(scene, camera, size.height, clippingPlanes), renderMode)
  const nativeScene: NativeRenderScene = {
    width: size.width,
    height: size.height,
    background: colorMode ? resolveBackground(scene, options) : [0, 0, 0, 1],
    backgroundIntensity: colorMode ? options.backgroundIntensity ?? scene.backgroundIntensity : undefined,
    viewport: pixelRectToArray(options.viewport),
    scissor: pixelRectToArray(options.scissor),
    backgroundTexture: backgroundTexture?.data,
    backgroundTextureWidth: backgroundTexture?.width,
    backgroundTextureHeight: backgroundTexture?.height,
    backgroundTextureWrapS: backgroundTexture?.wrapS,
    backgroundTextureWrapT: backgroundTexture?.wrapT,
    backgroundTextureMagFilter: backgroundTexture?.magFilter,
    backgroundTextureMinFilter: backgroundTexture?.minFilter,
    backgroundTextureTransform: backgroundTexture?.transform,
    backgroundTextureColorSpace: backgroundTexture?.colorSpace,
    backgroundTextureBlurriness: colorMode ? finiteOrUndefined(options.backgroundBlurriness ?? scene.backgroundBlurriness) : undefined,
    format: options.format ?? (options.target ? 'rgba' : 'png'),
    outputColorSpace: options.outputColorSpace,
    meshes,
    lights: colorMode ? extractLights(scene, camera) : [],
    ambientLight: colorMode ? extractAmbientLight(scene, camera) ?? undefined : undefined,
    ambientIntensity: colorMode ? extractAmbientIntensity(scene, camera) ?? undefined : undefined,
    lightProbe: colorMode ? extractLightProbe(scene, camera) ?? undefined : undefined,
    environmentMap: envMap?.data,
    environmentMapWidth: envMap?.width,
    environmentMapHeight: envMap?.height,
    environmentMapIntensity: envMap?.intensity,
    ...(colorMode ? fogToNative(scene.fog) : {}),
    ...(colorMode ? postProcessingToNative(options.postProcessing) : {}),
  }
  const nativeCamera: NativeCamera = {
    width: size.width,
    height: size.height,
    near: finiteOrUndefined(camera.near),
    far: finiteOrUndefined(camera.far),
    viewProjection: cameraViewProjection(camera),
    viewMatrix: cameraViewMatrix(camera),
    cameraPosition: cameraWorldPosition(camera),
  }

  return { nativeScene, nativeCamera }
}

function normalizedRenderMode(mode: RenderOptions['renderMode']): RenderMode {
  if (mode == null) return 'color'
  if (mode === 'color' || mode === 'mask' || mode === 'object-id') return mode
  throw new TypeError(
    `options.renderMode must be "color", "mask", or "object-id"; received ${String(mode)}`,
  )
}

function applyRenderMode(meshes: NativeSceneMesh[], mode: RenderMode): NativeSceneMesh[] {
  if (mode === 'color') return meshes
  return meshes.map((mesh, index) => renderModeMesh(mesh, mode, index))
}

function renderModeMesh(mesh: NativeSceneMesh, mode: Exclude<RenderMode, 'color'>, index: number): NativeSceneMesh {
  const color = mode === 'mask'
    ? [1, 1, 1, materialAlpha(mesh)] as Color4
    : objectIdColor(mesh, index)
  return {
    positions: mesh.positions,
    indices: mesh.indices,
    normals: mesh.normals,
    colors: mesh.colors,
    color,
    transform: mesh.transform,
    uvs: mesh.uvs,
    uvs2: mesh.uvs2,
    texture: mesh.texture,
    textureWidth: mesh.textureWidth,
    textureHeight: mesh.textureHeight,
    textureWrapS: mesh.textureWrapS,
    textureWrapT: mesh.textureWrapT,
    textureMagFilter: mesh.textureMagFilter,
    textureMinFilter: mesh.textureMinFilter,
    textureTransform: mesh.textureTransform,
    textureColorSpace: mesh.textureColorSpace,
    textureUsesUv2: mesh.textureUsesUv2,
    alphaMap: mesh.alphaMap,
    alphaMapWidth: mesh.alphaMapWidth,
    alphaMapHeight: mesh.alphaMapHeight,
    alphaMapWrapS: mesh.alphaMapWrapS,
    alphaMapWrapT: mesh.alphaMapWrapT,
    alphaMapMagFilter: mesh.alphaMapMagFilter,
    alphaMapMinFilter: mesh.alphaMapMinFilter,
    alphaMapTransform: mesh.alphaMapTransform,
    alphaMapUsesUv2: mesh.alphaMapUsesUv2,
    alphaTest: mesh.alphaTest,
    alphaHash: mesh.alphaHash,
    premultipliedAlpha: mesh.premultipliedAlpha,
    clippingPlanes: mesh.clippingPlanes,
    clippingUnionCount: mesh.clippingUnionCount,
    blending: 'none',
    depthTest: mesh.depthTest,
    depthWrite: true,
    colorWrite: true,
    polygonOffset: mesh.polygonOffset,
    polygonOffsetFactor: mesh.polygonOffsetFactor,
    polygonOffsetUnits: mesh.polygonOffsetUnits,
    stencilWrite: mesh.stencilWrite,
    stencilWriteMask: mesh.stencilWriteMask,
    stencilFunc: mesh.stencilFunc,
    stencilRef: mesh.stencilRef,
    stencilFuncMask: mesh.stencilFuncMask,
    stencilFail: mesh.stencilFail,
    stencilZFail: mesh.stencilZFail,
    stencilZPass: mesh.stencilZPass,
    transparent: false,
    side: mesh.side,
    shadingModel: 'basic',
    topology: mesh.topology,
    customFragmentShader: renderModeFragment(color),
    castShadow: false,
    receiveShadow: false,
    groupOrder: mesh.groupOrder,
    renderOrder: mesh.renderOrder,
    sortZ: mesh.sortZ,
    sortIndex: mesh.sortIndex,
    materialSortKey: mesh.materialSortKey,
  }
}

function materialAlpha(mesh: NativeSceneMesh): number {
  const alpha = mesh.color?.[3]
  return typeof alpha === 'number' && Number.isFinite(alpha) ? Math.min(1, Math.max(0, alpha)) : 1
}

function objectIdColor(mesh: NativeSceneMesh, index: number): Color4 {
  const sortIndex = typeof mesh.sortIndex === 'number' && Number.isSafeInteger(mesh.sortIndex) && mesh.sortIndex >= 0
    ? mesh.sortIndex
    : index
  const id = (sortIndex + 1) & 0xffffff
  const value = id === 0 ? 1 : id
  return [
    ((value >> 16) & 0xff) / 255,
    ((value >> 8) & 0xff) / 255,
    (value & 0xff) / 255,
    materialAlpha(mesh),
  ]
}

function renderModeFragment(color: Color4): string {
  return `return vec4<f32>(${formatWgslFloat(color[0])}, ${formatWgslFloat(color[1])}, ${formatWgslFloat(color[2])}, 1.0);`
}

function formatWgslFloat(value: number): string {
  if (value <= 0) return '0.0'
  if (value >= 1) return '1.0'
  return value.toFixed(10)
}

function fogToNative(fog: ThreeSceneRootLike['fog']): Partial<NativeRenderScene> {
  if (!fog) return {}
  const color = colorLikeToArray(fog.color)
  if (fog.isFogExp2) {
    return {
      fogType: 'exp2',
      fogColor: color ?? undefined,
      fogDensity: finiteOrUndefined(fog.density),
    }
  }
  if (fog.isFog) {
    return {
      fogType: 'linear',
      fogColor: color ?? undefined,
      fogNear: finiteOrUndefined(fog.near),
      fogFar: finiteOrUndefined(fog.far),
    }
  }
  return {}
}

function postProcessingToNative(post: RenderOptions['postProcessing']): Partial<NativeRenderScene> {
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

function pixelRectToArray(rect: RenderPixelRectLike | null | undefined): number[] | undefined {
  if (!rect) return undefined
  if (typeof (rect as ArrayLike<number>).length === 'number') {
    const values = rect as ArrayLike<number>
    return [values[0], values[1], values[2], values[3]]
  }
  const values = rect as { x?: number; y?: number; width?: number; height?: number }
  return [values.x!, values.y!, values.width!, values.height!]
}

function finiteOrUndefined(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function booleanOrNumber(value: unknown): number | undefined {
  if (typeof value === 'boolean') return value ? 1 : 0
  return finiteOrUndefined(value)
}

function validateUnsupportedSceneState(scene: ThreeSceneRootLike): void {
  if (hasNonZeroRotation(scene.backgroundRotation)) {
    throw new Error(
      'scene.backgroundRotation is not supported by @headless-three/renderer yet. Leave backgroundRotation at its default zero rotation or pre-rotate the background texture before rendering.',
    )
  }
  if (hasNonZeroRotation(scene.environmentRotation)) {
    throw new Error(
      'scene.environmentRotation is not supported by @headless-three/renderer yet. Leave environmentRotation at its default zero rotation or pre-rotate the environment texture before rendering.',
    )
  }
}

function hasNonZeroRotation(value: unknown): boolean {
  if (!value) return false
  const rotation = value as { x?: unknown; y?: unknown; z?: unknown; length?: unknown }
  if (
    nonZeroFinite(rotation.x) ||
    nonZeroFinite(rotation.y) ||
    nonZeroFinite(rotation.z)
  ) {
    return true
  }
  if (typeof rotation.length === 'number') {
    const values = value as ArrayLike<unknown>
    return nonZeroFinite(values[0]) || nonZeroFinite(values[1]) || nonZeroFinite(values[2])
  }
  return false
}

function nonZeroFinite(value: unknown): boolean {
  return typeof value === 'number' && Number.isFinite(value) && Math.abs(value) > 1e-12
}

function validateUnsupportedRenderOptions(options: RenderOptions): void {
  assertSupportedSampleCount(options.samples, 'options.samples')
  assertSupportedSampleCount(options.sampleCount, 'options.sampleCount')
  if (options.target) validateUnsupportedRenderTargetOptions(options.target)
}

function validateUnsupportedRenderTargetOptions(target: RenderTargetLike): void {
  if (target.depthTexture != null) {
    throw new Error(
      'Render target depthTexture output is not supported by @headless-three/renderer yet. Render depth with MeshDepthMaterial or omit target.depthTexture until depth readback support lands.',
    )
  }
  if (target.isWebGLMultipleRenderTargets === true || Array.isArray(target.texture)) {
    throw new Error(
      'Multiple render target color attachments are not supported by @headless-three/renderer yet. Render separate passes or use a single color target until MRT support lands.',
    )
  }
  if (Array.isArray(target.textures) && target.textures.length > 1) {
    throw new Error(
      'Multiple render target color attachments are not supported by @headless-three/renderer yet. Render separate passes or use a single color target until MRT support lands.',
    )
  }
  assertSupportedSampleCount(target.samples, 'target.samples')
  assertSupportedSampleCount(target.sampleCount, 'target.sampleCount')
}

function assertSupportedSampleCount(value: unknown, label: string): void {
  if (typeof value === 'number' && Number.isFinite(value) && value > 1) {
    throw new Error(
      `MSAA sample counts greater than 1 are not supported by @headless-three/renderer yet (${label}=${value}). Use the default single-sample render path until MSAA support lands.`,
    )
  }
}

function writeRenderTarget(
  target: RenderTargetLike,
  data: Buffer,
  width: number,
  height: number,
): RenderTargetLike {
  target.width = width
  target.height = height
  target.data = data

  const image = target.image ?? (target.image = {})
  image.data = data
  image.width = width
  image.height = height

  const texture = target.texture ?? target.textures?.[0]
  if (texture && !Array.isArray(texture)) {
    const textureImage = texture.image ?? (texture.image = {})
    textureImage.data = data
    textureImage.width = width
    textureImage.height = height

    if (texture.source?.data) {
      texture.source.data.data = data
      texture.source.data.width = width
      texture.source.data.height = height
    }
  }

  return target
}

function validateThreeSceneRoot(scene: unknown): asserts scene is ThreeSceneRootLike {
  const root = scene as any
  if (!root || (root.isScene !== true && root.isObject3D !== true)) {
    throw new TypeError('render(scene, camera) expects scene to be a THREE.Scene or THREE.Object3D root')
  }
}

function validateThreeCamera(camera: unknown): asserts camera is ThreeCameraLike {
  const cameraLike = camera as any
  if (cameraLike?.isCubeCamera === true || cameraLike?.type === 'CubeCamera') {
    throw new Error(
      'THREE.CubeCamera is not supported by @headless-three/renderer yet. Render each cube face with a regular camera until cube camera support lands.',
    )
  }
  if (!camera || cameraLike.isCamera !== true) {
    throw new TypeError('render(scene, camera) expects camera to be a THREE.Camera')
  }
  if (cameraLike.isArrayCamera === true || Array.isArray(cameraLike.cameras)) {
    throw new Error(
      'THREE.ArrayCamera is not supported by @headless-three/renderer yet. Render each sub-camera separately until array camera support lands.',
    )
  }
  if (!cameraLike.projectionMatrix || !cameraLike.matrixWorldInverse) {
    throw new TypeError('THREE.Camera must have projectionMatrix and matrixWorldInverse')
  }
}
