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
  NativeRenderScene,
  NativeCamera,
  NativeSceneMesh,
  NativeSceneLight,
  RenderMode,
  Color4,
  RenderObjectIdEntry,
  ThreeEulerLike,
  RenderSortFunction,
} from './types'

// eslint-disable-next-line @typescript-eslint/no-var-requires
const native = require('../native.js')

import { resolveSize, cameraViewProjection, cameraViewMatrix, cameraWorldPosition } from './camera'
import { resolveBackground, strictColorLikeToArray } from './color'
import { flattenScene, type ShadowMaterialMode } from './scene'
import { extractLights, extractAmbientLight, extractAmbientIntensity, extractLightProbe } from './lights'
import { extractBackgroundTexture, resolveEnvironmentMap } from './materials'
import { extractClippingPlanes } from './clipping'

export {
  EncodedImageTextureLoader,
  createEncodedImageTextureLoader,
  createNodeGltfLoader,
  installLocalFileFetch,
  loadGltfFromFile,
  loadVrmAnimationFromFile,
  loadVrmFromFile,
  resolveLocalAssetPath,
} from './loaders'

export type {
  ConfigureGltfLoader,
  LoadGltfFromFileOptions,
  LoadVrmAnimationFromFileOptions,
  LoadVrmFromFileOptions,
  NodeGltfLoaderBundle,
  NodeGltfLoaderOptions,
  ThreeGltfLoaderLike,
  ThreeLoadingManagerLike,
  VrmLoaderPluginConstructor,
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
  ThreeCubeCameraLike,
  ThreeRenderCameraLike,
  RenderOptions,
  RenderTargetLike,
  RenderObjectIdEntry,
  RenderSortFunction,
  RenderSortItem,
  PostProcessingOptions,
} from './types'

export class Renderer {
  private native: InstanceType<typeof native.NativeRenderer>
  private opaqueSort: RenderSortFunction | null = null
  private transparentSort: RenderSortFunction | null = null

  sortObjects = true

  constructor() {
    this.native = new native.NativeRenderer()
  }

  setOpaqueSort(method: RenderSortFunction | null): void {
    assertSortFunctionOrNull(method, 'Renderer.setOpaqueSort')
    this.opaqueSort = method
  }

  setTransparentSort(method: RenderSortFunction | null): void {
    assertSortFunctionOrNull(method, 'Renderer.setTransparentSort')
    this.transparentSort = method
  }

  render(scene: ThreeSceneRootLike, camera: ThreeRenderCameraLike, options: RenderOptions = {}): Buffer {
    assertRenderOptionsLike(options, 'options')
    const renderOptions = this.resolveRenderOptions(options)
    if (isCubeCamera(camera)) {
      const { buffer } = renderCubeCamera(
        scene,
        camera,
        renderOptions,
        (targetScene, targetCamera) => this.native.render(targetScene, targetCamera),
      )
      return buffer
    }

    if (isArrayCamera(camera)) {
      const { buffer, width, height, objectIdEntries, depthData } = renderArrayCamera(
        scene,
        camera,
        renderOptions,
        (targetScene, targetCamera) => this.native.render(targetScene, targetCamera),
      )
      if (renderOptions.target) {
        writeRenderTarget(renderOptions.target, buffer, width, height, objectIdEntries, depthData)
      }
      return buffer
    }

    const { buffer, nativeScene, nativeCamera, objectIdEntries } = this.renderNative(scene, camera, renderOptions)
    if (renderOptions.target) {
      const depthData = renderTargetDepthBuffer(
        renderOptions.target,
        nativeScene,
        nativeCamera,
        (targetScene, targetCamera) => this.native.render(targetScene, targetCamera),
      )
      writeRenderTarget(renderOptions.target, buffer, nativeScene.width!, nativeScene.height!, objectIdEntries, depthData)
    }
    return buffer
  }

  renderToTarget(
    scene: ThreeSceneRootLike,
    camera: ThreeRenderCameraLike,
    target: RenderTargetLike = {},
    options: RenderOptions = {},
  ): RenderTargetLike {
    assertRenderTargetLike(target, 'target')
    assertRenderOptionsLike(options, 'options')
    const targetOptions: RenderOptions = this.resolveRenderOptions({ ...options, target, format: options.format ?? 'rgba' })
    if (isCubeCamera(camera)) {
      const { target: cubeTarget } = renderCubeCamera(
        scene,
        camera,
        targetOptions,
        (targetScene, targetCamera) => this.native.render(targetScene, targetCamera),
      )
      return cubeTarget
    }

    if (isArrayCamera(camera)) {
      const { buffer, width, height, objectIdEntries, depthData } = renderArrayCamera(
        scene,
        camera,
        targetOptions,
        (targetScene, targetCamera) => this.native.render(targetScene, targetCamera),
      )
      return writeRenderTarget(target, buffer, width, height, objectIdEntries, depthData)
    }

    const { buffer, nativeScene, nativeCamera, objectIdEntries } = this.renderNative(scene, camera, targetOptions)
    const depthData = renderTargetDepthBuffer(
      target,
      nativeScene,
      nativeCamera,
      (targetScene, targetCamera) => this.native.render(targetScene, targetCamera),
    )
    return writeRenderTarget(target, buffer, nativeScene.width!, nativeScene.height!, objectIdEntries, depthData)
  }

  private renderNative(
    scene: ThreeSceneRootLike,
    camera: ThreeCameraLike,
    options: RenderOptions,
  ): { buffer: Buffer; nativeScene: NativeRenderScene; nativeCamera: NativeCamera; objectIdEntries?: RenderObjectIdEntry[] } {
    const { nativeScene, nativeCamera, objectIdEntries } = toNativeInput(scene, camera, options)
    return { buffer: this.native.render(nativeScene, nativeCamera), nativeScene, nativeCamera, objectIdEntries }
  }

  private resolveRenderOptions(options: RenderOptions): RenderOptions {
    return {
      ...options,
      sortObjects: options.sortObjects ?? this.sortObjects,
      opaqueSort: options.opaqueSort === undefined ? this.opaqueSort : options.opaqueSort,
      transparentSort: options.transparentSort === undefined ? this.transparentSort : options.transparentSort,
    }
  }
}

export function render(scene: ThreeSceneRootLike, camera: ThreeRenderCameraLike, options: RenderOptions = {}): Buffer {
  assertRenderOptionsLike(options, 'options')
  if (isCubeCamera(camera)) {
    const { buffer } = renderCubeCamera(scene, camera, options, native.renderNative)
    return buffer
  }

  if (isArrayCamera(camera)) {
    const { buffer, width, height, objectIdEntries, depthData } = renderArrayCamera(scene, camera, options, native.renderNative)
    if (options.target) {
      writeRenderTarget(options.target, buffer, width, height, objectIdEntries, depthData)
    }
    return buffer
  }

  const { nativeScene, nativeCamera, objectIdEntries } = toNativeInput(scene, camera, options)
  const buffer = native.renderNative(nativeScene, nativeCamera)
  if (options.target) {
    const depthData = renderTargetDepthBuffer(options.target, nativeScene, nativeCamera, native.renderNative)
    writeRenderTarget(options.target, buffer, nativeScene.width!, nativeScene.height!, objectIdEntries, depthData)
  }
  return buffer
}

export function renderToTarget(
  scene: ThreeSceneRootLike,
  camera: ThreeRenderCameraLike,
  target: RenderTargetLike = {},
  options: RenderOptions = {},
): RenderTargetLike {
  assertRenderTargetLike(target, 'target')
  assertRenderOptionsLike(options, 'options')
  const targetOptions: RenderOptions = { ...options, target, format: options.format ?? 'rgba' }
  if (isCubeCamera(camera)) {
    const { target: cubeTarget } = renderCubeCamera(scene, camera, targetOptions, native.renderNative)
    return cubeTarget
  }

  if (isArrayCamera(camera)) {
    const { buffer, width, height, objectIdEntries, depthData } = renderArrayCamera(scene, camera, targetOptions, native.renderNative)
    return writeRenderTarget(target, buffer, width, height, objectIdEntries, depthData)
  }

  const { nativeScene, nativeCamera, objectIdEntries } = toNativeInput(scene, camera, targetOptions)
  const buffer = native.renderNative(nativeScene, nativeCamera)
  const depthData = renderTargetDepthBuffer(target, nativeScene, nativeCamera, native.renderNative)
  return writeRenderTarget(target, buffer, nativeScene.width!, nativeScene.height!, objectIdEntries, depthData)
}

function toNativeInput(
  scene: ThreeSceneRootLike,
  camera: ThreeCameraLike,
  options: RenderOptions,
): { nativeScene: NativeRenderScene; nativeCamera: NativeCamera; objectIdEntries?: RenderObjectIdEntry[] } {
  validateThreeSceneRoot(scene)
  validateThreeCamera(camera)
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
  const environment = colorMode ? resolveEnvironmentMap(scene, options.environmentIntensity) : { envMap: null }
  const envMap = environment.envMap
  const hasEnvironmentRotationOverride = options.environmentRotation !== undefined
  const environmentRotation = environment.rotation ?? (
    hasEnvironmentRotationOverride ? options.environmentRotation : scene.environmentRotation
  )
  const environmentRotationLabel = environment.rotation
    ? 'material.envMapRotation'
    : hasEnvironmentRotationOverride ? 'options.environmentRotation' : 'scene.environmentRotation'
  const environmentMapRotation = colorMode
    ? environmentRotationToNative(environmentRotation, envMap, environmentRotationLabel)
    : undefined
  const hasBackgroundOverride = options.background !== undefined
  const optionBackgroundTexture = colorMode && hasBackgroundOverride
    ? extractBackgroundTexture(options.background, 'options.background')
    : null
  const backgroundTexture = colorMode
    ? optionBackgroundTexture ?? (hasBackgroundOverride ? null : extractBackgroundTexture(scene.background, 'scene.background'))
    : null
  const hasBackgroundRotationOverride = options.backgroundRotation !== undefined
  const backgroundRotation = hasBackgroundOverride
    ? options.backgroundRotation
    : hasBackgroundRotationOverride ? options.backgroundRotation : scene.backgroundRotation
  const backgroundTextureRotation = colorMode
    ? backgroundRotationToNative(
      backgroundRotation,
      backgroundTexture,
      hasBackgroundOverride || hasBackgroundRotationOverride ? 'options.backgroundRotation' : 'scene.backgroundRotation',
    )
    : undefined
  const backgroundTextureBlurriness = colorMode && backgroundTexture
    ? optionalFiniteNumber(
      hasBackgroundOverride ? options.backgroundBlurriness : options.backgroundBlurriness ?? scene.backgroundBlurriness,
      hasBackgroundOverride || options.backgroundBlurriness !== undefined ? 'options.backgroundBlurriness' : 'scene.backgroundBlurriness',
    )
    : undefined
  const backgroundIntensity = colorMode
    ? optionalFiniteNumber(
      hasBackgroundOverride ? options.backgroundIntensity : options.backgroundIntensity ?? scene.backgroundIntensity,
      hasBackgroundOverride || options.backgroundIntensity !== undefined ? 'options.backgroundIntensity' : 'scene.backgroundIntensity',
    )
    : undefined
  const clippingPlanes = extractClippingPlanes(options.clippingPlanes, 'options.clippingPlanes')
  const lights: NativeSceneLight[] | undefined = colorMode ? extractLights(scene, camera) : []
  const shadowMaterialMode = colorMode ? shadowMaterialModeForLights(lights) : undefined
  const flattenedMeshes = flattenScene(
    scene,
    camera,
    size.height,
    clippingPlanes,
    options.localClippingEnabled !== false,
    shadowMaterialMode,
    environment.materialContext,
    {
      sortObjects: options.sortObjects,
      opaqueSort: options.opaqueSort,
      transparentSort: options.transparentSort,
    },
  )
  const objectIdEntries = renderMode === 'object-id' ? objectIdEntriesForMeshes(flattenedMeshes) : undefined
  const meshes = applyRenderMode(flattenedMeshes, renderMode)
  const viewport = normalizeOptionalPixelRect(
    effectiveViewport(options),
    size.width,
    size.height,
    effectiveViewportLabel(options),
  )
  const scissor = normalizeOptionalPixelRect(
    effectiveScissor(options),
    size.width,
    size.height,
    effectiveScissorLabel(options),
  )
  const nativeScene: NativeRenderScene = {
    width: size.width,
    height: size.height,
    background: colorMode ? resolveBackground(scene, options, backgroundTexture != null) : [0, 0, 0, 1],
    backgroundIntensity,
    viewport: pixelRectToArray(viewport),
    scissor: pixelRectToArray(scissor),
    backgroundTexture: backgroundTexture?.data,
    backgroundTextureWidth: backgroundTexture?.width,
    backgroundTextureHeight: backgroundTexture?.height,
    backgroundTextureWrapS: backgroundTexture?.wrapS,
    backgroundTextureWrapT: backgroundTexture?.wrapT,
    backgroundTextureMagFilter: backgroundTexture?.magFilter,
    backgroundTextureMinFilter: backgroundTexture?.minFilter,
    backgroundTextureAnisotropy: backgroundTexture?.anisotropy,
    backgroundTextureTransform: backgroundTexture?.transform,
    backgroundTextureColorSpace: backgroundTexture?.colorSpace,
    backgroundTextureMapping: backgroundTexture?.mapping,
    backgroundTextureRotation,
    backgroundTextureBlurriness,
    format: options.format ?? (options.target ? 'rgba' : 'png'),
    outputColorSpace: options.outputColorSpace,
    sampleCount: resolveSampleCount(options),
    meshes,
    lights,
    ambientLight: colorMode ? extractAmbientLight(scene, camera) ?? undefined : undefined,
    ambientIntensity: colorMode ? extractAmbientIntensity(scene, camera) ?? undefined : undefined,
    lightProbe: colorMode ? extractLightProbe(scene, camera) ?? undefined : undefined,
    environmentMap: envMap?.data,
    environmentMapWidth: envMap?.width,
    environmentMapHeight: envMap?.height,
    environmentMapIntensity: envMap?.intensity,
    environmentMapColorSpace: envMap?.colorSpace,
    environmentMapRotation,
    ...(colorMode ? fogToNative(scene.fog) : {}),
    ...(colorMode ? postProcessingToNative(options.postProcessing) : {}),
  }
  const nativeCamera: NativeCamera = {
    width: size.width,
    height: size.height,
    near: optionalFiniteNumber(camera.near, 'camera.near'),
    far: optionalFiniteNumber(camera.far, 'camera.far'),
    viewProjection: cameraViewProjection(camera),
    viewMatrix: cameraViewMatrix(camera),
    cameraPosition: cameraWorldPosition(camera),
  }

  return { nativeScene, nativeCamera, objectIdEntries }
}

function normalizedRenderMode(mode: RenderOptions['renderMode']): RenderMode {
  if (mode == null) return 'color'
  if (mode === 'color' || mode === 'mask' || mode === 'object-id') return mode
  throw new TypeError(
    `options.renderMode must be "color", "mask", or "object-id"; received ${String(mode)}`,
  )
}

function shadowMaterialModeForLights(lights: NativeSceneLight[] | undefined): ShadowMaterialMode | undefined {
  const shadowLight = lights?.find((light) => light.castShadow === true)
  if (!shadowLight) return undefined
  return shadowLight.lightType === 'point' ? 'distance' : 'depth'
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
    alphaToCoverage: mesh.alphaToCoverage,
    premultipliedAlpha: mesh.premultipliedAlpha,
    clippingPlanes: mesh.clippingPlanes,
    clippingUnionCount: mesh.clippingUnionCount,
    blending: 'none',
    depthTest: mesh.depthTest,
    depthFunc: mesh.depthFunc,
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
    materialVariant: mesh.materialVariant,
    materialSortKey: mesh.materialSortKey,
  }
}

function materialAlpha(mesh: NativeSceneMesh): number {
  const alpha = mesh.color?.[3]
  return typeof alpha === 'number' && Number.isFinite(alpha) ? Math.min(1, Math.max(0, alpha)) : 1
}

function objectIdColor(mesh: NativeSceneMesh, index: number): Color4 {
  const value = encodedObjectId(mesh, index)
  return [
    ((value >> 16) & 0xff) / 255,
    ((value >> 8) & 0xff) / 255,
    (value & 0xff) / 255,
    materialAlpha(mesh),
  ]
}

function objectIdEntriesForMeshes(meshes: NativeSceneMesh[]): RenderObjectIdEntry[] {
  const entries = new Map<number, RenderObjectIdEntry>()
  meshes.forEach((mesh, index) => {
    const id = objectSortId(mesh, index)
    const encodedId = encodedObjectId(mesh, index)
    if (entries.has(encodedId)) return
    entries.set(encodedId, {
      id,
      encodedId,
      rgb: [
        (encodedId >> 16) & 0xff,
        (encodedId >> 8) & 0xff,
        encodedId & 0xff,
      ],
      hex: `#${encodedId.toString(16).padStart(6, '0')}`,
    })
  })
  return [...entries.values()].sort((a, b) => a.encodedId - b.encodedId)
}

function encodedObjectId(mesh: NativeSceneMesh, index: number): number {
  const encoded = (objectSortId(mesh, index) + 1) & 0xffffff
  return encoded === 0 ? 1 : encoded
}

function objectSortId(mesh: NativeSceneMesh, index: number): number {
  return typeof mesh.sortIndex === 'number' && Number.isSafeInteger(mesh.sortIndex) && mesh.sortIndex >= 0
    ? mesh.sortIndex
    : index
}

function renderModeFragment(color: Color4): string {
  return `return vec4<f32>(${formatWgslFloat(color[0])}, ${formatWgslFloat(color[1])}, ${formatWgslFloat(color[2])}, 1.0);`
}

const DEPTH_READBACK_FRAGMENT = [
  'let frag_depth = clamp(input.position.z, 0.0, 1.0);',
  'let depth = 1.0 - frag_depth;',
  'return vec4<f32>(depth, depth, depth, 1.0);',
].join('\n')

function renderTargetDepthBuffer(
  target: RenderTargetLike | undefined,
  nativeScene: NativeRenderScene,
  nativeCamera: NativeCamera,
  renderNativeScene: (scene: NativeRenderScene, camera: NativeCamera) => Buffer,
): Buffer | undefined {
  if (target?.depthTexture == null) return undefined
  return renderNativeScene(depthReadbackScene(nativeScene), nativeCamera)
}

type RenderNativeScene = (scene: NativeRenderScene, camera: NativeCamera) => Buffer

type PixelRect = {
  x: number
  y: number
  width: number
  height: number
}

const WEBGL_COORDINATE_SYSTEM = 2000
const CUBE_FACE_COUNT = 6
const UnsignedByteType = 1009
const UnsignedShortType = 1012
const UnsignedIntType = 1014
const FloatType = 1015
const HalfFloatType = 1016
const UnsignedInt248Type = 1020
const RGBAFormat = 1023
const DepthFormat = 1026
const DepthStencilFormat = 1027

function renderCubeCamera(
  scene: ThreeSceneRootLike,
  camera: ThreeCubeCameraLike,
  options: RenderOptions,
  renderNativeScene: RenderNativeScene,
): { buffer: Buffer; target: RenderTargetLike; width: number; height: number; faces: Buffer[] } {
  validateThreeSceneRoot(scene)
  validateCubeCamera(camera, options)
  const target = options.target ?? camera.renderTarget
  if (!target) {
    throw new Error('THREE.CubeCamera rendering requires a WebGLCubeRenderTarget via camera.renderTarget or options.target.')
  }
  validateUnsupportedRenderTargetOptions(target)

  const { width: targetWidth, height: targetHeight } = resolveCubeTargetSize(target, options)
  const activeMipmapLevel = resolveCubeMipmapLevel(camera, targetWidth)
  const { width, height } = cubeMipmapSize(targetWidth, targetHeight, activeMipmapLevel)
  const outputFormat = options.format ?? (options.target ? 'rgba' : 'png')
  const subCameras = cubeSubCameras(camera)
  const faceOptions: RenderOptions = {
    ...options,
    target,
    width,
    height,
    format: 'rgba',
    viewport: cubeMipmapViewport(options, target, activeMipmapLevel),
    scissor: cubeMipmapScissor(options, target, activeMipmapLevel),
  }
  const faces: Buffer[] = []
  const depthFaces: NonNullable<RenderTargetImageLike['data']>[] = []
  for (const subCamera of subCameras) {
    const { nativeScene, nativeCamera } = toNativeInput(scene, subCamera, faceOptions)
    faces.push(Buffer.from(renderNativeScene(nativeScene, nativeCamera)))
    const depthFace = renderTargetDepthBuffer(target, nativeScene, nativeCamera, renderNativeScene)
    if (depthFace) {
      depthFaces.push(cloneTargetData(depthTextureData(target.depthTexture!, depthFace)))
    }
  }

  writeCubeRenderTarget(
    target,
    faces,
    targetWidth,
    targetHeight,
    width,
    height,
    activeMipmapLevel,
    depthFaces.length > 0 ? depthFaces : undefined,
  )

  const buffer = outputFormat === 'png' ? native.encodePng(faces[0], width, height) : faces[0]
  return { buffer, target, width, height, faces }
}

function validateCubeCamera(camera: ThreeCubeCameraLike, options: RenderOptions): void {
  if (!isCubeCamera(camera)) {
    throw new TypeError('render(scene, camera) expected a THREE.CubeCamera-compatible object.')
  }
  assertSupportedOutputFormat(options.format, 'options.format')
}

function cubeSubCameras(camera: ThreeCubeCameraLike): ThreeCameraLike[] {
  if (typeof camera.updateCoordinateSystem === 'function' && camera.coordinateSystem !== WEBGL_COORDINATE_SYSTEM) {
    camera.coordinateSystem = WEBGL_COORDINATE_SYSTEM
    camera.updateCoordinateSystem()
  }
  if (typeof camera.updateMatrixWorld === 'function') {
    camera.updateMatrixWorld(true)
  }

  const children = camera.children
  if (!Array.isArray(children) || children.length < CUBE_FACE_COUNT) {
    throw new Error('THREE.CubeCamera requires six internal perspective cameras.')
  }
  const subCameras = children.slice(0, CUBE_FACE_COUNT)
  for (const subCamera of subCameras) {
    validateThreeCamera(subCamera)
    if (typeof subCamera.updateMatrixWorld === 'function') {
      subCamera.updateMatrixWorld(true)
    }
  }
  return subCameras
}

function resolveCubeTargetSize(target: RenderTargetLike, options: RenderOptions): { width: number; height: number } {
  const texture = cubeTargetTexture(target)
  const firstImage = Array.isArray(texture?.image) ? texture.image[0] : undefined
  const width = options.width ?? target.width ?? firstImage?.width
  const height = options.height ?? target.height ?? firstImage?.height ?? width
  if (!Number.isInteger(width) || width! <= 0) {
    throw new TypeError('THREE.CubeCamera target width must be a positive integer.')
  }
  if (!Number.isInteger(height) || height! <= 0) {
    throw new TypeError('THREE.CubeCamera target height must be a positive integer.')
  }
  if (width !== height) {
    throw new TypeError('THREE.CubeCamera target faces must be square.')
  }
  return { width: width!, height: height! }
}

function resolveCubeMipmapLevel(camera: ThreeCubeCameraLike, targetSize: number): number {
  const level = camera.activeMipmapLevel ?? 0
  if (!Number.isInteger(level) || level < 0) {
    throw new TypeError(`THREE.CubeCamera activeMipmapLevel must be a non-negative integer; received ${String(level)}.`)
  }
  const maxLevel = Math.floor(Math.log2(targetSize))
  if (level > maxLevel) {
    throw new Error(
      `THREE.CubeCamera activeMipmapLevel ${level} exceeds the maximum mip level ${maxLevel} for a ${targetSize}x${targetSize} cube target.`,
    )
  }
  return level
}

function cubeMipmapSize(width: number, height: number, activeMipmapLevel: number): { width: number; height: number } {
  if (activeMipmapLevel === 0) return { width, height }
  return {
    width: Math.max(1, width >> activeMipmapLevel),
    height: Math.max(1, height >> activeMipmapLevel),
  }
}

function cubeMipmapViewport(
  options: RenderOptions,
  target: RenderTargetLike,
  activeMipmapLevel: number,
): RenderPixelRectLike | null | undefined {
  if (options.viewport !== undefined) return options.viewport
  return cubeMipmapRect(target.viewport, activeMipmapLevel)
}

function cubeMipmapScissor(
  options: RenderOptions,
  target: RenderTargetLike,
  activeMipmapLevel: number,
): RenderPixelRectLike | null | undefined {
  if (options.scissor !== undefined) return options.scissor
  return target.scissorTest === true ? cubeMipmapRect(target.scissor, activeMipmapLevel) : undefined
}

function cubeMipmapRect(rect: RenderPixelRectLike | null | undefined, activeMipmapLevel: number): RenderPixelRectLike | null | undefined {
  if (!rect || activeMipmapLevel === 0) return rect
  const [x, y, width, height] = pixelRectComponents(rect)
  if (![x, y, width, height].every((value) => typeof value === 'number' && Number.isFinite(value))) {
    return { x, y, width, height }
  }
  return {
    x,
    y,
    width: Math.max(1, Math.floor(width / 2 ** activeMipmapLevel)),
    height: Math.max(1, Math.floor(height / 2 ** activeMipmapLevel)),
  }
}

function writeCubeRenderTarget(
  target: RenderTargetLike,
  faces: Buffer[],
  targetWidth: number,
  targetHeight: number,
  faceWidth: number,
  faceHeight: number,
  activeMipmapLevel: number,
  depthFaces?: NonNullable<RenderTargetImageLike['data']>[],
): RenderTargetLike {
  if (faces.length !== CUBE_FACE_COUNT) {
    throw new Error(`THREE.CubeCamera expected ${CUBE_FACE_COUNT} rendered faces, received ${faces.length}.`)
  }
  target.width = targetWidth
  target.height = targetHeight
  target.data = faces[0]

  const texture = ensureCubeTargetTexture(target)
  texture.isCubeTexture = true
  writeCubeTextureFaces(texture, faces, faceWidth, faceHeight, activeMipmapLevel)
  texture.needsPMREMUpdate = true
  if (target.depthTexture && depthFaces) {
    if (depthFaces.length !== CUBE_FACE_COUNT) {
      throw new Error(`THREE.CubeCamera expected ${CUBE_FACE_COUNT} rendered depth faces, received ${depthFaces.length}.`)
    }
    writeCubeTextureFaces(target.depthTexture, depthFaces, faceWidth, faceHeight, activeMipmapLevel)
  }
  return target
}

function writeCubeTextureFaces(
  texture: RenderTargetTextureLike,
  faces: NonNullable<RenderTargetImageLike['data']>[],
  width: number,
  height: number,
  activeMipmapLevel: number,
): void {
  const images = faces.map((data) => ({ data, width, height, depth: 1 }))
  if (activeMipmapLevel === 0) {
    texture.image = images
    texture.source ??= {}
    texture.source.data = images
  } else {
    const mipmaps = texture.mipmaps ?? (texture.mipmaps = [])
    for (let level = 0; level <= activeMipmapLevel; level += 1) {
      mipmaps[level] ??= {}
    }
    const mipmap = mipmaps[activeMipmapLevel]
    mipmap.image = images
    mipmap.width = width
    mipmap.height = height
    mipmap.depth = 1
  }
  texture.needsUpdate = true
}

function cubeTargetTexture(target: RenderTargetLike): RenderTargetTextureLike | undefined {
  return Array.isArray(target.texture)
    ? target.texture[0]
    : target.texture ?? target.textures?.[0]
}

function ensureCubeTargetTexture(target: RenderTargetLike): RenderTargetTextureLike {
  const texture = cubeTargetTexture(target)
  if (texture) return texture
  const images = Array.from({ length: CUBE_FACE_COUNT }, () => ({}))
  const created: RenderTargetTextureLike = { image: images, source: { data: images }, isCubeTexture: true }
  target.texture = created
  return created
}

function renderArrayCamera(
  scene: ThreeSceneRootLike,
  camera: ThreeCameraLike,
  options: RenderOptions,
  renderNativeScene: RenderNativeScene,
): { buffer: Buffer; width: number; height: number; objectIdEntries?: RenderObjectIdEntry[]; depthData?: Buffer } {
  validateThreeSceneRoot(scene)
  validateArrayCameraOutput(camera, options)
  validateUnsupportedRenderOptions(options)

  const size = resolveSize(camera, options)
  const subCameras = arraySubCameras(camera)
  const outputFormat = options.format ?? (options.target ? 'rgba' : 'png')
  const objectIdEntryMap = new Map<number, RenderObjectIdEntry>()
  let colorBuffer: Buffer | undefined
  let depthBuffer: Buffer | undefined

  for (const subCamera of subCameras) {
    const viewport = resolveSubCameraViewport(subCamera, options.viewport, size.width, size.height)
    const copyRect = viewport ?? { x: 0, y: 0, width: size.width, height: size.height }
    const subOptions: RenderOptions = {
      ...options,
      width: size.width,
      height: size.height,
      format: 'rgba',
      viewport: viewport ?? undefined,
    }
    const { nativeScene, nativeCamera, objectIdEntries } = toNativeInput(scene, subCamera, subOptions)
    const subBuffer = renderNativeScene(nativeScene, nativeCamera)
    if (colorBuffer == null) {
      colorBuffer = Buffer.from(subBuffer)
    } else {
      copyPixelRect(subBuffer, colorBuffer, size.width, copyRect)
    }

    if (objectIdEntries) {
      for (const entry of objectIdEntries) {
        objectIdEntryMap.set(entry.encodedId, entry)
      }
    }

    const subDepth = renderTargetDepthBuffer(options.target, nativeScene, nativeCamera, renderNativeScene)
    if (subDepth) {
      if (depthBuffer == null) {
        depthBuffer = Buffer.from(subDepth)
      } else {
        copyPixelRect(subDepth, depthBuffer, size.width, copyRect)
      }
    }
  }

  return {
    buffer: outputFormat === 'png' ? native.encodePng(colorBuffer!, size.width, size.height) : colorBuffer!,
    width: size.width,
    height: size.height,
    objectIdEntries: objectIdEntryMap.size > 0
      ? [...objectIdEntryMap.values()].sort((a, b) => a.encodedId - b.encodedId)
      : undefined,
    depthData: depthBuffer,
  }
}

function validateArrayCameraOutput(camera: ThreeCameraLike, options: RenderOptions): void {
  const cameraLike = camera as any
  if (cameraLike?.isCubeCamera === true || cameraLike?.type === 'CubeCamera') {
    throw new Error(
      'THREE.CubeCamera cannot be used as an ArrayCamera sub-camera. Pass the CubeCamera as the top-level camera with a cube render target.',
    )
  }
  if (!camera || cameraLike.isCamera !== true) {
    throw new TypeError('render(scene, camera) expects camera to be a THREE.Camera')
  }
  assertSupportedOutputFormat(options.format, 'options.format')
}

function arraySubCameras(camera: ThreeCameraLike): ThreeCameraLike[] {
  const cameras = (camera as any).cameras
  if (!Array.isArray(cameras) || cameras.length === 0) {
    throw new Error('THREE.ArrayCamera requires at least one sub-camera in camera.cameras.')
  }
  for (const subCamera of cameras) {
    validateThreeCamera(subCamera)
  }
  return cameras
}

function resolveSubCameraViewport(
  camera: ThreeCameraLike,
  fallback: RenderPixelRectLike | null | undefined,
  width: number,
  height: number,
): PixelRect | undefined {
  const viewport = cameraViewport(camera) ?? fallback
  return viewport ? normalizePixelRect(viewport, width, height, 'THREE.ArrayCamera sub-camera viewport') : undefined
}

function cameraViewport(camera: ThreeCameraLike): RenderPixelRectLike | undefined {
  const viewport = camera.viewport as any
  if (viewport == null) return undefined
  if (typeof viewport.length === 'number') {
    return [viewport[0], viewport[1], viewport[2], viewport[3]]
  }
  return {
    x: viewport.x,
    y: viewport.y,
    width: viewport.width ?? viewport.z,
    height: viewport.height ?? viewport.w,
  }
}

function normalizePixelRect(rect: RenderPixelRectLike, targetWidth: number, targetHeight: number, label: string): PixelRect {
  const [rawX, rawY, rawWidth, rawHeight] = pixelRectComponents(rect)
  if (![rawX, rawY, rawWidth, rawHeight].every((value) => typeof value === 'number' && Number.isFinite(value))) {
    throw new TypeError(`${label} must contain finite x, y, width, and height values.`)
  }
  const x = Math.round(rawX)
  const y = Math.round(rawY)
  const width = Math.round(rawWidth)
  const height = Math.round(rawHeight)
  if (x < 0 || y < 0) {
    throw new TypeError(`${label} x and y must be greater than or equal to 0.`)
  }
  if (width <= 0 || height <= 0) {
    throw new TypeError(`${label} width and height must be greater than 0.`)
  }
  if (x + width > targetWidth || y + height > targetHeight) {
    throw new TypeError(`${label} must fit inside the render target.`)
  }
  return { x, y, width, height }
}

function normalizeOptionalPixelRect(
  rect: RenderPixelRectLike | null | undefined,
  targetWidth: number,
  targetHeight: number,
  label: string,
): PixelRect | undefined {
  if (rect == null) return undefined
  return normalizePixelRect(rect, targetWidth, targetHeight, label)
}

function copyPixelRect(source: Buffer, destination: Buffer, imageWidth: number, rect: PixelRect): void {
  const rowBytes = rect.width * 4
  for (let row = 0; row < rect.height; row += 1) {
    const offset = ((rect.y + row) * imageWidth + rect.x) * 4
    source.copy(destination, offset, offset, offset + rowBytes)
  }
}

function depthReadbackScene(scene: NativeRenderScene): NativeRenderScene {
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

function depthReadbackMesh(mesh: NativeSceneMesh): NativeSceneMesh {
  const writesDepth = meshWritesDepth(mesh)
  return {
    ...mesh,
    blending: 'none',
    depthWrite: writesDepth,
    colorWrite: writesDepth,
    transparent: false,
    shadingModel: 'basic',
    alphaToCoverage: false,
    customFragmentShader: DEPTH_READBACK_FRAGMENT,
    castShadow: false,
    receiveShadow: false,
  }
}

function meshWritesDepth(mesh: NativeSceneMesh): boolean {
  if (mesh.depthTest === false) return false
  if (typeof mesh.depthWrite === 'boolean') return mesh.depthWrite
  return true
}

function formatWgslFloat(value: number): string {
  if (value <= 0) return '0.0'
  if (value >= 1) return '1.0'
  return value.toFixed(10)
}

function fogToNative(fog: ThreeSceneRootLike['fog']): Partial<NativeRenderScene> {
  if (!fog) return {}
  const color = strictColorLikeToArray(fog.color, 'scene.fog.color')
  if (fog.isFogExp2) {
    return {
      fogType: 'exp2',
      fogColor: color ?? undefined,
      fogDensity: optionalFiniteNumber(fog.density, 'scene.fog.density'),
    }
  }
  if (fog.isFog) {
    return {
      fogType: 'linear',
      fogColor: color ?? undefined,
      fogNear: optionalFiniteNumber(fog.near, 'scene.fog.near'),
      fogFar: optionalFiniteNumber(fog.far, 'scene.fog.far'),
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
  return pixelRectComponents(rect)
}

function effectiveViewport(options: RenderOptions): RenderPixelRectLike | null | undefined {
  return options.viewport !== undefined ? options.viewport : options.target?.viewport
}

function effectiveScissor(options: RenderOptions): RenderPixelRectLike | null | undefined {
  if (options.scissor !== undefined) return options.scissor
  return options.target?.scissorTest === true ? options.target.scissor : undefined
}

function effectiveViewportLabel(options: RenderOptions): string {
  return options.viewport !== undefined ? 'options.viewport' : 'target.viewport'
}

function effectiveScissorLabel(options: RenderOptions): string {
  return options.scissor !== undefined ? 'options.scissor' : 'target.scissor'
}

function pixelRectComponents(rect: RenderPixelRectLike): number[] {
  if (typeof (rect as ArrayLike<number>).length === 'number') {
    const values = rect as ArrayLike<number>
    return [values[0], values[1], values[2], values[3]]
  }
  const values = rect as { x?: number; y?: number; width?: number; height?: number; z?: number; w?: number }
  return [values.x!, values.y!, values.width ?? values.z!, values.height ?? values.w!]
}

function finiteOrUndefined(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function optionalFiniteNumber(value: unknown, label: string): number | undefined {
  if (value == null) return undefined
  if (typeof value === 'number' && Number.isFinite(value)) return value
  throw new TypeError(`${label} must be a finite number.`)
}

function booleanOrNumber(value: unknown): number | undefined {
  if (typeof value === 'boolean') return value ? 1 : 0
  return finiteOrUndefined(value)
}

type EulerOrder = 'XYZ' | 'YXZ' | 'ZXY' | 'ZYX' | 'YZX' | 'XZY'
type EulerComponents = { x: number; y: number; z: number; order: EulerOrder }

function backgroundRotationToNative(
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

function environmentRotationToNative(
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

function optionalEulerComponents(value: ThreeEulerLike | ArrayLike<number> | null | undefined, label: string): EulerComponents | null {
  if (!value) return null
  return eulerComponents(value, label)
}

function eulerComponents(value: ThreeEulerLike | ArrayLike<number>, label: string): EulerComponents {
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

function finiteRotationComponent(value: unknown, label: string): number {
  if (value == null) return 0
  if (typeof value === 'number' && Number.isFinite(value)) return value
  throw new TypeError(`${label} must be a finite number`)
}

function eulerOrder(value: unknown, label: string): EulerOrder {
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

function eulerRotationMatrix3Columns(x: number, y: number, z: number, order: EulerOrder): number[] {
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

function hasNonZeroEulerRotation(rotation: EulerComponents): boolean {
  return Math.abs(rotation.x) > 1e-12 || Math.abs(rotation.y) > 1e-12 || Math.abs(rotation.z) > 1e-12
}

function validateUnsupportedRenderOptions(options: RenderOptions): void {
  assertSupportedOutputFormat(options.format, 'options.format')
  assertSupportedOutputColorSpace(options.outputColorSpace)
  assertFiniteNumberOption(options.backgroundIntensity, 'options.backgroundIntensity')
  assertFiniteNumberOption(options.backgroundBlurriness, 'options.backgroundBlurriness')
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
  if (Object.prototype.hasOwnProperty.call(options, 'target') && options.target !== undefined) {
    assertRenderTargetLike(options.target, 'options.target')
  }
  if (options.target) validateUnsupportedRenderTargetOptions(options.target)
}

function assertRenderOptionsLike(value: unknown, label: string): asserts value is RenderOptions {
  if (value == null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be an options object.`)
  }
}

function assertRenderTargetLike(value: unknown, label: string): asserts value is RenderTargetLike {
  if (value == null || typeof value !== 'object') {
    throw new TypeError(`${label} must be a target-like object.`)
  }
}

function assertEulerOption(value: unknown, label: string): void {
  if (value == null) return
  if (typeof value !== 'object') {
    throw new TypeError(`${label} must be a rotation object or array.`)
  }
  eulerComponents(value as ThreeEulerLike | ArrayLike<number>, label)
}

function validateSortControls(options: RenderOptions): void {
  if (options.sortObjects != null && typeof options.sortObjects !== 'boolean') {
    throw new TypeError(`options.sortObjects must be a boolean; received ${String(options.sortObjects)}.`)
  }
  if (options.opaqueSort != null && typeof options.opaqueSort !== 'function') {
    throw new TypeError('options.opaqueSort must be a function or null.')
  }
  if (options.transparentSort != null && typeof options.transparentSort !== 'function') {
    throw new TypeError('options.transparentSort must be a function or null.')
  }
}

function assertSortFunctionOrNull(value: unknown, label: string): asserts value is RenderSortFunction | null {
  if (value != null && typeof value !== 'function') {
    throw new TypeError(`${label} expects a function or null.`)
  }
}

function validateUnsupportedRenderTargetOptions(target: RenderTargetLike): void {
  if (target.isWebGLMultipleRenderTargets === true) {
    throw new Error(
      'Multiple render target color attachments are not supported by @headless-three/renderer yet. Render separate passes or use a single color target until MRT support lands.',
    )
  }
  if (Array.isArray(target.texture) && target.texture.length > 1) {
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
  assertSupportedRenderTargetColorTexture(renderTargetColorTexture(target))
  assertSupportedDepthTextureType(target.depthTexture)
  assertSupportedDepthTextureFormat(target.depthTexture)
}

function assertSupportedSampleCount(value: unknown, label: string): void {
  if (value == null) return
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || Math.floor(value) !== value) {
    throw new Error(
      `${label} must be a non-negative integer sample count; received ${String(value)}.`,
    )
  }
  if (value > 1 && value !== 4) {
    throw new Error(
      `MSAA sample count ${value} is not supported by @headless-three/renderer yet (${label}=${value}). Use 4 for 4x MSAA or the default single-sample render path.`,
    )
  }
}

function assertSupportedOutputFormat(value: unknown, label: string): void {
  if (value == null) return
  if (value === 'png' || value === 'rgba') return
  throw new Error(
    `${label} ${String(value)} is not supported by @headless-three/renderer. Use "png" or "rgba".`,
  )
}

function assertSupportedOutputColorSpace(value: unknown): void {
  if (value == null) return
  if (
    value === 'srgb' ||
    value === 'srgb-linear' ||
    value === 'linear-srgb' ||
    value === 'linearsrgb' ||
    value === 'linear'
  ) return
  throw new Error(
    `options.outputColorSpace ${String(value)} is not supported by @headless-three/renderer. Use THREE.SRGBColorSpace or THREE.LinearSRGBColorSpace.`,
  )
}

function validatePostProcessingOptions(value: unknown): void {
  if (value == null || value === false) return
  if (typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError('options.postProcessing must be an object.')
  }
  const post = value as RenderOptions['postProcessing']
  if (post?.enabled != null && typeof post.enabled !== 'boolean') {
    throw new TypeError('options.postProcessing.enabled must be a boolean.')
  }
  if (post?.enabled === false) return
  assertFinitePostProcessingNumber(post?.exposure, 'options.postProcessing.exposure')
  assertFinitePostProcessingNumber(post?.contrast, 'options.postProcessing.contrast')
  assertFinitePostProcessingNumber(post?.saturation, 'options.postProcessing.saturation')
  assertFinitePostProcessingNumber(post?.vignette, 'options.postProcessing.vignette')
  assertFinitePostProcessingBlend(post?.grayscale, 'options.postProcessing.grayscale')
  assertFinitePostProcessingBlend(post?.invert, 'options.postProcessing.invert')
}

function assertFinitePostProcessingNumber(value: unknown, label: string): void {
  assertFiniteNumberOption(value, label)
}

function assertFinitePostProcessingBlend(value: unknown, label: string): void {
  if (value == null || typeof value === 'boolean') return
  if (typeof value === 'number' && Number.isFinite(value)) return
  throw new TypeError(`${label} must be a finite number or boolean.`)
}

function assertFiniteNumberOption(value: unknown, label: string): void {
  if (value == null) return
  if (typeof value === 'number' && Number.isFinite(value)) return
  throw new TypeError(`${label} must be a finite number.`)
}

function assertSupportedRenderTargetColorTexture(texture: RenderTargetTextureLike | undefined): void {
  if (!texture) return
  const format = texture.format
  if (format != null && format !== RGBAFormat) {
    throw new Error(
      `target color texture format ${String(format)} is not supported by @headless-three/renderer yet. Use RGBAFormat or omit format for RGBA8 readback.`,
    )
  }
  const type = texture.type
  if (type != null && type !== UnsignedByteType) {
    throw new Error(
      `target color texture type ${String(type)} is not supported by @headless-three/renderer yet. Use UnsignedByteType or omit type for RGBA8 readback.`,
    )
  }
}

function assertSupportedDepthTextureType(depthTexture: RenderTargetTextureLike | undefined): void {
  const type = depthTexture?.type
  if (type == null) return
  if (
    type === UnsignedByteType ||
    type === UnsignedShortType ||
    type === UnsignedIntType ||
    type === FloatType ||
    type === HalfFloatType ||
    type === UnsignedInt248Type
  ) return
  throw new Error(
    `target.depthTexture.type ${String(type)} is not supported by @headless-three/renderer yet. Use FloatType, HalfFloatType, UnsignedByteType, UnsignedShortType, UnsignedIntType, UnsignedInt248Type, or omit type for RGBA8 normalized depth readback.`,
  )
}

function assertSupportedDepthTextureFormat(depthTexture: RenderTargetTextureLike | undefined): void {
  const format = depthTexture?.format
  if (format == null) return
  if (format === DepthFormat) {
    if (depthTexture?.type === UnsignedInt248Type) {
      throw new Error(
        'target.depthTexture.format DepthFormat is not supported with UnsignedInt248Type by @headless-three/renderer. Use DepthStencilFormat with UnsignedInt248Type, or use DepthFormat with a scalar depth texture type.',
      )
    }
    return
  }
  if (format === DepthStencilFormat) {
    if (depthTexture?.type === UnsignedInt248Type) return
    throw new Error(
      'target.depthTexture.format DepthStencilFormat is only supported with UnsignedInt248Type by @headless-three/renderer. Use DepthFormat for scalar depth readback, or set type to UnsignedInt248Type for packed depth24-stencil8 readback.',
    )
  }
  throw new Error(
    `target.depthTexture.format ${String(format)} is not supported by @headless-three/renderer yet. Use DepthFormat, or DepthStencilFormat with UnsignedInt248Type.`,
  )
}

function resolveSampleCount(options: RenderOptions): number {
  const requested = options.target?.sampleCount
    ?? options.target?.samples
    ?? options.sampleCount
    ?? options.samples
    ?? 1
  return requested > 1 ? requested : 1
}

function writeRenderTarget(
  target: RenderTargetLike,
  data: Buffer,
  width: number,
  height: number,
  objectIdEntries?: RenderObjectIdEntry[],
  depthData?: Buffer,
): RenderTargetLike {
  target.width = width
  target.height = height
  target.data = data

  const image = target.image ?? (target.image = {})
  image.data = data
  image.width = width
  image.height = height

  const texture = renderTargetColorTexture(target)
  if (texture) {
    writeRenderTargetTexture(texture, data, width, height)
  }

  if (target.depthTexture != null && depthData) {
    writeRenderTargetTexture(target.depthTexture, depthTextureData(target.depthTexture, depthData), width, height)
  }

  if (objectIdEntries) {
    target.objectIdEntries = objectIdEntries
    target.objectIdMap = Object.fromEntries(objectIdEntries.map((entry) => [String(entry.encodedId), entry]))
  } else {
    delete target.objectIdEntries
    delete target.objectIdMap
  }

  return target
}

function renderTargetColorTexture(target: RenderTargetLike): RenderTargetTextureLike | undefined {
  return Array.isArray(target.texture)
    ? target.texture[0]
    : target.texture ?? target.textures?.[0]
}

function writeRenderTargetTexture(
  texture: RenderTargetTextureLike,
  data: NonNullable<RenderTargetImageLike['data']>,
  width: number,
  height: number,
): void {
  const textureImage = Array.isArray(texture.image)
    ? texture.image[0] ?? (texture.image[0] = {})
    : texture.image ?? (texture.image = {})
  textureImage.data = data
  textureImage.width = width
  textureImage.height = height

  if (texture.source?.data) {
    const sourceData = Array.isArray(texture.source.data)
      ? texture.source.data[0] ?? (texture.source.data[0] = {})
      : texture.source.data
    sourceData.data = data
    sourceData.width = width
    sourceData.height = height
  }
}

function depthTextureData(texture: RenderTargetTextureLike, rgbaDepth: Buffer): NonNullable<RenderTargetImageLike['data']> {
  if (texture.type === UnsignedByteType) {
    const depth = new Uint8Array(rgbaDepth.length / 4)
    for (let i = 0, p = 0; i < rgbaDepth.length; i += 4, p += 1) {
      depth[p] = rgbaDepth[i]
    }
    return depth
  }
  if (texture.type === UnsignedShortType) {
    const depth = new Uint16Array(rgbaDepth.length / 4)
    for (let i = 0, p = 0; i < rgbaDepth.length; i += 4, p += 1) {
      depth[p] = Math.round((rgbaDepth[i] / 255) * 0xffff)
    }
    return depth
  }
  if (texture.type === UnsignedIntType) {
    const depth = new Uint32Array(rgbaDepth.length / 4)
    for (let i = 0, p = 0; i < rgbaDepth.length; i += 4, p += 1) {
      depth[p] = Math.round((rgbaDepth[i] / 255) * 0xffffffff)
    }
    return depth
  }
  if (texture.type === UnsignedInt248Type) {
    const depth = new Uint32Array(rgbaDepth.length / 4)
    for (let i = 0, p = 0; i < rgbaDepth.length; i += 4, p += 1) {
      depth[p] = Math.round((rgbaDepth[i] / 255) * 0xffffff) * 0x100
    }
    return depth
  }
  if (texture.type === FloatType) {
    const depth = new Float32Array(rgbaDepth.length / 4)
    for (let i = 0, p = 0; i < rgbaDepth.length; i += 4, p += 1) {
      depth[p] = rgbaDepth[i] / 255
    }
    return depth
  }
  if (texture.type === HalfFloatType) {
    const depth = new Uint16Array(rgbaDepth.length / 4)
    for (let i = 0, p = 0; i < rgbaDepth.length; i += 4, p += 1) {
      depth[p] = normalizedFloatToHalf(rgbaDepth[i] / 255)
    }
    return depth
  }
  return rgbaDepth
}

function normalizedFloatToHalf(value: number): number {
  const clamped = Math.min(1, Math.max(0, value))
  if (clamped === 0) return 0
  if (clamped === 1) return 0x3c00

  const exponent = Math.floor(Math.log2(clamped))
  if (exponent < -14) {
    return Math.round(clamped * 0x1000000)
  }

  let mantissa = Math.round((clamped / (2 ** exponent) - 1) * 0x400)
  let biasedExponent = exponent + 15
  if (mantissa === 0x400) {
    mantissa = 0
    biasedExponent += 1
  }
  return (biasedExponent << 10) | mantissa
}

function cloneTargetData(data: NonNullable<RenderTargetImageLike['data']>): NonNullable<RenderTargetImageLike['data']> {
  if (Buffer.isBuffer(data)) return Buffer.from(data)
  if (data instanceof Float32Array) return new Float32Array(data)
  if (data instanceof Uint32Array) return new Uint32Array(data)
  if (data instanceof Uint16Array) return new Uint16Array(data)
  if (data instanceof Uint8ClampedArray) return new Uint8ClampedArray(data)
  return new Uint8Array(data)
}

function validateThreeSceneRoot(scene: unknown): asserts scene is ThreeSceneRootLike {
  const root = scene as any
  if (!root || (root.isScene !== true && root.isObject3D !== true)) {
    throw new TypeError('render(scene, camera) expects scene to be a THREE.Scene or THREE.Object3D root')
  }
}

function isArrayCamera(camera: unknown): camera is ThreeCameraLike {
  const cameraLike = camera as any
  return cameraLike?.isArrayCamera === true || Array.isArray(cameraLike?.cameras)
}

function isCubeCamera(camera: unknown): camera is ThreeCubeCameraLike {
  const cameraLike = camera as any
  return cameraLike?.isCubeCamera === true || cameraLike?.type === 'CubeCamera'
}

function validateThreeCamera(camera: unknown): asserts camera is ThreeCameraLike {
  const cameraLike = camera as any
  if (cameraLike?.isCubeCamera === true || cameraLike?.type === 'CubeCamera') {
    throw new Error(
      'THREE.CubeCamera cannot be used where a regular THREE.Camera is required. Pass the CubeCamera as the top-level camera with a cube render target.',
    )
  }
  if (!camera || cameraLike.isCamera !== true) {
    throw new TypeError('render(scene, camera) expects camera to be a THREE.Camera')
  }
  if (cameraLike.isArrayCamera === true || Array.isArray(cameraLike.cameras)) {
    throw new Error(
      'THREE.ArrayCamera cannot be used where a regular THREE.Camera is required. Pass the ArrayCamera as the top-level camera.',
    )
  }
  if (!cameraLike.projectionMatrix || !cameraLike.matrixWorldInverse) {
    throw new TypeError('THREE.Camera must have projectionMatrix and matrixWorldInverse')
  }
}
