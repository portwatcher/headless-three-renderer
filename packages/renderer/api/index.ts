import type {
  ThreeSceneRootLike,
  ThreeCameraLike,
  RenderOptions,
  RenderTargetLike,
  RenderTargetTextureLike,
  RenderPixelRectLike,
  NativeRenderScene,
  NativeCamera,
  NativeSceneMesh,
  NativeSceneLight,
  RenderMode,
  Color4,
  RenderObjectIdEntry,
  ThreeEulerLike,
} from './types'

// eslint-disable-next-line @typescript-eslint/no-var-requires
const native = require('../native.js')

import { resolveSize, cameraViewProjection, cameraViewMatrix, cameraWorldPosition } from './camera'
import { colorLikeToArray, resolveBackground } from './color'
import { flattenScene, type ShadowMaterialMode } from './scene'
import { extractLights, extractAmbientLight, extractAmbientIntensity, extractLightProbe } from './lights'
import { extractBackgroundTexture, resolveEnvironmentMap } from './materials'
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
  RenderObjectIdEntry,
  PostProcessingOptions,
} from './types'

export class Renderer {
  private native: InstanceType<typeof native.NativeRenderer>

  constructor() {
    this.native = new native.NativeRenderer()
  }

  render(scene: ThreeSceneRootLike, camera: ThreeCameraLike, options: RenderOptions = {}): Buffer {
    const { buffer, nativeScene, nativeCamera, objectIdEntries } = this.renderNative(scene, camera, options)
    if (options.target) {
      const depthData = renderTargetDepthBuffer(
        options.target,
        nativeScene,
        nativeCamera,
        (targetScene, targetCamera) => this.native.render(targetScene, targetCamera),
      )
      writeRenderTarget(options.target, buffer, nativeScene.width!, nativeScene.height!, objectIdEntries, depthData)
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
}

export function render(scene: ThreeSceneRootLike, camera: ThreeCameraLike, options: RenderOptions = {}): Buffer {
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
  camera: ThreeCameraLike,
  target: RenderTargetLike = {},
  options: RenderOptions = {},
): RenderTargetLike {
  const targetOptions: RenderOptions = { ...options, target, format: options.format ?? 'rgba' }
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
  const environment = colorMode ? resolveEnvironmentMap(scene) : { envMap: null }
  const envMap = environment.envMap
  const environmentRotation = environment.rotation ?? scene.environmentRotation
  const environmentRotationLabel = environment.rotation ? 'material.envMapRotation' : 'scene.environmentRotation'
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
  const backgroundTextureRotation = colorMode
    ? backgroundRotationToNative(scene.backgroundRotation, backgroundTexture)
    : undefined
  const clippingPlanes = extractClippingPlanes(options.clippingPlanes)
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
  )
  const objectIdEntries = renderMode === 'object-id' ? objectIdEntriesForMeshes(flattenedMeshes) : undefined
  const meshes = applyRenderMode(flattenedMeshes, renderMode)
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
    backgroundTextureAnisotropy: backgroundTexture?.anisotropy,
    backgroundTextureTransform: backgroundTexture?.transform,
    backgroundTextureColorSpace: backgroundTexture?.colorSpace,
    backgroundTextureMapping: backgroundTexture?.mapping,
    backgroundTextureRotation,
    backgroundTextureBlurriness: colorMode ? finiteOrUndefined(options.backgroundBlurriness ?? scene.backgroundBlurriness) : undefined,
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
    near: finiteOrUndefined(camera.near),
    far: finiteOrUndefined(camera.far),
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
    customFragmentShader: DEPTH_READBACK_FRAGMENT,
    castShadow: false,
    receiveShadow: false,
  }
}

function meshWritesDepth(mesh: NativeSceneMesh): boolean {
  if (mesh.depthTest === false) return false
  if (typeof mesh.depthWrite === 'boolean') return mesh.depthWrite
  return !meshDefaultsTransparent(mesh)
}

function meshDefaultsTransparent(mesh: NativeSceneMesh): boolean {
  if (mesh.alphaHash === true) return false
  return mesh.transparent === true || materialAlpha(mesh) < 0.999 || finitePositive(mesh.transmission)
}

function finitePositive(value: unknown): boolean {
  return typeof value === 'number' && Number.isFinite(value) && value > 0.0001
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

type EulerOrder = 'XYZ' | 'YXZ' | 'ZXY' | 'ZYX' | 'YZX' | 'XZY'

function backgroundRotationToNative(
  rotation: ThreeSceneRootLike['backgroundRotation'],
  backgroundTexture: { mapping?: string } | null,
): number[] | undefined {
  if (!hasNonZeroRotation(rotation)) return undefined
  if (backgroundTexture?.mapping !== 'equirectangular') {
    throw new Error(
      'scene.backgroundRotation is only supported for equirectangular or cube texture backgrounds by @headless-three/renderer. Leave backgroundRotation at its default for color/2D backgrounds or pre-rotate the background texture before rendering.',
    )
  }
  const { x, y, z, order } = eulerComponents(rotation, 'scene.backgroundRotation')
  // Three.js negates background Euler angles before producing the rotation matrix
  // to account for the background shader's left-handed frame.
  return eulerRotationMatrix3Columns(-x, -y, -z, order)
}

function environmentRotationToNative(
  rotation: ThreeSceneRootLike['environmentRotation'],
  envMap: { data?: Buffer } | null,
  label = 'scene.environmentRotation',
): number[] | undefined {
  if (!hasNonZeroRotation(rotation) || !envMap) return undefined
  const { x, y, z, order } = eulerComponents(rotation, label)
  return eulerRotationMatrix3Columns(-x, -y, -z, order)
}

function eulerComponents(value: ThreeEulerLike | ArrayLike<number> | null | undefined, label: string): { x: number; y: number; z: number; order: EulerOrder } {
  const rotation = value as (ThreeEulerLike & { length?: number }) | null | undefined
  if (!rotation) return { x: 0, y: 0, z: 0, order: 'XYZ' }
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

  const texture = Array.isArray(target.texture)
    ? target.texture[0]
    : target.texture ?? target.textures?.[0]
  if (texture) {
    writeRenderTargetTexture(texture, data, width, height)
  }

  if (target.depthTexture != null && depthData) {
    writeRenderTargetTexture(target.depthTexture, depthData, width, height)
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

function writeRenderTargetTexture(
  texture: RenderTargetTextureLike,
  data: Buffer,
  width: number,
  height: number,
): void {
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
