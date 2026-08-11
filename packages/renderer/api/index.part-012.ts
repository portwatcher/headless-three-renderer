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
import { WEBGL_COORDINATE_SYSTEM, native } from './index.part-001'
import { normalizedRenderMode, toNativeInput } from './index.part-011'
import { cubeMipmapScissor, cubeMipmapScissorLabel, cubeMipmapSize, cubeMipmapViewport, cubeMipmapViewportLabel, cubeTargetTexture, renderArrayCamera, resolveActiveMipmapLevel, writeCubeRenderTarget, writeCubeTextureFaces } from './index.part-013'
import { depthReadbackScene, formatWgslFloat } from './index.part-014'
import { assertRenderTargetLike } from './index.part-016'
import { assertSupportedOutputFormat, renderTargetTextureRenderMode, targetColorTextureLabel, validateUnsupportedRenderTargetOptions } from './index.part-017'
import { renderTargetColorTextures } from './index.part-019'
import { colorTextureData } from './index.part-020'
import { cloneTargetData, depthTextureData, isCubeCamera, validateThreeCamera, validateThreeSceneRoot } from './index.part-021'
export function objectSortId(mesh: NativeSceneMesh, index: number): number {
  return typeof mesh.sortIndex === 'number' && Number.isSafeInteger(mesh.sortIndex) && mesh.sortIndex >= 0
    ? mesh.sortIndex
    : index
}

export function renderModeFragment(mode: Exclude<RenderMode, 'color'>, color: Color4): string {
  if (mode === 'normal') {
    return [
      'let view_normal = normalize((uniforms.view * vec4<f32>(normal, 0.0)).xyz);',
      'return vec4<f32>(view_normal * 0.5 + vec3<f32>(0.5), 1.0);',
    ].join('\n')
  }
  if (mode === 'depth') return DEPTH_READBACK_FRAGMENT
  return `return vec4<f32>(${formatWgslFloat(color[0])}, ${formatWgslFloat(color[1])}, ${formatWgslFloat(color[2])}, 1.0);`
}

export const DEPTH_READBACK_FRAGMENT = [
  'let frag_depth = clamp(input.position.z, 0.0, 1.0);',
  'let depth = 1.0 - frag_depth;',
  'return vec4<f32>(depth, depth, depth, 1.0);',
].join('\n')

export function renderTargetDepthBuffer(
  target: RenderTargetLike | undefined,
  nativeScene: NativeRenderScene,
  nativeCamera: NativeCamera,
  renderNativeScene: (scene: NativeRenderScene, camera: NativeCamera) => Buffer,
): Buffer | undefined {
  if (target?.depthTexture == null) return undefined
  return renderNativeScene(depthReadbackScene(nativeScene), nativeCamera)
}

export function renderTargetHasExplicitSize(target: RenderTargetLike | null | undefined): boolean {
  if (!target) return false
  if (target.width != null || target.height != null) return true
  const texture = cubeTargetTexture(target)
  const firstImage = Array.isArray(texture?.image) ? texture.image[0] : undefined
  return firstImage?.width != null || firstImage?.height != null
}

export function renderAuxiliaryTargetAttachments(
  target: RenderTargetLike | undefined,
  options: RenderOptions,
  primaryData: Buffer,
  primaryObjectIdEntries: RenderObjectIdEntry[] | undefined,
  renderAttachment: (mode: RenderMode) => { data: Buffer; objectIdEntries?: RenderObjectIdEntry[] },
): { attachments?: RenderTargetAttachmentData[]; objectIdEntries?: RenderObjectIdEntry[] } {
  if (!target) return { objectIdEntries: primaryObjectIdEntries }
  const colorTextures = renderTargetColorTextures(target)
  if (colorTextures.length <= 1) return { objectIdEntries: primaryObjectIdEntries }

  const primaryMode = normalizedRenderMode(options.renderMode)
  let targetObjectIdEntries = primaryObjectIdEntries
  const attachments: RenderTargetAttachmentData[] = []

  for (let i = 1; i < colorTextures.length; i += 1) {
    const texture = colorTextures[i]
    const mode = renderTargetTextureRenderMode(texture, targetColorTextureLabel(i))!
    if (mode === primaryMode) {
      attachments.push({ texture, data: primaryData })
      if (mode === 'object-id') targetObjectIdEntries = primaryObjectIdEntries
      continue
    }

    const rendered = renderAttachment(mode)
    attachments.push({ texture, data: rendered.data })
    if (mode === 'object-id') targetObjectIdEntries = rendered.objectIdEntries
  }

  return { attachments, objectIdEntries: targetObjectIdEntries }
}

export function renderRegularCameraAuxiliaryTargetAttachments(
  scene: ThreeSceneRootLike,
  camera: ThreeCameraLike,
  target: RenderTargetLike | undefined,
  options: RenderOptions,
  primaryData: Buffer,
  primaryObjectIdEntries: RenderObjectIdEntry[] | undefined,
  renderNativeScene: (scene: NativeRenderScene, camera: NativeCamera) => Buffer,
): { attachments?: RenderTargetAttachmentData[]; objectIdEntries?: RenderObjectIdEntry[] } {
  return renderAuxiliaryTargetAttachments(
    target,
    options,
    primaryData,
    primaryObjectIdEntries,
    (mode) => {
      const { nativeScene, nativeCamera, objectIdEntries } = toNativeInput(scene, camera, {
        ...options,
        renderMode: mode,
        format: 'rgba',
      })
      return {
        data: renderNativeScene(nativeScene, nativeCamera),
        objectIdEntries,
      }
    },
  )
}

export function renderArrayCameraAuxiliaryTargetAttachments(
  scene: ThreeSceneRootLike,
  camera: ThreeCameraLike,
  target: RenderTargetLike | undefined,
  options: RenderOptions,
  primaryData: Buffer,
  primaryObjectIdEntries: RenderObjectIdEntry[] | undefined,
  renderNativeScene: RenderNativeScene,
): { attachments?: RenderTargetAttachmentData[]; objectIdEntries?: RenderObjectIdEntry[] } {
  return renderAuxiliaryTargetAttachments(
    target,
    options,
    primaryData,
    primaryObjectIdEntries,
    (mode) => {
      const rendered = renderArrayCamera(scene, camera, {
        ...options,
        renderMode: mode,
        format: 'rgba',
      }, renderNativeScene)
      return {
        data: rendered.buffer,
        objectIdEntries: rendered.objectIdEntries,
      }
    },
  )
}

export function renderCubeCameraAuxiliaryTargetAttachments(
  target: RenderTargetLike,
  options: RenderOptions,
  primaryFaces: Buffer[],
  primaryObjectIdEntries: RenderObjectIdEntry[] | undefined,
  renderAttachment: (mode: RenderMode) => { faces: Buffer[]; objectIdEntries?: RenderObjectIdEntry[] },
): { attachments?: RenderCubeTargetAttachmentData[]; objectIdEntries?: RenderObjectIdEntry[] } {
  const colorTextures = renderTargetColorTextures(target)
  if (colorTextures.length <= 1) return { objectIdEntries: primaryObjectIdEntries }

  const primaryMode = normalizedRenderMode(options.renderMode)
  let targetObjectIdEntries = primaryObjectIdEntries
  const attachments: RenderCubeTargetAttachmentData[] = []

  for (let i = 1; i < colorTextures.length; i += 1) {
    const texture = colorTextures[i]
    const mode = renderTargetTextureRenderMode(texture, targetColorTextureLabel(i))!
    if (mode === primaryMode) {
      attachments.push({ texture, faces: primaryFaces })
      if (mode === 'object-id') targetObjectIdEntries = primaryObjectIdEntries
      continue
    }

    const rendered = renderAttachment(mode)
    attachments.push({ texture, faces: rendered.faces })
    if (mode === 'object-id') targetObjectIdEntries = rendered.objectIdEntries
  }

  return { attachments, objectIdEntries: targetObjectIdEntries }
}

export function sortedObjectIdEntries(objectIdEntryMap: Map<number, RenderObjectIdEntry>): RenderObjectIdEntry[] | undefined {
  return objectIdEntryMap.size > 0
    ? [...objectIdEntryMap.values()].sort((a, b) => a.encodedId - b.encodedId)
    : undefined
}

export function renderCubeCameraFaces(
  scene: ThreeSceneRootLike,
  subCameras: ThreeCameraLike[],
  target: RenderTargetLike,
  faceOptions: InternalRenderOptions,
  renderNativeScene: RenderNativeScene,
  includeDepth: boolean,
): {
  faces: Buffer[]
  depthFaces?: NonNullable<RenderTargetImageLike['data']>[]
  objectIdEntries?: RenderObjectIdEntry[]
} {
  const objectIdEntryMap = new Map<number, RenderObjectIdEntry>()
  const faces: Buffer[] = []
  const depthFaces: NonNullable<RenderTargetImageLike['data']>[] = []

  for (const subCamera of subCameras) {
    const { nativeScene, nativeCamera, objectIdEntries } = toNativeInput(scene, subCamera, faceOptions)
    faces.push(Buffer.from(renderNativeScene(nativeScene, nativeCamera)))
    if (objectIdEntries) {
      for (const entry of objectIdEntries) {
        objectIdEntryMap.set(entry.encodedId, entry)
      }
    }
    if (includeDepth) {
      const depthFace = renderTargetDepthBuffer(target, nativeScene, nativeCamera, renderNativeScene)
      if (depthFace) {
        depthFaces.push(cloneTargetData(depthTextureData(target.depthTexture!, depthFace)))
      }
    }
  }

  return {
    faces,
    depthFaces: depthFaces.length > 0 ? depthFaces : undefined,
    objectIdEntries: sortedObjectIdEntries(objectIdEntryMap),
  }
}

export function assertCubeFaceCount(faces: unknown[], label: string): void {
  if (faces.length !== CUBE_FACE_COUNT) {
    throw new Error(
      `THREE.CubeCamera expected ${CUBE_FACE_COUNT} rendered ${label} faces, received ${faces.length}.`,
    )
  }
}

export function ensureCubeTargetAttachmentTexture(texture: RenderTargetTextureLike): void {
  texture.isCubeTexture = true
  texture.needsPMREMUpdate = true
  texture.pmremVersion = (texture.pmremVersion ?? 0) + 1
}

export function writeCubeTargetAttachmentFaces(
  attachment: RenderCubeTargetAttachmentData,
  faceWidth: number,
  faceHeight: number,
  activeMipmapLevel: number,
  label: string,
): void {
  assertCubeFaceCount(attachment.faces, label)
  ensureCubeTargetAttachmentTexture(attachment.texture)
  writeCubeTextureFaces(
    attachment.texture,
    attachment.faces.map((face) => colorTextureData(attachment.texture, face)),
    faceWidth,
    faceHeight,
    activeMipmapLevel,
    label,
  )
}

export type RenderNativeScene = (scene: NativeRenderScene, camera: NativeCamera) => Buffer

export type PixelRect = {
  x: number
  y: number
  width: number
  height: number
}
export type PixelSize = {
  width: number
  height: number
}
export type RenderTargetAttachmentData = {
  texture: RenderTargetTextureLike
  data: Buffer
}
export type RenderCubeTargetAttachmentData = {
  texture: RenderTargetTextureLike
  faces: Buffer[]
}
export type InternalRenderOptions = RenderOptions & {
  __headlessThreeViewportLabel?: string
  __headlessThreeScissorLabel?: string
  __headlessThreeRendererClearColor?: Color4
  __headlessThreeRendererViewport?: PixelRect | null
  __headlessThreeRendererScissor?: PixelRect | null
  __headlessThreeRendererScissorTest?: boolean
  __headlessThreeRendererShadowMapEnabled?: boolean
  __headlessThreeRendererShadowMapType?: number
  __headlessThreeClippingPlanesLabel?: string
  __headlessThreeRendererToneMapping?: number
  __headlessThreeRendererToneMappingExposure?: number
  __headlessThreeRendererTransmissionResolutionScale?: number
  __headlessThreeRendererOpaque?: boolean
  __headlessThreeRendererTransparent?: boolean
  __headlessThreeRenderer?: unknown
}

export const CUBE_FACE_COUNT = 6
export const UnsignedByteType = 1009
export const ByteType = 1010
export const ShortType = 1011
export const UnsignedShortType = 1012
export const IntType = 1013
export const UnsignedIntType = 1014
export const FloatType = 1015
export const HalfFloatType = 1016
export const UnsignedShort4444Type = 1017
export const UnsignedShort5551Type = 1018
export const UnsignedInt248Type = 1020
export const AlphaFormat = 1021
export const RGBFormat = 1022
export const RGBAFormat = 1023
export const LuminanceFormat = 1024
export const LuminanceAlphaFormat = 1025
export const DepthFormat = 1026
export const DepthStencilFormat = 1027
export const RedFormat = 1028
export const RedIntegerFormat = 1029
export const RGFormat = 1030
export const RGIntegerFormat = 1031
export const RGBIntegerFormat = 1032
export const RGBAIntegerFormat = 1033
export const UnsignedInt101111Type = 35899
export const UnsignedInt5999Type = 35902

export function renderCubeCamera(
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
  assertRenderTargetLike(target, options.target !== undefined ? 'options.target' : 'THREE.CubeCamera renderTarget')
  validateUnsupportedRenderTargetOptions(target)

  const { width: targetWidth, height: targetHeight } = resolveCubeTargetSize(target, options)
  const activeMipmapLevel = resolveCubeMipmapLevel(camera, targetWidth)
  const { width, height } = cubeMipmapSize(targetWidth, targetHeight, activeMipmapLevel)
  const outputFormat = options.format ?? (options.target ? 'rgba' : 'png')
  const subCameras = cubeSubCameras(camera)
  const faceOptions: InternalRenderOptions = {
    ...options,
    target,
    width,
    height,
    format: 'rgba',
    viewport: cubeMipmapViewport(options, target, activeMipmapLevel),
    scissor: cubeMipmapScissor(options, target, activeMipmapLevel),
    __headlessThreeViewportLabel: cubeMipmapViewportLabel(options),
    __headlessThreeScissorLabel: cubeMipmapScissorLabel(options, target),
  }
  const primary = renderCubeCameraFaces(scene, subCameras, target, faceOptions, renderNativeScene, true)
  const auxiliary = renderCubeCameraAuxiliaryTargetAttachments(
    target,
    options,
    primary.faces,
    primary.objectIdEntries,
    (mode) => renderCubeCameraFaces(
      scene,
      subCameras,
      target,
      {
        ...faceOptions,
        renderMode: mode,
        format: 'rgba',
      },
      renderNativeScene,
      false,
    ),
  )

  writeCubeRenderTarget(
    target,
    primary.faces,
    targetWidth,
    targetHeight,
    width,
    height,
    activeMipmapLevel,
    primary.depthFaces,
    auxiliary.objectIdEntries,
    auxiliary.attachments,
  )

  const buffer = outputFormat === 'png' ? native.encodePng(primary.faces[0], width, height) : primary.faces[0]
  return { buffer, target, width, height, faces: primary.faces }
}

export function validateCubeCamera(camera: ThreeCubeCameraLike, options: RenderOptions): void {
  if (!isCubeCamera(camera)) {
    throw new TypeError('render(scene, camera) expected a THREE.CubeCamera-compatible object.')
  }
  assertSupportedOutputFormat(options.format, 'options.format')
}

export function cubeSubCameras(camera: ThreeCubeCameraLike): ThreeCameraLike[] {
  const children = camera.children
  if (!Array.isArray(children)) {
    throw new TypeError('THREE.CubeCamera.children must be an array of internal perspective cameras.')
  }
  if (children.length < CUBE_FACE_COUNT) {
    throw new Error('THREE.CubeCamera requires six internal perspective cameras.')
  }
  const subCameras = children.slice(0, CUBE_FACE_COUNT)
  for (let index = 0; index < subCameras.length; index += 1) {
    validateThreeCamera(subCameras[index], `THREE.CubeCamera.children[${index}]`)
  }

  if (typeof camera.updateCoordinateSystem === 'function' && camera.coordinateSystem !== WEBGL_COORDINATE_SYSTEM) {
    camera.coordinateSystem = WEBGL_COORDINATE_SYSTEM
    camera.updateCoordinateSystem()
  }
  if (typeof camera.updateMatrixWorld === 'function') {
    camera.updateMatrixWorld(true)
  }
  for (const subCamera of subCameras) {
    if (typeof subCamera.updateMatrixWorld === 'function') {
      subCamera.updateMatrixWorld(true)
    }
  }
  return subCameras
}

export function resolveCubeTargetSize(target: RenderTargetLike, options: RenderOptions): { width: number; height: number } {
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

export function resolveCubeMipmapLevel(camera: ThreeCubeCameraLike, targetSize: number): number {
  return resolveActiveMipmapLevel(camera.activeMipmapLevel ?? 0, targetSize, 'THREE.CubeCamera activeMipmapLevel')
}
