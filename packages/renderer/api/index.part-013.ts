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
import { native } from './index.part-001'
import { toNativeInput } from './index.part-011'
import { CUBE_FACE_COUNT, PixelRect, RenderCubeTargetAttachmentData, RenderNativeScene, RenderTargetAttachmentData, assertCubeFaceCount, ensureCubeTargetAttachmentTexture, renderTargetDepthBuffer, writeCubeTargetAttachmentFaces } from './index.part-012'
import { pixelRectComponents } from './index.part-014'
import { validateUnsupportedRenderOptions } from './index.part-015'
import { assertSupportedOutputFormat, targetColorTextureLabel } from './index.part-017'
import { writeObjectIdMetadata } from './index.part-019'
import { colorTextureData } from './index.part-020'
import { validateThreeCamera, validateThreeSceneRoot } from './index.part-021'
export function resolveActiveMipmapLevel(level: number, targetSize: number, label: string): number {
  if (!Number.isInteger(level) || level < 0) {
    throw new TypeError(`${label} must be a non-negative integer; received ${String(level)}.`)
  }
  const maxLevel = Math.floor(Math.log2(targetSize))
  if (level > maxLevel) {
    throw new Error(
      `${label} ${level} exceeds the maximum mip level ${maxLevel} for a ${targetSize}x${targetSize} cube target.`,
    )
  }
  return level
}

export function assertActiveCubeFace(value: number, label: string): void {
  if (!Number.isInteger(value) || value < 0 || value >= CUBE_FACE_COUNT) {
    throw new TypeError(`${label} must be an integer from 0 to ${CUBE_FACE_COUNT - 1}; received ${String(value)}.`)
  }
}

export function assertActiveMipmapLevel(value: number, label: string): void {
  if (!Number.isInteger(value) || value < 0) {
    throw new TypeError(`${label} must be a non-negative integer; received ${String(value)}.`)
  }
}

export function cubeMipmapSize(width: number, height: number, activeMipmapLevel: number): { width: number; height: number } {
  if (activeMipmapLevel === 0) return { width, height }
  return {
    width: Math.max(1, width >> activeMipmapLevel),
    height: Math.max(1, height >> activeMipmapLevel),
  }
}

export function cubeMipmapViewport(
  options: RenderOptions,
  target: RenderTargetLike,
  activeMipmapLevel: number,
): RenderPixelRectLike | null | undefined {
  if (options.viewport !== undefined) return options.viewport
  return cubeMipmapRect(target.viewport, activeMipmapLevel)
}

export function cubeMipmapViewportLabel(options: RenderOptions): string | undefined {
  if (options.viewport !== undefined) return 'options.viewport'
  return 'target.viewport'
}

export function cubeMipmapScissor(
  options: RenderOptions,
  target: RenderTargetLike,
  activeMipmapLevel: number,
): RenderPixelRectLike | null | undefined {
  if (options.scissor !== undefined) return options.scissor
  return target.scissorTest === true ? cubeMipmapRect(target.scissor, activeMipmapLevel) : undefined
}

export function cubeMipmapScissorLabel(options: RenderOptions, target: RenderTargetLike): string | undefined {
  if (options.scissor !== undefined) return 'options.scissor'
  return target.scissorTest === true ? 'target.scissor' : undefined
}

export function cubeMipmapRect(rect: RenderPixelRectLike | null | undefined, activeMipmapLevel: number): RenderPixelRectLike | null | undefined {
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

export function writeCubeRenderTarget(
  target: RenderTargetLike,
  faces: Buffer[],
  targetWidth: number,
  targetHeight: number,
  faceWidth: number,
  faceHeight: number,
  activeMipmapLevel: number,
  depthFaces?: NonNullable<RenderTargetImageLike['data']>[],
  objectIdEntries?: RenderObjectIdEntry[],
  colorAttachments?: RenderCubeTargetAttachmentData[],
): RenderTargetLike {
  assertCubeFaceCount(faces, 'color')
  target.width = targetWidth
  target.height = targetHeight
  target.data = faces[0]

  const texture = ensureCubeTargetTexture(target)
  ensureCubeTargetAttachmentTexture(texture)
  writeCubeTextureFaces(texture, faces.map((face) => colorTextureData(texture, face)), faceWidth, faceHeight, activeMipmapLevel, 'target.texture')
  if (target.depthTexture && depthFaces) {
    assertCubeFaceCount(depthFaces, 'depth')
    writeCubeTextureFaces(target.depthTexture, depthFaces, faceWidth, faceHeight, activeMipmapLevel, 'target.depthTexture')
  }
  const attachments = colorAttachments ?? []
  for (let i = 0; i < attachments.length; i += 1) {
    writeCubeTargetAttachmentFaces(
      attachments[i],
      faceWidth,
      faceHeight,
      activeMipmapLevel,
      targetColorTextureLabel(i + 1),
    )
  }
  writeObjectIdMetadata(target, objectIdEntries)
  return target
}

export function writeCubeRenderTargetFace(
  target: RenderTargetLike,
  face: Buffer,
  targetWidth: number,
  targetHeight: number,
  faceWidth: number,
  faceHeight: number,
  activeCubeFace: number,
  activeMipmapLevel: number,
  depthFace?: NonNullable<RenderTargetImageLike['data']>,
  objectIdEntries?: RenderObjectIdEntry[],
  colorAttachments?: RenderTargetAttachmentData[],
): RenderTargetLike {
  assertActiveCubeFace(activeCubeFace, 'Renderer activeCubeFace')
  target.width = targetWidth
  target.height = targetHeight
  target.data = face

  const texture = ensureCubeTargetTexture(target)
  ensureCubeTargetAttachmentTexture(texture)
  writeCubeTextureFace(
    texture,
    colorTextureData(texture, face),
    faceWidth,
    faceHeight,
    activeCubeFace,
    activeMipmapLevel,
    'target.texture',
  )
  if (target.depthTexture && depthFace) {
    writeCubeTextureFace(target.depthTexture, depthFace, faceWidth, faceHeight, activeCubeFace, activeMipmapLevel, 'target.depthTexture')
  }
  const attachments = colorAttachments ?? []
  for (let i = 0; i < attachments.length; i += 1) {
    const attachment = attachments[i]
    ensureCubeTargetAttachmentTexture(attachment.texture)
    writeCubeTextureFace(
      attachment.texture,
      colorTextureData(attachment.texture, attachment.data),
      faceWidth,
      faceHeight,
      activeCubeFace,
      activeMipmapLevel,
      targetColorTextureLabel(i + 1),
    )
  }
  writeObjectIdMetadata(target, objectIdEntries)
  return target
}

export function writeCubeTextureFaces(
  texture: RenderTargetTextureLike,
  faces: NonNullable<RenderTargetImageLike['data']>[],
  width: number,
  height: number,
  activeMipmapLevel: number,
  label: string,
): void {
  const images = faces.map((data) => ({ data, width, height, depth: 1 }))
  if (activeMipmapLevel === 0) {
    texture.image = images
    texture.source ??= {}
    texture.source.data = images
  } else {
    if (texture.mipmaps != null && !Array.isArray(texture.mipmaps)) {
      throw new TypeError(`${label}.mipmaps must be an array of image-like objects.`)
    }
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

export function writeCubeTextureFace(
  texture: RenderTargetTextureLike,
  data: NonNullable<RenderTargetImageLike['data']>,
  width: number,
  height: number,
  activeCubeFace: number,
  activeMipmapLevel: number,
  label: string,
): void {
  const image = { data, width, height, depth: 1 }
  if (activeMipmapLevel === 0) {
    const images = cubeTextureImages(texture.image)
    images[activeCubeFace] = image
    texture.image = images
    texture.source ??= {}
    texture.source.data = images
  } else {
    if (texture.mipmaps != null && !Array.isArray(texture.mipmaps)) {
      throw new TypeError(`${label}.mipmaps must be an array of image-like objects.`)
    }
    const mipmaps = texture.mipmaps ?? (texture.mipmaps = [])
    for (let level = 0; level <= activeMipmapLevel; level += 1) {
      mipmaps[level] ??= {}
    }
    const mipmap = mipmaps[activeMipmapLevel]
    const images = cubeTextureImages(mipmap.image)
    images[activeCubeFace] = image
    mipmap.image = images
    mipmap.width = width
    mipmap.height = height
    mipmap.depth = 1
  }
  texture.needsUpdate = true
}

export function cubeTextureImages(value: RenderTargetImageLike | RenderTargetImageLike[] | undefined): RenderTargetImageLike[] {
  const images = Array.isArray(value) ? value.slice() : Array.from({ length: CUBE_FACE_COUNT }, () => ({}))
  while (images.length < CUBE_FACE_COUNT) {
    images.push({})
  }
  return images
}

export function cubeTargetTexture(target: RenderTargetLike): RenderTargetTextureLike | undefined {
  return Array.isArray(target.texture)
    ? target.texture[0]
    : target.textures?.[0] ?? target.texture
}

export function ensureCubeTargetTexture(target: RenderTargetLike): RenderTargetTextureLike {
  const texture = cubeTargetTexture(target)
  if (texture) return texture
  const images = Array.from({ length: CUBE_FACE_COUNT }, () => ({}))
  const created: RenderTargetTextureLike = { image: images, source: { data: images }, isCubeTexture: true }
  target.texture = created
  return created
}

export function isCubeRenderTarget(target: RenderTargetLike): boolean {
  return target.isWebGLCubeRenderTarget === true ||
    cubeTargetTexture(target)?.isCubeTexture === true ||
    target.depthTexture?.isCubeTexture === true
}

export function renderArrayCamera(
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

export function validateArrayCameraOutput(camera: ThreeCameraLike, options: RenderOptions): void {
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

export function arraySubCameras(camera: ThreeCameraLike): ThreeCameraLike[] {
  const cameras = (camera as any).cameras
  if (!Array.isArray(cameras)) {
    throw new TypeError('THREE.ArrayCamera.cameras must be an array.')
  }
  if (cameras.length === 0) {
    throw new Error('THREE.ArrayCamera requires at least one sub-camera in camera.cameras.')
  }
  for (let index = 0; index < cameras.length; index += 1) {
    validateThreeCamera(cameras[index], `THREE.ArrayCamera.cameras[${index}]`)
  }
  return cameras
}

export function resolveSubCameraViewport(
  camera: ThreeCameraLike,
  fallback: RenderPixelRectLike | null | undefined,
  width: number,
  height: number,
): PixelRect | undefined {
  const viewport = cameraViewport(camera) ?? fallback
  return viewport ? normalizePixelRect(viewport, width, height, 'THREE.ArrayCamera sub-camera viewport') : undefined
}

export function cameraViewport(camera: ThreeCameraLike): RenderPixelRectLike | undefined {
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

export function normalizePixelRect(rect: RenderPixelRectLike, targetWidth: number, targetHeight: number, label: string): PixelRect {
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

export function normalizeOptionalPixelRect(
  rect: RenderPixelRectLike | null | undefined,
  targetWidth: number,
  targetHeight: number,
  label: string,
): PixelRect | undefined {
  if (rect == null) return undefined
  return normalizePixelRect(rect, targetWidth, targetHeight, label)
}

export function copyPixelRect(source: Buffer, destination: Buffer, imageWidth: number, rect: PixelRect): void {
  const rowBytes = rect.width * 4
  for (let row = 0; row < rect.height; row += 1) {
    const offset = ((rect.y + row) * imageWidth + rect.x) * 4
    source.copy(destination, offset, offset, offset + rowBytes)
  }
}
