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
import { RendererRenderListItem } from './index.part-006'
import { checkedRenderMode } from './index.part-011'
import { eulerComponents } from './index.part-015'
import { assertSupportedDepthTextureFormat, assertSupportedDepthTextureType, assertSupportedRenderTargetColorTexture, assertSupportedRenderTargetTextureClass, assertSupportedRenderTargetTextureDimensionality } from './index.part-018'
import { renderTargetColorTextures } from './index.part-019'
export function textureCopyInteger(value: unknown, label: string): number {
  if (!Number.isInteger(value)) {
    throw new TypeError(`${label} must be an integer.`)
  }
  return value as number
}

export function textureCopyFlooredInteger(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a finite number.`)
  }
  return Math.floor(value)
}

export function textureCopyPositiveInteger(value: unknown, label: string): number {
  const integer = textureCopyInteger(value, label)
  if (integer <= 0) {
    throw new RangeError(`${label} must be a positive integer.`)
  }
  return integer
}

export function textureCopyPositiveFlooredInteger(value: unknown, label: string): number {
  const integer = textureCopyFlooredInteger(value, label)
  if (integer <= 0) {
    throw new RangeError(`${label} must be a positive integer.`)
  }
  return integer
}

export function assertEulerOption(value: unknown, label: string): void {
  if (value == null) return
  if (typeof value !== 'object') {
    throw new TypeError(`${label} must be a rotation object or array.`)
  }
  eulerComponents(value as ThreeEulerLike | ArrayLike<number>, label)
}

export function validateSortControls(options: RenderOptions): void {
  if (options.sortObjects != null && typeof options.sortObjects !== 'boolean') {
    throw new TypeError(`options.sortObjects must be a boolean; received ${String(options.sortObjects)}.`)
  }
  if (options.opaque != null && typeof options.opaque !== 'boolean') {
    throw new TypeError(`options.opaque must be a boolean; received ${String(options.opaque)}.`)
  }
  if (options.transparent != null && typeof options.transparent !== 'boolean') {
    throw new TypeError(`options.transparent must be a boolean; received ${String(options.transparent)}.`)
  }
  if (options.opaqueSort != null && typeof options.opaqueSort !== 'function') {
    throw new TypeError('options.opaqueSort must be a function or null.')
  }
  if (options.transparentSort != null && typeof options.transparentSort !== 'function') {
    throw new TypeError('options.transparentSort must be a function or null.')
  }
}

export function assertSortFunctionOrNull(value: unknown, label: string): asserts value is RenderSortFunction | null {
  if (value != null && typeof value !== 'function') {
    throw new TypeError(`${label} expects a function or null.`)
  }
}

export function validateUnsupportedRenderTargetOptions(target: RenderTargetLike): void {
  if (target.scissorTest != null && typeof target.scissorTest !== 'boolean') {
    throw new TypeError('target.scissorTest must be a boolean.')
  }
  if (target.image != null) assertRenderTargetImageLike(target.image, 'target.image')
  assertRenderTargetTextureSlot(target.texture, 'target.texture')
  assertRenderTargetTexturesSlot(target.textures, 'target.textures')
  if (target.depthTexture != null) assertRenderTargetTextureLike(target.depthTexture, 'target.depthTexture')
  assertSupportedSampleCount(target.samples, 'target.samples')
  assertSupportedSampleCount(target.sampleCount, 'target.sampleCount')
  const colorTextures = renderTargetColorTextures(target)
  assertAuxiliaryRenderTargetAttachments(colorTextures)
  for (let i = 0; i < colorTextures.length; i += 1) {
    const colorTexture = colorTextures[i]
    const label = targetColorTextureLabel(i)
    assertSupportedRenderTargetTextureDimensionality(colorTexture, label)
    assertSupportedRenderTargetTextureClass(colorTexture, label)
    assertSupportedRenderTargetColorTexture(colorTexture, label)
  }
  assertSupportedRenderTargetTextureDimensionality(target.depthTexture, 'target.depthTexture')
  assertSupportedRenderTargetTextureClass(target.depthTexture, 'target.depthTexture')
  assertSupportedDepthTextureType(target.depthTexture)
  assertSupportedDepthTextureFormat(target.depthTexture)
}

export function assertAuxiliaryRenderTargetAttachments(colorTextures: RenderTargetTextureLike[]): void {
  if (colorTextures.length <= 1) {
    for (let i = 0; i < colorTextures.length; i += 1) {
      renderTargetTextureRenderMode(colorTextures[i], targetColorTextureLabel(i))
    }
    return
  }

  for (let i = 0; i < colorTextures.length; i += 1) {
    const mode = renderTargetTextureRenderMode(colorTextures[i], targetColorTextureLabel(i))
    if (i > 0 && mode == null) {
      throw new Error(
        `${targetColorTextureLabel(i)} is a secondary color attachment and must declare userData.headlessThreeRenderer.renderMode as "color", "mask", "object-id", "normal", or "depth". Arbitrary native MRT shader outputs are not supported yet.`,
      )
    }
  }
}

export function renderTargetTextureRenderMode(texture: RenderTargetTextureLike, label: string): RenderMode | undefined {
  const hints = renderTargetTextureRendererHints(texture, label)
  if (!hints || hints.value.renderMode == null) return undefined
  return checkedRenderMode(hints.value.renderMode, `${hints.label}.renderMode`)
}

export function renderTargetTextureRendererHints(
  texture: RenderTargetTextureLike,
  label: string,
): { value: Record<string, unknown>; label: string } | undefined {
  const userData = texture.userData
  if (userData == null) return undefined
  assertPlainObject(userData, `${label}.userData`)

  const modernHints = userData.headlessThreeRenderer
  if (modernHints != null) {
    assertPlainObject(modernHints, `${label}.userData.headlessThreeRenderer`)
    return { value: modernHints, label: `${label}.userData.headlessThreeRenderer` }
  }

  const legacyHints = userData.headlessRenderer
  if (legacyHints != null) {
    assertPlainObject(legacyHints, `${label}.userData.headlessRenderer`)
    return { value: legacyHints, label: `${label}.userData.headlessRenderer` }
  }

  return undefined
}

export function assertPlainObject(value: unknown, label: string): asserts value is Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be an object.`)
  }
}

export function targetColorTextureLabel(index: number): string {
  return index === 0 ? 'target color texture' : `target color texture[${index}]`
}

export function assertSupportedSampleCount(value: unknown, label: string): void {
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

export function assertSupportedOutputFormat(value: unknown, label: string): void {
  if (value == null) return
  if (value === 'png' || value === 'rgba') return
  throw new Error(
    `${label} ${String(value)} is not supported by @headless-three/renderer. Use "png" or "rgba".`,
  )
}

export function assertSupportedOutputColorSpace(value: unknown): void {
  if (value == null) return
  checkedOutputColorSpace(value, 'options.outputColorSpace')
}

export function checkedOutputColorSpace(value: unknown, label: string): RenderOutputColorSpace {
  if (
    value === 'srgb' ||
    value === 'srgb-linear' ||
    value === 'linear-srgb' ||
    value === 'linearsrgb' ||
    value === 'linear'
  ) return value
  throw new Error(
    `${label} ${String(value)} is not supported by @headless-three/renderer. Use THREE.SRGBColorSpace or THREE.LinearSRGBColorSpace.`,
  )
}

export function assertRenderTargetTextureSlot(value: unknown, label: string): void {
  if (value == null) return
  if (Array.isArray(value)) {
    if (value.length === 0) {
      throw new TypeError(`${label} must contain one texture-like object when provided as an array.`)
    }
    for (let i = 0; i < value.length; i += 1) {
      assertRenderTargetTextureLike(value[i], `${label}[${i}]`)
    }
    return
  }
  assertRenderTargetTextureLike(value, label)
}

export function assertRenderTargetTexturesSlot(value: unknown, label: string): void {
  if (value == null) return
  if (!Array.isArray(value)) {
    throw new TypeError(`${label} must be an array of texture-like objects.`)
  }
  if (value.length === 0) {
    throw new TypeError(`${label} must contain one texture-like object when provided.`)
  }
  for (let i = 0; i < value.length; i += 1) {
    assertRenderTargetTextureLike(value[i], `${label}[${i}]`)
  }
}

export function assertRenderTargetTextureLike(value: unknown, label: string): asserts value is RenderTargetTextureLike {
  if (value == null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be a texture-like object.`)
  }
  const texture = value as RenderTargetTextureLike
  assertRenderTargetImageSlot(texture.image, `${label}.image`)
  assertRenderTargetMipmaps(texture.mipmaps, `${label}.mipmaps`)
  assertRenderTargetSource(texture.source, `${label}.source`)
}

export function assertRenderTargetSource(value: unknown, label: string): void {
  if (value == null) return
  if (typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be a source-like object.`)
  }
  assertRenderTargetImageSlot((value as { data?: unknown }).data, `${label}.data`)
}

export function assertRenderTargetMipmaps(value: unknown, label: string): void {
  if (value == null) return
  if (!Array.isArray(value)) {
    throw new TypeError(`${label} must be an array of image-like objects.`)
  }
  for (let index = 0; index < value.length; index += 1) {
    assertRenderTargetImageLike(value[index], `${label}[${index}]`)
  }
}

export function assertRenderTargetImageSlot(value: unknown, label: string): void {
  if (value == null) return
  if (Array.isArray(value)) {
    value.forEach((image, index) => {
      assertRenderTargetImageLike(image, `${label}[${index}]`)
    })
    return
  }
  assertRenderTargetImageLike(value, label)
}

export function assertRenderTargetImageLike(value: unknown, label: string): asserts value is RenderTargetImageLike {
  if (value == null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be an image-like object.`)
  }
}

export function assertWebGlExtensionName(value: unknown, label: string): asserts value is string {
  if (typeof value !== 'string' || value.length === 0) {
    throw new TypeError(`${label} must be a non-empty string.`)
  }
}

export function assertRendererProbeName(value: unknown, label: string): asserts value is string {
  if (typeof value !== 'string' || value.length === 0) {
    throw new TypeError(`${label} must be a non-empty string.`)
  }
}

export function assertEventListener(type: unknown, listener: unknown, label: string): void {
  if (typeof type !== 'string' || type.length === 0) {
    throw new TypeError(`${label} type must be a non-empty string.`)
  }
  if (typeof listener !== 'function') {
    throw new TypeError(`${label} listener must be a function.`)
  }
}

export function assertDomElementAttributeName(value: unknown, label: string): asserts value is string {
  if (typeof value !== 'string' || value.length === 0) {
    throw new TypeError(`${label} must be a non-empty string.`)
  }
}

export function domElementStylePropertyKey(value: unknown, label: string): string {
  if (typeof value !== 'string' || value.length === 0) {
    throw new TypeError(`${label} must be a non-empty string.`)
  }
  if (value.startsWith('--')) return value
  return value.replace(/-([a-z])/g, (_match, letter: string) => letter.toUpperCase())
}

export function domElementStyleWritablePropertyKey(value: unknown, label: string): string {
  const key = domElementStylePropertyKey(value, label)
  if (key === 'setProperty' || key === 'getPropertyValue' || key === 'removeProperty') {
    throw new TypeError(`${label} must not name a reserved style method.`)
  }
  return key
}

export function assertXrInputIndex(value: unknown, label: string): asserts value is number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
    throw new TypeError(`${label} must be a non-negative integer.`)
  }
}

export function assertWeakMapKey(value: unknown, label: string): asserts value is object {
  if (value == null || (typeof value !== 'object' && typeof value !== 'function')) {
    throw new TypeError(`${label} must be an object.`)
  }
}

export function assertWeakMapKeyArray(value: unknown, label: string): object[] {
  if (!Array.isArray(value) || value.length === 0) {
    throw new TypeError(`${label} must be a non-empty array of objects.`)
  }
  for (let i = 0; i < value.length; i += 1) {
    assertWeakMapKey(value[i], `${label}[${i}]`)
  }
  return value as object[]
}

export function assertPropertyKey(value: unknown, label: string): asserts value is string {
  if (typeof value !== 'string' || value.length === 0) {
    throw new TypeError(`${label} must be a non-empty string.`)
  }
}

export function assertNonEmptyString(value: unknown, label: string): asserts value is string {
  if (typeof value !== 'string' || value.length === 0) {
    throw new TypeError(`${label} must be a non-empty string.`)
  }
}

export function assertFunction(value: unknown, label: string): asserts value is (...args: unknown[]) => unknown {
  if (typeof value !== 'function') {
    throw new TypeError(`${label} must be a function.`)
  }
}

export function assertConstructorFunction(value: unknown, label: string): asserts value is new (...args: unknown[]) => Record<string, unknown> {
  if (typeof value !== 'function') {
    throw new TypeError(`${label} must be a constructor function.`)
  }
}

export function isNodeLike(value: unknown): boolean {
  return value != null && typeof value === 'object' && !Array.isArray(value) && (value as { isNode?: unknown }).isNode === true
}

export function rendererNodeFrame(): Record<string, unknown> {
  return {
    frameId: 0,
    renderId: 0,
    renderer: null,
    scene: null,
    object: null,
    camera: null,
    material: null,
    updateNode() {},
    updateBeforeNode() {},
    updateAfterNode() {},
  }
}

export function rendererSimpleHash(value: string): number {
  let hash = 0
  for (let i = 0; i < value.length; i += 1) {
    hash = ((hash << 5) - hash + value.charCodeAt(i)) | 0
  }
  return hash
}

export function assertFiniteInteger(value: unknown, label: string): asserts value is number {
  if (typeof value !== 'number' || !Number.isFinite(value) || Math.floor(value) !== value) {
    throw new TypeError(`${label} must be an integer.`)
  }
}

export function rendererRenderListId(object: unknown): unknown {
  return object && typeof object === 'object'
    ? (object as Record<string, unknown>).id
    : undefined
}

export function rendererRenderListRenderOrder(object: unknown): unknown {
  return object && typeof object === 'object'
    ? (object as Record<string, unknown>).renderOrder
    : undefined
}

export function rendererRenderListMaterialVariant(object: unknown): number {
  if (!object || typeof object !== 'object') return 0
  const record = object as Record<string, unknown>
  return (record.isInstancedMesh === true ? 2 : 0) + (record.isSkinnedMesh === true ? 1 : 0)
}

export function rendererRenderListOpaqueSort(a: RendererRenderListItem, b: RendererRenderListItem): number {
  const aMaterialId = rendererRenderListMaterialId(a.material)
  const bMaterialId = rendererRenderListMaterialId(b.material)
  return rendererRenderListSortNumber(a.groupOrder) - rendererRenderListSortNumber(b.groupOrder)
    || rendererRenderListSortNumber(a.renderOrder) - rendererRenderListSortNumber(b.renderOrder)
    || aMaterialId - bMaterialId
    || rendererRenderListSortNumber(a.z) - rendererRenderListSortNumber(b.z)
    || rendererRenderListSortNumber(a.id) - rendererRenderListSortNumber(b.id)
}

export function rendererRenderListTransparentSort(a: RendererRenderListItem, b: RendererRenderListItem): number {
  return rendererRenderListSortNumber(a.groupOrder) - rendererRenderListSortNumber(b.groupOrder)
    || rendererRenderListSortNumber(a.renderOrder) - rendererRenderListSortNumber(b.renderOrder)
    || rendererRenderListSortNumber(b.z) - rendererRenderListSortNumber(a.z)
    || rendererRenderListSortNumber(a.id) - rendererRenderListSortNumber(b.id)
}

export function rendererRenderListMaterialId(material: unknown): number {
  return material && typeof material === 'object'
    && typeof (material as { id?: unknown }).id === 'number'
    && Number.isFinite((material as { id: number }).id)
    ? (material as { id: number }).id
    : 0
}

export function rendererRenderListSortNumber(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

export function rendererRenderListNeedsDoublePass(material: Record<string, unknown> | undefined): boolean {
  const transmission = material?.transmission
  const hasTransmission = typeof transmission === 'number' && transmission > 0
  return hasTransmission && material?.side === 2 && material.forceSinglePass !== true
}
