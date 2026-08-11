import type { Color4, ThreeMaterialLike, PbrProperties, TextureInfo, ThreeTextureLike, ThreeSceneRootLike, ThreeObject3DLike } from './types'
import { clamp01 } from './math'
import { strictColorLikeToArray, validatedColorLikeToArray } from './color'
import { objectChildren } from './objects'
import { CubeReflectionMapping, CubeRefractionMapping, CubeUVReflectionMapping, DefaultOnBeforeCompileSource, EquirectangularReflectionMapping, EquirectangularRefractionMapping, MaterialExtractionContext, MaterialNodeHookProperties, TextureImageInput } from './materials.part-001'
import { sameUnknownArray, supportsNativeMaterialEnvironmentMap } from './materials.part-002'
import { optionalBoolean, optionalPositiveFiniteNumber, optionalWireframeLinecap, optionalWireframeLinejoin, textureUvChannel } from './materials.part-005'
import { cubeUvPackedImage, extractTextureFromSlot, filterModeToString, imageToRgbaTexture, minFilterModeToString, packedCubeUvTextureToFaceTextures, sampleCubeFace, textureLike, textureSourceData, wrapModeToString } from './materials.part-009'
import { assertSupportedBackgroundTexture, assertSupportedTextureInput } from './materials.part-010'
import { optionalTextureBoolean, textureAnisotropy, textureColorSpace, textureTransform, textureUnpackAlignment } from './materials.part-011'
export function activeMaterialNodeHooks(material: ThreeMaterialLike): string[] {
  const materialRecord = material as Record<string, unknown>
  const hookNames = new Set<string>(MaterialNodeHookProperties)

  for (const name of Object.keys(materialRecord)) {
    if (name.endsWith('Node')) hookNames.add(name)
  }

  const active: string[] = []
  for (const name of hookNames) {
    const value = materialRecord[name]
    if (value == null) continue
    if (MaterialNodeHookProperties.has(name) || materialNodeHookValue(value)) active.push(name)
  }
  return active
}

export function materialNodeHookValue(value: unknown): boolean {
  if (typeof value === 'function') return true
  if (typeof value !== 'object') return false
  return (value as { isNode?: unknown }).isNode === true ||
    (value as { isMRTNode?: unknown }).isMRTNode === true
}

export function isThreeCsmPatchedMaterial(material: ThreeMaterialLike): boolean {
  const defines = (material as { defines?: unknown }).defines
  return defines != null &&
    typeof defines === 'object' &&
    !Array.isArray(defines) &&
    (defines as Record<string, unknown>).USE_CSM != null &&
    (defines as Record<string, unknown>).CSM_CASCADES != null
}

export function assertSupportedMaterialState(
  material: ThreeMaterialLike,
  context: MaterialExtractionContext,
): void {
  optionalBoolean(material.wireframe, 'material.wireframe')
  optionalPositiveFiniteNumber(material.wireframeLinewidth, 'material.wireframeLinewidth')
  optionalWireframeLinecap(material.wireframeLinecap)
  optionalWireframeLinejoin(material.wireframeLinejoin)
  if (
    material.envMap != null &&
    supportsNativeMaterialEnvironmentMap(material) &&
    context.materialEnvironmentMaps?.has(material) !== true
  ) {
    throw new Error(
      'material.envMap on MeshBasicMaterial, MeshStandardMaterial, MeshPhysicalMaterial, MeshPhongMaterial, and MeshLambertMaterial requires one shared material envMap represented by the native IBL path. Use scene.environment, remove material.envMap when a scene environment or reflection probe is active, or render separate passes.',
    )
  }
}

export function assertSupportedMaterialClass(
  material: ThreeMaterialLike,
  customFragmentShader: string | undefined,
): void {
  if (customFragmentShader || supportedMaterialClass(material) || copyShaderMaterialInfo(material)) return

  const type = typeof material.type === 'string' && material.type.trim()
    ? material.type
    : 'Material'
  throw new Error(
    `${type} is not supported directly by @headless-three/renderer. Use a supported built-in Three.js material, or provide material.userData.headlessThreeRenderer.fragmentWgsl with a WGSL fragment body for the renderer's custom material path.`,
  )
}

export function supportedMaterialClass(material: ThreeMaterialLike): boolean {
  return material.isMeshBasicMaterial === true
    || material.isMeshDepthMaterial === true
    || material.isMeshDistanceMaterial === true
    || material.isMeshLambertMaterial === true
    || material.isMeshMatcapMaterial === true
    || material.isMeshNormalMaterial === true
    || material.isMeshPhongMaterial === true
    || material.isMeshStandardMaterial === true
    || material.isMeshPhysicalMaterial === true
    || material.isMeshToonMaterial === true
    || material.isShadowMaterial === true
    || material.isLineBasicMaterial === true
    || material.isLineDashedMaterial === true
    || material.isPointsMaterial === true
    || material.isSpriteMaterial === true
}

export function hasCustomOnBeforeCompile(material: ThreeMaterialLike): boolean {
  if (typeof material.onBeforeCompile !== 'function') return false
  return normalizeFunctionSource(material.onBeforeCompile) !== DefaultOnBeforeCompileSource
}

export function normalizeFunctionSource(fn: (...args: any[]) => unknown): string {
  return Function.prototype.toString.call(fn).replace(/\s+/g, ' ').trim()
}

export function shaderMaterialKind(material: ThreeMaterialLike): string | undefined {
  if (material.isRawShaderMaterial === true || material.type === 'RawShaderMaterial') {
    return 'RawShaderMaterial'
  }
  if (
    material.isNodeMaterial === true ||
    (typeof material.type === 'string' && material.type.includes('NodeMaterial'))
  ) {
    return 'NodeMaterial'
  }
  if (material.isShaderMaterial === true || material.type === 'ShaderMaterial') {
    return 'ShaderMaterial'
  }
  return undefined
}

export interface CopyShaderMaterialInfo {
  texture: unknown
  opacity: unknown
}

export function copyShaderMaterialInfo(material: ThreeMaterialLike | undefined): CopyShaderMaterialInfo | null {
  if (!material) return null
  const kind = shaderMaterialKind(material)
  if (kind !== 'ShaderMaterial' && kind !== 'RawShaderMaterial') return null
  if (kind === 'ShaderMaterial' && !isCopyShaderFragment(material.fragmentShader)) return null
  if (kind === 'RawShaderMaterial' && !isOutputShaderFragment(material.fragmentShader)) return null
  const uniforms = material.uniforms
  if (!uniforms || typeof uniforms !== 'object' || Array.isArray(uniforms)) return null
  return {
    texture: uniformValue((uniforms as Record<string, unknown>).tDiffuse),
    opacity: kind === 'ShaderMaterial'
      ? uniformValue((uniforms as Record<string, unknown>).opacity) ?? 1
      : 1,
  }
}

export function uniformValue(uniform: unknown): unknown {
  if (!uniform || typeof uniform !== 'object' || Array.isArray(uniform)) return undefined
  return (uniform as { value?: unknown }).value
}

export function isCopyShaderFragment(fragmentShader: unknown): boolean {
  if (typeof fragmentShader !== 'string') return false
  const compact = fragmentShader.replace(/\s+/g, '')
  return compact.includes('uniformfloatopacity;') &&
    compact.includes('uniformsampler2DtDiffuse;') &&
    compact.includes('texture2D(tDiffuse,vUv)') &&
    compact.includes('gl_FragColor=opacity*texel;')
}

export function isOutputShaderFragment(fragmentShader: unknown): boolean {
  if (typeof fragmentShader !== 'string') return false
  const compact = fragmentShader.replace(/\s+/g, '')
  return compact.includes('uniformsampler2DtDiffuse;') &&
    compact.includes('gl_FragColor=texture2D(tDiffuse,vUv);') &&
    compact.includes('tonemapping_pars_fragment') &&
    compact.includes('colorspace_pars_fragment')
}

export function extractTextureData(
  material: ThreeMaterialLike | undefined,
  context: MaterialExtractionContext = {},
): TextureInfo | null {
  const copyShader = copyShaderMaterialInfo(material)
  const slot = copyShader
    ? copyShader.texture as ThreeMaterialLike['map']
    : material?.isMeshMatcapMaterial
      ? material.matcap
      : material?.map
  const label = copyShader
    ? 'material.uniforms.tDiffuse.value'
    : material?.isMeshMatcapMaterial
      ? 'material.matcap'
      : 'material.map'
  const base = extractTextureFromSlot(slot, label, context.textureCache)
  if (!base) return null

  const map = slot as ThreeTextureLike | null | undefined
  const state = textureSamplerState(map, label, context, {
    includeWrap: material?.isMeshMatcapMaterial !== true,
    includeTransform: material?.isMeshMatcapMaterial !== true,
    includeUvChannel: material?.isMeshMatcapMaterial !== true,
  })
  return { ...base, ...state }
}

export interface CachedTextureStateExtraction {
  signature: TextureStateSignature
  state: TextureSamplerState
}

export interface TextureSamplerState {
  wrapS?: string
  wrapT?: string
  magFilter?: string
  minFilter?: string
  anisotropy?: number
  transform?: number[]
  colorSpace?: string
  usesUv2?: boolean
}

export interface TextureStateOptions {
  includeWrap?: boolean
  includeTransform?: boolean
  includeUvChannel?: boolean
}

export interface TextureStateSignature {
  includeWrap?: boolean
  includeTransform?: boolean
  includeUvChannel?: boolean
  wrapS?: unknown
  wrapT?: unknown
  magFilter?: unknown
  minFilter?: unknown
  generateMipmaps?: unknown
  mipmaps?: unknown
  mipmapCount?: number
  anisotropy?: unknown
  colorSpace?: unknown
  encoding?: unknown
  channel?: unknown
  flipY?: unknown
  matrixAutoUpdate?: unknown
  matrix?: unknown
  matrixElements?: unknown
  matrixValues?: unknown[]
  offset?: unknown
  offsetX?: unknown
  offsetY?: unknown
  repeat?: unknown
  repeatX?: unknown
  repeatY?: unknown
  rotation?: unknown
  center?: unknown
  centerX?: unknown
  centerY?: unknown
}

export function textureSamplerState(
  map: ThreeTextureLike | null | undefined,
  label: string,
  context: MaterialExtractionContext,
  options: TextureStateOptions = {},
): TextureSamplerState {
  const signature = map && context.textureStateCache
    ? textureStateSignature(map, options)
    : null
  if (map && signature) {
    const cached = context.textureStateCache?.get(map) as CachedTextureStateExtraction | undefined
    if (cached && sameTextureStateSignature(cached.signature, signature)) {
      return copyTextureSamplerState(cached.state)
    }
  }

  const state: TextureSamplerState = {
    wrapS: options.includeWrap === false ? undefined : wrapModeToString(map?.wrapS),
    wrapT: options.includeWrap === false ? undefined : wrapModeToString(map?.wrapT),
    magFilter: filterModeToString(map?.magFilter),
    minFilter: minFilterModeToString(map),
    anisotropy: textureAnisotropy(map, label),
    transform: options.includeTransform === false ? undefined : textureTransform(map, label),
    colorSpace: textureColorSpace(map),
    usesUv2: options.includeUvChannel === false ? false : textureUvChannel(map) > 0,
  }
  if (map && signature) {
    context.textureStateCache?.set(map, { signature, state: copyTextureSamplerState(state) })
  }
  return state
}

export function textureStateSignature(
  texture: ThreeTextureLike,
  options: TextureStateOptions,
): TextureStateSignature {
  const mipmaps = texture.mipmaps
  const matrix = texture.matrix
  const matrixElements = matrix?.elements
  return {
    includeWrap: options.includeWrap,
    includeTransform: options.includeTransform,
    includeUvChannel: options.includeUvChannel,
    wrapS: texture.wrapS,
    wrapT: texture.wrapT,
    magFilter: texture.magFilter,
    minFilter: texture.minFilter,
    generateMipmaps: texture.generateMipmaps,
    mipmaps,
    mipmapCount: Array.isArray(mipmaps) ? mipmaps.length : undefined,
    anisotropy: texture.anisotropy,
    colorSpace: texture.colorSpace,
    encoding: texture.encoding,
    channel: texture.channel,
    flipY: texture.flipY,
    matrixAutoUpdate: texture.matrixAutoUpdate,
    matrix,
    matrixElements,
    matrixValues: matrixElements ? Array.from(matrixElements as ArrayLike<unknown>) : undefined,
    offset: texture.offset,
    offsetX: texture.offset?.x,
    offsetY: texture.offset?.y,
    repeat: texture.repeat,
    repeatX: texture.repeat?.x,
    repeatY: texture.repeat?.y,
    rotation: texture.rotation,
    center: texture.center,
    centerX: texture.center?.x,
    centerY: texture.center?.y,
  }
}

export function copyTextureSamplerState(state: TextureSamplerState): TextureSamplerState {
  return {
    ...state,
    transform: state.transform ? state.transform.slice() : undefined,
  }
}

export function sameTextureStateSignature(a: TextureStateSignature, b: TextureStateSignature): boolean {
  return a.includeWrap === b.includeWrap
    && a.includeTransform === b.includeTransform
    && a.includeUvChannel === b.includeUvChannel
    && a.wrapS === b.wrapS
    && a.wrapT === b.wrapT
    && a.magFilter === b.magFilter
    && a.minFilter === b.minFilter
    && a.generateMipmaps === b.generateMipmaps
    && a.mipmaps === b.mipmaps
    && a.mipmapCount === b.mipmapCount
    && a.anisotropy === b.anisotropy
    && a.colorSpace === b.colorSpace
    && a.encoding === b.encoding
    && a.channel === b.channel
    && a.flipY === b.flipY
    && a.matrixAutoUpdate === b.matrixAutoUpdate
    && a.matrix === b.matrix
    && a.matrixElements === b.matrixElements
    && sameUnknownArray(a.matrixValues, b.matrixValues)
    && a.offset === b.offset
    && a.offsetX === b.offsetX
    && a.offsetY === b.offsetY
    && a.repeat === b.repeat
    && a.repeatX === b.repeatX
    && a.repeatY === b.repeatY
    && a.rotation === b.rotation
    && a.center === b.center
    && a.centerX === b.centerX
    && a.centerY === b.centerY
}

export function extractBackgroundTexture(
  background: ThreeSceneRootLike['background'] | ThreeTextureLike | number[] | undefined,
  label = 'background',
): TextureInfo | null {
  const map = textureLike(background)
  if (!map) return null
  if (isCubeBackgroundTexture(map)) {
    assertSupportedTextureInput(map, label)
    return extractCubeBackgroundTexture(map, label)
  }
  assertSupportedBackgroundTexture(map, label)

  const base = extractTextureFromSlot(map, label)
  if (!base) return null

  return {
    ...base,
    wrapS: wrapModeToString(map?.wrapS),
    wrapT: wrapModeToString(map?.wrapT),
    magFilter: filterModeToString(map?.magFilter),
    minFilter: minFilterModeToString(map),
    anisotropy: textureAnisotropy(map, label),
    transform: textureTransform(map, label),
    colorSpace: textureColorSpace(map),
    mapping: backgroundTextureMapping(map),
  }
}

export function backgroundTextureMapping(map: ThreeTextureLike): 'uv' | 'equirectangular' {
  return map.mapping === EquirectangularReflectionMapping || map.mapping === EquirectangularRefractionMapping
    ? 'equirectangular'
    : 'uv'
}

export function isCubeBackgroundTexture(map: ThreeTextureLike): boolean {
  return map.isCubeTexture === true ||
    map.mapping === CubeReflectionMapping ||
    map.mapping === CubeRefractionMapping ||
    map.mapping === CubeUVReflectionMapping
}

export function isCubeEnvironmentTexture(map: ThreeTextureLike, label = 'texture'): boolean {
  return map.isCubeTexture === true ||
    map.mapping === CubeReflectionMapping ||
    map.mapping === CubeRefractionMapping ||
    (
      map.mapping === CubeUVReflectionMapping &&
      (cubeFaceImages(map, label) !== null || cubeUvPackedImage(map, label) !== null)
    )
}

export function extractCubeBackgroundTexture(map: ThreeTextureLike, label: string): TextureInfo {
  const cube = cubeTextureToEquirectangular(map, label)
  return {
    ...cube,
    wrapS: 'repeat',
    wrapT: 'clamp',
    magFilter: filterModeToString(map.magFilter),
    minFilter: minFilterModeToString(map),
    anisotropy: textureAnisotropy(map, label),
    colorSpace: textureColorSpace(map),
    mapping: 'equirectangular',
  }
}

export function cubeTextureToEquirectangular(map: ThreeTextureLike, label: string): { data: Buffer; width: number; height: number } {
  textureUnpackAlignment(map, label)
  const faces = cubeFaceImages(map, label)
  if (!faces) {
    if (map.mapping === CubeUVReflectionMapping) {
      const packedFaceTextures = packedCubeUvTextureToFaceTextures(map, label)
      if (packedFaceTextures) return cubeFaceTexturesToEquirectangular(packedFaceTextures, label)
      throw new Error(`${label} uses PMREM/CubeUV mapping without readable six-face cube images or a readable packed CubeUV image.`)
    }
    throw new Error(
      `${label} uses a cube texture without six raw or encoded face images. Provide a CubeTexture with six DataTexture-style or encoded PNG/JPEG/WebP face images, use a 2D/equirectangular texture, or pre-render the background to a 2D image before rendering.`,
    )
  }

  const premultiplyAlpha = optionalTextureBoolean(map.premultiplyAlpha, `${label}.premultiplyAlpha`) === true
  const faceTextures = faces.map((face, index) => imageToRgbaTexture(face, `${label}.image[${index}]`, map.type, map.format, { premultiplyAlpha }))
  return cubeFaceTexturesToEquirectangular(faceTextures, label)
}

export function cubeFaceTexturesToEquirectangular(
  faceTextures: Array<{ rgba: Uint8Array; width: number; height: number }>,
  label: string,
): { data: Buffer; width: number; height: number } {
  const faceWidth = faceTextures[0].width
  const faceHeight = faceTextures[0].height
  if (faceWidth !== faceHeight) {
    throw new Error(`${label} cube background faces must be square raw RGBA images.`)
  }
  for (let i = 1; i < faceTextures.length; i += 1) {
    if (faceTextures[i].width !== faceWidth || faceTextures[i].height !== faceHeight) {
      throw new Error(`${label} cube background faces must all use the same dimensions.`)
    }
  }

  const width = Math.max(64, faceWidth * 4)
  const height = Math.max(32, faceHeight * 2)
  const out = new Uint8Array(width * height * 4)
  for (let y = 0; y < height; y += 1) {
    const v = (y + 0.5) / height
    const pitch = (v - 0.5) * Math.PI
    const dirY = Math.sin(pitch)
    const ring = Math.cos(pitch)
    for (let x = 0; x < width; x += 1) {
      const u = (x + 0.5) / width
      const yaw = (u - 0.5) * Math.PI * 2
      const dir = [
        Math.cos(yaw) * ring,
        dirY,
        Math.sin(yaw) * ring,
      ] as const
      const sample = sampleCubeFace(faceTextures, dir)
      out.set(sample, (y * width + x) * 4)
    }
  }

  return {
    data: Buffer.from(out.buffer, out.byteOffset, out.byteLength),
    width,
    height,
  }
}

export function cubeFaceImages(map: ThreeTextureLike, label = 'texture'): TextureImageInput[] | null {
  const sourceData = textureSourceData(map, label)
  const image = (map as any).image ?? sourceData
  if (Array.isArray(image) && image.length >= 6) return image.slice(0, 6) as TextureImageInput[]
  return null
}
