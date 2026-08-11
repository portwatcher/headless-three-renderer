import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { arrayBufferView, ensureDirectoryUrl, installAnimationPointerExtension, installEmbeddedGlbImageNormalizer, installEncodedImageSupportProbe, installLocalFileFetch, installNodeVisibilityExtension, isAbsoluteOrSpecialAssetUrl, registerEncodedImageHandlers, resolveLoadingManagerUrl, resolveLocalAssetPath, resolveLocalRootDir } from './loaders.part-002'
import { importGltfLoader, importVrmAnimationClipFactory, importVrmAnimationLoaderPlugin, importVrmLoaderPlugin, objectOptions, optionalBoolean, optionalFunction, optionalLoadingManager, optionalNonNegativeInteger, optionalNonNegativeNumber, optionalString, registerLoaderPlugin, requiredString, validateLoadGltfFromFileOptions, validateNodeGltfLoaderOptions } from './loaders.part-003'
// eslint-disable-next-line @typescript-eslint/no-var-requires
export const { AnimationMixer, ColorKeyframeTrack, InterpolateDiscrete, InterpolateLinear, LoadingManager, Texture } = require('three') as {
  AnimationMixer: AnimationMixerConstructor
  ColorKeyframeTrack: KeyframeTrackConstructor
  InterpolateDiscrete: number
  InterpolateLinear: number
  LoadingManager: new () => ThreeLoadingManagerLike
  Texture: new () => TextureLike
}

export type VrmAnimationActionLike = {
  play(): unknown
}
export type VrmAnimationMixerLike = {
  clipAction(clip: unknown): VrmAnimationActionLike
  setTime?(time: number): unknown
  update?(delta: number): unknown
}
export type AnimationMixerConstructor = new (root: unknown) => VrmAnimationMixerLike
export type VrmAnimationClipFactory = (vrmAnimation: unknown, vrm: unknown) => unknown
export type KeyframeTrackConstructor = new (name: string, times: ArrayLike<number>, values: ArrayLike<number>, interpolation?: number) => unknown
export type ApplyVrmAnimationOptions = {
  AnimationMixer?: AnimationMixerConstructor
  animationIndex?: number
  createVRMAnimationClip?: VrmAnimationClipFactory
  time?: number
  updateDelta?: number
  updateVrm?: boolean
}
export type AppliedVrmAnimation = {
  action: VrmAnimationActionLike
  clip: unknown
  mixer: VrmAnimationMixerLike
}
export type TextureLike = {
  image?: unknown
  source: { data?: unknown }
  needsUpdate?: boolean
  isTexture?: boolean
}
export type ThreeLoadingManagerLike = {
  addHandler(regex: RegExp, loader: unknown): unknown
  itemEnd?(url: string): unknown
  itemError?(url: string): unknown
  itemStart?(url: string): unknown
  resolveURL?(url: string): string
}
export type ThreeGltfLoaderLike = {
  parse(data: ArrayBuffer | string, path: string, onLoad: (gltf: unknown) => void, onError?: (error: unknown) => void): void
  register?(callback: unknown): unknown
  setPath?(path: string): ThreeGltfLoaderLike
}
export type GltfLoaderCtor = new (manager?: ThreeLoadingManagerLike) => ThreeGltfLoaderLike
export type GltfLoaderModule = {
  GLTFLoader: GltfLoaderCtor
}
export type VrmLoaderPluginConstructor = new (parser: unknown) => unknown
export type GltfRootLike = {
  scene?: unknown
  scenes?: unknown[]
}
export type VrmLike = {
  scene?: unknown
  update?: unknown
}
export type GltfNodeVisibilityParser = {
  associations?: Map<unknown, Record<string, number>>
  json?: {
    nodes?: unknown[]
  }
}
export type GltfAnimationPointerParser = {
  associations?: Map<unknown, Record<string, number>>
  getDependency?: (type: string, index: number) => Promise<GltfAccessorLike>
  json?: {
    animations?: GltfAnimationDef[]
  }
}
export type GltfAnimationDef = {
  channels?: GltfAnimationChannelDef[]
  samplers?: GltfAnimationSamplerDef[]
}
export type GltfAnimationChannelDef = {
  sampler?: number
  target?: {
    path?: string
    extensions?: {
      KHR_animation_pointer?: {
        pointer?: string
      }
    }
  }
}
export type GltfAnimationSamplerDef = {
  input?: number
  output?: number
  interpolation?: string
}
export type GltfAccessorLike = {
  array?: ArrayLike<number>
  itemSize?: number
}
export type AnimationClipLike = {
  tracks?: unknown[]
  resetDuration?: () => unknown
}
export type MaterialLike = {
  name?: string
}
export type MeshLike = TraversableObjectLike & {
  isMesh?: boolean
  name?: string
  uuid?: string
  material?: MaterialLike | MaterialLike[]
}
export type TraversableObjectLike = {
  traverse?: (callback: (object: unknown) => void) => void
  visible?: boolean
}
export type TextureLoadCallback = (texture: TextureLike) => void
export type TextureErrorCallback = (error: unknown) => void

export type ConfigureGltfLoader = (loader: ThreeGltfLoaderLike) => void | Promise<void>

export type NodeGltfLoaderOptions = {
  configureLoader?: ConfigureGltfLoader
  installFetch?: boolean
  manager?: ThreeLoadingManagerLike
  registerTextureHandlers?: boolean
}

export type LoadGltfFromFileOptions = NodeGltfLoaderOptions & {
  baseUrl?: string
  rootDir?: string
}

export type LoadVrmFromFileOptions = LoadGltfFromFileOptions & {
  VRMLoaderPlugin?: VrmLoaderPluginConstructor
}

export type LoadVrmAnimationFromFileOptions = LoadGltfFromFileOptions & {
  VRMAnimationLoaderPlugin?: VrmLoaderPluginConstructor
  VRMLoaderPlugin?: VrmLoaderPluginConstructor
}

export type NodeGltfLoaderBundle = {
  encodedImages: EncodedImageTextureLoader
  loader: ThreeGltfLoaderLike
  manager: ThreeLoadingManagerLike
  rootDir: string
}

export class EncodedImageTextureLoader {
  private readonly rootDir: string
  private readonly manager?: ThreeLoadingManagerLike
  private loaderPath = ''

  constructor(rootDir: string = process.cwd(), manager?: ThreeLoadingManagerLike) {
    this.rootDir = resolveLocalRootDir(rootDir, 'rootDir')
    this.manager = optionalLoadingManager(manager, 'manager')
  }

  setCrossOrigin(): this {
    return this
  }

  setRequestHeader(): this {
    return this
  }

  setWithCredentials(): this {
    return this
  }

  setPath(loaderPath: string): this {
    this.loaderPath = validatedTextureLoaderPath(loaderPath, 'loaderPath')
    return this
  }

  load(
    url: string,
    onLoad?: TextureLoadCallback,
    _onProgress?: unknown,
    onError?: TextureErrorCallback,
  ): TextureLike {
    const texture = new Texture()
    const requestedUrl = requiredString(url, 'url')
    const loadCallback = optionalFunction(onLoad, 'onLoad')
    const errorCallback = optionalFunction(onError, 'onError')
    const source = resolveLoadingManagerUrl(
      this.manager,
      resolveTextureLoaderSource(requestedUrl, this.loaderPath),
    )
    const encodedDataUri = encodedImageDataUriBuffer(source)
    const data = encodedDataUri
      ? Promise.resolve(encodedDataUri)
      : /^blob:/i.test(source)
        ? readBlobUrlBuffer(source)
        : readFile(resolveLocalAssetPath(source, this.rootDir))

    this.manager?.itemStart?.(source)
    data.then((buffer) => {
      texture.image = buffer
      texture.source.data = buffer
      texture.needsUpdate = true
      try {
        loadCallback?.(texture)
      } finally {
        this.manager?.itemEnd?.(source)
      }
    }, (error) => {
      this.manager?.itemError?.(source)
      this.manager?.itemEnd?.(source)
      errorCallback?.(error)
    })

    return texture
  }

  loadAsync(url: string, onProgress?: unknown): Promise<TextureLike> {
    return new Promise((resolve, reject) => {
      try {
        this.load(url, resolve, onProgress, reject)
      } catch (error) {
        reject(error)
      }
    })
  }
}

export function createEncodedImageTextureLoader(rootDir?: string, manager?: ThreeLoadingManagerLike): EncodedImageTextureLoader {
  return new EncodedImageTextureLoader(rootDir, manager)
}

export async function createNodeGltfLoader(
  rootDir: string = process.cwd(),
  options: NodeGltfLoaderOptions = {},
): Promise<NodeGltfLoaderBundle> {
  const loaderOptions = objectOptions(options, 'options') as NodeGltfLoaderOptions
  const root = resolveLocalRootDir(rootDir, 'rootDir')
  const {
    configureLoader,
    installFetch,
    manager,
    registerTextureHandlers,
  } = validateNodeGltfLoaderOptions(loaderOptions)
  const loadingManager = manager ?? new LoadingManager()

  if (installFetch !== false) {
    installLocalFileFetch()
  }
  installEncodedImageSupportProbe()

  const encodedImages = createEncodedImageTextureLoader(root, loadingManager)
  if (registerTextureHandlers !== false) {
    registerEncodedImageHandlers(loadingManager, encodedImages)
  }

  const { GLTFLoader } = await importGltfLoader()
  const loader = new GLTFLoader(loadingManager)
  loader.setPath?.(pathToFileURL(`${root}${path.sep}`).href)
  await configureLoader?.(loader)
  installAnimationPointerExtension(loader)
  installNodeVisibilityExtension(loader)
  installEmbeddedGlbImageNormalizer(loader)
  return { encodedImages, loader, manager: loadingManager, rootDir: root }
}

export async function loadGltfFromFile<T = unknown>(
  filePath: string,
  options: LoadGltfFromFileOptions = {},
): Promise<T> {
  const loaderOptions = objectOptions(options, 'options') as LoadGltfFromFileOptions
  validateLoadGltfFromFileOptions(loaderOptions)
  const absolute = resolveLocalAssetPath(requiredString(filePath, 'filePath'))
  const rootDirOption = optionalString(loaderOptions.rootDir, 'options.rootDir')
  const root = rootDirOption == null ? path.dirname(absolute) : resolveLocalRootDir(rootDirOption, 'options.rootDir')
  const baseUrl = optionalString(loaderOptions.baseUrl, 'options.baseUrl') ?? pathToFileURL(`${root}${path.sep}`).href
  const { loader } = await createNodeGltfLoader(root, loaderOptions)
  const bytes = await readFile(absolute)

  return await new Promise<T>((resolve, reject) => {
    loader.parse(arrayBufferView(bytes), baseUrl, (gltf) => resolve(gltf as T), reject)
  })
}

export async function loadVrmFromFile<T = unknown>(
  filePath: string,
  options: LoadVrmFromFileOptions = {},
): Promise<T> {
  const loaderOptions = objectOptions(options, 'options') as LoadVrmFromFileOptions
  validateLoadGltfFromFileOptions(loaderOptions)
  const validatedFilePath = requiredString(filePath, 'filePath')
  const {
    VRMLoaderPlugin,
    configureLoader,
    ...gltfOptions
  } = loaderOptions
  const configureLoaderCallback = optionalFunction(configureLoader, 'options.configureLoader')
  const VrmPlugin = optionalFunction(VRMLoaderPlugin, 'options.VRMLoaderPlugin') ?? await importVrmLoaderPlugin()

  return await loadGltfFromFile<T>(validatedFilePath, {
    ...gltfOptions,
    configureLoader: async (loader) => {
      registerLoaderPlugin(loader, (parser) => new VrmPlugin(parser), 'VRMLoaderPlugin')
      await configureLoaderCallback?.(loader)
    },
  })
}

export async function loadVrmAnimationFromFile<T = unknown>(
  filePath: string,
  options: LoadVrmAnimationFromFileOptions = {},
): Promise<T> {
  const loaderOptions = objectOptions(options, 'options') as LoadVrmAnimationFromFileOptions
  validateLoadGltfFromFileOptions(loaderOptions)
  const validatedFilePath = requiredString(filePath, 'filePath')
  const {
    VRMAnimationLoaderPlugin,
    VRMLoaderPlugin,
    configureLoader,
    ...gltfOptions
  } = loaderOptions
  const VrmPlugin = optionalFunction(VRMLoaderPlugin, 'options.VRMLoaderPlugin')
  const configureLoaderCallback = optionalFunction(configureLoader, 'options.configureLoader')
  const AnimationPlugin = optionalFunction(VRMAnimationLoaderPlugin, 'options.VRMAnimationLoaderPlugin') ?? await importVrmAnimationLoaderPlugin()

  return await loadGltfFromFile<T>(validatedFilePath, {
    ...gltfOptions,
    configureLoader: async (loader) => {
      if (VrmPlugin) {
        registerLoaderPlugin(loader, (parser) => new VrmPlugin(parser), 'VRMLoaderPlugin')
      }
      registerLoaderPlugin(loader, (parser) => new AnimationPlugin(parser), 'VRMAnimationLoaderPlugin')
      await configureLoaderCallback?.(loader)
    },
  })
}

export async function applyVrmAnimation(
  vrm: unknown,
  vrmAnimation: unknown,
  options: ApplyVrmAnimationOptions = {},
): Promise<AppliedVrmAnimation> {
  const helperOptions = objectOptions(options, 'options') as ApplyVrmAnimationOptions
  const model = resolveVrmModelInput(vrm)
  const animationIndex = optionalNonNegativeInteger(helperOptions.animationIndex, 'options.animationIndex') ?? 0
  const animation = resolveVrmAnimationInput(vrmAnimation, animationIndex)

  const time = optionalNonNegativeNumber(helperOptions.time, 'options.time') ?? 0
  const updateDelta = optionalNonNegativeNumber(helperOptions.updateDelta, 'options.updateDelta') ?? 0
  const updateVrm = optionalBoolean(helperOptions.updateVrm, 'options.updateVrm') ?? true
  const createClip = optionalFunction(helperOptions.createVRMAnimationClip, 'options.createVRMAnimationClip')
    ?? await importVrmAnimationClipFactory()
  const Mixer = optionalFunction(helperOptions.AnimationMixer, 'options.AnimationMixer') ?? AnimationMixer
  const clip = createClip(animation, model)
  const mixer = new Mixer(model.scene)
  if (typeof mixer.clipAction !== 'function') {
    throw new TypeError('AnimationMixer must provide a clipAction() function.')
  }
  const action = mixer.clipAction(clip)
  if (!action || typeof action.play !== 'function') {
    throw new TypeError('AnimationMixer.clipAction() must return an action with play().')
  }
  action.play()
  if (typeof mixer.setTime === 'function') {
    mixer.setTime(time)
  } else if (typeof mixer.update === 'function') {
    mixer.update(time)
  } else {
    throw new TypeError('AnimationMixer must provide setTime() or update().')
  }

  if (updateVrm && model.update != null) {
    if (typeof model.update !== 'function') {
      throw new TypeError('vrm.update must be a function when provided.')
    }
    model.update(updateDelta)
  }
  return { action, clip, mixer }
}

export function resolveVrmModelInput(vrm: unknown): VrmLike {
  const input = objectOptions(vrm, 'vrm') as Record<string, unknown>
  const userData = input.userData
  const wrapped = userData != null
    && typeof userData === 'object'
    && !Array.isArray(userData)
    && 'vrm' in userData
  const label = wrapped ? 'vrm.userData.vrm' : 'vrm'
  const model = objectOptions(wrapped ? (userData as Record<string, unknown>).vrm : vrm, label) as VrmLike
  if (model.scene == null || typeof model.scene !== 'object' || Array.isArray(model.scene)) {
    throw new TypeError(`${label}.scene must be an object.`)
  }
  return model
}

export function resolveVrmAnimationInput(vrmAnimation: unknown, animationIndex = 0): unknown {
  const input = objectOptions(vrmAnimation, 'vrmAnimation') as Record<string, unknown>
  const userData = input.userData
  if (userData == null || typeof userData !== 'object' || Array.isArray(userData) || !('vrmAnimations' in userData)) {
    return vrmAnimation
  }

  const animations = (userData as Record<string, unknown>).vrmAnimations
  if (!Array.isArray(animations)) {
    throw new TypeError('vrmAnimation.userData.vrmAnimations must be an array.')
  }
  objectOptions(animations[animationIndex], `vrmAnimation.userData.vrmAnimations[${animationIndex}]`)
  return animations[animationIndex]
}

export function encodedImageDataUriBuffer(url: string): Buffer | null {
  if (!/^data:/i.test(url)) return null
  const comma = url.indexOf(',')
  if (comma < 0) {
    throw new Error('Data URI texture is missing a comma separator.')
  }

  const metadata = url.slice(5, comma).toLowerCase()
  if (!/^image\/(?:png|jpe?g|webp)(?:;|$)/.test(metadata)) {
    throw new Error('Data URI texture is not a supported encoded image. Use PNG, JPEG, or WebP data URIs.')
  }
  const payload = url.slice(comma + 1)
  if (/(?:^|;)base64(?:;|$)/.test(metadata)) {
    return Buffer.from(payload, 'base64')
  }
  return Buffer.from(decodeURIComponent(payload), 'utf8')
}

export async function readBlobUrlBuffer(url: string): Promise<Buffer> {
  if (typeof fetch !== 'function') {
    throw new Error('Blob URL textures require global fetch support. Install a fetch polyfill or rewrite embedded images as files/data URIs before loading.')
  }
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Blob URL texture fetch failed with status ${response.status}.`)
  }
  const contentType = response.headers.get('content-type')?.split(';', 1)[0].trim().toLowerCase()
  if (contentType && !/^image\/(?:png|jpe?g|webp)$/.test(contentType)) {
    throw new Error(`Blob URL texture has unsupported content type "${contentType}". Use PNG, JPEG, or WebP embedded images.`)
  }
  return Buffer.from(await response.arrayBuffer())
}

export function resolveTextureLoaderSource(url: string, loaderPath: string): string {
  if (!loaderPath || isAbsoluteOrSpecialAssetUrl(url)) return url
  if (path.isAbsolute(loaderPath)) return path.join(loaderPath, url)
  if (path.win32.isAbsolute(loaderPath)) return path.win32.join(loaderPath, url)
  if (/^[a-z][a-z0-9+.-]*:/i.test(loaderPath)) {
    return new URL(url, ensureDirectoryUrl(loaderPath)).href
  }
  return path.join(loaderPath, url)
}

export function validatedTextureLoaderPath(loaderPath: unknown, label: string): string {
  const value = requiredString(loaderPath, label)
  if (!value || path.isAbsolute(value) || path.win32.isAbsolute(value) || /^file:/i.test(value)) return value
  if (/^[a-z][a-z0-9+.-]*:/i.test(value)) {
    throw new Error(`${label} is not a local directory path: ${value}`)
  }
  return value
}
