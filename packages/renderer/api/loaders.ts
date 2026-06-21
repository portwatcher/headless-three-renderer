import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

// eslint-disable-next-line @typescript-eslint/no-var-requires
const { AnimationMixer, ColorKeyframeTrack, InterpolateDiscrete, InterpolateLinear, LoadingManager, Texture } = require('three') as {
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
type KeyframeTrackConstructor = new (name: string, times: ArrayLike<number>, values: ArrayLike<number>, interpolation?: number) => unknown
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
type TextureLike = {
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
}
type GltfLoaderCtor = new (manager?: ThreeLoadingManagerLike) => ThreeGltfLoaderLike
type GltfLoaderModule = {
  GLTFLoader: GltfLoaderCtor
}
export type VrmLoaderPluginConstructor = new (parser: unknown) => unknown
type GltfRootLike = {
  scene?: unknown
  scenes?: unknown[]
}
type VrmLike = {
  scene?: unknown
  update?: unknown
}
type GltfNodeVisibilityParser = {
  associations?: Map<unknown, Record<string, number>>
  json?: {
    nodes?: unknown[]
  }
}
type GltfAnimationPointerParser = {
  associations?: Map<unknown, Record<string, number>>
  getDependency?: (type: string, index: number) => Promise<GltfAccessorLike>
  json?: {
    animations?: GltfAnimationDef[]
  }
}
type GltfAnimationDef = {
  channels?: GltfAnimationChannelDef[]
  samplers?: GltfAnimationSamplerDef[]
}
type GltfAnimationChannelDef = {
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
type GltfAnimationSamplerDef = {
  input?: number
  output?: number
  interpolation?: string
}
type GltfAccessorLike = {
  array?: ArrayLike<number>
  itemSize?: number
}
type AnimationClipLike = {
  tracks?: unknown[]
  resetDuration?: () => unknown
}
type MaterialLike = {
  name?: string
}
type MeshLike = TraversableObjectLike & {
  isMesh?: boolean
  name?: string
  uuid?: string
  material?: MaterialLike | MaterialLike[]
}
type TraversableObjectLike = {
  traverse?: (callback: (object: unknown) => void) => void
  visible?: boolean
}
type TextureLoadCallback = (texture: TextureLike) => void
type TextureErrorCallback = (error: unknown) => void

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
    this.loaderPath = requiredString(loaderPath, 'loaderPath')
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

function resolveVrmModelInput(vrm: unknown): VrmLike {
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

function resolveVrmAnimationInput(vrmAnimation: unknown, animationIndex = 0): unknown {
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

function encodedImageDataUriBuffer(url: string): Buffer | null {
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

async function readBlobUrlBuffer(url: string): Promise<Buffer> {
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

function resolveTextureLoaderSource(url: string, loaderPath: string): string {
  if (!loaderPath || isAbsoluteOrSpecialAssetUrl(url)) return url
  if (path.isAbsolute(loaderPath)) return path.join(loaderPath, url)
  if (path.win32.isAbsolute(loaderPath)) return path.win32.join(loaderPath, url)
  if (/^[a-z][a-z0-9+.-]*:/i.test(loaderPath)) {
    return new URL(url, ensureDirectoryUrl(loaderPath)).href
  }
  return path.join(loaderPath, url)
}

function resolveLoadingManagerUrl(manager: ThreeLoadingManagerLike | undefined, url: string): string {
  const resolveURL = manager?.resolveURL
  if (resolveURL == null) return url
  const resolved = resolveURL.call(manager, url)
  return requiredString(resolved, 'manager.resolveURL return value')
}

function isAbsoluteOrSpecialAssetUrl(url: string): boolean {
  return /^data:/i.test(url)
    || /^blob:/i.test(url)
    || path.isAbsolute(url)
    || path.win32.isAbsolute(url)
    || /^file:/i.test(url)
    || /^[a-z][a-z0-9+.-]*:/i.test(url)
}

function ensureDirectoryUrl(url: string): string {
  return url.endsWith('/') ? url : `${url}/`
}

export function resolveLocalAssetPath(url: string, rootDir: string = process.cwd()): string {
  const assetUrl = requiredString(url, 'url')
  requiredString(rootDir, 'rootDir')
  if (/^data:/i.test(assetUrl)) {
    throw new Error('Data URI textures should be decoded or written to files before loading in Node.')
  }
  if (path.isAbsolute(assetUrl) || path.win32.isAbsolute(assetUrl)) return path.normalize(assetUrl)
  if (/^file:/i.test(assetUrl)) return fileURLToPath(assetUrl)
  if (/^[a-z][a-z0-9+.-]*:/i.test(assetUrl)) {
    throw new Error(`Remote texture URL is not a local file: ${assetUrl}`)
  }
  const root = resolveLocalRootDir(rootDir, 'rootDir')
  return path.resolve(root, assetUrl)
}

function resolveLocalRootDir(rootDir: string, label: string): string {
  const value = requiredString(rootDir, label)
  if (/^data:/i.test(value)) {
    throw new Error(`${label} must be a local directory path, not a data URI.`)
  }
  if (path.isAbsolute(value) || path.win32.isAbsolute(value)) {
    return path.normalize(value)
  }
  if (/^file:/i.test(value)) {
    return fileURLToPath(value)
  }
  if (/^[a-z][a-z0-9+.-]*:/i.test(value)) {
    throw new Error(`${label} is not a local directory path: ${value}`)
  }
  return path.resolve(value)
}

export function installLocalFileFetch(): void {
  const marker = Symbol.for('headless-three-renderer.local-file-fetch')
  const globalScope = globalThis as any
  if (typeof globalScope.self === 'undefined') {
    globalScope.self = globalScope
  }
  if (globalScope[marker]) return

  if (typeof globalScope.ProgressEvent === 'undefined') {
    const EventCtor = typeof globalScope.Event === 'function'
      ? globalScope.Event
      : class Event {
        constructor(public readonly type: string) {}
      }
    globalScope.ProgressEvent = class ProgressEvent extends EventCtor {
      readonly lengthComputable: boolean
      readonly loaded: number
      readonly total: number

      constructor(type: string, init: { lengthComputable?: boolean; loaded?: number; total?: number } = {}) {
        super(type)
        this.lengthComputable = init.lengthComputable ?? false
        this.loaded = init.loaded ?? 0
        this.total = init.total ?? 0
      }
    }
  }

  const nativeFetch = globalScope.fetch
  globalScope.fetch = async (input: any, init?: any): Promise<any> => {
    const url = typeof input === 'string' || input instanceof URL ? String(input) : input?.url
    if (typeof url === 'string' && url.startsWith('file:')) {
      const buffer = await readFile(fileURLToPath(url))
      const ResponseCtor = globalScope.Response
      if (typeof ResponseCtor !== 'function') {
        throw new Error('global Response is not available; install a fetch polyfill before loading local files.')
      }
      return new ResponseCtor(buffer)
    }
    if (typeof nativeFetch !== 'function') {
      throw new Error('global fetch is not available; install a fetch polyfill before loading remote assets.')
    }
    return nativeFetch(input, init)
  }

  globalScope[marker] = true
}

function installEncodedImageSupportProbe(): void {
  const globalScope = globalThis as any
  if (typeof globalScope.Image === 'function') return
  const marker = Symbol.for('headless-three-renderer.encoded-image-probe')
  if (globalScope[marker]) return

  globalScope.Image = class EncodedImageProbe {
    height = 0
    onerror: (() => unknown) | null = null
    onload: (() => unknown) | null = null
    width = 0

    set src(value: unknown) {
      const url = String(value ?? '')
      const supported = /^data:image\/webp[;,]/i.test(url)
      this.width = supported ? 1 : 0
      this.height = supported ? 1 : 0
      queueMicrotask(() => {
        if (supported) {
          this.onload?.()
        } else {
          this.onerror?.()
        }
      })
    }
  }
  globalScope[marker] = true
}

function registerEncodedImageHandlers(manager: ThreeLoadingManagerLike, encodedImages: EncodedImageTextureLoader): void {
  manager.addHandler(/^blob:/i, encodedImages)
  manager.addHandler(/^data:image\/(?:png|jpe?g|webp)/i, encodedImages)
  manager.addHandler(/\.(png|jpe?g|webp)$/i, encodedImages)
}

function arrayBufferView(buffer: Buffer): ArrayBuffer {
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength) as ArrayBuffer
}

function installEmbeddedGlbImageNormalizer(loader: ThreeGltfLoaderLike): void {
  const originalParse = loader.parse.bind(loader)
  loader.parse = (data, parsePath, onLoad, onError) => {
    originalParse(normalizeEmbeddedGlbImages(data), parsePath, onLoad, onError)
  }
}

function installNodeVisibilityExtension(loader: ThreeGltfLoaderLike): void {
  registerLoaderPlugin(loader, (parser) => ({
    name: 'KHR_node_visibility',
    afterRoot(gltf: unknown) {
      applyNodeVisibilityExtension(gltf, parser as GltfNodeVisibilityParser)
    },
  }), 'KHR_node_visibility')
}

function installAnimationPointerExtension(loader: ThreeGltfLoaderLike): void {
  registerLoaderPlugin(loader, (parser) => ({
    name: 'KHR_animation_pointer',
    afterRoot(gltf: unknown) {
      return applyAnimationPointerExtension(gltf, parser as GltfAnimationPointerParser)
    },
  }), 'KHR_animation_pointer')
}

async function applyAnimationPointerExtension(gltf: unknown, parser: GltfAnimationPointerParser): Promise<void> {
  const animationDefs = parser.json?.animations
  const clips = (gltf as { animations?: AnimationClipLike[] }).animations
  if (!Array.isArray(animationDefs) || !Array.isArray(clips) || typeof parser.getDependency !== 'function') return

  for (let animationIndex = 0; animationIndex < animationDefs.length; animationIndex++) {
    const animationDef = animationDefs[animationIndex]
    const clip = clips[animationIndex]
    if (!clip || !Array.isArray(clip.tracks) || !Array.isArray(animationDef.channels) || !Array.isArray(animationDef.samplers)) continue

    for (const channel of animationDef.channels) {
      const pointer = channel.target?.extensions?.KHR_animation_pointer?.pointer
      const match = typeof pointer === 'string'
        ? pointer.match(/^\/materials\/(\d+)\/pbrMetallicRoughness\/baseColorFactor$/)
        : null
      if (!match) continue

      const samplerIndex = channel.sampler
      const sampler = typeof samplerIndex === 'number' ? animationDef.samplers[samplerIndex] : undefined
      if (!sampler || sampler.interpolation === 'CUBICSPLINE' || typeof sampler.input !== 'number' || typeof sampler.output !== 'number') continue

      const materialIndex = Number.parseInt(match[1], 10)
      const targets = materialAnimationTargets(gltf, parser, materialIndex)
      if (targets.length === 0) continue

      const [inputAccessor, outputAccessor] = await Promise.all([
        parser.getDependency('accessor', sampler.input),
        parser.getDependency('accessor', sampler.output),
      ])
      const times = inputAccessor.array
      const values = outputAccessor.array
      const itemSize = outputAccessor.itemSize ?? 0
      if (!times || !values || itemSize < 3) continue

      const colorValues = animationPointerColorValues(times.length, values, itemSize)
      const interpolation = sampler.interpolation === 'STEP' ? InterpolateDiscrete : InterpolateLinear
      for (const target of targets) {
        clip.tracks.push(new ColorKeyframeTrack(`${target}.material.color`, times, colorValues, interpolation))
      }
      clip.resetDuration?.()
    }
  }
}

function materialAnimationTargets(gltf: unknown, parser: GltfAnimationPointerParser, materialIndex: number): string[] {
  const associations = parser.associations
  if (!associations) return []

  const targets: string[] = []
  const seen = new Set<string>()
  for (const root of gltfRootScenes(gltf)) {
    root.traverse?.((object) => {
      const mesh = object as MeshLike
      if (mesh.isMesh !== true) return
      const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material]
      if (!materials.some((material) => associations.get(material)?.materials === materialIndex)) return

      const target = mesh.name || mesh.uuid
      if (target && !seen.has(target)) {
        seen.add(target)
        targets.push(target)
      }
    })
  }
  return targets
}

function animationPointerColorValues(keyframeCount: number, values: ArrayLike<number>, itemSize: number): Float32Array {
  const colors = new Float32Array(keyframeCount * 3)
  for (let keyframe = 0; keyframe < keyframeCount; keyframe++) {
    const source = keyframe * itemSize
    const target = keyframe * 3
    colors[target] = Number(values[source] ?? 0)
    colors[target + 1] = Number(values[source + 1] ?? 0)
    colors[target + 2] = Number(values[source + 2] ?? 0)
  }
  return colors
}

function applyNodeVisibilityExtension(gltf: unknown, parser: GltfNodeVisibilityParser): void {
  const associations = parser.associations
  const nodes = parser.json?.nodes
  if (!associations || !Array.isArray(nodes)) return
  const nodeIndexByName = uniqueNodeIndexByName(nodes)

  for (const root of gltfRootScenes(gltf)) {
    root.traverse?.((object) => {
      const association = associations.get(object)
      let nodeIndex = association?.nodes
      const objectName = typeof (object as { name?: unknown }).name === 'string'
        ? (object as { name: string }).name
        : ''
      const associatedNode = typeof nodeIndex === 'number' && Number.isInteger(nodeIndex)
        ? nodes[nodeIndex]
        : undefined
      const associatedName = typeof (associatedNode as { name?: unknown } | undefined)?.name === 'string'
        ? (associatedNode as { name: string }).name
        : ''
      if (objectName && associatedName !== objectName) {
        nodeIndex = nodeIndexByName.get(objectName) ?? nodeIndex
      }
      if (typeof nodeIndex !== 'number' || !Number.isInteger(nodeIndex)) return
      const node = nodes[nodeIndex]
      if (!node || typeof node !== 'object' || Array.isArray(node)) return
      const visible = (node as any).extensions?.KHR_node_visibility?.visible
      const objectLike = object as TraversableObjectLike
      if (visible === false) {
        objectLike.visible = false
      } else if (visible === true) {
        objectLike.visible = true
      }
    })
  }
}

function uniqueNodeIndexByName(nodes: unknown[]): Map<string, number> {
  const counts = new Map<string, number>()
  for (const node of nodes) {
    const name = typeof (node as { name?: unknown } | undefined)?.name === 'string'
      ? (node as { name: string }).name
      : ''
    if (name) counts.set(name, (counts.get(name) ?? 0) + 1)
  }

  const result = new Map<string, number>()
  nodes.forEach((node, index) => {
    const name = typeof (node as { name?: unknown } | undefined)?.name === 'string'
      ? (node as { name: string }).name
      : ''
    if (name && counts.get(name) === 1) result.set(name, index)
  })
  return result
}

function gltfRootScenes(gltf: unknown): TraversableObjectLike[] {
  const root = gltf as GltfRootLike
  const scenes = Array.isArray(root?.scenes) ? root.scenes : []
  const result = scenes.filter(isTraversableObject)
  if (isTraversableObject(root?.scene) && !result.includes(root.scene)) {
    result.push(root.scene)
  }
  return result
}

function isTraversableObject(value: unknown): value is TraversableObjectLike {
  return !!value && typeof value === 'object' && typeof (value as TraversableObjectLike).traverse === 'function'
}

function normalizeEmbeddedGlbImages(data: ArrayBuffer | string): ArrayBuffer | string {
  if (typeof data === 'string') {
    validateGltfTextImageReferences(data)
    return data
  }
  const bytes = Buffer.from(data)
  const normalized = rewriteGlbBufferViewImages(bytes)
  if (!normalized) {
    validateGltfTextImageReferences(bytes)
  }
  return normalized ? arrayBufferView(normalized) : data
}

function rewriteGlbBufferViewImages(bytes: Buffer): Buffer | null {
  if (bytes.byteLength < 20 || bytes.readUInt32LE(0) !== 0x46546c67 || bytes.readUInt32LE(4) !== 2) {
    return null
  }

  let offset = 12
  let jsonChunk: Buffer | null = null
  let binChunk: Buffer | null = null
  while (offset + 8 <= bytes.byteLength) {
    const chunkLength = bytes.readUInt32LE(offset)
    const chunkType = bytes.readUInt32LE(offset + 4)
    const chunkStart = offset + 8
    const chunkEnd = chunkStart + chunkLength
    if (chunkEnd > bytes.byteLength) return null
    const chunk = bytes.subarray(chunkStart, chunkEnd)
    if (chunkType === 0x4e4f534a) {
      jsonChunk = chunk
    } else if (chunkType === 0x004e4942 && binChunk === null) {
      binChunk = chunk
    }
    offset = chunkEnd
  }
  if (!jsonChunk || !binChunk) return null

  let json: any
  try {
    json = JSON.parse(jsonChunk.toString('utf8').trimEnd())
  } catch {
    return null
  }

  let changed = false
  for (const image of gltfImages(json, 'GLB')) {
    validateGltfImageUri(image)
    if (!image || !Number.isInteger(image.bufferView)) continue
    if (typeof image.mimeType !== 'string') {
      throw new Error('GLB bufferView image is missing mimeType. Embedded GLB images must declare PNG, JPEG, or WebP mimeType values.')
    }
    validateSupportedEmbeddedImageType('GLB bufferView image', image.mimeType)
    const imageBytes = glbBufferViewBytes(json, binChunk, image.bufferView)
    if (!imageBytes) continue
    image.uri = `data:${image.mimeType};base64,${imageBytes.toString('base64')}`
    delete image.bufferView
    changed = true
  }

  return changed ? encodeGlb(json, binChunk) : null
}

function validateGltfTextImageReferences(data: Buffer | string): void {
  const text = typeof data === 'string' ? data : data.toString('utf8')
  if (!text.trimStart().startsWith('{')) return

  let json: any
  try {
    json = JSON.parse(text)
  } catch {
    return
  }

  for (const image of gltfImages(json, 'glTF')) {
    validateGltfImageUri(image)
    if (image && Number.isInteger(image.bufferView)) {
      if (typeof image.mimeType !== 'string') {
        throw new Error('glTF bufferView image is missing mimeType. Embedded glTF images must declare PNG, JPEG, or WebP mimeType values.')
      }
      validateSupportedEmbeddedImageType('glTF bufferView image', image.mimeType)
    }
  }
}

function gltfImages(json: any, label: string): any[] {
  const images = json.images
  if (images == null) return []
  if (!Array.isArray(images)) {
    throw new Error(`${label}.images must be an array.`)
  }
  for (let i = 0; i < images.length; i += 1) {
    const image = images[i]
    if (!image || typeof image !== 'object' || Array.isArray(image)) {
      throw new Error(`${label}.images[${i}] must be an object.`)
    }
  }
  return images
}

function validateGltfImageUri(image: any): void {
  const uri = typeof image?.uri === 'string' ? image.uri : null
  if (!uri) return
  const mimeType = imageMimeTypeFromUri(uri)
  if (mimeType) {
    validateSupportedEmbeddedImageType('glTF image URI', mimeType)
  }
}

function imageMimeTypeFromUri(uri: string): string | null {
  const dataUriMatch = uri.match(/^data:([^;,]+)/i)
  if (dataUriMatch) return dataUriMatch[1].toLowerCase()
  if (/\.ktx2(?:$|[?#])/i.test(uri)) return 'image/ktx2'
  if (/\.basis(?:$|[?#])/i.test(uri)) return 'image/basis'
  return null
}

function validateSupportedEmbeddedImageType(label: string, mimeType: string): void {
  if (!/^image\/(?:png|jpe?g|webp)$/i.test(mimeType)) {
    throw unsupportedEmbeddedImageTypeError(label, mimeType)
  }
}

function unsupportedEmbeddedImageTypeError(label: string, mimeType: string): Error {
  if (/(?:ktx2|basis|compressed)/i.test(mimeType)) {
    return new Error(`${label} uses compressed texture MIME type "${mimeType}". KTX2, Basis, and compressed texture inputs are not decoded by @headless-three/renderer yet; pre-decode the texture to RGBA data or an encoded PNG/JPEG/WebP image before loading.`)
  }
  return new Error(`${label} uses unsupported MIME type "${mimeType}". Embedded GLB images must be PNG, JPEG, or WebP encoded images.`)
}

function glbBufferViewBytes(json: any, binChunk: Buffer, index: number): Buffer | null {
  const bufferView = json.bufferViews?.[index]
  if (!bufferView || bufferView.buffer !== 0 || !Number.isFinite(bufferView.byteLength)) {
    return null
  }
  const byteOffset = bufferView.byteOffset == null ? 0 : bufferView.byteOffset
  if (!Number.isFinite(byteOffset)) return null
  const start = Math.trunc(byteOffset)
  const end = start + Math.trunc(bufferView.byteLength)
  if (start < 0 || end < start || end > binChunk.byteLength) return null
  return binChunk.subarray(start, end)
}

function encodeGlb(json: any, binChunk: Buffer): Buffer {
  const jsonBytes = paddedChunk(Buffer.from(JSON.stringify(json), 'utf8'), 0x20)
  const binBytes = paddedChunk(binChunk, 0x00)
  const totalLength = 12 + 8 + jsonBytes.length + 8 + binBytes.length
  const glb = Buffer.alloc(totalLength)
  let offset = 0
  offset = writeGlbUint32(glb, offset, 0x46546c67)
  offset = writeGlbUint32(glb, offset, 2)
  offset = writeGlbUint32(glb, offset, totalLength)
  offset = writeGlbUint32(glb, offset, jsonBytes.length)
  offset = writeGlbUint32(glb, offset, 0x4e4f534a)
  jsonBytes.copy(glb, offset)
  offset += jsonBytes.length
  offset = writeGlbUint32(glb, offset, binBytes.length)
  offset = writeGlbUint32(glb, offset, 0x004e4942)
  binBytes.copy(glb, offset)
  return glb
}

function paddedChunk(buffer: Buffer, fill: number): Buffer {
  const padded = Buffer.alloc((buffer.length + 3) & ~3, fill)
  buffer.copy(padded)
  return padded
}

function writeGlbUint32(buffer: Buffer, offset: number, value: number): number {
  buffer.writeUInt32LE(value, offset)
  return offset + 4
}

function optionalBoolean(value: unknown, label: string): boolean | undefined {
  if (value == null) return undefined
  if (typeof value === 'boolean') return value
  throw new TypeError(`${label} must be a boolean.`)
}

function optionalNonNegativeNumber(value: unknown, label: string): number | undefined {
  if (value == null) return undefined
  if (typeof value === 'number' && Number.isFinite(value) && value >= 0) return value
  throw new TypeError(`${label} must be a finite non-negative number.`)
}

function optionalNonNegativeInteger(value: unknown, label: string): number | undefined {
  if (value == null) return undefined
  if (typeof value === 'number' && Number.isInteger(value) && value >= 0) return value
  throw new TypeError(`${label} must be a non-negative integer.`)
}

function validateNodeGltfLoaderOptions(options: NodeGltfLoaderOptions): {
  configureLoader?: ConfigureGltfLoader
  installFetch?: boolean
  manager?: ThreeLoadingManagerLike
  registerTextureHandlers?: boolean
} {
  return {
    configureLoader: optionalFunction(options.configureLoader, 'options.configureLoader'),
    installFetch: optionalBoolean(options.installFetch, 'options.installFetch'),
    manager: optionalLoadingManager(options.manager, 'options.manager'),
    registerTextureHandlers: optionalBoolean(options.registerTextureHandlers, 'options.registerTextureHandlers'),
  }
}

function validateLoadGltfFromFileOptions(options: LoadGltfFromFileOptions): void {
  validateNodeGltfLoaderOptions(options)
  optionalString(options.rootDir, 'options.rootDir')
  optionalString(options.baseUrl, 'options.baseUrl')
}

function requiredString(value: unknown, label: string): string {
  if (typeof value === 'string') return value
  throw new TypeError(`${label} must be a string.`)
}

function optionalString(value: unknown, label: string): string | undefined {
  if (value == null) return undefined
  return requiredString(value, label)
}

function objectOptions(value: unknown, label: string): Record<string, unknown> {
  if (value != null && typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>
  }
  throw new TypeError(`${label} must be an object.`)
}

function optionalFunction<T>(value: T | null | undefined, label: string): T | undefined {
  if (value == null) return undefined
  if (typeof value === 'function') return value
  throw new TypeError(`${label} must be a function.`)
}

function optionalLoadingManager(value: ThreeLoadingManagerLike | null | undefined, label: string): ThreeLoadingManagerLike | undefined {
  if (value == null) return undefined
  if (typeof (value as any).addHandler !== 'function') {
    throw new TypeError(`${label} must provide an addHandler() function.`)
  }
  for (const method of ['itemStart', 'itemEnd', 'itemError', 'resolveURL']) {
    if ((value as any)[method] != null && typeof (value as any)[method] !== 'function') {
      throw new TypeError(`${label}.${method} must be a function when provided.`)
    }
  }
  return value
}

function registerLoaderPlugin(
  loader: ThreeGltfLoaderLike,
  callback: (parser: unknown) => unknown,
  label: string,
): void {
  if (typeof loader.register !== 'function') {
    throw new Error(`GLTFLoader.register is required to install ${label}.`)
  }
  loader.register(callback)
}

const dynamicImport = new Function('specifier', 'return import(specifier)') as <T = unknown>(
  specifier: string,
) => Promise<T>

function importGltfLoader(): Promise<GltfLoaderModule> {
  return dynamicImport<GltfLoaderModule>('three/examples/jsm/loaders/GLTFLoader.js')
}

async function importVrmLoaderPlugin(): Promise<VrmLoaderPluginConstructor> {
  let module: any
  try {
    module = await dynamicImport('@pixiv/three-vrm')
  } catch {
    throw new Error('Missing optional dependency @pixiv/three-vrm. Install it in your project or pass VRMLoaderPlugin to loadVrmFromFile().')
  }

  if (typeof module.VRMLoaderPlugin !== 'function') {
    throw new Error('@pixiv/three-vrm did not export VRMLoaderPlugin.')
  }
  return module.VRMLoaderPlugin as VrmLoaderPluginConstructor
}

async function importVrmAnimationLoaderPlugin(): Promise<VrmLoaderPluginConstructor> {
  let module: any
  try {
    module = await dynamicImport('@pixiv/three-vrm-animation')
  } catch {
    throw new Error('Missing optional dependency @pixiv/three-vrm-animation. Install it in your project or pass VRMAnimationLoaderPlugin to loadVrmAnimationFromFile().')
  }

  if (typeof module.VRMAnimationLoaderPlugin !== 'function') {
    throw new Error('@pixiv/three-vrm-animation did not export VRMAnimationLoaderPlugin.')
  }
  return module.VRMAnimationLoaderPlugin as VrmLoaderPluginConstructor
}

async function importVrmAnimationClipFactory(): Promise<VrmAnimationClipFactory> {
  let module: any
  try {
    module = await dynamicImport('@pixiv/three-vrm-animation')
  } catch {
    throw new Error('Missing optional dependency @pixiv/three-vrm-animation. Install it in your project or pass createVRMAnimationClip to applyVrmAnimation().')
  }

  if (typeof module.createVRMAnimationClip !== 'function') {
    throw new Error('@pixiv/three-vrm-animation did not export createVRMAnimationClip.')
  }
  return module.createVRMAnimationClip as VrmAnimationClipFactory
}
