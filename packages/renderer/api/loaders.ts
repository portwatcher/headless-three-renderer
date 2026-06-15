import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

// eslint-disable-next-line @typescript-eslint/no-var-requires
const { LoadingManager, Texture } = require('three') as {
  LoadingManager: new () => ThreeLoadingManagerLike
  Texture: new () => TextureLike
}

type TextureLike = {
  image?: unknown
  source: { data?: unknown }
  needsUpdate?: boolean
  isTexture?: boolean
}
export type ThreeLoadingManagerLike = {
  addHandler(regex: RegExp, loader: unknown): unknown
}
export type ThreeGltfLoaderLike = {
  parse(data: ArrayBuffer, path: string, onLoad: (gltf: unknown) => void, onError?: (error: unknown) => void): void
  register?(callback: unknown): unknown
}
type GltfLoaderCtor = new (manager?: ThreeLoadingManagerLike) => ThreeGltfLoaderLike
type GltfLoaderModule = {
  GLTFLoader: GltfLoaderCtor
}
export type VrmLoaderPluginConstructor = new (parser: unknown) => unknown
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
  private loaderPath = ''

  constructor(private readonly rootDir: string = process.cwd()) {}

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
    this.loaderPath = loaderPath
    return this
  }

  load(
    url: string,
    onLoad?: TextureLoadCallback,
    _onProgress?: unknown,
    onError?: TextureErrorCallback,
  ): TextureLike {
    const texture = new Texture()
    const source = /^data:/i.test(url) ? url : (this.loaderPath ? `${this.loaderPath}${url}` : url)
    const encodedDataUri = encodedImageDataUriBuffer(source)
    const data = encodedDataUri
      ? Promise.resolve(encodedDataUri)
      : /^blob:/i.test(source)
        ? readBlobUrlBuffer(source)
        : readFile(resolveLocalAssetPath(source, this.rootDir))

    data.then((buffer) => {
      texture.image = buffer
      texture.source.data = buffer
      texture.needsUpdate = true
      onLoad?.(texture)
    }, onError)

    return texture
  }
}

export function createEncodedImageTextureLoader(rootDir?: string): EncodedImageTextureLoader {
  return new EncodedImageTextureLoader(rootDir)
}

export async function createNodeGltfLoader(
  rootDir: string = process.cwd(),
  options: NodeGltfLoaderOptions = {},
): Promise<NodeGltfLoaderBundle> {
  const root = path.resolve(rootDir)
  const installFetch = optionalBoolean(options.installFetch, 'options.installFetch')
  const registerTextureHandlers = optionalBoolean(options.registerTextureHandlers, 'options.registerTextureHandlers')

  if (installFetch !== false) {
    installLocalFileFetch()
  }

  const manager = options.manager ?? new LoadingManager()
  const encodedImages = createEncodedImageTextureLoader(root)
  if (registerTextureHandlers !== false) {
    registerEncodedImageHandlers(manager, encodedImages)
  }

  const { GLTFLoader } = await importGltfLoader()
  const loader = new GLTFLoader(manager)
  await options.configureLoader?.(loader)
  return { encodedImages, loader, manager, rootDir: root }
}

export async function loadGltfFromFile<T = unknown>(
  filePath: string,
  options: LoadGltfFromFileOptions = {},
): Promise<T> {
  const absolute = path.resolve(filePath)
  const root = path.resolve(options.rootDir ?? path.dirname(absolute))
  const { loader } = await createNodeGltfLoader(root, options)
  const bytes = await readFile(absolute)
  const baseUrl = options.baseUrl ?? pathToFileURL(`${root}${path.sep}`).href

  return await new Promise<T>((resolve, reject) => {
    loader.parse(arrayBufferView(bytes), baseUrl, (gltf) => resolve(gltf as T), reject)
  })
}

export async function loadVrmFromFile<T = unknown>(
  filePath: string,
  options: LoadVrmFromFileOptions = {},
): Promise<T> {
  const {
    VRMLoaderPlugin,
    configureLoader,
    ...gltfOptions
  } = options
  const VrmPlugin = VRMLoaderPlugin ?? await importVrmLoaderPlugin()

  return await loadGltfFromFile<T>(filePath, {
    ...gltfOptions,
    configureLoader: async (loader) => {
      registerLoaderPlugin(loader, (parser) => new VrmPlugin(parser), 'VRMLoaderPlugin')
      await configureLoader?.(loader)
    },
  })
}

export async function loadVrmAnimationFromFile<T = unknown>(
  filePath: string,
  options: LoadVrmAnimationFromFileOptions = {},
): Promise<T> {
  const {
    VRMAnimationLoaderPlugin,
    VRMLoaderPlugin,
    configureLoader,
    ...gltfOptions
  } = options
  const AnimationPlugin = VRMAnimationLoaderPlugin ?? await importVrmAnimationLoaderPlugin()

  return await loadGltfFromFile<T>(filePath, {
    ...gltfOptions,
    configureLoader: async (loader) => {
      if (VRMLoaderPlugin) {
        registerLoaderPlugin(loader, (parser) => new VRMLoaderPlugin(parser), 'VRMLoaderPlugin')
      }
      registerLoaderPlugin(loader, (parser) => new AnimationPlugin(parser), 'VRMAnimationLoaderPlugin')
      await configureLoader?.(loader)
    },
  })
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

export function resolveLocalAssetPath(url: string, rootDir: string = process.cwd()): string {
  if (/^data:/i.test(url)) {
    throw new Error('Data URI textures should be decoded or written to files before loading in Node.')
  }
  if (/^file:/i.test(url)) return fileURLToPath(url)
  if (/^[a-z][a-z0-9+.-]*:/i.test(url)) {
    throw new Error(`Remote texture URL is not a local file: ${url}`)
  }
  return path.isAbsolute(url) ? path.normalize(url) : path.resolve(rootDir, url)
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

function registerEncodedImageHandlers(manager: ThreeLoadingManagerLike, encodedImages: EncodedImageTextureLoader): void {
  manager.addHandler(/^blob:/i, encodedImages)
  manager.addHandler(/^data:image\/(?:png|jpe?g|webp)/i, encodedImages)
  manager.addHandler(/\.(png|jpe?g|webp)$/i, encodedImages)
}

function arrayBufferView(buffer: Buffer): ArrayBuffer {
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength) as ArrayBuffer
}

function optionalBoolean(value: unknown, label: string): boolean | undefined {
  if (value == null) return undefined
  if (typeof value === 'boolean') return value
  throw new TypeError(`${label} must be a boolean.`)
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
