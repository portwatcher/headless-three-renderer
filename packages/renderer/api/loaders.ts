import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

// eslint-disable-next-line @typescript-eslint/no-var-requires
const { Texture } = require('three') as { Texture: new () => TextureLike }

type TextureLike = {
  image?: unknown
  source: { data?: unknown }
  needsUpdate?: boolean
  isTexture?: boolean
}
type TextureLoadCallback = (texture: TextureLike) => void
type TextureErrorCallback = (error: unknown) => void

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
    const resolved = resolveLocalAssetPath(this.loaderPath ? `${this.loaderPath}${url}` : url, this.rootDir)

    readFile(resolved).then((buffer) => {
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
