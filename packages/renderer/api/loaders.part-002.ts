import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { AnimationClipLike, ColorKeyframeTrack, EncodedImageTextureLoader, GltfAnimationPointerParser, GltfNodeVisibilityParser, GltfRootLike, InterpolateDiscrete, InterpolateLinear, MeshLike, ThreeGltfLoaderLike, ThreeLoadingManagerLike, TraversableObjectLike } from './loaders.part-001'
import { registerLoaderPlugin, requiredString } from './loaders.part-003'
export function resolveLoadingManagerUrl(manager: ThreeLoadingManagerLike | undefined, url: string): string {
  const resolveURL = manager?.resolveURL
  if (resolveURL == null) return url
  const resolved = resolveURL.call(manager, url)
  return requiredString(resolved, 'manager.resolveURL return value')
}

export function isAbsoluteOrSpecialAssetUrl(url: string): boolean {
  return /^data:/i.test(url)
    || /^blob:/i.test(url)
    || path.isAbsolute(url)
    || path.win32.isAbsolute(url)
    || /^file:/i.test(url)
    || /^[a-z][a-z0-9+.-]*:/i.test(url)
}

export function ensureDirectoryUrl(url: string): string {
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

export function resolveLocalRootDir(rootDir: string, label: string): string {
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

export function installEncodedImageSupportProbe(): void {
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

export function registerEncodedImageHandlers(manager: ThreeLoadingManagerLike, encodedImages: EncodedImageTextureLoader): void {
  manager.addHandler(/^blob:/i, encodedImages)
  manager.addHandler(/^data:image\/(?:png|jpe?g|webp)/i, encodedImages)
  manager.addHandler(/\.(png|jpe?g|webp)$/i, encodedImages)
}

export function arrayBufferView(buffer: Buffer): ArrayBuffer {
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength) as ArrayBuffer
}

export function installEmbeddedGlbImageNormalizer(loader: ThreeGltfLoaderLike): void {
  const originalParse = loader.parse.bind(loader)
  loader.parse = (data, parsePath, onLoad, onError) => {
    originalParse(normalizeEmbeddedGlbImages(data), parsePath, onLoad, onError)
  }
}

export function installNodeVisibilityExtension(loader: ThreeGltfLoaderLike): void {
  registerLoaderPlugin(loader, (parser) => ({
    name: 'KHR_node_visibility',
    afterRoot(gltf: unknown) {
      applyNodeVisibilityExtension(gltf, parser as GltfNodeVisibilityParser)
    },
  }), 'KHR_node_visibility')
}

export function installAnimationPointerExtension(loader: ThreeGltfLoaderLike): void {
  registerLoaderPlugin(loader, (parser) => ({
    name: 'KHR_animation_pointer',
    afterRoot(gltf: unknown) {
      return applyAnimationPointerExtension(gltf, parser as GltfAnimationPointerParser)
    },
  }), 'KHR_animation_pointer')
}

export async function applyAnimationPointerExtension(gltf: unknown, parser: GltfAnimationPointerParser): Promise<void> {
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

export function materialAnimationTargets(gltf: unknown, parser: GltfAnimationPointerParser, materialIndex: number): string[] {
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

export function animationPointerColorValues(keyframeCount: number, values: ArrayLike<number>, itemSize: number): Float32Array {
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

export function applyNodeVisibilityExtension(gltf: unknown, parser: GltfNodeVisibilityParser): void {
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

export function uniqueNodeIndexByName(nodes: unknown[]): Map<string, number> {
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

export function gltfRootScenes(gltf: unknown): TraversableObjectLike[] {
  const root = gltf as GltfRootLike
  const scenes = Array.isArray(root?.scenes) ? root.scenes : []
  const result = scenes.filter(isTraversableObject)
  if (isTraversableObject(root?.scene) && !result.includes(root.scene)) {
    result.push(root.scene)
  }
  return result
}

export function isTraversableObject(value: unknown): value is TraversableObjectLike {
  return !!value && typeof value === 'object' && typeof (value as TraversableObjectLike).traverse === 'function'
}

export function normalizeEmbeddedGlbImages(data: ArrayBuffer | string): ArrayBuffer | string {
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

export function rewriteGlbBufferViewImages(bytes: Buffer): Buffer | null {
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

export function validateGltfTextImageReferences(data: Buffer | string): void {
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

export function gltfImages(json: any, label: string): any[] {
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

export function validateGltfImageUri(image: any): void {
  const uri = typeof image?.uri === 'string' ? image.uri : null
  if (!uri) return
  const mimeType = imageMimeTypeFromUri(uri)
  if (mimeType) {
    validateSupportedEmbeddedImageType('glTF image URI', mimeType)
  }
}

export function imageMimeTypeFromUri(uri: string): string | null {
  const dataUriMatch = uri.match(/^data:([^;,]+)/i)
  if (dataUriMatch) return dataUriMatch[1].toLowerCase()
  if (/\.ktx2(?:$|[?#])/i.test(uri)) return 'image/ktx2'
  if (/\.basis(?:$|[?#])/i.test(uri)) return 'image/basis'
  return null
}

export function validateSupportedEmbeddedImageType(label: string, mimeType: string): void {
  if (!/^image\/(?:png|jpe?g|webp)$/i.test(mimeType)) {
    throw unsupportedEmbeddedImageTypeError(label, mimeType)
  }
}

export function unsupportedEmbeddedImageTypeError(label: string, mimeType: string): Error {
  if (/(?:ktx2|basis|compressed)/i.test(mimeType)) {
    return new Error(`${label} uses compressed texture MIME type "${mimeType}". KTX2, Basis, and compressed texture inputs are not decoded by @headless-three/renderer yet; pre-decode the texture to RGBA data or an encoded PNG/JPEG/WebP image before loading.`)
  }
  return new Error(`${label} uses unsupported MIME type "${mimeType}". Embedded GLB images must be PNG, JPEG, or WebP encoded images.`)
}

export function glbBufferViewBytes(json: any, binChunk: Buffer, index: number): Buffer | null {
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

export function encodeGlb(json: any, binChunk: Buffer): Buffer {
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

export function paddedChunk(buffer: Buffer, fill: number): Buffer {
  const padded = Buffer.alloc((buffer.length + 3) & ~3, fill)
  buffer.copy(padded)
  return padded
}

export function writeGlbUint32(buffer: Buffer, offset: number, value: number): number {
  buffer.writeUInt32LE(value, offset)
  return offset + 4
}
