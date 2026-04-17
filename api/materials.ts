import type { Color4, ThreeMaterialLike, PbrProperties, TextureInfo, ThreeTextureLike, ThreeSceneLike } from './types'
import { clamp01 } from './math'
import { colorLikeToArray } from './color'

// Three.js wrapping constants
const RepeatWrapping = 1000
const MirroredRepeatWrapping = 1002

// Three.js side constants
const FrontSide = 0
const BackSide = 1
const DoubleSide = 2

// Three.js texture type constants
const UnsignedByteType = 1009
const HalfFloatType = 1016
const FloatType = 1015

export interface EnvironmentMapInfo {
  data: Buffer
  width: number
  height: number
  intensity: number
}

/**
 * Extract environment map data from scene.environment.
 * Supports DataTexture (equirectangular) with Uint8, Float16, Float32 pixel data.
 * Passes raw typed-array bytes to Rust which handles format detection.
 */
export function extractEnvironmentMap(scene: ThreeSceneLike): EnvironmentMapInfo | null {
  const envTex = scene.environment
  if (!envTex) return null

  const image = (envTex as any).image ?? (envTex as any).source?.data
  if (!image) return null

  const intensity = (scene as any).environmentIntensity ?? 1.0

  // DataTexture: { data, width, height }
  if (image.data && image.width > 0 && image.height > 0) {
    const texType = (envTex as any).type ?? UnsignedByteType
    const rawData = image.data as ArrayBufferView & { buffer: ArrayBuffer; byteOffset: number; byteLength: number }

    if (texType === HalfFloatType && rawData instanceof Uint16Array) {
      // Pass raw half-float bytes — Rust ibl.rs decodes them
      const channels = rawData.length / (image.width * image.height)
      let buf: Buffer
      if (channels === 3) {
        // Expand RGB16F → RGBA16F (half=0x3C00 is 1.0 for alpha)
        const pixels = image.width * image.height
        const out = new Uint16Array(pixels * 4)
        for (let i = 0; i < pixels; i++) {
          out[i * 4] = rawData[i * 3]
          out[i * 4 + 1] = rawData[i * 3 + 1]
          out[i * 4 + 2] = rawData[i * 3 + 2]
          out[i * 4 + 3] = 0x3C00 // 1.0 in half-float
        }
        buf = Buffer.from(out.buffer, out.byteOffset, out.byteLength)
      } else {
        buf = Buffer.from(rawData.buffer, rawData.byteOffset, rawData.byteLength)
      }
      return { data: buf, width: image.width, height: image.height, intensity }
    }

    if (texType === FloatType && rawData instanceof Float32Array) {
      const channels = rawData.length / (image.width * image.height)
      let buf: Buffer
      if (channels === 3) {
        const pixels = image.width * image.height
        const out = new Float32Array(pixels * 4)
        for (let i = 0; i < pixels; i++) {
          out[i * 4] = rawData[i * 3]
          out[i * 4 + 1] = rawData[i * 3 + 1]
          out[i * 4 + 2] = rawData[i * 3 + 2]
          out[i * 4 + 3] = 1.0
        }
        buf = Buffer.from(out.buffer, out.byteOffset, out.byteLength)
      } else {
        buf = Buffer.from(rawData.buffer, rawData.byteOffset, rawData.byteLength)
      }
      return { data: buf, width: image.width, height: image.height, intensity }
    }

    // UnsignedByteType / default: convert to RGBA8
    const rgba = toRgba8(rawData as any, image.width, image.height)
    if (rgba) {
      return { data: Buffer.from(rgba.buffer, rgba.byteOffset, rgba.byteLength), width: image.width, height: image.height, intensity }
    }
  }

  // Encoded image buffer (e.g. loaded HDR encoded as PNG/EXR)
  if (Buffer.isBuffer(image)) {
    return { data: image, width: 0, height: 0, intensity }
  }
  if (image instanceof Uint8Array && !((image as any).width > 0)) {
    return { data: Buffer.from(image.buffer, image.byteOffset, image.byteLength), width: 0, height: 0, intensity }
  }

  return null
}

export function materialForGroup(
  material: ThreeMaterialLike | ThreeMaterialLike[] | undefined,
  materialIndex: number,
): ThreeMaterialLike | undefined {
  if (Array.isArray(material)) {
    return material[materialIndex] ?? material[0]
  }
  return material
}

export function materialColor(material: ThreeMaterialLike | undefined): Color4 {
  const color = colorLikeToArray(material?.color) ?? [1, 1, 1, 1] as Color4
  color[3] = clamp01(material?.opacity ?? color[3] ?? 1)
  return color
}

export function extractPbrProperties(material: ThreeMaterialLike | undefined): PbrProperties {
  if (!material) return {}
  const props: PbrProperties = {}

  if (Number.isFinite(material.metalness)) {
    props.metallic = clamp01(material.metalness!)
  }
  if (Number.isFinite(material.roughness)) {
    props.roughness = clamp01(material.roughness!)
  }

  const emissive = colorLikeToArray(material.emissive)
  if (emissive) {
    props.emissive = [emissive[0], emissive[1], emissive[2]]
    props.emissiveIntensity = Number.isFinite(material.emissiveIntensity) ? material.emissiveIntensity! : 1
  }

  const normalMapInfo = extractTextureFromSlot(material.normalMap)
  if (normalMapInfo) {
    props.normalMap = normalMapInfo.data
    props.normalMapWidth = normalMapInfo.width
    props.normalMapHeight = normalMapInfo.height
  }
  if (material.normalScale) {
    props.normalScale = [material.normalScale.x ?? 1, material.normalScale.y ?? 1]
  }

  const mrMapInfo = extractTextureFromSlot(material.metalnessMap ?? material.roughnessMap)
  if (mrMapInfo) {
    props.metallicRoughnessTexture = mrMapInfo.data
    props.metallicRoughnessTextureWidth = mrMapInfo.width
    props.metallicRoughnessTextureHeight = mrMapInfo.height
  }

  const emissiveMapInfo = extractTextureFromSlot(material.emissiveMap)
  if (emissiveMapInfo) {
    props.emissiveMap = emissiveMapInfo.data
    props.emissiveMapWidth = emissiveMapInfo.width
    props.emissiveMapHeight = emissiveMapInfo.height
  }

  const aoMapInfo = extractTextureFromSlot(material.aoMap)
  if (aoMapInfo) {
    props.aoMap = aoMapInfo.data
    props.aoMapWidth = aoMapInfo.width
    props.aoMapHeight = aoMapInfo.height
    props.aoMapIntensity = Number.isFinite(material.aoMapIntensity) ? material.aoMapIntensity! : 1
  }

  if (Number.isFinite(material.alphaTest) && material.alphaTest! > 0) {
    props.alphaTest = clamp01(material.alphaTest!)
  }
  if (typeof material.transparent === 'boolean') {
    props.transparent = material.transparent
  }
  if (material.side === BackSide) {
    props.side = 'back'
  } else if (material.side === DoubleSide) {
    props.side = 'double'
  } else if (material.side === FrontSide) {
    props.side = 'front'
  }

  // Shading model: MeshBasicMaterial is unlit, MeshLambertMaterial is diffuse-only.
  // Everything else (MeshStandardMaterial / MeshPhysicalMaterial / unknown) uses the
  // default PBR path.
  if (material.isMeshBasicMaterial) {
    props.shadingModel = 'basic'
  } else if (material.isMeshLambertMaterial) {
    props.shadingModel = 'lambert'
  }

  return props
}

export function extractTextureData(material: ThreeMaterialLike | undefined): TextureInfo | null {
  const base = extractTextureFromSlot(material?.map)
  if (!base) return null

  const map = material!.map as ThreeTextureLike | null | undefined
  return {
    ...base,
    wrapS: wrapModeToString(map?.wrapS),
    wrapT: wrapModeToString(map?.wrapT),
  }
}

function wrapModeToString(mode: number | undefined): string | undefined {
  if (mode === RepeatWrapping) return 'repeat'
  if (mode === MirroredRepeatWrapping) return 'mirror'
  return undefined // default = clamp
}

function extractTextureFromSlot(map: ThreeMaterialLike['map']): TextureInfo | null {
  if (!map) return null

  const image = (map as any).image ?? (map as any).source?.data
  if (!image) return null

  // DataTexture style: { data: TypedArray, width, height }
  if (image.data && image.width > 0 && image.height > 0) {
    const rgba = toRgba8(image.data, image.width, image.height)
    if (rgba) {
      return { data: Buffer.from(rgba.buffer, rgba.byteOffset, rgba.byteLength), width: image.width, height: image.height }
    }
  }

  // Encoded image (PNG/JPEG/WebP Buffer from file loaders)
  if (Buffer.isBuffer(image)) {
    return { data: image, width: 0, height: 0 }
  }
  if (image instanceof Uint8Array && !((image as any).width > 0)) {
    return { data: Buffer.from(image.buffer, image.byteOffset, image.byteLength), width: 0, height: 0 }
  }

  // ImageData (canvas-based polyfill): { data: Uint8ClampedArray, width, height }
  if (image.data instanceof Uint8ClampedArray && image.width > 0 && image.height > 0) {
    return {
      data: Buffer.from(image.data.buffer, image.data.byteOffset, image.data.byteLength),
      width: image.width,
      height: image.height,
    }
  }

  return null
}

function toRgba8(data: ArrayBufferView & { length: number }, width: number, height: number): Uint8Array | null {
  const pixels = width * height

  if (data instanceof Uint8Array || data instanceof Uint8ClampedArray) {
    if (data.length === pixels * 4) return new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
    if (data.length === pixels * 3) {
      const out = new Uint8Array(pixels * 4)
      for (let i = 0; i < pixels; i++) {
        out[i * 4] = data[i * 3]
        out[i * 4 + 1] = data[i * 3 + 1]
        out[i * 4 + 2] = data[i * 3 + 2]
        out[i * 4 + 3] = 255
      }
      return out
    }
    return null
  }

  if (data instanceof Float32Array || data instanceof Float64Array) {
    if (data.length === pixels * 4) {
      const out = new Uint8Array(pixels * 4)
      for (let i = 0; i < pixels * 4; i++) {
        out[i] = Math.max(0, Math.min(255, Math.round(data[i] * 255)))
      }
      return out
    }
    if (data.length === pixels * 3) {
      const out = new Uint8Array(pixels * 4)
      for (let i = 0; i < pixels; i++) {
        out[i * 4] = Math.max(0, Math.min(255, Math.round(data[i * 3] * 255)))
        out[i * 4 + 1] = Math.max(0, Math.min(255, Math.round(data[i * 3 + 1] * 255)))
        out[i * 4 + 2] = Math.max(0, Math.min(255, Math.round(data[i * 3 + 2] * 255)))
        out[i * 4 + 3] = 255
      }
      return out
    }
    return null
  }

  // Uint16Array or other numeric typed arrays — treat as 8-bit range after clamping
  if (ArrayBuffer.isView(data) && data.length === pixels * 4) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels * 4; i++) {
      out[i] = Math.max(0, Math.min(255, (data as any)[i]))
    }
    return out
  }

  return null
}
