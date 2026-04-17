import type { Color4, ThreeMaterialLike, PbrProperties, TextureInfo } from './types'
import { clamp01 } from './math'
import { colorLikeToArray } from './color'

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

  return props
}

export function extractTextureData(material: ThreeMaterialLike | undefined): TextureInfo | null {
  return extractTextureFromSlot(material?.map)
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
