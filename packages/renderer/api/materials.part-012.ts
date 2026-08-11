import type { Color4, ThreeMaterialLike, PbrProperties, TextureInfo, ThreeTextureLike, ThreeSceneRootLike, ThreeObject3DLike } from './types'
import { clamp01 } from './math'
import { strictColorLikeToArray, validatedColorLikeToArray } from './color'
import { objectChildren } from './objects'
import { AlphaFormat, LuminanceAlphaFormat } from './materials.part-001'
import { normalizedUnsignedIntegerToByte } from './materials.part-011'
export function normalizedSignedIntegerDataToRgba8(
  data: Int8Array | Int16Array | Int32Array,
  pixels: number,
  allowNarrowChannels: boolean,
  maxValue: number,
  format: unknown,
): Uint8Array | null {
  if (data.length === pixels * 4) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels * 4; i++) {
      out[i] = normalizedSignedIntegerToByte(data[i], maxValue)
    }
    return out
  }
  if (data.length === pixels * 3) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels; i++) {
      out[i * 4] = normalizedSignedIntegerToByte(data[i * 3], maxValue)
      out[i * 4 + 1] = normalizedSignedIntegerToByte(data[i * 3 + 1], maxValue)
      out[i * 4 + 2] = normalizedSignedIntegerToByte(data[i * 3 + 2], maxValue)
      out[i * 4 + 3] = 255
    }
    return out
  }
  if (allowNarrowChannels && data.length === pixels * 2) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels; i++) {
      writeTwoChannelRgba8(
        out,
        i,
        normalizedSignedIntegerToByte(data[i * 2], maxValue),
        normalizedSignedIntegerToByte(data[i * 2 + 1], maxValue),
        format,
      )
    }
    return out
  }
  if (allowNarrowChannels && data.length === pixels) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels; i++) {
      const value = normalizedSignedIntegerToByte(data[i], maxValue)
      writeOneChannelRgba8(out, i, value, format)
    }
    return out
  }
  return null
}

export function normalizedSignedIntegerToByte(value: number, maxValue: number): number {
  return Math.max(0, Math.min(255, Math.round((Math.max(value, 0) / maxValue) * 255)))
}

export function packedUnsignedShort4444ToRgba8(data: Uint16Array, pixels: number): Uint8Array | null {
  if (data.length !== pixels) return null
  const out = new Uint8Array(pixels * 4)
  for (let i = 0; i < pixels; i++) {
    const value = data[i]
    out[i * 4] = ((value >> 12) & 0xf) * 17
    out[i * 4 + 1] = ((value >> 8) & 0xf) * 17
    out[i * 4 + 2] = ((value >> 4) & 0xf) * 17
    out[i * 4 + 3] = (value & 0xf) * 17
  }
  return out
}

export function packedUnsignedShort5551ToRgba8(data: Uint16Array, pixels: number): Uint8Array | null {
  if (data.length !== pixels) return null
  const out = new Uint8Array(pixels * 4)
  for (let i = 0; i < pixels; i++) {
    const value = data[i]
    out[i * 4] = normalizedUnsignedIntegerToByte((value >> 11) & 0x1f, 0x1f)
    out[i * 4 + 1] = normalizedUnsignedIntegerToByte((value >> 6) & 0x1f, 0x1f)
    out[i * 4 + 2] = normalizedUnsignedIntegerToByte((value >> 1) & 0x1f, 0x1f)
    out[i * 4 + 3] = (value & 0x1) === 1 ? 255 : 0
  }
  return out
}

export function packedUnsignedInt5999ToRgba8(data: Uint32Array, pixels: number): Uint8Array | null {
  if (data.length !== pixels) return null
  const out = new Uint8Array(pixels * 4)
  for (let i = 0; i < pixels; i++) {
    const value = data[i]
    const scale = 2 ** (((value >>> 27) & 0x1f) - 24)
    out[i * 4] = normalizedPackedRgb9E5ToByte(value & 0x1ff, scale)
    out[i * 4 + 1] = normalizedPackedRgb9E5ToByte((value >>> 9) & 0x1ff, scale)
    out[i * 4 + 2] = normalizedPackedRgb9E5ToByte((value >>> 18) & 0x1ff, scale)
    out[i * 4 + 3] = 255
  }
  return out
}

export function normalizedPackedRgb9E5ToByte(mantissa: number, scale: number): number {
  return Math.max(0, Math.min(255, Math.round(mantissa * scale * 255)))
}

export function packedUnsignedInt101111ToRgba8(data: Uint32Array, pixels: number): Uint8Array | null {
  if (data.length !== pixels) return null
  const out = new Uint8Array(pixels * 4)
  for (let i = 0; i < pixels; i++) {
    const value = data[i]
    out[i * 4] = unsignedPackedFloatToByte(value & 0x7ff, 6)
    out[i * 4 + 1] = unsignedPackedFloatToByte((value >>> 11) & 0x7ff, 6)
    out[i * 4 + 2] = unsignedPackedFloatToByte((value >>> 22) & 0x3ff, 5)
    out[i * 4 + 3] = 255
  }
  return out
}

export function unsignedPackedFloatToByte(bits: number, mantissaBits: 5 | 6): number {
  const exponent = bits >>> mantissaBits
  const mantissa = bits & ((1 << mantissaBits) - 1)
  let value: number
  if (exponent === 0) {
    value = (mantissa / (2 ** mantissaBits)) * (2 ** -14)
  } else if (exponent === 0x1f) {
    value = mantissa === 0 ? Infinity : Number.NaN
  } else {
    value = (1 + mantissa / (2 ** mantissaBits)) * (2 ** (exponent - 15))
  }
  if (!Number.isFinite(value)) return value > 0 ? 255 : 0
  return Math.max(0, Math.min(255, Math.round(value * 255)))
}

export function halfFloatDataToRgba8(
  data: Uint16Array,
  pixels: number,
  allowNarrowChannels: boolean,
  format: unknown,
): Uint8Array | null {
  if (data.length === pixels * 4) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels * 4; i++) {
      out[i] = halfFloatToByte(data[i])
    }
    return out
  }
  if (data.length === pixels * 3) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels; i++) {
      out[i * 4] = halfFloatToByte(data[i * 3])
      out[i * 4 + 1] = halfFloatToByte(data[i * 3 + 1])
      out[i * 4 + 2] = halfFloatToByte(data[i * 3 + 2])
      out[i * 4 + 3] = 255
    }
    return out
  }
  if (allowNarrowChannels && data.length === pixels * 2) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels; i++) {
      writeTwoChannelRgba8(out, i, halfFloatToByte(data[i * 2]), halfFloatToByte(data[i * 2 + 1]), format)
    }
    return out
  }
  if (allowNarrowChannels && data.length === pixels) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels; i++) {
      const value = halfFloatToByte(data[i])
      writeOneChannelRgba8(out, i, value, format)
    }
    return out
  }
  return null
}

export function writeOneChannelRgba8(
  out: Uint8Array,
  pixelIndex: number,
  value: number,
  format: unknown,
): void {
  const offset = pixelIndex * 4
  out[offset] = value
  out[offset + 1] = value
  out[offset + 2] = value
  out[offset + 3] = format === AlphaFormat ? value : 255
}

export function writeTwoChannelRgba8(
  out: Uint8Array,
  pixelIndex: number,
  first: number,
  second: number,
  format: unknown,
): void {
  const offset = pixelIndex * 4
  if (format === LuminanceAlphaFormat) {
    out[offset] = first
    out[offset + 1] = first
    out[offset + 2] = first
    out[offset + 3] = second
    return
  }
  out[offset] = first
  out[offset + 1] = second
  out[offset + 2] = 0
  out[offset + 3] = 255
}

export function halfFloatToByte(bits: number): number {
  const value = halfFloatToNumber(bits)
  if (!Number.isFinite(value)) return value > 0 ? 255 : 0
  return Math.max(0, Math.min(255, Math.round(value * 255)))
}

export function halfFloatToNumber(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1
  const exponent = (bits >> 10) & 0x1f
  const mantissa = bits & 0x03ff
  if (exponent === 0) {
    return sign * (mantissa / 0x400) * (2 ** -14)
  }
  if (exponent === 0x1f) {
    return mantissa === 0 ? sign * Infinity : Number.NaN
  }
  return sign * (1 + mantissa / 0x400) * (2 ** (exponent - 15))
}

export function numberToHalfFloat(value: number): number {
  if (Number.isNaN(value)) return 0x7e00
  const sign = value < 0 || Object.is(value, -0) ? 0x8000 : 0
  const abs = Math.abs(value)
  if (abs === 0) return sign
  if (!Number.isFinite(abs)) return sign | 0x7c00
  if (abs >= 65504) return sign | 0x7bff
  if (abs < 2 ** -14) {
    return sign | Math.round(abs / (2 ** -24))
  }

  const exponent = Math.floor(Math.log2(abs))
  let mantissa = Math.round((abs / (2 ** exponent) - 1) * 0x400)
  let biasedExponent = exponent + 15
  if (mantissa === 0x400) {
    mantissa = 0
    biasedExponent += 1
  }
  if (biasedExponent >= 31) return sign | 0x7bff
  return sign | (biasedExponent << 10) | mantissa
}
