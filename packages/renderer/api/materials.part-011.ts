import type { Color4, ThreeMaterialLike, PbrProperties, TextureInfo, ThreeTextureLike, ThreeSceneRootLike, ThreeObject3DLike } from './types'
import { clamp01 } from './math'
import { strictColorLikeToArray, validatedColorLikeToArray } from './color'
import { objectChildren } from './objects'
import { ByteType, HalfFloatType, IntType, LinearEncoding, ShortType, UnsignedByteType, UnsignedInt101111Type, UnsignedInt5999Type, UnsignedIntType, UnsignedShort4444Type, UnsignedShort5551Type, UnsignedShortType, sRGBEncoding } from './materials.part-001'
import { filterModeToString, minFilterModeToString, wrapModeToString } from './materials.part-009'
import { PhysicalMapFeatureGates, hasExplicitMipmaps } from './materials.part-010'
import { halfFloatDataToRgba8, normalizedSignedIntegerDataToRgba8, packedUnsignedInt101111ToRgba8, packedUnsignedInt5999ToRgba8, packedUnsignedShort4444ToRgba8, packedUnsignedShort5551ToRgba8, writeOneChannelRgba8, writeTwoChannelRgba8 } from './materials.part-012'
export function assertCompatiblePackedPhysicalMapSamplers(
  material: ThreeMaterialLike,
  features: PhysicalMapFeatureGates,
): void {
  const scalarSlots = activePackedPhysicalScalarMapSlots(material, features)
  const sheenSlots = features.sheen
    ? [
      ['sheenColorMap', material.sheenColorMap],
      ['sheenRoughnessMap', material.sheenRoughnessMap],
    ] satisfies Array<[string, ThreeTextureLike | null | undefined]>
    : []
  assertNoPackedPhysicalMapMipmaps('physical extension scalar maps', scalarSlots)
  assertNoPackedPhysicalMapMipmaps('physical extension sheen maps', sheenSlots)
  assertNoPackedPhysicalMapMipmaps('physical extension specular maps', [
    ['specularColorMap', material.specularColorMap],
    ['specularIntensityMap', material.specularIntensityMap],
  ])
  assertMatchingSamplerSettings('physical extension scalar maps', scalarSlots)
  assertMatchingSamplerSettings('physical extension sheen maps', sheenSlots)
  assertMatchingSamplerSettings('physical extension specular maps', [
    ['specularColorMap', material.specularColorMap],
    ['specularIntensityMap', material.specularIntensityMap],
  ])
}

export function activePackedPhysicalScalarMapSlots(
  material: ThreeMaterialLike,
  features: PhysicalMapFeatureGates,
): Array<[string, ThreeTextureLike | null | undefined]> {
  const slots: Array<[string, ThreeTextureLike | null | undefined]> = []
  if (features.clearcoat) {
    slots.push(
      ['clearcoatMap', material.clearcoatMap],
      ['clearcoatRoughnessMap', material.clearcoatRoughnessMap],
    )
  }
  if (features.transmission) {
    slots.push(
      ['transmissionMap', material.transmissionMap],
      ['thicknessMap', material.thicknessMap],
    )
  }
  if (features.anisotropy) {
    slots.push(['anisotropyMap', material.anisotropyMap])
  }
  if (features.iridescence) {
    slots.push(
      ['iridescenceMap', material.iridescenceMap],
      ['iridescenceThicknessMap', material.iridescenceThicknessMap],
    )
  }
  return slots
}

export function assertNoPackedPhysicalMapMipmaps(groupLabel: string, slots: Array<[string, ThreeTextureLike | null | undefined]>): void {
  for (const [label, texture] of slots) {
    if (!texture || !hasExplicitMipmaps(texture, `material.${label}`)) continue
    throw new Error(
      `${groupLabel} are packed into one native texture, and explicit mipmaps for ${label} are not supported by @headless-three/renderer yet. Remove texture.mipmaps from packed physical-extension maps or rely on generated mipmaps from the packed base level.`,
    )
  }
}

export function assertMatchingSamplerSettings(groupLabel: string, slots: Array<[string, ThreeTextureLike | null | undefined]>): void {
  let first: { label: string; signature: string } | null = null
  for (const [label, texture] of slots) {
    if (!texture) continue
    const signature = samplerSignature(texture, `material.${label}`)
    if (!first) {
      first = { label, signature }
      continue
    }
    if (signature !== first.signature) {
      throw new Error(
        `${groupLabel} are packed into one native texture and must use matching wrap/filter/anisotropy sampler settings. ${label} differs from ${first.label}; use matching wrapS/wrapT/magFilter/minFilter/anisotropy values or render separate passes until independent packed-channel samplers are supported.`,
      )
    }
  }
}

export function samplerSignature(texture: ThreeTextureLike, label: string): string {
  return [
    wrapModeToString(texture.wrapS) ?? 'clamp',
    wrapModeToString(texture.wrapT) ?? 'clamp',
    filterModeToString(texture.magFilter) ?? 'linear',
    minFilterModeToString(texture) ?? 'linear',
    String(textureAnisotropy(texture, label) ?? 1),
  ].join('|')
}

export function textureAnisotropy(map: ThreeTextureLike | null | undefined, label: string): number | undefined {
  const value = map?.anisotropy
  if (value == null) return undefined
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${label}.anisotropy must be a finite number.`)
  }
  if (value <= 1) return undefined
  return Math.max(1, Math.min(16, Math.floor(value)))
}

export function textureUnpackAlignment(map: ThreeTextureLike | null | undefined, label: string): number | undefined {
  const value = map?.unpackAlignment
  if (value == null) return undefined
  if (!Number.isInteger(value)) {
    throw new TypeError(`${label}.unpackAlignment must be an integer.`)
  }
  if (value === 1 || value === 2 || value === 4 || value === 8) return value
  throw new Error(
    `${label}.unpackAlignment ${value} is not supported by @headless-three/renderer. Use 1, 2, 4, or 8.`,
  )
}

export function textureTransform(map: ThreeTextureLike | null | undefined, label: string): number[] | undefined {
  const flipY = optionalTextureBoolean(map?.flipY, `${label}.flipY`) !== false
  const flipTransform = flipY ? undefined : [1, 0, 0, 0, -1, 1]
  if (!map) return flipTransform

  const matrixAutoUpdate = optionalTextureBoolean(map.matrixAutoUpdate, `${label}.matrixAutoUpdate`)
  if (matrixAutoUpdate === false) {
    const e = map.matrix?.elements
    if (!e || e.length !== 9) {
      throw new TypeError(`${label}.matrix.elements must contain nine finite numbers.`)
    }
    validateFiniteMatrix3(e, `${label}.matrix.elements`)
    return composeTextureTransformWithFlipY([e[0], e[3], e[6], e[1], e[4], e[7]], flipY)
  }

  const tx = textureVector2Component(map.offset, `${label}.offset`, 'x', 0)
  const ty = textureVector2Component(map.offset, `${label}.offset`, 'y', 0)
  const sx = textureVector2Component(map.repeat, `${label}.repeat`, 'x', 1)
  const sy = textureVector2Component(map.repeat, `${label}.repeat`, 'y', 1)
  const rotation = finiteTextureTransformNumber(map.rotation, `${label}.rotation`, 0)
  const cx = textureVector2Component(map.center, `${label}.center`, 'x', 0)
  const cy = textureVector2Component(map.center, `${label}.center`, 'y', 0)
  if (tx === 0 && ty === 0 && sx === 1 && sy === 1 && rotation === 0 && cx === 0 && cy === 0) {
    return flipTransform
  }

  const c = Math.cos(rotation)
  const s = Math.sin(rotation)
  return composeTextureTransformWithFlipY([
    sx * c,
    sx * s,
    -sx * (c * cx + s * cy) + cx + tx,
    -sy * s,
    sy * c,
    -sy * (-s * cx + c * cy) + cy + ty,
  ], flipY)
}

export function composeTextureTransformWithFlipY(transform: number[], flipY: boolean): number[] {
  if (flipY) return transform
  const [a, c, tx, b, d, ty] = transform
  return [a, -c, c + tx, b, -d, d + ty]
}

export function optionalTextureBoolean(value: unknown, label: string): boolean | undefined {
  if (value == null) return undefined
  if (typeof value === 'boolean') return value
  throw new TypeError(`${label} must be a boolean.`)
}

export function textureVector2Component(value: unknown, label: string, component: 'x' | 'y', fallback: number): number {
  if (value == null) return fallback
  if (typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be a vector-like object.`)
  }
  return finiteTextureTransformNumber((value as { x?: unknown; y?: unknown })[component], `${label}.${component}`, fallback)
}

export function textureColorSpace(map: ThreeTextureLike | null | undefined): string | undefined {
  if (!map) return undefined
  if (map.colorSpace === 'srgb') return 'srgb'
  if (
    map.colorSpace === 'srgb-linear' ||
    map.colorSpace === 'linear-srgb' ||
    map.colorSpace === 'linearsrgb' ||
    map.colorSpace === 'linear'
  ) {
    return 'linear'
  }
  if (map.colorSpace != null && map.colorSpace !== '') {
    throw new Error(
      `texture.colorSpace ${String(map.colorSpace)} is not supported by @headless-three/renderer. Use THREE.SRGBColorSpace, THREE.LinearSRGBColorSpace, or THREE.NoColorSpace.`,
    )
  }
  if (map.encoding === sRGBEncoding) return 'srgb'
  if (map.encoding != null && map.encoding !== LinearEncoding) {
    throw new Error(
      `texture.encoding ${String(map.encoding)} is not supported by @headless-three/renderer. Use sRGBEncoding, LinearEncoding, or texture.colorSpace with THREE.SRGBColorSpace/THREE.LinearSRGBColorSpace.`,
    )
  }
  return undefined
}

export function finiteOrDefault(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

export function finiteTextureTransformNumber(value: unknown, label: string, fallback: number): number {
  if (value == null) return fallback
  if (typeof value === 'number' && Number.isFinite(value)) return value
  throw new TypeError(`${label} must be a finite number.`)
}

export function validateFiniteMatrix3(values: ArrayLike<unknown>, label: string): void {
  for (let i = 0; i < 9; i += 1) {
    if (typeof values[i] !== 'number' || !Number.isFinite(values[i])) {
      throw new TypeError(`${label}[${i}] must be a finite number.`)
    }
  }
}

export function toRgba8(
  data: ArrayLike<number>,
  width: number,
  height: number,
  options: { narrowChannels?: boolean; type?: number; format?: unknown } = {},
): Uint8Array | null {
  const pixels = width * height
  const allowNarrowChannels = options.narrowChannels !== false
  const textureType = options.type ?? UnsignedByteType
  const textureFormat = options.format

  if (textureType === HalfFloatType) {
    if (!(data instanceof Uint16Array)) return null
    return halfFloatDataToRgba8(data, pixels, allowNarrowChannels, textureFormat)
  }
  if (textureType === ByteType) {
    if (!(data instanceof Int8Array)) return null
    return normalizedSignedIntegerDataToRgba8(data, pixels, allowNarrowChannels, 0x7f, textureFormat)
  }
  if (textureType === ShortType) {
    if (!(data instanceof Int16Array)) return null
    return normalizedSignedIntegerDataToRgba8(data, pixels, allowNarrowChannels, 0x7fff, textureFormat)
  }
  if (textureType === UnsignedShortType) {
    if (!(data instanceof Uint16Array)) return null
    return normalizedUnsignedIntegerDataToRgba8(data, pixels, allowNarrowChannels, 0xffff, textureFormat)
  }
  if (textureType === IntType) {
    if (!(data instanceof Int32Array)) return null
    return normalizedSignedIntegerDataToRgba8(data, pixels, allowNarrowChannels, 0x7fffffff, textureFormat)
  }
  if (textureType === UnsignedIntType) {
    if (!(data instanceof Uint32Array)) return null
    return normalizedUnsignedIntegerDataToRgba8(data, pixels, allowNarrowChannels, 0xffffffff, textureFormat)
  }
  if (textureType === UnsignedInt5999Type) {
    if (!(data instanceof Uint32Array)) return null
    return packedUnsignedInt5999ToRgba8(data, pixels)
  }
  if (textureType === UnsignedInt101111Type) {
    if (!(data instanceof Uint32Array)) return null
    return packedUnsignedInt101111ToRgba8(data, pixels)
  }
  if (textureType === UnsignedShort4444Type) {
    if (!(data instanceof Uint16Array)) return null
    return packedUnsignedShort4444ToRgba8(data, pixels)
  }
  if (textureType === UnsignedShort5551Type) {
    if (!(data instanceof Uint16Array)) return null
    return packedUnsignedShort5551ToRgba8(data, pixels)
  }

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
    if (allowNarrowChannels && data.length === pixels * 2) {
      const out = new Uint8Array(pixels * 4)
      for (let i = 0; i < pixels; i++) {
        writeTwoChannelRgba8(out, i, data[i * 2], data[i * 2 + 1], textureFormat)
      }
      return out
    }
    if (allowNarrowChannels && data.length === pixels) {
      const out = new Uint8Array(pixels * 4)
      for (let i = 0; i < pixels; i++) {
        writeOneChannelRgba8(out, i, data[i], textureFormat)
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
    if (allowNarrowChannels && data.length === pixels * 2) {
      const out = new Uint8Array(pixels * 4)
      for (let i = 0; i < pixels; i++) {
        writeTwoChannelRgba8(
          out,
          i,
          Math.max(0, Math.min(255, Math.round(data[i * 2] * 255))),
          Math.max(0, Math.min(255, Math.round(data[i * 2 + 1] * 255))),
          textureFormat,
        )
      }
      return out
    }
    if (allowNarrowChannels && data.length === pixels) {
      const out = new Uint8Array(pixels * 4)
      for (let i = 0; i < pixels; i++) {
        const value = Math.max(0, Math.min(255, Math.round(data[i] * 255)))
        writeOneChannelRgba8(out, i, value, textureFormat)
      }
      return out
    }
    return null
  }

  // Uint16Array or other numeric typed arrays — treat as 8-bit range after clamping
  if (ArrayBuffer.isView(data)) {
    if (data.length === pixels * 4) {
      const out = new Uint8Array(pixels * 4)
      for (let i = 0; i < pixels * 4; i++) {
        out[i] = Math.max(0, Math.min(255, (data as any)[i]))
      }
      return out
    }
    if (data.length === pixels * 3) {
      const out = new Uint8Array(pixels * 4)
      for (let i = 0; i < pixels; i++) {
        out[i * 4] = Math.max(0, Math.min(255, (data as any)[i * 3]))
        out[i * 4 + 1] = Math.max(0, Math.min(255, (data as any)[i * 3 + 1]))
        out[i * 4 + 2] = Math.max(0, Math.min(255, (data as any)[i * 3 + 2]))
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
          Math.max(0, Math.min(255, (data as any)[i * 2])),
          Math.max(0, Math.min(255, (data as any)[i * 2 + 1])),
          textureFormat,
        )
      }
      return out
    }
    if (allowNarrowChannels && data.length === pixels) {
      const out = new Uint8Array(pixels * 4)
      for (let i = 0; i < pixels; i++) {
        const value = Math.max(0, Math.min(255, (data as any)[i]))
        writeOneChannelRgba8(out, i, value, textureFormat)
      }
      return out
    }
  }

  if (data.length === pixels * 4) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels * 4; i++) {
      out[i] = Math.max(0, Math.min(255, data[i]))
    }
    return out
  }
  if (data.length === pixels * 3) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels; i++) {
      out[i * 4] = Math.max(0, Math.min(255, data[i * 3]))
      out[i * 4 + 1] = Math.max(0, Math.min(255, data[i * 3 + 1]))
      out[i * 4 + 2] = Math.max(0, Math.min(255, data[i * 3 + 2]))
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
        Math.max(0, Math.min(255, data[i * 2])),
        Math.max(0, Math.min(255, data[i * 2 + 1])),
        textureFormat,
      )
    }
    return out
  }
  if (allowNarrowChannels && data.length === pixels) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels; i++) {
      const value = Math.max(0, Math.min(255, data[i]))
      writeOneChannelRgba8(out, i, value, textureFormat)
    }
    return out
  }

  return null
}

export function normalizedUnsignedIntegerDataToRgba8(
  data: Uint16Array | Uint32Array,
  pixels: number,
  allowNarrowChannels: boolean,
  maxValue: number,
  format: unknown,
): Uint8Array | null {
  if (data.length === pixels * 4) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels * 4; i++) {
      out[i] = normalizedUnsignedIntegerToByte(data[i], maxValue)
    }
    return out
  }
  if (data.length === pixels * 3) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels; i++) {
      out[i * 4] = normalizedUnsignedIntegerToByte(data[i * 3], maxValue)
      out[i * 4 + 1] = normalizedUnsignedIntegerToByte(data[i * 3 + 1], maxValue)
      out[i * 4 + 2] = normalizedUnsignedIntegerToByte(data[i * 3 + 2], maxValue)
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
        normalizedUnsignedIntegerToByte(data[i * 2], maxValue),
        normalizedUnsignedIntegerToByte(data[i * 2 + 1], maxValue),
        format,
      )
    }
    return out
  }
  if (allowNarrowChannels && data.length === pixels) {
    const out = new Uint8Array(pixels * 4)
    for (let i = 0; i < pixels; i++) {
      const value = normalizedUnsignedIntegerToByte(data[i], maxValue)
      writeOneChannelRgba8(out, i, value, format)
    }
    return out
  }
  return null
}

export function normalizedUnsignedIntegerToByte(value: number, maxValue: number): number {
  return Math.max(0, Math.min(255, Math.round((value / maxValue) * 255)))
}
