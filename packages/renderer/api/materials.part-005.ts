import type { Color4, ThreeMaterialLike, PbrProperties, TextureInfo, ThreeTextureLike, ThreeSceneRootLike, ThreeObject3DLike } from './types'
import { clamp01 } from './math'
import { strictColorLikeToArray, validatedColorLikeToArray } from './color'
import { objectChildren } from './objects'
import { BackSide, BasicDepthPacking, DecrementStencilOp, DecrementWrapStencilOp, DoubleSide, FrontSide, IncrementStencilOp, IncrementWrapStencilOp, InvertStencilOp, KeepStencilOp, ObjectSpaceNormalMap, RGBADepthPacking, RGBDepthPacking, RGDepthPacking, ReplaceStencilOp, TangentSpaceNormalMap, ZeroStencilOp } from './materials.part-001'
export function materialStencilOperation(value: unknown, label: string): number {
  if (
    value === ZeroStencilOp ||
    value === KeepStencilOp ||
    value === ReplaceStencilOp ||
    value === IncrementStencilOp ||
    value === DecrementStencilOp ||
    value === IncrementWrapStencilOp ||
    value === DecrementWrapStencilOp ||
    value === InvertStencilOp
  ) {
    return value
  }
  throw new Error(
    `${label} ${String(value)} is not supported by @headless-three/renderer. Use a Three.js stencil operation constant such as KeepStencilOp, ReplaceStencilOp, or InvertStencilOp.`,
  )
}

export function materialSide(material: ThreeMaterialLike): string | undefined {
  if (material.side == null) return undefined
  switch (material.side) {
    case FrontSide:
      return 'front'
    case BackSide:
      return 'back'
    case DoubleSide:
      return 'double'
    default:
      throw new Error(
        `material.side ${String(material.side)} is not supported by @headless-three/renderer. Use FrontSide, BackSide, or DoubleSide.`,
      )
  }
}

export function materialDepthPacking(material: ThreeMaterialLike): number | undefined {
  if (material.depthPacking == null) return undefined
  switch (material.depthPacking) {
    case BasicDepthPacking:
    case RGBADepthPacking:
    case RGBDepthPacking:
    case RGDepthPacking:
      return material.depthPacking
    default:
      throw new Error(
        `material.depthPacking ${String(material.depthPacking)} is not supported by @headless-three/renderer. Use BasicDepthPacking, RGBADepthPacking, RGBDepthPacking, or RGDepthPacking.`,
      )
  }
}

export function materialNormalMapType(material: ThreeMaterialLike): 'tangent' | 'object' {
  if (material.normalMapType == null) return 'tangent'
  switch (material.normalMapType) {
    case TangentSpaceNormalMap:
      return 'tangent'
    case ObjectSpaceNormalMap:
      return 'object'
    default:
      throw new Error(
        `material.normalMapType ${String(material.normalMapType)} is not supported by @headless-three/renderer. Use TangentSpaceNormalMap or ObjectSpaceNormalMap.`,
      )
  }
}

export function materialShadowSide(material: ThreeMaterialLike | undefined): string | undefined {
  if (!material || material.shadowSide == null) return undefined
  switch (material.shadowSide) {
    case FrontSide:
      return 'front'
    case BackSide:
      return 'back'
    case DoubleSide:
      return 'double'
    default:
      throw new Error(
        `material.shadowSide ${String(material.shadowSide)} is not supported by @headless-three/renderer. Use FrontSide, BackSide, DoubleSide, null, or undefined.`,
      )
  }
}

export function finiteIntegerOrDefault(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? Math.trunc(value) : fallback
}

export function textureUvChannel(texture: ThreeTextureLike | null | undefined): number {
  if (texture?.channel == null) return 0
  if (!Number.isInteger(texture.channel)) {
    throw new TypeError('texture.channel must be an integer.')
  }
  const channel = texture!.channel!
  if (channel >= 0 && channel <= 3) return channel
  throw new Error(
    `texture.channel ${channel} is not supported by @headless-three/renderer yet. Use channel 0, 1, 2, or 3 for Three.js UV attributes.`,
  )
}

export function firstOptionalFiniteNumber(entries: Array<[unknown, string]>): number | undefined {
  for (const [value, label] of entries) {
    if (value != null) return optionalFiniteNumber(value, label)
  }
  return undefined
}

export function firstOptionalVector3LikeToArray(entries: Array<[unknown, string]>): number[] | undefined {
  for (const [value, label] of entries) {
    if (value != null) return requiredFiniteVector3LikeToArray(value, label)
  }
  return undefined
}

export function optionalFiniteNumber(value: unknown, label: string): number | undefined {
  if (value == null) return undefined
  if (typeof value === 'number' && Number.isFinite(value)) return value
  throw new TypeError(`${label} must be a finite number.`)
}

export function optionalBoolean(value: unknown, label: string): boolean | undefined {
  if (value == null) return undefined
  if (typeof value === 'boolean') return value
  throw new TypeError(`${label} must be a boolean.`)
}

export function optionalPositiveFiniteNumber(value: unknown, label: string): void {
  if (value == null) return
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a finite number.`)
  }
  if (value <= 0) {
    throw new TypeError(`${label} must be positive.`)
  }
}

export function optionalWireframeLinecap(value: unknown): void {
  if (value == null) return
  if (typeof value !== 'string') {
    throw new TypeError('material.wireframeLinecap must be a string.')
  }
  if (value !== 'butt' && value !== 'round' && value !== 'square') {
    throw new Error(
      `material.wireframeLinecap ${JSON.stringify(value)} is not supported by @headless-three/renderer. Use "butt", "round", "square", null, or undefined.`,
    )
  }
}

export function optionalWireframeLinejoin(value: unknown): void {
  if (value == null) return
  if (typeof value !== 'string') {
    throw new TypeError('material.wireframeLinejoin must be a string.')
  }
  if (value !== 'round' && value !== 'bevel' && value !== 'miter') {
    throw new Error(
      `material.wireframeLinejoin ${JSON.stringify(value)} is not supported by @headless-three/renderer. Use "round", "bevel", "miter", null, or undefined.`,
    )
  }
}

export function optionalMaterialPrecision(value: unknown): void {
  if (value == null) return
  if (typeof value !== 'string') {
    throw new TypeError('material.precision must be "highp", "mediump", "lowp", null, or undefined.')
  }
  if (value !== 'highp' && value !== 'mediump' && value !== 'lowp') {
    throw new Error(
      `material.precision ${JSON.stringify(value)} is not supported by @headless-three/renderer. Use "highp", "mediump", "lowp", null, or undefined.`,
    )
  }
}

export function optionalFiniteNumberOrInfinityDefault(value: unknown, label: string): number | undefined {
  if (value === Number.POSITIVE_INFINITY) return undefined
  return optionalFiniteNumber(value, label)
}

export function requiredFiniteNumber(value: unknown, label: string): number {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  throw new TypeError(`${label} must be a finite number.`)
}

export function materialRangePair(value: unknown, label: string): [number, number] | undefined {
  if (value == null) return undefined
  if (typeof value !== 'object') {
    throw new TypeError(`${label} must be an array-like pair.`)
  }
  const range = value as ArrayLike<unknown>
  if (typeof range.length !== 'number' || range.length < 2) {
    throw new TypeError(`${label} must contain at least two values.`)
  }
  return [
    requiredFiniteNumber(range[0], `${label}[0]`),
    requiredFiniteNumber(range[1], `${label}[1]`),
  ]
}

export function finiteNumberOrDefault(value: unknown, label: string, fallback: number): number {
  return optionalFiniteNumber(value, label) ?? fallback
}

export function vector3LikeToArray(value: unknown): number[] | undefined {
  if (!value || typeof value !== 'object') return undefined

  const arrayLike = value as ArrayLike<unknown>
  if (typeof arrayLike.length === 'number' && arrayLike.length >= 3) {
    const x = arrayLike[0]
    const y = arrayLike[1]
    const z = arrayLike[2]
    if (typeof x === 'number' && typeof y === 'number' && typeof z === 'number'
      && Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
      return [x, y, z]
    }
  }

  const vector = value as { x?: unknown; y?: unknown; z?: unknown }
  const { x, y, z } = vector
  if (typeof x === 'number' && typeof y === 'number' && typeof z === 'number'
    && Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
    return [x, y, z]
  }

  return undefined
}

export function requiredFiniteVector3LikeToArray(value: unknown, label: string): number[] {
  if (!value || typeof value !== 'object') {
    throw new TypeError(`${label} must be a finite Vector3-like value.`)
  }

  const arrayLike = value as ArrayLike<unknown>
  if (typeof arrayLike.length === 'number' && arrayLike.length >= 3) {
    return [
      requiredFiniteNumber(arrayLike[0], `${label}[0]`),
      requiredFiniteNumber(arrayLike[1], `${label}[1]`),
      requiredFiniteNumber(arrayLike[2], `${label}[2]`),
    ]
  }

  const vector = value as { x?: unknown; y?: unknown; z?: unknown }
  if ('x' in vector || 'y' in vector || 'z' in vector) {
    return [
      requiredFiniteNumber(vector.x, `${label}.x`),
      requiredFiniteNumber(vector.y, `${label}.y`),
      requiredFiniteNumber(vector.z, `${label}.z`),
    ]
  }

  throw new TypeError(`${label} must be a finite Vector3-like value.`)
}

export function materialEnvMapRotation(material: ThreeMaterialLike): ThreeMaterialLike['envMapRotation'] | undefined {
  const value = material.envMapRotation
  if (value == null) return undefined
  const components = requiredVector3LikeToArray(value, 'material.envMapRotation')
  return components.some((component) => Math.abs(component) > 1e-12)
    ? value
    : undefined
}

export function requiredVector3LikeToArray(value: unknown, label: string): number[] {
  if (!value || typeof value !== 'object') {
    throw new TypeError(`${label} must be a finite Vector3-like value.`)
  }

  const arrayLike = value as ArrayLike<unknown>
  if (typeof arrayLike.length === 'number' && arrayLike.length >= 3) {
    validateEulerLikeOrder(arrayLike[3], `${label}[3]`)
    return [
      requiredFiniteNumber(arrayLike[0], `${label}[0]`),
      requiredFiniteNumber(arrayLike[1], `${label}[1]`),
      requiredFiniteNumber(arrayLike[2], `${label}[2]`),
    ]
  }

  const vector = value as { x?: unknown; y?: unknown; z?: unknown }
  if ('x' in vector || 'y' in vector || 'z' in vector) {
    validateEulerLikeOrder((vector as { order?: unknown }).order, `${label}.order`)
    return [
      requiredFiniteNumber(vector.x, `${label}.x`),
      requiredFiniteNumber(vector.y, `${label}.y`),
      requiredFiniteNumber(vector.z, `${label}.z`),
    ]
  }

  throw new TypeError(`${label} must be a finite Vector3-like value.`)
}

export function validateEulerLikeOrder(value: unknown, label: string): void {
  if (value == null) return
  if (
    value === 'XYZ' ||
    value === 'YXZ' ||
    value === 'ZXY' ||
    value === 'ZYX' ||
    value === 'YZX' ||
    value === 'XZY'
  ) {
    return
  }
  throw new TypeError(`${label} must be one of XYZ, YXZ, ZXY, ZYX, YZX, or XZY.`)
}

export function sameVector3Like(left: unknown, right: unknown): boolean {
  const leftComponents = vector3LikeToArray(left)
  const rightComponents = vector3LikeToArray(right)
  if (!leftComponents || !rightComponents) return false
  return leftComponents.every((component, index) => Math.abs(component - rightComponents[index]) <= 1e-12)
    && eulerLikeOrder(left) === eulerLikeOrder(right)
}

export function eulerLikeOrder(value: unknown): string {
  if (!value || typeof value !== 'object') return 'XYZ'
  const arrayLike = value as ArrayLike<unknown>
  if (typeof arrayLike.length === 'number' && arrayLike.length >= 4 && typeof arrayLike[3] === 'string') {
    return arrayLike[3]
  }
  const order = (value as { order?: unknown }).order
  return typeof order === 'string' ? order : 'XYZ'
}

export function extractCustomFragmentShader(material: ThreeMaterialLike | undefined): string | undefined {
  if (!material) return undefined

  const candidates: Array<[unknown, string]> = [
    [material.customFragmentWgsl, 'material.customFragmentWgsl'],
    [material.customFragmentShader, 'material.customFragmentShader'],
    [material.headlessFragmentWgsl, 'material.headlessFragmentWgsl'],
    [material.headlessFragmentShader, 'material.headlessFragmentShader'],
  ]

  const hints = customFragmentHints(material.userData)
  if (hints) {
    candidates.push(
      [hints.value.fragmentWgsl, `${hints.label}.fragmentWgsl`],
      [hints.value.fragmentShader, `${hints.label}.fragmentShader`],
      [hints.value.customFragmentWgsl, `${hints.label}.customFragmentWgsl`],
      [hints.value.customFragmentShader, `${hints.label}.customFragmentShader`],
    )
  }

  for (const [value, label] of candidates) {
    const candidate = customFragmentCandidate(value, label)
    if (candidate) return candidate
  }

  return undefined
}

export function materialRendererHints(userData: Record<string, any> | undefined): { value: Record<string, unknown>; label: string } | undefined {
  if (userData == null) return undefined
  if (typeof userData !== 'object' || Array.isArray(userData)) {
    throw new TypeError('material.userData must be an object.')
  }
  const value = userData.headlessThreeRenderer ?? userData.headlessRenderer
  if (value == null) return undefined
  const label = userData.headlessThreeRenderer != null
    ? 'material.userData.headlessThreeRenderer'
    : 'material.userData.headlessRenderer'
  if (typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be an object.`)
  }
  return { value: value as Record<string, unknown>, label }
}

export function customFragmentHints(userData: Record<string, any> | undefined): { value: Record<string, unknown>; label: string } | undefined {
  return materialRendererHints(userData)
}

export function customFragmentCandidate(value: unknown, label: string): string | undefined {
  if (value == null) return undefined
  if (typeof value !== 'string') {
    throw new TypeError(`${label} must be a string.`)
  }
  const candidate = value.trim()
  return candidate.length > 0 ? candidate : undefined
}
