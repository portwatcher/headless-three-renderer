import type { ThreePlaneLike } from './types'

export const MAX_CLIPPING_PLANES = 8

export type NativeClippingPlane = [number, number, number, number]

export function extractClippingPlanes(
  input: readonly ThreePlaneLike[] | null | undefined,
  label = 'clippingPlanes',
  maxPlanes = MAX_CLIPPING_PLANES,
): NativeClippingPlane[] {
  if (input == null) return []
  if (!Array.isArray(input)) {
    throw new TypeError(`${label} must be an array of clipping planes.`)
  }
  if (input.length === 0) return []

  const planes: NativeClippingPlane[] = []
  for (let i = 0; i < input.length; i += 1) {
    const parsed = parseClippingPlane(input[i], `${label}[${i}]`)
    if (!parsed) continue
    if (planes.length >= maxPlanes) {
      throw new Error(
        `${label} exceeds the remaining native clipping plane budget (${maxPlanes} of ${MAX_CLIPPING_PLANES} remaining). @headless-three/renderer supports at most ${MAX_CLIPPING_PLANES} active global, group, and material clipping planes; reduce the active planes or render separate passes.`,
      )
    }
    planes.push(parsed)
  }
  return planes
}

export function flattenClippingPlanes(planes: readonly NativeClippingPlane[]): number[] | undefined {
  if (planes.length === 0) return undefined
  return planes.flatMap((plane) => plane)
}

function parseClippingPlane(plane: ThreePlaneLike | null | undefined, label: string): NativeClippingPlane | null {
  if (!plane) return null

  if (isArrayLike(plane)) {
    if (plane.length < 4) {
      throw new TypeError(`${label} must contain four finite numbers.`)
    }
    return normalizedPlane([
      requiredFiniteNumber(plane[0], `${label}[0]`),
      requiredFiniteNumber(plane[1], `${label}[1]`),
      requiredFiniteNumber(plane[2], `${label}[2]`),
      requiredFiniteNumber(plane[3], `${label}[3]`),
    ], label)
  }

  const normal = (plane as { normal?: unknown }).normal
  const constant = (plane as { constant?: unknown }).constant
  const finiteConstant = requiredFiniteNumber(constant, `${label}.constant`)

  if (isArrayLike(normal)) {
    if (normal.length < 3) {
      throw new TypeError(`${label}.normal must contain three finite numbers.`)
    }
    return normalizedPlane([
      requiredFiniteNumber(normal[0], `${label}.normal[0]`),
      requiredFiniteNumber(normal[1], `${label}.normal[1]`),
      requiredFiniteNumber(normal[2], `${label}.normal[2]`),
      finiteConstant,
    ], label)
  }

  const vector = normal as { x?: unknown; y?: unknown; z?: unknown } | undefined
  return normalizedPlane([
    requiredFiniteNumber(vector?.x, `${label}.normal.x`),
    requiredFiniteNumber(vector?.y, `${label}.normal.y`),
    requiredFiniteNumber(vector?.z, `${label}.normal.z`),
    finiteConstant,
  ], label)
}

function normalizedPlane(values: NativeClippingPlane, label: string): NativeClippingPlane {
  const [x, y, z, constant] = values
  const length = Math.hypot(x, y, z)
  if (length <= 1e-8) {
    throw new TypeError(`${label}.normal must have non-zero finite length.`)
  }
  return [x / length, y / length, z / length, constant / length]
}

function isArrayLike(value: unknown): value is ArrayLike<unknown> {
  return !!value && typeof (value as { length?: unknown }).length === 'number'
}

function requiredFiniteNumber(value: unknown, label: string): number {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  throw new TypeError(`${label} must be a finite number.`)
}
