import type { ThreePlaneLike } from './types'

export const MAX_CLIPPING_PLANES = 8

export type NativeClippingPlane = [number, number, number, number]

export function extractClippingPlanes(input: readonly ThreePlaneLike[] | null | undefined): NativeClippingPlane[] {
  if (!Array.isArray(input) || input.length === 0) return []

  const planes: NativeClippingPlane[] = []
  for (const plane of input) {
    const parsed = parseClippingPlane(plane)
    if (!parsed) continue
    planes.push(parsed)
    if (planes.length >= MAX_CLIPPING_PLANES) break
  }
  return planes
}

export function flattenClippingPlanes(planes: readonly NativeClippingPlane[]): number[] | undefined {
  if (planes.length === 0) return undefined
  return planes.flatMap((plane) => plane)
}

function parseClippingPlane(plane: ThreePlaneLike | null | undefined): NativeClippingPlane | null {
  if (!plane) return null

  if (isArrayLike(plane) && plane.length >= 4) {
    return normalizedPlane([numberAt(plane, 0), numberAt(plane, 1), numberAt(plane, 2), numberAt(plane, 3)])
  }

  const normal = (plane as { normal?: unknown }).normal
  const constant = (plane as { constant?: unknown }).constant
  if (typeof constant !== 'number' || !Number.isFinite(constant)) return null

  if (isArrayLike(normal) && normal.length >= 3) {
    return normalizedPlane([numberAt(normal, 0), numberAt(normal, 1), numberAt(normal, 2), constant])
  }

  const vector = normal as { x?: unknown; y?: unknown; z?: unknown } | undefined
  return normalizedPlane([numberFrom(vector?.x), numberFrom(vector?.y), numberFrom(vector?.z), constant])
}

function normalizedPlane(values: NativeClippingPlane): NativeClippingPlane | null {
  const [x, y, z, constant] = values
  if (![x, y, z, constant].every(Number.isFinite)) return null
  const length = Math.hypot(x, y, z)
  if (length <= 1e-8) return null
  return [x / length, y / length, z / length, constant / length]
}

function isArrayLike(value: unknown): value is ArrayLike<number> {
  return !!value && typeof (value as { length?: unknown }).length === 'number'
}

function numberAt(values: ArrayLike<number>, index: number): number {
  return numberFrom(values[index])
}

function numberFrom(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : Number.NaN
}
