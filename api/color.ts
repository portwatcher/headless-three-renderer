import type { Color4, ThreeColorLike, ThreeSceneLike, RenderOptions } from './types'
import { clamp01, assertFinite, areFiniteNumbers } from './math'

export function colorLikeToArray(value: unknown): Color4 | null {
  if (!value) return null
  if (Array.isArray(value)) return normalizeColorArray(value)
  const v = value as Partial<ThreeColorLike>
  if (v.isColor === true || areFiniteNumbers(v.r, v.g, v.b)) {
    return [clamp01(v.r!), clamp01(v.g!), clamp01(v.b!), 1]
  }
  return null
}

export function normalizeColorArray(values: number[]): Color4 {
  if (values.length !== 3 && values.length !== 4) {
    throw new TypeError('Color arrays must be [r, g, b] or [r, g, b, a]')
  }
  return [
    clamp01(assertFinite(values[0], 'color r')),
    clamp01(assertFinite(values[1], 'color g')),
    clamp01(assertFinite(values[2], 'color b')),
    clamp01(values.length === 4 ? assertFinite(values[3], 'color a') : 1),
  ]
}

export function resolveBackground(scene: ThreeSceneLike, options: RenderOptions): Color4 {
  if (Array.isArray(options.background)) {
    return normalizeColorArray(options.background)
  }

  const color = colorLikeToArray(options.background) ?? colorLikeToArray(scene.background)
  return color ?? [0.04, 0.045, 0.05, 1]
}
