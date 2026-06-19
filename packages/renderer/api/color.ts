import type { Color4, ThreeColorLike, ThreeSceneRootLike, RenderOptions } from './types'
import { clamp01, areFiniteNumbers } from './math'

type ColorLikeWithAlpha = Partial<ThreeColorLike> & { a?: unknown }
interface ThreeColorStyleParser {
  r: number
  g: number
  b: number
  setStyle(style: string): this
  convertLinearToSRGB(): this
}
interface ThreeColorManagementState {
  enabled: boolean
}

// eslint-disable-next-line @typescript-eslint/no-var-requires
const { Color: ThreeColor, ColorManagement: ThreeColorManagement } = require('three') as {
  Color: new () => ThreeColorStyleParser
  ColorManagement: ThreeColorManagementState
}

export const DEFAULT_BACKGROUND_COLOR: Color4 = [0.04, 0.045, 0.05, 1]

export function colorLikeToArray(value: unknown): Color4 | null {
  if (!value) return null
  if (Array.isArray(value)) return normalizeColorArray(value)
  const v = value as Partial<ThreeColorLike>
  if (v.isColor === true || areFiniteNumbers(v.r, v.g, v.b)) {
    return [clamp01(v.r!), clamp01(v.g!), clamp01(v.b!), 1]
  }
  return null
}

export function strictColorLikeToArray(value: unknown, label: string): Color4 | null {
  if (!value) return null
  if (Array.isArray(value)) return normalizeColorArray(value, label)
  if (typeof value === 'string') return cssColorStringToArray(value, label)
  if (typeof value !== 'object') return null

  const color = value as ColorLikeWithAlpha
  if (!isColorShaped(color)) return null

  return [
    clamp01(assertFiniteColorComponent(color.r, `${label}.r`)),
    clamp01(assertFiniteColorComponent(color.g, `${label}.g`)),
    clamp01(assertFiniteColorComponent(color.b, `${label}.b`)),
    clamp01(color.a === undefined ? 1 : assertFiniteColorComponent(color.a, `${label}.a`)),
  ]
}

export function validatedColorLikeToArray(value: unknown, label: string): Color4 | null {
  if (value == null) return null
  if (Array.isArray(value)) return normalizeColorArray(value, label)
  if (typeof value !== 'object') {
    throw new TypeError(`${label} must be a color-like object or [r, g, b].`)
  }

  const color = value as ColorLikeWithAlpha
  if (!isColorShaped(color)) {
    throw new TypeError(`${label} must be a color-like object or [r, g, b].`)
  }

  return [
    clamp01(assertFiniteColorComponent(color.r, `${label}.r`)),
    clamp01(assertFiniteColorComponent(color.g, `${label}.g`)),
    clamp01(assertFiniteColorComponent(color.b, `${label}.b`)),
    clamp01(color.a === undefined ? 1 : assertFiniteColorComponent(color.a, `${label}.a`)),
  ]
}

export function cssColorStringToArray(value: string, label: string): Color4 {
  if (value.trim() === '') {
    throw new TypeError(`${label} must be a non-empty CSS color string.`)
  }

  const warnings: string[] = []
  const originalWarn = console.warn
  const originalColorManagementEnabled = ThreeColorManagement.enabled
  console.warn = (...args: any[]) => { warnings.push(args.map(String).join(' ')) }
  try {
    ThreeColorManagement.enabled = true
    const color = new ThreeColor().setStyle(value)
    const invalidWarning = warnings.find((message) => /Unknown color|Invalid hex color/i.test(message))
    if (invalidWarning) {
      throw new TypeError(`${label} ${JSON.stringify(value)} is not a supported CSS color string.`)
    }
    color.convertLinearToSRGB()
    return [clamp01(color.r), clamp01(color.g), clamp01(color.b), 1]
  } finally {
    console.warn = originalWarn
    ThreeColorManagement.enabled = originalColorManagementEnabled
  }
}

export function normalizeColorArray(values: number[], label?: string): Color4 {
  if (values.length !== 3 && values.length !== 4) {
    throw new TypeError(label ? `${label} must be [r, g, b] or [r, g, b, a]` : 'Color arrays must be [r, g, b] or [r, g, b, a]')
  }
  return [
    clamp01(assertFiniteColorComponent(values[0], colorComponentLabel(label, 0, 'r'))),
    clamp01(assertFiniteColorComponent(values[1], colorComponentLabel(label, 1, 'g'))),
    clamp01(assertFiniteColorComponent(values[2], colorComponentLabel(label, 2, 'b'))),
    clamp01(values.length === 4 ? assertFiniteColorComponent(values[3], colorComponentLabel(label, 3, 'a')) : 1),
  ]
}

export function resolveBackground(
  scene: ThreeSceneRootLike,
  options: RenderOptions,
  hasBackgroundTexture = false,
  fallbackBackground: Color4 = DEFAULT_BACKGROUND_COLOR,
): Color4 {
  const hasBackgroundOverride = options.background !== undefined
  if (hasBackgroundOverride) {
    const color = strictColorLikeToArray(options.background, 'options.background')
    if (color) return color
    if (options.background == null || hasBackgroundTexture) return fallbackBackground
    throw new TypeError('options.background must be a color, texture, or null.')
  }
  const color = strictColorLikeToArray(scene.background, 'scene.background')
  if (color) return color
  if (scene.background == null || hasBackgroundTexture) return fallbackBackground
  throw new TypeError('scene.background must be a color, texture, or null.')
}

function isColorShaped(value: ColorLikeWithAlpha): boolean {
  return value.isColor === true || 'r' in value || 'g' in value || 'b' in value || 'a' in value
}

function colorComponentLabel(label: string | undefined, index: number, component: string): string {
  return label ? `${label}[${index}]` : `color ${component}`
}

function assertFiniteColorComponent(value: unknown, label: string): number {
  if (!Number.isFinite(value)) {
    throw new TypeError(`${label} must be a finite number.`)
  }
  return value as number
}
