import type { Mat4, ThreeMatrix4Like } from './types'

export const IDENTITY_4X4: Mat4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

// Three.js projection matrices use WebGL/OpenGL clip-space depth (-1..1).
// wgpu/WebGPU expects 0..1, so the public adapter converts clip-space here.
export const OPENGL_TO_WGPU_CLIP: Mat4 = [
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 0.5, 0,
  0, 0, 0.5, 1,
]

export function multiplyMatrices(a: Mat4, b: Mat4): Mat4 {
  const out = new Array<number>(16)
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      out[column * 4 + row] =
        a[row] * b[column * 4] +
        a[4 + row] * b[column * 4 + 1] +
        a[8 + row] * b[column * 4 + 2] +
        a[12 + row] * b[column * 4 + 3]
    }
  }
  return out
}

export function invertMatrix4(m: Mat4): Mat4 {
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3]
  const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7]
  const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11]
  const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15]

  const b00 = a00 * a11 - a01 * a10
  const b01 = a00 * a12 - a02 * a10
  const b02 = a00 * a13 - a03 * a10
  const b03 = a01 * a12 - a02 * a11
  const b04 = a01 * a13 - a03 * a11
  const b05 = a02 * a13 - a03 * a12
  const b06 = a20 * a31 - a21 * a30
  const b07 = a20 * a32 - a22 * a30
  const b08 = a20 * a33 - a23 * a30
  const b09 = a21 * a32 - a22 * a31
  const b10 = a21 * a33 - a23 * a31
  const b11 = a22 * a33 - a23 * a32

  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06
  if (Math.abs(det) < 1e-12) return IDENTITY_4X4.slice()
  det = 1.0 / det

  return [
    (a11 * b11 - a12 * b10 + a13 * b09) * det,
    (a02 * b10 - a01 * b11 - a03 * b09) * det,
    (a31 * b05 - a32 * b04 + a33 * b03) * det,
    (a22 * b04 - a21 * b05 - a23 * b03) * det,
    (a12 * b08 - a10 * b11 - a13 * b07) * det,
    (a00 * b11 - a02 * b08 + a03 * b07) * det,
    (a32 * b02 - a30 * b05 - a33 * b01) * det,
    (a20 * b05 - a22 * b02 + a23 * b01) * det,
    (a10 * b10 - a11 * b08 + a13 * b06) * det,
    (a01 * b08 - a00 * b10 - a03 * b06) * det,
    (a30 * b04 - a31 * b02 + a33 * b00) * det,
    (a21 * b02 - a20 * b04 - a23 * b00) * det,
    (a11 * b07 - a10 * b09 - a12 * b06) * det,
    (a00 * b09 - a01 * b07 + a02 * b06) * det,
    (a31 * b01 - a30 * b03 - a32 * b00) * det,
    (a20 * b03 - a21 * b01 + a22 * b00) * det,
  ]
}

export function blendBoneMatrices(
  boneMatrices: Mat4[],
  boneCount: number,
  ji0: number, ji1: number, ji2: number, ji3: number,
  jw0: number, jw1: number, jw2: number, jw3: number,
): Mat4 {
  const out = new Array<number>(16)
  for (let k = 0; k < 16; k++) {
    out[k] = 0
  }

  if (jw0 > 0 && ji0 >= 0 && ji0 < boneCount) {
    const m = boneMatrices[ji0]
    for (let k = 0; k < 16; k++) out[k] += jw0 * m[k]
  }
  if (jw1 > 0 && ji1 >= 0 && ji1 < boneCount) {
    const m = boneMatrices[ji1]
    for (let k = 0; k < 16; k++) out[k] += jw1 * m[k]
  }
  if (jw2 > 0 && ji2 >= 0 && ji2 < boneCount) {
    const m = boneMatrices[ji2]
    for (let k = 0; k < 16; k++) out[k] += jw2 * m[k]
  }
  if (jw3 > 0 && ji3 >= 0 && ji3 < boneCount) {
    const m = boneMatrices[ji3]
    for (let k = 0; k < 16; k++) out[k] += jw3 * m[k]
  }

  const sum = jw0 + jw1 + jw2 + jw3
  if (sum < 1e-8) {
    return IDENTITY_4X4
  }

  return out
}

export function matrixElements(matrix: ThreeMatrix4Like, label: string): Mat4 {
  const elements = matrix?.elements
  if (!elements || elements.length !== 16) {
    throw new TypeError(`${label} must be a THREE.Matrix4`)
  }
  return Array.from(elements, (value) => assertFinite(value, label))
}

export function clampInteger(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return max
  return Math.max(min, Math.min(max, Math.trunc(value)))
}

export function numberOrUndefined(value: unknown): number | undefined {
  return Number.isFinite(value) ? (value as number) : undefined
}

export function isFinitePositive(value: unknown): value is number {
  return Number.isFinite(value) && (value as number) > 0
}

export function assertFinite(value: number, label: string): number {
  if (!Number.isFinite(value)) {
    throw new TypeError(`${label} must be finite`)
  }
  return value
}

export function areFiniteNumbers(...values: unknown[]): boolean {
  return values.every(Number.isFinite)
}

export function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value))
}
