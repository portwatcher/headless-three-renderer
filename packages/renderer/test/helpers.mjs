import assert from 'node:assert/strict'

const PNG_MAGIC = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])

/**
 * Parse the width/height from a PNG buffer's IHDR chunk without decoding pixels.
 */
export function parsePngDimensions(buffer) {
  assert.ok(Buffer.isBuffer(buffer), 'expected a Buffer')
  assert.ok(buffer.length >= 24, `PNG too short: ${buffer.length} bytes`)
  assert.ok(buffer.subarray(0, 8).equals(PNG_MAGIC), 'missing PNG magic header')
  // IHDR starts at byte 8 (length=4, type=4), width/height at bytes 16-23
  return {
    width: buffer.readUInt32BE(16),
    height: buffer.readUInt32BE(20),
  }
}

/**
 * Assert that the buffer is a valid PNG of the given size.
 */
export function assertValidPng(buffer, { width, height }) {
  const dims = parsePngDimensions(buffer)
  assert.equal(dims.width, width, `PNG width mismatch`)
  assert.equal(dims.height, height, `PNG height mismatch`)
}

/**
 * Compute the mean RGB of a raw RGBA8 buffer.
 */
export function meanRgba(rgba) {
  assert.ok(Buffer.isBuffer(rgba))
  assert.equal(rgba.length % 4, 0, 'RGBA buffer length must be divisible by 4')
  let r = 0
  let g = 0
  let b = 0
  let a = 0
  const pixels = rgba.length / 4
  for (let i = 0; i < rgba.length; i += 4) {
    r += rgba[i]
    g += rgba[i + 1]
    b += rgba[i + 2]
    a += rgba[i + 3]
  }
  return {
    r: r / pixels,
    g: g / pixels,
    b: b / pixels,
    a: a / pixels,
  }
}

/**
 * Ratio of non-background pixels (any channel differing from the given color by > tolerance).
 */
export function nonBackgroundRatio(rgba, bg, tolerance = 2) {
  const pixels = rgba.length / 4
  let count = 0
  for (let i = 0; i < rgba.length; i += 4) {
    if (
      Math.abs(rgba[i] - bg[0]) > tolerance ||
      Math.abs(rgba[i + 1] - bg[1]) > tolerance ||
      Math.abs(rgba[i + 2] - bg[2]) > tolerance
    ) {
      count++
    }
  }
  return count / pixels
}

/**
 * Compute a short stable digest of a buffer (SHA-1, hex).
 */
export async function digest(buffer) {
  const { createHash } = await import('node:crypto')
  return createHash('sha1').update(buffer).digest('hex')
}
