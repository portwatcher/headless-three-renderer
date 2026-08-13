import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import test from 'node:test'
import { promisify } from 'node:util'
import { fileURLToPath } from 'node:url'
import * as THREE from 'three'

import { Renderer } from '../dist/index.js'

const execFileAsync = promisify(execFile)

function camera() {
  return new THREE.PerspectiveCamera(45, 1, 0.01, 100)
}

function patternScene() {
  const scene = new THREE.Scene()
  const pixels = new Uint8Array([
    255, 0, 0, 255, 0, 255, 0, 255,
    0, 0, 255, 255, 255, 255, 255, 255,
  ])
  const texture = new THREE.DataTexture(pixels, 2, 2, THREE.RGBAFormat, THREE.UnsignedByteType)
  texture.colorSpace = THREE.SRGBColorSpace
  texture.magFilter = THREE.NearestFilter
  texture.minFilter = THREE.NearestFilter
  texture.needsUpdate = true
  scene.background = texture
  return scene
}

function solidScene(r, g, b) {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(r, g, b)
  return scene
}

function code8(value) {
  return Math.max(0, Math.min(255, Math.round(value)))
}

function code10(value) {
  return Math.max(0, Math.min(1023, Math.round(value)))
}

function convertReference(rgba, width, height, format) {
  const tenBit = format === 'p010-planes'
  const y = new Uint16Array(width * height)
  const uv = new Uint16Array(width * height / 2)
  const rgbAt = (x, row) => {
    const offset = (row * width + x) * 4
    return [rgba[offset] / 255, rgba[offset + 1] / 255, rgba[offset + 2] / 255]
  }
  const luma = ([r, g, b]) => tenBit
    ? code10(64 + 876 * (0.2126 * r + 0.7152 * g + 0.0722 * b))
    : code8(16 + 219 * (0.2126 * r + 0.7152 * g + 0.0722 * b))
  const chroma = ([r, g, b]) => {
    const cb = -0.114572 * r - 0.385428 * g + 0.5 * b
    const cr = 0.5 * r - 0.454153 * g - 0.045847 * b
    return tenBit
      ? [code10(512 + 896 * cb), code10(512 + 896 * cr)]
      : [code8(128 + 224 * cb), code8(128 + 224 * cr)]
  }
  for (let row = 0; row < height; row += 1) {
    for (let x = 0; x < width; x += 1) y[row * width + x] = luma(rgbAt(x, row))
  }
  for (let row = 0; row < height; row += 2) {
    for (let x = 0; x < width; x += 2) {
      const samples = [rgbAt(x, row), rgbAt(x + 1, row), rgbAt(x, row + 1), rgbAt(x + 1, row + 1)]
      const average = [0, 1, 2].map((channel) => samples.reduce((sum, rgb) => sum + rgb[channel], 0) / 4)
      const [u, v] = chroma(average)
      const offset = (row / 2 * (width / 2) + x / 2) * 2
      uv[offset] = u
      uv[offset + 1] = v
    }
  }
  return { y, uv }
}

function convertI420Reference(rgba, width, height) {
  const y = new Uint8Array(width * height)
  const u = new Uint8Array(width * height / 4)
  const v = new Uint8Array(width * height / 4)
  const rgbAt = (x, row) => {
    const offset = (row * width + x) * 4
    return [rgba[offset] / 255, rgba[offset + 1] / 255, rgba[offset + 2] / 255]
  }
  const luma = ([r, g, b]) => code8(16 + 219 * (0.299 * r + 0.587 * g + 0.114 * b))
  const chroma = ([r, g, b]) => [
    code8(128 + 224 * (-0.168736 * r - 0.331264 * g + 0.5 * b)),
    code8(128 + 224 * (0.5 * r - 0.418688 * g - 0.081312 * b)),
  ]
  for (let row = 0; row < height; row += 1) {
    for (let x = 0; x < width; x += 1) y[row * width + x] = luma(rgbAt(x, row))
  }
  for (let row = 0; row < height; row += 2) {
    for (let x = 0; x < width; x += 2) {
      const samples = [rgbAt(x, row), rgbAt(x + 1, row), rgbAt(x, row + 1), rgbAt(x + 1, row + 1)]
      const average = [0, 1, 2].map((channel) => samples.reduce((sum, rgb) => sum + rgb[channel], 0) / 4)
      const index = row / 2 * (width / 2) + x / 2
      ;[u[index], v[index]] = chroma(average)
    }
  }
  return Buffer.concat([y, u, v])
}

function unpackPlane(plane, tenBit) {
  if (!tenBit) return Uint16Array.from(plane.data)
  const view = new DataView(plane.data.buffer, plane.data.byteOffset, plane.data.byteLength)
  const result = new Uint16Array(plane.data.byteLength / 2)
  for (let index = 0; index < result.length; index += 1) {
    const word = view.getUint16(index * 2, true)
    assert.equal(word & 63, 0, 'P010 low six bits must be zero')
    result[index] = word >> 6
  }
  return result
}

function assertNear(actual, expected, tolerance = 1) {
  assert.equal(actual.length, expected.length)
  for (let index = 0; index < actual.length; index += 1) {
    assert.ok(
      Math.abs(actual[index] - expected[index]) <= tolerance,
      `sample ${index}: expected ${expected[index]}, received ${actual[index]}`,
    )
  }
}

test('media capabilities distinguish writable planes from encoder-native surfaces', () => {
  const renderer = new Renderer()
  const capabilities = renderer.getGpuOutputCapabilities()
  const rgba = capabilities.mediaFormats.find(({ format }) => format === 'rgba8unorm')
  const nv12 = capabilities.mediaFormats.find(({ format }) => format === 'nv12-planes')
  const p010 = capabilities.mediaFormats.find(({ format }) => format === 'p010-planes')
  const i420 = capabilities.mediaFormats.find(({ format }) => format === 'i420-planes')
  assert.deepEqual(rgba?.planeFormats, ['rgba8unorm'])
  assert.equal(nv12?.storage, 'separate-textures')
  assert.deepEqual(nv12?.planeFormats, ['r8unorm-y', 'rg8unorm-uv'])
  assert.equal(typeof p010?.supported, 'boolean')
  assert.equal(i420?.colorMatrix, 'bt601')
  assert.equal(i420?.storage, 'separate-textures+packed-cpu-fallback')
  assert.equal(capabilities.encoderSurface.supported, false)
  assert.match(capabilities.encoderSurface.reason, /wgpu|encoder|surface|texture/i)
  assert.equal(typeof capabilities.encoderSurface.prerequisitesReady, 'boolean')
  assert.equal(typeof capabilities.encoderSurface.prerequisites, 'string')
  assert.equal(capabilities.dmaBuf.supported, false)
})

test('pool rendering is asynchronous and lets the Node event loop advance', async () => {
  const renderer = new Renderer()
  const pool = renderer.createGpuFramePool({ width: 128, height: 128 })
  const pending = pool.render(patternScene(), camera())
  let timerRan = false
  await new Promise((resolve) => setTimeout(() => { timerRan = true; resolve() }, 0))
  assert.equal(timerRan, true)
  const frame = await pending
  assert.equal(frame?.ready, true)
  frame?.release()
  pool.close()
})

test('GPU completion waits do not occupy the shared libuv worker pool', async () => {
  const fixture = fileURLToPath(new URL('./fixtures/media-worker-probe.mjs', import.meta.url))
  const { stdout } = await execFileAsync(process.execPath, [fixture], {
    env: { ...process.env, UV_THREADPOOL_SIZE: '1' },
    timeout: 30_000,
  })
  const result = JSON.parse(stdout)
  assert.equal(result.winner, 'pbkdf2')
  assert.equal(result.completedFrames, 4)
})

test('pool reserves synchronously and bounds burst work before libuv', async () => {
  const renderer = new Renderer()
  const pool = renderer.createGpuFramePool({
    width: 8,
    height: 8,
    capacity: 2,
    overflow: 'drop-newest',
  })
  const renders = Array.from({ length: 30 }, () => pool.render(solidScene(0.2, 0.4, 0.8), camera()))
  assert.equal(pool.stats().submitted, 2)
  assert.equal(pool.stats().inFlight, 2)
  assert.equal(pool.stats().dropped, 28)
  const results = await Promise.all(renders)
  assert.equal(results.filter(Boolean).length, 2)
  for (const frame of results) frame?.release()
  await new Promise((resolve) => setTimeout(resolve, 10))
  assert.equal(pool.stats().submitted, 2, 'dropped frames must not resurrect after slots free')
  assert.equal(pool.stats().available, 2)
  pool.close()
})

test('error overflow, release lifecycle, reuse, and shutdown are deterministic', async () => {
  const renderer = new Renderer()
  const pool = renderer.createGpuFramePool({ width: 4, height: 4, capacity: 1 })
  const first = await pool.render(solidScene(0, 0, 0), camera())
  await assert.rejects(() => pool.render(solidScene(1, 1, 1), camera()), /pool is exhausted/)
  first?.release()
  first?.release()
  const second = await pool.render(solidScene(1, 1, 1), camera())
  assert.equal(pool.stats().reused, 1)
  pool.close()
  assert.equal(pool.stats().closed, true)
  await assert.rejects(() => pool.render(solidScene(0, 0, 0), camera()), /pool is closed/)
  second?.release()
  assert.equal(pool.stats().available, 1)
})

test('readback state stays current and raw handle use requires external completion', async () => {
  const renderer = new Renderer()
  if (!renderer.getGpuOutputCapabilities().texture.supported) return
  const pool = renderer.createGpuFramePool({ width: 4, height: 4, capacity: 1 })
  const frame = await pool.render(solidScene(1, 0, 0), camera())
  assert.ok(frame)
  await frame.readPlanes()
  const readbackState = frame.planes[0].expectedStateBeforeUse
  if (frame.backend === 'vulkan') assert.match(readbackState, /TRANSFER_SRC_OPTIMAL/)
  else if (frame.backend === 'dx12') assert.match(readbackState, /COPY_SOURCE/)
  else assert.match(readbackState, /copy-source|untracked/i)
  assert.equal(frame.planes[0].requiredStateOnRelease, readbackState)
  assert.equal(typeof frame.planeHandle(0), 'bigint')
  assert.equal(frame.planes[0].expectedStateBeforeUse, readbackState)
  assert.throws(() => frame.release(), /external GPU use is pending/)
  assert.equal(frame.released, false)
  frame.completeExternalUse()
  frame.release()
  assert.equal(frame.released, true)
  assert.throws(() => frame.planeHandle(0), /lease has been released/)
  assert.throws(() => frame.completeExternalUse(), /lease has been released/)

  const reused = await pool.render(solidScene(0, 1, 0), camera())
  assert.ok(reused)
  const [rgba] = await reused.readPlanes()
  assert.ok(rgba.data.some((value) => value !== 0), 'tracked reuse must render into the prior copy-source texture')
  reused.release()
  assert.equal(pool.stats().reused, 1)
  pool.close()
})

for (const format of ['nv12-planes', 'p010-planes']) {
  test(`${format} GPU planes match a CPU BT.709 limited-range reference`, async (t) => {
    const renderer = new Renderer()
    const capability = renderer.getGpuOutputCapabilities().mediaFormats.find((item) => item.format === format)
    if (!capability?.supported) return t.skip(capability?.reason ?? `${format} unsupported`)
    const width = 6
    const height = 4
    const scene = patternScene()
    const options = { width, height, format: 'rgba', toneMapping: THREE.NoToneMapping }
    const rgba = renderer.render(scene, camera(), options)
    const reference = convertReference(rgba, width, height, format)
    const pool = renderer.createGpuFramePool({ width, height, format })
    const frame = await pool.render(scene, camera(), { toneMapping: THREE.NoToneMapping })
    assert.ok(frame)
    assert.equal(frame.format, format)
    assert.equal(frame.planes.length, 2)
    assert.equal(frame.planes[0].width, width)
    assert.equal(frame.planes[1].width, width / 2)
    assert.throws(() => frame.exportDmaBuf(), /DMA-BUF export is unsupported/)
    const [y, uv] = await frame.readPlanes()
    assertNear(unpackPlane(y, format === 'p010-planes'), reference.y)
    assertNear(unpackPlane(uv, format === 'p010-planes'), reference.uv)
    frame.release()
    assert.throws(() => frame.readPlanes(), /lease has been released/)
    pool.close()
  })
}

test('hundreds of frames reuse a fixed allocation set', async () => {
  const renderer = new Renderer()
  const pool = renderer.createGpuFramePool({ width: 2, height: 2, capacity: 3 })
  const scene = solidScene(0.1, 0.2, 0.3)
  for (let index = 0; index < 400; index += 1) {
    const frame = await pool.render(scene, camera())
    frame?.release()
  }
  const stats = pool.stats()
  assert.equal(stats.submitted, 400)
  assert.equal(stats.completed, 400)
  assert.equal(stats.allocations, 3)
  assert.equal(stats.reused, 399)
  assert.equal(stats.inFlight, 0)
  assert.equal(stats.available, 3)
  pool.close()
})

test('packed I420 fallback is wrtc-shaped BT.601 data with 1.5 B/pixel readback', async (t) => {
  const renderer = new Renderer()
  const capability = renderer.getGpuOutputCapabilities().mediaFormats.find(({ format }) => format === 'i420-planes')
  if (!capability?.supported) return t.skip(capability?.reason ?? 'i420-planes unsupported')
  const width = 6
  const height = 4
  const scene = patternScene()
  const options = { width, height, format: 'rgba', toneMapping: THREE.NoToneMapping }
  const rgba = renderer.render(scene, camera(), options)
  const reference = convertI420Reference(rgba, width, height)
  const target = Buffer.alloc(width * height * 1.5)
  const pool = renderer.createGpuFramePool({ width, height, capacity: 1, format: 'i420-planes' })
  const frame = await pool.renderI420(scene, camera(), { toneMapping: THREE.NoToneMapping }, target)
  assert.ok(frame)
  assert.strictEqual(frame.data, target)
  assert.equal(frame.data.length, reference.length)
  assert.equal(frame.byteLength, reference.length)
  assert.equal(frame.gpuReadbackBytes, Math.ceil(reference.length / 4) * 4)
  assert.deepEqual(frame.strides, [width, width / 2, width / 2])
  assert.deepEqual(frame.offsets, [0, width * height, width * height + width * height / 4])
  assert.equal(frame.colorMatrix, 'bt601')
  assert.equal(frame.colorRange, 'limited')
  assertNear(frame.data, reference)
  assert.equal(pool.stats().available, 1, 'CPU fallback releases its slot before resolving')

  const oversized = Buffer.alloc(target.length + 1).subarray(1)
  await assert.rejects(
    () => pool.renderI420(scene, camera(), {}, oversized),
    /exact standalone Buffer/,
  )
  pool.close()
})

test('I420 caller buffers can be reused without renderer allocations', async (t) => {
  const renderer = new Renderer()
  const capability = renderer.getGpuOutputCapabilities().mediaFormats.find(({ format }) => format === 'i420-planes')
  if (!capability?.supported) return t.skip(capability?.reason ?? 'i420-planes unsupported')
  const width = 8
  const height = 8
  const target = Buffer.alloc(width * height * 1.5)
  const pool = renderer.createGpuFramePool({ width, height, capacity: 2, format: 'i420-planes' })
  const pending = pool.renderI420(solidScene(0.1, 0.25, 0.5), camera(), {}, target)
  await assert.rejects(
    () => pool.renderI420(solidScene(0.2, 0.25, 0.5), camera(), {}, target),
    /target is already in use/,
  )
  assert.strictEqual((await pending)?.data, target)
  for (let index = 0; index < 120; index += 1) {
    const frame = await pool.renderI420(solidScene(index / 120, 0.25, 0.5), camera(), {}, target)
    assert.strictEqual(frame?.data, target)
  }
  const stats = pool.stats()
  assert.equal(stats.submitted, 121)
  assert.equal(stats.allocations, 2)
  assert.equal(stats.reused, 120)
  assert.equal(stats.inFlight, 0)
  pool.close()
})
