#!/usr/bin/env node

import { createRequire } from 'node:module'
import { performance } from 'node:perf_hooks'
import * as THREE from 'three'
import { Renderer } from '../dist/index.js'

const require = createRequire(import.meta.url)
const wrtc = require(process.env.WRTC_MODULE_PATH ?? '@roamhq/wrtc')
const width = numberArg('width', 1920)
const height = numberArg('height', 1080)
const frames = numberArg('frames', 60)
const warmup = numberArg('warmup', 10)
if (width % 2 || height % 2) throw new Error('width and height must be even')

const renderer = new Renderer()
const scene = new THREE.Scene()
scene.background = new THREE.Color(0.18, 0.42, 0.76)
const camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 100)
const renderOptions = { width, height, format: 'rgba', toneMapping: THREE.NoToneMapping }
const source = new wrtc.nonstandard.RTCVideoSource()
const track = source.createTrack()
const legacyI420 = new Uint8Array(width * height * 1.5)
const target = Buffer.alloc(width * height * 1.5)
const pool = renderer.createGpuFramePool({ width, height, capacity: 1, format: 'i420-planes' })

const legacy = await benchmark('legacy-rgba-libyuv-onFrame', () => {
  const rgba = renderer.render(scene, camera, renderOptions)
  wrtc.nonstandard.rgbaToI420(
    { width, height, data: rgba },
    { width, height, data: legacyI420 },
  )
  source.onFrame({ width, height, data: legacyI420 })
})
const packed = await benchmark('pooled-packed-i420-onFrame', async () => {
  const frame = await pool.renderI420(scene, camera, { toneMapping: THREE.NoToneMapping }, target)
  source.onFrame(frame)
})

track.stop()
pool.close()
const output = JSON.stringify({
  environment: { platform: process.platform, arch: process.arch, node: process.version },
  width,
  height,
  frames,
  warmup,
  legacy,
  packed,
  speedup: Number((legacy.meanMs / packed.meanMs).toFixed(3)),
}, null, 2)
// @roamhq/wrtc 0.10.0 may abort during Node 24 finalization after a track was
// stopped. Exit after flushing the benchmark so its upstream teardown bug does
// not discard valid results.
process.stdout.write(`${output}\n`, () => process.exit(0))

async function benchmark(name, run) {
  for (let index = 0; index < warmup; index += 1) await run()
  const samples = []
  for (let index = 0; index < frames; index += 1) {
    const start = performance.now()
    await run()
    samples.push(performance.now() - start)
  }
  samples.sort((a, b) => a - b)
  return {
    name,
    meanMs: round(samples.reduce((sum, value) => sum + value, 0) / samples.length),
    p50Ms: round(percentile(samples, 0.5)),
    p95Ms: round(percentile(samples, 0.95)),
    inputBytesPerPixel: name.startsWith('legacy') ? 4 : 1.5,
  }
}

function numberArg(name, fallback) {
  const prefix = `--${name}=`
  const value = process.argv.slice(2).find((arg) => arg.startsWith(prefix))?.slice(prefix.length)
  if (value === undefined) return fallback
  const parsed = Number(value)
  if (!Number.isInteger(parsed) || parsed <= 0) throw new Error(`${prefix}<positive integer> expected`)
  return parsed
}

function percentile(sorted, fraction) {
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * fraction))]
}

function round(value) {
  return Number(value.toFixed(3))
}
