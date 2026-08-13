#!/usr/bin/env node

import { pbkdf2 } from 'node:crypto'
import { performance } from 'node:perf_hooks'
import * as THREE from 'three'
import { Renderer } from '../dist/index.js'

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
const i420Capability = renderer.getGpuOutputCapabilities().mediaFormats
  .find(({ format }) => format === 'i420-planes')

const legacy = await benchmark('legacy-rgba-readback', async () => ({
  bytes: renderer.render(scene, camera, renderOptions).length,
}))

let i420 = null
if (i420Capability?.supported) {
  const target = Buffer.alloc(width * height * 1.5)
  const pool = renderer.createGpuFramePool({ width, height, capacity: 1, format: 'i420-planes' })
  i420 = await benchmark('pooled-packed-i420-readback', async () => {
    const frame = await pool.renderI420(scene, camera, { toneMapping: THREE.NoToneMapping }, target)
    return { bytes: frame.data.length, sameBuffer: frame.data === target }
  })
  i420.pool = pool.stats()
  pool.close()
}

const workerProbe = await libuvProbe()
console.log(JSON.stringify({
  environment: {
    platform: process.platform,
    arch: process.arch,
    node: process.version,
    backend: renderer.getGpuOutputCapabilities().backend,
    uvThreadpoolSize: process.env.UV_THREADPOOL_SIZE ?? 'default',
  },
  width,
  height,
  frames,
  warmup,
  legacy,
  i420,
  i420UnsupportedReason: i420 ? undefined : i420Capability?.reason,
  workerProbe,
}, null, 2))

async function benchmark(name, run) {
  for (let index = 0; index < warmup; index += 1) await run()
  const samples = []
  let result
  for (let index = 0; index < frames; index += 1) {
    const start = performance.now()
    result = await run()
    samples.push(performance.now() - start)
  }
  samples.sort((a, b) => a - b)
  return {
    name,
    meanMs: round(samples.reduce((sum, value) => sum + value, 0) / samples.length),
    p50Ms: round(percentile(samples, 0.5)),
    p95Ms: round(percentile(samples, 0.95)),
    logicalReadbackBytes: result.bytes,
    bytesPerPixel: result.bytes / (width * height),
    callerBufferReused: result.sameBuffer ?? false,
  }
}

async function libuvProbe() {
  const pbkdf2Ms = () => new Promise((resolve, reject) => {
    const start = performance.now()
    pbkdf2('media', 'probe', 1, 16, 'sha256', (error) => {
      if (error) reject(error)
      else resolve(performance.now() - start)
    })
  })
  const baselineMs = await pbkdf2Ms()
  if (!i420Capability?.supported) return { baselineMs: round(baselineMs) }
  const pool = renderer.createGpuFramePool({ width, height, capacity: 1, format: 'i420-planes' })
  const pending = pool.renderI420(scene, camera, { toneMapping: THREE.NoToneMapping })
  const contendedMs = await pbkdf2Ms()
  await pending
  pool.close()
  return {
    baselineMs: round(baselineMs),
    contendedMs: round(contendedMs),
    addedDelayMs: round(contendedMs - baselineMs),
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
