#!/usr/bin/env node

import { performance } from 'node:perf_hooks'
import * as THREE from 'three'
import { Renderer } from '../dist/index.js'

const options = parseArgs(process.argv.slice(2))
if (options.help) {
  printHelp()
  process.exit(0)
}

const renderer = new Renderer()
const { scene, camera, meshes, materials } = createScene(options.meshes)
const renderOptions = {
  width: options.width,
  height: options.height,
  format: 'rgba',
  outputColorSpace: THREE.LinearSRGBColorSpace,
}

let checksum = 0
for (let frame = 0; frame < options.warmup; frame += 1) {
  updateScene(scene, meshes, materials, frame)
  const rgba = renderer.render(scene, camera, renderOptions)
  checksum = updateChecksum(checksum, rgba)
}

const startMemory = process.memoryUsage()
const frameTimes = []
const totalStart = performance.now()
for (let frame = 0; frame < options.frames; frame += 1) {
  updateScene(scene, meshes, materials, frame + options.warmup)
  const start = performance.now()
  const rgba = renderer.render(scene, camera, renderOptions)
  frameTimes.push(performance.now() - start)
  checksum = updateChecksum(checksum, rgba)
}
const totalMs = performance.now() - totalStart
const endMemory = process.memoryUsage()

const summary = summarize({
  options,
  frameTimes,
  totalMs,
  checksum,
  startMemory,
  endMemory,
})

if (options.json) {
  console.log(JSON.stringify(summary, null, 2))
} else {
  printSummary(summary)
}

function parseArgs(args) {
  const parsed = {
    frames: 60,
    warmup: 5,
    meshes: 256,
    width: 128,
    height: 128,
    json: false,
    help: false,
  }

  for (const arg of args) {
    if (arg === '--') {
      continue
    }
    if (arg === '--help' || arg === '-h') {
      parsed.help = true
      continue
    }
    if (arg === '--json') {
      parsed.json = true
      continue
    }

    const match = arg.match(/^--([a-z-]+)=(.+)$/)
    if (!match) {
      throw new Error(`Unknown argument "${arg}". Use --help for usage.`)
    }
    const [, name, rawValue] = match
    if (!['frames', 'warmup', 'meshes', 'width', 'height'].includes(name)) {
      throw new Error(`Unknown option "--${name}". Use --help for usage.`)
    }
    parsed[name] = name === 'warmup'
      ? nonNegativeInteger(rawValue, `--${name}`)
      : positiveInteger(rawValue, `--${name}`)
  }

  return parsed
}

function positiveInteger(value, label) {
  const parsed = Number(value)
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive integer.`)
  }
  return parsed
}

function nonNegativeInteger(value, label) {
  const parsed = Number(value)
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`${label} must be a non-negative integer.`)
  }
  return parsed
}

function createScene(meshCount) {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.025, 0.035)

  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100)
  camera.position.set(0, 0, 14)
  camera.lookAt(0, 0, 0)

  const light = new THREE.DirectionalLight(0xffffff, 2.5)
  light.position.set(3, 5, 8)
  scene.add(light)
  scene.add(new THREE.AmbientLight(0x404050, 1.5))

  const textureData = new Uint8Array([
    255, 255, 255, 255,
    180, 220, 255, 255,
    255, 180, 220, 255,
    220, 255, 180, 255,
  ])
  const texture = new THREE.DataTexture(textureData, 2, 2, THREE.RGBAFormat)
  texture.colorSpace = THREE.LinearSRGBColorSpace
  texture.needsUpdate = true

  const geometry = new THREE.BoxGeometry(0.6, 0.6, 0.6)
  const materials = []
  const meshes = []
  const columns = Math.ceil(Math.sqrt(meshCount))
  const spacing = 0.78

  for (let index = 0; index < meshCount; index += 1) {
    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color().setHSL((index % 31) / 31, 0.7, 0.55),
      roughness: 0.7,
      metalness: 0.05,
      map: texture,
    })
    const mesh = new THREE.Mesh(geometry, material)
    const x = index % columns
    const y = Math.floor(index / columns)
    mesh.position.set(
      (x - (columns - 1) / 2) * spacing,
      (y - (columns - 1) / 2) * spacing,
      0,
    )
    mesh.userData.profileBaseX = mesh.position.x
    mesh.userData.profileBaseY = mesh.position.y
    scene.add(mesh)
    meshes.push(mesh)
    materials.push(material)
  }

  return { scene, camera, meshes, materials }
}

function updateScene(scene, meshes, materials, frame) {
  const time = frame * 0.071
  for (let index = 0; index < meshes.length; index += 1) {
    const mesh = meshes[index]
    const material = materials[index]
    const phase = time + index * 0.137
    mesh.position.x = mesh.userData.profileBaseX + Math.sin(phase) * 0.08
    mesh.position.y = mesh.userData.profileBaseY + Math.cos(phase * 0.73) * 0.08
    mesh.rotation.x = phase * 0.35
    mesh.rotation.y = phase * 0.21
    mesh.scale.setScalar(0.85 + Math.sin(phase * 0.47) * 0.08)
    material.color.setHSL((phase * 0.03 + index * 0.011) % 1, 0.72, 0.55)
    material.roughness = 0.55 + Math.sin(phase * 0.19) * 0.25
  }
  scene.updateMatrixWorld(true)
}

function updateChecksum(previous, rgba) {
  let checksum = previous
  const step = Math.max(4, Math.floor(rgba.length / 64))
  for (let offset = 0; offset < rgba.length; offset += step) {
    checksum = (checksum + rgba[offset]) >>> 0
  }
  return checksum
}

function summarize({ options, frameTimes, totalMs, checksum, startMemory, endMemory }) {
  const sorted = [...frameTimes].sort((a, b) => a - b)
  const meanMs = frameTimes.reduce((sum, value) => sum + value, 0) / frameTimes.length
  return {
    frames: options.frames,
    warmupFrames: options.warmup,
    meshes: options.meshes,
    width: options.width,
    height: options.height,
    totalMs: round(totalMs),
    meanFrameMs: round(meanMs),
    medianFrameMs: round(percentile(sorted, 0.5)),
    p95FrameMs: round(percentile(sorted, 0.95)),
    minFrameMs: round(sorted[0]),
    maxFrameMs: round(sorted[sorted.length - 1]),
    rssDeltaMb: round((endMemory.rss - startMemory.rss) / (1024 * 1024)),
    heapUsedDeltaMb: round((endMemory.heapUsed - startMemory.heapUsed) / (1024 * 1024)),
    checksum,
  }
}

function percentile(sortedValues, fraction) {
  const rawIndex = (sortedValues.length - 1) * fraction
  const lowerIndex = Math.floor(rawIndex)
  const upperIndex = Math.ceil(rawIndex)
  if (lowerIndex === upperIndex) return sortedValues[lowerIndex]
  const lower = sortedValues[lowerIndex]
  const upper = sortedValues[upperIndex]
  return lower + (upper - lower) * (rawIndex - lowerIndex)
}

function round(value) {
  return Math.round(value * 100) / 100
}

function printSummary(summary) {
  console.log('Animated scene profile')
  console.log(`frames=${summary.frames} warmup=${summary.warmupFrames} meshes=${summary.meshes} size=${summary.width}x${summary.height}`)
  console.log(`total=${summary.totalMs}ms mean=${summary.meanFrameMs}ms median=${summary.medianFrameMs}ms p95=${summary.p95FrameMs}ms`)
  console.log(`min=${summary.minFrameMs}ms max=${summary.maxFrameMs}ms rssDelta=${summary.rssDeltaMb}MB heapDelta=${summary.heapUsedDeltaMb}MB checksum=${summary.checksum}`)
}

function printHelp() {
  console.log(`Usage: pnpm -C packages/renderer run profile:animated -- [options]

Options:
  --frames=N   Measured frames to render. Default: 60
  --warmup=N   Warmup frames rendered before measurement. Default: 5
  --meshes=N   Animated mesh count. Default: 256
  --width=N    Output width in pixels. Default: 128
  --height=N   Output height in pixels. Default: 128
  --json       Print machine-readable JSON.
  --help       Show this message.
`)
}
