import { pbkdf2 } from 'node:crypto'
import { performance } from 'node:perf_hooks'
import * as THREE from 'three'
import { Renderer } from '../../dist/index.js'

const size = 1024
const renderer = new Renderer()
const scene = new THREE.Scene()
scene.background = new THREE.Color(0.2, 0.5, 0.8)
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
const pool = renderer.createGpuFramePool({
  width: size,
  height: size,
  capacity: 4,
  format: 'i420-planes',
})
const targets = Array.from({ length: 4 }, () => Buffer.alloc(size * size * 1.5))
const started = performance.now()
const frames = Promise.all(targets.map((target) => (
  pool.renderI420(scene, camera, { toneMapping: THREE.NoToneMapping }, target)
)))
const probe = new Promise((resolve, reject) => {
  pbkdf2('media', 'probe', 1, 16, 'sha256', (error) => {
    if (error) reject(error)
    else resolve('pbkdf2')
  })
})
const winner = await Promise.race([frames.then(() => 'frames'), probe])
const completed = await frames
pool.close()
console.log(JSON.stringify({
  winner,
  completedFrames: completed.filter(Boolean).length,
  totalMs: performance.now() - started,
}))
