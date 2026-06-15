#!/usr/bin/env node
import fs from 'node:fs/promises'
import * as THREE from 'three'
import {
  loadGltfFromFile,
  render,
} from '../packages/renderer/dist/index.js'

const [inputPath, outputPath = 'render.png'] = process.argv.slice(2)

if (!inputPath) {
  console.error('Usage: node examples/render-gltf.mjs <model.gltf|model.glb> [render.png]')
  process.exit(1)
}

const width = positiveInteger(process.env.WIDTH, 1024)
const height = positiveInteger(process.env.HEIGHT, width)

const gltf = await loadGltfFromFile(inputPath)
const scene = gltf.scene
const camera = findRenderableCamera(scene, width / height) ?? frameSceneCamera(scene, width / height)

scene.updateMatrixWorld(true)
camera.updateMatrixWorld(true)

const image = render(scene, camera, { width, height })
await fs.writeFile(outputPath, image)
console.log(`Rendered ${inputPath} to ${outputPath} (${width}x${height})`)

function findRenderableCamera(scene, aspect) {
  let camera = null
  scene.traverse((object) => {
    if (!camera && object.isCamera === true && object.isArrayCamera !== true && object.type !== 'CubeCamera') {
      camera = object
    }
  })
  if (!camera) return null
  if (camera.isPerspectiveCamera === true && Number.isFinite(aspect)) {
    camera.aspect = aspect
    camera.updateProjectionMatrix()
  }
  return camera
}

function frameSceneCamera(scene, aspect) {
  const box = new THREE.Box3().setFromObject(scene)
  const sphere = box.getBoundingSphere(new THREE.Sphere())
  const center = Number.isFinite(sphere.center.x) ? sphere.center : new THREE.Vector3()
  const radius = Number.isFinite(sphere.radius) && sphere.radius > 0 ? sphere.radius : 1

  const camera = new THREE.PerspectiveCamera(45, aspect, Math.max(0.01, radius / 1000), radius * 100)
  const fov = THREE.MathUtils.degToRad(camera.fov)
  const distance = radius / Math.sin(fov / 2)
  camera.position.set(
    center.x + distance * 0.55,
    center.y + distance * 0.35,
    center.z + distance,
  )
  camera.lookAt(center)
  camera.updateProjectionMatrix()
  return camera
}

function positiveInteger(value, fallback) {
  const parsed = Number.parseInt(value ?? '', 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback
}
