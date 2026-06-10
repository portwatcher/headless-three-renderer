import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'

const {
  Renderer,
  createEncodedImageTextureLoader,
  installLocalFileFetch,
} = pkg

const FIXTURE_DIR = fileURLToPath(new URL('./fixtures/', import.meta.url))
const SIMPLE_TRIANGLE = path.join(FIXTURE_DIR, 'simple-triangle.gltf')
const TEXTURED_QUAD = path.join(FIXTURE_DIR, 'textured-quad.gltf')

test('committed glTF fixture loads through GLTFLoader and renders', async () => {
  const gltf = await loadGltfFixture(SIMPLE_TRIANGLE)

  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'fixture should load a mesh')
  assert.equal(mesh.geometry.getAttribute('position').count, 3)
  assert.equal(mesh.material.isMeshStandardMaterial, true)

  const camera = gltf.cameras[0]
  assert.ok(camera, 'fixture should load a camera')
  camera.aspect = 1
  camera.updateProjectionMatrix()

  const scene = gltf.scene
  scene.add(new THREE.AmbientLight(0xffffff, 0.6))
  const light = new THREE.DirectionalLight(0xffffff, 1.4)
  light.position.set(2, 3, 4)
  scene.add(light)
  scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [0.02, 0.02, 0.03],
  })
  assert.equal(rgba.length, 96 * 96 * 4)
  assert.ok(nonBackgroundRatio(rgba, [5, 5, 8], 3) > 0.04, 'glTF triangle should render visible pixels')

  const mean = meanRgba(rgba)
  assert.ok(mean.b > mean.r, `loaded blue PBR material should contribute blue output (${mean.b} vs ${mean.r})`)
  assert.ok(mean.a > 240, `loaded glTF output should be opaque (${mean.a})`)
})

test('committed textured glTF fixture loads data URI image and renders texture', async () => {
  const gltf = await loadGltfFixture(TEXTURED_QUAD)

  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'textured fixture should load a mesh')
  assert.ok(mesh.material.map?.isTexture, 'textured fixture should load a base color texture')
  assert.ok(Buffer.isBuffer(mesh.material.map.image), 'encoded image helper should expose the data URI PNG as a Buffer')

  const camera = gltf.cameras[0]
  assert.ok(camera, 'textured fixture should load a camera')
  camera.aspect = 1
  camera.updateProjectionMatrix()

  const scene = gltf.scene
  scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })
  assert.equal(rgba.length, 96 * 96 * 4)
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.25, 'textured quad should render visible pixels')

  const left = meanRegion(rgba, 96, 96, 24, 36, 42, 60)
  const right = meanRegion(rgba, 96, 96, 54, 36, 72, 60)
  assert.ok(left.r > left.g + 80, `left half should sample the red data URI texel (${left.r} vs ${left.g})`)
  assert.ok(right.g > right.r + 80, `right half should sample the green data URI texel (${right.g} vs ${right.r})`)
})

async function loadGltfFixture(filePath) {
  installLocalFileFetch()

  const root = path.dirname(filePath)
  const bytes = await readFile(filePath)
  const manager = new THREE.LoadingManager()
  const encodedImages = createEncodedImageTextureLoader(root)
  manager.addHandler(/^blob:/i, encodedImages)
  manager.addHandler(/^data:image\/(?:png|jpe?g|webp)/i, encodedImages)
  manager.addHandler(/\.(png|jpe?g|webp)$/i, encodedImages)

  const loader = new GLTFLoader(manager)
  const baseUrl = pathToFileURL(`${root}${path.sep}`).href
  return await new Promise((resolve, reject) => {
    loader.parse(arrayBufferView(bytes), baseUrl, resolve, reject)
  })
}

function arrayBufferView(buffer) {
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength)
}

function findFirst(root, predicate) {
  let match = null
  root.traverse((object) => {
    if (!match && predicate(object)) match = object
  })
  return match
}

function meanRegion(rgba, width, _height, x0, y0, x1, y1) {
  let r = 0
  let g = 0
  let b = 0
  let a = 0
  let count = 0
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const i = (y * width + x) * 4
      r += rgba[i]
      g += rgba[i + 1]
      b += rgba[i + 2]
      a += rgba[i + 3]
      count++
    }
  }
  return { r: r / count, g: g / count, b: b / count, a: a / count }
}
