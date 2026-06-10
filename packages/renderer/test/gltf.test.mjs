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

  const scene = toRenderableScene(gltf.scene)
  scene.background = new THREE.Color(0.02, 0.02, 0.03)
  scene.add(new THREE.AmbientLight(0xffffff, 0.6))
  const light = new THREE.DirectionalLight(0xffffff, 1.4)
  light.position.set(2, 3, 4)
  scene.add(light)
  scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(scene, camera, { width: 96, height: 96, format: 'rgba' })
  assert.equal(rgba.length, 96 * 96 * 4)
  assert.ok(nonBackgroundRatio(rgba, [5, 5, 8], 3) > 0.04, 'glTF triangle should render visible pixels')

  const mean = meanRgba(rgba)
  assert.ok(mean.b > mean.r, `loaded blue PBR material should contribute blue output (${mean.b} vs ${mean.r})`)
  assert.ok(mean.a > 240, `loaded glTF output should be opaque (${mean.a})`)
})

async function loadGltfFixture(filePath) {
  installLocalFileFetch()

  const root = path.dirname(filePath)
  const bytes = await readFile(filePath)
  const manager = new THREE.LoadingManager()
  manager.addHandler(/\.(png|jpe?g|webp)$/i, createEncodedImageTextureLoader(root))

  const loader = new GLTFLoader(manager)
  const baseUrl = pathToFileURL(`${root}${path.sep}`).href
  return await new Promise((resolve, reject) => {
    loader.parse(arrayBufferView(bytes), baseUrl, resolve, reject)
  })
}

function arrayBufferView(buffer) {
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength)
}

function toRenderableScene(root) {
  if (root?.isScene === true) return root
  const scene = new THREE.Scene()
  scene.add(root)
  return scene
}

function findFirst(root, predicate) {
  let match = null
  root.traverse((object) => {
    if (!match && predicate(object)) match = object
  })
  return match
}
