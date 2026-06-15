import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'

const {
  Renderer,
  loadGltfFromFile,
} = pkg

const FIXTURE_DIR = fileURLToPath(new URL('./fixtures/', import.meta.url))
const SIMPLE_TRIANGLE = path.join(FIXTURE_DIR, 'simple-triangle.gltf')
const TEXTURED_QUAD = path.join(FIXTURE_DIR, 'textured-quad.gltf')

test('committed glTF fixture loads through GLTFLoader and renders', async () => {
  let configured = false
  const gltf = await loadGltfFixture(SIMPLE_TRIANGLE, {
    configureLoader(loader) {
      configured = typeof loader.parse === 'function'
    },
  })
  assert.equal(configured, true, 'loadGltfFromFile should expose the loader before parsing')

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

  assertTexturedQuadLoadsEncodedMap(gltf, 'textured fixture')
  assertTexturedQuadRendersTexture(gltf, 'textured quad')
})

test('loadGltfFromFile resolves external glTF image files from the model directory', async () => {
  const source = JSON.parse(await readFile(TEXTURED_QUAD, 'utf8'))
  const imageUri = source.images[0].uri
  const encodedImage = imageUri.slice(imageUri.indexOf(',') + 1)
  const tmp = await mkdtemp(path.join(os.tmpdir(), 'headless-three-gltf-image-'))
  try {
    const textureDir = path.join(tmp, 'textures')
    await mkdir(textureDir)
    await writeFile(path.join(textureDir, 'quad.png'), Buffer.from(encodedImage, 'base64'))
    source.images[0].uri = 'textures/quad.png'
    const modelPath = path.join(tmp, 'external-image.gltf')
    await writeFile(modelPath, JSON.stringify(source))

    const gltf = await loadGltfFixture(modelPath)
    assertTexturedQuadLoadsEncodedMap(gltf, 'external-image fixture')
    assertTexturedQuadRendersTexture(gltf, 'external-image quad')
  } finally {
    await rm(tmp, { recursive: true, force: true })
  }
})

test('loadGltfFromFile resolves external glTF buffers from the model directory', async () => {
  const source = JSON.parse(await readFile(SIMPLE_TRIANGLE, 'utf8'))
  const bufferUri = source.buffers[0].uri
  const encodedBuffer = bufferUri.slice(bufferUri.indexOf(',') + 1)
  const tmp = await mkdtemp(path.join(os.tmpdir(), 'headless-three-gltf-'))
  try {
    await writeFile(path.join(tmp, 'triangle.bin'), Buffer.from(encodedBuffer, 'base64'))
    source.buffers[0].uri = 'triangle.bin'
    const modelPath = path.join(tmp, 'external-buffer.gltf')
    await writeFile(modelPath, JSON.stringify(source))

    const gltf = await loadGltfFixture(modelPath)
    const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
    assert.ok(mesh, 'external-buffer fixture should load a mesh')

    const camera = gltf.cameras[0]
    assert.ok(camera, 'external-buffer fixture should load a camera')
    camera.aspect = 1
    camera.updateProjectionMatrix()
    gltf.scene.add(new THREE.AmbientLight(0xffffff, 1))
    gltf.scene.updateMatrixWorld(true)
    camera.updateMatrixWorld(true)

    const rgba = new Renderer().render(gltf.scene, camera, {
      width: 64,
      height: 64,
      format: 'rgba',
      background: [0, 0, 0],
    })
    assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.04, 'external buffer glTF should render visible pixels')
  } finally {
    await rm(tmp, { recursive: true, force: true })
  }
})

function assertTexturedQuadLoadsEncodedMap(gltf, label) {
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, `${label} should load a mesh`)
  assert.ok(mesh.material.map?.isTexture, `${label} should load a base color texture`)
  assert.ok(Buffer.isBuffer(mesh.material.map.image), 'encoded image helper should expose the PNG as a Buffer')
}

function assertTexturedQuadRendersTexture(gltf, label) {
  const camera = gltf.cameras[0]
  assert.ok(camera, `${label} should load a camera`)
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
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.25, `${label} should render visible pixels`)

  const left = meanRegion(rgba, 96, 96, 24, 36, 42, 60)
  const right = meanRegion(rgba, 96, 96, 54, 36, 72, 60)
  assert.ok(left.r > left.g + 80, `left half should sample the red texture texel (${left.r} vs ${left.g})`)
  assert.ok(right.g > right.r + 80, `right half should sample the green texture texel (${right.g} vs ${right.r})`)
}

async function loadGltfFixture(filePath, options) {
  return await loadGltfFromFile(filePath, options)
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
