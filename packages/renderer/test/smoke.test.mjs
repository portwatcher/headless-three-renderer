import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { pathToFileURL } from 'node:url'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { assertValidPng, parsePngDimensions } from './helpers.mjs'

const {
  Renderer,
  createEncodedImageTextureLoader,
  installLocalFileFetch,
  render,
  resolveLocalAssetPath,
} = pkg

test('module exports Renderer class and render function', () => {
  assert.equal(typeof Renderer, 'function')
  assert.equal(typeof render, 'function')
  assert.equal(typeof createEncodedImageTextureLoader, 'function')
  assert.equal(typeof installLocalFileFetch, 'function')
  assert.equal(typeof resolveLocalAssetPath, 'function')
})

test('Node loader helpers expose encoded image buffers and local file fetch', async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'headless-three-loader-'))
  try {
    const imagePath = path.join(dir, 'tex.png')
    const imageBytes = Buffer.from(
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR42mP8z8BQDwAFgwJ/l6g+WQAAAABJRU5ErkJggg==',
      'base64',
    )
    await writeFile(imagePath, imageBytes)

    const loader = createEncodedImageTextureLoader(dir)
    const texture = await new Promise((resolve, reject) => {
      loader.load('tex.png', resolve, undefined, reject)
    })

    assert.equal(texture.isTexture, true)
    assert.deepEqual(Buffer.from(texture.image), imageBytes)
    assert.equal(texture.source.data, texture.image)
    assert.equal(resolveLocalAssetPath('tex.png', dir), imagePath)

    installLocalFileFetch()
    const response = await fetch(pathToFileURL(imagePath).href)
    assert.deepEqual(Buffer.from(await response.arrayBuffer()), imageBytes)
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
})

test('renders a simple scene and returns a PNG buffer of the requested size', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.1, 0.1, 0.1)
  scene.add(
    new THREE.Mesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshBasicMaterial({ color: 0xff0000 }),
    ),
  )

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(2.5, 1.8, 3.2)
  camera.lookAt(0, 0, 0)

  const r = new Renderer()
  const buf = r.render(scene, camera, { width: 256, height: 256 })

  assert.ok(Buffer.isBuffer(buf), 'output should be a Buffer')
  assert.ok(buf.length > 0, 'output should be non-empty')
  assertValidPng(buf, { width: 256, height: 256 })
})

test('renderer is reusable across multiple calls', () => {
  const scene = new THREE.Scene()
  scene.add(
    new THREE.Mesh(new THREE.SphereGeometry(1, 16, 16), new THREE.MeshBasicMaterial({ color: 0x00ff00 })),
  )
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const r = new Renderer()
  for (let i = 0; i < 3; i++) {
    const buf = r.render(scene, camera, { width: 128, height: 128 })
    const { width, height } = parsePngDimensions(buf)
    assert.equal(width, 128)
    assert.equal(height, 128)
  }
})

test('top-level render() function works without a Renderer instance', () => {
  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial({ color: 0x0000ff })))
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const buf = render(scene, camera, { width: 64, height: 64 })
  assertValidPng(buf, { width: 64, height: 64 })
})

test('different sizes produce correctly sized outputs', () => {
  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial()))
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const r = new Renderer()
  for (const [w, h] of [
    [100, 100],
    [320, 240],
    [512, 256],
  ]) {
    const buf = r.render(scene, camera, { width: w, height: h })
    assertValidPng(buf, { width: w, height: h })
  }
})
