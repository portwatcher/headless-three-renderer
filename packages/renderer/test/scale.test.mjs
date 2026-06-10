import test from 'node:test'
import assert from 'node:assert/strict'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'

const { Renderer } = pkg

const SIZE = 96
const BACKGROUND = [5, 5, 5]

function makeCamera() {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(2.8, 2.2, 4.2)
  camera.lookAt(0, 0, 0)
  return camera
}

function makeTexture(index) {
  const size = 4
  const data = new Uint8Array(size * size * 4)
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const i = (y * size + x) * 4
      data[i] = (48 + index * 29 + x * 37) % 256
      data[i + 1] = (96 + index * 17 + y * 41) % 256
      data[i + 2] = (144 + index * 23 + (x + y) * 19) % 256
      data[i + 3] = 255
    }
  }
  const texture = new THREE.DataTexture(data, size, size, THREE.RGBAFormat)
  texture.colorSpace = THREE.SRGBColorSpace
  texture.needsUpdate = true
  return texture
}

function addSupportedLightBudget(scene) {
  scene.add(new THREE.AmbientLight(0xffffff, 0.08))
  for (let i = 0; i < 16; i += 1) {
    const angle = (i / 16) * Math.PI * 2
    const light = new THREE.PointLight(new THREE.Color().setHSL(i / 16, 0.55, 0.65), 0.42, 6, 1.6)
    light.position.set(Math.cos(angle) * 2.2, 1.2 + (i % 4) * 0.28, Math.sin(angle) * 2.2)
    scene.add(light)
  }
}

test('large scene budget renders many meshes, textures, and supported lights', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.02)

  const geometry = new THREE.BoxGeometry(0.14, 0.14, 0.14)
  const textures = Array.from({ length: 8 }, (_, i) => makeTexture(i))
  const materials = textures.map((map, i) => new THREE.MeshStandardMaterial({
    map,
    roughness: 0.48 + (i % 3) * 0.12,
    metalness: i % 2 === 0 ? 0.05 : 0.18,
  }))

  for (let row = 0; row < 8; row += 1) {
    for (let col = 0; col < 10; col += 1) {
      const mesh = new THREE.Mesh(geometry, materials[(row * 10 + col) % materials.length])
      mesh.position.set((col - 4.5) * 0.22, (row - 3.5) * 0.2, Math.sin(row * 0.8 + col * 0.45) * 0.2)
      mesh.rotation.set(row * 0.07, col * 0.05, (row + col) * 0.03)
      scene.add(mesh)
    }
  }
  addSupportedLightBudget(scene)

  const rgba = new Renderer().render(scene, makeCamera(), { width: SIZE, height: SIZE, format: 'rgba' })
  assert.equal(rgba.length, SIZE * SIZE * 4)
  const ratio = nonBackgroundRatio(rgba, BACKGROUND, 6)
  assert.ok(ratio > 0.04, `scale scene should render visible non-background pixels (${ratio})`)
  const mean = meanRgba(rgba)
  assert.ok(mean.a > 240, `scale scene should remain opaque on average (${mean.a})`)
})

test('more than 16 visible non-ambient lights fail clearly', () => {
  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshStandardMaterial({ color: 0xffffff })))
  for (let i = 0; i < 17; i += 1) {
    const light = new THREE.PointLight(0xffffff, 0.2)
    light.position.set((i % 5) - 2, 2, Math.floor(i / 5) - 1)
    scene.add(light)
  }

  assert.throws(
    () => new Renderer().render(scene, makeCamera(), { width: 32, height: 32, format: 'rgba' }),
    /More than 16 visible non-ambient lights/i,
  )
})
