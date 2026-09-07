import test from 'node:test'
import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { createRequire } from 'node:module'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { promisify } from 'node:util'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { assertValidPng, parsePngDimensions } from './helpers.mjs'
import { Renderer } from './smoke.test.part-001.mjs'
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


test('reused shader and texture caches observe live source and pixel changes', () => {
  const pixels = new Uint8Array([255, 0, 0, 255])
  const texture = new THREE.DataTexture(pixels, 1, 1)
  const source = `uniform float opacity;
    uniform sampler2D tDiffuse;
    varying vec2 vUv;
    void main() { vec4 texel = texture2D(tDiffuse, vUv); gl_FragColor = opacity * texel; }`
  const material = new THREE.ShaderMaterial({
    fragmentShader: source,
    uniforms: { tDiffuse: { value: texture }, opacity: { value: 1 } },
  })
  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10)
  camera.position.z = 2
  const renderer = new Renderer()
  const render = () => renderer.render(scene, camera, { width: 16, height: 16, format: 'rgba' })
  const center = (8 * 16 + 8) * 4
  assert.ok(render()[center] > 200)
  // Three.js texture version changes must invalidate both adapter and native caches.
  pixels.set([0, 255, 0, 255])
  texture.needsUpdate = true
  const changed = render()
  assert.ok(changed[center + 1] > changed[center] + 50)
  material.fragmentShader = 'void main() { gl_FragColor = vec4(1.0); }'
  assert.throws(render, /not supported|custom material/)
  material.fragmentShader = source
  assert.ok(render()[center + 1] > 200)
})
