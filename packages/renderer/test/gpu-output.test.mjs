import assert from 'node:assert/strict'
import test from 'node:test'
import * as THREE from 'three'

import { Renderer } from '../dist/index.js'

test('GPU output capability and lease lifetime contract', () => {
  const renderer = new Renderer()
  const capabilities = renderer.getGpuOutputCapabilities()
  assert.equal(typeof capabilities.backend, 'string')
  assert.equal(typeof capabilities.texture.supported, 'boolean')
  assert.equal(capabilities.texture.synchronization, 'submission-complete')
  assert.equal(capabilities.texture.scope, 'same-renderer-device')
  assert.equal(capabilities.texture.format, 'rgba8unorm')
  assert.equal(capabilities.texture.usage, 'copy-dst|copy-src|texture-binding')
  assert.equal(capabilities.texture.layout, 'backend-managed-copy-dst')
  assert.equal(typeof capabilities.dmaBuf.supported, 'boolean')

  if (!capabilities.texture.supported) {
    assert.equal(typeof capabilities.texture.reason, 'string')
    return
  }

  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial()))
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(2.5, 1.8, 3.2)
  camera.lookAt(0, 0, 0)

  const frame = renderer.renderGpuFrame(scene, camera, { width: 8, height: 6 })
  assert.equal(frame.width, 8)
  assert.equal(frame.height, 6)
  assert.equal(frame.format, 'rgba8unorm')
  assert.equal(frame.completed, true)
  assert.equal(frame.scope, 'same-renderer-device')
  assert.equal(frame.released, false)
  assert.equal(typeof frame.nativeHandle(), 'bigint')
  assert.notEqual(frame.nativeHandle(), 0n)
  if (!capabilities.dmaBuf.supported) {
    assert.throws(() => frame.exportDmaBuf(), /DMA-BUF export is unsupported/)
  }

  frame.release()
  frame.release()
  assert.equal(frame.released, true)
  assert.throws(() => frame.nativeHandle(), /lease has been released/)
  assert.throws(() => frame.exportDmaBuf(), /lease has been released/)
})
