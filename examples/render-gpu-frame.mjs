import * as THREE from 'three'
import { Renderer } from '@headless-three/renderer'

const renderer = new Renderer()
const capabilities = renderer.getGpuOutputCapabilities()
if (!capabilities.texture.supported) {
  throw new Error(capabilities.texture.reason)
}

const scene = new THREE.Scene()
scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial()))
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
camera.position.set(2.5, 1.8, 3.2)
camera.lookAt(0, 0, 0)

const frame = renderer.renderGpuFrame(scene, camera, { width: 512, height: 512 })
try {
  console.log({
    backend: frame.backend,
    handle: frame.nativeHandle(),
    width: frame.width,
    height: frame.height,
    format: frame.format,
  })
} finally {
  frame.release()
}
