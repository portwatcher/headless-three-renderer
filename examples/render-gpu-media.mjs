import * as THREE from 'three'
import { Renderer } from '@headless-three/renderer'

const renderer = new Renderer()
const capabilities = renderer.getGpuOutputCapabilities()
const nv12 = capabilities.mediaFormats.find(({ format }) => format === 'nv12-planes')
if (!nv12?.supported) throw new Error(nv12?.reason ?? 'nv12-planes is unsupported')

const pool = renderer.createGpuFramePool({
  width: 1280,
  height: 720,
  capacity: 3,
  format: 'nv12-planes',
  overflow: 'drop-newest',
})
const scene = new THREE.Scene()
scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshStandardMaterial()))
const camera = new THREE.PerspectiveCamera(45, 1280 / 720, 0.01, 100)
camera.position.set(2.5, 1.8, 3.2)
camera.lookAt(0, 0, 0)

const frame = await pool.render(scene, camera)
if (frame) {
  console.log({ format: frame.format, planes: frame.planes, stats: pool.stats() })
  // planeHandle() is only for a same-device integration that obeys the state
  // restoration contract documented in docs/gpu-native-output.md.
  frame.release()
}
pool.close()
