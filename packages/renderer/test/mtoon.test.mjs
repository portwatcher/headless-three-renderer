import test from 'node:test'
import assert from 'node:assert/strict'
import { fileURLToPath } from 'node:url'
import * as THREE from 'three'
import { MToonMaterial, VRMLoaderPlugin } from '@pixiv/three-vrm'
import pkg from '../dist/index.js'

const { Renderer, loadVrmFromFile } = pkg
const camera = () => {
  const result = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10)
  result.position.z = 2
  result.updateMatrixWorld(true)
  return result
}
const render = (renderer, scene) => {
  scene.updateMatrixWorld(true)
  return renderer.render(scene, camera(), { width: 32, height: 32, format: 'rgba', toneMapping: THREE.NoToneMapping, background: [0, 0, 0, 0] })
}
const center = (frame) => [...frame.subarray((16 * 32 + 16) * 4, (16 * 32 + 16) * 4 + 4)]
const surface = (material) => {
  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(1.5, 1.5), material))
  scene.add(new THREE.AmbientLight(0xffffff, 1))
  return scene
}

test('real Pixiv MToon preserves texture alpha and live expression color/opacity', () => {
  const renderer = new Renderer()
  const map = new THREE.DataTexture(new Uint8Array([255, 255, 255, 255]), 1, 1)
  map.needsUpdate = true
  const material = new MToonMaterial({ map, color: new THREE.Color(1, 0, 0), transparent: true })
  const scene = surface(material)
  try {
    const red = center(render(renderer, scene))
    assert.ok(red[0] > 150 && red[1] < 50 && red[2] < 50)
    // VRM expression bindings keep and mutate the original material and its uniforms.
    material.color.setRGB(0, 1, 0)
    const green = center(render(renderer, scene))
    assert.ok(green[1] > 150 && green[0] < 50 && green[2] < 50)
    material.opacity = 0
    assert.equal(center(render(renderer, scene))[3], 0)
    material.opacity = 1
    material.alphaTest = 0.5
    map.image.data[3] = 0
    map.needsUpdate = true
    assert.equal(center(render(renderer, scene))[3], 0)
    assert.equal(scene.children[0].material, material)
    assert.equal(material.isShaderMaterial, true)
  } finally {
    renderer.dispose()
    material.dispose()
    map.dispose()
  }
})

test('MToon uses ignoreVertexColor and skips unextruded outline groups without mutation', () => {
  const renderer = new Renderer()
  const material = new MToonMaterial()
  const outline = new MToonMaterial({ isOutline: true, side: THREE.DoubleSide })
  const scene = surface(material)
  const mesh = scene.children[0]
  const geometry = mesh.geometry
  geometry.setAttribute('color', new THREE.Float32BufferAttribute([
    1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
  ], 3))
  try {
    material.ignoreVertexColor = false
    assert.ok(center(render(renderer, scene))[1] < 50)
    material.ignoreVertexColor = true
    const plain = render(renderer, scene)
    assert.ok(center(plain)[1] > 150)
    mesh.material = [material, outline]
    geometry.clearGroups()
    geometry.addGroup(0, 6, 0)
    geometry.addGroup(0, 6, 1)
    assert.deepEqual(render(renderer, scene), plain)
    assert.equal(outline.visible, true)
  } finally {
    renderer.dispose()
    material.dispose()
    outline.dispose()
    geometry.dispose()
  }
})

test('real VRMLoaderPlugin renders the textured, skinned Seed-san MToon avatar', async () => {
  const path = fileURLToPath(new URL('./fixtures/vrm-specification/Seed-san/vrm/Seed-san.vrm', import.meta.url))
  const gltf = await loadVrmFromFile(path, { VRMLoaderPlugin })
  const vrm = gltf.userData.vrm
  assert.ok(vrm.materials.some((material) => material.isMToonMaterial))
  vrm.update(0)
  const scene = new THREE.Scene()
  scene.add(vrm.scene, new THREE.AmbientLight(0xffffff, 1))
  scene.updateMatrixWorld(true)
  const view = new THREE.PerspectiveCamera(35, 1, 0.01, 100)
  view.position.set(0, 1.3, 3)
  view.lookAt(0, 1, 0)
  view.updateMatrixWorld(true)
  const renderer = new Renderer()
  try {
    const frame = renderer.render(scene, view, { width: 128, height: 128, format: 'rgba', toneMapping: THREE.NoToneMapping, background: [0, 0, 0, 0] })
    let visible = 0
    for (let i = 3; i < frame.length; i += 4) {
      if (frame[i] > 0) visible++
    }
    assert.ok(visible > 500, `expected a visible avatar, got ${visible} pixels`)
    assert.ok(visible < 128 * 128 / 2, 'background should remain transparent')
  } finally {
    renderer.dispose()
  }
})

test('unrecognized ShaderMaterial still fails explicitly', () => {
  const renderer = new Renderer()
  try {
    assert.throws(() => render(renderer, surface(new THREE.ShaderMaterial())), /ShaderMaterial.*not supported directly/)
  } finally {
    renderer.dispose()
  }
})
