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

const colorNear = (actual, expected, tolerance = 2) => {
  actual.slice(0, 3).forEach((value, i) => assert.ok(
    Math.abs(value - expected[i]) <= tolerance,
    `channel ${i}: expected ${expected}, got ${actual}`,
  ))
}
const solidTexture = (rgba) => {
  const texture = new THREE.DataTexture(new Uint8Array(rgba), 1, 1)
  texture.needsUpdate = true
  return texture
}

test('MToon uses the browser Lambert normalization for ambient light and stays dark without light', () => {
  const renderer = new Renderer()
  const material = new MToonMaterial({ color: new THREE.Color(1, 0, 0) })
  const scene = surface(material)
  try {
    // Three.js WebGL: linear 1 / PI becomes sRGB 153, not full-strength ambient.
    colorNear(center(render(renderer, scene)), [153, 0, 0])
    scene.remove(scene.children[1])
    colorNear(center(render(renderer, scene)), [0, 0, 0])
  } finally { renderer.dispose(); material.dispose() }
})

test('MToon shades with authored color and texture, and observes live shade and shift changes', () => {
  const renderer = new Renderer()
  const material = new MToonMaterial({ color: new THREE.Color(1, 0, 0), shadeColorFactor: new THREE.Color(0, 0, 1) })
  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(1.5, 1.5), material))
  const light = new THREE.DirectionalLight(0xffffff, 1)
  light.position.z = -2
  scene.add(light)
  const shadeMap = solidTexture([255, 128, 255, 255])
  const shiftMap = solidTexture([255, 0, 0, 255])
  try {
    colorNear(center(render(renderer, scene)), [0, 0, 153])
    material.shadeColorFactor.setRGB(0, 1, 0)
    colorNear(center(render(renderer, scene)), [0, 153, 0])
    material.shadeMultiplyTexture = shadeMap
    colorNear(center(render(renderer, scene)), [0, 111, 0])
    shadeMap.colorSpace = THREE.SRGBColorSpace
    colorNear(center(render(renderer, scene)), [0, 74, 0])
    shadeMap.colorSpace = THREE.LinearSRGBColorSpace
    material.shadingShiftTexture = shiftMap
    material.shadingShiftTextureScale = 2
    colorNear(center(render(renderer, scene)), [153, 0, 0])
    material.shadingShiftTexture = null
    light.position.z = 2
    colorNear(center(render(renderer, scene)), [153, 0, 0])
    material.shadingShiftFactor = -2
    colorNear(center(render(renderer, scene)), [0, 111, 0])
    material.shadingShiftFactor = Number.NaN
    assert.throws(() => render(renderer, scene), /shadingShiftFactor.*finite/)
  } finally { renderer.dispose(); material.dispose(); shadeMap.dispose(); shiftMap.dispose() }
})

test('MToon preserves matcap, rim texture and live rim expression factors', () => {
  const renderer = new Renderer()
  const material = new MToonMaterial({ color: new THREE.Color(0, 0, 0), parametricRimColorFactor: new THREE.Color(0, 0, 1), parametricRimFresnelPowerFactor: 0 })
  const scene = surface(material)
  const rimMap = solidTexture([255, 255, 128, 255])
  const matcap = solidTexture([128, 0, 0, 255])
  try {
    colorNear(center(render(renderer, scene)), [0, 0, 153])
    material.rimMultiplyTexture = rimMap
    colorNear(center(render(renderer, scene)), [0, 0, 111])
    rimMap.colorSpace = THREE.SRGBColorSpace
    colorNear(center(render(renderer, scene)), [0, 0, 74])
    rimMap.colorSpace = THREE.LinearSRGBColorSpace
    material.parametricRimColorFactor.setRGB(0, 0, 0)
    material.matcapTexture = matcap
    colorNear(center(render(renderer, scene)), [111, 0, 0])
    material.matcapFactor.setRGB(0, 0, 0)
    colorNear(center(render(renderer, scene)), [0, 0, 0])
  } finally { renderer.dispose(); material.dispose(); rimMap.dispose(); matcap.dispose() }
})

test('MToon world outlines extrude back faces and sample the authored width texture', () => {
  const renderer = new Renderer()
  const material = new MToonMaterial({ color: new THREE.Color(1, 0, 0) })
  const widthMap = solidTexture([255, 255, 255, 255])
  const outline = new MToonMaterial({ isOutline: true, side: THREE.BackSide, outlineWidthMode: 'worldCoordinates', outlineWidthFactor: 0.12, outlineColorFactor: new THREE.Color(0, 1, 0), outlineLightingMixFactor: 0, outlineWidthMultiplyTexture: widthMap })
  const geometry = new THREE.SphereGeometry(0.6, 32, 24)
  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(geometry, material), new THREE.AmbientLight(0xffffff, 1))
  const visible = (frame) => { let n = 0; for (let i = 3; i < frame.length; i += 4) if (frame[i] > 0) n++; return n }
  try {
    const base = render(renderer, scene)
    scene.add(new THREE.Mesh(geometry, outline))
    const expanded = render(renderer, scene)
    assert.ok(visible(expanded) > visible(base) + 70)
    colorNear(center(expanded), [153, 0, 0])
    widthMap.image.data[1] = 0
    widthMap.needsUpdate = true
    assert.deepEqual(render(renderer, scene), base)
  } finally { renderer.dispose(); material.dispose(); outline.dispose(); widthMap.dispose(); geometry.dispose() }
})

test('skinned meshes retain their world transform, matching Three.js deformed vertices', () => {
  const renderer = new Renderer()
  const material = new THREE.MeshBasicMaterial({ color: 0xff0000 })
  const geometry = new THREE.PlaneGeometry(0.6, 0.8)
  const count = geometry.attributes.position.count
  const weights = new Float32Array(count * 4)
  for (let i = 0; i < count; i++) weights[i * 4] = 1
  geometry.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(new Uint16Array(count * 4), 4))
  geometry.setAttribute('skinWeight', new THREE.Float32BufferAttribute(weights, 4))
  const bone = new THREE.Bone()
  const mesh = new THREE.SkinnedMesh(geometry, material)
  const parent = new THREE.Group()
  parent.position.set(0.35, -0.15, 0)
  parent.rotation.z = 0.4
  parent.scale.setScalar(1.2)
  parent.add(mesh)
  mesh.add(bone)
  parent.updateMatrixWorld(true)
  mesh.bind(new THREE.Skeleton([bone]))
  bone.rotation.z = -0.25
  const scene = new THREE.Scene()
  scene.add(parent)
  scene.updateMatrixWorld(true)
  const expectedPositions = []
  for (let i = 0; i < count; i++) expectedPositions.push(...mesh.getVertexPosition(i, new THREE.Vector3()).applyMatrix4(mesh.matrixWorld).toArray())
  const baked = geometry.clone()
  baked.setAttribute('position', new THREE.Float32BufferAttribute(expectedPositions, 3))
  const expectedScene = new THREE.Scene()
  expectedScene.add(new THREE.Mesh(baked, material))
  try {
    const actual = render(renderer, scene)
    const expected = render(renderer, expectedScene)
    let differing = 0
    for (let i = 0; i < actual.length; i += 4) if (Math.abs(actual[i] - expected[i]) > 2) differing++
    assert.ok(differing <= 4, `world transform changed ${differing} pixels`)
  } finally { renderer.dispose(); material.dispose(); geometry.dispose(); baked.dispose(); mesh.skeleton.dispose() }
})
