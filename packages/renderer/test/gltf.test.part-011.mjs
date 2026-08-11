import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'
import { Renderer, SAMPLE_ASSET_CUBE_VISIBILITY, SAMPLE_ASSET_LANTERN, SAMPLE_ASSET_LIGHT_VISIBILITY, SAMPLE_ASSET_PLAYSET_LIGHT_TEST, SAMPLE_ASSET_POINT_LIGHT_INTENSITY_TEST } from './gltf.test.part-001.mjs'
import { assertVectorClose, isEffectivelyVisible, loadGltfFixture, meanRegion, pngDimensions } from './gltf.test.part-028.mjs'
test('committed Khronos glTF Sample Assets Lantern fixture loads multi-mesh textured PBR asset', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_LANTERN, 'utf8'))
  assert.equal(source.asset.generator, 'glTF Tools for Unity')
  assert.deepEqual(source.buffers, [{ uri: 'Lantern.bin', byteLength: 231324 }])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'Lantern_baseColor.png',
    'Lantern_roughnessMetallic.png',
    'Lantern_normal.png',
    'Lantern_emissive.png',
  ])
  assert.equal(source.materials.length, 1)
  assert.equal(source.materials[0].name, 'LanternPost_Mat')
  assert.deepEqual(source.materials[0].pbrMetallicRoughness, {
    baseColorTexture: { index: 0 },
    metallicRoughnessTexture: { index: 1 },
  })
  assert.deepEqual(source.materials[0].normalTexture, { index: 2 })
  assert.deepEqual(source.materials[0].emissiveFactor, [1, 1, 1])
  assert.deepEqual(source.materials[0].emissiveTexture, { index: 3 })
  assert.deepEqual(source.meshes.map((mesh) => mesh.name), [
    'LanternPole_Body',
    'LanternPole_Chain',
    'LanternPole_Lantern',
  ])
  assert.deepEqual(source.nodes.map((node) => node.name), [
    'LanternPole_Body',
    'LanternPole_Chain',
    'LanternPole_Lantern',
    'Lantern',
  ])
  assert.deepEqual(source.nodes[3].children, [0, 1, 2])
  assert.deepEqual(source.nodes[3].rotation, [0, 1, 0, 0])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_LANTERN)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'LanternPole_Body',
    'LanternPole_Chain',
    'LanternPole_Lantern',
  ])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [926, 756, 2463])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [926, 756, 2463])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('tangent')?.count), [926, 756, 2463])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [926, 756, 2463])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [2616, 3744, 9822])
  assert.deepEqual(meshes.map((mesh) => mesh.position.toArray()), [
    [-3.82315421, 13.01603, 0],
    [-9.582001, 21.0378723, 0],
    [-9.582007, 18.0091515, 0],
  ])

  const material = meshes[0].material
  assert.ok(meshes.every((mesh) => mesh.material === material), 'Lantern meshes should share one PBR material')
  assert.equal(material.isMeshStandardMaterial, true)
  assert.equal(material.name, 'LanternPost_Mat')
  assert.equal(material.metalness, 1)
  assert.equal(material.roughness, 1)
  assert.deepEqual(material.emissive.toArray(), [1, 1, 1])

  const assertLoadedTexture = (texture, name, colorSpace) => {
    assert.ok(texture?.isTexture, `${name} should load a texture`)
    assert.equal(texture.name, name)
    assert.equal(Buffer.isBuffer(texture.image), true, `${name} should load as an encoded Buffer`)
    assert.deepEqual(pngDimensions(texture.image), [2048, 2048])
    assert.equal(texture.wrapS, THREE.RepeatWrapping)
    assert.equal(texture.wrapT, THREE.RepeatWrapping)
    assert.equal(texture.magFilter, THREE.LinearFilter)
    assert.equal(texture.minFilter, THREE.LinearMipmapLinearFilter)
    assert.equal(texture.colorSpace, colorSpace)
    assert.equal(texture.flipY, false)
  }

  assertLoadedTexture(material.map, 'Lantern_baseColor.png', THREE.SRGBColorSpace)
  assertLoadedTexture(material.metalnessMap, 'Lantern_roughnessMetallic.png', THREE.NoColorSpace)
  assertLoadedTexture(material.normalMap, 'Lantern_normal.png', THREE.NoColorSpace)
  assertLoadedTexture(material.emissiveMap, 'Lantern_emissive.png', THREE.SRGBColorSpace)
  assert.equal(material.metalnessMap, material.roughnessMap, 'Lantern should reuse the packed metallic-roughness texture for roughness')

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.75))
  const light = new THREE.DirectionalLight(0xffffff, 1.8)
  light.position.set(5, 8, 6)
  gltf.scene.add(light)
  const camera = new THREE.OrthographicCamera(
    -size.x / 2 - 2,
    size.x / 2 + 2,
    size.y / 2 + 2,
    -size.y / 2 - 2,
    0.01,
    50,
  )
  camera.position.copy(center).add(new THREE.Vector3(0, 0, 30))
  camera.lookAt(center)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 128,
    height: 128,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.1, 'Lantern should render visible multi-mesh textured PBR geometry')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > mean.g && mean.g > mean.b, `Lantern texture should contribute warm metal and emissive pixels (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets PlaysetLightTest fixture loads textured punctual-light scene', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_PLAYSET_LIGHT_TEST, 'utf8'))
  assert.deepEqual(source.extensionsRequired, ['KHR_lights_punctual', 'KHR_materials_emissive_strength'])
  assert.deepEqual(source.extensionsUsed, ['KHR_lights_punctual', 'KHR_materials_emissive_strength'])
  assert.deepEqual(source.buffers, [{ uri: 'PlaysetLightTest_data.bin', byteLength: 3630672 }])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'PlaysetLightTest_img00.png',
    'PlaysetLightTest_img01.png',
    'PlaysetLightTest_img02.png',
    'PlaysetLightTest_img03.png',
    'PlaysetLightTest_img04.png',
    'PlaysetLightTest_img05.png',
    'PlaysetLightTest_img06.png',
    'PlaysetLightTest_img07.png',
    'PlaysetLightTest_img08.jpg',
    'PlaysetLightTest_img09.jpg',
    'PlaysetLightTest_img10.jpg',
    'PlaysetLightTest_img11.png',
    'PlaysetLightTest_img12.png',
    'PlaysetLightTest_img13.png',
    'PlaysetLightTest_img14.png',
    'PlaysetLightTest_img15.png',
    'PlaysetLightTest_img16.png',
    'PlaysetLightTest_img17.png',
    'PlaysetLightTest_img18.png',
  ])
  assert.deepEqual(source.meshes.map((mesh) => mesh.name), [
    'Mesh_0.001',
    'Mesh_0.002',
    'ProcessedGeometry',
    'Mesh_0',
    'Mesh_0.003',
    'Mesh_0.3065',
  ])
  assert.deepEqual(source.materials.map((material) => material.name), [
    'Material_0.001',
    'Material_0.002',
    'SimplygonCastMaterial.002',
    'Material_0',
    'emissive',
    'Material_0.3065',
  ])
  assert.deepEqual(source.materials[4].extensions, {
    KHR_materials_emissive_strength: {
      emissiveStrength: 1500,
    },
  })
  assert.equal(source.cameras.length, 1)
  assert.deepEqual(source.extensions.KHR_lights_punctual.lights.map((light) => light.type), ['directional', 'point'])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_PLAYSET_LIGHT_TEST)
  const meshes = []
  const lights = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
    if (object.isLight === true) lights.push(object)
  })

  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'abacus',
    'ball',
    'carpet',
    'giraffe',
    'lamp',
    'lamp_emissive',
  ])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [42410, 17056, 112, 17502, 8121, 3940])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [42410, 17056, 112, 17502, 8121, 3940])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [168693, 86346, 330, 88224, 23790, 21696])
  assert.deepEqual(meshes.map((mesh) => mesh.material.name), [
    'Material_0.001',
    'Material_0.002',
    'SimplygonCastMaterial.002',
    'Material_0',
    'Material_0.3065',
    'emissive',
  ])

  const abacus = gltf.scene.getObjectByName('abacus')
  assert.ok(abacus?.isMesh, 'PlaysetLightTest should load abacus mesh')
  assert.equal(abacus.material.map.name, 'Image_0')
  assert.equal(abacus.material.normalMap.name, 'Image_1')
  assert.equal(abacus.material.roughnessMap.name, 'Image_2')
  assert.equal(abacus.material.metalnessMap, abacus.material.roughnessMap)
  assert.equal(abacus.material.aoMap.name, 'Image_3')
  assert.equal(Buffer.isBuffer(abacus.material.map.image), true, 'PlaysetLightTest PNG base-color texture should load as an encoded Buffer')
  assert.deepEqual(pngDimensions(abacus.material.map.image), [1210, 1210])
  assert.equal(abacus.material.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(abacus.material.normalMap.colorSpace, THREE.NoColorSpace)

  const carpet = gltf.scene.getObjectByName('carpet')
  assert.ok(carpet?.isMesh, 'PlaysetLightTest should load carpet mesh')
  assert.equal(carpet.material.map.name, 'Basecolor_0')
  assert.equal(carpet.material.aoMap, carpet.material.roughnessMap, 'carpet should reuse its packed metallic-roughness map for AO')
  assert.equal(Buffer.isBuffer(carpet.material.map.image), true, 'PlaysetLightTest JPEG texture should load as an encoded Buffer')
  assert.equal(carpet.material.map.colorSpace, THREE.SRGBColorSpace)

  const emissive = gltf.scene.getObjectByName('lamp_emissive')
  assert.ok(emissive?.isMesh, 'PlaysetLightTest should load emissive lamp mesh')
  assert.equal(emissive.material.name, 'emissive')
  assert.deepEqual(emissive.material.emissive.toArray(), [1, 1, 1])
  assert.equal(emissive.material.emissiveIntensity, 1500)
  assert.equal(emissive.material.roughness, 0.5)

  assert.equal(gltf.cameras.length, 1)
  const camera = gltf.cameras[0]
  assert.equal(camera.isPerspectiveCamera, true)
  assert.equal(camera.name, 'Camera')
  assert.ok(Math.abs(camera.fov - 22.895193663949186) < 1e-10, `Playset camera should preserve imported yfov (${camera.fov})`)
  assert.equal(camera.near, 0.10000000149011612)
  assert.equal(camera.far, 1000)

  assert.deepEqual(lights.map((light) => light.name), ['light1', 'LEDlight'])
  assert.equal(lights[0].isDirectionalLight, true)
  assert.equal(lights[0].intensity, 512.25)
  assert.equal(lights[1].isPointLight, true)
  assert.ok(Math.abs(lights[1].intensity - 1500.0000192775694) < 1e-6, `LEDlight should preserve imported point-light intensity (${lights[1].intensity})`)

  camera.aspect = 16 / 9
  camera.updateProjectionMatrix()
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 160,
    height: 90,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.5, 'PlaysetLightTest should render visible imported-light scene through its camera')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 120 && mean.g > 120 && mean.b > 120, `PlaysetLightTest imported lights should strongly illuminate the scene (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets PointLightIntensityTest fixture loads KHR_lights_punctual point lights', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_POINT_LIGHT_INTENSITY_TEST)
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_lights_punctual'))
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_unlit'))

  const meshes = []
  const lights = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
    if (object.isLight === true) lights.push(object)
  })
  assert.equal(meshes.length, 13)
  assert.equal(lights.length, 8)
  assert.ok(lights.every((light) => light.isPointLight === true), 'all imported punctual lights should become PointLight objects')
  assert.deepEqual(lights.map((light) => light.name), [
    'Light_4_-_White',
    'Light_1_-_Red',
    'Light_3_-_Blue',
    'Light_2_-_Green',
    'Light_5_-_Gray',
    'Light_6_B',
    'Light_6_G',
    'Light_6_R',
  ])
  assert.deepEqual(lights.map((light) => light.color.toArray()), [
    [1, 1, 1],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0.5, 0.5, 0.5],
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
  ])
  assert.ok(lights.every((light) => light.intensity === 1 && light.distance === 1.125 && light.decay === 2))

  gltf.scene.updateMatrixWorld(true)
  const firstLightPosition = lights[0].getWorldPosition(new THREE.Vector3()).toArray()
  const rgbLightPositions = lights.slice(5).map((light) => light.getWorldPosition(new THREE.Vector3()).toArray())
  assert.deepEqual(firstLightPosition, [0, -2.5, 0.20000000298023224])
  assert.deepEqual(rgbLightPositions, [
    [-2.25, -2.5, 0.20000000298023224],
    [-2.25, -2.5, 0.20000000298023224],
    [-2.25, -2.5, 0.20000000298023224],
  ])

  const label = meshes.find((mesh) => mesh.name === 'Labels')
  assert.equal(label?.material.isMeshBasicMaterial, true)
  assert.equal(Buffer.isBuffer(label.material.map?.image), true, 'point-light label PNG should load as an encoded Buffer')
  assert.equal(label.material.map.name, 'LampColorNames')
  assert.equal(label.material.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(label.material.map.flipY, false)

  const litSurface = meshes.find((mesh) => mesh.material.name === 'Test Surface Material')
  const frame = meshes.find((mesh) => mesh.material.name === 'Frame Material')
  assert.equal(litSurface?.geometry.getAttribute('position')?.count, 24)
  assert.equal(litSurface.geometry.index?.count, 36)
  assert.equal(frame?.geometry.getAttribute('position')?.count, 248)
  assert.equal(frame.geometry.index?.count, 768)

  const camera = new THREE.OrthographicCamera(-4.1, 4.1, 1.4, -4.0, 0.01, 20)
  camera.position.set(0, -1.25, 8)
  camera.lookAt(0, -1.25, 0)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 180,
    height: 120,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.2, 'PointLightIntensityTest should render visible point-light panels')
})

test('committed Khronos glTF Sample Assets LightVisibility fixture applies KHR_node_visibility to imported lights', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_LIGHT_VISIBILITY, 'utf8'))
  assert.deepEqual(source.extensionsRequired, ['KHR_lights_punctual', 'KHR_node_visibility'])
  assert.deepEqual(source.extensionsUsed, ['KHR_animation_pointer', 'KHR_lights_punctual', 'KHR_node_visibility'])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_LIGHT_VISIBILITY)
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_lights_punctual'))
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_node_visibility'))

  const meshes = []
  const lights = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
    if (object.isLight === true) lights.push(object)
  })

  assert.deepEqual(meshes.map((mesh) => mesh.name), ['QuadMeshNode'])
  assert.equal(lights.length, 5)
  assert.ok(lights.every((light) => light.isSpotLight === true), 'all imported punctual lights should become SpotLight objects')
  assert.deepEqual(lights.map((light) => light.name), [
    'InvisibleLight',
    'ChildOfInvisibleShouldBeInvisible',
    'DescendantOfInvisibleShouldBeInvisible',
    'VisibleLight',
    'AnimatedVisibility',
  ])
  assert.deepEqual(lights.map((light) => light.color.toArray()), [
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0.125, 1],
  ])
  assert.deepEqual(lights.map((light) => light.intensity), [5, 5, 5, 5, 6])
  assert.ok(lights.every((light) => light.distance === 5 && light.decay === 2 && light.angle === 0.8 && light.penumbra === 0.1875))

  gltf.scene.updateMatrixWorld(true)
  assertVectorClose(lights[0].getWorldPosition(new THREE.Vector3()).toArray(), [-1.5, 0, 1], 'InvisibleLight world position')
  assertVectorClose(lights[3].getWorldPosition(new THREE.Vector3()).toArray(), [0, 0, 1], 'VisibleLight world position')
  assertVectorClose(lights[4].getWorldPosition(new THREE.Vector3()).toArray(), [1.5, 0, 1], 'AnimatedVisibility world position')

  assert.equal(lights[0].visible, false, 'InvisibleLight should import KHR_node_visibility false')
  assert.equal(lights[1].visible, true, 'child light should keep its own default visible flag')
  assert.equal(lights[2].visible, true, 'descendant light should keep its own default visible flag')
  assert.equal(lights[3].visible, true)
  assert.equal(lights[4].visible, true)
  assert.equal(isEffectivelyVisible(lights[0]), false, 'InvisibleLight should be effectively hidden')
  assert.equal(isEffectivelyVisible(lights[1]), false, 'child light should be hidden by its invisible parent')
  assert.equal(isEffectivelyVisible(lights[2]), false, 'descendant light should be hidden by its invisible ancestor')
  assert.equal(isEffectivelyVisible(lights[3]), true)
  assert.equal(isEffectivelyVisible(lights[4]), true)

  const mesh = meshes[0]
  assert.equal(mesh.geometry.getAttribute('position')?.count, 4)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 4)
  assert.equal(mesh.geometry.index?.count, 6)
  assert.equal(mesh.material.isMeshStandardMaterial, true)
  assert.equal(mesh.material.roughness, 1)
  assert.equal(mesh.material.metalness, 1)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 10)
  camera.position.set(0, -2.4, 2.1)
  camera.lookAt(0, 0, 0)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.3, 'LightVisibility should render visible green and blue spot-light contribution')
  const left = meanRegion(rgba, 96, 96, 12, 42, 34, 74)
  const center = meanRegion(rgba, 96, 96, 37, 42, 59, 74)
  const right = meanRegion(rgba, 96, 96, 62, 42, 84, 74)
  assert.ok(left.r < 10, `invisible red light branch should not tint the left panel red (${left.r}, ${left.g}, ${left.b})`)
  assert.ok(center.g > 80 && center.g > center.r + 80, `visible green light should tint the center panel (${center.r}, ${center.g}, ${center.b})`)
  assert.ok(right.b > 20 && right.b > right.r + 20, `visible animated blue light should tint the right panel (${right.r}, ${right.g}, ${right.b})`)
})

test('committed Khronos glTF Sample Assets CubeVisibility fixture applies KHR_node_visibility to meshes', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_CUBE_VISIBILITY, 'utf8'))
  assert.deepEqual(source.extensionsRequired, ['KHR_node_visibility'])
  assert.deepEqual(source.extensionsUsed, ['KHR_animation_pointer', 'KHR_node_visibility'])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_CUBE_VISIBILITY)
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_node_visibility'))

  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })

  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'InvisibleCube',
    'ChildOfInvisibleShouldBeInvisible',
    'DescendantOfInvisibleShouldBeInvisible',
    'VisibleCube',
    'AnimatedVisibility',
  ])
  assert.deepEqual(meshes.map((mesh) => mesh.material.color.toArray()), [
    [0.855, 0, 0],
    [0.855, 0, 0],
    [0.855, 0, 0],
    [0, 0.855, 0],
    [0, 0, 0.855],
  ])
  assert.ok(meshes.every((mesh) => mesh.geometry.getAttribute('position')?.count === 24))
  assert.ok(meshes.every((mesh) => mesh.geometry.index?.count === 36))

  assert.equal(meshes[0].visible, false, 'InvisibleCube should import KHR_node_visibility false')
  assert.equal(meshes[1].visible, true, 'child mesh should keep its own default visible flag')
  assert.equal(meshes[2].visible, true, 'descendant mesh should keep its own default visible flag')
  assert.equal(meshes[3].visible, true)
  assert.equal(meshes[4].visible, true)
  assert.equal(isEffectivelyVisible(meshes[0]), false, 'InvisibleCube should be effectively hidden')
  assert.equal(isEffectivelyVisible(meshes[1]), false, 'child mesh should be hidden by its invisible parent')
  assert.equal(isEffectivelyVisible(meshes[2]), false, 'descendant mesh should be hidden by its invisible ancestor')
  assert.equal(isEffectivelyVisible(meshes[3]), true)
  assert.equal(isEffectivelyVisible(meshes[4]), true)

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1.2))
  const camera = new THREE.OrthographicCamera(-2.4, 2.4, 1.2, -1.2, 0.01, 10)
  camera.position.set(0, 0, 4)
  camera.lookAt(0, 0, 0)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 160,
    height: 80,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.15, 'CubeVisibility should render the visible green and blue cubes')
  const left = meanRegion(rgba, 160, 80, 0, 25, 45, 55)
  const center = meanRegion(rgba, 160, 80, 55, 25, 105, 55)
  const right = meanRegion(rgba, 160, 80, 115, 25, 160, 55)
  assert.ok(left.r < 5 && left.g < 5 && left.b < 5, `invisible red branch should not render (${left.r}, ${left.g}, ${left.b})`)
  assert.ok(center.g > center.r + 50 && center.g > center.b + 50, `visible green cube should render in the center (${center.r}, ${center.g}, ${center.b})`)
  assert.ok(right.b > right.r + 50 && right.b > right.g + 50, `visible blue cube should render on the right (${right.r}, ${right.g}, ${right.b})`)
})
