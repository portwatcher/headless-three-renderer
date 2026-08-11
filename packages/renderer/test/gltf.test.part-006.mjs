import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'
import { Renderer, SAMPLE_ASSET_COMPARE_METALLIC, SAMPLE_ASSET_COMPARE_ROUGHNESS, SAMPLE_ASSET_COMPARE_SHEEN, SAMPLE_ASSET_COMPARE_SPECULAR, SAMPLE_ASSET_IRIDESCENCE_SUZANNE } from './gltf.test.part-001.mjs'
import { assertVectorClose, loadGltfFixture, pngDimensions } from './gltf.test.part-028.mjs'
test('committed Khronos glTF Sample Assets IridescenceSuzanne fixture loads iridescence thickness texture and punctual light', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_IRIDESCENCE_SUZANNE, 'utf8'))
  assert.deepEqual(source.extensionsUsed, [
    'KHR_lights_punctual',
    'KHR_materials_ior',
    'KHR_materials_transmission',
    'KHR_materials_volume',
    'KHR_materials_iridescence',
  ])
  assert.deepEqual(source.buffers, [{ byteLength: 129888, uri: 'IridescenceSuzanne.bin' }])
  assert.deepEqual(source.images.map((image) => image.uri), ['noise.png'])
  assert.deepEqual(source.extensions.KHR_lights_punctual.lights, [
    {
      color: [1, 1, 1],
      type: 'directional',
      intensity: 2,
    },
  ])
  assert.deepEqual(source.meshes.map((mesh) => mesh.name), ['Suzanne1', 'Suzanne2', 'Suzanne3'])
  assert.deepEqual(source.nodes.map((node) => node.name), ['Suzanne1', 'Suzanne2', 'Suzanne3', 'Light'])
  assert.deepEqual(source.nodes.map((node) => node.translation), [
    [-3, 0, 0],
    [0, 0, 0],
    [3, 0, 0],
    [5, 5, 5],
  ])

  const materials = source.materials
  assert.equal(materials.length, 3)
  assert.deepEqual(materials.map((material) => material.pbrMetallicRoughness?.metallicFactor), [0, 1, 0])
  assert.deepEqual(materials.map((material) => material.extensions?.KHR_materials_iridescence?.iridescenceIor), [1.33, 1.33, 1.8])
  assert.deepEqual(materials.map((material) => material.extensions?.KHR_materials_volume?.thicknessFactor), [2, 2, 2])
  assert.deepEqual(materials.map((material) => material.extensions?.KHR_materials_transmission?.transmissionFactor), [0, 0, 1])
  assert.deepEqual(materials[2].extensions.KHR_materials_iridescence.iridescenceThicknessTexture, { index: 0 })

  const gltf = await loadGltfFixture(SAMPLE_ASSET_IRIDESCENCE_SUZANNE)
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_lights_punctual'))
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_iridescence'))
  const meshes = []
  const lights = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
    if (object.isLight === true) lights.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), ['Suzanne1', 'Suzanne2', 'Suzanne3'])
  assert.equal(lights.length, 1)
  assert.equal(lights[0].isDirectionalLight, true)
  assert.equal(lights[0].name, 'Light')
  assert.deepEqual(lights[0].position.toArray(), [5, 5, 5])
  assert.deepEqual(lights[0].color.toArray(), [1, 1, 1])
  assert.equal(lights[0].intensity, 2)

  assert.ok(meshes.every((mesh) => mesh.material.isMeshPhysicalMaterial === true), 'all IridescenceSuzanne meshes should use MeshPhysicalMaterial')
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [3321, 3321, 3321])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [3321, 3321, 3321])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [3321, 3321, 3321])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [11808, 11808, 11808])
  assert.deepEqual(meshes.map((mesh) => mesh.position.toArray()), [
    [-3, 0, 0],
    [0, 0, 0],
    [3, 0, 0],
  ])

  const [dielectric, metallic, textured] = meshes.map((mesh) => mesh.material)
  assert.deepEqual([dielectric.metalness, metallic.metalness, textured.metalness], [0, 1, 0])
  assert.deepEqual([dielectric.ior, metallic.ior, textured.ior], [2, 1.5, 1.5])
  assert.deepEqual([dielectric.transmission, metallic.transmission, textured.transmission], [0, 0, 1])
  assert.deepEqual([dielectric.thickness, metallic.thickness, textured.thickness], [2, 2, 2])
  assert.deepEqual([dielectric.iridescence, metallic.iridescence, textured.iridescence], [1, 1, 1])
  assert.deepEqual([dielectric.iridescenceIOR, metallic.iridescenceIOR, textured.iridescenceIOR], [1.33, 1.33, 1.8])
  assert.deepEqual([dielectric.iridescenceThicknessRange, metallic.iridescenceThicknessRange, textured.iridescenceThicknessRange], [
    [100, 400],
    [100, 400],
    [200, 600],
  ])

  assert.equal(dielectric.iridescenceThicknessMap ?? null, null)
  assert.equal(metallic.iridescenceThicknessMap ?? null, null)
  assert.equal(Buffer.isBuffer(textured.iridescenceThicknessMap?.image), true, 'IridescenceSuzanne noise PNG should load as an encoded Buffer')
  assert.equal(textured.iridescenceThicknessMap.name, 'noise.png')
  assert.deepEqual(pngDimensions(textured.iridescenceThicknessMap.image), [1024, 1024])
  assert.equal(textured.iridescenceThicknessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(textured.iridescenceThicknessMap.flipY, false)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.35))
  const camera = new THREE.PerspectiveCamera(40, 1.6, 0.01, 50)
  camera.position.copy(center).add(new THREE.Vector3(0, -7, 3))
  camera.lookAt(center)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 160,
    height: 100,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.08, 'IridescenceSuzanne should render visible iridescence Suzanne meshes')
})

test('committed Khronos glTF Sample Assets CompareMetallic fixture loads metallic texture comparison variants', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_COMPARE_METALLIC, 'utf8'))
  assert.equal(source.buffers[0].uri, 'CompareMetallic.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'Compare_Metallic_img0.jpg',
    'Compare_Metallic_img1.jpg',
  ])
  assert.deepEqual(source.materials.map((material) => [
    material.name,
    material.pbrMetallicRoughness?.baseColorTexture?.index ?? null,
    material.pbrMetallicRoughness?.metallicRoughnessTexture?.index ?? null,
    material.pbrMetallicRoughness?.metallicFactor ?? null,
    material.pbrMetallicRoughness?.roughnessFactor ?? null,
  ]), [
    ['glTF Logo', 0, null, 0, 0.1],
    ['glTF Logo Metallic', 0, 1, null, 0.1],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_COMPARE_METALLIC)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), ['GeoSphere001', 'GeoSphere002'])
  assert.deepEqual(meshes.map((mesh) => mesh.material.name), ['glTF Logo', 'glTF Logo Metallic'])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [3840, 3840])

  const [dielectric, metallic] = meshes.map((mesh) => mesh.material)
  assert.equal(dielectric.metalness, 0)
  assert.equal(dielectric.roughness, 0.1)
  assert.equal(metallic.metalness, 1)
  assert.equal(metallic.roughness, 0.1)
  assert.equal(dielectric.map, metallic.map, 'both metallic comparison materials should share the base-color texture')
  assert.equal(Buffer.isBuffer(dielectric.map?.image), true, 'base-color JPEG should load as an encoded Buffer')
  assert.equal(dielectric.map.name, 'Compare_Metallic_img0.jpg')
  assert.equal(dielectric.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(dielectric.map.flipY, false)

  assert.equal(dielectric.metalnessMap ?? null, null)
  assert.equal(dielectric.roughnessMap ?? null, null)
  assert.equal(metallic.metalnessMap, metallic.roughnessMap, 'metalness and roughness should share the packed texture')
  assert.equal(Buffer.isBuffer(metallic.metalnessMap?.image), true, 'metallic-roughness JPEG should load as an encoded Buffer')
  assert.equal(metallic.metalnessMap.name, 'Compare_Metallic_img1.jpg')
  assert.equal(metallic.metalnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(metallic.metalnessMap.flipY, false)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.75))
  const light = new THREE.DirectionalLight(0xffffff, 2.2)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)
  const camera = new THREE.PerspectiveCamera(35, 1.5, 0.01, 20)
  camera.position.copy(center).add(new THREE.Vector3(0, -2.7, 1.4))
  camera.lookAt(center)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 144,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.25, 'CompareMetallic should render visible metallic comparison spheres')
})

test('committed Khronos glTF Sample Assets CompareRoughness fixture loads roughness texture comparison variants', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_COMPARE_ROUGHNESS, 'utf8'))
  assert.equal(source.buffers[0].uri, 'CompareRoughness.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'Compare_Roughness_img0.jpg',
    'Compare_Roughness_img1.jpg',
  ])
  assert.deepEqual(source.materials.map((material) => [
    material.name,
    material.pbrMetallicRoughness?.baseColorTexture?.index ?? null,
    material.pbrMetallicRoughness?.metallicRoughnessTexture?.index ?? null,
    material.pbrMetallicRoughness?.metallicFactor ?? null,
    material.pbrMetallicRoughness?.roughnessFactor ?? null,
  ]), [
    ['glTF Logo', 0, null, 0, 0],
    ['glTF Logo Roughness', 0, 1, 0, 0.5],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_COMPARE_ROUGHNESS)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), ['GeoSphere001', 'GeoSphere002'])
  assert.deepEqual(meshes.map((mesh) => mesh.material.name), ['glTF Logo', 'glTF Logo Roughness'])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [3840, 3840])

  const [smooth, rough] = meshes.map((mesh) => mesh.material)
  assert.equal(smooth.metalness, 0)
  assert.equal(smooth.roughness, 0)
  assert.equal(rough.metalness, 0)
  assert.equal(rough.roughness, 0.5)
  assert.equal(smooth.map, rough.map, 'both roughness comparison materials should share the base-color texture')
  assert.equal(Buffer.isBuffer(smooth.map?.image), true, 'base-color JPEG should load as an encoded Buffer')
  assert.equal(smooth.map.name, 'Compare_Roughness_img0.jpg')
  assert.equal(smooth.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(smooth.map.flipY, false)

  assert.equal(smooth.metalnessMap ?? null, null)
  assert.equal(smooth.roughnessMap ?? null, null)
  assert.equal(rough.metalnessMap, rough.roughnessMap, 'metalness and roughness should share the packed texture')
  assert.equal(Buffer.isBuffer(rough.roughnessMap?.image), true, 'metallic-roughness JPEG should load as an encoded Buffer')
  assert.equal(rough.roughnessMap.name, 'Compare_Roughness_img1.jpg')
  assert.equal(rough.roughnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(rough.roughnessMap.flipY, false)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 2))
  const light = new THREE.DirectionalLight(0xffffff, 6)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)
  const camera = new THREE.PerspectiveCamera(35, 1.5, 0.01, 20)
  camera.position.copy(center).add(new THREE.Vector3(0, -2.7, 1.4))
  camera.lookAt(center)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 144,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.SRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.045, 'CompareRoughness should render visible roughness comparison spheres')
})

test('committed Khronos glTF Sample Assets CompareSheen fixture loads sheen comparison variants', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_COMPARE_SHEEN, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_sheen'])
  assert.equal(source.buffers[0].uri, 'CompareSheen.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'Compare_Sheen_img0.jpg',
    'Compare_Sheen_img1.jpg',
    'Compare_Sheen_img2.jpg',
  ])
  assert.deepEqual(source.materials.map((material) => [
    material.name,
    material.normalTexture?.index ?? null,
    material.normalTexture?.scale ?? null,
    material.pbrMetallicRoughness?.baseColorTexture?.index ?? null,
    material.extensions?.KHR_materials_sheen?.sheenColorFactor ?? null,
    material.extensions?.KHR_materials_sheen?.sheenRoughnessFactor ?? null,
  ]), [
    ['glTF Logo', 1, 0.5, 2, null, null],
    ['glTF Logo Sheen', 1, 0.5, 0, [1, 0, 0], 0.3],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_COMPARE_SHEEN)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), ['GeoSphere001', 'GeoSphere002'])
  assert.deepEqual(meshes.map((mesh) => mesh.material.name), ['glTF Logo', 'glTF Logo Sheen'])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [3840, 3840])

  const [baseline, sheen] = meshes.map((mesh) => mesh.material)
  assert.equal(baseline.isMeshStandardMaterial, true)
  assert.equal(sheen.isMeshPhysicalMaterial, true)
  assert.equal(baseline.metalness, 0)
  assert.equal(sheen.metalness, 0)
  assert.equal(baseline.roughness, 0.75)
  assert.equal(sheen.roughness, 0.75)
  assert.equal(sheen.sheen, 1)
  assertVectorClose(sheen.sheenColor.toArray(), [1, 0, 0], 'CompareSheen sheenColorFactor')
  assert.equal(sheen.sheenRoughness, 0.3)
  assert.equal(baseline.sheenColorMap ?? null, null)
  assert.equal(sheen.sheenColorMap ?? null, null)
  assert.equal(sheen.sheenRoughnessMap ?? null, null)

  assert.notEqual(baseline.map, sheen.map, 'sheen comparison materials intentionally use different base-color textures')
  assert.equal(Buffer.isBuffer(baseline.map?.image), true, 'baseline base-color JPEG should load as an encoded Buffer')
  assert.equal(Buffer.isBuffer(sheen.map?.image), true, 'sheen base-color JPEG should load as an encoded Buffer')
  assert.equal(baseline.map.name, 'Compare_Sheen_img2.jpg')
  assert.equal(sheen.map.name, 'Compare_Sheen_img0.jpg')
  assert.equal(baseline.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(sheen.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(baseline.map.flipY, false)
  assert.equal(sheen.map.flipY, false)

  assert.equal(baseline.normalMap, sheen.normalMap, 'both sheen comparison materials should share the normal map')
  assert.equal(Buffer.isBuffer(baseline.normalMap?.image), true, 'shared normal JPEG should load as an encoded Buffer')
  assert.equal(baseline.normalMap.name, 'Compare_Sheen_img1.jpg')
  assert.equal(baseline.normalMap.colorSpace, THREE.NoColorSpace)
  assert.equal(baseline.normalMap.flipY, false)
  assertVectorClose(baseline.normalScale.toArray(), [0.5, -0.5], 'CompareSheen baseline normal scale')
  assertVectorClose(sheen.normalScale.toArray(), [0.5, -0.5], 'CompareSheen sheen normal scale')

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1.2))
  const light = new THREE.DirectionalLight(0xffffff, 4)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)
  const camera = new THREE.PerspectiveCamera(35, 1.5, 0.01, 20)
  camera.position.copy(center).add(new THREE.Vector3(0, -2.7, 1.4))
  camera.lookAt(center)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 144,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.SRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.25, 'CompareSheen should render visible sheen comparison spheres')
})

test('committed Khronos glTF Sample Assets CompareSpecular fixture loads specular extension comparison variants', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_COMPARE_SPECULAR, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_specular'])
  assert.equal(source.buffers[0].uri, 'CompareSpecular.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'Compare_Specular_img0.jpg',
    'Compare_Specular_img1.png',
  ])
  assert.deepEqual(source.materials.map((material) => [
    material.name,
    material.pbrMetallicRoughness?.baseColorTexture?.index ?? null,
    material.extensions?.KHR_materials_specular?.specularFactor ?? null,
    material.extensions?.KHR_materials_specular?.specularTexture?.index ?? null,
    material.extensions?.KHR_materials_specular?.specularColorTexture?.index ?? null,
  ]), [
    ['glTF Logo', 0, null, null, null],
    ['glTF Logo Specular', 0, 1, 1, 1],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_COMPARE_SPECULAR)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), ['GeoSphere001', 'GeoSphere002'])
  assert.deepEqual(meshes.map((mesh) => mesh.material.name), ['glTF Logo', 'glTF Logo Specular'])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [2625, 2625])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [2625, 2625])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [2625, 2625])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [15360, 15360])

  const [baseline, specular] = meshes.map((mesh) => mesh.material)
  assert.equal(baseline.isMeshStandardMaterial, true)
  assert.equal(specular.isMeshPhysicalMaterial, true)
  assert.equal(baseline.roughness, 0.4)
  assert.equal(specular.roughness, 0.4)
  assert.equal(baseline.map, specular.map, 'both specular comparison materials should share the base-color texture')
  assert.equal(Buffer.isBuffer(baseline.map?.image), true, 'base-color JPEG should load as an encoded Buffer')
  assert.equal(baseline.map.name, 'Compare_Specular_img0.jpg')
  assert.equal(baseline.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(baseline.map.flipY, false)

  assert.equal(baseline.specularIntensityMap ?? null, null)
  assert.equal(baseline.specularColorMap ?? null, null)
  assert.equal(specular.specularIntensity, 1)
  assertVectorClose(specular.specularColor.toArray(), [10, 10, 10], 'CompareSpecular specularColorFactor')
  assert.equal(specular.specularIntensityMap, specular.specularColorMap, 'specular intensity and color should share the extension texture')
  assert.equal(Buffer.isBuffer(specular.specularColorMap?.image), true, 'specular PNG should load as an encoded Buffer')
  assert.equal(specular.specularColorMap.name, 'Compare_Specular_img1.png')
  assert.deepEqual(pngDimensions(specular.specularColorMap.image), [1024, 512])
  assert.equal(specular.specularColorMap.colorSpace, THREE.SRGBColorSpace)
  assert.equal(specular.specularColorMap.flipY, false)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1))
  const light = new THREE.DirectionalLight(0xffffff, 4)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)
  const camera = new THREE.PerspectiveCamera(35, 1.5, 0.01, 20)
  camera.position.copy(center).add(new THREE.Vector3(0, -3, 1.4))
  camera.lookAt(center)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 144,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.SRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.2, 'CompareSpecular should render visible specular comparison spheres')
})
