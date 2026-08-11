import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'
import { Renderer, SAMPLE_ASSET_IOR_TEST_GRID, SAMPLE_ASSET_SHEEN_CHAIR, SAMPLE_ASSET_TRANSMISSION_ROUGHNESS_TEST, SAMPLE_ASSET_TRANSMISSION_THINWALL_TEST_GRID } from './gltf.test.part-001.mjs'
import { loadGltfFixture, meanRegion, pngDimensions } from './gltf.test.part-028.mjs'
test('committed Khronos glTF Sample Assets IORTestGrid fixture loads IOR, transmission, volume, and specular grids', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_IOR_TEST_GRID, 'utf8'))
  assert.deepEqual(source.extensionsUsed, [
    'KHR_materials_transmission',
    'KHR_materials_volume',
    'KHR_materials_specular',
    'KHR_materials_ior',
  ])
  assert.deepEqual(source.buffers, [
    { byteLength: 2599860, uri: 'IORTestGrid.bin' },
  ])
  assert.deepEqual(source.images.map((image) => image.uri), ['checker.png'])

  const sourcePhysicalMaterials = source.materials.filter((material) => material.name.startsWith('IOR'))
  assert.deepEqual(sourcePhysicalMaterials.map((material) => [
    material.name,
    material.extensions?.KHR_materials_ior?.ior ?? 1.5,
    material.extensions?.KHR_materials_transmission?.transmissionFactor ?? 0,
    material.extensions?.KHR_materials_specular?.specularFactor ?? null,
    material.extensions?.KHR_materials_volume?.thicknessFactor ?? 0,
  ]), [
    ['IOR1.0_Black_R0_M0_T0_S0.25', 1, 0, 0.25, 0.1],
    ['IOR1.33_Black_R0_M0_T0_S0.25', 1.33, 0, 0.25, 0.1],
    ['IOR1.5_Black_R0_M0_T0_S0.25', 1.5, 0, 0.25, 0.1],
    ['IOR1.76_Black_R0_M0_T0_S0.25', 1.76, 0, 0.25, 0.1],
    ['IOR2.42_Black_R0_M0_T0_S0.25', 2.42, 0, 0.25, 0.1],
    ['IOR2.42_Black_R0_M0_T0_S1', 2.42, 0, 1, 0.1],
    ['IOR1.76_Black_R0_M0_T0_S1', 1.76, 0, 1, 0.1],
    ['IOR1.5_Black_R0_M0_T0_S1', 1.5, 0, 1, 0.1],
    ['IOR1.33_Black_R0_M0_T0_S1', 1.33, 0, 1, 0.1],
    ['IOR1.0_Black_R0_M0_T0_S1', 1, 0, 1, 0.1],
    ['IOR2.42_White_R0_M0_T1_S0.25', 2.42, 1, 0.25, 0.1],
    ['IOR1.76_White_R0_M0_T1_S0.25', 1.76, 1, 0.25, 0.1],
    ['IOR1.5_White_R0_M0_T1_S0.25', 1.5, 1, 0.25, 0.1],
    ['IOR1.33_White_R0_M0_T1_S0.25', 1.33, 1, 0.25, 0.1],
    ['IOR1.0_White_R0_M0_T1_S0.25', 1, 1, 0.25, 0.1],
    ['IOR2.42_White_R0_M0_T1_S1', 2.42, 1, 1, 0.1],
    ['IOR1.76_White_R0_M0_T1_S1', 1.76, 1, 1, 0.1],
    ['IOR1.5_White_R0_M0_T1_S1', 1.5, 1, 1, 0.1],
    ['IOR1.33_White_R0_M0_T1_S1', 1.33, 1, 1, 0.1],
    ['IOR1.0_White_R0_M0_T1_S1', 1, 1, 1, 0.1],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_IOR_TEST_GRID)
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_ior'))
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_specular'))

  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 23)
  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'Backdrop',
    'IOR242_Black_R0_M0_T0_S025',
    'IOR242_Black_R0_M0_T0_S1',
    'IOR242_White_R0_M0_T1_S025',
    'IOR242_White_R0_M0_T1_S1',
    'IOR176_Black_R0_M0_T0_S025',
    'IOR176_Black_R0_M0_T0_S1',
    'IOR176_White_R0_M0_T1_S025',
    'IOR176_White_R0_M0_T1_S1',
    'IOR15_Black_R0_M0_T0_S025',
    'IOR15_Black_R0_M0_T0_S1',
    'IOR15_White_R0_M0_T1_S025',
    'IOR15_White_R0_M0_T1_S1',
    'IOR133_Black_R0_M0_T0_S025',
    'IOR133_Black_R0_M0_T0_S1',
    'IOR133_White_R0_M0_T1_S025',
    'IOR133_White_R0_M0_T1_S1',
    'IOR10_Black_R0_M0_T0_S025',
    'IOR10_Black_R0_M0_T0_S1',
    'IOR10_White_R0_M0_T1_S025',
    'IOR10_White_R0_M0_T1_S1',
    'Text',
    'Text_Backdrop',
  ])

  const samples = meshes.filter((mesh) => mesh.material.isMeshPhysicalMaterial === true)
  assert.equal(samples.length, 20)
  assert.ok(samples.every((mesh) => mesh.geometry.getAttribute('position')?.count === 3840))
  assert.ok(samples.every((mesh) => mesh.geometry.getAttribute('normal')?.count === 3840))
  assert.ok(samples.every((mesh) => mesh.geometry.index?.count === 3840))
  assert.deepEqual(samples.map((mesh) => mesh.material.ior), [
    2.42, 2.42, 2.42, 2.42,
    1.76, 1.76, 1.76, 1.76,
    1.5, 1.5, 1.5, 1.5,
    1.33, 1.33, 1.33, 1.33,
    1, 1, 1, 1,
  ])
  assert.deepEqual(samples.map((mesh) => mesh.material.transmission), [
    0, 0, 1, 1,
    0, 0, 1, 1,
    0, 0, 1, 1,
    0, 0, 1, 1,
    0, 0, 1, 1,
  ])
  assert.deepEqual(samples.map((mesh) => mesh.material.specularIntensity), [
    0.25, 1, 0.25, 1,
    0.25, 1, 0.25, 1,
    0.25, 1, 0.25, 1,
    0.25, 1, 0.25, 1,
    0.25, 1, 0.25, 1,
  ])
  assert.ok(samples.every((mesh) => mesh.material.thickness === 0.1))
  assert.ok(samples.every((mesh) => mesh.material.roughness === 0))
  assert.ok(samples.every((mesh) => mesh.material.metalness === 0))

  const backdrop = meshes[0]
  assert.equal(backdrop.material.name, 'Backdrop')
  assert.equal(Buffer.isBuffer(backdrop.material.map?.image), true, 'IORTestGrid checker PNG should load as an encoded Buffer')
  assert.equal(backdrop.material.map.name, 'checker.png')
  assert.deepEqual(pngDimensions(backdrop.material.map.image), [256, 256])
  assert.equal(backdrop.material.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(backdrop.material.map.flipY, false)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100)
  camera.position.copy(center).add(new THREE.Vector3(0, size.y * 0.15, Math.max(size.x, size.y, size.z) * 1.6))
  camera.lookAt(center)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.7))
  const light = new THREE.DirectionalLight(0xffffff, 2.2)
  light.position.copy(center).add(new THREE.Vector3(2, 3, 5))
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 128,
    height: 128,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.5, 'IORTestGrid should render visible physical material grid')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 50 && mean.g > 50 && mean.b > 50, `IORTestGrid should render lit grid pixels (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets TransmissionRoughnessTest fixture loads IOR and roughness texture inputs', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_TRANSMISSION_ROUGHNESS_TEST, 'utf8'))
  assert.deepEqual(source.extensionsUsed, [
    'KHR_materials_transmission',
    'KHR_materials_ior',
    'KHR_materials_volume',
  ])
  assert.equal(source.buffers[0].uri, 'TransmissionRoughnessTest.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'IOR_Labels.png',
    'RoughnessGrid.png',
    'RoughnessGrid-1.png',
    'GridWithDetails.png',
    'SmoothVsRough.png',
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_TRANSMISSION_ROUGHNESS_TEST)
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_transmission'))
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_ior'))
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_volume'))

  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'Labels',
    'IOR_10',
    'IOR_133',
    'IOR_150',
    'IOR_176',
    'IOR_242',
    'Opaque',
    'Flat_Backdrop',
    'SmoothRoughLabels',
  ])

  const byName = new Map(meshes.map((mesh) => [mesh.name, mesh]))
  const samples = ['IOR_10', 'IOR_133', 'IOR_150', 'IOR_176', 'IOR_242'].map((name) => byName.get(name))
  assert.deepEqual(samples.map((mesh) => mesh.geometry.getAttribute('position')?.count), [7866, 7866, 7866, 7866, 7866])
  assert.deepEqual(samples.map((mesh) => mesh.geometry.index?.count), [38880, 38880, 38880, 38880, 38880])

  const sampleMaterials = samples.map((mesh) => mesh.material)
  assert.ok(sampleMaterials.every((material) => material.isMeshPhysicalMaterial === true), 'IOR samples should load as MeshPhysicalMaterial')
  assert.deepEqual(sampleMaterials.map((material) => material.name), [
    'Mat_IOR_1.0',
    'Mat_IOR_1.33',
    'Mat_IOR_1.50',
    'Mat_IOR_1.76',
    'Mat_IOR_2.42',
  ])
  assert.deepEqual(sampleMaterials.map((material) => material.transmission), [1, 1, 1, 1, 1])
  assert.deepEqual(sampleMaterials.map((material) => material.ior), [1, 1.33, 1.5, 1.76, 2.42])
  assert.ok(sampleMaterials.every((material) => material.thickness === 0.005))
  assert.ok(sampleMaterials.every((material) => material.roughnessMap === material.metalnessMap))
  assert.equal(new Set(sampleMaterials.map((material) => material.roughnessMap)).size, 1, 'IOR samples should share the same roughness texture object')
  assert.equal(sampleMaterials[0].roughnessMap.name, 'RoughnessGrid')
  assert.equal(sampleMaterials[0].roughnessMap.colorSpace, THREE.NoColorSpace)
  assert.deepEqual(pngDimensions(sampleMaterials[0].roughnessMap.image), [64, 64])

  const labels = byName.get('Labels')
  assert.equal(labels.material.name, 'LabelMat')
  assert.equal(labels.material.map.name, 'IOR_Labels')
  assert.equal(labels.material.map.colorSpace, THREE.SRGBColorSpace)
  assert.deepEqual(pngDimensions(labels.material.map.image), [512, 512])

  const opaque = byName.get('Opaque')
  assert.equal(opaque.material.name, 'Mat_Opaque')
  assert.equal(opaque.material.isMeshStandardMaterial, true)
  assert.equal(opaque.material.roughnessMap.name, 'RoughnessGrid')
  assert.notEqual(opaque.material.roughnessMap, sampleMaterials[0].roughnessMap)
  assert.deepEqual(pngDimensions(opaque.material.roughnessMap.image), [64, 64])

  const backdrop = byName.get('Flat_Backdrop')
  assert.equal(backdrop.material.name, 'FlatBackdrop')
  assert.equal(backdrop.material.map.name, 'GridWithDetails')
  assert.deepEqual(pngDimensions(backdrop.material.map.image), [256, 256])

  const smoothRoughLabels = byName.get('SmoothRoughLabels')
  assert.equal(smoothRoughLabels.material.side, THREE.DoubleSide)
  assert.equal(smoothRoughLabels.material.map.name, 'SmoothVsRough')
  assert.deepEqual(pngDimensions(smoothRoughLabels.material.map.image), [256, 256])

  const camera = new THREE.OrthographicCamera(-1.1, 1.1, 0.65, -0.65, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1))
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 160,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.55, 'TransmissionRoughnessTest should render visible roughness and IOR samples')
  const center = meanRegion(rgba, 160, 96, 64, 24, 96, 72)
  assert.ok(center.r > 110 && center.g > 110 && center.b > 110, `TransmissionRoughnessTest center samples should render visible panels (${center.r}, ${center.g}, ${center.b})`)
})

test('committed Khronos glTF Sample Assets TransmissionThinwallTestGrid fixture loads thin-wall and volume IOR grids', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_TRANSMISSION_THINWALL_TEST_GRID, 'utf8'))
  assert.deepEqual(source.extensionsUsed, [
    'KHR_materials_transmission',
    'KHR_materials_volume',
    'KHR_materials_ior',
  ])
  assert.deepEqual(source.buffers, [
    { byteLength: 1174536, uri: 'TransmissionThinwallTestGrid.bin' },
  ])
  assert.deepEqual(source.images.map((image) => image.uri), ['checker.png'])

  const sourcePhysicalMaterials = source.materials.filter((material) => /^(ThinWall|Volume)_IOR/.test(material.name))
  assert.deepEqual(sourcePhysicalMaterials.map((material) => [
    material.name,
    material.extensions?.KHR_materials_ior?.ior ?? null,
    material.extensions?.KHR_materials_transmission?.transmissionFactor ?? null,
    material.extensions?.KHR_materials_volume?.thicknessFactor ?? 0,
  ]), [
    ['ThinWall_IOR1.00', 1, 1, 0],
    ['ThinWall_IOR1.33', 1.33, 1, 0],
    ['ThinWall_IOR1.50', null, 1, 0],
    ['ThinWall_IOR1.76', 1.75999, 1, 0],
    ['ThinWall_IOR2.42', 2.42, 1, 0],
    ['Volume_IOR2.42', 2.42, 1, 1],
    ['Volume_IOR1.76', 1.75999, 1, 1],
    ['Volume_IOR1.50', null, 1, 1],
    ['Volume_IOR1.33', 1.33, 1, 1],
    ['Volume_IOR1.00', 1, 1, 1],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_TRANSMISSION_THINWALL_TEST_GRID)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'CheckerBackdrop',
    'ThinWall_IOR100',
    'ThinWall_IOR133',
    'ThinWall_IOR150',
    'ThinWall_IOR176',
    'Volume_IOR100',
    'Volume_IOR133',
    'Volume_IOR150',
    'Volume_IOR176',
    'TextBackdrop',
    'TextXaxis',
    'TextYaxis',
    'ThinWall_IOR242',
    'Volume_IOR242',
  ])

  const byName = new Map(meshes.map((mesh) => [mesh.name, mesh]))
  const sampleNames = [
    'ThinWall_IOR100',
    'ThinWall_IOR133',
    'ThinWall_IOR150',
    'ThinWall_IOR176',
    'Volume_IOR100',
    'Volume_IOR133',
    'Volume_IOR150',
    'Volume_IOR176',
    'ThinWall_IOR242',
    'Volume_IOR242',
  ]
  const samples = sampleNames.map((name) => byName.get(name))
  assert.ok(samples.every(Boolean), 'TransmissionThinwallTestGrid should load all thin-wall and volume samples')
  assert.deepEqual(samples.map((mesh) => mesh.geometry.getAttribute('position')?.count), Array(10).fill(3840))
  assert.deepEqual(samples.map((mesh) => mesh.geometry.getAttribute('normal')?.count), Array(10).fill(3840))
  assert.deepEqual(samples.map((mesh) => mesh.geometry.index?.count), Array(10).fill(3840))

  const sampleMaterials = samples.map((mesh) => mesh.material)
  assert.ok(sampleMaterials.every((material) => material.isMeshPhysicalMaterial === true), 'all grid samples should load as MeshPhysicalMaterial')
  assert.deepEqual(sampleMaterials.map((material) => material.name), [
    'ThinWall_IOR1.00',
    'ThinWall_IOR1.33',
    'ThinWall_IOR1.50',
    'ThinWall_IOR1.76',
    'Volume_IOR1.00',
    'Volume_IOR1.33',
    'Volume_IOR1.50',
    'Volume_IOR1.76',
    'ThinWall_IOR2.42',
    'Volume_IOR2.42',
  ])
  assert.deepEqual(sampleMaterials.map((material) => material.ior), [1, 1.33, 1.5, 1.75999, 1, 1.33, 1.5, 1.75999, 2.42, 2.42])
  assert.deepEqual(sampleMaterials.map((material) => material.transmission), Array(10).fill(1))
  assert.deepEqual(sampleMaterials.map((material) => material.thickness), [0, 0, 0, 0, 1, 1, 1, 1, 0, 1])
  assert.ok(sampleMaterials.every((material) => material.roughness === 0))
  assert.ok(sampleMaterials.every((material) => material.metalness === 0))

  const checker = byName.get('CheckerBackdrop')
  assert.equal(checker.geometry.getAttribute('position')?.count, 6)
  assert.equal(checker.geometry.getAttribute('uv')?.count, 6)
  assert.equal(checker.geometry.index?.count, 6)
  assert.equal(checker.material.name, 'Backdrop')
  assert.equal(Buffer.isBuffer(checker.material.map?.image), true, 'TransmissionThinwallTestGrid checker PNG should load as an encoded Buffer')
  assert.equal(checker.material.map.name, 'checker.png')
  assert.deepEqual(pngDimensions(checker.material.map.image), [256, 256])
  assert.equal(checker.material.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(checker.material.map.flipY, false)
  assert.equal(checker.material.map.wrapS, THREE.RepeatWrapping)
  assert.equal(checker.material.map.wrapT, THREE.RepeatWrapping)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100)
  camera.position.copy(center).add(new THREE.Vector3(0, size.y * 0.15, Math.max(size.x, size.y, size.z) * 1.6))
  camera.lookAt(center)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.7))
  const light = new THREE.DirectionalLight(0xffffff, 2.2)
  light.position.copy(center).add(new THREE.Vector3(2, 3, 5))
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 128,
    height: 128,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.2, 'TransmissionThinwallTestGrid should render visible thin-wall and volume samples')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 30 && mean.g > 30 && mean.b > 30, `TransmissionThinwallTestGrid should render lit grid pixels (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets SheenChair fixture loads KHR_materials_sheen and variants metadata', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_SHEEN_CHAIR)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 4)
  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'SheenChair_fabric',
    'SheenChair_wood',
    'SheenChair_metal',
    'SheenChair_label',
  ])

  const variants = gltf.parser?.json?.extensions?.KHR_materials_variants?.variants
  assert.deepEqual(variants?.map((variant) => variant.name), ['Mango Velvet', 'Peacock Velvet'])
  const fabric = meshes.find((mesh) => mesh.name === 'SheenChair_fabric')
  assert.ok(fabric.userData.gltfExtensions?.KHR_materials_variants?.mappings?.length >= 2, 'fabric mesh should preserve material variant mappings')
  assert.equal(fabric.geometry.getAttribute('position')?.count, 14350)
  assert.equal(fabric.geometry.getAttribute('uv')?.count, 14350)

  const fabricMaterial = fabric.material
  assert.equal(fabricMaterial.isMeshPhysicalMaterial, true)
  assert.equal(fabricMaterial.sheen, 1)
  assert.deepEqual(fabricMaterial.sheenColor.toArray(), [1, 0.329, 0.1])
  assert.equal(fabricMaterial.sheenRoughness, 0.8)
  assert.equal(Buffer.isBuffer(fabricMaterial.map?.image), true, 'fabric base color PNG should load as an encoded Buffer')
  assert.equal(Buffer.isBuffer(fabricMaterial.normalMap?.image), true, 'fabric normal PNG should load as an encoded Buffer')
  assert.equal(Buffer.isBuffer(fabricMaterial.aoMap?.image), true, 'fabric occlusion PNG should load as an encoded Buffer')
  assert.equal(fabricMaterial.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(fabricMaterial.normalMap.colorSpace, THREE.NoColorSpace)
  assert.equal(fabricMaterial.aoMap.colorSpace, THREE.NoColorSpace)
  assert.equal(fabricMaterial.aoMap.channel, 1)
  assert.deepEqual(fabricMaterial.map.offset.toArray(), [-3, 3])
  assert.deepEqual(fabricMaterial.map.repeat.toArray(), [7, 7])

  const woodMaterial = meshes.find((mesh) => mesh.name === 'SheenChair_wood').material
  assert.equal(woodMaterial.isMeshStandardMaterial, true)
  assert.equal(Buffer.isBuffer(woodMaterial.map?.image), true, 'wood base color PNG should load as an encoded Buffer')
  assert.equal(Buffer.isBuffer(woodMaterial.roughnessMap?.image), true, 'wood roughness PNG should load as an encoded Buffer')
  assert.equal(Buffer.isBuffer(woodMaterial.metalnessMap?.image), true, 'wood metalness PNG should load as an encoded Buffer')
  assert.equal(woodMaterial.roughnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(woodMaterial.metalnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(woodMaterial.aoMap.channel, 1)

  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 20)
  camera.position.set(0.6, 0.8, 2.2)
  camera.lookAt(0, 0.35, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 1.3)
  light.position.set(1.5, 3, 4)
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 128,
    height: 128,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.1, 'SheenChair should render visible sheen material geometry')
})
