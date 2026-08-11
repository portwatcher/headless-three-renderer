import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'
import { Renderer, SAMPLE_ASSET_ATTENUATION_TEST, SAMPLE_ASSET_COMPARE_VOLUME, SAMPLE_ASSET_TRANSMISSION_ORDER_TEST, SAMPLE_ASSET_TRANSMISSION_TEST } from './gltf.test.part-001.mjs'
import { assertVectorClose, loadGltfFixture, meanRegion, pngDimensions } from './gltf.test.part-028.mjs'
test('committed Khronos glTF Sample Assets TransmissionOrderTest fixture loads alpha and transmission ordering cases', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_TRANSMISSION_ORDER_TEST, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_transmission', 'KHR_materials_volume'])
  assert.deepEqual(source.buffers, [{ byteLength: 2291932, uri: 'TransmissionOrderTest.bin' }])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'checkerboard.png',
    'alphaInACircle.png',
    'BlendMaskOpaqueLabels.png',
  ])
  assert.deepEqual(source.materials.map((material) => [
    material.name,
    material.alphaMode ?? 'OPAQUE',
    material.alphaCutoff ?? null,
    material.doubleSided ?? false,
    material.pbrMetallicRoughness?.baseColorTexture?.index ?? null,
    material.extensions?.KHR_materials_transmission?.transmissionFactor ?? null,
    material.extensions?.KHR_materials_volume?.thicknessFactor ?? null,
  ]), [
    ['Cloth Backdrop', 'OPAQUE', null, false, 0, null, null],
    ['Alpha Blend Material', 'BLEND', null, true, 1, null, null],
    ['Blue Glass Material', 'OPAQUE', null, false, null, 1, 0.4000000059604645],
    ['Alpha Mask Material', 'MASK', null, true, 2, null, null],
    ['Label Material', 'OPAQUE', null, false, 3, null, null],
    ['Opaque Material', 'OPAQUE', null, true, null, null, null],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_TRANSMISSION_ORDER_TEST)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 20)
  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'Cloth_Backdrop',
    'AlphaBlend',
    'Glass',
    'Glass001',
    'Glass002',
    'Glass003',
    'Glass004',
    'Glass005',
    'AlphaBlend001',
    'AlphaBlend002',
    'AlphaMask',
    'AlphaMask001',
    'AlphaMask002',
    'Labels',
    'Glass006',
    'Glass007',
    'Glass008',
    'Opaque',
    'Opaque001',
    'Opaque002',
  ])
  assert.deepEqual(meshes.slice(0, 4).map((mesh) => mesh.geometry.getAttribute('position')?.count), [62658, 4, 296, 296])
  assert.deepEqual(meshes.slice(0, 4).map((mesh) => mesh.geometry.index?.count), [131337, 6, 1764, 1764])

  const materials = new Map()
  for (const mesh of meshes) materials.set(mesh.material.name, mesh.material)

  const backdrop = materials.get('Cloth Backdrop')
  assert.equal(Buffer.isBuffer(backdrop.map?.image), true, 'checkerboard PNG should load as an encoded Buffer')
  assert.equal(backdrop.map.name, 'checkerboard')
  assert.deepEqual(pngDimensions(backdrop.map.image), [2048, 2048])
  assert.equal(backdrop.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(backdrop.map.flipY, false)

  const alphaBlend = materials.get('Alpha Blend Material')
  assert.equal(alphaBlend.transparent, true)
  assert.equal(alphaBlend.depthWrite, false)
  assert.equal(alphaBlend.side, THREE.DoubleSide)
  assert.equal(alphaBlend.alphaTest, 0)
  assert.equal(alphaBlend.map.name, 'alphaInACircle')
  assert.deepEqual(pngDimensions(alphaBlend.map.image), [256, 256])
  assert.equal(alphaBlend.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(alphaBlend.map.flipY, false)

  const alphaMask = materials.get('Alpha Mask Material')
  assert.equal(alphaMask.transparent, false)
  assert.equal(alphaMask.depthWrite, true)
  assert.equal(alphaMask.side, THREE.DoubleSide)
  assert.equal(alphaMask.alphaTest, 0.5)
  assert.equal(alphaMask.map.name, 'alphaInACircle')

  const glass = materials.get('Blue Glass Material')
  assert.equal(glass.isMeshPhysicalMaterial, true)
  assert.equal(glass.transmission, 1)
  assert.equal(glass.thickness, 0.4000000059604645)
  assert.equal(glass.attenuationDistance, 1)
  assertVectorClose(glass.attenuationColor.toArray(), [1, 1, 1], 'TransmissionOrderTest glass attenuation color')

  const label = materials.get('Label Material')
  assert.equal(Buffer.isBuffer(label.map?.image), true, 'label PNG should load as an encoded Buffer')
  assert.equal(label.map.name, 'BlendMaskOpaqueLabels')
  assert.deepEqual(pngDimensions(label.map.image), [256, 256])

  const opaque = materials.get('Opaque Material')
  assert.equal(opaque.side, THREE.DoubleSide)
  assert.equal(opaque.transparent, false)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 1.4)
  light.position.set(3, 4, 6)
  gltf.scene.add(light)
  const camera = new THREE.PerspectiveCamera(45, 1.5, 0.01, 100)
  camera.position.set(center.x, center.y - 6, center.z + 4)
  camera.lookAt(center)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 96,
    height: 64,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.15, 'TransmissionOrderTest should render visible transparent/transmissive ordering panels')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 15 && mean.g > 10 && mean.b > 10, `TransmissionOrderTest should render non-black layered output (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets TransmissionTest fixture loads texture-driven transmission grid', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_TRANSMISSION_TEST, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_transmission', 'KHR_xmp'])
  assert.deepEqual(source.asset.extensions, { KHR_xmp: { packet: 0 } })
  assert.equal(source.buffers[0].uri, 'TransmissionTest_binary.bin')
  assert.equal(source.buffers[0].byteLength, 1441156)
  assert.deepEqual(source.images.map((image) => image.uri), [
    'TransmissionTest_images/texture28577.png',
    'TransmissionTest_images/texture14184.png',
    'TransmissionTest_images/texture214190.png',
    'TransmissionTest_images/texture4086.png',
    'TransmissionTest_images/texture177328.png',
    'TransmissionTest_images/texture6807.png',
    'TransmissionTest_images/texture175763.png',
    'TransmissionTest_images/texture10487.png',
    'TransmissionTest_images/texture15366.png',
  ])
  assert.equal(source.materials.length, 14)
  assert.equal(source.materials.filter((material) => material.extensions?.KHR_materials_transmission).length, 12)
  assert.equal(source.materials.filter((material) => material.alphaMode === 'MASK').length, 6)

  const gltf = await loadGltfFixture(SAMPLE_ASSET_TRANSMISSION_TEST)
  assert.deepEqual(gltf.parser.json.extensionsUsed, ['KHR_materials_transmission', 'KHR_xmp'])
  assert.equal(gltf.cameras.length, 1)
  const importedCamera = gltf.cameras[0]
  assert.equal(importedCamera.name, 'render_camera_n3d')
  assert.equal(importedCamera.isPerspectiveCamera, true)
  assert.ok(Math.abs(importedCamera.fov - 34.515876027228366) < 1e-6, `TransmissionTest camera fov should load (${importedCamera.fov})`)

  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 22)
  assert.equal(meshes[0].name, 'Cloth_Backdrop_01')
  assert.equal(meshes[0].geometry.getAttribute('position')?.count, 22202)
  assert.equal(meshes[0].geometry.index?.count, 131337)
  assert.equal(meshes.filter((mesh) => mesh.name.startsWith('RedTransTexture')).length, 3)
  assert.equal(meshes.filter((mesh) => mesh.name.startsWith('BlueTransWithMask')).length, 3)
  assert.ok(meshes.slice(1, 13).every((mesh) => mesh.geometry.getAttribute('position')?.count === 3719), 'TransmissionTest sphere samples should share geometry density')
  assert.ok(meshes.slice(1, 13).every((mesh) => mesh.geometry.index?.count === 21240), 'TransmissionTest sphere samples should share indexed sphere geometry')

  const materials = [...new Set(meshes.map((mesh) => mesh.material))]
  assert.equal(materials.length, 14)
  assert.equal(materials.filter((material) => material.isMeshPhysicalMaterial === true).length, 12)
  assert.ok(materials.filter((material) => material.transmission === 1).length >= 12)
  assert.equal(materials.filter((material) => material.alphaTest === 0.5).length, 6)
  assert.equal(materials.filter((material) => material.transmissionMap?.name === 'texture14184').length, 6)
  assert.equal(materials.filter((material) => material.roughnessMap?.name === 'texture177328').length, 2)
  assert.equal(materials.filter((material) => material.roughnessMap?.name === 'texture175763').length, 3)

  const byName = new Map(materials.map((material) => [material.name, material]))
  const red = byName.get('RedTransTexture')
  const yellow = byName.get('YellowTrans')
  const blue = materials.find((material) => material.name === 'BlueTransWithMask' && material.map?.name === 'texture214190')
  const green = byName.get('GreenMask')
  assert.equal(red.transmission, 1)
  assert.equal(red.transmissionMap.name, 'texture14184')
  assert.equal(red.transmissionMap.colorSpace, THREE.NoColorSpace)
  assert.deepEqual(pngDimensions(red.transmissionMap.image), [256, 256])
  assert.equal(yellow.transmission, 1)
  assert.equal(yellow.transmissionMap ?? null, null)
  assert.equal(blue.alphaTest, 0.5)
  assert.equal(blue.map.name, 'texture214190')
  assert.equal(blue.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(green.alphaTest, 0.5)
  assert.equal(green.map.name, 'texture4086')
  assert.equal(green.transmission, 1)

  importedCamera.aspect = 4 / 3
  importedCamera.updateProjectionMatrix()
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 1.7)
  light.position.set(0, 2, 4)
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)
  importedCamera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, importedCamera, {
    width: 160,
    height: 120,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.8, 'TransmissionTest should render visible texture-driven transmission grid')
})

test('committed Khronos glTF Sample Assets CompareVolume fixture loads transmission volume variants', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_COMPARE_VOLUME, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_transmission', 'KHR_materials_volume'])
  assert.equal(source.buffers[0].uri, 'CompareVolume.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'Compare_Volume_img0.jpg',
    'Compare_Volume_img1.jpg',
    'Compare_Volume_img2.png',
    'Compare_Volume_img3.jpg',
  ])
  assert.deepEqual(source.materials.map((material) => [
    material.name,
    material.pbrMetallicRoughness?.baseColorTexture?.index ?? null,
    material.pbrMetallicRoughness?.metallicRoughnessTexture?.index ?? null,
    material.extensions?.KHR_materials_transmission?.transmissionFactor ?? null,
    material.extensions?.KHR_materials_volume?.thicknessFactor ?? null,
    material.extensions?.KHR_materials_volume?.thicknessTexture?.index ?? null,
  ]), [
    ['checker', 2, null, null, null, null],
    ['glTF Transmission', null, 0, 1, null, null],
    ['gold', null, null, null, null, null],
    ['glTF Volume', null, 0, 1, 0.75, 1],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_COMPARE_VOLUME)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'Sphere001_0',
    'Sphere001_1',
    'Sphere002_0',
    'Sphere002_1',
    'Checker',
  ])
  assert.deepEqual(meshes.map((mesh) => mesh.material.name), [
    'glTF Transmission',
    'gold',
    'glTF Volume',
    'gold',
    'checker',
  ])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [2732, 390, 2732, 390, 6])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [2732, 390, 2732, 390, 6])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [2732, 390, 2732, 390, 6])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [15744, 1920, 15744, 1920, 6])

  const [transmissionShell, goldLeft, volumeShell, goldRight, checker] = meshes.map((mesh) => mesh.material)
  assert.equal(transmissionShell.isMeshPhysicalMaterial, true)
  assert.equal(volumeShell.isMeshPhysicalMaterial, true)
  assert.equal(transmissionShell.transmission, 1)
  assert.equal(volumeShell.transmission, 1)
  assert.equal(transmissionShell.thickness, 0)
  assert.equal(volumeShell.thickness, 0.75)
  assert.equal(volumeShell.attenuationDistance, 0.25)
  assertVectorClose(volumeShell.attenuationColor.toArray(), [0.15, 1, 0.5], 'CompareVolume attenuation color')
  assert.equal(goldLeft, goldRight, 'both volume comparison cores should share the gold material instance')
  assertVectorClose(goldLeft.color.toArray(), [
    0.8823530077934265,
    0.5921568870544434,
    0.250980406999588,
  ], 'CompareVolume gold baseColorFactor')
  assert.equal(goldLeft.metalness, 1)
  assert.equal(goldLeft.roughness, 0.2)

  assert.equal(transmissionShell.roughnessMap, transmissionShell.metalnessMap)
  assert.equal(volumeShell.roughnessMap, volumeShell.metalnessMap)
  assert.equal(transmissionShell.roughnessMap, volumeShell.roughnessMap)
  assert.equal(Buffer.isBuffer(volumeShell.roughnessMap?.image), true, 'shared metallic-roughness JPEG should load as an encoded Buffer')
  assert.equal(volumeShell.roughnessMap.name, 'Compare_Volume_img0.jpg')
  assert.equal(volumeShell.roughnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(volumeShell.roughnessMap.flipY, false)

  assert.equal(transmissionShell.thicknessMap ?? null, null)
  assert.equal(Buffer.isBuffer(volumeShell.thicknessMap?.image), true, 'volume thickness JPEG should load as an encoded Buffer')
  assert.equal(volumeShell.thicknessMap.name, 'Compare_Volume_img1.jpg')
  assert.equal(volumeShell.thicknessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(volumeShell.thicknessMap.flipY, false)

  assert.equal(Buffer.isBuffer(checker.map?.image), true, 'checker PNG should load as an encoded Buffer')
  assert.equal(checker.map.name, 'Compare_Volume_img2.png')
  assert.deepEqual(pngDimensions(checker.map.image), [64, 64])
  assert.equal(checker.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(checker.map.flipY, false)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1))
  const light = new THREE.DirectionalLight(0xffffff, 3)
  light.position.set(2, 4, 5)
  gltf.scene.add(light)
  const camera = new THREE.PerspectiveCamera(35, 1.5, 0.01, 20)
  camera.position.copy(center).add(new THREE.Vector3(0, -3.2, 1.4))
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.35, 'CompareVolume should render visible volume comparison geometry')
})

test('committed Khronos glTF Sample Assets AttenuationTest fixture loads volume attenuation and thickness cases', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_ATTENUATION_TEST, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_transmission', 'KHR_materials_volume'])
  assert.deepEqual(source.buffers, [{ byteLength: 10584, uri: 'AttenuationTest.bin' }])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'AttenuationLabels.png',
    'ThicknessTexture.png',
    'PlainGrid.png',
  ])
  assert.equal(source.materials.length, 18)
  assert.deepEqual(source.materials.slice(0, 5).map((material) => [
    material.name,
    material.extensions?.KHR_materials_transmission?.transmissionFactor,
    material.extensions?.KHR_materials_volume?.thicknessFactor,
    material.extensions?.KHR_materials_volume?.attenuationDistance,
    material.extensions?.KHR_materials_volume?.attenuationColor,
  ]), [
    ['R2_and_R4_ThicknessFac_1.0', 1, 1, 1, [0.1, 0.5, 0.9]],
    ['R2_ThicknessFac_1.5', 1, 1.5, 1, [0.1, 0.5, 0.9]],
    ['R2_ThicknessFac_2.0', 1, 2, 1, [0.1, 0.5, 0.9]],
    ['R2_ThicknessFac_0.50', 1, 0.5, 1, [0.1, 0.5, 0.9]],
    ['R2_ThicknessFac_0.25', 1, 0.25, 1, [0.1, 0.5, 0.9]],
  ])
  assert.deepEqual(source.materials.slice(7, 12).map((material) => [
    material.name,
    material.extensions?.KHR_materials_volume?.thicknessFactor,
    material.extensions?.KHR_materials_volume?.attenuationDistance,
  ]), [
    ['R5_Attenuation_1.0', 1, 1],
    ['R5_Attenuation_1.5', 1, 0.6666666667],
    ['R5_Attenuation_2.0', 1, 0.5],
    ['R5_Attenuation_0.50', 1, 2],
    ['R5_Attenuation_0.25', 1, 4],
  ])
  assert.equal(source.materials[6].extensions.KHR_materials_volume.thicknessTexture.index, 1)

  const gltf = await loadGltfFixture(SAMPLE_ASSET_ATTENUATION_TEST)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 23)
  assert.deepEqual(meshes.slice(0, 6).map((mesh) => mesh.name), [
    'R2_Block_10',
    'R2_Block_15',
    'R2_Block_20',
    'R2_Block_050',
    'R2_Block_025',
    'Labels',
  ])

  const materials = new Map(meshes.map((mesh) => [mesh.material.name, mesh.material]))
  const thicknessOne = materials.get('R2_and_R4_ThicknessFac_1.0')
  assert.equal(thicknessOne.isMeshPhysicalMaterial, true)
  assert.equal(thicknessOne.transmission, 1)
  assert.equal(thicknessOne.thickness, 1)
  assert.equal(thicknessOne.attenuationDistance, 1)
  assertVectorClose(thicknessOne.attenuationColor.toArray(), [0.1, 0.5, 0.9], 'AttenuationTest attenuation color')
  assert.equal(materials.get('R2_ThicknessFac_2.0').thickness, 2)
  assert.equal(materials.get('R2_ThicknessFac_0.25').thickness, 0.25)
  assert.equal(materials.get('R5_Attenuation_2.0').attenuationDistance, 0.5)
  assert.equal(materials.get('R5_Attenuation_0.25').attenuationDistance, 4)

  const textureMaterial = materials.get('R3_ThicknessTex_Mat')
  assert.equal(textureMaterial.thickness, 2)
  assert.equal(Buffer.isBuffer(textureMaterial.thicknessMap?.image), true, 'thickness texture PNG should load as an encoded Buffer')
  assert.equal(textureMaterial.thicknessMap.name, 'ThicknessTexture')
  assert.deepEqual(pngDimensions(textureMaterial.thicknessMap.image), [256, 256])
  assert.equal(textureMaterial.thicknessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(textureMaterial.thicknessMap.flipY, false)

  const labelMaterial = materials.get('LabelMaterial')
  assert.equal(Buffer.isBuffer(labelMaterial.map?.image), true, 'attenuation label PNG should load as an encoded Buffer')
  assert.equal(labelMaterial.map.name, 'AttenuationLabels')
  assert.deepEqual(pngDimensions(labelMaterial.map.image), [512, 512])
  assert.equal(labelMaterial.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(labelMaterial.map.flipY, false)

  const backdrop = materials.get('FlatBackdrop')
  assert.equal(Buffer.isBuffer(backdrop.map?.image), true, 'plain grid PNG should load as an encoded Buffer')
  assert.equal(backdrop.map.name, 'PlainGrid')
  assert.deepEqual(pngDimensions(backdrop.map.image), [256, 256])
  assert.equal(backdrop.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(backdrop.map.flipY, false)

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.85))
  const light = new THREE.DirectionalLight(0xffffff, 1.3)
  light.position.set(2, 3, 8)
  gltf.scene.add(light)
  const camera = new THREE.OrthographicCamera(-10.8, 10.8, 10.8, -10.8, 0.01, 50)
  camera.position.set(0, 0, 22)
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.75, 'AttenuationTest should render visible attenuation and thickness panels')
  const center = meanRegion(rgba, 96, 96, 40, 40, 56, 56)
  assert.ok(center.r > 100 && center.g > 100 && center.b > 100, `AttenuationTest center panels should render visible transmission output (${center.r}, ${center.g}, ${center.b})`)
})
