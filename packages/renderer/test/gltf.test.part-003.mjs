import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'
import { Renderer, SAMPLE_ASSET_ALPHA_BLEND_MODE_TEST, SAMPLE_ASSET_COMPARE_ALPHA_COVERAGE, SAMPLE_ASSET_COMPARE_AMBIENT_OCCLUSION, SAMPLE_ASSET_COMPARE_ANISOTROPY, SAMPLE_ASSET_FLIGHT_HELMET } from './gltf.test.part-001.mjs'
import { assertVectorClose, loadGltfFixture, meanRegion, pngDimensions } from './gltf.test.part-028.mjs'
test('committed Khronos glTF Sample Assets FlightHelmet fixture loads transmission lens and PBR texture sets', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_FLIGHT_HELMET, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_transmission'])
  assert.deepEqual(source.buffers, [{ uri: 'FlightHelmet.bin', byteLength: 3227148 }])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'FlightHelmet_Materials_RubberWoodMat_Normal.png',
    'FlightHelmet_Materials_RubberWoodMat_OcclusionRoughMetal.png',
    'FlightHelmet_Materials_RubberWoodMat_BaseColor.png',
    'FlightHelmet_Materials_GlassPlasticMat_Normal.png',
    'FlightHelmet_Materials_GlassPlasticMat_OcclusionRoughMetal.png',
    'FlightHelmet_Materials_GlassPlasticMat_BaseColor.png',
    'FlightHelmet_Materials_MetalPartsMat_Normal.png',
    'FlightHelmet_Materials_MetalPartsMat_OcclusionRoughMetal.png',
    'FlightHelmet_Materials_MetalPartsMat_BaseColor.png',
    'FlightHelmet_Materials_LeatherPartsMat_Normal.png',
    'FlightHelmet_Materials_LeatherPartsMat_OcclusionRoughMetal.png',
    'FlightHelmet_Materials_LeatherPartsMat_BaseColor.png',
    'FlightHelmet_Materials_LensesMat_Normal.png',
    'FlightHelmet_Materials_LensesMat_OcclusionRoughMetal.png',
    'FlightHelmet_Materials_LensesMat_BaseColor.png',
  ])
  assert.deepEqual(source.materials.map((material) => material.name), [
    'HoseMat',
    'RubberWoodMat',
    'GlassPlasticMat',
    'MetalPartsMat',
    'LeatherPartsMat',
    'LensesMat',
  ])
  assert.deepEqual(source.materials[5].extensions, {
    KHR_materials_transmission: {
      transmissionFactor: 1,
    },
  })
  assert.deepEqual(source.meshes.map((mesh) => mesh.name), [
    'Hose_low',
    'RubberWood_low',
    'GlassPlastic_low',
    'MetalParts_low',
    'LeatherParts_low',
    'Lenses_low',
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_FLIGHT_HELMET)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'Hose_low',
    'RubberWood_low',
    'GlassPlastic_low',
    'MetalParts_low',
    'LeatherParts_low',
    'Lenses_low',
  ])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [10472, 13638, 4676, 13636, 12534, 436])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('tangent')?.count), [10472, 13638, 4676, 13636, 12534, 436])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [59040, 72534, 24408, 60288, 65688, 2208])

  const materials = new Map(meshes.map((mesh) => [mesh.material.name, mesh.material]))
  const assertTextureSet = (material, prefix, dimensions) => {
    assert.equal(material.map.name, `${prefix}_BaseColor.png`)
    assert.equal(material.roughnessMap.name, `${prefix}_OcclusionRoughMetal.png`)
    assert.equal(material.metalnessMap, material.roughnessMap)
    assert.equal(material.aoMap, material.roughnessMap)
    assert.equal(material.normalMap.name, `${prefix}_Normal.png`)
    assert.equal(Buffer.isBuffer(material.map.image), true, `${prefix} base-color PNG should load as an encoded Buffer`)
    assert.deepEqual(pngDimensions(material.map.image), dimensions)
    assert.deepEqual(pngDimensions(material.roughnessMap.image), dimensions)
    assert.deepEqual(pngDimensions(material.normalMap.image), dimensions)
    assert.equal(material.map.colorSpace, THREE.SRGBColorSpace)
    assert.equal(material.roughnessMap.colorSpace, THREE.NoColorSpace)
    assert.equal(material.normalMap.colorSpace, THREE.NoColorSpace)
    assert.equal(material.map.flipY, false)
  }

  const hose = materials.get('HoseMat')
  const rubber = materials.get('RubberWoodMat')
  assertTextureSet(hose, 'FlightHelmet_Materials_RubberWoodMat', [2048, 2048])
  assertTextureSet(rubber, 'FlightHelmet_Materials_RubberWoodMat', [2048, 2048])
  assert.equal(hose.map, rubber.map, 'FlightHelmet HoseMat should share RubberWood base-color texture')
  assertTextureSet(materials.get('GlassPlasticMat'), 'FlightHelmet_Materials_GlassPlasticMat', [2048, 2048])
  assertTextureSet(materials.get('MetalPartsMat'), 'FlightHelmet_Materials_MetalPartsMat', [2048, 2048])
  assertTextureSet(materials.get('LeatherPartsMat'), 'FlightHelmet_Materials_LeatherPartsMat', [2048, 2048])

  const lenses = materials.get('LensesMat')
  assert.equal(lenses.isMeshPhysicalMaterial, true)
  assert.equal(lenses.transmission, 1)
  assertTextureSet(lenses, 'FlightHelmet_Materials_LensesMat', [1024, 1024])

  gltf.scene.updateMatrixWorld(true)
  const box = new THREE.Box3().setFromObject(gltf.scene)
  const center = box.getCenter(new THREE.Vector3())
  const renderCamera = new THREE.PerspectiveCamera(35, 1, 0.001, 10)
  renderCamera.position.copy(center).add(new THREE.Vector3(0.8, 0.35, 1).normalize().multiplyScalar(1.0))
  renderCamera.lookAt(center)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.85))
  const light = new THREE.DirectionalLight(0xffffff, 1.9)
  light.position.copy(center).add(new THREE.Vector3(0.5, 0.8, 1))
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)
  renderCamera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, renderCamera, {
    width: 128,
    height: 128,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.3, 'FlightHelmet should render visible textured helmet geometry')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > mean.g && mean.g > mean.b, `FlightHelmet textures should render warm leather/metal colors (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets AlphaBlendModeTest fixture loads alpha modes and JPEG textures', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_ALPHA_BLEND_MODE_TEST)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 9)

  const meshesByName = new Map(meshes.map((mesh) => [mesh.name, mesh]))
  assert.equal(meshesByName.get('TestBlend')?.material.transparent, true)
  assert.equal(meshesByName.get('DecalBlend')?.material.transparent, true)
  assert.equal(meshesByName.get('TestOpaque')?.material.transparent, false)
  assert.equal(meshesByName.get('TestCutoff25')?.material.alphaTest, 0.25)
  assert.equal(meshesByName.get('TestCutoffDefault')?.material.alphaTest, 0.5)
  assert.equal(meshesByName.get('TestCutoff75')?.material.alphaTest, 0.75)
  assert.ok(
    meshes.every((mesh) => Buffer.isBuffer(mesh.material.map?.image)),
    'AlphaBlendModeTest PNG and JPEG material textures should load as encoded Buffers',
  )

  const bed = meshesByName.get('Bed')
  assert.ok(Buffer.isBuffer(bed?.material.normalMap?.image), 'AlphaBlendModeTest JPEG normal map should load as an encoded Buffer')
  assert.ok(Buffer.isBuffer(bed?.material.aoMap?.image), 'AlphaBlendModeTest JPEG ORM map should load as an encoded Buffer')
  assert.ok(Buffer.isBuffer(bed?.material.roughnessMap?.image), 'AlphaBlendModeTest JPEG roughness map should load as an encoded Buffer')
  assert.ok(Buffer.isBuffer(bed?.material.metalnessMap?.image), 'AlphaBlendModeTest JPEG metalness map should load as an encoded Buffer')

  const camera = new THREE.PerspectiveCamera(35, 4 / 3, 0.01, 50)
  camera.position.set(0, 1.4, 8)
  camera.lookAt(0, 0.8, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1.0))
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 160,
    height: 120,
    format: 'rgba',
    background: [0.04, 0.04, 0.04],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [10, 10, 10], 4) > 0.4, 'AlphaBlendModeTest should render visible alpha-mode geometry')
  const center = meanRegion(rgba, 160, 120, 60, 40, 100, 80)
  assert.ok(center.r > 80 && center.g > 80 && center.b > 70, `AlphaBlendModeTest render should include the textured material bed (${center.r}, ${center.g}, ${center.b})`)
})

test('committed Khronos glTF Sample Assets CompareAlphaCoverage fixture loads alpha coverage material variants', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_COMPARE_ALPHA_COVERAGE, 'utf8'))
  assert.equal(source.buffers[0].uri, 'CompareAlphaCoverage.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'FurBaseColorAlpha.png',
    'FurNormal.png',
    'FurEmissive.jpg',
    'FurORM.jpg',
    'FloorBaseColor.jpg',
    'FloorNormal.jpg',
  ])
  assert.deepEqual(source.materials.map((material) => [material.name, material.alphaMode ?? 'OPAQUE', material.alphaCutoff ?? null]), [
    ['fur_opaque', 'OPAQUE', null],
    ['fur floor', 'OPAQUE', null],
    ['fur_mask', 'MASK', 0.2],
    ['fur_blend', 'BLEND', null],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_COMPARE_ALPHA_COVERAGE)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'Fur001_0',
    'Fur001_1',
    'Fur002_0',
    'Fur002_1',
    'Fur003_0',
    'Fur003_1',
  ])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [32, 4, 32, 4, 4, 32])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [48, 6, 48, 6, 6, 48])

  const materials = new Map(meshes.map((mesh) => [mesh.material.name, mesh.material]))
  const opaque = materials.get('fur_opaque')
  const mask = materials.get('fur_mask')
  const blend = materials.get('fur_blend')
  const floor = materials.get('fur floor')
  assert.equal(opaque?.transparent, false)
  assert.equal(mask?.transparent, false)
  assert.equal(mask.alphaTest, 0.2)
  assert.equal(blend?.transparent, true)
  assert.equal(blend.alphaTest, 0)
  assert.equal(floor?.transparent, false)
  assert.ok([opaque, mask, blend, floor].every((material) => material.side === THREE.DoubleSide))

  for (const material of [opaque, mask, blend]) {
    assert.equal(Buffer.isBuffer(material.map?.image), true, `${material.name} base color/alpha PNG should load as an encoded Buffer`)
    assert.equal(material.map.name, 'FurBaseColorAlpha.png')
    assert.deepEqual(pngDimensions(material.map.image), [1024, 1024])
    assert.equal(material.map.colorSpace, THREE.SRGBColorSpace)
    assert.equal(material.map.flipY, false)

    assert.equal(Buffer.isBuffer(material.normalMap?.image), true, `${material.name} normal PNG should load as an encoded Buffer`)
    assert.equal(material.normalMap.name, 'FurNormal.png')
    assert.equal(material.normalMap.colorSpace, THREE.NoColorSpace)
    assert.equal(material.normalMap.flipY, false)

    assert.equal(Buffer.isBuffer(material.emissiveMap?.image), true, `${material.name} emissive JPEG should load as an encoded Buffer`)
    assert.equal(material.emissiveMap.name, 'FurEmissive.jpg')
    assert.equal(material.emissiveMap.colorSpace, THREE.SRGBColorSpace)
    assert.equal(material.emissiveMap.flipY, false)

    assert.equal(material.roughnessMap, material.metalnessMap, `${material.name} roughness and metalness should share the packed ORM texture`)
    assert.equal(material.aoMap, material.roughnessMap, `${material.name} occlusion and roughness should share the packed ORM texture`)
    assert.equal(Buffer.isBuffer(material.roughnessMap?.image), true, `${material.name} ORM JPEG should load as an encoded Buffer`)
    assert.equal(material.roughnessMap.name, 'FurORM.jpg')
    assert.equal(material.roughnessMap.colorSpace, THREE.NoColorSpace)
    assert.equal(material.roughnessMap.flipY, false)
  }

  assert.equal(Buffer.isBuffer(floor.map?.image), true, 'floor base color JPEG should load as an encoded Buffer')
  assert.equal(floor.map.name, 'FloorBaseColor.jpg')
  assert.equal(floor.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(floor.map.flipY, false)
  assert.equal(Buffer.isBuffer(floor.normalMap?.image), true, 'floor normal JPEG should load as an encoded Buffer')
  assert.equal(floor.normalMap.name, 'FloorNormal.jpg')
  assert.equal(floor.normalMap.colorSpace, THREE.NoColorSpace)
  assert.equal(floor.normalMap.flipY, false)

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 1.6)
  light.position.set(2, 4, 5)
  gltf.scene.add(light)
  const camera = new THREE.PerspectiveCamera(35, 10 / 7, 0.01, 20)
  camera.position.set(0, -7, 3.5)
  camera.lookAt(0, 0, 1)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 160,
    height: 112,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.03, 'CompareAlphaCoverage should render visible alpha coverage panels')
})

test('committed Khronos glTF Sample Assets CompareAmbientOcclusion fixture loads AO material pairs', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_COMPARE_AMBIENT_OCCLUSION, 'utf8'))
  assert.equal(source.buffers[0].uri, 'CompareAmbientOcclusion_data.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'FruitBaseColor.jpg',
    'FruitORM.jpg',
    'BasketORM.jpg',
  ])
  assert.deepEqual(source.materials.map((material) => [
    material.name,
    material.occlusionTexture?.index ?? null,
    material.pbrMetallicRoughness?.metallicRoughnessTexture?.index ?? null,
  ]), [
    ['Basket with Occlusion', 2, null],
    ['Fruit with Occlusion', 1, 1],
    ['Logo', null, null],
    ['Basket without Occlusion', null, null],
    ['Fruit without Occlusion', null, 1],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_COMPARE_AMBIENT_OCCLUSION)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'BasketRight',
    'FruitRight',
    'LogoRight',
    'BasketLeft',
    'FruitLeft',
    'LogoLeft',
  ])
  assert.deepEqual(meshes.map((mesh) => mesh.material.name), [
    'Basket with Occlusion',
    'Fruit with Occlusion',
    'Logo',
    'Basket without Occlusion',
    'Fruit without Occlusion',
    'Logo',
  ])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [17832, 28918, 1605, 11240, 28918, 1605])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [66828, 117600, 2865, 66828, 117600, 2865])

  const materials = new Map(meshes.map((mesh) => [mesh.material.name, mesh.material]))
  const basketWithAo = materials.get('Basket with Occlusion')
  const basketWithoutAo = materials.get('Basket without Occlusion')
  const fruitWithAo = materials.get('Fruit with Occlusion')
  const fruitWithoutAo = materials.get('Fruit without Occlusion')
  assert.equal(Buffer.isBuffer(basketWithAo?.aoMap?.image), true, 'basket AO JPEG should load as an encoded Buffer')
  assert.equal(basketWithAo.aoMap.name, 'BasketORM.jpg')
  assert.equal(basketWithAo.aoMap.colorSpace, THREE.NoColorSpace)
  assert.equal(basketWithAo.aoMap.flipY, false)
  assert.equal(basketWithoutAo?.aoMap ?? null, null)

  for (const material of [fruitWithAo, fruitWithoutAo]) {
    assert.equal(Buffer.isBuffer(material.map?.image), true, `${material.name} base color JPEG should load as an encoded Buffer`)
    assert.equal(material.map.name, 'FruitBaseColor.jpg')
    assert.equal(material.map.colorSpace, THREE.SRGBColorSpace)
    assert.equal(material.map.flipY, false)
    assert.equal(material.roughnessMap, material.metalnessMap, `${material.name} roughness and metalness should share the packed ORM texture`)
    assert.equal(Buffer.isBuffer(material.roughnessMap?.image), true, `${material.name} ORM JPEG should load as an encoded Buffer`)
    assert.equal(material.roughnessMap.name, 'FruitORM.jpg')
    assert.equal(material.roughnessMap.colorSpace, THREE.NoColorSpace)
    assert.equal(material.roughnessMap.flipY, false)
  }
  assert.equal(fruitWithAo.aoMap, fruitWithAo.roughnessMap, 'fruit AO should share the packed ORM texture when occlusion is enabled')
  assert.equal(fruitWithoutAo.aoMap ?? null, null)

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.55))
  const light = new THREE.DirectionalLight(0xffffff, 1.8)
  light.position.set(1.5, 3, 4)
  gltf.scene.add(light)
  const camera = new THREE.PerspectiveCamera(35, 1.5, 0.01, 10)
  camera.position.set(0, -1.4, 0.65)
  camera.lookAt(0, 0, 0.05)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 144,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.08, 'CompareAmbientOcclusion should render visible paired AO samples')
})

test('committed Khronos glTF Sample Assets CompareAnisotropy fixture loads anisotropy comparison variants', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_COMPARE_ANISOTROPY, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_texture_transform', 'KHR_materials_anisotropy'])
  assert.equal(source.buffers[0].uri, 'CompareAnisotropy.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'Compare_Anisotropy_img0.jpg',
    'Compare_Anisotropy_img1.png',
    'Compare_Anisotropy_img2.jpg',
    'Compare_Anisotropy_img3.png',
    'Compare_Anisotropy_img4.png',
  ])
  assert.deepEqual(source.materials.map((material) => [
    material.name,
    material.pbrMetallicRoughness?.baseColorTexture?.index ?? null,
    material.pbrMetallicRoughness?.metallicRoughnessTexture?.index ?? null,
    material.extensions?.KHR_materials_anisotropy?.anisotropyStrength ?? null,
    material.extensions?.KHR_materials_anisotropy?.anisotropyTexture?.index ?? null,
  ]), [
    ['grooved-anisotropy', 1, 3, 0.5, null],
    ['spiral-anisotropy', null, null, 0.5, 4],
    ['grooved', 1, 3, null, null],
    ['spiral', null, null, null, null],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_COMPARE_ANISOTROPY)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'Spheroid002_primitive0',
    'Spheroid002_primitive1',
    'Spheroid001_primitive0',
    'Spheroid001_primitive1',
  ])
  assert.deepEqual(meshes.map((mesh) => mesh.material.name), [
    'grooved-anisotropy',
    'spiral-anisotropy',
    'grooved',
    'spiral',
  ])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [5313, 4258, 5313, 4258])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [5313, 4258, 5313, 4258])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [5313, 4258, 5313, 4258])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [30720, 24576, 30720, 24576])

  const [groovedAniso, spiralAniso, grooved, spiral] = meshes.map((mesh) => mesh.material)
  assert.equal(groovedAniso.isMeshPhysicalMaterial, true)
  assert.equal(spiralAniso.isMeshPhysicalMaterial, true)
  assert.equal(grooved.isMeshStandardMaterial, true)
  assert.equal(spiral.isMeshStandardMaterial, true)
  assert.equal(groovedAniso.anisotropy, 0.5)
  assert.equal(groovedAniso.anisotropyRotation, 0)
  assert.equal(groovedAniso.anisotropyMap ?? null, null)
  assert.equal(spiralAniso.anisotropy, 0.5)
  assert.equal(spiralAniso.anisotropyRotation, 0)
  assert.equal(Buffer.isBuffer(spiralAniso.anisotropyMap?.image), true, 'anisotropy direction PNG should load as an encoded Buffer')
  assert.equal(spiralAniso.anisotropyMap.name, 'Compare_Anisotropy_img4.png')
  assert.deepEqual(pngDimensions(spiralAniso.anisotropyMap.image), [256, 256])
  assert.equal(spiralAniso.anisotropyMap.colorSpace, THREE.NoColorSpace)
  assert.equal(spiralAniso.anisotropyMap.flipY, false)
  assert.equal(grooved.anisotropyMap ?? null, null)
  assert.equal(spiral.anisotropyMap ?? null, null)

  assert.equal(groovedAniso.map, grooved.map, 'grooved anisotropy pair should share the base-color texture')
  assert.equal(Buffer.isBuffer(grooved.map?.image), true, 'grooved base-color PNG should load as an encoded Buffer')
  assert.equal(grooved.map.name, 'Compare_Anisotropy_img1.png')
  assert.deepEqual(pngDimensions(grooved.map.image), [2048, 1024])
  assert.equal(grooved.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(grooved.map.flipY, false)

  assert.equal(groovedAniso.roughnessMap, groovedAniso.metalnessMap)
  assert.equal(grooved.roughnessMap, grooved.metalnessMap)
  assert.equal(groovedAniso.roughnessMap, grooved.roughnessMap)
  assert.equal(Buffer.isBuffer(grooved.roughnessMap?.image), true, 'grooved metallic-roughness PNG should load as an encoded Buffer')
  assert.equal(grooved.roughnessMap.name, 'Compare_Anisotropy_img3.png')
  assert.deepEqual(pngDimensions(grooved.roughnessMap.image), [2048, 1024])
  assert.equal(grooved.roughnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(grooved.roughnessMap.flipY, false)

  assertVectorClose(spiralAniso.color.toArray(), [
    0.5795467495918274,
    0.2715774476528168,
    0.18354901671409607,
  ], 'CompareAnisotropy spiral anisotropic baseColorFactor')
  assert.deepEqual(spiralAniso.color.toArray(), spiral.color.toArray())
  assert.equal(spiralAniso.roughness, 0.1)
  assert.equal(spiral.roughness, 0.1)
  assert.equal(spiralAniso.metalness, 1)
  assert.equal(spiral.metalness, 1)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 3)
  light.position.set(2, 4, 5)
  gltf.scene.add(light)
  const camera = new THREE.PerspectiveCamera(35, 1.6, 0.01, 20)
  camera.position.copy(center).add(new THREE.Vector3(0, -3.2, 1.4))
  camera.lookAt(center)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 160,
    height: 100,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.SRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.15, 'CompareAnisotropy should render visible anisotropy comparison geometry')
})
