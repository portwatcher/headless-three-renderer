import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'
import { Renderer, SAMPLE_ASSET_COMPARE_BASE_COLOR, SAMPLE_ASSET_COMPARE_CLEARCOAT, SAMPLE_ASSET_COMPARE_DISPERSION, SAMPLE_ASSET_COMPARE_EMISSIVE_STRENGTH, SAMPLE_ASSET_DISPERSION_TEST, SAMPLE_ASSET_DRAGON_DISPERSION } from './gltf.test.part-001.mjs'
import { assertVectorClose, loadGltfFixture, pngDimensions } from './gltf.test.part-028.mjs'
test('committed Khronos glTF Sample Assets CompareBaseColor fixture loads base-color comparison variants', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_COMPARE_BASE_COLOR, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_texture_transform'])
  assert.equal(source.buffers[0].uri, 'CompareBasecolor.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'Compare_Basecolor_img0.png',
    'Compare_Basecolor_img1.png',
  ])
  assert.deepEqual(source.materials.map((material) => [
    material.name,
    material.pbrMetallicRoughness?.baseColorTexture?.index ?? null,
    material.emissiveTexture?.index ?? null,
  ]), [
    ['baseColor plain dielectric', null, 0],
    ['baseColor texture dielectric', 1, 0],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_COMPARE_BASE_COLOR)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), ['Sphere001', 'Sphere002', 'Sphere003'])
  assert.deepEqual(meshes.map((mesh) => mesh.material.name), [
    'baseColor plain dielectric',
    'baseColor texture dielectric',
    'baseColor texture dielectric',
  ])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [9216, 9216, 9216])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [9216, 9216, 9216])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('color')?.count ?? null), [null, null, 9216])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [9216, 9216, 9216])

  const [plain, textured, vertexColored] = meshes
  assert.equal(plain.material.vertexColors, false)
  assert.equal(textured.material.vertexColors, false)
  assert.equal(vertexColored.material.vertexColors, true)
  assertVectorClose(plain.material.color.toArray(), [
    0.23882800340652466,
    0.10615606606006622,
    0.0477757565677166,
  ], 'CompareBaseColor baseColorFactor')
  assert.deepEqual(textured.material.color.toArray(), [1, 1, 1])
  assert.deepEqual(vertexColored.material.color.toArray(), [1, 1, 1])

  for (const material of [plain.material, textured.material, vertexColored.material]) {
    assert.equal(material.metalness, 0)
    assert.equal(material.roughness, 0.25)
    assert.equal(Buffer.isBuffer(material.emissiveMap?.image), true, `${material.name} emissive PNG should load as an encoded Buffer`)
    assert.equal(material.emissiveMap.name, 'Compare_Basecolor_img0.png')
    assert.deepEqual(pngDimensions(material.emissiveMap.image), [2048, 1024])
    assert.equal(material.emissiveMap.colorSpace, THREE.SRGBColorSpace)
    assert.equal(material.emissiveMap.flipY, false)
    assertVectorClose(material.emissiveMap.offset.toArray(), [0.324, 0.137], `${material.name} emissive texture offset`)
    assertVectorClose(material.emissiveMap.repeat.toArray(), [0.349, 0.725], `${material.name} emissive texture scale`)
  }

  for (const material of [textured.material, vertexColored.material]) {
    assert.equal(Buffer.isBuffer(material.map?.image), true, `${material.name} base-color PNG should load as an encoded Buffer`)
    assert.equal(material.map.name, 'Compare_Basecolor_img1.png')
    assert.deepEqual(pngDimensions(material.map.image), [512, 512])
    assert.equal(material.map.colorSpace, THREE.SRGBColorSpace)
    assert.equal(material.map.flipY, false)
    assertVectorClose(material.map.offset.toArray(), [0.25, 0.25], `${material.name} base-color texture offset`)
    assertVectorClose(material.map.repeat.toArray(), [0.5, 0.5], `${material.name} base-color texture scale`)
  }
  assert.equal(plain.material.map ?? null, null)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.7))
  const light = new THREE.DirectionalLight(0xffffff, 1.5)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)
  const camera = new THREE.PerspectiveCamera(35, 1.5, 0.01, 20)
  camera.position.copy(center).add(new THREE.Vector3(0, -5, 2))
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.1, 'CompareBaseColor should render visible base-color comparison spheres')
})

test('committed Khronos glTF Sample Assets CompareClearcoat fixture loads clearcoat comparison variants', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_COMPARE_CLEARCOAT, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_clearcoat', 'KHR_materials_ior'])
  assert.equal(source.buffers[0].uri, 'CompareClearcoat.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'Compare_Clearcoat_img0.jpg',
    'Compare_Clearcoat_img1.jpg',
  ])
  assert.deepEqual(source.materials.map((material) => [
    material.name,
    material.pbrMetallicRoughness?.baseColorTexture?.index ?? null,
    material.pbrMetallicRoughness?.metallicRoughnessTexture?.index ?? null,
    material.extensions?.KHR_materials_clearcoat?.clearcoatFactor ?? null,
    material.extensions?.KHR_materials_ior?.ior ?? null,
  ]), [
    ['green glossy', 0, 1, null, null],
    ['green rough', 0, 1, null, null],
    ['green clearcoat', 0, 1, 1, 1.6],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_COMPARE_CLEARCOAT)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), ['GeoSphere001', 'GeoSphere002', 'GeoSphere003'])
  assert.deepEqual(meshes.map((mesh) => mesh.material.name), ['green glossy', 'green rough', 'green clearcoat'])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [673, 673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [673, 673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [673, 673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [3840, 3840, 3840])

  const [glossy, rough, clearcoat] = meshes.map((mesh) => mesh.material)
  assert.equal(glossy.isMeshStandardMaterial, true)
  assert.equal(rough.isMeshStandardMaterial, true)
  assert.equal(clearcoat.isMeshPhysicalMaterial, true)
  assert.equal(glossy.roughness, 0)
  assert.equal(rough.roughness, 0.5)
  assert.equal(clearcoat.roughness, 0.5)
  assert.equal(clearcoat.clearcoat, 1)
  assert.equal(clearcoat.clearcoatRoughness, 0)
  assert.equal(clearcoat.ior, 1.6)
  assert.ok([glossy, rough, clearcoat].every((material) => material.metalness === 1))

  assert.equal(glossy.map, rough.map)
  assert.equal(rough.map, clearcoat.map)
  assert.equal(Buffer.isBuffer(glossy.map?.image), true, 'clearcoat base-color JPEG should load as an encoded Buffer')
  assert.equal(glossy.map.name, 'Compare_Clearcoat_img0.jpg')
  assert.equal(glossy.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(glossy.map.flipY, false)

  assert.equal(glossy.roughnessMap, glossy.metalnessMap)
  assert.equal(rough.roughnessMap, rough.metalnessMap)
  assert.equal(clearcoat.roughnessMap, clearcoat.metalnessMap)
  assert.equal(glossy.roughnessMap, rough.roughnessMap)
  assert.equal(rough.roughnessMap, clearcoat.roughnessMap)
  assert.equal(Buffer.isBuffer(glossy.roughnessMap?.image), true, 'clearcoat metallic-roughness JPEG should load as an encoded Buffer')
  assert.equal(glossy.roughnessMap.name, 'Compare_Clearcoat_img1.jpg')
  assert.equal(glossy.roughnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(glossy.roughnessMap.flipY, false)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1))
  const light = new THREE.DirectionalLight(0xffffff, 4)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)
  const camera = new THREE.PerspectiveCamera(35, 1.7, 0.01, 20)
  camera.position.copy(center).add(new THREE.Vector3(0, -3.8, 1.6))
  camera.lookAt(center)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 160,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.SRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.18, 'CompareClearcoat should render visible clearcoat comparison spheres')
})

test('committed Khronos glTF Sample Assets CompareDispersion fixture loads dispersion comparison variants', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_COMPARE_DISPERSION, 'utf8'))
  assert.deepEqual(source.extensionsUsed, [
    'KHR_materials_transmission',
    'KHR_materials_volume',
    'KHR_materials_ior',
    'KHR_materials_dispersion',
  ])
  assert.equal(source.buffers[0].uri, 'CompareDispersion.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'Compare_Dispersion_img0.jpg',
    'Compare_Dispersion_img1.jpg',
  ])
  assert.deepEqual(source.materials.map((material) => [
    material.name,
    material.pbrMetallicRoughness?.baseColorTexture?.index ?? null,
    material.extensions?.KHR_materials_transmission?.transmissionTexture?.index ?? null,
    material.extensions?.KHR_materials_volume?.thicknessFactor ?? null,
    material.extensions?.KHR_materials_ior?.ior ?? null,
    material.extensions?.KHR_materials_dispersion?.dispersion ?? null,
  ]), [
    ['checker', 1, null, null, null, null],
    ['No Dispersion', null, 0, 0.5, 2.42, null],
    ['Dispersion', null, 0, 0.5, 2.42, 5],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_COMPARE_DISPERSION)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), ['Checker', 'GeoSphere001', 'GeoSphere002'])
  assert.deepEqual(meshes.map((mesh) => mesh.material.name), ['checker', 'No Dispersion', 'Dispersion'])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [6, 96, 96])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [6, 96, 96])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [6, 96, 96])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [6, 96, 96])

  const [checker, noDispersion, dispersion] = meshes.map((mesh) => mesh.material)
  assert.equal(checker.isMeshStandardMaterial, true)
  assert.equal(noDispersion.isMeshPhysicalMaterial, true)
  assert.equal(dispersion.isMeshPhysicalMaterial, true)
  assert.equal(noDispersion.transmission, 1)
  assert.equal(dispersion.transmission, 1)
  assert.equal(noDispersion.thickness, 0.5)
  assert.equal(dispersion.thickness, 0.5)
  assert.equal(noDispersion.attenuationDistance, 1)
  assert.equal(dispersion.attenuationDistance, 1)
  assert.equal(noDispersion.ior, 2.42)
  assert.equal(dispersion.ior, 2.42)
  assert.equal(noDispersion.dispersion, 0)
  assert.equal(dispersion.dispersion, 5)
  assert.equal(noDispersion.roughness, 0.1)
  assert.equal(dispersion.roughness, 0.1)

  assert.equal(noDispersion.transmissionMap, dispersion.transmissionMap)
  assert.equal(Buffer.isBuffer(noDispersion.transmissionMap?.image), true, 'dispersion transmission JPEG should load as an encoded Buffer')
  assert.equal(noDispersion.transmissionMap.name, 'Compare_Dispersion_img0.jpg')
  assert.equal(noDispersion.transmissionMap.colorSpace, THREE.NoColorSpace)
  assert.equal(noDispersion.transmissionMap.flipY, false)

  assert.equal(Buffer.isBuffer(checker.map?.image), true, 'dispersion checker JPEG should load as an encoded Buffer')
  assert.equal(checker.map.name, 'Compare_Dispersion_img1.jpg')
  assert.equal(checker.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(checker.map.flipY, false)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1))
  const light = new THREE.DirectionalLight(0xffffff, 3)
  light.position.set(2, 4, 5)
  gltf.scene.add(light)
  const camera = new THREE.PerspectiveCamera(35, 1.4, 0.01, 20)
  camera.position.copy(center).add(new THREE.Vector3(0, -2.6, 1.2))
  camera.lookAt(center)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 128,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.SRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.6, 'CompareDispersion should render visible dispersion comparison geometry')
})

test('committed Khronos glTF Sample Assets DispersionTest fixture loads IOR and dispersion prism grid', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_DISPERSION_TEST, 'utf8'))
  assert.deepEqual(source.extensionsUsed, [
    'KHR_materials_transmission',
    'KHR_materials_volume',
    'KHR_materials_ior',
    'KHR_materials_dispersion',
  ])
  assert.deepEqual(source.buffers, [{ byteLength: 2276324, uri: 'DispersionTest.bin' }])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'CheckerWithLines.png',
    'Dispersion_Labels2.png',
  ])
  assert.deepEqual(source.samplers, [{ magFilter: 9729, minFilter: 9987 }])
  assert.equal(source.materials.length, 27)
  assert.equal(source.meshes.length, 27)

  const prismMaterials = source.materials.filter((material) => material.extensions?.KHR_materials_dispersion)
  assert.equal(prismMaterials.length, 25)
  assert.deepEqual([...new Set(prismMaterials.map((material) => material.extensions.KHR_materials_dispersion.dispersion))].sort((a, b) => a - b), [0, 0.5, 1, 2, 5])
  assert.deepEqual([...new Set(prismMaterials.map((material) => Number((material.extensions.KHR_materials_ior?.ior ?? 1.5).toFixed(2))))].sort((a, b) => a - b), [1, 1.33, 1.5, 1.76, 2.42])
  assert.ok(prismMaterials.every((material) => material.extensions?.KHR_materials_transmission?.transmissionFactor === 1))
  assert.ok(prismMaterials.every((material) => material.extensions?.KHR_materials_volume?.thicknessFactor === 0.018478000536561012))

  const gltf = await loadGltfFixture(SAMPLE_ASSET_DISPERSION_TEST)
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_dispersion'))
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 27)

  const prismMeshes = meshes.filter((mesh) => mesh.name.startsWith('Prism_IOR'))
  assert.equal(prismMeshes.length, 25)
  assert.ok(prismMeshes.every((mesh) => mesh.material.isMeshPhysicalMaterial === true), 'all dispersion prisms should use MeshPhysicalMaterial')
  assert.deepEqual([...new Set(prismMeshes.map((mesh) => mesh.material.dispersion))].sort((a, b) => a - b), [0, 0.5, 1, 2, 5])
  assert.deepEqual([...new Set(prismMeshes.map((mesh) => Number(mesh.material.ior.toFixed(2))))].sort((a, b) => a - b), [1, 1.33, 1.5, 1.76, 2.42])
  assert.ok(prismMeshes.every((mesh) => mesh.material.transmission === 1))
  assert.ok(prismMeshes.every((mesh) => mesh.material.thickness === 0.018478000536561012))

  const firstPrism = prismMeshes[0]
  assert.equal(firstPrism.geometry.getAttribute('position')?.count, 162)
  assert.equal(firstPrism.geometry.getAttribute('normal')?.count, 162)
  assert.equal(firstPrism.geometry.index?.count, 960)

  const backdrop = meshes.find((mesh) => mesh.name === 'Cloth_Backdrop')
  assert.equal(backdrop?.material.name, 'Cloth Backdrop')
  assert.equal(Buffer.isBuffer(backdrop.material.map?.image), true, 'dispersion checker PNG should load as an encoded Buffer')
  assert.equal(backdrop.material.map.name, 'CheckerWithLines')
  assert.deepEqual(pngDimensions(backdrop.material.map.image), [256, 256])
  assert.equal(backdrop.material.map.colorSpace, THREE.SRGBColorSpace)

  const label = meshes.find((mesh) => mesh.name === 'IOR_Labels')
  assert.equal(label?.material.name, 'LabelMat')
  assert.equal(Buffer.isBuffer(label.material.map?.image), true, 'dispersion label PNG should load as an encoded Buffer')
  assert.equal(label.material.map.name, 'Dispersion_Labels2')
  assert.deepEqual(pngDimensions(label.material.map.image), [512, 512])
  assert.equal(label.material.map.colorSpace, THREE.SRGBColorSpace)

  const camera = new THREE.OrthographicCamera(-0.09, 0.09, 0.065, -0.055, 0.001, 5)
  camera.position.set(0, 0, 0.6)
  camera.lookAt(0, 0, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.7))
  const light = new THREE.DirectionalLight(0xffffff, 2)
  light.position.set(1, 2, 3)
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 160,
    height: 120,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.55, 'DispersionTest should render visible IOR and dispersion grid geometry')
})

test('committed Khronos glTF Sample Assets DragonDispersion fixture loads real dispersion dragon scene', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_DRAGON_DISPERSION, 'utf8'))
  assert.deepEqual(source.extensionsUsed, [
    'KHR_materials_transmission',
    'KHR_materials_volume',
    'KHR_materials_dispersion',
    'KHR_materials_ior',
  ])
  assert.deepEqual(source.buffers, [
    { byteLength: 5819636, uri: 'DragonDispersion.bin' },
  ])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'Dragon_ThicknessMap.jpg',
    'CheckerWithLines.png',
  ])
  assert.deepEqual(source.materials.map((material) => material.name), [
    'Dragon with Attenuation',
    'Cloth Backdrop',
  ])
  assert.deepEqual(source.materials[0].extensions, {
    KHR_materials_transmission: { transmissionFactor: 1 },
    KHR_materials_volume: {
      attenuationColor: [0.75, 0.8, 0.82],
      attenuationDistance: 0.1549999988913536,
      thicknessFactor: 2.2699999809265137,
      thicknessTexture: { index: 0 },
    },
    KHR_materials_dispersion: { dispersion: 2.04 },
    KHR_materials_ior: { ior: 1.75 },
  })

  const gltf = await loadGltfFixture(SAMPLE_ASSET_DRAGON_DISPERSION)
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_dispersion'))
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'Dragon',
    'Cloth_Backdrop',
  ])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [76809, 62640])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [76809, 62640])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [76809, 62640])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [273648, 131337])

  const materials = new Map(meshes.map((mesh) => [mesh.material.name, mesh.material]))
  const dragon = materials.get('Dragon with Attenuation')
  assert.equal(dragon?.isMeshPhysicalMaterial, true)
  assert.equal(dragon.metalness, 0)
  assert.equal(dragon.roughness, 0)
  assert.equal(dragon.transmission, 1)
  assert.equal(dragon.thickness, 2.2699999809265137)
  assert.equal(dragon.attenuationDistance, 0.1549999988913536)
  assert.deepEqual(dragon.attenuationColor.toArray(), [0.75, 0.8, 0.82])
  assert.equal(dragon.dispersion, 2.04)
  assert.equal(dragon.ior, 1.75)
  assert.equal(dragon.thicknessMap.name, 'Dragon_ThicknessMap')
  assert.equal(Buffer.isBuffer(dragon.thicknessMap.image), true, 'dragon dispersion thickness JPEG should load as an encoded Buffer')
  assert.equal(dragon.thicknessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(dragon.thicknessMap.flipY, false)

  const backdrop = materials.get('Cloth Backdrop')
  assert.equal(backdrop?.isMeshStandardMaterial, true)
  assert.equal(backdrop.map.name, 'CheckerWithLines')
  assert.equal(Buffer.isBuffer(backdrop.map.image), true, 'dragon dispersion checker PNG should load as an encoded Buffer')
  assert.deepEqual(pngDimensions(backdrop.map.image), [256, 256])
  assert.equal(backdrop.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(backdrop.map.flipY, false)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100)
  camera.position.copy(center).add(new THREE.Vector3(0, size.y * 0.12, Math.max(size.x, size.y, size.z) * 2.2))
  camera.lookAt(center)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.75))
  const light = new THREE.DirectionalLight(0xffffff, 2.6)
  light.position.copy(center).add(new THREE.Vector3(2, 3, 4))
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.15, 'DragonDispersion should render visible dispersion dragon and backdrop')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 10 && mean.g > 10 && mean.b > 10, `DragonDispersion should render lit dragon pixels (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets CompareEmissiveStrength fixture loads emissive strength variants', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_COMPARE_EMISSIVE_STRENGTH, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_emissive_strength'])
  assert.equal(source.buffers[0].uri, 'CompareEmissiveStrength.bin')
  assert.deepEqual(source.images.map((image) => image.uri), ['Compare_Emissive-Strength_img0.jpg'])
  assert.deepEqual(source.materials.map((material) => [
    material.name,
    material.emissiveTexture?.index ?? null,
    material.extensions?.KHR_materials_emissive_strength?.emissiveStrength ?? null,
  ]), [
    ['glTF Logo Emissive', 0, null],
    ['glTF Logo Emissive Strength', 0, 3],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_COMPARE_EMISSIVE_STRENGTH)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), ['GeoSphere001', 'GeoSphere002'])
  assert.deepEqual(meshes.map((mesh) => mesh.material.name), ['glTF Logo Emissive', 'glTF Logo Emissive Strength'])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [3840, 3840])

  const [baseline, strengthened] = meshes.map((mesh) => mesh.material)
  assert.deepEqual(baseline.color.toArray(), [0, 0, 0])
  assert.deepEqual(strengthened.color.toArray(), [0, 0, 0])
  assert.deepEqual(baseline.emissive.toArray(), [1, 1, 1])
  assert.deepEqual(strengthened.emissive.toArray(), [1, 1, 1])
  assert.equal(baseline.emissiveIntensity, 1)
  assert.equal(strengthened.emissiveIntensity, 3)
  assert.equal(baseline.emissiveMap, strengthened.emissiveMap, 'both emissive-strength materials should share the emissive texture')
  assert.equal(Buffer.isBuffer(baseline.emissiveMap?.image), true, 'emissive JPEG should load as an encoded Buffer')
  assert.equal(baseline.emissiveMap.name, 'Compare_Emissive-Strength_img0.jpg')
  assert.equal(baseline.emissiveMap.colorSpace, THREE.SRGBColorSpace)
  assert.equal(baseline.emissiveMap.flipY, false)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.02, 'CompareEmissiveStrength should render visible emissive comparison spheres')
})
