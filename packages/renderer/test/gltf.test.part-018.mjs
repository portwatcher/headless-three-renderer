import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'
import { Renderer, SAMPLE_ASSET_SHEEN_TEST_GRID, SAMPLE_ASSET_SPECULAR_SILK_POUF, SAMPLE_ASSET_SPECULAR_TEST, SAMPLE_ASSET_SUZANNE } from './gltf.test.part-001.mjs'
import { findFirst, loadGltfFixture, meanRegion, pngDimensions } from './gltf.test.part-028.mjs'
test('committed Khronos glTF Sample Assets SheenTestGrid fixture loads sheen color and roughness grid factors', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_SHEEN_TEST_GRID, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_sheen'])
  assert.deepEqual(source.buffers, [
    { byteLength: 1818312, uri: 'SheenTestGrid.bin' },
  ])
  assert.deepEqual(source.images.map((image) => image.uri), ['checker.png'])

  const sourceSheenMaterials = source.materials.filter((material) => material.name.startsWith('sheen'))
  assert.deepEqual(sourceSheenMaterials.map((material) => [
    material.name,
    material.extensions?.KHR_materials_sheen?.sheenColorFactor ?? null,
    material.extensions?.KHR_materials_sheen?.sheenRoughnessFactor ?? null,
  ]), [
    ['sheenColor0_sheenRough0', [0, 0, 0], 0],
    ['sheenColor0_sheenRough0.33', [0, 0, 0], 0.33],
    ['sheenColor0_sheenRough0.66', [0, 0, 0], 0.66],
    ['sheenColor0_sheenRough1', [0, 0, 0], 1],
    ['sheenColor0.33_sheenRough0', [0, 0.33, 0.33], 0],
    ['sheenColor0.33_sheenRough0.33', [0, 0.33, 0.33], 0.33],
    ['sheenColor0.33_sheenRough0.66', [0, 0.33, 0.33], 0.66],
    ['sheenColor0.33_sheenRough1', [0, 0.33, 0.33], 1],
    ['sheenColor0.66_sheenRough0', [0, 0.66, 0.66], 0],
    ['sheenColor0.66_sheenRough0.33', [0, 0.66, 0.66], 0.33],
    ['sheenColor0.66_sheenRough0.66', [0, 0.66, 0.66], 0.66],
    ['sheenColor0.66_sheenRough1', [0, 0.66, 0.66], 1],
    ['sheenColor1_sheenRough0', [0, 1, 1], 0],
    ['sheenColor1_sheenRough0.33', [0, 1, 1], 0.33],
    ['sheenColor1_sheenRough0.66', [0, 1, 1], 0.66],
    ['sheenColor1_sheenRough1', [0, 1, 1], 1],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_SHEEN_TEST_GRID)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'CheckerBackdrop',
    'sheenColor0_sheenRough0',
    'sheenColor033_sheenRough0',
    'sheenColor066_sheenRough0',
    'sheenColor1_sheenRough0',
    'sheenColor0_sheenRough033',
    'sheenColor033_sheenRough033',
    'sheenColor066_sheenRough0_1',
    'sheenColor1_sheenRough033',
    'sheenColor0_sheenRough066',
    'sheenColor033_sheenRough066',
    'sheenColor066_sheenRough066',
    'sheenColor1_sheenRough066',
    'sheenColor0_sheenRough1',
    'sheenColor033_sheenRough1',
    'sheenColor066_sheenRough1',
    'sheenColor1_sheenRough1',
    'TextBackdrop',
    'TextXaxis',
    'TextYaxis',
  ])

  const byName = new Map(meshes.map((mesh) => [mesh.name, mesh]))
  const sampleNames = [
    'sheenColor0_sheenRough0',
    'sheenColor033_sheenRough0',
    'sheenColor066_sheenRough0',
    'sheenColor1_sheenRough0',
    'sheenColor0_sheenRough033',
    'sheenColor033_sheenRough033',
    'sheenColor066_sheenRough0_1',
    'sheenColor1_sheenRough033',
    'sheenColor0_sheenRough066',
    'sheenColor033_sheenRough066',
    'sheenColor066_sheenRough066',
    'sheenColor1_sheenRough066',
    'sheenColor0_sheenRough1',
    'sheenColor033_sheenRough1',
    'sheenColor066_sheenRough1',
    'sheenColor1_sheenRough1',
  ]
  const samples = sampleNames.map((name) => byName.get(name))
  assert.ok(samples.every(Boolean), 'SheenTestGrid should load all sheen color/roughness samples')
  assert.deepEqual(samples.map((mesh) => mesh.geometry.getAttribute('position')?.count), Array(16).fill(3840))
  assert.deepEqual(samples.map((mesh) => mesh.geometry.getAttribute('normal')?.count), Array(16).fill(3840))
  assert.deepEqual(samples.map((mesh) => mesh.geometry.index?.count), Array(16).fill(3840))

  const sampleMaterials = samples.map((mesh) => mesh.material)
  assert.ok(sampleMaterials.every((material) => material.isMeshPhysicalMaterial === true), 'all sheen-grid samples should load as MeshPhysicalMaterial')
  assert.deepEqual(sampleMaterials.map((material) => material.name), [
    'sheenColor0_sheenRough0',
    'sheenColor0.33_sheenRough0',
    'sheenColor0.66_sheenRough0',
    'sheenColor1_sheenRough0',
    'sheenColor0_sheenRough0.33',
    'sheenColor0.33_sheenRough0.33',
    'sheenColor0.66_sheenRough0.33',
    'sheenColor1_sheenRough0.33',
    'sheenColor0_sheenRough0.66',
    'sheenColor0.33_sheenRough0.66',
    'sheenColor0.66_sheenRough0.66',
    'sheenColor1_sheenRough0.66',
    'sheenColor0_sheenRough1',
    'sheenColor0.33_sheenRough1',
    'sheenColor0.66_sheenRough1',
    'sheenColor1_sheenRough1',
  ])
  assert.deepEqual(sampleMaterials.map((material) => material.sheen), Array(16).fill(1))
  assert.deepEqual(sampleMaterials.map((material) => material.sheenRoughness), [
    0, 0, 0, 0,
    0.33, 0.33, 0.33, 0.33,
    0.66, 0.66, 0.66, 0.66,
    1, 1, 1, 1,
  ])
  assert.deepEqual(sampleMaterials.map((material) => material.sheenColor.toArray()), [
    [0, 0, 0],
    [0, 0.33, 0.33],
    [0, 0.66, 0.66],
    [0, 1, 1],
    [0, 0, 0],
    [0, 0.33, 0.33],
    [0, 0.66, 0.66],
    [0, 1, 1],
    [0, 0, 0],
    [0, 0.33, 0.33],
    [0, 0.66, 0.66],
    [0, 1, 1],
    [0, 0, 0],
    [0, 0.33, 0.33],
    [0, 0.66, 0.66],
    [0, 1, 1],
  ])
  assert.ok(sampleMaterials.every((material) => material.roughness === 0.75))
  assert.ok(sampleMaterials.every((material) => material.metalness === 0))

  const checker = byName.get('CheckerBackdrop')
  assert.equal(checker.geometry.getAttribute('position')?.count, 6)
  assert.equal(checker.geometry.getAttribute('uv')?.count, 6)
  assert.equal(checker.geometry.index?.count, 6)
  assert.equal(checker.material.name, 'Backdrop')
  assert.equal(Buffer.isBuffer(checker.material.map?.image), true, 'SheenTestGrid checker PNG should load as an encoded Buffer')
  assert.equal(checker.material.map.name, 'checker.png')
  assert.deepEqual(pngDimensions(checker.material.map.image), [256, 256])
  assert.equal(checker.material.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(checker.material.map.flipY, false)

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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.35, 'SheenTestGrid should render visible sheen color and roughness samples')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 20 && mean.g > 20 && mean.b > 35, `SheenTestGrid should render lit sheen grid pixels (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets SpecularSilkPouf fixture loads real sheen and specular silk material', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_SPECULAR_SILK_POUF, 'utf8'))
  assert.deepEqual(source.extensionsRequired, ['KHR_materials_specular', 'KHR_materials_sheen'])
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_specular', 'KHR_materials_sheen'])
  assert.deepEqual(source.buffers, [
    { byteLength: 2313984, uri: 'SpecularSilkPouf.bin' },
  ])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'SpecularSilkPouf_occlusion.png',
    'SpecularSilkPouf_normal.png',
  ])
  assert.deepEqual(source.meshes.map((mesh) => mesh.name), ['SpecularSilkPouf'])
  assert.deepEqual(source.meshes[0].primitives[0].attributes, {
    NORMAL: 1,
    POSITION: 0,
    TEXCOORD_0: 2,
  })
  assert.deepEqual(source.materials[0].extensions, {
    KHR_materials_sheen: {
      sheenColorFactor: [0.025, 0.03, 0.075],
      sheenRoughnessFactor: 0.6,
    },
    KHR_materials_specular: {
      specularColorFactor: [10, 0.6, 0],
      specularFactor: 0.5,
    },
  })

  const gltf = await loadGltfFixture(SAMPLE_ASSET_SPECULAR_SILK_POUF)
  assert.deepEqual(gltf.parser?.json?.extensionsRequired, ['KHR_materials_specular', 'KHR_materials_sheen'])
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'SpecularSilkPouf should load a mesh')
  assert.equal(mesh.name, 'SpecularSilkPouf')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 41832)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 41832)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 41832)
  assert.equal(mesh.geometry.index?.count, 243840)

  const material = mesh.material
  assert.equal(material.name, 'shot silk')
  assert.equal(material.isMeshPhysicalMaterial, true)
  assert.deepEqual(material.color.toArray(), [0.025, 0.03, 0.075])
  assert.equal(material.metalness, 0)
  assert.equal(material.roughness, 0.65)
  assert.equal(material.sheen, 1)
  assert.deepEqual(material.sheenColor.toArray(), [0.025, 0.03, 0.075])
  assert.equal(material.sheenRoughness, 0.6)
  assert.equal(material.specularIntensity, 0.5)
  assert.deepEqual(material.specularColor.toArray(), [10, 0.6, 0])
  assert.equal(Buffer.isBuffer(material.aoMap?.image), true, 'SpecularSilkPouf occlusion PNG should load as an encoded Buffer')
  assert.equal(Buffer.isBuffer(material.normalMap?.image), true, 'SpecularSilkPouf normal PNG should load as an encoded Buffer')
  assert.equal(material.aoMap.name, 'SpecularSilkPouf_occlusion')
  assert.equal(material.normalMap.name, 'SpecularSilkPouf_normal')
  assert.equal(material.aoMap.colorSpace, THREE.NoColorSpace)
  assert.equal(material.normalMap.colorSpace, THREE.NoColorSpace)
  assert.equal(material.normalMap.flipY, false)
  assert.deepEqual(pngDimensions(material.aoMap.image), [512, 512])
  assert.deepEqual(pngDimensions(material.normalMap.image), [1024, 1024])

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const camera = new THREE.PerspectiveCamera(35, 1, 0.001, 20)
  camera.position.copy(center).add(new THREE.Vector3(0, 1, 0).multiplyScalar(2))
  camera.lookAt(center)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1))
  const light = new THREE.DirectionalLight(0xffffff, 3)
  light.position.copy(center).add(new THREE.Vector3(0.5, 0.8, 1))
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 128,
    height: 128,
    format: 'rgba',
    background: [1, 1, 1],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [255, 255, 255], 3) > 0.15, 'SpecularSilkPouf should render visible dark silk geometry')
  const centerSample = meanRegion(rgba, 128, 128, 48, 48, 80, 80)
  assert.ok(centerSample.b > centerSample.r + 10 && centerSample.b > centerSample.g + 10 && centerSample.b < 35, `SpecularSilkPouf center should render dark blue silk (${centerSample.r}, ${centerSample.g}, ${centerSample.b})`)
})

test('committed Khronos glTF Sample Assets SpecularTest fixture loads KHR_materials_specular scalar and texture inputs', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_SPECULAR_TEST)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 24)
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_specular'))

  const materials = new Map(meshes.map((mesh) => [mesh.material.name, mesh.material]))
  assert.equal(materials.size, 24)

  const disabled = materials.get('M1.1_specFac')
  const enabled = materials.get('M1.5_specFac')
  assert.equal(disabled?.isMeshPhysicalMaterial, true)
  assert.equal(enabled?.isMeshPhysicalMaterial, true)
  assert.equal(disabled.specularIntensity, 0)
  assert.equal(enabled.specularIntensity, 1)
  assert.deepEqual(enabled.specularColor.toArray(), [1, 1, 1])

  const specularTexture = materials.get('M2_SpecTex')
  assert.equal(specularTexture?.isMeshPhysicalMaterial, true)
  assert.equal(specularTexture.specularIntensity, 1)
  assert.equal(Buffer.isBuffer(specularTexture.specularIntensityMap?.image), true, 'specular factor PNG should load as an encoded Buffer')
  assert.equal(specularTexture.specularIntensityMap.name, 'specularTextureGrid')
  assert.equal(specularTexture.specularIntensityMap.colorSpace, THREE.NoColorSpace)
  assert.equal(specularTexture.specularIntensityMap.magFilter, THREE.LinearFilter)
  assert.equal(specularTexture.specularIntensityMap.minFilter, THREE.LinearMipmapLinearFilter)
  assert.equal(specularTexture.specularIntensityMap.flipY, false)

  const whiteTexture = materials.get('M4_whiteTex')
  assert.equal(whiteTexture?.isMeshPhysicalMaterial, true)
  assert.equal(Buffer.isBuffer(whiteTexture.specularColorMap?.image), true, 'white specular color PNG should load as an encoded Buffer')
  assert.equal(whiteTexture.specularColorMap.name, 'WhiteGrid')
  assert.equal(whiteTexture.specularColorMap.colorSpace, THREE.SRGBColorSpace)
  assert.equal(whiteTexture.specularColorMap.flipY, false)

  const yellowTexture = materials.get('M6_yellowTex')
  assert.equal(yellowTexture?.isMeshPhysicalMaterial, true)
  assert.equal(Buffer.isBuffer(yellowTexture.specularColorMap?.image), true, 'yellow specular color PNG should load as an encoded Buffer')
  assert.equal(yellowTexture.specularColorMap.name, 'YellowGrid')
  assert.equal(yellowTexture.specularColorMap.colorSpace, THREE.SRGBColorSpace)

  const hdrFactor = materials.get('M7.5_HDR')
  assert.equal(hdrFactor?.isMeshPhysicalMaterial, true)
  assert.deepEqual(hdrFactor.specularColor.toArray(), [25, 25, 25])

  const specularTextureMesh = meshes.find((mesh) => mesh.material.name === 'M2_SpecTex')
  assert.equal(specularTextureMesh.geometry.getAttribute('position')?.count, 3645)
  assert.equal(specularTextureMesh.geometry.getAttribute('uv')?.count, 3645)
  assert.equal(specularTextureMesh.geometry.index?.count, 19200)

  const scalarMesh = meshes.find((mesh) => mesh.material.name === 'M1.5_specFac')
  assert.equal(scalarMesh.geometry.getAttribute('position')?.count, 642)
  assert.equal(scalarMesh.geometry.index?.count, 3840)

  const camera = new THREE.OrthographicCamera(-0.7, 0.7, 0.52, -0.52, 0.01, 20)
  camera.position.set(0, 0, 4)
  camera.lookAt(0, 0, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.25))
  const light = new THREE.DirectionalLight(0xffffff, 4)
  light.position.set(0.2, 0.5, 4)
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.18, 'SpecularTest should render visible specular material samples')
})

test('committed Khronos glTF Sample Assets Suzanne fixture loads dense textured PBR mesh', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_SUZANNE, 'utf8'))
  assert.deepEqual(source.buffers, [{ byteLength: 590400, uri: 'Suzanne.bin' }])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'Suzanne_BaseColor.png',
    'Suzanne_MetallicRoughness.png',
  ])
  assert.equal(source.meshes[0].name, 'Suzanne')
  assert.equal(source.materials[0].name, 'Suzanne')
  assert.equal(source.materials[0].pbrMetallicRoughness.baseColorTexture.index, 0)
  assert.equal(source.materials[0].pbrMetallicRoughness.metallicRoughnessTexture.index, 1)

  const gltf = await loadGltfFixture(SAMPLE_ASSET_SUZANNE)
  const mesh = gltf.scene.getObjectByName('Suzanne')
  assert.ok(mesh?.isMesh, 'Suzanne sample should load a named mesh')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 11808)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 11808)
  assert.equal(mesh.geometry.getAttribute('tangent')?.count, 11808)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 11808)
  assert.equal(mesh.geometry.index?.count, 11808)
  assert.equal(mesh.material.name, 'Suzanne')
  assert.equal(mesh.material.metalness, 1)
  assert.equal(mesh.material.roughness, 1)

  const { map, roughnessMap, metalnessMap } = mesh.material
  assert.ok(map?.isTexture, 'Suzanne sample should load a base-color texture')
  assert.ok(roughnessMap?.isTexture, 'Suzanne sample should load a roughness texture')
  assert.ok(metalnessMap?.isTexture, 'Suzanne sample should load a metalness texture')
  assert.equal(roughnessMap, metalnessMap, 'Suzanne metallic/roughness channels should share the packed texture')
  assert.equal(map.name, 'Suzanne_BaseColor.png')
  assert.equal(roughnessMap.name, 'Suzanne_MetallicRoughness.png')
  assert.deepEqual(pngDimensions(map.image), [1024, 1024])
  assert.deepEqual(pngDimensions(roughnessMap.image), [1024, 1024])
  assert.equal(map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(roughnessMap.colorSpace, THREE.NoColorSpace)

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 1.8)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)
  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const halfExtent = Math.max(size.x, size.y, size.z) / 2 + 0.1
  const camera = new THREE.OrthographicCamera(-halfExtent, halfExtent, halfExtent, -halfExtent, 0.01, 20)
  camera.position.set(center.x + 2, center.y + 1.5, center.z + 2.5)
  camera.lookAt(center)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.3, 'Suzanne should render visible dense textured PBR geometry')
})
