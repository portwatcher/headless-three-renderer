import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'
import { Renderer, SAMPLE_ASSET_CARBON_FIBRE, SAMPLE_ASSET_CHAIR_DAMASK_PURPLEGOLD, SAMPLE_ASSET_COMMERCIAL_REFRIGERATOR } from './gltf.test.part-001.mjs'
import { loadGltfFixture, meanRegion, pngDimensions } from './gltf.test.part-028.mjs'
test('committed Khronos glTF Sample Assets CarbonFibre fixture loads real anisotropy material maps', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_CARBON_FIBRE, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_anisotropy'])
  assert.deepEqual(source.buffers, [
    { uri: 'CarbonFibre.bin', byteLength: 91936 },
  ])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'CarbonFibre_occlusion.png',
    'CarbonFibre_normal.png',
    'CarbonFibre_anisotropy.png',
  ])
  assert.deepEqual(source.samplers, [
    { magFilter: 9729, minFilter: 9987 },
  ])
  assert.deepEqual(source.materials[0].extensions?.KHR_materials_anisotropy, {
    anisotropyStrength: 0.5,
    anisotropyRotation: 0,
    anisotropyTexture: { index: 2 },
  })

  const gltf = await loadGltfFixture(SAMPLE_ASSET_CARBON_FIBRE)
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_anisotropy'))

  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 1)
  const mesh = meshes[0]
  assert.equal(mesh.name, 'CarbonFibre')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 2129)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 2129)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 2129)
  assert.equal(mesh.geometry.index?.count, 11904)

  const material = mesh.material
  assert.equal(material.name, 'CarbonFibre')
  assert.equal(material.isMeshPhysicalMaterial, true)
  assert.deepEqual(material.color.toArray(), [0.009, 0.009, 0.009])
  assert.equal(material.metalness, 0)
  assert.equal(material.roughness, 0.4)
  assert.equal(material.anisotropy, 0.5)
  assert.equal(material.anisotropyRotation, 0)
  assert.deepEqual(material.normalScale.toArray(), [2, -2])

  const textureExpectations = [
    [material.aoMap, 'CarbonFibre_occlusion.png', [256, 256]],
    [material.normalMap, 'CarbonFibre_normal.png', [512, 512]],
    [material.anisotropyMap, 'CarbonFibre_anisotropy.png', [128, 128]],
  ]
  for (const [texture, name, dimensions] of textureExpectations) {
    assert.equal(Buffer.isBuffer(texture?.image), true, `${name} should load as an encoded Buffer`)
    assert.equal(texture.name, name)
    assert.deepEqual(pngDimensions(texture.image), dimensions)
    assert.equal(texture.colorSpace, THREE.NoColorSpace)
    assert.equal(texture.flipY, false)
    assert.equal(texture.wrapS, THREE.RepeatWrapping)
    assert.equal(texture.wrapT, THREE.RepeatWrapping)
    assert.equal(texture.magFilter, THREE.LinearFilter)
    assert.equal(texture.minFilter, THREE.LinearMipmapLinearFilter)
  }

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 50)
  camera.position.copy(center).add(new THREE.Vector3(0, size.y * 0.3, Math.max(size.x, size.y, size.z) * 2.2))
  camera.lookAt(center)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.9))
  const light = new THREE.DirectionalLight(0xffffff, 4)
  light.position.copy(center).add(new THREE.Vector3(2, 4, 5))
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

  assert.ok(nonBackgroundRatio(rgba, [255, 255, 255], 3) > 0.35, 'CarbonFibre should render visible dark anisotropic material against white background')
  const centerSample = meanRegion(rgba, 128, 128, 48, 48, 80, 80)
  assert.ok(centerSample.r < 20 && centerSample.g < 20 && centerSample.b < 20, `CarbonFibre center should render the near-black material (${centerSample.r}, ${centerSample.g}, ${centerSample.b})`)
})

test('committed Khronos glTF Sample Assets ChairDamaskPurplegold fixture loads transformed damask sheen and specular materials', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_CHAIR_DAMASK_PURPLEGOLD, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_texture_transform', 'KHR_materials_sheen', 'KHR_materials_specular'])
  assert.deepEqual(source.buffers, [
    { uri: 'ChairDamaskPurplegold.bin', byteLength: 310904 },
  ])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'chair_wood_normal.jpg',
    'chair_occlusion.jpg',
    'chair_wood_albedo.jpg',
    'chair_wood_roughness0.jpg',
    'chair_metal_roughness255.jpg',
    'chair_damask_normal.jpg',
    'chair_damask_basecolor.jpg',
    'chair_damask_roughmetal.jpg',
    'chair_label.jpg',
  ])
  assert.deepEqual(source.textures.map((texture) => [texture.name, texture.source]), [
    ['chair_wood_normal.jpg', 0],
    ['chair_occlusion.jpg', 1],
    ['chair_wood_albedo.jpg', 2],
    ['chair_wood_roughness0.jpg', 3],
    ['chair_metal_roughness255.jpg', 4],
    ['damask_multicolor_normal.jpg', 5],
    ['damask_multicolor_albedo.jpg', 6],
    ['damask_multicolor_roughnessdamask_multicolor_metalness.jpg', 7],
    ['chair_label.jpg', 8],
  ])
  assert.deepEqual(source.materials.map((material) => material.name), ['wood', 'metal', 'fabric', 'label'])
  assert.deepEqual(source.materials[0].pbrMetallicRoughness.baseColorTexture.extensions?.KHR_texture_transform, {
    rotation: 0.1,
    scale: [3, 3],
  })
  assert.deepEqual(source.materials[2].normalTexture.extensions?.KHR_texture_transform, {
    scale: [3, 3],
  })
  assert.deepEqual(source.materials[2].extensions, {
    KHR_materials_sheen: {
      sheenColorFactor: [0.2, 0, 1],
      sheenRoughnessFactor: 0.5,
    },
    KHR_materials_specular: {
      specularColorFactor: [1, 0.25, 2],
      specularFactor: 1,
    },
  })

  const gltf = await loadGltfFixture(SAMPLE_ASSET_CHAIR_DAMASK_PURPLEGOLD)
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_texture_transform'))
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_sheen'))
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_specular'))

  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 11)

  const expectedMeshes = [
    ['oval-tufted-chair_legs-frame', 912, 4320, 'wood'],
    ['oval-tufted-chair_legs-hardware', 792, 2880, 'metal'],
    ['oval-tufted-chair_back-panel', 371, 1680, 'wood'],
    ['oval-tufted-chair_back-fabric', 893, 4512, 'fabric'],
    ['oval-tufted-chair_back-buttons', 66, 240, 'fabric'],
    ['oval-tufted-chair_back-welt', 441, 2304, 'fabric'],
    ['oval-tufted-chair_seat-panel', 257, 1344, 'wood'],
    ['oval-tufted-chair_seat-label', 25, 96, 'label'],
    ['oval-tufted-chair_seat-fabric', 1747, 9072, 'fabric'],
    ['oval-tufted-chair_seat-buttons', 330, 1200, 'fabric'],
    ['oval-tufted-chair_seat-welt', 441, 2304, 'fabric'],
  ]
  for (const [name, vertexCount, indexCount, materialName] of expectedMeshes) {
    const mesh = meshes.find((candidate) => candidate.name === name)
    assert.ok(mesh, `${name} should load`)
    assert.equal(mesh.geometry.getAttribute('position')?.count, vertexCount)
    assert.equal(mesh.geometry.getAttribute('normal')?.count, vertexCount)
    assert.equal(mesh.geometry.getAttribute('uv')?.count, vertexCount)
    assert.equal(mesh.geometry.getAttribute('uv1')?.count, vertexCount)
    assert.equal(mesh.geometry.index?.count, indexCount)
    assert.equal(mesh.material.name, materialName)
  }

  const materials = new Map(meshes.map((mesh) => [mesh.material.name, mesh.material]))
  const wood = materials.get('wood')
  assert.equal(wood?.isMeshStandardMaterial, true)
  assert.deepEqual(wood.color.toArray(), [0.247, 0.109, 0.035])
  assert.equal(wood.metalness, 0)
  assert.equal(wood.roughness, 1)
  assert.equal(wood.map.name, 'chair_wood_albedo.jpg')
  assert.equal(wood.normalMap.name, 'chair_wood_normal.jpg')
  assert.equal(wood.roughnessMap.name, 'chair_wood_roughness0.jpg')
  assert.equal(wood.metalnessMap.name, 'chair_wood_roughness0.jpg')
  assert.equal(wood.aoMap.name, 'chair_occlusion.jpg')
  assert.equal(wood.aoMap.channel, 1)
  assert.equal(wood.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(wood.normalMap.colorSpace, THREE.NoColorSpace)
  assert.equal(wood.roughnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(wood.metalnessMap.colorSpace, THREE.NoColorSpace)
  assert.deepEqual(wood.map.repeat.toArray(), [3, 3])
  assert.equal(wood.map.rotation, 0.1)
  assert.deepEqual(wood.roughnessMap.repeat.toArray(), [3, 3])
  assert.deepEqual(wood.metalnessMap.repeat.toArray(), [3, 3])

  const fabric = materials.get('fabric')
  assert.equal(fabric?.isMeshPhysicalMaterial, true)
  assert.equal(fabric.metalness, 1)
  assert.equal(fabric.roughness, 1)
  assert.equal(fabric.sheen, 1)
  assert.deepEqual(fabric.sheenColor.toArray(), [0.2, 0, 1])
  assert.equal(fabric.sheenRoughness, 0.5)
  assert.equal(fabric.specularIntensity, 1)
  assert.deepEqual(fabric.specularColor.toArray(), [1, 0.25, 2])
  assert.equal(fabric.map.name, 'damask_multicolor_albedo.jpg')
  assert.equal(fabric.normalMap.name, 'damask_multicolor_normal.jpg')
  assert.equal(fabric.roughnessMap.name, 'damask_multicolor_roughnessdamask_multicolor_metalness.jpg')
  assert.equal(fabric.metalnessMap.name, 'damask_multicolor_roughnessdamask_multicolor_metalness.jpg')
  assert.equal(fabric.aoMap.name, 'chair_occlusion.jpg')
  assert.equal(fabric.aoMap.channel, 1)
  assert.equal(fabric.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(fabric.normalMap.colorSpace, THREE.NoColorSpace)
  assert.equal(fabric.roughnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(fabric.metalnessMap.colorSpace, THREE.NoColorSpace)
  assert.deepEqual(fabric.map.repeat.toArray(), [3, 3])
  assert.deepEqual(fabric.normalMap.repeat.toArray(), [3, 3])
  assert.deepEqual(fabric.roughnessMap.repeat.toArray(), [3, 3])
  assert.deepEqual(fabric.metalnessMap.repeat.toArray(), [3, 3])

  const metal = materials.get('metal')
  assert.equal(metal?.isMeshStandardMaterial, true)
  assert.equal(metal.metalness, 1)
  assert.equal(metal.roughness, 1)
  assert.equal(metal.roughnessMap.name, 'chair_metal_roughness255.jpg')
  assert.equal(metal.metalnessMap.name, 'chair_metal_roughness255.jpg')
  assert.equal(metal.roughnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(metal.metalnessMap.colorSpace, THREE.NoColorSpace)
  assert.deepEqual(metal.roughnessMap.repeat.toArray(), [3, 3])
  assert.deepEqual(metal.metalnessMap.repeat.toArray(), [3, 3])
  assert.equal(metal.aoMap.name, 'chair_occlusion.jpg')
  assert.equal(metal.aoMap.channel, 1)

  const label = materials.get('label')
  assert.equal(label?.isMeshStandardMaterial, true)
  assert.equal(label.map.name, 'chair_label.jpg')
  assert.equal(label.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(label.aoMap.channel, 1)

  for (const texture of [
    wood.map,
    wood.normalMap,
    wood.roughnessMap,
    wood.metalnessMap,
    wood.aoMap,
    fabric.map,
    fabric.normalMap,
    fabric.roughnessMap,
    fabric.metalnessMap,
    metal.roughnessMap,
    metal.metalnessMap,
    label.map,
  ]) {
    assert.equal(Buffer.isBuffer(texture.image), true, `${texture.name} should load as an encoded Buffer`)
    assert.equal(texture.flipY, false)
    assert.equal(texture.wrapS, THREE.RepeatWrapping)
    assert.equal(texture.wrapT, THREE.RepeatWrapping)
    assert.equal(texture.magFilter, THREE.LinearFilter)
    assert.equal(texture.minFilter, THREE.LinearMipmapLinearFilter)
  }

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 50)
  camera.position.copy(center).add(new THREE.Vector3(0, size.y * 0.25, Math.max(size.x, size.y, size.z) * 2.4))
  camera.lookAt(center)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.7))
  const light = new THREE.DirectionalLight(0xffffff, 2.4)
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.1, 'ChairDamaskPurplegold should render visible textured chair geometry')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 1 && mean.b > 1, `ChairDamaskPurplegold should render lit damask and wood pixels (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets CommercialRefrigerator fixture loads required transmission glass and door animation', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_COMMERCIAL_REFRIGERATOR, 'utf8'))
  assert.deepEqual(source.extensionsRequired, ['KHR_materials_transmission'])
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_clearcoat', 'KHR_materials_transmission'])
  assert.deepEqual(source.buffers, [
    { byteLength: 8029440, uri: 'CommercialRefrigerator.data.bin' },
  ])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'ChampagneLabel_Color.jpg',
    'Glass_Normal.jpg',
    'ChampagneLabel_RM.jpg',
    'ChampagneFoil_Normal.jpg',
    'Case_ORM.jpg',
    'Glass_RM.jpg',
    'Interior_Color.jpg',
    'Interior_Emissive.jpg',
    'Interior_ORM.jpg',
    'Case_Color.jpg',
  ])
  assert.deepEqual(source.materials.map((material) => material.name), [
    'FridgeCase',
    'FridgeInterior',
    'FridgeGlass',
    'ChampagneGlass',
    'ChampagneFoil',
    'ChampagneLabel',
  ])
  assert.deepEqual(source.materials[2].extensions, {
    KHR_materials_clearcoat: {
      clearcoatFactor: 1,
      clearcoatRoughnessFactor: 0,
    },
    KHR_materials_transmission: {
      transmissionFactor: 1,
    },
  })
  assert.deepEqual(source.animations.map((animation) => animation.channels.map((channel) => channel.target)), [
    [{ node: 1, path: 'rotation' }],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_COMMERCIAL_REFRIGERATOR)
  assert.deepEqual(gltf.parser?.json?.extensionsRequired, ['KHR_materials_transmission'])
  assert.equal(gltf.animations.length, 1)
  assert.equal(gltf.animations[0].name, 'animation_0')
  assert.ok(Math.abs(gltf.animations[0].duration - 4.666666507720947) < 1e-6)
  assert.deepEqual(gltf.animations[0].tracks.map((track) => track.name), ['FridgeDoor.quaternion'])

  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 8)

  const expectedMeshes = [
    ['FridgeCase', 720, 1488, 'FridgeCase', false],
    ['FridgeDoor', 582, 1164, 'FridgeCase', false],
    ['FridgeDoorInner', 42, 84, 'FridgeInterior', false],
    ['FridgeGlass', 4, 6, 'FridgeGlass', false],
    ['FridgeInterior', 2612, 6783, 'FridgeInterior', false],
    ['ChampagneBottles_1', 58216, 322848, 'ChampagneGlass', true],
    ['ChampagneBottles_2', 51224, 279072, 'ChampagneFoil', true],
    ['ChampagneBottles_3', 3420, 14592, 'ChampagneLabel', true],
  ]
  for (const [name, vertexCount, indexCount, materialName, hasTangents] of expectedMeshes) {
    const mesh = meshes.find((candidate) => candidate.name === name)
    assert.ok(mesh, `${name} should load`)
    assert.equal(mesh.geometry.getAttribute('position')?.count, vertexCount)
    assert.equal(mesh.geometry.getAttribute('normal')?.count, vertexCount)
    assert.equal(mesh.geometry.getAttribute('uv')?.count, vertexCount)
    assert.equal(mesh.geometry.getAttribute('tangent')?.count, hasTangents ? vertexCount : undefined)
    assert.equal(mesh.geometry.index?.count, indexCount)
    assert.equal(mesh.material.name, materialName)
  }

  const materials = new Map(meshes.map((mesh) => [mesh.material.name, mesh.material]))
  const fridgeCase = materials.get('FridgeCase')
  assert.equal(fridgeCase?.isMeshStandardMaterial, true)
  assert.equal(fridgeCase.metalness, 1)
  assert.equal(fridgeCase.roughness, 1)
  assert.equal(fridgeCase.map.name, 'Case_Color.jpg')
  assert.equal(fridgeCase.aoMap.name, 'Case_ORM.jpg')
  assert.equal(fridgeCase.roughnessMap.name, 'Case_ORM.jpg')
  assert.equal(fridgeCase.metalnessMap.name, 'Case_ORM.jpg')
  assert.equal(fridgeCase.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(fridgeCase.aoMap.colorSpace, THREE.NoColorSpace)
  assert.equal(fridgeCase.roughnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(fridgeCase.aoMap.channel, 0)

  const fridgeInterior = materials.get('FridgeInterior')
  assert.equal(fridgeInterior?.isMeshStandardMaterial, true)
  assert.equal(fridgeInterior.roughness, 0.75)
  assert.deepEqual(fridgeInterior.emissive.toArray(), [1, 1, 1])
  assert.equal(fridgeInterior.map.name, 'Interior_Color.jpg')
  assert.equal(fridgeInterior.emissiveMap.name, 'Interior_Emissive.jpg')
  assert.equal(fridgeInterior.aoMap.name, 'Interior_ORM.jpg')
  assert.equal(fridgeInterior.roughnessMap.name, 'Interior_ORM.jpg')
  assert.equal(fridgeInterior.metalnessMap.name, 'Interior_ORM.jpg')
  assert.equal(fridgeInterior.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(fridgeInterior.emissiveMap.colorSpace, THREE.SRGBColorSpace)
  assert.equal(fridgeInterior.roughnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(fridgeInterior.aoMap.channel, 0)

  const fridgeGlass = materials.get('FridgeGlass')
  assert.equal(fridgeGlass?.isMeshPhysicalMaterial, true)
  assert.equal(fridgeGlass.metalness, 0)
  assert.equal(fridgeGlass.roughness, 0.75)
  assert.equal(fridgeGlass.clearcoat, 1)
  assert.equal(fridgeGlass.clearcoatRoughness, 0)
  assert.equal(fridgeGlass.transmission, 1)
  assert.equal(fridgeGlass.roughnessMap.name, 'Glass_RM.jpg')
  assert.equal(fridgeGlass.metalnessMap.name, 'Glass_RM.jpg')
  assert.equal(fridgeGlass.roughnessMap.colorSpace, THREE.NoColorSpace)

  const champagneGlass = materials.get('ChampagneGlass')
  assert.equal(champagneGlass?.isMeshStandardMaterial, true)
  assert.deepEqual(champagneGlass.color.toArray(), [
    0.01568627543747425,
    0.0313725508749485,
    0.01568627543747425,
  ])
  assert.equal(champagneGlass.roughness, 0.5)
  assert.equal(champagneGlass.normalMap.name, 'Glass_Normal.jpg')
  assert.deepEqual(champagneGlass.normalScale.toArray(), [0.3, 0.3])
  assert.equal(champagneGlass.roughnessMap.name, 'Glass_RM.jpg')

  const champagneFoil = materials.get('ChampagneFoil')
  assert.equal(champagneFoil?.isMeshStandardMaterial, true)
  assert.deepEqual(champagneFoil.color.toArray(), [
    0.8666666746139526,
    0.6705882549285889,
    0.4156862795352936,
  ])
  assert.equal(champagneFoil.metalness, 1)
  assert.equal(champagneFoil.roughness, 0.4)
  assert.equal(champagneFoil.normalMap.name, 'ChampagneFoil_Normal.jpg')

  const champagneLabel = materials.get('ChampagneLabel')
  assert.equal(champagneLabel?.isMeshStandardMaterial, true)
  assert.equal(champagneLabel.map.name, 'ChampagneLabel_Color.jpg')
  assert.equal(champagneLabel.normalMap.name, 'Glass_Normal.jpg')
  assert.deepEqual(champagneLabel.normalScale.toArray(), [0.2, 0.2])
  assert.equal(champagneLabel.roughnessMap.name, 'ChampagneLabel_RM.jpg')
  assert.equal(champagneLabel.map.colorSpace, THREE.SRGBColorSpace)

  for (const texture of [
    fridgeCase.map,
    fridgeCase.aoMap,
    fridgeInterior.map,
    fridgeInterior.emissiveMap,
    fridgeInterior.aoMap,
    fridgeGlass.roughnessMap,
    champagneGlass.normalMap,
    champagneGlass.roughnessMap,
    champagneFoil.normalMap,
    champagneLabel.map,
    champagneLabel.normalMap,
    champagneLabel.roughnessMap,
  ]) {
    assert.equal(Buffer.isBuffer(texture.image), true, `${texture.name} should load as an encoded Buffer`)
    assert.equal(texture.flipY, false)
  }

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 80)
  camera.position.copy(center).add(new THREE.Vector3(0, size.y * 0.2, Math.max(size.x, size.y, size.z) * 2.2))
  camera.lookAt(center)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.65))
  const light = new THREE.DirectionalLight(0xffffff, 2.8)
  light.position.copy(center).add(new THREE.Vector3(2, 3, 5))
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 144,
    height: 144,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.2, 'CommercialRefrigerator should render visible refrigerator and bottle geometry')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 8 && mean.g > 8 && mean.b > 8, `CommercialRefrigerator should render lit refrigerator pixels (${mean.r}, ${mean.g}, ${mean.b})`)
})
