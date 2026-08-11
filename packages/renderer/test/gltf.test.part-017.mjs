import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'
import { Renderer, SAMPLE_ASSET_GLAM_VELVET_SOFA, SAMPLE_ASSET_MATERIALS_VARIANTS_SHOE, SAMPLE_ASSET_SHEEN_CLOTH } from './gltf.test.part-001.mjs'
import { loadGltfFixture, meanRegion, pngDimensions } from './gltf.test.part-028.mjs'
test('committed Khronos glTF Sample Assets GlamVelvetSofa fixture loads sheen variants and normal texture transforms', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_GLAM_VELVET_SOFA, 'utf8'))
  assert.deepEqual(source.extensionsRequired, ['KHR_texture_transform'])
  assert.deepEqual(source.extensionsUsed, [
    'KHR_texture_transform',
    'KHR_materials_sheen',
    'KHR_materials_specular',
    'KHR_materials_variants',
    'KHR_lights_punctual',
  ])
  assert.deepEqual(source.extensions?.KHR_materials_variants?.variants?.map((variant) => variant.name), [
    'Champagne',
    'Navy',
    'Gray',
    'Black',
    'Pale Pink',
  ])
  assert.deepEqual(source.extensions?.KHR_lights_punctual?.lights, [
    { type: 'directional', intensity: 3 },
  ])
  assert.deepEqual(source.buffers, [
    { uri: 'GlamVelvetSofa.bin', byteLength: 124952 },
  ])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'GlamVelvetSofa_occlusion.png',
    'GlamVelvetSofa_normal.png',
  ])
  assert.deepEqual(source.materials.map((material) => material.name), [
    'GlamVelvetSofa_legs',
    'GlamVelvetSofa_feet',
    'GlamVelvetSofa_fabric_champagne',
    'GlamVelvetSofa_fabric_navy',
    'GlamVelvetSofa_fabric_gray',
    'GlamVelvetSofa_fabric_black',
    'GlamVelvetSofa_fabric_palepink',
  ])
  assert.deepEqual(source.materials[3].normalTexture, {
    index: 1,
    scale: 0.75,
    extensions: {
      KHR_texture_transform: {
        rotation: 0.36,
        scale: [5, 5],
        texCoord: 0,
      },
    },
  })
  assert.deepEqual(source.materials[3].extensions, {
    KHR_materials_specular: {
      specularColorFactor: [0.1, 0.34, 1],
    },
    KHR_materials_sheen: {
      sheenColorFactor: [0.05, 0.17, 0.5],
      sheenRoughnessFactor: 0.6,
    },
  })
  assert.deepEqual(source.meshes[1].primitives[0].extensions?.KHR_materials_variants?.mappings, [
    { material: 2, variants: [0] },
    { material: 3, variants: [1] },
    { material: 4, variants: [2] },
    { material: 5, variants: [3] },
    { material: 6, variants: [4] },
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_GLAM_VELVET_SOFA)
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_sheen'))
  assert.deepEqual(gltf.parser?.json?.extensionsRequired, ['KHR_texture_transform'])
  const meshes = []
  const lights = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
    if (object.isLight === true) lights.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'GlamVelvetSofa_legs',
    'GlamVelvetSofa_fabric',
    'GlamVelvetSofa_feet',
  ])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [342, 2092, 684])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [342, 2092, 684])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [342, 2092, 684])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [918, 9726, 1944])
  assert.deepEqual(lights.map((light) => [light.name, light.type, light.intensity]), [
    ['Key_Light', 'DirectionalLight', 3],
  ])

  const fabric = meshes.find((mesh) => mesh.name === 'GlamVelvetSofa_fabric')
  assert.deepEqual(fabric.userData.gltfExtensions?.KHR_materials_variants?.mappings, [
    { material: 2, variants: [0] },
    { material: 3, variants: [1] },
    { material: 4, variants: [2] },
    { material: 5, variants: [3] },
    { material: 6, variants: [4] },
  ])

  const materials = new Map(meshes.map((mesh) => [mesh.material.name, mesh.material]))
  const legs = materials.get('GlamVelvetSofa_legs')
  assert.equal(legs?.isMeshStandardMaterial, true)
  assert.deepEqual(legs.color.toArray(), [0.02, 0.02, 0.02])
  assert.equal(legs.metalness, 0)
  assert.equal(legs.roughness, 0.4)
  assert.equal(legs.aoMap.name, 'GlamVelvetSofa_occlusion.png')
  assert.equal(legs.aoMap.channel, 0)

  const fabricMaterial = materials.get('GlamVelvetSofa_fabric_navy')
  assert.equal(fabricMaterial?.isMeshPhysicalMaterial, true)
  assert.deepEqual(fabricMaterial.color.toArray(), [0.01, 0.01, 0.01])
  assert.equal(fabricMaterial.metalness, 0)
  assert.equal(fabricMaterial.roughness, 0.7)
  assert.equal(fabricMaterial.sheen, 1)
  assert.deepEqual(fabricMaterial.sheenColor.toArray(), [0.05, 0.17, 0.5])
  assert.equal(fabricMaterial.sheenRoughness, 0.6)
  assert.deepEqual(fabricMaterial.specularColor.toArray(), [0.1, 0.34, 1])
  assert.equal(fabricMaterial.specularIntensity, 1)
  assert.equal(fabricMaterial.normalMap.name, 'GlamVelvetSofa_normal.png')
  assert.deepEqual(fabricMaterial.normalScale.toArray(), [0.75, -0.75])
  assert.deepEqual(fabricMaterial.normalMap.repeat.toArray(), [5, 5])
  assert.equal(fabricMaterial.normalMap.rotation, 0.36)
  assert.equal(fabricMaterial.aoMap.name, 'GlamVelvetSofa_occlusion.png')
  assert.equal(fabricMaterial.aoMap.channel, 0)

  const feet = materials.get('GlamVelvetSofa_feet')
  assert.equal(feet?.isMeshStandardMaterial, true)
  assert.deepEqual(feet.color.toArray(), [1, 0.8, 0.7])
  assert.equal(feet.metalness, 1)
  assert.equal(feet.roughness, 0.4)
  assert.equal(feet.aoMap.name, 'GlamVelvetSofa_occlusion.png')

  for (const texture of [
    legs.aoMap,
    fabricMaterial.aoMap,
    fabricMaterial.normalMap,
    feet.aoMap,
  ]) {
    assert.equal(Buffer.isBuffer(texture.image), true, `${texture.name} should load as an encoded Buffer`)
    assert.deepEqual(pngDimensions(texture.image), [1024, 1024])
    assert.equal(texture.colorSpace, THREE.NoColorSpace)
    assert.equal(texture.flipY, false)
  }

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100)
  camera.position.copy(center).add(new THREE.Vector3(0, size.y * 0.15, Math.max(size.x, size.y, size.z) * 2.6))
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.04, 'GlamVelvetSofa should render visible dark velvet furniture geometry')
})

test('committed Khronos glTF Sample Assets SheenCloth fixture loads transformed sheen texture inputs', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_SHEEN_CLOTH, 'utf8'))
  assert.deepEqual(source.extensionsRequired, ['KHR_texture_transform'])
  assert.deepEqual(source.extensionsUsed, ['KHR_texture_transform', 'KHR_materials_sheen'])
  assert.deepEqual(source.buffers, [
    { uri: 'SheenCloth.bin', byteLength: 3479088 },
  ])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'technicalFabricSmall_normal_256.png',
    'technicalFabricSmall_orm_256.png',
    'technicalFabricSmall_basecolor_256.png',
    'technicalFabricSmall_sheen_256.png',
    'SheenCloth_AO.jpg',
  ])
  assert.deepEqual(source.textures.map((texture) => [texture.name, texture.source]), [
    ['technicalFabricSmall_normal_256.png', 0],
    ['technicalFabricSmall_orm_256.png', 1],
    ['technicalFabricSmall_basecolor_256.png', 2],
    ['technicalFabricSmall_sheen_256.png', 3],
    ['SheenCloth_AO.jpg', 4],
  ])
  assert.deepEqual(source.meshes[0].primitives[0].attributes, {
    POSITION: 1,
    TANGENT: 2,
    NORMAL: 3,
    TEXCOORD_0: 4,
  })
  assert.deepEqual(source.materials[0].pbrMetallicRoughness.baseColorTexture.extensions?.KHR_texture_transform, {
    scale: [30, -30],
  })
  assert.deepEqual(source.materials[0].extensions?.KHR_materials_sheen, {
    sheenColorFactor: [1, 1, 1],
    sheenRoughnessFactor: 1,
    sheenColorTexture: {
      index: 3,
      extensions: {
        KHR_texture_transform: {
          scale: [30, -30],
        },
      },
    },
    sheenRoughnessTexture: {
      index: 3,
      extensions: {
        KHR_texture_transform: {
          scale: [30, -30],
        },
      },
    },
  })

  const gltf = await loadGltfFixture(SAMPLE_ASSET_SHEEN_CLOTH)
  assert.deepEqual(gltf.parser?.json?.extensionsRequired, ['KHR_texture_transform'])
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_sheen'))

  const mesh = gltf.scene.getObjectByName('SheenCloth_mesh')
  assert.ok(mesh?.isMesh, 'SheenCloth should load a named mesh')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 58081)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 58081)
  assert.equal(mesh.geometry.getAttribute('tangent')?.count, 58081)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 58081)
  assert.equal(mesh.geometry.index?.count, 345600)

  const material = mesh.material
  assert.equal(material.name, 'SheenClothMat')
  assert.equal(material.isMeshPhysicalMaterial, true)
  assert.equal(material.metalness, 1)
  assert.equal(material.roughness, 1)
  assert.equal(material.sheen, 1)
  assert.deepEqual(material.sheenColor.toArray(), [1, 1, 1])
  assert.equal(material.sheenRoughness, 1)
  assert.equal(material.map.name, 'technicalFabricSmall_basecolor_256.png')
  assert.equal(material.normalMap.name, 'technicalFabricSmall_normal_256.png')
  assert.equal(material.roughnessMap.name, 'technicalFabricSmall_orm_256.png')
  assert.equal(material.metalnessMap.name, 'technicalFabricSmall_orm_256.png')
  assert.equal(material.sheenColorMap.name, 'technicalFabricSmall_sheen_256.png')
  assert.equal(material.sheenRoughnessMap.name, 'technicalFabricSmall_sheen_256.png')
  assert.equal(material.aoMap.name, 'SheenCloth_AO.jpg')
  assert.equal(material.aoMap.channel, 0)
  assert.equal(material.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(material.normalMap.colorSpace, THREE.NoColorSpace)
  assert.equal(material.roughnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(material.metalnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(material.sheenColorMap.colorSpace, THREE.SRGBColorSpace)
  assert.equal(material.sheenRoughnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(material.aoMap.colorSpace, THREE.NoColorSpace)
  assert.deepEqual(material.map.repeat.toArray(), [30, -30])
  assert.deepEqual(material.normalMap.repeat.toArray(), [30, -30])
  assert.deepEqual(material.roughnessMap.repeat.toArray(), [30, -30])
  assert.deepEqual(material.metalnessMap.repeat.toArray(), [30, -30])
  assert.deepEqual(material.sheenColorMap.repeat.toArray(), [30, -30])
  assert.deepEqual(material.sheenRoughnessMap.repeat.toArray(), [30, -30])
  assert.deepEqual(material.aoMap.repeat.toArray(), [1, 1])

  const pngTextureExpectations = [
    material.map,
    material.normalMap,
    material.roughnessMap,
    material.metalnessMap,
    material.sheenColorMap,
    material.sheenRoughnessMap,
  ]
  for (const texture of pngTextureExpectations) {
    assert.equal(Buffer.isBuffer(texture.image), true, `${texture.name} should load as an encoded Buffer`)
    assert.deepEqual(pngDimensions(texture.image), [256, 256])
    assert.equal(texture.flipY, false)
  }
  assert.equal(Buffer.isBuffer(material.aoMap.image), true, 'SheenCloth AO JPEG should load as an encoded Buffer')
  assert.equal(material.aoMap.flipY, false)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 50)
  camera.position.copy(center).add(new THREE.Vector3(0, size.y * 0.25, Math.max(size.x, size.y, size.z) * 1.9))
  camera.lookAt(center)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 2.6)
  light.position.copy(center).add(new THREE.Vector3(1.5, 2, 4))
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.2, 'SheenCloth should render visible textured cloth geometry')
  const mean = meanRgba(rgba)
  assert.ok(mean.b > 8 && mean.g > 1, `SheenCloth should render lit blue sheen pixels (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets MaterialsVariantsShoe fixture preserves KHR_materials_variants mappings', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_MATERIALS_VARIANTS_SHOE, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_variants'])
  assert.deepEqual(source.extensions?.KHR_materials_variants?.variants?.map((variant) => variant.name), [
    'midnight',
    'beach',
    'street',
  ])
  assert.deepEqual(source.buffers, [
    { name: 'shoes-processed', byteLength: 705680, uri: 'MaterialsVariantsShoe.bin' },
  ])
  assert.deepEqual(source.images.map((image) => [image.mimeType, image.uri]), [
    ['image/jpeg', 'occlusionRougnessMetalness.jpg'],
    ['image/jpeg', 'diffuseMidnight.jpg'],
    ['image/jpeg', 'normal.jpg'],
    ['image/jpeg', 'diffuseBeach.jpg'],
    ['image/jpeg', 'diffuseStreet.jpg'],
  ])
  assert.deepEqual(source.materials.map((material) => [
    material.pbrMetallicRoughness?.baseColorTexture?.index,
    material.pbrMetallicRoughness?.metallicRoughnessTexture?.index,
    material.normalTexture?.index,
    material.occlusionTexture?.index,
  ]), [
    [1, 0, 2, 0],
    [3, 0, 2, 0],
    [4, 0, 2, 0],
  ])
  assert.deepEqual(source.meshes[0].primitives[0].extensions?.KHR_materials_variants?.mappings, [
    { material: 0, variants: [0] },
    { material: 1, variants: [1] },
    { material: 2, variants: [2] },
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_MATERIALS_VARIANTS_SHOE)
  assert.deepEqual(gltf.parser?.json?.extensions?.KHR_materials_variants?.variants?.map((variant) => variant.name), [
    'midnight',
    'beach',
    'street',
  ])

  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 1)
  const mesh = meshes[0]
  assert.equal(mesh.name, 'Shoe')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 13540)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 13540)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 13540)
  assert.equal(mesh.geometry.index?.count, 68100)
  assert.deepEqual(mesh.userData.gltfExtensions?.KHR_materials_variants?.mappings, [
    { material: 0, variants: [0] },
    { material: 1, variants: [1] },
    { material: 2, variants: [2] },
  ])

  const material = mesh.material
  assert.equal(material.name, 'phong1SG')
  assert.equal(material.isMeshStandardMaterial, true)
  assert.equal(material.metalness, 1)
  assert.equal(material.roughness, 1)
  assert.equal(material.map.name, 'diffuseMidnight.jpg')
  assert.equal(material.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(material.map.flipY, false)
  assert.equal(material.normalMap.name, 'normal.jpg')
  assert.equal(material.normalMap.colorSpace, THREE.NoColorSpace)
  assert.equal(material.normalMap.flipY, false)
  assert.equal(material.aoMap.name, 'occlusionRougnessMetalness.jpg')
  assert.equal(material.roughnessMap, material.aoMap)
  assert.equal(material.metalnessMap, material.aoMap)
  assert.equal(material.aoMap.colorSpace, THREE.NoColorSpace)
  assert.equal(material.aoMap.flipY, false)
  assert.equal(material.aoMap.channel, 0)
  for (const texture of [material.map, material.normalMap, material.aoMap]) {
    assert.equal(Buffer.isBuffer(texture.image), true, `${texture.name} should load as an encoded Buffer`)
  }

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const camera = new THREE.PerspectiveCamera(35, 1.2, 0.01, 50)
  camera.position.copy(center).add(new THREE.Vector3(0, size.y * 0.35, Math.max(size.x, size.y, size.z) * 2.2))
  camera.lookAt(center)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 2.5)
  light.position.copy(center).add(new THREE.Vector3(2, 4, 5))
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 144,
    height: 120,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.1, 'MaterialsVariantsShoe should render visible textured shoe geometry')
  const centerSample = meanRegion(rgba, 144, 120, 54, 42, 90, 78)
  assert.ok(centerSample.b > centerSample.g && centerSample.g > centerSample.r, `MaterialsVariantsShoe center should render the midnight diffuse texture (${centerSample.r}, ${centerSample.g}, ${centerSample.b})`)
})
