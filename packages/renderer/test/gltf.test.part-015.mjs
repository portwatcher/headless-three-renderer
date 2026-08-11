import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'
import { Renderer, SAMPLE_ASSET_CLEARCOAT_CAR_PAINT, SAMPLE_ASSET_CLEARCOAT_TEST, SAMPLE_ASSET_CLEARCOAT_WICKER, SAMPLE_ASSET_COMPARE_IOR, SAMPLE_ASSET_IRIDESCENCE_LAMP, SAMPLE_ASSET_POT_OF_COALS } from './gltf.test.part-001.mjs'
import { findFirst, loadGltfFixture, meanRegion, pngDimensions } from './gltf.test.part-028.mjs'
test('committed Khronos glTF Sample Assets ClearCoatTest fixture loads KHR_materials_clearcoat maps', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_CLEARCOAT_TEST)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 27)

  const materials = new Map(meshes.map((mesh) => [mesh.material.name, mesh.material]))
  const simpleCoated = materials.get('Simple_Coated')
  assert.equal(simpleCoated?.isMeshPhysicalMaterial, true)
  assert.equal(simpleCoated.clearcoat, 1)
  assert.equal(simpleCoated.clearcoatRoughness, 0.03)

  const partialCoated = materials.get('Partial_Coated')
  assert.equal(partialCoated?.isMeshPhysicalMaterial, true)
  assert.equal(Buffer.isBuffer(partialCoated.clearcoatMap?.image), true, 'clearcoat factor PNG should load as an encoded Buffer')
  assert.equal(partialCoated.clearcoatMap.colorSpace, THREE.NoColorSpace)

  const roughCoated = materials.get('RoughVariations_Coated')
  assert.equal(roughCoated?.isMeshPhysicalMaterial, true)
  assert.equal(roughCoated.clearcoatRoughness, 1)
  assert.equal(Buffer.isBuffer(roughCoated.clearcoatRoughnessMap?.image), true, 'clearcoat roughness PNG should load as an encoded Buffer')
  assert.equal(roughCoated.clearcoatRoughnessMap.colorSpace, THREE.NoColorSpace)

  const coatNormal = materials.get('CoatNorm_Coated')
  assert.equal(coatNormal?.isMeshPhysicalMaterial, true)
  assert.equal(Buffer.isBuffer(coatNormal.clearcoatNormalMap?.image), true, 'clearcoat normal PNG should load as an encoded Buffer')
  assert.equal(coatNormal.clearcoatNormalMap.colorSpace, THREE.NoColorSpace)

  const sharedNormal = materials.get('SharedNorm_Coated')
  assert.equal(sharedNormal?.isMeshPhysicalMaterial, true)
  assert.equal(Buffer.isBuffer(sharedNormal.clearcoatNormalMap?.image), true, 'shared clearcoat normal JPEG should load as an encoded Buffer')
  assert.equal(sharedNormal.clearcoatNormalMap.colorSpace, THREE.NoColorSpace)

  const camera = new THREE.PerspectiveCamera(35, 4 / 3, 0.01, 40)
  camera.position.set(0, 1.2, 12)
  camera.lookAt(0, 0.6, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.9))
  const light = new THREE.DirectionalLight(0xffffff, 1.4)
  light.position.set(2, 4, 6)
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.05, 'ClearCoatTest should render visible clearcoat panels')
})

test('committed Khronos glTF Sample Assets ClearCoatCarPaint fixture loads clearcoat normal texture transforms', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_CLEARCOAT_CAR_PAINT, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_texture_transform', 'KHR_materials_clearcoat'])
  assert.deepEqual(source.extensionsRequired, ['KHR_texture_transform', 'KHR_materials_clearcoat'])
  assert.deepEqual(source.buffers, [{ uri: 'ClearCoatCarPaint.bin', byteLength: 73728 }])
  assert.deepEqual(source.images.map((image) => image.uri), ['ClearCoatCarPaint_Normal.png'])

  const materialSource = source.materials[0]
  assert.equal(materialSource.name, 'Clear Coat Car Paint')
  assert.deepEqual(materialSource.pbrMetallicRoughness, {
    baseColorFactor: [0.7, 0, 0, 1],
    metallicFactor: 0.3,
    roughnessFactor: 0.4,
  })
  assert.deepEqual(materialSource.normalTexture, {
    index: 0,
    scale: 0.2,
    extensions: {
      KHR_texture_transform: {
        scale: [3, 3],
      },
    },
  })
  assert.deepEqual(materialSource.extensions.KHR_materials_clearcoat, {
    clearcoatFactor: 1,
    clearcoatRoughnessFactor: 0,
  })

  const gltf = await loadGltfFixture(SAMPLE_ASSET_CLEARCOAT_CAR_PAINT)
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_clearcoat'))
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'Khronos ClearCoatCarPaint sample should load a mesh')
  assert.equal(mesh.name, 'Sphere')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 1728)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 1728)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 1728)
  assert.equal(mesh.geometry.index?.count, 9216)
  assert.deepEqual(mesh.position.toArray(), [0, 0.5, 0])

  const material = mesh.material
  assert.equal(material.isMeshPhysicalMaterial, true)
  assert.equal(material.name, 'Clear Coat Car Paint')
  assert.deepEqual(material.color.toArray(), [0.7, 0, 0])
  assert.equal(material.metalness, 0.3)
  assert.equal(material.roughness, 0.4)
  assert.equal(material.clearcoat, 1)
  assert.equal(material.clearcoatRoughness, 0)
  assert.deepEqual(material.normalScale.toArray(), [0.2, -0.2])

  const normalMap = material.normalMap
  assert.ok(normalMap?.isTexture, 'ClearCoatCarPaint normal map should load')
  assert.equal(normalMap.name, 'ClearCoatCarPaint_Normal.png')
  assert.equal(Buffer.isBuffer(normalMap.image), true, 'ClearCoatCarPaint normal PNG should load as an encoded Buffer')
  assert.deepEqual(pngDimensions(normalMap.image), [128, 128])
  assert.equal(normalMap.colorSpace, THREE.NoColorSpace)
  assert.deepEqual(normalMap.repeat.toArray(), [3, 3])
  assert.deepEqual(normalMap.offset.toArray(), [0, 0])
  assert.equal(normalMap.rotation, 0)
  assert.deepEqual(normalMap.center.toArray(), [0, 0])
  assert.equal(normalMap.flipY, false)

  const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0.5, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 2)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.1, 'ClearCoatCarPaint should render visible clearcoat car-paint geometry')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > mean.g + 20 && mean.r > mean.b + 20, `ClearCoatCarPaint should render a red clearcoat material (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets ClearcoatWicker fixture loads textured clearcoat normal maps', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_CLEARCOAT_WICKER, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_clearcoat'])
  assert.deepEqual(source.buffers, [{ uri: 'ClearcoatWicker.bin', byteLength: 73728 }])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'wicker_basecolor.png',
    'wicker_normal.png',
    'wicker_occlusion-rough-metal.png',
    'clearcoat_normal.png',
  ])
  assert.deepEqual(source.samplers, [{ magFilter: 9729, minFilter: 9987 }])
  assert.deepEqual(source.meshes[0].primitives[0], {
    attributes: {
      POSITION: 1,
      NORMAL: 2,
      TEXCOORD_0: 3,
    },
    indices: 0,
    material: 0,
  })

  const materialSource = source.materials[0]
  assert.equal(materialSource.name, 'ClearcoatWicker')
  assert.deepEqual(materialSource.pbrMetallicRoughness, {
    baseColorTexture: { index: 0 },
    metallicFactor: 1,
    roughnessFactor: 1,
    metallicRoughnessTexture: { index: 2 },
  })
  assert.deepEqual(materialSource.normalTexture, { index: 1, scale: 1 })
  assert.deepEqual(materialSource.occlusionTexture, { index: 2 })
  assert.deepEqual(materialSource.extensions.KHR_materials_clearcoat, {
    clearcoatFactor: 1,
    clearcoatNormalTexture: { index: 3 },
    clearcoatRoughnessFactor: 0.1,
  })

  const gltf = await loadGltfFixture(SAMPLE_ASSET_CLEARCOAT_WICKER)
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_clearcoat'))
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'Khronos ClearcoatWicker sample should load a mesh')
  assert.equal(mesh.name, 'Sphere')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 1728)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 1728)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 1728)
  assert.equal(mesh.geometry.index?.count, 9216)
  assert.deepEqual(mesh.position.toArray(), [0, 0.5, 0])

  const material = mesh.material
  assert.equal(material.isMeshPhysicalMaterial, true)
  assert.equal(material.name, 'ClearcoatWicker')
  assert.equal(material.metalness, 1)
  assert.equal(material.roughness, 1)
  assert.equal(material.clearcoat, 1)
  assert.equal(material.clearcoatRoughness, 0.1)

  const assertLoadedTexture = (texture, name, colorSpace) => {
    assert.ok(texture?.isTexture, `${name} should load a texture`)
    assert.equal(texture.name, name)
    assert.equal(Buffer.isBuffer(texture.image), true, `${name} should load as an encoded Buffer`)
    assert.deepEqual(pngDimensions(texture.image), [512, 512])
    assert.equal(texture.wrapS, THREE.RepeatWrapping)
    assert.equal(texture.wrapT, THREE.RepeatWrapping)
    assert.equal(texture.magFilter, THREE.LinearFilter)
    assert.equal(texture.minFilter, THREE.LinearMipmapLinearFilter)
    assert.equal(texture.colorSpace, colorSpace)
    assert.equal(texture.flipY, false)
  }

  assertLoadedTexture(material.map, 'wicker_basecolor.png', THREE.SRGBColorSpace)
  assertLoadedTexture(material.normalMap, 'wicker_normal.png', THREE.NoColorSpace)
  assertLoadedTexture(material.metalnessMap, 'wicker_occlusion-rough-metal.png', THREE.NoColorSpace)
  assertLoadedTexture(material.clearcoatNormalMap, 'clearcoat_normal.png', THREE.NoColorSpace)
  assert.equal(material.metalnessMap, material.roughnessMap, 'ClearcoatWicker should reuse the ORM texture for roughness')
  assert.equal(material.metalnessMap, material.aoMap, 'ClearcoatWicker should reuse the ORM texture for occlusion')

  const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 2)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.1, 'ClearcoatWicker should render visible textured clearcoat geometry')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > mean.g + 5 && mean.g > mean.b, `ClearcoatWicker texture should contribute warm wicker pixels (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets PotOfCoals fixture loads emissive coals and copper clearcoat maps', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_POT_OF_COALS, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_clearcoat'])
  assert.deepEqual(source.buffers, [
    { uri: 'PotOfCoals.bin', byteLength: 1968084 },
  ])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'HotCoals_basecolor.jpg',
    'HotCoals_normal.jpg',
    'HotCoals_emissive.jpg',
    'HotCoals_occlusion.jpg',
    'CopperPot_basecolor.jpg',
    'CopperPot_normal.png',
    'CopperPot_orm.jpg',
    'CopperPot_clearcoat.jpg',
  ])
  assert.deepEqual(source.materials.map((material) => material.name), [
    'HotCoals',
    'CopperPot',
  ])
  assert.deepEqual(source.materials[0].emissiveTexture, { index: 2 })
  assert.deepEqual(source.materials[0].emissiveFactor, [1, 1, 1])
  assert.deepEqual(source.materials[1].extensions, {
    KHR_materials_clearcoat: {
      clearcoatFactor: 1,
      clearcoatTexture: { index: 7 },
    },
  })

  const gltf = await loadGltfFixture(SAMPLE_ASSET_POT_OF_COALS)
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_clearcoat'))
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'HotCoals',
    'CopperPot',
  ])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [38733, 15936])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [38733, 15936])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [38733, 15936])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [38733, 15936])

  const materials = new Map(meshes.map((mesh) => [mesh.material.name, mesh.material]))
  const coals = materials.get('HotCoals')
  assert.equal(coals?.isMeshStandardMaterial, true)
  assert.equal(coals.metalness, 0)
  assert.equal(coals.roughness, 0.712)
  assert.deepEqual(coals.emissive.toArray(), [1, 1, 1])
  assert.equal(coals.map.name, 'HotCoals_basecolor.jpg')
  assert.equal(coals.normalMap.name, 'HotCoals_normal.jpg')
  assert.equal(coals.emissiveMap.name, 'HotCoals_emissive.jpg')
  assert.equal(coals.aoMap.name, 'HotCoals_occlusion.jpg')
  assert.deepEqual(coals.normalScale.toArray(), [1, -1])

  const copper = materials.get('CopperPot')
  assert.equal(copper?.isMeshPhysicalMaterial, true)
  assert.equal(copper.metalness, 1)
  assert.equal(copper.roughness, 1)
  assert.equal(copper.clearcoat, 1)
  assert.equal(copper.clearcoatRoughness, 0)
  assert.equal(copper.map.name, 'CopperPot_basecolor.jpg')
  assert.equal(copper.normalMap.name, 'CopperPot_normal.png')
  assert.equal(copper.aoMap.name, 'CopperPot_orm.jpg')
  assert.equal(copper.roughnessMap.name, 'CopperPot_orm.jpg')
  assert.equal(copper.metalnessMap.name, 'CopperPot_orm.jpg')
  assert.equal(copper.clearcoatMap.name, 'CopperPot_clearcoat.jpg')
  assert.equal(copper.aoMap, copper.roughnessMap)
  assert.equal(copper.aoMap, copper.metalnessMap)

  for (const texture of [
    coals.map,
    coals.normalMap,
    coals.emissiveMap,
    coals.aoMap,
    copper.map,
    copper.normalMap,
    copper.aoMap,
    copper.clearcoatMap,
  ]) {
    assert.equal(Buffer.isBuffer(texture.image), true, `${texture.name} should load as an encoded Buffer`)
    assert.equal(texture.flipY, false)
  }
  assert.equal(coals.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(coals.emissiveMap.colorSpace, THREE.SRGBColorSpace)
  assert.equal(coals.normalMap.colorSpace, THREE.NoColorSpace)
  assert.equal(coals.aoMap.colorSpace, THREE.NoColorSpace)
  assert.equal(copper.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(copper.normalMap.colorSpace, THREE.NoColorSpace)
  assert.deepEqual(pngDimensions(copper.normalMap.image), [2048, 2048])
  assert.equal(copper.aoMap.colorSpace, THREE.NoColorSpace)
  assert.equal(copper.clearcoatMap.colorSpace, THREE.NoColorSpace)
  assert.equal(copper.aoMap.channel, 0)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 50)
  camera.position.copy(center).add(new THREE.Vector3(0, size.y * 0.15, Math.max(size.x, size.y, size.z) * 2.4))
  camera.lookAt(center)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 2.8)
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.2, 'PotOfCoals should render visible emissive coals and copper pot')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 10 && mean.g > 4, `PotOfCoals should render warm coal and copper pixels (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets IridescenceLamp fixture loads physical iridescence inputs', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_IRIDESCENCE_LAMP)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 3)

  const materials = new Map(meshes.map((mesh) => [mesh.material.name, mesh.material]))
  const base = materials.get('IridescenceLamp')
  assert.equal(base?.isMeshStandardMaterial, true)
  assert.equal(Buffer.isBuffer(base.map?.image), true, 'base color PNG should load as an encoded Buffer')
  assert.equal(Buffer.isBuffer(base.roughnessMap?.image), true, 'ORM roughness PNG should load as an encoded Buffer')
  assert.equal(Buffer.isBuffer(base.metalnessMap?.image), true, 'ORM metalness PNG should load as an encoded Buffer')
  assert.equal(Buffer.isBuffer(base.aoMap?.image), true, 'ORM occlusion PNG should load as an encoded Buffer')
  assert.equal(base.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(base.roughnessMap.colorSpace, THREE.NoColorSpace)

  const transmitted = materials.get('IridescenceLampTransmissionIridescence')
  assert.equal(transmitted?.isMeshPhysicalMaterial, true)
  assert.equal(transmitted.transmission, 1)
  assert.equal(transmitted.thickness, 0.005)
  assert.equal(transmitted.ior, 1.6)
  assert.equal(transmitted.iridescence, 1)
  assert.equal(transmitted.iridescenceIOR, 2)
  assert.deepEqual(transmitted.iridescenceThicknessRange, [385, 405])
  assert.equal(Buffer.isBuffer(transmitted.iridescenceThicknessMap?.image), true, 'iridescence thickness PNG should load as an encoded Buffer')
  assert.equal(transmitted.iridescenceThicknessMap.colorSpace, THREE.NoColorSpace)

  const iridescent = materials.get('IridescenceLampIridescence')
  assert.equal(iridescent?.isMeshPhysicalMaterial, true)
  assert.equal(iridescent.transmission, 0)
  assert.equal(iridescent.ior, 1.5)
  assert.equal(iridescent.iridescence, 1)
  assert.equal(iridescent.iridescenceIOR, 1.8)
  assert.deepEqual(iridescent.iridescenceThicknessRange, [485, 515])
  assert.equal(Buffer.isBuffer(iridescent.iridescenceThicknessMap?.image), true, 'second iridescence thickness PNG should load as an encoded Buffer')

  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 20)
  camera.position.set(0, 0.7, 2.4)
  camera.lookAt(0, 0.45, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1.0))
  const light = new THREE.DirectionalLight(0xffffff, 1.5)
  light.position.set(2, 4, 5)
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.035, 'IridescenceLamp should render visible physical-material geometry')
})

test('committed Khronos glTF Sample Assets CompareIor fixture loads transmission, volume, and IOR inputs', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_COMPARE_IOR, 'utf8'))
  assert.deepEqual(source.extensionsUsed, [
    'KHR_materials_transmission',
    'KHR_materials_volume',
    'KHR_materials_ior',
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_COMPARE_IOR)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 3, 'CompareIor should load two spheres plus checker backdrop')

  const meshesByName = new Map(meshes.map((mesh) => [mesh.name, mesh]))
  const baseline = meshesByName.get('GeoSphere001')
  const iorSphere = meshesByName.get('GeoSphere002')
  const checker = meshesByName.get('Checker')
  assert.equal(baseline?.geometry.getAttribute('position')?.count, 673)
  assert.equal(baseline.geometry.getAttribute('normal')?.count, 673)
  assert.equal(baseline.geometry.getAttribute('uv')?.count, 673)
  assert.equal(baseline.geometry.index?.count, 3840)
  assert.equal(iorSphere?.geometry.getAttribute('position')?.count, 673)
  assert.equal(iorSphere.geometry.index?.count, 3840)
  assert.equal(checker?.geometry.getAttribute('position')?.count, 4)
  assert.equal(checker.geometry.index?.count, 6)

  assert.equal(baseline.material.isMeshPhysicalMaterial, true)
  assert.equal(baseline.material.name, 'glTF Logo Transmission')
  assert.equal(baseline.material.transmission, 1)
  assert.equal(baseline.material.ior, 1.5)
  assert.equal(baseline.material.thickness, 0)
  assert.equal(baseline.material.roughness, 0.69999)
  assert.equal(iorSphere.material.isMeshPhysicalMaterial, true)
  assert.equal(iorSphere.material.name, 'glTF Logo Transmission IOR')
  assert.equal(iorSphere.material.transmission, 1)
  assert.equal(iorSphere.material.ior, 2.42)
  assert.equal(iorSphere.material.thickness, 1)
  assert.equal(iorSphere.material.attenuationDistance, 1)

  for (const material of [baseline.material, iorSphere.material]) {
    assert.equal(Buffer.isBuffer(material.map?.image), true, `${material.name} base color JPG should load as an encoded Buffer`)
    assert.equal(Buffer.isBuffer(material.roughnessMap?.image), true, `${material.name} roughness JPG should load as an encoded Buffer`)
    assert.equal(Buffer.isBuffer(material.metalnessMap?.image), true, `${material.name} metalness JPG should load as an encoded Buffer`)
    assert.equal(Buffer.isBuffer(material.transmissionMap?.image), true, `${material.name} transmission JPG should load as an encoded Buffer`)
    assert.equal(material.map.name, 'Compare_Ior_img1.jpg')
    assert.equal(material.roughnessMap.name, 'Compare_Ior_img2.jpg')
    assert.equal(material.metalnessMap.name, 'Compare_Ior_img2.jpg')
    assert.equal(material.transmissionMap.name, 'Compare_Ior_img3.jpg')
    assert.equal(material.map.colorSpace, THREE.SRGBColorSpace)
    assert.equal(material.roughnessMap.colorSpace, THREE.NoColorSpace)
    assert.equal(material.metalnessMap.colorSpace, THREE.NoColorSpace)
    assert.equal(material.transmissionMap.colorSpace, THREE.NoColorSpace)
    assert.equal(material.map.flipY, false)
    assert.equal(material.transmissionMap.flipY, false)
  }
  assert.equal(Buffer.isBuffer(checker.material.map?.image), true, 'CompareIor checker JPG should load as an encoded Buffer')
  assert.equal(checker.material.map.name, 'Compare_Ior_img0.jpg')

  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 20)
  camera.position.set(0, 0.1, 4)
  camera.lookAt(0, 0, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 1.5)
  light.position.set(2, 3, 4)
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.4, 'CompareIor should render visible physical material spheres')
  const left = meanRegion(rgba, 128, 128, 20, 48, 52, 82)
  const right = meanRegion(rgba, 128, 128, 76, 48, 108, 82)
  assert.ok(left.g > left.b + 15 && left.r > left.b + 5, `baseline transmission sphere should render lit textured pixels (${left.r}, ${left.g}, ${left.b})`)
  assert.ok(right.g > right.b + 15 && right.r > right.b + 5, `IOR transmission sphere should render lit textured pixels (${right.r}, ${right.g}, ${right.b})`)
})
