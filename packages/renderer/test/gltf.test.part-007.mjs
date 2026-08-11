import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'
import { Renderer, SAMPLE_ASSET_COMPARE_TRANSMISSION, SAMPLE_ASSET_GLASS_BROKEN_WINDOW, SAMPLE_ASSET_GLASS_HURRICANE_CANDLE_HOLDER, SAMPLE_ASSET_GLASS_VASE_FLOWERS } from './gltf.test.part-001.mjs'
import { assertVectorClose, loadGltfFixture, pngDimensions } from './gltf.test.part-028.mjs'
test('committed Khronos glTF Sample Assets CompareTransmission fixture loads alpha versus transmission variants', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_COMPARE_TRANSMISSION, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_transmission'])
  assert.equal(source.buffers[0].uri, 'CompareTransmission.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'Compare_Transmission_img0.jpg',
    'Compare_Transmission_img1.png',
    'Compare_Transmission_img2.png',
    'Compare_Transmission_img3.jpg',
  ])
  assert.deepEqual(source.materials.map((material) => [
    material.name,
    material.alphaMode ?? 'OPAQUE',
    material.pbrMetallicRoughness?.baseColorTexture?.index ?? null,
    material.pbrMetallicRoughness?.metallicRoughnessTexture?.index ?? null,
    material.extensions?.KHR_materials_transmission?.transmissionFactor ?? null,
  ]), [
    ['checker', 'OPAQUE', 2, null, null],
    ['glTF Alpha', 'BLEND', 1, 0, null],
    ['gold', 'OPAQUE', null, null, null],
    ['glTF Transmission', 'OPAQUE', null, 0, 1],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_COMPARE_TRANSMISSION)
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
    'glTF Alpha',
    'gold',
    'gold',
    'glTF Transmission',
    'checker',
  ])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [2732, 390, 390, 2732, 6])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [2732, 390, 390, 2732, 6])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [2732, 390, 390, 2732, 6])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [15744, 1920, 1920, 15744, 6])

  const [alphaShell, goldLeft, goldRight, transmissionShell, checker] = meshes.map((mesh) => mesh.material)
  assert.equal(alphaShell.transparent, true)
  assert.equal(alphaShell.isMeshStandardMaterial, true)
  assert.equal(transmissionShell.isMeshPhysicalMaterial, true)
  assert.equal(transmissionShell.transmission, 1)
  assert.equal(transmissionShell.transparent, false)
  assert.equal(goldLeft, goldRight, 'both comparison cores should share the gold material instance')
  assertVectorClose(goldLeft.color.toArray(), [
    0.8823530077934265,
    0.5921568870544434,
    0.250980406999588,
  ], 'CompareTransmission gold baseColorFactor')
  assert.equal(goldLeft.metalness, 1)
  assert.equal(goldLeft.roughness, 0.2)

  assert.equal(Buffer.isBuffer(alphaShell.map?.image), true, 'alpha-shell base-color PNG should load as an encoded Buffer')
  assert.equal(alphaShell.map.name, 'Compare_Transmission_img1.png')
  assert.deepEqual(pngDimensions(alphaShell.map.image), [2048, 1024])
  assert.equal(alphaShell.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(alphaShell.map.flipY, false)

  assert.equal(alphaShell.roughnessMap, alphaShell.metalnessMap)
  assert.equal(transmissionShell.roughnessMap, transmissionShell.metalnessMap)
  assert.equal(alphaShell.roughnessMap, transmissionShell.roughnessMap)
  assert.equal(Buffer.isBuffer(alphaShell.roughnessMap?.image), true, 'shared metallic-roughness JPEG should load as an encoded Buffer')
  assert.equal(alphaShell.roughnessMap.name, 'Compare_Transmission_img0.jpg')
  assert.equal(alphaShell.roughnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(alphaShell.roughnessMap.flipY, false)

  assert.equal(Buffer.isBuffer(checker.map?.image), true, 'checker PNG should load as an encoded Buffer')
  assert.equal(checker.map.name, 'Compare_Transmission_img2.png')
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.4, 'CompareTransmission should render visible alpha/transmission comparison geometry')
})

test('committed Khronos glTF Sample Assets GlassBrokenWindow fixture loads transmission glass with color-alpha texture', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_GLASS_BROKEN_WINDOW, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_transmission'])
  assert.deepEqual(source.buffers, [
    { uri: 'GlassBrokenWindow.bin', byteLength: 37412 },
  ])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'WindowGlass_Normal.png',
    'WindowGlass_ColorAlpha.png',
    'WindowGlass_OcclusionRoughMetal.jpg',
    'WindowFrame_Occlusion.jpg',
    'WindowClasp_Occlusion.jpg',
  ])
  assert.deepEqual(source.materials.map((material) => material.name), [
    'WindowFrame',
    'WindowGlass',
    'WindowClasp',
  ])
  assert.deepEqual(source.materials[1].extensions, {
    KHR_materials_transmission: {
      transmissionFactor: 1,
    },
  })

  const gltf = await loadGltfFixture(SAMPLE_ASSET_GLASS_BROKEN_WINDOW)
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_transmission'))
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'WindowFrame',
    'WindowsGlass',
    'WindowClasp',
  ])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [776, 4, 204])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [2208, 6, 747])

  const materials = new Map(meshes.map((mesh) => [mesh.material.name, mesh.material]))
  const frame = materials.get('WindowFrame')
  assert.equal(frame?.isMeshStandardMaterial, true)
  assert.deepEqual(frame.color.toArray(), [0.86, 0.86, 0.86])
  assert.equal(frame.metalness, 0)
  assert.equal(frame.roughness, 0.5)
  assert.equal(frame.aoMap.name, 'WindowFrame_Occlusion.jpg')
  assert.equal(frame.aoMap.channel, 0)

  const glass = materials.get('WindowGlass')
  assert.equal(glass?.isMeshPhysicalMaterial, true)
  assert.deepEqual(glass.color.toArray(), [0.6, 0.85, 1])
  assert.equal(glass.metalness, 1)
  assert.equal(glass.roughness, 0.05)
  assert.equal(glass.transmission, 1)
  assert.equal(glass.map.name, 'WindowGlass_ColorAlpha.png')
  assert.equal(glass.normalMap.name, 'WindowGlass_Normal.png')
  assert.deepEqual(glass.normalScale.toArray(), [1, -1])
  assert.equal(glass.roughnessMap.name, 'WindowGlass_OcclusionRoughMetal.jpg')
  assert.equal(glass.metalnessMap.name, 'WindowGlass_OcclusionRoughMetal.jpg')
  assert.equal(glass.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(glass.normalMap.colorSpace, THREE.NoColorSpace)
  assert.equal(glass.roughnessMap.colorSpace, THREE.NoColorSpace)
  assert.deepEqual(pngDimensions(glass.map.image), [1024, 1024])
  assert.deepEqual(pngDimensions(glass.normalMap.image), [1024, 1024])

  const clasp = materials.get('WindowClasp')
  assert.equal(clasp?.isMeshStandardMaterial, true)
  assert.deepEqual(clasp.color.toArray(), [0.18, 0.14, 0.09])
  assert.equal(clasp.metalness, 1)
  assert.equal(clasp.roughness, 0.5)
  assert.equal(clasp.aoMap.name, 'WindowClasp_Occlusion.jpg')
  assert.equal(clasp.aoMap.channel, 0)

  for (const texture of [
    frame.aoMap,
    glass.map,
    glass.normalMap,
    glass.roughnessMap,
    glass.metalnessMap,
    clasp.aoMap,
  ]) {
    assert.equal(Buffer.isBuffer(texture.image), true, `${texture.name} should load as an encoded Buffer`)
    assert.equal(texture.flipY, false)
  }

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 50)
  camera.position.copy(center).add(new THREE.Vector3(0, size.y * 0.2, Math.max(size.x, size.y, size.z) * 2.2))
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.12, 'GlassBrokenWindow should render visible frame and transmission glass')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 20 && mean.g > 20 && mean.b > 20, `GlassBrokenWindow should render lit window pixels (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets GlassHurricaneCandleHolder fixture loads volume thickness texture', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_GLASS_HURRICANE_CANDLE_HOLDER, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_transmission', 'KHR_materials_volume'])
  assert.deepEqual(source.buffers, [
    { uri: 'GlassHurricaneCandleHolder.bin', byteLength: 101108 },
  ])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'GlassHurricaneCandleHolder_orm.png',
    'GlassHurricaneCandleHolder_basecolor.png',
    'GlassHurricaneCandleHolder_thickness.png',
  ])
  assert.deepEqual(source.materials.map((material) => material.name), [
    'GlassHurricaneCandleHolder-opaque',
    'GlassHurricaneCandleHolder-glass',
  ])
  assert.deepEqual(source.materials[1].extensions, {
    KHR_materials_transmission: {
      transmissionFactor: 1,
    },
    KHR_materials_volume: {
      thicknessFactor: 0.1,
      thicknessTexture: { index: 2 },
      attenuationColor: [0.8, 0.95, 1],
      attenuationDistance: 0.001,
    },
  })

  const gltf = await loadGltfFixture(SAMPLE_ASSET_GLASS_HURRICANE_CANDLE_HOLDER)
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_volume'))
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'GlassHurricaneCandleHolder-opaque',
    'GlassHurricaneCandleHolder-glass',
  ])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [1006, 1380])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [1006, 1380])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [1006, 1380])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [5082, 7296])

  const materials = new Map(meshes.map((mesh) => [mesh.material.name, mesh.material]))
  const opaque = materials.get('GlassHurricaneCandleHolder-opaque')
  assert.equal(opaque?.isMeshStandardMaterial, true)
  assert.deepEqual(opaque.color.toArray(), [1, 1, 1])
  assert.equal(opaque.metalness, 1)
  assert.equal(opaque.roughness, 1)
  assert.equal(opaque.map.name, 'GlassHurricaneCandleHolder_basecolor.png')
  assert.equal(opaque.aoMap.name, 'GlassHurricaneCandleHolder_orm.png')
  assert.equal(opaque.roughnessMap.name, 'GlassHurricaneCandleHolder_orm.png')
  assert.equal(opaque.metalnessMap.name, 'GlassHurricaneCandleHolder_orm.png')

  const glass = materials.get('GlassHurricaneCandleHolder-glass')
  assert.equal(glass?.isMeshPhysicalMaterial, true)
  assert.deepEqual(glass.color.toArray(), [1, 1, 1])
  assert.equal(glass.metalness, 1)
  assert.equal(glass.roughness, 1)
  assert.equal(glass.transmission, 1)
  assert.equal(glass.thickness, 0.1)
  assert.equal(glass.attenuationDistance, 0.001)
  assert.deepEqual(glass.attenuationColor.toArray(), [0.8, 0.95, 1])
  assert.equal(glass.map.name, 'GlassHurricaneCandleHolder_basecolor.png')
  assert.equal(glass.aoMap.name, 'GlassHurricaneCandleHolder_orm.png')
  assert.equal(glass.roughnessMap.name, 'GlassHurricaneCandleHolder_orm.png')
  assert.equal(glass.metalnessMap.name, 'GlassHurricaneCandleHolder_orm.png')
  assert.equal(glass.thicknessMap.name, 'GlassHurricaneCandleHolder_thickness.png')

  for (const texture of [
    opaque.map,
    opaque.aoMap,
    opaque.roughnessMap,
    opaque.metalnessMap,
    glass.map,
    glass.aoMap,
    glass.roughnessMap,
    glass.metalnessMap,
    glass.thicknessMap,
  ]) {
    assert.equal(Buffer.isBuffer(texture.image), true, `${texture.name} should load as an encoded Buffer`)
    assert.deepEqual(pngDimensions(texture.image), [2048, 2048])
    assert.equal(texture.flipY, false)
  }
  assert.equal(opaque.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(glass.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(opaque.aoMap.colorSpace, THREE.NoColorSpace)
  assert.equal(glass.thicknessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(opaque.aoMap.channel, 0)
  assert.equal(glass.aoMap.channel, 0)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 50)
  camera.position.copy(center).add(new THREE.Vector3(0, size.y * 0.15, Math.max(size.x, size.y, size.z) * 2.2))
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.12, 'GlassHurricaneCandleHolder should render visible metal and volume glass')
  const mean = meanRgba(rgba)
  assert.ok(mean.b > 10 && mean.g > 10, `GlassHurricaneCandleHolder should render lit blue-tinted glass pixels (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets GlassVaseFlowers fixture loads alpha flowers and transmission vase', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_GLASS_VASE_FLOWERS, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_transmission', 'KHR_materials_volume'])
  assert.deepEqual(source.buffers, [
    { uri: 'GlassVaseFlowers.bin', byteLength: 178908 },
  ])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'shrub_sorrel_01_normal_1k.jpg',
    'shrub_sorrel_01_color_1k.png',
    'shrub_sorrel_01_rough_1k.jpg',
    'glass_vase_thickness_1k.png',
  ])
  assert.deepEqual(source.materials.map((material) => [
    material.name,
    material.alphaMode ?? 'OPAQUE',
    material.doubleSided ?? false,
    material.pbrMetallicRoughness?.baseColorTexture?.index ?? null,
    material.normalTexture?.index ?? null,
    material.pbrMetallicRoughness?.metallicRoughnessTexture?.index ?? null,
    material.extensions?.KHR_materials_transmission?.transmissionFactor ?? null,
    material.extensions?.KHR_materials_volume?.thicknessFactor ?? null,
  ]), [
    ['Flowers', 'MASK', true, 1, 0, 2, null, null],
    ['GlassAlpha', 'BLEND', false, null, null, null, null, null],
    ['GlassTransmission', 'OPAQUE', false, null, null, null, 1, 0.075],
  ])
  assert.deepEqual(source.materials[2].extensions, {
    KHR_materials_transmission: { transmissionFactor: 1 },
    KHR_materials_volume: { thicknessFactor: 0.075, thicknessTexture: { index: 3 } },
  })

  const gltf = await loadGltfFixture(SAMPLE_ASSET_GLASS_VASE_FLOWERS)
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_volume'))
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'Flowers1',
    'GlassAlpha',
    'Flowers2',
    'GlassTransmission',
  ])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [2561, 1714, 2561, 1714])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [2561, 1714, 2561, 1714])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [2561, 1714, 2561, 1714])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [11454, 9600, 11454, 9600])

  const materials = new Map(meshes.map((mesh) => [mesh.material.name, mesh.material]))
  const flowers = materials.get('Flowers')
  assert.equal(flowers?.isMeshStandardMaterial, true)
  assert.equal(flowers.metalness, 0)
  assert.equal(flowers.roughness, 1)
  assert.equal(flowers.alphaTest, 0.5)
  assert.equal(flowers.side, THREE.DoubleSide)
  assert.equal(flowers.transparent, false)
  assert.equal(flowers.map.name, 'shrub_sorrel_01_color_1k.png')
  assert.equal(flowers.normalMap.name, 'shrub_sorrel_01_normal_1k.jpg')
  assert.deepEqual(flowers.normalScale.toArray(), [1, -1])
  assert.equal(flowers.roughnessMap.name, 'shrub_sorrel_01_rough_1k.jpg')
  assert.equal(flowers.metalnessMap.name, 'shrub_sorrel_01_rough_1k.jpg')

  const alphaGlass = materials.get('GlassAlpha')
  assert.equal(alphaGlass?.isMeshStandardMaterial, true)
  assert.deepEqual(alphaGlass.color.toArray(), [0.15, 0.15, 0.15])
  assert.equal(alphaGlass.transparent, true)
  assert.equal(alphaGlass.roughness, 0)
  assert.equal(alphaGlass.metalness, 0)

  const transmissionGlass = materials.get('GlassTransmission')
  assert.equal(transmissionGlass?.isMeshPhysicalMaterial, true)
  assert.equal(transmissionGlass.transmission, 1)
  assert.equal(transmissionGlass.thickness, 0.075)
  assert.equal(transmissionGlass.ior, 1.5)
  assert.equal(transmissionGlass.roughness, 0)
  assert.equal(transmissionGlass.metalness, 0)
  assert.equal(transmissionGlass.thicknessMap.name, 'glass_vase_thickness_1k.jpg')

  assert.equal(Buffer.isBuffer(flowers.map.image), true, 'flower base-color PNG should load as an encoded Buffer')
  assert.deepEqual(pngDimensions(flowers.map.image), [1024, 1024])
  assert.equal(flowers.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(Buffer.isBuffer(flowers.normalMap.image), true, 'flower normal JPEG should load as an encoded Buffer')
  assert.equal(flowers.normalMap.colorSpace, THREE.NoColorSpace)
  assert.equal(Buffer.isBuffer(flowers.roughnessMap.image), true, 'flower roughness JPEG should load as an encoded Buffer')
  assert.equal(flowers.roughnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(Buffer.isBuffer(transmissionGlass.thicknessMap.image), true, 'vase thickness PNG should load as an encoded Buffer')
  assert.deepEqual(pngDimensions(transmissionGlass.thicknessMap.image), [1024, 1024])
  assert.equal(transmissionGlass.thicknessMap.colorSpace, THREE.NoColorSpace)
  for (const texture of [
    flowers.map,
    flowers.normalMap,
    flowers.roughnessMap,
    flowers.metalnessMap,
    transmissionGlass.thicknessMap,
  ]) {
    assert.equal(texture.flipY, false)
  }

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 50)
  camera.position.copy(center).add(new THREE.Vector3(0, size.y * 0.15, Math.max(size.x, size.y, size.z) * 2.3))
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.05, 'GlassVaseFlowers should render visible alpha flowers and glass vase')
})
