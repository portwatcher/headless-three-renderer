import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'
import { Renderer, SAMPLE_ASSET_A_BEAUTIFUL_GAME, SAMPLE_ASSET_CAR_CONCEPT, SAMPLE_ASSET_MOSQUITO_IN_AMBER, SAMPLE_ASSET_POT_OF_COALS_ANIMATION_POINTER, SAMPLE_ASSET_SCATTERING_SKULL, SAMPLE_ASSET_SPEC_GLOSS_VS_METAL_ROUGH, SAMPLE_ASSET_STAINED_GLASS_LAMP } from './gltf.test.part-001.mjs'
import { assertVectorClose, captureConsoleWarn, frameSceneCamera, loadGltfFixture, pngDimensions, uniqueMaterials } from './gltf.test.part-028.mjs'
test('committed Khronos glTF Sample Assets ScatteringSkull fixture loads volume-scatter metadata and thickness map', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_SCATTERING_SKULL, 'utf8'))
  assert.deepEqual(source.extensionsUsed, [
    'KHR_xmp_json_ld',
    'KHR_materials_diffuse_transmission',
    'KHR_materials_volume',
    'KHR_materials_dispersion',
    'KHR_materials_volume_scatter',
    'KHR_materials_ior',
  ])
  assert.equal(source.extensions.KHR_xmp_json_ld.packets[0]['dc:title'], 'Skattering Skull')
  assert.deepEqual(source.buffers, [{ uri: 'ScatteringSkull_binary.bin', byteLength: 6964692 }])
  assert.deepEqual(source.images.map((image) => image.uri), ['ScatteringSkull_images/aoThickness.png'])
  assert.equal(source.materials[0].extensions.KHR_materials_volume.thicknessFactor, 1)
  assert.equal(source.materials[0].extensions.KHR_materials_dispersion.dispersion, 0.5699999928474426)
  assert.equal(source.materials[0].extensions.KHR_materials_ior.ior, 1.3799999952316284)

  const gltf = await loadGltfFixture(SAMPLE_ASSET_SCATTERING_SKULL)
  const skull = gltf.scene.getObjectByName('Skull')
  assert.ok(skull?.isMesh, 'ScatteringSkull should load its skull mesh')
  assert.equal(skull.geometry.getAttribute('position')?.count, 97880)
  assert.equal(skull.geometry.index?.count, 566613)
  const material = skull.material
  assert.equal(material.name, 'subsurface_material')
  assert.equal(material.isMeshPhysicalMaterial, true)
  assert.equal(material.thickness, 1)
  assert.equal(material.dispersion, 0.5699999928474426)
  assert.equal(material.ior, 1.3799999952316284)
  assertVectorClose(material.attenuationColor.toArray(), [0.3678794503211975, 0.4857314527034759, 0.3964099884033203], 'ScatteringSkull attenuationColor')
  assert.equal(material.attenuationDistance, 0.016891848295927048)
  assert.deepEqual(material.userData.gltfExtensions.KHR_materials_diffuse_transmission, {
    diffuseTransmissionFactor: 1,
  })
  assert.deepEqual(material.userData.gltfExtensions.KHR_materials_volume_scatter, {
    multiscatterColorFactor: [0.16827136278152466, 0.5271198749542236, 0.5906253457069397],
  })
  assert.equal(material.aoMap?.name, 'aoThicknessTexture')
  assert.equal(material.thicknessMap, material.aoMap)
  assert.deepEqual(pngDimensions(material.thicknessMap.image), [2048, 2048])

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.75))
  const light = new THREE.DirectionalLight(0xffffff, 1.6)
  light.position.set(2, 4, 5)
  gltf.scene.add(light)
  const camera = frameSceneCamera(gltf.scene)
  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 128,
    height: 128,
    format: 'rgba',
    background: [1, 1, 1],
  })
  assert.ok(nonBackgroundRatio(rgba, [255, 255, 255], 3) > 0.2, 'ScatteringSkull should render visible skull geometry')
})

test('committed Khronos glTF Sample Assets SpecGlossVsMetalRough fixture preserves required spec-gloss extension metadata', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_SPEC_GLOSS_VS_METAL_ROUGH, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_pbrSpecularGlossiness'])
  assert.deepEqual(source.extensionsRequired, ['KHR_materials_pbrSpecularGlossiness'])
  assert.deepEqual(source.buffers, [
    { uri: 'WaterBottle.bin', byteLength: 149412 },
    { byteLength: 536, uri: 'SpecGlossVsMetalRoughLabel.bin' },
  ])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'WaterBottle_baseColor.png',
    'WaterBottle_roughnessMetallic.png',
    'WaterBottle_normal.png',
    'WaterBottle_emissive.png',
    'WaterBottle_occlusion.png',
    'WaterBottle_diffuse.png',
    'WaterBottle_specularGlossiness.png',
    'SpecGlossVsMetalRough.png',
  ])

  const { result: gltf, warnings } = await captureConsoleWarn(() => loadGltfFixture(SAMPLE_ASSET_SPEC_GLOSS_VS_METAL_ROUGH))
  assert.ok(
    warnings.some((warning) => warning.includes('KHR_materials_pbrSpecularGlossiness')),
    'GLTFLoader should warn that the required legacy spec-gloss extension is unknown',
  )
  const specGloss = gltf.scene.getObjectByName('WaterBottle_SpecGloss')
  const metalRough = gltf.scene.getObjectByName('WaterBottle_MR')
  assert.ok(specGloss?.isMesh, 'SpecGlossVsMetalRough should load the spec-gloss bottle mesh')
  assert.ok(metalRough?.isMesh, 'SpecGlossVsMetalRough should load the metal-rough bottle mesh')
  assert.equal(specGloss.geometry.getAttribute('position')?.count, 2549)
  assert.equal(specGloss.geometry.index?.count, 13530)
  assert.equal(metalRough.geometry.index?.count, 13530)
  assert.deepEqual(specGloss.material.userData.gltfExtensions.KHR_materials_pbrSpecularGlossiness, {
    diffuseTexture: { index: 5 },
    specularGlossinessTexture: { index: 6 },
  })
  assert.equal(specGloss.material.normalMap?.name, 'WaterBottle_normal.png')
  assert.deepEqual(pngDimensions(specGloss.material.normalMap.image), [2048, 2048])
  assert.equal(specGloss.material.aoMap?.name, 'WaterBottle_occlusion.png')
  assert.equal(specGloss.material.emissiveMap?.name, 'WaterBottle_emissive.png')
  assert.equal(metalRough.material.map?.name, 'WaterBottle_baseColor.png')
  assert.deepEqual(pngDimensions(metalRough.material.map.image), [2048, 2048])
  assert.equal(metalRough.material.roughnessMap?.name, 'WaterBottle_roughnessMetallic.png')
  assert.equal(metalRough.material.metalnessMap, metalRough.material.roughnessMap)

  const label = gltf.scene.getObjectByName('MetalRoughLabel')
  assert.equal(label?.material?.map?.name, 'SpecGlossVsMetalRough.png')
  assert.deepEqual(pngDimensions(label.material.map.image), [512, 128])

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.65))
  const light = new THREE.DirectionalLight(0xffffff, 1.7)
  light.position.set(2, 4, 5)
  gltf.scene.add(light)
  const camera = frameSceneCamera(gltf.scene, { distance: 2.1 })
  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 128,
    height: 128,
    format: 'rgba',
    background: [1, 1, 1],
  })
  assert.ok(nonBackgroundRatio(rgba, [255, 255, 255], 3) > 0.25, 'SpecGlossVsMetalRough should render visible bottle comparison geometry')
  const mean = meanRgba(rgba)
  assert.ok(mean.r < 240 && mean.g < 240 && mean.b < 235, `SpecGlossVsMetalRough should render non-white bottle pixels (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets CarConcept fixture loads variants, clearcoat paint, emissive dash, and glass', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_CAR_CONCEPT, 'utf8'))
  assert.deepEqual(source.extensionsUsed, [
    'KHR_materials_clearcoat',
    'KHR_materials_emissive_strength',
    'KHR_materials_iridescence',
    'KHR_materials_transmission',
    'KHR_materials_variants',
    'KHR_texture_transform',
  ])
  assert.deepEqual(source.buffers, [{ byteLength: 8670516, uri: 'CarConcept.data.bin' }])
  assert.equal(source.images.length, 14)
  assert.deepEqual(source.extensions.KHR_materials_variants.variants.map((variant) => variant.name), [
    'Carmine Candy',
    'Pearly Swirly',
    'Torched Graphite',
  ])
  const variantPrimitiveCount = source.meshes.reduce(
    (count, mesh) => count + mesh.primitives.filter((primitive) => primitive.extensions?.KHR_materials_variants).length,
    0,
  )
  assert.equal(variantPrimitiveCount, 25)
  assert.deepEqual(source.materials.find((material) => material.name === 'Glass').extensions, {
    KHR_materials_transmission: { transmissionFactor: 1 },
  })

  const gltf = await loadGltfFixture(SAMPLE_ASSET_CAR_CONCEPT)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 109)

  const windshield = gltf.scene.getObjectByName('BodyWindshield')
  assert.ok(windshield?.isMesh, 'CarConcept should load windshield glass')
  assert.equal(windshield.geometry.getAttribute('position')?.count, 1535)
  assert.equal(windshield.geometry.index?.count, 8640)
  assert.equal(windshield.material.name, 'Glass')
  assert.equal(windshield.material.isMeshPhysicalMaterial, true)
  assert.equal(windshield.material.transmission, 1)
  assert.equal(windshield.material.ior, 1.5)

  const variantPanel = gltf.scene.getObjectByName('BodyRoofPanel')
  assert.equal(variantPanel?.userData.gltfExtensions.KHR_materials_variants.mappings.length, 3)
  const paint = uniqueMaterials(gltf.scene).find((material) => material.name === 'Paint 2 Carmine')
  assert.equal(paint?.isMeshPhysicalMaterial, true)
  assert.equal(paint.clearcoat, 0.25)
  assert.equal(paint.metalness, 1)
  assert.equal(paint.normalMap?.name, 'Powdercoat_N.png')
  assert.deepEqual(pngDimensions(paint.normalMap.image), [128, 128])
  const dashboard = uniqueMaterials(gltf.scene).find((material) => material.name === 'Dashboard')
  assert.equal(dashboard?.emissiveIntensity, 3)
  assert.equal(dashboard.emissiveMap?.name, 'Dash_E.png')
  assert.deepEqual(pngDimensions(dashboard.emissiveMap.image), [1024, 256])
  const mechanical = uniqueMaterials(gltf.scene).find((material) => material.name === 'Mechanical')
  assert.equal(mechanical?.normalMap?.name, 'Mechanical_N.png')
  assert.deepEqual(pngDimensions(mechanical.normalMap.image), [512, 512])
  assert.equal(mechanical.roughnessMap?.name, 'Mechanical_ORM.png')
  assert.equal(mechanical.metalnessMap, mechanical.roughnessMap)

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.55))
  const light = new THREE.DirectionalLight(0xffffff, 1.8)
  light.position.set(2, 4, 5)
  gltf.scene.add(light)
  const camera = frameSceneCamera(gltf.scene, { distance: 2.8, yOffset: 0.3 })
  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 128,
    height: 128,
    format: 'rgba',
    background: [1, 1, 1],
  })
  assert.ok(nonBackgroundRatio(rgba, [255, 255, 255], 3) > 0.1, 'CarConcept should render visible vehicle geometry')
  const mean = meanRgba(rgba)
  assert.ok(mean.r < 245 && mean.g < 238 && mean.b < 238, `CarConcept should render non-white car pixels (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets PotOfCoalsAnimationPointer fixture preserves animation pointer source metadata', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_POT_OF_COALS_ANIMATION_POINTER, 'utf8'))
  assert.deepEqual(source.extensionsUsed, [
    'KHR_animation_pointer',
    'KHR_materials_clearcoat',
    'KHR_materials_specular',
    'KHR_materials_transmission',
    'KHR_materials_volume',
    'KHR_texture_transform',
  ])
  assert.deepEqual(source.extensionsRequired, [
    'KHR_materials_specular',
    'KHR_materials_transmission',
    'KHR_materials_volume',
    'KHR_texture_transform',
  ])
  assert.deepEqual(source.buffers, [{ byteLength: 1977628, uri: 'PotOfCoalsAnimationPointer.bin' }])
  assert.equal(source.animations[0].channels.length, 2)
  assert.deepEqual(source.animations[0].channels.map((channel) => channel.target.extensions.KHR_animation_pointer.pointer), [
    '/materials/2/normalTexture/extensions/KHR_texture_transform/rotation',
    '/materials/2/extensions/KHR_materials_volume/thicknessTexture/extensions/KHR_texture_transform/rotation',
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_POT_OF_COALS_ANIMATION_POINTER)
  assert.equal(gltf.animations.length, 1)
  assert.equal(gltf.animations[0].duration, 0)
  assert.equal(gltf.animations[0].tracks.length, 0, 'current Three.js loader exposes unsupported material animation pointers as an empty clip')
  const hotCoals = gltf.scene.getObjectByName('HotCoals')
  const copperPot = gltf.scene.getObjectByName('CopperPot')
  const heatDome = gltf.scene.getObjectByName('HeatDome')
  assert.equal(hotCoals?.geometry.getAttribute('position')?.count, 38733)
  assert.equal(copperPot?.geometry.getAttribute('position')?.count, 15936)
  assert.equal(heatDome?.geometry.getAttribute('position')?.count, 264)
  assert.equal(hotCoals.material.map?.name, 'HotCoals_basecolor.jpg')
  assert.equal(Buffer.isBuffer(hotCoals.material.emissiveMap?.image), true, 'HotCoals emissive JPEG should load as an encoded Buffer')
  assert.equal(copperPot.material.clearcoat, 1)
  assert.equal(copperPot.material.clearcoatMap?.name, 'CopperPot_clearcoat.jpg')
  assert.equal(copperPot.material.normalMap?.name, 'CopperPot_normal.png')
  assert.deepEqual(pngDimensions(copperPot.material.normalMap.image), [2048, 2048])
  assert.equal(heatDome.material.transmission, 1)
  assert.equal(heatDome.material.thickness, 0.01999)
  assert.equal(heatDome.material.thicknessMap?.name, 'Heatdome_thickness.jpg')

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.6))
  const light = new THREE.DirectionalLight(0xffffff, 1.8)
  light.position.set(2, 4, 5)
  gltf.scene.add(light)
  const camera = frameSceneCamera(gltf.scene)
  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 128,
    height: 128,
    format: 'rgba',
    background: [1, 1, 1],
  })
  assert.ok(nonBackgroundRatio(rgba, [255, 255, 255], 3) > 0.25, 'PotOfCoalsAnimationPointer should render visible coals and pot geometry')
  const mean = meanRgba(rgba)
  assert.ok(mean.r < 230 && mean.g < 220 && mean.b < 215, `PotOfCoalsAnimationPointer should render warm coal/pot pixels (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets StainedGlassLamp fixture loads glass variants and transmission materials', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_STAINED_GLASS_LAMP, 'utf8'))
  assert.deepEqual(source.extensionsUsed, [
    'KHR_materials_transmission',
    'KHR_materials_clearcoat',
    'KHR_materials_variants',
    'KHR_materials_ior',
    'KHR_materials_volume',
  ])
  assert.deepEqual(source.buffers, [{ byteLength: 1847592, uri: 'StainedGlassLamp.bin' }])
  assert.deepEqual(source.extensions.KHR_materials_variants.variants.map((variant) => variant.name), ['Lamp on', 'Lamp off'])
  const variantPrimitiveCount = source.meshes.reduce(
    (count, mesh) => count + mesh.primitives.filter((primitive) => primitive.extensions?.KHR_materials_variants).length,
    0,
  )
  assert.equal(variantPrimitiveCount, 5)
  assert.equal(source.images.length, 19)

  const gltf = await loadGltfFixture(SAMPLE_ASSET_STAINED_GLASS_LAMP)
  const stainedGlass = gltf.scene.getObjectByName('stainedglass')
  assert.ok(stainedGlass?.isMesh, 'StainedGlassLamp should load stained glass mesh')
  assert.equal(stainedGlass.geometry.getAttribute('position')?.count, 2616)
  assert.equal(stainedGlass.geometry.index?.count, 12096)
  assert.equal(stainedGlass.userData.gltfExtensions.KHR_materials_variants.mappings.length, 2)
  const material = stainedGlass.material
  assert.equal(material.name, 'stainedglass')
  assert.equal(material.isMeshPhysicalMaterial, true)
  assert.equal(material.transmission, 1)
  assert.equal(material.clearcoat, 1)
  assert.equal(material.ior, 1.5)
  assert.equal(material.map?.name, 'StainedGlassLamp_glass_basecolor-alpha.png')
  assert.deepEqual(pngDimensions(material.map.image), [2048, 1024])
  assert.equal(material.normalMap?.name, 'StainedGlassLamp_glass_normal.png')
  assert.deepEqual(pngDimensions(material.normalMap.image), [2048, 1024])
  assert.equal(material.clearcoatMap?.name, 'StainedGlassLamp_glass_transmission-clearcoat.png')
  assert.deepEqual(pngDimensions(material.clearcoatMap.image), [2048, 1024])

  const amber = gltf.scene.getObjectByName('amberbeads')
  assert.equal(amber?.material?.transmission, 1)
  assert.equal(amber.material.thickness, 0.02)
  assert.equal(amber.material.ior, 1.4)
  const red = gltf.scene.getObjectByName('redgems')
  assert.equal(red?.material?.transmission, 1)
  assert.equal(red.material.thickness, 0.03)
  assert.equal(red.material.ior, 1.52)
  const grill = gltf.scene.getObjectByName('grill')
  assert.equal(grill?.material?.map?.name, 'StainedGlassLamp_grill_basecolor-alpha.png')
  assert.deepEqual(pngDimensions(grill.material.map.image), [2048, 2048])

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.6))
  const light = new THREE.DirectionalLight(0xffffff, 1.8)
  light.position.set(2, 4, 5)
  gltf.scene.add(light)
  const camera = frameSceneCamera(gltf.scene)
  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 128,
    height: 128,
    format: 'rgba',
    background: [1, 1, 1],
  })
  assert.ok(nonBackgroundRatio(rgba, [255, 255, 255], 3) > 0.12, 'StainedGlassLamp should render visible lamp geometry')
  const mean = meanRgba(rgba)
  assert.ok(mean.r < 245 && mean.g < 245 && mean.b < 245, `StainedGlassLamp should render non-white glass and metal pixels (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets ABeautifulGame fixture loads chessboard transmission pieces', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_A_BEAUTIFUL_GAME, 'utf8'))
  assert.deepEqual(source.extensionsUsed, [
    'KHR_materials_transmission',
    'KHR_materials_volume',
  ])
  assert.deepEqual(source.buffers, [{ byteLength: 10829440, uri: 'ABeautifulGame.bin' }])
  assert.equal(source.nodes.length, 49)
  assert.equal(source.meshes.length, 15)
  assert.equal(source.materials.length, 15)
  assert.equal(source.images.length, 33)
  assert.deepEqual(source.materials.find((material) => material.name === 'Pawn_Top_White').extensions, {
    KHR_materials_transmission: { transmissionFactor: 1 },
    KHR_materials_volume: {
      attenuationColor: [0.800000011920929, 0.800000011920929, 0.800000011920929],
      thicknessFactor: 0.2199999988079071,
    },
  })

  const gltf = await loadGltfFixture(SAMPLE_ASSET_A_BEAUTIFUL_GAME)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 49)
  assert.equal(uniqueMaterials(gltf.scene).length, 15)

  const king = gltf.scene.getObjectByName('King_B')
  assert.ok(king?.isMesh, 'ABeautifulGame should load the black king mesh')
  assert.equal(king.geometry.getAttribute('position')?.count, 28901)
  assert.equal(king.geometry.index?.count, 121440)
  assert.equal(king.material.name, 'King_Black')
  assert.equal(king.material.map?.name, 'King_black_base_color')
  assert.equal(Buffer.isBuffer(king.material.map.image), true, 'King base-color JPEG should load as an encoded Buffer')
  assert.equal(king.material.normalMap?.name, 'King_black_normal')
  assert.equal(king.material.roughnessMap?.name, 'King_black_ORM')
  assert.equal(king.material.metalnessMap, king.material.roughnessMap)
  assert.equal(king.material.aoMap, king.material.roughnessMap)

  const chessboard = gltf.scene.getObjectByName('Chessboard')
  assert.equal(chessboard?.geometry.getAttribute('position')?.count, 108441)
  assert.equal(chessboard.geometry.index?.count, 277248)
  assert.equal(chessboard.material.map?.name, 'Chessboard_base_color')
  assert.equal(chessboard.material.normalMap?.name, 'Chessboard_normal')

  const pawnTop = gltf.scene.getObjectByName('Pawn_Top_W1')
  assert.ok(pawnTop?.isMesh, 'ABeautifulGame should load a translucent white pawn top')
  assert.equal(pawnTop.geometry.getAttribute('position')?.count, 1131)
  assert.equal(pawnTop.geometry.index?.count, 6624)
  assert.equal(pawnTop.material.name, 'Pawn_Top_White')
  assert.equal(pawnTop.material.isMeshPhysicalMaterial, true)
  assert.equal(pawnTop.material.transmission, 1)
  assert.equal(pawnTop.material.thickness, 0.2199999988079071)
  assert.equal(pawnTop.material.ior, 1.5)
  assertVectorClose(pawnTop.material.attenuationColor.toArray(), [
    0.800000011920929,
    0.800000011920929,
    0.800000011920929,
  ], 'ABeautifulGame pawn attenuation color')
  assert.equal(pawnTop.material.normalMap?.name, 'Pawn_normal')
  assert.equal(pawnTop.material.roughnessMap?.name, 'Pawn_ORM')
  assert.equal(pawnTop.material.metalnessMap, pawnTop.material.roughnessMap)

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.6))
  const light = new THREE.DirectionalLight(0xffffff, 1.8)
  light.position.set(2, 4, 5)
  gltf.scene.add(light)
  const camera = frameSceneCamera(gltf.scene)
  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [1, 1, 1],
  })
  assert.ok(nonBackgroundRatio(rgba, [255, 255, 255], 3) > 0.12, 'ABeautifulGame should render visible chessboard geometry')
  const mean = meanRgba(rgba)
  assert.ok(mean.r < 245 && mean.g < 245 && mean.b < 245, `ABeautifulGame should render non-white chess pixels (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets MosquitoInAmber fixture loads amber transmission volume', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_MOSQUITO_IN_AMBER, 'utf8'))
  assert.deepEqual(source.extensionsUsed, [
    'KHR_materials_transmission',
    'KHR_materials_ior',
    'KHR_materials_volume',
  ])
  assert.deepEqual(source.buffers, [{ name: 'MosquitoInAmber', byteLength: 1068732, uri: 'MosquitoInAmber.bin' }])
  assert.equal(source.meshes.length, 3)
  assert.equal(source.materials.length, 3)
  assert.deepEqual(source.images.map((image) => image.uri), [
    'MosquitoInAmber0.jpg',
    'MosquitoInAmber1.png',
    'MosquitoInAmber2.png',
    'MosquitoInAmber3.jpg',
    'MosquitoInAmber4.jpg',
  ])
  assert.deepEqual(source.materials.find((material) => material.name === 'material').extensions, {
    KHR_materials_transmission: { transmissionFactor: 0.75 },
    KHR_materials_ior: { ior: 1.55 },
    KHR_materials_volume: { thicknessFactor: 0.9 },
  })

  const gltf = await loadGltfFixture(SAMPLE_ASSET_MOSQUITO_IN_AMBER)
  const materials = uniqueMaterials(gltf.scene)
  assert.equal(materials.length, 3)

  const amber = gltf.scene.getObjectByName('5_amber_lr_PBR_0')
  assert.ok(amber?.isMesh, 'MosquitoInAmber should load the amber shell mesh')
  assert.equal(amber.geometry.getAttribute('position')?.count, 1085)
  assert.equal(amber.geometry.index?.count, 5556)
  assert.equal(amber.material.name, 'material')
  assert.equal(amber.material.isMeshPhysicalMaterial, true)
  assert.equal(amber.material.transmission, 0.75)
  assert.equal(amber.material.thickness, 0.9)
  assert.equal(amber.material.ior, 1.55)
  assert.equal(amber.material.roughness, 0.5)
  assert.equal(amber.material.map?.name, 'MosquitoInAmber0.jpg')
  assert.equal(Buffer.isBuffer(amber.material.map.image), true, 'amber base-color JPEG should load as an encoded Buffer')
  assert.equal(amber.material.roughnessMap?.name, 'MosquitoInAmber1.png')
  assert.deepEqual(pngDimensions(amber.material.roughnessMap.image), [4096, 4096])
  assert.equal(amber.material.metalnessMap, amber.material.roughnessMap)
  assert.equal(amber.material.normalMap?.name, 'MosquitoInAmber2.png')
  assert.deepEqual(pngDimensions(amber.material.normalMap.image), [2048, 2048])

  const shards = gltf.scene.getObjectByName('6_eclats_eclats_0')
  assert.equal(shards?.geometry.getAttribute('position')?.count, 3057)
  assert.equal(shards.geometry.index?.count, 3189)
  assert.equal(shards.material.metalness, 1)
  assert.equal(shards.material.roughness, 0.3922348485)
  const mosquito = gltf.scene.getObjectByName('2_mosquito_lr_originalo_material_0_0')
  assert.equal(mosquito?.geometry.getAttribute('position')?.count, 14536)
  assert.equal(mosquito.geometry.index?.count, 34302)
  assert.equal(mosquito.material.map?.name, 'MosquitoInAmber3.jpg')
  assert.equal(mosquito.material.normalMap?.name, 'MosquitoInAmber4.jpg')

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.6))
  const light = new THREE.DirectionalLight(0xffffff, 1.8)
  light.position.set(2, 4, 5)
  gltf.scene.add(light)
  const camera = frameSceneCamera(gltf.scene)
  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [1, 1, 1],
  })
  assert.ok(nonBackgroundRatio(rgba, [255, 255, 255], 3) > 0.3, 'MosquitoInAmber should render visible amber and insect geometry')
  const mean = meanRgba(rgba)
  assert.ok(mean.r < 250 && mean.g < 250 && mean.b < 250, `MosquitoInAmber should render non-white amber pixels (${mean.r}, ${mean.g}, ${mean.b})`)
})
