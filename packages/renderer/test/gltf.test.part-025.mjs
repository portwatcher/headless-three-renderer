import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'
import { Renderer, SAMPLE_ASSET_CHRONOGRAPH_WATCH, SAMPLE_ASSET_DIFFUSE_TRANSMISSION_PLANT, SAMPLE_ASSET_DIFFUSE_TRANSMISSION_TEACUP, SAMPLE_ASSET_DIFFUSE_TRANSMISSION_TEST, SAMPLE_ASSET_MANDARIN_ORANGE, SAMPLE_ASSET_SHEEN_WOOD_LEATHER_SOFA, SAMPLE_ASSET_TRIANGLE_WITHOUT_INDICES, SAMPLE_ASSET_USD_SHADER_BALL_FOR_GLTF } from './gltf.test.part-001.mjs'
import { assertWebpBuffer, findFirst, frameSceneCamera, loadGltfFixture, pngDimensions, uniqueMaterials, vectorFromAttribute } from './gltf.test.part-028.mjs'
test('committed Khronos glTF Sample Assets TriangleWithoutIndices fixture loads non-indexed geometry', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_TRIANGLE_WITHOUT_INDICES)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'Khronos TriangleWithoutIndices sample should load a mesh')
  assert.equal(mesh.geometry.index, null)
  assert.equal(mesh.geometry.getAttribute('position')?.count, 3)
  assert.deepEqual(vectorFromAttribute(mesh.geometry.getAttribute('position'), 0), [0, 0, 0])
  assert.deepEqual(vectorFromAttribute(mesh.geometry.getAttribute('position'), 1), [1, 0, 0])
  assert.deepEqual(vectorFromAttribute(mesh.geometry.getAttribute('position'), 2), [0, 1, 0])

  mesh.material = new THREE.MeshBasicMaterial({ color: 0xffffff })
  mesh.position.set(-0.5, -0.5, 0)

  const camera = new THREE.OrthographicCamera(-0.6, 0.6, 0.6, -0.6, 0.01, 10)
  camera.position.set(0, 0, 2)
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.25, 'TriangleWithoutIndices sample should render visible non-indexed geometry')
})

test('committed Khronos glTF Sample Assets DiffuseTransmissionTest fixture preserves diffuse-transmission metadata', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_DIFFUSE_TRANSMISSION_TEST, 'utf8'))
  assert.deepEqual(source.extensionsUsed, [
    'KHR_materials_diffuse_transmission',
    'KHR_materials_unlit',
    'KHR_lights_punctual',
  ])
  assert.deepEqual(source.extensionsRequired, ['KHR_materials_unlit', 'KHR_lights_punctual'])
  assert.deepEqual(source.buffers, [{ byteLength: 192896, uri: 'DiffuseTransmissionTest.bin' }])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'DiffuseTransmissionTexture.png',
    'glTF-Green.png',
    'Khronos-Red.png',
  ])
  assert.equal(source.materials.filter((material) => material.extensions?.KHR_materials_diffuse_transmission).length, 20)
  assert.equal(source.materials.filter((material) => material.extensions?.KHR_materials_unlit).length, 8)

  const gltf = await loadGltfFixture(SAMPLE_ASSET_DIFFUSE_TRANSMISSION_TEST)
  assert.deepEqual(gltf.parser.json.extensionsUsed, source.extensionsUsed)
  const meshes = []
  const materials = uniqueMaterials(gltf.scene)
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 33)
  assert.equal(materials.length, 29)

  const factor = gltf.scene.getObjectByName('Plane005')?.material
  assert.equal(factor?.name, 'Factor 1.0')
  assert.deepEqual(factor.userData.gltfExtensions.KHR_materials_diffuse_transmission, {
    diffuseTransmissionColorFactor: [1, 1, 1],
    diffuseTransmissionFactor: 1,
  })
  const colorTexture = materials.find((material) => material.name === 'ColorTexture 1.0')
  assert.equal(colorTexture?.map?.name, 'glTF-Green.png')
  assert.deepEqual(pngDimensions(colorTexture.map.image), [512, 512])
  const unlitTexture = materials.find((material) => material.name === 'TextureUnlit')
  assert.equal(unlitTexture?.isMeshBasicMaterial, true)
  assert.equal(unlitTexture.map?.name, 'DiffuseTransmissionTexture.png')
  assert.deepEqual(pngDimensions(unlitTexture.map.image), [64, 64])

  const importedLight = findFirst(gltf.scene, (object) => object.isDirectionalLight === true)
  assert.equal(importedLight?.name, 'DirectLight')
  assert.equal(importedLight.intensity, 1)

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.35))
  const camera = frameSceneCamera(gltf.scene, { distance: 2.1, yOffset: 0.3 })
  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 128,
    height: 128,
    format: 'rgba',
    background: [1, 1, 1],
  })
  assert.ok(nonBackgroundRatio(rgba, [255, 255, 255], 3) > 0.15, 'DiffuseTransmissionTest should render visible test-grid pixels')
  const mean = meanRgba(rgba)
  assert.ok(mean.g > mean.r - 20, `diffuse-transmission grid should include green/teal material output (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets MandarinOrange fixture loads real diffuse-transmission texture set', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_MANDARIN_ORANGE, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_diffuse_transmission'])
  assert.deepEqual(source.buffers, [{ byteLength: 1262448, uri: 'MandarinOrange.bin' }])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'MandarinOrange_Basecolor.jpg',
    'MandarinOrange_DiffuseTransmission.png',
    'MandarinOrange_Normal.png',
    'MandarinOrange_OcclusionRough.jpg',
  ])
  assert.deepEqual(source.materials[0].extensions.KHR_materials_diffuse_transmission, {
    diffuseTransmissionFactor: 1,
    diffuseTransmissionColorTexture: { index: 1 },
    diffuseTransmissionTexture: { index: 1 },
  })

  const gltf = await loadGltfFixture(SAMPLE_ASSET_MANDARIN_ORANGE)
  const mesh = gltf.scene.getObjectByName('MandarinOrange')
  assert.ok(mesh?.isMesh, 'MandarinOrange should load its fruit mesh')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 24138)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 24138)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 24138)
  assert.equal(mesh.geometry.index?.count, 122508)
  const material = mesh.material
  assert.equal(material.name, 'MandarinOrange')
  assert.equal(material.isMeshStandardMaterial, true)
  assert.equal(material.map?.name, 'MandarinOrange_Basecolor')
  assert.equal(Buffer.isBuffer(material.map.image), true, 'MandarinOrange JPEG base color should load as an encoded Buffer')
  assert.equal(material.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(material.normalMap?.name, 'MandarinOrange_Normal')
  assert.deepEqual(pngDimensions(material.normalMap.image), [1024, 1024])
  assert.equal(material.aoMap?.name, 'MandarinOrange_OcclusionRough')
  assert.equal(material.roughnessMap, material.aoMap)
  assert.equal(material.metalnessMap, material.aoMap)
  assert.deepEqual(material.userData.gltfExtensions.KHR_materials_diffuse_transmission, {
    diffuseTransmissionFactor: 1,
    diffuseTransmissionColorTexture: { index: 1 },
    diffuseTransmissionTexture: { index: 1 },
  })

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.7))
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
  assert.ok(nonBackgroundRatio(rgba, [255, 255, 255], 3) > 0.15, 'MandarinOrange should render visible textured fruit pixels')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > mean.b + 15 && mean.g > mean.b + 8, `MandarinOrange should render warm orange pixels (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets USDShaderBallForGltf fixture loads transmission volume thickness maps', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_USD_SHADER_BALL_FOR_GLTF, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_transmission', 'KHR_materials_volume'])
  assert.deepEqual(source.buffers, [{ uri: 'USDShaderBallForGltf.bin', byteLength: 1035192 }])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'core_ao.png',
    'material_ao.png',
    'material_thick.png',
    'sss_bars.png',
  ])
  assert.deepEqual(source.materials[1].extensions, {
    KHR_materials_transmission: { transmissionFactor: 1 },
    KHR_materials_volume: {
      attenuationColor: [0.9734452903978066, 0.9911020971136257, 0.982250550332711],
      attenuationDistance: 0.01,
      thicknessFactor: 8.9,
      thicknessTexture: { index: 2, texCoord: 0 },
    },
  })

  const gltf = await loadGltfFixture(SAMPLE_ASSET_USD_SHADER_BALL_FOR_GLTF)
  const surface = gltf.scene.getObjectByName('material_surface')
  assert.ok(surface?.isMesh, 'USDShaderBallForGltf should load the material surface mesh')
  assert.equal(surface.geometry.getAttribute('position')?.count, 12069)
  assert.equal(surface.geometry.index?.count, 69276)
  const material = surface.material
  assert.equal(material.name, 'material_surface')
  assert.equal(material.isMeshPhysicalMaterial, true)
  assert.equal(material.transmission, 1)
  assert.equal(material.thickness, 8.9)
  assert.equal(material.ior, 1.5)
  assert.equal(material.aoMap?.name, 'material_ao.png')
  assert.deepEqual(pngDimensions(material.aoMap.image), [512, 512])
  assert.equal(material.thicknessMap?.name, 'material_thick.png')
  assert.deepEqual(pngDimensions(material.thicknessMap.image), [512, 512])
  const bars = gltf.scene.getObjectByName('sss_bars')
  assert.equal(bars?.material?.map?.name, 'sss_bars.png')
  assert.deepEqual(pngDimensions(bars.material.map.image), [64, 512])

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.7))
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
  assert.ok(nonBackgroundRatio(rgba, [255, 255, 255], 3) > 0.25, 'USDShaderBallForGltf should render visible shader-ball geometry')
  const mean = meanRgba(rgba)
  assert.ok(mean.r < 230 && mean.g < 235 && mean.b < 230, `USD shader ball should render non-white geometry (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets ChronographWatch fixture loads variants, watch animation, and glass transmission', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_CHRONOGRAPH_WATCH, 'utf8'))
  assert.deepEqual(source.extensionsUsed, [
    'KHR_materials_transmission',
    'KHR_materials_variants',
    'KHR_texture_transform',
  ])
  assert.deepEqual(source.buffers, [{ byteLength: 4114564, uri: 'ChronographWatch.data.bin' }])
  assert.deepEqual(source.extensions.KHR_materials_variants.variants.map((variant) => variant.name), [
    'Surgical White',
    'Midnight Gold',
    'Commerce Green',
    'Khronos Red',
  ])
  assert.equal(source.animations[0].name, 'Anim_0')
  assert.equal(source.animations[0].channels[0].target.path, 'rotation')
  assert.equal(source.meshes.filter((mesh) => mesh.primitives.some((primitive) => primitive.extensions?.KHR_materials_variants)).length, 7)

  const gltf = await loadGltfFixture(SAMPLE_ASSET_CHRONOGRAPH_WATCH)
  assert.equal(gltf.parser.json.materials.length, 29)
  assert.equal(gltf.animations.length, 1)
  assert.equal(gltf.animations[0].duration, 60)
  assert.deepEqual(gltf.animations[0].tracks.map((track) => track.name), ['Hand_Seconds.quaternion'])
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 19)

  const band = gltf.scene.getObjectByName('Band_Carbon_Fiber')
  assert.ok(band?.isMesh, 'ChronographWatch should load the carbon-fiber band')
  assert.equal(band.geometry.getAttribute('position')?.count, 6113)
  assert.equal(band.userData.gltfExtensions.KHR_materials_variants.mappings.length, 4)
  assert.equal(band.material.name, 'Band Carbon Fiber Gold')
  assert.equal(band.material.normalMap?.name, 'carbonfiber_normal.png')
  assert.deepEqual(pngDimensions(band.material.normalMap.image), [256, 256])

  const glass = gltf.scene.getObjectByName('Glass_Face')
  assert.ok(glass?.isMesh, 'ChronographWatch should load a transmissive glass face')
  assert.equal(glass.material.name, 'Glass Face')
  assert.equal(glass.material.isMeshPhysicalMaterial, true)
  assert.equal(glass.material.transmission, 1)
  assert.equal(glass.geometry.index?.count, 186)
  const watchFace = gltf.scene.getObjectByName('Watch_Face')
  assert.equal(watchFace?.material?.map?.name, 'watchface_basecolor.png')
  assert.deepEqual(pngDimensions(watchFace.material.map.image), [2048, 2048])

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.55))
  const light = new THREE.DirectionalLight(0xffffff, 1.7)
  light.position.set(2, 4, 5)
  gltf.scene.add(light)
  const camera = frameSceneCamera(gltf.scene, { distance: 2.0, yOffset: 0.25 })
  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 128,
    height: 128,
    format: 'rgba',
    background: [1, 1, 1],
  })
  assert.ok(nonBackgroundRatio(rgba, [255, 255, 255], 3) > 0.3, 'ChronographWatch should render visible watch geometry')
  const mean = meanRgba(rgba)
  assert.ok(mean.r < 230 && mean.g < 230 && mean.b < 225, `ChronographWatch should render textured metal and face pixels (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets SheenWoodLeatherSofa fixture loads required WebP sheen/specular materials', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_SHEEN_WOOD_LEATHER_SOFA, 'utf8'))
  assert.deepEqual(source.extensionsUsed, [
    'KHR_materials_specular',
    'KHR_materials_sheen',
    'KHR_texture_transform',
    'EXT_texture_webp',
  ])
  assert.deepEqual(source.extensionsRequired, ['KHR_texture_transform', 'EXT_texture_webp'])
  assert.deepEqual(source.buffers, [{ byteLength: 6533376, uri: 'SheenWoodLeatherSofa.bin' }])
  assert.equal(source.images.length, 13)
  assert.equal(source.images.every((image) => image.mimeType === 'image/webp'), true)
  assert.equal(source.textures.every((texture) => texture.extensions?.EXT_texture_webp), true)
  assert.equal(source.materials.filter((material) => material.extensions?.KHR_materials_sheen).length, 5)
  assert.equal(source.materials.filter((material) => material.extensions?.KHR_materials_specular).length, 5)

  const gltf = await loadGltfFixture(SAMPLE_ASSET_SHEEN_WOOD_LEATHER_SOFA)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'Fringe',
    'Frame',
    'Frame_Fabric',
    'Paisley',
    'Stripes',
    'Brown',
  ])

  const fringe = gltf.scene.getObjectByName('Fringe')
  assert.equal(fringe.geometry.getAttribute('position')?.count, 53760)
  assert.equal(fringe.geometry.getAttribute('color')?.count, 53760)
  assert.equal(fringe.material.name, 'Fringe')
  assert.equal(fringe.material.isMeshPhysicalMaterial, true)
  assert.equal(fringe.material.transparent, true)
  assert.equal(fringe.material.sheen, 1)
  assert.equal(fringe.material.sheenRoughness, 0.4)
  assert.equal(fringe.material.specularIntensity, 0.3)
  assert.equal(fringe.material.map?.name, 'Fringe_BaseColor.webp')
  assertWebpBuffer(fringe.material.map.image, 'Fringe base color')
  assert.equal(fringe.material.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(fringe.material.normalMap?.name, 'Fringe_Normal.webp')
  assertWebpBuffer(fringe.material.normalMap.image, 'Fringe normal')

  const paisley = gltf.scene.getObjectByName('Paisley')
  assert.equal(paisley.material.map?.name, 'Paisley_BaseColor.webp')
  assert.equal(paisley.material.normalMap?.name, 'Paisley_Normal.webp')
  assert.equal(paisley.material.aoMap?.name, 'Cushions_Occlusion.webp')
  assert.equal(paisley.material.roughnessMap, paisley.material.aoMap)
  assertWebpBuffer(paisley.material.map.image, 'Paisley base color')

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.55))
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
  assert.ok(nonBackgroundRatio(rgba, [255, 255, 255], 3) > 0.18, 'SheenWoodLeatherSofa should render visible WebP-textured sofa pixels')
  const mean = meanRgba(rgba)
  assert.ok(mean.r < 235 && mean.g < 235 && mean.b < 235, `SheenWoodLeatherSofa should render non-white textured material output (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets DiffuseTransmissionPlant fixture loads animated fireflies and diffuse-transmission leaves', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_DIFFUSE_TRANSMISSION_PLANT, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_lights_punctual', 'KHR_materials_diffuse_transmission'])
  assert.deepEqual(source.buffers, [{ uri: 'DiffuseTransmissionPlant.bin', byteLength: 2156988 }])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'img0.png',
    'img1.jpg',
    'img2.jpg',
    'img3.jpg',
    'img4.jpg',
    'img5.jpg',
    'img6.jpg',
  ])
  assert.equal(source.extensions.KHR_lights_punctual.lights.length, 2)
  assert.equal(source.extensions.KHR_lights_punctual.lights.every((light) => light.type === 'point'), true)
  assert.deepEqual(source.materials[1].extensions.KHR_materials_diffuse_transmission, {
    diffuseTransmissionColorFactor: [1, 1, 1],
    diffuseTransmissionFactor: 0.1,
    diffuseTransmissionColorTexture: { index: 5 },
  })

  const gltf = await loadGltfFixture(SAMPLE_ASSET_DIFFUSE_TRANSMISSION_PLANT)
  assert.equal(gltf.animations.length, 1)
  assert.equal(gltf.animations[0].tracks.length, 30)
  assert.ok(gltf.animations[0].tracks.some((track) => track.name === 'pointlight_firefly1.position'))
  const lights = []
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isPointLight === true) lights.push(object)
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(lights.map((light) => light.name), ['pointlight_firefly1', 'pointlight_firefly2'])
  assert.equal(meshes.length, 9)

  const leaves = gltf.scene.getObjectByName('leaves')
  assert.ok(leaves?.isMesh, 'DiffuseTransmissionPlant should load its leaves mesh')
  assert.equal(leaves.geometry.getAttribute('position')?.count, 7077)
  assert.equal(leaves.geometry.index?.count, 31941)
  assert.equal(leaves.material.name, 'leaves')
  assert.equal(leaves.material.alphaTest, 0.5)
  assert.deepEqual(leaves.material.userData.gltfExtensions.KHR_materials_diffuse_transmission, {
    diffuseTransmissionColorFactor: [1, 1, 1],
    diffuseTransmissionFactor: 0.1,
    diffuseTransmissionColorTexture: { index: 5 },
  })
  assert.equal(leaves.material.map?.name, 'img0.png')
  assert.deepEqual(pngDimensions(leaves.material.map.image), [1024, 1024])
  assert.equal(leaves.material.normalMap?.name, 'img1.jpg')
  assert.equal(Buffer.isBuffer(leaves.material.normalMap.image), true, 'leaf normal JPEG should load as an encoded Buffer')

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.45))
  const light = new THREE.DirectionalLight(0xffffff, 1.6)
  light.position.set(2, 4, 5)
  gltf.scene.add(light)
  const camera = frameSceneCamera(gltf.scene, { distance: 2.2, yOffset: 0.4 })
  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 128,
    height: 128,
    format: 'rgba',
    background: [1, 1, 1],
  })
  assert.ok(nonBackgroundRatio(rgba, [255, 255, 255], 3) > 0.2, 'DiffuseTransmissionPlant should render visible plant and firefly geometry')
  const mean = meanRgba(rgba)
  assert.ok(mean.r < 240 && mean.g < 240 && mean.b < 235, `DiffuseTransmissionPlant should render non-white textured output (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets DiffuseTransmissionTeacup fixture loads diffuse-transmission ORM texture reuse', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_DIFFUSE_TRANSMISSION_TEACUP, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_diffuse_transmission'])
  assert.deepEqual(source.buffers, [{ byteLength: 3043872, uri: 'DiffuseTransmissionTeacup.bin' }])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'teasaucer_basecolor.jpg',
    'teasaucer_normal.png',
    'teasaucer_ormt.png',
    'teacup_basecolor.jpg',
    'teacup_normal.png',
    'teacup_ormt.png',
  ])
  assert.equal(source.materials.every((material) => material.extensions?.KHR_materials_diffuse_transmission?.diffuseTransmissionFactor === 1), true)

  const gltf = await loadGltfFixture(SAMPLE_ASSET_DIFFUSE_TRANSMISSION_TEACUP)
  const cup = gltf.scene.getObjectByName('tea_cup')
  const saucer = gltf.scene.getObjectByName('tea_saucer')
  assert.ok(cup?.isMesh, 'DiffuseTransmissionTeacup should load its cup mesh')
  assert.ok(saucer?.isMesh, 'DiffuseTransmissionTeacup should load its saucer mesh')
  assert.equal(cup.geometry.getAttribute('position')?.count, 49944)
  assert.equal(cup.geometry.index?.count, 49944)
  assert.equal(saucer.geometry.getAttribute('position')?.count, 34608)
  assert.equal(saucer.geometry.index?.count, 34608)

  assert.deepEqual(cup.material.userData.gltfExtensions.KHR_materials_diffuse_transmission, {
    diffuseTransmissionFactor: 1,
    diffuseTransmissionTexture: { index: 5 },
    diffuseTransmissionColorFactor: [0.84, 0.8, 0.74],
  })
  assert.equal(cup.material.map?.name, 'teacup_basecolor.jpg')
  assert.equal(Buffer.isBuffer(cup.material.map.image), true, 'teacup base color JPEG should load as an encoded Buffer')
  assert.equal(cup.material.normalMap?.name, 'teacup_normal.png')
  assert.deepEqual(pngDimensions(cup.material.normalMap.image), [1024, 1024])
  assert.equal(cup.material.aoMap?.name, 'teacup_ormt.png')
  assert.equal(cup.material.roughnessMap, cup.material.aoMap)
  assert.equal(cup.material.metalnessMap, cup.material.aoMap)
  assert.deepEqual(pngDimensions(cup.material.aoMap.image), [1024, 1024])

  assert.deepEqual(saucer.material.userData.gltfExtensions.KHR_materials_diffuse_transmission, {
    diffuseTransmissionFactor: 1,
    diffuseTransmissionTexture: { index: 2 },
    diffuseTransmissionColorFactor: [0.84, 0.8, 0.74],
  })
  assert.equal(saucer.material.normalMap?.name, 'teasaucer_normal.png')
  assert.deepEqual(pngDimensions(saucer.material.normalMap.image), [1024, 512])
  assert.equal(saucer.material.aoMap?.name, 'teasaucer_ormt.png')
  assert.deepEqual(pngDimensions(saucer.material.aoMap.image), [1024, 512])

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.65))
  const light = new THREE.DirectionalLight(0xffffff, 1.7)
  light.position.set(2, 4, 5)
  gltf.scene.add(light)
  const camera = frameSceneCamera(gltf.scene)
  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 128,
    height: 128,
    format: 'rgba',
    background: [1, 1, 1],
  })
  assert.ok(nonBackgroundRatio(rgba, [255, 255, 255], 3) > 0.18, 'DiffuseTransmissionTeacup should render visible ceramic geometry')
  const mean = meanRgba(rgba)
  assert.ok(mean.r < 245 && mean.g < 245 && mean.b < 242, `DiffuseTransmissionTeacup should render textured non-white output (${mean.r}, ${mean.g}, ${mean.b})`)
})
