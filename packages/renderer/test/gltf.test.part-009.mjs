import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'
import { Renderer, SAMPLE_ASSET_AVOCADO, SAMPLE_ASSET_BARRAMUNDI_FISH, SAMPLE_ASSET_BOOM_BOX, SAMPLE_ASSET_BOOM_BOX_WITH_AXES, SAMPLE_ASSET_BOX_INTERLEAVED, SAMPLE_ASSET_COMPARE_NORMAL, SAMPLE_ASSET_DRAGON_ATTENUATION } from './gltf.test.part-001.mjs'
import { assertVectorClose, findFirst, loadGltfFixture, meanRegion, pngDimensions, vectorFromAttribute } from './gltf.test.part-028.mjs'
test('committed Khronos glTF Sample Assets DragonAttenuation fixture loads attenuation variants and thickness maps', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_DRAGON_ATTENUATION, 'utf8'))
  assert.deepEqual(source.extensionsUsed, [
    'KHR_materials_transmission',
    'KHR_materials_volume',
    'KHR_materials_variants',
  ])
  assert.deepEqual(source.extensions?.KHR_materials_variants?.variants?.map((variant) => variant.name), [
    'Attenuation',
    'Surface Color',
  ])
  assert.deepEqual(source.buffers, [
    { byteLength: 5817396, uri: 'DragonAttenuation.bin' },
  ])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'checkerboard.png',
    'Dragon_ThicknessMap.jpg',
  ])
  assert.deepEqual(source.materials.map((material) => material.name), [
    'Cloth Backdrop',
    'Dragon with Attenuation',
    'Dragon with Surface Coloring Only',
  ])
  assert.deepEqual(source.materials[1].extensions, {
    KHR_materials_transmission: { transmissionFactor: 1 },
    KHR_materials_volume: {
      attenuationColor: [0.921, 0.64, 0.064],
      attenuationDistance: 0.155,
      thicknessFactor: 2.27,
      thicknessTexture: { index: 1, texCoord: 0 },
    },
  })
  assert.deepEqual(source.materials[2].extensions, {
    KHR_materials_transmission: { transmissionFactor: 1 },
    KHR_materials_volume: {
      thicknessFactor: 2.27,
      thicknessTexture: { index: 1, texCoord: 0 },
    },
  })
  assert.deepEqual(source.meshes[1].primitives[0].extensions?.KHR_materials_variants?.mappings, [
    { material: 1, variants: [0] },
    { material: 2, variants: [1] },
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_DRAGON_ATTENUATION)
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_volume'))
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'Cloth_Backdrop',
    'Dragon',
  ])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [62570, 76809])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [62570, 76809])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [62570, 76809])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [131337, 273648])

  const dragonMesh = meshes.find((mesh) => mesh.name === 'Dragon')
  assert.deepEqual(dragonMesh.userData.gltfExtensions?.KHR_materials_variants?.mappings, [
    { material: 1, variants: [0] },
    { material: 2, variants: [1] },
  ])

  const materials = new Map(meshes.map((mesh) => [mesh.material.name, mesh.material]))
  const backdrop = materials.get('Cloth Backdrop')
  assert.equal(backdrop?.isMeshStandardMaterial, true)
  assert.equal(backdrop.map.name, 'checkerboard.png')
  assert.equal(Buffer.isBuffer(backdrop.map.image), true, 'DragonAttenuation checker PNG should load as an encoded Buffer')
  assert.deepEqual(pngDimensions(backdrop.map.image), [2048, 2048])
  assert.equal(backdrop.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(backdrop.map.flipY, false)

  const dragon = materials.get('Dragon with Attenuation')
  assert.equal(dragon?.isMeshPhysicalMaterial, true)
  assert.equal(dragon.metalness, 0)
  assert.equal(dragon.roughness, 0)
  assert.equal(dragon.transmission, 1)
  assert.equal(dragon.thickness, 2.27)
  assert.equal(dragon.attenuationDistance, 0.155)
  assert.deepEqual(dragon.attenuationColor.toArray(), [0.921, 0.64, 0.064])
  assert.equal(dragon.thicknessMap.name, 'Dragon_ThicknessMap.jpg')
  assert.equal(Buffer.isBuffer(dragon.thicknessMap.image), true, 'DragonAttenuation thickness JPEG should load as an encoded Buffer')
  assert.equal(dragon.thicknessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(dragon.thicknessMap.flipY, false)

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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.2, 'DragonAttenuation should render visible attenuation dragon and checker backdrop')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 15 && mean.g > 15 && mean.b > 15, `DragonAttenuation should render lit dragon pixels (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets CompareNormal fixture loads normal-map comparison variants', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_COMPARE_NORMAL, 'utf8'))
  assert.equal(source.buffers[0].uri, 'CompareNormal.bin')
  assert.deepEqual(source.images.map((image) => image.uri), ['Compare_Normal_img0.jpg'])
  assert.deepEqual(source.materials.map((material) => [
    material.name,
    material.normalTexture?.index ?? null,
    material.pbrMetallicRoughness?.baseColorFactor,
  ]), [
    ['Wicker no Normal', null, [0.501960813999176, 0.4392157196998596, 0.3529411852359772, 1]],
    ['Wicker with Normal', 0, [0.501960813999176, 0.4431372880935669, 0.3529411852359772, 1]],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_COMPARE_NORMAL)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), ['Sphere001', 'Sphere002'])
  assert.deepEqual(meshes.map((mesh) => mesh.material.name), ['Wicker no Normal', 'Wicker with Normal'])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [1538, 1728])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [1538, 1728])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count ?? null), [null, 1728])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [9216, 9216])

  const [flat, normalMapped] = meshes
  assert.equal(flat.material.normalMap ?? null, null)
  assertVectorClose(flat.material.color.toArray(), [
    0.501960813999176,
    0.4392157196998596,
    0.3529411852359772,
  ], 'CompareNormal no-normal baseColorFactor')
  assertVectorClose(normalMapped.material.color.toArray(), [
    0.501960813999176,
    0.4431372880935669,
    0.3529411852359772,
  ], 'CompareNormal normal-mapped baseColorFactor')

  assert.equal(normalMapped.material.metalness, 0)
  assert.equal(normalMapped.material.roughness, 0.25)
  assert.equal(Buffer.isBuffer(normalMapped.material.normalMap?.image), true, 'normal-map JPEG should load as an encoded Buffer')
  assert.ok(normalMapped.material.normalMap.image.length > 0, 'normal-map JPEG buffer should not be empty')
  assert.equal(normalMapped.material.normalMap.name, 'Compare_Normal_img0.jpg')
  assert.equal(normalMapped.material.normalMap.colorSpace, THREE.NoColorSpace)
  assert.equal(normalMapped.material.normalMap.flipY, false)
  assertVectorClose(normalMapped.material.normalScale.toArray(), [1, -1], 'glTF normal map should use Three.js Y-flipped normal scale')

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.55))
  const light = new THREE.DirectionalLight(0xffffff, 1.8)
  light.position.set(2, 3, 4)
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
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.18, 'CompareNormal should render visible normal-map comparison spheres')
})

test('committed Khronos glTF Sample Assets Avocado fixture loads PBR texture maps', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_AVOCADO, 'utf8'))
  assert.equal(source.buffers[0].uri, 'Avocado.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'Avocado_baseColor.png',
    'Avocado_roughnessMetallic.png',
    'Avocado_normal.png',
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_AVOCADO)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'Khronos Avocado sample should load a mesh')
  assert.equal(mesh.name, 'Avocado')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 406)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 406)
  assert.equal(mesh.geometry.getAttribute('tangent')?.count, 406)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 406)
  assert.equal(mesh.geometry.index?.count, 2046)
  assert.equal(mesh.material.name, '2256_Avocado_d')

  const { map, roughnessMap, metalnessMap, normalMap } = mesh.material
  assert.ok(map?.isTexture, 'Avocado sample should load a base color texture')
  assert.ok(roughnessMap?.isTexture, 'Avocado sample should load a roughness texture')
  assert.ok(metalnessMap?.isTexture, 'Avocado sample should load a metalness texture')
  assert.ok(normalMap?.isTexture, 'Avocado sample should load a normal texture')
  assert.equal(roughnessMap, metalnessMap, 'Avocado metallic/roughness channels should share the packed texture')
  assert.deepEqual(pngDimensions(map.image), [2048, 2048])
  assert.deepEqual(pngDimensions(roughnessMap.image), [2048, 2048])
  assert.deepEqual(pngDimensions(normalMap.image), [2048, 2048])
  assert.equal(map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(roughnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(normalMap.colorSpace, THREE.NoColorSpace)
  assert.equal(map.flipY, false)
  assert.equal(roughnessMap.flipY, false)
  assert.equal(normalMap.flipY, false)

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 1.6)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const halfExtent = Math.max(size.x, size.y, size.z) * 0.75
  const camera = new THREE.OrthographicCamera(-halfExtent, halfExtent, halfExtent, -halfExtent, 0.001, 10)
  camera.position.set(center.x, center.y + 0.04, center.z + 0.14)
  camera.lookAt(center)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 128,
    height: 128,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.15, 'Khronos Avocado sample should render visible PBR textured pixels')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > mean.b + 10 && mean.g > mean.b + 10, `Avocado texture should contribute green/yellow output (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets BarramundiFish fixture loads organic mesh packed PBR maps', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_BARRAMUNDI_FISH, 'utf8'))
  assert.deepEqual(source.buffers, [{ uri: 'BarramundiFish.bin', byteLength: 128208 }])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'BarramundiFish_baseColor.png',
    'BarramundiFish_occlusionRoughnessMetallic.png',
    'BarramundiFish_normal.png',
  ])
  assert.equal(source.meshes[0].name, 'barramundi_fish_Hero')
  assert.equal(source.materials[0].name, '7288_barramundi fish_col')
  assert.equal(source.materials[0].normalTexture.index, 2)
  assert.equal(source.materials[0].occlusionTexture.index, 1)
  assert.equal(source.materials[0].pbrMetallicRoughness.baseColorTexture.index, 0)
  assert.equal(source.materials[0].pbrMetallicRoughness.metallicRoughnessTexture.index, 1)

  const gltf = await loadGltfFixture(SAMPLE_ASSET_BARRAMUNDI_FISH)
  const mesh = gltf.scene.getObjectByName('BarramundiFish')
  assert.ok(mesh?.isMesh, 'BarramundiFish sample should load a named fish mesh')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 2188)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 2188)
  assert.equal(mesh.geometry.getAttribute('tangent')?.count, 2188)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 2188)
  assert.equal(mesh.geometry.index?.count, 11592)
  assertVectorClose(mesh.quaternion.toArray(), [0, 1, 0, 0], 'BarramundiFish node rotation')
  assert.equal(mesh.material.name, '7288_barramundi fish_col')

  const { map, aoMap, roughnessMap, metalnessMap, normalMap } = mesh.material
  assert.ok(map?.isTexture, 'BarramundiFish sample should load a base color texture')
  assert.ok(aoMap?.isTexture, 'BarramundiFish sample should load an occlusion texture')
  assert.ok(roughnessMap?.isTexture, 'BarramundiFish sample should load a roughness texture')
  assert.ok(metalnessMap?.isTexture, 'BarramundiFish sample should load a metalness texture')
  assert.ok(normalMap?.isTexture, 'BarramundiFish sample should load a normal texture')
  assert.equal(aoMap, roughnessMap, 'BarramundiFish occlusion/roughness channels should share the packed texture')
  assert.equal(roughnessMap, metalnessMap, 'BarramundiFish metallic/roughness channels should share the packed texture')
  assert.equal(map.name, 'BarramundiFish_baseColor.png')
  assert.equal(aoMap.name, 'BarramundiFish_occlusionRoughnessMetallic.png')
  assert.equal(normalMap.name, 'BarramundiFish_normal.png')
  assert.deepEqual(pngDimensions(map.image), [2048, 2048])
  assert.deepEqual(pngDimensions(aoMap.image), [2048, 2048])
  assert.deepEqual(pngDimensions(normalMap.image), [2048, 2048])
  assert.equal(map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(aoMap.colorSpace, THREE.NoColorSpace)
  assert.equal(normalMap.colorSpace, THREE.NoColorSpace)

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.9))
  const light = new THREE.DirectionalLight(0xffffff, 2)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)
  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const halfHeight = Math.max(size.y, size.z) / 2 + 0.04
  const halfWidth = halfHeight * 1.5
  const camera = new THREE.OrthographicCamera(-halfWidth, halfWidth, halfHeight, -halfHeight, 0.01, 20)
  camera.position.set(center.x + 3, center.y, center.z)
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
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.07, 'BarramundiFish should render visible packed-PBR textured organic geometry')
})

test('committed Khronos glTF Sample Assets BoomBox fixture loads emissive and packed ORM maps', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_BOOM_BOX, 'utf8'))
  assert.equal(source.buffers[0].uri, 'BoomBox.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'BoomBox_baseColor.png',
    'BoomBox_occlusionRoughnessMetallic.png',
    'BoomBox_normal.png',
    'BoomBox_emissive.png',
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_BOOM_BOX)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'Khronos BoomBox sample should load a mesh')
  assert.equal(mesh.name, 'BoomBox')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 3575)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 3575)
  assert.equal(mesh.geometry.getAttribute('tangent')?.count, 3575)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 3575)
  assert.equal(mesh.geometry.index?.count, 18108)
  assert.equal(mesh.material.name, 'BoomBox_Mat')
  assert.deepEqual(mesh.material.emissive.toArray(), [1, 1, 1])

  const { map, aoMap, roughnessMap, metalnessMap, normalMap, emissiveMap } = mesh.material
  assert.ok(map?.isTexture, 'BoomBox sample should load a base color texture')
  assert.ok(aoMap?.isTexture, 'BoomBox sample should load an occlusion texture')
  assert.ok(roughnessMap?.isTexture, 'BoomBox sample should load a roughness texture')
  assert.ok(metalnessMap?.isTexture, 'BoomBox sample should load a metalness texture')
  assert.ok(normalMap?.isTexture, 'BoomBox sample should load a normal texture')
  assert.ok(emissiveMap?.isTexture, 'BoomBox sample should load an emissive texture')
  assert.equal(aoMap, roughnessMap, 'BoomBox occlusion/roughness channels should share the packed texture')
  assert.equal(roughnessMap, metalnessMap, 'BoomBox metallic/roughness channels should share the packed texture')
  assert.deepEqual(pngDimensions(map.image), [2048, 2048])
  assert.deepEqual(pngDimensions(aoMap.image), [2048, 2048])
  assert.deepEqual(pngDimensions(normalMap.image), [2048, 2048])
  assert.deepEqual(pngDimensions(emissiveMap.image), [2048, 2048])
  assert.equal(map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(aoMap.colorSpace, THREE.NoColorSpace)
  assert.equal(normalMap.colorSpace, THREE.NoColorSpace)
  assert.equal(emissiveMap.colorSpace, THREE.SRGBColorSpace)
  assert.equal(map.flipY, false)
  assert.equal(aoMap.flipY, false)
  assert.equal(normalMap.flipY, false)
  assert.equal(emissiveMap.flipY, false)

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 1.4)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const halfExtent = Math.max(size.x, size.y, size.z) * 0.72
  const camera = new THREE.OrthographicCamera(-halfExtent, halfExtent, halfExtent, -halfExtent, 0.001, 10)
  camera.position.set(center.x, center.y + 0.012, center.z + 0.05)
  camera.lookAt(center)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 128,
    height: 128,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.12, 'Khronos BoomBox sample should render visible textured pixels')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 8 && mean.g > 8 && mean.b > 8, `BoomBox textures should contribute non-black output (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets BoomBoxWithAxes fixture loads coordinate-system meshes and shared materials', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_BOOM_BOX_WITH_AXES, 'utf8'))
  assert.equal(source.buffers[0].uri, 'BoomBoxWithAxes.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'BoomBoxWithAxes_baseColor.png',
    'BoomBoxWithAxes_roughnessMetallic.png',
    'BoomBoxWithAxes_normal.png',
    'BoomBoxWithAxes_emissive.png',
    'BoomBoxWithAxes_baseColor1.png',
  ])
  assert.deepEqual(source.nodes[5].children, [0, 1, 2, 3, 4])
  assert.deepEqual(source.nodes[5].rotation, [0, 1, 0, 0])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_BOOM_BOX_WITH_AXES)
  const root = gltf.scene.getObjectByName('BoomBox_Coordinates')
  assert.deepEqual(root.children.map((child) => child.name), ['BoomBox', 'CoordinateSystem', 'X_axis', 'Y_axis', 'Z_axis'])
  assertVectorClose(root.quaternion.toArray(), [0, 1, 0, 0], 'BoomBoxWithAxes root rotation')

  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), ['BoomBox', 'CoordinateSystem', 'X_axis', 'Y_axis', 'Z_axis'])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [3575, 875, 2252, 1820, 1708])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [3575, 875, 2252, 1820, 1708])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('tangent')?.count), [3575, 875, 2252, 1820, 1708])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [3575, 875, 2252, 1820, 1708])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [18108, 3420, 11064, 8976, 8496])
  assert.ok(meshes.slice(2).every((mesh) => Math.abs(mesh.scale.x - 0.06) < 1e-12), 'axis meshes should retain imported scale transforms')

  const [boombox, coordinateSystem, xAxis, yAxis, zAxis] = meshes
  assert.equal(boombox.material.name, 'M_BoomBox')
  assert.ok([coordinateSystem, xAxis, yAxis, zAxis].every((mesh) => mesh.material.name === 'M_Coordinates'))
  const boomboxMaterial = boombox.material
  assert.equal(boomboxMaterial.map.name, 'BoomBoxWithAxes_baseColor.png')
  assert.deepEqual(pngDimensions(boomboxMaterial.map.image), [2048, 2048])
  assert.equal(boomboxMaterial.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(boomboxMaterial.roughnessMap, boomboxMaterial.metalnessMap)
  assert.equal(boomboxMaterial.roughnessMap.name, 'BoomBoxWithAxes_roughnessMetallic.png')
  assert.deepEqual(pngDimensions(boomboxMaterial.roughnessMap.image), [2048, 2048])
  assert.equal(boomboxMaterial.normalMap.name, 'BoomBoxWithAxes_normal.png')
  assert.deepEqual(pngDimensions(boomboxMaterial.normalMap.image), [2048, 2048])
  const coordinateMaterial = coordinateSystem.material
  assert.equal(coordinateMaterial.map.name, 'BoomBoxWithAxes_baseColor1.png')
  assert.deepEqual(pngDimensions(coordinateMaterial.map.image), [32, 32])
  assert.equal(coordinateMaterial.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(coordinateMaterial.metalness, 0)
  assert.equal(coordinateMaterial.roughness, 0.735)

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 1.6)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)
  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const halfExtent = Math.max(size.x, size.y, size.z) / 2 + 0.03
  const camera = new THREE.OrthographicCamera(-halfExtent, halfExtent, halfExtent, -halfExtent, 0.01, 20)
  camera.position.set(center.x + 0.3, center.y + 0.5, center.z + 0.9)
  camera.lookAt(center)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 128,
    height: 128,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.018, 'BoomBoxWithAxes should render visible boombox and coordinate-system geometry')
})

test('committed Khronos glTF Sample Assets BoxInterleaved fixture loads byteStride attributes', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_BOX_INTERLEAVED)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'Khronos BoxInterleaved sample should load a mesh')
  const position = mesh.geometry.getAttribute('position')
  const normal = mesh.geometry.getAttribute('normal')
  assert.equal(position?.count, 24)
  assert.equal(normal?.count, 24)
  assert.equal(mesh.geometry.index?.count, 36)
  assert.equal(position.isInterleavedBufferAttribute, true)
  assert.equal(normal.isInterleavedBufferAttribute, true)
  assert.equal(position.data.stride, 6)
  assert.deepEqual(vectorFromAttribute(position, 0), [-0.5, -0.5, 0.5])
  assert.deepEqual(vectorFromAttribute(normal, 0), [0, 0, 1])
  assert.equal(mesh.material.color.r, 0.800000011920929)
  assert.equal(mesh.material.color.g, 0)
  assert.equal(mesh.material.color.b, 0)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 10)
  camera.position.set(1.4, 1.1, 2.2)
  camera.lookAt(0, 0, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.55))
  const light = new THREE.DirectionalLight(0xffffff, 1.6)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.2, 'Khronos BoxInterleaved sample should render visible pixels')
  const center = meanRegion(rgba, 96, 96, 40, 40, 56, 56)
  assert.ok(center.r > center.b + 150 && center.r > center.g + 180, `BoxInterleaved sample should render a red cube (${center.r}, ${center.g}, ${center.b})`)
})
