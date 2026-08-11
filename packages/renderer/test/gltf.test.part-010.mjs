import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'
import { Renderer, SAMPLE_ASSET_CAMERAS, SAMPLE_ASSET_DAMAGED_HELMET, SAMPLE_ASSET_DIRECTIONAL_LIGHT, SAMPLE_ASSET_DUCK, SAMPLE_ASSET_VIRTUAL_CITY, SAMPLE_ASSET_WATER_BOTTLE } from './gltf.test.part-001.mjs'
import { findFirst, loadGltfFixture, meanRegion, pngDimensions } from './gltf.test.part-028.mjs'
test('committed Khronos glTF Sample Assets Cameras fixture loads and renders imported cameras', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_CAMERAS)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'Khronos Cameras sample should load a mesh')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 4)
  assert.equal(mesh.geometry.index?.count, 6)
  assert.equal(gltf.cameras.length, 2)

  const [perspective, orthographic] = gltf.cameras
  assert.equal(perspective.isPerspectiveCamera, true)
  assert.equal(perspective.near, 0.01)
  assert.equal(perspective.far, 100)
  assert.equal(perspective.aspect, 1)
  assert.ok(Math.abs(perspective.fov - THREE.MathUtils.radToDeg(0.7)) < 1e-10, `perspective camera should preserve glTF yfov (${perspective.fov})`)
  assert.deepEqual(perspective.position.toArray(), [0.5, 0.5, 3])

  assert.equal(orthographic.isOrthographicCamera, true)
  assert.equal(orthographic.near, 0.01)
  assert.equal(orthographic.far, 100)
  assert.equal(orthographic.left, -1)
  assert.equal(orthographic.right, 1)
  assert.equal(orthographic.top, 1)
  assert.equal(orthographic.bottom, -1)
  assert.deepEqual(orthographic.position.toArray(), [0.5, 0.5, 3])

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1.0))
  gltf.scene.updateMatrixWorld(true)

  const renderer = new Renderer()
  const renderWithCamera = (camera) => {
    camera.updateMatrixWorld(true)
    return renderer.render(gltf.scene, camera, {
      width: 96,
      height: 96,
      format: 'rgba',
      background: [0, 0, 0],
      outputColorSpace: THREE.LinearSRGBColorSpace,
    })
  }

  const perspectiveRgba = renderWithCamera(perspective)
  const orthographicRgba = renderWithCamera(orthographic)

  assert.ok(nonBackgroundRatio(perspectiveRgba, [0, 0, 0], 3) > 0.1, 'Cameras sample should render through imported perspective camera')
  assert.ok(nonBackgroundRatio(orthographicRgba, [0, 0, 0], 3) > 0.15, 'Cameras sample should render through imported orthographic camera')
  const perspectiveCenter = meanRegion(perspectiveRgba, 96, 96, 24, 24, 72, 72)
  const orthographicCenter = meanRegion(orthographicRgba, 96, 96, 24, 24, 72, 72)
  assert.ok(perspectiveCenter.r > 80 && perspectiveCenter.g > 80 && perspectiveCenter.b > 80, `perspective camera should see the white mesh (${perspectiveCenter.r}, ${perspectiveCenter.g}, ${perspectiveCenter.b})`)
  assert.ok(orthographicCenter.r > 80 && orthographicCenter.g > 80 && orthographicCenter.b > 80, `orthographic camera should see the white mesh (${orthographicCenter.r}, ${orthographicCenter.g}, ${orthographicCenter.b})`)
})

test('committed Khronos glTF Sample Assets VirtualCity fixture loads textured multi-camera animated city scene', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_VIRTUAL_CITY, 'utf8'))
  assert.deepEqual(source.buffers, [{ byteLength: 1967226, uri: 'VC0.bin' }])
  assert.deepEqual(source.images.map((image) => image.uri), [
    '001.jpg',
    'cockpit-map.jpg',
    's_08.jpg',
    's_06.jpg',
    's_04.jpg',
    's_02.jpg',
    's_07.jpg',
    's_03.jpg',
    's_05.jpg',
    's_01.jpg',
    '002.jpg',
    '11.jpg',
    'machine.jpg',
    'prop128.png',
    'scrapsurf03-red.jpg',
    'f22.jpg',
    'heli.jpg',
    'O21.jpg',
    '5.jpg',
    'surface01.jpg',
  ])
  assert.equal(source.textures.length, 28)
  assert.equal(source.materials.length, 167)
  assert.equal(source.meshes.length, 135)
  assert.equal(source.nodes.length, 234)
  assert.equal(source.cameras.length, 14)
  assert.equal(source.animations.length, 1)
  assert.equal(source.animations[0].channels.length, 73)
  assert.equal(source.animations[0].samplers.length, 73)

  const gltf = await loadGltfFixture(SAMPLE_ASSET_VIRTUAL_CITY)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 167)
  assert.equal(gltf.cameras.length, 14)
  assert.ok(gltf.cameras.every((camera) => camera.isPerspectiveCamera === true), 'VirtualCity should load perspective camera set')
  assert.equal(meshes.reduce((sum, mesh) => sum + mesh.geometry.getAttribute('position').count, 0), 19483)
  assert.equal(meshes.reduce((sum, mesh) => sum + (mesh.geometry.index?.count ?? 0), 0), 25149)
  assert.equal(meshes.filter((mesh) => mesh.material.map?.isTexture === true).length, 147)
  assert.equal(meshes.filter((mesh) => mesh.material.aoMap?.isTexture === true).length, 147)

  const tower = gltf.scene.getObjectByName('tower00')
  assert.ok(tower?.isMesh, 'VirtualCity should load named tower mesh')
  assert.equal(tower.geometry.getAttribute('position')?.count, 321)
  assert.equal(tower.geometry.getAttribute('uv')?.count, 321)
  assert.equal(tower.geometry.index?.count, 342)
  assert.equal(tower.material.name, 'Scrape04')
  assert.equal(tower.material.map.name, '001.jpg')
  assert.equal(Buffer.isBuffer(tower.material.map.image), true, 'VirtualCity JPEG base-color texture should load as an encoded Buffer')
  assert.equal(tower.material.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(tower.material.aoMap, tower.material.map, 'VirtualCity should preserve shared base-color/AO texture references')

  const redTower = gltf.scene.getObjectByName('tower02')
  assert.ok(redTower?.isMesh, 'VirtualCity should load alternate textured tower mesh')
  assert.equal(redTower.material.name, '_22jyj')
  assert.equal(redTower.material.map.name, 'scrapsurf03-red.jpg')
  assert.equal(Buffer.isBuffer(redTower.material.map.image), true, 'VirtualCity secondary JPEG texture should load as an encoded Buffer')

  const clip = gltf.animations[0]
  assert.equal(clip.name, 'animation_0')
  assert.equal(clip.duration, 30)
  assert.equal(clip.tracks.length, 73)
  assert.equal(clip.tracks.filter((track) => track.name.endsWith('.position')).length, 41)
  assert.equal(clip.tracks.filter((track) => track.name.endsWith('.quaternion')).length, 16)
  assert.equal(clip.tracks.filter((track) => track.name.endsWith('.scale')).length, 16)
  assert.ok(clip.tracks.every((track) => track.times.length === 901), 'every VirtualCity track should contain 901 keyframes')

  const ship = gltf.scene.getObjectByName('_ship-box01')
  assert.ok(ship?.isMesh, 'VirtualCity should load animated ship node')
  const mixer = new THREE.AnimationMixer(gltf.scene)
  mixer.clipAction(clip).play()
  mixer.setTime(0)
  const startPosition = ship.position.clone()
  mixer.setTime(clip.duration / 2)
  assert.ok(startPosition.distanceTo(ship.position) > 1000, 'VirtualCity animation should move the ship node across the city')

  const camera = gltf.cameras[1]
  camera.aspect = 1.5
  camera.updateProjectionMatrix()
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.9))
  const light = new THREE.DirectionalLight(0xffffff, 1.8)
  light.position.set(100, 300, 200)
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 144,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.55, 'VirtualCity should render visible animated city geometry through an imported camera')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > mean.g && mean.g > mean.b, `VirtualCity textures should render warm city colors (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets DirectionalLight fixture loads KHR_lights_punctual and renders with imported camera', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_DIRECTIONAL_LIGHT)
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_lights_punctual'))
  assert.ok(gltf.parser?.json?.extensionsRequired?.includes('KHR_lights_punctual'))

  const meshes = []
  const lights = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
    if (object.isLight === true) lights.push(object)
  })

  assert.deepEqual(meshes.map((mesh) => mesh.name), ['m0%_r0%', 'm0%_r16%', 'm0%_r33%'])
  assert.equal(lights.length, 1)

  const light = lights[0]
  assert.equal(light.isDirectionalLight, true)
  assert.equal(light.name, 'Sun_Orientation')
  assert.deepEqual(light.color.toArray(), [0.9, 0.8, 0.1])
  assert.equal(light.intensity, 1)

  const camera = gltf.cameras[0]
  assert.equal(camera?.isPerspectiveCamera, true)
  assert.equal(camera.name, 'Generated_Camera')
  assert.equal(camera.near, 0.3)
  assert.equal(camera.far, 5)
  assert.ok(Math.abs(camera.fov - THREE.MathUtils.radToDeg(0.65)) < 1e-10, `directional-light sample should preserve imported yfov (${camera.fov})`)
  assert.deepEqual(camera.position.toArray(), [0, 0, 2])

  const roughnesses = meshes.map((mesh) => mesh.material.roughness)
  assert.deepEqual(roughnesses, [0, 0.16, 0.33])
  for (const mesh of meshes) {
    assert.equal(mesh.geometry.getAttribute('position')?.count, 5374)
    assert.equal(mesh.geometry.getAttribute('normal')?.count, 5374)
    assert.equal(mesh.geometry.index?.count, 31800)
  }

  camera.aspect = 16 / 9
  camera.updateProjectionMatrix()
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 160,
    height: 90,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.1, 'DirectionalLight sample should render visible imported-light geometry')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > mean.b + 1 && mean.g > mean.b + 1, `imported yellow light should tint the rendered samples (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets Duck fixture loads textured external assets', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_DUCK, 'utf8'))
  assert.equal(source.buffers[0].uri, 'Duck0.bin')
  assert.equal(source.images[0].uri, 'DuckCM.png')
  assert.deepEqual(source.samplers, [
    {
      magFilter: 9729,
      minFilter: 9986,
      wrapS: 10497,
      wrapT: 10497,
    },
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_DUCK)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'Khronos Duck sample should load a mesh')
  assert.equal(mesh.name, 'LOD3spShape')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 2399)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 2399)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 2399)
  assert.equal(mesh.geometry.index?.count, 12636)
  assert.equal(mesh.material.name, 'blinn3-fx')
  assert.equal(mesh.material.metalness, 0)

  const texture = mesh.material.map
  assert.ok(texture?.isTexture, 'Duck sample should load a base color texture')
  assert.equal(texture.name, 'DuckCM.png')
  assert.equal(Buffer.isBuffer(texture.image), true, 'Duck external PNG should load as an encoded Buffer')
  assert.deepEqual(pngDimensions(texture.image), [512, 512])
  assert.equal(texture.wrapS, THREE.RepeatWrapping)
  assert.equal(texture.wrapT, THREE.RepeatWrapping)
  assert.equal(texture.magFilter, THREE.LinearFilter)
  assert.equal(texture.minFilter, THREE.NearestMipmapLinearFilter)
  assert.equal(texture.colorSpace, THREE.SRGBColorSpace)
  assert.equal(texture.flipY, false)

  const camera = gltf.cameras[0]
  assert.ok(camera, 'Khronos Duck sample should load an imported camera')
  camera.aspect = 1
  camera.updateProjectionMatrix()

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 1.2)
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.05, 'Khronos Duck sample should render visible textured pixels')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > mean.b + 8 && mean.g > mean.b + 6, `Duck texture should contribute warm yellow output (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets DamagedHelmet fixture loads canonical packed PBR texture set', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_DAMAGED_HELMET, 'utf8'))
  assert.equal(source.extensionsUsed, undefined)
  assert.deepEqual(source.buffers, [
    { byteLength: 558504, uri: 'DamagedHelmet.bin' },
  ])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'Default_albedo.jpg',
    'Default_metalRoughness.jpg',
    'Default_emissive.jpg',
    'Default_AO.jpg',
    'Default_normal.jpg',
  ])
  assert.deepEqual(source.materials.map((material) => [
    material.name,
    material.pbrMetallicRoughness?.baseColorTexture?.index,
    material.pbrMetallicRoughness?.metallicRoughnessTexture?.index,
    material.emissiveTexture?.index,
    material.occlusionTexture?.index,
    material.normalTexture?.index,
  ]), [
    ['Material_MR', 0, 1, 2, 3, 4],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_DAMAGED_HELMET)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 1)
  const mesh = meshes[0]
  assert.equal(mesh.name, 'node_damagedHelmet_-6514')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 14556)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 14556)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 14556)
  assert.equal(mesh.geometry.index?.count, 46356)

  const material = mesh.material
  assert.equal(material.name, 'Material_MR')
  assert.equal(material.isMeshStandardMaterial, true)
  assert.deepEqual(material.color.toArray(), [1, 1, 1])
  assert.equal(material.metalness, 1)
  assert.equal(material.roughness, 1)
  assert.deepEqual(material.emissive.toArray(), [1, 1, 1])
  assert.equal(material.emissiveIntensity, 1)
  assert.deepEqual(material.normalScale.toArray(), [1, -1])
  assert.equal(material.aoMapIntensity, 1)

  assert.equal(material.map.name, 'Default_albedo.jpg')
  assert.equal(material.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(material.emissiveMap.name, 'Default_emissive.jpg')
  assert.equal(material.emissiveMap.colorSpace, THREE.SRGBColorSpace)
  assert.equal(material.roughnessMap.name, 'Default_metalRoughness.jpg')
  assert.equal(material.metalnessMap, material.roughnessMap)
  assert.equal(material.roughnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(material.aoMap.name, 'Default_AO.jpg')
  assert.equal(material.aoMap.colorSpace, THREE.NoColorSpace)
  assert.equal(material.normalMap.name, 'Default_normal.jpg')
  assert.equal(material.normalMap.colorSpace, THREE.NoColorSpace)
  for (const texture of [material.map, material.emissiveMap, material.roughnessMap, material.aoMap, material.normalMap]) {
    assert.equal(Buffer.isBuffer(texture.image), true, `${texture.name} should load as an encoded Buffer`)
    assert.equal(texture.flipY, false)
  }

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 50)
  camera.position.copy(center).add(new THREE.Vector3(0, size.y * 0.1, Math.max(size.x, size.y, size.z) * 2.2))
  camera.lookAt(center)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.5))
  const light = new THREE.DirectionalLight(0xffffff, 2.5)
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.25, 'DamagedHelmet should render visible textured PBR geometry')
  const mean = meanRgba(rgba)
  assert.ok(mean.g > mean.r && mean.b > mean.r, `DamagedHelmet should render cool lit PBR output (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets WaterBottle fixture loads textured PBR maps', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_WATER_BOTTLE, 'utf8'))
  assert.equal(source.asset.generator, 'glTF Tools for Unity')
  assert.deepEqual(source.buffers, [{ uri: 'WaterBottle.bin', byteLength: 149412 }])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'WaterBottle_baseColor.png',
    'WaterBottle_occlusionRoughnessMetallic.png',
    'WaterBottle_normal.png',
    'WaterBottle_emissive.png',
  ])
  assert.deepEqual(source.textures, [
    { source: 0 },
    { source: 1 },
    { source: 2 },
    { source: 3 },
  ])
  assert.equal(source.materials.length, 1)
  assert.equal(source.materials[0].name, 'BottleMat')
  assert.deepEqual(source.materials[0].pbrMetallicRoughness, {
    baseColorTexture: { index: 0 },
    metallicRoughnessTexture: { index: 1 },
  })
  assert.deepEqual(source.materials[0].normalTexture, { index: 2 })
  assert.deepEqual(source.materials[0].occlusionTexture, { index: 1 })
  assert.deepEqual(source.materials[0].emissiveTexture, { index: 3 })
  assert.deepEqual(source.materials[0].emissiveFactor, [1, 1, 1])

  const primitive = source.meshes[0].primitives[0]
  assert.deepEqual(primitive.attributes, {
    TEXCOORD_0: 0,
    NORMAL: 1,
    TANGENT: 2,
    POSITION: 3,
  })
  assert.equal(primitive.indices, 4)
  assert.deepEqual(source.nodes[0].rotation, [0, 1, 0, 0])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_WATER_BOTTLE)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'Khronos WaterBottle sample should load a mesh')
  assert.equal(mesh.name, 'WaterBottle')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 2549)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 2549)
  assert.equal(mesh.geometry.getAttribute('tangent')?.count, 2549)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 2549)
  assert.equal(mesh.geometry.index?.count, 13530)
  assert.deepEqual(mesh.quaternion.toArray(), [0, 1, 0, 0])

  const material = mesh.material
  assert.equal(material.isMeshStandardMaterial, true)
  assert.equal(material.name, 'BottleMat')
  assert.equal(material.metalness, 1)
  assert.equal(material.roughness, 1)
  assert.deepEqual(material.emissive.toArray(), [1, 1, 1])

  const assertLoadedTexture = (texture, name, colorSpace) => {
    assert.ok(texture?.isTexture, `${name} should load a texture`)
    assert.equal(texture.name, name)
    assert.equal(Buffer.isBuffer(texture.image), true, `${name} should load as an encoded Buffer`)
    assert.deepEqual(pngDimensions(texture.image), [2048, 2048])
    assert.equal(texture.wrapS, THREE.RepeatWrapping)
    assert.equal(texture.wrapT, THREE.RepeatWrapping)
    assert.equal(texture.magFilter, THREE.LinearFilter)
    assert.equal(texture.minFilter, THREE.LinearMipmapLinearFilter)
    assert.equal(texture.colorSpace, colorSpace)
    assert.equal(texture.flipY, false)
  }

  assertLoadedTexture(material.map, 'WaterBottle_baseColor.png', THREE.SRGBColorSpace)
  assertLoadedTexture(material.metalnessMap, 'WaterBottle_occlusionRoughnessMetallic.png', THREE.NoColorSpace)
  assertLoadedTexture(material.normalMap, 'WaterBottle_normal.png', THREE.NoColorSpace)
  assertLoadedTexture(material.emissiveMap, 'WaterBottle_emissive.png', THREE.SRGBColorSpace)
  assert.equal(material.metalnessMap, material.roughnessMap, 'WaterBottle should reuse the ORM texture for roughness')
  assert.equal(material.metalnessMap, material.aoMap, 'WaterBottle should reuse the ORM texture for occlusion')

  const camera = new THREE.OrthographicCamera(-0.08, 0.08, 0.16, -0.16, 0.01, 2)
  camera.position.set(0, 0, 0.6)
  camera.lookAt(0, 0, 0)

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 1.5)
  light.position.set(0.2, 0.4, 0.7)
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.08, 'WaterBottle should render visible textured PBR geometry')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > mean.b + 15 && mean.g > mean.b + 15, `WaterBottle texture should contribute warm label pixels (${mean.r}, ${mean.g}, ${mean.b})`)
})
