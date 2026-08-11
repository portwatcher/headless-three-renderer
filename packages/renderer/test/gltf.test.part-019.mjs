import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'
import { Renderer, SAMPLE_ASSET_EMISSIVE_STRENGTH_TEST, SAMPLE_ASSET_ENVIRONMENT_TEST, SAMPLE_ASSET_SIMPLE_SKIN, SAMPLE_ASSET_SUNGLASSES_KHRONOS, SAMPLE_ASSET_TOY_CAR } from './gltf.test.part-001.mjs'
import { findFirst, loadGltfFixture, meanRegion, nonBackgroundBounds, pngDimensions } from './gltf.test.part-028.mjs'
test('committed Khronos glTF Sample Assets SunglassesKhronos fixture loads transmission, volume, IOR, and iridescence lenses', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_SUNGLASSES_KHRONOS, 'utf8'))
  assert.equal(source.extensionsRequired, undefined)
  assert.deepEqual(source.extensionsUsed, [
    'KHR_materials_transmission',
    'KHR_materials_volume',
    'KHR_materials_ior',
    'KHR_materials_iridescence',
  ])
  assert.deepEqual(source.buffers, [
    { uri: 'SunglassesKhronos_data.bin', byteLength: 277272 },
  ])
  assert.deepEqual(source.images, [{ uri: 'SunglassesKhronos.png' }])
  assert.deepEqual(source.textures, [{ source: 0 }])
  assert.deepEqual(source.materials.map((material) => material.name), [
    'earhooks',
    'temples',
    'nose_pads',
    'lens_interior',
    'lens_exterior',
  ])
  assert.deepEqual(source.materials[2].extensions, {
    KHR_materials_transmission: {
      transmissionFactor: 1,
    },
    KHR_materials_volume: {
      attenuationColor: [1, 1, 1],
      attenuationDistance: 0.006999999999999999,
      thicknessFactor: 0.01,
    },
  })
  assert.deepEqual(source.materials[4].extensions, {
    KHR_materials_ior: {
      ior: 1,
    },
    KHR_materials_iridescence: {
      iridescenceFactor: 1,
      iridescenceIor: 2,
      iridescenceThicknessMaximum: 300.01,
    },
    KHR_materials_transmission: {
      transmissionFactor: 1,
    },
  })
  assert.deepEqual(source.meshes.map((mesh) => mesh.name), [
    'EarhookRight',
    'TempleRight',
    'EarhookLeft',
    'TempleLeft',
    'Nosepads',
    'Frames',
    'LensesInterior',
    'LensesExterior',
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_SUNGLASSES_KHRONOS)
  assert.deepEqual(gltf.parser?.json?.extensionsUsed, source.extensionsUsed)

  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 8)

  const expectedMeshes = [
    ['EarhookRight_1', 1176, 6696, 'earhooks', true],
    ['TempleRight_1', 212, 858, 'temples', false],
    ['EarhookLeft_1', 1176, 6696, 'earhooks', true],
    ['TempleLeft_1', 212, 858, 'temples', false],
    ['Nosepads_1', 452, 2688, 'nose_pads', false],
    ['Frames_1', 3036, 16248, 'temples', false],
    ['LensesInterior_1', 578, 3072, 'lens_interior', false],
    ['LensesExterior_1', 578, 3072, 'lens_exterior', false],
  ]
  for (const [name, vertexCount, indexCount, materialName, hasUv] of expectedMeshes) {
    const mesh = meshes.find((candidate) => candidate.name === name)
    assert.ok(mesh, `${name} should load`)
    assert.equal(mesh.geometry.getAttribute('position')?.count, vertexCount)
    assert.equal(mesh.geometry.getAttribute('normal')?.count, vertexCount)
    assert.equal(mesh.geometry.getAttribute('uv')?.count, hasUv ? vertexCount : undefined)
    assert.equal(mesh.geometry.index?.count, indexCount)
    assert.equal(mesh.material.name, materialName)
  }

  const materials = new Map(meshes.map((mesh) => [mesh.material.name, mesh.material]))
  const earhooks = materials.get('earhooks')
  assert.equal(earhooks?.isMeshStandardMaterial, true)
  assert.deepEqual(earhooks.color.toArray(), [1, 1, 1])
  assert.equal(earhooks.metalness, 0)
  assert.equal(earhooks.roughness, 0.20000022649765015)
  assert.equal(Buffer.isBuffer(earhooks.map?.image), true, 'SunglassesKhronos PNG should load as an encoded Buffer')
  assert.equal(earhooks.map.name, 'SunglassesKhronos.png')
  assert.deepEqual(pngDimensions(earhooks.map.image), [1024, 128])
  assert.equal(earhooks.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(earhooks.map.flipY, false)

  const temples = materials.get('temples')
  assert.equal(temples?.isMeshStandardMaterial, true)
  assert.deepEqual(temples.color.toArray(), [
    0.9159365892410278,
    0.9159365892410278,
    0.9159365892410278,
  ])
  assert.equal(temples.metalness, 1)
  assert.equal(temples.roughness, 0.050000011920928955)

  const nosePads = materials.get('nose_pads')
  assert.equal(nosePads?.isMeshPhysicalMaterial, true)
  assert.equal(nosePads.transmission, 1)
  assert.equal(nosePads.thickness, 0.01)
  assert.equal(nosePads.attenuationDistance, 0.006999999999999999)
  assert.deepEqual(nosePads.attenuationColor.toArray(), [1, 1, 1])
  assert.equal(nosePads.ior, 1.5)

  const lensInterior = materials.get('lens_interior')
  assert.equal(lensInterior?.isMeshPhysicalMaterial, true)
  assert.deepEqual(lensInterior.color.toArray(), [
    0.01606770046055317,
    0.01606770046055317,
    0.01606770046055317,
  ])
  assert.equal(lensInterior.transmission, 1)
  assert.equal(lensInterior.thickness, 0)
  assert.equal(lensInterior.ior, 1.5)

  const lensExterior = materials.get('lens_exterior')
  assert.equal(lensExterior?.isMeshPhysicalMaterial, true)
  assert.deepEqual(lensExterior.color.toArray(), [
    0.009021490812301636,
    0.009021490812301636,
    0.009021490812301636,
  ])
  assert.equal(lensExterior.transmission, 1)
  assert.equal(lensExterior.ior, 1)
  assert.equal(lensExterior.iridescence, 1)
  assert.equal(lensExterior.iridescenceIOR, 2)
  assert.deepEqual(lensExterior.iridescenceThicknessRange, [100, 300.01])

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const camera = new THREE.PerspectiveCamera(35, 1.4, 0.01, 50)
  camera.position.copy(center).add(new THREE.Vector3(0, size.y * 0.25, Math.max(size.x, size.y, size.z) * 2))
  camera.lookAt(center)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.7))
  const light = new THREE.DirectionalLight(0xffffff, 3)
  light.position.copy(center).add(new THREE.Vector3(1, 2, 4))
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 160,
    height: 120,
    format: 'rgba',
    background: [1, 1, 1],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [255, 255, 255], 3) > 0.15, 'SunglassesKhronos should render visible glasses geometry')
  const mean = meanRgba(rgba)
  assert.ok(mean.r < 250 && mean.g < 250 && mean.b < 250, `SunglassesKhronos should render darker lens and frame pixels (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets ToyCar fixture loads clearcoat, fabric sheen, and transmission glass materials', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_TOY_CAR, 'utf8'))
  assert.equal(source.extensionsRequired, undefined)
  assert.deepEqual(source.extensionsUsed, [
    'KHR_texture_transform',
    'KHR_materials_clearcoat',
    'KHR_materials_transmission',
    'KHR_materials_sheen',
  ])
  assert.deepEqual(source.buffers, [
    { uri: 'ToyCar.bin', byteLength: 3664368 },
  ])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'ToyCar_normal.png',
    'ToyCar_emissive.png',
    'ToyCar_basecolor.png',
    'ToyCar_occlusion_roughness_metallic.png',
    'Fabric_normal.png',
    'Fabric_occlusion.png',
    'Fabric_baseColor.png',
    'ToyCar_clearcoat.png',
  ])
  assert.deepEqual(source.materials.map((material) => material.name), ['ToyCar', 'Fabric', 'Glass'])
  assert.deepEqual(source.materials[0].extensions, {
    KHR_materials_clearcoat: {
      clearcoatFactor: 1,
      clearcoatTexture: {
        index: 7,
        texCoord: 0,
      },
    },
  })
  assert.deepEqual(source.materials[1].pbrMetallicRoughness.baseColorTexture.extensions?.KHR_texture_transform, {
    offset: [0, 0],
    scale: [3, 3],
    texCoord: 0,
  })
  assert.deepEqual(source.materials[1].extensions, {
    KHR_materials_sheen: {
      sheenRoughnessFactor: 0.5,
      sheenColorFactor: [1, 0, 0],
    },
  })
  assert.deepEqual(source.materials[2].extensions, {
    KHR_materials_transmission: {
      transmissionFactor: 1,
    },
  })
  assert.deepEqual(source.meshes.map((mesh) => mesh.name), ['ToyCar', 'Fabric', 'Glass'])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_TOY_CAR)
  assert.deepEqual(gltf.parser?.json?.extensionsUsed, source.extensionsUsed)

  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 3)

  const expectedMeshes = [
    ['ToyCar', 66951, 266511, 'ToyCar'],
    ['Fabric', 8959, 52815, 'Fabric'],
    ['Glass', 1519, 7482, 'Glass'],
  ]
  for (const [name, vertexCount, indexCount, materialName] of expectedMeshes) {
    const mesh = meshes.find((candidate) => candidate.name === name)
    assert.ok(mesh, `${name} should load`)
    assert.equal(mesh.geometry.getAttribute('position')?.count, vertexCount)
    assert.equal(mesh.geometry.getAttribute('normal')?.count, vertexCount)
    assert.equal(mesh.geometry.getAttribute('uv')?.count, vertexCount)
    assert.equal(mesh.geometry.index?.count, indexCount)
    assert.equal(mesh.material.name, materialName)
  }

  const materials = new Map(meshes.map((mesh) => [mesh.material.name, mesh.material]))
  const car = materials.get('ToyCar')
  assert.equal(car?.isMeshPhysicalMaterial, true)
  assert.equal(car.metalness, 1)
  assert.equal(car.roughness, 1)
  assert.deepEqual(car.emissive.toArray(), [1, 1, 1])
  assert.equal(car.clearcoat, 1)
  assert.equal(car.map.name, 'ToyCar_basecolor.png')
  assert.equal(car.normalMap.name, 'ToyCar_normal.png')
  assert.equal(car.aoMap.name, 'ToyCar_occlusion_roughness_metallic.png')
  assert.equal(car.roughnessMap.name, 'ToyCar_occlusion_roughness_metallic.png')
  assert.equal(car.metalnessMap.name, 'ToyCar_occlusion_roughness_metallic.png')
  assert.equal(car.emissiveMap.name, 'ToyCar_emissive.png')
  assert.equal(car.clearcoatMap.name, 'ToyCar_clearcoat.png')
  assert.equal(car.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(car.normalMap.colorSpace, THREE.NoColorSpace)
  assert.equal(car.aoMap.colorSpace, THREE.NoColorSpace)
  assert.equal(car.roughnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(car.metalnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(car.emissiveMap.colorSpace, THREE.SRGBColorSpace)
  assert.equal(car.clearcoatMap.colorSpace, THREE.NoColorSpace)
  assert.equal(car.aoMap.channel, 0)

  const fabric = materials.get('Fabric')
  assert.equal(fabric?.isMeshPhysicalMaterial, true)
  assert.deepEqual(fabric.color.toArray(), [0.15, 0.15, 0.15])
  assert.equal(fabric.metalness, 0)
  assert.equal(fabric.roughness, 1)
  assert.equal(fabric.sheen, 1)
  assert.deepEqual(fabric.sheenColor.toArray(), [1, 0, 0])
  assert.equal(fabric.sheenRoughness, 0.5)
  assert.equal(fabric.map.name, 'Fabric_baseColor.png')
  assert.equal(fabric.normalMap.name, 'Fabric_normal.png')
  assert.equal(fabric.aoMap.name, 'Fabric_occlusion.png')
  assert.equal(fabric.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(fabric.normalMap.colorSpace, THREE.NoColorSpace)
  assert.equal(fabric.aoMap.colorSpace, THREE.NoColorSpace)
  assert.equal(fabric.aoMap.channel, 0)
  assert.deepEqual(fabric.map.repeat.toArray(), [3, 3])
  assert.deepEqual(fabric.normalMap.repeat.toArray(), [3, 3])
  assert.deepEqual(fabric.aoMap.repeat.toArray(), [1, 1])

  const glass = materials.get('Glass')
  assert.equal(glass?.isMeshPhysicalMaterial, true)
  assert.deepEqual(glass.color.toArray(), [0.3, 0.8, 0.3])
  assert.equal(glass.metalness, 0)
  assert.equal(glass.roughness, 0)
  assert.equal(glass.transmission, 1)

  const textureExpectations = [
    [car.map, [1024, 1024]],
    [car.normalMap, [1024, 1024]],
    [car.aoMap, [1024, 1024]],
    [car.roughnessMap, [1024, 1024]],
    [car.metalnessMap, [1024, 1024]],
    [car.emissiveMap, [1024, 1024]],
    [car.clearcoatMap, [1024, 1024]],
    [fabric.map, [512, 512]],
    [fabric.normalMap, [512, 512]],
    [fabric.aoMap, [1024, 1024]],
  ]
  for (const [texture, dimensions] of textureExpectations) {
    assert.equal(Buffer.isBuffer(texture.image), true, `${texture.name} should load as an encoded Buffer`)
    assert.deepEqual(pngDimensions(texture.image), dimensions)
    assert.equal(texture.flipY, false)
  }

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const camera = new THREE.PerspectiveCamera(35, 1.2, 0.01, 50)
  camera.position.copy(center).add(new THREE.Vector3(0, size.y * 0.35, Math.max(size.x, size.y, size.z) * 2.2))
  camera.lookAt(center)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.75))
  const light = new THREE.DirectionalLight(0xffffff, 2.6)
  light.position.copy(center).add(new THREE.Vector3(2, 3, 4))
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 160,
    height: 128,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.06, 'ToyCar should render visible clearcoated car geometry')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > mean.g && mean.r > mean.b, `ToyCar should render red car pixels (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets EmissiveStrengthTest fixture loads KHR_materials_emissive_strength factors', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_EMISSIVE_STRENGTH_TEST)
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_emissive_strength'))

  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'Cube4',
    'MeterGrid',
    'Cube2',
    'Cube1',
    'Cube8',
    'Cube16',
  ])

  const materials = new Map(meshes.map((mesh) => [mesh.material.name, mesh.material]))
  assert.deepEqual(
    ['Emit1', 'Emit2', 'Emit4', 'Emit8', 'Emit16'].map((name) => materials.get(name)?.emissiveIntensity),
    [1, 2, 4, 8, 16],
  )
  for (const name of ['Emit1', 'Emit2', 'Emit4', 'Emit8', 'Emit16']) {
    const material = materials.get(name)
    assert.equal(material?.isMeshStandardMaterial, true)
    assert.deepEqual(material.emissive.toArray(), [0.1, 0.5, 0.9])
  }

  const backdrop = materials.get('FlatBackdrop')
  assert.equal(backdrop?.isMeshStandardMaterial, true)
  assert.equal(Buffer.isBuffer(backdrop.map?.image), true, 'emissive-strength backdrop PNG should load as an encoded Buffer')
  assert.equal(backdrop.map.name, 'PlainGrid')
  assert.equal(backdrop.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(backdrop.map.flipY, false)

  const cube1 = meshes.find((mesh) => mesh.name === 'Cube1')
  const cube8 = meshes.find((mesh) => mesh.name === 'Cube8')
  assert.equal(cube1.geometry.getAttribute('position')?.count, 24)
  assert.equal(cube1.geometry.index?.count, 36)
  assert.equal(cube8.geometry.getAttribute('position')?.count, 24)
  assert.equal(cube8.geometry.index?.count, 36)

  const camera = new THREE.OrthographicCamera(-8.8, 8.8, 3.2, -4.6, 0.01, 30)
  camera.position.set(0, 0, 12)
  camera.lookAt(0, 0, 0)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 220,
    height: 110,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.8, 'EmissiveStrengthTest should render visible emissive-strength samples')
  const low = meanRegion(rgba, 220, 110, 28, 34, 50, 55)
  const high = meanRegion(rgba, 220, 110, 139, 34, 161, 55)
  assert.ok(high.g > low.g + 30, `higher emissive strength should brighten the green channel (${high.g} vs ${low.g})`)
  assert.ok(high.b > low.b + 20, `higher emissive strength should brighten the blue channel (${high.b} vs ${low.b})`)
})

test('committed Khronos glTF Sample Assets EnvironmentTest fixture loads imported camera and metallic-roughness sphere grids', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_ENVIRONMENT_TEST, 'utf8'))
  assert.equal(source.buffers[0].uri, 'EnvironmentTest_binary.bin')
  assert.equal(source.buffers[0].byteLength, 340472)
  assert.deepEqual(source.images.map((image) => image.uri), [
    'EnvironmentTest_images/roughness_metallic_0.png',
    'EnvironmentTest_images/roughness_metallic_1.png',
  ])
  assert.deepEqual(source.materials.map((material) => [material.name, material.doubleSided ?? false]), [
    ['MetallicSpheresMat', true],
    ['DielectricSpheresMat', true],
    ['DielectricSpheresMat', true],
  ])
  assert.equal(source.meshes.length, 3)
  assert.equal(source.cameras.length, 1)

  const gltf = await loadGltfFixture(SAMPLE_ASSET_ENVIRONMENT_TEST)
  assert.equal(gltf.cameras.length, 1)
  const importedCamera = gltf.cameras[0]
  assert.equal(importedCamera.name, 'render_camera_n3d')
  assert.equal(importedCamera.isPerspectiveCamera, true)
  assert.ok(Math.abs(importedCamera.fov - 34.515876027228366) < 1e-6, `EnvironmentTest camera fov should load (${importedCamera.fov})`)

  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), ['Metallic0', 'Dielectric0', 'Dielectric0-Black'])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [4598, 4598, 4598])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [4598, 4598, 4598])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [4598, 4598, 4598])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [25344, 25344, 25344])

  const [metallic, dielectric, black] = meshes.map((mesh) => mesh.material)
  assert.deepEqual([metallic.name, dielectric.name, black.name], ['MetallicSpheresMat', 'DielectricSpheresMat', 'DielectricSpheresMat'])
  assert.ok([metallic, dielectric, black].every((material) => material.side === THREE.DoubleSide))
  assert.deepEqual(black.color.toArray(), [0, 0, 0])
  assert.equal(Buffer.isBuffer(metallic.roughnessMap?.image), true, 'metallic roughness PNG should load as an encoded Buffer')
  assert.equal(Buffer.isBuffer(dielectric.roughnessMap?.image), true, 'dielectric roughness PNG should load as an encoded Buffer')
  assert.deepEqual(pngDimensions(metallic.roughnessMap.image), [512, 512])
  assert.deepEqual(pngDimensions(dielectric.roughnessMap.image), [512, 512])
  assert.equal(metallic.roughnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(dielectric.roughnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(metallic.roughnessMap, metallic.metalnessMap)
  assert.equal(dielectric.roughnessMap, dielectric.metalnessMap)
  assert.equal(black.roughnessMap, black.metalnessMap)

  importedCamera.aspect = 1.5
  importedCamera.updateProjectionMatrix()
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.7))
  const light = new THREE.DirectionalLight(0xffffff, 1.5)
  light.position.set(1, 5, 10)
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)
  importedCamera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, importedCamera, {
    width: 144,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.05, 'EnvironmentTest should render visible metallic-roughness sphere grids through its imported camera')
})

test('committed Khronos glTF Sample Assets SimpleSkin fixture applies skin animation', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_SIMPLE_SKIN)
  const mesh = findFirst(gltf.scene, (object) => object.isSkinnedMesh === true)
  assert.ok(mesh, 'Khronos SimpleSkin sample should load a SkinnedMesh')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 10)
  assert.equal(mesh.geometry.getAttribute('skinIndex')?.count, 10)
  assert.equal(mesh.geometry.getAttribute('skinWeight')?.count, 10)
  assert.equal(mesh.geometry.index?.count, 24)
  assert.equal(mesh.skeleton.bones.length, 2)
  assert.equal(gltf.animations.length, 1)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 10)
  camera.position.set(0, 1, 4)
  camera.lookAt(0, 1, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1.0))
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const renderer = new Renderer()
  const renderBounds = () => nonBackgroundBounds(renderer.render(gltf.scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  }), 96, 96, [0, 0, 0], 3)

  const base = renderBounds()
  const mixer = new THREE.AnimationMixer(gltf.scene)
  mixer.clipAction(gltf.animations[0]).play()
  mixer.setTime(1)
  gltf.scene.updateMatrixWorld(true)
  const animated = renderBounds()

  assert.ok(base.height > 50, `SimpleSkin base pose should render a tall strip (${base.height})`)
  assert.ok(animated.width > base.width + 10, `SimpleSkin animation should widen the skinned mesh (${animated.width} vs ${base.width})`)
  assert.ok(animated.minY > base.minY + 10, `SimpleSkin animation should bend the top downward (${animated.minY} vs ${base.minY})`)
})
