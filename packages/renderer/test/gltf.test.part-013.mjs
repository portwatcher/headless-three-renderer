import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'
import { Renderer, SAMPLE_ASSET_ANISOTROPY_BARN_LAMP, SAMPLE_ASSET_ANISOTROPY_DISC_TEST, SAMPLE_ASSET_ANISOTROPY_ROTATION_TEST, SAMPLE_ASSET_ANISOTROPY_STRENGTH_TEST, SAMPLE_ASSET_BOX_VERTEX_COLORS, SAMPLE_ASSET_VERTEX_COLOR_TEST } from './gltf.test.part-001.mjs'
import { assertVectorClose, findFirst, loadGltfFixture, meanRegion, pngDimensions } from './gltf.test.part-028.mjs'
test('committed Khronos glTF Sample Assets BoxVertexColors fixture renders COLOR_0 gradients', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_BOX_VERTEX_COLORS)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'Khronos BoxVertexColors sample should load a mesh')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 24)
  assert.equal(mesh.geometry.getAttribute('color')?.count, 24)
  assert.equal(mesh.geometry.index?.count, 36)
  assert.equal(mesh.material.vertexColors, true)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 10)
  camera.position.set(1.4, 1.1, 2.2)
  camera.lookAt(0, 0, 0)

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 1.0)
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.4, 'Khronos BoxVertexColors sample should render visible pixels')
  const topLeft = meanRegion(rgba, 96, 96, 24, 22, 36, 34)
  const bottomLeft = meanRegion(rgba, 96, 96, 24, 58, 36, 68)
  const bottomRight = meanRegion(rgba, 96, 96, 62, 58, 74, 68)
  assert.ok(topLeft.g > bottomLeft.g + 80, `vertex color gradient should make the upper-left face greener than lower-left (${topLeft.g} vs ${bottomLeft.g})`)
  assert.ok(bottomRight.r > bottomLeft.r + 80, `vertex color gradient should make the lower-right face redder than lower-left (${bottomRight.r} vs ${bottomLeft.r})`)
  assert.ok(bottomLeft.b > 170 && bottomRight.b > 170, `vertex color gradient should keep blue channel visible (${bottomLeft.b}, ${bottomRight.b})`)
})

test('committed Khronos glTF Sample Assets VertexColorTest fixture combines textures with COLOR_0 attributes', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_VERTEX_COLOR_TEST, 'utf8'))
  assert.equal(source.buffers[0].uri, 'VertexColorTest.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'VertexColorTestLabels.png',
    'VertexColorChecks.png',
  ])
  assert.deepEqual(source.meshes.map((mesh) => mesh.name), ['LabelMesh', 'VertexColorTestMesh'])
  assert.equal(source.meshes[1].primitives[0].attributes.COLOR_0, 10)

  const gltf = await loadGltfFixture(SAMPLE_ASSET_VERTEX_COLOR_TEST)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })

  assert.deepEqual(meshes.map((mesh) => ({
    name: mesh.name,
    material: mesh.material.name,
    positions: mesh.geometry.getAttribute('position')?.count,
    normals: mesh.geometry.getAttribute('normal')?.count,
    tangents: mesh.geometry.getAttribute('tangent')?.count,
    uvs: mesh.geometry.getAttribute('uv')?.count,
    colors: mesh.geometry.getAttribute('color')
      ? {
          count: mesh.geometry.getAttribute('color').count,
          itemSize: mesh.geometry.getAttribute('color').itemSize,
          normalized: mesh.geometry.getAttribute('color').normalized,
        }
      : null,
    index: mesh.geometry.index?.count,
    vertexColors: mesh.material.vertexColors,
    map: mesh.material.map?.name,
  })), [
    {
      name: 'Labels',
      material: 'Label_Mat',
      positions: 24,
      normals: 24,
      tangents: 24,
      uvs: 24,
      colors: null,
      index: 36,
      vertexColors: false,
      map: 'VertexColorTestLabels.png',
    },
    {
      name: 'VertexColorTest',
      material: 'VC_Checks_Mat',
      positions: 48,
      normals: 48,
      tangents: 48,
      uvs: 48,
      colors: { count: 48, itemSize: 4, normalized: false },
      index: 72,
      vertexColors: true,
      map: 'VertexColorChecks.png',
    },
  ])

  for (const mesh of meshes) {
    assert.equal(Buffer.isBuffer(mesh.material.map.image), true, `${mesh.name} should load an encoded PNG texture`)
    assert.deepEqual(pngDimensions(mesh.material.map.image), [256, 256])
    assert.equal(mesh.material.map.colorSpace, THREE.SRGBColorSpace)
    assert.equal(mesh.material.map.flipY, false)
  }

  const color = meshes[1].geometry.getAttribute('color')
  const min = [Infinity, Infinity, Infinity, Infinity]
  const max = [-Infinity, -Infinity, -Infinity, -Infinity]
  for (let i = 0; i < color.count; i += 1) {
    for (let component = 0; component < 4; component += 1) {
      const value = color.getComponent(i, component)
      min[component] = Math.min(min[component], value)
      max[component] = Math.max(max[component], value)
    }
  }
  assertVectorClose(min, [0, 0, 0, 1], 'VertexColorTest COLOR_0 minimum')
  assertVectorClose(max, [1, 1, 1, 1], 'VertexColorTest COLOR_0 maximum')

  const camera = new THREE.OrthographicCamera(-1.5, 1.5, 1.5, -1.5, 0.01, 20)
  camera.position.set(0, 0, 4)
  camera.lookAt(0, 0, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1.0))
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 160,
    height: 160,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.25, 'VertexColorTest should render visible textured vertex-color swatches')
  const center = meanRegion(rgba, 160, 160, 60, 60, 100, 100)
  assert.ok(center.b > center.r + 60 && center.b > center.g + 50, `VertexColorTest center should include the blue check texture (${center.r}, ${center.g}, ${center.b})`)
})

test('committed Khronos glTF Sample Assets AnisotropyBarnLamp fixture loads anisotropy, clearcoat, emissive, and glass materials', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_ANISOTROPY_BARN_LAMP, 'utf8'))
  assert.deepEqual(source.extensionsUsed, [
    'KHR_materials_anisotropy',
    'KHR_materials_clearcoat',
    'KHR_materials_emissive_strength',
    'KHR_materials_transmission',
    'KHR_materials_volume',
  ])
  assert.deepEqual(source.buffers, [{ uri: 'AnisotropyBarnLamp.bin', byteLength: 409580 }])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'AnisotropyBarnLamp_normalbump.png',
    'AnisotropyBarnLamp_occlusionroughnessmetal.png',
    'AnisotropyBarnLamp_basecolor.png',
    'AnisotropyBarnLamp_anisotropy.png',
  ])
  assert.deepEqual(source.meshes.map((mesh) => mesh.name), ['Lamp Metal', 'Lamp Filament', 'Lamp Glass'])
  assert.deepEqual(source.materials.map((material) => material.name), ['Lamp Metal', 'Lamp Filament', 'Lamp Glass'])
  assert.deepEqual(source.materials[0].extensions, {
    KHR_materials_anisotropy: {
      anisotropyStrength: 1,
      anisotropyRotation: 0,
      anisotropyTexture: { index: 3 },
    },
    KHR_materials_clearcoat: {
      clearcoatFactor: 0.25,
      clearcoatRoughnessFactor: 0.15,
      clearcoatNormalTexture: { index: 0 },
    },
  })
  assert.deepEqual(source.materials[1].extensions, {
    KHR_materials_emissive_strength: {
      emissiveStrength: 25,
    },
  })
  assert.deepEqual(source.materials[2].extensions, {
    KHR_materials_transmission: {
      transmissionFactor: 1,
    },
    KHR_materials_volume: {
      thicknessFactor: 0.01,
    },
  })

  const gltf = await loadGltfFixture(SAMPLE_ASSET_ANISOTROPY_BARN_LAMP)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), ['Lamp_Metal', 'Lamp_Filament', 'Lamp_Glass'])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [6803, 140, 769])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [6803, 140, 769])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [25257, 840, 4512])

  const [metalMesh, filamentMesh, glassMesh] = meshes
  const metal = metalMesh.material
  assert.equal(metal.name, 'Lamp Metal')
  assert.equal(metal.isMeshPhysicalMaterial, true)
  assert.equal(metal.metalness, 1)
  assert.equal(metal.roughness, 1)
  assert.equal(metal.anisotropy, 1)
  assert.equal(metal.anisotropyRotation, 0)
  assert.equal(metal.clearcoat, 0.25)
  assert.equal(metal.clearcoatRoughness, 0.15)
  assert.equal(metal.map.name, 'AnisotropyBarnLamp_basecolor.png')
  assert.equal(metal.normalMap.name, 'AnisotropyBarnLamp_normalbump.png')
  assert.equal(metal.roughnessMap.name, 'AnisotropyBarnLamp_occlusionroughnessmetal.png')
  assert.equal(metal.metalnessMap, metal.roughnessMap)
  assert.equal(metal.aoMap, metal.roughnessMap)
  assert.equal(metal.anisotropyMap.name, 'AnisotropyBarnLamp_anisotropy.png')
  assert.equal(Buffer.isBuffer(metal.map.image), true, 'BarnLamp base-color PNG should load as an encoded Buffer')
  assert.deepEqual(pngDimensions(metal.map.image), [2048, 2048])
  assert.deepEqual(pngDimensions(metal.anisotropyMap.image), [2048, 2048])
  assert.equal(metal.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(metal.normalMap.colorSpace, THREE.NoColorSpace)
  assert.equal(metal.anisotropyMap.colorSpace, THREE.NoColorSpace)

  const filament = filamentMesh.material
  assert.equal(filament.name, 'Lamp Filament')
  assert.equal(filament.isMeshStandardMaterial, true)
  assert.deepEqual(filament.color.toArray(), [0.09, 0.09, 0.09])
  assert.deepEqual(filament.emissive.toArray(), [1, 0.5, 0.25])
  assert.equal(filament.emissiveIntensity, 25)
  assert.equal(filament.roughness, 0.7)

  const glass = glassMesh.material
  assert.equal(glass.name, 'Lamp Glass')
  assert.equal(glass.isMeshPhysicalMaterial, true)
  assert.equal(glass.transmission, 1)
  assert.equal(glass.thickness, 0.01)
  assert.equal(glass.roughness, 0)

  gltf.scene.updateMatrixWorld(true)
  const box = new THREE.Box3().setFromObject(gltf.scene)
  const center = box.getCenter(new THREE.Vector3())
  const size = box.getSize(new THREE.Vector3())
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.95))
  const light = new THREE.DirectionalLight(0xffffff, 2.2)
  light.position.copy(center).add(new THREE.Vector3(0.5, 0.8, 1))
  gltf.scene.add(light)
  const padding = 0.02
  const halfHeight = size.y / 2 + padding
  const halfWidth = Math.max(size.x / 2 + padding, size.z / 2 + padding, halfHeight)
  const camera = new THREE.OrthographicCamera(-halfWidth, halfWidth, halfHeight, -halfHeight, 0.001, 10)
  camera.position.copy(center).add(new THREE.Vector3(0, 0, 2))
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.18, 'AnisotropyBarnLamp should render visible anisotropic lamp geometry')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > mean.g && mean.g > mean.b, `BarnLamp materials should render warm metal and filament colors (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets AnisotropyDiscTest fixture loads KHR_materials_anisotropy texture inputs', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_ANISOTROPY_DISC_TEST)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'Box001',
    'Box002',
    'Box003',
    'Box004',
    'Box005',
    'Box006',
    'Box007',
    'Box008',
    'Box009',
    'Box010',
    'Text',
    'Box000',
  ])
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_anisotropy'))

  const materials = new Map(meshes.map((mesh) => [mesh.material.name, mesh.material]))
  const smooth = materials.get('roughness 0.0')
  const rough = materials.get('roughness 1.0')
  assert.equal(smooth?.isMeshPhysicalMaterial, true)
  assert.equal(rough?.isMeshPhysicalMaterial, true)
  assert.equal(smooth.metalness, 1)
  assert.equal(smooth.roughness, 0)
  assert.equal(smooth.anisotropy, 1)
  assert.equal(smooth.anisotropyRotation, 0)
  assert.equal(rough.roughness, 1)
  assert.equal(rough.anisotropy, 1)

  const anisotropyMap = smooth.anisotropyMap
  assert.equal(Buffer.isBuffer(anisotropyMap?.image), true, 'anisotropy PNG should load as an encoded Buffer')
  assert.equal(anisotropyMap.name, 'AnisotropyDiscs')
  assert.equal(anisotropyMap.colorSpace, THREE.NoColorSpace)
  assert.equal(anisotropyMap.wrapS, THREE.RepeatWrapping)
  assert.equal(anisotropyMap.wrapT, THREE.RepeatWrapping)
  assert.equal(anisotropyMap.magFilter, THREE.LinearFilter)
  assert.equal(anisotropyMap.minFilter, THREE.LinearMipmapLinearFilter)
  assert.equal(anisotropyMap.flipY, false)

  const firstDisc = meshes[0]
  assert.equal(firstDisc.geometry.getAttribute('position')?.count, 9)
  assert.equal(firstDisc.geometry.getAttribute('normal')?.count, 9)
  assert.equal(firstDisc.geometry.getAttribute('uv')?.count, 9)
  assert.equal(firstDisc.geometry.getAttribute('tangent')?.count, 9)
  assert.equal(firstDisc.geometry.index?.count, 24)

  const camera = new THREE.OrthographicCamera(-4.2, 3.2, 3.0, -3.2, 0.01, 30)
  camera.position.set(-0.5, -0.1, 8)
  camera.lookAt(-0.5, -0.1, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.4))
  const light = new THREE.DirectionalLight(0xffffff, 3)
  light.position.set(0, 2, 8)
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.2, 'AnisotropyDiscTest should render visible anisotropic material panels')
})

test('committed Khronos glTF Sample Assets AnisotropyRotationTest fixture loads anisotropy rotations and direction textures', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_ANISOTROPY_ROTATION_TEST, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_anisotropy'])
  assert.equal(source.buffers[0].uri, 'AnisoDonuts.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'GridWithMarkers.png',
    'GridWithMarkers_30deg.png',
    'AnisoRotation30_Linear.png',
    'AnisoRotation10_Linear.png',
    'Heights_1d_Normals_v2.png',
    'AnisoDonutLabels.png',
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_ANISOTROPY_ROTATION_TEST)
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_anisotropy'))

  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'Band_1L',
    'Band_2L',
    'Band_4L',
    'Band_5L',
    'Band_1R',
    'Band_2R',
    'Band_4R',
    'Band_5R',
    'Band_3L',
    'Band_3R',
    'Labels',
  ])
  assert.ok(meshes.slice(0, 10).every((mesh) => mesh.geometry.getAttribute('position')?.count === 715))
  assert.ok(meshes.slice(0, 10).every((mesh) => mesh.geometry.getAttribute('normal')?.count === 715))
  assert.ok(meshes.slice(0, 10).every((mesh) => mesh.geometry.getAttribute('tangent')?.count === 715))
  assert.ok(meshes.slice(0, 10).every((mesh) => mesh.geometry.index?.count === 3840))

  const materials = new Map(meshes.map((mesh) => [mesh.material.name, mesh.material]))
  const base = materials.get('Aniso Tangents')
  const rotated = materials.get('Aniso Tan + Rotation')
  const textured = materials.get('Aniso Tan + Texture')
  const rotatedTextured = materials.get('Aniso Tan + Rotation + Texture')
  const normalSimulation = materials.get('Simulation via normal')
  assert.equal(base?.isMeshPhysicalMaterial, true)
  assert.equal(rotated?.isMeshPhysicalMaterial, true)
  assert.equal(textured?.isMeshPhysicalMaterial, true)
  assert.equal(rotatedTextured?.isMeshPhysicalMaterial, true)
  assert.equal(base.anisotropy, 0.5)
  assert.equal(base.anisotropyRotation, 0)
  assert.ok(Math.abs(rotated.anisotropyRotation - 0.523598775598) < 1e-12)
  assert.ok(Math.abs(rotatedTextured.anisotropyRotation - 0.349065850398866) < 1e-12)

  assert.equal(Buffer.isBuffer(base.map?.image), true, 'base anisotropy sample grid should load as an encoded Buffer')
  assert.equal(base.map.name, 'GridWithMarkers')
  assert.deepEqual(pngDimensions(base.map.image), [1024, 1024])
  assert.equal(rotated.map.name, 'GridWithMarkers_30deg')
  assert.deepEqual(pngDimensions(rotated.map.image), [1024, 1024])

  assert.equal(Buffer.isBuffer(textured.anisotropyMap?.image), true, '30 degree anisotropy direction map should load as an encoded Buffer')
  assert.equal(textured.anisotropyMap.name, 'AnisoRotation30_Linear')
  assert.equal(textured.anisotropyMap.colorSpace, THREE.NoColorSpace)
  assert.deepEqual(pngDimensions(textured.anisotropyMap.image), [4, 4])
  assert.equal(Buffer.isBuffer(rotatedTextured.anisotropyMap?.image), true, '10 degree anisotropy direction map should load as an encoded Buffer')
  assert.equal(rotatedTextured.anisotropyMap.name, 'AnisoRotation10_Linear')
  assert.equal(rotatedTextured.anisotropyMap.colorSpace, THREE.NoColorSpace)
  assert.deepEqual(pngDimensions(rotatedTextured.anisotropyMap.image), [4, 4])

  assert.equal(normalSimulation?.isMeshStandardMaterial, true)
  assert.equal(normalSimulation.normalMap.name, 'Heights_1d_Normals_v2')
  assert.equal(normalSimulation.normalMap.colorSpace, THREE.NoColorSpace)
  assert.deepEqual(pngDimensions(normalSimulation.normalMap.image), [2048, 1])

  const label = meshes.find((mesh) => mesh.name === 'Labels')
  assert.equal(label?.material.map.name, 'AnisoDonutLabels')
  assert.deepEqual(pngDimensions(label.material.map.image), [512, 512])

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.35))
  const light = new THREE.DirectionalLight(0xffffff, 3)
  light.position.set(0, 2, 8)
  gltf.scene.add(light)
  const camera = new THREE.OrthographicCamera(-2.8, 2.8, 2.7, -2.7, 0.01, 30)
  camera.position.set(0, 0, 8)
  camera.lookAt(0, 0, 0)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 128,
    height: 128,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.5, 'AnisotropyRotationTest should render visible rotated anisotropy bands')
})

test('committed Khronos glTF Sample Assets AnisotropyStrengthTest fixture loads anisotropy strength grid', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_ANISOTROPY_STRENGTH_TEST, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_anisotropy'])
  assert.equal(source.buffers[0].uri, 'AnisotropyStrengthTest_data.bin')
  assert.deepEqual(source.images.map((image) => image.uri), ['AnisotropySpheresLabels.png'])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_ANISOTROPY_STRENGTH_TEST)
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_anisotropy'))

  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 50)
  const spheres = meshes.filter((mesh) => /^mesh_\d+$/.test(mesh.name))
  assert.equal(spheres.length, 49)
  assert.ok(spheres.every((mesh) => mesh.material.isMeshPhysicalMaterial === true), 'all anisotropy-grid spheres should use MeshPhysicalMaterial')
  assert.ok(spheres.every((mesh) => mesh.geometry.getAttribute('position')?.count === 1087))
  assert.ok(spheres.every((mesh) => mesh.geometry.getAttribute('normal')?.count === 1087))
  assert.ok(spheres.every((mesh) => mesh.geometry.getAttribute('tangent')?.count === 1087))
  assert.ok(spheres.every((mesh) => mesh.geometry.index?.count === 5952))

  assert.deepEqual(spheres.slice(0, 7).map((mesh) => mesh.material.anisotropy), [
    0,
    1 / 6,
    1 / 3,
    0.5,
    2 / 3,
    5 / 6,
    1,
  ])
  assert.deepEqual([0, 7, 14, 21, 28, 35, 42].map((index) => spheres[index].material.roughness), [
    0,
    1 / 6,
    1 / 3,
    0.5,
    2 / 3,
    5 / 6,
    1,
  ])
  assert.equal(spheres[48].material.anisotropy, 1)
  assert.equal(spheres[48].material.roughness, 1)

  const label = meshes.find((mesh) => mesh.name === 'Labels')
  assert.equal(label?.material.name, 'Label Mat')
  assert.equal(label.material.map.name, 'AnisotropySpheresLabels')
  assert.equal(label.material.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(label.material.map.flipY, false)
  assert.deepEqual(pngDimensions(label.material.map.image), [512, 512])

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.35))
  const light = new THREE.DirectionalLight(0xffffff, 3)
  light.position.set(0, 2, 8)
  gltf.scene.add(light)
  const camera = new THREE.OrthographicCamera(-3.8, 3.8, 7.0, -0.8, 0.01, 30)
  camera.position.set(0, 3, 10)
  camera.lookAt(0, 3, 0)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 128,
    height: 128,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.2, 'AnisotropyStrengthTest should render visible anisotropy-strength grid spheres')
})
