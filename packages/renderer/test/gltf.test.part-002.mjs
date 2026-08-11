import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'
import { Renderer, SAMPLE_ASSET_ANTIQUE_CAMERA, SAMPLE_ASSET_BOX_TEXTURED, SAMPLE_ASSET_BOX_TEXTURED_NPOT, SAMPLE_ASSET_BOX_WITH_SPACES, SAMPLE_ASSET_CORSET, SAMPLE_ASSET_SCIFI_HELMET, SAMPLE_ASSET_UNICODE_TEST } from './gltf.test.part-001.mjs'
import { findFirst, loadGltfFixture, meanRegion, pngDimensions } from './gltf.test.part-028.mjs'
test('committed Khronos glTF Sample Assets Box With Spaces fixture resolves external paths with spaces', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_BOX_WITH_SPACES, 'utf8'))
  assert.equal(source.buffers[0].uri, 'Box With Spaces.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'Normal%20Map.png',
    'glTF%20Logo%20With%20Spaces.png',
    'Roughness%20Metallic.png',
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_BOX_WITH_SPACES)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'Khronos Box With Spaces sample should load a mesh')
  assert.equal(mesh.name, 'Cube')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 24)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 24)
  assert.equal(mesh.geometry.index?.count, 36)
  assert.equal(mesh.material.name, 'Material')

  const { map, normalMap, metalnessMap, roughnessMap } = mesh.material
  assert.equal(Buffer.isBuffer(map?.image), true, 'space-containing base color PNG path should load as an encoded Buffer')
  assert.equal(Buffer.isBuffer(normalMap?.image), true, 'space-containing normal PNG path should load as an encoded Buffer')
  assert.equal(Buffer.isBuffer(metalnessMap?.image), true, 'space-containing metallic-roughness PNG path should load as an encoded Buffer')
  assert.equal(Buffer.isBuffer(roughnessMap?.image), true, 'space-containing roughness PNG path should load as an encoded Buffer')
  assert.equal(map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(normalMap.colorSpace, THREE.NoColorSpace)
  assert.equal(metalnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(roughnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(map.flipY, false)
  assert.equal(normalMap.flipY, false)
  assert.equal(metalnessMap.flipY, false)
  assert.equal(roughnessMap.flipY, false)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 20)
  camera.position.set(3, 2.1, 4.5)
  camera.lookAt(0, 0, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 1.2)
  light.position.set(2, 4, 5)
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.07, 'Box With Spaces sample should render visible textured pixels')
  const center = meanRegion(rgba, 96, 96, 34, 34, 62, 62)
  assert.ok(center.r > 5 || center.g > 5 || center.b > 5, `Box With Spaces sample should render non-black center pixels (${center.r}, ${center.g}, ${center.b})`)
})

test('committed Khronos glTF Sample Assets Unicode❤♻Test fixture resolves Unicode external paths', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_UNICODE_TEST, 'utf8'))
  assert.deepEqual(source.buffers, [{ uri: 'Unicode❤♻Binary.bin', byteLength: 152 }])
  assert.deepEqual(source.images.map((image) => image.uri), ['Unicode❤♻Texture.png'])
  assert.equal(source.meshes[0].name, 'Unicode❤♻Mesh')
  assert.equal(source.materials[0].name, 'Unicode❤♻Material')

  const gltf = await loadGltfFixture(SAMPLE_ASSET_UNICODE_TEST)
  const mesh = gltf.scene.getObjectByName('Unicode❤♻Mesh')
  assert.ok(mesh?.isMesh, 'Unicode sample should load its Unicode-named mesh')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 4)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 4)
  assert.equal(mesh.geometry.index?.count, 6)
  assert.equal(mesh.material.name, 'Unicode❤♻Material')
  assert.equal(mesh.material.map?.name, 'Unicode❤♻Texture.png')
  assert.equal(Buffer.isBuffer(mesh.material.map?.image), true, 'Unicode texture path should load as an encoded Buffer')
  assert.deepEqual(pngDimensions(mesh.material.map.image), [256, 256])
  assert.equal(mesh.material.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(mesh.material.map.flipY, false)

  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 1.2, -1.2, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1))
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 64,
    height: 64,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.12, 'Unicode sample should render visible textured pixels')
  const center = meanRegion(rgba, 64, 64, 24, 24, 40, 40)
  assert.ok(center.b > 10 && center.g > 5 && center.b > center.r + 10, `Unicode texture center should render blue-green texels (${center.r}, ${center.g}, ${center.b})`)
})

test('committed Khronos glTF Sample Assets BoxTextured fixture loads POT texture sampler state', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_BOX_TEXTURED, 'utf8'))
  assert.deepEqual(source.buffers, [
    { byteLength: 840, uri: 'BoxTextured0.bin' },
  ])
  assert.equal(source.images[0].uri, 'CesiumLogoFlat.png')
  assert.deepEqual(source.samplers, [
    {
      magFilter: 9729,
      minFilter: 9986,
      wrapS: 10497,
      wrapT: 10497,
    },
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_BOX_TEXTURED)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'BoxTextured should load a mesh')
  assert.equal(mesh.name, 'Mesh')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 24)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 24)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 24)
  assert.equal(mesh.geometry.index?.count, 36)
  assert.equal(mesh.material.isMeshStandardMaterial, true)
  assert.equal(mesh.material.name, 'Texture')
  assert.equal(mesh.material.metalness, 0)

  const texture = mesh.material.map
  assert.ok(texture?.isTexture, 'BoxTextured should load a base color texture')
  assert.equal(texture.name, 'CesiumLogoFlat.png')
  assert.equal(Buffer.isBuffer(texture.image), true, 'BoxTextured PNG should load as an encoded Buffer')
  assert.deepEqual(pngDimensions(texture.image), [256, 256])
  assert.equal(texture.wrapS, THREE.RepeatWrapping)
  assert.equal(texture.wrapT, THREE.RepeatWrapping)
  assert.equal(texture.magFilter, THREE.LinearFilter)
  assert.equal(texture.minFilter, THREE.NearestMipmapLinearFilter)
  assert.equal(texture.colorSpace, THREE.SRGBColorSpace)
  assert.equal(texture.flipY, false)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 10)
  camera.position.set(1.3, 1.1, 2.2)
  camera.lookAt(0, 0, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 1.4)
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.25, 'BoxTextured should render visible textured pixels')
  const center = meanRegion(rgba, 96, 96, 34, 34, 62, 62)
  assert.ok(center.r > 70 && center.g > 90 && center.b > 100, `BoxTextured should render the repeated logo texture (${center.r}, ${center.g}, ${center.b})`)
})

test('committed Khronos glTF Sample Assets BoxTexturedNonPowerOfTwo fixture loads NPOT texture sampler state', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_BOX_TEXTURED_NPOT, 'utf8'))
  assert.equal(source.buffers[0].uri, 'BoxTextured0.bin')
  assert.equal(source.images[0].uri, 'CesiumLogoFlat.png')
  assert.deepEqual(source.samplers, [
    {
      magFilter: 9729,
      minFilter: 9986,
      wrapS: 10497,
      wrapT: 10497,
    },
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_BOX_TEXTURED_NPOT)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'BoxTexturedNonPowerOfTwo should load a mesh')
  assert.equal(mesh.name, 'Mesh')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 24)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 24)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 24)
  assert.equal(mesh.geometry.index?.count, 36)
  assert.equal(mesh.material.isMeshStandardMaterial, true)
  assert.equal(mesh.material.name, 'Texture')
  assert.equal(mesh.material.metalness, 0)

  const texture = mesh.material.map
  assert.ok(texture?.isTexture, 'BoxTexturedNonPowerOfTwo should load a base color texture')
  assert.equal(texture.name, 'CesiumLogoFlat.png')
  assert.equal(Buffer.isBuffer(texture.image), true, 'BoxTexturedNonPowerOfTwo NPOT PNG should load as an encoded Buffer')
  assert.deepEqual(pngDimensions(texture.image), [211, 211])
  assert.equal(texture.wrapS, THREE.RepeatWrapping)
  assert.equal(texture.wrapT, THREE.RepeatWrapping)
  assert.equal(texture.magFilter, THREE.LinearFilter)
  assert.equal(texture.minFilter, THREE.NearestMipmapLinearFilter)
  assert.equal(texture.colorSpace, THREE.SRGBColorSpace)
  assert.equal(texture.flipY, false)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 10)
  camera.position.set(1.3, 1.1, 2.2)
  camera.lookAt(0, 0, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 1.4)
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.25, 'BoxTexturedNonPowerOfTwo should render visible textured pixels')
  const center = meanRegion(rgba, 96, 96, 34, 34, 62, 62)
  assert.ok(center.r > 80 && center.g > 100 && center.b > 110, `BoxTexturedNonPowerOfTwo should render the NPOT logo texture (${center.r}, ${center.g}, ${center.b})`)
})

test('committed Khronos glTF Sample Assets AntiqueCamera fixture loads multi-mesh PBR texture sets', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_ANTIQUE_CAMERA, 'utf8'))
  assert.deepEqual(source.buffers, [
    { byteLength: 798092, uri: 'AntiqueCamera.bin' },
  ])
  assert.deepEqual(source.images.map((image) => [image.name, image.uri]), [
    ['camera_camera_Normal', 'camera_camera_Normal.png'],
    ['camera_camera_BaseColor', 'camera_camera_BaseColor.png'],
    ['camera_tripod_BaseColor', 'camera_tripod_BaseColor.png'],
    ['camera_tripod_Normal', 'camera_tripod_Normal.png'],
    ['camera_camera_Roughness', 'camera_camera_Roughness.png'],
    ['camera_tripod_Roughness', 'camera_tripod_Roughness.png'],
  ])
  assert.deepEqual(
    source.materials.map((material) => [
      material.name,
      material.pbrMetallicRoughness.baseColorTexture.index,
      material.pbrMetallicRoughness.metallicRoughnessTexture.index,
      material.normalTexture.index,
    ]),
    [
      ['camera', 1, 4, 0],
      ['tripod', 2, 5, 3],
    ],
  )

  const gltf = await loadGltfFixture(SAMPLE_ASSET_ANTIQUE_CAMERA)
  const cameraMesh = gltf.scene.getObjectByName('camera')
  const tripodMesh = gltf.scene.getObjectByName('tripod')
  assert.ok(cameraMesh?.isMesh, 'AntiqueCamera should load the camera mesh')
  assert.ok(tripodMesh?.isMesh, 'AntiqueCamera should load the tripod mesh')

  assert.equal(cameraMesh.geometry.getAttribute('position')?.count, 14668)
  assert.equal(cameraMesh.geometry.getAttribute('normal')?.count, 14668)
  assert.equal(cameraMesh.geometry.getAttribute('uv')?.count, 14668)
  assert.equal(cameraMesh.geometry.index?.count, 41838)
  assert.equal(tripodMesh.geometry.getAttribute('position')?.count, 6510)
  assert.equal(tripodMesh.geometry.getAttribute('normal')?.count, 6510)
  assert.equal(tripodMesh.geometry.getAttribute('uv')?.count, 6510)
  assert.equal(tripodMesh.geometry.index?.count, 18360)

  const assertAntiqueTexture = (texture, name, colorSpace) => {
    assert.ok(texture?.isTexture, `${name} should load as a texture`)
    assert.equal(texture.name, name)
    assert.equal(Buffer.isBuffer(texture.image), true, `${name} should load as an encoded Buffer`)
    assert.deepEqual(pngDimensions(texture.image), [2048, 2048])
    assert.equal(texture.colorSpace, colorSpace)
    assert.equal(texture.flipY, false)
  }

  assert.equal(cameraMesh.material.isMeshStandardMaterial, true)
  assert.equal(cameraMesh.material.name, 'camera')
  assertAntiqueTexture(cameraMesh.material.map, 'camera_camera_BaseColor', THREE.SRGBColorSpace)
  assertAntiqueTexture(cameraMesh.material.roughnessMap, 'camera_camera_Roughness', THREE.NoColorSpace)
  assert.equal(cameraMesh.material.metalnessMap, cameraMesh.material.roughnessMap)
  assertAntiqueTexture(cameraMesh.material.normalMap, 'camera_camera_Normal', THREE.NoColorSpace)

  assert.equal(tripodMesh.material.isMeshStandardMaterial, true)
  assert.equal(tripodMesh.material.name, 'tripod')
  assertAntiqueTexture(tripodMesh.material.map, 'camera_tripod_BaseColor', THREE.SRGBColorSpace)
  assertAntiqueTexture(tripodMesh.material.roughnessMap, 'camera_tripod_Roughness', THREE.NoColorSpace)
  assert.equal(tripodMesh.material.metalnessMap, tripodMesh.material.roughnessMap)
  assertAntiqueTexture(tripodMesh.material.normalMap, 'camera_tripod_Normal', THREE.NoColorSpace)

  gltf.scene.updateMatrixWorld(true)
  const box = new THREE.Box3().setFromObject(gltf.scene)
  const center = box.getCenter(new THREE.Vector3())
  const renderCamera = new THREE.PerspectiveCamera(35, 1, 0.01, 100)
  renderCamera.position.copy(center).add(new THREE.Vector3(0.8, 0.45, 1).normalize().multiplyScalar(9))
  renderCamera.lookAt(center)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.9))
  const light = new THREE.DirectionalLight(0xffffff, 1.6)
  light.position.copy(center).add(new THREE.Vector3(4, 6, 5))
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)
  renderCamera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, renderCamera, {
    width: 128,
    height: 128,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.15, 'AntiqueCamera should render visible textured geometry')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > mean.b + 1, `AntiqueCamera should render warm textured output (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets Corset fixture loads tangent-space ORM texture set', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_CORSET, 'utf8'))
  assert.deepEqual(source.buffers, [
    { uri: 'Corset.bin', byteLength: 662184 },
  ])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'Corset_baseColor.png',
    'Corset_occlusionRoughnessMetallic.png',
    'Corset_normal.png',
  ])
  assert.equal(source.meshes[0].name, 'pCube49')
  assert.deepEqual(source.meshes[0].primitives[0].attributes, {
    TEXCOORD_0: 0,
    NORMAL: 1,
    TANGENT: 2,
    POSITION: 3,
  })
  assert.deepEqual(source.materials[0], {
    pbrMetallicRoughness: {
      baseColorTexture: { index: 0 },
      metallicRoughnessTexture: { index: 1 },
    },
    normalTexture: { index: 2 },
    occlusionTexture: { index: 1 },
    name: 'Corset_O',
  })

  const gltf = await loadGltfFixture(SAMPLE_ASSET_CORSET)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'Corset should load a mesh')
  assert.equal(mesh.name, 'Corset')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 11505)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 11505)
  assert.equal(mesh.geometry.getAttribute('tangent')?.count, 11505)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 11505)
  assert.equal(mesh.geometry.index?.count, 54972)

  const material = mesh.material
  assert.equal(material.isMeshStandardMaterial, true)
  assert.equal(material.name, 'Corset_O')
  assert.deepEqual(material.color.toArray(), [1, 1, 1])
  assert.equal(material.metalness, 1)
  assert.equal(material.roughness, 1)

  const assertCorsetTexture = (texture, name, colorSpace) => {
    assert.ok(texture?.isTexture, `${name} should load as a texture`)
    assert.equal(texture.name, name)
    assert.equal(Buffer.isBuffer(texture.image), true, `${name} should load as an encoded Buffer`)
    assert.deepEqual(pngDimensions(texture.image), [2048, 2048])
    assert.equal(texture.colorSpace, colorSpace)
    assert.equal(texture.flipY, false)
  }

  assertCorsetTexture(material.map, 'Corset_baseColor.png', THREE.SRGBColorSpace)
  assertCorsetTexture(material.roughnessMap, 'Corset_occlusionRoughnessMetallic.png', THREE.NoColorSpace)
  assert.equal(material.metalnessMap, material.roughnessMap, 'Corset should reuse the ORM map for metalness')
  assert.equal(material.aoMap, material.roughnessMap, 'Corset should reuse the ORM map for ambient occlusion')
  assertCorsetTexture(material.normalMap, 'Corset_normal.png', THREE.NoColorSpace)

  gltf.scene.updateMatrixWorld(true)
  const box = new THREE.Box3().setFromObject(gltf.scene)
  const center = box.getCenter(new THREE.Vector3())
  const renderCamera = new THREE.PerspectiveCamera(35, 1, 0.001, 10)
  renderCamera.position.copy(center).add(new THREE.Vector3(0.8, 0.35, 1).normalize().multiplyScalar(0.09))
  renderCamera.lookAt(center)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 1.8)
  light.position.copy(center).add(new THREE.Vector3(0.4, 0.6, 0.7))
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)
  renderCamera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, renderCamera, {
    width: 128,
    height: 128,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.25, 'Corset should render visible textured geometry')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > mean.g && mean.g > mean.b, `Corset texture should render warm fabric colors (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets SciFiHelmet fixture loads separate AO and PBR texture maps', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_SCIFI_HELMET, 'utf8'))
  assert.deepEqual(source.buffers, [
    { byteLength: 3643848, uri: 'SciFiHelmet.bin' },
  ])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'SciFiHelmet_BaseColor.png',
    'SciFiHelmet_MetallicRoughness.png',
    'SciFiHelmet_Normal.png',
    'SciFiHelmet_AmbientOcclusion.png',
  ])
  assert.equal(source.meshes[0].name, 'SciFiHelmet')
  assert.deepEqual(source.meshes[0].primitives[0].attributes, {
    NORMAL: 2,
    POSITION: 1,
    TANGENT: 3,
    TEXCOORD_0: 4,
  })
  assert.deepEqual(source.materials[0], {
    name: 'SciFiHelmet',
    normalTexture: { index: 2 },
    occlusionTexture: { index: 3 },
    pbrMetallicRoughness: {
      baseColorTexture: { index: 0 },
      metallicRoughnessTexture: { index: 1 },
    },
  })

  const gltf = await loadGltfFixture(SAMPLE_ASSET_SCIFI_HELMET)
  assert.deepEqual(gltf.scene.children.map((child) => child.name), ['Camera', 'SciFiHelmet'])
  const mesh = gltf.scene.getObjectByName('SciFiHelmet')
  assert.ok(mesh?.isMesh, 'SciFiHelmet should load its mesh')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 70074)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 70074)
  assert.equal(mesh.geometry.getAttribute('tangent')?.count, 70074)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 70074)
  assert.equal(mesh.geometry.index?.count, 70074)

  const material = mesh.material
  assert.equal(material.isMeshStandardMaterial, true)
  assert.equal(material.name, 'SciFiHelmet')
  assert.deepEqual(material.color.toArray(), [1, 1, 1])
  assert.equal(material.metalness, 1)
  assert.equal(material.roughness, 1)

  const assertSciFiTexture = (texture, name, colorSpace) => {
    assert.ok(texture?.isTexture, `${name} should load as a texture`)
    assert.equal(texture.name, name)
    assert.equal(Buffer.isBuffer(texture.image), true, `${name} should load as an encoded Buffer`)
    assert.deepEqual(pngDimensions(texture.image), [2048, 2048])
    assert.equal(texture.colorSpace, colorSpace)
    assert.equal(texture.flipY, false)
  }

  assertSciFiTexture(material.map, 'SciFiHelmet_BaseColor.png', THREE.SRGBColorSpace)
  assertSciFiTexture(material.roughnessMap, 'SciFiHelmet_MetallicRoughness.png', THREE.NoColorSpace)
  assert.equal(material.metalnessMap, material.roughnessMap, 'SciFiHelmet should reuse the metallic-roughness map for metalness')
  assertSciFiTexture(material.normalMap, 'SciFiHelmet_Normal.png', THREE.NoColorSpace)
  assertSciFiTexture(material.aoMap, 'SciFiHelmet_AmbientOcclusion.png', THREE.NoColorSpace)
  assert.notEqual(material.aoMap, material.roughnessMap, 'SciFiHelmet should keep AO separate from metallic-roughness')

  gltf.scene.updateMatrixWorld(true)
  const box = new THREE.Box3().setFromObject(mesh)
  const center = box.getCenter(new THREE.Vector3())
  const renderCamera = new THREE.PerspectiveCamera(35, 1, 0.01, 50)
  renderCamera.position.copy(center).add(new THREE.Vector3(0.8, 0.45, 1).normalize().multiplyScalar(4))
  renderCamera.lookAt(center)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.75))
  const light = new THREE.DirectionalLight(0xffffff, 1.8)
  light.position.copy(center).add(new THREE.Vector3(3, 4, 5))
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)
  renderCamera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, renderCamera, {
    width: 128,
    height: 128,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.32, 'SciFiHelmet should render visible textured geometry')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > mean.b && mean.g > mean.b, `SciFiHelmet texture should render neutral metal colors (${mean.r}, ${mean.g}, ${mean.b})`)
})
