import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'
import { Renderer, SAMPLE_ASSET_NORMAL_TANGENT_MIRROR_TEST, SAMPLE_ASSET_NORMAL_TANGENT_TEST, SAMPLE_ASSET_SIMPLE_TEXTURE, SAMPLE_ASSET_TEXTURE_COORDINATE_TEST, SAMPLE_ASSET_TEXTURE_ENCODING_TEST, SAMPLE_ASSET_TEXTURE_LINEAR_INTERPOLATION_TEST, SAMPLE_ASSET_TEXTURE_SETTINGS_TEST, SAMPLE_ASSET_TWO_SIDED_PLANE } from './gltf.test.part-001.mjs'
import { assertTextureSampler, findFirst, loadGltfFixture, meanRegion } from './gltf.test.part-028.mjs'
test('committed Khronos glTF Sample Assets SimpleTexture fixture loads sampler state and renders mirrored texture repeats', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_SIMPLE_TEXTURE)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'Khronos SimpleTexture sample should load a mesh')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 4)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 4)
  assert.equal(mesh.geometry.index?.count, 6)

  const texture = mesh.material.map
  assert.ok(texture?.isTexture, 'SimpleTexture sample should load a base color texture')
  assert.equal(Buffer.isBuffer(texture.image), true, 'SimpleTexture external PNG should load as an encoded Buffer')
  assert.equal(texture.wrapS, THREE.MirroredRepeatWrapping)
  assert.equal(texture.wrapT, THREE.MirroredRepeatWrapping)
  assert.equal(texture.magFilter, THREE.LinearFilter)
  assert.equal(texture.minFilter, THREE.LinearMipmapLinearFilter)
  assert.equal(texture.flipY, false)

  texture.repeat.set(2, 2)
  mesh.material = new THREE.MeshBasicMaterial({ map: texture })
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.6, 'SimpleTexture sample should render a visible repeated texture')
  const topLeft = meanRegion(rgba, 96, 96, 18, 18, 38, 38)
  const topRight = meanRegion(rgba, 96, 96, 58, 18, 78, 38)
  const bottomLeft = meanRegion(rgba, 96, 96, 18, 58, 38, 78)
  const bottomRight = meanRegion(rgba, 96, 96, 58, 58, 78, 78)

  for (const [label, sample] of [
    ['top-right', topRight],
    ['bottom-left', bottomLeft],
    ['bottom-right', bottomRight],
  ]) {
    assert.ok(
      Math.abs(sample.r - topLeft.r) < 8 &&
        Math.abs(sample.g - topLeft.g) < 8 &&
        Math.abs(sample.b - topLeft.b) < 8,
      `mirrored-repeat ${label} sample should match top-left (${topLeft.r}, ${topLeft.g}, ${topLeft.b}) vs (${sample.r}, ${sample.g}, ${sample.b})`,
    )
  }

  const center = meanRegion(rgba, 96, 96, 38, 38, 58, 58)
  assert.ok(center.r > topLeft.r + 80 && center.g > topLeft.g + 80, `repeated texture center should sample brighter texels (${center.r}, ${center.g}, ${center.b})`)
})

test('committed Khronos glTF Sample Assets TextureSettingsTest fixture loads wrap modes and material sidedness', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_TEXTURE_SETTINGS_TEST)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })

  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'LabelMesh',
    'SingleSidedMesh',
    'DoubleSidedMesh',
    'TextureClampMeshS',
    'TextureRepeatMeshS',
    'BackgroundMesh',
    'TextureClampMeshT',
    'TextureRepeatMeshT',
    'TextureMirrorMeshS',
    'TextureMirrorMeshT',
  ])

  const meshByName = new Map(meshes.map((mesh) => [mesh.name, mesh]))
  assert.equal(meshByName.get('SingleSidedMesh')?.material.side, THREE.FrontSide)
  assert.equal(meshByName.get('DoubleSidedMesh')?.material.side, THREE.DoubleSide)
  assertTextureSampler(meshByName.get('SingleSidedMesh'), THREE.RepeatWrapping, THREE.RepeatWrapping)
  assertTextureSampler(meshByName.get('DoubleSidedMesh'), THREE.RepeatWrapping, THREE.RepeatWrapping)
  assertTextureSampler(meshByName.get('TextureClampMeshS'), THREE.ClampToEdgeWrapping, THREE.RepeatWrapping)
  assertTextureSampler(meshByName.get('TextureClampMeshT'), THREE.RepeatWrapping, THREE.ClampToEdgeWrapping)
  assertTextureSampler(meshByName.get('TextureRepeatMeshS'), THREE.RepeatWrapping, THREE.ClampToEdgeWrapping)
  assertTextureSampler(meshByName.get('TextureRepeatMeshT'), THREE.ClampToEdgeWrapping, THREE.RepeatWrapping)
  assertTextureSampler(meshByName.get('TextureMirrorMeshS'), THREE.MirroredRepeatWrapping, THREE.RepeatWrapping)
  assertTextureSampler(meshByName.get('TextureMirrorMeshT'), THREE.RepeatWrapping, THREE.MirroredRepeatWrapping)
  assertTextureSampler(meshByName.get('LabelMesh'), THREE.RepeatWrapping, THREE.RepeatWrapping)

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1.0))
  gltf.scene.updateMatrixWorld(true)
  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const halfExtent = Math.max(size.x, size.y) / 2 + 0.5
  const camera = new THREE.OrthographicCamera(-halfExtent, halfExtent, halfExtent, -halfExtent, 0.01, 40)
  camera.position.set(center.x, center.y, center.z + 15)
  camera.lookAt(center)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 160,
    height: 160,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.75, 'TextureSettingsTest should render visible sampler and sidedness panels')
})

test('committed Khronos glTF Sample Assets TwoSidedPlane fixture renders mapped double-sided PBR material', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_TWO_SIDED_PLANE, 'utf8'))
  assert.deepEqual(source.buffers, [{ byteLength: 300, uri: 'TwoSidedPlane.bin' }])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'TwoSidedPlane_BaseColor.png',
    'TwoSidedPlane_MetallicRoughness.png',
    'TwoSidedPlane_Normal.png',
  ])
  assert.equal(source.materials[0].doubleSided, true)
  assert.equal(source.materials[0].normalTexture.index, 2)
  assert.equal(source.materials[0].pbrMetallicRoughness.baseColorTexture.index, 0)
  assert.equal(source.materials[0].pbrMetallicRoughness.metallicRoughnessTexture.index, 1)

  const gltf = await loadGltfFixture(SAMPLE_ASSET_TWO_SIDED_PLANE)
  const mesh = gltf.scene.getObjectByName('TwoSidedPlane')
  assert.ok(mesh?.isMesh, 'TwoSidedPlane should load a named mesh')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 6)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 6)
  assert.equal(mesh.geometry.getAttribute('tangent')?.count, 6)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 6)
  assert.equal(mesh.geometry.index?.count, 6)

  const material = mesh.material
  assert.equal(material.side, THREE.DoubleSide)
  assert.equal(Buffer.isBuffer(material.map?.image), true, 'TwoSidedPlane base-color PNG should load as an encoded Buffer')
  assert.equal(material.map.name, 'TwoSidedPlane_BaseColor.png')
  assert.equal(material.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(Buffer.isBuffer(material.normalMap?.image), true, 'TwoSidedPlane normal PNG should load as an encoded Buffer')
  assert.equal(material.normalMap.name, 'TwoSidedPlane_Normal.png')
  assert.equal(material.normalMap.colorSpace, THREE.NoColorSpace)
  assert.equal(Buffer.isBuffer(material.roughnessMap?.image), true, 'TwoSidedPlane metallic-roughness PNG should load as an encoded Buffer')
  assert.equal(material.roughnessMap.name, 'TwoSidedPlane_MetallicRoughness.png')
  assert.equal(material.roughnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(material.metalnessMap, material.roughnessMap)

  const light = new THREE.DirectionalLight(0xffffff, 1.5)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1.2))
  gltf.scene.add(light)
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 20)
  const renderer = new Renderer()
  const renderRatio = (y) => {
    light.position.set(0, y, 2)
    camera.position.set(0, y, 0.2)
    camera.lookAt(0, 0, 0)
    gltf.scene.updateMatrixWorld(true)
    camera.updateMatrixWorld(true)
    return nonBackgroundRatio(renderer.render(gltf.scene, camera, {
      width: 96,
      height: 96,
      format: 'rgba',
      background: [0, 0, 0],
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), [0, 0, 0], 3)
  }

  const frontRatio = renderRatio(3)
  const backRatio = renderRatio(-3)
  assert.ok(frontRatio > 0.6, `TwoSidedPlane front side should render visibly (${frontRatio})`)
  assert.ok(backRatio > 0.6, `TwoSidedPlane back side should render visibly (${backRatio})`)
  assert.ok(Math.abs(frontRatio - backRatio) < 0.01, `TwoSidedPlane front/back coverage should match (${frontRatio} vs ${backRatio})`)
})

test('committed Khronos glTF Sample Assets TextureEncodingTest fixture preserves texture color roles', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_TEXTURE_ENCODING_TEST)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 14, 'TextureEncodingTest should load 12 spheres plus two label panels')

  const textures = []
  const addTexture = (texture) => {
    if (texture?.isTexture === true && !textures.includes(texture)) textures.push(texture)
  }
  for (const mesh of meshes) {
    addTexture(mesh.material.map)
    addTexture(mesh.material.emissiveMap)
    addTexture(mesh.material.roughnessMap)
    addTexture(mesh.material.metalnessMap)
  }

  assert.deepEqual(textures.map((texture) => texture.name), [
    '0_136_0.png',
    '0_136_0_gamma.png',
    '0_136_0_icc.png',
    '0_136_255.png',
    '0_136_255_gamma.png',
    '0_136_255_icc.png',
    'TestLabels.png',
    'SlotLabels.png',
  ])
  assert.deepEqual(textures.map((texture) => texture.colorSpace), [
    THREE.SRGBColorSpace,
    THREE.SRGBColorSpace,
    THREE.SRGBColorSpace,
    THREE.NoColorSpace,
    THREE.NoColorSpace,
    THREE.NoColorSpace,
    THREE.SRGBColorSpace,
    THREE.SRGBColorSpace,
  ])
  assert.deepEqual(textures.map((texture) => Buffer.isBuffer(texture.image)), Array.from({ length: 8 }, () => true))
  assert.deepEqual(textures.map((texture) => texture.flipY), Array.from({ length: 8 }, () => false))
  assert.equal(textures[6].wrapS, THREE.ClampToEdgeWrapping)
  assert.equal(textures[6].wrapT, THREE.ClampToEdgeWrapping)
  assert.equal(textures[7].wrapS, THREE.RepeatWrapping)
  assert.equal(textures[7].wrapT, THREE.RepeatWrapping)

  for (const index of [1, 2, 3]) {
    assert.equal(meshes[index].material.map.colorSpace, THREE.SRGBColorSpace, `base color texture ${index} should decode as sRGB`)
  }
  for (const index of [5, 6, 7]) {
    assert.equal(meshes[index].material.emissiveMap.colorSpace, THREE.SRGBColorSpace, `emissive texture ${index} should decode as sRGB`)
  }
  for (const index of [9, 10, 11]) {
    assert.equal(meshes[index].material.roughnessMap, meshes[index].material.metalnessMap)
    assert.equal(meshes[index].material.metalnessMap.colorSpace, THREE.NoColorSpace, `metallic-roughness texture ${index} should stay non-color`)
  }
  assert.equal(meshes[12].material.alphaTest, 0.5)
  assert.equal(meshes[12].material.side, THREE.DoubleSide)
  assert.equal(meshes[13].material.alphaTest, 0.5)
  assert.equal(meshes[13].material.side, THREE.DoubleSide)

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.7))
  const light = new THREE.DirectionalLight(0xffffff, 1.3)
  light.position.set(0, 4, 8)
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)

  const camera = new THREE.OrthographicCamera(-4.5, 8.5, 4.5, -5.5, 0.01, 50)
  camera.position.set(1.5, -0.5, 18)
  camera.lookAt(1.5, -0.5, 0)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 208,
    height: 160,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.25, 'TextureEncodingTest should render visible texture encoding panels')
  const mean = meanRgba(rgba)
  assert.ok(mean.g > mean.r + 8 && mean.g > mean.b + 8, `TextureEncodingTest render should preserve the green sample hue (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets TextureLinearInterpolationTest fixture loads linear sampler filters', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_TEXTURE_LINEAR_INTERPOLATION_TEST, 'utf8'))
  assert.deepEqual(source.samplers, [{ minFilter: 9729, magFilter: 9729 }])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_TEXTURE_LINEAR_INTERPOLATION_TEST)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 3, 'TextureLinearInterpolationTest should load two spheres and one label plane')

  const [solidSphere, texturedSphere, labels] = meshes
  assert.equal(solidSphere.geometry.getAttribute('position')?.count, 205)
  assert.equal(solidSphere.geometry.getAttribute('normal')?.count, 205)
  assert.equal(solidSphere.geometry.getAttribute('uv'), undefined)
  assert.equal(solidSphere.geometry.index?.count, 960)
  assert.deepEqual(solidSphere.material.emissive.toArray(), [0, 0.5, 0])

  assert.equal(texturedSphere.geometry.getAttribute('position')?.count, 205)
  assert.equal(texturedSphere.geometry.getAttribute('normal')?.count, 205)
  assert.equal(texturedSphere.geometry.getAttribute('uv')?.count, 205)
  assert.equal(texturedSphere.geometry.index?.count, 960)
  const texture = texturedSphere.material.emissiveMap
  assert.equal(texture?.name, '0_0_0-0_255_0.png')
  assert.equal(Buffer.isBuffer(texture.image), true, 'TextureLinearInterpolationTest tiny PNG should load as an encoded Buffer')
  assert.equal(texture.magFilter, THREE.LinearFilter)
  assert.equal(texture.minFilter, THREE.LinearFilter)
  assert.equal(texture.wrapS, THREE.RepeatWrapping)
  assert.equal(texture.wrapT, THREE.RepeatWrapping)
  assert.equal(texture.colorSpace, THREE.SRGBColorSpace)
  assert.equal(texture.flipY, false)

  assert.equal(labels.geometry.getAttribute('position')?.count, 4)
  assert.equal(labels.geometry.getAttribute('uv')?.count, 4)
  assert.equal(labels.geometry.index?.count, 6)
  assert.equal(labels.material.alphaTest, 0.5)
  assert.equal(labels.material.side, THREE.DoubleSide)
  assert.equal(Buffer.isBuffer(labels.material.map?.image), true, 'TextureLinearInterpolationTest labels PNG should load as an encoded Buffer')

  const camera = new THREE.OrthographicCamera(-3.6, 3.6, 1.8, -2.3, 0.01, 10)
  camera.position.set(0, -0.35, 4)
  camera.lookAt(0, -0.35, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 144,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.2, 'TextureLinearInterpolationTest should render visible green spheres')
  const left = meanRegion(rgba, 144, 96, 28, 50, 58, 78)
  const right = meanRegion(rgba, 144, 96, 86, 50, 116, 78)
  assert.ok(left.g > left.r + 80 && left.g > left.b + 80, `solid green sphere should render visibly green (${left.r}, ${left.g}, ${left.b})`)
  assert.ok(right.g > right.r + 50 && right.g > right.b + 50, `linear-sampled texture sphere should render visibly green (${right.r}, ${right.g}, ${right.b})`)
})

test('committed Khronos glTF Sample Assets NormalTangentTest fixture loads normal and ORM texture maps', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_NORMAL_TANGENT_TEST)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'NormalTangentTest should load a mesh')
  assert.equal(mesh.name, 'NormalTangentTest_low')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 3983)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 3983)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 3983)
  assert.equal(mesh.geometry.getAttribute('tangent'), undefined)
  assert.equal(mesh.geometry.index?.count, 23322)

  const material = mesh.material
  assert.equal(material.isMeshStandardMaterial, true)
  assert.equal(material.side, THREE.DoubleSide)
  assert.ok(Buffer.isBuffer(material.map?.image), 'NormalTangentTest base-color PNG should load as an encoded Buffer')
  assert.ok(Buffer.isBuffer(material.normalMap?.image), 'NormalTangentTest normal PNG should load as an encoded Buffer')
  assert.ok(Buffer.isBuffer(material.aoMap?.image), 'NormalTangentTest packed ORM PNG should load as an encoded Buffer')
  assert.equal(material.roughnessMap, material.aoMap)
  assert.equal(material.metalnessMap, material.aoMap)
  assert.equal(material.map.flipY, false)
  assert.equal(material.normalMap.flipY, false)
  assert.deepEqual(material.normalScale.toArray(), [1, -1])

  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 20)
  camera.position.set(0, 0, 4)
  camera.lookAt(0, -0.1, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 2.0)
  light.position.set(1, 2, 4)
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
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.35, 'NormalTangentTest should render visible textured geometry')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 45 && mean.g > 45 && mean.b > 40, `NormalTangentTest render should include textured material color (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets NormalTangentMirrorTest fixture loads mirrored tangent attributes', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_NORMAL_TANGENT_MIRROR_TEST)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'NormalTangentMirrorTest should load a mesh')
  assert.equal(mesh.name, 'NormalTangentTest_low')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 2770)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 2770)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 2770)
  assert.equal(mesh.geometry.index?.count, 15720)

  const tangent = mesh.geometry.getAttribute('tangent')
  assert.equal(tangent?.count, 2770)
  assert.equal(tangent.itemSize, 4)
  let positiveHandedness = 0
  let negativeHandedness = 0
  for (let index = 0; index < tangent.count; index += 1) {
    const handedness = tangent.getW(index)
    if (handedness > 0) positiveHandedness += 1
    if (handedness < 0) negativeHandedness += 1
  }
  assert.ok(positiveHandedness > 0, 'NormalTangentMirrorTest should include positive tangent handedness')
  assert.ok(negativeHandedness > 0, 'NormalTangentMirrorTest should include mirrored negative tangent handedness')

  const material = mesh.material
  assert.equal(material.isMeshStandardMaterial, true)
  assert.equal(material.side, THREE.DoubleSide)
  assert.ok(Buffer.isBuffer(material.map?.image), 'NormalTangentMirrorTest base-color PNG should load as an encoded Buffer')
  assert.ok(Buffer.isBuffer(material.normalMap?.image), 'NormalTangentMirrorTest normal PNG should load as an encoded Buffer')
  assert.ok(Buffer.isBuffer(material.aoMap?.image), 'NormalTangentMirrorTest packed ORM PNG should load as an encoded Buffer')
  assert.equal(material.roughnessMap, material.aoMap)
  assert.equal(material.metalnessMap, material.aoMap)
  assert.equal(material.map.flipY, false)
  assert.equal(material.normalMap.flipY, false)
  assert.deepEqual(material.normalScale.toArray(), [1, 1])

  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 20)
  camera.position.set(0, 0, 4)
  camera.lookAt(0, -0.05, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 2.0)
  light.position.set(1, 2, 4)
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
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.45, 'NormalTangentMirrorTest should render visible mirrored tangent geometry')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 50 && mean.g > 50 && mean.b > 45, `NormalTangentMirrorTest render should include textured material color (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets TextureCoordinateTest fixture renders external PNG UV quadrants', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_TEXTURE_COORDINATE_TEST)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 5, 'Khronos TextureCoordinateTest sample should load four textured planes plus a back plane')
  assert.equal(meshes.filter((mesh) => mesh.material.map?.isTexture === true).length, 4)
  assert.ok(
    meshes.filter((mesh) => mesh.material.map?.isTexture === true).every((mesh) => Buffer.isBuffer(mesh.material.map.image)),
    'external PNG textures should be exposed as encoded Buffers',
  )

  const camera = new THREE.OrthographicCamera(-1.45, 1.45, 1.45, -1.45, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1.0))
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.4, 'TextureCoordinateTest sample should render visible textured planes')
  const topLeft = meanRegion(rgba, 96, 96, 18, 18, 38, 38)
  const topRight = meanRegion(rgba, 96, 96, 58, 18, 78, 38)
  const bottomLeft = meanRegion(rgba, 96, 96, 18, 58, 38, 78)
  const bottomRight = meanRegion(rgba, 96, 96, 58, 58, 78, 78)
  assert.ok(topLeft.r > 130 && topLeft.g > 120 && topLeft.b < 50, `top-left UV quadrant should sample yellow texels (${topLeft.r}, ${topLeft.g}, ${topLeft.b})`)
  assert.ok(topRight.r > topRight.g + 140 && topRight.r > topRight.b + 140, `top-right UV quadrant should sample red texels (${topRight.r}, ${topRight.g}, ${topRight.b})`)
  assert.ok(bottomLeft.b > bottomLeft.r + 120 && bottomLeft.b > bottomLeft.g + 120, `bottom-left UV quadrant should sample blue texels (${bottomLeft.r}, ${bottomLeft.g}, ${bottomLeft.b})`)
  assert.ok(bottomRight.g > bottomRight.r + 100 && bottomRight.g > bottomRight.b + 100, `bottom-right UV quadrant should sample green texels (${bottomRight.r}, ${bottomRight.g}, ${bottomRight.b})`)
})
