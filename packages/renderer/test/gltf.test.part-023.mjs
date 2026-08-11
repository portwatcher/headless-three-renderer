import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'
import { Renderer, SAMPLE_ASSET_MESHOPT_CUBE_TEST, SAMPLE_ASSET_MESH_PRIMITIVE_MODES, SAMPLE_ASSET_METAL_ROUGH_SPHERES, SAMPLE_ASSET_METAL_ROUGH_SPHERES_NO_TEXTURES, SAMPLE_ASSET_MULTI_UV_TEST, SAMPLE_ASSET_TEXTURE_TRANSFORM_MULTI_TEST, SAMPLE_ASSET_TEXTURE_TRANSFORM_TEST } from './gltf.test.part-001.mjs'
import { assertVectorClose, findFirst, loadGltfFixture, meanRegion, pngDimensions, renderSingleObjectRatio } from './gltf.test.part-028.mjs'
test('committed Khronos glTF Sample Assets MultiUVTest fixture loads primary and secondary texture UVs', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_MULTI_UV_TEST)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'Khronos MultiUVTest sample should load a mesh')
  assert.equal(mesh.name, 'Cube')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 24)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 24)
  assert.equal(mesh.geometry.getAttribute('tangent')?.count, 24)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 24)
  assert.equal(mesh.geometry.getAttribute('uv1')?.count, 24)
  assert.equal(mesh.geometry.index?.count, 36)

  const { material } = mesh
  assert.equal(material.isMeshStandardMaterial, true)
  assert.equal(material.name, 'Material')
  assert.deepEqual(material.emissive.toArray(), [1, 1, 1])
  assert.equal(material.emissiveIntensity, 1)

  const { map, emissiveMap } = material
  assert.ok(map?.isTexture, 'MultiUVTest sample should load a base color texture')
  assert.ok(emissiveMap?.isTexture, 'MultiUVTest sample should load an emissive texture')
  assert.equal(map.name, 'uv0.png')
  assert.equal(emissiveMap.name, 'uv1.png')
  assert.equal(Buffer.isBuffer(map.image), true, 'MultiUVTest base color PNG should load as an encoded Buffer')
  assert.equal(Buffer.isBuffer(emissiveMap.image), true, 'MultiUVTest emissive PNG should load as an encoded Buffer')
  assert.equal(map.channel, 0)
  assert.equal(emissiveMap.channel, 1)
  assert.equal(map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(emissiveMap.colorSpace, THREE.SRGBColorSpace)
  assert.equal(map.flipY, false)
  assert.equal(emissiveMap.flipY, false)

  const camera = gltf.cameras[0]
  assert.ok(camera?.isPerspectiveCamera, 'MultiUVTest sample should load its camera')
  camera.aspect = 1
  camera.updateProjectionMatrix()
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.7))
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 128,
    height: 128,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.18, 'MultiUVTest should render visible multi-UV textured geometry')
  const center = meanRegion(rgba, 128, 128, 48, 48, 80, 80)
  assert.ok(center.r > 80 && center.g > 90 && center.b > 100, `MultiUVTest center should include textured/emissive color (${center.r}, ${center.g}, ${center.b})`)
  const lowerLeft = meanRegion(rgba, 128, 128, 20, 80, 48, 108)
  assert.ok(lowerLeft.r > lowerLeft.b + 20 && lowerLeft.g > lowerLeft.b + 20, `MultiUVTest secondary UV sample should contribute warm emissive texels (${lowerLeft.r}, ${lowerLeft.g}, ${lowerLeft.b})`)
})

test('committed Khronos glTF Sample Assets TextureTransformTest fixture loads KHR_texture_transform', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_TEXTURE_TRANSFORM_TEST)
  const offsetU = gltf.scene.getObjectByName('Offset_U')
  const offsetV = gltf.scene.getObjectByName('Offset_V')
  const offsetUv = gltf.scene.getObjectByName('Offset_UV')
  const rotation = gltf.scene.getObjectByName('Rotation')
  const scale = gltf.scene.getObjectByName('Scale')
  const all = gltf.scene.getObjectByName('All')
  assert.ok(offsetU?.isMesh, 'TextureTransformTest should load Offset_U mesh')
  assert.ok(offsetV?.isMesh, 'TextureTransformTest should load Offset_V mesh')
  assert.ok(offsetUv?.isMesh, 'TextureTransformTest should load Offset_UV mesh')
  assert.ok(rotation?.isMesh, 'TextureTransformTest should load Rotation mesh')
  assert.ok(scale?.isMesh, 'TextureTransformTest should load Scale mesh')
  assert.ok(all?.isMesh, 'TextureTransformTest should load All mesh')

  assert.equal(Buffer.isBuffer(offsetU.material.map.image), true)
  assert.deepEqual(offsetU.material.map.offset.toArray(), [0.5, 0])
  assert.deepEqual(offsetV.material.map.offset.toArray(), [0, 0.5])
  assert.deepEqual(offsetUv.material.map.offset.toArray(), [0.5, 0.5])
  assert.ok(Math.abs(rotation.material.map.rotation - 0.39269908169872414) < 1e-12)
  assert.deepEqual(scale.material.map.repeat.toArray(), [1.5, 1.5])
  assert.deepEqual(all.material.map.offset.toArray(), [-0.2, -0.1])
  assert.deepEqual(all.material.map.repeat.toArray(), [1.5, 1.5])
  assert.ok(Math.abs(all.material.map.rotation - 0.3) < 1e-12)

  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 12, 'TextureTransformTest should load transformed samples and reference badges')
  assert.ok(
    meshes.every((mesh) => Buffer.isBuffer(mesh.material.map?.image)),
    'TextureTransformTest external PNG textures should load as encoded Buffers',
  )

  const camera = new THREE.OrthographicCamera(-1.8, 1.8, 1.2, -1.2, 0.01, 20)
  camera.position.set(0, 0, 10)
  camera.lookAt(0, 0, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1.0))
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 144,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.6, 'TextureTransformTest should render visible transformed texture samples')
  const topLeft = meanRegion(rgba, 144, 96, 18, 16, 38, 36)
  const topCenter = meanRegion(rgba, 144, 96, 62, 16, 82, 36)
  const topRight = meanRegion(rgba, 144, 96, 106, 16, 126, 36)
  assert.ok(topLeft.g > topLeft.r + 60 && topLeft.g > topLeft.b + 60, `offset-U sample should expose green-dominant texels (${topLeft.r}, ${topLeft.g}, ${topLeft.b})`)
  assert.ok(topCenter.b > topCenter.r + 80 && topCenter.b > topCenter.g + 80, `offset-V sample should expose blue-dominant texels (${topCenter.r}, ${topCenter.g}, ${topCenter.b})`)
  assert.ok(topRight.g > topRight.r + 60 && topRight.b > topRight.r + 60, `offset-UV sample should expose cyan texels (${topRight.r}, ${topRight.g}, ${topRight.b})`)
})

test('committed Khronos glTF Sample Assets TextureTransformMultiTest fixture loads KHR_texture_transform across texture slots', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_TEXTURE_TRANSFORM_MULTI_TEST, 'utf8'))
  assert.deepEqual(source.extensionsUsed, [
    'KHR_materials_clearcoat',
    'KHR_materials_unlit',
    'KHR_texture_transform',
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_TEXTURE_TRANSFORM_MULTI_TEST)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 29, 'TextureTransformMultiTest should load transform panels plus labels/background')

  const meshesByName = new Map(meshes.map((mesh) => [mesh.name, mesh]))
  const transformedOffset = [0.7049999535083774, 0.28500004152502995]
  const transformedRepeat = [0.3499999940395355, 0.3499999940395355]
  const transformedRotation = 1.5707963705062866
  const assertTransformedTexture = ({
    meshName,
    slot,
    channel,
    textureName = 'TestMap',
    colorSpace,
    materialType,
  }) => {
    const mesh = meshesByName.get(meshName)
    assert.ok(mesh?.isMesh, `${meshName} should load a mesh`)
    if (materialType) {
      assert.equal(mesh.material.type, materialType)
    }
    const positionCount = mesh.geometry.getAttribute('position')?.count
    assert.ok(positionCount > 0, `${meshName} should load positions`)
    assert.equal(mesh.geometry.getAttribute('uv')?.count, positionCount, `${meshName} should load primary UVs`)
    assert.equal(mesh.geometry.getAttribute('uv1')?.count, positionCount, `${meshName} should load secondary UVs`)

    const texture = mesh.material[slot]
    assert.ok(texture?.isTexture, `${meshName}.${slot} should load a texture`)
    assert.equal(texture.name, textureName)
    assert.equal(Buffer.isBuffer(texture.image), true, `${meshName}.${slot} should load an encoded PNG Buffer`)
    assert.equal(texture.channel, channel)
    assertVectorClose(texture.offset.toArray(), transformedOffset, `${meshName}.${slot}.offset`, 1e-7)
    assertVectorClose(texture.repeat.toArray(), transformedRepeat, `${meshName}.${slot}.repeat`, 1e-7)
    assert.ok(Math.abs(texture.rotation - transformedRotation) < 1e-7, `${meshName}.${slot}.rotation should preserve KHR_texture_transform`)
    assertVectorClose(texture.center.toArray(), [0, 0], `${meshName}.${slot}.center`)
    assert.equal(texture.flipY, false)
    if (colorSpace !== undefined) {
      assert.equal(texture.colorSpace, colorSpace)
    }
  }

  assertTransformedTexture({ meshName: 'BaseColorUV0', slot: 'map', channel: 0, colorSpace: THREE.SRGBColorSpace, materialType: 'MeshStandardMaterial' })
  assertTransformedTexture({ meshName: 'BaseColorUV1', slot: 'map', channel: 1, colorSpace: THREE.SRGBColorSpace, materialType: 'MeshStandardMaterial' })
  assertTransformedTexture({ meshName: 'EmissionUV1', slot: 'emissiveMap', channel: 1, colorSpace: THREE.SRGBColorSpace, materialType: 'MeshStandardMaterial' })
  assertTransformedTexture({ meshName: 'NormalUV1', slot: 'normalMap', channel: 1, textureName: 'TestMap_Normal', colorSpace: THREE.NoColorSpace, materialType: 'MeshStandardMaterial' })
  assertTransformedTexture({ meshName: 'MetalRoughUV1', slot: 'roughnessMap', channel: 1, materialType: 'MeshStandardMaterial' })
  assertTransformedTexture({ meshName: 'MetalRoughUV1', slot: 'metalnessMap', channel: 1, materialType: 'MeshStandardMaterial' })
  assert.equal(meshesByName.get('MetalRoughUV1').material.roughnessMap.source, meshesByName.get('MetalRoughUV1').material.metalnessMap.source)
  assertTransformedTexture({ meshName: 'OcclusionUV1', slot: 'aoMap', channel: 1, materialType: 'MeshStandardMaterial' })
  assertTransformedTexture({ meshName: 'UnlitUV1', slot: 'map', channel: 1, materialType: 'MeshBasicMaterial' })
  assertTransformedTexture({ meshName: 'ClearcoatUV1', slot: 'clearcoatMap', channel: 1, materialType: 'MeshPhysicalMaterial' })
  assertTransformedTexture({ meshName: 'ClearcoatRoughUV1', slot: 'clearcoatRoughnessMap', channel: 1, colorSpace: THREE.NoColorSpace, materialType: 'MeshPhysicalMaterial' })
  assertTransformedTexture({ meshName: 'ClearcoatNormalUV1', slot: 'clearcoatNormalMap', channel: 1, textureName: 'TestMap_Normal', colorSpace: THREE.NoColorSpace, materialType: 'MeshPhysicalMaterial' })

  const camera = new THREE.OrthographicCamera(-0.05, 0.75, 0.95, -1.45, 0.01, 10)
  camera.position.set(0.35, -0.25, 2)
  camera.lookAt(0.35, -0.25, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.9))
  const light = new THREE.DirectionalLight(0xffffff, 1.6)
  light.position.set(0.2, 1, 2)
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 180,
    height: 420,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.6, 'TextureTransformMultiTest should render the transformed texture grid')
  const baseColorRow = meanRegion(rgba, 180, 420, 38, 36, 142, 70)
  assert.ok(baseColorRow.b > baseColorRow.r + 30 && baseColorRow.b > baseColorRow.g + 30, `TextureTransformMultiTest should render blue background and transformed panels (${baseColorRow.r}, ${baseColorRow.g}, ${baseColorRow.b})`)
})

test('committed Khronos glTF Sample Assets MetalRoughSpheres fixture loads packed metallic-roughness maps', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_METAL_ROUGH_SPHERES, 'utf8'))
  assert.equal(source.buffers[0].uri, 'MetalRoughSpheres0.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'Spheres_BaseColor.png',
    'Spheres_MetalRough.png',
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_METAL_ROUGH_SPHERES)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), ['Spheres', 'Spheres001', 'Spheres002', 'Spheres003', 'Spheres004'])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [36590, 62664, 62664, 62664, 31332])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [215088, 368640, 368640, 368640, 184320])

  const material = meshes[0].material
  assert.equal(material.isMeshStandardMaterial, true)
  assert.ok(material.map?.isTexture, 'MetalRoughSpheres should load a base color texture')
  assert.ok(material.roughnessMap?.isTexture, 'MetalRoughSpheres should load a roughness texture')
  assert.ok(material.metalnessMap?.isTexture, 'MetalRoughSpheres should load a metalness texture')
  assert.equal(material.roughnessMap, material.metalnessMap, 'packed metallic-roughness channels should share one texture')
  assert.deepEqual(pngDimensions(material.map.image), [1024, 1024])
  assert.deepEqual(pngDimensions(material.roughnessMap.image), [1024, 1024])
  assert.equal(material.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(material.roughnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(material.map.flipY, false)
  assert.equal(material.roughnessMap.flipY, false)

  const ratio = renderSingleObjectRatio(new Renderer(), meshes[0])
  assert.ok(ratio > 0.03, `MetalRoughSpheres representative mesh should render visible pixels (${ratio})`)
})

test('committed Khronos glTF Sample Assets MetalRoughSpheresNoTextures fixture loads scalar metallic-roughness grids', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_METAL_ROUGH_SPHERES_NO_TEXTURES, 'utf8'))
  assert.deepEqual(source.buffers, [{ byteLength: 241588, uri: 'MetalRoughSpheresNoTextures.bin' }])
  assert.equal(source.images, undefined)
  assert.equal(source.textures, undefined)
  assert.equal(source.meshes.length, 102)
  assert.equal(source.materials.length, 98)
  assert.equal(
    source.materials.every((material) => (
      material.pbrMetallicRoughness?.baseColorTexture === undefined &&
      material.pbrMetallicRoughness?.metallicRoughnessTexture === undefined
    )),
    true,
    'MetalRoughSpheresNoTextures should rely on scalar PBR factors instead of textures',
  )

  const expectedSteps = [0, 0.1666666716337204, 0.3333333432674408, 0.5, 0.6666666865348816, 0.8333333134651184, 1]
  assert.deepEqual(source.materials.slice(0, 7).map((material) => material.pbrMetallicRoughness.roughnessFactor), expectedSteps)
  assert.deepEqual(source.materials.slice(0, 49).filter((_, index) => index % 7 === 0).map((material) => material.pbrMetallicRoughness.metallicFactor), expectedSteps)
  assert.deepEqual(source.materials.slice(49, 56).map((material) => material.pbrMetallicRoughness.roughnessFactor), expectedSteps)
  assert.deepEqual(source.materials.slice(49).filter((_, index) => index % 7 === 0).map((material) => material.pbrMetallicRoughness.metallicFactor), expectedSteps)

  const gltf = await loadGltfFixture(SAMPLE_ASSET_METAL_ROUGH_SPHERES_NO_TEXTURES)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })

  assert.equal(meshes.length, 123)
  const materialGrid = meshes.slice(0, 98)
  assert.deepEqual(materialGrid.slice(0, 7).map((mesh) => mesh.name), [
    'm0%_r0%',
    'm0%_r16%',
    'm0%_r33%',
    'm0%_r50%',
    'm0%_r66%',
    'm0%_r83%',
    'm0%_r100%',
  ])
  assert.deepEqual(materialGrid.slice(49, 56).map((mesh) => mesh.name), [
    'g_m0%_r0%',
    'g_m0%_r16%',
    'g_m0%_r33%',
    'g_m0%_r50%',
    'g_m0%_r66%',
    'g_m0%_r83%',
    'g_m0%_r100%',
  ])
  assert.deepEqual(materialGrid.slice(0, 3).map((mesh) => mesh.geometry.getAttribute('position')?.count), [5374, 5374, 5374])
  assert.deepEqual(materialGrid.slice(0, 3).map((mesh) => mesh.geometry.index?.count), [31800, 31800, 31800])
  assert.equal(new Set(materialGrid.map((mesh) => mesh.material.uuid)).size, 98)
  assert.equal(materialGrid.every((mesh) => mesh.material.isMeshStandardMaterial === true), true)
  assert.equal(materialGrid.every((mesh) => mesh.material.map === null && mesh.material.roughnessMap === null && mesh.material.metalnessMap === null), true)

  assert.equal(materialGrid[0].material.metalness, 0)
  assert.equal(materialGrid[0].material.roughness, 0)
  assert.equal(materialGrid[6].material.metalness, 0)
  assert.equal(materialGrid[6].material.roughness, 1)
  assert.equal(materialGrid[48].material.metalness, 1)
  assert.equal(materialGrid[48].material.roughness, 1)
  assert.equal(materialGrid[97].material.metalness, 1)
  assert.equal(materialGrid[97].material.roughness, 1)
  assertVectorClose(materialGrid[0].material.color.toArray(), [0.6038269996643066, 0.6038269996643066, 0.6038269996643066], 'neutral scalar sphere color')
  assertVectorClose(materialGrid[97].material.color.toArray(), [0.6038274168968201, 0.4396572411060333, 0.01228648703545332], 'gold scalar sphere color')

  const neutralRatio = renderSingleObjectRatio(new Renderer(), materialGrid[0], 0.001)
  assert.ok(neutralRatio > 0.03, `MetalRoughSpheresNoTextures neutral representative mesh should render visible pixels (${neutralRatio})`)
  const goldRatio = renderSingleObjectRatio(new Renderer(), materialGrid[97], 0.001)
  assert.ok(goldRatio > 0.03, `MetalRoughSpheresNoTextures gold representative mesh should render visible pixels (${goldRatio})`)
})

test('committed Khronos glTF Sample Assets MeshPrimitiveModes fixture loads and renders primitive modes', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_MESH_PRIMITIVE_MODES)
  const renderables = []
  gltf.scene.traverse((object) => {
    if (
      object.isMesh === true ||
      object.isLine === true ||
      object.isLineSegments === true ||
      object.isLineLoop === true ||
      object.isPoints === true
    ) {
      renderables.push(object)
    }
  })

  assert.deepEqual(renderables.map((object) => ({
    name: object.name,
    type: object.type,
    index: object.geometry.index?.count,
    positions: object.geometry.getAttribute('position')?.count,
  })), [
    { name: 'mesh_with_POINTS', type: 'Points', index: 7, positions: 7 },
    { name: 'mesh_with_LINES', type: 'LineSegments', index: 12, positions: 7 },
    { name: 'mesh_with_LINE_LOOP', type: 'LineLoop', index: 7, positions: 7 },
    { name: 'mesh_with_LINE_STRIP', type: 'Line', index: 7, positions: 7 },
    { name: 'mesh_with_TRIANGLES', type: 'Mesh', index: 18, positions: 7 },
    { name: 'mesh_with_GL_TRIANGLE_STRIP', type: 'Mesh', index: 12, positions: 7 },
    { name: 'mesh_with_GL_TRIANGLE_FAN', type: 'Mesh', index: 18, positions: 7 },
  ])

  for (const object of renderables) {
    if (object.material?.color) object.material.color.set(0xffffff)
    if (object.isPoints === true) object.material.size = 10
    if (object.isLine === true || object.isLineSegments === true || object.isLineLoop === true) {
      object.material.linewidth = 4
    }
  }

  const camera = new THREE.OrthographicCamera(-4, 4, 4, -4, 0.01, 10)
  camera.position.set(0, 0, 4)
  camera.lookAt(0, 0, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1.0))
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 128,
    height: 128,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.2, 'MeshPrimitiveModes sample should render visible points, lines, and meshes')
  const points = meanRegion(rgba, 128, 128, 56, 8, 72, 24)
  const lineLoop = meanRegion(rgba, 128, 128, 56, 56, 72, 72)
  const triangleFan = meanRegion(rgba, 128, 128, 88, 104, 104, 120)
  assert.ok(points.r > 60 && points.g > 60 && points.b > 60, `POINTS primitive should render visible pixels (${points.r}, ${points.g}, ${points.b})`)
  assert.ok(lineLoop.r > 40 && lineLoop.g > 40 && lineLoop.b > 40, `LINE_LOOP primitive should render visible pixels (${lineLoop.r}, ${lineLoop.g}, ${lineLoop.b})`)
  assert.ok(triangleFan.r > 120 && triangleFan.g > 120 && triangleFan.b > 120, `TRIANGLE_FAN primitive should render visible pixels (${triangleFan.r}, ${triangleFan.g}, ${triangleFan.b})`)
})

test('committed Khronos glTF Sample Assets MeshoptCubeTest fixture loads quantized fallback cube grid', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_MESHOPT_CUBE_TEST, 'utf8'))
  assert.deepEqual(source.extensionsRequired, ['KHR_mesh_quantization'])
  assert.deepEqual(source.extensionsUsed, ['KHR_mesh_quantization', 'KHR_meshopt_compression'])
  assert.deepEqual(source.buffers, [
    { uri: 'MeshoptCubeTest.bin', byteLength: 10528 },
    {
      uri: 'MeshoptCubeTestFallback.bin',
      byteLength: 9984,
      extensions: {
        KHR_meshopt_compression: {
          fallback: true,
        },
      },
    },
  ])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'row0.png',
    'row1.png',
    'row2.png',
    'row3.png',
    'row4.png',
    'col0.png',
    'col1.png',
    'col2.png',
    'col3.png',
    'col4.png',
  ])
  assert.equal(source.meshes.length, 35)
  assert.equal(source.animations.length, 1)
  assert.equal(source.animations[0].channels.length, 5)

  const gltf = await loadGltfFixture(SAMPLE_ASSET_MESHOPT_CUBE_TEST)
  assert.deepEqual(gltf.parser?.json?.extensionsRequired, ['KHR_mesh_quantization'])
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_meshopt_compression'))
  assert.equal(gltf.animations.length, 1)
  assert.equal(gltf.animations[0].name, 'RotateCubes')
  assert.equal(gltf.animations[0].duration, 2)
  assert.deepEqual(gltf.animations[0].tracks.map((track) => track.name), [
    'Cube_4_animated_rotation.quaternion',
    'Cube_9_animated_rotation_compressed_indices.quaternion',
    'Cube_14_animated_rotation_compressed_triangles.quaternion',
    'Cube_19_animated_rotation_compressed_filtered.quaternion',
    'Cube_24_animated_rotation_compressed_filtered_v1.quaternion',
  ])

  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 35)

  const labels = meshes.filter((mesh) => mesh.name.includes('Label'))
  assert.equal(labels.length, 10)
  assert.deepEqual(labels.map((mesh) => mesh.name), [
    'RowLabel_0',
    'RowLabel_1',
    'RowLabel_2',
    'RowLabel_3',
    'RowLabel_4',
    'ColLabel_0',
    'ColLabel_1',
    'ColLabel_2',
    'ColLabel_3',
    'ColLabel_4',
  ])
  for (const label of labels) {
    assert.equal(label.geometry.getAttribute('position')?.count, 4)
    assert.equal(label.geometry.getAttribute('normal')?.count, 4)
    assert.equal(label.geometry.getAttribute('uv')?.count, 4)
    assert.equal(label.geometry.index?.count, 6)
    assert.equal(Buffer.isBuffer(label.material.map?.image), true, `${label.name} label PNG should load as an encoded Buffer`)
    assert.deepEqual(pngDimensions(label.material.map.image), [256, 128])
    assert.equal(label.material.map.colorSpace, THREE.SRGBColorSpace)
    assert.equal(label.material.map.flipY, false)
  }

  const cubes = meshes.filter((mesh) => mesh.name.startsWith('Cube_'))
  assert.equal(cubes.length, 25)
  assert.equal(cubes.filter((mesh) => mesh.geometry.getAttribute('color')).length, 20)
  assert.equal(cubes.filter((mesh) => !mesh.geometry.getAttribute('color')).length, 5)
  assert.deepEqual(cubes.slice(0, 5).map((mesh) => mesh.name), [
    'Cube_0_interleaved_u8norm_u8color_u16index',
    'Cube_1_deinterleaved_u8norm_u8color_u16index',
    'Cube_2_deinterleaved_u16norm_u16color_u16index',
    'Cube_3_deinterleaved_u8norm_u8color_u32index',
    'Cube_4_animated_rotation',
  ])
  assert.deepEqual(cubes.slice(-5).map((mesh) => mesh.name), [
    'Cube_20_interleaved_u8norm_u8color_u16index_compressed_filtered_v1',
    'Cube_21_deinterleaved_u8norm_u8color_u16index_compressed_filtered_v1',
    'Cube_22_deinterleaved_u16norm_u16color_u16index_compressed_filtered_v1',
    'Cube_23_deinterleaved_u8norm_u8color_u32index_compressed_filtered_v1',
    'Cube_24_animated_rotation_compressed_filtered_v1',
  ])
  for (const cube of cubes) {
    assert.equal(cube.geometry.getAttribute('position')?.count, 24)
    assert.equal(cube.geometry.getAttribute('normal')?.count, 24)
    assert.equal(cube.geometry.index?.count, 36)
  }

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100)
  camera.position.copy(center).add(new THREE.Vector3(0, size.y * 0.2, Math.max(size.x, size.y, size.z) * 1.9))
  camera.lookAt(center)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 2.2)
  light.position.copy(center).add(new THREE.Vector3(2, 3, 5))
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.07, 'MeshoptCubeTest should render visible quantized cube grid')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 10 && mean.g > 10 && mean.b > 10, `MeshoptCubeTest should render lit cube-grid pixels (${mean.r}, ${mean.g}, ${mean.b})`)
})
