import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'
import { Renderer, SAMPLE_ASSET_ANIMATED_COLORS_CUBE, SAMPLE_ASSET_ANIMATED_CUBE, SAMPLE_ASSET_ANIMATED_TRIANGLE, SAMPLE_ASSET_ANIMATION_POINTER_UVS, SAMPLE_ASSET_BOX_ANIMATED, SAMPLE_ASSET_INTERPOLATION_TEST, SAMPLE_ASSET_LIGHTS_PUNCTUAL_LAMP } from './gltf.test.part-001.mjs'
import { assertVectorClose, findFirst, loadGltfFixture, meanRegion, nonBackgroundBounds, pngDimensions } from './gltf.test.part-028.mjs'
test('committed Khronos glTF Sample Assets LightsPunctualLamp fixture loads textured point-light scene', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_LIGHTS_PUNCTUAL_LAMP, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_transmission', 'KHR_lights_punctual'])
  assert.equal(source.buffers[0].uri, 'LightsPunctualLamp.data.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'material0_basecolor.jpeg',
    'material0_normal.png',
    'material0_emissive.jpeg',
    'material0_metallic_roughness.jpeg',
    'material1_basecolor.png',
    'material1_normal.png',
    'material2_transmission.jpeg',
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_LIGHTS_PUNCTUAL_LAMP)
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_lights_punctual'))
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_transmission'))

  const meshes = []
  const lights = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
    if (object.isLight === true) lights.push(object)
  })

  assert.deepEqual(meshes.map((mesh) => mesh.name), ['mesh_0', 'mesh_1', 'mesh_2'])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [3212, 18, 1325])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('tangent')?.count), [3212, 18, 1325])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [12210, 42, 5748])

  assert.equal(lights.length, 5)
  assert.ok(lights.every((light) => light.isPointLight === true), 'all imported punctual lights should become PointLight objects')
  assert.deepEqual(lights.map((light) => light.name), [
    'Point_Orientation',
    'Point002_Orientation',
    'Point001_Orientation',
    'Point003_Orientation',
    'Point004_Orientation',
  ])
  assert.deepEqual(lights.map((light) => light.intensity), [15, 1.5, 80, 80, 180])
  assert.ok(lights.every((light) => light.distance === 0 && light.decay === 2))
  assertVectorClose(lights[0].color.toArray(), [1, 0.6318749785423279, 0.23909975588321689], 'warm lamp point-light color')
  assertVectorClose(lights[2].color.toArray(), [0.21223080158233645, 0.5906190276145935, 0.5583405494689941], 'cyan lamp point-light color')

  gltf.scene.updateMatrixWorld(true)
  assertVectorClose(lights[0].getWorldPosition(new THREE.Vector3()).toArray(), [0.04622355476021767, 0.9077973365783693, 0.006696629337966442], 'first lamp light position')
  assertVectorClose(lights[4].getWorldPosition(new THREE.Vector3()).toArray(), [0.2920210361480713, 1.0323998928070068, 1.5589159727096558], 'last lamp light position')

  const [body, shade, glass] = meshes
  assert.equal(body.material.isMeshStandardMaterial, true)
  assert.equal(body.material.side, THREE.DoubleSide)
  assert.equal(body.material.emissiveMap.name, 'material0_emissive.jpeg')
  assert.equal(body.material.map.name, 'material0_basecolor.jpeg')
  assert.equal(body.material.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(body.material.normalMap.name, 'material0_normal.png')
  assert.equal(body.material.normalMap.colorSpace, THREE.NoColorSpace)
  assert.equal(body.material.metalnessMap.name, 'material0_metallic_roughness.jpeg')
  assert.equal(body.material.roughnessMap, body.material.metalnessMap)
  assert.deepEqual(pngDimensions(body.material.normalMap.image), [2048, 2048])

  assert.equal(shade.material.transparent, true)
  assert.equal(shade.material.side, THREE.DoubleSide)
  assert.equal(shade.material.metalness, 0)
  assert.equal(shade.material.roughness, 0.5)
  assert.equal(shade.material.map.name, 'material1_basecolor.png')
  assert.equal(shade.material.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(shade.material.normalMap.name, 'material1_normal.png')
  assert.deepEqual(pngDimensions(shade.material.map.image), [512, 512])
  assert.deepEqual(pngDimensions(shade.material.normalMap.image), [512, 512])

  assert.equal(glass.material.isMeshPhysicalMaterial, true)
  assert.equal(glass.material.side, THREE.DoubleSide)
  assert.equal(glass.material.transmission, 1)
  assert.equal(glass.material.map, body.material.map)
  assert.equal(glass.material.normalMap, body.material.normalMap)
  assert.equal(glass.material.transmissionMap.name, 'material2_transmission.jpeg')
  assert.equal(Buffer.isBuffer(glass.material.transmissionMap.image), true)
  assert.equal(glass.material.transmissionMap.colorSpace, THREE.NoColorSpace)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 20)
  camera.position.copy(center).add(new THREE.Vector3(0, -3.1, 1.2))
  camera.lookAt(center)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.15, 'LightsPunctualLamp should render visible textured geometry')
  const centerRegion = meanRegion(rgba, 96, 96, 32, 32, 64, 64)
  assert.ok(centerRegion.r > 60 && centerRegion.g > 45 && centerRegion.b > 40, `lamp render should include warm textured light contribution (${centerRegion.r}, ${centerRegion.g}, ${centerRegion.b})`)
})

test('committed Khronos glTF Sample Assets InterpolationTest fixture applies animation interpolation modes', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_INTERPOLATION_TEST)
  assert.deepEqual(gltf.animations.map((clip) => clip.name), [
    'Step Scale',
    'Linear Scale',
    'CubicSpline Scale',
    'Step Rotation',
    'CubicSpline Rotation',
    'Linear Rotation',
    'Step Translation',
    'CubicSpline Translation',
    'Linear Translation',
  ])

  const tracksByClip = new Map(gltf.animations.map((clip) => [clip.name, clip.tracks[0]]))
  assert.equal(tracksByClip.get('Step Scale')?.name, 'Cube.scale')
  assert.equal(tracksByClip.get('Step Scale')?.getInterpolation(), THREE.InterpolateDiscrete)
  assert.equal(tracksByClip.get('Linear Scale')?.getInterpolation(), THREE.InterpolateLinear)
  assert.equal(tracksByClip.get('CubicSpline Scale')?.getValueSize(), 9)
  assert.equal(tracksByClip.get('Step Rotation')?.name, 'Cube003.quaternion')
  assert.equal(tracksByClip.get('Linear Rotation')?.getInterpolation(), THREE.InterpolateLinear)
  assert.equal(tracksByClip.get('CubicSpline Rotation')?.getValueSize(), 12)
  assert.equal(tracksByClip.get('Step Translation')?.name, 'Cube006.position')
  assert.equal(tracksByClip.get('Linear Translation')?.getInterpolation(), THREE.InterpolateLinear)
  assert.equal(tracksByClip.get('CubicSpline Translation')?.getValueSize(), 9)

  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 10, 'InterpolationTest should load nine animated cubes plus one textured plane')
  const plane = gltf.scene.getObjectByName('Plane')
  assert.ok(Buffer.isBuffer(plane?.material?.map?.image), 'InterpolationTest external PNG should load as an encoded Buffer')

  const mixer = new THREE.AnimationMixer(gltf.scene)
  for (const clip of gltf.animations) mixer.clipAction(clip).play()
  mixer.setTime(0.25)
  gltf.scene.updateMatrixWorld(true)

  assert.ok(Math.abs(gltf.scene.getObjectByName('Cube').scale.x - 1) < 1e-6, 'STEP scale should hold the previous keyframe at t=0.25')
  assert.ok(Math.abs(gltf.scene.getObjectByName('Cube001').scale.x - 0.5) < 1e-6, 'LINEAR scale should interpolate halfway at t=0.25')
  assert.ok(Math.abs(gltf.scene.getObjectByName('Cube002').scale.x - 0.5) < 1e-6, 'CUBICSPLINE scale should interpolate halfway at t=0.25')
  assert.ok(Math.abs(gltf.scene.getObjectByName('Cube003').quaternion.z) < 1e-6, 'STEP rotation should hold the previous keyframe at t=0.25')
  assert.ok(Math.abs(gltf.scene.getObjectByName('Cube005').quaternion.z + 0.19509032) < 1e-5, 'LINEAR rotation should slerp at t=0.25')
  assert.ok(Math.abs(gltf.scene.getObjectByName('Cube006').position.y - 6.80000019) < 1e-5, 'STEP translation should hold the previous keyframe at t=0.25')
  assert.ok(Math.abs(gltf.scene.getObjectByName('Cube009').position.y - 8.80000019) < 1e-5, 'LINEAR translation should interpolate halfway at t=0.25')

  const camera = new THREE.OrthographicCamera(-6, 6, 10, -2.5, 0.01, 20)
  camera.position.set(0, 3.6, 10)
  camera.lookAt(0, 3.6, 0)
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
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.1, 'InterpolationTest animated fixture should render visible geometry')
})

test('committed Khronos glTF Sample Assets AnimatedTriangle fixture loads external animation buffer', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_ANIMATED_TRIANGLE, 'utf8'))
  assert.deepEqual(source.buffers.map((buffer) => buffer.uri), [
    'AnimatedTriangle_geometry.bin',
    'AnimatedTriangle_animation.bin',
  ])
  assert.equal(source.accessors[2].count, 5)
  assert.equal(source.accessors[3].type, 'VEC4')

  const gltf = await loadGltfFixture(SAMPLE_ASSET_ANIMATED_TRIANGLE)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'AnimatedTriangle should load a mesh')
  assert.equal(mesh.name, 'mesh_0')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 3)
  assert.equal(mesh.geometry.index?.count, 3)
  assert.equal(mesh.material.isMeshStandardMaterial, true)

  assert.equal(gltf.animations.length, 1)
  const clip = gltf.animations[0]
  assert.equal(clip.name, 'animation_0')
  assert.equal(clip.duration, 1)
  assert.equal(clip.tracks.length, 1)
  const track = clip.tracks[0]
  assert.equal(track.name, 'mesh_0.quaternion')
  assert.equal(track.getInterpolation(), THREE.InterpolateLinear)
  assert.equal(track.getValueSize(), 4)
  assert.deepEqual(Array.from(track.times), [0, 0.25, 0.5, 0.75, 1])
  assertVectorClose(Array.from(track.values.slice(4, 8)), [0, 0, 0.7070000171661377, 0.7070000171661377], 'quarter-turn quaternion key')

  const mixer = new THREE.AnimationMixer(gltf.scene)
  mixer.clipAction(clip).play()
  mixer.setTime(0.5)
  assertVectorClose(mesh.quaternion.toArray(), [0, 0, 1, 0], 'AnimatedTriangle half-turn pose')
  mixer.setTime(0)

  const camera = new THREE.OrthographicCamera(-0.2, 1.2, 1.2, -0.2, 0.01, 10)
  camera.position.set(0.5, 0.5, 2)
  camera.lookAt(0.5, 0.5, 0)
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
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.035, 'AnimatedTriangle should render visible animated geometry')
})

test('committed Khronos glTF Sample Assets AnimatedCube fixture loads textured quaternion animation', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_ANIMATED_CUBE, 'utf8'))
  assert.deepEqual(source.buffers, [{ byteLength: 1860, uri: 'AnimatedCube.bin' }])
  assert.deepEqual(source.images, [{ uri: 'AnimatedCube_BaseColor.png' }])
  assert.equal(source.accessors[0].count, 3)
  assert.equal(source.accessors[1].type, 'VEC4')
  assert.equal(source.animations[0].name, 'animation_AnimatedCube')
  assert.deepEqual(source.animations[0].channels, [
    { sampler: 0, target: { node: 0, path: 'rotation' } },
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_ANIMATED_CUBE)
  const mesh = gltf.scene.getObjectByName('AnimatedCube')
  assert.ok(mesh?.isMesh, 'AnimatedCube should load a named mesh')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 36)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 36)
  assert.equal(mesh.geometry.getAttribute('tangent')?.count, 36)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 36)
  assert.equal(mesh.geometry.index?.count, 36)
  assert.equal(mesh.material.name, 'AnimatedCube')
  assert.equal(mesh.material.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(mesh.material.metalness, 0)
  assert.equal(mesh.material.roughness, 0.079)

  assert.equal(gltf.animations.length, 1)
  const clip = gltf.animations[0]
  assert.equal(clip.name, 'animation_AnimatedCube')
  assert.equal(clip.duration, 2)
  assert.equal(clip.tracks.length, 1)
  const track = clip.tracks[0]
  assert.equal(track.name, 'AnimatedCube.quaternion')
  assert.equal(track.getInterpolation(), THREE.InterpolateLinear)
  assert.equal(track.getValueSize(), 4)
  assert.deepEqual(Array.from(track.times), [0, 1, 2])
  assertVectorClose(Array.from(track.values.slice(4, 8)), [0, 1, 0, -4.371138828673793e-8], 'AnimatedCube middle quaternion key')

  const mixer = new THREE.AnimationMixer(gltf.scene)
  mixer.clipAction(clip).play()
  mixer.setTime(1)
  gltf.scene.updateMatrixWorld(true)
  assertVectorClose(mesh.quaternion.toArray(), [0, 1, 0, -4.371138828673793e-8], 'AnimatedCube half-turn pose')

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1))
  const camera = new THREE.OrthographicCamera(-2.2, 2.2, 2.2, -2.2, 0.01, 20)
  camera.position.set(0, 0, 6)
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
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.15, 'AnimatedCube should render visible textured cube geometry')
})

test('committed Khronos glTF Sample Assets AnimatedColorsCube fixture applies material color animation pointers', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_ANIMATED_COLORS_CUBE, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_animation_pointer'])
  assert.equal(source.buffers[0].uri, 'AnimatedColorsCube.bin')

  const gltf = await loadGltfFixture(SAMPLE_ASSET_ANIMATED_COLORS_CUBE)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), ['TestCube', '1-RedCube', '2-GreenCube', '3-BlueCube'])
  assert.deepEqual(meshes.map((mesh) => mesh.material.name), ['AnimatedColorMaterial', 'Red', 'Green', 'Blue'])

  assert.equal(gltf.animations.length, 1)
  const clip = gltf.animations[0]
  assert.equal(clip.name, 'Cube Animation')
  assert.deepEqual(clip.tracks.map((track) => track.name), [
    'TestCube.position',
    'TestCube.quaternion',
    'TestCube.material.color',
  ])
  const colorTrack = clip.tracks[2]
  assert.equal(colorTrack.getValueSize(), 3)
  assert.equal(colorTrack.getInterpolation(), THREE.InterpolateLinear)

  const animated = meshes[0]
  const mixer = new THREE.AnimationMixer(gltf.scene)
  mixer.clipAction(clip).play()
  mixer.setTime(1.5)
  gltf.scene.updateMatrixWorld(true)
  assertVectorClose(animated.position.toArray(), [3, 3, 0], 'AnimatedColorsCube translation at t=1.5')
  assertVectorClose(animated.material.color.toArray(), [0.019999999552965164, 0.019999999552965164, 0.800000011920929], 'AnimatedColorsCube material color at t=1.5')

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1.1))
  const camera = new THREE.OrthographicCamera(-5, 5, 4.8, -2, 0.01, 20)
  camera.position.set(0, 0, 10)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 160,
    height: 110,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.15, 'AnimatedColorsCube should render visible animated colored cubes')
})

test('committed Khronos glTF Sample Assets AnimationPointerUVs fixture loads animation-pointer texture transform coverage', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_ANIMATION_POINTER_UVS, 'utf8'))
  assert.deepEqual(source.extensionsRequired, ['KHR_materials_unlit', 'KHR_lights_punctual'])
  assert.deepEqual(source.extensionsUsed, [
    'KHR_materials_transmission',
    'KHR_materials_volume',
    'KHR_materials_specular',
    'KHR_materials_sheen',
    'KHR_materials_clearcoat',
    'KHR_texture_transform',
    'KHR_animation_pointer',
    'KHR_materials_anisotropy',
    'KHR_materials_iridescence',
    'KHR_materials_diffuse_transmission',
    'KHR_materials_unlit',
    'KHR_lights_punctual',
  ])
  assert.deepEqual(source.buffers, [{ byteLength: 5329724, uri: 'AnimationPointerUVs.bin' }])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'AnimationPointerUVs_BaseColor.png',
    'AnimationPointerUVs_DiffuseTransmission.png',
    'AnimationPointerUVs_Orm.png',
    'AnimationPointerUVs_Emissive.png',
    'AnimationPointerUVs_NormalFlat.png',
    'AnimationPointerUVs_Clearcoat.png',
    'AnimationPointerUVs_Normal.png',
    'AnimationPointerUVs_Anisotropy.png',
    'AnimationPointerUVs_ClearcoatNormal.png',
    'AnimationPointerUVs_Iridescence.png',
    'AnimationPointerUVs_Sheen.png',
    'AnimationPointerUVs_Specular.png',
    'AnimationPointerUVs_TransmissionVolume.png',
  ])
  assert.equal(source.textures.length, 61)
  assert.equal(source.materials.length, 82)
  assert.equal(source.meshes.length, 106)

  const clipSource = source.animations[0]
  assert.equal(source.animations.length, 1)
  assert.equal(clipSource.channels.length, 103)
  assert.equal(clipSource.samplers.length, 103)
  assert.equal(clipSource.channels.every((channel) => channel.target.path === 'pointer'), true)
  const pointers = clipSource.channels.map((channel) => channel.target.extensions.KHR_animation_pointer.pointer)
  assert.equal(new Set(pointers).size, 99)
  for (const pointer of [
    '/materials/11/pbrMetallicRoughness/baseColorTexture/extensions/KHR_texture_transform/scale',
    '/materials/27/extensions/KHR_materials_anisotropy/anisotropyTexture/extensions/KHR_texture_transform/rotation',
    '/materials/57/extensions/KHR_materials_sheen/sheenColorTexture/extensions/KHR_texture_transform/rotation',
    '/materials/67/extensions/KHR_materials_specular/specularTexture/extensions/KHR_texture_transform/offset',
    '/materials/72/extensions/KHR_materials_transmission/transmissionTexture/extensions/KHR_texture_transform/rotation',
    '/materials/77/extensions/KHR_materials_volume/thicknessTexture/extensions/KHR_texture_transform/scale',
    '/materials/8/extensions/KHR_materials_diffuse_transmission/diffuseTransmissionTexture/extensions/KHR_texture_transform/scale',
  ]) {
    assert.ok(pointers.includes(pointer), `AnimationPointerUVs should include pointer target ${pointer}`)
  }

  const gltf = await loadGltfFixture(SAMPLE_ASSET_ANIMATION_POINTER_UVS)
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_animation_pointer'))
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_texture_transform'))
  assert.equal(gltf.cameras.length, 11)
  assert.equal(gltf.animations.length, 1)

  const meshes = []
  const lights = []
  const materials = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) {
      meshes.push(object)
      if (!materials.includes(object.material)) materials.push(object.material)
    }
    if (object.isLight === true) lights.push(object)
  })
  assert.equal(meshes.length, 132)
  assert.deepEqual(lights.map((light) => [light.type, light.name, light.intensity]), [
    ['DirectionalLight', 'light_rear', 50],
  ])
  assert.deepEqual(materials.reduce((counts, material) => {
    counts[material.type] = (counts[material.type] ?? 0) + 1
    return counts
  }, {}), {
    MeshStandardMaterial: 27,
    MeshPhysicalMaterial: 51,
    MeshBasicMaterial: 3,
  })

  const materialsByName = new Map(materials.map((material) => [material.name, material]))
  const assertTexture = (materialName, slot, textureName, colorSpace = THREE.NoColorSpace, dimensions = [512, 512]) => {
    const texture = materialsByName.get(materialName)?.[slot]
    assert.equal(texture?.name, textureName, `${materialName}.${slot} should load ${textureName}`)
    assert.equal(Buffer.isBuffer(texture.image), true, `${textureName} should load as an encoded Buffer`)
    assert.deepEqual(pngDimensions(texture.image), dimensions)
    assert.equal(texture.colorSpace, colorSpace)
    assert.equal(texture.flipY, false)
  }

  assertTexture('Material #60', 'map', 'AnimationPointerUVs_BaseColor.png', THREE.SRGBColorSpace)
  assertTexture('Material #57', 'emissiveMap', 'AnimationPointerUVs_Emissive.png', THREE.SRGBColorSpace)
  assertTexture('Material #99', 'normalMap', 'AnimationPointerUVs_Normal.png')
  assertTexture('Material #99', 'anisotropyMap', 'AnimationPointerUVs_Anisotropy.png')
  assertTexture('Material #120', 'clearcoatMap', 'AnimationPointerUVs_Clearcoat.png')
  assertTexture('Material #120', 'clearcoatNormalMap', 'AnimationPointerUVs_Normal.png')
  assertTexture('Material #133', 'clearcoatNormalMap', 'AnimationPointerUVs_ClearcoatNormal.png')
  assertTexture('Material #148', 'iridescenceMap', 'AnimationPointerUVs_Iridescence.png')
  assertTexture('Material #158', 'sheenColorMap', 'AnimationPointerUVs_Sheen.png', THREE.SRGBColorSpace)
  assertTexture('Material #167', 'specularColorMap', 'AnimationPointerUVs_Specular.png', THREE.SRGBColorSpace)
  assertTexture('Material #176', 'transmissionMap', 'AnimationPointerUVs_TransmissionVolume.png')
  assertTexture('Material #120', 'normalMap', 'AnimationPointerUVs_NormalFlat.png', THREE.NoColorSpace, [4, 4])
  assert.equal(materialsByName.get('Material #120').clearcoatMap.source, materialsByName.get('Material #120').clearcoatRoughnessMap.source)
  assert.equal(materialsByName.get('Material #148').iridescenceMap.source, materialsByName.get('Material #148').iridescenceThicknessMap.source)
  assert.equal(materialsByName.get('Material #158').sheenColorMap.source, materialsByName.get('Material #158').sheenRoughnessMap.source)
  assert.equal(materialsByName.get('Material #167').specularColorMap.source, materialsByName.get('Material #167').specularIntensityMap.source)
  assert.equal(materialsByName.get('Material #176').transmissionMap.source, materialsByName.get('Material #176').thicknessMap.source)

  const camera = gltf.cameras.find((candidate) => candidate.name === 'camera_all')
  assert.ok(camera?.isPerspectiveCamera, 'AnimationPointerUVs should load the all-panels camera')
  camera.aspect = 1.5
  camera.updateProjectionMatrix()
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 96,
    height: 64,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.12, 'AnimationPointerUVs should render visible physical texture-transform panels')
})

test('committed Khronos glTF Sample Assets BoxAnimated fixture applies transform animation', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_BOX_ANIMATED)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 2, 'Khronos BoxAnimated sample should load inner and outer meshes')
  assert.deepEqual(meshes.map((mesh) => mesh.material.name).sort(), ['inner', 'outer'])
  assert.equal(gltf.animations.length, 1)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 20)
  camera.position.set(1.7, 1.7, 4.4)
  camera.lookAt(0, 0.8, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.7))
  const light = new THREE.DirectionalLight(0xffffff, 1.2)
  light.position.set(3, 4, 5)
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const renderer = new Renderer()
  const renderBounds = () => nonBackgroundBounds(renderer.render(gltf.scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
  }), 96, 96, [0, 0, 0], 3)

  const base = renderBounds()
  const mixer = new THREE.AnimationMixer(gltf.scene)
  mixer.clipAction(gltf.animations[0]).play()
  mixer.setTime(1.25)
  gltf.scene.updateMatrixWorld(true)
  const animated = renderBounds()

  assert.ok(base.height > 25, `BoxAnimated base pose should render visible box bounds (${base.height})`)
  assert.ok(animated.height > base.height + 40, `BoxAnimated translation track should expand vertical bounds (${animated.height} vs ${base.height})`)
  assert.ok(animated.minY < base.minY - 40, `BoxAnimated translation track should move the animated mesh upward (${animated.minY} vs ${base.minY})`)
})
