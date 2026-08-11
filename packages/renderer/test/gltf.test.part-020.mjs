import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'
import { Renderer, SAMPLE_ASSET_BRAIN_STEM, SAMPLE_ASSET_CESIUM_MAN, SAMPLE_ASSET_CESIUM_MILK_TRUCK, SAMPLE_ASSET_RIGGED_FIGURE, SAMPLE_ASSET_RIGGED_SIMPLE } from './gltf.test.part-001.mjs'
import { findFirst, loadGltfFixture, nonBackgroundBounds } from './gltf.test.part-028.mjs'
test('committed Khronos glTF Sample Assets RiggedSimple fixture applies skinned bone animation', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_RIGGED_SIMPLE)
  const mesh = findFirst(gltf.scene, (object) => object.isSkinnedMesh === true)
  assert.ok(mesh, 'Khronos RiggedSimple sample should load a SkinnedMesh')
  assert.equal(mesh.name, 'Cylinder')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 160)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 160)
  assert.equal(mesh.geometry.getAttribute('skinIndex')?.count, 160)
  assert.equal(mesh.geometry.getAttribute('skinWeight')?.count, 160)
  assert.equal(mesh.geometry.index?.count, 564)
  assert.deepEqual(mesh.skeleton.bones.map((bone) => bone.name), ['Bone', 'Bone001'])
  assert.equal(gltf.animations.length, 1)
  assert.deepEqual(gltf.animations[0].tracks.map((track) => track.name), [
    'Bone001.position',
    'Bone001.quaternion',
    'Bone001.scale',
  ])

  const camera = new THREE.OrthographicCamera(-3, 3, 5, -1, 0.01, 30)
  camera.position.set(8, 0, 3)
  camera.lookAt(0, 0, 1.8)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1.0))
  const light = new THREE.DirectionalLight(0xffffff, 1.0)
  light.position.set(2, 3, 5)
  gltf.scene.add(light)
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

  const mixer = new THREE.AnimationMixer(gltf.scene)
  mixer.clipAction(gltf.animations[0]).play()
  mixer.setTime(0)
  gltf.scene.updateMatrixWorld(true)
  const base = renderBounds()

  mixer.setTime(1)
  gltf.scene.updateMatrixWorld(true)
  const animated = renderBounds()
  const animatedBone = gltf.scene.getObjectByName('Bone001')

  assert.ok(animatedBone.quaternion.x > 0.25, `RiggedSimple peak pose should rotate the animated bone (${animatedBone.quaternion.x})`)
  assert.ok(base.width > 80 && base.height < 30, `RiggedSimple base pose should render a long straight cylinder (${base.width}x${base.height})`)
  assert.ok(animated.height > base.height + 25, `RiggedSimple bone animation should bend the cylinder taller in side view (${animated.height} vs ${base.height})`)
  assert.ok(animated.minY < base.minY - 25, `RiggedSimple bone animation should lift the bent tip upward (${animated.minY} vs ${base.minY})`)
})

test('committed Khronos glTF Sample Assets RiggedFigure fixture loads full skinned animation hierarchy', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_RIGGED_FIGURE, 'utf8'))
  assert.deepEqual(source.buffers, [{ byteLength: 22184, uri: 'RiggedFigure0.bin' }])
  assert.equal(source.nodes.length, 22)
  assert.equal(source.skins.length, 1)
  assert.equal(source.skins[0].joints.length, 19)
  assert.equal(source.skins[0].skeleton, 2)
  assert.equal(source.meshes[0].name, 'Proxy')
  assert.equal(source.animations.length, 1)
  assert.equal(source.animations[0].channels.length, 57)
  assert.equal(source.animations[0].samplers.length, 57)

  const gltf = await loadGltfFixture(SAMPLE_ASSET_RIGGED_FIGURE)
  const skinnedMeshes = []
  const bones = []
  gltf.scene.traverse((object) => {
    if (object.isSkinnedMesh === true) skinnedMeshes.push(object)
    if (object.isBone === true) bones.push(object)
  })

  assert.equal(skinnedMeshes.length, 1)
  const mesh = skinnedMeshes[0]
  assert.equal(mesh.name, 'Proxy')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 370)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 370)
  assert.equal(mesh.geometry.getAttribute('skinIndex')?.count, 370)
  assert.equal(mesh.geometry.getAttribute('skinWeight')?.count, 370)
  assert.equal(mesh.geometry.index?.count, 768)
  assert.equal(mesh.material.name, 'Default-effect')
  assert.equal(mesh.skeleton.bones.length, 19)
  assert.deepEqual(bones.map((bone) => bone.name), [
    'torso_joint_1',
    'torso_joint_2',
    'torso_joint_3',
    'neck_joint_1',
    'neck_joint_2',
    'arm_joint_L_1',
    'arm_joint_L_2',
    'arm_joint_L_3',
    'arm_joint_R_1',
    'arm_joint_R_2',
    'arm_joint_R_3',
    'leg_joint_L_1',
    'leg_joint_L_2',
    'leg_joint_L_3',
    'leg_joint_L_5',
    'leg_joint_R_1',
    'leg_joint_R_2',
    'leg_joint_R_3',
    'leg_joint_R_5',
  ])

  assert.equal(gltf.animations.length, 1)
  const clip = gltf.animations[0]
  assert.equal(clip.name, 'animation_0')
  assert.equal(clip.tracks.length, 57)
  assert.equal(clip.duration, 1.25)
  assert.equal(clip.tracks.filter((track) => track.name.endsWith('.position')).length, 19)
  assert.equal(clip.tracks.filter((track) => track.name.endsWith('.quaternion')).length, 19)
  assert.equal(clip.tracks.filter((track) => track.name.endsWith('.scale')).length, 19)
  assert.ok(clip.tracks.every((track) => track.times.length === 2), 'every RiggedFigure track should contain 2 keyframes')

  const mixer = new THREE.AnimationMixer(gltf.scene)
  mixer.clipAction(clip).play()
  mixer.setTime(clip.duration / 2)
  gltf.scene.updateMatrixWorld(true)
  const torso = gltf.scene.getObjectByName('torso_joint_1')
  const leftArm = gltf.scene.getObjectByName('arm_joint_L_2')
  const rightLeg = gltf.scene.getObjectByName('leg_joint_R_3')
  assert.ok(Math.abs(torso.quaternion.x) > 0.03, `RiggedFigure torso should rotate at mid animation (${torso.quaternion.x})`)
  assert.ok(Math.abs(leftArm.quaternion.z) > 0.25, `RiggedFigure left arm should rotate at mid animation (${leftArm.quaternion.z})`)
  assert.ok(rightLeg.quaternion.x > 0.8, `RiggedFigure right leg should rotate at mid animation (${rightLeg.quaternion.x})`)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1.0))
  const light = new THREE.DirectionalLight(0xffffff, 2.0)
  light.position.set(2, 4, 5)
  gltf.scene.add(light)
  const halfHeight = size.y / 2 + 0.1
  const halfWidth = Math.max(size.x / 2 + 0.1, halfHeight)
  const camera = new THREE.OrthographicCamera(-halfWidth, halfWidth, halfHeight, -halfHeight, 0.01, 20)
  camera.position.set(center.x, center.y, center.z + 8)
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.04, 'RiggedFigure should render visible skinned figure geometry')
})

test('committed Khronos glTF Sample Assets BrainStem fixture loads multi-primitive skinned animation', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_BRAIN_STEM, 'utf8'))
  assert.deepEqual(source.buffers, [{ byteLength: 3105104, uri: 'BrainStem0.bin' }])
  assert.equal(source.images, undefined)
  assert.equal(source.materials.length, 59)
  assert.deepEqual(source.materials.slice(0, 5).map((material) => material.name), [
    'frameInStem-effect',
    'componentsStem-effect',
    'Stem-effect',
    'footStem-effect',
    'footFlangeStem-effect',
  ])
  assert.deepEqual(source.materials.slice(-5).map((material) => material.name), [
    'headCaseStem-effect',
    'headGrillStem-effect',
    'headHornStem-effect',
    'eyeStem-effect',
    'eyeRimStem-effect',
  ])
  assert.equal(source.meshes[0].name, 'Figure_2_geometry')
  assert.equal(source.meshes[0].primitives.length, 59)
  assert.deepEqual(source.meshes[0].primitives[0].attributes, {
    JOINTS_0: 1,
    NORMAL: 2,
    POSITION: 3,
    WEIGHTS_0: 4,
  })
  assert.ok(source.meshes[0].primitives.every((primitive, index) => primitive.material === index), 'BrainStem primitives should map one-to-one to materials')
  assert.equal(source.nodes.length, 22)
  assert.equal(source.skins.length, 1)
  assert.equal(source.skins[0].joints.length, 18)
  assert.equal(source.animations.length, 1)
  assert.equal(source.animations[0].channels.length, 57)
  assert.equal(source.animations[0].samplers.length, 57)

  const gltf = await loadGltfFixture(SAMPLE_ASSET_BRAIN_STEM)
  const skinnedMeshes = []
  gltf.scene.traverse((object) => {
    if (object.isSkinnedMesh === true) skinnedMeshes.push(object)
  })
  assert.equal(skinnedMeshes.length, 59)
  assert.deepEqual(skinnedMeshes.slice(0, 5).map((mesh) => mesh.name), [
    'Figure_2_geometry',
    'Figure_2_geometry_1',
    'Figure_2_geometry_2',
    'Figure_2_geometry_3',
    'Figure_2_geometry_4',
  ])
  assert.deepEqual(skinnedMeshes.slice(-5).map((mesh) => mesh.name), [
    'Figure_2_geometry_54',
    'Figure_2_geometry_55',
    'Figure_2_geometry_56',
    'Figure_2_geometry_57',
    'Figure_2_geometry_58',
  ])
  assert.deepEqual(skinnedMeshes.slice(0, 5).map((mesh) => mesh.material.name), [
    'frameInStem-effect',
    'componentsStem-effect',
    'Stem-effect',
    'footStem-effect',
    'footFlangeStem-effect',
  ])
  assert.ok(skinnedMeshes.every((mesh) => mesh.material.isMeshStandardMaterial === true), 'BrainStem materials should load as MeshStandardMaterial instances')
  assert.ok(skinnedMeshes.every((mesh) => mesh.material.metalness === 0 && mesh.material.roughness === 1), 'BrainStem materials should load scalar PBR defaults')
  assert.ok(skinnedMeshes.every((mesh) => mesh.skeleton.bones.length === 18), 'every BrainStem primitive should retain the shared 18-bone skeleton')
  assert.ok(skinnedMeshes.every((mesh) => mesh.geometry.getAttribute('position')?.count === mesh.geometry.getAttribute('normal')?.count))
  assert.ok(skinnedMeshes.every((mesh) => mesh.geometry.getAttribute('position')?.count === mesh.geometry.getAttribute('skinIndex')?.count))
  assert.ok(skinnedMeshes.every((mesh) => mesh.geometry.getAttribute('position')?.count === mesh.geometry.getAttribute('skinWeight')?.count))
  assert.equal(skinnedMeshes.reduce((sum, mesh) => sum + mesh.geometry.getAttribute('position').count, 0), 34159)
  assert.equal(skinnedMeshes.reduce((sum, mesh) => sum + (mesh.geometry.index?.count ?? 0), 0), 184998)

  const clip = gltf.animations[0]
  assert.equal(clip.name, 'animation_0')
  assert.ok(Math.abs(clip.duration - 34.880001068115234) < 1e-6, `BrainStem clip duration should match source timing (${clip.duration})`)
  assert.equal(clip.tracks.length, 57)
  assert.equal(clip.tracks.filter((track) => track.name.endsWith('.position')).length, 19)
  assert.equal(clip.tracks.filter((track) => track.name.endsWith('.quaternion')).length, 19)
  assert.equal(clip.tracks.filter((track) => track.name.endsWith('.scale')).length, 19)
  assert.ok(clip.tracks.every((track) => track.times.length === 1309), 'every BrainStem track should contain 1309 keyframes')

  const movingQuaternionTrack = clip.tracks.find((track) => {
    if (!track.name.endsWith('.quaternion')) return false
    const valueSize = track.getValueSize()
    const lastOffset = track.values.length - valueSize
    for (let i = 0; i < valueSize; i += 1) {
      if (Math.abs(track.values[i] - track.values[lastOffset + i]) > 1e-4) return true
    }
    return false
  })
  assert.ok(movingQuaternionTrack, 'BrainStem should include at least one changing quaternion track')
  const animatedObject = gltf.scene.getObjectByProperty('uuid', movingQuaternionTrack.name.replace(/\.quaternion$/, ''))
  assert.equal(animatedObject?.isBone, true, 'changing BrainStem quaternion track should target a bone')

  const mixer = new THREE.AnimationMixer(gltf.scene)
  mixer.clipAction(clip).play()
  mixer.setTime(0)
  const startQuaternion = animatedObject.quaternion.clone()
  mixer.setTime(clip.duration * 0.75)
  const endQuaternion = animatedObject.quaternion.clone()
  assert.ok(startQuaternion.angleTo(endQuaternion) > 0.1, 'BrainStem animation should move the selected bone')

  gltf.scene.updateMatrixWorld(true)
  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1.0))
  const light = new THREE.DirectionalLight(0xffffff, 2.0)
  light.position.copy(center).add(new THREE.Vector3(2, 4, 5))
  gltf.scene.add(light)
  const halfHeight = Math.max(size.y, size.z) / 2 + 0.08
  const halfWidth = Math.max(size.x / 2 + 0.08, halfHeight)
  const camera = new THREE.OrthographicCamera(-halfWidth, halfWidth, halfHeight, -halfHeight, 0.01, 20)
  camera.position.copy(center).add(new THREE.Vector3(0, 0, 5))
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.07, 'BrainStem should render visible multi-primitive skinned geometry')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > mean.b, `BrainStem material colors should render warm highlights (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets CesiumMan fixture loads textured skinned character animation', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_CESIUM_MAN, 'utf8'))
  assert.deepEqual(source.buffers, [{ uri: 'CesiumMan_data.bin', byteLength: 252664 }])
  assert.deepEqual(source.images, [{ uri: 'CesiumMan_img0.jpg' }])
  assert.equal(source.meshes[0].name, 'Cesium_Man')
  assert.equal(source.skins.length, 1)
  assert.equal(source.skins[0].joints.length, 19)
  assert.equal(source.skins[0].skeleton, 3)
  assert.equal(source.animations.length, 1)
  assert.equal(source.animations[0].channels.length, 57)
  assert.equal(source.animations[0].samplers.length, 57)

  const gltf = await loadGltfFixture(SAMPLE_ASSET_CESIUM_MAN)
  const skinnedMeshes = []
  const bones = []
  gltf.scene.traverse((object) => {
    if (object.isSkinnedMesh === true) skinnedMeshes.push(object)
    if (object.isBone === true) bones.push(object)
  })

  assert.equal(skinnedMeshes.length, 1)
  const mesh = skinnedMeshes[0]
  assert.equal(mesh.name, 'Cesium_Man')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 3273)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 3273)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 3273)
  assert.equal(mesh.geometry.getAttribute('skinIndex')?.count, 3273)
  assert.equal(mesh.geometry.getAttribute('skinWeight')?.count, 3273)
  assert.equal(mesh.geometry.index?.count, 14016)
  assert.equal(mesh.material.name, 'Cesium_Man-effect')
  assert.equal(mesh.material.metalness, 0)
  assert.equal(mesh.material.roughness, 1)
  assert.equal(Buffer.isBuffer(mesh.material.map?.image), true, 'CesiumMan base-color JPEG should load as an encoded Buffer')
  assert.equal(mesh.material.map.name, 'CesiumMan_img0.jpg')
  assert.equal(mesh.material.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(mesh.skeleton.bones.length, 19)
  assert.deepEqual(bones.map((bone) => bone.name), [
    'Skeleton_torso_joint_1',
    'Skeleton_torso_joint_2',
    'torso_joint_3',
    'Skeleton_neck_joint_1',
    'Skeleton_neck_joint_2',
    'Skeleton_arm_joint_L__4_',
    'Skeleton_arm_joint_L__3_',
    'Skeleton_arm_joint_L__2_',
    'Skeleton_arm_joint_R',
    'Skeleton_arm_joint_R__2_',
    'Skeleton_arm_joint_R__3_',
    'leg_joint_L_1',
    'leg_joint_L_2',
    'leg_joint_L_3',
    'leg_joint_L_5',
    'leg_joint_R_1',
    'leg_joint_R_2',
    'leg_joint_R_3',
    'leg_joint_R_5',
  ])

  const clip = gltf.animations[0]
  assert.equal(clip.name, 'animation_0')
  assert.equal(clip.duration, 2)
  assert.equal(clip.tracks.length, 57)
  assert.equal(clip.tracks.filter((track) => track.name.endsWith('.position')).length, 19)
  assert.equal(clip.tracks.filter((track) => track.name.endsWith('.quaternion')).length, 19)
  assert.equal(clip.tracks.filter((track) => track.name.endsWith('.scale')).length, 19)
  assert.ok(clip.tracks.every((track) => track.times.length === 48), 'every CesiumMan track should contain 48 keyframes')

  const mixer = new THREE.AnimationMixer(gltf.scene)
  mixer.clipAction(clip).play()
  mixer.setTime(clip.duration / 2)
  gltf.scene.updateMatrixWorld(true)
  const torso = gltf.scene.getObjectByName('Skeleton_torso_joint_1')
  const rightArm = gltf.scene.getObjectByName('Skeleton_arm_joint_R__2_')
  const leftLeg = gltf.scene.getObjectByName('leg_joint_L_3')
  assert.ok(Math.abs(torso.position.y + 0.025) < 0.001, `CesiumMan torso should translate at mid animation (${torso.position.y})`)
  assert.ok(rightArm.quaternion.y > 0.9, `CesiumMan right arm should rotate at mid animation (${rightArm.quaternion.y})`)
  assert.ok(leftLeg.quaternion.y < -0.85, `CesiumMan left leg should rotate at mid animation (${leftLeg.quaternion.y})`)

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1.2))
  const light = new THREE.DirectionalLight(0xffffff, 1.5)
  light.position.set(2, 4, 5)
  gltf.scene.add(light)
  const camera = new THREE.PerspectiveCamera(35, 0.75, 0.01, 20)
  camera.position.set(0, -3, 1.4)
  camera.lookAt(0, 0, 0.7)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 96,
    height: 128,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.12, 'CesiumMan should render visible textured skinned character geometry')
})

test('committed Khronos glTF Sample Assets Cesium Milk Truck fixture loads textured wheel animation', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_CESIUM_MILK_TRUCK, 'utf8'))
  assert.deepEqual(source.buffers, [{ uri: 'CesiumMilkTruck_data.bin', byteLength: 146092 }])
  assert.deepEqual(source.images, [{ name: 'CesiumMilkTruck.jpg', uri: 'CesiumMilkTruck.jpg' }])
  assert.deepEqual(source.materials.map((material) => material.name), ['wheels', 'truck', 'glass', 'window_trim'])
  assert.deepEqual(source.meshes.map((mesh) => mesh.name), ['Wheels', 'Cesium_Milk_Truck'])
  assert.equal(source.animations.length, 1)
  assert.equal(source.animations[0].name, 'Wheels')
  assert.equal(source.animations[0].channels.length, 2)
  assert.equal(source.animations[0].samplers.length, 2)

  const gltf = await loadGltfFixture(SAMPLE_ASSET_CESIUM_MILK_TRUCK)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'Cesium_Milk_Truck_1',
    'Cesium_Milk_Truck_2',
    'Cesium_Milk_Truck_3',
    'Wheels',
    'Wheels001',
  ])
  assert.deepEqual(meshes.map((mesh) => mesh.material.name), [
    'truck',
    'glass',
    'window_trim',
    'wheels',
    'wheels',
  ])
  assert.deepEqual(
    meshes.map((mesh) => [
      mesh.geometry.getAttribute('position')?.count,
      mesh.geometry.getAttribute('normal')?.count,
      mesh.geometry.getAttribute('uv')?.count,
      mesh.geometry.index?.count,
    ]),
    [
      [2366, 2366, 2366, 5232],
      [151, 151, 151, 168],
      [650, 650, 650, 864],
      [828, 828, 828, 2304],
      [828, 828, 828, 2304],
    ],
  )

  const truck = meshes.find((mesh) => mesh.material.name === 'truck')
  const wheels = meshes.find((mesh) => mesh.name === 'Wheels')
  for (const mesh of [truck, wheels]) {
    const texture = mesh.material.map
    assert.ok(texture?.isTexture, `${mesh.material.name} should load a base-color texture`)
    assert.equal(texture.name, 'CesiumMilkTruck.jpg')
    assert.equal(Buffer.isBuffer(texture.image), true, 'CesiumMilkTruck JPEG should load as an encoded Buffer')
    assert.equal(texture.image[0], 0xff, 'CesiumMilkTruck texture should start with a JPEG marker')
    assert.equal(texture.image[1], 0xd8, 'CesiumMilkTruck texture should start with a JPEG marker')
    assert.equal(texture.colorSpace, THREE.SRGBColorSpace)
    assert.equal(texture.flipY, false)
    assert.equal(mesh.material.metalness, 0)
    assert.equal(mesh.material.roughness, 1)
  }

  const glass = meshes.find((mesh) => mesh.material.name === 'glass')
  assert.deepEqual(glass.material.color.toArray().map((value) => Number(value.toFixed(6))), [0, 0.040506, 0.021241])
  const trim = meshes.find((mesh) => mesh.material.name === 'window_trim')
  assert.deepEqual(trim.material.color.toArray().map((value) => Number(value.toFixed(6))), [0.064, 0.064, 0.064])

  assert.equal(gltf.animations.length, 1)
  const clip = gltf.animations[0]
  assert.equal(clip.name, 'Wheels')
  assert.equal(clip.duration, 1.25)
  assert.deepEqual(clip.tracks.map((track) => track.name), ['Wheels.quaternion', 'Wheels001.quaternion'])
  assert.ok(clip.tracks.every((track) => track.times.length === 31), 'every CesiumMilkTruck track should contain 31 keyframes')

  const mixer = new THREE.AnimationMixer(gltf.scene)
  mixer.clipAction(clip).play()
  mixer.setTime(clip.duration / 2)
  gltf.scene.updateMatrixWorld(true)
  assert.ok(gltf.scene.getObjectByName('Wheels').quaternion.y > 0.99, 'CesiumMilkTruck front wheels should rotate at mid animation')
  assert.ok(gltf.scene.getObjectByName('Wheels001').quaternion.y > 0.99, 'CesiumMilkTruck rear wheels should rotate at mid animation')

  const box = new THREE.Box3().setFromObject(gltf.scene)
  const center = box.getCenter(new THREE.Vector3())
  const renderCamera = new THREE.PerspectiveCamera(35, 1, 0.01, 100)
  renderCamera.position.copy(center).add(new THREE.Vector3(0.9, 0.45, 1).normalize().multiplyScalar(7))
  renderCamera.lookAt(center)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.9))
  const light = new THREE.DirectionalLight(0xffffff, 1.8)
  light.position.copy(center).add(new THREE.Vector3(4, 5, 6))
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.4, 'CesiumMilkTruck should render visible textured vehicle geometry')
  const mean = meanRgba(rgba)
  assert.ok(mean.g > mean.r && mean.b > mean.r, `CesiumMilkTruck texture should contribute cool label colors (${mean.r}, ${mean.g}, ${mean.b})`)
})
