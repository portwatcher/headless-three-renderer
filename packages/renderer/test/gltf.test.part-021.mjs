import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'
import { Renderer, SAMPLE_ASSET_ANIMATED_MORPH_CUBE, SAMPLE_ASSET_FOX, SAMPLE_ASSET_MORPH_STRESS_TEST, SAMPLE_ASSET_RECURSIVE_SKELETONS, SAMPLE_ASSET_SIMPLE_INSTANCING, SAMPLE_ASSET_SIMPLE_MORPH, SAMPLE_ASSET_SIMPLE_SPARSE_ACCESSOR } from './gltf.test.part-001.mjs'
import { findFirst, loadGltfFixture, nonBackgroundBounds, pngDimensions, vectorFromAttribute } from './gltf.test.part-028.mjs'
test('committed Khronos glTF Sample Assets Fox fixture loads textured multi-clip skinned animal animation', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_FOX, 'utf8'))
  assert.deepEqual(source.buffers, [{ uri: 'Fox.bin', byteLength: 119904 }])
  assert.deepEqual(source.images, [{ uri: 'Texture.png', mimeType: 'image/png' }])
  assert.equal(source.meshes[0].name, 'fox1')
  assert.equal(source.skins.length, 1)
  assert.equal(source.skins[0].joints.length, 24)
  assert.equal(source.skins[0].skeleton, 2)
  assert.deepEqual(source.animations.map((animation) => animation.name), ['Survey', 'Walk', 'Run'])
  assert.ok(source.animations.every((animation) => animation.channels.length === 21), 'every Fox animation should have 21 channels')

  const gltf = await loadGltfFixture(SAMPLE_ASSET_FOX)
  const skinnedMeshes = []
  const bones = []
  gltf.scene.traverse((object) => {
    if (object.isSkinnedMesh === true) skinnedMeshes.push(object)
    if (object.isBone === true) bones.push(object)
  })

  assert.equal(skinnedMeshes.length, 1)
  const mesh = skinnedMeshes[0]
  assert.equal(mesh.name, 'fox')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 1728)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 1728)
  assert.equal(mesh.geometry.getAttribute('skinIndex')?.count, 1728)
  assert.equal(mesh.geometry.getAttribute('skinWeight')?.count, 1728)
  assert.equal(mesh.geometry.getAttribute('normal') ?? null, null)
  assert.equal(mesh.geometry.index ?? null, null)
  assert.equal(mesh.material.name, 'fox_material')
  assert.equal(mesh.material.metalness, 0)
  assert.equal(mesh.material.roughness, 0.58)
  assert.equal(Buffer.isBuffer(mesh.material.map?.image), true, 'Fox base-color PNG should load as an encoded Buffer')
  assert.equal(mesh.material.map.name, 'Texture.png')
  assert.deepEqual(pngDimensions(mesh.material.map.image), [1024, 1024])
  assert.equal(mesh.material.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(mesh.skeleton.bones.length, 24)
  assert.deepEqual(bones.map((bone) => bone.name), [
    '_rootJoint',
    'b_Root_00',
    'b_Hip_01',
    'b_Spine01_02',
    'b_Spine02_03',
    'b_Neck_04',
    'b_Head_05',
    'b_RightUpperArm_06',
    'b_RightForeArm_07',
    'b_RightHand_08',
    'b_LeftUpperArm_09',
    'b_LeftForeArm_010',
    'b_LeftHand_011',
    'b_Tail01_012',
    'b_Tail02_013',
    'b_Tail03_014',
    'b_LeftLeg01_015',
    'b_LeftLeg02_016',
    'b_LeftFoot01_017',
    'b_LeftFoot02_018',
    'b_RightLeg01_019',
    'b_RightLeg02_020',
    'b_RightFoot01_021',
    'b_RightFoot02_022',
  ])

  assert.deepEqual(gltf.animations.map((clip) => clip.name), ['Survey', 'Walk', 'Run'])
  assert.deepEqual(gltf.animations.map((clip) => clip.tracks.length), [21, 21, 21])
  assert.deepEqual(gltf.animations.map((clip) => Number(clip.duration.toFixed(6))), [3.416667, 0.708333, 1.158333])
  for (const clip of gltf.animations) {
    assert.equal(clip.tracks.filter((track) => track.name.endsWith('.quaternion')).length, 20)
    assert.equal(clip.tracks.filter((track) => track.name.endsWith('.position')).length, 1)
  }

  const runClip = gltf.animations.find((clip) => clip.name === 'Run')
  const mixer = new THREE.AnimationMixer(gltf.scene)
  mixer.clipAction(runClip).play()
  mixer.setTime(runClip.duration / 2)
  gltf.scene.updateMatrixWorld(true)
  const head = gltf.scene.getObjectByName('b_Head_05')
  const tail = gltf.scene.getObjectByName('b_Tail02_013')
  const leftLeg = gltf.scene.getObjectByName('b_LeftLeg02_016')
  assert.ok(Math.abs(head.quaternion.z) > 0.1, `Fox run pose should rotate the head (${head.quaternion.z})`)
  assert.ok(tail.quaternion.z > 0.06, `Fox run pose should rotate the tail (${tail.quaternion.z})`)
  assert.ok(leftLeg.quaternion.z < -0.8, `Fox run pose should rotate the left leg (${leftLeg.quaternion.z})`)

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1.4))
  const light = new THREE.DirectionalLight(0xffffff, 2)
  light.position.set(2, 4, 5)
  gltf.scene.add(light)
  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const halfHeight = Math.max(size.x, size.y, size.z) / 2 + 5
  const halfWidth = halfHeight * 1.4
  const camera = new THREE.OrthographicCamera(-halfWidth, halfWidth, halfHeight, -halfHeight, 0.01, 500)
  camera.position.set(center.x + 150, center.y + 90, center.z + 120)
  camera.lookAt(center)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 140,
    height: 100,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.08, 'Fox should render visible textured skinned animal geometry')
})

test('committed Khronos glTF Sample Assets RecursiveSkeletons fixture loads recursive skinned hierarchies', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_RECURSIVE_SKELETONS, 'utf8'))
  assert.equal(source.buffers[0].uri, 'RecursiveSkeletons.bin')
  assert.equal(source.nodes.length, 924)
  assert.equal(source.skins.length, 84)
  assert.equal(source.skins.every((skin) => skin.joints.length === 10), true)
  assert.equal(source.animations[0].channels.length, 840)
  assert.equal(source.animations[0].samplers.length, 840)

  const gltf = await loadGltfFixture(SAMPLE_ASSET_RECURSIVE_SKELETONS)
  const skinnedMeshes = []
  const bones = []
  gltf.scene.traverse((object) => {
    if (object.isSkinnedMesh === true) skinnedMeshes.push(object)
    if (object.isBone === true) bones.push(object)
  })

  assert.equal(skinnedMeshes.length, 84, 'RecursiveSkeletons should load every skinned mesh instance')
  assert.equal(bones.length, 840, 'RecursiveSkeletons should load every recursive bone')
  assert.equal(gltf.animations.length, 1)
  assert.equal(gltf.animations[0].tracks.length, 840)

  const first = skinnedMeshes[0]
  assert.equal(first.name, 'skinned_mesh_instance_0')
  assert.equal(first.skeleton.bones.length, 10)
  assert.equal(first.geometry.getAttribute('position')?.count, 40)
  assert.equal(first.geometry.getAttribute('skinIndex')?.count, 40)
  assert.equal(first.geometry.getAttribute('skinWeight')?.count, 40)
  assert.equal(first.geometry.index?.count, 228)
  assert.equal(first.geometry.getAttribute('color')?.itemSize, 4)
  assert.equal(first.geometry.getAttribute('color')?.normalized, true)
  assert.equal(first.material.vertexColors, true)

  const mixer = new THREE.AnimationMixer(gltf.scene)
  mixer.clipAction(gltf.animations[0]).play()
  const skinnedBounds = (time) => {
    mixer.setTime(time)
    gltf.scene.updateMatrixWorld(true)
    const box = new THREE.Box3()
    const vertex = new THREE.Vector3()
    for (const mesh of skinnedMeshes) {
      assert.equal(typeof mesh.applyBoneTransform, 'function', `${mesh.name} should expose Three.js skinning transforms`)
      mesh.skeleton.update()
      const position = mesh.geometry.getAttribute('position')
      for (let i = 0; i < position.count; i += 1) {
        vertex.fromBufferAttribute(position, i)
        mesh.applyBoneTransform(i, vertex)
        vertex.applyMatrix4(mesh.matrixWorld)
        box.expandByPoint(vertex)
      }
    }
    return box.getSize(new THREE.Vector3())
  }

  const baseSize = skinnedBounds(0)
  const animatedSize = skinnedBounds(1)
  assert.ok(baseSize.x < 70 && baseSize.z < 70, `RecursiveSkeletons base pose should stay compact (${baseSize.x}, ${baseSize.z})`)
  assert.ok(animatedSize.x > baseSize.x + 100, `RecursiveSkeletons animation should spread recursively in X (${animatedSize.x} vs ${baseSize.x})`)
  assert.ok(animatedSize.z > baseSize.z + 100, `RecursiveSkeletons animation should spread recursively in Z (${animatedSize.z} vs ${baseSize.z})`)

  mixer.setTime(0)
  const camera = new THREE.OrthographicCamera(-80, 80, 140, -20, 0.1, 500)
  camera.position.set(0, 62, 180)
  camera.lookAt(0, 62, 0)
  camera.updateProjectionMatrix()
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.04, 'RecursiveSkeletons should render visible normalized-color skinned meshes')
})

test('committed Khronos glTF Sample Assets SimpleMorph fixture applies morph weight animation', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_SIMPLE_MORPH)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'Khronos SimpleMorph sample should load a mesh')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 3)
  assert.equal(mesh.geometry.index?.count, 3)
  assert.equal(mesh.geometry.morphAttributes.position?.length, 2)
  assert.deepEqual(mesh.morphTargetInfluences, [0.5, 0.5])
  assert.equal(gltf.animations.length, 1)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 10)
  camera.position.set(0.5, 0.5, 3.2)
  camera.lookAt(0.45, 0.4, 0)
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

  mesh.morphTargetInfluences = [0, 0]
  gltf.scene.updateMatrixWorld(true)
  const base = renderBounds()

  const mixer = new THREE.AnimationMixer(gltf.scene)
  mixer.clipAction(gltf.animations[0]).play()
  mixer.setTime(2)
  gltf.scene.updateMatrixWorld(true)
  const animated = renderBounds()

  assert.ok(base.height > 10, `SimpleMorph base triangle should render visible bounds (${base.height})`)
  assert.ok(animated.height > base.height + 35, `SimpleMorph animation should expand rendered height (${animated.height} vs ${base.height})`)
  assert.ok(animated.minY < base.minY - 35, `SimpleMorph animation should lift the triangle top (${animated.minY} vs ${base.minY})`)
})

test('committed Khronos glTF Sample Assets AnimatedMorphCube fixture applies animated morph normals', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_ANIMATED_MORPH_CUBE)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'Khronos AnimatedMorphCube sample should load a mesh')
  assert.equal(mesh.name, 'AnimatedMorphCube')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 24)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 24)
  assert.equal(mesh.geometry.getAttribute('tangent')?.count, 24)
  assert.equal(mesh.geometry.index?.count, 36)
  assert.equal(mesh.geometry.morphAttributes.position?.length, 2)
  assert.equal(mesh.geometry.morphAttributes.normal?.length, 2)
  assert.deepEqual(mesh.morphTargetInfluences, [0, 0])
  assert.deepEqual(mesh.morphTargetDictionary, { 0: 0, 1: 1 })
  assert.equal(gltf.animations.length, 1)
  assert.equal(gltf.animations[0].name, 'Square')
  assert.equal(gltf.animations[0].tracks[0].name, 'AnimatedMorphCube.morphTargetInfluences')
  assert.equal(gltf.animations[0].tracks[0].getInterpolation(), THREE.InterpolateLinear)
  assert.equal(gltf.animations[0].tracks[0].getValueSize(), 2)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 30)
  camera.position.set(3, 2.5, 5)
  camera.lookAt(0, 0, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 1.4)
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
    outputColorSpace: THREE.LinearSRGBColorSpace,
  }), 96, 96, [0, 0, 0], 3)

  const base = renderBounds()
  const mixer = new THREE.AnimationMixer(gltf.scene)
  mixer.clipAction(gltf.animations[0]).play()
  mixer.setTime(2)
  gltf.scene.updateMatrixWorld(true)
  const animated = renderBounds()

  assert.ok(mesh.morphTargetInfluences[0] > 0.75, `AnimatedMorphCube first morph target should be active at t=2 (${mesh.morphTargetInfluences[0]})`)
  assert.ok(mesh.morphTargetInfluences[1] > 0.15, `AnimatedMorphCube second morph target should be active at t=2 (${mesh.morphTargetInfluences[1]})`)
  assert.ok(base.width > 45 && base.height > 45, `AnimatedMorphCube base pose should render a broad cube (${base.width}x${base.height})`)
  assert.ok(animated.width < base.width - 10, `AnimatedMorphCube morph animation should narrow the rendered cube (${animated.width} vs ${base.width})`)
  assert.ok(animated.height < base.height - 10, `AnimatedMorphCube morph animation should shorten the rendered cube (${animated.height} vs ${base.height})`)
})

test('committed Khronos glTF Sample Assets MorphStressTest fixture loads dense morph weight tracks', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_MORPH_STRESS_TEST, 'utf8'))
  assert.equal(source.buffers[0].uri, 'MorphStressTest.bin')
  assert.equal(source.buffers[0].byteLength, 388084)
  assert.deepEqual(source.images.map((image) => image.uri), [
    'Base_AO.png',
    'TinyGrid.png',
    'ColorSwatches.png',
  ])
  const primitive = source.meshes[0].primitives[0]
  assert.equal(primitive.targets.length, 8)
  assert.equal(source.meshes[0].weights.length, 8)
  assert.deepEqual(source.animations.map((animation) => [
    animation.name,
    animation.channels.length,
    animation.samplers.length,
    animation.channels[0].target.path,
  ]), [
    ['Individuals', 1, 1, 'weights'],
    ['TheWave', 1, 1, 'weights'],
    ['Pulse', 1, 1, 'weights'],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_MORPH_STRESS_TEST)
  const mesh = gltf.scene.getObjectByName('Cube')
  assert.ok(mesh?.isMesh, 'MorphStressTest should load its cube mesh')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 24)
  assert.equal(mesh.geometry.index?.count, 36)
  assert.equal(mesh.geometry.morphAttributes.position?.length, 8)
  assert.equal(mesh.geometry.morphAttributes.normal?.length, 8)
  assert.deepEqual(mesh.geometry.morphAttributes.position.map((attribute) => attribute.count), [24, 24, 24, 24, 24, 24, 24, 24])
  assert.deepEqual(mesh.morphTargetInfluences, [0, 0, 0, 0, 0, 0, 0, 0])
  assert.deepEqual(mesh.morphTargetDictionary, {
    'Key 1': 0,
    'Key 2': 1,
    'Key 3': 2,
    'Key 4': 3,
    'Key 5': 4,
    'Key 6': 5,
    'Key 7': 6,
    'Key 8': 7,
  })

  assert.equal(Buffer.isBuffer(mesh.material.map?.image), true, 'MorphStressTest tiny grid PNG should load as an encoded Buffer')
  assert.equal(mesh.material.map.name, 'TinyGrid')
  assert.deepEqual(pngDimensions(mesh.material.map.image), [64, 64])
  assert.equal(mesh.material.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(mesh.material.map.flipY, false)
  assert.equal(Buffer.isBuffer(mesh.material.aoMap?.image), true, 'MorphStressTest AO PNG should load as an encoded Buffer')
  assert.equal(mesh.material.aoMap.name, 'Base_AO')
  assert.deepEqual(pngDimensions(mesh.material.aoMap.image), [1024, 1024])
  assert.equal(mesh.material.aoMap.colorSpace, THREE.NoColorSpace)
  assert.equal(mesh.material.aoMap.flipY, false)

  assert.deepEqual(gltf.animations.map((clip) => [clip.name, clip.tracks.length]), [
    ['Individuals', 2],
    ['TheWave', 2],
    ['Pulse', 2],
  ])
  const waveTrack = gltf.animations.find((clip) => clip.name === 'TheWave').tracks[0]
  assert.equal(waveTrack.name, 'Cube.morphTargetInfluences')
  assert.equal(waveTrack.getValueSize(), 8)
  assert.equal(waveTrack.times.length, 59)
  assert.equal(waveTrack.values.length, 472)
  const pulseTrack = gltf.animations.find((clip) => clip.name === 'Pulse').tracks[0]
  assert.equal(pulseTrack.times.length, 191)
  assert.equal(pulseTrack.values.length, 1528)

  const mixer = new THREE.AnimationMixer(gltf.scene)
  const wave = gltf.animations.find((clip) => clip.name === 'TheWave')
  mixer.clipAction(wave).play()
  mixer.setTime(wave.duration / 2)
  assert.ok(mesh.morphTargetInfluences[3] > 0.9, `MorphStressTest wave should strongly activate middle targets (${mesh.morphTargetInfluences.join(', ')})`)
  assert.ok(mesh.morphTargetInfluences[0] < 0.1 && mesh.morphTargetInfluences[7] < 0.1, `MorphStressTest wave should leave edge targets low (${mesh.morphTargetInfluences.join(', ')})`)
  mixer.stopAllAction()
  mesh.morphTargetInfluences.fill(0)
  const pulse = gltf.animations.find((clip) => clip.name === 'Pulse')
  mixer.clipAction(pulse).play()
  mixer.setTime(pulse.duration / 2)
  assert.deepEqual(mesh.morphTargetInfluences, [1, 1, 1, 1, 1, 1, 1, 1])

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.7))
  const light = new THREE.DirectionalLight(0xffffff, 1.5)
  light.position.set(3, 4, 5)
  gltf.scene.add(light)
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 50)
  camera.position.copy(center).add(new THREE.Vector3(0, -8, 5))
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.1, 'MorphStressTest should render visible morphed geometry')
})

test('committed Khronos glTF Sample Assets SimpleSparseAccessor fixture applies sparse POSITION overrides', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_SIMPLE_SPARSE_ACCESSOR)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'Khronos SimpleSparseAccessor sample should load a mesh')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 14)
  assert.equal(mesh.geometry.index?.count, 36)

  const position = mesh.geometry.getAttribute('position')
  assert.deepEqual(vectorFromAttribute(position, 8), [1, 2, 0])
  assert.deepEqual(vectorFromAttribute(position, 10), [3, 3, 0])
  assert.deepEqual(vectorFromAttribute(position, 12), [5, 4, 0])

  const camera = new THREE.OrthographicCamera(-0.5, 6.5, 4.5, -0.5, 0.01, 10)
  camera.position.set(3, 2, 5)
  camera.lookAt(3, 2, 0)
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.05, 'SimpleSparseAccessor sample should render visible sparse geometry')
})

test('committed Khronos glTF Sample Assets SimpleInstancing fixture loads EXT_mesh_gpu_instancing', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_SIMPLE_INSTANCING)
  const mesh = findFirst(gltf.scene, (object) => object.isInstancedMesh === true)
  assert.ok(mesh, 'Khronos SimpleInstancing sample should load an InstancedMesh')
  assert.equal(mesh.count, 125)
  assert.equal(mesh.geometry.getAttribute('position')?.count, 24)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 24)
  assert.equal(mesh.geometry.index?.count, 36)
  assert.ok(mesh.instanceMatrix?.isInstancedBufferAttribute, 'EXT_mesh_gpu_instancing should populate instance matrices')
  assert.deepEqual(Array.from(mesh.instanceMatrix.array.slice(0, 16)), [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ])
  assert.deepEqual(Array.from(mesh.instanceMatrix.array.slice(124 * 16, 125 * 16)), [
    0, 2, 0, 0,
    0, 0, 2, 0,
    2, 0, 0, 0,
    10, 10, 10, 1,
  ])

  const camera = new THREE.OrthographicCamera(-1, 12, 12, -1, 0.01, 50)
  camera.position.set(6, 6, 20)
  camera.lookAt(6, 6, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 1.2)
  light.position.set(10, 12, 20)
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.2, 'SimpleInstancing sample should render visible instanced geometry')
})
