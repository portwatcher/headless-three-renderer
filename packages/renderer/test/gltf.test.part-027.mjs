import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'
import { MORPHED_TRIANGLE, REAL_VRM_SEED_SAN_SAMPLE, Renderer, SAMPLE_ASSET_SPONZA, SAMPLE_ASSET_XMP_METADATA_ROUNDED_CUBE, SKINNED_QUAD, SYNTHETIC_HUMANOID_VRM, SYNTHETIC_HUMANOID_VRMA, SYNTHETIC_VRM, SYNTHETIC_VRMA, TEXTURED_QUAD, VERTEX_COLOR_QUAD, loadVrmAnimationFromFile, loadVrmFromFile } from './gltf.test.part-001.mjs'
import { assertRejectsMutatedGltfSource, assertTexturedQuadLoadsEncodedMap, assertTexturedQuadRendersTexture, assertVectorClose, buildTexturedQuadGlb, findFirst, frameSceneCamera, loadGltfFixture, meanRegion, nonBackgroundBounds, pngDimensions, uniqueMaterials } from './gltf.test.part-028.mjs'
test('committed Khronos glTF Sample Assets Sponza fixture loads large textured architectural scene', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_SPONZA, 'utf8'))
  assert.deepEqual(source.buffers, [{ uri: 'Sponza.bin', byteLength: 9528220 }])
  assert.equal(source.nodes.length, 1)
  assert.equal(source.meshes.length, 1)
  assert.equal(source.materials.length, 25)
  assert.equal(source.textures.length, 69)
  assert.equal(source.images.length, 69)
  assert.deepEqual(source.materials[0].pbrMetallicRoughness.baseColorFactor, [
    0.5879999995231628,
    0.5879999995231628,
    0.5879999995231628,
    1,
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_SPONZA)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 103)
  assert.equal(uniqueMaterials(gltf.scene).length, 25)

  const first = gltf.scene.getObjectByName('mesh_0')
  assert.ok(first?.isMesh, 'Sponza should load split mesh_0 geometry')
  assert.equal(first.geometry.getAttribute('position')?.count, 3175)
  assert.equal(first.geometry.getAttribute('uv')?.count, 3175)
  assert.equal(first.geometry.index?.count, 10920)
  assert.equal(first.material.isMeshStandardMaterial, true)
  assert.equal(first.material.map?.name, '5061699253647017043.png')
  assert.deepEqual(pngDimensions(first.material.map.image), [1024, 1024])
  assert.equal(first.material.normalMap?.name, '8773302468495022225.jpg')
  assert.equal(first.material.roughnessMap?.name, '11872827283454512094.jpg')
  assert.equal(first.material.metalnessMap, first.material.roughnessMap)

  const second = gltf.scene.getObjectByName('mesh_0_1')
  assert.equal(second?.geometry.getAttribute('position')?.count, 533)
  assert.equal(second.geometry.index?.count, 1404)
  assert.equal(second.material.map?.name, '8006627369776289000.png')
  assert.deepEqual(pngDimensions(second.material.map.image), [1024, 1024])
  assert.equal(second.material.normalMap?.name, '12501374198249454378.jpg')

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.6))
  const light = new THREE.DirectionalLight(0xffffff, 1.8)
  light.position.set(2, 4, 5)
  gltf.scene.add(light)
  const camera = frameSceneCamera(gltf.scene, { distance: 1.8, yOffset: 0.1 })
  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [1, 1, 1],
  })
  assert.ok(nonBackgroundRatio(rgba, [255, 255, 255], 3) > 0.5, 'Sponza should render visible architectural geometry')
  const mean = meanRgba(rgba)
  assert.ok(mean.r < 220 && mean.g < 220 && mean.b < 220, `Sponza should render textured architecture pixels (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed textured glTF fixture loads data URI image and renders texture', async () => {
  const gltf = await loadGltfFixture(TEXTURED_QUAD)

  assertTexturedQuadLoadsEncodedMap(gltf, 'textured fixture')
  assertTexturedQuadRendersTexture(gltf, 'textured quad')
})

test('loadGltfFromFile loads helper-normalized GLB bufferView images', async () => {
  const source = JSON.parse(await readFile(TEXTURED_QUAD, 'utf8'))
  const glbBytes = buildTexturedQuadGlb(source)
  const tmp = await mkdtemp(path.join(os.tmpdir(), 'headless-three-glb-image-'))
  try {
    const modelPath = path.join(tmp, 'buffer-view-image.glb')
    await writeFile(modelPath, glbBytes)

    const gltf = await loadGltfFixture(modelPath)
    assertTexturedQuadLoadsEncodedMap(gltf, 'GLB bufferView-image fixture')
    assertTexturedQuadRendersTexture(gltf, 'GLB bufferView-image quad')
  } finally {
    await rm(tmp, { recursive: true, force: true })
  }
})

test('loadGltfFromFile rejects compressed GLB bufferView images with pre-decode guidance', async () => {
  const source = JSON.parse(await readFile(TEXTURED_QUAD, 'utf8'))
  source.images[0].mimeType = 'image/ktx2'
  const glbBytes = buildTexturedQuadGlb(source)
  const tmp = await mkdtemp(path.join(os.tmpdir(), 'headless-three-glb-compressed-image-'))
  try {
    const modelPath = path.join(tmp, 'compressed-buffer-view-image.glb')
    await writeFile(modelPath, glbBytes)

    await assert.rejects(
      () => loadGltfFixture(modelPath),
      /GLB bufferView image.*compressed texture.*KTX2.*Basis.*pre-decode/i,
    )
  } finally {
    await rm(tmp, { recursive: true, force: true })
  }
})

test('loadGltfFromFile rejects external compressed glTF image references with pre-decode guidance', async () => {
  const source = JSON.parse(await readFile(TEXTURED_QUAD, 'utf8'))
  source.images[0].uri = 'textures/albedo.ktx2'
  const tmp = await mkdtemp(path.join(os.tmpdir(), 'headless-three-gltf-compressed-image-'))
  try {
    const modelPath = path.join(tmp, 'compressed-image-reference.gltf')
    await writeFile(modelPath, JSON.stringify(source))

    await assert.rejects(
      () => loadGltfFixture(modelPath),
      /glTF image URI.*compressed texture.*KTX2.*Basis.*pre-decode/i,
    )
  } finally {
    await rm(tmp, { recursive: true, force: true })
  }
})

test('loadGltfFromFile rejects malformed glTF image metadata clearly', async () => {
  await assertRejectsMutatedGltfSource((source) => {
    source.images = 'images'
  }, /glTF\.images must be an array/i)

  await assertRejectsMutatedGltfSource((source) => {
    source.images[0] = 'image'
  }, /glTF\.images\[0\] must be an object/i)

  await assertRejectsMutatedGltfSource((source) => {
    source.images[0] = { bufferView: 0 }
  }, /glTF bufferView image is missing mimeType/i)

  await assertRejectsMutatedGltfSource((source) => {
    source.images[0] = { bufferView: 0, mimeType: 'image/ktx2' }
  }, /glTF bufferView image.*compressed texture.*KTX2.*Basis.*pre-decode/i)
})

test('committed vertex-color glTF fixture renders COLOR_0 attributes', async () => {
  const gltf = await loadGltfFixture(VERTEX_COLOR_QUAD)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'vertex-color fixture should load a mesh')
  assert.equal(mesh.geometry.getAttribute('color')?.count, 4)
  assert.equal(mesh.material.vertexColors, true)

  const camera = gltf.cameras[0]
  assert.ok(camera, 'vertex-color fixture should load a camera')
  camera.aspect = 1
  camera.updateProjectionMatrix()
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })
  assert.equal(rgba.length, 96 * 96 * 4)
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.25, 'vertex-color quad should render visible pixels')

  const left = meanRegion(rgba, 96, 96, 24, 36, 42, 60)
  const right = meanRegion(rgba, 96, 96, 54, 36, 72, 60)
  assert.ok(left.r > left.g + 60, `left half should be dominated by COLOR_0 red (${left.r} vs ${left.g})`)
  assert.ok(right.g > right.r + 60, `right half should be dominated by COLOR_0 green (${right.g} vs ${right.r})`)
})

test('committed morph-target glTF fixture applies POSITION targets', async () => {
  const gltf = await loadGltfFixture(MORPHED_TRIANGLE)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'morph fixture should load a mesh')
  assert.equal(mesh.geometry.morphAttributes.position?.length, 1)
  assert.equal(mesh.morphTargetInfluences?.length, 1)

  const camera = gltf.cameras[0]
  assert.ok(camera, 'morph fixture should load a camera')
  camera.aspect = 1
  camera.updateProjectionMatrix()
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  function renderBounds(influence) {
    mesh.morphTargetInfluences[0] = influence
    const rgba = new Renderer().render(gltf.scene, camera, {
      width: 96,
      height: 96,
      format: 'rgba',
      background: [0, 0, 0],
      outputColorSpace: THREE.LinearSRGBColorSpace,
    })
    return nonBackgroundBounds(rgba, 96, 96, [0, 0, 0], 3)
  }

  const flat = renderBounds(0)
  const morphed = renderBounds(1)
  assert.ok(flat.height > 10, `flat triangle should render visible bounds (${flat.height})`)
  assert.ok(morphed.minY < flat.minY - 12, `morph target should move the triangle top upward (${morphed.minY} vs ${flat.minY})`)
  assert.ok(morphed.height > flat.height + 10, `morph target should expand rendered height (${morphed.height} vs ${flat.height})`)
})

test('committed skinned glTF fixture applies JOINTS_0 and WEIGHTS_0 attributes', async () => {
  const gltf = await loadGltfFixture(SKINNED_QUAD)
  const mesh = findFirst(gltf.scene, (object) => object.isSkinnedMesh === true)
  assert.ok(mesh, 'skinned fixture should load a SkinnedMesh')
  assert.equal(mesh.geometry.getAttribute('skinIndex')?.count, 4)
  assert.equal(mesh.geometry.getAttribute('skinWeight')?.count, 4)
  assert.equal(mesh.skeleton.bones.length, 1)

  const camera = gltf.cameras[0]
  assert.ok(camera, 'skinned fixture should load a camera')
  camera.aspect = 1
  camera.updateProjectionMatrix()
  camera.updateMatrixWorld(true)

  function renderBounds(jointY) {
    mesh.skeleton.bones[0].position.y = jointY
    gltf.scene.updateMatrixWorld(true)
    const rgba = new Renderer().render(gltf.scene, camera, {
      width: 96,
      height: 96,
      format: 'rgba',
      background: [0, 0, 0],
      outputColorSpace: THREE.LinearSRGBColorSpace,
    })
    return nonBackgroundBounds(rgba, 96, 96, [0, 0, 0], 3)
  }

  const base = renderBounds(0)
  const moved = renderBounds(0.55)
  assert.ok(base.height > 20, `base skinned quad should render visible bounds (${base.height})`)
  assert.ok(moved.minY < base.minY - 12, `joint translation should move the skinned quad upward (${moved.minY} vs ${base.minY})`)
  assert.ok(Math.abs(moved.height - base.height) <= 4, `single-joint translation should preserve quad height (${moved.height} vs ${base.height})`)
})

test('committed Khronos glTF Sample Assets XmpMetadataRoundedCube fixture preserves XMP extension metadata and split buffers', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_XMP_METADATA_ROUNDED_CUBE, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_xmp_json_ld'])
  assert.deepEqual(source.asset.extensions, { KHR_xmp_json_ld: { packet: 0 } })
  assert.equal(source.extensions.KHR_xmp_json_ld.packets[0]['dc:title']['rdf:_1']['@value'], 'Sample glTF with XMP metadata')
  assert.equal(source.extensions.KHR_xmp_json_ld.packets[1]['dc:title']['rdf:_1']['@value'], 'My Cube Mesh')
  assert.deepEqual(source.buffers.map((buffer) => buffer.uri), [
    'MODEL_ROUNDED_CUBE_PART_1/positions.bin',
    'MODEL_ROUNDED_CUBE_PART_1/normals.bin',
    'MODEL_ROUNDED_CUBE_PART_1/indices.bin',
  ])
  assert.deepEqual(source.buffers.map((buffer) => buffer.byteLength), [41472, 41472, 20688])
  assert.deepEqual(source.meshes[0].extensions, { KHR_xmp_json_ld: { packet: 1 } })

  const gltf = await loadGltfFixture(SAMPLE_ASSET_XMP_METADATA_ROUNDED_CUBE)
  assert.deepEqual(gltf.parser.json.extensionsUsed, ['KHR_xmp_json_ld'])
  assert.deepEqual(gltf.parser.json.asset.extensions, { KHR_xmp_json_ld: { packet: 0 } })
  const mesh = gltf.scene.getObjectByName('MODEL_ROUNDED_CUBE_PART_1model_N3D')
  assert.ok(mesh?.isMesh, 'XmpMetadataRoundedCube should load its rounded cube mesh')
  assert.deepEqual(mesh.userData.gltfExtensions, { KHR_xmp_json_ld: { packet: 1 } })
  assert.equal(mesh.geometry.getAttribute('position')?.count, 3456)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 3456)
  assert.equal(mesh.geometry.index?.count, 5172)
  assert.equal(mesh.material.name, 'Rounded Cube Material')
  assert.equal(mesh.material.side, THREE.DoubleSide)
  assertVectorClose(mesh.material.color.toArray(), [0.6307567954063416, 0.6307567954063416, 0.6307567954063416], 'XmpMetadataRoundedCube material color')
  assert.equal(mesh.material.metalness, 0)
  assert.equal(mesh.material.roughness, 0.503000020980835)

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 1.7)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)
  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const halfExtent = Math.max(size.x, size.y, size.z) / 2 + 0.1
  const camera = new THREE.OrthographicCamera(-halfExtent, halfExtent, halfExtent, -halfExtent, 0.01, 20)
  camera.position.set(center.x + 2, center.y + 2, center.z + 3)
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
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.9, 'XmpMetadataRoundedCube should render visible rounded cube geometry')
})

test('VRM loader helpers register supplied Pixiv-style plugins', async () => {
  let vrmPluginParser = null
  let animationPluginParser = null
  let modelPluginParser = null

  class FakeVRMLoaderPlugin {
    constructor(parser) {
      this.name = 'FakeVRMLoaderPlugin'
      vrmPluginParser = parser
    }
  }

  class FakeModelLoaderPlugin {
    constructor(parser) {
      this.name = 'FakeModelLoaderPlugin'
      modelPluginParser = parser
    }
  }

  class FakeVRMAnimationLoaderPlugin {
    constructor(parser) {
      this.name = 'FakeVRMAnimationLoaderPlugin'
      animationPluginParser = parser
    }
  }

  const vrmGltf = await loadVrmFromFile(SYNTHETIC_VRM, {
    VRMLoaderPlugin: FakeVRMLoaderPlugin,
  })
  assert.ok(findFirst(vrmGltf.scene, (object) => object.isMesh === true), 'VRM helper should still parse glTF scenes')
  assert.ok(vrmPluginParser, 'VRM helper should install the supplied VRMLoaderPlugin')
  assert.ok(vrmPluginParser.json?.extensionsUsed?.includes('VRMC_vrm'), 'VRM fixture should expose VRMC_vrm metadata to the plugin')

  const animationGltf = await loadVrmAnimationFromFile(SYNTHETIC_VRMA, {
    VRMLoaderPlugin: FakeModelLoaderPlugin,
    VRMAnimationLoaderPlugin: FakeVRMAnimationLoaderPlugin,
  })
  assert.ok(findFirst(animationGltf.scene, (object) => object.isMesh === true), 'VRMA helper should still parse glTF scenes')
  assert.ok(modelPluginParser, 'VRMA helper should install the supplied VRMLoaderPlugin when provided')
  assert.ok(animationPluginParser, 'VRMA helper should install the supplied VRMAnimationLoaderPlugin')
  assert.ok(
    animationPluginParser.json?.extensionsUsed?.includes('VRMC_vrm_animation'),
    'VRMA fixture should expose VRMC_vrm_animation metadata to the plugin',
  )
})

test('synthetic humanoid VRM and VRMA fixtures expose avatar-scale skeleton metadata', async () => {
  let vrmPluginParser = null
  let animationPluginParser = null

  class CaptureVRMLoaderPlugin {
    constructor(parser) {
      this.name = 'CaptureVRMLoaderPlugin'
      vrmPluginParser = parser
    }
  }

  class CaptureVRMAnimationLoaderPlugin {
    constructor(parser) {
      this.name = 'CaptureVRMAnimationLoaderPlugin'
      animationPluginParser = parser
    }
  }

  const vrmGltf = await loadVrmFromFile(SYNTHETIC_HUMANOID_VRM, {
    VRMLoaderPlugin: CaptureVRMLoaderPlugin,
  })
  const skinnedMesh = findFirst(vrmGltf.scene, (object) => object.isSkinnedMesh === true)
  assert.ok(skinnedMesh, 'synthetic humanoid VRM should load skinned mesh primitives')
  assert.equal(skinnedMesh.skeleton.bones.length, 17)
  assert.equal(skinnedMesh.geometry.getAttribute('skinIndex')?.count, 44)
  assert.equal(skinnedMesh.geometry.getAttribute('skinWeight')?.count, 44)
  assert.ok(vrmPluginParser, 'synthetic humanoid VRM should initialize the supplied loader plugin')
  assert.ok(vrmPluginParser.json?.extensionsUsed?.includes('VRMC_vrm'), 'synthetic humanoid VRM should expose VRMC_vrm metadata')
  assert.ok(
    vrmPluginParser.json?.extensionsUsed?.includes('VRMC_materials_mtoon'),
    'synthetic humanoid VRM should expose VRM MToon material metadata',
  )

  const vrmExtension = vrmPluginParser.json?.extensions?.VRMC_vrm
  const humanBones = vrmExtension?.humanoid?.humanBones ?? {}
  assert.equal(vrmExtension?.meta?.name, 'Synthetic Humanoid Avatar')
  assert.equal(humanBones.hips?.node, 2)
  assert.equal(humanBones.head?.node, 6)
  assert.equal(humanBones.leftUpperArm?.node, 7)
  assert.equal(humanBones.rightFoot?.node, 18)
  assert.equal(vrmExtension?.expressions?.preset?.happy?.morphTargetBinds?.[0]?.node, 1)
  assert.equal(vrmExtension?.expressions?.preset?.blink?.isBinary, true)

  const camera = vrmGltf.cameras[0]
  assert.ok(camera, 'synthetic humanoid VRM should load a camera')
  camera.aspect = 1
  camera.updateProjectionMatrix()
  camera.updateMatrixWorld(true)
  vrmGltf.scene.updateMatrixWorld(true)
  const rgba = new Renderer().render(vrmGltf.scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.05, 'synthetic humanoid VRM should render visible avatar geometry')

  const animationGltf = await loadVrmAnimationFromFile(SYNTHETIC_HUMANOID_VRMA, {
    VRMAnimationLoaderPlugin: CaptureVRMAnimationLoaderPlugin,
  })
  assert.equal(animationGltf.animations.length, 1)
  assert.ok(animationPluginParser, 'synthetic humanoid VRMA should initialize the supplied animation loader plugin')
  assert.ok(
    animationPluginParser.json?.extensionsUsed?.includes('VRMC_vrm_animation'),
    'synthetic humanoid VRMA should expose VRMC_vrm_animation metadata',
  )
  assert.equal(animationPluginParser.json?.nodes?.length, 19)
  assert.equal(animationPluginParser.json?.animations?.[0]?.channels?.length, 3)
  assert.equal(animationPluginParser.json?.animations?.[0]?.samplers?.length, 3)

  const vrmaExtension = animationPluginParser.json?.extensions?.VRMC_vrm_animation
  const animationHumanBones = vrmaExtension?.humanoid?.humanBones ?? {}
  assert.equal(vrmaExtension?.specVersion, '1.0')
  assert.equal(animationHumanBones.hips?.node, 0)
  assert.equal(animationHumanBones.head?.node, 4)
  assert.equal(animationHumanBones.leftUpperArm?.node, 5)
  assert.equal(animationHumanBones.rightFoot?.node, 16)
  assert.equal(vrmaExtension?.lookAt?.node, 17)
  assert.equal(vrmaExtension?.expressions?.preset?.happy?.node, 18)
})

test('real full-avatar VRM fixture loads practical avatar extension coverage', async () => {
  let vrmPluginParser = null

  class CaptureVRMLoaderPlugin {
    constructor(parser) {
      this.name = 'CaptureVRMLoaderPlugin'
      vrmPluginParser = parser
    }
  }

  const vrmGltf = await loadVrmFromFile(REAL_VRM_SEED_SAN_SAMPLE, {
    VRMLoaderPlugin: CaptureVRMLoaderPlugin,
  })
  assert.ok(vrmPluginParser, 'Seed-san VRM should initialize the supplied VRM loader plugin')
  assert.ok(vrmPluginParser.json?.extensionsUsed?.includes('VRMC_vrm'), 'Seed-san VRM should expose VRMC_vrm metadata')
  assert.ok(vrmPluginParser.json?.extensionsUsed?.includes('VRMC_springBone'), 'Seed-san VRM should expose VRMC_springBone metadata')
  assert.ok(
    vrmPluginParser.json?.extensionsUsed?.includes('VRMC_materials_mtoon'),
    'Seed-san VRM should expose VRMC_materials_mtoon metadata',
  )
  assert.ok(
    vrmPluginParser.json?.extensionsUsed?.includes('VRMC_node_constraint'),
    'Seed-san VRM should expose VRMC_node_constraint metadata',
  )
  assert.deepEqual(
    {
      nodes: vrmPluginParser.json?.nodes?.length,
      meshes: vrmPluginParser.json?.meshes?.length,
      skins: vrmPluginParser.json?.skins?.length,
      materials: vrmPluginParser.json?.materials?.length,
      images: vrmPluginParser.json?.images?.length,
    },
    { nodes: 147, meshes: 5, skins: 5, materials: 17, images: 15 },
  )

  const skinnedMeshes = []
  vrmGltf.scene.traverse((object) => {
    if (object.isSkinnedMesh === true) skinnedMeshes.push(object)
  })
  assert.equal(skinnedMeshes.length, 21)

  const vrmExtension = vrmPluginParser.json?.extensions?.VRMC_vrm
  const humanBones = vrmExtension?.humanoid?.humanBones ?? {}
  assert.equal(vrmExtension?.specVersion, '1.0')
  assert.equal(vrmExtension?.meta?.name, 'Seed-san')
  assert.equal(vrmExtension?.meta?.licenseUrl, 'https://vrm.dev/licenses/1.0/')
  assert.equal(vrmExtension?.meta?.allowRedistribution, true)
  assert.equal(vrmExtension?.meta?.creditNotation, 'required')
  assert.ok('hips' in humanBones, 'Seed-san VRM should map humanoid hips')
  assert.ok('head' in humanBones, 'Seed-san VRM should map humanoid head')
  assert.ok('leftHand' in humanBones, 'Seed-san VRM should map humanoid left hand')
  assert.ok('rightToes' in humanBones, 'Seed-san VRM should map humanoid right toes')
  assert.equal(Object.keys(humanBones).length, 51)
  assert.equal(Object.keys(vrmExtension?.expressions?.preset ?? {}).length, 18)
  assert.ok(vrmExtension?.lookAt, 'Seed-san VRM should include look-at metadata')
  assert.ok(vrmExtension?.firstPerson, 'Seed-san VRM should include first-person metadata')

  const springBoneExtension = vrmPluginParser.json?.extensions?.VRMC_springBone
  assert.equal(springBoneExtension?.springs?.length, 9)
  assert.equal(springBoneExtension?.colliders?.length, 8)
  assert.equal(springBoneExtension?.colliderGroups?.length, 2)
  assert.equal(
    vrmPluginParser.json?.nodes?.filter((node) => node.extensions?.VRMC_node_constraint).length,
    23,
  )
  assert.equal(vrmPluginParser.json?.materials?.filter((material) => material.extensions?.VRMC_materials_mtoon).length, 10)

  const camera = frameSceneCamera(vrmGltf.scene)
  const rgba = new Renderer().render(vrmGltf.scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.05, 'Seed-san VRM should render visible full-avatar geometry')
})
