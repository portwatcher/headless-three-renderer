import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'

const {
  Renderer,
  loadGltfFromFile,
  loadVrmAnimationFromFile,
  loadVrmFromFile,
} = pkg

const FIXTURE_DIR = fileURLToPath(new URL('./fixtures/', import.meta.url))
const SIMPLE_TRIANGLE = path.join(FIXTURE_DIR, 'simple-triangle.gltf')
const TEXTURED_QUAD = path.join(FIXTURE_DIR, 'textured-quad.gltf')
const VERTEX_COLOR_QUAD = path.join(FIXTURE_DIR, 'vertex-color-quad.gltf')
const MORPHED_TRIANGLE = path.join(FIXTURE_DIR, 'morphed-triangle.gltf')
const SKINNED_QUAD = path.join(FIXTURE_DIR, 'skinned-quad.gltf')
const SYNTHETIC_VRM = path.join(FIXTURE_DIR, 'synthetic-avatar.vrm')
const SYNTHETIC_VRMA = path.join(FIXTURE_DIR, 'synthetic-animation.vrma')
const REAL_VRM_EXPRESSION_SAMPLE = path.join(
  FIXTURE_DIR,
  'vrm-specification',
  'VRMC_vrm_expressions_isBinary_Overridden',
  'VRMC_vrm_expressions_isBinary_Overridden.vrm',
)
const REAL_VRMA_ANIMATION_SAMPLE = path.join(FIXTURE_DIR, 'three-vrm-animation', 'test.vrma')
const SAMPLE_ASSET_ANIMATED_CUBE = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'AnimatedCube', 'glTF', 'AnimatedCube.gltf')
const SAMPLE_ASSET_ANIMATED_COLORS_CUBE = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'AnimatedColorsCube', 'glTF', 'AnimatedColorsCube.gltf')
const SAMPLE_ASSET_ANIMATED_MORPH_CUBE = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'AnimatedMorphCube', 'glTF', 'AnimatedMorphCube.gltf')
const SAMPLE_ASSET_ANIMATED_TRIANGLE = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'AnimatedTriangle', 'glTF', 'AnimatedTriangle.gltf')
const SAMPLE_ASSET_ANIMATION_POINTER_UVS = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'AnimationPointerUVs', 'glTF', 'AnimationPointerUVs.gltf')
const SAMPLE_ASSET_ALPHA_BLEND_MODE_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'AlphaBlendModeTest', 'glTF', 'AlphaBlendModeTest.gltf')
const SAMPLE_ASSET_ANISOTROPY_DISC_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'AnisotropyDiscTest', 'glTF', 'AnisotropyDiscTest.gltf')
const SAMPLE_ASSET_ANISOTROPY_ROTATION_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'AnisotropyRotationTest', 'glTF', 'AnisotropyRotationTest.gltf')
const SAMPLE_ASSET_ANISOTROPY_STRENGTH_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'AnisotropyStrengthTest', 'glTF', 'AnisotropyStrengthTest.gltf')
const SAMPLE_ASSET_ATTENUATION_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'AttenuationTest', 'glTF', 'AttenuationTest.gltf')
const SAMPLE_ASSET_AVOCADO = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'Avocado', 'glTF', 'Avocado.gltf')
const SAMPLE_ASSET_BARRAMUNDI_FISH = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'BarramundiFish', 'glTF', 'BarramundiFish.gltf')
const SAMPLE_ASSET_BOOM_BOX = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'BoomBox', 'glTF', 'BoomBox.gltf')
const SAMPLE_ASSET_BOOM_BOX_WITH_AXES = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'BoomBoxWithAxes', 'glTF', 'BoomBoxWithAxes.gltf')
const SAMPLE_ASSET_BOX_ANIMATED = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'BoxAnimated', 'glTF', 'BoxAnimated.gltf')
const SAMPLE_ASSET_BOX = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'Box', 'glTF', 'Box.gltf')
const SAMPLE_ASSET_BOX_WITH_SPACES = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'Box With Spaces', 'glTF', 'Box With Spaces.gltf')
const SAMPLE_ASSET_BOX_INTERLEAVED = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'BoxInterleaved', 'glTF', 'BoxInterleaved.gltf')
const SAMPLE_ASSET_BOX_TEXTURED_NPOT = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'BoxTexturedNonPowerOfTwo', 'glTF', 'BoxTexturedNonPowerOfTwo.gltf')
const SAMPLE_ASSET_BOX_VERTEX_COLORS = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'BoxVertexColors', 'glTF', 'BoxVertexColors.gltf')
const SAMPLE_ASSET_CAMERAS = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'Cameras', 'glTF', 'Cameras.gltf')
const SAMPLE_ASSET_CESIUM_MAN = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CesiumMan', 'glTF', 'CesiumMan.gltf')
const SAMPLE_ASSET_CLEARCOAT_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'ClearCoatTest', 'glTF', 'ClearCoatTest.gltf')
const SAMPLE_ASSET_COMPARE_ALPHA_COVERAGE = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CompareAlphaCoverage', 'glTF', 'CompareAlphaCoverage.gltf')
const SAMPLE_ASSET_COMPARE_AMBIENT_OCCLUSION = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CompareAmbientOcclusion', 'glTF', 'CompareAmbientOcclusion.gltf')
const SAMPLE_ASSET_COMPARE_ANISOTROPY = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CompareAnisotropy', 'glTF', 'CompareAnisotropy.gltf')
const SAMPLE_ASSET_COMPARE_BASE_COLOR = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CompareBaseColor', 'glTF', 'CompareBaseColor.gltf')
const SAMPLE_ASSET_COMPARE_CLEARCOAT = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CompareClearcoat', 'glTF', 'CompareClearcoat.gltf')
const SAMPLE_ASSET_COMPARE_DISPERSION = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CompareDispersion', 'glTF', 'CompareDispersion.gltf')
const SAMPLE_ASSET_COMPARE_EMISSIVE_STRENGTH = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CompareEmissiveStrength', 'glTF', 'CompareEmissiveStrength.gltf')
const SAMPLE_ASSET_COMPARE_IOR = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CompareIor', 'glTF', 'CompareIor.gltf')
const SAMPLE_ASSET_COMPARE_IRIDESCENCE = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CompareIridescence', 'glTF', 'CompareIridescence.gltf')
const SAMPLE_ASSET_COMPARE_METALLIC = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CompareMetallic', 'glTF', 'CompareMetallic.gltf')
const SAMPLE_ASSET_COMPARE_NORMAL = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CompareNormal', 'glTF', 'CompareNormal.gltf')
const SAMPLE_ASSET_COMPARE_ROUGHNESS = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CompareRoughness', 'glTF', 'CompareRoughness.gltf')
const SAMPLE_ASSET_COMPARE_SHEEN = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CompareSheen', 'glTF', 'CompareSheen.gltf')
const SAMPLE_ASSET_COMPARE_SPECULAR = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CompareSpecular', 'glTF', 'CompareSpecular.gltf')
const SAMPLE_ASSET_COMPARE_TRANSMISSION = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CompareTransmission', 'glTF', 'CompareTransmission.gltf')
const SAMPLE_ASSET_COMPARE_VOLUME = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CompareVolume', 'glTF', 'CompareVolume.gltf')
const SAMPLE_ASSET_CUBE_VISIBILITY = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CubeVisibility', 'glTF', 'CubeVisibility.gltf')
const SAMPLE_ASSET_DIRECTIONAL_LIGHT = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'DirectionalLight', 'glTF', 'DirectionalLight.gltf')
const SAMPLE_ASSET_DUCK = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'Duck', 'glTF', 'Duck.gltf')
const SAMPLE_ASSET_EMISSIVE_STRENGTH_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'EmissiveStrengthTest', 'glTF', 'EmissiveStrengthTest.gltf')
const SAMPLE_ASSET_ENVIRONMENT_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'EnvironmentTest', 'glTF', 'EnvironmentTest.gltf')
const SAMPLE_ASSET_FOX = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'Fox', 'glTF', 'Fox.gltf')
const SAMPLE_ASSET_INTERPOLATION_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'InterpolationTest', 'glTF', 'InterpolationTest.gltf')
const SAMPLE_ASSET_IRIDESCENCE_LAMP = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'IridescenceLamp', 'glTF', 'IridescenceLamp.gltf')
const SAMPLE_ASSET_LIGHT_VISIBILITY = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'LightVisibility', 'glTF', 'LightVisibility.gltf')
const SAMPLE_ASSET_LIGHTS_PUNCTUAL_LAMP = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'LightsPunctualLamp', 'glTF', 'LightsPunctualLamp.gltf')
const SAMPLE_ASSET_METAL_ROUGH_SPHERES = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'MetalRoughSpheres', 'glTF', 'MetalRoughSpheres.gltf')
const SAMPLE_ASSET_METAL_ROUGH_SPHERES_NO_TEXTURES = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'MetalRoughSpheresNoTextures', 'glTF', 'MetalRoughSpheresNoTextures.gltf')
const SAMPLE_ASSET_MESH_PRIMITIVE_MODES = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'MeshPrimitiveModes', 'glTF', 'MeshPrimitiveModes.gltf')
const SAMPLE_ASSET_MORPH_PRIMITIVES_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'MorphPrimitivesTest', 'glTF', 'MorphPrimitivesTest.gltf')
const SAMPLE_ASSET_MORPH_STRESS_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'MorphStressTest', 'glTF', 'MorphStressTest.gltf')
const SAMPLE_ASSET_MULTI_UV_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'MultiUVTest', 'glTF', 'MultiUVTest.gltf')
const SAMPLE_ASSET_MULTIPLE_SCENES = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'MultipleScenes', 'glTF', 'MultipleScenes.gltf')
const SAMPLE_ASSET_NEGATIVE_SCALE_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'NegativeScaleTest', 'glTF', 'NegativeScaleTest.gltf')
const SAMPLE_ASSET_NORMAL_TANGENT_MIRROR_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'NormalTangentMirrorTest', 'glTF', 'NormalTangentMirrorTest.gltf')
const SAMPLE_ASSET_NORMAL_TANGENT_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'NormalTangentTest', 'glTF', 'NormalTangentTest.gltf')
const SAMPLE_ASSET_ORIENTATION_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'OrientationTest', 'glTF', 'OrientationTest.gltf')
const SAMPLE_ASSET_POINT_LIGHT_INTENSITY_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'PointLightIntensityTest', 'glTF', 'PointLightIntensityTest.gltf')
const SAMPLE_ASSET_PRIMITIVE_MODE_NORMALS_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'PrimitiveModeNormalsTest', 'glTF', 'PrimitiveModeNormalsTest.gltf')
const SAMPLE_ASSET_RECURSIVE_SKELETONS = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'RecursiveSkeletons', 'glTF', 'RecursiveSkeletons.gltf')
const SAMPLE_ASSET_RIGGED_FIGURE = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'RiggedFigure', 'glTF', 'RiggedFigure.gltf')
const SAMPLE_ASSET_RIGGED_SIMPLE = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'RiggedSimple', 'glTF', 'RiggedSimple.gltf')
const SAMPLE_ASSET_SHEEN_CHAIR = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'SheenChair', 'glTF', 'SheenChair.gltf')
const SAMPLE_ASSET_SIMPLE_INSTANCING = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'SimpleInstancing', 'glTF', 'SimpleInstancing.gltf')
const SAMPLE_ASSET_SIMPLE_MATERIAL = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'SimpleMaterial', 'glTF', 'SimpleMaterial.gltf')
const SAMPLE_ASSET_SIMPLE_MESHES = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'SimpleMeshes', 'glTF', 'SimpleMeshes.gltf')
const SAMPLE_ASSET_SIMPLE_MORPH = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'SimpleMorph', 'glTF', 'SimpleMorph.gltf')
const SAMPLE_ASSET_SIMPLE_SKIN = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'SimpleSkin', 'glTF', 'SimpleSkin.gltf')
const SAMPLE_ASSET_SIMPLE_SPARSE_ACCESSOR = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'SimpleSparseAccessor', 'glTF', 'SimpleSparseAccessor.gltf')
const SAMPLE_ASSET_SIMPLE_TEXTURE = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'SimpleTexture', 'glTF', 'SimpleTexture.gltf')
const SAMPLE_ASSET_SPECULAR_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'SpecularTest', 'glTF', 'SpecularTest.gltf')
const SAMPLE_ASSET_SUZANNE = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'Suzanne', 'glTF', 'Suzanne.gltf')
const SAMPLE_ASSET_TEXTURE_COORDINATE_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'TextureCoordinateTest', 'glTF', 'TextureCoordinateTest.gltf')
const SAMPLE_ASSET_TEXTURE_ENCODING_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'TextureEncodingTest', 'glTF', 'TextureEncodingTest.gltf')
const SAMPLE_ASSET_TEXTURE_LINEAR_INTERPOLATION_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'TextureLinearInterpolationTest', 'glTF', 'TextureLinearInterpolationTest.gltf')
const SAMPLE_ASSET_TEXTURE_SETTINGS_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'TextureSettingsTest', 'glTF', 'TextureSettingsTest.gltf')
const SAMPLE_ASSET_TEXTURE_TRANSFORM_MULTI_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'TextureTransformMultiTest', 'glTF', 'TextureTransformMultiTest.gltf')
const SAMPLE_ASSET_TEXTURE_TRANSFORM_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'TextureTransformTest', 'glTF', 'TextureTransformTest.gltf')
const SAMPLE_ASSET_TRANSMISSION_ORDER_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'TransmissionOrderTest', 'glTF', 'TransmissionOrderTest.gltf')
const SAMPLE_ASSET_TRANSMISSION_ROUGHNESS_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'TransmissionRoughnessTest', 'glTF', 'TransmissionRoughnessTest.gltf')
const SAMPLE_ASSET_TRIANGLE = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'Triangle', 'glTF', 'Triangle.gltf')
const SAMPLE_ASSET_TRIANGLE_WITHOUT_INDICES = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'TriangleWithoutIndices', 'glTF', 'TriangleWithoutIndices.gltf')
const SAMPLE_ASSET_TWO_SIDED_PLANE = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'TwoSidedPlane', 'glTF', 'TwoSidedPlane.gltf')
const SAMPLE_ASSET_UNICODE_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'Unicode❤♻Test', 'glTF', 'Unicode❤♻Test.gltf')
const SAMPLE_ASSET_UNLIT_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'UnlitTest', 'glTF', 'UnlitTest.gltf')
const SAMPLE_ASSET_VERTEX_COLOR_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'VertexColorTest', 'glTF', 'VertexColorTest.gltf')
const SAMPLE_ASSET_XMP_METADATA_ROUNDED_CUBE = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'XmpMetadataRoundedCube', 'glTF', 'XmpMetadataRoundedCube.gltf')

test('committed glTF fixture loads through GLTFLoader and renders', async () => {
  let configured = false
  const gltf = await loadGltfFixture(SIMPLE_TRIANGLE, {
    configureLoader(loader) {
      configured = typeof loader.parse === 'function'
    },
  })
  assert.equal(configured, true, 'loadGltfFromFile should expose the loader before parsing')

  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'fixture should load a mesh')
  assert.equal(mesh.geometry.getAttribute('position').count, 3)
  assert.equal(mesh.material.isMeshStandardMaterial, true)

  const camera = gltf.cameras[0]
  assert.ok(camera, 'fixture should load a camera')
  camera.aspect = 1
  camera.updateProjectionMatrix()

  const scene = gltf.scene
  scene.add(new THREE.AmbientLight(0xffffff, 0.6))
  const light = new THREE.DirectionalLight(0xffffff, 1.4)
  light.position.set(2, 3, 4)
  scene.add(light)
  scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [0.02, 0.02, 0.03],
  })
  assert.equal(rgba.length, 96 * 96 * 4)
  assert.ok(nonBackgroundRatio(rgba, [5, 5, 8], 3) > 0.04, 'glTF triangle should render visible pixels')

  const mean = meanRgba(rgba)
  assert.ok(mean.b > mean.r, `loaded blue PBR material should contribute blue output (${mean.b} vs ${mean.r})`)
  assert.ok(mean.a > 240, `loaded glTF output should be opaque (${mean.a})`)
})

test('committed Khronos glTF Sample Assets Box fixture loads external buffer and renders', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_BOX)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'Khronos Box sample should load a mesh')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 24)
  assert.equal(mesh.geometry.index?.count, 36)
  assert.equal(mesh.material.name, 'Red')

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 10)
  camera.position.set(1.4, 1.1, 2.2)
  camera.lookAt(0, 0, 0)

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.55))
  const light = new THREE.DirectionalLight(0xffffff, 1.6)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.2, 'Khronos Box sample should render visible pixels')
  const center = meanRegion(rgba, 96, 96, 40, 40, 56, 56)
  assert.ok(center.r > center.b + 150 && center.r > center.g + 180, `Khronos Box sample should render a red cube (${center.r}, ${center.g}, ${center.b})`)
})

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

test('committed Khronos glTF Sample Assets AlphaBlendModeTest fixture loads alpha modes and JPEG textures', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_ALPHA_BLEND_MODE_TEST)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 9)

  const meshesByName = new Map(meshes.map((mesh) => [mesh.name, mesh]))
  assert.equal(meshesByName.get('TestBlend')?.material.transparent, true)
  assert.equal(meshesByName.get('DecalBlend')?.material.transparent, true)
  assert.equal(meshesByName.get('TestOpaque')?.material.transparent, false)
  assert.equal(meshesByName.get('TestCutoff25')?.material.alphaTest, 0.25)
  assert.equal(meshesByName.get('TestCutoffDefault')?.material.alphaTest, 0.5)
  assert.equal(meshesByName.get('TestCutoff75')?.material.alphaTest, 0.75)
  assert.ok(
    meshes.every((mesh) => Buffer.isBuffer(mesh.material.map?.image)),
    'AlphaBlendModeTest PNG and JPEG material textures should load as encoded Buffers',
  )

  const bed = meshesByName.get('Bed')
  assert.ok(Buffer.isBuffer(bed?.material.normalMap?.image), 'AlphaBlendModeTest JPEG normal map should load as an encoded Buffer')
  assert.ok(Buffer.isBuffer(bed?.material.aoMap?.image), 'AlphaBlendModeTest JPEG ORM map should load as an encoded Buffer')
  assert.ok(Buffer.isBuffer(bed?.material.roughnessMap?.image), 'AlphaBlendModeTest JPEG roughness map should load as an encoded Buffer')
  assert.ok(Buffer.isBuffer(bed?.material.metalnessMap?.image), 'AlphaBlendModeTest JPEG metalness map should load as an encoded Buffer')

  const camera = new THREE.PerspectiveCamera(35, 4 / 3, 0.01, 50)
  camera.position.set(0, 1.4, 8)
  camera.lookAt(0, 0.8, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1.0))
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 160,
    height: 120,
    format: 'rgba',
    background: [0.04, 0.04, 0.04],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [10, 10, 10], 4) > 0.4, 'AlphaBlendModeTest should render visible alpha-mode geometry')
  const center = meanRegion(rgba, 160, 120, 60, 40, 100, 80)
  assert.ok(center.r > 80 && center.g > 80 && center.b > 70, `AlphaBlendModeTest render should include the textured material bed (${center.r}, ${center.g}, ${center.b})`)
})

test('committed Khronos glTF Sample Assets CompareAlphaCoverage fixture loads alpha coverage material variants', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_COMPARE_ALPHA_COVERAGE, 'utf8'))
  assert.equal(source.buffers[0].uri, 'CompareAlphaCoverage.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'FurBaseColorAlpha.png',
    'FurNormal.png',
    'FurEmissive.jpg',
    'FurORM.jpg',
    'FloorBaseColor.jpg',
    'FloorNormal.jpg',
  ])
  assert.deepEqual(source.materials.map((material) => [material.name, material.alphaMode ?? 'OPAQUE', material.alphaCutoff ?? null]), [
    ['fur_opaque', 'OPAQUE', null],
    ['fur floor', 'OPAQUE', null],
    ['fur_mask', 'MASK', 0.2],
    ['fur_blend', 'BLEND', null],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_COMPARE_ALPHA_COVERAGE)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'Fur001_0',
    'Fur001_1',
    'Fur002_0',
    'Fur002_1',
    'Fur003_0',
    'Fur003_1',
  ])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [32, 4, 32, 4, 4, 32])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [48, 6, 48, 6, 6, 48])

  const materials = new Map(meshes.map((mesh) => [mesh.material.name, mesh.material]))
  const opaque = materials.get('fur_opaque')
  const mask = materials.get('fur_mask')
  const blend = materials.get('fur_blend')
  const floor = materials.get('fur floor')
  assert.equal(opaque?.transparent, false)
  assert.equal(mask?.transparent, false)
  assert.equal(mask.alphaTest, 0.2)
  assert.equal(blend?.transparent, true)
  assert.equal(blend.alphaTest, 0)
  assert.equal(floor?.transparent, false)
  assert.ok([opaque, mask, blend, floor].every((material) => material.side === THREE.DoubleSide))

  for (const material of [opaque, mask, blend]) {
    assert.equal(Buffer.isBuffer(material.map?.image), true, `${material.name} base color/alpha PNG should load as an encoded Buffer`)
    assert.equal(material.map.name, 'FurBaseColorAlpha.png')
    assert.deepEqual(pngDimensions(material.map.image), [1024, 1024])
    assert.equal(material.map.colorSpace, THREE.SRGBColorSpace)
    assert.equal(material.map.flipY, false)

    assert.equal(Buffer.isBuffer(material.normalMap?.image), true, `${material.name} normal PNG should load as an encoded Buffer`)
    assert.equal(material.normalMap.name, 'FurNormal.png')
    assert.equal(material.normalMap.colorSpace, THREE.NoColorSpace)
    assert.equal(material.normalMap.flipY, false)

    assert.equal(Buffer.isBuffer(material.emissiveMap?.image), true, `${material.name} emissive JPEG should load as an encoded Buffer`)
    assert.equal(material.emissiveMap.name, 'FurEmissive.jpg')
    assert.equal(material.emissiveMap.colorSpace, THREE.SRGBColorSpace)
    assert.equal(material.emissiveMap.flipY, false)

    assert.equal(material.roughnessMap, material.metalnessMap, `${material.name} roughness and metalness should share the packed ORM texture`)
    assert.equal(material.aoMap, material.roughnessMap, `${material.name} occlusion and roughness should share the packed ORM texture`)
    assert.equal(Buffer.isBuffer(material.roughnessMap?.image), true, `${material.name} ORM JPEG should load as an encoded Buffer`)
    assert.equal(material.roughnessMap.name, 'FurORM.jpg')
    assert.equal(material.roughnessMap.colorSpace, THREE.NoColorSpace)
    assert.equal(material.roughnessMap.flipY, false)
  }

  assert.equal(Buffer.isBuffer(floor.map?.image), true, 'floor base color JPEG should load as an encoded Buffer')
  assert.equal(floor.map.name, 'FloorBaseColor.jpg')
  assert.equal(floor.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(floor.map.flipY, false)
  assert.equal(Buffer.isBuffer(floor.normalMap?.image), true, 'floor normal JPEG should load as an encoded Buffer')
  assert.equal(floor.normalMap.name, 'FloorNormal.jpg')
  assert.equal(floor.normalMap.colorSpace, THREE.NoColorSpace)
  assert.equal(floor.normalMap.flipY, false)

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 1.6)
  light.position.set(2, 4, 5)
  gltf.scene.add(light)
  const camera = new THREE.PerspectiveCamera(35, 10 / 7, 0.01, 20)
  camera.position.set(0, -7, 3.5)
  camera.lookAt(0, 0, 1)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 160,
    height: 112,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.03, 'CompareAlphaCoverage should render visible alpha coverage panels')
})

test('committed Khronos glTF Sample Assets CompareAmbientOcclusion fixture loads AO material pairs', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_COMPARE_AMBIENT_OCCLUSION, 'utf8'))
  assert.equal(source.buffers[0].uri, 'CompareAmbientOcclusion_data.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'FruitBaseColor.jpg',
    'FruitORM.jpg',
    'BasketORM.jpg',
  ])
  assert.deepEqual(source.materials.map((material) => [
    material.name,
    material.occlusionTexture?.index ?? null,
    material.pbrMetallicRoughness?.metallicRoughnessTexture?.index ?? null,
  ]), [
    ['Basket with Occlusion', 2, null],
    ['Fruit with Occlusion', 1, 1],
    ['Logo', null, null],
    ['Basket without Occlusion', null, null],
    ['Fruit without Occlusion', null, 1],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_COMPARE_AMBIENT_OCCLUSION)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'BasketRight',
    'FruitRight',
    'LogoRight',
    'BasketLeft',
    'FruitLeft',
    'LogoLeft',
  ])
  assert.deepEqual(meshes.map((mesh) => mesh.material.name), [
    'Basket with Occlusion',
    'Fruit with Occlusion',
    'Logo',
    'Basket without Occlusion',
    'Fruit without Occlusion',
    'Logo',
  ])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [17832, 28918, 1605, 11240, 28918, 1605])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [66828, 117600, 2865, 66828, 117600, 2865])

  const materials = new Map(meshes.map((mesh) => [mesh.material.name, mesh.material]))
  const basketWithAo = materials.get('Basket with Occlusion')
  const basketWithoutAo = materials.get('Basket without Occlusion')
  const fruitWithAo = materials.get('Fruit with Occlusion')
  const fruitWithoutAo = materials.get('Fruit without Occlusion')
  assert.equal(Buffer.isBuffer(basketWithAo?.aoMap?.image), true, 'basket AO JPEG should load as an encoded Buffer')
  assert.equal(basketWithAo.aoMap.name, 'BasketORM.jpg')
  assert.equal(basketWithAo.aoMap.colorSpace, THREE.NoColorSpace)
  assert.equal(basketWithAo.aoMap.flipY, false)
  assert.equal(basketWithoutAo?.aoMap ?? null, null)

  for (const material of [fruitWithAo, fruitWithoutAo]) {
    assert.equal(Buffer.isBuffer(material.map?.image), true, `${material.name} base color JPEG should load as an encoded Buffer`)
    assert.equal(material.map.name, 'FruitBaseColor.jpg')
    assert.equal(material.map.colorSpace, THREE.SRGBColorSpace)
    assert.equal(material.map.flipY, false)
    assert.equal(material.roughnessMap, material.metalnessMap, `${material.name} roughness and metalness should share the packed ORM texture`)
    assert.equal(Buffer.isBuffer(material.roughnessMap?.image), true, `${material.name} ORM JPEG should load as an encoded Buffer`)
    assert.equal(material.roughnessMap.name, 'FruitORM.jpg')
    assert.equal(material.roughnessMap.colorSpace, THREE.NoColorSpace)
    assert.equal(material.roughnessMap.flipY, false)
  }
  assert.equal(fruitWithAo.aoMap, fruitWithAo.roughnessMap, 'fruit AO should share the packed ORM texture when occlusion is enabled')
  assert.equal(fruitWithoutAo.aoMap ?? null, null)

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.55))
  const light = new THREE.DirectionalLight(0xffffff, 1.8)
  light.position.set(1.5, 3, 4)
  gltf.scene.add(light)
  const camera = new THREE.PerspectiveCamera(35, 1.5, 0.01, 10)
  camera.position.set(0, -1.4, 0.65)
  camera.lookAt(0, 0, 0.05)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 144,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.08, 'CompareAmbientOcclusion should render visible paired AO samples')
})

test('committed Khronos glTF Sample Assets CompareAnisotropy fixture loads anisotropy comparison variants', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_COMPARE_ANISOTROPY, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_texture_transform', 'KHR_materials_anisotropy'])
  assert.equal(source.buffers[0].uri, 'CompareAnisotropy.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'Compare_Anisotropy_img0.jpg',
    'Compare_Anisotropy_img1.png',
    'Compare_Anisotropy_img2.jpg',
    'Compare_Anisotropy_img3.png',
    'Compare_Anisotropy_img4.png',
  ])
  assert.deepEqual(source.materials.map((material) => [
    material.name,
    material.pbrMetallicRoughness?.baseColorTexture?.index ?? null,
    material.pbrMetallicRoughness?.metallicRoughnessTexture?.index ?? null,
    material.extensions?.KHR_materials_anisotropy?.anisotropyStrength ?? null,
    material.extensions?.KHR_materials_anisotropy?.anisotropyTexture?.index ?? null,
  ]), [
    ['grooved-anisotropy', 1, 3, 0.5, null],
    ['spiral-anisotropy', null, null, 0.5, 4],
    ['grooved', 1, 3, null, null],
    ['spiral', null, null, null, null],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_COMPARE_ANISOTROPY)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'Spheroid002_primitive0',
    'Spheroid002_primitive1',
    'Spheroid001_primitive0',
    'Spheroid001_primitive1',
  ])
  assert.deepEqual(meshes.map((mesh) => mesh.material.name), [
    'grooved-anisotropy',
    'spiral-anisotropy',
    'grooved',
    'spiral',
  ])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [5313, 4258, 5313, 4258])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [5313, 4258, 5313, 4258])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [5313, 4258, 5313, 4258])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [30720, 24576, 30720, 24576])

  const [groovedAniso, spiralAniso, grooved, spiral] = meshes.map((mesh) => mesh.material)
  assert.equal(groovedAniso.isMeshPhysicalMaterial, true)
  assert.equal(spiralAniso.isMeshPhysicalMaterial, true)
  assert.equal(grooved.isMeshStandardMaterial, true)
  assert.equal(spiral.isMeshStandardMaterial, true)
  assert.equal(groovedAniso.anisotropy, 0.5)
  assert.equal(groovedAniso.anisotropyRotation, 0)
  assert.equal(groovedAniso.anisotropyMap ?? null, null)
  assert.equal(spiralAniso.anisotropy, 0.5)
  assert.equal(spiralAniso.anisotropyRotation, 0)
  assert.equal(Buffer.isBuffer(spiralAniso.anisotropyMap?.image), true, 'anisotropy direction PNG should load as an encoded Buffer')
  assert.equal(spiralAniso.anisotropyMap.name, 'Compare_Anisotropy_img4.png')
  assert.deepEqual(pngDimensions(spiralAniso.anisotropyMap.image), [256, 256])
  assert.equal(spiralAniso.anisotropyMap.colorSpace, THREE.NoColorSpace)
  assert.equal(spiralAniso.anisotropyMap.flipY, false)
  assert.equal(grooved.anisotropyMap ?? null, null)
  assert.equal(spiral.anisotropyMap ?? null, null)

  assert.equal(groovedAniso.map, grooved.map, 'grooved anisotropy pair should share the base-color texture')
  assert.equal(Buffer.isBuffer(grooved.map?.image), true, 'grooved base-color PNG should load as an encoded Buffer')
  assert.equal(grooved.map.name, 'Compare_Anisotropy_img1.png')
  assert.deepEqual(pngDimensions(grooved.map.image), [2048, 1024])
  assert.equal(grooved.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(grooved.map.flipY, false)

  assert.equal(groovedAniso.roughnessMap, groovedAniso.metalnessMap)
  assert.equal(grooved.roughnessMap, grooved.metalnessMap)
  assert.equal(groovedAniso.roughnessMap, grooved.roughnessMap)
  assert.equal(Buffer.isBuffer(grooved.roughnessMap?.image), true, 'grooved metallic-roughness PNG should load as an encoded Buffer')
  assert.equal(grooved.roughnessMap.name, 'Compare_Anisotropy_img3.png')
  assert.deepEqual(pngDimensions(grooved.roughnessMap.image), [2048, 1024])
  assert.equal(grooved.roughnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(grooved.roughnessMap.flipY, false)

  assertVectorClose(spiralAniso.color.toArray(), [
    0.5795467495918274,
    0.2715774476528168,
    0.18354901671409607,
  ], 'CompareAnisotropy spiral anisotropic baseColorFactor')
  assert.deepEqual(spiralAniso.color.toArray(), spiral.color.toArray())
  assert.equal(spiralAniso.roughness, 0.1)
  assert.equal(spiral.roughness, 0.1)
  assert.equal(spiralAniso.metalness, 1)
  assert.equal(spiral.metalness, 1)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 3)
  light.position.set(2, 4, 5)
  gltf.scene.add(light)
  const camera = new THREE.PerspectiveCamera(35, 1.6, 0.01, 20)
  camera.position.copy(center).add(new THREE.Vector3(0, -3.2, 1.4))
  camera.lookAt(center)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 160,
    height: 100,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.SRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.15, 'CompareAnisotropy should render visible anisotropy comparison geometry')
})

test('committed Khronos glTF Sample Assets CompareBaseColor fixture loads base-color comparison variants', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_COMPARE_BASE_COLOR, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_texture_transform'])
  assert.equal(source.buffers[0].uri, 'CompareBasecolor.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'Compare_Basecolor_img0.png',
    'Compare_Basecolor_img1.png',
  ])
  assert.deepEqual(source.materials.map((material) => [
    material.name,
    material.pbrMetallicRoughness?.baseColorTexture?.index ?? null,
    material.emissiveTexture?.index ?? null,
  ]), [
    ['baseColor plain dielectric', null, 0],
    ['baseColor texture dielectric', 1, 0],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_COMPARE_BASE_COLOR)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), ['Sphere001', 'Sphere002', 'Sphere003'])
  assert.deepEqual(meshes.map((mesh) => mesh.material.name), [
    'baseColor plain dielectric',
    'baseColor texture dielectric',
    'baseColor texture dielectric',
  ])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [9216, 9216, 9216])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [9216, 9216, 9216])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('color')?.count ?? null), [null, null, 9216])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [9216, 9216, 9216])

  const [plain, textured, vertexColored] = meshes
  assert.equal(plain.material.vertexColors, false)
  assert.equal(textured.material.vertexColors, false)
  assert.equal(vertexColored.material.vertexColors, true)
  assertVectorClose(plain.material.color.toArray(), [
    0.23882800340652466,
    0.10615606606006622,
    0.0477757565677166,
  ], 'CompareBaseColor baseColorFactor')
  assert.deepEqual(textured.material.color.toArray(), [1, 1, 1])
  assert.deepEqual(vertexColored.material.color.toArray(), [1, 1, 1])

  for (const material of [plain.material, textured.material, vertexColored.material]) {
    assert.equal(material.metalness, 0)
    assert.equal(material.roughness, 0.25)
    assert.equal(Buffer.isBuffer(material.emissiveMap?.image), true, `${material.name} emissive PNG should load as an encoded Buffer`)
    assert.equal(material.emissiveMap.name, 'Compare_Basecolor_img0.png')
    assert.deepEqual(pngDimensions(material.emissiveMap.image), [2048, 1024])
    assert.equal(material.emissiveMap.colorSpace, THREE.SRGBColorSpace)
    assert.equal(material.emissiveMap.flipY, false)
    assertVectorClose(material.emissiveMap.offset.toArray(), [0.324, 0.137], `${material.name} emissive texture offset`)
    assertVectorClose(material.emissiveMap.repeat.toArray(), [0.349, 0.725], `${material.name} emissive texture scale`)
  }

  for (const material of [textured.material, vertexColored.material]) {
    assert.equal(Buffer.isBuffer(material.map?.image), true, `${material.name} base-color PNG should load as an encoded Buffer`)
    assert.equal(material.map.name, 'Compare_Basecolor_img1.png')
    assert.deepEqual(pngDimensions(material.map.image), [512, 512])
    assert.equal(material.map.colorSpace, THREE.SRGBColorSpace)
    assert.equal(material.map.flipY, false)
    assertVectorClose(material.map.offset.toArray(), [0.25, 0.25], `${material.name} base-color texture offset`)
    assertVectorClose(material.map.repeat.toArray(), [0.5, 0.5], `${material.name} base-color texture scale`)
  }
  assert.equal(plain.material.map ?? null, null)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.7))
  const light = new THREE.DirectionalLight(0xffffff, 1.5)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)
  const camera = new THREE.PerspectiveCamera(35, 1.5, 0.01, 20)
  camera.position.copy(center).add(new THREE.Vector3(0, -5, 2))
  camera.lookAt(center)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 144,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.1, 'CompareBaseColor should render visible base-color comparison spheres')
})

test('committed Khronos glTF Sample Assets CompareClearcoat fixture loads clearcoat comparison variants', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_COMPARE_CLEARCOAT, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_clearcoat', 'KHR_materials_ior'])
  assert.equal(source.buffers[0].uri, 'CompareClearcoat.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'Compare_Clearcoat_img0.jpg',
    'Compare_Clearcoat_img1.jpg',
  ])
  assert.deepEqual(source.materials.map((material) => [
    material.name,
    material.pbrMetallicRoughness?.baseColorTexture?.index ?? null,
    material.pbrMetallicRoughness?.metallicRoughnessTexture?.index ?? null,
    material.extensions?.KHR_materials_clearcoat?.clearcoatFactor ?? null,
    material.extensions?.KHR_materials_ior?.ior ?? null,
  ]), [
    ['green glossy', 0, 1, null, null],
    ['green rough', 0, 1, null, null],
    ['green clearcoat', 0, 1, 1, 1.6],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_COMPARE_CLEARCOAT)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), ['GeoSphere001', 'GeoSphere002', 'GeoSphere003'])
  assert.deepEqual(meshes.map((mesh) => mesh.material.name), ['green glossy', 'green rough', 'green clearcoat'])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [673, 673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [673, 673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [673, 673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [3840, 3840, 3840])

  const [glossy, rough, clearcoat] = meshes.map((mesh) => mesh.material)
  assert.equal(glossy.isMeshStandardMaterial, true)
  assert.equal(rough.isMeshStandardMaterial, true)
  assert.equal(clearcoat.isMeshPhysicalMaterial, true)
  assert.equal(glossy.roughness, 0)
  assert.equal(rough.roughness, 0.5)
  assert.equal(clearcoat.roughness, 0.5)
  assert.equal(clearcoat.clearcoat, 1)
  assert.equal(clearcoat.clearcoatRoughness, 0)
  assert.equal(clearcoat.ior, 1.6)
  assert.ok([glossy, rough, clearcoat].every((material) => material.metalness === 1))

  assert.equal(glossy.map, rough.map)
  assert.equal(rough.map, clearcoat.map)
  assert.equal(Buffer.isBuffer(glossy.map?.image), true, 'clearcoat base-color JPEG should load as an encoded Buffer')
  assert.equal(glossy.map.name, 'Compare_Clearcoat_img0.jpg')
  assert.equal(glossy.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(glossy.map.flipY, false)

  assert.equal(glossy.roughnessMap, glossy.metalnessMap)
  assert.equal(rough.roughnessMap, rough.metalnessMap)
  assert.equal(clearcoat.roughnessMap, clearcoat.metalnessMap)
  assert.equal(glossy.roughnessMap, rough.roughnessMap)
  assert.equal(rough.roughnessMap, clearcoat.roughnessMap)
  assert.equal(Buffer.isBuffer(glossy.roughnessMap?.image), true, 'clearcoat metallic-roughness JPEG should load as an encoded Buffer')
  assert.equal(glossy.roughnessMap.name, 'Compare_Clearcoat_img1.jpg')
  assert.equal(glossy.roughnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(glossy.roughnessMap.flipY, false)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1))
  const light = new THREE.DirectionalLight(0xffffff, 4)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)
  const camera = new THREE.PerspectiveCamera(35, 1.7, 0.01, 20)
  camera.position.copy(center).add(new THREE.Vector3(0, -3.8, 1.6))
  camera.lookAt(center)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 160,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.SRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.18, 'CompareClearcoat should render visible clearcoat comparison spheres')
})

test('committed Khronos glTF Sample Assets CompareDispersion fixture loads dispersion comparison variants', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_COMPARE_DISPERSION, 'utf8'))
  assert.deepEqual(source.extensionsUsed, [
    'KHR_materials_transmission',
    'KHR_materials_volume',
    'KHR_materials_ior',
    'KHR_materials_dispersion',
  ])
  assert.equal(source.buffers[0].uri, 'CompareDispersion.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'Compare_Dispersion_img0.jpg',
    'Compare_Dispersion_img1.jpg',
  ])
  assert.deepEqual(source.materials.map((material) => [
    material.name,
    material.pbrMetallicRoughness?.baseColorTexture?.index ?? null,
    material.extensions?.KHR_materials_transmission?.transmissionTexture?.index ?? null,
    material.extensions?.KHR_materials_volume?.thicknessFactor ?? null,
    material.extensions?.KHR_materials_ior?.ior ?? null,
    material.extensions?.KHR_materials_dispersion?.dispersion ?? null,
  ]), [
    ['checker', 1, null, null, null, null],
    ['No Dispersion', null, 0, 0.5, 2.42, null],
    ['Dispersion', null, 0, 0.5, 2.42, 5],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_COMPARE_DISPERSION)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), ['Checker', 'GeoSphere001', 'GeoSphere002'])
  assert.deepEqual(meshes.map((mesh) => mesh.material.name), ['checker', 'No Dispersion', 'Dispersion'])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [6, 96, 96])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [6, 96, 96])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [6, 96, 96])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [6, 96, 96])

  const [checker, noDispersion, dispersion] = meshes.map((mesh) => mesh.material)
  assert.equal(checker.isMeshStandardMaterial, true)
  assert.equal(noDispersion.isMeshPhysicalMaterial, true)
  assert.equal(dispersion.isMeshPhysicalMaterial, true)
  assert.equal(noDispersion.transmission, 1)
  assert.equal(dispersion.transmission, 1)
  assert.equal(noDispersion.thickness, 0.5)
  assert.equal(dispersion.thickness, 0.5)
  assert.equal(noDispersion.attenuationDistance, 1)
  assert.equal(dispersion.attenuationDistance, 1)
  assert.equal(noDispersion.ior, 2.42)
  assert.equal(dispersion.ior, 2.42)
  assert.equal(noDispersion.dispersion, 0)
  assert.equal(dispersion.dispersion, 5)
  assert.equal(noDispersion.roughness, 0.1)
  assert.equal(dispersion.roughness, 0.1)

  assert.equal(noDispersion.transmissionMap, dispersion.transmissionMap)
  assert.equal(Buffer.isBuffer(noDispersion.transmissionMap?.image), true, 'dispersion transmission JPEG should load as an encoded Buffer')
  assert.equal(noDispersion.transmissionMap.name, 'Compare_Dispersion_img0.jpg')
  assert.equal(noDispersion.transmissionMap.colorSpace, THREE.NoColorSpace)
  assert.equal(noDispersion.transmissionMap.flipY, false)

  assert.equal(Buffer.isBuffer(checker.map?.image), true, 'dispersion checker JPEG should load as an encoded Buffer')
  assert.equal(checker.map.name, 'Compare_Dispersion_img1.jpg')
  assert.equal(checker.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(checker.map.flipY, false)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1))
  const light = new THREE.DirectionalLight(0xffffff, 3)
  light.position.set(2, 4, 5)
  gltf.scene.add(light)
  const camera = new THREE.PerspectiveCamera(35, 1.4, 0.01, 20)
  camera.position.copy(center).add(new THREE.Vector3(0, -2.6, 1.2))
  camera.lookAt(center)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 128,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.SRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.6, 'CompareDispersion should render visible dispersion comparison geometry')
})

test('committed Khronos glTF Sample Assets CompareEmissiveStrength fixture loads emissive strength variants', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_COMPARE_EMISSIVE_STRENGTH, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_emissive_strength'])
  assert.equal(source.buffers[0].uri, 'CompareEmissiveStrength.bin')
  assert.deepEqual(source.images.map((image) => image.uri), ['Compare_Emissive-Strength_img0.jpg'])
  assert.deepEqual(source.materials.map((material) => [
    material.name,
    material.emissiveTexture?.index ?? null,
    material.extensions?.KHR_materials_emissive_strength?.emissiveStrength ?? null,
  ]), [
    ['glTF Logo Emissive', 0, null],
    ['glTF Logo Emissive Strength', 0, 3],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_COMPARE_EMISSIVE_STRENGTH)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), ['GeoSphere001', 'GeoSphere002'])
  assert.deepEqual(meshes.map((mesh) => mesh.material.name), ['glTF Logo Emissive', 'glTF Logo Emissive Strength'])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [3840, 3840])

  const [baseline, strengthened] = meshes.map((mesh) => mesh.material)
  assert.deepEqual(baseline.color.toArray(), [0, 0, 0])
  assert.deepEqual(strengthened.color.toArray(), [0, 0, 0])
  assert.deepEqual(baseline.emissive.toArray(), [1, 1, 1])
  assert.deepEqual(strengthened.emissive.toArray(), [1, 1, 1])
  assert.equal(baseline.emissiveIntensity, 1)
  assert.equal(strengthened.emissiveIntensity, 3)
  assert.equal(baseline.emissiveMap, strengthened.emissiveMap, 'both emissive-strength materials should share the emissive texture')
  assert.equal(Buffer.isBuffer(baseline.emissiveMap?.image), true, 'emissive JPEG should load as an encoded Buffer')
  assert.equal(baseline.emissiveMap.name, 'Compare_Emissive-Strength_img0.jpg')
  assert.equal(baseline.emissiveMap.colorSpace, THREE.SRGBColorSpace)
  assert.equal(baseline.emissiveMap.flipY, false)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const camera = new THREE.PerspectiveCamera(35, 1.5, 0.01, 20)
  camera.position.copy(center).add(new THREE.Vector3(0, -2.7, 1.4))
  camera.lookAt(center)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 144,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.SRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.02, 'CompareEmissiveStrength should render visible emissive comparison spheres')
})

test('committed Khronos glTF Sample Assets CompareIridescence fixture loads iridescence comparison variants', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_COMPARE_IRIDESCENCE, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_iridescence'])
  assert.equal(source.buffers[0].uri, 'CompareIridescence.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'Compare_Iridescence_img0.jpg',
    'Compare_Iridescence_img1.jpg',
    'Compare_Iridescence_img2.jpg',
  ])
  assert.deepEqual(source.materials.map((material) => [
    material.name,
    material.pbrMetallicRoughness?.baseColorTexture?.index ?? null,
    material.pbrMetallicRoughness?.metallicRoughnessTexture?.index ?? null,
    material.extensions?.KHR_materials_iridescence?.iridescenceFactor ?? null,
    material.extensions?.KHR_materials_iridescence?.iridescenceIor ?? null,
    material.extensions?.KHR_materials_iridescence?.iridescenceTexture?.index ?? null,
  ]), [
    ['glTF Logo', 0, 1, null, null, null],
    ['glTF Logo Iridescence', 0, 1, 1, 1.5, 2],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_COMPARE_IRIDESCENCE)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), ['GeoSphere001', 'GeoSphere002'])
  assert.deepEqual(meshes.map((mesh) => mesh.material.name), ['glTF Logo', 'glTF Logo Iridescence'])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [3840, 3840])

  const [baseline, iridescent] = meshes.map((mesh) => mesh.material)
  assert.equal(baseline.isMeshStandardMaterial, true)
  assert.equal(iridescent.isMeshPhysicalMaterial, true)
  assert.equal(baseline.metalness, 1)
  assert.equal(iridescent.metalness, 1)
  assert.equal(baseline.roughness, 0.69999)
  assert.equal(iridescent.roughness, 0.69999)
  assert.equal(iridescent.iridescence, 1)
  assert.equal(iridescent.iridescenceIOR, 1.5)
  assert.deepEqual(iridescent.iridescenceThicknessRange, [100, 400])
  assert.equal(baseline.iridescenceMap ?? null, null)
  assert.equal(Buffer.isBuffer(iridescent.iridescenceMap?.image), true, 'iridescence JPEG should load as an encoded Buffer')
  assert.equal(iridescent.iridescenceMap.name, 'Compare_Iridescence_img2.jpg')
  assert.equal(iridescent.iridescenceMap.colorSpace, THREE.NoColorSpace)
  assert.equal(iridescent.iridescenceMap.flipY, false)

  assert.equal(baseline.map, iridescent.map, 'iridescence comparison materials should share the base-color texture')
  assert.equal(Buffer.isBuffer(baseline.map?.image), true, 'iridescence base-color JPEG should load as an encoded Buffer')
  assert.equal(baseline.map.name, 'Compare_Iridescence_img0.jpg')
  assert.equal(baseline.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(baseline.map.flipY, false)

  assert.equal(baseline.roughnessMap, baseline.metalnessMap)
  assert.equal(iridescent.roughnessMap, iridescent.metalnessMap)
  assert.equal(baseline.roughnessMap, iridescent.roughnessMap)
  assert.equal(Buffer.isBuffer(baseline.roughnessMap?.image), true, 'iridescence metallic-roughness JPEG should load as an encoded Buffer')
  assert.equal(baseline.roughnessMap.name, 'Compare_Iridescence_img1.jpg')
  assert.equal(baseline.roughnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(baseline.roughnessMap.flipY, false)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1))
  const light = new THREE.DirectionalLight(0xffffff, 4)
  light.position.set(2, 4, 5)
  gltf.scene.add(light)
  const camera = new THREE.PerspectiveCamera(35, 1.5, 0.01, 20)
  camera.position.copy(center).add(new THREE.Vector3(0, -2.7, 1.4))
  camera.lookAt(center)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 144,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.SRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.02, 'CompareIridescence should render visible iridescence comparison geometry')
})

test('committed Khronos glTF Sample Assets CompareMetallic fixture loads metallic texture comparison variants', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_COMPARE_METALLIC, 'utf8'))
  assert.equal(source.buffers[0].uri, 'CompareMetallic.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'Compare_Metallic_img0.jpg',
    'Compare_Metallic_img1.jpg',
  ])
  assert.deepEqual(source.materials.map((material) => [
    material.name,
    material.pbrMetallicRoughness?.baseColorTexture?.index ?? null,
    material.pbrMetallicRoughness?.metallicRoughnessTexture?.index ?? null,
    material.pbrMetallicRoughness?.metallicFactor ?? null,
    material.pbrMetallicRoughness?.roughnessFactor ?? null,
  ]), [
    ['glTF Logo', 0, null, 0, 0.1],
    ['glTF Logo Metallic', 0, 1, null, 0.1],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_COMPARE_METALLIC)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), ['GeoSphere001', 'GeoSphere002'])
  assert.deepEqual(meshes.map((mesh) => mesh.material.name), ['glTF Logo', 'glTF Logo Metallic'])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [3840, 3840])

  const [dielectric, metallic] = meshes.map((mesh) => mesh.material)
  assert.equal(dielectric.metalness, 0)
  assert.equal(dielectric.roughness, 0.1)
  assert.equal(metallic.metalness, 1)
  assert.equal(metallic.roughness, 0.1)
  assert.equal(dielectric.map, metallic.map, 'both metallic comparison materials should share the base-color texture')
  assert.equal(Buffer.isBuffer(dielectric.map?.image), true, 'base-color JPEG should load as an encoded Buffer')
  assert.equal(dielectric.map.name, 'Compare_Metallic_img0.jpg')
  assert.equal(dielectric.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(dielectric.map.flipY, false)

  assert.equal(dielectric.metalnessMap ?? null, null)
  assert.equal(dielectric.roughnessMap ?? null, null)
  assert.equal(metallic.metalnessMap, metallic.roughnessMap, 'metalness and roughness should share the packed texture')
  assert.equal(Buffer.isBuffer(metallic.metalnessMap?.image), true, 'metallic-roughness JPEG should load as an encoded Buffer')
  assert.equal(metallic.metalnessMap.name, 'Compare_Metallic_img1.jpg')
  assert.equal(metallic.metalnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(metallic.metalnessMap.flipY, false)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.75))
  const light = new THREE.DirectionalLight(0xffffff, 2.2)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)
  const camera = new THREE.PerspectiveCamera(35, 1.5, 0.01, 20)
  camera.position.copy(center).add(new THREE.Vector3(0, -2.7, 1.4))
  camera.lookAt(center)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 144,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.25, 'CompareMetallic should render visible metallic comparison spheres')
})

test('committed Khronos glTF Sample Assets CompareRoughness fixture loads roughness texture comparison variants', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_COMPARE_ROUGHNESS, 'utf8'))
  assert.equal(source.buffers[0].uri, 'CompareRoughness.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'Compare_Roughness_img0.jpg',
    'Compare_Roughness_img1.jpg',
  ])
  assert.deepEqual(source.materials.map((material) => [
    material.name,
    material.pbrMetallicRoughness?.baseColorTexture?.index ?? null,
    material.pbrMetallicRoughness?.metallicRoughnessTexture?.index ?? null,
    material.pbrMetallicRoughness?.metallicFactor ?? null,
    material.pbrMetallicRoughness?.roughnessFactor ?? null,
  ]), [
    ['glTF Logo', 0, null, 0, 0],
    ['glTF Logo Roughness', 0, 1, 0, 0.5],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_COMPARE_ROUGHNESS)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), ['GeoSphere001', 'GeoSphere002'])
  assert.deepEqual(meshes.map((mesh) => mesh.material.name), ['glTF Logo', 'glTF Logo Roughness'])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [3840, 3840])

  const [smooth, rough] = meshes.map((mesh) => mesh.material)
  assert.equal(smooth.metalness, 0)
  assert.equal(smooth.roughness, 0)
  assert.equal(rough.metalness, 0)
  assert.equal(rough.roughness, 0.5)
  assert.equal(smooth.map, rough.map, 'both roughness comparison materials should share the base-color texture')
  assert.equal(Buffer.isBuffer(smooth.map?.image), true, 'base-color JPEG should load as an encoded Buffer')
  assert.equal(smooth.map.name, 'Compare_Roughness_img0.jpg')
  assert.equal(smooth.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(smooth.map.flipY, false)

  assert.equal(smooth.metalnessMap ?? null, null)
  assert.equal(smooth.roughnessMap ?? null, null)
  assert.equal(rough.metalnessMap, rough.roughnessMap, 'metalness and roughness should share the packed texture')
  assert.equal(Buffer.isBuffer(rough.roughnessMap?.image), true, 'metallic-roughness JPEG should load as an encoded Buffer')
  assert.equal(rough.roughnessMap.name, 'Compare_Roughness_img1.jpg')
  assert.equal(rough.roughnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(rough.roughnessMap.flipY, false)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 2))
  const light = new THREE.DirectionalLight(0xffffff, 6)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)
  const camera = new THREE.PerspectiveCamera(35, 1.5, 0.01, 20)
  camera.position.copy(center).add(new THREE.Vector3(0, -2.7, 1.4))
  camera.lookAt(center)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 144,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.SRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.045, 'CompareRoughness should render visible roughness comparison spheres')
})

test('committed Khronos glTF Sample Assets CompareSheen fixture loads sheen comparison variants', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_COMPARE_SHEEN, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_sheen'])
  assert.equal(source.buffers[0].uri, 'CompareSheen.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'Compare_Sheen_img0.jpg',
    'Compare_Sheen_img1.jpg',
    'Compare_Sheen_img2.jpg',
  ])
  assert.deepEqual(source.materials.map((material) => [
    material.name,
    material.normalTexture?.index ?? null,
    material.normalTexture?.scale ?? null,
    material.pbrMetallicRoughness?.baseColorTexture?.index ?? null,
    material.extensions?.KHR_materials_sheen?.sheenColorFactor ?? null,
    material.extensions?.KHR_materials_sheen?.sheenRoughnessFactor ?? null,
  ]), [
    ['glTF Logo', 1, 0.5, 2, null, null],
    ['glTF Logo Sheen', 1, 0.5, 0, [1, 0, 0], 0.3],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_COMPARE_SHEEN)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), ['GeoSphere001', 'GeoSphere002'])
  assert.deepEqual(meshes.map((mesh) => mesh.material.name), ['glTF Logo', 'glTF Logo Sheen'])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [3840, 3840])

  const [baseline, sheen] = meshes.map((mesh) => mesh.material)
  assert.equal(baseline.isMeshStandardMaterial, true)
  assert.equal(sheen.isMeshPhysicalMaterial, true)
  assert.equal(baseline.metalness, 0)
  assert.equal(sheen.metalness, 0)
  assert.equal(baseline.roughness, 0.75)
  assert.equal(sheen.roughness, 0.75)
  assert.equal(sheen.sheen, 1)
  assertVectorClose(sheen.sheenColor.toArray(), [1, 0, 0], 'CompareSheen sheenColorFactor')
  assert.equal(sheen.sheenRoughness, 0.3)
  assert.equal(baseline.sheenColorMap ?? null, null)
  assert.equal(sheen.sheenColorMap ?? null, null)
  assert.equal(sheen.sheenRoughnessMap ?? null, null)

  assert.notEqual(baseline.map, sheen.map, 'sheen comparison materials intentionally use different base-color textures')
  assert.equal(Buffer.isBuffer(baseline.map?.image), true, 'baseline base-color JPEG should load as an encoded Buffer')
  assert.equal(Buffer.isBuffer(sheen.map?.image), true, 'sheen base-color JPEG should load as an encoded Buffer')
  assert.equal(baseline.map.name, 'Compare_Sheen_img2.jpg')
  assert.equal(sheen.map.name, 'Compare_Sheen_img0.jpg')
  assert.equal(baseline.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(sheen.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(baseline.map.flipY, false)
  assert.equal(sheen.map.flipY, false)

  assert.equal(baseline.normalMap, sheen.normalMap, 'both sheen comparison materials should share the normal map')
  assert.equal(Buffer.isBuffer(baseline.normalMap?.image), true, 'shared normal JPEG should load as an encoded Buffer')
  assert.equal(baseline.normalMap.name, 'Compare_Sheen_img1.jpg')
  assert.equal(baseline.normalMap.colorSpace, THREE.NoColorSpace)
  assert.equal(baseline.normalMap.flipY, false)
  assertVectorClose(baseline.normalScale.toArray(), [0.5, -0.5], 'CompareSheen baseline normal scale')
  assertVectorClose(sheen.normalScale.toArray(), [0.5, -0.5], 'CompareSheen sheen normal scale')

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1.2))
  const light = new THREE.DirectionalLight(0xffffff, 4)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)
  const camera = new THREE.PerspectiveCamera(35, 1.5, 0.01, 20)
  camera.position.copy(center).add(new THREE.Vector3(0, -2.7, 1.4))
  camera.lookAt(center)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 144,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.SRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.25, 'CompareSheen should render visible sheen comparison spheres')
})

test('committed Khronos glTF Sample Assets CompareSpecular fixture loads specular extension comparison variants', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_COMPARE_SPECULAR, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_specular'])
  assert.equal(source.buffers[0].uri, 'CompareSpecular.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'Compare_Specular_img0.jpg',
    'Compare_Specular_img1.png',
  ])
  assert.deepEqual(source.materials.map((material) => [
    material.name,
    material.pbrMetallicRoughness?.baseColorTexture?.index ?? null,
    material.extensions?.KHR_materials_specular?.specularFactor ?? null,
    material.extensions?.KHR_materials_specular?.specularTexture?.index ?? null,
    material.extensions?.KHR_materials_specular?.specularColorTexture?.index ?? null,
  ]), [
    ['glTF Logo', 0, null, null, null],
    ['glTF Logo Specular', 0, 1, 1, 1],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_COMPARE_SPECULAR)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), ['GeoSphere001', 'GeoSphere002'])
  assert.deepEqual(meshes.map((mesh) => mesh.material.name), ['glTF Logo', 'glTF Logo Specular'])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [2625, 2625])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [2625, 2625])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [2625, 2625])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [15360, 15360])

  const [baseline, specular] = meshes.map((mesh) => mesh.material)
  assert.equal(baseline.isMeshStandardMaterial, true)
  assert.equal(specular.isMeshPhysicalMaterial, true)
  assert.equal(baseline.roughness, 0.4)
  assert.equal(specular.roughness, 0.4)
  assert.equal(baseline.map, specular.map, 'both specular comparison materials should share the base-color texture')
  assert.equal(Buffer.isBuffer(baseline.map?.image), true, 'base-color JPEG should load as an encoded Buffer')
  assert.equal(baseline.map.name, 'Compare_Specular_img0.jpg')
  assert.equal(baseline.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(baseline.map.flipY, false)

  assert.equal(baseline.specularIntensityMap ?? null, null)
  assert.equal(baseline.specularColorMap ?? null, null)
  assert.equal(specular.specularIntensity, 1)
  assertVectorClose(specular.specularColor.toArray(), [10, 10, 10], 'CompareSpecular specularColorFactor')
  assert.equal(specular.specularIntensityMap, specular.specularColorMap, 'specular intensity and color should share the extension texture')
  assert.equal(Buffer.isBuffer(specular.specularColorMap?.image), true, 'specular PNG should load as an encoded Buffer')
  assert.equal(specular.specularColorMap.name, 'Compare_Specular_img1.png')
  assert.deepEqual(pngDimensions(specular.specularColorMap.image), [1024, 512])
  assert.equal(specular.specularColorMap.colorSpace, THREE.SRGBColorSpace)
  assert.equal(specular.specularColorMap.flipY, false)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1))
  const light = new THREE.DirectionalLight(0xffffff, 4)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)
  const camera = new THREE.PerspectiveCamera(35, 1.5, 0.01, 20)
  camera.position.copy(center).add(new THREE.Vector3(0, -3, 1.4))
  camera.lookAt(center)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 144,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.SRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.2, 'CompareSpecular should render visible specular comparison spheres')
})

test('committed Khronos glTF Sample Assets CompareTransmission fixture loads alpha versus transmission variants', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_COMPARE_TRANSMISSION, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_transmission'])
  assert.equal(source.buffers[0].uri, 'CompareTransmission.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'Compare_Transmission_img0.jpg',
    'Compare_Transmission_img1.png',
    'Compare_Transmission_img2.png',
    'Compare_Transmission_img3.jpg',
  ])
  assert.deepEqual(source.materials.map((material) => [
    material.name,
    material.alphaMode ?? 'OPAQUE',
    material.pbrMetallicRoughness?.baseColorTexture?.index ?? null,
    material.pbrMetallicRoughness?.metallicRoughnessTexture?.index ?? null,
    material.extensions?.KHR_materials_transmission?.transmissionFactor ?? null,
  ]), [
    ['checker', 'OPAQUE', 2, null, null],
    ['glTF Alpha', 'BLEND', 1, 0, null],
    ['gold', 'OPAQUE', null, null, null],
    ['glTF Transmission', 'OPAQUE', null, 0, 1],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_COMPARE_TRANSMISSION)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'Sphere001_0',
    'Sphere001_1',
    'Sphere002_0',
    'Sphere002_1',
    'Checker',
  ])
  assert.deepEqual(meshes.map((mesh) => mesh.material.name), [
    'glTF Alpha',
    'gold',
    'gold',
    'glTF Transmission',
    'checker',
  ])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [2732, 390, 390, 2732, 6])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [2732, 390, 390, 2732, 6])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [2732, 390, 390, 2732, 6])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [15744, 1920, 1920, 15744, 6])

  const [alphaShell, goldLeft, goldRight, transmissionShell, checker] = meshes.map((mesh) => mesh.material)
  assert.equal(alphaShell.transparent, true)
  assert.equal(alphaShell.isMeshStandardMaterial, true)
  assert.equal(transmissionShell.isMeshPhysicalMaterial, true)
  assert.equal(transmissionShell.transmission, 1)
  assert.equal(transmissionShell.transparent, false)
  assert.equal(goldLeft, goldRight, 'both comparison cores should share the gold material instance')
  assertVectorClose(goldLeft.color.toArray(), [
    0.8823530077934265,
    0.5921568870544434,
    0.250980406999588,
  ], 'CompareTransmission gold baseColorFactor')
  assert.equal(goldLeft.metalness, 1)
  assert.equal(goldLeft.roughness, 0.2)

  assert.equal(Buffer.isBuffer(alphaShell.map?.image), true, 'alpha-shell base-color PNG should load as an encoded Buffer')
  assert.equal(alphaShell.map.name, 'Compare_Transmission_img1.png')
  assert.deepEqual(pngDimensions(alphaShell.map.image), [2048, 1024])
  assert.equal(alphaShell.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(alphaShell.map.flipY, false)

  assert.equal(alphaShell.roughnessMap, alphaShell.metalnessMap)
  assert.equal(transmissionShell.roughnessMap, transmissionShell.metalnessMap)
  assert.equal(alphaShell.roughnessMap, transmissionShell.roughnessMap)
  assert.equal(Buffer.isBuffer(alphaShell.roughnessMap?.image), true, 'shared metallic-roughness JPEG should load as an encoded Buffer')
  assert.equal(alphaShell.roughnessMap.name, 'Compare_Transmission_img0.jpg')
  assert.equal(alphaShell.roughnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(alphaShell.roughnessMap.flipY, false)

  assert.equal(Buffer.isBuffer(checker.map?.image), true, 'checker PNG should load as an encoded Buffer')
  assert.equal(checker.map.name, 'Compare_Transmission_img2.png')
  assert.deepEqual(pngDimensions(checker.map.image), [64, 64])
  assert.equal(checker.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(checker.map.flipY, false)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1))
  const light = new THREE.DirectionalLight(0xffffff, 3)
  light.position.set(2, 4, 5)
  gltf.scene.add(light)
  const camera = new THREE.PerspectiveCamera(35, 1.5, 0.01, 20)
  camera.position.copy(center).add(new THREE.Vector3(0, -3.2, 1.4))
  camera.lookAt(center)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 144,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.SRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.4, 'CompareTransmission should render visible alpha/transmission comparison geometry')
})

test('committed Khronos glTF Sample Assets TransmissionOrderTest fixture loads alpha and transmission ordering cases', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_TRANSMISSION_ORDER_TEST, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_transmission', 'KHR_materials_volume'])
  assert.deepEqual(source.buffers, [{ byteLength: 2291932, uri: 'TransmissionOrderTest.bin' }])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'checkerboard.png',
    'alphaInACircle.png',
    'BlendMaskOpaqueLabels.png',
  ])
  assert.deepEqual(source.materials.map((material) => [
    material.name,
    material.alphaMode ?? 'OPAQUE',
    material.alphaCutoff ?? null,
    material.doubleSided ?? false,
    material.pbrMetallicRoughness?.baseColorTexture?.index ?? null,
    material.extensions?.KHR_materials_transmission?.transmissionFactor ?? null,
    material.extensions?.KHR_materials_volume?.thicknessFactor ?? null,
  ]), [
    ['Cloth Backdrop', 'OPAQUE', null, false, 0, null, null],
    ['Alpha Blend Material', 'BLEND', null, true, 1, null, null],
    ['Blue Glass Material', 'OPAQUE', null, false, null, 1, 0.4000000059604645],
    ['Alpha Mask Material', 'MASK', null, true, 2, null, null],
    ['Label Material', 'OPAQUE', null, false, 3, null, null],
    ['Opaque Material', 'OPAQUE', null, true, null, null, null],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_TRANSMISSION_ORDER_TEST)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 20)
  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'Cloth_Backdrop',
    'AlphaBlend',
    'Glass',
    'Glass001',
    'Glass002',
    'Glass003',
    'Glass004',
    'Glass005',
    'AlphaBlend001',
    'AlphaBlend002',
    'AlphaMask',
    'AlphaMask001',
    'AlphaMask002',
    'Labels',
    'Glass006',
    'Glass007',
    'Glass008',
    'Opaque',
    'Opaque001',
    'Opaque002',
  ])
  assert.deepEqual(meshes.slice(0, 4).map((mesh) => mesh.geometry.getAttribute('position')?.count), [62658, 4, 296, 296])
  assert.deepEqual(meshes.slice(0, 4).map((mesh) => mesh.geometry.index?.count), [131337, 6, 1764, 1764])

  const materials = new Map()
  for (const mesh of meshes) materials.set(mesh.material.name, mesh.material)

  const backdrop = materials.get('Cloth Backdrop')
  assert.equal(Buffer.isBuffer(backdrop.map?.image), true, 'checkerboard PNG should load as an encoded Buffer')
  assert.equal(backdrop.map.name, 'checkerboard')
  assert.deepEqual(pngDimensions(backdrop.map.image), [2048, 2048])
  assert.equal(backdrop.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(backdrop.map.flipY, false)

  const alphaBlend = materials.get('Alpha Blend Material')
  assert.equal(alphaBlend.transparent, true)
  assert.equal(alphaBlend.depthWrite, false)
  assert.equal(alphaBlend.side, THREE.DoubleSide)
  assert.equal(alphaBlend.alphaTest, 0)
  assert.equal(alphaBlend.map.name, 'alphaInACircle')
  assert.deepEqual(pngDimensions(alphaBlend.map.image), [256, 256])
  assert.equal(alphaBlend.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(alphaBlend.map.flipY, false)

  const alphaMask = materials.get('Alpha Mask Material')
  assert.equal(alphaMask.transparent, false)
  assert.equal(alphaMask.depthWrite, true)
  assert.equal(alphaMask.side, THREE.DoubleSide)
  assert.equal(alphaMask.alphaTest, 0.5)
  assert.equal(alphaMask.map.name, 'alphaInACircle')

  const glass = materials.get('Blue Glass Material')
  assert.equal(glass.isMeshPhysicalMaterial, true)
  assert.equal(glass.transmission, 1)
  assert.equal(glass.thickness, 0.4000000059604645)
  assert.equal(glass.attenuationDistance, 1)
  assertVectorClose(glass.attenuationColor.toArray(), [1, 1, 1], 'TransmissionOrderTest glass attenuation color')

  const label = materials.get('Label Material')
  assert.equal(Buffer.isBuffer(label.map?.image), true, 'label PNG should load as an encoded Buffer')
  assert.equal(label.map.name, 'BlendMaskOpaqueLabels')
  assert.deepEqual(pngDimensions(label.map.image), [256, 256])

  const opaque = materials.get('Opaque Material')
  assert.equal(opaque.side, THREE.DoubleSide)
  assert.equal(opaque.transparent, false)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 1.4)
  light.position.set(3, 4, 6)
  gltf.scene.add(light)
  const camera = new THREE.PerspectiveCamera(45, 1.5, 0.01, 100)
  camera.position.set(center.x, center.y - 6, center.z + 4)
  camera.lookAt(center)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 96,
    height: 64,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.15, 'TransmissionOrderTest should render visible transparent/transmissive ordering panels')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 15 && mean.g > 10 && mean.b > 10, `TransmissionOrderTest should render non-black layered output (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets CompareVolume fixture loads transmission volume variants', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_COMPARE_VOLUME, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_transmission', 'KHR_materials_volume'])
  assert.equal(source.buffers[0].uri, 'CompareVolume.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'Compare_Volume_img0.jpg',
    'Compare_Volume_img1.jpg',
    'Compare_Volume_img2.png',
    'Compare_Volume_img3.jpg',
  ])
  assert.deepEqual(source.materials.map((material) => [
    material.name,
    material.pbrMetallicRoughness?.baseColorTexture?.index ?? null,
    material.pbrMetallicRoughness?.metallicRoughnessTexture?.index ?? null,
    material.extensions?.KHR_materials_transmission?.transmissionFactor ?? null,
    material.extensions?.KHR_materials_volume?.thicknessFactor ?? null,
    material.extensions?.KHR_materials_volume?.thicknessTexture?.index ?? null,
  ]), [
    ['checker', 2, null, null, null, null],
    ['glTF Transmission', null, 0, 1, null, null],
    ['gold', null, null, null, null, null],
    ['glTF Volume', null, 0, 1, 0.75, 1],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_COMPARE_VOLUME)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'Sphere001_0',
    'Sphere001_1',
    'Sphere002_0',
    'Sphere002_1',
    'Checker',
  ])
  assert.deepEqual(meshes.map((mesh) => mesh.material.name), [
    'glTF Transmission',
    'gold',
    'glTF Volume',
    'gold',
    'checker',
  ])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [2732, 390, 2732, 390, 6])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [2732, 390, 2732, 390, 6])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [2732, 390, 2732, 390, 6])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [15744, 1920, 15744, 1920, 6])

  const [transmissionShell, goldLeft, volumeShell, goldRight, checker] = meshes.map((mesh) => mesh.material)
  assert.equal(transmissionShell.isMeshPhysicalMaterial, true)
  assert.equal(volumeShell.isMeshPhysicalMaterial, true)
  assert.equal(transmissionShell.transmission, 1)
  assert.equal(volumeShell.transmission, 1)
  assert.equal(transmissionShell.thickness, 0)
  assert.equal(volumeShell.thickness, 0.75)
  assert.equal(volumeShell.attenuationDistance, 0.25)
  assertVectorClose(volumeShell.attenuationColor.toArray(), [0.15, 1, 0.5], 'CompareVolume attenuation color')
  assert.equal(goldLeft, goldRight, 'both volume comparison cores should share the gold material instance')
  assertVectorClose(goldLeft.color.toArray(), [
    0.8823530077934265,
    0.5921568870544434,
    0.250980406999588,
  ], 'CompareVolume gold baseColorFactor')
  assert.equal(goldLeft.metalness, 1)
  assert.equal(goldLeft.roughness, 0.2)

  assert.equal(transmissionShell.roughnessMap, transmissionShell.metalnessMap)
  assert.equal(volumeShell.roughnessMap, volumeShell.metalnessMap)
  assert.equal(transmissionShell.roughnessMap, volumeShell.roughnessMap)
  assert.equal(Buffer.isBuffer(volumeShell.roughnessMap?.image), true, 'shared metallic-roughness JPEG should load as an encoded Buffer')
  assert.equal(volumeShell.roughnessMap.name, 'Compare_Volume_img0.jpg')
  assert.equal(volumeShell.roughnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(volumeShell.roughnessMap.flipY, false)

  assert.equal(transmissionShell.thicknessMap ?? null, null)
  assert.equal(Buffer.isBuffer(volumeShell.thicknessMap?.image), true, 'volume thickness JPEG should load as an encoded Buffer')
  assert.equal(volumeShell.thicknessMap.name, 'Compare_Volume_img1.jpg')
  assert.equal(volumeShell.thicknessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(volumeShell.thicknessMap.flipY, false)

  assert.equal(Buffer.isBuffer(checker.map?.image), true, 'checker PNG should load as an encoded Buffer')
  assert.equal(checker.map.name, 'Compare_Volume_img2.png')
  assert.deepEqual(pngDimensions(checker.map.image), [64, 64])
  assert.equal(checker.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(checker.map.flipY, false)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1))
  const light = new THREE.DirectionalLight(0xffffff, 3)
  light.position.set(2, 4, 5)
  gltf.scene.add(light)
  const camera = new THREE.PerspectiveCamera(35, 1.5, 0.01, 20)
  camera.position.copy(center).add(new THREE.Vector3(0, -3.2, 1.4))
  camera.lookAt(center)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 144,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.SRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.35, 'CompareVolume should render visible volume comparison geometry')
})

test('committed Khronos glTF Sample Assets AttenuationTest fixture loads volume attenuation and thickness cases', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_ATTENUATION_TEST, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_transmission', 'KHR_materials_volume'])
  assert.deepEqual(source.buffers, [{ byteLength: 10584, uri: 'AttenuationTest.bin' }])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'AttenuationLabels.png',
    'ThicknessTexture.png',
    'PlainGrid.png',
  ])
  assert.equal(source.materials.length, 18)
  assert.deepEqual(source.materials.slice(0, 5).map((material) => [
    material.name,
    material.extensions?.KHR_materials_transmission?.transmissionFactor,
    material.extensions?.KHR_materials_volume?.thicknessFactor,
    material.extensions?.KHR_materials_volume?.attenuationDistance,
    material.extensions?.KHR_materials_volume?.attenuationColor,
  ]), [
    ['R2_and_R4_ThicknessFac_1.0', 1, 1, 1, [0.1, 0.5, 0.9]],
    ['R2_ThicknessFac_1.5', 1, 1.5, 1, [0.1, 0.5, 0.9]],
    ['R2_ThicknessFac_2.0', 1, 2, 1, [0.1, 0.5, 0.9]],
    ['R2_ThicknessFac_0.50', 1, 0.5, 1, [0.1, 0.5, 0.9]],
    ['R2_ThicknessFac_0.25', 1, 0.25, 1, [0.1, 0.5, 0.9]],
  ])
  assert.deepEqual(source.materials.slice(7, 12).map((material) => [
    material.name,
    material.extensions?.KHR_materials_volume?.thicknessFactor,
    material.extensions?.KHR_materials_volume?.attenuationDistance,
  ]), [
    ['R5_Attenuation_1.0', 1, 1],
    ['R5_Attenuation_1.5', 1, 0.6666666667],
    ['R5_Attenuation_2.0', 1, 0.5],
    ['R5_Attenuation_0.50', 1, 2],
    ['R5_Attenuation_0.25', 1, 4],
  ])
  assert.equal(source.materials[6].extensions.KHR_materials_volume.thicknessTexture.index, 1)

  const gltf = await loadGltfFixture(SAMPLE_ASSET_ATTENUATION_TEST)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 23)
  assert.deepEqual(meshes.slice(0, 6).map((mesh) => mesh.name), [
    'R2_Block_10',
    'R2_Block_15',
    'R2_Block_20',
    'R2_Block_050',
    'R2_Block_025',
    'Labels',
  ])

  const materials = new Map(meshes.map((mesh) => [mesh.material.name, mesh.material]))
  const thicknessOne = materials.get('R2_and_R4_ThicknessFac_1.0')
  assert.equal(thicknessOne.isMeshPhysicalMaterial, true)
  assert.equal(thicknessOne.transmission, 1)
  assert.equal(thicknessOne.thickness, 1)
  assert.equal(thicknessOne.attenuationDistance, 1)
  assertVectorClose(thicknessOne.attenuationColor.toArray(), [0.1, 0.5, 0.9], 'AttenuationTest attenuation color')
  assert.equal(materials.get('R2_ThicknessFac_2.0').thickness, 2)
  assert.equal(materials.get('R2_ThicknessFac_0.25').thickness, 0.25)
  assert.equal(materials.get('R5_Attenuation_2.0').attenuationDistance, 0.5)
  assert.equal(materials.get('R5_Attenuation_0.25').attenuationDistance, 4)

  const textureMaterial = materials.get('R3_ThicknessTex_Mat')
  assert.equal(textureMaterial.thickness, 2)
  assert.equal(Buffer.isBuffer(textureMaterial.thicknessMap?.image), true, 'thickness texture PNG should load as an encoded Buffer')
  assert.equal(textureMaterial.thicknessMap.name, 'ThicknessTexture')
  assert.deepEqual(pngDimensions(textureMaterial.thicknessMap.image), [256, 256])
  assert.equal(textureMaterial.thicknessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(textureMaterial.thicknessMap.flipY, false)

  const labelMaterial = materials.get('LabelMaterial')
  assert.equal(Buffer.isBuffer(labelMaterial.map?.image), true, 'attenuation label PNG should load as an encoded Buffer')
  assert.equal(labelMaterial.map.name, 'AttenuationLabels')
  assert.deepEqual(pngDimensions(labelMaterial.map.image), [512, 512])
  assert.equal(labelMaterial.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(labelMaterial.map.flipY, false)

  const backdrop = materials.get('FlatBackdrop')
  assert.equal(Buffer.isBuffer(backdrop.map?.image), true, 'plain grid PNG should load as an encoded Buffer')
  assert.equal(backdrop.map.name, 'PlainGrid')
  assert.deepEqual(pngDimensions(backdrop.map.image), [256, 256])
  assert.equal(backdrop.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(backdrop.map.flipY, false)

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.85))
  const light = new THREE.DirectionalLight(0xffffff, 1.3)
  light.position.set(2, 3, 8)
  gltf.scene.add(light)
  const camera = new THREE.OrthographicCamera(-10.8, 10.8, 10.8, -10.8, 0.01, 50)
  camera.position.set(0, 0, 22)
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.75, 'AttenuationTest should render visible attenuation and thickness panels')
  const center = meanRegion(rgba, 96, 96, 40, 40, 56, 56)
  assert.ok(center.r > 100 && center.g > 100 && center.b > 100, `AttenuationTest center panels should render visible transmission output (${center.r}, ${center.g}, ${center.b})`)
})

test('committed Khronos glTF Sample Assets CompareNormal fixture loads normal-map comparison variants', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_COMPARE_NORMAL, 'utf8'))
  assert.equal(source.buffers[0].uri, 'CompareNormal.bin')
  assert.deepEqual(source.images.map((image) => image.uri), ['Compare_Normal_img0.jpg'])
  assert.deepEqual(source.materials.map((material) => [
    material.name,
    material.normalTexture?.index ?? null,
    material.pbrMetallicRoughness?.baseColorFactor,
  ]), [
    ['Wicker no Normal', null, [0.501960813999176, 0.4392157196998596, 0.3529411852359772, 1]],
    ['Wicker with Normal', 0, [0.501960813999176, 0.4431372880935669, 0.3529411852359772, 1]],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_COMPARE_NORMAL)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), ['Sphere001', 'Sphere002'])
  assert.deepEqual(meshes.map((mesh) => mesh.material.name), ['Wicker no Normal', 'Wicker with Normal'])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [1538, 1728])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [1538, 1728])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count ?? null), [null, 1728])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [9216, 9216])

  const [flat, normalMapped] = meshes
  assert.equal(flat.material.normalMap ?? null, null)
  assertVectorClose(flat.material.color.toArray(), [
    0.501960813999176,
    0.4392157196998596,
    0.3529411852359772,
  ], 'CompareNormal no-normal baseColorFactor')
  assertVectorClose(normalMapped.material.color.toArray(), [
    0.501960813999176,
    0.4431372880935669,
    0.3529411852359772,
  ], 'CompareNormal normal-mapped baseColorFactor')

  assert.equal(normalMapped.material.metalness, 0)
  assert.equal(normalMapped.material.roughness, 0.25)
  assert.equal(Buffer.isBuffer(normalMapped.material.normalMap?.image), true, 'normal-map JPEG should load as an encoded Buffer')
  assert.ok(normalMapped.material.normalMap.image.length > 0, 'normal-map JPEG buffer should not be empty')
  assert.equal(normalMapped.material.normalMap.name, 'Compare_Normal_img0.jpg')
  assert.equal(normalMapped.material.normalMap.colorSpace, THREE.NoColorSpace)
  assert.equal(normalMapped.material.normalMap.flipY, false)
  assertVectorClose(normalMapped.material.normalScale.toArray(), [1, -1], 'glTF normal map should use Three.js Y-flipped normal scale')

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.55))
  const light = new THREE.DirectionalLight(0xffffff, 1.8)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)
  const camera = new THREE.PerspectiveCamera(35, 1.5, 0.01, 20)
  camera.position.copy(center).add(new THREE.Vector3(0, -3.2, 1.4))
  camera.lookAt(center)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 144,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.18, 'CompareNormal should render visible normal-map comparison spheres')
})

test('committed Khronos glTF Sample Assets Avocado fixture loads PBR texture maps', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_AVOCADO, 'utf8'))
  assert.equal(source.buffers[0].uri, 'Avocado.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'Avocado_baseColor.png',
    'Avocado_roughnessMetallic.png',
    'Avocado_normal.png',
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_AVOCADO)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'Khronos Avocado sample should load a mesh')
  assert.equal(mesh.name, 'Avocado')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 406)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 406)
  assert.equal(mesh.geometry.getAttribute('tangent')?.count, 406)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 406)
  assert.equal(mesh.geometry.index?.count, 2046)
  assert.equal(mesh.material.name, '2256_Avocado_d')

  const { map, roughnessMap, metalnessMap, normalMap } = mesh.material
  assert.ok(map?.isTexture, 'Avocado sample should load a base color texture')
  assert.ok(roughnessMap?.isTexture, 'Avocado sample should load a roughness texture')
  assert.ok(metalnessMap?.isTexture, 'Avocado sample should load a metalness texture')
  assert.ok(normalMap?.isTexture, 'Avocado sample should load a normal texture')
  assert.equal(roughnessMap, metalnessMap, 'Avocado metallic/roughness channels should share the packed texture')
  assert.deepEqual(pngDimensions(map.image), [2048, 2048])
  assert.deepEqual(pngDimensions(roughnessMap.image), [2048, 2048])
  assert.deepEqual(pngDimensions(normalMap.image), [2048, 2048])
  assert.equal(map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(roughnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(normalMap.colorSpace, THREE.NoColorSpace)
  assert.equal(map.flipY, false)
  assert.equal(roughnessMap.flipY, false)
  assert.equal(normalMap.flipY, false)

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 1.6)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const halfExtent = Math.max(size.x, size.y, size.z) * 0.75
  const camera = new THREE.OrthographicCamera(-halfExtent, halfExtent, halfExtent, -halfExtent, 0.001, 10)
  camera.position.set(center.x, center.y + 0.04, center.z + 0.14)
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.15, 'Khronos Avocado sample should render visible PBR textured pixels')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > mean.b + 10 && mean.g > mean.b + 10, `Avocado texture should contribute green/yellow output (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets BarramundiFish fixture loads organic mesh packed PBR maps', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_BARRAMUNDI_FISH, 'utf8'))
  assert.deepEqual(source.buffers, [{ uri: 'BarramundiFish.bin', byteLength: 128208 }])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'BarramundiFish_baseColor.png',
    'BarramundiFish_occlusionRoughnessMetallic.png',
    'BarramundiFish_normal.png',
  ])
  assert.equal(source.meshes[0].name, 'barramundi_fish_Hero')
  assert.equal(source.materials[0].name, '7288_barramundi fish_col')
  assert.equal(source.materials[0].normalTexture.index, 2)
  assert.equal(source.materials[0].occlusionTexture.index, 1)
  assert.equal(source.materials[0].pbrMetallicRoughness.baseColorTexture.index, 0)
  assert.equal(source.materials[0].pbrMetallicRoughness.metallicRoughnessTexture.index, 1)

  const gltf = await loadGltfFixture(SAMPLE_ASSET_BARRAMUNDI_FISH)
  const mesh = gltf.scene.getObjectByName('BarramundiFish')
  assert.ok(mesh?.isMesh, 'BarramundiFish sample should load a named fish mesh')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 2188)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 2188)
  assert.equal(mesh.geometry.getAttribute('tangent')?.count, 2188)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 2188)
  assert.equal(mesh.geometry.index?.count, 11592)
  assertVectorClose(mesh.quaternion.toArray(), [0, 1, 0, 0], 'BarramundiFish node rotation')
  assert.equal(mesh.material.name, '7288_barramundi fish_col')

  const { map, aoMap, roughnessMap, metalnessMap, normalMap } = mesh.material
  assert.ok(map?.isTexture, 'BarramundiFish sample should load a base color texture')
  assert.ok(aoMap?.isTexture, 'BarramundiFish sample should load an occlusion texture')
  assert.ok(roughnessMap?.isTexture, 'BarramundiFish sample should load a roughness texture')
  assert.ok(metalnessMap?.isTexture, 'BarramundiFish sample should load a metalness texture')
  assert.ok(normalMap?.isTexture, 'BarramundiFish sample should load a normal texture')
  assert.equal(aoMap, roughnessMap, 'BarramundiFish occlusion/roughness channels should share the packed texture')
  assert.equal(roughnessMap, metalnessMap, 'BarramundiFish metallic/roughness channels should share the packed texture')
  assert.equal(map.name, 'BarramundiFish_baseColor.png')
  assert.equal(aoMap.name, 'BarramundiFish_occlusionRoughnessMetallic.png')
  assert.equal(normalMap.name, 'BarramundiFish_normal.png')
  assert.deepEqual(pngDimensions(map.image), [2048, 2048])
  assert.deepEqual(pngDimensions(aoMap.image), [2048, 2048])
  assert.deepEqual(pngDimensions(normalMap.image), [2048, 2048])
  assert.equal(map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(aoMap.colorSpace, THREE.NoColorSpace)
  assert.equal(normalMap.colorSpace, THREE.NoColorSpace)

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.9))
  const light = new THREE.DirectionalLight(0xffffff, 2)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)
  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const halfHeight = Math.max(size.y, size.z) / 2 + 0.04
  const halfWidth = halfHeight * 1.5
  const camera = new THREE.OrthographicCamera(-halfWidth, halfWidth, halfHeight, -halfHeight, 0.01, 20)
  camera.position.set(center.x + 3, center.y, center.z)
  camera.lookAt(center)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 144,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.07, 'BarramundiFish should render visible packed-PBR textured organic geometry')
})

test('committed Khronos glTF Sample Assets BoomBox fixture loads emissive and packed ORM maps', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_BOOM_BOX, 'utf8'))
  assert.equal(source.buffers[0].uri, 'BoomBox.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'BoomBox_baseColor.png',
    'BoomBox_occlusionRoughnessMetallic.png',
    'BoomBox_normal.png',
    'BoomBox_emissive.png',
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_BOOM_BOX)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'Khronos BoomBox sample should load a mesh')
  assert.equal(mesh.name, 'BoomBox')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 3575)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 3575)
  assert.equal(mesh.geometry.getAttribute('tangent')?.count, 3575)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 3575)
  assert.equal(mesh.geometry.index?.count, 18108)
  assert.equal(mesh.material.name, 'BoomBox_Mat')
  assert.deepEqual(mesh.material.emissive.toArray(), [1, 1, 1])

  const { map, aoMap, roughnessMap, metalnessMap, normalMap, emissiveMap } = mesh.material
  assert.ok(map?.isTexture, 'BoomBox sample should load a base color texture')
  assert.ok(aoMap?.isTexture, 'BoomBox sample should load an occlusion texture')
  assert.ok(roughnessMap?.isTexture, 'BoomBox sample should load a roughness texture')
  assert.ok(metalnessMap?.isTexture, 'BoomBox sample should load a metalness texture')
  assert.ok(normalMap?.isTexture, 'BoomBox sample should load a normal texture')
  assert.ok(emissiveMap?.isTexture, 'BoomBox sample should load an emissive texture')
  assert.equal(aoMap, roughnessMap, 'BoomBox occlusion/roughness channels should share the packed texture')
  assert.equal(roughnessMap, metalnessMap, 'BoomBox metallic/roughness channels should share the packed texture')
  assert.deepEqual(pngDimensions(map.image), [2048, 2048])
  assert.deepEqual(pngDimensions(aoMap.image), [2048, 2048])
  assert.deepEqual(pngDimensions(normalMap.image), [2048, 2048])
  assert.deepEqual(pngDimensions(emissiveMap.image), [2048, 2048])
  assert.equal(map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(aoMap.colorSpace, THREE.NoColorSpace)
  assert.equal(normalMap.colorSpace, THREE.NoColorSpace)
  assert.equal(emissiveMap.colorSpace, THREE.SRGBColorSpace)
  assert.equal(map.flipY, false)
  assert.equal(aoMap.flipY, false)
  assert.equal(normalMap.flipY, false)
  assert.equal(emissiveMap.flipY, false)

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 1.4)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const halfExtent = Math.max(size.x, size.y, size.z) * 0.72
  const camera = new THREE.OrthographicCamera(-halfExtent, halfExtent, halfExtent, -halfExtent, 0.001, 10)
  camera.position.set(center.x, center.y + 0.012, center.z + 0.05)
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.12, 'Khronos BoomBox sample should render visible textured pixels')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 8 && mean.g > 8 && mean.b > 8, `BoomBox textures should contribute non-black output (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets BoomBoxWithAxes fixture loads coordinate-system meshes and shared materials', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_BOOM_BOX_WITH_AXES, 'utf8'))
  assert.equal(source.buffers[0].uri, 'BoomBoxWithAxes.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'BoomBoxWithAxes_baseColor.png',
    'BoomBoxWithAxes_roughnessMetallic.png',
    'BoomBoxWithAxes_normal.png',
    'BoomBoxWithAxes_emissive.png',
    'BoomBoxWithAxes_baseColor1.png',
  ])
  assert.deepEqual(source.nodes[5].children, [0, 1, 2, 3, 4])
  assert.deepEqual(source.nodes[5].rotation, [0, 1, 0, 0])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_BOOM_BOX_WITH_AXES)
  const root = gltf.scene.getObjectByName('BoomBox_Coordinates')
  assert.deepEqual(root.children.map((child) => child.name), ['BoomBox', 'CoordinateSystem', 'X_axis', 'Y_axis', 'Z_axis'])
  assertVectorClose(root.quaternion.toArray(), [0, 1, 0, 0], 'BoomBoxWithAxes root rotation')

  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), ['BoomBox', 'CoordinateSystem', 'X_axis', 'Y_axis', 'Z_axis'])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [3575, 875, 2252, 1820, 1708])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [3575, 875, 2252, 1820, 1708])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('tangent')?.count), [3575, 875, 2252, 1820, 1708])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [3575, 875, 2252, 1820, 1708])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [18108, 3420, 11064, 8976, 8496])
  assert.ok(meshes.slice(2).every((mesh) => Math.abs(mesh.scale.x - 0.06) < 1e-12), 'axis meshes should retain imported scale transforms')

  const [boombox, coordinateSystem, xAxis, yAxis, zAxis] = meshes
  assert.equal(boombox.material.name, 'M_BoomBox')
  assert.ok([coordinateSystem, xAxis, yAxis, zAxis].every((mesh) => mesh.material.name === 'M_Coordinates'))
  const boomboxMaterial = boombox.material
  assert.equal(boomboxMaterial.map.name, 'BoomBoxWithAxes_baseColor.png')
  assert.deepEqual(pngDimensions(boomboxMaterial.map.image), [2048, 2048])
  assert.equal(boomboxMaterial.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(boomboxMaterial.roughnessMap, boomboxMaterial.metalnessMap)
  assert.equal(boomboxMaterial.roughnessMap.name, 'BoomBoxWithAxes_roughnessMetallic.png')
  assert.deepEqual(pngDimensions(boomboxMaterial.roughnessMap.image), [2048, 2048])
  assert.equal(boomboxMaterial.normalMap.name, 'BoomBoxWithAxes_normal.png')
  assert.deepEqual(pngDimensions(boomboxMaterial.normalMap.image), [2048, 2048])
  const coordinateMaterial = coordinateSystem.material
  assert.equal(coordinateMaterial.map.name, 'BoomBoxWithAxes_baseColor1.png')
  assert.deepEqual(pngDimensions(coordinateMaterial.map.image), [32, 32])
  assert.equal(coordinateMaterial.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(coordinateMaterial.metalness, 0)
  assert.equal(coordinateMaterial.roughness, 0.735)

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 1.6)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)
  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const halfExtent = Math.max(size.x, size.y, size.z) / 2 + 0.03
  const camera = new THREE.OrthographicCamera(-halfExtent, halfExtent, halfExtent, -halfExtent, 0.01, 20)
  camera.position.set(center.x + 0.3, center.y + 0.5, center.z + 0.9)
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
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.018, 'BoomBoxWithAxes should render visible boombox and coordinate-system geometry')
})

test('committed Khronos glTF Sample Assets BoxInterleaved fixture loads byteStride attributes', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_BOX_INTERLEAVED)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'Khronos BoxInterleaved sample should load a mesh')
  const position = mesh.geometry.getAttribute('position')
  const normal = mesh.geometry.getAttribute('normal')
  assert.equal(position?.count, 24)
  assert.equal(normal?.count, 24)
  assert.equal(mesh.geometry.index?.count, 36)
  assert.equal(position.isInterleavedBufferAttribute, true)
  assert.equal(normal.isInterleavedBufferAttribute, true)
  assert.equal(position.data.stride, 6)
  assert.deepEqual(vectorFromAttribute(position, 0), [-0.5, -0.5, 0.5])
  assert.deepEqual(vectorFromAttribute(normal, 0), [0, 0, 1])
  assert.equal(mesh.material.color.r, 0.800000011920929)
  assert.equal(mesh.material.color.g, 0)
  assert.equal(mesh.material.color.b, 0)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 10)
  camera.position.set(1.4, 1.1, 2.2)
  camera.lookAt(0, 0, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.55))
  const light = new THREE.DirectionalLight(0xffffff, 1.6)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.2, 'Khronos BoxInterleaved sample should render visible pixels')
  const center = meanRegion(rgba, 96, 96, 40, 40, 56, 56)
  assert.ok(center.r > center.b + 150 && center.r > center.g + 180, `BoxInterleaved sample should render a red cube (${center.r}, ${center.g}, ${center.b})`)
})

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

test('committed Khronos glTF Sample Assets PointLightIntensityTest fixture loads KHR_lights_punctual point lights', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_POINT_LIGHT_INTENSITY_TEST)
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_lights_punctual'))
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_unlit'))

  const meshes = []
  const lights = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
    if (object.isLight === true) lights.push(object)
  })
  assert.equal(meshes.length, 13)
  assert.equal(lights.length, 8)
  assert.ok(lights.every((light) => light.isPointLight === true), 'all imported punctual lights should become PointLight objects')
  assert.deepEqual(lights.map((light) => light.name), [
    'Light_4_-_White',
    'Light_1_-_Red',
    'Light_3_-_Blue',
    'Light_2_-_Green',
    'Light_5_-_Gray',
    'Light_6_B',
    'Light_6_G',
    'Light_6_R',
  ])
  assert.deepEqual(lights.map((light) => light.color.toArray()), [
    [1, 1, 1],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0.5, 0.5, 0.5],
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
  ])
  assert.ok(lights.every((light) => light.intensity === 1 && light.distance === 1.125 && light.decay === 2))

  gltf.scene.updateMatrixWorld(true)
  const firstLightPosition = lights[0].getWorldPosition(new THREE.Vector3()).toArray()
  const rgbLightPositions = lights.slice(5).map((light) => light.getWorldPosition(new THREE.Vector3()).toArray())
  assert.deepEqual(firstLightPosition, [0, -2.5, 0.20000000298023224])
  assert.deepEqual(rgbLightPositions, [
    [-2.25, -2.5, 0.20000000298023224],
    [-2.25, -2.5, 0.20000000298023224],
    [-2.25, -2.5, 0.20000000298023224],
  ])

  const label = meshes.find((mesh) => mesh.name === 'Labels')
  assert.equal(label?.material.isMeshBasicMaterial, true)
  assert.equal(Buffer.isBuffer(label.material.map?.image), true, 'point-light label PNG should load as an encoded Buffer')
  assert.equal(label.material.map.name, 'LampColorNames')
  assert.equal(label.material.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(label.material.map.flipY, false)

  const litSurface = meshes.find((mesh) => mesh.material.name === 'Test Surface Material')
  const frame = meshes.find((mesh) => mesh.material.name === 'Frame Material')
  assert.equal(litSurface?.geometry.getAttribute('position')?.count, 24)
  assert.equal(litSurface.geometry.index?.count, 36)
  assert.equal(frame?.geometry.getAttribute('position')?.count, 248)
  assert.equal(frame.geometry.index?.count, 768)

  const camera = new THREE.OrthographicCamera(-4.1, 4.1, 1.4, -4.0, 0.01, 20)
  camera.position.set(0, -1.25, 8)
  camera.lookAt(0, -1.25, 0)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 180,
    height: 120,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.2, 'PointLightIntensityTest should render visible point-light panels')
})

test('committed Khronos glTF Sample Assets LightVisibility fixture applies KHR_node_visibility to imported lights', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_LIGHT_VISIBILITY, 'utf8'))
  assert.deepEqual(source.extensionsRequired, ['KHR_lights_punctual', 'KHR_node_visibility'])
  assert.deepEqual(source.extensionsUsed, ['KHR_animation_pointer', 'KHR_lights_punctual', 'KHR_node_visibility'])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_LIGHT_VISIBILITY)
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_lights_punctual'))
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_node_visibility'))

  const meshes = []
  const lights = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
    if (object.isLight === true) lights.push(object)
  })

  assert.deepEqual(meshes.map((mesh) => mesh.name), ['QuadMeshNode'])
  assert.equal(lights.length, 5)
  assert.ok(lights.every((light) => light.isSpotLight === true), 'all imported punctual lights should become SpotLight objects')
  assert.deepEqual(lights.map((light) => light.name), [
    'InvisibleLight',
    'ChildOfInvisibleShouldBeInvisible',
    'DescendantOfInvisibleShouldBeInvisible',
    'VisibleLight',
    'AnimatedVisibility',
  ])
  assert.deepEqual(lights.map((light) => light.color.toArray()), [
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0.125, 1],
  ])
  assert.deepEqual(lights.map((light) => light.intensity), [5, 5, 5, 5, 6])
  assert.ok(lights.every((light) => light.distance === 5 && light.decay === 2 && light.angle === 0.8 && light.penumbra === 0.1875))

  gltf.scene.updateMatrixWorld(true)
  assertVectorClose(lights[0].getWorldPosition(new THREE.Vector3()).toArray(), [-1.5, 0, 1], 'InvisibleLight world position')
  assertVectorClose(lights[3].getWorldPosition(new THREE.Vector3()).toArray(), [0, 0, 1], 'VisibleLight world position')
  assertVectorClose(lights[4].getWorldPosition(new THREE.Vector3()).toArray(), [1.5, 0, 1], 'AnimatedVisibility world position')

  assert.equal(lights[0].visible, false, 'InvisibleLight should import KHR_node_visibility false')
  assert.equal(lights[1].visible, true, 'child light should keep its own default visible flag')
  assert.equal(lights[2].visible, true, 'descendant light should keep its own default visible flag')
  assert.equal(lights[3].visible, true)
  assert.equal(lights[4].visible, true)
  assert.equal(isEffectivelyVisible(lights[0]), false, 'InvisibleLight should be effectively hidden')
  assert.equal(isEffectivelyVisible(lights[1]), false, 'child light should be hidden by its invisible parent')
  assert.equal(isEffectivelyVisible(lights[2]), false, 'descendant light should be hidden by its invisible ancestor')
  assert.equal(isEffectivelyVisible(lights[3]), true)
  assert.equal(isEffectivelyVisible(lights[4]), true)

  const mesh = meshes[0]
  assert.equal(mesh.geometry.getAttribute('position')?.count, 4)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 4)
  assert.equal(mesh.geometry.index?.count, 6)
  assert.equal(mesh.material.isMeshStandardMaterial, true)
  assert.equal(mesh.material.roughness, 1)
  assert.equal(mesh.material.metalness, 1)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 10)
  camera.position.set(0, -2.4, 2.1)
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.3, 'LightVisibility should render visible green and blue spot-light contribution')
  const left = meanRegion(rgba, 96, 96, 12, 42, 34, 74)
  const center = meanRegion(rgba, 96, 96, 37, 42, 59, 74)
  const right = meanRegion(rgba, 96, 96, 62, 42, 84, 74)
  assert.ok(left.r < 10, `invisible red light branch should not tint the left panel red (${left.r}, ${left.g}, ${left.b})`)
  assert.ok(center.g > 80 && center.g > center.r + 80, `visible green light should tint the center panel (${center.r}, ${center.g}, ${center.b})`)
  assert.ok(right.b > 20 && right.b > right.r + 20, `visible animated blue light should tint the right panel (${right.r}, ${right.g}, ${right.b})`)
})

test('committed Khronos glTF Sample Assets CubeVisibility fixture applies KHR_node_visibility to meshes', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_CUBE_VISIBILITY, 'utf8'))
  assert.deepEqual(source.extensionsRequired, ['KHR_node_visibility'])
  assert.deepEqual(source.extensionsUsed, ['KHR_animation_pointer', 'KHR_node_visibility'])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_CUBE_VISIBILITY)
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_node_visibility'))

  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })

  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'InvisibleCube',
    'ChildOfInvisibleShouldBeInvisible',
    'DescendantOfInvisibleShouldBeInvisible',
    'VisibleCube',
    'AnimatedVisibility',
  ])
  assert.deepEqual(meshes.map((mesh) => mesh.material.color.toArray()), [
    [0.855, 0, 0],
    [0.855, 0, 0],
    [0.855, 0, 0],
    [0, 0.855, 0],
    [0, 0, 0.855],
  ])
  assert.ok(meshes.every((mesh) => mesh.geometry.getAttribute('position')?.count === 24))
  assert.ok(meshes.every((mesh) => mesh.geometry.index?.count === 36))

  assert.equal(meshes[0].visible, false, 'InvisibleCube should import KHR_node_visibility false')
  assert.equal(meshes[1].visible, true, 'child mesh should keep its own default visible flag')
  assert.equal(meshes[2].visible, true, 'descendant mesh should keep its own default visible flag')
  assert.equal(meshes[3].visible, true)
  assert.equal(meshes[4].visible, true)
  assert.equal(isEffectivelyVisible(meshes[0]), false, 'InvisibleCube should be effectively hidden')
  assert.equal(isEffectivelyVisible(meshes[1]), false, 'child mesh should be hidden by its invisible parent')
  assert.equal(isEffectivelyVisible(meshes[2]), false, 'descendant mesh should be hidden by its invisible ancestor')
  assert.equal(isEffectivelyVisible(meshes[3]), true)
  assert.equal(isEffectivelyVisible(meshes[4]), true)

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1.2))
  const camera = new THREE.OrthographicCamera(-2.4, 2.4, 1.2, -1.2, 0.01, 10)
  camera.position.set(0, 0, 4)
  camera.lookAt(0, 0, 0)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 160,
    height: 80,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.15, 'CubeVisibility should render the visible green and blue cubes')
  const left = meanRegion(rgba, 160, 80, 0, 25, 45, 55)
  const center = meanRegion(rgba, 160, 80, 55, 25, 105, 55)
  const right = meanRegion(rgba, 160, 80, 115, 25, 160, 55)
  assert.ok(left.r < 5 && left.g < 5 && left.b < 5, `invisible red branch should not render (${left.r}, ${left.g}, ${left.b})`)
  assert.ok(center.g > center.r + 50 && center.g > center.b + 50, `visible green cube should render in the center (${center.r}, ${center.g}, ${center.b})`)
  assert.ok(right.b > right.r + 50 && right.b > right.g + 50, `visible blue cube should render on the right (${right.r}, ${right.g}, ${right.b})`)
})

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

test('committed Khronos glTF Sample Assets ClearCoatTest fixture loads KHR_materials_clearcoat maps', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_CLEARCOAT_TEST)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 27)

  const materials = new Map(meshes.map((mesh) => [mesh.material.name, mesh.material]))
  const simpleCoated = materials.get('Simple_Coated')
  assert.equal(simpleCoated?.isMeshPhysicalMaterial, true)
  assert.equal(simpleCoated.clearcoat, 1)
  assert.equal(simpleCoated.clearcoatRoughness, 0.03)

  const partialCoated = materials.get('Partial_Coated')
  assert.equal(partialCoated?.isMeshPhysicalMaterial, true)
  assert.equal(Buffer.isBuffer(partialCoated.clearcoatMap?.image), true, 'clearcoat factor PNG should load as an encoded Buffer')
  assert.equal(partialCoated.clearcoatMap.colorSpace, THREE.NoColorSpace)

  const roughCoated = materials.get('RoughVariations_Coated')
  assert.equal(roughCoated?.isMeshPhysicalMaterial, true)
  assert.equal(roughCoated.clearcoatRoughness, 1)
  assert.equal(Buffer.isBuffer(roughCoated.clearcoatRoughnessMap?.image), true, 'clearcoat roughness PNG should load as an encoded Buffer')
  assert.equal(roughCoated.clearcoatRoughnessMap.colorSpace, THREE.NoColorSpace)

  const coatNormal = materials.get('CoatNorm_Coated')
  assert.equal(coatNormal?.isMeshPhysicalMaterial, true)
  assert.equal(Buffer.isBuffer(coatNormal.clearcoatNormalMap?.image), true, 'clearcoat normal PNG should load as an encoded Buffer')
  assert.equal(coatNormal.clearcoatNormalMap.colorSpace, THREE.NoColorSpace)

  const sharedNormal = materials.get('SharedNorm_Coated')
  assert.equal(sharedNormal?.isMeshPhysicalMaterial, true)
  assert.equal(Buffer.isBuffer(sharedNormal.clearcoatNormalMap?.image), true, 'shared clearcoat normal JPEG should load as an encoded Buffer')
  assert.equal(sharedNormal.clearcoatNormalMap.colorSpace, THREE.NoColorSpace)

  const camera = new THREE.PerspectiveCamera(35, 4 / 3, 0.01, 40)
  camera.position.set(0, 1.2, 12)
  camera.lookAt(0, 0.6, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.9))
  const light = new THREE.DirectionalLight(0xffffff, 1.4)
  light.position.set(2, 4, 6)
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 160,
    height: 120,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.05, 'ClearCoatTest should render visible clearcoat panels')
})

test('committed Khronos glTF Sample Assets IridescenceLamp fixture loads physical iridescence inputs', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_IRIDESCENCE_LAMP)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 3)

  const materials = new Map(meshes.map((mesh) => [mesh.material.name, mesh.material]))
  const base = materials.get('IridescenceLamp')
  assert.equal(base?.isMeshStandardMaterial, true)
  assert.equal(Buffer.isBuffer(base.map?.image), true, 'base color PNG should load as an encoded Buffer')
  assert.equal(Buffer.isBuffer(base.roughnessMap?.image), true, 'ORM roughness PNG should load as an encoded Buffer')
  assert.equal(Buffer.isBuffer(base.metalnessMap?.image), true, 'ORM metalness PNG should load as an encoded Buffer')
  assert.equal(Buffer.isBuffer(base.aoMap?.image), true, 'ORM occlusion PNG should load as an encoded Buffer')
  assert.equal(base.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(base.roughnessMap.colorSpace, THREE.NoColorSpace)

  const transmitted = materials.get('IridescenceLampTransmissionIridescence')
  assert.equal(transmitted?.isMeshPhysicalMaterial, true)
  assert.equal(transmitted.transmission, 1)
  assert.equal(transmitted.thickness, 0.005)
  assert.equal(transmitted.ior, 1.6)
  assert.equal(transmitted.iridescence, 1)
  assert.equal(transmitted.iridescenceIOR, 2)
  assert.deepEqual(transmitted.iridescenceThicknessRange, [385, 405])
  assert.equal(Buffer.isBuffer(transmitted.iridescenceThicknessMap?.image), true, 'iridescence thickness PNG should load as an encoded Buffer')
  assert.equal(transmitted.iridescenceThicknessMap.colorSpace, THREE.NoColorSpace)

  const iridescent = materials.get('IridescenceLampIridescence')
  assert.equal(iridescent?.isMeshPhysicalMaterial, true)
  assert.equal(iridescent.transmission, 0)
  assert.equal(iridescent.ior, 1.5)
  assert.equal(iridescent.iridescence, 1)
  assert.equal(iridescent.iridescenceIOR, 1.8)
  assert.deepEqual(iridescent.iridescenceThicknessRange, [485, 515])
  assert.equal(Buffer.isBuffer(iridescent.iridescenceThicknessMap?.image), true, 'second iridescence thickness PNG should load as an encoded Buffer')

  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 20)
  camera.position.set(0, 0.7, 2.4)
  camera.lookAt(0, 0.45, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1.0))
  const light = new THREE.DirectionalLight(0xffffff, 1.5)
  light.position.set(2, 4, 5)
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.035, 'IridescenceLamp should render visible physical-material geometry')
})

test('committed Khronos glTF Sample Assets CompareIor fixture loads transmission, volume, and IOR inputs', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_COMPARE_IOR, 'utf8'))
  assert.deepEqual(source.extensionsUsed, [
    'KHR_materials_transmission',
    'KHR_materials_volume',
    'KHR_materials_ior',
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_COMPARE_IOR)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 3, 'CompareIor should load two spheres plus checker backdrop')

  const meshesByName = new Map(meshes.map((mesh) => [mesh.name, mesh]))
  const baseline = meshesByName.get('GeoSphere001')
  const iorSphere = meshesByName.get('GeoSphere002')
  const checker = meshesByName.get('Checker')
  assert.equal(baseline?.geometry.getAttribute('position')?.count, 673)
  assert.equal(baseline.geometry.getAttribute('normal')?.count, 673)
  assert.equal(baseline.geometry.getAttribute('uv')?.count, 673)
  assert.equal(baseline.geometry.index?.count, 3840)
  assert.equal(iorSphere?.geometry.getAttribute('position')?.count, 673)
  assert.equal(iorSphere.geometry.index?.count, 3840)
  assert.equal(checker?.geometry.getAttribute('position')?.count, 4)
  assert.equal(checker.geometry.index?.count, 6)

  assert.equal(baseline.material.isMeshPhysicalMaterial, true)
  assert.equal(baseline.material.name, 'glTF Logo Transmission')
  assert.equal(baseline.material.transmission, 1)
  assert.equal(baseline.material.ior, 1.5)
  assert.equal(baseline.material.thickness, 0)
  assert.equal(baseline.material.roughness, 0.69999)
  assert.equal(iorSphere.material.isMeshPhysicalMaterial, true)
  assert.equal(iorSphere.material.name, 'glTF Logo Transmission IOR')
  assert.equal(iorSphere.material.transmission, 1)
  assert.equal(iorSphere.material.ior, 2.42)
  assert.equal(iorSphere.material.thickness, 1)
  assert.equal(iorSphere.material.attenuationDistance, 1)

  for (const material of [baseline.material, iorSphere.material]) {
    assert.equal(Buffer.isBuffer(material.map?.image), true, `${material.name} base color JPG should load as an encoded Buffer`)
    assert.equal(Buffer.isBuffer(material.roughnessMap?.image), true, `${material.name} roughness JPG should load as an encoded Buffer`)
    assert.equal(Buffer.isBuffer(material.metalnessMap?.image), true, `${material.name} metalness JPG should load as an encoded Buffer`)
    assert.equal(Buffer.isBuffer(material.transmissionMap?.image), true, `${material.name} transmission JPG should load as an encoded Buffer`)
    assert.equal(material.map.name, 'Compare_Ior_img1.jpg')
    assert.equal(material.roughnessMap.name, 'Compare_Ior_img2.jpg')
    assert.equal(material.metalnessMap.name, 'Compare_Ior_img2.jpg')
    assert.equal(material.transmissionMap.name, 'Compare_Ior_img3.jpg')
    assert.equal(material.map.colorSpace, THREE.SRGBColorSpace)
    assert.equal(material.roughnessMap.colorSpace, THREE.NoColorSpace)
    assert.equal(material.metalnessMap.colorSpace, THREE.NoColorSpace)
    assert.equal(material.transmissionMap.colorSpace, THREE.NoColorSpace)
    assert.equal(material.map.flipY, false)
    assert.equal(material.transmissionMap.flipY, false)
  }
  assert.equal(Buffer.isBuffer(checker.material.map?.image), true, 'CompareIor checker JPG should load as an encoded Buffer')
  assert.equal(checker.material.map.name, 'Compare_Ior_img0.jpg')

  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 20)
  camera.position.set(0, 0.1, 4)
  camera.lookAt(0, 0, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 1.5)
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.4, 'CompareIor should render visible physical material spheres')
  const left = meanRegion(rgba, 128, 128, 20, 48, 52, 82)
  const right = meanRegion(rgba, 128, 128, 76, 48, 108, 82)
  assert.ok(left.g > left.b + 15 && left.r > left.b + 5, `baseline transmission sphere should render lit textured pixels (${left.r}, ${left.g}, ${left.b})`)
  assert.ok(right.g > right.b + 15 && right.r > right.b + 5, `IOR transmission sphere should render lit textured pixels (${right.r}, ${right.g}, ${right.b})`)
})

test('committed Khronos glTF Sample Assets TransmissionRoughnessTest fixture loads IOR and roughness texture inputs', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_TRANSMISSION_ROUGHNESS_TEST, 'utf8'))
  assert.deepEqual(source.extensionsUsed, [
    'KHR_materials_transmission',
    'KHR_materials_ior',
    'KHR_materials_volume',
  ])
  assert.equal(source.buffers[0].uri, 'TransmissionRoughnessTest.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'IOR_Labels.png',
    'RoughnessGrid.png',
    'RoughnessGrid-1.png',
    'GridWithDetails.png',
    'SmoothVsRough.png',
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_TRANSMISSION_ROUGHNESS_TEST)
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_transmission'))
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_ior'))
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_volume'))

  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'Labels',
    'IOR_10',
    'IOR_133',
    'IOR_150',
    'IOR_176',
    'IOR_242',
    'Opaque',
    'Flat_Backdrop',
    'SmoothRoughLabels',
  ])

  const byName = new Map(meshes.map((mesh) => [mesh.name, mesh]))
  const samples = ['IOR_10', 'IOR_133', 'IOR_150', 'IOR_176', 'IOR_242'].map((name) => byName.get(name))
  assert.deepEqual(samples.map((mesh) => mesh.geometry.getAttribute('position')?.count), [7866, 7866, 7866, 7866, 7866])
  assert.deepEqual(samples.map((mesh) => mesh.geometry.index?.count), [38880, 38880, 38880, 38880, 38880])

  const sampleMaterials = samples.map((mesh) => mesh.material)
  assert.ok(sampleMaterials.every((material) => material.isMeshPhysicalMaterial === true), 'IOR samples should load as MeshPhysicalMaterial')
  assert.deepEqual(sampleMaterials.map((material) => material.name), [
    'Mat_IOR_1.0',
    'Mat_IOR_1.33',
    'Mat_IOR_1.50',
    'Mat_IOR_1.76',
    'Mat_IOR_2.42',
  ])
  assert.deepEqual(sampleMaterials.map((material) => material.transmission), [1, 1, 1, 1, 1])
  assert.deepEqual(sampleMaterials.map((material) => material.ior), [1, 1.33, 1.5, 1.76, 2.42])
  assert.ok(sampleMaterials.every((material) => material.thickness === 0.005))
  assert.ok(sampleMaterials.every((material) => material.roughnessMap === material.metalnessMap))
  assert.equal(new Set(sampleMaterials.map((material) => material.roughnessMap)).size, 1, 'IOR samples should share the same roughness texture object')
  assert.equal(sampleMaterials[0].roughnessMap.name, 'RoughnessGrid')
  assert.equal(sampleMaterials[0].roughnessMap.colorSpace, THREE.NoColorSpace)
  assert.deepEqual(pngDimensions(sampleMaterials[0].roughnessMap.image), [64, 64])

  const labels = byName.get('Labels')
  assert.equal(labels.material.name, 'LabelMat')
  assert.equal(labels.material.map.name, 'IOR_Labels')
  assert.equal(labels.material.map.colorSpace, THREE.SRGBColorSpace)
  assert.deepEqual(pngDimensions(labels.material.map.image), [512, 512])

  const opaque = byName.get('Opaque')
  assert.equal(opaque.material.name, 'Mat_Opaque')
  assert.equal(opaque.material.isMeshStandardMaterial, true)
  assert.equal(opaque.material.roughnessMap.name, 'RoughnessGrid')
  assert.notEqual(opaque.material.roughnessMap, sampleMaterials[0].roughnessMap)
  assert.deepEqual(pngDimensions(opaque.material.roughnessMap.image), [64, 64])

  const backdrop = byName.get('Flat_Backdrop')
  assert.equal(backdrop.material.name, 'FlatBackdrop')
  assert.equal(backdrop.material.map.name, 'GridWithDetails')
  assert.deepEqual(pngDimensions(backdrop.material.map.image), [256, 256])

  const smoothRoughLabels = byName.get('SmoothRoughLabels')
  assert.equal(smoothRoughLabels.material.side, THREE.DoubleSide)
  assert.equal(smoothRoughLabels.material.map.name, 'SmoothVsRough')
  assert.deepEqual(pngDimensions(smoothRoughLabels.material.map.image), [256, 256])

  const camera = new THREE.OrthographicCamera(-1.1, 1.1, 0.65, -0.65, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1))
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 160,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.55, 'TransmissionRoughnessTest should render visible roughness and IOR samples')
  const center = meanRegion(rgba, 160, 96, 64, 24, 96, 72)
  assert.ok(center.r > 110 && center.g > 110 && center.b > 110, `TransmissionRoughnessTest center samples should render visible panels (${center.r}, ${center.g}, ${center.b})`)
})

test('committed Khronos glTF Sample Assets SheenChair fixture loads KHR_materials_sheen and variants metadata', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_SHEEN_CHAIR)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 4)
  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'SheenChair_fabric',
    'SheenChair_wood',
    'SheenChair_metal',
    'SheenChair_label',
  ])

  const variants = gltf.parser?.json?.extensions?.KHR_materials_variants?.variants
  assert.deepEqual(variants?.map((variant) => variant.name), ['Mango Velvet', 'Peacock Velvet'])
  const fabric = meshes.find((mesh) => mesh.name === 'SheenChair_fabric')
  assert.ok(fabric.userData.gltfExtensions?.KHR_materials_variants?.mappings?.length >= 2, 'fabric mesh should preserve material variant mappings')
  assert.equal(fabric.geometry.getAttribute('position')?.count, 14350)
  assert.equal(fabric.geometry.getAttribute('uv')?.count, 14350)

  const fabricMaterial = fabric.material
  assert.equal(fabricMaterial.isMeshPhysicalMaterial, true)
  assert.equal(fabricMaterial.sheen, 1)
  assert.deepEqual(fabricMaterial.sheenColor.toArray(), [1, 0.329, 0.1])
  assert.equal(fabricMaterial.sheenRoughness, 0.8)
  assert.equal(Buffer.isBuffer(fabricMaterial.map?.image), true, 'fabric base color PNG should load as an encoded Buffer')
  assert.equal(Buffer.isBuffer(fabricMaterial.normalMap?.image), true, 'fabric normal PNG should load as an encoded Buffer')
  assert.equal(Buffer.isBuffer(fabricMaterial.aoMap?.image), true, 'fabric occlusion PNG should load as an encoded Buffer')
  assert.equal(fabricMaterial.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(fabricMaterial.normalMap.colorSpace, THREE.NoColorSpace)
  assert.equal(fabricMaterial.aoMap.colorSpace, THREE.NoColorSpace)
  assert.equal(fabricMaterial.aoMap.channel, 1)
  assert.deepEqual(fabricMaterial.map.offset.toArray(), [-3, 3])
  assert.deepEqual(fabricMaterial.map.repeat.toArray(), [7, 7])

  const woodMaterial = meshes.find((mesh) => mesh.name === 'SheenChair_wood').material
  assert.equal(woodMaterial.isMeshStandardMaterial, true)
  assert.equal(Buffer.isBuffer(woodMaterial.map?.image), true, 'wood base color PNG should load as an encoded Buffer')
  assert.equal(Buffer.isBuffer(woodMaterial.roughnessMap?.image), true, 'wood roughness PNG should load as an encoded Buffer')
  assert.equal(Buffer.isBuffer(woodMaterial.metalnessMap?.image), true, 'wood metalness PNG should load as an encoded Buffer')
  assert.equal(woodMaterial.roughnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(woodMaterial.metalnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(woodMaterial.aoMap.channel, 1)

  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 20)
  camera.position.set(0.6, 0.8, 2.2)
  camera.lookAt(0, 0.35, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 1.3)
  light.position.set(1.5, 3, 4)
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.1, 'SheenChair should render visible sheen material geometry')
})

test('committed Khronos glTF Sample Assets SpecularTest fixture loads KHR_materials_specular scalar and texture inputs', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_SPECULAR_TEST)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 24)
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_specular'))

  const materials = new Map(meshes.map((mesh) => [mesh.material.name, mesh.material]))
  assert.equal(materials.size, 24)

  const disabled = materials.get('M1.1_specFac')
  const enabled = materials.get('M1.5_specFac')
  assert.equal(disabled?.isMeshPhysicalMaterial, true)
  assert.equal(enabled?.isMeshPhysicalMaterial, true)
  assert.equal(disabled.specularIntensity, 0)
  assert.equal(enabled.specularIntensity, 1)
  assert.deepEqual(enabled.specularColor.toArray(), [1, 1, 1])

  const specularTexture = materials.get('M2_SpecTex')
  assert.equal(specularTexture?.isMeshPhysicalMaterial, true)
  assert.equal(specularTexture.specularIntensity, 1)
  assert.equal(Buffer.isBuffer(specularTexture.specularIntensityMap?.image), true, 'specular factor PNG should load as an encoded Buffer')
  assert.equal(specularTexture.specularIntensityMap.name, 'specularTextureGrid')
  assert.equal(specularTexture.specularIntensityMap.colorSpace, THREE.NoColorSpace)
  assert.equal(specularTexture.specularIntensityMap.magFilter, THREE.LinearFilter)
  assert.equal(specularTexture.specularIntensityMap.minFilter, THREE.LinearMipmapLinearFilter)
  assert.equal(specularTexture.specularIntensityMap.flipY, false)

  const whiteTexture = materials.get('M4_whiteTex')
  assert.equal(whiteTexture?.isMeshPhysicalMaterial, true)
  assert.equal(Buffer.isBuffer(whiteTexture.specularColorMap?.image), true, 'white specular color PNG should load as an encoded Buffer')
  assert.equal(whiteTexture.specularColorMap.name, 'WhiteGrid')
  assert.equal(whiteTexture.specularColorMap.colorSpace, THREE.SRGBColorSpace)
  assert.equal(whiteTexture.specularColorMap.flipY, false)

  const yellowTexture = materials.get('M6_yellowTex')
  assert.equal(yellowTexture?.isMeshPhysicalMaterial, true)
  assert.equal(Buffer.isBuffer(yellowTexture.specularColorMap?.image), true, 'yellow specular color PNG should load as an encoded Buffer')
  assert.equal(yellowTexture.specularColorMap.name, 'YellowGrid')
  assert.equal(yellowTexture.specularColorMap.colorSpace, THREE.SRGBColorSpace)

  const hdrFactor = materials.get('M7.5_HDR')
  assert.equal(hdrFactor?.isMeshPhysicalMaterial, true)
  assert.deepEqual(hdrFactor.specularColor.toArray(), [25, 25, 25])

  const specularTextureMesh = meshes.find((mesh) => mesh.material.name === 'M2_SpecTex')
  assert.equal(specularTextureMesh.geometry.getAttribute('position')?.count, 3645)
  assert.equal(specularTextureMesh.geometry.getAttribute('uv')?.count, 3645)
  assert.equal(specularTextureMesh.geometry.index?.count, 19200)

  const scalarMesh = meshes.find((mesh) => mesh.material.name === 'M1.5_specFac')
  assert.equal(scalarMesh.geometry.getAttribute('position')?.count, 642)
  assert.equal(scalarMesh.geometry.index?.count, 3840)

  const camera = new THREE.OrthographicCamera(-0.7, 0.7, 0.52, -0.52, 0.01, 20)
  camera.position.set(0, 0, 4)
  camera.lookAt(0, 0, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.25))
  const light = new THREE.DirectionalLight(0xffffff, 4)
  light.position.set(0.2, 0.5, 4)
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 160,
    height: 120,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.18, 'SpecularTest should render visible specular material samples')
})

test('committed Khronos glTF Sample Assets Suzanne fixture loads dense textured PBR mesh', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_SUZANNE, 'utf8'))
  assert.deepEqual(source.buffers, [{ byteLength: 590400, uri: 'Suzanne.bin' }])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'Suzanne_BaseColor.png',
    'Suzanne_MetallicRoughness.png',
  ])
  assert.equal(source.meshes[0].name, 'Suzanne')
  assert.equal(source.materials[0].name, 'Suzanne')
  assert.equal(source.materials[0].pbrMetallicRoughness.baseColorTexture.index, 0)
  assert.equal(source.materials[0].pbrMetallicRoughness.metallicRoughnessTexture.index, 1)

  const gltf = await loadGltfFixture(SAMPLE_ASSET_SUZANNE)
  const mesh = gltf.scene.getObjectByName('Suzanne')
  assert.ok(mesh?.isMesh, 'Suzanne sample should load a named mesh')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 11808)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 11808)
  assert.equal(mesh.geometry.getAttribute('tangent')?.count, 11808)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 11808)
  assert.equal(mesh.geometry.index?.count, 11808)
  assert.equal(mesh.material.name, 'Suzanne')
  assert.equal(mesh.material.metalness, 1)
  assert.equal(mesh.material.roughness, 1)

  const { map, roughnessMap, metalnessMap } = mesh.material
  assert.ok(map?.isTexture, 'Suzanne sample should load a base-color texture')
  assert.ok(roughnessMap?.isTexture, 'Suzanne sample should load a roughness texture')
  assert.ok(metalnessMap?.isTexture, 'Suzanne sample should load a metalness texture')
  assert.equal(roughnessMap, metalnessMap, 'Suzanne metallic/roughness channels should share the packed texture')
  assert.equal(map.name, 'Suzanne_BaseColor.png')
  assert.equal(roughnessMap.name, 'Suzanne_MetallicRoughness.png')
  assert.deepEqual(pngDimensions(map.image), [1024, 1024])
  assert.deepEqual(pngDimensions(roughnessMap.image), [1024, 1024])
  assert.equal(map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(roughnessMap.colorSpace, THREE.NoColorSpace)

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 1.8)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)
  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const halfExtent = Math.max(size.x, size.y, size.z) / 2 + 0.1
  const camera = new THREE.OrthographicCamera(-halfExtent, halfExtent, halfExtent, -halfExtent, 0.01, 20)
  camera.position.set(center.x + 2, center.y + 1.5, center.z + 2.5)
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
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.3, 'Suzanne should render visible dense textured PBR geometry')
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

test('committed Khronos glTF Sample Assets PrimitiveModeNormalsTest fixture loads primitive modes with normals and colors', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_PRIMITIVE_MODE_NORMALS_TEST, 'utf8'))
  assert.deepEqual(source.buffers.map((buffer) => buffer.uri), [
    'Points.bin',
    'Lines.bin',
    'Triangles.bin',
    'Colors.bin',
    'Plane.bin',
  ])
  assert.deepEqual(source.buffers.map((buffer) => buffer.byteLength), [786432, 786432, 4380, 262144, 92])
  assert.deepEqual(source.images.map((image) => image.uri), ['Labels.png'])
  assert.equal(source.meshes.length, 25)
  assert.deepEqual(source.meshes.slice(0, 6).map((mesh) => ({
    mode: mesh.primitives[0].mode,
    attributes: Object.keys(mesh.primitives[0].attributes),
  })), [
    { mode: 0, attributes: ['POSITION'] },
    { mode: 3, attributes: ['POSITION'] },
    { mode: 4, attributes: ['POSITION'] },
    { mode: 0, attributes: ['POSITION', 'COLOR_0'] },
    { mode: 3, attributes: ['POSITION', 'COLOR_0'] },
    { mode: 4, attributes: ['POSITION', 'COLOR_0'] },
  ])
  assert.deepEqual(source.meshes.slice(12, 18).map((mesh) => Object.keys(mesh.primitives[0].attributes)), [
    ['POSITION', 'NORMAL'],
    ['POSITION', 'NORMAL'],
    ['POSITION', 'NORMAL'],
    ['POSITION', 'COLOR_0', 'NORMAL'],
    ['POSITION', 'COLOR_0', 'NORMAL'],
    ['POSITION', 'COLOR_0', 'NORMAL'],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_PRIMITIVE_MODE_NORMALS_TEST)
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

  assert.equal(renderables.length, 25)
  assert.deepEqual(renderables.reduce((counts, object) => {
    counts[object.type] = (counts[object.type] ?? 0) + 1
    return counts
  }, {}), { Points: 8, Line: 8, Mesh: 9 })

  const points = renderables[0]
  const coloredPoints = renderables[3]
  const normalMesh = renderables[14]
  const coloredNormalMesh = renderables[17]
  const labelPlane = renderables[24]
  assert.equal(points.geometry.getAttribute('position')?.count, 65536)
  assert.equal(coloredPoints.geometry.getAttribute('color')?.count, 65536)
  assert.equal(coloredPoints.geometry.getAttribute('color')?.itemSize, 4)
  assert.equal(coloredPoints.geometry.getAttribute('color')?.normalized, true)
  assert.equal(normalMesh.geometry.getAttribute('normal')?.count, 205)
  assert.equal(coloredNormalMesh.geometry.getAttribute('normal')?.count, 205)
  assert.equal(coloredNormalMesh.geometry.getAttribute('color')?.normalized, true)
  assert.equal(labelPlane.material.map?.name, 'Labels.png')
  assert.deepEqual(pngDimensions(labelPlane.material.map.image), [1024, 1024])
  assert.equal(labelPlane.material.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(labelPlane.material.map.flipY, false)

  for (const object of renderables) {
    if (object.material?.color) object.material.color.set(0xffffff)
    if (object.isPoints === true) object.material.size = 2.5
    if (object.isLine === true || object.isLineSegments === true || object.isLineLoop === true) {
      object.material.linewidth = 3
    }
  }

  const camera = new THREE.OrthographicCamera(-7, 11, 8, -8, 0.01, 30)
  camera.position.set(0, 0, 12)
  camera.lookAt(0, 0, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1.0))
  const light = new THREE.DirectionalLight(0xffffff, 1.2)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 180,
    height: 180,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.45, 'PrimitiveModeNormalsTest should render visible primitive-mode grids')
  const topLeft = meanRegion(rgba, 180, 180, 20, 20, 50, 50)
  const center = meanRegion(rgba, 180, 180, 75, 75, 105, 105)
  assert.ok(topLeft.r > 150 && topLeft.g > 150 && topLeft.b > 150, `upper primitive samples should render bright points/lines (${topLeft.r}, ${topLeft.g}, ${topLeft.b})`)
  assert.ok(center.r > 50 && center.g > 45 && center.b < 20, `center primitive sample should include normalized color attributes (${center.r}, ${center.g}, ${center.b})`)
})

test('committed Khronos glTF Sample Assets MorphPrimitivesTest fixture preserves morph targets across split primitives', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_MORPH_PRIMITIVES_TEST, 'utf8'))
  assert.equal(source.buffers[0].uri, 'MorphPrimitivesTest.bin')
  assert.deepEqual(source.images.map((image) => image.uri), ['uv_texture.jpg'])
  assert.deepEqual(source.meshes[0].weights, [0.5])
  assert.deepEqual(source.meshes[0].primitives.map((primitive) => ({
    mode: primitive.mode,
    material: primitive.material,
    targetAttributes: Object.keys(primitive.targets[0]),
  })), [
    { mode: 4, material: 0, targetAttributes: ['POSITION'] },
    { mode: 4, material: 1, targetAttributes: ['POSITION'] },
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_MORPH_PRIMITIVES_TEST)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })

  assert.deepEqual(meshes.map((mesh) => ({
    name: mesh.name,
    material: mesh.material.name,
    positions: mesh.geometry.getAttribute('position')?.count,
    normals: mesh.geometry.getAttribute('normal')?.count,
    uvs: mesh.geometry.getAttribute('uv')?.count,
    index: mesh.geometry.index?.count,
    morphPositions: mesh.geometry.morphAttributes.position?.map((attribute) => attribute.count),
    influences: mesh.morphTargetInfluences,
    morphTargetsRelative: mesh.geometry.morphTargetsRelative,
  })), [
    {
      name: 'mesh_1',
      material: 'red',
      positions: 21,
      normals: 21,
      uvs: 21,
      index: 72,
      morphPositions: [21],
      influences: [0.5],
      morphTargetsRelative: true,
    },
    {
      name: 'mesh_2',
      material: 'green',
      positions: 9,
      normals: 9,
      uvs: 9,
      index: 24,
      morphPositions: [9],
      influences: [0.5],
      morphTargetsRelative: true,
    },
  ])

  assertVectorClose(meshes[0].material.color.toArray(), [1, 0, 0], 'MorphPrimitivesTest red material')
  assertVectorClose(meshes[1].material.color.toArray(), [0, 1, 0], 'MorphPrimitivesTest green material')
  for (const mesh of meshes) {
    assert.equal(mesh.material.isMeshStandardMaterial, true)
    assert.equal(mesh.material.map?.name, 'uv_texture.jpg')
    assert.equal(Buffer.isBuffer(mesh.material.map.image), true, `${mesh.name} should load the external JPEG as an encoded Buffer`)
    assert.equal(mesh.material.map.colorSpace, THREE.SRGBColorSpace)
    assert.equal(mesh.material.map.flipY, false)

    const position = mesh.geometry.getAttribute('position')
    const morphPosition = mesh.geometry.morphAttributes.position[0]
    let morphedMaxY = -Infinity
    for (let i = 0; i < position.count; i += 1) {
      morphedMaxY = Math.max(morphedMaxY, position.getY(i) + morphPosition.getY(i))
    }
    assert.ok(Math.abs(morphedMaxY - 0.20000000298023224) < 1e-8, `${mesh.name} should preserve its upward morph target`)
  }

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 20)
  camera.position.set(1.8, 1.4, 3.2)
  camera.lookAt(0, 0, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1.2))
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.03, 'MorphPrimitivesTest should render visible morphed primitive meshes')
  const redRegion = meanRegion(rgba, 128, 128, 52, 52, 76, 76)
  const greenRegion = meanRegion(rgba, 128, 128, 76, 52, 100, 76)
  assert.ok(redRegion.r > redRegion.g + 70 && redRegion.r > redRegion.b + 90, `MorphPrimitivesTest should render the red primitive (${redRegion.r}, ${redRegion.g}, ${redRegion.b})`)
  assert.ok(greenRegion.g > greenRegion.r && greenRegion.g > greenRegion.b + 10, `MorphPrimitivesTest should render the green primitive (${greenRegion.r}, ${greenRegion.g}, ${greenRegion.b})`)
})

test('committed Khronos glTF Sample Assets NegativeScaleTest fixture preserves negative node determinants', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_NEGATIVE_SCALE_TEST)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })

  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'NegativeScaleBack',
    'BackgroundMesh',
    'Labels',
    'PositiveScaleTest',
    'NegativeScaleFront',
    'NotShiny1',
    'NotShinyMinus1',
    'Shiny1',
    'ShinyMinus1',
    'Dark1',
    'DarkMinus1',
  ])

  const positivePanel = gltf.scene.getObjectByName('PositiveScaleTest')
  const negativeFrontPanel = gltf.scene.getObjectByName('NegativeScaleFront')
  const labelPanel = gltf.scene.getObjectByName('Labels')
  const notShinyMinusOne = gltf.scene.getObjectByName('NotShinyMinus1')
  const shinyOne = gltf.scene.getObjectByName('Shiny1')
  const shinyMinusOne = gltf.scene.getObjectByName('ShinyMinus1')
  assert.ok(positivePanel?.isMesh, 'NegativeScaleTest should load the positive front-face panel')
  assert.ok(negativeFrontPanel?.isMesh, 'NegativeScaleTest should load the negative-scale front-face panel')
  assert.ok(labelPanel?.isMesh, 'NegativeScaleTest should load the external PNG label panel')
  assert.ok(notShinyMinusOne?.isMesh, 'NegativeScaleTest should load the negative-scale double-sided sphere')
  assert.ok(shinyOne?.isMesh, 'NegativeScaleTest should load a child under a negative-scale parent')
  assert.ok(shinyMinusOne?.isMesh, 'NegativeScaleTest should load a negative-scale child under a negative-scale parent')

  assert.equal(positivePanel.material.side, THREE.FrontSide)
  assert.equal(negativeFrontPanel.material.side, THREE.FrontSide)
  assert.equal(notShinyMinusOne.material.side, THREE.DoubleSide)
  assert.equal(Buffer.isBuffer(negativeFrontPanel.material.map?.image), true, 'NegativeScaleTest check/X PNG should load as an encoded Buffer')
  assert.equal(Buffer.isBuffer(labelPanel.material.map?.image), true, 'NegativeScaleTest label PNG should load as an encoded Buffer')

  assert.ok(worldDeterminant(positivePanel) > 0, 'positive-scale panel should keep positive world winding')
  assert.ok(worldDeterminant(negativeFrontPanel) < 0, 'negative-scale panel should expose negative world winding')
  assert.ok(worldDeterminant(shinyOne) < 0, 'child under a negative-scale parent should inherit negative world winding')
  assert.ok(worldDeterminant(shinyMinusOne) > 0, 'negative-scale child under a negative-scale parent should recover positive world winding')

  const renderer = new Renderer()
  assert.ok(
    renderSingleObjectRatio(renderer, positivePanel) > 0.3,
    'NegativeScaleTest positive-scale front-face panel should render visible pixels',
  )
  assert.ok(
    renderSingleObjectRatio(renderer, negativeFrontPanel) > 0.15,
    'NegativeScaleTest negative-scale front-face panel should render visible pixels',
  )
})

test('committed Khronos glTF Sample Assets OrientationTest fixture preserves quaternion and matrix rotations', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_ORIENTATION_TEST)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 13)

  const arrowX1 = gltf.scene.getObjectByName('ArrowX1')
  const arrowY1 = gltf.scene.getObjectByName('ArrowY1')
  const arrowZ1 = gltf.scene.getObjectByName('ArrowZ1')
  const arrowX2 = gltf.scene.getObjectByName('ArrowX2')
  const arrowY2 = gltf.scene.getObjectByName('ArrowY2')
  const arrowZ2 = gltf.scene.getObjectByName('ArrowZ2')
  for (const arrow of [arrowX1, arrowY1, arrowZ1, arrowX2, arrowY2, arrowZ2]) {
    assert.ok(arrow?.isMesh, 'OrientationTest should load all quaternion and matrix arrow meshes')
  }

  assertVectorClose(arrowX1.position.toArray(), [5, 0, 0], 'ArrowX1 quaternion translation')
  assertVectorClose(arrowY1.position.toArray(), [0, 5, 0], 'ArrowY1 quaternion translation')
  assertVectorClose(arrowZ1.position.toArray(), [0, 0, 5], 'ArrowZ1 quaternion translation')
  assertVectorClose(arrowX2.position.toArray(), [-5, 0, 0], 'ArrowX2 matrix translation')
  assertVectorClose(arrowY2.position.toArray(), [0, -5, 0], 'ArrowY2 matrix translation')
  assertVectorClose(arrowZ2.position.toArray(), [0, 0, -5], 'ArrowZ2 matrix translation')

  assert.ok(arrowX1.quaternion.x < -0.29 && Math.abs(arrowX1.quaternion.y) < 1e-6 && Math.abs(arrowX1.quaternion.z) < 1e-6, 'ArrowX1 should keep its X-axis quaternion rotation')
  assert.ok(arrowY1.quaternion.y < -0.57 && Math.abs(arrowY1.quaternion.x) < 1e-6 && Math.abs(arrowY1.quaternion.z) < 1e-6, 'ArrowY1 should keep its Y-axis quaternion rotation')
  assert.ok(arrowZ1.quaternion.z > 0.13 && Math.abs(arrowZ1.quaternion.x) < 1e-6 && Math.abs(arrowZ1.quaternion.y) < 1e-6, 'ArrowZ1 should keep its Z-axis quaternion rotation')
  assert.ok(arrowX2.quaternion.x > 0.04 && Math.abs(arrowX2.quaternion.y) < 1e-6 && Math.abs(arrowX2.quaternion.z) < 1e-6, 'ArrowX2 should decompose its matrix into an X-axis rotation')
  assert.ok(arrowY2.quaternion.y < -0.10 && Math.abs(arrowY2.quaternion.x) < 1e-6 && Math.abs(arrowY2.quaternion.z) < 1e-6, 'ArrowY2 should decompose its matrix into a Y-axis rotation')
  assert.ok(arrowZ2.quaternion.z < -0.14 && Math.abs(arrowZ2.quaternion.x) < 1e-6 && Math.abs(arrowZ2.quaternion.y) < 1e-6, 'ArrowZ2 should decompose its matrix into a Z-axis rotation')

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.75))
  const light = new THREE.DirectionalLight(0xffffff, 1.0)
  light.position.set(4, 5, 6)
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)
  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const halfExtent = Math.max(size.x, size.y, size.z) / 2 + 0.5
  const camera = new THREE.OrthographicCamera(-halfExtent, halfExtent, halfExtent, -halfExtent, 0.01, 40)
  camera.position.set(center.x + 6, center.y + 4, center.z + 6)
  camera.lookAt(center)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 160,
    height: 160,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.9, 'OrientationTest should render visible rotated arrows and targets')
})

test('committed Khronos glTF Sample Assets MultipleScenes fixture preserves default and alternate scenes', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_MULTIPLE_SCENES)
  assert.equal(gltf.scenes.length, 2)
  assert.equal(gltf.scene, gltf.scenes[1], 'MultipleScenes should select glTF scene index 1 as the default scene')

  const triangleMesh = findFirst(gltf.scenes[0], (object) => object.isMesh === true)
  const squareMesh = findFirst(gltf.scenes[1], (object) => object.isMesh === true)
  assert.ok(triangleMesh, 'MultipleScenes first scene should load a triangle mesh')
  assert.ok(squareMesh, 'MultipleScenes default scene should load a square mesh')
  assert.equal(triangleMesh.geometry.getAttribute('position')?.count, 3)
  assert.equal(triangleMesh.geometry.index?.count, 3)
  assert.equal(squareMesh.geometry.getAttribute('position')?.count, 4)
  assert.equal(squareMesh.geometry.index?.count, 6)

  triangleMesh.material = new THREE.MeshBasicMaterial({ color: 0xffffff })
  squareMesh.material = new THREE.MeshBasicMaterial({ color: 0xffffff })

  const camera = new THREE.OrthographicCamera(-0.6, 0.6, 0.6, -0.6, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  const renderer = new Renderer()
  const renderScene = (scene) => {
    scene.position.set(-0.5, -0.5, 0)
    scene.updateMatrixWorld(true)
    return renderer.render(scene, camera, {
      width: 96,
      height: 96,
      format: 'rgba',
      background: [0, 0, 0],
      outputColorSpace: THREE.LinearSRGBColorSpace,
    })
  }

  const triangleRatio = nonBackgroundRatio(renderScene(gltf.scenes[0]), [0, 0, 0], 3)
  const squareRatio = nonBackgroundRatio(renderScene(gltf.scene), [0, 0, 0], 3)
  assert.ok(triangleRatio > 0.25, `alternate triangle scene should render visible pixels (${triangleRatio})`)
  assert.ok(squareRatio > 0.6, `default square scene should render visible pixels (${squareRatio})`)
  assert.ok(squareRatio > triangleRatio + 0.25, `default square scene should cover more pixels than alternate triangle scene (${squareRatio} vs ${triangleRatio})`)
})

test('committed Khronos glTF Sample Assets SimpleMaterial fixture loads scalar PBR material factors', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_SIMPLE_MATERIAL)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'Khronos SimpleMaterial sample should load a mesh')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 3)
  assert.equal(mesh.geometry.index?.count, 3)
  assert.equal(mesh.material.isMeshStandardMaterial, true)
  assert.deepEqual(mesh.material.color.toArray(), [1, 0.766, 0.336])
  assert.equal(mesh.material.metalness, 0.5)
  assert.equal(mesh.material.roughness, 0.1)

  mesh.position.set(-0.5, -0.5, 0)
  const camera = new THREE.OrthographicCamera(-0.6, 0.6, 0.6, -0.6, 0.01, 10)
  camera.position.set(0, 0, 2)
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.25, 'SimpleMaterial sample should render visible PBR geometry')
  const center = meanRegion(rgba, 96, 96, 34, 34, 62, 62)
  assert.ok(center.r > center.b + 50 && center.g > center.b + 35, `SimpleMaterial sample should render warm base-color pixels (${center.r}, ${center.g}, ${center.b})`)
})

test('committed Khronos glTF Sample Assets SimpleMeshes fixture reuses a mesh across nodes', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_SIMPLE_MESHES)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), ['mesh_0_instance_0', 'mesh_0_instance_1'])
  assert.equal(meshes[0].geometry, meshes[1].geometry, 'SimpleMeshes nodes should share one loaded geometry')
  assert.deepEqual(meshes[0].position.toArray(), [0, 0, 0])
  assert.deepEqual(meshes[1].position.toArray(), [1, 0, 0])
  assert.equal(meshes[0].geometry.getAttribute('position')?.count, 3)
  assert.equal(meshes[0].geometry.getAttribute('normal')?.count, 3)
  assert.equal(meshes[0].geometry.index?.count, 3)

  for (const mesh of meshes) mesh.material = new THREE.MeshBasicMaterial({ color: 0xffffff })
  gltf.scene.position.set(-0.75, -0.5, 0)
  const camera = new THREE.OrthographicCamera(-0.85, 0.85, 0.65, -0.65, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 128,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.35, 'SimpleMeshes sample should render both shared-geometry mesh instances')
  const left = meanRegion(rgba, 128, 96, 20, 45, 48, 75)
  const right = meanRegion(rgba, 128, 96, 80, 45, 108, 75)
  assert.ok(left.r > 120 && left.g > 120 && left.b > 120, `first shared mesh instance should render visibly (${left.r}, ${left.g}, ${left.b})`)
  assert.ok(right.r > 120 && right.g > 120 && right.b > 120, `second shared mesh instance should render visibly (${right.r}, ${right.g}, ${right.b})`)
})

test('committed Khronos glTF Sample Assets UnlitTest fixture loads KHR_materials_unlit', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_UNLIT_TEST)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), ['Orange_Object', 'Blue_Object'])

  const [orange, blue] = meshes
  assert.equal(orange.material.isMeshBasicMaterial, true)
  assert.equal(blue.material.isMeshBasicMaterial, true)
  assert.equal(orange.geometry.getAttribute('position')?.count, 96)
  assert.equal(orange.geometry.getAttribute('normal')?.count, 96)
  assert.equal(orange.geometry.index?.count, 132)
  assert.deepEqual(orange.material.color.toArray(), [1, 0.217637640824031, 0])
  assert.deepEqual(blue.material.color.toArray(), [0, 0.217637640824031, 1])

  const camera = new THREE.PerspectiveCamera(45, 4 / 3, 0.01, 20)
  camera.position.set(0, 0, 6)
  camera.lookAt(0, 0, 0)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 128,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.3, 'UnlitTest should render visible objects without scene lights')
  const left = meanRegion(rgba, 128, 96, 24, 32, 54, 64)
  const right = meanRegion(rgba, 128, 96, 74, 32, 104, 64)
  assert.ok(left.r > left.g + 180 && left.r > left.b + 200, `unlit orange mesh should render orange without lights (${left.r}, ${left.g}, ${left.b})`)
  assert.ok(right.b > right.g + 150 && right.b > right.r + 180, `unlit blue mesh should render blue without lights (${right.r}, ${right.g}, ${right.b})`)
})

test('committed Khronos glTF Sample Assets Triangle fixture loads minimal indexed primitive', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_TRIANGLE, 'utf8'))
  assert.deepEqual(source.buffers, [{ uri: 'Triangle.bin', byteLength: 44 }])
  assert.deepEqual(source.meshes[0].primitives, [
    {
      attributes: { POSITION: 1 },
      indices: 0,
    },
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_TRIANGLE)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'Khronos Triangle sample should load a mesh')
  assert.equal(mesh.name, 'mesh_0')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 3)
  assert.equal(mesh.geometry.getAttribute('normal') ?? null, null)
  assert.equal(mesh.geometry.getAttribute('uv') ?? null, null)
  assert.equal(mesh.geometry.index?.count, 3)
  assert.equal(mesh.material.isMeshStandardMaterial, true)
  assert.deepEqual(mesh.material.color.toArray(), [1, 1, 1])
  assert.equal(mesh.material.metalness, 1)
  assert.equal(mesh.material.roughness, 1)

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
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.035, 'Triangle sample should render visible minimal indexed geometry')
})

test('committed Khronos glTF Sample Assets TriangleWithoutIndices fixture loads non-indexed geometry', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_TRIANGLE_WITHOUT_INDICES)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'Khronos TriangleWithoutIndices sample should load a mesh')
  assert.equal(mesh.geometry.index, null)
  assert.equal(mesh.geometry.getAttribute('position')?.count, 3)
  assert.deepEqual(vectorFromAttribute(mesh.geometry.getAttribute('position'), 0), [0, 0, 0])
  assert.deepEqual(vectorFromAttribute(mesh.geometry.getAttribute('position'), 1), [1, 0, 0])
  assert.deepEqual(vectorFromAttribute(mesh.geometry.getAttribute('position'), 2), [0, 1, 0])

  mesh.material = new THREE.MeshBasicMaterial({ color: 0xffffff })
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.25, 'TriangleWithoutIndices sample should render visible non-indexed geometry')
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

test('real external VRM and VRMA fixtures expose extension metadata through loader helpers', async () => {
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

  const vrmGltf = await loadVrmFromFile(REAL_VRM_EXPRESSION_SAMPLE, {
    VRMLoaderPlugin: CaptureVRMLoaderPlugin,
  })
  assert.ok(findFirst(vrmGltf.scene, (object) => object.isMesh === true), 'real VRM fixture should parse renderable meshes')
  assert.ok(vrmPluginParser, 'real VRM fixture should initialize the supplied VRM loader plugin')
  assert.ok(vrmPluginParser.json?.extensionsUsed?.includes('VRMC_vrm'), 'real VRM fixture should expose VRMC_vrm metadata')
  assert.ok(
    vrmPluginParser.json?.extensionsUsed?.includes('KHR_texture_transform'),
    'real VRM fixture should expose its texture transform extension metadata',
  )
  assert.equal(vrmPluginParser.json?.meshes?.length, 4)

  const vrmExtension = vrmPluginParser.json?.extensions?.VRMC_vrm
  assert.equal(vrmExtension?.specVersion, '1.0')
  assert.equal(vrmExtension?.meta?.name, 'isBinary overridden')
  assert.equal(vrmExtension?.meta?.licenseUrl, 'https://vrm.dev/licenses/1.0/')
  assert.equal(vrmExtension?.meta?.allowRedistribution, true)
  assert.equal(vrmExtension?.expressions?.preset?.happy?.overrideBlink, 'blend')
  assert.equal(vrmExtension?.expressions?.preset?.blink?.isBinary, true)

  const animationGltf = await loadVrmAnimationFromFile(REAL_VRMA_ANIMATION_SAMPLE, {
    VRMAnimationLoaderPlugin: CaptureVRMAnimationLoaderPlugin,
  })
  assert.equal(animationGltf.animations.length, 1)
  assert.ok(animationPluginParser, 'real VRMA fixture should initialize the supplied VRM animation loader plugin')
  assert.ok(
    animationPluginParser.json?.extensionsUsed?.includes('VRMC_vrm_animation'),
    'real VRMA fixture should expose VRMC_vrm_animation metadata',
  )
  assert.equal(animationPluginParser.json?.nodes?.length, 53)
  assert.equal(animationPluginParser.json?.animations?.[0]?.channels?.length, 3)
  assert.equal(animationPluginParser.json?.animations?.[0]?.samplers?.length, 3)

  const vrmaExtension = animationPluginParser.json?.extensions?.VRMC_vrm_animation
  const humanBones = vrmaExtension?.humanoid?.humanBones ?? {}
  assert.equal(vrmaExtension?.specVersion, '1.0')
  assert.ok('hips' in humanBones, 'real VRMA fixture should map humanoid hips')
  assert.ok('leftUpperArm' in humanBones, 'real VRMA fixture should map humanoid upper-body bones')
  assert.ok('rightFoot' in humanBones, 'real VRMA fixture should map humanoid lower-body bones')
  assert.equal(vrmaExtension?.lookAt?.node, 52)
  assert.equal(vrmaExtension?.expressions?.preset?.happy?.node, 51)
})

test('loadGltfFromFile resolves external glTF image files from the model directory', async () => {
  const source = JSON.parse(await readFile(TEXTURED_QUAD, 'utf8'))
  const imageUri = source.images[0].uri
  const encodedImage = imageUri.slice(imageUri.indexOf(',') + 1)
  const tmp = await mkdtemp(path.join(os.tmpdir(), 'headless-three-gltf-image-'))
  try {
    const textureDir = path.join(tmp, 'textures')
    await mkdir(textureDir)
    await writeFile(path.join(textureDir, 'quad.png'), Buffer.from(encodedImage, 'base64'))
    source.images[0].uri = 'textures/quad.png'
    const modelPath = path.join(tmp, 'external-image.gltf')
    await writeFile(modelPath, JSON.stringify(source))

    const gltf = await loadGltfFixture(modelPath)
    assertTexturedQuadLoadsEncodedMap(gltf, 'external-image fixture')
    assertTexturedQuadRendersTexture(gltf, 'external-image quad')
  } finally {
    await rm(tmp, { recursive: true, force: true })
  }
})

test('loadGltfFromFile resolves external glTF buffers from the model directory', async () => {
  const source = JSON.parse(await readFile(SIMPLE_TRIANGLE, 'utf8'))
  const bufferUri = source.buffers[0].uri
  const encodedBuffer = bufferUri.slice(bufferUri.indexOf(',') + 1)
  const tmp = await mkdtemp(path.join(os.tmpdir(), 'headless-three-gltf-'))
  try {
    await writeFile(path.join(tmp, 'triangle.bin'), Buffer.from(encodedBuffer, 'base64'))
    source.buffers[0].uri = 'triangle.bin'
    const modelPath = path.join(tmp, 'external-buffer.gltf')
    await writeFile(modelPath, JSON.stringify(source))

    const gltf = await loadGltfFixture(modelPath)
    const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
    assert.ok(mesh, 'external-buffer fixture should load a mesh')

    const camera = gltf.cameras[0]
    assert.ok(camera, 'external-buffer fixture should load a camera')
    camera.aspect = 1
    camera.updateProjectionMatrix()
    gltf.scene.add(new THREE.AmbientLight(0xffffff, 1))
    gltf.scene.updateMatrixWorld(true)
    camera.updateMatrixWorld(true)

    const rgba = new Renderer().render(gltf.scene, camera, {
      width: 64,
      height: 64,
      format: 'rgba',
      background: [0, 0, 0],
    })
    assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.04, 'external buffer glTF should render visible pixels')
  } finally {
    await rm(tmp, { recursive: true, force: true })
  }
})

function assertTexturedQuadLoadsEncodedMap(gltf, label) {
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, `${label} should load a mesh`)
  assert.ok(mesh.material.map?.isTexture, `${label} should load a base color texture`)
  assert.ok(Buffer.isBuffer(mesh.material.map.image), 'encoded image helper should expose the PNG as a Buffer')
}

function assertTexturedQuadRendersTexture(gltf, label) {
  const camera = gltf.cameras[0]
  assert.ok(camera, `${label} should load a camera`)
  camera.aspect = 1
  camera.updateProjectionMatrix()

  const scene = gltf.scene
  scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })
  assert.equal(rgba.length, 96 * 96 * 4)
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.25, `${label} should render visible pixels`)

  const left = meanRegion(rgba, 96, 96, 24, 36, 42, 60)
  const right = meanRegion(rgba, 96, 96, 54, 36, 72, 60)
  assert.ok(left.r > left.g + 80, `left half should sample the red texture texel (${left.r} vs ${left.g})`)
  assert.ok(right.g > right.r + 80, `right half should sample the green texture texel (${right.g} vs ${right.r})`)
}

function assertTextureSampler(mesh, wrapS, wrapT) {
  assert.ok(mesh?.isMesh, 'texture sampler assertion requires a mesh')
  const texture = mesh.material.map
  assert.ok(texture?.isTexture, `${mesh.name} should load a base color texture`)
  assert.equal(Buffer.isBuffer(texture.image), true, `${mesh.name} texture should load as an encoded Buffer`)
  assert.equal(texture.wrapS, wrapS, `${mesh.name} should preserve sampler wrapS`)
  assert.equal(texture.wrapT, wrapT, `${mesh.name} should preserve sampler wrapT`)
  assert.equal(texture.magFilter, THREE.LinearFilter)
  assert.equal(texture.minFilter, THREE.NearestMipmapLinearFilter)
  assert.equal(texture.flipY, false)
}

async function loadGltfFixture(filePath, options) {
  return await loadGltfFromFile(filePath, options)
}

function vectorFromAttribute(attribute, index) {
  return [attribute.getX(index), attribute.getY(index), attribute.getZ(index)]
}

function pngDimensions(buffer) {
  assert.equal(Buffer.isBuffer(buffer), true, 'PNG source should be an encoded Buffer')
  assert.equal(buffer.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])), true, 'PNG source should start with a PNG signature')
  return [buffer.readUInt32BE(16), buffer.readUInt32BE(20)]
}

function assertVectorClose(actual, expected, label, tolerance = 1e-6) {
  assert.equal(actual.length, expected.length, `${label} should have ${expected.length} components`)
  for (let i = 0; i < expected.length; i++) {
    assert.ok(Math.abs(actual[i] - expected[i]) <= tolerance, `${label}[${i}] should be close to ${expected[i]} (${actual[i]})`)
  }
}

function isEffectivelyVisible(object) {
  let current = object
  while (current) {
    if (current.visible === false) return false
    current = current.parent
  }
  return true
}

function worldDeterminant(object) {
  object.updateWorldMatrix(true, false)
  return object.matrixWorld.determinant()
}

function renderSingleObjectRatio(renderer, object, padding = 0.2) {
  object.updateWorldMatrix(true, true)
  const bounds = new THREE.Box3().setFromObject(object)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())

  const scene = new THREE.Scene()
  scene.add(object.clone(true))
  scene.add(new THREE.AmbientLight(0xffffff, 1.0))

  const camera = new THREE.OrthographicCamera(
    -size.x / 2 - padding,
    size.x / 2 + padding,
    size.y / 2 + padding,
    -size.y / 2 - padding,
    0.01,
    20,
  )
  camera.position.set(center.x, center.y, center.z + 8)
  camera.lookAt(center)
  scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = renderer.render(scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })
  return nonBackgroundRatio(rgba, [0, 0, 0], 3)
}

async function assertRejectsMutatedGltfSource(mutator, pattern) {
  const source = JSON.parse(await readFile(TEXTURED_QUAD, 'utf8'))
  mutator(source)
  const tmp = await mkdtemp(path.join(os.tmpdir(), 'headless-three-gltf-malformed-image-'))
  try {
    const modelPath = path.join(tmp, 'malformed-image-metadata.gltf')
    await writeFile(modelPath, JSON.stringify(source))
    await assert.rejects(
      () => loadGltfFixture(modelPath),
      pattern,
    )
  } finally {
    await rm(tmp, { recursive: true, force: true })
  }
}

function buildTexturedQuadGlb(source) {
  const geometryBytes = decodeDataUriBuffer(source.buffers[0].uri, 'textured fixture geometry buffer')
  const imageBytes = decodeDataUriBuffer(source.images[0].uri, 'textured fixture image')
  const imageOffset = alignedLength(geometryBytes.length)
  const binLength = imageOffset + imageBytes.length
  const bin = Buffer.alloc(alignedLength(binLength))
  geometryBytes.copy(bin, 0)
  imageBytes.copy(bin, imageOffset)

  const glb = structuredClone(source)
  delete glb.buffers[0].uri
  glb.buffers[0].byteLength = binLength
  glb.bufferViews.push({
    buffer: 0,
    byteOffset: imageOffset,
    byteLength: imageBytes.length,
  })
  glb.images[0] = {
    name: source.images[0].name,
    mimeType: source.images[0].mimeType,
    bufferView: glb.bufferViews.length - 1,
  }

  return encodeGlb(glb, bin)
}

function decodeDataUriBuffer(uri, label) {
  assert.equal(typeof uri, 'string', `${label} should be a data URI`)
  const comma = uri.indexOf(',')
  assert.notEqual(comma, -1, `${label} should contain a comma separator`)
  const metadata = uri.slice(5, comma)
  const payload = uri.slice(comma + 1)
  return /(?:^|;)base64(?:;|$)/i.test(metadata)
    ? Buffer.from(payload, 'base64')
    : Buffer.from(decodeURIComponent(payload), 'utf8')
}

function encodeGlb(json, bin) {
  const jsonChunk = paddedBuffer(Buffer.from(JSON.stringify(json), 'utf8'), 0x20)
  const binChunk = paddedBuffer(bin, 0x00)
  const totalLength = 12 + 8 + jsonChunk.length + 8 + binChunk.length
  const glb = Buffer.alloc(totalLength)
  let offset = 0
  offset = writeUint32(glb, offset, 0x46546c67)
  offset = writeUint32(glb, offset, 2)
  offset = writeUint32(glb, offset, totalLength)
  offset = writeUint32(glb, offset, jsonChunk.length)
  offset = writeUint32(glb, offset, 0x4e4f534a)
  jsonChunk.copy(glb, offset)
  offset += jsonChunk.length
  offset = writeUint32(glb, offset, binChunk.length)
  offset = writeUint32(glb, offset, 0x004e4942)
  binChunk.copy(glb, offset)
  return glb
}

function paddedBuffer(buffer, fill) {
  const padded = Buffer.alloc(alignedLength(buffer.length), fill)
  buffer.copy(padded)
  return padded
}

function alignedLength(length) {
  return (length + 3) & ~3
}

function writeUint32(buffer, offset, value) {
  buffer.writeUInt32LE(value, offset)
  return offset + 4
}

function findFirst(root, predicate) {
  let match = null
  root.traverse((object) => {
    if (!match && predicate(object)) match = object
  })
  return match
}

function meanRegion(rgba, width, _height, x0, y0, x1, y1) {
  let r = 0
  let g = 0
  let b = 0
  let a = 0
  let count = 0
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const i = (y * width + x) * 4
      r += rgba[i]
      g += rgba[i + 1]
      b += rgba[i + 2]
      a += rgba[i + 3]
      count++
    }
  }
  return { r: r / count, g: g / count, b: b / count, a: a / count }
}

function nonBackgroundBounds(rgba, width, height, bg, tolerance = 2) {
  let minX = width
  let minY = height
  let maxX = -1
  let maxY = -1
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const i = (y * width + x) * 4
      if (
        Math.abs(rgba[i] - bg[0]) > tolerance ||
        Math.abs(rgba[i + 1] - bg[1]) > tolerance ||
        Math.abs(rgba[i + 2] - bg[2]) > tolerance
      ) {
        minX = Math.min(minX, x)
        minY = Math.min(minY, y)
        maxX = Math.max(maxX, x)
        maxY = Math.max(maxY, y)
      }
    }
  }
  return {
    minX: maxX >= minX ? minX : 0,
    minY: maxY >= minY ? minY : 0,
    maxX,
    maxY,
    width: maxX >= minX ? maxX - minX + 1 : 0,
    height: maxY >= minY ? maxY - minY + 1 : 0,
  }
}
