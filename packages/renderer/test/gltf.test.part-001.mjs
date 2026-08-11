import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'
import { findFirst, frameSceneCamera, loadGltfFixture, meanRegion, pngDimensions } from './gltf.test.part-028.mjs'
export const {
  Renderer,
  loadGltfFromFile,
  loadVrmAnimationFromFile,
  loadVrmFromFile,
} = pkg

export const FIXTURE_DIR = fileURLToPath(new URL('./fixtures/', import.meta.url))
export const SIMPLE_TRIANGLE = path.join(FIXTURE_DIR, 'simple-triangle.gltf')
export const TEXTURED_QUAD = path.join(FIXTURE_DIR, 'textured-quad.gltf')
export const VERTEX_COLOR_QUAD = path.join(FIXTURE_DIR, 'vertex-color-quad.gltf')
export const MORPHED_TRIANGLE = path.join(FIXTURE_DIR, 'morphed-triangle.gltf')
export const SKINNED_QUAD = path.join(FIXTURE_DIR, 'skinned-quad.gltf')
export const SYNTHETIC_VRM = path.join(FIXTURE_DIR, 'synthetic-avatar.vrm')
export const SYNTHETIC_VRMA = path.join(FIXTURE_DIR, 'synthetic-animation.vrma')
export const SYNTHETIC_HUMANOID_VRM = path.join(FIXTURE_DIR, 'synthetic-humanoid-avatar.vrm')
export const SYNTHETIC_HUMANOID_VRMA = path.join(FIXTURE_DIR, 'synthetic-humanoid-animation.vrma')
export const REAL_VRM_SEED_SAN_SAMPLE = path.join(
  FIXTURE_DIR,
  'vrm-specification',
  'Seed-san',
  'vrm',
  'Seed-san.vrm',
)
export const REAL_VRM_EXPRESSION_SAMPLE = path.join(
  FIXTURE_DIR,
  'vrm-specification',
  'VRMC_vrm_expressions_isBinary_Overridden',
  'VRMC_vrm_expressions_isBinary_Overridden.vrm',
)
export const REAL_VRM_EXPRESSION_OVERRIDES_SAMPLE = path.join(
  FIXTURE_DIR,
  'vrm-specification',
  'VRMC_vrm_expressions_isBinary_Overrides',
  'VRMC_vrm_expressions_isBinary_Overrides.vrm',
)
export const REAL_VRMA_ANIMATION_SAMPLE = path.join(FIXTURE_DIR, 'three-vrm-animation', 'test.vrma')
export const SAMPLE_ASSET_A_BEAUTIFUL_GAME = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'ABeautifulGame', 'glTF', 'ABeautifulGame.gltf')
export const SAMPLE_ASSET_ANIMATED_CUBE = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'AnimatedCube', 'glTF', 'AnimatedCube.gltf')
export const SAMPLE_ASSET_ANIMATED_COLORS_CUBE = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'AnimatedColorsCube', 'glTF', 'AnimatedColorsCube.gltf')
export const SAMPLE_ASSET_ANIMATED_MORPH_CUBE = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'AnimatedMorphCube', 'glTF', 'AnimatedMorphCube.gltf')
export const SAMPLE_ASSET_ANIMATED_TRIANGLE = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'AnimatedTriangle', 'glTF', 'AnimatedTriangle.gltf')
export const SAMPLE_ASSET_ANIMATION_POINTER_UVS = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'AnimationPointerUVs', 'glTF', 'AnimationPointerUVs.gltf')
export const SAMPLE_ASSET_ALPHA_BLEND_MODE_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'AlphaBlendModeTest', 'glTF', 'AlphaBlendModeTest.gltf')
export const SAMPLE_ASSET_ANISOTROPY_BARN_LAMP = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'AnisotropyBarnLamp', 'glTF', 'AnisotropyBarnLamp.gltf')
export const SAMPLE_ASSET_ANISOTROPY_DISC_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'AnisotropyDiscTest', 'glTF', 'AnisotropyDiscTest.gltf')
export const SAMPLE_ASSET_ANISOTROPY_ROTATION_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'AnisotropyRotationTest', 'glTF', 'AnisotropyRotationTest.gltf')
export const SAMPLE_ASSET_ANISOTROPY_STRENGTH_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'AnisotropyStrengthTest', 'glTF', 'AnisotropyStrengthTest.gltf')
export const SAMPLE_ASSET_ANTIQUE_CAMERA = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'AntiqueCamera', 'glTF', 'AntiqueCamera.gltf')
export const SAMPLE_ASSET_ATTENUATION_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'AttenuationTest', 'glTF', 'AttenuationTest.gltf')
export const SAMPLE_ASSET_AVOCADO = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'Avocado', 'glTF', 'Avocado.gltf')
export const SAMPLE_ASSET_BARRAMUNDI_FISH = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'BarramundiFish', 'glTF', 'BarramundiFish.gltf')
export const SAMPLE_ASSET_BOOM_BOX = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'BoomBox', 'glTF', 'BoomBox.gltf')
export const SAMPLE_ASSET_BOOM_BOX_WITH_AXES = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'BoomBoxWithAxes', 'glTF', 'BoomBoxWithAxes.gltf')
export const SAMPLE_ASSET_BOX_ANIMATED = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'BoxAnimated', 'glTF', 'BoxAnimated.gltf')
export const SAMPLE_ASSET_BOX = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'Box', 'glTF', 'Box.gltf')
export const SAMPLE_ASSET_BOX_WITH_SPACES = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'Box With Spaces', 'glTF', 'Box With Spaces.gltf')
export const SAMPLE_ASSET_BOX_INTERLEAVED = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'BoxInterleaved', 'glTF', 'BoxInterleaved.gltf')
export const SAMPLE_ASSET_BOX_TEXTURED = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'BoxTextured', 'glTF', 'BoxTextured.gltf')
export const SAMPLE_ASSET_BOX_TEXTURED_NPOT = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'BoxTexturedNonPowerOfTwo', 'glTF', 'BoxTexturedNonPowerOfTwo.gltf')
export const SAMPLE_ASSET_BOX_VERTEX_COLORS = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'BoxVertexColors', 'glTF', 'BoxVertexColors.gltf')
export const SAMPLE_ASSET_BRAIN_STEM = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'BrainStem', 'glTF', 'BrainStem.gltf')
export const SAMPLE_ASSET_CAMERAS = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'Cameras', 'glTF', 'Cameras.gltf')
export const SAMPLE_ASSET_CARBON_FIBRE = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CarbonFibre', 'glTF', 'CarbonFibre.gltf')
export const SAMPLE_ASSET_CAR_CONCEPT = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CarConcept', 'glTF', 'CarConcept.gltf')
export const SAMPLE_ASSET_CESIUM_MAN = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CesiumMan', 'glTF', 'CesiumMan.gltf')
export const SAMPLE_ASSET_CESIUM_MILK_TRUCK = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CesiumMilkTruck', 'glTF', 'CesiumMilkTruck.gltf')
export const SAMPLE_ASSET_CHAIR_DAMASK_PURPLEGOLD = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'ChairDamaskPurplegold', 'glTF', 'ChairDamaskPurplegold.gltf')
export const SAMPLE_ASSET_CHRONOGRAPH_WATCH = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'ChronographWatch', 'glTF', 'ChronographWatch.gltf')
export const SAMPLE_ASSET_CLEARCOAT_CAR_PAINT = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'ClearCoatCarPaint', 'glTF', 'ClearCoatCarPaint.gltf')
export const SAMPLE_ASSET_CLEARCOAT_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'ClearCoatTest', 'glTF', 'ClearCoatTest.gltf')
export const SAMPLE_ASSET_CLEARCOAT_WICKER = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'ClearcoatWicker', 'glTF', 'ClearcoatWicker.gltf')
export const SAMPLE_ASSET_COMMERCIAL_REFRIGERATOR = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CommercialRefrigerator', 'glTF', 'CommercialRefrigerator.gltf')
export const SAMPLE_ASSET_COMPARE_ALPHA_COVERAGE = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CompareAlphaCoverage', 'glTF', 'CompareAlphaCoverage.gltf')
export const SAMPLE_ASSET_COMPARE_AMBIENT_OCCLUSION = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CompareAmbientOcclusion', 'glTF', 'CompareAmbientOcclusion.gltf')
export const SAMPLE_ASSET_COMPARE_ANISOTROPY = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CompareAnisotropy', 'glTF', 'CompareAnisotropy.gltf')
export const SAMPLE_ASSET_COMPARE_BASE_COLOR = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CompareBaseColor', 'glTF', 'CompareBaseColor.gltf')
export const SAMPLE_ASSET_COMPARE_CLEARCOAT = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CompareClearcoat', 'glTF', 'CompareClearcoat.gltf')
export const SAMPLE_ASSET_COMPARE_DISPERSION = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CompareDispersion', 'glTF', 'CompareDispersion.gltf')
export const SAMPLE_ASSET_COMPARE_EMISSIVE_STRENGTH = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CompareEmissiveStrength', 'glTF', 'CompareEmissiveStrength.gltf')
export const SAMPLE_ASSET_COMPARE_IOR = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CompareIor', 'glTF', 'CompareIor.gltf')
export const SAMPLE_ASSET_COMPARE_IRIDESCENCE = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CompareIridescence', 'glTF', 'CompareIridescence.gltf')
export const SAMPLE_ASSET_COMPARE_METALLIC = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CompareMetallic', 'glTF', 'CompareMetallic.gltf')
export const SAMPLE_ASSET_COMPARE_NORMAL = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CompareNormal', 'glTF', 'CompareNormal.gltf')
export const SAMPLE_ASSET_COMPARE_ROUGHNESS = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CompareRoughness', 'glTF', 'CompareRoughness.gltf')
export const SAMPLE_ASSET_COMPARE_SHEEN = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CompareSheen', 'glTF', 'CompareSheen.gltf')
export const SAMPLE_ASSET_COMPARE_SPECULAR = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CompareSpecular', 'glTF', 'CompareSpecular.gltf')
export const SAMPLE_ASSET_COMPARE_TRANSMISSION = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CompareTransmission', 'glTF', 'CompareTransmission.gltf')
export const SAMPLE_ASSET_COMPARE_VOLUME = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CompareVolume', 'glTF', 'CompareVolume.gltf')
export const SAMPLE_ASSET_CORSET = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'Corset', 'glTF', 'Corset.gltf')
export const SAMPLE_ASSET_CUBE = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'Cube', 'glTF', 'Cube.gltf')
export const SAMPLE_ASSET_CUBE_VISIBILITY = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'CubeVisibility', 'glTF', 'CubeVisibility.gltf')
export const SAMPLE_ASSET_DAMAGED_HELMET = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'DamagedHelmet', 'glTF', 'DamagedHelmet.gltf')
export const SAMPLE_ASSET_DIRECTIONAL_LIGHT = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'DirectionalLight', 'glTF', 'DirectionalLight.gltf')
export const SAMPLE_ASSET_DIFFUSE_TRANSMISSION_PLANT = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'DiffuseTransmissionPlant', 'glTF', 'DiffuseTransmissionPlant.gltf')
export const SAMPLE_ASSET_DIFFUSE_TRANSMISSION_TEACUP = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'DiffuseTransmissionTeacup', 'glTF', 'DiffuseTransmissionTeacup.gltf')
export const SAMPLE_ASSET_DIFFUSE_TRANSMISSION_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'DiffuseTransmissionTest', 'glTF', 'DiffuseTransmissionTest.gltf')
export const SAMPLE_ASSET_DISPERSION_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'DispersionTest', 'glTF', 'DispersionTest.gltf')
export const SAMPLE_ASSET_DRAGON_ATTENUATION = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'DragonAttenuation', 'glTF', 'DragonAttenuation.gltf')
export const SAMPLE_ASSET_DRAGON_DISPERSION = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'DragonDispersion', 'glTF', 'DragonDispersion.gltf')
export const SAMPLE_ASSET_DUCK = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'Duck', 'glTF', 'Duck.gltf')
export const SAMPLE_ASSET_EMISSIVE_STRENGTH_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'EmissiveStrengthTest', 'glTF', 'EmissiveStrengthTest.gltf')
export const SAMPLE_ASSET_ENVIRONMENT_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'EnvironmentTest', 'glTF', 'EnvironmentTest.gltf')
export const SAMPLE_ASSET_FLIGHT_HELMET = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'FlightHelmet', 'glTF', 'FlightHelmet.gltf')
export const SAMPLE_ASSET_FOX = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'Fox', 'glTF', 'Fox.gltf')
export const SAMPLE_ASSET_GLAM_VELVET_SOFA = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'GlamVelvetSofa', 'glTF', 'GlamVelvetSofa.gltf')
export const SAMPLE_ASSET_GLASS_BROKEN_WINDOW = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'GlassBrokenWindow', 'glTF', 'GlassBrokenWindow.gltf')
export const SAMPLE_ASSET_GLASS_HURRICANE_CANDLE_HOLDER = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'GlassHurricaneCandleHolder', 'glTF', 'GlassHurricaneCandleHolder.gltf')
export const SAMPLE_ASSET_GLASS_VASE_FLOWERS = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'GlassVaseFlowers', 'glTF', 'GlassVaseFlowers.gltf')
export const SAMPLE_ASSET_INTERPOLATION_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'InterpolationTest', 'glTF', 'InterpolationTest.gltf')
export const SAMPLE_ASSET_IRIDESCENCE_ABALONE = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'IridescenceAbalone', 'glTF', 'IridescenceAbalone.gltf')
export const SAMPLE_ASSET_IOR_TEST_GRID = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'IORTestGrid', 'glTF', 'IORTestGrid.gltf')
export const SAMPLE_ASSET_IRIDESCENCE_DIELECTRIC_SPHERES = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'IridescenceDielectricSpheres', 'glTF', 'IridescenceDielectricSpheres.gltf')
export const SAMPLE_ASSET_IRIDESCENCE_LAMP = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'IridescenceLamp', 'glTF', 'IridescenceLamp.gltf')
export const SAMPLE_ASSET_IRIDESCENCE_METALLIC_SPHERES = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'IridescenceMetallicSpheres', 'glTF', 'IridescenceMetallicSpheres.gltf')
export const SAMPLE_ASSET_IRIDESCENCE_SUZANNE = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'IridescenceSuzanne', 'glTF', 'IridescenceSuzanne.gltf')
export const SAMPLE_ASSET_IRIDESCENT_DISH_WITH_OLIVES = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'IridescentDishWithOlives', 'glTF', 'IridescentDishWithOlives.gltf')
export const SAMPLE_ASSET_LANTERN = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'Lantern', 'glTF', 'Lantern.gltf')
export const SAMPLE_ASSET_LIGHT_VISIBILITY = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'LightVisibility', 'glTF', 'LightVisibility.gltf')
export const SAMPLE_ASSET_LIGHTS_PUNCTUAL_LAMP = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'LightsPunctualLamp', 'glTF', 'LightsPunctualLamp.gltf')
export const SAMPLE_ASSET_MANDARIN_ORANGE = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'MandarinOrange', 'glTF', 'MandarinOrange.gltf')
export const SAMPLE_ASSET_MATERIALS_VARIANTS_SHOE = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'MaterialsVariantsShoe', 'glTF', 'MaterialsVariantsShoe.gltf')
export const SAMPLE_ASSET_METAL_ROUGH_SPHERES = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'MetalRoughSpheres', 'glTF', 'MetalRoughSpheres.gltf')
export const SAMPLE_ASSET_METAL_ROUGH_SPHERES_NO_TEXTURES = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'MetalRoughSpheresNoTextures', 'glTF', 'MetalRoughSpheresNoTextures.gltf')
export const SAMPLE_ASSET_MESH_PRIMITIVE_MODES = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'MeshPrimitiveModes', 'glTF', 'MeshPrimitiveModes.gltf')
export const SAMPLE_ASSET_MESHOPT_CUBE_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'MeshoptCubeTest', 'glTF', 'MeshoptCubeTest.gltf')
export const SAMPLE_ASSET_MOSQUITO_IN_AMBER = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'MosquitoInAmber', 'glTF', 'MosquitoInAmber.gltf')
export const SAMPLE_ASSET_MORPH_PRIMITIVES_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'MorphPrimitivesTest', 'glTF', 'MorphPrimitivesTest.gltf')
export const SAMPLE_ASSET_MORPH_STRESS_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'MorphStressTest', 'glTF', 'MorphStressTest.gltf')
export const SAMPLE_ASSET_MULTI_UV_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'MultiUVTest', 'glTF', 'MultiUVTest.gltf')
export const SAMPLE_ASSET_MULTIPLE_SCENES = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'MultipleScenes', 'glTF', 'MultipleScenes.gltf')
export const SAMPLE_ASSET_NEGATIVE_SCALE_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'NegativeScaleTest', 'glTF', 'NegativeScaleTest.gltf')
export const SAMPLE_ASSET_NORMAL_TANGENT_MIRROR_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'NormalTangentMirrorTest', 'glTF', 'NormalTangentMirrorTest.gltf')
export const SAMPLE_ASSET_NORMAL_TANGENT_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'NormalTangentTest', 'glTF', 'NormalTangentTest.gltf')
export const SAMPLE_ASSET_ORIENTATION_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'OrientationTest', 'glTF', 'OrientationTest.gltf')
export const SAMPLE_ASSET_PLAYSET_LIGHT_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'PlaysetLightTest', 'glTF', 'PlaysetLightTest.gltf')
export const SAMPLE_ASSET_POINT_LIGHT_INTENSITY_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'PointLightIntensityTest', 'glTF', 'PointLightIntensityTest.gltf')
export const SAMPLE_ASSET_POT_OF_COALS = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'PotOfCoals', 'glTF', 'PotOfCoals.gltf')
export const SAMPLE_ASSET_POT_OF_COALS_ANIMATION_POINTER = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'PotOfCoalsAnimationPointer', 'glTF', 'PotOfCoalsAnimationPointer.gltf')
export const SAMPLE_ASSET_PRIMITIVE_MODE_NORMALS_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'PrimitiveModeNormalsTest', 'glTF', 'PrimitiveModeNormalsTest.gltf')
export const SAMPLE_ASSET_RECURSIVE_SKELETONS = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'RecursiveSkeletons', 'glTF', 'RecursiveSkeletons.gltf')
export const SAMPLE_ASSET_RIGGED_FIGURE = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'RiggedFigure', 'glTF', 'RiggedFigure.gltf')
export const SAMPLE_ASSET_RIGGED_SIMPLE = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'RiggedSimple', 'glTF', 'RiggedSimple.gltf')
export const SAMPLE_ASSET_SCATTERING_SKULL = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'ScatteringSkull', 'glTF', 'ScatteringSkull.gltf')
export const SAMPLE_ASSET_SCIFI_HELMET = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'SciFiHelmet', 'glTF', 'SciFiHelmet.gltf')
export const SAMPLE_ASSET_SHEEN_CHAIR = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'SheenChair', 'glTF', 'SheenChair.gltf')
export const SAMPLE_ASSET_SHEEN_CLOTH = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'SheenCloth', 'glTF', 'SheenCloth.gltf')
export const SAMPLE_ASSET_SHEEN_TEST_GRID = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'SheenTestGrid', 'glTF', 'SheenTestGrid.gltf')
export const SAMPLE_ASSET_SHEEN_WOOD_LEATHER_SOFA = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'SheenWoodLeatherSofa', 'glTF', 'SheenWoodLeatherSofa.gltf')
export const SAMPLE_ASSET_SIMPLE_INSTANCING = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'SimpleInstancing', 'glTF', 'SimpleInstancing.gltf')
export const SAMPLE_ASSET_SIMPLE_MATERIAL = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'SimpleMaterial', 'glTF', 'SimpleMaterial.gltf')
export const SAMPLE_ASSET_SIMPLE_MESHES = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'SimpleMeshes', 'glTF', 'SimpleMeshes.gltf')
export const SAMPLE_ASSET_SIMPLE_MORPH = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'SimpleMorph', 'glTF', 'SimpleMorph.gltf')
export const SAMPLE_ASSET_SIMPLE_SKIN = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'SimpleSkin', 'glTF', 'SimpleSkin.gltf')
export const SAMPLE_ASSET_SIMPLE_SPARSE_ACCESSOR = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'SimpleSparseAccessor', 'glTF', 'SimpleSparseAccessor.gltf')
export const SAMPLE_ASSET_SIMPLE_TEXTURE = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'SimpleTexture', 'glTF', 'SimpleTexture.gltf')
export const SAMPLE_ASSET_SPEC_GLOSS_VS_METAL_ROUGH = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'SpecGlossVsMetalRough', 'glTF', 'SpecGlossVsMetalRough.gltf')
export const SAMPLE_ASSET_SPECULAR_SILK_POUF = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'SpecularSilkPouf', 'glTF', 'SpecularSilkPouf.gltf')
export const SAMPLE_ASSET_SPECULAR_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'SpecularTest', 'glTF', 'SpecularTest.gltf')
export const SAMPLE_ASSET_SPONZA = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'Sponza', 'glTF', 'Sponza.gltf')
export const SAMPLE_ASSET_STAINED_GLASS_LAMP = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'StainedGlassLamp', 'glTF', 'StainedGlassLamp.gltf')
export const SAMPLE_ASSET_SUNGLASSES_KHRONOS = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'SunglassesKhronos', 'glTF', 'SunglassesKhronos.gltf')
export const SAMPLE_ASSET_SUZANNE = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'Suzanne', 'glTF', 'Suzanne.gltf')
export const SAMPLE_ASSET_TEXTURE_COORDINATE_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'TextureCoordinateTest', 'glTF', 'TextureCoordinateTest.gltf')
export const SAMPLE_ASSET_TEXTURE_ENCODING_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'TextureEncodingTest', 'glTF', 'TextureEncodingTest.gltf')
export const SAMPLE_ASSET_TEXTURE_LINEAR_INTERPOLATION_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'TextureLinearInterpolationTest', 'glTF', 'TextureLinearInterpolationTest.gltf')
export const SAMPLE_ASSET_TEXTURE_SETTINGS_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'TextureSettingsTest', 'glTF', 'TextureSettingsTest.gltf')
export const SAMPLE_ASSET_TEXTURE_TRANSFORM_MULTI_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'TextureTransformMultiTest', 'glTF', 'TextureTransformMultiTest.gltf')
export const SAMPLE_ASSET_TEXTURE_TRANSFORM_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'TextureTransformTest', 'glTF', 'TextureTransformTest.gltf')
export const SAMPLE_ASSET_TOY_CAR = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'ToyCar', 'glTF', 'ToyCar.gltf')
export const SAMPLE_ASSET_TRANSMISSION_ORDER_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'TransmissionOrderTest', 'glTF', 'TransmissionOrderTest.gltf')
export const SAMPLE_ASSET_TRANSMISSION_ROUGHNESS_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'TransmissionRoughnessTest', 'glTF', 'TransmissionRoughnessTest.gltf')
export const SAMPLE_ASSET_TRANSMISSION_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'TransmissionTest', 'glTF', 'TransmissionTest.gltf')
export const SAMPLE_ASSET_TRANSMISSION_THINWALL_TEST_GRID = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'TransmissionThinwallTestGrid', 'glTF', 'TransmissionThinwallTestGrid.gltf')
export const SAMPLE_ASSET_TRIANGLE = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'Triangle', 'glTF', 'Triangle.gltf')
export const SAMPLE_ASSET_TRIANGLE_WITHOUT_INDICES = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'TriangleWithoutIndices', 'glTF', 'TriangleWithoutIndices.gltf')
export const SAMPLE_ASSET_TWO_SIDED_PLANE = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'TwoSidedPlane', 'glTF', 'TwoSidedPlane.gltf')
export const SAMPLE_ASSET_UNICODE_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'Unicode❤♻Test', 'glTF', 'Unicode❤♻Test.gltf')
export const SAMPLE_ASSET_UNLIT_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'UnlitTest', 'glTF', 'UnlitTest.gltf')
export const SAMPLE_ASSET_USD_SHADER_BALL_FOR_GLTF = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'USDShaderBallForGltf', 'glTF', 'USDShaderBallForGltf.gltf')
export const SAMPLE_ASSET_VERTEX_COLOR_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'VertexColorTest', 'glTF', 'VertexColorTest.gltf')
export const SAMPLE_ASSET_VIRTUAL_CITY = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'VirtualCity', 'glTF', 'VirtualCity.gltf')
export const SAMPLE_ASSET_WATER_BOTTLE = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'WaterBottle', 'glTF', 'WaterBottle.gltf')
export const SAMPLE_ASSET_XMP_METADATA_ROUNDED_CUBE = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'XmpMetadataRoundedCube', 'glTF', 'XmpMetadataRoundedCube.gltf')

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

test('committed Khronos glTF Sample Assets Cube fixture loads canonical textured cube', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_CUBE, 'utf8'))
  assert.deepEqual(source.buffers, [{ byteLength: 1800, uri: 'Cube.bin' }])
  assert.deepEqual(source.images, [{ uri: 'Cube_BaseColor.png' }])
  assert.equal(source.materials[0].name, 'Cube')
  assert.equal(source.materials[0].pbrMetallicRoughness.roughnessFactor, 0.079)

  const gltf = await loadGltfFixture(SAMPLE_ASSET_CUBE)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'Khronos Cube sample should load a mesh')
  assert.equal(mesh.name, 'Cube')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 36)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 36)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 36)
  assert.equal(mesh.geometry.index?.count, 36)
  assert.equal(mesh.material.isMeshStandardMaterial, true)
  assert.equal(mesh.material.name, 'Cube')
  assert.equal(mesh.material.roughness, 0.079)
  assert.equal(mesh.material.metalness, 0)
  assert.equal(mesh.material.map?.name, 'Cube_BaseColor.png')
  assert.deepEqual(pngDimensions(mesh.material.map.image), [512, 512])
  assert.equal(mesh.material.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(mesh.material.map.flipY, false)

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.65))
  const light = new THREE.DirectionalLight(0xffffff, 1.4)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)
  const camera = frameSceneCamera(gltf.scene)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [1, 1, 1],
  })

  assert.ok(nonBackgroundRatio(rgba, [255, 255, 255], 3) > 0.5, 'Cube sample should render visible textured cube pixels')
  const mean = meanRgba(rgba)
  assert.ok(mean.r < 230 && mean.g < 230 && mean.b < 230, `Cube texture should darken the white background (${mean.r}, ${mean.g}, ${mean.b})`)
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
