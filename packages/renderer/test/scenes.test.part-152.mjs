import nodeTest, { afterEach } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import path from 'node:path'
import * as THREE from 'three'
import { PMREMGenerator } from 'three'
import * as THREE_WEBGPU from 'three/webgpu'
import { Timer } from 'three/src/core/Timer.js'
import { AnimationClipCreator } from 'three/examples/jsm/animation/AnimationClipCreator.js'
import { CCDIKSolver } from 'three/examples/jsm/animation/CCDIKSolver.js'
import WebGL from 'three/examples/jsm/capabilities/WebGL.js'
import WebGPU from 'three/examples/jsm/capabilities/WebGPU.js'
import { ArcballControls } from 'three/examples/jsm/controls/ArcballControls.js'
import { DragControls } from 'three/examples/jsm/controls/DragControls.js'
import { FirstPersonControls } from 'three/examples/jsm/controls/FirstPersonControls.js'
import { FlyControls } from 'three/examples/jsm/controls/FlyControls.js'
import { MapControls } from 'three/examples/jsm/controls/MapControls.js'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { PointerLockControls } from 'three/examples/jsm/controls/PointerLockControls.js'
import { TrackballControls } from 'three/examples/jsm/controls/TrackballControls.js'
import { TransformControls } from 'three/examples/jsm/controls/TransformControls.js'
import { CSM } from 'three/examples/jsm/csm/CSM.js'
import { CSMFrustum } from 'three/examples/jsm/csm/CSMFrustum.js'
import { CSMHelper } from 'three/examples/jsm/csm/CSMHelper.js'
import { CSMShader } from 'three/examples/jsm/csm/CSMShader.js'
import { CSMShadowNode } from 'three/examples/jsm/csm/CSMShadowNode.js'
import { AnaglyphEffect } from 'three/examples/jsm/effects/AnaglyphEffect.js'
import { AsciiEffect } from 'three/examples/jsm/effects/AsciiEffect.js'
import { OutlineEffect } from 'three/examples/jsm/effects/OutlineEffect.js'
import { ParallaxBarrierEffect } from 'three/examples/jsm/effects/ParallaxBarrierEffect.js'
import { StereoEffect } from 'three/examples/jsm/effects/StereoEffect.js'
import { DRACOExporter } from 'three/examples/jsm/exporters/DRACOExporter.js'
import { EXRExporter, NO_COMPRESSION } from 'three/examples/jsm/exporters/EXRExporter.js'
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js'
import { KTX2Exporter } from 'three/examples/jsm/exporters/KTX2Exporter.js'
import { OBJExporter } from 'three/examples/jsm/exporters/OBJExporter.js'
import { PLYExporter } from 'three/examples/jsm/exporters/PLYExporter.js'
import { STLExporter } from 'three/examples/jsm/exporters/STLExporter.js'
import { USDZExporter } from 'three/examples/jsm/exporters/USDZExporter.js'
import { DebugEnvironment } from 'three/examples/jsm/environments/DebugEnvironment.js'
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js'
import { TrefoilKnot } from 'three/examples/jsm/curves/CurveExtras.js'
import { NURBSCurve } from 'three/examples/jsm/curves/NURBSCurve.js'
import { NURBSSurface } from 'three/examples/jsm/curves/NURBSSurface.js'
import * as NURBSUtils from 'three/examples/jsm/curves/NURBSUtils.js'
import { NURBSVolume } from 'three/examples/jsm/curves/NURBSVolume.js'
import { BoxLineGeometry } from 'three/examples/jsm/geometries/BoxLineGeometry.js'
import { ConvexGeometry } from 'three/examples/jsm/geometries/ConvexGeometry.js'
import { DecalGeometry } from 'three/examples/jsm/geometries/DecalGeometry.js'
import { ParametricGeometry } from 'three/examples/jsm/geometries/ParametricGeometry.js'
import * as ParametricFunctions from 'three/examples/jsm/geometries/ParametricFunctions.js'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'
import { TeapotGeometry } from 'three/examples/jsm/geometries/TeapotGeometry.js'
import { TextGeometry } from 'three/examples/jsm/geometries/TextGeometry.js'
import { LightProbeHelper } from 'three/examples/jsm/helpers/LightProbeHelper.js'
import { OctreeHelper } from 'three/examples/jsm/helpers/OctreeHelper.js'
import { PositionalAudioHelper } from 'three/examples/jsm/helpers/PositionalAudioHelper.js'
import { RectAreaLightHelper } from 'three/examples/jsm/helpers/RectAreaLightHelper.js'
import { TextureHelper } from 'three/examples/jsm/helpers/TextureHelper.js'
import { VertexNormalsHelper } from 'three/examples/jsm/helpers/VertexNormalsHelper.js'
import { VertexTangentsHelper } from 'three/examples/jsm/helpers/VertexTangentsHelper.js'
import { ViewHelper } from 'three/examples/jsm/helpers/ViewHelper.js'
import { HTMLMesh } from 'three/examples/jsm/interactive/HTMLMesh.js'
import { InteractiveGroup } from 'three/examples/jsm/interactive/InteractiveGroup.js'
import { SelectionBox } from 'three/examples/jsm/interactive/SelectionBox.js'
import { SelectionHelper } from 'three/examples/jsm/interactive/SelectionHelper.js'
import { Line2 } from 'three/examples/jsm/lines/Line2.js'
import { LineGeometry } from 'three/examples/jsm/lines/LineGeometry.js'
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js'
import { LineSegments2 } from 'three/examples/jsm/lines/LineSegments2.js'
import { LineSegmentsGeometry } from 'three/examples/jsm/lines/LineSegmentsGeometry.js'
import { Wireframe } from 'three/examples/jsm/lines/Wireframe.js'
import { WireframeGeometry2 } from 'three/examples/jsm/lines/WireframeGeometry2.js'
import { Line2 as WebGPULine2 } from 'three/examples/jsm/lines/webgpu/Line2.js'
import { LineSegments2 as WebGPULineSegments2 } from 'three/examples/jsm/lines/webgpu/LineSegments2.js'
import { Wireframe as WebGPUWireframe } from 'three/examples/jsm/lines/webgpu/Wireframe.js'
import { TiledLighting } from 'three/examples/jsm/lighting/TiledLighting.js'
import { LightProbeGenerator } from 'three/examples/jsm/lights/LightProbeGenerator.js'
import { RectAreaLightTexturesLib } from 'three/examples/jsm/lights/RectAreaLightTexturesLib.js'
import { RectAreaLightUniformsLib } from 'three/examples/jsm/lights/RectAreaLightUniformsLib.js'
import { BVHLoader } from 'three/examples/jsm/loaders/BVHLoader.js'
import { FontLoader } from 'three/examples/jsm/loaders/FontLoader.js'
import { GCodeLoader } from 'three/examples/jsm/loaders/GCodeLoader.js'
import { IESLoader } from 'three/examples/jsm/loaders/IESLoader.js'
import { KTX2Loader } from 'three/examples/jsm/loaders/KTX2Loader.js'
import { LUT3dlLoader } from 'three/examples/jsm/loaders/LUT3dlLoader.js'
import { LUTCubeLoader } from 'three/examples/jsm/loaders/LUTCubeLoader.js'
import { MDDLoader } from 'three/examples/jsm/loaders/MDDLoader.js'
import { MTLLoader } from 'three/examples/jsm/loaders/MTLLoader.js'
import { NRRDLoader } from 'three/examples/jsm/loaders/NRRDLoader.js'
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js'
import { PCDLoader } from 'three/examples/jsm/loaders/PCDLoader.js'
import { PDBLoader } from 'three/examples/jsm/loaders/PDBLoader.js'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'
import { RGBELoader } from 'three/examples/jsm/loaders/RGBELoader.js'
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js'
import { SVGLoader } from 'three/examples/jsm/loaders/SVGLoader.js'
import { TGALoader } from 'three/examples/jsm/loaders/TGALoader.js'
import { TTFLoader } from 'three/examples/jsm/loaders/TTFLoader.js'
import { VTKLoader } from 'three/examples/jsm/loaders/VTKLoader.js'
import { VRMLLoader } from 'three/examples/jsm/loaders/VRMLLoader.js'
import { XYZLoader } from 'three/examples/jsm/loaders/XYZLoader.js'
import { LDrawConditionalLineMaterial } from 'three/examples/jsm/materials/LDrawConditionalLineMaterial.js'
import { Capsule } from 'three/examples/jsm/math/Capsule.js'
import { ColorConverter } from 'three/examples/jsm/math/ColorConverter.js'
import { DisplayP3ColorSpace, LinearDisplayP3ColorSpace, LinearRec2020ColorSpace } from 'three/examples/jsm/math/ColorSpaces.js'
import { ConvexHull } from 'three/examples/jsm/math/ConvexHull.js'
import { ImprovedNoise } from 'three/examples/jsm/math/ImprovedNoise.js'
import { Lut } from 'three/examples/jsm/math/Lut.js'
import { MeshSurfaceSampler } from 'three/examples/jsm/math/MeshSurfaceSampler.js'
import { OBB } from 'three/examples/jsm/math/OBB.js'
import { Octree } from 'three/examples/jsm/math/Octree.js'
import { SimplexNoise } from 'three/examples/jsm/math/SimplexNoise.js'
import { ConvexObjectBreaker } from 'three/examples/jsm/misc/ConvexObjectBreaker.js'
import { GPUComputationRenderer } from 'three/examples/jsm/misc/GPUComputationRenderer.js'
import { Gyroscope } from 'three/examples/jsm/misc/Gyroscope.js'
import { MD2Character } from 'three/examples/jsm/misc/MD2Character.js'
import { MD2CharacterComplex } from 'three/examples/jsm/misc/MD2CharacterComplex.js'
import { MorphAnimMesh } from 'three/examples/jsm/misc/MorphAnimMesh.js'
import { MorphBlendMesh } from 'three/examples/jsm/misc/MorphBlendMesh.js'
import { ProgressiveLightMap } from 'three/examples/jsm/misc/ProgressiveLightMap.js'
import { ProgressiveLightMap as ProgressiveLightMapGPU } from 'three/examples/jsm/misc/ProgressiveLightMapGPU.js'
import { RollerCoasterGeometry, RollerCoasterLiftersGeometry, RollerCoasterShadowGeometry, SkyGeometry, TreesGeometry } from 'three/examples/jsm/misc/RollerCoaster.js'
import { TubePainter } from 'three/examples/jsm/misc/TubePainter.js'
import { Volume } from 'three/examples/jsm/misc/Volume.js'
import { VolumeSlice } from 'three/examples/jsm/misc/VolumeSlice.js'
import { Flow, InstancedFlow } from 'three/examples/jsm/modifiers/CurveModifier.js'
import { Flow as GPUFlow } from 'three/examples/jsm/modifiers/CurveModifierGPU.js'
import { EdgeSplitModifier } from 'three/examples/jsm/modifiers/EdgeSplitModifier.js'
import { SimplifyModifier } from 'three/examples/jsm/modifiers/SimplifyModifier.js'
import { TessellateModifier } from 'three/examples/jsm/modifiers/TessellateModifier.js'
import { GroundedSkybox } from 'three/examples/jsm/objects/GroundedSkybox.js'
import { Lensflare } from 'three/examples/jsm/objects/Lensflare.js'
import { LensflareMesh } from 'three/examples/jsm/objects/LensflareMesh.js'
import { MarchingCubes } from 'three/examples/jsm/objects/MarchingCubes.js'
import { Reflector } from 'three/examples/jsm/objects/Reflector.js'
import { ReflectorForSSRPass } from 'three/examples/jsm/objects/ReflectorForSSRPass.js'
import { Refractor } from 'three/examples/jsm/objects/Refractor.js'
import { ShadowMesh } from 'three/examples/jsm/objects/ShadowMesh.js'
import { Sky } from 'three/examples/jsm/objects/Sky.js'
import { SkyMesh } from 'three/examples/jsm/objects/SkyMesh.js'
import { Water } from 'three/examples/jsm/objects/Water.js'
import { Water as FlowWater } from 'three/examples/jsm/objects/Water2.js'
import { WaterMesh } from 'three/examples/jsm/objects/WaterMesh.js'
import { WaterMesh as FlowWaterMesh } from 'three/examples/jsm/objects/Water2Mesh.js'
import { AfterimagePass } from 'three/examples/jsm/postprocessing/AfterimagePass.js'
import { BloomPass } from 'three/examples/jsm/postprocessing/BloomPass.js'
import { BokehPass } from 'three/examples/jsm/postprocessing/BokehPass.js'
import { ClearPass } from 'three/examples/jsm/postprocessing/ClearPass.js'
import { CubeTexturePass } from 'three/examples/jsm/postprocessing/CubeTexturePass.js'
import { DotScreenPass } from 'three/examples/jsm/postprocessing/DotScreenPass.js'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { FilmPass } from 'three/examples/jsm/postprocessing/FilmPass.js'
import { GlitchPass } from 'three/examples/jsm/postprocessing/GlitchPass.js'
import { GTAOPass } from 'three/examples/jsm/postprocessing/GTAOPass.js'
import { HalftonePass } from 'three/examples/jsm/postprocessing/HalftonePass.js'
import { LUTPass } from 'three/examples/jsm/postprocessing/LUTPass.js'
import { OutlinePass } from 'three/examples/jsm/postprocessing/OutlinePass.js'
import { ClearMaskPass, MaskPass } from 'three/examples/jsm/postprocessing/MaskPass.js'
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js'
import { FullScreenQuad, Pass } from 'three/examples/jsm/postprocessing/Pass.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { RenderPixelatedPass } from 'three/examples/jsm/postprocessing/RenderPixelatedPass.js'
import { RenderTransitionPass } from 'three/examples/jsm/postprocessing/RenderTransitionPass.js'
import { SAOPass } from 'three/examples/jsm/postprocessing/SAOPass.js'
import { SavePass } from 'three/examples/jsm/postprocessing/SavePass.js'
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js'
import { SMAAPass } from 'three/examples/jsm/postprocessing/SMAAPass.js'
import { SSRPass } from 'three/examples/jsm/postprocessing/SSRPass.js'
import { SSAOPass } from 'three/examples/jsm/postprocessing/SSAOPass.js'
import { SSAARenderPass } from 'three/examples/jsm/postprocessing/SSAARenderPass.js'
import { TAARenderPass } from 'three/examples/jsm/postprocessing/TAARenderPass.js'
import { TexturePass } from 'three/examples/jsm/postprocessing/TexturePass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'
import { CSS2DObject, CSS2DRenderer } from 'three/examples/jsm/renderers/CSS2DRenderer.js'
import { CSS3DObject, CSS3DRenderer, CSS3DSprite } from 'three/examples/jsm/renderers/CSS3DRenderer.js'
import { Projector, RenderableFace, RenderableLine, RenderableSprite } from 'three/examples/jsm/renderers/Projector.js'
import { SVGObject, SVGRenderer } from 'three/examples/jsm/renderers/SVGRenderer.js'
import { CopyShader } from 'three/examples/jsm/shaders/CopyShader.js'
import { FlakesTexture } from 'three/examples/jsm/textures/FlakesTexture.js'
import * as TranspilerAST from 'three/examples/jsm/transpiler/AST.js'
import GLSLDecoder from 'three/examples/jsm/transpiler/GLSLDecoder.js'
import ShaderToyDecoder from 'three/examples/jsm/transpiler/ShaderToyDecoder.js'
import TSLEncoder from 'three/examples/jsm/transpiler/TSLEncoder.js'
import Transpiler from 'three/examples/jsm/transpiler/Transpiler.js'
import * as BufferGeometryUtils from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import { frameCorners } from 'three/examples/jsm/utils/CameraUtils.js'
import * as GeometryCompressionUtils from 'three/examples/jsm/utils/GeometryCompressionUtils.js'
import { hilbert2D } from 'three/examples/jsm/utils/GeometryUtils.js'
import { LDrawUtils } from 'three/examples/jsm/utils/LDrawUtils.js'
import * as SceneUtils from 'three/examples/jsm/utils/SceneUtils.js'
import { SceneOptimizer } from 'three/examples/jsm/utils/SceneOptimizer.js'
import * as SkeletonUtils from 'three/examples/jsm/utils/SkeletonUtils.js'
import { radixSort } from 'three/examples/jsm/utils/SortUtils.js'
import { ShadowMapViewer } from 'three/examples/jsm/utils/ShadowMapViewer.js'
import { UVsDebug } from 'three/examples/jsm/utils/UVsDebug.js'
import * as WebGLTextureUtils from 'three/examples/jsm/utils/WebGLTextureUtils.js'
import { WorkerPool } from 'three/examples/jsm/utils/WorkerPool.js'
import { createText } from 'three/examples/jsm/webxr/Text2D.js'
import CommonCubeRenderTarget from 'three/src/renderers/common/CubeRenderTarget.js'
import pkg from '../dist/index.js'
import lightsApi from '../dist/lights.js'
import materialsApi from '../dist/materials.js'
import { assertValidPng, meanRgba, nonBackgroundRatio } from './helpers.mjs'
import { test } from './scenes.test.part-001.mjs'
import { cubeTexture, makeCamera, meanRegion, renderRgba, rgbaTexture, solidTexture } from './scenes.test.part-002.mjs'
test('backgroundBlurriness softens 2D texture backgrounds', () => {
  function renderBackground(blurriness) {
    const texture = rgbaTexture([
      255, 0, 0, 255,
      0, 255, 0, 255,
    ], 2, 1)
    texture.magFilter = THREE.NearestFilter
    texture.minFilter = THREE.NearestFilter

    const scene = new THREE.Scene()
    scene.background = texture
    scene.backgroundBlurriness = blurriness

    const camera = makeCamera()
    return renderRgba(scene, camera, { width: 64, height: 64 })
  }

  const sharp = meanRegion(renderBackground(0), 64, 64, 28, 20, 31, 44)
  const blurred = meanRegion(renderBackground(1), 64, 64, 28, 20, 31, 44)
  assert.ok(sharp.r > sharp.g + 120, `sharp background should sample the red texel (${sharp.r} vs ${sharp.g})`)
  assert.ok(blurred.g > sharp.g + 80, `blurred background should mix in the green texel (${blurred.g} vs ${sharp.g})`)
  assert.ok(sharp.r > blurred.r + 20, `blurred background should soften the red texel (${sharp.r} vs ${blurred.r})`)
})

test('options.backgroundBlurriness overrides scene backgroundBlurriness', () => {
  const texture = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  texture.magFilter = THREE.NearestFilter
  texture.minFilter = THREE.NearestFilter

  const scene = new THREE.Scene()
  scene.background = texture
  scene.backgroundBlurriness = 0
  const sharp = meanRegion(renderRgba(scene, makeCamera(), { width: 64, height: 64 }), 64, 64, 28, 20, 31, 44)
  const blurred = meanRegion(renderRgba(scene, makeCamera(), {
    width: 64,
    height: 64,
    backgroundBlurriness: 1,
  }), 64, 64, 28, 20, 31, 44)

  assert.ok(sharp.r > sharp.g + 120, `scene blurriness 0 should keep the red texel sharp (${sharp.r} vs ${sharp.g})`)
  assert.ok(blurred.g > sharp.g + 80, `options.backgroundBlurriness should soften in the green texel (${blurred.g} vs ${sharp.g})`)
})

test('backgroundBlurriness softens equirectangular and cube texture backgrounds', () => {
  function renderBackground(background, blurriness) {
    const scene = new THREE.Scene()
    scene.background = background
    scene.backgroundBlurriness = blurriness

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 0)
    camera.lookAt(new THREE.Vector3(0, 0, -1))
    return meanRegion(renderRgba(scene, camera, {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 64, 64, 28, 28, 36, 36)
  }

  const equirect = rgbaTexture([
    255, 0, 0, 255,
    255, 0, 0, 255,
    255, 0, 0, 255,
    255, 0, 0, 255,
    0, 255, 0, 255,
    0, 255, 0, 255,
    0, 255, 0, 255,
    0, 255, 0, 255,
  ], 8, 1)
  equirect.mapping = THREE.EquirectangularReflectionMapping
  equirect.magFilter = THREE.NearestFilter
  equirect.minFilter = THREE.NearestFilter

  const cube = cubeTexture([
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0],
    [255, 0, 0],
  ])
  cube.magFilter = THREE.NearestFilter
  cube.minFilter = THREE.NearestFilter

  for (const [name, background] of [['equirect', equirect], ['cube', cube]]) {
    const sharp = renderBackground(background, 0)
    const blurred = renderBackground(background, 1)
    assert.ok(sharp.r > sharp.g + 80, `${name} sharp background should sample red (${sharp.r} vs ${sharp.g})`)
    assert.ok(blurred.g > sharp.g + 30, `${name} blurred background should mix in green (${blurred.g} vs ${sharp.g})`)
    assert.ok(sharp.r > blurred.r + 20, `${name} blurred background should soften red (${sharp.r} vs ${blurred.r})`)
  }
})

test('unsupported scene background rotations fail clearly', () => {
  const cases = [
    ['color backgroundRotation', (scene) => {
      scene.background = new THREE.Color(0, 0, 0)
      scene.backgroundRotation = new THREE.Euler(0, Math.PI / 4, 0)
    }, /scene\.backgroundRotation.*equirectangular or cube texture backgrounds/i],
    ['2D backgroundRotation', (scene) => {
      scene.background = solidTexture(0, 255, 0)
      scene.backgroundRotation = new THREE.Euler(0, Math.PI / 4, 0)
    }, /scene\.backgroundRotation.*equirectangular or cube texture backgrounds/i],
  ]

  for (const [name, setup, pattern] of cases) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    setup(scene)

    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      pattern,
      name,
    )
  }
})

test('background textures apply UV transforms', () => {
  const background = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  background.offset.set(0.5, 0)

  const scene = new THREE.Scene()
  scene.background = background
  const camera = makeCamera()

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  assert.ok(mean.g > mean.r + 40, `background texture offset should shift the sampled texel from red to green (${mean.g} vs ${mean.r})`)
})

test('background textures honor flipY', () => {
  function renderBackground(flipY) {
    const background = rgbaTexture([
      255, 0, 0, 255,
      0, 255, 0, 255,
    ], 1, 2)
    background.magFilter = THREE.NearestFilter
    background.minFilter = THREE.NearestFilter
    background.flipY = flipY

    const scene = new THREE.Scene()
    scene.background = background
    const rgba = renderRgba(scene, makeCamera(), { width: 64, height: 64 })
    return {
      top: meanRegion(rgba, 64, 64, 24, 8, 40, 24),
      bottom: meanRegion(rgba, 64, 64, 24, 40, 40, 56),
    }
  }

  const unflipped = renderBackground(false)
  const flipped = renderBackground(true)
  assert.ok(unflipped.top.g > unflipped.top.r + 80, `unflipped background top should sample green (${unflipped.top.g} vs ${unflipped.top.r})`)
  assert.ok(unflipped.bottom.r > unflipped.bottom.g + 80, `unflipped background bottom should sample red (${unflipped.bottom.r} vs ${unflipped.bottom.g})`)
  assert.ok(flipped.top.r > flipped.top.g + 80, `flipped background top should sample red (${flipped.top.r} vs ${flipped.top.g})`)
  assert.ok(flipped.bottom.g > flipped.bottom.r + 80, `flipped background bottom should sample green (${flipped.bottom.g} vs ${flipped.bottom.r})`)
})

test('background textures honor explicit texture matrices', () => {
  const background = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  background.magFilter = THREE.NearestFilter
  background.minFilter = THREE.NearestFilter
  background.matrixAutoUpdate = false
  background.matrix.set(
    0, 0, 0.25,
    0, 0, 0.5,
    0, 0, 1,
  )

  const scene = new THREE.Scene()
  scene.background = background
  const mean = meanRgba(renderRgba(scene, makeCamera(), { width: 64, height: 64 }))
  assert.ok(mean.r > mean.g + 80, `explicit background matrix should pin sampling to the red texel (${mean.r} vs ${mean.g})`)
})

test('background textures honor horizontal wrap modes', () => {
  function renderWrap(wrapS) {
    const background = rgbaTexture([
      255, 0, 0, 255,
      0, 255, 0, 255,
    ], 2, 1)
    background.magFilter = THREE.NearestFilter
    background.minFilter = THREE.NearestFilter
    background.offset.set(1, 0)
    if (wrapS != null) background.wrapS = wrapS

    const scene = new THREE.Scene()
    scene.background = background
    return meanRegion(renderRgba(scene, makeCamera(), { width: 64, height: 64 }), 64, 64, 8, 20, 24, 44)
  }

  const clamped = renderWrap(undefined)
  const repeated = renderWrap(THREE.RepeatWrapping)
  const mirrored = renderWrap(THREE.MirroredRepeatWrapping)
  assert.ok(clamped.g > clamped.r + 80, `clamped offset should hold the green edge texel (${clamped.g} vs ${clamped.r})`)
  assert.ok(repeated.r > repeated.g + 80, `repeated offset should wrap back to the red texel (${repeated.r} vs ${repeated.g})`)
  assert.ok(mirrored.g > mirrored.r + 80, `mirrored repeat should reflect the offset into the green texel (${mirrored.g} vs ${mirrored.r})`)
})

test('background textures honor vertical wrap modes', () => {
  function renderWrap(wrapT) {
    const background = rgbaTexture([
      255, 0, 0, 255,
      255, 0, 0, 255,
      0, 255, 0, 255,
      0, 255, 0, 255,
    ], 2, 2)
    background.magFilter = THREE.NearestFilter
    background.minFilter = THREE.NearestFilter
    background.offset.set(0, 0.5)
    if (wrapT != null) background.wrapT = wrapT

    const scene = new THREE.Scene()
    scene.background = background
    return meanRgba(renderRgba(scene, makeCamera(), { width: 64, height: 64 }))
  }

  const clamped = renderWrap(undefined)
  const repeated = renderWrap(THREE.RepeatWrapping)
  const mirrored = renderWrap(THREE.MirroredRepeatWrapping)
  assert.ok(clamped.g > clamped.r + 80, `clamped vertical offset should hold the green edge texel (${clamped.g} vs ${clamped.r})`)
  assert.ok(repeated.r > clamped.r + 80, `repeated vertical offset should wrap red texels back into view (${repeated.r} vs ${clamped.r})`)
  assert.ok(repeated.g < clamped.g - 80, `repeated vertical offset should no longer be fully clamped green (${repeated.g} vs ${clamped.g})`)
  assert.ok(mirrored.g > mirrored.r + 80, `mirrored repeat should reflect the vertical offset into green texels (${mirrored.g} vs ${mirrored.r})`)
})

test('background texture anisotropy renders with native sampler settings', () => {
  const background = solidTexture(32, 180, 64)
  background.anisotropy = 4

  const scene = new THREE.Scene()
  scene.background = background
  const mean = meanRgba(renderRgba(scene, makeCamera(), { width: 64, height: 64 }))
  assert.ok(mean.g > mean.r + 80 && mean.g > mean.b + 80, `anisotropic background texture should render green (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('background textures decode sRGB colorSpace before output conversion', () => {
  function renderColorSpace(colorSpace) {
    const background = solidTexture(128, 128, 128)
    background.colorSpace = colorSpace

    const scene = new THREE.Scene()
    scene.background = background
    return meanRgba(renderRgba(scene, makeCamera(), {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }))
  }

  const srgb = renderColorSpace(THREE.SRGBColorSpace)
  const linear = renderColorSpace(THREE.LinearSRGBColorSpace)
  assert.ok(linear.r > srgb.r + 50, `linear background texture should remain brighter than decoded sRGB (${linear.r} vs ${srgb.r})`)
})

test('equirect background textures sample from camera direction', () => {
  const background = rgbaTexture([
    255, 0, 0, 255,
    255, 0, 0, 255,
    255, 0, 0, 255,
    255, 0, 0, 255,
    0, 255, 0, 255,
    0, 255, 0, 255,
    0, 255, 0, 255,
    0, 255, 0, 255,
  ], 8, 1)
  background.mapping = THREE.EquirectangularReflectionMapping
  background.magFilter = THREE.NearestFilter
  background.minFilter = THREE.NearestFilter

  function renderFacing(target) {
    const scene = new THREE.Scene()
    scene.background = background
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 0)
    camera.lookAt(target)
    return meanRegion(renderRgba(scene, camera, {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 64, 64, 28, 28, 36, 36)
  }

  const negativeZ = renderFacing(new THREE.Vector3(0, 0, -1))
  const positiveZ = renderFacing(new THREE.Vector3(0, 0, 1))
  assert.ok(negativeZ.r > negativeZ.g + 80, `-Z view should sample the red equirect half (${negativeZ.r} vs ${negativeZ.g})`)
  assert.ok(positiveZ.g > positiveZ.r + 80, `+Z view should sample the green equirect half (${positiveZ.g} vs ${positiveZ.r})`)
})
