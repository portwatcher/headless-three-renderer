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
import { Renderer, test } from './scenes.test.part-001.mjs'
import { assertRgbClose, makeCamera, meanRegion, renderRgba, rgbaTexture, solidTexture } from './scenes.test.part-002.mjs'
test('malformed packed CubeUV background texture layouts fail clearly', () => {
  const malformedLayout = /packed PMREM\/CubeUV image height must be divisible by 4|packed PMREM\/CubeUV image must use Three\.js' 3-column by 4-row layout/i
  const scene = new THREE.Scene()
  scene.background = Object.assign(solidTexture(0, 255, 0), { mapping: THREE.CubeUVReflectionMapping })

  assert.throws(
    () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
    malformedLayout,
    'scene.background',
  )

  const optionScene = new THREE.Scene()
  optionScene.background = new THREE.Color(0, 0, 0)
  assert.throws(
    () => renderRgba(optionScene, makeCamera(), {
      width: 64,
      height: 64,
      background: Object.assign(solidTexture(0, 255, 0), { mapping: THREE.CubeUVReflectionMapping }),
    }),
    malformedLayout,
    'options.background',
  )
})

test('render options accept texture backgrounds', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(1, 0, 0)
  scene.backgroundIntensity = 0
  scene.backgroundRotation = new THREE.Euler(0, Math.PI / 4, 0)
  const camera = makeCamera()

  const mean = meanRgba(renderRgba(scene, camera, {
    width: 64,
    height: 64,
    background: solidTexture(0, 0, 255),
  }))
  assert.ok(mean.b > mean.r + 80, `options.background texture should override scene background (${mean.b} vs ${mean.r})`)
})

test('render option texture backgrounds honor their own UV transforms', () => {
  function renderOptionBackground(offset) {
    const scene = new THREE.Scene()
    scene.background = solidTexture(255, 0, 0)
    scene.backgroundIntensity = 0
    scene.backgroundBlurriness = 1
    scene.backgroundRotation = new THREE.Euler(0, Math.PI / 4, 0)

    const background = rgbaTexture([
      255, 0, 0, 255,
      0, 255, 0, 255,
    ], 2, 1)
    background.magFilter = THREE.NearestFilter
    background.minFilter = THREE.NearestFilter
    background.offset.set(offset, 0)

    return meanRgba(renderRgba(scene, makeCamera(), {
      width: 64,
      height: 64,
      background,
    }))
  }

  const unshifted = renderOptionBackground(0)
  const shifted = renderOptionBackground(0.5)
  assert.ok(shifted.g > unshifted.g + 80, `options.background texture offset should increase green coverage (${shifted.g} vs ${unshifted.g})`)
  assert.ok(shifted.r < unshifted.r - 80, `options.background texture offset should reduce red coverage (${shifted.r} vs ${unshifted.r})`)
})

test('render option texture backgrounds honor option intensity and blurriness controls', () => {
  const background = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  background.magFilter = THREE.NearestFilter
  background.minFilter = THREE.NearestFilter

  const scene = new THREE.Scene()
  scene.background = solidTexture(0, 0, 255)
  scene.backgroundIntensity = 0
  scene.backgroundBlurriness = 1

  const options = {
    width: 64,
    height: 64,
    background,
  }

  const sharp = meanRegion(renderRgba(scene, makeCamera(), options), 64, 64, 28, 20, 31, 44)
  const dimmed = meanRegion(renderRgba(scene, makeCamera(), {
    ...options,
    backgroundIntensity: 0.25,
  }), 64, 64, 28, 20, 31, 44)
  const blurred = meanRegion(renderRgba(scene, makeCamera(), {
    ...options,
    backgroundBlurriness: 1,
  }), 64, 64, 28, 20, 31, 44)

  assert.ok(sharp.r > sharp.g + 120, `options.background texture should keep its red texel sharp (${sharp.r} vs ${sharp.g})`)
  assert.ok(sharp.r > 180, `scene.backgroundIntensity should not dim an option texture background (${sharp.r})`)
  assert.ok(dimmed.r < sharp.r - 60, `options.backgroundIntensity should dim the option texture background (${dimmed.r} vs ${sharp.r})`)
  assert.ok(blurred.g > sharp.g + 80, `options.backgroundBlurriness should mix in the green texel (${blurred.g} vs ${sharp.g})`)
  assert.ok(sharp.r > blurred.r + 20, `options.backgroundBlurriness should soften the red texel (${sharp.r} vs ${blurred.r})`)
})

test('CSS color backgrounds render for scene and options', () => {
  const camera = makeCamera()
  const scene = new THREE.Scene()
  scene.background = 'rgb(32, 64, 128)'

  const sceneMean = meanRgba(renderRgba(scene, camera, { width: 32, height: 32 }))
  assertRgbClose(sceneMean, [0x20, 0x40, 0x80], 'scene.background CSS color')

  scene.background = solidTexture(0, 255, 0)
  const optionMean = meanRgba(renderRgba(scene, camera, {
    width: 32,
    height: 32,
    background: '#204080',
  }))
  assertRgbClose(optionMean, [0x20, 0x40, 0x80], 'options.background CSS color')
})

test('render option color backgrounds override scene texture backgrounds', () => {
  const scene = new THREE.Scene()
  scene.background = Object.assign(solidTexture(0, 255, 0), { mapping: THREE.EquirectangularReflectionMapping })
  scene.backgroundIntensity = 0
  scene.backgroundRotation = new THREE.Euler(0, Math.PI / 4, 0)
  const camera = makeCamera()

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64, background: [1, 0, 0] }))
  assert.ok(mean.r > 200, `options.background color should override scene texture background (${mean.r})`)
  assert.ok(mean.g < 30, `options.background color should suppress scene texture background (${mean.g})`)
})

test('render option null background clears scene backgrounds', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(1, 0, 0)
  const camera = makeCamera()

  const sceneBackground = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  scene.backgroundIntensity = 'ignored'
  scene.backgroundRotation = new THREE.Euler(0, Math.PI / 4, 0)
  const cleared = meanRgba(renderRgba(scene, camera, { width: 64, height: 64, background: null }))

  assert.ok(sceneBackground.r > 200, `scene color background should render red (${sceneBackground.r})`)
  assert.ok(cleared.r < sceneBackground.r - 120, `options.background null should clear scene color background (${cleared.r} vs ${sceneBackground.r})`)
  assert.ok(cleared.g > 5 && cleared.b > 5, `cleared background should use renderer default color (${cleared.g}, ${cleared.b})`)

  const textureScene = new THREE.Scene()
  textureScene.background = solidTexture(0, 255, 0)
  const textureBackground = meanRgba(renderRgba(textureScene, camera, { width: 64, height: 64 }))
  textureScene.backgroundIntensity = 'ignored'
  textureScene.backgroundBlurriness = 'ignored'
  const textureCleared = meanRgba(renderRgba(textureScene, camera, { width: 64, height: 64, background: null }))

  assert.ok(textureBackground.g > 180, `scene texture background should render green (${textureBackground.g})`)
  assert.ok(
    textureCleared.g < textureBackground.g - 120,
    `options.background null should clear scene texture background (${textureCleared.g} vs ${textureBackground.g})`,
  )
})

test('Renderer clear color state applies as background fallback', () => {
  const scene = new THREE.Scene()
  const camera = makeCamera()
  const renderer = new Renderer()

  renderer.setClearColor(0x204080, 0.5)
  assert.equal(renderer.getClearAlpha(), 0.5)

  const clearColor = renderer.getClearColor()
  assert.ok(Math.abs(clearColor.r - 0x20 / 255) < 1e-6, `clear red should match hex input (${clearColor.r})`)
  assert.ok(Math.abs(clearColor.g - 0x40 / 255) < 1e-6, `clear green should match hex input (${clearColor.g})`)
  assert.ok(Math.abs(clearColor.b - 0x80 / 255) < 1e-6, `clear blue should match hex input (${clearColor.b})`)

  const colorTarget = new THREE.Color()
  assert.strictEqual(renderer.getClearColor(colorTarget), colorTarget)
  assert.ok(Math.abs(colorTarget.r - 0x20 / 255) < 1e-6, `target clear red should match hex input (${colorTarget.r})`)
  assert.ok(Math.abs(colorTarget.g - 0x40 / 255) < 1e-6, `target clear green should match hex input (${colorTarget.g})`)
  assert.ok(Math.abs(colorTarget.b - 0x80 / 255) < 1e-6, `target clear blue should match hex input (${colorTarget.b})`)

  const clear = meanRgba(renderer.render(scene, camera, { width: 32, height: 32, format: 'rgba' }))
  assertRgbClose(clear, [0x20, 0x40, 0x80], 'Renderer clear color fallback')
  assert.ok(Math.abs(clear.a - 128) <= 1, `Renderer clear alpha fallback should be half opacity (${clear.a})`)

  renderer.setClearColor('rgb(32, 64, 128)', 0.5)
  const cssClearColor = renderer.getClearColor()
  assert.ok(Math.abs(cssClearColor.r - 0x20 / 255) < 1e-3, `CSS clear red should match rgb() input (${cssClearColor.r})`)
  assert.ok(Math.abs(cssClearColor.g - 0x40 / 255) < 1e-3, `CSS clear green should match rgb() input (${cssClearColor.g})`)
  assert.ok(Math.abs(cssClearColor.b - 0x80 / 255) < 1e-3, `CSS clear blue should match rgb() input (${cssClearColor.b})`)
  const cssClear = meanRgba(renderer.render(scene, camera, { width: 32, height: 32, format: 'rgba' }))
  assertRgbClose(cssClear, [0x20, 0x40, 0x80], 'Renderer CSS clear color fallback')
  assert.ok(Math.abs(cssClear.a - 128) <= 1, `Renderer CSS clear alpha fallback should be half opacity (${cssClear.a})`)

  const originalColorManagementEnabled = THREE.ColorManagement.enabled
  try {
    THREE.ColorManagement.enabled = false
    renderer.setClearColor('rgb(32, 64, 128)', 0.5)
    const disabledManagementClearColor = renderer.getClearColor()
    assert.ok(
      Math.abs(disabledManagementClearColor.r - 0x20 / 255) < 1e-3,
      `CSS clear red should ignore external ColorManagement state (${disabledManagementClearColor.r})`,
    )
  } finally {
    THREE.ColorManagement.enabled = originalColorManagementEnabled
  }

  scene.background = new THREE.Color(1, 0, 0)
  const sceneBackground = meanRgba(renderer.render(scene, camera, { width: 32, height: 32, format: 'rgba' }))
  assertRgbClose(sceneBackground, [255, 0, 0], 'scene background should override Renderer clear color')
  assert.ok(Math.abs(sceneBackground.a - 255) <= 1, `scene background alpha should remain opaque (${sceneBackground.a})`)

  renderer.setClearAlpha(0.25)
  const cleared = meanRgba(renderer.render(scene, camera, { width: 32, height: 32, format: 'rgba', background: null }))
  assertRgbClose(cleared, [0x20, 0x40, 0x80], 'options.background null should use Renderer clear color')
  assert.ok(Math.abs(cleared.a - 64) <= 1, `setClearAlpha should update fallback alpha (${cleared.a})`)

  const optionBackground = meanRgba(renderer.render(scene, camera, {
    width: 32,
    height: 32,
    format: 'rgba',
    background: [0, 1, 0, 0.75],
  }))
  assertRgbClose(optionBackground, [0, 255, 0], 'options.background color should override Renderer clear color')
  assert.ok(Math.abs(optionBackground.a - 191) <= 1, `options.background alpha should override Renderer clear alpha (${optionBackground.a})`)
})
