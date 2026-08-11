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
import { Renderer, renderToTarget, test } from './scenes.test.part-001.mjs'
import { getRenderer, makeCamera, meanRegion, renderRgba } from './scenes.test.part-002.mjs'
test('render options viewport confines draws to an output rectangle', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 1)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(4, 4),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, {
    width: 64,
    height: 64,
    viewport: { x: 32, y: 16, width: 32, height: 32 },
  })
  const inside = meanRegion(rgba, 64, 64, 40, 24, 56, 40)
  const outside = meanRegion(rgba, 64, 64, 8, 24, 24, 40)
  assert.ok(inside.r > inside.b + 80, `viewport region should contain the red mesh (${inside.r} vs ${inside.b})`)
  assert.ok(outside.b > outside.r + 80, `outside viewport should retain blue background (${outside.b} vs ${outside.r})`)
})

test('render options scissor clips draws to an output rectangle', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 1)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(4, 4),
    new THREE.MeshBasicMaterial({ color: 0x00ff00 }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, {
    width: 64,
    height: 64,
    scissor: [16, 16, 32, 32],
  })
  const inside = meanRegion(rgba, 64, 64, 24, 24, 40, 40)
  const outsideLeft = meanRegion(rgba, 64, 64, 4, 24, 12, 40)
  const outsideTop = meanRegion(rgba, 64, 64, 24, 4, 40, 12)
  assert.ok(inside.g > inside.b + 80, `scissor region should contain the green mesh (${inside.g} vs ${inside.b})`)
  assert.ok(outsideLeft.b > outsideLeft.g + 80, `left of scissor should retain blue background (${outsideLeft.b} vs ${outsideLeft.g})`)
  assert.ok(outsideTop.b > outsideTop.g + 80, `above scissor should retain blue background (${outsideTop.b} vs ${outsideTop.g})`)
})

test('render options accept Vector4 viewport and scissor rectangles', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 1)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(4, 4),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, {
    width: 64,
    height: 64,
    viewport: new THREE.Vector4(16, 16, 40, 32),
    scissor: new THREE.Vector4(24, 20, 24, 24),
  })
  const inside = meanRegion(rgba, 64, 64, 30, 26, 42, 38)
  const viewportOutside = meanRegion(rgba, 64, 64, 4, 26, 12, 38)
  const scissorOutside = meanRegion(rgba, 64, 64, 18, 26, 22, 38)
  assert.ok(inside.r > inside.b + 80, `Vector4 viewport/scissor region should contain the red mesh (${inside.r} vs ${inside.b})`)
  assert.ok(viewportOutside.b > viewportOutside.r + 80, `outside Vector4 viewport should retain blue background (${viewportOutside.b} vs ${viewportOutside.r})`)
  assert.ok(scissorOutside.b > scissorOutside.r + 80, `outside Vector4 scissor should retain blue background (${scissorOutside.b} vs ${scissorOutside.r})`)
})

test('Renderer viewport and scissor state apply as render fallbacks', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 1)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(4, 4),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const renderer = new Renderer()
  renderer.setViewport(16, 16, 40, 32)
  renderer.setViewport(16, 16, 40, 32, 0, 1)
  renderer.setScissor(new THREE.Vector4(24, 20, 24, 24))
  renderer.setScissorTest(true)

  assert.deepEqual(renderer.getViewport(), { x: 16, y: 16, width: 40, height: 32 })
  assert.deepEqual(renderer.getCurrentViewport(), { x: 16, y: 16, width: 40, height: 32 })
  assert.deepEqual(renderer.getScissor(), { x: 24, y: 20, width: 24, height: 24 })
  assert.equal(renderer.getScissorTest(), true)
  const viewportTarget = new THREE.Vector4()
  const currentViewportTarget = new THREE.Vector4()
  const scissorTarget = { x: 0, y: 0, width: 0, height: 0 }
  const viewportArray = [0, 0, 0, 0]
  assert.strictEqual(renderer.getViewport(viewportTarget), viewportTarget)
  assert.deepEqual(viewportTarget.toArray(), [16, 16, 40, 32])
  assert.strictEqual(renderer.getCurrentViewport(currentViewportTarget), currentViewportTarget)
  assert.deepEqual(currentViewportTarget.toArray(), [16, 16, 40, 32])
  assert.strictEqual(renderer.getScissor(scissorTarget), scissorTarget)
  assert.deepEqual(scissorTarget, { x: 24, y: 20, width: 24, height: 24, z: 24, w: 24 })
  assert.strictEqual(renderer.getViewport(viewportArray), viewportArray)
  assert.deepEqual(viewportArray, [16, 16, 40, 32])

  const rgba = renderer.render(scene, camera, { width: 64, height: 64, format: 'rgba' })
  const inside = meanRegion(rgba, 64, 64, 30, 26, 42, 38)
  const viewportOutside = meanRegion(rgba, 64, 64, 4, 26, 12, 38)
  const scissorOutside = meanRegion(rgba, 64, 64, 18, 26, 22, 38)
  assert.ok(inside.r > inside.b + 80, `Renderer viewport/scissor state should contain the red mesh (${inside.r} vs ${inside.b})`)
  assert.ok(viewportOutside.b > viewportOutside.r + 80, `outside Renderer viewport should retain blue background (${viewportOutside.b} vs ${viewportOutside.r})`)
  assert.ok(scissorOutside.b > scissorOutside.r + 80, `outside Renderer scissor should retain blue background (${scissorOutside.b} vs ${scissorOutside.r})`)

  const override = renderer.render(scene, camera, {
    width: 64,
    height: 64,
    format: 'rgba',
    viewport: { x: 0, y: 0, width: 24, height: 24 },
    scissor: { x: 0, y: 0, width: 24, height: 24 },
  })
  const optionInside = meanRegion(override, 64, 64, 4, 4, 16, 16)
  const stateInside = meanRegion(override, 64, 64, 30, 26, 42, 38)
  assert.ok(optionInside.r > optionInside.b + 80, `options.viewport should override Renderer viewport state (${optionInside.r} vs ${optionInside.b})`)
  assert.ok(stateInside.b > stateInside.r + 80, `Renderer viewport state should not leak when options override it (${stateInside.b} vs ${stateInside.r})`)

  renderer.setScissorTest(false)
  const unclipped = renderer.render(scene, camera, { width: 64, height: 64, format: 'rgba' })
  const previouslyClipped = meanRegion(unclipped, 64, 64, 18, 26, 22, 38)
  assert.ok(previouslyClipped.r > previouslyClipped.b + 80, `disabled Renderer scissor should stop clipping inside the viewport (${previouslyClipped.r} vs ${previouslyClipped.b})`)

  renderer.setViewport(null)
  renderer.setScissor(null)
  assert.equal(renderer.getViewport(), null)
  assert.equal(renderer.getCurrentViewport(), null)
  assert.equal(renderer.getScissor(), null)
  assert.equal(renderer.getViewport(new THREE.Vector4()), null)
  assert.equal(renderer.getCurrentViewport(new THREE.Vector4()), null)
  assert.equal(renderer.getScissor({}), null)
})

test('invalid viewport and scissor rectangles fail clearly', () => {
  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial({ color: 0xffffff })))
  const camera = makeCamera()
  const renderer = new Renderer()

  assert.throws(
    () => getRenderer().render(scene, camera, { width: 32, height: 32, viewport: [0, 0, 0, 16] }),
    /options\.viewport width and height must be greater than 0/i,
  )
  assert.throws(
    () => getRenderer().render(scene, camera, { width: 32, height: 32, viewport: { x: '0', y: 0, width: 16, height: 16 } }),
    /options\.viewport must contain finite x, y, width, and height values/i,
  )
  assert.throws(
    () => getRenderer().render(scene, camera, { width: 32, height: 32, scissor: { x: 0, y: 0, width: 64, height: 16 } }),
    /options\.scissor must fit inside the render target/i,
  )
  assert.throws(
    () => renderToTarget(scene, camera, { viewport: { x: 0, y: 0, width: Number.NaN, height: 16 } }, { width: 32, height: 32 }),
    /target\.viewport must contain finite x, y, width, and height values/i,
  )
  assert.throws(
    () => renderToTarget(scene, camera, { scissorTest: true, scissor: [0, 0, 16, 0] }, { width: 32, height: 32 }),
    /target\.scissor width and height must be greater than 0/i,
  )
  assert.throws(
    () => renderToTarget(scene, camera, { scissorTest: 'yes', scissor: [0, 0, 16, 16] }, { width: 32, height: 32 }),
    /target\.scissorTest must be a boolean/i,
  )
  assert.throws(
    () => renderer.setViewport(0, 0, 0, 16),
    /Renderer\.setViewport width and height must be greater than 0/i,
  )
  assert.throws(
    () => renderer.setViewport(0, 0, 16, 16, Number.NaN, 1),
    /Renderer\.setViewport minDepth must be a finite number/i,
  )
  assert.throws(
    () => renderer.setViewport(0, 0, 16, 16, 0, 2),
    /Renderer\.setViewport maxDepth must be between 0 and 1/i,
  )
  assert.throws(
    () => renderer.setViewport(0, 0, 16, 16, 0.25, 1),
    /Renderer\.setViewport depth ranges other than 0\.\.1 are not supported/i,
  )
  assert.throws(
    () => renderer.setScissor({ x: '0', y: 0, width: 16, height: 16 }),
    /Renderer\.setScissor must contain finite x, y, width, and height values/i,
  )
  assert.throws(
    () => renderer.setScissorTest('yes'),
    /Renderer\.setScissorTest enabled must be a boolean/i,
  )
  assert.throws(
    () => renderer.setSize('32', 16),
    /Renderer\.setSize width must be a finite number/i,
  )
  assert.throws(
    () => renderer.setSize(32, 16.5),
    /Renderer\.setSize height must be a positive integer/i,
  )
  assert.throws(
    () => renderer.setSize(32, 16, 'yes'),
    /Renderer\.setSize updateStyle must be a boolean/i,
  )
  assert.throws(
    () => renderer.setClearColor('not-a-color'),
    /Renderer\.setClearColor color "not-a-color" is not a supported CSS color string/i,
  )
  assert.throws(
    () => renderer.setClearColor(0x1000000),
    /Renderer\.setClearColor color must be between 0x000000 and 0xffffff/i,
  )
  assert.throws(
    () => renderer.setClearColor(0xffffff, 'opaque'),
    /Renderer\.setClearColor alpha must be a finite number/i,
  )
  assert.throws(
    () => renderer.setClearAlpha(Number.NaN),
    /Renderer\.setClearAlpha alpha must be a finite number/i,
  )
  assert.throws(
    () => renderer.setClearDepth('near'),
    /Renderer\.setClearDepth depth must be a finite number/i,
  )
  assert.throws(
    () => renderer.setClearStencil(1.5),
    /Renderer\.setClearStencil stencil must be a finite integer/i,
  )
  renderer.setViewport(0, 0, 64, 16)
  assert.throws(
    () => renderer.render(scene, camera, { width: 32, height: 32, format: 'rgba' }),
    /Renderer\.viewport must fit inside the render target/i,
  )

  const cubeTarget = new THREE.WebGLCubeRenderTarget(32)
  const cubeCamera = new THREE.CubeCamera(0.01, 100, cubeTarget)
  cubeCamera.activeMipmapLevel = 1
  cubeTarget.viewport = { x: 0, y: 0, width: '32', height: 32 }
  assert.throws(
    () => renderToTarget(scene, cubeCamera, cubeTarget),
    /target\.viewport must contain finite x, y, width, and height values/i,
  )

  cubeTarget.viewport = undefined
  cubeTarget.scissorTest = true
  cubeTarget.scissor = { x: 0, y: 0, width: 64, height: 32 }
  assert.throws(
    () => renderToTarget(scene, cubeCamera, cubeTarget),
    /target\.scissor must fit inside the render target/i,
  )
})
