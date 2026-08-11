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
import { assertRgbClose, makeCamera, meanRegion } from './scenes.test.part-002.mjs'
test('Renderer info exposes inert compatibility counters', () => {
  const scene = new THREE.Scene()
  const camera = makeCamera()
  const renderer = new Renderer()

  assert.equal(renderer.info.autoReset, true)
  assert.equal(renderer.info.calls, 0)
  assert.equal(renderer.info.frame, 0)
  assert.deepEqual(renderer.info.memory, { geometries: 0, textures: 0 })
  assert.deepEqual(renderer.info.render, {
    calls: 0,
    frameCalls: 0,
    drawCalls: 0,
    triangles: 0,
    points: 0,
    lines: 0,
    timestamp: 0,
    previousFrameCalls: 0,
    timestampCalls: 0,
    frame: 0,
  })
  assert.deepEqual(renderer.info.compute, {
    calls: 0,
    frameCalls: 0,
    timestamp: 0,
    previousFrameCalls: 0,
    timestampCalls: 0,
  })
  assert.equal(renderer.info.programs, null)

  renderer.info.autoReset = false
  assert.equal(renderer.info.autoReset, false)
  renderer.info.update(6, 0x0004)
  renderer.info.update(4, 0x0001, 2)
  renderer.info.update(3, 0x0003)
  renderer.info.update(3, 0x0002)
  renderer.info.update(5, 0x0000)
  assert.equal(renderer.info.render.calls, 5)
  assert.equal(renderer.info.render.drawCalls, 5)
  assert.equal(renderer.info.render.frameCalls, 0)
  assert.equal(renderer.info.render.triangles, 2)
  assert.equal(renderer.info.render.lines, 9)
  assert.equal(renderer.info.render.points, 5)
  renderer.info.update({ isMesh: true }, 6, 2)
  renderer.info.update({ isSprite: true }, 6)
  renderer.info.update({ isPoints: true }, 2, 3)
  renderer.info.update({ isLineSegments: true }, 4)
  renderer.info.update({ isLine: true }, 4)
  renderer.info.update({ isLineLoop: true }, 4)
  assert.equal(renderer.info.render.calls, 5)
  assert.equal(renderer.info.render.drawCalls, 11)
  assert.equal(renderer.info.render.triangles, 8)
  assert.equal(renderer.info.render.lines, 18)
  assert.equal(renderer.info.render.points, 11)
  renderer.info.render.previousFrameCalls = 2
  renderer.info.updateTimestamp('render', 1.25)
  assert.equal(renderer.info.render.timestamp, 1.25)
  assert.equal(renderer.info.render.timestampCalls, 1)
  renderer.info.updateTimestamp('render', 2.5)
  assert.equal(renderer.info.render.timestamp, 3.75)
  assert.equal(renderer.info.render.timestampCalls, 0)
  renderer.info.updateTimestamp('render', 4)
  assert.equal(renderer.info.render.timestamp, 4)
  assert.equal(renderer.info.render.timestampCalls, 1)
  renderer.info.compute.previousFrameCalls = 1
  renderer.info.updateTimestamp('compute', 7)
  assert.equal(renderer.info.compute.timestamp, 7)
  assert.equal(renderer.info.compute.timestampCalls, 0)
  renderer.info.render.frame = 7
  renderer.info.render.frameCalls = 3
  renderer.info.render.timestamp = 123
  renderer.info.render.timestampCalls = 1
  renderer.info.compute.calls = 4
  renderer.info.compute.frameCalls = 2
  renderer.info.compute.timestamp = 456
  renderer.info.compute.timestampCalls = 1
  renderer.info.calls = 9
  renderer.info.memory.geometries = 1
  renderer.info.memory.textures = 2
  renderer.info.reset()
  assert.equal(renderer.info.render.calls, 0)
  assert.equal(renderer.info.render.drawCalls, 0)
  assert.equal(renderer.info.render.frameCalls, 0)
  assert.equal(renderer.info.render.triangles, 0)
  assert.equal(renderer.info.render.points, 0)
  assert.equal(renderer.info.render.lines, 0)
  assert.equal(renderer.info.render.frame, 7)
  assert.equal(renderer.info.render.timestamp, 123)
  assert.equal(renderer.info.render.previousFrameCalls, 3)
  assert.equal(renderer.info.render.timestampCalls, 1)
  assert.equal(renderer.info.compute.calls, 4)
  assert.equal(renderer.info.compute.frameCalls, 0)
  assert.equal(renderer.info.compute.timestamp, 456)
  assert.equal(renderer.info.compute.previousFrameCalls, 2)
  assert.equal(renderer.info.compute.timestampCalls, 1)
  assert.equal(renderer.info.calls, 9)
  assert.deepEqual(renderer.info.memory, { geometries: 1, textures: 2 })
  renderer.info.dispose()
  assert.equal(renderer.info.calls, 0)
  assert.equal(renderer.info.render.calls, 0)
  assert.equal(renderer.info.compute.calls, 0)
  assert.equal(renderer.info.render.timestamp, 0)
  assert.equal(renderer.info.render.previousFrameCalls, 0)
  assert.equal(renderer.info.render.timestampCalls, 0)
  assert.equal(renderer.info.compute.timestamp, 0)
  assert.equal(renderer.info.compute.previousFrameCalls, 0)
  assert.equal(renderer.info.compute.timestampCalls, 0)
  assert.deepEqual(renderer.info.memory, { geometries: 0, textures: 0 })

  assert.throws(
    () => { renderer.info.autoReset = 'yes' },
    /Renderer\.info\.autoReset must be a boolean/i,
  )
  assert.throws(
    () => renderer.info.update({}, 1),
    /Renderer\.info\.update object type is not supported/i,
  )
  assert.throws(
    () => renderer.info.update(-1, 0x0004),
    /Renderer\.info\.update count must be a non-negative integer/i,
  )
  assert.throws(
    () => renderer.info.update(3, 0x0005),
    /Renderer\.info\.update mode 5 is not supported/i,
  )
  assert.throws(
    () => renderer.info.update(3, 0x0004, Number.NaN),
    /Renderer\.info\.update instanceCount must be a non-negative integer/i,
  )
  assert.throws(
    () => renderer.info.updateTimestamp('frame', 1),
    /Renderer\.info\.updateTimestamp type must be "render" or "compute"; received "frame"/i,
  )
  assert.throws(
    () => renderer.info.updateTimestamp('render', Number.NaN),
    /Renderer\.info\.updateTimestamp time must be a finite number/i,
  )
  assert.throws(
    () => renderer.info.updateTimestamp('render', -1),
    /Renderer\.info\.updateTimestamp time must be non-negative/i,
  )

  renderer.setClearColor(0x204080, 0.5)
  const clear = meanRgba(renderer.render(scene, camera, { width: 32, height: 32, format: 'rgba' }))
  assertRgbClose(clear, [0x20, 0x40, 0x80], 'Renderer info compatibility state should not affect rendering')
  assert.ok(Math.abs(clear.a - 128) <= 1, `Renderer info compatibility state should preserve clear alpha (${clear.a})`)
})

test('Renderer size state applies as render fallback', () => {
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
  assert.equal(renderer.getSize(), null)
  assert.equal(renderer.getSize(new THREE.Vector2()), null)

  renderer.setSize(40, 28)
  assert.deepEqual(renderer.getSize(), { width: 40, height: 28 })

  const vectorTarget = new THREE.Vector2()
  const objectTarget = { width: 0, height: 0 }
  const arrayTarget = [0, 0]
  assert.strictEqual(renderer.getSize(vectorTarget), vectorTarget)
  assert.deepEqual(vectorTarget.toArray(), [40, 28])
  assert.strictEqual(renderer.getSize(objectTarget), objectTarget)
  assert.deepEqual(objectTarget, { width: 40, height: 28 })
  assert.strictEqual(renderer.getSize(arrayTarget), arrayTarget)
  assert.deepEqual(arrayTarget, [40, 28])

  const rgba = renderer.render(scene, camera, { format: 'rgba' })
  assert.equal(rgba.length, 40 * 28 * 4)
  const mean = meanRegion(rgba, 40, 28, 12, 8, 28, 20)
  assert.ok(mean.r > mean.b + 80, `Renderer size fallback should render the red mesh (${mean.r} vs ${mean.b})`)

  const override = renderer.render(scene, camera, { width: 24, height: 16, format: 'rgba' })
  assert.equal(override.length, 24 * 16 * 4)

  const target = renderer.renderToTarget(scene, camera)
  assert.equal(target.width, 40)
  assert.equal(target.height, 28)
  assert.equal(target.data.length, 40 * 28 * 4)

  const sizedTarget = { width: 32, height: 20 }
  renderer.renderToTarget(scene, camera, sizedTarget)
  assert.equal(sizedTarget.width, 32)
  assert.equal(sizedTarget.height, 20)
  assert.equal(sizedTarget.data.length, 32 * 20 * 4)

  const activeTarget = { width: 30, height: 18 }
  renderer.setRenderTarget(activeTarget)
  const targetBuffer = renderer.render(scene, camera, { format: 'rgba' })
  assert.equal(targetBuffer.length, 30 * 18 * 4)
  assert.equal(activeTarget.data.length, 30 * 18 * 4)
  renderer.setRenderTarget(null)

  const cubeImages = Array.from({ length: 6 }, () => ({ width: 16, height: 16 }))
  const cubeTarget = {
    texture: { isCubeTexture: true, image: cubeImages, source: { data: cubeImages } },
  }
  const cubeCamera = new THREE.CubeCamera(0.01, 100, new THREE.WebGLCubeRenderTarget(16))
  cubeCamera.renderTarget = cubeTarget
  const cubeFace = renderer.render(scene, cubeCamera, { format: 'rgba' })
  assert.equal(cubeFace.length, 16 * 16 * 4)
  assert.equal(cubeTarget.width, 16)
  assert.equal(cubeTarget.height, 16)
})

test('Renderer pixel ratio is validated compatibility state', () => {
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
  assert.equal(renderer.getPixelRatio(), 1)
  renderer.setPixelRatio(2)
  assert.equal(renderer.getPixelRatio(), 2)
  assert.equal(renderer.setPixelRatio(undefined), undefined)
  assert.equal(renderer.getPixelRatio(), 2)

  renderer.setSize(24, 16)
  const rgba = renderer.render(scene, camera, { format: 'rgba' })
  assert.equal(rgba.length, 24 * 16 * 4)
  assert.deepEqual(renderer.getSize(), { width: 24, height: 16 })

  const mean = meanRegion(rgba, 24, 16, 7, 5, 17, 11)
  assert.ok(mean.r > mean.b + 80, `Renderer pixel ratio state should preserve output-pixel size fallback (${mean.r} vs ${mean.b})`)

  assert.throws(
    () => renderer.setPixelRatio('2'),
    /Renderer\.setPixelRatio value must be a finite number/i,
  )
  assert.throws(
    () => renderer.setPixelRatio(0),
    /Renderer\.setPixelRatio value must be greater than 0/i,
  )
})

test('Renderer drawing buffer size state applies as render fallback', () => {
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
  assert.equal(renderer.getDrawingBufferSize(), null)
  assert.equal(renderer.getDrawingBufferSize(new THREE.Vector2()), null)

  renderer.setDrawingBufferSize(36, 24, 2)
  assert.equal(renderer.getPixelRatio(), 2)
  assert.deepEqual(renderer.getSize(), { width: 36, height: 24 })
  assert.deepEqual(renderer.getDrawingBufferSize(), { width: 36, height: 24 })

  const vectorTarget = new THREE.Vector2()
  const objectTarget = { width: 0, height: 0 }
  const arrayTarget = [0, 0]
  assert.strictEqual(renderer.getDrawingBufferSize(vectorTarget), vectorTarget)
  assert.deepEqual(vectorTarget.toArray(), [36, 24])
  assert.strictEqual(renderer.getDrawingBufferSize(objectTarget), objectTarget)
  assert.deepEqual(objectTarget, { width: 36, height: 24 })
  assert.strictEqual(renderer.getDrawingBufferSize(arrayTarget), arrayTarget)
  assert.deepEqual(arrayTarget, [36, 24])

  const rgba = renderer.render(scene, camera, { format: 'rgba' })
  assert.equal(rgba.length, 36 * 24 * 4)
  const mean = meanRegion(rgba, 36, 24, 10, 7, 26, 17)
  assert.ok(mean.r > mean.b + 80, `Renderer drawing buffer fallback should render the red mesh (${mean.r} vs ${mean.b})`)

  const override = renderer.render(scene, camera, { width: 20, height: 12, format: 'rgba' })
  assert.equal(override.length, 20 * 12 * 4)

  assert.throws(
    () => renderer.setDrawingBufferSize(0, 24, 1),
    /Renderer\.setDrawingBufferSize width must be a positive integer/i,
  )
  assert.throws(
    () => renderer.setDrawingBufferSize(36, 24, '2'),
    /Renderer\.setDrawingBufferSize pixelRatio value must be a finite number/i,
  )
})
