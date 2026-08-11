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
import { renderRgba } from './scenes.test.part-002.mjs'
test('examples WebXR planes and estimated light helpers update renderable scene state', async () => {
  const { XREstimatedLight } = await import('three/examples/jsm/webxr/XREstimatedLight.js')
  const { XRPlanes } = await import('three/examples/jsm/webxr/XRPlanes.js')

  const renderer = new Renderer()
  const planes = new XRPlanes(renderer)
  let planeChanges = 0
  planes.addEventListener('planeschanged', () => {
    planeChanges += 1
  })
  const detectedPlane = {
    planeSpace: { id: 'plane-space' },
    polygon: [
      { x: -0.45, z: -0.25 },
      { x: 0.45, z: -0.25 },
      { x: 0.45, z: 0.25 },
      { x: -0.45, z: 0.25 },
    ],
  }
  const planeMatrix = new THREE.Matrix4().makeTranslation(0, 0, -1).toArray()
  renderer.xr.dispatchEvent({
    type: 'planesdetected',
    data: {
      detectedPlanes: new Set([detectedPlane]),
      getPose(planeSpace, referenceSpace) {
        assert.equal(planeSpace, detectedPlane.planeSpace)
        assert.equal(referenceSpace, null)
        return { transform: { matrix: planeMatrix } }
      },
    },
  })
  assert.equal(planeChanges, 1)
  assert.equal(planes.children.length, 1)
  const planeMesh = planes.children[0]
  assert.equal(planeMesh.isMesh, true)
  assert.equal(planeMesh.geometry.parameters.width, 0.9)
  assert.equal(planeMesh.geometry.parameters.height, 0.01)
  assert.equal(planeMesh.geometry.parameters.depth, 0.5)

  const planeScene = new THREE.Scene()
  planeScene.background = new THREE.Color(0x000000)
  planeScene.add(planes)
  const planeCamera = new THREE.PerspectiveCamera(45, 1, 0.01, 10)
  planeCamera.position.set(0, 0.8, 1.8)
  planeCamera.lookAt(0, 0, -1)
  const planeRgba = renderRgba(planeScene, planeCamera, { width: 64, height: 64 })
  assert.ok(
    nonBackgroundRatio(planeRgba, [0, 0, 0], 3) > 0.01,
    'XRPlanes should create renderable built-in mesh plane geometry',
  )

  renderer.xr.dispatchEvent({
    type: 'planesdetected',
    data: {
      detectedPlanes: new Set(),
      getPose() {
        throw new Error('removed planes should not request poses')
      },
    },
  })
  assert.equal(planeChanges, 2)
  assert.equal(planes.children.length, 0)
  renderer.dispose()

  const listeners = new Map()
  const frameCallbacks = []
  const lightProbeHandle = { id: 'light-probe' }
  let requestedLightProbe = null
  const session = {
    preferredReflectionFormat: 'srgba8',
    requestLightProbe(init) {
      requestedLightProbe = init
      return Promise.resolve(lightProbeHandle)
    },
    requestAnimationFrame(callback) {
      frameCallbacks.push(callback)
    },
  }
  const xr = {
    addEventListener(type, listener) {
      if (!listeners.has(type)) listeners.set(type, [])
      listeners.get(type).push(listener)
    },
    getSession() {
      return session
    },
    dispatch(type) {
      for (const listener of listeners.get(type) ?? []) listener({ type })
    },
  }
  const estimatedLight = new XREstimatedLight({ xr }, false)
  let estimationStarts = 0
  let estimationEnds = 0
  estimatedLight.addEventListener('estimationstart', () => {
    estimationStarts += 1
  })
  estimatedLight.addEventListener('estimationend', () => {
    estimationEnds += 1
  })
  assert.equal(estimatedLight.lightProbe.intensity, 0)
  assert.equal(estimatedLight.directionalLight.intensity, 0)
  assert.equal(estimatedLight.environment, null)

  xr.dispatch('sessionstart')
  await Promise.resolve()
  await Promise.resolve()
  assert.deepEqual(requestedLightProbe, { reflectionFormat: 'srgba8' })
  assert.equal(frameCallbacks.length, 1)

  const sphericalHarmonics = Array.from({ length: 27 }, (_, index) => (index === 0 ? 0.5 : 0.01))
  frameCallbacks[0](0, {
    session,
    getLightEstimate(probe) {
      assert.equal(probe, lightProbeHandle)
      return {
        sphericalHarmonicsCoefficients: sphericalHarmonics,
        primaryLightIntensity: { x: 3, y: 1.5, z: 0.75 },
        primaryLightDirection: new THREE.Vector3(0, -1, 0),
      }
    },
  })
  assert.equal(frameCallbacks.length, 2)
  assert.equal(estimationStarts, 1)
  assert.equal(estimatedLight.lightProbe.intensity, 1)
  assert.equal(estimatedLight.directionalLight.intensity, 3)
  assert.ok(Math.abs(estimatedLight.directionalLight.color.r - 1) < 1e-6)
  assert.ok(Math.abs(estimatedLight.directionalLight.color.g - 0.5) < 1e-6)
  assert.ok(Math.abs(estimatedLight.directionalLight.color.b - 0.25) < 1e-6)
  assert.equal(estimatedLight.directionalLight.position.y, -1)

  xr.dispatch('sessionend')
  assert.equal(estimationEnds, 1)
  estimatedLight.dispose()
  assert.equal(estimatedLight.lightProbe, null)
  assert.equal(estimatedLight.directionalLight, null)
})

test('examples Addons barrel imports in Node and exposes covered helper modules', async () => {
  const Addons = await import('three/examples/jsm/Addons.js')

  assert.ok(Object.keys(Addons).length > 250)
  assert.equal(Addons.WebGL, WebGL)
  assert.equal(Addons.AnimationClipCreator, AnimationClipCreator)
  assert.equal(Addons.CCDIKSolver, CCDIKSolver)
  assert.equal(Addons.CSS2DRenderer, CSS2DRenderer)
  assert.equal(Addons.CSS3DRenderer, CSS3DRenderer)
  assert.equal(Addons.Projector, Projector)
  assert.equal(Addons.SVGRenderer, SVGRenderer)
  assert.equal(Addons.Pass, Pass)
  assert.equal(Addons.FullScreenQuad, FullScreenQuad)
  assert.equal(Addons.FlakesTexture, FlakesTexture)
  assert.equal(Addons.WorkerPool, WorkerPool)
})

test('examples transpiler utilities parse GLSL and emit TSL source', () => {
  const program = new TranspilerAST.Program()
  const variable = new TranspilerAST.VariableDeclaration('float', 'value', new TranspilerAST.Number('1.0'))
  const ternary = new TranspilerAST.Ternary(
    new TranspilerAST.Operator('>', new TranspilerAST.Accessor('value'), new TranspilerAST.Number('0.0')),
    new TranspilerAST.String('positive'),
    new TranspilerAST.String('zero'),
  )
  const accessorElements = new TranspilerAST.AccessorElements(new TranspilerAST.Accessor('coord'), [
    new TranspilerAST.StaticElement(new TranspilerAST.Accessor('x')),
    new TranspilerAST.DynamicElement(new TranspilerAST.Number('0', 'int')),
  ])
  const loop = new TranspilerAST.For(
    new TranspilerAST.VariableDeclaration('int', 'i', new TranspilerAST.Number('0', 'int')),
    new TranspilerAST.Operator('<', new TranspilerAST.Accessor('i'), new TranspilerAST.Number('2', 'int')),
    new TranspilerAST.Unary('++', new TranspilerAST.Accessor('i'), true),
  )
  const func = new TranspilerAST.FunctionDeclaration('float', 'manual', [
    new TranspilerAST.FunctionParameter('float', 'input', 'in'),
  ])
  func.body.push(new TranspilerAST.Return(new TranspilerAST.FunctionCall('float', [new TranspilerAST.Accessor('input')])))
  program.body.push(new TranspilerAST.Uniform('float', 'amount'), new TranspilerAST.Varying('vec2', 'vUv'), variable, ternary, accessorElements, loop, func)

  assert.equal(program.isProgram, true)
  assert.equal(variable.isVariableDeclaration, true)
  assert.equal(ternary.isTernary, true)
  assert.equal(accessorElements.isAccessorElements, true)
  assert.equal(loop.isFor, true)
  assert.equal(func.isFunctionDeclaration, true)

  const decoder = new GLSLDecoder().addPolyfill('customValue', 'float customValue = 2.0;')
  const ast = decoder.parse(`
    uniform float amount;
    varying vec2 vUv;
    float shade(inout float value) {
      value += amount;
      if (value > 1.0) {
        value = inversesqrt(value);
      } else {
        value = value * customValue;
      }
      return value;
    }
  `)
  const shade = ast.body.find((node) => node.isFunctionDeclaration && node.name === 'shade')
  assert.equal(ast.isProgram, true)
  assert.ok(ast.body.some((node) => node.isUniform && node.name === 'amount'))
  assert.ok(ast.body.some((node) => node.isVarying && node.name === 'vUv'))
  assert.equal(shade.params[0].qualifier, 'inout')
  assert.equal(shade.params[0].immutable, false)
  assert.ok(shade.body.some((node) => node.isConditional))

  const encodableAst = new TranspilerAST.Program()
  encodableAst.body.push(...ast.body.filter((node) => !node.isVarying))
  const encoded = new TSLEncoder().emit(encodableAst)
  assert.match(encoded, /Three\.js Transpiler/)
  assert.match(encoded, /import \{[^}]*uniform[^}]*Fn/)
  assert.match(encoded, /const customValue = float\( 2\.0 \)/)
  assert.match(encoded, /inverseSqrt/)
  assert.match(encoded, /If\(/)
  assert.match(encoded, /return value/)

  const transpiled = new Transpiler(new GLSLDecoder(), new TSLEncoder()).parse('float halfValue(float value) { return value * 0.5; }')
  assert.match(transpiled, /const halfValue/)
  assert.match(transpiled, /return value\.mul\( 0\.5 \)/)

  const shaderToy = new Transpiler(new ShaderToyDecoder(), new TSLEncoder()).parse(`
    void mainImage(out vec4 fragColor, in vec2 fragCoord) {
      fragColor = vec4(iTime / iResolution.x);
    }
  `)
  assert.match(shaderToy, /const mainImage/)
  assert.match(shaderToy, /const fragColor = (?:vec4\(\)\.toVar\(\)|property\( 'vec4' \))/)
  assert.match(shaderToy, /return fragColor/)
  assert.match(shaderToy, /screenSize/)
  assert.match(shaderToy, /time/)
})

test('Projector produces CPU render data for supported scene objects', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)

  const meshGeometry = new THREE.PlaneGeometry(0.8, 0.8)
  const meshMaterial = new THREE.MeshBasicMaterial({ color: 0xff3344, side: THREE.DoubleSide })
  const mesh = new THREE.Mesh(meshGeometry, meshMaterial)
  mesh.position.x = -0.4

  const lineGeometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-0.2, -0.6, 0),
    new THREE.Vector3(0.6, -0.6, 0),
  ])
  const lineMaterial = new THREE.LineBasicMaterial({ color: 0x33ff66 })
  const line = new THREE.LineSegments(lineGeometry, lineMaterial)

  const pointsGeometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0.45, 0.35, 0),
  ])
  const pointsMaterial = new THREE.PointsMaterial({ color: 0x4488ff, size: 0.25 })
  const points = new THREE.Points(pointsGeometry, pointsMaterial)

  const spriteMaterial = new THREE.SpriteMaterial({ color: 0xffff44 })
  const sprite = new THREE.Sprite(spriteMaterial)
  sprite.position.set(0.2, 0.15, 0)
  sprite.scale.setScalar(0.35)

  const light = new THREE.DirectionalLight(0xffffff, 1)
  scene.add(mesh, line, points, sprite, light)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10)
  camera.position.z = 3
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  try {
    const data = new Projector().projectScene(scene, camera, true, true)
    const faces = data.elements.filter((element) => element instanceof RenderableFace)
    const lines = data.elements.filter((element) => element instanceof RenderableLine)
    const sprites = data.elements.filter((element) => element instanceof RenderableSprite)

    assert.ok(data.objects.some((entry) => entry.object === mesh), 'Projector should include mesh objects')
    assert.ok(data.objects.some((entry) => entry.object === line), 'Projector should include line objects')
    assert.equal(data.lights[0], light)
    assert.equal(faces.length, 2)
    assert.equal(lines.length, 1)
    assert.equal(sprites.length, 2)
    assert.equal(faces[0].material, meshMaterial)
    assert.equal(lines[0].material, lineMaterial)
    assert.equal(sprites.some((entry) => entry.object === points), true)
    assert.equal(sprites.some((entry) => entry.object === sprite), true)
    assert.ok(data.elements.every((element) => Number.isFinite(element.z)), 'projected element z values should be finite')

    const rgba = renderRgba(scene, camera, { width: 64, height: 64 })
    assert.ok(
      nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.04,
      'Projector source scene objects should still render through the normal renderer path',
    )
  } finally {
    meshGeometry.dispose()
    meshMaterial.dispose()
    lineGeometry.dispose()
    lineMaterial.dispose()
    pointsGeometry.dispose()
    pointsMaterial.dispose()
    spriteMaterial.dispose()
  }
})
