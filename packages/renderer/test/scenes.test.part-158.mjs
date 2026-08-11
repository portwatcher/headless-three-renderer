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
import { makeCamera, meanRegion } from './scenes.test.part-002.mjs'
test('Renderer exposes inert render-list and resource helper objects', async () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 1)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(4, 4),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  ))
  const camera = makeCamera()
  const renderer = new Renderer()
  function ToneMappingNode() {}
  class PointLightNode {}
  class PointLightClass {}

  const object = {}
  assert.equal(renderer.properties.has(object), false)
  const propertyBag = renderer.properties.get(object)
  assert.equal(renderer.properties.has(object), true)
  propertyBag.__webglTexture = 'ignored'
  assert.equal(renderer.properties.get(object).__webglTexture, 'ignored')
  renderer.properties.update(object, 'custom', 42)
  assert.equal(renderer.properties.get(object).custom, 42)
  renderer.properties.remove(object)
  assert.equal(renderer.properties.has(object), false)

  const renderList = renderer.renderLists.get(scene, 0)
  assert.equal(renderer.renderLists.lighting, renderer.lighting)
  assert.equal(renderer.renderLists.get(scene, 0), renderList)
  assert.equal(renderer.renderLists.get(scene, 1) === renderList, false)
  const cameraRenderList = renderer.renderLists.get(scene, camera)
  assert.equal(renderer.renderLists.get(scene, camera), cameraRenderList)
  assert.equal(renderer.renderLists.lists.get([scene, camera]), cameraRenderList)
  assert.notEqual(cameraRenderList, renderList)
  assert.equal(cameraRenderList.scene, scene)
  assert.equal(cameraRenderList.camera, camera)
  assert.equal(cameraRenderList.lightsNode, renderer.lighting.getNode(scene, camera))
  assert.equal(cameraRenderList.begin(), cameraRenderList)
  const renderListLight = { id: 'render-list-light' }
  cameraRenderList.pushBundle({ id: 'bundle' })
  cameraRenderList.pushLight(renderListLight)
  assert.deepEqual(cameraRenderList.bundles, [{ id: 'bundle' }])
  assert.deepEqual(cameraRenderList.lightsArray, [renderListLight])
  cameraRenderList.finish()
  assert.deepEqual(cameraRenderList.lightsNode.getLights(), [renderListLight])
  cameraRenderList.begin()
  assert.deepEqual(cameraRenderList.bundles, [])
  assert.deepEqual(cameraRenderList.lightsArray, [])
  renderList.init()
  renderList.push({ id: 2, renderOrder: 0, occlusionTest: true }, {}, { id: 5, transparent: true }, 0, 2, null, { clip: 1 })
  renderList.unshift(
    { id: 1, renderOrder: 0, isSkinnedMesh: true },
    {},
    { id: 4, transmission: 0.5, side: THREE.DoubleSide },
    0,
    1,
    null,
  )
  assert.equal(renderList.opaque.length, 0)
  assert.equal(renderList.transparent.length, 2)
  assert.equal(renderList.transmissive.length, 1)
  assert.equal(renderList.transparentDoublePass.length, 1)
  assert.equal(renderList.occlusionQueryCount, 1)
  assert.equal(renderList.transparent.find((item) => item.id === 2).clippingContext.clip, 1)
  assert.equal(renderList.transmissive[0].materialVariant, 1)
  renderList.sort()
  assert.deepEqual(renderList.transparent.map((item) => item.id), [2, 1])
  renderList.sort(null, (a, b) => b.z - a.z)
  renderList.finish()
  const renderState = renderer.renderStates.get(scene, 0)
  assert.equal(renderer.renderStates.get(scene, 0), renderState)
  assert.equal(renderer.renderStates.get(scene, 1) === renderState, false)
  renderState.init(camera)
  assert.equal(renderState.state.camera, camera)
  assert.equal(renderState.state.lights.state.version, 0)
  assert.equal(renderState.state.lights.state.hash.directionalLength, -1)
  assert.equal(renderState.state.lights.state.probe.length, 9)
  assert.deepEqual(renderState.state.lights.state.ambient, [0, 0, 0])
  const lightProbe = { id: 1, type: 'PointLight' }
  const shadowLight = { id: 2, castShadow: true }
  renderState.pushLight(lightProbe)
  renderState.pushShadow(shadowLight)
  assert.deepEqual(renderState.state.lightsArray, [lightProbe])
  assert.deepEqual(renderState.state.shadowsArray, [shadowLight])
  assert.equal(renderState.setupLights(), undefined)
  assert.equal(renderState.setupLightsView(camera), undefined)
  renderState.init(camera)
  assert.deepEqual(renderState.state.lightsArray, [])
  assert.deepEqual(renderState.state.shadowsArray, [])
  renderState.state.transmissionRenderTarget[camera.id] = { texture: true }
  assert.deepEqual(renderState.state.transmissionRenderTarget[camera.id], { texture: true })
  const disposableNodeKey = {}
  renderer.nodes.get(disposableNodeKey).value = true
  assert.equal(renderer.nodes.has(disposableNodeKey), true)
  renderer.dispose()
  assert.equal(renderer.nodes.has(disposableNodeKey), false)
  assert.equal(renderer.properties.has(object), false)
  assert.equal(renderer.renderLists.get(scene, 0) === renderList, false)
  assert.equal(renderer.renderLists.lists.get([scene, camera]), undefined)
  assert.equal(renderer.renderLists.get(scene, camera) === cameraRenderList, false)
  assert.equal(renderer.renderStates.get(scene, 0) === renderState, false)

  const rgba = renderer.render(scene, camera, { width: 32, height: 32, format: 'rgba' })
  const mean = meanRegion(rgba, 32, 32, 10, 10, 22, 22)
  assert.ok(mean.r > mean.b + 80, `helper object probes should not alter later rendering (${mean.r} vs ${mean.b})`)

  assert.throws(
    () => renderer.extensions.has(''),
    /Renderer\.extensions\.has name must be a non-empty string/i,
  )
  assert.throws(
    () => renderer.extensions.get(null),
    /Renderer\.extensions\.get name must be a non-empty string/i,
  )
  assert.throws(
    () => renderer.hasFeature(''),
    /Renderer\.hasFeature name must be a non-empty string/i,
  )
  await assert.rejects(
    () => renderer.hasFeatureAsync(1),
    /Renderer\.hasFeature name must be a non-empty string/i,
  )
  assert.throws(
    () => renderer.backend.getContext(),
    /Renderer\.backend\.getContext\(\) is not supported.*WebGL or WebGPU context/i,
  )
  assert.throws(
    () => renderer.backend.hasFeature(''),
    /Renderer\.backend\.hasFeature name must be a non-empty string/i,
  )
  assert.throws(
    () => renderer.backend.get(null),
    /Renderer\.backend\.get object must be an object/i,
  )
  assert.throws(
    () => renderer.backend.set({}, null),
    /Renderer\.backend\.set value must be an object/i,
  )
  assert.throws(
    () => renderer.backend.setScissorTest(1),
    /Renderer\.backend\.setScissorTest value must be a boolean/i,
  )
  assert.throws(
    () => renderer.nodes.get(null),
    /Renderer\.nodes\.get object must be an object/i,
  )
  assert.throws(
    () => { renderer.nodes.nodeFrame = null },
    /Renderer\.nodes\.nodeFrame must be an object/i,
  )
  assert.throws(
    () => renderer.nodes.updateGroup({}),
    /Renderer\.nodes\.updateGroup nodeUniformsGroup\.groupNode must be an object/i,
  )
  assert.throws(
    () => renderer.nodes.getForRender({}),
    /Renderer\.nodes\.getForRender\(\) is not supported.*shader-node builder creation.*custom WGSL fragment path/i,
  )
  assert.throws(
    () => renderer.nodes.getForCompute({ isComputeNode: true }),
    /Renderer\.nodes\.getForCompute\(\) is not supported.*compute shader-node builder creation/i,
  )
  assert.throws(
    () => renderer.nodes._createNodeBuilderState({}),
    /Renderer\.nodes\._createNodeBuilderState\(\) is not supported.*shader-node builder state creation/i,
  )
  assert.throws(
    () => renderer.nodes.updateBefore({}),
    /Renderer\.nodes\.updateBefore\(\) is not supported.*updateBefore lifecycle dispatch/i,
  )
  assert.throws(
    () => renderer.nodes.updateAfter({}),
    /Renderer\.nodes\.updateAfter\(\) is not supported.*updateAfter lifecycle dispatch/i,
  )
  assert.throws(
    () => renderer.nodes.updateForCompute({ isComputeNode: true }),
    /Renderer\.nodes\.updateForCompute\(\) is not supported.*compute shader-node update lifecycle dispatch/i,
  )
  assert.throws(
    () => renderer.nodes.updateForRender({}),
    /Renderer\.nodes\.updateForRender\(\) is not supported.*render shader-node update lifecycle dispatch/i,
  )
  assert.throws(
    () => renderer.nodes.getCacheNode('', {}, () => {}),
    /Renderer\.nodes\.getCacheNode type must be a non-empty string/i,
  )
  assert.throws(
    () => renderer.backend.isOccluded({}, null),
    /Renderer\.isOccluded object must be an object-like value/i,
  )
  for (const method of [
    'clear',
    '_getDefaultRenderPassDescriptor',
    '_getRenderPassDescriptor',
    'beginBundle',
    'finishBundle',
    'addBundle',
    'draw',
    'compute',
    'createProgram',
    '_handleSource',
    '_getShaderErrors',
    '_logProgramError',
    '_completeCompile',
    'createBindings',
    'updateBindings',
    'updateBinding',
    '_setupBindings',
    '_bindUniforms',
    'createRenderPipeline',
    'createComputePipeline',
    'createNodeBuilder',
    'createSampler',
    'createDefaultTexture',
    'createTexture',
    'updateTexture',
    'generateMipmaps',
    'copyTextureToBuffer',
    'copyTextureToTexture',
    'copyFramebufferToTexture',
    'createAttribute',
    '_getVaoKey',
    '_createVao',
    'createIndexAttribute',
    'createStorageAttribute',
    'createIndirectStorageAttribute',
    '_getTransformFeedback',
    'updateAttribute',
    '_setFramebuffer',
  ]) {
    assert.throws(
      () => renderer.backend[method]({}),
      new RegExp(`Renderer\\.backend\\.${method}\\(\\) is not supported.*WebGL\\/WebGPU resource state.*Renderer\\.render\\(\\) or renderToTarget\\(\\)`, 'i'),
    )
  }
  assert.throws(
    () => renderer.library.fromMaterial(null),
    /Renderer\.library\.fromMaterial material must be a material-like object/i,
  )
  assert.throws(
    () => renderer.library.addToneMapping('tone', THREE.LinearToneMapping),
    /Renderer\.library\.addToneMapping toneMappingNode must be a function/i,
  )
  assert.throws(
    () => renderer.library.getToneMappingFunction('linear'),
    /Renderer\.library\.getToneMappingFunction toneMapping must be an integer/i,
  )
  assert.throws(
    () => renderer.library.addMaterial({}, 'MeshBasicMaterial'),
    /Renderer\.library\.addMaterial materialNodeClass must be a constructor function/i,
  )
  assert.throws(
    () => renderer.library.getMaterialNodeClass(''),
    /Renderer\.library\.getMaterialNodeClass materialType must be a non-empty string/i,
  )
  assert.throws(
    () => renderer.library.addLight(PointLightNode, {}),
    /Renderer\.library\.addLight lightClass must be a constructor function/i,
  )
  assert.throws(
    () => renderer.library.getLightNodeClass(null),
    /Renderer\.library\.getLightNodeClass light must be an object/i,
  )
  assert.throws(
    () => renderer.library.addType(ToneMappingNode, {}, renderer.library.materialNodes),
    /Renderer\.library\.addType type must be a non-empty string or integer/i,
  )
  assert.throws(
    () => renderer.library.addClass({}, PointLightClass, renderer.library.lightNodes),
    /Renderer\.library\.addClass nodeClass must be a function/i,
  )
  assert.throws(
    () => renderer.lighting.createNode({}),
    /Renderer\.lighting lights must be an array/i,
  )
  assert.throws(
    () => renderer.lighting.getNode(null, camera),
    /Renderer\.lighting\.getNode scene must be an object/i,
  )
  assert.throws(
    () => renderer.lighting.get(scene),
    /Renderer\.lighting\.get keys must be a non-empty array of objects/i,
  )
  assert.throws(
    () => renderer.lighting.set([scene, null], {}),
    /Renderer\.lighting\.set keys\[1\] must be an object/i,
  )
  assert.throws(
    () => renderer.lighting.delete([]),
    /Renderer\.lighting\.delete keys must be a non-empty array of objects/i,
  )
  await assert.rejects(
    () => renderer.backend.getArrayBufferAsync({ isStorageBufferAttribute: true }),
    /Renderer\.backend\.getArrayBufferAsync\(\) is not supported.*storage-buffer GPU readback.*Renderer\.readRenderTargetPixels\(\)/i,
  )
  await assert.rejects(
    () => renderer.backend.getArrayBufferAsync({}),
    /Renderer\.backend\.getArrayBufferAsync attribute must be a storage buffer attribute-like object/i,
  )
  await assert.rejects(
    () => renderer.backend.resolveTimestampAsync({}, 'render'),
    /Renderer\.backend\.resolveTimestampAsync\(\) is not supported.*timestamp queries/i,
  )
  await assert.rejects(
    () => renderer.backend.resolveTimestampAsync({}, 'frame'),
    /Renderer\.backend\.resolveTimestampAsync type must be "render" or "compute"; received "frame"/i,
  )
  await assert.rejects(
    () => renderer.backend.waitForGPU(),
    /Renderer\.backend\.waitForGPU\(\) is not supported.*GPU task synchronization/i,
  )
  assert.throws(
    () => renderer.hasCompatibility(null),
    /Renderer\.hasCompatibility name must be a non-empty string/i,
  )
  assert.throws(
    () => renderer.isOccluded([]),
    /Renderer\.isOccluded object must be an object-like value/i,
  )
  assert.throws(
    () => renderer.capabilities.getMaxPrecision('ultrap'),
    /Renderer\.capabilities\.getMaxPrecision precision ultrap is not supported/i,
  )
  assert.throws(
    () => renderer.capabilities.textureFormatReadable('rgba'),
    /Renderer\.capabilities\.textureFormatReadable format must be an integer/i,
  )
  assert.throws(
    () => renderer.capabilities.textureTypeReadable(Number.POSITIVE_INFINITY),
    /Renderer\.capabilities\.textureTypeReadable type must be an integer/i,
  )
  assert.throws(
    () => renderer.properties.has(null),
    /Renderer\.properties\.has object must be an object/i,
  )
  assert.throws(
    () => renderer.properties.get(null),
    /Renderer\.properties\.get object must be an object/i,
  )
  assert.throws(
    () => renderer.properties.remove(null),
    /Renderer\.properties\.remove object must be an object/i,
  )
  assert.throws(
    () => renderer.properties.update(null, 'custom', 1),
    /Renderer\.properties\.update object must be an object/i,
  )
  assert.throws(
    () => renderer.properties.update({}, '', 1),
    /Renderer\.properties\.update key must be a non-empty string/i,
  )
  assert.throws(
    () => renderer.renderLists.get(null),
    /Renderer\.renderLists\.get scene must be an object/i,
  )
  assert.throws(
    () => renderer.renderLists.get(scene, -1),
    /Renderer\.renderLists\.get renderCallDepth must be a non-negative integer/i,
  )
  assert.throws(
    () => renderer.renderLists.get(scene, null),
    /Renderer\.renderLists\.get camera must be an object/i,
  )
  assert.throws(
    () => renderer.renderLists.get(scene, 'camera'),
    /Renderer\.renderLists\.get camera must be an object/i,
  )
  assert.throws(
    () => renderList.sort('front-to-back'),
    /Renderer\.renderLists list opaque sort must be a function or null/i,
  )
  assert.throws(
    () => renderer.renderStates.get(null),
    /Renderer\.renderStates\.get scene must be an object/i,
  )
  assert.throws(
    () => renderer.renderStates.get(scene, -1),
    /Renderer\.renderStates\.get renderCallDepth must be a non-negative integer/i,
  )
})
