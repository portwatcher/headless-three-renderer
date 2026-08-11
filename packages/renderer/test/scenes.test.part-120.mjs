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
import { countRegionPixels, meanRegion, renderRgba, rgbaTexture } from './scenes.test.part-002.mjs'
test('renderToTarget and options.target use target viewport and scissor fields', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 1)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(4, 4),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  function makeTarget() {
    const depthTexture = {}
    return {
      depthTexture,
      target: {
        texture: {},
        depthTexture,
        viewport: new THREE.Vector4(16, 16, 40, 32),
        scissor: new THREE.Vector4(24, 20, 24, 24),
        scissorTest: true,
      },
    }
  }

  function assertTargetClip(label, target, depthTexture) {
    const inside = meanRegion(target.data, 64, 64, 30, 26, 42, 38)
    const viewportOutside = meanRegion(target.data, 64, 64, 4, 26, 12, 38)
    const scissorOutside = meanRegion(target.data, 64, 64, 18, 26, 22, 38)
    assert.ok(inside.r > inside.b + 80, `${label} target viewport/scissor region should contain the red mesh (${inside.r} vs ${inside.b})`)
    assert.ok(viewportOutside.b > viewportOutside.r + 80, `${label} outside target viewport should retain blue background (${viewportOutside.b} vs ${viewportOutside.r})`)
    assert.ok(scissorOutside.b > scissorOutside.r + 80, `${label} outside target scissor should retain blue background (${scissorOutside.b} vs ${scissorOutside.r})`)

    const depthInside = meanRegion(depthTexture.image.data, 64, 64, 30, 26, 42, 38)
    const depthOutside = meanRegion(depthTexture.image.data, 64, 64, 18, 26, 22, 38)
    assert.ok(depthInside.r > 0, `${label} target viewport/scissor depth should contain visible geometry (${depthInside.r})`)
    assert.ok(depthOutside.r < 2, `${label} outside target scissor depth should keep background depth (${depthOutside.r})`)
  }

  const direct = makeTarget()
  renderToTarget(scene, camera, direct.target, { width: 64, height: 64 })
  assertTargetClip('renderToTarget', direct.target, direct.depthTexture)

  const options = makeTarget()
  const returned = renderRgba(scene, camera, { width: 64, height: 64, target: options.target })
  assert.equal(returned, options.target.data, 'options.target should return target.data')
  assertTargetClip('options.target', options.target, options.depthTexture)

  const rendererTarget = makeTarget()
  const renderer = new Renderer()
  renderer.setRenderTarget(rendererTarget.target)
  const rendererReturned = renderer.render(scene, camera, { width: 64, height: 64 })
  assert.equal(rendererReturned, rendererTarget.target.data, 'Renderer.setRenderTarget should return target.data')
  assertTargetClip('Renderer.setRenderTarget', rendererTarget.target, rendererTarget.depthTexture)
  renderer.setRenderTarget(null)

  const disabledScissor = makeTarget()
  disabledScissor.target.scissorTest = false
  renderToTarget(scene, camera, disabledScissor.target, { width: 64, height: 64 })
  const previouslyScissored = meanRegion(disabledScissor.target.data, 64, 64, 18, 26, 22, 38)
  const viewportOutside = meanRegion(disabledScissor.target.data, 64, 64, 4, 26, 12, 38)
  const depthPreviouslyScissored = meanRegion(disabledScissor.depthTexture.image.data, 64, 64, 18, 26, 22, 38)
  assert.ok(previouslyScissored.r > previouslyScissored.b + 80, `disabled target scissor should render inside the viewport (${previouslyScissored.r} vs ${previouslyScissored.b})`)
  assert.ok(viewportOutside.b > viewportOutside.r + 80, `disabled target scissor should still preserve the viewport boundary (${viewportOutside.b} vs ${viewportOutside.r})`)
  assert.ok(depthPreviouslyScissored.r > 0, `disabled target scissor should write depth inside the viewport (${depthPreviouslyScissored.r})`)
})

test('renderToTarget depthTexture preserves alphaMap cutouts', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const alphaMap = rgbaTexture([
    255, 0, 255, 255,
    255, 255, 255, 255,
  ], 2, 1)
  alphaMap.magFilter = THREE.NearestFilter
  alphaMap.minFilter = THREE.NearestFilter
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(3.2, 2.4),
    new THREE.MeshBasicMaterial({
      color: 0xffffff,
      alphaMap,
      alphaTest: 0.5,
    }),
  ))

  const camera = new THREE.OrthographicCamera(-2, 2, 2, -2, 0.1, 10)
  camera.position.set(0, 0, 5)
  camera.lookAt(0, 0, 0)

  const depthTexture = {}
  renderToTarget(scene, camera, { texture: {}, depthTexture }, { width: 64, height: 64 })

  const discarded = meanRegion(depthTexture.image.data, 64, 64, 14, 26, 24, 38)
  const visible = meanRegion(depthTexture.image.data, 64, 64, 40, 26, 50, 38)
  assert.ok(discarded.r < 2, `alphaMap cutout should keep background depth (${discarded.r})`)
  assert.ok(visible.r > 80, `opaque alphaMap region should write visible mesh depth (${visible.r})`)
})

test('renderToTarget depthTexture preserves base texture alpha cutouts', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const map = rgbaTexture([
    255, 255, 255, 0,
    255, 255, 255, 255,
  ], 2, 1)
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(3.2, 2.4),
    new THREE.MeshBasicMaterial({
      color: 0xffffff,
      map,
      alphaTest: 0.5,
    }),
  ))

  const camera = new THREE.OrthographicCamera(-2, 2, 2, -2, 0.1, 10)
  camera.position.set(0, 0, 5)
  camera.lookAt(0, 0, 0)

  const depthTexture = {}
  renderToTarget(scene, camera, { texture: {}, depthTexture }, { width: 64, height: 64 })

  const discarded = meanRegion(depthTexture.image.data, 64, 64, 14, 26, 24, 38)
  const visible = meanRegion(depthTexture.image.data, 64, 64, 40, 26, 50, 38)
  assert.ok(discarded.r < 2, `base texture cutout should keep background depth (${discarded.r})`)
  assert.ok(visible.r > 80, `opaque base texture region should write visible mesh depth (${visible.r})`)
})

test('renderToTarget depthTexture preserves alphaHash cutouts', () => {
  function renderAlphaHashDepth(alphaHash) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({
        alphaHash,
        color: 0xffffff,
        opacity: alphaHash ? 0.35 : 1,
      }),
    ))

    const camera = new THREE.OrthographicCamera(-2, 2, 2, -2, 0.1, 10)
    camera.position.set(0, 0, 5)
    camera.lookAt(0, 0, 0)

    const depthTexture = {}
    renderToTarget(scene, camera, { texture: {}, depthTexture }, { width: 64, height: 64 })
    return depthTexture.image.data
  }

  const opaque = renderAlphaHashDepth(false)
  const hashed = renderAlphaHashDepth(true)
  const depthPixel = (r) => r > 2
  const opaquePixels = countRegionPixels(opaque, 64, 64, 20, 20, 44, 44, depthPixel)
  const hashedPixels = countRegionPixels(hashed, 64, 64, 20, 20, 44, 44, depthPixel)

  assert.ok(opaquePixels > 520, `opaque target depth should fill the sampled region (${opaquePixels})`)
  assert.ok(hashedPixels > 40, `alphaHash target depth should retain some depth pixels (${hashedPixels})`)
  assert.ok(hashedPixels < opaquePixels - 120, `alphaHash target depth should discard depth pixels (${hashedPixels} vs ${opaquePixels})`)
})

test('renderToTarget depthTexture honors transparent default depthWrite', () => {
  function renderTransparentDepth(depthWrite) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const material = new THREE.MeshBasicMaterial({
      color: 0xff0000,
      opacity: 0.5,
      transparent: true,
    })
    if (depthWrite !== undefined) material.depthWrite = depthWrite
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    const depthTexture = {}
    renderToTarget(scene, camera, { texture: {}, depthTexture }, { width: 64, height: 64 })
    return meanRegion(depthTexture.image.data, 64, 64, 24, 24, 40, 40)
  }

  const defaultDepth = renderTransparentDepth(undefined)
  const disabledDepth = renderTransparentDepth(false)

  assert.ok(defaultDepth.r > disabledDepth.r + 0.5, `transparent default depthWrite should populate target depth (${defaultDepth.r} vs ${disabledDepth.r})`)
  assert.ok(disabledDepth.r < 0.5, `transparent depthWrite=false should leave target depth clear (${disabledDepth.r})`)
})
