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
import { assertRgbClose, makeCamera, meanRegion, objectIdBytes } from './scenes.test.part-002.mjs'
import { makeCubeCaptureScene } from './scenes.test.part-010.mjs'
test('EffectComposer TAARenderPass accumulates low-opacity CopyShader samples into the active target', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 1)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(4, 4),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  ))
  const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 10)
  camera.position.z = 2

  const renderer = new Renderer()
  renderer.setSize(32, 32)
  const previousTarget = { texture: {} }
  renderer.setRenderTarget(previousTarget)

  const renderTarget = new THREE.WebGLRenderTarget(32, 32, {
    format: THREE.RGBAFormat,
    type: THREE.UnsignedByteType,
  })
  const composer = new EffectComposer(renderer, renderTarget)
  composer.renderToScreen = false
  const pass = new TAARenderPass(scene, camera)
  pass.sampleLevel = 2
  pass.accumulate = true
  composer.addPass(pass)

  for (let frame = 0; frame < 8; frame += 1) {
    composer.render(0)
  }

  assert.strictEqual(renderer.getRenderTarget(), previousTarget)
  assert.equal(camera.view?.enabled, false)
  assert.equal(pass.accumulateIndex, 32)

  const pixels = Buffer.alloc(32 * 32 * 4)
  renderer.readRenderTargetPixels(composer.readBuffer, 0, 0, 32, 32, pixels)
  const mean = meanRegion(pixels, 32, 32, 10, 10, 22, 22)
  assert.ok(mean.r > 200, `TAARenderPass target should accumulate red CopyShader samples (${mean.r})`)
  assert.ok(mean.r > mean.b + 80, `TAARenderPass target should preserve red scene output (${mean.r} vs ${mean.b})`)
  assert.ok(mean.a > 220, `TAARenderPass target should accumulate alpha (${mean.a})`)
})

test('CubeCamera.update works with Renderer render-target state', async () => {
  const scene = makeCubeCaptureScene()
  const cubeTarget = new THREE.WebGLCubeRenderTarget(32, { generateMipmaps: true })
  const cubeCamera = new THREE.CubeCamera(0.01, 100, cubeTarget)
  const renderer = new Renderer()
  const eventCalls = []
  function listener(event) {
    eventCalls.push([event.type, this === renderer.xr])
  }
  assert.equal(renderer.xr.enabled, false)
  assert.equal(renderer.xr.isPresenting, false)
  assert.equal(renderer.xr.cameraAutoUpdate, true)
  assert.equal(renderer.xr.getSession(), null)
  assert.equal(renderer.xr.getReferenceSpace(), null)
  assert.equal(renderer.xr.getReferenceSpaceType(), 'local-floor')
  assert.equal(renderer.xr.setReferenceSpaceType('local'), undefined)
  assert.equal(renderer.xr.getReferenceSpaceType(), 'local')
  assert.equal(renderer.xr.getBaseLayer(), null)
  assert.equal(renderer.xr.getBinding(), null)
  assert.equal(renderer.xr.getFrame(), null)
  assert.equal(renderer.xr.setFramebufferScaleFactor(0.5), undefined)
  assert.equal(renderer.xr.getEnvironmentBlendMode(), 'opaque')
  assert.equal(renderer.xr.getDepthTexture(), null)
  assert.equal(renderer.xr.hasDepthSensing(), false)
  assert.equal(renderer.xr.getDepthSensingMesh(), null)
  assert.equal(renderer.xr.getCamera(), null)
  assert.equal(renderer.xr.getCameraTexture({ isArrayCamera: true }), null)
  assert.equal(renderer.xr.getFoveation(), undefined)
  assert.equal(renderer.xr.setFoveation(0.75), undefined)
  assert.equal(renderer.xr.getFoveation(), 0.75)
  assert.equal(renderer.xr.updateCamera(makeCamera()), undefined)
  assert.equal(renderer.xr.setAnimationLoop(() => {}), undefined)
  assert.equal(renderer.xr.setAnimationLoop(null), undefined)
  assert.equal(renderer.xr.addEventListener('sessionstart', listener), undefined)
  assert.equal(renderer.xr.hasEventListener('sessionstart', listener), true)
  assert.equal(renderer.xr.dispatchEvent({ type: 'sessionstart' }), undefined)
  assert.equal(renderer.xr.removeEventListener('sessionstart', listener), undefined)
  assert.equal(renderer.xr.hasEventListener('sessionstart', listener), false)
  assert.equal(renderer.xr.dispatchEvent({ type: 'sessionstart' }), undefined)
  assert.deepEqual(eventCalls, [['sessionstart', true]])
  assert.equal(renderer.xr.dispose(), undefined)
  assert.throws(
    () => { renderer.xr.enabled = 'yes' },
    /Renderer\.xr\.enabled must be a boolean/i,
  )
  assert.throws(
    () => { renderer.xr.cameraAutoUpdate = 'yes' },
    /Renderer\.xr\.cameraAutoUpdate must be a boolean/i,
  )
  assert.throws(
    () => renderer.xr.setReferenceSpaceType(''),
    /Renderer\.xr\.setReferenceSpaceType type must be a non-empty string/i,
  )
  assert.throws(
    () => renderer.xr.setReferenceSpace(null),
    /Renderer\.xr\.setReferenceSpace space must be a WebXR reference-space-like object/i,
  )
  assert.throws(
    () => renderer.xr.setReferenceSpace({}),
    /Renderer\.xr\.setReferenceSpace\(\) is not supported.*WebXR runtime/i,
  )
  assert.throws(
    () => renderer.xr.setFramebufferScaleFactor(0),
    /Renderer\.xr\.setFramebufferScaleFactor value must be greater than 0/i,
  )
  assert.throws(
    () => renderer.xr.getController(-1),
    /Renderer\.xr\.getController index must be a non-negative integer/i,
  )
  assert.throws(
    () => renderer.xr.getController(0),
    /Renderer\.xr\.getController\(\) is not supported.*WebXR runtime/i,
  )
  assert.throws(
    () => renderer.xr.getControllerGrip(0),
    /Renderer\.xr\.getControllerGrip\(\) is not supported.*WebXR runtime/i,
  )
  assert.throws(
    () => renderer.xr.getHand(0),
    /Renderer\.xr\.getHand\(\) is not supported.*WebXR runtime/i,
  )
  assert.throws(
    () => renderer.xr.getCameraTexture(null),
    /Renderer\.xr\.getCameraTexture camera must be an XR camera-like object/i,
  )
  assert.throws(
    () => renderer.xr.setFoveation('full'),
    /Renderer\.xr\.setFoveation value must be a finite number/i,
  )
  assert.throws(
    () => renderer.xr.setFoveation(1.5),
    /Renderer\.xr\.setFoveation value must be between 0 and 1/i,
  )
  assert.throws(
    () => renderer.xr.addEventListener('', listener),
    /Renderer\.xr\.addEventListener type must be a non-empty string/i,
  )
  assert.throws(
    () => renderer.xr.removeEventListener('sessionstart', null),
    /Renderer\.xr\.removeEventListener listener must be a function/i,
  )
  assert.throws(
    () => renderer.xr.dispatchEvent({ type: '' }),
    /Renderer\.xr\.dispatchEvent event\.type must be a non-empty string/i,
  )
  assert.throws(
    () => renderer.xr.dispatchEvent(null),
    /Renderer\.xr\.dispatchEvent event must be an event-like object/i,
  )
  assert.throws(
    () => renderer.xr.setAnimationLoop('loop'),
    /Renderer\.xr\.setAnimationLoop callback must be a function or null/i,
  )
  await assert.rejects(
    () => renderer.xr.setSession(null),
    /Renderer\.xr\.setSession session must be a WebXR session-like object/i,
  )
  await assert.rejects(
    () => renderer.xr.setSession({}),
    /Renderer\.xr\.setSession\(\) is not supported.*WebXR runtime/i,
  )
  renderer.xr.enabled = true

  cubeCamera.update(renderer, scene)

  assert.equal(renderer.getRenderTarget(), null)
  assert.equal(renderer.getActiveCubeFace(), 0)
  assert.equal(renderer.getActiveMipmapLevel(), 0)
  assert.equal(renderer.xr.enabled, true)
  assert.equal(cubeTarget.texture.generateMipmaps, true)
  assert.equal(cubeTarget.texture.image.length, 6)
  assert.strictEqual(cubeTarget.texture.source.data, cubeTarget.texture.image)
  assert.ok(cubeTarget.texture.pmremVersion > 0, 'CubeCamera.update should request a PMREM refresh')

  const px = meanRegion(cubeTarget.texture.image[0].data, 32, 32, 12, 12, 20, 20)
  const nx = meanRegion(cubeTarget.texture.image[1].data, 32, 32, 12, 12, 20, 20)
  assert.ok(px.r > px.g + 80 && px.r > px.b + 80, `+X update face should capture red (${px.r}, ${px.g}, ${px.b})`)
  assert.ok(nx.g > nx.r + 60 && nx.g > nx.b + 60, `-X update face should capture green (${nx.r}, ${nx.g}, ${nx.b})`)

  const previousTarget = { width: 8, height: 8, texture: {} }
  renderer.setRenderTarget(previousTarget, 2, 1)
  cubeCamera.activeMipmapLevel = 1
  cubeCamera.update(renderer, scene)

  assert.strictEqual(renderer.getRenderTarget(), previousTarget)
  assert.equal(renderer.getActiveCubeFace(), 2)
  assert.equal(renderer.getActiveMipmapLevel(), 1)

  const mip = cubeTarget.texture.mipmaps[1]
  assert.equal(mip.image.length, 6)
  assert.equal(mip.image[0].width, 16)
  assert.equal(mip.image[0].height, 16)
  const mipPx = meanRegion(mip.image[0].data, 16, 16, 5, 5, 11, 11)
  assert.ok(mipPx.r > mipPx.g + 80 && mipPx.r > mipPx.b + 80, `+X update mip face should capture red (${mipPx.r}, ${mipPx.g}, ${mipPx.b})`)
})

test('CubeCamera object-id target includes reverse lookup metadata', () => {
  const scene = makeCubeCaptureScene()
  const cubeTarget = new THREE.WebGLCubeRenderTarget(32)
  const cubeCamera = new THREE.CubeCamera(0.01, 100, cubeTarget)
  const [positiveX, negativeX] = scene.children

  renderToTarget(scene, cubeCamera, cubeTarget, { renderMode: 'object-id' })

  const positiveXEncoded = positiveX.id + 1
  const negativeXEncoded = negativeX.id + 1
  const px = meanRegion(cubeTarget.texture.image[0].data, 32, 32, 12, 12, 20, 20)
  const nx = meanRegion(cubeTarget.texture.image[1].data, 32, 32, 12, 12, 20, 20)
  assertRgbClose(px, objectIdBytes(positiveXEncoded), '+X CubeCamera object id')
  assertRgbClose(nx, objectIdBytes(negativeXEncoded), '-X CubeCamera object id')
  assert.ok(cubeTarget.objectIdEntries.length >= 2, 'cube object-id target should expose rendered object metadata')
  assert.equal(cubeTarget.objectIdMap[String(positiveXEncoded)].id, positiveX.id)
  assert.equal(cubeTarget.objectIdMap[String(negativeXEncoded)].id, negativeX.id)

  renderToTarget(scene, cubeCamera, cubeTarget)
  assert.equal(cubeTarget.objectIdEntries, undefined)
  assert.equal(cubeTarget.objectIdMap, undefined)
})

test('CubeCamera supports auxiliary MRT-shaped target attachments', () => {
  const scene = makeCubeCaptureScene()
  const cubeTarget = new THREE.WebGLCubeRenderTarget(32)
  cubeTarget.textures = [
    cubeTarget.texture,
    { userData: { headlessThreeRenderer: { renderMode: 'color' } } },
    { userData: { headlessThreeRenderer: { renderMode: 'mask' } } },
    { userData: { headlessThreeRenderer: { renderMode: 'object-id' } } },
    { userData: { headlessThreeRenderer: { renderMode: 'normal' } } },
    { userData: { headlessThreeRenderer: { renderMode: 'depth' } } },
  ]
  const cubeCamera = new THREE.CubeCamera(0.01, 100, cubeTarget)
  const [positiveX, negativeX] = scene.children

  renderToTarget(scene, cubeCamera, cubeTarget, {
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.equal(cubeTarget.textures[0].image.length, 6)
  assert.equal(cubeTarget.textures[1].image.length, 6)
  assert.equal(cubeTarget.textures[2].image.length, 6)
  assert.equal(cubeTarget.textures[3].image.length, 6)
  assert.equal(cubeTarget.textures[4].image.length, 6)
  assert.equal(cubeTarget.textures[5].image.length, 6)

  const colorPx = meanRegion(cubeTarget.textures[0].image[0].data, 32, 32, 12, 12, 20, 20)
  const colorNx = meanRegion(cubeTarget.textures[0].image[1].data, 32, 32, 12, 12, 20, 20)
  assert.ok(colorPx.r > colorPx.g + 80 && colorPx.r > colorPx.b + 80, `primary +X cube attachment should capture red (${colorPx.r}, ${colorPx.g}, ${colorPx.b})`)
  assert.ok(colorNx.g > colorNx.r + 60 && colorNx.g > colorNx.b + 60, `primary -X cube attachment should capture green (${colorNx.r}, ${colorNx.g}, ${colorNx.b})`)

  const colorCopyPx = meanRegion(cubeTarget.textures[1].image[0].data, 32, 32, 12, 12, 20, 20)
  const colorCopyNx = meanRegion(cubeTarget.textures[1].image[1].data, 32, 32, 12, 12, 20, 20)
  assert.ok(colorCopyPx.r > colorCopyPx.g + 80 && colorCopyPx.r > colorCopyPx.b + 80, `secondary +X cube color attachment should capture red (${colorCopyPx.r}, ${colorCopyPx.g}, ${colorCopyPx.b})`)
  assert.ok(colorCopyNx.g > colorCopyNx.r + 60 && colorCopyNx.g > colorCopyNx.b + 60, `secondary -X cube color attachment should capture green (${colorCopyNx.r}, ${colorCopyNx.g}, ${colorCopyNx.b})`)

  const colorCopyReadback = Buffer.alloc(8 * 4 * 4)
  new Renderer().readRenderTargetPixels(cubeTarget, 12, 12, 8, 4, colorCopyReadback, 1, 1)
  const expectedColorCopyReadback = Buffer.alloc(colorCopyReadback.length)
  const colorCopySource = Buffer.from(cubeTarget.textures[1].image[1].data)
  for (let row = 0; row < 4; row += 1) {
    const sourceStart = (((12 + row) * 32) + 12) * 4
    colorCopySource.copy(expectedColorCopyReadback, row * 8 * 4, sourceStart, sourceStart + 8 * 4)
  }
  assert.deepEqual(colorCopyReadback, expectedColorCopyReadback)

  const maskPx = meanRegion(cubeTarget.textures[2].image[0].data, 32, 32, 12, 12, 20, 20)
  const maskNx = meanRegion(cubeTarget.textures[2].image[1].data, 32, 32, 12, 12, 20, 20)
  assert.ok(maskPx.r > 250 && maskNx.r > 250, `cube mask attachment should capture visible faces (${maskPx.r}, ${maskNx.r})`)

  const objectIdPx = meanRegion(cubeTarget.textures[3].image[0].data, 32, 32, 12, 12, 20, 20)
  const objectIdNx = meanRegion(cubeTarget.textures[3].image[1].data, 32, 32, 12, 12, 20, 20)
  const positiveXEncoded = positiveX.id + 1
  const negativeXEncoded = negativeX.id + 1
  assertRgbClose(objectIdPx, objectIdBytes(positiveXEncoded), '+X auxiliary CubeCamera object id')
  assertRgbClose(objectIdNx, objectIdBytes(negativeXEncoded), '-X auxiliary CubeCamera object id')
  assert.ok(cubeTarget.objectIdEntries.length >= 2, 'cube auxiliary object-id target should expose rendered object metadata')
  assert.equal(cubeTarget.objectIdMap[String(positiveXEncoded)].id, positiveX.id)
  assert.equal(cubeTarget.objectIdMap[String(negativeXEncoded)].id, negativeX.id)

  const normalPx = meanRegion(cubeTarget.textures[4].image[0].data, 32, 32, 12, 12, 20, 20)
  const normalNx = meanRegion(cubeTarget.textures[4].image[1].data, 32, 32, 12, 12, 20, 20)
  assert.ok(normalPx.b > 250 && normalNx.b > 250, `cube normal attachment should capture face normals (${normalPx.b}, ${normalNx.b})`)

  const depthPx = meanRegion(cubeTarget.textures[5].image[0].data, 32, 32, 12, 12, 20, 20)
  const depthNx = meanRegion(cubeTarget.textures[5].image[1].data, 32, 32, 12, 12, 20, 20)
  assert.ok(depthPx.r > 0 && depthNx.r > 0, `cube depth attachment should capture visible faces (${depthPx.r}, ${depthNx.r})`)
})
