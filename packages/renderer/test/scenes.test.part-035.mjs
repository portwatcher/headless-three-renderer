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
import { PeppersGhostEffect, Renderer, test } from './scenes.test.part-001.mjs'
import { countRegionPixels, meanRegion, solidTexture } from './scenes.test.part-002.mjs'
test('StereoEffect renders scissored eye cameras without forced matrix updates', () => {
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
  const target = { texture: {} }
  renderer.setRenderTarget(target)
  renderer.autoClear = true

  const effect = new StereoEffect(renderer)
  effect.setSize(64, 32)
  effect.render(scene, camera)

  assert.ok(target.data instanceof Uint8Array, 'StereoEffect should write into the active target')
  assert.equal(target.width, 64)
  assert.equal(target.height, 32)

  const left = meanRegion(target.data, 64, 32, 8, 8, 24, 24)
  const right = meanRegion(target.data, 64, 32, 40, 8, 56, 24)
  assert.ok(left.r > left.b + 80, `left eye should contain the red mesh (${left.r} vs ${left.b})`)
  assert.ok(right.r > right.b + 80, `right eye should contain the red mesh (${right.r} vs ${right.b})`)
  assert.equal(renderer.autoClear, true)
  assert.equal(renderer.getScissorTest(), false)
  assert.deepEqual(renderer.getViewport(), { x: 32, y: 0, width: 32, height: 32 })
  assert.deepEqual(renderer.getScissor(), { x: 32, y: 0, width: 32, height: 32 })
})

test('PeppersGhostEffect renders scissored cameras into the active target', {
  skip: PeppersGhostEffect ? false : 'PeppersGhostEffect is not shipped by this Three.js revision',
}, () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  scene.add(new THREE.AmbientLight(0xffffff, 2))
  scene.add(new THREE.Mesh(
    new THREE.BoxGeometry(1, 1, 1),
    new THREE.MeshBasicMaterial({ color: 0x00ff00 }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 5)
  camera.lookAt(0, 0, 0)

  const renderer = new Renderer()
  const target = { texture: {} }
  renderer.setRenderTarget(target)

  const effect = new PeppersGhostEffect(renderer)
  effect.setSize(96, 96)
  effect.cameraDistance = 5
  effect.render(scene, camera)

  assert.ok(target.data instanceof Uint8Array, 'PeppersGhostEffect should write into the active target')
  assert.equal(target.width, 96)
  assert.equal(target.height, 96)
  assert.ok(nonBackgroundRatio(target.data, [0, 0, 0], 3) > 0.01, 'PeppersGhostEffect should render visible helper panels')
  assert.equal(renderer.getRenderTarget(), target)
  assert.equal(renderer.getScissorTest(), false)
})

test('AnaglyphEffect internal shader pass fails clearly', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const renderer = new Renderer()
  const effect = new AnaglyphEffect(renderer, 32, 32)
  effect.setSize(32, 32)

  try {
    assert.throws(
      () => effect.render(scene, camera),
      /AnaglyphEffect internal ShaderMaterial.*not translated.*StereoEffect.*PeppersGhostEffect/i,
    )
  } finally {
    effect.dispose()
  }
})

test('ParallaxBarrierEffect internal shader pass fails clearly', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0x00ff00 }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const renderer = new Renderer()
  const effect = new ParallaxBarrierEffect(renderer)
  effect.setSize(32, 32)

  try {
    assert.throws(
      () => effect.render(scene, camera),
      /ParallaxBarrierEffect internal ShaderMaterial.*not translated.*StereoEffect.*PeppersGhostEffect/i,
    )
  } finally {
    effect.dispose()
  }
})

test('Postprocessing Pass base state and FullScreenQuad render through renderer state', () => {
  const pass = new Pass()
  assert.equal(pass.isPass, true)
  assert.equal(pass.enabled, true)
  assert.equal(pass.needsSwap, true)
  assert.equal(pass.clear, false)
  assert.equal(pass.renderToScreen, false)
  assert.equal(pass.setSize(16, 16), undefined)
  assert.equal(pass.dispose(), undefined)

  const previousError = console.error
  const errors = []
  try {
    console.error = (message) => errors.push(String(message))
    assert.equal(pass.render(), undefined)
  } finally {
    console.error = previousError
  }
  assert.ok(errors.some((message) => message.includes('THREE.Pass: .render() must be implemented')))

  const renderer = new Renderer()
  const target = { texture: {} }
  renderer.setSize(48, 48)
  renderer.setRenderTarget(target)
  const material = new THREE.MeshBasicMaterial({ color: 0x33ccff, side: THREE.DoubleSide })
  const quad = new FullScreenQuad(material)
  assert.equal(quad.material, material)

  try {
    quad.render(renderer)
    assert.ok(target.data instanceof Uint8Array, 'FullScreenQuad should write into the active render target')
    assert.equal(target.width, 48)
    assert.equal(target.height, 48)
    assert.ok(
      countRegionPixels(target.data, 48, 48, 0, 0, 48, 48, (r, g, b) => b > 120 && g > 100 && r < 130) > 1800,
      'FullScreenQuad should render its material through the renderer scene path',
    )
  } finally {
    material.dispose()
    renderer.dispose()
  }
})

test('Three.js scene helper shader materials fail clearly with helper guidance', () => {
  function makeFlatCamera() {
    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 10)
    camera.position.z = 2
    camera.updateProjectionMatrix()
    camera.updateMatrixWorld()
    return camera
  }

  function makePlaneScene(material = new THREE.MeshBasicMaterial({ color: 0xff0000 })) {
    const scene = new THREE.Scene()
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(1, 1), material))
    return scene
  }

  const cases = [
    [
      'OutlineEffect',
      () => {
        const renderer = new Renderer()
        renderer.setSize(16, 16)
        const effect = new OutlineEffect(renderer)
        effect.autoClear = false
        const scene = makePlaneScene()
        const camera = makeFlatCamera()
        return () => effect.render(scene, camera)
      },
      /OutlineEffect internal outline ShaderMaterial.*not translated.*renderMode.*custom WGSL/i,
    ],
    [
      'Sky',
      () => {
        const renderer = new Renderer()
        const scene = new THREE.Scene()
        scene.add(new Sky())
        const camera = makeFlatCamera()
        return () => renderer.render(scene, camera, { width: 16, height: 16, format: 'rgba' })
      },
      /(?:Sky internal SkyShader ShaderMaterial.*not translated.*scene\.background.*custom WGSL|ShaderMaterial "SkyShader" is not supported directly.*fragmentWgsl)/i,
    ],
    [
      'Water',
      () => {
        const renderer = new Renderer()
        const scene = new THREE.Scene()
        scene.add(new Water(new THREE.PlaneGeometry(1, 1), {
          textureWidth: 16,
          textureHeight: 16,
          waterNormals: solidTexture(128, 128, 255),
        }))
        const camera = makeFlatCamera()
        return () => renderer.render(scene, camera, { width: 16, height: 16, format: 'rgba' })
      },
      /Water internal MirrorShader ShaderMaterial.*not translated.*reflection targets.*custom WGSL/i,
    ],
    [
      'Water2',
      () => {
        const renderer = new Renderer()
        const scene = new THREE.Scene()
        scene.add(new FlowWater(new THREE.PlaneGeometry(1, 1), {
          textureWidth: 16,
          textureHeight: 16,
          normalMap0: solidTexture(128, 128, 255),
          normalMap1: solidTexture(128, 128, 255),
        }))
        const camera = makeFlatCamera()
        return () => renderer.render(scene, camera, { width: 16, height: 16, format: 'rgba' })
      },
      /Water2 internal WaterShader ShaderMaterial.*not translated.*reflection\/refraction targets.*custom WGSL/i,
    ],
    [
      'Reflector',
      () => {
        const renderer = new Renderer()
        const scene = new THREE.Scene()
        scene.add(new Reflector(new THREE.PlaneGeometry(1, 1), {
          textureWidth: 16,
          textureHeight: 16,
          multisample: 0,
        }))
        const camera = makeFlatCamera()
        return () => renderer.render(scene, camera, { width: 16, height: 16, format: 'rgba' })
      },
      /Reflector internal ReflectorShader ShaderMaterial.*not translated.*target prepass.*custom WGSL/i,
    ],
    [
      'Refractor',
      () => {
        const renderer = new Renderer()
        const scene = new THREE.Scene()
        scene.add(new Refractor(new THREE.PlaneGeometry(1, 1), {
          textureWidth: 16,
          textureHeight: 16,
          multisample: 0,
        }))
        const camera = makeFlatCamera()
        return () => renderer.render(scene, camera, { width: 16, height: 16, format: 'rgba' })
      },
      /Refractor internal RefractorShader ShaderMaterial.*not translated.*target prepass.*custom WGSL/i,
    ],
    [
      'ReflectorForSSRPass',
      () => {
        const renderer = new Renderer()
        const scene = new THREE.Scene()
        scene.add(new ReflectorForSSRPass(new THREE.PlaneGeometry(1, 1), {
          textureWidth: 16,
          textureHeight: 16,
          resolution: new THREE.Vector2(16, 16),
        }))
        const camera = makeFlatCamera()
        return () => renderer.render(scene, camera, { width: 16, height: 16, format: 'rgba' })
      },
      /ReflectorForSSRPass internal ReflectorShader ShaderMaterial.*not translated.*custom WGSL.*SSR reflector/i,
    ],
  ]

  for (const [name, createRender, pattern] of cases) {
    assert.throws(
      createRender(),
      pattern,
      `${name} should fail with helper-specific guidance`,
    )
  }
})
