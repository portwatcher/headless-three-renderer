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
import { halfFloatToNumber, meanRegion, meanScalarRegion, renderRgba } from './scenes.test.part-002.mjs'
test('Renderer.setRenderTarget state writes regular targets', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const renderer = new Renderer()
  const target = { texture: {} }
  renderer.setRenderTarget(target)

  const returned = renderer.render(scene, camera, {
    width: 32,
    height: 32,
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })
  assert.equal(returned, target.data)
  assert.strictEqual(renderer.getRenderTarget(), target)
  assert.equal(renderer.getActiveCubeFace(), 0)
  assert.equal(renderer.getActiveMipmapLevel(), 0)
  assert.equal(target.width, 32)
  assert.equal(target.height, 32)
  assert.equal(target.texture.image.data, target.data)
  assert.equal(target.texture.source.data.data, target.data)

  const mean = meanRegion(target.data, 32, 32, 12, 12, 20, 20)
  assert.ok(mean.r > mean.g + 80 && mean.r > mean.b + 80, `active render target should capture red (${mean.r}, ${mean.g}, ${mean.b})`)

  renderer.setRenderTarget(null)
  assert.equal(renderer.getRenderTarget(), null)
})

test('Renderer.setRenderTarget state honors typed single-attachment target arrays', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const renderer = new Renderer()
  const options = { width: 32, height: 32, outputColorSpace: THREE.LinearSRGBColorSpace }
  const center = ((16 * 32) + 16) * 2
  const cases = [
    ['target.texture array', { texture: [{ format: THREE.RGFormat, type: THREE.FloatType }] }, (target) => target.texture[0]],
    ['target.textures array', { textures: [{ format: THREE.RGFormat, type: THREE.FloatType }] }, (target) => target.textures[0]],
    ['single-attachment MRT target', { isWebGLMultipleRenderTargets: true, textures: [{ format: THREE.RGFormat, type: THREE.FloatType }] }, (target) => target.textures[0]],
  ]

  for (const [label, target, colorTexture] of cases) {
    renderer.setRenderTarget(target)
    const returned = renderer.render(scene, camera, options)
    const data = colorTexture(target).image.data

    assert.equal(returned, target.data, `${label} render should return target.data`)
    assert.strictEqual(renderer.getRenderTarget(), target, `${label} should remain the active target`)
    assert.equal(target.width, 32, `${label} should receive render width`)
    assert.equal(target.height, 32, `${label} should receive render height`)
    assert.ok(Buffer.isBuffer(target.data), `${label} top-level target.data should remain RGBA8`)
    assert.ok(data instanceof Float32Array, `${label} should receive Float32Array data`)
    assert.equal(data.length, 32 * 32 * 2, `${label} should receive two channels per pixel`)
    assert.equal(colorTexture(target).source.data.data, data, `${label} source should reference typed data`)
    assert.ok(data[center] > 0.5, `${label} red channel should be normalized (${data[center]})`)
    assert.ok(data[center + 1] < 0.05, `${label} green channel should stay near zero (${data[center + 1]})`)
  }

  renderer.setRenderTarget(null)
  assert.equal(renderer.getRenderTarget(), null)
})

test('renderToTarget and options.target populate depthTexture with normalized RGBA depth', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const near = new THREE.Mesh(
    new THREE.PlaneGeometry(0.9, 1.2),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  )
  near.position.set(-0.7, 0, 1)

  const far = new THREE.Mesh(
    new THREE.PlaneGeometry(0.9, 1.2),
    new THREE.MeshBasicMaterial({ color: 0x0000ff }),
  )
  far.position.set(0.7, 0, -3)
  scene.add(near, far)

  const camera = new THREE.OrthographicCamera(-2, 2, 2, -2, 0.1, 10)
  camera.position.set(0, 0, 5)
  camera.lookAt(0, 0, 0)

  const depthTexture = { source: { data: {} } }
  const target = { texture: {}, depthTexture }
  renderToTarget(scene, camera, target, { width: 64, height: 64 })

  assert.equal(target.data.length, 64 * 64 * 4)
  assert.equal(target.texture.image.data, target.data)
  assert.equal(depthTexture.image.data.length, 64 * 64 * 4)
  assert.notStrictEqual(depthTexture.image.data, target.data)
  assert.equal(depthTexture.source.data.data, depthTexture.image.data)
  assert.equal(depthTexture.source.data.width, 64)
  assert.equal(depthTexture.source.data.height, 64)
  assert.equal(depthTexture.needsUpdate, true)

  const leftDepth = meanRegion(depthTexture.image.data, 64, 64, 18, 26, 26, 38)
  const rightDepth = meanRegion(depthTexture.image.data, 64, 64, 38, 26, 46, 38)
  assert.ok(
    leftDepth.r > rightDepth.r + 80,
    `near depth should be brighter than far depth (${leftDepth.r} vs ${rightDepth.r})`,
  )
  assert.ok(Math.abs(leftDepth.r - leftDepth.g) <= 1, 'depth red and green channels should match')
  assert.ok(Math.abs(leftDepth.r - leftDepth.b) <= 1, 'depth red and blue channels should match')

  const optionsDepthTexture = { source: { data: {} } }
  const optionsTarget = { texture: {}, depthTexture: optionsDepthTexture }
  const returned = renderRgba(scene, camera, { width: 64, height: 64, target: optionsTarget })
  assert.equal(returned, optionsTarget.data, 'options.target should return target.data')
  assert.equal(optionsTarget.texture.image.data, optionsTarget.data)
  assert.equal(optionsDepthTexture.image.data.length, 64 * 64 * 4)
  assert.notStrictEqual(optionsDepthTexture.image.data, optionsTarget.data)
  assert.equal(optionsDepthTexture.source.data.data, optionsDepthTexture.image.data)
  assert.equal(optionsDepthTexture.source.data.width, 64)
  assert.equal(optionsDepthTexture.source.data.height, 64)
  assert.equal(optionsDepthTexture.needsUpdate, true)

  const optionsLeftDepth = meanRegion(optionsDepthTexture.image.data, 64, 64, 18, 26, 26, 38)
  const optionsRightDepth = meanRegion(optionsDepthTexture.image.data, 64, 64, 38, 26, 46, 38)
  assert.ok(
    optionsLeftDepth.r > optionsRightDepth.r + 80,
    `options.target near depth should be brighter than far depth (${optionsLeftDepth.r} vs ${optionsRightDepth.r})`,
  )
})

test('Renderer.setRenderTarget state populates FloatType depthTexture', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const near = new THREE.Mesh(
    new THREE.PlaneGeometry(0.9, 1.2),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  )
  near.position.set(-0.7, 0, 1)

  const far = new THREE.Mesh(
    new THREE.PlaneGeometry(0.9, 1.2),
    new THREE.MeshBasicMaterial({ color: 0x0000ff }),
  )
  far.position.set(0.7, 0, -3)
  scene.add(near, far)

  const camera = new THREE.OrthographicCamera(-2, 2, 2, -2, 0.1, 10)
  camera.position.set(0, 0, 5)
  camera.lookAt(0, 0, 0)

  const renderer = new Renderer()
  const depthTexture = { type: THREE.FloatType, source: { data: {} } }
  const target = { texture: {}, depthTexture }
  renderer.setRenderTarget(target)
  const returned = renderer.render(scene, camera, { width: 64, height: 64 })

  assert.equal(returned, target.data)
  assert.strictEqual(renderer.getRenderTarget(), target)
  assert.equal(target.texture.image.data, target.data)
  assert.ok(depthTexture.image.data instanceof Float32Array, 'FloatType depthTexture should receive Float32Array data')
  assert.equal(depthTexture.image.data.length, 64 * 64)
  assert.equal(depthTexture.source.data.data, depthTexture.image.data)
  assert.equal(depthTexture.source.data.width, 64)
  assert.equal(depthTexture.source.data.height, 64)

  const leftDepth = meanScalarRegion(depthTexture.image.data, 64, 64, 18, 26, 26, 38)
  const rightDepth = meanScalarRegion(depthTexture.image.data, 64, 64, 38, 26, 46, 38)
  assert.ok(leftDepth > rightDepth + 0.3, `active target near float depth should be greater than far depth (${leftDepth} vs ${rightDepth})`)
  assert.ok(leftDepth <= 1 && rightDepth >= 0, `active target float depth values should be normalized (${leftDepth}, ${rightDepth})`)

  renderer.setRenderTarget(null)
  assert.equal(renderer.getRenderTarget(), null)
})

test('Renderer.setRenderTarget state populates scalar, half-float, and packed depth textures', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const near = new THREE.Mesh(
    new THREE.PlaneGeometry(0.9, 1.2),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  )
  near.position.set(-0.7, 0, 1)

  const far = new THREE.Mesh(
    new THREE.PlaneGeometry(0.9, 1.2),
    new THREE.MeshBasicMaterial({ color: 0x0000ff }),
  )
  far.position.set(0.7, 0, -3)
  scene.add(near, far)

  const camera = new THREE.OrthographicCamera(-2, 2, 2, -2, 0.1, 10)
  camera.position.set(0, 0, 5)
  camera.lookAt(0, 0, 0)

  const renderer = new Renderer()
  const cases = [
    {
      label: 'UnsignedByteType',
      depthTexture: { type: THREE.UnsignedByteType, source: { data: {} } },
      assertData(depthTexture) {
        assert.ok(depthTexture.image.data instanceof Uint8Array, 'UnsignedByteType depthTexture should receive Uint8Array data')
        const leftDepth = meanScalarRegion(depthTexture.image.data, 64, 64, 18, 26, 26, 38)
        const rightDepth = meanScalarRegion(depthTexture.image.data, 64, 64, 38, 26, 46, 38)
        assert.ok(leftDepth > rightDepth + 80, `active target near byte depth should be greater than far depth (${leftDepth} vs ${rightDepth})`)
        assert.ok(leftDepth <= 0xff && rightDepth >= 0, `active target byte depth values should be normalized (${leftDepth}, ${rightDepth})`)
      },
    },
    {
      label: 'UnsignedShortType',
      depthTexture: { type: THREE.UnsignedShortType, source: { data: {} } },
      assertData(depthTexture) {
        assert.ok(depthTexture.image.data instanceof Uint16Array, 'UnsignedShortType depthTexture should receive Uint16Array data')
        const leftDepth = meanScalarRegion(depthTexture.image.data, 64, 64, 18, 26, 26, 38)
        const rightDepth = meanScalarRegion(depthTexture.image.data, 64, 64, 38, 26, 46, 38)
        assert.ok(leftDepth > rightDepth + 20000, `active target near ushort depth should be greater than far depth (${leftDepth} vs ${rightDepth})`)
        assert.ok(leftDepth <= 0xffff && rightDepth >= 0, `active target ushort depth values should be normalized (${leftDepth}, ${rightDepth})`)
      },
    },
    {
      label: 'UnsignedIntType',
      depthTexture: { type: THREE.UnsignedIntType, source: { data: {} } },
      assertData(depthTexture) {
        assert.ok(depthTexture.image.data instanceof Uint32Array, 'UnsignedIntType depthTexture should receive Uint32Array data')
        const leftDepth = meanScalarRegion(depthTexture.image.data, 64, 64, 18, 26, 26, 38)
        const rightDepth = meanScalarRegion(depthTexture.image.data, 64, 64, 38, 26, 46, 38)
        assert.ok(leftDepth > rightDepth + 1_000_000_000, `active target near uint depth should be greater than far depth (${leftDepth} vs ${rightDepth})`)
        assert.ok(leftDepth <= 0xffffffff && rightDepth >= 0, `active target uint depth values should be normalized (${leftDepth}, ${rightDepth})`)
      },
    },
    {
      label: 'HalfFloatType',
      depthTexture: { type: THREE.HalfFloatType, source: { data: {} } },
      assertData(depthTexture) {
        assert.ok(depthTexture.image.data instanceof Uint16Array, 'HalfFloatType depthTexture should receive Uint16Array half-float data')
        const leftDepth = halfFloatToNumber(Math.round(meanScalarRegion(depthTexture.image.data, 64, 64, 18, 26, 26, 38)))
        const rightDepth = halfFloatToNumber(Math.round(meanScalarRegion(depthTexture.image.data, 64, 64, 38, 26, 46, 38)))
        assert.ok(leftDepth > rightDepth + 0.3, `active target near half-float depth should be greater than far depth (${leftDepth} vs ${rightDepth})`)
        assert.ok(leftDepth <= 1 && rightDepth >= 0, `active target half-float depth values should be normalized (${leftDepth}, ${rightDepth})`)
      },
    },
    {
      label: 'UnsignedInt248Type',
      depthTexture: { type: THREE.UnsignedInt248Type, format: THREE.DepthStencilFormat, source: { data: {} } },
      assertData(depthTexture) {
        assert.ok(depthTexture.image.data instanceof Uint32Array, 'UnsignedInt248Type depthTexture should receive Uint32Array data')
        for (let i = 0; i < depthTexture.image.data.length; i += 197) {
          assert.equal(depthTexture.image.data[i] & 0xff, 0, `stencil byte should be zero at ${i}`)
        }
        const leftDepth24 = meanScalarRegion(depthTexture.image.data, 64, 64, 18, 26, 26, 38) / 0x100
        const rightDepth24 = meanScalarRegion(depthTexture.image.data, 64, 64, 38, 26, 46, 38) / 0x100
        assert.ok(leftDepth24 > rightDepth24 + 1_000_000, `active target near depth24 should be greater than far depth (${leftDepth24} vs ${rightDepth24})`)
        assert.ok(leftDepth24 <= 0xffffff && rightDepth24 >= 0, `active target depth24 values should be normalized (${leftDepth24}, ${rightDepth24})`)
      },
    },
  ]

  for (const { label, depthTexture, assertData } of cases) {
    const target = { texture: {}, depthTexture }
    renderer.setRenderTarget(target)
    const returned = renderer.render(scene, camera, { width: 64, height: 64 })

    assert.equal(returned, target.data, `${label} active target render should return target.data`)
    assert.strictEqual(renderer.getRenderTarget(), target, `${label} should remain the active target`)
    assert.equal(target.texture.image.data, target.data, `${label} color texture should receive target.data`)
    assert.equal(depthTexture.image.data.length, 64 * 64, `${label} depthTexture should receive scalar data`)
    assert.equal(depthTexture.source.data.data, depthTexture.image.data, `${label} source should reference depth data`)
    assert.equal(depthTexture.source.data.width, 64, `${label} source should receive width`)
    assert.equal(depthTexture.source.data.height, 64, `${label} source should receive height`)
    assertData(depthTexture)
  }

  renderer.setRenderTarget(null)
  assert.equal(renderer.getRenderTarget(), null)
})
