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
import { makeCamera } from './scenes.test.part-002.mjs'
test('Renderer exposes inert WebGLRenderer helper objects', async () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 1)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(4, 4),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  ))
  const camera = makeCamera()
  const renderer = new Renderer()

  assert.equal(renderer.isRenderer, true)
  assert.equal(renderer.isWebGLRenderer, true)
  assert.equal(renderer.isWebGPURenderer, false)
  assert.equal(renderer.initialized, true)
  assert.equal(renderer.hasInitialized(), true)
  assert.equal(renderer.alpha, false)
  assert.equal(renderer.depth, true)
  assert.equal(renderer.stencil, false)
  assert.equal(renderer.logarithmicDepthBuffer, false)
  assert.equal(renderer.currentToneMapping, THREE.ACESFilmicToneMapping)
  assert.equal(renderer.needsFrameBufferTarget, false)
  renderer.currentToneMapping = THREE.NoToneMapping
  assert.equal(renderer.toneMapping, THREE.NoToneMapping)
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  assert.equal(renderer.currentToneMapping, THREE.ACESFilmicToneMapping)
  assert.equal(renderer.highPrecision, false)
  renderer.highPrecision = false
  assert.equal(renderer.highPrecision, false)
  assert.equal(renderer.samples, 0)
  assert.equal(renderer.currentSamples, 0)
  assert.equal(renderer.isOutputTarget, true)
  renderer.setRenderTarget({ texture: {}, sampleCount: 4 })
  assert.equal(renderer.samples, 0)
  assert.equal(renderer.currentSamples, 4)
  assert.equal(renderer.isOutputTarget, false)
  renderer.setRenderTarget(null)
  assert.equal(renderer.currentSamples, 0)
  assert.equal(renderer.isOutputTarget, true)
  assert.equal(renderer.coordinateSystem, THREE.WebGLCoordinateSystem)
  assert.throws(
    () => { renderer.coordinateSystem = THREE.WebGPUCoordinateSystem },
    /coordinateSystem/i,
  )
  assert.throws(
    () => { renderer.highPrecision = true },
    /Renderer\.highPrecision = true is not supported.*shader-node state/i,
  )
  assert.throws(
    () => { renderer.highPrecision = 1 },
    /Renderer\.highPrecision must be a boolean/i,
  )
  assert.throws(
    () => { renderer.currentToneMapping = 'aces' },
    /Renderer\.toneMapping must be a Three\.js tone mapping constant/i,
  )
  assert.equal(renderer.capabilities.isWebGL2, false)
  assert.equal(renderer.capabilities.drawBuffers, false)
  assert.equal(renderer.capabilities.precision, 'highp')
  assert.equal(renderer.capabilities.logarithmicDepthBuffer, false)
  assert.equal(renderer.capabilities.reversedDepthBuffer, false)
  assert.equal(renderer.capabilities.reverseDepthBuffer, false)
  assert.equal(renderer.capabilities.vertexTextures, false)
  assert.equal(renderer.capabilities.floatFragmentTextures, false)
  assert.equal(renderer.capabilities.floatVertexTextures, false)
  assert.equal(renderer.capabilities.maxTextures, 0)
  assert.equal(renderer.capabilities.maxVertexTextures, 0)
  assert.equal(renderer.capabilities.maxTextureSize, 0)
  assert.equal(renderer.capabilities.maxCubemapSize, 0)
  assert.equal(renderer.capabilities.maxAttributes, 0)
  assert.equal(renderer.capabilities.maxVertexUniforms, 0)
  assert.equal(renderer.capabilities.maxVaryings, 0)
  assert.equal(renderer.capabilities.maxFragmentUniforms, 0)
  assert.equal(renderer.capabilities.maxDrawBuffers, 1)
  assert.equal(renderer.capabilities.maxColorAttachments, 1)
  assert.equal(renderer.capabilities.maxSamples, 4)
  assert.equal(renderer.capabilities.samples, 0)
  assert.equal(renderer.getMaxAnisotropy(), 0)
  assert.equal(renderer.hasFeature('timestamp-query'), false)
  assert.equal(await renderer.hasFeatureAsync('timestamp-query'), false)
  assert.equal(renderer.hasCompatibility('float32-filterable'), false)
  assert.equal(renderer.isOccluded(scene), false)
  assert.equal(renderer.capabilities.getMaxAnisotropy(), 0)
  assert.equal(renderer.capabilities.getMaxPrecision('highp'), 'highp')
  assert.equal(renderer.capabilities.getMaxPrecision('mediump'), 'mediump')
  assert.equal(renderer.capabilities.getMaxPrecision('lowp'), 'lowp')
  assert.equal(renderer.capabilities.textureFormatReadable(THREE.RGBAFormat), true)
  assert.equal(renderer.capabilities.textureFormatReadable(THREE.DepthFormat), false)
  assert.equal(renderer.capabilities.textureTypeReadable(THREE.UnsignedByteType), true)
  assert.equal(renderer.capabilities.textureTypeReadable(THREE.UnsignedInt248Type), false)
  assert.equal(renderer.backend.renderer, renderer)
  assert.equal(renderer.backend.isWebGLBackend, false)
  assert.equal(renderer.backend.isWebGPUBackend, false)
  assert.equal(renderer.backend.coordinateSystem, THREE.WebGLCoordinateSystem)
  assert.deepEqual(renderer.backend.parameters, {})
  assert.equal(renderer.backend.domElement, renderer.domElement)
  assert.equal(renderer.backend.getDomElement(), renderer.domElement)
  assert.equal(renderer.backend.getMaxAnisotropy(), 0)
  assert.equal(renderer.backend.hasFeature('timestamp-query'), false)
  assert.equal(await renderer.backend.hasFeatureAsync('timestamp-query'), false)
  assert.equal(await renderer.backend.init(renderer), undefined)
  assert.equal(renderer.backend.beginRender({ id: 'render' }), undefined)
  assert.equal(renderer.backend.finishRender({ id: 'render' }), undefined)
  assert.equal(renderer.backend.beginCompute({ id: 'compute' }), undefined)
  assert.equal(renderer.backend.finishCompute({ id: 'compute' }), undefined)
  assert.equal(renderer.backend.updateSize(), undefined)
  assert.equal(renderer.backend.updateViewport({ viewport: true }), undefined)
  assert.equal(await renderer.backend.resolveOccludedAsync({ id: 'render' }), undefined)
  assert.equal(renderer.backend.initTimestampQuery({ id: 'render' }, { type: 'timestamp' }), undefined)
  assert.equal(renderer.backend.prepareTimestampBuffer({ id: 'render' }, { type: 'encoder' }), undefined)
  assert.equal(renderer.backend.isOccluded({}, scene), false)
  assert.equal(renderer.backend.isOccluded(scene), false)
  renderer.setClearColor(0x204080, 0.5)
  const backendClearColor = renderer.backend.getClearColor()
  assert.ok(Math.abs(backendClearColor.r - 0x20 / 255) < 1e-6, `backend clear red should match renderer state (${backendClearColor.r})`)
  assert.ok(Math.abs(backendClearColor.g - 0x40 / 255) < 1e-6, `backend clear green should match renderer state (${backendClearColor.g})`)
  assert.ok(Math.abs(backendClearColor.b - 0x80 / 255) < 1e-6, `backend clear blue should match renderer state (${backendClearColor.b})`)
  assert.equal(backendClearColor.a, 0.5)
  const backendClearTarget = {}
  assert.strictEqual(backendClearColor.getRGB(backendClearTarget), backendClearTarget)
  assert.ok(Math.abs(backendClearTarget.r - 0x20 / 255) < 1e-6, `backend getRGB red should match renderer state (${backendClearTarget.r})`)
  assert.equal(backendClearTarget.a, 0.5)
  renderer.backend.setScissorTest(true)
  assert.equal(renderer.getScissorTest(), true)
  renderer.backend.setScissorTest(false)
  assert.equal(renderer.getScissorTest(), false)
  assert.equal(renderer.backend.getDrawingBufferSize(), null)
  const backendSizeTarget = new THREE.Vector2()
  assert.equal(renderer.backend.getDrawingBufferSize(backendSizeTarget), null)
  renderer.setSize(20, 10)
  assert.strictEqual(renderer.backend.getDrawingBufferSize(backendSizeTarget), backendSizeTarget)
  assert.deepEqual(backendSizeTarget.toArray(), [20, 10])
  const backendKey = {}
  assert.equal(renderer.backend.has(backendKey), false)
  const backendData = renderer.backend.get(backendKey)
  backendData.ready = true
  assert.deepEqual(renderer.backend.get(backendKey), { ready: true })
  assert.equal(renderer.backend.has(backendKey), true)
  renderer.backend.set(backendKey, { slot: 3 })
  assert.deepEqual(renderer.backend.get(backendKey), { slot: 3 })
  renderer.backend.delete(backendKey)
  assert.equal(renderer.backend.has(backendKey), false)
  renderer.backend.set(backendKey, { reset: false })
  renderer.backend.dispose()
  assert.equal(renderer.backend.has(backendKey), false)
  assert.equal(renderer.backend.destroyProgram({}), undefined)
  assert.equal(renderer.backend.destroySampler({}), undefined)
  assert.equal(renderer.backend.destroyTexture({}), undefined)
  assert.equal(renderer.backend.destroyAttribute({}), undefined)
  assert.equal(renderer.backend.needsRenderUpdate({}), false)
  assert.equal(renderer.backend.getRenderCacheKey({}), 'headless-three-renderer')
  assert.equal(renderer.nodes.modelViewMatrix, null)
  assert.equal(renderer.nodes.modelNormalViewMatrix, null)
  assert.equal(renderer.nodes.renderer, renderer)
  assert.equal(renderer.nodes.backend, renderer.backend)
  const modelViewMatrixNode = { isNode: true }
  renderer.nodes.modelViewMatrix = modelViewMatrixNode
  assert.equal(renderer.nodes.modelViewMatrix, modelViewMatrixNode)
  renderer.nodes.modelViewMatrix = null
  const nodeKey = {}
  assert.equal(renderer.nodes.has(nodeKey), false)
  const nodeData = renderer.nodes.get(nodeKey)
  nodeData.ready = true
  assert.deepEqual(renderer.nodes.get(nodeKey), { ready: true })
  assert.equal(renderer.nodes.has(nodeKey), true)
  assert.deepEqual(renderer.nodes.delete(nodeKey), { ready: true })
  assert.equal(renderer.nodes.has(nodeKey), false)
  assert.equal(renderer.nodes.delete(nodeKey), null)
  const renderGroup = { groupNode: { name: 'render' } }
  assert.equal(renderer.nodes.updateGroup(renderGroup), true)
  assert.equal(renderer.nodes.updateGroup(renderGroup), false)
  renderer.nodes.nodeFrame.renderId = 1
  assert.equal(renderer.nodes.updateGroup(renderGroup), true)
  assert.equal(renderer.nodes.getForRenderCacheKey({ initialCacheKey: 'cache-1' }), 'cache-1')
  let cacheNodeCalls = 0
  const cacheNodeObject = {}
  const cachedNode = renderer.nodes.getCacheNode('background', cacheNodeObject, () => ({ id: ++cacheNodeCalls }))
  assert.equal(renderer.nodes.getCacheNode('background', cacheNodeObject, () => ({ id: ++cacheNodeCalls })), cachedNode)
  const forcedCachedNode = renderer.nodes.getCacheNode('background', cacheNodeObject, () => ({ id: ++cacheNodeCalls }), true)
  assert.notEqual(forcedCachedNode, cachedNode)
  assert.equal(cacheNodeCalls, 2)
  const frameMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff })
  const nodeFrame = renderer.nodes.getNodeFrame(renderer, scene, scene.children[0], camera, frameMaterial)
  assert.equal(nodeFrame.renderer, renderer)
  assert.equal(nodeFrame.scene, scene)
  assert.equal(nodeFrame.object, scene.children[0])
  assert.equal(nodeFrame.camera, camera)
  assert.equal(nodeFrame.material, frameMaterial)
  assert.equal(renderer.nodes.getNodeFrameForRender({
    renderer,
    scene,
    object: scene.children[0],
    camera,
    material: frameMaterial,
  }), nodeFrame)
  assert.equal(renderer.nodes.getOutputCacheKey(), `${renderer.toneMapping},${renderer.currentColorSpace}`)
  const outputTarget = {}
  assert.equal(renderer.nodes.hasOutputChange(outputTarget), true)
  const outputNode = renderer.nodes.getOutputNode(outputTarget)
  assert.equal(outputNode.isNode, true)
  assert.equal(outputNode.isHeadlessRendererOutputNode, true)
  assert.equal(outputNode.outputTarget, outputTarget)
  assert.equal(renderer.nodes.hasOutputChange(outputTarget), false)
  const environmentNode = { isNode: true }
  const backgroundNode = { isNode: true }
  const fogNode = { isNode: true }
  scene.environmentNode = environmentNode
  scene.backgroundNode = backgroundNode
  scene.fogNode = fogNode
  assert.equal(renderer.nodes.getEnvironmentNode(scene), environmentNode)
  assert.equal(renderer.nodes.getBackgroundNode(scene), backgroundNode)
  assert.equal(renderer.nodes.getFogNode(scene), fogNode)
  assert.equal(typeof renderer.nodes.getCacheKey(scene, { getCacheKey: () => 'lights' }), 'number')
  assert.equal(renderer.nodes.isToneMappingState, true)
  assert.equal(renderer.nodes.needsRefresh({}), false)
  function ToneMappingNode() {}
  class BasicNodeMaterial {
    constructor() {
      this.isNodeMaterial = true
      this.type = 'BasicNodeMaterial'
    }
  }
  class PointLightNode {}
  class PointLightClass {}
  renderer.library.addToneMapping(ToneMappingNode, THREE.LinearToneMapping)
  assert.equal(renderer.library.getToneMappingFunction(THREE.LinearToneMapping), ToneMappingNode)
  assert.equal(renderer.library.getToneMappingFunction(THREE.ReinhardToneMapping), null)
  renderer.library.addMaterial(BasicNodeMaterial, 'MeshBasicMaterial')
  assert.equal(renderer.library.getMaterialNodeClass('MeshBasicMaterial'), BasicNodeMaterial)
  assert.equal(renderer.library.getMaterialNodeClass('MeshPhongMaterial'), null)
  const sourceMaterial = new THREE.MeshBasicMaterial({ color: 0x204080 })
  sourceMaterial.customCompatibilityValue = 7
  const nodeMaterial = renderer.library.fromMaterial(sourceMaterial)
  assert.equal(nodeMaterial.isNodeMaterial, true)
  assert.equal(nodeMaterial.customCompatibilityValue, 7)
  assert.equal(renderer.library.fromMaterial(nodeMaterial), nodeMaterial)
  assert.equal(renderer.library.fromMaterial({ type: 'UnregisteredMaterial' }), null)
  renderer.library.addLight(PointLightNode, PointLightClass)
  assert.equal(renderer.library.getLightNodeClass(PointLightClass), PointLightNode)
  renderer.library.addType(ToneMappingNode, 'custom-tone', renderer.library.materialNodes)
  assert.equal(renderer.library.materialNodes.get('custom-tone'), ToneMappingNode)
  const lightingNode = renderer.lighting.createNode([scene])
  assert.equal(lightingNode.isLightsNode, true)
  assert.deepEqual(lightingNode.getLights(), [scene])
  assert.equal(lightingNode.setLights([camera]), lightingNode)
  assert.deepEqual(lightingNode.getLights(), [camera])
  const cachedLightingNode = renderer.lighting.getNode(scene, camera)
  assert.equal(renderer.lighting.getNode(scene, camera), cachedLightingNode)
  assert.deepEqual(cachedLightingNode.getLights(), [])
  assert.ok(renderer.lighting.weakMap instanceof WeakMap)
  assert.equal(renderer.lighting.get([scene, camera]), cachedLightingNode)
  const alternateCamera = makeCamera()
  const manualLightingNode = renderer.lighting.createNode([scene, camera])
  assert.strictEqual(renderer.lighting.set([scene, alternateCamera], manualLightingNode), renderer.lighting)
  assert.equal(renderer.lighting.get([scene, alternateCamera]), manualLightingNode)
  assert.equal(renderer.lighting.getNode(scene, alternateCamera), manualLightingNode)
  assert.equal(renderer.lighting.delete([scene, alternateCamera]), true)
  assert.equal(renderer.lighting.get([scene, alternateCamera]), undefined)
  assert.notEqual(renderer.lighting.getNode(scene, alternateCamera), manualLightingNode)
  assert.equal(renderer.lighting.delete([scene, alternateCamera]), true)
  assert.equal(renderer.lighting.delete([scene, {}]), false)
  const quadLightingNode = renderer.lighting.getNode({ isQuadMesh: true }, null)
  assert.equal(renderer.lighting.getNode({ isQuadMesh: true }, undefined), quadLightingNode)
  assert.equal(quadLightingNode.isLightsNode, true)
  assert.deepEqual(quadLightingNode.getLights(), [])
  assert.equal(renderer.extensions.has('EXT_texture_filter_anisotropic'), false)
  assert.equal(renderer.extensions.get('EXT_texture_filter_anisotropic'), null)
  assert.equal(renderer.extensions.init(), undefined)
  const defaultInspector = renderer.inspector
  assert.equal(defaultInspector.currentFrame, null)
  assert.equal(defaultInspector.getRenderer(), renderer)
  assert.equal(defaultInspector.init(), undefined)
  assert.equal(defaultInspector.begin(), undefined)
  assert.equal(defaultInspector.finish(), undefined)
  assert.equal(defaultInspector.inspect({}), undefined)
  assert.equal(defaultInspector.beginRender('render-1', scene, camera, null), undefined)
  assert.equal(defaultInspector.finishRender('render-1'), undefined)
  assert.equal(defaultInspector.beginCompute('compute-1', {}), undefined)
  assert.equal(defaultInspector.finishCompute('compute-1'), undefined)
  assert.equal(defaultInspector.computeAsync({}), undefined)

  const inspectorEvents = []
  const customInspector = {
    currentFrame: { id: 1 },
    setRenderer(value) {
      inspectorEvents.push(['setRenderer', value])
      this.renderer = value
      return this
    },
    getRenderer() {
      return this.renderer
    },
    copyTextureToTexture(srcTexture, dstTexture) {
      inspectorEvents.push(['copyTextureToTexture', srcTexture, dstTexture])
    },
  }
  renderer.inspector = customInspector
  assert.equal(defaultInspector.getRenderer(), null)
  assert.equal(customInspector.getRenderer(), renderer)
  const inspectorSource = new THREE.DataTexture(new Uint8Array([1, 2, 3, 255]), 1, 1, THREE.RGBAFormat)
  const inspectorDestination = new THREE.DataTexture(new Uint8Array([0, 0, 0, 0]), 1, 1, THREE.RGBAFormat)
  renderer.copyTextureToTexture(inspectorSource, inspectorDestination)
  assert.deepEqual(Array.from(inspectorDestination.image.data), [1, 2, 3, 255])
  assert.equal(inspectorEvents.length, 2)
  assert.deepEqual(inspectorEvents[0], ['setRenderer', renderer])
  assert.deepEqual(inspectorEvents[1], ['copyTextureToTexture', inspectorSource, inspectorDestination])
  assert.throws(
    () => { renderer.inspector = null },
    /Renderer\.inspector must be an inspector-like object/i,
  )
  assert.throws(
    () => { renderer.inspector = { setRenderer: 'renderer' } },
    /Renderer\.inspector\.setRenderer must be a function/i,
  )
  assert.throws(
    () => { renderer.inspector = { setRenderer() {}, beginRender: true } },
    /Renderer\.inspector\.beginRender must be a function/i,
  )
  assert.equal(renderer.state.buffers.color.setMask(false), undefined)
  assert.equal(renderer.state.buffers.color.setLocked(true), undefined)
  assert.equal(renderer.state.buffers.color.setClear(0.1, 0.2, 0.3, 0.4, true), undefined)
  assert.equal(renderer.state.buffers.color.reset(), undefined)
  assert.equal(renderer.state.color, renderer.state.buffers.color)
  assert.equal(renderer.state.depth, renderer.state.buffers.depth)
  assert.equal(renderer.state.stencil, renderer.state.buffers.stencil)
  assert.equal(renderer.state.color.setMask(false), undefined)
  assert.equal(renderer.state.depth.setClear(0.5), undefined)
  assert.equal(renderer.state.stencil.setClear(1), undefined)
  assert.equal(renderer.state.buffers.depth.getReversed(), false)
  assert.equal(renderer.state.buffers.depth.setReversed(false), undefined)
  assert.equal(renderer.state.buffers.depth.getReversed(), false)
  assert.equal(renderer.state.buffers.depth.setTest(true), undefined)
  assert.equal(renderer.state.buffers.depth.setMask(true), undefined)
  assert.equal(renderer.state.buffers.depth.setFunc(THREE.LessEqualDepth), undefined)
  assert.equal(renderer.state.buffers.depth.setClear(0.5), undefined)
  assert.equal(renderer.state.buffers.depth.setLocked(false), undefined)
  assert.equal(renderer.state.buffers.depth.reset(), undefined)
  assert.equal(renderer.state.buffers.stencil.setTest(true), undefined)
  assert.equal(renderer.state.buffers.stencil.setMask(0xff), undefined)
  assert.equal(renderer.state.buffers.stencil.setFunc(THREE.AlwaysStencilFunc, 1, 0xff), undefined)
  assert.equal(renderer.state.buffers.stencil.setOp(
    THREE.ReplaceStencilOp,
    THREE.KeepStencilOp,
    THREE.KeepStencilOp,
  ), undefined)
  assert.equal(renderer.state.buffers.stencil.setClear(1), undefined)
  assert.equal(renderer.state.buffers.stencil.setLocked(false), undefined)
  assert.equal(renderer.state.buffers.stencil.reset(), undefined)
  assert.equal(renderer.state.setBlending(
    THREE.CustomBlending,
    THREE.AddEquation,
    THREE.SrcAlphaFactor,
    THREE.OneMinusSrcAlphaFactor,
    THREE.AddEquation,
    THREE.OneFactor,
    THREE.OneMinusSrcAlphaFactor,
    new THREE.Color(0, 0, 0),
    0,
    true,
  ), undefined)
  assert.equal(renderer.state.setMaterial(new THREE.MeshBasicMaterial(), false), undefined)
  assert.equal(renderer.state.setFlipSided(false), undefined)
  assert.equal(renderer.state.setCullFace(THREE.CullFaceBack), undefined)
  assert.equal(renderer.state.setLineWidth(2), undefined)
  assert.equal(renderer.state.setPolygonOffset(true, 1, -1), undefined)
  assert.equal(renderer.state.setScissorTest(true), undefined)
  assert.equal(renderer.state.setColorMask(false), undefined)
  assert.equal(renderer.state.setDepthTest(true), undefined)
  assert.equal(renderer.state.setDepthMask(true), undefined)
  assert.equal(renderer.state.setDepthFunc(THREE.LessEqualDepth), undefined)
  assert.equal(renderer.state.setReversedDepth(false), undefined)
  assert.equal(renderer.state.setStencilTest(true), undefined)
  assert.equal(renderer.state.setStencilMask(0xff), undefined)
  assert.equal(renderer.state.setStencilFunc(THREE.AlwaysStencilFunc, 1, 0xff), undefined)
  assert.equal(renderer.state.setStencilOp(
    THREE.ReplaceStencilOp,
    THREE.KeepStencilOp,
    THREE.KeepStencilOp,
  ), undefined)
  assert.equal(renderer.state.buffers.color.setLocked(true), undefined)
  assert.equal(renderer.state.buffers.color.setLocked(false), undefined)
  assert.equal(renderer.state.buffers.depth.setLocked(true), undefined)
  assert.equal(renderer.state.buffers.depth.setLocked(false), undefined)
  assert.equal(renderer.state.buffers.stencil.setLocked(true), undefined)
  assert.equal(renderer.state.buffers.stencil.setLocked(false), undefined)
  assert.equal(renderer.state.scissor(new THREE.Vector4(0, 0, 8, 8)), undefined)
  assert.equal(renderer.state.scissor(0, 0, 8, 8), undefined)
  assert.equal(renderer.state.viewport({ x: 0, y: 0, width: 16, height: 16 }), undefined)
  assert.equal(renderer.state.viewport(0, 0, 16, 16), undefined)
  assert.equal(renderer.state.resetVertexState(), undefined)
  assert.equal(renderer.state.reset(), undefined)
  assert.equal(renderer.state.unbindTexture(), undefined)
  assert.throws(
    () => renderer.state.buffers.color.setMask(1),
    /Renderer\.state\.buffers\.color\.setMask mask must be a boolean/i,
  )
  assert.throws(
    () => renderer.state.buffers.color.setLocked(1),
    /Renderer\.state\.buffers\.color\.setLocked lock must be a boolean/i,
  )
  assert.throws(
    () => renderer.state.buffers.depth.setFunc('less'),
    /Renderer\.state\.buffers\.depth\.setFunc func must be a finite integer/i,
  )
  assert.throws(
    () => renderer.state.buffers.depth.setLocked(1),
    /Renderer\.state\.buffers\.depth\.setLocked lock must be a boolean/i,
  )
  assert.throws(
    () => renderer.state.buffers.depth.setReversed(0),
    /Renderer\.state\.buffers\.depth\.setReversed reversed must be a boolean/i,
  )
  assert.throws(
    () => renderer.state.buffers.depth.setReversed(true),
    /Renderer\.state\.buffers\.depth\.setReversed\(true\) is not supported.*reversed depth buffers/i,
  )
  assert.throws(
    () => renderer.state.buffers.stencil.setTest(1),
    /Renderer\.state\.buffers\.stencil\.setTest test must be a boolean/i,
  )
  assert.throws(
    () => renderer.state.buffers.stencil.setLocked(1),
    /Renderer\.state\.buffers\.stencil\.setLocked lock must be a boolean/i,
  )
  assert.throws(
    () => renderer.state.buffers.stencil.setClear(0.5),
    /Renderer\.state\.buffers\.stencil\.setClear stencil must be a finite integer/i,
  )
  assert.throws(
    () => renderer.state.setBlending('normal'),
    /Renderer\.state\.setBlending blending normal is not supported/i,
  )
  assert.throws(
    () => renderer.state.setBlending(
      THREE.NormalBlending,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      1,
    ),
    /Renderer\.state\.setBlending blendColor must be a color-like object/i,
  )
  assert.throws(
    () => renderer.state.setBlending(
      THREE.NormalBlending,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      'opaque',
    ),
    /Renderer\.state\.setBlending blendAlpha must be a finite number/i,
  )
  assert.throws(
    () => renderer.state.setBlending(
      THREE.NormalBlending,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      1,
    ),
    /Renderer\.state\.setBlending premultipliedAlpha must be a boolean/i,
  )
  assert.throws(
    () => renderer.state.setMaterial(null),
    /Renderer\.state\.setMaterial material must be a material-like object/i,
  )
  assert.throws(
    () => renderer.state.setMaterial(new THREE.MeshBasicMaterial(), 1),
    /Renderer\.state\.setMaterial frontFaceCW must be a boolean/i,
  )
  assert.throws(
    () => renderer.state.setFlipSided(1),
    /Renderer\.state\.setFlipSided flipSided must be a boolean/i,
  )
  assert.throws(
    () => renderer.state.setCullFace(99),
    /Renderer\.state\.setCullFace cullFace 99 is not supported/i,
  )
  assert.throws(
    () => renderer.state.setLineWidth(0),
    /Renderer\.state\.setLineWidth width must be greater than 0/i,
  )
  assert.throws(
    () => renderer.state.setPolygonOffset(1, 0, 0),
    /Renderer\.state\.setPolygonOffset polygonOffset must be a boolean/i,
  )
  assert.throws(
    () => renderer.state.setScissorTest(1),
    /Renderer\.state\.setScissorTest scissorTest must be a boolean/i,
  )
  assert.throws(
    () => renderer.state.setColorMask(1),
    /Renderer\.state\.buffers\.color\.setMask mask must be a boolean/i,
  )
  assert.throws(
    () => renderer.state.setReversedDepth(true),
    /Renderer\.state\.buffers\.depth\.setReversed\(true\) is not supported.*reversed depth buffers/i,
  )
  assert.throws(
    () => renderer.state.setStencilTest(1),
    /Renderer\.state\.buffers\.stencil\.setTest test must be a boolean/i,
  )
  assert.throws(
    () => renderer.state.viewport({ x: 0, y: 0, width: 0, height: 1 }),
    /Renderer\.state\.viewport width and height must be greater than 0/i,
  )
  assert.throws(
    () => renderer.state.scissor(0, 0, 0, 1),
    /Renderer\.state\.scissor width and height must be greater than 0/i,
  )
  for (const [method, pattern] of [
    ['enable', /WebGL capability flags/i],
    ['disable', /WebGL capability flags/i],
    ['bindFramebuffer', /WebGL framebuffer binding/i],
    ['drawBuffers', /WebGL draw-buffer binding/i],
    ['useProgram', /WebGL program binding/i],
    ['setMRTBlending', /WebGL MRT indexed blending/i],
    ['setVertexState', /WebGL vertex-array binding/i],
    ['activeTexture', /WebGL texture-unit binding/i],
    ['bindTexture', /WebGL texture binding/i],
    ['compressedTexImage2D', /WebGL texture uploads/i],
    ['compressedTexImage3D', /WebGL texture uploads/i],
    ['texImage2D', /WebGL texture uploads/i],
    ['texImage3D', /WebGL texture uploads/i],
    ['texStorage2D', /WebGL texture storage/i],
    ['texStorage3D', /WebGL texture storage/i],
    ['texSubImage2D', /WebGL texture uploads/i],
    ['texSubImage3D', /WebGL texture uploads/i],
    ['compressedTexSubImage2D', /WebGL texture uploads/i],
    ['compressedTexSubImage3D', /WebGL texture uploads/i],
    ['updateUBOMapping', /WebGL uniform-buffer binding/i],
    ['uniformBlockBinding', /WebGL uniform-buffer binding/i],
    ['bindBufferBase', /WebGL uniform-buffer binding/i],
  ]) {
    assert.throws(
      () => renderer.state[method](),
      new RegExp(`Renderer\\.state\\.${method}\\(\\) is not supported.*${pattern.source}`, 'i'),
    )
  }

})
