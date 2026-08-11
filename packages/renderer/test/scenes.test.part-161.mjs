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
test('Renderer framebuffer and texture handle APIs fail clearly', () => {
  const renderer = new Renderer()
  const target = { texture: {}, depthTexture: {} }
  const externalColorTexture = {}
  const externalDepthTexture = {}
  const externalFramebuffer = {}
  const source = new THREE.DataTexture(new Uint8Array([255, 0, 0, 255]), 1, 1, THREE.RGBAFormat)
  const destination = new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1, THREE.RGBAFormat)
  const compressedTexture = () => new THREE.CompressedTexture([
    { data: new Uint8Array(16), width: 4, height: 4 },
  ], 4, 4, THREE.RGBAFormat)
  const compressedFormatTexture = () => new THREE.DataTexture(
    new Uint8Array(16),
    4,
    4,
    THREE.RGBA_S3TC_DXT5_Format,
  )

  assert.throws(
    () => renderer.setRenderTargetTextures(target, externalColorTexture, externalDepthTexture),
    /Renderer\.setRenderTargetTextures\(\).*not supported.*WebGLTexture handles/i,
  )
  assert.throws(
    () => renderer.setRenderTargetFramebuffer(target, externalFramebuffer),
    /Renderer\.setRenderTargetFramebuffer\(\).*not supported.*WebGL framebuffer/i,
  )
  assert.throws(
    () => renderer.copyFramebufferToTexture(source),
    /Renderer\.copyFramebufferToTexture\(\) requires an active render target.*setRenderTarget/i,
  )
  renderer.setRenderTarget({ texture: {} })
  assert.throws(
    () => renderer.copyFramebufferToTexture(source),
    /Renderer\.copyFramebufferToTexture target has no readable color data.*Render into the target/i,
  )
  assert.throws(
    () => renderer.copyFramebufferToTexture(source, null, -1),
    /Renderer\.copyFramebufferToTexture level must be a non-negative integer/i,
  )
  renderer.setRenderTarget({ width: 1, height: 1, texture: {}, data: new Uint8Array([255, 0, 0, 255]) })
  assert.throws(
    () => renderer.copyFramebufferToTexture(new THREE.FramebufferTexture(1, 1)),
    /Renderer\.copyFramebufferToTexture texture uses a FramebufferTexture/i,
  )
  assert.throws(
    () => renderer.copyFramebufferToTexture(new THREE.DepthTexture(1, 1)),
    /Renderer\.copyFramebufferToTexture texture uses a DepthTexture/i,
  )
  assert.throws(
    () => renderer.copyFramebufferToTexture(new THREE_WEBGPU.StorageTexture(1, 1)),
    /Renderer\.copyFramebufferToTexture texture uses a StorageTexture.*backing data.*not directly readable/i,
  )
  assert.throws(
    () => renderer.copyFramebufferToTexture(compressedTexture()),
    /Renderer\.copyFramebufferToTexture texture uses a compressed texture.*texture copy.*Pre-decode/i,
  )
  assert.throws(
    () => renderer.copyFramebufferToTexture(compressedFormatTexture()),
    /Renderer\.copyFramebufferToTexture texture uses a compressed texture format.*texture copy.*Pre-decode/i,
  )
  assert.throws(
    () => renderer.copyFramebufferToTexture(source, { x: 1, y: 0 }),
    /Renderer\.copyFramebufferToTexture source rectangle must fit inside the active framebuffer bounds/i,
  )
  assert.throws(
    () => renderer.copyFramebufferToTexture(source, { x: 0, y: 0, width: 0, height: 1 }),
    /Renderer\.copyFramebufferToTexture source rectangle\.width must be a positive integer/i,
  )
  assert.throws(
    () => renderer.copyFramebufferToTexture(source, 'position'),
    /Renderer\.copyFramebufferToTexture source rectangle must be a vector, rectangle object, array, or null/i,
  )
  renderer.setRenderTarget(null)
  assert.throws(
    () => renderer.setRenderTargetTextures(null, externalColorTexture),
    /Renderer\.setRenderTargetTextures renderTarget must be a target-like object/i,
  )
  assert.throws(
    () => renderer.setRenderTargetTextures(target, null),
    /Renderer\.setRenderTargetTextures colorTexture must be an external WebGL object-like handle/i,
  )
  assert.throws(
    () => renderer.setRenderTargetTextures(target, externalColorTexture, []),
    /Renderer\.setRenderTargetTextures depthTexture must be an external WebGL object-like handle/i,
  )
  assert.throws(
    () => renderer.setRenderTargetFramebuffer(null, externalFramebuffer),
    /Renderer\.setRenderTargetFramebuffer renderTarget must be a target-like object/i,
  )
  assert.throws(
    () => renderer.setRenderTargetFramebuffer(target, []),
    /Renderer\.setRenderTargetFramebuffer defaultFramebuffer must be an external WebGL object-like handle/i,
  )
  assert.throws(
    () => renderer.copyFramebufferToTexture(null),
    /Renderer\.copyFramebufferToTexture texture must be a texture-like object/i,
  )
  assert.throws(
    () => renderer.copyTextureToTexture(null, destination),
    /Renderer\.copyTextureToTexture source texture must be a texture-like object/i,
  )
  assert.throws(
    () => renderer.copyTextureToTexture(source, null),
    /Renderer\.copyTextureToTexture destination texture must be a texture-like object/i,
  )
  assert.throws(
    () => renderer.copyTextureToTexture({ isTexture: true, image: Buffer.from([1, 2, 3, 4]) }, destination),
    /Renderer\.copyTextureToTexture source texture must provide a readable image object.*raw data.*canvas-like pixel access/i,
  )
  assert.throws(
    () => renderer.copyTextureToTexture(new THREE.FramebufferTexture(1, 1), destination),
    /Renderer\.copyTextureToTexture source texture uses a FramebufferTexture/i,
  )
  assert.throws(
    () => renderer.copyTextureToTexture(source, new THREE.FramebufferTexture(1, 1)),
    /Renderer\.copyTextureToTexture destination texture uses a FramebufferTexture/i,
  )
  assert.throws(
    () => renderer.copyTextureToTexture(new THREE.DepthTexture(1, 1), destination),
    /Renderer\.copyTextureToTexture source texture uses a DepthTexture/i,
  )
  assert.throws(
    () => renderer.copyTextureToTexture(source, new THREE.DepthTexture(1, 1)),
    /Renderer\.copyTextureToTexture destination texture uses a DepthTexture/i,
  )
  assert.throws(
    () => renderer.copyTextureToTexture(new THREE.VideoTexture({ videoWidth: 1, videoHeight: 1 }), destination),
    /Renderer\.copyTextureToTexture source texture uses a VideoTexture.*live video frames.*not directly readable/i,
  )
  assert.throws(
    () => renderer.copyTextureToTexture(source, new THREE.VideoTexture({ videoWidth: 1, videoHeight: 1 })),
    /Renderer\.copyTextureToTexture destination texture uses a VideoTexture.*live video frames.*not directly readable/i,
  )
  assert.throws(
    () => renderer.copyTextureToTexture(new THREE_WEBGPU.StorageTexture(1, 1), destination),
    /Renderer\.copyTextureToTexture source texture uses a StorageTexture.*backing data.*not directly readable/i,
  )
  assert.throws(
    () => renderer.copyTextureToTexture(source, new THREE_WEBGPU.StorageTexture(1, 1)),
    /Renderer\.copyTextureToTexture destination texture uses a StorageTexture.*backing data.*not directly readable/i,
  )
  assert.throws(
    () => renderer.copyTextureToTexture(compressedTexture(), destination),
    /Renderer\.copyTextureToTexture source texture uses a compressed texture.*texture copy.*Pre-decode/i,
  )
  assert.throws(
    () => renderer.copyTextureToTexture(source, compressedTexture()),
    /Renderer\.copyTextureToTexture destination texture uses a compressed texture.*texture copy.*Pre-decode/i,
  )
  assert.throws(
    () => renderer.copyTextureToTexture(compressedFormatTexture(), destination),
    /Renderer\.copyTextureToTexture source texture uses a compressed texture format.*texture copy.*Pre-decode/i,
  )
  assert.throws(
    () => renderer.copyTextureToTexture(source, destination, null, null, 1, 0),
    /Renderer\.copyTextureToTexture source texture\.mipmaps\[0\] must provide a readable raw image object/i,
  )
  assert.throws(
    () => renderer.copyTextureToTexture(source, destination, null, null, -1, 0),
    /Renderer\.copyTextureToTexture source level must be a non-negative integer/i,
  )
  assert.throws(
    () => renderer.copyTextureToTexture(source, destination, { x: 0, y: 0, width: 2, height: 1 }),
    /Renderer\.copyTextureToTexture source region must fit inside the source texture bounds/i,
  )
  const rgbSource = new THREE.DataTexture(new Uint8Array([255, 0, 0]), 1, 1, THREE.RGBFormat)
  assert.throws(
    () => renderer.copyTextureToTexture(rgbSource, destination),
    /same raw channel count/i,
  )
  const canvasDestination = new THREE.Texture({
    width: 1,
    height: 1,
    getContext(type) {
      if (type !== '2d') return null
      return {
        getImageData() {
          return { data: new Uint8ClampedArray([0, 0, 0, 255]), width: 1, height: 1 }
        },
      }
    },
  })
  canvasDestination.needsUpdate = true
  assert.throws(
    () => renderer.copyTextureToTexture(source, canvasDestination),
    /Renderer\.copyTextureToTexture destination texture must provide a readable raw image object/i,
  )
  assert.throws(
    () => renderer.copyTextureToTexture(new THREE.DataArrayTexture(new Uint8Array([255, 0, 0, 255]), 1, 1, 1), destination),
    /Renderer\.copyTextureToTexture source texture uses an array or 3D texture/i,
  )
  assert.throws(
    () => renderer.copyTextureToTexture(source, new THREE.Data3DTexture(new Uint8Array([0, 0, 0, 255]), 1, 1, 1)),
    /Renderer\.copyTextureToTexture destination texture uses an array or 3D texture/i,
  )
  assert.throws(
    () => renderer.copyTextureToTexture3D(source, destination),
    /Renderer\.copyTextureToTexture3D\(\) is not supported.*3D and array texture GPU copies.*Renderer\.copyTextureToTexture\(\)/i,
  )
  assert.throws(
    () => renderer.copyTextureToTexture3D(
      { min: { x: 0, y: 0, z: 0 }, max: { x: 1, y: 1, z: 1 }, isBox3: true },
      { x: 0, y: 0, z: 0 },
      source,
      destination,
    ),
    /Renderer\.copyTextureToTexture3D\(\) is not supported.*3D and array texture GPU copies.*Renderer\.copyTextureToTexture\(\)/i,
  )
  assert.throws(
    () => renderer.copyTextureToTexture3D(null, destination),
    /Renderer\.copyTextureToTexture3D source texture must be a texture-like object/i,
  )
  assert.throws(
    () => renderer.copyTextureToTexture3D(source, null),
    /Renderer\.copyTextureToTexture3D destination texture must be a texture-like object/i,
  )
  assert.throws(
    () => renderer.copyTextureToTexture3D(source, destination, null, null, -1),
    /Renderer\.copyTextureToTexture3D level must be a non-negative integer/i,
  )
})

test('Renderer copyTextureToTexture copies readable raw texture data on the CPU', () => {
  const renderer = new Renderer()
  const source = new THREE.DataTexture(new Uint8Array([
    255, 0, 0, 255,
    0, 255, 0, 255,
    0, 0, 255, 255,
    255, 255, 0, 255,
  ]), 2, 2, THREE.RGBAFormat)
  const destinationData = new Uint8Array(3 * 3 * 4)
  destinationData.fill(9)
  const destination = new THREE.DataTexture(destinationData, 3, 3, THREE.RGBAFormat)
  const initialVersion = destination.version

  renderer.copyTextureToTexture(source, destination, { x: 1, y: 0, width: 1, height: 2 }, { x: 1, y: 1 })

  function pixel(x, y) {
    const offset = (y * 3 + x) * 4
    return Array.from(destination.image.data.slice(offset, offset + 4))
  }

  assert.deepEqual(pixel(1, 1), [0, 255, 0, 255])
  assert.deepEqual(pixel(1, 2), [255, 255, 0, 255])
  assert.deepEqual(pixel(0, 0), [9, 9, 9, 9])
  assert.ok(destination.version > initialVersion, 'destination texture should be marked dirty after CPU copy')
})

test('Renderer copyTextureToTexture copies readable texture source.data on the CPU', () => {
  const renderer = new Renderer()
  const source = new THREE.Texture()
  source.source = {
    data: {
      data: new Uint8Array([
        11, 21, 31, 255,
        41, 51, 61, 255,
        71, 81, 91, 255,
        101, 111, 121, 255,
      ]),
      width: 2,
      height: 2,
    },
  }

  const destinationData = new Uint8Array(3 * 2 * 4)
  destinationData.fill(3)
  const destination = new THREE.Texture()
  destination.source = {
    data: {
      data: destinationData,
      width: 3,
      height: 2,
    },
  }
  const initialVersion = destination.version

  renderer.copyTextureToTexture(source, destination, { x: 0, y: 1, width: 2, height: 1 }, { x: 1, y: 0 })

  function pixel(x, y) {
    const width = destination.source.data.width
    const offset = (y * width + x) * 4
    return Array.from(destination.source.data.data.slice(offset, offset + 4))
  }

  assert.deepEqual(pixel(1, 0), [71, 81, 91, 255])
  assert.deepEqual(pixel(2, 0), [101, 111, 121, 255])
  assert.deepEqual(pixel(0, 0), [3, 3, 3, 3])
  assert.ok(destination.version > initialVersion, 'destination texture should be marked dirty after source.data CPU copy')
})
