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
import { makeCamera, meanAbsDiff, renderRgba, solidTexture } from './scenes.test.part-002.mjs'
test('unsupported render target MRT and invalid MSAA requests fail clearly', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.1, 0.1, 0.1)
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial({ color: 0x00ffaa })))
  const camera = makeCamera()

  assert.throws(
    () => renderToTarget(scene, camera, null, { width: 32, height: 32 }),
    /target must be a target-like object/i,
  )
  assert.throws(
    () => renderToTarget(scene, camera, [], { width: 32, height: 32 }),
    /target must be a target-like object/i,
  )
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32, target: 'bad' }),
    /options\.target must be a target-like object/i,
  )
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32, target: [] }),
    /options\.target must be a target-like object/i,
  )

  const renderer = new Renderer()
  assert.throws(
    () => renderer.setRenderTarget('bad'),
    /Renderer\.setRenderTarget target must be a target-like object/i,
  )
  assert.throws(
    () => renderer.setRenderTarget({}, 6),
    /Renderer\.setRenderTarget activeCubeFace must be an integer from 0 to 5/i,
  )
  assert.throws(
    () => renderer.setRenderTarget({}, 0, -1),
    /Renderer\.setRenderTarget activeMipmapLevel must be a non-negative integer/i,
  )
  assert.throws(
    () => renderer.setRenderTarget({ texture: [{}, {}] }),
    /secondary color attachment.*renderMode/i,
  )

  const arrayOr3DRenderTargetClasses = [
    ['THREE.RenderTargetArray', THREE.RenderTargetArray],
    ['THREE.RenderTarget3D', THREE.RenderTarget3D],
    ['THREE.WebGLArrayRenderTarget', THREE.WebGLArrayRenderTarget],
    ['THREE.WebGL3DRenderTarget', THREE.WebGL3DRenderTarget],
  ].filter(([, TargetClass]) => typeof TargetClass === 'function')

  assert.ok(arrayOr3DRenderTargetClasses.length > 0, 'Expected installed Three.js to expose at least one array or 3D render target class.')
  for (const [label, TargetClass] of arrayOr3DRenderTargetClasses) {
    const directTarget = new TargetClass(4, 4, 2)
    assert.throws(
      () => renderToTarget(scene, camera, directTarget, { width: 32, height: 32 }),
      /target color texture uses an array or 3D texture/i,
      label,
    )
    assert.throws(
      () => renderRgba(scene, camera, { width: 32, height: 32, target: new TargetClass(4, 4, 2) }),
      /target color texture uses an array or 3D texture/i,
      `${label} options.target`,
    )
    assert.throws(
      () => renderer.setRenderTarget(new TargetClass(4, 4, 2)),
      /target color texture uses an array or 3D texture/i,
      `${label} Renderer.setRenderTarget`,
    )
  }

  const targetCases = [
    [{ image: 'bad' }, /target\.image must be an image-like object/i, 'target image container'],
    [{ texture: 'bad' }, /target\.texture must be a texture-like object/i, 'color texture container'],
    [{ texture: [] }, /target\.texture must contain one texture-like object/i, 'empty texture array'],
    [{ texture: ['bad'] }, /target\.texture\[0\] must be a texture-like object/i, 'texture array element'],
    [{ textures: 'bad' }, /target\.textures must be an array of texture-like objects/i, 'textures container'],
    [{ textures: [] }, /target\.textures must contain one texture-like object/i, 'empty textures array'],
    [{ textures: ['bad'] }, /target\.textures\[0\] must be a texture-like object/i, 'textures array element'],
    [{ depthTexture: 'bad' }, /target\.depthTexture must be a texture-like object/i, 'depth texture container'],
    [{ texture: { image: 'bad' } }, /target\.texture\.image must be an image-like object/i, 'texture image container'],
    [{ texture: { mipmaps: ['bad'] } }, /target\.texture\.mipmaps\[0\] must be an image-like object/i, 'texture mipmap container'],
    [{ texture: { source: 'bad' } }, /target\.texture\.source must be a source-like object/i, 'texture source container'],
    [{ texture: { source: { data: 'bad' } } }, /target\.texture\.source\.data must be an image-like object/i, 'texture source data container'],
    [{ texture: [{}, {}] }, /secondary color attachment.*renderMode/i, 'texture array'],
    [{ textures: [{}, {}] }, /secondary color attachment.*renderMode/i, 'textures array'],
    [{ textures: [{}, { userData: { headlessThreeRenderer: { renderMode: 'albedo' } } }] }, /target color texture\[1\]\.userData\.headlessThreeRenderer\.renderMode must be "color", "mask", "object-id", "normal", or "depth"/i, 'secondary renderMode value'],
    [{ textures: [{}, { userData: { headlessThreeRenderer: 'mask' } }] }, /target color texture\[1\]\.userData\.headlessThreeRenderer must be an object/i, 'secondary renderMode hints'],
    [{ texture: new THREE.DataArrayTexture(new Uint8Array([255, 0, 0, 255]), 1, 1, 1) }, /target color texture uses an array or 3D texture/i, 'color array texture'],
    [{ depthTexture: new THREE.Data3DTexture(new Uint8Array([255, 0, 0, 255]), 1, 1, 1) }, /target\.depthTexture uses an array or 3D texture/i, 'depth 3D texture'],
    [{ texture: new THREE.FramebufferTexture(1, 1) }, /target color texture uses a FramebufferTexture/i, 'color framebuffer texture'],
    [{ depthTexture: new THREE.FramebufferTexture(1, 1) }, /target\.depthTexture uses a FramebufferTexture/i, 'depth framebuffer texture'],
    [{ texture: new THREE.DepthTexture(1, 1) }, /target color texture uses a DepthTexture as a color attachment/i, 'color depth texture'],
    [{ texture: new THREE_WEBGPU.StorageTexture(1, 1) }, /target color texture uses a StorageTexture.*scene-oriented output contract/i, 'color storage texture'],
    [{ depthTexture: new THREE_WEBGPU.StorageTexture(1, 1) }, /target\.depthTexture uses a StorageTexture.*scene-oriented output contract/i, 'depth storage texture'],
    [{ texture: new THREE.CompressedTexture([], 1, 1, THREE.RGBAFormat) }, /target color texture uses a compressed texture/i, 'color compressed texture'],
    [{ depthTexture: new THREE.CompressedTexture([], 1, 1, THREE.RGBAFormat) }, /target\.depthTexture uses a compressed texture/i, 'depth compressed texture'],
    [{ texture: { format: THREE.RGBA_S3TC_DXT5_Format } }, /target color texture format uses a compressed texture format/i, 'color compressed format'],
    [{ depthTexture: { format: THREE.RGBA_S3TC_DXT5_Format } }, /target\.depthTexture\.format uses a compressed texture format/i, 'depth compressed format'],
    [{ texture: { isCubeTexture: true } }, /target color texture uses a cube texture.*THREE\.CubeCamera/i, 'regular camera cube color texture'],
    [{ depthTexture: { isCubeTexture: true } }, /target\.depthTexture uses a cube texture.*THREE\.CubeCamera/i, 'regular camera cube depth texture'],
    [{ samples: 2 }, /MSAA sample count 2.*not supported/i, 'target samples'],
    [{ sampleCount: 8 }, /MSAA sample count 8.*not supported/i, 'target sampleCount'],
    [{ texture: { format: THREE.DepthFormat } }, /target color texture format .*not supported.*AlphaFormat.*LuminanceFormat.*LuminanceAlphaFormat.*RedFormat.*RedIntegerFormat.*RGFormat.*RGIntegerFormat.*RGBFormat.*RGBIntegerFormat.*RGBAFormat.*RGBAIntegerFormat/i, 'color texture format'],
    [{ texture: { type: THREE.UnsignedInt248Type } }, /target color texture type .*not supported.*UnsignedByteType.*ByteType.*ShortType.*UnsignedShortType.*IntType.*UnsignedIntType.*HalfFloatType.*FloatType.*UnsignedShort4444Type.*UnsignedShort5551Type.*UnsignedInt101111Type.*UnsignedInt5999Type/i, 'color texture type'],
    [{ depthTexture: { type: THREE.ByteType } }, /target\.depthTexture\.type .*not supported/i, 'depth texture type'],
    [{ depthTexture: { format: THREE.RGBAFormat } }, /target\.depthTexture\.format .*not supported/i, 'depth texture format'],
    [{ depthTexture: { type: THREE.FloatType, format: THREE.DepthStencilFormat } }, /DepthStencilFormat.*UnsignedInt248Type/i, 'depth-stencil format with scalar type'],
    [{ depthTexture: { type: THREE.UnsignedInt248Type, format: THREE.DepthFormat } }, /DepthFormat.*UnsignedInt248Type/i, 'depth format with packed depth-stencil type'],
  ]

  for (const [target, pattern, label] of targetCases) {
    assert.throws(
      () => renderToTarget(scene, camera, target, { width: 32, height: 32 }),
      pattern,
      label,
    )
  }

  const optionsTargetCases = [
    [{ texture: [{}, {}] }, /secondary color attachment.*renderMode/i, 'options.target texture array'],
    [{ texture: new THREE.DataArrayTexture(new Uint8Array([255, 0, 0, 255]), 1, 1, 1) }, /target color texture uses an array or 3D texture/i, 'options.target color array texture'],
    [{ texture: new THREE.FramebufferTexture(1, 1) }, /target color texture uses a FramebufferTexture/i, 'options.target color framebuffer texture'],
    [{ texture: new THREE.DepthTexture(1, 1) }, /target color texture uses a DepthTexture as a color attachment/i, 'options.target color depth texture'],
    [{ texture: new THREE_WEBGPU.StorageTexture(1, 1) }, /target color texture uses a StorageTexture.*scene-oriented output contract/i, 'options.target color storage texture'],
    [{ depthTexture: new THREE_WEBGPU.StorageTexture(1, 1) }, /target\.depthTexture uses a StorageTexture.*scene-oriented output contract/i, 'options.target depth storage texture'],
    [{ texture: new THREE.CompressedTexture([], 1, 1, THREE.RGBAFormat) }, /target color texture uses a compressed texture/i, 'options.target compressed color texture'],
    [{ texture: { format: THREE.RGBA_S3TC_DXT5_Format } }, /target color texture format uses a compressed texture format/i, 'options.target compressed color format'],
    [{ depthTexture: { format: THREE.RGBA_S3TC_DXT5_Format } }, /target\.depthTexture\.format uses a compressed texture format/i, 'options.target compressed depth format'],
    [{ sampleCount: 8 }, /MSAA sample count 8.*not supported/i, 'options.target sampleCount'],
    [{ texture: { format: THREE.DepthFormat } }, /target color texture format .*not supported.*AlphaFormat.*LuminanceFormat.*LuminanceAlphaFormat.*RedFormat.*RedIntegerFormat.*RGFormat.*RGIntegerFormat.*RGBFormat.*RGBIntegerFormat.*RGBAFormat.*RGBAIntegerFormat/i, 'options.target color texture format'],
    [{ depthTexture: { type: THREE.ByteType } }, /target\.depthTexture\.type .*not supported/i, 'options.target depth texture type'],
    [{ depthTexture: { format: THREE.RGBAFormat } }, /target\.depthTexture\.format .*not supported/i, 'options.target depth texture format'],
    [{ depthTexture: { type: THREE.FloatType, format: THREE.DepthStencilFormat } }, /DepthStencilFormat.*UnsignedInt248Type/i, 'options.target depth-stencil format with scalar type'],
    [{ depthTexture: { type: THREE.UnsignedInt248Type, format: THREE.DepthFormat } }, /DepthFormat.*UnsignedInt248Type/i, 'options.target depth format with packed depth-stencil type'],
  ]

  for (const [target, pattern, label] of optionsTargetCases) {
    assert.throws(
      () => renderRgba(scene, camera, { width: 32, height: 32, target }),
      pattern,
      label,
    )
  }

  for (const options of [{ samples: 2 }, { sampleCount: 8 }]) {
    assert.throws(
      () => renderRgba(scene, camera, { width: 32, height: 32, ...options }),
      /MSAA sample count .*not supported/i,
      JSON.stringify(options),
    )
  }
})

test('post-processing options modify the final image', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(1, 0, 0)
  const camera = makeCamera()

  const base = renderRgba(scene, camera, { width: 64, height: 64 })
  const processed = renderRgba(scene, camera, {
    width: 64,
    height: 64,
    postProcessing: { invert: 1, saturation: 1.5, vignette: 0.25 },
  })
  const diff = meanAbsDiff(base, processed)
  const mean = meanRgba(processed)
  assert.ok(diff > 20, `expected post processing to change image, diff=${diff.toFixed(2)}`)
  assert.ok(mean.g > mean.r, `inverted red background should have stronger green than red (${mean.g} vs ${mean.r})`)
})

test('post-processing exposure contrast and grayscale controls modify output', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.25, 0.5, 0.75)
  const camera = makeCamera()
  const options = { width: 64, height: 64, outputColorSpace: THREE.LinearSRGBColorSpace }

  const base = meanRgba(renderRgba(scene, camera, options))
  const exposed = meanRgba(renderRgba(scene, camera, {
    ...options,
    postProcessing: { exposure: 1 },
  }))
  const contrasted = meanRgba(renderRgba(scene, camera, {
    ...options,
    postProcessing: { contrast: 2 },
  }))
  const grayscale = meanRgba(renderRgba(scene, camera, {
    ...options,
    postProcessing: { grayscale: true },
  }))

  assert.ok(exposed.r > base.r + 20, `exposure should brighten red (${exposed.r} vs ${base.r})`)
  assert.ok(exposed.g > base.g + 20, `exposure should brighten green (${exposed.g} vs ${base.g})`)
  assert.ok(contrasted.r < base.r - 20, `contrast should darken below-mid red (${contrasted.r} vs ${base.r})`)
  assert.ok(contrasted.b > base.b + 20, `contrast should brighten above-mid blue (${contrasted.b} vs ${base.b})`)
  assert.ok(Math.max(grayscale.r, grayscale.g, grayscale.b) - Math.min(grayscale.r, grayscale.g, grayscale.b) < 3, `boolean grayscale should equalize color channels (${grayscale.r}, ${grayscale.g}, ${grayscale.b})`)
})

test('post-processing enabled false bypasses configured effects', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.25, 0.5, 0.75)
  const camera = makeCamera()
  const options = { width: 32, height: 32, outputColorSpace: THREE.LinearSRGBColorSpace }

  const base = renderRgba(scene, camera, options)
  const disabled = renderRgba(scene, camera, {
    ...options,
    postProcessing: {
      enabled: false,
      exposure: 4,
      contrast: 4,
      grayscale: true,
      invert: true,
      saturation: 0,
      vignette: 1,
    },
  })

  assert.deepEqual(disabled, base, 'postProcessing.enabled=false should ignore configured effects')
})

test('reusable renderer updates cached background and post-processing uniforms', () => {
  const renderer = new Renderer()
  const camera = makeCamera()

  const backgroundScene = new THREE.Scene()
  backgroundScene.background = solidTexture(0, 255, 0)
  const backgroundOptions = { width: 64, height: 64, format: 'rgba' }
  const dimmed = meanRgba(renderer.render(backgroundScene, camera, {
    ...backgroundOptions,
    backgroundIntensity: 0.25,
  }))
  const full = meanRgba(renderer.render(backgroundScene, camera, {
    ...backgroundOptions,
    backgroundIntensity: 1,
  }))
  const dimmedAgain = meanRgba(renderer.render(backgroundScene, camera, {
    ...backgroundOptions,
    backgroundIntensity: 0.25,
  }))
  assert.ok(full.g > dimmed.g + 60, `updated background uniform should brighten cached texture background (${full.g} vs ${dimmed.g})`)
  assert.ok(Math.abs(dimmedAgain.g - dimmed.g) < 4, `cached background uniform should update back to dimmed intensity (${dimmedAgain.g} vs ${dimmed.g})`)

  const postScene = new THREE.Scene()
  postScene.background = new THREE.Color(1, 0, 0)
  const postOptions = {
    width: 64,
    height: 64,
    format: 'rgba',
    outputColorSpace: THREE.LinearSRGBColorSpace,
  }
  const inverted = meanRgba(renderer.render(postScene, camera, {
    ...postOptions,
    postProcessing: { invert: 1 },
  }))
  const grayscale = meanRgba(renderer.render(postScene, camera, {
    ...postOptions,
    postProcessing: { grayscale: true },
  }))
  postScene.background = new THREE.Color(0, 0, 1)
  const invertedBlue = meanRgba(renderer.render(postScene, camera, {
    ...postOptions,
    postProcessing: { invert: 1 },
  }))
  postScene.background = new THREE.Color(1, 0, 0)
  const invertedAgain = meanRgba(renderer.render(postScene, camera, {
    ...postOptions,
    postProcessing: { invert: 1 },
  }))

  assert.ok(inverted.g > inverted.r + 80 && inverted.b > inverted.r + 80, `invert uniform should turn red toward cyan (${inverted.r}, ${inverted.g}, ${inverted.b})`)
  assert.ok(Math.max(grayscale.r, grayscale.g, grayscale.b) - Math.min(grayscale.r, grayscale.g, grayscale.b) < 3, `grayscale uniform should replace prior invert settings (${grayscale.r}, ${grayscale.g}, ${grayscale.b})`)
  assert.ok(invertedBlue.r > invertedBlue.b + 80 && invertedBlue.g > invertedBlue.b + 80, `cached post bind group should sample updated blue scene color as yellow after invert (${invertedBlue.r}, ${invertedBlue.g}, ${invertedBlue.b})`)
  assert.ok(Math.abs(invertedAgain.g - inverted.g) < 4 && Math.abs(invertedAgain.b - inverted.b) < 4, `post uniform buffer should update back to invert settings (${invertedAgain.r}, ${invertedAgain.g}, ${invertedAgain.b})`)
})

test('reusable renderer reflects mutated background texture bytes', () => {
  const renderer = new Renderer()
  const scene = new THREE.Scene()
  const camera = makeCamera()
  const data = new Uint8Array([255, 0, 0, 255])
  const background = new THREE.DataTexture(data, 1, 1, THREE.RGBAFormat)
  background.needsUpdate = true
  scene.background = background

  const options = { width: 64, height: 64, format: 'rgba' }
  const sample = () => meanRgba(renderer.render(scene, camera, options))

  const red = sample()
  data.set([0, 255, 0, 255])
  background.needsUpdate = true
  const green = sample()
  data.set([255, 0, 0, 255])
  background.needsUpdate = true
  const redAgain = sample()

  assert.ok(red.r > red.g + 80, `initial background texture should render red (${red.r}, ${red.g}, ${red.b})`)
  assert.ok(green.g > green.r + 80, `mutated background texture should render green (${green.r}, ${green.g}, ${green.b})`)
  assert.ok(redAgain.r > redAgain.g + 80, `background texture upload should update back to red (${redAgain.r}, ${redAgain.g}, ${redAgain.b})`)
})
