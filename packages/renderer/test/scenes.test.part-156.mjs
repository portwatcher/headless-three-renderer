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
import { Renderer, extractCommonBackendMethodNames, extractCommonRendererNodeSurfaceNames, extractJsClassSurfaceNames, extractJsFunctionReturnSurfaceNames, extractJsFunctionThisSurfaceNames, extractWebGlStateBufferMethodNames, extractWebGlStateMethodNames, objectSurfaceNames, test } from './scenes.test.part-001.mjs'
import { makeCamera, meanRegion } from './scenes.test.part-002.mjs'
test('Renderer domElement is an inert output-size mirror', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 1)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(4, 4),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  ))
  const camera = makeCamera()
  const renderer = new Renderer()

  assert.equal(renderer.domElement.width, 0)
  assert.equal(renderer.domElement.height, 0)
  assert.equal(renderer.domElement.clientWidth, 0)
  assert.equal(renderer.domElement.clientHeight, 0)
  assert.equal(renderer.domElement.offsetWidth, 0)
  assert.equal(renderer.domElement.offsetHeight, 0)
  assert.deepEqual(renderer.domElement.getBoundingClientRect(), {
    x: 0,
    y: 0,
    width: 0,
    height: 0,
    top: 0,
    right: 0,
    bottom: 0,
    left: 0,
  })
  assert.deepEqual(renderer.domElement.style, { width: '0px', height: '0px' })
  assert.throws(
    () => renderer.domElement.getContext('webgl'),
    /Renderer\.domElement\.getContext\(\) is not supported.*inert offscreen compatibility object/i,
  )
  assert.throws(
    () => renderer.domElement.toDataURL('image/png'),
    /Renderer\.domElement\.toDataURL\(\) is not supported.*Renderer\.render\(\).*PNG Buffer/i,
  )
  assert.throws(
    () => renderer.domElement.toBlob(() => {}, 'image/png'),
    /Renderer\.domElement\.toBlob\(\) is not supported.*Renderer\.render\(\).*PNG Buffer/i,
  )
  assert.throws(
    () => renderer.domElement.toBlob(null),
    /Renderer\.domElement\.toBlob callback must be a function/i,
  )
  assert.throws(
    () => renderer.domElement.captureStream(),
    /Renderer\.domElement\.captureStream\(\) is not supported.*inert offscreen compatibility object/i,
  )
  assert.throws(
    () => renderer.domElement.transferToImageBitmap(),
    /Renderer\.domElement\.transferToImageBitmap\(\) is not supported.*not an OffscreenCanvas/i,
  )

  renderer.setSize(64, 32)
  assert.equal(renderer.domElement.width, 64)
  assert.equal(renderer.domElement.height, 32)
  assert.equal(renderer.domElement.clientWidth, 64)
  assert.equal(renderer.domElement.clientHeight, 32)
  assert.equal(renderer.domElement.offsetWidth, 64)
  assert.equal(renderer.domElement.offsetHeight, 32)
  assert.deepEqual(renderer.domElement.getBoundingClientRect(), {
    x: 0,
    y: 0,
    width: 64,
    height: 32,
    top: 0,
    right: 64,
    bottom: 32,
    left: 0,
  })
  assert.deepEqual(renderer.domElement.style, { width: '64px', height: '32px' })

  renderer.setSize(48, 24, false)
  assert.equal(renderer.domElement.width, 48)
  assert.equal(renderer.domElement.height, 24)
  assert.equal(renderer.domElement.clientWidth, 64)
  assert.equal(renderer.domElement.clientHeight, 32)
  assert.deepEqual(renderer.domElement.style, { width: '64px', height: '32px' })

  renderer.setDrawingBufferSize(40, 20, 2)
  assert.equal(renderer.domElement.width, 40)
  assert.equal(renderer.domElement.height, 20)
  assert.equal(renderer.domElement.clientWidth, 40)
  assert.equal(renderer.domElement.clientHeight, 20)
  assert.deepEqual(renderer.domElement.style, { width: '40px', height: '20px' })
  assert.equal(renderer.getPixelRatio(), 2)
  renderer.domElement.style.width = '12.5px'
  renderer.domElement.style.height = 'invalid'
  assert.equal(renderer.domElement.clientWidth, 13)
  assert.equal(renderer.domElement.clientHeight, 20)
  assert.equal(renderer.domElement.offsetWidth, 13)
  assert.equal(renderer.domElement.offsetHeight, 20)
  renderer.domElement.style.width = '40px'
  renderer.domElement.style.height = '20px'
  assert.equal(renderer.domElement.style.getPropertyValue('width'), '40px')
  renderer.domElement.style.setProperty('width', '12.5px')
  assert.equal(renderer.domElement.clientWidth, 13)
  assert.equal(renderer.domElement.style.removeProperty('width'), '12.5px')
  assert.equal(renderer.domElement.clientWidth, 40)
  renderer.domElement.style.setProperty('width', '40px')
  renderer.domElement.style.setProperty('touch-action', 'none')
  assert.equal(renderer.domElement.style.touchAction, 'none')
  assert.equal(renderer.domElement.style.getPropertyValue('touch-action'), 'none')
  assert.equal(renderer.domElement.style.removeProperty('touch-action'), 'none')
  assert.equal(renderer.domElement.style.getPropertyValue('touch-action'), '')
  renderer.domElement.style.setProperty('--renderer-mode', 'headless')
  assert.equal(renderer.domElement.style.getPropertyValue('--renderer-mode'), 'headless')
  assert.equal(renderer.domElement.style.removeProperty('--renderer-mode'), 'headless')
  assert.deepEqual(renderer.domElement.style, { width: '40px', height: '20px' })

  assert.equal(renderer.domElement.getAttribute('data-renderer'), null)
  assert.equal(renderer.domElement.hasAttribute('data-renderer'), false)
  renderer.domElement.setAttribute('data-renderer', 'headless')
  renderer.domElement.setAttribute('data-count', 3)
  assert.equal(renderer.domElement.getAttribute('data-renderer'), 'headless')
  assert.equal(renderer.domElement.getAttribute('data-count'), '3')
  assert.equal(renderer.domElement.hasAttribute('data-renderer'), true)
  renderer.domElement.removeAttribute('data-renderer')
  assert.equal(renderer.domElement.getAttribute('data-renderer'), null)
  assert.equal(renderer.domElement.hasAttribute('data-renderer'), false)
  assert.throws(
    () => renderer.domElement.setAttribute('', 'value'),
    /Renderer\.domElement\.setAttribute name must be a non-empty string/i,
  )
  assert.throws(
    () => renderer.domElement.getAttribute(null),
    /Renderer\.domElement\.getAttribute name must be a non-empty string/i,
  )
  assert.throws(
    () => renderer.domElement.style.setProperty('', 'none'),
    /Renderer\.domElement\.style\.setProperty propertyName must be a non-empty string/i,
  )
  assert.throws(
    () => renderer.domElement.style.getPropertyValue(null),
    /Renderer\.domElement\.style\.getPropertyValue propertyName must be a non-empty string/i,
  )
  assert.throws(
    () => renderer.domElement.style.removeProperty('set-property'),
    /Renderer\.domElement\.style\.removeProperty propertyName must not name a reserved style method/i,
  )

  const eventCalls = []
  function onContextLost(event) {
    eventCalls.push([event.type, this === renderer.domElement])
  }
  renderer.domElement.addEventListener('webglcontextlost', onContextLost)
  assert.equal(renderer.domElement.dispatchEvent({ type: 'webglcontextlost' }), true)
  renderer.domElement.removeEventListener('webglcontextlost', onContextLost)
  assert.equal(renderer.domElement.dispatchEvent({ type: 'webglcontextlost' }), true)
  assert.deepEqual(eventCalls, [['webglcontextlost', true]])
  assert.throws(
    () => renderer.domElement.addEventListener('', onContextLost),
    /Renderer\.domElement\.addEventListener type must be a non-empty string/i,
  )
  assert.throws(
    () => renderer.domElement.removeEventListener('webglcontextlost', null),
    /Renderer\.domElement\.removeEventListener listener must be a function/i,
  )
  assert.throws(
    () => renderer.domElement.dispatchEvent({ type: '' }),
    /Renderer\.domElement\.dispatchEvent event\.type must be a non-empty string/i,
  )
  assert.throws(
    () => renderer.domElement.dispatchEvent(null),
    /Renderer\.domElement\.dispatchEvent event must be an event-like object/i,
  )

  const rgba = renderer.render(scene, camera, { format: 'rgba' })
  assert.equal(rgba.length, 40 * 20 * 4)
  const mean = meanRegion(rgba, 40, 20, 12, 6, 28, 14)
  assert.ok(mean.r > mean.b + 80, `domElement size mirroring should preserve normal rendering (${mean.r} vs ${mean.b})`)
})

test('Renderer.backend tracks the installed CommonRenderer Backend method surface', () => {
  const renderer = new Renderer()
  const backendMethods = extractCommonBackendMethodNames()
  assert.ok(backendMethods.size > 40, 'Expected to find installed Three.js CommonRenderer Backend methods.')

  const missingBackendMethods = [...backendMethods]
    .filter((methodName) => !objectSurfaceNames(renderer.backend).has(methodName))
    .sort()
  assert.deepEqual(
    missingBackendMethods,
    [],
    `Renderer.backend is missing installed Three.js CommonRenderer Backend methods: ${missingBackendMethods.join(', ')}`,
  )
})

test('Renderer.state tracks the installed WebGLState method surface', () => {
  const renderer = new Renderer()
  const webGlStateMethods = extractWebGlStateMethodNames()
  assert.ok(webGlStateMethods.size > 20, 'Expected to find installed Three.js WebGLState methods.')

  const missingStateMethods = [...webGlStateMethods]
    .filter((methodName) => !objectSurfaceNames(renderer.state).has(methodName))
    .sort()
  assert.deepEqual(missingStateMethods, [], `Renderer.state is missing installed Three.js WebGLState methods: ${missingStateMethods.join(', ')}`)

  for (const [label, functionName, stateBuffer] of [
    ['color', 'ColorBuffer', renderer.state.buffers.color],
    ['depth', 'DepthBuffer', renderer.state.buffers.depth],
    ['stencil', 'StencilBuffer', renderer.state.buffers.stencil],
  ]) {
    const webGlBufferMethods = extractWebGlStateBufferMethodNames(functionName)
    assert.ok(webGlBufferMethods.size > 3, `Expected to find installed Three.js WebGLState ${label} buffer methods.`)
    const missingBufferMethods = [...webGlBufferMethods]
      .filter((methodName) => !objectSurfaceNames(stateBuffer).has(methodName))
      .sort()
    assert.deepEqual(
      missingBufferMethods,
      [],
      `Renderer.state.buffers.${label} is missing installed Three.js WebGLState ${label} buffer methods: ${missingBufferMethods.join(', ')}`,
    )
  }
})

test('Renderer render lists and node registries track installed CommonRenderer surfaces', () => {
  const renderer = new Renderer()
  const scene = new THREE.Scene()
  const camera = makeCamera()
  const renderList = renderer.renderLists.get(scene, camera)

  for (const [label, names, actual, minimum] of [
    ['nodes', extractCommonRendererNodeSurfaceNames(), renderer.nodes, 20],
    ['library', extractJsClassSurfaceNames('src/renderers/common/nodes/NodeLibrary.js', 'NodeLibrary'), renderer.library, 8],
    ['lighting', extractJsClassSurfaceNames('src/renderers/common/Lighting.js', 'Lighting'), renderer.lighting, 2],
    ['renderLists', extractJsClassSurfaceNames('src/renderers/common/RenderLists.js', 'RenderLists'), renderer.renderLists, 4],
    ['renderList', extractJsClassSurfaceNames('src/renderers/common/RenderList.js', 'RenderList'), renderList, 10],
  ]) {
    assert.ok(names.size >= minimum, `Expected to find installed Three.js CommonRenderer ${label} surface names.`)
    const missing = [...names]
      .filter((name) => !objectSurfaceNames(actual).has(name))
      .sort()
    assert.deepEqual(missing, [], `Renderer ${label} is missing installed Three.js CommonRenderer ${label} names: ${missing.join(', ')}`)
  }

  assert.equal(renderer.renderLists.lists.get([scene, camera]), renderList)
  const manualScene = {}
  const manualCamera = {}
  const manualList = { id: 'manual' }
  assert.strictEqual(renderer.renderLists.lists.set([manualScene, manualCamera], manualList), renderer.renderLists.lists)
  assert.equal(renderer.renderLists.lists.get([manualScene, manualCamera]), manualList)
  assert.equal(renderer.renderLists.get(manualScene, manualCamera), manualList)
  assert.equal(renderer.renderLists.lists.delete([manualScene, manualCamera]), true)
  assert.equal(renderer.renderLists.lists.get([manualScene, manualCamera]), undefined)
  assert.equal(renderer.renderLists.lists.delete([manualScene, manualCamera]), false)
  assert.throws(
    () => renderer.renderLists.lists.get([]),
    /Renderer\.renderLists\.lists\.get keys must be a non-empty array of objects/i,
  )
  assert.throws(
    () => renderer.renderLists.lists.set([scene, null], {}),
    /Renderer\.renderLists\.lists\.set keys\[1\] must be an object/i,
  )
  assert.throws(
    () => renderer.renderLists.lists.delete('keys'),
    /Renderer\.renderLists\.lists\.delete keys must be a non-empty array of objects/i,
  )
})

test('Renderer helper objects track installed WebGL and WebXR helper surfaces', () => {
  const renderer = new Renderer()
  const scene = new THREE.Scene()
  const renderState = renderer.renderStates.get(scene, 0)

  for (const [label, names, actual, minimum] of [
    ['extensions', extractJsFunctionReturnSurfaceNames('src/renderers/webgl/WebGLExtensions.js', 'WebGLExtensions'), renderer.extensions, 3],
    ['capabilities', extractJsFunctionReturnSurfaceNames('src/renderers/webgl/WebGLCapabilities.js', 'WebGLCapabilities'), renderer.capabilities, 10],
    ['properties', extractJsFunctionReturnSurfaceNames('src/renderers/webgl/WebGLProperties.js', 'WebGLProperties'), renderer.properties, 4],
    ['renderStates', extractJsFunctionReturnSurfaceNames('src/renderers/webgl/WebGLRenderStates.js', 'WebGLRenderStates'), renderer.renderStates, 2],
    ['renderState', extractJsFunctionReturnSurfaceNames('src/renderers/webgl/WebGLRenderStates.js', 'WebGLRenderState'), renderState, 5],
    ['shadowMap', extractJsFunctionThisSurfaceNames('src/renderers/webgl/WebGLShadowMap.js', 'WebGLShadowMap'), renderer.shadowMap, 5],
    ['xr', extractJsClassSurfaceNames('src/renderers/webxr/WebXRManager.js', 'WebXRManager'), renderer.xr, 20],
  ]) {
    assert.ok(names.size >= minimum, `Expected to find installed Three.js ${label} helper surface names.`)
    const missing = [...names]
      .filter((name) => !objectSurfaceNames(actual).has(name))
      .sort()
    assert.deepEqual(missing, [], `Renderer ${label} is missing installed Three.js ${label} helper names: ${missing.join(', ')}`)
  }
})
