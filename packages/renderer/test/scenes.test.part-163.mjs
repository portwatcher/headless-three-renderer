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
import { assertRgbClose, meanRegion, meanScalarRegion } from './scenes.test.part-002.mjs'
import { makeCubeCaptureScene } from './scenes.test.part-010.mjs'
test('Renderer copyFramebufferToTexture copies active framebuffer source rectangles on the CPU', () => {
  const renderer = new Renderer()

  function sourcePixel(x, y, width) {
    return [10 + x, 20 + y, 30 + x + y * width, 255]
  }

  function patternedTarget(width, height) {
    const data = new Uint8Array(width * height * 4)
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        data.set(sourcePixel(x, y, width), (y * width + x) * 4)
      }
    }
    return { width, height, texture: {}, data }
  }

  const target = patternedTarget(5, 4)
  renderer.setRenderTarget(target)

  const destinationData = new Uint8Array(3 * 3 * 4)
  destinationData.fill(9)
  const destination = new THREE.DataTexture(destinationData, 3, 3, THREE.RGBAFormat)
  const initialVersion = destination.version

  renderer.copyFramebufferToTexture(destination, new THREE.Vector2(1.8, 1.2))

  function pixel(x, y) {
    const offset = (y * 3 + x) * 4
    return Array.from(destination.image.data.slice(offset, offset + 4))
  }

  assert.deepEqual(pixel(0, 0), sourcePixel(1, 1, 5))
  assert.deepEqual(pixel(2, 2), sourcePixel(3, 3, 5))
  assert.ok(destination.version > initialVersion, 'destination texture should be marked dirty after floored framebuffer position copy')

  destination.image.data.fill(9)
  const versionAfterVector2Copy = destination.version
  renderer.copyFramebufferToTexture(destination, new THREE.Vector4(2, 0, 2, 2))

  assert.deepEqual(pixel(0, 0), sourcePixel(2, 0, 5))
  assert.deepEqual(pixel(1, 1), sourcePixel(3, 1, 5))
  assert.deepEqual(pixel(2, 2), [9, 9, 9, 9])
  assert.ok(destination.version > versionAfterVector2Copy, 'destination texture should be marked dirty after framebuffer rectangle copy')

  const mipData = new Uint8Array(2 * 2 * 4)
  mipData.fill(11)
  destination.mipmaps = [{ data: mipData, width: 2, height: 2 }]
  const versionAfterRectangleCopy = destination.version
  renderer.copyFramebufferToTexture(destination, [0, 2], 1)

  function mipPixel(x, y) {
    const offset = (y * 2 + x) * 4
    return Array.from(destination.mipmaps[0].data.slice(offset, offset + 4))
  }

  assert.deepEqual(mipPixel(0, 0), sourcePixel(0, 2, 5))
  assert.deepEqual(mipPixel(1, 1), sourcePixel(1, 3, 5))
  assert.ok(destination.version > versionAfterRectangleCopy, 'destination texture should be marked dirty after framebuffer mip copy')

  mipData.fill(12)
  const versionAfterMipCopy = destination.version
  renderer.copyFramebufferToTexture([1, 1], destination, 1)

  assert.deepEqual(mipPixel(0, 0), sourcePixel(1, 1, 5))
  assert.deepEqual(mipPixel(1, 1), sourcePixel(2, 2, 5))
  assert.ok(destination.version > versionAfterMipCopy, 'destination texture should be marked dirty after legacy framebuffer mip copy')

  const rawDestinationData = new Uint8Array(2 * 2 * 4)
  rawDestinationData.fill(6)
  const rawDestination = { image: { data: rawDestinationData, width: 2, height: 2 } }
  renderer.copyFramebufferToTexture(rawDestination, [2, 0])

  function rawPixel(x, y) {
    const offset = (y * 2 + x) * 4
    return Array.from(rawDestination.image.data.slice(offset, offset + 4))
  }

  assert.deepEqual(rawPixel(0, 0), sourcePixel(2, 0, 5))
  assert.deepEqual(rawPixel(1, 1), sourcePixel(3, 1, 5))
  assert.equal(rawDestination.needsUpdate, true, 'plain raw destination texture should still be marked dirty')

  const sourceBackedDestinationData = new Uint8Array(2 * 2 * 4)
  sourceBackedDestinationData.fill(7)
  const sourceBackedDestination = new THREE.Texture()
  sourceBackedDestination.source = {
    data: {
      data: sourceBackedDestinationData,
      width: 2,
      height: 2,
    },
  }
  const sourceBackedVersion = sourceBackedDestination.version
  renderer.copyFramebufferToTexture(sourceBackedDestination, { x: 1, y: 0, width: 2, height: 2 })

  function sourceBackedPixel(x, y) {
    const offset = (y * 2 + x) * 4
    return Array.from(sourceBackedDestination.source.data.data.slice(offset, offset + 4))
  }

  assert.deepEqual(sourceBackedPixel(0, 0), sourcePixel(1, 0, 5))
  assert.deepEqual(sourceBackedPixel(1, 1), sourcePixel(2, 1, 5))
  assert.ok(sourceBackedDestination.version > sourceBackedVersion, 'source-backed destination texture should be marked dirty after framebuffer copy')

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  ))
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)
  const actualTarget = new THREE.WebGLRenderTarget(8, 6)
  renderToTarget(scene, camera, actualTarget, { outputColorSpace: THREE.LinearSRGBColorSpace })
  renderer.setRenderTarget(actualTarget)

  const actualDestinationData = new Uint8Array(3 * 2 * 4)
  actualDestinationData.fill(7)
  const actualDestination = new THREE.DataTexture(actualDestinationData, 3, 2, THREE.RGBAFormat)
  const actualDestinationVersion = actualDestination.version
  renderer.copyFramebufferToTexture(actualDestination, { x: 2, y: 1, width: 3, height: 2 })

  const expectedActualCopy = new Uint8Array(3 * 2 * 4)
  for (let row = 0; row < 2; row += 1) {
    const sourceStart = (((1 + row) * 8) + 2) * 4
    expectedActualCopy.set(actualTarget.data.subarray(sourceStart, sourceStart + 3 * 4), row * 3 * 4)
  }
  assert.deepEqual(actualDestination.image.data, expectedActualCopy)
  assert.ok(actualDestination.version > actualDestinationVersion, 'destination texture should be marked dirty after actual render target framebuffer copy')
  renderer.setRenderTarget(null)
})

test('Renderer copyFramebufferToTexture copies active cube mip face data on the CPU', () => {
  const scene = makeCubeCaptureScene()
  const cubeTarget = new THREE.WebGLCubeRenderTarget(32)
  const cubeCamera = new THREE.CubeCamera(0.01, 100, cubeTarget)
  cubeCamera.activeMipmapLevel = 1
  renderToTarget(scene, cubeCamera, cubeTarget)

  const renderer = new Renderer()
  renderer.setRenderTarget(cubeTarget, 1, 1)
  const destinationData = new Uint8Array(16 * 16 * 4)
  destinationData.fill(3)
  const destination = new THREE.DataTexture(destinationData, 16, 16, THREE.RGBAFormat)
  const initialVersion = destination.version

  renderer.copyFramebufferToTexture(destination)

  const copied = meanRegion(destination.image.data, 16, 16, 5, 5, 11, 11)
  assert.ok(copied.g > copied.r + 30 && copied.g > copied.b + 50, `active cube mip face copy should read the selected -X mip face (${copied.r}, ${copied.g}, ${copied.b})`)
  assert.ok(destination.version > initialVersion, 'destination texture should be marked dirty after cube mip framebuffer copy')
  renderer.setRenderTarget(null)
})

test('Renderer clear honors active render target scissor state', () => {
  const renderer = new Renderer()
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  ))
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)

  const target = { texture: {} }
  renderer.setRenderTarget(target)
  renderer.render(scene, camera, {
    width: 32,
    height: 32,
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  target.scissorTest = true
  target.scissor = { x: 16, y: 0, width: 16, height: 32 }
  renderer.setClearColor(0x00ff00, 1)
  renderer.clear()

  const targetLeft = meanRegion(target.data, 32, 32, 4, 10, 12, 22)
  const targetRight = meanRegion(target.data, 32, 32, 20, 10, 28, 22)
  assert.ok(targetLeft.r > targetLeft.g + 80, `target scissor clear should preserve red outside the rectangle (${targetLeft.r}, ${targetLeft.g})`)
  assert.ok(targetRight.g > targetRight.r + 80, `target scissor clear should write green inside the rectangle (${targetRight.r}, ${targetRight.g})`)

  target.scissorTest = false
  renderer.setScissor(0, 0, 16, 32)
  renderer.setScissorTest(true)
  renderer.setClearColor(0x0000ff, 1)
  renderer.clearColor()

  const rendererLeft = meanRegion(target.data, 32, 32, 4, 10, 12, 22)
  const rendererRight = meanRegion(target.data, 32, 32, 20, 10, 28, 22)
  assert.ok(rendererLeft.b > rendererLeft.r + 80, `renderer scissor clear should write blue inside fallback rectangle (${rendererLeft.r}, ${rendererLeft.b})`)
  assert.ok(rendererRight.g > rendererRight.b + 80, `renderer scissor clear should preserve green outside fallback rectangle (${rendererRight.g}, ${rendererRight.b})`)

  renderer.setScissorTest(false)
  renderer.setRenderTarget(null)
})

test('Renderer clear writes actual Three.js RenderTarget classes', () => {
  const renderer = new Renderer()
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  ))
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)

  const clearTarget = new THREE.RenderTarget(16, 10)
  const clearTextureVersion = clearTarget.texture.version
  const clearSourceVersion = clearTarget.texture.source.version
  renderer.setClearColor(0x204080, 0.5)
  renderer.clearTarget(clearTarget, true, false, false)

  assert.equal(clearTarget.data.length, 16 * 10 * 4)
  assert.strictEqual(clearTarget.texture.source.data, clearTarget.texture.image)
  assert.strictEqual(clearTarget.texture.image.data, clearTarget.data)
  assert.strictEqual(clearTarget.texture.source.data.data, clearTarget.data)
  assert.ok(clearTarget.texture.version > clearTextureVersion, 'clearTarget should mark the Three texture version dirty')
  assert.ok(clearTarget.texture.source.version > clearSourceVersion, 'clearTarget should mark the Three source version dirty')
  const cleared = meanRegion(clearTarget.data, 16, 10, 4, 2, 12, 8)
  assertRgbClose(cleared, [0x20, 0x40, 0x80], 'clearTarget should write clear color into THREE.RenderTarget')
  assert.ok(Math.abs(cleared.a - 128) <= 1, `clearTarget should write clear alpha (${cleared.a})`)

  const activeTarget = new THREE.WebGLRenderTarget(32, 32)
  renderer.setRenderTarget(activeTarget)
  renderer.render(scene, camera, {
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  activeTarget.scissorTest = true
  activeTarget.scissor = { x: 16, y: 0, width: 16, height: 32 }
  const activeTextureVersion = activeTarget.texture.version
  const activeSourceVersion = activeTarget.texture.source.version
  renderer.setClearColor(0x00ff00, 1)
  renderer.clearColor()

  const activeLeft = meanRegion(activeTarget.data, 32, 32, 4, 10, 12, 22)
  const activeRight = meanRegion(activeTarget.data, 32, 32, 20, 10, 28, 22)
  assert.ok(activeLeft.r > activeLeft.g + 80, `active THREE.WebGLRenderTarget clear should preserve red outside scissor (${activeLeft.r}, ${activeLeft.g})`)
  assert.ok(activeRight.g > activeRight.r + 80, `active THREE.WebGLRenderTarget clear should write green inside scissor (${activeRight.r}, ${activeRight.g})`)
  assert.strictEqual(activeTarget.texture.source.data, activeTarget.texture.image)
  assert.strictEqual(activeTarget.texture.image.data, activeTarget.data)
  assert.ok(activeTarget.texture.version > activeTextureVersion, 'active clear should mark the Three texture version dirty')
  assert.ok(activeTarget.texture.source.version > activeSourceVersion, 'active clear should mark the Three source version dirty')

  renderer.setRenderTarget(null)
})

test('Renderer clear writes active render target depth textures', () => {
  const renderer = new Renderer()
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  ))
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)

  const depthTexture = { type: THREE.FloatType, source: { data: {} } }
  const target = { texture: {}, depthTexture }
  renderer.setRenderTarget(target)
  renderer.render(scene, camera, {
    width: 32,
    height: 32,
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  renderer.setClearColor(0x00ff00, 1)
  renderer.setClearDepth(0.25)
  renderer.clear(false, true, false)

  const colorMean = meanRegion(target.data, 32, 32, 10, 10, 22, 22)
  const depthMean = meanScalarRegion(depthTexture.image.data, 32, 32, 10, 10, 22, 22)
  assert.ok(colorMean.r > colorMean.g + 80, `depth-only clear should preserve color output (${colorMean.r}, ${colorMean.g})`)
  assert.ok(Math.abs(depthMean - 0.25) < 1e-6, `depth-only clear should write configured clear depth (${depthMean})`)
  assert.equal(depthTexture.source.data.data, depthTexture.image.data)
  assert.equal(depthTexture.needsUpdate, true)

  renderer.setClearDepth(0.75)
  renderer.clearDepth()
  const helperDepthMean = meanScalarRegion(depthTexture.image.data, 32, 32, 10, 10, 22, 22)
  const helperColorMean = meanRegion(target.data, 32, 32, 10, 10, 22, 22)
  assert.ok(Math.abs(helperDepthMean - 0.75) < 1e-6, `clearDepth should write configured clear depth (${helperDepthMean})`)
  assert.ok(helperColorMean.r > helperColorMean.g + 80, `clearDepth should preserve color output (${helperColorMean.r}, ${helperColorMean.g})`)

  renderer.setRenderTarget(null)
})
