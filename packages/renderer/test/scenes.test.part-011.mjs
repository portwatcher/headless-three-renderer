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
import { assertRgbClose, getRenderer, makeCamera, meanRegion, meanScalarRegion, renderRgba } from './scenes.test.part-002.mjs'
import { makeCubeCaptureScene, makeLayeredArrayCamera, makeLayeredSplitScene } from './scenes.test.part-010.mjs'
test('malformed ArrayCamera sub-camera containers fail clearly', () => {
  const scene = makeLayeredSplitScene()
  const arrayCamera = makeLayeredArrayCamera()

  arrayCamera.cameras = 'bad'
  assert.throws(
    () => renderRgba(scene, arrayCamera, { width: 64, height: 64 }),
    /THREE\.ArrayCamera\.cameras must be an array/i,
  )

  arrayCamera.cameras = []
  assert.throws(
    () => renderRgba(scene, arrayCamera, { width: 64, height: 64 }),
    /THREE\.ArrayCamera requires at least one sub-camera/i,
  )

  arrayCamera.cameras = [null]
  assert.throws(
    () => renderRgba(scene, arrayCamera, { width: 64, height: 64 }),
    /THREE\.ArrayCamera\.cameras\[0\] must be a THREE\.Camera/i,
  )

  arrayCamera.cameras = [new THREE.ArrayCamera([])]
  assert.throws(
    () => renderRgba(scene, arrayCamera, { width: 64, height: 64 }),
    /THREE\.ArrayCamera\.cameras\[0\] cannot be a THREE\.ArrayCamera/i,
  )

  const matrixArrayCamera = makeLayeredArrayCamera()
  matrixArrayCamera.cameras[0].matrixWorldInverse.elements[1] = Number.NaN
  assert.throws(
    () => renderRgba(scene, matrixArrayCamera, { width: 64, height: 64 }),
    /THREE\.ArrayCamera\.cameras\[0\]\.matrixWorldInverse\.elements\[1\] must be a finite number/i,
  )
})

test('CubeCamera renders cube target faces', async () => {
  const scene = makeCubeCaptureScene()
  const cubeTarget = new THREE.WebGLCubeRenderTarget(32)
  const cubeCamera = new THREE.CubeCamera(0.01, 100, cubeTarget)
  assertValidPng(getRenderer().render(scene, cubeCamera, { width: 32, height: 32 }), { width: 32, height: 32 })

  cubeTarget.depthTexture = {
    type: THREE.FloatType,
    source: { data: Array.from({ length: 6 }, () => ({})) },
  }
  const returned = renderToTarget(scene, cubeCamera, cubeTarget)
  assert.equal(returned, cubeTarget)
  assert.equal(cubeTarget.texture.image.length, 6)

  const px = meanRegion(cubeTarget.texture.image[0].data, 32, 32, 12, 12, 20, 20)
  const nx = meanRegion(cubeTarget.texture.image[1].data, 32, 32, 12, 12, 20, 20)
  const py = meanRegion(cubeTarget.texture.image[2].data, 32, 32, 12, 12, 20, 20)
  const pz = meanRegion(cubeTarget.texture.image[4].data, 32, 32, 12, 12, 20, 20)
  assert.ok(px.r > px.g + 80 && px.r > px.b + 80, `+X face should capture red (${px.r}, ${px.g}, ${px.b})`)
  assert.ok(nx.g > nx.r + 60 && nx.g > nx.b + 60, `-X face should capture green (${nx.r}, ${nx.g}, ${nx.b})`)
  assert.ok(py.b > py.r + 80 && py.b > py.g + 80, `+Y face should capture blue (${py.r}, ${py.g}, ${py.b})`)
  assert.ok(pz.r > pz.g + 80 && pz.b > pz.g + 80, `+Z face should capture magenta (${pz.r}, ${pz.g}, ${pz.b})`)
  assert.notStrictEqual(cubeTarget.texture.image[0], cubeTarget.texture.image[1])
  assert.strictEqual(cubeTarget.texture.source.data, cubeTarget.texture.image)

  assert.equal(cubeTarget.depthTexture.image.length, 6)
  assert.ok(cubeTarget.depthTexture.image[0].data instanceof Float32Array, 'cube depth face should use Float32Array data')
  assert.equal(cubeTarget.depthTexture.image[0].data.length, 32 * 32)
  assert.strictEqual(cubeTarget.depthTexture.source.data, cubeTarget.depthTexture.image)
  const depthPx = meanScalarRegion(cubeTarget.depthTexture.image[0].data, 32, 32, 12, 12, 20, 20)
  assert.ok(depthPx > 0 && depthPx <= 1, `cube depth face should contain normalized depth (${depthPx})`)

  const renderer = new Renderer()
  const positiveFace = Buffer.alloc(32 * 32 * 4)
  renderer.readRenderTargetPixels(cubeTarget, 0, 0, 32, 32, positiveFace, 0)
  assert.deepEqual(positiveFace, Buffer.from(cubeTarget.texture.image[0].data))

  const negativeFaceRect = Buffer.alloc(8 * 4 * 4)
  renderer.readRenderTargetPixels(cubeTarget, 12, 12, 8, 4, negativeFaceRect, 1)
  const expectedNegativeFaceRect = Buffer.alloc(negativeFaceRect.length)
  const negativeFace = Buffer.from(cubeTarget.texture.image[1].data)
  for (let row = 0; row < 4; row += 1) {
    const sourceStart = (((12 + row) * 32) + 12) * 4
    negativeFace.copy(expectedNegativeFaceRect, row * 8 * 4, sourceStart, sourceStart + 8 * 4)
  }
  assert.deepEqual(negativeFaceRect, expectedNegativeFaceRect)

  const asyncPositiveFace = await renderer.readRenderTargetPixelsAsync(cubeTarget, 0, 0, 32, 32, undefined, 0)
  assert.ok(Buffer.isBuffer(asyncPositiveFace), 'async cube-face readback should allocate a Buffer for Buffer-backed cube faces')
  assert.deepEqual(asyncPositiveFace, Buffer.from(cubeTarget.texture.image[0].data))

  const asyncCommonFaceRect = await renderer.readRenderTargetPixelsAsync(cubeTarget, 12, 12, 8, 4, 0, 1)
  assert.ok(Buffer.isBuffer(asyncCommonFaceRect), 'common async cube-face readback should allocate a Buffer')
  assert.deepEqual(asyncCommonFaceRect, expectedNegativeFaceRect)

  assert.throws(
    () => renderer.readRenderTargetPixels(
      { texture: { image: [{ data: Buffer.alloc(4), width: 1, height: 1 }] } },
      0,
      0,
      1,
      1,
      Buffer.alloc(4),
      1,
    ),
    /target has no readable color data/i,
  )
})

test('WebGLCubeRenderTarget.clear uses Renderer target state for all faces', () => {
  const renderer = new Renderer()
  const cubeTarget = new THREE.WebGLCubeRenderTarget(8)
  const previousTarget = {
    width: 4,
    height: 4,
    texture: {
      image: {
        width: 4,
        height: 4,
        data: Buffer.alloc(4 * 4 * 4),
      },
    },
  }

  renderer.setRenderTarget(previousTarget)
  renderer.setClearColor(0x123456, 0.5)
  assert.equal(cubeTarget.clear(renderer), undefined)
  assert.equal(renderer.getRenderTarget(), previousTarget)

  for (let face = 0; face < 6; face += 1) {
    const data = Buffer.alloc(8 * 8 * 4)
    renderer.readRenderTargetPixels(cubeTarget, 0, 0, 8, 8, data, face)
    const mean = meanRgba(data)
    assertRgbClose(mean, [0x12, 0x34, 0x56], `cleared cube face ${face}`)
    assert.ok(Math.abs(mean.a - 128) <= 1, `cleared cube face ${face} should preserve clear alpha (${mean.a})`)
  }
})

test('Common CubeRenderTarget.clear uses Renderer target state for all faces', () => {
  const renderer = new Renderer()
  const cubeTarget = new CommonCubeRenderTarget(8)
  const previousTarget = { width: 4, height: 4, texture: {} }

  renderer.setRenderTarget(previousTarget)
  renderer.setClearColor(0x2a4c6e, 0.75)
  assert.equal(cubeTarget.clear(renderer), undefined)
  assert.equal(renderer.getRenderTarget(), previousTarget)

  for (let face = 0; face < 6; face += 1) {
    const data = Buffer.alloc(8 * 8 * 4)
    renderer.readRenderTargetPixels(cubeTarget, 0, 0, 8, 8, data, face)
    const mean = meanRgba(data)
    assertRgbClose(mean, [0x2a, 0x4c, 0x6e], `cleared common cube face ${face}`)
    assert.ok(Math.abs(mean.a - 191) <= 1, `cleared common cube face ${face} should preserve clear alpha (${mean.a})`)
  }

  assert.equal(cubeTarget.texture.image.length, 6)
  assert.strictEqual(cubeTarget.texture.source.data, cubeTarget.texture.image)
})

test('LightProbeGenerator reads cube targets through the WebGLRenderer marker path', async () => {
  const scene = makeCubeCaptureScene()
  const cubeTarget = new THREE.WebGLCubeRenderTarget(16)
  const cubeCamera = new THREE.CubeCamera(0.01, 100, cubeTarget)
  renderToTarget(scene, cubeCamera, cubeTarget)

  const renderer = new Renderer()
  const probe = await LightProbeGenerator.fromCubeRenderTarget(renderer, cubeTarget)
  assert.equal(probe.isLightProbe, true)
  const coefficients = probe.sh.coefficients
  assert.equal(coefficients.length, 9)
  const energy = coefficients.reduce((sum, coefficient) => (
    sum + coefficient.x ** 2 + coefficient.y ** 2 + coefficient.z ** 2
  ), 0)
  assert.ok(Number.isFinite(energy), `generated LightProbe coefficients should stay finite (${energy})`)
  assert.ok(energy > 0.01, `generated LightProbe should contain captured cube radiance (${energy})`)
})

test('LightProbeHelper shader fails clearly', () => {
  const scene = new THREE.Scene()
  const probe = new THREE.LightProbe()
  scene.add(probe)
  scene.add(new LightProbeHelper(probe, 0.6))

  assert.throws(
    () => renderRgba(scene, makeCamera(), { width: 32, height: 32 }),
    /LightProbeHelper internal LightProbeHelperMaterial ShaderMaterial.*not translated.*Native THREE\.LightProbe lighting.*LightProbeGenerator/i,
  )
})

test('TextureHelper shader fails clearly', () => {
  const scene = new THREE.Scene()
  const texture = new THREE.DataTexture(new Uint8Array([255, 0, 0, 255]), 1, 1)
  texture.needsUpdate = true
  scene.add(new TextureHelper(texture, 1, 1, 1))

  assert.throws(
    () => renderRgba(scene, makeCamera(), { width: 32, height: 32 }),
    /TextureHelper internal TextureHelperMaterial ShaderMaterial.*not translated.*supported material, background, scene\.environment.*target pixels.*custom WGSL/i,
  )
})

test('WebGLTextureUtils decompression shader path fails clearly', () => {
  const renderer = new Renderer()
  const texture = new THREE.DataTexture(new Uint8Array([255, 0, 0, 255]), 1, 1)
  texture.needsUpdate = true

  try {
    assert.throws(
      () => WebGLTextureUtils.decompress(texture, Infinity, renderer),
      /ShaderMaterial is not supported directly.*fragmentWgsl/i,
    )
  } finally {
    renderer.dispose?.()
    texture.dispose()
  }
})

test('examples WebGL and WebGPU capability helpers expose browser-context boundaries', () => {
  const hadWindow = Object.prototype.hasOwnProperty.call(globalThis, 'window')
  const previousWindow = globalThis.window
  const hadDocument = Object.prototype.hasOwnProperty.call(globalThis, 'document')
  const previousDocument = globalThis.document
  const previousWarn = console.warn
  const warnings = []

  try {
    delete globalThis.window
    delete globalThis.document
    console.warn = (message) => warnings.push(String(message))

    assert.equal(WebGL.isWebGL2Available(), false)
    assert.equal(WebGL.isColorSpaceAvailable('display-p3'), false)
    if (typeof WebGL.isWebGLAvailable === 'function') assert.equal(WebGL.isWebGLAvailable(), false)
    assert.equal(WebGPU.isAvailable(), false)
    if (typeof WebGPU.getStaticAdapter === 'function') assert.equal(WebGPU.getStaticAdapter(), false)
    if (typeof WebGL.isWebGLAvailable === 'function') {
      assert.ok(warnings.some((message) => message.includes('isWebGLAvailable() has been deprecated')))
    }

    const elements = []
    const makeElement = (tagName) => ({
      tagName,
      style: {},
      children: [],
      id: '',
      innerHTML: '',
      appendChild(child) {
        this.children.push(child)
      },
      getContext(type) {
        if (tagName !== 'canvas') return null
        if (type === 'webgl2' || type === 'webgl' || type === 'experimental-webgl') {
          return { drawingBufferColorSpace: 'srgb' }
        }
        return null
      },
    })
    globalThis.window = {
      WebGLRenderingContext: function WebGLRenderingContext() {},
      WebGL2RenderingContext: function WebGL2RenderingContext() {},
    }
    globalThis.document = {
      createElement(tagName) {
        const element = makeElement(tagName)
        elements.push(element)
        return element
      },
    }

    assert.equal(WebGL.isWebGL2Available(), true)
    assert.equal(WebGL.isColorSpaceAvailable('display-p3'), true)
    if (typeof WebGL.isWebGLAvailable === 'function') assert.equal(WebGL.isWebGLAvailable(), true)

    const webglMessage = WebGL.getWebGL2ErrorMessage()
    assert.equal(webglMessage.id, 'webglmessage')
    assert.match(webglMessage.innerHTML, /graphics card.*WebGL 2/)
    assert.equal(webglMessage.style.fontFamily, 'monospace')

    const webgpuMessage = WebGPU.getErrorMessage()
    assert.equal(webgpuMessage.id, 'webgpumessage')
    assert.match(webgpuMessage.innerHTML, /WebGPU/)
    assert.equal(webgpuMessage.style.maxWidth, '400px')
    assert.ok(elements.some((element) => element.tagName === 'canvas'))
  } finally {
    console.warn = previousWarn
    if (hadWindow) {
      globalThis.window = previousWindow
    } else {
      delete globalThis.window
    }
    if (hadDocument) {
      globalThis.document = previousDocument
    } else {
      delete globalThis.document
    }
  }
})
