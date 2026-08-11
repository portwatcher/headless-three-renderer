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
import { BG, Renderer, render, renderToTarget, test } from './scenes.test.part-001.mjs'
import { assertRgbClose, makeCamera, meanAbsDiff, meanRegion, meanScalarRegion, objectIdBytes, renderRgba } from './scenes.test.part-002.mjs'
import { makeCubeCaptureScene } from './scenes.test.part-010.mjs'
test('CubeCamera renders active mip target faces', () => {
  const scene = makeCubeCaptureScene()
  const cubeTarget = new THREE.WebGLCubeRenderTarget(32)
  const cubeCamera = new THREE.CubeCamera(0.01, 100, cubeTarget)

  renderToTarget(scene, cubeCamera, cubeTarget)
  const basePositiveX = cubeTarget.texture.image[0]

  cubeCamera.activeMipmapLevel = 1
  cubeTarget.depthTexture = { type: THREE.UnsignedShortType, mipmaps: [] }
  const returned = renderToTarget(scene, cubeCamera, cubeTarget)
  assert.equal(returned, cubeTarget)
  assert.equal(cubeTarget.width, 32)
  assert.equal(cubeTarget.height, 32)
  assert.equal(cubeTarget.data.length, 16 * 16 * 4)
  assert.strictEqual(cubeTarget.texture.image[0], basePositiveX)
  assert.equal(cubeTarget.texture.image[0].data.length, 32 * 32 * 4)

  const mip = cubeTarget.texture.mipmaps[1]
  assert.equal(mip.image.length, 6)
  assert.equal(mip.image[0].width, 16)
  assert.equal(mip.image[0].height, 16)
  assert.equal(mip.image[0].data.length, 16 * 16 * 4)
  assert.ok(cubeTarget.texture.pmremVersion > 0, 'cube target texture should request PMREM refresh')

  const px = meanRegion(mip.image[0].data, 16, 16, 5, 5, 11, 11)
  assert.ok(px.r > px.g + 80 && px.r > px.b + 80, `+X mip face should capture red (${px.r}, ${px.g}, ${px.b})`)

  const depthMip = cubeTarget.depthTexture.mipmaps[1]
  assert.equal(depthMip.image.length, 6)
  assert.ok(depthMip.image[0].data instanceof Uint16Array, 'cube depth mip face should use Uint16Array data')
  assert.equal(depthMip.image[0].data.length, 16 * 16)
  const depthPx = meanScalarRegion(depthMip.image[0].data, 16, 16, 5, 5, 11, 11)
  assert.ok(depthPx > 0, `cube depth mip face should contain scalar depth (${depthPx})`)
})

test('CubeCamera active mip target faces honor scissor clipping', () => {
  const scene = makeCubeCaptureScene()
  const cubeTarget = new THREE.WebGLCubeRenderTarget(32)
  cubeTarget.scissorTest = true
  cubeTarget.scissor = { x: 4, y: 4, width: 16, height: 16 }
  const cubeCamera = new THREE.CubeCamera(0.01, 100, cubeTarget)
  cubeCamera.activeMipmapLevel = 1

  renderToTarget(scene, cubeCamera, cubeTarget)

  const face = cubeTarget.texture.mipmaps[1].image[0].data
  const inside = meanRegion(face, 16, 16, 6, 6, 10, 10)
  const outside = meanRegion(face, 16, 16, 0, 6, 3, 10)
  assert.ok(inside.r > 180 && inside.g < 40 && inside.b < 40, `cube scissor should keep red inside the active mip rectangle (${inside.r}, ${inside.g}, ${inside.b})`)
  assert.ok(outside.r < 20 && outside.g < 20 && outside.b < 20, `cube scissor should leave pixels outside the active mip rectangle clear (${outside.r}, ${outside.g}, ${outside.b})`)
})

test('Renderer clear preserves active cube mip face scissor data', () => {
  const scene = makeCubeCaptureScene()
  const cubeTarget = new THREE.WebGLCubeRenderTarget(32)
  cubeTarget.depthTexture = { type: THREE.FloatType, mipmaps: [] }
  const cubeCamera = new THREE.CubeCamera(0.01, 100, cubeTarget)
  cubeCamera.activeMipmapLevel = 1

  renderToTarget(scene, cubeCamera, cubeTarget)

  const depthBefore = meanScalarRegion(cubeTarget.depthTexture.mipmaps[1].image[1].data, 16, 16, 10, 6, 14, 10)
  cubeTarget.scissorTest = true
  cubeTarget.scissor = { x: 0, y: 0, width: 16, height: 32 }
  const renderer = new Renderer()
  renderer.setClearColor(0x0000ff, 1)
  renderer.setClearDepth(0.25)
  renderer.setRenderTarget(cubeTarget, 1, 1)
  renderer.clear(true, true, false)

  const negativeFace = cubeTarget.texture.mipmaps[1].image[1].data
  const clearedColor = meanRegion(negativeFace, 16, 16, 2, 6, 6, 10)
  const preservedColor = meanRegion(negativeFace, 16, 16, 10, 6, 14, 10)
  assert.ok(clearedColor.b > clearedColor.r + 80 && clearedColor.b > clearedColor.g + 80, `cube clear should write blue inside the active mip face scissor (${clearedColor.r}, ${clearedColor.g}, ${clearedColor.b})`)
  assert.ok(preservedColor.g > preservedColor.r + 30 && preservedColor.g > preservedColor.b + 50, `cube clear should preserve the active face outside scissor (${preservedColor.r}, ${preservedColor.g}, ${preservedColor.b})`)

  const negativeDepth = cubeTarget.depthTexture.mipmaps[1].image[1].data
  const clearedDepth = meanScalarRegion(negativeDepth, 16, 16, 2, 6, 6, 10)
  const preservedDepth = meanScalarRegion(negativeDepth, 16, 16, 10, 6, 14, 10)
  assert.ok(Math.abs(clearedDepth - 0.25) < 1e-6, `cube clear should write depth inside the active mip face scissor (${clearedDepth})`)
  assert.ok(Math.abs(preservedDepth - depthBefore) < 1e-6, `cube clear should preserve depth outside the active face scissor (${preservedDepth} vs ${depthBefore})`)

  const positiveFace = meanRegion(cubeTarget.texture.mipmaps[1].image[0].data, 16, 16, 5, 5, 11, 11)
  assert.ok(positiveFace.r > positiveFace.g + 80 && positiveFace.r > positiveFace.b + 80, `cube clear should not alter other active mip faces (${positiveFace.r}, ${positiveFace.g}, ${positiveFace.b})`)
  renderer.setRenderTarget(null)

  const packedDepth = Math.round(0.5 * 0xffffff) * 0x100
  const packedFaces = Array.from({ length: 6 }, () => ({
    data: new Uint32Array(16 * 16).fill(packedDepth + 5),
    width: 16,
    height: 16,
  }))
  const packedCubeTarget = {
    width: 32,
    height: 32,
    texture: { isCubeTexture: true, image: Array.from({ length: 6 }, () => ({ width: 32, height: 32 })) },
    depthTexture: {
      isCubeTexture: true,
      type: THREE.UnsignedInt248Type,
      format: THREE.DepthStencilFormat,
      mipmaps: [{}, { image: packedFaces }],
    },
  }
  renderer.setClearStencil(11)
  renderer.setRenderTarget(packedCubeTarget, 2, 1)
  renderer.clearStencil()
  const clearedPackedFace = packedCubeTarget.depthTexture.mipmaps[1].image[2].data
  assert.equal(clearedPackedFace[0] & 0xff, 11, 'packed cube mip stencil clear should write the active face')
  assert.equal(Math.floor(clearedPackedFace[0] / 0x100) * 0x100, packedDepth, 'packed cube mip stencil clear should preserve active face depth bits')
  assert.equal(packedCubeTarget.depthTexture.mipmaps[1].image[0].data[0] & 0xff, 5, 'packed cube mip stencil clear should not alter inactive faces')
  renderer.setRenderTarget(null)
})

test('CubeCamera captured target textures can be reused as cube inputs', () => {
  const captureTarget = {}
  const cubeCamera = new THREE.CubeCamera(0.01, 100, new THREE.WebGLCubeRenderTarget(32))
  renderToTarget(makeCubeCaptureScene(), cubeCamera, captureTarget, { width: 32, height: 32 })
  assert.equal(captureTarget.texture.isCubeTexture, true)
  assert.strictEqual(captureTarget.texture.source.data, captureTarget.texture.image)

  const backgroundScene = new THREE.Scene()
  backgroundScene.background = captureTarget.texture
  const backgroundCamera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  backgroundCamera.position.set(0, 0, 0)
  backgroundCamera.lookAt(new THREE.Vector3(1, 0, 0))
  const background = meanRegion(renderRgba(backgroundScene, backgroundCamera, {
    width: 64,
    height: 64,
    outputColorSpace: THREE.LinearSRGBColorSpace,
  }), 64, 64, 28, 28, 36, 36)
  assert.ok(background.r > background.g + 80 && background.r > background.b + 80, `captured +X cube background should render red (${background.r}, ${background.g}, ${background.b})`)

  function makeEnvironmentScene(environment) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    if (environment) {
      scene.environment = environment
      scene.environmentIntensity = 4
    }
    scene.add(new THREE.Mesh(
      new THREE.SphereGeometry(1, 32, 32),
      new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0.2 }),
    ))
    return scene
  }

  const environmentCamera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  environmentCamera.position.set(0, 0, 3)
  environmentCamera.lookAt(0, 0, 0)
  const noEnvironment = renderRgba(makeEnvironmentScene(null), environmentCamera, { width: 64, height: 64 })
  const withEnvironment = renderRgba(makeEnvironmentScene(captureTarget.texture), environmentCamera, { width: 64, height: 64 })
  const diff = meanAbsDiff(noEnvironment, withEnvironment)
  assert.ok(diff > 1, `captured cube environment should affect metallic IBL, diff=${diff.toFixed(3)}`)
})

test('CubeCamera malformed render targets fail clearly', () => {
  const scene = makeCubeCaptureScene()
  const cubeCamera = new THREE.CubeCamera(0.01, 100, new THREE.WebGLCubeRenderTarget(32))

  cubeCamera.renderTarget = 'bad'
  assert.throws(
    () => renderRgba(scene, cubeCamera, { width: 32, height: 32 }),
    /THREE\.CubeCamera renderTarget must be a target-like object/i,
  )

  assert.throws(
    () => render(scene, cubeCamera, { width: 32, height: 32, target: 'bad' }),
    /options\.target must be a target-like object/i,
  )

  const colorMipTarget = new THREE.WebGLCubeRenderTarget(32)
  const colorMipCamera = new THREE.CubeCamera(0.01, 100, colorMipTarget)
  colorMipCamera.activeMipmapLevel = 1
  colorMipTarget.texture.mipmaps = 'bad'
  assert.throws(
    () => renderToTarget(scene, colorMipCamera, colorMipTarget),
    /target\.texture\.mipmaps must be an array of image-like objects/i,
  )

  const depthMipTarget = new THREE.WebGLCubeRenderTarget(32)
  const depthMipCamera = new THREE.CubeCamera(0.01, 100, depthMipTarget)
  depthMipCamera.activeMipmapLevel = 1
  depthMipTarget.depthTexture = { type: THREE.UnsignedShortType, mipmaps: 'bad' }
  assert.throws(
    () => renderToTarget(scene, depthMipCamera, depthMipTarget),
    /target\.depthTexture\.mipmaps must be an array of image-like objects/i,
  )
})

test('malformed CubeCamera child camera containers fail clearly', () => {
  const scene = makeCubeCaptureScene()
  const cubeCamera = new THREE.CubeCamera(0.01, 100, new THREE.WebGLCubeRenderTarget(32))

  cubeCamera.children = 'bad'
  assert.throws(
    () => renderRgba(scene, cubeCamera, { width: 32, height: 32 }),
    /THREE\.CubeCamera\.children must be an array/i,
  )

  cubeCamera.children = []
  assert.throws(
    () => renderRgba(scene, cubeCamera, { width: 32, height: 32 }),
    /THREE\.CubeCamera requires six internal perspective cameras/i,
  )

  cubeCamera.children = [null, null, null, null, null, null]
  assert.throws(
    () => renderRgba(scene, cubeCamera, { width: 32, height: 32 }),
    /THREE\.CubeCamera\.children\[0\] must be a THREE\.Camera/i,
  )

  cubeCamera.children = [
    new THREE.CubeCamera(0.01, 100, new THREE.WebGLCubeRenderTarget(16)),
    null,
    null,
    null,
    null,
    null,
  ]
  assert.throws(
    () => renderRgba(scene, cubeCamera, { width: 32, height: 32 }),
    /THREE\.CubeCamera\.children\[0\] cannot be a THREE\.CubeCamera/i,
  )

  const matrixCubeCamera = new THREE.CubeCamera(0.01, 100, new THREE.WebGLCubeRenderTarget(32))
  matrixCubeCamera.children[0].projectionMatrix = { elements: [1, 2, 3] }
  assert.throws(
    () => renderRgba(scene, matrixCubeCamera, { width: 32, height: 32 }),
    /THREE\.CubeCamera\.children\[0\]\.projectionMatrix must be a THREE\.Matrix4/i,
  )
})

test('MeshBasicMaterial renders foreground pixels distinct from background', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.1, 0.1, 0.1)
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial({ color: 0xffaa00 })))

  const rgba = renderRgba(scene, makeCamera())
  const ratio = nonBackgroundRatio(rgba, BG)
  assert.ok(ratio > 0.05, `expected mesh to cover >5% of frame, got ${(ratio * 100).toFixed(1)}%`)
  assert.ok(ratio < 0.95, `expected background to be visible, got ${(ratio * 100).toFixed(1)}% non-bg`)
})

test('renderMode mask outputs white visible geometry over black', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(1, 0, 0)
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(1.2, 1.2), new THREE.MeshBasicMaterial({ color: 0x0088ff })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 64, height: 64, renderMode: 'mask' })
  const center = meanRegion(rgba, 64, 64, 28, 28, 36, 36)
  const corner = meanRegion(rgba, 64, 64, 0, 0, 8, 8)
  assert.ok(center.r > 250 && center.g > 250 && center.b > 250, `mask center should be white (${center.r}, ${center.g}, ${center.b})`)
  assert.ok(corner.r < 2 && corner.g < 2 && corner.b < 2, `mask background should be black (${corner.r}, ${corner.g}, ${corner.b})`)
})

test('renderMode object-id outputs stable per-object RGB IDs', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(1, 0, 0)
  const left = new THREE.Mesh(new THREE.PlaneGeometry(0.75, 0.8), new THREE.MeshBasicMaterial({ color: 0xff0000 }))
  const right = new THREE.Mesh(new THREE.PlaneGeometry(0.75, 0.8), new THREE.MeshBasicMaterial({ color: 0x00ff00 }))
  left.position.x = -0.5
  right.position.x = 0.5
  scene.add(left, right)

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 64, height: 64, renderMode: 'object-id' })
  const leftMean = meanRegion(rgba, 64, 64, 16, 28, 23, 36)
  const rightMean = meanRegion(rgba, 64, 64, 41, 28, 48, 36)
  const background = meanRegion(rgba, 64, 64, 0, 0, 8, 8)
  assertRgbClose(leftMean, objectIdBytes(left.id + 1), 'left object id')
  assertRgbClose(rightMean, objectIdBytes(right.id + 1), 'right object id')
  assert.notDeepEqual(objectIdBytes(left.id + 1), objectIdBytes(right.id + 1))
  assert.ok(background.r < 2 && background.g < 2 && background.b < 2, `object-id background should be black (${background.r}, ${background.g}, ${background.b})`)
})
