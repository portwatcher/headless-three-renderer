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
import { Renderer, extractCommonInfoSurfaceNames, extractWebGlInfoSurfaceNames, objectSurfaceNames, test } from './scenes.test.part-001.mjs'
import { assertRgbClose, makeCamera, meanScalarRegion } from './scenes.test.part-002.mjs'
test('Renderer clear writes packed depth-stencil stencil bytes', () => {
  const renderer = new Renderer()
  const initialDepth = Math.round(0.5 * 0xffffff) * 0x100
  const initialData = new Uint32Array(4 * 4)
  initialData.fill(initialDepth + 3)
  const depthTexture = {
    type: THREE.UnsignedInt248Type,
    format: THREE.DepthStencilFormat,
    image: { data: initialData, width: 4, height: 4 },
    source: { data: { data: initialData, width: 4, height: 4 } },
  }
  const target = { width: 4, height: 4, texture: {}, depthTexture }

  renderer.setRenderTarget(target)
  renderer.setScissor(1, 1, 2, 2)
  renderer.setScissorTest(true)
  renderer.setClearStencil(9)
  renderer.clearStencil()

  const stencilData = depthTexture.image.data
  assert.ok(stencilData instanceof Uint32Array, 'packed depth-stencil clear should keep Uint32Array data')
  for (let y = 0; y < 4; y += 1) {
    for (let x = 0; x < 4; x += 1) {
      const value = stencilData[y * 4 + x]
      const inside = x >= 1 && x < 3 && y >= 1 && y < 3
      assert.equal(value & 0xff, inside ? 9 : 3, `stencil byte should ${inside ? 'clear' : 'preserve'} at ${x},${y}`)
      assert.equal(Math.floor(value / 0x100) * 0x100, initialDepth, `stencil-only clear should preserve depth bits at ${x},${y}`)
    }
  }

  renderer.setScissorTest(false)
  renderer.setClearDepth(0.25)
  renderer.clearDepth()

  const depthCleared = depthTexture.image.data
  const expectedDepth = Math.round(0.25 * 0xffffff) * 0x100
  for (let y = 0; y < 4; y += 1) {
    for (let x = 0; x < 4; x += 1) {
      const value = depthCleared[y * 4 + x]
      const wasStencilCleared = x >= 1 && x < 3 && y >= 1 && y < 3
      assert.equal(Math.floor(value / 0x100) * 0x100, expectedDepth, `depth clear should update depth bits at ${x},${y}`)
      assert.equal(value & 0xff, wasStencilCleared ? 9 : 3, `depth-only clear should preserve stencil byte at ${x},${y}`)
    }
  }
  assert.equal(depthTexture.source.data.data, depthTexture.image.data)
  assert.equal(depthTexture.needsUpdate, true)
  renderer.setRenderTarget(null)
})

test('Renderer clear methods validate compatibility hooks', async () => {
  const scene = new THREE.Scene()
  const camera = makeCamera()
  const renderer = new Renderer()

  assert.equal(renderer.getClearDepth(), 1)
  assert.equal(renderer.getClearStencil(), 0)
  renderer.setClearColor(0x204080, 0.5)
  renderer.setClearDepth(0.25)
  renderer.setClearStencil(7)
  assert.equal(renderer.getClearDepth(), 0.25)
  assert.equal(renderer.getClearStencil(), 7)
  renderer.clear()
  renderer.clear(false, true, false)
  renderer.clearColor()
  renderer.clearDepth()
  renderer.clearStencil()
  assert.equal(await renderer.clearAsync(), undefined)
  assert.equal(await renderer.clearAsync(false, true, false), undefined)
  assert.equal(await renderer.clearColorAsync(), undefined)
  assert.equal(await renderer.clearDepthAsync(), undefined)
  assert.equal(await renderer.clearStencilAsync(), undefined)
  const previousTarget = { texture: {} }
  const clearTarget = { texture: {} }
  renderer.setRenderTarget(previousTarget, 2, 1)
  assert.equal(renderer.clearTarget(clearTarget, true, false, true), undefined)
  assert.strictEqual(renderer.getRenderTarget(), previousTarget)
  assert.equal(renderer.getActiveCubeFace(), 2)
  assert.equal(renderer.getActiveMipmapLevel(), 1)
  assert.equal(renderer.clearTarget(null), undefined)
  assert.strictEqual(renderer.getRenderTarget(), previousTarget)
  assert.equal(renderer.getActiveCubeFace(), 2)
  assert.equal(renderer.getActiveMipmapLevel(), 1)

  const sizedClearTarget = {
    width: 4,
    height: 4,
    texture: {},
    depthTexture: { type: THREE.FloatType, source: { data: {} } },
  }
  renderer.setClearColor(0x204080, 0.5)
  renderer.setClearDepth(0.375)
  assert.equal(renderer.clearTarget(sizedClearTarget, true, true, false), undefined)
  assert.strictEqual(renderer.getRenderTarget(), previousTarget)
  assert.equal(renderer.getActiveCubeFace(), 2)
  assert.equal(renderer.getActiveMipmapLevel(), 1)
  const targetClear = meanRgba(sizedClearTarget.data)
  assertRgbClose(targetClear, [0x20, 0x40, 0x80], 'Renderer.clearTarget should write color into sized targets')
  assert.ok(Math.abs(targetClear.a - 128) <= 1, `Renderer.clearTarget should write clear alpha (${targetClear.a})`)
  const targetDepthClear = meanScalarRegion(sizedClearTarget.depthTexture.image.data, 4, 4, 0, 0, 4, 4)
  assert.ok(Math.abs(targetDepthClear - 0.375) < 1e-6, `Renderer.clearTarget should write clear depth (${targetDepthClear})`)

  renderer.setRenderTarget(null)
  assert.equal(renderer.getClearDepth(), 0.375)
  assert.equal(renderer.getClearStencil(), 7)

  const clear = meanRgba(renderer.render(scene, camera, { width: 32, height: 32, format: 'rgba' }))
  assertRgbClose(clear, [0x20, 0x40, 0x80], 'Renderer clear hooks should preserve clear color state')
  assert.ok(Math.abs(clear.a - 128) <= 1, `Renderer clear hooks should preserve clear alpha state (${clear.a})`)

  assert.throws(
    () => renderer.clear('yes'),
    /Renderer\.clear color must be a boolean/i,
  )
  assert.throws(
    () => renderer.clear(true, 'depth'),
    /Renderer\.clear depth must be a boolean/i,
  )
  assert.throws(
    () => renderer.clear(true, true, 'stencil'),
    /Renderer\.clear stencil must be a boolean/i,
  )
  await assert.rejects(
    () => renderer.clearAsync('yes'),
    /Renderer\.clear color must be a boolean/i,
  )
  assert.throws(
    () => renderer.clearTarget('target'),
    /Renderer\.clearTarget target must be a target-like object/i,
  )
  assert.throws(
    () => renderer.clearTarget(clearTarget, 'yes'),
    /Renderer\.clearTarget color must be a boolean/i,
  )
  assert.throws(
    () => renderer.clearTarget(clearTarget, true, 'depth'),
    /Renderer\.clearTarget depth must be a boolean/i,
  )
  assert.throws(
    () => renderer.clearTarget(clearTarget, true, true, 'stencil'),
    /Renderer\.clearTarget stencil must be a boolean/i,
  )
})

test('Renderer setAnimationLoop is an inert validated compatibility hook', () => {
  const scene = new THREE.Scene()
  const camera = makeCamera()
  const renderer = new Renderer()
  let calls = 0

  renderer.setAnimationLoop(() => { calls += 1 })
  assert.equal(typeof renderer.getAnimationLoop(), 'function')
  renderer.render(scene, camera, { width: 32, height: 32, format: 'rgba' })
  assert.equal(calls, 0)

  renderer.setAnimationLoop(null)
  assert.equal(renderer.getAnimationLoop(), null)
  renderer.render(scene, camera, { width: 32, height: 32, format: 'rgba' })
  assert.equal(calls, 0)

  assert.throws(
    () => renderer.setAnimationLoop('loop'),
    /Renderer\.setAnimationLoop callback must be a function or null/i,
  )
})

test('Renderer autoClear flags are validated compatibility state', () => {
  const scene = new THREE.Scene()
  const camera = makeCamera()
  const renderer = new Renderer()

  assert.equal(renderer.autoClear, true)
  assert.equal(renderer.autoClearColor, true)
  assert.equal(renderer.autoClearDepth, true)
  assert.equal(renderer.autoClearStencil, true)

  renderer.autoClear = false
  renderer.autoClearColor = false
  renderer.autoClearDepth = false
  renderer.autoClearStencil = false
  assert.equal(renderer.autoClear, false)
  assert.equal(renderer.autoClearColor, false)
  assert.equal(renderer.autoClearDepth, false)
  assert.equal(renderer.autoClearStencil, false)

  renderer.setClearColor(0x204080, 0.5)
  const clear = meanRgba(renderer.render(scene, camera, { width: 32, height: 32, format: 'rgba' }))
  assertRgbClose(clear, [0x20, 0x40, 0x80], 'Renderer autoClear flags should not change pass-owned clear color')
  assert.ok(Math.abs(clear.a - 128) <= 1, `Renderer autoClear flags should not change pass-owned clear alpha (${clear.a})`)

  for (const [property, value, pattern] of [
    ['autoClear', 'yes', /Renderer\.autoClear must be a boolean/i],
    ['autoClearColor', 'yes', /Renderer\.autoClearColor must be a boolean/i],
    ['autoClearDepth', 'yes', /Renderer\.autoClearDepth must be a boolean/i],
    ['autoClearStencil', 'yes', /Renderer\.autoClearStencil must be a boolean/i],
  ]) {
    assert.throws(
      () => { renderer[property] = value },
      pattern,
    )
  }
})

test('Renderer dispose is a no-op compatibility hook', () => {
  const scene = new THREE.Scene()
  const camera = makeCamera()
  const renderer = new Renderer()

  renderer.setClearColor(0x204080, 0.5)
  renderer.dispose()

  const clear = meanRgba(renderer.render(scene, camera, { width: 32, height: 32, format: 'rgba' }))
  assertRgbClose(clear, [0x20, 0x40, 0x80], 'Renderer dispose should preserve renderer state')
  assert.ok(Math.abs(clear.a - 128) <= 1, `Renderer dispose should preserve renderer clear alpha (${clear.a})`)
})

test('Renderer context loss hooks are no-op compatibility hooks', () => {
  const scene = new THREE.Scene()
  const camera = makeCamera()
  const renderer = new Renderer()
  const deviceLostInfo = { api: 'headless', message: 'test loss' }
  let observedDeviceLostInfo = null

  assert.equal(typeof renderer.onDeviceLost, 'function')
  assert.equal(typeof renderer._onDeviceLost, 'function')
  assert.equal(renderer.isDeviceLost, false)
  assert.equal(renderer.onDeviceLost(deviceLostInfo), undefined)
  assert.equal(renderer.isDeviceLost, true)
  assert.equal(renderer.forceContextRestore(), undefined)
  assert.equal(renderer.isDeviceLost, false)
  assert.equal(renderer._onDeviceLost(deviceLostInfo), undefined)
  assert.equal(renderer.isDeviceLost, true)
  assert.equal(renderer.forceContextRestore(), undefined)
  assert.equal(renderer.isDeviceLost, false)
  renderer.onDeviceLost = (info) => { observedDeviceLostInfo = info }
  assert.equal(renderer.onDeviceLost(deviceLostInfo), undefined)
  assert.equal(observedDeviceLostInfo, deviceLostInfo)
  assert.equal(renderer.isDeviceLost, false)
  observedDeviceLostInfo = null
  assert.equal(renderer._onDeviceLost(deviceLostInfo), undefined)
  assert.equal(observedDeviceLostInfo, deviceLostInfo)
  assert.equal(renderer.isDeviceLost, true)
  assert.equal(renderer.forceContextRestore(), undefined)
  assert.equal(renderer.isDeviceLost, false)

  renderer.setClearColor(0x204080, 0.5)
  assert.equal(renderer.forceContextLoss(), undefined)
  assert.equal(renderer.isDeviceLost, false)
  assert.equal(renderer.forceContextRestore(), undefined)
  assert.equal(renderer.isDeviceLost, false)

  const clear = meanRgba(renderer.render(scene, camera, { width: 32, height: 32, format: 'rgba' }))
  assertRgbClose(clear, [0x20, 0x40, 0x80], 'Renderer context loss hooks should preserve renderer state')
  assert.ok(Math.abs(clear.a - 128) <= 1, `Renderer context loss hooks should preserve clear alpha (${clear.a})`)
  assert.throws(
    () => { renderer.onDeviceLost = null },
    /Renderer\.onDeviceLost must be a function/i,
  )
})

test('Renderer resetState is a no-op compatibility hook', () => {
  const scene = new THREE.Scene()
  const camera = makeCamera()
  const renderer = new Renderer()

  renderer.setClearColor(0x204080, 0.5)
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace
  renderer.localClippingEnabled = false
  renderer.resetState()
  renderer.resetGLState()

  assert.equal(renderer.outputColorSpace, THREE.LinearSRGBColorSpace)
  assert.equal(renderer.localClippingEnabled, false)

  const clear = meanRgba(renderer.render(scene, camera, { width: 32, height: 32, format: 'rgba' }))
  assertRgbClose(clear, [0x20, 0x40, 0x80], 'Renderer resetState should preserve renderer state')
  assert.ok(Math.abs(clear.a - 128) <= 1, `Renderer resetState should preserve renderer clear alpha (${clear.a})`)
})

test('Renderer.info tracks installed Three info surfaces', () => {
  const renderer = new Renderer()
  const infoSurface = objectSurfaceNames(renderer.info)

  for (const [label, names, minimum] of [
    ['WebGLInfo', extractWebGlInfoSurfaceNames(), 5],
    ['CommonRenderer Info', extractCommonInfoSurfaceNames(), 8],
  ]) {
    assert.ok(names.size >= minimum, `Expected to find installed Three.js ${label} surface names.`)
    const missing = [...names].filter((name) => !infoSurface.has(name)).sort()
    assert.deepEqual(missing, [], `Renderer.info is missing installed Three.js ${label} names: ${missing.join(', ')}`)
  }
})
