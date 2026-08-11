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
import { test } from './scenes.test.part-001.mjs'
import { constantUvPlane, countRegionPixels, meanRegion, renderRgba, rgbaTexture, setConstantUvAttribute } from './scenes.test.part-002.mjs'
test('alphaMap samples selected uv1-uv3 texture channels', () => {
  function renderAlphaChannel(channel) {
    const alphaMap = rgbaTexture([
      255, 0, 255, 255,
      255, 255, 255, 255,
    ], 2, 1)
    alphaMap.channel = channel

    const geometry = constantUvPlane(0.25, 0.5)
    if (channel > 0) {
      setConstantUvAttribute(geometry, `uv${channel}`, 0.75, 0.5)
    }

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 1)
    scene.add(new THREE.Mesh(
      geometry,
      new THREE.MeshBasicMaterial({
        color: 0xff0000,
        alphaMap,
        alphaTest: 0.5,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const primary = renderAlphaChannel(0)
  assert.ok(primary.b > primary.r + 80, `alphaMap channel=0 should sample the transparent primary UV texel (${primary.b} vs ${primary.r})`)
  for (const channel of [1, 2, 3]) {
    const secondary = renderAlphaChannel(channel)
    assert.ok(secondary.r > secondary.b + 40, `alphaMap channel=${channel} should sample the opaque uv${channel} texel (${secondary.r} vs ${secondary.b})`)
  }
})

test('alphaMap honors nearest texture filters before alpha testing', () => {
  function renderWithFilter(filter) {
    const alphaMap = rgbaTexture([
      255, 0, 0, 255,
      255, 255, 0, 255,
    ], 2, 1)
    alphaMap.magFilter = filter
    alphaMap.minFilter = filter

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 1)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.45, 0.5),
      new THREE.MeshBasicMaterial({
        color: 0xff0000,
        alphaMap,
        alphaTest: 0.2,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const nearest = renderWithFilter(THREE.NearestFilter)
  const linear = renderWithFilter(THREE.LinearFilter)
  assert.ok(nearest.b > nearest.r + 80, `NearestFilter should choose the transparent alpha texel (${nearest.b} vs ${nearest.r})`)
  assert.ok(linear.r > linear.b + 40, `LinearFilter should blend enough green-channel alpha to pass the test (${linear.r} vs ${linear.b})`)
})

test('alphaMap honors horizontal and vertical repeat wrapping before alpha testing', () => {
  function renderWithWrapping({ wrapS, wrapT, vertical = false }) {
    const alphaMap = vertical
      ? rgbaTexture([
        255, 255, 255, 255,
        255, 0, 255, 255,
      ], 1, 2)
      : rgbaTexture([
        255, 255, 255, 255,
        255, 0, 255, 255,
      ], 2, 1)
    if (wrapS != null) alphaMap.wrapS = wrapS
    if (wrapT != null) alphaMap.wrapT = wrapT
    alphaMap.magFilter = THREE.NearestFilter
    alphaMap.minFilter = THREE.NearestFilter

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 1)
    scene.add(new THREE.Mesh(
      constantUvPlane(vertical ? 0.5 : 1.25, vertical ? 1.25 : 0.5),
      new THREE.MeshBasicMaterial({
        color: 0xff0000,
        alphaMap,
        alphaTest: 0.5,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const clamped = renderWithWrapping({ wrapS: THREE.ClampToEdgeWrapping })
  const repeated = renderWithWrapping({ wrapS: THREE.RepeatWrapping })
  const mirrored = renderWithWrapping({ wrapS: THREE.MirroredRepeatWrapping })
  assert.ok(clamped.b > clamped.r + 80, `clamped alphaMap U coordinates should sample the transparent edge texel (${clamped.b} vs ${clamped.r})`)
  assert.ok(repeated.r > repeated.b + 40, `repeated alphaMap U coordinates should wrap to the opaque texel (${repeated.r} vs ${repeated.b})`)
  assert.ok(mirrored.b > mirrored.r + 80, `mirrored alphaMap U coordinates should reflect to the transparent texel (${mirrored.b} vs ${mirrored.r})`)

  const clampedVertical = renderWithWrapping({ wrapT: THREE.ClampToEdgeWrapping, vertical: true })
  const repeatedVertical = renderWithWrapping({ wrapT: THREE.RepeatWrapping, vertical: true })
  const mirroredVertical = renderWithWrapping({ wrapT: THREE.MirroredRepeatWrapping, vertical: true })
  assert.ok(clampedVertical.b > clampedVertical.r + 80, `clamped alphaMap V coordinates should sample the transparent edge texel (${clampedVertical.b} vs ${clampedVertical.r})`)
  assert.ok(repeatedVertical.r > repeatedVertical.b + 40, `repeated alphaMap V coordinates should wrap to the opaque texel (${repeatedVertical.r} vs ${repeatedVertical.b})`)
  assert.ok(mirroredVertical.b > mirroredVertical.r + 80, `mirrored alphaMap V coordinates should reflect to the transparent texel (${mirroredVertical.b} vs ${mirroredVertical.r})`)
})

test('material alphaHash produces stochastic coverage without transparent blending', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 1, 0)
  const front = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({
      color: 0xff0000,
      opacity: 0.5,
      alphaHash: true,
    }),
  )
  front.position.z = 0.1
  scene.add(front)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 64, height: 64 })
  const redPixels = countRegionPixels(rgba, 64, 64, 24, 24, 40, 40, (r, g, b) => r > 160 && g < 60 && b < 60)
  const greenPixels = countRegionPixels(rgba, 64, 64, 24, 24, 40, 40, (r, g, b) => g > 160 && r < 60 && b < 60)
  assert.ok(redPixels > 40, `alphaHash should leave red covered pixels (${redPixels})`)
  assert.ok(greenPixels > 120, `alphaHash should reveal green pixels through hashed discards (${greenPixels})`)
})

test('material alphaToCoverage uses MSAA coverage from output alpha', () => {
  function renderCoverage(alphaToCoverage, sampleCount = 4) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({
        color: 0xffffff,
        opacity: 0.5,
        transparent: false,
        alphaToCoverage,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, {
      width: 64,
      height: 64,
      sampleCount,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 64, 64, 24, 24, 40, 40)
  }

  const noCoverage = renderCoverage(false)
  const coverage = renderCoverage(true)
  const singleSample = renderCoverage(true, 1)
  assert.ok(noCoverage.r > 170, `opaque non-A2C path should keep bright RGB despite opacity alpha (${noCoverage.r})`)
  assert.ok(Math.abs(singleSample.r - noCoverage.r) < 5, `single-sample alphaToCoverage should not alter RGB coverage (${singleSample.r} vs ${noCoverage.r})`)
  assert.ok(coverage.r > 30 && coverage.r < noCoverage.r - 80, `4x alphaToCoverage should resolve partial RGB coverage (${coverage.r} vs ${noCoverage.r})`)
})

test('material alphaToCoverage smooths alphaTest thresholds', () => {
  function thresholdAlphaMap() {
    const texture = rgbaTexture([
      255, 0, 255, 255,
      255, 255, 255, 255,
    ], 2, 1)
    texture.magFilter = THREE.LinearFilter
    texture.minFilter = THREE.LinearFilter
    return texture
  }

  function renderThreshold(material) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return renderRgba(scene, camera, {
      width: 64,
      height: 64,
      sampleCount: 4,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    })
  }

  function softThresholdPixels(rgba) {
    return countRegionPixels(rgba, 64, 64, 14, 14, 50, 50, (r, g, b) => {
      return r > 40 && r < 180 && Math.abs(r - g) < 3 && Math.abs(r - b) < 3
    })
  }

  const hard = softThresholdPixels(renderThreshold(new THREE.MeshBasicMaterial({
    color: 0xffffff,
    alphaMap: thresholdAlphaMap(),
    alphaTest: 0.5,
    alphaToCoverage: false,
  })))

  const builtIn = softThresholdPixels(renderThreshold(new THREE.MeshBasicMaterial({
    color: 0xffffff,
    alphaMap: thresholdAlphaMap(),
    alphaTest: 0.5,
    alphaToCoverage: true,
  })))
  assert.ok(builtIn > hard + 20, `built-in material alphaTest should add partial coverage pixels (${builtIn} vs ${hard})`)

  const custom = new THREE.ShaderMaterial()
  custom.alphaMap = thresholdAlphaMap()
  custom.alphaTest = 0.5
  custom.alphaToCoverage = true
  custom.userData.headlessThreeRenderer = {
    fragmentWgsl: 'return vec4<f32>(1.0, 1.0, 1.0, alpha);',
  }
  const customSoft = softThresholdPixels(renderThreshold(custom))
  assert.ok(customSoft > hard + 20, `custom WGSL alphaTest should add partial coverage pixels (${customSoft} vs ${hard})`)
})

test('material alphaToCoverage smooths clipping plane edges', () => {
  function renderClipped(material) {
    material.clippingPlanes = [new THREE.Plane(new THREE.Vector3(1, 1, 0).normalize(), 0)]

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return renderRgba(scene, camera, {
      width: 64,
      height: 64,
      sampleCount: 4,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    })
  }

  function softClipPixels(rgba) {
    return countRegionPixels(rgba, 64, 64, 14, 14, 50, 50, (r, g, b) => {
      return r > 40 && r < 180 && Math.abs(r - g) < 3 && Math.abs(r - b) < 3
    })
  }

  const hard = softClipPixels(renderClipped(new THREE.MeshBasicMaterial({
    color: 0xffffff,
    alphaToCoverage: false,
  })))

  const builtIn = softClipPixels(renderClipped(new THREE.MeshBasicMaterial({
    color: 0xffffff,
    alphaToCoverage: true,
  })))
  assert.ok(builtIn > hard + 20, `built-in material clipping should add partial coverage pixels (${builtIn} vs ${hard})`)

  const custom = new THREE.ShaderMaterial()
  custom.alphaToCoverage = true
  custom.userData.headlessThreeRenderer = {
    fragmentWgsl: 'return vec4<f32>(1.0, 1.0, 1.0, alpha);',
  }
  const customSoft = softClipPixels(renderClipped(custom))
  assert.ok(customSoft > hard + 20, `custom WGSL clipping should add partial coverage pixels (${customSoft} vs ${hard})`)
})

test('material clippingPlanes discard the negative plane side', () => {
  const material = new THREE.MeshBasicMaterial({ color: 0xff0000 })
  material.clippingPlanes = [new THREE.Plane(new THREE.Vector3(1, 0, 0), 0)]

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 1)
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 64, height: 64 })
  const clippedLeft = meanRegion(rgba, 64, 64, 12, 22, 24, 42)
  const visibleRight = meanRegion(rgba, 64, 64, 40, 22, 52, 42)

  assert.ok(clippedLeft.b > clippedLeft.r + 80, `left side should reveal blue background (${clippedLeft.b} vs ${clippedLeft.r})`)
  assert.ok(visibleRight.r > visibleRight.b + 80, `right side should keep the red plane (${visibleRight.r} vs ${visibleRight.b})`)
})
