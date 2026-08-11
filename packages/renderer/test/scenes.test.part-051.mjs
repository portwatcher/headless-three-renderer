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
import { constantUvPlane, makeCamera, meanRegion, renderRgba, rgbaTexture, setConstantUvAttribute, setTextureMatrixOffset, solidTexture } from './scenes.test.part-002.mjs'
test('displacementMap honors horizontal and vertical repeat wrapping before depth output', () => {
  function renderDisplaced({ wrapS, wrapT, vertical = false }) {
    const displacementMap = vertical
      ? rgbaTexture([
        0, 0, 0, 255,
        255, 255, 255, 255,
      ], 1, 2)
      : rgbaTexture([
        255, 255, 255, 255,
        0, 0, 0, 255,
      ], 2, 1)
    if (wrapS != null) displacementMap.wrapS = wrapS
    if (wrapT != null) displacementMap.wrapT = wrapT
    displacementMap.magFilter = THREE.NearestFilter
    displacementMap.minFilter = THREE.NearestFilter

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(vertical ? 0.5 : 1.25, vertical ? 1.25 : 0.5),
      new THREE.MeshDepthMaterial({
        displacementMap,
        displacementScale: 2.5,
        displacementBias: 0,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 20, 20, 44, 44)
  }

  const clamped = renderDisplaced({ wrapS: THREE.ClampToEdgeWrapping })
  const repeated = renderDisplaced({ wrapS: THREE.RepeatWrapping })
  const mirrored = renderDisplaced({ wrapS: THREE.MirroredRepeatWrapping })
  assert.ok(repeated.r > clamped.r + 15, `repeat wrapping should wrap displacement U coordinates to the high texel (${repeated.r} vs ${clamped.r})`)
  assert.ok(repeated.r > mirrored.r + 15, `mirrored displacement U coordinates should reflect away from the high texel (${mirrored.r} vs ${repeated.r})`)

  const clampedVertical = renderDisplaced({ wrapT: THREE.ClampToEdgeWrapping, vertical: true })
  const repeatedVertical = renderDisplaced({ wrapT: THREE.RepeatWrapping, vertical: true })
  const mirroredVertical = renderDisplaced({ wrapT: THREE.MirroredRepeatWrapping, vertical: true })
  assert.ok(repeatedVertical.r > clampedVertical.r + 15, `repeat wrapping should wrap displacement V coordinates to the high texel (${repeatedVertical.r} vs ${clampedVertical.r})`)
  assert.ok(repeatedVertical.r > mirroredVertical.r + 15, `mirrored displacement V coordinates should reflect away from the high texel (${mirroredVertical.r} vs ${repeatedVertical.r})`)
})

test('displacementMap applies displacementBias independently of sampled height', () => {
  function renderDisplacementBias(displacementBias) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.25, 0.5),
      new THREE.MeshDepthMaterial({
        displacementMap: solidTexture(0, 0, 0),
        displacementScale: 0,
        displacementBias,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 20, 20, 44, 44)
  }

  const flat = renderDisplacementBias(0)
  const biased = renderDisplacementBias(2.4)
  assert.ok(biased.r > flat.r + 25, `positive displacementBias should move the plane nearer (${biased.r} vs ${flat.r})`)
})

test('displacementMap honors explicit texture matrices before depth output', () => {
  function renderDisplaced(matrixOffsetX) {
    const displacementMap = rgbaTexture([
      0, 0, 0, 255,
      255, 255, 255, 255,
    ], 2, 1)
    displacementMap.magFilter = THREE.NearestFilter
    displacementMap.minFilter = THREE.NearestFilter
    if (matrixOffsetX !== 0) setTextureMatrixOffset(displacementMap, matrixOffsetX)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.25, 0.5),
      new THREE.MeshDepthMaterial({
        displacementMap,
        displacementScale: 2.5,
        displacementBias: 0,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 20, 20, 44, 44)
  }

  const flat = renderDisplaced(0)
  const displaced = renderDisplaced(0.5)
  assert.ok(displaced.r > flat.r + 15, `explicit displacementMap matrix should move the plane nearer (${displaced.r} vs ${flat.r})`)
})

test('displacementMap samples selected uv1-uv3 texture channels before depth output', () => {
  function renderDisplaced(channel) {
    const displacementMap = rgbaTexture([
      0, 0, 0, 255,
      255, 255, 255, 255,
    ], 2, 1)
    displacementMap.channel = channel

    const geometry = constantUvPlane(0.25, 0.5)
    if (channel > 0) {
      setConstantUvAttribute(geometry, `uv${channel}`, 0.75, 0.5)
    }

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      geometry,
      new THREE.MeshDepthMaterial({
        displacementMap,
        displacementScale: 2.5,
        displacementBias: 0,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 20, 20, 44, 44)
  }

  const primary = renderDisplaced(0)
  for (const channel of [1, 2, 3]) {
    const secondary = renderDisplaced(channel)
    assert.ok(secondary.r > primary.r + 15, `displacementMap channel=${channel} should sample uv${channel}'s displaced texel (${secondary.r} vs ${primary.r})`)
  }
})

test('MeshDistanceMaterial renders farther fragments with higher red distance', () => {
  function renderDistanceAt(z) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), new THREE.MeshDistanceMaterial())
    mesh.position.z = z
    scene.add(mesh)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 8)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const near = renderDistanceAt(2.8)
  const far = renderDistanceAt(-4)
  assert.ok(far.r > near.r + 60, `far distance plane should write a higher red distance (${far.r} vs ${near.r})`)
  assert.ok(far.g < 5 && far.b < 5, `distance material should write distance in red only (${far.g}, ${far.b})`)
})

test('MeshDistanceMaterial wireframe renders distance on triangle edges', () => {
  function renderDistanceWireframe(wireframe) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const material = new THREE.MeshDistanceMaterial()
    material.wireframe = wireframe
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 8)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return renderRgba(scene, camera, { width: 64, height: 64 })
  }

  const solidRatio = nonBackgroundRatio(renderDistanceWireframe(false), [0, 0, 0])
  const wireRatio = nonBackgroundRatio(renderDistanceWireframe(true), [0, 0, 0])
  assert.ok(solidRatio > 0.4, `solid distance material should fill the plane (${solidRatio})`)
  assert.ok(wireRatio > 0.005, `wireframe distance material should draw visible edges (${wireRatio})`)
  assert.ok(wireRatio < solidRatio * 0.35, `wireframe distance material should not fill faces (${wireRatio} vs ${solidRatio})`)
})

test('MeshBasicMaterial wireframe renders triangle edges without filling faces', () => {
  function renderBasicWireframe(wireframe) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 8)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return renderRgba(scene, camera, { width: 64, height: 64 })
  }

  const solidRatio = nonBackgroundRatio(renderBasicWireframe(false), [0, 0, 0])
  const wireRatio = nonBackgroundRatio(renderBasicWireframe(true), [0, 0, 0])
  assert.ok(solidRatio > 0.4, `solid basic material should fill the plane (${solidRatio})`)
  assert.ok(wireRatio > 0.005, `wireframe basic material should draw visible edges (${wireRatio})`)
  assert.ok(wireRatio < solidRatio * 0.35, `wireframe basic material should not fill faces (${wireRatio} vs ${solidRatio})`)
})

test('MeshDistanceMaterial honors referencePosition and distance range', () => {
  function renderDistanceAt(z) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const material = new THREE.MeshDistanceMaterial()
    material.referencePosition = new THREE.Vector3(0, 0, -4)
    material.nearDistance = 0
    material.farDistance = 7
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), material)
    mesh.position.z = z
    scene.add(mesh)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 8)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 20, 20, 44, 44)
  }

  const nearReference = renderDistanceAt(-4)
  const farReference = renderDistanceAt(2.8)
  assert.ok(
    farReference.r > nearReference.r + 100,
    `distance material should measure from referencePosition (${farReference.r} vs ${nearReference.r})`,
  )
})

test('invalid MeshDistanceMaterial range and reference values fail clearly', () => {
  const cases = [
    ['referencePosition', (material) => {
      material.referencePosition = [0, Number.NaN, 0]
    }, /material\.referencePosition\[1\] must be a finite number/i],
    ['nearDistance', (material) => {
      material.nearDistance = 'near'
    }, /material\.nearDistance must be a finite number/i],
    ['farDistance', (material) => {
      material.farDistance = Number.NaN
    }, /material\.farDistance must be a finite number/i],
    ['hint nearDistance', (material) => {
      material.userData.headlessThreeRenderer = { nearDistance: 'near' }
    }, /material\.userData\.headlessThreeRenderer\.nearDistance must be a finite number/i],
    ['hint distanceFar', (material) => {
      material.userData.headlessThreeRenderer = { distanceFar: Number.POSITIVE_INFINITY }
    }, /material\.userData\.headlessThreeRenderer\.distanceFar must be a finite number/i],
    ['hint distanceReferencePosition', (material) => {
      material.userData.headlessThreeRenderer = { distanceReferencePosition: { x: 0, y: 'near', z: 0 } }
    }, /material\.userData\.headlessThreeRenderer\.distanceReferencePosition\.y must be a finite number/i],
    ['userData container', (material) => {
      material.userData = 'distance'
    }, /material\.userData must be an object/i],
    ['modern hint container', (material) => {
      material.userData.headlessThreeRenderer = 'distance'
    }, /material\.userData\.headlessThreeRenderer must be an object/i],
    ['legacy hint container', (material) => {
      material.userData.headlessRenderer = []
    }, /material\.userData\.headlessRenderer must be an object/i],
  ]

  for (const [name, mutate, pattern] of cases) {
    const scene = new THREE.Scene()
    const material = new THREE.MeshDistanceMaterial()
    mutate(material)
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      pattern,
      `${name} should fail clearly`,
    )
  }
})

test('MeshDistanceMaterial alphaMap cuts out discarded fragments', () => {
  const alphaMap = rgbaTexture([
    255, 0, 255, 255,
    255, 255, 255, 255,
  ], 2, 1)
  alphaMap.magFilter = THREE.NearestFilter
  alphaMap.minFilter = THREE.NearestFilter

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshDistanceMaterial({ alphaMap, alphaTest: 0.5 }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 8)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 64, height: 64 })
  const discarded = meanRegion(rgba, 64, 64, 14, 24, 28, 40)
  const visible = meanRegion(rgba, 64, 64, 36, 24, 50, 40)
  assert.ok(discarded.r < 2, `alphaMap cutout should keep background distance (${discarded.r})`)
  assert.ok(visible.r > 60, `opaque alphaMap region should write distance (${visible.r})`)
})
