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
import { Renderer, extractLights, test } from './scenes.test.part-001.mjs'
import { makeCamera, meanAbsDiff, meanRegion, renderRgba } from './scenes.test.part-002.mjs'
test('Renderer transmissionResolutionScale controls transmission scene-color resolution', () => {
  const width = 64
  const height = 64
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  function makeScene() {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)

    const left = new THREE.Mesh(
      new THREE.PlaneGeometry(1.6, 3),
      new THREE.MeshBasicMaterial({ color: 0xff0000 }),
    )
    left.position.set(-0.8, 0, -0.1)
    scene.add(left)

    const right = new THREE.Mesh(
      new THREE.PlaneGeometry(1.6, 3),
      new THREE.MeshBasicMaterial({ color: 0x0000ff }),
    )
    right.position.set(0.8, 0, -0.1)
    scene.add(right)

    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(3, 3),
      new THREE.MeshPhysicalMaterial({
        color: 0xffffff,
        metalness: 0,
        roughness: 0.02,
        transmission: 1,
        thickness: 0,
        ior: 1.5,
      }),
    ))
    return scene
  }

  function centerEdgeContrast(rgba) {
    const left = meanRegion(rgba, width, height, 25, 20, 31, 44)
    const right = meanRegion(rgba, width, height, 33, 20, 39, 44)
    return Math.abs((left.r - left.b) - (right.r - right.b))
  }

  const renderer = new Renderer()
  renderer.transmissionResolutionScale = 1
  const fullResolution = renderer.render(makeScene(), camera, { width, height, format: 'rgba' })
  renderer.transmissionResolutionScale = 0.125
  const lowResolution = renderer.render(makeScene(), camera, { width, height, format: 'rgba' })
  renderer.transmissionResolutionScale = 1
  const optionLowResolution = renderer.render(makeScene(), camera, { width, height, format: 'rgba', transmissionResolutionScale: 0.125 })
  assert.equal(renderer.transmissionResolutionScale, 1)
  renderer.transmissionResolutionScale = 0.125
  const optionFullResolution = renderer.render(makeScene(), camera, { width, height, format: 'rgba', transmissionResolutionScale: 1 })
  assert.equal(renderer.transmissionResolutionScale, 0.125)

  const fullContrast = centerEdgeContrast(fullResolution)
  const lowContrast = centerEdgeContrast(lowResolution)
  const optionLowContrast = centerEdgeContrast(optionLowResolution)
  const optionFullContrast = centerEdgeContrast(optionFullResolution)
  assert.ok(fullContrast > 80, `full-resolution transmission scene color should preserve the edge (${fullContrast.toFixed(1)})`)
  assert.ok(
    lowContrast < fullContrast - 20,
    `low transmissionResolutionScale should soften the scene-color edge (${lowContrast.toFixed(1)} vs ${fullContrast.toFixed(1)})`,
  )
  assert.ok(
    optionLowContrast < fullContrast - 20,
    `options.transmissionResolutionScale should soften the scene-color edge without mutating renderer state (${optionLowContrast.toFixed(1)} vs ${fullContrast.toFixed(1)})`,
  )
  assert.ok(
    optionFullContrast > 80,
    `options.transmissionResolutionScale should override low renderer state for a full-resolution edge (${optionFullContrast.toFixed(1)})`,
  )
})

test('physical transmission dispersion separates transmitted color channels', () => {
  const width = 64
  const height = 64
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  function makeScene(dispersion) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)

    const left = new THREE.Mesh(
      new THREE.PlaneGeometry(1.6, 3),
      new THREE.MeshBasicMaterial({ color: 0xff0000 }),
    )
    left.position.set(-0.8, 0, -0.2)
    scene.add(left)

    const right = new THREE.Mesh(
      new THREE.PlaneGeometry(1.6, 3),
      new THREE.MeshBasicMaterial({ color: 0x0000ff }),
    )
    right.position.set(0.8, 0, -0.2)
    scene.add(right)

    const glass = new THREE.Mesh(
      new THREE.SphereGeometry(0.95, 48, 24),
      new THREE.MeshPhysicalMaterial({
        color: 0xffffff,
        metalness: 0,
        roughness: 0.02,
        transmission: 1,
        thickness: 40,
        ior: 2.2,
        dispersion,
      }),
    )
    scene.add(glass)
    return scene
  }

  const normal = renderRgba(makeScene(0), camera, { width, height })
  const dispersed = renderRgba(makeScene(10), camera, { width, height })
  const diff = meanAbsDiff(normal, dispersed)
  const normalEdge = meanRegion(normal, width, height, 28, 22, 36, 42)
  const dispersedEdge = meanRegion(dispersed, width, height, 28, 22, 36, 42)
  const normalSeparation = Math.abs(normalEdge.r - normalEdge.b)
  const dispersedSeparation = Math.abs(dispersedEdge.r - dispersedEdge.b)

  assert.ok(diff > 10, `dispersion should affect transmitted color, diff=${diff.toFixed(2)}`)
  assert.ok(
    Math.abs(dispersedSeparation - normalSeparation) > 20,
    `dispersion should change edge channel separation (${dispersedSeparation.toFixed(1)} vs ${normalSeparation.toFixed(1)})`,
  )
})

test('directional cascaded shadow hints render successfully', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.05, 0.05, 0.05)
  scene.add(new THREE.AmbientLight(0xffffff, 0.2))

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(8, 8),
    new THREE.MeshStandardMaterial({ color: 0x888888, roughness: 0.8 }),
  )
  ground.rotation.x = -Math.PI / 2
  ground.receiveShadow = true
  scene.add(ground)

  const box = new THREE.Mesh(
    new THREE.BoxGeometry(1, 1, 1),
    new THREE.MeshStandardMaterial({ color: 0xff5533 }),
  )
  box.position.y = 0.5
  box.castShadow = true
  scene.add(box)

  const light = new THREE.DirectionalLight(0xffffff, 1.5)
  light.position.set(4, 6, 3)
  light.target.position.set(0, 0, 0)
  light.castShadow = true
  light.shadow.mapSize.set(256, 256)
  light.userData.headlessThreeRenderer = {
    shadowCascades: [
      { left: -3, right: 3, top: 3, bottom: -3, near: 0.1, far: 16, split: 4 },
      { left: -7, right: 7, top: 7, bottom: -7, near: 0.1, far: 32, split: 12 },
    ],
  }
  scene.add(light)
  scene.add(light.target)

  const rgba = renderRgba(scene, makeCamera(), { width: 64, height: 64 })
  assert.equal(rgba.length, 64 * 64 * 4)
})

test('directional shadow cascade hints over four valid cascades fail clearly', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshStandardMaterial({ color: 0xffffff }),
  ))

  const light = new THREE.DirectionalLight(0xffffff, 1)
  light.position.set(4, 6, 3)
  light.target.position.set(0, 0, 0)
  light.castShadow = true
  light.userData.headlessThreeRenderer = {
    shadowCascades: Array.from({ length: 5 }, (_, index) => ({
      left: -2 - index,
      right: 2 + index,
      top: 2 + index,
      bottom: -2 - index,
      near: 0.1,
      far: 12 + index,
      split: 2 + index,
    })),
  }
  scene.add(light)
  scene.add(light.target)

  assert.throws(
    () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
    /shadow cascade hints.*at most 4 valid cascades/i,
  )
})

test('invalid directional shadow cascade hints fail clearly', () => {
  const validCascade = (index = 0) => ({
    left: -2 - index,
    right: 2 + index,
    top: 2 + index,
    bottom: -2 - index,
    near: 0.1,
    far: 12 + index,
    split: 2 + index,
  })

  const containerCases = [
    ['light userData container', (light) => {
      light.userData = 'cascades'
    }, /light\.userData must be an object/i],
    ['modern hint container', (light) => {
      light.userData.headlessThreeRenderer = 'cascades'
    }, /light\.userData\.headlessThreeRenderer must be an object/i],
    ['modern shadowCascades', (light) => {
      light.userData.headlessThreeRenderer = { shadowCascades: 'cascades' }
    }, /light\.userData\.headlessThreeRenderer\.shadowCascades must be an array/i],
    ['legacy hint container', (light) => {
      light.userData.headlessRenderer = []
    }, /light\.userData\.headlessRenderer must be an object/i],
    ['legacy cascades', (light) => {
      light.userData.headlessRenderer = { cascades: 'cascades' }
    }, /light\.userData\.headlessRenderer\.cascades must be an array/i],
    ['shadow cascades', (light) => {
      light.shadow.cascades = 'cascades'
    }, /light\.shadow\.cascades must be an array/i],
  ]
  for (const [name, setup, pattern] of containerCases) {
    const scene = new THREE.Scene()
    const light = new THREE.DirectionalLight(0xffffff, 1)
    light.castShadow = true
    setup(light)
    scene.add(light)

    assert.throws(
      () => extractLights(scene),
      pattern,
      `${name} should fail clearly`,
    )
  }

  const cases = [
    ['non-object cascade', [validCascade(), null], /shadowCascades\[1\] must be an object/i],
    ['missing far bound', [{ ...validCascade(), far: undefined }, validCascade(1)], /shadowCascades\[0\]\.far must be a finite number/i],
    ['invalid split', [{ ...validCascade(), split: 'near' }, validCascade(1)], /shadowCascades\[0\]\.split must be a finite number/i],
    ['invalid distance alias', [{ ...validCascade(), split: undefined, distance: Number.NaN }, validCascade(1)], /shadowCascades\[0\]\.distance must be a finite number/i],
  ]

  for (const [name, shadowCascades, pattern] of cases) {
    const scene = new THREE.Scene()
    const light = new THREE.DirectionalLight(0xffffff, 1)
    light.castShadow = true
    light.userData.headlessThreeRenderer = { shadowCascades }
    scene.add(light)

    assert.throws(
      () => extractLights(scene),
      pattern,
      `${name} should fail clearly`,
    )
  }
})
