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
import { countRegionPixels, meanRegion, renderRgba } from './scenes.test.part-002.mjs'
test('GroundedSkybox renders generated textured sky geometry', () => {
  const texture = new THREE.DataTexture(new Uint8Array([
    255, 0, 0, 255,
    0, 255, 0, 255,
    0, 0, 255, 255,
    255, 255, 0, 255,
  ]), 2, 2, THREE.RGBAFormat)
  texture.colorSpace = THREE.SRGBColorSpace
  texture.needsUpdate = true

  const skybox = new GroundedSkybox(texture, 1, 8, 8)
  skybox.position.y = 1

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  scene.add(skybox)

  const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 50)
  camera.position.set(0, 1, 0.1)
  camera.lookAt(0, 1, -1)
  camera.updateMatrixWorld(true)

  try {
    const rgba = renderRgba(scene, camera, { width: 64, height: 64 })
    const mean = meanRgba(rgba)
    assert.equal(skybox.material.map, texture)
    assert.ok(
      nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.9,
      'GroundedSkybox should fill the view with generated textured geometry',
    )
    assert.ok(
      mean.r > 40 && mean.b > 40,
      `GroundedSkybox should sample visible texture colors (${mean.r}, ${mean.g}, ${mean.b})`,
    )
  } finally {
    skybox.geometry.dispose()
    skybox.material.dispose()
    texture.dispose()
  }
})

test('examples DebugEnvironment and RoomEnvironment render generated scene geometry', () => {
  for (const [label, Environment, minVisibleRatio] of [
    ['DebugEnvironment', DebugEnvironment, 0.9],
    ['RoomEnvironment', RoomEnvironment, 0.75],
  ]) {
    const scene = new Environment()
    scene.background = new THREE.Color(0x000000)

    const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100)
    camera.position.set(0, 2, 12)
    camera.lookAt(0, 2, 0)
    camera.updateMatrixWorld(true)

    try {
      const rgba = renderRgba(scene, camera, { width: 64, height: 64 })
      const mean = meanRgba(rgba)
      assert.ok(
        nonBackgroundRatio(rgba, [0, 0, 0], 3) > minVisibleRatio,
        `${label} should render substantial generated environment geometry`,
      )
      assert.ok(
        mean.r > 40 && mean.g > 40 && mean.b > 40,
        `${label} should render lit built-in materials (${mean.r}, ${mean.g}, ${mean.b})`,
      )
    } finally {
      scene.dispose?.()
    }
  }
})

test('MarchingCubes renders generated BufferGeometry with built-in materials', () => {
  const material = new THREE.MeshBasicMaterial({ color: 0x00ff80 })
  const cubes = new MarchingCubes(16, material, false, false, 2000)
  cubes.isolation = 50
  cubes.addBall(0.5, 0.5, 0.5, 1.2, 12)
  cubes.update()

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  scene.add(cubes)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  try {
    const rgba = renderRgba(scene, camera, { width: 64, height: 64 })
    const center = meanRegion(rgba, 64, 64, 24, 24, 40, 40)
    assert.equal(cubes.isMarchingCubes, true)
    assert.ok(cubes.count > 0, 'MarchingCubes should generate triangles after adding a metaball')
    assert.equal(cubes.geometry.drawRange.count, cubes.count)
    assert.ok(
      nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.02,
      'MarchingCubes generated geometry should render visible pixels',
    )
    assert.ok(
      center.g > center.r + 20 && center.b > center.r,
      `MarchingCubes material color should appear in the generated mesh (${center.r}, ${center.g}, ${center.b})`,
    )
  } finally {
    cubes.geometry.dispose()
    material.dispose()
  }
})

test('examples RollerCoaster geometries render generated track, lifter, and shadow meshes', () => {
  const curve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-1.2, 0.35, -0.5),
    new THREE.Vector3(-0.45, 0.9, 0.1),
    new THREE.Vector3(0.35, 0.5, 0.5),
    new THREE.Vector3(1.15, 0.75, -0.45),
  ])
  const trackGeometry = new RollerCoasterGeometry(curve, 18)
  const liftersGeometry = new RollerCoasterLiftersGeometry(curve, 18)
  const shadowGeometry = new RollerCoasterShadowGeometry(curve, 18)
  const trackMaterial = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide })
  const liftersMaterial = new THREE.MeshBasicMaterial({ color: 0xff6622, side: THREE.DoubleSide })
  const shadowMaterial = new THREE.MeshBasicMaterial({ color: 0x222222, side: THREE.DoubleSide })
  const track = new THREE.Mesh(trackGeometry, trackMaterial)
  const lifters = new THREE.Mesh(liftersGeometry, liftersMaterial)
  const shadow = new THREE.Mesh(shadowGeometry, shadowMaterial)
  shadow.position.y = -0.02

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  scene.add(shadow, lifters, track)

  const camera = new THREE.OrthographicCamera(-1.8, 1.8, 1.3, -0.5, 0.01, 10)
  camera.position.set(0, 2.0, 3.2)
  camera.lookAt(0, 0.35, 0)
  camera.updateMatrixWorld(true)

  try {
    const width = 128
    const height = 88
    const rgba = renderRgba(scene, camera, { width, height })
    assert.ok(trackGeometry.getAttribute('position').count > 1000)
    assert.ok(trackGeometry.getAttribute('normal').count > 1000)
    assert.equal(trackGeometry.getAttribute('color').count, trackGeometry.getAttribute('position').count)
    assert.ok(liftersGeometry.getAttribute('position').count > 100)
    assert.ok(shadowGeometry.getAttribute('position').count > 50)
    assert.ok(
      nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.2,
      'RollerCoaster generated geometries should render visible pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => r > 180 && g > 180 && b > 180) > 120,
      'RollerCoasterGeometry vertex colors should render white track pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => r > 150 && g > 70 && g < 170 && b < 90) > 300,
      'RollerCoasterLiftersGeometry should render orange support pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => r > 20 && r < 80 && g > 20 && g < 80 && b > 20 && b < 80) > 120,
      'RollerCoasterShadowGeometry should render dark projected pixels',
    )
  } finally {
    trackGeometry.dispose()
    liftersGeometry.dispose()
    shadowGeometry.dispose()
    trackMaterial.dispose()
    liftersMaterial.dispose()
    shadowMaterial.dispose()
  }
})

test('examples RollerCoaster sky and tree geometries render seeded generated meshes', () => {
  const originalRandom = Math.random
  const seededRandom = () => {
    let seed = 1234
    return () => {
      seed = (seed * 1664525 + 1013904223) >>> 0
      return seed / 0x100000000
    }
  }

  let skyGeometry
  let treesGeometry
  const landscapeGeometry = new THREE.PlaneGeometry(600, 600)
  const landscapeMaterial = new THREE.MeshBasicMaterial()
  const landscape = new THREE.Mesh(landscapeGeometry, landscapeMaterial)
  landscape.rotation.x = -Math.PI / 2
  landscape.updateMatrixWorld(true)

  try {
    Math.random = seededRandom()
    skyGeometry = new SkyGeometry()
    treesGeometry = new TreesGeometry(landscape)
  } finally {
    Math.random = originalRandom
  }

  const skyMaterial = new THREE.MeshBasicMaterial({ color: 0x3366ff, side: THREE.DoubleSide })
  const treesMaterial = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide })
  const sky = new THREE.Mesh(skyGeometry, skyMaterial)
  sky.scale.setScalar(0.005)
  sky.position.y = 0.15
  const trees = new THREE.Mesh(treesGeometry, treesMaterial)
  trees.scale.set(0.004, 0.06, 0.004)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  scene.add(sky, trees)

  const camera = new THREE.OrthographicCamera(-1.5, 1.5, 1.0, -0.35, 0.01, 10)
  camera.position.set(0, 1.2, 3.2)
  camera.lookAt(0, 0.2, 0)
  camera.updateMatrixWorld(true)

  try {
    const width = 128
    const height = 88
    const rgba = renderRgba(scene, camera, { width, height })
    assert.equal(skyGeometry.getAttribute('position').count, 600)
    assert.equal(treesGeometry.getAttribute('position').count, 12000)
    assert.equal(treesGeometry.getAttribute('color').count, treesGeometry.getAttribute('position').count)
    assert.ok(
      nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.5,
      'RollerCoaster SkyGeometry and TreesGeometry should render visible generated meshes',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => b > 140 && r < 120 && g > 70 && g < 180) > 300,
      'SkyGeometry should render blue sky pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => g > 70 && g > r + 20 && g > b + 20) > 300,
      'TreesGeometry vertex colors should render green tree pixels',
    )
  } finally {
    skyGeometry.dispose()
    treesGeometry.dispose()
    landscapeGeometry.dispose()
    landscapeMaterial.dispose()
    skyMaterial.dispose()
    treesMaterial.dispose()
  }
})
