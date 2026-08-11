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
import { countRegionPixels, renderRgba } from './scenes.test.part-002.mjs'
test('examples CCDIKSolver updates and renders supported helper geometry', () => {
  const geometry = new THREE.BufferGeometry()
  const material = new THREE.MeshBasicMaterial()
  const mesh = new THREE.SkinnedMesh(geometry, material)
  const rootBone = new THREE.Bone()
  rootBone.name = 'root'
  const targetBone = new THREE.Bone()
  targetBone.name = 'target'
  targetBone.position.set(0.4, 0.35, 0)
  rootBone.add(targetBone)
  const linkBone = new THREE.Bone()
  linkBone.name = 'link'
  rootBone.add(linkBone)
  const effectorBone = new THREE.Bone()
  effectorBone.name = 'effector'
  effectorBone.position.set(0, 0.55, 0)
  linkBone.add(effectorBone)
  mesh.add(rootBone)
  mesh.bind(new THREE.Skeleton([targetBone, linkBone, effectorBone]))
  mesh.updateMatrixWorld(true)

  const targetPosition = new THREE.Vector3().setFromMatrixPosition(targetBone.matrixWorld)
  const beforeEffectorPosition = new THREE.Vector3().setFromMatrixPosition(effectorBone.matrixWorld)
  const beforeDistance = beforeEffectorPosition.distanceTo(targetPosition)

  const ik = {
    target: 0,
    effector: 2,
    links: [{ index: 1 }],
    iteration: 8,
    minAngle: 0,
    maxAngle: 0.35,
  }
  const solver = new CCDIKSolver(mesh, [ik])
  solver.update()
  mesh.updateMatrixWorld(true)
  const afterEffectorPosition = new THREE.Vector3().setFromMatrixPosition(effectorBone.matrixWorld)
  const afterDistance = afterEffectorPosition.distanceTo(targetPosition)

  const helper = solver.createHelper(0.1)
  helper.lineMaterial.linewidth = 4
  helper.updateMatrixWorld(true)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  scene.add(helper)

  const camera = new THREE.OrthographicCamera(-0.8, 0.8, 0.8, -0.2, 0.01, 10)
  camera.position.set(0, 0, 4)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  try {
    const width = 96
    const height = 72
    const rgba = renderRgba(scene, camera, { width, height })

    assert.ok(afterDistance < beforeDistance, `CCDIKSolver should move effector closer to target (${beforeDistance} -> ${afterDistance})`)
    assert.ok(Math.abs(linkBone.rotation.z) > 0.2, 'CCDIKSolver should rotate the IK link bone')
    assert.equal(helper.children.length, 4, 'CCDIKHelper should create target, effector, link, and line children')
    assert.ok(helper.children.some((child) => child.isLine), 'CCDIKHelper should include a line path')
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => r > 150 && g < 120 && b < 120) > 100,
      'CCDIKHelper line path should render red pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => g > 140 && g > r + 30 && g > b + 20) > 80,
      'CCDIKHelper effector sphere should render green pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => b > 140 && b > r + 20 && b > g + 20) > 80,
      'CCDIKHelper link sphere should render blue pixels',
    )
  } finally {
    helper.dispose()
    geometry.dispose()
    material.dispose()
  }
})

test('examples CameraUtils.frameCorners renders off-axis framed scene content', () => {
  const width = 96
  const height = 64
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  const planeGeometry = new THREE.PlaneGeometry(1, 0.6)
  const planeMaterial = new THREE.MeshBasicMaterial({ color: 0x44aaff, side: THREE.DoubleSide })
  scene.add(new THREE.Mesh(planeGeometry, planeMaterial))

  const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 10)
  camera.position.set(0, 0, 1)
  frameCorners(
    camera,
    new THREE.Vector3(-0.5, -0.3, 0),
    new THREE.Vector3(0.5, -0.3, 0),
    new THREE.Vector3(-0.5, 0.3, 0),
    true,
  )
  camera.updateMatrixWorld(true)

  try {
    const rgba = renderRgba(scene, camera, { width, height })
    const projectionIdentity = camera.projectionMatrix.clone().multiply(camera.projectionMatrixInverse)

    assert.ok(camera.fov > 0 && camera.fov < 90, `frameCorners should estimate a usable culling fov (${camera.fov})`)
    assert.ok(Math.abs(projectionIdentity.elements[0] - 1) < 1e-6)
    assert.ok(Math.abs(projectionIdentity.elements[5] - 1) < 1e-6)
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => b > 150 && g > 100 && r < 120) > 5800,
      'CameraUtils.frameCorners should frame the blue plane across the render target',
    )
  } finally {
    planeGeometry.dispose()
    planeMaterial.dispose()
  }
})

test('examples camera controls drive renderable still-frame camera and helper state', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)

  const redGeometry = new THREE.PlaneGeometry(0.5, 0.5)
  const redMaterial = new THREE.MeshBasicMaterial({ color: 0xff3344, side: THREE.DoubleSide })
  const red = new THREE.Mesh(redGeometry, redMaterial)
  red.position.x = -0.45
  scene.add(red)

  const blueGeometry = new THREE.PlaneGeometry(0.5, 0.5)
  const blueMaterial = new THREE.MeshBasicMaterial({ color: 0x44aaff, side: THREE.DoubleSide })
  const blue = new THREE.Mesh(blueGeometry, blueMaterial)
  blue.position.x = 0.45
  scene.add(blue)

  const orbitCamera = new THREE.PerspectiveCamera(50, 1, 0.1, 10)
  orbitCamera.position.set(0, 0, 3)
  const orbit = new OrbitControls(orbitCamera, null)
  orbit.target.copy(red.position)
  orbit.update()
  const orbitDirection = orbitCamera.getWorldDirection(new THREE.Vector3())
  assert.ok(orbitDirection.x < -0.1, 'OrbitControls should orient the camera toward its target')

  const mapCamera = orbitCamera.clone()
  mapCamera.position.set(0, 0, 3)
  const mapControls = new MapControls(mapCamera, null)
  mapControls.target.copy(blue.position)
  mapControls.update()
  const mapDirection = mapCamera.getWorldDirection(new THREE.Vector3())
  assert.equal(mapControls.screenSpacePanning, false)
  assert.ok(mapDirection.x > 0.1, 'MapControls should orient the camera toward its target')

  const trackballCamera = orbitCamera.clone()
  trackballCamera.position.set(0, 0, 3)
  const trackball = new TrackballControls(trackballCamera, null)
  trackball.target.copy(blue.position)
  trackball.update()
  const trackballDirection = trackballCamera.getWorldDirection(new THREE.Vector3())
  assert.ok(trackballDirection.x > 0.1, 'TrackballControls should orient the camera toward its target')

  const firstPersonCamera = new THREE.PerspectiveCamera(50, 1, 0.1, 10)
  firstPersonCamera.position.set(0, 0, 3)
  const firstPerson = new FirstPersonControls(firstPersonCamera, null)
  firstPerson.autoForward = true
  firstPerson.lookAt(0, 0, 0)
  firstPerson.update(0.5)
  assert.ok(firstPersonCamera.position.z < 2.6, 'FirstPersonControls should advance an auto-forward still frame')

  const flyCamera = new THREE.PerspectiveCamera(50, 1, 0.1, 10)
  flyCamera.position.set(0, 0, 3)
  const fly = new FlyControls(flyCamera, null)
  fly.movementSpeed = 2
  fly._moveState.forward = 1
  fly._moveState.right = 1
  fly._updateMovementVector()
  fly.update(0.5)
  assert.ok(flyCamera.position.z < 2.2 && flyCamera.position.x > 0.8, 'FlyControls should apply deterministic movement vectors')

  const pointerCamera = new THREE.PerspectiveCamera(50, 1, 0.1, 10)
  pointerCamera.position.set(0, 0, 3)
  const pointerLock = new PointerLockControls(pointerCamera, null)
  pointerLock.moveForward(0.5)
  pointerLock.moveRight(0.25)
  assert.ok(pointerCamera.position.z < 2.6 && pointerCamera.position.x > 0.2, 'PointerLockControls should move the camera on the XZ plane')

  const hadWindow = Object.prototype.hasOwnProperty.call(globalThis, 'window')
  const previousWindow = globalThis.window
  globalThis.window = {
    devicePixelRatio: 1,
    addEventListener() {},
    removeEventListener() {},
  }
  const arcballScene = new THREE.Scene()
  arcballScene.background = new THREE.Color(0x000000)
  const arcballCamera = new THREE.PerspectiveCamera(45, 1, 0.1, 10)
  arcballCamera.position.set(0, 0, 4)
  arcballCamera.lookAt(0, 0, 0)
  arcballCamera.updateMatrixWorld(true)
  const arcballDom = {
    style: {},
    addEventListener() {},
    removeEventListener() {},
    getBoundingClientRect() {
      return { left: 0, top: 0, width: 100, height: 100 }
    },
  }
  const arcball = new ArcballControls(arcballCamera, arcballDom, arcballScene)
  arcball.setGizmosVisible(true)
  arcball.update()
  arcballScene.updateMatrixWorld(true)

  const dragScene = new THREE.Scene()
  dragScene.background = new THREE.Color(0x000000)
  const dragCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  dragCamera.position.set(0, 0, 3)
  dragCamera.lookAt(0, 0, 0)
  dragCamera.updateMatrixWorld(true)
  const dragGeometry = new THREE.PlaneGeometry(0.25, 0.25)
  const dragMaterial = new THREE.MeshBasicMaterial({ color: 0x33ff66, side: THREE.DoubleSide })
  const draggable = new THREE.Mesh(dragGeometry, dragMaterial)
  dragScene.add(draggable)
  dragScene.updateMatrixWorld(true)
  const dragListeners = new Map()
  const dragDom = {
    style: {},
    addEventListener(type, listener) {
      dragListeners.set(type, listener)
    },
    removeEventListener() {},
    getBoundingClientRect() {
      return { left: 0, top: 0, width: 100, height: 100 }
    },
  }
  const drag = new DragControls([draggable], dragCamera, dragDom)
  const dragEvents = []
  drag.addEventListener('dragstart', (event) => dragEvents.push(event.type))
  drag.addEventListener('drag', (event) => dragEvents.push(event.type))
  drag.addEventListener('dragend', (event) => dragEvents.push(event.type))
  dragListeners.get('pointerdown')({ clientX: 50, clientY: 50, pointerType: 'mouse', button: 0 })
  dragListeners.get('pointermove')({ clientX: 70, clientY: 50, pointerType: 'mouse', button: 0 })
  dragListeners.get('pointerup')({})
  dragScene.updateMatrixWorld(true)
  assert.deepEqual(dragEvents, ['dragstart', 'drag', 'dragend'])
  assert.ok(draggable.position.x > 0.35, 'DragControls pointer listeners should move selected objects')

  const transformCamera = new THREE.PerspectiveCamera(45, 1, 0.1, 10)
  transformCamera.position.set(0, 0, 4)
  transformCamera.lookAt(0, 0, 0)
  transformCamera.updateMatrixWorld(true)
  const transformScene = new THREE.Scene()
  transformScene.background = new THREE.Color(0x000000)
  const transformTargetGeometry = new THREE.BoxGeometry(0.1, 0.1, 0.1)
  const transformTargetMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff })
  const transformTarget = new THREE.Mesh(transformTargetGeometry, transformTargetMaterial)
  transformTarget.visible = false
  transformScene.add(transformTarget)
  const transformControlsDom = {
    style: {},
    addEventListener() {},
    removeEventListener() {},
  }
  const transformControls = new TransformControls(transformCamera, transformControlsDom)
  transformControls.attach(transformTarget)
  transformControls.setMode('translate')
  transformControls.setSpace('world')
  transformControls.setSize(1.5)
  transformControls.getHelper().updateMatrixWorld(true)
  transformScene.add(transformControls.getHelper())

  try {
    const width = 128
    const height = 96
    const controlsRgba = renderRgba(scene, orbitCamera, { width, height })
    const arcballRgba = renderRgba(arcballScene, arcballCamera, { width, height })
    const dragRgba = renderRgba(dragScene, dragCamera, { width, height })
    const helperRgba = renderRgba(transformScene, transformCamera, { width, height })

    assert.ok(
      countRegionPixels(controlsRgba, width, height, 0, 0, width, height, (r, g, b) => r > 150 && r > g + 40 && r > b + 30) > 200,
      'OrbitControls-targeted camera should render the red target mesh',
    )
    assert.ok(
      countRegionPixels(arcballRgba, width, height, 0, 0, width, height, (r, g, b) => r > 20 || g > 20 || b > 20) > 250,
      'ArcballControls should render built-in gizmo geometry',
    )
    assert.ok(
      countRegionPixels(dragRgba, width, height, width / 2, 0, width, height, (r, g, b) => g > 180 && g > r + 50 && b < 170) > 100,
      'DragControls-moved object should render on the right side',
    )
    assert.ok(
      countRegionPixels(helperRgba, width, height, 0, 0, width, height, (r, g, b) => r > 20 || g > 20 || b > 20) > 80,
      'TransformControls helper should render built-in transform gizmo geometry',
    )
  } finally {
    arcball.dispose()
    drag.dispose()
    if (hadWindow) {
      globalThis.window = previousWindow
    } else {
      delete globalThis.window
    }
    transformControls.dispose()
    redGeometry.dispose()
    redMaterial.dispose()
    blueGeometry.dispose()
    blueMaterial.dispose()
    dragGeometry.dispose()
    dragMaterial.dispose()
    transformTargetGeometry.dispose()
    transformTargetMaterial.dispose()
  }
})
