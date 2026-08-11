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
import { assertRgbClose, getRenderer, makeCamera, meanRegion, objectIdBytes, renderRgba } from './scenes.test.part-002.mjs'
test('invalid output dimensions fail clearly', () => {
  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial({ color: 0xffffff })))
  const camera = makeCamera()

  assert.throws(
    () => getRenderer().render(scene, camera, { width: '64', height: 32 }),
    /options\.width must be a finite number/i,
  )
  assert.throws(
    () => getRenderer().render(scene, camera, { width: 32, height: 0 }),
    /options\.height must be a positive integer/i,
  )
  assert.throws(
    () => renderToTarget(scene, camera, { width: Number.NaN, height: 32 }),
    /target\.width must be a finite number/i,
  )

  const userDataCamera = makeCamera()
  userDataCamera.userData.width = 32.5
  assert.throws(
    () => getRenderer().render(scene, userDataCamera, { format: 'rgba' }),
    /camera\.userData\.width must be a positive integer/i,
  )

  const userDataContainerCamera = makeCamera()
  userDataContainerCamera.userData = 'size'
  assert.throws(
    () => getRenderer().render(scene, userDataContainerCamera, { format: 'rgba' }),
    /camera\.userData must be an object/i,
  )

  const invalidAspectCamera = makeCamera()
  invalidAspectCamera.aspect = Number.NaN
  assert.throws(
    () => getRenderer().render(scene, invalidAspectCamera, { width: 32, format: 'rgba' }),
    /camera\.aspect must be a finite number/i,
  )

  const zeroAspectCamera = makeCamera()
  zeroAspectCamera.aspect = 0
  assert.throws(
    () => getRenderer().render(scene, zeroAspectCamera, { height: 32, format: 'rgba' }),
    /camera\.aspect must be positive/i,
  )
})

test('invalid camera clipping distances fail clearly', () => {
  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial({ color: 0xffffff })))
  const camera = makeCamera()
  camera.near = Number.NaN

  assert.throws(
    () => getRenderer().render(scene, camera, { width: 32, height: 32 }),
    /camera\.near must be a finite number/i,
  )

  camera.near = 0
  assert.throws(
    () => getRenderer().render(scene, camera, { width: 32, height: 32 }),
    /camera\.near must be positive/i,
  )

  camera.near = 0.01
  camera.far = 'deep'
  assert.throws(
    () => getRenderer().render(scene, camera, { width: 32, height: 32 }),
    /camera\.far must be a finite number/i,
  )

  camera.far = -1
  assert.throws(
    () => getRenderer().render(scene, camera, { width: 32, height: 32 }),
    /camera\.far must be positive/i,
  )

  camera.near = 10
  camera.far = 1
  assert.throws(
    () => getRenderer().render(scene, camera, { width: 32, height: 32 }),
    /camera\.far must be greater than camera\.near/i,
  )

  camera.near = 0.01
  camera.far = 100
  camera.projectionMatrix.elements[0] = Number.NaN
  assert.throws(
    () => getRenderer().render(scene, camera, { width: 32, height: 32 }),
    /camera\.projectionMatrix\.elements\[0\] must be a finite number/i,
  )

  const missingMatrixCamera = makeCamera()
  missingMatrixCamera.projectionMatrix = null
  assert.throws(
    () => getRenderer().render(scene, missingMatrixCamera, { width: 32, height: 32 }),
    /THREE\.Camera must have projectionMatrix and matrixWorldInverse/i,
  )

  const projectionContainerCamera = makeCamera()
  projectionContainerCamera.projectionMatrix = { elements: [1, 2, 3] }
  assert.throws(
    () => getRenderer().render(scene, projectionContainerCamera, { width: 32, height: 32 }),
    /camera\.projectionMatrix must be a THREE\.Matrix4/i,
  )

  const viewMatrixCamera = makeCamera()
  viewMatrixCamera.matrixWorldInverse.elements[4] = Number.NaN
  assert.throws(
    () => getRenderer().render(scene, viewMatrixCamera, { width: 32, height: 32 }),
    /camera\.matrixWorldInverse\.elements\[4\] must be a finite number/i,
  )

  const worldMatrixCamera = makeCamera()
  worldMatrixCamera.updateMatrixWorld = () => {}
  worldMatrixCamera.matrixWorld.elements[12] = Number.NaN
  assert.throws(
    () => getRenderer().render(scene, worldMatrixCamera, { width: 32, height: 32 }),
    /camera\.matrixWorld\.elements\[12\] must be a finite number/i,
  )

  const matrixContainerCamera = makeCamera()
  matrixContainerCamera.updateMatrixWorld = () => {}
  matrixContainerCamera.matrixWorld = 'world'
  assert.throws(
    () => getRenderer().render(scene, matrixContainerCamera, { width: 32, height: 32 }),
    /camera\.matrixWorld must be a THREE\.Matrix4/i,
  )
})

export function makeLayeredArrayCamera(width = 64, height = 64) {
  const leftCamera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  leftCamera.position.set(0, 0, 3)
  leftCamera.lookAt(0, 0, 0)
  leftCamera.layers.set(1)
  leftCamera.viewport = new THREE.Vector4(0, 0, width / 2, height)

  const rightCamera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  rightCamera.position.set(0, 0, 3)
  rightCamera.lookAt(0, 0, 0)
  rightCamera.layers.set(2)
  rightCamera.viewport = new THREE.Vector4(width / 2, 0, width / 2, height)

  return new THREE.ArrayCamera([leftCamera, rightCamera])
}

export function makeLayeredSplitScene() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 1)
  const red = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ color: 0xff0000 }))
  red.layers.set(1)
  scene.add(red)
  const green = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ color: 0x00ff00 }))
  green.layers.set(2)
  scene.add(green)
  return scene
}

export function makeCubeCaptureScene() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const addPlane = (position, rotation, color) => {
    const plane = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide }),
    )
    plane.position.set(...position)
    plane.rotation.set(...rotation)
    scene.add(plane)
  }
  addPlane([2, 0, 0], [0, Math.PI / 2, 0], 0xff0000)
  addPlane([-2, 0, 0], [0, Math.PI / 2, 0], 0x00ff00)
  addPlane([0, 2, 0], [Math.PI / 2, 0, 0], 0x0000ff)
  addPlane([0, -2, 0], [Math.PI / 2, 0, 0], 0xffff00)
  addPlane([0, 0, 2], [0, 0, 0], 0xff00ff)
  addPlane([0, 0, -2], [0, 0, 0], 0x00ffff)
  return scene
}

test('ArrayCamera renders sub-camera viewports', () => {
  const scene = makeLayeredSplitScene()
  const arrayCamera = makeLayeredArrayCamera()

  const rgba = renderRgba(scene, arrayCamera, { width: 64, height: 64 })
  const left = meanRegion(rgba, 64, 64, 8, 20, 24, 44)
  const right = meanRegion(rgba, 64, 64, 40, 20, 56, 44)
  assert.ok(left.r > left.g + 80 && left.r > left.b + 80, `left ArrayCamera viewport should render the red layer (${left.r}, ${left.g}, ${left.b})`)
  assert.ok(right.g > right.r + 80 && right.g > right.b + 80, `right ArrayCamera viewport should render the green layer (${right.r}, ${right.g}, ${right.b})`)

  const target = { texture: {}, depthTexture: {} }
  renderToTarget(scene, arrayCamera, target, { width: 64, height: 64 })
  assert.equal(target.data.length, 64 * 64 * 4)
  assert.equal(target.depthTexture.image.data.length, 64 * 64 * 4)
  const targetLeft = meanRegion(target.data, 64, 64, 8, 20, 24, 44)
  const targetRight = meanRegion(target.data, 64, 64, 40, 20, 56, 44)
  const depthLeft = meanRegion(target.depthTexture.image.data, 64, 64, 8, 20, 24, 44)
  const depthRight = meanRegion(target.depthTexture.image.data, 64, 64, 40, 20, 56, 44)
  assert.ok(targetLeft.r > targetLeft.g + 80, `target left ArrayCamera viewport should render red (${targetLeft.r}, ${targetLeft.g})`)
  assert.ok(targetRight.g > targetRight.r + 80, `target right ArrayCamera viewport should render green (${targetRight.g}, ${targetRight.r})`)
  assert.ok(depthLeft.r > 0 && depthRight.r > 0, `ArrayCamera depth target should include both viewports (${depthLeft.r}, ${depthRight.r})`)

  const rendererTarget = { texture: {}, depthTexture: {} }
  const renderer = new Renderer()
  renderer.setRenderTarget(rendererTarget)
  const rendererReturned = renderer.render(scene, arrayCamera, { width: 64, height: 64 })
  assert.equal(rendererReturned, rendererTarget.data)
  const rendererLeft = meanRegion(rendererTarget.data, 64, 64, 8, 20, 24, 44)
  const rendererRight = meanRegion(rendererTarget.data, 64, 64, 40, 20, 56, 44)
  const rendererDepthLeft = meanRegion(rendererTarget.depthTexture.image.data, 64, 64, 8, 20, 24, 44)
  const rendererDepthRight = meanRegion(rendererTarget.depthTexture.image.data, 64, 64, 40, 20, 56, 44)
  assert.ok(rendererLeft.r > rendererLeft.g + 80, `Renderer.setRenderTarget ArrayCamera left viewport should render red (${rendererLeft.r}, ${rendererLeft.g})`)
  assert.ok(rendererRight.g > rendererRight.r + 80, `Renderer.setRenderTarget ArrayCamera right viewport should render green (${rendererRight.g}, ${rendererRight.r})`)
  assert.ok(rendererDepthLeft.r > 0 && rendererDepthRight.r > 0, `Renderer.setRenderTarget ArrayCamera depth should include both viewports (${rendererDepthLeft.r}, ${rendererDepthRight.r})`)
  renderer.setRenderTarget(null)
})

test('ArrayCamera supports PNG output', () => {
  const scene = makeLayeredSplitScene()
  const arrayCamera = makeLayeredArrayCamera()
  assertValidPng(getRenderer().render(scene, arrayCamera, { width: 64, height: 64 }), { width: 64, height: 64 })
})

test('ArrayCamera object-id target merges sub-camera metadata', () => {
  const scene = makeLayeredSplitScene()
  const arrayCamera = makeLayeredArrayCamera()
  const [red, green] = scene.children
  const target = { texture: {} }

  renderToTarget(scene, arrayCamera, target, { width: 64, height: 64, renderMode: 'object-id' })

  const redEncoded = red.id + 1
  const greenEncoded = green.id + 1
  const left = meanRegion(target.data, 64, 64, 8, 20, 24, 44)
  const right = meanRegion(target.data, 64, 64, 40, 20, 56, 44)
  assertRgbClose(left, objectIdBytes(redEncoded), 'left ArrayCamera object id')
  assertRgbClose(right, objectIdBytes(greenEncoded), 'right ArrayCamera object id')
  assert.equal(target.objectIdEntries.length, 2)
  assert.equal(target.objectIdMap[String(redEncoded)].id, red.id)
  assert.equal(target.objectIdMap[String(greenEncoded)].id, green.id)
})

test('ArrayCamera supports auxiliary MRT-shaped target attachments', () => {
  const scene = makeLayeredSplitScene()
  const arrayCamera = makeLayeredArrayCamera()
  const [red, green] = scene.children
  const target = {
    textures: [
      {},
      { userData: { headlessThreeRenderer: { renderMode: 'color' } } },
      { userData: { headlessThreeRenderer: { renderMode: 'mask' } } },
      { userData: { headlessThreeRenderer: { renderMode: 'object-id' } } },
      { userData: { headlessThreeRenderer: { renderMode: 'normal' } } },
      { userData: { headlessThreeRenderer: { renderMode: 'depth' } } },
    ],
  }

  renderToTarget(scene, arrayCamera, target, {
    width: 64,
    height: 64,
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  const colorLeft = meanRegion(target.textures[0].image.data, 64, 64, 8, 20, 24, 44)
  const colorRight = meanRegion(target.textures[0].image.data, 64, 64, 40, 20, 56, 44)
  assert.ok(colorLeft.r > colorLeft.g + 80, `primary ArrayCamera attachment left viewport should render red (${colorLeft.r}, ${colorLeft.g})`)
  assert.ok(colorRight.g > colorRight.r + 80, `primary ArrayCamera attachment right viewport should render green (${colorRight.g}, ${colorRight.r})`)

  const colorCopyLeft = meanRegion(target.textures[1].image.data, 64, 64, 8, 20, 24, 44)
  const colorCopyRight = meanRegion(target.textures[1].image.data, 64, 64, 40, 20, 56, 44)
  assert.ok(colorCopyLeft.r > colorCopyLeft.g + 80, `secondary color ArrayCamera attachment left viewport should render red (${colorCopyLeft.r}, ${colorCopyLeft.g})`)
  assert.ok(colorCopyRight.g > colorCopyRight.r + 80, `secondary color ArrayCamera attachment right viewport should render green (${colorCopyRight.g}, ${colorCopyRight.r})`)

  const maskLeft = meanRegion(target.textures[2].image.data, 64, 64, 8, 20, 24, 44)
  const maskRight = meanRegion(target.textures[2].image.data, 64, 64, 40, 20, 56, 44)
  assert.ok(maskLeft.r > 250 && maskRight.r > 250, `mask attachment should compose both viewports (${maskLeft.r}, ${maskRight.r})`)

  const objectIdLeft = meanRegion(target.textures[3].image.data, 64, 64, 8, 20, 24, 44)
  const objectIdRight = meanRegion(target.textures[3].image.data, 64, 64, 40, 20, 56, 44)
  const redEncoded = red.id + 1
  const greenEncoded = green.id + 1
  assertRgbClose(objectIdLeft, objectIdBytes(redEncoded), 'left auxiliary ArrayCamera object id')
  assertRgbClose(objectIdRight, objectIdBytes(greenEncoded), 'right auxiliary ArrayCamera object id')
  assert.equal(target.objectIdEntries.length, 2)
  assert.equal(target.objectIdMap[String(redEncoded)].id, red.id)
  assert.equal(target.objectIdMap[String(greenEncoded)].id, green.id)

  const normalLeft = meanRegion(target.textures[4].image.data, 64, 64, 8, 20, 24, 44)
  const normalRight = meanRegion(target.textures[4].image.data, 64, 64, 40, 20, 56, 44)
  assert.ok(normalLeft.b > 250 && normalRight.b > 250, `normal attachment should compose both viewports (${normalLeft.b}, ${normalRight.b})`)

  const depthLeft = meanRegion(target.textures[5].image.data, 64, 64, 8, 20, 24, 44)
  const depthRight = meanRegion(target.textures[5].image.data, 64, 64, 40, 20, 56, 44)
  assert.ok(depthLeft.r > 0 && depthRight.r > 0, `depth attachment should compose both viewports (${depthLeft.r}, ${depthRight.r})`)
})
