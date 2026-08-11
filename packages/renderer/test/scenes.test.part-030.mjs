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
test('examples color and noise math utilities produce renderable scene inputs', () => {
  let seed = 17
  const random = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0
    return seed / 0x100000000
  }

  const hues = [0, 1 / 3, 2 / 3]
  const planeGeometry = new THREE.PlaneGeometry(0.36, 0.34)
  const colorMeshes = []
  const colorMaterials = []

  for (let i = 0; i < hues.length; i += 1) {
    const color = ColorConverter.setHSV(new THREE.Color(), hues[i], 1, 1)
    const material = new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide })
    const mesh = new THREE.Mesh(planeGeometry, material)
    mesh.position.set(-0.62 + i * 0.62, 0.24, 0)
    colorMeshes.push(mesh)
    colorMaterials.push(material)
  }

  const hsv = ColorConverter.getHSV(colorMaterials[1].color, {})
  assert.ok(Math.abs(hsv.h - 1 / 3) < 1e-6)
  assert.ok(Math.abs(hsv.s - 1) < 1e-6)
  assert.ok(Math.abs(hsv.v - 1) < 1e-6)

  const improvedNoise = new ImprovedNoise()
  const simplexNoise = new SimplexNoise({ random })
  const points = []
  const yValues = []
  for (let i = 0; i < 32; i += 1) {
    const x = -0.9 + (i / 31) * 1.8
    const noiseValue = improvedNoise.noise(i * 0.21, 0.4, 0.9) + simplexNoise.noise(i * 0.16, 0.3)
    const y = -0.42 + noiseValue * 0.08
    points.push(new THREE.Vector3(x, y, 0))
    yValues.push(y)
  }
  assert.ok(Math.max(...yValues) - Math.min(...yValues) > 0.02, 'noise helpers should perturb the generated path')

  const lineGeometry = new THREE.BufferGeometry().setFromPoints(points)
  const lineMaterial = new THREE.LineBasicMaterial({ color: 0xffff44, linewidth: 4 })
  const line = new THREE.Line(lineGeometry, lineMaterial)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  scene.add(...colorMeshes, line)

  const camera = new THREE.OrthographicCamera(-1, 1, 0.75, -0.75, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  try {
    const width = 128
    const height = 72
    const rgba = renderRgba(scene, camera, { width, height })

    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => r > 150 && g < 120 && b < 120) > 150,
      'ColorConverter HSV red should render visible red pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => g > 120 && g > r + 20 && g > b + 20) > 150,
      'ColorConverter HSV green should render visible green pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => b > 150 && r < 120 && g < 170) > 150,
      'ColorConverter HSV blue should render visible blue pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => r > 150 && g > 140 && b < 140) > 20,
      'noise-generated line path should render visible yellow pixels',
    )
  } finally {
    planeGeometry.dispose()
    for (const material of colorMaterials) material.dispose()
    lineGeometry.dispose()
    lineMaterial.dispose()
  }
})

test('examples Gyroscope preserves child orientation while rendering normal mesh paths', () => {
  const geometry = new THREE.PlaneGeometry(0.46, 0.18)
  const redMaterial = new THREE.MeshBasicMaterial({ color: 0xff3344, side: THREE.DoubleSide })
  const greenMaterial = new THREE.MeshBasicMaterial({ color: 0x44ff66, side: THREE.DoubleSide })

  const normalParent = new THREE.Object3D()
  normalParent.position.x = -0.45
  normalParent.rotation.z = Math.PI / 2
  const normalMesh = new THREE.Mesh(geometry, redMaterial)
  normalParent.add(normalMesh)

  const gyroParent = new THREE.Object3D()
  gyroParent.position.x = 0.45
  gyroParent.rotation.z = Math.PI / 2
  const gyroscope = new Gyroscope()
  const gyroMesh = new THREE.Mesh(geometry, greenMaterial)
  gyroscope.add(gyroMesh)
  gyroParent.add(gyroscope)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  scene.add(normalParent, gyroParent)

  const camera = new THREE.OrthographicCamera(-1, 1, 0.7, -0.7, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  try {
    const width = 128
    const height = 72
    const rgba = renderRgba(scene, camera, { width, height })

    scene.updateMatrixWorld(true)
    const normalQuaternion = new THREE.Quaternion()
    const gyroQuaternion = new THREE.Quaternion()
    normalMesh.matrixWorld.decompose(new THREE.Vector3(), normalQuaternion, new THREE.Vector3())
    gyroMesh.matrixWorld.decompose(new THREE.Vector3(), gyroQuaternion, new THREE.Vector3())

    assert.ok(Math.abs(normalQuaternion.angleTo(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI / 2))) < 1e-6)
    assert.ok(Math.abs(gyroQuaternion.angleTo(new THREE.Quaternion())) < 1e-6)
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => r > 150 && g < 120 && b < 140) > 100,
      'normally parented mesh should render red pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => g > 150 && g > r + 20 && g > b + 20) > 100,
      'Gyroscope child mesh should render green pixels',
    )
  } finally {
    geometry.dispose()
    redMaterial.dispose()
    greenMaterial.dispose()
  }
})

test('Three.js Timer utility can drive still-frame render state', () => {
  const timer = new Timer()
  timer.setTimescale(2)
  timer.update(timer._startTime + 250)

  const secondTimer = new Timer()
  secondTimer.setTimescale(1)
  secondTimer.update(secondTimer._startTime + 500)

  assert.ok(Math.abs(timer.getDelta() - 0.5) < 1e-6)
  assert.ok(Math.abs(timer.getElapsed() - 0.5) < 1e-6)
  assert.ok(Math.abs(secondTimer.getDelta() - 0.5) < 1e-6)
  assert.ok(Math.abs(secondTimer.getElapsed() - 0.5) < 1e-6)

  const geometry = new THREE.PlaneGeometry(0.36, 0.36)
  const redMaterial = new THREE.MeshBasicMaterial({ color: 0xff3344, side: THREE.DoubleSide })
  const greenMaterial = new THREE.MeshBasicMaterial({ color: 0x44ff66, side: THREE.DoubleSide })
  const red = new THREE.Mesh(geometry, redMaterial)
  red.position.x = -timer.getElapsed()
  const green = new THREE.Mesh(geometry, greenMaterial)
  green.position.x = secondTimer.getElapsed() + 0.2

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  scene.add(red, green)

  const camera = new THREE.OrthographicCamera(-1, 1, 0.7, -0.7, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  try {
    const width = 128
    const height = 72
    const rgba = renderRgba(scene, camera, { width, height })
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width / 2, height, (r, g, b) => r > 150 && g < 120 && b < 140) > 120,
      'Timer-driven red mesh should render on the left',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, width / 2, 0, width, height, (r, g, b) => g > 150 && g > r + 20 && g > b + 20) > 120,
      'second Timer-driven green mesh should render on the right',
    )
  } finally {
    timer.dispose()
    secondTimer.dispose()
    geometry.dispose()
    redMaterial.dispose()
    greenMaterial.dispose()
  }
})

test('examples morph animation helpers drive renderable CPU morph target state', () => {
  const createMorphPlaneGeometry = (targetNames, offsetX) => {
    const geometry = new THREE.PlaneGeometry(0.34, 0.34)
    const position = geometry.getAttribute('position')
    geometry.morphTargetsRelative = true
    geometry.morphAttributes.position = targetNames.map((name) => {
      const values = new Float32Array(position.count * 3)
      for (let i = 0; i < position.count; i += 1) values[i * 3] = offsetX
      const attribute = new THREE.Float32BufferAttribute(values, 3)
      attribute.name = name
      return attribute
    })
    return geometry
  }

  const animGeometry = createMorphPlaneGeometry(['enter'], 2.6)
  animGeometry.animations = [
    new THREE.AnimationClip('enter', 1, [
      new THREE.NumberKeyframeTrack('.morphTargetInfluences[0]', [0, 1], [0, 1]),
    ]),
  ]
  const animMaterial = new THREE.MeshBasicMaterial({ color: 0x44ff66, side: THREE.DoubleSide })
  const animMesh = new MorphAnimMesh(animGeometry, animMaterial)
  animMesh.position.x = -1.8
  animMesh.frustumCulled = false
  animMesh.playAnimation('enter', 1)
  animMesh.updateAnimation(0.5)

  const blendGeometry = createMorphPlaneGeometry(['pulse_0', 'pulse_1'], -1.3)
  const blendMaterial = new THREE.MeshBasicMaterial({ color: 0xff4455, side: THREE.DoubleSide })
  const blendMesh = new MorphBlendMesh(blendGeometry, blendMaterial)
  blendMesh.position.x = 1.8
  blendMesh.frustumCulled = false
  blendMesh.autoCreateAnimations(2)
  blendMesh.playAnimation('pulse')
  blendMesh.update(0.2)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  scene.add(animMesh, blendMesh)

  const camera = new THREE.OrthographicCamera(-1, 1, 0.7, -0.7, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  try {
    const width = 128
    const height = 72
    const rgba = renderRgba(scene, camera, { width, height })

    assert.ok(animMesh.morphTargetInfluences[0] > 0.45 && animMesh.morphTargetInfluences[0] < 0.55)
    assert.ok(blendMesh.morphTargetInfluences[0] > 0.9)
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => g > 150 && g > r + 20 && g > b + 20) > 100,
      'MorphAnimMesh helper-updated morph target should render green pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => r > 150 && g < 140 && b < 140) > 100,
      'MorphBlendMesh helper-updated morph target should render red pixels',
    )
  } finally {
    animGeometry.dispose()
    animMaterial.dispose()
    blendGeometry.dispose()
    blendMaterial.dispose()
  }
})
