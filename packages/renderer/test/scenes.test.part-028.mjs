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
import { countRegionPixels, makeCamera, renderRgba, solidTexture } from './scenes.test.part-002.mjs'
test('examples NURBSUtils low-level samplers produce renderable geometry paths', () => {
  const curveKnots = [0, 0, 0, 1, 1, 1]
  const curveControls = [
    new THREE.Vector4(-0.42, -0.26, 0, 1),
    new THREE.Vector4(-0.05, 0.38, 0.18, 0.75),
    new THREE.Vector4(0.42, -0.18, 0, 1),
  ]

  const span = NURBSUtils.findSpan(2, 0.5, curveKnots)
  const basis = NURBSUtils.calcBasisFunctions(span, 0.5, 2, curveKnots)
  const derivatives = NURBSUtils.calcNURBSDerivatives(2, curveKnots, curveControls, 0.5, 1)
  assert.equal(span, 2)
  assert.ok(Math.abs(basis.reduce((sum, value) => sum + value, 0) - 1) < 1e-12, 'basis functions should partition unity')
  assert.ok(derivatives[1].length() > 0.1, 'curve derivative should expose a usable tangent')
  assert.equal(NURBSUtils.calcKoverI(4, 2), 6)

  const curvePoints = []
  for (let i = 0; i <= 24; i++) {
    const point = NURBSUtils.calcBSplinePoint(2, curveKnots, curveControls, i / 24)
    curvePoints.push(new THREE.Vector3(point.x / point.w, point.y / point.w, point.z / point.w))
  }
  const curveGeometry = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(curvePoints), 32, 0.022, 6, false)
  const curveMaterial = new THREE.MeshBasicMaterial({ color: 0xff4444, side: THREE.DoubleSide })
  const curveMesh = new THREE.Mesh(curveGeometry, curveMaterial)
  curveMesh.position.x = -0.78

  const surfaceKnots = [0, 0, 0, 1, 1, 1]
  const surfaceControls = [
    [
      new THREE.Vector4(-0.3, -0.28, 0.02, 1),
      new THREE.Vector4(-0.32, 0, 0.18, 1),
      new THREE.Vector4(-0.3, 0.28, 0.02, 1),
    ],
    [
      new THREE.Vector4(0, -0.3, 0.16, 1),
      new THREE.Vector4(0, 0, 0.34, 0.8),
      new THREE.Vector4(0, 0.3, 0.16, 1),
    ],
    [
      new THREE.Vector4(0.3, -0.28, 0.02, 1),
      new THREE.Vector4(0.32, 0, 0.18, 1),
      new THREE.Vector4(0.3, 0.28, 0.02, 1),
    ],
  ]
  const surfaceGeometry = new ParametricGeometry((u, v, target) => {
    NURBSUtils.calcSurfacePoint(2, 2, surfaceKnots, surfaceKnots, surfaceControls, u, v, target)
  }, 10, 10)
  const surfaceMaterial = new THREE.MeshBasicMaterial({ color: 0x55ff66, side: THREE.DoubleSide })
  const surfaceMesh = new THREE.Mesh(surfaceGeometry, surfaceMaterial)

  const volumeKnots = [0, 0, 1, 1]
  const volumeControls = [
    [
      [new THREE.Vector4(-0.22, -0.22, -0.05, 1), new THREE.Vector4(-0.18, -0.14, 0.18, 1)],
      [new THREE.Vector4(-0.2, 0.22, 0.02, 1), new THREE.Vector4(-0.12, 0.24, 0.22, 1)],
    ],
    [
      [new THREE.Vector4(0.2, -0.2, 0.02, 1), new THREE.Vector4(0.28, -0.14, 0.18, 1)],
      [new THREE.Vector4(0.18, 0.2, 0.08, 1), new THREE.Vector4(0.3, 0.24, 0.26, 1)],
    ],
  ]
  const volumePoints = [
    [0, 0, 0],
    [0.5, 0.5, 0.5],
    [1, 1, 1],
    [0, 1, 0],
    [1, 0, 1],
  ].map(([u, v, w]) => {
    const point = new THREE.Vector3()
    NURBSUtils.calcVolumePoint(1, 1, 1, volumeKnots, volumeKnots, volumeKnots, volumeControls, u, v, w, point)
    return point.add(new THREE.Vector3(0.78, 0, 0))
  })
  const volumeGeometry = new THREE.BufferGeometry().setFromPoints(volumePoints)
  const volumeMaterial = new THREE.PointsMaterial({ color: 0x6688ff, size: 8, sizeAttenuation: false })
  const volumePointsObject = new THREE.Points(volumeGeometry, volumeMaterial)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  scene.add(curveMesh, surfaceMesh, volumePointsObject)

  const camera = new THREE.OrthographicCamera(-1.4, 1.4, 0.9, -0.9, 0.01, 10)
  camera.position.set(0, 0, 4)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  try {
    const width = 128
    const height = 80
    const rgba = renderRgba(scene, camera, { width, height })
    assert.equal(curveGeometry.getAttribute('position')?.count > 0, true)
    assert.equal(surfaceGeometry.getAttribute('normal')?.count > 0, true)
    assert.equal(volumeGeometry.getAttribute('position')?.count, 5)
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => r > 180 && g < 120 && b < 120) > 40,
      'NURBSUtils curve samples should render red tube pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => g > 120 && g > r + 20 && g > b + 20) > 70,
      'NURBSUtils surface samples should render green mesh pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => b > 170 && r < 140 && g < 170) > 40,
      'NURBSUtils volume samples should render blue point pixels',
    )
  } finally {
    curveGeometry.dispose()
    curveMaterial.dispose()
    surfaceGeometry.dispose()
    surfaceMaterial.dispose()
    volumeGeometry.dispose()
    volumeMaterial.dispose()
  }
})

test('examples geometry modifiers and BufferGeometryUtils render CPU-transformed geometry paths', () => {
  const edgeSourceGeometry = new THREE.BoxGeometry(0.48, 0.48, 0.48)
  const edgeGeometry = new EdgeSplitModifier().modify(edgeSourceGeometry, Math.PI / 6)
  const edgeMaterial = new THREE.MeshBasicMaterial({ color: 0xff4466, side: THREE.DoubleSide })
  const edgeMesh = new THREE.Mesh(edgeGeometry, edgeMaterial)
  edgeMesh.position.x = -1.25
  edgeMesh.rotation.set(0.45, 0.45, 0.1)

  const simplifySourceGeometry = new THREE.IcosahedronGeometry(0.48, 2)
  const simplifyGeometry = new SimplifyModifier().modify(simplifySourceGeometry, 24)
  simplifyGeometry.computeVertexNormals()
  const simplifyMaterial = new THREE.MeshBasicMaterial({ color: 0x55ff66, side: THREE.DoubleSide })
  const simplifyMesh = new THREE.Mesh(simplifyGeometry, simplifyMaterial)
  simplifyMesh.position.x = -0.42
  simplifyMesh.rotation.set(0.2, -0.5, 0.1)

  const tessellateSourceGeometry = new THREE.PlaneGeometry(0.72, 0.72, 1, 1).toNonIndexed()
  const tessellateGeometry = new TessellateModifier(0.28, 3).modify(tessellateSourceGeometry)
  tessellateGeometry.computeVertexNormals()
  const tessellateMaterial = new THREE.MeshBasicMaterial({ color: 0x6688ff, side: THREE.DoubleSide })
  const tessellateMesh = new THREE.Mesh(tessellateGeometry, tessellateMaterial)
  tessellateMesh.position.x = 0.42
  tessellateMesh.rotation.z = 0.2

  const mergeLeftGeometry = new THREE.BoxGeometry(0.2, 0.2, 0.2)
  mergeLeftGeometry.translate(-0.16, 0, 0)
  const mergeRightGeometry = new THREE.BoxGeometry(0.2, 0.2, 0.2)
  mergeRightGeometry.translate(0.16, 0, 0)
  const mergedGeometry = BufferGeometryUtils.mergeGeometries([mergeLeftGeometry, mergeRightGeometry], true)
  assert.ok(mergedGeometry, 'BufferGeometryUtils.mergeGeometries should produce merged geometry')
  const mergedMaterial = new THREE.MeshBasicMaterial({ color: 0xffdd44, side: THREE.DoubleSide })
  const mergedMesh = new THREE.Mesh(mergedGeometry, mergedMaterial)
  mergedMesh.position.x = 1.25
  mergedMesh.rotation.set(0.3, -0.35, 0.1)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  scene.add(edgeMesh, simplifyMesh, tessellateMesh, mergedMesh)

  const camera = new THREE.OrthographicCamera(-1.8, 1.8, 0.9, -0.9, 0.01, 10)
  camera.position.set(0, 0, 4)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  try {
    const width = 128
    const height = 72
    const rgba = renderRgba(scene, camera, { width, height })
    assert.ok(
      edgeGeometry.getAttribute('position').count > edgeSourceGeometry.getAttribute('position').count,
      'EdgeSplitModifier should split vertices around sharp edges',
    )
    assert.ok(
      simplifyGeometry.getAttribute('position').count < simplifySourceGeometry.getAttribute('position').count,
      'SimplifyModifier should reduce source vertices',
    )
    assert.ok(
      tessellateGeometry.getAttribute('position').count > tessellateSourceGeometry.getAttribute('position').count,
      'TessellateModifier should add vertices for long edges',
    )
    assert.equal(mergedGeometry.isBufferGeometry, true, 'mergeGeometries should return BufferGeometry')
    assert.ok(mergedGeometry.getAttribute('position').count > mergeLeftGeometry.getAttribute('position').count)
    assert.ok(mergedGeometry.groups.length >= 2, 'mergeGeometries(useGroups=true) should keep source geometry groups')
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => r > 160 && g < 120 && b < 140) > 120,
      'EdgeSplitModifier output should render red pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => g > 170 && g > r + 40 && g > b + 40) > 120,
      'SimplifyModifier output should render green pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => b > 180 && r < 150 && g < 180) > 120,
      'TessellateModifier output should render blue pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => r > 150 && g > 120 && b < 180) > 40,
      'BufferGeometryUtils.mergeGeometries output should render yellow pixels',
    )
  } finally {
    edgeSourceGeometry.dispose()
    edgeGeometry.dispose()
    edgeMaterial.dispose()
    simplifySourceGeometry.dispose()
    simplifyGeometry.dispose()
    simplifyMaterial.dispose()
    tessellateSourceGeometry.dispose()
    tessellateGeometry.dispose()
    tessellateMaterial.dispose()
    mergeLeftGeometry.dispose()
    mergeRightGeometry.dispose()
    mergedGeometry.dispose()
    mergedMaterial.dispose()
  }
})

test('examples GeometryCompressionUtils packed attributes fail clearly', () => {
  const camera = makeCamera()
  const cases = [
    ['positions', (geometry) => {
      GeometryCompressionUtils.compressPositions(geometry)
      return new THREE.MeshBasicMaterial({ color: 0xff0000, side: THREE.DoubleSide })
    }, /geometry\.attributes\.position uses packed GeometryCompressionUtils data.*Decode packed position, normal, or UV attributes/i],
    ['normals', (geometry) => {
      GeometryCompressionUtils.compressNormals(geometry, 'DEFAULT')
      return new THREE.MeshLambertMaterial({ color: 0xff0000, side: THREE.DoubleSide })
    }, /geometry\.attributes\.normal uses packed GeometryCompressionUtils data \(DEFAULT\).*Decode packed position, normal, or UV attributes/i],
    ['uvs', (geometry) => {
      GeometryCompressionUtils.compressUvs(geometry)
      return new THREE.MeshBasicMaterial({ map: solidTexture(255, 0, 0), side: THREE.DoubleSide })
    }, /geometry\.attributes\.uv uses packed GeometryCompressionUtils data.*Decode packed position, normal, or UV attributes/i],
  ]

  for (const [name, configure, pattern] of cases) {
    const geometry = new THREE.PlaneGeometry(1, 1)
    const material = configure(geometry)
    const scene = new THREE.Scene()
    scene.add(new THREE.Mesh(geometry, material))
    scene.add(new THREE.AmbientLight(0xffffff, 1))

    try {
      assert.throws(
        () => renderRgba(scene, camera, { width: 16, height: 16 }),
        pattern,
        `${name} should fail with packed-attribute decode guidance`,
      )
    } finally {
      geometry.dispose()
      material.map?.dispose?.()
      material.dispose()
    }
  }
})

test('examples ConvexObjectBreaker cuts debris into renderable mesh geometry', () => {
  const sourceGeometry = new THREE.BoxGeometry(0.8, 0.5, 0.4).toNonIndexed()
  const sourceMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide })
  const source = new THREE.Mesh(sourceGeometry, sourceMaterial)
  const breaker = new ConvexObjectBreaker(0.05)
  breaker.prepareBreakableObject(source, 2, new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), true)

  const output = {}
  const pieces = breaker.cutByPlane(source, new THREE.Plane(new THREE.Vector3(1, 0, 0), 0), output)
  const leftPiece = output.object1
  const rightPiece = output.object2
  const redMaterial = new THREE.MeshBasicMaterial({ color: 0xff4455, side: THREE.DoubleSide })
  const cyanMaterial = new THREE.MeshBasicMaterial({ color: 0x44e8ff, side: THREE.DoubleSide })

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)

  try {
    assert.equal(pieces, 2)
    assert.ok(leftPiece?.isMesh)
    assert.ok(rightPiece?.isMesh)
    assert.equal(leftPiece.userData.mass, 1)
    assert.equal(rightPiece.userData.mass, 1)
    assert.equal(leftPiece.userData.breakable, true)
    assert.equal(rightPiece.userData.breakable, true)
    assert.ok(leftPiece.geometry.getAttribute('position').count > 12)
    assert.ok(rightPiece.geometry.getAttribute('position').count > 12)

    leftPiece.material = redMaterial
    rightPiece.material = cyanMaterial
    scene.add(leftPiece, rightPiece)

    const camera = new THREE.OrthographicCamera(-1, 1, 0.7, -0.7, 0.01, 10)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    camera.updateMatrixWorld(true)

    const width = 128
    const height = 72
    const rgba = renderRgba(scene, camera, { width, height })

    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => r > 150 && g < 140 && b < 140) > 150,
      'ConvexObjectBreaker negative-side debris should render red pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => g > 120 && b > 120 && b > r + 30 && g > r + 30) > 150,
      'ConvexObjectBreaker positive-side debris should render cyan pixels',
    )
  } finally {
    sourceGeometry.dispose()
    sourceMaterial.dispose()
    leftPiece?.geometry?.dispose()
    rightPiece?.geometry?.dispose()
    redMaterial.dispose()
    cyanMaterial.dispose()
  }
})
