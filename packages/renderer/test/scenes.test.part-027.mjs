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
test('examples PCDLoader and VTKLoader parse renderable point and mesh geometry paths', () => {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  const pcdPointCloud = new PCDLoader().parse(new TextEncoder().encode([
    '# .PCD v0.7 - Point Cloud Data file format',
    'VERSION 0.7',
    'FIELDS x y z rgb',
    'SIZE 4 4 4 4',
    'TYPE F F F U',
    'COUNT 1 1 1 1',
    'WIDTH 3',
    'HEIGHT 1',
    'VIEWPOINT 0 0 0 1 0 0 0',
    'POINTS 3',
    'DATA ascii',
    '-0.8 0 0 16711680',
    '0 0.8 0 65280',
    '0.8 0 0 255',
  ].join('\n')).buffer)
  pcdPointCloud.material.size = 5
  pcdPointCloud.material.sizeAttenuation = false
  const pcdScene = new THREE.Scene()
  pcdScene.background = new THREE.Color(0x000000)
  pcdScene.add(pcdPointCloud)

  const vtkGeometry = new VTKLoader().parse(new TextEncoder().encode([
    '# vtk DataFile Version 3.0',
    'tri',
    'ASCII',
    'DATASET POLYDATA',
    'POINTS 3 float',
    '-0.6 -0.4 0',
    '0.6 -0.4 0',
    '0 0.6 0',
    'POLYGONS 1 4',
    '3 0 1 2',
    'POINT_DATA 3',
    'COLOR_SCALARS color 3',
    '1 0 0',
    '0 1 0',
    '0 0 1',
    'NORMALS normals float',
    '0 0 1',
    '0 0 1',
    '0 0 1',
  ].join('\n').padEnd(260, '\n')).buffer)
  const vtkMaterial = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide, vertexColors: true })
  const vtkMesh = new THREE.Mesh(vtkGeometry, vtkMaterial)
  const vtkScene = new THREE.Scene()
  vtkScene.background = new THREE.Color(0x000000)
  vtkScene.add(vtkMesh)

  try {
    assert.equal(pcdPointCloud.isPoints, true)
    assert.equal(pcdPointCloud.geometry.getAttribute('position').count, 3)
    assert.equal(pcdPointCloud.geometry.getAttribute('color').count, 3)
    assert.equal(pcdPointCloud.material.vertexColors, true)
    assert.ok(
      nonBackgroundRatio(renderRgba(pcdScene, camera, { width: 64, height: 64 }), [0, 0, 0], 3) > 0.01,
      'PCDLoader point cloud output should render visible colored points',
    )

    assert.equal(vtkGeometry.isBufferGeometry, true)
    assert.equal(vtkGeometry.getAttribute('position').count, 3)
    assert.equal(vtkGeometry.index.count, 3)
    assert.equal(vtkGeometry.getAttribute('color').count, 3)
    assert.equal(vtkGeometry.getAttribute('normal').count, 3)
    assert.ok(
      nonBackgroundRatio(renderRgba(vtkScene, camera, { width: 64, height: 64 }), [0, 0, 0], 3) > 0.08,
      'VTKLoader mesh output should render visible vertex-colored geometry',
    )
  } finally {
    pcdPointCloud.geometry.dispose()
    pcdPointCloud.material.dispose()
    vtkGeometry.dispose()
    vtkMaterial.dispose()
  }
})

test('examples ParametricFunctions render generated geometry paths', () => {
  const entries = [
    {
      label: 'ParametricFunctions.klein',
      geometry: new ParametricGeometry(ParametricFunctions.klein, 10, 10),
      color: 0xff44ff,
      x: -0.85,
      scale: 0.07,
      isColor: (r, g, b) => r > 120 && b > 120 && g < 120,
    },
    {
      label: 'THREE.TorusKnotGeometry',
      geometry: new THREE.TorusKnotGeometry(0.28, 0.06, 28, 8),
      color: 0x55ff66,
      x: 0,
      scale: 1,
      isColor: (r, g, b) => g > 130 && g > r + 40 && g > b + 40,
    },
    {
      label: 'THREE.SphereGeometry',
      geometry: new THREE.SphereGeometry(0.32, 12, 8),
      color: 0x4488ff,
      x: 0.85,
      scale: 1,
      isColor: (r, g, b) => b > 130 && b > r + 50 && b > g + 20,
    },
  ]

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  const materials = []
  const meshes = []
  for (const entry of entries) {
    const material = new THREE.MeshBasicMaterial({ color: entry.color, side: THREE.DoubleSide })
    const mesh = new THREE.Mesh(entry.geometry, material)
    mesh.position.x = entry.x
    mesh.scale.setScalar(entry.scale)
    scene.add(mesh)
    materials.push(material)
    meshes.push(mesh)
  }

  const camera = new THREE.OrthographicCamera(-1.8, 1.8, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 4)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  try {
    const width = 128
    const height = 72
    const rgba = renderRgba(scene, camera, { width, height })
    for (const entry of entries) {
      assert.equal(entry.geometry.isBufferGeometry, true, `${entry.label} should produce BufferGeometry`)
      assert.ok(entry.geometry.getAttribute('position')?.count > 0, `${entry.label} should generate vertices`)
      assert.ok(entry.geometry.getAttribute('normal'), `${entry.label} should generate normals`)
      assert.ok(
        countRegionPixels(rgba, width, height, 0, 0, width, height, entry.isColor) > 30,
        `${entry.label} should render visible material color`,
      )
    }
  } finally {
    for (const mesh of meshes) scene.remove(mesh)
    for (const entry of entries) entry.geometry.dispose()
    for (const material of materials) material.dispose()
  }
})

test('examples CurveExtras and NURBS helpers render generated geometry paths', () => {
  const curveGeometry = new THREE.TubeGeometry(new TrefoilKnot(0.18), 64, 0.025, 6, true)
  const curveMaterial = new THREE.MeshBasicMaterial({ color: 0xff55aa, side: THREE.DoubleSide })
  const curveMesh = new THREE.Mesh(curveGeometry, curveMaterial)
  curveMesh.position.x = -1.2

  const nurbsCurve = new NURBSCurve(3, [0, 0, 0, 0, 1, 1, 1, 1], [
    new THREE.Vector4(-0.42, -0.28, 0, 1),
    new THREE.Vector4(-0.12, 0.42, 0.18, 1),
    new THREE.Vector4(0.18, -0.42, -0.08, 1),
    new THREE.Vector4(0.42, 0.28, 0, 1),
  ])
  const nurbsCurveGeometry = new THREE.TubeGeometry(nurbsCurve, 32, 0.022, 6, false)
  const nurbsCurveMaterial = new THREE.MeshBasicMaterial({ color: 0x55ffff, side: THREE.DoubleSide })
  const nurbsCurveMesh = new THREE.Mesh(nurbsCurveGeometry, nurbsCurveMaterial)
  nurbsCurveMesh.position.x = -0.35

  const surfaceKnots = [0, 0, 0, 1, 1, 1]
  const nurbsSurface = new NURBSSurface(2, 2, surfaceKnots, surfaceKnots, [
    [
      new THREE.Vector4(-0.34, -0.34, 0.02, 1),
      new THREE.Vector4(-0.34, 0, 0.22, 1),
      new THREE.Vector4(-0.34, 0.34, 0.02, 1),
    ],
    [
      new THREE.Vector4(0, -0.34, 0.18, 1),
      new THREE.Vector4(0, 0, 0.34, 1),
      new THREE.Vector4(0, 0.34, 0.18, 1),
    ],
    [
      new THREE.Vector4(0.34, -0.34, 0.02, 1),
      new THREE.Vector4(0.34, 0, 0.22, 1),
      new THREE.Vector4(0.34, 0.34, 0.02, 1),
    ],
  ])
  const surfaceGeometry = new ParametricGeometry((u, v, target) => {
    nurbsSurface.getPoint(u, v, target)
  }, 10, 10)
  const surfaceMaterial = new THREE.MeshBasicMaterial({ color: 0x66ff77, side: THREE.DoubleSide })
  const surfaceMesh = new THREE.Mesh(surfaceGeometry, surfaceMaterial)
  surfaceMesh.position.x = 0.45

  const volumeKnots = [0, 0, 1, 1]
  const nurbsVolume = new NURBSVolume(1, 1, 1, volumeKnots, volumeKnots, volumeKnots, [
    [
      [new THREE.Vector4(-0.28, -0.24, -0.08, 1), new THREE.Vector4(-0.18, -0.18, 0.16, 1)],
      [new THREE.Vector4(-0.24, 0.24, 0.04, 1), new THREE.Vector4(-0.12, 0.28, 0.22, 1)],
    ],
    [
      [new THREE.Vector4(0.22, -0.22, -0.02, 1), new THREE.Vector4(0.28, -0.18, 0.18, 1)],
      [new THREE.Vector4(0.18, 0.22, 0.08, 1), new THREE.Vector4(0.3, 0.26, 0.26, 1)],
    ],
  ])
  const volumeSamples = [
    [0, 0, 0],
    [1, 1, 1],
    [0, 1, 0],
    [1, 0, 1],
    [0, 0.5, 1],
    [1, 0.5, 0],
  ].map(([u, v, w]) => {
    const point = new THREE.Vector3()
    nurbsVolume.getPoint(u, v, w, point)
    return point
  })
  const volumeGeometry = new THREE.BufferGeometry().setFromPoints(volumeSamples)
  const volumeMaterial = new THREE.LineBasicMaterial({ color: 0x6688ff, linewidth: 4 })
  const volumeLines = new THREE.LineSegments(volumeGeometry, volumeMaterial)
  volumeLines.position.x = 1.25

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  scene.add(curveMesh, nurbsCurveMesh, surfaceMesh, volumeLines)

  const camera = new THREE.OrthographicCamera(-1.8, 1.8, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 4)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  try {
    const width = 128
    const height = 72
    const rgba = renderRgba(scene, camera, { width, height })
    assert.equal(curveGeometry.isBufferGeometry, true)
    assert.ok(curveGeometry.getAttribute('position')?.count > 0, 'CurveExtras TubeGeometry should generate vertices')
    assert.ok(nurbsCurveGeometry.getAttribute('position')?.count > 0, 'NURBSCurve TubeGeometry should generate vertices')
    assert.ok(surfaceGeometry.getAttribute('normal'), 'NURBSSurface ParametricGeometry should generate normals')
    assert.equal(volumeGeometry.getAttribute('position')?.count, 6)
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => r > 130 && b > 100 && g < 120) > 60,
      'CurveExtras TrefoilKnot tube should render pink pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => g > 180 && b > 180 && r < 190) > 25,
      'NURBSCurve tube should render cyan pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => g > 130 && g > r + 40 && g > b + 20) > 60,
      'NURBSSurface mesh should render green pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => b > 150 && r < 130 && g < 170) > 20,
      'NURBSVolume sampled line segments should render blue pixels',
    )
  } finally {
    curveGeometry.dispose()
    curveMaterial.dispose()
    nurbsCurveGeometry.dispose()
    nurbsCurveMaterial.dispose()
    surfaceGeometry.dispose()
    surfaceMaterial.dispose()
    volumeGeometry.dispose()
    volumeMaterial.dispose()
  }
})
