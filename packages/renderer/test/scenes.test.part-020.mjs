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
import { Renderer, test } from './scenes.test.part-001.mjs'
import { countRegionPixels, makeCamera, meanRegion, renderRgba } from './scenes.test.part-002.mjs'
test('examples collision math utilities produce renderable helper geometry', () => {
  const sourceGeometry = new THREE.BoxGeometry(0.6, 0.42, 0.36).toNonIndexed()
  const sourceMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff })
  const sourceMesh = new THREE.Mesh(sourceGeometry, sourceMaterial)
  sourceMesh.rotation.set(0.15, 0.3, -0.1)
  sourceMesh.updateMatrixWorld(true)

  const octree = new Octree().fromGraphNode(sourceMesh)
  const octreeHelper = new OctreeHelper(octree, 0xffff00)
  octreeHelper.position.x = -1.05

  const capsule = new Capsule(
    new THREE.Vector3(-0.15, -0.65, 0),
    new THREE.Vector3(0.15, 0.65, 0),
    0.2,
  )
  const capsuleGeometry = new THREE.BufferGeometry().setAttribute(
    'position',
    new THREE.Float32BufferAttribute([
      capsule.start.x, capsule.start.y, capsule.start.z,
      capsule.end.x, capsule.end.y, capsule.end.z,
    ], 3),
  )
  const capsuleMaterial = new THREE.LineBasicMaterial({ color: 0xff44ff, linewidth: 6 })
  const capsuleLine = new THREE.LineSegments(capsuleGeometry, capsuleMaterial)
  capsuleLine.position.x = 1.05

  const obb = new OBB().fromBox3(octree.box)
  const obbSize = obb.getSize(new THREE.Vector3())
  const obbGeometry = new THREE.BoxGeometry(obbSize.x, obbSize.y, obbSize.z)
  const obbMaterial = new THREE.MeshBasicMaterial({ color: 0x0044ff, side: THREE.DoubleSide })
  const obbMesh = new THREE.Mesh(obbGeometry, obbMaterial)
  obbMesh.position.copy(obb.center)
  obbMesh.position.y = -0.72

  const hull = new ConvexHull().setFromObject(sourceMesh)
  const hullPositions = []
  for (const face of hull.faces) {
    let edge = face.edge
    do {
      const a = edge.tail().point
      const b = edge.head().point
      hullPositions.push(a.x, a.y, a.z, b.x, b.y, b.z)
      edge = edge.next
    } while (edge !== face.edge)
  }
  const hullGeometry = new THREE.BufferGeometry().setAttribute(
    'position',
    new THREE.Float32BufferAttribute(hullPositions, 3),
  )
  const hullMaterial = new THREE.LineBasicMaterial({ color: 0x44ffcc, linewidth: 5 })
  const hullLine = new THREE.LineSegments(hullGeometry, hullMaterial)
  hullLine.position.y = 0.72

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  scene.add(octreeHelper, capsuleLine, obbMesh, hullLine)

  const camera = new THREE.OrthographicCamera(-1.8, 1.8, 1.2, -1.2, 0.01, 10)
  camera.position.set(0, 0, 4)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  try {
    const width = 128
    const height = 88
    const rgba = renderRgba(scene, camera, { width, height })
    assert.ok(octree.subTrees.length > 0, 'Octree.fromGraphNode should build visible helper boxes')
    assert.ok(octree.box.intersectsBox(new THREE.Box3().setFromObject(sourceMesh)))
    assert.ok(capsule.intersectsBox(octree.box), 'Capsule should intersect the Octree bounds')
    assert.ok(obb.containsPoint(octree.box.getCenter(new THREE.Vector3())), 'OBB should contain the Octree center')
    assert.ok(hull.faces.length > 0, 'ConvexHull.setFromObject should build hull faces')
    assert.ok(hull.containsPoint(sourceMesh.position), 'ConvexHull should contain the source mesh center')
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => r > 150 && g > 130 && b < 80) > 20,
      'Octree helper boxes should render yellow line pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 44, width, height, (r, g, b) => b > 120 && r < 80 && g < 120) > 120,
      'OBB-derived mesh should render blue pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => g > 180 && b > 160 && g > r + 40 && b > r + 20) > 20,
      'ConvexHull-derived line segments should render cyan pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => r > 150 && b > 120 && g < 120) > 10,
      'Capsule segment should render magenta line pixels',
    )
  } finally {
    sourceGeometry.dispose()
    sourceMaterial.dispose()
    octreeHelper.dispose()
    capsuleGeometry.dispose()
    capsuleMaterial.dispose()
    obbGeometry.dispose()
    obbMaterial.dispose()
    hullGeometry.dispose()
    hullMaterial.dispose()
  }
})

test('core Three helpers render supported line and basic material geometry', () => {
  const makeCameraForHelper = () => {
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 10)
    camera.position.set(1.4, 1.2, 2.4)
    camera.lookAt(0, 0, 0)
    return camera
  }
  const helperCases = [
    ['AxesHelper', () => new THREE.AxesHelper(0.8)],
    ['GridHelper', () => new THREE.GridHelper(1, 4)],
    ['PolarGridHelper', () => new THREE.PolarGridHelper(0.55, 6, 2, 12)],
    ['BoxHelper', () => {
      const mesh = new THREE.Mesh(
        new THREE.BoxGeometry(0.55, 0.45, 0.35),
        new THREE.MeshBasicMaterial({ color: 0xffffff }),
      )
      mesh.updateMatrixWorld(true)
      return new THREE.BoxHelper(mesh, 0xff8800)
    }],
    ['Box3Helper', () => new THREE.Box3Helper(
      new THREE.Box3(
        new THREE.Vector3(-0.35, -0.25, -0.2),
        new THREE.Vector3(0.35, 0.25, 0.2),
      ),
      0x00ffaa,
    )],
    ['PlaneHelper', () => new THREE.PlaneHelper(
      new THREE.Plane(new THREE.Vector3(0, 0, 1), 0),
      0.8,
      0x4488ff,
    )],
    ['ArrowHelper', () => new THREE.ArrowHelper(
      new THREE.Vector3(1, 0.25, 0).normalize(),
      new THREE.Vector3(-0.3, -0.1, 0),
      0.8,
      0xffff00,
    )],
    ['CameraHelper', () => {
      const sourceCamera = new THREE.PerspectiveCamera(50, 1, 0.1, 1.2)
      sourceCamera.position.set(0, 0, 0.55)
      sourceCamera.lookAt(0, 0, -0.4)
      sourceCamera.updateMatrixWorld(true)
      return new THREE.CameraHelper(sourceCamera)
    }],
    ['SkeletonHelper', () => {
      const root = new THREE.Bone()
      root.position.set(-0.35, -0.25, 0)
      const elbow = new THREE.Bone()
      elbow.position.set(0.35, 0.5, 0)
      const tip = new THREE.Bone()
      tip.position.set(0.28, -0.15, 0.2)
      root.add(elbow)
      elbow.add(tip)
      root.updateMatrixWorld(true)
      return new THREE.SkeletonHelper(root)
    }],
    ['DirectionalLightHelper', () => {
      const light = new THREE.DirectionalLight(0xffffff, 1)
      light.position.set(0, 0, 0.35)
      light.target.position.set(0, 0, -0.4)
      light.updateMatrixWorld(true)
      light.target.updateMatrixWorld(true)
      return new THREE.DirectionalLightHelper(light, 0.55)
    }],
    ['PointLightHelper', () => {
      const light = new THREE.PointLight(0xffffff, 1)
      return new THREE.PointLightHelper(light, 0.45)
    }],
    ['SpotLightHelper', () => {
      const light = new THREE.SpotLight(0xffffff, 1)
      light.position.set(0, 0, 0.5)
      light.target.position.set(0, 0, -0.5)
      light.updateMatrixWorld(true)
      light.target.updateMatrixWorld(true)
      return new THREE.SpotLightHelper(light, 0x00ffff)
    }],
    ['HemisphereLightHelper', () => {
      const light = new THREE.HemisphereLight(0xffffaa, 0x3333ff, 1)
      return new THREE.HemisphereLightHelper(light, 0.45)
    }],
  ]
  assert.deepEqual(
    helperCases.map(([label]) => label).sort(),
    Object.keys(THREE).filter((name) => name.endsWith('Helper')).sort(),
    'core helper coverage should track installed Three.js helper exports',
  )

  for (const [label, makeHelper] of helperCases) {
    const helper = makeHelper()
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x000000)
    scene.add(helper)
    helper.update?.()
    helper.updateMatrixWorld?.(true)

    try {
      const rgba = renderRgba(scene, makeCameraForHelper(), { width: 64, height: 64 })
      assert.ok(
        nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.0005,
        `${label} should render visible supported helper geometry`,
      )
    } finally {
      helper.dispose?.()
    }
  }
})

test('Reflector and Refractor prepasses use Renderer target state and restore flags', () => {
  for (const [label, Helper] of [
    ['Reflector', Reflector],
    ['Refractor', Refractor],
  ]) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 1)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(4, 4),
      new THREE.MeshBasicMaterial({ color: 0xff0000 }),
    ))

    const helper = new Helper(new THREE.PlaneGeometry(2, 2), {
      textureWidth: 16,
      textureHeight: 16,
    })
    scene.add(helper)

    const camera = makeCamera()
    camera.updateMatrixWorld()
    helper.updateMatrixWorld()

    const renderer = new Renderer()
    renderer.setSize(32, 32)
    renderer.setViewport(3, 4, 12, 14)
    const previousTarget = { texture: {} }
    renderer.setRenderTarget(previousTarget)
    renderer.xr.enabled = true
    renderer.shadowMap.autoUpdate = true
    renderer.autoClear = false

    helper.onBeforeRender(renderer, scene, camera)

    const helperTarget = helper.getRenderTarget()
    const helperData = helperTarget.data ?? helperTarget.texture?.image?.data ?? helperTarget.texture?.source?.data?.data
    assert.ok(helperData instanceof Uint8Array, `${label} should render its helper-owned target`)
    assert.equal(helperTarget.width, 16)
    assert.equal(helperTarget.height, 16)
    assert.strictEqual(renderer.getRenderTarget(), previousTarget)
    assert.deepEqual(renderer.getViewport(), { x: 3, y: 4, width: 12, height: 14 })
    assert.equal(renderer.xr.enabled, true)
    assert.equal(renderer.shadowMap.autoUpdate, true)
    assert.equal(renderer.autoClear, false)
    assert.equal(helper.visible, true)
    helper.dispose()
  }
})

test('ShadowMesh renders projected helper geometry with built-in material state', () => {
  const sourceGeometry = new THREE.BoxGeometry(0.7, 0.7, 0.7)
  const sourceMaterial = new THREE.MeshBasicMaterial({ color: 0xff5533 })
  const source = new THREE.Mesh(sourceGeometry, sourceMaterial)
  source.position.set(0, 0.45, 0)
  source.rotation.set(0.2, 0.4, 0.1)
  source.updateMatrixWorld(true)

  const shadow = new ShadowMesh(source)
  shadow.update(
    new THREE.Plane(new THREE.Vector3(0, 1, 0), 0),
    new THREE.Vector4(-2, 4, 3, 1),
  )

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0xffffff)
  scene.add(shadow)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 10)
  camera.position.set(0, 2.4, 3.2)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  try {
    const rgba = renderRgba(scene, camera, { width: 64, height: 64 })
    const center = meanRegion(rgba, 64, 64, 20, 20, 44, 44)
    assert.equal(shadow.isShadowMesh, true)
    assert.equal(shadow.frustumCulled, false)
    assert.equal(shadow.matrixAutoUpdate, false)
    assert.ok(
      nonBackgroundRatio(rgba, [255, 255, 255], 3) > 0.03,
      'ShadowMesh should project visible helper geometry onto the plane',
    )
    assert.ok(
      center.r < 235 && center.g < 235 && center.b < 235,
      `ShadowMesh transparent material should darken the projected region (${center.r}, ${center.g}, ${center.b})`,
    )
  } finally {
    shadow.material.dispose()
    sourceGeometry.dispose()
    sourceMaterial.dispose()
  }
})
