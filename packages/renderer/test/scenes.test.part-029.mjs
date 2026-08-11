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
test('examples LDrawUtils merges mesh and line objects into renderable geometry', () => {
  const meshMaterial = new THREE.MeshBasicMaterial({ color: 0xff4455, side: THREE.DoubleSide })
  const lineMaterial = new THREE.LineBasicMaterial({ color: 0x44ffff, linewidth: 8 })
  const root = new THREE.Group()
  const meshGeometries = []
  for (const x of [-0.35, 0.15]) {
    const geometry = new THREE.BoxGeometry(0.25, 0.25, 0.25).toNonIndexed()
    meshGeometries.push(geometry)
    const mesh = new THREE.Mesh(geometry, meshMaterial)
    mesh.position.x = x
    root.add(mesh)
  }
  const lineGeometry = new THREE.BufferGeometry()
  lineGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.6, -0.35, 0,
    0.45, 0.35, 0,
  ]), 3))
  root.add(new THREE.LineSegments(lineGeometry, lineMaterial))

  const merged = LDrawUtils.mergeObject(root)
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  scene.add(merged)

  const camera = new THREE.OrthographicCamera(-1, 1, 0.7, -0.7, 0.01, 10)
  camera.position.set(0, 0, 4)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  try {
    const width = 96
    const height = 72
    const rgba = renderRgba(scene, camera, { width, height })
    const mergedMesh = merged.children.find((child) => child.isMesh)
    const mergedLine = merged.children.find((child) => child.isLineSegments)

    assert.equal(merged.userData.constructionStep, 0)
    assert.equal(merged.userData.numConstructionSteps, 1)
    assert.equal(merged.children.length, 2, 'LDrawUtils.mergeObject should merge by mesh and line material')
    assert.equal(mergedMesh?.geometry.getAttribute('position').count, 72)
    assert.equal(mergedLine?.geometry.getAttribute('position').count, 2)
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => r > 150 && g < 120 && b < 140) > 250,
      'LDrawUtils merged mesh geometry should render red pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => g > 180 && b > 180 && g > r + 40) > 300,
      'LDrawUtils merged line geometry should render cyan pixels',
    )
  } finally {
    for (const geometry of meshGeometries) geometry.dispose()
    lineGeometry.dispose()
    for (const child of merged.children) child.geometry?.dispose()
    meshMaterial.dispose()
    lineMaterial.dispose()
  }
})

test('examples SceneUtils and MeshSurfaceSampler produce renderable scene data', () => {
  const splitGeometry = new THREE.PlaneGeometry(0.7, 0.7, 1, 1)
  splitGeometry.clearGroups()
  splitGeometry.addGroup(0, 3, 0)
  splitGeometry.addGroup(3, 3, 1)
  const redMaterial = new THREE.MeshBasicMaterial({ color: 0xff4444, side: THREE.DoubleSide })
  const greenMaterial = new THREE.MeshBasicMaterial({ color: 0x44ff66, side: THREE.DoubleSide })
  const splitSource = new THREE.Mesh(splitGeometry, [redMaterial, greenMaterial])
  const splitGroup = SceneUtils.createMeshesFromMultiMaterialMesh(splitSource)
  splitGroup.position.x = -1.15

  const instancedGeometry = new THREE.BoxGeometry(0.18, 0.18, 0.18)
  const instancedMaterial = new THREE.MeshBasicMaterial({ color: 0x4488ff })
  const instancedMesh = new THREE.InstancedMesh(instancedGeometry, instancedMaterial, 3)
  const instanceMatrix = new THREE.Matrix4()
  for (const [index, position] of [
    [0, new THREE.Vector3(-0.22, -0.08, 0)],
    [1, new THREE.Vector3(0, 0.16, 0)],
    [2, new THREE.Vector3(0.22, -0.08, 0)],
  ]) {
    instanceMatrix.makeTranslation(position.x, position.y, position.z)
    instancedMesh.setMatrixAt(index, instanceMatrix)
  }
  const instancedGroup = SceneUtils.createMeshesFromInstancedMesh(instancedMesh)

  const sampleSourceGeometry = new THREE.PlaneGeometry(0.72, 0.72, 1, 1)
  const sampleSource = new THREE.Mesh(sampleSourceGeometry, new THREE.MeshBasicMaterial())
  const randomValues = [
    0.05, 0.1, 0.25, 0.2, 0.45, 0.3, 0.65, 0.4,
    0.85, 0.5, 0.15, 0.6, 0.35, 0.7, 0.55, 0.8,
    0.75, 0.9, 0.95, 0.12, 0.22, 0.32, 0.42, 0.52,
  ]
  let randomIndex = 0
  const sampler = new MeshSurfaceSampler(sampleSource)
    .setRandomGenerator(() => randomValues[randomIndex++ % randomValues.length])
    .build()
  const sampledPositions = []
  const sampledNormal = new THREE.Vector3()
  const sampledUv = new THREE.Vector2()
  for (let i = 0; i < 24; i += 1) {
    const sampledPosition = new THREE.Vector3()
    sampler.sample(sampledPosition, sampledNormal, undefined, sampledUv)
    sampledPositions.push(sampledPosition.x, sampledPosition.y, sampledPosition.z)
  }
  const pointsGeometry = new THREE.BufferGeometry()
  pointsGeometry.setAttribute('position', new THREE.Float32BufferAttribute(sampledPositions, 3))
  const pointsMaterial = new THREE.PointsMaterial({ color: 0xffdd44, size: 8, sizeAttenuation: false })
  const sampledPoints = new THREE.Points(pointsGeometry, pointsMaterial)
  sampledPoints.position.x = 1.15

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  scene.add(splitGroup, instancedGroup, sampledPoints)

  const camera = new THREE.OrthographicCamera(-1.8, 1.8, 0.9, -0.9, 0.01, 10)
  camera.position.set(0, 0, 4)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  try {
    const width = 128
    const height = 72
    const rgba = renderRgba(scene, camera, { width, height })

    assert.equal(splitGroup.children.length, 2, 'SceneUtils should split a two-group material mesh')
    assert.equal(instancedGroup.children.length, 3, 'SceneUtils should expand three instanced meshes')
    assert.equal(pointsGeometry.getAttribute('position').count, 24)
    assert.ok(sampledNormal.z > 0.9, 'MeshSurfaceSampler should sample source normals')
    assert.ok(sampledUv.x >= 0 && sampledUv.x <= 1 && sampledUv.y >= 0 && sampledUv.y <= 1)

    const maxSplitX = SceneUtils.reduceVertices(splitGroup, (max, vertex) => Math.max(max, vertex.x), -Infinity)
    assert.ok(maxSplitX > -0.9 && maxSplitX < -0.75, `reduceVertices should see transformed split geometry (${maxSplitX})`)
    assert.equal([...SceneUtils.traverseVisibleGenerator(splitGroup)].length, 3)
    assert.strictEqual([...SceneUtils.traverseAncestorsGenerator(splitGroup.children[0])][0], splitGroup)

    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => r > 150 && g < 120 && b < 120) > 40,
      'SceneUtils split red material should render visible pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => g > 120 && g > r + 20 && g > b + 20) > 30,
      'SceneUtils split green material should render visible pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => b > 150 && r < 140 && g < 170) > 35,
      'SceneUtils expanded instanced meshes should render blue pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => r > 150 && g > 130 && b < 170) > 35,
      'MeshSurfaceSampler points should render yellow pixels',
    )
  } finally {
    for (const child of splitGroup.children) child.geometry?.dispose()
    splitGeometry.dispose()
    redMaterial.dispose()
    greenMaterial.dispose()
    instancedGeometry.dispose()
    instancedMaterial.dispose()
    sampleSourceGeometry.dispose()
    sampleSource.material.dispose()
    pointsGeometry.dispose()
    pointsMaterial.dispose()
  }
})

test('examples SceneOptimizer batches compatible meshes into renderable BatchedMesh output', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  const colors = [0xff4455, 0x44ff66, 0x4488ff]
  const positions = [-0.55, 0, 0.55]
  for (let i = 0; i < colors.length; i += 1) {
    const geometry = new THREE.BoxGeometry(0.28, 0.28, 0.28)
    const material = new THREE.MeshBasicMaterial({ color: colors[i], side: THREE.DoubleSide })
    const mesh = new THREE.Mesh(geometry, material)
    mesh.name = `optimizer-piece-${i}`
    mesh.position.x = positions[i]
    scene.add(mesh)
  }

  const optimizedScene = new SceneOptimizer(scene).toBatchedMesh()
  const batchedMesh = optimizedScene.children.find((child) => child.isBatchedMesh)

  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 0.7, -0.7, 0.01, 10)
  camera.position.set(0, 0, 4)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  try {
    const width = 128
    const height = 72
    const rgba = renderRgba(optimizedScene, camera, { width, height })

    assert.strictEqual(optimizedScene, scene)
    assert.equal(optimizedScene.children.length, 1, 'SceneOptimizer should replace compatible meshes with one batch')
    assert.ok(batchedMesh?.isBatchedMesh, 'SceneOptimizer should create a BatchedMesh')
    assert.equal(batchedMesh.name, 'optimizer-piece-0_batch')
    assert.equal(batchedMesh._maxInstanceCount, 3)
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => r > 150 && g < 120 && b < 140) > 150,
      'SceneOptimizer BatchedMesh should render red instance pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => g > 150 && g > r + 30 && g > b + 20) > 150,
      'SceneOptimizer BatchedMesh should render green instance pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => b > 150 && r < 120 && g < 170) > 150,
      'SceneOptimizer BatchedMesh should render blue instance pixels',
    )
  } finally {
    for (const child of optimizedScene.children) {
      child.geometry?.dispose()
      if (Array.isArray(child.material)) {
        for (const material of child.material) material.dispose()
      } else {
        child.material?.dispose()
      }
    }
  }
})

test('examples Lut maps scalar values into renderable vertex colors', () => {
  const lut = new Lut('rainbow', 8).setMin(-1).setMax(1)
  const scalarValues = [-1, 0, 1]
  const centers = [-0.62, 0, 0.62]
  const positions = []
  const colors = []
  const halfWidth = 0.22
  const halfHeight = 0.42

  for (let i = 0; i < scalarValues.length; i += 1) {
    const x = centers[i]
    const corners = [
      [x - halfWidth, -halfHeight, 0],
      [x + halfWidth, -halfHeight, 0],
      [x + halfWidth, halfHeight, 0],
      [x - halfWidth, -halfHeight, 0],
      [x + halfWidth, halfHeight, 0],
      [x - halfWidth, halfHeight, 0],
    ]
    const color = lut.getColor(scalarValues[i])
    assert.ok(color?.isColor, 'Lut should produce THREE.Color values')
    for (const corner of corners) {
      positions.push(...corner)
      colors.push(color.r, color.g, color.b)
    }
  }

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3))
  const material = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide })
  const mesh = new THREE.Mesh(geometry, material)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  scene.add(mesh)

  const camera = new THREE.OrthographicCamera(-1, 1, 0.7, -0.7, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  try {
    const width = 128
    const height = 72
    const rgba = renderRgba(scene, camera, { width, height })

    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => b > 150 && r < 120 && g < 170) > 200,
      'Lut low scalar vertex colors should render blue pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => g > 120 && g > r + 20 && g > b + 20) > 200,
      'Lut midpoint vertex colors should render green pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => r > 150 && g < 140 && b < 140) > 200,
      'Lut high scalar vertex colors should render red pixels',
    )
  } finally {
    geometry.dispose()
    material.dispose()
  }
})
