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
import { getRenderer, makeCamera, meanRegion, renderRgba } from './scenes.test.part-002.mjs'
test('malformed BatchedMesh inputs fail clearly', () => {
  const camera = makeCamera()
  const makeBatchedScene = () => {
    const scene = new THREE.Scene()
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ color: 0xffffff }))
    mesh.isBatchedMesh = true
    scene.add(mesh)
    return { scene, mesh }
  }

  const { scene } = makeBatchedScene()
  assert.throws(
    () => getRenderer().render(scene, camera, { width: 32, height: 32, format: 'rgba' }),
    /THREE\.BatchedMesh instance table is not readable.*ordinary Mesh or InstancedMesh/i,
  )

  const instanceCases = [
    ['instance entry', [null], /THREE\.BatchedMesh\._instanceInfo\[0\] must be an object/i],
    ['active flag', [{ geometryIndex: 0, active: 'yes' }], /THREE\.BatchedMesh\._instanceInfo\[0\]\.active must be a boolean/i],
    ['visible flag', [{ geometryIndex: 0, visible: 'yes' }], /THREE\.BatchedMesh\._instanceInfo\[0\]\.visible must be a boolean/i],
    ['geometry index', [{ geometryIndex: -1 }], /THREE\.BatchedMesh\._instanceInfo\[0\]\.geometryIndex must be a non-negative integer/i],
  ]

  for (const [name, instanceInfo, pattern] of instanceCases) {
    const { scene, mesh } = makeBatchedScene()
    mesh._instanceInfo = instanceInfo
    assert.throws(
      () => getRenderer().render(scene, camera, { width: 32, height: 32, format: 'rgba' }),
      pattern,
      `${name} should fail clearly`,
    )
  }

  const matrixCases = [
    ['matrix texture container', (mesh) => {
      mesh._instanceInfo = [{ geometryIndex: 0 }]
      mesh._geometryInfo = [{ start: 0, count: 6 }]
      mesh._matricesTexture = 'matrices'
    }, /THREE\.BatchedMesh\._matricesTexture must be a texture-like object/i],
    ['matrix texture image container', (mesh) => {
      mesh._instanceInfo = [{ geometryIndex: 0 }]
      mesh._geometryInfo = [{ start: 0, count: 6 }]
      mesh._matricesTexture = { image: 'matrices' }
    }, /THREE\.BatchedMesh\._matricesTexture\.image must be an image-like object/i],
    ['matrix texture data container', (mesh) => {
      mesh._instanceInfo = [{ geometryIndex: 0 }]
      mesh._geometryInfo = [{ start: 0, count: 6 }]
      mesh._matricesTexture = { image: { data: 'matrices' } }
    }, /THREE\.BatchedMesh\._matricesTexture\.image\.data must be an array-like object/i],
    ['matrix texture finite values', (mesh) => {
      mesh._instanceInfo = [{ geometryIndex: 0 }]
      mesh._geometryInfo = [{ start: 0, count: 6 }]
      const matrix = new THREE.Matrix4().toArray()
      matrix[5] = Number.NaN
      mesh._matricesTexture = { image: { data: matrix } }
    }, /THREE\.BatchedMesh\._matricesTexture\.image\.data\[5\] must be a finite number/i],
    ['color texture image container', (mesh) => {
      mesh._instanceInfo = [{ geometryIndex: 0 }]
      mesh._geometryInfo = [{ start: 0, count: 6 }]
      mesh._matricesTexture = { image: { data: new Float32Array(16) } }
      mesh._colorsTexture = { image: 'colors' }
    }, /THREE\.BatchedMesh\._colorsTexture\.image must be an image-like object/i],
    ['color texture data container', (mesh) => {
      mesh._instanceInfo = [{ geometryIndex: 0 }]
      mesh._geometryInfo = [{ start: 0, count: 6 }]
      mesh._matricesTexture = { image: { data: new Float32Array(16) } }
      mesh._colorsTexture = { image: { data: 'colors' } }
    }, /THREE\.BatchedMesh\._colorsTexture\.image\.data must be an array-like object/i],
    ['color texture finite values', (mesh) => {
      mesh._instanceInfo = [{ geometryIndex: 0 }]
      mesh._geometryInfo = [{ start: 0, count: 6 }]
      mesh._matricesTexture = { image: { data: new THREE.Matrix4().toArray() } }
      mesh._colorsTexture = { image: { data: [1, Number.POSITIVE_INFINITY, 0, 1] } }
    }, /THREE\.BatchedMesh\._colorsTexture\.image\.data\[1\] must be a finite number/i],
  ]

  for (const [name, setup, pattern] of matrixCases) {
    const { scene, mesh } = makeBatchedScene()
    setup(mesh)
    assert.throws(
      () => getRenderer().render(scene, camera, { width: 32, height: 32, format: 'rgba' }),
      pattern,
      `${name} should fail clearly`,
    )
  }
})

test('BatchedMesh renders visible instance transforms and colors', () => {
  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 1.2, -1.2, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const source = new THREE.PlaneGeometry(0.45, 0.45)
  const material = new THREE.MeshBasicMaterial({ color: 0xffffff })
  const batched = new THREE.BatchedMesh(
    4,
    source.getAttribute('position').count,
    source.index.count,
    material,
  )
  const geometryId = batched.addGeometry(source)
  const left = batched.addInstance(geometryId)
  const right = batched.addInstance(geometryId)
  const hidden = batched.addInstance(geometryId)
  const deleted = batched.addInstance(geometryId)
  batched.setMatrixAt(left, new THREE.Matrix4().makeTranslation(-0.55, 0, 0))
  batched.setMatrixAt(right, new THREE.Matrix4().makeTranslation(0.55, 0, 0))
  batched.setMatrixAt(hidden, new THREE.Matrix4().makeTranslation(0, 0, 0))
  batched.setMatrixAt(deleted, new THREE.Matrix4().makeTranslation(0, -0.55, 0))
  batched.setColorAt(left, new THREE.Color(1, 0, 0))
  batched.setColorAt(right, new THREE.Color(0, 1, 0))
  batched.setColorAt(hidden, new THREE.Color(0, 0, 1))
  batched.setColorAt(deleted, new THREE.Color(1, 1, 0))
  batched.setVisibleAt(hidden, false)
  batched.deleteInstance(deleted)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(batched)

  const rgba = renderRgba(scene, camera, { width: 96, height: 64 })
  const leftMean = meanRegion(rgba, 96, 64, 20, 28, 30, 36)
  const rightMean = meanRegion(rgba, 96, 64, 66, 28, 76, 36)
  const centerMean = meanRegion(rgba, 96, 64, 44, 28, 52, 36)
  const bottomMean = meanRegion(rgba, 96, 64, 44, 42, 52, 50)

  assert.ok(leftMean.r > leftMean.g + 80 && leftMean.r > leftMean.b + 80, `left BatchedMesh instance should render red (${leftMean.r}, ${leftMean.g}, ${leftMean.b})`)
  assert.ok(rightMean.g > rightMean.r + 80 && rightMean.g > rightMean.b + 80, `right BatchedMesh instance should render green (${rightMean.r}, ${rightMean.g}, ${rightMean.b})`)
  assert.ok(centerMean.b < 5 && centerMean.r < 5 && centerMean.g < 5, `hidden BatchedMesh instance should not render at center (${centerMean.r}, ${centerMean.g}, ${centerMean.b})`)
  assert.ok(bottomMean.r < 5 && bottomMean.g < 5 && bottomMean.b < 5, `deleted BatchedMesh instance should not render at bottom (${bottomMean.r}, ${bottomMean.g}, ${bottomMean.b})`)
})

test('BatchedMesh skips deleted geometry ranges', () => {
  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 1.2, -1.2, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const source = new THREE.PlaneGeometry(0.45, 0.45)
  const batched = new THREE.BatchedMesh(
    2,
    source.getAttribute('position').count * 2,
    source.index.count * 2,
    new THREE.MeshBasicMaterial({ color: 0xffffff }),
  )
  const activeGeometryId = batched.addGeometry(source)
  const inactiveGeometryId = batched.addGeometry(source.clone())
  const left = batched.addInstance(activeGeometryId)
  const right = batched.addInstance(inactiveGeometryId)
  batched.setMatrixAt(left, new THREE.Matrix4().makeTranslation(-0.55, 0, 0))
  batched.setMatrixAt(right, new THREE.Matrix4().makeTranslation(0.55, 0, 0))
  batched.setColorAt(left, new THREE.Color(1, 0, 0))
  batched.setColorAt(right, new THREE.Color(0, 1, 0))
  batched.deleteGeometry(inactiveGeometryId)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(batched)

  const rgba = renderRgba(scene, camera, { width: 96, height: 64 })
  const leftMean = meanRegion(rgba, 96, 64, 20, 28, 30, 36)
  const rightMean = meanRegion(rgba, 96, 64, 66, 28, 76, 36)

  assert.ok(leftMean.r > leftMean.g + 80, `active BatchedMesh geometry should render red (${leftMean.r} vs ${leftMean.g})`)
  assert.ok(rightMean.r < 5 && rightMean.g < 5 && rightMean.b < 5, `deleted BatchedMesh geometry should skip its visible instance (${rightMean.r}, ${rightMean.g}, ${rightMean.b})`)
})

test('BatchedMesh optimize preserves repacked active geometry ranges', () => {
  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 1.2, -1.2, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const source = new THREE.PlaneGeometry(0.45, 0.8)
  const batched = new THREE.BatchedMesh(
    3,
    source.getAttribute('position').count * 3,
    source.index.count * 3,
    new THREE.MeshBasicMaterial({ color: 0xffffff }),
  )
  const leftGeometryId = batched.addGeometry(source)
  const middleGeometryId = batched.addGeometry(source.clone())
  const rightGeometryId = batched.addGeometry(source.clone())
  const left = batched.addInstance(leftGeometryId)
  const middle = batched.addInstance(middleGeometryId)
  const right = batched.addInstance(rightGeometryId)
  batched.setMatrixAt(left, new THREE.Matrix4().makeTranslation(-0.55, 0, 0))
  batched.setMatrixAt(middle, new THREE.Matrix4())
  batched.setMatrixAt(right, new THREE.Matrix4().makeTranslation(0.55, 0, 0))
  batched.setColorAt(left, new THREE.Color(1, 0, 0))
  batched.setColorAt(middle, new THREE.Color(0, 1, 0))
  batched.setColorAt(right, new THREE.Color(0, 0, 1))
  batched.frustumCulled = false
  batched.perObjectFrustumCulled = false
  batched.sortObjects = false

  const rightRangeBefore = batched.getGeometryRangeAt(rightGeometryId, {})
  batched.deleteGeometry(middleGeometryId)
  batched.optimize()
  const rightRangeAfter = batched.getGeometryRangeAt(rightGeometryId, {})
  assert.ok(rightRangeAfter.start < rightRangeBefore.start, 'optimize should move the right geometry into the deleted packed range')
  assert.equal(rightRangeAfter.count, rightRangeBefore.count)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(batched)

  const rgba = renderRgba(scene, camera, { width: 96, height: 64 })
  const leftMean = meanRegion(rgba, 96, 64, 20, 28, 30, 36)
  const centerMean = meanRegion(rgba, 96, 64, 43, 28, 53, 36)
  const rightMean = meanRegion(rgba, 96, 64, 66, 28, 76, 36)
  assert.ok(leftMean.r > leftMean.g + 80 && leftMean.r > leftMean.b + 80, `left optimized BatchedMesh geometry should remain red (${leftMean.r}, ${leftMean.g}, ${leftMean.b})`)
  assert.ok(centerMean.r < 5 && centerMean.g < 5 && centerMean.b < 5, `deleted optimized BatchedMesh geometry should leave the center empty (${centerMean.r}, ${centerMean.g}, ${centerMean.b})`)
  assert.ok(rightMean.b > rightMean.r + 80 && rightMean.b > rightMean.g + 80, `repacked BatchedMesh geometry should render blue on the right (${rightMean.r}, ${rightMean.g}, ${rightMean.b})`)
})

test('BatchedMesh material arrays honor packed geometry groups', () => {
  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 1.2, -1.2, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const source = new THREE.PlaneGeometry(0.45, 0.45)
  const batched = new THREE.BatchedMesh(
    2,
    source.getAttribute('position').count * 2,
    source.index.count * 2,
    [
      new THREE.MeshBasicMaterial({ color: 0xff0000 }),
      new THREE.MeshBasicMaterial({ color: 0x00ff00 }),
    ],
  )
  const leftGeometryId = batched.addGeometry(source)
  const rightGeometryId = batched.addGeometry(source.clone())
  const left = batched.addInstance(leftGeometryId)
  const right = batched.addInstance(rightGeometryId)
  batched.setMatrixAt(left, new THREE.Matrix4().makeTranslation(-0.55, 0, 0))
  batched.setMatrixAt(right, new THREE.Matrix4().makeTranslation(0.55, 0, 0))

  const leftRange = batched.getGeometryRangeAt(leftGeometryId, {})
  const rightRange = batched.getGeometryRangeAt(rightGeometryId, {})
  batched.geometry.clearGroups()
  batched.geometry.addGroup(leftRange.start, leftRange.count, 0)
  batched.geometry.addGroup(rightRange.start, rightRange.count, 1)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(batched)

  const rgba = renderRgba(scene, camera, { width: 96, height: 64 })
  const leftMean = meanRegion(rgba, 96, 64, 20, 28, 30, 36)
  const rightMean = meanRegion(rgba, 96, 64, 66, 28, 76, 36)
  assert.ok(leftMean.r > leftMean.g + 80 && leftMean.r > leftMean.b + 80, `left BatchedMesh geometry group should use the red material (${leftMean.r}, ${leftMean.g}, ${leftMean.b})`)
  assert.ok(rightMean.g > rightMean.r + 80 && rightMean.g > rightMean.b + 80, `right BatchedMesh geometry group should use the green material (${rightMean.r}, ${rightMean.g}, ${rightMean.b})`)
})
