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
import { countRegionPixels, meanRegion, renderRgba, renderRgbaIsolated } from './scenes.test.part-002.mjs'
test('examples WorkerPool can drive still-frame render input preparation', async () => {
  const workers = []
  const pool = new WorkerPool(1)
  pool.setWorkerCreator(() => {
    const worker = {
      messages: [],
      terminated: false,
      listener: null,
      addEventListener(type, listener) {
        assert.equal(type, 'message')
        this.listener = listener
      },
      postMessage(message, transfer) {
        this.messages.push({ message, transfer })
        queueMicrotask(() => {
          this.listener({
            data: {
              id: message.id,
              x: message.x,
              color: message.color,
            },
          })
        })
      },
      terminate() {
        this.terminated = true
      },
    }
    workers.push(worker)
    return worker
  })

  const transferBuffer = new ArrayBuffer(4)
  const [left, center, right] = await Promise.all([
    pool.postMessage({ id: 'left', x: -0.55, color: 0xff3344 }, [transferBuffer]),
    pool.postMessage({ id: 'center', x: 0, color: 0x44ff66 }),
    pool.postMessage({ id: 'right', x: 0.55, color: 0x4488ff }),
  ])

  assert.equal(workers.length, 1)
  assert.equal(workers[0].messages.length, 3)
  assert.strictEqual(workers[0].messages[0].transfer[0], transferBuffer)
  assert.deepEqual([left.data.id, center.data.id, right.data.id], ['left', 'center', 'right'])
  assert.equal(pool.queue.length, 0)
  assert.equal(pool.workerStatus, 0)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  const geometry = new THREE.PlaneGeometry(0.34, 0.5)
  const materials = []
  for (const result of [left, center, right]) {
    const material = new THREE.MeshBasicMaterial({ color: result.data.color, side: THREE.DoubleSide })
    materials.push(material)
    const mesh = new THREE.Mesh(geometry, material)
    mesh.position.x = result.data.x
    scene.add(mesh)
  }

  const camera = new THREE.OrthographicCamera(-1, 1, 0.7, -0.7, 0.01, 10)
  camera.position.set(0, 0, 4)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  try {
    const width = 96
    const height = 72
    const rgba = renderRgba(scene, camera, { width, height })
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => r > 160 && g < 120 && b < 120) > 120,
      'WorkerPool-prepared red mesh should render visible pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => g > 130 && g > r + 30 && g > b + 20) > 120,
      'WorkerPool-prepared green mesh should render visible pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => b > 160 && r < 140 && g < 170) > 120,
      'WorkerPool-prepared blue mesh should render visible pixels',
    )
  } finally {
    pool.dispose()
    geometry.dispose()
    for (const material of materials) material.dispose()
  }

  assert.equal(workers[0].terminated, true)
  assert.equal(pool.workers.length, 0)
  assert.equal(pool.workersResolve.length, 0)
})

test('BatchedMesh default onBeforeRender does not re-enter customSort', () => {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const source = new THREE.PlaneGeometry(1, 1)
  const material = new THREE.MeshBasicMaterial({ color: 0xff0000, transparent: true })
  const batched = new THREE.BatchedMesh(
    2,
    source.getAttribute('position').count,
    source.index.count,
    material,
  )
  const geometryId = batched.addGeometry(source)
  const left = batched.addInstance(geometryId)
  const right = batched.addInstance(geometryId)
  batched.setMatrixAt(left, new THREE.Matrix4().makeTranslation(-0.35, 0, 0))
  batched.setMatrixAt(right, new THREE.Matrix4().makeTranslation(0.35, 0, 0))

  let customSortCalls = 0
  batched.setCustomSort((list) => {
    customSortCalls += 1
    list.sort((a, b) => a.index - b.index)
  })

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(batched)

  renderRgba(scene, camera, { width: 64, height: 64 })
  assert.equal(customSortCalls, 1, 'CPU BatchedMesh sorting should not invoke the built-in multidraw onBeforeRender path')

  let userBeforeRenderCalls = 0
  batched.onBeforeRender = function () {
    userBeforeRenderCalls += 1
  }
  renderRgba(scene, camera, { width: 64, height: 64 })
  assert.equal(customSortCalls, 2, 'explicit BatchedMesh onBeforeRender callbacks should not suppress CPU customSort')
  assert.ok(userBeforeRenderCalls > 0, 'explicit BatchedMesh onBeforeRender callbacks should still run')
})

test('BatchedMesh renderer sort callbacks receive the source object', () => {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const source = new THREE.PlaneGeometry(1, 1)
  const material = new THREE.MeshBasicMaterial({
    color: 0xffffff,
    depthWrite: false,
    transparent: true,
  })
  const batched = new THREE.BatchedMesh(
    2,
    source.getAttribute('position').count,
    source.index.count,
    material,
  )
  const geometryId = batched.addGeometry(source)
  const left = batched.addInstance(geometryId)
  const right = batched.addInstance(geometryId)
  batched.setMatrixAt(left, new THREE.Matrix4().makeTranslation(-0.25, 0, 0))
  batched.setMatrixAt(right, new THREE.Matrix4().makeTranslation(0.25, 0, 0))

  const scene = new THREE.Scene()
  scene.add(batched)

  let calls = 0
  renderRgba(scene, camera, {
    width: 64,
    height: 64,
    transparentSort: (a, b) => {
      calls += 1
      assert.equal(a.object, batched)
      assert.equal(b.object, batched)
      assert.equal(a.material, material)
      assert.equal(b.material, material)
      return 0
    },
  })

  assert.ok(calls > 0, 'transparentSort should compare BatchedMesh-expanded draw items')
})

test('BatchedMesh renderer sort callbacks receive range-local depth values', () => {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const nearGeometry = new THREE.PlaneGeometry(2, 2)
  nearGeometry.translate(0, 0, 0.35)
  const farGeometry = new THREE.PlaneGeometry(2, 2)
  farGeometry.translate(0, 0, -0.35)

  const material = new THREE.MeshBasicMaterial({
    color: 0xffffff,
    depthWrite: false,
    transparent: true,
  })
  const batched = new THREE.BatchedMesh(
    2,
    nearGeometry.getAttribute('position').count + farGeometry.getAttribute('position').count,
    nearGeometry.index.count + farGeometry.index.count,
    material,
  )
  const nearGeometryId = batched.addGeometry(nearGeometry)
  const farGeometryId = batched.addGeometry(farGeometry)
  const near = batched.addInstance(nearGeometryId)
  const far = batched.addInstance(farGeometryId)
  batched.setMatrixAt(near, new THREE.Matrix4())
  batched.setMatrixAt(far, new THREE.Matrix4())
  batched.setColorAt(near, new THREE.Color(1, 0, 0))
  batched.setColorAt(far, new THREE.Color(0, 0, 1))

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(batched)

  const seenZ = new Set()
  let calls = 0
  const rgba = renderRgbaIsolated(scene, camera, {
    width: 64,
    height: 64,
    transparentSort: (a, b) => {
      calls += 1
      assert.equal(a.object, batched)
      assert.equal(b.object, batched)
      seenZ.add(a.z)
      seenZ.add(b.z)
      return a.z - b.z
    },
  })

  assert.ok(calls > 0, 'transparentSort should compare BatchedMesh-expanded draw depths')
  assert.equal(seenZ.size, 2, 'BatchedMesh-expanded render items should expose per-range z values')
  const mean = meanRegion(rgba, 64, 64, 24, 24, 40, 40)
  assert.ok(mean.b > mean.r + 80, `custom renderer sort should draw farther blue BatchedMesh range last (${mean.b} vs ${mean.r})`)
})

test('BatchedMesh sort callbacks receive packed geometry group render items', () => {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const source = new THREE.BufferGeometry()
  source.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -1, -1, 0,
    1, -1, 0,
    1, 1, 0,
    -1, 1, 0,
    -1, -1, 0,
    1, -1, 0,
    1, 1, 0,
    -1, 1, 0,
  ]), 3))
  source.setIndex([
    0, 1, 2,
    0, 2, 3,
    4, 5, 6,
    4, 6, 7,
  ])

  const materials = [
    new THREE.MeshBasicMaterial({ color: 0xff0000, transparent: true, depthTest: false, depthWrite: false }),
    new THREE.MeshBasicMaterial({ color: 0x0000ff, transparent: true, depthTest: false, depthWrite: false }),
  ]
  const batched = new THREE.BatchedMesh(
    1,
    source.getAttribute('position').count,
    source.index.count,
    materials,
  )
  const geometryId = batched.addGeometry(source)
  batched.addInstance(geometryId)

  const range = batched.getGeometryRangeAt(geometryId, {})
  batched.geometry.clearGroups()
  batched.geometry.addGroup(range.start, 6, 0)
  batched.geometry.addGroup(range.start + 6, 6, 1)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(batched)

  const seenGroups = new Set()
  const seenMaterials = new Set()
  let calls = 0
  const rgba = renderRgba(scene, camera, {
    width: 64,
    height: 64,
    transparentSort: (a, b) => {
      calls += 1
      assert.equal(a.object, batched)
      assert.equal(b.object, batched)
      assert.ok(a.group)
      assert.ok(b.group)
      seenGroups.add(a.group.materialIndex)
      seenGroups.add(b.group.materialIndex)
      seenMaterials.add(a.material)
      seenMaterials.add(b.material)
      return b.group.materialIndex - a.group.materialIndex
    },
  })

  assert.ok(calls > 0, 'transparentSort should compare BatchedMesh packed group items')
  assert.deepEqual([...seenGroups].sort(), [0, 1])
  assert.deepEqual([...seenMaterials].sort((a, b) => materials.indexOf(a) - materials.indexOf(b)), materials)
  const mean = meanRegion(rgba, 64, 64, 24, 24, 40, 40)
  assert.ok(mean.r > mean.b + 160, `custom group-aware BatchedMesh sort should draw red after blue (${mean.r} vs ${mean.b})`)
})
