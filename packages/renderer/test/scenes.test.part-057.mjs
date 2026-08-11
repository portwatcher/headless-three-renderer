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
import { getRenderer, meanRegion, renderRgba } from './scenes.test.part-002.mjs'
test('transparent sort callbacks receive line and point group render items', () => {
  function assertGroupedSortItems(object, materials, label) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(object)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    const seenGroups = new Set()
    const seenMaterials = new Set()
    const transparentSort = (a, b) => {
      for (const item of [a, b]) {
        assert.equal(item.object, object, `${label} transparent sort item should expose source object`)
        assert.equal(item.geometry, object.geometry, `${label} transparent sort item should expose source geometry`)
        assert.ok(item.group, `${label} transparent sort item should expose geometry group`)
        assert.equal(item.material, materials[item.group.materialIndex], `${label} transparent sort item should expose grouped material`)
        seenGroups.add(item.group.materialIndex)
        seenMaterials.add(item.material)
      }
      return b.group.materialIndex - a.group.materialIndex
    }

    renderRgba(scene, camera, { width: 64, height: 64, transparentSort })
    assert.deepEqual([...seenGroups].sort(), [0, 1], `${label} transparent sort should see both geometry groups`)
    assert.deepEqual([...seenMaterials].sort((a, b) => materials.indexOf(a) - materials.indexOf(b)), materials)
  }

  const lineGeometry = new THREE.BufferGeometry()
  lineGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -1.3, -0.2, 0,
    -0.2, -0.2, 0,
    0.2, -0.2, 0,
    1.3, -0.2, 0,
  ]), 3))
  lineGeometry.addGroup(0, 2, 0)
  lineGeometry.addGroup(2, 2, 1)
  const lineMaterials = [
    new THREE.LineBasicMaterial({ color: 0xff0000, transparent: true, opacity: 0.75 }),
    new THREE.LineBasicMaterial({ color: 0x0000ff, transparent: true, opacity: 0.75 }),
  ]
  assertGroupedSortItems(new THREE.LineSegments(lineGeometry, lineMaterials), lineMaterials, 'line')

  const pointGeometry = new THREE.BufferGeometry()
  pointGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.5, 0.35, 0,
    0.5, 0.35, 0,
  ]), 3))
  pointGeometry.addGroup(0, 1, 0)
  pointGeometry.addGroup(1, 1, 1)
  const pointMaterials = [
    new THREE.PointsMaterial({ color: 0xff0000, transparent: true, opacity: 0.75, size: 18, sizeAttenuation: false }),
    new THREE.PointsMaterial({ color: 0x0000ff, transparent: true, opacity: 0.75, size: 18, sizeAttenuation: false }),
  ]
  assertGroupedSortItems(new THREE.Points(pointGeometry, pointMaterials), pointMaterials, 'point')
})

test('sortObjects=false preserves traversal order within transparent bucket', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const red = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xff0000, transparent: true, depthWrite: false }),
  )
  red.position.z = 0.35
  scene.add(red)

  const blue = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0x0000ff, transparent: true, depthWrite: false }),
  )
  blue.position.z = -0.35
  scene.add(blue)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const renderer = getRenderer()
  renderer.sortObjects = false
  let mean
  try {
    mean = meanRegion(renderer.render(scene, camera, { width: 64, height: 64, format: 'rgba' }), 64, 64, 24, 24, 40, 40)
  } finally {
    renderer.sortObjects = true
  }
  assert.ok(mean.b > mean.r + 160, `sortObjects=false should leave blue after red traversal order (${mean.b} vs ${mean.r})`)
})

test('Renderer opaque and transparent flags gate render buckets', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const opaque = new THREE.Mesh(
    new THREE.PlaneGeometry(0.8, 0.8),
    new THREE.MeshBasicMaterial({ color: 0x00ff00, depthTest: false, toneMapped: false }),
  )
  opaque.position.x = -1.1
  scene.add(opaque)

  const transmissive = new THREE.Mesh(
    new THREE.PlaneGeometry(0.8, 0.8),
    new THREE.MeshPhysicalMaterial({
      color: 0x000000,
      emissive: 0x0000ff,
      emissiveIntensity: 1,
      transmission: 0.5,
      depthTest: false,
      depthWrite: false,
      toneMapped: false,
    }),
  )
  scene.add(transmissive)

  const transparent = new THREE.Mesh(
    new THREE.PlaneGeometry(0.8, 0.8),
    new THREE.MeshBasicMaterial({ color: 0xff0000, transparent: true, depthTest: false, depthWrite: false, toneMapped: false }),
  )
  transparent.position.x = 1.1
  scene.add(transparent)

  const camera = new THREE.OrthographicCamera(-2, 2, 2, -2, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  camera.updateProjectionMatrix()

  const renderer = new Renderer()
  assert.equal(renderer.opaque, true)
  assert.equal(renderer.transparent, true)

  function bucketMeans(options = {}) {
    const rgba = renderer.render(scene, camera, { width: 64, height: 64, format: 'rgba', ...options })
    return {
      opaque: meanRegion(rgba, 64, 64, 11, 27, 21, 37),
      transmissive: meanRegion(rgba, 64, 64, 27, 27, 37, 37),
      transparent: meanRegion(rgba, 64, 64, 43, 27, 53, 37),
    }
  }

  const all = bucketMeans()
  assert.ok(all.opaque.g > all.opaque.r + 160, `default renderer bucket flags should draw opaque green (${all.opaque.r}, ${all.opaque.g}, ${all.opaque.b})`)
  assert.ok(all.transmissive.b > all.transmissive.r + 80, `default renderer bucket flags should draw transmissive blue (${all.transmissive.r}, ${all.transmissive.g}, ${all.transmissive.b})`)
  assert.ok(all.transparent.r > all.transparent.g + 160, `default renderer bucket flags should draw transparent red (${all.transparent.r}, ${all.transparent.g}, ${all.transparent.b})`)

  renderer.transparent = false
  const opaqueOnly = bucketMeans()
  assert.ok(opaqueOnly.opaque.g > opaqueOnly.opaque.r + 160, `transparent=false should keep the opaque green bucket (${opaqueOnly.opaque.r}, ${opaqueOnly.opaque.g}, ${opaqueOnly.opaque.b})`)
  assert.ok(opaqueOnly.transmissive.b < 5, `transparent=false should skip the transmissive blue bucket (${opaqueOnly.transmissive.r}, ${opaqueOnly.transmissive.g}, ${opaqueOnly.transmissive.b})`)
  assert.ok(opaqueOnly.transparent.r < 5, `transparent=false should skip the ordinary transparent red bucket (${opaqueOnly.transparent.r}, ${opaqueOnly.transparent.g}, ${opaqueOnly.transparent.b})`)

  renderer.opaque = false
  renderer.transparent = true
  const transparentOnly = bucketMeans()
  assert.ok(transparentOnly.opaque.g < 5, `opaque=false should skip the opaque green bucket (${transparentOnly.opaque.r}, ${transparentOnly.opaque.g}, ${transparentOnly.opaque.b})`)
  assert.ok(transparentOnly.transmissive.b > transparentOnly.transmissive.r + 80, `opaque=false should keep the transmissive blue bucket (${transparentOnly.transmissive.r}, ${transparentOnly.transmissive.g}, ${transparentOnly.transmissive.b})`)
  assert.ok(transparentOnly.transparent.r > transparentOnly.transparent.g + 160, `opaque=false should keep the ordinary transparent red bucket (${transparentOnly.transparent.r}, ${transparentOnly.transparent.g}, ${transparentOnly.transparent.b})`)

  renderer.transparent = false
  const backgroundOnly = bucketMeans()
  for (const [label, mean] of Object.entries(backgroundOnly)) {
    assert.ok(mean.r < 5 && mean.g < 5 && mean.b < 5, `opaque=false and transparent=false should leave only the background in the ${label} region (${mean.r}, ${mean.g}, ${mean.b})`)
  }

  const optionOpaqueOnly = bucketMeans({ opaque: true, transparent: false })
  assert.ok(optionOpaqueOnly.opaque.g > optionOpaqueOnly.opaque.r + 160, `options.opaque=true should override renderer state for the opaque green bucket (${optionOpaqueOnly.opaque.r}, ${optionOpaqueOnly.opaque.g}, ${optionOpaqueOnly.opaque.b})`)
  assert.ok(optionOpaqueOnly.transmissive.b < 5, `options.transparent=false should skip the transmissive blue bucket (${optionOpaqueOnly.transmissive.r}, ${optionOpaqueOnly.transmissive.g}, ${optionOpaqueOnly.transmissive.b})`)
  assert.ok(optionOpaqueOnly.transparent.r < 5, `options.transparent=false should skip the ordinary transparent red bucket (${optionOpaqueOnly.transparent.r}, ${optionOpaqueOnly.transparent.g}, ${optionOpaqueOnly.transparent.b})`)
  assert.equal(renderer.opaque, false)
  assert.equal(renderer.transparent, false)

  const optionTransparentOnly = bucketMeans({ opaque: false, transparent: true })
  assert.ok(optionTransparentOnly.opaque.g < 5, `options.opaque=false should skip the opaque green bucket (${optionTransparentOnly.opaque.r}, ${optionTransparentOnly.opaque.g}, ${optionTransparentOnly.opaque.b})`)
  assert.ok(optionTransparentOnly.transmissive.b > optionTransparentOnly.transmissive.r + 80, `options.transparent=true should override renderer state for the transmissive blue bucket (${optionTransparentOnly.transmissive.r}, ${optionTransparentOnly.transmissive.g}, ${optionTransparentOnly.transmissive.b})`)
  assert.ok(optionTransparentOnly.transparent.r > optionTransparentOnly.transparent.g + 160, `options.transparent=true should override renderer state for the ordinary transparent red bucket (${optionTransparentOnly.transparent.r}, ${optionTransparentOnly.transparent.g}, ${optionTransparentOnly.transparent.b})`)
  assert.equal(renderer.opaque, false)
  assert.equal(renderer.transparent, false)

  assert.throws(
    () => { renderer.opaque = 'yes' },
    /Renderer\.opaque must be a boolean/i,
  )
  assert.throws(
    () => { renderer.transparent = 'yes' },
    /Renderer\.transparent must be a boolean/i,
  )
})

test('invalid sort controls fail clearly', () => {
  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial()))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32, sortObjects: 'yes' }),
    /options\.sortObjects must be a boolean/i,
  )
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32, opaque: 'yes' }),
    /options\.opaque must be a boolean/i,
  )
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32, transparent: 1 }),
    /options\.transparent must be a boolean/i,
  )
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32, opaqueSort: 'front' }),
    /options\.opaqueSort must be a function or null/i,
  )
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32, transparentSort: 1 }),
    /options\.transparentSort must be a function or null/i,
  )

  const renderer = getRenderer()
  assert.throws(
    () => { renderer.sortObjects = 'yes' },
    /Renderer\.sortObjects must be a boolean/i,
  )
  assert.throws(
    () => renderer.setOpaqueSort('front'),
    /Renderer\.setOpaqueSort expects a function or null/i,
  )
  assert.throws(
    () => renderer.setTransparentSort(1),
    /Renderer\.setTransparentSort expects a function or null/i,
  )
})

test('invalid renderOrder values fail clearly', () => {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const cases = [
    ['mesh renderOrder', () => {
      const scene = new THREE.Scene()
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial())
      mesh.renderOrder = Number.NaN
      scene.add(mesh)
      return scene
    }],
    ['group renderOrder', () => {
      const scene = new THREE.Scene()
      const group = new THREE.Group()
      group.renderOrder = 'front'
      group.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial()))
      scene.add(group)
      return scene
    }],
  ]

  for (const [label, makeScene] of cases) {
    assert.throws(
      () => renderRgba(makeScene(), camera, { width: 64, height: 64 }),
      /object\.renderOrder must be a finite number/i,
      label,
    )
  }
})

test('transparent sort depth uses geometry bounding sphere center', () => {
  function offsetPlane(zOffset, color) {
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
      -1, -1, zOffset,
      1, -1, zOffset,
      1, 1, zOffset,
      -1, 1, zOffset,
    ]), 3))
    geometry.setIndex([0, 1, 2, 0, 2, 3])

    return new THREE.Mesh(
      geometry,
      new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.6,
        depthWrite: false,
      }),
    )
  }

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(offsetPlane(0.45, 0xff0000))
  scene.add(offsetPlane(-0.45, 0x0000ff))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 24, 24, 40, 40)
  assert.ok(mean.r > mean.b + 20, `near red geometry center should sort over far blue despite matching object origins (${mean.r} vs ${mean.b})`)
})
