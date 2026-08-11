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
import { assertRgbClose, getRenderer, makeCamera, meanRegion, renderRgba, rgbaTexture, solidTexture } from './scenes.test.part-002.mjs'
test('Points with InstancedBufferGeometry expands selected instanced map UV channels', () => {
  const geometry = new THREE.InstancedBufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    0, 0, 0,
  ]), 3))
  geometry.setAttribute('instanceOffset', new THREE.InstancedBufferAttribute(new Float32Array([
    -0.35, 0, 0,
    0.35, 0, 0,
  ]), 3))
  geometry.setAttribute('uv', new THREE.InstancedBufferAttribute(new Float32Array([
    0.25, 0.5,
    0.25, 0.5,
  ]), 2))
  geometry.setAttribute('uv1', new THREE.InstancedBufferAttribute(new Float32Array([
    0.25, 0.5,
    0.75, 0.5,
  ]), 2))

  const map = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter
  map.channel = 1

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({
    color: 0xffffff,
    map,
    size: 24,
    sizeAttenuation: false,
  })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 96, height: 96 })
  const left = meanRegion(rgba, 96, 96, 24, 38, 42, 58)
  const right = meanRegion(rgba, 96, 96, 54, 38, 72, 58)
  assert.ok(left.r > left.g + 50, `left selected instanced point uv1 should sample red (${left.r} vs ${left.g})`)
  assert.ok(right.g > right.r + 50, `right selected instanced point uv1 should sample green (${right.g} vs ${right.r})`)
})

test('Points receiveShadow is accepted as an unlit WebGL-compatible no-op', () => {
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))

  const receiveScene = new THREE.Scene()
  receiveScene.background = new THREE.Color(0, 0, 0)
  const receiver = new THREE.Points(geometry, new THREE.PointsMaterial({ color: 0xffffff, size: 12 }))
  receiver.receiveShadow = true
  receiveScene.add(receiver)

  const mean = meanRegion(renderRgba(receiveScene, makeCamera(), { width: 64, height: 64 }), 64, 64, 28, 28, 36, 36)
  assert.ok(mean.r > 180 && mean.g > 180 && mean.b > 180, `points receiveShadow no-op should still render the unlit billboard (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('Points cast point-light shadows from expanded billboard quads', () => {
  function renderPointShadow(castShadow) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 2.2, 1.8]), 3))
    const caster = new THREE.Points(geometry, new THREE.PointsMaterial({
      color: 0xffffff,
      size: 48,
      sizeAttenuation: false,
    }))
    caster.castShadow = castShadow
    scene.add(caster)

    const light = new THREE.PointLight(0xffffff, 2)
    light.position.set(0, 5, 4)
    light.distance = 12
    light.castShadow = true
    light.shadow.mapSize.set(256, 256)
    light.shadow.camera.near = 0.1
    light.shadow.camera.far = 12
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, { width: 96, height: 96 }), 96, 96, 28, 42, 68, 82)
  }

  const unshadowed = renderPointShadow(false)
  const shadowed = renderPointShadow(true)
  const unshadowedLum = unshadowed.r + unshadowed.g + unshadowed.b
  const shadowedLum = shadowed.r + shadowed.g + shadowed.b
  assert.ok(shadowedLum < unshadowedLum - 10, `point billboard point-light shadow should darken the receiver (${shadowedLum} vs ${unshadowedLum})`)
})

test('empty scene renders the background color', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(1, 0, 0)
  const camera = makeCamera()

  const rgba = getRenderer().render(scene, camera, { width: 64, height: 64, format: 'rgba' })
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 200, `expected red background, got r=${mean.r}`)
  assert.ok(mean.g < 20, `expected red background, got g=${mean.g}`)
  assert.ok(mean.b < 20, `expected red background, got b=${mean.b}`)
})

test('THREE.Color backgrounds are encoded for sRGB output clears', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.015, 0.015, 0.02)
  const camera = makeCamera()

  const srgb = meanRgba(renderRgba(scene, camera, { width: 32, height: 32 }))
  const linear = meanRgba(renderRgba(scene, camera, {
    width: 32,
    height: 32,
    outputColorSpace: THREE.LinearSRGBColorSpace,
  }))
  const explicitArray = meanRgba(renderRgba(scene, camera, {
    width: 32,
    height: 32,
    background: [0.015, 0.015, 0.02],
  }))

  assertRgbClose(srgb, [33, 33, 39], 'THREE.Color scene background sRGB output')
  assertRgbClose(linear, [4, 4, 5], 'THREE.Color scene background linear output')
  assertRgbClose(explicitArray, [4, 4, 5], 'explicit array background channels')
})

test('backgroundIntensity scales background color clears', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(1, 0, 0)
  scene.backgroundIntensity = 0.5

  const camera = makeCamera()
  const dimmed = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  const override = meanRgba(renderRgba(scene, camera, { width: 64, height: 64, backgroundIntensity: 1 }))
  assert.ok(dimmed.r > 90 && dimmed.r < 170, `backgroundIntensity should dim red clears to about half strength (${dimmed.r})`)
  assert.ok(override.r > dimmed.r + 80, `options.backgroundIntensity should override scene.backgroundIntensity (${override.r} vs ${dimmed.r})`)
})

test('background color clears preserve alpha components', () => {
  const scene = new THREE.Scene()
  scene.background = { r: 1, g: 0, b: 0, a: 0.25 }
  const camera = makeCamera()

  const sceneAlpha = meanRgba(renderRgba(scene, camera, { width: 32, height: 32 }))
  const optionAlpha = meanRgba(renderRgba(scene, camera, {
    width: 32,
    height: 32,
    background: [0, 1, 0, 0.5],
  }))

  assert.ok(sceneAlpha.r > 200, `scene RGBA background should keep red color (${sceneAlpha.r})`)
  assert.ok(sceneAlpha.a > 60 && sceneAlpha.a < 70, `scene RGBA background should preserve quarter alpha (${sceneAlpha.a})`)
  assert.ok(optionAlpha.g > 200, `options.background RGBA clear should override to green (${optionAlpha.g})`)
  assert.ok(optionAlpha.a > 124 && optionAlpha.a < 132, `options.background RGBA clear should preserve half alpha (${optionAlpha.a})`)
})

test('invalid background control values fail clearly', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(1, 0, 0)
  const camera = makeCamera()

  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32, backgroundIntensity: Number.POSITIVE_INFINITY }),
    /options\.backgroundIntensity must be a finite number/i,
  )
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32, backgroundIntensity: -0.1 }),
    /options\.backgroundIntensity must be non-negative/i,
  )

  scene.backgroundIntensity = 'bright'
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /scene\.backgroundIntensity must be a finite number/i,
  )
  scene.backgroundIntensity = -0.1
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /scene\.backgroundIntensity must be non-negative/i,
  )
  scene.backgroundIntensity = 1

  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32, backgroundBlurriness: 'strong' }),
    /options\.backgroundBlurriness must be a finite number/i,
  )
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32, backgroundBlurriness: 1.5 }),
    /options\.backgroundBlurriness must be between 0 and 1/i,
  )

  scene.background = solidTexture(0, 255, 0)
  scene.backgroundBlurriness = 'soft'
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /scene\.backgroundBlurriness must be a finite number/i,
  )
  scene.backgroundBlurriness = -0.1
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /scene\.backgroundBlurriness must be between 0 and 1/i,
  )
})

test('invalid background color values fail clearly', () => {
  const camera = makeCamera()

  assert.throws(
    () => renderRgba(new THREE.Scene(), camera, { width: 32, height: 32, background: [1, 'green', 0] }),
    /options\.background\[1\] must be a finite number/i,
  )

  assert.throws(
    () => renderRgba(new THREE.Scene(), camera, { width: 32, height: 32, background: [1, 0] }),
    /options\.background must be \[r, g, b\] or \[r, g, b, a\]/i,
  )
  assert.throws(
    () => renderRgba(new THREE.Scene(), camera, { width: 32, height: 32, background: 'not-a-color' }),
    /options\.background "not-a-color" is not a supported CSS color string/i,
  )
  assert.throws(
    () => renderRgba(new THREE.Scene(), camera, { width: 32, height: 32, background: {} }),
    /options\.background must be a color, texture, or null/i,
  )

  const scene = new THREE.Scene()
  scene.background = { isColor: true, r: 0, g: Number.NaN, b: 1 }
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /scene\.background\.g must be a finite number/i,
  )
  scene.background = 'not-a-color'
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /scene\.background "not-a-color" is not a supported CSS color string/i,
  )
  scene.background = {}
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /scene\.background must be a color, texture, or null/i,
  )

  scene.background = solidTexture(0, 255, 0)
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32, background: { r: 1, g: 0 } }),
    /options\.background\.b must be a finite number/i,
  )
})

test('empty scene renders a texture background', () => {
  const scene = new THREE.Scene()
  scene.background = solidTexture(0, 255, 0)
  const camera = makeCamera()

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  assert.ok(mean.g > mean.r + 80, `expected green texture background, got ${mean.g} vs ${mean.r}`)
  assert.ok(mean.g > mean.b + 80, `expected green texture background, got ${mean.g} vs ${mean.b}`)
})

test('backgroundIntensity scales texture backgrounds', () => {
  const scene = new THREE.Scene()
  scene.background = solidTexture(0, 255, 0)

  const camera = makeCamera()
  const full = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  scene.backgroundIntensity = 0.25
  const dimmed = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  assert.ok(full.g > dimmed.g + 60, `backgroundIntensity should dim texture backgrounds (${full.g} vs ${dimmed.g})`)
  assert.ok(dimmed.g > dimmed.r + 40, `dimmed texture background should keep the sampled green hue (${dimmed.g} vs ${dimmed.r})`)
})
