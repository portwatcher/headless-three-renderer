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
import { meanRegion, renderRgba } from './scenes.test.part-002.mjs'
test('Renderer outputColorSpace state applies as render fallback', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(4, 4),
    new THREE.MeshBasicMaterial({ color: new THREE.Color(0.5, 0.5, 0.5) }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const renderer = new Renderer()
  assert.equal(renderer.outputColorSpace, THREE.SRGBColorSpace)
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace
  assert.equal(renderer.outputColorSpace, THREE.LinearSRGBColorSpace)

  const fallback = meanRgba(renderer.render(scene, camera, { width: 64, height: 64, format: 'rgba' }))
  const explicit = meanRgba(renderer.render(scene, camera, {
    width: 64,
    height: 64,
    format: 'rgba',
    outputColorSpace: THREE.SRGBColorSpace,
  }))
  assert.ok(
    explicit.r > fallback.r + 20,
    `explicit outputColorSpace should override Renderer outputColorSpace fallback (${explicit.r} vs ${fallback.r})`,
  )

  const target = renderer.renderToTarget(scene, camera, {}, { width: 32, height: 32 })
  const targetMean = meanRgba(target.data)
  assert.ok(
    explicit.r > targetMean.r + 20,
    `renderToTarget should use Renderer outputColorSpace fallback (${explicit.r} vs ${targetMean.r})`,
  )

  assert.throws(
    () => { renderer.outputColorSpace = 'display-p3' },
    /Renderer\.outputColorSpace display-p3 is not supported.*SRGBColorSpace.*LinearSRGBColorSpace/i,
  )
})

test('material.toneMapped=false skips material tone mapping before output conversion', () => {
  function renderToneMapped(toneMapped) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(4, 4),
      new THREE.MeshBasicMaterial({
        color: new THREE.Color(1, 1, 1),
        toneMapped,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }))
  }

  const mapped = renderToneMapped(true)
  const unmapped = renderToneMapped(false)
  assert.ok(
    unmapped.r > mapped.r + 35,
    `toneMapped=false should keep brighter linear white (${unmapped.r} vs ${mapped.r})`,
  )
  assert.ok(unmapped.r > 245, `toneMapped=false should preserve white output (${unmapped.r})`)
})

test('unlit sprite, point, and line materials honor toneMapped=false', () => {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  function renderPrimitive(kind, toneMapped) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)

    if (kind === 'sprite') {
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
        color: 0xffffff,
        toneMapped,
      }))
      sprite.scale.set(2, 2, 1)
      scene.add(sprite)
    } else if (kind === 'points') {
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))
      scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({
        color: 0xffffff,
        size: 48,
        sizeAttenuation: false,
        toneMapped,
      })))
    } else {
      const geometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-1.5, 0, 0),
        new THREE.Vector3(1.5, 0, 0),
      ])
      const material = kind === 'dashed-line'
        ? new THREE.LineDashedMaterial({
          color: 0xffffff,
          dashSize: 10,
          gapSize: 0,
          linewidth: 8,
          scale: 1,
          toneMapped,
        })
        : new THREE.LineBasicMaterial({
          color: 0xffffff,
          linewidth: 8,
          toneMapped,
        })
      const line = new THREE.Line(geometry, material)
      if (kind === 'dashed-line') line.computeLineDistances()
      scene.add(line)
    }

    const rgba = renderRgba(scene, camera, {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    })
    return kind === 'sprite' || kind === 'points'
      ? meanRegion(rgba, 64, 64, 20, 20, 44, 44)
      : meanRegion(rgba, 64, 64, 0, 30, 64, 34)
  }

  for (const kind of ['sprite', 'points', 'line', 'dashed-line']) {
    const mapped = renderPrimitive(kind, true)
    const unmapped = renderPrimitive(kind, false)
    assert.ok(
      unmapped.r > mapped.r + 35,
      `${kind} toneMapped=false should keep brighter linear white (${unmapped.r} vs ${mapped.r})`,
    )
    assert.ok(unmapped.r > 245, `${kind} toneMapped=false should preserve white output (${unmapped.r})`)
  }
})

test('Renderer toneMapping state controls material tone mapping exposure', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(4, 4),
    new THREE.MeshBasicMaterial({ color: new THREE.Color(1, 1, 1) }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const renderer = new Renderer()
  const renderToneMappingState = (options = {}) => meanRgba(renderer.render(scene, camera, {
    width: 64,
    height: 64,
    format: 'rgba',
    outputColorSpace: THREE.LinearSRGBColorSpace,
    ...options,
  }))

  assert.equal(renderer.toneMapping, THREE.ACESFilmicToneMapping)
  assert.equal(renderer.toneMappingExposure, 1)
  const mapped = renderToneMappingState()

  renderer.toneMapping = THREE.NoToneMapping
  const unmapped = renderToneMappingState()
  assert.ok(
    unmapped.r > mapped.r + 35,
    `Renderer.toneMapping=NoToneMapping should keep brighter linear white (${unmapped.r} vs ${mapped.r})`,
  )
  assert.ok(unmapped.r > 245, `Renderer.toneMapping=NoToneMapping should preserve white output (${unmapped.r})`)
  const optionMapped = renderToneMappingState({ toneMapping: THREE.ACESFilmicToneMapping })
  assert.equal(renderer.toneMapping, THREE.NoToneMapping)
  assert.ok(
    unmapped.r > optionMapped.r + 35,
    `options.toneMapping should override NoToneMapping renderer state for one render (${unmapped.r} vs ${optionMapped.r})`,
  )

  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.toneMappingExposure = 0.25
  const dimmed = renderToneMappingState()
  const optionBrightened = renderToneMappingState({ toneMappingExposure: 2 })
  assert.equal(renderer.toneMappingExposure, 0.25)
  renderer.toneMappingExposure = 2
  const brightened = renderToneMappingState()
  assert.ok(
    brightened.r > dimmed.r + 60,
    `Renderer.toneMappingExposure should scale ACES tone mapping (${brightened.r} vs ${dimmed.r})`,
  )
  assert.ok(
    optionBrightened.r > dimmed.r + 60,
    `options.toneMappingExposure should scale ACES tone mapping without mutating renderer state (${optionBrightened.r} vs ${dimmed.r})`,
  )

  renderer.toneMappingExposure = 1
  renderer.toneMapping = THREE.LinearToneMapping
  const linear = renderToneMappingState()
  renderer.toneMapping = THREE.ReinhardToneMapping
  const reinhard = renderToneMappingState()
  renderer.toneMapping = THREE.CineonToneMapping
  const cineon = renderToneMappingState()
  renderer.toneMapping = THREE.CustomToneMapping
  const custom = renderToneMappingState()
  renderer.toneMapping = THREE.AgXToneMapping
  const agx = renderToneMappingState()
  renderer.toneMapping = THREE.NeutralToneMapping
  const neutral = renderToneMappingState()
  assert.ok(linear.r > 245, `LinearToneMapping should preserve unclipped white at exposure 1 (${linear.r})`)
  assert.ok(
    reinhard.r < linear.r - 70,
    `ReinhardToneMapping should compress white below linear output (${reinhard.r} vs ${linear.r})`,
  )
  assert.ok(
    cineon.r > reinhard.r + 20 && cineon.r < linear.r - 20,
    `CineonToneMapping should land between Reinhard and linear white (${cineon.r}, ${reinhard.r}, ${linear.r})`,
  )
  assert.ok(
    Math.abs(custom.r - linear.r) < 2,
    `CustomToneMapping should use Three.js' default identity custom function (${custom.r} vs ${linear.r})`,
  )
  assert.ok(
    agx.r > 0 && agx.r < linear.r - 20,
    `AgXToneMapping should render finite compressed white (${agx.r} vs ${linear.r})`,
  )
  assert.ok(
    neutral.r > 0 && neutral.r < linear.r - 5,
    `NeutralToneMapping should render finite compressed white (${neutral.r} vs ${linear.r})`,
  )

  assert.throws(
    () => { renderer.toneMapping = 'aces' },
    /Renderer\.toneMapping must be a Three\.js tone mapping constant/i,
  )
  assert.throws(
    () => { renderer.toneMapping = 99 },
    /Renderer\.toneMapping 99 is not supported.*NoToneMapping.*LinearToneMapping.*ReinhardToneMapping.*CineonToneMapping.*ACESFilmicToneMapping.*CustomToneMapping.*AgXToneMapping.*NeutralToneMapping/i,
  )
  assert.throws(
    () => { renderer.toneMappingExposure = Number.NaN },
    /Renderer\.toneMappingExposure must be a finite number/i,
  )
  assert.throws(
    () => { renderer.toneMappingExposure = -0.1 },
    /Renderer\.toneMappingExposure must be non-negative/i,
  )
  assert.throws(
    () => renderToneMappingState({ toneMapping: 'aces' }),
    /options\.toneMapping must be a Three\.js tone mapping constant/i,
  )
  assert.throws(
    () => renderToneMappingState({ toneMapping: 99 }),
    /options\.toneMapping 99 is not supported.*NoToneMapping.*LinearToneMapping.*ReinhardToneMapping.*CineonToneMapping.*ACESFilmicToneMapping.*CustomToneMapping.*AgXToneMapping.*NeutralToneMapping/i,
  )
  assert.throws(
    () => renderToneMappingState({ toneMappingExposure: Number.NaN }),
    /options\.toneMappingExposure must be a finite number/i,
  )
  assert.throws(
    () => renderToneMappingState({ toneMappingExposure: -0.1 }),
    /options\.toneMappingExposure must be non-negative/i,
  )
})

test('outputColorSpace string aliases match Three.js constants', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.04, 0.08, 0.12)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(1.5, 1.5),
    new THREE.MeshBasicMaterial({ color: new THREE.Color(0.5, 0.28, 0.12) }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const renderColorSpace = (outputColorSpace) => renderRgba(scene, camera, {
    width: 32,
    height: 32,
    outputColorSpace,
  })
  const linear = renderColorSpace(THREE.LinearSRGBColorSpace)
  for (const alias of ['srgb-linear', 'linear-srgb', 'linearsrgb', 'linear']) {
    assert.deepEqual(renderColorSpace(alias), linear, `${alias} should match THREE.LinearSRGBColorSpace`)
  }
  assert.deepEqual(
    renderColorSpace('srgb'),
    renderColorSpace(THREE.SRGBColorSpace),
    'srgb should match THREE.SRGBColorSpace',
  )
})
