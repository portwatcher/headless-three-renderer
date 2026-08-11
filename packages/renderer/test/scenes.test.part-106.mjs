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
import { SIZE, test } from './scenes.test.part-001.mjs'
import { addLights, countRegionPixels, makeCamera, makeEnvironmentTexture, meanAbsDiff, meanRegion, renderRgba, solidTexture } from './scenes.test.part-002.mjs'
test('Fog uses view-space depth rather than Euclidean camera distance', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.fog = new THREE.Fog(0x00ff00, 2.9, 3.1)

  for (const x of [0, 2]) {
    const plane = new THREE.Mesh(
      new THREE.PlaneGeometry(0.8, 0.8),
      new THREE.MeshBasicMaterial({ color: 0xff0000 }),
    )
    plane.position.x = x
    scene.add(plane)
  }

  const camera = new THREE.OrthographicCamera(-3, 3, 2, -2, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 96, height: 96 })
  const center = meanRegion(rgba, 96, 96, 42, 42, 54, 54)
  const offAxis = meanRegion(rgba, 96, 96, 74, 42, 86, 54)
  assert.ok(Math.abs(center.g - offAxis.g) < 15, `same view-depth planes should receive similar linear fog (${center.g} vs ${offAxis.g})`)
  assert.ok(Math.abs(center.r - offAxis.r) < 15, `same view-depth planes should retain similar red output (${center.r} vs ${offAxis.r})`)
})

test('invalid fog parameter values fail clearly', () => {
  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ color: 0xff0000 })))
  const camera = makeCamera()

  scene.fog = 'fog'
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /scene\.fog must be a THREE\.Fog or THREE\.FogExp2 object/i,
  )

  scene.fog = { color: new THREE.Color(0x00ff00) }
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /scene\.fog must be a THREE\.Fog or THREE\.FogExp2 object/i,
  )

  scene.fog = new THREE.Fog(0x00ff00, 0, 1)
  scene.fog.near = Number.NaN
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /scene\.fog\.near must be a finite number/i,
  )

  scene.fog = new THREE.FogExp2(0x0000ff, 1)
  scene.fog.density = 'dense'
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /scene\.fog\.density must be a finite number/i,
  )

  scene.fog = new THREE.FogExp2(0x0000ff, 1)
  scene.fog.density = -0.1
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /scene\.fog\.density must be non-negative/i,
  )

  scene.fog = new THREE.Fog(0x00ff00, 10, 1)
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /scene\.fog\.far must be greater than scene\.fog\.near/i,
  )

  scene.fog = new THREE.Fog(0x00ff00, 1001, 2000)
  delete scene.fog.far
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /scene\.fog\.near must be less than the effective scene\.fog\.far/i,
  )

  scene.fog = new THREE.Fog(0x00ff00, 0, 1)
  scene.fog.color = { isColor: true, r: 0, g: Number.POSITIVE_INFINITY, b: 0 }
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /scene\.fog\.color\.g must be a finite number/i,
  )

  scene.fog = new THREE.Fog(0x00ff00, 0, 1)
  scene.fog.color = 123
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /scene\.fog\.color must be a color-like object, CSS color string, or \[r, g, b\]/i,
  )

  scene.fog = new THREE.Fog(0x00ff00, 0, 1)
  scene.fog.color = 'not-a-color'
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /scene\.fog\.color "not-a-color" is not a supported CSS color string/i,
  )
})

test('Fog and FogExp2 affect sprites, points, and lines with material fog opt-out', () => {
  function renderObject(object, fog = new THREE.Fog(0x00ff00, 0, 1)) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.fog = fog
    scene.add(object)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return renderRgba(scene, camera, { width: 64, height: 64 })
  }

  function makeSprite(fog) {
    const material = new THREE.SpriteMaterial({ color: 0xff0000 })
    material.fog = fog
    const sprite = new THREE.Sprite(material)
    sprite.scale.set(1.2, 1.2, 1)
    return sprite
  }

  function makePoint(fog) {
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))
    const material = new THREE.PointsMaterial({
      color: 0xff0000,
      size: 34,
      sizeAttenuation: false,
    })
    material.fog = fog
    return new THREE.Points(geometry, material)
  }

  function makeLine(fog) {
    const geometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-1, 0, 0),
      new THREE.Vector3(1, 0, 0),
    ])
    const material = new THREE.LineBasicMaterial({ color: 0xff0000 })
    material.fog = fog
    return new THREE.Line(geometry, material)
  }

  for (const [label, makeObject] of [
    ['sprite', makeSprite],
    ['point', makePoint],
  ]) {
    const fogged = meanRegion(renderObject(makeObject(true)), 64, 64, 24, 24, 40, 40)
    const unfogged = meanRegion(renderObject(makeObject(false)), 64, 64, 24, 24, 40, 40)
    assert.ok(fogged.g > fogged.r + 40, `${label} should be mixed toward green fog (${fogged.g} vs ${fogged.r})`)
    assert.ok(
      unfogged.r > unfogged.g + 40,
      `${label} fog=false should keep the red material color (${unfogged.r} vs ${unfogged.g})`,
    )

    const exp2 = meanRegion(
      renderObject(makeObject(true), new THREE.FogExp2(0x0000ff, 1.0)),
      64,
      64,
      24,
      24,
      40,
      40,
    )
    assert.ok(exp2.b > exp2.r + 40, `${label} should be mixed toward blue FogExp2 (${exp2.b} vs ${exp2.r})`)
  }

  const foggedLine = renderObject(makeLine(true))
  const unfoggedLine = renderObject(makeLine(false))
  const greenLinePixels = countRegionPixels(
    foggedLine,
    64,
    64,
    8,
    28,
    56,
    36,
    (r, g, b) => g > r + 30 && g > b + 30,
  )
  const redLinePixels = countRegionPixels(
    unfoggedLine,
    64,
    64,
    8,
    28,
    56,
    36,
    (r, g, b) => r > g + 30 && r > b + 30,
  )
  assert.ok(greenLinePixels > 2, `line should be mixed toward green fog (${greenLinePixels})`)
  assert.ok(redLinePixels > 2, `line fog=false should keep the red material color (${redLinePixels})`)

  const exp2Line = renderObject(makeLine(true), new THREE.FogExp2(0x0000ff, 1.0))
  const blueLinePixels = countRegionPixels(
    exp2Line,
    64,
    64,
    8,
    28,
    56,
    36,
    (r, g, b) => b > r + 30 && b > g + 30,
  )
  assert.ok(blueLinePixels > 2, `line should be mixed toward blue FogExp2 (${blueLinePixels})`)
})

test('PBR scene with lights renders and shows lighting variation', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.05, 0.05, 0.05)
  addLights(scene)
  scene.add(
    new THREE.Mesh(
      new THREE.SphereGeometry(1, 32, 32),
      new THREE.MeshStandardMaterial({ color: 0xdddddd, metalness: 0.1, roughness: 0.4 }),
    ),
  )

  const rgba = renderRgba(scene, makeCamera())
  const ratio = nonBackgroundRatio(rgba, [13, 13, 13])
  assert.ok(ratio > 0.05, 'sphere should be visible')

  // Sample both sides of the sphere — the lit side should be brighter than the shadowed side.
  // Top-right quadrant vs bottom-left quadrant of the image.
  let litSum = 0
  let litCount = 0
  let darkSum = 0
  let darkCount = 0
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const i = (y * SIZE + x) * 4
      const lum = rgba[i] + rgba[i + 1] + rgba[i + 2]
      if (lum < 50) continue // skip background
      if (x > SIZE * 0.6 && y < SIZE * 0.4) {
        litSum += lum
        litCount++
      } else if (x < SIZE * 0.4 && y > SIZE * 0.6) {
        darkSum += lum
        darkCount++
      }
    }
  }
  if (litCount > 0 && darkCount > 0) {
    const litAvg = litSum / litCount
    const darkAvg = darkSum / darkCount
    assert.ok(litAvg > darkAvg, `lit side (${litAvg.toFixed(1)}) should be brighter than shadowed side (${darkAvg.toFixed(1)})`)
  }
})

test('MeshPhysicalMaterial extensions and maps affect rendered output', () => {
  const camera = makeCamera()

  function makeScene(material) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0.04, 0.04, 0.045)
    scene.environment = makeEnvironmentTexture()
    scene.environmentIntensity = 0.8
    addLights(scene)
    scene.add(new THREE.Mesh(new THREE.SphereGeometry(1, 32, 32), material))
    return scene
  }

  const base = renderRgba(
    makeScene(new THREE.MeshPhysicalMaterial({
      color: 0x7aa7ff,
      roughness: 0.35,
      metalness: 0.0,
    })),
    camera,
  )
  const physical = renderRgba(
    makeScene(new THREE.MeshPhysicalMaterial({
      color: 0x7aa7ff,
      roughness: 0.35,
      metalness: 0.0,
      clearcoat: 1.0,
      clearcoatMap: solidTexture(255, 0, 0),
      clearcoatRoughness: 0.04,
      clearcoatRoughnessMap: solidTexture(0, 96, 0),
      clearcoatNormalMap: solidTexture(128, 180, 240),
      clearcoatNormalScale: new THREE.Vector2(0.6, 0.4),
      sheen: 0.8,
      sheenColor: new THREE.Color(1.0, 0.25, 0.12),
      sheenColorMap: solidTexture(255, 128, 96),
      sheenRoughness: 0.35,
      sheenRoughnessMap: solidTexture(0, 0, 0, 160),
      anisotropy: 0.85,
      anisotropyRotation: Math.PI / 4,
      anisotropyMap: solidTexture(255, 128, 255),
      transmission: 0.25,
      transmissionMap: solidTexture(180, 0, 0),
      ior: 1.45,
      thickness: 0.4,
      thicknessMap: solidTexture(0, 255, 0),
      attenuationColor: new THREE.Color(0.8, 0.95, 1.0),
      attenuationDistance: 1.5,
    })),
    camera,
  )

  const ratio = nonBackgroundRatio(physical, [10, 10, 11])
  assert.ok(ratio > 0.05, 'physical material sphere should be visible')
  const diff = meanAbsDiff(base, physical)
  assert.ok(diff > 0.5, `expected physical extensions to change output, mean abs diff=${diff.toFixed(3)}`)
})
