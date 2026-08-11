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
import { Renderer, renderToTarget, test } from './scenes.test.part-001.mjs'
import { assertRgbClose, countRegionPixels, makeCamera, meanAbsDiff, meanRegion, objectIdBytes, renderRgba, solidTexture } from './scenes.test.part-002.mjs'
test('renderMode normal outputs view-space normal colors', () => {
  function makeScene(material) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 1.2), material)
    mesh.rotation.y = Math.PI * 0.25
    scene.add(mesh)
    return scene
  }

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const renderModeNormals = renderRgba(
    makeScene(new THREE.MeshBasicMaterial({ color: 0xff0000 })),
    camera,
    { width: 64, height: 64, renderMode: 'normal' },
  )
  const materialNormals = renderRgba(
    makeScene(new THREE.MeshNormalMaterial()),
    camera,
    { width: 64, height: 64 },
  )

  const diff = meanAbsDiff(renderModeNormals, materialNormals)
  const center = meanRegion(renderModeNormals, 64, 64, 28, 28, 36, 36)
  const background = meanRegion(renderModeNormals, 64, 64, 0, 0, 8, 8)
  assert.ok(diff < 1, `renderMode normal should match MeshNormalMaterial output (diff=${diff.toFixed(2)})`)
  assert.ok(center.r > 120 && center.b > 200, `normal pass center should encode tilted view normal (${center.r}, ${center.g}, ${center.b})`)
  assert.ok(background.r < 2 && background.g < 2 && background.b < 2, `normal background should be black (${background.r}, ${background.g}, ${background.b})`)
})

test('renderMode depth outputs normalized depth grayscale', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(1, 0, 0)
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(1.2, 1.2), new THREE.MeshBasicMaterial({ color: 0x0088ff })))

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 64, height: 64, renderMode: 'depth' })
  const center = meanRegion(rgba, 64, 64, 28, 28, 36, 36)
  const corner = meanRegion(rgba, 64, 64, 0, 0, 8, 8)
  assert.ok(center.r > 150 && center.g > 150 && center.b > 150, `depth center should encode visible geometry (${center.r}, ${center.g}, ${center.b})`)
  assert.ok(Math.abs(center.r - center.g) < 2 && Math.abs(center.r - center.b) < 2, `depth output should be grayscale (${center.r}, ${center.g}, ${center.b})`)
  assert.ok(corner.r < 2 && corner.g < 2 && corner.b < 2, `depth background should be black (${corner.r}, ${corner.g}, ${corner.b})`)
})

test('renderMode object-id target includes reverse lookup metadata', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const left = new THREE.Mesh(new THREE.PlaneGeometry(0.75, 0.8), new THREE.MeshBasicMaterial({ color: 0xff0000 }))
  const right = new THREE.Mesh(new THREE.PlaneGeometry(0.75, 0.8), new THREE.MeshBasicMaterial({ color: 0x00ff00 }))
  left.position.x = -0.5
  right.position.x = 0.5
  scene.add(left, right)

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const target = { texture: {} }
  renderToTarget(scene, camera, target, { width: 64, height: 64, renderMode: 'object-id' })
  const leftEncoded = left.id + 1
  const rightEncoded = right.id + 1
  assert.equal(target.objectIdEntries.length, 2)
  assert.deepEqual(target.objectIdMap[String(leftEncoded)].rgb, objectIdBytes(leftEncoded))
  assert.deepEqual(target.objectIdMap[String(rightEncoded)].rgb, objectIdBytes(rightEncoded))
  assert.equal(target.objectIdMap[String(leftEncoded)].id, left.id)
  assert.equal(target.objectIdMap[String(rightEncoded)].hex, `#${rightEncoded.toString(16).padStart(6, '0')}`)

  const optionsTarget = { texture: {} }
  const returned = renderRgba(scene, camera, {
    width: 64,
    height: 64,
    renderMode: 'object-id',
    target: optionsTarget,
  })
  assert.equal(optionsTarget.data, returned)
  assert.equal(optionsTarget.objectIdEntries.length, 2)
  assert.equal(optionsTarget.objectIdMap[String(leftEncoded)].id, left.id)
  assert.equal(optionsTarget.objectIdMap[String(rightEncoded)].id, right.id)

  const rendererTarget = { texture: {} }
  const renderer = new Renderer()
  renderer.setRenderTarget(rendererTarget)
  const rendererReturned = renderer.render(scene, camera, {
    width: 64,
    height: 64,
    renderMode: 'object-id',
  })
  assert.equal(rendererTarget.data, rendererReturned)
  assert.equal(rendererTarget.objectIdEntries.length, 2)
  assert.equal(rendererTarget.objectIdMap[String(leftEncoded)].id, left.id)
  assert.equal(rendererTarget.objectIdMap[String(rightEncoded)].id, right.id)
  const rendererLeft = meanRegion(rendererTarget.data, 64, 64, 12, 24, 24, 40)
  assertRgbClose(rendererLeft, objectIdBytes(leftEncoded), 'Renderer.setRenderTarget object-id left mesh')

  renderer.render(scene, camera, { width: 64, height: 64 })
  assert.equal(rendererTarget.objectIdEntries, undefined)
  assert.equal(rendererTarget.objectIdMap, undefined)
  renderer.setRenderTarget(null)

  renderToTarget(scene, camera, target, { width: 64, height: 64 })
  assert.equal(target.objectIdEntries, undefined)
  assert.equal(target.objectIdMap, undefined)
})

test('renderMode auxiliary passes bypass post-processing', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(1, 0, 0)
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 1.2), new THREE.MeshBasicMaterial({ color: 0x0088ff }))
  mesh.rotation.y = Math.PI * 0.2
  scene.add(mesh)

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  for (const renderMode of ['mask', 'object-id', 'normal', 'depth']) {
    const base = renderRgba(scene, camera, { width: 64, height: 64, renderMode })
    const processed = renderRgba(scene, camera, {
      width: 64,
      height: 64,
      renderMode,
      postProcessing: {
        exposure: 4,
        contrast: 4,
        grayscale: true,
        invert: true,
        saturation: 0,
        vignette: 1,
      },
    })
    assert.deepEqual(processed, base, `${renderMode} should ignore post-processing effects`)
  }
})

test('renderMode auxiliary passes preserve texture alpha cutouts', () => {
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const makeBaseAlphaMaterial = (alpha) => new THREE.MeshBasicMaterial({
    map: solidTexture(255, 255, 255, alpha),
    alphaTest: 0.5,
  })
  const makeAlphaMapMaterial = (green) => new THREE.MeshBasicMaterial({
    alphaMap: solidTexture(255, green, 255),
    alphaTest: 0.5,
  })
  const cases = [
    ['base texture alpha', () => makeBaseAlphaMaterial(0), () => makeBaseAlphaMaterial(255)],
    ['alphaMap green channel', () => makeAlphaMapMaterial(0), () => makeAlphaMapMaterial(255)],
  ]

  for (const [label, makeDiscardedMaterial, makeVisibleMaterial] of cases) {
    for (const renderMode of ['mask', 'object-id', 'normal', 'depth']) {
      const scene = new THREE.Scene()
      const discarded = new THREE.Mesh(new THREE.PlaneGeometry(0.75, 0.8), makeDiscardedMaterial())
      const visible = new THREE.Mesh(new THREE.PlaneGeometry(0.75, 0.8), makeVisibleMaterial())
      discarded.position.x = -0.5
      visible.position.x = 0.5
      scene.add(discarded, visible)

      const rgba = renderRgba(scene, camera, { width: 64, height: 64, renderMode })
      const leftMean = meanRegion(rgba, 64, 64, 16, 28, 23, 36)
      const rightMean = meanRegion(rgba, 64, 64, 41, 28, 48, 36)
      assert.ok(leftMean.r < 2 && leftMean.g < 2 && leftMean.b < 2, `${renderMode} should discard ${label} pixels (${leftMean.r}, ${leftMean.g}, ${leftMean.b})`)

      if (renderMode === 'mask') {
        assert.ok(rightMean.r > 250 && rightMean.g > 250 && rightMean.b > 250, `mask should keep opaque ${label} pixels (${rightMean.r}, ${rightMean.g}, ${rightMean.b})`)
      } else if (renderMode === 'object-id') {
        assertRgbClose(rightMean, objectIdBytes(visible.id + 1), `object-id should keep opaque ${label} pixels`)
      } else if (renderMode === 'normal') {
        assert.ok(rightMean.r > 120 && rightMean.g > 120 && rightMean.b > 250, `normal should keep opaque ${label} pixels (${rightMean.r}, ${rightMean.g}, ${rightMean.b})`)
      } else {
        assert.ok(rightMean.r > 150 && rightMean.g > 150 && rightMean.b > 150, `depth should keep opaque ${label} pixels (${rightMean.r}, ${rightMean.g}, ${rightMean.b})`)
      }
    }
  }
})

test('renderMode auxiliary passes preserve alphaHash cutouts', () => {
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  function renderAuxiliary(renderMode, alphaHash) {
    const scene = new THREE.Scene()
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(1.2, 1.2),
      new THREE.MeshBasicMaterial({
        alphaHash,
        color: 0xffffff,
        opacity: alphaHash ? 0.35 : 1,
      }),
    ))
    return renderRgba(scene, camera, { width: 64, height: 64, renderMode })
  }

  for (const renderMode of ['mask', 'object-id', 'normal', 'depth']) {
    const opaque = renderAuxiliary(renderMode, false)
    const hashed = renderAuxiliary(renderMode, true)
    const visiblePixel = (r, g, b) => r > 0 || g > 0 || b > 0
    const opaquePixels = countRegionPixels(opaque, 64, 64, 20, 20, 44, 44, visiblePixel)
    const hashedPixels = countRegionPixels(hashed, 64, 64, 20, 20, 44, 44, visiblePixel)

    assert.ok(opaquePixels > 520, `${renderMode} opaque pass should fill the sampled region (${opaquePixels})`)
    assert.ok(hashedPixels > 40, `${renderMode} alphaHash pass should retain some visible pixels (${hashedPixels})`)
    assert.ok(hashedPixels < opaquePixels - 120, `${renderMode} alphaHash pass should discard visible pixels (${hashedPixels} vs ${opaquePixels})`)
  }
})

test('invalid renderMode values fail clearly', () => {
  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial()))
  assert.throws(
    () => renderRgba(scene, makeCamera(), { width: 32, height: 32, renderMode: 'normals' }),
    /options\.renderMode must be "color", "mask", "object-id", "normal", or "depth"/i,
  )
})

test('invalid material alphaTest values fail clearly', () => {
  const cases = [
    ['mesh', () => {
      const material = new THREE.MeshBasicMaterial({ color: 0xffffff })
      material.alphaTest = 'cutout'
      return new THREE.Mesh(new THREE.PlaneGeometry(1, 1), material)
    }],
    ['line', () => {
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-0.5, 0, 0, 0.5, 0, 0]), 3))
      const material = new THREE.LineBasicMaterial({ color: 0xffffff })
      material.alphaTest = Number.NaN
      return new THREE.Line(geometry, material)
    }],
  ]

  for (const [name, object] of cases) {
    const scene = new THREE.Scene()
    scene.add(object())
    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      /material\.alphaTest must be a finite number/i,
      `${name} alphaTest should fail clearly`,
    )
  }
})
