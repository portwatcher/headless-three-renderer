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
import { extractAmbientLight, extractLights, test } from './scenes.test.part-001.mjs'
import { renderRgba } from './scenes.test.part-002.mjs'
test('invalid light color values fail clearly', () => {
  const directScene = new THREE.Scene()
  const directional = new THREE.DirectionalLight(0xffffff, 1)
  directional.color = { isColor: true, r: 1, g: 'green', b: 0 }
  directScene.add(directional)
  assert.throws(
    () => extractLights(directScene),
    /light\.color\.g must be a finite number/i,
  )

  const hemisphereScene = new THREE.Scene()
  const hemisphere = new THREE.HemisphereLight(0xffffff, 0x222222, 1)
  hemisphere.groundColor = { isColor: true, r: 0, g: 0, b: Number.NaN }
  hemisphereScene.add(hemisphere)
  assert.throws(
    () => extractLights(hemisphereScene),
    /HemisphereLight\.groundColor\.b must be a finite number/i,
  )

  const primitiveColorScene = new THREE.Scene()
  const primitiveColor = new THREE.PointLight(0xffffff, 1)
  primitiveColor.color = 123
  primitiveColorScene.add(primitiveColor)
  assert.throws(
    () => extractLights(primitiveColorScene),
    /light\.color must be a color-like object, CSS color string, or \[r, g, b\]/i,
  )

  const invalidCssScene = new THREE.Scene()
  const invalidCss = new THREE.PointLight(0xffffff, 1)
  invalidCss.color = 'not-a-color'
  invalidCssScene.add(invalidCss)
  assert.throws(
    () => extractLights(invalidCssScene),
    /light\.color "not-a-color" is not a supported CSS color string/i,
  )

  const ambientScene = new THREE.Scene()
  const ambient = new THREE.AmbientLight(0xffffff, 1)
  ambient.color = { isColor: true, r: 1, g: 1, b: 'blue' }
  ambientScene.add(ambient)
  assert.throws(
    () => extractAmbientLight(ambientScene),
    /AmbientLight\.color\.b must be a finite number/i,
  )
})

test('LOD selects object level from active camera distance', () => {
  const lod = new THREE.LOD()
  lod.addLevel(
    new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color: 0xff0000 })),
    0,
  )
  lod.addLevel(
    new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color: 0x0000ff })),
    4,
  )

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(lod)

  const nearCamera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  nearCamera.position.set(0, 0, 3)
  nearCamera.lookAt(0, 0, 0)

  const farCamera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  farCamera.position.set(0, 0, 6)
  farCamera.lookAt(0, 0, 0)

  const near = meanRgba(renderRgba(scene, nearCamera, { width: 64, height: 64 }))
  const far = meanRgba(renderRgba(scene, farCamera, { width: 64, height: 64 }))

  assert.ok(near.r > near.b + 10, `near LOD should render the red level (${near.r} vs ${near.b})`)
  assert.ok(far.b > far.r + 5, `far LOD should render the blue level (${far.b} vs ${far.r})`)
})

test('LOD selection accounts for camera zoom', () => {
  function makeScene() {
    const lod = new THREE.LOD()
    lod.addLevel(
      new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color: 0xff0000 })),
      0,
    )
    lod.addLevel(
      new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color: 0x0000ff })),
      4,
    )

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(lod)
    return scene
  }

  const farCamera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  farCamera.position.set(0, 0, 6)
  farCamera.lookAt(0, 0, 0)

  const zoomedCamera = farCamera.clone()
  zoomedCamera.zoom = 2

  const far = meanRgba(renderRgba(makeScene(), farCamera, { width: 64, height: 64 }))
  const zoomed = meanRgba(renderRgba(makeScene(), zoomedCamera, { width: 64, height: 64 }))

  assert.ok(far.b > far.r + 5, `unzoomed far LOD should render the blue level (${far.b} vs ${far.r})`)
  assert.ok(zoomed.r > zoomed.b + 10, `zoomed LOD distance should render the red level (${zoomed.r} vs ${zoomed.b})`)
})

test('LOD hysteresis preserves the previous farther level inside its threshold band', () => {
  function makeScene() {
    const lod = new THREE.LOD()
    lod.addLevel(
      new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color: 0xff0000 })),
      0,
    )
    const blueFar = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color: 0x0000ff }))
    blueFar.visible = false
    lod.addLevel(
      blueFar,
      4,
      0.25,
    )

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(lod)
    return scene
  }

  function makeDistanceCamera(distance) {
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, distance)
    camera.lookAt(0, 0, 0)
    return camera
  }

  const bandCamera = makeDistanceCamera(3.2)
  const baseline = meanRgba(renderRgba(makeScene(), bandCamera, { width: 64, height: 64 }))
  assert.ok(baseline.r > baseline.b + 10, `fresh LOD should render the red near level before hysteresis is active (${baseline.r} vs ${baseline.b})`)

  const scene = makeScene()
  const far = meanRgba(renderRgba(scene, makeDistanceCamera(6), { width: 64, height: 64 }))
  const retained = meanRgba(renderRgba(scene, bandCamera, { width: 64, height: 64 }))
  const released = meanRgba(renderRgba(scene, makeDistanceCamera(2.8), { width: 64, height: 64 }))

  assert.ok(far.b > far.r + 5, `far camera should render the blue LOD level (${far.b} vs ${far.r})`)
  assert.ok(retained.b > retained.r + 5, `hysteresis band should retain the blue LOD level (${retained.b} vs ${retained.r})`)
  assert.ok(released.r > released.b + 10, `inside hysteresis threshold should release back to the red LOD level (${released.r} vs ${released.b})`)
})

test('LOD autoUpdate=false preserves manual level visibility', () => {
  const redNear = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color: 0xff0000 }))
  const blueFar = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color: 0x0000ff }))
  redNear.visible = false
  blueFar.visible = true

  const lod = new THREE.LOD()
  lod.addLevel(redNear, 0)
  lod.addLevel(blueFar, 4)
  lod.autoUpdate = false

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(lod)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  assert.ok(mean.b > mean.r + 5, `manual blue LOD level should remain visible when autoUpdate=false (${mean.b} vs ${mean.r})`)
})

test('invalid LOD level values fail clearly', () => {
  function makeLodScene(mutator) {
    const lod = new THREE.LOD()
    lod.addLevel(
      new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color: 0xff0000 })),
      0,
    )
    lod.addLevel(
      new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color: 0x0000ff })),
      4,
    )
    mutator(lod)

    const scene = new THREE.Scene()
    scene.add(lod)
    return scene
  }

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 6)
  camera.lookAt(0, 0, 0)

  const malformedLevelsScene = new THREE.Scene()
  const malformedLevelsLod = new THREE.Object3D()
  malformedLevelsLod.isLOD = true
  malformedLevelsLod.levels = 'levels'
  malformedLevelsScene.add(malformedLevelsLod)
  assert.throws(
    () => renderRgba(malformedLevelsScene, camera, { width: 64, height: 64 }),
    /LOD\.levels must be an array/i,
    'levels container',
  )

  const cases = [
    ['autoUpdate', (lod) => {
      lod.autoUpdate = 'yes'
    }, /LOD\.autoUpdate must be a boolean/i],
    ['level entry', (lod) => {
      lod.levels[1] = null
    }, /LOD\.levels\[1\] must be an object/i],
    ['level object', (lod) => {
      lod.levels[1].object = null
    }, /LOD\.levels\[1\]\.object must be a THREE\.Object3D-like object/i],
    ['distance', (lod) => {
      lod.levels[1].distance = 'far'
    }, /LOD\.levels\[1\]\.distance must be a finite number/i],
    ['distance negative', (lod) => {
      lod.levels[1].distance = -1
    }, /LOD\.levels\[1\]\.distance must be non-negative/i],
    ['hysteresis', (lod) => {
      lod.levels[1].hysteresis = Number.POSITIVE_INFINITY
    }, /LOD\.levels\[1\]\.hysteresis must be a finite number/i],
    ['hysteresis negative', (lod) => {
      lod.levels[1].hysteresis = -0.1
    }, /LOD\.levels\[1\]\.hysteresis must be between 0 and 1/i],
    ['hysteresis above one', (lod) => {
      lod.levels[1].hysteresis = 1.5
    }, /LOD\.levels\[1\]\.hysteresis must be between 0 and 1/i],
  ]

  for (const [label, mutate, pattern] of cases) {
    assert.throws(
      () => renderRgba(makeLodScene(mutate), camera, { width: 64, height: 64 }),
      pattern,
      label,
    )
  }

  const invalidZoomCamera = camera.clone()
  invalidZoomCamera.zoom = Number.NaN
  assert.throws(
    () => renderRgba(makeLodScene(() => {}), invalidZoomCamera, { width: 64, height: 64 }),
    /camera\.zoom must be a finite number/i,
  )

  const zeroZoomCamera = camera.clone()
  zeroZoomCamera.zoom = 0
  assert.throws(
    () => renderRgba(makeLodScene(() => {}), zeroZoomCamera, { width: 64, height: 64 }),
    /camera\.zoom must be positive/i,
  )
})

test('Fog and FogExp2 affect material output', () => {
  function renderFogged(fog, materialFog = true) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.fog = fog
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ color: 0xff0000, fog: materialFog }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const linear = renderFogged(new THREE.Fog(0x00ff00, 0, 1))
  assert.ok(linear.g > linear.r + 40, `linear fog should mix the red plane toward green (${linear.g} vs ${linear.r})`)

  const exp2 = renderFogged(new THREE.FogExp2(0x0000ff, 1.0))
  assert.ok(exp2.b > exp2.r + 40, `FogExp2 should mix the red plane toward blue (${exp2.b} vs ${exp2.r})`)

  const optOut = renderFogged(new THREE.Fog(0x00ff00, 0, 1), false)
  assert.ok(
    optOut.r > optOut.g + 40,
    `material.fog=false should keep the red material color (${optOut.r} vs ${optOut.g})`,
  )
})

test('CSS fog color strings render with linear color semantics', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.fog = new THREE.Fog(0x000000, 0, 1)
  scene.fog.color = 'rgb(0, 255, 0)'
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  assert.ok(mean.g > mean.r + 40, `CSS fog color should mix the red plane toward green (${mean.g} vs ${mean.r})`)
})
