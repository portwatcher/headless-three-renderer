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
import { makeCamera, renderRgba } from './scenes.test.part-002.mjs'
test('examples LUT loaders parse 3D LUT textures and fail clearly as material maps', () => {
  const lut3dl = new LUT3dlLoader().parse([
    '0 1',
    '0 0 0',
    '0 0 1',
    '0 1 0',
    '0 1 1',
    '1 0 0',
    '1 0 1',
    '1 1 0',
    '1 1 1',
  ].join('\n'))
  const cube = new LUTCubeLoader().parse([
    'TITLE "tiny"',
    'LUT_3D_SIZE 2',
    '0 0 0',
    '0 0 1',
    '0 1 0',
    '0 1 1',
    '1 0 0',
    '1 0 1',
    '1 1 0',
    '1 1 1',
  ].join('\n'))

  const geometry = new THREE.PlaneGeometry(1, 1)
  const material = new THREE.MeshBasicMaterial({ map: cube.texture3D })
  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(geometry, material))
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10)
  camera.position.z = 2
  camera.updateMatrixWorld(true)

  try {
    assert.equal(lut3dl.size, 2)
    assert.equal(lut3dl.texture3D.isData3DTexture, true)
    assert.equal(lut3dl.texture3D.image.data.length, 32)
    assert.equal(cube.title, 'tiny')
    assert.equal(cube.size, 2)
    assert.equal(cube.texture3D.isData3DTexture, true)
    assert.throws(
      () => renderRgba(scene, camera, { width: 32, height: 32 }),
      /material\.map uses an array or 3D texture.*not supported.*2D texture image/i,
    )
  } finally {
    geometry.dispose()
    material.dispose()
    lut3dl.texture3D.dispose()
    cube.texture3D.dispose()
  }
})

test('examples IESLoader parses photometric textures and fails clearly as material maps', () => {
  const texture = new IESLoader().parse([
    'IESNA:LM-63-1995',
    'TILT=NONE',
    '1 1000 1 2 2 1 1 1 1 1',
    '1 1 1',
    '0 90',
    '0 90',
    '1 0.5',
    '0.5 0.25',
  ].join('\n'))

  const geometry = new THREE.PlaneGeometry(1, 1)
  const material = new THREE.MeshBasicMaterial({ map: texture })
  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(geometry, material))
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10)
  camera.position.z = 2
  camera.updateMatrixWorld(true)

  try {
    assert.equal(texture.isDataTexture, true)
    assert.equal(texture.format, THREE.RedFormat)
    assert.equal(texture.type, THREE.HalfFloatType)
    assert.equal(texture.image.width, 180)
    assert.equal(texture.image.height, 1)
    assert.equal(texture.image.data instanceof Uint16Array, true)
    assert.equal(texture.image.data.length, 360 * 180)
    assert.throws(
      () => renderRgba(scene, camera, { width: 64, height: 64 }),
      /material\.map raw texture data must contain one-channel, two-channel, RGB, or RGBA numeric pixel data.*mismatched data lengths/i,
    )
  } finally {
    geometry.dispose()
    material.dispose()
    texture.dispose()
  }
})

test('GPUComputationRenderer stops at conservative vertex texture support detection', () => {
  const renderer = new Renderer()
  const gpuCompute = new GPUComputationRenderer(2, 2, renderer)
  const initialState = gpuCompute.createTexture()
  assert.equal(initialState.image.data.length, 2 * 2 * 4)

  gpuCompute.addVariable('textureState', 'void main() { gl_FragColor = vec4( 1.0 ); }', initialState)
  assert.equal(gpuCompute.init(), 'No support for vertex shader textures.')
  gpuCompute.dispose()
})

test('examples TiledLighting builds tiled point-light metadata before compute boundary', () => {
  const point = new THREE.PointLight(0x804020, 2, 5, 1.5)
  point.position.set(1, 2, 3)
  point.updateMatrixWorld(true)
  const directional = new THREE.DirectionalLight(0xffffff, 1)

  const node = new TiledLighting().createNode([point, directional])
  assert.equal(node.tiledLights.length, 1)
  assert.equal(node.tiledLights[0], point)
  assert.equal(node.materialLights.length, 1)
  assert.equal(node.materialLights[0], directional)
  assert.equal(node.hasLights, true)

  node.setSize(33, 63)
  assert.equal(node._bufferSize.width, 64)
  assert.equal(node._bufferSize.height, 64)
  assert.equal(node._lightsTexture.image.width, 1024)
  assert.equal(node._lightsTexture.image.height, 2)

  node.updateLightsTexture()
  const positionLine = node._lightsTexture.image.data
  const colorLineOffset = node._lightsTexture.image.width * 4
  assert.deepEqual(Array.from(positionLine.slice(0, 4)), [1, 2, 3, 5])
  assert.ok(positionLine[colorLineOffset] > positionLine[colorLineOffset + 1])
  assert.equal(positionLine[colorLineOffset + 3], 1.5)

  const renderer = new Renderer({ width: 16, height: 16 })
  const camera = makeCamera()
  try {
    assert.throws(
      () => node.updateBefore({ renderer, camera }),
      /Renderer\.compute\(\) is not supported.*WebGPU compute pipelines/i,
    )
  } finally {
    renderer.dispose()
  }
})

test('CSM internals expose frustum splits, shader chunks, and shadow-node cascade state', () => {
  const camera = new THREE.PerspectiveCamera(55, 1, 0.5, 20)
  camera.updateProjectionMatrix()
  camera.position.set(0, 0, 4)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  const frustum = new CSMFrustum({ projectionMatrix: camera.projectionMatrix, maxFar: 12 })
  const cascades = []
  frustum.split([0.25, 0.65, 1], cascades)
  assert.equal(cascades.length, 3)
  assert.ok(cascades[0].vertices.near[0].distanceTo(frustum.vertices.near[0]) < 1e-6)
  assert.ok(cascades[2].vertices.far[0].distanceTo(frustum.vertices.far[0]) < 1e-6)
  assert.ok(cascades[1].vertices.near[0].distanceTo(cascades[0].vertices.far[0]) < 1e-6)

  const translated = new CSMFrustum()
  cascades[0].toSpace(new THREE.Matrix4().makeTranslation(1, 2, 3), translated)
  assert.ok(translated.vertices.near[0].distanceTo(cascades[0].vertices.near[0].clone().add(new THREE.Vector3(1, 2, 3))) < 1e-6)
  assert.match(CSMShader.lights_fragment_begin, /USE_CSM/)
  assert.match(CSMShader.lights_pars_begin, /CSM_cascades/)

  const scene = new THREE.Scene()
  const light = new THREE.DirectionalLight(0xffffff, 1)
  light.position.set(4, 5, 6)
  light.target.position.set(0, 0, 0)
  scene.add(light, light.target)

  const renderer = new Renderer()
  const shadowNode = new CSMShadowNode(light, { cascades: 2, maxFar: 12, mode: 'uniform', lightMargin: 3 })
  try {
    const initializeShadowNode = shadowNode.init ?? shadowNode._init ?? shadowNode.setup
    assert.equal(typeof initializeShadowNode, 'function')
    initializeShadowNode.call(shadowNode, { camera, renderer })
    const expectedBreak = (camera.near + (Math.min(camera.far, shadowNode.maxFar) - camera.near) / 2) / Math.min(camera.far, shadowNode.maxFar)
    assert.equal(shadowNode.mainFrustum instanceof CSMFrustum, true)
    assert.equal(shadowNode.lights.length, 2)
    assert.equal(shadowNode.frustums.length, 2)
    assert.deepEqual(shadowNode.breaks, [expectedBreak, 1])
    assert.deepEqual(shadowNode._cascades.map((entry) => [entry.x, entry.y]), [[0, expectedBreak], [expectedBreak, 1]])
    assert.ok(shadowNode.lights.every((cascadeLight) => cascadeLight.parent === null || cascadeLight.parent === scene))
    shadowNode.updateBefore()
    assert.ok(shadowNode.lights.every((cascadeLight) => Number.isFinite(cascadeLight.position.x)))
    assert.ok(shadowNode.lights.every((cascadeLight) => cascadeLight.parent === scene))
  } finally {
    if (shadowNode.lights.every((cascadeLight) => cascadeLight.parent !== null)) shadowNode.dispose()
    renderer.dispose()
  }

  assert.ok(shadowNode.lights.every((cascadeLight) => cascadeLight.parent === null))
})

test('CSM material shader injection fails clearly', () => {
  const scene = new THREE.Scene()
  const camera = makeCamera()
  const previousLightsFragmentBegin = THREE.ShaderChunk.lights_fragment_begin
  const previousLightsParsBegin = THREE.ShaderChunk.lights_pars_begin
  const csm = new CSM({
    camera,
    parent: scene,
    cascades: 2,
    shadowMapSize: 16,
  })
  const material = new THREE.MeshStandardMaterial({ color: 0xffffff })
  csm.setupMaterial(material)
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(1, 1), material))

  try {
    assert.throws(
      () => renderRgba(scene, camera, { width: 16, height: 16 }),
      /CSM material onBeforeCompile customization.*not translated.*native lights and shadows.*fragmentWgsl/i,
    )
  } finally {
    csm.dispose()
    csm.remove()
    THREE.ShaderChunk.lights_fragment_begin = previousLightsFragmentBegin
    THREE.ShaderChunk.lights_pars_begin = previousLightsParsBegin
  }
})

test('CurveModifier Flow shader injection fails clearly', () => {
  const curve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-0.8, 0, 0),
    new THREE.Vector3(0, 0.4, 0),
    new THREE.Vector3(0.8, 0, 0),
  ])
  const geometry = new THREE.BoxGeometry(0.5, 0.18, 0.18)
  const material = new THREE.MeshBasicMaterial({ color: 0xff5533 })
  const flow = new Flow(new THREE.Mesh(geometry, material))
  flow.updateCurve(0, curve)
  const instancedFlow = new InstancedFlow(2, 1, geometry, material)
  instancedFlow.updateCurve(0, curve)
  instancedFlow.moveIndividualAlongCurve(1, 0.3)

  const camera = makeCamera()
  try {
    for (const [label, object3D] of [
      ['Flow', flow.object3D],
      ['InstancedFlow', instancedFlow.object3D],
    ]) {
      const scene = new THREE.Scene()
      scene.add(object3D)
      assert.throws(
        () => renderRgba(scene, camera, { width: 16, height: 16 }),
        /material\.onBeforeCompile customizations.*fragmentWgsl/i,
        `${label} should fail clearly on its shader-injection path`,
      )
    }
  } finally {
    geometry.dispose()
    material.dispose()
    for (const object3D of [flow.object3D, instancedFlow.object3D]) {
      object3D.traverse((child) => {
        if (Array.isArray(child.material)) {
          for (const childMaterial of child.material) childMaterial.dispose()
        } else {
          child.material?.dispose?.()
        }
      })
    }
  }
})
