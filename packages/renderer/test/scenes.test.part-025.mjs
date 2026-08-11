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
import { test, threeRendererFile, threeRendererSource } from './scenes.test.part-001.mjs'
import { countRegionPixels, renderRgba } from './scenes.test.part-002.mjs'
test('examples BoxLineGeometry and TeapotGeometry render generated geometry paths', () => {
  const lineGeometry = new BoxLineGeometry(1, 1, 1, 2, 2, 2)
  const lineMaterial = new THREE.LineBasicMaterial({ color: 0xff00ff })
  const line = new THREE.LineSegments(lineGeometry, lineMaterial)
  line.position.x = -0.7

  const teapotGeometry = new TeapotGeometry(0.4, 4, true, true, true, true, true)
  const teapotMaterial = new THREE.MeshBasicMaterial({ color: 0x44ff88, side: THREE.DoubleSide })
  const teapot = new THREE.Mesh(teapotGeometry, teapotMaterial)
  teapot.position.x = 0.7

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  scene.add(line, teapot)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  try {
    const rgba = renderRgba(scene, camera, { width: 96, height: 96 })
    assert.equal(lineGeometry.isBufferGeometry, true)
    assert.ok(lineGeometry.getAttribute('position')?.count >= 72)
    assert.equal(teapotGeometry.isBufferGeometry, true)
    assert.ok(teapotGeometry.getAttribute('position')?.count > 0)
    assert.ok(teapotGeometry.getAttribute('normal'), 'TeapotGeometry should generate normals')
    assert.ok(teapotGeometry.getAttribute('uv'), 'TeapotGeometry should generate UVs')
    assert.ok(
      nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.12,
      'BoxLineGeometry and TeapotGeometry should produce visible pixels',
    )
    assert.ok(
      countRegionPixels(rgba, 96, 96, 0, 0, 96, 96, (r, g, b) => r > 100 && b > 100 && g < 100) > 150,
      'BoxLineGeometry should render visible magenta line pixels',
    )
    assert.ok(
      countRegionPixels(rgba, 96, 96, 0, 0, 96, 96, (r, g, b) => g > 120 && g > r + 30 && g > b + 20) > 150,
      'TeapotGeometry should render visible green mesh pixels',
    )
  } finally {
    lineGeometry.dispose()
    lineMaterial.dispose()
    teapotGeometry.dispose()
    teapotMaterial.dispose()
  }
})

test('TextGeometry renders parsed example fonts through built-in mesh materials', () => {
  const font = new FontLoader().parse(JSON.parse(threeRendererSource('examples/fonts/helvetiker_regular.typeface.json')))
  const geometry = new TextGeometry('Node', {
    font,
    size: 0.42,
    depth: 0.08,
    curveSegments: 4,
    bevelEnabled: false,
  })
  geometry.center()
  const material = new THREE.MeshBasicMaterial({ color: 0xffaa33, side: THREE.DoubleSide })
  const mesh = new THREE.Mesh(geometry, material)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  scene.add(mesh)

  const camera = new THREE.PerspectiveCamera(45, 1.5, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  try {
    const width = 96
    const height = 64
    const rgba = renderRgba(scene, camera, { width, height })
    assert.equal(geometry.isBufferGeometry, true)
    assert.ok(geometry.getAttribute('position')?.count > 0)
    assert.ok(geometry.getAttribute('normal'), 'TextGeometry should generate normals')
    assert.ok(geometry.getAttribute('uv'), 'TextGeometry should generate UVs')
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => {
        return r > 100 && g > 50 && r > g && b < 90
      }) > 100,
      'TextGeometry should render visible text-colored pixels',
    )
  } finally {
    geometry.dispose()
    material.dispose()
  }
})

test('TTFLoader parses example fonts into renderable TextGeometry', () => {
  const buffer = threeRendererFile('examples/fonts/ttf/kenpixel.ttf')
  const arrayBuffer = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength)
  const typeface = new TTFLoader().parse(arrayBuffer)
  const font = new FontLoader().parse(typeface)
  const geometry = new TextGeometry('HUD', {
    font,
    size: 0.42,
    depth: 0.04,
    curveSegments: 1,
    bevelEnabled: false,
  })
  geometry.center()
  const material = new THREE.MeshBasicMaterial({ color: 0x66ddff, side: THREE.DoubleSide })
  const mesh = new THREE.Mesh(geometry, material)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  scene.add(mesh)

  const camera = new THREE.PerspectiveCamera(45, 1.5, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  try {
    const width = 96
    const height = 64
    const rgba = renderRgba(scene, camera, { width, height })
    assert.match(typeface.familyName, /KenPixel/i)
    assert.equal(geometry.isBufferGeometry, true)
    assert.ok(geometry.getAttribute('position')?.count > 0)
    assert.ok(geometry.getAttribute('normal'), 'TTF-derived TextGeometry should generate normals')
    assert.ok(geometry.getAttribute('uv'), 'TTF-derived TextGeometry should generate UVs')
    assert.ok(
      nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.02,
      'TTF-derived TextGeometry should render visible font geometry',
    )
  } finally {
    geometry.dispose()
    material.dispose()
  }
})

test('examples XYZLoader GCodeLoader and PDBLoader parse renderable geometry paths', () => {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 20)
  camera.position.set(0, 0, 6)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  const xyzGeometry = new XYZLoader().parse(`
    # XYZRGB point cloud
    -0.8 0 0 255 0 0
    0 0.8 0 0 255 0
    0.8 0 0 0 0 255
  `)
  const xyzMaterial = new THREE.PointsMaterial({ size: 4, sizeAttenuation: false, vertexColors: true })
  const xyzPoints = new THREE.Points(xyzGeometry, xyzMaterial)
  const xyzScene = new THREE.Scene()
  xyzScene.background = new THREE.Color(0x000000)
  xyzScene.add(xyzPoints)

  const gcodeLoader = new GCodeLoader()
  gcodeLoader.splitLayer = true
  const gcodeGroup = gcodeLoader.parse([
    'G90',
    'G1 X0 Y0 Z0 F1500',
    'G1 X1 Y0 Z0 E0.2',
    'G1 X1 Y1 Z0 E0.4',
    'G1 X0 Y1 Z0',
    'G91',
    'G1 X0 Y0 Z1',
    'G1 X-1 Y0 Z0 E0.6',
  ].join('\n'))
  gcodeGroup.scale.setScalar(1.2)
  gcodeGroup.position.set(-0.5, -0.5, 0)
  const gcodeScene = new THREE.Scene()
  gcodeScene.background = new THREE.Color(0x000000)
  gcodeScene.add(gcodeGroup)

  const atomLine = (serial, name, residue, chain, sequence, x, y, z, element) => {
    return `HETATM${String(serial).padStart(5)} ${name.padEnd(4)} ${residue.padStart(3)} ${chain}${String(sequence).padStart(4)}    ${x.toFixed(3).padStart(7)} ${y.toFixed(3).padStart(7)} ${z.toFixed(3).padStart(7)}  1.00 20.00           ${element.padStart(2)}`
  }
  const pdb = new PDBLoader().parse([
    atomLine(1, 'C', 'MOL', 'A', 1, -0.6, 0, 0, 'C'),
    atomLine(2, 'O', 'MOL', 'A', 1, 0.6, 0, 0, 'O'),
    'CONECT    1    2',
    'END',
  ].join('\n'))
  const pdbAtomMaterial = new THREE.PointsMaterial({ size: 6, sizeAttenuation: false, vertexColors: true })
  const pdbBondMaterial = new THREE.LineBasicMaterial({ color: 0xffffff })
  const pdbAtoms = new THREE.Points(pdb.geometryAtoms, pdbAtomMaterial)
  const pdbBonds = new THREE.LineSegments(pdb.geometryBonds, pdbBondMaterial)
  const pdbScene = new THREE.Scene()
  pdbScene.background = new THREE.Color(0x000000)
  pdbScene.add(pdbBonds, pdbAtoms)

  try {
    const xyzRgba = renderRgba(xyzScene, camera, { width: 64, height: 64 })
    assert.equal(xyzGeometry.isBufferGeometry, true)
    assert.equal(xyzGeometry.getAttribute('position').count, 3)
    assert.equal(xyzGeometry.getAttribute('color').count, 3)
    assert.ok(
      nonBackgroundRatio(xyzRgba, [0, 0, 0], 3) > 0.008,
      'XYZLoader point cloud output should render visible colored points',
    )

    const gcodeRgba = renderRgba(gcodeScene, camera, { width: 64, height: 64 })
    assert.equal(gcodeGroup.name, 'gcode')
    assert.equal(gcodeGroup.children.length, 4)
    assert.ok(gcodeGroup.children.some((child) => child.material.name === 'extruded'))
    assert.ok(gcodeGroup.children.some((child) => child.material.name === 'path'))
    assert.ok(
      nonBackgroundRatio(gcodeRgba, [0, 0, 0], 3) > 0.008,
      'GCodeLoader line output should render visible tool paths',
    )

    const pdbRgba = renderRgba(pdbScene, camera, { width: 64, height: 64 })
    assert.equal(pdb.geometryAtoms.getAttribute('position').count, 2)
    assert.equal(pdb.geometryAtoms.getAttribute('color').count, 2)
    assert.equal(pdb.geometryBonds.getAttribute('position').count, 2)
    assert.deepEqual(pdb.json.atoms.map((atom) => atom[4]), ['C', 'O'])
    assert.ok(
      nonBackgroundRatio(pdbRgba, [0, 0, 0], 3) > 0.003,
      'PDBLoader atom and bond geometry should render visible points and bonds',
    )
  } finally {
    xyzGeometry.dispose()
    xyzMaterial.dispose()
    for (const child of gcodeGroup.children) child.geometry.dispose()
    for (const material of new Set(gcodeGroup.children.map((child) => child.material))) material.dispose()
    pdb.geometryAtoms.dispose()
    pdb.geometryBonds.dispose()
    pdbAtomMaterial.dispose()
    pdbBondMaterial.dispose()
  }
})
