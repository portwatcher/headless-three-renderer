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
import { renderRgba } from './scenes.test.part-002.mjs'
test('examples OBJLoader STLLoader and PLYLoader parse renderable mesh geometry paths', () => {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  const objGroup = new OBJLoader().parse([
    'o tri',
    'v -0.6 -0.4 0 1 0 0',
    'v 0.6 -0.4 0 0 1 0',
    'v 0 0.6 0 0 0 1',
    'vt 0 0',
    'vt 1 0',
    'vt 0.5 1',
    'vn 0 0 1',
    'usemtl none',
    'f 1/1/1 2/2/1 3/3/1',
  ].join('\n'))
  const objScene = new THREE.Scene()
  objScene.background = new THREE.Color(0x000000)
  objScene.add(new THREE.AmbientLight(0xffffff, 1), objGroup)

  const stlGeometry = new STLLoader().parse(new TextEncoder().encode([
    'solid tri',
    'facet normal 0 0 1',
    'outer loop',
    'vertex -0.6 -0.4 0',
    'vertex 0.6 -0.4 0',
    'vertex 0 0.6 0',
    'endloop',
    'endfacet',
    'endsolid tri',
  ].join('\n')).buffer)
  const stlMaterial = new THREE.MeshBasicMaterial({ color: 0xff8844, side: THREE.DoubleSide })
  const stlMesh = new THREE.Mesh(stlGeometry, stlMaterial)
  const stlScene = new THREE.Scene()
  stlScene.background = new THREE.Color(0x000000)
  stlScene.add(stlMesh)

  const plyGeometry = new PLYLoader().parse([
    'ply',
    'format ascii 1.0',
    'element vertex 3',
    'property float x',
    'property float y',
    'property float z',
    'property uchar red',
    'property uchar green',
    'property uchar blue',
    'element face 1',
    'property list uchar int vertex_indices',
    'end_header',
    '-0.6 -0.4 0 255 0 0',
    '0.6 -0.4 0 0 255 0',
    '0 0.6 0 0 0 255',
    '3 0 1 2',
  ].join('\n'))
  const plyMaterial = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide, vertexColors: true })
  const plyMesh = new THREE.Mesh(plyGeometry, plyMaterial)
  const plyScene = new THREE.Scene()
  plyScene.background = new THREE.Color(0x000000)
  plyScene.add(plyMesh)

  try {
    const objMesh = objGroup.children[0]
    assert.equal(objGroup.children.length, 1)
    assert.equal(objMesh.isMesh, true)
    assert.equal(objMesh.geometry.getAttribute('position').count, 3)
    assert.equal(objMesh.geometry.getAttribute('normal').count, 3)
    assert.equal(objMesh.geometry.getAttribute('uv').count, 3)
    assert.equal(objMesh.geometry.getAttribute('color').count, 3)
    assert.equal(objMesh.material.name, 'none')
    assert.ok(
      nonBackgroundRatio(renderRgba(objScene, camera, { width: 64, height: 64 }), [0, 0, 0], 3) > 0.08,
      'OBJLoader mesh output should render visible vertex-colored geometry',
    )

    assert.equal(stlGeometry.getAttribute('position').count, 3)
    assert.equal(stlGeometry.getAttribute('normal').count, 3)
    assert.equal(stlGeometry.groups.length, 1)
    assert.ok(
      nonBackgroundRatio(renderRgba(stlScene, camera, { width: 64, height: 64 }), [0, 0, 0], 3) > 0.08,
      'STLLoader mesh output should render visible triangle geometry',
    )

    assert.equal(plyGeometry.getAttribute('position').count, 3)
    assert.equal(plyGeometry.getAttribute('color').count, 3)
    assert.equal(plyGeometry.index.count, 3)
    assert.ok(
      nonBackgroundRatio(renderRgba(plyScene, camera, { width: 64, height: 64 }), [0, 0, 0], 3) > 0.08,
      'PLYLoader mesh output should render visible indexed vertex-colored geometry',
    )
  } finally {
    for (const child of objGroup.children) {
      child.geometry?.dispose?.()
      if (Array.isArray(child.material)) {
        for (const material of child.material) material.dispose()
      } else {
        child.material?.dispose?.()
      }
    }
    stlGeometry.dispose()
    stlMaterial.dispose()
    plyGeometry.dispose()
    plyMaterial.dispose()
  }
})

test('examples MTLLoader parses material libraries for renderable mesh paths', () => {
  const materialCreator = new MTLLoader().setMaterialOptions({ normalizeRGB: true }).parse([
    'newmtl panel',
    'Kd 64 200 96',
    'Ks 0 0 0',
    'Ns 16',
    'd 0.9',
  ].join('\n'), '')
  materialCreator.preload()
  const materials = materialCreator.getAsArray()
  const material = materialCreator.create('panel')
  const geometry = new THREE.PlaneGeometry(1, 1)
  const mesh = new THREE.Mesh(geometry, material)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  scene.add(new THREE.AmbientLight(0xffffff, 1), mesh)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  try {
    const rgba = renderRgba(scene, camera, { width: 64, height: 64 })
    assert.equal(material.type, 'MeshPhongMaterial')
    assert.equal(material.name, 'panel')
    assert.equal(materialCreator.getIndex('panel'), 0)
    assert.equal(materials[0], material)
    assert.equal(material.transparent, true)
    assert.equal(material.opacity, 0.9)
    assert.ok(material.color.g > material.color.r)
    assert.ok(
      nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.12,
      'MTLLoader-created MeshPhongMaterial should render visible mesh pixels',
    )
  } finally {
    geometry.dispose()
    material.dispose()
  }
})

test('examples VRMLLoader parses renderable scene mesh paths', () => {
  const scene = new VRMLLoader().parse([
    '#VRML V2.0 utf8',
    'Shape {',
    '  appearance Appearance { material Material { diffuseColor 0 1 0 } }',
    '  geometry Box { size 1 1 1 }',
    '}',
  ].join('\n'), '')
  scene.background = new THREE.Color(0x000000)
  scene.add(new THREE.AmbientLight(0xffffff, 1))

  const mesh = scene.children.find((child) => child.isMesh)
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  try {
    assert.equal(scene.isScene, true)
    assert.ok(mesh, 'VRMLLoader should create a mesh child')
    assert.equal(mesh.geometry.type, 'BoxGeometry')
    assert.equal(mesh.material.type, 'MeshPhongMaterial')
    assert.equal(mesh.material.color.getHexString(), '00ff00')
    assert.ok(
      nonBackgroundRatio(renderRgba(scene, camera, { width: 64, height: 64 }), [0, 0, 0], 3) > 0.12,
      'VRMLLoader scene output should render visible mesh pixels',
    )
  } finally {
    for (const child of scene.children) {
      child.geometry?.dispose?.()
      child.material?.dispose?.()
    }
  }
})

test('examples RGBELoader parses HDR buffers into renderable DataTexture payloads', () => {
  const header = new TextEncoder().encode([
    '#?RADIANCE',
    'FORMAT=32-bit_rle_rgbe',
    '-Y 1 +X 2',
    '',
  ].join('\n'))
  const buffer = new Uint8Array(header.length + 8)
  buffer.set(header)
  buffer.set([
    255, 0, 0, 128,
    0, 255, 0, 128,
  ], header.length)
  const texData = new RGBELoader().setDataType(THREE.FloatType).parse(buffer.buffer)
  const texture = new THREE.DataTexture(texData.data, texData.width, texData.height, THREE.RGBAFormat, texData.type)
  texture.colorSpace = THREE.LinearSRGBColorSpace
  texture.needsUpdate = true
  const geometry = new THREE.PlaneGeometry(1, 1)
  const material = new THREE.MeshBasicMaterial({ map: texture })
  const mesh = new THREE.Mesh(geometry, material)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  scene.add(mesh)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  try {
    assert.equal(texData.width, 2)
    assert.equal(texData.height, 1)
    assert.equal(texData.type, THREE.FloatType)
    assert.ok(texData.data instanceof Float32Array)
    assert.deepEqual(Array.from(texData.data), [
      1, 0, 0, 1,
      0, 1, 0, 1,
    ])
    assert.match(texData.header, /FORMAT=32-bit_rle_rgbe/)
    assert.ok(
      nonBackgroundRatio(renderRgba(scene, camera, { width: 64, height: 64 }), [0, 0, 0], 3) > 0.12,
      'RGBELoader-decoded DataTexture should render visible HDR texture pixels',
    )
  } finally {
    geometry.dispose()
    material.dispose()
    texture.dispose()
  }
})

test('examples TGALoader parses TGA buffers into renderable DataTexture payloads', () => {
  const header = new Uint8Array(18)
  header[2] = 2
  header[12] = 2
  header[14] = 1
  header[16] = 24
  header[17] = 0x20

  const buffer = new Uint8Array(18 + 6)
  buffer.set(header)
  buffer.set([
    0, 0, 255,
    0, 255, 0,
  ], 18)

  const texData = new TGALoader().parse(buffer.buffer)
  const texture = new THREE.DataTexture(texData.data, texData.width, texData.height, THREE.RGBAFormat)
  texture.flipY = texData.flipY
  texture.generateMipmaps = texData.generateMipmaps
  texture.minFilter = texData.minFilter
  texture.needsUpdate = true

  const geometry = new THREE.PlaneGeometry(1, 1)
  const material = new THREE.MeshBasicMaterial({ map: texture })
  const mesh = new THREE.Mesh(geometry, material)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  scene.add(mesh)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  try {
    assert.equal(texData.width, 2)
    assert.equal(texData.height, 1)
    assert.equal(texData.flipY, true)
    assert.equal(texData.generateMipmaps, true)
    assert.equal(texData.minFilter, THREE.LinearMipmapLinearFilter)
    assert.deepEqual(Array.from(texData.data), [
      255, 0, 0, 255,
      0, 255, 0, 255,
    ])
    assert.ok(
      nonBackgroundRatio(renderRgba(scene, camera, { width: 64, height: 64 }), [0, 0, 0], 3) > 0.12,
      'TGALoader-decoded DataTexture should render visible TGA texture pixels',
    )
  } finally {
    geometry.dispose()
    material.dispose()
    texture.dispose()
  }
})
