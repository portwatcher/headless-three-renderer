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
import { getRenderer, makeCamera, renderRgba } from './scenes.test.part-002.mjs'
test('invalid transform matrix values fail clearly', () => {
  const camera = makeCamera()

  const scene = new THREE.Scene()
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial({ color: 0xffffff }))
  mesh.matrixAutoUpdate = false
  mesh.matrixWorldAutoUpdate = false
  mesh.matrixWorld.elements[12] = Number.NaN
  scene.add(mesh)

  assert.throws(
    () => getRenderer().render(scene, camera, { width: 32, height: 32, format: 'rgba' }),
    /mesh\.matrixWorld\.elements\[12\] must be a finite number/i,
  )
})

test('invalid geometry attribute values fail clearly', () => {
  const camera = makeCamera()

  const scene = new THREE.Scene()
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.75, -0.5, 0,
    Number.NaN, -0.5, 0,
    0, 0.75, 0,
  ]), 3))
  scene.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: 0xffffff })))

  assert.throws(
    () => getRenderer().render(scene, camera, { width: 32, height: 32, format: 'rgba' }),
    /geometry\.attributes\.position\[1\]\.x must be a finite number/i,
  )
})

test('invalid geometry attribute count values fail clearly', () => {
  const camera = makeCamera()

  const containerScene = new THREE.Scene()
  const containerGeometry = new THREE.BufferGeometry()
  containerGeometry.attributes = 'attributes'
  containerScene.add(new THREE.Mesh(containerGeometry, new THREE.MeshBasicMaterial({ color: 0xffffff })))
  assert.throws(
    () => renderRgba(containerScene, camera, { width: 32, height: 32 }),
    /geometry\.attributes must be an object/i,
  )

  const positionScene = new THREE.Scene()
  const positionGeometry = new THREE.BufferGeometry()
  positionGeometry.setAttribute('position', {
    count: '3',
    itemSize: 3,
    array: new Float32Array([
      -0.75, -0.5, 0,
      0.75, -0.5, 0,
      0, 0.75, 0,
    ]),
  })
  positionScene.add(new THREE.Mesh(positionGeometry, new THREE.MeshBasicMaterial({ color: 0xffffff })))
  assert.throws(
    () => renderRgba(positionScene, camera, { width: 32, height: 32 }),
    /geometry\.attributes\.position\.count must be a non-negative integer/i,
  )

  const indexScene = new THREE.Scene()
  const indexGeometry = new THREE.BufferGeometry()
  indexGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.75, -0.5, 0,
    0.75, -0.5, 0,
    0, 0.75, 0,
  ]), 3))
  indexGeometry.index = {
    count: Number.NaN,
    itemSize: 1,
    array: new Uint16Array([0, 1, 2]),
  }
  indexScene.add(new THREE.Mesh(indexGeometry, new THREE.MeshBasicMaterial({ color: 0xffffff })))
  assert.throws(
    () => renderRgba(indexScene, camera, { width: 32, height: 32 }),
    /geometry\.index\.count must be a non-negative integer/i,
  )
})

test('malformed geometry bounding spheres fail clearly', () => {
  const camera = makeCamera()
  const cases = [
    ['container', 'sphere', /geometry\.boundingSphere must be a THREE\.Sphere-like object/i],
    ['center', { center: { x: Number.NaN, y: 0, z: 0 }, radius: 1 }, /geometry\.boundingSphere\.center must be a finite Vector3-like value/i],
  ]

  for (const [name, boundingSphere, pattern] of cases) {
    const scene = new THREE.Scene()
    const geometry = new THREE.PlaneGeometry(1, 1)
    geometry.boundingSphere = boundingSphere
    scene.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: 0xffffff })))

    assert.throws(
      () => renderRgba(scene, camera, { width: 32, height: 32 }),
      pattern,
      `${name} should fail clearly`,
    )
  }
})

test('malformed object bounding spheres fail clearly', () => {
  const camera = makeCamera()
  const cases = [
    ['container', 'sphere', /object\.boundingSphere must be a THREE\.Sphere-like object/i],
    ['center', { center: { x: Number.NaN, y: 0, z: 0 }, radius: 1 }, /object\.boundingSphere\.center must be a finite Vector3-like value/i],
    ['radius', { center: { x: 0, y: 0, z: 0 }, radius: -1 }, /object\.boundingSphere\.radius must be non-negative/i],
  ]

  for (const [name, boundingSphere, pattern] of cases) {
    const scene = new THREE.Scene()
    const mesh = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1),
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    )
    mesh.boundingSphere = boundingSphere
    scene.add(mesh)

    assert.throws(
      () => renderRgba(scene, camera, { width: 32, height: 32 }),
      pattern,
      `${name} should fail clearly`,
    )
  }
})

test('invalid geometry attribute layout values fail clearly', () => {
  const camera = makeCamera()
  const values = new Float32Array([
    -0.75, -0.5, 0,
    0.75, -0.5, 0,
    0, 0.75, 0,
  ])
  const makeScene = (positionAttribute) => {
    const scene = new THREE.Scene()
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', positionAttribute)
    scene.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: 0xffffff })))
    return scene
  }

  const cases = [
    ['itemSize', { count: 3, itemSize: '3', array: values }, /geometry\.attributes\.position\.itemSize must be a positive integer/i],
    ['data container', { count: 3, itemSize: 3, data: 'packed' }, /geometry\.attributes\.position\.data must be an object/i],
    ['stride', { count: 3, itemSize: 3, data: { array: values, stride: '3' } }, /geometry\.attributes\.position\.data\.stride must be a positive integer/i],
    ['offset', { count: 3, itemSize: 3, array: values, offset: -1 }, /geometry\.attributes\.position\.offset must be a non-negative integer/i],
    ['normalized', { count: 3, itemSize: 3, array: values, normalized: 'yes' }, /geometry\.attributes\.position\.normalized must be a boolean/i],
  ]

  for (const [name, positionAttribute, pattern] of cases) {
    assert.throws(
      () => renderRgba(makeScene(positionAttribute), camera, { width: 32, height: 32 }),
      pattern,
      name,
    )
  }
})

test('invalid geometry index values fail clearly', () => {
  const camera = makeCamera()
  const makeScene = (indexAttribute) => {
    const scene = new THREE.Scene()
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
      -0.75, -0.5, 0,
      0.75, -0.5, 0,
      0, 0.75, 0,
    ]), 3))
    geometry.index = indexAttribute
    scene.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: 0xffffff })))
    return scene
  }

  const cases = [
    ['finite', new THREE.BufferAttribute(new Float32Array([0, Number.NaN, 2]), 1), /geometry\.index\[1\]\.x must be a finite number/i],
    ['negative', new THREE.BufferAttribute(new Float32Array([0, -1, 2]), 1), /geometry\.index\[1\]\.x must be a non-negative integer/i],
    ['fractional', new THREE.BufferAttribute(new Float32Array([0, 1.5, 2]), 1), /geometry\.index\[1\]\.x must be a non-negative integer/i],
    ['out of range', new THREE.BufferAttribute(new Uint16Array([0, 1, 3]), 1), /geometry\.index\[2\]\.x must reference a vertex below geometry\.attributes\.position\.count \(3\)/i],
  ]

  for (const [name, indexAttribute, pattern] of cases) {
    assert.throws(
      () => renderRgba(makeScene(indexAttribute), camera, { width: 32, height: 32 }),
      pattern,
      name,
    )
  }
})

test('invalid geometry group values fail clearly', () => {
  const camera = makeCamera()
  const makeScene = (groups) => {
    const scene = new THREE.Scene()
    const geometry = new THREE.PlaneGeometry(1, 1)
    geometry.groups = groups
    scene.add(new THREE.Mesh(geometry, [
      new THREE.MeshBasicMaterial({ color: 0xff0000 }),
      new THREE.MeshBasicMaterial({ color: 0x0000ff }),
    ]))
    return scene
  }

  const cases = [
    ['container', 'groups', /geometry\.groups must be an array/i],
    ['group object', [null], /geometry\.groups\[0\] must be an object/i],
    ['start', [{ start: '0', count: 6, materialIndex: 0 }], /geometry\.groups\[0\]\.start must be a non-negative integer/i],
    ['count', [{ start: 0, count: Number.NaN, materialIndex: 0 }], /geometry\.groups\[0\]\.count must be a non-negative integer/i],
    ['materialIndex', [{ start: 0, count: 6, materialIndex: -1 }], /geometry\.groups\[0\]\.materialIndex must be a non-negative integer/i],
  ]

  for (const [name, groups, pattern] of cases) {
    assert.throws(
      () => renderRgba(makeScene(groups), camera, { width: 32, height: 32 }),
      pattern,
      name,
    )
  }
})

test('invalid geometry drawRange values fail clearly', () => {
  const camera = makeCamera()
  const makeScene = (drawRange) => {
    const scene = new THREE.Scene()
    const geometry = new THREE.PlaneGeometry(1, 1)
    geometry.drawRange = drawRange
    scene.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: 0xffffff })))
    return scene
  }

  const cases = [
    ['container', 'range', /geometry\.drawRange must be an object/i],
    ['start', { start: '0', count: 6 }, /geometry\.drawRange\.start must be a non-negative integer/i],
    ['negative start', { start: -1, count: 6 }, /geometry\.drawRange\.start must be a non-negative integer/i],
    ['count', { start: 0, count: Number.NaN }, /geometry\.drawRange\.count must be a non-negative integer/i],
    ['fractional count', { start: 0, count: 1.5 }, /geometry\.drawRange\.count must be a non-negative integer/i],
  ]

  for (const [name, drawRange, pattern] of cases) {
    assert.throws(
      () => renderRgba(makeScene(drawRange), camera, { width: 32, height: 32 }),
      pattern,
      name,
    )
  }
})

test('malformed object children containers fail clearly', () => {
  const scene = new THREE.Scene()
  scene.children = 'children'
  assert.throws(
    () => renderRgba(scene, makeCamera(), { width: 32, height: 32 }),
    /object\.children must be an array/i,
  )

  const childEntryScene = new THREE.Scene()
  childEntryScene.children = [null]
  assert.throws(
    () => renderRgba(childEntryScene, makeCamera(), { width: 32, height: 32 }),
    /object\.children\[0\] must be an object/i,
  )
})
