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
import { constantUvPlane, makeCamera, renderRgba, rgbaTexture, setConstantUvAttribute } from './scenes.test.part-002.mjs'
test('invalid instanced attributes fail clearly', () => {
  const camera = makeCamera()
  const box = new THREE.BoxGeometry(0.5, 0.5, 0.5)
  const material = new THREE.MeshBasicMaterial({ color: 0xffffff })

  const matrixCountScene = new THREE.Scene()
  const matrixCountMesh = new THREE.InstancedMesh(box, material, 1)
  matrixCountMesh.instanceMatrix.count = 'many'
  matrixCountScene.add(matrixCountMesh)
  assert.throws(
    () => renderRgba(matrixCountScene, camera, { width: 64, height: 64 }),
    /InstancedMesh\.instanceMatrix\.count must be a non-negative integer/i,
  )

  const matrixValueScene = new THREE.Scene()
  const matrixValueMesh = new THREE.InstancedMesh(box, material, 1)
  matrixValueMesh.instanceMatrix.array[0] = Number.NaN
  matrixValueScene.add(matrixValueMesh)
  assert.throws(
    () => renderRgba(matrixValueScene, camera, { width: 64, height: 64 }),
    /InstancedMesh\.instanceMatrix\[0\]\.x must be a finite number/i,
  )

  const instanceColorScene = new THREE.Scene()
  const instanceColorMesh = new THREE.InstancedMesh(box, material, 1)
  instanceColorMesh.setColorAt(0, new THREE.Color(1, 1, 1))
  instanceColorMesh.instanceColor.count = 'many'
  instanceColorScene.add(instanceColorMesh)
  assert.throws(
    () => renderRgba(instanceColorScene, camera, { width: 64, height: 64 }),
    /InstancedMesh\.instanceColor\.count must be a non-negative integer/i,
  )

  const base = new THREE.PlaneGeometry(0.85, 0.85)
  const offsetGeometry = new THREE.InstancedBufferGeometry()
  offsetGeometry.index = base.index
  offsetGeometry.setAttribute('position', base.getAttribute('position'))
  offsetGeometry.setAttribute('uv', base.getAttribute('uv'))
  offsetGeometry.instanceCount = 1
  const instanceOffset = new THREE.InstancedBufferAttribute(new Float32Array([Number.NaN, 0, 0]), 3)
  offsetGeometry.setAttribute('instanceOffset', instanceOffset)
  const offsetScene = new THREE.Scene()
  offsetScene.add(new THREE.Mesh(offsetGeometry, material))
  assert.throws(
    () => renderRgba(offsetScene, camera, { width: 64, height: 64 }),
    /geometry\.attributes\.instanceOffset\[0\]\.x must be a finite number/i,
  )

  const scaleGeometry = new THREE.InstancedBufferGeometry()
  scaleGeometry.index = base.index
  scaleGeometry.setAttribute('position', base.getAttribute('position'))
  scaleGeometry.setAttribute('uv', base.getAttribute('uv'))
  scaleGeometry.instanceCount = 1
  scaleGeometry.setAttribute('instanceScale', new THREE.InstancedBufferAttribute(new Float32Array([Number.NaN]), 1))
  const scaleScene = new THREE.Scene()
  scaleScene.add(new THREE.Mesh(scaleGeometry, material))
  assert.throws(
    () => renderRgba(scaleScene, camera, { width: 64, height: 64 }),
    /geometry\.attributes\.instanceScale\[0\]\.x must be a finite number/i,
  )

  const colorCountGeometry = new THREE.InstancedBufferGeometry()
  colorCountGeometry.index = base.index
  colorCountGeometry.setAttribute('position', base.getAttribute('position'))
  colorCountGeometry.setAttribute('uv', base.getAttribute('uv'))
  colorCountGeometry.instanceCount = 1
  const instanceColor = new THREE.InstancedBufferAttribute(new Float32Array([1, 0, 0]), 3)
  instanceColor.count = 'many'
  colorCountGeometry.setAttribute('color', instanceColor)
  const colorCountScene = new THREE.Scene()
  colorCountScene.add(new THREE.Mesh(colorCountGeometry, new THREE.MeshBasicMaterial({ color: 0xffffff, vertexColors: true })))
  assert.throws(
    () => renderRgba(colorCountScene, camera, { width: 64, height: 64 }),
    /geometry\.attributes\.color\.count must be a non-negative integer/i,
  )

  const colorValueGeometry = new THREE.InstancedBufferGeometry()
  colorValueGeometry.index = base.index
  colorValueGeometry.setAttribute('position', base.getAttribute('position'))
  colorValueGeometry.setAttribute('uv', base.getAttribute('uv'))
  colorValueGeometry.instanceCount = 1
  colorValueGeometry.setAttribute('color', new THREE.InstancedBufferAttribute(new Float32Array([Number.NaN, 0, 0]), 3))
  const colorValueScene = new THREE.Scene()
  colorValueScene.add(new THREE.Mesh(colorValueGeometry, new THREE.MeshBasicMaterial({ color: 0xffffff, vertexColors: true })))
  assert.throws(
    () => renderRgba(colorValueScene, camera, { width: 64, height: 64 }),
    /geometry\.attributes\.color\[0\]\.x must be a finite number/i,
  )

  const uvCountGeometry = new THREE.InstancedBufferGeometry()
  uvCountGeometry.index = base.index
  uvCountGeometry.setAttribute('position', base.getAttribute('position'))
  const uvCount = new THREE.InstancedBufferAttribute(new Float32Array([0.5, 0.5]), 2)
  uvCount.count = 'many'
  uvCountGeometry.setAttribute('uv', uvCount)
  const uvCountScene = new THREE.Scene()
  uvCountScene.add(new THREE.Mesh(uvCountGeometry, material))
  assert.throws(
    () => renderRgba(uvCountScene, camera, { width: 64, height: 64 }),
    /geometry\.attributes\.uv\.count must be a non-negative integer/i,
  )

  const uvValueGeometry = new THREE.InstancedBufferGeometry()
  uvValueGeometry.index = base.index
  uvValueGeometry.setAttribute('position', base.getAttribute('position'))
  uvValueGeometry.instanceCount = 1
  uvValueGeometry.setAttribute('uv', new THREE.InstancedBufferAttribute(new Float32Array([Number.NaN, 0.5]), 2))
  const uvValueScene = new THREE.Scene()
  uvValueScene.add(new THREE.Mesh(uvValueGeometry, material))
  assert.throws(
    () => renderRgba(uvValueScene, camera, { width: 64, height: 64 }),
    /geometry\.attributes\.uv\[0\]\.x must be a finite number/i,
  )

  const uvRepeatGeometry = new THREE.InstancedBufferGeometry()
  uvRepeatGeometry.index = base.index
  uvRepeatGeometry.setAttribute('position', base.getAttribute('position'))
  const uvRepeat = new THREE.InstancedBufferAttribute(new Float32Array([0.5, 0.5]), 2)
  uvRepeat.meshPerAttribute = 0
  uvRepeatGeometry.setAttribute('uv', uvRepeat)
  const uvRepeatScene = new THREE.Scene()
  uvRepeatScene.add(new THREE.Mesh(uvRepeatGeometry, material))
  assert.throws(
    () => renderRgba(uvRepeatScene, camera, { width: 64, height: 64 }),
    /geometry\.attributes\.uv\.meshPerAttribute must be a positive finite number/i,
  )
})

test('invalid morph target influence values fail clearly', () => {
  function sceneWithInfluence(influence) {
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
      -0.75, -0.5, 0,
      0.75, -0.5, 0,
      0, 0.75, 0,
    ]), 3))
    geometry.setAttribute('normal', new THREE.BufferAttribute(new Float32Array([
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
    ]), 3))
    geometry.morphTargetsRelative = true
    geometry.morphAttributes.position = [new THREE.BufferAttribute(new Float32Array([
      0, 0.25, 0,
      0, 0.25, 0,
      0, 0.25, 0,
    ]), 3)]

    const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: 0xffffff }))
    mesh.morphTargetInfluences = [influence]

    const scene = new THREE.Scene()
    scene.add(mesh)
    return scene
  }

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  for (const influence of [Number.NaN, Number.POSITIVE_INFINITY, 'active']) {
    assert.throws(
      () => renderRgba(sceneWithInfluence(influence), camera, { width: 64, height: 64 }),
      /morphTargetInfluences\[0\] must be a finite number/i,
    )
  }

  const invalidRelative = sceneWithInfluence(1)
  invalidRelative.children[0].geometry.morphTargetsRelative = 'yes'
  assert.throws(
    () => renderRgba(invalidRelative, camera, { width: 64, height: 64 }),
    /geometry\.morphTargetsRelative must be a boolean/i,
  )

  const invalidAttributes = sceneWithInfluence(1)
  invalidAttributes.children[0].geometry.morphAttributes = 'morphs'
  assert.throws(
    () => renderRgba(invalidAttributes, camera, { width: 64, height: 64 }),
    /geometry\.morphAttributes must be an object/i,
  )

  const invalidPositionArray = sceneWithInfluence(1)
  invalidPositionArray.children[0].geometry.morphAttributes.position = 'positions'
  assert.throws(
    () => renderRgba(invalidPositionArray, camera, { width: 64, height: 64 }),
    /geometry\.morphAttributes\.position must be an array/i,
  )

  const invalidPositionEntry = sceneWithInfluence(1)
  invalidPositionEntry.children[0].geometry.morphAttributes.position = ['position']
  assert.throws(
    () => renderRgba(invalidPositionEntry, camera, { width: 64, height: 64 }),
    /geometry\.morphAttributes\.position\[0\] must be an attribute-like object/i,
  )
})

test('invalid skinning matrix values fail clearly', () => {
  function sceneWithSkinning(mutator) {
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
      -0.75, -0.5, 0,
      0.75, -0.5, 0,
      0, 0.75, 0,
    ]), 3))
    geometry.setAttribute('normal', new THREE.BufferAttribute(new Float32Array([
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
    ]), 3))
    geometry.setAttribute('skinIndex', new THREE.BufferAttribute(new Uint16Array([
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
    ]), 4))
    geometry.setAttribute('skinWeight', new THREE.BufferAttribute(new Float32Array([
      1, 0, 0, 0,
      1, 0, 0, 0,
      1, 0, 0, 0,
    ]), 4))

    const mesh = new THREE.SkinnedMesh(geometry, new THREE.MeshBasicMaterial({ color: 0xffffff }))
    const bone = new THREE.Bone()
    const skeleton = new THREE.Skeleton([bone])
    mesh.add(bone)
    const scene = new THREE.Scene()
    scene.add(mesh)
    mesh.bind(skeleton)
    mutator({ mesh, bone, skeleton })
    scene.updateMatrixWorld = () => {}
    return scene
  }

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const cases = [
    ['skeleton container', ({ mesh }) => {
      mesh.skeleton = 'skeleton'
    }, /mesh\.skeleton must be an object/i],
    ['bones container', ({ skeleton }) => {
      skeleton.bones = 'bones'
    }, /skeleton\.bones must be an array/i],
    ['bone inverse container', ({ skeleton }) => {
      skeleton.boneInverses = 'inverses'
    }, /skeleton\.boneInverses must be an array/i],
    ['bone entry', ({ skeleton }) => {
      skeleton.bones[0] = 'bone'
    }, /skeleton\.bones\[0\] must be an object/i],
    ['bone world matrix', ({ bone }) => {
      bone.matrixWorld.elements[13] = Number.NaN
    }, /skeleton\.bones\[0\]\.matrixWorld\.elements\[13\] must be a finite number/i],
    ['bone inverse matrix', ({ skeleton }) => {
      skeleton.boneInverses[0].elements[0] = Number.NaN
    }, /skeleton\.boneInverses\[0\]\.elements\[0\] must be a finite number/i],
    ['bind matrix', ({ mesh }) => {
      mesh.bindMatrix.elements[5] = Number.POSITIVE_INFINITY
    }, /mesh\.bindMatrix\.elements\[5\] must be a finite number/i],
    ['bind inverse matrix', ({ mesh }) => {
      mesh.bindMatrixInverse.elements[10] = Number.NEGATIVE_INFINITY
    }, /mesh\.bindMatrixInverse\.elements\[10\] must be a finite number/i],
  ]

  for (const [name, mutate, pattern] of cases) {
    assert.throws(
      () => renderRgba(sceneWithSkinning(mutate), camera, { width: 64, height: 64 }),
      pattern,
      `${name} should fail clearly`,
    )
  }
})

test('aoMap samples selected uv1-uv3 texture channels', () => {
  function renderWithChannel(channel) {
    const aoMap = rgbaTexture([
      255, 255, 255, 255,
      255, 255, 255, 255,
      0, 0, 0, 255,
      0, 0, 0, 255,
    ], 4, 1)
    aoMap.channel = channel

    const geometry = constantUvPlane(0.125, 0.5)
    if (channel > 0) {
      setConstantUvAttribute(geometry, `uv${channel}`, 0.875, 0.5)
    }

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      geometry,
      new THREE.MeshBasicMaterial({ color: 0xffffff, aoMap, aoMapIntensity: 1 }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const primary = renderWithChannel(0)
  for (const channel of [1, 2, 3]) {
    const secondary = renderWithChannel(channel)
    assert.ok(primary.r > secondary.r + 100, `aoMap channel=0 should sample bright primary UVs over channel=${channel} (${primary.r} vs ${secondary.r})`)
    assert.ok(secondary.r < 20, `aoMap channel=${channel} should darken the plane through uv${channel} (${secondary.r})`)
  }
})
