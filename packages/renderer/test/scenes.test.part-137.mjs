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
import { extractLights, test } from './scenes.test.part-001.mjs'
import { makeCamera, renderRgba } from './scenes.test.part-002.mjs'
test('non-square point-light shadow map sizes fail clearly', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const light = new THREE.PointLight(0xffffff, 1)
  light.position.set(2, 4, 2)
  light.castShadow = true
  light.shadow.mapSize.set(512, 256)
  scene.add(light)

  assert.throws(
    () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
    /non-square PointLight shadow map sizes.*not supported/i,
  )
})

test('invalid object and light shadow flag values fail clearly', () => {
  const camera = makeCamera()
  const objectCases = [
    ['mesh castShadow', () => {
      const scene = new THREE.Scene()
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial())
      mesh.castShadow = 'yes'
      scene.add(mesh)
      return scene
    }, /object\.castShadow must be a boolean/i],
    ['mesh receiveShadow', () => {
      const scene = new THREE.Scene()
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial())
      mesh.receiveShadow = 1
      scene.add(mesh)
      return scene
    }, /object\.receiveShadow must be a boolean/i],
    ['sprite castShadow', () => {
      const scene = new THREE.Scene()
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial())
      sprite.castShadow = 'yes'
      scene.add(sprite)
      return scene
    }, /object\.castShadow must be a boolean/i],
    ['sprite receiveShadow', () => {
      const scene = new THREE.Scene()
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial())
      sprite.receiveShadow = 'yes'
      scene.add(sprite)
      return scene
    }, /object\.receiveShadow must be a boolean/i],
    ['points castShadow', () => {
      const scene = new THREE.Scene()
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))
      const points = new THREE.Points(geometry, new THREE.PointsMaterial())
      points.castShadow = 1
      scene.add(points)
      return scene
    }, /object\.castShadow must be a boolean/i],
    ['line receiveShadow', () => {
      const scene = new THREE.Scene()
      const geometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-1, 0, 0),
        new THREE.Vector3(1, 0, 0),
      ])
      const line = new THREE.Line(geometry, new THREE.LineBasicMaterial())
      line.receiveShadow = 'yes'
      scene.add(line)
      return scene
    }, /object\.receiveShadow must be a boolean/i],
  ]

  for (const [name, makeScene, pattern] of objectCases) {
    assert.throws(
      () => renderRgba(makeScene(), camera, { width: 32, height: 32 }),
      pattern,
      name,
    )
  }

  const lightScene = new THREE.Scene()
  const light = new THREE.DirectionalLight(0xffffff, 1)
  light.castShadow = 'yes'
  lightScene.add(light)
  assert.throws(
    () => extractLights(lightScene),
    /light\.castShadow must be a boolean/i,
  )

  for (const [name, mutate, pattern] of [
    ['shadow.autoUpdate', (shadow) => {
      shadow.autoUpdate = 'yes'
    }, /light\.shadow\.autoUpdate must be a boolean/i],
    ['shadow.needsUpdate', (shadow) => {
      shadow.needsUpdate = 'yes'
    }, /light\.shadow\.needsUpdate must be a boolean/i],
  ]) {
    const scene = new THREE.Scene()
    const shadowLight = new THREE.DirectionalLight(0xffffff, 1)
    shadowLight.castShadow = true
    mutate(shadowLight.shadow)
    scene.add(shadowLight)
    assert.throws(
      () => extractLights(scene),
      pattern,
      name,
    )
  }
})

test('invalid shadow numeric values fail clearly', () => {
  const cases = [
    ['shadow container', (light) => {
      light.shadow = 'shadow'
    }, /light\.shadow must be an object/i],
    ['mapSize container', (light) => {
      light.shadow.mapSize = [512, 512]
    }, /light\.shadow\.mapSize must be an object/i],
    ['mapSize.x', (light) => {
      light.shadow.mapSize.x = 'wide'
    }, /light\.shadow\.mapSize\.x must be a finite number/i],
    ['mapSize.y', (light) => {
      light.shadow.mapSize.y = Number.NaN
    }, /light\.shadow\.mapSize\.y must be a finite number/i],
    ['mapSize.x zero', (light) => {
      light.shadow.mapSize.x = 0
    }, /light\.shadow\.mapSize\.x must be positive/i],
    ['mapSize.width', (light) => {
      light.shadow.mapSize = { width: 'wide', height: 512 }
    }, /light\.shadow\.mapSize\.width must be a finite number/i],
    ['mapSize.height', (light) => {
      light.shadow.mapSize = { width: 512, height: Number.NaN }
    }, /light\.shadow\.mapSize\.height must be a finite number/i],
    ['mapSize.height zero', (light) => {
      light.shadow.mapSize = { width: 512, height: 0 }
    }, /light\.shadow\.mapSize\.height must be positive/i],
    ['bias', (light) => {
      light.shadow.bias = 'biased'
    }, /light\.shadow\.bias must be a finite number/i],
    ['normalBias', (light) => {
      light.shadow.normalBias = Number.POSITIVE_INFINITY
    }, /light\.shadow\.normalBias must be a finite number/i],
    ['radius', (light) => {
      light.shadow.radius = Number.NEGATIVE_INFINITY
    }, /light\.shadow\.radius must be a finite number/i],
    ['radius negative', (light) => {
      light.shadow.radius = -1
    }, /light\.shadow\.radius must be non-negative/i],
    ['intensity', (light) => {
      light.shadow.intensity = 'dark'
    }, /light\.shadow\.intensity must be a finite number/i],
    ['intensity negative', (light) => {
      light.shadow.intensity = -1
    }, /light\.shadow\.intensity must be non-negative/i],
    ['blurSamples', (light) => {
      light.shadow.blurSamples = 'many'
    }, /light\.shadow\.blurSamples must be a finite number/i],
    ['blurSamples negative', (light) => {
      light.shadow.blurSamples = -1
    }, /light\.shadow\.blurSamples must be non-negative/i],
    ['map cache', (light) => {
      light.shadow.map = 'target'
    }, /light\.shadow\.map must be an object/i],
    ['mapPass cache', (light) => {
      light.shadow.mapPass = ['target']
    }, /light\.shadow\.mapPass must be an object/i],
    ['matrix container', (light) => {
      light.shadow.matrix = { elements: [1, 0, 0] }
    }, /light\.shadow\.matrix must be a THREE\.Matrix4/i],
    ['matrix element', (light) => {
      light.shadow.matrix.elements[0] = Number.NaN
    }, /light\.shadow\.matrix\.elements\[0\] must be a finite number/i],
    ['camera.left', (light) => {
      light.shadow.camera.left = 'left'
    }, /light\.shadow\.camera\.left must be a finite number/i],
    ['camera.right before left', (light) => {
      light.shadow.camera.left = 4
      light.shadow.camera.right = 4
    }, /light\.shadow\.camera\.right must be greater than light\.shadow\.camera\.left/i],
    ['camera.left beyond default right', (light) => {
      light.shadow.camera.left = 10
      delete light.shadow.camera.right
    }, /light\.shadow\.camera\.left must be less than the effective light\.shadow\.camera\.right/i],
    ['camera.top below bottom', (light) => {
      light.shadow.camera.top = -6
      light.shadow.camera.bottom = -6
    }, /light\.shadow\.camera\.top must be greater than light\.shadow\.camera\.bottom/i],
    ['camera.bottom beyond default top', (light) => {
      light.shadow.camera.bottom = 10
      delete light.shadow.camera.top
    }, /light\.shadow\.camera\.bottom must be less than the effective light\.shadow\.camera\.top/i],
    ['camera container', (light) => {
      light.shadow.camera = 'camera'
    }, /light\.shadow\.camera must be an object/i],
    ['camera.near', (light) => {
      light.shadow.camera.near = Number.NaN
    }, /light\.shadow\.camera\.near must be a finite number/i],
    ['camera.far', (light) => {
      light.shadow.camera.far = 'far'
    }, /light\.shadow\.camera\.far must be a finite number/i],
    ['camera.near negative', (light) => {
      light.shadow.camera.near = -0.1
    }, /light\.shadow\.camera\.near must be non-negative/i],
    ['camera.far zero', (light) => {
      light.shadow.camera.far = 0
    }, /light\.shadow\.camera\.far must be positive/i],
    ['camera.far before near', (light) => {
      light.shadow.camera.near = 10
      light.shadow.camera.far = 1
    }, /light\.shadow\.camera\.far must be greater than light\.shadow\.camera\.near/i],
    ['camera.near beyond default far', (light) => {
      light.shadow.camera.near = 600
      delete light.shadow.camera.far
    }, /light\.shadow\.camera\.near must be less than the effective light\.shadow\.camera\.far/i],
  ]

  for (const [name, mutate, pattern] of cases) {
    const scene = new THREE.Scene()
    const light = new THREE.DirectionalLight(0xffffff, 1)
    light.castShadow = true
    mutate(light)
    scene.add(light)
    assert.throws(
      () => extractLights(scene),
      pattern,
      `${name} should fail clearly`,
    )
  }

  for (const [name, makeLight] of [
    ['point near zero', () => new THREE.PointLight(0xffffff, 1)],
    ['spot near zero', () => new THREE.SpotLight(0xffffff, 1)],
  ]) {
    const scene = new THREE.Scene()
    const light = makeLight()
    light.castShadow = true
    light.shadow.camera.near = 0
    scene.add(light)
    assert.throws(
      () => extractLights(scene),
      /light\.shadow\.camera\.near must be positive for point and spot shadows/i,
      `${name} should fail clearly`,
    )
  }

  const directionalScene = new THREE.Scene()
  const directionalLight = new THREE.DirectionalLight(0xffffff, 1)
  directionalLight.castShadow = true
  directionalLight.shadow.camera.near = 0
  directionalLight.shadow.camera.far = 24
  directionalScene.add(directionalLight)
  const [nativeDirectionalLight] = extractLights(directionalScene) ?? []
  assert.equal(nativeDirectionalLight.shadowCameraNear, 0)
})

test('shadow radius values render PCF shadows', () => {
  function renderRadiusShadow(castShadow) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const caster = new THREE.Mesh(
      new THREE.BoxGeometry(3, 3, 3),
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    )
    caster.position.y = 1.5
    caster.castShadow = castShadow
    scene.add(caster)

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(8, 6, 0)
    light.target.position.set(0, 0, 0)
    light.castShadow = true
    light.shadow.radius = 4
    light.shadow.camera.left = -7
    light.shadow.camera.right = 7
    light.shadow.camera.top = 7
    light.shadow.camera.bottom = -7
    light.shadow.camera.near = 0.1
    light.shadow.camera.far = 16
    scene.add(light)
    scene.add(light.target)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 96, height: 96 }))
  }

  const unshadowed = renderRadiusShadow(false)
  const shadowed = renderRadiusShadow(true)
  const unshadowedLum = unshadowed.r + unshadowed.g + unshadowed.b
  const shadowedLum = shadowed.r + shadowed.g + shadowed.b
  assert.ok(shadowedLum < unshadowedLum - 20, `shadow radius should still render received shadows (${shadowedLum} vs ${unshadowedLum})`)
})
