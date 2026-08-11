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
import { countRegionPixels, renderRgba, solidTexture } from './scenes.test.part-002.mjs'
test('examples MD2 character helpers manage renderable synthetic parts', () => {
  const createSimplePart = (map, x) => {
    const geometry = new THREE.BoxGeometry(0.32, 0.32, 0.18)
    const materialTexture = new THREE.MeshBasicMaterial({ map, side: THREE.DoubleSide })
    const materialWireframe = new THREE.MeshBasicMaterial({ color: 0xffaa00, wireframe: true, side: THREE.DoubleSide })
    const mesh = new THREE.Mesh(geometry, materialTexture)
    mesh.position.x = x
    mesh.materialTexture = materialTexture
    mesh.materialWireframe = materialWireframe
    mesh.animations = [new THREE.AnimationClip('idle', 1, [])]
    return mesh
  }

  const createMd2MorphGeometry = () => {
    const geometry = new THREE.BoxGeometry(0.28, 0.28, 0.18)
    const position = geometry.getAttribute('position')
    geometry.morphTargetsRelative = true
    geometry.morphAttributes.position = ['idle_01', 'idle_02', 'move_01', 'move_02'].map((name, frame) => {
      const values = new Float32Array(position.count * 3)
      for (let i = 0; i < position.count; i += 1) values[i * 3] = frame >= 2 ? 0.04 : 0
      const attribute = new THREE.Float32BufferAttribute(values, 3)
      attribute.name = name
      return attribute
    })
    return geometry
  }

  const simpleBodySkin = solidTexture(255, 80, 80)
  const simpleBodyReplacementSkin = solidTexture(60, 255, 90)
  const simpleWeaponSkin = solidTexture(80, 130, 255)
  const simpleWeaponReplacementSkin = solidTexture(255, 220, 60)
  const complexBodySkin = solidTexture(255, 80, 220)
  const complexBodyReplacementSkin = solidTexture(80, 230, 255)
  const complexWeaponSkin = solidTexture(255, 80, 255)

  const simple = new MD2Character()
  const simpleBody = createSimplePart(simpleBodySkin, -0.8)
  const simpleWeapon = createSimplePart(simpleWeaponSkin, -0.45)
  const simpleActiveWeapon = createSimplePart(simpleWeaponReplacementSkin, -0.45)
  simpleWeapon.visible = false
  simpleActiveWeapon.visible = false
  simple.skinsBody = [simpleBodySkin, simpleBodyReplacementSkin]
  simple.skinsWeapon = [simpleWeaponSkin, simpleWeaponReplacementSkin]
  simple.meshBody = simpleBody
  simple.meshWeapon = simpleWeapon
  simple.weapons = [simpleWeapon, simpleActiveWeapon]
  simple.mixer = new THREE.AnimationMixer(simpleBody)
  simple.root.add(simpleBody, simpleWeapon, simpleActiveWeapon)

  const complexSource = new MD2CharacterComplex()
  complexSource.animations = {
    idle: 'idle',
    move: 'move',
    crouchIdle: 'idle',
    crouchMove: 'move',
    jump: 'idle',
    attack: 'move',
    crouchAttack: 'move',
  }
  complexSource.walkSpeed = 1
  complexSource.crouchSpeed = 0.5
  complexSource.skinsBody = [complexBodySkin, complexBodyReplacementSkin]
  complexSource.skinsWeapon = [complexWeaponSkin]
  const complexBodyGeometry = createMd2MorphGeometry()
  const complexWeaponGeometry = createMd2MorphGeometry()
  const complexSourceBody = complexSource._createPart(complexBodyGeometry, complexBodySkin)
  const complexSourceWeapon = complexSource._createPart(complexWeaponGeometry, complexWeaponSkin)
  complexSourceWeapon.name = 'synthetic-md2-weapon'
  complexSource.root.add(complexSourceBody, complexSourceWeapon)
  complexSource.meshBody = complexSourceBody
  complexSource.meshWeapon = complexSourceWeapon
  complexSource.weapons = [complexSourceWeapon]
  complexSource.meshes = [complexSourceBody, complexSourceWeapon]

  const complex = new MD2CharacterComplex()
  complex.shareParts(complexSource)
  complex.root.position.x = 0.45
  complex.meshBody.position.x = -0.15
  complex.meshWeapon.position.x = 0.25

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  scene.add(new THREE.AmbientLight(0xffffff, 1), simple.root, complex.root)

  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 0.75, -0.75, 0.01, 10)
  camera.position.set(0, 0, 4)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  try {
    simple.setPlaybackRate(2)
    assert.equal(simple.mixer.timeScale, 0.5)
    simple.setPlaybackRate(0)
    assert.equal(simple.mixer.timeScale, 0)
    simple.setPlaybackRate(1)
    simple.setWireframe(true)
    assert.equal(simpleBody.material, simpleBody.materialWireframe)
    simple.setWireframe(false)
    assert.equal(simpleBody.material, simpleBody.materialTexture)
    simple.setSkin(1)
    assert.equal(simpleBody.material.map, simpleBodyReplacementSkin)
    simple.setAnimation('idle')
    assert.ok(simpleBody.activeAction)
    simple.setWeapon(1)
    assert.equal(simpleWeapon.visible, false)
    assert.equal(simpleActiveWeapon.visible, true)
    assert.ok(simpleActiveWeapon.activeAction)
    simple.update(0.05)

    complex.enableShadows(true)
    assert.equal(complex.meshes.every((mesh) => mesh.castShadow && mesh.receiveShadow), true)
    complex.setVisible(true)
    complex.setWireframe(true)
    assert.equal(complex.meshBody.material, complex.meshBody.materialWireframe)
    complex.setWireframe(false)
    complex.setSkin(1)
    assert.equal(complex.currentSkin, 1)
    assert.equal(complex.meshBody.material.map, complexBodyReplacementSkin)
    complex.meshBody.baseDuration = 2
    complex.meshWeapon.baseDuration = 2
    complex.setPlaybackRate(2)
    assert.equal(complex.meshBody.duration, 1)
    complex.setAnimation('idle')
    complex.setWeapon(0)
    complex.controls = {
      moveForward: true,
      moveBackward: false,
      moveLeft: false,
      moveRight: false,
      crouch: false,
      jump: false,
      attack: false,
    }
    const previousZ = complex.root.position.z
    complex.update(0.16)
    assert.equal(complex.activeAnimation, 'move')
    assert.ok(complex.root.position.z > previousZ)
    assert.equal(complex.meshBody.animationsMap.move.active, true)

    const width = 128
    const height = 80
    const rgba = renderRgba(scene, camera, { width, height })
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => g > 210 && r < 200 && b < 205) > 50,
      'MD2Character skin switching should render the replacement green body skin',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => r > 210 && g > 200 && b < 190) > 50,
      'MD2Character weapon switching should render the active yellow weapon skin',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => g > 200 && b > 210 && r < 205) > 30,
      'MD2CharacterComplex shared body parts should render the replacement cyan skin',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => r > 210 && b > 210 && g < 200) > 30,
      'MD2CharacterComplex active weapon parts should render the magenta skin',
    )
  } finally {
    const materials = new Set()
    const geometries = new Set()
    const textures = new Set([
      simpleBodySkin,
      simpleBodyReplacementSkin,
      simpleWeaponSkin,
      simpleWeaponReplacementSkin,
      complexBodySkin,
      complexBodyReplacementSkin,
      complexWeaponSkin,
    ])
    for (const mesh of [
      simpleBody,
      simpleWeapon,
      simpleActiveWeapon,
      complexSourceBody,
      complexSourceWeapon,
      ...complex.meshes,
    ]) {
      geometries.add(mesh.geometry)
      materials.add(mesh.material)
      materials.add(mesh.materialTexture)
      materials.add(mesh.materialWireframe)
    }
    for (const material of materials) material?.dispose()
    for (const geometry of geometries) geometry?.dispose()
    for (const texture of textures) texture.dispose()
  }
})

test('examples GeometryUtils and TubePainter produce renderable geometry paths', () => {
  const hilbertGeometry = new THREE.BufferGeometry().setFromPoints(hilbert2D(new THREE.Vector3(), 0.75, 2))
  const hilbertMaterial = new THREE.LineBasicMaterial({ color: 0xff55ff, linewidth: 4 })
  const hilbertLine = new THREE.Line(hilbertGeometry, hilbertMaterial)
  hilbertLine.position.x = -0.8

  const painter = new TubePainter()
  painter.setSize(10)
  painter.moveTo(new THREE.Vector3(-0.35, -0.25, 0))
  painter.lineTo(new THREE.Vector3(0, 0.3, 0))
  painter.lineTo(new THREE.Vector3(0.35, -0.25, 0))
  painter.update()
  painter.mesh.position.x = 0.8
  painter.mesh.material.color.set(0xffaa44)
  painter.mesh.material.side = THREE.DoubleSide

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  scene.add(new THREE.AmbientLight(0xffffff, 2))
  scene.add(hilbertLine, painter.mesh)

  const camera = new THREE.OrthographicCamera(-1.8, 1.8, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 4)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  try {
    const width = 128
    const height = 72
    const rgba = renderRgba(scene, camera, { width, height })

    assert.equal(hilbertGeometry.getAttribute('position').count, 64)
    assert.ok(painter.mesh.geometry.drawRange.count >= 120)
    assert.ok(painter.mesh.geometry.getAttribute('position')?.count >= 120)
    assert.ok(painter.mesh.geometry.getAttribute('normal'), 'TubePainter should generate normals')
    assert.ok(painter.mesh.geometry.getAttribute('color'), 'TubePainter should generate vertex colors')
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => r > 180 && b > 160 && g < 140) > 100,
      'GeometryUtils.hilbert2D line should render magenta pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => r > 200 && g > 150 && b < 180) > 120,
      'TubePainter mesh should render orange pixels',
    )
  } finally {
    hilbertGeometry.dispose()
    hilbertMaterial.dispose()
    painter.mesh.geometry.dispose()
    painter.mesh.material.dispose()
  }
})
