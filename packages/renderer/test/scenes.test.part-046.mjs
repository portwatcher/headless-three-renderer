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
import { constantUvPlane, cubeTexture, meanRegion, packedCubeUvTexture, renderRgba, rgbaTexture, solidTexture } from './scenes.test.part-002.mjs'
import { maxLuminance } from './scenes.test.part-003.mjs'
test('MeshPhongMaterial specularMap honors nearest texture filters', () => {
  function renderWithFilter(filter) {
    const specularMap = rgbaTexture([
      0, 0, 0, 255,
      255, 0, 0, 255,
    ], 2, 1)
    specularMap.magFilter = filter
    specularMap.minFilter = filter

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.45, 0.5),
      new THREE.MeshPhongMaterial({
        color: 0x000000,
        specular: 0xffffff,
        shininess: 4,
        specularMap,
      }),
    ))

    const light = new THREE.DirectionalLight(0xffffff, 8)
    light.position.set(0, 0, 3)
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const nearest = renderWithFilter(THREE.NearestFilter)
  const linear = renderWithFilter(THREE.LinearFilter)
  assert.ok(linear.r > nearest.r + 25, `LinearFilter should blend in the enabled specular texel (${linear.r} vs ${nearest.r})`)
})

test('MeshPhongMaterial specularMap honors horizontal and vertical repeat wrapping', () => {
  function renderWithWrapping({ wrapS, wrapT, vertical = false }) {
    const specularMap = vertical
      ? rgbaTexture([
        255, 0, 0, 255,
        0, 0, 0, 255,
      ], 1, 2)
      : rgbaTexture([
        255, 0, 0, 255,
        0, 0, 0, 255,
      ], 2, 1)
    if (wrapS != null) specularMap.wrapS = wrapS
    if (wrapT != null) specularMap.wrapT = wrapT
    specularMap.magFilter = THREE.NearestFilter
    specularMap.minFilter = THREE.NearestFilter

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(vertical ? 0.5 : 1.25, vertical ? 1.25 : 0.5),
      new THREE.MeshPhongMaterial({
        color: 0x000000,
        specular: 0xffffff,
        shininess: 4,
        specularMap,
      }),
    ))

    const light = new THREE.DirectionalLight(0xffffff, 8)
    light.position.set(0, 0, 3)
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const clamped = renderWithWrapping({ wrapS: THREE.ClampToEdgeWrapping })
  const repeated = renderWithWrapping({ wrapS: THREE.RepeatWrapping })
  const mirrored = renderWithWrapping({ wrapS: THREE.MirroredRepeatWrapping })
  assert.ok(clamped.r < 5, `clamped specularMap U coordinates should sample the disabled edge texel (${clamped.r})`)
  assert.ok(repeated.r > clamped.r + 80, `repeated specularMap U coordinates should wrap to the enabled texel (${repeated.r} vs ${clamped.r})`)
  assert.ok(mirrored.r < 5, `mirrored specularMap U coordinates should reflect to the disabled texel (${mirrored.r})`)
  assert.ok(repeated.r > mirrored.r + 80, `mirrored specularMap U coordinates should differ from RepeatWrapping (${mirrored.r} vs ${repeated.r})`)

  const clampedVertical = renderWithWrapping({ wrapT: THREE.ClampToEdgeWrapping, vertical: true })
  const repeatedVertical = renderWithWrapping({ wrapT: THREE.RepeatWrapping, vertical: true })
  const mirroredVertical = renderWithWrapping({ wrapT: THREE.MirroredRepeatWrapping, vertical: true })
  assert.ok(clampedVertical.r < 5, `clamped specularMap V coordinates should sample the disabled edge texel (${clampedVertical.r})`)
  assert.ok(repeatedVertical.r > clampedVertical.r + 80, `repeated specularMap V coordinates should wrap to the enabled texel (${repeatedVertical.r} vs ${clampedVertical.r})`)
  assert.ok(mirroredVertical.r < 5, `mirrored specularMap V coordinates should reflect to the disabled texel (${mirroredVertical.r})`)
  assert.ok(repeatedVertical.r > mirroredVertical.r + 80, `mirrored specularMap V coordinates should differ from RepeatWrapping (${mirroredVertical.r} vs ${repeatedVertical.r})`)
})

test('MeshPhongMaterial scene environment feeds specular reflection', () => {
  function renderPhongEnvironment(specularMap, useEnvironment) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    if (useEnvironment) {
      const environment = solidTexture(255, 255, 255)
      environment.mapping = THREE.EquirectangularReflectionMapping
      scene.environment = environment
    }
    scene.add(new THREE.Mesh(
      new THREE.SphereGeometry(1, 48, 24),
      new THREE.MeshPhongMaterial({
        color: 0x000000,
        specular: 0xffffff,
        shininess: 120,
        specularMap,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return renderRgba(scene, camera, { width: 96, height: 96 })
  }

  const noEnvironment = maxLuminance(renderPhongEnvironment(null, false))
  const environment = maxLuminance(renderPhongEnvironment(null, true))
  const maskedEnvironment = maxLuminance(renderPhongEnvironment(solidTexture(0, 0, 0), true))
  assert.ok(environment > noEnvironment + 40, `scene environment should add a Phong reflection (${environment} vs ${noEnvironment})`)
  assert.ok(environment > maskedEnvironment + 40, `specularMap should suppress Phong environment reflection (${environment} vs ${maskedEnvironment})`)
})

test('MeshPhongMaterial material envMap feeds specular reflection', () => {
  function renderPhongMaterialEnvironment(intensity) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const envMap = solidTexture(255, 255, 255)
    envMap.mapping = THREE.EquirectangularReflectionMapping
    const material = new THREE.MeshPhongMaterial({
      color: 0x000000,
      specular: 0xffffff,
      shininess: 120,
      envMap,
    })
    if (intensity != null) material.envMapIntensity = intensity
    scene.add(new THREE.Mesh(new THREE.SphereGeometry(1, 48, 24), material))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return maxLuminance(renderRgba(scene, camera, { width: 96, height: 96 }))
  }

  const disabled = renderPhongMaterialEnvironment(0)
  const reflected = renderPhongMaterialEnvironment(1)
  assert.ok(reflected > disabled + 40, `material envMap should add Phong reflection (${reflected} vs ${disabled})`)
})

test('MeshPhongMaterial material envMap honors legacy combine and reflectivity', () => {
  function renderPhongMaterialEnvironment(combine, reflectivity = 1) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const envMap = solidTexture(0, 255, 0)
    envMap.mapping = THREE.EquirectangularReflectionMapping
    const material = new THREE.MeshPhongMaterial({
      color: 0x000000,
      specular: 0xffffff,
      shininess: 120,
      envMap,
      combine,
      reflectivity,
    })
    material.envMapIntensity = 0.5
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 64, 64, 24, 24, 40, 40)
  }

  const disabled = renderPhongMaterialEnvironment(THREE.MultiplyOperation, 0)
  const multiply = renderPhongMaterialEnvironment(THREE.MultiplyOperation)
  const add = renderPhongMaterialEnvironment(THREE.AddOperation)

  assert.ok(multiply.g > disabled.g + 10, `reflectivity should scale Phong env reflection (${multiply.g} vs ${disabled.g})`)
  assert.ok(add.g > multiply.g + 10, `AddOperation should add extra Phong env reflection (${add.g} vs ${multiply.g})`)
})

test('MeshLambertMaterial material envMap honors legacy mix reflectivity', () => {
  function renderLambertMaterialEnvironment(reflectivity) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const envMap = solidTexture(0, 255, 0)
    envMap.mapping = THREE.EquirectangularReflectionMapping
    const material = new THREE.MeshLambertMaterial({
      color: 0xff0000,
      envMap,
      combine: THREE.MixOperation,
      reflectivity,
    })
    material.envMapIntensity = 0.5
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

    const light = new THREE.DirectionalLight(0xffffff, 4)
    light.position.set(0, 0, 3)
    scene.add(light)

    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 64, 64, 24, 24, 40, 40)
  }

  const disabled = renderLambertMaterialEnvironment(0)
  const mixed = renderLambertMaterialEnvironment(1)

  assert.ok(disabled.r > disabled.g + 20, `reflectivity 0 should preserve direct Lambert color (${disabled.r}, ${disabled.g})`)
  assert.ok(mixed.g > mixed.r + 20, `MixOperation should replace Lambert output with green env reflection (${mixed.r}, ${mixed.g})`)
})

test('MeshBasicMaterial material envMap uses legacy combine modes', () => {
  function renderBasicMaterialEnvironment(combine, reflectivity = 1) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const envMap = solidTexture(0, 255, 0)
    envMap.mapping = THREE.EquirectangularReflectionMapping
    const material = new THREE.MeshBasicMaterial({
      color: 0xff0000,
      envMap,
      combine,
      reflectivity,
    })
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 64, 64, 24, 24, 40, 40)
  }

  const multiply = renderBasicMaterialEnvironment(THREE.MultiplyOperation)
  const add = renderBasicMaterialEnvironment(THREE.AddOperation)
  const mixZero = renderBasicMaterialEnvironment(THREE.MixOperation, 0)
  const mixFull = renderBasicMaterialEnvironment(THREE.MixOperation, 1)

  assert.ok(add.g > multiply.g + 40, `AddOperation should add green env reflection (${add.g} vs ${multiply.g})`)
  assert.ok(mixZero.r > mixZero.g + 40, `reflectivity 0 should preserve Basic color (${mixZero.r}, ${mixZero.g})`)
  assert.ok(mixFull.g > mixFull.r + 40, `MixOperation should replace with green env reflection (${mixFull.r}, ${mixFull.g})`)
})

test('CubeUV-mapped cube material envMap feeds supported material paths', () => {
  const envMap = cubeTexture([
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0],
  ])
  envMap.mapping = THREE.CubeUVReflectionMapping

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({
      color: 0xff0000,
      envMap,
      combine: THREE.MixOperation,
      reflectivity: 1,
    }),
  ))

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  const mean = meanRegion(renderRgba(scene, camera, {
    width: 64,
    height: 64,
    outputColorSpace: THREE.LinearSRGBColorSpace,
  }), 64, 64, 24, 24, 40, 40)

  assert.ok(mean.g > mean.r + 40, `CubeUV-mapped cube material envMap should replace with green (${mean.r}, ${mean.g})`)

  const packedEnvMap = packedCubeUvTexture([
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0],
  ])
  scene.children[0].material.envMap = packedEnvMap
  const packedMean = meanRegion(renderRgba(scene, camera, {
    width: 64,
    height: 64,
    outputColorSpace: THREE.LinearSRGBColorSpace,
  }), 64, 64, 24, 24, 40, 40)
  assert.ok(packedMean.g > packedMean.r + 40, `packed CubeUV material envMap should replace with green (${packedMean.r}, ${packedMean.g})`)
})
