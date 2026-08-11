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
import { makeCamera, meanRegion, renderRgba } from './scenes.test.part-002.mjs'
test('multiple shadow-casting directional lights render separate shadow maps', () => {
  function renderDirectionalShadows(lightXs) {
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
      new THREE.BoxGeometry(1.5, 1.5, 1.5),
      new THREE.MeshBasicMaterial({
        color: 0xffffff,
        colorWrite: false,
        depthWrite: false,
      }),
    )
    caster.position.y = 0.75
    caster.castShadow = true
    scene.add(caster)

    for (const x of lightXs) {
      const light = new THREE.DirectionalLight(0xffffff, 2)
      light.position.set(x, 5, 0)
      light.target.position.set(0, 0, 0)
      light.castShadow = true
      light.shadow.mapSize.set(512, 512)
      light.shadow.camera.left = -6
      light.shadow.camera.right = 6
      light.shadow.camera.top = 6
      light.shadow.camera.bottom = -6
      light.shadow.camera.near = 0.1
      light.shadow.camera.far = 12
      scene.add(light)
      scene.add(light.target)
    }

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 10, 0)
    camera.up.set(0, 0, -1)
    camera.lookAt(0, 0, 0)
    return renderRgba(scene, camera, { width: 128, height: 128 })
  }

  const firstOnly = renderDirectionalShadows([5])
  const secondOnly = renderDirectionalShadows([-5])
  const both = renderDirectionalShadows([5, -5])
  const luminance = (mean) => mean.r + mean.g + mean.b
  const left = [28, 54, 48, 74]
  const right = [80, 54, 100, 74]

  const firstLeft = luminance(meanRegion(firstOnly, 128, 128, ...left))
  const firstRight = luminance(meanRegion(firstOnly, 128, 128, ...right))
  const secondLeft = luminance(meanRegion(secondOnly, 128, 128, ...left))
  const secondRight = luminance(meanRegion(secondOnly, 128, 128, ...right))
  const bothLeft = luminance(meanRegion(both, 128, 128, ...left))
  const bothRight = luminance(meanRegion(both, 128, 128, ...right))

  assert.ok(firstLeft < firstRight - 30, `first light should cast the left shadow (${firstLeft} vs ${firstRight})`)
  assert.ok(secondRight < secondLeft - 30, `second light should cast the right shadow (${secondRight} vs ${secondLeft})`)
  assert.ok(bothLeft < secondLeft - 30, `dual shadow maps should keep the first light's left shadow (${bothLeft} vs ${secondLeft})`)
  assert.ok(bothRight < firstRight - 30, `dual shadow maps should add the second light's right shadow (${bothRight} vs ${firstRight})`)
})

test('point and directional shadow lights render within the expanded layer budget', () => {
  function renderMixedShadow(castShadow) {
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
      new THREE.BoxGeometry(2, 2, 2),
      new THREE.MeshBasicMaterial({
        color: 0xffffff,
        colorWrite: false,
        depthWrite: false,
      }),
    )
    caster.position.y = 1
    caster.castShadow = castShadow
    scene.add(caster)

    const point = new THREE.PointLight(0xffffff, 2.5, 16)
    point.position.set(0, 5, 4)
    point.castShadow = true
    point.shadow.mapSize.set(256, 256)
    point.shadow.camera.near = 0.1
    point.shadow.camera.far = 16
    scene.add(point)

    const directional = new THREE.DirectionalLight(0xffffff, 2)
    directional.position.set(5, 6, 0)
    directional.target.position.set(0, 0, 0)
    directional.castShadow = true
    directional.shadow.mapSize.set(256, 256)
    directional.shadow.camera.left = -7
    directional.shadow.camera.right = 7
    directional.shadow.camera.top = 7
    directional.shadow.camera.bottom = -7
    directional.shadow.camera.near = 0.1
    directional.shadow.camera.far = 16
    scene.add(directional)
    scene.add(directional.target)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 7, 8)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, { width: 96, height: 96 }), 96, 96, 28, 42, 68, 82)
  }

  const unshadowed = renderMixedShadow(false)
  const shadowed = renderMixedShadow(true)
  const unshadowedLum = unshadowed.r + unshadowed.g + unshadowed.b
  const shadowedLum = shadowed.r + shadowed.g + shadowed.b
  assert.ok(shadowedLum < unshadowedLum - 20, `mixed point/directional shadows should darken the receiver (${shadowedLum} vs ${unshadowedLum})`)
})

test('Renderer shadowMap enabled gates reusable renderer shadows', () => {
  function renderWithShadowMap(enabled) {
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
      new THREE.BoxGeometry(2, 2, 2),
      new THREE.MeshBasicMaterial({
        color: 0xffffff,
        colorWrite: false,
        depthWrite: false,
      }),
    )
    caster.position.y = 1
    caster.castShadow = true
    scene.add(caster)

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(5, 6, 0)
    light.target.position.set(0, 0, 0)
    light.castShadow = true
    light.shadow.mapSize.set(512, 512)
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

    const renderer = new Renderer()
    if (enabled !== undefined) renderer.shadowMap.enabled = enabled
    const rgba = renderer.render(scene, camera, { width: 96, height: 96, format: 'rgba' })
    return meanRegion(rgba, 96, 96, 28, 42, 68, 82)
  }

  const renderer = new Renderer()
  assert.equal(renderer.shadowMap.enabled, true)
  assert.equal(renderer.shadowMap.autoUpdate, true)
  assert.equal(renderer.shadowMap.needsUpdate, false)
  assert.equal(renderer.shadowMap.transmitted, false)
  assert.equal(renderer.shadowMap.type, THREE.PCFShadowMap)

  for (const shadowMapType of [THREE.BasicShadowMap, THREE.PCFShadowMap, THREE.PCFSoftShadowMap, THREE.VSMShadowMap]) {
    renderer.shadowMap.type = shadowMapType
    assert.equal(renderer.shadowMap.type, shadowMapType)
  }

  renderer.shadowMap.autoUpdate = false
  renderer.shadowMap.needsUpdate = true
  renderer.shadowMap.transmitted = true
  renderer.shadowMap.type = THREE.PCFSoftShadowMap
  assert.equal(renderer.shadowMap.autoUpdate, false)
  assert.equal(renderer.shadowMap.needsUpdate, true)
  assert.equal(renderer.shadowMap.transmitted, true)
  assert.equal(renderer.shadowMap.type, THREE.PCFSoftShadowMap)
  const shadowScene = new THREE.Scene()
  const shadowCamera = makeCamera()
  assert.equal(renderer.shadowMap.render([], shadowScene, shadowCamera), undefined)

  for (const [property, value, pattern] of [
    ['enabled', 'yes', /Renderer\.shadowMap\.enabled must be a boolean/i],
    ['autoUpdate', 'yes', /Renderer\.shadowMap\.autoUpdate must be a boolean/i],
    ['needsUpdate', 'yes', /Renderer\.shadowMap\.needsUpdate must be a boolean/i],
    ['transmitted', 'yes', /Renderer\.shadowMap\.transmitted must be a boolean/i],
    ['type', 'soft', /Renderer\.shadowMap\.type must be a Three\.js shadow map type constant/i],
    ['type', 999, /Renderer\.shadowMap\.type 999 is not supported.*BasicShadowMap.*PCFShadowMap.*PCFSoftShadowMap.*VSMShadowMap/i],
  ]) {
    assert.throws(
      () => { renderer.shadowMap[property] = value },
      pattern,
    )
  }
  assert.throws(
    () => renderer.shadowMap.render(null, shadowScene, shadowCamera),
    /Renderer\.shadowMap\.render lights must be an array/i,
  )
  assert.throws(
    () => renderer.shadowMap.render([], null, shadowCamera),
    /render\(scene, camera\) expects scene to be a THREE\.Scene or THREE\.Object3D root/i,
  )
  assert.throws(
    () => renderer.shadowMap.render([], shadowScene, null),
    /render\(scene, camera\) expects camera to be a THREE\.Camera, THREE\.ArrayCamera, or THREE\.CubeCamera/i,
  )

  const defaultShadowed = renderWithShadowMap(undefined)
  const explicitShadowed = renderWithShadowMap(true)
  const disabled = renderWithShadowMap(false)
  const defaultLum = defaultShadowed.r + defaultShadowed.g + defaultShadowed.b
  const explicitLum = explicitShadowed.r + explicitShadowed.g + explicitShadowed.b
  const disabledLum = disabled.r + disabled.g + disabled.b

  assert.ok(defaultLum < disabledLum - 20, `default Renderer shadowMap state should keep shadows enabled (${defaultLum} vs ${disabledLum})`)
  assert.ok(explicitLum < disabledLum - 20, `Renderer shadowMap.enabled=true should keep shadows enabled (${explicitLum} vs ${disabledLum})`)
})

test('Renderer shadowMap type controls Basic versus filtered sampling', () => {
  function renderShadowType(type, radius) {
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
    caster.castShadow = true
    scene.add(caster)

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(8, 6, 0)
    light.target.position.set(0, 0, 0)
    light.castShadow = true
    light.shadow.radius = radius
    light.shadow.mapSize.set(128, 128)
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

    const renderer = new Renderer()
    renderer.shadowMap.type = type
    const rgba = renderer.render(scene, camera, { width: 96, height: 96, format: 'rgba' })
    const mean = meanRegion(rgba, 96, 96, 28, 42, 68, 82)
    return mean.r + mean.g + mean.b
  }

  const basicSmallRadius = renderShadowType(THREE.BasicShadowMap, 0)
  const basicLargeRadius = renderShadowType(THREE.BasicShadowMap, 4)
  const pcfLargeRadius = renderShadowType(THREE.PCFShadowMap, 4)
  const pcfSoftLargeRadius = renderShadowType(THREE.PCFSoftShadowMap, 4)
  const vsmLargeRadius = renderShadowType(THREE.VSMShadowMap, 4)

  assert.ok(
    Math.abs(basicSmallRadius - basicLargeRadius) < 1,
    `BasicShadowMap should ignore PCF radius (${basicSmallRadius} vs ${basicLargeRadius})`,
  )
  assert.ok(
    pcfLargeRadius < basicLargeRadius - 10,
    `PCFShadowMap should use radius-based PCF sampling (${pcfLargeRadius} vs ${basicLargeRadius})`,
  )
  assert.ok(
    pcfSoftLargeRadius < basicLargeRadius - 10,
    `PCFSoftShadowMap should use the current filtered shadow path (${pcfSoftLargeRadius} vs ${basicLargeRadius})`,
  )
  assert.ok(
    vsmLargeRadius < basicLargeRadius - 10,
    `VSMShadowMap should use the current filtered shadow path (${vsmLargeRadius} vs ${basicLargeRadius})`,
  )
})
