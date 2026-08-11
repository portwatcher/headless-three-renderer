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
import { extractAmbientIntensity, extractLightProbe, extractLights, test } from './scenes.test.part-001.mjs'
import { renderRgba, solidTexture } from './scenes.test.part-002.mjs'
import { maxLuminance } from './scenes.test.part-003.mjs'
test('LightProbe combines with scene environment across lit material models', () => {
  function makeGreenEnvironment() {
    const texture = solidTexture(0, 255, 0)
    texture.mapping = THREE.EquirectangularReflectionMapping
    return texture
  }

  function makeRedProbe() {
    const probe = new THREE.LightProbe(undefined, 1.5)
    for (const coefficient of probe.sh.coefficients) {
      coefficient.set(0, 0, 0)
    }
    probe.sh.coefficients[0].set(1, 0, 0)
    return probe
  }

  function renderMaterial(material, { environment = false, probe = false } = {}) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    if (environment) {
      scene.environment = makeGreenEnvironment()
      scene.environmentIntensity = 2.5
    }
    if (probe) {
      scene.add(makeRedProbe())
    }
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const cases = [
    ['Standard', () => new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, metalness: 0 })],
    ['Physical', () => new THREE.MeshPhysicalMaterial({ color: 0xffffff, roughness: 1, metalness: 0 })],
    ['Lambert', () => new THREE.MeshLambertMaterial({ color: 0xffffff })],
    ['Phong', () => new THREE.MeshPhongMaterial({ color: 0xffffff, shininess: 20 })],
    ['Toon', () => new THREE.MeshToonMaterial({ color: 0xffffff })],
  ]

  for (const [name, makeMaterial] of cases) {
    const environmentOnly = renderMaterial(makeMaterial(), { environment: true })
    const probeOnly = renderMaterial(makeMaterial(), { probe: true })
    const combined = renderMaterial(makeMaterial(), { environment: true, probe: true })
    assert.ok(combined.r > environmentOnly.r + 5, `${name} combined LightProbe/environment should add red probe diffuse lighting (${combined.r} vs ${environmentOnly.r})`)
    assert.ok(combined.g > probeOnly.g + 80, `${name} combined LightProbe/environment should keep green environment lighting (${combined.g} vs ${probeOnly.g})`)
  }
})

test('RectAreaLight approximates finite one-sided area lighting', () => {
  function renderRectArea(width, height, targetZ) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, metalness: 0 }),
    ))

    const light = new THREE.RectAreaLight(0xffffff, 20, width, height)
    light.position.set(0, 0, 2)
    light.lookAt(0, 0, targetZ)
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return maxLuminance(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const smallForward = renderRectArea(0.5, 0.5, 0)
  const largeForward = renderRectArea(3, 3, 0)
  const backward = renderRectArea(3, 3, 4)

  assert.ok(smallForward > backward + 10, `forward RectAreaLight should illuminate its front side (${smallForward} vs ${backward})`)
  assert.ok(largeForward > smallForward + 10, `larger RectAreaLight should contribute more radiance (${largeForward} vs ${smallForward})`)
})

test('invalid light numeric values fail clearly', () => {
  const directCases = [
    ['directional intensity', () => {
      const light = new THREE.DirectionalLight(0xffffff, 1)
      light.intensity = 'bright'
      return light
    }, /light\.intensity must be a finite number/i],
    ['point intensity', () => {
      const light = new THREE.PointLight(0xffffff, 1)
      light.intensity = Number.NaN
      return light
    }, /light\.intensity must be a finite number/i],
    ['spot intensity', () => {
      const light = new THREE.SpotLight(0xffffff, 1)
      light.intensity = Number.POSITIVE_INFINITY
      return light
    }, /light\.intensity must be a finite number/i],
    ['hemisphere intensity', () => {
      const light = new THREE.HemisphereLight(0xffffff, 0x222222, 1)
      light.intensity = 'bright'
      return light
    }, /light\.intensity must be a finite number/i],
    ['rect intensity', () => {
      const light = new THREE.RectAreaLight(0xffffff, 1, 1, 1)
      light.intensity = Number.NEGATIVE_INFINITY
      return light
    }, /light\.intensity must be a finite number/i],
    ['directional target matrix', () => {
      const light = new THREE.DirectionalLight(0xffffff, 1)
      light.target.matrixWorld.elements[14] = Number.NaN
      return light
    }, /DirectionalLight\.target\.matrixWorld\.elements\[14\] must be a finite number/i],
    ['directional transform matrix', () => {
      const light = new THREE.DirectionalLight(0xffffff, 1)
      light.matrixWorld.elements[12] = Number.NaN
      return light
    }, /DirectionalLight\.matrixWorld\.elements\[12\] must be a finite number/i],
    ['directional target container', () => {
      const light = new THREE.DirectionalLight(0xffffff, 1)
      light.target = 'target'
      return light
    }, /DirectionalLight\.target must be an object/i],
    ['point transform matrix', () => {
      const light = new THREE.PointLight(0xffffff, 1)
      light.matrixWorld.elements[12] = Number.NaN
      return light
    }, /PointLight\.matrixWorld\.elements\[12\] must be a finite number/i],
    ['point distance', () => {
      const light = new THREE.PointLight(0xffffff, 1)
      light.distance = 'far'
      return light
    }, /PointLight\.distance must be a finite number/i],
    ['point distance negative', () => {
      const light = new THREE.PointLight(0xffffff, 1)
      light.distance = -1
      return light
    }, /PointLight\.distance must be non-negative/i],
    ['point decay', () => {
      const light = new THREE.PointLight(0xffffff, 1)
      light.decay = Number.POSITIVE_INFINITY
      return light
    }, /PointLight\.decay must be a finite number/i],
    ['point decay negative', () => {
      const light = new THREE.PointLight(0xffffff, 1)
      light.decay = -0.5
      return light
    }, /PointLight\.decay must be non-negative/i],
    ['spot distance', () => {
      const light = new THREE.SpotLight(0xffffff, 1)
      light.distance = Number.NaN
      return light
    }, /SpotLight\.distance must be a finite number/i],
    ['spot distance negative', () => {
      const light = new THREE.SpotLight(0xffffff, 1)
      light.distance = -1
      return light
    }, /SpotLight\.distance must be non-negative/i],
    ['spot target container', () => {
      const light = new THREE.SpotLight(0xffffff, 1)
      light.target = []
      return light
    }, /SpotLight\.target must be an object/i],
    ['spot transform matrix', () => {
      const light = new THREE.SpotLight(0xffffff, 1)
      light.matrixWorld.elements[12] = Number.NaN
      return light
    }, /SpotLight\.matrixWorld\.elements\[12\] must be a finite number/i],
    ['spot target matrix', () => {
      const light = new THREE.SpotLight(0xffffff, 1)
      light.target.matrixWorld.elements[13] = Number.NaN
      return light
    }, /SpotLight\.target\.matrixWorld\.elements\[13\] must be a finite number/i],
    ['spot decay negative', () => {
      const light = new THREE.SpotLight(0xffffff, 1)
      light.decay = -0.5
      return light
    }, /SpotLight\.decay must be non-negative/i],
    ['spot angle', () => {
      const light = new THREE.SpotLight(0xffffff, 1)
      light.angle = 'wide'
      return light
    }, /SpotLight\.angle must be a finite number/i],
    ['spot angle negative', () => {
      const light = new THREE.SpotLight(0xffffff, 1)
      light.angle = -0.1
      return light
    }, /SpotLight\.angle must be between 0 and Math\.PI \/ 2/i],
    ['spot angle too wide', () => {
      const light = new THREE.SpotLight(0xffffff, 1)
      light.angle = Math.PI
      return light
    }, /SpotLight\.angle must be between 0 and Math\.PI \/ 2/i],
    ['spot penumbra', () => {
      const light = new THREE.SpotLight(0xffffff, 1)
      light.penumbra = Number.NEGATIVE_INFINITY
      return light
    }, /SpotLight\.penumbra must be a finite number/i],
    ['spot penumbra negative', () => {
      const light = new THREE.SpotLight(0xffffff, 1)
      light.penumbra = -0.1
      return light
    }, /SpotLight\.penumbra must be between 0 and 1/i],
    ['spot penumbra above one', () => {
      const light = new THREE.SpotLight(0xffffff, 1)
      light.penumbra = 1.5
      return light
    }, /SpotLight\.penumbra must be between 0 and 1/i],
    ['hemisphere transform matrix', () => {
      const light = new THREE.HemisphereLight(0xffffff, 0x222222, 1)
      light.matrixWorld.elements[5] = Number.NaN
      return light
    }, /HemisphereLight\.matrixWorld\.elements\[5\] must be a finite number/i],
    ['rect width', () => {
      const light = new THREE.RectAreaLight(0xffffff, 1, 1, 1)
      light.width = 'wide'
      return light
    }, /RectAreaLight\.width must be a finite number/i],
    ['rect width zero', () => {
      const light = new THREE.RectAreaLight(0xffffff, 1, 1, 1)
      light.width = 0
      return light
    }, /RectAreaLight\.width must be positive/i],
    ['rect height', () => {
      const light = new THREE.RectAreaLight(0xffffff, 1, 1, 1)
      light.height = Number.NaN
      return light
    }, /RectAreaLight\.height must be a finite number/i],
    ['rect height negative', () => {
      const light = new THREE.RectAreaLight(0xffffff, 1, 1, 1)
      light.height = -1
      return light
    }, /RectAreaLight\.height must be positive/i],
    ['rect transform matrix', () => {
      const light = new THREE.RectAreaLight(0xffffff, 1, 1, 1)
      light.matrixWorld.elements[8] = Number.NEGATIVE_INFINITY
      return light
    }, /RectAreaLight\.matrixWorld\.elements\[8\] must be a finite number/i],
  ]

  for (const [name, makeLight, pattern] of directCases) {
    const scene = new THREE.Scene()
    scene.add(makeLight())
    assert.throws(
      () => extractLights(scene),
      pattern,
      `${name} should fail clearly`,
    )
  }

  const ambientScene = new THREE.Scene()
  const ambient = new THREE.AmbientLight(0xffffff, 1)
  ambient.intensity = 'bright'
  ambientScene.add(ambient)
  assert.throws(
    () => extractAmbientIntensity(ambientScene),
    /AmbientLight\.intensity must be a finite number/i,
  )

  const probeScene = new THREE.Scene()
  const probe = new THREE.LightProbe(undefined, 1)
  probe.intensity = Number.NaN
  probeScene.add(probe)
  assert.throws(
    () => extractLightProbe(probeScene),
    /LightProbe\.intensity must be a finite number/i,
  )

  const vectorCoefficientScene = new THREE.Scene()
  const vectorCoefficientProbe = new THREE.LightProbe(undefined, 1)
  vectorCoefficientProbe.sh.coefficients[0] = { x: 1, y: 'green', z: 0 }
  vectorCoefficientScene.add(vectorCoefficientProbe)
  assert.throws(
    () => extractLightProbe(vectorCoefficientScene),
    /LightProbe\.sh\.coefficients\[0\]\.y must be a finite number/i,
  )

  const arrayCoefficientScene = new THREE.Scene()
  const arrayCoefficientProbe = new THREE.LightProbe(undefined, 1)
  arrayCoefficientProbe.sh.coefficients[0] = [1, Number.NEGATIVE_INFINITY, 0]
  arrayCoefficientScene.add(arrayCoefficientProbe)
  assert.throws(
    () => extractLightProbe(arrayCoefficientScene),
    /LightProbe\.sh\.coefficients\[0\]\[1\] must be a finite number/i,
  )

  const missingCoefficientsScene = new THREE.Scene()
  const missingCoefficientsProbe = new THREE.LightProbe(undefined, 1)
  missingCoefficientsProbe.sh.coefficients = [{ x: 1, y: 0, z: 0 }]
  missingCoefficientsScene.add(missingCoefficientsProbe)
  assert.throws(
    () => extractLightProbe(missingCoefficientsScene),
    /LightProbe\.sh\.coefficients must contain 9 coefficients/i,
  )

  const invalidCoefficientsScene = new THREE.Scene()
  const invalidCoefficientsProbe = new THREE.LightProbe(undefined, 1)
  invalidCoefficientsProbe.sh.coefficients = 'bright'
  invalidCoefficientsScene.add(invalidCoefficientsProbe)
  assert.throws(
    () => extractLightProbe(invalidCoefficientsScene),
    /LightProbe\.sh\.coefficients must be an array of 9 coefficients/i,
  )
})
