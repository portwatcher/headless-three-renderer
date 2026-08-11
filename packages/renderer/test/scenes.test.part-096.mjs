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
import { cubeTexture, makeCamera, renderRgba, solidTexture } from './scenes.test.part-002.mjs'
test('unsupported framebuffer and depth texture scene inputs fail clearly', () => {
  function framebufferTexture() {
    const texture = new THREE.FramebufferTexture(1, 1)
    texture.needsUpdate = true
    return texture
  }

  function depthTexture() {
    const texture = new THREE.DepthTexture(1, 1)
    texture.needsUpdate = true
    return texture
  }

  const textureFactories = [
    ['FramebufferTexture', framebufferTexture, /uses a FramebufferTexture.*texture slots/i],
    ['DepthTexture', depthTexture, /uses a DepthTexture.*target\.depthTexture/i],
  ]
  const slots = [
    ['material map', (scene, texture) => {
      scene.add(new THREE.Mesh(
        new THREE.PlaneGeometry(2, 2),
        new THREE.MeshBasicMaterial({ map: texture }),
      ))
    }, (pattern) => new RegExp(`material\\.map ${pattern.source}`, 'i')],
    ['background', (scene, texture) => {
      scene.background = texture
    }, (pattern) => new RegExp(`background ${pattern.source}`, 'i')],
    ['environment', (scene, texture) => {
      texture.mapping = THREE.EquirectangularReflectionMapping
      scene.environment = texture
    }, (pattern) => new RegExp(`scene\\.environment ${pattern.source}`, 'i')],
    ['material envMap', (scene, texture) => {
      texture.mapping = THREE.EquirectangularReflectionMapping
      scene.add(new THREE.Mesh(
        new THREE.PlaneGeometry(2, 2),
        new THREE.MeshBasicMaterial({ envMap: texture }),
      ))
    }, (pattern) => new RegExp(`material\\.envMap ${pattern.source}`, 'i')],
    ['reflection probe', (scene, texture) => {
      texture.mapping = THREE.EquirectangularReflectionMapping
      scene.userData.headlessThreeRenderer = { reflectionProbe: { texture } }
    }, (pattern) => new RegExp(`reflectionProbe\\.texture ${pattern.source}`, 'i')],
  ]

  for (const [textureName, makeTexture, pattern] of textureFactories) {
    for (const [slotName, setup, slotPattern] of slots) {
      const scene = new THREE.Scene()
      setup(scene, makeTexture())
      assert.throws(
        () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
        slotPattern(pattern),
        `${slotName} ${textureName} should fail clearly`,
      )
    }
  }
})

test('unsupported VideoTexture scene inputs fail clearly in Node slots', () => {
  function videoTexture() {
    const texture = new THREE.VideoTexture({ videoWidth: 1, videoHeight: 1 })
    texture.needsUpdate = true
    return texture
  }

  const slots = [
    ['material map', (scene, texture) => {
      scene.add(new THREE.Mesh(
        new THREE.PlaneGeometry(2, 2),
        new THREE.MeshBasicMaterial({ map: texture }),
      ))
    }, /material\.map uses a VideoTexture.*live video frames.*not directly readable/i],
    ['background', (scene, texture) => {
      scene.background = texture
    }, /background uses a VideoTexture.*live video frames.*not directly readable/i],
    ['environment', (scene, texture) => {
      texture.mapping = THREE.EquirectangularReflectionMapping
      scene.environment = texture
    }, /scene\.environment uses a VideoTexture.*live video frames.*not directly readable/i],
    ['material envMap', (scene, texture) => {
      texture.mapping = THREE.EquirectangularReflectionMapping
      scene.add(new THREE.Mesh(
        new THREE.PlaneGeometry(2, 2),
        new THREE.MeshBasicMaterial({ envMap: texture }),
      ))
    }, /material\.envMap uses a VideoTexture.*live video frames.*not directly readable/i],
    ['reflection probe', (scene, texture) => {
      texture.mapping = THREE.EquirectangularReflectionMapping
      scene.userData.headlessThreeRenderer = { reflectionProbe: { texture } }
    }, /reflectionProbe\.texture uses a VideoTexture.*live video frames.*not directly readable/i],
  ]

  for (const [slotName, setup, pattern] of slots) {
    const scene = new THREE.Scene()
    setup(scene, videoTexture())
    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      pattern,
      `${slotName} VideoTexture should fail clearly`,
    )
  }
})

test('unsupported StorageTexture scene inputs fail clearly in Node slots', () => {
  function storageTexture() {
    const texture = new THREE_WEBGPU.StorageTexture(1, 1)
    texture.needsUpdate = true
    return texture
  }

  const slots = [
    ['material map', (scene, texture) => {
      scene.add(new THREE.Mesh(
        new THREE.PlaneGeometry(2, 2),
        new THREE.MeshBasicMaterial({ map: texture }),
      ))
    }, /material\.map uses a StorageTexture.*backing data.*not directly readable/i],
    ['background', (scene, texture) => {
      scene.background = texture
    }, /background uses a StorageTexture.*backing data.*not directly readable/i],
    ['environment', (scene, texture) => {
      texture.mapping = THREE.EquirectangularReflectionMapping
      scene.environment = texture
    }, /scene\.environment uses a StorageTexture.*backing data.*not directly readable/i],
    ['material envMap', (scene, texture) => {
      texture.mapping = THREE.EquirectangularReflectionMapping
      scene.add(new THREE.Mesh(
        new THREE.PlaneGeometry(2, 2),
        new THREE.MeshBasicMaterial({ envMap: texture }),
      ))
    }, /material\.envMap uses a StorageTexture.*backing data.*not directly readable/i],
    ['reflection probe', (scene, texture) => {
      texture.mapping = THREE.EquirectangularReflectionMapping
      scene.userData.headlessThreeRenderer = { reflectionProbe: { texture } }
    }, /reflectionProbe\.texture uses a StorageTexture.*backing data.*not directly readable/i],
  ]

  for (const [slotName, setup, pattern] of slots) {
    const scene = new THREE.Scene()
    setup(scene, storageTexture())
    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      pattern,
      `${slotName} StorageTexture should fail clearly`,
    )
  }
})

test('unsupported cube texture material slots fail clearly', () => {
  function makeCubeMap() {
    return cubeTexture([
      [255, 0, 0],
      [0, 255, 0],
      [0, 0, 255],
      [255, 255, 0],
      [255, 0, 255],
      [0, 255, 255],
    ])
  }

  function makeCubeUvMap() {
    const texture = solidTexture(255, 255, 255)
    texture.mapping = THREE.CubeUVReflectionMapping
    return texture
  }

  const textureFactories = [
    ['CubeTexture', makeCubeMap],
    ['CubeUV texture', makeCubeUvMap],
  ]
  const unsupported2DSlotPattern = (slot) => new RegExp(`material\\.${slot} uses a cube or PMREM\\/CubeUV texture mapping.*2D material texture slots`, 'i')
  const physicalMapSlot = (slot, extra = {}) => (texture) => new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshPhysicalMaterial({
      color: 0xffffff,
      ...extra,
      [slot]: texture,
    }),
  )
  const slots = [
    ['MeshBasicMaterial.map', (texture) => new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ map: texture }),
    ), unsupported2DSlotPattern('map')],
    ['MeshBasicMaterial.alphaMap', (texture) => new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ alphaMap: texture, alphaTest: 0.5 }),
    ), unsupported2DSlotPattern('alphaMap')],
    ['MeshStandardMaterial.normalMap', (texture) => new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshStandardMaterial({ normalMap: texture }),
    ), unsupported2DSlotPattern('normalMap')],
    ['MeshPhongMaterial.specularMap', (texture) => new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshPhongMaterial({ specularMap: texture }),
    ), unsupported2DSlotPattern('specularMap')],
    ['MeshPhysicalMaterial.clearcoatMap', physicalMapSlot('clearcoatMap', { clearcoat: 1 }), unsupported2DSlotPattern('clearcoatMap')],
    ['MeshPhysicalMaterial.clearcoatRoughnessMap', physicalMapSlot('clearcoatRoughnessMap', { clearcoat: 1 }), unsupported2DSlotPattern('clearcoatRoughnessMap')],
    ['MeshPhysicalMaterial.clearcoatNormalMap', physicalMapSlot('clearcoatNormalMap', { clearcoat: 1 }), unsupported2DSlotPattern('clearcoatNormalMap')],
    ['MeshPhysicalMaterial.sheenColorMap', physicalMapSlot('sheenColorMap', { sheen: 1 }), unsupported2DSlotPattern('sheenColorMap')],
    ['MeshPhysicalMaterial.sheenRoughnessMap', physicalMapSlot('sheenRoughnessMap', { sheen: 1 }), unsupported2DSlotPattern('sheenRoughnessMap')],
    ['MeshPhysicalMaterial.anisotropyMap', physicalMapSlot('anisotropyMap', { anisotropy: 1 }), unsupported2DSlotPattern('anisotropyMap')],
    ['MeshPhysicalMaterial.iridescenceMap', physicalMapSlot('iridescenceMap', { iridescence: 1 }), unsupported2DSlotPattern('iridescenceMap')],
    ['MeshPhysicalMaterial.iridescenceThicknessMap', physicalMapSlot('iridescenceThicknessMap', { iridescence: 1 }), unsupported2DSlotPattern('iridescenceThicknessMap')],
    ['MeshPhysicalMaterial.transmissionMap', physicalMapSlot('transmissionMap', { transmission: 1 }), unsupported2DSlotPattern('transmissionMap')],
    ['MeshPhysicalMaterial.thicknessMap', physicalMapSlot('thicknessMap', { transmission: 1, thickness: 1 }), unsupported2DSlotPattern('thicknessMap')],
    ['MeshPhysicalMaterial.specularColorMap', physicalMapSlot('specularColorMap', { specularIntensity: 1 }), unsupported2DSlotPattern('specularColorMap')],
    ['MeshPhysicalMaterial.specularIntensityMap', physicalMapSlot('specularIntensityMap', { specularIntensity: 1 }), unsupported2DSlotPattern('specularIntensityMap')],
    ['SpriteMaterial.map', (texture) => new THREE.Sprite(new THREE.SpriteMaterial({ map: texture })),
      unsupported2DSlotPattern('map')],
    ['PointsMaterial.alphaMap', (texture) => {
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))
      return new THREE.Points(geometry, new THREE.PointsMaterial({ alphaMap: texture, alphaTest: 0.5 }))
    }, unsupported2DSlotPattern('alphaMap')],
    ['LineBasicMaterial.map', (texture) => {
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-1, 0, 0, 1, 0, 0]), 3))
      return new THREE.Line(geometry, new THREE.LineBasicMaterial({ map: texture }))
    }, unsupported2DSlotPattern('map')],
  ]

  for (const [textureName, makeTexture] of textureFactories) {
    for (const [slotName, makeObject, pattern] of slots) {
      const scene = new THREE.Scene()
      scene.add(makeObject(makeTexture()))

      assert.throws(
        () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
        pattern,
        `${slotName} ${textureName} should fail clearly`,
      )
    }
  }
})

test('cube textures remain supported for background and environment slots', () => {
  const scene = new THREE.Scene()
  scene.background = cubeTexture([
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
  ])
  scene.environment = scene.background
  scene.add(new THREE.Mesh(
    new THREE.SphereGeometry(1, 16, 8),
    new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0.25 }),
  ))

  const mean = meanRgba(renderRgba(scene, makeCamera(), { width: 64, height: 64 }))
  assert.ok(mean.r > 20 || mean.g > 20 || mean.b > 20, `cube background/environment should still render visibly (${mean.r}, ${mean.g}, ${mean.b})`)
})
