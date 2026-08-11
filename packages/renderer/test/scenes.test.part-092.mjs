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
import { LuminanceAlphaFormat, UnsignedInt101111Type, extractEnvironmentMap, test } from './scenes.test.part-001.mjs'
import { halfFloatToNumber, makeCamera, meanAbsDiff, meanRegion, renderRgba, solidTexture } from './scenes.test.part-002.mjs'
import { packR11G11B10F, packRgb9E5 } from './scenes.test.part-003.mjs'
test('float raw environment textures decode for IBL', () => {
  function byteEnvironmentTexture() {
    const texture = solidTexture(128, 64, 255)
    texture.colorSpace = THREE.LinearSRGBColorSpace
    texture.mapping = THREE.EquirectangularReflectionMapping
    return texture
  }

  function floatEnvironmentTexture() {
    const texture = new THREE.DataTexture(
      new Float32Array([0.5, 0.25, 1, 1]),
      1,
      1,
      THREE.RGBAFormat,
      THREE.FloatType,
    )
    texture.colorSpace = THREE.LinearSRGBColorSpace
    texture.mapping = THREE.EquirectangularReflectionMapping
    texture.needsUpdate = true
    return texture
  }

  function halfFloatEnvironmentTexture() {
    const texture = new THREE.DataTexture(
      new Uint16Array([0x3800, 0x3400, 0x3c00, 0x3c00]),
      1,
      1,
      THREE.RGBAFormat,
      THREE.HalfFloatType,
    )
    texture.colorSpace = THREE.LinearSRGBColorSpace
    texture.mapping = THREE.EquirectangularReflectionMapping
    texture.needsUpdate = true
    return texture
  }

  function renderEnvironment(kind, texture) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const material = new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0.25 })
    if (kind === 'scene') {
      scene.environment = texture
      scene.environmentIntensity = 2.5
    } else if (kind === 'reflectionProbe') {
      scene.userData.headlessThreeRenderer = {
        reflectionProbe: { texture, intensity: 2.5 },
      }
    } else {
      material.envMap = texture
      material.envMapIntensity = 2.5
    }
    scene.add(new THREE.Mesh(
      new THREE.SphereGeometry(1, 32, 16),
      material,
    ))
    return renderRgba(scene, makeCamera(), {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    })
  }

  for (const kind of ['scene', 'reflectionProbe', 'materialEnvMap']) {
    const byteRender = renderEnvironment(kind, byteEnvironmentTexture())
    for (const [label, makeTexture] of [
      ['FloatType', floatEnvironmentTexture],
      ['HalfFloatType', halfFloatEnvironmentTexture],
    ]) {
      const floatRender = renderEnvironment(kind, makeTexture())
      const diff = meanAbsDiff(byteRender, floatRender)
      assert.ok(diff < 3, `${kind} ${label} environment should match equivalent linear RGBA8 IBL (diff=${diff.toFixed(3)})`)
    }
  }
})

test('float raw environment textures honor premultiplyAlpha before IBL upload', () => {
  const scene = new THREE.Scene()

  const floatEnvironment = new THREE.DataTexture(
    new Float32Array([0.5, 0.25, 1, 0.5]),
    1,
    1,
    THREE.RGBAFormat,
    THREE.FloatType,
  )
  floatEnvironment.mapping = THREE.EquirectangularReflectionMapping
  floatEnvironment.premultiplyAlpha = true
  floatEnvironment.needsUpdate = true
  scene.environment = floatEnvironment

  const extractedFloat = extractEnvironmentMap(scene)
  assert.ok(extractedFloat)
  const floatData = new Float32Array(extractedFloat.data.buffer, extractedFloat.data.byteOffset, extractedFloat.data.byteLength / 4)
  assert.ok(Math.abs(floatData[0] - 0.25) < 0.001, `FloatType premultiplied red should be 0.25 (${floatData[0]})`)
  assert.ok(Math.abs(floatData[1] - 0.125) < 0.001, `FloatType premultiplied green should be 0.125 (${floatData[1]})`)
  assert.ok(Math.abs(floatData[2] - 0.5) < 0.001, `FloatType premultiplied blue should be 0.5 (${floatData[2]})`)
  assert.ok(Math.abs(floatData[3] - 0.5) < 0.001, `FloatType alpha should remain 0.5 (${floatData[3]})`)

  const halfEnvironment = new THREE.DataTexture(
    new Uint16Array([0x3800, 0x3400, 0x3c00, 0x3800]),
    1,
    1,
    THREE.RGBAFormat,
    THREE.HalfFloatType,
  )
  halfEnvironment.mapping = THREE.EquirectangularReflectionMapping
  halfEnvironment.premultiplyAlpha = true
  halfEnvironment.needsUpdate = true
  scene.environment = halfEnvironment

  const extractedHalf = extractEnvironmentMap(scene)
  assert.ok(extractedHalf)
  const halfData = new Uint16Array(extractedHalf.data.buffer, extractedHalf.data.byteOffset, extractedHalf.data.byteLength / 2)
  assert.ok(Math.abs(halfFloatToNumber(halfData[0]) - 0.25) < 0.001, `HalfFloatType premultiplied red should be 0.25 (${halfFloatToNumber(halfData[0])})`)
  assert.ok(Math.abs(halfFloatToNumber(halfData[1]) - 0.125) < 0.001, `HalfFloatType premultiplied green should be 0.125 (${halfFloatToNumber(halfData[1])})`)
  assert.ok(Math.abs(halfFloatToNumber(halfData[2]) - 0.5) < 0.001, `HalfFloatType premultiplied blue should be 0.5 (${halfFloatToNumber(halfData[2])})`)
  assert.ok(Math.abs(halfFloatToNumber(halfData[3]) - 0.5) < 0.001, `HalfFloatType alpha should remain 0.5 (${halfFloatToNumber(halfData[3])})`)

  const floatLuminanceAlpha = new THREE.DataTexture(
    new Float32Array([0.5, 0.5]),
    1,
    1,
    LuminanceAlphaFormat,
    THREE.FloatType,
  )
  floatLuminanceAlpha.mapping = THREE.EquirectangularReflectionMapping
  floatLuminanceAlpha.premultiplyAlpha = true
  floatLuminanceAlpha.needsUpdate = true
  scene.environment = floatLuminanceAlpha

  const extractedFloatLuminanceAlpha = extractEnvironmentMap(scene)
  assert.ok(extractedFloatLuminanceAlpha)
  const floatLuminanceAlphaData = new Float32Array(
    extractedFloatLuminanceAlpha.data.buffer,
    extractedFloatLuminanceAlpha.data.byteOffset,
    extractedFloatLuminanceAlpha.data.byteLength / 4,
  )
  assert.ok(Math.abs(floatLuminanceAlphaData[0] - 0.25) < 0.001, `FloatType luminance-alpha premultiplied red should be 0.25 (${floatLuminanceAlphaData[0]})`)
  assert.ok(Math.abs(floatLuminanceAlphaData[1] - 0.25) < 0.001, `FloatType luminance-alpha premultiplied green should be 0.25 (${floatLuminanceAlphaData[1]})`)
  assert.ok(Math.abs(floatLuminanceAlphaData[2] - 0.25) < 0.001, `FloatType luminance-alpha premultiplied blue should be 0.25 (${floatLuminanceAlphaData[2]})`)
  assert.ok(Math.abs(floatLuminanceAlphaData[3] - 0.5) < 0.001, `FloatType luminance-alpha alpha should remain 0.5 (${floatLuminanceAlphaData[3]})`)

  const halfLuminanceAlpha = new THREE.DataTexture(
    new Uint16Array([0x3800, 0x3800]),
    1,
    1,
    LuminanceAlphaFormat,
    THREE.HalfFloatType,
  )
  halfLuminanceAlpha.mapping = THREE.EquirectangularReflectionMapping
  halfLuminanceAlpha.premultiplyAlpha = true
  halfLuminanceAlpha.needsUpdate = true
  scene.environment = halfLuminanceAlpha

  const extractedHalfLuminanceAlpha = extractEnvironmentMap(scene)
  assert.ok(extractedHalfLuminanceAlpha)
  const halfLuminanceAlphaData = new Uint16Array(
    extractedHalfLuminanceAlpha.data.buffer,
    extractedHalfLuminanceAlpha.data.byteOffset,
    extractedHalfLuminanceAlpha.data.byteLength / 2,
  )
  assert.ok(Math.abs(halfFloatToNumber(halfLuminanceAlphaData[0]) - 0.25) < 0.001, `HalfFloatType luminance-alpha premultiplied red should be 0.25 (${halfFloatToNumber(halfLuminanceAlphaData[0])})`)
  assert.ok(Math.abs(halfFloatToNumber(halfLuminanceAlphaData[1]) - 0.25) < 0.001, `HalfFloatType luminance-alpha premultiplied green should be 0.25 (${halfFloatToNumber(halfLuminanceAlphaData[1])})`)
  assert.ok(Math.abs(halfFloatToNumber(halfLuminanceAlphaData[2]) - 0.25) < 0.001, `HalfFloatType luminance-alpha premultiplied blue should be 0.25 (${halfFloatToNumber(halfLuminanceAlphaData[2])})`)
  assert.ok(Math.abs(halfFloatToNumber(halfLuminanceAlphaData[3]) - 0.5) < 0.001, `HalfFloatType luminance-alpha alpha should remain 0.5 (${halfFloatToNumber(halfLuminanceAlphaData[3])})`)
})

test('raw environment textures honor premultiplyAlpha for IBL', () => {
  function environmentTexture(premultiplyAlpha) {
    const texture = solidTexture(220, 120, 60, 128)
    texture.colorSpace = THREE.LinearSRGBColorSpace
    texture.mapping = THREE.EquirectangularReflectionMapping
    texture.premultiplyAlpha = premultiplyAlpha
    return texture
  }

  function renderEnvironment(kind, premultiplyAlpha) {
    const texture = environmentTexture(premultiplyAlpha)
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const material = new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0.2 })
    if (kind === 'scene') {
      scene.environment = texture
      scene.environmentIntensity = 2.5
    } else if (kind === 'reflectionProbe') {
      scene.userData.headlessThreeRenderer = {
        reflectionProbe: { texture, intensity: 2.5 },
      }
    } else {
      material.envMap = texture
      material.envMapIntensity = 2.5
    }
    scene.add(new THREE.Mesh(new THREE.SphereGeometry(1, 32, 16), material))
    return meanRegion(renderRgba(scene, makeCamera(), {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 64, 64, 24, 24, 40, 40)
  }

  for (const kind of ['scene', 'reflectionProbe', 'materialEnvMap']) {
    const straight = renderEnvironment(kind, false)
    const premultiplied = renderEnvironment(kind, true)
    assert.ok(straight.r > premultiplied.r + 7, `${kind} premultiplyAlpha should reduce red IBL (${straight.r} vs ${premultiplied.r})`)
    assert.ok(straight.g > premultiplied.g + 9, `${kind} premultiplyAlpha should reduce green IBL (${straight.g} vs ${premultiplied.g})`)
  }
})

test('packed raw environment textures unpack for IBL', () => {
  const cases = [
    ['UnsignedShort4444Type', THREE.UnsignedShort4444Type, THREE.RGBAFormat, new Uint16Array([0x84ff]), [136, 68, 255]],
    ['UnsignedShort5551Type', THREE.UnsignedShort5551Type, THREE.RGBAFormat, new Uint16Array([0x823f]), [132, 66, 255]],
    ['UnsignedInt5999Type', THREE.UnsignedInt5999Type, THREE.RGBFormat, new Uint32Array([packRgb9E5(0.5, 0.25, 1)]), [128, 64, 255]],
    ['UnsignedInt101111Type', UnsignedInt101111Type, THREE.RGBFormat, new Uint32Array([packR11G11B10F(0x380, 0x340, 0x1e0)]), [128, 64, 255]],
  ]

  function byteEnvironmentTexture([r, g, b]) {
    const texture = solidTexture(r, g, b)
    texture.colorSpace = THREE.LinearSRGBColorSpace
    texture.mapping = THREE.EquirectangularReflectionMapping
    return texture
  }

  function packedEnvironmentTexture(type, format, data) {
    const texture = new THREE.DataTexture(data, 1, 1, format, type)
    texture.colorSpace = THREE.LinearSRGBColorSpace
    texture.mapping = THREE.EquirectangularReflectionMapping
    texture.needsUpdate = true
    return texture
  }

  function renderEnvironment(kind, texture) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const material = new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0.25 })
    if (kind === 'scene') {
      scene.environment = texture
      scene.environmentIntensity = 2.5
    } else if (kind === 'reflectionProbe') {
      scene.userData.headlessThreeRenderer = {
        reflectionProbe: { texture, intensity: 2.5 },
      }
    } else {
      material.envMap = texture
      material.envMapIntensity = 2.5
    }
    scene.add(new THREE.Mesh(
      new THREE.SphereGeometry(1, 32, 16),
      material,
    ))
    return renderRgba(scene, makeCamera(), {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    })
  }

  for (const [label, type, format, data, equivalentByteColor] of cases) {
    for (const kind of ['scene', 'reflectionProbe', 'materialEnvMap']) {
      const byteRender = renderEnvironment(kind, byteEnvironmentTexture(equivalentByteColor))
      const packedRender = renderEnvironment(kind, packedEnvironmentTexture(type, format, data))
      const diff = meanAbsDiff(byteRender, packedRender)
      assert.ok(diff < 3, `${kind} ${label} environment should match equivalent linear RGBA8 IBL (diff=${diff.toFixed(3)})`)
    }
  }
})
