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
import { AlphaFormat, LuminanceAlphaFormat, LuminanceFormat, Renderer, UnsignedInt101111Type, renderToTarget, test } from './scenes.test.part-001.mjs'
import { halfFloatToNumber, unsignedFloatToNumber } from './scenes.test.part-002.mjs'
test('renderToTarget color textures honor typed readback requests', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const center = ((32 * 64) + 32) * 4
  const options = { width: 64, height: 64, outputColorSpace: THREE.LinearSRGBColorSpace }

  const redTarget = { texture: { format: THREE.RedFormat } }
  renderToTarget(scene, camera, redTarget, options)
  const redData = redTarget.texture.image.data
  const redCenter = (32 * 64) + 32
  assert.ok(redData instanceof Uint8Array, 'RedFormat color target should receive Uint8Array data')
  assert.equal(redData.length, 64 * 64, 'RedFormat color target should receive one channel per pixel')
  assert.ok(redData[redCenter] > 128, `RedFormat red channel should keep the source red (${redData[redCenter]})`)

  const luminanceTarget = { texture: { format: LuminanceFormat } }
  renderToTarget(scene, camera, luminanceTarget, options)
  const luminanceData = luminanceTarget.texture.image.data
  assert.ok(luminanceData instanceof Uint8Array, 'LuminanceFormat color target should receive Uint8Array data')
  assert.equal(luminanceData.length, 64 * 64, 'LuminanceFormat color target should receive one channel per pixel')
  assert.ok(luminanceData[redCenter] > 128, `LuminanceFormat luminance should keep the source red channel (${luminanceData[redCenter]})`)

  const alphaScene = new THREE.Scene()
  alphaScene.background = new THREE.Color(0, 0, 1)
  alphaScene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0x0000ff }),
  ))
  const alphaTarget = { texture: { format: AlphaFormat } }
  renderToTarget(alphaScene, camera, alphaTarget, options)
  const alphaData = alphaTarget.texture.image.data
  assert.ok(alphaData instanceof Uint8Array, 'AlphaFormat color target should receive Uint8Array data')
  assert.equal(alphaData.length, 64 * 64, 'AlphaFormat color target should receive one channel per pixel')
  assert.ok(alphaData[redCenter] > 250, `AlphaFormat should extract opaque alpha instead of red (${alphaData[redCenter]})`)

  const luminanceAlphaScene = new THREE.Scene()
  luminanceAlphaScene.background = { r: 1, g: 0, b: 0, a: 0.5 }
  const luminanceAlphaTarget = { texture: { format: LuminanceAlphaFormat, type: THREE.FloatType } }
  renderToTarget(luminanceAlphaScene, camera, luminanceAlphaTarget, options)
  const luminanceAlphaData = luminanceAlphaTarget.texture.image.data
  const luminanceAlphaCenter = redCenter * 2
  assert.ok(luminanceAlphaData instanceof Float32Array, 'LuminanceAlphaFormat FloatType color target should receive Float32Array data')
  assert.equal(luminanceAlphaData.length, 64 * 64 * 2, 'LuminanceAlphaFormat color target should receive two channels per pixel')
  assert.ok(luminanceAlphaData[luminanceAlphaCenter] > 0.99, `LuminanceAlphaFormat luminance should keep red clear color (${luminanceAlphaData[luminanceAlphaCenter]})`)
  assert.ok(Math.abs(luminanceAlphaData[luminanceAlphaCenter + 1] - 0.5) < 0.01, `LuminanceAlphaFormat alpha should keep clear alpha (${luminanceAlphaData[luminanceAlphaCenter + 1]})`)

  const floatTarget = { texture: { type: THREE.FloatType } }
  renderToTarget(scene, camera, floatTarget, options)
  const floatData = floatTarget.texture.image.data
  assert.ok(floatData instanceof Float32Array, 'FloatType color target should receive Float32Array data')
  assert.ok(Buffer.isBuffer(floatTarget.data), 'target.data should remain raw RGBA8 for compatibility')
  assert.ok(floatData[center] > 0.5, `FloatType red channel should be normalized (${floatData[center]})`)
  assert.ok(floatData[center + 1] < 0.05, `FloatType green channel should stay near zero (${floatData[center + 1]})`)
  assert.ok(floatData[center + 3] > 0.99, `FloatType alpha channel should stay opaque (${floatData[center + 3]})`)

  const byteTarget = { texture: { type: THREE.ByteType } }
  renderToTarget(scene, camera, byteTarget, options)
  const byteData = byteTarget.texture.image.data
  assert.ok(byteData instanceof Int8Array, 'ByteType color target should receive Int8Array data')
  assert.ok(byteData[center] > 63, `ByteType red channel should be normalized (${byteData[center]})`)
  assert.ok(byteData[center + 1] < 8, `ByteType green channel should stay near zero (${byteData[center + 1]})`)
  assert.ok(byteData[center + 3] > 120, `ByteType alpha channel should stay opaque (${byteData[center + 3]})`)

  const rgFloatTarget = { texture: { format: THREE.RGFormat, type: THREE.FloatType } }
  renderToTarget(scene, camera, rgFloatTarget, options)
  const rgFloatData = rgFloatTarget.texture.image.data
  const rgCenter = ((32 * 64) + 32) * 2
  assert.ok(rgFloatData instanceof Float32Array, 'RGFormat FloatType color target should receive Float32Array data')
  assert.equal(rgFloatData.length, 64 * 64 * 2, 'RGFormat color target should receive two channels per pixel')
  assert.ok(rgFloatData[rgCenter] > 0.5, `RGFormat FloatType red channel should be normalized (${rgFloatData[rgCenter]})`)
  assert.ok(rgFloatData[rgCenter + 1] < 0.05, `RGFormat FloatType green channel should stay near zero (${rgFloatData[rgCenter + 1]})`)

  for (const [label, format, channels] of [
    ['RedIntegerFormat', THREE.RedIntegerFormat, 1],
    ['RGIntegerFormat', THREE.RGIntegerFormat, 2],
    ['RGBIntegerFormat', THREE.RGBIntegerFormat, 3],
    ['RGBAIntegerFormat', THREE.RGBAIntegerFormat, 4],
  ]) {
    const target = { texture: { format, type: THREE.UnsignedShortType } }
    renderToTarget(scene, camera, target, options)
    const data = target.texture.image.data
    const pixel = redCenter * channels
    assert.ok(data instanceof Uint16Array, `${label} color target should receive Uint16Array data`)
    assert.equal(data.length, 64 * 64 * channels, `${label} color target should receive ${channels} channel(s) per pixel`)
    assert.ok(data[pixel] > 0x8000, `${label} red channel should be normalized (${data[pixel]})`)
    if (channels > 1) assert.ok(data[pixel + 1] < 0x1000, `${label} green channel should stay near zero (${data[pixel + 1]})`)
    if (channels > 2) assert.ok(data[pixel + 2] < 0x1000, `${label} blue channel should stay near zero (${data[pixel + 2]})`)
    if (channels > 3) assert.ok(data[pixel + 3] > 0xff00, `${label} alpha channel should stay opaque (${data[pixel + 3]})`)
  }

  const ushortTarget = { texture: { type: THREE.UnsignedShortType } }
  renderToTarget(scene, camera, ushortTarget, options)
  const ushortData = ushortTarget.texture.image.data
  assert.ok(ushortData instanceof Uint16Array, 'UnsignedShortType color target should receive Uint16Array data')
  assert.ok(ushortData[center] > 0x8000, `UnsignedShortType red channel should be normalized (${ushortData[center]})`)
  assert.ok(ushortData[center + 1] < 0x1000, `UnsignedShortType green channel should stay near zero (${ushortData[center + 1]})`)
  assert.ok(ushortData[center + 3] > 0xff00, `UnsignedShortType alpha channel should stay opaque (${ushortData[center + 3]})`)

  const shortTarget = { texture: { type: THREE.ShortType } }
  renderToTarget(scene, camera, shortTarget, options)
  const shortData = shortTarget.texture.image.data
  assert.ok(shortData instanceof Int16Array, 'ShortType color target should receive Int16Array data')
  assert.ok(shortData[center] > 0x4000, `ShortType red channel should be normalized (${shortData[center]})`)
  assert.ok(shortData[center + 1] < 0x1000, `ShortType green channel should stay near zero (${shortData[center + 1]})`)
  assert.ok(shortData[center + 3] > 0x7f00, `ShortType alpha channel should stay opaque (${shortData[center + 3]})`)

  const rgbUshortTarget = { texture: { format: THREE.RGBFormat, type: THREE.UnsignedShortType } }
  renderToTarget(scene, camera, rgbUshortTarget, options)
  const rgbUshortData = rgbUshortTarget.texture.image.data
  const rgbCenter = ((32 * 64) + 32) * 3
  assert.ok(rgbUshortData instanceof Uint16Array, 'RGBFormat UnsignedShortType color target should receive Uint16Array data')
  assert.equal(rgbUshortData.length, 64 * 64 * 3, 'RGBFormat color target should receive three channels per pixel')
  assert.ok(rgbUshortData[rgbCenter] > 0x8000, `RGBFormat red channel should be normalized (${rgbUshortData[rgbCenter]})`)
  assert.ok(rgbUshortData[rgbCenter + 1] < 0x1000, `RGBFormat green channel should stay near zero (${rgbUshortData[rgbCenter + 1]})`)
  assert.ok(rgbUshortData[rgbCenter + 2] < 0x1000, `RGBFormat blue channel should stay near zero (${rgbUshortData[rgbCenter + 2]})`)

  const packed4444Target = { texture: { type: THREE.UnsignedShort4444Type } }
  renderToTarget(scene, camera, packed4444Target, options)
  const packed4444Data = packed4444Target.texture.image.data
  const packed4444 = packed4444Data[redCenter]
  assert.ok(packed4444Data instanceof Uint16Array, 'UnsignedShort4444Type color target should receive Uint16Array data')
  assert.ok(((packed4444 >> 12) & 0xf) > 7, `UnsignedShort4444Type red channel should be packed (${packed4444.toString(16)})`)
  assert.ok(((packed4444 >> 8) & 0xf) < 2, `UnsignedShort4444Type green channel should stay near zero (${packed4444.toString(16)})`)
  assert.ok(((packed4444 >> 4) & 0xf) < 2, `UnsignedShort4444Type blue channel should stay near zero (${packed4444.toString(16)})`)
  assert.equal(packed4444 & 0xf, 0xf, `UnsignedShort4444Type alpha channel should stay opaque (${packed4444.toString(16)})`)

  const packed5551Target = { texture: { type: THREE.UnsignedShort5551Type } }
  renderToTarget(scene, camera, packed5551Target, options)
  const packed5551Data = packed5551Target.texture.image.data
  const packed5551 = packed5551Data[redCenter]
  assert.ok(packed5551Data instanceof Uint16Array, 'UnsignedShort5551Type color target should receive Uint16Array data')
  assert.ok(((packed5551 >> 11) & 0x1f) > 15, `UnsignedShort5551Type red channel should be packed (${packed5551.toString(16)})`)
  assert.ok(((packed5551 >> 6) & 0x1f) < 2, `UnsignedShort5551Type green channel should stay near zero (${packed5551.toString(16)})`)
  assert.ok(((packed5551 >> 1) & 0x1f) < 2, `UnsignedShort5551Type blue channel should stay near zero (${packed5551.toString(16)})`)
  assert.equal(packed5551 & 0x1, 1, `UnsignedShort5551Type alpha channel should stay opaque (${packed5551.toString(16)})`)

  const rgb9e5Target = { texture: { type: THREE.UnsignedInt5999Type } }
  renderToTarget(scene, camera, rgb9e5Target, options)
  const rgb9e5Data = rgb9e5Target.texture.image.data
  const rgb9e5 = rgb9e5Data[redCenter]
  const rgb9e5Scale = 2 ** (((rgb9e5 >>> 27) & 0x1f) - 24)
  const rgb9e5Red = (rgb9e5 & 0x1ff) * rgb9e5Scale
  const rgb9e5Green = ((rgb9e5 >>> 9) & 0x1ff) * rgb9e5Scale
  const rgb9e5Blue = ((rgb9e5 >>> 18) & 0x1ff) * rgb9e5Scale
  assert.ok(rgb9e5Data instanceof Uint32Array, 'UnsignedInt5999Type color target should receive Uint32Array data')
  assert.ok(rgb9e5Red > 0.5, `UnsignedInt5999Type red channel should be packed (${rgb9e5Red})`)
  assert.ok(rgb9e5Green < 0.05, `UnsignedInt5999Type green channel should stay near zero (${rgb9e5Green})`)
  assert.ok(rgb9e5Blue < 0.05, `UnsignedInt5999Type blue channel should stay near zero (${rgb9e5Blue})`)

  const r11Target = { texture: { type: UnsignedInt101111Type } }
  renderToTarget(scene, camera, r11Target, options)
  const r11Data = r11Target.texture.image.data
  const r11 = r11Data[redCenter]
  const r11Red = unsignedFloatToNumber(r11 & 0x7ff, 6)
  const r11Green = unsignedFloatToNumber((r11 >>> 11) & 0x7ff, 6)
  const r11Blue = unsignedFloatToNumber((r11 >>> 22) & 0x3ff, 5)
  assert.ok(r11Data instanceof Uint32Array, 'UnsignedInt101111Type color target should receive Uint32Array data')
  assert.ok(r11Red > 0.5, `UnsignedInt101111Type red channel should be packed (${r11Red})`)
  assert.ok(r11Green < 0.05, `UnsignedInt101111Type green channel should stay near zero (${r11Green})`)
  assert.ok(r11Blue < 0.05, `UnsignedInt101111Type blue channel should stay near zero (${r11Blue})`)

  const uintTarget = { texture: { type: THREE.UnsignedIntType } }
  renderToTarget(scene, camera, uintTarget, options)
  const uintData = uintTarget.texture.image.data
  assert.ok(uintData instanceof Uint32Array, 'UnsignedIntType color target should receive Uint32Array data')
  assert.ok(uintData[center] > 0x80000000, `UnsignedIntType red channel should be normalized (${uintData[center]})`)
  assert.ok(uintData[center + 1] < 0x10000000, `UnsignedIntType green channel should stay near zero (${uintData[center + 1]})`)
  assert.ok(uintData[center + 3] > 0xff000000, `UnsignedIntType alpha channel should stay opaque (${uintData[center + 3]})`)

  const intTarget = { texture: { type: THREE.IntType } }
  renderToTarget(scene, camera, intTarget, options)
  const intData = intTarget.texture.image.data
  assert.ok(intData instanceof Int32Array, 'IntType color target should receive Int32Array data')
  assert.ok(intData[center] > 0x40000000, `IntType red channel should be normalized (${intData[center]})`)
  assert.ok(intData[center + 1] < 0x10000000, `IntType green channel should stay near zero (${intData[center + 1]})`)
  assert.ok(intData[center + 3] > 0x7f000000, `IntType alpha channel should stay opaque (${intData[center + 3]})`)

  const halfTarget = { texture: { type: THREE.HalfFloatType } }
  renderToTarget(scene, camera, halfTarget, options)
  const halfData = halfTarget.texture.image.data
  assert.ok(halfData instanceof Uint16Array, 'HalfFloatType color target should receive Uint16Array half-float data')
  const halfRed = halfFloatToNumber(halfData[center])
  const halfGreen = halfFloatToNumber(halfData[center + 1])
  const halfAlpha = halfFloatToNumber(halfData[center + 3])
  assert.ok(halfRed > 0.5, `HalfFloatType red channel should be normalized (${halfRed})`)
  assert.ok(halfGreen < 0.05, `HalfFloatType green channel should stay near zero (${halfGreen})`)
  assert.ok(halfAlpha > 0.99, `HalfFloatType alpha channel should stay opaque (${halfAlpha})`)
})

test('Three.js exporters read targets through the WebGLRenderer marker path', async () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const renderer = new Renderer()
  const renderOptions = { outputColorSpace: THREE.LinearSRGBColorSpace }

  const exrTarget = new THREE.WebGLRenderTarget(16, 16, {
    format: THREE.RGBAFormat,
    type: THREE.HalfFloatType,
  })
  exrTarget.texture.colorSpace = THREE.LinearSRGBColorSpace
  renderToTarget(scene, camera, exrTarget, renderOptions)
  assert.ok(exrTarget.texture.image.data instanceof Uint16Array, 'EXR target should receive half-float render data')

  const exr = await new EXRExporter().parse(renderer, exrTarget, {
    compression: NO_COMPRESSION,
    type: THREE.HalfFloatType,
  })
  assert.ok(exr instanceof Uint8Array, 'EXRExporter should return Uint8Array data')
  assert.ok(exr.length > 16 * 16 * 8, `EXR output should include header and pixel data (${exr.length})`)
  assert.deepEqual(
    Array.from(exr.subarray(0, 4)),
    [0x76, 0x2f, 0x31, 0x01],
    'EXRExporter output should start with the OpenEXR magic number',
  )

  const ktxTarget = new THREE.WebGLRenderTarget(16, 16, {
    format: THREE.RGBAFormat,
    type: THREE.UnsignedByteType,
  })
  ktxTarget.texture.colorSpace = THREE.SRGBColorSpace
  renderToTarget(scene, camera, ktxTarget, renderOptions)
  assert.ok(ktxTarget.texture.image.data instanceof Uint8Array, 'KTX2 target should receive byte render data')

  const ktx = await new KTX2Exporter().parse(renderer, ktxTarget)
  assert.ok(ktx instanceof Uint8Array, 'KTX2Exporter should return Uint8Array data')
  assert.ok(ktx.length > 16 * 16 * 4, `KTX2 output should include header and pixel data (${ktx.length})`)
  assert.deepEqual(
    Array.from(ktx.subarray(0, 12)),
    [0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a],
    'KTX2Exporter output should start with the KTX2 identifier',
  )
})
