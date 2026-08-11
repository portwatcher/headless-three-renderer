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
import { AlphaFormat, LuminanceAlphaFormat, LuminanceFormat, extractBackgroundTexture, extractTextureData, test } from './scenes.test.part-001.mjs'
import { meanRegion, renderRgba, rgbaTexture } from './scenes.test.part-002.mjs'
test('one-, luminance-, alpha-, two-channel, and luminance-alpha raw DataTexture maps and backgrounds expand before texture upload', () => {
  function extractMap(map) {
    map.needsUpdate = true
    const info = extractTextureData(new THREE.MeshBasicMaterial({ map }))
    assert.ok(info)
    return {
      r: info.data[0],
      g: info.data[1],
      b: info.data[2],
      a: info.data[3],
    }
  }

  function extractBackground(background) {
    background.needsUpdate = true
    const info = extractBackgroundTexture(background)
    assert.ok(info)
    return {
      r: info.data[0],
      g: info.data[1],
      b: info.data[2],
      a: info.data[3],
    }
  }

  const redMap = new THREE.DataTexture(new Uint8Array([220]), 1, 1, THREE.RedFormat)
  const red = extractMap(redMap)
  assert.ok(red.r > 180 && red.g > 180 && red.b > 180, `one-channel raw texture should expand to grayscale (${red.r}, ${red.g}, ${red.b})`)

  const luminanceMap = new THREE.DataTexture(new Uint8Array([180]), 1, 1, LuminanceFormat)
  const luminance = extractMap(luminanceMap)
  assert.deepEqual(
    [luminance.r, luminance.g, luminance.b, luminance.a],
    [180, 180, 180, 255],
    'LuminanceFormat raw texture should copy the single channel into RGB with opaque alpha',
  )

  const alphaMap = new THREE.DataTexture(new Uint8Array([96]), 1, 1, AlphaFormat)
  const alpha = extractMap(alphaMap)
  assert.deepEqual(
    [alpha.r, alpha.g, alpha.b, alpha.a],
    [96, 96, 96, 96],
    'AlphaFormat raw texture should copy the single channel into RGB and alpha',
  )

  const redBackground = extractBackground(new THREE.DataTexture(new Uint8Array([220]), 1, 1, THREE.RedFormat))
  assert.ok(
    redBackground.r > 180 && redBackground.g > 180 && redBackground.b > 180,
    `one-channel raw background should expand to grayscale (${redBackground.r}, ${redBackground.g}, ${redBackground.b})`,
  )

  const luminanceBackground = extractBackground(new THREE.DataTexture(new Uint8Array([180]), 1, 1, LuminanceFormat))
  assert.deepEqual(
    [luminanceBackground.r, luminanceBackground.g, luminanceBackground.b, luminanceBackground.a],
    [180, 180, 180, 255],
    'LuminanceFormat raw background should copy the single channel into RGB with opaque alpha',
  )

  const alphaBackground = extractBackground(new THREE.DataTexture(new Uint8Array([96]), 1, 1, AlphaFormat))
  assert.deepEqual(
    [alphaBackground.r, alphaBackground.g, alphaBackground.b, alphaBackground.a],
    [96, 96, 96, 96],
    'AlphaFormat raw background should copy the single channel into RGB and alpha',
  )

  const luminanceAlphaMap = new THREE.DataTexture(new Uint8Array([220, 255]), 1, 1, LuminanceAlphaFormat)
  const luminanceAlpha = extractMap(luminanceAlphaMap)
  assert.ok(
    luminanceAlpha.r > 180 && luminanceAlpha.g > 180 && luminanceAlpha.b > 180,
    `luminance-alpha raw texture should expand luminance to RGB (${luminanceAlpha.r}, ${luminanceAlpha.g}, ${luminanceAlpha.b})`,
  )

  const luminanceAlphaBackground = extractBackground(new THREE.DataTexture(new Uint8Array([220, 255]), 1, 1, LuminanceAlphaFormat))
  assert.ok(
    luminanceAlphaBackground.r > 180 && luminanceAlphaBackground.g > 180 && luminanceAlphaBackground.b > 180,
    `luminance-alpha raw background should expand luminance to RGB (${luminanceAlphaBackground.r}, ${luminanceAlphaBackground.g}, ${luminanceAlphaBackground.b})`,
  )

  const rgMap = new THREE.DataTexture(new Uint8Array([230, 24]), 1, 1, THREE.RGFormat)
  const rg = extractMap(rgMap)
  assert.ok(rg.r > 190, `two-channel raw texture should preserve red (${rg.r})`)
  assert.ok(rg.g < 80, `two-channel raw texture should preserve green (${rg.g})`)
  assert.ok(rg.b < 40, `two-channel raw texture should leave blue empty (${rg.b})`)

  const rgBackground = extractBackground(new THREE.DataTexture(new Uint8Array([230, 24]), 1, 1, THREE.RGFormat))
  assert.ok(rgBackground.r > 190, `two-channel raw background should preserve red (${rgBackground.r})`)
  assert.ok(rgBackground.g < 80, `two-channel raw background should preserve green (${rgBackground.g})`)
  assert.ok(rgBackground.b < 40, `two-channel raw background should leave blue empty (${rgBackground.b})`)
})

test('HalfFloatType raw DataTexture maps decode for material and background textures', () => {
  function halfRgbaTexture() {
    const texture = new THREE.DataTexture(
      new Uint16Array([0x3800, 0x3400, 0x3c00, 0x3c00]),
      1,
      1,
      THREE.RGBAFormat,
      THREE.HalfFloatType,
    )
    texture.needsUpdate = true
    return texture
  }

  function renderTexture(kind) {
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    if (kind === 'material') {
      scene.background = new THREE.Color(0, 0, 0)
      scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map: halfRgbaTexture() })))
    } else {
      scene.background = halfRgbaTexture()
    }
    return meanRegion(
      renderRgba(scene, camera, { width: 64, height: 64, outputColorSpace: THREE.LinearSRGBColorSpace }),
      64,
      64,
      24,
      24,
      40,
      40,
    )
  }

  for (const kind of ['material', 'background']) {
    const mean = renderTexture(kind)
    assert.ok(mean.r > 105 && mean.r < 150, `${kind} half-float red should decode near 0.5 (${mean.r})`)
    assert.ok(mean.g > 45 && mean.g < 100, `${kind} half-float green should decode near 0.25 (${mean.g})`)
    assert.ok(mean.b > 180, `${kind} half-float blue should decode near 1.0 (${mean.b})`)
  }
})

test('FloatType raw DataTexture maps decode for material and background textures', () => {
  function floatRgbaTexture() {
    const texture = new THREE.DataTexture(
      new Float32Array([0.5, 0.25, 1, 1]),
      1,
      1,
      THREE.RGBAFormat,
      THREE.FloatType,
    )
    texture.needsUpdate = true
    return texture
  }

  function renderTexture(kind) {
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    if (kind === 'material') {
      scene.background = new THREE.Color(0, 0, 0)
      scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map: floatRgbaTexture() })))
    } else {
      scene.background = floatRgbaTexture()
    }
    return meanRegion(
      renderRgba(scene, camera, { width: 64, height: 64, outputColorSpace: THREE.LinearSRGBColorSpace }),
      64,
      64,
      24,
      24,
      40,
      40,
    )
  }

  for (const kind of ['material', 'background']) {
    const mean = renderTexture(kind)
    assert.ok(mean.r > 105 && mean.r < 150, `${kind} float red should decode near 0.5 (${mean.r})`)
    assert.ok(mean.g > 45 && mean.g < 100, `${kind} float green should decode near 0.25 (${mean.g})`)
    assert.ok(mean.b > 180, `${kind} float blue should decode near 1.0 (${mean.b})`)
  }
})

test('normalized unsigned integer raw DataTexture maps decode for material and background textures', () => {
  const cases = [
    [
      'UnsignedShortType',
      THREE.UnsignedShortType,
      () => new Uint16Array([0x8000, 0x4000, 0xffff, 0xffff]),
    ],
    [
      'UnsignedIntType',
      THREE.UnsignedIntType,
      () => new Uint32Array([0x80000000, 0x40000000, 0xffffffff, 0xffffffff]),
    ],
  ]

  function rgbaTexture(type, data) {
    const texture = new THREE.DataTexture(data, 1, 1, THREE.RGBAFormat, type)
    texture.needsUpdate = true
    return texture
  }

  function renderTexture(kind, type, data) {
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    if (kind === 'material') {
      scene.background = new THREE.Color(0, 0, 0)
      scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map: rgbaTexture(type, data) })))
    } else {
      scene.background = rgbaTexture(type, data)
    }
    return meanRegion(
      renderRgba(scene, camera, { width: 64, height: 64, outputColorSpace: THREE.LinearSRGBColorSpace }),
      64,
      64,
      24,
      24,
      40,
      40,
    )
  }

  for (const [name, type, makeData] of cases) {
    for (const kind of ['material', 'background']) {
      const mean = renderTexture(kind, type, makeData())
      assert.ok(mean.r > 105 && mean.r < 150, `${kind} ${name} red should normalize near 0.5 (${mean.r})`)
      assert.ok(mean.g > 45 && mean.g < 100, `${kind} ${name} green should normalize near 0.25 (${mean.g})`)
      assert.ok(mean.b > 180, `${kind} ${name} blue should normalize near 1.0 (${mean.b})`)
    }
  }
})

test('normalized signed integer raw DataTexture maps decode for material and background textures', () => {
  const cases = [
    [
      'ByteType',
      THREE.ByteType,
      () => new Int8Array([64, 32, 127, 127]),
    ],
    [
      'ShortType',
      THREE.ShortType,
      () => new Int16Array([0x4000, 0x2000, 0x7fff, 0x7fff]),
    ],
    [
      'IntType',
      THREE.IntType,
      () => new Int32Array([0x40000000, 0x20000000, 0x7fffffff, 0x7fffffff]),
    ],
  ]

  function rgbaTexture(type, data) {
    const texture = new THREE.DataTexture(data, 1, 1, THREE.RGBAFormat, type)
    texture.needsUpdate = true
    return texture
  }

  function renderTexture(kind, type, data) {
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    if (kind === 'material') {
      scene.background = new THREE.Color(0, 0, 0)
      scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map: rgbaTexture(type, data) })))
    } else {
      scene.background = rgbaTexture(type, data)
    }
    return meanRegion(
      renderRgba(scene, camera, { width: 64, height: 64, outputColorSpace: THREE.LinearSRGBColorSpace }),
      64,
      64,
      24,
      24,
      40,
      40,
    )
  }

  for (const [name, type, makeData] of cases) {
    for (const kind of ['material', 'background']) {
      const mean = renderTexture(kind, type, makeData())
      assert.ok(mean.r > 105 && mean.r < 155, `${kind} ${name} red should normalize near 0.5 (${mean.r})`)
      assert.ok(mean.g > 45 && mean.g < 100, `${kind} ${name} green should normalize near 0.25 (${mean.g})`)
      assert.ok(mean.b > 180, `${kind} ${name} blue should normalize near 1.0 (${mean.b})`)
    }
  }
})
