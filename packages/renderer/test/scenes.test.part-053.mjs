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
import { countRegionPixels, meanRegion, renderRgba, rgbaTexture, setTextureMatrixOffset, solidTexture } from './scenes.test.part-002.mjs'
test('SpriteMaterial alphaMap honors explicit texture matrices', () => {
  const alphaMap = rgbaTexture([
    255, 0, 255, 255,
    255, 255, 255, 255,
  ], 2, 1)
  alphaMap.magFilter = THREE.NearestFilter
  alphaMap.minFilter = THREE.NearestFilter
  setTextureMatrixOffset(alphaMap, 0.5)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 1)
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    color: 0x00ff00,
    alphaMap,
    alphaTest: 0.5,
  }))
  sprite.scale.set(2, 2, 1)
  scene.add(sprite)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 18, 28, 26, 36)
  assert.ok(mean.g > mean.b + 40, `explicit sprite alphaMap matrix should shift left sprite UVs into the opaque texel (${mean.g} vs ${mean.b})`)
})

test('SpriteMaterial alphaMap honors horizontal and vertical repeat wrapping', () => {
  const renderer = new Renderer()

  function renderWithWrapping({ wrapS = THREE.ClampToEdgeWrapping, wrapT = THREE.ClampToEdgeWrapping, vertical = false }) {
    const alphaMap = vertical
      ? rgbaTexture([
        255, 255, 255, 255,
        255, 255, 255, 255,
        255, 0, 255, 255,
        255, 0, 255, 255,
      ], 2, 2)
      : rgbaTexture([
        255, 255, 255, 255,
        255, 0, 255, 255,
        255, 255, 255, 255,
        255, 0, 255, 255,
      ], 2, 2)
    alphaMap.wrapS = wrapS
    alphaMap.wrapT = wrapT
    alphaMap.offset.set(vertical ? 0 : 1, vertical ? 1 : 0)
    alphaMap.magFilter = THREE.NearestFilter
    alphaMap.minFilter = THREE.NearestFilter

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 1)
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
      color: 0x00ff00,
      alphaMap,
      alphaTest: 0.5,
    }))
    sprite.scale.set(2, 2, 1)
    scene.add(sprite)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    const rgba = renderer.render(scene, camera, { width: 64, height: 64, format: 'rgba' })
    return vertical
      ? meanRegion(rgba, 64, 64, 28, 38, 36, 46)
      : meanRegion(rgba, 64, 64, 18, 28, 26, 36)
  }

  const clamped = renderWithWrapping({ wrapS: THREE.ClampToEdgeWrapping })
  const repeated = renderWithWrapping({ wrapS: THREE.RepeatWrapping })
  const mirrored = renderWithWrapping({ wrapS: THREE.MirroredRepeatWrapping })
  assert.ok(clamped.b > clamped.g + 40, `clamped sprite alphaMap U coordinates should discard against the blue background (${clamped.b} vs ${clamped.g})`)
  assert.ok(repeated.g > repeated.b + 40, `repeated sprite alphaMap U coordinates should wrap to the opaque texel (${repeated.g} vs ${repeated.b})`)
  assert.ok(mirrored.b > mirrored.g + 40, `mirrored sprite alphaMap U coordinates should reflect to the transparent texel (${mirrored.b} vs ${mirrored.g})`)

  const clampedVertical = renderWithWrapping({ wrapT: THREE.ClampToEdgeWrapping, vertical: true })
  const repeatedVertical = renderWithWrapping({ wrapT: THREE.RepeatWrapping, vertical: true })
  const mirroredVertical = renderWithWrapping({ wrapT: THREE.MirroredRepeatWrapping, vertical: true })
  assert.ok(clampedVertical.b > clampedVertical.g + 40, `clamped sprite alphaMap V coordinates should discard against the blue background (${clampedVertical.b} vs ${clampedVertical.g})`)
  assert.ok(repeatedVertical.g > repeatedVertical.b + 40, `repeated sprite alphaMap V coordinates should wrap to the opaque texel (${repeatedVertical.g} vs ${repeatedVertical.b})`)
  assert.ok(mirroredVertical.b > mirroredVertical.g + 40, `mirrored sprite alphaMap V coordinates should reflect to the transparent texel (${mirroredVertical.b} vs ${mirroredVertical.g})`)
})

test('SpriteMaterial and PointsMaterial alphaMap decode sRGB colorSpace before alpha testing', () => {
  function frontCamera() {
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return camera
  }

  function renderBillboard(kind, colorSpace) {
    const alphaMap = solidTexture(255, 128, 255, 255)
    alphaMap.colorSpace = colorSpace

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 1)
    const materialProps = {
      alphaMap,
      alphaTest: 0.3,
      color: 0x00ff00,
    }

    if (kind === 'sprite') {
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial(materialProps))
      sprite.scale.set(2, 2, 1)
      scene.add(sprite)
    } else {
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))
      scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({
        ...materialProps,
        size: 48,
        sizeAttenuation: false,
      })))
    }

    return meanRegion(renderRgba(scene, frontCamera(), { width: 64, height: 64 }), 64, 64, 20, 20, 44, 44)
  }

  const spriteSrgb = renderBillboard('sprite', THREE.SRGBColorSpace)
  const spriteLinear = renderBillboard('sprite', THREE.LinearSRGBColorSpace)
  assert.ok(spriteSrgb.b > 240 && spriteSrgb.g < 5, `decoded sRGB sprite alphaMap should fall below alphaTest and show blue background (${spriteSrgb.b} vs ${spriteSrgb.g})`)
  assert.ok(spriteLinear.g > spriteSrgb.g + 80, `linear sprite alphaMap should stay visible after alpha testing (${spriteLinear.g} vs ${spriteSrgb.g})`)
  assert.ok(spriteLinear.b < spriteSrgb.b - 40, `linear sprite alphaMap should reduce blue background coverage (${spriteLinear.b} vs ${spriteSrgb.b})`)

  const pointSrgb = renderBillboard('points', THREE.SRGBColorSpace)
  const pointLinear = renderBillboard('points', THREE.LinearSRGBColorSpace)
  assert.ok(pointSrgb.b > pointSrgb.g + 80, `decoded sRGB point alphaMap should fall below alphaTest and show blue background (${pointSrgb.b} vs ${pointSrgb.g})`)
  assert.ok(pointLinear.g > pointLinear.b + 40, `linear point alphaMap should stay visible after alpha testing (${pointLinear.g} vs ${pointLinear.b})`)
})

test('SpriteMaterial and PointsMaterial maps honor nearest and linear filters', () => {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  function renderBillboard(kind, slot, filter) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, slot === 'alphaMap' ? 1 : 0)
    const materialProps = {
      color: slot === 'alphaMap' ? 0x00ff00 : 0xffffff,
      transparent: false,
    }

    if (slot === 'map') {
      const map = rgbaTexture([
        255, 0, 0, 255,
        0, 255, 0, 255,
      ], 2, 1)
      map.magFilter = filter
      map.minFilter = filter
      map.offset.set(-0.05, 0)
      materialProps.map = map
    } else {
      const alphaMap = rgbaTexture([
        255, 0, 255, 255,
        255, 255, 255, 255,
      ], 2, 1)
      alphaMap.magFilter = filter
      alphaMap.minFilter = filter
      alphaMap.offset.set(-0.05, 0)
      materialProps.alphaMap = alphaMap
      materialProps.alphaTest = 0.3
    }

    if (kind === 'sprite') {
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial(materialProps))
      sprite.scale.set(2, 2, 1)
      scene.add(sprite)
      return meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 30, 30, 34, 34)
    }

    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))
    scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({
      ...materialProps,
      size: 48,
      sizeAttenuation: false,
    })))
    return meanRegion(renderRgba(scene, camera, { width: 96, height: 96 }), 96, 96, 46, 46, 50, 50)
  }

  for (const kind of ['sprite', 'points']) {
    const nearestMap = renderBillboard(kind, 'map', THREE.NearestFilter)
    const linearMap = renderBillboard(kind, 'map', THREE.LinearFilter)
    assert.ok(nearestMap.r > nearestMap.g + 100, `${kind} NearestFilter color map should choose the red texel (${nearestMap.r} vs ${nearestMap.g})`)
    assert.ok(linearMap.g > nearestMap.g + 80, `${kind} LinearFilter color map should blend in the green texel (${linearMap.g} vs ${nearestMap.g})`)
    assert.ok(nearestMap.r > linearMap.r + 25, `${kind} NearestFilter color map should preserve a stronger red texel (${nearestMap.r} vs ${linearMap.r})`)

    const nearestAlpha = renderBillboard(kind, 'alphaMap', THREE.NearestFilter)
    const linearAlpha = renderBillboard(kind, 'alphaMap', THREE.LinearFilter)
    assert.ok(nearestAlpha.b > nearestAlpha.g + 100, `${kind} NearestFilter alphaMap should choose the transparent texel (${nearestAlpha.b} vs ${nearestAlpha.g})`)
    assert.ok(linearAlpha.g > linearAlpha.b + 80, `${kind} LinearFilter alphaMap should blend in enough opacity to pass alphaTest (${linearAlpha.g} vs ${linearAlpha.b})`)
    assert.ok(linearAlpha.g > nearestAlpha.g + 120, `${kind} LinearFilter alphaMap should keep the billboard visible (${linearAlpha.g} vs ${nearestAlpha.g})`)
  }
})

test('SpriteMaterial and PointsMaterial alphaHash produce main-pass stochastic coverage', () => {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  function renderBillboard(kind, alphaHash) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const materialProps = {
      alphaHash,
      color: 0xffffff,
      opacity: alphaHash ? 0.35 : 1,
    }

    if (kind === 'sprite') {
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial(materialProps))
      sprite.scale.set(1.2, 1.2, 1)
      scene.add(sprite)
    } else {
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))
      scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({
        ...materialProps,
        size: 48,
        sizeAttenuation: false,
      })))
    }

    return renderRgba(scene, camera, { width: 64, height: 64 })
  }

  for (const kind of ['sprite', 'points']) {
    const opaque = renderBillboard(kind, false)
    const hashed = renderBillboard(kind, true)
    const visiblePixel = (r, g, b) => r > 20 || g > 20 || b > 20
    const opaquePixels = countRegionPixels(opaque, 64, 64, 16, 16, 48, 48, visiblePixel)
    const hashedPixels = countRegionPixels(hashed, 64, 64, 16, 16, 48, 48, visiblePixel)

    assert.ok(opaquePixels > 700, `${kind} opaque billboard should fill the sampled region (${opaquePixels})`)
    assert.ok(hashedPixels > 80, `${kind} alphaHash billboard should retain some visible pixels (${hashedPixels})`)
    assert.ok(hashedPixels < opaquePixels - 180, `${kind} alphaHash billboard should discard visible pixels (${hashedPixels} vs ${opaquePixels})`)
  }
})

test('SpriteMaterial and PointsMaterial alphaToCoverage produce 4x-MSAA main-pass coverage', () => {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  function renderBillboard(kind, alphaToCoverage, sampleCount = 4) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const materialProps = {
      alphaToCoverage,
      color: 0xffffff,
      opacity: 0.5,
      transparent: false,
    }

    if (kind === 'sprite') {
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial(materialProps))
      sprite.scale.set(1.2, 1.2, 1)
      scene.add(sprite)
    } else {
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))
      scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({
        ...materialProps,
        size: 48,
        sizeAttenuation: false,
      })))
    }

    return meanRegion(renderRgba(scene, camera, {
      width: 64,
      height: 64,
      sampleCount,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 64, 64, 24, 24, 40, 40)
  }

  for (const kind of ['sprite', 'points']) {
    const noCoverage = renderBillboard(kind, false)
    const coverage = renderBillboard(kind, true)
    const singleSample = renderBillboard(kind, true, 1)

    assert.ok(noCoverage.r > 170, `${kind} non-A2C path should keep bright RGB despite opacity alpha (${noCoverage.r})`)
    assert.ok(Math.abs(singleSample.r - noCoverage.r) < 5, `${kind} single-sample alphaToCoverage should not alter RGB coverage (${singleSample.r} vs ${noCoverage.r})`)
    assert.ok(coverage.r > 30 && coverage.r < noCoverage.r - 80, `${kind} 4x alphaToCoverage should resolve partial RGB coverage (${coverage.r} vs ${noCoverage.r})`)
  }
})
