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
import { meanRegion, rgbaTexture } from './scenes.test.part-002.mjs'
test('reusable renderer reflects mutated mesh material texture and transform state', () => {
  const renderer = new Renderer()
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)

  const texture = rgbaTexture([255, 255, 255, 255], 1, 1)
  const material = new THREE.MeshBasicMaterial({ color: 0xff0000, map: texture })
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(0.75, 0.75), material)
  mesh.position.x = -0.5
  scene.add(mesh)

  const options = { width: 64, height: 64, format: 'rgba' }
  const first = renderer.render(scene, camera, options)
  const firstLeft = meanRegion(first, 64, 64, 8, 24, 28, 40)
  const firstRight = meanRegion(first, 64, 64, 36, 24, 56, 40)

  assert.ok(firstLeft.r > 180, `initial material color should render red on the left (${firstLeft.r}, ${firstLeft.g}, ${firstLeft.b})`)
  assert.ok(firstLeft.g < 40 && firstLeft.b < 40, `initial left region should not include stale green/blue (${firstLeft.r}, ${firstLeft.g}, ${firstLeft.b})`)
  assert.ok(firstRight.r < 20 && firstRight.g < 20 && firstRight.b < 20, `initial right region should remain background (${firstRight.r}, ${firstRight.g}, ${firstRight.b})`)

  texture.image.data.set([0, 255, 0, 255])
  texture.needsUpdate = true
  material.color.set(0xffffff)
  mesh.position.x = 0.5

  const second = renderer.render(scene, camera, options)
  const secondLeft = meanRegion(second, 64, 64, 8, 24, 28, 40)
  const secondRight = meanRegion(second, 64, 64, 36, 24, 56, 40)

  assert.ok(secondLeft.r < 20 && secondLeft.g < 20 && secondLeft.b < 20, `updated transform should clear the previous left region (${secondLeft.r}, ${secondLeft.g}, ${secondLeft.b})`)
  assert.ok(secondRight.g > 180, `updated texture data should render green on the right (${secondRight.r}, ${secondRight.g}, ${secondRight.b})`)
  assert.ok(secondRight.g > secondRight.r + 50 && secondRight.g > secondRight.b + 80, `updated material and texture state should produce a green-dominant right region (${secondRight.r}, ${secondRight.g}, ${secondRight.b})`)

  texture.image.data.set([255, 255, 255, 255])
  texture.needsUpdate = true
  material.color.set(0xff0000)
  mesh.position.x = -0.5

  const third = renderer.render(scene, camera, options)
  const thirdLeft = meanRegion(third, 64, 64, 8, 24, 28, 40)
  const thirdRight = meanRegion(third, 64, 64, 36, 24, 56, 40)

  assert.ok(thirdLeft.r > 180, `dynamic mesh uniform slot should update back to red on the left (${thirdLeft.r}, ${thirdLeft.g}, ${thirdLeft.b})`)
  assert.ok(thirdLeft.g < 40 && thirdLeft.b < 40, `updated uniform slot should not retain prior green texture state (${thirdLeft.r}, ${thirdLeft.g}, ${thirdLeft.b})`)
  assert.ok(thirdRight.r < 20 && thirdRight.g < 20 && thirdRight.b < 20, `updated transform should clear the previous right region (${thirdRight.r}, ${thirdRight.g}, ${thirdRight.b})`)
})

test('reusable renderer reuses cached material texture payload until texture version changes', () => {
  const renderer = new Renderer()
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)

  const sourceData = new Uint8Array([255, 255, 255])
  let textureReads = 0
  const trackedData = new Proxy(sourceData, {
    get(target, property) {
      if (typeof property === 'string' && /^(0|[1-9]\d*)$/.test(property)) {
        textureReads += 1
      }
      return Reflect.get(target, property, target)
    },
    set(target, property, value) {
      return Reflect.set(target, property, value, target)
    },
  })
  const texture = new THREE.DataTexture(trackedData, 1, 1, THREE.RGBFormat)
  texture.needsUpdate = true

  const material = new THREE.MeshBasicMaterial({ color: 0xff0000, map: texture })
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(0.75, 0.75), material)
  mesh.frustumCulled = false
  mesh.position.x = -0.5
  scene.add(mesh)

  const options = {
    width: 64,
    height: 64,
    format: 'rgba',
    outputColorSpace: THREE.LinearSRGBColorSpace,
    sortObjects: false,
  }

  const first = renderer.render(scene, camera, options)
  const firstLeft = meanRegion(first, 64, 64, 8, 24, 28, 40)
  const readsAfterFirstRender = textureReads
  assert.ok(firstLeft.r > 180 && firstLeft.g < 40, `initial cached texture render should be red (${firstLeft.r}, ${firstLeft.g}, ${firstLeft.b})`)
  assert.ok(readsAfterFirstRender > 0, 'initial render should extract texture payload bytes')

  material.color.set(0x00ff00)
  mesh.position.x = 0.5
  const second = renderer.render(scene, camera, options)
  const secondRight = meanRegion(second, 64, 64, 36, 24, 56, 40)
  assert.ok(secondRight.g > secondRight.r + 80, `material and transform animation should remain live while texture payload is cached (${secondRight.r}, ${secondRight.g}, ${secondRight.b})`)
  assert.equal(textureReads, readsAfterFirstRender, 'material and transform animation should reuse cached texture payload extraction')

  sourceData.set([0, 0, 255])
  texture.needsUpdate = true
  material.color.set(0xffffff)
  mesh.position.x = -0.5
  const third = renderer.render(scene, camera, options)
  const thirdLeft = meanRegion(third, 64, 64, 8, 24, 28, 40)
  assert.ok(thirdLeft.b > thirdLeft.r + 120, `texture version changes should render updated blue payload (${thirdLeft.r}, ${thirdLeft.g}, ${thirdLeft.b})`)
  assert.ok(textureReads > readsAfterFirstRender, 'texture version changes should invalidate cached texture payload extraction')
})

test('reusable renderer reuses cached base texture sampler state until texture state changes', () => {
  const renderer = new Renderer()
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)

  const texture = rgbaTexture([255, 255, 255, 255], 1, 1)
  texture.rotation = 0.37
  const material = new THREE.MeshBasicMaterial({ color: 0xff0000, map: texture })
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(0.75, 0.75), material)
  mesh.frustumCulled = false
  scene.add(mesh)

  const options = {
    width: 64,
    height: 64,
    format: 'rgba',
    outputColorSpace: THREE.LinearSRGBColorSpace,
    sortObjects: false,
  }

  const originalCos = Math.cos
  let rotationCosCalls = 0
  Math.cos = (value) => {
    if (value === texture.rotation) {
      rotationCosCalls += 1
    }
    return originalCos(value)
  }
  try {
    const first = renderer.render(scene, camera, options)
    const firstCenter = meanRegion(first, 64, 64, 24, 24, 40, 40)
    const callsAfterFirstRender = rotationCosCalls
    assert.ok(firstCenter.r > 180 && firstCenter.g < 40, `initial mapped material should render red (${firstCenter.r}, ${firstCenter.g}, ${firstCenter.b})`)
    assert.ok(callsAfterFirstRender > 0, 'initial render should compute texture transform state')

    material.color.set(0x00ff00)
    const second = renderer.render(scene, camera, options)
    const secondCenter = meanRegion(second, 64, 64, 24, 24, 40, 40)
    assert.ok(secondCenter.g > secondCenter.r + 80, `material color should remain live while base texture state is cached (${secondCenter.r}, ${secondCenter.g}, ${secondCenter.b})`)
    assert.equal(rotationCosCalls, callsAfterFirstRender, 'material-only animation should reuse cached base texture sampler state')

    texture.rotation = 0.61
    renderer.render(scene, camera, options)
    const callsAfterTransformStateChange = rotationCosCalls
    assert.ok(callsAfterTransformStateChange > callsAfterFirstRender, 'texture transform changes should invalidate cached base texture sampler state')

    texture.wrapS = THREE.RepeatWrapping
    renderer.render(scene, camera, options)
    assert.ok(rotationCosCalls > callsAfterTransformStateChange, 'texture sampler changes should invalidate cached base texture sampler state')
  } finally {
    Math.cos = originalCos
  }
})

test('reusable renderer reuses cached material render state until render state changes', () => {
  const renderer = new Renderer()
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)

  const material = new THREE.MeshBasicMaterial({ color: 0xff0000 })
  material.stencilRef = 7.25
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(0.75, 0.75), material)
  mesh.frustumCulled = false
  scene.add(mesh)

  const options = {
    width: 64,
    height: 64,
    format: 'rgba',
    outputColorSpace: THREE.LinearSRGBColorSpace,
    sortObjects: false,
  }

  const originalTrunc = Math.trunc
  let stencilRefTruncCalls = 0
  Math.trunc = (value) => {
    if (value === material.stencilRef) {
      stencilRefTruncCalls += 1
    }
    return originalTrunc(value)
  }
  try {
    const first = renderer.render(scene, camera, options)
    const firstCenter = meanRegion(first, 64, 64, 24, 24, 40, 40)
    const callsAfterFirstRender = stencilRefTruncCalls
    assert.ok(firstCenter.r > 180 && firstCenter.g < 40, `initial material should render red (${firstCenter.r}, ${firstCenter.g}, ${firstCenter.b})`)
    assert.ok(callsAfterFirstRender > 0, 'initial render should extract material render state')

    material.color.set(0x00ff00)
    const second = renderer.render(scene, camera, options)
    const secondCenter = meanRegion(second, 64, 64, 24, 24, 40, 40)
    assert.ok(secondCenter.g > secondCenter.r + 80, `material color should remain live while render state is cached (${secondCenter.r}, ${secondCenter.g}, ${secondCenter.b})`)
    assert.equal(stencilRefTruncCalls, callsAfterFirstRender, 'material-color animation should reuse cached material render state')

    material.stencilRef = 11.75
    renderer.render(scene, camera, options)
    assert.ok(stencilRefTruncCalls > callsAfterFirstRender, 'render-state changes should invalidate cached material render state')
  } finally {
    Math.trunc = originalTrunc
  }
})

test('reusable renderer reuses cached material scalar feature state until scalar changes', () => {
  const renderer = new Renderer()
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)

  const material = new THREE.MeshBasicMaterial({ color: 0xff0000 })
  material.thickness = 0.42
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(0.75, 0.75), material)
  mesh.frustumCulled = false
  scene.add(mesh)

  const options = {
    width: 64,
    height: 64,
    format: 'rgba',
    outputColorSpace: THREE.LinearSRGBColorSpace,
    sortObjects: false,
  }

  const originalMax = Math.max
  let thicknessMaxCalls = 0
  Math.max = (...values) => {
    if (values[0] === 0 && values[1] === material.thickness) {
      thicknessMaxCalls += 1
    }
    return originalMax(...values)
  }
  try {
    const first = renderer.render(scene, camera, options)
    const firstCenter = meanRegion(first, 64, 64, 24, 24, 40, 40)
    const callsAfterFirstRender = thicknessMaxCalls
    assert.ok(firstCenter.r > 180 && firstCenter.g < 40, `initial scalar-feature material should render red (${firstCenter.r}, ${firstCenter.g}, ${firstCenter.b})`)
    assert.ok(callsAfterFirstRender > 0, 'initial render should extract material scalar feature state')

    material.color.set(0x00ff00)
    const second = renderer.render(scene, camera, options)
    const secondCenter = meanRegion(second, 64, 64, 24, 24, 40, 40)
    assert.ok(secondCenter.g > secondCenter.r + 80, `material color should remain live while scalar feature state is cached (${secondCenter.r}, ${secondCenter.g}, ${secondCenter.b})`)
    assert.equal(thicknessMaxCalls, callsAfterFirstRender, 'material-color animation should reuse cached material scalar feature state')

    material.thickness = 0.73
    renderer.render(scene, camera, options)
    assert.ok(thicknessMaxCalls > callsAfterFirstRender, 'material scalar changes should invalidate cached scalar feature state')
  } finally {
    Math.max = originalMax
  }
})
