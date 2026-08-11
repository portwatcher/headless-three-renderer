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
import { Renderer, renderToTarget, test } from './scenes.test.part-001.mjs'
import { makeCamera, meanRegion, renderRgba } from './scenes.test.part-002.mjs'
test('ShaderMaterial custom WGSL honors premultipliedAlpha output', () => {
  function renderCustom(premultipliedAlpha) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const material = new THREE.ShaderMaterial({
      blending: THREE.NoBlending,
      premultipliedAlpha,
      transparent: true,
    })
    material.userData.headlessThreeRenderer = {
      fragmentWgsl: 'return vec4<f32>(0.0, 1.0, 0.0, alpha * 0.5);',
    }
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 20, 20, 44, 44)
  }

  const straight = renderCustom(false)
  const premultiplied = renderCustom(true)
  assert.ok(straight.g > premultiplied.g + 60, `premultiplied custom WGSL should reduce raw green output (${straight.g} vs ${premultiplied.g})`)
  assert.ok(premultiplied.g > 60, `premultiplied custom WGSL output should retain source contribution (${premultiplied.g})`)
  assert.ok(premultiplied.a > 120 && premultiplied.a < 140, `premultiplied custom WGSL should preserve returned alpha (${premultiplied.a})`)
})

test('material onBeforeCompile customizations fail clearly', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const material = new THREE.MeshBasicMaterial({ color: 0xffffff })
  material.onBeforeCompile = (shader) => {
    shader.fragmentShader = shader.fragmentShader.replace('vec4', 'vec4')
  }
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  assert.throws(
    () => renderRgba(scene, camera, { width: 64, height: 64 }),
    /onBeforeCompile.*fragmentWgsl/i,
  )
})

test('material onBeforeCompile can opt into custom WGSL fragment output', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const material = new THREE.MeshBasicMaterial({ color: 0xffffff })
  material.onBeforeCompile = (shader) => {
    shader.fragmentShader = shader.fragmentShader.replace('vec4', 'vec4')
  }
  material.userData.headlessThreeRenderer = {
    fragmentWgsl: 'return vec4<f32>(1.0, 0.0, 1.0, alpha);',
  }
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  assert.ok(mean.r > mean.g + 40, `onBeforeCompile WGSL override should render magenta red (${mean.r} vs ${mean.g})`)
  assert.ok(mean.b > mean.g + 40, `onBeforeCompile WGSL override should render magenta blue (${mean.b} vs ${mean.g})`)
})

test('renderToTarget populates a target-like object with raw RGBA', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.1, 0.1, 0.1)
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial({ color: 0x00ffaa })))

  const target = { texture: {} }
  const out = renderToTarget(scene, makeCamera(), target, { width: 64, height: 32 })
  assert.equal(out, target)
  assert.equal(target.width, 64)
  assert.equal(target.height, 32)
  assert.equal(target.data.length, 64 * 32 * 4)
  assert.equal(target.texture.image.data, target.data)
  assert.equal(target.texture.source.data, target.texture.image)
  assert.equal(target.texture.source.data.data, target.data)
  assert.equal(target.texture.source.data.width, 64)
  assert.equal(target.texture.source.data.height, 32)
  assert.equal(target.texture.needsUpdate, true)

  const singleTextureArrayTarget = { texture: [{}] }
  renderToTarget(scene, makeCamera(), singleTextureArrayTarget, { width: 32, height: 16 })
  assert.equal(singleTextureArrayTarget.width, 32)
  assert.equal(singleTextureArrayTarget.height, 16)
  assert.equal(singleTextureArrayTarget.texture[0].image.data, singleTextureArrayTarget.data)
  assert.equal(singleTextureArrayTarget.texture[0].source.data, singleTextureArrayTarget.texture[0].image)
  assert.equal(singleTextureArrayTarget.texture[0].needsUpdate, true)

  const texturesTarget = { textures: [{}] }
  renderToTarget(scene, makeCamera(), texturesTarget, { width: 16, height: 8 })
  assert.equal(texturesTarget.width, 16)
  assert.equal(texturesTarget.height, 8)
  assert.equal(texturesTarget.textures[0].image.data, texturesTarget.data)
  assert.equal(texturesTarget.textures[0].source.data, texturesTarget.textures[0].image)
  assert.equal(texturesTarget.textures[0].needsUpdate, true)

  const singleAttachmentMrtTarget = { isWebGLMultipleRenderTargets: true, textures: [{}] }
  renderToTarget(scene, makeCamera(), singleAttachmentMrtTarget, { width: 8, height: 4 })
  assert.equal(singleAttachmentMrtTarget.width, 8)
  assert.equal(singleAttachmentMrtTarget.height, 4)
  assert.equal(singleAttachmentMrtTarget.textures[0].image.data, singleAttachmentMrtTarget.data)
  assert.equal(singleAttachmentMrtTarget.textures[0].source.data, singleAttachmentMrtTarget.textures[0].image)
  assert.equal(singleAttachmentMrtTarget.textures[0].needsUpdate, true)
})

test('render targets write actual Three.js RenderTarget classes', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  function assertThreeTargetWrite(label, target, width, height, initialTextureVersion, initialSourceVersion) {
    assert.equal(target.width, width, `${label} should preserve target width`)
    assert.equal(target.height, height, `${label} should preserve target height`)
    assert.ok(Buffer.isBuffer(target.data), `${label} should receive Buffer-backed target data`)
    assert.equal(target.data.length, width * height * 4, `${label} should receive RGBA8 data`)
    assert.strictEqual(target.texture, target.textures[0], `${label} texture should remain the primary texture`)
    assert.strictEqual(target.texture.source.data, target.texture.image, `${label} source.data should remain the texture image object`)
    assert.strictEqual(target.texture.image.data, target.data, `${label} image should reference target.data`)
    assert.strictEqual(target.texture.source.data.data, target.data, `${label} source image should reference target.data`)
    assert.equal(target.texture.image.width, width, `${label} image width should update`)
    assert.equal(target.texture.image.height, height, `${label} image height should update`)
    assert.equal(target.texture.image.depth, 1, `${label} image depth should keep Three.js target shape`)
    assert.ok(target.texture.version > initialTextureVersion, `${label} texture version should advance`)
    assert.ok(target.texture.source.version > initialSourceVersion, `${label} texture source version should advance`)

    const center = ((Math.floor(height / 2) * width) + Math.floor(width / 2)) * 4
    assert.ok(
      target.data[center] > target.data[center + 1] + 80 && target.data[center] > target.data[center + 2] + 80,
      `${label} should capture the red mesh (${target.data[center]}, ${target.data[center + 1]}, ${target.data[center + 2]})`,
    )

    const readback = Buffer.alloc(target.data.length)
    new Renderer().readRenderTargetPixels(target, 0, 0, width, height, readback)
    assert.deepEqual(readback, target.data, `${label} should be readable through Renderer.readRenderTargetPixels`)
  }

  const directTarget = new THREE.RenderTarget(17, 9)
  const directTextureVersion = directTarget.texture.version
  const directSourceVersion = directTarget.texture.source.version
  assert.strictEqual(
    renderToTarget(scene, camera, directTarget, { outputColorSpace: THREE.LinearSRGBColorSpace }),
    directTarget,
  )
  assertThreeTargetWrite('THREE.RenderTarget renderToTarget', directTarget, 17, 9, directTextureVersion, directSourceVersion)

  const optionsTarget = new THREE.RenderTarget(11, 6)
  const optionsTextureVersion = optionsTarget.texture.version
  const optionsSourceVersion = optionsTarget.texture.source.version
  const optionsReturned = new Renderer().render(scene, camera, {
    target: optionsTarget,
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })
  assert.strictEqual(optionsReturned, optionsTarget.data)
  assertThreeTargetWrite('THREE.RenderTarget options.target', optionsTarget, 11, 6, optionsTextureVersion, optionsSourceVersion)

  const rendererTarget = new THREE.WebGLRenderTarget(13, 7)
  const rendererTextureVersion = rendererTarget.texture.version
  const rendererSourceVersion = rendererTarget.texture.source.version
  const renderer = new Renderer()
  renderer.setRenderTarget(rendererTarget)
  const rendererReturned = renderer.render(scene, camera, { outputColorSpace: THREE.LinearSRGBColorSpace })
  assert.strictEqual(rendererReturned, rendererTarget.data)
  assert.strictEqual(renderer.getRenderTarget(), rendererTarget)
  assertThreeTargetWrite('THREE.WebGLRenderTarget Renderer.setRenderTarget', rendererTarget, 13, 7, rendererTextureVersion, rendererSourceVersion)
  renderer.setRenderTarget(null)
})

test('Renderer readRenderTargetPixels reads stored target color data', async () => {
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
  const target = { texture: {} }
  renderer.renderToTarget(scene, camera, target, {
    width: 16,
    height: 8,
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  const full = Buffer.alloc(16 * 8 * 4)
  renderer.readRenderTargetPixels(target, 0, 0, 16, 8, full)
  assert.deepEqual(full, target.data)

  const rect = Buffer.alloc(4 * 3 * 4)
  renderer.readRenderTargetPixels(target, 2, 1, 4, 3, rect)
  const expectedRect = Buffer.alloc(rect.length)
  for (let row = 0; row < 3; row += 1) {
    const sourceStart = (((1 + row) * 16) + 2) * 4
    target.data.copy(expectedRect, row * 4 * 4, sourceStart, sourceStart + 4 * 4)
  }
  assert.deepEqual(rect, expectedRect)

  const asyncBuffer = new Uint8Array(16 * 8 * 4)
  const returned = await renderer.readRenderTargetPixelsAsync(target, 0, 0, 16, 8, asyncBuffer)
  assert.strictEqual(returned, asyncBuffer)
  assert.deepEqual(Buffer.from(asyncBuffer), target.data)

  const allocatedAsync = await renderer.readRenderTargetPixelsAsync(target, 2, 1, 4, 3)
  assert.ok(Buffer.isBuffer(allocatedAsync), 'async readback without a buffer should allocate a Buffer for Buffer-backed targets')
  assert.deepEqual(allocatedAsync, expectedRect)

  const mrtTarget = {
    isWebGLMultipleRenderTargets: true,
    textures: [
      {},
      { format: THREE.RGFormat, type: THREE.FloatType, userData: { headlessThreeRenderer: { renderMode: 'color' } } },
    ],
  }
  renderer.renderToTarget(scene, camera, mrtTarget, {
    width: 16,
    height: 8,
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })
  const typedAttachment = mrtTarget.textures[1].image.data
  const typedBuffer = new Float32Array(16 * 8 * 2)
  renderer.readRenderTargetPixels(mrtTarget, 0, 0, 16, 8, typedBuffer, undefined, 1)
  assert.deepEqual([...typedBuffer], [...typedAttachment])

  const allocatedTyped = await renderer.readRenderTargetPixelsAsync(mrtTarget, 0, 0, 16, 8, undefined, undefined, 1)
  assert.ok(allocatedTyped instanceof Float32Array, 'async MRT readback should allocate the attachment typed-array class')
  assert.deepEqual([...allocatedTyped], [...typedAttachment])

  const commonSignatureTyped = await renderer.readRenderTargetPixelsAsync(mrtTarget, 0, 0, 16, 8, 1)
  assert.ok(commonSignatureTyped instanceof Float32Array, 'async common-renderer readback signature should allocate the attachment typed-array class')
  assert.deepEqual([...commonSignatureTyped], [...typedAttachment])

  assert.throws(
    () => renderer.readRenderTargetPixels('target', 0, 0, 1, 1, Buffer.alloc(4)),
    /Renderer\.readRenderTargetPixels target must be a target-like object/i,
  )
  assert.throws(
    () => renderer.readRenderTargetPixels({ texture: {} }, 0, 0, 1, 1, Buffer.alloc(4)),
    /target has no readable color data/i,
  )
  assert.throws(
    () => renderer.readRenderTargetPixels({
      texture: new THREE.CompressedTexture([], 1, 1, THREE.RGBAFormat),
      width: 1,
      height: 1,
      data: Buffer.alloc(4),
    }, 0, 0, 1, 1, Buffer.alloc(4)),
    /target color texture uses a compressed texture/i,
  )
  await assert.rejects(
    () => renderer.readRenderTargetPixelsAsync({
      texture: { format: THREE.RGBA_S3TC_DXT5_Format },
      width: 1,
      height: 1,
      data: Buffer.alloc(4),
    }, 0, 0, 1, 1),
    /target color texture format uses a compressed texture format/i,
  )
  assert.throws(
    () => renderer.readRenderTargetPixels(target, 15, 0, 2, 1, Buffer.alloc(8)),
    /requested read bounds are out of range/i,
  )
  assert.throws(
    () => renderer.readRenderTargetPixels(target, 0, 0, 1.5, 1, Buffer.alloc(8)),
    /x, y, width, and height must be integers/i,
  )
  assert.throws(
    () => renderer.readRenderTargetPixels(target, 0, 0, 2, 2, Buffer.alloc(4)),
    /buffer length is too small/i,
  )
  assert.throws(
    () => renderer.readRenderTargetPixels(target, 0, 0, 1, 1, {}, undefined, -1),
    /textureIndex must be a non-negative integer/i,
  )
  await assert.rejects(
    () => renderer.readRenderTargetPixelsAsync(target, 0, 0, 1, 1, {}),
    /buffer must be a mutable typed array or Buffer/i,
  )
  await assert.rejects(
    () => renderer.readRenderTargetPixelsAsync(target, 0, 0, 1, 1, null),
    /buffer must be a mutable typed array or Buffer/i,
  )
  await assert.rejects(
    () => renderer.readRenderTargetPixelsAsync(target, 0, 0, 1, 1, -1),
    /textureIndex must be a non-negative integer/i,
  )
})
