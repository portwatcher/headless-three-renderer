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
import { countRegionPixels, makeCamera, meanRegion, renderRgba } from './scenes.test.part-002.mjs'
test('BatchedMesh per-object frustum culling honors geometry bounds', () => {
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  function renderCulling(perObjectFrustumCulled) {
    const source = new THREE.PlaneGeometry(2, 2)
    source.boundingSphere = new THREE.Sphere(new THREE.Vector3(4, 0, 0), 0.1)
    const batched = new THREE.BatchedMesh(
      1,
      source.getAttribute('position').count,
      source.index.count,
      new THREE.MeshBasicMaterial({ color: 0xff0000 }),
    )
    const geometryId = batched.addGeometry(source)
    const instanceId = batched.addInstance(geometryId)
    batched.setMatrixAt(instanceId, new THREE.Matrix4())
    batched.frustumCulled = false
    batched.perObjectFrustumCulled = perObjectFrustumCulled

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(batched)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const culled = renderCulling(true)
  const uncullable = renderCulling(false)
  assert.ok(culled.r < 5 && culled.g < 5 && culled.b < 5, `cached out-of-frustum BatchedMesh bounds should cull the draw (${culled.r}, ${culled.g}, ${culled.b})`)
  assert.ok(uncullable.r > 200, `perObjectFrustumCulled=false should render the oversized batch draw (${uncullable.r})`)
})

test('BatchedMesh per-object frustum culling combines object and instance transforms', () => {
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const source = new THREE.PlaneGeometry(0.5, 0.5)
  const batched = new THREE.BatchedMesh(
    1,
    source.getAttribute('position').count,
    source.index.count,
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  )
  const geometryId = batched.addGeometry(source)
  const instanceId = batched.addInstance(geometryId)
  batched.setMatrixAt(instanceId, new THREE.Matrix4().makeTranslation(-3, 0, 0))
  batched.position.set(3, 0, 0)
  batched.frustumCulled = false
  batched.perObjectFrustumCulled = true

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(batched)

  const rgba = renderRgba(scene, camera, { width: 64, height: 64 })
  const mean = meanRegion(rgba, 64, 64, 28, 28, 36, 36)
  assert.ok(mean.r > mean.g + 150 && mean.r > mean.b + 150, `combined BatchedMesh object and instance transform should keep the draw visible (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('BatchedMesh object frustum culling computes aggregate bounds with per-object culling disabled', () => {
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  function renderBatched(frustumCulled) {
    const source = new THREE.PlaneGeometry(2, 2)
    source.boundingSphere = new THREE.Sphere(new THREE.Vector3(4, 0, 0), 0.1)
    const batched = new THREE.BatchedMesh(
      1,
      source.getAttribute('position').count,
      source.index.count,
      new THREE.MeshBasicMaterial({ color: 0xff0000 }),
    )
    const geometryId = batched.addGeometry(source)
    const instanceId = batched.addInstance(geometryId)
    batched.setMatrixAt(instanceId, new THREE.Matrix4())
    batched.perObjectFrustumCulled = false
    batched.frustumCulled = frustumCulled

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(batched)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const topLevelCulled = renderBatched(true)
  const uncullable = renderBatched(false)
  assert.ok(topLevelCulled.r < 5 && topLevelCulled.g < 5 && topLevelCulled.b < 5, `BatchedMesh aggregate bounds should cull before per-object expansion (${topLevelCulled.r}, ${topLevelCulled.g}, ${topLevelCulled.b})`)
  assert.ok(uncullable.r > 100, `frustumCulled=false should bypass aggregate BatchedMesh culling (${uncullable.r})`)
})

test('renderable object frustum culling honors geometry bounds and frustumCulled opt-out', () => {
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  function renderObject(object, frustumCulled = true) {
    object.frustumCulled = frustumCulled
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(object)
    return renderRgba(scene, camera, { width: 64, height: 64 })
  }

  function visiblePixels(rgba) {
    return countRegionPixels(rgba, 64, 64, 12, 12, 52, 52, (r, g, b) => r > 180 || g > 180 || b > 180)
  }

  const cases = [
    ['Mesh', () => {
      const geometry = new THREE.PlaneGeometry(1.2, 1.2)
      geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(4, 0, 0), 0.1)
      return new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: 0xff0000 }))
    }],
    ['Mesh object boundingSphere', () => {
      const mesh = new THREE.Mesh(
        new THREE.PlaneGeometry(1.2, 1.2),
        new THREE.MeshBasicMaterial({ color: 0xff0000 }),
      )
      mesh.boundingSphere = new THREE.Sphere(new THREE.Vector3(4, 0, 0), 0.1)
      return mesh
    }],
    ['Line', () => {
      const geometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-0.6, 0, 0),
        new THREE.Vector3(0.6, 0, 0),
      ])
      geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(4, 0, 0), 0.1)
      return new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: 0xffffff }))
    }],
    ['Points', () => {
      const geometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, 0),
      ])
      geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(4, 0, 0), 0.1)
      return new THREE.Points(geometry, new THREE.PointsMaterial({ color: 0xffffff, size: 20, sizeAttenuation: false }))
    }],
  ]

  for (const [label, createObject] of cases) {
    const culledPixels = visiblePixels(renderObject(createObject(), true))
    const uncullablePixels = visiblePixels(renderObject(createObject(), false))
    assert.equal(culledPixels, 0, `${label} bounding sphere outside the frustum should cull the centered geometry (${culledPixels})`)
    assert.ok(uncullablePixels > 2, `${label} frustumCulled=false should render centered geometry (${uncullablePixels})`)
  }
})

test('Points frustum culling accounts for rendered billboard size', () => {
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  function renderPoint(frustumCulled = true) {
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([1.25, 0, 0]), 3))
    geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(1.25, 0, 0), 0)

    const points = new THREE.Points(
      geometry,
      new THREE.PointsMaterial({ color: 0xffffff, size: 48, sizeAttenuation: false }),
    )
    points.frustumCulled = frustumCulled

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(points)
    return renderRgba(scene, camera, { width: 64, height: 64 })
  }

  function visiblePixels(rgba) {
    return countRegionPixels(rgba, 64, 64, 0, 0, 64, 64, (r, g, b) => r > 180 || g > 180 || b > 180)
  }

  const culledPixels = visiblePixels(renderPoint(true))
  const uncullablePixels = visiblePixels(renderPoint(false))
  assert.ok(culledPixels > 20, `point billboard should remain visible when its expanded quad intersects the frustum (${culledPixels})`)
  assert.ok(uncullablePixels > 20, `frustumCulled=false control point should remain visible (${uncullablePixels})`)
})

test('invalid renderable object frustumCulled values fail clearly', () => {
  const scene = new THREE.Scene()
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(1, 1),
    new THREE.MeshBasicMaterial({ color: 0xffffff }),
  )
  mesh.frustumCulled = 'yes'
  scene.add(mesh)

  assert.throws(
    () => renderRgba(scene, makeCamera(), { width: 32, height: 32 }),
    /object\.frustumCulled must be a boolean/i,
  )
})

test('invalid BatchedMesh perObjectFrustumCulled values fail clearly', () => {
  const camera = makeCamera()
  const source = new THREE.PlaneGeometry(1, 1)
  const batched = new THREE.BatchedMesh(
    1,
    source.getAttribute('position').count,
    source.index.count,
    new THREE.MeshBasicMaterial({ color: 0xffffff }),
  )
  const geometryId = batched.addGeometry(source)
  batched.addInstance(geometryId)
  batched.perObjectFrustumCulled = 'yes'

  const scene = new THREE.Scene()
  scene.add(batched)

  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /THREE\.BatchedMesh\.perObjectFrustumCulled must be a boolean/i,
  )
})

test('malformed BatchedMesh culling bounds fail clearly', () => {
  const camera = makeCamera()
  const source = new THREE.PlaneGeometry(1, 1)

  const cases = [
    ['container', 'sphere', /THREE\.BatchedMesh\._geometryInfo\[0\]\.boundingSphere must be a THREE\.Sphere-like object/i],
    ['center', { center: { x: Number.NaN, y: 0, z: 0 }, radius: 1 }, /THREE\.BatchedMesh\._geometryInfo\[0\]\.boundingSphere\.center must be a finite Vector3-like value/i],
    ['radius', { center: new THREE.Vector3(0, 0, 0), radius: -1 }, /THREE\.BatchedMesh\._geometryInfo\[0\]\.boundingSphere\.radius must be non-negative/i],
  ]

  for (const [label, boundingSphere, pattern] of cases) {
    const batched = new THREE.BatchedMesh(
      1,
      source.getAttribute('position').count,
      source.index.count,
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    )
    const geometryId = batched.addGeometry(source)
    batched.addInstance(geometryId)
    batched._geometryInfo[geometryId].boundingSphere = boundingSphere

    const scene = new THREE.Scene()
    scene.add(batched)

    assert.throws(
      () => renderRgba(scene, camera, { width: 32, height: 32 }),
      pattern,
      label,
    )
  }
})

test('malformed BatchedMesh geometry ranges fail clearly', () => {
  const camera = makeCamera()
  const source = new THREE.PlaneGeometry(1, 1)
  const makeScene = (range) => {
    const batched = new THREE.BatchedMesh(
      1,
      source.getAttribute('position').count,
      source.index.count,
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    )
    const geometryId = batched.addGeometry(source)
    batched.addInstance(geometryId)
    batched.getGeometryRangeAt = () => range
    const scene = new THREE.Scene()
    scene.add(batched)
    return scene
  }

  const cases = [
    ['missing range', null, /THREE\.BatchedMesh geometry range 0 is not readable/i],
    ['active flag', { start: 0, count: 6, active: 'yes' }, /THREE\.BatchedMesh\._geometryInfo\[0\]\.active must be a boolean/i],
    ['negative start', { start: -1, count: 6 }, /THREE\.BatchedMesh\._geometryInfo\[0\]\.start must be a non-negative integer/i],
    ['non-integer count', { start: 0, count: 1.5 }, /THREE\.BatchedMesh\._geometryInfo\[0\]\.count must be a non-negative integer/i],
    ['start past packed geometry', { start: 7, count: 0 }, /THREE\.BatchedMesh\._geometryInfo\[0\]\.start must be less than or equal to packed geometry count \(6\)/i],
    ['count past packed geometry', { start: 4, count: 3 }, /THREE\.BatchedMesh\._geometryInfo\[0\]\.count must fit within packed geometry count \(6\) from start 4/i],
  ]

  for (const [label, range, pattern] of cases) {
    assert.throws(
      () => renderRgba(makeScene(range), camera, { width: 32, height: 32 }),
      pattern,
      label,
    )
  }
})
