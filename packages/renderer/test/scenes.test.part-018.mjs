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
import { MeshGouraudMaterial, MeshPostProcessingMaterial, Renderer, test } from './scenes.test.part-001.mjs'
import { makeCamera, renderRgba, solidTexture } from './scenes.test.part-002.mjs'
test('CurveModifierGPU Flow writes spline textures and fails clearly on material node hooks', () => {
  const curve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-0.8, 0, 0),
    new THREE.Vector3(0, 0.4, 0),
    new THREE.Vector3(0.8, 0, 0),
  ])
  const geometry = new THREE.BoxGeometry(0.5, 0.18, 0.18)
  const material = new THREE.MeshBasicMaterial({ color: 0xff5533 })
  const flow = new GPUFlow(new THREE.Mesh(geometry, material), 2)
  try {
    const texture = flow.splineTexture
    assert.equal(texture.isDataTexture, true)
    assert.equal(texture.image.width, 1024)
    assert.equal(texture.image.height, 8)
    assert.ok(texture.image.data instanceof Uint16Array)
    assert.equal(texture.format, THREE.RGBAFormat)
    assert.equal(texture.type, THREE.HalfFloatType)
    assert.equal(texture.wrapS, THREE.RepeatWrapping)
    assert.equal(texture.wrapY, THREE.RepeatWrapping)
    assert.equal(texture.magFilter, THREE.LinearFilter)
    assert.equal(texture.minFilter, THREE.LinearFilter)

    const { uniforms } = flow
    assert.equal(uniforms.spineTexture, texture)
    assert.equal(uniforms.pathOffset, 0)
    assert.equal(uniforms.pathSegment, 1)
    assert.equal(uniforms.spineOffset, 161)
    assert.equal(uniforms.spineLength, 400)
    assert.equal(uniforms.flow, 1)

    const version = texture.version
    flow.updateCurve(0, curve)
    assert.ok(texture.version > version, 'updating a CurveModifierGPU spline texture should mark it dirty')
    assert.ok(
      Array.from(texture.image.data.slice(0, 64)).some((value) => value !== 0),
      'curve rows should receive packed half-float spline data',
    )

    flow.moveAlongCurve(0.25)

    assert.equal(flow.curveArray[0], curve)
    assert.ok(Math.abs(flow.curveLengthArray[0] - curve.getLength()) < 1e-5)
    assert.ok(Math.abs(flow.uniforms.spineLength - curve.getLength()) < 1e-5)
    assert.equal(flow.uniforms.pathOffset, 0.25)
    assert.notEqual(flow.object3D.material, material)
    assert.equal(flow.object3D.material.positionNode?.isNode, true)
    assert.equal(flow.object3D.material.normalNode?.isNode, true)

    const scene = new THREE.Scene()
    scene.add(flow.object3D)
    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 16, height: 16 }),
      /material node hooks.*normalNode.*positionNode.*TSL.*fragmentWgsl/i,
    )
  } finally {
    geometry.dispose()
    material.dispose()
    flow.splineTexture.dispose()
    flow.object3D.traverse((child) => {
      if (Array.isArray(child.material)) {
        for (const childMaterial of child.material) childMaterial.dispose()
      } else {
        child.material?.dispose?.()
      }
    })
  }
})

test('CSMHelper renders supported cascade visualization geometry', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  const camera = makeCamera()
  camera.updateMatrixWorld()
  const viewCamera = new THREE.PerspectiveCamera(50, 1, 0.01, 100)
  viewCamera.position.set(4, 3, 5)
  viewCamera.lookAt(0, 0, 0)

  const previousLightsFragmentBegin = THREE.ShaderChunk.lights_fragment_begin
  const previousLightsParsBegin = THREE.ShaderChunk.lights_pars_begin
  const csm = new CSM({
    camera,
    parent: scene,
    cascades: 2,
    shadowMapSize: 16,
  })
  let helper

  try {
    csm.update()
    helper = new CSMHelper(csm)
    helper.update()
    scene.add(helper)

    const rgba = renderRgba(scene, viewCamera, { width: 64, height: 64 })
    assert.ok(
      nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.001,
      'CSMHelper should render visible cascade helper geometry',
    )
  } finally {
    if (helper) {
      scene.remove(helper)
      helper.dispose()
    }
    csm.dispose()
    csm.remove()
    THREE.ShaderChunk.lights_fragment_begin = previousLightsFragmentBegin
    THREE.ShaderChunk.lights_pars_begin = previousLightsParsBegin
  }
})

test('examples Line2, LineSegments2, and Wireframe LineMaterial shaders fail clearly', () => {
  const cases = [
    ['Line2', () => {
      const geometry = new LineGeometry()
      geometry.setPositions([-0.5, 0, 0, 0.5, 0, 0])
      return new Line2(geometry, new LineMaterial({ color: 0xff0000, linewidth: 2 }))
    }],
    ['LineSegments2', () => {
      const geometry = new LineSegmentsGeometry()
      geometry.setPositions([-0.5, 0, 0, 0.5, 0, 0])
      return new LineSegments2(geometry, new LineMaterial({ color: 0xff0000, linewidth: 2 }))
    }],
    ['Wireframe', () => {
      const geometry = new WireframeGeometry2(new THREE.BoxGeometry(0.75, 0.75, 0.75))
      return new Wireframe(geometry, new LineMaterial({ color: 0xff0000, linewidth: 2 }))
    }],
  ]

  for (const [label, makeLine] of cases) {
    const line = makeLine()
    const scene = new THREE.Scene()
    scene.add(line)

    try {
      assert.throws(
        () => renderRgba(scene, makeCamera(), { width: 32, height: 32 }),
        /LineMaterial ShaderMaterial.*not translated.*LineBasicMaterial.*LineDashedMaterial/i,
        `${label} should fail with built-in line-material guidance`,
      )
    } finally {
      line.geometry.dispose()
      line.material.dispose()
    }
  }
})

test('examples custom material helpers fail clearly on shader customization boundaries', () => {
  const cases = [
    ...(MeshGouraudMaterial ? [['MeshGouraudMaterial', () => new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1),
      new MeshGouraudMaterial({ color: 0xff5533 }),
    ), /ShaderMaterial.*fragmentWgsl/i]] : []),
    ['LDrawConditionalLineMaterial', () => new THREE.LineSegments(
      new THREE.BufferGeometry().setAttribute(
        'position',
        new THREE.Float32BufferAttribute([-0.5, 0, 0, 0.5, 0, 0], 3),
      ),
      new LDrawConditionalLineMaterial({ color: 0x33ff66 }),
    ), /ShaderMaterial.*fragmentWgsl/i],
    ...(MeshPostProcessingMaterial ? [['MeshPostProcessingMaterial', () => new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1),
      new MeshPostProcessingMaterial({ color: 0x3355ff }),
    ), /onBeforeCompile customizations.*fragmentWgsl/i]] : []),
  ]

  for (const [label, makeObject, pattern] of cases) {
    const scene = new THREE.Scene()
    const object = makeObject()
    scene.add(object)

    try {
      assert.throws(
        () => renderRgba(scene, makeCamera(), { width: 16, height: 16 }),
        pattern,
        `${label} should fail with custom material guidance`,
      )
    } finally {
      object.geometry.dispose()
      object.material.dispose()
    }
  }
})

test('examples LDrawConditionalLineNodeMaterial imports under installed TSL entrypoint', async () => {
  const module = await import('three/examples/jsm/materials/LDrawConditionalLineNodeMaterial.js')
  assert.equal(typeof module.LDrawConditionalLineMaterial, 'function')
})

test('WebGPU Line2, LineSegments2, and Wireframe helpers fail clearly on NodeMaterial paths', () => {
  const cases = [
    ['WebGPU Line2', () => {
      const geometry = new LineGeometry()
      geometry.setPositions([-0.5, 0, 0, 0.5, 0, 0])
      return new WebGPULine2(geometry)
    }],
    ['WebGPU LineSegments2', () => {
      const geometry = new LineSegmentsGeometry()
      geometry.setPositions([-0.5, 0, 0, 0.5, 0, 0])
      return new WebGPULineSegments2(geometry)
    }],
    ['WebGPU Wireframe', () => {
      const geometry = new LineSegmentsGeometry()
      geometry.setPositions([-0.5, 0, 0, 0.5, 0, 0])
      return new WebGPUWireframe(geometry)
    }],
  ]

  for (const [label, makeLine] of cases) {
    const line = makeLine()
    line.computeLineDistances()
    const distanceStart = line.geometry.getAttribute('instanceDistanceStart')
    const distanceEnd = line.geometry.getAttribute('instanceDistanceEnd')
    assert.equal(distanceStart.getX(0), 0)
    assert.equal(distanceEnd.getX(0), 1)

    const scene = new THREE.Scene()
    scene.add(line)

    try {
      assert.throws(
        () => renderRgba(scene, makeCamera(), { width: 32, height: 32 }),
        /NodeMaterial is not supported directly.*fragmentWgsl/i,
        `${label} should fail with NodeMaterial guidance`,
      )
    } finally {
      line.geometry.dispose()
      line.material.dispose()
    }
  }
})

test('ProgressiveLightMap internal shader rewrite fails clearly', () => {
  const renderer = new Renderer()
  const scene = new THREE.Scene()
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(1, 1),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  )
  scene.add(mesh)

  const lightMap = new ProgressiveLightMap(renderer, 16)
  lightMap.addObjectsToLightMap([mesh])

  assert.throws(
    () => lightMap.update(makeCamera(), 1, false),
    /material\.onBeforeCompile customizations.*fragmentWgsl/i,
  )
})

test('ProgressiveLightMapGPU NodeMaterial path fails clearly', () => {
  const renderer = new Renderer()
  const geometry = new THREE.PlaneGeometry(1, 1)
  const material = new THREE.MeshBasicMaterial({ color: 0xff0000 })
  const mesh = new THREE.Mesh(geometry, material)
  const scene = new THREE.Scene()
  scene.add(mesh)

  const lightMap = new ProgressiveLightMapGPU(renderer, 16)
  lightMap.addObjectsToLightMap([mesh])

  try {
    assert.throws(
      () => lightMap.update(makeCamera(), 1, false),
      /NodeMaterial is not supported directly.*fragmentWgsl/i,
    )
  } finally {
    lightMap.dispose()
    renderer.dispose?.()
    geometry.dispose()
    material.dispose()
  }
})

test('ShadowMapViewer depth-unpack shader fails clearly', () => {
  const hadWindow = Object.prototype.hasOwnProperty.call(globalThis, 'window')
  const previousWindow = globalThis.window
  globalThis.window = { innerWidth: 64, innerHeight: 64 }

  try {
    const renderer = new Renderer()
    renderer.setSize(64, 64)
    const light = new THREE.DirectionalLight(0xffffff, 1)
    light.shadow.map = { texture: solidTexture(255, 255, 255) }

    const viewer = new ShadowMapViewer(light)
    assert.throws(
      () => viewer.render(renderer),
      /ShaderMaterial is not supported directly.*fragmentWgsl/i,
    )
  } finally {
    if (hadWindow) {
      globalThis.window = previousWindow
    } else {
      delete globalThis.window
    }
  }
})
