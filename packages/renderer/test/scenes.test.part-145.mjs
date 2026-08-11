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
import { countRegionPixels, makeCamera, meanRegion, renderRgba, rgbaTexture } from './scenes.test.part-002.mjs'
test('LineDashedMaterial alphaMap honors horizontal and vertical repeat wrapping', () => {
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
    alphaMap.magFilter = THREE.NearestFilter
    alphaMap.minFilter = THREE.NearestFilter

    const geom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-1.5, 0, 0),
      new THREE.Vector3(1.5, 0, 0),
    ])
    geom.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([
      vertical ? 0.25 : 1.25, vertical ? 1.25 : 0.25,
      vertical ? 0.25 : 1.25, vertical ? 1.25 : 0.25,
    ]), 2))

    const material = new THREE.LineDashedMaterial({
      alphaTest: 0.5,
      color: 0x00ff00,
      dashSize: 10,
      gapSize: 0,
      linewidth: 8,
      scale: 1,
    })
    material.alphaMap = alphaMap
    const line = new THREE.Line(geom, material)
    line.computeLineDistances()

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 1)
    scene.add(line)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return renderer.render(scene, camera, { width: 96, height: 96, format: 'rgba' })
  }

  const clamped = renderWithWrapping({ wrapS: THREE.ClampToEdgeWrapping })
  const repeated = renderWithWrapping({ wrapS: THREE.RepeatWrapping })
  const mirrored = renderWithWrapping({ wrapS: THREE.MirroredRepeatWrapping })
  const clampedGreen = countRegionPixels(clamped, 96, 96, 0, 0, 96, 96, (r, g, b) => g > b + 40 && g > r + 40)
  const repeatedGreen = countRegionPixels(repeated, 96, 96, 0, 0, 96, 96, (r, g, b) => g > b + 40 && g > r + 40)
  const mirroredGreen = countRegionPixels(mirrored, 96, 96, 0, 0, 96, 96, (r, g, b) => g > b + 40 && g > r + 40)
  assert.ok(clampedGreen < 20, `clamped dashed-line alphaMap U coordinates should discard the line (${clampedGreen})`)
  assert.ok(repeatedGreen > 600, `repeated dashed-line alphaMap U coordinates should wrap to the opaque texel (${repeatedGreen})`)
  assert.ok(mirroredGreen < 20, `mirrored dashed-line alphaMap U coordinates should reflect to the transparent texel (${mirroredGreen})`)

  const clampedVertical = renderWithWrapping({ wrapT: THREE.ClampToEdgeWrapping, vertical: true })
  const repeatedVertical = renderWithWrapping({ wrapT: THREE.RepeatWrapping, vertical: true })
  const mirroredVertical = renderWithWrapping({ wrapT: THREE.MirroredRepeatWrapping, vertical: true })
  const clampedVerticalGreen = countRegionPixels(clampedVertical, 96, 96, 0, 0, 96, 96, (r, g, b) => g > b + 40 && g > r + 40)
  const repeatedVerticalGreen = countRegionPixels(repeatedVertical, 96, 96, 0, 0, 96, 96, (r, g, b) => g > b + 40 && g > r + 40)
  const mirroredVerticalGreen = countRegionPixels(mirroredVertical, 96, 96, 0, 0, 96, 96, (r, g, b) => g > b + 40 && g > r + 40)
  assert.ok(clampedVerticalGreen < 20, `clamped dashed-line alphaMap V coordinates should discard the line (${clampedVerticalGreen})`)
  assert.ok(repeatedVerticalGreen > 600, `repeated dashed-line alphaMap V coordinates should wrap to the opaque texel (${repeatedVerticalGreen})`)
  assert.ok(mirroredVerticalGreen < 20, `mirrored dashed-line alphaMap V coordinates should reflect to the transparent texel (${mirroredVerticalGreen})`)
})

test('LineBasicMaterial and LineDashedMaterial maps honor nearest and linear filters', () => {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  function renderLine(kind, slot, filter) {
    const geometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-1.5, 0, 0),
      new THREE.Vector3(1.5, 0, 0),
    ])
    geometry.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([
      0.45, 0.5,
      0.45, 0.5,
    ]), 2))

    const materialProps = {
      alphaTest: slot === 'alphaMap' ? 0.3 : 0,
      color: slot === 'alphaMap' ? 0x00ff00 : 0xffffff,
      linewidth: 8,
      transparent: false,
    }

    if (slot === 'map') {
      const map = rgbaTexture([
        255, 0, 0, 255,
        0, 255, 0, 255,
      ], 2, 1)
      map.magFilter = filter
      map.minFilter = filter
      materialProps.map = map
    }

    const material = kind === 'basic'
      ? new THREE.LineBasicMaterial(materialProps)
      : new THREE.LineDashedMaterial({
        ...materialProps,
        dashSize: 10,
        gapSize: 0,
        scale: 1,
      })

    if (slot === 'alphaMap') {
      const alphaMap = rgbaTexture([
        255, 0, 255, 255,
        255, 255, 255, 255,
      ], 2, 1)
      alphaMap.magFilter = filter
      alphaMap.minFilter = filter
      material.alphaMap = alphaMap
    }

    const line = new THREE.Line(geometry, material)
    if (kind === 'dashed') line.computeLineDistances()

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, slot === 'alphaMap' ? 1 : 0)
    scene.add(line)
    return meanRegion(renderRgba(scene, camera, { width: 96, height: 96 }), 96, 96, 0, 44, 96, 52)
  }

  for (const kind of ['basic', 'dashed']) {
    const nearestMap = renderLine(kind, 'map', THREE.NearestFilter)
    const linearMap = renderLine(kind, 'map', THREE.LinearFilter)
    assert.ok(nearestMap.r > nearestMap.g + 60, `${kind} NearestFilter color map should choose the red texel (${nearestMap.r} vs ${nearestMap.g})`)
    assert.ok(linearMap.g > nearestMap.g + 30, `${kind} LinearFilter color map should blend in the green texel (${linearMap.g} vs ${nearestMap.g})`)
    assert.ok(nearestMap.r > linearMap.r + 15, `${kind} NearestFilter color map should preserve a stronger red texel (${nearestMap.r} vs ${linearMap.r})`)

    const nearestAlpha = renderLine(kind, 'alphaMap', THREE.NearestFilter)
    const linearAlpha = renderLine(kind, 'alphaMap', THREE.LinearFilter)
    assert.ok(nearestAlpha.b > nearestAlpha.g + 80, `${kind} NearestFilter alphaMap should choose the transparent texel (${nearestAlpha.b} vs ${nearestAlpha.g})`)
    assert.ok(linearAlpha.g > linearAlpha.b + 30, `${kind} LinearFilter alphaMap should blend in enough opacity to pass alphaTest (${linearAlpha.g} vs ${linearAlpha.b})`)
    assert.ok(linearAlpha.g > nearestAlpha.g + 50, `${kind} LinearFilter alphaMap should keep the line visible (${linearAlpha.g} vs ${nearestAlpha.g})`)
  }
})

test('LineDashedMaterial map samples selected uv1-uv3 texture channels', () => {
  const map = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)

  function renderLine(channel) {
    map.channel = channel

    const geom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-1.5, 0, 0),
      new THREE.Vector3(1.5, 0, 0),
    ])
    geom.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([
      0.25, 0.5,
      0.25, 0.5,
    ]), 2))
    for (const index of [1, 2, 3]) {
      const u = index === channel ? 0.75 : 0.25
      geom.setAttribute(`uv${index}`, new THREE.BufferAttribute(new Float32Array([
        u, 0.5,
        u, 0.5,
      ]), 2))
    }

    const line = new THREE.Line(geom, new THREE.LineDashedMaterial({
      color: 0xffffff,
      map,
      dashSize: 0.5,
      gapSize: 0.2,
      scale: 1,
    }))
    line.computeLineDistances()

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(line)
    return renderRgba(scene, makeCamera(), { width: 96, height: 96 })
  }

  for (const channel of [1, 2, 3]) {
    const rgba = renderLine(channel)
    const greenPixels = countRegionPixels(rgba, 96, 96, 0, 0, 96, 96, (r, g, b) => g > r + 40 && g > b + 40)
    assert.ok(greenPixels > 2, `dashed line map channel=${channel} should sample uv${channel} green texel (${greenPixels})`)
  }
})

test('line and point maps sample texture channel 3 from uv3 attributes', () => {
  const map = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter
  map.channel = 3

  function setUvChannels(geometry, count) {
    const redUvs = new Float32Array(count * 2)
    const greenUvs = new Float32Array(count * 2)
    for (let i = 0; i < count; i += 1) {
      redUvs[i * 2] = 0.25
      redUvs[i * 2 + 1] = 0.5
      greenUvs[i * 2] = 0.75
      greenUvs[i * 2 + 1] = 0.5
    }
    geometry.setAttribute('uv', new THREE.BufferAttribute(redUvs.slice(), 2))
    geometry.setAttribute('uv1', new THREE.BufferAttribute(redUvs.slice(), 2))
    geometry.setAttribute('uv2', new THREE.BufferAttribute(redUvs.slice(), 2))
    geometry.setAttribute('uv3', new THREE.BufferAttribute(greenUvs, 2))
  }

  function renderLine(material) {
    const geometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-1.5, 0, 0),
      new THREE.Vector3(1.5, 0, 0),
    ])
    setUvChannels(geometry, 2)
    const line = new THREE.Line(geometry, material)
    if (material.isLineDashedMaterial === true) line.computeLineDistances()

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(line)
    return renderRgba(scene, makeCamera(), { width: 96, height: 96 })
  }

  const basic = renderLine(new THREE.LineBasicMaterial({ color: 0xffffff, map }))
  const dashed = renderLine(new THREE.LineDashedMaterial({
    color: 0xffffff,
    dashSize: 4,
    gapSize: 0,
    map,
    scale: 1,
  }))
  const greenLinePixels = countRegionPixels(basic, 96, 96, 0, 42, 96, 54, (r, g, b) => g > r + 40 && g > b + 40)
  const greenDashedPixels = countRegionPixels(dashed, 96, 96, 0, 42, 96, 54, (r, g, b) => g > r + 40 && g > b + 40)
  assert.ok(greenLinePixels > 2, `line map channel=3 should sample uv3 green texel (${greenLinePixels})`)
  assert.ok(greenDashedPixels > 2, `dashed line map channel=3 should sample uv3 green texel (${greenDashedPixels})`)

  const pointGeometry = new THREE.BufferGeometry()
  pointGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))
  setUvChannels(pointGeometry, 1)
  const pointScene = new THREE.Scene()
  pointScene.background = new THREE.Color(0, 0, 0)
  pointScene.add(new THREE.Points(pointGeometry, new THREE.PointsMaterial({
    color: 0xffffff,
    map,
    size: 48,
    sizeAttenuation: false,
  })))
  const pointMean = meanRegion(renderRgba(pointScene, makeCamera(), { width: 96, height: 96 }), 96, 96, 40, 40, 56, 56)
  assert.ok(pointMean.g > pointMean.r + 60, `point map channel=3 should sample uv3 green texel (${pointMean.g} vs ${pointMean.r})`)
})

test('LineDashedMaterial interpolates vertex colors across dash segments', () => {
  const geom = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-1.5, 0, 0),
    new THREE.Vector3(1.5, 0, 0),
  ])
  geom.setAttribute('color', new THREE.BufferAttribute(new Float32Array([
    1, 0, 0,
    0, 1, 0,
  ]), 3))

  const line = new THREE.Line(geom, new THREE.LineDashedMaterial({
    color: 0xffffff,
    vertexColors: true,
    dashSize: 0.5,
    gapSize: 0.2,
    scale: 1,
  }))
  line.computeLineDistances()

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(line)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 96, height: 96 })
  const redPixels = countRegionPixels(rgba, 96, 96, 8, 36, 44, 60, (r, g, b) => r > g + 20 && r > b + 20)
  const greenPixels = countRegionPixels(rgba, 96, 96, 52, 36, 88, 60, (r, g, b) => g > r + 20 && g > b + 20)
  assert.ok(redPixels > 2, `left dash segments should retain red vertex colors (${redPixels})`)
  assert.ok(greenPixels > 2, `right dash segments should retain green vertex colors (${greenPixels})`)
})
