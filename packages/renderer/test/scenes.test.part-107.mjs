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
import { cubeTexture, makeCamera, meanAbsDiff, renderRgba, solidTexture } from './scenes.test.part-002.mjs'
import { maxLuminance } from './scenes.test.part-003.mjs'
test('invalid physical material scalar values fail clearly', () => {
  const cases = [
    ['metalness', (material) => {
      material.metalness = 'metal'
    }, /material\.metalness must be a finite number/i],
    ['roughness', (material) => {
      material.roughness = Number.NaN
    }, /material\.roughness must be a finite number/i],
    ['clearcoat', (material) => {
      material.clearcoat = 'coat'
    }, /material\.clearcoat must be a finite number/i],
    ['clearcoatRoughness', (material) => {
      material.clearcoatRoughness = Number.POSITIVE_INFINITY
    }, /material\.clearcoatRoughness must be a finite number/i],
    ['clearcoatNormalScale.x', (material) => {
      material.clearcoatNormalScale = new THREE.Vector2(1, 1)
      material.clearcoatNormalScale.x = 'wide'
    }, /material\.clearcoatNormalScale\.x must be a finite number/i],
    ['clearcoatNormalScale container', (material) => {
      material.clearcoatNormalScale = 'wide'
    }, /material\.clearcoatNormalScale must be a Vector2-like object/i],
    ['sheen', (material) => {
      material.sheen = 'soft'
    }, /material\.sheen must be a finite number/i],
    ['sheenRoughness', (material) => {
      material.sheenRoughness = Number.NaN
    }, /material\.sheenRoughness must be a finite number/i],
    ['anisotropy', (material) => {
      material.anisotropy = 'aligned'
    }, /material\.anisotropy must be a finite number/i],
    ['anisotropyRotation', (material) => {
      material.anisotropyRotation = Number.NEGATIVE_INFINITY
    }, /material\.anisotropyRotation must be a finite number/i],
    ['iridescence', (material) => {
      material.iridescence = 'rainbow'
    }, /material\.iridescence must be a finite number/i],
    ['iridescenceIOR', (material) => {
      material.iridescenceIOR = Number.NaN
    }, /material\.iridescenceIOR must be a finite number/i],
    ['iridescenceThicknessRange container', (material) => {
      material.iridescenceThicknessRange = 'range'
    }, /material\.iridescenceThicknessRange must be an array-like pair/i],
    ['iridescenceThicknessRange length', (material) => {
      material.iridescenceThicknessRange = [100]
    }, /material\.iridescenceThicknessRange must contain at least two values/i],
    ['iridescenceThicknessRange value', (material) => {
      material.iridescenceThicknessRange = [100, 'thick']
    }, /material\.iridescenceThicknessRange\[1\] must be a finite number/i],
    ['transmission', (material) => {
      material.transmission = 'glass'
    }, /material\.transmission must be a finite number/i],
    ['dispersion', (material) => {
      material.dispersion = Number.NaN
    }, /material\.dispersion must be a finite number/i],
    ['ior', (material) => {
      material.ior = 'dense'
    }, /material\.ior must be a finite number/i],
    ['thickness', (material) => {
      material.thickness = Number.POSITIVE_INFINITY
    }, /material\.thickness must be a finite number/i],
    ['attenuationDistance', (material) => {
      material.attenuationDistance = 'short'
    }, /material\.attenuationDistance must be a finite number/i],
    ['specularIntensity', (material) => {
      material.specularIntensity = Number.NaN
    }, /material\.specularIntensity must be a finite number/i],
  ]

  for (const [name, mutate, pattern] of cases) {
    const material = new THREE.MeshPhysicalMaterial({ color: 0xffffff })
    mutate(material)
    const scene = new THREE.Scene()
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(1, 1), material))

    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      pattern,
      `${name} should fail clearly`,
    )
  }
})

test('invalid physical material color values fail clearly', () => {
  const cases = [
    ['sheenColor component', (material) => {
      material.sheenColor = { isColor: true, r: 1, g: 'warm', b: 0 }
    }, /material\.sheenColor\.g must be a finite number/i],
    ['sheenColor container', (material) => {
      material.sheenColor = 42
    }, /material\.sheenColor must be a color-like object, CSS color string, or \[r, g, b\]/i],
    ['attenuationColor component', (material) => {
      material.attenuationColor = [0.2, Number.NaN, 1]
    }, /material\.attenuationColor\[1\] must be a finite number/i],
    ['attenuationColor container', (material) => {
      material.attenuationColor = {}
    }, /material\.attenuationColor must be a color-like object, CSS color string, or \[r, g, b\]/i],
    ['specularColor CSS', (material) => {
      material.specularColor = 'not-a-color'
    }, /material\.specularColor "not-a-color" is not a supported CSS color string/i],
  ]

  for (const [name, mutate, pattern] of cases) {
    const material = new THREE.MeshPhysicalMaterial({
      color: 0xffffff,
      sheen: 1,
      transmission: 1,
      specularIntensity: 1,
    })
    mutate(material)
    const scene = new THREE.Scene()
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(1, 1), material))

    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      pattern,
      `${name} should fail clearly`,
    )
  }
})

test('MeshPhysicalMaterial specular intensity and color affect direct specular', () => {
  function renderMaterial(material) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(new THREE.SphereGeometry(1, 48, 24), material))

    const light = new THREE.DirectionalLight(0xffffff, 8)
    light.position.set(0, 0, 3)
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return renderRgba(scene, camera, { width: 64, height: 64 })
  }

  const disabled = renderMaterial(new THREE.MeshPhysicalMaterial({
    color: 0x000000,
    roughness: 0.08,
    metalness: 0,
    specularIntensity: 0,
  }))
  const enabled = renderMaterial(new THREE.MeshPhysicalMaterial({
    color: 0x000000,
    roughness: 0.08,
    metalness: 0,
    specularIntensity: 1,
  }))
  assert.ok(maxLuminance(enabled) > maxLuminance(disabled) + 20, 'specularIntensity should control the direct specular highlight')

  const red = meanRgba(renderMaterial(new THREE.MeshPhysicalMaterial({
    color: 0x000000,
    roughness: 0.08,
    metalness: 0,
    specularIntensity: 1,
    specularColor: new THREE.Color(1, 0, 0),
  })))
  const green = meanRgba(renderMaterial(new THREE.MeshPhysicalMaterial({
    color: 0x000000,
    roughness: 0.08,
    metalness: 0,
    specularIntensity: 1,
    specularColor: new THREE.Color(0, 1, 0),
  })))
  assert.ok(red.r > red.g + 0.1, `red specularColor should tint the highlight red (${red.r} vs ${red.g})`)
  assert.ok(green.g > green.r + 0.1, `green specularColor should tint the highlight green (${green.g} vs ${green.r})`)
})

test('MeshPhysicalMaterial negative anisotropy matches Three.js positive-only feature gating', () => {
  function renderAnisotropy(anisotropy) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.SphereGeometry(1, 64, 32),
      new THREE.MeshPhysicalMaterial({
        color: 0x000000,
        roughness: 0.22,
        metalness: 0,
        specularIntensity: 1,
        anisotropy,
        anisotropyRotation: Math.PI / 3,
        anisotropyMap: solidTexture(255, 128, 255),
      }),
    ))

    const light = new THREE.PointLight(0xffffff, 450)
    light.position.set(0.5, 0.35, 2.2)
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return renderRgba(scene, camera, { width: 64, height: 64 })
  }

  const zero = renderAnisotropy(0)
  const negative = renderAnisotropy(-0.9)
  const positive = renderAnisotropy(0.9)

  const negativeDiff = meanAbsDiff(zero, negative)
  const positiveDiff = meanAbsDiff(zero, positive)
  assert.ok(negativeDiff < 0.05, `negative anisotropy should render like zero anisotropy, diff=${negativeDiff.toFixed(3)}`)
  assert.ok(positiveDiff > 0.5, `positive anisotropy should still affect the physical BRDF, diff=${positiveDiff.toFixed(3)}`)
})

test('inactive MeshPhysicalMaterial extension maps are ignored before texture validation', () => {
  function ignoredCubeMap() {
    return cubeTexture([
      [255, 0, 0],
      [0, 255, 0],
      [0, 0, 255],
      [255, 255, 0],
      [255, 0, 255],
      [0, 255, 255],
    ])
  }

  function mipmappedTexture() {
    const texture = solidTexture(255, 0, 0)
    texture.mipmaps = [{ data: new Uint8Array([0, 255, 0, 255]), width: 1, height: 1 }]
    return texture
  }

  function renderMaterial(params = {}) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.AmbientLight(0xffffff, 1))
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshPhysicalMaterial({
        color: 0x6688aa,
        roughness: 0.65,
        metalness: 0,
        ...params,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return renderRgba(scene, camera, { width: 64, height: 64 })
  }

  const baseline = renderMaterial()
  const cases = [
    ['clearcoat maps', {
      clearcoat: 0,
      clearcoatMap: ignoredCubeMap(),
      clearcoatRoughnessMap: mipmappedTexture(),
      clearcoatNormalMap: ignoredCubeMap(),
    }],
    ['sheen maps', {
      sheen: 0,
      sheenColor: new THREE.Color(1, 0, 0),
      sheenColorMap: ignoredCubeMap(),
      sheenRoughnessMap: mipmappedTexture(),
    }],
    ['anisotropy map', {
      anisotropy: 0,
      anisotropyMap: ignoredCubeMap(),
    }],
    ['iridescence maps', {
      iridescence: 0,
      iridescenceMap: ignoredCubeMap(),
      iridescenceThicknessMap: mipmappedTexture(),
    }],
    ['transmission maps', {
      transmission: 0,
      thickness: 8,
      transmissionMap: ignoredCubeMap(),
      thicknessMap: mipmappedTexture(),
    }],
  ]

  for (const [label, params] of cases) {
    const diff = meanAbsDiff(baseline, renderMaterial(params))
    assert.ok(diff < 0.05, `${label} should be ignored while its controlling scalar is zero, diff=${diff.toFixed(3)}`)
  }
})
