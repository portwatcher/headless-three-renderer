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
import { makeCamera, meanRegion, renderRgba } from './scenes.test.part-002.mjs'
import { nonBackgroundBounds } from './scenes.test.part-003.mjs'
test('SpriteMaterial honors sprite scale and material rotation', () => {
  function renderRotatedSprite(rotation) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const material = new THREE.SpriteMaterial({ color: 0xffffff, rotation })
    const sprite = new THREE.Sprite(material)
    sprite.scale.set(1.8, 0.45, 1)
    scene.add(sprite)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return renderRgba(scene, camera, { width: 96, height: 96 })
  }

  const horizontal = nonBackgroundBounds(renderRotatedSprite(0), 96, 96, [0, 0, 0])
  const vertical = nonBackgroundBounds(renderRotatedSprite(Math.PI / 2), 96, 96, [0, 0, 0])
  assert.ok(horizontal.width > horizontal.height * 2, `unrotated sprite should be wide (${horizontal.width}x${horizontal.height})`)
  assert.ok(vertical.height > vertical.width * 2, `rotated sprite should be tall (${vertical.width}x${vertical.height})`)
})

test('SpriteMaterial sizeAttenuation=false keeps perspective sprite size depth independent', () => {
  function renderSprite(z, sizeAttenuation) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const material = new THREE.SpriteMaterial({ color: 0xffffff, sizeAttenuation })
    const sprite = new THREE.Sprite(material)
    sprite.position.z = z
    sprite.scale.set(0.2, 0.2, 1)
    scene.add(sprite)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return nonBackgroundBounds(renderRgba(scene, camera, { width: 96, height: 96 }), 96, 96, [0, 0, 0])
  }

  const nearAttenuated = renderSprite(0, true)
  const farAttenuated = renderSprite(-3, true)
  const nearFixed = renderSprite(0, false)
  const farFixed = renderSprite(-3, false)

  assert.ok(
    nearAttenuated.width >= farAttenuated.width * 1.7,
    `default perspective sprite should shrink with distance (${nearAttenuated.width} vs ${farAttenuated.width})`,
  )
  assert.ok(
    Math.abs(nearFixed.width - farFixed.width) <= 2,
    `sizeAttenuation=false sprite should keep width stable with distance (${nearFixed.width} vs ${farFixed.width})`,
  )
  assert.ok(
    Math.abs(nearFixed.height - farFixed.height) <= 2,
    `sizeAttenuation=false sprite should keep height stable with distance (${nearFixed.height} vs ${farFixed.height})`,
  )
})

test('Sprite center shifts billboard anchoring around object position', () => {
  function renderCenteredSprite(centerX, centerY) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ color: 0xffffff }))
    sprite.center.set(centerX, centerY)
    sprite.scale.set(0.8, 0.8, 1)
    scene.add(sprite)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return nonBackgroundBounds(renderRgba(scene, camera, { width: 96, height: 96 }), 96, 96, [0, 0, 0])
  }

  const centered = renderCenteredSprite(0.5, 0.5)
  const lowerLeft = renderCenteredSprite(0, 0)
  const upperRight = renderCenteredSprite(1, 1)

  assert.ok(lowerLeft.minX > centered.minX + 10, `center=(0,0) should anchor the sprite to the right of its origin (${lowerLeft.minX} vs ${centered.minX})`)
  assert.ok(lowerLeft.maxY < centered.maxY - 10, `center=(0,0) should anchor the sprite above its origin (${lowerLeft.maxY} vs ${centered.maxY})`)
  assert.ok(upperRight.maxX < centered.maxX - 10, `center=(1,1) should anchor the sprite to the left of its origin (${upperRight.maxX} vs ${centered.maxX})`)
  assert.ok(upperRight.minY > centered.minY + 10, `center=(1,1) should anchor the sprite below its origin (${upperRight.minY} vs ${centered.minY})`)
})

test('invalid billboard and line scalar values fail clearly', () => {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  function spriteScene(mutator) {
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ color: 0xffffff }))
    mutator(sprite, sprite.material)
    const scene = new THREE.Scene()
    scene.add(sprite)
    return scene
  }

  function pointsScene(mutator) {
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))
    const material = new THREE.PointsMaterial({ color: 0xffffff })
    mutator(material)
    const scene = new THREE.Scene()
    scene.add(new THREE.Points(geometry, material))
    return scene
  }

  function lineScene(material, configureGeometry = () => {}) {
    const geometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-0.5, 0, 0),
      new THREE.Vector3(0.5, 0, 0),
    ])
    configureGeometry(geometry)
    const scene = new THREE.Scene()
    scene.add(new THREE.Line(geometry, material))
    return scene
  }

  const cases = [
    ['sprite center', () => spriteScene((sprite) => {
      sprite.center.x = 'left'
    }), /Sprite\.center\.x must be a finite number/i],
    ['sprite scale x', () => spriteScene((sprite) => {
      sprite.scale.x = 'wide'
    }), /Sprite\.scale\.x must be a finite number/i],
    ['sprite scale z', () => spriteScene((sprite) => {
      sprite.scale.z = Number.NaN
    }), /Sprite\.scale\.z must be a finite number/i],
    ['sprite rotation', () => spriteScene((_sprite, material) => {
      material.rotation = Number.NaN
    }), /material\.rotation must be a finite number/i],
    ['point size', () => pointsScene((material) => {
      material.size = 'large'
    }), /material\.size must be a finite number/i],
    ['point size zero', () => pointsScene((material) => {
      material.size = 0
    }), /material\.size must be positive/i],
    ['point size negative', () => pointsScene((material) => {
      material.size = -1
    }), /material\.size must be positive/i],
    ['line width', () => {
      const material = new THREE.LineBasicMaterial({ color: 0xffffff })
      material.linewidth = Number.POSITIVE_INFINITY
      return lineScene(material)
    }, /material\.linewidth must be a finite number/i],
    ['line width zero', () => {
      const material = new THREE.LineBasicMaterial({ color: 0xffffff })
      material.linewidth = 0
      return lineScene(material)
    }, /material\.linewidth must be positive/i],
    ['dash size', () => {
      const material = new THREE.LineDashedMaterial({ color: 0xffffff })
      material.dashSize = 'long'
      return lineScene(material)
    }, /material\.dashSize must be a finite number/i],
    ['dash size negative', () => {
      const material = new THREE.LineDashedMaterial({ color: 0xffffff })
      material.dashSize = -0.1
      return lineScene(material)
    }, /material\.dashSize must be positive/i],
    ['dash gap', () => {
      const material = new THREE.LineDashedMaterial({ color: 0xffffff })
      material.gapSize = Number.NaN
      return lineScene(material)
    }, /material\.gapSize must be a finite number/i],
    ['dash gap negative', () => {
      const material = new THREE.LineDashedMaterial({ color: 0xffffff })
      material.gapSize = -0.1
      return lineScene(material)
    }, /material\.gapSize must be non-negative/i],
    ['dash scale', () => {
      const material = new THREE.LineDashedMaterial({ color: 0xffffff })
      material.scale = 'fast'
      return lineScene(material)
    }, /material\.scale must be a finite number/i],
    ['dash scale negative', () => {
      const material = new THREE.LineDashedMaterial({ color: 0xffffff })
      material.scale = -0.1
      return lineScene(material)
    }, /material\.scale must be non-negative/i],
    ['line distance finite', () => {
      const material = new THREE.LineDashedMaterial({ color: 0xffffff })
      return lineScene(material, (geometry) => {
        geometry.setAttribute('lineDistance', new THREE.BufferAttribute(new Float32Array([0, Number.NaN]), 1))
      })
    }, /geometry\.attributes\.lineDistance\[1\]\.x must be a finite number/i],
    ['line distance itemSize', () => {
      const material = new THREE.LineDashedMaterial({ color: 0xffffff })
      return lineScene(material, (geometry) => {
        geometry.setAttribute('lineDistance', {
          count: 2,
          itemSize: 0,
          array: new Float32Array([0, 1]),
        })
      })
    }, /geometry\.attributes\.lineDistance\.itemSize must be a positive integer/i],
  ]

  for (const [label, makeScene, pattern] of cases) {
    assert.throws(
      () => renderRgba(makeScene(), camera, { width: 64, height: 64 }),
      pattern,
      label,
    )
  }
})

test('Sprite receiveShadow is accepted as an unlit WebGL-compatible no-op', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const receiver = new THREE.Sprite(new THREE.SpriteMaterial({ color: 0xffffff }))
  receiver.receiveShadow = true
  receiver.scale.set(1.2, 1.2, 1)
  scene.add(receiver)

  const mean = meanRegion(renderRgba(scene, makeCamera(), { width: 64, height: 64 }), 64, 64, 24, 24, 40, 40)
  assert.ok(mean.r > 180 && mean.g > 180 && mean.b > 180, `sprite receiveShadow no-op should still render the unlit billboard (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('Sprite casts point-light shadows from expanded billboard quads', () => {
  function renderSpriteShadow(castShadow) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const caster = new THREE.Sprite(new THREE.SpriteMaterial({ color: 0xffffff }))
    caster.position.set(0, 2.2, 1.8)
    caster.scale.set(2.8, 2.8, 1)
    caster.castShadow = castShadow
    scene.add(caster)

    const light = new THREE.PointLight(0xffffff, 2)
    light.position.set(0, 5, 4)
    light.distance = 12
    light.castShadow = true
    light.shadow.mapSize.set(256, 256)
    light.shadow.camera.near = 0.1
    light.shadow.camera.far = 12
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, { width: 96, height: 96 }), 96, 96, 28, 42, 68, 82)
  }

  const unshadowed = renderSpriteShadow(false)
  const shadowed = renderSpriteShadow(true)
  const unshadowedLum = unshadowed.r + unshadowed.g + unshadowed.b
  const shadowedLum = shadowed.r + shadowed.g + shadowed.b
  assert.ok(shadowedLum < unshadowedLum - 15, `sprite point-light shadow should darken the receiver (${shadowedLum} vs ${unshadowedLum})`)
})

test('camera layers filter renderable objects', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const redOccluder = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  )
  redOccluder.position.z = 0.1
  scene.add(redOccluder)

  const greenVisible = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0x00ff00 }),
  )
  greenVisible.layers.set(1)
  scene.add(greenVisible)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  camera.layers.set(1)

  const rgba = renderRgba(scene, camera, { width: 64, height: 64 })
  const mean = meanRgba(rgba)
  assert.ok(mean.g > mean.r + 20, `layer 1 object should dominate over filtered layer 0 object (${mean.g} vs ${mean.r})`)
})
