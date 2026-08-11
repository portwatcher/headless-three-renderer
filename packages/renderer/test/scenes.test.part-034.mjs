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
import { countRegionPixels, makeCamera, renderRgba } from './scenes.test.part-002.mjs'
test('examples interactive selection utilities produce renderable selected scene state', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)

  const redGeometry = new THREE.PlaneGeometry(0.34, 0.34)
  const redMaterial = new THREE.MeshBasicMaterial({ color: 0xff3344, side: THREE.DoubleSide })
  const red = new THREE.Mesh(redGeometry, redMaterial)
  red.name = 'selectable-red'
  red.position.x = -0.45
  scene.add(red)

  const blueGeometry = new THREE.PlaneGeometry(0.34, 0.34)
  const blueMaterial = new THREE.MeshBasicMaterial({ color: 0x4488ff, side: THREE.DoubleSide })
  const blue = new THREE.Mesh(blueGeometry, blueMaterial)
  blue.name = 'unselected-blue'
  blue.position.x = 0.45
  scene.add(blue)

  const instancedGeometry = new THREE.BoxGeometry(0.12, 0.12, 0.12)
  const instancedMaterial = new THREE.MeshBasicMaterial({ color: 0xffff44 })
  const instanced = new THREE.InstancedMesh(instancedGeometry, instancedMaterial, 2)
  instanced.setMatrixAt(0, new THREE.Matrix4().makeTranslation(-0.72, -0.38, 0))
  instanced.setMatrixAt(1, new THREE.Matrix4().makeTranslation(0.72, -0.38, 0))
  scene.add(instanced)

  const interactiveGroup = new InteractiveGroup()
  const interactiveGeometry = new THREE.PlaneGeometry(0.3, 0.3)
  const interactiveMaterial = new THREE.MeshBasicMaterial({ color: 0xaa66ff, side: THREE.DoubleSide })
  const interactiveMesh = new THREE.Mesh(interactiveGeometry, interactiveMaterial)
  interactiveMesh.position.y = 0.45
  interactiveGroup.add(interactiveMesh)
  scene.add(interactiveGroup)

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)
  scene.updateMatrixWorld(true)

  const selectionBox = new SelectionBox(camera, scene)
  const selected = selectionBox.select(
    new THREE.Vector3(-1, 0.2, 0),
    new THREE.Vector3(0.05, -1, 0),
  )
  assert.deepEqual(selected.map((object) => object.name), ['selectable-red'])
  assert.deepEqual(selectionBox.instances[instanced.uuid], [0])
  red.material.color.set(0x33ff66)
  instanced.setColorAt(0, new THREE.Color(0xffaa33))
  instanced.setColorAt(1, new THREE.Color(0x333333))

  const listeners = new Map()
  const fakeRenderer = {
    domElement: {
      addEventListener(type, listener) {
        listeners.set(type, listener)
      },
      getBoundingClientRect() {
        return { left: 0, top: 0, width: 100, height: 100 }
      },
    },
  }
  let clickUv = null
  interactiveMesh.addEventListener('click', (event) => {
    clickUv = event.data.clone()
  })
  interactiveGroup.listenToPointerEvents(fakeRenderer, camera)
  listeners.get('click')({
    type: 'click',
    clientX: 50,
    clientY: 28,
    stopPropagation() {},
  })
  assert.ok(clickUv, 'InteractiveGroup should dispatch pointer events to intersected child meshes')
  assert.ok(Math.abs(clickUv.x - 0.5) < 0.05)
  assert.ok(Math.abs(clickUv.y - 0.47) < 0.08)

  const hadDocument = Object.prototype.hasOwnProperty.call(globalThis, 'document')
  const previousDocument = globalThis.document
  let selectionHelper = null

  try {
    const createOverlayElement = () => ({
      classList: {
        values: [],
        add(value) {
          this.values.push(value)
        },
      },
      parentElement: null,
      style: {},
      remove() {
        this.parentElement?.removeChild(this)
      },
    })
    globalThis.document = {
      createElement(tagName) {
        assert.equal(tagName, 'div')
        return createOverlayElement()
      },
    }
    const overlayParent = {
      children: [],
      appendChild(element) {
        this.children.push(element)
        element.parentElement = this
      },
      removeChild(element) {
        this.children = this.children.filter((child) => child !== element)
        element.parentElement = null
      },
    }
    const selectionListeners = new Map()
    const selectionRenderer = {
      domElement: {
        parentElement: overlayParent,
        addEventListener(type, listener) {
          selectionListeners.set(type, listener)
        },
        removeEventListener(type, listener) {
          if (selectionListeners.get(type) === listener) selectionListeners.delete(type)
        },
      },
    }
    selectionHelper = new SelectionHelper(selectionRenderer, 'selection-box')
    assert.deepEqual(selectionHelper.element.classList.values, ['selection-box'])
    assert.equal(selectionHelper.element.style.pointerEvents, 'none')
    selectionListeners.get('pointerdown')({ clientX: 72, clientY: 38 })
    selectionListeners.get('pointermove')({ clientX: 40, clientY: 80 })
    assert.equal(selectionHelper.element.style.display, 'block')
    assert.equal(selectionHelper.element.style.left, '40px')
    assert.equal(selectionHelper.element.style.top, '38px')
    assert.equal(selectionHelper.element.style.width, '32px')
    assert.equal(selectionHelper.element.style.height, '42px')
    assert.equal(overlayParent.children.length, 1)
    selectionListeners.get('pointerup')()
    assert.equal(selectionHelper.isDown, false)
    assert.equal(overlayParent.children.length, 0)

    const width = 128
    const height = 96
    const rgba = renderRgba(scene, camera, { width, height })

    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width / 2, height, (r, g, b) => g > 150 && g > r + 30 && g > b + 20) > 150,
      'SelectionBox-selected mesh should render with updated green material state',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, width / 2, 0, width, height, (r, g, b) => b > 150 && r < 120) > 150,
      'Objects outside the SelectionBox frustum should remain blue',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => r > 150 && g > 100 && b < 100) > 20,
      'SelectionBox instance IDs can drive visible selected instance color state',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => r > 120 && b > 150 && g < 140) > 100,
      'InteractiveGroup child mesh should render through normal group traversal',
    )
  } finally {
    if (selectionHelper) selectionHelper.dispose()
    if (hadDocument) {
      globalThis.document = previousDocument
    } else {
      delete globalThis.document
    }
    redGeometry.dispose()
    redMaterial.dispose()
    blueGeometry.dispose()
    blueMaterial.dispose()
    instancedGeometry.dispose()
    instancedMaterial.dispose()
    interactiveGeometry.dispose()
    interactiveMaterial.dispose()
  }
})

test('examples DOM-backed interactive helpers require browser DOM APIs', () => {
  const hadDocument = Object.prototype.hasOwnProperty.call(globalThis, 'document')
  const previousDocument = globalThis.document
  delete globalThis.document

  try {
    const renderer = {
      domElement: {
        addEventListener() {},
        removeEventListener() {},
        parentElement: {
          appendChild() {},
          removeChild() {},
        },
      },
    }

    assert.throws(
      () => new SelectionHelper(renderer, 'selection-box'),
      /document is not defined/i,
      'SelectionHelper should require a browser document for its overlay element',
    )
    assert.throws(
      () => new HTMLMesh({}),
      /document is not defined/i,
      'HTMLMesh should require browser document/canvas APIs for html2canvas texture extraction',
    )
  } finally {
    if (hadDocument) {
      globalThis.document = previousDocument
    } else {
      delete globalThis.document
    }
  }
})

test('AsciiEffect delegates rendering and writes browser-style DOM output', () => {
  function makeDocument() {
    return {
      createElement(type) {
        if (type === 'canvas') {
          return {
            width: 0,
            height: 0,
            getContext(contextType) {
              if (contextType !== '2d') return null
              return {
                clearRect() {},
                drawImage() {},
                getImageData(_x, _y, width, height) {
                  const data = new Uint8ClampedArray(width * height * 4)
                  for (let y = 0; y < height; y += 1) {
                    for (let x = 0; x < width; x += 1) {
                      const offset = (y * width + x) * 4
                      const value = x < width / 2 ? 0 : 255
                      data[offset] = value
                      data[offset + 1] = value
                      data[offset + 2] = value
                      data[offset + 3] = 255
                    }
                  }
                  return { data, width, height }
                },
              }
            },
          }
        }

        return {
          type,
          style: {},
          children: [],
          rows: [],
          innerHTML: '',
          appendChild(child) {
            this.children.push(child)
          },
        }
      },
    }
  }

  const hadDocument = Object.prototype.hasOwnProperty.call(globalThis, 'document')
  const previousDocument = globalThis.document

  try {
    delete globalThis.document
    assert.throws(
      () => new AsciiEffect({ domElement: { style: {} } }),
      /document is not defined/i,
      'AsciiEffect should require browser-style document creation',
    )

    globalThis.document = makeDocument()
    const sizes = []
    let renderCalls = 0
    const renderer = {
      domElement: { style: {} },
      setSize(width, height) {
        sizes.push([width, height])
      },
      render(scene, camera) {
        assert.equal(scene.isScene, true)
        assert.equal(camera.isCamera, true)
        renderCalls += 1
      },
    }

    const effect = new AsciiEffect(renderer, ' .#', { resolution: 0.5 })
    const scene = new THREE.Scene()
    const camera = makeCamera()
    effect.setSize(8, 4)
    effect.render(scene, camera)

    const table = effect.domElement.children[0]
    assert.deepEqual(sizes, [[8, 4]])
    assert.equal(renderCalls, 1)
    assert.equal(table.type, 'table')
    assert.match(table.innerHTML, /<tr><td/)
    assert.match(table.innerHTML, /#/)
    assert.match(table.innerHTML, /&nbsp;/)
  } finally {
    if (hadDocument) {
      globalThis.document = previousDocument
    } else {
      delete globalThis.document
    }
  }
})
