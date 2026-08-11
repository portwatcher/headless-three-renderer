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
import { renderRgba } from './scenes.test.part-002.mjs'
test('CSS2DRenderer and CSS3DRenderer maintain browser DOM overlay state', () => {
  class FakeElement {}

  function makeElement(tagName, ownerDocument) {
    const element = new FakeElement()
    Object.assign(element, {
      tagName,
      ownerDocument,
      style: {},
      attributes: new Map(),
      children: [],
      parentNode: null,
      setAttribute(name, value) {
        this.attributes.set(name, value)
      },
      appendChild(child) {
        if (child.parentNode && child.parentNode !== this) child.parentNode.removeChild(child)
        child.parentNode = this
        this.children.push(child)
      },
      removeChild(child) {
        this.children = this.children.filter((entry) => entry !== child)
        child.parentNode = null
      },
      remove() {
        this.parentNode?.removeChild(this)
      },
      cloneNode() {
        return makeElement(tagName, ownerDocument)
      },
    })
    return element
  }

  function makeDocument() {
    const document = {
      defaultView: { Element: FakeElement },
      createElement(tagName) {
        return makeElement(tagName, document)
      },
    }
    return document
  }

  const hadDocument = Object.prototype.hasOwnProperty.call(globalThis, 'document')
  const previousDocument = globalThis.document

  try {
    delete globalThis.document
    assert.throws(() => new CSS2DObject(), /document is not defined/i)
    assert.throws(() => new CSS2DRenderer(), /document is not defined/i)
    assert.throws(() => new CSS3DObject(), /document is not defined/i)
    assert.throws(() => new CSS3DRenderer(), /document is not defined/i)

    globalThis.document = makeDocument()
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10)
    camera.position.z = 3
    camera.lookAt(0, 0, 0)

    const labelElement = document.createElement('label')
    const label = new CSS2DObject(labelElement)
    label.position.set(-0.3, 0.2, 0)
    scene.add(label)

    const css2d = new CSS2DRenderer()
    css2d.setSize(100, 50)
    css2d.render(scene, camera)
    assert.deepEqual(css2d.getSize(), { width: 100, height: 50 })
    assert.equal(css2d.domElement.style.width, '100px')
    assert.equal(css2d.domElement.style.height, '50px')
    assert.equal(labelElement.parentNode, css2d.domElement)
    assert.match(labelElement.style.transform, /translate/)
    assert.equal(labelElement.attributes.get('draggable'), false)

    const panelElement = document.createElement('panel')
    const panel = new CSS3DObject(panelElement)
    panel.position.set(0.2, -0.1, 0)
    const spriteElement = document.createElement('sprite')
    const sprite = new CSS3DSprite(spriteElement)
    sprite.position.set(0.1, 0.3, 0)
    scene.add(panel, sprite)

    const css3d = new CSS3DRenderer()
    css3d.setSize(120, 90)
    css3d.render(scene, camera)
    const viewElement = css3d.domElement.children[0]
    const cameraElement = viewElement.children[0]
    assert.deepEqual(css3d.getSize(), { width: 120, height: 90 })
    assert.equal(css3d.domElement.style.width, '120px')
    assert.equal(viewElement.style.width, '120px')
    assert.equal(cameraElement.style.transformStyle, 'preserve-3d')
    assert.equal(panelElement.parentNode, cameraElement)
    assert.equal(spriteElement.parentNode, cameraElement)
    assert.match(panelElement.style.transform, /matrix3d/)
    assert.match(spriteElement.style.transform, /matrix3d/)
  } finally {
    if (hadDocument) {
      globalThis.document = previousDocument
    } else {
      delete globalThis.document
    }
  }
})

test('SVGRenderer serializes supported Projector output into SVG DOM nodes', () => {
  function makeSvgElement(namespaceURI, tagName) {
    const childNodes = []
    const element = {
      namespaceURI,
      tagName,
      style: {},
      attributes: new Map(),
      childNodes,
      children: childNodes,
      parentNode: null,
      setAttribute(name, value) {
        this.attributes.set(name, String(value))
      },
      appendChild(child) {
        if (child.parentNode && child.parentNode !== this) child.parentNode.removeChild(child)
        child.parentNode = this
        childNodes.push(child)
        return child
      },
      removeChild(child) {
        const index = childNodes.indexOf(child)
        if (index >= 0) childNodes.splice(index, 1)
        child.parentNode = null
        return child
      },
    }
    return element
  }

  function makeDocument() {
    return {
      createElementNS(namespaceURI, tagName) {
        assert.equal(namespaceURI, 'http://www.w3.org/2000/svg')
        return makeSvgElement(namespaceURI, tagName)
      },
    }
  }

  const hadDocument = Object.prototype.hasOwnProperty.call(globalThis, 'document')
  const previousDocument = globalThis.document
  const meshGeometry = new THREE.PlaneGeometry(0.8, 0.7)
  const meshMaterial = new THREE.MeshBasicMaterial({ color: 0xff3344, side: THREE.DoubleSide })
  const lineGeometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-0.55, -0.45, 0),
    new THREE.Vector3(0.45, -0.45, 0),
  ])
  const lineMaterial = new THREE.LineBasicMaterial({ color: 0x44ff66 })

  try {
    delete globalThis.document
    assert.throws(() => new SVGRenderer(), /document is not defined/i)

    globalThis.document = makeDocument()
    const renderer = new SVGRenderer()
    renderer.setQuality('low')
    renderer.setClearColor(0x112233)
    renderer.setSize(120, 80)
    renderer.clear()

    assert.equal(renderer.domElement.tagName, 'svg')
    assert.equal(renderer.domElement.attributes.get('width'), '120')
    assert.equal(renderer.domElement.attributes.get('height'), '80')
    assert.equal(renderer.domElement.attributes.get('viewBox'), '-60 -40 120 80')
    assert.match(renderer.domElement.style.backgroundColor, /^rgb/)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x000000)
    const mesh = new THREE.Mesh(meshGeometry, meshMaterial)
    mesh.position.x = -0.2
    const line = new THREE.LineSegments(lineGeometry, lineMaterial)
    const customNode = document.createElementNS('http://www.w3.org/2000/svg', 'g')
    const svgObject = new SVGObject(customNode)
    svgObject.position.set(0.35, 0.2, 0)
    scene.add(mesh, line, svgObject)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10)
    camera.position.z = 3
    camera.lookAt(0, 0, 0)
    camera.updateMatrixWorld(true)

    renderer.setPrecision(2)
    renderer.render(scene, camera)
    const pathNodes = renderer.domElement.childNodes.filter((node) => node.tagName === 'path')
    assert.ok(pathNodes.length >= 2, 'SVGRenderer should append projected path nodes')
    assert.ok(pathNodes.some((node) => node.attributes.get('style').includes('fill:')), 'mesh faces should produce fill paths')
    assert.ok(pathNodes.some((node) => node.attributes.get('style').includes('stroke:')), 'line segments should produce stroke paths')
    assert.ok(pathNodes.some((node) => node.attributes.get('shape-rendering') === 'crispEdges'))
    assert.ok(pathNodes.every((node) => node.attributes.get('d').startsWith('M')))
    assert.equal(customNode.parentNode, renderer.domElement)
    assert.match(customNode.attributes.get('transform'), /^translate\(/)
    assert.equal(renderer.info.render.faces, 2)
    assert.equal(renderer.info.render.vertices, 6)

    const rgba = renderRgba(scene, camera, { width: 64, height: 64 })
    assert.ok(
      nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.03,
      'SVGRenderer source scene objects should still render through the normal renderer path',
    )
  } finally {
    meshGeometry.dispose()
    meshMaterial.dispose()
    lineGeometry.dispose()
    lineMaterial.dispose()
    if (hadDocument) {
      globalThis.document = previousDocument
    } else {
      delete globalThis.document
    }
  }
})

test('SVGLoader exposes its DOMParser dependency in plain Node', () => {
  const hadDOMParser = Object.prototype.hasOwnProperty.call(globalThis, 'DOMParser')
  const previousDOMParser = globalThis.DOMParser

  try {
    delete globalThis.DOMParser
    assert.throws(
      () => new SVGLoader().parse('<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0 L1 0 L1 1 Z"/></svg>'),
      /DOMParser is not defined|DOMParser is not a constructor/i,
    )
  } finally {
    if (hadDOMParser) {
      globalThis.DOMParser = previousDOMParser
    }
  }
})

test('KTX2Loader detects conservative renderer texture compression support', async () => {
  const renderer = new Renderer()
  const expectedSupport = {
    astcSupported: false,
    astcHDRSupported: false,
    etc1Supported: false,
    etc2Supported: false,
    dxtSupported: false,
    bptcSupported: false,
    pvrtcSupported: false,
  }
  const syncFeatureChecks = []
  const syncExtensionChecks = []
  const originalHasFeature = renderer.hasFeature.bind(renderer)
  const originalExtensionsHas = renderer.extensions.has.bind(renderer.extensions)
  renderer.hasFeature = (name) => {
    syncFeatureChecks.push(name)
    return originalHasFeature(name)
  }
  renderer.extensions.has = (name) => {
    syncExtensionChecks.push(name)
    return originalExtensionsHas(name)
  }

  const loader = new KTX2Loader()
  assert.equal(renderer.isWebGPURenderer, false)
  assert.equal(loader.detectSupport(renderer), loader)
  assert.deepEqual(loader.workerConfig, expectedSupport)
  assert.deepEqual(
    syncFeatureChecks,
    [],
    'synchronous KTX2 detection should stay on WebGL extension probes when isWebGPURenderer is false',
  )
  assert.deepEqual(syncExtensionChecks, [
    'WEBGL_compressed_texture_astc',
    'WEBGL_compressed_texture_astc',
    'WEBGL_compressed_texture_etc1',
    'WEBGL_compressed_texture_etc',
    'WEBGL_compressed_texture_s3tc',
    'EXT_texture_compression_bptc',
    'WEBGL_compressed_texture_pvrtc',
    'WEBKIT_WEBGL_compressed_texture_pvrtc',
  ])

  const asyncRenderer = new Renderer()
  const asyncFeatureChecks = []
  const asyncExtensionChecks = []
  const originalAsyncHasFeature = asyncRenderer.hasFeature.bind(asyncRenderer)
  const originalAsyncExtensionsHas = asyncRenderer.extensions.has.bind(asyncRenderer.extensions)
  asyncRenderer.hasFeature = (name) => {
    asyncFeatureChecks.push(name)
    return originalAsyncHasFeature(name)
  }
  asyncRenderer.extensions.has = (name) => {
    asyncExtensionChecks.push(name)
    return originalAsyncExtensionsHas(name)
  }
  const asyncLoader = new KTX2Loader()
  assert.equal(await asyncLoader.detectSupportAsync(asyncRenderer), asyncLoader)
  assert.deepEqual(asyncLoader.workerConfig, expectedSupport)
  assert.deepEqual(asyncFeatureChecks, [])
  assert.deepEqual(asyncExtensionChecks, syncExtensionChecks)
})
