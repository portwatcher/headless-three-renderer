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
import { renderToTarget, test } from './scenes.test.part-001.mjs'
import { countRegionPixels, renderRgba } from './scenes.test.part-002.mjs'
test('Three.js scene graph exporters serialize renderer-visible geometry', async () => {
  class FileReaderShim {
    result = null
    onloadend = null

    async readAsArrayBuffer(blob) {
      this.result = await blob.arrayBuffer()
      this.onloadend?.({ target: this })
    }

    async readAsDataURL(blob) {
      this.result = `data:${blob.type || 'application/octet-stream'};base64,${Buffer.from(await blob.arrayBuffer()).toString('base64')}`
      this.onloadend?.({ target: this })
    }
  }

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)

  const meshGeometry = new THREE.PlaneGeometry(0.6, 0.6)
  meshGeometry.setAttribute('color', new THREE.Float32BufferAttribute([
    1, 0, 0,
    1, 0, 0,
    0, 1, 0,
    0, 1, 0,
  ], 3))
  const meshMaterial = new THREE.MeshBasicMaterial({
    name: 'export-red-green',
    vertexColors: true,
    side: THREE.DoubleSide,
  })
  const mesh = new THREE.Mesh(meshGeometry, meshMaterial)
  mesh.name = 'export-mesh'
  mesh.position.x = -0.45

  const lineGeometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-0.2, -0.35, 0),
    new THREE.Vector3(0.2, 0.35, 0),
  ])
  const lineMaterial = new THREE.LineBasicMaterial({ color: 0x44aaff })
  const line = new THREE.Line(lineGeometry, lineMaterial)
  line.name = 'export-line'
  line.position.x = 0.35

  const pointsGeometry = new THREE.BufferGeometry()
  pointsGeometry.setAttribute('position', new THREE.Float32BufferAttribute([
    0, -0.18, 0,
    0.12, 0.16, 0,
    -0.12, 0.16, 0,
  ], 3))
  pointsGeometry.setAttribute('color', new THREE.Float32BufferAttribute([
    1, 1, 0,
    1, 1, 0,
    1, 1, 0,
  ], 3))
  const pointsMaterial = new THREE.PointsMaterial({
    vertexColors: true,
    size: 8,
    sizeAttenuation: false,
  })
  const points = new THREE.Points(pointsGeometry, pointsMaterial)
  points.name = 'export-points'
  points.position.x = 0.75

  scene.add(mesh, line, points)
  scene.updateMatrixWorld(true)

  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 0.8, -0.8, 0.01, 10)
  camera.position.set(0, 0, 4)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  try {
    const width = 128
    const height = 80
    const rgba = renderRgba(scene, camera, { width, height })
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => r > 160 && g < 130 && b < 130) > 80,
      'exporter mesh scene should render red vertex-colored pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => g > 130 && g > r + 20 && g > b + 20) > 80,
      'exporter mesh scene should render green vertex-colored pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => b > 150 && r < 140 && g > 120) > 20,
      'exporter line scene should render blue line pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => r > 150 && g > 130 && b < 140) > 20,
      'exporter point scene should render yellow point pixels',
    )

    const obj = new OBJExporter().parse(scene)
    assert.match(obj, /^o export-mesh$/m)
    assert.match(obj, /^usemtl export-red-green$/m)
    assert.match(obj, /^f \d+\/\d+\/\d+ \d+\/\d+\/\d+ \d+\/\d+\/\d+$/m)
    assert.match(obj, /^o export-line$/m)
    assert.match(obj, /^l \d+ \d+ $/m)
    assert.match(obj, /^o export-points$/m)
    assert.match(obj, /^p \d+ \d+ \d+ $/m)

    const stl = new STLExporter().parse(scene)
    assert.match(stl, /^solid exported/)
    assert.match(stl, /\bfacet normal\b/)
    assert.match(stl, /\bvertex -?[\d.]+ -?[\d.]+ 0\b/)
    assert.match(stl, /endsolid exported\s*$/)

    const ply = new PLYExporter().parse(mesh)
    assert.match(ply, /^ply\nformat ascii 1\.0/m)
    assert.match(ply, /^element vertex 4$/m)
    assert.match(ply, /^element face 2$/m)
    assert.match(ply, /^property uchar red$/m)
    assert.match(ply, /^3 0 2 1$/m)

    const previousFileReader = globalThis.FileReader
    try {
      globalThis.FileReader = FileReaderShim
      const gltf = await new GLTFExporter().parseAsync(mesh, { binary: false })
      assert.equal(gltf.asset.version, '2.0')
      assert.equal(gltf.meshes.length, 1)
      assert.equal(gltf.nodes.some((node) => node.name === 'export-mesh'), true)
      assert.match(gltf.buffers[0].uri, /^data:application\/octet-stream;base64,/)

      const glb = await new GLTFExporter().parseAsync(mesh, { binary: true })
      const glbHeader = new DataView(glb)
      assert.equal(glbHeader.getUint32(0, true), 0x46546c67)
      assert.equal(glbHeader.getUint32(4, true), 2)
      assert.equal(glbHeader.getUint32(8, true), glb.byteLength)
    } finally {
      if (previousFileReader === undefined) {
        delete globalThis.FileReader
      } else {
        globalThis.FileReader = previousFileReader
      }
    }

    const usdzScene = new THREE.Scene()
    usdzScene.background = new THREE.Color(0x000000)
    const usdzGeometry = new THREE.BoxGeometry(0.7, 0.7, 0.7)
    const usdzMaterial = new THREE.MeshStandardMaterial({
      color: 0xff3344,
      roughness: 0.6,
      metalness: 0,
    })
    const usdzMesh = new THREE.Mesh(usdzGeometry, usdzMaterial)
    usdzMesh.name = 'export-usdz-standard'
    usdzScene.add(new THREE.AmbientLight(0xffffff, 1.5), usdzMesh)
    usdzScene.updateMatrixWorld(true)

    try {
      const usdzRgba = renderRgba(usdzScene, camera, { width, height })
      assert.ok(
        countRegionPixels(usdzRgba, width, height, 0, 0, width, height, (r, g, b) => r > 80 && r > g + 20 && r > b + 10) > 100,
        'USDZ-exportable Standard material scene should render visible red pixels',
      )
      const usdz = await new USDZExporter().parseAsync(usdzScene)
      assert.ok(usdz instanceof Uint8Array, 'USDZExporter should return Uint8Array data')
      assert.ok(usdz.length > 512, `USDZ output should contain zipped USDA content (${usdz.length})`)
      assert.deepEqual(
        Array.from(usdz.subarray(0, 4)),
        [0x50, 0x4b, 0x03, 0x04],
        'USDZ output should start with a ZIP local file header',
      )
      const usdzText = new TextDecoder().decode(usdz)
      assert.match(usdzText, /model\.usda/)
      assert.match(usdzText, /geometries\/Geometry_/)
    } finally {
      usdzGeometry.dispose()
      usdzMaterial.dispose()
    }
  } finally {
    meshGeometry.dispose()
    meshMaterial.dispose()
    lineGeometry.dispose()
    lineMaterial.dispose()
    pointsGeometry.dispose()
    pointsMaterial.dispose()
  }
})

test('DRACOExporter external encoder requirement fails clearly', () => {
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial())

  try {
    assert.throws(
      () => new DRACOExporter().parse(mesh),
      /required the draco_encoder to work|DracoEncoderModule is not defined/i,
    )
  } finally {
    mesh.geometry.dispose()
    mesh.material.dispose()
  }
})

test('single-attachment target array paths honor typed color readback requests', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const options = { width: 32, height: 32, outputColorSpace: THREE.LinearSRGBColorSpace }
  const center = ((16 * 32) + 16) * 2
  const cases = [
    ['target.texture array', { texture: [{ format: THREE.RGFormat, type: THREE.FloatType }] }, (target) => target.texture[0]],
    ['target.textures array', { textures: [{ format: THREE.RGFormat, type: THREE.FloatType }] }, (target) => target.textures[0]],
    ['single-attachment MRT target', { isWebGLMultipleRenderTargets: true, textures: [{ format: THREE.RGFormat, type: THREE.FloatType }] }, (target) => target.textures[0]],
  ]

  for (const [label, target, colorTexture] of cases) {
    renderToTarget(scene, camera, target, options)
    const data = colorTexture(target).image.data
    assert.ok(Buffer.isBuffer(target.data), `${label} top-level target.data should remain RGBA8`)
    assert.ok(data instanceof Float32Array, `${label} should receive Float32Array data`)
    assert.equal(data.length, 32 * 32 * 2, `${label} should receive two channels per pixel`)
    assert.equal(colorTexture(target).source.data.data, data, `${label} source should reference typed data`)
    assert.ok(data[center] > 0.5, `${label} red channel should be normalized (${data[center]})`)
    assert.ok(data[center + 1] < 0.05, `${label} green channel should stay near zero (${data[center + 1]})`)
  }

  const optionsTarget = { textures: [{ format: THREE.RGFormat, type: THREE.FloatType }] }
  const returned = renderRgba(scene, camera, { ...options, target: optionsTarget })
  const data = optionsTarget.textures[0].image.data
  assert.equal(optionsTarget.data, returned)
  assert.ok(data instanceof Float32Array, 'options.target target.textures[0] should receive Float32Array data')
  assert.equal(data.length, 32 * 32 * 2)
  assert.ok(data[center] > 0.5, `options.target red channel should be normalized (${data[center]})`)
  assert.ok(data[center + 1] < 0.05, `options.target green channel should stay near zero (${data[center + 1]})`)
})
