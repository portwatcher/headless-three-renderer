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
import { countRegionPixels, renderRgba } from './scenes.test.part-002.mjs'
test('examples Volume slices render canvas-backed grayscale texture meshes', () => {
  function makeCanvas() {
    const canvas = {
      _width: 0,
      _height: 0,
      _pixels: new Uint8ClampedArray(0),
      get width() {
        return this._width
      },
      set width(value) {
        this._width = Math.max(0, Math.trunc(value))
        this._pixels = new Uint8ClampedArray(this._width * this._height * 4)
      },
      get height() {
        return this._height
      },
      set height(value) {
        this._height = Math.max(0, Math.trunc(value))
        this._pixels = new Uint8ClampedArray(this._width * this._height * 4)
      },
      getContext(type) {
        if (type !== '2d') return null
        return {
          getImageData: (x, y, width, height) => {
            assert.equal(x, 0)
            assert.equal(y, 0)
            assert.equal(width, canvas.width)
            assert.equal(height, canvas.height)
            return {
              data: new Uint8ClampedArray(canvas._pixels),
              width,
              height,
            }
          },
          putImageData: (imageData, x, y) => {
            assert.equal(x, 0)
            assert.equal(y, 0)
            assert.equal(imageData.data.length, canvas._pixels.length)
            canvas._pixels.set(imageData.data)
          },
          drawImage: (source, sx, sy, sw, sh, dx, dy, dw, dh) => {
            assert.equal(sx, 0)
            assert.equal(sy, 0)
            assert.equal(dx, 0)
            assert.equal(dy, 0)
            assert.equal(sw, source.width)
            assert.equal(sh, source.height)
            assert.equal(dw, canvas.width)
            assert.equal(dh, canvas.height)
            for (let y = 0; y < canvas.height; y += 1) {
              const sourceY = Math.min(source.height - 1, Math.floor(y * source.height / canvas.height))
              for (let x = 0; x < canvas.width; x += 1) {
                const sourceX = Math.min(source.width - 1, Math.floor(x * source.width / canvas.width))
                const sourceOffset = (sourceY * source.width + sourceX) * 4
                const targetOffset = (y * canvas.width + x) * 4
                canvas._pixels[targetOffset] = source._pixels[sourceOffset]
                canvas._pixels[targetOffset + 1] = source._pixels[sourceOffset + 1]
                canvas._pixels[targetOffset + 2] = source._pixels[sourceOffset + 2]
                canvas._pixels[targetOffset + 3] = source._pixels[sourceOffset + 3]
              }
            }
          },
        }
      },
    }
    return canvas
  }

  const previousDocument = globalThis.document
  let slice
  try {
    globalThis.document = {
      createElement(type) {
        assert.equal(type, 'canvas')
        return makeCanvas()
      },
    }

    const values = new Uint8Array([
      0, 32, 96,
      128, 180, 220,
      255, 64, 160,
      32, 96, 160,
      80, 144, 208,
      250, 120, 40,
      12, 48, 84,
      120, 156, 192,
      228, 240, 252,
    ])
    const volume = new Volume(3, 3, 3, 'uint8', values.buffer)
    volume.RASDimensions = [3, 3, 3]
    volume.inverseMatrix = new THREE.Matrix4().identity()
    volume.windowLow = 0
    volume.windowHigh = 255
    volume.lowerThreshold = 1

    slice = volume.extractSlice('z', 1)
    slice.updateGeometry()
    slice.repaint()
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x000000)
    scene.add(slice.mesh)

    const camera = new THREE.OrthographicCamera(-2, 2, 2, -2, 0.01, 10)
    camera.position.set(0, 0, 4)
    camera.lookAt(0, 0, 0)
    camera.updateMatrixWorld(true)

    const width = 96
    const height = 96
    const rgba = renderRgba(scene, camera, { width, height })

    assert.equal(volume.getData(2, 1, 1), 208)
    assert.deepEqual(volume.reverseAccess(volume.access(2, 1, 1)), [2, 1, 1])
    assert.equal(volume.sliceList[0], slice)
    assert.equal(slice instanceof VolumeSlice, true)
    assert.equal(slice.iLength, 3)
    assert.equal(slice.jLength, 3)
    assert.equal(slice.canvas.width, 3)
    assert.equal(slice.canvas.height, 3)
    assert.equal(slice.canvasBuffer.width, 3)
    assert.equal(slice.canvasBuffer.height, 3)
    assert.equal(slice.geometry.getAttribute('position').count, 4)
    assert.equal(slice.mesh.material.map.image, slice.canvas)
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => r > 30 || g > 30 || b > 30) > 3000,
      'VolumeSlice canvas texture should render visible grayscale pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => r > 170 && g > 170 && b > 170) > 400,
      'VolumeSlice canvas texture should render bright voxels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => r > 70 && r < 170 && Math.abs(r - g) < 4 && Math.abs(g - b) < 4) > 700,
      'VolumeSlice canvas texture should render midtone grayscale voxels',
    )

    volume.lowerThreshold = 80
    assert.equal(slice.geometryNeedsUpdate, true)
    volume.repaintAllSlices()
    assert.equal(slice.geometryNeedsUpdate, false)
  } finally {
    if (slice) {
      slice.mesh.material.map.dispose()
      slice.mesh.material.dispose()
      slice.mesh.geometry.dispose()
    }
    if (previousDocument === undefined) {
      delete globalThis.document
    } else {
      globalThis.document = previousDocument
    }
  }
})

test('examples NRRDLoader parses volumes into renderable slice meshes', () => {
  function makeCanvas() {
    const canvas = {
      _width: 0,
      _height: 0,
      _pixels: new Uint8ClampedArray(0),
      get width() {
        return this._width
      },
      set width(value) {
        this._width = Math.max(0, Math.trunc(value))
        this._pixels = new Uint8ClampedArray(this._width * this._height * 4)
      },
      get height() {
        return this._height
      },
      set height(value) {
        this._height = Math.max(0, Math.trunc(value))
        this._pixels = new Uint8ClampedArray(this._width * this._height * 4)
      },
      getContext(type) {
        if (type !== '2d') return null
        return {
          getImageData: (x, y, width, height) => {
            assert.equal(x, 0)
            assert.equal(y, 0)
            assert.equal(width, canvas.width)
            assert.equal(height, canvas.height)
            return {
              data: new Uint8ClampedArray(canvas._pixels),
              width,
              height,
            }
          },
          putImageData: (imageData, x, y) => {
            assert.equal(x, 0)
            assert.equal(y, 0)
            assert.equal(imageData.data.length, canvas._pixels.length)
            canvas._pixels.set(imageData.data)
          },
          drawImage: (source, sx, sy, sw, sh, dx, dy, dw, dh) => {
            assert.equal(sx, 0)
            assert.equal(sy, 0)
            assert.equal(dx, 0)
            assert.equal(dy, 0)
            assert.equal(sw, source.width)
            assert.equal(sh, source.height)
            assert.equal(dw, canvas.width)
            assert.equal(dh, canvas.height)
            for (let row = 0; row < canvas.height; row += 1) {
              const sourceY = Math.min(source.height - 1, Math.floor(row * source.height / canvas.height))
              for (let col = 0; col < canvas.width; col += 1) {
                const sourceX = Math.min(source.width - 1, Math.floor(col * source.width / canvas.width))
                const sourceOffset = (sourceY * source.width + sourceX) * 4
                const targetOffset = (row * canvas.width + col) * 4
                canvas._pixels[targetOffset] = source._pixels[sourceOffset]
                canvas._pixels[targetOffset + 1] = source._pixels[sourceOffset + 1]
                canvas._pixels[targetOffset + 2] = source._pixels[sourceOffset + 2]
                canvas._pixels[targetOffset + 3] = source._pixels[sourceOffset + 3]
              }
            }
          },
        }
      },
    }
    return canvas
  }

  const values = [
    0, 32, 96,
    128, 180, 220,
    255, 64, 160,
    32, 96, 160,
    80, 144, 208,
    250, 120, 40,
    12, 48, 84,
    120, 156, 192,
    228, 240, 252,
  ]
  const payload = new TextEncoder().encode([
    'NRRD0005',
    'type: uchar',
    'dimension: 3',
    'sizes: 3 3 3',
    'encoding: ascii',
    '# payload follows',
    '',
    values.join(' '),
  ].join('\n'))

  const previousDocument = globalThis.document
  let slice
  try {
    globalThis.document = {
      createElement(type) {
        assert.equal(type, 'canvas')
        return makeCanvas()
      },
    }

    const volume = new NRRDLoader().parse(payload.buffer.slice(payload.byteOffset, payload.byteOffset + payload.byteLength))
    volume.lowerThreshold = 1
    slice = volume.extractSlice('z', 1)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x000000)
    scene.add(slice.mesh)

    const camera = new THREE.OrthographicCamera(-2, 2, 2, -2, 0.01, 10)
    camera.position.set(0, 0, 4)
    camera.lookAt(0, 0, 0)
    camera.updateMatrixWorld(true)

    const width = 96
    const height = 96
    const rgba = renderRgba(scene, camera, { width, height })

    assert.deepEqual(volume.dimensions, [3, 3, 3])
    assert.deepEqual(volume.spacing, [1, 1, 1])
    assert.deepEqual(volume.RASDimensions, [3, 3, 3])
    assert.equal(volume.data instanceof Uint8Array, true)
    assert.equal(volume.getData(2, 1, 1), 208)
    assert.equal(slice instanceof VolumeSlice, true)
    assert.equal(slice.canvas.width, 3)
    assert.equal(slice.canvas.height, 3)
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => r > 30 || g > 30 || b > 30) > 3000,
      'NRRDLoader parsed VolumeSlice should render visible grayscale pixels',
    )
    assert.ok(
      countRegionPixels(rgba, width, height, 0, 0, width, height, (r, g, b) => r > 170 && g > 170 && b > 170) > 400,
      'NRRDLoader parsed VolumeSlice should render bright voxels',
    )
  } finally {
    if (slice) {
      slice.mesh.material.map.dispose()
      slice.mesh.material.dispose()
      slice.mesh.geometry.dispose()
    }
    if (previousDocument === undefined) {
      delete globalThis.document
    } else {
      globalThis.document = previousDocument
    }
  }
})
