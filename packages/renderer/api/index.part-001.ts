import type {
  ThreeSceneRootLike,
  ThreeCameraLike,
  ThreeCubeCameraLike,
  ThreeRenderCameraLike,
  RenderOptions,
  RenderTargetLike,
  RenderTargetTextureLike,
  RenderTargetImageLike,
  RenderPixelRectLike,
  RenderSizeLike,
  ThreeColorLike,
  NativeRenderScene,
  NativeCamera,
  NativeSceneMesh,
  NativeSceneLight,
  RenderMode,
  RenderOutputColorSpace,
  Color4,
  RenderObjectIdEntry,
  ThreeEulerLike,
  ThreePlaneLike,
  ThreeTextureLike,
  ThreeMaterialLike,
  ThreeObject3DLike,
  RenderSortFunction,
  RenderAnimationLoopCallback,
  RendererParametersLike,
  RendererContextAttributesLike,
  RendererInspectorLike,
} from './types'
import { resolveSize, cameraViewProjection, cameraViewMatrix, cameraWorldPosition } from './camera'
import { DEFAULT_BACKGROUND_COLOR, cssColorStringToArray, resolveBackground, validatedColorLikeToArray } from './color'
import { commitNativeMeshPayloadCache, createSceneExtractionCache, flattenScene, type SceneExtractionCache, type ShadowMaterialMode } from './scene'
import { extractLights, extractAmbientLight, extractAmbientIntensity, extractLightProbe } from './lights'
import { canvasLikeImageToRgba, extractBackgroundTexture, extractTextureData, isCompressedTextureFormat, resolveEnvironmentMap, resolveSceneOverrideMaterial, type MaterialExtractionContext } from './materials'
import { extractClippingPlanes } from './clipping'
import { validateObjectChildrenTree } from './objects'
import { clamp01, matrixElements } from './math'
import { rendererInfoDrawCount, rendererInfoDrawMode, rendererInfoInstanceCount, rendererInfoTimestampTime, rendererStateBoolean } from './index.part-014'
import { rendererStateShadowMapType } from './index.part-015'
import { assertTimestampQueryType } from './index.part-016'
import { validateThreeSceneRoot, validateTopLevelRenderCamera } from './index.part-021'
// eslint-disable-next-line @typescript-eslint/no-var-requires
export const native = require('../native.js')

export const WEBGL_COORDINATE_SYSTEM = 2000
export const BasicShadowMap = 0
export const PCFShadowMap = 1
export const PCFSoftShadowMap = 2
export const VSMShadowMap = 3
export const SupportedRendererShadowMapTypes = new Set([BasicShadowMap, PCFShadowMap, PCFSoftShadowMap, VSMShadowMap])
export const CullFaceNone = 0
export const CullFaceBack = 1
export const CullFaceFront = 2
export const CullFaceFrontBack = 3
export const SupportedRendererStateCullFaces = new Set([CullFaceNone, CullFaceBack, CullFaceFront, CullFaceFrontBack])
export const NoBlending = 0
export const NormalBlending = 1
export const AdditiveBlending = 2
export const SubtractiveBlending = 3
export const MultiplyBlending = 4
export const CustomBlending = 5
export const SupportedRendererStateBlendingModes = new Set([
  NoBlending,
  NormalBlending,
  AdditiveBlending,
  SubtractiveBlending,
  MultiplyBlending,
  CustomBlending,
])
export const WebGLDrawModePoints = 0x0000
export const WebGLDrawModeLines = 0x0001
export const WebGLDrawModeLineLoop = 0x0002
export const WebGLDrawModeLineStrip = 0x0003
export const WebGLDrawModeTriangles = 0x0004
export const SupportedRendererInfoDrawModes = new Set([
  WebGLDrawModePoints,
  WebGLDrawModeLines,
  WebGLDrawModeLineLoop,
  WebGLDrawModeLineStrip,
  WebGLDrawModeTriangles,
])
export const NoToneMapping = 0
export const LinearToneMapping = 1
export const ReinhardToneMapping = 2
export const CineonToneMapping = 3
export const ACESFilmicToneMapping = 4
export const CustomToneMapping = 5
export const AgXToneMapping = 6
export const NeutralToneMapping = 7
export const SupportedRendererToneMappings = new Set([
  NoToneMapping,
  LinearToneMapping,
  ReinhardToneMapping,
  CineonToneMapping,
  ACESFilmicToneMapping,
  CustomToneMapping,
  AgXToneMapping,
  NeutralToneMapping,
])
export const SupportedRendererPowerPreferences = new Set(['default', 'high-performance', 'low-power'])
export const SupportedTimestampQueryTypes = new Set(['render', 'compute'])
export const RendererBooleanParameters = [
  'alpha',
  'depth',
  'stencil',
  'antialias',
  'premultipliedAlpha',
  'preserveDrawingBuffer',
  'failIfMajorPerformanceCaveat',
] as const
export const DefaultRendererContextAttributes: RendererContextAttributesLike = {
  alpha: false,
  depth: true,
  stencil: false,
  antialias: false,
  premultipliedAlpha: true,
  preserveDrawingBuffer: false,
  powerPreference: 'default',
  failIfMajorPerformanceCaveat: false,
}
export const RendererInspectorOptionalMethods = [
  'getRenderer',
  'init',
  'begin',
  'finish',
  'inspect',
  'computeAsync',
  'beginCompute',
  'finishCompute',
  'beginRender',
  'finishRender',
  'copyTextureToTexture',
  'copyFramebufferToTexture',
] as const

export class RendererShadowMapState {
  private enabledValue = true
  private autoUpdateValue = true
  private needsUpdateValue = false
  private transmittedValue = false
  private typeValue = PCFShadowMap

  get enabled(): boolean {
    return this.enabledValue
  }

  set enabled(value: boolean) {
    this.enabledValue = rendererStateBoolean(value, 'Renderer.shadowMap.enabled')
  }

  get autoUpdate(): boolean {
    return this.autoUpdateValue
  }

  set autoUpdate(value: boolean) {
    this.autoUpdateValue = rendererStateBoolean(value, 'Renderer.shadowMap.autoUpdate')
  }

  get needsUpdate(): boolean {
    return this.needsUpdateValue
  }

  set needsUpdate(value: boolean) {
    this.needsUpdateValue = rendererStateBoolean(value, 'Renderer.shadowMap.needsUpdate')
  }

  get transmitted(): boolean {
    return this.transmittedValue
  }

  set transmitted(value: boolean) {
    this.transmittedValue = rendererStateBoolean(value, 'Renderer.shadowMap.transmitted')
  }

  get type(): number {
    return this.typeValue
  }

  set type(value: number) {
    this.typeValue = rendererStateShadowMapType(value)
  }

  render(lights: unknown, scene: unknown, camera: unknown): void {
    if (!Array.isArray(lights)) {
      throw new TypeError('Renderer.shadowMap.render lights must be an array.')
    }
    validateThreeSceneRoot(scene)
    validateTopLevelRenderCamera(camera)
  }
}

export class RendererInfoState {
  private autoResetValue = true

  calls = 0
  frame = 0

  readonly memory = {
    geometries: 0,
    textures: 0,
  }

  readonly render = {
    calls: 0,
    frameCalls: 0,
    drawCalls: 0,
    triangles: 0,
    points: 0,
    lines: 0,
    timestamp: 0,
    previousFrameCalls: 0,
    timestampCalls: 0,
    frame: 0,
  }

  readonly compute = {
    calls: 0,
    frameCalls: 0,
    timestamp: 0,
    previousFrameCalls: 0,
    timestampCalls: 0,
  }

  programs: unknown[] | null = null

  get autoReset(): boolean {
    return this.autoResetValue
  }

  set autoReset(value: boolean) {
    this.autoResetValue = rendererStateBoolean(value, 'Renderer.info.autoReset')
  }

  update(objectOrCount: unknown, modeOrCount: unknown, instanceCount: unknown = 1): void {
    if (objectOrCount !== null && typeof objectOrCount === 'object' && !Array.isArray(objectOrCount)) {
      this.updateCommonRendererObject(objectOrCount as ThreeObject3DLike, modeOrCount, instanceCount)
      return
    }

    const drawCount = rendererInfoDrawCount(objectOrCount, 'Renderer.info.update count')
    const drawMode = rendererInfoDrawMode(modeOrCount, 'Renderer.info.update mode')
    const instances = rendererInfoInstanceCount(instanceCount, 'Renderer.info.update instanceCount')

    this.render.calls += 1
    this.render.drawCalls += 1

    switch (drawMode) {
      case WebGLDrawModeTriangles:
        this.render.triangles += instances * (drawCount / 3)
        break
      case WebGLDrawModeLines:
        this.render.lines += instances * (drawCount / 2)
        break
      case WebGLDrawModeLineStrip:
        this.render.lines += instances * (drawCount - 1)
        break
      case WebGLDrawModeLineLoop:
        this.render.lines += instances * drawCount
        break
      case WebGLDrawModePoints:
        this.render.points += instances * drawCount
        break
    }
  }

  reset(): void {
    this.render.previousFrameCalls = this.render.frameCalls
    this.compute.previousFrameCalls = this.compute.frameCalls
    this.render.calls = 0
    this.render.frameCalls = 0
    this.render.drawCalls = 0
    this.render.triangles = 0
    this.render.points = 0
    this.render.lines = 0
    this.compute.frameCalls = 0
  }

  dispose(): void {
    this.reset()
    this.calls = 0
    this.compute.calls = 0
    this.render.timestamp = 0
    this.render.previousFrameCalls = 0
    this.render.timestampCalls = 0
    this.compute.timestamp = 0
    this.compute.previousFrameCalls = 0
    this.compute.timestampCalls = 0
    this.memory.geometries = 0
    this.memory.textures = 0
  }

  updateTimestamp(type: unknown, time: unknown): void {
    assertTimestampQueryType(type, 'Renderer.info.updateTimestamp type')
    const elapsed = rendererInfoTimestampTime(time, 'Renderer.info.updateTimestamp time')
    const target = type === 'render' ? this.render : this.compute

    if (target.timestampCalls === 0) {
      target.timestamp = 0
    }

    target.timestamp += elapsed
    target.timestampCalls += 1

    if (target.timestampCalls >= target.previousFrameCalls) {
      target.timestampCalls = 0
    }
  }

  private updateCommonRendererObject(object: ThreeObject3DLike, count: unknown, instanceCount: unknown): void {
    const drawCount = rendererInfoDrawCount(count, 'Renderer.info.update count')
    const instances = rendererInfoInstanceCount(instanceCount, 'Renderer.info.update instanceCount')

    this.render.drawCalls += 1

    if (object.isMesh === true || object.isSprite === true) {
      this.render.triangles += instances * (drawCount / 3)
    } else if (object.isPoints === true) {
      this.render.points += instances * drawCount
    } else if (object.isLineSegments === true) {
      this.render.lines += instances * (drawCount / 2)
    } else if (object.isLineLoop === true) {
      this.render.lines += instances * drawCount
    } else if (object.isLine === true) {
      this.render.lines += instances * (drawCount - 1)
    } else {
      throw new Error(
        'Renderer.info.update object type is not supported. Use a mesh, sprite, points, line, line segments, or line loop object.',
      )
    }
  }
}
