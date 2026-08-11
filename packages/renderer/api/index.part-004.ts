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
import { RendererBackendState } from './index.part-003'
import { Renderer } from './index.part-008'
import { rendererStateBoolean } from './index.part-014'
import { assertComputeNodeLike, unsupportedNodeOperationError } from './index.part-016'
import { assertConstructorFunction, assertFiniteInteger, assertFunction, assertNonEmptyString, assertWeakMapKey, assertWeakMapKeyArray, isNodeLike, rendererNodeFrame, rendererSimpleHash } from './index.part-017'
export class RendererNodesState {
  readonly nodeBuilderCache = new Map<unknown, unknown>()
  readonly callHashCache = new Map<unknown, unknown>()
  readonly groupsData = new WeakMap<object, Record<string, unknown>>()
  readonly cacheLib: Record<string, WeakMap<object, unknown>> = {}
  modelViewMatrix: unknown = null
  modelNormalViewMatrix: unknown = null
  private data = new WeakMap<object, Record<string, unknown>>()
  private nodeFrameValue = rendererNodeFrame()
  private outputNodeCacheKeys = new WeakMap<object, string>()

  constructor(readonly renderer: Renderer, readonly backend: RendererBackendState) {}

  get nodeFrame(): Record<string, unknown> {
    return this.nodeFrameValue
  }

  set nodeFrame(value: Record<string, unknown>) {
    if (value == null || typeof value !== 'object' || Array.isArray(value)) {
      throw new TypeError('Renderer.nodes.nodeFrame must be an object.')
    }
    this.nodeFrameValue = value
  }

  get(object: unknown): Record<string, unknown> {
    assertWeakMapKey(object, 'Renderer.nodes.get object')
    let map = this.data.get(object)
    if (map === undefined) {
      map = {}
      this.data.set(object, map)
    }
    return map
  }

  has(object: unknown): boolean {
    assertWeakMapKey(object, 'Renderer.nodes.has object')
    return this.data.has(object)
  }

  delete(object: unknown): Record<string, unknown> | null {
    assertWeakMapKey(object, 'Renderer.nodes.delete object')
    const map = this.data.get(object) ?? null
    this.data.delete(object)
    return map
  }

  dispose(): void {
    this.data = new WeakMap()
    this.nodeBuilderCache.clear()
    this.callHashCache.clear()
    for (const key of Object.keys(this.cacheLib)) {
      delete this.cacheLib[key]
    }
    this.nodeFrameValue = rendererNodeFrame()
    this.outputNodeCacheKeys = new WeakMap()
  }

  updateGroup(nodeUniformsGroup: unknown): boolean {
    if (nodeUniformsGroup == null || typeof nodeUniformsGroup !== 'object' || Array.isArray(nodeUniformsGroup)) {
      throw new TypeError('Renderer.nodes.updateGroup nodeUniformsGroup must be an object.')
    }
    const groupNode = (nodeUniformsGroup as { groupNode?: unknown }).groupNode
    if (groupNode == null || typeof groupNode !== 'object' || Array.isArray(groupNode)) {
      throw new TypeError('Renderer.nodes.updateGroup nodeUniformsGroup.groupNode must be an object.')
    }
    const name = (groupNode as { name?: unknown }).name
    if (name === 'object') return true
    const groupData = this.get(nodeUniformsGroup)
    if (name === 'render') {
      const renderId = this.nodeFrameValue.renderId
      if (groupData.renderId !== renderId) {
        groupData.renderId = renderId
        return true
      }
      return false
    }
    if (name === 'frame') {
      const frameId = this.nodeFrameValue.frameId
      if (groupData.frameId !== frameId) {
        groupData.frameId = frameId
        return true
      }
      return false
    }
    const version = (groupNode as { version?: unknown }).version
    if (groupData.version !== version) {
      groupData.version = version
      return true
    }
    return false
  }

  getForRenderCacheKey(renderObject: unknown): unknown {
    assertWeakMapKey(renderObject, 'Renderer.nodes.getForRenderCacheKey renderObject')
    return (renderObject as { initialCacheKey?: unknown }).initialCacheKey
      ?? this.backend.getRenderCacheKey(renderObject)
  }

  getForRender(renderObject: unknown): never {
    assertWeakMapKey(renderObject, 'Renderer.nodes.getForRender renderObject')
    throw unsupportedNodeOperationError('Renderer.nodes.getForRender', 'render-object shader-node builder creation')
  }

  getForCompute(computeNode: unknown): never {
    assertComputeNodeLike(computeNode, 'Renderer.nodes.getForCompute computeNode')
    throw unsupportedNodeOperationError('Renderer.nodes.getForCompute', 'compute shader-node builder creation')
  }

  _createNodeBuilderState(_nodeBuilder: unknown): never {
    throw unsupportedNodeOperationError('Renderer.nodes._createNodeBuilderState', 'shader-node builder state creation')
  }

  updateEnvironment(scene: unknown): void {
    const sceneData = this.getSceneNodeData(scene, 'Renderer.nodes.updateEnvironment scene')
    const environmentNode = (scene as { environmentNode?: unknown }).environmentNode
    if (isNodeLike(environmentNode)) {
      sceneData.environmentNode = environmentNode
    } else {
      delete sceneData.environmentNode
    }
  }

  updateBackground(scene: unknown): void {
    const sceneData = this.getSceneNodeData(scene, 'Renderer.nodes.updateBackground scene')
    const backgroundNode = (scene as { backgroundNode?: unknown }).backgroundNode
    if (isNodeLike(backgroundNode)) {
      sceneData.backgroundNode = backgroundNode
    } else {
      delete sceneData.backgroundNode
    }
  }

  updateFog(scene: unknown): void {
    const sceneData = this.getSceneNodeData(scene, 'Renderer.nodes.updateFog scene')
    const fogNode = (scene as { fogNode?: unknown }).fogNode
    if (isNodeLike(fogNode)) {
      sceneData.fogNode = fogNode
    } else {
      delete sceneData.fogNode
    }
  }

  getEnvironmentNode(scene: unknown): unknown {
    this.updateEnvironment(scene)
    return (scene as { environmentNode?: unknown }).environmentNode
      ?? this.get(scene).environmentNode
      ?? null
  }

  getBackgroundNode(scene: unknown): unknown {
    this.updateBackground(scene)
    return (scene as { backgroundNode?: unknown }).backgroundNode
      ?? this.get(scene).backgroundNode
      ?? null
  }

  getFogNode(scene: unknown): unknown {
    this.updateFog(scene)
    return (scene as { fogNode?: unknown }).fogNode
      ?? this.get(scene).fogNode
      ?? null
  }

  getCacheKey(scene: unknown, lightsNode: unknown = null): number {
    assertWeakMapKey(scene, 'Renderer.nodes.getCacheKey scene')
    let key = this.renderer.shadowMap.enabled ? 1 : 0
    if (lightsNode != null) {
      assertWeakMapKey(lightsNode, 'Renderer.nodes.getCacheKey lightsNode')
      const lightCacheKey = (lightsNode as { getCacheKey?: unknown }).getCacheKey
      if (typeof lightCacheKey === 'function') {
        key = rendererSimpleHash(`${key}:${String(lightCacheKey.call(lightsNode, true))}`)
      }
    }
    return key
  }

  get isToneMappingState(): boolean {
    return this.renderer.getRenderTarget() == null
  }

  getCacheNode(type: unknown, object: unknown, callback: unknown, forceUpdate: unknown = false): unknown {
    assertNonEmptyString(type, 'Renderer.nodes.getCacheNode type')
    assertWeakMapKey(object, 'Renderer.nodes.getCacheNode object')
    assertFunction(callback, 'Renderer.nodes.getCacheNode callback')
    const shouldForceUpdate = rendererStateBoolean(forceUpdate, 'Renderer.nodes.getCacheNode forceUpdate')
    let nodeCache = this.cacheLib[type]
    if (nodeCache === undefined) {
      nodeCache = new WeakMap()
      this.cacheLib[type] = nodeCache
    }
    let node = nodeCache.get(object)
    if (node === undefined || shouldForceUpdate) {
      node = callback()
      nodeCache.set(object, node)
    }
    return node
  }

  getNodeFrame(
    renderer: unknown = this.renderer,
    scene: unknown = null,
    object: unknown = null,
    camera: unknown = null,
    material: unknown = null,
  ): Record<string, unknown> {
    const nodeFrame = this.nodeFrameValue
    nodeFrame.renderer = renderer
    nodeFrame.scene = scene
    nodeFrame.object = object
    nodeFrame.camera = camera
    nodeFrame.material = material
    return nodeFrame
  }

  getNodeFrameForRender(renderObject: unknown): Record<string, unknown> {
    assertWeakMapKey(renderObject, 'Renderer.nodes.getNodeFrameForRender renderObject')
    return this.getNodeFrame(
      (renderObject as { renderer?: unknown }).renderer ?? this.renderer,
      (renderObject as { scene?: unknown }).scene ?? null,
      (renderObject as { object?: unknown }).object ?? null,
      (renderObject as { camera?: unknown }).camera ?? null,
      (renderObject as { material?: unknown }).material ?? null,
    )
  }

  getOutputCacheKey(): string {
    return `${this.renderer.toneMapping},${this.renderer.currentColorSpace}`
  }

  hasOutputChange(outputTarget: unknown): boolean {
    assertWeakMapKey(outputTarget, 'Renderer.nodes.hasOutputChange outputTarget')
    return this.outputNodeCacheKeys.get(outputTarget) !== this.getOutputCacheKey()
  }

  getOutputNode(outputTarget: unknown): Record<string, unknown> {
    assertWeakMapKey(outputTarget, 'Renderer.nodes.getOutputNode outputTarget')
    const cacheKey = this.getOutputCacheKey()
    this.outputNodeCacheKeys.set(outputTarget, cacheKey)
    return {
      isNode: true,
      isHeadlessRendererOutputNode: true,
      outputTarget,
      toneMapping: this.renderer.toneMapping,
      outputColorSpace: this.renderer.currentColorSpace,
    }
  }

  updateBefore(renderObject: unknown): never {
    assertWeakMapKey(renderObject, 'Renderer.nodes.updateBefore renderObject')
    throw unsupportedNodeOperationError('Renderer.nodes.updateBefore', 'shader-node updateBefore lifecycle dispatch')
  }

  updateAfter(renderObject: unknown): never {
    assertWeakMapKey(renderObject, 'Renderer.nodes.updateAfter renderObject')
    throw unsupportedNodeOperationError('Renderer.nodes.updateAfter', 'shader-node updateAfter lifecycle dispatch')
  }

  updateForCompute(computeNode: unknown): never {
    assertComputeNodeLike(computeNode, 'Renderer.nodes.updateForCompute computeNode')
    throw unsupportedNodeOperationError('Renderer.nodes.updateForCompute', 'compute shader-node update lifecycle dispatch')
  }

  updateForRender(renderObject: unknown): never {
    assertWeakMapKey(renderObject, 'Renderer.nodes.updateForRender renderObject')
    throw unsupportedNodeOperationError('Renderer.nodes.updateForRender', 'render shader-node update lifecycle dispatch')
  }

  needsRefresh(renderObject: unknown): boolean {
    assertWeakMapKey(renderObject, 'Renderer.nodes.needsRefresh renderObject')
    return false
  }

  private getSceneNodeData(scene: unknown, label: string): Record<string, unknown> {
    assertWeakMapKey(scene, label)
    return this.get(scene)
  }
}

export class RendererNodeLibraryState {
  readonly lightNodes = new WeakMap<object, (...args: unknown[]) => unknown>()
  readonly materialNodes = new Map<string, new (...args: unknown[]) => Record<string, unknown>>()
  readonly toneMappingNodes = new Map<number, (...args: unknown[]) => unknown>()

  fromMaterial(material: unknown): unknown {
    if (material == null || typeof material !== 'object' || Array.isArray(material)) {
      throw new TypeError('Renderer.library.fromMaterial material must be a material-like object.')
    }
    if ((material as { isNodeMaterial?: unknown }).isNodeMaterial === true) return material
    const materialType = (material as { type?: unknown }).type
    if (typeof materialType !== 'string' || materialType.length === 0) return null
    const NodeMaterialClass = this.getMaterialNodeClass(materialType)
    if (NodeMaterialClass === null) return null
    const nodeMaterial = new NodeMaterialClass()
    Object.assign(nodeMaterial, material)
    return nodeMaterial
  }

  addToneMapping(toneMappingNode: unknown, toneMapping: unknown): void {
    assertFunction(toneMappingNode, 'Renderer.library.addToneMapping toneMappingNode')
    assertFiniteInteger(toneMapping, 'Renderer.library.addToneMapping toneMapping')
    this.addType(toneMappingNode, toneMapping, this.toneMappingNodes)
  }

  getToneMappingFunction(toneMapping: unknown): ((...args: unknown[]) => unknown) | null {
    assertFiniteInteger(toneMapping, 'Renderer.library.getToneMappingFunction toneMapping')
    return this.toneMappingNodes.get(toneMapping as number) ?? null
  }

  getMaterialNodeClass(materialType: unknown): (new (...args: unknown[]) => Record<string, unknown>) | null {
    assertNonEmptyString(materialType, 'Renderer.library.getMaterialNodeClass materialType')
    return this.materialNodes.get(materialType) ?? null
  }

  addMaterial(materialNodeClass: unknown, materialClassType: unknown): void {
    assertConstructorFunction(materialNodeClass, 'Renderer.library.addMaterial materialNodeClass')
    assertNonEmptyString(materialClassType, 'Renderer.library.addMaterial materialClassType')
    this.addType(materialNodeClass, materialClassType, this.materialNodes)
  }

  getLightNodeClass(light: unknown): ((...args: unknown[]) => unknown) | null {
    assertWeakMapKey(light, 'Renderer.library.getLightNodeClass light')
    return this.lightNodes.get(light) ?? null
  }

  addLight(lightNodeClass: unknown, lightClass: unknown): void {
    assertFunction(lightNodeClass, 'Renderer.library.addLight lightNodeClass')
    assertConstructorFunction(lightClass, 'Renderer.library.addLight lightClass')
    this.addClass(lightNodeClass, lightClass, this.lightNodes)
  }

  addType<T>(nodeClass: unknown, type: unknown, library: Map<any, T>): void {
    assertFunction(nodeClass, 'Renderer.library.addType nodeClass')
    if ((typeof type !== 'string' && typeof type !== 'number') || (typeof type === 'string' && type.length === 0)) {
      throw new TypeError('Renderer.library.addType type must be a non-empty string or integer.')
    }
    if (typeof type === 'number') assertFiniteInteger(type, 'Renderer.library.addType type')
    if (!library.has(type)) {
      library.set(type, nodeClass as T)
    }
  }

  addClass<T>(nodeClass: unknown, baseClass: unknown, library: WeakMap<object, T>): void {
    assertFunction(nodeClass, 'Renderer.library.addClass nodeClass')
    assertConstructorFunction(baseClass, 'Renderer.library.addClass baseClass')
    if (!library.has(baseClass)) {
      library.set(baseClass, nodeClass as T)
    }
  }
}

export class RendererLightingNodeState {
  readonly isLightsNode = true
  private lightsValue: unknown[] = []

  constructor(lights: unknown[] = []) {
    this.setLights(lights)
  }

  setLights(lights: unknown[] = []): this {
    if (!Array.isArray(lights)) {
      throw new TypeError('Renderer.lighting lights must be an array.')
    }
    this.lightsValue = [...lights]
    return this
  }

  getLights(): unknown[] {
    return [...this.lightsValue]
  }
}

export class RendererLightingState {
  readonly weakMap = new WeakMap<object, WeakMap<object, unknown>>()
  readonly defaultLightsNode = new RendererLightingNodeState()

  createNode(lights: unknown[] = []): RendererLightingNodeState {
    return new RendererLightingNodeState(lights)
  }

  get(keys: unknown): unknown {
    const keyPath = assertWeakMapKeyArray(keys, 'Renderer.lighting.get keys')
    let map: WeakMap<object, unknown> | undefined = this.weakMap
    for (let i = 0; i < keyPath.length; i += 1) {
      map = map.get(keyPath[i]) as WeakMap<object, unknown> | undefined
      if (map === undefined) return undefined
    }
    return map.get(keyPath[keyPath.length - 1])
  }

  set(keys: unknown, value: unknown): this {
    const keyPath = assertWeakMapKeyArray(keys, 'Renderer.lighting.set keys')
    let map: WeakMap<object, unknown> = this.weakMap
    for (const key of keyPath) {
      if (!map.has(key)) {
        map.set(key, new WeakMap<object, unknown>())
      }
      map = map.get(key) as WeakMap<object, unknown>
    }
    map.set(keyPath[keyPath.length - 1], value)
    return this
  }

  delete(keys: unknown): boolean {
    const keyPath = assertWeakMapKeyArray(keys, 'Renderer.lighting.delete keys')
    let map: WeakMap<object, unknown> | undefined = this.weakMap
    for (let i = 0; i < keyPath.length; i += 1) {
      map = map.get(keyPath[i]) as WeakMap<object, unknown> | undefined
      if (map === undefined) return false
    }
    return map.delete(keyPath[keyPath.length - 1])
  }

  getNode(scene: unknown, camera: unknown): RendererLightingNodeState {
    assertWeakMapKey(scene, 'Renderer.lighting.getNode scene')
    if ((scene as { isQuadMesh?: unknown }).isQuadMesh === true) return this.defaultLightsNode
    assertWeakMapKey(camera, 'Renderer.lighting.getNode camera')
    const keys = [scene, camera]
    let node = this.get(keys) as RendererLightingNodeState | undefined
    if (node === undefined) {
      node = this.createNode()
      this.set(keys, node)
    }
    return node
  }
}
