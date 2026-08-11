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
import { RendererLightingNodeState, RendererLightingState } from './index.part-004'
import { RendererRenderListItem, RendererRenderListSort } from './index.part-006'
import { assertWeakMapKey, assertWeakMapKeyArray, rendererRenderListId, rendererRenderListMaterialVariant, rendererRenderListNeedsDoublePass, rendererRenderListOpaqueSort, rendererRenderListRenderOrder, rendererRenderListTransparentSort } from './index.part-017'
import { assertFiniteNumberOption } from './index.part-018'
export class RendererRenderList {
  readonly renderItems: RendererRenderListItem[] = []
  renderItemsIndex = 0

  readonly opaque: RendererRenderListItem[] = []
  readonly transmissive: RendererRenderListItem[] = []
  readonly transparentDoublePass: RendererRenderListItem[] = []
  readonly transparent: RendererRenderListItem[] = []
  readonly bundles: unknown[] = []
  readonly lightsArray: unknown[] = []
  readonly lightsNode: RendererLightingNodeState
  occlusionQueryCount = 0

  constructor(
    lighting: RendererLightingState | null = null,
    readonly scene: unknown = null,
    readonly camera: unknown = null,
  ) {
    this.lightsNode = lighting && scene && typeof scene === 'object' && camera && typeof camera === 'object'
      ? lighting.getNode(scene, camera)
      : new RendererLightingNodeState()
  }

  init(): void {
    this.begin()
  }

  begin(): this {
    this.renderItemsIndex = 0
    this.opaque.length = 0
    this.transmissive.length = 0
    this.transparentDoublePass.length = 0
    this.transparent.length = 0
    this.bundles.length = 0
    this.lightsArray.length = 0
    this.occlusionQueryCount = 0
    return this
  }

  push(
    object: unknown,
    geometry: unknown,
    material: unknown,
    groupOrder = 0,
    z = 0,
    group: unknown = null,
    clippingContext: unknown = null,
  ): void {
    const renderItem = this.getNextRenderItem(object, geometry, material, groupOrder, z, group, clippingContext)
    if ((object as { occlusionTest?: unknown } | null)?.occlusionTest === true) {
      this.occlusionQueryCount += 1
    }
    this.pushRenderItem(renderItem, material, false)
  }

  unshift(
    object: unknown,
    geometry: unknown,
    material: unknown,
    groupOrder = 0,
    z = 0,
    group: unknown = null,
    clippingContext: unknown = null,
  ): void {
    const renderItem = this.getNextRenderItem(object, geometry, material, groupOrder, z, group, clippingContext)
    this.pushRenderItem(renderItem, material, true)
  }

  pushBundle(group: unknown): void {
    this.bundles.push(group)
  }

  pushLight(light: unknown): void {
    this.lightsArray.push(light)
  }

  sort(customOpaqueSort?: RendererRenderListSort | null, customTransparentSort?: RendererRenderListSort | null): void {
    if (customOpaqueSort !== undefined && customOpaqueSort !== null && typeof customOpaqueSort !== 'function') {
      throw new TypeError('Renderer.renderLists list opaque sort must be a function or null.')
    }
    if (customTransparentSort !== undefined && customTransparentSort !== null && typeof customTransparentSort !== 'function') {
      throw new TypeError('Renderer.renderLists list transparent sort must be a function or null.')
    }
    this.opaque.sort(customOpaqueSort ?? rendererRenderListOpaqueSort)
    this.transmissive.sort(customTransparentSort ?? rendererRenderListTransparentSort)
    this.transparentDoublePass.sort(customTransparentSort ?? rendererRenderListTransparentSort)
    this.transparent.sort(customTransparentSort ?? rendererRenderListTransparentSort)
  }

  finish(): void {
    this.lightsNode.setLights(this.lightsArray)
    for (let i = this.renderItemsIndex; i < this.renderItems.length; i += 1) {
      this.renderItems[i].id = null
      this.renderItems[i].object = null
      this.renderItems[i].geometry = null
      this.renderItems[i].material = null
      this.renderItems[i].materialVariant = null
      this.renderItems[i].groupOrder = null
      this.renderItems[i].renderOrder = null
      this.renderItems[i].z = null
      this.renderItems[i].group = null
      this.renderItems[i].clippingContext = null
    }
  }

  private getNextRenderItem(
    object: unknown,
    geometry: unknown,
    material: unknown,
    groupOrder: number,
    z: number,
    group: unknown,
    clippingContext: unknown,
  ): RendererRenderListItem {
    assertFiniteNumberOption(groupOrder, 'Renderer.renderLists list groupOrder')
    assertFiniteNumberOption(z, 'Renderer.renderLists list z')
    let renderItem = this.renderItems[this.renderItemsIndex]
    if (renderItem === undefined) {
      renderItem = {
        id: rendererRenderListId(object),
        object,
        geometry,
        material,
        materialVariant: rendererRenderListMaterialVariant(object),
        groupOrder,
        renderOrder: rendererRenderListRenderOrder(object),
        z,
        group,
        clippingContext,
      }
      this.renderItems[this.renderItemsIndex] = renderItem
    } else {
      renderItem.id = rendererRenderListId(object)
      renderItem.object = object
      renderItem.geometry = geometry
      renderItem.material = material
      renderItem.materialVariant = rendererRenderListMaterialVariant(object)
      renderItem.groupOrder = groupOrder
      renderItem.renderOrder = rendererRenderListRenderOrder(object)
      renderItem.z = z
      renderItem.group = group
      renderItem.clippingContext = clippingContext
    }
    this.renderItemsIndex += 1
    return renderItem
  }

  private pushRenderItem(renderItem: RendererRenderListItem, material: unknown, unshift: boolean): void {
    const record = material && typeof material === 'object' ? material as Record<string, unknown> : undefined
    const hasTransmission = typeof record?.transmission === 'number' && record.transmission > 0
    const transparent = record?.transparent === true || hasTransmission
    if (hasTransmission) {
      this.pushInto(this.transmissive, renderItem, unshift)
    }
    if (transparent) {
      if (rendererRenderListNeedsDoublePass(record)) {
        this.pushInto(this.transparentDoublePass, renderItem, unshift)
      }
      this.pushInto(this.transparent, renderItem, unshift)
      return
    }
    this.pushInto(this.opaque, renderItem, unshift)
  }

  private pushInto(list: RendererRenderListItem[], item: RendererRenderListItem, unshift: boolean): void {
    if (unshift) list.unshift(item)
    else list.push(item)
  }
}

export class RendererChainMapState {
  weakMap = new WeakMap<object, WeakMap<object, unknown>>()

  get(keys: unknown): unknown {
    const keyPath = assertWeakMapKeyArray(keys, 'Renderer.renderLists.lists.get keys')
    let map: WeakMap<object, unknown> | undefined = this.weakMap
    for (let i = 0; i < keyPath.length; i += 1) {
      map = map.get(keyPath[i]) as WeakMap<object, unknown> | undefined
      if (map === undefined) return undefined
    }
    return map.get(keyPath[keyPath.length - 1])
  }

  set(keys: unknown, value: unknown): this {
    const keyPath = assertWeakMapKeyArray(keys, 'Renderer.renderLists.lists.set keys')
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
    const keyPath = assertWeakMapKeyArray(keys, 'Renderer.renderLists.lists.delete keys')
    let map: WeakMap<object, unknown> | undefined = this.weakMap
    for (let i = 0; i < keyPath.length; i += 1) {
      map = map.get(keyPath[i]) as WeakMap<object, unknown> | undefined
      if (map === undefined) return false
    }
    return map.delete(keyPath[keyPath.length - 1])
  }

  clear(): void {
    this.weakMap = new WeakMap()
  }
}

export class RendererRenderListsState {
  readonly lighting: RendererLightingState
  readonly lists = new RendererChainMapState()
  private depthLists = new WeakMap<object, RendererRenderList[]>()

  constructor(lighting: RendererLightingState) {
    this.lighting = lighting
  }

  get(scene: object, renderCallDepthOrCamera: unknown = 0): RendererRenderList {
    assertWeakMapKey(scene, 'Renderer.renderLists.get scene')
    if (typeof renderCallDepthOrCamera !== 'number') {
      assertWeakMapKey(renderCallDepthOrCamera, 'Renderer.renderLists.get camera')
      const keys = [scene, renderCallDepthOrCamera]
      let cameraList = this.lists.get(keys) as RendererRenderList | undefined
      if (cameraList === undefined) {
        cameraList = new RendererRenderList(this.lighting, scene, renderCallDepthOrCamera)
        this.lists.set(keys, cameraList)
      }
      return cameraList
    }

    const renderCallDepth = renderCallDepthOrCamera
    if (!Number.isInteger(renderCallDepth) || renderCallDepth < 0) {
      throw new TypeError(`Renderer.renderLists.get renderCallDepth must be a non-negative integer; received ${String(renderCallDepth)}.`)
    }
    let listArray = this.depthLists.get(scene)
    if (listArray === undefined) {
      listArray = []
      this.depthLists.set(scene, listArray)
    }
    let list = listArray[renderCallDepth]
    if (list === undefined) {
      list = new RendererRenderList(this.lighting, scene, null)
      listArray[renderCallDepth] = list
    }
    return list
  }

  dispose(): void {
    this.depthLists = new WeakMap()
    this.lists.clear()
  }
}

export class RendererRenderLightsState {
  readonly state = {
    version: 0,
    hash: {
      directionalLength: -1,
      pointLength: -1,
      spotLength: -1,
      rectAreaLength: -1,
      hemiLength: -1,
      numDirectionalShadows: -1,
      numPointShadows: -1,
      numSpotShadows: -1,
      numSpotMaps: -1,
      numLightProbes: -1,
    },
    ambient: [0, 0, 0],
    probe: Array.from({ length: 9 }, () => ({ x: 0, y: 0, z: 0 })),
    directional: [],
    directionalShadow: [],
    directionalShadowMap: [],
    directionalShadowMatrix: [],
    spot: [],
    spotLightMap: [],
    spotShadow: [],
    spotShadowMap: [],
    spotLightMatrix: [],
    rectArea: [],
    rectAreaLTC1: null,
    rectAreaLTC2: null,
    point: [],
    pointShadow: [],
    pointShadowMap: [],
    pointShadowMatrix: [],
    hemi: [],
    numSpotLightShadowsWithMaps: 0,
    numLightProbes: 0,
  }

  setup(_lights: unknown[] = []): void {}

  setupView(_lights: unknown[] = [], _camera?: unknown): void {}
}

export class RendererRenderState {
  readonly state = {
    lightsArray: [] as unknown[],
    shadowsArray: [] as unknown[],
    camera: null as unknown,
    lights: new RendererRenderLightsState(),
    transmissionRenderTarget: {} as Record<PropertyKey, unknown>,
  }

  init(camera: unknown): void {
    this.state.camera = camera
    this.state.lightsArray.length = 0
    this.state.shadowsArray.length = 0
  }

  pushLight(light: unknown): void {
    this.state.lightsArray.push(light)
  }

  pushShadow(shadowLight: unknown): void {
    this.state.shadowsArray.push(shadowLight)
  }

  setupLights(): void {
    this.state.lights.setup(this.state.lightsArray)
  }

  setupLightsView(camera: unknown): void {
    this.state.lights.setupView(this.state.lightsArray, camera)
  }
}

export class RendererRenderStatesState {
  private states = new WeakMap<object, RendererRenderState[]>()

  get(scene: object, renderCallDepth = 0): RendererRenderState {
    assertWeakMapKey(scene, 'Renderer.renderStates.get scene')
    if (!Number.isInteger(renderCallDepth) || renderCallDepth < 0) {
      throw new TypeError(`Renderer.renderStates.get renderCallDepth must be a non-negative integer; received ${String(renderCallDepth)}.`)
    }
    let stateArray = this.states.get(scene)
    if (stateArray === undefined) {
      stateArray = []
      this.states.set(scene, stateArray)
    }
    let renderState = stateArray[renderCallDepth]
    if (renderState === undefined) {
      renderState = new RendererRenderState()
      stateArray[renderCallDepth] = renderState
    }
    return renderState
  }

  dispose(): void {
    this.states = new WeakMap()
  }
}

export function collectCompileMaterials(scene: ThreeSceneRootLike): Set<ThreeMaterialLike> {
  const materials = new Set<ThreeMaterialLike>()
  collectObjectCompileMaterials(scene, materials, 'Renderer.compile scene')
  return materials
}

export function collectObjectCompileMaterials(
  object: ThreeObject3DLike,
  materials: Set<ThreeMaterialLike>,
  label: string,
): void {
  if (isCompileRenderableObject(object) && object.material != null) {
    if (Array.isArray(object.material)) {
      for (let i = 0; i < object.material.length; i += 1) {
        addCompileMaterial(object.material[i], materials, `${label}.material[${i}]`)
      }
    } else {
      addCompileMaterial(object.material, materials, `${label}.material`)
    }
  }

  const children = object.children ?? []
  for (let i = 0; i < children.length; i += 1) {
    collectObjectCompileMaterials(children[i], materials, `${label}.children[${i}]`)
  }
}

export function isCompileRenderableObject(object: ThreeObject3DLike): boolean {
  return object.isMesh === true
    || object.isPoints === true
    || object.isLine === true
    || object.isLineSegments === true
    || object.isLineLoop === true
    || object.isSprite === true
}

export function addCompileMaterial(material: unknown, materials: Set<ThreeMaterialLike>, label: string): void {
  if (material === null || typeof material !== 'object' || Array.isArray(material)) {
    throw new TypeError(`${label} must be a material-like object.`)
  }
  materials.add(material as ThreeMaterialLike)
}
