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
import { ACESFilmicToneMapping, NoToneMapping, PCFShadowMap, native } from './index.part-001'
import { InternalRenderOptions, objectSortId, renderArrayCameraAuxiliaryTargetAttachments, renderCubeCamera, renderModeFragment, renderRegularCameraAuxiliaryTargetAttachments, renderTargetDepthBuffer } from './index.part-012'
import { normalizeOptionalPixelRect, renderArrayCamera } from './index.part-013'
import { depthReadbackMesh, effectiveScissor, effectiveScissorLabel, effectiveViewport, effectiveViewportLabel, fogToNative, pixelRectToArray, postProcessingToNative } from './index.part-014'
import { assertRenderOptionsLike, backgroundRotationToNative, cameraClipDistances, environmentRotationToNative, optionalNonNegativeFiniteNumber, optionalNormalizedFiniteNumber, validateUnsupportedRenderOptions } from './index.part-015'
import { assertRenderTargetLike } from './index.part-016'
import { assertNonCubeCameraRenderTargetTextures, resolveSampleCount, writeRenderTarget } from './index.part-018'
import { isArrayCamera, isCubeCamera, validateThreeCamera, validateThreeSceneRoot, validateTopLevelRenderCamera } from './index.part-021'
export function render(scene: ThreeSceneRootLike, camera: ThreeRenderCameraLike, options: RenderOptions = {}): Buffer {
  validateThreeSceneRoot(scene)
  validateTopLevelRenderCamera(camera)
  assertRenderOptionsLike(options, 'options')
  if (isCubeCamera(camera)) {
    const { buffer } = renderCubeCamera(scene, camera, options, native.renderNative)
    return buffer
  }

  if (options.target) assertNonCubeCameraRenderTargetTextures(options.target)

  if (isArrayCamera(camera)) {
    const { buffer, width, height, objectIdEntries, depthData } = renderArrayCamera(scene, camera, options, native.renderNative)
    if (options.target) {
      const auxiliary = renderArrayCameraAuxiliaryTargetAttachments(
        scene,
        camera,
        options.target,
        options,
        buffer,
        objectIdEntries,
        native.renderNative,
      )
      writeRenderTarget(
        options.target,
        buffer,
        width,
        height,
        auxiliary.objectIdEntries,
        depthData,
        auxiliary.attachments,
      )
    }
    return buffer
  }

  const { nativeScene, nativeCamera, objectIdEntries } = toNativeInput(scene, camera, options)
  const buffer = native.renderNative(nativeScene, nativeCamera)
  if (options.target) {
    const depthData = renderTargetDepthBuffer(options.target, nativeScene, nativeCamera, native.renderNative)
    const auxiliary = renderRegularCameraAuxiliaryTargetAttachments(
      scene,
      camera,
      options.target,
      options,
      buffer,
      objectIdEntries,
      native.renderNative,
    )
    writeRenderTarget(
      options.target,
      buffer,
      nativeScene.width!,
      nativeScene.height!,
      auxiliary.objectIdEntries,
      depthData,
      auxiliary.attachments,
    )
  }
  return buffer
}

export function renderToTarget(
  scene: ThreeSceneRootLike,
  camera: ThreeRenderCameraLike,
  target: RenderTargetLike = {},
  options: RenderOptions = {},
): RenderTargetLike {
  validateThreeSceneRoot(scene)
  validateTopLevelRenderCamera(camera)
  assertRenderTargetLike(target, 'target')
  assertRenderOptionsLike(options, 'options')
  const targetOptions: RenderOptions = { ...options, target, format: options.format ?? 'rgba' }
  if (isCubeCamera(camera)) {
    const { target: cubeTarget } = renderCubeCamera(scene, camera, targetOptions, native.renderNative)
    return cubeTarget
  }

  assertNonCubeCameraRenderTargetTextures(target)

  if (isArrayCamera(camera)) {
    const { buffer, width, height, objectIdEntries, depthData } = renderArrayCamera(scene, camera, targetOptions, native.renderNative)
    const auxiliary = renderArrayCameraAuxiliaryTargetAttachments(
      scene,
      camera,
      target,
      targetOptions,
      buffer,
      objectIdEntries,
      native.renderNative,
    )
    return writeRenderTarget(
      target,
      buffer,
      width,
      height,
      auxiliary.objectIdEntries,
      depthData,
      auxiliary.attachments,
    )
  }

  const { nativeScene, nativeCamera, objectIdEntries } = toNativeInput(scene, camera, targetOptions)
  const buffer = native.renderNative(nativeScene, nativeCamera)
  const depthData = renderTargetDepthBuffer(target, nativeScene, nativeCamera, native.renderNative)
  const auxiliary = renderRegularCameraAuxiliaryTargetAttachments(
    scene,
    camera,
    target,
    targetOptions,
    buffer,
    objectIdEntries,
    native.renderNative,
  )
  return writeRenderTarget(
    target,
    buffer,
    nativeScene.width!,
    nativeScene.height!,
    auxiliary.objectIdEntries,
    depthData,
    auxiliary.attachments,
  )
}

export function toNativeInput(
  scene: ThreeSceneRootLike,
  camera: ThreeCameraLike,
  options: RenderOptions,
  sceneExtractionCache?: SceneExtractionCache,
): { nativeScene: NativeRenderScene; nativeCamera: NativeCamera; objectIdEntries?: RenderObjectIdEntry[] } {
  validateThreeSceneRoot(scene)
  validateThreeCamera(camera)
  validateUnsupportedRenderOptions(options)
  validateObjectChildrenTree(scene)
  const renderMode = normalizedRenderMode(options.renderMode)
  const colorMode = renderMode === 'color'

  if (typeof scene.updateMatrixWorld === 'function') {
    scene.updateMatrixWorld(true)
  }
  if (typeof camera.updateMatrixWorld === 'function') {
    camera.updateMatrixWorld()
  }

  const size = resolveSize(camera, options)
  const overrideMaterial = colorMode ? resolveSceneOverrideMaterial(scene) : undefined
  const environment = colorMode ? resolveEnvironmentMap(scene, options.environmentIntensity, overrideMaterial) : { envMap: null }
  const envMap = environment.envMap
  const hasEnvironmentRotationOverride = options.environmentRotation !== undefined
  const environmentRotation = environment.rotation ?? (
    hasEnvironmentRotationOverride ? options.environmentRotation : scene.environmentRotation
  )
  const environmentRotationLabel = environment.rotation
    ? 'material.envMapRotation'
    : hasEnvironmentRotationOverride ? 'options.environmentRotation' : 'scene.environmentRotation'
  const environmentMapRotation = colorMode
    ? environmentRotationToNative(environmentRotation, envMap, environmentRotationLabel)
    : undefined
  const hasBackgroundOverride = options.background !== undefined
  const optionBackgroundTexture = colorMode && hasBackgroundOverride
    ? extractBackgroundTexture(options.background, 'options.background')
    : null
  const backgroundTexture = colorMode
    ? optionBackgroundTexture ?? (hasBackgroundOverride ? null : extractBackgroundTexture(scene.background, 'scene.background'))
    : null
  const hasBackgroundRotationOverride = options.backgroundRotation !== undefined
  const backgroundRotation = hasBackgroundOverride
    ? options.backgroundRotation
    : hasBackgroundRotationOverride ? options.backgroundRotation : scene.backgroundRotation
  const backgroundTextureRotation = colorMode
    ? backgroundRotationToNative(
      backgroundRotation,
      backgroundTexture,
      hasBackgroundOverride || hasBackgroundRotationOverride ? 'options.backgroundRotation' : 'scene.backgroundRotation',
    )
    : undefined
  const backgroundTextureBlurriness = colorMode && backgroundTexture
    ? optionalNormalizedFiniteNumber(
      hasBackgroundOverride ? options.backgroundBlurriness : options.backgroundBlurriness ?? scene.backgroundBlurriness,
      hasBackgroundOverride || options.backgroundBlurriness !== undefined ? 'options.backgroundBlurriness' : 'scene.backgroundBlurriness',
    )
    : undefined
  const backgroundIntensity = colorMode
    ? optionalNonNegativeFiniteNumber(
      hasBackgroundOverride ? options.backgroundIntensity : options.backgroundIntensity ?? scene.backgroundIntensity,
      hasBackgroundOverride || options.backgroundIntensity !== undefined ? 'options.backgroundIntensity' : 'scene.backgroundIntensity',
    )
    : undefined
  const clippingPlanes = extractClippingPlanes(
    options.clippingPlanes,
    (options as InternalRenderOptions).__headlessThreeClippingPlanesLabel ?? 'options.clippingPlanes',
  )
  const rendererShadowMapEnabled = (options as InternalRenderOptions).__headlessThreeRendererShadowMapEnabled !== false
  const rendererShadowMapType = (options as InternalRenderOptions).__headlessThreeRendererShadowMapType ?? PCFShadowMap
  const rendererToneMapping = (options as InternalRenderOptions).__headlessThreeRendererToneMapping ?? ACESFilmicToneMapping
  const toneMappingExposure = (options as InternalRenderOptions).__headlessThreeRendererToneMappingExposure ?? 1
  const rendererCallbackContext = colorMode && (options as InternalRenderOptions).__headlessThreeRenderer !== undefined
    ? { renderer: (options as InternalRenderOptions).__headlessThreeRenderer, scene }
    : undefined
  const extractedLights: NativeSceneLight[] | undefined = colorMode ? extractLights(scene, camera) : []
  const lights = rendererShadowMapEnabled ? extractedLights : nativeLightsWithoutShadows(extractedLights)
  const shadowMaterialMode = colorMode ? shadowMaterialModeForLights(lights) : undefined
  const materialContext: MaterialExtractionContext = {
    ...(environment.materialContext ?? {}),
    textureCache: sceneExtractionCache?.texturePayloads,
    materialColorCache: sceneExtractionCache?.materialColors,
    textureStateCache: sceneExtractionCache?.textureStates,
    materialRenderStateCache: sceneExtractionCache?.materialRenderStates,
    materialScalarFeatureCache: sceneExtractionCache?.materialScalarFeatures,
  }
  const flattenedMeshes = flattenScene(
    scene,
    camera,
    size.height,
    clippingPlanes,
    options.localClippingEnabled !== false,
    shadowMaterialMode,
    materialContext,
    {
      sortObjects: options.sortObjects,
      opaqueSort: options.opaqueSort,
      transparentSort: options.transparentSort,
      opaque: (options as InternalRenderOptions).__headlessThreeRendererOpaque,
      transparent: (options as InternalRenderOptions).__headlessThreeRendererTransparent,
    },
    overrideMaterial,
    sceneExtractionCache,
    rendererCallbackContext,
  )
  const objectIdEntries = renderMode === 'object-id' ? objectIdEntriesForMeshes(flattenedMeshes) : undefined
  const meshes = renderMode === 'depth'
    ? flattenedMeshes.map(depthReadbackMesh)
    : applyRendererToneMapping(applyRenderMode(flattenedMeshes, renderMode), rendererToneMapping)
  const viewport = normalizeOptionalPixelRect(
    effectiveViewport(options),
    size.width,
    size.height,
    effectiveViewportLabel(options),
  )
  const scissor = normalizeOptionalPixelRect(
    effectiveScissor(options),
    size.width,
    size.height,
    effectiveScissorLabel(options),
  )
  const nativeScene: NativeRenderScene = {
    width: size.width,
    height: size.height,
    background: colorMode
      ? resolveBackground(
        scene,
        options,
        backgroundTexture != null,
        (options as InternalRenderOptions).__headlessThreeRendererClearColor,
        options.outputColorSpace,
      )
      : [0, 0, 0, 1],
    backgroundIntensity,
    viewport: pixelRectToArray(viewport),
    scissor: pixelRectToArray(scissor),
    backgroundTexture: backgroundTexture?.data,
    backgroundTextureWidth: backgroundTexture?.width,
    backgroundTextureHeight: backgroundTexture?.height,
    backgroundTextureWrapS: backgroundTexture?.wrapS,
    backgroundTextureWrapT: backgroundTexture?.wrapT,
    backgroundTextureMagFilter: backgroundTexture?.magFilter,
    backgroundTextureMinFilter: backgroundTexture?.minFilter,
    backgroundTextureAnisotropy: backgroundTexture?.anisotropy,
    backgroundTextureTransform: backgroundTexture?.transform,
    backgroundTextureColorSpace: backgroundTexture?.colorSpace,
    backgroundTextureMapping: backgroundTexture?.mapping,
    backgroundTextureRotation,
    backgroundTextureBlurriness,
    format: options.format ?? (options.target ? 'rgba' : 'png'),
    outputColorSpace: renderMode === 'depth' ? 'srgb-linear' : options.outputColorSpace,
    toneMapping: renderMode === 'depth' ? undefined : rendererToneMapping,
    toneMappingExposure: renderMode === 'depth' ? undefined : toneMappingExposure,
    transmissionResolutionScale: (options as InternalRenderOptions).__headlessThreeRendererTransmissionResolutionScale,
    sampleCount: renderMode === 'depth' ? 1 : resolveSampleCount(options),
    shadowMapType: rendererShadowMapType,
    meshes,
    lights,
    ambientLight: colorMode ? extractAmbientLight(scene, camera) ?? undefined : undefined,
    ambientIntensity: colorMode ? extractAmbientIntensity(scene, camera) ?? undefined : undefined,
    lightProbe: colorMode ? extractLightProbe(scene, camera) ?? undefined : undefined,
    environmentMap: envMap?.data,
    environmentMapWidth: envMap?.width,
    environmentMapHeight: envMap?.height,
    environmentMapIntensity: envMap?.intensity,
    environmentMapColorSpace: envMap?.colorSpace,
    environmentMapRotation,
    ...(colorMode ? fogToNative(scene.fog) : {}),
    ...(colorMode ? postProcessingToNative(options.postProcessing) : {}),
  }
  const clipDistances = cameraClipDistances(camera)
  const nativeCamera: NativeCamera = {
    width: size.width,
    height: size.height,
    near: clipDistances.near,
    far: clipDistances.far,
    viewProjection: cameraViewProjection(camera),
    viewMatrix: cameraViewMatrix(camera),
    cameraPosition: cameraWorldPosition(camera),
  }

  return { nativeScene, nativeCamera, objectIdEntries }
}

export function normalizedRenderMode(mode: RenderOptions['renderMode']): RenderMode {
  if (mode == null) return 'color'
  return checkedRenderMode(mode, 'options.renderMode')
}

export function checkedRenderMode(mode: unknown, label: string): RenderMode {
  if (mode === 'color' || mode === 'mask' || mode === 'object-id' || mode === 'normal' || mode === 'depth') return mode
  throw new TypeError(
    `${label} must be "color", "mask", "object-id", "normal", or "depth"; received ${String(mode)}`,
  )
}

export function shadowMaterialModeForLights(lights: NativeSceneLight[] | undefined): ShadowMaterialMode | undefined {
  const shadowLight = lights?.find((light) => light.castShadow === true)
  if (!shadowLight) return undefined
  return shadowLight.lightType === 'point' ? 'distance' : 'depth'
}

export function nativeLightsWithoutShadows(lights: NativeSceneLight[] | undefined): NativeSceneLight[] | undefined {
  if (!lights) return undefined
  return lights.map((light) => {
    const withoutShadow = { ...light }
    delete withoutShadow.castShadow
    return withoutShadow
  })
}

export function applyRenderMode(meshes: NativeSceneMesh[], mode: RenderMode): NativeSceneMesh[] {
  if (mode === 'color') return meshes
  return meshes.map((mesh, index) => renderModeMesh(mesh, mode, index))
}

export function applyRendererToneMapping(meshes: NativeSceneMesh[], toneMapping: number): NativeSceneMesh[] {
  if (toneMapping !== NoToneMapping) return meshes
  return meshes.map((mesh) => (mesh.toneMapped === false ? mesh : { ...mesh, toneMapped: false }))
}

export function renderModeMesh(mesh: NativeSceneMesh, mode: Exclude<RenderMode, 'color'>, index: number): NativeSceneMesh {
  const color = mode === 'mask'
    ? [1, 1, 1, materialAlpha(mesh)] as Color4
    : mode === 'object-id'
      ? objectIdColor(mesh, index)
      : [1, 1, 1, materialAlpha(mesh)] as Color4
  return {
    nativeMeshKey: mesh.nativeMeshKey,
    nativeVertexCount: mesh.nativeVertexCount,
    nativeIndexCount: mesh.nativeIndexCount,
    positions: mesh.positions,
    indices: mesh.indices,
    normals: mesh.normals,
    colors: mesh.colors,
    color,
    transform: mesh.transform,
    uvs: mesh.uvs,
    uvs2: mesh.uvs2,
    texture: mesh.texture,
    textureWidth: mesh.textureWidth,
    textureHeight: mesh.textureHeight,
    textureWrapS: mesh.textureWrapS,
    textureWrapT: mesh.textureWrapT,
    textureMagFilter: mesh.textureMagFilter,
    textureMinFilter: mesh.textureMinFilter,
    textureTransform: mesh.textureTransform,
    textureColorSpace: mesh.textureColorSpace,
    textureUsesUv2: mesh.textureUsesUv2,
    alphaMap: mesh.alphaMap,
    alphaMapWidth: mesh.alphaMapWidth,
    alphaMapHeight: mesh.alphaMapHeight,
    alphaMapWrapS: mesh.alphaMapWrapS,
    alphaMapWrapT: mesh.alphaMapWrapT,
    alphaMapMagFilter: mesh.alphaMapMagFilter,
    alphaMapMinFilter: mesh.alphaMapMinFilter,
    alphaMapTransform: mesh.alphaMapTransform,
    alphaMapColorSpace: mesh.alphaMapColorSpace,
    alphaMapUsesUv2: mesh.alphaMapUsesUv2,
    alphaTest: mesh.alphaTest,
    alphaHash: mesh.alphaHash,
    alphaToCoverage: mesh.alphaToCoverage,
    premultipliedAlpha: mesh.premultipliedAlpha,
    toneMapped: false,
    clippingPlanes: mesh.clippingPlanes,
    clippingUnionCount: mesh.clippingUnionCount,
    blending: 'none',
    depthTest: mesh.depthTest,
    depthFunc: mesh.depthFunc,
    depthWrite: true,
    colorWrite: true,
    polygonOffset: mesh.polygonOffset,
    polygonOffsetFactor: mesh.polygonOffsetFactor,
    polygonOffsetUnits: mesh.polygonOffsetUnits,
    stencilWrite: mesh.stencilWrite,
    stencilWriteMask: mesh.stencilWriteMask,
    stencilFunc: mesh.stencilFunc,
    stencilRef: mesh.stencilRef,
    stencilFuncMask: mesh.stencilFuncMask,
    stencilFail: mesh.stencilFail,
    stencilZFail: mesh.stencilZFail,
    stencilZPass: mesh.stencilZPass,
    transparent: false,
    side: mesh.side,
    shadingModel: 'basic',
    topology: mesh.topology,
    customFragmentShader: renderModeFragment(mode, color),
    castShadow: false,
    receiveShadow: false,
    groupOrder: mesh.groupOrder,
    renderOrder: mesh.renderOrder,
    sortZ: mesh.sortZ,
    sortIndex: mesh.sortIndex,
    materialVariant: mesh.materialVariant,
    materialSortKey: mesh.materialSortKey,
  }
}

export function materialAlpha(mesh: NativeSceneMesh): number {
  const alpha = mesh.color?.[3]
  return typeof alpha === 'number' && Number.isFinite(alpha) ? Math.min(1, Math.max(0, alpha)) : 1
}

export function objectIdColor(mesh: NativeSceneMesh, index: number): Color4 {
  const value = encodedObjectId(mesh, index)
  return [
    ((value >> 16) & 0xff) / 255,
    ((value >> 8) & 0xff) / 255,
    (value & 0xff) / 255,
    materialAlpha(mesh),
  ]
}

export function objectIdEntriesForMeshes(meshes: NativeSceneMesh[]): RenderObjectIdEntry[] {
  const entries = new Map<number, RenderObjectIdEntry>()
  meshes.forEach((mesh, index) => {
    const id = objectSortId(mesh, index)
    const encodedId = encodedObjectId(mesh, index)
    if (entries.has(encodedId)) return
    entries.set(encodedId, {
      id,
      encodedId,
      rgb: [
        (encodedId >> 16) & 0xff,
        (encodedId >> 8) & 0xff,
        encodedId & 0xff,
      ],
      hex: `#${encodedId.toString(16).padStart(6, '0')}`,
    })
  })
  return [...entries.values()].sort((a, b) => a.encodedId - b.encodedId)
}

export function encodedObjectId(mesh: NativeSceneMesh, index: number): number {
  const encoded = (objectSortId(mesh, index) + 1) & 0xffffff
  return encoded === 0 ? 1 : encoded
}
