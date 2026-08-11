import type {
  ThreeObject3DLike,
  ThreeBufferAttributeLike,
  ThreeBufferGeometryLike,
  ThreeCameraLike,
  ThreeMaterialLike,
  ThreeSphereLike,
  NativeSceneMesh,
  GeometryGroup,
  Color4,
  PbrProperties,
  RenderSortFunction,
  RenderSortItem,
} from './types'
import { IDENTITY_4X4, matrixElements, clampInteger, clamp01 } from './math'
import {
  attributeComponent,
  attributeCount,
  getAttribute,
  readVec3Attribute,
  readVec2Attribute,
  readColorAttribute,
  readIndexAttribute,
  geometryAttributes,
} from './attributes'
import {
  materialForGroup,
  materialColor,
  extractPbrProperties,
  extractTextureData,
  materialShadowSide,
  textureUvChannel,
  assertMaterialLike,
  type MaterialExtractionContext,
  type TextureExtractionCache,
  type MaterialColorExtractionCache,
  type TextureStateExtractionCache,
  type MaterialRenderStateExtractionCache,
  type MaterialScalarFeatureExtractionCache,
} from './materials'
import { applyCpuSkinning } from './skinning'
import { applyMorphTargets } from './morphs'
import { objectLayersMatchCamera } from './layers'
import { objectChildren } from './objects'
import {
  MAX_CLIPPING_PLANES,
  type NativeClippingPlane,
  extractClippingPlanes,
  flattenClippingPlanes,
} from './clipping'
import { ClippingContext, FlattenedMesh, SceneExtractionCache, ShadowMaterialMode, SpriteBillboardExpansion, SpriteBillboardSignature } from './scene.part-001'
import { RenderCallbackContext } from './scene.part-002'
import { customShadowMaterialForMode, materialForObjectGroup, meshGeometryExtraction } from './scene.part-003'
import { appendShadowOnlyBillboardMesh, expandPointBillboardUvStream, matrixValues, optionalObjectBoolean, pointBillboardExpansion, sameNumberArray, sameOptionalNumberArray, shadowMaterialWithSourceShadowState } from './scene.part-005'
import { optionalSceneBoolean } from './scene.part-006'
import { clipShadowsForMaterial, clippingState, pushMesh, sortInfoForObject } from './scene.part-007'
import { cameraBillboardAxes, columnLength3, finiteMaterialOrObjectNumber, positiveMaterialOrObjectNumber, spriteOutsideFrustum, viewSpaceZ } from './scene.part-008'
import { assertSupportedCustomFragmentInstancedAttributes } from './scene.part-009'
import { textureUvStreamsForMapAlphaMaterial } from './scene.part-010'
export function invokeObjectRenderCallback(
  callback: unknown,
  name: 'onBeforeRender' | 'onAfterRender',
  context: RenderCallbackContext | undefined,
  object: ThreeObject3DLike,
  camera: ThreeCameraLike | undefined,
  geometry: ThreeBufferGeometryLike | undefined,
  material: ThreeMaterialLike | undefined,
  group?: GeometryGroup,
): void {
  if (callback == null) return
  if (typeof callback !== 'function') {
    throw new TypeError(`THREE.Object3D.${name} must be a function when provided.`)
  }
  if (isInternalBatchedMeshRenderCallback(object, callback, name)) return
  if (!context) return
  callback.call(object, context.renderer, context.scene, camera, geometry, material, group)
}

export function isInternalBatchedMeshRenderCallback(
  object: ThreeObject3DLike,
  callback: Function,
  name: 'onBeforeRender' | 'onAfterRender',
): boolean {
  if (name !== 'onBeforeRender' || object.isBatchedMesh !== true) return false
  if (Object.prototype.hasOwnProperty.call(object, name)) return false

  let prototype = Object.getPrototypeOf(object)
  while (prototype && prototype !== Object.prototype) {
    if (prototype.constructor?.name === 'BatchedMesh' && typeof prototype[name] === 'function') {
      return callback === prototype[name]
    }
    prototype = Object.getPrototypeOf(prototype)
  }
  return false
}

export function shadowOnlyMainPassState(): Pick<
  NativeSceneMesh,
  'blending' | 'colorWrite' | 'depthTest' | 'depthWrite' | 'stencilWrite' | 'transparent'
> {
  return {
    blending: 'none',
    colorWrite: false,
    depthTest: false,
    depthWrite: false,
    stencilWrite: false,
    transparent: false,
  }
}

export function appendSprite(
  object: ThreeObject3DLike,
  camera: ThreeCameraLike | undefined,
  meshes: FlattenedMesh[],
  groupOrder: number,
  clippingContext: ClippingContext,
  localClippingEnabled: boolean,
  shadowMaterialMode: ShadowMaterialMode | undefined,
  materialContext: MaterialExtractionContext,
  overrideMaterial: ThreeMaterialLike | undefined,
  cache: SceneExtractionCache | undefined,
  callbackContext: RenderCallbackContext | undefined,
): void {
  const objectCastsShadow = optionalObjectBoolean(object.castShadow, 'object.castShadow') === true
  optionalObjectBoolean(object.receiveShadow, 'object.receiveShadow')

  const material = materialForObjectGroup(object, 0, overrideMaterial)
  if (material?.visible === false) return

  validateSpriteScale(object)
  const matrix = matrixElements(object.matrixWorld!, 'sprite.matrixWorld')
  const center: [number, number] = [
    finiteMaterialOrObjectNumber(object.center?.x, 'Sprite.center.x', 0.5),
    finiteMaterialOrObjectNumber(object.center?.y, 'Sprite.center.y', 0.5),
  ]
  if (spriteOutsideFrustum(object, camera, matrix, center)) return

  invokeObjectRenderCallback(object.onBeforeRender, 'onBeforeRender', callbackContext, object, camera, object.geometry, material)

  const worldPosition = [matrix[12], matrix[13], matrix[14]]
  let scaleX = columnLength3(matrix, 0)
  let scaleY = columnLength3(matrix, 4)

  const sizeAttenuation = optionalSceneBoolean(material?.sizeAttenuation, 'material.sizeAttenuation')
  if (sizeAttenuation === false && camera?.isPerspectiveCamera === true) {
    const viewZ = viewSpaceZ(worldPosition, camera)
    if (Number.isFinite(viewZ)) {
      scaleX *= -viewZ
      scaleY *= -viewZ
    }
  }

  if (scaleX <= 0 || scaleY <= 0) return

  const axes = cameraBillboardAxes(camera)
  const rotation = finiteMaterialOrObjectNumber(material?.rotation, 'material.rotation', 0)
  const billboard = spriteBillboardExpansion(
    object,
    matrix,
    worldPosition,
    center,
    scaleX,
    scaleY,
    rotation,
    sizeAttenuation,
    axes,
    camera,
    cache,
  )
  const positions = billboard.positions
  const indices = billboard.indices
  const uvs = billboard.uvs

  const textureInfo = extractTextureData(material, materialContext)
  const sortInfo = sortInfoForObject(object, material, camera, meshes.length, groupOrder)
  const pbrProps = extractPbrProperties(material, materialContext)
  pbrProps.alphaMapUsesUv2 = false
  const clipping = clippingState(clippingContext, material, localClippingEnabled)
  const customShadowMaterial = customShadowMaterialForMode(object, shadowMaterialMode)
  const usesCustomShadowMaterial = objectCastsShadow && customShadowMaterial != null

  pushMesh(meshes, {
    positions,
    indices,
    uvs,
    color: materialColor(material, materialContext),
    texture: textureInfo?.data,
    textureWidth: textureInfo?.width ?? undefined,
    textureHeight: textureInfo?.height ?? undefined,
    textureWrapS: textureInfo?.wrapS,
    textureWrapT: textureInfo?.wrapT,
    textureMagFilter: textureInfo?.magFilter,
    textureMinFilter: textureInfo?.minFilter,
    textureAnisotropy: textureInfo?.anisotropy,
    textureTransform: textureInfo?.transform,
    textureColorSpace: textureInfo?.colorSpace,
    textureUsesUv2: false,
    transform: IDENTITY_4X4.slice(),
    transparent: material?.transparent !== false,
    castShadow: objectCastsShadow && !usesCustomShadowMaterial ? true : undefined,
    receiveShadow: undefined,
    clipShadows: clipShadowsForMaterial(material, clippingContext),
    ...clipping,
    ...sortInfo.keys,
    ...pbrProps,
  }, sortInfo.item)

  if (usesCustomShadowMaterial && customShadowMaterial?.visible !== false) {
    appendShadowOnlyBillboardMesh(
      object,
      camera,
      meshes,
      groupOrder,
      clippingContext,
      localClippingEnabled,
      customShadowMaterial,
      material,
      materialContext,
      positions,
      indices,
      uvs,
    )
  }

  invokeObjectRenderCallback(object.onAfterRender, 'onAfterRender', callbackContext, object, camera, object.geometry, material)
}

export function spriteBillboardExpansion(
  object: ThreeObject3DLike,
  matrix: number[],
  worldPosition: number[],
  center: [number, number],
  scaleX: number,
  scaleY: number,
  rotation: number,
  sizeAttenuation: boolean | undefined,
  axes: { right: [number, number, number]; up: [number, number, number] },
  camera: ThreeCameraLike | undefined,
  cache: SceneExtractionCache | undefined,
): SpriteBillboardExpansion {
  const signature = spriteBillboardSignature(
    matrix,
    center,
    scaleX,
    scaleY,
    rotation,
    sizeAttenuation,
    axes,
    camera,
  )
  if (cache) {
    const cached = cache.spriteBillboards.get(object)
    if (cached && sameSpriteBillboardSignature(cached.signature, signature)) {
      return cached.expansion
    }
  }

  const expansion = readSpriteBillboardExpansion(worldPosition, center, scaleX, scaleY, rotation, axes)
  if (cache) {
    cache.spriteBillboards.set(object, { signature, expansion })
  }
  return expansion
}

export function readSpriteBillboardExpansion(
  worldPosition: number[],
  center: [number, number],
  scaleX: number,
  scaleY: number,
  rotation: number,
  axes: { right: [number, number, number]; up: [number, number, number] },
): SpriteBillboardExpansion {
  const cos = Math.cos(rotation)
  const sin = Math.sin(rotation)
  const corners = [
    [-0.5, -0.5, 0, 0],
    [0.5, -0.5, 1, 0],
    [0.5, 0.5, 1, 1],
    [-0.5, 0.5, 0, 1],
  ]
  const positions: number[] = []
  const uvs: number[] = []
  for (const [x, y, u, v] of corners) {
    const alignedX = (x - (center[0] - 0.5)) * scaleX
    const alignedY = (y - (center[1] - 0.5)) * scaleY
    const rotatedX = cos * alignedX - sin * alignedY
    const rotatedY = sin * alignedX + cos * alignedY
    positions.push(
      worldPosition[0] + axes.right[0] * rotatedX + axes.up[0] * rotatedY,
      worldPosition[1] + axes.right[1] * rotatedX + axes.up[1] * rotatedY,
      worldPosition[2] + axes.right[2] * rotatedX + axes.up[2] * rotatedY,
    )
    uvs.push(u, v)
  }
  return {
    positions,
    indices: [0, 1, 2, 0, 2, 3],
    uvs,
  }
}

export function spriteBillboardSignature(
  matrix: number[],
  center: [number, number],
  scaleX: number,
  scaleY: number,
  rotation: number,
  sizeAttenuation: boolean | undefined,
  axes: { right: [number, number, number]; up: [number, number, number] },
  camera: ThreeCameraLike | undefined,
): SpriteBillboardSignature {
  return {
    matrix: matrix.slice(0, 16),
    center: center.slice() as [number, number],
    scaleX,
    scaleY,
    rotation,
    sizeAttenuation,
    cameraRight: axes.right.slice() as [number, number, number],
    cameraUp: axes.up.slice() as [number, number, number],
    cameraView: matrixValues(camera?.matrixWorldInverse?.elements),
    cameraIsPerspective: camera?.isPerspectiveCamera,
  }
}

export function sameSpriteBillboardSignature(a: SpriteBillboardSignature, b: SpriteBillboardSignature): boolean {
  return sameNumberArray(a.matrix, b.matrix)
    && sameNumberArray(a.center, b.center)
    && a.scaleX === b.scaleX
    && a.scaleY === b.scaleY
    && a.rotation === b.rotation
    && a.sizeAttenuation === b.sizeAttenuation
    && sameNumberArray(a.cameraRight, b.cameraRight)
    && sameNumberArray(a.cameraUp, b.cameraUp)
    && sameOptionalNumberArray(a.cameraView, b.cameraView)
    && a.cameraIsPerspective === b.cameraIsPerspective
}

export function validateSpriteScale(object: ThreeObject3DLike): void {
  finiteMaterialOrObjectNumber(object.scale?.x, 'Sprite.scale.x', 1)
  finiteMaterialOrObjectNumber(object.scale?.y, 'Sprite.scale.y', 1)
  finiteMaterialOrObjectNumber(object.scale?.z, 'Sprite.scale.z', 1)
}

export function appendPoints(
  object: ThreeObject3DLike,
  camera: ThreeCameraLike | undefined,
  meshes: FlattenedMesh[],
  groupOrder: number,
  viewportHeight: number,
  clippingContext: ClippingContext,
  localClippingEnabled: boolean,
  shadowMaterialMode: ShadowMaterialMode | undefined,
  materialContext: MaterialExtractionContext,
  overrideMaterial: ThreeMaterialLike | undefined,
  cache: SceneExtractionCache | undefined,
  callbackContext: RenderCallbackContext | undefined,
): void {
  const objectCastsShadow = optionalObjectBoolean(object.castShadow, 'object.castShadow') === true
  optionalObjectBoolean(object.receiveShadow, 'object.receiveShadow')

  const geometry = object.geometry!
  const geometryExtraction = meshGeometryExtraction(geometry, cache)
  if (!geometryExtraction) return

  const {
    position,
    positions,
    vertexColors,
    index,
    sourceIndex,
    groups,
    instancedGeometryCount,
    instancedPositionOffset,
    instancedPositionScale,
  } = geometryExtraction
  const pointUvChannels = geometryExtraction.uvChannels
  const transform = matrixElements(object.matrixWorld!, 'points.matrixWorld')
  const axes = cameraBillboardAxes(camera)

  for (const group of groups) {
    const material = materialForObjectGroup(object, group.materialIndex, overrideMaterial)
    if (material?.visible === false) continue
    invokeObjectRenderCallback(object.onBeforeRender, 'onBeforeRender', callbackContext, object, camera, geometry, material, group)
    const pointUvStreams = material?.map || material?.alphaMap
      ? textureUvStreamsForMapAlphaMaterial(pointUvChannels, {
        map: material.map,
        alphaMap: material.alphaMap,
      })
      : null

    const baseColor = materialColor(material, materialContext)
    const useVertexColors = vertexColors && material?.vertexColors !== false
    const pointSize = positiveMaterialOrObjectNumber(material?.size, 'material.size', 1)
    const sizeAttenuation = optionalSceneBoolean(material?.sizeAttenuation, 'material.sizeAttenuation')
    const billboard = pointBillboardExpansion(
      object,
      group,
      position,
      positions,
      index ?? sourceIndex,
      instancedGeometryCount,
      instancedPositionOffset,
      instancedPositionScale,
      transform,
      axes,
      camera,
      viewportHeight,
      pointSize,
      sizeAttenuation,
      pointUvStreams,
      vertexColors,
      useVertexColors ? baseColor : undefined,
      cache,
    )

    if (billboard.positions.length === 0) continue
    const outputPositions = billboard.positions
    const outputUvs = billboard.uvs
    const outputUvs2 = billboard.uvs2
    const outputColors = billboard.colors
    const outputIndices = billboard.indices
    const outputPointRefs = billboard.pointRefs

    const textureInfo = extractTextureData(material, materialContext)
    const sortInfo = sortInfoForObject(object, material, camera, meshes.length, groupOrder, undefined, geometry, group)
    const pbrProps = extractPbrProperties(material, materialContext)
    assertSupportedCustomFragmentInstancedAttributes(geometry, pbrProps)
    pbrProps.alphaMapUsesUv2 = pointUvStreams?.alphaMapUsesUv2 ?? false
    const clipping = clippingState(clippingContext, material, localClippingEnabled)
    const customShadowMaterial = customShadowMaterialForMode(object, shadowMaterialMode)
    const usesCustomShadowMaterial = objectCastsShadow && customShadowMaterial != null
    const effectiveCustomShadowMaterial = usesCustomShadowMaterial
      ? shadowMaterialWithSourceShadowState(customShadowMaterial, material)
      : null
    const pointShadowUvStreams = effectiveCustomShadowMaterial && (
      effectiveCustomShadowMaterial.map ||
      effectiveCustomShadowMaterial.alphaMap
    )
      ? textureUvStreamsForMapAlphaMaterial(pointUvChannels, {
        map: effectiveCustomShadowMaterial.map,
        alphaMap: effectiveCustomShadowMaterial.alphaMap,
      })
      : null

    pushMesh(meshes, {
      positions: outputPositions,
      indices: outputIndices,
      uvs: outputUvs,
      uvs2: outputUvs2,
      color: baseColor,
      colors: outputColors,
      texture: textureInfo?.data,
      textureWidth: textureInfo?.width ?? undefined,
      textureHeight: textureInfo?.height ?? undefined,
      textureWrapS: textureInfo?.wrapS,
      textureWrapT: textureInfo?.wrapT,
      textureMagFilter: textureInfo?.magFilter,
      textureMinFilter: textureInfo?.minFilter,
      textureAnisotropy: textureInfo?.anisotropy,
      textureTransform: textureInfo?.transform,
      textureColorSpace: textureInfo?.colorSpace,
      textureUsesUv2: pointUvStreams?.textureUsesUv2 ?? false,
      transform: IDENTITY_4X4.slice(),
      transparent: material?.transparent === true || (material?.opacity != null && material.opacity < 1),
      topology: 'triangles',
      castShadow: objectCastsShadow && !usesCustomShadowMaterial ? true : undefined,
      receiveShadow: false,
      clipShadows: clipShadowsForMaterial(material, clippingContext),
      ...clipping,
      ...sortInfo.keys,
      ...pbrProps,
      shadingModel: 'basic',
    }, sortInfo.item)

    if (usesCustomShadowMaterial && customShadowMaterial?.visible !== false) {
      appendShadowOnlyBillboardMesh(
        object,
        camera,
        meshes,
        groupOrder,
        clippingContext,
        localClippingEnabled,
        customShadowMaterial,
        material,
        materialContext,
        outputPositions,
        outputIndices,
        pointShadowUvStreams?.uvs ? expandPointBillboardUvStream(pointShadowUvStreams.uvs, outputPointRefs) : outputUvs,
        pointShadowUvStreams?.uvs2 ? expandPointBillboardUvStream(pointShadowUvStreams.uvs2, outputPointRefs) : undefined,
        pointShadowUvStreams?.textureUsesUv2 ?? false,
        pointShadowUvStreams?.alphaMapUsesUv2 ?? false,
      )
    }
    invokeObjectRenderCallback(object.onAfterRender, 'onAfterRender', callbackContext, object, camera, geometry, material, group)
  }
}
