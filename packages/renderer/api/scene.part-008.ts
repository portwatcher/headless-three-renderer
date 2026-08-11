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
import { NativeRenderOrderLimit } from './scene.part-001'
import { MAX_POINT_SPRITE_SIZE } from './scene.part-002'
import { optionalObjectBoolean } from './scene.part-005'
import { batchedGeometryBoundingSphere, batchedGeometryRange } from './scene.part-012'
import { batchedInstanceMatrix, batchedNonNegativeInteger, batchedOptionalBoolean, cameraFrustumIntersectsSphere, sphereLike } from './scene.part-013'
import { readMat4Attribute } from './scene.part-014'
export function validateInstancedMeshMatrices(object: ThreeObject3DLike): void {
  const instanceMatrix = object.instanceMatrix
  if (!instanceMatrix || instanceMatrix.count == null) return
  const instanceMatrixCount = attributeCount(instanceMatrix, 'InstancedMesh.instanceMatrix')
  const count = clampInteger(
    integerCountOrDefault(object.count, 'InstancedMesh.count', instanceMatrixCount),
    0,
    instanceMatrixCount,
  )
  for (let i = 0; i < count; i += 1) {
    readMat4Attribute(instanceMatrix, i)
  }
}

export function renderableMatrixWorldLabel(object: ThreeObject3DLike): string {
  if (object.isBatchedMesh === true) return 'batchedMesh.matrixWorld'
  if (object.isMesh === true) return 'mesh.matrixWorld'
  if (object.isPoints === true) return 'points.matrixWorld'
  return 'object.matrixWorld'
}

export function objectBoundingSphere(object: ThreeObject3DLike): { center: [number, number, number]; radius: number } | null {
  if (object.isBatchedMesh === true) {
    if (object.boundingSphere != null) return sphereLike(object.boundingSphere, 'object.boundingSphere')
    return batchedMeshBoundingSphere(object)
  }

  if (object.boundingSphere !== undefined) {
    const hadBoundingSphere = object.boundingSphere != null
    if (object.boundingSphere == null && typeof object.computeBoundingSphere === 'function') {
      if (object.isInstancedMesh === true) {
        validateInstancedMeshMatrices(object)
      }
      try {
        object.computeBoundingSphere()
      } catch {
        return null
      }
    }
    if (object.boundingSphere == null) return null
    try {
      return object.isInstancedMesh === true
        ? instancedMeshBoundingSphere(object)
        : sphereLike(object.boundingSphere, 'object.boundingSphere')
    } catch (error) {
      if (!hadBoundingSphere) return null
      throw error
    }
  }

  const geometry = object.geometry
  if (!geometry) return null
  const hadBoundingSphere = geometry.boundingSphere != null
  if (geometry.boundingSphere == null && typeof geometry.computeBoundingSphere === 'function') {
    try {
      geometry.computeBoundingSphere()
    } catch {
      return null
    }
  }
  if (geometry.boundingSphere == null) return null
  try {
    return sphereLike(geometry.boundingSphere, 'geometry.boundingSphere')
  } catch (error) {
    if (!hadBoundingSphere) return null
    throw error
  }
}

export function batchedMeshBoundingSphere(
  object: ThreeObject3DLike,
): { center: [number, number, number]; radius: number } | null {
  const geometry = object.geometry
  if (!geometry) return null
  const instanceInfo = object._instanceInfo
  if (!Array.isArray(instanceInfo)) {
    throw new Error(
      'THREE.BatchedMesh instance table is not readable. Use a real THREE.BatchedMesh or expand the batch to ordinary Mesh or InstancedMesh objects before rendering.',
    )
  }

  let bounds: { center: [number, number, number]; radius: number } | null = null
  for (let instanceId = 0; instanceId < instanceInfo.length; instanceId += 1) {
    const info = instanceInfo[instanceId]
    if (!info || typeof info !== 'object') {
      throw new TypeError(`THREE.BatchedMesh._instanceInfo[${instanceId}] must be an object.`)
    }
    if (!batchedOptionalBoolean(info.active, `THREE.BatchedMesh._instanceInfo[${instanceId}].active`, true)) continue
    batchedOptionalBoolean(info.visible, `THREE.BatchedMesh._instanceInfo[${instanceId}].visible`, true)

    const geometryId = batchedNonNegativeInteger(
      info.geometryIndex,
      `THREE.BatchedMesh._instanceInfo[${instanceId}].geometryIndex`,
    )
    const range = batchedGeometryRange(object, geometry, geometryId)
    if (range.count === 0) continue

    const sphere = batchedGeometryBoundingSphere(object, geometry, geometryId, range)
    if (!sphere) continue
    bounds = unionSpheres(bounds, transformSphere(sphere, batchedInstanceMatrix(object, instanceId)))
  }
  return bounds
}

export function transformSphere(
  sphere: { center: [number, number, number]; radius: number },
  transform: ArrayLike<number>,
): { center: [number, number, number]; radius: number } {
  const center = transformPoint(transform, sphere.center)
  const scale = Math.max(columnLength3(transform, 0), columnLength3(transform, 4), columnLength3(transform, 8))
  return { center, radius: sphere.radius * scale }
}

export function unionSpheres(
  a: { center: [number, number, number]; radius: number } | null,
  b: { center: [number, number, number]; radius: number },
): { center: [number, number, number]; radius: number } {
  if (!a) return { center: [...b.center], radius: b.radius }
  const dx = b.center[0] - a.center[0]
  const dy = b.center[1] - a.center[1]
  const dz = b.center[2] - a.center[2]
  const distance = Math.hypot(dx, dy, dz)
  if (distance <= Math.abs(a.radius - b.radius)) {
    return a.radius >= b.radius ? a : { center: [...b.center], radius: b.radius }
  }
  if (distance === 0) return { center: [...a.center], radius: Math.max(a.radius, b.radius) }

  const radius = (distance + a.radius + b.radius) * 0.5
  const centerShift = (radius - a.radius) / distance
  return {
    center: [
      a.center[0] + dx * centerShift,
      a.center[1] + dy * centerShift,
      a.center[2] + dz * centerShift,
    ],
    radius,
  }
}

export function instancedMeshBoundingSphere(object: ThreeObject3DLike): { center: [number, number, number]; radius: number } {
  try {
    return sphereLike(object.boundingSphere!, 'object.boundingSphere')
  } catch (error) {
    validateInstancedMeshMatrices(object)
    throw error
  }
}

export function pointBillboardCullRadius(
  object: ThreeObject3DLike,
  camera: ThreeCameraLike,
  transform: ArrayLike<number>,
  sphere: { center: [number, number, number]; radius: number },
  viewportHeight: number,
  overrideMaterial: ThreeMaterialLike | undefined,
): number {
  const center = transformPoint(transform, sphere.center)
  let radius = 0
  for (const material of pointCullMaterials(object, overrideMaterial)) {
    const pointSize = safePositiveNumber(material?.size, 1)
    const sizeAttenuation = typeof material?.sizeAttenuation === 'boolean'
      ? material.sizeAttenuation
      : undefined
    const worldSize = pointWorldSize(pointSize, center, sizeAttenuation, camera, viewportHeight)
    radius = Math.max(radius, worldSize * Math.SQRT1_2)
  }
  return radius
}

export function pointCullMaterials(
  object: ThreeObject3DLike,
  overrideMaterial: ThreeMaterialLike | undefined,
): Array<ThreeMaterialLike | undefined> {
  if (overrideMaterial !== undefined) return [overrideMaterial]
  if (Array.isArray(object.material)) {
    return object.material.filter((material): material is ThreeMaterialLike => material != null && typeof material === 'object' && !Array.isArray(material))
  }
  return object.material != null && typeof object.material === 'object' && !Array.isArray(object.material)
    ? [object.material]
    : [undefined]
}

export function spriteOutsideFrustum(
  object: ThreeObject3DLike,
  camera: ThreeCameraLike | undefined,
  transform: ArrayLike<number>,
  center: [number, number],
): boolean {
  if (!camera) return false
  const frustumCulled = optionalObjectBoolean(object.frustumCulled, 'object.frustumCulled')
  if (frustumCulled === false) return false

  const offset = Math.hypot(center[0] - 0.5, center[1] - 0.5)
  return transformedSphereOutsideFrustum(camera, transform, {
    center: [0, 0, 0],
    radius: 0.7071067811865476 + offset,
  })
}

export function transformedSphereOutsideFrustum(
  camera: ThreeCameraLike,
  transform: ArrayLike<number>,
  sphere: { center: [number, number, number]; radius: number },
  extraRadius = 0,
): boolean {
  const center = transformPoint(transform, sphere.center)
  const scale = Math.max(columnLength3(transform, 0), columnLength3(transform, 4), columnLength3(transform, 8))
  const radius = sphere.radius * scale + extraRadius
  if (!Number.isFinite(radius) || radius < 0) return false
  return !cameraFrustumIntersectsSphere(camera, center, radius)
}

export function vec3Like(value: { x?: number; y?: number; z?: number } | ArrayLike<number> | undefined): [number, number, number] | null {
  if (!value) return null
  const objectValue = value as { x?: unknown; y?: unknown; z?: unknown }
  const x = typeof objectValue.x === 'number' ? objectValue.x : (value as ArrayLike<unknown>)[0]
  const y = typeof objectValue.y === 'number' ? objectValue.y : (value as ArrayLike<unknown>)[1]
  const z = typeof objectValue.z === 'number' ? objectValue.z : (value as ArrayLike<unknown>)[2]
  return typeof x === 'number' && Number.isFinite(x)
    && typeof y === 'number' && Number.isFinite(y)
    && typeof z === 'number' && Number.isFinite(z)
    ? [x, y, z]
    : null
}

export function requiredVec3Like(value: { x?: number; y?: number; z?: number } | ArrayLike<number> | undefined, label: string): [number, number, number] {
  const vector = vec3Like(value)
  if (vector) return vector
  throw new TypeError(`${label} must be a finite Vector3-like value.`)
}

export function columnLength3(matrix: ArrayLike<number>, start: number): number {
  const x = matrix[start]
  const y = matrix[start + 1]
  const z = matrix[start + 2]
  return Math.hypot(x, y, z)
}

export function cameraBillboardAxes(camera: ThreeCameraLike | undefined): { right: [number, number, number]; up: [number, number, number] } {
  const matrix = camera?.matrixWorld?.elements
  if (!matrix || matrix.length < 16) {
    return { right: [1, 0, 0], up: [0, 1, 0] }
  }
  return {
    right: normalizeVec3([matrix[0], matrix[1], matrix[2]], [1, 0, 0]),
    up: normalizeVec3([matrix[4], matrix[5], matrix[6]], [0, 1, 0]),
  }
}

export function normalizeVec3(value: [number, number, number], fallback: [number, number, number]): [number, number, number] {
  const length = Math.hypot(value[0], value[1], value[2])
  if (!Number.isFinite(length) || length <= 1e-8) return fallback
  return [value[0] / length, value[1] / length, value[2] / length]
}

export function viewSpaceZ(worldPosition: number[], camera: ThreeCameraLike): number {
  const view = camera.matrixWorldInverse?.elements
  if (!view || view.length < 16) return Number.NaN
  return view[2] * worldPosition[0] + view[6] * worldPosition[1] + view[10] * worldPosition[2] + view[14]
}

export function transformPoint(matrix: ArrayLike<number>, point: [number, number, number]): [number, number, number] {
  const x = point[0]
  const y = point[1]
  const z = point[2]
  return [
    matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12],
    matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13],
    matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14],
  ]
}

export function pointWorldSize(
  pointSize: number,
  worldPosition: [number, number, number],
  sizeAttenuation: boolean | undefined,
  camera: ThreeCameraLike | undefined,
  viewportHeight: number,
): number {
  const projectionY = Math.abs(finiteOrDefault(camera?.projectionMatrix?.elements?.[5], 1))
  if (projectionY <= 0) return 0

  if (camera?.isPerspectiveCamera === true && sizeAttenuation !== false) {
    const viewZ = viewSpaceZ(worldPosition, camera)
    const depth = Number.isFinite(viewZ) ? Math.max(0.0001, Math.abs(viewZ)) : 1
    const pixelSize = clampPointSpriteSize(pointSize * Math.max(1, viewportHeight) / (2 * depth))
    return pixelSize * 2 * depth / Math.max(1, viewportHeight) / projectionY
  }

  const cappedPointSize = clampPointSpriteSize(pointSize)
  if (camera?.isPerspectiveCamera !== true) {
    return cappedPointSize * 2 / Math.max(1, viewportHeight) / projectionY
  }

  const viewZ = camera ? viewSpaceZ(worldPosition, camera) : -1
  const depth = Number.isFinite(viewZ) ? Math.max(0.0001, Math.abs(viewZ)) : 1
  return cappedPointSize * 2 * depth / Math.max(1, viewportHeight) / projectionY
}

export function clampPointSpriteSize(pixelSize: number): number {
  return Math.min(pixelSize, MAX_POINT_SPRITE_SIZE)
}

export function finiteOrDefault(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

export function safePositiveNumber(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : fallback
}

export function finiteMaterialOrObjectNumber(value: unknown, label: string, fallback: number): number {
  if (value == null) return fallback
  if (typeof value === 'number' && Number.isFinite(value)) return value
  throw new TypeError(`${label} must be a finite number.`)
}

export function renderOrderNumber(value: unknown, label: string, fallback: number): number {
  if (value == null) return fallback
  if (typeof value === 'number' && (Number.isFinite(value) || value === Infinity || value === -Infinity)) {
    return value
  }
  throw new TypeError(`${label} must be a finite number or +/-Infinity.`)
}

export function nativeRenderOrderKey(value: number): number {
  if (value === Infinity) return NativeRenderOrderLimit
  if (value === -Infinity) return -NativeRenderOrderLimit
  return value
}

export function nonNegativeMaterialOrObjectNumber(value: unknown, label: string, fallback: number): number {
  const number = finiteMaterialOrObjectNumber(value, label, fallback)
  if (number < 0) {
    throw new TypeError(`${label} must be non-negative.`)
  }
  return number
}

export function positiveMaterialOrObjectNumber(value: unknown, label: string, fallback: number): number {
  const number = finiteMaterialOrObjectNumber(value, label, fallback)
  if (number <= 0) {
    throw new TypeError(`${label} must be positive.`)
  }
  return number
}

export function validateLineMaterialCompatibilityHints(material: ThreeMaterialLike | undefined): void {
  if (!material) return
  optionalLineCap(material.linecap)
  optionalLineJoin(material.linejoin)
}

export function optionalLineCap(value: unknown): void {
  if (value == null) return
  if (typeof value !== 'string') {
    throw new TypeError('material.linecap must be a string.')
  }
  if (value !== 'butt' && value !== 'round' && value !== 'square') {
    throw new Error(
      `material.linecap ${JSON.stringify(value)} is not supported by @headless-three/renderer. Use "butt", "round", "square", null, or undefined.`,
    )
  }
}

export function optionalLineJoin(value: unknown): void {
  if (value == null) return
  if (typeof value !== 'string') {
    throw new TypeError('material.linejoin must be a string.')
  }
  if (value !== 'round' && value !== 'bevel' && value !== 'miter') {
    throw new Error(
      `material.linejoin ${JSON.stringify(value)} is not supported by @headless-three/renderer. Use "round", "bevel", "miter", null, or undefined.`,
    )
  }
}

export function normalizedMaterialOrObjectNumber(value: unknown, label: string, fallback: number): number {
  const number = finiteMaterialOrObjectNumber(value, label, fallback)
  if (number < 0 || number > 1) {
    throw new TypeError(`${label} must be between 0 and 1.`)
  }
  return number
}

export function cameraZoomOrDefault(value: unknown): number {
  if (value == null) return 1
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError('camera.zoom must be a finite number.')
  }
  if (value <= 0) {
    throw new TypeError('camera.zoom must be positive.')
  }
  return value
}

export function finiteCountOrDefault(value: unknown, label: string, fallback: number, allowInfinity = false): number {
  if (value == null) return fallback
  if (allowInfinity && value === Infinity) return value
  if (typeof value === 'number' && Number.isFinite(value)) return value
  throw new TypeError(`${label} must be a finite number.`)
}

export function integerCountOrDefault(value: unknown, label: string, fallback: number, allowInfinity = false): number {
  const number = finiteCountOrDefault(value, label, fallback, allowInfinity)
  if (number === Infinity) return number
  if (!Number.isInteger(number)) {
    throw new TypeError(`${label} must be an integer.`)
  }
  if (number < 0) {
    throw new TypeError(`${label} must be non-negative.`)
  }
  return number
}

export function unsignedSortKey(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? value : fallback
}
