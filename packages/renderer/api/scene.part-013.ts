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
import { BatchedGeometryViewSignature, DashedLineExpansion, LineSegmentDistance, MeshInstance, SceneExtractionCache } from './scene.part-001'
import { attributeSignature, attributeSignatureCacheable, sameAttributeSignature } from './scene.part-003'
import { integerCountOrDefault, nonNegativeMaterialOrObjectNumber, vec3Like } from './scene.part-008'
import { instancedMeshSignature, multiplyMat4, readLocalInstancedMeshInstances, sameInstancedMeshSignature } from './scene.part-014'
export function sphereLike(
  sphere: ThreeSphereLike,
  label: string,
): { center: [number, number, number]; radius: number } {
  if (!sphere || typeof sphere !== 'object') {
    throw new TypeError(`${label} must be a THREE.Sphere-like object.`)
  }
  const center = vec3Like(sphere.center)
  if (!center) {
    throw new TypeError(`${label}.center must be a finite Vector3-like value.`)
  }
  const radius = nonNegativeMaterialOrObjectNumber(sphere.radius, `${label}.radius`, 0)
  return { center, radius }
}

export function batchedGeometryView(
  geometry: ThreeBufferGeometryLike,
  range: { start: number; count: number },
  cache?: SceneExtractionCache,
): ThreeBufferGeometryLike {
  const signature = batchedGeometryViewSignature(geometry, range)
  if (cache && signature.cacheable) {
    let views = cache.batchedGeometryViews.get(geometry)
    if (!views) {
      views = new Map()
      cache.batchedGeometryViews.set(geometry, views)
    }
    const key = batchedGeometryViewKey(range)
    const cached = views.get(key)
    if (cached && sameBatchedGeometryViewSignature(cached.signature, signature)) {
      return cached.view
    }
    const view = cached?.view ?? createBatchedGeometryView(geometry)
    updateBatchedGeometryView(view, geometry, range)
    views.set(key, { signature, view })
    return view
  }

  const view = createBatchedGeometryView(geometry)
  updateBatchedGeometryView(view, geometry, range)
  return view
}

export function createBatchedGeometryView(geometry: ThreeBufferGeometryLike): ThreeBufferGeometryLike {
  return Object.create(geometry) as ThreeBufferGeometryLike
}

export function updateBatchedGeometryView(
  view: ThreeBufferGeometryLike,
  geometry: ThreeBufferGeometryLike,
  range: { start: number; count: number },
): void {
  view.drawRange = { start: range.start, count: range.count }
  view.boundingSphere = batchedGeometryRangeBoundingSphere(geometry, range)
}

export function batchedGeometryViewSignature(
  geometry: ThreeBufferGeometryLike,
  range: { start: number; count: number },
): BatchedGeometryViewSignature {
  const signature: BatchedGeometryViewSignature = {
    cacheable: true,
    geometryVersion: geometry.version,
    rangeStart: range.start,
    rangeCount: range.count,
    position: attributeSignature(getAttribute(geometry, 'position')),
    index: attributeSignature(geometry.index),
  }
  signature.cacheable = attributeSignatureCacheable(signature.position)
    && attributeSignatureCacheable(signature.index)
  return signature
}

export function sameBatchedGeometryViewSignature(
  a: BatchedGeometryViewSignature,
  b: BatchedGeometryViewSignature,
): boolean {
  return a.cacheable === b.cacheable
    && a.geometryVersion === b.geometryVersion
    && a.rangeStart === b.rangeStart
    && a.rangeCount === b.rangeCount
    && sameAttributeSignature(a.position, b.position)
    && sameAttributeSignature(a.index, b.index)
}

export function batchedGeometryViewKey(range: { start: number; count: number }): string {
  return `${range.start}:${range.count}`
}

export function batchedGeometryRangeBoundingSphere(
  geometry: ThreeBufferGeometryLike,
  range: { start: number; count: number },
): { center: { x: number; y: number; z: number }; radius: number } | undefined {
  const position = getAttribute(geometry, 'position')
  if (!position || range.count <= 0) return undefined

  const index = geometry.index ? readIndexAttribute(geometry.index, 'geometry.index', position.count) : null
  const start = Math.max(0, range.start)
  const end = Math.max(start, range.start + range.count)
  let minX = Number.POSITIVE_INFINITY
  let minY = Number.POSITIVE_INFINITY
  let minZ = Number.POSITIVE_INFINITY
  let maxX = Number.NEGATIVE_INFINITY
  let maxY = Number.NEGATIVE_INFINITY
  let maxZ = Number.NEGATIVE_INFINITY
  const vertexIndices: number[] = []

  for (let offset = start; offset < end; offset += 1) {
    const vertexIndex = index ? index[offset] : offset
    if (!Number.isInteger(vertexIndex) || vertexIndex < 0 || vertexIndex >= position.count) continue
    vertexIndices.push(vertexIndex)
    const x = attributeComponent(position, vertexIndex, 0, 'geometry.attributes.position')
    const y = attributeComponent(position, vertexIndex, 1, 'geometry.attributes.position')
    const z = attributeComponent(position, vertexIndex, 2, 'geometry.attributes.position')
    minX = Math.min(minX, x)
    minY = Math.min(minY, y)
    minZ = Math.min(minZ, z)
    maxX = Math.max(maxX, x)
    maxY = Math.max(maxY, y)
    maxZ = Math.max(maxZ, z)
  }

  if (vertexIndices.length === 0) return undefined

  const center = {
    x: (minX + maxX) * 0.5,
    y: (minY + maxY) * 0.5,
    z: (minZ + maxZ) * 0.5,
  }
  let radiusSq = 0
  for (const vertexIndex of vertexIndices) {
    const dx = attributeComponent(position, vertexIndex, 0, 'geometry.attributes.position') - center.x
    const dy = attributeComponent(position, vertexIndex, 1, 'geometry.attributes.position') - center.y
    const dz = attributeComponent(position, vertexIndex, 2, 'geometry.attributes.position') - center.z
    radiusSq = Math.max(radiusSq, dx * dx + dy * dy + dz * dz)
  }

  return { center, radius: Math.sqrt(radiusSq) }
}

export function batchedInstanceMatrix(object: ThreeObject3DLike, instanceId: number): number[] {
  const data = batchedTextureImageData(
    object._matricesTexture,
    'THREE.BatchedMesh._matricesTexture',
    instanceId,
  )
  const offset = instanceId * 16
  if (typeof data.length !== 'number' || data.length < offset + 16) {
    throw new Error(
      `THREE.BatchedMesh matrix texture is not readable for instance ${instanceId}. Use a real THREE.BatchedMesh or expand the batch before rendering.`,
    )
  }

  const matrix = new Array<number>(16)
  for (let component = 0; component < 16; component += 1) {
    matrix[component] = finiteArrayValue(data, offset + component, 'THREE.BatchedMesh._matricesTexture.image.data')
  }
  return matrix
}

export function batchedInstanceColor(object: ThreeObject3DLike, instanceId: number): Color4 | undefined {
  const texture = object._colorsTexture
  if (texture == null) return undefined
  const data = batchedTextureImageData(texture, 'THREE.BatchedMesh._colorsTexture', instanceId)

  const offset = instanceId * 4
  if (typeof data.length !== 'number' || data.length < offset + 4) {
    throw new Error(
      `THREE.BatchedMesh color texture is not readable for instance ${instanceId}. Use a real THREE.BatchedMesh or expand the batch before rendering.`,
    )
  }

  return [
    finiteArrayValue(data, offset, 'THREE.BatchedMesh._colorsTexture.image.data'),
    finiteArrayValue(data, offset + 1, 'THREE.BatchedMesh._colorsTexture.image.data'),
    finiteArrayValue(data, offset + 2, 'THREE.BatchedMesh._colorsTexture.image.data'),
    finiteArrayValue(data, offset + 3, 'THREE.BatchedMesh._colorsTexture.image.data'),
  ]
}

export function batchedTextureImageData(texture: unknown, label: string, instanceId: number): ArrayLike<number> {
  if (!texture || typeof texture !== 'object') {
    throw new TypeError(`${label} must be a texture-like object for instance ${instanceId}.`)
  }
  const image = (texture as { image?: unknown }).image
  if (!image || typeof image !== 'object') {
    throw new TypeError(`${label}.image must be an image-like object for instance ${instanceId}.`)
  }
  const data = (image as { data?: unknown }).data
  if (!data || typeof data !== 'object') {
    throw new TypeError(`${label}.image.data must be an array-like object for instance ${instanceId}.`)
  }
  return data as ArrayLike<number>
}

export function batchedOptionalBoolean(value: unknown, label: string, fallback: boolean): boolean {
  if (value == null) return fallback
  if (typeof value === 'boolean') return value
  throw new TypeError(`${label} must be a boolean.`)
}

export function batchedNonNegativeInteger(value: unknown, label: string): number {
  if (typeof value === 'number' && Number.isFinite(value) && Number.isInteger(value) && value >= 0) {
    return value
  }
  throw new TypeError(`${label} must be a non-negative integer.`)
}

export function finiteArrayValue(values: ArrayLike<number>, index: number, label: string): number {
  const value = values[index]
  if (typeof value === 'number' && Number.isFinite(value)) return value
  throw new TypeError(`${label}[${index}] must be a finite number.`)
}

export function appendNumberArray(out: number[], values: number[]): void {
  for (const value of values) out.push(value)
}

export function createDashedLineExpansion(
  uvs: number[] | null,
  uvs2: number[] | null,
  colors: number[] | undefined,
): DashedLineExpansion {
  return {
    positions: [],
    uvs: uvs ? [] : undefined,
    uvs2: uvs2 ? [] : undefined,
    colors: colors ? [] : undefined,
  }
}

export function cameraFrustumIntersectsSphere(
  camera: ThreeCameraLike,
  center: [number, number, number],
  radius: number,
): boolean {
  const view = camera.matrixWorldInverse?.elements
  const projection = camera.projectionMatrix?.elements
  if (!view || view.length < 16 || !projection || projection.length < 16) return true
  const viewProjection = multiplyMat4(projection, view)

  const planes: Array<[number, number, number, number]> = [
    [viewProjection[3] + viewProjection[0], viewProjection[7] + viewProjection[4], viewProjection[11] + viewProjection[8], viewProjection[15] + viewProjection[12]],
    [viewProjection[3] - viewProjection[0], viewProjection[7] - viewProjection[4], viewProjection[11] - viewProjection[8], viewProjection[15] - viewProjection[12]],
    [viewProjection[3] + viewProjection[1], viewProjection[7] + viewProjection[5], viewProjection[11] + viewProjection[9], viewProjection[15] + viewProjection[13]],
    [viewProjection[3] - viewProjection[1], viewProjection[7] - viewProjection[5], viewProjection[11] - viewProjection[9], viewProjection[15] - viewProjection[13]],
    [viewProjection[3] + viewProjection[2], viewProjection[7] + viewProjection[6], viewProjection[11] + viewProjection[10], viewProjection[15] + viewProjection[14]],
    [viewProjection[3] - viewProjection[2], viewProjection[7] - viewProjection[6], viewProjection[11] - viewProjection[10], viewProjection[15] - viewProjection[14]],
  ]

  for (const [a, b, c, d] of planes) {
    const length = Math.hypot(a, b, c)
    if (!Number.isFinite(length) || length <= 1e-8) continue
    const distance = (a * center[0] + b * center[1] + c * center[2] + d) / length
    if (distance < -radius) return false
  }

  return true
}

export function lineSegmentsWithDistances(
  positions: number[],
  source: number[],
  start: number,
  end: number,
  object: ThreeObject3DLike,
  lineDistance: ThreeBufferAttributeLike | undefined,
): LineSegmentDistance[] {
  const count = end - start
  const segments: LineSegmentDistance[] = []
  if (count < 2) return segments

  if (object.isLineSegments === true) {
    let cumulative = 0
    const aligned = count - (count % 2)
    for (let i = 0; i < aligned; i += 2) {
      const a = source[start + i]
      const b = source[start + i + 1]
      const length = vertexDistance(positions, a, b)
      const d0 = lineDistance ? attributeComponent(lineDistance, a, 0, 'geometry.attributes.lineDistance') : cumulative
      const d1 = lineDistance ? attributeComponent(lineDistance, b, 0, 'geometry.attributes.lineDistance') : d0 + length
      segments.push({ a, b, d0, d1 })
      cumulative = d1
    }
    return segments
  }

  let previous = source[start]
  let previousDistance = lineDistance ? attributeComponent(lineDistance, previous, 0, 'geometry.attributes.lineDistance') : 0
  for (let i = 1; i < count; i += 1) {
    const current = source[start + i]
    const length = vertexDistance(positions, previous, current)
    const currentDistance = lineDistance ? attributeComponent(lineDistance, current, 0, 'geometry.attributes.lineDistance') : previousDistance + length
    segments.push({ a: previous, b: current, d0: previousDistance, d1: currentDistance })
    previous = current
    previousDistance = currentDistance
  }
  if (object.isLineLoop === true && count >= 2) {
    const first = source[start]
    segments.push({
      a: previous,
      b: first,
      d0: previousDistance,
      d1: lineDistance
        ? attributeComponent(lineDistance, first, 0, 'geometry.attributes.lineDistance')
        : previousDistance + vertexDistance(positions, previous, first),
    })
  }
  return segments
}

export function appendDashedSegment(
  out: DashedLineExpansion,
  positions: number[],
  uvs: number[] | null,
  uvs2: number[] | null,
  colors: number[] | undefined,
  segment: LineSegmentDistance,
  scale: number,
  dashSize: number,
  totalSize: number,
): void {
  const s0 = segment.d0 * scale
  const s1 = segment.d1 * scale
  const span = s1 - s0
  const direction = Math.sign(span)
  if (Math.abs(span) <= 1e-6 || direction === 0) return

  let cursor = s0
  let guard = 0
  while ((direction > 0 ? cursor < s1 - 1e-6 : cursor > s1 + 1e-6) && guard < 10000) {
    guard += 1
    const cycle = Math.floor(cursor / totalSize)
    const cycleStart = cycle * totalSize
    const inCycle = cursor - cycleStart
    const visible = inCycle <= dashSize
    const boundary = direction > 0
      ? cycleStart + (visible ? dashSize : totalSize)
      : cycleStart + (visible ? 0 : dashSize)
    const next = direction > 0
      ? Math.min(s1, boundary <= cursor + 1e-6 ? cursor + 1e-6 : boundary)
      : Math.max(s1, boundary >= cursor - 1e-6 ? cursor - 1e-6 : boundary)
    const hasVisibleSpan = direction > 0 ? next > cursor + 1e-6 : next < cursor - 1e-6
    if (visible && hasVisibleSpan) {
      const t0 = (cursor - s0) / span
      const t1 = (next - s0) / span
      appendInterpolatedLine(out, positions, uvs, uvs2, colors, segment.a, segment.b, t0, t1)
    }
    cursor = next
  }
}

export function appendInterpolatedLine(
  out: DashedLineExpansion,
  positions: number[],
  uvs: number[] | null,
  uvs2: number[] | null,
  colors: number[] | undefined,
  a: number,
  b: number,
  t0: number,
  t1: number,
): void {
  appendInterpolatedAttribute(out.positions, positions, 3, a, b, t0)
  appendInterpolatedAttribute(out.positions, positions, 3, a, b, t1)
  if (out.uvs && uvs) {
    appendInterpolatedAttribute(out.uvs, uvs, 2, a, b, t0)
    appendInterpolatedAttribute(out.uvs, uvs, 2, a, b, t1)
  }
  if (out.uvs2 && uvs2) {
    appendInterpolatedAttribute(out.uvs2, uvs2, 2, a, b, t0)
    appendInterpolatedAttribute(out.uvs2, uvs2, 2, a, b, t1)
  }
  if (out.colors && colors) {
    appendInterpolatedAttribute(out.colors, colors, 4, a, b, t0)
    appendInterpolatedAttribute(out.colors, colors, 4, a, b, t1)
  }
}

export function appendInterpolatedAttribute(
  out: number[],
  values: number[],
  itemSize: number,
  a: number,
  b: number,
  t: number,
): void {
  const aBase = a * itemSize
  const bBase = b * itemSize
  for (let component = 0; component < itemSize; component += 1) {
    const av = values[aBase + component]
    const bv = values[bBase + component]
    out.push(av + (bv - av) * t)
  }
}

export function vertexDistance(positions: number[], a: number, b: number): number {
  const dx = positions[a * 3] - positions[b * 3]
  const dy = positions[a * 3 + 1] - positions[b * 3 + 1]
  const dz = positions[a * 3 + 2] - positions[b * 3 + 2]
  return Math.sqrt(dx * dx + dy * dy + dz * dz)
}

export function meshInstances(
  object: ThreeObject3DLike,
  baseTransform: number[],
  cache?: SceneExtractionCache,
): MeshInstance[] {
  if (object.isInstancedMesh !== true) {
    return [{ transform: baseTransform }]
  }

  const instanceMatrix = object.instanceMatrix
  if (!instanceMatrix || instanceMatrix.count == null) return []
  const instanceMatrixCount = attributeCount(instanceMatrix, 'InstancedMesh.instanceMatrix')
  const count = clampInteger(
    integerCountOrDefault(object.count, 'InstancedMesh.count', instanceMatrixCount),
    0,
    instanceMatrixCount,
  )
  const signature = instancedMeshSignature(object, instanceMatrix, count)
  let localInstances: MeshInstance[] | undefined

  if (cache && signature.cacheable) {
    const cached = cache.instancedMeshes.get(object)
    if (cached && sameInstancedMeshSignature(cached.signature, signature)) {
      localInstances = cached.localInstances
    }
  }

  if (!localInstances) {
    localInstances = readLocalInstancedMeshInstances(object, instanceMatrix, count)
    if (cache && signature.cacheable) {
      cache.instancedMeshes.set(object, { signature, localInstances })
    }
  }

  return localInstances.map((instance) => ({
    transform: multiplyMat4(baseTransform, instance.transform),
    color: instance.color,
  }))
}
