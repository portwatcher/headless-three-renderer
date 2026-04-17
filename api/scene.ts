import type { ThreeObject3DLike, ThreeBufferGeometryLike, NativeSceneMesh, GeometryGroup } from './types'
import { IDENTITY_4X4, matrixElements, clampInteger } from './math'
import { getAttribute, readVec3Attribute, readVec2Attribute, readColorAttribute, readIndexAttribute } from './attributes'
import { materialForGroup, materialColor, extractPbrProperties, extractTextureData } from './materials'
import { applyCpuSkinning } from './skinning'
import { applyMorphTargets } from './morphs'

export function flattenScene(scene: ThreeObject3DLike): NativeSceneMesh[] {
  const meshes: NativeSceneMesh[] = []
  visitObject(scene, true, meshes)
  return meshes
}

function visitObject(object: ThreeObject3DLike, parentVisible: boolean, meshes: NativeSceneMesh[]): void {
  if (!object || !parentVisible || object.visible === false) return

  if (object.isMesh === true && object.geometry) {
    appendMesh(object, meshes)
  } else if ((object.isLineSegments === true || object.isLineLoop === true || object.isLine === true) && object.geometry) {
    appendLineOrPoints(object, meshes, 'lines')
  } else if (object.isPoints === true && object.geometry) {
    appendLineOrPoints(object, meshes, 'points')
  }

  const children = Array.isArray(object.children) ? object.children : []
  for (const child of children) {
    visitObject(child, true, meshes)
  }
}

function appendMesh(object: ThreeObject3DLike, meshes: NativeSceneMesh[]): void {
  const geometry = object.geometry!
  const position = getAttribute(geometry, 'position')
  if (!position) return

  let positions = readVec3Attribute(position)
  const uvAttribute = getAttribute(geometry, 'uv')
  const uvs = uvAttribute ? readVec2Attribute(uvAttribute) : null
  const normalAttribute = getAttribute(geometry, 'normal')
  let normals = normalAttribute ? readVec3Attribute(normalAttribute) : null
  const vertexColors = getAttribute(geometry, 'color')
  const index = geometry.index ? readIndexAttribute(geometry.index) : null
  const groups = effectiveGroups(geometry, index, position.count)

  // CPU-side morph targets (blend shapes / shape keys / VRM blendshapes)
  if (object.morphTargetInfluences && object.morphTargetInfluences.length > 0) {
    const morphed = applyMorphTargets(object, positions, normals)
    positions = morphed.positions
    normals = morphed.normals
  }

  // CPU-side skinning for SkinnedMesh (Three.js, VRM, VRMA)
  if (object.isSkinnedMesh === true && object.skeleton) {
    const skinned = applyCpuSkinning(object, positions, normals)
    positions = skinned.positions
    normals = skinned.normals
  }

  // For skinned meshes, positions are already in world space after CPU skinning.
  const isSkinned = object.isSkinnedMesh === true && object.skeleton
  const meshTransform = isSkinned
    ? IDENTITY_4X4.slice()
    : matrixElements(object.matrixWorld!, 'mesh.matrixWorld')

  for (const group of groups) {
    const material = materialForGroup(object.material, group.materialIndex)
    if (material?.visible === false) continue

    const color = materialColor(material)
    const useVertexColors = vertexColors && material?.vertexColors !== false
    const pbrProps = extractPbrProperties(material)
    const textureInfo = extractTextureData(material)

    if (index) {
      const indices = index.slice(group.start, group.start + group.count)
      if (indices.length % 3 !== 0) {
        throw new Error(`THREE.Mesh "${object.name || object.uuid || '<unnamed>'}" has a non-triangle index range`)
      }

      meshes.push({
        positions,
        indices,
        normals: normals ?? undefined,
        color,
        colors: useVertexColors ? readColorAttribute(vertexColors!, color) : undefined,
        uvs: uvs ?? undefined,
        texture: textureInfo?.data,
        textureWidth: textureInfo?.width ?? undefined,
        textureHeight: textureInfo?.height ?? undefined,
        textureWrapS: textureInfo?.wrapS,
        textureWrapT: textureInfo?.wrapT,
        transform: meshTransform,
        ...pbrProps,
      })
    } else {
      if (group.count % 3 !== 0) {
        throw new Error(`THREE.Mesh "${object.name || object.uuid || '<unnamed>'}" has a non-triangle vertex range`)
      }

      meshes.push({
        positions: positions.slice(group.start * 3, (group.start + group.count) * 3),
        normals: normals ? normals.slice(group.start * 3, (group.start + group.count) * 3) : undefined,
        color,
        colors: useVertexColors
          ? readColorAttribute(vertexColors!, color).slice(group.start * 4, (group.start + group.count) * 4)
          : undefined,
        uvs: uvs ? uvs.slice(group.start * 2, (group.start + group.count) * 2) : undefined,
        texture: textureInfo?.data,
        textureWidth: textureInfo?.width ?? undefined,
        textureHeight: textureInfo?.height ?? undefined,
        textureWrapS: textureInfo?.wrapS,
        textureWrapT: textureInfo?.wrapT,
        transform: meshTransform,
        ...pbrProps,
      })
    }
  }
}

function effectiveGroups(
  geometry: ThreeBufferGeometryLike,
  index: number[] | null,
  vertexCount: number,
): GeometryGroup[] {
  const range = geometry.drawRange ?? {}
  const maxCount = index ? index.length : vertexCount
  const drawStart = clampInteger(range.start ?? 0, 0, maxCount)
  const requestedCount = range.count == null || range.count === Infinity ? maxCount : range.count
  const drawEnd = clampInteger(drawStart + requestedCount, drawStart, maxCount)
  const sourceGroups = Array.isArray(geometry.groups) && geometry.groups.length
    ? geometry.groups
    : [{ start: drawStart, count: drawEnd - drawStart, materialIndex: 0 }]

  const groups: GeometryGroup[] = []
  for (const group of sourceGroups) {
    const start = Math.max(drawStart, clampInteger(group.start ?? 0, 0, maxCount))
    const end = Math.min(drawEnd, clampInteger((group.start ?? 0) + (group.count ?? 0), 0, maxCount))
    if (end > start) {
      groups.push({
        start,
        count: end - start,
        materialIndex: group.materialIndex ?? 0,
      })
    }
  }
  return groups
}

/**
 * Emit a `NativeSceneMesh` with `topology: 'lines'` or `'points'` for
 * `THREE.Line` / `THREE.LineSegments` / `THREE.LineLoop` / `THREE.Points`.
 * Lines are always expanded to a LineList (pairs of vertex indices) so the
 * Rust side only has to deal with one line topology.
 */
function appendLineOrPoints(
  object: ThreeObject3DLike,
  meshes: NativeSceneMesh[],
  topology: 'lines' | 'points',
): void {
  const geometry = object.geometry!
  const position = getAttribute(geometry, 'position')
  if (!position) return

  const material = materialForGroup(object.material, 0)
  if (material?.visible === false) return

  const positions = readVec3Attribute(position)
  const vertexColors = getAttribute(geometry, 'color')
  const indexAttr = geometry.index ? readIndexAttribute(geometry.index) : null
  const vertexCount = position.count

  const range = geometry.drawRange ?? {}
  const sourceCount = indexAttr ? indexAttr.length : vertexCount
  const drawStart = clampInteger(range.start ?? 0, 0, sourceCount)
  const requestedCount = range.count == null || range.count === Infinity ? sourceCount : range.count
  const drawEnd = clampInteger(drawStart + requestedCount, drawStart, sourceCount)

  let indices: number[] | null = null
  if (topology === 'lines') {
    const source = indexAttr ?? rangeIndices(vertexCount)
    indices = expandLineIndices(source, drawStart, drawEnd, object)
    if (indices.length < 2) return
  } else if (indexAttr) {
    indices = indexAttr.slice(drawStart, drawEnd)
    if (indices.length === 0) return
  }

  const color = materialColor(material)
  const useVertexColors = vertexColors && material?.vertexColors !== false
  const colors = useVertexColors ? readColorAttribute(vertexColors!, color) : undefined

  meshes.push({
    positions,
    indices: indices ?? undefined,
    color,
    colors,
    transform: matrixElements(object.matrixWorld!, 'object.matrixWorld'),
    transparent: material?.transparent === true || (material?.opacity != null && material.opacity < 1),
    alphaTest: material && Number.isFinite(material.alphaTest) && material.alphaTest! > 0 ? material.alphaTest : undefined,
    shadingModel: 'basic',
    topology,
  })
}

function rangeIndices(count: number): number[] {
  const out = new Array<number>(count)
  for (let i = 0; i < count; i++) out[i] = i
  return out
}

/**
 * Convert a LineStrip / LineSegments / LineLoop index stream into a flat
 * LineList `[a, b, b, c, ...]` array.
 */
function expandLineIndices(
  source: number[],
  start: number,
  end: number,
  object: ThreeObject3DLike,
): number[] {
  const count = end - start
  if (count < 2) return []

  if (object.isLineSegments === true) {
    // already pairs; just validate alignment
    const aligned = count - (count % 2)
    return source.slice(start, start + aligned)
  }

  const out: number[] = []
  for (let i = 0; i < count - 1; i++) {
    out.push(source[start + i], source[start + i + 1])
  }
  if (object.isLineLoop === true && count >= 2) {
    out.push(source[start + count - 1], source[start])
  }
  return out
}
