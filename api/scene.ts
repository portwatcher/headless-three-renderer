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
