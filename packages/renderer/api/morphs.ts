import type { ThreeObject3DLike, ThreeBufferAttributeLike } from './types'
import { attributeComponent } from './attributes'

/**
 * CPU-side morph target (blend shape / shape key) application.
 *
 * Supports:
 * - Three.js `BufferGeometry.morphAttributes` with `mesh.morphTargetInfluences`
 * - VRM blend shapes (`@pixiv/three-vrm` sets `morphTargetInfluences` on the mesh)
 * - Blender shape keys exported via glTF (`GLTFLoader` populates the same attributes)
 *
 * Three.js morph target formula per vertex:
 *   morphedPosition = basePosition + sum(influence[i] * morphPosition[i][vertex])
 *   morphedNormal   = normalize(baseNormal + sum(influence[i] * morphNormal[i][vertex]))
 *
 * When `geometry.morphTargetsRelative` is true (the default for glTF), the morph
 * attributes store deltas. When false (legacy Three.js), they store absolute
 * positions and we compute the delta as `morphAttr[i][v] - baseAttr[v]`.
 */
export function applyMorphTargets(
  mesh: ThreeObject3DLike,
  positions: number[],
  normals: number[] | null,
): { positions: number[]; normals: number[] | null } {
  const geometry = mesh.geometry
  if (!geometry) return { positions, normals }

  const influences = mesh.morphTargetInfluences
  if (!influences || influences.length === 0) return { positions, normals }

  const morphAttributes = geometry.morphAttributes
  if (!morphAttributes) return { positions, normals }

  const morphPositions: ThreeBufferAttributeLike[] | undefined = morphAttributes.position as any
  const morphNormals: ThreeBufferAttributeLike[] | undefined = morphAttributes.normal as any

  if (!morphPositions || morphPositions.length === 0) return { positions, normals }

  const targetCount = Math.min(influences.length, morphPositions.length)
  const weights: number[] = []
  let hasEffect = false
  for (let t = 0; t < targetCount; t++) {
    const weight = morphTargetInfluence(influences[t], `morphTargetInfluences[${t}]`)
    weights.push(weight)
    if (weight !== 0) {
      hasEffect = true
    }
  }
  if (!hasEffect) return { positions, normals }

  const isRelative = optionalBoolean(geometry.morphTargetsRelative, 'geometry.morphTargetsRelative') !== false
  const vertexCount = positions.length / 3

  const morphedPositions = positions.slice()
  const morphedNormals = normals ? normals.slice() : null

  for (let t = 0; t < targetCount; t++) {
    const weight = weights[t]
    if (weight === 0) continue

    const morphPosAttr = morphPositions[t]
    if (!morphPosAttr) continue

    const morphNormAttr = morphNormals?.[t]
    const count = Math.min(vertexCount, morphPosAttr.count ?? 0)

    for (let vi = 0; vi < count; vi++) {
      const mx = attributeComponent(morphPosAttr, vi, 0, `geometry.morphAttributes.position[${t}]`)
      const my = attributeComponent(morphPosAttr, vi, 1, `geometry.morphAttributes.position[${t}]`)
      const mz = attributeComponent(morphPosAttr, vi, 2, `geometry.morphAttributes.position[${t}]`)

      if (isRelative) {
        // Morph attributes store deltas
        morphedPositions[vi * 3] += weight * mx
        morphedPositions[vi * 3 + 1] += weight * my
        morphedPositions[vi * 3 + 2] += weight * mz
      } else {
        // Legacy absolute mode: delta = morphPos - basePos
        morphedPositions[vi * 3] += weight * (mx - positions[vi * 3])
        morphedPositions[vi * 3 + 1] += weight * (my - positions[vi * 3 + 1])
        morphedPositions[vi * 3 + 2] += weight * (mz - positions[vi * 3 + 2])
      }

      if (morphedNormals && morphNormAttr) {
        const nx = attributeComponent(morphNormAttr, vi, 0, `geometry.morphAttributes.normal[${t}]`)
        const ny = attributeComponent(morphNormAttr, vi, 1, `geometry.morphAttributes.normal[${t}]`)
        const nz = attributeComponent(morphNormAttr, vi, 2, `geometry.morphAttributes.normal[${t}]`)

        if (isRelative) {
          morphedNormals[vi * 3] += weight * nx
          morphedNormals[vi * 3 + 1] += weight * ny
          morphedNormals[vi * 3 + 2] += weight * nz
        } else {
          morphedNormals[vi * 3] += weight * (nx - normals![vi * 3])
          morphedNormals[vi * 3 + 1] += weight * (ny - normals![vi * 3 + 1])
          morphedNormals[vi * 3 + 2] += weight * (nz - normals![vi * 3 + 2])
        }
      }
    }
  }

  // Re-normalize morphed normals
  if (morphedNormals) {
    for (let vi = 0; vi < vertexCount; vi++) {
      const nx = morphedNormals[vi * 3]
      const ny = morphedNormals[vi * 3 + 1]
      const nz = morphedNormals[vi * 3 + 2]
      const len = Math.sqrt(nx * nx + ny * ny + nz * nz)
      if (len > 1e-8) {
        morphedNormals[vi * 3] = nx / len
        morphedNormals[vi * 3 + 1] = ny / len
        morphedNormals[vi * 3 + 2] = nz / len
      }
    }
  }

  return { positions: morphedPositions, normals: morphedNormals }
}

function morphTargetInfluence(value: unknown, label: string): number {
  if (value == null) return 0
  if (typeof value === 'number' && Number.isFinite(value)) return value
  throw new TypeError(`${label} must be a finite number.`)
}

function optionalBoolean(value: unknown, label: string): boolean | undefined {
  if (value == null) return undefined
  if (typeof value === 'boolean') return value
  throw new TypeError(`${label} must be a boolean.`)
}
