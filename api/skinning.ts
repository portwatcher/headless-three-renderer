import type { ThreeObject3DLike, Mat4 } from './types'
import { IDENTITY_4X4, multiplyMatrices, invertMatrix4, blendBoneMatrices } from './math'
import { getAttribute, attributeComponent } from './attributes'

/**
 * CPU-side skeletal skinning for SkinnedMesh.
 * Supports Three.js SkinnedMesh, @pixiv/three-vrm, and VRMA animations.
 *
 * Three.js bone transform formula per vertex:
 *   boneMatrix[i] = skeleton.bones[i].matrixWorld * skeleton.boneInverses[i]
 *   skinMatrix = sum(weight[i] * boneMatrix[skinIndex[i]])
 *   worldPos = mesh.bindMatrixInverse * skinMatrix * mesh.bindMatrix * localPos
 */
export function applyCpuSkinning(
  mesh: ThreeObject3DLike,
  positions: number[],
  normals: number[] | null,
): { positions: number[]; normals: number[] | null } {
  const geometry = mesh.geometry!
  const skeleton = mesh.skeleton!

  const skinIndexAttr = getAttribute(geometry, 'skinIndex')
  const skinWeightAttr = getAttribute(geometry, 'skinWeight')

  if (!skinIndexAttr || !skinWeightAttr || !skeleton) {
    return { positions, normals }
  }

  if (typeof skeleton.update === 'function') {
    skeleton.update()
  }

  const bones = skeleton.bones
  const boneInverses = skeleton.boneInverses
  if (!bones || !boneInverses || bones.length === 0) {
    return { positions, normals }
  }

  const boneCount = bones.length
  const boneMatrices = new Array<Mat4>(boneCount)
  for (let i = 0; i < boneCount; i++) {
    const boneWorld = bones[i]?.matrixWorld?.elements
    const boneInv = boneInverses[i]?.elements
    if (!boneWorld || !boneInv) {
      boneMatrices[i] = IDENTITY_4X4
      continue
    }
    boneMatrices[i] = multiplyMatrices(Array.from(boneWorld), Array.from(boneInv))
  }

  const bindMatrix = mesh.bindMatrix?.elements
    ? Array.from(mesh.bindMatrix.elements)
    : IDENTITY_4X4
  const bindMatrixInverse = mesh.bindMatrixInverse?.elements
    ? Array.from(mesh.bindMatrixInverse.elements)
    : invertMatrix4(bindMatrix)

  const vertexCount = positions.length / 3
  const skinnedPositions = new Array<number>(positions.length)
  const skinnedNormals = normals ? new Array<number>(normals.length) : null

  for (let vi = 0; vi < vertexCount; vi++) {
    const ji0 = Math.floor(attributeComponent(skinIndexAttr, vi, 0))
    const ji1 = Math.floor(attributeComponent(skinIndexAttr, vi, 1))
    const ji2 = Math.floor(attributeComponent(skinIndexAttr, vi, 2))
    const ji3 = Math.floor(attributeComponent(skinIndexAttr, vi, 3))
    const jw0 = attributeComponent(skinWeightAttr, vi, 0)
    const jw1 = attributeComponent(skinWeightAttr, vi, 1)
    const jw2 = attributeComponent(skinWeightAttr, vi, 2)
    const jw3 = attributeComponent(skinWeightAttr, vi, 3)

    const skinMatrix = blendBoneMatrices(
      boneMatrices, boneCount,
      ji0, ji1, ji2, ji3,
      jw0, jw1, jw2, jw3,
    )

    const finalMatrix = multiplyMatrices(bindMatrixInverse, multiplyMatrices(skinMatrix, bindMatrix))

    const px = positions[vi * 3]
    const py = positions[vi * 3 + 1]
    const pz = positions[vi * 3 + 2]
    skinnedPositions[vi * 3] = finalMatrix[0] * px + finalMatrix[4] * py + finalMatrix[8] * pz + finalMatrix[12]
    skinnedPositions[vi * 3 + 1] = finalMatrix[1] * px + finalMatrix[5] * py + finalMatrix[9] * pz + finalMatrix[13]
    skinnedPositions[vi * 3 + 2] = finalMatrix[2] * px + finalMatrix[6] * py + finalMatrix[10] * pz + finalMatrix[14]

    if (normals && skinnedNormals) {
      const nx = normals[vi * 3]
      const ny = normals[vi * 3 + 1]
      const nz = normals[vi * 3 + 2]
      let snx = finalMatrix[0] * nx + finalMatrix[4] * ny + finalMatrix[8] * nz
      let sny = finalMatrix[1] * nx + finalMatrix[5] * ny + finalMatrix[9] * nz
      let snz = finalMatrix[2] * nx + finalMatrix[6] * ny + finalMatrix[10] * nz
      const len = Math.sqrt(snx * snx + sny * sny + snz * snz)
      if (len > 1e-8) {
        snx /= len
        sny /= len
        snz /= len
      }
      skinnedNormals[vi * 3] = snx
      skinnedNormals[vi * 3 + 1] = sny
      skinnedNormals[vi * 3 + 2] = snz
    }
  }

  return { positions: skinnedPositions, normals: skinnedNormals }
}
