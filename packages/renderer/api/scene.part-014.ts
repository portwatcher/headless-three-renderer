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
import { InstancedMeshSignature, MeshInstance } from './scene.part-001'
import { attributeSignature, attributeSignatureCacheable, sameAttributeSignature } from './scene.part-003'
export function instancedMeshSignature(
  object: ThreeObject3DLike,
  instanceMatrix: NonNullable<ThreeObject3DLike['instanceMatrix']>,
  count: number,
): InstancedMeshSignature {
  const signature: InstancedMeshSignature = {
    cacheable: true,
    count,
    instanceMatrix: attributeSignature(instanceMatrix),
    instanceColor: attributeSignature(object.instanceColor),
  }
  signature.cacheable = attributeSignatureCacheable(signature.instanceMatrix)
    && attributeSignatureCacheable(signature.instanceColor)
  return signature
}

export function sameInstancedMeshSignature(a: InstancedMeshSignature, b: InstancedMeshSignature): boolean {
  return a.cacheable === b.cacheable
    && a.count === b.count
    && sameAttributeSignature(a.instanceMatrix, b.instanceMatrix)
    && sameAttributeSignature(a.instanceColor, b.instanceColor)
}

export function readLocalInstancedMeshInstances(
  object: ThreeObject3DLike,
  instanceMatrix: NonNullable<ThreeObject3DLike['instanceMatrix']>,
  count: number,
): MeshInstance[] {
  const instances = new Array<MeshInstance>(count)
  for (let i = 0; i < count; i += 1) {
    instances[i] = {
      transform: readMat4Attribute(instanceMatrix, i),
      color: readInstanceColor(object.instanceColor, i),
    }
  }
  return instances
}

export function readMat4Attribute(attribute: ThreeObject3DLike['instanceMatrix'], index: number): number[] {
  if (!attribute) return IDENTITY_4X4.slice()
  const matrix = new Array<number>(16)
  for (let component = 0; component < 16; component += 1) {
    matrix[component] = attributeComponent(attribute, index, component, 'InstancedMesh.instanceMatrix')
  }
  return matrix
}

export function readInstanceColor(attribute: ThreeObject3DLike['instanceColor'], index: number): Color4 | undefined {
  if (!attribute || index >= attributeCount(attribute, 'InstancedMesh.instanceColor')) return undefined
  return [
    attributeComponent(attribute, index, 0, 'InstancedMesh.instanceColor'),
    attributeComponent(attribute, index, 1, 'InstancedMesh.instanceColor'),
    attributeComponent(attribute, index, 2, 'InstancedMesh.instanceColor'),
    attribute.itemSize && attribute.itemSize >= 4 ? attributeComponent(attribute, index, 3, 'InstancedMesh.instanceColor') : 1,
  ]
}

export function instanceColor(baseColor: Color4, instance: MeshInstance): Color4 {
  if (!instance.color) return baseColor
  return [
    baseColor[0] * instance.color[0],
    baseColor[1] * instance.color[1],
    baseColor[2] * instance.color[2],
    baseColor[3] * instance.color[3],
  ]
}

export function multiplyMat4(a: ArrayLike<number>, b: ArrayLike<number>): number[] {
  const out = new Array<number>(16)
  for (let col = 0; col < 4; col += 1) {
    for (let row = 0; row < 4; row += 1) {
      out[col * 4 + row] =
        a[row] * b[col * 4]
        + a[4 + row] * b[col * 4 + 1]
        + a[8 + row] * b[col * 4 + 2]
        + a[12 + row] * b[col * 4 + 3]
    }
  }
  return out
}

export function rangeIndices(count: number): number[] {
  const out = new Array<number>(count)
  for (let i = 0; i < count; i++) out[i] = i
  return out
}

/**
 * Convert a LineStrip / LineSegments / LineLoop index stream into a flat
 * LineList `[a, b, b, c, ...]` array.
 */
export function expandLineIndices(
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
