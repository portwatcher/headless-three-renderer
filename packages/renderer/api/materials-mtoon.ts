import type { ThreeMaterialLike } from './types'

/** MToon outline groups need vertex extrusion; do not draw them as filled surfaces. */
export const isMaterialSurfaceVisible = function (material: ThreeMaterialLike | undefined): boolean {
  return material?.visible !== false && !(material?.isMToonMaterial === true && material.isOutline === true)
}

/** Pixiv uses ignoreVertexColor instead of ShaderMaterial.vertexColors. */
export const usesMaterialVertexColors = function (material: ThreeMaterialLike | undefined): boolean {
  return material?.isMToonMaterial === true
    ? material.ignoreVertexColor !== true
    : material?.vertexColors !== false
}
