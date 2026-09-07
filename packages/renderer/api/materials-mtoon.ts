import type { PbrProperties, ThreeMaterialLike } from './types'
import type { MaterialExtractionContext } from './materials.part-001'
import { materialSlotColor } from './materials.part-002'
import { assignPbrTextureSamplerState } from './materials.part-003'
import { finiteNumberOrDefault } from './materials.part-005'
import { extractTextureFromSlot } from './materials.part-009'

/** Preserve Pixiv's live expression-bound factors without replacing its material. */
export const extractMtoonProperties = function (
  material: ThreeMaterialLike,
  props: PbrProperties,
  context: MaterialExtractionContext,
): void {
  if (material.isMToonMaterial !== true) {
    return
  }
  const color = (slot: string, value: unknown, fallback: number[]) =>
    materialSlotColor(material, slot, value, `material.${slot}`, context)?.slice(0, 3) ?? fallback
  const scalar = (value: unknown, name: string, fallback: number) =>
    finiteNumberOrDefault(value, `material.${name}`, fallback)
  props.mtoon = {
    shadeColor: color('shadeColorFactor', material.shadeColorFactor, [0, 0, 0]),
    shadingShift: scalar(material.shadingShiftFactor, 'shadingShiftFactor', 0),
    shadingToony: scalar(material.shadingToonyFactor, 'shadingToonyFactor', 0.9),
    shadingShiftTextureScale: material.shadingShiftTexture
      ? scalar(material.shadingShiftTextureScale, 'shadingShiftTextureScale', 1) : 0,
    rimColor: color('parametricRimColorFactor', material.parametricRimColorFactor, [0, 0, 0]),
    rimFresnelPower: scalar(material.parametricRimFresnelPowerFactor, 'parametricRimFresnelPowerFactor', 5),
    rimLift: scalar(material.parametricRimLiftFactor, 'parametricRimLiftFactor', 0),
    matcapColor: material.matcapTexture
      ? color('matcapFactor', material.matcapFactor, [1, 1, 1]) : [0, 0, 0],
    v0CompatShade: material.v0CompatShade === true,
    outlineColor: color('outlineColorFactor', material.outlineColorFactor, [0, 0, 0]),
    outlineLightingMix: scalar(material.outlineLightingMixFactor, 'outlineLightingMixFactor', 1),
    outlineWidth: material.isOutline === true
      ? scalar(material.outlineWidthFactor, 'outlineWidthFactor', 0) : 0,
    outlineMode: material.isOutline !== true ? 0
      : material.outlineWidthMode === 'screenCoordinates' ? 2
      : material.outlineWidthMode === 'worldCoordinates' ? 1 : 0,
  }
  // Reuse directly uploaded texture slots, preserving their UVs and samplers.
  // PBR sheen/specular-color slots would repack large textures on the CPU each frame.
  const slots = [
    ['shadeMultiplyTexture', 'matcapMap'],
    ['rimMultiplyTexture', 'specularMap'],
    ['matcapTexture', 'lightMap'],
    ['shadingShiftTexture', 'metallicRoughnessTexture'],
    ['outlineWidthMultiplyTexture', 'aoMap'],
  ] as const
  const target = props as Record<string, unknown>
  for (const [source, destination] of slots) {
    const texture = material[source]
    const label = `material.${source}`
    const info = extractTextureFromSlot(texture, label, context.textureCache)
    if (!info) {
      continue
    }
    target[destination] = info.data
    target[`${destination}Width`] = info.width
    target[`${destination}Height`] = info.height
    assignPbrTextureSamplerState(props, destination, texture, label, context)
  }
}

/** An outline is drawable only when it has a supported, nonzero extrusion. */
export const isMaterialSurfaceVisible = function (material: ThreeMaterialLike | undefined): boolean {
  if (material?.visible === false) {
    return false
  }
  if (material?.isMToonMaterial === true && material.isOutline === true) {
    return (material.outlineWidthFactor ?? 0) > 0
      && (material.outlineWidthMode === 'worldCoordinates' || material.outlineWidthMode === 'screenCoordinates')
  }
  return true
}

/** Pixiv uses ignoreVertexColor instead of ShaderMaterial.vertexColors. */
export const usesMaterialVertexColors = function (material: ThreeMaterialLike | undefined): boolean {
  return material?.isMToonMaterial === true
    ? material.ignoreVertexColor !== true
    : material?.vertexColors !== false
}
