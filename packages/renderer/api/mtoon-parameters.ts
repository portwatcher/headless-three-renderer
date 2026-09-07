/** Native MToon lighting factors. Texture data uses the otherwise idle PBR slots. */
export interface MtoonParameters {
  shadeColor: number[]
  shadingShift: number
  shadingToony: number
  shadingShiftTextureScale: number
  rimColor: number[]
  rimFresnelPower: number
  rimLift: number
  matcapColor: number[]
  v0CompatShade: boolean
  outlineColor: number[]
  outlineLightingMix: number
  outlineWidth: number
  outlineMode: number
}
