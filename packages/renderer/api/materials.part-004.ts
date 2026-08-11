import type { Color4, ThreeMaterialLike, PbrProperties, TextureInfo, ThreeTextureLike, ThreeSceneRootLike, ThreeObject3DLike } from './types'
import { clamp01 } from './math'
import { strictColorLikeToArray, validatedColorLikeToArray } from './color'
import { objectChildren } from './objects'
import { AddEquation, AdditiveBlending, AlwaysDepth, AlwaysStencilFunc, ConstantAlphaFactor, ConstantColorFactor, CustomBlending, DstAlphaFactor, DstColorFactor, EqualDepth, EqualStencilFunc, GreaterDepth, GreaterEqualDepth, GreaterEqualStencilFunc, GreaterStencilFunc, LessDepth, LessEqualDepth, LessEqualStencilFunc, LessStencilFunc, MaterialExtractionContext, MaxEquation, MinEquation, MultiplyBlending, NeverDepth, NeverStencilFunc, NoBlending, NormalBlending, NotEqualDepth, NotEqualStencilFunc, OneFactor, OneMinusConstantAlphaFactor, OneMinusConstantColorFactor, OneMinusDstAlphaFactor, OneMinusDstColorFactor, OneMinusSrcAlphaFactor, OneMinusSrcColorFactor, ReverseSubtractEquation, SrcAlphaFactor, SrcAlphaSaturateFactor, SrcColorFactor, SubtractEquation, SubtractiveBlending, ZeroFactor } from './materials.part-001'
import { MaterialColorSignature, materialSlotColor, materialSlotColorSignature, sameMaterialColorSignature, sameUnknownArray } from './materials.part-002'
import { MaterialScalarFeatureSignature } from './materials.part-003'
import { materialShadowSide, materialSide, materialStencilOperation, optionalBoolean, optionalFiniteNumber, optionalMaterialPrecision } from './materials.part-005'
import { copyShaderMaterialInfo, shaderMaterialKind } from './materials.part-008'
export function materialScalarFeatureSignature(
  material: ThreeMaterialLike,
  context: MaterialExtractionContext,
): MaterialScalarFeatureSignature {
  const iridescenceThicknessRange = material.iridescenceThicknessRange as ArrayLike<unknown> | undefined
  const hintInfo = material.isMeshDistanceMaterial ? materialHintSignatureInfo(material.userData) : []
  return {
    values: [
      context.materialEnvironmentSource,
      context.materialEnvironmentMaps?.has(material),
      material.envMap,
      material.envMap?.mapping,
      material.envMapIntensity,
      material.combine,
      material.reflectivity,
      material.refractionRatio,
      material.metalness,
      material.roughness,
      material.clearcoat,
      material.clearcoatRoughness,
      ...vector2SignatureValues(material.clearcoatNormalScale),
      material.sheenRoughness,
      material.anisotropy,
      material.anisotropyRotation,
      material.iridescence,
      material.iridescenceIOR,
      iridescenceThicknessRange,
      iridescenceThicknessRange?.length,
      iridescenceThicknessRange?.[0],
      iridescenceThicknessRange?.[1],
      material.transmission,
      material.dispersion,
      material.ior,
      material.thickness,
      material.attenuationDistance,
      material.specularIntensity,
      material.isMeshPhongMaterial,
      material.shininess,
      ...vector2SignatureValues(material.normalScale),
      material.isMeshDepthMaterial,
      material.depthPacking,
      material.isMeshDistanceMaterial,
      material.referencePosition,
      ...vector3SignatureValues(material.referencePosition),
      material.nearDistance,
      material.farDistance,
      ...hintInfo,
    ],
  }
}

export function materialHintSignatureInfo(userData: Record<string, any> | undefined): unknown[] {
  const values: unknown[] = [userData]
  if (userData && typeof userData === 'object' && !Array.isArray(userData)) {
    const hints = userData.headlessThreeRenderer ?? userData.headlessRenderer
    values.push(hints)
    if (hints && typeof hints === 'object' && !Array.isArray(hints)) {
      const hintRecord = hints as Record<string, unknown>
      values.push(
        hintRecord.referencePosition,
        ...vector3SignatureValues(hintRecord.referencePosition),
        hintRecord.distanceReferencePosition,
        ...vector3SignatureValues(hintRecord.distanceReferencePosition),
        hintRecord.nearDistance,
        hintRecord.distanceNear,
        hintRecord.farDistance,
        hintRecord.distanceFar,
      )
    }
  }
  return values
}

export function vector2SignatureValues(value: unknown): unknown[] {
  if (!value || typeof value !== 'object') return [value]
  const arrayLike = value as ArrayLike<unknown>
  return [
    value,
    arrayLike.length,
    arrayLike[0],
    arrayLike[1],
    (value as { x?: unknown }).x,
    (value as { y?: unknown }).y,
  ]
}

export function vector3SignatureValues(value: unknown): unknown[] {
  if (!value || typeof value !== 'object') return [value]
  const arrayLike = value as ArrayLike<unknown>
  return [
    value,
    arrayLike.length,
    arrayLike[0],
    arrayLike[1],
    arrayLike[2],
    (value as { x?: unknown }).x,
    (value as { y?: unknown }).y,
    (value as { z?: unknown }).z,
  ]
}

export function copyMaterialScalarFeatureProperties(props: PbrProperties): PbrProperties {
  return {
    ...props,
    clearcoatNormalScale: props.clearcoatNormalScale ? props.clearcoatNormalScale.slice() : undefined,
    normalScale: props.normalScale ? props.normalScale.slice() : undefined,
    distanceReferencePosition: props.distanceReferencePosition ? props.distanceReferencePosition.slice() : undefined,
  }
}

export function sameMaterialScalarFeatureSignature(
  a: MaterialScalarFeatureSignature,
  b: MaterialScalarFeatureSignature,
): boolean {
  return sameUnknownArray(a.values, b.values)
}

export interface CachedMaterialRenderStateExtraction {
  signature: MaterialRenderStateSignature
  props: PbrProperties
}

export interface MaterialRenderStateSignature {
  values: unknown[]
  blendColor: MaterialColorSignature
}

export function materialRenderStateProperties(
  material: ThreeMaterialLike,
  customFragmentShader: string | undefined,
  context: MaterialExtractionContext,
): PbrProperties {
  const signature = context.materialRenderStateCache
    ? materialRenderStateSignature(material, customFragmentShader)
    : null
  if (signature) {
    const cached = context.materialRenderStateCache?.get(material) as CachedMaterialRenderStateExtraction | undefined
    if (cached && sameMaterialRenderStateSignature(cached.signature, signature)) {
      return copyMaterialRenderStateProperties(cached.props)
    }
  }

  const props: PbrProperties = {}
  const alphaTest = optionalFiniteNumber(material.alphaTest, 'material.alphaTest')
  if (alphaTest !== undefined && alphaTest > 0) {
    props.alphaTest = clamp01(alphaTest)
  }
  if (optionalBoolean(material.alphaHash, 'material.alphaHash') === true) {
    props.alphaHash = true
  }
  if (optionalBoolean(material.alphaToCoverage, 'material.alphaToCoverage') === true) {
    props.alphaToCoverage = true
  }
  if (optionalBoolean(material.premultipliedAlpha, 'material.premultipliedAlpha') === true) {
    props.premultipliedAlpha = true
  }
  if (optionalBoolean(material.toneMapped, 'material.toneMapped') === false) {
    props.toneMapped = false
  }
  optionalBoolean(material.dithering, 'material.dithering')
  optionalMaterialPrecision(material.precision)
  const transparent = optionalBoolean(material.transparent, 'material.transparent')
  if (transparent !== undefined) {
    props.transparent = transparent
  }
  optionalBoolean(material.forceSinglePass, 'material.forceSinglePass')
  const blending = materialBlending(material)
  if (blending) {
    props.blending = blending
    if (blending === 'custom') {
      props.blendEquation = materialBlendEquationOrDefault(material.blendEquation, 'material.blendEquation', AddEquation)
      props.blendSrc = materialBlendFactorOrDefault(material.blendSrc, 'material.blendSrc', SrcAlphaFactor)
      props.blendDst = materialBlendFactorOrDefault(material.blendDst, 'material.blendDst', OneMinusSrcAlphaFactor)
      if (material.blendEquationAlpha != null) {
        props.blendEquationAlpha = materialBlendEquation(material.blendEquationAlpha, 'material.blendEquationAlpha')
      }
      if (material.blendSrcAlpha != null) {
        props.blendSrcAlpha = materialBlendFactor(material.blendSrcAlpha, 'material.blendSrcAlpha')
      }
      if (material.blendDstAlpha != null) {
        props.blendDstAlpha = materialBlendFactor(material.blendDstAlpha, 'material.blendDstAlpha')
      }
      const blendColor = materialSlotColor(material, 'blendColor', material.blendColor, 'material.blendColor', context)
      if (blendColor) {
        props.blendColor = [blendColor[0], blendColor[1], blendColor[2]]
      }
      const blendAlpha = optionalFiniteNumber(material.blendAlpha, 'material.blendAlpha')
      if (blendAlpha !== undefined) {
        props.blendAlpha = clamp01(blendAlpha)
      }
    }
  }
  const depthTest = optionalBoolean(material.depthTest, 'material.depthTest')
  if (depthTest !== undefined) {
    props.depthTest = depthTest
  }
  const depthFunc = materialDepthFunc(material)
  if (depthFunc) {
    props.depthFunc = depthFunc
  }
  const depthWrite = optionalBoolean(material.depthWrite, 'material.depthWrite')
  if (depthWrite !== undefined) {
    props.depthWrite = depthWrite
  }
  const colorWrite = optionalBoolean(material.colorWrite, 'material.colorWrite')
  if (colorWrite !== undefined) {
    props.colorWrite = colorWrite
  }
  const polygonOffset = optionalBoolean(material.polygonOffset, 'material.polygonOffset')
  if (polygonOffset !== undefined) {
    props.polygonOffset = polygonOffset
    const polygonOffsetFactor = optionalFiniteNumber(material.polygonOffsetFactor, 'material.polygonOffsetFactor')
    if (polygonOffsetFactor !== undefined) {
      props.polygonOffsetFactor = polygonOffsetFactor
    }
    const polygonOffsetUnits = optionalFiniteNumber(material.polygonOffsetUnits, 'material.polygonOffsetUnits')
    if (polygonOffsetUnits !== undefined) {
      props.polygonOffsetUnits = polygonOffsetUnits
    }
  }
  const stencilWrite = optionalBoolean(material.stencilWrite, 'material.stencilWrite')
  if (stencilWrite !== undefined) {
    props.stencilWrite = stencilWrite
  }
  const stencilWriteMask = optionalFiniteNumber(material.stencilWriteMask, 'material.stencilWriteMask')
  if (stencilWriteMask !== undefined) {
    props.stencilWriteMask = Math.trunc(stencilWriteMask)
  }
  if (material.stencilFunc != null) {
    props.stencilFunc = materialStencilFunc(material.stencilFunc, 'material.stencilFunc')
  }
  const stencilRef = optionalFiniteNumber(material.stencilRef, 'material.stencilRef')
  if (stencilRef !== undefined) {
    props.stencilRef = Math.trunc(stencilRef)
  }
  const stencilFuncMask = optionalFiniteNumber(material.stencilFuncMask, 'material.stencilFuncMask')
  if (stencilFuncMask !== undefined) {
    props.stencilFuncMask = Math.trunc(stencilFuncMask)
  }
  if (material.stencilFail != null) {
    props.stencilFail = materialStencilOperation(material.stencilFail, 'material.stencilFail')
  }
  if (material.stencilZFail != null) {
    props.stencilZFail = materialStencilOperation(material.stencilZFail, 'material.stencilZFail')
  }
  if (material.stencilZPass != null) {
    props.stencilZPass = materialStencilOperation(material.stencilZPass, 'material.stencilZPass')
  }
  const side = materialSide(material)
  if (side) {
    props.side = side
  }
  const shadowSide = materialShadowSide(material)
  if (shadowSide) {
    props.shadowSide = shadowSide
  }
  if (optionalBoolean(material.flatShading, 'material.flatShading') === true) {
    props.flatShading = true
  }
  if (optionalBoolean(material.fog, 'material.fog') === false) {
    props.fog = false
  }

  if (customFragmentShader && shaderMaterialKind(material)) {
    props.shadingModel = 'basic'
  } else if (copyShaderMaterialInfo(material)) {
    props.shadingModel = 'basic'
  } else if (material.isMeshBasicMaterial || material.isSpriteMaterial) {
    props.shadingModel = 'basic'
  } else if (material.isMeshDepthMaterial) {
    props.shadingModel = 'depth'
  } else if (material.isMeshDistanceMaterial) {
    props.shadingModel = 'distance'
  } else if (material.isMeshLambertMaterial) {
    props.shadingModel = 'lambert'
  } else if (material.isMeshNormalMaterial) {
    props.shadingModel = 'normal'
  } else if (material.isMeshMatcapMaterial) {
    props.shadingModel = 'matcap'
  } else if (material.isMeshPhongMaterial) {
    props.shadingModel = 'phong'
  } else if (material.isMeshToonMaterial) {
    props.shadingModel = 'toon'
  } else if (material.isShadowMaterial) {
    props.shadingModel = 'shadow'
  }

  if (customFragmentShader) {
    props.customFragmentShader = customFragmentShader
  }

  if (signature) {
    context.materialRenderStateCache?.set(material, {
      signature,
      props: copyMaterialRenderStateProperties(props),
    })
  }
  return props
}

export function materialRenderStateSignature(
  material: ThreeMaterialLike,
  customFragmentShader: string | undefined,
): MaterialRenderStateSignature {
  return {
    values: [
      customFragmentShader,
      material.alphaTest,
      material.alphaHash,
      material.alphaToCoverage,
      material.premultipliedAlpha,
      material.toneMapped,
      material.dithering,
      material.precision,
      material.transparent,
      material.forceSinglePass,
      material.blending,
      material.blendEquation,
      material.blendSrc,
      material.blendDst,
      material.blendEquationAlpha,
      material.blendSrcAlpha,
      material.blendDstAlpha,
      material.blendAlpha,
      material.depthTest,
      material.depthFunc,
      material.depthWrite,
      material.colorWrite,
      material.polygonOffset,
      material.polygonOffsetFactor,
      material.polygonOffsetUnits,
      material.stencilWrite,
      material.stencilWriteMask,
      material.stencilFunc,
      material.stencilRef,
      material.stencilFuncMask,
      material.stencilFail,
      material.stencilZFail,
      material.stencilZPass,
      material.side,
      material.shadowSide,
      material.flatShading,
      material.fog,
      material.type,
      copyShaderMaterialInfo(material) != null,
      material.isRawShaderMaterial,
      material.isNodeMaterial,
      material.isShaderMaterial,
      material.isMeshBasicMaterial,
      material.isSpriteMaterial,
      material.isMeshDepthMaterial,
      material.isMeshDistanceMaterial,
      material.isMeshLambertMaterial,
      material.isMeshNormalMaterial,
      material.isMeshMatcapMaterial,
      material.isMeshPhongMaterial,
      material.isMeshToonMaterial,
      material.isShadowMaterial,
    ],
    blendColor: materialSlotColorSignature(material.blendColor),
  }
}

export function copyMaterialRenderStateProperties(props: PbrProperties): PbrProperties {
  return {
    ...props,
    blendColor: props.blendColor ? props.blendColor.slice() : undefined,
  }
}

export function sameMaterialRenderStateSignature(
  a: MaterialRenderStateSignature,
  b: MaterialRenderStateSignature,
): boolean {
  return sameUnknownArray(a.values, b.values)
    && sameMaterialColorSignature(a.blendColor, b.blendColor)
}

export function materialBlending(material: ThreeMaterialLike): string | undefined {
  switch (material.blending) {
    case NoBlending:
      return 'none'
    case NormalBlending:
      return 'normal'
    case AdditiveBlending:
      return 'additive'
    case SubtractiveBlending:
      return 'subtractive'
    case MultiplyBlending:
      return 'multiply'
    case CustomBlending:
      return 'custom'
    default:
      if (material.blending != null) {
        throw new Error(
          `material.blending ${String(material.blending)} is not supported by @headless-three/renderer. Use a Three.js blending constant such as NormalBlending, AdditiveBlending, or CustomBlending.`,
        )
      }
      return undefined
  }
}

export function materialBlendEquationOrDefault(value: unknown, label: string, fallback: number): number {
  if (value == null) return fallback
  return materialBlendEquation(value, label)
}

export function materialBlendEquation(value: unknown, label: string): number {
  if (
    value === AddEquation ||
    value === SubtractEquation ||
    value === ReverseSubtractEquation ||
    value === MinEquation ||
    value === MaxEquation
  ) {
    return value
  }
  throw new Error(
    `${label} ${String(value)} is not supported by @headless-three/renderer. Use a Three.js blend equation constant such as AddEquation, SubtractEquation, ReverseSubtractEquation, MinEquation, or MaxEquation.`,
  )
}

export function materialBlendFactorOrDefault(value: unknown, label: string, fallback: number): number {
  if (value == null) return fallback
  return materialBlendFactor(value, label)
}

export function materialBlendFactor(value: unknown, label: string): number {
  if (
    value === ZeroFactor ||
    value === OneFactor ||
    value === SrcColorFactor ||
    value === OneMinusSrcColorFactor ||
    value === SrcAlphaFactor ||
    value === OneMinusSrcAlphaFactor ||
    value === DstAlphaFactor ||
    value === OneMinusDstAlphaFactor ||
    value === DstColorFactor ||
    value === OneMinusDstColorFactor ||
    value === SrcAlphaSaturateFactor ||
    value === ConstantColorFactor ||
    value === OneMinusConstantColorFactor ||
    value === ConstantAlphaFactor ||
    value === OneMinusConstantAlphaFactor
  ) {
    return value
  }
  throw new Error(
    `${label} ${String(value)} is not supported by @headless-three/renderer. Use a Three.js blend factor constant such as SrcAlphaFactor, OneMinusSrcAlphaFactor, OneFactor, or ZeroFactor.`,
  )
}

export function materialDepthFunc(material: ThreeMaterialLike): string | undefined {
  if (material.depthFunc == null) return undefined
  switch (material.depthFunc) {
    case NeverDepth:
      return 'never'
    case AlwaysDepth:
      return 'always'
    case LessDepth:
      return 'less'
    case LessEqualDepth:
      return 'less-equal'
    case EqualDepth:
      return 'equal'
    case GreaterEqualDepth:
      return 'greater-equal'
    case GreaterDepth:
      return 'greater'
    case NotEqualDepth:
      return 'not-equal'
    default:
      throw new Error(
        `material.depthFunc ${String(material.depthFunc)} is not supported by @headless-three/renderer. Use a Three.js depth comparison constant such as LessEqualDepth, GreaterDepth, or AlwaysDepth.`,
      )
  }
}

export function materialStencilFunc(value: unknown, label: string): number {
  if (
    value === NeverStencilFunc ||
    value === LessStencilFunc ||
    value === EqualStencilFunc ||
    value === LessEqualStencilFunc ||
    value === GreaterStencilFunc ||
    value === NotEqualStencilFunc ||
    value === GreaterEqualStencilFunc ||
    value === AlwaysStencilFunc
  ) {
    return value
  }
  throw new Error(
    `${label} ${String(value)} is not supported by @headless-three/renderer. Use a Three.js stencil function constant such as AlwaysStencilFunc, EqualStencilFunc, or NotEqualStencilFunc.`,
  )
}
