import type { Color4, ThreeMaterialLike, PbrProperties, TextureInfo, ThreeTextureLike, ThreeSceneRootLike, ThreeObject3DLike } from './types'
import { clamp01 } from './math'
import { strictColorLikeToArray, validatedColorLikeToArray } from './color'
import { objectChildren } from './objects'
import { isThreeCubeTexturePassShaderMaterial, isThreeGtaoPassShaderMaterial, isThreeLightProbeHelperShaderMaterial, isThreeLineMaterialShaderMaterial, isThreeOutlineEffectShaderMaterial, isThreeOutlinePassPrepareMaskShaderMaterial, isThreeReflectorForSsrPassShaderMaterial, isThreeReflectorShaderMaterial, isThreeRefractorShaderMaterial, isThreeSaoPassShaderMaterial, isThreeShadowMapViewerShaderMaterial, isThreeSkyShaderMaterial, isThreeSmaaPassEdgesShaderMaterial, isThreeSsaoPassShaderMaterial, isThreeSsrPassShaderMaterial, isThreeTextureHelperShaderMaterial, isThreeUnrealBloomPassHighPassShaderMaterial, isThreeWater2ShaderMaterial, isThreeWaterMirrorShaderMaterial, namedShaderMaterialLabel } from './materials.part-007'
import { copyShaderMaterialInfo, shaderMaterialKind } from './materials.part-008'
export function assertSupportedShaderMaterial(
  material: ThreeMaterialLike,
  customFragmentShader: string | undefined,
): void {
  const kind = shaderMaterialKind(material)
  if (!kind || customFragmentShader || copyShaderMaterialInfo(material)) return

  if (isThreePmremShaderMaterial(material)) {
    throw new Error(
      `${material.name} is a Three.js PMREMGenerator internal ShaderMaterial and is not translated by @headless-three/renderer yet. Use readable scene.environment, scene.background, material.envMap, or reflection-probe textures directly so the renderer can run its native CPU IBL precompute path, or precompute PMREM/CubeUV assets before rendering.`,
    )
  }

  if (isThreeAnaglyphEffectShaderMaterial(material)) {
    throw new Error(
      'THREE.AnaglyphEffect internal ShaderMaterial is not translated by @headless-three/renderer yet. Use StereoEffect or PeppersGhostEffect for covered stereo helper renders, or render the left and right eye targets separately and compose the anaglyph image outside this helper.',
    )
  }

  if (isThreeParallaxBarrierEffectShaderMaterial(material)) {
    throw new Error(
      'THREE.ParallaxBarrierEffect internal ShaderMaterial is not translated by @headless-three/renderer yet. Use StereoEffect or PeppersGhostEffect for covered stereo helper renders, or render the left and right eye targets separately and compose the parallax-barrier image outside this helper.',
    )
  }

  if (isThreeAfterimagePassShaderMaterial(material)) {
    throw new Error(
      'THREE.AfterimagePass internal ShaderMaterial is not translated by @headless-three/renderer yet. Use the covered CopyShader/OutputPass fullscreen helpers for simple copies, provide a custom WGSL fragment for an equivalent pass, or compose the afterimage effect outside this helper.',
    )
  }

  if (isThreeBloomPassConvolutionShaderMaterial(material)) {
    throw new Error(
      "THREE.BloomPass internal convolution ShaderMaterial is not translated by @headless-three/renderer yet. Use the renderer's postProcessing controls for supported image effects, provide a custom WGSL fragment for an equivalent blur/composite pass, or compose bloom outside this helper.",
    )
  }

  if (isThreeFilmPassShaderMaterial(material)) {
    throw new Error(
      "THREE.FilmPass internal FilmShader ShaderMaterial is not translated by @headless-three/renderer yet. Use the renderer's postProcessing controls for supported image effects, provide a custom WGSL fragment for an equivalent film/noise pass, or compose the film effect outside this helper.",
    )
  }

  if (isThreeDotScreenPassShaderMaterial(material)) {
    throw new Error(
      "THREE.DotScreenPass internal DotScreenShader ShaderMaterial is not translated by @headless-three/renderer yet. Use the renderer's postProcessing controls for supported image effects, provide a custom WGSL fragment for an equivalent dot-screen pass, or compose the dot-screen effect outside this helper.",
    )
  }

  if (isThreeGlitchPassShaderMaterial(material)) {
    throw new Error(
      "THREE.GlitchPass internal DigitalGlitch ShaderMaterial is not translated by @headless-three/renderer yet. Use the renderer's postProcessing controls for supported image effects, provide a custom WGSL fragment for an equivalent glitch pass, or compose the glitch effect outside this helper.",
    )
  }

  if (isThreeHalftonePassShaderMaterial(material)) {
    throw new Error(
      "THREE.HalftonePass internal HalftoneShader ShaderMaterial is not translated by @headless-three/renderer yet. Use the renderer's postProcessing controls for supported image effects, provide a custom WGSL fragment for an equivalent halftone pass, or compose the halftone effect outside this helper.",
    )
  }

  if (isThreeLutPassShaderMaterial(material)) {
    throw new Error(
      "THREE.LUTPass internal LUTShader ShaderMaterial is not translated by @headless-three/renderer yet. Use the renderer's postProcessing controls for supported color adjustments, bake color grading into input textures, provide a custom WGSL fragment for an equivalent LUT pass, or compose LUT color grading outside this helper.",
    )
  }

  if (isThreeBokehPassShaderMaterial(material)) {
    throw new Error(
      "THREE.BokehPass internal BokehShader ShaderMaterial is not translated by @headless-three/renderer yet. Use the renderer's postProcessing controls for supported image effects, provide a custom WGSL fragment for an equivalent depth-of-field pass, or compose the bokeh effect outside this helper.",
    )
  }

  if (isThreeRenderPixelatedPassShaderMaterial(material)) {
    throw new Error(
      "THREE.RenderPixelatedPass internal pixelation ShaderMaterial is not translated by @headless-three/renderer yet. Use options.renderMode for supported depth/normal auxiliary outputs, provide a custom WGSL fragment for an equivalent pixelated composite pass, or compose the pixelated effect outside this helper.",
    )
  }

  if (isThreeRenderTransitionPassShaderMaterial(material)) {
    throw new Error(
      "THREE.RenderTransitionPass internal transition ShaderMaterial is not translated by @headless-three/renderer yet. Render the two scenes separately and blend them outside this helper, or provide a custom WGSL fragment for an equivalent transition pass.",
    )
  }

  if (isThreeCubeTexturePassShaderMaterial(material)) {
    throw new Error(
      "THREE.CubeTexturePass internal cube ShaderMaterial is not translated by @headless-three/renderer yet. Use scene.background, options.background, or a supported cube background texture directly, or provide a custom WGSL fragment for an equivalent cube-texture pass.",
    )
  }

  if (isThreeSaoPassShaderMaterial(material)) {
    throw new Error(
      "THREE.SAOPass internal SAOShader ShaderMaterial is not translated by @headless-three/renderer yet. Use the renderer's postProcessing controls for supported image effects, provide a custom WGSL fragment for an equivalent ambient-occlusion pass, or compose the SAO effect outside this helper.",
    )
  }

  if (isThreeSsaoPassShaderMaterial(material)) {
    throw new Error(
      "THREE.SSAOPass internal SSAOShader ShaderMaterial is not translated by @headless-three/renderer yet. Use the renderer's postProcessing controls for supported image effects, provide a custom WGSL fragment for an equivalent screen-space ambient-occlusion pass, or compose the SSAO effect outside this helper.",
    )
  }

  if (isThreeGtaoPassShaderMaterial(material)) {
    throw new Error(
      "THREE.GTAOPass internal GTAOShader ShaderMaterial is not translated by @headless-three/renderer yet. Use the renderer's postProcessing controls for supported image effects, provide a custom WGSL fragment for an equivalent ground-truth ambient-occlusion pass, or compose the GTAO effect outside this helper.",
    )
  }

  if (isThreeUnrealBloomPassHighPassShaderMaterial(material)) {
    throw new Error(
      "THREE.UnrealBloomPass internal LuminosityHighPassShader ShaderMaterial is not translated by @headless-three/renderer yet. Use the renderer's postProcessing controls for supported image effects, provide a custom WGSL fragment for an equivalent bloom pass, or compose UnrealBloom outside this helper.",
    )
  }

  if (isThreeSsrPassShaderMaterial(material)) {
    throw new Error(
      "THREE.SSRPass internal SSRShader ShaderMaterial is not translated by @headless-three/renderer yet. Render reflections through supported scene/environment/material inputs, provide a custom WGSL fragment for an equivalent screen-space reflection pass, or compose SSR outside this helper.",
    )
  }

  if (isThreeSmaaPassEdgesShaderMaterial(material)) {
    throw new Error(
      "THREE.SMAAPass internal SMAAEdgesShader ShaderMaterial is not translated by @headless-three/renderer yet. Use native MSAA through render options or targets when applicable, provide a custom WGSL fragment for an equivalent antialiasing pass, or compose SMAA outside this helper.",
    )
  }

  if (isThreeOutlinePassPrepareMaskShaderMaterial(material)) {
    throw new Error(
      "THREE.OutlinePass internal prepare-mask ShaderMaterial is not translated by @headless-three/renderer yet. Use options.renderMode mask/object-id outputs for supported object isolation, provide a custom WGSL fragment for an equivalent outline pass, or compose outlines outside this helper.",
    )
  }

  if (isThreeOutlineEffectShaderMaterial(material)) {
    throw new Error(
      "THREE.OutlineEffect internal outline ShaderMaterial is not translated by @headless-three/renderer yet. Use options.renderMode mask/object-id outputs for supported object isolation, provide a custom WGSL fragment for an equivalent outline material, or compose outlines outside this helper.",
    )
  }

  if (isThreeSkyShaderMaterial(material)) {
    throw new Error(
      "THREE.Sky internal SkyShader ShaderMaterial is not translated by @headless-three/renderer yet. Use supported scene.background/environment textures or colors, pre-render a sky texture before rendering, or provide a custom WGSL fragment for an equivalent sky material.",
    )
  }

  if (isThreeWaterMirrorShaderMaterial(material)) {
    throw new Error(
      "THREE.Water internal MirrorShader ShaderMaterial is not translated by @headless-three/renderer yet. Use supported scene/environment/material inputs, render reflection targets separately, provide a custom WGSL fragment for an equivalent water material, or compose water outside this helper.",
    )
  }

  if (isThreeWater2ShaderMaterial(material)) {
    throw new Error(
      "THREE.Water2 internal WaterShader ShaderMaterial is not translated by @headless-three/renderer yet. Use supported scene/environment/material inputs, render reflection/refraction targets separately, provide a custom WGSL fragment for an equivalent flow-water material, or compose water outside this helper.",
    )
  }

  if (isThreeReflectorForSsrPassShaderMaterial(material)) {
    throw new Error(
      "THREE.ReflectorForSSRPass internal ReflectorShader ShaderMaterial is not translated by @headless-three/renderer yet. Use supported scene/environment/material inputs, render reflector targets separately, provide a custom WGSL fragment for an equivalent SSR reflector material, or compose SSR reflections outside this helper.",
    )
  }

  if (isThreeReflectorShaderMaterial(material)) {
    throw new Error(
      "THREE.Reflector internal ReflectorShader ShaderMaterial is not translated by @headless-three/renderer yet. The helper-owned target prepass is covered, but the final projective reflection material still needs a custom WGSL fragment or an external composition step.",
    )
  }

  if (isThreeRefractorShaderMaterial(material)) {
    throw new Error(
      "THREE.Refractor internal RefractorShader ShaderMaterial is not translated by @headless-three/renderer yet. The helper-owned target prepass is covered, but the final projective refraction material still needs a custom WGSL fragment or an external composition step.",
    )
  }

  if (isThreeShadowMapViewerShaderMaterial(material)) {
    throw new Error(
      "THREE.ShadowMapViewer internal UnpackDepthRGBAShader ShaderMaterial is not translated by @headless-three/renderer yet. Use supported shadow/depth render-target readback paths, provide a custom WGSL fragment for an equivalent depth visualization, or compose the shadow-map preview outside this helper.",
    )
  }

  if (isThreeLightProbeHelperShaderMaterial(material)) {
    throw new Error(
      'THREE.LightProbeHelper internal LightProbeHelperMaterial ShaderMaterial is not translated by @headless-three/renderer yet. Native THREE.LightProbe lighting and LightProbeGenerator cube-target readback are supported directly; use a built-in material, provide a custom WGSL fragment for an equivalent probe visualization, or inspect spherical-harmonics coefficients outside this helper.',
    )
  }

  if (isThreeTextureHelperShaderMaterial(material)) {
    throw new Error(
      'THREE.TextureHelper internal TextureHelperMaterial ShaderMaterial is not translated by @headless-three/renderer yet. Use supported material, background, scene.environment, or render-target texture inputs directly, read or copy renderer-owned target pixels, or provide a custom WGSL fragment for an equivalent texture visualizer.',
    )
  }

  if (isThreeLineMaterialShaderMaterial(material)) {
    throw new Error(
      'THREE.LineMaterial ShaderMaterial used by Line2, LineSegments2, and Wireframe is not translated by @headless-three/renderer yet. Use THREE.Line, THREE.LineSegments, or THREE.LineLoop with LineBasicMaterial or LineDashedMaterial for covered CPU-expanded line rendering, or provide a custom WGSL fragment for an equivalent line material.',
    )
  }

  const label = namedShaderMaterialLabel(kind, material)
  throw new Error(
    `${label} is not supported directly by @headless-three/renderer. Use a built-in Three.js material, or provide material.userData.headlessThreeRenderer.fragmentWgsl with a WGSL fragment body for the renderer's custom material path.`,
  )
}

export function isThreePmremShaderMaterial(material: ThreeMaterialLike): material is ThreeMaterialLike & { name: string } {
  return material.name === 'EquirectangularToCubeUV' ||
    material.name === 'CubemapToCubeUV' ||
    material.name === 'SphericalGaussianBlur'
}

export function isThreeAnaglyphEffectShaderMaterial(material: ThreeMaterialLike): boolean {
  const uniforms = material.uniforms
  if (!uniforms || typeof uniforms !== 'object' || Array.isArray(uniforms)) return false

  const values = uniforms as Record<string, unknown>
  if (
    values.mapLeft == null ||
    values.mapRight == null ||
    values.colorMatrixLeft == null ||
    values.colorMatrixRight == null
  ) {
    return false
  }

  if (typeof material.fragmentShader !== 'string') return false
  const compact = material.fragmentShader.replace(/\s+/g, '')
  return compact.includes('uniformsampler2DmapLeft;') &&
    compact.includes('uniformsampler2DmapRight;') &&
    compact.includes('uniformmat3colorMatrixLeft;') &&
    compact.includes('uniformmat3colorMatrixRight;') &&
    compact.includes('colorMatrixLeft*colorL.rgb+') &&
    compact.includes('colorMatrixRight*colorR.rgb')
}

export function isThreeParallaxBarrierEffectShaderMaterial(material: ThreeMaterialLike): boolean {
  const uniforms = material.uniforms
  if (!uniforms || typeof uniforms !== 'object' || Array.isArray(uniforms)) return false

  const values = uniforms as Record<string, unknown>
  if (values.mapLeft == null || values.mapRight == null) return false

  if (typeof material.fragmentShader !== 'string') return false
  const compact = material.fragmentShader.replace(/\s+/g, '')
  return compact.includes('uniformsampler2DmapLeft;') &&
    compact.includes('uniformsampler2DmapRight;') &&
    compact.includes('mod(gl_FragCoord.y,2.0)') &&
    compact.includes('gl_FragColor=texture2D(mapLeft,uv);') &&
    compact.includes('gl_FragColor=texture2D(mapRight,uv);')
}

export function isThreeAfterimagePassShaderMaterial(material: ThreeMaterialLike): boolean {
  const uniforms = material.uniforms
  if (!uniforms || typeof uniforms !== 'object' || Array.isArray(uniforms)) return false

  const values = uniforms as Record<string, unknown>
  if (values.damp == null || values.tOld == null || values.tNew == null) return false

  if (typeof material.fragmentShader !== 'string') return false
  const compact = material.fragmentShader.replace(/\s+/g, '')
  return compact.includes('uniformfloatdamp;') &&
    compact.includes('uniformsampler2DtOld;') &&
    compact.includes('uniformsampler2DtNew;') &&
    compact.includes('texelOld*=damp*when_gt(texelOld,0.1);') &&
    compact.includes('gl_FragColor=max(texelNew,texelOld);')
}

export function isThreeBloomPassConvolutionShaderMaterial(material: ThreeMaterialLike): boolean {
  const uniforms = material.uniforms
  if (!uniforms || typeof uniforms !== 'object' || Array.isArray(uniforms)) return false

  const values = uniforms as Record<string, unknown>
  if (values.tDiffuse == null || values.uImageIncrement == null || values.cKernel == null) return false

  if (typeof material.fragmentShader !== 'string') return false
  const compact = material.fragmentShader.replace(/\s+/g, '')
  return compact.includes('uniformfloatcKernel[KERNEL_SIZE_INT];') &&
    compact.includes('uniformsampler2DtDiffuse;') &&
    compact.includes('uniformvec2uImageIncrement;') &&
    compact.includes('sum+=texture2D(tDiffuse,imageCoord)*cKernel[i];') &&
    compact.includes('imageCoord+=uImageIncrement;') &&
    compact.includes('gl_FragColor=sum;')
}

export function isThreeFilmPassShaderMaterial(material: ThreeMaterialLike): boolean {
  const uniforms = material.uniforms
  if (!uniforms || typeof uniforms !== 'object' || Array.isArray(uniforms)) return false

  const values = uniforms as Record<string, unknown>
  if (values.tDiffuse == null || values.time == null || values.intensity == null || values.grayscale == null) {
    return false
  }

  if (typeof material.fragmentShader !== 'string') return false
  const compact = material.fragmentShader.replace(/\s+/g, '')
  return compact.includes('uniformfloatintensity;') &&
    compact.includes('uniformboolgrayscale;') &&
    compact.includes('uniformfloattime;') &&
    compact.includes('floatnoise=rand(fract(vUv+time));') &&
    compact.includes('color=mix(base.rgb,color,intensity);') &&
    compact.includes('gl_FragColor=vec4(color,base.a);')
}

export function isThreeDotScreenPassShaderMaterial(material: ThreeMaterialLike): boolean {
  const uniforms = material.uniforms
  if (!uniforms || typeof uniforms !== 'object' || Array.isArray(uniforms)) return false

  const values = uniforms as Record<string, unknown>
  if (
    values.tDiffuse == null ||
    values.tSize == null ||
    values.center == null ||
    values.angle == null ||
    values.scale == null
  ) {
    return false
  }

  if (typeof material.fragmentShader !== 'string') return false
  const compact = material.fragmentShader.replace(/\s+/g, '')
  return compact.includes('uniformvec2center;') &&
    compact.includes('uniformfloatscale;') &&
    compact.includes('uniformvec2tSize;') &&
    compact.includes('floatpattern()') &&
    compact.includes('vec2tex=vUv*tSize-center;') &&
    compact.includes('gl_FragColor=vec4(vec3(average*10.0-5.0+pattern()),color.a);')
}

export function isThreeGlitchPassShaderMaterial(material: ThreeMaterialLike): boolean {
  const uniforms = material.uniforms
  if (!uniforms || typeof uniforms !== 'object' || Array.isArray(uniforms)) return false

  const values = uniforms as Record<string, unknown>
  if (
    values.tDiffuse == null ||
    values.tDisp == null ||
    values.byp == null ||
    values.amount == null ||
    values.angle == null ||
    values.seed == null ||
    values.seed_x == null ||
    values.seed_y == null ||
    values.distortion_x == null ||
    values.distortion_y == null ||
    values.col_s == null
  ) {
    return false
  }

  if (typeof material.fragmentShader !== 'string') return false
  const compact = material.fragmentShader.replace(/\s+/g, '')
  return compact.includes('uniformintbyp;') &&
    compact.includes('uniformsampler2DtDisp;') &&
    compact.includes('uniformfloatdistortion_x;') &&
    compact.includes('floatdisp=texture2D(tDisp,p*seed*seed).r;') &&
    compact.includes('vec4cr=texture2D(tDiffuse,p+offset);') &&
    compact.includes('gl_FragColor=vec4(cr.r,cga.g,cb.b,cga.a);')
}

export function isThreeHalftonePassShaderMaterial(material: ThreeMaterialLike): boolean {
  const uniforms = material.uniforms
  if (!uniforms || typeof uniforms !== 'object' || Array.isArray(uniforms)) return false

  const values = uniforms as Record<string, unknown>
  if (
    values.tDiffuse == null ||
    values.shape == null ||
    values.radius == null ||
    values.rotateR == null ||
    values.rotateG == null ||
    values.rotateB == null ||
    values.scatter == null ||
    values.width == null ||
    values.height == null ||
    values.blending == null ||
    values.blendingMode == null ||
    values.greyscale == null ||
    values.disable == null
  ) {
    return false
  }

  if (typeof material.fragmentShader !== 'string') return false
  const compact = material.fragmentShader.replace(/\s+/g, '')
  return compact.includes('uniformfloatradius;') &&
    compact.includes('uniformfloatrotateR;') &&
    compact.includes('uniformintshape;') &&
    compact.includes('structCell{') &&
    compact.includes('vec4getSample(vec2point)') &&
    compact.includes('CellgetReferenceCell(vec2p,vec2origin,floatgrid_angle,floatstep)') &&
    compact.includes('gl_FragColor=vec4(r,g,b,1.0);')
}

export function isThreeLutPassShaderMaterial(material: ThreeMaterialLike): boolean {
  const uniforms = material.uniforms
  if (!uniforms || typeof uniforms !== 'object' || Array.isArray(uniforms)) return false

  const values = uniforms as Record<string, unknown>
  if (values.lut == null || values.lutSize == null || values.tDiffuse == null || values.intensity == null) {
    return false
  }

  if (typeof material.fragmentShader !== 'string') return false
  const compact = material.fragmentShader.replace(/\s+/g, '')
  return compact.includes('uniformfloatlutSize;') &&
    compact.includes('uniformsampler3Dlut;') &&
    compact.includes('uniformfloatintensity;') &&
    compact.includes('uniformsampler2DtDiffuse;') &&
    compact.includes('vec4val=texture2D(tDiffuse,vUv);') &&
    compact.includes('floatpixelWidth=1.0/lutSize;') &&
    compact.includes('vec3uvw=vec3(halfPixelWidth)+val.rgb*(1.0-pixelWidth);') &&
    compact.includes('lutVal=vec4(texture(lut,uvw).rgb,val.a);') &&
    compact.includes('gl_FragColor=vec4(mix(val,lutVal,intensity));')
}

export function isThreeBokehPassShaderMaterial(material: ThreeMaterialLike): boolean {
  const uniforms = material.uniforms
  if (!uniforms || typeof uniforms !== 'object' || Array.isArray(uniforms)) return false

  const values = uniforms as Record<string, unknown>
  if (
    values.tColor == null ||
    values.tDepth == null ||
    values.focus == null ||
    values.aspect == null ||
    values.aperture == null ||
    values.maxblur == null ||
    values.nearClip == null ||
    values.farClip == null
  ) {
    return false
  }

  if (typeof material.fragmentShader !== 'string') return false
  const compact = material.fragmentShader.replace(/\s+/g, '')
  return compact.includes('uniformsampler2DtColor;') &&
    compact.includes('uniformsampler2DtDepth;') &&
    compact.includes('uniformfloataperture;') &&
    compact.includes('uniformfloatnearClip;') &&
    compact.includes('floatviewZ=getViewZ(getDepth(vUv));') &&
    compact.includes('vec2dofblur=vec2(clamp(factor*aperture,-maxblur,maxblur));') &&
    compact.includes('gl_FragColor=col/41.0;')
}

export function isThreeRenderPixelatedPassShaderMaterial(material: ThreeMaterialLike): boolean {
  const uniforms = material.uniforms
  if (!uniforms || typeof uniforms !== 'object' || Array.isArray(uniforms)) return false

  const values = uniforms as Record<string, unknown>
  if (
    values.tDiffuse == null ||
    values.tDepth == null ||
    values.tNormal == null ||
    values.resolution == null ||
    values.normalEdgeStrength == null ||
    values.depthEdgeStrength == null
  ) {
    return false
  }

  if (typeof material.fragmentShader !== 'string') return false
  const compact = material.fragmentShader.replace(/\s+/g, '')
  return compact.includes('uniformsampler2DtNormal;') &&
    compact.includes('uniformvec4resolution;') &&
    compact.includes('floatgetDepth(intx,inty)') &&
    compact.includes('vec3getNormal(intx,inty)') &&
    compact.includes('floatdepthEdgeIndicator(floatdepth,vec3normal)') &&
    compact.includes('floatnormalEdgeIndicator(floatdepth,vec3normal)') &&
    compact.includes('gl_FragColor=texel*Strength;')
}

export function isThreeRenderTransitionPassShaderMaterial(material: ThreeMaterialLike): boolean {
  const uniforms = material.uniforms
  if (!uniforms || typeof uniforms !== 'object' || Array.isArray(uniforms)) return false

  const values = uniforms as Record<string, unknown>
  if (
    values.tDiffuse1 == null ||
    values.tDiffuse2 == null ||
    values.mixRatio == null ||
    values.threshold == null ||
    values.useTexture == null ||
    values.tMixTexture == null
  ) {
    return false
  }

  if (typeof material.fragmentShader !== 'string') return false
  const compact = material.fragmentShader.replace(/\s+/g, '')
  return compact.includes('uniformsampler2DtDiffuse1;') &&
    compact.includes('uniformsampler2DtDiffuse2;') &&
    compact.includes('uniformsampler2DtMixTexture;') &&
    compact.includes('uniformfloatmixRatio;') &&
    compact.includes('floatmixf=clamp((transitionTexel.r-r)*(1.0/threshold),0.0,1.0);') &&
    compact.includes('gl_FragColor=mix(texel1,texel2,mixf);') &&
    compact.includes('gl_FragColor=mix(texel2,texel1,mixRatio);')
}
