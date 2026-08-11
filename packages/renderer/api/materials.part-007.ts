import type { Color4, ThreeMaterialLike, PbrProperties, TextureInfo, ThreeTextureLike, ThreeSceneRootLike, ThreeObject3DLike } from './types'
import { clamp01 } from './math'
import { strictColorLikeToArray, validatedColorLikeToArray } from './color'
import { objectChildren } from './objects'
import { activeMaterialNodeHooks, hasCustomOnBeforeCompile, isThreeCsmPatchedMaterial } from './materials.part-008'
export function isThreeCubeTexturePassShaderMaterial(material: ThreeMaterialLike): boolean {
  const uniforms = material.uniforms
  if (!uniforms || typeof uniforms !== 'object' || Array.isArray(uniforms)) return false

  const values = uniforms as Record<string, unknown>
  if (values.tCube == null || values.tFlip == null || values.opacity == null) return false

  if (typeof material.fragmentShader !== 'string') return false
  const compact = material.fragmentShader.replace(/\s+/g, '')
  return compact.includes('uniformsamplerCubetCube;') &&
    compact.includes('uniformfloattFlip;') &&
    compact.includes('uniformfloatopacity;') &&
    compact.includes('textureCube(tCube,vec3(tFlip*vWorldDirection.x,vWorldDirection.yz))') &&
    compact.includes('gl_FragColor.a*=opacity;')
}

export function isThreeSaoPassShaderMaterial(material: ThreeMaterialLike): boolean {
  const uniforms = material.uniforms
  if (!uniforms || typeof uniforms !== 'object' || Array.isArray(uniforms)) return false

  const values = uniforms as Record<string, unknown>
  if (
    values.tDepth == null ||
    values.tNormal == null ||
    values.size == null ||
    values.cameraNear == null ||
    values.cameraFar == null ||
    values.cameraProjectionMatrix == null ||
    values.cameraInverseProjectionMatrix == null ||
    values.scale == null ||
    values.intensity == null ||
    values.bias == null ||
    values.minResolution == null ||
    values.kernelRadius == null ||
    values.randomSeed == null
  ) {
    return false
  }

  if (typeof material.fragmentShader !== 'string') return false
  const compact = material.fragmentShader.replace(/\s+/g, '')
  return compact.includes('uniformhighpsampler2DtDepth;') &&
    compact.includes('uniformhighpsampler2DtNormal;') &&
    compact.includes('uniformfloatkernelRadius;') &&
    compact.includes('floatgetAmbientOcclusion(constinvec3centerViewPosition)') &&
    compact.includes('scaleDividedByCameraFar=scale/cameraFar;') &&
    compact.includes('occlusionSum+=getOcclusion(centerViewPosition,centerViewNormal,sampleViewPosition);') &&
    compact.includes('gl_FragColor.xyz*=1.0-ambientOcclusion;')
}

export function isThreeSsaoPassShaderMaterial(material: ThreeMaterialLike): boolean {
  const uniforms = material.uniforms
  if (!uniforms || typeof uniforms !== 'object' || Array.isArray(uniforms)) return false

  const values = uniforms as Record<string, unknown>
  if (
    values.tNormal == null ||
    values.tDepth == null ||
    values.tNoise == null ||
    values.kernel == null ||
    values.cameraNear == null ||
    values.cameraFar == null ||
    values.resolution == null ||
    values.cameraProjectionMatrix == null ||
    values.cameraInverseProjectionMatrix == null ||
    values.kernelRadius == null ||
    values.minDistance == null ||
    values.maxDistance == null
  ) {
    return false
  }

  if (typeof material.fragmentShader !== 'string') return false
  const compact = material.fragmentShader.replace(/\s+/g, '')
  return compact.includes('uniformhighpsampler2DtNormal;') &&
    compact.includes('uniformhighpsampler2DtDepth;') &&
    compact.includes('uniformsampler2DtNoise;') &&
    compact.includes('uniformvec3kernel[KERNEL_SIZE];') &&
    compact.includes('vec2noiseScale=vec2(resolution.x/4.0,resolution.y/4.0);') &&
    compact.includes('vec3sampleVector=kernelMatrix*kernel[i];') &&
    compact.includes('if(delta>minDistance&&delta<maxDistance)') &&
    compact.includes('gl_FragColor=vec4(vec3(1.0-occlusion),1.0);')
}

export function isThreeGtaoPassShaderMaterial(material: ThreeMaterialLike): boolean {
  const uniforms = material.uniforms
  if (!uniforms || typeof uniforms !== 'object' || Array.isArray(uniforms)) return false

  const values = uniforms as Record<string, unknown>
  if (
    values.tNormal == null ||
    values.tDepth == null ||
    values.tNoise == null ||
    values.resolution == null ||
    values.cameraNear == null ||
    values.cameraFar == null ||
    values.cameraProjectionMatrix == null ||
    values.cameraProjectionMatrixInverse == null ||
    values.cameraWorldMatrix == null ||
    values.radius == null ||
    values.distanceExponent == null ||
    values.thickness == null ||
    values.distanceFallOff == null ||
    values.scale == null
  ) {
    return false
  }

  if (typeof material.fragmentShader !== 'string') return false
  const compact = material.fragmentShader.replace(/\s+/g, '')
  return compact.includes('uniformhighpsampler2DtNormal;') &&
    compact.includes('uniformhighpsampler2DtDepth;') &&
    compact.includes('uniformsampler2DtNoise;') &&
    compact.includes('uniformmat4cameraProjectionMatrixInverse;') &&
    compact.includes('floatgetDepth(constvec2uv)') &&
    compact.includes('returntextureLod(tDepth,uv.xy,0.0).DEPTH_SWIZZLING;') &&
    compact.includes('constintDIRECTIONS=SAMPLES<30?3:5;') &&
    compact.includes('gl_FragColor=FRAGMENT_OUTPUT;')
}

export function isThreeUnrealBloomPassHighPassShaderMaterial(material: ThreeMaterialLike): boolean {
  const uniforms = material.uniforms
  if (!uniforms || typeof uniforms !== 'object' || Array.isArray(uniforms)) return false

  const values = uniforms as Record<string, unknown>
  if (
    values.tDiffuse == null ||
    values.luminosityThreshold == null ||
    values.smoothWidth == null ||
    values.defaultColor == null ||
    values.defaultOpacity == null
  ) {
    return false
  }

  if (typeof material.fragmentShader !== 'string') return false
  const compact = material.fragmentShader.replace(/\s+/g, '')
  return compact.includes('uniformvec3defaultColor;') &&
    compact.includes('uniformfloatdefaultOpacity;') &&
    compact.includes('uniformfloatluminosityThreshold;') &&
    compact.includes('floatv=luminance(texel.xyz);') &&
    compact.includes('vec4outputColor=vec4(defaultColor.rgb,defaultOpacity);') &&
    compact.includes('floatalpha=smoothstep(luminosityThreshold,luminosityThreshold+smoothWidth,v);') &&
    compact.includes('gl_FragColor=mix(outputColor,texel,alpha);')
}

export function isThreeSsrPassShaderMaterial(material: ThreeMaterialLike): boolean {
  const uniforms = material.uniforms
  if (!uniforms || typeof uniforms !== 'object' || Array.isArray(uniforms)) return false

  const values = uniforms as Record<string, unknown>
  if (
    values.tDiffuse == null ||
    values.tNormal == null ||
    values.tMetalness == null ||
    values.tDepth == null ||
    values.cameraNear == null ||
    values.cameraFar == null ||
    values.resolution == null ||
    values.cameraProjectionMatrix == null ||
    values.cameraInverseProjectionMatrix == null ||
    values.opacity == null ||
    values.maxDistance == null ||
    values.cameraRange == null ||
    values.thickness == null
  ) {
    return false
  }

  if (typeof material.fragmentShader !== 'string') return false
  const compact = material.fragmentShader.replace(/\s+/g, '')
  return compact.includes('precisionhighpsampler2D;') &&
    compact.includes('uniformsampler2DtMetalness;') &&
    compact.includes('uniformfloatmaxDistance;') &&
    compact.includes('floatpointToLineDistance(vec3x0,vec3x1,vec3x2)') &&
    compact.includes('vec2viewPositionToXY(vec3viewPosition)') &&
    compact.includes('vec3viewReflectDir=reflect(viewIncidentDir,viewNormal);') &&
    compact.includes('vec4reflectColor=texture2D(tDiffuse,uv);') &&
    compact.includes('gl_FragColor.xyz=reflectColor.xyz;')
}

export function isThreeSmaaPassEdgesShaderMaterial(material: ThreeMaterialLike): boolean {
  const uniforms = material.uniforms
  if (!uniforms || typeof uniforms !== 'object' || Array.isArray(uniforms)) return false

  const values = uniforms as Record<string, unknown>
  if (values.tDiffuse == null || values.resolution == null) return false

  if (typeof material.fragmentShader !== 'string') return false
  const compact = material.fragmentShader.replace(/\s+/g, '')
  return compact.includes('varyingvec4vOffset[3];') &&
    compact.includes('vec4SMAAColorEdgeDetectionPS(vec2texcoord,vec4offset[3],sampler2DcolorTex)') &&
    compact.includes('vec2threshold=vec2(SMAA_THRESHOLD,SMAA_THRESHOLD);') &&
    compact.includes('vec2edges=step(threshold,delta.xy);') &&
    compact.includes('edges.xy*=step(0.5*maxDelta,delta.xy);') &&
    compact.includes('gl_FragColor=SMAAColorEdgeDetectionPS(vUv,vOffset,tDiffuse);')
}

export function isThreeOutlinePassPrepareMaskShaderMaterial(material: ThreeMaterialLike): boolean {
  const uniforms = material.uniforms
  if (!uniforms || typeof uniforms !== 'object' || Array.isArray(uniforms)) return false

  const values = uniforms as Record<string, unknown>
  if (values.depthTexture == null || values.cameraNearFar == null || values.textureMatrix == null) return false

  if (typeof material.fragmentShader !== 'string') return false
  const compact = material.fragmentShader.replace(/\s+/g, '')
  return compact.includes('uniformsampler2DdepthTexture;') &&
    compact.includes('uniformvec2cameraNearFar;') &&
    compact.includes('floatdepth=unpackRGBAToDepth(texture2DProj(depthTexture,projTexCoord));') &&
    compact.includes('DepthToViewZ(depth,cameraNearFar.x,cameraNearFar.y);') &&
    compact.includes('floatdepthTest=(-vPosition.z>viewZ)?1.0:0.0;') &&
    compact.includes('gl_FragColor=vec4(0.0,depthTest,1.0,1.0);')
}

export function isThreeOutlineEffectShaderMaterial(material: ThreeMaterialLike): boolean {
  if (material.type !== 'OutlineEffect') return false

  const uniforms = material.uniforms
  if (!uniforms || typeof uniforms !== 'object' || Array.isArray(uniforms)) return false

  const values = uniforms as Record<string, unknown>
  if (values.outlineThickness == null || values.outlineColor == null || values.outlineAlpha == null) return false

  if (typeof material.vertexShader !== 'string' || typeof material.fragmentShader !== 'string') return false
  const vertexCompact = material.vertexShader.replace(/\s+/g, '')
  const fragmentCompact = material.fragmentShader.replace(/\s+/g, '')
  return vertexCompact.includes('vec4calculateOutline(vec4pos,vec3normal,vec4skinned)') &&
    vertexCompact.includes('gl_Position=calculateOutline(gl_Position,outlineNormal,vec4(transformed,1.0));') &&
    fragmentCompact.includes('uniformvec3outlineColor;') &&
    fragmentCompact.includes('uniformfloatoutlineAlpha;') &&
    fragmentCompact.includes('gl_FragColor=vec4(outlineColor,outlineAlpha);')
}

export function isThreeSkyShaderMaterial(material: ThreeMaterialLike): boolean {
  const uniforms = material.uniforms
  if (!uniforms || typeof uniforms !== 'object' || Array.isArray(uniforms)) return false

  const values = uniforms as Record<string, unknown>
  if (
    material.name !== 'SkyShader' ||
    values.turbidity == null ||
    values.rayleigh == null ||
    values.mieCoefficient == null ||
    values.mieDirectionalG == null ||
    values.sunPosition == null ||
    values.up == null
  ) {
    return false
  }

  if (typeof material.fragmentShader !== 'string') return false
  const compact = material.fragmentShader.replace(/\s+/g, '')
  return compact.includes('uniformfloatmieDirectionalG;') &&
    compact.includes('constfloatrayleighZenithLength=8.4E3;') &&
    compact.includes('floatsundisk=smoothstep(sunAngularDiameterCos,sunAngularDiameterCos+0.00002,cosTheta);') &&
    compact.includes('vec3retColor=pow(texColor,vec3(1.0/(1.2+(1.2*vSunfade))));') &&
    compact.includes('gl_FragColor=vec4(retColor,1.0);')
}

export function isThreeWaterMirrorShaderMaterial(material: ThreeMaterialLike): boolean {
  const uniforms = material.uniforms
  if (!uniforms || typeof uniforms !== 'object' || Array.isArray(uniforms)) return false

  const values = uniforms as Record<string, unknown>
  if (
    material.name !== 'MirrorShader' ||
    values.normalSampler == null ||
    values.mirrorSampler == null ||
    values.alpha == null ||
    values.time == null ||
    values.size == null ||
    values.distortionScale == null ||
    values.textureMatrix == null ||
    values.sunColor == null ||
    values.sunDirection == null ||
    values.eye == null ||
    values.waterColor == null
  ) {
    return false
  }

  if (typeof material.fragmentShader !== 'string') return false
  const compact = material.fragmentShader.replace(/\s+/g, '')
  return compact.includes('uniformsampler2DmirrorSampler;') &&
    compact.includes('uniformsampler2DnormalSampler;') &&
    compact.includes('vec4getNoise(vec2uv)') &&
    compact.includes('voidsunLight(constvec3surfaceNormal,constvec3eyeDirection,floatshiny,floatspec,floatdiffuse,inoutvec3diffuseColor,inoutvec3specularColor)') &&
    compact.includes('vec3reflectionSample=vec3(texture2D(mirrorSampler,mirrorCoord.xy/mirrorCoord.w+distortion));') &&
    compact.includes('gl_FragColor=vec4(outgoingLight,alpha);')
}

export function isThreeWater2ShaderMaterial(material: ThreeMaterialLike): boolean {
  const uniforms = material.uniforms
  if (!uniforms || typeof uniforms !== 'object' || Array.isArray(uniforms)) return false

  const values = uniforms as Record<string, unknown>
  if (
    material.name !== 'WaterShader' ||
    values.color == null ||
    values.reflectivity == null ||
    values.tReflectionMap == null ||
    values.tRefractionMap == null ||
    values.tNormalMap0 == null ||
    values.tNormalMap1 == null ||
    values.textureMatrix == null ||
    values.config == null
  ) {
    return false
  }

  if (typeof material.fragmentShader !== 'string') return false
  const compact = material.fragmentShader.replace(/\s+/g, '')
  return compact.includes('uniformsampler2DtReflectionMap;') &&
    compact.includes('uniformsampler2DtRefractionMap;') &&
    compact.includes('uniformsampler2DtNormalMap0;') &&
    compact.includes('floatflowMapOffset0=config.x;') &&
    compact.includes('vec4normalColor0=texture2D(tNormalMap0,(vUv*scale)+flow*flowMapOffset0);') &&
    compact.includes('vec4reflectColor=texture2D(tReflectionMap,vec2(1.0-uv.x,uv.y));') &&
    compact.includes('gl_FragColor=vec4(color,1.0)*mix(refractColor,reflectColor,reflectance);')
}

export function isThreeReflectorForSsrPassShaderMaterial(material: ThreeMaterialLike): boolean {
  const uniforms = material.uniforms
  if (!uniforms || typeof uniforms !== 'object' || Array.isArray(uniforms)) return false

  const values = uniforms as Record<string, unknown>
  if (
    material.name !== 'ReflectorShader' ||
    values.color == null ||
    values.tDiffuse == null ||
    values.tDepth == null ||
    values.textureMatrix == null ||
    values.maxDistance == null ||
    values.opacity == null ||
    values.fresnelCoe == null ||
    values.virtualCameraNear == null ||
    values.virtualCameraFar == null ||
    values.virtualCameraProjectionMatrix == null ||
    values.virtualCameraMatrixWorld == null ||
    values.virtualCameraProjectionMatrixInverse == null ||
    values.resolution == null
  ) {
    return false
  }

  if (typeof material.fragmentShader !== 'string') return false
  const compact = material.fragmentShader.replace(/\s+/g, '')
  return compact.includes('uniformsampler2DtDepth;') &&
    compact.includes('uniformfloatmaxDistance;') &&
    compact.includes('floatgetViewZ(constinfloatdepth)') &&
    compact.includes('returnperspectiveDepthToViewZ(depth,virtualCameraNear,virtualCameraFar);') &&
    compact.includes('vec3worldPosition=(virtualCameraMatrixWorld*vec4(viewPosition,1)).xyz;') &&
    compact.includes('gl_FragColor=vec4(blendOverlay(base.rgb,color),op);')
}

export function isThreeReflectorShaderMaterial(material: ThreeMaterialLike): boolean {
  const uniforms = material.uniforms
  if (!uniforms || typeof uniforms !== 'object' || Array.isArray(uniforms)) return false

  const values = uniforms as Record<string, unknown>
  if (
    material.name !== 'ReflectorShader' ||
    values.color == null ||
    values.tDiffuse == null ||
    values.textureMatrix == null ||
    values.tDepth != null
  ) {
    return false
  }

  if (typeof material.fragmentShader !== 'string') return false
  const compact = material.fragmentShader.replace(/\s+/g, '')
  return compact.includes('uniformsampler2DtDiffuse;') &&
    compact.includes('floatblendOverlay(floatbase,floatblend)') &&
    compact.includes('vec3blendOverlay(vec3base,vec3blend)') &&
    compact.includes('vec4base=texture2DProj(tDiffuse,vUv);') &&
    compact.includes('gl_FragColor=vec4(blendOverlay(base.rgb,color),1.0);')
}

export function isThreeRefractorShaderMaterial(material: ThreeMaterialLike): boolean {
  const uniforms = material.uniforms
  if (!uniforms || typeof uniforms !== 'object' || Array.isArray(uniforms)) return false

  const values = uniforms as Record<string, unknown>
  if (
    material.name !== 'RefractorShader' ||
    values.color == null ||
    values.tDiffuse == null ||
    values.textureMatrix == null
  ) {
    return false
  }

  if (typeof material.fragmentShader !== 'string') return false
  const compact = material.fragmentShader.replace(/\s+/g, '')
  return compact.includes('uniformsampler2DtDiffuse;') &&
    compact.includes('floatblendOverlay(floatbase,floatblend)') &&
    compact.includes('vec3blendOverlay(vec3base,vec3blend)') &&
    compact.includes('vec4base=texture2DProj(tDiffuse,vUv);') &&
    compact.includes('gl_FragColor=vec4(blendOverlay(base.rgb,color),1.0);')
}

export function isThreeShadowMapViewerShaderMaterial(material: ThreeMaterialLike): boolean {
  const uniforms = material.uniforms
  if (!uniforms || typeof uniforms !== 'object' || Array.isArray(uniforms)) return false

  const values = uniforms as Record<string, unknown>
  if (values.tDiffuse == null || values.opacity == null) return false

  if (typeof material.fragmentShader !== 'string') return false
  const compact = material.fragmentShader.replace(/\s+/g, '')
  return compact.includes('uniformfloatopacity;') &&
    compact.includes('uniformsampler2DtDiffuse;') &&
    compact.includes('floatdepth=1.0-unpackRGBAToDepth(texture2D(tDiffuse,vUv));') &&
    compact.includes('gl_FragColor=vec4(vec3(depth),opacity);')
}

export function isThreeLightProbeHelperShaderMaterial(material: ThreeMaterialLike): boolean {
  if (material.type === 'LightProbeHelperMaterial') return true

  const uniforms = material.uniforms
  if (!uniforms || typeof uniforms !== 'object' || Array.isArray(uniforms)) return false

  const values = uniforms as Record<string, unknown>
  if (values.sh == null || values.intensity == null) return false

  if (typeof material.fragmentShader !== 'string') return false
  const compact = material.fragmentShader.replace(/\s+/g, '')
  return compact.includes('uniformvec3sh[9];') &&
    compact.includes('vec3shGetIrradianceAt(invec3normal,invec3shCoefficients[9])') &&
    compact.includes('vec3outgoingLight=RECIPROCAL_PI*irradiance*intensity;')
}

export function isThreeTextureHelperShaderMaterial(material: ThreeMaterialLike): boolean {
  if (material.type === 'TextureHelperMaterial') return true

  const uniforms = material.uniforms
  if (!uniforms || typeof uniforms !== 'object' || Array.isArray(uniforms)) return false

  const values = uniforms as Record<string, unknown>
  if (values.map == null || values.alpha == null) return false

  if (typeof material.fragmentShader !== 'string') return false
  const compact = material.fragmentShader.replace(/\s+/g, '')
  return compact.includes('uniformfloatalpha;') &&
    compact.includes('varyingvec3vUvw;') &&
    compact.includes('vec4textureHelper(insampler2Dmap){returntexture(map,vUvw.xy);}') &&
    compact.includes('gl_FragColor=linearToOutputTexel(vec4(textureHelper(map).xyz,alpha));')
}

export function isThreeLineMaterialShaderMaterial(material: ThreeMaterialLike): boolean {
  return (material as { isLineMaterial?: unknown }).isLineMaterial === true ||
    material.type === 'LineMaterial'
}

export function namedShaderMaterialLabel(kind: string, material: ThreeMaterialLike): string {
  return typeof material.name === 'string' && material.name.trim().length > 0
    ? `${kind} "${material.name}"`
    : kind
}

export function assertSupportedOnBeforeCompile(
  material: ThreeMaterialLike,
  customFragmentShader: string | undefined,
): void {
  if (customFragmentShader || !hasCustomOnBeforeCompile(material)) return

  if (isThreeCsmPatchedMaterial(material)) {
    throw new Error(
      'THREE.CSM material onBeforeCompile customization is not translated by @headless-three/renderer yet. Use regular supported native lights and shadows, pre-bake cascaded shadowing into textures, or provide material.userData.headlessThreeRenderer.fragmentWgsl with a WGSL fragment body for the renderer custom material path.',
    )
  }

  throw new Error(
    'material.onBeforeCompile customizations are not translated by @headless-three/renderer yet. Provide material.userData.headlessThreeRenderer.fragmentWgsl with a WGSL fragment body for the renderer custom material path.',
  )
}

export function assertSupportedMaterialNodeHooks(material: ThreeMaterialLike): void {
  const nodeHooks = activeMaterialNodeHooks(material)
  if (nodeHooks.length === 0) return

  throw new Error(
    `material node hooks (${nodeHooks.join(', ')}) are not translated by @headless-three/renderer yet. Use a supported built-in material without Three.js TSL node hooks, bake the effect into geometry/textures, or provide equivalent fragment output through material.userData.headlessThreeRenderer.fragmentWgsl when only fragment color needs customization.`,
  )
}
