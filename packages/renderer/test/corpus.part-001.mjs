import * as THREE from 'three'
import { alphaToCoverageClippingCorpus, backgroundOverrideCorpus, customBlendingCorpus, lightProbeCorpus, materialRenderStateNoopCorpus, optionBackgroundTextureControlsCorpus, rendererClearColorFallbackCorpus, signedRawTextureCorpus, stencilRenderStateCorpus } from './corpus.part-002.mjs'
import { fogExp2MixedObjectCorpus, lightMapCorpus, lightProbeEnvironmentMaterialModelsCorpus, lightProbeMaterialModelsCorpus, linearFogCorpus, linearOutputColorSpaceCorpus, phongSpecularMapMatrixCorpus, textureMatrixColorSpaceCorpus, textureSlotMatrixCorpus } from './corpus.part-003.mjs'
import { customWgslPremultipliedCorpus, depthRenderModeCorpus, maskRenderModeCorpus, normalRenderModeCorpus, objectIdRenderModeCorpus, postProcessingOptionsCorpus, renderModeAlphaHashCutoutCorpus, sceneOverrideMaterialCorpus, toneMappingStateCorpus } from './corpus.part-004.mjs'
import { billboardAlphaCutoutCorpus, billboardReceiveShadowNoopCorpus, renderModeMrtAuxiliaryCorpus, renderModeTextureAlphaCutoutCorpus, spriteAlphaMapCorpus, spriteMaterialCorpus, twoDimensionalBackgroundTextureCorpus } from './corpus.part-005.mjs'
import { billboardCustomShadowCutoutCorpus, billboardPointLightShadowCorpus, globalClippingPlaneCorpus, materialLocalClippingCorpus, pointSpotLightCorpus, rectAreaLightCorpus, spriteShadowCorpus } from './corpus.part-006.mjs'
import { cubeEnvironmentOptionRotationCorpus, cubeUvMaterialEnvMapCorpus, materialEnvMapBasicLambertCorpus, materialEnvMapCorpus, materialEnvMapPbrCorpus, meshBasicMaterialWireframeCorpus, meshDepthMaterialCorpus, narrowRawIblCorpus, nestedClippingGroupCorpus, packedCubeUvMaterialEnvMapCorpus, rendererClippingStateCorpus } from './corpus.part-007.mjs'
import { meshDepthDisplacementMapCorpus, meshDepthMaterialWireframeCorpus, meshDepthPackingVariantsCorpus, meshDistanceDisplacementMapCorpus, meshDistanceMaterialCorpus, meshDistanceMaterialWireframeCorpus, meshNormalMaterialCorpus, meshNormalMaterialNormalMapCorpus, meshStandardMaterialDisplacementCorpus } from './corpus.part-008.mjs'
import { meshMatcapMaterialBumpMapCorpus, meshMatcapMaterialCorpus, meshMatcapMaterialFlatShadingCorpus, meshMatcapMaterialNormalMapCorpus, meshMatcapMaterialObjectSpaceNormalMapCorpus, meshNormalMaterialBumpMapCorpus, meshNormalMaterialObjectSpaceNormalMapCorpus, meshToonMaterialCorpus, meshToonMaterialFallbackBandsCorpus } from './corpus.part-009.mjs'
import { arrayCameraViewportCorpus, cameraLayerFilteringCorpus, cubeCameraCaptureCorpus, cubeCameraUpdateCorpus, meshToonAlphaMapCorpus, meshToonMaterialBumpMapCorpus, meshToonMaterialNormalMapCorpus, meshToonTextureSlotsCorpus, viewportScissorCorpus } from './corpus.part-010.mjs'
import { cubeBackgroundOptionRotationCorpus, cubeBackgroundTextureCorpus, cubeUvBackgroundTextureCorpus, customSortGroupCorpus, customTransparentSortGroupCorpus, equirectangularBackgroundCorpus, packedCubeUvBackgroundTextureCorpus, rendererBucketFlagsCorpus, skinnedMorphCorpus } from './corpus.part-011.mjs'
import { avatarLikeCorpus, physicalClearcoatMapCorpus, physicalIblShadowCorpus, physicalSheenMapCorpus, physicalSpecularMapCorpus } from './corpus.part-012.mjs'
import { physicalAnisotropyMapCorpus, physicalIridescenceMapCorpus, physicalTransmissionDispersionCorpus, physicalTransmissionMapCorpus, transmissionResolutionScaleCorpus } from './corpus.part-013.mjs'
import { customShadowDisplacementCorpus, mixedShadowLightTypesCorpus, multipleDirectionalShadowCorpus, shadowMapEnabledGatingCorpus, shadowMapTypeFilteringCorpus } from './corpus.part-014.mjs'
import { dashedLineMaterialCorpus, dashedLineMaterialTextureCorpus, dashedLineMaterialUvChannelCorpus, shadowMaterialFogOptOutCorpus, shadowMaterialOpacityCorpus, shadowMaterialOutputColorSpaceCorpus, shadowMaterialReceiverCorpus } from './corpus.part-015.mjs'
import { dashedLineMaterialCustomDistanceCorpus, dashedLineMaterialLineLoopDistanceCorpus, dashedLineMaterialWideLineCorpus, lineBasicMaterialUvChannelCorpus, lineMaterialNoopCorpus, pointsMaterialTextureCorpus, pointsMaterialUvChannelCorpus } from './corpus.part-016.mjs'
import { batchedMeshCorpus, batchedMeshInactiveGeometryCorpus, batchedMeshIndexedGroupsCorpus, batchedMeshOptimizedRangeCorpus, instancedLineNoBridgeCorpus, instancedLinesPointsCorpus, instancedTextureUvCorpus, renderableFrustumCullingCorpus } from './corpus.part-017.mjs'
import { batchedMeshCullingCorpus, batchedMeshCullingOptOutCorpus, batchedMeshDefaultGroupMaterialCorpus, batchedMeshMultiSourceGroupOffsetsCorpus, batchedMeshNonIndexedGroupsCorpus, batchedMeshPartialGroupRangeCorpus, batchedMeshSparseMaterialGroupsCorpus } from './corpus.part-018.mjs'
import { batchedMeshCustomSortCorpus, lodAndGroupsCorpus, lodZoomCorpus, pathologicalGeometryCorpus } from './corpus.part-019.mjs'
export const CORPUS_RENDER_SIZE = 96

export function createSceneCorpus() {
  return [
    transparentLayerCorpus(),
    alphaToCoverageCorpus(),
    alphaToCoverageAlphaTestCorpus(),
    alphaToCoverageClippingCorpus(),
    stencilRenderStateCorpus(),
    customBlendingCorpus(),
    materialRenderStateNoopCorpus(),
    backgroundOverrideCorpus(),
    optionBackgroundTextureControlsCorpus(),
    rendererClearColorFallbackCorpus(),
    twoDimensionalBackgroundTextureCorpus(),
    signedRawTextureCorpus(),
    equirectangularBackgroundCorpus(),
    cubeBackgroundTextureCorpus(),
    cubeBackgroundOptionRotationCorpus(),
    cubeUvBackgroundTextureCorpus(),
    packedCubeUvBackgroundTextureCorpus(),
    arrayCameraViewportCorpus(),
    cubeCameraCaptureCorpus(),
    cubeCameraUpdateCorpus(),
    viewportScissorCorpus(),
    cameraLayerFilteringCorpus(),
    customSortGroupCorpus(),
    customTransparentSortGroupCorpus(),
    rendererBucketFlagsCorpus(),
    materialEnvMapCorpus(),
    materialEnvMapBasicLambertCorpus(),
    materialEnvMapPbrCorpus(),
    cubeUvMaterialEnvMapCorpus(),
    packedCubeUvMaterialEnvMapCorpus(),
    cubeEnvironmentOptionRotationCorpus(),
    narrowRawIblCorpus(),
    meshBasicMaterialWireframeCorpus(),
    meshDepthMaterialCorpus(),
    meshDepthPackingVariantsCorpus(),
    meshDepthDisplacementMapCorpus(),
    meshDepthMaterialWireframeCorpus(),
    meshDistanceMaterialCorpus(),
    meshDistanceDisplacementMapCorpus(),
    meshDistanceMaterialWireframeCorpus(),
    meshStandardMaterialDisplacementCorpus(),
    meshNormalMaterialCorpus(),
    meshNormalMaterialNormalMapCorpus(),
    meshNormalMaterialObjectSpaceNormalMapCorpus(),
    meshNormalMaterialBumpMapCorpus(),
    meshMatcapMaterialCorpus(),
    meshMatcapMaterialFlatShadingCorpus(),
    meshMatcapMaterialNormalMapCorpus(),
    meshMatcapMaterialObjectSpaceNormalMapCorpus(),
    meshMatcapMaterialBumpMapCorpus(),
    meshToonMaterialFallbackBandsCorpus(),
    meshToonMaterialCorpus(),
    meshToonMaterialNormalMapCorpus(),
    meshToonMaterialBumpMapCorpus(),
    meshToonTextureSlotsCorpus(),
    meshToonAlphaMapCorpus(),
    globalClippingPlaneCorpus(),
    materialLocalClippingCorpus(),
    rendererClippingStateCorpus(),
    nestedClippingGroupCorpus(),
    lightProbeCorpus(),
    lightProbeMaterialModelsCorpus(),
    lightProbeEnvironmentMaterialModelsCorpus(),
    linearFogCorpus(),
    fogExp2MixedObjectCorpus(),
    textureMatrixColorSpaceCorpus(),
    phongSpecularMapMatrixCorpus(),
    textureSlotMatrixCorpus(),
    lightMapCorpus(),
    linearOutputColorSpaceCorpus(),
    toneMappingStateCorpus(),
    postProcessingOptionsCorpus(),
    customWgslPremultipliedCorpus(),
    sceneOverrideMaterialCorpus(),
    maskRenderModeCorpus(),
    objectIdRenderModeCorpus(),
    normalRenderModeCorpus(),
    depthRenderModeCorpus(),
    renderModeAlphaHashCutoutCorpus(),
    renderModeTextureAlphaCutoutCorpus(),
    renderModeMrtAuxiliaryCorpus(),
    spriteMaterialCorpus(),
    spriteAlphaMapCorpus(),
    billboardAlphaCutoutCorpus(),
    billboardReceiveShadowNoopCorpus(),
    spriteShadowCorpus(),
    billboardPointLightShadowCorpus(),
    billboardCustomShadowCutoutCorpus(),
    pointSpotLightCorpus(),
    rectAreaLightCorpus(),
    skinnedMorphCorpus(),
    avatarLikeCorpus(),
    physicalIblShadowCorpus(),
    physicalClearcoatMapCorpus(),
    physicalSheenMapCorpus(),
    physicalSpecularMapCorpus(),
    physicalAnisotropyMapCorpus(),
    physicalIridescenceMapCorpus(),
    physicalTransmissionMapCorpus(),
    physicalTransmissionDispersionCorpus(),
    transmissionResolutionScaleCorpus(),
    multipleDirectionalShadowCorpus(),
    mixedShadowLightTypesCorpus(),
    shadowMapEnabledGatingCorpus(),
    shadowMapTypeFilteringCorpus(),
    customShadowDisplacementCorpus(),
    shadowMaterialReceiverCorpus(),
    shadowMaterialOpacityCorpus(),
    shadowMaterialOutputColorSpaceCorpus(),
    shadowMaterialFogOptOutCorpus(),
    dashedLineMaterialCorpus(),
    dashedLineMaterialTextureCorpus(),
    dashedLineMaterialUvChannelCorpus(),
    dashedLineMaterialCustomDistanceCorpus(),
    dashedLineMaterialLineLoopDistanceCorpus(),
    dashedLineMaterialWideLineCorpus(),
    lineMaterialNoopCorpus(),
    lineBasicMaterialUvChannelCorpus(),
    pointsMaterialTextureCorpus(),
    pointsMaterialUvChannelCorpus(),
    instancedLinesPointsCorpus(),
    instancedLineNoBridgeCorpus(),
    instancedTextureUvCorpus(),
    renderableFrustumCullingCorpus(),
    batchedMeshCorpus(),
    batchedMeshInactiveGeometryCorpus(),
    batchedMeshOptimizedRangeCorpus(),
    batchedMeshIndexedGroupsCorpus(),
    batchedMeshMultiSourceGroupOffsetsCorpus(),
    batchedMeshNonIndexedGroupsCorpus(),
    batchedMeshDefaultGroupMaterialCorpus(),
    batchedMeshPartialGroupRangeCorpus(),
    batchedMeshSparseMaterialGroupsCorpus(),
    batchedMeshCullingCorpus(),
    batchedMeshCullingOptOutCorpus(),
    batchedMeshCustomSortCorpus(),
    lodAndGroupsCorpus(),
    lodZoomCorpus(),
    pathologicalGeometryCorpus(),
  ]
}

export function makeCamera(position = [2.2, 1.6, 3.1], target = [0, 0, 0]) {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(position[0], position[1], position[2])
  camera.lookAt(target[0], target[1], target[2])
  return camera
}

export function addBasicLights(scene) {
  scene.add(new THREE.AmbientLight(0xffffff, 0.25))
  const dir = new THREE.DirectionalLight(0xffffff, 1.2)
  dir.position.set(3, 5, 2)
  dir.target.position.set(0, 0, 0)
  scene.add(dir)
  scene.add(dir.target)
}

export function solidTexture(r, g, b, a = 255) {
  const texture = new THREE.DataTexture(new Uint8Array([r, g, b, a]), 1, 1, THREE.RGBAFormat)
  texture.needsUpdate = true
  return texture
}

export function environmentTexture() {
  const data = new Uint8Array([
    255, 255, 255, 255,
    64, 128, 255, 255,
    255, 180, 96, 255,
    16, 24, 40, 255,
  ])
  const texture = new THREE.DataTexture(data, 2, 2, THREE.RGBAFormat)
  texture.needsUpdate = true
  return texture
}

export function gradientTexture() {
  const texture = new THREE.DataTexture(new Uint8Array([
    88, 88, 120, 255,
    255, 226, 178, 255,
  ]), 2, 1, THREE.RGBAFormat)
  texture.magFilter = THREE.NearestFilter
  texture.minFilter = THREE.NearestFilter
  texture.needsUpdate = true
  return texture
}

export function constantUvPlane(u, v, width = 2, height = 2) {
  const geometry = new THREE.PlaneGeometry(width, height)
  const uv = geometry.getAttribute('uv')
  for (let i = 0; i < uv.count; i += 1) {
    uv.setXY(i, u, v)
  }
  return geometry
}

export function cubeTexture(faceColors) {
  const faces = faceColors.map(([r, g, b, a = 255]) => ({
    data: new Uint8Array([r, g, b, a]),
    width: 1,
    height: 1,
  }))
  const texture = new THREE.CubeTexture(faces)
  texture.needsUpdate = true
  return texture
}

export function cubeUvGreenCubeTexture() {
  const texture = cubeTexture([
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0],
  ])
  texture.mapping = THREE.CubeUVReflectionMapping
  return texture
}

export function packedCubeUvTexture(faceColors, faceSize = 16) {
  const width = 3 * Math.max(faceSize, 16 * 7)
  const height = 4 * faceSize
  const data = new Uint8Array(width * height * 4)
  const atlasFaceToCubeFace = [0, 2, 4, 1, 3, 5]
  for (let atlasFace = 0; atlasFace < 6; atlasFace += 1) {
    const [r, g, b, a = 255] = faceColors[atlasFaceToCubeFace[atlasFace]]
    const col = atlasFace % 3
    const row = atlasFace > 2 ? 1 : 0
    for (let y = 0; y < faceSize; y += 1) {
      for (let x = 0; x < faceSize; x += 1) {
        const offset = (((row * faceSize + y) * width) + (col * faceSize + x)) * 4
        data[offset] = r
        data[offset + 1] = g
        data[offset + 2] = b
        data[offset + 3] = a
      }
    }
  }
  const texture = new THREE.DataTexture(data, width, height, THREE.RGBAFormat)
  texture.mapping = THREE.CubeUVReflectionMapping
  texture.needsUpdate = true
  return texture
}

export function packedCubeUvGreenTexture() {
  return packedCubeUvTexture([
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0],
  ])
}

export function coloredCubeBackgroundTexture() {
  return cubeTexture([
    [48, 80, 255],
    [255, 225, 72],
    [255, 64, 220],
    [32, 210, 220],
    [32, 200, 96],
    [255, 48, 32],
  ])
}

export function packedCubeUvColoredBackgroundTexture() {
  return packedCubeUvTexture([
    [48, 80, 255],
    [255, 225, 72],
    [255, 64, 220],
    [32, 210, 220],
    [32, 200, 96],
    [255, 48, 32],
  ])
}

export function spriteMapTexture() {
  const texture = new THREE.DataTexture(new Uint8Array([
    255, 80, 60, 255,
    40, 210, 120, 255,
    55, 95, 240, 255,
    250, 230, 80, 255,
  ]), 2, 2, THREE.RGBAFormat)
  texture.colorSpace = THREE.SRGBColorSpace
  texture.magFilter = THREE.NearestFilter
  texture.minFilter = THREE.NearestFilter
  texture.needsUpdate = true
  return texture
}

export function meanRegion(rgba, width, x0, y0, x1, y1) {
  let r = 0
  let g = 0
  let b = 0
  let count = 0
  for (let y = y0; y < y1; y += 1) {
    for (let x = x0; x < x1; x += 1) {
      const offset = (y * width + x) * 4
      r += rgba[offset]
      g += rgba[offset + 1]
      b += rgba[offset + 2]
      count += 1
    }
  }
  return { r: r / count, g: g / count, b: b / count }
}

export function meanAbsDiff(a, b) {
  let total = 0
  let count = 0
  for (let i = 0; i < a.length; i += 4) {
    total += Math.abs(a[i] - b[i])
    total += Math.abs(a[i + 1] - b[i + 1])
    total += Math.abs(a[i + 2] - b[i + 2])
    count += 3
  }
  return total / count
}

export function countRegionPixels(rgba, width, x0, y0, x1, y1, predicate) {
  let count = 0
  for (let y = y0; y < y1; y += 1) {
    for (let x = x0; x < x1; x += 1) {
      const offset = (y * width + x) * 4
      if (predicate(rgba[offset], rgba[offset + 1], rgba[offset + 2], rgba[offset + 3])) {
        count += 1
      }
    }
  }
  return count
}

export function alphaCoverageBandPixels(rgba, width) {
  return countRegionPixels(rgba, width, 20, 20, 76, 76, (r, g, b) => {
    return r > 35 && r < 180 && Math.abs(r - g) < 4 && Math.abs(r - b) < 4
  })
}

export function pixelAt(rgba, width, x, y) {
  const offset = (y * width + x) * 4
  return {
    r: rgba[offset],
    g: rgba[offset + 1],
    b: rgba[offset + 2],
    a: rgba[offset + 3],
  }
}

export function setTextureMatrixOffset(texture, x, y = 0) {
  texture.matrixAutoUpdate = false
  texture.matrix.set(
    1, 0, x,
    0, 1, y,
    0, 0, 1,
  )
}

export function transparentLayerCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.08, 0.08, 0.1)

  const back = new THREE.Mesh(
    new THREE.PlaneGeometry(1.6, 1.6),
    new THREE.MeshBasicMaterial({ color: 0xff5522, transparent: true, opacity: 0.65 }),
  )
  back.position.z = -0.04
  back.renderOrder = 1

  const front = new THREE.Mesh(
    new THREE.PlaneGeometry(1.2, 1.2),
    new THREE.MeshBasicMaterial({ color: 0x2266ff, transparent: true, opacity: 0.55 }),
  )
  front.position.z = 0.04
  front.renderOrder = 2

  scene.add(back, front)
  return {
    name: 'transparent-layer-stack',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [80, 80, 89],
    validate(rgba, { width }) {
      const center = pixelAt(rgba, width, 48, 48)
      const corner = pixelAt(rgba, width, 4, 4)
      const cornerMatchesBackground = Math.abs(corner.r - 80) <= 1 && Math.abs(corner.g - 80) <= 1 && Math.abs(corner.b - 89) <= 1
      if (!(center.r > center.g + 8 && center.b > center.r + 30 && center.b > center.g + 40 && cornerMatchesBackground)) {
        throw new Error(`transparent layer corpus should blend the blue front over the orange back, got center=${JSON.stringify(center)} corner=${JSON.stringify(corner)}`)
      }
    },
  }
}

export function alphaToCoverageCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(1.7, 1.7),
    new THREE.MeshBasicMaterial({
      color: 0xffffff,
      opacity: 0.5,
      transparent: false,
      alphaToCoverage: true,
    }),
  ))

  return {
    name: 'alpha-to-coverage-msaa-plane',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: {
      width: CORPUS_RENDER_SIZE,
      height: CORPUS_RENDER_SIZE,
      format: 'rgba',
      sampleCount: 4,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.08,
    minMeanAlpha: 190,
    validate(rgba, { width }) {
      const center = pixelAt(rgba, width, 48, 48)
      const corner = pixelAt(rgba, width, 4, 4)
      if (!(center.r > 80 && center.r < 120 && center.g > 80 && center.g < 120 && center.b > 80 && center.b < 120 && center.a > 180 && center.a < 210 && corner.r === 0 && corner.g === 0 && corner.b === 0 && corner.a === 255)) {
        throw new Error(`alpha-to-coverage corpus should resolve a partial gray plane over black, got center=${JSON.stringify(center)} corner=${JSON.stringify(corner)}`)
      }
    },
  }
}

export function alphaToCoverageAlphaTestCorpus() {
  const alphaMap = new THREE.DataTexture(new Uint8Array([
    255, 0, 255, 255,
    255, 255, 255, 255,
  ]), 2, 1, THREE.RGBAFormat)
  alphaMap.magFilter = THREE.LinearFilter
  alphaMap.minFilter = THREE.LinearFilter
  alphaMap.needsUpdate = true

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(1.7, 1.7),
    new THREE.MeshBasicMaterial({
      color: 0xffffff,
      alphaMap,
      alphaTest: 0.5,
      alphaToCoverage: true,
    }),
  ))

  return {
    name: 'alpha-to-coverage-alpha-test-threshold',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: {
      width: CORPUS_RENDER_SIZE,
      height: CORPUS_RENDER_SIZE,
      format: 'rgba',
      sampleCount: 4,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.08,
    validate(rgba, { width }) {
      const softPixels = alphaCoverageBandPixels(rgba, width)
      if (softPixels < 24) {
        throw new Error(`alpha-to-coverage alphaTest corpus should produce a soft threshold band, got ${softPixels} partial pixels`)
      }
    },
  }
}
