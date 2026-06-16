import * as THREE from 'three'

export const CORPUS_RENDER_SIZE = 96

export function createSceneCorpus() {
  return [
    transparentLayerCorpus(),
    alphaToCoverageCorpus(),
    stencilRenderStateCorpus(),
    customBlendingCorpus(),
    backgroundOverrideCorpus(),
    twoDimensionalBackgroundTextureCorpus(),
    signedRawTextureCorpus(),
    equirectangularBackgroundCorpus(),
    cubeBackgroundTextureCorpus(),
    arrayCameraViewportCorpus(),
    cubeCameraCaptureCorpus(),
    viewportScissorCorpus(),
    customSortGroupCorpus(),
    materialEnvMapCorpus(),
    materialEnvMapBasicLambertCorpus(),
    materialEnvMapPbrCorpus(),
    meshBasicMaterialWireframeCorpus(),
    meshDepthMaterialCorpus(),
    meshDepthMaterialWireframeCorpus(),
    meshDistanceMaterialCorpus(),
    meshDistanceMaterialWireframeCorpus(),
    meshNormalMaterialCorpus(),
    meshMatcapMaterialCorpus(),
    meshToonMaterialCorpus(),
    meshToonAlphaMapCorpus(),
    globalClippingPlaneCorpus(),
    materialLocalClippingCorpus(),
    nestedClippingGroupCorpus(),
    lightProbeCorpus(),
    lightProbeMaterialModelsCorpus(),
    linearFogCorpus(),
    textureMatrixColorSpaceCorpus(),
    linearOutputColorSpaceCorpus(),
    maskRenderModeCorpus(),
    objectIdRenderModeCorpus(),
    normalRenderModeCorpus(),
    spriteMaterialCorpus(),
    spriteShadowCorpus(),
    pointSpotLightCorpus(),
    rectAreaLightCorpus(),
    skinnedMorphCorpus(),
    avatarLikeCorpus(),
    physicalIblShadowCorpus(),
    shadowMaterialReceiverCorpus(),
    shadowMaterialFogOptOutCorpus(),
    dashedLineMaterialCorpus(),
    dashedLineMaterialTextureCorpus(),
    pointsMaterialTextureCorpus(),
    instancedLinesPointsCorpus(),
    instancedTextureUvCorpus(),
    batchedMeshCorpus(),
    batchedMeshInactiveGeometryCorpus(),
    batchedMeshCullingCorpus(),
    batchedMeshCustomSortCorpus(),
    lodAndGroupsCorpus(),
    lodZoomCorpus(),
    pathologicalGeometryCorpus(),
  ]
}

function makeCamera(position = [2.2, 1.6, 3.1], target = [0, 0, 0]) {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(position[0], position[1], position[2])
  camera.lookAt(target[0], target[1], target[2])
  return camera
}

function addBasicLights(scene) {
  scene.add(new THREE.AmbientLight(0xffffff, 0.25))
  const dir = new THREE.DirectionalLight(0xffffff, 1.2)
  dir.position.set(3, 5, 2)
  dir.target.position.set(0, 0, 0)
  scene.add(dir)
  scene.add(dir.target)
}

function solidTexture(r, g, b, a = 255) {
  const texture = new THREE.DataTexture(new Uint8Array([r, g, b, a]), 1, 1, THREE.RGBAFormat)
  texture.needsUpdate = true
  return texture
}

function environmentTexture() {
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

function gradientTexture() {
  const texture = new THREE.DataTexture(new Uint8Array([
    88, 88, 120, 255,
    255, 226, 178, 255,
  ]), 2, 1, THREE.RGBAFormat)
  texture.magFilter = THREE.NearestFilter
  texture.minFilter = THREE.NearestFilter
  texture.needsUpdate = true
  return texture
}

function cubeTexture(faceColors) {
  const faces = faceColors.map(([r, g, b, a = 255]) => ({
    data: new Uint8Array([r, g, b, a]),
    width: 1,
    height: 1,
  }))
  const texture = new THREE.CubeTexture(faces)
  texture.needsUpdate = true
  return texture
}

function spriteMapTexture() {
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

function meanRegion(rgba, width, x0, y0, x1, y1) {
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

function countRegionPixels(rgba, width, x0, y0, x1, y1, predicate) {
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

function pixelAt(rgba, width, x, y) {
  const offset = (y * width + x) * 4
  return {
    r: rgba[offset],
    g: rgba[offset + 1],
    b: rgba[offset + 2],
    a: rgba[offset + 3],
  }
}

function transparentLayerCorpus() {
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
    background: [20, 20, 26],
    validate(rgba, { width }) {
      const center = pixelAt(rgba, width, 48, 48)
      const corner = pixelAt(rgba, width, 4, 4)
      if (!(center.r > center.g + 40 && center.b > center.g + 50 && corner.r === 20 && corner.g === 20 && corner.b === 26)) {
        throw new Error(`transparent layer corpus should blend the blue front over the orange back, got center=${JSON.stringify(center)} corner=${JSON.stringify(corner)}`)
      }
    },
  }
}

function alphaToCoverageCorpus() {
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
    minMeanAlpha: 220,
    validate(rgba, { width }) {
      const center = pixelAt(rgba, width, 48, 48)
      const corner = pixelAt(rgba, width, 4, 4)
      if (!(center.r > 35 && center.r < 70 && center.g > 35 && center.g < 70 && center.b > 35 && center.b < 70 && center.a < 240 && corner.r === 0 && corner.g === 0 && corner.b === 0 && corner.a === 255)) {
        throw new Error(`alpha-to-coverage corpus should resolve a partial gray plane over black, got center=${JSON.stringify(center)} corner=${JSON.stringify(corner)}`)
      }
    },
  }
}

function signedRawTextureCorpus() {
  const background = new THREE.DataTexture(
    new Int16Array([0, 0x3000, 0x7fff, 0x7fff]),
    1,
    1,
    THREE.RGBAFormat,
    THREE.ShortType,
  )
  background.needsUpdate = true

  const map = new THREE.DataTexture(
    new Int8Array([
      80, 20, 127, 127,
      20, 80, 127, 127,
      127, 20, 80, 127,
      80, 127, 20, 127,
    ]),
    2,
    2,
    THREE.RGBAFormat,
    THREE.ByteType,
  )
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter
  map.needsUpdate = true

  const scene = new THREE.Scene()
  scene.background = background
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(1.5, 1.5),
    new THREE.MeshBasicMaterial({ map }),
  )
  scene.add(mesh)

  return {
    name: 'signed-raw-datatexture-material-background',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: {
      width: CORPUS_RENDER_SIZE,
      height: CORPUS_RENDER_SIZE,
      format: 'rgba',
      outputColorSpace: THREE.LinearSRGBColorSpace,
    },
    background: [0, 96, 255],
    minNonBackgroundRatio: 0.08,
    browserReference: false,
    validate(rgba, { width }) {
      const background = pixelAt(rgba, width, 4, 4)
      const yellow = pixelAt(rgba, width, 48, 28)
      const magenta = pixelAt(rgba, width, 28, 48)
      const cyan = pixelAt(rgba, width, 68, 48)
      if (!(background.r === 0 && background.g === 96 && background.b === 255 && yellow.r > 140 && yellow.g > 160 && yellow.b < 120 && magenta.r > 150 && magenta.b > 150 && magenta.g < 90 && cyan.g > 140 && cyan.b > 160 && cyan.r < 120)) {
        throw new Error(`signed raw texture corpus should render normalized signed material texels and background, got background=${JSON.stringify(background)} yellow=${JSON.stringify(yellow)} magenta=${JSON.stringify(magenta)} cyan=${JSON.stringify(cyan)}`)
      }
    },
  }
}

function stencilRenderStateCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const mask = new THREE.Mesh(
    new THREE.PlaneGeometry(1, 2),
    new THREE.MeshBasicMaterial({
      color: 0xff0000,
      colorWrite: false,
      depthWrite: false,
      stencilWrite: true,
      stencilFunc: THREE.AlwaysStencilFunc,
      stencilRef: 1,
      stencilZPass: THREE.ReplaceStencilOp,
    }),
  )
  mask.position.x = -0.5
  mask.renderOrder = 0
  scene.add(mask)

  const fill = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({
      color: 0x2255ff,
      stencilWrite: true,
      stencilFunc: THREE.EqualStencilFunc,
      stencilRef: 1,
      stencilFail: THREE.KeepStencilOp,
      stencilZFail: THREE.KeepStencilOp,
      stencilZPass: THREE.KeepStencilOp,
      stencilWriteMask: 0,
    }),
  )
  fill.renderOrder = 1
  scene.add(fill)

  return {
    name: 'stencil-masked-render-state',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.08,
    validate(rgba, { width }) {
      const masked = meanRegion(rgba, width, 16, 24, 44, 72)
      const unmasked = meanRegion(rgba, width, 52, 24, 80, 72)
      if (!(masked.b > masked.g + 180 && masked.b > masked.r + 200 && unmasked.r < 2 && unmasked.g < 2 && unmasked.b < 2)) {
        throw new Error(`stencil corpus should render only the masked blue side, got masked=${JSON.stringify(masked)} unmasked=${JSON.stringify(unmasked)}`)
      }
    },
  }
}

function customBlendingCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const back = new THREE.Mesh(
    new THREE.PlaneGeometry(1.7, 1.7),
    new THREE.MeshBasicMaterial({ color: 0xffffff }),
  )
  back.position.z = -0.1
  scene.add(back)

  const front = new THREE.Mesh(
    new THREE.PlaneGeometry(1.7, 1.7),
    new THREE.MeshBasicMaterial({
      color: 0xff0000,
      transparent: true,
      blending: THREE.CustomBlending,
      blendEquation: THREE.ReverseSubtractEquation,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneFactor,
      blendEquationAlpha: THREE.AddEquation,
      blendSrcAlpha: THREE.OneFactor,
      blendDstAlpha: THREE.ZeroFactor,
    }),
  )
  front.position.z = 0.1
  scene.add(front)

  return {
    name: 'custom-blending-reverse-subtract',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.08,
    validate(rgba, { width }) {
      const center = meanRegion(rgba, width, 32, 32, 64, 64)
      if (!(center.r < 4 && center.g > 180 && center.b > 180)) {
        throw new Error(`reverse-subtract blending corpus should render cyan in the overlap, got ${JSON.stringify(center)}`)
      }
    },
  }
}

function backgroundOverrideCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.7, 0.05, 0.05)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(1.2, 1.2),
    new THREE.MeshBasicMaterial({ color: 0x33cc88 }),
  ))

  return {
    name: 'option-background-override-color',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: {
      width: CORPUS_RENDER_SIZE,
      height: CORPUS_RENDER_SIZE,
      format: 'rgba',
      background: new THREE.Color(0, 0, 0),
    },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.06,
    validate(rgba, { width }) {
      const center = meanRegion(rgba, width, 32, 32, 64, 64)
      const corner = meanRegion(rgba, width, 4, 4, 20, 20)
      if (!(center.g > center.r + 120 && center.g > center.b + 80 && corner.r < 2 && corner.g < 2 && corner.b < 2)) {
        throw new Error(`background override corpus should render green mesh on black option background, got center=${JSON.stringify(center)} corner=${JSON.stringify(corner)}`)
      }
    },
  }
}

function lightProbeCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.03, 0.03, 0.04)

  const probe = new THREE.LightProbe()
  probe.sh.coefficients[0].set(0.7, 0.45, 0.25)
  probe.sh.coefficients[1].set(0.15, 0.05, 0.0)
  probe.intensity = 1.4
  scene.add(probe)

  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(0.72, 24, 16),
    new THREE.MeshLambertMaterial({ color: 0xffffff }),
  )
  mesh.rotation.y = -0.25
  scene.add(mesh)

  return {
    name: 'light-probe-diffuse',
    scene,
    camera: makeCamera([0.8, 0.4, 3.0], [0, 0, 0]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [8, 8, 10],
    minNonBackgroundRatio: 0.02,
    validate(rgba, { width }) {
      const center = meanRegion(rgba, width, 40, 40, 56, 56)
      if (!(center.r > 180 && center.g > 160 && center.r > center.b + 40 && center.g > center.b + 20)) {
        throw new Error(`LightProbe diffuse corpus should render a warm lit sphere, got ${JSON.stringify(center)}`)
      }
    },
  }
}

function lightProbeMaterialModelsCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const probe = new THREE.LightProbe(undefined, 1.5)
  for (const coefficient of probe.sh.coefficients) {
    coefficient.set(0, 0, 0)
  }
  probe.sh.coefficients[0].set(1.0, 0.18, 0.08)
  scene.add(probe)

  const materials = [
    new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, metalness: 0 }),
    new THREE.MeshPhysicalMaterial({ color: 0xffffff, roughness: 1, metalness: 0 }),
    new THREE.MeshPhongMaterial({ color: 0xffffff, shininess: 20 }),
    new THREE.MeshToonMaterial({ color: 0xffffff }),
  ]

  for (const [index, material] of materials.entries()) {
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(0.42, 1.2), material)
    mesh.position.x = (index - 1.5) * 0.5
    scene.add(mesh)
  }

  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  return {
    name: 'light-probe-lit-material-models',
    scene,
    camera,
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.08,
    validate(rgba, { width }) {
      const regions = [
        ['standard', meanRegion(rgba, width, 14, 30, 22, 66)],
        ['physical', meanRegion(rgba, width, 34, 30, 42, 66)],
        ['phong', meanRegion(rgba, width, 54, 30, 62, 66)],
        ['toon', meanRegion(rgba, width, 74, 30, 82, 66)],
      ]
      for (const [label, mean] of regions) {
        if (!(mean.r > mean.g + 20 && mean.r > mean.b + 20)) {
          throw new Error(`LightProbe should tint ${label} corpus material red (${mean.r}, ${mean.g}, ${mean.b})`)
        }
      }
    },
  }
}

function linearFogCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.03)
  scene.fog = new THREE.Fog(0x3366ff, 1.2, 3.2)

  const fogged = new THREE.Mesh(
    new THREE.PlaneGeometry(0.82, 1.45),
    new THREE.MeshBasicMaterial({ color: 0xff4422 }),
  )
  fogged.position.set(-0.48, 0, 0)
  scene.add(fogged)

  const unfogged = new THREE.Mesh(
    new THREE.PlaneGeometry(0.82, 1.45),
    new THREE.MeshBasicMaterial({ color: 0xff4422, fog: false }),
  )
  unfogged.position.set(0.48, 0, 0)
  scene.add(unfogged)

  return {
    name: 'linear-fog-material-opt-out',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [5, 5, 8],
    minNonBackgroundRatio: 0.08,
    validate(rgba, { width }) {
      const fogged = meanRegion(rgba, width, 20, 24, 38, 72)
      const unfogged = meanRegion(rgba, width, 58, 24, 76, 72)
      if (!(fogged.b > fogged.r + 180 && unfogged.r > unfogged.b + 180)) {
        throw new Error(`linear fog corpus should keep only the opt-out panel red, got fogged=${JSON.stringify(fogged)} unfogged=${JSON.stringify(unfogged)}`)
      }
    },
  }
}

function textureMatrixColorSpaceCorpus() {
  const texture = new THREE.DataTexture(new Uint8Array([
    64, 64, 64, 255,
    64, 64, 64, 255,
    224, 224, 224, 255,
    224, 224, 224, 255,
  ]), 2, 2, THREE.RGBAFormat)
  texture.colorSpace = THREE.SRGBColorSpace
  texture.magFilter = THREE.NearestFilter
  texture.minFilter = THREE.NearestFilter
  texture.wrapS = THREE.RepeatWrapping
  texture.wrapT = THREE.RepeatWrapping
  texture.matrixAutoUpdate = false
  texture.matrix.setUvTransform(0.12, 0.18, 1.7, 1.7, Math.PI / 2, 0.5, 0.5)
  texture.needsUpdate = true

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.025)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(1.7, 1.7),
    new THREE.MeshBasicMaterial({ color: 0xffffff, map: texture }),
  ))

  return {
    name: 'texture-matrix-srgb-map',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [5, 5, 6],
    minNonBackgroundRatio: 0.1,
    validate(rgba, { width }) {
      const transformedBright = pixelAt(rgba, width, 48, 48)
      const transformedDark = pixelAt(rgba, width, 30, 37)
      if (!(transformedBright.r > 180 && transformedBright.g > 180 && transformedBright.b > 180 && transformedDark.r < 80 && transformedDark.g < 80 && transformedDark.b < 80)) {
        throw new Error(`texture matrix corpus should sample distinct sRGB bright/dark texels, got bright=${JSON.stringify(transformedBright)} dark=${JSON.stringify(transformedDark)}`)
      }
    },
  }
}

function linearOutputColorSpaceCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.18, 0.18, 0.18)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(1.55, 1.55),
    new THREE.MeshBasicMaterial({ color: new THREE.Color(0.5, 0.22, 0.08) }),
  ))

  return {
    name: 'linear-output-color-space',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: {
      width: CORPUS_RENDER_SIZE,
      height: CORPUS_RENDER_SIZE,
      format: 'rgba',
      outputColorSpace: THREE.LinearSRGBColorSpace,
    },
    background: [46, 46, 46],
    minNonBackgroundRatio: 0.08,
    validate(rgba, { width }) {
      const center = pixelAt(rgba, width, 48, 48)
      const corner = pixelAt(rgba, width, 4, 4)
      if (!(center.r > 60 && center.r < 90 && center.g < 20 && center.b < 10 && corner.r > 40 && corner.r < 55 && corner.g > 40 && corner.g < 55 && corner.b > 40 && corner.b < 55)) {
        throw new Error(`linear output corpus should preserve linear RGB values, got center=${JSON.stringify(center)} corner=${JSON.stringify(corner)}`)
      }
    },
  }
}

function maskRenderModeCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.6, 0.05, 0.05)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(1.25, 1.25),
    new THREE.MeshBasicMaterial({ color: 0x0088ff }),
  ))

  return {
    name: 'mask-render-mode-plane',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: {
      width: CORPUS_RENDER_SIZE,
      height: CORPUS_RENDER_SIZE,
      format: 'rgba',
      renderMode: 'mask',
    },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.08,
    validate(rgba, { width }) {
      const center = meanRegion(rgba, width, 32, 32, 64, 64)
      const corner = meanRegion(rgba, width, 4, 4, 20, 20)
      if (!(center.r > 245 && center.g > 245 && center.b > 245 && corner.r < 2 && corner.g < 2 && corner.b < 2)) {
        throw new Error(`mask render corpus should render white geometry on black, got center=${JSON.stringify(center)} corner=${JSON.stringify(corner)}`)
      }
    },
  }
}

function objectIdRenderModeCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(1, 0, 0)

  const left = new THREE.Mesh(
    new THREE.PlaneGeometry(0.8, 1.0),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  )
  left.position.x = -0.5
  scene.add(left)

  const right = new THREE.Mesh(
    new THREE.PlaneGeometry(0.8, 1.0),
    new THREE.MeshBasicMaterial({ color: 0x00ff00 }),
  )
  right.position.x = 0.5
  scene.add(right)

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  return {
    name: 'object-id-render-mode-planes',
    scene,
    camera,
    options: {
      width: CORPUS_RENDER_SIZE,
      height: CORPUS_RENDER_SIZE,
      format: 'rgba',
      renderMode: 'object-id',
    },
    background: [0, 0, 0],
    backgroundTolerance: 0,
    minNonBackgroundRatio: 0.08,
    browserReference: false,
    validate(rgba, { width }) {
      const leftId = pixelAt(rgba, width, 30, 48)
      const rightId = pixelAt(rgba, width, 66, 48)
      const background = pixelAt(rgba, width, 8, 8)
      const leftCode = (leftId.r << 16) | (leftId.g << 8) | leftId.b
      const rightCode = (rightId.r << 16) | (rightId.g << 8) | rightId.b
      const backgroundCode = (background.r << 16) | (background.g << 8) | background.b
      if (!(leftCode > 0 && rightCode > 0 && leftCode !== rightCode && backgroundCode === 0)) {
        throw new Error(`object-id corpus should encode two distinct objects on zero background, got left=${leftCode} right=${rightCode} background=${backgroundCode}`)
      }
    },
  }
}

function normalRenderModeCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(1.45, 1.45),
    new THREE.MeshBasicMaterial({ color: 0xff6633 }),
  )
  mesh.rotation.y = Math.PI * 0.28
  mesh.rotation.x = -Math.PI * 0.08
  scene.add(mesh)

  return {
    name: 'normal-render-mode-tilted-plane',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: {
      width: CORPUS_RENDER_SIZE,
      height: CORPUS_RENDER_SIZE,
      format: 'rgba',
      renderMode: 'normal',
    },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.08,
    validate(rgba, { width }) {
      const center = meanRegion(rgba, width, 32, 32, 64, 64)
      const corner = meanRegion(rgba, width, 4, 4, 20, 20)
      if (!(center.r > center.g + 60 && center.b > center.g + 40 && corner.r < 2 && corner.g < 2 && corner.b < 2)) {
        throw new Error(`normal render corpus should render tilted view-normal colors on black, got center=${JSON.stringify(center)} corner=${JSON.stringify(corner)}`)
      }
    },
  }
}

function twoDimensionalBackgroundTextureCorpus() {
  const background = new THREE.DataTexture(new Uint8Array([
    255, 48, 32, 255,
    32, 200, 96, 255,
    48, 80, 255, 255,
    255, 225, 72, 255,
  ]), 2, 2, THREE.RGBAFormat)
  background.colorSpace = THREE.SRGBColorSpace
  background.wrapS = THREE.RepeatWrapping
  background.wrapT = THREE.RepeatWrapping
  background.repeat.set(2, 1)
  background.offset.set(0.25, 0)
  background.magFilter = THREE.NearestFilter
  background.minFilter = THREE.NearestFilter
  background.needsUpdate = true

  const scene = new THREE.Scene()
  scene.background = background
  scene.backgroundIntensity = 0.85

  return {
    name: 'two-dimensional-background-texture-transform',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.95,
    validate(rgba, { width }) {
      const top = meanRegion(rgba, width, 4, 4, 24, 24)
      const bottom = meanRegion(rgba, width, 4, 72, 24, 92)
      if (!(top.r > top.b + 20 && bottom.g > bottom.b + 40 && Math.abs(top.r - bottom.r) > 30)) {
        throw new Error(`2D background corpus should show transformed repeated texture colors, got top=${JSON.stringify(top)} bottom=${JSON.stringify(bottom)}`)
      }
    },
  }
}

function spriteMaterialCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.025, 0.03)

  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    color: 0xffffff,
    map: spriteMapTexture(),
    opacity: 0.85,
    transparent: true,
    rotation: 0.3,
  }))
  sprite.center.set(0.4, 0.55)
  sprite.scale.set(1.25, 0.9, 1)
  scene.add(sprite)

  return {
    name: 'sprite-material-map-billboard',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [5, 6, 8],
    minNonBackgroundRatio: 0.02,
    validate(rgba, { width, height }) {
      const colors = {
        red: countRegionPixels(rgba, width, 0, 0, width, height, (r, g, b) => r > 150 && r > g + 40 && r > b + 40),
        green: countRegionPixels(rgba, width, 0, 0, width, height, (r, g, b) => g > 120 && g > r + 40 && g > b + 20),
        blue: countRegionPixels(rgba, width, 0, 0, width, height, (r, g, b) => b > 150 && b > r + 40 && b > g + 40),
        yellow: countRegionPixels(rgba, width, 0, 0, width, height, (r, g, b) => r > 150 && g > 120 && r > b + 40 && g > b + 40),
      }
      if (colors.red < 100 || colors.green < 100 || colors.blue < 100 || colors.yellow < 100) {
        throw new Error(`mapped sprite corpus should render all texture quadrants, got ${JSON.stringify(colors)}`)
      }
    },
  }
}

function spriteShadowCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(1, 1, 1)

  const receiver = new THREE.Mesh(
    new THREE.PlaneGeometry(12, 12),
    new THREE.ShadowMaterial({ opacity: 1 }),
  )
  receiver.rotation.x = -Math.PI / 2
  receiver.receiveShadow = true
  scene.add(receiver)

  const caster = new THREE.Sprite(new THREE.SpriteMaterial({ color: 0xffffff }))
  caster.position.set(-1.6, 4, 0)
  caster.scale.set(4, 4, 1)
  caster.castShadow = true
  scene.add(caster)

  const light = new THREE.DirectionalLight(0xffffff, 2)
  light.position.set(0, 6, 8)
  light.target.position.set(0, 0, 0)
  light.castShadow = true
  light.shadow.mapSize.set(256, 256)
  light.shadow.camera.left = -7
  light.shadow.camera.right = 7
  light.shadow.camera.top = 7
  light.shadow.camera.bottom = -7
  light.shadow.camera.near = 0.1
  light.shadow.camera.far = 16
  scene.add(light, light.target)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 6, 8)
  camera.lookAt(0, 0, 0)

  return {
    name: 'sprite-shadow-caster',
    scene,
    camera,
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [255, 255, 255],
    minNonBackgroundRatio: 0.05,
    validate(rgba, { width }) {
      const shadowed = meanRegion(rgba, width, 18, 22, 42, 46)
      const lit = meanRegion(rgba, width, 70, 22, 90, 46)
      const shadowedLum = shadowed.r + shadowed.g + shadowed.b
      const litLum = lit.r + lit.g + lit.b
      if (!(shadowedLum < litLum - 120)) {
        throw new Error(`sprite shadow corpus should darken the receiver (${shadowedLum} vs ${litLum})`)
      }
    },
  }
}

function pointSpotLightCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.025, 0.03)
  scene.add(new THREE.AmbientLight(0xffffff, 0.08))

  const material = new THREE.MeshLambertMaterial({ color: 0xffffff })
  const left = new THREE.Mesh(new THREE.SphereGeometry(0.42, 24, 16), material)
  left.position.x = -0.45
  scene.add(left)

  const right = new THREE.Mesh(new THREE.SphereGeometry(0.42, 24, 16), material.clone())
  right.position.x = 0.45
  scene.add(right)

  const point = new THREE.PointLight(0xff5533, 6, 4, 2)
  point.position.set(-1.2, 0.75, 1.5)
  scene.add(point)

  const spot = new THREE.SpotLight(0x44aaff, 7, 4, Math.PI / 5, 0.25, 2)
  spot.position.set(1.1, 1.1, 1.8)
  spot.target.position.set(0.35, 0, 0)
  scene.add(spot, spot.target)

  return {
    name: 'point-spot-light-materials',
    scene,
    camera: makeCamera([0, 0.2, 3.1], [0, 0, 0]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [5, 6, 8],
    minNonBackgroundRatio: 0.02,
    validate(rgba, { width }) {
      const pointLit = meanRegion(rgba, width, 20, 28, 42, 62)
      const spotLit = meanRegion(rgba, width, 54, 28, 76, 62)
      if (!(pointLit.r > pointLit.g + 25 && spotLit.b > spotLit.r + 8 && spotLit.b > spotLit.g + 15)) {
        throw new Error(`point/spot corpus should tint the two spheres red and blue, got point=${JSON.stringify(pointLit)} spot=${JSON.stringify(spotLit)}`)
      }
    },
  }
}

function rectAreaLightCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(1.75, 1.75),
    new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, metalness: 0 }),
  ))

  const light = new THREE.RectAreaLight(0xffddaa, 18, 2.8, 1.4)
  light.position.set(0, 0, 2)
  light.lookAt(0, 0, 0)
  scene.add(light)

  return {
    name: 'rect-area-light-plane',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.18,
    validate(rgba, { width }) {
      const center = meanRegion(rgba, width, 32, 32, 64, 64)
      if (!(center.r > 220 && center.g > 210 && center.b > 190 && center.r > center.b + 10)) {
        throw new Error(`RectAreaLight corpus should render a warm lit plane, got ${JSON.stringify(center)}`)
      }
    },
  }
}

function globalClippingPlaneCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0.12)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xff5533 }),
  ))

  return {
    name: 'global-clipping-plane',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: {
      width: CORPUS_RENDER_SIZE,
      height: CORPUS_RENDER_SIZE,
      format: 'rgba',
      clippingPlanes: [new THREE.Plane(new THREE.Vector3(1, 0, 0), 0)],
    },
    background: [0, 0, 31],
    minNonBackgroundRatio: 0.05,
    validate(rgba, { width }) {
      const clipped = meanRegion(rgba, width, 16, 32, 40, 64)
      const visible = meanRegion(rgba, width, 56, 32, 80, 64)
      if (!(clipped.r < 5 && clipped.g < 5 && clipped.b > 25 && visible.r > visible.g + 180 && visible.r > visible.b + 180)) {
        throw new Error(`global clipping corpus should keep only the red right half, got clipped=${JSON.stringify(clipped)} visible=${JSON.stringify(visible)}`)
      }
    },
  }
}

function materialLocalClippingCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.08)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({
      color: 0x22ccff,
      clippingPlanes: [new THREE.Plane(new THREE.Vector3(0, 1, 0), 0)],
    }),
  ))

  return {
    name: 'material-local-clipping-plane',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: {
      width: CORPUS_RENDER_SIZE,
      height: CORPUS_RENDER_SIZE,
      format: 'rgba',
      localClippingEnabled: true,
    },
    background: [5, 5, 20],
    minNonBackgroundRatio: 0.05,
    validate(rgba, { width }) {
      const visible = meanRegion(rgba, width, 32, 16, 64, 40)
      const clipped = meanRegion(rgba, width, 32, 56, 64, 80)
      if (!(visible.b > visible.r + 120 && visible.g > visible.r + 80 && clipped.r < 10 && clipped.g < 10 && clipped.b < 25)) {
        throw new Error(`local clipping corpus should keep only the cyan top half, got visible=${JSON.stringify(visible)} clipped=${JSON.stringify(clipped)}`)
      }
    },
  }
}

function nestedClippingGroupCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0.12)

  const parent = new THREE.Group()
  parent.isClippingGroup = true
  parent.clippingPlanes = [new THREE.Plane(new THREE.Vector3(1, 0, 0), 0)]

  const child = new THREE.Group()
  child.isClippingGroup = true
  child.clipIntersection = true
  child.clippingPlanes = [
    new THREE.Plane(new THREE.Vector3(0, 1, 0), 0),
    new THREE.Plane(new THREE.Vector3(-1, 0, 0), 0),
  ]

  child.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xff5533 }),
  ))
  parent.add(child)
  scene.add(parent)

  return {
    name: 'nested-clipping-groups',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 31],
    minNonBackgroundRatio: 0.04,
    browserReference: false,
    validate(rgba, { width }) {
      const visible = pixelAt(rgba, width, 64, 32)
      const clippedLeft = pixelAt(rgba, width, 32, 32)
      const clippedBottom = pixelAt(rgba, width, 64, 64)
      if (!(visible.r > visible.g + 180 && visible.r > visible.b + 180 && clippedLeft.r < 5 && clippedLeft.g < 5 && clippedLeft.b > 25 && clippedBottom.r < 5 && clippedBottom.g < 5 && clippedBottom.b > 25)) {
        throw new Error(`nested clipping corpus should keep only the red upper-right quadrant, got visible=${JSON.stringify(visible)} clippedLeft=${JSON.stringify(clippedLeft)} clippedBottom=${JSON.stringify(clippedBottom)}`)
      }
    },
  }
}

function materialEnvMapCorpus() {
  const envMap = new THREE.DataTexture(new Uint8Array([
    40, 220, 120, 255,
    40, 220, 120, 255,
  ]), 2, 1, THREE.RGBAFormat)
  envMap.mapping = THREE.EquirectangularReflectionMapping
  envMap.needsUpdate = true

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.04, 0.04, 0.05)
  scene.add(new THREE.AmbientLight(0xffffff, 0.25))
  const key = new THREE.DirectionalLight(0xffffff, 1.1)
  key.position.set(2, 3, 4)
  scene.add(key)

  scene.add(new THREE.Mesh(
    new THREE.SphereGeometry(0.7, 24, 16),
    new THREE.MeshPhongMaterial({
      color: 0x884433,
      envMap,
      combine: THREE.MixOperation,
      reflectivity: 0.65,
      shininess: 48,
    }),
  ))

  return {
    name: 'material-env-map-phong',
    scene,
    camera: makeCamera([0.8, 0.35, 3.0], [0, 0, 0]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [10, 10, 13],
    minNonBackgroundRatio: 0.02,
    validate(rgba, { width }) {
      const center = meanRegion(rgba, width, 32, 32, 64, 64)
      if (!(center.g > center.r + 60 && center.g > center.b + 25)) {
        throw new Error(`Phong material envMap corpus should render green reflected IBL, got ${JSON.stringify(center)}`)
      }
    },
  }
}

function materialEnvMapBasicLambertCorpus() {
  const envMap = new THREE.DataTexture(new Uint8Array([
    40, 220, 120, 255,
    40, 220, 120, 255,
  ]), 2, 1, THREE.RGBAFormat)
  envMap.mapping = THREE.EquirectangularReflectionMapping
  envMap.needsUpdate = true

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.025, 0.025, 0.03)

  const basic = new THREE.Mesh(
    new THREE.PlaneGeometry(0.8, 1.1),
    new THREE.MeshBasicMaterial({
      color: 0xaa3322,
      envMap,
      combine: THREE.MixOperation,
      reflectivity: 0.9,
    }),
  )
  basic.position.x = -0.48
  scene.add(basic)

  const lambert = new THREE.Mesh(
    new THREE.PlaneGeometry(0.8, 1.1),
    new THREE.MeshLambertMaterial({
      color: 0xaa3322,
      envMap,
      combine: THREE.MixOperation,
      reflectivity: 0.9,
    }),
  )
  lambert.position.x = 0.48
  scene.add(lambert)

  const key = new THREE.DirectionalLight(0xffffff, 1.4)
  key.position.set(0, 0, 3)
  scene.add(key)

  return {
    name: 'material-env-map-basic-lambert',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [6, 6, 8],
    minNonBackgroundRatio: 0.06,
    validate(rgba, { width }) {
      const basic = meanRegion(rgba, width, 20, 30, 40, 66)
      const lambert = meanRegion(rgba, width, 56, 30, 76, 66)
      for (const [label, mean] of [
        ['basic', basic],
        ['lambert', lambert],
      ]) {
        if (!(mean.g > mean.r + 45 && mean.g > mean.b + 20)) {
          throw new Error(`shared material envMap corpus should render green ${label} IBL, got ${JSON.stringify(mean)}`)
        }
      }
    },
  }
}

function materialEnvMapPbrCorpus() {
  const envMap = new THREE.DataTexture(new Uint8Array([
    0, 255, 0, 255,
    0, 255, 0, 255,
  ]), 2, 1, THREE.RGBAFormat)
  envMap.mapping = THREE.EquirectangularReflectionMapping
  envMap.needsUpdate = true

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.025)

  const standard = new THREE.Mesh(
    new THREE.PlaneGeometry(0.8, 1.1),
    new THREE.MeshStandardMaterial({
      color: 0xffffff,
      metalness: 1,
      roughness: 0.12,
      envMap,
      envMapIntensity: 2,
    }),
  )
  standard.position.x = -0.48
  scene.add(standard)

  const physical = new THREE.Mesh(
    new THREE.PlaneGeometry(0.8, 1.1),
    new THREE.MeshPhysicalMaterial({
      color: 0xffffff,
      metalness: 1,
      roughness: 0.16,
      clearcoat: 0.45,
      envMap,
      envMapIntensity: 2,
    }),
  )
  physical.position.x = 0.48
  scene.add(physical)

  return {
    name: 'material-env-map-pbr',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [5, 5, 6],
    minNonBackgroundRatio: 0.06,
    validate(rgba, { width }) {
      const standardMean = meanRegion(rgba, width, 20, 30, 40, 66)
      const physicalMean = meanRegion(rgba, width, 56, 30, 76, 66)
      for (const [label, mean] of [
        ['standard', standardMean],
        ['physical', physicalMean],
      ]) {
        if (!(mean.g > mean.r + 20 && mean.g > mean.b + 8)) {
          throw new Error(`PBR material envMap should render green ${label} IBL (${mean.r}, ${mean.g}, ${mean.b})`)
        }
      }
    },
  }
}

function meshBasicMaterialWireframeCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(1.7, 1.7, 4, 4),
    new THREE.MeshBasicMaterial({ color: 0xffdd66, wireframe: true }),
  ))

  return {
    name: 'mesh-basic-material-wireframe',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.004,
    validate(rgba, { width, height }) {
      const yellowPixels = countRegionPixels(rgba, width, 0, 0, width, height, (r, g, b) => r > 120 && g > 100 && b < 120)
      const center = pixelAt(rgba, width, 48, 48)
      if (!(yellowPixels > 700 && yellowPixels < 1200 && center.r < 5 && center.g < 5 && center.b < 5)) {
        throw new Error(`basic wireframe corpus should render sparse yellow grid lines, got yellow=${yellowPixels} center=${JSON.stringify(center)}`)
      }
    },
  }
}

function meshDepthMaterialCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const material = new THREE.MeshDepthMaterial({ depthPacking: THREE.BasicDepthPacking })
  scene.add(new THREE.Mesh(
    new THREE.SphereGeometry(0.72, 24, 16),
    material,
  ))
  const camera = new THREE.PerspectiveCamera(45, 1, 0.5, 4)
  camera.position.set(0.7, 0.25, 3.0)
  camera.lookAt(0, 0, 0)

  return {
    name: 'mesh-depth-material-basic',
    scene,
    camera,
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.02,
    validate(rgba, { width }) {
      const center = pixelAt(rgba, width, 48, 48)
      const corner = pixelAt(rgba, width, 4, 4)
      if (!(center.r > 18 && center.r < 35 && Math.abs(center.r - center.g) <= 1 && Math.abs(center.r - center.b) <= 1 && corner.r === 0 && corner.g === 0 && corner.b === 0)) {
        throw new Error(`depth material corpus should render a low grayscale depth sphere, got center=${JSON.stringify(center)} corner=${JSON.stringify(corner)}`)
      }
    },
  }
}

function meshDepthMaterialWireframeCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(0.58, 0.58, 4, 4),
    new THREE.MeshDepthMaterial({ depthPacking: THREE.BasicDepthPacking, wireframe: true }),
  )
  mesh.position.z = 2.25
  scene.add(mesh)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 4)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  return {
    name: 'mesh-depth-material-wireframe',
    scene,
    camera,
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.004,
    validate(rgba, { width, height }) {
      const grayPixels = countRegionPixels(rgba, width, 0, 0, width, height, (r, g, b) => r > 2 && Math.abs(r - g) <= 1 && Math.abs(r - b) <= 1)
      const center = pixelAt(rgba, width, 48, 48)
      if (!(grayPixels > 900 && grayPixels < 1600 && center.r === 0 && center.g === 0 && center.b === 0)) {
        throw new Error(`depth wireframe corpus should render sparse grayscale depth lines, got gray=${grayPixels} center=${JSON.stringify(center)}`)
      }
    },
  }
}

function meshDistanceMaterialCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const material = new THREE.MeshDistanceMaterial()
  material.referencePosition = new THREE.Vector3(0, 0, -4)
  material.nearDistance = 0
  material.farDistance = 7

  const near = new THREE.Mesh(new THREE.PlaneGeometry(0.8, 1.1), material)
  near.position.set(-0.5, 0, -3.6)
  scene.add(near)

  const far = new THREE.Mesh(new THREE.PlaneGeometry(0.8, 1.1), material.clone())
  far.material.referencePosition = material.referencePosition.clone()
  far.material.nearDistance = material.nearDistance
  far.material.farDistance = material.farDistance
  far.position.set(0.5, 0, 1.8)
  scene.add(far)

  return {
    name: 'mesh-distance-material-range',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.08,
    browserReference: false,
    validate(rgba, { width }) {
      const near = meanRegion(rgba, width, 16, 36, 36, 60)
      const far = meanRegion(rgba, width, 60, 36, 80, 60)
      if (!(near.r < 30 && near.g < 5 && near.b < 5 && far.r > 180 && far.g < 5 && far.b < 5)) {
        throw new Error(`distance material corpus should render the far plane bright red and near plane dark, got near=${JSON.stringify(near)} far=${JSON.stringify(far)}`)
      }
    },
  }
}

function meshDistanceMaterialWireframeCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const material = new THREE.MeshDistanceMaterial()
  material.wireframe = true
  material.referencePosition = new THREE.Vector3(0, 0, 3)
  material.nearDistance = 0
  material.farDistance = 4

  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(1.7, 1.7, 4, 4),
    material,
  ))

  return {
    name: 'mesh-distance-material-wireframe',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.004,
    browserReference: false,
    validate(rgba, { width, height }) {
      const redPixels = countRegionPixels(rgba, width, 0, 0, width, height, (r, g, b) => r > 120 && g < 5 && b < 5)
      const center = pixelAt(rgba, width, 48, 48)
      if (!(redPixels > 700 && redPixels < 1200 && center.r === 0 && center.g === 0 && center.b === 0)) {
        throw new Error(`distance wireframe corpus should render sparse red distance lines, got red=${redPixels} center=${JSON.stringify(center)}`)
      }
    },
  }
}

function meshNormalMaterialCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.015, 0.015, 0.02)
  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(0.95, 0.95, 0.95),
    new THREE.MeshNormalMaterial({ flatShading: true }),
  )
  mesh.rotation.set(0.25, -0.55, 0.18)
  scene.add(mesh)

  return {
    name: 'mesh-normal-material-flat',
    scene,
    camera: makeCamera([1.1, 0.7, 3.0], [0, 0, 0]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [4, 4, 5],
    minNonBackgroundRatio: 0.04,
    validate(rgba, { width, height }) {
      const blueFaces = countRegionPixels(rgba, width, 0, 0, width, height, (r, g, b) => b > r + 40 && b > g + 40 && b > 100)
      const greenFaces = countRegionPixels(rgba, width, 0, 0, width, height, (r, g, b) => g > r + 40 && g > b + 40 && g > 100)
      if (blueFaces < 300 || greenFaces < 80) {
        throw new Error(`normal material corpus should render distinct normal-color faces, got blue=${blueFaces} green=${greenFaces}`)
      }
    },
  }
}

function meshMatcapMaterialCorpus() {
  const matcap = new THREE.DataTexture(new Uint8Array([
    40, 70, 130, 255,
    245, 210, 140, 255,
    90, 170, 210, 255,
    255, 255, 240, 255,
  ]), 2, 2, THREE.RGBAFormat)
  matcap.colorSpace = THREE.SRGBColorSpace
  matcap.magFilter = THREE.LinearFilter
  matcap.minFilter = THREE.LinearFilter
  matcap.needsUpdate = true

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.025, 0.03)
  scene.add(new THREE.Mesh(
    new THREE.SphereGeometry(0.72, 24, 16),
    new THREE.MeshMatcapMaterial({
      color: 0xffffff,
      matcap,
    }),
  ))

  return {
    name: 'mesh-matcap-material-map',
    scene,
    camera: makeCamera([0.8, 0.35, 3.0], [0, 0, 0]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [5, 6, 8],
    minNonBackgroundRatio: 0.02,
    validate(rgba, { width }) {
      const center = meanRegion(rgba, width, 32, 32, 64, 64)
      if (!(center.b > center.r + 20 && center.g > center.r + 10)) {
        throw new Error(`matcap corpus should sample the blue-green matcap blend, got ${JSON.stringify(center)}`)
      }
    },
  }
}

function meshToonMaterialCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.03, 0.025, 0.035)
  scene.add(new THREE.AmbientLight(0xffffff, 0.12))
  const key = new THREE.DirectionalLight(0xffffff, 1.8)
  key.position.set(2, 3, 3)
  scene.add(key)

  scene.add(new THREE.Mesh(
    new THREE.SphereGeometry(0.72, 24, 16),
    new THREE.MeshToonMaterial({
      color: 0x66ccff,
      gradientMap: gradientTexture(),
    }),
  ))

  return {
    name: 'mesh-toon-gradient-map',
    scene,
    camera: makeCamera([0.8, 0.35, 3.0], [0, 0, 0]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [8, 6, 9],
    minNonBackgroundRatio: 0.02,
    validate(rgba, { width }) {
      const center = meanRegion(rgba, width, 32, 32, 64, 64)
      if (!(center.b > center.g + 40 && center.g > center.r + 80)) {
        throw new Error(`toon gradient corpus should sample the blue-green ramp, got ${JSON.stringify(center)}`)
      }
    },
  }
}

function meshToonAlphaMapCorpus() {
  const alphaMap = new THREE.DataTexture(new Uint8Array([
    255, 0, 255, 255,
    255, 255, 255, 255,
  ]), 2, 1, THREE.RGBAFormat)
  alphaMap.magFilter = THREE.NearestFilter
  alphaMap.minFilter = THREE.NearestFilter
  alphaMap.needsUpdate = true

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0.16)
  scene.add(new THREE.AmbientLight(0xffffff, 0.2))
  const key = new THREE.DirectionalLight(0xffffff, 2.2)
  key.position.set(0, 0, 3)
  scene.add(key)

  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(1.8, 1.5),
    new THREE.MeshToonMaterial({
      color: 0xff4422,
      alphaMap,
      alphaTest: 0.5,
    }),
  ))

  return {
    name: 'mesh-toon-alpha-map-cutout',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 41],
    minNonBackgroundRatio: 0.04,
    validate(rgba, { width }) {
      const cutout = meanRegion(rgba, width, 12, 24, 40, 72)
      const visible = meanRegion(rgba, width, 56, 24, 84, 72)
      if (!(cutout.b > cutout.r + 30 && visible.r > visible.b + 60)) {
        throw new Error(`toon alpha corpus should cut out the left side and keep the right side red, got cutout=${JSON.stringify(cutout)} visible=${JSON.stringify(visible)}`)
      }
    },
  }
}

function viewportScissorCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0.12)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(4, 4),
    new THREE.MeshBasicMaterial({ color: 0xffcc22 }),
  ))

  return {
    name: 'viewport-scissor-rectangle',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: {
      width: CORPUS_RENDER_SIZE,
      height: CORPUS_RENDER_SIZE,
      format: 'rgba',
      viewport: { x: 24, y: 16, width: 48, height: 56 },
      scissor: { x: 36, y: 28, width: 24, height: 24 },
    },
    background: [0, 0, 31],
    validate(rgba, { width }) {
      const filled = pixelAt(rgba, width, 48, 40)
      const outsideScissor = pixelAt(rgba, width, 32, 32)
      const outsideViewport = pixelAt(rgba, width, 8, 8)
      if (!(filled.r > 200 && filled.g > 120 && filled.b < 90 && outsideScissor.r < 5 && outsideScissor.g < 5 && outsideScissor.b > 25 && outsideViewport.r < 5 && outsideViewport.g < 5 && outsideViewport.b > 25)) {
        throw new Error(`viewport/scissor corpus should fill only the yellow clipped rectangle, got filled=${JSON.stringify(filled)} outsideScissor=${JSON.stringify(outsideScissor)} outsideViewport=${JSON.stringify(outsideViewport)}`)
      }
    },
  }
}

function arrayCameraViewportCorpus() {
  const width = CORPUS_RENDER_SIZE
  const height = CORPUS_RENDER_SIZE
  const leftCamera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  leftCamera.position.set(0, 0, 3)
  leftCamera.lookAt(0, 0, 0)
  leftCamera.layers.set(1)
  leftCamera.viewport = new THREE.Vector4(0, 0, width / 2, height)

  const rightCamera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  rightCamera.position.set(0, 0, 3)
  rightCamera.lookAt(0, 0, 0)
  rightCamera.layers.set(2)
  rightCamera.viewport = new THREE.Vector4(width / 2, 0, width / 2, height)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0.08)

  const red = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xff3333 }),
  )
  red.layers.set(1)
  scene.add(red)

  const green = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0x33ff66 }),
  )
  green.layers.set(2)
  scene.add(green)

  return {
    name: 'array-camera-viewport-split',
    scene,
    camera: new THREE.ArrayCamera([leftCamera, rightCamera]),
    options: { width, height, format: 'rgba' },
    background: [0, 0, 20],
    validate(rgba, { width }) {
      const left = meanRegion(rgba, width, 16, 32, 40, 64)
      const right = meanRegion(rgba, width, 56, 32, 80, 64)
      if (!(left.r > left.g + 180 && left.r > left.b + 180 && right.g > right.r + 70 && right.g > right.b + 100)) {
        throw new Error(`ArrayCamera corpus should render red left and green right viewports, got left=${JSON.stringify(left)} right=${JSON.stringify(right)}`)
      }
    },
  }
}

function cubeCameraCaptureCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const addPlane = (position, rotation, color) => {
    const plane = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide }),
    )
    plane.position.set(position[0], position[1], position[2])
    plane.rotation.set(rotation[0], rotation[1], rotation[2])
    scene.add(plane)
  }
  addPlane([2, 0, 0], [0, Math.PI / 2, 0], 0xff0000)
  addPlane([-2, 0, 0], [0, Math.PI / 2, 0], 0x00ff00)
  addPlane([0, 2, 0], [Math.PI / 2, 0, 0], 0x0000ff)
  addPlane([0, -2, 0], [Math.PI / 2, 0, 0], 0xffff00)
  addPlane([0, 0, 2], [0, 0, 0], 0xff00ff)
  addPlane([0, 0, -2], [0, 0, 0], 0x00ffff)

  const target = new THREE.WebGLCubeRenderTarget(CORPUS_RENDER_SIZE)
  const camera = new THREE.CubeCamera(0.01, 100, target)

  return {
    name: 'cube-camera-face-capture',
    scene,
    camera,
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.08,
    browserReference: false,
    validate(rgba, { width }) {
      const center = pixelAt(rgba, width, 48, 48)
      const corner = pixelAt(rgba, width, 4, 4)
      if (!(center.r > center.g + 180 && center.r > center.b + 180 && corner.r === 0 && corner.g === 0 && corner.b === 0)) {
        throw new Error(`cube camera corpus should capture the red +X face into output, got center=${JSON.stringify(center)} corner=${JSON.stringify(corner)}`)
      }
    },
  }
}

function equirectangularBackgroundCorpus() {
  const background = new THREE.DataTexture(new Uint8Array([
    255, 48, 32, 255,
    255, 48, 32, 255,
    255, 48, 32, 255,
    255, 48, 32, 255,
    32, 200, 96, 255,
    32, 200, 96, 255,
    32, 200, 96, 255,
    32, 200, 96, 255,
  ]), 8, 1, THREE.RGBAFormat)
  background.mapping = THREE.EquirectangularReflectionMapping
  background.magFilter = THREE.NearestFilter
  background.minFilter = THREE.NearestFilter
  background.needsUpdate = true

  const scene = new THREE.Scene()
  scene.background = background
  scene.backgroundIntensity = 0.85
  scene.backgroundRotation = new THREE.Euler(0, Math.PI, 0)

  return {
    name: 'equirectangular-background-rotation',
    scene,
    camera: makeCamera([0, 0, 0], [0, 0, -1]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.95,
    validate(rgba, { width }) {
      const center = meanRegion(rgba, width, 32, 32, 64, 64)
      if (!(center.g > center.r + 100 && center.g > center.b + 40)) {
        throw new Error(`equirectangular background rotation corpus should sample the green half, got ${JSON.stringify(center)}`)
      }
    },
  }
}

function cubeBackgroundTextureCorpus() {
  const scene = new THREE.Scene()
  scene.background = cubeTexture([
    [48, 80, 255],
    [255, 225, 72],
    [255, 64, 220],
    [32, 210, 220],
    [32, 200, 96],
    [255, 48, 32],
  ])
  scene.background.magFilter = THREE.NearestFilter
  scene.background.minFilter = THREE.NearestFilter
  scene.backgroundRotation = new THREE.Euler(0, Math.PI, 0)

  return {
    name: 'cube-background-texture-rotation',
    scene,
    camera: makeCamera([0, 0, 0], [0, 0, -1]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.95,
    browserReference: false,
    validate(rgba, { width }) {
      const center = meanRegion(rgba, width, 32, 32, 64, 64)
      if (!(center.g > center.r + 100 && center.g > center.b + 40)) {
        throw new Error(`cube background rotation corpus should sample the green face, got ${JSON.stringify(center)}`)
      }
    },
  }
}

function customSortGroupCorpus() {
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.9, -0.9, 0,
    0.9, -0.9, 0,
    0.9, 0.9, 0,
    -0.9, -0.9, 0,
    0.9, 0.9, 0,
    -0.9, 0.9, 0,
    -0.9, -0.9, 0,
    0.9, -0.9, 0,
    0.9, 0.9, 0,
    -0.9, -0.9, 0,
    0.9, 0.9, 0,
    -0.9, 0.9, 0,
  ]), 3))
  geometry.addGroup(0, 6, 0)
  geometry.addGroup(6, 6, 1)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(geometry, [
    new THREE.MeshBasicMaterial({ color: 0xff3344, depthTest: false }),
    new THREE.MeshBasicMaterial({ color: 0x2266ff, depthTest: false }),
  ]))

  return {
    name: 'custom-opaque-sort-group-items',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: {
      width: CORPUS_RENDER_SIZE,
      height: CORPUS_RENDER_SIZE,
      format: 'rgba',
      opaqueSort: (a, b) => b.group.materialIndex - a.group.materialIndex,
    },
    background: [0, 0, 0],
    validate(rgba, { width }) {
      const center = pixelAt(rgba, width, 48, 48)
      const corner = pixelAt(rgba, width, 4, 4)
      if (!(center.r > center.b + 180 && center.r > center.g + 180 && corner.r === 0 && corner.g === 0 && corner.b === 0)) {
        throw new Error(`custom sort corpus should draw the red group last on black background, got center=${JSON.stringify(center)} corner=${JSON.stringify(corner)}`)
      }
    },
  }
}

function skinnedMorphCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.05, 0.06, 0.08)
  addBasicLights(scene)

  const geometry = new THREE.PlaneGeometry(1, 1, 1, 1)
  const count = geometry.getAttribute('position').count
  geometry.setAttribute('skinIndex', new THREE.BufferAttribute(new Uint16Array(count * 4), 4))
  const skinWeights = new Float32Array(count * 4)
  for (let i = 0; i < count; i += 1) {
    skinWeights[i * 4] = 1
  }
  geometry.setAttribute('skinWeight', new THREE.BufferAttribute(skinWeights, 4))
  geometry.morphTargetsRelative = true
  geometry.morphAttributes.position = [
    new THREE.BufferAttribute(new Float32Array([
      0, 0, 0,
      0.15, 0, 0,
      0, 0.2, 0,
      0.15, 0.2, 0,
    ]), 3),
  ]

  const material = new THREE.MeshStandardMaterial({ color: 0x77ccff, roughness: 0.55, metalness: 0.05 })
  const mesh = new THREE.SkinnedMesh(geometry, material)
  const bone = new THREE.Bone()
  mesh.add(bone)
  const skeleton = new THREE.Skeleton([bone])
  mesh.bind(skeleton)
  mesh.morphTargetInfluences = [0.6]
  bone.position.set(0.12, 0.05, 0)
  bone.updateMatrixWorld(true)
  mesh.rotation.y = -0.25
  scene.add(mesh)

  return {
    name: 'skinned-morphed-plane',
    scene,
    camera: makeCamera([0.2, 0.1, 2.5]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [13, 15, 20],
    validate(rgba, { width }) {
      const center = pixelAt(rgba, width, 48, 48)
      const corner = pixelAt(rgba, width, 4, 4)
      if (!(center.b > center.g + 50 && center.g > 80 && center.r < 20 && corner.r === 13 && corner.g === 15 && corner.b === 20)) {
        throw new Error(`skinned morph corpus should render the deformed cyan plane over background, got center=${JSON.stringify(center)} corner=${JSON.stringify(corner)}`)
      }
    },
  }
}

function avatarLikeCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.06, 0.07, 0.1)
  scene.environment = environmentTexture()
  scene.environmentIntensity = 0.8
  scene.fog = new THREE.Fog(0x111827, 3.5, 7)
  scene.add(new THREE.HemisphereLight(0xbfd7ff, 0x443322, 0.6))

  const key = new THREE.DirectionalLight(0xffffff, 1.1)
  key.position.set(2.5, 3.5, 2)
  key.target.position.set(0, 0.45, 0)
  scene.add(key, key.target)

  const bodyGeometry = new THREE.BoxGeometry(0.58, 1.24, 0.28, 1, 2, 1)
  const position = bodyGeometry.getAttribute('position')
  const vertexCount = position.count
  const skinIndex = new Uint16Array(vertexCount * 4)
  const skinWeight = new Float32Array(vertexCount * 4)
  const morph = new Float32Array(vertexCount * 3)
  for (let i = 0; i < vertexCount; i += 1) {
    const y = position.getY(i)
    const topWeight = Math.max(0, Math.min(1, (y + 0.15) / 0.9))
    skinIndex[i * 4] = 0
    skinIndex[i * 4 + 1] = 1
    skinWeight[i * 4] = 1 - topWeight
    skinWeight[i * 4 + 1] = topWeight
    if (y > 0.15) {
      morph[i * 3] = position.getX(i) * 0.08
      morph[i * 3 + 1] = 0.04
    }
  }
  bodyGeometry.setAttribute('skinIndex', new THREE.BufferAttribute(skinIndex, 4))
  bodyGeometry.setAttribute('skinWeight', new THREE.BufferAttribute(skinWeight, 4))
  bodyGeometry.morphTargetsRelative = true
  bodyGeometry.morphAttributes.position = [new THREE.BufferAttribute(morph, 3)]

  const body = new THREE.SkinnedMesh(bodyGeometry, new THREE.MeshToonMaterial({
    color: 0x8fc7ff,
    gradientMap: gradientTexture(),
  }))
  const hips = new THREE.Bone()
  hips.name = 'hips'
  hips.position.y = -0.55
  const chest = new THREE.Bone()
  chest.name = 'chest'
  chest.position.y = 0.85
  chest.rotation.z = -0.12
  hips.add(chest)
  body.add(hips)
  body.bind(new THREE.Skeleton([hips, chest]))
  body.morphTargetInfluences = [0.55]
  body.rotation.y = -0.25
  scene.add(body)

  const head = new THREE.Mesh(
    new THREE.SphereGeometry(0.34, 20, 12),
    new THREE.MeshPhongMaterial({
      color: 0xffd8b8,
      specular: 0x222222,
      shininess: 24,
    }),
  )
  head.position.set(0, 0.88, 0.02)
  head.rotation.y = -0.25
  scene.add(head)

  const hair = new THREE.Mesh(
    new THREE.SphereGeometry(0.38, 16, 10, 0, Math.PI * 2, 0, Math.PI * 0.62),
    new THREE.MeshBasicMaterial({
      color: 0x2f2448,
      transparent: true,
      opacity: 0.78,
      side: THREE.DoubleSide,
      alphaHash: true,
    }),
  )
  hair.position.set(0, 0.98, -0.02)
  hair.rotation.y = -0.25
  scene.add(hair)

  const eyeGeometry = new THREE.BufferGeometry()
  eyeGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.12, 0.92, 0.34,
    0.12, 0.92, 0.34,
  ]), 3))
  scene.add(new THREE.Points(eyeGeometry, new THREE.PointsMaterial({
    color: 0x102033,
    size: 5,
    sizeAttenuation: false,
  })))

  const outline = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(0.66, 1.32, 0.34)),
    new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.35 }),
  )
  outline.position.y = 0.02
  outline.rotation.y = -0.25
  scene.add(outline)

  return {
    name: 'avatar-like-skinned-toon',
    scene,
    camera: makeCamera([0.95, 0.75, 3.2], [0, 0.25, 0]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [15, 18, 26],
    backgroundTolerance: 8,
    minNonBackgroundRatio: 0.035,
    validate(rgba, { width }) {
      const head = pixelAt(rgba, width, 48, 28)
      const body = pixelAt(rgba, width, 48, 54)
      const corner = pixelAt(rgba, width, 4, 4)
      if (!(head.r > head.b + 50 && head.g > head.b + 20 && body.b > body.r + 80 && body.g > body.r + 40 && corner.r < 25 && corner.g < 30 && corner.b < 40)) {
        throw new Error(`avatar corpus should render warm head and blue toon body, got head=${JSON.stringify(head)} body=${JSON.stringify(body)} corner=${JSON.stringify(corner)}`)
      }
    },
  }
}

function physicalIblShadowCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.04, 0.04, 0.05)
  scene.environment = environmentTexture()
  scene.environmentIntensity = 1.6
  scene.add(new THREE.AmbientLight(0xffffff, 0.15))

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(4, 4),
    new THREE.ShadowMaterial({ opacity: 0.65 }),
  )
  ground.rotation.x = -Math.PI / 2
  ground.position.y = -0.65
  ground.receiveShadow = true
  scene.add(ground)

  const sphere = new THREE.Mesh(
    new THREE.SphereGeometry(0.7, 24, 16),
    new THREE.MeshPhysicalMaterial({
      color: 0xffffff,
      metalness: 0.35,
      roughness: 0.22,
      clearcoat: 0.5,
      transmission: 0.15,
      thickness: 0.2,
      ior: 1.35,
    }),
  )
  sphere.castShadow = true
  sphere.receiveShadow = true
  scene.add(sphere)

  const light = new THREE.DirectionalLight(0xffffff, 2)
  light.position.set(3, 5, 2)
  light.target.position.set(0, 0, 0)
  light.castShadow = true
  light.shadow.mapSize.set(256, 256)
  light.shadow.camera.left = -3
  light.shadow.camera.right = 3
  light.shadow.camera.top = 3
  light.shadow.camera.bottom = -3
  light.shadow.camera.near = 0.1
  light.shadow.camera.far = 12
  scene.add(light, light.target)

  return {
    name: 'physical-ibl-shadow',
    scene,
    camera: makeCamera([2.2, 1.4, 3.2]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [10, 10, 13],
  }
}

function shadowMaterialReceiverCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(1, 1, 1)

  const receiver = new THREE.Mesh(
    new THREE.PlaneGeometry(4, 4),
    new THREE.ShadowMaterial({ color: 0x204080, opacity: 0.75 }),
  )
  receiver.rotation.x = -Math.PI / 2
  receiver.position.y = -0.6
  receiver.receiveShadow = true
  scene.add(receiver)

  const caster = new THREE.Mesh(
    new THREE.BoxGeometry(0.8, 0.8, 0.8),
    new THREE.MeshBasicMaterial({ color: 0xffffff }),
  )
  caster.position.y = 0.05
  caster.castShadow = true
  scene.add(caster)

  const light = new THREE.DirectionalLight(0xffffff, 2)
  light.position.set(3, 4, 2)
  light.target.position.set(0, -0.4, 0)
  light.castShadow = true
  light.shadow.mapSize.set(256, 256)
  light.shadow.camera.left = -3
  light.shadow.camera.right = 3
  light.shadow.camera.top = 3
  light.shadow.camera.bottom = -3
  light.shadow.camera.near = 0.1
  light.shadow.camera.far = 10
  scene.add(light, light.target)

  return {
    name: 'shadow-material-receiver',
    scene,
    camera: makeCamera([0.8, 1.5, 3.0], [0, -0.35, 0]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [255, 255, 255],
    minNonBackgroundRatio: 0.01,
    validate(rgba, { width }) {
      const shadow = meanRegion(rgba, width, 24, 42, 30, 54)
      if (!(shadow.b > shadow.g + 18 && shadow.g > shadow.r + 1)) {
        throw new Error(`ShadowMaterial corpus should tint received shadows blue-purple (${shadow.r}, ${shadow.g}, ${shadow.b})`)
      }
    },
  }
}

function shadowMaterialFogOptOutCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(1, 1, 1)
  scene.fog = new THREE.Fog(0x3366ff, 1.2, 3.0)

  const fogged = new THREE.Mesh(
    new THREE.PlaneGeometry(1.8, 3.2),
    new THREE.ShadowMaterial({ color: 0x204080, opacity: 0.85 }),
  )
  fogged.rotation.x = -Math.PI / 2
  fogged.position.set(-0.85, -0.6, 0)
  fogged.receiveShadow = true
  scene.add(fogged)

  const unfogged = new THREE.Mesh(
    new THREE.PlaneGeometry(1.8, 3.2),
    new THREE.ShadowMaterial({ color: 0x204080, opacity: 0.85, fog: false }),
  )
  unfogged.rotation.x = -Math.PI / 2
  unfogged.position.set(0.85, -0.6, 0)
  unfogged.receiveShadow = true
  scene.add(unfogged)

  const caster = new THREE.Mesh(
    new THREE.BoxGeometry(1.8, 0.8, 0.8),
    new THREE.MeshBasicMaterial({ color: 0xffffff }),
  )
  caster.position.y = 0.05
  caster.castShadow = true
  scene.add(caster)

  const light = new THREE.DirectionalLight(0xffffff, 2)
  light.position.set(2.5, 4, 2)
  light.target.position.set(0, -0.4, 0)
  light.castShadow = true
  light.shadow.mapSize.set(256, 256)
  light.shadow.camera.left = -3
  light.shadow.camera.right = 3
  light.shadow.camera.top = 3
  light.shadow.camera.bottom = -3
  light.shadow.camera.near = 0.1
  light.shadow.camera.far = 10
  scene.add(light, light.target)

  return {
    name: 'shadow-material-fog-opt-out',
    scene,
    camera: makeCamera([0.7, 1.5, 3.2], [0, -0.35, 0]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [255, 255, 255],
    minNonBackgroundRatio: 0.01,
  }
}

function dashedLineMaterialCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.72, -0.25, 0,
    -0.15, 0.35, 0,
    0.35, -0.1, 0,
    0.72, 0.42, 0,
  ]), 3))
  geometry.setAttribute('lineDistance', new THREE.BufferAttribute(new Float32Array([
    0,
    0.82,
    1.49,
    2.13,
  ]), 1))

  scene.add(new THREE.Line(
    geometry,
    new THREE.LineDashedMaterial({
      color: 0xffee55,
      dashSize: 0.16,
      gapSize: 0.09,
      scale: 1,
    }),
  ))

  return {
    name: 'line-dashed-material-pattern',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.001,
    validate(rgba, { width, height }) {
      const yellowPixels = countRegionPixels(
        rgba,
        width,
        0,
        0,
        width,
        height,
        (r, g, b) => r > 120 && g > 120 && b < 100,
      )
      const gap = pixelAt(rgba, width, 48, 48)
      if (!(yellowPixels > 30 && yellowPixels < 90 && gap.r < 5 && gap.g < 5 && gap.b < 5)) {
        throw new Error(`dashed-line corpus should render sparse yellow dashes with visible gaps, got yellow=${yellowPixels} gap=${JSON.stringify(gap)}`)
      }
    },
  }
}

function dashedLineMaterialTextureCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const map = new THREE.DataTexture(new Uint8Array([
    255, 0, 0, 0,
    0, 255, 0, 255,
  ]), 2, 1, THREE.RGBAFormat)
  map.colorSpace = THREE.SRGBColorSpace
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter
  map.needsUpdate = true

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.85, -0.25, 0,
    -0.3, 0.3, 0,
    0.32, -0.15, 0,
    0.85, 0.3, 0,
  ]), 3))
  geometry.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([
    0, 0.5,
    0.35, 0.5,
    0.65, 0.5,
    1, 0.5,
  ]), 2))
  geometry.setAttribute('lineDistance', new THREE.BufferAttribute(new Float32Array([
    0,
    0.78,
    1.55,
    2.25,
  ]), 1))

  scene.add(new THREE.Line(
    geometry,
    new THREE.LineDashedMaterial({
      alphaTest: 0.5,
      color: 0xffffff,
      dashSize: 0.2,
      gapSize: 0.1,
      map,
      scale: 1,
    }),
  ))

  return {
    name: 'line-dashed-material-textured-alpha',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.0004,
    validate(rgba, { width, height }) {
      const greenPixels = countRegionPixels(
        rgba,
        width,
        0,
        0,
        width,
        height,
        (r, g, b) => g > 120 && g > r + 50 && g > b + 50,
      )
      const redPixels = countRegionPixels(
        rgba,
        width,
        0,
        0,
        width,
        height,
        (r, g, b) => r > 120 && r > g + 50 && r > b + 50,
      )
      if (greenPixels < 3 || redPixels > 1) {
        throw new Error(`textured dashed-line corpus should render green alpha-tested dashes, got green=${greenPixels} red=${redPixels}`)
      }
    },
  }
}

function pointsMaterialTextureCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const map = new THREE.DataTexture(new Uint8Array([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ]), 2, 1, THREE.RGBAFormat)
  map.colorSpace = THREE.SRGBColorSpace
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter
  map.needsUpdate = true

  const alphaMap = new THREE.DataTexture(new Uint8Array([
    255, 0, 255, 255,
    255, 255, 255, 255,
  ]), 2, 1, THREE.RGBAFormat)
  alphaMap.magFilter = THREE.NearestFilter
  alphaMap.minFilter = THREE.NearestFilter
  alphaMap.needsUpdate = true

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))
  scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({
    alphaMap,
    alphaTest: 0.5,
    color: 0xffffff,
    map,
    size: 48,
    sizeAttenuation: false,
  })))

  return {
    name: 'points-material-textured-alpha',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.05,
    validate(rgba, { width, height }) {
      const greenPixels = countRegionPixels(
        rgba,
        width,
        0,
        0,
        width,
        height,
        (r, g, b) => g > 120 && g > r + 60 && g > b + 60,
      )
      const redPixels = countRegionPixels(
        rgba,
        width,
        0,
        0,
        width,
        height,
        (r, g, b) => r > 120 && r > g + 60 && r > b + 60,
      )
      if (greenPixels < 400 || redPixels > 4) {
        throw new Error(`textured point corpus should render green alpha-tested point-sprite UVs, got green=${greenPixels} red=${redPixels}`)
      }
    },
  }
}

function instancedLinesPointsCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const pointGeometry = new THREE.InstancedBufferGeometry()
  pointGeometry.instanceCount = 3
  pointGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0.35, 0]), 3))
  pointGeometry.setAttribute('instanceOffset', new THREE.InstancedBufferAttribute(new Float32Array([
    -0.55, 0, 0,
    0, 0, 0,
    0.55, 0, 0,
  ]), 3))
  pointGeometry.setAttribute('color', new THREE.InstancedBufferAttribute(new Float32Array([
    1, 0, 0,
    0, 1, 0,
    0, 0.4, 1,
  ]), 3))
  scene.add(new THREE.Points(pointGeometry, new THREE.PointsMaterial({
    color: 0xffffff,
    vertexColors: true,
    size: 18,
    sizeAttenuation: false,
    map: solidTexture(255, 255, 255),
  })))

  const lineGeometry = new THREE.InstancedBufferGeometry()
  lineGeometry.instanceCount = 2
  lineGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.35, -0.35, 0,
    0.35, -0.35, 0,
  ]), 3))
  lineGeometry.setAttribute('instanceOffset', new THREE.InstancedBufferAttribute(new Float32Array([
    -0.35, 0, 0,
    0.35, 0, 0,
  ]), 3))
  lineGeometry.setAttribute('color', new THREE.InstancedBufferAttribute(new Float32Array([
    1, 1, 0,
    0, 1, 1,
  ]), 3))
  scene.add(new THREE.LineSegments(lineGeometry, new THREE.LineBasicMaterial({
    color: 0xffffff,
    vertexColors: true,
  })))

  return {
    name: 'instanced-lines-and-points',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    validate(rgba, { width, height }) {
      const redPixels = countRegionPixels(rgba, width, 0, 0, width, height, (r, g, b) => r > 120 && r > g + 60 && r > b + 60)
      const greenPixels = countRegionPixels(rgba, width, 0, 0, width, height, (r, g, b) => g > 120 && g > r + 60 && g > b + 60)
      const bluePixels = countRegionPixels(rgba, width, 0, 0, width, height, (r, g, b) => b > 120 && b > r + 35 && b > g + 35)
      const yellowPixels = countRegionPixels(rgba, width, 0, 0, width, height, (r, g, b) => r > 120 && g > 120 && b < 120)
      const cyanPixels = countRegionPixels(rgba, width, 0, 0, width, height, (r, g, b) => g > 120 && b > 120 && r < 120)
      if (!(redPixels > 250 && greenPixels > 250 && bluePixels > 250 && yellowPixels > 250 && cyanPixels > 250)) {
        throw new Error(`instanced line/point corpus should render all per-instance colors, got red=${redPixels} green=${greenPixels} blue=${bluePixels} yellow=${yellowPixels} cyan=${cyanPixels}`)
      }
    },
  }
}

function instancedTextureUvCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const map = new THREE.DataTexture(new Uint8Array([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ]), 2, 1, THREE.RGBAFormat)
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter
  map.channel = 1
  map.needsUpdate = true

  const base = new THREE.PlaneGeometry(0.34, 0.34)
  const meshGeometry = new THREE.InstancedBufferGeometry()
  meshGeometry.index = base.index
  meshGeometry.setAttribute('position', base.getAttribute('position'))
  meshGeometry.setAttribute('uv', base.getAttribute('uv'))
  meshGeometry.setAttribute('instanceOffset', new THREE.InstancedBufferAttribute(new Float32Array([
    -0.52, 0.35, 0,
    0.52, 0.35, 0,
  ]), 3))
  meshGeometry.setAttribute('uv1', new THREE.InstancedBufferAttribute(new Float32Array([
    0.25, 0.5,
    0.75, 0.5,
  ]), 2))
  scene.add(new THREE.Mesh(
    meshGeometry,
    new THREE.MeshBasicMaterial({ color: 0xffffff, map }),
  ))

  const lineGeometry = new THREE.InstancedBufferGeometry()
  lineGeometry.instanceCount = 2
  lineGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.18, -0.08, 0,
    0.18, -0.08, 0,
  ]), 3))
  lineGeometry.setAttribute('instanceOffset', new THREE.InstancedBufferAttribute(new Float32Array([
    -0.52, 0, 0,
    0.52, 0, 0,
  ]), 3))
  lineGeometry.setAttribute('uv1', new THREE.InstancedBufferAttribute(new Float32Array([
    0.25, 0.5,
    0.75, 0.5,
  ]), 2))
  scene.add(new THREE.LineSegments(
    lineGeometry,
    new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 8, map }),
  ))

  const pointGeometry = new THREE.InstancedBufferGeometry()
  pointGeometry.instanceCount = 2
  pointGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, -0.48, 0]), 3))
  pointGeometry.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([0.5, 0.5]), 2))
  pointGeometry.setAttribute('instanceOffset', new THREE.InstancedBufferAttribute(new Float32Array([
    -0.52, 0, 0,
    0.52, 0, 0,
  ]), 3))
  pointGeometry.setAttribute('uv1', new THREE.InstancedBufferAttribute(new Float32Array([
    0.25, 0.5,
    0.75, 0.5,
  ]), 2))
  scene.add(new THREE.Points(pointGeometry, new THREE.PointsMaterial({
    color: 0xffffff,
    map,
    size: 18,
    sizeAttenuation: false,
  })))

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  return {
    name: 'instanced-texture-uv-streams',
    scene,
    camera,
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.02,
    browserReference: false,
    validate(rgba, { width }) {
      const samples = [
        ['left mesh', meanRegion(rgba, width, 20, 28, 30, 38), 'red'],
        ['right mesh', meanRegion(rgba, width, 66, 28, 76, 38), 'green'],
        ['left line', meanRegion(rgba, width, 17, 48, 31, 57), 'red'],
        ['right line', meanRegion(rgba, width, 65, 48, 79, 57), 'green'],
        ['left point', meanRegion(rgba, width, 18, 64, 32, 78), 'red'],
        ['right point', meanRegion(rgba, width, 66, 64, 80, 78), 'green'],
      ]
      for (const [label, color, expected] of samples) {
        if (expected === 'red' && color.r <= color.g + 45) {
          throw new Error(`${label} should sample the red instanced UV texel, got rgb(${color.r}, ${color.g}, ${color.b})`)
        }
        if (expected === 'green' && color.g <= color.r + 45) {
          throw new Error(`${label} should sample the green instanced UV texel, got rgb(${color.r}, ${color.g}, ${color.b})`)
        }
      }
    },
  }
}

function batchedMeshCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const source = new THREE.PlaneGeometry(0.42, 0.42)
  const batch = new THREE.BatchedMesh(
    3,
    source.getAttribute('position').count,
    source.index.count,
    new THREE.MeshBasicMaterial({ color: 0xffffff }),
  )
  const geometryId = batch.addGeometry(source)
  const left = batch.addInstance(geometryId)
  const right = batch.addInstance(geometryId)
  const hidden = batch.addInstance(geometryId)
  batch.setMatrixAt(left, new THREE.Matrix4().makeTranslation(-0.52, 0, 0))
  batch.setMatrixAt(right, new THREE.Matrix4().makeTranslation(0.52, 0, 0))
  batch.setMatrixAt(hidden, new THREE.Matrix4().makeTranslation(0, 0, 0))
  batch.setColorAt(left, new THREE.Color(1, 0.15, 0.05))
  batch.setColorAt(right, new THREE.Color(0.05, 0.9, 0.25))
  batch.setColorAt(hidden, new THREE.Color(0.1, 0.2, 1))
  batch.setVisibleAt(hidden, false)
  scene.add(batch)

  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 1.2, -1.2, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  return {
    name: 'batched-mesh-instance-colors',
    scene,
    camera,
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.03,
    validate(rgba, { width, height }) {
      const redPixels = countRegionPixels(rgba, width, 0, 0, width, height, (r, g, b) => r > 120 && r > g + 50 && r > b + 50)
      const greenPixels = countRegionPixels(rgba, width, 0, 0, width, height, (r, g, b) => g > 120 && g > r + 50 && g > b + 50)
      const bluePixels = countRegionPixels(rgba, width, 0, 0, width, height, (r, g, b) => b > 120 && b > r + 50 && b > g + 50)
      const hidden = pixelAt(rgba, width, 48, 48)
      if (!(redPixels > 200 && greenPixels > 200 && bluePixels < 5 && hidden.r < 5 && hidden.g < 5 && hidden.b < 5)) {
        throw new Error(`BatchedMesh corpus should render red/green visible instances and hide the blue instance, got red=${redPixels} green=${greenPixels} blue=${bluePixels} hidden=${JSON.stringify(hidden)}`)
      }
    },
  }
}

function batchedMeshInactiveGeometryCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const source = new THREE.PlaneGeometry(0.42, 0.42)
  const batch = new THREE.BatchedMesh(
    2,
    source.getAttribute('position').count * 2,
    source.index.count * 2,
    new THREE.MeshBasicMaterial({ color: 0xffffff }),
  )
  const activeGeometryId = batch.addGeometry(source)
  const inactiveGeometryId = batch.addGeometry(source.clone())
  const left = batch.addInstance(activeGeometryId)
  const right = batch.addInstance(inactiveGeometryId)
  batch.setMatrixAt(left, new THREE.Matrix4().makeTranslation(-0.52, 0, 0))
  batch.setMatrixAt(right, new THREE.Matrix4().makeTranslation(0.52, 0, 0))
  batch.setColorAt(left, new THREE.Color(1, 0.05, 0.05))
  batch.setColorAt(right, new THREE.Color(0.05, 1, 0.05))
  batch._geometryInfo[inactiveGeometryId].active = false
  scene.add(batch)

  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 1.2, -1.2, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  return {
    name: 'batched-mesh-inactive-geometry',
    scene,
    camera,
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.01,
    validate(rgba, { width, height }) {
      const y = Math.floor(height / 2)
      const leftOffset = (y * width + Math.floor(width * 0.29)) * 4
      const rightOffset = (y * width + Math.floor(width * 0.71)) * 4
      const leftR = rgba[leftOffset]
      const leftG = rgba[leftOffset + 1]
      const rightR = rgba[rightOffset]
      const rightG = rgba[rightOffset + 1]
      const rightB = rgba[rightOffset + 2]
      if (leftR <= leftG + 80) {
        throw new Error(`active BatchedMesh geometry should render red, got red=${leftR} green=${leftG}`)
      }
      if (rightR > 8 || rightG > 8 || rightB > 8) {
        throw new Error(`inactive BatchedMesh geometry should remain black, got rgb(${rightR}, ${rightG}, ${rightB})`)
      }
    },
  }
}

function batchedMeshCullingCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const visibleSource = new THREE.PlaneGeometry(0.6, 0.6)
  const culledSource = new THREE.PlaneGeometry(2.2, 2.2)
  culledSource.boundingSphere = new THREE.Sphere(new THREE.Vector3(5, 0, 0), 0.05)

  const batch = new THREE.BatchedMesh(
    2,
    visibleSource.getAttribute('position').count + culledSource.getAttribute('position').count,
    visibleSource.index.count + culledSource.index.count,
    new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false }),
  )
  const visibleGeometryId = batch.addGeometry(visibleSource)
  const culledGeometryId = batch.addGeometry(culledSource)
  const visible = batch.addInstance(visibleGeometryId)
  const culled = batch.addInstance(culledGeometryId)
  batch.setColorAt(visible, new THREE.Color(0.05, 0.95, 0.1))
  batch.setColorAt(culled, new THREE.Color(1, 0, 0))
  scene.add(batch)

  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 1.2, -1.2, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  return {
    name: 'batched-mesh-per-object-culling',
    scene,
    camera,
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.03,
    browserReference: false,
    validate(rgba, { width, height }) {
      const x = Math.floor(width / 2)
      const y = Math.floor(height / 2)
      const offset = (y * width + x) * 4
      const r = rgba[offset]
      const g = rgba[offset + 1]
      const b = rgba[offset + 2]
      if (g <= r + 80 || g <= b + 80) {
        throw new Error(`batched culling should leave the center green, got rgb(${r}, ${g}, ${b})`)
      }
    },
  }
}

function batchedMeshCustomSortCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const nearGeometry = new THREE.PlaneGeometry(1.5, 1.5)
  nearGeometry.translate(0, 0, 0.35)
  const farGeometry = new THREE.PlaneGeometry(1.5, 1.5)
  farGeometry.translate(0, 0, -0.35)

  const batch = new THREE.BatchedMesh(
    2,
    nearGeometry.getAttribute('position').count + farGeometry.getAttribute('position').count,
    nearGeometry.index.count + farGeometry.index.count,
    new THREE.MeshBasicMaterial({
      color: 0xffffff,
      depthWrite: false,
      transparent: true,
    }),
  )
  const nearGeometryId = batch.addGeometry(nearGeometry)
  const farGeometryId = batch.addGeometry(farGeometry)
  const near = batch.addInstance(nearGeometryId)
  const far = batch.addInstance(farGeometryId)
  batch.setColorAt(near, new THREE.Color(1, 0, 0))
  batch.setColorAt(far, new THREE.Color(0, 0, 1))
  batch.setCustomSort((list) => {
    list.sort((a, b) => a.index - b.index)
  })
  scene.add(batch)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  return {
    name: 'batched-mesh-custom-sort',
    scene,
    camera,
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.1,
    validate(rgba, { width, height }) {
      const x = Math.floor(width / 2)
      const y = Math.floor(height / 2)
      const offset = (y * width + x) * 4
      const r = rgba[offset]
      const g = rgba[offset + 1]
      const b = rgba[offset + 2]
      if (b <= r + 80 || b <= g + 80) {
        throw new Error(`batched customSort should draw the blue instance last, got rgb(${r}, ${g}, ${b})`)
      }
    },
  }
}

function lodAndGroupsCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.08, 0.08, 0.08)
  addBasicLights(scene)

  const group = new THREE.Group()
  group.renderOrder = 2
  group.add(new THREE.Mesh(
    new THREE.BoxGeometry(0.65, 0.65, 0.65),
    [
      new THREE.MeshLambertMaterial({ color: 0xff4444 }),
      new THREE.MeshLambertMaterial({ color: 0x44ff44 }),
      new THREE.MeshLambertMaterial({ color: 0x4444ff }),
      new THREE.MeshLambertMaterial({ color: 0xffff44 }),
      new THREE.MeshLambertMaterial({ color: 0xff44ff }),
      new THREE.MeshLambertMaterial({ color: 0x44ffff }),
    ],
  ))
  group.position.x = -0.45
  scene.add(group)

  const lod = new THREE.LOD()
  lod.position.x = 0.65
  lod.addLevel(
    new THREE.Mesh(new THREE.SphereGeometry(0.32, 16, 12), new THREE.MeshBasicMaterial({ color: 0x00aaff })),
    0,
  )
  lod.addLevel(
    new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.5, 0.5), new THREE.MeshBasicMaterial({ color: 0xffaa00 })),
    4,
  )
  scene.add(lod)

  return {
    name: 'lod-groups-material-array',
    scene,
    camera: makeCamera([1.4, 1.2, 3.2]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [20, 20, 20],
    validate(rgba, { width }) {
      const group = meanRegion(rgba, width, 16, 36, 36, 60)
      const lod = meanRegion(rgba, width, 60, 36, 80, 60)
      if (!(group.r > 80 && group.b > 80 && group.g < 30 && lod.b > lod.r + 100 && lod.g > lod.r + 40)) {
        throw new Error(`LOD/groups corpus should render the material-array group and near LOD sphere, got group=${JSON.stringify(group)} lod=${JSON.stringify(lod)}`)
      }
    },
  }
}

function lodZoomCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const lod = new THREE.LOD()
  lod.addLevel(
    new THREE.Mesh(new THREE.SphereGeometry(0.48, 24, 16), new THREE.MeshBasicMaterial({ color: 0xff0000 })),
    0,
  )
  lod.addLevel(
    new THREE.Mesh(new THREE.BoxGeometry(0.75, 0.75, 0.75), new THREE.MeshBasicMaterial({ color: 0x0000ff })),
    4,
  )
  scene.add(lod)

  const camera = makeCamera([0, 0, 6])
  camera.zoom = 2
  camera.updateProjectionMatrix()

  return {
    name: 'lod-zoom-selection',
    scene,
    camera,
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.02,
    validate(rgba, { width, height }) {
      const x = Math.floor(width / 2)
      const y = Math.floor(height / 2)
      const offset = (y * width + x) * 4
      const r = rgba[offset]
      const b = rgba[offset + 2]
      if (r <= b + 80) {
        throw new Error(`zoomed LOD corpus should render the red near level, got red=${r} blue=${b}`)
      }
    },
  }
}

function pathologicalGeometryCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.05, 0.05, 0.05)

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.8, -0.55, 0,
    0.8, -0.55, 0,
    -0.7, 0.55, 0,
    0.65, 0.5, 0.25,
  ]), 3))
  geometry.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([
    0, 0,
    1, 0,
    0, 1,
    1, 1,
  ]), 2))
  geometry.setAttribute('normal', new THREE.BufferAttribute(new Float32Array([
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
  ]), 3))
  geometry.setIndex([0, 1, 2, 1, 3, 2, 3, 3, 3])

  scene.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
    color: 0xffffff,
    map: solidTexture(64, 180, 255),
    side: THREE.DoubleSide,
  })))

  return {
    name: 'pathological-degenerate-geometry',
    scene,
    camera: makeCamera([0, 0, 2.6]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [13, 13, 13],
  }
}
