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
    pointSpotLightCorpus(),
    rectAreaLightCorpus(),
    skinnedMorphCorpus(),
    avatarLikeCorpus(),
    physicalIblShadowCorpus(),
    shadowMaterialReceiverCorpus(),
    shadowMaterialFogOptOutCorpus(),
    dashedLineMaterialCorpus(),
    instancedLinesPointsCorpus(),
    batchedMeshCorpus(),
    batchedMeshCullingCorpus(),
    lodAndGroupsCorpus(),
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
    new THREE.MeshPhongMaterial({ color: 0xffffff, shininess: 20 }),
    new THREE.MeshToonMaterial({ color: 0xffffff }),
  ]

  for (const [index, material] of materials.entries()) {
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(0.55, 1.2), material)
    mesh.position.x = (index - 1) * 0.65
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
