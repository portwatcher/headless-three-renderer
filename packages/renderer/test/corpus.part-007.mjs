import * as THREE from 'three'
import { CORPUS_RENDER_SIZE, countRegionPixels, cubeTexture, cubeUvGreenCubeTexture, makeCamera, meanAbsDiff, meanRegion, packedCubeUvGreenTexture, pixelAt } from './corpus.part-001.mjs'
export function rendererClippingStateCorpus() {
  const globalScene = new THREE.Scene()
  globalScene.background = new THREE.Color(0, 0, 1)
  globalScene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0x00ff00 }),
  ))

  const localMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 })
  localMaterial.clippingPlanes = [new THREE.Plane(new THREE.Vector3(1, 0, 0), 0)]
  const localScene = new THREE.Scene()
  localScene.background = new THREE.Color(0, 0, 1)
  localScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), localMaterial))

  const camera = makeCamera([0, 0, 3])
  const options = { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' }
  const stats = {}

  function regions(rgba) {
    return {
      top: meanRegion(rgba, options.width, 33, 18, 63, 36),
      bottom: meanRegion(rgba, options.width, 33, 60, 63, 78),
      left: meanRegion(rgba, options.width, 18, 33, 36, 63),
      right: meanRegion(rgba, options.width, 60, 33, 78, 63),
    }
  }

  function isGreen(mean) {
    return mean.g > mean.b + 80 && mean.g > mean.r + 80
  }

  function isRed(mean) {
    return mean.r > mean.b + 80 && mean.r > mean.g + 80
  }

  function isBlue(mean) {
    return mean.b > mean.r + 80 && mean.b > mean.g + 80
  }

  return {
    name: 'renderer-clipping-state-options',
    scene: globalScene,
    camera,
    options,
    background: [0, 0, 255],
    minNonBackgroundRatio: 0.2,
    render(renderer) {
      const previousClippingPlanes = renderer.clippingPlanes
      const previousLocalClippingEnabled = renderer.localClippingEnabled
      try {
        renderer.clippingPlanes = [new THREE.Plane(new THREE.Vector3(0, 1, 0), 0)]
        const fallback = renderer.render(globalScene, camera, options)
        stats.globalFallback = regions(fallback)

        const explicitEmpty = renderer.render(globalScene, camera, { ...options, clippingPlanes: [] })
        stats.globalExplicitEmpty = regions(explicitEmpty)

        renderer.clippingPlanes = []
        renderer.localClippingEnabled = false
        const localDisabled = renderer.render(localScene, camera, options)
        stats.localDisabled = regions(localDisabled)

        const localExplicitEnabled = renderer.render(localScene, camera, { ...options, localClippingEnabled: true })
        stats.localExplicitEnabled = regions(localExplicitEnabled)

        return fallback
      } finally {
        renderer.clippingPlanes = previousClippingPlanes
        renderer.localClippingEnabled = previousLocalClippingEnabled
      }
    },
    validate() {
      if (!(isGreen(stats.globalFallback.top) && isBlue(stats.globalFallback.bottom))) {
        throw new Error(`Renderer clippingPlanes fallback should clip the bottom half, stats=${JSON.stringify(stats.globalFallback)}`)
      }
      if (!isGreen(stats.globalExplicitEmpty.bottom)) {
        throw new Error(`explicit empty clippingPlanes should override renderer state, stats=${JSON.stringify(stats.globalExplicitEmpty)}`)
      }
      if (!(isRed(stats.localDisabled.left) && isRed(stats.localDisabled.right))) {
        throw new Error(`Renderer localClippingEnabled=false should disable material-local clipping, stats=${JSON.stringify(stats.localDisabled)}`)
      }
      if (!(isBlue(stats.localExplicitEnabled.left) && isRed(stats.localExplicitEnabled.right))) {
        throw new Error(`explicit localClippingEnabled=true should restore material-local clipping, stats=${JSON.stringify(stats.localExplicitEnabled)}`)
      }
    },
  }
}

export function nestedClippingGroupCorpus() {
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
      if (!(visible.r > visible.g + 130 && visible.r > visible.b + 170 && clippedLeft.r < 5 && clippedLeft.g < 5 && clippedLeft.b > 25 && clippedBottom.r < 5 && clippedBottom.g < 5 && clippedBottom.b > 25)) {
        throw new Error(`nested clipping corpus should keep only the red upper-right quadrant, got visible=${JSON.stringify(visible)} clippedLeft=${JSON.stringify(clippedLeft)} clippedBottom=${JSON.stringify(clippedBottom)}`)
      }
    },
  }
}

export function materialEnvMapCorpus() {
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
    background: [56, 56, 63],
    minNonBackgroundRatio: 0.02,
    validate(rgba, { width }) {
      const center = meanRegion(rgba, width, 32, 32, 64, 64)
      if (!(center.g > center.r + 60 && center.g > center.b + 25)) {
        throw new Error(`Phong material envMap corpus should render green reflected IBL, got ${JSON.stringify(center)}`)
      }
    },
  }
}

export function materialEnvMapBasicLambertCorpus() {
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
    background: [44, 44, 48],
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

export function materialEnvMapPbrCorpus() {
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
    background: [39, 39, 44],
    minNonBackgroundRatio: 0.06,
    browserReference: false,
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

export function cubeUvMaterialEnvMapCorpus() {
  const envMap = cubeUvGreenCubeTexture()

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.025)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(1.45, 1.45),
    new THREE.MeshBasicMaterial({
      color: 0xaa3322,
      envMap,
      combine: THREE.MixOperation,
      reflectivity: 1,
    }),
  ))

  return {
    name: 'cubeuv-cube-material-env-map',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [39, 39, 44],
    minNonBackgroundRatio: 0.08,
    browserReference: false,
    validate(rgba, { width }) {
      const center = meanRegion(rgba, width, 32, 32, 64, 64)
      if (!(center.g > center.r + 45 && center.g > center.b + 20)) {
        throw new Error(`CubeUV cube material envMap corpus should render green IBL, got ${JSON.stringify(center)}`)
      }
    },
  }
}

export function packedCubeUvMaterialEnvMapCorpus() {
  const envMap = packedCubeUvGreenTexture()

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.025)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(1.45, 1.45),
    new THREE.MeshBasicMaterial({
      color: 0xaa3322,
      envMap,
      combine: THREE.MixOperation,
      reflectivity: 1,
    }),
  ))

  return {
    name: 'packed-cubeuv-material-env-map',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [39, 39, 44],
    minNonBackgroundRatio: 0.08,
    browserReference: false,
    validate(rgba, { width }) {
      const center = meanRegion(rgba, width, 32, 32, 64, 64)
      if (!(center.g > center.r + 45 && center.g > center.b + 20)) {
        throw new Error(`packed CubeUV material envMap corpus should render green IBL, got ${JSON.stringify(center)}`)
      }
    },
  }
}

export function cubeEnvironmentOptionRotationCorpus() {
  const environment = cubeTexture([
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
  ])

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.environment = environment
  scene.environmentIntensity = 4
  scene.environmentRotation = new THREE.Euler(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0 }),
  ))

  const camera = makeCamera([0, 0, 3])
  const baseOptions = {
    width: CORPUS_RENDER_SIZE,
    height: CORPUS_RENDER_SIZE,
    format: 'rgba',
    outputColorSpace: THREE.LinearSRGBColorSpace,
  }
  const options = {
    ...baseOptions,
    environmentRotation: new THREE.Euler(0, -Math.PI / 2, 0),
  }
  let rotationDiff = 0

  return {
    name: 'cube-environment-option-rotation',
    scene,
    camera,
    options,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.08,
    browserReference: false,
    render(renderer) {
      const sceneRotation = renderer.render(scene, camera, baseOptions).slice()
      const optionRotation = renderer.render(scene, camera, options)
      rotationDiff = meanAbsDiff(sceneRotation, optionRotation)
      return optionRotation
    },
    validate() {
      if (!(rotationDiff > 1.0)) {
        throw new Error(`cube environment option rotation corpus should change IBL reflections, diff=${rotationDiff.toFixed(3)}`)
      }
    },
  }
}

export function narrowRawIblCorpus() {
  const environment = new THREE.DataTexture(new Uint8Array([220, 64]), 1, 1, THREE.RGFormat)
  environment.colorSpace = THREE.LinearSRGBColorSpace
  environment.mapping = THREE.EquirectangularReflectionMapping
  environment.needsUpdate = true

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.015, 0.015, 0.02)
  scene.environment = environment
  scene.environmentIntensity = 2.4
  scene.add(new THREE.Mesh(
    new THREE.SphereGeometry(0.8, 32, 18),
    new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0.2 }),
  ))

  return {
    name: 'narrow-raw-ibl-environment',
    scene,
    camera: makeCamera([0.8, 0.25, 3.0], [0, 0, 0]),
    options: {
      width: CORPUS_RENDER_SIZE,
      height: CORPUS_RENDER_SIZE,
      format: 'rgba',
      outputColorSpace: THREE.LinearSRGBColorSpace,
    },
    background: [4, 4, 5],
    minNonBackgroundRatio: 0.02,
    browserReference: false,
    validate(rgba, { width }) {
      const center = meanRegion(rgba, width, 34, 34, 62, 62)
      if (!(center.r > 180 && center.g > 140 && center.b < 120 && center.r > center.b + 110 && center.g > center.b + 70)) {
        throw new Error(`narrow raw IBL corpus should expand RG environment data into warm reflected light, got ${JSON.stringify(center)}`)
      }
    },
  }
}

export function meshBasicMaterialWireframeCorpus() {
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
      const yellowPixels = countRegionPixels(rgba, width, 0, 0, width, height, (r, g, b) => r > 120 && g > 100 && r > b + 35 && g > b + 15)
      const center = pixelAt(rgba, width, 48, 48)
      if (!(yellowPixels > 700 && yellowPixels < 1200 && center.r < 5 && center.g < 5 && center.b < 5)) {
        throw new Error(`basic wireframe corpus should render sparse yellow grid lines, got yellow=${yellowPixels} center=${JSON.stringify(center)}`)
      }
    },
  }
}

export function meshDepthMaterialCorpus() {
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
