import * as THREE from 'three'
import { CORPUS_RENDER_SIZE, constantUvPlane, makeCamera, meanRegion, pixelAt, solidTexture } from './corpus.part-001.mjs'
export function meshToonMaterialNormalMapCorpus() {
  function makeScene(normalScale) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshToonMaterial({
        color: 0xffffff,
        normalMap: solidTexture(255, 128, 128),
        normalScale: new THREE.Vector2(normalScale, normalScale),
      }),
    ))

    const light = new THREE.DirectionalLight(0xffffff, 3)
    light.position.set(3, 0, 0.25)
    scene.add(light)
    return scene
  }

  const flatScene = makeScene(0)
  const mappedScene = makeScene(1)
  const camera = makeCamera([0, 0, 3])
  const options = { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' }
  let flatCenter = null
  let mappedCenter = null

  return {
    name: 'mesh-toon-material-normal-map',
    scene: mappedScene,
    camera,
    options,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.35,
    render(renderer) {
      const flat = renderer.render(flatScene, camera, options)
      flatCenter = meanRegion(flat, options.width, 32, 32, 64, 64)
      const mapped = renderer.render(mappedScene, camera, options)
      mappedCenter = meanRegion(mapped, options.width, 32, 32, 64, 64)
      return mapped
    },
    validate() {
      if (!(mappedCenter.r > flatCenter.r + 5)) {
        throw new Error(`toon normal-map corpus should tilt lighting toward the oblique light, flat=${JSON.stringify(flatCenter)} mapped=${JSON.stringify(mappedCenter)}`)
      }
    },
  }
}

export function meshToonMaterialBumpMapCorpus() {
  function makeBumpMap() {
    const texture = new THREE.DataTexture(new Uint8Array([
      0, 0, 0, 255,
      255, 255, 255, 255,
    ]), 2, 1, THREE.RGBAFormat)
    texture.magFilter = THREE.LinearFilter
    texture.minFilter = THREE.LinearFilter
    texture.needsUpdate = true
    return texture
  }

  function makeGradientMap() {
    const texture = new THREE.DataTexture(new Uint8Array([
      0, 0, 0, 255,
      255, 255, 255, 255,
    ]), 2, 1, THREE.RGBAFormat)
    texture.magFilter = THREE.LinearFilter
    texture.minFilter = THREE.LinearFilter
    texture.needsUpdate = true
    return texture
  }

  function makeScene(bumpScale) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshToonMaterial({
        color: 0xffffff,
        bumpMap: makeBumpMap(),
        bumpScale,
        gradientMap: makeGradientMap(),
      }),
    ))

    const light = new THREE.DirectionalLight(0xffffff, 3)
    light.position.set(3, 0, 0.25)
    scene.add(light)
    return scene
  }

  const flatScene = makeScene(0)
  const bumpedScene = makeScene(8)
  const camera = makeCamera([0, 0, 3])
  const options = { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' }
  let flatCenter = null
  let bumpedCenter = null

  return {
    name: 'mesh-toon-material-bump-map',
    scene: bumpedScene,
    camera,
    options,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.35,
    render(renderer) {
      const flat = renderer.render(flatScene, camera, options)
      flatCenter = meanRegion(flat, options.width, 32, 32, 64, 64)
      const bumped = renderer.render(bumpedScene, camera, options)
      bumpedCenter = meanRegion(bumped, options.width, 32, 32, 64, 64)
      return bumped
    },
    validate() {
      if (!(flatCenter.r > bumpedCenter.r + 8)) {
        throw new Error(`toon bump-map corpus should perturb the ramp lookup, flat=${JSON.stringify(flatCenter)} bumped=${JSON.stringify(bumpedCenter)}`)
      }
    },
  }
}

export function meshToonTextureSlotsCorpus() {
  const camera = makeCamera([0, 0, 3])
  const options = { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' }
  const stats = new Map()

  function secondaryUvPlane() {
    const geometry = constantUvPlane(0.25, 0.5)
    const uv1 = new Float32Array(geometry.getAttribute('position').count * 2)
    for (let i = 0; i < geometry.getAttribute('position').count; i += 1) {
      uv1[i * 2] = 0.75
      uv1[i * 2 + 1] = 0.5
    }
    geometry.setAttribute('uv1', new THREE.BufferAttribute(uv1, 2))
    return geometry
  }

  function channelTexture(data) {
    const texture = new THREE.DataTexture(new Uint8Array(data), 2, 1, THREE.RGBAFormat)
    texture.channel = 1
    texture.magFilter = THREE.NearestFilter
    texture.minFilter = THREE.NearestFilter
    texture.needsUpdate = true
    return texture
  }

  function makeScene(kind) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const geometry = secondaryUvPlane()
    let material

    if (kind === 'map') {
      material = new THREE.MeshToonMaterial({
        color: 0xffffff,
        map: channelTexture([
          0, 255, 0, 255,
          255, 0, 0, 255,
        ]),
      })
      const light = new THREE.DirectionalLight(0xffffff, 3)
      light.position.set(0, 0, 3)
      scene.add(light)
    } else if (kind === 'emissive') {
      material = new THREE.MeshToonMaterial({
        color: 0x000000,
        emissive: 0xffffff,
        emissiveMap: channelTexture([
          0, 255, 0, 255,
          255, 0, 0, 255,
        ]),
      })
    } else {
      material = new THREE.MeshToonMaterial({
        color: 0xffffff,
        lightMap: channelTexture([
          0, 0, 0, 255,
          255, 255, 255, 255,
        ]),
        lightMapIntensity: 4,
      })
    }

    scene.add(new THREE.Mesh(geometry, material))
    return scene
  }

  function centerMean(rgba) {
    return meanRegion(rgba, options.width, 28, 28, 68, 68)
  }

  return {
    name: 'mesh-toon-texture-slot-uv-channel',
    scene: makeScene('map'),
    camera,
    options,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.35,
    render(renderer) {
      const map = renderer.render(makeScene('map'), camera, options).slice()
      const emissive = renderer.render(makeScene('emissive'), camera, options).slice()
      const lightMap = renderer.render(makeScene('lightMap'), camera, options).slice()
      stats.set('map', centerMean(map))
      stats.set('emissive', centerMean(emissive))
      stats.set('lightMap', centerMean(lightMap))
      return map
    },
    validate() {
      const map = stats.get('map')
      const emissive = stats.get('emissive')
      const lightMap = stats.get('lightMap')
      if (!(map && map.r > map.g + 40 && map.r > map.b + 40)) {
        throw new Error(`toon map should sample the secondary-UV red texel, got ${JSON.stringify(map)}`)
      }
      if (!(emissive && emissive.r > emissive.g + 40 && emissive.r > emissive.b + 40)) {
        throw new Error(`toon emissiveMap should sample the secondary-UV red texel, got ${JSON.stringify(emissive)}`)
      }
      if (!(lightMap && lightMap.r > 100 && lightMap.g > 100 && lightMap.b > 100)) {
        throw new Error(`toon lightMap should sample the secondary-UV bright texel, got ${JSON.stringify(lightMap)}`)
      }
    },
  }
}

export function meshToonAlphaMapCorpus() {
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

export function viewportScissorCorpus() {
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

export function cameraLayerFilteringCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const filteredObject = new THREE.Mesh(
    new THREE.PlaneGeometry(0.95, 1.35),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  )
  filteredObject.position.set(-0.55, 0, 0.1)
  scene.add(filteredObject)

  const visibleObject = new THREE.Mesh(
    new THREE.PlaneGeometry(0.95, 1.35),
    new THREE.MeshBasicMaterial({ color: 0x00ff00 }),
  )
  visibleObject.position.set(-0.55, 0, 0)
  visibleObject.layers.set(1)
  scene.add(visibleObject)

  const litObject = new THREE.Mesh(
    new THREE.PlaneGeometry(0.95, 1.35),
    new THREE.MeshLambertMaterial({ color: 0xffffff }),
  )
  litObject.position.set(0.55, 0, 0)
  litObject.layers.set(1)
  scene.add(litObject)

  const filteredLight = new THREE.DirectionalLight(0xff0000, 8)
  filteredLight.position.set(0, 0, 3)
  filteredLight.layers.set(0)
  scene.add(filteredLight)

  const visibleLight = new THREE.DirectionalLight(0x00ff00, 4)
  visibleLight.position.set(0, 0, 3)
  visibleLight.layers.set(1)
  scene.add(visibleLight)

  const camera = makeCamera([0, 0, 3])
  camera.layers.set(1)

  return {
    name: 'camera-layer-filtering',
    scene,
    camera,
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.12,
    validate(rgba, { width }) {
      const objectPanel = meanRegion(rgba, width, 18, 28, 42, 68)
      const lightPanel = meanRegion(rgba, width, 54, 28, 78, 68)
      if (!(objectPanel.g > objectPanel.r + 80 && objectPanel.g > objectPanel.b + 80)) {
        throw new Error(`camera layer corpus should hide the red layer-0 object and show green layer-1 object, got ${JSON.stringify(objectPanel)}`)
      }
      if (!(lightPanel.g > lightPanel.r + 45 && lightPanel.g > lightPanel.b + 45)) {
        throw new Error(`camera layer corpus should hide the red layer-0 light and apply green layer-1 light, got ${JSON.stringify(lightPanel)}`)
      }
    },
  }
}

export function arrayCameraViewportCorpus() {
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

  const camera = new THREE.ArrayCamera([leftCamera, rightCamera])
  camera.layers.enable(1)
  camera.layers.enable(2)

  return {
    name: 'array-camera-viewport-split',
    scene,
    camera,
    options: { width, height, format: 'rgba' },
    background: [0, 0, 20],
    validate(rgba, { width }) {
      const left = meanRegion(rgba, width, 16, 32, 40, 64)
      const right = meanRegion(rgba, width, 56, 32, 80, 64)
      if (!(left.r > left.g + 170 && left.r > left.b + 180 && right.g > right.r + 60 && right.g > right.b + 70)) {
        throw new Error(`ArrayCamera corpus should render red left and green right viewports, got left=${JSON.stringify(left)} right=${JSON.stringify(right)}`)
      }
    },
  }
}

export function cubeCameraCaptureCorpus() {
  const scene = makeCubeCaptureScene()
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

export function cubeCameraUpdateCorpus() {
  const scene = makeCubeCaptureScene()
  const target = new THREE.WebGLCubeRenderTarget(CORPUS_RENDER_SIZE)
  const camera = new THREE.CubeCamera(0.01, 100, target)
  camera.activeMipmapLevel = 1

  return {
    name: 'cube-camera-update-active-mip',
    scene,
    camera,
    options: { width: CORPUS_RENDER_SIZE / 2, height: CORPUS_RENDER_SIZE / 2, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.08,
    browserReference: false,
    render(renderer) {
      camera.update(renderer, scene)
      return target.texture.mipmaps[1].image[0].data
    },
    validate(rgba, { width }) {
      const center = pixelAt(rgba, width, 24, 24)
      const corner = pixelAt(rgba, width, 2, 2)
      if (!(center.r > center.g + 180 && center.r > center.b + 180 && corner.r === 0 && corner.g === 0 && corner.b === 0)) {
        throw new Error(`cube camera update corpus should capture the red +X active mip face, got center=${JSON.stringify(center)} corner=${JSON.stringify(corner)}`)
      }
    },
  }
}

export function makeCubeCaptureScene() {
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
  return scene
}
