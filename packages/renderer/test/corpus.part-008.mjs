import * as THREE from 'three'
import { CORPUS_RENDER_SIZE, constantUvPlane, countRegionPixels, makeCamera, meanRegion, pixelAt, setTextureMatrixOffset, solidTexture } from './corpus.part-001.mjs'
export function meshDepthPackingVariantsCorpus() {
  function packDepthToRG(v) {
    if (v <= 0) return [0, 0, 0, 255]
    if (v >= 1) return [255, 255, 0, 255]
    const vuf = Math.floor(v * 256)
    const gf = (v * 256) - vuf
    return [vuf, gf * 255, 0, 255]
  }

  function assertChannels(actual, expected, label, tolerance = 3) {
    for (const [channel, expectedValue] of [['r', expected[0]], ['g', expected[1]], ['b', expected[2]], ['a', expected[3]]]) {
      if (Math.abs(actual[channel] - expectedValue) > tolerance) {
        throw new Error(`depth packing corpus ${label}.${channel} expected ${expectedValue}, got ${actual[channel]}`)
      }
    }
  }

  function assertPrefix(actual, expected, label) {
    if (Math.abs(actual.r - expected[0]) > 8 || Math.abs(actual.g - expected[1]) > 8) {
      throw new Error(`depth packing corpus ${label} expected rg=${expected[0]},${expected[1]}, got mean=${JSON.stringify(actual)}`)
    }
  }

  function makeDepthPackingScene(depthPacking) {
    const z = 2.5
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshDepthMaterial({ depthPacking }))
    mesh.position.z = z
    scene.add(mesh)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    camera.updateMatrixWorld()
    camera.updateProjectionMatrix()

    const ndc = new THREE.Vector3(0, 0, z).project(camera)
    return {
      scene,
      camera,
      fragDepth: ndc.z * 0.5 + 0.5,
    }
  }

  const options = { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' }
  const rgbaFixture = makeDepthPackingScene(THREE.RGBADepthPacking)
  const rgbFixture = makeDepthPackingScene(THREE.RGBDepthPacking)
  const rgFixture = makeDepthPackingScene(THREE.RGDepthPacking)
  let rgbaMean = null
  let rgbMean = null
  let rgMean = null

  return {
    name: 'mesh-depth-material-packed-depth-variants',
    scene: rgbFixture.scene,
    camera: rgbFixture.camera,
    options,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.35,
    render(renderer) {
      const rgba = renderer.render(rgbaFixture.scene, rgbaFixture.camera, options)
      rgbaMean = pixelAt(rgba, options.width, 48, 48)
      const rgb = renderer.render(rgbFixture.scene, rgbFixture.camera, options)
      rgbMean = pixelAt(rgb, options.width, 48, 48)
      const rg = renderer.render(rgFixture.scene, rgFixture.camera, options)
      rgMean = pixelAt(rg, options.width, 48, 48)
      return rgb
    },
    validate() {
      assertPrefix(rgbaMean, packDepthToRG(rgbaFixture.fragDepth), 'rgba')
      if (!(rgbaMean.b > 10 && rgbaMean.a < 5)) {
        throw new Error(`depth packing corpus rgba should carry lower packed depth bits in b/a, got mean=${JSON.stringify(rgbaMean)}`)
      }

      assertPrefix(rgbMean, packDepthToRG(rgbFixture.fragDepth), 'rgb')
      if (!(rgbMean.b > 10 && rgbMean.a > 250)) {
        throw new Error(`depth packing corpus rgb should carry lower packed depth bits with opaque alpha, got mean=${JSON.stringify(rgbMean)}`)
      }

      assertChannels(rgMean, packDepthToRG(rgFixture.fragDepth), 'rg', 8)
    },
  }
}

export function meshDepthDisplacementMapCorpus() {
  function makeDisplacementMap() {
    const texture = new THREE.DataTexture(new Uint8Array([
      0, 0, 0, 255,
      255, 255, 255, 255,
    ]), 2, 1, THREE.RGBAFormat)
    texture.magFilter = THREE.NearestFilter
    texture.minFilter = THREE.NearestFilter
    texture.needsUpdate = true
    return texture
  }

  function makeScene(u) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(u, 0.5),
      new THREE.MeshDepthMaterial({
        displacementMap: makeDisplacementMap(),
        displacementScale: 2.5,
        displacementBias: 0,
        depthPacking: THREE.BasicDepthPacking,
      }),
    ))
    return scene
  }

  const flatScene = makeScene(0.25)
  const displacedScene = makeScene(0.75)
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  const options = { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' }
  let flatCenter = null
  let displacedCenter = null

  return {
    name: 'mesh-depth-material-displacement-map',
    scene: displacedScene,
    camera,
    options,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.35,
    render(renderer) {
      const flat = renderer.render(flatScene, camera, options)
      flatCenter = meanRegion(flat, options.width, 32, 32, 64, 64)
      const displaced = renderer.render(displacedScene, camera, options)
      displacedCenter = meanRegion(displaced, options.width, 32, 32, 64, 64)
      return displaced
    },
    validate() {
      if (!(displacedCenter.r > flatCenter.r + 15)) {
        throw new Error(`depth displacement corpus should move the plane nearer, flat=${JSON.stringify(flatCenter)} displaced=${JSON.stringify(displacedCenter)}`)
      }
    },
  }
}

export function meshDepthMaterialWireframeCorpus() {
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

export function meshDistanceMaterialCorpus() {
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
    validate(rgba, { width }) {
      const near = meanRegion(rgba, width, 16, 36, 36, 60)
      const far = meanRegion(rgba, width, 60, 36, 80, 60)
      if (!(near.r < 30 && near.g < 5 && near.b < 5 && far.r > 180 && far.g < 5 && far.b < 5)) {
        throw new Error(`distance material corpus should render the far plane bright red and near plane dark, got near=${JSON.stringify(near)} far=${JSON.stringify(far)}`)
      }
    },
  }
}

export function meshDistanceDisplacementMapCorpus() {
  function makeDisplacementMap(channel) {
    const texture = new THREE.DataTexture(new Uint8Array([
      0, 0, 0, 255,
      255, 255, 255, 255,
    ]), 2, 1, THREE.RGBAFormat)
    texture.channel = channel
    texture.magFilter = THREE.NearestFilter
    texture.minFilter = THREE.NearestFilter
    texture.needsUpdate = true
    return texture
  }

  function makeScene(channel) {
    const geometry = constantUvPlane(0.25, 0.5)
    const uv1 = new Float32Array(geometry.getAttribute('position').count * 2)
    for (let i = 0; i < geometry.getAttribute('position').count; i += 1) {
      uv1[i * 2] = 0.75
      uv1[i * 2 + 1] = 0.5
    }
    geometry.setAttribute('uv1', new THREE.BufferAttribute(uv1, 2))

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      geometry,
      new THREE.MeshDistanceMaterial({
        displacementMap: makeDisplacementMap(channel),
        displacementScale: 1.2,
        displacementBias: 0,
      }),
    ))
    return scene
  }

  const primaryScene = makeScene(0)
  const secondaryScene = makeScene(1)
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 8)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  const options = { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' }
  let primaryCenter = null
  let secondaryCenter = null

  return {
    name: 'mesh-distance-material-displacement-map',
    scene: secondaryScene,
    camera,
    options,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.35,
    render(renderer) {
      const primary = renderer.render(primaryScene, camera, options)
      primaryCenter = meanRegion(primary, options.width, 32, 32, 64, 64)
      const secondary = renderer.render(secondaryScene, camera, options)
      secondaryCenter = meanRegion(secondary, options.width, 32, 32, 64, 64)
      return secondary
    },
    validate() {
      if (!(primaryCenter.r > secondaryCenter.r + 15)) {
        throw new Error(`distance displacement corpus should move the uv1-selected plane closer, primary=${JSON.stringify(primaryCenter)} secondary=${JSON.stringify(secondaryCenter)}`)
      }
    },
  }
}

export function meshDistanceMaterialWireframeCorpus() {
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
    validate(rgba, { width, height }) {
      const redPixels = countRegionPixels(rgba, width, 0, 0, width, height, (r, g, b) => r > 120 && g < 5 && b < 5)
      const center = pixelAt(rgba, width, 48, 48)
      if (!(redPixels > 700 && redPixels < 1200 && center.r === 0 && center.g === 0 && center.b === 0)) {
        throw new Error(`distance wireframe corpus should render sparse red distance lines, got red=${redPixels} center=${JSON.stringify(center)}`)
      }
    },
  }
}

export function meshStandardMaterialDisplacementCorpus() {
  const camera = makeCamera([0, 0, 3])
  const options = { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' }
  let flatPixels = 0
  let displacedPixels = 0

  function makeDisplacementMap(offsetX) {
    const displacementMap = new THREE.DataTexture(new Uint8Array([
      0, 0, 0, 255,
      255, 0, 0, 255,
    ]), 2, 1, THREE.RGBAFormat)
    displacementMap.magFilter = THREE.NearestFilter
    displacementMap.minFilter = THREE.NearestFilter
    setTextureMatrixOffset(displacementMap, offsetX)
    displacementMap.needsUpdate = true
    return displacementMap
  }

  function makeScene(displacementScale) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.AmbientLight(0xffffff, 0.35))

    const light = new THREE.DirectionalLight(0xffffff, 2.4)
    light.position.set(0.8, 1.4, 3)
    scene.add(light)

    scene.add(new THREE.Mesh(
      constantUvPlane(0.25, 0.5, 1.2, 1.2),
      new THREE.MeshStandardMaterial({
        color: 0x66ccff,
        roughness: 0.35,
        metalness: 0,
        displacementMap: makeDisplacementMap(0.5),
        displacementScale,
        displacementBias: 0,
      }),
    ))
    return scene
  }

  function visiblePixels(rgba) {
    return countRegionPixels(
      rgba,
      options.width,
      0,
      0,
      options.width,
      options.height,
      (r, g, b) => r > 20 || g > 20 || b > 20,
    )
  }

  return {
    name: 'mesh-standard-displacement-map',
    scene: makeScene(0.8),
    camera,
    options,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.2,
    render(renderer) {
      const flat = renderer.render(makeScene(0), camera, options).slice()
      const displaced = renderer.render(makeScene(0.8), camera, options).slice()
      flatPixels = visiblePixels(flat)
      displacedPixels = visiblePixels(displaced)
      return displaced
    },
    validate() {
      if (!(displacedPixels > flatPixels + 750)) {
        throw new Error(`standard displacement corpus should expand the visible main-pass plane, flat=${flatPixels} displaced=${displacedPixels}`)
      }
    },
  }
}

export function meshNormalMaterialCorpus() {
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
    background: [33, 33, 39],
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

export function meshNormalMaterialNormalMapCorpus() {
  function makeScene(normalMap) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshNormalMaterial({ normalMap }),
    ))
    return scene
  }

  const flatScene = makeScene(null)
  const mappedScene = makeScene(solidTexture(255, 128, 128))
  const camera = makeCamera([0, 0, 3])
  const options = { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' }
  let flatCenter = null
  let mappedCenter = null

  return {
    name: 'mesh-normal-material-normal-map',
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
      if (!(mappedCenter.r > flatCenter.r + 40 && flatCenter.b > mappedCenter.b + 40)) {
        throw new Error(`normal-map corpus should tilt MeshNormalMaterial output, flat=${JSON.stringify(flatCenter)} mapped=${JSON.stringify(mappedCenter)}`)
      }
    },
  }
}
