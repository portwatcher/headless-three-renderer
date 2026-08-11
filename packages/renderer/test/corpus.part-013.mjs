import * as THREE from 'three'
import { CORPUS_RENDER_SIZE, constantUvPlane, makeCamera, meanAbsDiff, meanRegion, setTextureMatrixOffset } from './corpus.part-001.mjs'
export function physicalAnisotropyMapCorpus() {
  const camera = makeCamera([0, 0, 3])
  const options = { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' }
  const stats = {}

  function makeMap(offsetX) {
    const texture = new THREE.DataTexture(new Uint8Array([
      128, 128, 0, 255,
      255, 128, 255, 255,
    ]), 2, 1, THREE.RGBAFormat)
    texture.magFilter = THREE.NearestFilter
    texture.minFilter = THREE.NearestFilter
    setTextureMatrixOffset(texture, offsetX)
    texture.needsUpdate = true
    return texture
  }

  function makeScene(offsetX) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.25, 0.5),
      new THREE.MeshPhysicalMaterial({
        color: 0x111111,
        roughness: 0.2,
        metalness: 0,
        anisotropy: 1,
        anisotropyRotation: Math.PI / 4,
        anisotropyMap: makeMap(offsetX),
      }),
    ))
    const light = new THREE.PointLight(0xffffff, 250)
    light.position.set(0.8, 0.8, 2)
    scene.add(light)
    return scene
  }

  return {
    name: 'physical-anisotropy-map-slot',
    scene: makeScene(0.5),
    camera,
    options,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.08,
    browserReference: false,
    render(renderer) {
      const primary = renderer.render(makeScene(0), camera, options)
      const shifted = renderer.render(makeScene(0.5), camera, options)
      stats.diff = meanAbsDiff(primary, shifted)
      return shifted
    },
    validate() {
      if (!(stats.diff > 1)) {
        throw new Error(`physical anisotropy corpus should sample the shifted anisotropyMap texel, diff=${stats.diff}`)
      }
    },
  }
}

export function physicalIridescenceMapCorpus() {
  const camera = makeCamera([0, 0, 3])
  const options = { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' }
  let factorDiff = 0
  let thicknessDiff = 0
  let factorCenter = null
  let thicknessCenter = null

  function physicalLight(scene) {
    const light = new THREE.PointLight(0xffffff, 300)
    light.position.set(0, 0, 2)
    scene.add(light)
  }

  function makeMap(data, offsetX) {
    const texture = new THREE.DataTexture(new Uint8Array(data), 2, 1, THREE.RGBAFormat)
    texture.magFilter = THREE.NearestFilter
    texture.minFilter = THREE.NearestFilter
    setTextureMatrixOffset(texture, offsetX)
    texture.needsUpdate = true
    return texture
  }

  function makeFactorScene(offsetX) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.25, 0.5),
      new THREE.MeshPhysicalMaterial({
        color: 0x000000,
        roughness: 0.08,
        metalness: 0,
        specularIntensity: 1,
        iridescence: 1,
        iridescenceMap: makeMap([
          0, 0, 0, 255,
          255, 0, 0, 255,
        ], offsetX),
        iridescenceIOR: 1.8,
        iridescenceThicknessRange: [250, 650],
      }),
    ))
    physicalLight(scene)
    return scene
  }

  function makeThicknessScene(offsetX) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.25, 0.5),
      new THREE.MeshPhysicalMaterial({
        color: 0x000000,
        roughness: 0.08,
        metalness: 0,
        specularIntensity: 1,
        iridescence: 1,
        iridescenceIOR: 1.8,
        iridescenceThicknessRange: [120, 760],
        iridescenceThicknessMap: makeMap([
          0, 0, 0, 255,
          0, 255, 0, 255,
        ], offsetX),
      }),
    ))
    physicalLight(scene)
    return scene
  }

  return {
    name: 'physical-iridescence-texture-maps',
    scene: makeFactorScene(0.5),
    camera,
    options,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.2,
    browserReference: false,
    render(renderer) {
      const factorDisabled = renderer.render(makeFactorScene(0), camera, options).slice()
      const factorEnabled = renderer.render(makeFactorScene(0.5), camera, options).slice()
      const thicknessLow = renderer.render(makeThicknessScene(0), camera, options).slice()
      const thicknessHigh = renderer.render(makeThicknessScene(0.5), camera, options).slice()
      factorDiff = meanAbsDiff(factorDisabled, factorEnabled)
      thicknessDiff = meanAbsDiff(thicknessLow, thicknessHigh)
      factorCenter = meanRegion(factorEnabled, options.width, 24, 24, 72, 72)
      thicknessCenter = meanRegion(thicknessHigh, options.width, 24, 24, 72, 72)
      return factorEnabled
    },
    validate() {
      if (!(factorDiff > 10)) {
        throw new Error(`iridescenceMap corpus should modulate scalar iridescence, diff=${factorDiff.toFixed(3)} center=${JSON.stringify(factorCenter)}`)
      }
      if (!(thicknessDiff > 5)) {
        throw new Error(`iridescenceThicknessMap corpus should select a different film thickness, diff=${thicknessDiff.toFixed(3)} center=${JSON.stringify(thicknessCenter)}`)
      }
      if (!(factorCenter && factorCenter.r > 15 && thicknessCenter && Math.max(thicknessCenter.r, thicknessCenter.g, thicknessCenter.b) > 15)) {
        throw new Error(`iridescence texture corpus should render visible specular highlights, factor=${JSON.stringify(factorCenter)} thickness=${JSON.stringify(thicknessCenter)}`)
      }
    },
  }
}

export function physicalTransmissionMapCorpus() {
  const camera = makeCamera([0, 0, 3])
  const options = { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' }
  const stats = {}

  function makeMap(data, offsetX) {
    const texture = new THREE.DataTexture(new Uint8Array(data), 2, 1, THREE.RGBAFormat)
    texture.magFilter = THREE.NearestFilter
    texture.minFilter = THREE.NearestFilter
    setTextureMatrixOffset(texture, offsetX)
    texture.needsUpdate = true
    return texture
  }

  function makeTransmissionScene(offsetX) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const back = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ color: 0x0000ff }),
    )
    back.position.z = -0.2
    scene.add(back)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.25, 0.5),
      new THREE.MeshPhysicalMaterial({
        color: 0xff0000,
        roughness: 0.1,
        metalness: 0,
        transmission: 1,
        transmissionMap: makeMap([
          0, 0, 0, 255,
          255, 0, 0, 255,
        ], offsetX),
        ior: 1.5,
        thickness: 0,
      }),
    ))
    return scene
  }

  function makeThicknessScene(offsetX) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const back = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    )
    back.position.z = -0.2
    scene.add(back)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.25, 0.5),
      new THREE.MeshPhysicalMaterial({
        color: 0xffffff,
        roughness: 0.1,
        metalness: 0,
        transmission: 1,
        ior: 1.5,
        thickness: 8,
        thicknessMap: makeMap([
          0, 0, 0, 255,
          0, 255, 0, 255,
        ], offsetX),
        attenuationColor: new THREE.Color(0.02, 0.02, 1),
        attenuationDistance: 1,
      }),
    ))
    return scene
  }

  return {
    name: 'physical-transmission-map-slots',
    scene: makeTransmissionScene(0.5),
    camera,
    options,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.2,
    browserReference: false,
    render(renderer) {
      const transmissionPrimary = renderer.render(makeTransmissionScene(0), camera, options)
      const transmissionShifted = renderer.render(makeTransmissionScene(0.5), camera, options)
      const thicknessPrimary = renderer.render(makeThicknessScene(0), camera, options)
      const thicknessShifted = renderer.render(makeThicknessScene(0.5), camera, options)
      stats.transmissionPrimary = meanRegion(transmissionPrimary, options.width, 0, 0, options.width, options.height)
      stats.transmissionShifted = meanRegion(transmissionShifted, options.width, 0, 0, options.width, options.height)
      stats.thicknessPrimary = meanRegion(thicknessPrimary, options.width, 0, 0, options.width, options.height)
      stats.thicknessShifted = meanRegion(thicknessShifted, options.width, 0, 0, options.width, options.height)
      return transmissionShifted
    },
    validate() {
      if (!(stats.transmissionPrimary.r > stats.transmissionPrimary.b + 30)) {
        throw new Error(`physical transmission corpus should keep the primary transmissionMap texel opaque red, stats=${JSON.stringify(stats)}`)
      }
      if (!(stats.transmissionShifted.b > stats.transmissionShifted.r + 40)) {
        throw new Error(`physical transmission corpus should sample the shifted transmissionMap texel, stats=${JSON.stringify(stats)}`)
      }
      if (!(stats.thicknessPrimary.r > stats.thicknessPrimary.b - 15)) {
        throw new Error(`physical transmission corpus should keep the primary thicknessMap texel thin, stats=${JSON.stringify(stats)}`)
      }
      if (!(stats.thicknessShifted.b > stats.thicknessShifted.r + 40)) {
        throw new Error(`physical transmission corpus should sample the shifted attenuating thicknessMap texel, stats=${JSON.stringify(stats)}`)
      }
    },
  }
}

export function physicalTransmissionDispersionCorpus() {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  const options = { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' }
  let normalOutput = null

  function makeScene(dispersion) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)

    const left = new THREE.Mesh(
      new THREE.PlaneGeometry(1.6, 3),
      new THREE.MeshBasicMaterial({ color: 0xff0000 }),
    )
    left.position.set(-0.8, 0, -0.2)
    scene.add(left)

    const right = new THREE.Mesh(
      new THREE.PlaneGeometry(1.6, 3),
      new THREE.MeshBasicMaterial({ color: 0x0000ff }),
    )
    right.position.set(0.8, 0, -0.2)
    scene.add(right)

    scene.add(new THREE.Mesh(
      new THREE.SphereGeometry(0.95, 48, 24),
      new THREE.MeshPhysicalMaterial({
        color: 0xffffff,
        metalness: 0,
        roughness: 0.02,
        transmission: 1,
        thickness: 40,
        ior: 2.2,
        dispersion,
      }),
    ))
    return scene
  }

  const dispersedScene = makeScene(10)

  return {
    name: 'physical-transmission-dispersion',
    scene: dispersedScene,
    camera,
    options,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.15,
    render(renderer) {
      normalOutput = renderer.render(makeScene(0), camera, options)
      return renderer.render(dispersedScene, camera, options)
    },
    validate(rgba, { width }) {
      if (!normalOutput) {
        throw new Error('physical dispersion corpus did not render the normal reference output')
      }
      const diff = meanAbsDiff(normalOutput, rgba)
      const normalEdge = meanRegion(normalOutput, width, 42, 32, 54, 64)
      const dispersedEdge = meanRegion(rgba, width, 42, 32, 54, 64)
      const normalSeparation = Math.abs(normalEdge.r - normalEdge.b)
      const dispersedSeparation = Math.abs(dispersedEdge.r - dispersedEdge.b)
      if (!(diff > 8 && Math.abs(dispersedSeparation - normalSeparation) > 18)) {
        throw new Error(`physical dispersion corpus should shift transmitted color channels, diff=${diff.toFixed(2)} normal=${JSON.stringify(normalEdge)} dispersed=${JSON.stringify(dispersedEdge)}`)
      }
    },
  }
}

export function transmissionResolutionScaleCorpus() {
  const width = CORPUS_RENDER_SIZE
  const height = CORPUS_RENDER_SIZE
  const camera = makeCamera([0, 0, 3])
  const options = { width, height, format: 'rgba' }
  let fullContrast = 0
  let lowContrast = 0
  let optionFullContrast = 0

  function makeScene() {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)

    const left = new THREE.Mesh(
      new THREE.PlaneGeometry(1.6, 3),
      new THREE.MeshBasicMaterial({ color: 0xff0000 }),
    )
    left.position.set(-0.8, 0, -0.1)
    scene.add(left)

    const right = new THREE.Mesh(
      new THREE.PlaneGeometry(1.6, 3),
      new THREE.MeshBasicMaterial({ color: 0x0000ff }),
    )
    right.position.set(0.8, 0, -0.1)
    scene.add(right)

    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(3, 3),
      new THREE.MeshPhysicalMaterial({
        color: 0xffffff,
        metalness: 0,
        roughness: 0.02,
        transmission: 1,
        thickness: 0,
        ior: 1.5,
      }),
    ))
    return scene
  }

  function centerEdgeContrast(rgba) {
    const left = meanRegion(rgba, width, 38, 30, 46, 66)
    const right = meanRegion(rgba, width, 50, 30, 58, 66)
    return Math.abs((left.r - left.b) - (right.r - right.b))
  }

  return {
    name: 'physical-transmission-resolution-scale',
    scene: makeScene(),
    camera,
    options: { ...options, transmissionResolutionScale: 0.125 },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.9,
    browserReference: false,
    render(renderer) {
      const previousScale = renderer.transmissionResolutionScale
      try {
        renderer.transmissionResolutionScale = 1
        const fullResolution = renderer.render(makeScene(), camera, options)
        renderer.transmissionResolutionScale = 0.125
        const lowResolution = renderer.render(makeScene(), camera, options)
        const optionLowResolution = renderer.render(makeScene(), camera, {
          ...options,
          transmissionResolutionScale: 0.125,
        })
        const optionFullResolution = renderer.render(makeScene(), camera, {
          ...options,
          transmissionResolutionScale: 1,
        })

        fullContrast = centerEdgeContrast(fullResolution)
        lowContrast = centerEdgeContrast(lowResolution)
        optionFullContrast = centerEdgeContrast(optionFullResolution)
        return optionLowResolution
      } finally {
        renderer.transmissionResolutionScale = previousScale
      }
    },
    validate(rgba) {
      const optionLowContrast = centerEdgeContrast(rgba)
      if (!(fullContrast > 80)) {
        throw new Error(`full-resolution transmission scene color should preserve the edge, contrast=${fullContrast.toFixed(1)}`)
      }
      if (!(lowContrast < fullContrast - 20)) {
        throw new Error(`low renderer transmissionResolutionScale should soften the scene-color edge, low=${lowContrast.toFixed(1)} full=${fullContrast.toFixed(1)}`)
      }
      if (!(optionLowContrast < fullContrast - 20)) {
        throw new Error(`options.transmissionResolutionScale should soften the scene-color edge, optionLow=${optionLowContrast.toFixed(1)} full=${fullContrast.toFixed(1)}`)
      }
      if (!(optionFullContrast > 80)) {
        throw new Error(`options.transmissionResolutionScale should override low renderer state, optionFull=${optionFullContrast.toFixed(1)}`)
      }
    },
  }
}
