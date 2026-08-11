import * as THREE from 'three'
import { CORPUS_RENDER_SIZE, countRegionPixels, makeCamera, meanAbsDiff, meanRegion, solidTexture, spriteMapTexture } from './corpus.part-001.mjs'
export function renderModeTextureAlphaCutoutCorpus() {
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  const options = { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' }
  const renderModes = ['mask', 'object-id', 'normal', 'depth']
  const stats = {}
  const leftRegion = [24, 42, 35, 54]
  const rightRegion = [61, 42, 72, 54]

  const cases = [
    {
      name: 'baseTextureAlpha',
      makeDiscardedMaterial: () => new THREE.MeshBasicMaterial({
        map: solidTexture(255, 255, 255, 0),
        alphaTest: 0.5,
      }),
      makeVisibleMaterial: () => new THREE.MeshBasicMaterial({
        map: solidTexture(255, 255, 255, 255),
        alphaTest: 0.5,
      }),
    },
    {
      name: 'alphaMapGreen',
      makeDiscardedMaterial: () => new THREE.MeshBasicMaterial({
        alphaMap: solidTexture(255, 0, 255),
        alphaTest: 0.5,
      }),
      makeVisibleMaterial: () => new THREE.MeshBasicMaterial({
        alphaMap: solidTexture(255, 255, 255),
        alphaTest: 0.5,
      }),
    },
  ]

  function makeScene(makeDiscardedMaterial, makeVisibleMaterial) {
    const scene = new THREE.Scene()
    const discarded = new THREE.Mesh(new THREE.PlaneGeometry(0.75, 0.8), makeDiscardedMaterial())
    const visible = new THREE.Mesh(new THREE.PlaneGeometry(0.75, 0.8), makeVisibleMaterial())
    discarded.position.x = -0.5
    visible.position.x = 0.5
    scene.add(discarded, visible)
    return scene
  }

  function visiblePixels(rgba, region) {
    return countRegionPixels(rgba, options.width, ...region, (r, g, b) => r > 0 || g > 0 || b > 0)
  }

  return {
    name: 'render-mode-texture-alpha-cutouts',
    scene: makeScene(cases[0].makeDiscardedMaterial, cases[0].makeVisibleMaterial),
    camera,
    options,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.04,
    browserReference: false,
    render(renderer) {
      let referenceOutput
      for (const testCase of cases) {
        for (const renderMode of renderModes) {
          const rgba = renderer.render(
            makeScene(testCase.makeDiscardedMaterial, testCase.makeVisibleMaterial),
            camera,
            { ...options, renderMode },
          )
          stats[`${testCase.name}:${renderMode}`] = {
            leftPixels: visiblePixels(rgba, leftRegion),
            rightPixels: visiblePixels(rgba, rightRegion),
          }
          if (testCase.name === 'baseTextureAlpha' && renderMode === 'mask') {
            referenceOutput = rgba
          }
        }
      }
      return referenceOutput
    },
    validate() {
      for (const [label, { leftPixels, rightPixels }] of Object.entries(stats)) {
        if (!(leftPixels < 3)) {
          throw new Error(`${label} render-mode texture cutout should discard the left region, left=${leftPixels}`)
        }
        if (!(rightPixels > 110)) {
          throw new Error(`${label} render-mode texture cutout should keep the right region, right=${rightPixels}`)
        }
      }
    },
  }
}

export function renderModeMrtAuxiliaryCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(1.3, 1.3),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  )
  mesh.rotation.y = Math.PI * 0.22
  scene.add(mesh)

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const options = {
    width: CORPUS_RENDER_SIZE,
    height: CORPUS_RENDER_SIZE,
    format: 'rgba',
    outputColorSpace: THREE.LinearSRGBColorSpace,
  }
  let target

  function objectIdBytes(id) {
    const value = Math.max(1, Math.trunc(id)) & 0xffffff
    return [(value >> 16) & 0xff, (value >> 8) & 0xff, value & 0xff]
  }

  function assertRgbClose(mean, expected, label) {
    for (const [channel, index] of [['r', 0], ['g', 1], ['b', 2]]) {
      if (Math.abs(mean[channel] - expected[index]) > 1) {
        throw new Error(`${label} ${channel} should be ${expected[index]}, got ${mean[channel]}`)
      }
    }
  }

  return {
    name: 'render-mode-mrt-auxiliary-attachments',
    scene,
    camera,
    options,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.08,
    browserReference: false,
    render(renderer) {
      target = {
        isWebGLMultipleRenderTargets: true,
        textures: [
          {},
          { format: THREE.RGFormat, type: THREE.FloatType, userData: { headlessThreeRenderer: { renderMode: 'color' } } },
          { userData: { headlessThreeRenderer: { renderMode: 'mask' } } },
          { userData: { headlessThreeRenderer: { renderMode: 'object-id' } } },
          { userData: { headlessThreeRenderer: { renderMode: 'normal' } } },
          { userData: { headlessThreeRenderer: { renderMode: 'depth' } } },
        ],
      }
      return renderer.render(scene, camera, { ...options, target })
    },
    validate(rgba, { width }) {
      if (!target) {
        throw new Error('MRT auxiliary corpus did not render a target')
      }

      const primary = meanRegion(rgba, width, 40, 40, 56, 56)
      if (!(primary.r > 180 && primary.g < 10 && primary.b < 10)) {
        throw new Error(`primary MRT color attachment should render the red mesh, got ${JSON.stringify(primary)}`)
      }
      if (target.textures[0].image.data !== target.data) {
        throw new Error('primary MRT texture should reference the target RGBA data')
      }

      const colorCopy = target.textures[1].image.data
      const center = ((48 * width) + 48) * 2
      if (!(colorCopy instanceof Float32Array && colorCopy[center] > 0.7 && colorCopy[center + 1] < 0.05)) {
        throw new Error(`secondary MRT color attachment should render normalized RG floats, red=${colorCopy?.[center]} green=${colorCopy?.[center + 1]}`)
      }

      const maskCenter = meanRegion(target.textures[2].image.data, width, 40, 40, 56, 56)
      const maskCorner = meanRegion(target.textures[2].image.data, width, 0, 0, 8, 8)
      if (!(maskCenter.r > 250 && maskCenter.g > 250 && maskCenter.b > 250 && maskCorner.r < 2 && maskCorner.g < 2 && maskCorner.b < 2)) {
        throw new Error(`mask MRT attachment should render white geometry on black, center=${JSON.stringify(maskCenter)} corner=${JSON.stringify(maskCorner)}`)
      }

      const objectIdCenter = meanRegion(target.textures[3].image.data, width, 40, 40, 56, 56)
      const encoded = mesh.id + 1
      assertRgbClose(objectIdCenter, objectIdBytes(encoded), 'object-id MRT attachment')
      if (target.objectIdMap?.[String(encoded)]?.id !== mesh.id) {
        throw new Error('object-id MRT attachment should expose reverse lookup metadata')
      }

      const normalCenter = meanRegion(target.textures[4].image.data, width, 40, 40, 56, 56)
      if (!(normalCenter.r > 140 && normalCenter.b > 200)) {
        throw new Error(`normal MRT attachment should encode the tilted view normal, got ${JSON.stringify(normalCenter)}`)
      }

      const depthCenter = meanRegion(target.textures[5].image.data, width, 40, 40, 56, 56)
      const depthCorner = meanRegion(target.textures[5].image.data, width, 0, 0, 8, 8)
      if (!(depthCenter.r > depthCorner.r + 20 && depthCenter.r > 150)) {
        throw new Error(`depth MRT attachment should encode nearer mesh depth, center=${JSON.stringify(depthCenter)} corner=${JSON.stringify(depthCorner)}`)
      }
    },
  }
}

export function twoDimensionalBackgroundTextureCorpus() {
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

export function spriteMaterialCorpus() {
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
    background: [39, 44, 48],
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

export function spriteAlphaMapCorpus() {
  const alphaMap = new THREE.DataTexture(new Uint8Array([
    255, 0, 255, 255,
    255, 255, 255, 255,
  ]), 2, 1, THREE.RGBAFormat)
  alphaMap.magFilter = THREE.NearestFilter
  alphaMap.minFilter = THREE.NearestFilter
  alphaMap.needsUpdate = true

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0.16)
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    alphaMap,
    alphaTest: 0.5,
    color: 0x22ff88,
  }))
  sprite.scale.set(1.6, 1.2, 1)
  scene.add(sprite)

  return {
    name: 'sprite-material-alpha-map-cutout',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 41],
    minNonBackgroundRatio: 0.04,
    validate(rgba, { width }) {
      const cutout = meanRegion(rgba, width, 20, 34, 40, 62)
      const visible = meanRegion(rgba, width, 56, 34, 76, 62)
      if (!(cutout.b > cutout.g + 25 && visible.g > visible.b + 45 && visible.g > visible.r + 65)) {
        throw new Error(`sprite alpha-map corpus should cut out the left side and keep the right side green, got cutout=${JSON.stringify(cutout)} visible=${JSON.stringify(visible)}`)
      }
    },
  }
}

export function billboardAlphaCutoutCorpus() {
  function makeBillboardScene(kind, materialProps) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    if (kind === 'sprite') {
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial(materialProps))
      sprite.scale.set(1.2, 1.2, 1)
      scene.add(sprite)
    } else {
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))
      scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({
        ...materialProps,
        size: 48,
        sizeAttenuation: false,
      })))
    }
    return scene
  }

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  const options = { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' }
  const a2cOptions = { ...options, sampleCount: 4, outputColorSpace: THREE.LinearSRGBColorSpace }
  const opaqueProps = { color: 0xffffff, opacity: 1 }
  const hashedProps = { alphaHash: true, color: 0xffffff, opacity: 0.35 }
  const a2cProps = { alphaToCoverage: true, color: 0xffffff, opacity: 0.5, transparent: false }
  const opaqueScenes = {
    points: makeBillboardScene('points', opaqueProps),
    sprite: makeBillboardScene('sprite', opaqueProps),
  }
  const hashedScenes = {
    points: makeBillboardScene('points', hashedProps),
    sprite: makeBillboardScene('sprite', hashedProps),
  }
  const a2cScenes = {
    points: makeBillboardScene('points', a2cProps),
    sprite: makeBillboardScene('sprite', a2cProps),
  }
  const stats = new Map()

  function visiblePixels(rgba) {
    return countRegionPixels(
      rgba,
      options.width,
      24,
      24,
      72,
      72,
      (r, g, b) => r > 20 || g > 20 || b > 20,
    )
  }

  function centerMean(rgba) {
    return meanRegion(rgba, options.width, 36, 36, 60, 60)
  }

  return {
    name: 'billboard-alpha-cutouts',
    scene: a2cScenes.points,
    camera,
    options: a2cOptions,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.04,
    minMeanAlpha: 238,
    render(renderer) {
      let output = null
      for (const kind of ['sprite', 'points']) {
        const opaque = renderer.render(opaqueScenes[kind], camera, options)
        const hashed = renderer.render(hashedScenes[kind], camera, options)
        const a2c = renderer.render(a2cScenes[kind], camera, a2cOptions)
        stats.set(kind, {
          a2c: centerMean(a2c),
          hashedPixels: visiblePixels(hashed),
          opaquePixels: visiblePixels(opaque),
          opaque: centerMean(opaque),
        })
        if (kind === 'points') {
          output = a2c
        }
      }
      return output
    },
    validate() {
      for (const kind of ['sprite', 'points']) {
        const result = stats.get(kind)
        if (!result) {
          throw new Error(`billboard alpha corpus did not record ${kind} stats`)
        }
        if (!(result.opaquePixels > 1400)) {
          throw new Error(`${kind} opaque billboard should fill sampled region, got ${result.opaquePixels}`)
        }
        if (!(result.hashedPixels > 100 && result.hashedPixels < result.opaquePixels - 260)) {
          throw new Error(`${kind} alphaHash should keep sparse visible pixels, got hashed=${result.hashedPixels} opaque=${result.opaquePixels}`)
        }
        if (!(result.opaque.r > 170 && result.a2c.r > 30 && result.a2c.r < result.opaque.r - 80)) {
          throw new Error(`${kind} alphaToCoverage should resolve partial billboard coverage, got opaque=${JSON.stringify(result.opaque)} a2c=${JSON.stringify(result.a2c)}`)
        }
      }
    },
  }
}

export function billboardReceiveShadowNoopCorpus() {
  const camera = makeCamera([0, 0, 3])
  const options = { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' }
  const stats = { diffs: {}, means: {}, visiblePixels: {} }

  function makeScene(kind, receiveShadow = false) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    if (kind === 'sprite') {
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ color: 0xffffff }))
      sprite.receiveShadow = receiveShadow
      sprite.scale.set(1.2, 1.2, 1)
      scene.add(sprite)
    } else {
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))
      const points = new THREE.Points(geometry, new THREE.PointsMaterial({
        color: 0xffffff,
        size: 48,
        sizeAttenuation: false,
      }))
      points.receiveShadow = receiveShadow
      scene.add(points)
    }
    return scene
  }

  function renderBillboard(renderer, kind, receiveShadow = false) {
    return renderer.render(makeScene(kind, receiveShadow), camera, options)
  }

  return {
    name: 'billboard-receive-shadow-noop',
    scene: makeScene('sprite'),
    camera,
    options,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.04,
    render(renderer) {
      let returned
      for (const kind of ['sprite', 'points']) {
        const baseline = renderBillboard(renderer, kind)
        const receiveShadow = renderBillboard(renderer, kind, true)
        stats.diffs[kind] = meanAbsDiff(baseline, receiveShadow)
        stats.means[kind] = meanRegion(baseline, options.width, 36, 36, 60, 60)
        stats.visiblePixels[kind] = countRegionPixels(
          baseline,
          options.width,
          0,
          0,
          options.width,
          options.height,
          (r, g, b) => r > 160 && g > 160 && b > 160,
        )
        if (kind === 'sprite') {
          returned = baseline
        }
      }
      return returned
    },
    validate() {
      for (const kind of ['sprite', 'points']) {
        if (!(stats.visiblePixels[kind] > 1000)) {
          throw new Error(`${kind} receiveShadow no-op corpus should render a visible white billboard, stats=${JSON.stringify(stats)}`)
        }
        const mean = stats.means[kind]
        if (!(mean.r > 180 && mean.g > 180 && mean.b > 180)) {
          throw new Error(`${kind} receiveShadow no-op corpus should keep the unlit billboard bright, stats=${JSON.stringify(stats)}`)
        }
        if (!(stats.diffs[kind] < 0.1)) {
          throw new Error(`${kind} receiveShadow should be accepted as an unlit billboard no-op, stats=${JSON.stringify(stats)}`)
        }
      }
    },
  }
}
