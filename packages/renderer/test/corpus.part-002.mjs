import * as THREE from 'three'
import { CORPUS_RENDER_SIZE, alphaCoverageBandPixels, makeCamera, meanAbsDiff, meanRegion, pixelAt, solidTexture } from './corpus.part-001.mjs'
export function alphaToCoverageClippingCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(1.7, 1.7),
    new THREE.MeshBasicMaterial({
      color: 0xffffff,
      alphaToCoverage: true,
      clippingPlanes: [new THREE.Plane(new THREE.Vector3(1, 1, 0).normalize(), 0)],
    }),
  ))

  return {
    name: 'alpha-to-coverage-clipping-plane',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: {
      width: CORPUS_RENDER_SIZE,
      height: CORPUS_RENDER_SIZE,
      format: 'rgba',
      sampleCount: 4,
      outputColorSpace: THREE.LinearSRGBColorSpace,
      localClippingEnabled: true,
    },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.08,
    validate(rgba, { width }) {
      const softPixels = alphaCoverageBandPixels(rgba, width)
      if (softPixels < 24) {
        throw new Error(`alpha-to-coverage clipping corpus should produce a soft clipping band, got ${softPixels} partial pixels`)
      }
    },
  }
}

export function signedRawTextureCorpus() {
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

export function stencilRenderStateCorpus() {
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
      if (!(masked.b > masked.g + 120 && masked.b > masked.r + 160 && unmasked.r < 2 && unmasked.g < 2 && unmasked.b < 2)) {
        throw new Error(`stencil corpus should render only the masked blue side, got masked=${JSON.stringify(masked)} unmasked=${JSON.stringify(unmasked)}`)
      }
    },
  }
}

export function customBlendingCorpus() {
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

export function materialRenderStateNoopCorpus() {
  const camera = makeCamera([0, 0, 3])
  const options = {
    width: CORPUS_RENDER_SIZE,
    height: CORPUS_RENDER_SIZE,
    format: 'rgba',
    outputColorSpace: THREE.LinearSRGBColorSpace,
    toneMapping: THREE.NoToneMapping,
  }
  const stats = { precisionDiffs: {} }

  function planeScene(configure = () => {}) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const material = new THREE.MeshBasicMaterial({ color: 0x40a0ff })
    configure(material)
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))
    return scene
  }

  function wireframeScene(hints = {}) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.BoxGeometry(1.5, 1.5, 1.5),
      new THREE.MeshBasicMaterial({
        color: 0xffffff,
        wireframe: true,
        ...hints,
      }),
    ))
    return scene
  }

  function renderPlane(renderer, configure) {
    return renderer.render(planeScene(configure), camera, options)
  }

  function renderWireframe(renderer, hints) {
    return renderer.render(wireframeScene(hints), makeCamera([0, 0, 4]), options)
  }

  return {
    name: 'material-render-state-noops',
    scene: planeScene(),
    camera,
    options,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.1,
    render(renderer) {
      const baseline = renderPlane(renderer)
      stats.center = meanRegion(baseline, options.width, 32, 32, 64, 64)
      stats.ditheringDiff = meanAbsDiff(baseline, renderPlane(renderer, (material) => {
        material.dithering = true
      }))
      for (const precision of ['highp', 'mediump', 'lowp']) {
        stats.precisionDiffs[precision] = meanAbsDiff(baseline, renderPlane(renderer, (material) => {
          material.precision = precision
        }))
      }

      const transparentBase = renderPlane(renderer, (material) => {
        material.opacity = 0.5
        material.side = THREE.DoubleSide
        material.transparent = true
      })
      const forceSinglePass = renderPlane(renderer, (material) => {
        material.forceSinglePass = true
        material.opacity = 0.5
        material.side = THREE.DoubleSide
        material.transparent = true
      })
      stats.forceSinglePassDiff = meanAbsDiff(transparentBase, forceSinglePass)

      const wireframeBase = renderWireframe(renderer, {})
      const wireframeHints = renderWireframe(renderer, {
        wireframeLinewidth: 4,
        wireframeLinecap: 'butt',
        wireframeLinejoin: 'bevel',
      })
      stats.wireframeHintDiff = meanAbsDiff(wireframeBase, wireframeHints)

      return baseline
    },
    validate() {
      if (!(stats.center.b > 240 && stats.center.g > 80 && stats.center.g < 105 && stats.center.r > 8 && stats.center.r < 25)) {
        throw new Error(`render-state no-op baseline should render the blue plane, stats=${JSON.stringify(stats)}`)
      }
      if (!(stats.ditheringDiff < 0.1)) {
        throw new Error(`material.dithering should not alter native output, stats=${JSON.stringify(stats)}`)
      }
      for (const [precision, diff] of Object.entries(stats.precisionDiffs)) {
        if (!(diff < 0.1)) {
          throw new Error(`material.precision=${precision} should not alter native output, stats=${JSON.stringify(stats)}`)
        }
      }
      if (!(stats.forceSinglePassDiff < 0.1)) {
        throw new Error(`material.forceSinglePass should not alter native output, stats=${JSON.stringify(stats)}`)
      }
      if (!(stats.wireframeHintDiff < 0.1)) {
        throw new Error(`mesh wireframe line hints should not alter native output, stats=${JSON.stringify(stats)}`)
      }
    },
  }
}

export function backgroundOverrideCorpus() {
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
      if (!(center.g > center.r + 80 && center.g > center.b + 40 && corner.r < 2 && corner.g < 2 && corner.b < 2)) {
        throw new Error(`background override corpus should render green mesh on black option background, got center=${JSON.stringify(center)} corner=${JSON.stringify(corner)}`)
      }
    },
  }
}

export function optionBackgroundTextureControlsCorpus() {
  const background = new THREE.DataTexture(new Uint8Array([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ]), 2, 1, THREE.RGBAFormat)
  background.magFilter = THREE.NearestFilter
  background.minFilter = THREE.NearestFilter
  background.needsUpdate = true

  const scene = new THREE.Scene()
  scene.background = solidTexture(0, 0, 255)
  scene.backgroundIntensity = 0
  scene.backgroundBlurriness = 1

  const camera = makeCamera([0, 0, 3])
  const options = {
    width: CORPUS_RENDER_SIZE,
    height: CORPUS_RENDER_SIZE,
    format: 'rgba',
    background,
    backgroundIntensity: 1,
  }
  const samples = {}

  return {
    name: 'option-background-texture-controls',
    scene,
    camera,
    options,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.95,
    render(renderer) {
      const sharp = renderer.render(scene, camera, options)
      const dimmed = renderer.render(scene, camera, {
        ...options,
        backgroundIntensity: 0.25,
      })
      const blurred = renderer.render(scene, camera, {
        ...options,
        backgroundBlurriness: 1,
      })
      samples.sharp = meanRegion(sharp, options.width, 42, 28, 47, 68)
      samples.dimmed = meanRegion(dimmed, options.width, 42, 28, 47, 68)
      samples.blurred = meanRegion(blurred, options.width, 42, 28, 47, 68)
      return sharp
    },
    validate() {
      if (!(samples.sharp.r > samples.sharp.g + 120 && samples.sharp.r > 180)) {
        throw new Error(`option background texture controls should keep the option texture sharp and ignore scene intensity, samples=${JSON.stringify(samples)}`)
      }
      if (!(samples.dimmed.r < samples.sharp.r - 60)) {
        throw new Error(`options.backgroundIntensity should dim the option texture, samples=${JSON.stringify(samples)}`)
      }
      if (!(samples.blurred.g > samples.sharp.g + 80 && samples.sharp.r > samples.blurred.r + 20)) {
        throw new Error(`options.backgroundBlurriness should soften the option texture, samples=${JSON.stringify(samples)}`)
      }
    },
  }
}

export function rendererClearColorFallbackCorpus() {
  const emptyScene = new THREE.Scene()
  const backgroundScene = new THREE.Scene()
  backgroundScene.background = new THREE.Color(1, 0, 0)
  const camera = makeCamera([0, 0, 3])
  const options = { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' }
  const samples = {}

  function sampleCenter(rgba) {
    return pixelAt(rgba, options.width, 48, 48)
  }

  function assertPixel(pixel, expected, label) {
    const close = Math.abs(pixel.r - expected[0]) <= 1
      && Math.abs(pixel.g - expected[1]) <= 1
      && Math.abs(pixel.b - expected[2]) <= 1
      && Math.abs(pixel.a - expected[3]) <= 1
    if (!close) {
      throw new Error(`${label} expected rgba(${expected.join(', ')}), got ${JSON.stringify(pixel)}`)
    }
  }

  return {
    name: 'renderer-clear-color-fallback',
    scene: emptyScene,
    camera,
    options,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.9,
    minMeanAlpha: 180,
    browserReference: false,
    render(renderer) {
      const previousColor = renderer.getClearColor(new THREE.Color())
      const previousAlpha = renderer.getClearAlpha()
      try {
        renderer.setClearColor(0x204080, 0.5)
        const clearFallback = renderer.render(emptyScene, camera, options)
        samples.clearFallback = sampleCenter(clearFallback)

        const sceneBackground = renderer.render(backgroundScene, camera, options)
        samples.sceneBackground = sampleCenter(sceneBackground)

        renderer.setClearAlpha(0.25)
        const nullBackground = renderer.render(backgroundScene, camera, { ...options, background: null })
        samples.nullBackground = sampleCenter(nullBackground)

        const optionBackground = renderer.render(backgroundScene, camera, {
          ...options,
          background: [0, 1, 0, 0.75],
        })
        samples.optionBackground = sampleCenter(optionBackground)
        return optionBackground
      } finally {
        renderer.setClearColor(previousColor, previousAlpha)
      }
    },
    validate() {
      assertPixel(samples.clearFallback, [0x20, 0x40, 0x80, 128], 'Renderer clear color fallback')
      assertPixel(samples.sceneBackground, [255, 0, 0, 255], 'scene background precedence')
      assertPixel(samples.nullBackground, [0x20, 0x40, 0x80, 64], 'options.background null clear fallback')
      assertPixel(samples.optionBackground, [0, 255, 0, 191], 'options.background override')
    },
  }
}

export function lightProbeCorpus() {
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
    background: [48, 48, 56],
    minNonBackgroundRatio: 0.02,
    validate(rgba, { width }) {
      const center = meanRegion(rgba, width, 40, 40, 56, 56)
      if (!(center.r > 140 && center.r < 210 && center.g > 110 && center.r > center.b + 40 && center.g > center.b + 20)) {
        throw new Error(`LightProbe diffuse corpus should render a warm lit sphere, got ${JSON.stringify(center)}`)
      }
    },
  }
}
