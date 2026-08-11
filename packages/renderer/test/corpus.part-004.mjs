import * as THREE from 'three'
import { CORPUS_RENDER_SIZE, countRegionPixels, makeCamera, meanAbsDiff, meanRegion, pixelAt } from './corpus.part-001.mjs'
export function toneMappingStateCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(4, 4),
    new THREE.MeshBasicMaterial({ color: new THREE.Color(1, 1, 1) }),
  ))

  const camera = makeCamera([0, 0, 3])
  const options = {
    width: CORPUS_RENDER_SIZE,
    height: CORPUS_RENDER_SIZE,
    format: 'rgba',
    outputColorSpace: THREE.LinearSRGBColorSpace,
  }
  const stats = {}

  function luminance(rgba) {
    const mean = meanRegion(rgba, options.width, 32, 32, 64, 64)
    return mean.r
  }

  return {
    name: 'renderer-tone-mapping-state',
    scene,
    camera,
    options,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.4,
    browserReference: false,
    render(renderer) {
      const previousToneMapping = renderer.toneMapping
      const previousToneMappingExposure = renderer.toneMappingExposure
      try {
        stats.mapped = luminance(renderer.render(scene, camera, options))

        renderer.toneMapping = THREE.NoToneMapping
        const unmapped = renderer.render(scene, camera, options)
        stats.unmapped = luminance(unmapped)
        stats.optionMapped = luminance(renderer.render(scene, camera, {
          ...options,
          toneMapping: THREE.ACESFilmicToneMapping,
        }))

        renderer.toneMapping = THREE.ACESFilmicToneMapping
        renderer.toneMappingExposure = 0.25
        stats.dimmed = luminance(renderer.render(scene, camera, options))
        stats.optionBrightened = luminance(renderer.render(scene, camera, {
          ...options,
          toneMappingExposure: 2,
        }))
        renderer.toneMappingExposure = 2
        stats.brightened = luminance(renderer.render(scene, camera, options))

        renderer.toneMappingExposure = 1
        renderer.toneMapping = THREE.LinearToneMapping
        stats.linear = luminance(renderer.render(scene, camera, options))
        renderer.toneMapping = THREE.ReinhardToneMapping
        stats.reinhard = luminance(renderer.render(scene, camera, options))
        renderer.toneMapping = THREE.CineonToneMapping
        stats.cineon = luminance(renderer.render(scene, camera, options))
        renderer.toneMapping = THREE.CustomToneMapping
        stats.custom = luminance(renderer.render(scene, camera, options))
        renderer.toneMapping = THREE.AgXToneMapping
        stats.agx = luminance(renderer.render(scene, camera, options))
        renderer.toneMapping = THREE.NeutralToneMapping
        stats.neutral = luminance(renderer.render(scene, camera, options))

        return unmapped
      } finally {
        renderer.toneMapping = previousToneMapping
        renderer.toneMappingExposure = previousToneMappingExposure
      }
    },
    validate() {
      if (!(stats.unmapped > stats.mapped + 35 && stats.unmapped > 245)) {
        throw new Error(`NoToneMapping should preserve bright linear output, stats=${JSON.stringify(stats)}`)
      }
      if (!(stats.unmapped > stats.optionMapped + 35)) {
        throw new Error(`options.toneMapping should override renderer state, stats=${JSON.stringify(stats)}`)
      }
      if (!(stats.brightened > stats.dimmed + 60 && stats.optionBrightened > stats.dimmed + 60)) {
        throw new Error(`toneMappingExposure state/options should scale output, stats=${JSON.stringify(stats)}`)
      }
      if (!(stats.linear > 245)) {
        throw new Error(`LinearToneMapping should preserve white output, stats=${JSON.stringify(stats)}`)
      }
      if (!(stats.reinhard < stats.linear - 70)) {
        throw new Error(`ReinhardToneMapping should compress white, stats=${JSON.stringify(stats)}`)
      }
      if (!(stats.cineon > stats.reinhard + 20 && stats.cineon < stats.linear - 20)) {
        throw new Error(`CineonToneMapping should land between Reinhard and Linear, stats=${JSON.stringify(stats)}`)
      }
      if (!(Math.abs(stats.custom - stats.linear) < 2)) {
        throw new Error(`CustomToneMapping should use the default identity mapping, stats=${JSON.stringify(stats)}`)
      }
      if (!(stats.agx > 0 && stats.agx < stats.linear - 20 && stats.neutral > 0 && stats.neutral < stats.linear - 5)) {
        throw new Error(`AgX and Neutral tone mapping should produce finite compressed output, stats=${JSON.stringify(stats)}`)
      }
    },
  }
}

export function postProcessingOptionsCorpus() {
  const colorScene = new THREE.Scene()
  colorScene.background = new THREE.Color(0.25, 0.5, 0.75)

  const redScene = new THREE.Scene()
  redScene.background = new THREE.Color(1, 0, 0)

  const camera = makeCamera([0, 0, 3])
  const options = {
    width: CORPUS_RENDER_SIZE,
    height: CORPUS_RENDER_SIZE,
    format: 'rgba',
    outputColorSpace: THREE.LinearSRGBColorSpace,
    toneMapping: THREE.NoToneMapping,
  }
  const stats = {}

  function frameMean(rgba) {
    return meanRegion(rgba, options.width, 0, 0, options.width, options.height)
  }

  return {
    name: 'post-processing-options',
    scene: redScene,
    camera,
    options,
    background: [255, 0, 0],
    minNonBackgroundRatio: 0.9,
    browserReference: false,
    render(renderer) {
      const base = renderer.render(colorScene, camera, options)
      stats.base = frameMean(base)
      stats.exposed = frameMean(renderer.render(colorScene, camera, {
        ...options,
        postProcessing: { exposure: 1 },
      }))
      stats.contrasted = frameMean(renderer.render(colorScene, camera, {
        ...options,
        postProcessing: { contrast: 2 },
      }))
      stats.grayscale = frameMean(renderer.render(colorScene, camera, {
        ...options,
        postProcessing: { grayscale: true },
      }))
      const disabled = renderer.render(colorScene, camera, {
        ...options,
        postProcessing: {
          enabled: false,
          exposure: 4,
          contrast: 4,
          saturation: 0,
          vignette: 1,
          grayscale: true,
          invert: true,
        },
      })
      stats.disabledDiff = meanAbsDiff(base, disabled)

      const vignette = renderer.render(redScene, camera, {
        ...options,
        postProcessing: { vignette: 1 },
      })
      stats.vignetteCenter = meanRegion(vignette, options.width, 36, 36, 60, 60)
      stats.vignetteCorner = meanRegion(vignette, options.width, 0, 0, 16, 16)

      const processed = renderer.render(redScene, camera, {
        ...options,
        postProcessing: { invert: 1, saturation: 1.5, vignette: 0.25 },
      })
      stats.processedCenter = meanRegion(processed, options.width, 36, 36, 60, 60)
      return processed
    },
    validate() {
      if (!(stats.exposed.r > stats.base.r + 45 && stats.exposed.g > stats.base.g + 80)) {
        throw new Error(`post-processing exposure should brighten output, stats=${JSON.stringify(stats)}`)
      }
      if (!(stats.contrasted.r < stats.base.r - 45 && stats.contrasted.b > stats.base.b + 45)) {
        throw new Error(`post-processing contrast should expand color values around mid gray, stats=${JSON.stringify(stats)}`)
      }
      if (!(Math.max(stats.grayscale.r, stats.grayscale.g, stats.grayscale.b) - Math.min(stats.grayscale.r, stats.grayscale.g, stats.grayscale.b) < 3)) {
        throw new Error(`post-processing grayscale should equalize color channels, stats=${JSON.stringify(stats)}`)
      }
      if (!(stats.disabledDiff < 0.1)) {
        throw new Error(`postProcessing.enabled=false should bypass effects, stats=${JSON.stringify(stats)}`)
      }
      if (!(stats.processedCenter.g > stats.processedCenter.r + 80 && stats.processedCenter.b > stats.processedCenter.r + 80)) {
        throw new Error(`post-processing invert/saturation should turn red toward cyan, stats=${JSON.stringify(stats)}`)
      }
      if (!(stats.vignetteCenter.r > stats.vignetteCorner.r + 180 && stats.vignetteCorner.r < 40)) {
        throw new Error(`post-processing vignette should darken image corners, stats=${JSON.stringify(stats)}`)
      }
    },
  }
}

export function customWgslPremultipliedCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const material = new THREE.ShaderMaterial({
    blending: THREE.NoBlending,
    premultipliedAlpha: true,
    transparent: true,
  })
  material.userData.headlessThreeRenderer = {
    fragmentWgsl: 'return vec4<f32>(0.0, 1.0, 0.0, alpha * 0.5);',
  }
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

  return {
    name: 'custom-wgsl-premultiplied-alpha',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minMeanAlpha: 120,
    browserReference: false,
    validate(rgba, { width }) {
      const center = pixelAt(rgba, width, 48, 48)
      if (!(center.g > 60 && center.g < 150 && center.a > 120 && center.a < 140)) {
        throw new Error(`custom WGSL premultiplied corpus should output half-alpha premultiplied green, got ${JSON.stringify(center)}`)
      }
    },
  }
}

export function sceneOverrideMaterialCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const left = new THREE.Mesh(
    new THREE.PlaneGeometry(0.85, 1.35),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  )
  left.position.x = -0.45
  scene.add(left)
  const right = new THREE.Mesh(
    new THREE.PlaneGeometry(0.85, 1.35),
    new THREE.MeshBasicMaterial({ color: 0x0000ff }),
  )
  right.position.x = 0.45
  scene.add(right)
  scene.overrideMaterial = new THREE.MeshBasicMaterial({ color: 0x00ff00, toneMapped: false })

  return {
    name: 'scene-override-material',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.08,
    validate(rgba, { width }) {
      const leftMean = meanRegion(rgba, width, 24, 30, 42, 66)
      const rightMean = meanRegion(rgba, width, 54, 30, 72, 66)
      for (const [label, mean] of [['left', leftMean], ['right', rightMean]]) {
        if (!(mean.g > mean.r + 100 && mean.g > mean.b + 100)) {
          throw new Error(`scene override-material corpus should replace ${label} source material with green, got ${JSON.stringify(mean)}`)
        }
      }
    },
  }
}

export function maskRenderModeCorpus() {
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

export function objectIdRenderModeCorpus() {
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

export function normalRenderModeCorpus() {
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

export function depthRenderModeCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.85, 0.1, 0.1)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(1.35, 1.35),
    new THREE.MeshBasicMaterial({ color: 0x0088ff }),
  ))

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  return {
    name: 'depth-render-mode-plane',
    scene,
    camera,
    options: {
      width: CORPUS_RENDER_SIZE,
      height: CORPUS_RENDER_SIZE,
      format: 'rgba',
      renderMode: 'depth',
    },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.08,
    validate(rgba, { width }) {
      const center = meanRegion(rgba, width, 32, 32, 64, 64)
      const corner = meanRegion(rgba, width, 0, 0, 8, 8)
      if (!(center.r > 150 && Math.abs(center.r - center.g) < 2 && Math.abs(center.r - center.b) < 2 && corner.r < 2 && corner.g < 2 && corner.b < 2)) {
        throw new Error(`depth render corpus should render grayscale depth on black, got center=${JSON.stringify(center)} corner=${JSON.stringify(corner)}`)
      }
    },
  }
}

export function renderModeAlphaHashCutoutCorpus() {
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  const options = { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' }
  const stats = {}
  const renderModes = ['mask', 'object-id', 'normal', 'depth']

  function makeScene(alphaHash) {
    const scene = new THREE.Scene()
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(1.2, 1.2),
      new THREE.MeshBasicMaterial({
        alphaHash,
        color: 0xffffff,
        opacity: alphaHash ? 0.35 : 1,
      }),
    ))
    return scene
  }

  function visiblePixels(rgba) {
    return countRegionPixels(rgba, options.width, 30, 30, 66, 66, (r, g, b) => r > 0 || g > 0 || b > 0)
  }

  return {
    name: 'render-mode-alpha-hash-cutouts',
    scene: makeScene(true),
    camera,
    options,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.02,
    browserReference: false,
    render(renderer) {
      let hashedMask
      for (const renderMode of renderModes) {
        const opaque = renderer.render(makeScene(false), camera, { ...options, renderMode })
        const hashed = renderer.render(makeScene(true), camera, { ...options, renderMode })
        stats[renderMode] = {
          opaquePixels: visiblePixels(opaque),
          hashedPixels: visiblePixels(hashed),
        }
        if (renderMode === 'mask') {
          hashedMask = hashed
        }
      }
      return hashedMask
    },
    validate() {
      for (const [renderMode, { opaquePixels, hashedPixels }] of Object.entries(stats)) {
        if (!(opaquePixels > 1100)) {
          throw new Error(`${renderMode} render mode should fill the sampled opaque region, opaque=${opaquePixels}`)
        }
        if (!(hashedPixels > 100)) {
          throw new Error(`${renderMode} render mode should retain some alphaHash pixels, hashed=${hashedPixels}`)
        }
        if (!(hashedPixels < opaquePixels - 250)) {
          throw new Error(`${renderMode} render mode should discard alphaHash pixels, hashed=${hashedPixels} opaque=${opaquePixels}`)
        }
      }
    },
  }
}
