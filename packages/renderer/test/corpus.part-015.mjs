import * as THREE from 'three'
import { CORPUS_RENDER_SIZE, countRegionPixels, makeCamera, meanRegion, pixelAt } from './corpus.part-001.mjs'
export function shadowMaterialReceiverCorpus() {
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

export function shadowMaterialOpacityCorpus() {
  const camera = makeCamera([0.8, 1.5, 3.0], [0, -0.35, 0])
  const options = { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' }
  const stats = {}

  function makeScene(opacity) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(4, 4),
      new THREE.ShadowMaterial({ opacity }),
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

    return scene
  }

  function shadowLuminance(rgba) {
    const shadow = meanRegion(rgba, options.width, 24, 42, 30, 54)
    return shadow.r + shadow.g + shadow.b
  }

  return {
    name: 'shadow-material-opacity-scaling',
    scene: makeScene(0.35),
    camera,
    options,
    background: [255, 255, 255],
    minNonBackgroundRatio: 0.01,
    render(renderer) {
      const opaque = renderer.render(makeScene(1), camera, options)
      const translucent = renderer.render(makeScene(0.35), camera, options)
      stats.opaqueLum = shadowLuminance(opaque)
      stats.translucentLum = shadowLuminance(translucent)
      return opaque
    },
    validate() {
      if (!(stats.opaqueLum < 720)) {
        throw new Error(`opaque ShadowMaterial should render a visible received shadow, stats=${JSON.stringify(stats)}`)
      }
      if (!(stats.translucentLum > stats.opaqueLum + 40)) {
        throw new Error(`lower ShadowMaterial opacity should blend more background through received shadows, stats=${JSON.stringify(stats)}`)
      }
    },
  }
}

export function shadowMaterialOutputColorSpaceCorpus() {
  const camera = makeCamera([0, 6, 8], [0, 0, 0])
  const options = {
    width: CORPUS_RENDER_SIZE,
    height: CORPUS_RENDER_SIZE,
    format: 'rgba',
    outputColorSpace: THREE.LinearSRGBColorSpace,
  }
  const stats = {}

  function makeScene() {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ color: 0x808080, opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const caster = new THREE.Mesh(
      new THREE.BoxGeometry(3, 3, 3),
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    )
    caster.position.y = 1.5
    caster.castShadow = true
    scene.add(caster)

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(8, 6, 0)
    light.target.position.set(0, 0, 0)
    light.castShadow = true
    light.shadow.mapSize.set(512, 512)
    light.shadow.camera.left = -7
    light.shadow.camera.right = 7
    light.shadow.camera.top = 7
    light.shadow.camera.bottom = -7
    light.shadow.camera.near = 0.1
    light.shadow.camera.far = 16
    scene.add(light, light.target)

    return scene
  }

  function shadowMean(rgba) {
    return meanRegion(rgba, options.width, 32, 32, 64, 64)
  }

  return {
    name: 'shadow-material-output-color-space',
    scene: makeScene(),
    camera,
    options,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.01,
    render(renderer) {
      const srgb = renderer.render(makeScene(), camera, { ...options, outputColorSpace: THREE.SRGBColorSpace })
      const linear = renderer.render(makeScene(), camera, { ...options, outputColorSpace: THREE.LinearSRGBColorSpace })
      stats.srgb = shadowMean(srgb)
      stats.linear = shadowMean(linear)
      return srgb
    },
    validate() {
      if (!(stats.srgb.r > stats.linear.r + 15)) {
        throw new Error(`sRGB ShadowMaterial output should apply display conversion, stats=${JSON.stringify(stats)}`)
      }
      if (!(Math.abs(stats.srgb.r - stats.srgb.g) < 2 && Math.abs(stats.linear.r - stats.linear.g) < 2)) {
        throw new Error(`ShadowMaterial gray output should stay neutral across color spaces, stats=${JSON.stringify(stats)}`)
      }
    },
  }
}

export function shadowMaterialFogOptOutCorpus() {
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

export function dashedLineMaterialCorpus() {
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
        (r, g, b) => r > 120 && g > 120 && r > b + 35 && g > b + 15,
      )
      const gap = pixelAt(rgba, width, 48, 48)
      if (!(yellowPixels > 30 && yellowPixels < 90 && gap.r < 5 && gap.g < 5 && gap.b < 5)) {
        throw new Error(`dashed-line corpus should render sparse yellow dashes with visible gaps, got yellow=${yellowPixels} gap=${JSON.stringify(gap)}`)
      }
    },
  }
}

export function dashedLineMaterialTextureCorpus() {
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

export function dashedLineMaterialUvChannelCorpus() {
  const map = new THREE.DataTexture(new Uint8Array([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ]), 2, 1, THREE.RGBAFormat)
  map.channel = 1
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter
  map.needsUpdate = true

  const alphaMap = new THREE.DataTexture(new Uint8Array([
    255, 255, 255, 255,
    255, 0, 255, 255,
  ]), 2, 1, THREE.RGBAFormat)
  alphaMap.channel = 2
  alphaMap.magFilter = THREE.NearestFilter
  alphaMap.minFilter = THREE.NearestFilter
  alphaMap.needsUpdate = true

  const geometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-1.45, 0, 0),
    new THREE.Vector3(1.45, 0, 0),
  ])
  geometry.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([
    0.25, 0.5,
    0.25, 0.5,
  ]), 2))
  geometry.setAttribute('uv1', new THREE.BufferAttribute(new Float32Array([
    0.75, 0.5,
    0.75, 0.5,
  ]), 2))
  geometry.setAttribute('uv2', new THREE.BufferAttribute(new Float32Array([
    0.25, 0.5,
    0.25, 0.5,
  ]), 2))
  geometry.setAttribute('lineDistance', new THREE.BufferAttribute(new Float32Array([0, 2.9]), 1))

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const material = new THREE.LineDashedMaterial({
    alphaTest: 0.5,
    color: 0xffffff,
    dashSize: 0.38,
    gapSize: 0.2,
    linewidth: 8,
    map,
    scale: 1,
  })
  material.alphaMap = alphaMap
  scene.add(new THREE.Line(
    geometry,
    material,
  ))

  return {
    name: 'line-dashed-material-uv-channel-selection',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.001,
    validate(rgba, { width, height }) {
      const greenPixels = countRegionPixels(
        rgba,
        width,
        0,
        0,
        width,
        height,
        (r, g, b) => g > 100 && g > r + 40 && g > b + 40,
      )
      const redPixels = countRegionPixels(
        rgba,
        width,
        0,
        0,
        width,
        height,
        (r, g, b) => r > 100 && r > g + 40 && r > b + 40,
      )
      if (!(greenPixels > 80 && redPixels < 5)) {
        throw new Error(`dashed-line UV-channel corpus should render green uv1-selected dashes with uv2 alpha kept opaque, green=${greenPixels} red=${redPixels}`)
      }
    },
  }
}
