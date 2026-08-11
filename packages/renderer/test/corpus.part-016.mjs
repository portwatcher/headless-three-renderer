import * as THREE from 'three'
import { CORPUS_RENDER_SIZE, countRegionPixels, makeCamera, meanAbsDiff, meanRegion } from './corpus.part-001.mjs'
export function dashedLineMaterialCustomDistanceCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const grouped = new THREE.BufferGeometry()
  grouped.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -1.55, 0.38, 0,
    -0.35, 0.38, 0,
    0.35, 0.38, 0,
    1.55, 0.38, 0,
  ]), 3))
  grouped.setAttribute('lineDistance', new THREE.BufferAttribute(new Float32Array([
    0, 1,
    0, 1,
  ]), 1))
  grouped.addGroup(0, 2, 0)
  grouped.addGroup(2, 2, 1)
  scene.add(new THREE.LineSegments(grouped, [
    new THREE.LineBasicMaterial({ color: 0xff3333, linewidth: 7 }),
    new THREE.LineDashedMaterial({
      color: 0x44ff44,
      dashSize: 0.45,
      gapSize: 10,
      linewidth: 7,
      scale: 1,
    }),
  ]))

  const descending = new THREE.BufferGeometry()
  descending.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -1.2, -0.38, 0,
    1.2, -0.38, 0,
  ]), 3))
  descending.setAttribute('lineDistance', new THREE.BufferAttribute(new Float32Array([
    2,
    0,
  ]), 1))
  scene.add(new THREE.Line(
    descending,
    new THREE.LineDashedMaterial({
      color: 0xffff66,
      dashSize: 0.6,
      gapSize: 10,
      linewidth: 7,
      scale: 1,
    }),
  ))

  const camera = new THREE.OrthographicCamera(-1.8, 1.8, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  return {
    name: 'line-dashed-material-custom-distance',
    scene,
    camera,
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.002,
    browserReference: false,
    validate(rgba, { width }) {
      const redPixels = countRegionPixels(rgba, width, 9, 25, 42, 37, (r, g, b) => r > g + 45 && r > b + 45)
      const greenDashPixels = countRegionPixels(rgba, width, 58, 25, 70, 37, (r, g, b) => g > r + 45 && g > b + 45)
      const greenGapPixels = countRegionPixels(rgba, width, 77, 25, 90, 37, (r, g, b) => g > r + 45 && g > b + 45)
      const descendingGapPixels = countRegionPixels(rgba, width, 15, 60, 30, 70, (r, g, b) => r > 120 && g > 120 && r > b + 25 && g > b + 25)
      const descendingDashPixels = countRegionPixels(rgba, width, 56, 60, 80, 70, (r, g, b) => r > 120 && g > 120 && r > b + 25 && g > b + 25)
      if (redPixels < 20) {
        throw new Error(`custom-distance corpus should keep the solid red material-array group visible (${redPixels})`)
      }
      if (greenDashPixels < 20 || greenGapPixels > 1) {
        throw new Error(`custom-distance corpus should preserve the dashed material-array gap, dash=${greenDashPixels} gap=${greenGapPixels}`)
      }
      if (descendingGapPixels > 1 || descendingDashPixels < 20) {
        throw new Error(`custom-distance corpus should honor descending lineDistance spans, gap=${descendingGapPixels} dash=${descendingDashPixels}`)
      }
    },
  }
}

export function dashedLineMaterialLineLoopDistanceCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const geometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-0.8, -0.8, 0),
    new THREE.Vector3(0.8, -0.8, 0),
    new THREE.Vector3(0.8, 0.8, 0),
    new THREE.Vector3(-0.8, 0.8, 0),
  ])
  const line = new THREE.LineLoop(geometry, new THREE.LineDashedMaterial({
    color: 0x66ddff,
    dashSize: 2,
    gapSize: 10,
    linewidth: 1,
    scale: 1,
  }))
  line.computeLineDistances()
  scene.add(line)

  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 1.2, -1.2, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  return {
    name: 'line-dashed-material-lineloop-distance',
    scene,
    camera,
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.002,
    validate(rgba, { width }) {
      const isCyan = (r, g, b) => b > 120 && g > 100 && b > r + 35 && g > r + 25
      const lowerClosingPixels = countRegionPixels(rgba, width, 12, 48, 22, 76, isCyan)
      const upperClosingPixels = countRegionPixels(rgba, width, 12, 18, 22, 34, isCyan)
      if (lowerClosingPixels < 8 || upperClosingPixels > 1) {
        throw new Error(`LineLoop dashed corpus should interpolate closing lineDistance into lower dash and upper gap, lower=${lowerClosingPixels} upper=${upperClosingPixels}`)
      }
    },
  }
}

export function dashedLineMaterialWideLineCorpus() {
  function makeScene(linewidth) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)

    const geometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-1.45, 0, 0),
      new THREE.Vector3(1.45, 0, 0),
    ])
    geometry.setAttribute('lineDistance', new THREE.BufferAttribute(new Float32Array([0, 2.9]), 1))

    scene.add(new THREE.Line(
      geometry,
      new THREE.LineDashedMaterial({
        color: 0x55ccff,
        dashSize: 0.4,
        gapSize: 0.22,
        linewidth,
        scale: 1,
      }),
    ))
    return scene
  }

  const camera = makeCamera([0, 0, 3])
  const options = { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' }
  const thinScene = makeScene(1)
  const wideScene = makeScene(10)
  let thinPixels = 0
  let widePixels = 0

  return {
    name: 'line-dashed-material-wide-linewidth',
    scene: wideScene,
    camera,
    options,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.002,
    browserReference: false,
    render(renderer) {
      const isCyan = (r, g, b) => b > 120 && g > 90 && b > r + 30 && g > r + 20
      const thin = renderer.render(thinScene, camera, options)
      thinPixels = countRegionPixels(thin, options.width, 0, 0, options.width, options.height, isCyan)
      const wide = renderer.render(wideScene, camera, options)
      widePixels = countRegionPixels(wide, options.width, 0, 0, options.width, options.height, isCyan)
      return wide
    },
    validate() {
      if (!(thinPixels > 0 && widePixels > thinPixels * 3)) {
        throw new Error(`wide dashed-line corpus should expand linewidth coverage, thin=${thinPixels} wide=${widePixels}`)
      }
    },
  }
}

export function lineMaterialNoopCorpus() {
  const camera = makeCamera([0, 0, 3])
  const options = { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' }
  const stats = { capJoinDiffs: {}, receiveShadowDiffs: {} }

  function makeLine(kind, configureMaterial = () => {}, receiveShadow = false) {
    const geometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-1.2, 0, 0),
      new THREE.Vector3(0, 0.5, 0),
      new THREE.Vector3(1.2, 0, 0),
    ])
    const material = kind === 'basic'
      ? new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 8 })
      : new THREE.LineDashedMaterial({
        color: 0xffffff,
        dashSize: 0.3,
        gapSize: 0.15,
        linewidth: 8,
        scale: 1,
      })
    configureMaterial(material)

    const line = new THREE.Line(geometry, material)
    line.receiveShadow = receiveShadow
    if (kind === 'dashed') line.computeLineDistances()
    return line
  }

  function makeScene(kind, configureMaterial = () => {}, receiveShadow = false) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(makeLine(kind, configureMaterial, receiveShadow))
    return scene
  }

  function renderLine(renderer, kind, configureMaterial = () => {}, receiveShadow = false) {
    return renderer.render(makeScene(kind, configureMaterial, receiveShadow), camera, options)
  }

  return {
    name: 'line-material-noop-state',
    scene: makeScene('basic'),
    camera,
    options,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.004,
    browserReference: false,
    render(renderer) {
      let returned
      for (const kind of ['basic', 'dashed']) {
        const baseline = renderLine(renderer, kind)
        if (kind === 'basic') returned = baseline
        stats.capJoinDiffs[kind] = meanAbsDiff(baseline, renderLine(renderer, kind, (material) => {
          material.linecap = 'butt'
          material.linejoin = 'bevel'
        }))
        stats.receiveShadowDiffs[kind] = meanAbsDiff(baseline, renderLine(renderer, kind, () => {}, true))
      }
      return returned
    },
    validate(rgba, { width, height }) {
      const whitePixels = countRegionPixels(rgba, width, 0, 0, width, height, (r, g, b) => r > 160 && g > 160 && b > 160)
      if (!(whitePixels > 400)) {
        throw new Error(`line no-op corpus should render a visible wide white line, whitePixels=${whitePixels}`)
      }
      for (const [kind, diff] of Object.entries(stats.capJoinDiffs)) {
        if (!(diff < 0.1)) {
          throw new Error(`${kind} linecap/linejoin should be accepted as no-op state, stats=${JSON.stringify(stats)}`)
        }
      }
      for (const [kind, diff] of Object.entries(stats.receiveShadowDiffs)) {
        if (!(diff < 0.1)) {
          throw new Error(`${kind} receiveShadow should be accepted as an unlit line no-op, stats=${JSON.stringify(stats)}`)
        }
      }
    },
  }
}

export function lineBasicMaterialUvChannelCorpus() {
  const map = new THREE.DataTexture(new Uint8Array([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ]), 2, 1, THREE.RGBAFormat)
  map.channel = 1
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter
  map.needsUpdate = true

  const alphaMap = new THREE.DataTexture(new Uint8Array([
    255, 0, 255, 255,
    255, 255, 255, 255,
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
    0.75, 0.5,
    0.75, 0.5,
  ]), 2))

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const material = new THREE.LineBasicMaterial({
    alphaTest: 0.5,
    color: 0xffffff,
    linewidth: 8,
    map,
  })
  material.alphaMap = alphaMap
  scene.add(new THREE.Line(
    geometry,
    material,
  ))

  return {
    name: 'line-basic-material-uv-channel-selection',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.01,
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
      if (!(greenPixels > 500 && redPixels < 5)) {
        throw new Error(`LineBasicMaterial UV-channel corpus should render green uv1-selected color while uv2 keeps alpha opaque, green=${greenPixels} red=${redPixels}`)
      }
    },
  }
}

export function pointsMaterialTextureCorpus() {
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

export function pointsMaterialUvChannelCorpus() {
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

  const material = new THREE.PointsMaterial({
    color: 0xffffff,
    map,
    size: 32,
    sizeAttenuation: false,
  })

  const spriteUvGeometry = new THREE.BufferGeometry()
  spriteUvGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.58, 0, 0,
  ]), 3))
  scene.add(new THREE.Points(spriteUvGeometry, material))

  const selectedUvGeometry = new THREE.BufferGeometry()
  selectedUvGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    0.58, 0, 0,
  ]), 3))
  selectedUvGeometry.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([
    0.25, 0.5,
  ]), 2))
  selectedUvGeometry.setAttribute('uv1', new THREE.BufferAttribute(new Float32Array([
    0.75, 0.5,
  ]), 2))
  scene.add(new THREE.Points(selectedUvGeometry, material.clone()))

  return {
    name: 'points-material-uv-channel-selection',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.08,
    validate(rgba, { width }) {
      const spriteLeft = meanRegion(rgba, width, 16, 40, 26, 56)
      const spriteRight = meanRegion(rgba, width, 32, 40, 42, 56)
      const selected = meanRegion(rgba, width, 60, 40, 76, 56)
      if (!(spriteLeft.r > spriteLeft.g + 60 && spriteRight.g > spriteRight.r + 60 && selected.g > selected.r + 60)) {
        throw new Error(`points UV corpus should use point-sprite UVs without geometry UVs and selected uv1 when present, got spriteLeft=${JSON.stringify(spriteLeft)} spriteRight=${JSON.stringify(spriteRight)} selected=${JSON.stringify(selected)}`)
      }
    },
  }
}
