import * as THREE from 'three'
import { CORPUS_RENDER_SIZE, constantUvPlane, environmentTexture, gradientTexture, makeCamera, meanAbsDiff, meanRegion, pixelAt, setTextureMatrixOffset } from './corpus.part-001.mjs'
export function avatarLikeCorpus() {
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
    background: [69, 75, 89],
    backgroundTolerance: 8,
    minNonBackgroundRatio: 0.035,
    validate(rgba, { width }) {
      const head = pixelAt(rgba, width, 48, 28)
      const body = pixelAt(rgba, width, 48, 54)
      const corner = pixelAt(rgba, width, 4, 4)
      if (!(head.r > head.b + 20 && head.g > head.b + 5 && body.b > body.r + 45 && body.g > body.r + 20 && Math.abs(corner.r - 69) <= 1 && Math.abs(corner.g - 75) <= 1 && Math.abs(corner.b - 89) <= 1)) {
        throw new Error(`avatar corpus should render warm head and blue toon body, got head=${JSON.stringify(head)} body=${JSON.stringify(body)} corner=${JSON.stringify(corner)}`)
      }
    },
  }
}

export function physicalIblShadowCorpus() {
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
    background: [56, 56, 63],
    validate(rgba, { width }) {
      const sphere = pixelAt(rgba, width, 48, 48)
      const ground = meanRegion(rgba, width, 60, 42, 76, 58)
      const corner = pixelAt(rgba, width, 4, 4)
      if (!(sphere.r > 200 && sphere.g > 210 && sphere.b > 220 && ground.r > 80 && ground.g > 80 && ground.b > 90 && corner.r === 56 && corner.g === 56 && corner.b === 63)) {
        throw new Error(`physical IBL shadow corpus should render a bright physical sphere and visible shadowed ground, got sphere=${JSON.stringify(sphere)} ground=${JSON.stringify(ground)} corner=${JSON.stringify(corner)}`)
      }
    },
  }
}

export function physicalClearcoatMapCorpus() {
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

  function makeScene(parameters) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.environment = environmentTexture()
    scene.environmentIntensity = 2
    scene.add(new THREE.Mesh(
      constantUvPlane(0.25, 0.5),
      new THREE.MeshPhysicalMaterial({
        color: 0x000000,
        roughness: 1,
        metalness: 0,
        ...parameters,
      }),
    ))
    return scene
  }

  function makeClearcoatScene(offsetX) {
    return makeScene({
      clearcoat: 1,
      clearcoatRoughness: 0.04,
      clearcoatMap: makeMap([
        0, 0, 0, 255,
        255, 0, 0, 255,
      ], offsetX),
    })
  }

  function makeClearcoatRoughnessScene(offsetX) {
    return makeScene({
      clearcoat: 1,
      clearcoatRoughness: 1,
      clearcoatRoughnessMap: makeMap([
        0, 0, 0, 255,
        0, 255, 0, 255,
      ], offsetX),
    })
  }

  function makeClearcoatNormalScene(offsetX) {
    return makeScene({
      clearcoat: 1,
      clearcoatRoughness: 0.04,
      clearcoatNormalMap: makeMap([
        128, 128, 255, 255,
        255, 128, 128, 255,
      ], offsetX),
      clearcoatNormalScale: new THREE.Vector2(1, 1),
    })
  }

  function luminance(mean) {
    return 0.2126 * mean.r + 0.7152 * mean.g + 0.0722 * mean.b
  }

  return {
    name: 'physical-clearcoat-map-slots',
    scene: makeClearcoatScene(0.5),
    camera,
    options,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.08,
    browserReference: false,
    render(renderer) {
      const clearcoatPrimary = renderer.render(makeClearcoatScene(0), camera, options)
      const clearcoatShifted = renderer.render(makeClearcoatScene(0.5), camera, options)
      const roughnessPrimary = renderer.render(makeClearcoatRoughnessScene(0), camera, options)
      const roughnessShifted = renderer.render(makeClearcoatRoughnessScene(0.5), camera, options)
      const normalPrimary = renderer.render(makeClearcoatNormalScene(0), camera, options)
      const normalShifted = renderer.render(makeClearcoatNormalScene(0.5), camera, options)
      stats.clearcoatPrimary = luminance(meanRegion(clearcoatPrimary, options.width, 0, 0, options.width, options.height))
      stats.clearcoatShifted = luminance(meanRegion(clearcoatShifted, options.width, 0, 0, options.width, options.height))
      stats.roughnessPrimary = luminance(meanRegion(roughnessPrimary, options.width, 0, 0, options.width, options.height))
      stats.roughnessShifted = luminance(meanRegion(roughnessShifted, options.width, 0, 0, options.width, options.height))
      stats.normalDiff = meanAbsDiff(normalPrimary, normalShifted)
      return clearcoatShifted
    },
    validate() {
      if (!(stats.clearcoatShifted > stats.clearcoatPrimary + 50)) {
        throw new Error(`physical clearcoat corpus should enable shifted clearcoatMap highlights, stats=${JSON.stringify(stats)}`)
      }
      if (!(stats.roughnessPrimary > stats.roughnessShifted + 10)) {
        throw new Error(`physical clearcoat corpus should sample shifted rough clearcoatRoughnessMap texels, stats=${JSON.stringify(stats)}`)
      }
      if (!(stats.normalDiff > 2)) {
        throw new Error(`physical clearcoat corpus should sample shifted clearcoatNormalMap texels, stats=${JSON.stringify(stats)}`)
      }
    },
  }
}

export function physicalSheenMapCorpus() {
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

  function makeScene(material) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.environment = environmentTexture()
    scene.environmentIntensity = 2
    scene.add(new THREE.Mesh(constantUvPlane(0.25, 0.5), material))
    return scene
  }

  function makeSheenColorScene(offsetX) {
    return makeScene(new THREE.MeshPhysicalMaterial({
      color: 0x000000,
      roughness: 1,
      metalness: 0,
      sheen: 1,
      sheenColor: new THREE.Color(1, 1, 1),
      sheenRoughness: 0.35,
      sheenColorMap: makeMap([
        0, 0, 0, 255,
        255, 0, 0, 255,
      ], offsetX),
    }))
  }

  function makeSheenRoughnessScene(offsetX) {
    return makeScene(new THREE.MeshPhysicalMaterial({
      color: 0x000000,
      roughness: 1,
      metalness: 0,
      sheen: 1,
      sheenColor: new THREE.Color(1, 0, 0),
      sheenRoughness: 1,
      sheenRoughnessMap: makeMap([
        0, 0, 0, 0,
        0, 0, 0, 255,
      ], offsetX),
    }))
  }

  return {
    name: 'physical-sheen-map-slots',
    scene: makeSheenColorScene(0.5),
    camera,
    options,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.08,
    browserReference: false,
    render(renderer) {
      const colorPrimary = renderer.render(makeSheenColorScene(0), camera, options)
      const colorShifted = renderer.render(makeSheenColorScene(0.5), camera, options)
      const roughnessPrimary = renderer.render(makeSheenRoughnessScene(0), camera, options)
      const roughnessShifted = renderer.render(makeSheenRoughnessScene(0.5), camera, options)
      stats.colorPrimary = meanRegion(colorPrimary, options.width, 0, 0, options.width, options.height)
      stats.colorShifted = meanRegion(colorShifted, options.width, 0, 0, options.width, options.height)
      stats.roughnessDiff = meanAbsDiff(roughnessPrimary, roughnessShifted)
      return colorShifted
    },
    validate() {
      if (!(stats.colorShifted.r > stats.colorPrimary.r + 3 && stats.colorShifted.r > stats.colorShifted.g + 3)) {
        throw new Error(`physical sheen corpus should tint sheenColorMap output red, stats=${JSON.stringify(stats)}`)
      }
      if (!(stats.roughnessDiff > 5)) {
        throw new Error(`physical sheen corpus should sample shifted sheenRoughnessMap texel, stats=${JSON.stringify(stats)}`)
      }
    },
  }
}

export function physicalSpecularMapCorpus() {
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

  function addSpecularLight(scene) {
    const light = new THREE.PointLight(0xffffff, 300)
    light.position.set(0, 0, 2)
    scene.add(light)
  }

  function makeSpecularColorScene(offsetX) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.25, 0.5),
      new THREE.MeshPhysicalMaterial({
        color: 0x000000,
        roughness: 0.08,
        metalness: 0,
        specularIntensity: 1,
        specularColor: new THREE.Color(1, 1, 1),
        specularColorMap: makeMap([
          0, 0, 0, 255,
          255, 0, 0, 255,
        ], offsetX),
      }),
    ))
    addSpecularLight(scene)
    return scene
  }

  function makeSpecularIntensityScene(offsetX) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.25, 0.5),
      new THREE.MeshPhysicalMaterial({
        color: 0x000000,
        roughness: 0.08,
        metalness: 0,
        specularIntensity: 1,
        specularColor: new THREE.Color(1, 1, 1),
        specularIntensityMap: makeMap([
          0, 0, 0, 0,
          0, 0, 0, 255,
        ], offsetX),
      }),
    ))
    addSpecularLight(scene)
    return scene
  }

  function maxLuminance(rgba) {
    let max = 0
    for (let i = 0; i < rgba.length; i += 4) {
      const lum = 0.2126 * rgba[i] + 0.7152 * rgba[i + 1] + 0.0722 * rgba[i + 2]
      if (lum > max) max = lum
    }
    return max
  }

  return {
    name: 'physical-specular-map-slots',
    scene: makeSpecularColorScene(0.5),
    camera,
    options,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.08,
    browserReference: false,
    render(renderer) {
      const colorPrimary = renderer.render(makeSpecularColorScene(0), camera, options)
      const colorShifted = renderer.render(makeSpecularColorScene(0.5), camera, options)
      const intensityPrimary = renderer.render(makeSpecularIntensityScene(0), camera, options)
      const intensityShifted = renderer.render(makeSpecularIntensityScene(0.5), camera, options)
      stats.colorPrimary = meanRegion(colorPrimary, options.width, 0, 0, options.width, options.height)
      stats.colorShifted = meanRegion(colorShifted, options.width, 0, 0, options.width, options.height)
      stats.intensityPrimary = maxLuminance(intensityPrimary)
      stats.intensityShifted = maxLuminance(intensityShifted)
      return colorShifted
    },
    validate() {
      if (!(stats.colorShifted.r > stats.colorPrimary.r + 4 && stats.colorShifted.r > stats.colorShifted.g + 4)) {
        throw new Error(`physical specular corpus should tint specularColorMap highlights red, stats=${JSON.stringify(stats)}`)
      }
      if (!(stats.intensityShifted > stats.intensityPrimary + 40)) {
        throw new Error(`physical specular corpus should enable shifted specularIntensityMap highlights, stats=${JSON.stringify(stats)}`)
      }
    },
  }
}
