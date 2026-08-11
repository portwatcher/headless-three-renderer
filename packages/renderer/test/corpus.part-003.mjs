import * as THREE from 'three'
import { CORPUS_RENDER_SIZE, constantUvPlane, countRegionPixels, makeCamera, meanRegion, pixelAt, setTextureMatrixOffset, solidTexture } from './corpus.part-001.mjs'
export function lightProbeMaterialModelsCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const probe = new THREE.LightProbe(undefined, 1.5)
  for (const coefficient of probe.sh.coefficients) {
    coefficient.set(0, 0, 0)
  }
  probe.sh.coefficients[0].set(1.0, 0.18, 0.08)
  scene.add(probe)

  const materials = [
    new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, metalness: 0 }),
    new THREE.MeshPhysicalMaterial({ color: 0xffffff, roughness: 1, metalness: 0 }),
    new THREE.MeshPhongMaterial({ color: 0xffffff, shininess: 20 }),
    new THREE.MeshToonMaterial({ color: 0xffffff }),
  ]

  for (const [index, material] of materials.entries()) {
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(0.42, 1.2), material)
    mesh.position.x = (index - 1.5) * 0.5
    scene.add(mesh)
  }

  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  return {
    name: 'light-probe-lit-material-models',
    scene,
    camera,
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.08,
    validate(rgba, { width }) {
      const regions = [
        ['standard', meanRegion(rgba, width, 14, 30, 22, 66)],
        ['physical', meanRegion(rgba, width, 34, 30, 42, 66)],
        ['phong', meanRegion(rgba, width, 54, 30, 62, 66)],
        ['toon', meanRegion(rgba, width, 74, 30, 82, 66)],
      ]
      for (const [label, mean] of regions) {
        if (!(mean.r > mean.g + 20 && mean.r > mean.b + 20)) {
          throw new Error(`LightProbe should tint ${label} corpus material red (${mean.r}, ${mean.g}, ${mean.b})`)
        }
      }
    },
  }
}

export function lightProbeEnvironmentMaterialModelsCorpus() {
  function makeGreenEnvironment() {
    const texture = solidTexture(0, 255, 0)
    texture.mapping = THREE.EquirectangularReflectionMapping
    return texture
  }

  function makeRedProbe() {
    const probe = new THREE.LightProbe(undefined, 1.5)
    for (const coefficient of probe.sh.coefficients) {
      coefficient.set(0, 0, 0)
    }
    probe.sh.coefficients[0].set(1, 0, 0)
    return probe
  }

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.environment = makeGreenEnvironment()
  scene.environmentIntensity = 2.5
  scene.add(makeRedProbe())

  const materials = [
    ['standard', new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, metalness: 0 })],
    ['physical', new THREE.MeshPhysicalMaterial({ color: 0xffffff, roughness: 1, metalness: 0 })],
    ['lambert', new THREE.MeshLambertMaterial({ color: 0xffffff })],
    ['phong', new THREE.MeshPhongMaterial({ color: 0xffffff, shininess: 20 })],
    ['toon', new THREE.MeshToonMaterial({ color: 0xffffff })],
  ]

  for (const [index, [, material]] of materials.entries()) {
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(0.34, 1.2), material)
    mesh.position.x = (index - 2) * 0.42
    scene.add(mesh)
  }

  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  return {
    name: 'light-probe-environment-material-models',
    scene,
    camera,
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.1,
    browserReference: false,
    validate(rgba, { width }) {
      const sampleXs = [16, 32, 48, 64, 80]
      for (const [index, [label]] of materials.entries()) {
        const x = sampleXs[index]
        const mean = meanRegion(rgba, width, x - 4, 30, x + 4, 66)
        if (!(mean.r > 180 && mean.g > 200 && mean.g > mean.b + 65)) {
          throw new Error(`LightProbe/environment corpus should light ${label} with red probe plus green environment, got ${JSON.stringify(mean)}`)
        }
      }
    },
  }
}

export function linearFogCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.03)
  scene.fog = new THREE.Fog(0x3366ff, 1.2, 3.2)

  const fogged = new THREE.Mesh(
    new THREE.PlaneGeometry(0.82, 1.45),
    new THREE.MeshBasicMaterial({ color: 0xff4422 }),
  )
  fogged.position.set(-0.48, 0, 0)
  scene.add(fogged)

  const unfogged = new THREE.Mesh(
    new THREE.PlaneGeometry(0.82, 1.45),
    new THREE.MeshBasicMaterial({ color: 0xff4422, fog: false }),
  )
  unfogged.position.set(0.48, 0, 0)
  scene.add(unfogged)

  return {
    name: 'linear-fog-material-opt-out',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [39, 39, 48],
    minNonBackgroundRatio: 0.08,
    validate(rgba, { width }) {
      const fogged = meanRegion(rgba, width, 20, 24, 38, 72)
      const unfogged = meanRegion(rgba, width, 58, 24, 76, 72)
      if (!(fogged.b > fogged.r + 180 && unfogged.r > unfogged.b + 180)) {
        throw new Error(`linear fog corpus should keep only the opt-out panel red, got fogged=${JSON.stringify(fogged)} unfogged=${JSON.stringify(unfogged)}`)
      }
    },
  }
}

export function fogExp2MixedObjectCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.fog = new THREE.FogExp2(0x2255ff, 0.45)

  const foggedColor = 0xff2200
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(0.44, 0.44),
    new THREE.MeshBasicMaterial({ color: foggedColor }),
  )
  mesh.position.set(-0.55, 0.42, 0)
  scene.add(mesh)

  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ color: foggedColor, fog: true }))
  sprite.position.set(0.55, 0.42, 0)
  sprite.scale.set(0.46, 0.46, 1)
  scene.add(sprite)

  const pointGeometry = new THREE.BufferGeometry()
  pointGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-0.55, -0.42, 0]), 3))
  scene.add(new THREE.Points(pointGeometry, new THREE.PointsMaterial({
    color: foggedColor,
    fog: true,
    size: 28,
    sizeAttenuation: false,
  })))

  const lineGeometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0.26, -0.55, 0),
    new THREE.Vector3(0.84, -0.25, 0),
  ])
  scene.add(new THREE.Line(lineGeometry, new THREE.LineBasicMaterial({
    color: foggedColor,
    fog: true,
    linewidth: 8,
  })))

  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 1.2, -1.2, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  return {
    name: 'fog-exp2-mixed-object-types',
    scene,
    camera,
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.02,
    validate(rgba, { width, height }) {
      const isFoggedBlue = (r, g, b) => b > 130 && b > r + 60 && b > g + 50
      const regions = [
        ['mesh', countRegionPixels(rgba, width, 18, 20, 34, 36, isFoggedBlue)],
        ['sprite', countRegionPixels(rgba, width, 62, 20, 78, 36, isFoggedBlue)],
        ['point', countRegionPixels(rgba, width, 15, 58, 37, 80, isFoggedBlue)],
        ['line', countRegionPixels(rgba, width, 58, 52, 84, height, isFoggedBlue)],
      ]
      for (const [label, pixels] of regions) {
        if (pixels < 20) {
          throw new Error(`FogExp2 mixed-object corpus should tint ${label} toward blue, got ${pixels} blue pixels`)
        }
      }
    },
  }
}

export function textureMatrixColorSpaceCorpus() {
  const texture = new THREE.DataTexture(new Uint8Array([
    64, 64, 64, 255,
    64, 64, 64, 255,
    224, 224, 224, 255,
    224, 224, 224, 255,
  ]), 2, 2, THREE.RGBAFormat)
  texture.colorSpace = THREE.SRGBColorSpace
  texture.magFilter = THREE.NearestFilter
  texture.minFilter = THREE.NearestFilter
  texture.wrapS = THREE.RepeatWrapping
  texture.wrapT = THREE.RepeatWrapping
  texture.matrixAutoUpdate = false
  texture.matrix.setUvTransform(0.12, 0.18, 1.7, 1.7, Math.PI / 2, 0.5, 0.5)
  texture.needsUpdate = true

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.025)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(1.7, 1.7),
    new THREE.MeshBasicMaterial({ color: 0xffffff, map: texture }),
  ))

  return {
    name: 'texture-matrix-srgb-map',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [39, 39, 44],
    minNonBackgroundRatio: 0.1,
    validate(rgba, { width }) {
      const transformedBright = pixelAt(rgba, width, 48, 48)
      const transformedDark = pixelAt(rgba, width, 30, 37)
      if (!(transformedBright.r > 180 && transformedBright.g > 180 && transformedBright.b > 180 && transformedDark.r < 80 && transformedDark.g < 80 && transformedDark.b < 80)) {
        throw new Error(`texture matrix corpus should sample distinct sRGB bright/dark texels, got bright=${JSON.stringify(transformedBright)} dark=${JSON.stringify(transformedDark)}`)
      }
    },
  }
}

export function phongSpecularMapMatrixCorpus() {
  const specularMap = new THREE.DataTexture(new Uint8Array([
    0, 0, 0, 255,
    255, 0, 0, 255,
  ]), 2, 1, THREE.RGBAFormat)
  specularMap.magFilter = THREE.NearestFilter
  specularMap.minFilter = THREE.NearestFilter
  setTextureMatrixOffset(specularMap, 0.5)
  specularMap.needsUpdate = true

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    constantUvPlane(0.25, 0.5, 1.3, 1.3),
    new THREE.MeshPhongMaterial({
      color: 0x000000,
      specular: 0xffffff,
      shininess: 4,
      specularMap,
    }),
  ))

  const light = new THREE.DirectionalLight(0xffffff, 8)
  light.position.set(0, 0, 3)
  scene.add(light)

  return {
    name: 'phong-specular-map-matrix',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.08,
    validate(rgba, { width }) {
      const center = meanRegion(rgba, width, 31, 31, 65, 65)
      if (!(center.r > 80 && center.g > 80 && center.b > 80)) {
        throw new Error(`specularMap explicit matrix should sample the bright specular texel, got ${JSON.stringify(center)}`)
      }
    },
  }
}

export function matrixSlotTexture(data) {
  const texture = new THREE.DataTexture(new Uint8Array(data), 2, 1, THREE.RGBAFormat)
  texture.magFilter = THREE.NearestFilter
  texture.minFilter = THREE.NearestFilter
  setTextureMatrixOffset(texture, 0.5)
  texture.needsUpdate = true
  return texture
}

export function textureSlotMatrixCorpus() {
  const alphaMap = matrixSlotTexture([
    255, 0, 0, 255,
    255, 255, 0, 255,
  ])
  const aoMap = matrixSlotTexture([
    255, 255, 255, 255,
    0, 0, 0, 255,
  ])
  const emissiveMap = matrixSlotTexture([
    0, 0, 0, 255,
    0, 255, 0, 255,
  ])

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0.16)
  const panelGeometry = () => constantUvPlane(0.25, 0.5, 0.72, 0.9)

  const alphaPanel = new THREE.Mesh(
    panelGeometry(),
    new THREE.MeshBasicMaterial({
      color: 0xff3311,
      alphaMap,
      alphaTest: 0.5,
    }),
  )
  alphaPanel.position.x = -0.86

  const aoPanel = new THREE.Mesh(
    panelGeometry(),
    new THREE.MeshBasicMaterial({
      color: 0xffffaa,
      aoMap,
      aoMapIntensity: 1,
    }),
  )

  const emissivePanel = new THREE.Mesh(
    panelGeometry(),
    new THREE.MeshStandardMaterial({
      color: 0x000000,
      emissive: 0x00ff00,
      emissiveIntensity: 2,
      emissiveMap,
    }),
  )
  emissivePanel.position.x = 0.86

  scene.add(alphaPanel, aoPanel, emissivePanel)

  return {
    name: 'texture-slot-explicit-matrices',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 113],
    minNonBackgroundRatio: 0.06,
    validate(rgba, { width }) {
      const alpha = meanRegion(rgba, width, 8, 34, 26, 62)
      const ao = meanRegion(rgba, width, 38, 34, 56, 62)
      const emissive = meanRegion(rgba, width, 67, 34, 85, 62)
      if (!(alpha.r > alpha.b + 55 && alpha.r > alpha.g + 35)) {
        throw new Error(`alphaMap explicit matrix should reveal the red panel, got ${JSON.stringify(alpha)}`)
      }
      if (!(ao.r < 12 && ao.g < 12 && ao.b < 12)) {
        throw new Error(`aoMap explicit matrix should darken the center panel, got ${JSON.stringify(ao)}`)
      }
      if (!(emissive.g > emissive.r + 35 && emissive.g > emissive.b + 55)) {
        throw new Error(`emissiveMap explicit matrix should light the green panel, got ${JSON.stringify(emissive)}`)
      }
    },
  }
}

export function lightMapCorpus() {
  const lightMap = new THREE.DataTexture(new Uint8Array([
    255, 48, 16, 255,
    24, 255, 96, 255,
  ]), 2, 1, THREE.RGBAFormat)
  lightMap.magFilter = THREE.NearestFilter
  lightMap.minFilter = THREE.NearestFilter
  lightMap.needsUpdate = true

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.015, 0.015, 0.018)

  const red = new THREE.Mesh(
    constantUvPlane(0.25, 0.5, 0.78, 1.15),
    new THREE.MeshBasicMaterial({
      color: 0xffffff,
      lightMap,
      lightMapIntensity: 1.25,
    }),
  )
  red.position.x = -0.48

  const green = new THREE.Mesh(
    constantUvPlane(0.75, 0.5, 0.78, 1.15),
    new THREE.MeshBasicMaterial({
      color: 0xffffff,
      lightMap,
      lightMapIntensity: 1.25,
    }),
  )
  green.position.x = 0.48

  scene.add(red, green)

  return {
    name: 'light-map-material-texture',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [14, 14, 17],
    minNonBackgroundRatio: 0.16,
    validate(rgba, { width }) {
      const redPanel = meanRegion(rgba, width, 19, 28, 39, 68)
      const greenPanel = meanRegion(rgba, width, 57, 28, 77, 68)
      if (!(redPanel.r > redPanel.g + 80 && redPanel.r > redPanel.b + 100)) {
        throw new Error(`lightMap corpus should tint the left panel red, got ${JSON.stringify(redPanel)}`)
      }
      if (!(greenPanel.g > greenPanel.r + 80 && greenPanel.g > greenPanel.b + 50)) {
        throw new Error(`lightMap corpus should tint the right panel green, got ${JSON.stringify(greenPanel)}`)
      }
    },
  }
}

export function linearOutputColorSpaceCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.18, 0.18, 0.18)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(1.55, 1.55),
    new THREE.MeshBasicMaterial({ color: new THREE.Color(0.5, 0.22, 0.08) }),
  ))

  return {
    name: 'linear-output-color-space',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: {
      width: CORPUS_RENDER_SIZE,
      height: CORPUS_RENDER_SIZE,
      format: 'rgba',
      outputColorSpace: THREE.LinearSRGBColorSpace,
    },
    background: [46, 46, 46],
    minNonBackgroundRatio: 0.08,
    validate(rgba, { width }) {
      const center = pixelAt(rgba, width, 48, 48)
      const corner = pixelAt(rgba, width, 4, 4)
      if (!(center.r > 130 && center.r < 160 && center.g > 60 && center.g < 85 && center.b > 15 && center.b < 35 && corner.r > 40 && corner.r < 55 && corner.g > 40 && corner.g < 55 && corner.b > 40 && corner.b < 55)) {
        throw new Error(`linear output corpus should preserve linear RGB values, got center=${JSON.stringify(center)} corner=${JSON.stringify(corner)}`)
      }
    },
  }
}
