import * as THREE from 'three'
import { CORPUS_RENDER_SIZE, gradientTexture, makeCamera, meanAbsDiff, meanRegion, solidTexture } from './corpus.part-001.mjs'
export function meshNormalMaterialObjectSpaceNormalMapCorpus() {
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -1, -1, 0,
    1, -1, 0,
    -1, 1, 0,
    1, -1, 0,
    1, 1, 0,
    -1, 1, 0,
  ]), 3))
  geometry.setAttribute('normal', new THREE.BufferAttribute(new Float32Array([
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
  ]), 3))
  geometry.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([
    0, 0,
    0, 1,
    1, 0,
    0, 1,
    1, 1,
    1, 0,
  ]), 2))

  function makeScene(normalMapType) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      geometry,
      new THREE.MeshNormalMaterial({
        normalMap: solidTexture(255, 128, 128),
        normalMapType,
      }),
    ))
    return scene
  }

  const tangentScene = makeScene(THREE.TangentSpaceNormalMap)
  const objectScene = makeScene(THREE.ObjectSpaceNormalMap)
  const camera = makeCamera([0, 0, 3])
  const options = { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' }
  let tangentCenter = null
  let objectCenter = null

  return {
    name: 'mesh-normal-material-object-space-normal-map',
    scene: objectScene,
    camera,
    options,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.35,
    render(renderer) {
      const tangent = renderer.render(tangentScene, camera, options)
      tangentCenter = meanRegion(tangent, options.width, 32, 32, 64, 64)
      const objectSpace = renderer.render(objectScene, camera, options)
      objectCenter = meanRegion(objectSpace, options.width, 32, 32, 64, 64)
      return objectSpace
    },
    validate() {
      if (!(tangentCenter.g > tangentCenter.r + 35 && objectCenter.r > objectCenter.g + 35)) {
        throw new Error(`object-space normal-map corpus should distinguish tangent/object normal interpretation, tangent=${JSON.stringify(tangentCenter)} object=${JSON.stringify(objectCenter)}`)
      }
    },
  }
}

export function meshNormalMaterialBumpMapCorpus() {
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

  function makeScene(bumpScale) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshNormalMaterial({
        bumpMap: makeBumpMap(),
        bumpScale,
      }),
    ))
    return scene
  }

  const flatScene = makeScene(0)
  const bumpedScene = makeScene(8)
  const camera = makeCamera([0, 0, 3])
  const options = { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' }
  let bumpDiff = 0

  return {
    name: 'mesh-normal-material-bump-map',
    scene: bumpedScene,
    camera,
    options,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.35,
    render(renderer) {
      const flat = renderer.render(flatScene, camera, options).slice()
      const bumped = renderer.render(bumpedScene, camera, options)
      bumpDiff = meanAbsDiff(flat, bumped)
      return bumped
    },
    validate() {
      if (!(bumpDiff > 2)) {
        throw new Error(`bump-map corpus should perturb MeshNormalMaterial output, diff=${bumpDiff.toFixed(3)}`)
      }
    },
  }
}

export function meshMatcapMaterialCorpus() {
  const matcap = new THREE.DataTexture(new Uint8Array([
    40, 70, 130, 255,
    245, 210, 140, 255,
    90, 170, 210, 255,
    255, 255, 240, 255,
  ]), 2, 2, THREE.RGBAFormat)
  matcap.colorSpace = THREE.SRGBColorSpace
  matcap.magFilter = THREE.LinearFilter
  matcap.minFilter = THREE.LinearFilter
  matcap.needsUpdate = true

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.025, 0.03)
  scene.add(new THREE.Mesh(
    new THREE.SphereGeometry(0.72, 24, 16),
    new THREE.MeshMatcapMaterial({
      color: 0xffffff,
      matcap,
    }),
  ))

  return {
    name: 'mesh-matcap-material-map',
    scene,
    camera: makeCamera([0.8, 0.35, 3.0], [0, 0, 0]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [39, 44, 48],
    minNonBackgroundRatio: 0.02,
    validate(rgba, { width }) {
      const center = meanRegion(rgba, width, 32, 32, 64, 64)
      if (!(center.b > center.r + 20 && center.g > center.r + 10)) {
        throw new Error(`matcap corpus should sample the blue-green matcap blend, got ${JSON.stringify(center)}`)
      }
    },
  }
}

export function meshMatcapMaterialFlatShadingCorpus() {
  function makeGeometry() {
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
      -1, -1, 0,
      1, -1, 0,
      -1, 1, 0,
      1, 1, 1,
    ]), 3))
    geometry.setIndex([0, 1, 2, 1, 3, 2])
    return geometry
  }

  function makeMatcap() {
    const data = []
    for (let y = 0; y < 4; y += 1) {
      for (let x = 0; x < 4; x += 1) {
        data.push(x * 85, y * 85, 255 - x * 85, 255)
      }
    }
    const texture = new THREE.DataTexture(new Uint8Array(data), 4, 4, THREE.RGBAFormat)
    texture.needsUpdate = true
    return texture
  }

  function makeScene(flatShading) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      makeGeometry(),
      new THREE.MeshMatcapMaterial({
        color: 0xffffff,
        matcap: makeMatcap(),
        flatShading,
        side: THREE.DoubleSide,
      }),
    ))
    return scene
  }

  const smoothScene = makeScene(false)
  const flatScene = makeScene(true)
  const camera = makeCamera([0, 0, 4], [0, 0, 0.2])
  const options = { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' }
  let shadingDiff = 0

  return {
    name: 'mesh-matcap-material-flat-shading',
    scene: flatScene,
    camera,
    options,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.08,
    render(renderer) {
      const smooth = renderer.render(smoothScene, camera, options).slice()
      const flat = renderer.render(flatScene, camera, options)
      shadingDiff = meanAbsDiff(smooth, flat)
      return flat
    },
    validate() {
      if (!(shadingDiff > 1)) {
        throw new Error(`matcap flat-shading corpus should change face-normal lookup, diff=${shadingDiff.toFixed(3)}`)
      }
    },
  }
}

export function meshMatcapMaterialNormalMapCorpus() {
  function makeMatcap() {
    const texture = new THREE.DataTexture(new Uint8Array([
      255, 0, 0, 255,
      0, 255, 0, 255,
    ]), 2, 1, THREE.RGBAFormat)
    texture.needsUpdate = true
    return texture
  }

  function makeScene(normalMap) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshMatcapMaterial({
        color: 0xffffff,
        matcap: makeMatcap(),
        normalMap,
      }),
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
    name: 'mesh-matcap-material-normal-map',
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
      if (!(flatCenter.r > flatCenter.g + 40 && mappedCenter.g > mappedCenter.r + 40)) {
        throw new Error(`matcap normal-map corpus should shift lookup from red to green, flat=${JSON.stringify(flatCenter)} mapped=${JSON.stringify(mappedCenter)}`)
      }
    },
  }
}

export function meshMatcapMaterialObjectSpaceNormalMapCorpus() {
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -1, -1, 0,
    1, -1, 0,
    -1, 1, 0,
    1, -1, 0,
    1, 1, 0,
    -1, 1, 0,
  ]), 3))
  geometry.setAttribute('normal', new THREE.BufferAttribute(new Float32Array([
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
  ]), 3))
  geometry.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([
    0, 0,
    0, 1,
    1, 0,
    0, 1,
    1, 1,
    1, 0,
  ]), 2))

  function makeMatcap() {
    const texture = new THREE.DataTexture(new Uint8Array([
      255, 0, 0, 255,
      0, 255, 0, 255,
      0, 0, 255, 255,
      255, 255, 0, 255,
    ]), 2, 2, THREE.RGBAFormat)
    texture.magFilter = THREE.LinearFilter
    texture.minFilter = THREE.LinearFilter
    texture.needsUpdate = true
    return texture
  }

  function makeScene(normalMapType) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      geometry,
      new THREE.MeshMatcapMaterial({
        color: 0xffffff,
        matcap: makeMatcap(),
        normalMap: solidTexture(255, 128, 128),
        normalMapType,
      }),
    ))
    return scene
  }

  const tangentScene = makeScene(THREE.TangentSpaceNormalMap)
  const objectScene = makeScene(THREE.ObjectSpaceNormalMap)
  const camera = makeCamera([0, 0, 3])
  const options = { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' }
  let normalTypeDiff = 0

  return {
    name: 'mesh-matcap-material-object-space-normal-map',
    scene: objectScene,
    camera,
    options,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.35,
    render(renderer) {
      const tangent = renderer.render(tangentScene, camera, options).slice()
      const objectSpace = renderer.render(objectScene, camera, options)
      normalTypeDiff = meanAbsDiff(tangent, objectSpace)
      return objectSpace
    },
    validate() {
      if (!(normalTypeDiff > 20)) {
        throw new Error(`matcap object-space normal-map corpus should change lookup from tangent-space output, diff=${normalTypeDiff.toFixed(3)}`)
      }
    },
  }
}

export function meshMatcapMaterialBumpMapCorpus() {
  function makeMatcap() {
    const texture = new THREE.DataTexture(new Uint8Array([
      255, 0, 0, 255,
      0, 255, 0, 255,
    ]), 2, 1, THREE.RGBAFormat)
    texture.needsUpdate = true
    return texture
  }

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

  function makeScene(bumpScale) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshMatcapMaterial({
        color: 0xffffff,
        matcap: makeMatcap(),
        bumpMap: makeBumpMap(),
        bumpScale,
      }),
    ))
    return scene
  }

  const flatScene = makeScene(0)
  const bumpedScene = makeScene(8)
  const camera = makeCamera([0, 0, 3])
  const options = { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' }
  let bumpDiff = 0

  return {
    name: 'mesh-matcap-material-bump-map',
    scene: bumpedScene,
    camera,
    options,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.35,
    render(renderer) {
      const flat = renderer.render(flatScene, camera, options).slice()
      const bumped = renderer.render(bumpedScene, camera, options)
      bumpDiff = meanAbsDiff(flat, bumped)
      return bumped
    },
    validate() {
      if (!(bumpDiff > 2)) {
        throw new Error(`matcap bump-map corpus should perturb the matcap lookup, diff=${bumpDiff.toFixed(3)}`)
      }
    },
  }
}

export function meshToonMaterialCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.03, 0.025, 0.035)
  scene.add(new THREE.AmbientLight(0xffffff, 0.12))
  const key = new THREE.DirectionalLight(0xffffff, 1.8)
  key.position.set(2, 3, 3)
  scene.add(key)

  scene.add(new THREE.Mesh(
    new THREE.SphereGeometry(0.72, 24, 16),
    new THREE.MeshToonMaterial({
      color: 0x66ccff,
      gradientMap: gradientTexture(),
    }),
  ))

  return {
    name: 'mesh-toon-gradient-map',
    scene,
    camera: makeCamera([0.8, 0.35, 3.0], [0, 0, 0]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [48, 44, 53],
    minNonBackgroundRatio: 0.02,
    validate(rgba, { width }) {
      const center = meanRegion(rgba, width, 32, 32, 64, 64)
      if (!(center.b > center.g + 20 && center.g > center.r + 60)) {
        throw new Error(`toon gradient corpus should sample the blue-green ramp, got ${JSON.stringify(center)}`)
      }
    },
  }
}

export function meshToonMaterialFallbackBandsCorpus() {
  function makeScene(material) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(new THREE.SphereGeometry(1, 48, 24), material))

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(2, 0, 3)
    scene.add(light)
    return scene
  }

  const toonScene = makeScene(new THREE.MeshToonMaterial({ color: 0xffffff }))
  const lambertScene = makeScene(new THREE.MeshLambertMaterial({ color: 0xffffff }))
  const camera = makeCamera([0, 0, 3])
  const options = { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' }
  let toonMean = null
  let lambertMean = null

  return {
    name: 'mesh-toon-fallback-bands',
    scene: toonScene,
    camera,
    options,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.1,
    render(renderer) {
      const lambert = renderer.render(lambertScene, camera, options)
      lambertMean = meanRegion(lambert, options.width, 0, 0, options.width, options.height)
      const toon = renderer.render(toonScene, camera, options)
      toonMean = meanRegion(toon, options.width, 0, 0, options.width, options.height)
      return toon
    },
    validate() {
      if (!(toonMean.r > lambertMean.r + 8)) {
        throw new Error(`toon fallback corpus should produce broader lit bands than Lambert, toon=${JSON.stringify(toonMean)} lambert=${JSON.stringify(lambertMean)}`)
      }
    },
  }
}
