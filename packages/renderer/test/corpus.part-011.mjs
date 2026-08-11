import * as THREE from 'three'
import { CORPUS_RENDER_SIZE, addBasicLights, coloredCubeBackgroundTexture, makeCamera, meanRegion, packedCubeUvColoredBackgroundTexture, pixelAt } from './corpus.part-001.mjs'
export function equirectangularBackgroundCorpus() {
  const background = new THREE.DataTexture(new Uint8Array([
    255, 48, 32, 255,
    255, 48, 32, 255,
    255, 48, 32, 255,
    255, 48, 32, 255,
    32, 200, 96, 255,
    32, 200, 96, 255,
    32, 200, 96, 255,
    32, 200, 96, 255,
  ]), 8, 1, THREE.RGBAFormat)
  background.mapping = THREE.EquirectangularReflectionMapping
  background.magFilter = THREE.NearestFilter
  background.minFilter = THREE.NearestFilter
  background.needsUpdate = true

  const scene = new THREE.Scene()
  scene.background = background
  scene.backgroundIntensity = 0.85
  scene.backgroundRotation = new THREE.Euler(0, Math.PI, 0)

  return {
    name: 'equirectangular-background-rotation',
    scene,
    camera: makeCamera([0, 0, 0], [0, 0, -1]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.95,
    validate(rgba, { width }) {
      const center = meanRegion(rgba, width, 32, 32, 64, 64)
      if (!(center.g > center.r + 100 && center.g > center.b + 40)) {
        throw new Error(`equirectangular background rotation corpus should sample the green half, got ${JSON.stringify(center)}`)
      }
    },
  }
}

export function cubeBackgroundTextureCorpus() {
  const scene = new THREE.Scene()
  scene.background = coloredCubeBackgroundTexture()
  scene.background.magFilter = THREE.NearestFilter
  scene.background.minFilter = THREE.NearestFilter
  scene.backgroundRotation = new THREE.Euler(0, Math.PI, 0)

  return {
    name: 'cube-background-texture-rotation',
    scene,
    camera: makeCamera([0, 0, 0], [0, 0, -1]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.95,
    validate(rgba, { width }) {
      const center = meanRegion(rgba, width, 32, 32, 64, 64)
      if (!(center.g > center.r + 100 && center.g > center.b + 40)) {
        throw new Error(`cube background rotation corpus should sample the green face, got ${JSON.stringify(center)}`)
      }
    },
  }
}

export function cubeBackgroundOptionRotationCorpus() {
  const scene = new THREE.Scene()
  scene.background = coloredCubeBackgroundTexture()
  scene.background.magFilter = THREE.NearestFilter
  scene.background.minFilter = THREE.NearestFilter
  scene.backgroundRotation = new THREE.Euler(0, 0, 0)

  return {
    name: 'cube-background-option-rotation',
    scene,
    camera: makeCamera([0, 0, 0], [0, 0, -1]),
    options: {
      width: CORPUS_RENDER_SIZE,
      height: CORPUS_RENDER_SIZE,
      format: 'rgba',
      backgroundRotation: new THREE.Euler(0, Math.PI, 0),
    },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.95,
    validate(rgba, { width }) {
      const center = meanRegion(rgba, width, 32, 32, 64, 64)
      if (!(center.g > center.r + 100 && center.g > center.b + 40)) {
        throw new Error(`cube background option rotation corpus should sample the green face, got ${JSON.stringify(center)}`)
      }
    },
  }
}

export function cubeUvBackgroundTextureCorpus() {
  const scene = new THREE.Scene()
  scene.background = coloredCubeBackgroundTexture()
  scene.background.mapping = THREE.CubeUVReflectionMapping
  scene.background.magFilter = THREE.NearestFilter
  scene.background.minFilter = THREE.NearestFilter
  scene.backgroundRotation = new THREE.Euler(0, Math.PI, 0)

  return {
    name: 'cubeuv-cube-background-texture',
    scene,
    camera: makeCamera([0, 0, 0], [0, 0, -1]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.95,
    browserReference: false,
    validate(rgba, { width }) {
      const center = meanRegion(rgba, width, 32, 32, 64, 64)
      if (!(center.g > center.r + 100 && center.g > center.b + 40)) {
        throw new Error(`CubeUV cube background corpus should sample the green face, got ${JSON.stringify(center)}`)
      }
    },
  }
}

export function packedCubeUvBackgroundTextureCorpus() {
  const scene = new THREE.Scene()
  scene.background = packedCubeUvColoredBackgroundTexture()
  scene.background.magFilter = THREE.NearestFilter
  scene.background.minFilter = THREE.NearestFilter
  scene.backgroundRotation = new THREE.Euler(0, Math.PI, 0)

  return {
    name: 'packed-cubeuv-background-texture',
    scene,
    camera: makeCamera([0, 0, 0], [0, 0, -1]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.95,
    validate(rgba, { width }) {
      const center = meanRegion(rgba, width, 32, 32, 64, 64)
      if (!(center.g > center.r + 100 && center.g > center.b + 40)) {
        throw new Error(`packed CubeUV background corpus should sample the green face, got ${JSON.stringify(center)}`)
      }
    },
  }
}

export function customSortGroupCorpus() {
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.9, -0.9, 0,
    0.9, -0.9, 0,
    0.9, 0.9, 0,
    -0.9, -0.9, 0,
    0.9, 0.9, 0,
    -0.9, 0.9, 0,
    -0.9, -0.9, 0,
    0.9, -0.9, 0,
    0.9, 0.9, 0,
    -0.9, -0.9, 0,
    0.9, 0.9, 0,
    -0.9, 0.9, 0,
  ]), 3))
  geometry.addGroup(0, 6, 0)
  geometry.addGroup(6, 6, 1)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(geometry, [
    new THREE.MeshBasicMaterial({ color: 0xff3344, depthTest: false }),
    new THREE.MeshBasicMaterial({ color: 0x2266ff, depthTest: false }),
  ]))

  return {
    name: 'custom-opaque-sort-group-items',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: {
      width: CORPUS_RENDER_SIZE,
      height: CORPUS_RENDER_SIZE,
      format: 'rgba',
      opaqueSort: (a, b) => b.group.materialIndex - a.group.materialIndex,
    },
    background: [0, 0, 0],
    validate(rgba, { width }) {
      const center = pixelAt(rgba, width, 48, 48)
      const corner = pixelAt(rgba, width, 4, 4)
      if (!(center.r > center.b + 160 && center.r > center.g + 170 && corner.r === 0 && corner.g === 0 && corner.b === 0)) {
        throw new Error(`custom sort corpus should draw the red group last on black background, got center=${JSON.stringify(center)} corner=${JSON.stringify(corner)}`)
      }
    },
  }
}

export function customTransparentSortGroupCorpus() {
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.9, -0.9, 0,
    0.9, -0.9, 0,
    0.9, 0.9, 0,
    -0.9, -0.9, 0,
    0.9, 0.9, 0,
    -0.9, 0.9, 0,
    -0.9, -0.9, 0,
    0.9, -0.9, 0,
    0.9, 0.9, 0,
    -0.9, -0.9, 0,
    0.9, 0.9, 0,
    -0.9, 0.9, 0,
  ]), 3))
  geometry.addGroup(0, 6, 0)
  geometry.addGroup(6, 6, 1)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(geometry, [
    new THREE.MeshBasicMaterial({ color: 0xff3344, opacity: 0.55, transparent: true, depthWrite: false }),
    new THREE.MeshBasicMaterial({ color: 0x2266ff, opacity: 0.55, transparent: true, depthWrite: false }),
  ]))

  return {
    name: 'custom-transparent-sort-group-items',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: {
      width: CORPUS_RENDER_SIZE,
      height: CORPUS_RENDER_SIZE,
      format: 'rgba',
      transparentSort: (a, b) => b.group.materialIndex - a.group.materialIndex,
    },
    background: [0, 0, 0],
    validate(rgba, { width }) {
      const center = pixelAt(rgba, width, 48, 48)
      const corner = pixelAt(rgba, width, 4, 4)
      if (!(center.r > center.b + 10 && center.r > center.g + 60 && corner.r === 0 && corner.g === 0 && corner.b === 0)) {
        throw new Error(`custom transparent sort corpus should draw the red group last on black background, got center=${JSON.stringify(center)} corner=${JSON.stringify(corner)}`)
      }
    },
  }
}

export function rendererBucketFlagsCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const opaque = new THREE.Mesh(
    new THREE.PlaneGeometry(0.8, 0.8),
    new THREE.MeshBasicMaterial({ color: 0x00ff00, depthTest: false, toneMapped: false }),
  )
  opaque.position.x = -1.1
  scene.add(opaque)

  const transmissive = new THREE.Mesh(
    new THREE.PlaneGeometry(0.8, 0.8),
    new THREE.MeshPhysicalMaterial({
      color: 0x000000,
      emissive: 0x0000ff,
      emissiveIntensity: 1,
      transmission: 0.5,
      depthTest: false,
      depthWrite: false,
      toneMapped: false,
    }),
  )
  scene.add(transmissive)

  const transparent = new THREE.Mesh(
    new THREE.PlaneGeometry(0.8, 0.8),
    new THREE.MeshBasicMaterial({
      color: 0xff0000,
      depthTest: false,
      depthWrite: false,
      toneMapped: false,
      transparent: true,
    }),
  )
  transparent.position.x = 1.1
  scene.add(transparent)

  const camera = new THREE.OrthographicCamera(-2, 2, 2, -2, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const options = { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' }
  let opaqueOnly
  let transparentOnly
  let backgroundOnly

  function bucketMeans(rgba) {
    return {
      opaque: meanRegion(rgba, options.width, 17, 40, 31, 56),
      transmissive: meanRegion(rgba, options.width, 41, 40, 55, 56),
      transparent: meanRegion(rgba, options.width, 65, 40, 79, 56),
    }
  }

  function assertDark(mean, label) {
    if (!(mean.r < 5 && mean.g < 5 && mean.b < 5)) {
      throw new Error(`${label} should stay background black, got rgb(${mean.r}, ${mean.g}, ${mean.b})`)
    }
  }

  return {
    name: 'renderer-bucket-flag-options',
    scene,
    camera,
    options,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.04,
    browserReference: false,
    render(renderer) {
      const all = renderer.render(scene, camera, options)
      opaqueOnly = renderer.render(scene, camera, { ...options, opaque: true, transparent: false })
      transparentOnly = renderer.render(scene, camera, { ...options, opaque: false, transparent: true })
      backgroundOnly = renderer.render(scene, camera, { ...options, opaque: false, transparent: false })
      return all
    },
    validate(rgba) {
      const all = bucketMeans(rgba)
      if (!(all.opaque.g > all.opaque.r + 140 && all.opaque.g > all.opaque.b + 140)) {
        throw new Error(`default bucket flags should render opaque green, got ${JSON.stringify(all.opaque)}`)
      }
      if (!(all.transmissive.b > all.transmissive.r + 70 && all.transmissive.b > all.transmissive.g + 70)) {
        throw new Error(`default bucket flags should render transmissive blue, got ${JSON.stringify(all.transmissive)}`)
      }
      if (!(all.transparent.r > all.transparent.g + 140 && all.transparent.r > all.transparent.b + 140)) {
        throw new Error(`default bucket flags should render transparent red, got ${JSON.stringify(all.transparent)}`)
      }

      const opaqueOnlyMeans = bucketMeans(opaqueOnly)
      if (!(opaqueOnlyMeans.opaque.g > opaqueOnlyMeans.opaque.r + 140)) {
        throw new Error(`opaque-only bucket options should keep green, got ${JSON.stringify(opaqueOnlyMeans.opaque)}`)
      }
      assertDark(opaqueOnlyMeans.transmissive, 'opaque-only transmissive bucket')
      assertDark(opaqueOnlyMeans.transparent, 'opaque-only transparent bucket')

      const transparentOnlyMeans = bucketMeans(transparentOnly)
      assertDark(transparentOnlyMeans.opaque, 'transparent-only opaque bucket')
      if (!(transparentOnlyMeans.transmissive.b > transparentOnlyMeans.transmissive.r + 70)) {
        throw new Error(`transparent-only bucket options should keep transmissive blue, got ${JSON.stringify(transparentOnlyMeans.transmissive)}`)
      }
      if (!(transparentOnlyMeans.transparent.r > transparentOnlyMeans.transparent.g + 140)) {
        throw new Error(`transparent-only bucket options should keep transparent red, got ${JSON.stringify(transparentOnlyMeans.transparent)}`)
      }

      for (const [label, mean] of Object.entries(bucketMeans(backgroundOnly))) {
        assertDark(mean, `background-only ${label} bucket`)
      }
    },
  }
}

export function skinnedMorphCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.05, 0.06, 0.08)
  addBasicLights(scene)

  const geometry = new THREE.PlaneGeometry(1, 1, 1, 1)
  const count = geometry.getAttribute('position').count
  geometry.setAttribute('skinIndex', new THREE.BufferAttribute(new Uint16Array(count * 4), 4))
  const skinWeights = new Float32Array(count * 4)
  for (let i = 0; i < count; i += 1) {
    skinWeights[i * 4] = 1
  }
  geometry.setAttribute('skinWeight', new THREE.BufferAttribute(skinWeights, 4))
  geometry.morphTargetsRelative = true
  geometry.morphAttributes.position = [
    new THREE.BufferAttribute(new Float32Array([
      0, 0, 0,
      0.15, 0, 0,
      0, 0.2, 0,
      0.15, 0.2, 0,
    ]), 3),
  ]

  const material = new THREE.MeshStandardMaterial({ color: 0x77ccff, roughness: 0.55, metalness: 0.05 })
  const mesh = new THREE.SkinnedMesh(geometry, material)
  const bone = new THREE.Bone()
  mesh.add(bone)
  const skeleton = new THREE.Skeleton([bone])
  mesh.bind(skeleton)
  mesh.morphTargetInfluences = [0.6]
  bone.position.set(0.12, 0.05, 0)
  bone.updateMatrixWorld(true)
  mesh.rotation.y = -0.25
  scene.add(mesh)

  return {
    name: 'skinned-morphed-plane',
    scene,
    camera: makeCamera([0.2, 0.1, 2.5]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [63, 69, 80],
    validate(rgba, { width }) {
      const center = pixelAt(rgba, width, 48, 48)
      const corner = pixelAt(rgba, width, 4, 4)
      if (!(center.b > center.g + 25 && center.g > center.r + 50 && center.b > center.r + 80 && corner.r === 63 && corner.g === 69 && corner.b === 80)) {
        throw new Error(`skinned morph corpus should render the deformed cyan plane over background, got center=${JSON.stringify(center)} corner=${JSON.stringify(corner)}`)
      }
    },
  }
}
