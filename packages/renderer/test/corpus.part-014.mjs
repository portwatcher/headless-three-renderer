import * as THREE from 'three'
import { CORPUS_RENDER_SIZE, meanAbsDiff, meanRegion, solidTexture } from './corpus.part-001.mjs'
export function multipleDirectionalShadowCorpus() {
  function makeScene(lightXs) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const caster = new THREE.Mesh(
      new THREE.BoxGeometry(1.5, 1.5, 1.5),
      new THREE.MeshBasicMaterial({
        color: 0xffffff,
        colorWrite: false,
        depthWrite: false,
      }),
    )
    caster.position.y = 0.75
    caster.castShadow = true
    scene.add(caster)

    for (const x of lightXs) {
      const light = new THREE.DirectionalLight(0xffffff, 2)
      light.position.set(x, 5, 0)
      light.target.position.set(0, 0, 0)
      light.castShadow = true
      light.shadow.mapSize.set(256, 256)
      light.shadow.camera.left = -6
      light.shadow.camera.right = 6
      light.shadow.camera.top = 6
      light.shadow.camera.bottom = -6
      light.shadow.camera.near = 0.1
      light.shadow.camera.far = 12
      scene.add(light)
      scene.add(light.target)
    }

    return scene
  }

  const firstScene = makeScene([5])
  const secondScene = makeScene([-5])
  const bothScene = makeScene([5, -5])
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 10, 0)
  camera.up.set(0, 0, -1)
  camera.lookAt(0, 0, 0)
  const options = { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' }
  const left = [21, 40, 36, 56]
  const right = [60, 40, 75, 56]
  const stats = {}

  function luminance(rgba, region) {
    const mean = meanRegion(rgba, options.width, ...region)
    return mean.r + mean.g + mean.b
  }

  return {
    name: 'multiple-directional-shadows',
    scene: bothScene,
    camera,
    options,
    background: [255, 255, 255],
    minNonBackgroundRatio: 0.02,
    render(renderer) {
      const first = renderer.render(firstScene, camera, options)
      const second = renderer.render(secondScene, camera, options)
      const both = renderer.render(bothScene, camera, options)
      stats.firstLeft = luminance(first, left)
      stats.firstRight = luminance(first, right)
      stats.secondLeft = luminance(second, left)
      stats.secondRight = luminance(second, right)
      stats.bothLeft = luminance(both, left)
      stats.bothRight = luminance(both, right)
      return both
    },
    validate() {
      if (!(stats.firstLeft < stats.firstRight - 25)) {
        throw new Error(`first directional light should cast the left shadow, stats=${JSON.stringify(stats)}`)
      }
      if (!(stats.secondRight < stats.secondLeft - 25)) {
        throw new Error(`second directional light should cast the right shadow, stats=${JSON.stringify(stats)}`)
      }
      if (!(stats.bothLeft < stats.secondLeft - 25 && stats.bothRight < stats.firstRight - 25)) {
        throw new Error(`dual directional shadow maps should preserve both shadow regions, stats=${JSON.stringify(stats)}`)
      }
    },
  }
}

export function mixedShadowLightTypesCorpus() {
  const lightTypes = ['directional', 'spot', 'point']
  const regions = {
    directional: [32, 40, 40, 56],
    spot: [56, 40, 72, 56],
    point: [40, 24, 56, 32],
  }
  const stats = {}

  function makeScene(activeLightTypes, castShadow) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const caster = new THREE.Mesh(
      new THREE.BoxGeometry(1.5, 1.5, 1.5),
      new THREE.MeshBasicMaterial({
        color: 0xffffff,
        colorWrite: false,
        depthWrite: false,
      }),
    )
    caster.position.y = 0.75
    caster.castShadow = castShadow
    scene.add(caster)

    if (activeLightTypes.includes('directional')) {
      const light = new THREE.DirectionalLight(0xffffff, 2)
      light.position.set(5, 6, 0)
      light.target.position.set(0, 0, 0)
      light.castShadow = true
      light.shadow.mapSize.set(256, 256)
      light.shadow.camera.left = -7
      light.shadow.camera.right = 7
      light.shadow.camera.top = 7
      light.shadow.camera.bottom = -7
      light.shadow.camera.near = 0.1
      light.shadow.camera.far = 16
      scene.add(light)
      scene.add(light.target)
    }

    if (activeLightTypes.includes('spot')) {
      const light = new THREE.SpotLight(0xffffff, 3.2, 16, Math.PI / 4, 0.1, 1)
      light.position.set(-5, 6, 0)
      light.target.position.set(0, 0, 0)
      light.castShadow = true
      light.shadow.mapSize.set(256, 256)
      light.shadow.camera.near = 0.1
      light.shadow.camera.far = 16
      scene.add(light)
      scene.add(light.target)
    }

    if (activeLightTypes.includes('point')) {
      const light = new THREE.PointLight(0xffffff, 2.5, 16)
      light.position.set(0, 5, 4)
      light.castShadow = true
      light.shadow.mapSize.set(256, 256)
      light.shadow.camera.near = 0.1
      light.shadow.camera.far = 16
      scene.add(light)
    }

    return scene
  }

  const mixedScene = makeScene(lightTypes, true)
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 10, 0)
  camera.up.set(0, 0, -1)
  camera.lookAt(0, 0, 0)
  const options = { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' }

  function luminance(rgba, region) {
    const mean = meanRegion(rgba, options.width, ...region)
    return mean.r + mean.g + mean.b
  }

  return {
    name: 'mixed-shadow-light-types',
    scene: mixedScene,
    camera,
    options,
    background: [255, 255, 255],
    minNonBackgroundRatio: 0.04,
    render(renderer) {
      const unshadowedMixed = renderer.render(makeScene(lightTypes, false), camera, options)
      const shadowedMixed = renderer.render(mixedScene, camera, options)

      for (const lightType of lightTypes) {
        const singleUnshadowed = renderer.render(makeScene([lightType], false), camera, options)
        const singleShadowed = renderer.render(makeScene([lightType], true), camera, options)
        stats[lightType] = {
          unshadowed: luminance(singleUnshadowed, regions[lightType]),
          shadowed: luminance(singleShadowed, regions[lightType]),
          mixedUnshadowed: luminance(unshadowedMixed, regions[lightType]),
          mixedShadowed: luminance(shadowedMixed, regions[lightType]),
        }
      }

      return shadowedMixed
    },
    validate() {
      for (const lightType of lightTypes) {
        const result = stats[lightType]
        if (!result) {
          throw new Error(`mixed shadow-light corpus did not record ${lightType} stats`)
        }
        if (!(result.shadowed < result.unshadowed - 50)) {
          throw new Error(`${lightType} light should cast an isolated shadow, stats=${JSON.stringify(stats)}`)
        }
        if (!(result.mixedShadowed < result.mixedUnshadowed - 50)) {
          throw new Error(`${lightType} shadow region should remain dark in the mixed shadow-light scene, stats=${JSON.stringify(stats)}`)
        }
      }
    },
  }
}

export function shadowMapEnabledGatingCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(1, 1, 1)

  const receiver = new THREE.Mesh(
    new THREE.PlaneGeometry(12, 12),
    new THREE.ShadowMaterial({ opacity: 1 }),
  )
  receiver.rotation.x = -Math.PI / 2
  receiver.receiveShadow = true
  scene.add(receiver)

  const caster = new THREE.Mesh(
    new THREE.BoxGeometry(1.5, 1.5, 1.5),
    new THREE.MeshBasicMaterial({
      color: 0xffffff,
      colorWrite: false,
      depthWrite: false,
    }),
  )
  caster.position.y = 0.75
  caster.castShadow = true
  scene.add(caster)

  const light = new THREE.DirectionalLight(0xffffff, 2)
  light.position.set(5, 6, 0)
  light.target.position.set(0, 0, 0)
  light.castShadow = true
  light.shadow.mapSize.set(256, 256)
  light.shadow.camera.left = -6
  light.shadow.camera.right = 6
  light.shadow.camera.top = 6
  light.shadow.camera.bottom = -6
  light.shadow.camera.near = 0.1
  light.shadow.camera.far = 14
  scene.add(light)
  scene.add(light.target)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 6, 8)
  camera.lookAt(0, 0, 0)
  const options = { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' }
  let disabledOutput

  function luminance(rgba) {
    const mean = meanRegion(rgba, options.width, 28, 42, 68, 82)
    return mean.r + mean.g + mean.b
  }

  return {
    name: 'shadow-map-enabled-gating',
    scene,
    camera,
    options,
    background: [255, 255, 255],
    minNonBackgroundRatio: 0.02,
    render(renderer) {
      const previousEnabled = renderer.shadowMap.enabled
      try {
        renderer.shadowMap.enabled = true
        const enabled = renderer.render(scene, camera, options)
        renderer.shadowMap.enabled = false
        disabledOutput = renderer.render(scene, camera, options)
        return enabled
      } finally {
        renderer.shadowMap.enabled = previousEnabled
      }
    },
    validate(rgba) {
      const enabledLum = luminance(rgba)
      const disabledLum = luminance(disabledOutput)
      if (!(enabledLum < disabledLum - 25)) {
        throw new Error(`shadowMap.enabled corpus should darken the receiver only when enabled, enabled=${enabledLum.toFixed(1)} disabled=${disabledLum.toFixed(1)}`)
      }
    },
  }
}

export function shadowMapTypeFilteringCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(1, 1, 1)

  const receiver = new THREE.Mesh(
    new THREE.PlaneGeometry(12, 12),
    new THREE.ShadowMaterial({ opacity: 1 }),
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
  light.shadow.mapSize.set(128, 128)
  light.shadow.camera.left = -7
  light.shadow.camera.right = 7
  light.shadow.camera.top = 7
  light.shadow.camera.bottom = -7
  light.shadow.camera.near = 0.1
  light.shadow.camera.far = 16
  scene.add(light)
  scene.add(light.target)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 6, 8)
  camera.lookAt(0, 0, 0)
  const options = { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' }
  const stats = {}

  function luminance(rgba) {
    const mean = meanRegion(rgba, options.width, 28, 42, 68, 82)
    return mean.r + mean.g + mean.b
  }

  return {
    name: 'shadow-map-type-filtering',
    scene,
    camera,
    options,
    background: [255, 255, 255],
    minNonBackgroundRatio: 0.02,
    render(renderer) {
      const previousType = renderer.shadowMap.type
      const previousRadius = light.shadow.radius
      try {
        renderer.shadowMap.type = THREE.BasicShadowMap
        light.shadow.radius = 0
        stats.basicSmallRadius = luminance(renderer.render(scene, camera, options))
        light.shadow.radius = 4
        const basicLarge = renderer.render(scene, camera, options)
        stats.basicLargeRadius = luminance(basicLarge)
        renderer.shadowMap.type = THREE.PCFShadowMap
        stats.pcfLargeRadius = luminance(renderer.render(scene, camera, options))
        return basicLarge
      } finally {
        renderer.shadowMap.type = previousType
        light.shadow.radius = previousRadius
      }
    },
    validate() {
      if (!(Math.abs(stats.basicSmallRadius - stats.basicLargeRadius) < 1)) {
        throw new Error(`BasicShadowMap should ignore PCF radius in corpus, stats=${JSON.stringify(stats)}`)
      }
      if (!(stats.pcfLargeRadius < stats.basicLargeRadius - 10)) {
        throw new Error(`PCFShadowMap should use radius-based PCF sampling in corpus, stats=${JSON.stringify(stats)}`)
      }
    },
  }
}

export function customShadowDisplacementCorpus() {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 6, 8)
  camera.lookAt(0, 0, 0)
  const options = { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' }
  const stats = new Map()

  function addReceiver(scene) {
    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)
  }

  function displacementMaterial(displacementScale) {
    return new THREE.MeshStandardMaterial({
      color: 0xffffff,
      displacementMap: solidTexture(255, 0, 0),
      displacementScale,
      displacementBias: 0,
      colorWrite: false,
      depthWrite: false,
    })
  }

  function makeCaster(kind, displacementScale, inheritFromSource) {
    const caster = new THREE.Mesh(
      new THREE.PlaneGeometry(2.5, 2.5, 8, 8),
      inheritFromSource
        ? displacementMaterial(displacementScale)
        : new THREE.MeshBasicMaterial({
          color: 0xffffff,
          colorWrite: false,
          depthWrite: false,
        }),
    )
    caster.position.set(0, 1.7, 0)
    caster.rotation.x = -Math.PI / 2
    caster.castShadow = true

    if (kind === 'depth') {
      caster.customDepthMaterial = inheritFromSource
        ? new THREE.MeshDepthMaterial()
        : new THREE.MeshDepthMaterial({
          displacementMap: solidTexture(255, 0, 0),
          displacementScale,
          displacementBias: 0,
        })
    } else {
      caster.customDistanceMaterial = inheritFromSource
        ? new THREE.MeshDistanceMaterial()
        : new THREE.MeshDistanceMaterial({
          displacementMap: solidTexture(255, 0, 0),
          displacementScale,
          displacementBias: 0,
        })
    }

    return caster
  }

  function addShadowLight(scene, kind) {
    if (kind === 'depth') {
      const light = new THREE.DirectionalLight(0xffffff, 2)
      light.castShadow = true
      light.position.set(8, 6, 0)
      light.target.position.set(0, 0, 0)
      light.shadow.camera.left = -7
      light.shadow.camera.right = 7
      light.shadow.camera.top = 7
      light.shadow.camera.bottom = -7
      light.shadow.camera.near = 0.1
      light.shadow.camera.far = 16
      scene.add(light)
      scene.add(light.target)
      return
    }

    const light = new THREE.PointLight(0xffffff, 2)
    light.position.set(0, 5, 4)
    light.distance = 12
    light.castShadow = true
    light.shadow.mapSize.set(256, 256)
    light.shadow.camera.near = 0.1
    light.shadow.camera.far = 12
    scene.add(light)
  }

  function makeScene(kind, displacementScale, inheritFromSource) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)
    addReceiver(scene)
    scene.add(makeCaster(kind, displacementScale, inheritFromSource))
    addShadowLight(scene, kind)
    return scene
  }

  return {
    name: 'custom-shadow-displacement-maps',
    scene: makeScene('depth', 2, false),
    camera,
    options,
    background: [255, 255, 255],
    minNonBackgroundRatio: 0.008,
    browserReference: false,
    render(renderer) {
      for (const [kind, inheritFromSource] of [
        ['depth', false],
        ['depth', true],
        ['distance', false],
        ['distance', true],
      ]) {
        const flat = renderer.render(makeScene(kind, 0, inheritFromSource), camera, options)
        const displaced = renderer.render(makeScene(kind, 2, inheritFromSource), camera, options)
        stats.set(`${kind}:${inheritFromSource ? 'source' : 'custom'}`, meanAbsDiff(flat, displaced))
      }
      return renderer.render(makeScene('depth', 2, false), camera, options)
    },
    validate() {
      for (const [label, diff] of stats) {
        if (!(diff > 5)) {
          throw new Error(`custom shadow displacement corpus expected ${label} to move the shadow, diff=${diff.toFixed(3)}`)
        }
      }
    },
  }
}
