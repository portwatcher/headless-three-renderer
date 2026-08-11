import * as THREE from 'three'
import { CORPUS_RENDER_SIZE, makeCamera, meanRegion, solidTexture } from './corpus.part-001.mjs'
export function spriteShadowCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(1, 1, 1)

  const receiver = new THREE.Mesh(
    new THREE.PlaneGeometry(12, 12),
    new THREE.ShadowMaterial({ opacity: 1 }),
  )
  receiver.rotation.x = -Math.PI / 2
  receiver.receiveShadow = true
  scene.add(receiver)

  const caster = new THREE.Sprite(new THREE.SpriteMaterial({ color: 0xffffff }))
  caster.position.set(-1.6, 4, 0)
  caster.scale.set(4, 4, 1)
  caster.castShadow = true
  scene.add(caster)

  const light = new THREE.DirectionalLight(0xffffff, 2)
  light.position.set(0, 6, 8)
  light.target.position.set(0, 0, 0)
  light.castShadow = true
  light.shadow.mapSize.set(256, 256)
  light.shadow.camera.left = -7
  light.shadow.camera.right = 7
  light.shadow.camera.top = 7
  light.shadow.camera.bottom = -7
  light.shadow.camera.near = 0.1
  light.shadow.camera.far = 16
  scene.add(light, light.target)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 6, 8)
  camera.lookAt(0, 0, 0)

  return {
    name: 'sprite-shadow-caster',
    scene,
    camera,
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [255, 255, 255],
    minNonBackgroundRatio: 0.05,
    validate(rgba, { width }) {
      const shadowed = meanRegion(rgba, width, 18, 22, 42, 46)
      const lit = meanRegion(rgba, width, 70, 22, 90, 46)
      const shadowedLum = shadowed.r + shadowed.g + shadowed.b
      const litLum = lit.r + lit.g + lit.b
      if (!(shadowedLum < litLum - 120)) {
        throw new Error(`sprite shadow corpus should darken the receiver (${shadowedLum} vs ${litLum})`)
      }
    },
  }
}

export function billboardPointLightShadowCorpus() {
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

  function addBillboard(scene, kind, castShadow) {
    if (kind === 'sprite') {
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ color: 0xffffff }))
      sprite.position.set(0, 2.2, 1.8)
      sprite.scale.set(4, 4, 1)
      sprite.castShadow = castShadow
      scene.add(sprite)
      return
    }

    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 2.2, 1.8]), 3))
    const points = new THREE.Points(geometry, new THREE.PointsMaterial({
      color: 0xffffff,
      size: 48,
      sizeAttenuation: false,
    }))
    points.castShadow = castShadow
    scene.add(points)
  }

  function addPointLight(scene) {
    const light = new THREE.PointLight(0xffffff, 2)
    light.position.set(0, 5, 4)
    light.distance = 12
    light.castShadow = true
    light.shadow.mapSize.set(256, 256)
    light.shadow.camera.near = 0.1
    light.shadow.camera.far = 12
    scene.add(light)
  }

  function makeScene(kind, castShadow) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)
    addReceiver(scene)
    addBillboard(scene, kind, castShadow)
    addPointLight(scene)
    return scene
  }

  function shadowLuminance(rgba) {
    const mean = meanRegion(rgba, options.width, 28, 42, 68, 82)
    return mean.r + mean.g + mean.b
  }

  return {
    name: 'billboard-point-light-shadow-casters',
    scene: makeScene('points', true),
    camera,
    options,
    background: [255, 255, 255],
    minNonBackgroundRatio: 0.02,
    browserReference: false,
    render(renderer) {
      let output = null
      for (const kind of ['sprite', 'points']) {
        const unshadowed = renderer.render(makeScene(kind, false), camera, options)
        const shadowed = renderer.render(makeScene(kind, true), camera, options)
        stats.set(kind, {
          shadowedLum: shadowLuminance(shadowed),
          unshadowedLum: shadowLuminance(unshadowed),
        })
        if (kind === 'points') {
          output = shadowed
        }
      }
      return output
    },
    validate() {
      for (const kind of ['sprite', 'points']) {
        const result = stats.get(kind)
        if (!result) {
          throw new Error(`billboard point-light shadow corpus did not record ${kind} stats`)
        }
        if (!(result.shadowedLum < result.unshadowedLum - 10)) {
          throw new Error(`${kind} point-light billboard shadow should darken the receiver, stats=${JSON.stringify(Object.fromEntries(stats))}`)
        }
      }
    },
  }
}

export function billboardCustomShadowCutoutCorpus() {
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

  function addBillboard(scene, kind, shadowKind, alphaMapGreen) {
    const sourceMaterial = kind === 'sprite'
      ? new THREE.SpriteMaterial({ color: 0xffffff })
      : new THREE.PointsMaterial({
        color: 0xffffff,
        size: 48,
        sizeAttenuation: false,
      })

    const shadowMaterial = shadowKind === 'point'
      ? new THREE.MeshDistanceMaterial({
        alphaMap: solidTexture(255, alphaMapGreen, 255),
        alphaTest: 0.5,
      })
      : new THREE.MeshDepthMaterial({
        alphaMap: solidTexture(255, alphaMapGreen, 255),
        alphaTest: 0.5,
      })

    if (kind === 'sprite') {
      const sprite = new THREE.Sprite(sourceMaterial)
      sprite.position.set(0, shadowKind === 'point' ? 2.2 : 4, shadowKind === 'point' ? 1.8 : 0)
      sprite.scale.set(4, 4, 1)
      sprite.castShadow = true
      if (shadowKind === 'point') {
        sprite.customDistanceMaterial = shadowMaterial
      } else {
        sprite.customDepthMaterial = shadowMaterial
      }
      scene.add(sprite)
      return
    }

    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array([0, shadowKind === 'point' ? 2.2 : 4, shadowKind === 'point' ? 1.8 : 0]), 3),
    )
    const points = new THREE.Points(geometry, sourceMaterial)
    points.castShadow = true
    if (shadowKind === 'point') {
      points.customDistanceMaterial = shadowMaterial
    } else {
      points.customDepthMaterial = shadowMaterial
    }
    scene.add(points)
  }

  function addLight(scene, shadowKind) {
    if (shadowKind === 'point') {
      const light = new THREE.PointLight(0xffffff, 2)
      light.position.set(0, 5, 4)
      light.distance = 12
      light.castShadow = true
      light.shadow.mapSize.set(256, 256)
      light.shadow.camera.near = 0.1
      light.shadow.camera.far = 12
      scene.add(light)
      return
    }

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(0, 6, 8)
    light.target.position.set(0, 0, 0)
    light.castShadow = true
    light.shadow.mapSize.set(256, 256)
    light.shadow.camera.left = -7
    light.shadow.camera.right = 7
    light.shadow.camera.top = 7
    light.shadow.camera.bottom = -7
    light.shadow.camera.near = 0.1
    light.shadow.camera.far = 16
    scene.add(light, light.target)
  }

  function makeScene(kind, shadowKind, alphaMapGreen) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)
    addReceiver(scene)
    addBillboard(scene, kind, shadowKind, alphaMapGreen)
    addLight(scene, shadowKind)
    return scene
  }

  function sampledMean(rgba, shadowKind) {
    if (shadowKind === 'point') {
      return meanRegion(rgba, options.width, 28, 42, 68, 82)
    }
    return meanRegion(rgba, options.width, 0, 0, options.width, options.height)
  }

  function luminance(mean) {
    return mean.r + mean.g + mean.b
  }

  return {
    name: 'billboard-custom-shadow-alpha-map-cutouts',
    scene: makeScene('points', 'point', 255),
    camera,
    options,
    background: [255, 255, 255],
    minNonBackgroundRatio: 0.02,
    browserReference: false,
    render(renderer) {
      let output = null
      const cases = [
        ['sprite', 'directional'],
        ['sprite', 'point'],
        ['points', 'directional'],
        ['points', 'point'],
      ]
      for (const [kind, shadowKind] of cases) {
        const opaque = renderer.render(makeScene(kind, shadowKind, 255), camera, options)
        const cutout = renderer.render(makeScene(kind, shadowKind, 0), camera, options)
        stats.set(`${kind}-${shadowKind}`, {
          opaque: sampledMean(opaque, shadowKind),
          cutout: sampledMean(cutout, shadowKind),
        })
        if (kind === 'points' && shadowKind === 'point') {
          output = opaque
        }
      }
      return output
    },
    validate() {
      const cases = [
        ['sprite', 'directional'],
        ['sprite', 'point'],
        ['points', 'directional'],
        ['points', 'point'],
      ]
      for (const [kind, shadowKind] of cases) {
        const result = stats.get(`${kind}-${shadowKind}`)
        if (!result) {
          throw new Error(`billboard custom shadow corpus did not record ${kind} ${shadowKind} stats`)
        }
        const opaqueLum = luminance(result.opaque)
        const cutoutLum = luminance(result.cutout)
        if (!(cutoutLum > opaqueLum + 10)) {
          throw new Error(`${kind} ${shadowKind} custom shadow alpha-map cutout should remove the caster shadow, got opaque=${opaqueLum} cutout=${cutoutLum}`)
        }
      }
    },
  }
}

export function pointSpotLightCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.025, 0.03)
  scene.add(new THREE.AmbientLight(0xffffff, 0.08))

  const material = new THREE.MeshLambertMaterial({ color: 0xffffff })
  const left = new THREE.Mesh(new THREE.SphereGeometry(0.42, 24, 16), material)
  left.position.x = -0.45
  scene.add(left)

  const right = new THREE.Mesh(new THREE.SphereGeometry(0.42, 24, 16), material.clone())
  right.position.x = 0.45
  scene.add(right)

  const point = new THREE.PointLight(0xff5533, 6, 4, 2)
  point.position.set(-1.2, 0.75, 1.5)
  scene.add(point)

  const spot = new THREE.SpotLight(0x44aaff, 7, 4, Math.PI / 5, 0.25, 2)
  spot.position.set(1.1, 1.1, 1.8)
  spot.target.position.set(0.35, 0, 0)
  scene.add(spot, spot.target)

  return {
    name: 'point-spot-light-materials',
    scene,
    camera: makeCamera([0, 0.2, 3.1], [0, 0, 0]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [39, 44, 48],
    minNonBackgroundRatio: 0.02,
    validate(rgba, { width }) {
      const pointLit = meanRegion(rgba, width, 20, 28, 42, 62)
      const spotLit = meanRegion(rgba, width, 54, 28, 76, 62)
      if (!(pointLit.r > pointLit.g + 25 && spotLit.b > spotLit.r + 8 && spotLit.b > spotLit.g + 15)) {
        throw new Error(`point/spot corpus should tint the two spheres red and blue, got point=${JSON.stringify(pointLit)} spot=${JSON.stringify(spotLit)}`)
      }
    },
  }
}

export function rectAreaLightCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(1.75, 1.75),
    new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, metalness: 0 }),
  ))

  const light = new THREE.RectAreaLight(0xffddaa, 18, 2.8, 1.4)
  light.position.set(0, 0, 2)
  light.lookAt(0, 0, 0)
  scene.add(light)

  return {
    name: 'rect-area-light-plane',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.18,
    validate(rgba, { width }) {
      const center = meanRegion(rgba, width, 32, 32, 64, 64)
      if (!(center.r > 220 && center.g > 210 && center.b > 190 && center.r > center.b + 10)) {
        throw new Error(`RectAreaLight corpus should render a warm lit plane, got ${JSON.stringify(center)}`)
      }
    },
  }
}

export function globalClippingPlaneCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0.12)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xff5533 }),
  ))

  return {
    name: 'global-clipping-plane',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: {
      width: CORPUS_RENDER_SIZE,
      height: CORPUS_RENDER_SIZE,
      format: 'rgba',
      clippingPlanes: [new THREE.Plane(new THREE.Vector3(1, 0, 0), 0)],
    },
    background: [0, 0, 31],
    minNonBackgroundRatio: 0.05,
    validate(rgba, { width }) {
      const clipped = meanRegion(rgba, width, 16, 32, 40, 64)
      const visible = meanRegion(rgba, width, 56, 32, 80, 64)
      if (!(clipped.r < 5 && clipped.g < 5 && clipped.b > 25 && visible.r > visible.g + 130 && visible.r > visible.b + 170)) {
        throw new Error(`global clipping corpus should keep only the red right half, got clipped=${JSON.stringify(clipped)} visible=${JSON.stringify(visible)}`)
      }
    },
  }
}

export function materialLocalClippingCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.08)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({
      color: 0x22ccff,
      clippingPlanes: [new THREE.Plane(new THREE.Vector3(0, 1, 0), 0)],
    }),
  ))

  return {
    name: 'material-local-clipping-plane',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: {
      width: CORPUS_RENDER_SIZE,
      height: CORPUS_RENDER_SIZE,
      format: 'rgba',
      localClippingEnabled: true,
    },
    background: [39, 39, 80],
    minNonBackgroundRatio: 0.05,
    validate(rgba, { width }) {
      const visible = meanRegion(rgba, width, 32, 16, 64, 40)
      const clipped = meanRegion(rgba, width, 32, 56, 64, 80)
      const clippedMatchesBackground = Math.abs(clipped.r - 39) <= 1 && Math.abs(clipped.g - 39) <= 1 && Math.abs(clipped.b - 80) <= 1
      if (!(visible.b > visible.r + 90 && visible.g > visible.r + 80 && clippedMatchesBackground)) {
        throw new Error(`local clipping corpus should keep only the cyan top half, got visible=${JSON.stringify(visible)} clipped=${JSON.stringify(clipped)}`)
      }
    },
  }
}
