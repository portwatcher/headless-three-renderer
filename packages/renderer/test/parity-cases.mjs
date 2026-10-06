// WebGL parity cases. Each case builds one small Three.js scene and names the pixels that
// test/parity.test.mjs compares with Chrome WebGLRenderer values in parity-references.json.
// browser-reference/parity.html renders the same cases to regenerate those values.
import { createParityHelpers } from './parity-scenes.mjs'

export const createParityCases = function ({ THREE, MToonMaterial }) {
  const h = createParityHelpers(THREE)
  const cases = []
  const add = (name, width, height, samples, build, tolerance = 2) => cases.push({ name, width, height, samples, build, tolerance })
  const sphereCase = (name, makeMaterial, lights = h.avatarLights, tolerance = 2) =>
    add(name, 64, 64, h.spherePoints, () => ({ scene: h.sceneWith(h.sphere(makeMaterial()), ...lights()), camera: h.persp() }), tolerance)

  // sRGB output uses the sRGB transfer function, also for very dark tones.
  const darkRamp = [0, 0.0002, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.006, 0.008, 0.012, 0.02, 0.04, 0.1, 0.3, 1]
  add('srgb-output-dark-ramp', 64, 4, h.rowPoints(64, 16, 2), () => ({
    scene: h.sceneWith(h.stripes(darkRamp.map((v) => new THREE.MeshBasicMaterial({ color: h.linearColor(v) })))),
    camera: h.ortho(),
  }))
  // sRGB textures decode before filtering and average mipmaps in linear light (SRGB8_ALPHA8).
  add('srgb-texture-bilinear', 64, 4, h.rowPoints(64, 16, 2), () => {
    const map = h.dataTexture(2, 1, (x) => (x === 0 ? [0, 0, 0, 255] : [255, 255, 255, 255]), { colorSpace: THREE.SRGBColorSpace })
    return { scene: h.sceneWith(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map }))), camera: h.ortho() }
  })
  add('srgb-texture-mipmap', 16, 16, h.gridPoints(16, 2), () => {
    const map = h.dataTexture(64, 64, (x, y) => ((x + y) % 2 === 0 ? [0, 0, 0, 255] : [255, 255, 255, 255]), {
      colorSpace: THREE.SRGBColorSpace,
      minFilter: THREE.LinearMipmapLinearFilter,
      generateMipmaps: true,
    })
    return { scene: h.sceneWith(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map }))), camera: h.ortho() }
  })
  // Texture transforms keep Three.js orientation for flipY true and false.
  for (const flipY of [true, false]) {
    add(`texture-transform-flipy-${flipY}`, 32, 32, h.gridPoints(32, 4), () => {
      const map = h.gradientTexture({ flipY, wrap: THREE.RepeatWrapping })
      map.offset.set(0.25, 0.1)
      map.repeat.set(1.5, 0.75)
      map.rotation = 0.3
      map.center.set(0.5, 0.5)
      return { scene: h.sceneWith(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map }))), camera: h.ortho() }
    })
  }
  add('matcap-flipy-true', 64, 64, h.spherePoints, () => ({
    scene: h.sceneWith(h.sphere(new THREE.MeshMatcapMaterial({ matcap: h.gradientTexture({ flipY: true }) }))),
    camera: h.persp(),
  }))

  // Direct PBR light: Three.js BRDF_GGX (V_GGX_SmithCorrelated, D_GGX, F_Schlick) and roughness
  // max(r, 0.0525) + geometryRoughness.
  sphereCase('pbr-metal-r0.5', () => new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0.5 }))
  sphereCase('pbr-gold-r0.3', () => new THREE.MeshStandardMaterial({ color: 0xffc35a, metalness: 1, roughness: 0.3 }), h.avatarLights, 3)
  sphereCase('pbr-dielectric-r0.2', () => new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 0, roughness: 0.2 }), h.avatarLights, 3)
  sphereCase('pbr-clearcoat', () => new THREE.MeshPhysicalMaterial({ color: 0xa02020, roughness: 0.6, clearcoat: 1, clearcoatRoughness: 0.15 }), h.avatarLights, 3)
  // roughnessMap (G) and metalnessMap (B) are separate slots: one map changes only its factor.
  sphereCase('pbr-metalness-map-only', () => new THREE.MeshStandardMaterial({
    color: 0xffffff,
    metalness: 1,
    roughness: 0.6,
    metalnessMap: h.dataTexture(16, 16, (x) => [255, 0, x * 17, 255]),
  }))
  sphereCase('pbr-roughness-map-only', () => new THREE.MeshStandardMaterial({
    color: 0xffffff,
    metalness: 0.6,
    roughness: 1,
    roughnessMap: h.dataTexture(16, 16, (x, y) => [0, 64 + y * 12, 0, 255]),
  }), h.avatarLights, 3)
  sphereCase('pbr-sheen', () => new THREE.MeshPhysicalMaterial({ color: 0x404080, roughness: 0.8, sheen: 1, sheenColor: 0xffffff, sheenRoughness: 0.4 }))
  add('pbr-grazing-plane', 64, 64, h.gridPoints(64, 4).filter(([, y]) => y > 20), () => {
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(3, 3), new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0.4 }))
    mesh.rotation.x = -1.25
    return { scene: h.sceneWith(mesh, ...h.avatarLights()), camera: h.persp([0, 0.4, 3], [0, -0.2, 0]) }
  })
  // Orthographic cameras use a constant view direction (geometryViewDir).
  add('ortho-view-direction', 64, 64, h.gridPoints(64, 4), () => {
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(3, 3), new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0.3 }))
    mesh.rotation.y = 0.6
    const point = new THREE.PointLight(0xffffff, 8, 0, 2)
    point.position.set(1.2, 0.6, 1.5)
    return { scene: h.sceneWith(mesh, point), camera: h.ortho() }
  })

  // scene.environment: PMREM (CubeUV) lighting for MeshStandardMaterial/MeshPhysicalMaterial only.
  const envCase = (name, makeMaterial, configure = () => {}, kind = 'hdr', tolerance = 2) =>
    add(name, 64, 64, h.spherePoints, () => {
      const scene = h.sceneWith(h.sphere(makeMaterial()))
      scene.environment = h.environment(kind)
      configure(scene)
      return { scene, camera: h.persp() }
    }, tolerance)
  envCase('env-dielectric-r0.6', () => new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 0, roughness: 0.6 }))
  envCase('env-metal-r0.3', () => new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0.3 }))
  envCase('env-mirror-rotated', () => new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0 }), (scene) => {
    scene.environmentRotation.set(0, 1.2, 0)
  }, 'hdr', 3)
  envCase('env-intensity', () => new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 0.5, roughness: 0.5, envMapIntensity: 3 }), (scene) => {
    scene.environmentIntensity = 0.4
  })
  envCase('env-srgb-ldr', () => new THREE.MeshStandardMaterial({ color: 0xd0d0d0, metalness: 0.2, roughness: 0.5 }), () => {}, 'srgb')
  envCase('env-clearcoat', () => new THREE.MeshPhysicalMaterial({ color: 0x206040, roughness: 0.7, clearcoat: 1, clearcoatRoughness: 0.1 }))
  for (const [name, make] of [
    ['env-ignored-lambert', () => new THREE.MeshLambertMaterial({ color: 0xffffff })],
    ['env-ignored-phong', () => new THREE.MeshPhongMaterial({ color: 0xffffff, shininess: 30 })],
    ['env-ignored-toon', () => new THREE.MeshToonMaterial({ color: 0xffffff })],
  ]) {
    envCase(name, make, (scene) => scene.add(...h.avatarLights()))
  }
  // PMREM gives no image-based light for equirectangular inputs narrower than 64 px or cube faces
  // smaller than 16 px; larger cubes are mirrored like flipEnvMap = -1.
  const envWithLights = (name, makeEnvironment) => envCase(name, () => new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 0.7, roughness: 0.25 }), (scene) => {
    scene.environment = makeEnvironment()
    scene.add(...h.avatarLights())
  }, 'hdr', 3)
  envWithLights('env-narrow-equirect', () => h.environment('hdr', 32))
  envWithLights('env-cube-8px-faces', () => h.cubeEnvironment(8))
  envWithLights('env-cube-16px-faces', () => h.cubeEnvironment(16))
  add('envmap-standard-material', 64, 64, h.spherePoints, () => {
    const material = new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 0.6, roughness: 0.4, envMapIntensity: 0.7 })
    material.envMap = h.environment('hdr')
    return { scene: h.sceneWith(h.sphere(material)), camera: h.persp() }
  })
  // Legacy MeshBasic/Lambert/Phong envMap combine; NoColorSpace 8-bit data stays linear.
  add('envmap-lambert-mix', 64, 64, h.spherePoints, () => {
    const material = new THREE.MeshLambertMaterial({ color: 0x806040, combine: THREE.MixOperation, reflectivity: 0.5 })
    material.envMap = h.environment('srgb')
    return { scene: h.sceneWith(h.sphere(material), ...h.avatarLights()), camera: h.persp() }
  })
  add('envmap-basic-linear-data', 64, 64, h.spherePoints, () => {
    const material = new THREE.MeshBasicMaterial({ color: 0xffffff, combine: THREE.MultiplyOperation, reflectivity: 0.8 })
    material.envMap = h.environment('linear')
    return { scene: h.sceneWith(h.sphere(material)), camera: h.persp() }
  })
  add('background-equirect', 64, 32, h.gridPoints(32, 4).map(([x, y]) => [x * 2, y]), () => {
    const scene = new THREE.Scene()
    scene.background = h.environment('srgb')
    return { scene, camera: h.persp([0, 0, 0.01], [0, 0, -1], 90, 2) }
  })
  // Texture backgrounds without the sRGB transfer are tone mapped (toneMapped = transfer !== SRGB).
  add('background-linear-aces', 64, 32, h.gridPoints(32, 4).map(([x, y]) => [x * 2, y]), () => {
    const scene = new THREE.Scene()
    scene.background = h.environment('linear')
    return { scene, camera: h.persp([0, 0, 0.01], [0, 0, -1], 90, 2), toneMapping: THREE.ACESFilmicToneMapping }
  })

  // Fog mixes the output-encoded fog color after the sRGB conversion.
  add('fog-exp2-dark', 64, 64, h.spherePoints, () => {
    const scene = h.sceneWith(h.sphere(new THREE.MeshLambertMaterial({ color: 0xffffff })), ...h.avatarLights())
    scene.fog = new THREE.FogExp2(0x101418, 0.35)
    return { scene, camera: h.persp() }
  })
  // HDR material colors are not clamped before tone mapping.
  add('hdr-color-aces', 64, 4, h.rowPoints(64, 8, 2), () => ({
    scene: h.sceneWith(h.stripes([0.5, 1, 1.5, 2.5, 4, 8, 16, 32].map((v) => new THREE.MeshBasicMaterial({ color: h.linearColor(v, v * 0.8, v * 0.6) })))),
    camera: h.ortho(),
    toneMapping: THREE.ACESFilmicToneMapping,
  }))

  // Normal-map frames: vertex tangents or getTangentFrame derivatives, with the DoubleSide
  // and BackSide face flips of normal_fragment_begin; bump maps carry faceDirection.
  const backFacePlane = (name, { tangents, side, bump }) => add(name, 64, 64, h.gridPoints(64, 4), () => {
    const geometry = new THREE.PlaneGeometry(2, 2, 4, 4)
    if (tangents) geometry.computeTangents()
    const parameters = { color: 0xc0c0c0, roughness: 0.4, side }
    if (bump) {
      parameters.bumpMap = h.bumpyNormalMap()
      parameters.bumpScale = 3
    } else {
      parameters.normalMap = h.bumpyNormalMap()
      parameters.normalScale = new THREE.Vector2(1, -0.7)
    }
    const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial(parameters))
    mesh.rotation.y = Math.PI
    mesh.rotation.x = 0.3
    return { scene: h.sceneWith(mesh, ...h.avatarLights()), camera: h.ortho() }
  }, 3)
  backFacePlane('tbn-double-back-tangents', { tangents: true, side: THREE.DoubleSide })
  backFacePlane('tbn-double-back-derivatives', { tangents: false, side: THREE.DoubleSide })
  backFacePlane('tbn-backside-tangents', { tangents: true, side: THREE.BackSide })
  backFacePlane('bump-double-back', { tangents: false, side: THREE.DoubleSide, bump: true })
  // BackSide keeps faceDirection = 1 (WebGL flips the front-face winding) and negates bumpScale.
  backFacePlane('bump-backside', { tangents: false, side: THREE.BackSide, bump: true })
  sphereCase('normal-map-derivative-frame', () => new THREE.MeshStandardMaterial({ color: 0xc0c0c0, roughness: 0.4, normalMap: h.bumpyNormalMap() }), h.avatarLights, 3)

  if (MToonMaterial) {
    add('mtoon-matcap', 64, 64, h.spherePoints, () => {
      const material = new MToonMaterial()
      material.color.set(0x705050)
      material.matcapTexture = h.gradientTexture({ flipY: false })
      material.matcapFactor.set(0xffffff)
      return { scene: h.sceneWith(h.sphere(material), ...h.avatarLights()), camera: h.persp() }
    })
    // Outlines draw back faces but three-vrm lights them with the outward normal.
    add('mtoon-outline-lit', 96, 96, [[10, 48], [48, 10], [85, 48], [48, 85], [48, 48]], () => {
      const material = new MToonMaterial()
      material.color.set(0xd0c0b0)
      material.outlineWidthMode = 'worldCoordinates'
      material.outlineWidthFactor = 0.06
      material.outlineColorFactor.set(0x203040)
      material.outlineLightingMixFactor = 0.5
      const outline = material.clone()
      outline.isOutline = true
      outline.side = THREE.BackSide
      const geometry = new THREE.SphereGeometry(0.85, 64, 32)
      geometry.clearGroups()
      geometry.addGroup(0, geometry.index.count, 0)
      geometry.addGroup(0, geometry.index.count, 1)
      return { scene: h.sceneWith(new THREE.Mesh(geometry, [material, outline]), ...h.avatarLights()), camera: h.persp() }
    })
  }
  return cases
}
