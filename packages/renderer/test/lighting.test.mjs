import test from 'node:test'
import assert from 'node:assert/strict'
import * as THREE from 'three'
import pkg from '../dist/index.js'

// Expected values follow Three.js r155+ physical light units (checked against r180
// WebGLRenderer output, see also parity.test.mjs): ambient, hemisphere, LightProbe, and light-map irradiance reach lit
// materials as irradiance * BRDF_Lambert(diffuseColor), with diffuseColor = color * (1 - metalness).
// Linear output compares the lighting itself; expected values are linear radiance x 255.
const { Renderer } = pkg
const SIZE = 32
const INV_PI = 1 / Math.PI

const camera = () => {
  const result = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10)
  result.position.z = 2
  result.updateMatrixWorld(true)
  return result
}
const sceneWith = (material, ...objects) => {
  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(1.5, 1.5), material), ...objects)
  scene.updateMatrixWorld(true)
  return scene
}
const center = (renderer, scene, outputColorSpace = THREE.LinearSRGBColorSpace) => {
  const frame = renderer.render(scene, camera(), {
    width: SIZE, height: SIZE, format: 'rgba', toneMapping: THREE.NoToneMapping, outputColorSpace, background: [0, 0, 0, 0],
  })
  const at = (SIZE / 2 * SIZE + SIZE / 2) * 4
  return [...frame.subarray(at, at + 3)]
}
const assertLinear = (actual, expected, label, tolerance = 2) => {
  const bytes = (Array.isArray(expected) ? expected : [expected, expected, expected])
    .map((value) => Math.round(Math.min(1, Math.max(0, value)) * 255))
  actual.forEach((value, i) => assert.ok(
    Math.abs(value - bytes[i]) <= tolerance,
    `${label}: channel ${i} expected ${bytes} (linear x 255), got ${actual}`,
  ))
}
const solidTexture = (rgba) => {
  const texture = new THREE.DataTexture(new Uint8Array(rgba), 1, 1)
  texture.needsUpdate = true
  return texture
}
const litMaterials = {
  standard: () => new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.5, metalness: 0 }),
  physical: () => new THREE.MeshPhysicalMaterial({ color: 0xffffff, roughness: 0.5, metalness: 0 }),
  lambert: () => new THREE.MeshLambertMaterial({ color: 0xffffff }),
  phong: () => new THREE.MeshPhongMaterial({ color: 0xffffff }),
  toon: () => new THREE.MeshToonMaterial({ color: 0xffffff }),
}
const directionalAlongNormal = (intensity) => {
  const light = new THREE.DirectionalLight(0xffffff, intensity)
  light.position.set(0, 0, 1)
  return light
}

test('lit materials divide AmbientLight irradiance by PI and sum ambient colors', () => {
  const renderer = new Renderer()
  try {
    for (const [name, make] of Object.entries(litMaterials)) {
      // 0.4.3 rendered the full ambient color here (linear 255 instead of 81).
      assertLinear(center(renderer, sceneWith(make(), new THREE.AmbientLight(0xffffff, 1))), INV_PI, `${name} ambient`)
      assertLinear(
        center(renderer, sceneWith(make(), new THREE.AmbientLight(0xff0000, 1), new THREE.AmbientLight(0xffffff, 0.5))),
        [1.5 * INV_PI, 0.5 * INV_PI, 0.5 * INV_PI],
        `${name} red + white ambient`,
      )
      assertLinear(center(renderer, sceneWith(make(), new THREE.AmbientLight(0xffffff, 0))), 0, `${name} zero ambient`)
    }
  } finally {
    renderer.dispose()
  }
})

test('the no-light fallback stays for scenes without any light source', () => {
  const renderer = new Renderer()
  const lightMap = solidTexture([255, 255, 255, 255])
  try {
    const unlit = center(renderer, sceneWith(litMaterials.standard()))
    assert.ok(unlit.every((value) => value > 40), `scenes without light keep the preview fallback, got ${unlit}`)
    // A baked light map is a light source: Three.js shows only lightMap * diffuse / PI.
    const baked = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.5, metalness: 0.5, lightMap })
    assertLinear(center(renderer, sceneWith(baked)), 0.5 * INV_PI, 'light map without lights')
  } finally {
    renderer.dispose()
    lightMap.dispose()
  }
})

test('standard indirect diffuse excludes metalness and applies aoMap like Three.js', () => {
  const renderer = new Renderer()
  const aoMap = solidTexture([128, 128, 128, 255])
  const lightMap = solidTexture([255, 255, 255, 255])
  const ambient = () => new THREE.AmbientLight(0xffffff, 1)
  const standard = (parameters) => new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.5, ...parameters })
  try {
    assertLinear(center(renderer, sceneWith(standard({ metalness: 1 }), ambient())), 0, 'metalness 1')
    assertLinear(center(renderer, sceneWith(standard({ metalness: 0.5 }), ambient())), 0.5 * INV_PI, 'metalness 0.5')
    assertLinear(center(renderer, sceneWith(standard({ metalness: 0, aoMap }), ambient())), (128 / 255) * INV_PI, 'aoMap')
    assertLinear(
      center(renderer, sceneWith(standard({ metalness: 0.5, lightMap }), ambient())),
      0.5 * 2 * INV_PI,
      'light map adds to ambient irradiance',
    )
  } finally {
    renderer.dispose()
    aoMap.dispose()
    lightMap.dispose()
  }
})

test('HemisphereLight irradiance uses the Three.js position direction and PI normalization', () => {
  const renderer = new Renderer()
  const hemisphere = (configure) => {
    const light = new THREE.HemisphereLight(0xffffff, 0x000000, 1)
    configure?.(light)
    return light
  }
  try {
    for (const [name, make] of Object.entries(litMaterials)) {
      // The plane normal is +Z: the default light at +Y gives sky weight 0.5.
      assertLinear(center(renderer, sceneWith(make(), hemisphere())), 0.5 * INV_PI, `${name} default hemisphere`)
      assertLinear(
        center(renderer, sceneWith(make(), hemisphere((light) => light.position.set(0, 0, 3)))),
        INV_PI,
        `${name} hemisphere positioned on the normal`,
      )
      // Three.js ignores the light rotation and a light at the origin has no sky direction.
      assertLinear(
        center(renderer, sceneWith(make(), hemisphere((light) => { light.rotation.x = Math.PI / 2 }))),
        0.5 * INV_PI,
        `${name} rotated hemisphere`,
      )
      assertLinear(
        center(renderer, sceneWith(make(), hemisphere((light) => light.position.set(0, 0, 0)))),
        0.5 * INV_PI,
        `${name} hemisphere at the origin`,
      )
    }
    const metal = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.5, metalness: 1 })
    assertLinear(center(renderer, sceneWith(metal, hemisphere())), 0, 'metallic hemisphere')
  } finally {
    renderer.dispose()
  }
})

test('LightProbe irradiance stays Lambert-normalized and metalness-weighted', () => {
  const renderer = new Renderer()
  const probe = () => {
    const light = new THREE.LightProbe()
    light.sh.coefficients[0].set(1, 1, 1)
    return light
  }
  const irradiance = 0.886227
  try {
    for (const [name, make] of Object.entries(litMaterials)) {
      assertLinear(center(renderer, sceneWith(make(), probe())), irradiance * INV_PI, `${name} probe`)
    }
    const halfMetal = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.5, metalness: 0.5 })
    assertLinear(center(renderer, sceneWith(halfMetal, probe())), 0.5 * irradiance * INV_PI, 'metalness 0.5 probe')
  } finally {
    renderer.dispose()
  }
})

test('AmbientLight still applies when the scene has an environment map', () => {
  const renderer = new Renderer()
  const environment = new THREE.DataTexture(new Uint8Array(8 * 4 * 4), 8, 4)
  environment.mapping = THREE.EquirectangularReflectionMapping
  environment.needsUpdate = true
  try {
    for (const name of ['standard', 'physical']) {
      const scene = sceneWith(litMaterials[name](), new THREE.AmbientLight(0xffffff, 1))
      scene.environment = environment
      // A black environment adds nothing; 0.4.3 dropped the ambient term and rendered black.
      assertLinear(center(renderer, scene), INV_PI, `${name} ambient with environment`)
    }
  } finally {
    renderer.dispose()
    environment.dispose()
  }
})

test('direct PBR diffuse uses BRDF_Lambert without a Fresnel weight like RE_Direct_Physical', () => {
  const renderer = new Renderer()
  // Light, view, and normal coincide: GGX D = 1 / (PI alpha^2) and Smith V = 0.25 at roughness 0.5.
  const specular = (f0) => f0 * 0.25 / (Math.PI * 0.0625)
  try {
    const dielectric = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.5, metalness: 0 })
    assertLinear(center(renderer, sceneWith(dielectric, directionalAlongNormal(1))), INV_PI + specular(0.04), 'dielectric')
    const halfMetal = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.5, metalness: 0.5 })
    // The native GGX terms keep small epsilons, so this specular-heavy sample gets 3 levels.
    assertLinear(
      center(renderer, sceneWith(halfMetal, directionalAlongNormal(1))),
      0.5 * INV_PI + specular(0.52),
      'metalness 0.5',
      3,
    )
  } finally {
    renderer.dispose()
  }
})

test('a white PBR face under avatar viewer lights matches Three.js r180 WebGL output', () => {
  const renderer = new Renderer()
  const viewerLights = (scale) => {
    const key = new THREE.DirectionalLight(0xffffff, 1.2 * scale)
    key.position.set(1, 1.5, 1)
    return [new THREE.AmbientLight(0xffffff, 0.7 * scale), key]
  }
  const face = () => new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.5, metalness: 0 })
  try {
    // sRGB values measured with THREE.WebGLRenderer (r180 and r183); 0.4.3 rendered 241 and 255.
    const current = center(renderer, sceneWith(face(), ...viewerLights(1)), THREE.SRGBColorSpace)
    current.forEach((value) => assert.ok(Math.abs(value - 172) <= 3, `ambient 0.7 + key 1.2: expected 172, got ${current}`))
    const raised = center(renderer, sceneWith(face(), ...viewerLights(Math.PI / 1.9)), THREE.SRGBColorSpace)
    raised.forEach((value) => assert.ok(Math.abs(value - 215) <= 3, `lights x PI / 1.9: expected 215, got ${raised}`))
  } finally {
    renderer.dispose()
  }
})
