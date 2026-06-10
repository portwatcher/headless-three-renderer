import test from 'node:test'
import assert from 'node:assert/strict'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { assertValidPng, meanRgba, nonBackgroundRatio } from './helpers.mjs'

const { Renderer, renderToTarget } = pkg

const SIZE = 128
const BG = [26, 26, 26] // 0.1 * 255

function makeCamera() {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(2.5, 2, 3)
  camera.lookAt(0, 0, 0)
  return camera
}

function addLights(scene) {
  scene.add(new THREE.AmbientLight(0xffffff, 0.3))
  const dir = new THREE.DirectionalLight(0xffffff, 1.2)
  dir.position.set(3, 4, 2)
  scene.add(dir)
}

function renderRgba(scene, camera, options = {}) {
  const r = new Renderer()
  return r.render(scene, camera, { width: SIZE, height: SIZE, format: 'rgba', ...options })
}

function makeEnvironmentTexture() {
  const data = new Uint8Array([
    255, 255, 255, 255,
    64, 128, 255, 255,
    255, 180, 96, 255,
    18, 24, 36, 255,
  ])
  const texture = new THREE.DataTexture(data, 2, 2, THREE.RGBAFormat)
  texture.needsUpdate = true
  return texture
}

function solidTexture(r, g, b, a = 255) {
  const texture = new THREE.DataTexture(new Uint8Array([r, g, b, a]), 1, 1, THREE.RGBAFormat)
  texture.needsUpdate = true
  return texture
}

function rgbaTexture(data, width, height) {
  const texture = new THREE.DataTexture(new Uint8Array(data), width, height, THREE.RGBAFormat)
  texture.needsUpdate = true
  return texture
}

function constantUvPlane(u, v) {
  const geometry = new THREE.PlaneGeometry(2, 2)
  const uv = new Float32Array(geometry.getAttribute('uv').count * 2)
  for (let i = 0; i < geometry.getAttribute('uv').count; i++) {
    uv[i * 2] = u
    uv[i * 2 + 1] = v
  }
  geometry.setAttribute('uv', new THREE.BufferAttribute(uv, 2))
  return geometry
}

function foldedIndexedGeometry() {
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

function setConstantUvAttribute(geometry, name, u, v) {
  const uv = new Float32Array(geometry.getAttribute('uv').count * 2)
  for (let i = 0; i < geometry.getAttribute('uv').count; i++) {
    uv[i * 2] = u
    uv[i * 2 + 1] = v
  }
  geometry.setAttribute(name, new THREE.BufferAttribute(uv, 2))
}

function meanAbsDiff(a, b) {
  assert.equal(a.length, b.length)
  let total = 0
  for (let i = 0; i < a.length; i += 4) {
    total += Math.abs(a[i] - b[i])
    total += Math.abs(a[i + 1] - b[i + 1])
    total += Math.abs(a[i + 2] - b[i + 2])
  }
  return total / ((a.length / 4) * 3)
}

function meanRegion(rgba, width, height, x0, y0, x1, y1) {
  const sum = { r: 0, g: 0, b: 0, a: 0 }
  let count = 0
  for (let y = y0; y < y1; y += 1) {
    assert.ok(y >= 0 && y < height)
    for (let x = x0; x < x1; x += 1) {
      assert.ok(x >= 0 && x < width)
      const i = (y * width + x) * 4
      sum.r += rgba[i]
      sum.g += rgba[i + 1]
      sum.b += rgba[i + 2]
      sum.a += rgba[i + 3]
      count += 1
    }
  }
  return {
    r: sum.r / count,
    g: sum.g / count,
    b: sum.b / count,
    a: sum.a / count,
  }
}

function countRegionPixels(rgba, width, height, x0, y0, x1, y1, predicate) {
  let count = 0
  for (let y = y0; y < y1; y += 1) {
    assert.ok(y >= 0 && y < height)
    for (let x = x0; x < x1; x += 1) {
      assert.ok(x >= 0 && x < width)
      const i = (y * width + x) * 4
      if (predicate(rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3])) {
        count += 1
      }
    }
  }
  return count
}

function maxLuminance(rgba) {
  let max = 0
  for (let i = 0; i < rgba.length; i += 4) {
    max = Math.max(max, 0.2126 * rgba[i] + 0.7152 * rgba[i + 1] + 0.0722 * rgba[i + 2])
  }
  return max
}

function nonBackgroundBounds(rgba, width, height, bg, tolerance = 2) {
  let minX = width
  let minY = height
  let maxX = -1
  let maxY = -1
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const i = (y * width + x) * 4
      if (
        Math.abs(rgba[i] - bg[0]) > tolerance ||
        Math.abs(rgba[i + 1] - bg[1]) > tolerance ||
        Math.abs(rgba[i + 2] - bg[2]) > tolerance
      ) {
        minX = Math.min(minX, x)
        minY = Math.min(minY, y)
        maxX = Math.max(maxX, x)
        maxY = Math.max(maxY, y)
      }
    }
  }
  return {
    width: maxX >= minX ? maxX - minX + 1 : 0,
    height: maxY >= minY ? maxY - minY + 1 : 0,
  }
}

test('rgba format returns raw pixel buffer of the expected byte length', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.1, 0.1, 0.1)
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial({ color: 0xff00ff })))

  const buf = renderRgba(scene, makeCamera())
  assert.equal(buf.length, SIZE * SIZE * 4, 'rgba buffer must be width*height*4 bytes')
})

test('ArrayCamera and CubeCamera inputs fail clearly', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.1, 0.1, 0.1)
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial({ color: 0xff00ff })))

  const subCamera = makeCamera()
  const arrayCamera = new THREE.ArrayCamera([subCamera])
  arrayCamera.projectionMatrix.copy(subCamera.projectionMatrix)
  arrayCamera.matrixWorldInverse.copy(subCamera.matrixWorldInverse)
  assert.throws(
    () => renderRgba(scene, arrayCamera),
    /ArrayCamera.*not supported/i,
  )

  const cubeTarget = new THREE.WebGLCubeRenderTarget(16)
  const cubeCamera = new THREE.CubeCamera(0.01, 100, cubeTarget)
  assert.throws(
    () => renderRgba(scene, cubeCamera),
    /CubeCamera.*not supported/i,
  )
})

test('MeshBasicMaterial renders foreground pixels distinct from background', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.1, 0.1, 0.1)
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial({ color: 0xffaa00 })))

  const rgba = renderRgba(scene, makeCamera())
  const ratio = nonBackgroundRatio(rgba, BG)
  assert.ok(ratio > 0.05, `expected mesh to cover >5% of frame, got ${(ratio * 100).toFixed(1)}%`)
  assert.ok(ratio < 0.95, `expected background to be visible, got ${(ratio * 100).toFixed(1)}% non-bg`)
})

test('different materials produce visibly different outputs', async () => {
  const camera = makeCamera()

  const sceneA = new THREE.Scene()
  sceneA.background = new THREE.Color(0.1, 0.1, 0.1)
  sceneA.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial({ color: 0xff0000 })))

  const sceneB = new THREE.Scene()
  sceneB.background = new THREE.Color(0.1, 0.1, 0.1)
  sceneB.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial({ color: 0x00ff00 })))

  const a = renderRgba(sceneA, camera)
  const b = renderRgba(sceneB, camera)
  const meanA = meanRgba(a)
  const meanB = meanRgba(b)

  assert.ok(meanA.r > meanB.r + 5, `red scene should have higher red channel mean (${meanA.r} vs ${meanB.r})`)
  assert.ok(meanB.g > meanA.g + 5, `green scene should have higher green channel mean (${meanB.g} vs ${meanA.g})`)
})

test('MeshNormalMaterial renders view-space normal colors', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshNormalMaterial()))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  assert.ok(mean.b > mean.r + 20, `front-facing normal plane should have a dominant blue channel (${mean.b} vs ${mean.r})`)
  assert.ok(mean.b > mean.g + 20, `front-facing normal plane should have a dominant blue channel (${mean.b} vs ${mean.g})`)
})

test('MeshNormalMaterial normalMap perturbs output normals', () => {
  function renderNormalMaterial(normalMap) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshNormalMaterial({ normalMap }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const unperturbed = renderNormalMaterial(null)
  const tangentRight = renderNormalMaterial(solidTexture(255, 128, 128))
  assert.ok(tangentRight.r > unperturbed.r + 40, `normalMap should tilt normal output toward red (${tangentRight.r} vs ${unperturbed.r})`)
  assert.ok(unperturbed.b > tangentRight.b + 40, `normalMap should reduce the front-facing blue normal channel (${unperturbed.b} vs ${tangentRight.b})`)
})

test('MeshNormalMaterial bumpMap perturbs output normals', () => {
  function renderBumpMaterial(bumpScale) {
    const bumpMap = rgbaTexture([
      0, 0, 0, 255,
      255, 255, 255, 255,
    ], 2, 1)
    bumpMap.magFilter = THREE.LinearFilter
    bumpMap.minFilter = THREE.LinearFilter

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshNormalMaterial({ bumpMap, bumpScale }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return renderRgba(scene, camera, { width: 64, height: 64 })
  }

  const flat = renderBumpMaterial(0)
  const bumped = renderBumpMaterial(4)
  const diff = meanAbsDiff(flat, bumped)
  assert.ok(diff > 2, `bumpMap should perturb MeshNormalMaterial output normals (diff=${diff.toFixed(2)})`)
})

test('MeshNormalMaterial normalMap samples the selected secondary UV channel', () => {
  function renderNormalMaterial(channel) {
    const normalMap = rgbaTexture([
      128, 128, 255, 255,
      255, 128, 128, 255,
    ], 2, 1)
    normalMap.channel = channel

    const geometry = new THREE.PlaneGeometry(2, 2)
    setConstantUvAttribute(geometry, 'uv1', 0.75, 0.5)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      geometry,
      new THREE.MeshNormalMaterial({ normalMap }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const primary = renderNormalMaterial(0)
  const secondary = renderNormalMaterial(1)
  assert.ok(secondary.r > primary.r + 20, `normalMap channel=1 should sample uv1's tangent-right texel (${secondary.r} vs ${primary.r})`)
  assert.ok(primary.b > secondary.b + 20, `normalMap channel=0 should retain more front-facing blue normal output (${primary.b} vs ${secondary.b})`)
})

test('MeshNormalMaterial bumpMap samples the selected secondary UV channel', () => {
  function renderBumpMaterial(channel) {
    const bumpMap = rgbaTexture([
      0, 0, 0, 255,
      255, 255, 255, 255,
    ], 2, 1)
    bumpMap.channel = channel
    bumpMap.magFilter = THREE.LinearFilter
    bumpMap.minFilter = THREE.LinearFilter

    const geometry = new THREE.PlaneGeometry(2, 2)
    setConstantUvAttribute(geometry, 'uv1', 0.25, 0.5)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      geometry,
      new THREE.MeshNormalMaterial({ bumpMap, bumpScale: 4 }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return renderRgba(scene, camera, { width: 64, height: 64 })
  }

  const primary = renderBumpMaterial(0)
  const secondary = renderBumpMaterial(1)
  const diff = meanAbsDiff(primary, secondary)
  assert.ok(diff > 2, `bumpMap channel=1 should use uv1 and change the bump perturbation (diff=${diff.toFixed(2)})`)
})

test('MeshNormalMaterial flatShading uses per-face normals on indexed geometry', () => {
  function renderFlatShading(flatShading) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      foldedIndexedGeometry(),
      new THREE.MeshNormalMaterial({ flatShading, side: THREE.DoubleSide }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 4)
    camera.lookAt(0, 0, 0.2)
    return renderRgba(scene, camera, { width: 96, height: 96 })
  }

  const smooth = renderFlatShading(false)
  const flat = renderFlatShading(true)
  const diff = meanAbsDiff(smooth, flat)
  assert.ok(diff > 3, `flatShading should change MeshNormalMaterial face normals on indexed geometry (diff=${diff.toFixed(2)})`)
})

test('MeshMatcapMaterial samples matcap texture without lights', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshMatcapMaterial({ color: 0xffffff, matcap: solidTexture(0, 255, 0) }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  assert.ok(mean.g > mean.r + 40, `matcap texture should drive green output without lights (${mean.g} vs ${mean.r})`)
  assert.ok(mean.g > mean.b + 40, `matcap texture should drive green output without lights (${mean.g} vs ${mean.b})`)
})

test('MeshMatcapMaterial normalMap changes matcap lookup', () => {
  function renderMatcap(normalMap) {
    const matcap = rgbaTexture([
      255, 0, 0, 255,
      0, 255, 0, 255,
    ], 2, 1)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshMatcapMaterial({
        color: 0xffffff,
        matcap,
        normalMap,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const unperturbed = renderMatcap(null)
  const tangentRight = renderMatcap(solidTexture(255, 128, 128))
  assert.ok(unperturbed.r > unperturbed.g + 40, `unperturbed matcap lookup should sample the red center texel (${unperturbed.r} vs ${unperturbed.g})`)
  assert.ok(tangentRight.g > tangentRight.r + 40, `normalMap should shift matcap lookup toward the green texel (${tangentRight.g} vs ${tangentRight.r})`)
})

test('MeshMatcapMaterial flatShading changes face-normal matcap lookup', () => {
  const data = []
  for (let y = 0; y < 4; y += 1) {
    for (let x = 0; x < 4; x += 1) {
      data.push(x * 85, y * 85, 255 - x * 85, 255)
    }
  }

  function renderFlatShading(flatShading) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      foldedIndexedGeometry(),
      new THREE.MeshMatcapMaterial({
        color: 0xffffff,
        matcap: rgbaTexture(data, 4, 4),
        flatShading,
        side: THREE.DoubleSide,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 4)
    camera.lookAt(0, 0, 0.2)
    return renderRgba(scene, camera, { width: 96, height: 96 })
  }

  const smooth = renderFlatShading(false)
  const flat = renderFlatShading(true)
  const diff = meanAbsDiff(smooth, flat)
  assert.ok(diff > 1, `flatShading should change MeshMatcapMaterial matcap lookup on indexed geometry (diff=${diff.toFixed(2)})`)
})

test('MeshMatcapMaterial map multiplies matcap color and applies UV transforms', () => {
  function renderMatcapMap(offsetX) {
    const map = rgbaTexture([
      255, 0, 0, 255,
      0, 255, 0, 255,
    ], 2, 1)
    map.offset.set(offsetX, 0)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.25, 0.5),
      new THREE.MeshMatcapMaterial({
        color: 0xffffff,
        matcap: solidTexture(255, 255, 255),
        map,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const red = renderMatcapMap(0)
  const green = renderMatcapMap(0.5)
  assert.ok(red.r > red.g + 40, `matcap color map should multiply output with the red texel (${red.r} vs ${red.g})`)
  assert.ok(green.g > green.r + 40, `matcap color map offset should sample the green texel (${green.g} vs ${green.r})`)
})

test('MeshMatcapMaterial map samples the selected secondary UV channel', () => {
  function renderMatcapMap(channel) {
    const map = rgbaTexture([
      0, 255, 0, 255,
      255, 0, 0, 255,
    ], 2, 1)
    map.channel = channel

    const geometry = constantUvPlane(0.25, 0.5)
    setConstantUvAttribute(geometry, 'uv1', 0.75, 0.5)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      geometry,
      new THREE.MeshMatcapMaterial({
        color: 0xffffff,
        matcap: solidTexture(255, 255, 255),
        map,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const primary = renderMatcapMap(0)
  const secondary = renderMatcapMap(1)
  assert.ok(primary.g > primary.r + 40, `matcap map channel=0 should sample the primary UV green texel (${primary.g} vs ${primary.r})`)
  assert.ok(secondary.r > secondary.g + 40, `matcap map channel=1 should sample the uv1 red texel (${secondary.r} vs ${secondary.g})`)
})

test('normalMap applies texture UV transforms', () => {
  function renderWithOffset(offsetX) {
    const normalMap = rgbaTexture([
      128, 128, 255, 255,
      255, 128, 128, 255,
    ], 2, 1)
    normalMap.offset.set(offsetX, 0)
    normalMap.magFilter = THREE.LinearFilter
    normalMap.minFilter = THREE.LinearFilter

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshStandardMaterial({
        color: 0xffffff,
        roughness: 1,
        metalness: 0,
        normalMap,
      }),
    ))

    const light = new THREE.DirectionalLight(0xffffff, 4)
    light.position.set(3, 0, 3)
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return renderRgba(scene, camera, { width: 64, height: 64 })
  }

  const unshifted = renderWithOffset(0)
  const shifted = renderWithOffset(0.5)
  const diff = meanAbsDiff(unshifted, shifted)
  assert.ok(diff > 2, `normalMap offset should change the sampled tangent-space normals (diff=${diff.toFixed(2)})`)
})

test('normalMap honors nearest texture filters', () => {
  function renderWithFilter(filter) {
    const normalMap = rgbaTexture([
      128, 128, 255, 255,
      255, 128, 128, 255,
    ], 2, 1)
    normalMap.magFilter = filter
    normalMap.minFilter = filter

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.45, 0.5),
      new THREE.MeshNormalMaterial({ normalMap }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const nearest = renderWithFilter(THREE.NearestFilter)
  const linear = renderWithFilter(THREE.LinearFilter)
  assert.ok(nearest.b > nearest.r + 20, `NearestFilter should choose the flat blue normal texel (${nearest.b} vs ${nearest.r})`)
  assert.ok(linear.r > nearest.r + 20, `LinearFilter should blend in the tangent-right red normal texel (${linear.r} vs ${nearest.r})`)
  assert.ok(nearest.b > linear.b + 10, `NearestFilter should preserve a stronger blue normal output (${nearest.b} vs ${linear.b})`)
})

test('MeshPhongMaterial renders Blinn-Phong specular and honors specularMap', () => {
  function renderPhong(specularMap) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.SphereGeometry(1, 48, 24),
      new THREE.MeshPhongMaterial({
        color: 0x000000,
        specular: 0xffffff,
        shininess: 120,
        specularMap,
      }),
    ))

    const light = new THREE.DirectionalLight(0xffffff, 4)
    light.position.set(0, 0, 3)
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return renderRgba(scene, camera, { width: 96, height: 96 })
  }

  const fullSpecular = maxLuminance(renderPhong(null))
  const maskedSpecular = maxLuminance(renderPhong(solidTexture(0, 0, 0)))
  assert.ok(fullSpecular > maskedSpecular + 80, `specularMap should suppress Phong highlight (${fullSpecular} vs ${maskedSpecular})`)
})

test('MeshPhongMaterial specularMap samples the selected secondary UV channel', () => {
  const specularMap = rgbaTexture([
    0, 0, 0, 255,
    255, 0, 0, 255,
  ], 2, 1)
  specularMap.channel = 1

  const geometry = constantUvPlane(0.25, 0.5)
  const uv1 = new Float32Array(geometry.getAttribute('uv').count * 2)
  for (let i = 0; i < geometry.getAttribute('uv').count; i++) {
    uv1[i * 2] = 0.75
    uv1[i * 2 + 1] = 0.5
  }
  geometry.setAttribute('uv1', new THREE.BufferAttribute(uv1, 2))

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    geometry,
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

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  assert.ok(mean.r > 35, `specularMap.channel should sample uv1's enabled texel (${mean.r})`)
})

test('MeshPhongMaterial specularMap applies texture UV transforms', () => {
  const specularMap = rgbaTexture([
    0, 0, 0, 255,
    255, 0, 0, 255,
  ], 2, 1)
  specularMap.channel = 1
  specularMap.offset.set(0.5, 0)

  const geometry = constantUvPlane(0.75, 0.5)
  setConstantUvAttribute(geometry, 'uv1', 0.25, 0.5)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    geometry,
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

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  assert.ok(mean.r > 35, `specularMap offset should sample uv1's enabled texel (${mean.r})`)
})

test('MeshPhongMaterial specularMap honors nearest texture filters', () => {
  function renderWithFilter(filter) {
    const specularMap = rgbaTexture([
      0, 0, 0, 255,
      255, 0, 0, 255,
    ], 2, 1)
    specularMap.magFilter = filter
    specularMap.minFilter = filter

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.45, 0.5),
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

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const nearest = renderWithFilter(THREE.NearestFilter)
  const linear = renderWithFilter(THREE.LinearFilter)
  assert.ok(linear.r > nearest.r + 25, `LinearFilter should blend in the enabled specular texel (${linear.r} vs ${nearest.r})`)
})

test('material envMap reflection inputs fail clearly', () => {
  const envMap = Object.assign(solidTexture(255, 255, 255), {
    mapping: THREE.EquirectangularReflectionMapping,
  })
  const cases = [
    new THREE.MeshPhongMaterial({ color: 0xffffff, envMap }),
    new THREE.MeshBasicMaterial({ color: 0xffffff, envMap }),
  ]

  for (const material of cases) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(new THREE.SphereGeometry(1, 16, 16), material))

    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      /material\.envMap.*not supported/i,
      material.type,
    )
  }
})

test('MeshToonMaterial renders broad toon diffuse bands', () => {
  function renderMaterial(material) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(new THREE.SphereGeometry(1, 48, 24), material))

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(2, 0, 3)
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 96, height: 96 }))
  }

  const toon = renderMaterial(new THREE.MeshToonMaterial({ color: 0xffffff }))
  const lambert = renderMaterial(new THREE.MeshLambertMaterial({ color: 0xffffff }))
  assert.ok(toon.r > lambert.r + 8, `toon fallback should produce a broader lit band than Lambert (${toon.r} vs ${lambert.r})`)
})

test('MeshToonMaterial gradientMap controls toon diffuse ramp', () => {
  function renderGradientMap(gradientMap) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.SphereGeometry(1, 48, 24),
      new THREE.MeshToonMaterial({ color: 0xffffff, gradientMap }),
    ))

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(2, 0, 3)
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 96, height: 96 }))
  }

  const blackRamp = renderGradientMap(solidTexture(0, 0, 0))
  const whiteRamp = renderGradientMap(solidTexture(255, 255, 255))
  assert.ok(whiteRamp.r > blackRamp.r + 30, `white toon gradient ramp should brighten diffuse output (${whiteRamp.r} vs ${blackRamp.r})`)
})

test('MeshDepthMaterial renders nearer fragments brighter than farther fragments', () => {
  function renderDepthAt(z) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), new THREE.MeshDepthMaterial())
    mesh.position.z = z
    scene.add(mesh)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const near = renderDepthAt(2.85)
  const far = renderDepthAt(-6)
  assert.ok(near.r > far.r + 40, `near depth plane should be brighter than far plane (${near.r} vs ${far.r})`)
})

test('MeshDepthMaterial depthPacking encodes packed depth variants', () => {
  function packDepthToRG(v) {
    if (v <= 0) return [0, 0, 0, 255]
    if (v >= 1) return [255, 255, 0, 255]
    const vuf = Math.floor(v * 256)
    const gf = (v * 256) - vuf
    return [vuf, gf * 255, 0, 255]
  }

  function renderPackedDepth(depthPacking) {
    const z = 2.5
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshDepthMaterial({ depthPacking }))
    mesh.position.z = z
    scene.add(mesh)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    camera.updateMatrixWorld()
    camera.updateProjectionMatrix()

    const ndc = new THREE.Vector3(0, 0, z).project(camera)
    const fragDepth = ndc.z * 0.5 + 0.5
    const mean = meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 24, 24, 40, 40)
    return { fragDepth, mean }
  }

  function assertChannels(actual, expected, label, tolerance = 3) {
    for (const [channel, expectedValue] of [['r', expected[0]], ['g', expected[1]], ['b', expected[2]], ['a', expected[3]]]) {
      assert.ok(Math.abs(actual[channel] - expectedValue) <= tolerance, `${label}.${channel} expected ${expectedValue}, got ${actual[channel]}`)
    }
  }

  function assertPrefix(actual, expected, label) {
    assert.ok(Math.abs(actual.r - expected[0]) <= 8, `${label}.r expected ${expected[0]}, got ${actual.r}`)
    assert.ok(Math.abs(actual.g - expected[1]) <= 8, `${label}.g expected ${expected[1]}, got ${actual.g}`)
  }

  const basic = renderPackedDepth(THREE.BasicDepthPacking)
  assertChannels(basic.mean, [255 * (1 - basic.fragDepth), 255 * (1 - basic.fragDepth), 255 * (1 - basic.fragDepth), 255], 'basic')

  const rgba = renderPackedDepth(THREE.RGBADepthPacking)
  assertPrefix(rgba.mean, packDepthToRG(rgba.fragDepth), 'rgba')
  assert.ok(rgba.mean.b > 10, `rgba.b should carry packed lower depth bits, got ${rgba.mean.b}`)
  assert.ok(rgba.mean.a < 5, `rgba.a should carry the remaining packed depth bits for this depth, got ${rgba.mean.a}`)

  const rgb = renderPackedDepth(THREE.RGBDepthPacking)
  assertPrefix(rgb.mean, packDepthToRG(rgb.fragDepth), 'rgb')
  assert.ok(rgb.mean.b > 10, `rgb.b should carry packed lower depth bits, got ${rgb.mean.b}`)
  assert.ok(rgb.mean.a > 250, `rgb.a should remain opaque, got ${rgb.mean.a}`)

  const rg = renderPackedDepth(THREE.RGDepthPacking)
  assertChannels(rg.mean, packDepthToRG(rg.fragDepth), 'rg', 8)
})

test('MeshDepthMaterial wireframe renders triangle edges without filling faces', () => {
  function renderDepthWireframe(wireframe) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshDepthMaterial({ wireframe }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 8)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return renderRgba(scene, camera, { width: 64, height: 64 })
  }

  const solidRatio = nonBackgroundRatio(renderDepthWireframe(false), [0, 0, 0])
  const wireRatio = nonBackgroundRatio(renderDepthWireframe(true), [0, 0, 0])
  assert.ok(solidRatio > 0.4, `solid depth material should fill the plane (${solidRatio})`)
  assert.ok(wireRatio > 0.005, `wireframe depth material should draw visible edges (${wireRatio})`)
  assert.ok(wireRatio < solidRatio * 0.35, `wireframe depth material should not fill faces (${wireRatio} vs ${solidRatio})`)
})

test('displacementMap applies texture UV transforms before depth output', () => {
  function renderDisplaced(offsetX) {
    const displacementMap = rgbaTexture([
      0, 0, 0, 255,
      255, 255, 255, 255,
    ], 2, 1)
    displacementMap.offset.set(offsetX, 0)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.25, 0.5),
      new THREE.MeshDepthMaterial({
        displacementMap,
        displacementScale: 2.5,
        displacementBias: 0,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 20, 20, 44, 44)
  }

  const flat = renderDisplaced(0)
  const displaced = renderDisplaced(0.5)
  assert.ok(displaced.r > flat.r + 15, `displaced depth plane should move nearer and render brighter (${displaced.r} vs ${flat.r})`)
})

test('displacementMap samples the selected secondary UV channel before depth output', () => {
  function renderDisplaced(channel) {
    const displacementMap = rgbaTexture([
      0, 0, 0, 255,
      255, 255, 255, 255,
    ], 2, 1)
    displacementMap.channel = channel

    const geometry = constantUvPlane(0.25, 0.5)
    setConstantUvAttribute(geometry, 'uv1', 0.75, 0.5)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      geometry,
      new THREE.MeshDepthMaterial({
        displacementMap,
        displacementScale: 2.5,
        displacementBias: 0,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 20, 20, 44, 44)
  }

  const primary = renderDisplaced(0)
  const secondary = renderDisplaced(1)
  assert.ok(secondary.r > primary.r + 15, `displacementMap channel=1 should sample uv1's displaced texel (${secondary.r} vs ${primary.r})`)
})

test('MeshDistanceMaterial renders farther fragments with higher red distance', () => {
  function renderDistanceAt(z) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), new THREE.MeshDistanceMaterial())
    mesh.position.z = z
    scene.add(mesh)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 8)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const near = renderDistanceAt(2.8)
  const far = renderDistanceAt(-4)
  assert.ok(far.r > near.r + 60, `far distance plane should write a higher red distance (${far.r} vs ${near.r})`)
  assert.ok(far.g < 5 && far.b < 5, `distance material should write distance in red only (${far.g}, ${far.b})`)
})

test('MeshDistanceMaterial wireframe renders distance on triangle edges', () => {
  function renderDistanceWireframe(wireframe) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const material = new THREE.MeshDistanceMaterial()
    material.wireframe = wireframe
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 8)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return renderRgba(scene, camera, { width: 64, height: 64 })
  }

  const solidRatio = nonBackgroundRatio(renderDistanceWireframe(false), [0, 0, 0])
  const wireRatio = nonBackgroundRatio(renderDistanceWireframe(true), [0, 0, 0])
  assert.ok(solidRatio > 0.4, `solid distance material should fill the plane (${solidRatio})`)
  assert.ok(wireRatio > 0.005, `wireframe distance material should draw visible edges (${wireRatio})`)
  assert.ok(wireRatio < solidRatio * 0.35, `wireframe distance material should not fill faces (${wireRatio} vs ${solidRatio})`)
})

test('MeshDistanceMaterial honors referencePosition and distance range', () => {
  function renderDistanceAt(z) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const material = new THREE.MeshDistanceMaterial()
    material.referencePosition = new THREE.Vector3(0, 0, -4)
    material.nearDistance = 0
    material.farDistance = 7
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), material)
    mesh.position.z = z
    scene.add(mesh)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 8)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 20, 20, 44, 44)
  }

  const nearReference = renderDistanceAt(-4)
  const farReference = renderDistanceAt(2.8)
  assert.ok(
    farReference.r > nearReference.r + 100,
    `distance material should measure from referencePosition (${farReference.r} vs ${nearReference.r})`,
  )
})

test('SpriteMaterial renders texture maps and opacity as a camera-facing billboard', () => {
  function renderSprite(opacity) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
      map: solidTexture(0, 255, 0),
      color: 0xffffff,
      opacity,
      transparent: true,
    }))
    sprite.scale.set(2, 2, 1)
    scene.add(sprite)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const opaque = renderSprite(1)
  const translucent = renderSprite(0.35)
  assert.ok(opaque.g > opaque.r + 40, `sprite map should drive green output (${opaque.g} vs ${opaque.r})`)
  assert.ok(translucent.g > translucent.r + 5, `transparent sprite should still render green (${translucent.g} vs ${translucent.r})`)
  assert.ok(opaque.g > translucent.g + 20, `sprite opacity should reduce output intensity (${opaque.g} vs ${translucent.g})`)
})

test('SpriteMaterial honors sprite scale and material rotation', () => {
  function renderRotatedSprite(rotation) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const material = new THREE.SpriteMaterial({ color: 0xffffff, rotation })
    const sprite = new THREE.Sprite(material)
    sprite.scale.set(1.8, 0.45, 1)
    scene.add(sprite)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return renderRgba(scene, camera, { width: 96, height: 96 })
  }

  const horizontal = nonBackgroundBounds(renderRotatedSprite(0), 96, 96, [0, 0, 0])
  const vertical = nonBackgroundBounds(renderRotatedSprite(Math.PI / 2), 96, 96, [0, 0, 0])
  assert.ok(horizontal.width > horizontal.height * 2, `unrotated sprite should be wide (${horizontal.width}x${horizontal.height})`)
  assert.ok(vertical.height > vertical.width * 2, `rotated sprite should be tall (${vertical.width}x${vertical.height})`)
})

test('camera layers filter renderable objects', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const redOccluder = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  )
  redOccluder.position.z = 0.1
  scene.add(redOccluder)

  const greenVisible = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0x00ff00 }),
  )
  greenVisible.layers.set(1)
  scene.add(greenVisible)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  camera.layers.set(1)

  const rgba = renderRgba(scene, camera, { width: 64, height: 64 })
  const mean = meanRgba(rgba)
  assert.ok(mean.g > mean.r + 20, `layer 1 object should dominate over filtered layer 0 object (${mean.g} vs ${mean.r})`)
})

test('transparent renderOrder overrides traversal order', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const red = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xff0000, opacity: 0.55, transparent: true }),
  )
  red.renderOrder = 2
  scene.add(red)

  const blue = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0x0000ff, opacity: 0.55, transparent: true }),
  )
  blue.renderOrder = 1
  scene.add(blue)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 64, height: 64 })
  const mean = meanRgba(rgba)
  assert.ok(mean.r > mean.b + 10, `higher renderOrder red plane should render on top (${mean.r} vs ${mean.b})`)
})

test('Group renderOrder supplies groupOrder for transparent children', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const redGroup = new THREE.Group()
  redGroup.renderOrder = 2
  redGroup.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xff0000, opacity: 0.55, transparent: true }),
  ))
  scene.add(redGroup)

  const blueGroup = new THREE.Group()
  blueGroup.renderOrder = 1
  blueGroup.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0x0000ff, opacity: 0.55, transparent: true }),
  ))
  scene.add(blueGroup)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 64, height: 64 })
  const mean = meanRgba(rgba)
  assert.ok(mean.r > mean.b + 10, `higher groupOrder red plane should render on top (${mean.r} vs ${mean.b})`)
})

test('material depthTest=false renders over earlier depth', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const front = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  )
  front.renderOrder = 0
  scene.add(front)

  const behind = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0x0000ff, depthTest: false }),
  )
  behind.position.z = -0.2
  behind.renderOrder = 1
  scene.add(behind)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 24, 24, 40, 40)
  assert.ok(mean.b > mean.r + 80, `depthTest=false behind plane should draw over red (${mean.b} vs ${mean.r})`)
})

test('material depthWrite=false avoids occluding later depth-tested draws', () => {
  function renderFront(depthWrite) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)

    const front = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ color: 0xff0000, depthWrite }),
    )
    front.renderOrder = 0
    scene.add(front)

    const behind = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ color: 0x0000ff }),
    )
    behind.position.z = -0.2
    behind.renderOrder = 1
    scene.add(behind)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 24, 24, 40, 40)
  }

  const defaultWrite = renderFront(true)
  const disabledWrite = renderFront(false)
  assert.ok(defaultWrite.r > defaultWrite.b + 80, `default depthWrite should keep front red visible (${defaultWrite.r} vs ${defaultWrite.b})`)
  assert.ok(disabledWrite.b > disabledWrite.r + 80, `depthWrite=false should let later blue draw pass (${disabledWrite.b} vs ${disabledWrite.r})`)
})

test('material colorWrite=false writes depth without changing color', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const mask = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xff0000, colorWrite: false }),
  )
  mask.renderOrder = 0
  scene.add(mask)

  const behind = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0x0000ff }),
  )
  behind.position.z = -0.2
  behind.renderOrder = 1
  scene.add(behind)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 24, 24, 40, 40)
  assert.ok(mean.r < 5 && mean.g < 5 && mean.b < 5, `colorWrite=false depth mask should leave background visible (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('material polygonOffset applies depth bias', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const red = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  )
  red.renderOrder = 0
  scene.add(red)

  const blue = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({
      color: 0x0000ff,
      polygonOffset: true,
      polygonOffsetFactor: 0,
      polygonOffsetUnits: 1,
    }),
  )
  blue.renderOrder = 1
  scene.add(blue)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 24, 24, 40, 40)
  assert.ok(mean.r > mean.b + 80, `positive polygonOffset should bias the later blue plane behind red (${mean.r} vs ${mean.b})`)
})

test('material stencil state masks later draws', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const mask = new THREE.Mesh(
    new THREE.PlaneGeometry(1, 2),
    new THREE.MeshBasicMaterial({
      color: 0xff0000,
      colorWrite: false,
      depthWrite: false,
      stencilWrite: true,
      stencilFunc: THREE.AlwaysStencilFunc,
      stencilRef: 1,
      stencilZPass: THREE.ReplaceStencilOp,
    }),
  )
  mask.position.x = -0.5
  mask.renderOrder = 0
  scene.add(mask)

  const fill = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({
      color: 0x0000ff,
      stencilWrite: true,
      stencilFunc: THREE.EqualStencilFunc,
      stencilRef: 1,
      stencilFail: THREE.KeepStencilOp,
      stencilZFail: THREE.KeepStencilOp,
      stencilZPass: THREE.KeepStencilOp,
      stencilWriteMask: 0,
    }),
  )
  fill.renderOrder = 1
  scene.add(fill)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 64, height: 64 })
  const left = meanRegion(rgba, 64, 64, 16, 24, 28, 40)
  const right = meanRegion(rgba, 64, 64, 42, 24, 54, 40)
  assert.ok(left.b > 150, `stencil fill should render inside the mask (${left.b})`)
  assert.ok(right.b < 10, `stencil fill should be rejected outside the mask (${right.b})`)
})

test('NoBlending disables blending even for transparent materials', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 1)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({
      color: 0xff0000,
      opacity: 0.25,
      transparent: true,
      blending: THREE.NoBlending,
    }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 24, 24, 40, 40)
  assert.ok(mean.r > mean.b + 80, `NoBlending should overwrite destination color (${mean.r} vs ${mean.b})`)
})

test('material premultipliedAlpha premultiplies shader output before blending', () => {
  function renderNoBlending(premultipliedAlpha) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({
        color: 0xff0000,
        opacity: 0.5,
        transparent: true,
        blending: THREE.NoBlending,
        premultipliedAlpha,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 24, 24, 40, 40)
  }

  const straight = renderNoBlending(false)
  const premultiplied = renderNoBlending(true)
  assert.ok(straight.r > premultiplied.r + 60, `premultipliedAlpha should reduce raw RGB output (${straight.r} vs ${premultiplied.r})`)
  assert.ok(premultiplied.r > 60, `premultiplied output should retain source contribution (${premultiplied.r})`)
})

test('AdditiveBlending adds source color to destination', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const back = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0x0000ff }),
  )
  back.position.z = -0.1
  scene.add(back)
  const front = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({
      color: 0xff0000,
      opacity: 0.5,
      transparent: true,
      blending: THREE.AdditiveBlending,
    }),
  )
  front.position.z = 0.1
  scene.add(front)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 24, 24, 40, 40)
  assert.ok(mean.r > 60, `AdditiveBlending should add red source contribution (${mean.r})`)
  assert.ok(mean.b > 180, `AdditiveBlending should preserve bright blue destination (${mean.b})`)
})

test('premultipliedAlpha uses premultiplied additive blend factors', () => {
  function renderAdditive(premultipliedAlpha) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const back = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ color: 0x0000ff }),
    )
    back.position.z = -0.1
    scene.add(back)
    const front = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({
        color: 0xff0000,
        opacity: 0.5,
        transparent: true,
        blending: THREE.AdditiveBlending,
        premultipliedAlpha,
      }),
    )
    front.position.z = 0.1
    scene.add(front)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 24, 24, 40, 40)
  }

  const straight = renderAdditive(false)
  const premultiplied = renderAdditive(true)
  assert.ok(Math.abs(straight.r - premultiplied.r) < 20, `premultiplied additive red should match straight additive (${straight.r} vs ${premultiplied.r})`)
  assert.ok(premultiplied.b > 180, `premultiplied additive should preserve bright blue destination (${premultiplied.b})`)
})

test('CustomBlending honors custom factors and equation', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const back = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xffffff }),
  )
  back.position.z = -0.1
  scene.add(back)
  const front = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({
      color: 0xff0000,
      opacity: 0.5,
      transparent: true,
      blending: THREE.CustomBlending,
      blendEquation: THREE.ReverseSubtractEquation,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneFactor,
      blendEquationAlpha: THREE.AddEquation,
      blendSrcAlpha: THREE.OneFactor,
      blendDstAlpha: THREE.ZeroFactor,
    }),
  )
  front.position.z = 0.1
  scene.add(front)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 24, 24, 40, 40)
  assert.ok(mean.r < 20, `ReverseSubtractEquation should subtract red from the white destination (${mean.r})`)
  assert.ok(mean.g > 180, `ReverseSubtractEquation should preserve the green destination channel (${mean.g})`)
  assert.ok(mean.b > 180, `ReverseSubtractEquation should preserve the blue destination channel (${mean.b})`)
})

test('InstancedMesh expands instance matrices and colors', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const mesh = new THREE.InstancedMesh(
    new THREE.BoxGeometry(0.75, 0.75, 0.75),
    new THREE.MeshBasicMaterial({ color: 0xffffff }),
    2,
  )
  const matrix = new THREE.Matrix4()
  mesh.setMatrixAt(0, matrix.makeTranslation(-0.55, 0, 0))
  mesh.setMatrixAt(1, matrix.makeTranslation(0.55, 0, 0))
  mesh.setColorAt(0, new THREE.Color(1, 0, 0))
  mesh.setColorAt(1, new THREE.Color(0, 1, 0))
  scene.add(mesh)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 96, height: 96 })
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 8, `red instance should contribute to output mean (${mean.r})`)
  assert.ok(mean.g > 8, `green instance should contribute to output mean (${mean.g})`)
  assert.ok(mean.b < Math.max(mean.r, mean.g) * 0.5, `white material should be modulated by instanceColor (${mean.b})`)
})

test('InstancedBufferGeometry expands per-instance offsets and colors', () => {
  const base = new THREE.PlaneGeometry(0.85, 0.85)
  const geometry = new THREE.InstancedBufferGeometry()
  geometry.index = base.index
  geometry.setAttribute('position', base.getAttribute('position'))
  geometry.setAttribute('uv', base.getAttribute('uv'))
  geometry.instanceCount = 2
  geometry.setAttribute('instanceOffset', new THREE.InstancedBufferAttribute(
    new Float32Array([-0.55, 0, 0, 0.55, 0, 0]),
    3,
  ))
  geometry.setAttribute('color', new THREE.InstancedBufferAttribute(
    new Float32Array([1, 0, 0, 0, 1, 0]),
    3,
  ))

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: 0xffffff, vertexColors: true })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 96, height: 96 })
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 8, `red instanced attribute should contribute to output mean (${mean.r})`)
  assert.ok(mean.g > 8, `green instanced attribute should contribute to output mean (${mean.g})`)
  assert.ok(mean.b < Math.max(mean.r, mean.g) * 0.5, `instance colors should avoid blue contribution (${mean.b})`)
})

test('aoMap samples the selected secondary UV channel', () => {
  const aoMap = rgbaTexture([
    255, 255, 255, 255,
    255, 255, 255, 255,
    0, 0, 0, 255,
    0, 0, 0, 255,
  ], 4, 1)
  aoMap.channel = 1

  const geometry = new THREE.PlaneGeometry(2, 2)
  const primaryUv = new Float32Array(geometry.getAttribute('uv').count * 2)
  const secondaryUv = new Float32Array(geometry.getAttribute('uv').count * 2)
  for (let i = 0; i < geometry.getAttribute('uv').count; i++) {
    primaryUv[i * 2] = 0.125
    primaryUv[i * 2 + 1] = 0.5
    secondaryUv[i * 2] = 0.875
    secondaryUv[i * 2 + 1] = 0.5
  }
  geometry.setAttribute('uv', new THREE.BufferAttribute(primaryUv, 2))
  geometry.setAttribute('uv1', new THREE.BufferAttribute(secondaryUv, 2))

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    geometry,
    new THREE.MeshBasicMaterial({ color: 0xffffff, aoMap, aoMapIntensity: 1 }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 64, height: 64 })
  const mean = meanRgba(rgba)
  assert.ok(mean.r < 20, `aoMap channel=1 should darken the plane through uv1 (${mean.r})`)
})

test('aoMap applies texture UV transforms on the selected channel', () => {
  const aoMap = rgbaTexture([
    255, 255, 255, 255,
    0, 0, 0, 255,
  ], 2, 1)
  aoMap.channel = 1
  aoMap.offset.set(0.5, 0)

  const geometry = constantUvPlane(0.75, 0.5)
  setConstantUvAttribute(geometry, 'uv1', 0.25, 0.5)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    geometry,
    new THREE.MeshBasicMaterial({ color: 0xffffff, aoMap, aoMapIntensity: 1 }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  assert.ok(mean.r < 20, `aoMap offset should darken the plane through uv1 (${mean.r})`)
})

test('aoMap honors nearest texture filters', () => {
  function renderWithFilter(filter) {
    const aoMap = rgbaTexture([
      0, 0, 0, 255,
      255, 255, 255, 255,
    ], 2, 1)
    aoMap.magFilter = filter
    aoMap.minFilter = filter

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.45, 0.5),
      new THREE.MeshBasicMaterial({ color: 0xffffff, aoMap, aoMapIntensity: 1 }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const nearest = renderWithFilter(THREE.NearestFilter)
  const linear = renderWithFilter(THREE.LinearFilter)
  assert.ok(nearest.r < 20, `NearestFilter should choose the dark AO texel (${nearest.r})`)
  assert.ok(linear.r > nearest.r + 40, `LinearFilter should blend in the bright AO texel (${linear.r} vs ${nearest.r})`)
})

test('alphaMap green channel contributes to alpha testing', () => {
  const alphaMap = solidTexture(255, 0, 255, 255)
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 1, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({
      color: 0xff0000,
      alphaMap,
      alphaTest: 0.5,
    }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 64, height: 64 })
  const mean = meanRgba(rgba)
  assert.ok(mean.g > mean.r + 80, `green-channel alpha map should discard the red plane (${mean.g} vs ${mean.r})`)
})

test('alphaMap applies texture UV transforms before alpha testing', () => {
  const alphaMap = rgbaTexture([
    255, 0, 0, 255,
    255, 255, 0, 255,
  ], 2, 1)
  alphaMap.offset.set(0.5, 0)

  const geometry = constantUvPlane(0.25, 0.5)
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 1)
  scene.add(new THREE.Mesh(
    geometry,
    new THREE.MeshBasicMaterial({
      color: 0xff0000,
      alphaMap,
      alphaTest: 0.5,
    }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  assert.ok(mean.r > mean.b + 40, `alphaMap offset should sample the visible texel before alpha testing (${mean.r} vs ${mean.b})`)
})

test('alphaMap samples the selected secondary UV channel', () => {
  function renderAlphaChannel(channel) {
    const alphaMap = rgbaTexture([
      255, 0, 255, 255,
      255, 255, 255, 255,
    ], 2, 1)
    alphaMap.channel = channel

    const geometry = constantUvPlane(0.25, 0.5)
    setConstantUvAttribute(geometry, 'uv1', 0.75, 0.5)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 1)
    scene.add(new THREE.Mesh(
      geometry,
      new THREE.MeshBasicMaterial({
        color: 0xff0000,
        alphaMap,
        alphaTest: 0.5,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const primary = renderAlphaChannel(0)
  const secondary = renderAlphaChannel(1)
  assert.ok(primary.b > primary.r + 80, `alphaMap channel=0 should sample the transparent primary UV texel (${primary.b} vs ${primary.r})`)
  assert.ok(secondary.r > secondary.b + 40, `alphaMap channel=1 should sample the opaque uv1 texel (${secondary.r} vs ${secondary.b})`)
})

test('alphaMap honors nearest texture filters before alpha testing', () => {
  function renderWithFilter(filter) {
    const alphaMap = rgbaTexture([
      255, 0, 0, 255,
      255, 255, 0, 255,
    ], 2, 1)
    alphaMap.magFilter = filter
    alphaMap.minFilter = filter

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 1)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.45, 0.5),
      new THREE.MeshBasicMaterial({
        color: 0xff0000,
        alphaMap,
        alphaTest: 0.2,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const nearest = renderWithFilter(THREE.NearestFilter)
  const linear = renderWithFilter(THREE.LinearFilter)
  assert.ok(nearest.b > nearest.r + 80, `NearestFilter should choose the transparent alpha texel (${nearest.b} vs ${nearest.r})`)
  assert.ok(linear.r > linear.b + 40, `LinearFilter should blend enough green-channel alpha to pass the test (${linear.r} vs ${linear.b})`)
})

test('material alphaHash produces stochastic coverage without transparent blending', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 1, 0)
  const front = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({
      color: 0xff0000,
      opacity: 0.5,
      alphaHash: true,
    }),
  )
  front.position.z = 0.1
  scene.add(front)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 64, height: 64 })
  const redPixels = countRegionPixels(rgba, 64, 64, 24, 24, 40, 40, (r, g, b) => r > 160 && g < 60 && b < 60)
  const greenPixels = countRegionPixels(rgba, 64, 64, 24, 24, 40, 40, (r, g, b) => g > 160 && r < 60 && b < 60)
  assert.ok(redPixels > 40, `alphaHash should leave red covered pixels (${redPixels})`)
  assert.ok(greenPixels > 120, `alphaHash should reveal green pixels through hashed discards (${greenPixels})`)
})

test('material alphaToCoverage fails clearly', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xff0000, alphaToCoverage: true }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  assert.throws(
    () => renderRgba(scene, camera, { width: 64, height: 64 }),
    /alphaToCoverage.*not supported/i,
  )
})

test('material clippingPlanes discard the negative plane side', () => {
  const material = new THREE.MeshBasicMaterial({ color: 0xff0000 })
  material.clippingPlanes = [new THREE.Plane(new THREE.Vector3(1, 0, 0), 0)]

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 1)
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 64, height: 64 })
  const clippedLeft = meanRegion(rgba, 64, 64, 12, 22, 24, 42)
  const visibleRight = meanRegion(rgba, 64, 64, 40, 22, 52, 42)

  assert.ok(clippedLeft.b > clippedLeft.r + 80, `left side should reveal blue background (${clippedLeft.b} vs ${clippedLeft.r})`)
  assert.ok(visibleRight.r > visibleRight.b + 80, `right side should keep the red plane (${visibleRight.r} vs ${visibleRight.b})`)
})

test('render option clippingPlanes apply as global union planes', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 1)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0x00ff00 }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, {
    width: 64,
    height: 64,
    clippingPlanes: [new THREE.Plane(new THREE.Vector3(0, 1, 0), 0)],
  })
  const visibleTop = meanRegion(rgba, 64, 64, 22, 12, 42, 24)
  const clippedBottom = meanRegion(rgba, 64, 64, 22, 40, 42, 52)

  assert.ok(visibleTop.g > visibleTop.b + 80, `top side should keep the green plane (${visibleTop.g} vs ${visibleTop.b})`)
  assert.ok(clippedBottom.b > clippedBottom.g + 80, `bottom side should reveal blue background (${clippedBottom.b} vs ${clippedBottom.g})`)
})

test('clipIntersection requires all local clipping planes to reject a fragment', () => {
  const material = new THREE.MeshBasicMaterial({ color: 0xff0000 })
  material.clippingPlanes = [
    new THREE.Plane(new THREE.Vector3(1, 0, 0), 0),
    new THREE.Plane(new THREE.Vector3(0, 1, 0), 0),
  ]
  material.clipIntersection = true

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 1)
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 64, height: 64 })
  const visibleTopLeft = meanRegion(rgba, 64, 64, 12, 12, 24, 24)
  const clippedBottomLeft = meanRegion(rgba, 64, 64, 12, 40, 24, 52)
  const visibleBottomRight = meanRegion(rgba, 64, 64, 40, 40, 52, 52)

  assert.ok(visibleTopLeft.r > visibleTopLeft.b + 80, `top-left should remain visible with intersection clipping (${visibleTopLeft.r} vs ${visibleTopLeft.b})`)
  assert.ok(clippedBottomLeft.b > clippedBottomLeft.r + 80, `bottom-left should be clipped by both planes (${clippedBottomLeft.b} vs ${clippedBottomLeft.r})`)
  assert.ok(visibleBottomRight.r > visibleBottomRight.b + 80, `bottom-right should remain visible with intersection clipping (${visibleBottomRight.r} vs ${visibleBottomRight.b})`)
})

test('material clipShadows fails clearly', () => {
  const material = new THREE.MeshBasicMaterial({ color: 0xff0000 })
  material.clipShadows = true
  material.clippingPlanes = [new THREE.Plane(new THREE.Vector3(1, 0, 0), 0)]

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material)
  mesh.castShadow = true
  scene.add(mesh)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  assert.throws(
    () => renderRgba(scene, camera, { width: 64, height: 64 }),
    /clipShadows.*not supported/i,
  )
})

test('base color map applies texture UV transforms', () => {
  const map = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  map.offset.set(0.5, 0)

  const geometry = new THREE.PlaneGeometry(2, 2)
  const uv = new Float32Array(geometry.getAttribute('uv').count * 2)
  for (let i = 0; i < geometry.getAttribute('uv').count; i++) {
    uv[i * 2] = 0.25
    uv[i * 2 + 1] = 0.5
  }
  geometry.setAttribute('uv', new THREE.BufferAttribute(uv, 2))

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ map })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  assert.ok(mean.g > mean.r + 40, `texture offset should shift the sampled texel from red to green (${mean.g} vs ${mean.r})`)
})

test('base color map samples the selected secondary UV channel', () => {
  function renderWithChannel(channel) {
    const map = rgbaTexture([
      0, 255, 0, 255,
      255, 0, 0, 255,
    ], 2, 1)
    map.channel = channel

    const geometry = constantUvPlane(0.25, 0.5)
    setConstantUvAttribute(geometry, 'uv1', 0.75, 0.5)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ map })))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const primary = renderWithChannel(0)
  const secondary = renderWithChannel(1)
  assert.ok(primary.g > primary.r + 40, `map channel=0 should sample the primary UV green texel (${primary.g} vs ${primary.r})`)
  assert.ok(secondary.r > secondary.g + 40, `map channel=1 should sample the uv1 red texel (${secondary.r} vs ${secondary.g})`)
})

test('emissiveMap applies texture UV transforms', () => {
  const emissiveMap = rgbaTexture([
    0, 0, 0, 255,
    255, 0, 0, 255,
  ], 2, 1)
  emissiveMap.offset.set(0.5, 0)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    constantUvPlane(0.25, 0.5),
    new THREE.MeshStandardMaterial({
      color: 0x000000,
      emissive: 0xff0000,
      emissiveMap,
    }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  assert.ok(mean.r > mean.g + 40, `emissiveMap offset should sample the red texel (${mean.r} vs ${mean.g})`)
})

test('emissiveMap samples the selected secondary UV channel', () => {
  function renderWithChannel(channel) {
    const emissiveMap = rgbaTexture([
      0, 255, 0, 255,
      255, 0, 0, 255,
    ], 2, 1)
    emissiveMap.channel = channel

    const geometry = constantUvPlane(0.25, 0.5)
    setConstantUvAttribute(geometry, 'uv1', 0.75, 0.5)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      geometry,
      new THREE.MeshStandardMaterial({
        color: 0x000000,
        emissive: 0xffffff,
        emissiveMap,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const primary = renderWithChannel(0)
  const secondary = renderWithChannel(1)

  assert.ok(primary.g > primary.r + 40, `emissiveMap channel=0 should sample the primary UV green texel (${primary.g} vs ${primary.r})`)
  assert.ok(secondary.r > secondary.g + 40, `emissiveMap channel=1 should sample the uv1 red texel (${secondary.r} vs ${secondary.g})`)
})

test('emissiveMap honors nearest texture filters', () => {
  function renderWithFilter(filter) {
    const emissiveMap = rgbaTexture([
      255, 0, 0, 255,
      0, 255, 0, 255,
    ], 2, 1)
    emissiveMap.magFilter = filter
    emissiveMap.minFilter = filter

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.45, 0.5),
      new THREE.MeshStandardMaterial({
        color: 0x000000,
        emissive: 0xffffff,
        emissiveMap,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const nearest = renderWithFilter(THREE.NearestFilter)
  const linear = renderWithFilter(THREE.LinearFilter)
  assert.ok(nearest.r > nearest.g + 80, `NearestFilter should choose the red emissive texel (${nearest.r} vs ${nearest.g})`)
  assert.ok(linear.g > nearest.g + 40, `LinearFilter should blend in the green emissive texel (${linear.g} vs ${nearest.g})`)
  assert.ok(nearest.r > linear.r + 20, `NearestFilter should preserve a stronger red emissive texel (${nearest.r} vs ${linear.r})`)
})

test('metallicRoughness maps apply texture UV transforms', () => {
  function renderWithOffset(offsetX) {
    const roughnessMap = rgbaTexture([
      0, 255, 0, 255,
      0, 0, 0, 255,
    ], 2, 1)
    roughnessMap.offset.set(offsetX, 0)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.25, 0.5),
      new THREE.MeshStandardMaterial({
        color: 0x000000,
        roughness: 1,
        metalness: 0,
        roughnessMap,
      }),
    ))

    const light = new THREE.DirectionalLight(0xffffff, 12)
    light.position.set(0, 0, 3)
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return renderRgba(scene, camera, { width: 64, height: 64 })
  }

  const rough = maxLuminance(renderWithOffset(0))
  const smooth = maxLuminance(renderWithOffset(0.5))
  assert.ok(Math.abs(smooth - rough) > 20, `roughnessMap offset should change the sampled texel (${smooth} vs ${rough})`)
})

test('metallicRoughness maps sample the selected secondary UV channel', () => {
  function renderWithChannel(channel) {
    const roughnessMap = rgbaTexture([
      0, 255, 0, 255,
      0, 0, 0, 255,
    ], 2, 1)
    roughnessMap.channel = channel

    const geometry = constantUvPlane(0.25, 0.5)
    setConstantUvAttribute(geometry, 'uv1', 0.75, 0.5)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      geometry,
      new THREE.MeshStandardMaterial({
        color: 0x000000,
        roughness: 1,
        metalness: 0,
        roughnessMap,
      }),
    ))

    const light = new THREE.DirectionalLight(0xffffff, 12)
    light.position.set(0, 0, 3)
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return maxLuminance(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const primary = renderWithChannel(0)
  const secondary = renderWithChannel(1)
  assert.ok(Math.abs(secondary - primary) > 20, `roughnessMap channel=1 should sample uv1's different texel (${secondary} vs ${primary})`)
})

test('base color maps honor texture flipY', () => {
  const data = [
    255, 0, 0, 255,
    255, 0, 0, 255,
    0, 255, 0, 255,
    0, 255, 0, 255,
  ]

  function renderFlipY(flipY) {
    const map = rgbaTexture(data, 2, 2)
    map.flipY = flipY

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(constantUvPlane(0.25, 0.25), new THREE.MeshBasicMaterial({ map })))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 22, 22, 42, 42)
  }

  const unflipped = renderFlipY(false)
  const flipped = renderFlipY(true)
  assert.ok(unflipped.r > unflipped.g + 40, `flipY=false should sample the first texture row as red (${unflipped.r} vs ${unflipped.g})`)
  assert.ok(flipped.g > flipped.r + 40, `flipY=true should sample the opposite texture row as green (${flipped.g} vs ${flipped.r})`)
})

test('base color maps honor nearest texture filters', () => {
  function renderWithFilter(filter) {
    const map = rgbaTexture([
      255, 0, 0, 255,
      0, 255, 0, 255,
    ], 2, 1)
    map.magFilter = filter
    map.minFilter = filter

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(constantUvPlane(0.45, 0.5), new THREE.MeshBasicMaterial({ map })))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const nearest = renderWithFilter(THREE.NearestFilter)
  const linear = renderWithFilter(THREE.LinearFilter)
  assert.ok(nearest.r > nearest.g + 80, `NearestFilter should choose the red texel (${nearest.r} vs ${nearest.g})`)
  assert.ok(linear.g > nearest.g + 40, `LinearFilter should blend in the green texel (${linear.g} vs ${nearest.g})`)
  assert.ok(nearest.r > linear.r + 20, `NearestFilter should preserve a stronger red texel (${nearest.r} vs ${linear.r})`)
})

test('compressed texture inputs fail with a clear pre-decode error', () => {
  const compressedTexture = {
    isTexture: true,
    isCompressedTexture: true,
    image: { width: 4, height: 4 },
    mipmaps: [{ data: new Uint8Array(16), width: 4, height: 4 }],
  }

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const material = new THREE.MeshBasicMaterial({ color: 0xffffff })
  material.map = compressedTexture
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  assert.throws(
    () => renderRgba(scene, camera, { width: 64, height: 64 }),
    /compressed texture.*pre-decode/i,
  )
})

test('base color maps decode sRGB colorSpace before shading', () => {
  function renderColorSpace(colorSpace) {
    const map = solidTexture(128, 128, 128)
    map.colorSpace = colorSpace

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map })))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const srgb = renderColorSpace(THREE.SRGBColorSpace)
  const linear = renderColorSpace(THREE.LinearSRGBColorSpace)
  assert.ok(linear.r > srgb.r + 10, `linear texture should render brighter than decoded sRGB texture (${linear.r} vs ${srgb.r})`)
})

test('outputColorSpace controls material and texture background output conversion', () => {
  function renderMaterialOutput(outputColorSpace) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(4, 4),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(0.5, 0.5, 0.5) }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64, outputColorSpace }))
  }

  function renderBackgroundOutput(outputColorSpace) {
    const scene = new THREE.Scene()
    scene.background = solidTexture(128, 128, 128)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64, outputColorSpace }))
  }

  const srgbMaterial = renderMaterialOutput(THREE.SRGBColorSpace)
  const linearMaterial = renderMaterialOutput(THREE.LinearSRGBColorSpace)
  assert.ok(
    srgbMaterial.r > linearMaterial.r + 20,
    `sRGB material output should apply display conversion (${srgbMaterial.r} vs ${linearMaterial.r})`,
  )

  const srgbBackground = renderBackgroundOutput(THREE.SRGBColorSpace)
  const linearBackground = renderBackgroundOutput(THREE.LinearSRGBColorSpace)
  assert.ok(
    srgbBackground.r > linearBackground.r + 40,
    `sRGB background output should apply display conversion (${srgbBackground.r} vs ${linearBackground.r})`,
  )
})

test('emissiveMap decodes sRGB colorSpace before shading', () => {
  function renderColorSpace(colorSpace) {
    const emissiveMap = solidTexture(128, 128, 128)
    emissiveMap.colorSpace = colorSpace

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshStandardMaterial({
        color: 0x000000,
        emissive: 0xffffff,
        emissiveMap,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const srgb = renderColorSpace(THREE.SRGBColorSpace)
  const linear = renderColorSpace(THREE.LinearSRGBColorSpace)
  assert.ok(linear.r > srgb.r + 10, `linear emissiveMap should render brighter than decoded sRGB texture (${linear.r} vs ${srgb.r})`)
})

test('lightMap decodes sRGB colorSpace before shading', () => {
  function renderColorSpace(colorSpace) {
    const lightMap = solidTexture(128, 128, 128)
    lightMap.colorSpace = colorSpace

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({
      color: 0xffffff,
      lightMap,
      lightMapIntensity: 4,
    })))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const srgb = renderColorSpace(THREE.SRGBColorSpace)
  const linear = renderColorSpace(THREE.LinearSRGBColorSpace)
  assert.ok(linear.r > srgb.r + 10, `linear lightMap should render brighter than decoded sRGB texture (${linear.r} vs ${srgb.r})`)
})

test('lightMap samples the selected secondary UV channel', () => {
  const lightMap = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  lightMap.channel = 1

  const geometry = constantUvPlane(0.25, 0.5)
  const uv1 = new Float32Array(geometry.getAttribute('uv').count * 2)
  for (let i = 0; i < geometry.getAttribute('uv').count; i++) {
    uv1[i * 2] = 0.75
    uv1[i * 2 + 1] = 0.5
  }
  geometry.setAttribute('uv1', new THREE.BufferAttribute(uv1, 2))

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
    color: 0xffffff,
    lightMap,
    lightMapIntensity: 3,
  })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  assert.ok(mean.g > mean.r + 40, `lightMap.channel should sample uv1 green texel, got ${mean.g} vs ${mean.r}`)
})

test('lightMap applies texture UV transforms on the selected channel', () => {
  const lightMap = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  lightMap.channel = 1
  lightMap.offset.set(0.5, 0)

  const geometry = constantUvPlane(0.75, 0.5)
  setConstantUvAttribute(geometry, 'uv1', 0.25, 0.5)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
    color: 0xffffff,
    lightMap,
    lightMapIntensity: 3,
  })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  assert.ok(mean.g > mean.r + 40, `lightMap offset should sample uv1 green texel, got ${mean.g} vs ${mean.r}`)
})

test('lightMap honors nearest texture filters', () => {
  function renderWithFilter(filter) {
    const lightMap = rgbaTexture([
      255, 0, 0, 255,
      0, 255, 0, 255,
    ], 2, 1)
    lightMap.magFilter = filter
    lightMap.minFilter = filter

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.45, 0.5),
      new THREE.MeshBasicMaterial({
        color: 0xffffff,
        lightMap,
        lightMapIntensity: 8,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const nearest = renderWithFilter(THREE.NearestFilter)
  const linear = renderWithFilter(THREE.LinearFilter)
  assert.ok(nearest.r > nearest.g + 40, `NearestFilter should choose the red light-map texel (${nearest.r} vs ${nearest.g})`)
  assert.ok(linear.g > nearest.g + 30, `LinearFilter should blend in the green light-map texel (${linear.g} vs ${nearest.g})`)
})

test('LightProbe spherical harmonics contribute diffuse lighting', () => {
  const probe = new THREE.LightProbe(undefined, 1.5)
  for (const coefficient of probe.sh.coefficients) {
    coefficient.set(0, 0, 0)
  }
  probe.sh.coefficients[0].set(1, 0, 0)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(probe)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, metalness: 0 }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  assert.ok(mean.r > mean.g + 40, `LightProbe should tint diffuse lighting red (${mean.r} vs ${mean.g})`)
  assert.ok(mean.r > mean.b + 40, `LightProbe should tint diffuse lighting red (${mean.r} vs ${mean.b})`)
})

test('RectAreaLight approximates finite one-sided area lighting', () => {
  function renderRectArea(width, height, targetZ) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, metalness: 0 }),
    ))

    const light = new THREE.RectAreaLight(0xffffff, 20, width, height)
    light.position.set(0, 0, 2)
    light.lookAt(0, 0, targetZ)
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return maxLuminance(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const smallForward = renderRectArea(0.5, 0.5, 0)
  const largeForward = renderRectArea(3, 3, 0)
  const backward = renderRectArea(3, 3, 4)

  assert.ok(smallForward > backward + 10, `forward RectAreaLight should illuminate its front side (${smallForward} vs ${backward})`)
  assert.ok(largeForward > smallForward + 10, `larger RectAreaLight should contribute more radiance (${largeForward} vs ${smallForward})`)
})

test('LOD selects object level from active camera distance', () => {
  const lod = new THREE.LOD()
  lod.addLevel(
    new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color: 0xff0000 })),
    0,
  )
  lod.addLevel(
    new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color: 0x0000ff })),
    4,
  )

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(lod)

  const nearCamera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  nearCamera.position.set(0, 0, 3)
  nearCamera.lookAt(0, 0, 0)

  const farCamera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  farCamera.position.set(0, 0, 6)
  farCamera.lookAt(0, 0, 0)

  const near = meanRgba(renderRgba(scene, nearCamera, { width: 64, height: 64 }))
  const far = meanRgba(renderRgba(scene, farCamera, { width: 64, height: 64 }))

  assert.ok(near.r > near.b + 10, `near LOD should render the red level (${near.r} vs ${near.b})`)
  assert.ok(far.b > far.r + 5, `far LOD should render the blue level (${far.b} vs ${far.r})`)
})

test('Fog and FogExp2 affect material output', () => {
  function renderFogged(fog) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.fog = fog
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ color: 0xff0000 }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const linear = renderFogged(new THREE.Fog(0x00ff00, 0, 1))
  assert.ok(linear.g > linear.r + 40, `linear fog should mix the red plane toward green (${linear.g} vs ${linear.r})`)

  const exp2 = renderFogged(new THREE.FogExp2(0x0000ff, 1.0))
  assert.ok(exp2.b > exp2.r + 40, `FogExp2 should mix the red plane toward blue (${exp2.b} vs ${exp2.r})`)
})

test('PBR scene with lights renders and shows lighting variation', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.05, 0.05, 0.05)
  addLights(scene)
  scene.add(
    new THREE.Mesh(
      new THREE.SphereGeometry(1, 32, 32),
      new THREE.MeshStandardMaterial({ color: 0xdddddd, metalness: 0.1, roughness: 0.4 }),
    ),
  )

  const rgba = renderRgba(scene, makeCamera())
  const ratio = nonBackgroundRatio(rgba, [13, 13, 13])
  assert.ok(ratio > 0.05, 'sphere should be visible')

  // Sample both sides of the sphere — the lit side should be brighter than the shadowed side.
  // Top-right quadrant vs bottom-left quadrant of the image.
  let litSum = 0
  let litCount = 0
  let darkSum = 0
  let darkCount = 0
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const i = (y * SIZE + x) * 4
      const lum = rgba[i] + rgba[i + 1] + rgba[i + 2]
      if (lum < 50) continue // skip background
      if (x > SIZE * 0.6 && y < SIZE * 0.4) {
        litSum += lum
        litCount++
      } else if (x < SIZE * 0.4 && y > SIZE * 0.6) {
        darkSum += lum
        darkCount++
      }
    }
  }
  if (litCount > 0 && darkCount > 0) {
    const litAvg = litSum / litCount
    const darkAvg = darkSum / darkCount
    assert.ok(litAvg > darkAvg, `lit side (${litAvg.toFixed(1)}) should be brighter than shadowed side (${darkAvg.toFixed(1)})`)
  }
})

test('MeshPhysicalMaterial extensions and maps affect rendered output', () => {
  const camera = makeCamera()

  function makeScene(material) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0.04, 0.04, 0.045)
    scene.environment = makeEnvironmentTexture()
    scene.environmentIntensity = 0.8
    addLights(scene)
    scene.add(new THREE.Mesh(new THREE.SphereGeometry(1, 32, 32), material))
    return scene
  }

  const base = renderRgba(
    makeScene(new THREE.MeshPhysicalMaterial({
      color: 0x7aa7ff,
      roughness: 0.35,
      metalness: 0.0,
    })),
    camera,
  )
  const physical = renderRgba(
    makeScene(new THREE.MeshPhysicalMaterial({
      color: 0x7aa7ff,
      roughness: 0.35,
      metalness: 0.0,
      clearcoat: 1.0,
      clearcoatMap: solidTexture(255, 0, 0),
      clearcoatRoughness: 0.04,
      clearcoatRoughnessMap: solidTexture(0, 96, 0),
      clearcoatNormalMap: solidTexture(128, 180, 240),
      clearcoatNormalScale: new THREE.Vector2(0.6, 0.4),
      sheen: 0.8,
      sheenColor: new THREE.Color(1.0, 0.25, 0.12),
      sheenColorMap: solidTexture(255, 128, 96),
      sheenRoughness: 0.35,
      sheenRoughnessMap: solidTexture(0, 0, 0, 160),
      anisotropy: 0.85,
      anisotropyRotation: Math.PI / 4,
      anisotropyMap: solidTexture(255, 128, 255),
      transmission: 0.25,
      transmissionMap: solidTexture(180, 0, 0),
      ior: 1.45,
      thickness: 0.4,
      thicknessMap: solidTexture(0, 255, 0),
      attenuationColor: new THREE.Color(0.8, 0.95, 1.0),
      attenuationDistance: 1.5,
    })),
    camera,
  )

  const ratio = nonBackgroundRatio(physical, [10, 10, 11])
  assert.ok(ratio > 0.05, 'physical material sphere should be visible')
  const diff = meanAbsDiff(base, physical)
  assert.ok(diff > 0.5, `expected physical extensions to change output, mean abs diff=${diff.toFixed(3)}`)
})

test('MeshPhysicalMaterial specular intensity and color affect direct specular', () => {
  function renderMaterial(material) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(new THREE.SphereGeometry(1, 48, 24), material))

    const light = new THREE.DirectionalLight(0xffffff, 8)
    light.position.set(0, 0, 3)
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return renderRgba(scene, camera, { width: 64, height: 64 })
  }

  const disabled = renderMaterial(new THREE.MeshPhysicalMaterial({
    color: 0x000000,
    roughness: 0.08,
    metalness: 0,
    specularIntensity: 0,
  }))
  const enabled = renderMaterial(new THREE.MeshPhysicalMaterial({
    color: 0x000000,
    roughness: 0.08,
    metalness: 0,
    specularIntensity: 1,
  }))
  assert.ok(maxLuminance(enabled) > maxLuminance(disabled) + 20, 'specularIntensity should control the direct specular highlight')

  const red = meanRgba(renderMaterial(new THREE.MeshPhysicalMaterial({
    color: 0x000000,
    roughness: 0.08,
    metalness: 0,
    specularIntensity: 1,
    specularColor: new THREE.Color(1, 0, 0),
  })))
  const green = meanRgba(renderMaterial(new THREE.MeshPhysicalMaterial({
    color: 0x000000,
    roughness: 0.08,
    metalness: 0,
    specularIntensity: 1,
    specularColor: new THREE.Color(0, 1, 0),
  })))
  assert.ok(red.r > red.g + 0.1, `red specularColor should tint the highlight red (${red.r} vs ${red.g})`)
  assert.ok(green.g > green.r + 0.1, `green specularColor should tint the highlight green (${green.g} vs ${green.r})`)
})

test('MeshPhysicalMaterial iridescence and dispersion fail clearly', () => {
  const cases = []

  const iridescence = new THREE.MeshPhysicalMaterial({ color: 0xffffff })
  iridescence.iridescence = 0.5
  cases.push([iridescence, /iridescence.*not supported/i, 'iridescence'])

  const iridescenceMap = new THREE.MeshPhysicalMaterial({ color: 0xffffff })
  iridescenceMap.iridescenceMap = solidTexture(255, 255, 255)
  cases.push([iridescenceMap, /iridescence.*not supported/i, 'iridescenceMap'])

  const iridescenceThicknessMap = new THREE.MeshPhysicalMaterial({ color: 0xffffff })
  iridescenceThicknessMap.iridescenceThicknessMap = solidTexture(255, 255, 255)
  cases.push([iridescenceThicknessMap, /iridescence.*not supported/i, 'iridescenceThicknessMap'])

  const dispersion = new THREE.MeshPhysicalMaterial({ color: 0xffffff })
  dispersion.dispersion = 0.25
  cases.push([dispersion, /dispersion.*not supported/i, 'dispersion'])

  for (const [material, pattern, label] of cases) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(new THREE.SphereGeometry(1, 16, 16), material))

    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      pattern,
      label,
    )
  }
})

test('physical extension maps apply texture UV transforms', () => {
  const transmissionMap = rgbaTexture([
    0, 0, 0, 255,
    255, 0, 0, 255,
  ], 2, 1)
  transmissionMap.offset.set(0.5, 0)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const back = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0x0000ff }),
  )
  back.position.z = -0.2
  scene.add(back)
  scene.add(new THREE.Mesh(
    constantUvPlane(0.25, 0.5),
    new THREE.MeshPhysicalMaterial({
      color: 0xff0000,
      roughness: 0.1,
      metalness: 0,
      transmission: 1,
      transmissionMap,
      ior: 1.5,
      thickness: 0,
    }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  assert.ok(mean.b > mean.r + 40, `transmissionMap offset should sample the transmitting texel (${mean.b} vs ${mean.r})`)
})

test('physical extension maps honor nearest texture filters', () => {
  function filteredTexture(data, filter) {
    const texture = rgbaTexture(data, 2, 1)
    texture.magFilter = filter
    texture.minFilter = filter
    return texture
  }

  function renderClearcoat(filter) {
    const clearcoatMap = filteredTexture([
      0, 0, 0, 255,
      255, 0, 0, 255,
    ], filter)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.environment = makeEnvironmentTexture()
    scene.environmentIntensity = 2
    scene.add(new THREE.Mesh(
      constantUvPlane(0.45, 0.5),
      new THREE.MeshPhysicalMaterial({
        color: 0x000000,
        roughness: 1,
        metalness: 0,
        clearcoat: 1,
        clearcoatRoughness: 0.04,
        clearcoatMap,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return maxLuminance(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  function renderSheen(filter) {
    const sheenColorMap = filteredTexture([
      0, 0, 0, 255,
      255, 0, 0, 255,
    ], filter)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.environment = makeEnvironmentTexture()
    scene.environmentIntensity = 2
    scene.add(new THREE.Mesh(
      constantUvPlane(0.45, 0.5),
      new THREE.MeshPhysicalMaterial({
        color: 0x000000,
        roughness: 1,
        metalness: 0,
        sheen: 1,
        sheenColor: new THREE.Color(1, 1, 1),
        sheenRoughness: 0.35,
        sheenColorMap,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  function renderSpecularIntensity(filter) {
    const specularIntensityMap = filteredTexture([
      0, 0, 0, 0,
      0, 0, 0, 255,
    ], filter)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.45, 0.5),
      new THREE.MeshPhysicalMaterial({
        color: 0x000000,
        roughness: 0.05,
        metalness: 0,
        specularIntensity: 1,
        specularIntensityMap,
      }),
    ))
    const light = new THREE.PointLight(0xffffff, 450)
    light.position.set(0, 0, 2)
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return maxLuminance(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  function renderTransmission(filter) {
    const transmissionMap = filteredTexture([
      0, 0, 0, 255,
      255, 0, 0, 255,
    ], filter)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const back = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ color: 0x0000ff }),
    )
    back.position.z = -0.2
    scene.add(back)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.45, 0.5),
      new THREE.MeshPhysicalMaterial({
        color: 0xff0000,
        roughness: 0.1,
        metalness: 0,
        transmission: 1,
        transmissionMap,
        ior: 1.5,
        thickness: 0,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  function renderClearcoatNormal(filter) {
    const clearcoatNormalMap = filteredTexture([
      128, 128, 255, 255,
      255, 128, 128, 255,
    ], filter)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.environment = makeEnvironmentTexture()
    scene.environmentIntensity = 2
    scene.add(new THREE.Mesh(
      constantUvPlane(0.45, 0.5),
      new THREE.MeshPhysicalMaterial({
        color: 0x000000,
        roughness: 1,
        metalness: 0,
        clearcoat: 1,
        clearcoatRoughness: 0.04,
        clearcoatNormalMap,
        clearcoatNormalScale: new THREE.Vector2(1, 1),
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return renderRgba(scene, camera, { width: 64, height: 64 })
  }

  const nearestClearcoat = renderClearcoat(THREE.NearestFilter)
  const linearClearcoat = renderClearcoat(THREE.LinearFilter)
  assert.ok(linearClearcoat > nearestClearcoat + 25, `LinearFilter should blend in the clearcoat texel (${linearClearcoat} vs ${nearestClearcoat})`)

  const nearestSheen = renderSheen(THREE.NearestFilter)
  const linearSheen = renderSheen(THREE.LinearFilter)
  assert.ok(linearSheen.r > nearestSheen.r + 1.5, `LinearFilter should blend in red sheen (${linearSheen.r} vs ${nearestSheen.r})`)

  const nearestSpecular = renderSpecularIntensity(THREE.NearestFilter)
  const linearSpecular = renderSpecularIntensity(THREE.LinearFilter)
  assert.ok(linearSpecular > nearestSpecular + 20, `LinearFilter should blend in specular intensity (${linearSpecular} vs ${nearestSpecular})`)

  const nearestTransmission = renderTransmission(THREE.NearestFilter)
  const linearTransmission = renderTransmission(THREE.LinearFilter)
  assert.ok(linearTransmission.b > nearestTransmission.b + 20, `LinearFilter should blend in transmission (${linearTransmission.b} vs ${nearestTransmission.b})`)

  const nearestNormal = renderClearcoatNormal(THREE.NearestFilter)
  const linearNormal = renderClearcoatNormal(THREE.LinearFilter)
  const normalDiff = meanAbsDiff(nearestNormal, linearNormal)
  assert.ok(normalDiff > 2, `LinearFilter should blend clearcoat normals differently than NearestFilter, diff=${normalDiff.toFixed(2)}`)
})

test('specularColorMap samples the selected secondary UV channel', () => {
  function renderWithChannel(channel) {
    const specularColorMap = rgbaTexture([
      0, 0, 0, 255,
      255, 0, 0, 255,
    ], 2, 1)
    specularColorMap.channel = channel

    const geometry = constantUvPlane(0.25, 0.5)
    setConstantUvAttribute(geometry, 'uv1', 0.75, 0.5)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      geometry,
      new THREE.MeshPhysicalMaterial({
        color: 0x000000,
        roughness: 0.08,
        metalness: 0,
        specularIntensity: 1,
        specularColor: new THREE.Color(1, 1, 1),
        specularColorMap,
      }),
    ))
    const light = new THREE.PointLight(0xffffff, 300)
    light.position.set(0, 0, 2)
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const primary = renderWithChannel(0)
  const secondary = renderWithChannel(1)
  assert.ok(secondary.r > primary.r + 4, `specularColorMap channel=1 should sample uv1's red texel (${secondary.r} vs ${primary.r})`)
  assert.ok(secondary.r > secondary.g + 4, `specularColorMap channel=1 should tint the specular response red (${secondary.r} vs ${secondary.g})`)
})

test('specularColorMap decodes sRGB colorSpace before shading', () => {
  function renderColorSpace(colorSpace) {
    const specularColorMap = solidTexture(128, 128, 128)
    specularColorMap.colorSpace = colorSpace

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshPhysicalMaterial({
        color: 0x000000,
        roughness: 0.05,
        metalness: 0,
        specularIntensity: 1,
        specularColor: new THREE.Color(1, 1, 1),
        specularColorMap,
      }),
    ))
    const light = new THREE.PointLight(0xffffff, 450)
    light.position.set(0, 0, 2)
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return maxLuminance(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const srgb = renderColorSpace(THREE.SRGBColorSpace)
  const linear = renderColorSpace(THREE.LinearSRGBColorSpace)
  assert.ok(linear > srgb + 5, `linear specularColorMap should produce brighter highlights than decoded sRGB (${linear} vs ${srgb})`)
})

test('specularIntensityMap samples the selected secondary UV channel', () => {
  function renderWithChannel(channel) {
    const specularIntensityMap = rgbaTexture([
      0, 0, 0, 0,
      0, 0, 0, 255,
    ], 2, 1)
    specularIntensityMap.channel = channel

    const geometry = constantUvPlane(0.25, 0.5)
    setConstantUvAttribute(geometry, 'uv1', 0.75, 0.5)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      geometry,
      new THREE.MeshPhysicalMaterial({
        color: 0x000000,
        roughness: 0.08,
        metalness: 0,
        specularIntensity: 1,
        specularColor: new THREE.Color(1, 1, 1),
        specularIntensityMap,
      }),
    ))
    const light = new THREE.PointLight(0xffffff, 300)
    light.position.set(0, 0, 2)
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return renderRgba(scene, camera, { width: 64, height: 64 })
  }

  const primary = renderWithChannel(0)
  const secondary = renderWithChannel(1)
  assert.ok(maxLuminance(secondary) > maxLuminance(primary) + 40, 'specularIntensityMap channel=1 should enable the uv1 specular texel')
})

test('transmissionMap samples the selected secondary UV channel', () => {
  function renderWithChannel(channel) {
    const transmissionMap = rgbaTexture([
      0, 0, 0, 255,
      255, 0, 0, 255,
    ], 2, 1)
    transmissionMap.channel = channel

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const back = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ color: 0x0000ff }),
    )
    back.position.z = -0.2
    scene.add(back)

    const geometry = constantUvPlane(0.25, 0.5)
    setConstantUvAttribute(geometry, 'uv1', 0.75, 0.5)
    scene.add(new THREE.Mesh(
      geometry,
      new THREE.MeshPhysicalMaterial({
        color: 0xff0000,
        roughness: 0.1,
        metalness: 0,
        transmission: 1,
        transmissionMap,
        ior: 1.5,
        thickness: 0,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const primary = renderWithChannel(0)
  const secondary = renderWithChannel(1)
  assert.ok(primary.r > primary.b + 30, `transmissionMap channel=0 should sample the opaque primary UV texel (${primary.r} vs ${primary.b})`)
  assert.ok(secondary.b > secondary.r + 40, `transmissionMap channel=1 should sample the transmitting uv1 texel (${secondary.b} vs ${secondary.r})`)
})

test('clearcoatMap samples the selected secondary UV channel', () => {
  function renderWithChannel(channel) {
    const clearcoatMap = rgbaTexture([
      0, 0, 0, 255,
      255, 0, 0, 255,
    ], 2, 1)
    clearcoatMap.channel = channel

    const geometry = constantUvPlane(0.25, 0.5)
    setConstantUvAttribute(geometry, 'uv1', 0.75, 0.5)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.environment = makeEnvironmentTexture()
    scene.environmentIntensity = 2
    scene.add(new THREE.Mesh(
      geometry,
      new THREE.MeshPhysicalMaterial({
        color: 0x000000,
        roughness: 1,
        metalness: 0,
        clearcoat: 1,
        clearcoatRoughness: 0.04,
        clearcoatMap,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const primary = renderWithChannel(0)
  const secondary = renderWithChannel(1)
  const primaryLum = 0.2126 * primary.r + 0.7152 * primary.g + 0.0722 * primary.b
  const secondaryLum = 0.2126 * secondary.r + 0.7152 * secondary.g + 0.0722 * secondary.b
  assert.ok(secondaryLum > primaryLum + 80, `clearcoatMap channel=1 should enable stronger clearcoat IBL (${secondaryLum.toFixed(1)} vs ${primaryLum.toFixed(1)})`)
})

test('clearcoatRoughnessMap samples the selected secondary UV channel', () => {
  function renderWithChannel(channel) {
    const clearcoatRoughnessMap = rgbaTexture([
      0, 0, 0, 255,
      0, 255, 0, 255,
    ], 2, 1)
    clearcoatRoughnessMap.channel = channel

    const geometry = constantUvPlane(0.25, 0.5)
    setConstantUvAttribute(geometry, 'uv1', 0.75, 0.5)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.environment = makeEnvironmentTexture()
    scene.environmentIntensity = 2
    scene.add(new THREE.Mesh(
      geometry,
      new THREE.MeshPhysicalMaterial({
        color: 0x000000,
        roughness: 1,
        metalness: 0,
        clearcoat: 1,
        clearcoatRoughness: 1,
        clearcoatRoughnessMap,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const primary = renderWithChannel(0)
  const secondary = renderWithChannel(1)
  const primaryLum = 0.2126 * primary.r + 0.7152 * primary.g + 0.0722 * primary.b
  const secondaryLum = 0.2126 * secondary.r + 0.7152 * secondary.g + 0.0722 * secondary.b
  assert.ok(primaryLum > secondaryLum + 20, `clearcoatRoughnessMap channel=0 should keep the clearcoat IBL sharper/brighter (${primaryLum.toFixed(1)} vs ${secondaryLum.toFixed(1)})`)
})

test('clearcoatNormalMap samples the selected secondary UV channel', () => {
  function renderWithChannel(channel) {
    const clearcoatNormalMap = rgbaTexture([
      128, 128, 255, 255,
      255, 128, 128, 255,
    ], 2, 1)
    clearcoatNormalMap.channel = channel

    const geometry = constantUvPlane(0.25, 0.5)
    setConstantUvAttribute(geometry, 'uv1', 0.75, 0.5)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.environment = makeEnvironmentTexture()
    scene.environmentIntensity = 2
    scene.add(new THREE.Mesh(
      geometry,
      new THREE.MeshPhysicalMaterial({
        color: 0x000000,
        roughness: 1,
        metalness: 0,
        clearcoat: 1,
        clearcoatRoughness: 0.04,
        clearcoatNormalMap,
        clearcoatNormalScale: new THREE.Vector2(1, 1),
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return renderRgba(scene, camera, { width: 64, height: 64 })
  }

  const primary = renderWithChannel(0)
  const secondary = renderWithChannel(1)
  const diff = meanAbsDiff(primary, secondary)
  assert.ok(diff > 5, `clearcoatNormalMap channel=1 should sample the tilted uv1 normal, mean diff=${diff.toFixed(2)}`)
})

test('sheenColorMap samples the selected secondary UV channel', () => {
  function renderWithChannel(channel) {
    const sheenColorMap = rgbaTexture([
      0, 0, 0, 255,
      255, 0, 0, 255,
    ], 2, 1)
    sheenColorMap.channel = channel

    const geometry = constantUvPlane(0.25, 0.5)
    setConstantUvAttribute(geometry, 'uv1', 0.75, 0.5)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.environment = makeEnvironmentTexture()
    scene.environmentIntensity = 2
    scene.add(new THREE.Mesh(
      geometry,
      new THREE.MeshPhysicalMaterial({
        color: 0x000000,
        roughness: 1,
        metalness: 0,
        sheen: 1,
        sheenColor: new THREE.Color(1, 1, 1),
        sheenRoughness: 0.35,
        sheenColorMap,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const primary = renderWithChannel(0)
  const secondary = renderWithChannel(1)
  assert.ok(secondary.r > primary.r + 3, `sheenColorMap channel=1 should add red sheen from uv1 (${secondary.r} vs ${primary.r})`)
  assert.ok(secondary.r > secondary.g + 3, `sheenColorMap channel=1 should keep the sampled red sheen tint (${secondary.r} vs ${secondary.g})`)
})

test('sheenColorMap decodes sRGB colorSpace before shading', () => {
  function renderColorSpace(colorSpace) {
    const sheenColorMap = solidTexture(128, 128, 128)
    sheenColorMap.colorSpace = colorSpace

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.environment = makeEnvironmentTexture()
    scene.environmentIntensity = 3
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshPhysicalMaterial({
        color: 0x000000,
        roughness: 1,
        metalness: 0,
        sheen: 1,
        sheenColor: new THREE.Color(1, 1, 1),
        sheenRoughness: 0.35,
        sheenColorMap,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return maxLuminance(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const srgb = renderColorSpace(THREE.SRGBColorSpace)
  const linear = renderColorSpace(THREE.LinearSRGBColorSpace)
  assert.ok(linear > srgb + 3, `linear sheenColorMap should produce brighter sheen than decoded sRGB (${linear} vs ${srgb})`)
})

test('sheenRoughnessMap samples the selected secondary UV channel', () => {
  function renderWithChannel(channel) {
    const sheenRoughnessMap = rgbaTexture([
      0, 0, 0, 0,
      0, 0, 0, 255,
    ], 2, 1)
    sheenRoughnessMap.channel = channel

    const geometry = constantUvPlane(0.25, 0.5)
    setConstantUvAttribute(geometry, 'uv1', 0.75, 0.5)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.environment = makeEnvironmentTexture()
    scene.environmentIntensity = 2
    scene.add(new THREE.Mesh(
      geometry,
      new THREE.MeshPhysicalMaterial({
        color: 0x000000,
        roughness: 1,
        metalness: 0,
        sheen: 1,
        sheenColor: new THREE.Color(1, 0, 0),
        sheenRoughness: 1,
        sheenRoughnessMap,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return renderRgba(scene, camera, { width: 64, height: 64 })
  }

  const primary = renderWithChannel(0)
  const secondary = renderWithChannel(1)
  const diff = meanAbsDiff(primary, secondary)
  assert.ok(diff > 5, `sheenRoughnessMap channel=1 should sample the rough uv1 texel, mean diff=${diff.toFixed(2)}`)
})

test('anisotropyMap samples the selected secondary UV channel', () => {
  function renderWithChannel(channel) {
    const anisotropyMap = rgbaTexture([
      128, 128, 0, 255,
      255, 128, 255, 255,
    ], 2, 1)
    anisotropyMap.channel = channel

    const geometry = constantUvPlane(0.25, 0.5)
    setConstantUvAttribute(geometry, 'uv1', 0.75, 0.5)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      geometry,
      new THREE.MeshPhysicalMaterial({
        color: 0x111111,
        roughness: 0.2,
        metalness: 0,
        anisotropy: 1,
        anisotropyRotation: Math.PI / 4,
        anisotropyMap,
      }),
    ))
    const light = new THREE.PointLight(0xffffff, 250)
    light.position.set(0.8, 0.8, 2)
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return renderRgba(scene, camera, { width: 64, height: 64 })
  }

  const primary = renderWithChannel(0)
  const secondary = renderWithChannel(1)
  const diff = meanAbsDiff(primary, secondary)
  assert.ok(diff > 1, `anisotropyMap channel=1 should sample the anisotropic uv1 texel, mean diff=${diff.toFixed(2)}`)
})

test('thicknessMap samples the selected secondary UV channel', () => {
  function renderWithChannel(channel) {
    const thicknessMap = rgbaTexture([
      0, 0, 0, 255,
      0, 255, 0, 255,
    ], 2, 1)
    thicknessMap.channel = channel

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const back = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    )
    back.position.z = -0.2
    scene.add(back)

    const geometry = constantUvPlane(0.25, 0.5)
    setConstantUvAttribute(geometry, 'uv1', 0.75, 0.5)
    scene.add(new THREE.Mesh(
      geometry,
      new THREE.MeshPhysicalMaterial({
        color: 0xffffff,
        roughness: 0.1,
        metalness: 0,
        transmission: 1,
        ior: 1.5,
        thickness: 8,
        thicknessMap,
        attenuationColor: new THREE.Color(0.02, 0.02, 1),
        attenuationDistance: 1,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const primary = renderWithChannel(0)
  const secondary = renderWithChannel(1)
  assert.ok(primary.r > primary.b - 15, `thicknessMap channel=0 should sample the thin primary UV texel (${primary.r} vs ${primary.b})`)
  assert.ok(secondary.b > secondary.r + 40, `thicknessMap channel=1 should sample the attenuating uv1 texel (${secondary.b} vs ${secondary.r})`)
})

test('custom WGSL fragment material affects rendered output', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.02)
  const material = new THREE.MeshBasicMaterial({ color: 0xffffff })
  material.userData.headlessThreeRenderer = {
    fragmentWgsl: 'return vec4<f32>(0.0, 1.0, 1.0, alpha);',
  }
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(), material))

  const rgba = renderRgba(scene, makeCamera())
  const mean = meanRgba(rgba)
  assert.ok(mean.g > mean.r + 5, `custom shader should raise green over red (${mean.g} vs ${mean.r})`)
  assert.ok(mean.b > mean.r + 5, `custom shader should raise blue over red (${mean.b} vs ${mean.r})`)
})

test('ShaderMaterial without headless WGSL override fails clearly', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.ShaderMaterial({
      vertexShader: 'void main() { gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
      fragmentShader: 'void main() { gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); }',
    }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  assert.throws(
    () => renderRgba(scene, camera, { width: 64, height: 64 }),
    /ShaderMaterial.*fragmentWgsl/i,
  )
})

test('ShaderMaterial can opt into custom WGSL fragment output', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const material = new THREE.ShaderMaterial()
  material.userData.headlessThreeRenderer = {
    fragmentWgsl: 'return vec4<f32>(0.0, 1.0, 0.0, alpha);',
  }
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  assert.ok(mean.g > mean.r + 40, `ShaderMaterial WGSL override should render green output (${mean.g} vs ${mean.r})`)
  assert.ok(mean.g > mean.b + 40, `ShaderMaterial WGSL override should render green output (${mean.g} vs ${mean.b})`)
})

test('material onBeforeCompile customizations fail clearly', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const material = new THREE.MeshBasicMaterial({ color: 0xffffff })
  material.onBeforeCompile = (shader) => {
    shader.fragmentShader = shader.fragmentShader.replace('vec4', 'vec4')
  }
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  assert.throws(
    () => renderRgba(scene, camera, { width: 64, height: 64 }),
    /onBeforeCompile.*fragmentWgsl/i,
  )
})

test('material onBeforeCompile can opt into custom WGSL fragment output', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const material = new THREE.MeshBasicMaterial({ color: 0xffffff })
  material.onBeforeCompile = (shader) => {
    shader.fragmentShader = shader.fragmentShader.replace('vec4', 'vec4')
  }
  material.userData.headlessThreeRenderer = {
    fragmentWgsl: 'return vec4<f32>(1.0, 0.0, 1.0, alpha);',
  }
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  assert.ok(mean.r > mean.g + 40, `onBeforeCompile WGSL override should render magenta red (${mean.r} vs ${mean.g})`)
  assert.ok(mean.b > mean.g + 40, `onBeforeCompile WGSL override should render magenta blue (${mean.b} vs ${mean.g})`)
})

test('renderToTarget populates a target-like object with raw RGBA', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.1, 0.1, 0.1)
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial({ color: 0x00ffaa })))

  const target = { texture: {} }
  const out = renderToTarget(scene, makeCamera(), target, { width: 64, height: 32 })
  assert.equal(out, target)
  assert.equal(target.width, 64)
  assert.equal(target.height, 32)
  assert.equal(target.data.length, 64 * 32 * 4)
  assert.equal(target.texture.image.data, target.data)
})

test('unsupported render target depth, MRT, and MSAA requests fail clearly', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.1, 0.1, 0.1)
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial({ color: 0x00ffaa })))
  const camera = makeCamera()

  const targetCases = [
    [{ depthTexture: {} }, /depthTexture output.*not supported/i, 'depthTexture'],
    [{ texture: [{}, {}] }, /Multiple render target color attachments.*not supported/i, 'texture array'],
    [{ textures: [{}, {}] }, /Multiple render target color attachments.*not supported/i, 'textures array'],
    [{ isWebGLMultipleRenderTargets: true, texture: {} }, /Multiple render target color attachments.*not supported/i, 'MRT flag'],
    [{ samples: 4 }, /MSAA sample counts.*not supported/i, 'target samples'],
    [{ sampleCount: 4 }, /MSAA sample counts.*not supported/i, 'target sampleCount'],
  ]

  for (const [target, pattern, label] of targetCases) {
    assert.throws(
      () => renderToTarget(scene, camera, target, { width: 32, height: 32 }),
      pattern,
      label,
    )
  }

  for (const options of [{ samples: 4 }, { sampleCount: 4 }]) {
    assert.throws(
      () => renderRgba(scene, camera, { width: 32, height: 32, ...options }),
      /MSAA sample counts.*not supported/i,
      JSON.stringify(options),
    )
  }
})

test('post-processing options modify the final image', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(1, 0, 0)
  const camera = makeCamera()

  const base = renderRgba(scene, camera, { width: 64, height: 64 })
  const processed = renderRgba(scene, camera, {
    width: 64,
    height: 64,
    postProcessing: { invert: 1, saturation: 1.5, vignette: 0.25 },
  })
  const diff = meanAbsDiff(base, processed)
  const mean = meanRgba(processed)
  assert.ok(diff > 20, `expected post processing to change image, diff=${diff.toFixed(2)}`)
  assert.ok(mean.g > mean.r, `inverted red background should have stronger green than red (${mean.g} vs ${mean.r})`)
})

test('scene-level reflection probe feeds physical IBL when scene.environment is absent', () => {
  const camera = makeCamera()

  function makeScene(withProbe) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0.04, 0.04, 0.045)
    addLights(scene)
    if (withProbe) {
      scene.userData.headlessThreeRenderer = {
        reflectionProbe: {
          texture: makeEnvironmentTexture(),
          intensity: 1.0,
        },
      }
    }
    scene.add(new THREE.Mesh(
      new THREE.SphereGeometry(1, 32, 32),
      new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1.0, roughness: 0.2 }),
    ))
    return scene
  }

  const withoutProbe = renderRgba(makeScene(false), camera)
  const withProbe = renderRgba(makeScene(true), camera)
  const diff = meanAbsDiff(withoutProbe, withProbe)
  assert.ok(diff > 0.5, `expected reflection probe to affect metallic IBL, diff=${diff.toFixed(3)}`)
})

test('physical transmission samples the already-rendered scene color', () => {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  function makeScene(material) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 0, 0)
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(3, 3), material))
    return scene
  }

  const opaque = renderRgba(
    makeScene(new THREE.MeshPhysicalMaterial({ color: 0xffffff, roughness: 0.2 })),
    camera,
    { width: 64, height: 64 },
  )
  const transmissive = renderRgba(
    makeScene(new THREE.MeshPhysicalMaterial({
      color: 0xffffff,
      roughness: 0.05,
      transmission: 1.0,
      thickness: 0.2,
      ior: 1.5,
    })),
    camera,
    { width: 64, height: 64 },
  )

  const diff = meanAbsDiff(opaque, transmissive)
  const mean = meanRgba(transmissive)
  assert.ok(diff > 5, `expected transmission to differ from opaque material, diff=${diff.toFixed(2)}`)
  assert.ok(mean.r > mean.g + 30, `transmission should reveal red scene color (${mean.r} vs ${mean.g})`)
})

test('directional cascaded shadow hints render successfully', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.05, 0.05, 0.05)
  scene.add(new THREE.AmbientLight(0xffffff, 0.2))

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(8, 8),
    new THREE.MeshStandardMaterial({ color: 0x888888, roughness: 0.8 }),
  )
  ground.rotation.x = -Math.PI / 2
  ground.receiveShadow = true
  scene.add(ground)

  const box = new THREE.Mesh(
    new THREE.BoxGeometry(1, 1, 1),
    new THREE.MeshStandardMaterial({ color: 0xff5533 }),
  )
  box.position.y = 0.5
  box.castShadow = true
  scene.add(box)

  const light = new THREE.DirectionalLight(0xffffff, 1.5)
  light.position.set(4, 6, 3)
  light.target.position.set(0, 0, 0)
  light.castShadow = true
  light.shadow.mapSize.set(256, 256)
  light.userData.headlessThreeRenderer = {
    shadowCascades: [
      { left: -3, right: 3, top: 3, bottom: -3, near: 0.1, far: 16, split: 4 },
      { left: -7, right: 7, top: 7, bottom: -7, near: 0.1, far: 32, split: 12 },
    ],
  }
  scene.add(light)
  scene.add(light.target)

  const rgba = renderRgba(scene, makeCamera(), { width: 64, height: 64 })
  assert.equal(rgba.length, 64 * 64 * 4)
})

test('ShadowMaterial is transparent except for received shadows', () => {
  function renderShadowMaterial(castShadow) {
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
    caster.castShadow = castShadow
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
    scene.add(light)
    scene.add(light.target)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 96, height: 96 }))
  }

  const unshadowed = renderShadowMaterial(false)
  const shadowed = renderShadowMaterial(true)
  const unshadowedLum = unshadowed.r + unshadowed.g + unshadowed.b
  const shadowedLum = shadowed.r + shadowed.g + shadowed.b
  assert.ok(unshadowedLum > 650, `unshadowed ShadowMaterial receiver should be mostly transparent (${unshadowedLum})`)
  assert.ok(shadowedLum < unshadowedLum - 30, `received shadow should darken the transparent receiver (${shadowedLum} vs ${unshadowedLum})`)
})

test('ShadowMaterial honors material.fog opt-out', () => {
  function renderShadowMaterialFog(fog) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)
    scene.fog = new THREE.Fog(0x0000ff, 0, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1, fog }),
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
    scene.add(light)
    scene.add(light.target)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 96, height: 96 }))
  }

  const fogged = renderShadowMaterialFog(true)
  const unfogged = renderShadowMaterialFog(false)
  assert.ok(fogged.b > fogged.r + 15, `fogged ShadowMaterial should tint received shadows blue (${fogged.b} vs ${fogged.r})`)
  assert.ok(fogged.b > unfogged.b + 10, `fog=false should skip the fog color tint (${fogged.b} vs ${unfogged.b})`)
})

test('lines topology renders successfully', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.1, 0.1, 0.1)
  const geom = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-1, 0, 0),
    new THREE.Vector3(1, 0, 0),
    new THREE.Vector3(0, 1, 0),
    new THREE.Vector3(0, -1, 0),
  ])
  scene.add(new THREE.LineSegments(geom, new THREE.LineBasicMaterial({ color: 0xffffff })))

  const camera = makeCamera()
  const r = new Renderer()
  const buf = r.render(scene, camera, { width: SIZE, height: SIZE })
  assertValidPng(buf, { width: SIZE, height: SIZE })
})

test('LineBasicMaterial map alpha samples line UVs', () => {
  const map = rgbaTexture([
    255, 255, 255, 0,
    255, 255, 255, 255,
  ], 2, 1)

  function renderLine(u) {
    const geom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-1.5, 0, 0),
      new THREE.Vector3(1.5, 0, 0),
    ])
    geom.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([
      u, 0.5,
      u, 0.5,
    ]), 2))

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0.1, 0.1, 0.1)
    scene.add(new THREE.Line(
      geom,
      new THREE.LineBasicMaterial({ color: 0xffffff, map, alphaTest: 0.5 }),
    ))
    return renderRgba(scene, makeCamera(), { width: 96, height: 96 })
  }

  const discarded = nonBackgroundRatio(renderLine(0.25), BG)
  const visible = nonBackgroundRatio(renderLine(0.75), BG)
  assert.ok(visible > 0.001, `opaque map alpha texel should leave visible line pixels (${visible})`)
  assert.ok(discarded < visible * 0.3, `transparent map alpha texel should discard most line pixels (${discarded} vs ${visible})`)
})

test('LineDashedMaterial renders fewer visible line pixels than a solid line', () => {
  function makeScene(material) {
    const geom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-1.5, 0, 0),
      new THREE.Vector3(1.5, 0, 0),
    ])
    const line = new THREE.Line(geom, material)
    line.computeLineDistances()

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0.1, 0.1, 0.1)
    scene.add(line)
    return scene
  }

  const camera = makeCamera()
  const solid = renderRgba(makeScene(new THREE.LineBasicMaterial({ color: 0xffffff })), camera)
  const dashed = renderRgba(makeScene(new THREE.LineDashedMaterial({
    color: 0xffffff,
    dashSize: 0.15,
    gapSize: 0.15,
    scale: 1,
  })), camera)

  const solidRatio = nonBackgroundRatio(solid, BG)
  const dashedRatio = nonBackgroundRatio(dashed, BG)
  assert.ok(solidRatio > 0.001, `expected solid line pixels, got ratio ${solidRatio}`)
  assert.ok(dashedRatio > 0.001, `expected dashed line pixels, got ratio ${dashedRatio}`)
  assert.ok(dashedRatio < solidRatio * 0.85, `dashed line should cover less than solid (${dashedRatio} vs ${solidRatio})`)
})

test('line materials with non-default linewidth fail clearly', () => {
  const cases = [
    new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 2 }),
    new THREE.LineDashedMaterial({ color: 0xffffff, linewidth: 2, dashSize: 0.2, gapSize: 0.1 }),
  ]

  for (const material of cases) {
    const geom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-1, 0, 0),
      new THREE.Vector3(1, 0, 0),
    ])
    const line = new THREE.Line(geom, material)
    line.computeLineDistances()

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(line)

    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      /linewidth.*not supported/i,
      material.type,
    )
  }
})

test('LineDashedMaterial map alpha samples reconstructed dash UVs', () => {
  const map = rgbaTexture([
    255, 255, 255, 0,
    255, 255, 255, 255,
  ], 2, 1)
  const geom = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-1.5, 0, 0),
    new THREE.Vector3(1.5, 0, 0),
  ])
  geom.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([
    0, 0.5,
    1, 0.5,
  ]), 2))

  const line = new THREE.Line(geom, new THREE.LineDashedMaterial({
    color: 0xffffff,
    map,
    alphaTest: 0.5,
    dashSize: 0.5,
    gapSize: 0.2,
    scale: 1,
  }))
  line.computeLineDistances()

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.1, 0.1, 0.1)
  scene.add(line)

  const ratio = nonBackgroundRatio(renderRgba(scene, makeCamera(), { width: 96, height: 96 }), BG)
  assert.ok(ratio > 0.0005, `dashed line UVs should sample the opaque map region (${ratio})`)
})

test('LineDashedMaterial interpolates vertex colors across dash segments', () => {
  const geom = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-1.5, 0, 0),
    new THREE.Vector3(1.5, 0, 0),
  ])
  geom.setAttribute('color', new THREE.BufferAttribute(new Float32Array([
    1, 0, 0,
    0, 1, 0,
  ]), 3))

  const line = new THREE.Line(geom, new THREE.LineDashedMaterial({
    color: 0xffffff,
    vertexColors: true,
    dashSize: 0.5,
    gapSize: 0.2,
    scale: 1,
  }))
  line.computeLineDistances()

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(line)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 96, height: 96 })
  const redPixels = countRegionPixels(rgba, 96, 96, 8, 36, 44, 60, (r, g, b) => r > g + 20 && r > b + 20)
  const greenPixels = countRegionPixels(rgba, 96, 96, 52, 36, 88, 60, (r, g, b) => g > r + 20 && g > b + 20)
  assert.ok(redPixels > 2, `left dash segments should retain red vertex colors (${redPixels})`)
  assert.ok(greenPixels > 2, `right dash segments should retain green vertex colors (${greenPixels})`)
})

test('points topology renders successfully', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.1, 0.1, 0.1)
  const positions = new Float32Array(30)
  for (let i = 0; i < 10; i++) {
    positions[i * 3 + 0] = Math.cos(i) * 0.8
    positions[i * 3 + 1] = Math.sin(i) * 0.8
    positions[i * 3 + 2] = 0
  }
  const geom = new THREE.BufferGeometry()
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  scene.add(new THREE.Points(geom, new THREE.PointsMaterial({ color: 0xffffff, size: 0.1 })))

  const camera = makeCamera()
  const r = new Renderer()
  const buf = r.render(scene, camera, { width: SIZE, height: SIZE })
  assertValidPng(buf, { width: SIZE, height: SIZE })
})

test('PointsMaterial size controls billboard pixel bounds', () => {
  function renderPoint(size) {
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({
      color: 0xffffff,
      size,
      sizeAttenuation: false,
    })))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return nonBackgroundBounds(renderRgba(scene, camera, { width: 96, height: 96 }), 96, 96, [0, 0, 0])
  }

  const small = renderPoint(10)
  const large = renderPoint(34)
  assert.ok(small.width >= 8 && small.height >= 8, `small point should render as a visible billboard (${small.width}x${small.height})`)
  assert.ok(large.width > small.width * 2, `larger point should produce wider bounds (${large.width} vs ${small.width})`)
  assert.ok(large.height > small.height * 2, `larger point should produce taller bounds (${large.height} vs ${small.height})`)
})

test('PointsMaterial maps, alpha maps, and vertex colors affect billboards', () => {
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.35, 0, 0,
    0.35, 0, 0,
  ]), 3))
  geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array([
    1, 0, 0,
    0, 1, 0,
  ]), 3))

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({
    color: 0xffffff,
    vertexColors: true,
    map: solidTexture(255, 255, 255),
    size: 24,
    sizeAttenuation: false,
  })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const colored = meanRgba(renderRgba(scene, camera, { width: 96, height: 96 }))
  assert.ok(colored.r > 8, `red vertex-colored point should contribute to output (${colored.r})`)
  assert.ok(colored.g > 8, `green vertex-colored point should contribute to output (${colored.g})`)
  assert.ok(colored.b < Math.max(colored.r, colored.g) * 0.3, `vertex colors should avoid blue contribution (${colored.b})`)

  const alphaScene = new THREE.Scene()
  alphaScene.background = new THREE.Color(0, 0, 1)
  alphaScene.add(new THREE.Points(
    geometry,
    new THREE.PointsMaterial({
      color: 0x00ff00,
      alphaMap: solidTexture(255, 0, 255),
      alphaTest: 0.5,
      size: 36,
      sizeAttenuation: false,
    }),
  ))
  const discarded = meanRgba(renderRgba(alphaScene, camera, { width: 64, height: 64 }))
  assert.ok(discarded.b > discarded.g + 80, `alphaMap green channel should discard point billboards (${discarded.b} vs ${discarded.g})`)
})

test('empty scene renders the background color', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(1, 0, 0)
  const camera = makeCamera()

  const r = new Renderer()
  const rgba = r.render(scene, camera, { width: 64, height: 64, format: 'rgba' })
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 200, `expected red background, got r=${mean.r}`)
  assert.ok(mean.g < 20, `expected red background, got g=${mean.g}`)
  assert.ok(mean.b < 20, `expected red background, got b=${mean.b}`)
})

test('backgroundIntensity scales background color clears', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(1, 0, 0)
  scene.backgroundIntensity = 0.5

  const camera = makeCamera()
  const dimmed = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  const override = meanRgba(renderRgba(scene, camera, { width: 64, height: 64, backgroundIntensity: 1 }))
  assert.ok(dimmed.r > 90 && dimmed.r < 170, `backgroundIntensity should dim red clears to about half strength (${dimmed.r})`)
  assert.ok(override.r > dimmed.r + 80, `options.backgroundIntensity should override scene.backgroundIntensity (${override.r} vs ${dimmed.r})`)
})

test('empty scene renders a texture background', () => {
  const scene = new THREE.Scene()
  scene.background = solidTexture(0, 255, 0)
  const camera = makeCamera()

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  assert.ok(mean.g > mean.r + 80, `expected green texture background, got ${mean.g} vs ${mean.r}`)
  assert.ok(mean.g > mean.b + 80, `expected green texture background, got ${mean.g} vs ${mean.b}`)
})

test('backgroundIntensity scales texture backgrounds', () => {
  const scene = new THREE.Scene()
  scene.background = solidTexture(0, 255, 0)

  const camera = makeCamera()
  const full = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  scene.backgroundIntensity = 0.25
  const dimmed = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  assert.ok(full.g > dimmed.g + 60, `backgroundIntensity should dim texture backgrounds (${full.g} vs ${dimmed.g})`)
  assert.ok(dimmed.g > dimmed.r + 40, `dimmed texture background should keep the sampled green hue (${dimmed.g} vs ${dimmed.r})`)
})

test('backgroundBlurriness softens 2D texture backgrounds', () => {
  function renderBackground(blurriness) {
    const texture = rgbaTexture([
      255, 0, 0, 255,
      0, 255, 0, 255,
    ], 2, 1)
    texture.magFilter = THREE.NearestFilter
    texture.minFilter = THREE.NearestFilter

    const scene = new THREE.Scene()
    scene.background = texture
    scene.backgroundBlurriness = blurriness

    const camera = makeCamera()
    return renderRgba(scene, camera, { width: 64, height: 64 })
  }

  const sharp = meanRegion(renderBackground(0), 64, 64, 28, 20, 31, 44)
  const blurred = meanRegion(renderBackground(1), 64, 64, 28, 20, 31, 44)
  assert.ok(sharp.r > sharp.g + 120, `sharp background should sample the red texel (${sharp.r} vs ${sharp.g})`)
  assert.ok(blurred.g > sharp.g + 80, `blurred background should mix in the green texel (${blurred.g} vs ${sharp.g})`)
  assert.ok(sharp.r > blurred.r + 20, `blurred background should soften the red texel (${sharp.r} vs ${blurred.r})`)
})

test('background textures apply UV transforms', () => {
  const background = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  background.offset.set(0.5, 0)

  const scene = new THREE.Scene()
  scene.background = background
  const camera = makeCamera()

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  assert.ok(mean.g > mean.r + 40, `background texture offset should shift the sampled texel from red to green (${mean.g} vs ${mean.r})`)
})

test('cube and equirect background texture mappings fail clearly', () => {
  const camera = makeCamera()
  const cases = [
    ['equirect scene background', { sceneBackground: Object.assign(solidTexture(0, 255, 0), { mapping: THREE.EquirectangularReflectionMapping }) }],
    ['cube scene background', { sceneBackground: Object.assign(solidTexture(0, 255, 0), { mapping: THREE.CubeReflectionMapping }) }],
    ['cube texture flag scene background', { sceneBackground: Object.assign(solidTexture(0, 255, 0), { isCubeTexture: true }) }],
    ['equirect option background', { optionBackground: Object.assign(solidTexture(0, 255, 0), { mapping: THREE.EquirectangularReflectionMapping }) }],
  ]

  for (const [name, { sceneBackground, optionBackground }] of cases) {
    const scene = new THREE.Scene()
    scene.background = sceneBackground ?? new THREE.Color(0, 0, 0)
    assert.throws(
      () => renderRgba(scene, camera, { width: 64, height: 64, background: optionBackground }),
      /background.*cube\/equirectangular.*not supported/i,
      name,
    )
  }
})

test('render options accept texture backgrounds', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(1, 0, 0)
  const camera = makeCamera()

  const mean = meanRgba(renderRgba(scene, camera, {
    width: 64,
    height: 64,
    background: solidTexture(0, 0, 255),
  }))
  assert.ok(mean.b > mean.r + 80, `options.background texture should override scene background (${mean.b} vs ${mean.r})`)
})

test('render option color backgrounds override scene texture backgrounds', () => {
  const scene = new THREE.Scene()
  scene.background = Object.assign(solidTexture(0, 255, 0), { mapping: THREE.EquirectangularReflectionMapping })
  const camera = makeCamera()

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64, background: [1, 0, 0] }))
  assert.ok(mean.r > 200, `options.background color should override scene texture background (${mean.r})`)
  assert.ok(mean.g < 30, `options.background color should suppress scene texture background (${mean.g})`)
})

test('render options viewport confines draws to an output rectangle', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 1)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(4, 4),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, {
    width: 64,
    height: 64,
    viewport: { x: 32, y: 16, width: 32, height: 32 },
  })
  const inside = meanRegion(rgba, 64, 64, 40, 24, 56, 40)
  const outside = meanRegion(rgba, 64, 64, 8, 24, 24, 40)
  assert.ok(inside.r > inside.b + 80, `viewport region should contain the red mesh (${inside.r} vs ${inside.b})`)
  assert.ok(outside.b > outside.r + 80, `outside viewport should retain blue background (${outside.b} vs ${outside.r})`)
})

test('render options scissor clips draws to an output rectangle', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 1)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(4, 4),
    new THREE.MeshBasicMaterial({ color: 0x00ff00 }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, {
    width: 64,
    height: 64,
    scissor: [16, 16, 32, 32],
  })
  const inside = meanRegion(rgba, 64, 64, 24, 24, 40, 40)
  const outsideLeft = meanRegion(rgba, 64, 64, 4, 24, 12, 40)
  const outsideTop = meanRegion(rgba, 64, 64, 24, 4, 40, 12)
  assert.ok(inside.g > inside.b + 80, `scissor region should contain the green mesh (${inside.g} vs ${inside.b})`)
  assert.ok(outsideLeft.b > outsideLeft.g + 80, `left of scissor should retain blue background (${outsideLeft.b} vs ${outsideLeft.g})`)
  assert.ok(outsideTop.b > outsideTop.g + 80, `above scissor should retain blue background (${outsideTop.b} vs ${outsideTop.g})`)
})
