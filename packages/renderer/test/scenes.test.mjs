import test from 'node:test'
import assert from 'node:assert/strict'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import lightsApi from '../dist/lights.js'
import { assertValidPng, meanRgba, nonBackgroundRatio } from './helpers.mjs'

const { Renderer, render, renderToTarget } = pkg
const { extractLights, extractAmbientLight, extractAmbientIntensity, extractLightProbe } = lightsApi

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

let sharedRenderer

function getRenderer() {
  sharedRenderer ??= new Renderer()
  return sharedRenderer
}

function renderRgba(scene, camera, options = {}) {
  return getRenderer().render(scene, camera, { width: SIZE, height: SIZE, format: 'rgba', ...options })
}

function renderRgbaIsolated(scene, camera, options = {}) {
  return new Renderer().render(scene, camera, { width: SIZE, height: SIZE, format: 'rgba', ...options })
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

function setTextureMatrixOffset(texture, x, y = 0) {
  texture.matrixAutoUpdate = false
  texture.matrix.set(
    1, 0, x,
    0, 1, y,
    0, 0, 1,
  )
}

function halfFloatToNumber(bits) {
  const sign = bits & 0x8000 ? -1 : 1
  const exponent = (bits >> 10) & 0x1f
  const mantissa = bits & 0x03ff
  if (exponent === 0) return sign * (mantissa / 0x400) * (2 ** -14)
  if (exponent === 0x1f) return mantissa ? Number.NaN : sign * Infinity
  return sign * (1 + mantissa / 0x400) * (2 ** (exponent - 15))
}

function splitEnvironmentTexture() {
  const data = []
  for (let y = 0; y < 2; y++) {
    for (let x = 0; x < 8; x++) {
      if (x < 4) {
        data.push(255, 0, 0, 255)
      } else {
        data.push(0, 255, 0, 255)
      }
    }
  }
  const texture = rgbaTexture(data, 8, 2)
  texture.mapping = THREE.EquirectangularReflectionMapping
  return texture
}

function cubeTexture(faceColors) {
  const faces = faceColors.map(([r, g, b, a = 255]) => ({
    data: new Uint8Array([r, g, b, a]),
    width: 1,
    height: 1,
  }))
  const texture = new THREE.CubeTexture(faces)
  texture.needsUpdate = true
  return texture
}

function encodedCubeTexture() {
  const faces = [
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYPj/HwADAgH/5ncLrgAAAABJRU5ErkJggg==',
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4/5/hPwAH/QL+ppTFtAAAAABJRU5ErkJggg==',
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z/D/PwAG/gL+DHWJ3gAAAABJRU5ErkJggg==',
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNg+P//PwAF/wL+Xg47rQAAAABJRU5ErkJggg==',
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNg+M/wHwAEAQH/cetH5QAAAABJRU5ErkJggg==',
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==',
  ].map((base64) => Buffer.from(base64, 'base64'))
  const texture = new THREE.CubeTexture(faces)
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

function meanScalarRegion(data, width, height, x0, y0, x1, y1) {
  let sum = 0
  let count = 0
  for (let y = y0; y < y1; y += 1) {
    assert.ok(y >= 0 && y < height)
    for (let x = x0; x < x1; x += 1) {
      assert.ok(x >= 0 && x < width)
      sum += data[y * width + x]
      count += 1
    }
  }
  return sum / count
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

function objectIdBytes(id) {
  const masked = Math.max(1, Math.trunc(id)) & 0xffffff
  const value = masked === 0 ? 1 : masked
  return [(value >> 16) & 0xff, (value >> 8) & 0xff, value & 0xff]
}

function assertRgbClose(mean, expected, label) {
  assert.ok(Math.abs(mean.r - expected[0]) <= 1, `${label} red should be ${expected[0]}, got ${mean.r}`)
  assert.ok(Math.abs(mean.g - expected[1]) <= 1, `${label} green should be ${expected[1]}, got ${mean.g}`)
  assert.ok(Math.abs(mean.b - expected[2]) <= 1, `${label} blue should be ${expected[2]}, got ${mean.b}`)
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
    minX: maxX >= minX ? minX : 0,
    minY: maxY >= minY ? minY : 0,
    maxX,
    maxY,
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

test('unsupported output format values fail clearly', () => {
  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial({ color: 0xff00ff })))
  const camera = makeCamera()

  assert.throws(
    () => getRenderer().render(scene, camera, { width: 32, height: 32, format: 'webp' }),
    /options\.format webp is not supported.*png.*rgba/i,
  )
  assert.throws(
    () => renderToTarget(scene, camera, {}, { width: 32, height: 32, format: 'webp' }),
    /options\.format webp is not supported.*png.*rgba/i,
  )
})

test('invalid render options containers fail clearly', () => {
  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial({ color: 0xff00ff })))
  const camera = makeCamera()

  assert.throws(
    () => render(scene, camera, null),
    /options must be an options object/i,
  )
  assert.throws(
    () => getRenderer().render(scene, camera, 'bad'),
    /options must be an options object/i,
  )
  assert.throws(
    () => renderToTarget(scene, camera, {}, null),
    /options must be an options object/i,
  )
  assert.throws(
    () => getRenderer().renderToTarget(scene, camera, {}, []),
    /options must be an options object/i,
  )
})

test('invalid render scene and camera containers fail clearly', () => {
  const scene = new THREE.Scene()
  const camera = makeCamera()

  assert.throws(
    () => render(null, camera, { width: 32, height: 32, format: 'rgba' }),
    /render\(scene, camera\) expects scene to be a THREE\.Scene or THREE\.Object3D root/i,
  )
  assert.throws(
    () => getRenderer().render([], camera, { width: 32, height: 32, format: 'rgba' }),
    /render\(scene, camera\) expects scene to be a THREE\.Scene or THREE\.Object3D root/i,
  )
  assert.throws(
    () => render(scene, null, { width: 32, height: 32, format: 'rgba' }),
    /render\(scene, camera\) expects camera to be a THREE\.Camera, THREE\.ArrayCamera, or THREE\.CubeCamera/i,
  )
  assert.throws(
    () => getRenderer().render(scene, { cameras: [] }, { width: 32, height: 32, format: 'rgba' }),
    /render\(scene, camera\) expects camera to be a THREE\.Camera, THREE\.ArrayCamera, or THREE\.CubeCamera/i,
  )
  assert.throws(
    () => renderToTarget(null, camera, {}, { width: 32, height: 32 }),
    /render\(scene, camera\) expects scene to be a THREE\.Scene or THREE\.Object3D root/i,
  )
  assert.throws(
    () => getRenderer().renderToTarget(scene, [], {}, { width: 32, height: 32 }),
    /render\(scene, camera\) expects camera to be a THREE\.Camera, THREE\.ArrayCamera, or THREE\.CubeCamera/i,
  )
})

test('invalid transform matrix values fail clearly', () => {
  const camera = makeCamera()

  const scene = new THREE.Scene()
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial({ color: 0xffffff }))
  mesh.matrixAutoUpdate = false
  mesh.matrixWorldAutoUpdate = false
  mesh.matrixWorld.elements[12] = Number.NaN
  scene.add(mesh)

  assert.throws(
    () => getRenderer().render(scene, camera, { width: 32, height: 32, format: 'rgba' }),
    /mesh\.matrixWorld\.elements\[12\] must be a finite number/i,
  )
})

test('invalid geometry attribute values fail clearly', () => {
  const camera = makeCamera()

  const scene = new THREE.Scene()
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.75, -0.5, 0,
    Number.NaN, -0.5, 0,
    0, 0.75, 0,
  ]), 3))
  scene.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: 0xffffff })))

  assert.throws(
    () => getRenderer().render(scene, camera, { width: 32, height: 32, format: 'rgba' }),
    /geometry\.attributes\.position\[1\]\.x must be a finite number/i,
  )
})

test('invalid geometry attribute count values fail clearly', () => {
  const camera = makeCamera()

  const containerScene = new THREE.Scene()
  const containerGeometry = new THREE.BufferGeometry()
  containerGeometry.attributes = 'attributes'
  containerScene.add(new THREE.Mesh(containerGeometry, new THREE.MeshBasicMaterial({ color: 0xffffff })))
  assert.throws(
    () => renderRgba(containerScene, camera, { width: 32, height: 32 }),
    /geometry\.attributes must be an object/i,
  )

  const positionScene = new THREE.Scene()
  const positionGeometry = new THREE.BufferGeometry()
  positionGeometry.setAttribute('position', {
    count: '3',
    itemSize: 3,
    array: new Float32Array([
      -0.75, -0.5, 0,
      0.75, -0.5, 0,
      0, 0.75, 0,
    ]),
  })
  positionScene.add(new THREE.Mesh(positionGeometry, new THREE.MeshBasicMaterial({ color: 0xffffff })))
  assert.throws(
    () => renderRgba(positionScene, camera, { width: 32, height: 32 }),
    /geometry\.attributes\.position\.count must be a non-negative integer/i,
  )

  const indexScene = new THREE.Scene()
  const indexGeometry = new THREE.BufferGeometry()
  indexGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.75, -0.5, 0,
    0.75, -0.5, 0,
    0, 0.75, 0,
  ]), 3))
  indexGeometry.index = {
    count: Number.NaN,
    itemSize: 1,
    array: new Uint16Array([0, 1, 2]),
  }
  indexScene.add(new THREE.Mesh(indexGeometry, new THREE.MeshBasicMaterial({ color: 0xffffff })))
  assert.throws(
    () => renderRgba(indexScene, camera, { width: 32, height: 32 }),
    /geometry\.index\.count must be a non-negative integer/i,
  )
})

test('malformed geometry bounding spheres fail clearly', () => {
  const camera = makeCamera()
  const cases = [
    ['container', 'sphere', /geometry\.boundingSphere must be a THREE\.Sphere-like object/i],
    ['center', { center: { x: Number.NaN, y: 0, z: 0 }, radius: 1 }, /geometry\.boundingSphere\.center must be a finite Vector3-like value/i],
  ]

  for (const [name, boundingSphere, pattern] of cases) {
    const scene = new THREE.Scene()
    const geometry = new THREE.PlaneGeometry(1, 1)
    geometry.boundingSphere = boundingSphere
    scene.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: 0xffffff })))

    assert.throws(
      () => renderRgba(scene, camera, { width: 32, height: 32 }),
      pattern,
      `${name} should fail clearly`,
    )
  }
})

test('invalid geometry attribute layout values fail clearly', () => {
  const camera = makeCamera()
  const values = new Float32Array([
    -0.75, -0.5, 0,
    0.75, -0.5, 0,
    0, 0.75, 0,
  ])
  const makeScene = (positionAttribute) => {
    const scene = new THREE.Scene()
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', positionAttribute)
    scene.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: 0xffffff })))
    return scene
  }

  const cases = [
    ['itemSize', { count: 3, itemSize: '3', array: values }, /geometry\.attributes\.position\.itemSize must be a positive integer/i],
    ['data container', { count: 3, itemSize: 3, data: 'packed' }, /geometry\.attributes\.position\.data must be an object/i],
    ['stride', { count: 3, itemSize: 3, data: { array: values, stride: '3' } }, /geometry\.attributes\.position\.data\.stride must be a positive integer/i],
    ['offset', { count: 3, itemSize: 3, array: values, offset: -1 }, /geometry\.attributes\.position\.offset must be a non-negative integer/i],
    ['normalized', { count: 3, itemSize: 3, array: values, normalized: 'yes' }, /geometry\.attributes\.position\.normalized must be a boolean/i],
  ]

  for (const [name, positionAttribute, pattern] of cases) {
    assert.throws(
      () => renderRgba(makeScene(positionAttribute), camera, { width: 32, height: 32 }),
      pattern,
      name,
    )
  }
})

test('invalid geometry index values fail clearly', () => {
  const camera = makeCamera()
  const makeScene = (indexAttribute) => {
    const scene = new THREE.Scene()
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
      -0.75, -0.5, 0,
      0.75, -0.5, 0,
      0, 0.75, 0,
    ]), 3))
    geometry.index = indexAttribute
    scene.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: 0xffffff })))
    return scene
  }

  const cases = [
    ['finite', new THREE.BufferAttribute(new Float32Array([0, Number.NaN, 2]), 1), /geometry\.index\[1\]\.x must be a finite number/i],
    ['negative', new THREE.BufferAttribute(new Float32Array([0, -1, 2]), 1), /geometry\.index\[1\]\.x must be a non-negative integer/i],
    ['fractional', new THREE.BufferAttribute(new Float32Array([0, 1.5, 2]), 1), /geometry\.index\[1\]\.x must be a non-negative integer/i],
    ['out of range', new THREE.BufferAttribute(new Uint16Array([0, 1, 3]), 1), /geometry\.index\[2\]\.x must reference a vertex below geometry\.attributes\.position\.count \(3\)/i],
  ]

  for (const [name, indexAttribute, pattern] of cases) {
    assert.throws(
      () => renderRgba(makeScene(indexAttribute), camera, { width: 32, height: 32 }),
      pattern,
      name,
    )
  }
})

test('invalid geometry group values fail clearly', () => {
  const camera = makeCamera()
  const makeScene = (groups) => {
    const scene = new THREE.Scene()
    const geometry = new THREE.PlaneGeometry(1, 1)
    geometry.groups = groups
    scene.add(new THREE.Mesh(geometry, [
      new THREE.MeshBasicMaterial({ color: 0xff0000 }),
      new THREE.MeshBasicMaterial({ color: 0x0000ff }),
    ]))
    return scene
  }

  const cases = [
    ['container', 'groups', /geometry\.groups must be an array/i],
    ['group object', [null], /geometry\.groups\[0\] must be an object/i],
    ['start', [{ start: '0', count: 6, materialIndex: 0 }], /geometry\.groups\[0\]\.start must be a non-negative integer/i],
    ['count', [{ start: 0, count: Number.NaN, materialIndex: 0 }], /geometry\.groups\[0\]\.count must be a non-negative integer/i],
    ['materialIndex', [{ start: 0, count: 6, materialIndex: -1 }], /geometry\.groups\[0\]\.materialIndex must be a non-negative integer/i],
  ]

  for (const [name, groups, pattern] of cases) {
    assert.throws(
      () => renderRgba(makeScene(groups), camera, { width: 32, height: 32 }),
      pattern,
      name,
    )
  }
})

test('invalid geometry drawRange values fail clearly', () => {
  const camera = makeCamera()
  const makeScene = (drawRange) => {
    const scene = new THREE.Scene()
    const geometry = new THREE.PlaneGeometry(1, 1)
    geometry.drawRange = drawRange
    scene.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: 0xffffff })))
    return scene
  }

  const cases = [
    ['container', 'range', /geometry\.drawRange must be an object/i],
    ['start', { start: '0', count: 6 }, /geometry\.drawRange\.start must be a non-negative integer/i],
    ['negative start', { start: -1, count: 6 }, /geometry\.drawRange\.start must be a non-negative integer/i],
    ['count', { start: 0, count: Number.NaN }, /geometry\.drawRange\.count must be a non-negative integer/i],
    ['fractional count', { start: 0, count: 1.5 }, /geometry\.drawRange\.count must be a non-negative integer/i],
  ]

  for (const [name, drawRange, pattern] of cases) {
    assert.throws(
      () => renderRgba(makeScene(drawRange), camera, { width: 32, height: 32 }),
      pattern,
      name,
    )
  }
})

test('malformed object children containers fail clearly', () => {
  const scene = new THREE.Scene()
  scene.children = 'children'
  assert.throws(
    () => renderRgba(scene, makeCamera(), { width: 32, height: 32 }),
    /object\.children must be an array/i,
  )

  const childEntryScene = new THREE.Scene()
  childEntryScene.children = [null]
  assert.throws(
    () => renderRgba(childEntryScene, makeCamera(), { width: 32, height: 32 }),
    /object\.children\[0\] must be an object/i,
  )
})

test('malformed BatchedMesh inputs fail clearly', () => {
  const camera = makeCamera()
  const makeBatchedScene = () => {
    const scene = new THREE.Scene()
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ color: 0xffffff }))
    mesh.isBatchedMesh = true
    scene.add(mesh)
    return { scene, mesh }
  }

  const { scene } = makeBatchedScene()
  assert.throws(
    () => getRenderer().render(scene, camera, { width: 32, height: 32, format: 'rgba' }),
    /THREE\.BatchedMesh instance table is not readable.*ordinary Mesh or InstancedMesh/i,
  )

  const instanceCases = [
    ['instance entry', [null], /THREE\.BatchedMesh\._instanceInfo\[0\] must be an object/i],
    ['active flag', [{ geometryIndex: 0, active: 'yes' }], /THREE\.BatchedMesh\._instanceInfo\[0\]\.active must be a boolean/i],
    ['visible flag', [{ geometryIndex: 0, visible: 'yes' }], /THREE\.BatchedMesh\._instanceInfo\[0\]\.visible must be a boolean/i],
    ['geometry index', [{ geometryIndex: -1 }], /THREE\.BatchedMesh\._instanceInfo\[0\]\.geometryIndex must be a non-negative integer/i],
  ]

  for (const [name, instanceInfo, pattern] of instanceCases) {
    const { scene, mesh } = makeBatchedScene()
    mesh._instanceInfo = instanceInfo
    assert.throws(
      () => getRenderer().render(scene, camera, { width: 32, height: 32, format: 'rgba' }),
      pattern,
      `${name} should fail clearly`,
    )
  }

  const matrixCases = [
    ['matrix texture container', (mesh) => {
      mesh._instanceInfo = [{ geometryIndex: 0 }]
      mesh._geometryInfo = [{ start: 0, count: 6 }]
      mesh._matricesTexture = 'matrices'
    }, /THREE\.BatchedMesh\._matricesTexture must be a texture-like object/i],
    ['matrix texture image container', (mesh) => {
      mesh._instanceInfo = [{ geometryIndex: 0 }]
      mesh._geometryInfo = [{ start: 0, count: 6 }]
      mesh._matricesTexture = { image: 'matrices' }
    }, /THREE\.BatchedMesh\._matricesTexture\.image must be an image-like object/i],
    ['matrix texture data container', (mesh) => {
      mesh._instanceInfo = [{ geometryIndex: 0 }]
      mesh._geometryInfo = [{ start: 0, count: 6 }]
      mesh._matricesTexture = { image: { data: 'matrices' } }
    }, /THREE\.BatchedMesh\._matricesTexture\.image\.data must be an array-like object/i],
    ['color texture image container', (mesh) => {
      mesh._instanceInfo = [{ geometryIndex: 0 }]
      mesh._geometryInfo = [{ start: 0, count: 6 }]
      mesh._matricesTexture = { image: { data: new Float32Array(16) } }
      mesh._colorsTexture = { image: 'colors' }
    }, /THREE\.BatchedMesh\._colorsTexture\.image must be an image-like object/i],
    ['color texture data container', (mesh) => {
      mesh._instanceInfo = [{ geometryIndex: 0 }]
      mesh._geometryInfo = [{ start: 0, count: 6 }]
      mesh._matricesTexture = { image: { data: new Float32Array(16) } }
      mesh._colorsTexture = { image: { data: 'colors' } }
    }, /THREE\.BatchedMesh\._colorsTexture\.image\.data must be an array-like object/i],
  ]

  for (const [name, setup, pattern] of matrixCases) {
    const { scene, mesh } = makeBatchedScene()
    setup(mesh)
    assert.throws(
      () => getRenderer().render(scene, camera, { width: 32, height: 32, format: 'rgba' }),
      pattern,
      `${name} should fail clearly`,
    )
  }
})

test('BatchedMesh renders visible instance transforms and colors', () => {
  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 1.2, -1.2, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const source = new THREE.PlaneGeometry(0.45, 0.45)
  const material = new THREE.MeshBasicMaterial({ color: 0xffffff })
  const batched = new THREE.BatchedMesh(
    4,
    source.getAttribute('position').count,
    source.index.count,
    material,
  )
  const geometryId = batched.addGeometry(source)
  const left = batched.addInstance(geometryId)
  const right = batched.addInstance(geometryId)
  const hidden = batched.addInstance(geometryId)
  const deleted = batched.addInstance(geometryId)
  batched.setMatrixAt(left, new THREE.Matrix4().makeTranslation(-0.55, 0, 0))
  batched.setMatrixAt(right, new THREE.Matrix4().makeTranslation(0.55, 0, 0))
  batched.setMatrixAt(hidden, new THREE.Matrix4().makeTranslation(0, 0, 0))
  batched.setMatrixAt(deleted, new THREE.Matrix4().makeTranslation(0, -0.55, 0))
  batched.setColorAt(left, new THREE.Color(1, 0, 0))
  batched.setColorAt(right, new THREE.Color(0, 1, 0))
  batched.setColorAt(hidden, new THREE.Color(0, 0, 1))
  batched.setColorAt(deleted, new THREE.Color(1, 1, 0))
  batched.setVisibleAt(hidden, false)
  batched.deleteInstance(deleted)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(batched)

  const rgba = renderRgba(scene, camera, { width: 96, height: 64 })
  const leftMean = meanRegion(rgba, 96, 64, 20, 28, 30, 36)
  const rightMean = meanRegion(rgba, 96, 64, 66, 28, 76, 36)
  const centerMean = meanRegion(rgba, 96, 64, 44, 28, 52, 36)
  const bottomMean = meanRegion(rgba, 96, 64, 44, 42, 52, 50)

  assert.ok(leftMean.r > leftMean.g + 80 && leftMean.r > leftMean.b + 80, `left BatchedMesh instance should render red (${leftMean.r}, ${leftMean.g}, ${leftMean.b})`)
  assert.ok(rightMean.g > rightMean.r + 80 && rightMean.g > rightMean.b + 80, `right BatchedMesh instance should render green (${rightMean.r}, ${rightMean.g}, ${rightMean.b})`)
  assert.ok(centerMean.b < 5 && centerMean.r < 5 && centerMean.g < 5, `hidden BatchedMesh instance should not render at center (${centerMean.r}, ${centerMean.g}, ${centerMean.b})`)
  assert.ok(bottomMean.r < 5 && bottomMean.g < 5 && bottomMean.b < 5, `deleted BatchedMesh instance should not render at bottom (${bottomMean.r}, ${bottomMean.g}, ${bottomMean.b})`)
})

test('BatchedMesh skips deleted geometry ranges', () => {
  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 1.2, -1.2, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const source = new THREE.PlaneGeometry(0.45, 0.45)
  const batched = new THREE.BatchedMesh(
    2,
    source.getAttribute('position').count * 2,
    source.index.count * 2,
    new THREE.MeshBasicMaterial({ color: 0xffffff }),
  )
  const activeGeometryId = batched.addGeometry(source)
  const inactiveGeometryId = batched.addGeometry(source.clone())
  const left = batched.addInstance(activeGeometryId)
  const right = batched.addInstance(inactiveGeometryId)
  batched.setMatrixAt(left, new THREE.Matrix4().makeTranslation(-0.55, 0, 0))
  batched.setMatrixAt(right, new THREE.Matrix4().makeTranslation(0.55, 0, 0))
  batched.setColorAt(left, new THREE.Color(1, 0, 0))
  batched.setColorAt(right, new THREE.Color(0, 1, 0))
  batched.deleteGeometry(inactiveGeometryId)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(batched)

  const rgba = renderRgba(scene, camera, { width: 96, height: 64 })
  const leftMean = meanRegion(rgba, 96, 64, 20, 28, 30, 36)
  const rightMean = meanRegion(rgba, 96, 64, 66, 28, 76, 36)

  assert.ok(leftMean.r > leftMean.g + 80, `active BatchedMesh geometry should render red (${leftMean.r} vs ${leftMean.g})`)
  assert.ok(rightMean.r < 5 && rightMean.g < 5 && rightMean.b < 5, `deleted BatchedMesh geometry should skip its visible instance (${rightMean.r}, ${rightMean.g}, ${rightMean.b})`)
})

test('BatchedMesh material arrays honor packed geometry groups', () => {
  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 1.2, -1.2, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const source = new THREE.PlaneGeometry(0.45, 0.45)
  const batched = new THREE.BatchedMesh(
    2,
    source.getAttribute('position').count * 2,
    source.index.count * 2,
    [
      new THREE.MeshBasicMaterial({ color: 0xff0000 }),
      new THREE.MeshBasicMaterial({ color: 0x00ff00 }),
    ],
  )
  const leftGeometryId = batched.addGeometry(source)
  const rightGeometryId = batched.addGeometry(source.clone())
  const left = batched.addInstance(leftGeometryId)
  const right = batched.addInstance(rightGeometryId)
  batched.setMatrixAt(left, new THREE.Matrix4().makeTranslation(-0.55, 0, 0))
  batched.setMatrixAt(right, new THREE.Matrix4().makeTranslation(0.55, 0, 0))

  const leftRange = batched.getGeometryRangeAt(leftGeometryId, {})
  const rightRange = batched.getGeometryRangeAt(rightGeometryId, {})
  batched.geometry.clearGroups()
  batched.geometry.addGroup(leftRange.start, leftRange.count, 0)
  batched.geometry.addGroup(rightRange.start, rightRange.count, 1)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(batched)

  const rgba = renderRgba(scene, camera, { width: 96, height: 64 })
  const leftMean = meanRegion(rgba, 96, 64, 20, 28, 30, 36)
  const rightMean = meanRegion(rgba, 96, 64, 66, 28, 76, 36)
  assert.ok(leftMean.r > leftMean.g + 80 && leftMean.r > leftMean.b + 80, `left BatchedMesh geometry group should use the red material (${leftMean.r}, ${leftMean.g}, ${leftMean.b})`)
  assert.ok(rightMean.g > rightMean.r + 80 && rightMean.g > rightMean.b + 80, `right BatchedMesh geometry group should use the green material (${rightMean.r}, ${rightMean.g}, ${rightMean.b})`)
})

test('BatchedMesh packed geometry groups clip to partial draw ranges', () => {
  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 1.2, -1.2, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const source = new THREE.BufferGeometry()
  source.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.9, -0.45, 0,
    -0.25, -0.45, 0,
    -0.25, 0.45, 0,
    -0.9, 0.45, 0,
    0.25, -0.45, 0,
    0.9, -0.45, 0,
    0.9, 0.45, 0,
    0.25, 0.45, 0,
  ]), 3))
  source.setIndex([
    0, 1, 2,
    0, 2, 3,
    4, 5, 6,
    4, 6, 7,
  ])

  const batched = new THREE.BatchedMesh(
    1,
    source.getAttribute('position').count,
    source.index.count,
    [
      new THREE.MeshBasicMaterial({ color: 0xff0000 }),
      new THREE.MeshBasicMaterial({ color: 0x00ff00 }),
    ],
  )
  const geometryId = batched.addGeometry(source)
  batched.addInstance(geometryId)
  batched.perObjectFrustumCulled = false

  const range = batched.getGeometryRangeAt(geometryId, {})
  batched.geometry.clearGroups()
  batched.geometry.addGroup(range.start, 6, 0)
  batched.geometry.addGroup(range.start + 6, 6, 1)

  // Force the active BatchedMesh draw range to start and end inside source groups.
  batched._geometryInfo[geometryId].start = range.start + 3
  batched._geometryInfo[geometryId].count = 6

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(batched)

  const rgba = renderRgba(scene, camera, { width: 96, height: 64 })
  const redPixels = countRegionPixels(rgba, 96, 64, 12, 14, 42, 34, (r, g, b) => r > g + 40 && r > b + 40)
  const greenPixels = countRegionPixels(rgba, 96, 64, 62, 30, 90, 52, (r, g, b) => g > r + 40 && g > b + 40)
  const clippedLeft = meanRegion(rgba, 96, 64, 30, 36, 40, 46)
  const clippedRight = meanRegion(rgba, 96, 64, 56, 18, 66, 28)

  assert.ok(redPixels > 40, `clipped left group should keep visible red pixels (${redPixels})`)
  assert.ok(greenPixels > 40, `clipped right group should keep visible green pixels (${greenPixels})`)
  assert.ok(clippedLeft.r < 5 && clippedLeft.g < 5 && clippedLeft.b < 5, `first skipped triangle should stay clipped out (${clippedLeft.r}, ${clippedLeft.g}, ${clippedLeft.b})`)
  assert.ok(clippedRight.r < 5 && clippedRight.g < 5 && clippedRight.b < 5, `last skipped triangle should stay clipped out (${clippedRight.r}, ${clippedRight.g}, ${clippedRight.b})`)
})

test('BatchedMesh per-object frustum culling honors geometry bounds', () => {
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  function renderCulling(perObjectFrustumCulled) {
    const source = new THREE.PlaneGeometry(2, 2)
    source.boundingSphere = new THREE.Sphere(new THREE.Vector3(4, 0, 0), 0.1)
    const batched = new THREE.BatchedMesh(
      1,
      source.getAttribute('position').count,
      source.index.count,
      new THREE.MeshBasicMaterial({ color: 0xff0000 }),
    )
    const geometryId = batched.addGeometry(source)
    const instanceId = batched.addInstance(geometryId)
    batched.setMatrixAt(instanceId, new THREE.Matrix4())
    batched.perObjectFrustumCulled = perObjectFrustumCulled

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(batched)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const culled = renderCulling(true)
  const uncullable = renderCulling(false)
  assert.ok(culled.r < 5 && culled.g < 5 && culled.b < 5, `cached out-of-frustum BatchedMesh bounds should cull the draw (${culled.r}, ${culled.g}, ${culled.b})`)
  assert.ok(uncullable.r > 200, `perObjectFrustumCulled=false should render the oversized batch draw (${uncullable.r})`)
})

test('invalid BatchedMesh perObjectFrustumCulled values fail clearly', () => {
  const camera = makeCamera()
  const source = new THREE.PlaneGeometry(1, 1)
  const batched = new THREE.BatchedMesh(
    1,
    source.getAttribute('position').count,
    source.index.count,
    new THREE.MeshBasicMaterial({ color: 0xffffff }),
  )
  const geometryId = batched.addGeometry(source)
  batched.addInstance(geometryId)
  batched.perObjectFrustumCulled = 'yes'

  const scene = new THREE.Scene()
  scene.add(batched)

  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /THREE\.BatchedMesh\.perObjectFrustumCulled must be a boolean/i,
  )
})

test('malformed BatchedMesh culling bounds fail clearly', () => {
  const camera = makeCamera()
  const source = new THREE.PlaneGeometry(1, 1)

  const cases = [
    ['container', 'sphere', /THREE\.BatchedMesh\._geometryInfo\[0\]\.boundingSphere must be a THREE\.Sphere-like object/i],
    ['center', { center: { x: Number.NaN, y: 0, z: 0 }, radius: 1 }, /THREE\.BatchedMesh\._geometryInfo\[0\]\.boundingSphere\.center must be a finite Vector3-like value/i],
    ['radius', { center: new THREE.Vector3(0, 0, 0), radius: -1 }, /THREE\.BatchedMesh\._geometryInfo\[0\]\.boundingSphere\.radius must be non-negative/i],
  ]

  for (const [label, boundingSphere, pattern] of cases) {
    const batched = new THREE.BatchedMesh(
      1,
      source.getAttribute('position').count,
      source.index.count,
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    )
    const geometryId = batched.addGeometry(source)
    batched.addInstance(geometryId)
    batched._geometryInfo[geometryId].boundingSphere = boundingSphere

    const scene = new THREE.Scene()
    scene.add(batched)

    assert.throws(
      () => renderRgba(scene, camera, { width: 32, height: 32 }),
      pattern,
      label,
    )
  }
})

test('malformed BatchedMesh geometry ranges fail clearly', () => {
  const camera = makeCamera()
  const source = new THREE.PlaneGeometry(1, 1)
  const makeScene = (range) => {
    const batched = new THREE.BatchedMesh(
      1,
      source.getAttribute('position').count,
      source.index.count,
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    )
    const geometryId = batched.addGeometry(source)
    batched.addInstance(geometryId)
    batched.getGeometryRangeAt = () => range
    const scene = new THREE.Scene()
    scene.add(batched)
    return scene
  }

  const cases = [
    ['missing range', null, /THREE\.BatchedMesh geometry range 0 is not readable/i],
    ['active flag', { start: 0, count: 6, active: 'yes' }, /THREE\.BatchedMesh\._geometryInfo\[0\]\.active must be a boolean/i],
    ['negative start', { start: -1, count: 6 }, /THREE\.BatchedMesh\._geometryInfo\[0\]\.start must be a non-negative integer/i],
    ['non-integer count', { start: 0, count: 1.5 }, /THREE\.BatchedMesh\._geometryInfo\[0\]\.count must be a non-negative integer/i],
    ['start past packed geometry', { start: 7, count: 0 }, /THREE\.BatchedMesh\._geometryInfo\[0\]\.start must be less than or equal to packed geometry count \(6\)/i],
    ['count past packed geometry', { start: 4, count: 3 }, /THREE\.BatchedMesh\._geometryInfo\[0\]\.count must fit within packed geometry count \(6\) from start 4/i],
  ]

  for (const [label, range, pattern] of cases) {
    assert.throws(
      () => renderRgba(makeScene(range), camera, { width: 32, height: 32 }),
      pattern,
      label,
    )
  }
})

test('invalid BatchedMesh sort controls fail clearly', () => {
  const camera = makeCamera()
  const source = new THREE.PlaneGeometry(1, 1)
  const makeScene = (instanceCount = 1) => {
    const batched = new THREE.BatchedMesh(
      instanceCount,
      source.getAttribute('position').count * instanceCount,
      source.index.count * instanceCount,
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    )
    const geometryId = batched.addGeometry(source)
    for (let i = 0; i < instanceCount; i += 1) {
      const instanceId = batched.addInstance(geometryId)
      batched.setMatrixAt(instanceId, new THREE.Matrix4().makeTranslation(i * 0.01, 0, 0))
    }
    const scene = new THREE.Scene()
    scene.add(batched)
    return { batched, scene }
  }

  {
    const { batched, scene } = makeScene()
    batched.sortObjects = 'yes'
    assert.throws(
      () => renderRgba(scene, camera, { width: 32, height: 32 }),
      /THREE\.BatchedMesh\.sortObjects must be a boolean/i,
    )
  }

  {
    const { batched, scene } = makeScene()
    batched.customSort = 'front'
    assert.throws(
      () => renderRgba(scene, camera, { width: 32, height: 32 }),
      /THREE\.BatchedMesh\.customSort must be a function or null/i,
    )
  }

  {
    const { batched, scene } = makeScene(2)
    batched.setCustomSort((list) => {
      list[0] = null
    })
    assert.throws(
      () => renderRgba(scene, camera, { width: 32, height: 32 }),
      /THREE\.BatchedMesh\.customSort list\[0\] must be an object/i,
    )
  }

  {
    const { batched, scene } = makeScene(2)
    batched.setCustomSort((list) => {
      list.pop()
    })
    assert.throws(
      () => renderRgba(scene, camera, { width: 32, height: 32 }),
      /THREE\.BatchedMesh\.customSort must keep 2 draw items; received 1/i,
    )
  }

  {
    const { batched, scene } = makeScene(2)
    batched.setCustomSort((list) => {
      list[0].index = 99
    })
    assert.throws(
      () => renderRgba(scene, camera, { width: 32, height: 32 }),
      /THREE\.BatchedMesh\.customSort returned unknown instance index 99/i,
    )
  }

  {
    const { batched, scene } = makeScene(2)
    batched.setCustomSort((list) => {
      list[1].index = list[0].index
    })
    assert.throws(
      () => renderRgba(scene, camera, { width: 32, height: 32 }),
      /THREE\.BatchedMesh\.customSort returned duplicate instance index/i,
    )
  }
})

test('BatchedMesh transparent sorting uses each geometry range center', () => {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const nearGeometry = new THREE.PlaneGeometry(2, 2)
  nearGeometry.translate(0, 0, 0.35)
  const farGeometry = new THREE.PlaneGeometry(2, 2)
  farGeometry.translate(0, 0, -0.35)

  const material = new THREE.MeshBasicMaterial({
    color: 0xffffff,
    depthWrite: false,
    transparent: true,
  })
  const batched = new THREE.BatchedMesh(
    2,
    nearGeometry.getAttribute('position').count + farGeometry.getAttribute('position').count,
    nearGeometry.index.count + farGeometry.index.count,
    material,
  )
  const nearGeometryId = batched.addGeometry(nearGeometry)
  const farGeometryId = batched.addGeometry(farGeometry)
  const near = batched.addInstance(nearGeometryId)
  const far = batched.addInstance(farGeometryId)
  batched.setMatrixAt(near, new THREE.Matrix4())
  batched.setMatrixAt(far, new THREE.Matrix4())
  batched.setColorAt(near, new THREE.Color(1, 0, 0))
  batched.setColorAt(far, new THREE.Color(0, 0, 1))

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(batched)

  const mean = meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 24, 24, 40, 40)
  assert.ok(mean.r > mean.b + 80, `near red BatchedMesh range should sort over far blue range (${mean.r} vs ${mean.b})`)
})

test('BatchedMesh transparent sorting scans material arrays', () => {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const nearGeometry = new THREE.PlaneGeometry(2, 2)
  nearGeometry.translate(0, 0, 0.35)
  const farGeometry = new THREE.PlaneGeometry(2, 2)
  farGeometry.translate(0, 0, -0.35)

  const materials = [
    new THREE.MeshBasicMaterial({ color: 0x00ff00 }),
    new THREE.MeshBasicMaterial({
      color: 0xffffff,
      depthWrite: false,
      transparent: true,
    }),
  ]
  const batched = new THREE.BatchedMesh(
    2,
    nearGeometry.getAttribute('position').count + farGeometry.getAttribute('position').count,
    nearGeometry.index.count + farGeometry.index.count,
    materials,
  )
  const nearGeometryId = batched.addGeometry(nearGeometry)
  const farGeometryId = batched.addGeometry(farGeometry)
  const near = batched.addInstance(nearGeometryId)
  const far = batched.addInstance(farGeometryId)
  batched.setMatrixAt(near, new THREE.Matrix4())
  batched.setMatrixAt(far, new THREE.Matrix4())
  batched.setColorAt(near, new THREE.Color(1, 0, 0))
  batched.setColorAt(far, new THREE.Color(0, 0, 1))

  const nearRange = batched.getGeometryRangeAt(nearGeometryId, {})
  const farRange = batched.getGeometryRangeAt(farGeometryId, {})
  batched.geometry.clearGroups()
  batched.geometry.addGroup(nearRange.start, nearRange.count, 1)
  batched.geometry.addGroup(farRange.start, farRange.count, 1)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(batched)

  const mean = meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 24, 24, 40, 40)
  assert.ok(mean.r > mean.b + 80, `transparent material-array BatchedMesh groups should sort near red over far blue (${mean.r} vs ${mean.b})`)
  assert.ok(mean.r > mean.g + 160, `unused opaque material-array entry should not dominate the grouped draw color (${mean.r} vs ${mean.g})`)
})

test('BatchedMesh sortObjects=false preserves instance draw order', () => {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const nearGeometry = new THREE.PlaneGeometry(2, 2)
  nearGeometry.translate(0, 0, 0.35)
  const farGeometry = new THREE.PlaneGeometry(2, 2)
  farGeometry.translate(0, 0, -0.35)

  const batched = new THREE.BatchedMesh(
    2,
    nearGeometry.getAttribute('position').count + farGeometry.getAttribute('position').count,
    nearGeometry.index.count + farGeometry.index.count,
    new THREE.MeshBasicMaterial({
      color: 0xffffff,
      depthWrite: false,
      transparent: true,
    }),
  )
  batched.sortObjects = false
  const nearGeometryId = batched.addGeometry(nearGeometry)
  const farGeometryId = batched.addGeometry(farGeometry)
  const near = batched.addInstance(nearGeometryId)
  const far = batched.addInstance(farGeometryId)
  batched.setMatrixAt(near, new THREE.Matrix4())
  batched.setMatrixAt(far, new THREE.Matrix4())
  batched.setColorAt(near, new THREE.Color(1, 0, 0))
  batched.setColorAt(far, new THREE.Color(0, 0, 1))

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(batched)

  const mean = meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 24, 24, 40, 40)
  assert.ok(mean.b > mean.r + 80, `BatchedMesh sortObjects=false should draw later far blue instance over near red (${mean.b} vs ${mean.r})`)
})

test('BatchedMesh customSort controls instance draw order', () => {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const nearGeometry = new THREE.PlaneGeometry(2, 2)
  nearGeometry.translate(0, 0, 0.35)
  const farGeometry = new THREE.PlaneGeometry(2, 2)
  farGeometry.translate(0, 0, -0.35)

  const batched = new THREE.BatchedMesh(
    2,
    nearGeometry.getAttribute('position').count + farGeometry.getAttribute('position').count,
    nearGeometry.index.count + farGeometry.index.count,
    new THREE.MeshBasicMaterial({
      color: 0xffffff,
      depthWrite: false,
      transparent: true,
    }),
  )
  const nearGeometryId = batched.addGeometry(nearGeometry)
  const farGeometryId = batched.addGeometry(farGeometry)
  const near = batched.addInstance(nearGeometryId)
  const far = batched.addInstance(farGeometryId)
  batched.setMatrixAt(near, new THREE.Matrix4())
  batched.setMatrixAt(far, new THREE.Matrix4())
  batched.setColorAt(near, new THREE.Color(1, 0, 0))
  batched.setColorAt(far, new THREE.Color(0, 0, 1))
  let callbackThis = null
  let callbackCamera = null
  let callbackList = null
  batched.setCustomSort(function (list, sortCamera) {
    callbackThis = this
    callbackCamera = sortCamera
    callbackList = list.map((item) => ({ ...item }))
    list.sort((a, b) => a.index - b.index)
  })

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(batched)

  const mean = meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 24, 24, 40, 40)
  assert.equal(callbackThis, batched)
  assert.equal(callbackCamera, camera)
  assert.deepEqual(callbackList.map((item) => item.index).sort(), [near, far].sort())
  assert.ok(callbackList.every((item) => item.count > 0 && item.start >= 0 && Number.isFinite(item.z)))
  assert.ok(mean.b > mean.r + 80, `BatchedMesh customSort should draw custom-ordered blue instance last (${mean.b} vs ${mean.r})`)
})

test('BatchedMesh renderer sort callbacks receive the source object', () => {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const source = new THREE.PlaneGeometry(1, 1)
  const material = new THREE.MeshBasicMaterial({
    color: 0xffffff,
    depthWrite: false,
    transparent: true,
  })
  const batched = new THREE.BatchedMesh(
    2,
    source.getAttribute('position').count,
    source.index.count,
    material,
  )
  const geometryId = batched.addGeometry(source)
  const left = batched.addInstance(geometryId)
  const right = batched.addInstance(geometryId)
  batched.setMatrixAt(left, new THREE.Matrix4().makeTranslation(-0.25, 0, 0))
  batched.setMatrixAt(right, new THREE.Matrix4().makeTranslation(0.25, 0, 0))

  const scene = new THREE.Scene()
  scene.add(batched)

  let calls = 0
  renderRgba(scene, camera, {
    width: 64,
    height: 64,
    transparentSort: (a, b) => {
      calls += 1
      assert.equal(a.object, batched)
      assert.equal(b.object, batched)
      assert.equal(a.material, material)
      assert.equal(b.material, material)
      return 0
    },
  })

  assert.ok(calls > 0, 'transparentSort should compare BatchedMesh-expanded draw items')
})

test('BatchedMesh sort callbacks receive packed geometry group render items', () => {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const source = new THREE.BufferGeometry()
  source.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -1, -1, 0,
    1, -1, 0,
    1, 1, 0,
    -1, 1, 0,
    -1, -1, 0,
    1, -1, 0,
    1, 1, 0,
    -1, 1, 0,
  ]), 3))
  source.setIndex([
    0, 1, 2,
    0, 2, 3,
    4, 5, 6,
    4, 6, 7,
  ])

  const materials = [
    new THREE.MeshBasicMaterial({ color: 0xff0000, transparent: true, depthTest: false, depthWrite: false }),
    new THREE.MeshBasicMaterial({ color: 0x0000ff, transparent: true, depthTest: false, depthWrite: false }),
  ]
  const batched = new THREE.BatchedMesh(
    1,
    source.getAttribute('position').count,
    source.index.count,
    materials,
  )
  const geometryId = batched.addGeometry(source)
  batched.addInstance(geometryId)

  const range = batched.getGeometryRangeAt(geometryId, {})
  batched.geometry.clearGroups()
  batched.geometry.addGroup(range.start, 6, 0)
  batched.geometry.addGroup(range.start + 6, 6, 1)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(batched)

  const seenGroups = new Set()
  const seenMaterials = new Set()
  let calls = 0
  const rgba = renderRgba(scene, camera, {
    width: 64,
    height: 64,
    transparentSort: (a, b) => {
      calls += 1
      assert.equal(a.object, batched)
      assert.equal(b.object, batched)
      assert.ok(a.group)
      assert.ok(b.group)
      seenGroups.add(a.group.materialIndex)
      seenGroups.add(b.group.materialIndex)
      seenMaterials.add(a.material)
      seenMaterials.add(b.material)
      return b.group.materialIndex - a.group.materialIndex
    },
  })

  assert.ok(calls > 0, 'transparentSort should compare BatchedMesh packed group items')
  assert.deepEqual([...seenGroups].sort(), [0, 1])
  assert.deepEqual([...seenMaterials].sort((a, b) => materials.indexOf(a) - materials.indexOf(b)), materials)
  const mean = meanRegion(rgba, 64, 64, 24, 24, 40, 40)
  assert.ok(mean.r > mean.b + 160, `custom group-aware BatchedMesh sort should draw red after blue (${mean.r} vs ${mean.b})`)
})

test('invalid output dimensions fail clearly', () => {
  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial({ color: 0xffffff })))
  const camera = makeCamera()

  assert.throws(
    () => getRenderer().render(scene, camera, { width: '64', height: 32 }),
    /options\.width must be a finite number/i,
  )
  assert.throws(
    () => getRenderer().render(scene, camera, { width: 32, height: 0 }),
    /options\.height must be a positive integer/i,
  )
  assert.throws(
    () => renderToTarget(scene, camera, { width: Number.NaN, height: 32 }),
    /target\.width must be a finite number/i,
  )

  const userDataCamera = makeCamera()
  userDataCamera.userData.width = 32.5
  assert.throws(
    () => getRenderer().render(scene, userDataCamera, { format: 'rgba' }),
    /camera\.userData\.width must be a positive integer/i,
  )

  const userDataContainerCamera = makeCamera()
  userDataContainerCamera.userData = 'size'
  assert.throws(
    () => getRenderer().render(scene, userDataContainerCamera, { format: 'rgba' }),
    /camera\.userData must be an object/i,
  )

  const invalidAspectCamera = makeCamera()
  invalidAspectCamera.aspect = Number.NaN
  assert.throws(
    () => getRenderer().render(scene, invalidAspectCamera, { width: 32, format: 'rgba' }),
    /camera\.aspect must be a finite number/i,
  )

  const zeroAspectCamera = makeCamera()
  zeroAspectCamera.aspect = 0
  assert.throws(
    () => getRenderer().render(scene, zeroAspectCamera, { height: 32, format: 'rgba' }),
    /camera\.aspect must be positive/i,
  )
})

test('invalid camera clipping distances fail clearly', () => {
  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial({ color: 0xffffff })))
  const camera = makeCamera()
  camera.near = Number.NaN

  assert.throws(
    () => getRenderer().render(scene, camera, { width: 32, height: 32 }),
    /camera\.near must be a finite number/i,
  )

  camera.near = 0
  assert.throws(
    () => getRenderer().render(scene, camera, { width: 32, height: 32 }),
    /camera\.near must be positive/i,
  )

  camera.near = 0.01
  camera.far = 'deep'
  assert.throws(
    () => getRenderer().render(scene, camera, { width: 32, height: 32 }),
    /camera\.far must be a finite number/i,
  )

  camera.far = -1
  assert.throws(
    () => getRenderer().render(scene, camera, { width: 32, height: 32 }),
    /camera\.far must be positive/i,
  )

  camera.near = 10
  camera.far = 1
  assert.throws(
    () => getRenderer().render(scene, camera, { width: 32, height: 32 }),
    /camera\.far must be greater than camera\.near/i,
  )

  camera.near = 0.01
  camera.far = 100
  camera.projectionMatrix.elements[0] = Number.NaN
  assert.throws(
    () => getRenderer().render(scene, camera, { width: 32, height: 32 }),
    /camera\.projectionMatrix\.elements\[0\] must be a finite number/i,
  )

  const missingMatrixCamera = makeCamera()
  missingMatrixCamera.projectionMatrix = null
  assert.throws(
    () => getRenderer().render(scene, missingMatrixCamera, { width: 32, height: 32 }),
    /THREE\.Camera must have projectionMatrix and matrixWorldInverse/i,
  )

  const projectionContainerCamera = makeCamera()
  projectionContainerCamera.projectionMatrix = { elements: [1, 2, 3] }
  assert.throws(
    () => getRenderer().render(scene, projectionContainerCamera, { width: 32, height: 32 }),
    /camera\.projectionMatrix must be a THREE\.Matrix4/i,
  )

  const viewMatrixCamera = makeCamera()
  viewMatrixCamera.matrixWorldInverse.elements[4] = Number.NaN
  assert.throws(
    () => getRenderer().render(scene, viewMatrixCamera, { width: 32, height: 32 }),
    /camera\.matrixWorldInverse\.elements\[4\] must be a finite number/i,
  )

  const worldMatrixCamera = makeCamera()
  worldMatrixCamera.updateMatrixWorld = () => {}
  worldMatrixCamera.matrixWorld.elements[12] = Number.NaN
  assert.throws(
    () => getRenderer().render(scene, worldMatrixCamera, { width: 32, height: 32 }),
    /camera\.matrixWorld\.elements\[12\] must be a finite number/i,
  )

  const matrixContainerCamera = makeCamera()
  matrixContainerCamera.updateMatrixWorld = () => {}
  matrixContainerCamera.matrixWorld = 'world'
  assert.throws(
    () => getRenderer().render(scene, matrixContainerCamera, { width: 32, height: 32 }),
    /camera\.matrixWorld must be a THREE\.Matrix4/i,
  )
})

function makeLayeredArrayCamera(width = 64, height = 64) {
  const leftCamera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  leftCamera.position.set(0, 0, 3)
  leftCamera.lookAt(0, 0, 0)
  leftCamera.layers.set(1)
  leftCamera.viewport = new THREE.Vector4(0, 0, width / 2, height)

  const rightCamera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  rightCamera.position.set(0, 0, 3)
  rightCamera.lookAt(0, 0, 0)
  rightCamera.layers.set(2)
  rightCamera.viewport = new THREE.Vector4(width / 2, 0, width / 2, height)

  return new THREE.ArrayCamera([leftCamera, rightCamera])
}

function makeLayeredSplitScene() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 1)
  const red = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ color: 0xff0000 }))
  red.layers.set(1)
  scene.add(red)
  const green = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ color: 0x00ff00 }))
  green.layers.set(2)
  scene.add(green)
  return scene
}

function makeCubeCaptureScene() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const addPlane = (position, rotation, color) => {
    const plane = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide }),
    )
    plane.position.set(...position)
    plane.rotation.set(...rotation)
    scene.add(plane)
  }
  addPlane([2, 0, 0], [0, Math.PI / 2, 0], 0xff0000)
  addPlane([-2, 0, 0], [0, Math.PI / 2, 0], 0x00ff00)
  addPlane([0, 2, 0], [Math.PI / 2, 0, 0], 0x0000ff)
  addPlane([0, -2, 0], [Math.PI / 2, 0, 0], 0xffff00)
  addPlane([0, 0, 2], [0, 0, 0], 0xff00ff)
  addPlane([0, 0, -2], [0, 0, 0], 0x00ffff)
  return scene
}

test('ArrayCamera renders sub-camera viewports', () => {
  const scene = makeLayeredSplitScene()
  const arrayCamera = makeLayeredArrayCamera()

  const rgba = renderRgba(scene, arrayCamera, { width: 64, height: 64 })
  const left = meanRegion(rgba, 64, 64, 8, 20, 24, 44)
  const right = meanRegion(rgba, 64, 64, 40, 20, 56, 44)
  assert.ok(left.r > left.g + 80 && left.r > left.b + 80, `left ArrayCamera viewport should render the red layer (${left.r}, ${left.g}, ${left.b})`)
  assert.ok(right.g > right.r + 80 && right.g > right.b + 80, `right ArrayCamera viewport should render the green layer (${right.r}, ${right.g}, ${right.b})`)

  const target = { texture: {}, depthTexture: {} }
  renderToTarget(scene, arrayCamera, target, { width: 64, height: 64 })
  assert.equal(target.data.length, 64 * 64 * 4)
  assert.equal(target.depthTexture.image.data.length, 64 * 64 * 4)
  const targetLeft = meanRegion(target.data, 64, 64, 8, 20, 24, 44)
  const targetRight = meanRegion(target.data, 64, 64, 40, 20, 56, 44)
  const depthLeft = meanRegion(target.depthTexture.image.data, 64, 64, 8, 20, 24, 44)
  const depthRight = meanRegion(target.depthTexture.image.data, 64, 64, 40, 20, 56, 44)
  assert.ok(targetLeft.r > targetLeft.g + 80, `target left ArrayCamera viewport should render red (${targetLeft.r}, ${targetLeft.g})`)
  assert.ok(targetRight.g > targetRight.r + 80, `target right ArrayCamera viewport should render green (${targetRight.g}, ${targetRight.r})`)
  assert.ok(depthLeft.r > 0 && depthRight.r > 0, `ArrayCamera depth target should include both viewports (${depthLeft.r}, ${depthRight.r})`)
})

test('ArrayCamera supports PNG output', () => {
  const scene = makeLayeredSplitScene()
  const arrayCamera = makeLayeredArrayCamera()
  assertValidPng(getRenderer().render(scene, arrayCamera, { width: 64, height: 64 }), { width: 64, height: 64 })
})

test('ArrayCamera object-id target merges sub-camera metadata', () => {
  const scene = makeLayeredSplitScene()
  const arrayCamera = makeLayeredArrayCamera()
  const [red, green] = scene.children
  const target = { texture: {} }

  renderToTarget(scene, arrayCamera, target, { width: 64, height: 64, renderMode: 'object-id' })

  const redEncoded = red.id + 1
  const greenEncoded = green.id + 1
  const left = meanRegion(target.data, 64, 64, 8, 20, 24, 44)
  const right = meanRegion(target.data, 64, 64, 40, 20, 56, 44)
  assertRgbClose(left, objectIdBytes(redEncoded), 'left ArrayCamera object id')
  assertRgbClose(right, objectIdBytes(greenEncoded), 'right ArrayCamera object id')
  assert.equal(target.objectIdEntries.length, 2)
  assert.equal(target.objectIdMap[String(redEncoded)].id, red.id)
  assert.equal(target.objectIdMap[String(greenEncoded)].id, green.id)
})

test('malformed ArrayCamera sub-camera containers fail clearly', () => {
  const scene = makeLayeredSplitScene()
  const arrayCamera = makeLayeredArrayCamera()

  arrayCamera.cameras = 'bad'
  assert.throws(
    () => renderRgba(scene, arrayCamera, { width: 64, height: 64 }),
    /THREE\.ArrayCamera\.cameras must be an array/i,
  )

  arrayCamera.cameras = []
  assert.throws(
    () => renderRgba(scene, arrayCamera, { width: 64, height: 64 }),
    /THREE\.ArrayCamera requires at least one sub-camera/i,
  )

  arrayCamera.cameras = [null]
  assert.throws(
    () => renderRgba(scene, arrayCamera, { width: 64, height: 64 }),
    /THREE\.ArrayCamera\.cameras\[0\] must be a THREE\.Camera/i,
  )

  arrayCamera.cameras = [new THREE.ArrayCamera([])]
  assert.throws(
    () => renderRgba(scene, arrayCamera, { width: 64, height: 64 }),
    /THREE\.ArrayCamera\.cameras\[0\] cannot be a THREE\.ArrayCamera/i,
  )

  const matrixArrayCamera = makeLayeredArrayCamera()
  matrixArrayCamera.cameras[0].matrixWorldInverse.elements[1] = Number.NaN
  assert.throws(
    () => renderRgba(scene, matrixArrayCamera, { width: 64, height: 64 }),
    /THREE\.ArrayCamera\.cameras\[0\]\.matrixWorldInverse\.elements\[1\] must be a finite number/i,
  )
})

test('CubeCamera renders cube target faces', () => {
  const scene = makeCubeCaptureScene()
  const cubeTarget = new THREE.WebGLCubeRenderTarget(32)
  const cubeCamera = new THREE.CubeCamera(0.01, 100, cubeTarget)
  assertValidPng(getRenderer().render(scene, cubeCamera, { width: 32, height: 32 }), { width: 32, height: 32 })

  cubeTarget.depthTexture = {
    type: THREE.FloatType,
    source: { data: Array.from({ length: 6 }, () => ({})) },
  }
  const returned = renderToTarget(scene, cubeCamera, cubeTarget)
  assert.equal(returned, cubeTarget)
  assert.equal(cubeTarget.texture.image.length, 6)

  const px = meanRegion(cubeTarget.texture.image[0].data, 32, 32, 12, 12, 20, 20)
  const nx = meanRegion(cubeTarget.texture.image[1].data, 32, 32, 12, 12, 20, 20)
  const py = meanRegion(cubeTarget.texture.image[2].data, 32, 32, 12, 12, 20, 20)
  const pz = meanRegion(cubeTarget.texture.image[4].data, 32, 32, 12, 12, 20, 20)
  assert.ok(px.r > px.g + 80 && px.r > px.b + 80, `+X face should capture red (${px.r}, ${px.g}, ${px.b})`)
  assert.ok(nx.g > nx.r + 60 && nx.g > nx.b + 60, `-X face should capture green (${nx.r}, ${nx.g}, ${nx.b})`)
  assert.ok(py.b > py.r + 80 && py.b > py.g + 80, `+Y face should capture blue (${py.r}, ${py.g}, ${py.b})`)
  assert.ok(pz.r > pz.g + 80 && pz.b > pz.g + 80, `+Z face should capture magenta (${pz.r}, ${pz.g}, ${pz.b})`)
  assert.notStrictEqual(cubeTarget.texture.image[0], cubeTarget.texture.image[1])
  assert.strictEqual(cubeTarget.texture.source.data, cubeTarget.texture.image)

  assert.equal(cubeTarget.depthTexture.image.length, 6)
  assert.ok(cubeTarget.depthTexture.image[0].data instanceof Float32Array, 'cube depth face should use Float32Array data')
  assert.equal(cubeTarget.depthTexture.image[0].data.length, 32 * 32)
  assert.strictEqual(cubeTarget.depthTexture.source.data, cubeTarget.depthTexture.image)
  const depthPx = meanScalarRegion(cubeTarget.depthTexture.image[0].data, 32, 32, 12, 12, 20, 20)
  assert.ok(depthPx > 0 && depthPx <= 1, `cube depth face should contain normalized depth (${depthPx})`)
})

test('CubeCamera object-id target includes reverse lookup metadata', () => {
  const scene = makeCubeCaptureScene()
  const cubeTarget = new THREE.WebGLCubeRenderTarget(32)
  const cubeCamera = new THREE.CubeCamera(0.01, 100, cubeTarget)
  const [positiveX, negativeX] = scene.children

  renderToTarget(scene, cubeCamera, cubeTarget, { renderMode: 'object-id' })

  const positiveXEncoded = positiveX.id + 1
  const negativeXEncoded = negativeX.id + 1
  const px = meanRegion(cubeTarget.texture.image[0].data, 32, 32, 12, 12, 20, 20)
  const nx = meanRegion(cubeTarget.texture.image[1].data, 32, 32, 12, 12, 20, 20)
  assertRgbClose(px, objectIdBytes(positiveXEncoded), '+X CubeCamera object id')
  assertRgbClose(nx, objectIdBytes(negativeXEncoded), '-X CubeCamera object id')
  assert.ok(cubeTarget.objectIdEntries.length >= 2, 'cube object-id target should expose rendered object metadata')
  assert.equal(cubeTarget.objectIdMap[String(positiveXEncoded)].id, positiveX.id)
  assert.equal(cubeTarget.objectIdMap[String(negativeXEncoded)].id, negativeX.id)

  renderToTarget(scene, cubeCamera, cubeTarget)
  assert.equal(cubeTarget.objectIdEntries, undefined)
  assert.equal(cubeTarget.objectIdMap, undefined)
})

test('CubeCamera renders active mip target faces', () => {
  const scene = makeCubeCaptureScene()
  const cubeTarget = new THREE.WebGLCubeRenderTarget(32)
  const cubeCamera = new THREE.CubeCamera(0.01, 100, cubeTarget)

  renderToTarget(scene, cubeCamera, cubeTarget)
  const basePositiveX = cubeTarget.texture.image[0]

  cubeCamera.activeMipmapLevel = 1
  cubeTarget.depthTexture = { type: THREE.UnsignedShortType, mipmaps: [] }
  const returned = renderToTarget(scene, cubeCamera, cubeTarget)
  assert.equal(returned, cubeTarget)
  assert.equal(cubeTarget.width, 32)
  assert.equal(cubeTarget.height, 32)
  assert.equal(cubeTarget.data.length, 16 * 16 * 4)
  assert.strictEqual(cubeTarget.texture.image[0], basePositiveX)
  assert.equal(cubeTarget.texture.image[0].data.length, 32 * 32 * 4)

  const mip = cubeTarget.texture.mipmaps[1]
  assert.equal(mip.image.length, 6)
  assert.equal(mip.image[0].width, 16)
  assert.equal(mip.image[0].height, 16)
  assert.equal(mip.image[0].data.length, 16 * 16 * 4)
  assert.ok(cubeTarget.texture.pmremVersion > 0, 'cube target texture should request PMREM refresh')

  const px = meanRegion(mip.image[0].data, 16, 16, 5, 5, 11, 11)
  assert.ok(px.r > px.g + 80 && px.r > px.b + 80, `+X mip face should capture red (${px.r}, ${px.g}, ${px.b})`)

  const depthMip = cubeTarget.depthTexture.mipmaps[1]
  assert.equal(depthMip.image.length, 6)
  assert.ok(depthMip.image[0].data instanceof Uint16Array, 'cube depth mip face should use Uint16Array data')
  assert.equal(depthMip.image[0].data.length, 16 * 16)
  const depthPx = meanScalarRegion(depthMip.image[0].data, 16, 16, 5, 5, 11, 11)
  assert.ok(depthPx > 0, `cube depth mip face should contain scalar depth (${depthPx})`)
})

test('CubeCamera active mip target faces honor scissor clipping', () => {
  const scene = makeCubeCaptureScene()
  const cubeTarget = new THREE.WebGLCubeRenderTarget(32)
  cubeTarget.scissorTest = true
  cubeTarget.scissor = { x: 4, y: 4, width: 16, height: 16 }
  const cubeCamera = new THREE.CubeCamera(0.01, 100, cubeTarget)
  cubeCamera.activeMipmapLevel = 1

  renderToTarget(scene, cubeCamera, cubeTarget)

  const face = cubeTarget.texture.mipmaps[1].image[0].data
  const inside = meanRegion(face, 16, 16, 6, 6, 10, 10)
  const outside = meanRegion(face, 16, 16, 0, 6, 3, 10)
  assert.ok(inside.r > 180 && inside.g < 40 && inside.b < 40, `cube scissor should keep red inside the active mip rectangle (${inside.r}, ${inside.g}, ${inside.b})`)
  assert.ok(outside.r < 20 && outside.g < 20 && outside.b < 20, `cube scissor should leave pixels outside the active mip rectangle clear (${outside.r}, ${outside.g}, ${outside.b})`)
})

test('CubeCamera captured target textures can be reused as cube inputs', () => {
  const captureTarget = {}
  const cubeCamera = new THREE.CubeCamera(0.01, 100, new THREE.WebGLCubeRenderTarget(32))
  renderToTarget(makeCubeCaptureScene(), cubeCamera, captureTarget, { width: 32, height: 32 })
  assert.equal(captureTarget.texture.isCubeTexture, true)
  assert.strictEqual(captureTarget.texture.source.data, captureTarget.texture.image)

  const backgroundScene = new THREE.Scene()
  backgroundScene.background = captureTarget.texture
  const backgroundCamera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  backgroundCamera.position.set(0, 0, 0)
  backgroundCamera.lookAt(new THREE.Vector3(1, 0, 0))
  const background = meanRegion(renderRgba(backgroundScene, backgroundCamera, {
    width: 64,
    height: 64,
    outputColorSpace: THREE.LinearSRGBColorSpace,
  }), 64, 64, 28, 28, 36, 36)
  assert.ok(background.r > background.g + 80 && background.r > background.b + 80, `captured +X cube background should render red (${background.r}, ${background.g}, ${background.b})`)

  function makeEnvironmentScene(environment) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    if (environment) {
      scene.environment = environment
      scene.environmentIntensity = 4
    }
    scene.add(new THREE.Mesh(
      new THREE.SphereGeometry(1, 32, 32),
      new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0.2 }),
    ))
    return scene
  }

  const environmentCamera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  environmentCamera.position.set(0, 0, 3)
  environmentCamera.lookAt(0, 0, 0)
  const noEnvironment = renderRgba(makeEnvironmentScene(null), environmentCamera, { width: 64, height: 64 })
  const withEnvironment = renderRgba(makeEnvironmentScene(captureTarget.texture), environmentCamera, { width: 64, height: 64 })
  const diff = meanAbsDiff(noEnvironment, withEnvironment)
  assert.ok(diff > 1, `captured cube environment should affect metallic IBL, diff=${diff.toFixed(3)}`)
})

test('CubeCamera malformed render targets fail clearly', () => {
  const scene = makeCubeCaptureScene()
  const cubeCamera = new THREE.CubeCamera(0.01, 100, new THREE.WebGLCubeRenderTarget(32))

  cubeCamera.renderTarget = 'bad'
  assert.throws(
    () => renderRgba(scene, cubeCamera, { width: 32, height: 32 }),
    /THREE\.CubeCamera renderTarget must be a target-like object/i,
  )

  assert.throws(
    () => render(scene, cubeCamera, { width: 32, height: 32, target: 'bad' }),
    /options\.target must be a target-like object/i,
  )

  const colorMipTarget = new THREE.WebGLCubeRenderTarget(32)
  const colorMipCamera = new THREE.CubeCamera(0.01, 100, colorMipTarget)
  colorMipCamera.activeMipmapLevel = 1
  colorMipTarget.texture.mipmaps = 'bad'
  assert.throws(
    () => renderToTarget(scene, colorMipCamera, colorMipTarget),
    /target\.texture\.mipmaps must be an array of image-like objects/i,
  )

  const depthMipTarget = new THREE.WebGLCubeRenderTarget(32)
  const depthMipCamera = new THREE.CubeCamera(0.01, 100, depthMipTarget)
  depthMipCamera.activeMipmapLevel = 1
  depthMipTarget.depthTexture = { type: THREE.UnsignedShortType, mipmaps: 'bad' }
  assert.throws(
    () => renderToTarget(scene, depthMipCamera, depthMipTarget),
    /target\.depthTexture\.mipmaps must be an array of image-like objects/i,
  )
})

test('malformed CubeCamera child camera containers fail clearly', () => {
  const scene = makeCubeCaptureScene()
  const cubeCamera = new THREE.CubeCamera(0.01, 100, new THREE.WebGLCubeRenderTarget(32))

  cubeCamera.children = 'bad'
  assert.throws(
    () => renderRgba(scene, cubeCamera, { width: 32, height: 32 }),
    /THREE\.CubeCamera\.children must be an array/i,
  )

  cubeCamera.children = []
  assert.throws(
    () => renderRgba(scene, cubeCamera, { width: 32, height: 32 }),
    /THREE\.CubeCamera requires six internal perspective cameras/i,
  )

  cubeCamera.children = [null, null, null, null, null, null]
  assert.throws(
    () => renderRgba(scene, cubeCamera, { width: 32, height: 32 }),
    /THREE\.CubeCamera\.children\[0\] must be a THREE\.Camera/i,
  )

  cubeCamera.children = [
    new THREE.CubeCamera(0.01, 100, new THREE.WebGLCubeRenderTarget(16)),
    null,
    null,
    null,
    null,
    null,
  ]
  assert.throws(
    () => renderRgba(scene, cubeCamera, { width: 32, height: 32 }),
    /THREE\.CubeCamera\.children\[0\] cannot be a THREE\.CubeCamera/i,
  )

  const matrixCubeCamera = new THREE.CubeCamera(0.01, 100, new THREE.WebGLCubeRenderTarget(32))
  matrixCubeCamera.children[0].projectionMatrix = { elements: [1, 2, 3] }
  assert.throws(
    () => renderRgba(scene, matrixCubeCamera, { width: 32, height: 32 }),
    /THREE\.CubeCamera\.children\[0\]\.projectionMatrix must be a THREE\.Matrix4/i,
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

test('renderMode mask outputs white visible geometry over black', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(1, 0, 0)
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(1.2, 1.2), new THREE.MeshBasicMaterial({ color: 0x0088ff })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 64, height: 64, renderMode: 'mask' })
  const center = meanRegion(rgba, 64, 64, 28, 28, 36, 36)
  const corner = meanRegion(rgba, 64, 64, 0, 0, 8, 8)
  assert.ok(center.r > 250 && center.g > 250 && center.b > 250, `mask center should be white (${center.r}, ${center.g}, ${center.b})`)
  assert.ok(corner.r < 2 && corner.g < 2 && corner.b < 2, `mask background should be black (${corner.r}, ${corner.g}, ${corner.b})`)
})

test('renderMode object-id outputs stable per-object RGB IDs', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(1, 0, 0)
  const left = new THREE.Mesh(new THREE.PlaneGeometry(0.75, 0.8), new THREE.MeshBasicMaterial({ color: 0xff0000 }))
  const right = new THREE.Mesh(new THREE.PlaneGeometry(0.75, 0.8), new THREE.MeshBasicMaterial({ color: 0x00ff00 }))
  left.position.x = -0.5
  right.position.x = 0.5
  scene.add(left, right)

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 64, height: 64, renderMode: 'object-id' })
  const leftMean = meanRegion(rgba, 64, 64, 16, 28, 23, 36)
  const rightMean = meanRegion(rgba, 64, 64, 41, 28, 48, 36)
  const background = meanRegion(rgba, 64, 64, 0, 0, 8, 8)
  assertRgbClose(leftMean, objectIdBytes(left.id + 1), 'left object id')
  assertRgbClose(rightMean, objectIdBytes(right.id + 1), 'right object id')
  assert.notDeepEqual(objectIdBytes(left.id + 1), objectIdBytes(right.id + 1))
  assert.ok(background.r < 2 && background.g < 2 && background.b < 2, `object-id background should be black (${background.r}, ${background.g}, ${background.b})`)
})

test('renderMode normal outputs view-space normal colors', () => {
  function makeScene(material) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 1.2), material)
    mesh.rotation.y = Math.PI * 0.25
    scene.add(mesh)
    return scene
  }

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const renderModeNormals = renderRgba(
    makeScene(new THREE.MeshBasicMaterial({ color: 0xff0000 })),
    camera,
    { width: 64, height: 64, renderMode: 'normal' },
  )
  const materialNormals = renderRgba(
    makeScene(new THREE.MeshNormalMaterial()),
    camera,
    { width: 64, height: 64 },
  )

  const diff = meanAbsDiff(renderModeNormals, materialNormals)
  const center = meanRegion(renderModeNormals, 64, 64, 28, 28, 36, 36)
  const background = meanRegion(renderModeNormals, 64, 64, 0, 0, 8, 8)
  assert.ok(diff < 1, `renderMode normal should match MeshNormalMaterial output (diff=${diff.toFixed(2)})`)
  assert.ok(center.r > 120 && center.b > 200, `normal pass center should encode tilted view normal (${center.r}, ${center.g}, ${center.b})`)
  assert.ok(background.r < 2 && background.g < 2 && background.b < 2, `normal background should be black (${background.r}, ${background.g}, ${background.b})`)
})

test('renderMode object-id target includes reverse lookup metadata', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const left = new THREE.Mesh(new THREE.PlaneGeometry(0.75, 0.8), new THREE.MeshBasicMaterial({ color: 0xff0000 }))
  const right = new THREE.Mesh(new THREE.PlaneGeometry(0.75, 0.8), new THREE.MeshBasicMaterial({ color: 0x00ff00 }))
  left.position.x = -0.5
  right.position.x = 0.5
  scene.add(left, right)

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const target = { texture: {} }
  renderToTarget(scene, camera, target, { width: 64, height: 64, renderMode: 'object-id' })
  const leftEncoded = left.id + 1
  const rightEncoded = right.id + 1
  assert.equal(target.objectIdEntries.length, 2)
  assert.deepEqual(target.objectIdMap[String(leftEncoded)].rgb, objectIdBytes(leftEncoded))
  assert.deepEqual(target.objectIdMap[String(rightEncoded)].rgb, objectIdBytes(rightEncoded))
  assert.equal(target.objectIdMap[String(leftEncoded)].id, left.id)
  assert.equal(target.objectIdMap[String(rightEncoded)].hex, `#${rightEncoded.toString(16).padStart(6, '0')}`)

  const optionsTarget = { texture: {} }
  const returned = renderRgba(scene, camera, {
    width: 64,
    height: 64,
    renderMode: 'object-id',
    target: optionsTarget,
  })
  assert.equal(optionsTarget.data, returned)
  assert.equal(optionsTarget.objectIdEntries.length, 2)
  assert.equal(optionsTarget.objectIdMap[String(leftEncoded)].id, left.id)
  assert.equal(optionsTarget.objectIdMap[String(rightEncoded)].id, right.id)

  renderToTarget(scene, camera, target, { width: 64, height: 64 })
  assert.equal(target.objectIdEntries, undefined)
  assert.equal(target.objectIdMap, undefined)
})

test('renderMode auxiliary passes bypass post-processing', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(1, 0, 0)
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 1.2), new THREE.MeshBasicMaterial({ color: 0x0088ff }))
  mesh.rotation.y = Math.PI * 0.2
  scene.add(mesh)

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  for (const renderMode of ['mask', 'object-id', 'normal']) {
    const base = renderRgba(scene, camera, { width: 64, height: 64, renderMode })
    const processed = renderRgba(scene, camera, {
      width: 64,
      height: 64,
      renderMode,
      postProcessing: {
        exposure: 4,
        contrast: 4,
        grayscale: true,
        invert: true,
        saturation: 0,
        vignette: 1,
      },
    })
    assert.deepEqual(processed, base, `${renderMode} should ignore post-processing effects`)
  }
})

test('renderMode auxiliary passes preserve texture alpha cutouts', () => {
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const makeBaseAlphaMaterial = (alpha) => new THREE.MeshBasicMaterial({
    map: solidTexture(255, 255, 255, alpha),
    alphaTest: 0.5,
  })
  const makeAlphaMapMaterial = (green) => new THREE.MeshBasicMaterial({
    alphaMap: solidTexture(255, green, 255),
    alphaTest: 0.5,
  })
  const cases = [
    ['base texture alpha', () => makeBaseAlphaMaterial(0), () => makeBaseAlphaMaterial(255)],
    ['alphaMap green channel', () => makeAlphaMapMaterial(0), () => makeAlphaMapMaterial(255)],
  ]

  for (const [label, makeDiscardedMaterial, makeVisibleMaterial] of cases) {
    for (const renderMode of ['mask', 'object-id', 'normal']) {
      const scene = new THREE.Scene()
      const discarded = new THREE.Mesh(new THREE.PlaneGeometry(0.75, 0.8), makeDiscardedMaterial())
      const visible = new THREE.Mesh(new THREE.PlaneGeometry(0.75, 0.8), makeVisibleMaterial())
      discarded.position.x = -0.5
      visible.position.x = 0.5
      scene.add(discarded, visible)

      const rgba = renderRgba(scene, camera, { width: 64, height: 64, renderMode })
      const leftMean = meanRegion(rgba, 64, 64, 16, 28, 23, 36)
      const rightMean = meanRegion(rgba, 64, 64, 41, 28, 48, 36)
      assert.ok(leftMean.r < 2 && leftMean.g < 2 && leftMean.b < 2, `${renderMode} should discard ${label} pixels (${leftMean.r}, ${leftMean.g}, ${leftMean.b})`)

      if (renderMode === 'mask') {
        assert.ok(rightMean.r > 250 && rightMean.g > 250 && rightMean.b > 250, `mask should keep opaque ${label} pixels (${rightMean.r}, ${rightMean.g}, ${rightMean.b})`)
      } else if (renderMode === 'object-id') {
        assertRgbClose(rightMean, objectIdBytes(visible.id + 1), `object-id should keep opaque ${label} pixels`)
      } else {
        assert.ok(rightMean.r > 120 && rightMean.g > 120 && rightMean.b > 250, `normal should keep opaque ${label} pixels (${rightMean.r}, ${rightMean.g}, ${rightMean.b})`)
      }
    }
  }
})

test('renderMode auxiliary passes preserve alphaHash cutouts', () => {
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  function renderAuxiliary(renderMode, alphaHash) {
    const scene = new THREE.Scene()
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(1.2, 1.2),
      new THREE.MeshBasicMaterial({
        alphaHash,
        color: 0xffffff,
        opacity: alphaHash ? 0.35 : 1,
      }),
    ))
    return renderRgba(scene, camera, { width: 64, height: 64, renderMode })
  }

  for (const renderMode of ['mask', 'object-id', 'normal']) {
    const opaque = renderAuxiliary(renderMode, false)
    const hashed = renderAuxiliary(renderMode, true)
    const visiblePixel = (r, g, b) => r > 0 || g > 0 || b > 0
    const opaquePixels = countRegionPixels(opaque, 64, 64, 20, 20, 44, 44, visiblePixel)
    const hashedPixels = countRegionPixels(hashed, 64, 64, 20, 20, 44, 44, visiblePixel)

    assert.ok(opaquePixels > 520, `${renderMode} opaque pass should fill the sampled region (${opaquePixels})`)
    assert.ok(hashedPixels > 40, `${renderMode} alphaHash pass should retain some visible pixels (${hashedPixels})`)
    assert.ok(hashedPixels < opaquePixels - 120, `${renderMode} alphaHash pass should discard visible pixels (${hashedPixels} vs ${opaquePixels})`)
  }
})

test('invalid renderMode values fail clearly', () => {
  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial()))
  assert.throws(
    () => renderRgba(scene, makeCamera(), { width: 32, height: 32, renderMode: 'normals' }),
    /options\.renderMode must be "color", "mask", "object-id", or "normal"/i,
  )
})

test('invalid material alphaTest values fail clearly', () => {
  const cases = [
    ['mesh', () => {
      const material = new THREE.MeshBasicMaterial({ color: 0xffffff })
      material.alphaTest = 'cutout'
      return new THREE.Mesh(new THREE.PlaneGeometry(1, 1), material)
    }],
    ['line', () => {
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-0.5, 0, 0, 0.5, 0, 0]), 3))
      const material = new THREE.LineBasicMaterial({ color: 0xffffff })
      material.alphaTest = Number.NaN
      return new THREE.Line(geometry, material)
    }],
  ]

  for (const [name, object] of cases) {
    const scene = new THREE.Scene()
    scene.add(object())
    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      /material\.alphaTest must be a finite number/i,
      `${name} alphaTest should fail clearly`,
    )
  }
})

test('invalid material scalar values fail clearly', () => {
  const texture = solidTexture(255, 255, 255)
  const cases = [
    ['shininess', () => {
      const material = new THREE.MeshPhongMaterial({ color: 0xffffff })
      material.shininess = 'glossy'
      return material
    }, /material\.shininess must be a finite number/i],
    ['emissiveIntensity', () => {
      const material = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xff0000 })
      material.emissiveIntensity = Number.POSITIVE_INFINITY
      return material
    }, /material\.emissiveIntensity must be a finite number/i],
    ['opacity', () => {
      const material = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true })
      material.opacity = 'clear'
      return material
    }, /material\.opacity must be a finite number/i],
    ['bumpScale', () => {
      const material = new THREE.MeshNormalMaterial({ bumpMap: texture })
      material.bumpScale = 'strong'
      return material
    }, /material\.bumpScale must be a finite number/i],
    ['normalScale.x', () => {
      const material = new THREE.MeshStandardMaterial({ color: 0xffffff, normalMap: texture })
      material.normalScale.x = 'wide'
      return material
    }, /material\.normalScale\.x must be a finite number/i],
    ['normalScale container', () => {
      const material = new THREE.MeshStandardMaterial({ color: 0xffffff, normalMap: texture })
      material.normalScale = 'wide'
      return material
    }, /material\.normalScale must be a Vector2-like object/i],
    ['displacementScale', () => {
      const material = new THREE.MeshStandardMaterial({ color: 0xffffff, displacementMap: texture })
      material.displacementScale = Number.NaN
      return material
    }, /material\.displacementScale must be a finite number/i],
    ['displacementBias', () => {
      const material = new THREE.MeshStandardMaterial({ color: 0xffffff, displacementMap: texture })
      material.displacementBias = 'nearer'
      return material
    }, /material\.displacementBias must be a finite number/i],
    ['aoMapIntensity', () => {
      const material = new THREE.MeshBasicMaterial({ color: 0xffffff, aoMap: texture })
      material.aoMapIntensity = 'dark'
      return material
    }, /material\.aoMapIntensity must be a finite number/i],
    ['lightMapIntensity', () => {
      const material = new THREE.MeshBasicMaterial({ color: 0xffffff, lightMap: texture })
      material.lightMapIntensity = Number.NEGATIVE_INFINITY
      return material
    }, /material\.lightMapIntensity must be a finite number/i],
  ]

  for (const [name, material, pattern] of cases) {
    const scene = new THREE.Scene()
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(1, 1), material()))
    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      pattern,
      `${name} should fail clearly`,
    )
  }
})

test('invalid material color values fail clearly', () => {
  const cases = [
    ['base color', () => {
      const material = new THREE.MeshBasicMaterial({ color: 0xffffff })
      material.color = { isColor: true, r: 1, g: 'green', b: 0 }
      return material
    }, /material\.color\.g must be a finite number/i],
    ['base color container', () => {
      const material = new THREE.MeshBasicMaterial({ color: 0xffffff })
      material.color = 'red'
      return material
    }, /material\.color must be a color-like object or \[r, g, b\]/i],
    ['emissive', () => {
      const material = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xff0000 })
      material.emissive = { isColor: true, r: 1, g: 0, b: Number.NaN }
      return material
    }, /material\.emissive\.b must be a finite number/i],
    ['physical specularColor', () => {
      const material = new THREE.MeshPhysicalMaterial({ color: 0xffffff })
      material.specularColor = { isColor: true, r: 1, g: Number.POSITIVE_INFINITY, b: 1 }
      return material
    }, /material\.specularColor\.g must be a finite number/i],
    ['blendColor', () => {
      const material = new THREE.MeshBasicMaterial({ color: 0xffffff })
      material.blending = THREE.CustomBlending
      material.blendColor = { isColor: true, r: 'red', g: 0, b: 0 }
      return material
    }, /material\.blendColor\.r must be a finite number/i],
  ]

  for (const [name, material, pattern] of cases) {
    const scene = new THREE.Scene()
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(1, 1), material()))
    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      pattern,
      `${name} should fail clearly`,
    )
  }
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

test('MeshNormalMaterial supports object-space normal maps', () => {
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

  function renderNormalType(normalMapType) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      geometry,
      new THREE.MeshNormalMaterial({
        normalMap: solidTexture(255, 128, 128),
        normalMapType,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const tangentSpace = renderNormalType(THREE.TangentSpaceNormalMap)
  const objectSpace = renderNormalType(THREE.ObjectSpaceNormalMap)
  assert.ok(tangentSpace.g > tangentSpace.r + 35, `swapped UV tangent normal should point toward green (${tangentSpace.g} vs ${tangentSpace.r})`)
  assert.ok(objectSpace.r > objectSpace.g + 35, `object-space normal should point toward red (${objectSpace.r} vs ${objectSpace.g})`)
})

test('unsupported normalMapType values fail clearly', () => {
  const scene = new THREE.Scene()
  const material = new THREE.MeshNormalMaterial({ normalMap: solidTexture(128, 128, 255) })
  material.normalMapType = 999
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  assert.throws(
    () => renderRgba(scene, camera, { width: 64, height: 64 }),
    /material\.normalMapType 999.*not supported/i,
  )
})

test('MeshNormalMaterial normalMap honors normalScale strength', () => {
  function renderNormalScale(scale) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshNormalMaterial({
        normalMap: solidTexture(255, 128, 128),
        normalScale: new THREE.Vector2(scale, scale),
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const disabled = renderNormalScale(0)
  const strong = renderNormalScale(2)
  assert.ok(strong.r > disabled.r + 40, `larger normalScale should strengthen tangent-red output (${strong.r} vs ${disabled.r})`)
  assert.ok(disabled.b > strong.b + 40, `normalScale 0 should preserve more front-facing blue output (${disabled.b} vs ${strong.b})`)
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

test('MeshNormalMaterial bumpMap honors explicit texture matrices', () => {
  function renderBumpMaterial(matrixOffsetX) {
    const bumpMap = rgbaTexture([
      0, 0, 0, 255,
      255, 255, 255, 255,
    ], 2, 1)
    bumpMap.magFilter = THREE.LinearFilter
    bumpMap.minFilter = THREE.LinearFilter
    if (matrixOffsetX !== 0) setTextureMatrixOffset(bumpMap, matrixOffsetX)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshNormalMaterial({ bumpMap, bumpScale: 4 }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return renderRgba(scene, camera, { width: 64, height: 64 })
  }

  const unshifted = renderBumpMaterial(0)
  const shifted = renderBumpMaterial(0.5)
  const diff = meanAbsDiff(unshifted, shifted)
  assert.ok(diff > 2, `explicit bumpMap matrix should change the bump perturbation (diff=${diff.toFixed(2)})`)
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

test('MeshNormalMaterial bumpMap honors horizontal and vertical repeat wrapping', () => {
  const renderer = new Renderer()

  function offsetUvPlane(offsetU, offsetV) {
    const geometry = new THREE.PlaneGeometry(2, 2)
    const uv = geometry.getAttribute('uv')
    for (let i = 0; i < uv.count; i++) {
      uv.setXY(i, uv.getX(i) + offsetU, uv.getY(i) + offsetV)
    }
    uv.needsUpdate = true
    return geometry
  }

  function renderWithWrapping({ wrapS = THREE.ClampToEdgeWrapping, wrapT = THREE.ClampToEdgeWrapping, vertical = false }) {
    const bumpMap = vertical
      ? rgbaTexture([
        0, 0, 0, 255,
        0, 0, 0, 255,
        255, 255, 255, 255,
        255, 255, 255, 255,
      ], 2, 2)
      : rgbaTexture([
        0, 0, 0, 255,
        255, 255, 255, 255,
        0, 0, 0, 255,
        255, 255, 255, 255,
      ], 2, 2)
    bumpMap.wrapS = wrapS
    bumpMap.wrapT = wrapT
    bumpMap.magFilter = THREE.LinearFilter
    bumpMap.minFilter = THREE.LinearFilter

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      offsetUvPlane(vertical ? 0 : 1, vertical ? 1 : 0),
      new THREE.MeshNormalMaterial({ bumpMap, bumpScale: 8 }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return renderer.render(scene, camera, { width: 64, height: 64, format: 'rgba' })
  }

  const clamped = renderWithWrapping({ wrapS: THREE.ClampToEdgeWrapping })
  const repeated = renderWithWrapping({ wrapS: THREE.RepeatWrapping })
  const mirrored = renderWithWrapping({ wrapS: THREE.MirroredRepeatWrapping })
  const diff = meanAbsDiff(clamped, repeated)
  const mirroredDiff = meanAbsDiff(clamped, mirrored)
  const repeatedMirroredDiff = meanAbsDiff(repeated, mirrored)
  assert.ok(diff > 5, `RepeatWrapping should wrap bumpMap U coordinates before perturbing normals (diff=${diff.toFixed(2)})`)
  assert.ok(mirroredDiff > 2, `MirroredRepeatWrapping should differ from clamped bumpMap U coordinates before perturbing normals (diff=${mirroredDiff.toFixed(2)})`)
  assert.ok(repeatedMirroredDiff > 5, `MirroredRepeatWrapping should reflect bumpMap U coordinates differently than RepeatWrapping (diff=${repeatedMirroredDiff.toFixed(2)})`)

  const clampedVertical = renderWithWrapping({ wrapT: THREE.ClampToEdgeWrapping, vertical: true })
  const repeatedVertical = renderWithWrapping({ wrapT: THREE.RepeatWrapping, vertical: true })
  const mirroredVertical = renderWithWrapping({ wrapT: THREE.MirroredRepeatWrapping, vertical: true })
  const verticalDiff = meanAbsDiff(clampedVertical, repeatedVertical)
  const mirroredVerticalDiff = meanAbsDiff(clampedVertical, mirroredVertical)
  const repeatedMirroredVerticalDiff = meanAbsDiff(repeatedVertical, mirroredVertical)
  assert.ok(verticalDiff > 5, `RepeatWrapping should wrap bumpMap V coordinates before perturbing normals (diff=${verticalDiff.toFixed(2)})`)
  assert.ok(mirroredVerticalDiff > 2, `MirroredRepeatWrapping should differ from clamped bumpMap V coordinates before perturbing normals (diff=${mirroredVerticalDiff.toFixed(2)})`)
  assert.ok(repeatedMirroredVerticalDiff > 5, `MirroredRepeatWrapping should reflect bumpMap V coordinates differently than RepeatWrapping (diff=${repeatedMirroredVerticalDiff.toFixed(2)})`)
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

test('MeshMatcapMaterial map honors explicit texture matrices', () => {
  const map = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter
  setTextureMatrixOffset(map, 0.5)

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

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  assert.ok(mean.g > mean.r + 40, `explicit matcap map matrix should sample the green texel (${mean.g} vs ${mean.r})`)
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

test('MeshMatcapMaterial map honors nearest and linear filters', () => {
  function renderWithFilter(filter) {
    const map = rgbaTexture([
      0, 0, 0, 255,
      255, 255, 255, 255,
    ], 2, 1)
    map.magFilter = filter
    map.minFilter = filter

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.45, 0.5),
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

  const nearest = renderWithFilter(THREE.NearestFilter)
  const linear = renderWithFilter(THREE.LinearFilter)
  assert.ok(linear.r > nearest.r + 30, `LinearFilter should blend in the bright matcap map texel (${linear.r} vs ${nearest.r})`)
})

test('MeshMatcapMaterial map honors horizontal and vertical repeat wrapping', () => {
  function renderWithWrapping({ wrapS, wrapT, vertical = false }) {
    const map = vertical
      ? rgbaTexture([
        255, 0, 0, 255,
        0, 255, 0, 255,
      ], 1, 2)
      : rgbaTexture([
        255, 0, 0, 255,
        0, 255, 0, 255,
      ], 2, 1)
    if (wrapS != null) map.wrapS = wrapS
    if (wrapT != null) map.wrapT = wrapT
    map.magFilter = THREE.NearestFilter
    map.minFilter = THREE.NearestFilter

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(vertical ? 0.5 : 1.25, vertical ? 1.25 : 0.5),
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

  const clamped = renderWithWrapping({ wrapS: THREE.ClampToEdgeWrapping })
  const repeated = renderWithWrapping({ wrapS: THREE.RepeatWrapping })
  const mirrored = renderWithWrapping({ wrapS: THREE.MirroredRepeatWrapping })
  assert.ok(clamped.g > clamped.r + 30, `clamped matcap map U coordinates should sample the green edge texel (${clamped.g} vs ${clamped.r})`)
  assert.ok(repeated.r > repeated.g + 80, `repeated matcap map U coordinates should wrap to the red texel (${repeated.r} vs ${repeated.g})`)
  assert.ok(mirrored.g > mirrored.r + 30, `mirrored matcap map U coordinates should reflect to the green texel (${mirrored.g} vs ${mirrored.r})`)

  const clampedVertical = renderWithWrapping({ wrapT: THREE.ClampToEdgeWrapping, vertical: true })
  const repeatedVertical = renderWithWrapping({ wrapT: THREE.RepeatWrapping, vertical: true })
  const mirroredVertical = renderWithWrapping({ wrapT: THREE.MirroredRepeatWrapping, vertical: true })
  assert.ok(clampedVertical.g > clampedVertical.r + 30, `clamped matcap map V coordinates should sample the green edge texel (${clampedVertical.g} vs ${clampedVertical.r})`)
  assert.ok(repeatedVertical.r > repeatedVertical.g + 80, `repeated matcap map V coordinates should wrap to the red texel (${repeatedVertical.r} vs ${repeatedVertical.g})`)
  assert.ok(mirroredVertical.g > mirroredVertical.r + 30, `mirrored matcap map V coordinates should reflect to the green texel (${mirroredVertical.g} vs ${mirroredVertical.r})`)
})

test('MeshMatcapMaterial map decodes sRGB colorSpace before shading', () => {
  function renderColorSpace(colorSpace) {
    const map = solidTexture(128, 128, 128)
    map.colorSpace = colorSpace

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
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

  const srgb = renderColorSpace(THREE.SRGBColorSpace)
  const linear = renderColorSpace(THREE.LinearSRGBColorSpace)
  assert.ok(linear.r > srgb.r + 15, `linear matcap map should render brighter than decoded sRGB texture (${linear.r} vs ${srgb.r})`)
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

test('normalMap honors explicit texture matrices', () => {
  function renderWithMatrix(matrixOffsetX) {
    const normalMap = rgbaTexture([
      128, 128, 255, 255,
      255, 128, 128, 255,
    ], 2, 1)
    normalMap.magFilter = THREE.LinearFilter
    normalMap.minFilter = THREE.LinearFilter
    if (matrixOffsetX !== 0) setTextureMatrixOffset(normalMap, matrixOffsetX)

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

  const unshifted = renderWithMatrix(0)
  const shifted = renderWithMatrix(0.5)
  const diff = meanAbsDiff(unshifted, shifted)
  assert.ok(diff > 2, `explicit normalMap matrix should change sampled tangent-space normals (diff=${diff.toFixed(2)})`)
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

test('normalMap honors horizontal and vertical repeat wrapping', () => {
  function renderWithWrapping({ wrapS, wrapT, vertical = false }) {
    const normalMap = vertical
      ? rgbaTexture([
        255, 128, 128, 255,
        128, 128, 255, 255,
      ], 1, 2)
      : rgbaTexture([
        255, 128, 128, 255,
        128, 128, 255, 255,
      ], 2, 1)
    if (wrapS != null) normalMap.wrapS = wrapS
    if (wrapT != null) normalMap.wrapT = wrapT
    normalMap.magFilter = THREE.NearestFilter
    normalMap.minFilter = THREE.NearestFilter

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(vertical ? 0.5 : 1.25, vertical ? 1.25 : 0.5),
      new THREE.MeshNormalMaterial({ normalMap }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const clamped = renderWithWrapping({ wrapS: THREE.ClampToEdgeWrapping })
  const repeated = renderWithWrapping({ wrapS: THREE.RepeatWrapping })
  const mirrored = renderWithWrapping({ wrapS: THREE.MirroredRepeatWrapping })
  assert.ok(clamped.b > clamped.r + 20, `clamped normalMap U coordinates should sample the flat blue texel (${clamped.b} vs ${clamped.r})`)
  assert.ok(repeated.r > repeated.b + 20, `repeated normalMap U coordinates should wrap to the tangent-right red texel (${repeated.r} vs ${repeated.b})`)
  assert.ok(mirrored.b > mirrored.r + 20, `mirrored normalMap U coordinates should reflect to the flat blue texel (${mirrored.b} vs ${mirrored.r})`)

  const clampedVertical = renderWithWrapping({ wrapT: THREE.ClampToEdgeWrapping, vertical: true })
  const repeatedVertical = renderWithWrapping({ wrapT: THREE.RepeatWrapping, vertical: true })
  const mirroredVertical = renderWithWrapping({ wrapT: THREE.MirroredRepeatWrapping, vertical: true })
  assert.ok(clampedVertical.b > clampedVertical.r + 20, `clamped normalMap V coordinates should sample the flat blue texel (${clampedVertical.b} vs ${clampedVertical.r})`)
  assert.ok(repeatedVertical.r > repeatedVertical.b + 20, `repeated normalMap V coordinates should wrap to the tangent-right red texel (${repeatedVertical.r} vs ${repeatedVertical.b})`)
  assert.ok(mirroredVertical.b > mirroredVertical.r + 20, `mirrored normalMap V coordinates should reflect to the flat blue texel (${mirroredVertical.b} vs ${mirroredVertical.r})`)
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

test('MeshPhongMaterial specularMap keeps primary UVs when another map uses a secondary channel', () => {
  const specularMap = rgbaTexture([
    0, 0, 0, 255,
    255, 0, 0, 255,
  ], 2, 1)
  specularMap.magFilter = THREE.NearestFilter
  specularMap.minFilter = THREE.NearestFilter

  const normalMap = rgbaTexture([
    128, 128, 255, 255,
    128, 128, 255, 255,
  ], 2, 1)
  normalMap.channel = 1
  normalMap.magFilter = THREE.NearestFilter
  normalMap.minFilter = THREE.NearestFilter

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
      normalMap,
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
  assert.ok(mean.r > 35, `specularMap channel 0 should stay on primary UVs (${mean.r})`)
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

test('MeshPhongMaterial specularMap honors explicit texture matrices', () => {
  const specularMap = rgbaTexture([
    0, 0, 0, 255,
    255, 0, 0, 255,
  ], 2, 1)
  specularMap.channel = 1
  specularMap.magFilter = THREE.NearestFilter
  specularMap.minFilter = THREE.NearestFilter
  setTextureMatrixOffset(specularMap, 0.5)

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
  assert.ok(mean.r > 35, `explicit specularMap matrix should sample uv1's enabled texel (${mean.r})`)
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

test('MeshPhongMaterial specularMap honors horizontal and vertical repeat wrapping', () => {
  function renderWithWrapping({ wrapS, wrapT, vertical = false }) {
    const specularMap = vertical
      ? rgbaTexture([
        255, 0, 0, 255,
        0, 0, 0, 255,
      ], 1, 2)
      : rgbaTexture([
        255, 0, 0, 255,
        0, 0, 0, 255,
      ], 2, 1)
    if (wrapS != null) specularMap.wrapS = wrapS
    if (wrapT != null) specularMap.wrapT = wrapT
    specularMap.magFilter = THREE.NearestFilter
    specularMap.minFilter = THREE.NearestFilter

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(vertical ? 0.5 : 1.25, vertical ? 1.25 : 0.5),
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

  const clamped = renderWithWrapping({ wrapS: THREE.ClampToEdgeWrapping })
  const repeated = renderWithWrapping({ wrapS: THREE.RepeatWrapping })
  const mirrored = renderWithWrapping({ wrapS: THREE.MirroredRepeatWrapping })
  assert.ok(clamped.r < 5, `clamped specularMap U coordinates should sample the disabled edge texel (${clamped.r})`)
  assert.ok(repeated.r > clamped.r + 80, `repeated specularMap U coordinates should wrap to the enabled texel (${repeated.r} vs ${clamped.r})`)
  assert.ok(mirrored.r < 5, `mirrored specularMap U coordinates should reflect to the disabled texel (${mirrored.r})`)
  assert.ok(repeated.r > mirrored.r + 80, `mirrored specularMap U coordinates should differ from RepeatWrapping (${mirrored.r} vs ${repeated.r})`)

  const clampedVertical = renderWithWrapping({ wrapT: THREE.ClampToEdgeWrapping, vertical: true })
  const repeatedVertical = renderWithWrapping({ wrapT: THREE.RepeatWrapping, vertical: true })
  const mirroredVertical = renderWithWrapping({ wrapT: THREE.MirroredRepeatWrapping, vertical: true })
  assert.ok(clampedVertical.r < 5, `clamped specularMap V coordinates should sample the disabled edge texel (${clampedVertical.r})`)
  assert.ok(repeatedVertical.r > clampedVertical.r + 80, `repeated specularMap V coordinates should wrap to the enabled texel (${repeatedVertical.r} vs ${clampedVertical.r})`)
  assert.ok(mirroredVertical.r < 5, `mirrored specularMap V coordinates should reflect to the disabled texel (${mirroredVertical.r})`)
  assert.ok(repeatedVertical.r > mirroredVertical.r + 80, `mirrored specularMap V coordinates should differ from RepeatWrapping (${mirroredVertical.r} vs ${repeatedVertical.r})`)
})

test('MeshPhongMaterial scene environment feeds specular reflection', () => {
  function renderPhongEnvironment(specularMap, useEnvironment) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    if (useEnvironment) {
      const environment = solidTexture(255, 255, 255)
      environment.mapping = THREE.EquirectangularReflectionMapping
      scene.environment = environment
    }
    scene.add(new THREE.Mesh(
      new THREE.SphereGeometry(1, 48, 24),
      new THREE.MeshPhongMaterial({
        color: 0x000000,
        specular: 0xffffff,
        shininess: 120,
        specularMap,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return renderRgba(scene, camera, { width: 96, height: 96 })
  }

  const noEnvironment = maxLuminance(renderPhongEnvironment(null, false))
  const environment = maxLuminance(renderPhongEnvironment(null, true))
  const maskedEnvironment = maxLuminance(renderPhongEnvironment(solidTexture(0, 0, 0), true))
  assert.ok(environment > noEnvironment + 40, `scene environment should add a Phong reflection (${environment} vs ${noEnvironment})`)
  assert.ok(environment > maskedEnvironment + 40, `specularMap should suppress Phong environment reflection (${environment} vs ${maskedEnvironment})`)
})

test('MeshPhongMaterial material envMap feeds specular reflection', () => {
  function renderPhongMaterialEnvironment(intensity) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const envMap = solidTexture(255, 255, 255)
    envMap.mapping = THREE.EquirectangularReflectionMapping
    const material = new THREE.MeshPhongMaterial({
      color: 0x000000,
      specular: 0xffffff,
      shininess: 120,
      envMap,
    })
    if (intensity != null) material.envMapIntensity = intensity
    scene.add(new THREE.Mesh(new THREE.SphereGeometry(1, 48, 24), material))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return maxLuminance(renderRgba(scene, camera, { width: 96, height: 96 }))
  }

  const disabled = renderPhongMaterialEnvironment(0)
  const reflected = renderPhongMaterialEnvironment(1)
  assert.ok(reflected > disabled + 40, `material envMap should add Phong reflection (${reflected} vs ${disabled})`)
})

test('MeshPhongMaterial material envMap honors legacy combine and reflectivity', () => {
  function renderPhongMaterialEnvironment(combine, reflectivity = 1) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const envMap = solidTexture(0, 255, 0)
    envMap.mapping = THREE.EquirectangularReflectionMapping
    const material = new THREE.MeshPhongMaterial({
      color: 0x000000,
      specular: 0xffffff,
      shininess: 120,
      envMap,
      combine,
      reflectivity,
    })
    material.envMapIntensity = 0.5
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 64, 64, 24, 24, 40, 40)
  }

  const disabled = renderPhongMaterialEnvironment(THREE.MultiplyOperation, 0)
  const multiply = renderPhongMaterialEnvironment(THREE.MultiplyOperation)
  const add = renderPhongMaterialEnvironment(THREE.AddOperation)

  assert.ok(multiply.g > disabled.g + 10, `reflectivity should scale Phong env reflection (${multiply.g} vs ${disabled.g})`)
  assert.ok(add.g > multiply.g + 10, `AddOperation should add extra Phong env reflection (${add.g} vs ${multiply.g})`)
})

test('MeshLambertMaterial material envMap honors legacy mix reflectivity', () => {
  function renderLambertMaterialEnvironment(reflectivity) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const envMap = solidTexture(0, 255, 0)
    envMap.mapping = THREE.EquirectangularReflectionMapping
    const material = new THREE.MeshLambertMaterial({
      color: 0xff0000,
      envMap,
      combine: THREE.MixOperation,
      reflectivity,
    })
    material.envMapIntensity = 0.5
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

    const light = new THREE.DirectionalLight(0xffffff, 4)
    light.position.set(0, 0, 3)
    scene.add(light)

    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 64, 64, 24, 24, 40, 40)
  }

  const disabled = renderLambertMaterialEnvironment(0)
  const mixed = renderLambertMaterialEnvironment(1)

  assert.ok(disabled.r > disabled.g + 20, `reflectivity 0 should preserve direct Lambert color (${disabled.r}, ${disabled.g})`)
  assert.ok(mixed.g > mixed.r + 20, `MixOperation should replace Lambert output with green env reflection (${mixed.r}, ${mixed.g})`)
})

test('MeshBasicMaterial material envMap uses legacy combine modes', () => {
  function renderBasicMaterialEnvironment(combine, reflectivity = 1) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const envMap = solidTexture(0, 255, 0)
    envMap.mapping = THREE.EquirectangularReflectionMapping
    const material = new THREE.MeshBasicMaterial({
      color: 0xff0000,
      envMap,
      combine,
      reflectivity,
    })
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 64, 64, 24, 24, 40, 40)
  }

  const multiply = renderBasicMaterialEnvironment(THREE.MultiplyOperation)
  const add = renderBasicMaterialEnvironment(THREE.AddOperation)
  const mixZero = renderBasicMaterialEnvironment(THREE.MixOperation, 0)
  const mixFull = renderBasicMaterialEnvironment(THREE.MixOperation, 1)

  assert.ok(add.g > multiply.g + 40, `AddOperation should add green env reflection (${add.g} vs ${multiply.g})`)
  assert.ok(mixZero.r > mixZero.g + 40, `reflectivity 0 should preserve Basic color (${mixZero.r}, ${mixZero.g})`)
  assert.ok(mixFull.g > mixFull.r + 40, `MixOperation should replace with green env reflection (${mixFull.r}, ${mixFull.g})`)
})

test('MeshBasicMaterial material envMap supports refraction mapping', () => {
  function renderBasicEnvironmentMapping(mapping) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const envMap = splitEnvironmentTexture()
    envMap.mapping = mapping
    const material = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      envMap,
      combine: THREE.MixOperation,
      reflectivity: 1,
      refractionRatio: 0.5,
      side: THREE.DoubleSide,
    })
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material)
    mesh.rotation.y = 0.5
    scene.add(mesh)

    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 64, 64, 24, 24, 40, 40)
  }

  const reflected = renderBasicEnvironmentMapping(THREE.EquirectangularReflectionMapping)
  const refracted = renderBasicEnvironmentMapping(THREE.EquirectangularRefractionMapping)
  assert.ok(reflected.g > reflected.r + 15, `reflection should sample the green environment half (${reflected.r}, ${reflected.g})`)
  assert.ok(refracted.r > refracted.g + 15, `refraction should sample the red environment half (${refracted.r}, ${refracted.g})`)
})

test('material envMap colorSpace controls LDR IBL decode', () => {
  function renderColorSpace(colorSpace) {
    const envMap = solidTexture(128, 128, 128)
    envMap.colorSpace = colorSpace
    envMap.mapping = THREE.EquirectangularReflectionMapping

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({
        color: 0xffffff,
        envMap,
        combine: THREE.MixOperation,
        reflectivity: 1,
      }),
    ))

    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 64, 64, 24, 24, 40, 40)
  }

  const srgb = renderColorSpace(THREE.SRGBColorSpace)
  const linear = renderColorSpace(THREE.LinearSRGBColorSpace)
  assert.ok(linear.r > srgb.r + 20, `linear material envMap should render brighter than decoded sRGB (${linear.r} vs ${srgb.r})`)
  for (const alias of ['srgb-linear', 'linear-srgb', 'linearsrgb', 'linear']) {
    assert.deepEqual(renderColorSpace(alias), linear, `${alias} should match THREE.LinearSRGBColorSpace for material envMap IBL`)
  }
})

test('scene environment does not affect MeshBasicMaterial without material envMap', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const environment = solidTexture(0, 255, 0)
  environment.mapping = THREE.EquirectangularReflectionMapping
  scene.environment = environment
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  ))

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  const mean = meanRegion(renderRgba(scene, camera, {
    width: 64,
    height: 64,
    outputColorSpace: THREE.LinearSRGBColorSpace,
  }), 64, 64, 24, 24, 40, 40)
  assert.ok(mean.r > mean.g + 40, `scene.environment should not drive Basic material reflection (${mean.r}, ${mean.g})`)
})

test('material envMapRotation rotates shared IBL', () => {
  function renderWithRotation(yRotation) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const envMap = splitEnvironmentTexture()
    const material = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      metalness: 1,
      roughness: 0,
      envMap,
    })
    material.envMapIntensity = 4
    material.envMapRotation = new THREE.Euler(0, yRotation, 0)
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 64, 64, 24, 24, 40, 40)
  }

  const unrotated = renderWithRotation(0)
  const rotated = renderWithRotation(-Math.PI / 2)
  assert.ok(unrotated.r > unrotated.g + 15, `unrotated material reflection should sample the red environment half (${unrotated.r} vs ${unrotated.g})`)
  assert.ok(rotated.g > rotated.r + 15, `rotated material reflection should sample the green environment half (${rotated.g} vs ${rotated.r})`)
})

test('material envMap fallback does not light unrelated materials', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  scene.add(new THREE.Mesh(
    new THREE.SphereGeometry(1, 48, 24),
    new THREE.MeshPhongMaterial({
      color: 0x000000,
      specular: 0xffffff,
      shininess: 120,
    }),
  ))

  const envMap = solidTexture(255, 255, 255)
  envMap.mapping = THREE.EquirectangularReflectionMapping
  const envMapped = new THREE.Mesh(
    new THREE.SphereGeometry(1, 16, 8),
    new THREE.MeshPhongMaterial({
      color: 0x000000,
      specular: 0xffffff,
      shininess: 120,
      envMap,
    }),
  )
  envMapped.position.x = 6
  scene.add(envMapped)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  const mean = meanRgba(renderRgba(scene, camera, { width: 96, height: 96 }))
  assert.ok(mean.r + mean.g + mean.b < 5, `material envMap should not affect unrelated Phong material (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('invalid material envMap scalar values fail clearly', () => {
  const cases = [
    ['envMapIntensity', (material) => {
      material.envMapIntensity = 'bright'
    }, /material\.envMapIntensity must be a finite number/i],
    ['reflectivity', (material) => {
      material.reflectivity = Number.NaN
    }, /material\.reflectivity must be a finite number/i],
    ['refractionRatio', (material) => {
      material.refractionRatio = Number.POSITIVE_INFINITY
    }, /material\.refractionRatio must be a finite number/i],
  ]

  for (const [name, mutate, pattern] of cases) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const envMap = solidTexture(255, 255, 255)
    envMap.mapping = name === 'refractionRatio'
      ? THREE.EquirectangularRefractionMapping
      : THREE.EquirectangularReflectionMapping
    const material = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      envMap,
      combine: THREE.MixOperation,
    })
    mutate(material)
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      pattern,
      `${name} should fail clearly`,
    )
  }
})

test('envMap on non-env-map material classes is ignored', () => {
  const envMap = Object.assign(solidTexture(255, 255, 255), {
    mapping: THREE.EquirectangularReflectionMapping,
  })
  const cases = [
    ['MeshDistanceMaterial', () => new THREE.MeshDistanceMaterial()],
    ['MeshMatcapMaterial', () => new THREE.MeshMatcapMaterial({ color: 0xffffff })],
    ['MeshNormalMaterial', () => new THREE.MeshNormalMaterial()],
    ['MeshDepthMaterial', () => new THREE.MeshDepthMaterial()],
    ['MeshToonMaterial', () => new THREE.MeshToonMaterial({ color: 0xffffff })],
  ]

  function renderMaterial(makeMaterial, envMapped) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const material = makeMaterial()
    if (envMapped) material.envMap = envMap
    scene.add(new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), material))
    return renderRgba(scene, makeCamera(), { width: 64, height: 64 })
  }

  for (const [name, makeMaterial] of cases) {
    const withoutEnvMap = renderMaterial(makeMaterial, false)
    const withEnvMap = renderMaterial(makeMaterial, true)
    const diff = meanAbsDiff(withoutEnvMap, withEnvMap)
    assert.ok(diff < 0.5, `${name} envMap should be ignored like Three.js (${diff.toFixed(3)})`)
  }
})

test('unsupported material envMap inputs fail clearly', () => {
  const envMap = Object.assign(solidTexture(255, 255, 255), {
    mapping: THREE.EquirectangularReflectionMapping,
  })
  const refractionEnvMap = Object.assign(solidTexture(255, 255, 255), {
    mapping: THREE.EquirectangularRefractionMapping,
  })

  function assertMaterialEnvMapFailure(materialEnvMap, pattern, label) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.SphereGeometry(1, 16, 16),
      new THREE.MeshBasicMaterial({ color: 0xffffff, envMap: materialEnvMap }),
    ))

    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      pattern,
      label,
    )
  }

  assertMaterialEnvMapFailure(
    {
      isTexture: true,
      isCompressedTexture: true,
      mapping: THREE.EquirectangularReflectionMapping,
      image: { width: 4, height: 4 },
      mipmaps: [{ data: new Uint8Array(16), width: 4, height: 4 }],
    },
    /material\.envMap uses a compressed texture.*pre-decode/i,
    'material envMap compressed texture',
  )

  assertMaterialEnvMapFailure(
    Object.assign(new THREE.DataArrayTexture(new Uint8Array([255, 0, 0, 255]), 1, 1, 1), {
      mapping: THREE.EquirectangularReflectionMapping,
    }),
    /material\.envMap uses an array or 3D texture/i,
    'material envMap array texture',
  )

  {
    const mipmappedEnvMap = Object.assign(solidTexture(255, 255, 255), {
      mapping: THREE.EquirectangularReflectionMapping,
    })
    mipmappedEnvMap.mipmaps = [{ data: new Uint8Array([255, 255, 255, 255]), width: 1, height: 1 }]
    assertMaterialEnvMapFailure(
      mipmappedEnvMap,
      /material\.envMap provides explicit texture mipmaps.*not uploaded/i,
      'material envMap explicit mipmaps',
    )
  }

  {
    const invalidColorSpace = Object.assign(solidTexture(255, 255, 255), {
      mapping: THREE.EquirectangularReflectionMapping,
    })
    invalidColorSpace.colorSpace = 'display-p3'
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.SphereGeometry(1, 16, 16),
      new THREE.MeshBasicMaterial({ color: 0xffffff, envMap: invalidColorSpace }),
    ))

    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      /texture\.colorSpace display-p3.*not supported.*SRGBColorSpace.*LinearSRGBColorSpace.*NoColorSpace/i,
      'material envMap colorSpace',
    )
  }

  {
    const invalidEncoding = Object.assign(solidTexture(255, 255, 255), {
      mapping: THREE.EquirectangularReflectionMapping,
    })
    invalidEncoding.encoding = 999
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.SphereGeometry(1, 16, 16),
      new THREE.MeshBasicMaterial({ color: 0xffffff, envMap: invalidEncoding }),
    ))

    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      /texture\.encoding 999.*not supported.*sRGBEncoding.*LinearEncoding.*texture\.colorSpace/i,
      'material envMap encoding',
    )
  }

  {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.SphereGeometry(1, 16, 16),
      new THREE.MeshPhongMaterial({ color: 0xffffff, envMap: refractionEnvMap }),
    ))

    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      /refraction mappings are only supported for MeshBasicMaterial/i,
      'MeshPhongMaterial refraction envMap',
    )
  }

  {
    const invalidCombine = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      envMap,
      combine: 999,
    })
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(new THREE.SphereGeometry(1, 16, 16), invalidCombine))

    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      /material\.envMap combine.*MultiplyOperation.*MixOperation.*AddOperation/i,
      'invalid material envMap combine',
    )
  }

  {
    const cubeUvEnvMap = Object.assign(solidTexture(255, 255, 255), {
      mapping: THREE.CubeUVReflectionMapping,
    })
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.SphereGeometry(1, 16, 16),
      new THREE.MeshBasicMaterial({ color: 0xffffff, envMap: cubeUvEnvMap }),
    ))

    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      /material\.envMap.*refraction or PMREM\/CubeUV environment mapping.*not supported/i,
      'material envMap CubeUV mapping',
    )
  }

  {
    const invalidRotation = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      envMap,
    })
    invalidRotation.envMapRotation = { x: 'left', y: 0, z: 0 }
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(new THREE.SphereGeometry(1, 16, 16), invalidRotation))

    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      /material\.envMapRotation\.x must be a finite number/i,
      'invalid material envMapRotation component',
    )
  }

  {
    const invalidRotation = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      envMap,
    })
    invalidRotation.envMapRotation = 'spin'
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(new THREE.SphereGeometry(1, 16, 16), invalidRotation))

    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      /material\.envMapRotation must be a finite Vector3-like value/i,
      'invalid material envMapRotation container',
    )
  }

  {
    const invalidRotation = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      envMap,
    })
    invalidRotation.envMapRotation = [0, 0, 0, 'BAD']
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(new THREE.SphereGeometry(1, 16, 16), invalidRotation))

    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      /material\.envMapRotation\[3\] must be one of XYZ, YXZ, ZXY, ZYX, YZX, or XZY/i,
      'invalid material envMapRotation order',
    )
  }

  const firstEnvMap = Object.assign(solidTexture(255, 255, 255), {
    mapping: THREE.EquirectangularReflectionMapping,
  })
  const secondEnvMap = Object.assign(solidTexture(128, 128, 128), {
    mapping: THREE.EquirectangularReflectionMapping,
  })
  {
    const scene = new THREE.Scene()
    const first = new THREE.Mesh(
      new THREE.SphereGeometry(1, 16, 16),
      new THREE.MeshPhongMaterial({ color: 0xffffff, envMap: firstEnvMap }),
    )
    first.position.x = -1.5
    const second = new THREE.Mesh(
      new THREE.SphereGeometry(1, 16, 16),
      new THREE.MeshPhongMaterial({ color: 0xffffff, envMap: secondEnvMap }),
    )
    second.position.x = 1.5
    scene.add(first, second)
    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      /Multiple distinct material\.envMap textures.*not supported/i,
      'multiple material env maps',
    )
  }

  const sharedEnvMap = Object.assign(solidTexture(255, 255, 255), {
    mapping: THREE.EquirectangularReflectionMapping,
  })
  {
    const scene = new THREE.Scene()
    const firstMaterial = new THREE.MeshPhongMaterial({ color: 0xffffff, envMap: sharedEnvMap })
    firstMaterial.envMapRotation = new THREE.Euler(0, 0.25, 0)
    const secondMaterial = new THREE.MeshPhongMaterial({ color: 0xffffff, envMap: sharedEnvMap })
    secondMaterial.envMapRotation = new THREE.Euler(0, -0.25, 0)
    const first = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 16), firstMaterial)
    first.position.x = -1.5
    const second = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 16), secondMaterial)
    second.position.x = 1.5
    scene.add(first, second)
    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      /Multiple material\.envMapRotation values.*not supported/i,
      'multiple material env rotations',
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

test('MeshToonMaterial gradientMap honors nearest and linear filters', () => {
  function renderWithFilter(filter) {
    const gradientMap = rgbaTexture([
      0, 0, 0, 255,
      255, 255, 255, 255,
    ], 2, 1)
    gradientMap.magFilter = filter
    gradientMap.minFilter = filter

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshToonMaterial({ color: 0xffffff, gradientMap }),
    ))

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(Math.sqrt(0.99), 0, -0.1)
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const nearest = renderWithFilter(THREE.NearestFilter)
  const linear = renderWithFilter(THREE.LinearFilter)
  assert.ok(linear.r > nearest.r + 30, `LinearFilter should blend in the bright toon ramp texel (${linear.r} vs ${nearest.r})`)
})

test('MeshToonMaterial gradientMap honors horizontal repeat wrapping', () => {
  const renderer = new Renderer()

  function renderWithWrapping(wrapS) {
    const gradientMap = rgbaTexture([
      0, 0, 0, 255,
      255, 255, 255, 255,
    ], 2, 1)
    gradientMap.wrapS = wrapS
    gradientMap.magFilter = THREE.NearestFilter
    gradientMap.minFilter = THREE.NearestFilter

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshToonMaterial({ color: 0xffffff, gradientMap }),
    ))

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(0, 0, 3)
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderer.render(scene, camera, { width: 64, height: 64, format: 'rgba' }))
  }

  const clamped = renderWithWrapping(THREE.ClampToEdgeWrapping)
  const repeated = renderWithWrapping(THREE.RepeatWrapping)
  const mirrored = renderWithWrapping(THREE.MirroredRepeatWrapping)
  assert.ok(clamped.r > repeated.r + 80, `clamped toon gradientMap should sample the bright edge texel (${clamped.r} vs ${repeated.r})`)
  assert.ok(repeated.r < 5, `repeated toon gradientMap should wrap the lit ramp coordinate to the dark texel (${repeated.r})`)
  assert.ok(mirrored.r > repeated.r + 80, `mirrored toon gradientMap should reflect the lit ramp coordinate to the bright texel (${mirrored.r} vs ${repeated.r})`)
})

test('MeshToonMaterial gradientMap decodes sRGB colorSpace before shading', () => {
  function renderColorSpace(colorSpace) {
    const gradientMap = solidTexture(128, 128, 128)
    gradientMap.colorSpace = colorSpace

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshToonMaterial({ color: 0xffffff, gradientMap }),
    ))

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(0, 0, 3)
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }))
  }

  const srgb = renderColorSpace(THREE.SRGBColorSpace)
  const linear = renderColorSpace(THREE.LinearSRGBColorSpace)
  assert.ok(linear.r > srgb.r + 20, `linear toon gradient ramp should render brighter than decoded sRGB texture (${linear.r} vs ${srgb.r})`)
})

test('MeshToonMaterial normalMap perturbs toon lighting', () => {
  function renderNormalScale(normalScale) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshToonMaterial({
        color: 0xffffff,
        normalMap: solidTexture(255, 128, 128),
        normalScale: new THREE.Vector2(normalScale, normalScale),
      }),
    ))

    const light = new THREE.DirectionalLight(0xffffff, 3)
    light.position.set(3, 0, 0.25)
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const flat = renderNormalScale(0)
  const tilted = renderNormalScale(1)
  assert.ok(tilted.r > flat.r + 5, `normalMap should tilt toon lighting toward the oblique light (${tilted.r} vs ${flat.r})`)
})

test('MeshToonMaterial bumpMap perturbs toon lighting', () => {
  function renderBumpScale(bumpScale) {
    const bumpMap = rgbaTexture([
      0, 0, 0, 255,
      255, 255, 255, 255,
    ], 2, 1)
    bumpMap.magFilter = THREE.LinearFilter
    bumpMap.minFilter = THREE.LinearFilter

    const gradientMap = rgbaTexture([
      0, 0, 0, 255,
      255, 255, 255, 255,
    ], 2, 1)
    gradientMap.magFilter = THREE.LinearFilter
    gradientMap.minFilter = THREE.LinearFilter

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshToonMaterial({
        color: 0xffffff,
        bumpMap,
        bumpScale,
        gradientMap,
      }),
    ))

    const light = new THREE.DirectionalLight(0xffffff, 3)
    light.position.set(3, 0, 0.25)
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const flat = renderBumpScale(0)
  const bumped = renderBumpScale(8)
  assert.ok(flat.r > bumped.r + 8, `bumpMap should perturb the toon ramp lookup (${flat.r} vs ${bumped.r})`)
})

test('MeshToonMaterial map samples the selected secondary UV channel', () => {
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
    scene.add(new THREE.Mesh(
      geometry,
      new THREE.MeshToonMaterial({ color: 0xffffff, map }),
    ))

    const light = new THREE.DirectionalLight(0xffffff, 3)
    light.position.set(0, 0, 3)
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const primary = renderWithChannel(0)
  const secondary = renderWithChannel(1)
  assert.ok(primary.g > primary.r + 40, `toon map channel=0 should sample the primary UV green texel (${primary.g} vs ${primary.r})`)
  assert.ok(secondary.r > secondary.g + 40, `toon map channel=1 should sample the uv1 red texel (${secondary.r} vs ${secondary.g})`)
})

test('MeshToonMaterial emissiveMap samples the selected secondary UV channel', () => {
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
      new THREE.MeshToonMaterial({
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
  assert.ok(primary.g > primary.r + 40, `toon emissiveMap channel=0 should sample the primary UV green texel (${primary.g} vs ${primary.r})`)
  assert.ok(secondary.r > secondary.g + 40, `toon emissiveMap channel=1 should sample the uv1 red texel (${secondary.r} vs ${secondary.g})`)
})

test('MeshToonMaterial lightMap contributes through secondary UVs', () => {
  const lightMap = rgbaTexture([
    0, 0, 0, 255,
    255, 255, 255, 255,
  ], 2, 1)
  lightMap.channel = 1

  const geometry = constantUvPlane(0.25, 0.5)
  setConstantUvAttribute(geometry, 'uv1', 0.75, 0.5)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    geometry,
    new THREE.MeshToonMaterial({
      color: 0xffffff,
      lightMap,
      lightMapIntensity: 4,
    }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  assert.ok(mean.r > 100 && mean.g > 100 && mean.b > 100, `toon lightMap should add the bright uv1 texel (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('MeshToonMaterial alphaMap cutouts participate in alpha testing', () => {
  function renderAlpha(alphaMapGreen) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 1)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshToonMaterial({
        color: 0xff0000,
        alphaMap: solidTexture(255, alphaMapGreen, 255),
        alphaTest: 0.5,
      }),
    ))

    const light = new THREE.DirectionalLight(0xffffff, 3)
    light.position.set(0, 0, 3)
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const discarded = renderAlpha(0)
  const visible = renderAlpha(255)
  assert.ok(discarded.b > discarded.r + 40, `toon alphaMap green=0 should discard to the blue background (${discarded.b} vs ${discarded.r})`)
  assert.ok(visible.r > visible.b + 40, `toon alphaMap green=255 should keep the red toon surface (${visible.r} vs ${visible.b})`)
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

test('unsupported MeshDepthMaterial depthPacking values fail clearly', () => {
  const material = new THREE.MeshDepthMaterial()
  material.depthPacking = 999
  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

  assert.throws(
    () => renderRgba(scene, makeCamera(), { width: 32, height: 32 }),
    /material\.depthPacking 999.*not supported/i,
  )
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

test('MeshDepthMaterial base and alpha maps cut out discarded fragments', () => {
  function renderDepthMaterial(makeMaterial) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), makeMaterial()))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 8)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return renderRgba(scene, camera, { width: 64, height: 64 })
  }

  const map = rgbaTexture([
    255, 255, 255, 0,
    255, 255, 255, 255,
  ], 2, 1)
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter
  const alphaMap = rgbaTexture([
    255, 0, 255, 255,
    255, 255, 255, 255,
  ], 2, 1)
  alphaMap.magFilter = THREE.NearestFilter
  alphaMap.minFilter = THREE.NearestFilter

  for (const [label, makeMaterial] of [
    ['base texture', () => new THREE.MeshDepthMaterial({ map, alphaTest: 0.5 })],
    ['alphaMap', () => new THREE.MeshDepthMaterial({ alphaMap, alphaTest: 0.5 })],
  ]) {
    const rgba = renderDepthMaterial(makeMaterial)
    const discarded = meanRegion(rgba, 64, 64, 14, 24, 28, 40)
    const visible = meanRegion(rgba, 64, 64, 36, 24, 50, 40)
    assert.ok(discarded.r < 2, `${label} cutout should keep background depth (${discarded.r})`)
    assert.ok(visible.r > discarded.r + 3, `opaque ${label} region should write depth (${visible.r} vs ${discarded.r})`)
  }
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

test('displacementMap honors texture filters before depth output', () => {
  function renderDisplaced(filter) {
    const displacementMap = rgbaTexture([
      0, 0, 0, 255,
      255, 255, 255, 255,
    ], 2, 1)
    displacementMap.magFilter = filter
    displacementMap.minFilter = filter

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.5, 0.5),
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

  const linear = renderDisplaced(THREE.LinearFilter)
  const nearest = renderDisplaced(THREE.NearestFilter)
  assert.ok(nearest.r > linear.r + 8, `nearest displacement should sample the high texel more strongly than linear filtering (${nearest.r} vs ${linear.r})`)
})

test('displacementMap honors horizontal and vertical repeat wrapping before depth output', () => {
  function renderDisplaced({ wrapS, wrapT, vertical = false }) {
    const displacementMap = vertical
      ? rgbaTexture([
        0, 0, 0, 255,
        255, 255, 255, 255,
      ], 1, 2)
      : rgbaTexture([
        255, 255, 255, 255,
        0, 0, 0, 255,
      ], 2, 1)
    if (wrapS != null) displacementMap.wrapS = wrapS
    if (wrapT != null) displacementMap.wrapT = wrapT
    displacementMap.magFilter = THREE.NearestFilter
    displacementMap.minFilter = THREE.NearestFilter

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(vertical ? 0.5 : 1.25, vertical ? 1.25 : 0.5),
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

  const clamped = renderDisplaced({ wrapS: THREE.ClampToEdgeWrapping })
  const repeated = renderDisplaced({ wrapS: THREE.RepeatWrapping })
  const mirrored = renderDisplaced({ wrapS: THREE.MirroredRepeatWrapping })
  assert.ok(repeated.r > clamped.r + 15, `repeat wrapping should wrap displacement U coordinates to the high texel (${repeated.r} vs ${clamped.r})`)
  assert.ok(repeated.r > mirrored.r + 15, `mirrored displacement U coordinates should reflect away from the high texel (${mirrored.r} vs ${repeated.r})`)

  const clampedVertical = renderDisplaced({ wrapT: THREE.ClampToEdgeWrapping, vertical: true })
  const repeatedVertical = renderDisplaced({ wrapT: THREE.RepeatWrapping, vertical: true })
  const mirroredVertical = renderDisplaced({ wrapT: THREE.MirroredRepeatWrapping, vertical: true })
  assert.ok(repeatedVertical.r > clampedVertical.r + 15, `repeat wrapping should wrap displacement V coordinates to the high texel (${repeatedVertical.r} vs ${clampedVertical.r})`)
  assert.ok(repeatedVertical.r > mirroredVertical.r + 15, `mirrored displacement V coordinates should reflect away from the high texel (${mirroredVertical.r} vs ${repeatedVertical.r})`)
})

test('displacementMap applies displacementBias independently of sampled height', () => {
  function renderDisplacementBias(displacementBias) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.25, 0.5),
      new THREE.MeshDepthMaterial({
        displacementMap: solidTexture(0, 0, 0),
        displacementScale: 0,
        displacementBias,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 20, 20, 44, 44)
  }

  const flat = renderDisplacementBias(0)
  const biased = renderDisplacementBias(2.4)
  assert.ok(biased.r > flat.r + 25, `positive displacementBias should move the plane nearer (${biased.r} vs ${flat.r})`)
})

test('displacementMap honors explicit texture matrices before depth output', () => {
  function renderDisplaced(matrixOffsetX) {
    const displacementMap = rgbaTexture([
      0, 0, 0, 255,
      255, 255, 255, 255,
    ], 2, 1)
    displacementMap.magFilter = THREE.NearestFilter
    displacementMap.minFilter = THREE.NearestFilter
    if (matrixOffsetX !== 0) setTextureMatrixOffset(displacementMap, matrixOffsetX)

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
  assert.ok(displaced.r > flat.r + 15, `explicit displacementMap matrix should move the plane nearer (${displaced.r} vs ${flat.r})`)
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

test('MeshBasicMaterial wireframe renders triangle edges without filling faces', () => {
  function renderBasicWireframe(wireframe) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 8)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return renderRgba(scene, camera, { width: 64, height: 64 })
  }

  const solidRatio = nonBackgroundRatio(renderBasicWireframe(false), [0, 0, 0])
  const wireRatio = nonBackgroundRatio(renderBasicWireframe(true), [0, 0, 0])
  assert.ok(solidRatio > 0.4, `solid basic material should fill the plane (${solidRatio})`)
  assert.ok(wireRatio > 0.005, `wireframe basic material should draw visible edges (${wireRatio})`)
  assert.ok(wireRatio < solidRatio * 0.35, `wireframe basic material should not fill faces (${wireRatio} vs ${solidRatio})`)
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

test('invalid MeshDistanceMaterial range and reference values fail clearly', () => {
  const cases = [
    ['referencePosition', (material) => {
      material.referencePosition = [0, Number.NaN, 0]
    }, /material\.referencePosition\[1\] must be a finite number/i],
    ['nearDistance', (material) => {
      material.nearDistance = 'near'
    }, /material\.nearDistance must be a finite number/i],
    ['farDistance', (material) => {
      material.farDistance = Number.NaN
    }, /material\.farDistance must be a finite number/i],
    ['hint nearDistance', (material) => {
      material.userData.headlessThreeRenderer = { nearDistance: 'near' }
    }, /material\.userData\.headlessThreeRenderer\.nearDistance must be a finite number/i],
    ['hint distanceFar', (material) => {
      material.userData.headlessThreeRenderer = { distanceFar: Number.POSITIVE_INFINITY }
    }, /material\.userData\.headlessThreeRenderer\.distanceFar must be a finite number/i],
    ['hint distanceReferencePosition', (material) => {
      material.userData.headlessThreeRenderer = { distanceReferencePosition: { x: 0, y: 'near', z: 0 } }
    }, /material\.userData\.headlessThreeRenderer\.distanceReferencePosition\.y must be a finite number/i],
    ['userData container', (material) => {
      material.userData = 'distance'
    }, /material\.userData must be an object/i],
    ['modern hint container', (material) => {
      material.userData.headlessThreeRenderer = 'distance'
    }, /material\.userData\.headlessThreeRenderer must be an object/i],
    ['legacy hint container', (material) => {
      material.userData.headlessRenderer = []
    }, /material\.userData\.headlessRenderer must be an object/i],
  ]

  for (const [name, mutate, pattern] of cases) {
    const scene = new THREE.Scene()
    const material = new THREE.MeshDistanceMaterial()
    mutate(material)
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      pattern,
      `${name} should fail clearly`,
    )
  }
})

test('MeshDistanceMaterial alphaMap cuts out discarded fragments', () => {
  const alphaMap = rgbaTexture([
    255, 0, 255, 255,
    255, 255, 255, 255,
  ], 2, 1)
  alphaMap.magFilter = THREE.NearestFilter
  alphaMap.minFilter = THREE.NearestFilter

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshDistanceMaterial({ alphaMap, alphaTest: 0.5 }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 8)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 64, height: 64 })
  const discarded = meanRegion(rgba, 64, 64, 14, 24, 28, 40)
  const visible = meanRegion(rgba, 64, 64, 36, 24, 50, 40)
  assert.ok(discarded.r < 2, `alphaMap cutout should keep background distance (${discarded.r})`)
  assert.ok(visible.r > 60, `opaque alphaMap region should write distance (${visible.r})`)
})

test('MeshDistanceMaterial base texture alpha cuts out discarded fragments', () => {
  const map = rgbaTexture([
    255, 255, 255, 0,
    255, 255, 255, 255,
  ], 2, 1)
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshDistanceMaterial({ map, alphaTest: 0.5 }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 8)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 64, height: 64 })
  const discarded = meanRegion(rgba, 64, 64, 14, 24, 28, 40)
  const visible = meanRegion(rgba, 64, 64, 36, 24, 50, 40)
  assert.ok(discarded.r < 2, `base texture cutout should keep background distance (${discarded.r})`)
  assert.ok(visible.r > 60, `opaque base texture region should write distance (${visible.r})`)
})

test('MeshDistanceMaterial displacementMap samples the selected secondary UV channel', () => {
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
      new THREE.MeshDistanceMaterial({
        displacementMap,
        displacementScale: 1.2,
        displacementBias: 0,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 8)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 20, 20, 44, 44)
  }

  const primary = renderDisplaced(0)
  const secondary = renderDisplaced(1)
  assert.ok(primary.r > secondary.r + 15, `displacementMap channel=1 should move the distance plane closer (${primary.r} vs ${secondary.r})`)
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

test('SpriteMaterial map decodes sRGB colorSpace before shading', () => {
  function renderColorSpace(colorSpace) {
    const map = solidTexture(128, 128, 128)
    map.colorSpace = colorSpace

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
      map,
      color: 0xffffff,
    }))
    sprite.scale.set(2, 2, 1)
    scene.add(sprite)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 64, 64, 24, 24, 40, 40)
  }

  const srgb = renderColorSpace(THREE.SRGBColorSpace)
  const linear = renderColorSpace(THREE.LinearSRGBColorSpace)
  assert.ok(linear.r > srgb.r + 15, `linear sprite map should render brighter than decoded sRGB texture (${linear.r} vs ${srgb.r})`)
})

test('SpriteMaterial map applies texture UV transforms', () => {
  const map = rgbaTexture([
    255, 0, 0, 255,
    255, 0, 0, 255,
    0, 255, 0, 255,
    0, 255, 0, 255,
  ], 4, 1)
  map.offset.set(0.5, 0)
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map,
    color: 0xffffff,
  }))
  sprite.scale.set(2, 2, 1)
  scene.add(sprite)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 18, 28, 26, 36)
  assert.ok(mean.g > mean.r + 40, `sprite map offset should shift left sprite UVs from red to green (${mean.g} vs ${mean.r})`)
})

test('SpriteMaterial map honors explicit texture matrices', () => {
  const map = rgbaTexture([
    255, 0, 0, 255,
    255, 0, 0, 255,
    0, 255, 0, 255,
    0, 255, 0, 255,
  ], 4, 1)
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter
  setTextureMatrixOffset(map, 0.5)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map,
    color: 0xffffff,
  }))
  sprite.scale.set(2, 2, 1)
  scene.add(sprite)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 18, 28, 26, 36)
  assert.ok(mean.g > mean.r + 40, `explicit sprite map matrix should shift left sprite UVs from red to green (${mean.g} vs ${mean.r})`)
})

test('SpriteMaterial map honors horizontal and vertical repeat wrapping', () => {
  const renderer = new Renderer()

  function renderWithWrapping({ wrapS = THREE.ClampToEdgeWrapping, wrapT = THREE.ClampToEdgeWrapping, vertical = false }) {
    const map = vertical
      ? rgbaTexture([
        0, 255, 0, 255,
        0, 255, 0, 255,
        255, 0, 0, 255,
        255, 0, 0, 255,
      ], 2, 2)
      : rgbaTexture([
        0, 255, 0, 255,
        255, 0, 0, 255,
        0, 255, 0, 255,
        255, 0, 0, 255,
      ], 2, 2)
    map.wrapS = wrapS
    map.wrapT = wrapT
    map.offset.set(vertical ? 0 : 1, vertical ? 1 : 0)
    map.magFilter = THREE.NearestFilter
    map.minFilter = THREE.NearestFilter

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
      map,
      color: 0xffffff,
    }))
    sprite.scale.set(2, 2, 1)
    scene.add(sprite)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    const rgba = renderer.render(scene, camera, { width: 64, height: 64, format: 'rgba' })
    return vertical
      ? meanRegion(rgba, 64, 64, 28, 38, 36, 46)
      : meanRegion(rgba, 64, 64, 18, 28, 26, 36)
  }

  const clamped = renderWithWrapping({ wrapS: THREE.ClampToEdgeWrapping })
  const repeated = renderWithWrapping({ wrapS: THREE.RepeatWrapping })
  const mirrored = renderWithWrapping({ wrapS: THREE.MirroredRepeatWrapping })
  assert.ok(clamped.r > clamped.g + 100, `clamped sprite map U coordinates should sample the red edge texel (${clamped.r} vs ${clamped.g})`)
  assert.ok(repeated.g > repeated.r + 60, `repeated sprite map U coordinates should wrap to the green texel (${repeated.g} vs ${repeated.r})`)
  assert.ok(mirrored.r > mirrored.g + 100, `mirrored sprite map U coordinates should reflect to the red texel (${mirrored.r} vs ${mirrored.g})`)

  const clampedVertical = renderWithWrapping({ wrapT: THREE.ClampToEdgeWrapping, vertical: true })
  const repeatedVertical = renderWithWrapping({ wrapT: THREE.RepeatWrapping, vertical: true })
  const mirroredVertical = renderWithWrapping({ wrapT: THREE.MirroredRepeatWrapping, vertical: true })
  assert.ok(clampedVertical.r > clampedVertical.g + 100, `clamped sprite map V coordinates should sample the red edge texel (${clampedVertical.r} vs ${clampedVertical.g})`)
  assert.ok(repeatedVertical.g > repeatedVertical.r + 60, `repeated sprite map V coordinates should wrap to the green texel (${repeatedVertical.g} vs ${repeatedVertical.r})`)
  assert.ok(mirroredVertical.r > mirroredVertical.g + 100, `mirrored sprite map V coordinates should reflect to the red texel (${mirroredVertical.r} vs ${mirroredVertical.g})`)
})

test('SpriteMaterial maps use generated sprite UVs for non-primary texture channels', () => {
  const map = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter
  map.channel = 1

  const alphaMap = rgbaTexture([
    255, 255, 255, 255,
    255, 255, 255, 255,
  ], 2, 1)
  alphaMap.channel = 1

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map,
    alphaMap,
    color: 0xffffff,
  }))
  sprite.scale.set(2, 2, 1)
  scene.add(sprite)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 64, height: 64 })
  const left = meanRegion(rgba, 64, 64, 16, 26, 26, 38)
  const right = meanRegion(rgba, 64, 64, 38, 26, 48, 38)
  assert.ok(left.r > left.g + 40, `left generated sprite UVs should sample red (${left.r} vs ${left.g})`)
  assert.ok(right.g > right.r + 40, `right generated sprite UVs should sample green (${right.g} vs ${right.r})`)
})

test('SpriteMaterial alphaMap applies texture UV transforms', () => {
  const alphaMap = rgbaTexture([
    255, 0, 255, 255,
    255, 255, 255, 255,
  ], 2, 1)
  alphaMap.offset.set(0.5, 0)
  alphaMap.magFilter = THREE.NearestFilter
  alphaMap.minFilter = THREE.NearestFilter

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 1)
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    color: 0x00ff00,
    alphaMap,
    alphaTest: 0.5,
  }))
  sprite.scale.set(2, 2, 1)
  scene.add(sprite)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 18, 28, 26, 36)
  assert.ok(mean.g > mean.b + 40, `sprite alphaMap offset should shift left sprite UVs into the opaque texel (${mean.g} vs ${mean.b})`)
})

test('SpriteMaterial alphaMap honors explicit texture matrices', () => {
  const alphaMap = rgbaTexture([
    255, 0, 255, 255,
    255, 255, 255, 255,
  ], 2, 1)
  alphaMap.magFilter = THREE.NearestFilter
  alphaMap.minFilter = THREE.NearestFilter
  setTextureMatrixOffset(alphaMap, 0.5)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 1)
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    color: 0x00ff00,
    alphaMap,
    alphaTest: 0.5,
  }))
  sprite.scale.set(2, 2, 1)
  scene.add(sprite)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 18, 28, 26, 36)
  assert.ok(mean.g > mean.b + 40, `explicit sprite alphaMap matrix should shift left sprite UVs into the opaque texel (${mean.g} vs ${mean.b})`)
})

test('SpriteMaterial alphaMap honors horizontal and vertical repeat wrapping', () => {
  const renderer = new Renderer()

  function renderWithWrapping({ wrapS = THREE.ClampToEdgeWrapping, wrapT = THREE.ClampToEdgeWrapping, vertical = false }) {
    const alphaMap = vertical
      ? rgbaTexture([
        255, 255, 255, 255,
        255, 255, 255, 255,
        255, 0, 255, 255,
        255, 0, 255, 255,
      ], 2, 2)
      : rgbaTexture([
        255, 255, 255, 255,
        255, 0, 255, 255,
        255, 255, 255, 255,
        255, 0, 255, 255,
      ], 2, 2)
    alphaMap.wrapS = wrapS
    alphaMap.wrapT = wrapT
    alphaMap.offset.set(vertical ? 0 : 1, vertical ? 1 : 0)
    alphaMap.magFilter = THREE.NearestFilter
    alphaMap.minFilter = THREE.NearestFilter

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 1)
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
      color: 0x00ff00,
      alphaMap,
      alphaTest: 0.5,
    }))
    sprite.scale.set(2, 2, 1)
    scene.add(sprite)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    const rgba = renderer.render(scene, camera, { width: 64, height: 64, format: 'rgba' })
    return vertical
      ? meanRegion(rgba, 64, 64, 28, 38, 36, 46)
      : meanRegion(rgba, 64, 64, 18, 28, 26, 36)
  }

  const clamped = renderWithWrapping({ wrapS: THREE.ClampToEdgeWrapping })
  const repeated = renderWithWrapping({ wrapS: THREE.RepeatWrapping })
  const mirrored = renderWithWrapping({ wrapS: THREE.MirroredRepeatWrapping })
  assert.ok(clamped.b > clamped.g + 40, `clamped sprite alphaMap U coordinates should discard against the blue background (${clamped.b} vs ${clamped.g})`)
  assert.ok(repeated.g > repeated.b + 40, `repeated sprite alphaMap U coordinates should wrap to the opaque texel (${repeated.g} vs ${repeated.b})`)
  assert.ok(mirrored.b > mirrored.g + 40, `mirrored sprite alphaMap U coordinates should reflect to the transparent texel (${mirrored.b} vs ${mirrored.g})`)

  const clampedVertical = renderWithWrapping({ wrapT: THREE.ClampToEdgeWrapping, vertical: true })
  const repeatedVertical = renderWithWrapping({ wrapT: THREE.RepeatWrapping, vertical: true })
  const mirroredVertical = renderWithWrapping({ wrapT: THREE.MirroredRepeatWrapping, vertical: true })
  assert.ok(clampedVertical.b > clampedVertical.g + 40, `clamped sprite alphaMap V coordinates should discard against the blue background (${clampedVertical.b} vs ${clampedVertical.g})`)
  assert.ok(repeatedVertical.g > repeatedVertical.b + 40, `repeated sprite alphaMap V coordinates should wrap to the opaque texel (${repeatedVertical.g} vs ${repeatedVertical.b})`)
  assert.ok(mirroredVertical.b > mirroredVertical.g + 40, `mirrored sprite alphaMap V coordinates should reflect to the transparent texel (${mirroredVertical.b} vs ${mirroredVertical.g})`)
})

test('SpriteMaterial and PointsMaterial alphaHash produce main-pass stochastic coverage', () => {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  function renderBillboard(kind, alphaHash) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const materialProps = {
      alphaHash,
      color: 0xffffff,
      opacity: alphaHash ? 0.35 : 1,
    }

    if (kind === 'sprite') {
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial(materialProps))
      sprite.scale.set(1.2, 1.2, 1)
      scene.add(sprite)
    } else {
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))
      scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({
        ...materialProps,
        size: 48,
        sizeAttenuation: false,
      })))
    }

    return renderRgba(scene, camera, { width: 64, height: 64 })
  }

  for (const kind of ['sprite', 'points']) {
    const opaque = renderBillboard(kind, false)
    const hashed = renderBillboard(kind, true)
    const visiblePixel = (r, g, b) => r > 20 || g > 20 || b > 20
    const opaquePixels = countRegionPixels(opaque, 64, 64, 16, 16, 48, 48, visiblePixel)
    const hashedPixels = countRegionPixels(hashed, 64, 64, 16, 16, 48, 48, visiblePixel)

    assert.ok(opaquePixels > 700, `${kind} opaque billboard should fill the sampled region (${opaquePixels})`)
    assert.ok(hashedPixels > 80, `${kind} alphaHash billboard should retain some visible pixels (${hashedPixels})`)
    assert.ok(hashedPixels < opaquePixels - 180, `${kind} alphaHash billboard should discard visible pixels (${hashedPixels} vs ${opaquePixels})`)
  }
})

test('SpriteMaterial and PointsMaterial alphaToCoverage produce 4x-MSAA main-pass coverage', () => {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  function renderBillboard(kind, alphaToCoverage, sampleCount = 4) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const materialProps = {
      alphaToCoverage,
      color: 0xffffff,
      opacity: 0.5,
      transparent: false,
    }

    if (kind === 'sprite') {
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial(materialProps))
      sprite.scale.set(1.2, 1.2, 1)
      scene.add(sprite)
    } else {
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))
      scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({
        ...materialProps,
        size: 48,
        sizeAttenuation: false,
      })))
    }

    return meanRegion(renderRgba(scene, camera, {
      width: 64,
      height: 64,
      sampleCount,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 64, 64, 24, 24, 40, 40)
  }

  for (const kind of ['sprite', 'points']) {
    const noCoverage = renderBillboard(kind, false)
    const coverage = renderBillboard(kind, true)
    const singleSample = renderBillboard(kind, true, 1)

    assert.ok(noCoverage.r > 170, `${kind} non-A2C path should keep bright RGB despite opacity alpha (${noCoverage.r})`)
    assert.ok(Math.abs(singleSample.r - noCoverage.r) < 5, `${kind} single-sample alphaToCoverage should not alter RGB coverage (${singleSample.r} vs ${noCoverage.r})`)
    assert.ok(coverage.r > 30 && coverage.r < noCoverage.r - 80, `${kind} 4x alphaToCoverage should resolve partial RGB coverage (${coverage.r} vs ${noCoverage.r})`)
  }
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

test('SpriteMaterial sizeAttenuation=false keeps perspective sprite size depth independent', () => {
  function renderSprite(z, sizeAttenuation) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const material = new THREE.SpriteMaterial({ color: 0xffffff, sizeAttenuation })
    const sprite = new THREE.Sprite(material)
    sprite.position.z = z
    sprite.scale.set(0.2, 0.2, 1)
    scene.add(sprite)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return nonBackgroundBounds(renderRgba(scene, camera, { width: 96, height: 96 }), 96, 96, [0, 0, 0])
  }

  const nearAttenuated = renderSprite(0, true)
  const farAttenuated = renderSprite(-3, true)
  const nearFixed = renderSprite(0, false)
  const farFixed = renderSprite(-3, false)

  assert.ok(
    nearAttenuated.width >= farAttenuated.width * 1.7,
    `default perspective sprite should shrink with distance (${nearAttenuated.width} vs ${farAttenuated.width})`,
  )
  assert.ok(
    Math.abs(nearFixed.width - farFixed.width) <= 2,
    `sizeAttenuation=false sprite should keep width stable with distance (${nearFixed.width} vs ${farFixed.width})`,
  )
  assert.ok(
    Math.abs(nearFixed.height - farFixed.height) <= 2,
    `sizeAttenuation=false sprite should keep height stable with distance (${nearFixed.height} vs ${farFixed.height})`,
  )
})

test('Sprite center shifts billboard anchoring around object position', () => {
  function renderCenteredSprite(centerX, centerY) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ color: 0xffffff }))
    sprite.center.set(centerX, centerY)
    sprite.scale.set(0.8, 0.8, 1)
    scene.add(sprite)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return nonBackgroundBounds(renderRgba(scene, camera, { width: 96, height: 96 }), 96, 96, [0, 0, 0])
  }

  const centered = renderCenteredSprite(0.5, 0.5)
  const lowerLeft = renderCenteredSprite(0, 0)
  const upperRight = renderCenteredSprite(1, 1)

  assert.ok(lowerLeft.minX > centered.minX + 10, `center=(0,0) should anchor the sprite to the right of its origin (${lowerLeft.minX} vs ${centered.minX})`)
  assert.ok(lowerLeft.maxY < centered.maxY - 10, `center=(0,0) should anchor the sprite above its origin (${lowerLeft.maxY} vs ${centered.maxY})`)
  assert.ok(upperRight.maxX < centered.maxX - 10, `center=(1,1) should anchor the sprite to the left of its origin (${upperRight.maxX} vs ${centered.maxX})`)
  assert.ok(upperRight.minY > centered.minY + 10, `center=(1,1) should anchor the sprite below its origin (${upperRight.minY} vs ${centered.minY})`)
})

test('invalid billboard and line scalar values fail clearly', () => {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  function spriteScene(mutator) {
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ color: 0xffffff }))
    mutator(sprite, sprite.material)
    const scene = new THREE.Scene()
    scene.add(sprite)
    return scene
  }

  function pointsScene(mutator) {
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))
    const material = new THREE.PointsMaterial({ color: 0xffffff })
    mutator(material)
    const scene = new THREE.Scene()
    scene.add(new THREE.Points(geometry, material))
    return scene
  }

  function lineScene(material, configureGeometry = () => {}) {
    const geometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-0.5, 0, 0),
      new THREE.Vector3(0.5, 0, 0),
    ])
    configureGeometry(geometry)
    const scene = new THREE.Scene()
    scene.add(new THREE.Line(geometry, material))
    return scene
  }

  const cases = [
    ['sprite center', () => spriteScene((sprite) => {
      sprite.center.x = 'left'
    }), /Sprite\.center\.x must be a finite number/i],
    ['sprite scale x', () => spriteScene((sprite) => {
      sprite.scale.x = 'wide'
    }), /Sprite\.scale\.x must be a finite number/i],
    ['sprite scale z', () => spriteScene((sprite) => {
      sprite.scale.z = Number.NaN
    }), /Sprite\.scale\.z must be a finite number/i],
    ['sprite rotation', () => spriteScene((_sprite, material) => {
      material.rotation = Number.NaN
    }), /material\.rotation must be a finite number/i],
    ['point size', () => pointsScene((material) => {
      material.size = 'large'
    }), /material\.size must be a finite number/i],
    ['point size zero', () => pointsScene((material) => {
      material.size = 0
    }), /material\.size must be positive/i],
    ['point size negative', () => pointsScene((material) => {
      material.size = -1
    }), /material\.size must be positive/i],
    ['line width', () => {
      const material = new THREE.LineBasicMaterial({ color: 0xffffff })
      material.linewidth = Number.POSITIVE_INFINITY
      return lineScene(material)
    }, /material\.linewidth must be a finite number/i],
    ['line width zero', () => {
      const material = new THREE.LineBasicMaterial({ color: 0xffffff })
      material.linewidth = 0
      return lineScene(material)
    }, /material\.linewidth must be positive/i],
    ['dash size', () => {
      const material = new THREE.LineDashedMaterial({ color: 0xffffff })
      material.dashSize = 'long'
      return lineScene(material)
    }, /material\.dashSize must be a finite number/i],
    ['dash size negative', () => {
      const material = new THREE.LineDashedMaterial({ color: 0xffffff })
      material.dashSize = -0.1
      return lineScene(material)
    }, /material\.dashSize must be positive/i],
    ['dash gap', () => {
      const material = new THREE.LineDashedMaterial({ color: 0xffffff })
      material.gapSize = Number.NaN
      return lineScene(material)
    }, /material\.gapSize must be a finite number/i],
    ['dash gap negative', () => {
      const material = new THREE.LineDashedMaterial({ color: 0xffffff })
      material.gapSize = -0.1
      return lineScene(material)
    }, /material\.gapSize must be non-negative/i],
    ['dash scale', () => {
      const material = new THREE.LineDashedMaterial({ color: 0xffffff })
      material.scale = 'fast'
      return lineScene(material)
    }, /material\.scale must be a finite number/i],
    ['dash scale zero', () => {
      const material = new THREE.LineDashedMaterial({ color: 0xffffff })
      material.scale = 0
      return lineScene(material)
    }, /material\.scale must be positive/i],
    ['line distance finite', () => {
      const material = new THREE.LineDashedMaterial({ color: 0xffffff })
      return lineScene(material, (geometry) => {
        geometry.setAttribute('lineDistance', new THREE.BufferAttribute(new Float32Array([0, Number.NaN]), 1))
      })
    }, /geometry\.attributes\.lineDistance\[1\]\.x must be a finite number/i],
    ['line distance itemSize', () => {
      const material = new THREE.LineDashedMaterial({ color: 0xffffff })
      return lineScene(material, (geometry) => {
        geometry.setAttribute('lineDistance', {
          count: 2,
          itemSize: 0,
          array: new Float32Array([0, 1]),
        })
      })
    }, /geometry\.attributes\.lineDistance\.itemSize must be a positive integer/i],
  ]

  for (const [label, makeScene, pattern] of cases) {
    assert.throws(
      () => renderRgba(makeScene(), camera, { width: 64, height: 64 }),
      pattern,
      label,
    )
  }
})

test('Sprite receiveShadow is accepted as an unlit WebGL-compatible no-op', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const receiver = new THREE.Sprite(new THREE.SpriteMaterial({ color: 0xffffff }))
  receiver.receiveShadow = true
  receiver.scale.set(1.2, 1.2, 1)
  scene.add(receiver)

  const mean = meanRegion(renderRgba(scene, makeCamera(), { width: 64, height: 64 }), 64, 64, 24, 24, 40, 40)
  assert.ok(mean.r > 180 && mean.g > 180 && mean.b > 180, `sprite receiveShadow no-op should still render the unlit billboard (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('Sprite casts point-light shadows from expanded billboard quads', () => {
  function renderSpriteShadow(castShadow) {
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
    caster.position.set(0, 2.2, 1.8)
    caster.scale.set(2.8, 2.8, 1)
    caster.castShadow = castShadow
    scene.add(caster)

    const light = new THREE.PointLight(0xffffff, 2)
    light.position.set(0, 5, 4)
    light.distance = 12
    light.castShadow = true
    light.shadow.mapSize.set(256, 256)
    light.shadow.camera.near = 0.1
    light.shadow.camera.far = 12
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, { width: 96, height: 96 }), 96, 96, 28, 42, 68, 82)
  }

  const unshadowed = renderSpriteShadow(false)
  const shadowed = renderSpriteShadow(true)
  const unshadowedLum = unshadowed.r + unshadowed.g + unshadowed.b
  const shadowedLum = shadowed.r + shadowed.g + shadowed.b
  assert.ok(shadowedLum < unshadowedLum - 15, `sprite point-light shadow should darken the receiver (${shadowedLum} vs ${unshadowedLum})`)
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

test('camera layers filter direct lights', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(3, 3),
    new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, metalness: 0 }),
  )
  mesh.layers.set(1)
  scene.add(mesh)

  const redFiltered = new THREE.DirectionalLight(0xff0000, 8)
  redFiltered.position.set(0, 0, 3)
  redFiltered.layers.set(0)
  scene.add(redFiltered)

  const greenVisible = new THREE.DirectionalLight(0x00ff00, 4)
  greenVisible.position.set(0, 0, 3)
  greenVisible.layers.set(1)
  scene.add(greenVisible)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  camera.layers.set(1)

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  assert.ok(mean.g > mean.r + 25, `camera layer should select the green light and ignore red (${mean.g} vs ${mean.r})`)
})

test('invalid layer containers and masks fail clearly', () => {
  const camera = makeCamera()

  const objectContainerScene = new THREE.Scene()
  const objectWithBadLayers = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial())
  objectWithBadLayers.layers = 'visible'
  objectContainerScene.add(objectWithBadLayers)
  assert.throws(
    () => renderRgba(objectContainerScene, camera, { width: 32, height: 32 }),
    /object\.layers must be a layers-like object/i,
  )

  const cameraContainerScene = new THREE.Scene()
  cameraContainerScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial()))
  const badLayersCamera = makeCamera()
  badLayersCamera.layers = []
  assert.throws(
    () => renderRgba(cameraContainerScene, badLayersCamera, { width: 32, height: 32 }),
    /camera\.layers must be a layers-like object/i,
  )

  const objectScene = new THREE.Scene()
  const object = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial())
  object.layers.mask = 'visible'
  objectScene.add(object)
  assert.throws(
    () => renderRgba(objectScene, camera, { width: 32, height: 32 }),
    /object\.layers\.mask must be a finite number/i,
  )

  const cameraScene = new THREE.Scene()
  cameraScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial()))
  const invalidCamera = makeCamera()
  invalidCamera.layers.mask = 'camera'
  assert.throws(
    () => renderRgba(cameraScene, invalidCamera, { width: 32, height: 32 }),
    /camera\.layers\.mask must be a finite number/i,
  )

  const lightScene = new THREE.Scene()
  const light = new THREE.DirectionalLight(0xffffff, 1)
  light.layers.mask = 'bright'
  lightScene.add(light)
  assert.throws(
    () => extractLights(lightScene, camera),
    /object\.layers\.mask must be a finite number/i,
  )

  const ambientScene = new THREE.Scene()
  const ambient = new THREE.AmbientLight(0xffffff, 1)
  ambient.layers.mask = 'ambient'
  ambientScene.add(ambient)
  assert.throws(
    () => extractAmbientLight(ambientScene, camera),
    /object\.layers\.mask must be a finite number/i,
  )

  const probeScene = new THREE.Scene()
  const probe = new THREE.LightProbe(undefined, 1)
  probe.layers.mask = Number.NaN
  probeScene.add(probe)
  assert.throws(
    () => extractLightProbe(probeScene, camera),
    /object\.layers\.mask must be a finite number/i,
  )
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

test('opaque sorting honors material variant before depth', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const material = new THREE.MeshBasicMaterial({ color: 0xffffff, vertexColors: true })

  const instanced = new THREE.InstancedMesh(new THREE.PlaneGeometry(2, 2), material, 1)
  instanced.setMatrixAt(0, new THREE.Matrix4())
  instanced.setColorAt(0, new THREE.Color(0, 0, 1))
  scene.add(instanced)

  const redGeometry = new THREE.PlaneGeometry(2, 2)
  const redColors = new Float32Array(redGeometry.getAttribute('position').count * 3)
  for (let i = 0; i < redColors.length; i += 3) {
    redColors[i] = 1
  }
  redGeometry.setAttribute('color', new THREE.BufferAttribute(redColors, 3))
  scene.add(new THREE.Mesh(redGeometry, material))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 24, 24, 40, 40)
  assert.ok(mean.b > mean.r + 60, `instanced material variant should draw after the normal mesh (${mean.b} vs ${mean.r})`)
})

test('transmissive bucket renders before ordinary transparent bucket', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 1)

  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({
      color: 0xff0000,
      opacity: 1,
      transparent: true,
      depthWrite: false,
    }),
  ))

  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshPhysicalMaterial({
      color: 0xffffff,
      roughness: 0.05,
      transmission: 1,
      thickness: 0.2,
    }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 24, 24, 40, 40)
  assert.ok(mean.r > mean.b + 160, `ordinary transparent red should draw after transmissive blue (${mean.r} vs ${mean.b})`)
})

test('Renderer.setOpaqueSort overrides opaque draw ordering', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const material = new THREE.MeshBasicMaterial({ color: 0xffffff, vertexColors: true, depthWrite: false })
  const redGeometry = new THREE.PlaneGeometry(2, 2)
  const redColors = new Float32Array(redGeometry.getAttribute('position').count * 3)
  for (let i = 0; i < redColors.length; i += 3) {
    redColors[i] = 1
  }
  redGeometry.setAttribute('color', new THREE.BufferAttribute(redColors, 3))
  const red = new THREE.Mesh(redGeometry, material)
  scene.add(red)

  const blueGeometry = new THREE.PlaneGeometry(2, 2)
  const blueColors = new Float32Array(blueGeometry.getAttribute('position').count * 3)
  for (let i = 0; i < blueColors.length; i += 3) {
    blueColors[i + 2] = 1
  }
  blueGeometry.setAttribute('color', new THREE.BufferAttribute(blueColors, 3))
  const blue = new THREE.Mesh(blueGeometry, material)
  scene.add(blue)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const renderer = getRenderer()
  renderer.setOpaqueSort((a, b) => b.id - a.id)
  let mean
  try {
    mean = meanRegion(renderer.render(scene, camera, { width: 64, height: 64, format: 'rgba' }), 64, 64, 24, 24, 40, 40)
  } finally {
    renderer.setOpaqueSort(null)
  }
  assert.ok(mean.r > mean.b + 160, `custom opaque sort should draw red after blue (${mean.r} vs ${mean.b})`)
})

test('Renderer.setTransparentSort overrides transparent depth sorting', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const red = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xff0000, transparent: true, depthWrite: false }),
  )
  red.position.z = 0.35
  scene.add(red)

  const blue = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0x0000ff, transparent: true, depthWrite: false }),
  )
  blue.position.z = -0.35
  scene.add(blue)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const renderer = getRenderer()
  renderer.setTransparentSort((a, b) => a.id - b.id)
  let mean
  try {
    mean = meanRegion(renderer.render(scene, camera, { width: 64, height: 64, format: 'rgba' }), 64, 64, 24, 24, 40, 40)
  } finally {
    renderer.setTransparentSort(null)
  }
  assert.ok(mean.b > mean.r + 160, `custom transparent sort should draw blue after red (${mean.b} vs ${mean.r})`)
})

test('opaque sort callbacks receive geometry group render items', () => {
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -1, -1, 0,
    1, -1, 0,
    1, 1, 0,
    -1, -1, 0,
    1, 1, 0,
    -1, 1, 0,
    -1, -1, 0,
    1, -1, 0,
    1, 1, 0,
    -1, -1, 0,
    1, 1, 0,
    -1, 1, 0,
  ]), 3))
  geometry.addGroup(0, 6, 0)
  geometry.addGroup(6, 6, 1)

  const materials = [
    new THREE.MeshBasicMaterial({ color: 0xff0000, depthTest: false }),
    new THREE.MeshBasicMaterial({ color: 0x0000ff, depthTest: false }),
  ]

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(geometry, materials))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const seenGroups = new Set()
  const seenMaterials = new Set()
  const opaqueSort = (a, b) => {
    assert.equal(a.geometry, geometry)
    assert.equal(b.geometry, geometry)
    assert.ok(a.group)
    assert.ok(b.group)
    seenGroups.add(a.group.materialIndex)
    seenGroups.add(b.group.materialIndex)
    seenMaterials.add(a.material)
    seenMaterials.add(b.material)
    return b.group.materialIndex - a.group.materialIndex
  }

  const rgba = renderRgba(scene, camera, { width: 64, height: 64, opaqueSort })
  assert.deepEqual([...seenGroups].sort(), [0, 1])
  assert.deepEqual([...seenMaterials].sort((a, b) => materials.indexOf(a) - materials.indexOf(b)), materials)
  const mean = meanRegion(rgba, 64, 64, 24, 24, 40, 40)
  assert.ok(mean.r > mean.b + 160, `custom group-aware opaque sort should draw red after blue (${mean.r} vs ${mean.b})`)
})

test('transparent sort callbacks receive geometry group render items', () => {
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -1, -1, 0,
    1, -1, 0,
    1, 1, 0,
    -1, -1, 0,
    1, 1, 0,
    -1, 1, 0,
    -1, -1, 0,
    1, -1, 0,
    1, 1, 0,
    -1, -1, 0,
    1, 1, 0,
    -1, 1, 0,
  ]), 3))
  geometry.addGroup(0, 6, 0)
  geometry.addGroup(6, 6, 1)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(geometry, [
    new THREE.MeshBasicMaterial({ color: 0xff0000, opacity: 0.55, transparent: true, depthWrite: false }),
    new THREE.MeshBasicMaterial({ color: 0x0000ff, opacity: 0.55, transparent: true, depthWrite: false }),
  ]))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const seenGroups = new Set()
  const transparentSort = (a, b) => {
    assert.equal(a.geometry, geometry)
    assert.equal(b.geometry, geometry)
    assert.ok(a.group)
    assert.ok(b.group)
    seenGroups.add(a.group.materialIndex)
    seenGroups.add(b.group.materialIndex)
    return b.group.materialIndex - a.group.materialIndex
  }

  const rgba = renderRgba(scene, camera, { width: 64, height: 64, transparentSort })
  assert.deepEqual([...seenGroups].sort(), [0, 1])
  const mean = meanRegion(rgba, 64, 64, 24, 24, 40, 40)
  assert.ok(mean.r > mean.b + 10, `custom group-aware transparent sort should draw red after blue (${mean.r} vs ${mean.b})`)
})

test('sortObjects=false preserves traversal order within transparent bucket', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const red = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xff0000, transparent: true, depthWrite: false }),
  )
  red.position.z = 0.35
  scene.add(red)

  const blue = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0x0000ff, transparent: true, depthWrite: false }),
  )
  blue.position.z = -0.35
  scene.add(blue)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const renderer = getRenderer()
  renderer.sortObjects = false
  let mean
  try {
    mean = meanRegion(renderer.render(scene, camera, { width: 64, height: 64, format: 'rgba' }), 64, 64, 24, 24, 40, 40)
  } finally {
    renderer.sortObjects = true
  }
  assert.ok(mean.b > mean.r + 160, `sortObjects=false should leave blue after red traversal order (${mean.b} vs ${mean.r})`)
})

test('invalid sort controls fail clearly', () => {
  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial()))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32, sortObjects: 'yes' }),
    /options\.sortObjects must be a boolean/i,
  )
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32, opaqueSort: 'front' }),
    /options\.opaqueSort must be a function or null/i,
  )
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32, transparentSort: 1 }),
    /options\.transparentSort must be a function or null/i,
  )

  const renderer = getRenderer()
  assert.throws(
    () => { renderer.sortObjects = 'yes' },
    /Renderer\.sortObjects must be a boolean/i,
  )
  assert.throws(
    () => renderer.setOpaqueSort('front'),
    /Renderer\.setOpaqueSort expects a function or null/i,
  )
  assert.throws(
    () => renderer.setTransparentSort(1),
    /Renderer\.setTransparentSort expects a function or null/i,
  )
})

test('invalid renderOrder values fail clearly', () => {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const cases = [
    ['mesh renderOrder', () => {
      const scene = new THREE.Scene()
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial())
      mesh.renderOrder = Number.NaN
      scene.add(mesh)
      return scene
    }],
    ['group renderOrder', () => {
      const scene = new THREE.Scene()
      const group = new THREE.Group()
      group.renderOrder = 'front'
      group.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial()))
      scene.add(group)
      return scene
    }],
  ]

  for (const [label, makeScene] of cases) {
    assert.throws(
      () => renderRgba(makeScene(), camera, { width: 64, height: 64 }),
      /object\.renderOrder must be a finite number/i,
      label,
    )
  }
})

test('transparent sort depth uses geometry bounding sphere center', () => {
  function offsetPlane(zOffset, color) {
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
      -1, -1, zOffset,
      1, -1, zOffset,
      1, 1, zOffset,
      -1, 1, zOffset,
    ]), 3))
    geometry.setIndex([0, 1, 2, 0, 2, 3])

    return new THREE.Mesh(
      geometry,
      new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.6,
        depthWrite: false,
      }),
    )
  }

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(offsetPlane(0.45, 0xff0000))
  scene.add(offsetPlane(-0.45, 0x0000ff))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 24, 24, 40, 40)
  assert.ok(mean.r > mean.b + 20, `near red geometry center should sort over far blue despite matching object origins (${mean.r} vs ${mean.b})`)
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

test('transparent materials honor default depthWrite=true', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const front = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xff0000, opacity: 0.75, transparent: true }),
  )
  front.position.z = 0.2
  front.renderOrder = 0
  scene.add(front)

  const behind = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0x0000ff, opacity: 0.75, transparent: true }),
  )
  behind.position.z = -0.2
  behind.renderOrder = 1
  scene.add(behind)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 24, 24, 40, 40)
  assert.ok(mean.r > mean.b + 60, `default transparent depthWrite should reject the later blue draw behind red (${mean.r} vs ${mean.b})`)
})

test('material depthFunc controls depth comparison', () => {
  function renderBehind(depthFunc) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)

    const front = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ color: 0xff0000 }),
    )
    front.position.z = 0.2
    front.renderOrder = 0
    scene.add(front)

    const behind = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ color: 0x0000ff, depthFunc }),
    )
    behind.position.z = -0.2
    behind.renderOrder = 1
    scene.add(behind)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 24, 24, 40, 40)
  }

  const lessEqual = renderBehind(THREE.LessEqualDepth)
  const always = renderBehind(THREE.AlwaysDepth)
  const greater = renderBehind(THREE.GreaterDepth)

  assert.ok(lessEqual.r > lessEqual.b + 80, `LessEqualDepth should reject the later blue plane behind red (${lessEqual.r} vs ${lessEqual.b})`)
  assert.ok(always.b > always.r + 80, `AlwaysDepth should render the later blue plane over red (${always.b} vs ${always.r})`)
  assert.ok(greater.b > greater.r + 80, `GreaterDepth should pass the farther blue depth over red (${greater.b} vs ${greater.r})`)
})

test('unsupported material depthFunc values fail clearly', () => {
  const material = new THREE.MeshBasicMaterial({ color: 0xffffff })
  material.depthFunc = 999

  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

  assert.throws(
    () => renderRgba(scene, makeCamera(), { width: 32, height: 32 }),
    /material\.depthFunc 999.*not supported/i,
  )
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

function assertMaterialRenderStateFails(material, pattern) {
  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

  assert.throws(
    () => renderRgba(scene, makeCamera(), { width: 32, height: 32 }),
    pattern,
  )
}

test('unsupported material blending values fail clearly', () => {
  const material = new THREE.MeshBasicMaterial({ color: 0xffffff })
  material.blending = 999

  assertMaterialRenderStateFails(
    material,
    /material\.blending 999.*not supported/i,
  )
})

test('unsupported custom blending constants fail clearly', () => {
  for (const field of ['blendEquation', 'blendEquationAlpha']) {
    const material = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      blending: THREE.CustomBlending,
    })
    material[field] = 999

    assertMaterialRenderStateFails(
      material,
      new RegExp(`material\\.${field} 999.*not supported`, 'i'),
    )
  }

  for (const field of ['blendSrc', 'blendDst', 'blendSrcAlpha', 'blendDstAlpha']) {
    const material = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      blending: THREE.CustomBlending,
    })
    material[field] = 999

    assertMaterialRenderStateFails(
      material,
      new RegExp(`material\\.${field} 999.*not supported`, 'i'),
    )
  }
})

test('invalid material render-state numeric values fail clearly', () => {
  const cases = [
    ['blendAlpha', (material) => {
      material.blending = THREE.CustomBlending
      material.blendSrc = THREE.ConstantAlphaFactor
      material.blendAlpha = 'opaque'
    }, /material\.blendAlpha must be a finite number/i],
    ['polygonOffsetFactor', (material) => {
      material.polygonOffset = true
      material.polygonOffsetFactor = 'front'
    }, /material\.polygonOffsetFactor must be a finite number/i],
    ['polygonOffsetUnits', (material) => {
      material.polygonOffset = true
      material.polygonOffsetUnits = Number.NaN
    }, /material\.polygonOffsetUnits must be a finite number/i],
    ['stencilWriteMask', (material) => {
      material.stencilWriteMask = 'mask'
    }, /material\.stencilWriteMask must be a finite number/i],
    ['stencilRef', (material) => {
      material.stencilRef = Number.POSITIVE_INFINITY
    }, /material\.stencilRef must be a finite number/i],
    ['stencilFuncMask', (material) => {
      material.stencilFuncMask = 'mask'
    }, /material\.stencilFuncMask must be a finite number/i],
  ]

  for (const [name, mutate, pattern] of cases) {
    const material = new THREE.MeshBasicMaterial({ color: 0xffffff })
    mutate(material)
    assertMaterialRenderStateFails(material, pattern, `${name} should fail clearly`)
  }
})

test('invalid material render-state boolean values fail clearly', () => {
  const cases = [
    ['alphaHash', (material) => {
      material.alphaHash = 'yes'
    }, /material\.alphaHash must be a boolean/i],
    ['alphaToCoverage', (material) => {
      material.alphaToCoverage = 1
    }, /material\.alphaToCoverage must be a boolean/i],
    ['premultipliedAlpha', (material) => {
      material.premultipliedAlpha = 'yes'
    }, /material\.premultipliedAlpha must be a boolean/i],
    ['toneMapped', (material) => {
      material.toneMapped = 'yes'
    }, /material\.toneMapped must be a boolean/i],
    ['transparent', (material) => {
      material.transparent = 'yes'
    }, /material\.transparent must be a boolean/i],
    ['vertexColors', (material) => {
      material.vertexColors = 'yes'
    }, /material\.vertexColors must be a boolean/i],
    ['depthTest', (material) => {
      material.depthTest = 'yes'
    }, /material\.depthTest must be a boolean/i],
    ['depthWrite', (material) => {
      material.depthWrite = 1
    }, /material\.depthWrite must be a boolean/i],
    ['colorWrite', (material) => {
      material.colorWrite = 'no'
    }, /material\.colorWrite must be a boolean/i],
    ['polygonOffset', (material) => {
      material.polygonOffset = 'yes'
    }, /material\.polygonOffset must be a boolean/i],
    ['stencilWrite', (material) => {
      material.stencilWrite = 1
    }, /material\.stencilWrite must be a boolean/i],
    ['flatShading', (material) => {
      material.flatShading = 'flat'
    }, /material\.flatShading must be a boolean/i],
    ['fog', (material) => {
      material.fog = 'scene'
    }, /material\.fog must be a boolean/i],
    ['wireframe', (material) => {
      material.wireframe = 'yes'
    }, /material\.wireframe must be a boolean/i],
  ]

  for (const [name, mutate, pattern] of cases) {
    const material = new THREE.MeshBasicMaterial({ color: 0xffffff })
    mutate(material)
    assertMaterialRenderStateFails(material, pattern, `${name} should fail clearly`)
  }
})

test('invalid material sizeAttenuation values fail clearly', () => {
  const spriteScene = new THREE.Scene()
  const spriteMaterial = new THREE.SpriteMaterial({ color: 0xffffff })
  spriteMaterial.sizeAttenuation = 'no'
  spriteScene.add(new THREE.Sprite(spriteMaterial))
  assert.throws(
    () => renderRgba(spriteScene, makeCamera(), { width: 32, height: 32 }),
    /material\.sizeAttenuation must be a boolean/i,
  )

  const pointsScene = new THREE.Scene()
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))
  const pointsMaterial = new THREE.PointsMaterial({ color: 0xffffff })
  pointsMaterial.sizeAttenuation = 1
  pointsScene.add(new THREE.Points(geometry, pointsMaterial))
  assert.throws(
    () => renderRgba(pointsScene, makeCamera(), { width: 32, height: 32 }),
    /material\.sizeAttenuation must be a boolean/i,
  )
})

test('unsupported material stencil constants fail clearly', () => {
  for (const field of ['stencilFunc', 'stencilFail', 'stencilZFail', 'stencilZPass']) {
    const material = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      stencilWrite: true,
    })
    material[field] = 999

    assertMaterialRenderStateFails(
      material,
      new RegExp(`material\\.${field} 999.*not supported`, 'i'),
    )
  }
})

test('unsupported material side values fail clearly', () => {
  const material = new THREE.MeshBasicMaterial({ color: 0xffffff })
  material.side = 999

  assertMaterialRenderStateFails(
    material,
    /material\.side 999.*not supported/i,
  )
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

test('SubtractiveBlending subtracts source color from destination', () => {
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
      opacity: 1,
      transparent: true,
      blending: THREE.SubtractiveBlending,
    }),
  )
  front.position.z = 0.1
  scene.add(front)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRegion(
    renderRgba(scene, camera, { width: 64, height: 64 }),
    64,
    64,
    24,
    24,
    40,
    40,
  )
  assert.ok(
    mean.g > mean.r + 80,
    `SubtractiveBlending should subtract red from the destination (${mean.g} vs ${mean.r})`,
  )
  assert.ok(
    mean.b > mean.r + 80,
    `SubtractiveBlending should preserve non-source destination channels (${mean.b} vs ${mean.r})`,
  )
})

test('MultiplyBlending multiplies source and destination colors', () => {
  function renderBlend(blending) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const back = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ color: 0xff0000 }),
    )
    back.position.z = -0.1
    scene.add(back)
    const front = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({
        color: 0x00ff00,
        opacity: 1,
        transparent: true,
        blending,
      }),
    )
    front.position.z = 0.1
    scene.add(front)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return meanRegion(
      renderRgba(scene, camera, { width: 64, height: 64 }),
      64,
      64,
      24,
      24,
      40,
      40,
    )
  }

  const normal = renderBlend(THREE.NormalBlending)
  const multiply = renderBlend(THREE.MultiplyBlending)
  assert.ok(
    multiply.g < normal.g - 150,
    `MultiplyBlending should suppress the green source (${multiply.g} vs ${normal.g})`,
  )
  assert.ok(
    multiply.r > multiply.g + 80,
    `MultiplyBlending should preserve more destination red than source green (${multiply.r} vs ${multiply.g})`,
  )
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

test('CustomBlending honors constant color and alpha factors', () => {
  function renderConstantBlend(blendSrc, blendColor, blendAlpha) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({
        color: 0xffffff,
        transparent: true,
        blending: THREE.CustomBlending,
        blendEquation: THREE.AddEquation,
        blendSrc,
        blendDst: THREE.ZeroFactor,
        blendEquationAlpha: THREE.AddEquation,
        blendSrcAlpha: THREE.OneFactor,
        blendDstAlpha: THREE.ZeroFactor,
        blendColor,
        blendAlpha,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 24, 24, 40, 40)
  }

  const colorConstant = renderConstantBlend(THREE.ConstantColorFactor, new THREE.Color(0, 1, 0), 0.2)
  assert.ok(colorConstant.g > colorConstant.r + 40, `ConstantColorFactor should use blendColor green over red (${colorConstant.g} vs ${colorConstant.r})`)
  assert.ok(colorConstant.g > colorConstant.b + 40, `ConstantColorFactor should use blendColor green over blue (${colorConstant.g} vs ${colorConstant.b})`)

  const alphaConstant = renderConstantBlend(THREE.ConstantAlphaFactor, new THREE.Color(0, 0, 1), 0.35)
  assert.ok(
    Math.abs(alphaConstant.r - alphaConstant.g) < 12 && Math.abs(alphaConstant.g - alphaConstant.b) < 12,
    `ConstantAlphaFactor should use blendAlpha as the RGB constant (${alphaConstant.r}, ${alphaConstant.g}, ${alphaConstant.b})`,
  )
  assert.ok(alphaConstant.r > 20, `ConstantAlphaFactor should keep visible source contribution (${alphaConstant.r})`)
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

test('InstancedBufferGeometry default instanceCount expands per-instance offsets and colors', () => {
  const base = new THREE.PlaneGeometry(0.85, 0.85)
  const geometry = new THREE.InstancedBufferGeometry()
  geometry.index = base.index
  geometry.setAttribute('position', base.getAttribute('position'))
  geometry.setAttribute('uv', base.getAttribute('uv'))
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

test('normalized integer vertex colors render at full intensity', () => {
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.8, -0.5, 0,
    0, -0.5, 0,
    -0.8, 0.5, 0,
    0, -0.5, 0,
    0, 0.5, 0,
    -0.8, 0.5, 0,
    0, -0.5, 0,
    0.8, -0.5, 0,
    0, 0.5, 0,
    0.8, -0.5, 0,
    0.8, 0.5, 0,
    0, 0.5, 0,
  ]), 3))
  geometry.setAttribute('color', new THREE.BufferAttribute(new Uint8Array([
    255, 0, 0, 255,
    255, 0, 0, 255,
    255, 0, 0, 255,
    255, 0, 0, 255,
    255, 0, 0, 255,
    255, 0, 0, 255,
    0, 255, 0, 255,
    0, 255, 0, 255,
    0, 255, 0, 255,
    0, 255, 0, 255,
    0, 255, 0, 255,
    0, 255, 0, 255,
  ]), 4, true))

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: 0xffffff, vertexColors: true })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 96, height: 96 })
  const left = meanRegion(rgba, 96, 96, 28, 36, 42, 60)
  const right = meanRegion(rgba, 96, 96, 54, 36, 68, 60)
  assert.ok(left.r > 140 && left.r > left.g + 120, `normalized red vertex colors should stay bright (${left.r}, ${left.g}, ${left.b})`)
  assert.ok(right.g > 200 && right.g > right.r + 50, `normalized green vertex colors should stay bright (${right.r}, ${right.g}, ${right.b})`)
})

test('InstancedBufferGeometry honors meshPerAttribute repeat values for offsets and colors', () => {
  const base = new THREE.PlaneGeometry(0.35, 0.5)
  const geometry = new THREE.InstancedBufferGeometry()
  geometry.index = base.index
  geometry.setAttribute('position', base.getAttribute('position'))
  geometry.setAttribute('uv', base.getAttribute('uv'))
  geometry.instanceCount = 4

  const offsets = new THREE.InstancedBufferAttribute(
    new Float32Array([-0.45, 0, 0, 0.45, 0, 0]),
    3,
  )
  offsets.meshPerAttribute = 2
  geometry.setAttribute('instanceOffset', offsets)

  const colors = new THREE.InstancedBufferAttribute(
    new Float32Array([1, 0, 0, 0, 1, 0]),
    3,
  )
  colors.meshPerAttribute = 2
  geometry.setAttribute('color', colors)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: 0xffffff, vertexColors: true })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 96, height: 64 })
  const left = meanRegion(rgba, 96, 64, 26, 28, 36, 36)
  const right = meanRegion(rgba, 96, 64, 60, 28, 70, 36)
  assert.ok(left.r > left.g + 40, `repeated first instanced attributes should draw red on the left (${left.r} vs ${left.g})`)
  assert.ok(right.g > right.r + 40, `repeated second instanced attributes should draw green on the right (${right.g} vs ${right.r})`)
})

test('InstancedBufferGeometry expands instanced mesh texture UV attributes', () => {
  const base = new THREE.PlaneGeometry(0.45, 0.45)
  const geometry = new THREE.InstancedBufferGeometry()
  geometry.index = base.index
  geometry.setAttribute('position', base.getAttribute('position'))
  geometry.setAttribute('instanceOffset', new THREE.InstancedBufferAttribute(
    new Float32Array([-0.45, 0, 0, 0.45, 0, 0]),
    3,
  ))
  geometry.setAttribute('uv', new THREE.InstancedBufferAttribute(
    new Float32Array([0.25, 0.5, 0.75, 0.5]),
    2,
  ))

  const map = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: 0xffffff, map })))

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 96, height: 64 })
  const left = meanRegion(rgba, 96, 64, 20, 28, 32, 36)
  const right = meanRegion(rgba, 96, 64, 64, 28, 76, 36)
  assert.ok(left.r > left.g + 60, `left instanced mesh uv should sample red (${left.r} vs ${left.g})`)
  assert.ok(right.g > right.r + 60, `right instanced mesh uv should sample green (${right.g} vs ${right.r})`)
})

test('InstancedBufferGeometry honors meshPerAttribute for instanced texture UV attributes', () => {
  const base = new THREE.PlaneGeometry(0.25, 0.5)
  const geometry = new THREE.InstancedBufferGeometry()
  geometry.index = base.index
  geometry.setAttribute('position', base.getAttribute('position'))
  geometry.instanceCount = 4
  geometry.setAttribute('instanceOffset', new THREE.InstancedBufferAttribute(
    new Float32Array([
      -0.6, 0, 0,
      -0.2, 0, 0,
      0.2, 0, 0,
      0.6, 0, 0,
    ]),
    3,
  ))
  const uvs = new THREE.InstancedBufferAttribute(
    new Float32Array([0.25, 0.5, 0.75, 0.5]),
    2,
  )
  uvs.meshPerAttribute = 2
  geometry.setAttribute('uv', uvs)

  const map = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: 0xffffff, map })))

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 128, height: 64 })
  const first = meanRegion(rgba, 128, 64, 20, 28, 28, 36)
  const second = meanRegion(rgba, 128, 64, 45, 28, 53, 36)
  const third = meanRegion(rgba, 128, 64, 75, 28, 83, 36)
  const fourth = meanRegion(rgba, 128, 64, 100, 28, 108, 36)
  assert.ok(first.r > first.g + 50, `first repeated uv should sample red (${first.r} vs ${first.g})`)
  assert.ok(second.r > second.g + 50, `second repeated uv should sample red (${second.r} vs ${second.g})`)
  assert.ok(third.g > third.r + 50, `third repeated uv should sample green (${third.g} vs ${third.r})`)
  assert.ok(fourth.g > fourth.r + 50, `fourth repeated uv should sample green (${fourth.g} vs ${fourth.r})`)
})

test('InstancedBufferGeometry expands selected instanced texture UV channels', () => {
  const base = new THREE.PlaneGeometry(0.45, 0.45)
  const geometry = new THREE.InstancedBufferGeometry()
  geometry.index = base.index
  geometry.setAttribute('position', base.getAttribute('position'))
  geometry.setAttribute('uv', base.getAttribute('uv'))
  geometry.setAttribute('instanceOffset', new THREE.InstancedBufferAttribute(
    new Float32Array([-0.45, 0, 0, 0.45, 0, 0]),
    3,
  ))
  geometry.setAttribute('uv1', new THREE.InstancedBufferAttribute(
    new Float32Array([0.25, 0.5, 0.75, 0.5]),
    2,
  ))

  const map = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter
  map.channel = 1

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: 0xffffff, map })))

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 96, height: 64 })
  const left = meanRegion(rgba, 96, 64, 20, 28, 32, 36)
  const right = meanRegion(rgba, 96, 64, 64, 28, 76, 36)
  assert.ok(left.r > left.g + 60, `left selected instanced uv1 should sample red (${left.r} vs ${left.g})`)
  assert.ok(right.g > right.r + 60, `right selected instanced uv1 should sample green (${right.g} vs ${right.r})`)
})

test('invalid instance counts fail clearly', () => {
  const camera = makeCamera()

  const meshScene = new THREE.Scene()
  const instancedMesh = new THREE.InstancedMesh(
    new THREE.BoxGeometry(0.5, 0.5, 0.5),
    new THREE.MeshBasicMaterial({ color: 0xffffff }),
    2,
  )
  instancedMesh.count = 'many'
  meshScene.add(instancedMesh)
  assert.throws(
    () => renderRgba(meshScene, camera, { width: 64, height: 64 }),
    /InstancedMesh\.count must be a finite number/i,
  )

  instancedMesh.count = -1
  assert.throws(
    () => renderRgba(meshScene, camera, { width: 64, height: 64 }),
    /InstancedMesh\.count must be non-negative/i,
  )

  instancedMesh.count = 1.5
  assert.throws(
    () => renderRgba(meshScene, camera, { width: 64, height: 64 }),
    /InstancedMesh\.count must be an integer/i,
  )

  const base = new THREE.PlaneGeometry(0.85, 0.85)
  const geometry = new THREE.InstancedBufferGeometry()
  geometry.index = base.index
  geometry.setAttribute('position', base.getAttribute('position'))
  geometry.setAttribute('uv', base.getAttribute('uv'))
  geometry.instanceCount = Number.NaN
  const geometryScene = new THREE.Scene()
  geometryScene.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: 0xffffff })))

  assert.throws(
    () => renderRgba(geometryScene, camera, { width: 64, height: 64 }),
    /geometry\.instanceCount must be a finite number/i,
  )

  geometry.instanceCount = -1
  assert.throws(
    () => renderRgba(geometryScene, camera, { width: 64, height: 64 }),
    /geometry\.instanceCount must be non-negative/i,
  )

  geometry.instanceCount = 1.5
  assert.throws(
    () => renderRgba(geometryScene, camera, { width: 64, height: 64 }),
    /geometry\.instanceCount must be an integer/i,
  )

  const meshPerAttributeScene = new THREE.Scene()
  const meshPerAttributeGeometry = new THREE.InstancedBufferGeometry()
  meshPerAttributeGeometry.index = base.index
  meshPerAttributeGeometry.setAttribute('position', base.getAttribute('position'))
  meshPerAttributeGeometry.setAttribute('uv', base.getAttribute('uv'))
  const instanceOffset = new THREE.InstancedBufferAttribute(new Float32Array([0, 0, 0]), 3)
  instanceOffset.meshPerAttribute = 'many'
  meshPerAttributeGeometry.setAttribute('instanceOffset', instanceOffset)
  meshPerAttributeScene.add(new THREE.Mesh(meshPerAttributeGeometry, new THREE.MeshBasicMaterial({ color: 0xffffff })))

  assert.throws(
    () => renderRgba(meshPerAttributeScene, camera, { width: 64, height: 64 }),
    /geometry\.attributes\.instanceOffset\.meshPerAttribute must be a positive finite number/i,
  )

  instanceOffset.meshPerAttribute = 0
  assert.throws(
    () => renderRgba(meshPerAttributeScene, camera, { width: 64, height: 64 }),
    /geometry\.attributes\.instanceOffset\.meshPerAttribute must be a positive finite number/i,
  )

  instanceOffset.meshPerAttribute = 1.5
  assert.throws(
    () => renderRgba(meshPerAttributeScene, camera, { width: 64, height: 64 }),
    /geometry\.attributes\.instanceOffset\.meshPerAttribute must be a positive integer/i,
  )
})

test('invalid instanced attributes fail clearly', () => {
  const camera = makeCamera()
  const box = new THREE.BoxGeometry(0.5, 0.5, 0.5)
  const material = new THREE.MeshBasicMaterial({ color: 0xffffff })

  const matrixCountScene = new THREE.Scene()
  const matrixCountMesh = new THREE.InstancedMesh(box, material, 1)
  matrixCountMesh.instanceMatrix.count = 'many'
  matrixCountScene.add(matrixCountMesh)
  assert.throws(
    () => renderRgba(matrixCountScene, camera, { width: 64, height: 64 }),
    /InstancedMesh\.instanceMatrix\.count must be a non-negative integer/i,
  )

  const matrixValueScene = new THREE.Scene()
  const matrixValueMesh = new THREE.InstancedMesh(box, material, 1)
  matrixValueMesh.instanceMatrix.array[0] = Number.NaN
  matrixValueScene.add(matrixValueMesh)
  assert.throws(
    () => renderRgba(matrixValueScene, camera, { width: 64, height: 64 }),
    /InstancedMesh\.instanceMatrix\[0\]\.x must be a finite number/i,
  )

  const instanceColorScene = new THREE.Scene()
  const instanceColorMesh = new THREE.InstancedMesh(box, material, 1)
  instanceColorMesh.setColorAt(0, new THREE.Color(1, 1, 1))
  instanceColorMesh.instanceColor.count = 'many'
  instanceColorScene.add(instanceColorMesh)
  assert.throws(
    () => renderRgba(instanceColorScene, camera, { width: 64, height: 64 }),
    /InstancedMesh\.instanceColor\.count must be a non-negative integer/i,
  )

  const base = new THREE.PlaneGeometry(0.85, 0.85)
  const offsetGeometry = new THREE.InstancedBufferGeometry()
  offsetGeometry.index = base.index
  offsetGeometry.setAttribute('position', base.getAttribute('position'))
  offsetGeometry.setAttribute('uv', base.getAttribute('uv'))
  offsetGeometry.instanceCount = 1
  const instanceOffset = new THREE.InstancedBufferAttribute(new Float32Array([Number.NaN, 0, 0]), 3)
  offsetGeometry.setAttribute('instanceOffset', instanceOffset)
  const offsetScene = new THREE.Scene()
  offsetScene.add(new THREE.Mesh(offsetGeometry, material))
  assert.throws(
    () => renderRgba(offsetScene, camera, { width: 64, height: 64 }),
    /geometry\.attributes\.instanceOffset\[0\]\.x must be a finite number/i,
  )

  const colorCountGeometry = new THREE.InstancedBufferGeometry()
  colorCountGeometry.index = base.index
  colorCountGeometry.setAttribute('position', base.getAttribute('position'))
  colorCountGeometry.setAttribute('uv', base.getAttribute('uv'))
  colorCountGeometry.instanceCount = 1
  const instanceColor = new THREE.InstancedBufferAttribute(new Float32Array([1, 0, 0]), 3)
  instanceColor.count = 'many'
  colorCountGeometry.setAttribute('color', instanceColor)
  const colorCountScene = new THREE.Scene()
  colorCountScene.add(new THREE.Mesh(colorCountGeometry, new THREE.MeshBasicMaterial({ color: 0xffffff, vertexColors: true })))
  assert.throws(
    () => renderRgba(colorCountScene, camera, { width: 64, height: 64 }),
    /geometry\.attributes\.color\.count must be a non-negative integer/i,
  )

  const colorValueGeometry = new THREE.InstancedBufferGeometry()
  colorValueGeometry.index = base.index
  colorValueGeometry.setAttribute('position', base.getAttribute('position'))
  colorValueGeometry.setAttribute('uv', base.getAttribute('uv'))
  colorValueGeometry.instanceCount = 1
  colorValueGeometry.setAttribute('color', new THREE.InstancedBufferAttribute(new Float32Array([Number.NaN, 0, 0]), 3))
  const colorValueScene = new THREE.Scene()
  colorValueScene.add(new THREE.Mesh(colorValueGeometry, new THREE.MeshBasicMaterial({ color: 0xffffff, vertexColors: true })))
  assert.throws(
    () => renderRgba(colorValueScene, camera, { width: 64, height: 64 }),
    /geometry\.attributes\.color\[0\]\.x must be a finite number/i,
  )

  const uvCountGeometry = new THREE.InstancedBufferGeometry()
  uvCountGeometry.index = base.index
  uvCountGeometry.setAttribute('position', base.getAttribute('position'))
  const uvCount = new THREE.InstancedBufferAttribute(new Float32Array([0.5, 0.5]), 2)
  uvCount.count = 'many'
  uvCountGeometry.setAttribute('uv', uvCount)
  const uvCountScene = new THREE.Scene()
  uvCountScene.add(new THREE.Mesh(uvCountGeometry, material))
  assert.throws(
    () => renderRgba(uvCountScene, camera, { width: 64, height: 64 }),
    /geometry\.attributes\.uv\.count must be a non-negative integer/i,
  )

  const uvValueGeometry = new THREE.InstancedBufferGeometry()
  uvValueGeometry.index = base.index
  uvValueGeometry.setAttribute('position', base.getAttribute('position'))
  uvValueGeometry.instanceCount = 1
  uvValueGeometry.setAttribute('uv', new THREE.InstancedBufferAttribute(new Float32Array([Number.NaN, 0.5]), 2))
  const uvValueScene = new THREE.Scene()
  uvValueScene.add(new THREE.Mesh(uvValueGeometry, material))
  assert.throws(
    () => renderRgba(uvValueScene, camera, { width: 64, height: 64 }),
    /geometry\.attributes\.uv\[0\]\.x must be a finite number/i,
  )

  const uvRepeatGeometry = new THREE.InstancedBufferGeometry()
  uvRepeatGeometry.index = base.index
  uvRepeatGeometry.setAttribute('position', base.getAttribute('position'))
  const uvRepeat = new THREE.InstancedBufferAttribute(new Float32Array([0.5, 0.5]), 2)
  uvRepeat.meshPerAttribute = 0
  uvRepeatGeometry.setAttribute('uv', uvRepeat)
  const uvRepeatScene = new THREE.Scene()
  uvRepeatScene.add(new THREE.Mesh(uvRepeatGeometry, material))
  assert.throws(
    () => renderRgba(uvRepeatScene, camera, { width: 64, height: 64 }),
    /geometry\.attributes\.uv\.meshPerAttribute must be a positive finite number/i,
  )
})

test('invalid morph target influence values fail clearly', () => {
  function sceneWithInfluence(influence) {
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
      -0.75, -0.5, 0,
      0.75, -0.5, 0,
      0, 0.75, 0,
    ]), 3))
    geometry.setAttribute('normal', new THREE.BufferAttribute(new Float32Array([
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
    ]), 3))
    geometry.morphTargetsRelative = true
    geometry.morphAttributes.position = [new THREE.BufferAttribute(new Float32Array([
      0, 0.25, 0,
      0, 0.25, 0,
      0, 0.25, 0,
    ]), 3)]

    const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: 0xffffff }))
    mesh.morphTargetInfluences = [influence]

    const scene = new THREE.Scene()
    scene.add(mesh)
    return scene
  }

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  for (const influence of [Number.NaN, Number.POSITIVE_INFINITY, 'active']) {
    assert.throws(
      () => renderRgba(sceneWithInfluence(influence), camera, { width: 64, height: 64 }),
      /morphTargetInfluences\[0\] must be a finite number/i,
    )
  }

  const invalidRelative = sceneWithInfluence(1)
  invalidRelative.children[0].geometry.morphTargetsRelative = 'yes'
  assert.throws(
    () => renderRgba(invalidRelative, camera, { width: 64, height: 64 }),
    /geometry\.morphTargetsRelative must be a boolean/i,
  )

  const invalidAttributes = sceneWithInfluence(1)
  invalidAttributes.children[0].geometry.morphAttributes = 'morphs'
  assert.throws(
    () => renderRgba(invalidAttributes, camera, { width: 64, height: 64 }),
    /geometry\.morphAttributes must be an object/i,
  )

  const invalidPositionArray = sceneWithInfluence(1)
  invalidPositionArray.children[0].geometry.morphAttributes.position = 'positions'
  assert.throws(
    () => renderRgba(invalidPositionArray, camera, { width: 64, height: 64 }),
    /geometry\.morphAttributes\.position must be an array/i,
  )

  const invalidPositionEntry = sceneWithInfluence(1)
  invalidPositionEntry.children[0].geometry.morphAttributes.position = ['position']
  assert.throws(
    () => renderRgba(invalidPositionEntry, camera, { width: 64, height: 64 }),
    /geometry\.morphAttributes\.position\[0\] must be an attribute-like object/i,
  )
})

test('invalid skinning matrix values fail clearly', () => {
  function sceneWithSkinning(mutator) {
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
      -0.75, -0.5, 0,
      0.75, -0.5, 0,
      0, 0.75, 0,
    ]), 3))
    geometry.setAttribute('normal', new THREE.BufferAttribute(new Float32Array([
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
    ]), 3))
    geometry.setAttribute('skinIndex', new THREE.BufferAttribute(new Uint16Array([
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
    ]), 4))
    geometry.setAttribute('skinWeight', new THREE.BufferAttribute(new Float32Array([
      1, 0, 0, 0,
      1, 0, 0, 0,
      1, 0, 0, 0,
    ]), 4))

    const mesh = new THREE.SkinnedMesh(geometry, new THREE.MeshBasicMaterial({ color: 0xffffff }))
    const bone = new THREE.Bone()
    const skeleton = new THREE.Skeleton([bone])
    mesh.add(bone)
    const scene = new THREE.Scene()
    scene.add(mesh)
    mesh.bind(skeleton)
    mutator({ mesh, bone, skeleton })
    scene.updateMatrixWorld = () => {}
    return scene
  }

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const cases = [
    ['skeleton container', ({ mesh }) => {
      mesh.skeleton = 'skeleton'
    }, /mesh\.skeleton must be an object/i],
    ['bones container', ({ skeleton }) => {
      skeleton.bones = 'bones'
    }, /skeleton\.bones must be an array/i],
    ['bone inverse container', ({ skeleton }) => {
      skeleton.boneInverses = 'inverses'
    }, /skeleton\.boneInverses must be an array/i],
    ['bone entry', ({ skeleton }) => {
      skeleton.bones[0] = 'bone'
    }, /skeleton\.bones\[0\] must be an object/i],
    ['bone world matrix', ({ bone }) => {
      bone.matrixWorld.elements[13] = Number.NaN
    }, /skeleton\.bones\[0\]\.matrixWorld\.elements\[13\] must be a finite number/i],
    ['bone inverse matrix', ({ skeleton }) => {
      skeleton.boneInverses[0].elements[0] = Number.NaN
    }, /skeleton\.boneInverses\[0\]\.elements\[0\] must be a finite number/i],
    ['bind matrix', ({ mesh }) => {
      mesh.bindMatrix.elements[5] = Number.POSITIVE_INFINITY
    }, /mesh\.bindMatrix\.elements\[5\] must be a finite number/i],
    ['bind inverse matrix', ({ mesh }) => {
      mesh.bindMatrixInverse.elements[10] = Number.NEGATIVE_INFINITY
    }, /mesh\.bindMatrixInverse\.elements\[10\] must be a finite number/i],
  ]

  for (const [name, mutate, pattern] of cases) {
    assert.throws(
      () => renderRgba(sceneWithSkinning(mutate), camera, { width: 64, height: 64 }),
      pattern,
      `${name} should fail clearly`,
    )
  }
})

test('aoMap samples the selected UV channel', () => {
  function renderWithChannel(channel) {
    const aoMap = rgbaTexture([
      255, 255, 255, 255,
      255, 255, 255, 255,
      0, 0, 0, 255,
      0, 0, 0, 255,
    ], 4, 1)
    aoMap.channel = channel

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

    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const primary = renderWithChannel(0)
  const secondary = renderWithChannel(1)
  assert.ok(primary.r > secondary.r + 100, `aoMap channel=0 should sample bright primary UVs (${primary.r} vs ${secondary.r})`)
  assert.ok(secondary.r < 20, `aoMap channel=1 should darken the plane through uv1 (${secondary.r})`)
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

test('aoMap honors explicit texture matrices on the selected channel', () => {
  const aoMap = rgbaTexture([
    255, 255, 255, 255,
    0, 0, 0, 255,
  ], 2, 1)
  aoMap.channel = 1
  aoMap.magFilter = THREE.NearestFilter
  aoMap.minFilter = THREE.NearestFilter
  setTextureMatrixOffset(aoMap, 0.5)

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
  assert.ok(mean.r < 20, `explicit aoMap matrix should darken the plane through uv1 (${mean.r})`)
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

test('aoMap honors horizontal and vertical repeat wrapping', () => {
  function renderWithWrapping({ wrapS, wrapT, vertical = false }) {
    const aoMap = vertical
      ? rgbaTexture([
        255, 255, 255, 255,
        0, 0, 0, 255,
      ], 1, 2)
      : rgbaTexture([
        255, 255, 255, 255,
        0, 0, 0, 255,
      ], 2, 1)
    if (wrapS != null) aoMap.wrapS = wrapS
    if (wrapT != null) aoMap.wrapT = wrapT
    aoMap.magFilter = THREE.NearestFilter
    aoMap.minFilter = THREE.NearestFilter

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(vertical ? 0.5 : 1.25, vertical ? 1.25 : 0.5),
      new THREE.MeshBasicMaterial({ color: 0xffffff, aoMap, aoMapIntensity: 1 }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const clamped = renderWithWrapping({ wrapS: THREE.ClampToEdgeWrapping })
  const repeated = renderWithWrapping({ wrapS: THREE.RepeatWrapping })
  const mirrored = renderWithWrapping({ wrapS: THREE.MirroredRepeatWrapping })
  assert.ok(clamped.r < 20, `clamped aoMap U coordinates should sample the dark edge texel (${clamped.r})`)
  assert.ok(repeated.r > clamped.r + 80, `repeated aoMap U coordinates should wrap to the bright texel (${repeated.r} vs ${clamped.r})`)
  assert.ok(mirrored.r < 20, `mirrored aoMap U coordinates should reflect to the dark texel (${mirrored.r})`)
  assert.ok(repeated.r > mirrored.r + 80, `mirrored aoMap U coordinates should differ from RepeatWrapping (${mirrored.r} vs ${repeated.r})`)

  const clampedVertical = renderWithWrapping({ wrapT: THREE.ClampToEdgeWrapping, vertical: true })
  const repeatedVertical = renderWithWrapping({ wrapT: THREE.RepeatWrapping, vertical: true })
  const mirroredVertical = renderWithWrapping({ wrapT: THREE.MirroredRepeatWrapping, vertical: true })
  assert.ok(clampedVertical.r < 20, `clamped aoMap V coordinates should sample the dark edge texel (${clampedVertical.r})`)
  assert.ok(repeatedVertical.r > clampedVertical.r + 80, `repeated aoMap V coordinates should wrap to the bright texel (${repeatedVertical.r} vs ${clampedVertical.r})`)
  assert.ok(mirroredVertical.r < 20, `mirrored aoMap V coordinates should reflect to the dark texel (${mirroredVertical.r})`)
  assert.ok(repeatedVertical.r > mirroredVertical.r + 80, `mirrored aoMap V coordinates should differ from RepeatWrapping (${mirroredVertical.r} vs ${repeatedVertical.r})`)
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

test('alphaMap honors explicit texture matrices before alpha testing', () => {
  const alphaMap = rgbaTexture([
    255, 0, 0, 255,
    255, 255, 0, 255,
  ], 2, 1)
  alphaMap.magFilter = THREE.NearestFilter
  alphaMap.minFilter = THREE.NearestFilter
  setTextureMatrixOffset(alphaMap, 0.5)

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
  assert.ok(mean.r > mean.b + 40, `explicit alphaMap matrix should sample the visible texel before alpha testing (${mean.r} vs ${mean.b})`)
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

test('alphaMap honors horizontal and vertical repeat wrapping before alpha testing', () => {
  function renderWithWrapping({ wrapS, wrapT, vertical = false }) {
    const alphaMap = vertical
      ? rgbaTexture([
        255, 255, 255, 255,
        255, 0, 255, 255,
      ], 1, 2)
      : rgbaTexture([
        255, 255, 255, 255,
        255, 0, 255, 255,
      ], 2, 1)
    if (wrapS != null) alphaMap.wrapS = wrapS
    if (wrapT != null) alphaMap.wrapT = wrapT
    alphaMap.magFilter = THREE.NearestFilter
    alphaMap.minFilter = THREE.NearestFilter

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 1)
    scene.add(new THREE.Mesh(
      constantUvPlane(vertical ? 0.5 : 1.25, vertical ? 1.25 : 0.5),
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

  const clamped = renderWithWrapping({ wrapS: THREE.ClampToEdgeWrapping })
  const repeated = renderWithWrapping({ wrapS: THREE.RepeatWrapping })
  const mirrored = renderWithWrapping({ wrapS: THREE.MirroredRepeatWrapping })
  assert.ok(clamped.b > clamped.r + 80, `clamped alphaMap U coordinates should sample the transparent edge texel (${clamped.b} vs ${clamped.r})`)
  assert.ok(repeated.r > repeated.b + 40, `repeated alphaMap U coordinates should wrap to the opaque texel (${repeated.r} vs ${repeated.b})`)
  assert.ok(mirrored.b > mirrored.r + 80, `mirrored alphaMap U coordinates should reflect to the transparent texel (${mirrored.b} vs ${mirrored.r})`)

  const clampedVertical = renderWithWrapping({ wrapT: THREE.ClampToEdgeWrapping, vertical: true })
  const repeatedVertical = renderWithWrapping({ wrapT: THREE.RepeatWrapping, vertical: true })
  const mirroredVertical = renderWithWrapping({ wrapT: THREE.MirroredRepeatWrapping, vertical: true })
  assert.ok(clampedVertical.b > clampedVertical.r + 80, `clamped alphaMap V coordinates should sample the transparent edge texel (${clampedVertical.b} vs ${clampedVertical.r})`)
  assert.ok(repeatedVertical.r > repeatedVertical.b + 40, `repeated alphaMap V coordinates should wrap to the opaque texel (${repeatedVertical.r} vs ${repeatedVertical.b})`)
  assert.ok(mirroredVertical.b > mirroredVertical.r + 80, `mirrored alphaMap V coordinates should reflect to the transparent texel (${mirroredVertical.b} vs ${mirroredVertical.r})`)
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

test('material alphaToCoverage uses MSAA coverage from output alpha', () => {
  function renderCoverage(alphaToCoverage, sampleCount = 4) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({
        color: 0xffffff,
        opacity: 0.5,
        transparent: false,
        alphaToCoverage,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, {
      width: 64,
      height: 64,
      sampleCount,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 64, 64, 24, 24, 40, 40)
  }

  const noCoverage = renderCoverage(false)
  const coverage = renderCoverage(true)
  const singleSample = renderCoverage(true, 1)
  assert.ok(noCoverage.r > 170, `opaque non-A2C path should keep bright RGB despite opacity alpha (${noCoverage.r})`)
  assert.ok(Math.abs(singleSample.r - noCoverage.r) < 5, `single-sample alphaToCoverage should not alter RGB coverage (${singleSample.r} vs ${noCoverage.r})`)
  assert.ok(coverage.r > 30 && coverage.r < noCoverage.r - 80, `4x alphaToCoverage should resolve partial RGB coverage (${coverage.r} vs ${noCoverage.r})`)
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

test('clippingPlanes over the native plane budget fail clearly', () => {
  function planes(count) {
    return Array.from({ length: count }, (_, index) => new THREE.Plane(new THREE.Vector3(1, 0, 0), -index - 1))
  }

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const cases = [
    ['options', () => {
      const scene = new THREE.Scene()
      scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial()))
      return () => renderRgba(scene, camera, { width: 64, height: 64, clippingPlanes: planes(9) })
    }, /options\.clippingPlanes.*at most 8 active/i],
    ['group', () => {
      const scene = new THREE.Scene()
      const group = new THREE.Group()
      group.isClippingGroup = true
      group.clippingPlanes = planes(9)
      group.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial()))
      scene.add(group)
      return () => renderRgba(scene, camera, { width: 64, height: 64 })
    }, /ClippingGroup\.clippingPlanes.*at most 8 active/i],
    ['material', () => {
      const scene = new THREE.Scene()
      const material = new THREE.MeshBasicMaterial()
      material.clippingPlanes = [new THREE.Plane(new THREE.Vector3(0, 1, 0), 0)]
      scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))
      return () => renderRgba(scene, camera, { width: 64, height: 64, clippingPlanes: planes(8) })
    }, /material\.clippingPlanes.*at most 8 active/i],
  ]

  for (const [label, makeRender, pattern] of cases) {
    assert.throws(makeRender(), pattern, label)
  }

  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial()))
  assert.throws(
    () => renderRgba(scene, camera, { width: 64, height: 64, localClippingEnabled: 'no' }),
    /options\.localClippingEnabled must be a boolean/i,
  )
})

test('invalid clipping control boolean values fail clearly', () => {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  function renderMaterialWith(mutator) {
    const scene = new THREE.Scene()
    const material = new THREE.MeshBasicMaterial({ color: 0xffffff })
    material.clippingPlanes = [new THREE.Plane(new THREE.Vector3(1, 0, 0), 0)]
    mutator(material)
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))
    return () => renderRgba(scene, camera, { width: 64, height: 64 })
  }

  function renderGroupWith(mutator) {
    const scene = new THREE.Scene()
    const group = new THREE.Group()
    group.isClippingGroup = true
    group.clippingPlanes = [new THREE.Plane(new THREE.Vector3(1, 0, 0), 0)]
    mutator(group)
    group.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial()))
    scene.add(group)
    return () => renderRgba(scene, camera, { width: 64, height: 64 })
  }

  const cases = [
    ['material clipIntersection', renderMaterialWith((material) => {
      material.clipIntersection = 'yes'
    }), /material\.clipIntersection must be a boolean/i],
    ['material clipShadows', renderMaterialWith((material) => {
      material.clipShadows = 'yes'
    }), /material\.clipShadows must be a boolean/i],
    ['group enabled', renderGroupWith((group) => {
      group.enabled = 'yes'
    }), /ClippingGroup\.enabled must be a boolean/i],
    ['group clipIntersection', renderGroupWith((group) => {
      group.clipIntersection = 'yes'
    }), /ClippingGroup\.clipIntersection must be a boolean/i],
    ['group clipShadows', renderGroupWith((group) => {
      group.clipShadows = 'yes'
    }), /ClippingGroup\.clipShadows must be a boolean/i],
  ]

  for (const [label, render, pattern] of cases) {
    assert.throws(render, pattern, label)
  }
})

test('invalid clippingPlane values fail clearly', () => {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const cases = [
    ['options container', () => {
      const scene = new THREE.Scene()
      scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial()))
      return () => renderRgba(scene, camera, { width: 64, height: 64, clippingPlanes: 'planes' })
    }, /options\.clippingPlanes must be an array of clipping planes/i],
    ['options constant', () => {
      const scene = new THREE.Scene()
      scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial()))
      const plane = new THREE.Plane(new THREE.Vector3(1, 0, 0), 0)
      plane.constant = Number.NaN
      return () => renderRgba(scene, camera, { width: 64, height: 64, clippingPlanes: [plane] })
    }, /options\.clippingPlanes\[0\]\.constant must be a finite number/i],
    ['material normal', () => {
      const scene = new THREE.Scene()
      const material = new THREE.MeshBasicMaterial()
      const plane = new THREE.Plane(new THREE.Vector3(1, 0, 0), 0)
      plane.normal.x = 'right'
      material.clippingPlanes = [plane]
      scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))
      return () => renderRgba(scene, camera, { width: 64, height: 64 })
    }, /material\.clippingPlanes\[0\]\.normal\.x must be a finite number/i],
    ['material container', () => {
      const scene = new THREE.Scene()
      const material = new THREE.MeshBasicMaterial()
      material.clippingPlanes = 'planes'
      scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))
      return () => renderRgba(scene, camera, { width: 64, height: 64 })
    }, /material\.clippingPlanes must be an array of clipping planes/i],
    ['group container', () => {
      const scene = new THREE.Scene()
      const group = new THREE.Group()
      group.isClippingGroup = true
      group.clippingPlanes = 'planes'
      group.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial()))
      scene.add(group)
      return () => renderRgba(scene, camera, { width: 64, height: 64 })
    }, /ClippingGroup\.clippingPlanes must be an array of clipping planes/i],
    ['group zero normal', () => {
      const scene = new THREE.Scene()
      const group = new THREE.Group()
      group.isClippingGroup = true
      group.clippingPlanes = [[0, 0, 0, 0]]
      group.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial()))
      scene.add(group)
      return () => renderRgba(scene, camera, { width: 64, height: 64 })
    }, /ClippingGroup\.clippingPlanes\[0\]\.normal must have non-zero finite length/i],
  ]

  for (const [label, makeRender, pattern] of cases) {
    assert.throws(makeRender(), pattern, label)
  }
})

test('localClippingEnabled false ignores material planes but preserves global planes', () => {
  const material = new THREE.MeshBasicMaterial({ color: 0xff0000 })
  material.clippingPlanes = [new THREE.Plane(new THREE.Vector3(1, 0, 0), 0)]

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 1)
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, {
    width: 64,
    height: 64,
    localClippingEnabled: false,
    clippingPlanes: [new THREE.Plane(new THREE.Vector3(0, 1, 0), 0)],
  })
  const topLeft = meanRegion(rgba, 64, 64, 12, 12, 24, 24)
  const bottomRight = meanRegion(rgba, 64, 64, 40, 40, 52, 52)

  assert.ok(topLeft.r > topLeft.b + 80, `top-left should ignore the material x-plane (${topLeft.r} vs ${topLeft.b})`)
  assert.ok(bottomRight.b > bottomRight.r + 80, `bottom-right should still be clipped by the global y-plane (${bottomRight.b} vs ${bottomRight.r})`)
})

test('localClippingEnabled false ignores line material planes but preserves global planes', () => {
  const material = new THREE.LineBasicMaterial({ color: 0xff0000, linewidth: 8 })
  material.clippingPlanes = [new THREE.Plane(new THREE.Vector3(1, 0, 0), 0)]

  const geometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-1.5, 0.5, 0),
    new THREE.Vector3(1.5, 0.5, 0),
    new THREE.Vector3(-1.5, -0.5, 0),
    new THREE.Vector3(1.5, -0.5, 0),
  ])
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 1)
  scene.add(new THREE.LineSegments(geometry, material))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, {
    width: 64,
    height: 64,
    localClippingEnabled: false,
    clippingPlanes: [new THREE.Plane(new THREE.Vector3(0, 1, 0), 0)],
  })
  const topLeftRedPixels = countRegionPixels(
    rgba,
    64,
    64,
    8,
    18,
    30,
    30,
    (r, g, b) => r > 120 && r > b + 40 && g < 60,
  )
  const bottomRight = meanRegion(rgba, 64, 64, 38, 38, 56, 50)

  assert.ok(topLeftRedPixels > 4, `top-left line should ignore the material x-plane (${topLeftRedPixels})`)
  assert.ok(bottomRight.b > bottomRight.r + 80, `bottom-right line should still be clipped by the global y-plane (${bottomRight.b} vs ${bottomRight.r})`)
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

test('line clipIntersection requires all clipping planes to reject a fragment', () => {
  function makeLine() {
    const geometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-1.5, 0.7, 0),
      new THREE.Vector3(-0.3, 0.7, 0),
      new THREE.Vector3(-1.5, -0.7, 0),
      new THREE.Vector3(-0.3, -0.7, 0),
      new THREE.Vector3(0.3, -0.7, 0),
      new THREE.Vector3(1.5, -0.7, 0),
    ])
    return new THREE.LineSegments(
      geometry,
      new THREE.LineBasicMaterial({ color: 0xff0000, linewidth: 8 }),
    )
  }

  function renderLine(source) {
    const planes = [
      new THREE.Plane(new THREE.Vector3(1, 0, 0), 0),
      new THREE.Plane(new THREE.Vector3(0, 1, 0), 0),
    ]
    const line = makeLine()
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 1)
    if (source === 'material') {
      line.material.clippingPlanes = planes
      line.material.clipIntersection = true
      scene.add(line)
    } else {
      const group = new THREE.Group()
      group.isClippingGroup = true
      group.clipIntersection = true
      group.clippingPlanes = planes
      group.add(line)
      scene.add(group)
    }

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return renderRgba(scene, camera, { width: 64, height: 64 })
  }

  for (const source of ['material', 'group']) {
    const rgba = renderLine(source)
    const visibleTopLeft = countRegionPixels(rgba, 64, 64, 8, 14, 30, 28, (r, g, b) => r > 120 && r > b + 40 && g < 60)
    const clippedBottomLeft = meanRegion(rgba, 64, 64, 8, 36, 30, 50)
    const visibleBottomRight = countRegionPixels(rgba, 64, 64, 34, 36, 56, 50, (r, g, b) => r > 120 && r > b + 40 && g < 60)

    assert.ok(visibleTopLeft > 4, `${source} top-left line should remain visible with intersection clipping (${visibleTopLeft})`)
    assert.ok(clippedBottomLeft.b > clippedBottomLeft.r + 80, `${source} bottom-left line should be clipped by both planes (${clippedBottomLeft.b} vs ${clippedBottomLeft.r})`)
    assert.ok(visibleBottomRight > 4, `${source} bottom-right line should remain visible with intersection clipping (${visibleBottomRight})`)
  }
})

test('scene ClippingGroup planes clip descendants', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 1)

  const group = new THREE.Group()
  group.name = 'clip-group'
  group.isClippingGroup = true
  group.clippingPlanes = [new THREE.Plane(new THREE.Vector3(1, 0, 0), 0)]
  group.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  ))
  scene.add(group)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 64, height: 64 })
  const clippedLeft = meanRegion(rgba, 64, 64, 12, 28, 24, 36)
  const visibleRight = meanRegion(rgba, 64, 64, 40, 28, 52, 36)
  assert.ok(clippedLeft.b > clippedLeft.r + 80, `left descendant pixels should be clipped by group plane (${clippedLeft.b} vs ${clippedLeft.r})`)
  assert.ok(visibleRight.r > visibleRight.b + 80, `right descendant pixels should remain visible (${visibleRight.r} vs ${visibleRight.b})`)
})

test('scene ClippingGroup clipIntersection requires all group planes', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 1)

  const group = new THREE.Group()
  group.isClippingGroup = true
  group.clipIntersection = true
  group.clippingPlanes = [
    new THREE.Plane(new THREE.Vector3(1, 0, 0), 0),
    new THREE.Plane(new THREE.Vector3(0, 1, 0), 0),
  ]
  group.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  ))
  scene.add(group)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 64, height: 64 })
  const visibleTopLeft = meanRegion(rgba, 64, 64, 12, 12, 24, 24)
  const clippedBottomLeft = meanRegion(rgba, 64, 64, 12, 40, 24, 52)
  const visibleBottomRight = meanRegion(rgba, 64, 64, 40, 40, 52, 52)
  assert.ok(visibleTopLeft.r > visibleTopLeft.b + 80, `top-left should remain visible with group intersection clipping (${visibleTopLeft.r} vs ${visibleTopLeft.b})`)
  assert.ok(clippedBottomLeft.b > clippedBottomLeft.r + 80, `bottom-left should be clipped by both group planes (${clippedBottomLeft.b} vs ${clippedBottomLeft.r})`)
  assert.ok(visibleBottomRight.r > visibleBottomRight.b + 80, `bottom-right should remain visible with group intersection clipping (${visibleBottomRight.r} vs ${visibleBottomRight.b})`)
})

test('nested ClippingGroup planes compose inherited union and child intersection rules', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 1)

  const parent = new THREE.Group()
  parent.isClippingGroup = true
  parent.clippingPlanes = [new THREE.Plane(new THREE.Vector3(1, 0, 0), 0)]

  const child = new THREE.Group()
  child.isClippingGroup = true
  child.clipIntersection = true
  child.clippingPlanes = [
    new THREE.Plane(new THREE.Vector3(0, 1, 0), 0),
    new THREE.Plane(new THREE.Vector3(-1, 0, 0), 0),
  ]

  child.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  ))
  parent.add(child)
  scene.add(parent)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 64, height: 64 })
  const clippedTopLeft = meanRegion(rgba, 64, 64, 12, 12, 24, 24)
  const visibleTopRight = meanRegion(rgba, 64, 64, 40, 12, 52, 24)
  const clippedBottomRight = meanRegion(rgba, 64, 64, 40, 40, 52, 52)

  assert.ok(clippedTopLeft.b > clippedTopLeft.r + 80, `top-left should inherit the parent union clip (${clippedTopLeft.b} vs ${clippedTopLeft.r})`)
  assert.ok(visibleTopRight.r > visibleTopRight.b + 80, `top-right should survive the child intersection group (${visibleTopRight.r} vs ${visibleTopRight.b})`)
  assert.ok(clippedBottomRight.b > clippedBottomRight.r + 80, `bottom-right should be clipped by both child intersection planes (${clippedBottomRight.b} vs ${clippedBottomRight.r})`)
})

test('scene ClippingGroup clipShadows clips descendant shadow casters', () => {
  function renderGroupClipShadows(clipShadows) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const group = new THREE.Group()
    group.isClippingGroup = true
    group.clipShadows = clipShadows
    group.clippingPlanes = [new THREE.Plane(new THREE.Vector3(0, 1, 0), -10)]

    const caster = new THREE.Mesh(
      new THREE.BoxGeometry(3, 3, 3),
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    )
    caster.position.y = 1.5
    caster.castShadow = true
    group.add(caster)
    scene.add(group)

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(8, 6, 0)
    light.target.position.set(0, 0, 0)
    light.castShadow = true
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

  const unclippedShadow = renderGroupClipShadows(false)
  const clippedShadow = renderGroupClipShadows(true)
  const unclippedLum = unclippedShadow.r + unclippedShadow.g + unclippedShadow.b
  const clippedLum = clippedShadow.r + clippedShadow.g + clippedShadow.b
  assert.ok(clippedLum > unclippedLum + 30, `group clipShadows should remove descendant caster shadow (${clippedLum} vs ${unclippedLum})`)
})

test('material clipShadows clips shadow caster fragments', () => {
  function renderClipShadows(clipShadows) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const material = new THREE.MeshBasicMaterial({ color: 0xffffff })
    material.clippingPlanes = [new THREE.Plane(new THREE.Vector3(0, 1, 0), -10)]
    material.clipShadows = clipShadows

    const caster = new THREE.Mesh(new THREE.BoxGeometry(3, 3, 3), material)
    caster.position.y = 1.5
    caster.castShadow = true
    scene.add(caster)

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(8, 6, 0)
    light.target.position.set(0, 0, 0)
    light.castShadow = true
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

  const unclippedShadow = renderClipShadows(false)
  const clippedShadow = renderClipShadows(true)
  const unclippedLum = unclippedShadow.r + unclippedShadow.g + unclippedShadow.b
  const clippedLum = clippedShadow.r + clippedShadow.g + clippedShadow.b
  assert.ok(clippedLum > unclippedLum + 30, `clipShadows should remove the clipped caster shadow (${clippedLum} vs ${unclippedLum})`)
})

test('source material clipShadows applies to custom shadow material casters', () => {
  function sourceMaterial(clipShadows) {
    const material = new THREE.MeshBasicMaterial({ color: 0xffffff })
    material.clippingPlanes = [new THREE.Plane(new THREE.Vector3(0, 1, 0), -10)]
    material.clipShadows = clipShadows
    material.colorWrite = false
    material.depthWrite = false
    return material
  }

  function renderDirectionalCustomDepthClipShadows(clipShadows) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const caster = new THREE.Mesh(new THREE.BoxGeometry(3, 3, 3), sourceMaterial(clipShadows))
    caster.position.y = 1.5
    caster.castShadow = true
    caster.customDepthMaterial = new THREE.MeshDepthMaterial()
    scene.add(caster)

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(8, 6, 0)
    light.target.position.set(0, 0, 0)
    light.castShadow = true
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

  function renderPointCustomDistanceClipShadows(clipShadows) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const caster = new THREE.Mesh(new THREE.BoxGeometry(2.5, 2.5, 2.5), sourceMaterial(clipShadows))
    caster.position.y = 1.25
    caster.castShadow = true
    caster.customDistanceMaterial = new THREE.MeshDistanceMaterial()
    scene.add(caster)

    const light = new THREE.PointLight(0xffffff, 2)
    light.position.set(0, 5, 4)
    light.distance = 12
    light.castShadow = true
    light.shadow.mapSize.set(256, 256)
    light.shadow.camera.near = 0.1
    light.shadow.camera.far = 12
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 96, height: 96 }))
  }

  const unclippedDepth = renderDirectionalCustomDepthClipShadows(false)
  const clippedDepth = renderDirectionalCustomDepthClipShadows(true)
  const unclippedDepthLum = unclippedDepth.r + unclippedDepth.g + unclippedDepth.b
  const clippedDepthLum = clippedDepth.r + clippedDepth.g + clippedDepth.b
  assert.ok(clippedDepthLum > unclippedDepthLum + 30, `source clipShadows should remove the customDepthMaterial caster shadow (${clippedDepthLum} vs ${unclippedDepthLum})`)

  const unclippedDistance = renderPointCustomDistanceClipShadows(false)
  const clippedDistance = renderPointCustomDistanceClipShadows(true)
  const unclippedDistanceLum = unclippedDistance.r + unclippedDistance.g + unclippedDistance.b
  const clippedDistanceLum = clippedDistance.r + clippedDistance.g + clippedDistance.b
  assert.ok(clippedDistanceLum > unclippedDistanceLum + 20, `source clipShadows should remove the customDistanceMaterial caster shadow (${clippedDistanceLum} vs ${unclippedDistanceLum})`)
})

test('Line, LineSegments, and LineLoop objects cast directional shadows', () => {
  function makeLineObject(kind, castShadow) {
    const material = new THREE.LineBasicMaterial({ color: 0xffffff })
    material.colorWrite = false
    material.depthWrite = false

    if (kind === 'segments') {
      const positions = []
      for (let offset = -3; offset <= 3; offset += 0.5) {
        positions.push(-3, 2, offset, 3, 2, offset)
        positions.push(offset, 2, -3, offset, 2, 3)
      }
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
      const line = new THREE.LineSegments(geometry, material)
      line.castShadow = castShadow
      return line
    }

    if (kind === 'loop') {
      const positions = []
      for (let row = 0; row < 18; row += 1) {
        const z = -3 + row * (6 / 17)
        if (row % 2 === 0) {
          positions.push(-3, 2, z, 3, 2, z)
        } else {
          positions.push(3, 2, z, -3, 2, z)
        }
      }
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
      const line = new THREE.LineLoop(geometry, material)
      line.castShadow = castShadow
      return line
    }

    const positions = []
    for (let row = 0; row < 12; row += 1) {
      const z = -3 + row * (6 / 11)
      if (row % 2 === 0) {
        positions.push(-3, 2, z, 3, 2, z)
      } else {
        positions.push(3, 2, z, -3, 2, z)
      }
    }
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
    const line = new THREE.Line(geometry, material)
    line.castShadow = castShadow
    return line
  }

  function renderLineShadow(kind, castShadow) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)
    scene.add(makeLineObject(kind, castShadow))

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(3, 6, 4)
    light.target.position.set(0, 0, 0)
    light.castShadow = true
    light.shadow.mapSize.set(1024, 1024)
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

  for (const kind of ['line', 'segments', 'loop']) {
    const unshadowed = renderLineShadow(kind, false)
    const shadowed = renderLineShadow(kind, true)
    const unshadowedLum = unshadowed.r + unshadowed.g + unshadowed.b
    const shadowedLum = shadowed.r + shadowed.g + shadowed.b
    assert.ok(shadowedLum < unshadowedLum - 2, `${kind} should cast visible line shadows (${shadowedLum} vs ${unshadowedLum})`)
  }
})

test('LineSegments objects cast spot and point shadows', () => {
  function makeLineSegments(castShadow) {
    const material = new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 6 })
    material.colorWrite = false
    material.depthWrite = false

    const positions = []
    for (let offset = -3; offset <= 3; offset += 0.35) {
      positions.push(-3, 2, offset, 3, 2, offset)
      positions.push(offset, 2, -3, offset, 2, 3)
    }
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
    const line = new THREE.LineSegments(geometry, material)
    line.castShadow = castShadow
    return line
  }

  function renderLineSegmentsShadow(lightKind, castShadow) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)
    scene.add(makeLineSegments(castShadow))

    if (lightKind === 'spot') {
      const light = new THREE.SpotLight(0xffffff, 4, 20, Math.PI / 4, 0.2, 2)
      light.position.set(0, 7, 8)
      light.target.position.set(0, 0, 0)
      light.castShadow = true
      light.shadow.mapSize.set(512, 512)
      light.shadow.camera.near = 0.1
      light.shadow.camera.far = 20
      scene.add(light)
      scene.add(light.target)
    } else {
      const light = new THREE.PointLight(0xffffff, 2)
      light.position.set(0, 5, 4)
      light.distance = 12
      light.castShadow = true
      light.shadow.mapSize.set(256, 256)
      light.shadow.camera.near = 0.1
      light.shadow.camera.far = 12
      scene.add(light)
    }

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 96, height: 96 }))
  }

  for (const lightKind of ['spot', 'point']) {
    const unshadowed = renderLineSegmentsShadow(lightKind, false)
    const shadowed = renderLineSegmentsShadow(lightKind, true)
    const unshadowedLum = unshadowed.r + unshadowed.g + unshadowed.b
    const shadowedLum = shadowed.r + shadowed.g + shadowed.b
    assert.ok(shadowedLum < unshadowedLum - 4, `${lightKind} line shadow should darken the receiver (${shadowedLum} vs ${unshadowedLum})`)
  }
})

test('LineBasicMaterial and LineDashedMaterial cutouts affect directional shadows', () => {
  function makeLineSegments(materialKind, cutoutKind, opaque) {
    const material = materialKind === 'dashed'
      ? new THREE.LineDashedMaterial({
        color: 0xffffff,
        dashSize: 10,
        gapSize: 0,
        scale: 1,
      })
      : new THREE.LineBasicMaterial({ color: 0xffffff })
    material.colorWrite = false
    material.depthWrite = false
    material.alphaTest = 0.5
    if (cutoutKind === 'map') {
      material.map = solidTexture(255, 255, 255, opaque ? 255 : 0)
    } else {
      material.alphaMap = solidTexture(255, opaque ? 255 : 0, 255)
    }

    const positions = []
    for (let offset = -3; offset <= 3; offset += 0.25) {
      positions.push(-3, 2, offset, 3, 2, offset)
      positions.push(offset, 2, -3, offset, 2, 3)
    }
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
    const line = new THREE.LineSegments(geometry, material)
    if (materialKind === 'dashed') line.computeLineDistances()
    line.castShadow = true
    return line
  }

  function renderLineCutoutShadow(materialKind, cutoutKind, opaque) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)
    scene.add(makeLineSegments(materialKind, cutoutKind, opaque))

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(3, 6, 4)
    light.target.position.set(0, 0, 0)
    light.castShadow = true
    light.shadow.mapSize.set(1024, 1024)
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

  for (const materialKind of ['basic', 'dashed']) {
    for (const cutoutKind of ['map', 'alphaMap']) {
      const opaqueCaster = renderLineCutoutShadow(materialKind, cutoutKind, true)
      const cutoutCaster = renderLineCutoutShadow(materialKind, cutoutKind, false)
      const opaqueLum = opaqueCaster.r + opaqueCaster.g + opaqueCaster.b
      const cutoutLum = cutoutCaster.r + cutoutCaster.g + cutoutCaster.b
      assert.ok(cutoutLum > opaqueLum + 2, `${materialKind} ${cutoutKind} should remove the line caster shadow (${cutoutLum} vs ${opaqueLum})`)
    }
  }
})

test('LineBasicMaterial alphaToCoverage approximates shadow caster cutouts', () => {
  function renderLineAlphaCoverageShadow(alphaToCoverage) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const material = new THREE.LineBasicMaterial({ color: 0xffffff })
    material.alphaMap = solidTexture(255, 0, 255)
    material.alphaToCoverage = alphaToCoverage
    material.colorWrite = false
    material.depthWrite = false

    const positions = []
    for (let offset = -3; offset <= 3; offset += 0.25) {
      positions.push(-3, 2, offset, 3, 2, offset)
      positions.push(offset, 2, -3, offset, 2, 3)
    }
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
    const line = new THREE.LineSegments(geometry, material)
    line.castShadow = true
    scene.add(line)

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(3, 6, 4)
    light.target.position.set(0, 0, 0)
    light.castShadow = true
    light.shadow.mapSize.set(1024, 1024)
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

  const fullShadow = renderLineAlphaCoverageShadow(false)
  const cutoutShadow = renderLineAlphaCoverageShadow(true)
  const fullLum = fullShadow.r + fullShadow.g + fullShadow.b
  const cutoutLum = cutoutShadow.r + cutoutShadow.g + cutoutShadow.b
  assert.ok(cutoutLum > fullLum + 2, `line alphaToCoverage shadow cutoff should let more receiver light through (${cutoutLum} vs ${fullLum})`)
})

test('LineBasicMaterial and LineDashedMaterial alphaHash approximate shadow caster opacity cutouts', () => {
  function renderLineAlphaHashShadow(materialKind, alphaHash) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const material = materialKind === 'dashed'
      ? new THREE.LineDashedMaterial({
        color: 0xffffff,
        opacity: alphaHash ? 0.35 : 1,
        alphaHash,
        dashSize: 10,
        gapSize: 0,
        scale: 1,
      })
      : new THREE.LineBasicMaterial({
        color: 0xffffff,
        opacity: alphaHash ? 0.35 : 1,
        alphaHash,
      })
    material.colorWrite = false
    material.depthWrite = false

    const positions = []
    for (let offset = -3; offset <= 3; offset += 0.25) {
      positions.push(-3, 2, offset, 3, 2, offset)
      positions.push(offset, 2, -3, offset, 2, 3)
    }
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
    const line = new THREE.LineSegments(geometry, material)
    if (materialKind === 'dashed') line.computeLineDistances()
    line.castShadow = true
    scene.add(line)

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(3, 6, 4)
    light.target.position.set(0, 0, 0)
    light.castShadow = true
    light.shadow.mapSize.set(1024, 1024)
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

  for (const materialKind of ['basic', 'dashed']) {
    const fullShadow = renderLineAlphaHashShadow(materialKind, false)
    const hashedShadow = renderLineAlphaHashShadow(materialKind, true)
    const fullLum = fullShadow.r + fullShadow.g + fullShadow.b
    const hashedLum = hashedShadow.r + hashedShadow.g + hashedShadow.b
    assert.ok(hashedLum > fullLum + 2, `${materialKind} line alphaHash shadow cutoff should let more receiver light through (${hashedLum} vs ${fullLum})`)
  }
})

test('Line shadow casters honor material and group clipShadows', () => {
  function makeLineSegments() {
    const material = new THREE.LineBasicMaterial({ color: 0xffffff })
    material.colorWrite = false
    material.depthWrite = false

    const positions = []
    for (let offset = -3; offset <= 3; offset += 0.25) {
      positions.push(-3, 2, offset, 3, 2, offset)
      positions.push(offset, 2, -3, offset, 2, 3)
    }
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
    const line = new THREE.LineSegments(geometry, material)
    line.castShadow = true
    return line
  }

  function renderLineClipShadows(source, clipShadows) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const line = makeLineSegments()
    const clippingPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -10)
    if (source === 'material') {
      line.material.clippingPlanes = [clippingPlane]
      line.material.clipShadows = clipShadows
      scene.add(line)
    } else {
      const group = new THREE.Group()
      group.isClippingGroup = true
      group.clippingPlanes = [clippingPlane]
      group.clipShadows = clipShadows
      group.add(line)
      scene.add(group)
    }

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(3, 6, 4)
    light.target.position.set(0, 0, 0)
    light.castShadow = true
    light.shadow.mapSize.set(1024, 1024)
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

  for (const source of ['material', 'group']) {
    const unclipped = renderLineClipShadows(source, false)
    const clipped = renderLineClipShadows(source, true)
    const unclippedLum = unclipped.r + unclipped.g + unclipped.b
    const clippedLum = clipped.r + clipped.g + clipped.b
    assert.ok(clippedLum > unclippedLum + 2, `${source} clipShadows should remove the line caster shadow (${clippedLum} vs ${unclippedLum})`)
  }
})

test('alpha-tested shadow casters honor alphaMap cutouts', () => {
  function renderAlphaShadow(alphaMapGreen) {
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
      new THREE.MeshBasicMaterial({
        color: 0xffffff,
        alphaMap: solidTexture(255, alphaMapGreen, 255),
        alphaTest: 0.5,
      }),
    )
    caster.position.y = 1.5
    caster.castShadow = true
    scene.add(caster)

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(8, 6, 0)
    light.target.position.set(0, 0, 0)
    light.castShadow = true
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

  const opaqueCaster = renderAlphaShadow(255)
  const cutoutCaster = renderAlphaShadow(0)
  const opaqueLum = opaqueCaster.r + opaqueCaster.g + opaqueCaster.b
  const cutoutLum = cutoutCaster.r + cutoutCaster.g + cutoutCaster.b
  assert.ok(cutoutLum > opaqueLum + 30, `alphaMap cutout should remove the caster shadow (${cutoutLum} vs ${opaqueLum})`)
})

test('alpha-tested shadow casters honor base-map alpha cutouts', () => {
  function renderMapAlphaShadow(mapAlpha) {
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
      new THREE.MeshBasicMaterial({
        color: 0xffffff,
        map: solidTexture(255, 255, 255, mapAlpha),
        alphaTest: 0.5,
      }),
    )
    caster.position.y = 1.5
    caster.castShadow = true
    scene.add(caster)

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(8, 6, 0)
    light.target.position.set(0, 0, 0)
    light.castShadow = true
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

  const opaqueCaster = renderMapAlphaShadow(255)
  const cutoutCaster = renderMapAlphaShadow(0)
  const opaqueLum = opaqueCaster.r + opaqueCaster.g + opaqueCaster.b
  const cutoutLum = cutoutCaster.r + cutoutCaster.g + cutoutCaster.b
  assert.ok(cutoutLum > opaqueLum + 30, `base map alpha cutout should remove the caster shadow (${cutoutLum} vs ${opaqueLum})`)
})

test('SpriteMaterial casts directional shadows from expanded billboards', () => {
  function renderSpriteShadow(castShadow) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ color: 0xffffff }))
    sprite.position.set(0, 4, 0)
    sprite.scale.set(4, 4, 1)
    sprite.castShadow = castShadow
    scene.add(sprite)

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(0, 6, 8)
    light.target.position.set(0, 0, 0)
    light.castShadow = true
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

  const unshadowed = renderSpriteShadow(false)
  const shadowed = renderSpriteShadow(true)
  const unshadowedLum = unshadowed.r + unshadowed.g + unshadowed.b
  const shadowedLum = shadowed.r + shadowed.g + shadowed.b
  assert.ok(shadowedLum < unshadowedLum - 15, `sprite billboard shadow should darken the receiver (${shadowedLum} vs ${unshadowedLum})`)
})

test('PointsMaterial casts directional shadows from expanded billboards', () => {
  function renderPointShadow(castShadow) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 4, 0]), 3))
    const points = new THREE.Points(geometry, new THREE.PointsMaterial({
      color: 0xffffff,
      size: 48,
      sizeAttenuation: false,
    }))
    points.castShadow = castShadow
    scene.add(points)

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(0, 6, 8)
    light.target.position.set(0, 0, 0)
    light.castShadow = true
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

  const unshadowed = renderPointShadow(false)
  const shadowed = renderPointShadow(true)
  const unshadowedLum = unshadowed.r + unshadowed.g + unshadowed.b
  const shadowedLum = shadowed.r + shadowed.g + shadowed.b
  assert.ok(shadowedLum < unshadowedLum - 15, `point billboard shadow should darken the receiver (${shadowedLum} vs ${unshadowedLum})`)
})

test('SpriteMaterial and PointsMaterial alphaMap cutouts affect directional shadows', () => {
  function renderBillboardAlphaShadow(kind, alphaMapGreen) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    if (kind === 'sprite') {
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
        color: 0xffffff,
        alphaMap: solidTexture(255, alphaMapGreen, 255),
        alphaTest: 0.5,
      }))
      sprite.position.set(0, 4, 0)
      sprite.scale.set(4, 4, 1)
      sprite.castShadow = true
      scene.add(sprite)
    } else {
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 4, 0]), 3))
      const points = new THREE.Points(geometry, new THREE.PointsMaterial({
        color: 0xffffff,
        size: 48,
        sizeAttenuation: false,
        alphaMap: solidTexture(255, alphaMapGreen, 255),
        alphaTest: 0.5,
      }))
      points.castShadow = true
      scene.add(points)
    }

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(0, 6, 8)
    light.target.position.set(0, 0, 0)
    light.castShadow = true
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

  for (const kind of ['sprite', 'points']) {
    const opaqueCaster = renderBillboardAlphaShadow(kind, 255)
    const cutoutCaster = renderBillboardAlphaShadow(kind, 0)
    const opaqueLum = opaqueCaster.r + opaqueCaster.g + opaqueCaster.b
    const cutoutLum = cutoutCaster.r + cutoutCaster.g + cutoutCaster.b
    assert.ok(cutoutLum > opaqueLum + 10, `${kind} material alphaMap should remove the caster shadow (${cutoutLum} vs ${opaqueLum})`)
  }
})

test('SpriteMaterial and PointsMaterial base-map alpha cutouts affect directional shadows', () => {
  function renderBillboardMapAlphaShadow(kind, mapAlpha) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    if (kind === 'sprite') {
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
        color: 0xffffff,
        map: solidTexture(255, 255, 255, mapAlpha),
        alphaTest: 0.5,
      }))
      sprite.position.set(0, 4, 0)
      sprite.scale.set(4, 4, 1)
      sprite.castShadow = true
      scene.add(sprite)
    } else {
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 4, 0]), 3))
      const points = new THREE.Points(geometry, new THREE.PointsMaterial({
        color: 0xffffff,
        size: 48,
        sizeAttenuation: false,
        map: solidTexture(255, 255, 255, mapAlpha),
        alphaTest: 0.5,
      }))
      points.castShadow = true
      scene.add(points)
    }

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(0, 6, 8)
    light.target.position.set(0, 0, 0)
    light.castShadow = true
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

  for (const kind of ['sprite', 'points']) {
    const opaqueCaster = renderBillboardMapAlphaShadow(kind, 255)
    const cutoutCaster = renderBillboardMapAlphaShadow(kind, 0)
    const opaqueLum = opaqueCaster.r + opaqueCaster.g + opaqueCaster.b
    const cutoutLum = cutoutCaster.r + cutoutCaster.g + cutoutCaster.b
    assert.ok(cutoutLum > opaqueLum + 10, `${kind} material map alpha should remove the caster shadow (${cutoutLum} vs ${opaqueLum})`)
  }
})

test('SpriteMaterial and PointsMaterial alphaToCoverage approximate shadow caster cutouts', () => {
  function renderBillboardAlphaCoverageShadow(kind, alphaToCoverage) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    if (kind === 'sprite') {
      const material = new THREE.SpriteMaterial({ color: 0xffffff })
      material.alphaMap = solidTexture(255, 0, 255)
      material.alphaToCoverage = alphaToCoverage
      const sprite = new THREE.Sprite(material)
      sprite.position.set(0, 4, 0)
      sprite.scale.set(4, 4, 1)
      sprite.castShadow = true
      scene.add(sprite)
    } else {
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 4, 0]), 3))
      const material = new THREE.PointsMaterial({
        color: 0xffffff,
        size: 48,
        sizeAttenuation: false,
      })
      material.alphaMap = solidTexture(255, 0, 255)
      material.alphaToCoverage = alphaToCoverage
      const points = new THREE.Points(geometry, material)
      points.castShadow = true
      scene.add(points)
    }

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(0, 6, 8)
    light.target.position.set(0, 0, 0)
    light.castShadow = true
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

  for (const kind of ['sprite', 'points']) {
    const fullShadow = renderBillboardAlphaCoverageShadow(kind, false)
    const cutoutShadow = renderBillboardAlphaCoverageShadow(kind, true)
    const fullLum = fullShadow.r + fullShadow.g + fullShadow.b
    const cutoutLum = cutoutShadow.r + cutoutShadow.g + cutoutShadow.b
    assert.ok(cutoutLum > fullLum + 10, `${kind} alphaToCoverage shadow cutoff should let more receiver light through (${cutoutLum} vs ${fullLum})`)
  }
})

test('SpriteMaterial and PointsMaterial alphaHash approximate shadow caster opacity cutouts', () => {
  function renderBillboardAlphaHashShadow(kind, alphaHash) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    if (kind === 'sprite') {
      const material = new THREE.SpriteMaterial({
        color: 0xffffff,
        opacity: alphaHash ? 0.35 : 1,
        alphaHash,
      })
      const sprite = new THREE.Sprite(material)
      sprite.position.set(0, 4, 0)
      sprite.scale.set(4, 4, 1)
      sprite.castShadow = true
      scene.add(sprite)
    } else {
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 4, 0]), 3))
      const material = new THREE.PointsMaterial({
        color: 0xffffff,
        opacity: alphaHash ? 0.35 : 1,
        alphaHash,
        size: 48,
        sizeAttenuation: false,
      })
      const points = new THREE.Points(geometry, material)
      points.castShadow = true
      scene.add(points)
    }

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(0, 6, 8)
    light.target.position.set(0, 0, 0)
    light.castShadow = true
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

  for (const kind of ['sprite', 'points']) {
    const fullShadow = renderBillboardAlphaHashShadow(kind, false)
    const hashedShadow = renderBillboardAlphaHashShadow(kind, true)
    const fullLum = fullShadow.r + fullShadow.g + fullShadow.b
    const hashedLum = hashedShadow.r + hashedShadow.g + hashedShadow.b
    assert.ok(hashedLum > fullLum + 10, `${kind} alphaHash shadow cutoff should let more receiver light through (${hashedLum} vs ${fullLum})`)
  }
})

test('SpriteMaterial and PointsMaterial cast spot-light shadows from expanded billboards', () => {
  function renderSpotBillboardShadow(kind, castShadow) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    if (kind === 'sprite') {
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ color: 0xffffff }))
      sprite.position.set(0, 4, 0)
      sprite.scale.set(4, 4, 1)
      sprite.castShadow = castShadow
      scene.add(sprite)
    } else {
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 4, 0]), 3))
      const points = new THREE.Points(geometry, new THREE.PointsMaterial({
        color: 0xffffff,
        size: 48,
        sizeAttenuation: false,
      }))
      points.castShadow = castShadow
      scene.add(points)
    }

    const light = new THREE.SpotLight(0xffffff, 4, 20, Math.PI / 4, 0.2, 2)
    light.position.set(0, 7, 8)
    light.target.position.set(0, 0, 0)
    light.castShadow = true
    light.shadow.mapSize.set(256, 256)
    light.shadow.camera.near = 0.1
    light.shadow.camera.far = 20
    scene.add(light)
    scene.add(light.target)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 96, height: 96 }))
  }

  for (const kind of ['sprite', 'points']) {
    const unshadowed = renderSpotBillboardShadow(kind, false)
    const shadowed = renderSpotBillboardShadow(kind, true)
    const unshadowedLum = unshadowed.r + unshadowed.g + unshadowed.b
    const shadowedLum = shadowed.r + shadowed.g + shadowed.b
    assert.ok(shadowedLum < unshadowedLum - 10, `${kind} spot-light shadow should darken the receiver (${shadowedLum} vs ${unshadowedLum})`)
  }
})

test('Sprite and Points customDepthMaterial alphaMap controls spot-light shadow casters', () => {
  function renderSpotBillboardCustomDepthShadow(kind, alphaMapGreen) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    if (kind === 'sprite') {
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ color: 0xffffff }))
      sprite.position.set(0, 4, 0)
      sprite.scale.set(4, 4, 1)
      sprite.castShadow = true
      sprite.customDepthMaterial = new THREE.MeshDepthMaterial({
        alphaMap: solidTexture(255, alphaMapGreen, 255),
        alphaTest: 0.5,
      })
      scene.add(sprite)
    } else {
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 4, 0]), 3))
      const points = new THREE.Points(geometry, new THREE.PointsMaterial({
        color: 0xffffff,
        size: 48,
        sizeAttenuation: false,
      }))
      points.castShadow = true
      points.customDepthMaterial = new THREE.MeshDepthMaterial({
        alphaMap: solidTexture(255, alphaMapGreen, 255),
        alphaTest: 0.5,
      })
      scene.add(points)
    }

    const light = new THREE.SpotLight(0xffffff, 4, 20, Math.PI / 4, 0.2, 2)
    light.position.set(0, 7, 8)
    light.target.position.set(0, 0, 0)
    light.castShadow = true
    light.shadow.mapSize.set(256, 256)
    light.shadow.camera.near = 0.1
    light.shadow.camera.far = 20
    scene.add(light)
    scene.add(light.target)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 96, height: 96 }))
  }

  for (const kind of ['sprite', 'points']) {
    const opaqueCaster = renderSpotBillboardCustomDepthShadow(kind, 255)
    const cutoutCaster = renderSpotBillboardCustomDepthShadow(kind, 0)
    const opaqueLum = opaqueCaster.r + opaqueCaster.g + opaqueCaster.b
    const cutoutLum = cutoutCaster.r + cutoutCaster.g + cutoutCaster.b
    assert.ok(cutoutLum > opaqueLum + 10, `${kind} customDepthMaterial alphaMap should remove the spot-shadow caster (${cutoutLum} vs ${opaqueLum})`)
  }
})

test('Sprite customDepthMaterial alphaMap controls directional shadow casters', () => {
  function renderSpriteCustomDepthShadow(alphaMapGreen) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ color: 0xffffff }))
    sprite.position.set(0, 4, 0)
    sprite.scale.set(4, 4, 1)
    sprite.castShadow = true
    sprite.customDepthMaterial = new THREE.MeshDepthMaterial({
      alphaMap: solidTexture(255, alphaMapGreen, 255),
      alphaTest: 0.5,
    })
    scene.add(sprite)

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(0, 6, 8)
    light.target.position.set(0, 0, 0)
    light.castShadow = true
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

  const opaqueCaster = renderSpriteCustomDepthShadow(255)
  const cutoutCaster = renderSpriteCustomDepthShadow(0)
  const opaqueLum = opaqueCaster.r + opaqueCaster.g + opaqueCaster.b
  const cutoutLum = cutoutCaster.r + cutoutCaster.g + cutoutCaster.b
  assert.ok(cutoutLum > opaqueLum + 10, `sprite customDepthMaterial alphaMap should remove the caster shadow (${cutoutLum} vs ${opaqueLum})`)
})

test('Points customDepthMaterial alphaMap controls directional shadow casters', () => {
  function renderPointsCustomDepthShadow(alphaMapGreen) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 4, 0]), 3))
    const points = new THREE.Points(geometry, new THREE.PointsMaterial({
      color: 0xffffff,
      size: 48,
      sizeAttenuation: false,
    }))
    points.castShadow = true
    points.customDepthMaterial = new THREE.MeshDepthMaterial({
      alphaMap: solidTexture(255, alphaMapGreen, 255),
      alphaTest: 0.5,
    })
    scene.add(points)

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(0, 6, 8)
    light.target.position.set(0, 0, 0)
    light.castShadow = true
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

  const opaqueCaster = renderPointsCustomDepthShadow(255)
  const cutoutCaster = renderPointsCustomDepthShadow(0)
  const opaqueLum = opaqueCaster.r + opaqueCaster.g + opaqueCaster.b
  const cutoutLum = cutoutCaster.r + cutoutCaster.g + cutoutCaster.b
  assert.ok(cutoutLum > opaqueLum + 10, `points customDepthMaterial alphaMap should remove the caster shadow (${cutoutLum} vs ${opaqueLum})`)
})

test('Sprite customDistanceMaterial alphaMap controls point-light shadow casters', () => {
  function renderSpriteCustomDistanceShadow(alphaMapGreen) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ color: 0xffffff }))
    sprite.position.set(0, 2.2, 1.8)
    sprite.scale.set(4, 4, 1)
    sprite.castShadow = true
    sprite.customDistanceMaterial = new THREE.MeshDistanceMaterial({
      alphaMap: solidTexture(255, alphaMapGreen, 255),
      alphaTest: 0.5,
    })
    scene.add(sprite)

    const light = new THREE.PointLight(0xffffff, 2)
    light.position.set(0, 5, 4)
    light.distance = 12
    light.castShadow = true
    light.shadow.mapSize.set(256, 256)
    light.shadow.camera.near = 0.1
    light.shadow.camera.far = 12
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, { width: 96, height: 96 }), 96, 96, 28, 42, 68, 82)
  }

  const opaqueCaster = renderSpriteCustomDistanceShadow(255)
  const cutoutCaster = renderSpriteCustomDistanceShadow(0)
  const opaqueLum = opaqueCaster.r + opaqueCaster.g + opaqueCaster.b
  const cutoutLum = cutoutCaster.r + cutoutCaster.g + cutoutCaster.b
  assert.ok(cutoutLum > opaqueLum + 10, `sprite customDistanceMaterial alphaMap should remove the caster shadow (${cutoutLum} vs ${opaqueLum})`)
})

test('Points customDistanceMaterial alphaMap controls point-light shadow casters', () => {
  function renderPointsCustomDistanceShadow(alphaMapGreen) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 2.2, 1.8]), 3))
    const points = new THREE.Points(geometry, new THREE.PointsMaterial({
      color: 0xffffff,
      size: 48,
      sizeAttenuation: false,
    }))
    points.castShadow = true
    points.customDistanceMaterial = new THREE.MeshDistanceMaterial({
      alphaMap: solidTexture(255, alphaMapGreen, 255),
      alphaTest: 0.5,
    })
    scene.add(points)

    const light = new THREE.PointLight(0xffffff, 2)
    light.position.set(0, 5, 4)
    light.distance = 12
    light.castShadow = true
    light.shadow.mapSize.set(256, 256)
    light.shadow.camera.near = 0.1
    light.shadow.camera.far = 12
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, { width: 96, height: 96 }), 96, 96, 28, 42, 68, 82)
  }

  const opaqueCaster = renderPointsCustomDistanceShadow(255)
  const cutoutCaster = renderPointsCustomDistanceShadow(0)
  const opaqueLum = opaqueCaster.r + opaqueCaster.g + opaqueCaster.b
  const cutoutLum = cutoutCaster.r + cutoutCaster.g + cutoutCaster.b
  assert.ok(cutoutLum > opaqueLum + 10, `points customDistanceMaterial alphaMap should remove the caster shadow (${cutoutLum} vs ${opaqueLum})`)
})

test('Sprite and Points source alphaHash applies to custom shadow material casters', () => {
  function sourceBillboardMaterial(kind, alphaHash) {
    const params = {
      color: 0xffffff,
      opacity: alphaHash ? 0.35 : 1,
      alphaHash,
    }
    if (kind === 'sprite') return new THREE.SpriteMaterial(params)
    return new THREE.PointsMaterial({
      ...params,
      size: 48,
      sizeAttenuation: false,
    })
  }

  function addBillboard(scene, kind, alphaHash, shadowProperty, shadowMaterial, position) {
    const material = sourceBillboardMaterial(kind, alphaHash)
    if (kind === 'sprite') {
      const sprite = new THREE.Sprite(material)
      sprite.position.set(...position)
      sprite.scale.set(4, 4, 1)
      sprite.castShadow = true
      sprite[shadowProperty] = shadowMaterial
      scene.add(sprite)
      return
    }

    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(position), 3))
    const points = new THREE.Points(geometry, material)
    points.castShadow = true
    points[shadowProperty] = shadowMaterial
    scene.add(points)
  }

  function renderDirectionalCustomDepthSourceAlphaHash(kind, alphaHash) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    addBillboard(scene, kind, alphaHash, 'customDepthMaterial', new THREE.MeshDepthMaterial(), [0, 4, 0])

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(0, 6, 8)
    light.target.position.set(0, 0, 0)
    light.castShadow = true
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

  function renderPointCustomDistanceSourceAlphaHash(kind, alphaHash) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    addBillboard(scene, kind, alphaHash, 'customDistanceMaterial', new THREE.MeshDistanceMaterial(), [0, 2.2, 1.8])

    const light = new THREE.PointLight(0xffffff, 2)
    light.position.set(0, 5, 4)
    light.distance = 12
    light.castShadow = true
    light.shadow.mapSize.set(256, 256)
    light.shadow.camera.near = 0.1
    light.shadow.camera.far = 12
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, { width: 96, height: 96 }), 96, 96, 28, 42, 68, 82)
  }

  for (const kind of ['sprite', 'points']) {
    const fullDepth = renderDirectionalCustomDepthSourceAlphaHash(kind, false)
    const hashedDepth = renderDirectionalCustomDepthSourceAlphaHash(kind, true)
    const fullDepthLum = fullDepth.r + fullDepth.g + fullDepth.b
    const hashedDepthLum = hashedDepth.r + hashedDepth.g + hashedDepth.b
    assert.ok(hashedDepthLum > fullDepthLum + 10, `${kind} source alphaHash should lighten the customDepthMaterial caster shadow (${hashedDepthLum} vs ${fullDepthLum})`)

    const fullDistance = renderPointCustomDistanceSourceAlphaHash(kind, false)
    const hashedDistance = renderPointCustomDistanceSourceAlphaHash(kind, true)
    const fullDistanceLum = fullDistance.r + fullDistance.g + fullDistance.b
    const hashedDistanceLum = hashedDistance.r + hashedDistance.g + hashedDistance.b
    assert.ok(hashedDistanceLum > fullDistanceLum + 10, `${kind} source alphaHash should lighten the customDistanceMaterial caster shadow (${hashedDistanceLum} vs ${fullDistanceLum})`)
  }
})

test('Sprite and Points source alphaToCoverage opacity applies to custom shadow material casters', () => {
  function sourceBillboardMaterial(kind, alphaToCoverage) {
    const params = {
      color: 0xffffff,
      opacity: alphaToCoverage ? 0.35 : 1,
      alphaToCoverage,
    }
    if (kind === 'sprite') return new THREE.SpriteMaterial(params)
    return new THREE.PointsMaterial({
      ...params,
      size: 48,
      sizeAttenuation: false,
    })
  }

  function addBillboard(scene, kind, alphaToCoverage, shadowProperty, shadowMaterial, position) {
    const material = sourceBillboardMaterial(kind, alphaToCoverage)
    if (kind === 'sprite') {
      const sprite = new THREE.Sprite(material)
      sprite.position.set(...position)
      sprite.scale.set(4, 4, 1)
      sprite.castShadow = true
      sprite[shadowProperty] = shadowMaterial
      scene.add(sprite)
      return
    }

    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(position), 3))
    const points = new THREE.Points(geometry, material)
    points.castShadow = true
    points[shadowProperty] = shadowMaterial
    scene.add(points)
  }

  function renderDirectionalCustomDepthSourceAlphaCoverage(kind, alphaToCoverage) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    addBillboard(scene, kind, alphaToCoverage, 'customDepthMaterial', new THREE.MeshDepthMaterial(), [0, 4, 0])

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(0, 6, 8)
    light.target.position.set(0, 0, 0)
    light.castShadow = true
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

  function renderPointCustomDistanceSourceAlphaCoverage(kind, alphaToCoverage) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    addBillboard(scene, kind, alphaToCoverage, 'customDistanceMaterial', new THREE.MeshDistanceMaterial(), [0, 2.2, 1.8])

    const light = new THREE.PointLight(0xffffff, 2)
    light.position.set(0, 5, 4)
    light.distance = 12
    light.castShadow = true
    light.shadow.mapSize.set(256, 256)
    light.shadow.camera.near = 0.1
    light.shadow.camera.far = 12
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, { width: 96, height: 96 }), 96, 96, 28, 42, 68, 82)
  }

  for (const kind of ['sprite', 'points']) {
    const fullDepth = renderDirectionalCustomDepthSourceAlphaCoverage(kind, false)
    const coveredDepth = renderDirectionalCustomDepthSourceAlphaCoverage(kind, true)
    const fullDepthLum = fullDepth.r + fullDepth.g + fullDepth.b
    const coveredDepthLum = coveredDepth.r + coveredDepth.g + coveredDepth.b
    assert.ok(coveredDepthLum > fullDepthLum + 10, `${kind} source alphaToCoverage opacity should lighten the customDepthMaterial caster shadow (${coveredDepthLum} vs ${fullDepthLum})`)

    const fullDistance = renderPointCustomDistanceSourceAlphaCoverage(kind, false)
    const coveredDistance = renderPointCustomDistanceSourceAlphaCoverage(kind, true)
    const fullDistanceLum = fullDistance.r + fullDistance.g + fullDistance.b
    const coveredDistanceLum = coveredDistance.r + coveredDistance.g + coveredDistance.b
    assert.ok(coveredDistanceLum > fullDistanceLum + 10, `${kind} source alphaToCoverage opacity should lighten the customDistanceMaterial caster shadow (${coveredDistanceLum} vs ${fullDistanceLum})`)
  }
})

test('customDepthMaterial alphaMap controls directional shadow casters', () => {
  function renderCustomDepthShadow(alphaMapGreen) {
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
    const customDepthMaterial = new THREE.MeshDepthMaterial({
      alphaMap: solidTexture(255, alphaMapGreen, 255),
      alphaTest: 0.5,
    })
    caster.customDepthMaterial = customDepthMaterial
    scene.add(caster)

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

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 96, height: 96 }))
  }

  const opaqueCaster = renderCustomDepthShadow(255)
  const cutoutCaster = renderCustomDepthShadow(0)
  const opaqueLum = opaqueCaster.r + opaqueCaster.g + opaqueCaster.b
  const cutoutLum = cutoutCaster.r + cutoutCaster.g + cutoutCaster.b
  assert.ok(cutoutLum > opaqueLum + 30, `customDepthMaterial alphaMap should remove the caster shadow (${cutoutLum} vs ${opaqueLum})`)
})

test('customDepthMaterial alphaMap controls spot-light shadow casters', () => {
  function renderSpotCustomDepthShadow(alphaMapGreen) {
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
      new THREE.BoxGeometry(2.5, 2.5, 2.5),
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    )
    caster.position.y = 1.25
    caster.castShadow = true
    const customDepthMaterial = new THREE.MeshDepthMaterial({
      alphaMap: solidTexture(255, alphaMapGreen, 255),
      alphaTest: 0.5,
    })
    caster.customDepthMaterial = customDepthMaterial
    scene.add(caster)

    const light = new THREE.SpotLight(0xffffff, 4, 20, Math.PI / 4, 0.2, 2)
    light.position.set(0, 7, 8)
    light.target.position.set(0, 0, 0)
    light.castShadow = true
    light.shadow.mapSize.set(256, 256)
    light.shadow.camera.near = 0.1
    light.shadow.camera.far = 20
    scene.add(light)
    scene.add(light.target)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, { width: 96, height: 96 }), 96, 96, 28, 31, 68, 57)
  }

  const opaqueCaster = renderSpotCustomDepthShadow(255)
  const cutoutCaster = renderSpotCustomDepthShadow(0)
  const opaqueLum = opaqueCaster.r + opaqueCaster.g + opaqueCaster.b
  const cutoutLum = cutoutCaster.r + cutoutCaster.g + cutoutCaster.b
  assert.ok(cutoutLum > opaqueLum + 8, `customDepthMaterial alphaMap should remove the spot-shadow caster (${cutoutLum} vs ${opaqueLum})`)
})

test('customDistanceMaterial alphaMap controls point-light shadow casters', () => {
  function renderCustomDistanceShadow(alphaMapGreen) {
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
      new THREE.BoxGeometry(2.5, 2.5, 2.5),
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    )
    caster.position.y = 1.25
    caster.castShadow = true
    const customDistanceMaterial = new THREE.MeshDistanceMaterial({
      alphaMap: solidTexture(255, alphaMapGreen, 255),
      alphaTest: 0.5,
    })
    caster.customDistanceMaterial = customDistanceMaterial
    scene.add(caster)

    const light = new THREE.PointLight(0xffffff, 2)
    light.position.set(0, 5, 4)
    light.distance = 12
    light.castShadow = true
    light.shadow.mapSize.set(256, 256)
    light.shadow.camera.near = 0.1
    light.shadow.camera.far = 12
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 96, height: 96 }))
  }

  const opaqueCaster = renderCustomDistanceShadow(255)
  const cutoutCaster = renderCustomDistanceShadow(0)
  const opaqueLum = opaqueCaster.r + opaqueCaster.g + opaqueCaster.b
  const cutoutLum = cutoutCaster.r + cutoutCaster.g + cutoutCaster.b
  assert.ok(cutoutLum > opaqueLum + 20, `customDistanceMaterial alphaMap should remove the point-shadow caster (${cutoutLum} vs ${opaqueLum})`)
})

test('custom shadow material base-map alpha controls mesh shadow casters', () => {
  function renderDirectionalCustomDepthMapShadow(mapAlpha) {
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
    caster.customDepthMaterial = new THREE.MeshDepthMaterial({
      map: solidTexture(255, 255, 255, mapAlpha),
      alphaTest: 0.5,
    })
    scene.add(caster)

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

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 96, height: 96 }))
  }

  function renderPointCustomDistanceMapShadow(mapAlpha) {
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
      new THREE.BoxGeometry(2.5, 2.5, 2.5),
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    )
    caster.position.y = 1.25
    caster.castShadow = true
    caster.customDistanceMaterial = new THREE.MeshDistanceMaterial({
      map: solidTexture(255, 255, 255, mapAlpha),
      alphaTest: 0.5,
    })
    scene.add(caster)

    const light = new THREE.PointLight(0xffffff, 2)
    light.position.set(0, 5, 4)
    light.distance = 12
    light.castShadow = true
    light.shadow.mapSize.set(256, 256)
    light.shadow.camera.near = 0.1
    light.shadow.camera.far = 12
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 96, height: 96 }))
  }

  const opaqueDepth = renderDirectionalCustomDepthMapShadow(255)
  const cutoutDepth = renderDirectionalCustomDepthMapShadow(0)
  const opaqueDepthLum = opaqueDepth.r + opaqueDepth.g + opaqueDepth.b
  const cutoutDepthLum = cutoutDepth.r + cutoutDepth.g + cutoutDepth.b
  assert.ok(cutoutDepthLum > opaqueDepthLum + 30, `customDepthMaterial map alpha should remove the caster shadow (${cutoutDepthLum} vs ${opaqueDepthLum})`)

  const opaqueDistance = renderPointCustomDistanceMapShadow(255)
  const cutoutDistance = renderPointCustomDistanceMapShadow(0)
  const opaqueDistanceLum = opaqueDistance.r + opaqueDistance.g + opaqueDistance.b
  const cutoutDistanceLum = cutoutDistance.r + cutoutDistance.g + cutoutDistance.b
  assert.ok(cutoutDistanceLum > opaqueDistanceLum + 20, `customDistanceMaterial map alpha should remove the caster shadow (${cutoutDistanceLum} vs ${opaqueDistanceLum})`)
})

test('source material alpha cutouts apply to custom shadow material casters', () => {
  function renderDirectionalCustomDepthSourceAlpha(alphaMapGreen) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const sourceMaterial = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      alphaMap: solidTexture(255, alphaMapGreen, 255),
      alphaTest: 0.5,
      colorWrite: false,
      depthWrite: false,
    })
    const caster = new THREE.Mesh(new THREE.BoxGeometry(3, 3, 3), sourceMaterial)
    caster.position.y = 1.5
    caster.castShadow = true
    caster.customDepthMaterial = new THREE.MeshDepthMaterial()
    scene.add(caster)

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

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 96, height: 96 }))
  }

  function renderPointCustomDistanceSourceAlpha(mapAlpha) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const sourceMaterial = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      map: solidTexture(255, 255, 255, mapAlpha),
      alphaTest: 0.5,
      colorWrite: false,
      depthWrite: false,
    })
    const caster = new THREE.Mesh(new THREE.BoxGeometry(2.5, 2.5, 2.5), sourceMaterial)
    caster.position.y = 1.25
    caster.castShadow = true
    caster.customDistanceMaterial = new THREE.MeshDistanceMaterial()
    scene.add(caster)

    const light = new THREE.PointLight(0xffffff, 2)
    light.position.set(0, 5, 4)
    light.distance = 12
    light.castShadow = true
    light.shadow.mapSize.set(256, 256)
    light.shadow.camera.near = 0.1
    light.shadow.camera.far = 12
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 96, height: 96 }))
  }

  const opaqueDepth = renderDirectionalCustomDepthSourceAlpha(255)
  const cutoutDepth = renderDirectionalCustomDepthSourceAlpha(0)
  const opaqueDepthLum = opaqueDepth.r + opaqueDepth.g + opaqueDepth.b
  const cutoutDepthLum = cutoutDepth.r + cutoutDepth.g + cutoutDepth.b
  assert.ok(cutoutDepthLum > opaqueDepthLum + 30, `source alphaMap should remove the customDepthMaterial caster shadow (${cutoutDepthLum} vs ${opaqueDepthLum})`)

  const opaqueDistance = renderPointCustomDistanceSourceAlpha(255)
  const cutoutDistance = renderPointCustomDistanceSourceAlpha(0)
  const opaqueDistanceLum = opaqueDistance.r + opaqueDistance.g + opaqueDistance.b
  const cutoutDistanceLum = cutoutDistance.r + cutoutDistance.g + cutoutDistance.b
  assert.ok(cutoutDistanceLum > opaqueDistanceLum + 20, `source map alpha should remove the customDistanceMaterial caster shadow (${cutoutDistanceLum} vs ${opaqueDistanceLum})`)
})

test('source material alphaHash applies to custom shadow material casters', () => {
  function sourceMaterial(alphaHash) {
    return new THREE.MeshBasicMaterial({
      color: 0xffffff,
      opacity: alphaHash ? 0.35 : 1,
      alphaHash,
      colorWrite: false,
      depthWrite: false,
    })
  }

  function renderDirectionalCustomDepthSourceAlphaHash(alphaHash) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const caster = new THREE.Mesh(new THREE.BoxGeometry(3, 3, 3), sourceMaterial(alphaHash))
    caster.position.y = 1.5
    caster.castShadow = true
    caster.customDepthMaterial = new THREE.MeshDepthMaterial()
    scene.add(caster)

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

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 96, height: 96 }))
  }

  function renderPointCustomDistanceSourceAlphaHash(alphaHash) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const caster = new THREE.Mesh(new THREE.BoxGeometry(2.5, 2.5, 2.5), sourceMaterial(alphaHash))
    caster.position.y = 1.25
    caster.castShadow = true
    caster.customDistanceMaterial = new THREE.MeshDistanceMaterial()
    scene.add(caster)

    const light = new THREE.PointLight(0xffffff, 2)
    light.position.set(0, 5, 4)
    light.distance = 12
    light.castShadow = true
    light.shadow.mapSize.set(256, 256)
    light.shadow.camera.near = 0.1
    light.shadow.camera.far = 12
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 96, height: 96 }))
  }

  const fullDepth = renderDirectionalCustomDepthSourceAlphaHash(false)
  const hashedDepth = renderDirectionalCustomDepthSourceAlphaHash(true)
  const fullDepthLum = fullDepth.r + fullDepth.g + fullDepth.b
  const hashedDepthLum = hashedDepth.r + hashedDepth.g + hashedDepth.b
  assert.ok(hashedDepthLum > fullDepthLum + 20, `source alphaHash should lighten the customDepthMaterial caster shadow (${hashedDepthLum} vs ${fullDepthLum})`)

  const fullDistance = renderPointCustomDistanceSourceAlphaHash(false)
  const hashedDistance = renderPointCustomDistanceSourceAlphaHash(true)
  const fullDistanceLum = fullDistance.r + fullDistance.g + fullDistance.b
  const hashedDistanceLum = hashedDistance.r + hashedDistance.g + hashedDistance.b
  assert.ok(hashedDistanceLum > fullDistanceLum + 10, `source alphaHash should lighten the customDistanceMaterial caster shadow (${hashedDistanceLum} vs ${fullDistanceLum})`)
})

test('source material alphaToCoverage opacity applies to custom shadow material casters', () => {
  function sourceMaterial(alphaToCoverage) {
    return new THREE.MeshBasicMaterial({
      color: 0xffffff,
      opacity: alphaToCoverage ? 0.35 : 1,
      alphaToCoverage,
      colorWrite: false,
      depthWrite: false,
    })
  }

  function renderDirectionalCustomDepthSourceAlphaCoverage(alphaToCoverage) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const caster = new THREE.Mesh(new THREE.BoxGeometry(3, 3, 3), sourceMaterial(alphaToCoverage))
    caster.position.y = 1.5
    caster.castShadow = true
    caster.customDepthMaterial = new THREE.MeshDepthMaterial()
    scene.add(caster)

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

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 96, height: 96 }))
  }

  function renderPointCustomDistanceSourceAlphaCoverage(alphaToCoverage) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const caster = new THREE.Mesh(new THREE.BoxGeometry(2.5, 2.5, 2.5), sourceMaterial(alphaToCoverage))
    caster.position.y = 1.25
    caster.castShadow = true
    caster.customDistanceMaterial = new THREE.MeshDistanceMaterial()
    scene.add(caster)

    const light = new THREE.PointLight(0xffffff, 2)
    light.position.set(0, 5, 4)
    light.distance = 12
    light.castShadow = true
    light.shadow.mapSize.set(256, 256)
    light.shadow.camera.near = 0.1
    light.shadow.camera.far = 12
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 96, height: 96 }))
  }

  const fullDepth = renderDirectionalCustomDepthSourceAlphaCoverage(false)
  const coveredDepth = renderDirectionalCustomDepthSourceAlphaCoverage(true)
  const fullDepthLum = fullDepth.r + fullDepth.g + fullDepth.b
  const coveredDepthLum = coveredDepth.r + coveredDepth.g + coveredDepth.b
  assert.ok(coveredDepthLum > fullDepthLum + 20, `source alphaToCoverage opacity should lighten the customDepthMaterial caster shadow (${coveredDepthLum} vs ${fullDepthLum})`)

  const fullDistance = renderPointCustomDistanceSourceAlphaCoverage(false)
  const coveredDistance = renderPointCustomDistanceSourceAlphaCoverage(true)
  const fullDistanceLum = fullDistance.r + fullDistance.g + fullDistance.b
  const coveredDistanceLum = coveredDistance.r + coveredDistance.g + coveredDistance.b
  assert.ok(coveredDistanceLum > fullDistanceLum + 10, `source alphaToCoverage opacity should lighten the customDistanceMaterial caster shadow (${coveredDistanceLum} vs ${fullDistanceLum})`)
})

test('customDepthMaterial displacement shifts directional shadow casters', () => {
  function renderCustomDepthDisplacement(displacementScale) {
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
      new THREE.PlaneGeometry(2.5, 2.5, 8, 8),
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    )
    caster.position.set(0, 1.7, 0)
    caster.rotation.x = -Math.PI / 2
    caster.castShadow = true
    caster.customDepthMaterial = new THREE.MeshDepthMaterial({
      displacementMap: solidTexture(255, 0, 0),
      displacementScale,
      displacementBias: 0,
    })
    scene.add(caster)

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

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    return renderRgba(scene, camera, { width: 96, height: 96 })
  }

  const flat = renderCustomDepthDisplacement(0)
  const displaced = renderCustomDepthDisplacement(2)
  const diff = meanAbsDiff(flat, displaced)
  assert.ok(diff > 5, `customDepthMaterial displacement should move the directional caster shadow, diff=${diff.toFixed(3)}`)
})

test('customDistanceMaterial displacement shifts point-light shadow casters', () => {
  function renderCustomDistanceDisplacement(displacementScale) {
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
      new THREE.PlaneGeometry(2.5, 2.5, 8, 8),
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    )
    caster.position.set(0, 1.7, 0)
    caster.rotation.x = -Math.PI / 2
    caster.castShadow = true
    caster.customDistanceMaterial = new THREE.MeshDistanceMaterial({
      displacementMap: solidTexture(255, 0, 0),
      displacementScale,
      displacementBias: 0,
    })
    scene.add(caster)

    const light = new THREE.PointLight(0xffffff, 2)
    light.position.set(0, 5, 4)
    light.distance = 12
    light.castShadow = true
    light.shadow.mapSize.set(256, 256)
    light.shadow.camera.near = 0.1
    light.shadow.camera.far = 12
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    return renderRgba(scene, camera, { width: 96, height: 96 })
  }

  const flat = renderCustomDistanceDisplacement(0)
  const displaced = renderCustomDistanceDisplacement(2)
  const diff = meanAbsDiff(flat, displaced)
  assert.ok(diff > 5, `customDistanceMaterial displacement should move the point-light caster shadow, diff=${diff.toFixed(3)}`)
})

test('source material displacement applies to custom shadow material casters', () => {
  function renderDirectionalCustomDepthSourceDisplacement(displacementScale) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const sourceMaterial = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      displacementMap: solidTexture(255, 0, 0),
      displacementScale,
      displacementBias: 0,
      colorWrite: false,
      depthWrite: false,
    })
    const caster = new THREE.Mesh(
      new THREE.PlaneGeometry(2.5, 2.5, 8, 8),
      sourceMaterial,
    )
    caster.position.set(0, 1.7, 0)
    caster.rotation.x = -Math.PI / 2
    caster.castShadow = true
    caster.customDepthMaterial = new THREE.MeshDepthMaterial()
    scene.add(caster)

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

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    return renderRgba(scene, camera, { width: 96, height: 96 })
  }

  function renderPointCustomDistanceSourceDisplacement(displacementScale) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const sourceMaterial = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      displacementMap: solidTexture(255, 0, 0),
      displacementScale,
      displacementBias: 0,
      colorWrite: false,
      depthWrite: false,
    })
    const caster = new THREE.Mesh(
      new THREE.PlaneGeometry(2.5, 2.5, 8, 8),
      sourceMaterial,
    )
    caster.position.set(0, 1.7, 0)
    caster.rotation.x = -Math.PI / 2
    caster.castShadow = true
    caster.customDistanceMaterial = new THREE.MeshDistanceMaterial()
    scene.add(caster)

    const light = new THREE.PointLight(0xffffff, 2)
    light.position.set(0, 5, 4)
    light.distance = 12
    light.castShadow = true
    light.shadow.mapSize.set(256, 256)
    light.shadow.camera.near = 0.1
    light.shadow.camera.far = 12
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    return renderRgba(scene, camera, { width: 96, height: 96 })
  }

  const flatDepth = renderDirectionalCustomDepthSourceDisplacement(0)
  const displacedDepth = renderDirectionalCustomDepthSourceDisplacement(2)
  const depthDiff = meanAbsDiff(flatDepth, displacedDepth)
  assert.ok(depthDiff > 5, `source material displacement should move the customDepthMaterial shadow, diff=${depthDiff.toFixed(3)}`)

  const flatDistance = renderPointCustomDistanceSourceDisplacement(0)
  const displacedDistance = renderPointCustomDistanceSourceDisplacement(2)
  const distanceDiff = meanAbsDiff(flatDistance, displacedDistance)
  assert.ok(distanceDiff > 5, `source material displacement should move the customDistanceMaterial shadow, diff=${distanceDiff.toFixed(3)}`)
})

test('malformed custom shadow material containers fail clearly', () => {
  function makeScene(property, customShadowMaterial, light) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const caster = new THREE.Mesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    )
    caster.castShadow = true
    caster[property] = customShadowMaterial
    scene.add(caster)
    scene.add(light)
    if (light.target) scene.add(light.target)
    return scene
  }

  const directional = new THREE.DirectionalLight(0xffffff, 1)
  directional.castShadow = true
  directional.position.set(2, 4, 3)
  directional.target.position.set(0, 0, 0)
  assert.throws(
    () => renderRgba(
      makeScene('customDepthMaterial', 'depth', directional),
      makeCamera(),
      { width: 64, height: 64 },
    ),
    /Object3D\.customDepthMaterial must be a material-like object/i,
  )

  const point = new THREE.PointLight(0xffffff, 1)
  point.castShadow = true
  point.position.set(2, 4, 3)
  assert.throws(
    () => renderRgba(
      makeScene('customDistanceMaterial', [], point),
      makeCamera(),
      { width: 64, height: 64 },
    ),
    /Object3D\.customDistanceMaterial must be a material-like object/i,
  )
})

test('custom shadow material wireframes cast thinner shadows than filled custom casters', () => {
  function renderDirectionalCustomDepthShadow(wireframe, castShadow = true) {
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
      new THREE.BoxGeometry(3, 3, 3, 4, 4, 4),
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    )
    caster.position.y = 1.5
    caster.castShadow = castShadow
    caster.customDepthMaterial = new THREE.MeshDepthMaterial({ wireframe })
    scene.add(caster)

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

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    const mean = meanRgba(renderRgba(scene, camera, { width: 96, height: 96 }))
    return mean.r + mean.g + mean.b
  }

  function renderPointCustomDistanceShadow(wireframe, castShadow = true) {
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
      new THREE.BoxGeometry(2.5, 2.5, 2.5, 4, 4, 4),
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    )
    caster.position.y = 1.25
    caster.castShadow = castShadow
    const customDistanceMaterial = new THREE.MeshDistanceMaterial()
    customDistanceMaterial.wireframe = wireframe
    caster.customDistanceMaterial = customDistanceMaterial
    scene.add(caster)

    const light = new THREE.PointLight(0xffffff, 2)
    light.position.set(0, 5, 4)
    light.distance = 12
    light.castShadow = true
    light.shadow.mapSize.set(256, 256)
    light.shadow.camera.near = 0.1
    light.shadow.camera.far = 12
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    const mean = meanRgba(renderRgba(scene, camera, { width: 96, height: 96 }))
    return mean.r + mean.g + mean.b
  }

  const filledDepthLum = renderDirectionalCustomDepthShadow(false)
  const wireframeDepthLum = renderDirectionalCustomDepthShadow(true)
  const unshadowedDepthLum = renderDirectionalCustomDepthShadow(true, false)
  assert.ok(
    wireframeDepthLum > filledDepthLum + 20,
    `customDepthMaterial wireframe shadow should be lighter than filled caster shadow (${wireframeDepthLum} vs ${filledDepthLum})`,
  )
  assert.ok(
    wireframeDepthLum < unshadowedDepthLum - 2,
    `customDepthMaterial wireframe shadow should still cast visible shadow (${wireframeDepthLum} vs ${unshadowedDepthLum})`,
  )

  const filledDistanceLum = renderPointCustomDistanceShadow(false)
  const wireframeDistanceLum = renderPointCustomDistanceShadow(true)
  const unshadowedDistanceLum = renderPointCustomDistanceShadow(true, false)
  assert.ok(
    wireframeDistanceLum > filledDistanceLum + 10,
    `customDistanceMaterial wireframe shadow should be lighter than filled caster shadow (${wireframeDistanceLum} vs ${filledDistanceLum})`,
  )
  assert.ok(
    wireframeDistanceLum < unshadowedDistanceLum - 5,
    `customDistanceMaterial wireframe shadow should still cast visible shadow (${wireframeDistanceLum} vs ${unshadowedDistanceLum})`,
  )
})

test('source material wireframe applies to custom shadow material casters', () => {
  function sourceMaterial(wireframe) {
    const material = new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe })
    material.colorWrite = false
    material.depthWrite = false
    return material
  }

  function renderDirectionalCustomDepthShadow(wireframe, castShadow = true) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const caster = new THREE.Mesh(new THREE.BoxGeometry(3, 3, 3), sourceMaterial(wireframe))
    caster.position.y = 1.5
    caster.castShadow = castShadow
    caster.customDepthMaterial = new THREE.MeshDepthMaterial()
    scene.add(caster)

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

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    const mean = meanRgba(renderRgba(scene, camera, { width: 96, height: 96 }))
    return mean.r + mean.g + mean.b
  }

  function renderPointCustomDistanceShadow(wireframe, castShadow = true) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const caster = new THREE.Mesh(new THREE.BoxGeometry(2.5, 2.5, 2.5), sourceMaterial(wireframe))
    caster.position.y = 1.25
    caster.castShadow = castShadow
    caster.customDistanceMaterial = new THREE.MeshDistanceMaterial()
    scene.add(caster)

    const light = new THREE.PointLight(0xffffff, 2)
    light.position.set(0, 5, 4)
    light.distance = 12
    light.castShadow = true
    light.shadow.mapSize.set(256, 256)
    light.shadow.camera.near = 0.1
    light.shadow.camera.far = 12
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    const mean = meanRgba(renderRgba(scene, camera, { width: 96, height: 96 }))
    return mean.r + mean.g + mean.b
  }

  const filledDepthLum = renderDirectionalCustomDepthShadow(false)
  const wireframeDepthLum = renderDirectionalCustomDepthShadow(true)
  const unshadowedDepthLum = renderDirectionalCustomDepthShadow(false, false)
  assert.ok(
    wireframeDepthLum > filledDepthLum + 20,
    `source material wireframe should lighten the customDepthMaterial caster shadow (${wireframeDepthLum} vs ${filledDepthLum})`,
  )
  assert.ok(
    wireframeDepthLum < unshadowedDepthLum - 2,
    `source material wireframe customDepthMaterial shadow should still be visible (${wireframeDepthLum} vs ${unshadowedDepthLum})`,
  )

  const filledDistanceLum = renderPointCustomDistanceShadow(false)
  const wireframeDistanceLum = renderPointCustomDistanceShadow(true)
  const unshadowedDistanceLum = renderPointCustomDistanceShadow(false, false)
  assert.ok(
    wireframeDistanceLum > filledDistanceLum + 10,
    `source material wireframe should lighten the customDistanceMaterial caster shadow (${wireframeDistanceLum} vs ${filledDistanceLum})`,
  )
  assert.ok(
    wireframeDistanceLum < unshadowedDistanceLum - 5,
    `source material wireframe customDistanceMaterial shadow should still be visible (${wireframeDistanceLum} vs ${unshadowedDistanceLum})`,
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

test('base color map samples texture channels 2 and 3 from matching UV attributes', () => {
  function renderWithChannel(channel) {
    const map = rgbaTexture([
      0, 255, 0, 255,
      255, 0, 0, 255,
    ], 2, 1)
    map.channel = channel

    const geometry = constantUvPlane(0.25, 0.5)
    setConstantUvAttribute(geometry, 'uv1', 0.25, 0.5)
    setConstantUvAttribute(geometry, 'uv2', channel === 2 ? 0.75 : 0.25, 0.5)
    setConstantUvAttribute(geometry, 'uv3', channel === 3 ? 0.75 : 0.25, 0.5)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ map })))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const uv2 = renderWithChannel(2)
  const uv3 = renderWithChannel(3)
  assert.ok(uv2.r > uv2.g + 40, `map channel=2 should sample the uv2 red texel (${uv2.r} vs ${uv2.g})`)
  assert.ok(uv3.r > uv3.g + 40, `map channel=3 should sample the uv3 red texel (${uv3.r} vs ${uv3.g})`)
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

test('emissiveMap honors explicit texture matrices', () => {
  const emissiveMap = rgbaTexture([
    0, 0, 0, 255,
    255, 0, 0, 255,
  ], 2, 1)
  emissiveMap.magFilter = THREE.NearestFilter
  emissiveMap.minFilter = THREE.NearestFilter
  setTextureMatrixOffset(emissiveMap, 0.5)

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
  assert.ok(mean.r > mean.g + 40, `explicit emissiveMap matrix should sample the red texel (${mean.r} vs ${mean.g})`)
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

test('emissiveMap honors horizontal and vertical repeat wrapping', () => {
  function renderWithWrapping({ wrapS, wrapT, vertical = false }) {
    const emissiveMap = vertical
      ? rgbaTexture([
        0, 255, 0, 255,
        255, 0, 0, 255,
      ], 1, 2)
      : rgbaTexture([
        0, 255, 0, 255,
        255, 0, 0, 255,
      ], 2, 1)
    if (wrapS != null) emissiveMap.wrapS = wrapS
    if (wrapT != null) emissiveMap.wrapT = wrapT
    emissiveMap.magFilter = THREE.NearestFilter
    emissiveMap.minFilter = THREE.NearestFilter

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(vertical ? 0.5 : 1.25, vertical ? 1.25 : 0.5),
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

  const clamped = renderWithWrapping({ wrapS: THREE.ClampToEdgeWrapping })
  const repeated = renderWithWrapping({ wrapS: THREE.RepeatWrapping })
  const mirrored = renderWithWrapping({ wrapS: THREE.MirroredRepeatWrapping })
  assert.ok(clamped.r > clamped.g + 80, `clamped emissiveMap U coordinates should sample the red edge texel (${clamped.r} vs ${clamped.g})`)
  assert.ok(repeated.g > repeated.r + 30, `repeated emissiveMap U coordinates should wrap to the green texel (${repeated.g} vs ${repeated.r})`)
  assert.ok(mirrored.r > mirrored.g + 80, `mirrored emissiveMap U coordinates should reflect to the red texel (${mirrored.r} vs ${mirrored.g})`)

  const clampedVertical = renderWithWrapping({ wrapT: THREE.ClampToEdgeWrapping, vertical: true })
  const repeatedVertical = renderWithWrapping({ wrapT: THREE.RepeatWrapping, vertical: true })
  const mirroredVertical = renderWithWrapping({ wrapT: THREE.MirroredRepeatWrapping, vertical: true })
  assert.ok(clampedVertical.r > clampedVertical.g + 80, `clamped emissiveMap V coordinates should sample the red edge texel (${clampedVertical.r} vs ${clampedVertical.g})`)
  assert.ok(repeatedVertical.g > repeatedVertical.r + 30, `repeated emissiveMap V coordinates should wrap to the green texel (${repeatedVertical.g} vs ${repeatedVertical.r})`)
  assert.ok(mirroredVertical.r > mirroredVertical.g + 80, `mirrored emissiveMap V coordinates should reflect to the red texel (${mirroredVertical.r} vs ${mirroredVertical.g})`)
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

test('metallicRoughness maps honor explicit texture matrices', () => {
  function renderWithMatrix(matrixOffsetX) {
    const roughnessMap = rgbaTexture([
      0, 255, 0, 255,
      0, 0, 0, 255,
    ], 2, 1)
    roughnessMap.magFilter = THREE.NearestFilter
    roughnessMap.minFilter = THREE.NearestFilter
    if (matrixOffsetX !== 0) setTextureMatrixOffset(roughnessMap, matrixOffsetX)

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

  const rough = maxLuminance(renderWithMatrix(0))
  const smooth = maxLuminance(renderWithMatrix(0.5))
  assert.ok(Math.abs(smooth - rough) > 20, `explicit roughnessMap matrix should change the sampled texel (${smooth} vs ${rough})`)
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

test('metallicRoughness maps honor horizontal and vertical repeat wrapping', () => {
  const renderer = new Renderer()

  function renderWithWrapping({ wrapS = THREE.ClampToEdgeWrapping, wrapT = THREE.ClampToEdgeWrapping, vertical = false }) {
    const roughnessMap = rgbaTexture([
      0, 255, 0, 255,
      0, 0, 0, 255,
      0, 0, 0, 255,
      0, 0, 0, 255,
    ], 2, 2)
    roughnessMap.wrapS = wrapS
    roughnessMap.wrapT = wrapT
    roughnessMap.magFilter = THREE.NearestFilter
    roughnessMap.minFilter = THREE.NearestFilter

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(vertical ? 0.25 : 1.25, vertical ? 1.25 : 0.25),
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

    return maxLuminance(renderer.render(scene, camera, { width: 64, height: 64, format: 'rgba' }))
  }

  const clamped = renderWithWrapping({ wrapS: THREE.ClampToEdgeWrapping })
  const repeated = renderWithWrapping({ wrapS: THREE.RepeatWrapping })
  const mirrored = renderWithWrapping({ wrapS: THREE.MirroredRepeatWrapping })

  const clampedVertical = renderWithWrapping({ wrapT: THREE.ClampToEdgeWrapping, vertical: true })
  const repeatedVertical = renderWithWrapping({ wrapT: THREE.RepeatWrapping, vertical: true })
  const mirroredVertical = renderWithWrapping({ wrapT: THREE.MirroredRepeatWrapping, vertical: true })
  assert.ok(clamped < 10, `clamped roughnessMap U coordinates should sample the smooth edge texel (${clamped})`)
  assert.ok(repeated > clamped + 20, `repeated roughnessMap U coordinates should wrap to the rough texel (${repeated} vs ${clamped})`)
  assert.ok(mirrored < 10, `mirrored roughnessMap U coordinates should reflect to the smooth texel (${mirrored})`)
  assert.ok(repeated > mirrored + 20, `mirrored roughnessMap U coordinates should differ from RepeatWrapping (${mirrored} vs ${repeated})`)
  assert.ok(clampedVertical < 10, `clamped roughnessMap V coordinates should sample the smooth edge texel (${clampedVertical})`)
  assert.ok(repeatedVertical > clampedVertical + 20, `repeated roughnessMap V coordinates should wrap to the rough texel (${repeatedVertical} vs ${clampedVertical})`)
  assert.ok(mirroredVertical < 10, `mirrored roughnessMap V coordinates should reflect to the smooth texel (${mirroredVertical})`)
  assert.ok(repeatedVertical > mirroredVertical + 20, `mirrored roughnessMap V coordinates should differ from RepeatWrapping (${mirroredVertical} vs ${repeatedVertical})`)
})

test('metallicRoughness metalnessMap honors horizontal and vertical repeat wrapping', () => {
  const renderer = new Renderer()

  function renderWithWrapping({ wrapS = THREE.ClampToEdgeWrapping, wrapT = THREE.ClampToEdgeWrapping, vertical = false }) {
    const metalnessMap = rgbaTexture([
      0, 0, 255, 255,
      0, 0, 0, 255,
      0, 0, 0, 255,
      0, 0, 0, 255,
    ], 2, 2)
    metalnessMap.wrapS = wrapS
    metalnessMap.wrapT = wrapT
    metalnessMap.magFilter = THREE.NearestFilter
    metalnessMap.minFilter = THREE.NearestFilter

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(vertical ? 0.25 : 1.25, vertical ? 1.25 : 0.25),
      new THREE.MeshStandardMaterial({
        color: 0xffffff,
        roughness: 1,
        metalness: 1,
        metalnessMap,
      }),
    ))

    const light = new THREE.DirectionalLight(0xffffff, 12)
    light.position.set(0, 0, 3)
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return maxLuminance(renderer.render(scene, camera, { width: 64, height: 64, format: 'rgba' }))
  }

  const clamped = renderWithWrapping({ wrapS: THREE.ClampToEdgeWrapping })
  const repeated = renderWithWrapping({ wrapS: THREE.RepeatWrapping })
  const mirrored = renderWithWrapping({ wrapS: THREE.MirroredRepeatWrapping })
  assert.ok(clamped > 180, `clamped metalnessMap U coordinates should sample the non-metal edge texel (${clamped})`)
  assert.ok(repeated < clamped - 80, `repeated metalnessMap U coordinates should wrap to the metallic texel (${repeated} vs ${clamped})`)
  assert.ok(mirrored > 180, `mirrored metalnessMap U coordinates should reflect to the non-metal texel (${mirrored})`)
  assert.ok(repeated < mirrored - 80, `mirrored metalnessMap U coordinates should differ from RepeatWrapping (${mirrored} vs ${repeated})`)

  const clampedVertical = renderWithWrapping({ wrapT: THREE.ClampToEdgeWrapping, vertical: true })
  const repeatedVertical = renderWithWrapping({ wrapT: THREE.RepeatWrapping, vertical: true })
  const mirroredVertical = renderWithWrapping({ wrapT: THREE.MirroredRepeatWrapping, vertical: true })
  assert.ok(clampedVertical > 180, `clamped metalnessMap V coordinates should sample the non-metal edge texel (${clampedVertical})`)
  assert.ok(repeatedVertical < clampedVertical - 80, `repeated metalnessMap V coordinates should wrap to the metallic texel (${repeatedVertical} vs ${clampedVertical})`)
  assert.ok(mirroredVertical > 180, `mirrored metalnessMap V coordinates should reflect to the non-metal texel (${mirroredVertical})`)
  assert.ok(repeatedVertical < mirroredVertical - 80, `mirrored metalnessMap V coordinates should differ from RepeatWrapping (${mirroredVertical} vs ${repeatedVertical})`)
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

test('base color maps honor explicit texture matrices', () => {
  const map = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter
  setTextureMatrixOffset(map, 0.5)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    constantUvPlane(0.25, 0.5),
    new THREE.MeshBasicMaterial({ color: 0xffffff, map }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 22, 22, 42, 42)
  assert.ok(mean.g > mean.r + 80, `explicit texture matrix should shift the base map to green (${mean.g} vs ${mean.r})`)
})

test('base color maps honor horizontal and vertical repeat wrapping', () => {
  const renderer = new Renderer()

  function renderWithWrapping({ wrapS = THREE.ClampToEdgeWrapping, wrapT = THREE.ClampToEdgeWrapping, vertical = false }) {
    const map = vertical
      ? rgbaTexture([
        0, 255, 0, 255,
        0, 255, 0, 255,
        255, 0, 0, 255,
        255, 0, 0, 255,
      ], 2, 2)
      : rgbaTexture([
        0, 255, 0, 255,
        255, 0, 0, 255,
        0, 255, 0, 255,
        255, 0, 0, 255,
      ], 2, 2)
    map.wrapS = wrapS
    map.wrapT = wrapT
    map.magFilter = THREE.NearestFilter
    map.minFilter = THREE.NearestFilter

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(vertical ? 0.25 : 1.25, vertical ? 1.25 : 0.25),
      new THREE.MeshBasicMaterial({ map }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderer.render(scene, camera, { width: 64, height: 64, format: 'rgba' }), 64, 64, 22, 22, 42, 42)
  }

  const clamped = renderWithWrapping({ wrapS: THREE.ClampToEdgeWrapping })
  const repeated = renderWithWrapping({ wrapS: THREE.RepeatWrapping })
  const mirrored = renderWithWrapping({ wrapS: THREE.MirroredRepeatWrapping })
  assert.ok(clamped.r > clamped.g + 80, `clamped base map U coordinates should sample the red edge texel (${clamped.r} vs ${clamped.g})`)
  assert.ok(repeated.g > repeated.r + 80, `repeated base map U coordinates should wrap to the green texel (${repeated.g} vs ${repeated.r})`)
  assert.ok(mirrored.r > mirrored.g + 80, `mirrored base map U coordinates should reflect to the red texel (${mirrored.r} vs ${mirrored.g})`)

  const clampedVertical = renderWithWrapping({ wrapT: THREE.ClampToEdgeWrapping, vertical: true })
  const repeatedVertical = renderWithWrapping({ wrapT: THREE.RepeatWrapping, vertical: true })
  const mirroredVertical = renderWithWrapping({ wrapT: THREE.MirroredRepeatWrapping, vertical: true })
  assert.ok(clampedVertical.r > clampedVertical.g + 80, `clamped base map V coordinates should sample the red edge texel (${clampedVertical.r} vs ${clampedVertical.g})`)
  assert.ok(repeatedVertical.g > repeatedVertical.r + 80, `repeated base map V coordinates should wrap to the green texel (${repeatedVertical.g} vs ${repeatedVertical.r})`)
  assert.ok(mirroredVertical.r > mirroredVertical.g + 80, `mirrored base map V coordinates should reflect to the red texel (${mirroredVertical.r} vs ${mirroredVertical.g})`)
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

test('base color maps generate mip chains for mipmap min filters', () => {
  function checkerTexture(generateMipmaps) {
    const size = 16
    const data = []
    for (let y = 0; y < size; y += 1) {
      for (let x = 0; x < size; x += 1) {
        const value = (x + y) % 2 === 0 ? 0 : 255
        data.push(value, value, value, 255)
      }
    }
    const map = rgbaTexture(data, size, size)
    map.wrapS = THREE.RepeatWrapping
    map.wrapT = THREE.RepeatWrapping
    map.repeat.set(128, 128)
    map.magFilter = THREE.NearestFilter
    map.minFilter = THREE.NearestMipmapNearestFilter
    map.generateMipmaps = generateMipmaps
    return map
  }

  function renderChecker(generateMipmaps) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ map: checkerTexture(generateMipmaps) }),
    ))
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10)
    camera.position.set(0, 0, 3)
    return renderRgba(scene, camera, { width: 64, height: 64, outputColorSpace: THREE.LinearSRGBColorSpace })
  }

  const mipmapped = renderChecker(true)
  const baseOnly = renderChecker(false)
  const gray = (r, g, b) => r > 96 && r < 160 && Math.abs(r - g) < 3 && Math.abs(r - b) < 3
  const mipmappedGray = countRegionPixels(mipmapped, 64, 64, 4, 4, 60, 60, gray)
  const baseOnlyGray = countRegionPixels(baseOnly, 64, 64, 4, 4, 60, 60, gray)

  assert.ok(mipmappedGray > 2400, `generated mip chain should minify the checker to gray (${mipmappedGray})`)
  assert.ok(
    mipmappedGray > baseOnlyGray + 1800,
    `mipmapped texture should have far more gray pixels than base-only sampling (${mipmappedGray} vs ${baseOnlyGray})`,
  )
})

test('invalid texture mipmap boolean values fail clearly', () => {
  const map = solidTexture(255, 255, 255)
  map.minFilter = THREE.NearestMipmapNearestFilter
  map.generateMipmaps = 'yes'

  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map })))

  assert.throws(
    () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
    /texture\.generateMipmaps must be a boolean/i,
  )
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

test('texture anisotropy inputs render with native samplers', () => {
  const map = solidTexture(255, 255, 255)
  map.anisotropy = 4

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  assert.ok(mean.r > 120 && mean.g > 120 && mean.b > 120, `anisotropic mapped plane should render visibly (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('invalid texture anisotropy values fail clearly', () => {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const cases = [
    ['material map', () => {
      const map = solidTexture(255, 255, 255)
      map.anisotropy = 'high'
      const scene = new THREE.Scene()
      scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map })))
      return scene
    }, /material\.map\.anisotropy must be a finite number/i],
    ['background texture', () => {
      const background = solidTexture(0, 0, 255)
      background.anisotropy = Number.POSITIVE_INFINITY
      const scene = new THREE.Scene()
      scene.background = background
      return scene
    }, /background\.anisotropy must be a finite number/i],
    ['packed physical map', () => {
      const clearcoatMap = solidTexture(255, 255, 255)
      clearcoatMap.anisotropy = Number.NaN
      const scene = new THREE.Scene()
      const material = new THREE.MeshPhysicalMaterial({ color: 0xffffff, clearcoat: 1 })
      material.clearcoatMap = clearcoatMap
      scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))
      return scene
    }, /material\.clearcoatMap\.anisotropy must be a finite number/i],
  ]

  for (const [name, makeScene, pattern] of cases) {
    assert.throws(
      () => renderRgba(makeScene(), camera, { width: 64, height: 64 }),
      pattern,
      `${name} should fail clearly`,
    )
  }
})

test('invalid texture transform values fail clearly', () => {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const cases = [
    ['material map offset', () => {
      const map = solidTexture(255, 255, 255)
      map.offset.x = 'left'
      const scene = new THREE.Scene()
      scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map })))
      return scene
    }, /material\.map\.offset\.x must be a finite number/i],
    ['material map offset container', () => {
      const map = solidTexture(255, 255, 255)
      map.offset = 'left'
      const scene = new THREE.Scene()
      scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map })))
      return scene
    }, /material\.map\.offset must be a vector-like object/i],
    ['normalMap matrix', () => {
      const normalMap = solidTexture(128, 128, 255)
      normalMap.matrixAutoUpdate = false
      normalMap.matrix.elements[0] = Number.NaN
      const scene = new THREE.Scene()
      scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshStandardMaterial({ normalMap })))
      scene.add(new THREE.AmbientLight(0xffffff, 1))
      return scene
    }, /material\.normalMap\.matrix\.elements\[0\] must be a finite number/i],
    ['material map flipY', () => {
      const map = solidTexture(255, 255, 255)
      map.flipY = 'no'
      const scene = new THREE.Scene()
      scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map })))
      return scene
    }, /material\.map\.flipY must be a boolean/i],
    ['material map matrixAutoUpdate', () => {
      const map = solidTexture(255, 255, 255)
      map.matrixAutoUpdate = 'no'
      const scene = new THREE.Scene()
      scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map })))
      return scene
    }, /material\.map\.matrixAutoUpdate must be a boolean/i],
    ['background rotation', () => {
      const background = solidTexture(0, 0, 255)
      background.rotation = Number.POSITIVE_INFINITY
      const scene = new THREE.Scene()
      scene.background = background
      return scene
    }, /background\.rotation must be a finite number/i],
    ['physical extension repeat', () => {
      const clearcoatMap = solidTexture(255, 255, 255)
      clearcoatMap.repeat.y = 'tall'
      const material = new THREE.MeshPhysicalMaterial({ color: 0xffffff, clearcoat: 1 })
      material.clearcoatMap = clearcoatMap
      const scene = new THREE.Scene()
      scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))
      return scene
    }, /material\.clearcoatMap\.repeat\.y must be a finite number/i],
  ]

  for (const [label, makeScene, pattern] of cases) {
    assert.throws(
      () => renderRgba(makeScene(), camera, { width: 64, height: 64 }),
      pattern,
      label,
    )
  }
})

test('one- and two-channel raw DataTexture maps and backgrounds expand for texture rendering', () => {
  function renderMap(map) {
    map.needsUpdate = true
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map })))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    const rgba = renderRgba(scene, camera, { width: 64, height: 64, outputColorSpace: THREE.LinearSRGBColorSpace })
    return meanRegion(rgba, 64, 64, 24, 24, 40, 40)
  }

  function renderBackground(background) {
    background.needsUpdate = true
    const scene = new THREE.Scene()
    scene.background = background

    const rgba = renderRgba(scene, makeCamera(), { width: 64, height: 64, outputColorSpace: THREE.LinearSRGBColorSpace })
    return meanRgba(rgba)
  }

  const redMap = new THREE.DataTexture(new Uint8Array([220]), 1, 1, THREE.RedFormat)
  const red = renderMap(redMap)
  assert.ok(red.r > 180 && red.g > 180 && red.b > 180, `one-channel raw texture should expand to grayscale (${red.r}, ${red.g}, ${red.b})`)

  const redBackground = renderBackground(new THREE.DataTexture(new Uint8Array([220]), 1, 1, THREE.RedFormat))
  assert.ok(
    redBackground.r > 180 && redBackground.g > 180 && redBackground.b > 180,
    `one-channel raw background should expand to grayscale (${redBackground.r}, ${redBackground.g}, ${redBackground.b})`,
  )

  const rgMap = new THREE.DataTexture(new Uint8Array([230, 24]), 1, 1, THREE.RGFormat)
  const rg = renderMap(rgMap)
  assert.ok(rg.r > 190, `two-channel raw texture should preserve red (${rg.r})`)
  assert.ok(rg.g < 80, `two-channel raw texture should preserve green (${rg.g})`)
  assert.ok(rg.b < 40, `two-channel raw texture should leave blue empty (${rg.b})`)

  const rgBackground = renderBackground(new THREE.DataTexture(new Uint8Array([230, 24]), 1, 1, THREE.RGFormat))
  assert.ok(rgBackground.r > 190, `two-channel raw background should preserve red (${rgBackground.r})`)
  assert.ok(rgBackground.g < 80, `two-channel raw background should preserve green (${rgBackground.g})`)
  assert.ok(rgBackground.b < 40, `two-channel raw background should leave blue empty (${rgBackground.b})`)
})

test('HalfFloatType raw DataTexture maps decode for material and background textures', () => {
  function halfRgbaTexture() {
    const texture = new THREE.DataTexture(
      new Uint16Array([0x3800, 0x3400, 0x3c00, 0x3c00]),
      1,
      1,
      THREE.RGBAFormat,
      THREE.HalfFloatType,
    )
    texture.needsUpdate = true
    return texture
  }

  function renderTexture(kind) {
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    if (kind === 'material') {
      scene.background = new THREE.Color(0, 0, 0)
      scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map: halfRgbaTexture() })))
    } else {
      scene.background = halfRgbaTexture()
    }
    return meanRegion(
      renderRgba(scene, camera, { width: 64, height: 64, outputColorSpace: THREE.LinearSRGBColorSpace }),
      64,
      64,
      24,
      24,
      40,
      40,
    )
  }

  for (const kind of ['material', 'background']) {
    const mean = renderTexture(kind)
    assert.ok(mean.r > 105 && mean.r < 150, `${kind} half-float red should decode near 0.5 (${mean.r})`)
    assert.ok(mean.g > 45 && mean.g < 100, `${kind} half-float green should decode near 0.25 (${mean.g})`)
    assert.ok(mean.b > 180, `${kind} half-float blue should decode near 1.0 (${mean.b})`)
  }
})

test('FloatType raw DataTexture maps decode for material and background textures', () => {
  function floatRgbaTexture() {
    const texture = new THREE.DataTexture(
      new Float32Array([0.5, 0.25, 1, 1]),
      1,
      1,
      THREE.RGBAFormat,
      THREE.FloatType,
    )
    texture.needsUpdate = true
    return texture
  }

  function renderTexture(kind) {
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    if (kind === 'material') {
      scene.background = new THREE.Color(0, 0, 0)
      scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map: floatRgbaTexture() })))
    } else {
      scene.background = floatRgbaTexture()
    }
    return meanRegion(
      renderRgba(scene, camera, { width: 64, height: 64, outputColorSpace: THREE.LinearSRGBColorSpace }),
      64,
      64,
      24,
      24,
      40,
      40,
    )
  }

  for (const kind of ['material', 'background']) {
    const mean = renderTexture(kind)
    assert.ok(mean.r > 105 && mean.r < 150, `${kind} float red should decode near 0.5 (${mean.r})`)
    assert.ok(mean.g > 45 && mean.g < 100, `${kind} float green should decode near 0.25 (${mean.g})`)
    assert.ok(mean.b > 180, `${kind} float blue should decode near 1.0 (${mean.b})`)
  }
})

test('normalized unsigned integer raw DataTexture maps decode for material and background textures', () => {
  const cases = [
    [
      'UnsignedShortType',
      THREE.UnsignedShortType,
      () => new Uint16Array([0x8000, 0x4000, 0xffff, 0xffff]),
    ],
    [
      'UnsignedIntType',
      THREE.UnsignedIntType,
      () => new Uint32Array([0x80000000, 0x40000000, 0xffffffff, 0xffffffff]),
    ],
  ]

  function rgbaTexture(type, data) {
    const texture = new THREE.DataTexture(data, 1, 1, THREE.RGBAFormat, type)
    texture.needsUpdate = true
    return texture
  }

  function renderTexture(kind, type, data) {
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    if (kind === 'material') {
      scene.background = new THREE.Color(0, 0, 0)
      scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map: rgbaTexture(type, data) })))
    } else {
      scene.background = rgbaTexture(type, data)
    }
    return meanRegion(
      renderRgba(scene, camera, { width: 64, height: 64, outputColorSpace: THREE.LinearSRGBColorSpace }),
      64,
      64,
      24,
      24,
      40,
      40,
    )
  }

  for (const [name, type, makeData] of cases) {
    for (const kind of ['material', 'background']) {
      const mean = renderTexture(kind, type, makeData())
      assert.ok(mean.r > 105 && mean.r < 150, `${kind} ${name} red should normalize near 0.5 (${mean.r})`)
      assert.ok(mean.g > 45 && mean.g < 100, `${kind} ${name} green should normalize near 0.25 (${mean.g})`)
      assert.ok(mean.b > 180, `${kind} ${name} blue should normalize near 1.0 (${mean.b})`)
    }
  }
})

test('normalized signed integer raw DataTexture maps decode for material and background textures', () => {
  const cases = [
    [
      'ByteType',
      THREE.ByteType,
      () => new Int8Array([64, 32, 127, 127]),
    ],
    [
      'ShortType',
      THREE.ShortType,
      () => new Int16Array([0x4000, 0x2000, 0x7fff, 0x7fff]),
    ],
    [
      'IntType',
      THREE.IntType,
      () => new Int32Array([0x40000000, 0x20000000, 0x7fffffff, 0x7fffffff]),
    ],
  ]

  function rgbaTexture(type, data) {
    const texture = new THREE.DataTexture(data, 1, 1, THREE.RGBAFormat, type)
    texture.needsUpdate = true
    return texture
  }

  function renderTexture(kind, type, data) {
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    if (kind === 'material') {
      scene.background = new THREE.Color(0, 0, 0)
      scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map: rgbaTexture(type, data) })))
    } else {
      scene.background = rgbaTexture(type, data)
    }
    return meanRegion(
      renderRgba(scene, camera, { width: 64, height: 64, outputColorSpace: THREE.LinearSRGBColorSpace }),
      64,
      64,
      24,
      24,
      40,
      40,
    )
  }

  for (const [name, type, makeData] of cases) {
    for (const kind of ['material', 'background']) {
      const mean = renderTexture(kind, type, makeData())
      assert.ok(mean.r > 105 && mean.r < 155, `${kind} ${name} red should normalize near 0.5 (${mean.r})`)
      assert.ok(mean.g > 45 && mean.g < 100, `${kind} ${name} green should normalize near 0.25 (${mean.g})`)
      assert.ok(mean.b > 180, `${kind} ${name} blue should normalize near 1.0 (${mean.b})`)
    }
  }
})

test('normalized unsigned integer raw environment textures decode for IBL', () => {
  function byteEnvironmentTexture() {
    const texture = solidTexture(128, 64, 255)
    texture.mapping = THREE.EquirectangularReflectionMapping
    return texture
  }

  function unsignedShortEnvironmentTexture() {
    const texture = new THREE.DataTexture(
      new Uint16Array([0x8080, 0x4040, 0xffff, 0xffff]),
      1,
      1,
      THREE.RGBAFormat,
      THREE.UnsignedShortType,
    )
    texture.mapping = THREE.EquirectangularReflectionMapping
    texture.needsUpdate = true
    return texture
  }

  function renderEnvironment(kind, texture) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const material = new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0.25 })
    if (kind === 'scene') {
      scene.environment = texture
      scene.environmentIntensity = 2.5
    } else if (kind === 'reflectionProbe') {
      scene.userData.headlessThreeRenderer = {
        reflectionProbe: { texture, intensity: 2.5 },
      }
    } else {
      material.envMap = texture
      material.envMapIntensity = 2.5
    }
    scene.add(new THREE.Mesh(
      new THREE.SphereGeometry(1, 32, 16),
      material,
    ))
    return renderRgba(scene, makeCamera(), {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    })
  }

  for (const kind of ['scene', 'reflectionProbe', 'materialEnvMap']) {
    const byteRender = renderEnvironment(kind, byteEnvironmentTexture())
    const unsignedRender = renderEnvironment(kind, unsignedShortEnvironmentTexture())
    const diff = meanAbsDiff(byteRender, unsignedRender)
    assert.ok(diff < 2, `${kind} unsigned integer environment should match equivalent RGBA8 IBL (diff=${diff.toFixed(3)})`)
  }
})

test('normalized signed integer raw environment textures decode for IBL', () => {
  function byteEnvironmentTexture() {
    const texture = solidTexture(129, 64, 255)
    texture.mapping = THREE.EquirectangularReflectionMapping
    return texture
  }

  function signedShortEnvironmentTexture() {
    const texture = new THREE.DataTexture(
      new Int16Array([0x4000, 0x2000, 0x7fff, 0x7fff]),
      1,
      1,
      THREE.RGBAFormat,
      THREE.ShortType,
    )
    texture.mapping = THREE.EquirectangularReflectionMapping
    texture.needsUpdate = true
    return texture
  }

  function renderEnvironment(kind, texture) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const material = new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0.25 })
    if (kind === 'scene') {
      scene.environment = texture
      scene.environmentIntensity = 2.5
    } else if (kind === 'reflectionProbe') {
      scene.userData.headlessThreeRenderer = {
        reflectionProbe: { texture, intensity: 2.5 },
      }
    } else {
      material.envMap = texture
      material.envMapIntensity = 2.5
    }
    scene.add(new THREE.Mesh(
      new THREE.SphereGeometry(1, 32, 16),
      material,
    ))
    return renderRgba(scene, makeCamera(), {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    })
  }

  for (const kind of ['scene', 'reflectionProbe', 'materialEnvMap']) {
    const byteRender = renderEnvironment(kind, byteEnvironmentTexture())
    const signedRender = renderEnvironment(kind, signedShortEnvironmentTexture())
    const diff = meanAbsDiff(byteRender, signedRender)
    assert.ok(diff < 2, `${kind} signed integer environment should match equivalent RGBA8 IBL (diff=${diff.toFixed(3)})`)
  }
})

test('float raw environment textures decode for IBL', () => {
  function byteEnvironmentTexture() {
    const texture = solidTexture(128, 64, 255)
    texture.colorSpace = THREE.LinearSRGBColorSpace
    texture.mapping = THREE.EquirectangularReflectionMapping
    return texture
  }

  function floatEnvironmentTexture() {
    const texture = new THREE.DataTexture(
      new Float32Array([0.5, 0.25, 1, 1]),
      1,
      1,
      THREE.RGBAFormat,
      THREE.FloatType,
    )
    texture.colorSpace = THREE.LinearSRGBColorSpace
    texture.mapping = THREE.EquirectangularReflectionMapping
    texture.needsUpdate = true
    return texture
  }

  function halfFloatEnvironmentTexture() {
    const texture = new THREE.DataTexture(
      new Uint16Array([0x3800, 0x3400, 0x3c00, 0x3c00]),
      1,
      1,
      THREE.RGBAFormat,
      THREE.HalfFloatType,
    )
    texture.colorSpace = THREE.LinearSRGBColorSpace
    texture.mapping = THREE.EquirectangularReflectionMapping
    texture.needsUpdate = true
    return texture
  }

  function renderEnvironment(kind, texture) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const material = new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0.25 })
    if (kind === 'scene') {
      scene.environment = texture
      scene.environmentIntensity = 2.5
    } else if (kind === 'reflectionProbe') {
      scene.userData.headlessThreeRenderer = {
        reflectionProbe: { texture, intensity: 2.5 },
      }
    } else {
      material.envMap = texture
      material.envMapIntensity = 2.5
    }
    scene.add(new THREE.Mesh(
      new THREE.SphereGeometry(1, 32, 16),
      material,
    ))
    return renderRgba(scene, makeCamera(), {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    })
  }

  for (const kind of ['scene', 'reflectionProbe', 'materialEnvMap']) {
    const byteRender = renderEnvironment(kind, byteEnvironmentTexture())
    for (const [label, makeTexture] of [
      ['FloatType', floatEnvironmentTexture],
      ['HalfFloatType', halfFloatEnvironmentTexture],
    ]) {
      const floatRender = renderEnvironment(kind, makeTexture())
      const diff = meanAbsDiff(byteRender, floatRender)
      assert.ok(diff < 3, `${kind} ${label} environment should match equivalent linear RGBA8 IBL (diff=${diff.toFixed(3)})`)
    }
  }
})

test('packed raw environment textures unpack for IBL', () => {
  const cases = [
    ['UnsignedShort4444Type', THREE.UnsignedShort4444Type, 0x84ff, [136, 68, 255]],
    ['UnsignedShort5551Type', THREE.UnsignedShort5551Type, 0x823f, [132, 66, 255]],
  ]

  function byteEnvironmentTexture([r, g, b]) {
    const texture = solidTexture(r, g, b)
    texture.colorSpace = THREE.LinearSRGBColorSpace
    texture.mapping = THREE.EquirectangularReflectionMapping
    return texture
  }

  function packedEnvironmentTexture(type, value) {
    const texture = new THREE.DataTexture(new Uint16Array([value]), 1, 1, THREE.RGBAFormat, type)
    texture.colorSpace = THREE.LinearSRGBColorSpace
    texture.mapping = THREE.EquirectangularReflectionMapping
    texture.needsUpdate = true
    return texture
  }

  function renderEnvironment(kind, texture) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const material = new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0.25 })
    if (kind === 'scene') {
      scene.environment = texture
      scene.environmentIntensity = 2.5
    } else if (kind === 'reflectionProbe') {
      scene.userData.headlessThreeRenderer = {
        reflectionProbe: { texture, intensity: 2.5 },
      }
    } else {
      material.envMap = texture
      material.envMapIntensity = 2.5
    }
    scene.add(new THREE.Mesh(
      new THREE.SphereGeometry(1, 32, 16),
      material,
    ))
    return renderRgba(scene, makeCamera(), {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    })
  }

  for (const [label, type, value, equivalentByteColor] of cases) {
    for (const kind of ['scene', 'reflectionProbe', 'materialEnvMap']) {
      const byteRender = renderEnvironment(kind, byteEnvironmentTexture(equivalentByteColor))
      const packedRender = renderEnvironment(kind, packedEnvironmentTexture(type, value))
      const diff = meanAbsDiff(byteRender, packedRender)
      assert.ok(diff < 3, `${kind} ${label} environment should match equivalent linear RGBA8 IBL (diff=${diff.toFixed(3)})`)
    }
  }
})

test('packed unsigned short raw DataTexture maps unpack RGBA channels', () => {
  const cases = [
    ['UnsignedShort4444Type', THREE.UnsignedShort4444Type, 0x842f, 'red-dominant'],
    ['UnsignedShort5551Type', THREE.UnsignedShort5551Type, 0x823f, 'blue-dominant'],
  ]

  function packedTexture(type, value) {
    const texture = new THREE.DataTexture(new Uint16Array([value]), 1, 1, THREE.RGBAFormat, type)
    texture.needsUpdate = true
    return texture
  }

  function renderTexture(kind, type, value) {
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    if (kind === 'material') {
      scene.background = new THREE.Color(0, 0, 0)
      scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map: packedTexture(type, value) })))
    } else {
      scene.background = packedTexture(type, value)
    }
    return meanRegion(
      renderRgba(scene, camera, { width: 64, height: 64, outputColorSpace: THREE.LinearSRGBColorSpace }),
      64,
      64,
      24,
      24,
      40,
      40,
    )
  }

  for (const [name, type, value, expectation] of cases) {
    for (const kind of ['material', 'background']) {
      const mean = renderTexture(kind, type, value)
      if (expectation === 'red-dominant') {
        assert.ok(mean.r > 100, `${kind} ${name} red channel should unpack strongly (${mean.r})`)
        assert.ok(mean.r > mean.g + 20, `${kind} ${name} red should exceed green (${mean.r} vs ${mean.g})`)
        assert.ok(mean.g > mean.b + 15, `${kind} ${name} green should exceed blue (${mean.g} vs ${mean.b})`)
      } else {
        assert.ok(mean.b > 180, `${kind} ${name} blue channel should unpack strongly (${mean.b})`)
        assert.ok(mean.b > mean.r + 30, `${kind} ${name} blue should exceed red (${mean.b} vs ${mean.r})`)
        assert.ok(mean.r > mean.g + 20, `${kind} ${name} red should exceed green (${mean.r} vs ${mean.g})`)
      }
    }
  }
})

test('explicit raw texture mipmaps upload for material and background maps', () => {
  function mipmappedCheckerTexture() {
    const size = 16
    const data = []
    for (let y = 0; y < size; y += 1) {
      for (let x = 0; x < size; x += 1) {
        if ((x + y) % 2 === 0) data.push(255, 0, 0, 255)
        else data.push(0, 255, 0, 255)
      }
    }
    const map = rgbaTexture(data, size, size)
    map.wrapS = THREE.RepeatWrapping
    map.wrapT = THREE.RepeatWrapping
    map.repeat.set(128, 128)
    map.magFilter = THREE.NearestFilter
    map.minFilter = THREE.NearestMipmapNearestFilter
    map.generateMipmaps = false
    map.mipmaps = [8, 4, 2, 1].map((levelSize) => ({
      data: new Uint8Array(levelSize * levelSize * 4).fill(0).map((_, index) => (
        index % 4 === 2 || index % 4 === 3 ? 255 : 0
      )),
      width: levelSize,
      height: levelSize,
    }))
    return map
  }

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10)
  camera.position.set(0, 0, 3)

  const materialScene = new THREE.Scene()
  materialScene.background = new THREE.Color(0, 0, 0)
  materialScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map: mipmappedCheckerTexture() })))

  const backgroundScene = new THREE.Scene()
  backgroundScene.background = mipmappedCheckerTexture()

  for (const [name, scene] of [['material', materialScene], ['background', backgroundScene]]) {
    const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64, outputColorSpace: THREE.LinearSRGBColorSpace }))
    assert.ok(mean.b > 180, `${name} explicit mipmap levels should drive minified sampling (${mean.r}, ${mean.g}, ${mean.b})`)
    assert.ok(mean.r < 80 && mean.g < 80, `${name} base checker colors should not dominate explicit mip sampling (${mean.r}, ${mean.g}, ${mean.b})`)
  }
})

test('HalfFloatType explicit raw texture mipmaps decode before upload', () => {
  const size = 16
  const data = new Uint16Array(size * size * 4)
  for (let i = 0; i < size * size; i += 1) {
    data[i * 4] = 0x3c00
    data[i * 4 + 1] = 0x0000
    data[i * 4 + 2] = 0x0000
    data[i * 4 + 3] = 0x3c00
  }
  const map = new THREE.DataTexture(data, size, size, THREE.RGBAFormat, THREE.HalfFloatType)
  map.wrapS = THREE.RepeatWrapping
  map.wrapT = THREE.RepeatWrapping
  map.repeat.set(128, 128)
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestMipmapNearestFilter
  map.generateMipmaps = false
  map.mipmaps = [8, 4, 2, 1].map((levelSize) => {
    const level = new Uint16Array(levelSize * levelSize * 4)
    for (let i = 0; i < levelSize * levelSize; i += 1) {
      level[i * 4] = 0x0000
      level[i * 4 + 1] = 0x0000
      level[i * 4 + 2] = 0x3800
      level[i * 4 + 3] = 0x3c00
    }
    return { data: level, width: levelSize, height: levelSize }
  })
  map.needsUpdate = true

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map })))

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10)
  camera.position.set(0, 0, 3)

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64, outputColorSpace: THREE.LinearSRGBColorSpace }))
  assert.ok(mean.b > 100 && mean.b < 160, `half-float explicit mipmap blue should decode near 0.5 (${mean.b})`)
  assert.ok(mean.r < 60 && mean.g < 60, `half-float base red should not dominate explicit mip sampling (${mean.r}, ${mean.g})`)
})

test('malformed explicit texture mipmaps fail clearly', () => {
  const map = rgbaTexture(new Uint8Array(4 * 4 * 4).fill(255), 4, 4)
  map.mipmaps = [{ data: new Uint8Array([255, 255, 255, 255]), width: 1, height: 1 }]
  map.minFilter = THREE.LinearMipmapLinearFilter

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  assert.throws(
    () => renderRgba(scene, camera, { width: 64, height: 64 }),
    /mipmaps\[0\].*2x2/i,
  )

  const malformedMap = rgbaTexture(new Uint8Array(4 * 4 * 4).fill(255), 4, 4)
  malformedMap.mipmaps = 'mips'
  const malformedScene = new THREE.Scene()
  malformedScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map: malformedMap })))
  assert.throws(
    () => renderRgba(malformedScene, camera, { width: 64, height: 64 }),
    /material\.map\.mipmaps must be an array of image-like mip levels/i,
  )

  const malformedBackground = rgbaTexture(new Uint8Array(4 * 4 * 4).fill(255), 4, 4)
  malformedBackground.mipmaps = {}
  const backgroundScene = new THREE.Scene()
  backgroundScene.background = malformedBackground
  assert.throws(
    () => renderRgba(backgroundScene, camera, { width: 64, height: 64 }),
    /background\.mipmaps must be an array of image-like mip levels/i,
  )
})

test('packed physical extension maps reject explicit texture mipmaps clearly', () => {
  const clearcoatMap = rgbaTexture([
    255, 0, 0, 255,
    255, 0, 0, 255,
    255, 0, 0, 255,
    255, 0, 0, 255,
  ], 2, 2)
  clearcoatMap.minFilter = THREE.LinearMipmapLinearFilter
  clearcoatMap.mipmaps = [{ data: new Uint8Array([255, 0, 0, 255]), width: 1, height: 1 }]

  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(
    new THREE.SphereGeometry(1, 16, 8),
    new THREE.MeshPhysicalMaterial({ color: 0xffffff, clearcoat: 1, clearcoatMap }),
  ))

  assert.throws(
    () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
    /physical extension scalar maps.*explicit mipmaps.*clearcoatMap/i,
  )
})

test('unsupported texture inputs fail clearly for background and environment slots', () => {
  const compressedTexture = {
    isTexture: true,
    isCompressedTexture: true,
    image: { width: 4, height: 4 },
    mipmaps: [{ data: new Uint8Array(16), width: 4, height: 4 }],
  }
  const mipmappedTexture = solidTexture(255, 255, 255)
  mipmappedTexture.mipmaps = [{ data: new Uint8Array([255, 255, 255, 255]), width: 1, height: 1 }]

  function addMaterialEnvMap(scene, envMap) {
    envMap.mapping = THREE.EquirectangularReflectionMapping
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ envMap }),
    ))
  }

  const cases = [
    ['compressed material map', (scene) => {
      scene.background = new THREE.Color(0, 0, 0)
      scene.add(new THREE.Mesh(
        new THREE.PlaneGeometry(2, 2),
        new THREE.MeshBasicMaterial({ map: compressedTexture }),
      ))
    }, /compressed texture.*pre-decode/i],
    ['compressed background', (scene) => { scene.background = compressedTexture }, /compressed texture.*pre-decode/i],
    ['compressed environment', (scene) => { scene.environment = compressedTexture }, /compressed texture.*pre-decode/i],
    ['compressed material envMap', (scene) => {
      addMaterialEnvMap(scene, compressedTexture)
    }, /compressed texture.*pre-decode/i],
    ['compressed reflection probe', (scene) => {
      scene.userData.headlessThreeRenderer = { reflectionProbe: { texture: compressedTexture } }
    }, /compressed texture.*pre-decode/i],
    ['mipmapped environment', (scene) => { scene.environment = mipmappedTexture }, /explicit texture mipmaps.*not uploaded/i],
    ['mipmapped material envMap', (scene) => {
      addMaterialEnvMap(scene, mipmappedTexture)
    }, /explicit texture mipmaps.*not uploaded/i],
    ['mipmapped reflection probe', (scene) => {
      scene.userData.headlessThreeRenderer = { reflectionProbe: { texture: mipmappedTexture } }
    }, /explicit texture mipmaps.*not uploaded/i],
  ]

  for (const [name, setup, pattern] of cases) {
    const scene = new THREE.Scene()
    setup(scene)
    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      pattern,
      name,
    )
  }
})

test('unsupported array and 3D texture inputs fail clearly', () => {
  function dataArrayTexture() {
    const texture = new THREE.DataArrayTexture(new Uint8Array([255, 0, 0, 255]), 1, 1, 1)
    texture.needsUpdate = true
    return texture
  }

  function data3dTexture() {
    const texture = new THREE.Data3DTexture(new Uint8Array([255, 0, 0, 255]), 1, 1, 1)
    texture.needsUpdate = true
    return texture
  }

  const cases = [
    ['material DataArrayTexture', (scene) => {
      scene.add(new THREE.Mesh(
        new THREE.PlaneGeometry(2, 2),
        new THREE.MeshBasicMaterial({ map: dataArrayTexture() }),
      ))
    }, /material\.map uses an array or 3D texture/i],
    ['background DataArrayTexture', (scene) => {
      scene.background = dataArrayTexture()
    }, /background uses an array or 3D texture/i],
    ['environment Data3DTexture', (scene) => {
      scene.environment = data3dTexture()
    }, /scene\.environment uses an array or 3D texture/i],
    ['material envMap Data3DTexture', (scene) => {
      const envMap = data3dTexture()
      envMap.mapping = THREE.EquirectangularReflectionMapping
      scene.add(new THREE.Mesh(
        new THREE.PlaneGeometry(2, 2),
        new THREE.MeshBasicMaterial({ envMap }),
      ))
    }, /material\.envMap uses an array or 3D texture/i],
    ['reflection probe Data3DTexture', (scene) => {
      scene.userData.headlessThreeRenderer = { reflectionProbe: { texture: data3dTexture() } }
    }, /reflectionProbe\.texture uses an array or 3D texture/i],
  ]

  for (const [name, setup, pattern] of cases) {
    const scene = new THREE.Scene()
    setup(scene)
    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      pattern,
      name,
    )
  }
})

test('unsupported cube texture material slots fail clearly', () => {
  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({
      map: cubeTexture([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
      ]),
    }),
  ))

  assert.throws(
    () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
    /material\.map uses a cube or PMREM\/CubeUV texture mapping.*2D material texture slots/i,
  )
})

test('malformed environment and reflection probe texture values fail clearly', () => {
  const cases = [
    ['string scene environment', (scene) => {
      scene.environment = 'bright'
    }, /scene\.environment must be a Three\.js texture or null/i],
    ['empty scene environment', (scene) => {
      scene.environment = {}
    }, /scene\.environment must be a Three\.js texture or null/i],
    ['array scene environment', (scene) => {
      scene.environment = []
    }, /scene\.environment must be a Three\.js texture or null/i],
    ['image-less scene environment texture', (scene) => {
      scene.environment = new THREE.Texture()
    }, /scene\.environment.*texture image object.*not readable.*environment map rendering/i],
    ['string reflection probe texture', (scene) => {
      scene.userData.headlessThreeRenderer = { reflectionProbe: { texture: 'bright' } }
    }, /reflectionProbe\.texture must be a Three\.js texture or null/i],
    ['empty reflection probe object', (scene) => {
      scene.userData.headlessThreeRenderer = { reflectionProbe: {} }
    }, /reflectionProbe must be a Three\.js texture or null/i],
    ['malformed reflection probe list', (scene) => {
      scene.userData.headlessThreeRenderer = { reflectionProbes: 'probes' }
    }, /scene\.userData\.headlessThreeRenderer\.reflectionProbes must be an array/i],
    ['malformed legacy reflection probe list', (scene) => {
      scene.userData.headlessRenderer = { probes: {} }
    }, /scene\.userData\.headlessRenderer\.probes must be an array/i],
    ['scene userData container', (scene) => {
      scene.userData = 'probes'
    }, /scene\.userData must be an object/i],
    ['modern reflection hint container', (scene) => {
      scene.userData.headlessThreeRenderer = 'probes'
    }, /scene\.userData\.headlessThreeRenderer must be an object/i],
    ['legacy reflection hint container', (scene) => {
      scene.userData.headlessRenderer = []
    }, /scene\.userData\.headlessRenderer must be an object/i],
    ['malformed reflection probe map', (scene) => {
      scene.userData.headlessThreeRenderer = { reflectionProbes: [{ map: {} }] }
    }, /reflectionProbe\.map must be a Three\.js texture or null/i],
  ]

  for (const [name, setup, pattern] of cases) {
    const scene = new THREE.Scene()
    setup(scene)
    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      pattern,
      name,
    )
  }
})

test('unsupported raw DataTexture channel layouts fail clearly', () => {
  function invalidRawTexture(data, type = THREE.UnsignedByteType) {
    const texture = new THREE.DataTexture(data, 1, 1, THREE.RGBAFormat, type)
    texture.needsUpdate = true
    return texture
  }

  const cases = [
    ['material map', (scene) => {
      scene.background = new THREE.Color(0, 0, 0)
      scene.add(new THREE.Mesh(
        new THREE.PlaneGeometry(2, 2),
        new THREE.MeshBasicMaterial({ map: invalidRawTexture(new Uint8Array([255, 0, 0, 255, 1])) }),
      ))
    }, /material\.map raw texture data.*one-channel.*two-channel.*RGB.*RGBA.*texture rendering.*mismatched/i],
    ['background', (scene) => {
      scene.background = invalidRawTexture(new Uint8Array([255, 0, 0, 255, 1]))
    }, /background.*raw texture data.*one-channel.*two-channel.*RGB.*RGBA.*texture rendering.*mismatched/i],
    ['environment', (scene) => {
      scene.environment = invalidRawTexture(new Uint8Array([255, 0]))
    }, /scene\.environment raw texture data.*RGB or RGBA.*environment map rendering/i],
    ['mismatched environment', (scene) => {
      scene.environment = invalidRawTexture(new Uint8Array([255, 0, 0, 255, 1]))
    }, /scene\.environment raw texture data.*RGB or RGBA.*environment map rendering.*mismatched/i],
    ['material envMap', (scene) => {
      const envMap = invalidRawTexture(new Uint8Array([255, 0]))
      envMap.mapping = THREE.EquirectangularReflectionMapping
      scene.add(new THREE.Mesh(
        new THREE.PlaneGeometry(2, 2),
        new THREE.MeshBasicMaterial({ envMap }),
      ))
    }, /material\.envMap raw texture data.*RGB or RGBA.*environment map rendering/i],
    ['mismatched material envMap', (scene) => {
      const envMap = invalidRawTexture(new Uint8Array([255, 0, 0, 255, 1]))
      envMap.mapping = THREE.EquirectangularReflectionMapping
      scene.add(new THREE.Mesh(
        new THREE.PlaneGeometry(2, 2),
        new THREE.MeshBasicMaterial({ envMap }),
      ))
    }, /material\.envMap raw texture data.*RGB or RGBA.*environment map rendering.*mismatched/i],
    ['reflection probe', (scene) => {
      scene.userData.headlessThreeRenderer = {
        reflectionProbe: { texture: invalidRawTexture(new Uint8Array([255, 0])) },
      }
    }, /reflectionProbe\.texture raw texture data.*RGB or RGBA.*environment map rendering/i],
    ['mismatched reflection probe', (scene) => {
      scene.userData.headlessThreeRenderer = {
        reflectionProbe: { texture: invalidRawTexture(new Uint8Array([255, 0, 0, 255, 1])) },
      }
    }, /reflectionProbe\.texture raw texture data.*RGB or RGBA.*environment map rendering.*mismatched/i],
    ['FloatType environment', (scene) => {
      scene.environment = invalidRawTexture(new Float32Array([1, 0]), THREE.FloatType)
    }, /scene\.environment raw texture data.*RGB or RGBA.*environment map rendering/i],
    ['FloatType material envMap', (scene) => {
      const envMap = invalidRawTexture(new Float32Array([1, 0]), THREE.FloatType)
      envMap.mapping = THREE.EquirectangularReflectionMapping
      scene.add(new THREE.Mesh(
        new THREE.PlaneGeometry(2, 2),
        new THREE.MeshBasicMaterial({ envMap }),
      ))
    }, /material\.envMap raw texture data.*RGB or RGBA.*environment map rendering/i],
    ['FloatType reflection probe', (scene) => {
      scene.userData.headlessThreeRenderer = {
        reflectionProbe: { texture: invalidRawTexture(new Float32Array([1, 0]), THREE.FloatType) },
      }
    }, /reflectionProbe\.texture raw texture data.*RGB or RGBA.*environment map rendering/i],
  ]

  for (const [name, setup, pattern] of cases) {
    const scene = new THREE.Scene()
    setup(scene)
    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      pattern,
      name,
    )
  }
})

test('unsupported packed-depth raw DataTexture type constants fail clearly', () => {
  function rawTexture(data, type) {
    const texture = new THREE.DataTexture(data, 1, 1, THREE.RGBAFormat, type)
    texture.needsUpdate = true
    return texture
  }

  const cases = [
    ['material map packed depth type', (scene) => {
      scene.background = new THREE.Color(0, 0, 0)
      scene.add(new THREE.Mesh(
        new THREE.PlaneGeometry(2, 2),
        new THREE.MeshBasicMaterial({ map: rawTexture(new Uint32Array([0xffffffff]), THREE.UnsignedInt248Type) }),
      ))
    }, /material\.map raw texture type UnsignedInt248Type.*not supported/i],
    ['background packed depth type', (scene) => {
      scene.background = rawTexture(new Uint32Array([0xffffffff]), THREE.UnsignedInt248Type)
    }, /background raw texture type UnsignedInt248Type.*not supported/i],
    ['environment packed depth type', (scene) => {
      scene.environment = rawTexture(new Uint32Array([0xffffffff]), THREE.UnsignedInt248Type)
    }, /scene\.environment raw texture type UnsignedInt248Type.*not supported/i],
    ['material envMap packed depth type', (scene) => {
      const envMap = rawTexture(new Uint32Array([0xffffffff]), THREE.UnsignedInt248Type)
      envMap.mapping = THREE.EquirectangularReflectionMapping
      scene.add(new THREE.Mesh(
        new THREE.PlaneGeometry(2, 2),
        new THREE.MeshBasicMaterial({ envMap }),
      ))
    }, /material\.envMap raw texture type UnsignedInt248Type.*not supported/i],
    ['reflection probe packed depth type', (scene) => {
      scene.userData.headlessThreeRenderer = {
        reflectionProbe: { texture: rawTexture(new Uint32Array([0xffffffff]), THREE.UnsignedInt248Type) },
      }
    }, /reflectionProbe\.texture raw texture type UnsignedInt248Type.*not supported/i],
  ]

  for (const [name, setup, pattern] of cases) {
    const scene = new THREE.Scene()
    setup(scene)
    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      pattern,
      name,
    )
  }
})

test('browser-like texture image objects fail clearly in Node slots', () => {
  function browserLikeTexture() {
    const texture = new THREE.Texture({ width: 1, height: 1, complete: true })
    texture.needsUpdate = true
    return texture
  }
  function sourcedTexture(source) {
    const texture = new THREE.Texture()
    texture.source = source
    return texture
  }

  const cases = [
    ['material map', (scene) => {
      scene.background = new THREE.Color(0, 0, 0)
      scene.add(new THREE.Mesh(
        new THREE.PlaneGeometry(2, 2),
        new THREE.MeshBasicMaterial({ map: browserLikeTexture() }),
      ))
    }, /texture image object.*not readable.*texture rendering/i],
    ['background', (scene) => {
      scene.background = browserLikeTexture()
    }, /background.*texture image object.*not readable.*texture rendering/i],
    ['environment', (scene) => {
      scene.environment = browserLikeTexture()
    }, /scene\.environment.*texture image object.*not readable.*environment map rendering/i],
    ['material envMap', (scene) => {
      const envMap = browserLikeTexture()
      envMap.mapping = THREE.EquirectangularReflectionMapping
      scene.add(new THREE.Mesh(
        new THREE.PlaneGeometry(2, 2),
        new THREE.MeshBasicMaterial({ envMap }),
      ))
    }, /material\.envMap.*texture image object.*not readable.*environment map rendering/i],
    ['reflection probe', (scene) => {
      scene.userData.headlessThreeRenderer = {
        reflectionProbe: { texture: browserLikeTexture() },
      }
    }, /reflectionProbe\.texture.*texture image object.*not readable.*environment map rendering/i],
    ['material map source container', (scene) => {
      scene.add(new THREE.Mesh(
        new THREE.PlaneGeometry(2, 2),
        new THREE.MeshBasicMaterial({ map: sourcedTexture('source') }),
      ))
    }, /material\.map\.source must be a source-like object/i],
    ['background source data container', (scene) => {
      scene.background = sourcedTexture({ data: 'source' })
    }, /scene\.background\.source\.data must be an image-like object/i],
    ['environment source container', (scene) => {
      scene.environment = sourcedTexture('source')
    }, /scene\.environment\.source must be a source-like object/i],
    ['material envMap source data container', (scene) => {
      const envMap = sourcedTexture({ data: 'source' })
      envMap.mapping = THREE.EquirectangularReflectionMapping
      scene.add(new THREE.Mesh(
        new THREE.PlaneGeometry(2, 2),
        new THREE.MeshBasicMaterial({ envMap }),
      ))
    }, /material\.envMap\.source\.data must be an image-like object/i],
    ['reflection probe source data container', (scene) => {
      scene.userData.headlessThreeRenderer = {
        reflectionProbe: { texture: sourcedTexture({ data: 'source' }) },
      }
    }, /reflectionProbe\.texture\.source\.data must be an image-like object/i],
  ]

  for (const [name, setup, pattern] of cases) {
    const scene = new THREE.Scene()
    setup(scene)
    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      pattern,
      name,
    )
  }
})

test('unsupported texture channel indices fail clearly', () => {
  const map = solidTexture(255, 255, 255)
  map.channel = 4

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ map }),
  ))

  assert.throws(
    () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
    /texture\.channel 4.*not supported.*channel 0.*1.*2.*3/i,
  )

  const malformedMap = solidTexture(255, 255, 255)
  malformedMap.channel = '1'
  const malformedScene = new THREE.Scene()
  malformedScene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ map: malformedMap }),
  ))
  assert.throws(
    () => renderRgba(malformedScene, makeCamera(), { width: 64, height: 64 }),
    /texture\.channel must be an integer/i,
  )
})

test('MeshBasicMaterial map and alphaMap can sample distinct non-primary UV channels', () => {
  const map = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter
  map.channel = 1

  const alphaMap = rgbaTexture([
    255, 0, 255, 255,
    255, 255, 255, 255,
  ], 2, 1)
  alphaMap.magFilter = THREE.NearestFilter
  alphaMap.minFilter = THREE.NearestFilter
  alphaMap.channel = 2

  const geometry = constantUvPlane(0.25, 0.5)
  setConstantUvAttribute(geometry, 'uv1', 0.75, 0.5)
  setConstantUvAttribute(geometry, 'uv2', 0.75, 0.5)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    geometry,
    new THREE.MeshBasicMaterial({
      alphaMap,
      alphaTest: 0.5,
      color: 0xffffff,
      map,
    }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRegion(renderRgba(scene, camera, { width: 96, height: 96 }), 96, 96, 40, 40, 56, 56)
  assert.ok(
    mean.g > mean.r + 60,
    `map channel=1 and alphaMap channel=2 should render green from uv1 while uv2 keeps it opaque (${mean.g} vs ${mean.r})`,
  )
})

test('mixed non-primary texture channels fail clearly', () => {
  const map = solidTexture(255, 255, 255)
  map.channel = 1
  const alphaMap = solidTexture(255, 255, 255)
  alphaMap.channel = 2
  const metalnessMap = solidTexture(0, 0, 255)
  metalnessMap.channel = 3

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshStandardMaterial({
      alphaMap,
      alphaTest: 0.5,
      map,
      metalnessMap,
      metalness: 1,
      roughness: 0.5,
    }),
  ))

  assert.throws(
    () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
    /multiple non-primary texture\.channel values.*1.*3.*one secondary UV attribute/i,
  )
})

test('unsupported texture sampler constants fail clearly', () => {
  function assertMaterialSamplerFailure(configure, pattern, label) {
    const map = solidTexture(255, 255, 255)
    configure(map)
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ map }),
    ))

    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 32, height: 32 }),
      pattern,
      label,
    )
  }

  assertMaterialSamplerFailure(
    (map) => { map.wrapS = 999 },
    /texture wrap mode 999.*not supported.*ClampToEdgeWrapping.*RepeatWrapping.*MirroredRepeatWrapping/i,
    'wrapS',
  )
  assertMaterialSamplerFailure(
    (map) => { map.magFilter = 999 },
    /texture\.magFilter 999.*not supported.*NearestFilter.*LinearFilter/i,
    'magFilter',
  )
  assertMaterialSamplerFailure(
    (map) => { map.minFilter = 999 },
    /texture\.minFilter 999.*not supported.*NearestFilter.*LinearFilter.*mipmap/i,
    'minFilter',
  )

  const background = solidTexture(255, 255, 255)
  background.wrapT = 999
  const scene = new THREE.Scene()
  scene.background = background
  assert.throws(
    () => renderRgba(scene, makeCamera(), { width: 32, height: 32 }),
    /texture wrap mode 999.*not supported.*ClampToEdgeWrapping.*RepeatWrapping.*MirroredRepeatWrapping/i,
    'background wrapT',
  )
})

test('unsupported line texture channel indices fail clearly', () => {
  const map = solidTexture(255, 255, 255)
  map.channel = 4

  const geometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-1, 0, 0),
    new THREE.Vector3(1, 0, 0),
  ])
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Line(geometry, new THREE.LineBasicMaterial({ map })))

  assert.throws(
    () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
    /texture\.channel 4.*not supported.*channel 0.*1.*2.*3/i,
  )
})

test('unsupported texture colorSpace and encoding values fail clearly', () => {
  function assertMaterialColorSpaceFailure(configure, pattern, label) {
    const map = solidTexture(255, 255, 255)
    configure(map)
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ map }),
    ))

    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 32, height: 32 }),
      pattern,
      label,
    )
  }

  assertMaterialColorSpaceFailure(
    (map) => { map.colorSpace = 'display-p3' },
    /texture\.colorSpace display-p3.*not supported.*SRGBColorSpace.*LinearSRGBColorSpace.*NoColorSpace/i,
    'material colorSpace',
  )
  assertMaterialColorSpaceFailure(
    (map) => { map.encoding = 999 },
    /texture\.encoding 999.*not supported.*sRGBEncoding.*LinearEncoding.*texture\.colorSpace/i,
    'material encoding',
  )

  const background = solidTexture(255, 255, 255)
  background.colorSpace = 'display-p3'
  const scene = new THREE.Scene()
  scene.background = background
  assert.throws(
    () => renderRgba(scene, makeCamera(), { width: 32, height: 32 }),
    /texture\.colorSpace display-p3.*not supported.*SRGBColorSpace.*LinearSRGBColorSpace.*NoColorSpace/i,
    'background colorSpace',
  )
  const encodedBackground = solidTexture(255, 255, 255)
  encodedBackground.encoding = 999
  const encodedBackgroundScene = new THREE.Scene()
  encodedBackgroundScene.background = encodedBackground
  assert.throws(
    () => renderRgba(encodedBackgroundScene, makeCamera(), { width: 32, height: 32 }),
    /texture\.encoding 999.*not supported.*sRGBEncoding.*LinearEncoding.*texture\.colorSpace/i,
    'background encoding',
  )

  const environment = solidTexture(255, 255, 255)
  environment.colorSpace = 'display-p3'
  environment.mapping = THREE.EquirectangularReflectionMapping
  const environmentScene = new THREE.Scene()
  environmentScene.environment = environment
  assert.throws(
    () => renderRgba(environmentScene, makeCamera(), { width: 32, height: 32 }),
    /texture\.colorSpace display-p3.*not supported.*SRGBColorSpace.*LinearSRGBColorSpace.*NoColorSpace/i,
    'environment colorSpace',
  )
  const encodedEnvironment = solidTexture(255, 255, 255)
  encodedEnvironment.encoding = 999
  encodedEnvironment.mapping = THREE.EquirectangularReflectionMapping
  const encodedEnvironmentScene = new THREE.Scene()
  encodedEnvironmentScene.environment = encodedEnvironment
  assert.throws(
    () => renderRgba(encodedEnvironmentScene, makeCamera(), { width: 32, height: 32 }),
    /texture\.encoding 999.*not supported.*sRGBEncoding.*LinearEncoding.*texture\.colorSpace/i,
    'environment encoding',
  )

  const reflectionProbe = cubeTexture([
    [255, 255, 255],
    [255, 255, 255],
    [255, 255, 255],
    [255, 255, 255],
    [255, 255, 255],
    [255, 255, 255],
  ])
  reflectionProbe.colorSpace = 'display-p3'
  const reflectionProbeScene = new THREE.Scene()
  reflectionProbeScene.userData.headlessThreeRenderer = { reflectionProbe: { texture: reflectionProbe } }
  assert.throws(
    () => renderRgba(reflectionProbeScene, makeCamera(), { width: 32, height: 32 }),
    /texture\.colorSpace display-p3.*not supported.*SRGBColorSpace.*LinearSRGBColorSpace.*NoColorSpace/i,
    'reflection probe colorSpace',
  )
  const encodedReflectionProbe = cubeTexture([
    [255, 255, 255],
    [255, 255, 255],
    [255, 255, 255],
    [255, 255, 255],
    [255, 255, 255],
    [255, 255, 255],
  ])
  encodedReflectionProbe.encoding = 999
  const encodedReflectionProbeScene = new THREE.Scene()
  encodedReflectionProbeScene.userData.headlessThreeRenderer = { reflectionProbe: { texture: encodedReflectionProbe } }
  assert.throws(
    () => renderRgba(encodedReflectionProbeScene, makeCamera(), { width: 32, height: 32 }),
    /texture\.encoding 999.*not supported.*sRGBEncoding.*LinearEncoding.*texture\.colorSpace/i,
    'reflection probe encoding',
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

test('texture colorSpace string aliases match LinearSRGBColorSpace', () => {
  function renderTextureColorSpace(colorSpace) {
    const map = solidTexture(128, 128, 128)
    map.colorSpace = colorSpace

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map })))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return renderRgba(scene, camera, {
      width: 32,
      height: 32,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    })
  }

  const linear = renderTextureColorSpace(THREE.LinearSRGBColorSpace)
  for (const alias of ['srgb-linear', 'linear-srgb', 'linearsrgb', 'linear']) {
    assert.deepEqual(renderTextureColorSpace(alias), linear, `${alias} should match THREE.LinearSRGBColorSpace`)
  }
})

test('background texture colorSpace string aliases match LinearSRGBColorSpace', () => {
  function renderBackgroundColorSpace(colorSpace) {
    const background = solidTexture(128, 128, 128)
    background.colorSpace = colorSpace

    const scene = new THREE.Scene()
    scene.background = background

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return renderRgba(scene, camera, {
      width: 32,
      height: 32,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    })
  }

  const linear = renderBackgroundColorSpace(THREE.LinearSRGBColorSpace)
  for (const alias of ['srgb-linear', 'linear-srgb', 'linearsrgb', 'linear']) {
    assert.deepEqual(renderBackgroundColorSpace(alias), linear, `${alias} should match THREE.LinearSRGBColorSpace`)
  }
})

test('color-space decoding composes with explicit texture matrices', () => {
  function transformedGrayTexture(colorSpace) {
    const texture = rgbaTexture([
      0, 0, 0, 255,
      128, 128, 128, 255,
    ], 2, 1)
    texture.colorSpace = colorSpace
    texture.magFilter = THREE.NearestFilter
    texture.minFilter = THREE.NearestFilter
    setTextureMatrixOffset(texture, 0.5)
    return texture
  }

  function frontCamera() {
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return camera
  }

  function assertLinearBrighter(label, renderValue, threshold) {
    const srgb = renderValue(THREE.SRGBColorSpace)
    const linear = renderValue(THREE.LinearSRGBColorSpace)
    assert.ok(
      linear > srgb + threshold,
      `${label} should decode the matrix-selected sRGB texel before shading (${linear} vs ${srgb})`,
    )
  }

  assertLinearBrighter('base color map', (colorSpace) => {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.25, 0.5),
      new THREE.MeshBasicMaterial({ map: transformedGrayTexture(colorSpace) }),
    ))
    return meanRgba(renderRgba(scene, frontCamera(), {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    })).r
  }, 40)

  assertLinearBrighter('background texture', (colorSpace) => {
    const scene = new THREE.Scene()
    scene.background = transformedGrayTexture(colorSpace)
    return meanRgba(renderRgba(scene, frontCamera(), {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    })).r
  }, 40)

  assertLinearBrighter('sprite map', (colorSpace) => {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
      map: transformedGrayTexture(colorSpace),
      color: 0xffffff,
    }))
    sprite.scale.set(2, 2, 1)
    scene.add(sprite)
    return meanRegion(renderRgba(scene, frontCamera(), {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 64, 64, 18, 28, 26, 36).r
  }, 40)

  assertLinearBrighter('point map', (colorSpace) => {
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({
      color: 0xffffff,
      map: transformedGrayTexture(colorSpace),
      size: 48,
      sizeAttenuation: false,
    })))
    return meanRegion(renderRgba(scene, frontCamera(), {
      width: 96,
      height: 96,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 96, 96, 30, 44, 38, 52).r
  }, 40)

  assertLinearBrighter('line map', (colorSpace) => {
    const geom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-1.5, 0, 0),
      new THREE.Vector3(1.5, 0, 0),
    ])
    geom.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([
      0.25, 0.5,
      0.25, 0.5,
    ]), 2))

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Line(
      geom,
      new THREE.LineBasicMaterial({ color: 0xffffff, map: transformedGrayTexture(colorSpace) }),
    ))
    return meanRegion(renderRgba(scene, frontCamera(), {
      width: 96,
      height: 96,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 96, 96, 0, 46, 96, 50).r
  }, 10)

  assertLinearBrighter('dashed-line map', (colorSpace) => {
    const geom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-1.5, 0, 0),
      new THREE.Vector3(1.5, 0, 0),
    ])
    geom.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([
      0.25, 0.5,
      0.25, 0.5,
    ]), 2))

    const line = new THREE.Line(geom, new THREE.LineDashedMaterial({
      color: 0xffffff,
      dashSize: 10,
      gapSize: 0,
      map: transformedGrayTexture(colorSpace),
      scale: 1,
    }))
    line.computeLineDistances()

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(line)
    return meanRegion(renderRgba(scene, frontCamera(), {
      width: 96,
      height: 96,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 96, 96, 0, 46, 96, 50).r
  }, 10)

  assertLinearBrighter('matcap color map', (colorSpace) => {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.25, 0.5),
      new THREE.MeshMatcapMaterial({
        color: 0xffffff,
        matcap: solidTexture(255, 255, 255),
        map: transformedGrayTexture(colorSpace),
      }),
    ))
    return meanRgba(renderRgba(scene, frontCamera(), {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    })).r
  }, 40)

  assertLinearBrighter('emissive map', (colorSpace) => {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.25, 0.5),
      new THREE.MeshStandardMaterial({
        color: 0x000000,
        emissive: 0xffffff,
        emissiveMap: transformedGrayTexture(colorSpace),
      }),
    ))
    return meanRgba(renderRgba(scene, frontCamera(), {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    })).r
  }, 40)

  assertLinearBrighter('light map', (colorSpace) => {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.25, 0.5),
      new THREE.MeshBasicMaterial({
        color: 0xffffff,
        lightMap: transformedGrayTexture(colorSpace),
        lightMapIntensity: 2,
      }),
    ))
    return meanRgba(renderRgba(scene, frontCamera(), {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    })).r
  }, 40)

  assertLinearBrighter('specular color map', (colorSpace) => {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.25, 0.5),
      new THREE.MeshPhysicalMaterial({
        color: 0x000000,
        roughness: 0.05,
        metalness: 0,
        specularIntensity: 1,
        specularColor: new THREE.Color(1, 1, 1),
        specularColorMap: transformedGrayTexture(colorSpace),
      }),
    ))
    const light = new THREE.PointLight(0xffffff, 450)
    light.position.set(0, 0, 2)
    scene.add(light)
    return maxLuminance(renderRgba(scene, frontCamera(), { width: 64, height: 64 }))
  }, 5)

  assertLinearBrighter('sheen color map', (colorSpace) => {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.environment = makeEnvironmentTexture()
    scene.environmentIntensity = 3
    scene.add(new THREE.Mesh(
      constantUvPlane(0.25, 0.5),
      new THREE.MeshPhysicalMaterial({
        color: 0x000000,
        roughness: 1,
        metalness: 0,
        sheen: 1,
        sheenColor: new THREE.Color(1, 1, 1),
        sheenRoughness: 0.35,
        sheenColorMap: transformedGrayTexture(colorSpace),
      }),
    ))
    return maxLuminance(renderRgba(scene, frontCamera(), { width: 64, height: 64 }))
  }, 3)
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

test('material.toneMapped=false skips material tone mapping before output conversion', () => {
  function renderToneMapped(toneMapped) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(4, 4),
      new THREE.MeshBasicMaterial({
        color: new THREE.Color(1, 1, 1),
        toneMapped,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }))
  }

  const mapped = renderToneMapped(true)
  const unmapped = renderToneMapped(false)
  assert.ok(
    unmapped.r > mapped.r + 35,
    `toneMapped=false should keep brighter linear white (${unmapped.r} vs ${mapped.r})`,
  )
  assert.ok(unmapped.r > 245, `toneMapped=false should preserve white output (${unmapped.r})`)
})

test('outputColorSpace string aliases match Three.js constants', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.04, 0.08, 0.12)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(1.5, 1.5),
    new THREE.MeshBasicMaterial({ color: new THREE.Color(0.5, 0.28, 0.12) }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const renderColorSpace = (outputColorSpace) => renderRgba(scene, camera, {
    width: 32,
    height: 32,
    outputColorSpace,
  })
  const linear = renderColorSpace(THREE.LinearSRGBColorSpace)
  for (const alias of ['srgb-linear', 'linear-srgb', 'linearsrgb', 'linear']) {
    assert.deepEqual(renderColorSpace(alias), linear, `${alias} should match THREE.LinearSRGBColorSpace`)
  }
  assert.deepEqual(
    renderColorSpace('srgb'),
    renderColorSpace(THREE.SRGBColorSpace),
    'srgb should match THREE.SRGBColorSpace',
  )
})

test('unsupported outputColorSpace values fail clearly', () => {
  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial({ color: 0xffffff })))
  const camera = makeCamera()

  assert.throws(
    () => getRenderer().render(scene, camera, { width: 32, height: 32, outputColorSpace: 'display-p3' }),
    /options\.outputColorSpace display-p3 is not supported.*SRGBColorSpace.*LinearSRGBColorSpace/i,
  )
  assert.throws(
    () => renderToTarget(scene, camera, {}, { width: 32, height: 32, outputColorSpace: 'display-p3' }),
    /options\.outputColorSpace display-p3 is not supported.*SRGBColorSpace.*LinearSRGBColorSpace/i,
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

test('lightMap samples the selected UV channel', () => {
  function renderWithChannel(channel) {
    const lightMap = rgbaTexture([
      255, 0, 0, 255,
      0, 255, 0, 255,
    ], 2, 1)
    lightMap.channel = channel

    const geometry = constantUvPlane(0.25, 0.5)
    setConstantUvAttribute(geometry, 'uv1', 0.75, 0.5)

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

    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const primary = renderWithChannel(0)
  const secondary = renderWithChannel(1)
  assert.ok(primary.r > primary.g + 40, `lightMap channel=0 should sample primary red texel, got ${primary.r} vs ${primary.g}`)
  assert.ok(secondary.g > secondary.r + 40, `lightMap channel=1 should sample uv1 green texel, got ${secondary.g} vs ${secondary.r}`)
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

test('lightMap honors explicit texture matrices on the selected channel', () => {
  const lightMap = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  lightMap.channel = 1
  lightMap.magFilter = THREE.NearestFilter
  lightMap.minFilter = THREE.NearestFilter
  setTextureMatrixOffset(lightMap, 0.5)

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
  assert.ok(mean.g > mean.r + 40, `explicit lightMap matrix should sample uv1 green texel, got ${mean.g} vs ${mean.r}`)
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

test('lightMap honors horizontal and vertical repeat wrapping', () => {
  function renderWithWrapping({ wrapS, wrapT, vertical = false }) {
    const lightMap = vertical
      ? rgbaTexture([
        0, 255, 0, 255,
        255, 0, 0, 255,
      ], 1, 2)
      : rgbaTexture([
        0, 255, 0, 255,
        255, 0, 0, 255,
      ], 2, 1)
    if (wrapS != null) lightMap.wrapS = wrapS
    if (wrapT != null) lightMap.wrapT = wrapT
    lightMap.magFilter = THREE.NearestFilter
    lightMap.minFilter = THREE.NearestFilter

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(vertical ? 0.5 : 1.25, vertical ? 1.25 : 0.5),
      new THREE.MeshBasicMaterial({
        color: 0xffffff,
        lightMap,
        lightMapIntensity: 3,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const clamped = renderWithWrapping({ wrapS: THREE.ClampToEdgeWrapping })
  const repeated = renderWithWrapping({ wrapS: THREE.RepeatWrapping })
  const mirrored = renderWithWrapping({ wrapS: THREE.MirroredRepeatWrapping })
  assert.ok(clamped.r > clamped.g + 80, `clamped lightMap U coordinates should sample the red edge texel (${clamped.r} vs ${clamped.g})`)
  assert.ok(repeated.g > repeated.r + 30, `repeated lightMap U coordinates should wrap to the green texel (${repeated.g} vs ${repeated.r})`)
  assert.ok(mirrored.r > mirrored.g + 80, `mirrored lightMap U coordinates should reflect to the red texel (${mirrored.r} vs ${mirrored.g})`)

  const clampedVertical = renderWithWrapping({ wrapT: THREE.ClampToEdgeWrapping, vertical: true })
  const repeatedVertical = renderWithWrapping({ wrapT: THREE.RepeatWrapping, vertical: true })
  const mirroredVertical = renderWithWrapping({ wrapT: THREE.MirroredRepeatWrapping, vertical: true })
  assert.ok(clampedVertical.r > clampedVertical.g + 80, `clamped lightMap V coordinates should sample the red edge texel (${clampedVertical.r} vs ${clampedVertical.g})`)
  assert.ok(repeatedVertical.g > repeatedVertical.r + 30, `repeated lightMap V coordinates should wrap to the green texel (${repeatedVertical.g} vs ${repeatedVertical.r})`)
  assert.ok(mirroredVertical.r > mirroredVertical.g + 80, `mirrored lightMap V coordinates should reflect to the red texel (${mirroredVertical.r} vs ${mirroredVertical.g})`)
})

test('AmbientLight honors camera layer filtering', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const redFiltered = new THREE.AmbientLight(0xff0000, 4)
  redFiltered.layers.set(0)
  scene.add(redFiltered)

  const greenVisible = new THREE.AmbientLight(0x00ff00, 1.5)
  greenVisible.layers.set(1)
  scene.add(greenVisible)

  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, metalness: 0 }),
  )
  mesh.layers.set(1)
  scene.add(mesh)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  camera.layers.set(1)

  assert.deepEqual(extractAmbientLight(scene, camera), [0, 1, 0])
  assert.equal(extractAmbientIntensity(scene, camera), 1.5)

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  assert.ok(mean.g > mean.r + 15, `camera layer should select green AmbientLight and ignore red (${mean.g} vs ${mean.r})`)
})

test('HemisphereLight honors camera layer filtering', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const redFiltered = new THREE.HemisphereLight(0xff0000, 0x000000, 4)
  redFiltered.layers.set(0)
  scene.add(redFiltered)

  const greenVisible = new THREE.HemisphereLight(0x00ff00, 0x000000, 4)
  greenVisible.layers.set(1)
  scene.add(greenVisible)

  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(1, 32, 16),
    new THREE.MeshLambertMaterial({ color: 0xffffff }),
  )
  mesh.layers.set(1)
  scene.add(mesh)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  camera.layers.set(1)

  const [light] = extractLights(scene, camera) ?? []
  assert.equal(light?.lightType, 'hemisphere')
  assert.deepEqual(light?.color, [0, 1, 0])

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  assert.ok(mean.g > mean.r + 20, `camera layer should select green HemisphereLight and ignore red (${mean.g} vs ${mean.r})`)
  assert.ok(mean.g > mean.b + 35, `green HemisphereLight should tint the lit sphere (${mean.g} vs ${mean.b})`)
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

test('LightProbe ignores invisible probes and hidden ancestors', () => {
  function makeProbe(r, g, b) {
    const probe = new THREE.LightProbe(undefined, 1.8)
    for (const coefficient of probe.sh.coefficients) {
      coefficient.set(0, 0, 0)
    }
    probe.sh.coefficients[0].set(r, g, b)
    return probe
  }

  const hiddenProbe = makeProbe(1, 0, 0)
  const hiddenGroup = new THREE.Group()
  hiddenGroup.visible = false
  hiddenGroup.add(hiddenProbe)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(hiddenGroup)
  scene.add(makeProbe(0, 1, 0))
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, metalness: 0 }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  assert.ok(mean.g > mean.r + 25, `visible green LightProbe should ignore hidden red probe (${mean.g} vs ${mean.r})`)
  assert.ok(mean.g > mean.b + 40, `visible LightProbe should tint diffuse lighting green (${mean.g} vs ${mean.b})`)
})

test('LightProbe honors camera layer filtering', () => {
  function makeProbe(r, g, b, layer) {
    const probe = new THREE.LightProbe(undefined, 1.8)
    for (const coefficient of probe.sh.coefficients) {
      coefficient.set(0, 0, 0)
    }
    probe.sh.coefficients[0].set(r, g, b)
    probe.layers.set(layer)
    return probe
  }

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(makeProbe(1, 0, 0, 0))
  scene.add(makeProbe(0, 1, 0, 1))

  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, metalness: 0 }),
  )
  mesh.layers.set(1)
  scene.add(mesh)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  camera.layers.set(1)

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  assert.ok(mean.g > mean.r + 25, `camera layer should select the green LightProbe and ignore red (${mean.g} vs ${mean.r})`)
  assert.ok(mean.g > mean.b + 40, `camera layer should tint diffuse lighting green (${mean.g} vs ${mean.b})`)
})

test('LightProbe contributes diffuse lighting across lit material models', () => {
  function renderMaterial(material) {
    const probe = new THREE.LightProbe(undefined, 1.5)
    for (const coefficient of probe.sh.coefficients) {
      coefficient.set(0, 0, 0)
    }
    probe.sh.coefficients[0].set(1, 0, 0)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(probe)
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const cases = [
    ['Lambert', new THREE.MeshLambertMaterial({ color: 0xffffff })],
    ['Physical', new THREE.MeshPhysicalMaterial({ color: 0xffffff, roughness: 1, metalness: 0 })],
    ['Phong', new THREE.MeshPhongMaterial({ color: 0xffffff, shininess: 20 })],
    ['Toon', new THREE.MeshToonMaterial({ color: 0xffffff })],
  ]

  for (const [name, material] of cases) {
    const mean = renderMaterial(material)
    assert.ok(mean.r > mean.g + 25, `${name} should receive red LightProbe diffuse lighting (${mean.r} vs ${mean.g})`)
    assert.ok(mean.r > mean.b + 25, `${name} should receive red LightProbe diffuse lighting (${mean.r} vs ${mean.b})`)
  }
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

test('invalid light numeric values fail clearly', () => {
  const directCases = [
    ['directional intensity', () => {
      const light = new THREE.DirectionalLight(0xffffff, 1)
      light.intensity = 'bright'
      return light
    }, /light\.intensity must be a finite number/i],
    ['point intensity', () => {
      const light = new THREE.PointLight(0xffffff, 1)
      light.intensity = Number.NaN
      return light
    }, /light\.intensity must be a finite number/i],
    ['spot intensity', () => {
      const light = new THREE.SpotLight(0xffffff, 1)
      light.intensity = Number.POSITIVE_INFINITY
      return light
    }, /light\.intensity must be a finite number/i],
    ['hemisphere intensity', () => {
      const light = new THREE.HemisphereLight(0xffffff, 0x222222, 1)
      light.intensity = 'bright'
      return light
    }, /light\.intensity must be a finite number/i],
    ['rect intensity', () => {
      const light = new THREE.RectAreaLight(0xffffff, 1, 1, 1)
      light.intensity = Number.NEGATIVE_INFINITY
      return light
    }, /light\.intensity must be a finite number/i],
    ['directional target matrix', () => {
      const light = new THREE.DirectionalLight(0xffffff, 1)
      light.target.matrixWorld.elements[14] = Number.NaN
      return light
    }, /DirectionalLight\.target\.matrixWorld\.elements\[14\] must be a finite number/i],
    ['directional transform matrix', () => {
      const light = new THREE.DirectionalLight(0xffffff, 1)
      light.matrixWorld.elements[12] = Number.NaN
      return light
    }, /DirectionalLight\.matrixWorld\.elements\[12\] must be a finite number/i],
    ['directional target container', () => {
      const light = new THREE.DirectionalLight(0xffffff, 1)
      light.target = 'target'
      return light
    }, /DirectionalLight\.target must be an object/i],
    ['point transform matrix', () => {
      const light = new THREE.PointLight(0xffffff, 1)
      light.matrixWorld.elements[12] = Number.NaN
      return light
    }, /PointLight\.matrixWorld\.elements\[12\] must be a finite number/i],
    ['point distance', () => {
      const light = new THREE.PointLight(0xffffff, 1)
      light.distance = 'far'
      return light
    }, /PointLight\.distance must be a finite number/i],
    ['point distance negative', () => {
      const light = new THREE.PointLight(0xffffff, 1)
      light.distance = -1
      return light
    }, /PointLight\.distance must be non-negative/i],
    ['point decay', () => {
      const light = new THREE.PointLight(0xffffff, 1)
      light.decay = Number.POSITIVE_INFINITY
      return light
    }, /PointLight\.decay must be a finite number/i],
    ['point decay negative', () => {
      const light = new THREE.PointLight(0xffffff, 1)
      light.decay = -0.5
      return light
    }, /PointLight\.decay must be non-negative/i],
    ['spot distance', () => {
      const light = new THREE.SpotLight(0xffffff, 1)
      light.distance = Number.NaN
      return light
    }, /SpotLight\.distance must be a finite number/i],
    ['spot distance negative', () => {
      const light = new THREE.SpotLight(0xffffff, 1)
      light.distance = -1
      return light
    }, /SpotLight\.distance must be non-negative/i],
    ['spot target container', () => {
      const light = new THREE.SpotLight(0xffffff, 1)
      light.target = []
      return light
    }, /SpotLight\.target must be an object/i],
    ['spot transform matrix', () => {
      const light = new THREE.SpotLight(0xffffff, 1)
      light.matrixWorld.elements[12] = Number.NaN
      return light
    }, /SpotLight\.matrixWorld\.elements\[12\] must be a finite number/i],
    ['spot target matrix', () => {
      const light = new THREE.SpotLight(0xffffff, 1)
      light.target.matrixWorld.elements[13] = Number.NaN
      return light
    }, /SpotLight\.target\.matrixWorld\.elements\[13\] must be a finite number/i],
    ['spot decay negative', () => {
      const light = new THREE.SpotLight(0xffffff, 1)
      light.decay = -0.5
      return light
    }, /SpotLight\.decay must be non-negative/i],
    ['spot angle', () => {
      const light = new THREE.SpotLight(0xffffff, 1)
      light.angle = 'wide'
      return light
    }, /SpotLight\.angle must be a finite number/i],
    ['spot angle negative', () => {
      const light = new THREE.SpotLight(0xffffff, 1)
      light.angle = -0.1
      return light
    }, /SpotLight\.angle must be between 0 and Math\.PI \/ 2/i],
    ['spot angle too wide', () => {
      const light = new THREE.SpotLight(0xffffff, 1)
      light.angle = Math.PI
      return light
    }, /SpotLight\.angle must be between 0 and Math\.PI \/ 2/i],
    ['spot penumbra', () => {
      const light = new THREE.SpotLight(0xffffff, 1)
      light.penumbra = Number.NEGATIVE_INFINITY
      return light
    }, /SpotLight\.penumbra must be a finite number/i],
    ['spot penumbra negative', () => {
      const light = new THREE.SpotLight(0xffffff, 1)
      light.penumbra = -0.1
      return light
    }, /SpotLight\.penumbra must be between 0 and 1/i],
    ['spot penumbra above one', () => {
      const light = new THREE.SpotLight(0xffffff, 1)
      light.penumbra = 1.5
      return light
    }, /SpotLight\.penumbra must be between 0 and 1/i],
    ['hemisphere transform matrix', () => {
      const light = new THREE.HemisphereLight(0xffffff, 0x222222, 1)
      light.matrixWorld.elements[5] = Number.NaN
      return light
    }, /HemisphereLight\.matrixWorld\.elements\[5\] must be a finite number/i],
    ['rect width', () => {
      const light = new THREE.RectAreaLight(0xffffff, 1, 1, 1)
      light.width = 'wide'
      return light
    }, /RectAreaLight\.width must be a finite number/i],
    ['rect width zero', () => {
      const light = new THREE.RectAreaLight(0xffffff, 1, 1, 1)
      light.width = 0
      return light
    }, /RectAreaLight\.width must be positive/i],
    ['rect height', () => {
      const light = new THREE.RectAreaLight(0xffffff, 1, 1, 1)
      light.height = Number.NaN
      return light
    }, /RectAreaLight\.height must be a finite number/i],
    ['rect height negative', () => {
      const light = new THREE.RectAreaLight(0xffffff, 1, 1, 1)
      light.height = -1
      return light
    }, /RectAreaLight\.height must be positive/i],
    ['rect transform matrix', () => {
      const light = new THREE.RectAreaLight(0xffffff, 1, 1, 1)
      light.matrixWorld.elements[8] = Number.NEGATIVE_INFINITY
      return light
    }, /RectAreaLight\.matrixWorld\.elements\[8\] must be a finite number/i],
  ]

  for (const [name, makeLight, pattern] of directCases) {
    const scene = new THREE.Scene()
    scene.add(makeLight())
    assert.throws(
      () => extractLights(scene),
      pattern,
      `${name} should fail clearly`,
    )
  }

  const ambientScene = new THREE.Scene()
  const ambient = new THREE.AmbientLight(0xffffff, 1)
  ambient.intensity = 'bright'
  ambientScene.add(ambient)
  assert.throws(
    () => extractAmbientIntensity(ambientScene),
    /AmbientLight\.intensity must be a finite number/i,
  )

  const probeScene = new THREE.Scene()
  const probe = new THREE.LightProbe(undefined, 1)
  probe.intensity = Number.NaN
  probeScene.add(probe)
  assert.throws(
    () => extractLightProbe(probeScene),
    /LightProbe\.intensity must be a finite number/i,
  )

  const vectorCoefficientScene = new THREE.Scene()
  const vectorCoefficientProbe = new THREE.LightProbe(undefined, 1)
  vectorCoefficientProbe.sh.coefficients[0] = { x: 1, y: 'green', z: 0 }
  vectorCoefficientScene.add(vectorCoefficientProbe)
  assert.throws(
    () => extractLightProbe(vectorCoefficientScene),
    /LightProbe\.sh\.coefficients\[0\]\.y must be a finite number/i,
  )

  const arrayCoefficientScene = new THREE.Scene()
  const arrayCoefficientProbe = new THREE.LightProbe(undefined, 1)
  arrayCoefficientProbe.sh.coefficients[0] = [1, Number.NEGATIVE_INFINITY, 0]
  arrayCoefficientScene.add(arrayCoefficientProbe)
  assert.throws(
    () => extractLightProbe(arrayCoefficientScene),
    /LightProbe\.sh\.coefficients\[0\]\[1\] must be a finite number/i,
  )

  const missingCoefficientsScene = new THREE.Scene()
  const missingCoefficientsProbe = new THREE.LightProbe(undefined, 1)
  missingCoefficientsProbe.sh.coefficients = [{ x: 1, y: 0, z: 0 }]
  missingCoefficientsScene.add(missingCoefficientsProbe)
  assert.throws(
    () => extractLightProbe(missingCoefficientsScene),
    /LightProbe\.sh\.coefficients must contain 9 coefficients/i,
  )

  const invalidCoefficientsScene = new THREE.Scene()
  const invalidCoefficientsProbe = new THREE.LightProbe(undefined, 1)
  invalidCoefficientsProbe.sh.coefficients = 'bright'
  invalidCoefficientsScene.add(invalidCoefficientsProbe)
  assert.throws(
    () => extractLightProbe(invalidCoefficientsScene),
    /LightProbe\.sh\.coefficients must be an array of 9 coefficients/i,
  )
})

test('invalid light color values fail clearly', () => {
  const directScene = new THREE.Scene()
  const directional = new THREE.DirectionalLight(0xffffff, 1)
  directional.color = { isColor: true, r: 1, g: 'green', b: 0 }
  directScene.add(directional)
  assert.throws(
    () => extractLights(directScene),
    /light\.color\.g must be a finite number/i,
  )

  const hemisphereScene = new THREE.Scene()
  const hemisphere = new THREE.HemisphereLight(0xffffff, 0x222222, 1)
  hemisphere.groundColor = { isColor: true, r: 0, g: 0, b: Number.NaN }
  hemisphereScene.add(hemisphere)
  assert.throws(
    () => extractLights(hemisphereScene),
    /HemisphereLight\.groundColor\.b must be a finite number/i,
  )

  const primitiveColorScene = new THREE.Scene()
  const primitiveColor = new THREE.PointLight(0xffffff, 1)
  primitiveColor.color = 'white'
  primitiveColorScene.add(primitiveColor)
  assert.throws(
    () => extractLights(primitiveColorScene),
    /light\.color must be a color-like object or \[r, g, b\]/i,
  )

  const ambientScene = new THREE.Scene()
  const ambient = new THREE.AmbientLight(0xffffff, 1)
  ambient.color = { isColor: true, r: 1, g: 1, b: 'blue' }
  ambientScene.add(ambient)
  assert.throws(
    () => extractAmbientLight(ambientScene),
    /AmbientLight\.color\.b must be a finite number/i,
  )
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

test('LOD selection accounts for camera zoom', () => {
  function makeScene() {
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
    return scene
  }

  const farCamera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  farCamera.position.set(0, 0, 6)
  farCamera.lookAt(0, 0, 0)

  const zoomedCamera = farCamera.clone()
  zoomedCamera.zoom = 2

  const far = meanRgba(renderRgba(makeScene(), farCamera, { width: 64, height: 64 }))
  const zoomed = meanRgba(renderRgba(makeScene(), zoomedCamera, { width: 64, height: 64 }))

  assert.ok(far.b > far.r + 5, `unzoomed far LOD should render the blue level (${far.b} vs ${far.r})`)
  assert.ok(zoomed.r > zoomed.b + 10, `zoomed LOD distance should render the red level (${zoomed.r} vs ${zoomed.b})`)
})

test('LOD autoUpdate=false preserves manual level visibility', () => {
  const redNear = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color: 0xff0000 }))
  const blueFar = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color: 0x0000ff }))
  redNear.visible = false
  blueFar.visible = true

  const lod = new THREE.LOD()
  lod.addLevel(redNear, 0)
  lod.addLevel(blueFar, 4)
  lod.autoUpdate = false

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(lod)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  assert.ok(mean.b > mean.r + 5, `manual blue LOD level should remain visible when autoUpdate=false (${mean.b} vs ${mean.r})`)
})

test('invalid LOD level values fail clearly', () => {
  function makeLodScene(mutator) {
    const lod = new THREE.LOD()
    lod.addLevel(
      new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color: 0xff0000 })),
      0,
    )
    lod.addLevel(
      new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color: 0x0000ff })),
      4,
    )
    mutator(lod)

    const scene = new THREE.Scene()
    scene.add(lod)
    return scene
  }

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 6)
  camera.lookAt(0, 0, 0)

  const malformedLevelsScene = new THREE.Scene()
  const malformedLevelsLod = new THREE.Object3D()
  malformedLevelsLod.isLOD = true
  malformedLevelsLod.levels = 'levels'
  malformedLevelsScene.add(malformedLevelsLod)
  assert.throws(
    () => renderRgba(malformedLevelsScene, camera, { width: 64, height: 64 }),
    /LOD\.levels must be an array/i,
    'levels container',
  )

  const cases = [
    ['autoUpdate', (lod) => {
      lod.autoUpdate = 'yes'
    }, /LOD\.autoUpdate must be a boolean/i],
    ['level entry', (lod) => {
      lod.levels[1] = null
    }, /LOD\.levels\[1\] must be an object/i],
    ['level object', (lod) => {
      lod.levels[1].object = null
    }, /LOD\.levels\[1\]\.object must be a THREE\.Object3D-like object/i],
    ['distance', (lod) => {
      lod.levels[1].distance = 'far'
    }, /LOD\.levels\[1\]\.distance must be a finite number/i],
    ['distance negative', (lod) => {
      lod.levels[1].distance = -1
    }, /LOD\.levels\[1\]\.distance must be non-negative/i],
    ['hysteresis', (lod) => {
      lod.levels[1].hysteresis = Number.POSITIVE_INFINITY
    }, /LOD\.levels\[1\]\.hysteresis must be a finite number/i],
    ['hysteresis negative', (lod) => {
      lod.levels[1].hysteresis = -0.1
    }, /LOD\.levels\[1\]\.hysteresis must be between 0 and 1/i],
    ['hysteresis above one', (lod) => {
      lod.levels[1].hysteresis = 1.5
    }, /LOD\.levels\[1\]\.hysteresis must be between 0 and 1/i],
  ]

  for (const [label, mutate, pattern] of cases) {
    assert.throws(
      () => renderRgba(makeLodScene(mutate), camera, { width: 64, height: 64 }),
      pattern,
      label,
    )
  }

  const invalidZoomCamera = camera.clone()
  invalidZoomCamera.zoom = Number.NaN
  assert.throws(
    () => renderRgba(makeLodScene(() => {}), invalidZoomCamera, { width: 64, height: 64 }),
    /camera\.zoom must be a finite number/i,
  )

  const zeroZoomCamera = camera.clone()
  zeroZoomCamera.zoom = 0
  assert.throws(
    () => renderRgba(makeLodScene(() => {}), zeroZoomCamera, { width: 64, height: 64 }),
    /camera\.zoom must be positive/i,
  )
})

test('Fog and FogExp2 affect material output', () => {
  function renderFogged(fog, materialFog = true) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.fog = fog
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ color: 0xff0000, fog: materialFog }),
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

  const optOut = renderFogged(new THREE.Fog(0x00ff00, 0, 1), false)
  assert.ok(
    optOut.r > optOut.g + 40,
    `material.fog=false should keep the red material color (${optOut.r} vs ${optOut.g})`,
  )
})

test('Fog uses view-space depth rather than Euclidean camera distance', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.fog = new THREE.Fog(0x00ff00, 2.9, 3.1)

  for (const x of [0, 2]) {
    const plane = new THREE.Mesh(
      new THREE.PlaneGeometry(0.8, 0.8),
      new THREE.MeshBasicMaterial({ color: 0xff0000 }),
    )
    plane.position.x = x
    scene.add(plane)
  }

  const camera = new THREE.OrthographicCamera(-3, 3, 2, -2, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 96, height: 96 })
  const center = meanRegion(rgba, 96, 96, 42, 42, 54, 54)
  const offAxis = meanRegion(rgba, 96, 96, 74, 42, 86, 54)
  assert.ok(Math.abs(center.g - offAxis.g) < 15, `same view-depth planes should receive similar linear fog (${center.g} vs ${offAxis.g})`)
  assert.ok(Math.abs(center.r - offAxis.r) < 15, `same view-depth planes should retain similar red output (${center.r} vs ${offAxis.r})`)
})

test('invalid fog parameter values fail clearly', () => {
  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ color: 0xff0000 })))
  const camera = makeCamera()

  scene.fog = 'fog'
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /scene\.fog must be a THREE\.Fog or THREE\.FogExp2 object/i,
  )

  scene.fog = { color: new THREE.Color(0x00ff00) }
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /scene\.fog must be a THREE\.Fog or THREE\.FogExp2 object/i,
  )

  scene.fog = new THREE.Fog(0x00ff00, 0, 1)
  scene.fog.near = Number.NaN
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /scene\.fog\.near must be a finite number/i,
  )

  scene.fog = new THREE.FogExp2(0x0000ff, 1)
  scene.fog.density = 'dense'
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /scene\.fog\.density must be a finite number/i,
  )

  scene.fog = new THREE.FogExp2(0x0000ff, 1)
  scene.fog.density = -0.1
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /scene\.fog\.density must be non-negative/i,
  )

  scene.fog = new THREE.Fog(0x00ff00, 10, 1)
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /scene\.fog\.far must be greater than scene\.fog\.near/i,
  )

  scene.fog = new THREE.Fog(0x00ff00, 1001, 2000)
  delete scene.fog.far
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /scene\.fog\.near must be less than the effective scene\.fog\.far/i,
  )

  scene.fog = new THREE.Fog(0x00ff00, 0, 1)
  scene.fog.color = { isColor: true, r: 0, g: Number.POSITIVE_INFINITY, b: 0 }
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /scene\.fog\.color\.g must be a finite number/i,
  )

  scene.fog = new THREE.Fog(0x00ff00, 0, 1)
  scene.fog.color = 'green'
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /scene\.fog\.color must be a color-like object or \[r, g, b\]/i,
  )
})

test('Fog affects sprites, points, and lines with material fog opt-out', () => {
  function renderObject(object) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.fog = new THREE.Fog(0x00ff00, 0, 1)
    scene.add(object)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return renderRgba(scene, camera, { width: 64, height: 64 })
  }

  function makeSprite(fog) {
    const material = new THREE.SpriteMaterial({ color: 0xff0000 })
    material.fog = fog
    const sprite = new THREE.Sprite(material)
    sprite.scale.set(1.2, 1.2, 1)
    return sprite
  }

  function makePoint(fog) {
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))
    const material = new THREE.PointsMaterial({
      color: 0xff0000,
      size: 34,
      sizeAttenuation: false,
    })
    material.fog = fog
    return new THREE.Points(geometry, material)
  }

  function makeLine(fog) {
    const geometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-1, 0, 0),
      new THREE.Vector3(1, 0, 0),
    ])
    const material = new THREE.LineBasicMaterial({ color: 0xff0000 })
    material.fog = fog
    return new THREE.Line(geometry, material)
  }

  for (const [label, makeObject] of [
    ['sprite', makeSprite],
    ['point', makePoint],
  ]) {
    const fogged = meanRegion(renderObject(makeObject(true)), 64, 64, 24, 24, 40, 40)
    const unfogged = meanRegion(renderObject(makeObject(false)), 64, 64, 24, 24, 40, 40)
    assert.ok(fogged.g > fogged.r + 40, `${label} should be mixed toward green fog (${fogged.g} vs ${fogged.r})`)
    assert.ok(
      unfogged.r > unfogged.g + 40,
      `${label} fog=false should keep the red material color (${unfogged.r} vs ${unfogged.g})`,
    )
  }

  const foggedLine = renderObject(makeLine(true))
  const unfoggedLine = renderObject(makeLine(false))
  const greenLinePixels = countRegionPixels(
    foggedLine,
    64,
    64,
    8,
    28,
    56,
    36,
    (r, g, b) => g > r + 30 && g > b + 30,
  )
  const redLinePixels = countRegionPixels(
    unfoggedLine,
    64,
    64,
    8,
    28,
    56,
    36,
    (r, g, b) => r > g + 30 && r > b + 30,
  )
  assert.ok(greenLinePixels > 2, `line should be mixed toward green fog (${greenLinePixels})`)
  assert.ok(redLinePixels > 2, `line fog=false should keep the red material color (${redLinePixels})`)
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

test('invalid physical material scalar values fail clearly', () => {
  const cases = [
    ['metalness', (material) => {
      material.metalness = 'metal'
    }, /material\.metalness must be a finite number/i],
    ['roughness', (material) => {
      material.roughness = Number.NaN
    }, /material\.roughness must be a finite number/i],
    ['clearcoat', (material) => {
      material.clearcoat = 'coat'
    }, /material\.clearcoat must be a finite number/i],
    ['clearcoatRoughness', (material) => {
      material.clearcoatRoughness = Number.POSITIVE_INFINITY
    }, /material\.clearcoatRoughness must be a finite number/i],
    ['clearcoatNormalScale.x', (material) => {
      material.clearcoatNormalScale = new THREE.Vector2(1, 1)
      material.clearcoatNormalScale.x = 'wide'
    }, /material\.clearcoatNormalScale\.x must be a finite number/i],
    ['clearcoatNormalScale container', (material) => {
      material.clearcoatNormalScale = 'wide'
    }, /material\.clearcoatNormalScale must be a Vector2-like object/i],
    ['sheen', (material) => {
      material.sheen = 'soft'
    }, /material\.sheen must be a finite number/i],
    ['sheenRoughness', (material) => {
      material.sheenRoughness = Number.NaN
    }, /material\.sheenRoughness must be a finite number/i],
    ['anisotropy', (material) => {
      material.anisotropy = 'aligned'
    }, /material\.anisotropy must be a finite number/i],
    ['anisotropyRotation', (material) => {
      material.anisotropyRotation = Number.NEGATIVE_INFINITY
    }, /material\.anisotropyRotation must be a finite number/i],
    ['iridescence', (material) => {
      material.iridescence = 'rainbow'
    }, /material\.iridescence must be a finite number/i],
    ['iridescenceIOR', (material) => {
      material.iridescenceIOR = Number.NaN
    }, /material\.iridescenceIOR must be a finite number/i],
    ['iridescenceThicknessRange container', (material) => {
      material.iridescenceThicknessRange = 'range'
    }, /material\.iridescenceThicknessRange must be an array-like pair/i],
    ['iridescenceThicknessRange length', (material) => {
      material.iridescenceThicknessRange = [100]
    }, /material\.iridescenceThicknessRange must contain at least two values/i],
    ['iridescenceThicknessRange value', (material) => {
      material.iridescenceThicknessRange = [100, 'thick']
    }, /material\.iridescenceThicknessRange\[1\] must be a finite number/i],
    ['transmission', (material) => {
      material.transmission = 'glass'
    }, /material\.transmission must be a finite number/i],
    ['dispersion', (material) => {
      material.dispersion = Number.NaN
    }, /material\.dispersion must be a finite number/i],
    ['ior', (material) => {
      material.ior = 'dense'
    }, /material\.ior must be a finite number/i],
    ['thickness', (material) => {
      material.thickness = Number.POSITIVE_INFINITY
    }, /material\.thickness must be a finite number/i],
    ['attenuationDistance', (material) => {
      material.attenuationDistance = 'short'
    }, /material\.attenuationDistance must be a finite number/i],
    ['specularIntensity', (material) => {
      material.specularIntensity = Number.NaN
    }, /material\.specularIntensity must be a finite number/i],
  ]

  for (const [name, mutate, pattern] of cases) {
    const material = new THREE.MeshPhysicalMaterial({ color: 0xffffff })
    mutate(material)
    const scene = new THREE.Scene()
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(1, 1), material))

    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      pattern,
      `${name} should fail clearly`,
    )
  }
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

test('MeshPhysicalMaterial scalar clearcoat and roughness affect IBL specular', () => {
  function renderClearcoat(clearcoat, clearcoatRoughness) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.environment = makeEnvironmentTexture()
    scene.environmentIntensity = 2
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshPhysicalMaterial({
        color: 0x000000,
        roughness: 1,
        metalness: 0,
        clearcoat,
        clearcoatRoughness,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const disabled = renderClearcoat(0, 0.04)
  const glossy = renderClearcoat(1, 0.04)
  const rough = renderClearcoat(1, 1)
  const luminance = (mean) => 0.2126 * mean.r + 0.7152 * mean.g + 0.0722 * mean.b

  assert.ok(
    luminance(glossy) > luminance(disabled) + 80,
    `clearcoat should add environment specular (${luminance(glossy).toFixed(1)} vs ${luminance(disabled).toFixed(1)})`,
  )
  assert.ok(
    luminance(glossy) > luminance(rough) + 60,
    `lower clearcoatRoughness should keep a stronger environment highlight (${luminance(glossy).toFixed(1)} vs ${luminance(rough).toFixed(1)})`,
  )
})

test('MeshPhysicalMaterial transmission volume attenuation honors color and distance', () => {
  function renderAttenuated(attenuationColor, attenuationDistance) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const back = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    )
    back.position.z = -0.2
    scene.add(back)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshPhysicalMaterial({
        color: 0xffffff,
        roughness: 0.1,
        metalness: 0,
        transmission: 1,
        ior: 1.5,
        thickness: 8,
        attenuationColor: new THREE.Color(attenuationColor),
        attenuationDistance,
      }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const blueShort = renderAttenuated(0x0505ff, 1)
  const blueLong = renderAttenuated(0x0505ff, 100)
  const redShort = renderAttenuated(0xff0505, 1)

  assert.ok(blueShort.b > blueShort.r + 80, `short blue attenuation should tint transmission blue (${blueShort.b} vs ${blueShort.r})`)
  assert.ok(blueLong.r > blueShort.r + 60, `long attenuation distance should preserve more transmitted red (${blueLong.r} vs ${blueShort.r})`)
  assert.ok(redShort.r > redShort.b + 80, `red attenuationColor should tint the same volume red (${redShort.r} vs ${redShort.b})`)
})

test('MeshPhysicalMaterial scalar iridescence tints direct specular', () => {
  function renderIridescence(iridescence) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshPhysicalMaterial({
        color: 0x000000,
        roughness: 0.08,
        metalness: 0,
        specularIntensity: 1,
        iridescence,
        iridescenceIOR: 1.8,
        iridescenceThicknessRange: [250, 650],
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

  const plain = meanRgba(renderIridescence(0))
  const iridescent = meanRgba(renderIridescence(1))

  assert.ok(
    Math.abs(iridescent.r - iridescent.g) > 2 || Math.abs(iridescent.g - iridescent.b) > 2,
    `iridescence should tint the highlight, got ${JSON.stringify(iridescent)}`,
  )
  assert.ok(
    meanAbsDiff(renderIridescence(0), renderIridescence(1)) > 0.5,
    'scalar iridescence should change physical material output',
  )
})

test('MeshPhysicalMaterial iridescenceMap modulates scalar iridescence', () => {
  function renderIridescenceMap(matrixOffsetX) {
    const iridescenceMap = rgbaTexture([
      0, 0, 0, 255,
      255, 0, 0, 255,
    ], 2, 1)
    setTextureMatrixOffset(iridescenceMap, matrixOffsetX)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.25, 0.5),
      new THREE.MeshPhysicalMaterial({
        color: 0x000000,
        roughness: 0.08,
        metalness: 0,
        specularIntensity: 1,
        iridescence: 1,
        iridescenceMap,
        iridescenceIOR: 1.8,
        iridescenceThicknessRange: [250, 650],
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

  assert.ok(
    meanAbsDiff(renderIridescenceMap(0), renderIridescenceMap(0.5)) > 10,
    'explicit iridescenceMap matrix should select the texel that enables iridescence',
  )
})

test('MeshPhysicalMaterial iridescenceMap samples the selected secondary UV channel', () => {
  function renderIridescenceMap(channel) {
    const iridescenceMap = rgbaTexture([
      0, 0, 0, 255,
      255, 0, 0, 255,
    ], 2, 1)
    iridescenceMap.channel = channel

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
        iridescence: 1,
        iridescenceMap,
        iridescenceIOR: 1.8,
        iridescenceThicknessRange: [250, 650],
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

  assert.ok(
    meanAbsDiff(renderIridescenceMap(0), renderIridescenceMap(1)) > 10,
    'iridescenceMap channel=1 should sample the uv1 texel that enables iridescence',
  )
})

test('MeshPhysicalMaterial iridescenceThicknessMap selects film thickness range', () => {
  function renderThicknessMap(matrixOffsetX) {
    const iridescenceThicknessMap = rgbaTexture([
      0, 0, 0, 255,
      0, 255, 0, 255,
    ], 2, 1)
    setTextureMatrixOffset(iridescenceThicknessMap, matrixOffsetX)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.25, 0.5),
      new THREE.MeshPhysicalMaterial({
        color: 0x000000,
        roughness: 0.08,
        metalness: 0,
        specularIntensity: 1,
        iridescence: 1,
        iridescenceIOR: 1.8,
        iridescenceThicknessRange: [120, 760],
        iridescenceThicknessMap,
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

  assert.ok(
    meanAbsDiff(renderThicknessMap(0), renderThicknessMap(0.5)) > 5,
    'explicit iridescenceThicknessMap matrix should select a different film thickness',
  )
})

test('MeshPhysicalMaterial iridescenceThicknessMap samples the selected secondary UV channel', () => {
  function renderThicknessMap(channel) {
    const iridescenceThicknessMap = rgbaTexture([
      0, 0, 0, 255,
      0, 255, 0, 255,
    ], 2, 1)
    iridescenceThicknessMap.channel = channel

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
        iridescence: 1,
        iridescenceIOR: 1.8,
        iridescenceThicknessRange: [120, 760],
        iridescenceThicknessMap,
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

  assert.ok(
    meanAbsDiff(renderThicknessMap(0), renderThicknessMap(1)) > 5,
    'iridescenceThicknessMap channel=1 should sample the uv1 texel for the upper film thickness range',
  )
})

test('MeshPhysicalMaterial iridescence maps honor horizontal and vertical wrap modes', () => {
  function iridescenceWrapMap(vertical) {
    return vertical
      ? rgbaTexture([
        0, 0, 0, 255,
        0, 0, 0, 255,
        255, 0, 0, 255,
        255, 0, 0, 255,
      ], 2, 2)
      : rgbaTexture([
        0, 0, 0, 255,
        255, 0, 0, 255,
      ], 2, 1)
  }

  function thicknessWrapMap(vertical) {
    return vertical
      ? rgbaTexture([
        0, 0, 0, 255,
        0, 0, 0, 255,
        0, 255, 0, 255,
        0, 255, 0, 255,
      ], 2, 2)
      : rgbaTexture([
        0, 0, 0, 255,
        0, 255, 0, 255,
      ], 2, 1)
  }

  function renderIridescenceMap({ wrapS, wrapT, offsetX = 0, offsetY = 0, vertical = false }) {
    const iridescenceMap = iridescenceWrapMap(vertical)
    iridescenceMap.magFilter = THREE.NearestFilter
    iridescenceMap.minFilter = THREE.NearestFilter
    iridescenceMap.offset.set(offsetX, offsetY)
    if (wrapS != null) iridescenceMap.wrapS = wrapS
    if (wrapT != null) iridescenceMap.wrapT = wrapT

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.25, vertical ? 0.25 : 0.5),
      new THREE.MeshPhysicalMaterial({
        color: 0x000000,
        roughness: 0.08,
        metalness: 0,
        specularIntensity: 1,
        iridescence: 1,
        iridescenceMap,
        iridescenceIOR: 1.8,
        iridescenceThicknessRange: [250, 650],
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

  function renderThicknessMap({ wrapS, wrapT, offsetX = 0, offsetY = 0, vertical = false }) {
    const iridescenceThicknessMap = thicknessWrapMap(vertical)
    iridescenceThicknessMap.magFilter = THREE.NearestFilter
    iridescenceThicknessMap.minFilter = THREE.NearestFilter
    iridescenceThicknessMap.offset.set(offsetX, offsetY)
    if (wrapS != null) iridescenceThicknessMap.wrapS = wrapS
    if (wrapT != null) iridescenceThicknessMap.wrapT = wrapT

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.25, vertical ? 0.25 : 0.5),
      new THREE.MeshPhysicalMaterial({
        color: 0x000000,
        roughness: 0.08,
        metalness: 0,
        specularIntensity: 1,
        iridescence: 1,
        iridescenceIOR: 1.8,
        iridescenceThicknessRange: [120, 760],
        iridescenceThicknessMap,
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

  const clampedIridescence = renderIridescenceMap({ offsetX: 1 })
  const repeatedIridescence = renderIridescenceMap({ wrapS: THREE.RepeatWrapping, offsetX: 1 })
  const iridescenceDiff = meanAbsDiff(clampedIridescence, repeatedIridescence)
  assert.ok(iridescenceDiff > 0.5, `RepeatWrapping should wrap iridescenceMap UVs before sampling, diff=${iridescenceDiff.toFixed(2)}`)

  const repeatedHighIridescence = renderIridescenceMap({ wrapS: THREE.RepeatWrapping, offsetX: 1.5 })
  const mirroredIridescence = renderIridescenceMap({ wrapS: THREE.MirroredRepeatWrapping, offsetX: 1.5 })
  const mirroredIridescenceDiff = meanAbsDiff(repeatedHighIridescence, mirroredIridescence)
  assert.ok(mirroredIridescenceDiff > 0.5, `MirroredRepeatWrapping should reflect iridescenceMap U coordinates differently than RepeatWrapping, diff=${mirroredIridescenceDiff.toFixed(2)}`)

  const clampedThickness = renderThicknessMap({ offsetX: 1 })
  const repeatedThickness = renderThicknessMap({ wrapS: THREE.RepeatWrapping, offsetX: 1 })
  const thicknessDiff = meanAbsDiff(clampedThickness, repeatedThickness)
  assert.ok(thicknessDiff > 0.5, `RepeatWrapping should wrap iridescenceThicknessMap UVs before sampling, diff=${thicknessDiff.toFixed(2)}`)

  const repeatedHighThickness = renderThicknessMap({ wrapS: THREE.RepeatWrapping, offsetX: 1.5 })
  const mirroredThickness = renderThicknessMap({ wrapS: THREE.MirroredRepeatWrapping, offsetX: 1.5 })
  const mirroredThicknessDiff = meanAbsDiff(repeatedHighThickness, mirroredThickness)
  assert.ok(mirroredThicknessDiff > 0.5, `MirroredRepeatWrapping should reflect iridescenceThicknessMap U coordinates differently than RepeatWrapping, diff=${mirroredThicknessDiff.toFixed(2)}`)

  const clampedVerticalIridescence = renderIridescenceMap({ offsetY: 1, vertical: true })
  const repeatedVerticalIridescence = renderIridescenceMap({ wrapT: THREE.RepeatWrapping, offsetY: 1, vertical: true })
  const verticalIridescenceDiff = meanAbsDiff(clampedVerticalIridescence, repeatedVerticalIridescence)
  assert.ok(verticalIridescenceDiff > 0.5, `RepeatWrapping should wrap iridescenceMap V coordinates before sampling, diff=${verticalIridescenceDiff.toFixed(2)}`)

  const repeatedHighVerticalIridescence = renderIridescenceMap({ wrapT: THREE.RepeatWrapping, offsetY: 1.5, vertical: true })
  const mirroredVerticalIridescence = renderIridescenceMap({ wrapT: THREE.MirroredRepeatWrapping, offsetY: 1.5, vertical: true })
  const mirroredVerticalIridescenceDiff = meanAbsDiff(repeatedHighVerticalIridescence, mirroredVerticalIridescence)
  assert.ok(mirroredVerticalIridescenceDiff > 0.5, `MirroredRepeatWrapping should reflect iridescenceMap V coordinates differently than RepeatWrapping, diff=${mirroredVerticalIridescenceDiff.toFixed(2)}`)

  const clampedVerticalThickness = renderThicknessMap({ offsetY: 1, vertical: true })
  const repeatedVerticalThickness = renderThicknessMap({ wrapT: THREE.RepeatWrapping, offsetY: 1, vertical: true })
  const verticalThicknessDiff = meanAbsDiff(clampedVerticalThickness, repeatedVerticalThickness)
  assert.ok(verticalThicknessDiff > 0.5, `RepeatWrapping should wrap iridescenceThicknessMap V coordinates before sampling, diff=${verticalThicknessDiff.toFixed(2)}`)

  const repeatedHighVerticalThickness = renderThicknessMap({ wrapT: THREE.RepeatWrapping, offsetY: 1.5, vertical: true })
  const mirroredVerticalThickness = renderThicknessMap({ wrapT: THREE.MirroredRepeatWrapping, offsetY: 1.5, vertical: true })
  const mirroredVerticalThicknessDiff = meanAbsDiff(repeatedHighVerticalThickness, mirroredVerticalThickness)
  assert.ok(mirroredVerticalThicknessDiff > 0.5, `MirroredRepeatWrapping should reflect iridescenceThicknessMap V coordinates differently than RepeatWrapping, diff=${mirroredVerticalThicknessDiff.toFixed(2)}`)
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

test('physical extension maps honor explicit texture matrices', () => {
  function makeMap(data, matrixOffsetX = 0) {
    const texture = rgbaTexture(data, 2, 1)
    texture.magFilter = THREE.NearestFilter
    texture.minFilter = THREE.NearestFilter
    if (matrixOffsetX !== 0) setTextureMatrixOffset(texture, matrixOffsetX)
    return texture
  }

  function frontCamera() {
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return camera
  }

  function luminance(mean) {
    return 0.2126 * mean.r + 0.7152 * mean.g + 0.0722 * mean.b
  }

  function renderSpecularColor(matrixOffsetX) {
    const specularColorMap = makeMap([
      0, 0, 0, 255,
      255, 0, 0, 255,
    ], matrixOffsetX)

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
        specularColorMap,
      }),
    ))
    const light = new THREE.PointLight(0xffffff, 300)
    light.position.set(0, 0, 2)
    scene.add(light)
    return meanRgba(renderRgba(scene, frontCamera(), { width: 64, height: 64 }))
  }

  const specularColorPrimary = renderSpecularColor(0)
  const specularColorShifted = renderSpecularColor(0.5)
  assert.ok(
    specularColorShifted.r > specularColorPrimary.r + 4,
    `explicit specularColorMap matrix should sample the red texel (${specularColorShifted.r} vs ${specularColorPrimary.r})`,
  )
  assert.ok(
    specularColorShifted.r > specularColorShifted.g + 4,
    `explicit specularColorMap matrix should tint the specular response red (${specularColorShifted.r} vs ${specularColorShifted.g})`,
  )

  function renderSpecularIntensity(matrixOffsetX) {
    const specularIntensityMap = makeMap([
      0, 0, 0, 0,
      0, 0, 0, 255,
    ], matrixOffsetX)

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
        specularIntensityMap,
      }),
    ))
    const light = new THREE.PointLight(0xffffff, 300)
    light.position.set(0, 0, 2)
    scene.add(light)
    return renderRgba(scene, frontCamera(), { width: 64, height: 64 })
  }

  assert.ok(
    maxLuminance(renderSpecularIntensity(0.5)) > maxLuminance(renderSpecularIntensity(0)) + 40,
    'explicit specularIntensityMap matrix should enable the shifted specular texel',
  )

  function renderTransmission(matrixOffsetX) {
    const transmissionMap = makeMap([
      0, 0, 0, 255,
      255, 0, 0, 255,
    ], matrixOffsetX)

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
    return meanRgba(renderRgba(scene, frontCamera(), { width: 64, height: 64 }))
  }

  const transmissionPrimary = renderTransmission(0)
  const transmissionShifted = renderTransmission(0.5)
  assert.ok(
    transmissionPrimary.r > transmissionPrimary.b + 30,
    `primary transmissionMap texel should keep the physical surface opaque red (${transmissionPrimary.r} vs ${transmissionPrimary.b})`,
  )
  assert.ok(
    transmissionShifted.b > transmissionShifted.r + 40,
    `explicit transmissionMap matrix should sample the transmitting texel (${transmissionShifted.b} vs ${transmissionShifted.r})`,
  )

  function renderClearcoat(matrixOffsetX) {
    const clearcoatMap = makeMap([
      0, 0, 0, 255,
      255, 0, 0, 255,
    ], matrixOffsetX)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.environment = makeEnvironmentTexture()
    scene.environmentIntensity = 2
    scene.add(new THREE.Mesh(
      constantUvPlane(0.25, 0.5),
      new THREE.MeshPhysicalMaterial({
        color: 0x000000,
        roughness: 1,
        metalness: 0,
        clearcoat: 1,
        clearcoatRoughness: 0.04,
        clearcoatMap,
      }),
    ))
    return meanRgba(renderRgba(scene, frontCamera(), { width: 64, height: 64 }))
  }

  assert.ok(
    luminance(renderClearcoat(0.5)) > luminance(renderClearcoat(0)) + 80,
    'explicit clearcoatMap matrix should enable stronger clearcoat IBL',
  )

  function renderClearcoatRoughness(matrixOffsetX) {
    const clearcoatRoughnessMap = makeMap([
      0, 0, 0, 255,
      0, 255, 0, 255,
    ], matrixOffsetX)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.environment = makeEnvironmentTexture()
    scene.environmentIntensity = 2
    scene.add(new THREE.Mesh(
      constantUvPlane(0.25, 0.5),
      new THREE.MeshPhysicalMaterial({
        color: 0x000000,
        roughness: 1,
        metalness: 0,
        clearcoat: 1,
        clearcoatRoughness: 1,
        clearcoatRoughnessMap,
      }),
    ))
    return meanRgba(renderRgba(scene, frontCamera(), { width: 64, height: 64 }))
  }

  assert.ok(
    luminance(renderClearcoatRoughness(0)) > luminance(renderClearcoatRoughness(0.5)) + 20,
    'explicit clearcoatRoughnessMap matrix should sample the rougher shifted texel',
  )

  function renderClearcoatNormal(matrixOffsetX) {
    const clearcoatNormalMap = makeMap([
      128, 128, 255, 255,
      255, 128, 128, 255,
    ], matrixOffsetX)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.environment = makeEnvironmentTexture()
    scene.environmentIntensity = 2
    scene.add(new THREE.Mesh(
      constantUvPlane(0.25, 0.5),
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
    return renderRgba(scene, frontCamera(), { width: 64, height: 64 })
  }

  assert.ok(
    meanAbsDiff(renderClearcoatNormal(0), renderClearcoatNormal(0.5)) > 5,
    'explicit clearcoatNormalMap matrix should sample the tilted normal texel',
  )

  function renderSheenColor(matrixOffsetX) {
    const sheenColorMap = makeMap([
      0, 0, 0, 255,
      255, 0, 0, 255,
    ], matrixOffsetX)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.environment = makeEnvironmentTexture()
    scene.environmentIntensity = 2
    scene.add(new THREE.Mesh(
      constantUvPlane(0.25, 0.5),
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
    return meanRgba(renderRgba(scene, frontCamera(), { width: 64, height: 64 }))
  }

  const sheenColorPrimary = renderSheenColor(0)
  const sheenColorShifted = renderSheenColor(0.5)
  assert.ok(
    sheenColorShifted.r > sheenColorPrimary.r + 3,
    `explicit sheenColorMap matrix should add red sheen (${sheenColorShifted.r} vs ${sheenColorPrimary.r})`,
  )
  assert.ok(
    sheenColorShifted.r > sheenColorShifted.g + 3,
    `explicit sheenColorMap matrix should keep the sampled red sheen tint (${sheenColorShifted.r} vs ${sheenColorShifted.g})`,
  )

  function renderSheenRoughness(matrixOffsetX) {
    const sheenRoughnessMap = makeMap([
      0, 0, 0, 0,
      0, 0, 0, 255,
    ], matrixOffsetX)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.environment = makeEnvironmentTexture()
    scene.environmentIntensity = 2
    scene.add(new THREE.Mesh(
      constantUvPlane(0.25, 0.5),
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
    return renderRgba(scene, frontCamera(), { width: 64, height: 64 })
  }

  assert.ok(
    meanAbsDiff(renderSheenRoughness(0), renderSheenRoughness(0.5)) > 5,
    'explicit sheenRoughnessMap matrix should sample the rough shifted texel',
  )

  function renderAnisotropy(matrixOffsetX) {
    const anisotropyMap = makeMap([
      128, 128, 0, 255,
      255, 128, 255, 255,
    ], matrixOffsetX)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.25, 0.5),
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
    return renderRgba(scene, frontCamera(), { width: 64, height: 64 })
  }

  assert.ok(
    meanAbsDiff(renderAnisotropy(0), renderAnisotropy(0.5)) > 1,
    'explicit anisotropyMap matrix should sample the anisotropic shifted texel',
  )

  function renderThickness(matrixOffsetX) {
    const thicknessMap = makeMap([
      0, 0, 0, 255,
      0, 255, 0, 255,
    ], matrixOffsetX)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const back = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    )
    back.position.z = -0.2
    scene.add(back)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.25, 0.5),
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
    return meanRgba(renderRgba(scene, frontCamera(), { width: 64, height: 64 }))
  }

  const thicknessPrimary = renderThickness(0)
  const thicknessShifted = renderThickness(0.5)
  assert.ok(
    thicknessPrimary.r > thicknessPrimary.b - 15,
    `primary thicknessMap texel should leave the transmitted plane mostly white (${thicknessPrimary.r} vs ${thicknessPrimary.b})`,
  )
  assert.ok(
    thicknessShifted.b > thicknessShifted.r + 40,
    `explicit thicknessMap matrix should sample the attenuating texel (${thicknessShifted.b} vs ${thicknessShifted.r})`,
  )
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

  function renderIridescence(filter) {
    const iridescenceMap = filteredTexture([
      0, 0, 0, 255,
      255, 0, 0, 255,
    ], filter)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.45, 0.5),
      new THREE.MeshPhysicalMaterial({
        color: 0x000000,
        roughness: 0.08,
        metalness: 0,
        specularIntensity: 1,
        iridescence: 1,
        iridescenceMap,
        iridescenceIOR: 1.8,
        iridescenceThicknessRange: [250, 650],
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

  function renderIridescenceThickness(filter) {
    const iridescenceThicknessMap = filteredTexture([
      0, 0, 0, 255,
      0, 255, 0, 255,
    ], filter)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      constantUvPlane(0.45, 0.5),
      new THREE.MeshPhysicalMaterial({
        color: 0x000000,
        roughness: 0.08,
        metalness: 0,
        specularIntensity: 1,
        iridescence: 1,
        iridescenceIOR: 1.8,
        iridescenceThicknessRange: [120, 760],
        iridescenceThicknessMap,
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

  const nearestIridescence = renderIridescence(THREE.NearestFilter)
  const linearIridescence = renderIridescence(THREE.LinearFilter)
  const iridescenceDiff = meanAbsDiff(nearestIridescence, linearIridescence)
  assert.ok(iridescenceDiff > 0.5, `LinearFilter should blend in iridescence factor, diff=${iridescenceDiff.toFixed(2)}`)

  const nearestThickness = renderIridescenceThickness(THREE.NearestFilter)
  const linearThickness = renderIridescenceThickness(THREE.LinearFilter)
  const thicknessDiff = meanAbsDiff(nearestThickness, linearThickness)
  assert.ok(thicknessDiff > 0.5, `LinearFilter should blend iridescence thickness differently than NearestFilter, diff=${thicknessDiff.toFixed(2)}`)
})

test('conflicting packed physical texture samplers fail clearly', () => {
  const clearcoatMap = solidTexture(255, 0, 0)
  clearcoatMap.magFilter = THREE.NearestFilter
  clearcoatMap.minFilter = THREE.NearestFilter
  const clearcoatRoughnessMap = solidTexture(0, 255, 0)
  clearcoatRoughnessMap.magFilter = THREE.LinearFilter
  clearcoatRoughnessMap.minFilter = THREE.LinearFilter

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshPhysicalMaterial({
      clearcoat: 1,
      clearcoatMap,
      clearcoatRoughness: 0.5,
      clearcoatRoughnessMap,
    }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  assert.throws(
    () => renderRgba(scene, camera, { width: 64, height: 64 }),
    /physical extension scalar maps.*packed.*sampler settings.*clearcoatRoughnessMap.*clearcoatMap/i,
  )
})

test('conflicting packed physical texture anisotropy settings fail clearly', () => {
  const clearcoatMap = solidTexture(255, 0, 0)
  clearcoatMap.anisotropy = 4
  const clearcoatRoughnessMap = solidTexture(0, 255, 0)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshPhysicalMaterial({
      clearcoat: 1,
      clearcoatMap,
      clearcoatRoughness: 0.5,
      clearcoatRoughnessMap,
    }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  assert.throws(
    () => renderRgba(scene, camera, { width: 64, height: 64 }),
    /physical extension scalar maps.*packed.*anisotropy.*clearcoatRoughnessMap.*clearcoatMap/i,
  )
})

test('specularColorMap samples selected uv1-uv3 texture channels', () => {
  function renderWithChannel(channel) {
    const specularColorMap = rgbaTexture([
      0, 0, 0, 255,
      255, 0, 0, 255,
    ], 2, 1)
    specularColorMap.channel = channel

    const geometry = constantUvPlane(0.25, 0.5)
    setConstantUvAttribute(geometry, 'uv1', channel === 1 ? 0.75 : 0.25, 0.5)
    setConstantUvAttribute(geometry, 'uv2', channel === 2 ? 0.75 : 0.25, 0.5)
    setConstantUvAttribute(geometry, 'uv3', channel === 3 ? 0.75 : 0.25, 0.5)

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
  const uv1 = renderWithChannel(1)
  const uv2 = renderWithChannel(2)
  const uv3 = renderWithChannel(3)
  for (const [label, sample] of [['uv1', uv1], ['uv2', uv2], ['uv3', uv3]]) {
    assert.ok(sample.r > primary.r + 4, `specularColorMap ${label} should sample its red texel (${sample.r} vs ${primary.r})`)
    assert.ok(sample.r > sample.g + 4, `specularColorMap ${label} should tint the specular response red (${sample.r} vs ${sample.g})`)
  }
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

test('custom WGSL fragment material can read the expanded light budget', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const material = new THREE.MeshBasicMaterial({ color: 0xffffff })
  material.userData.headlessThreeRenderer = {
    fragmentWgsl: `
      if uniforms.num_lights == 64u && uniforms.lights[63].color_intensity.r > 0.5 {
        return vec4<f32>(0.0, 1.0, 0.0, alpha);
      }
      return vec4<f32>(1.0, 0.0, 0.0, alpha);
    `,
  }
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))
  for (let i = 0; i < 64; i += 1) {
    const light = new THREE.PointLight(i === 63 ? 0xff0000 : 0xffffff, 1)
    light.position.set((i % 8) - 3.5, 2, Math.floor(i / 8) - 1.5)
    scene.add(light)
  }

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  assert.ok(mean.g > mean.r + 40, `custom WGSL should read light slot 63 and render green (${mean.g} vs ${mean.r})`)
})

test('over native direct light budget fails clearly', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xffffff }),
  ))
  for (let i = 0; i < 65; i += 1) {
    const light = new THREE.PointLight(0xffffff, 1)
    light.position.set((i % 8) - 3.5, 2, Math.floor(i / 8) - 2)
    scene.add(light)
  }

  assert.throws(
    () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
    /More than 64 visible non-ambient lights.*65 found/i,
  )
})

test('camera-layer-filtered lights do not count toward the direct light budget', () => {
  const scene = new THREE.Scene()
  const camera = makeCamera()
  camera.layers.set(1)

  for (let i = 0; i < 65; i += 1) {
    const light = new THREE.PointLight(0xffffff, 1)
    light.position.set((i % 8) - 3.5, 2, Math.floor(i / 8) - 2)
    light.layers.set(i < 64 ? 1 : 0)
    scene.add(light)
  }

  assert.equal(extractLights(scene, camera)?.length, 64)
})

test('ShaderMaterial without headless WGSL override fails clearly', () => {
  const cases = [
    ['ShaderMaterial', new THREE.ShaderMaterial({
      vertexShader: 'void main() { gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
      fragmentShader: 'void main() { gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); }',
    }), /ShaderMaterial.*fragmentWgsl/i],
    ['RawShaderMaterial', new THREE.RawShaderMaterial(), /RawShaderMaterial.*fragmentWgsl/i],
    ['NodeMaterial', Object.assign(new THREE.MeshBasicMaterial({ color: 0xffffff }), {
      isNodeMaterial: true,
      type: 'MeshBasicNodeMaterial',
    }), /NodeMaterial.*fragmentWgsl/i],
  ]

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  for (const [name, material, pattern] of cases) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))
    assert.throws(
      () => renderRgba(scene, camera, { width: 64, height: 64 }),
      pattern,
      name,
    )
  }
})

test('custom WGSL fragment override values fail clearly', () => {
  const cases = [
    ['top-level customFragmentWgsl', (material) => {
      material.customFragmentWgsl = {}
    }, /material\.customFragmentWgsl must be a string/i],
    ['documented fragmentWgsl', (material) => {
      material.userData.headlessThreeRenderer = { fragmentWgsl: 1 }
    }, /material\.userData\.headlessThreeRenderer\.fragmentWgsl must be a string/i],
    ['legacy fragmentWgsl', (material) => {
      material.userData.headlessRenderer = { fragmentWgsl: false }
    }, /material\.userData\.headlessRenderer\.fragmentWgsl must be a string/i],
    ['material userData container', (material) => {
      material.userData = 'fragment'
    }, /material\.userData must be an object/i],
    ['modern hint container', (material) => {
      material.userData.headlessThreeRenderer = 1
    }, /material\.userData\.headlessThreeRenderer must be an object/i],
    ['legacy hint container', (material) => {
      material.userData.headlessRenderer = []
    }, /material\.userData\.headlessRenderer must be an object/i],
  ]

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  for (const [name, configure, pattern] of cases) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const material = new THREE.MeshBasicMaterial({ color: 0xffffff })
    configure(material)
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

    assert.throws(
      () => renderRgba(scene, camera, { width: 64, height: 64 }),
      pattern,
      name,
    )
  }
})

test('unsupported base Material without headless WGSL override fails clearly', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.Material()))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  assert.throws(
    () => renderRgba(scene, camera, { width: 64, height: 64 }),
    /Material.*not supported.*fragmentWgsl/i,
  )
})

test('malformed material containers fail clearly', () => {
  const cases = [
    ['direct material', 'material', /material must be a material-like object/i],
    ['material array entry', [
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
      'material',
    ], /material\[1\] must be a material-like object/i],
  ]

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  for (const [name, material, pattern] of cases) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const geometry = new THREE.PlaneGeometry(2, 2)
    if (Array.isArray(material)) {
      geometry.clearGroups()
      geometry.addGroup(0, geometry.index.count, 1)
    }
    scene.add(new THREE.Mesh(geometry, material))

    assert.throws(
      () => renderRgba(scene, camera, { width: 64, height: 64 }),
      pattern,
      name,
    )
  }
})

test('ShaderMaterial, RawShaderMaterial, NodeMaterial, and base Material can opt into custom WGSL fragment output', () => {
  function renderCustom(material) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    material.userData.headlessThreeRenderer = {
      fragmentWgsl: 'return vec4<f32>(0.0, 1.0, 0.0, alpha);',
    }
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    return meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  }

  const cases = [
    ['ShaderMaterial', new THREE.ShaderMaterial()],
    ['RawShaderMaterial', new THREE.RawShaderMaterial()],
    ['NodeMaterial', Object.assign(new THREE.MeshBasicMaterial({ color: 0xffffff }), {
      isNodeMaterial: true,
      type: 'MeshBasicNodeMaterial',
    })],
    ['Material', new THREE.Material()],
  ]

  for (const [name, material] of cases) {
    const mean = renderCustom(material)
    assert.ok(mean.g > mean.r + 40, `${name} WGSL override should render green output (${mean.g} vs ${mean.r})`)
    assert.ok(mean.g > mean.b + 40, `${name} WGSL override should render green output (${mean.g} vs ${mean.b})`)
  }
})

test('ShaderMaterial custom WGSL preserves output alpha', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const material = new THREE.ShaderMaterial()
  material.userData.headlessThreeRenderer = {
    fragmentWgsl: 'return vec4<f32>(0.0, 1.0, 0.0, alpha * 0.5);',
  }
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 20, 20, 44, 44)
  assert.ok(mean.g > mean.r + 40, `ShaderMaterial WGSL override should render green output (${mean.g} vs ${mean.r})`)
  assert.ok(mean.a > 120 && mean.a < 140, `ShaderMaterial WGSL override should preserve returned alpha (${mean.a})`)
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
  assert.equal(target.texture.source.data, target.texture.image)
  assert.equal(target.texture.source.data.data, target.data)
  assert.equal(target.texture.source.data.width, 64)
  assert.equal(target.texture.source.data.height, 32)

  const singleTextureArrayTarget = { texture: [{}] }
  renderToTarget(scene, makeCamera(), singleTextureArrayTarget, { width: 32, height: 16 })
  assert.equal(singleTextureArrayTarget.width, 32)
  assert.equal(singleTextureArrayTarget.height, 16)
  assert.equal(singleTextureArrayTarget.texture[0].image.data, singleTextureArrayTarget.data)
  assert.equal(singleTextureArrayTarget.texture[0].source.data, singleTextureArrayTarget.texture[0].image)

  const texturesTarget = { textures: [{}] }
  renderToTarget(scene, makeCamera(), texturesTarget, { width: 16, height: 8 })
  assert.equal(texturesTarget.width, 16)
  assert.equal(texturesTarget.height, 8)
  assert.equal(texturesTarget.textures[0].image.data, texturesTarget.data)
  assert.equal(texturesTarget.textures[0].source.data, texturesTarget.textures[0].image)

  const singleAttachmentMrtTarget = { isWebGLMultipleRenderTargets: true, textures: [{}] }
  renderToTarget(scene, makeCamera(), singleAttachmentMrtTarget, { width: 8, height: 4 })
  assert.equal(singleAttachmentMrtTarget.width, 8)
  assert.equal(singleAttachmentMrtTarget.height, 4)
  assert.equal(singleAttachmentMrtTarget.textures[0].image.data, singleAttachmentMrtTarget.data)
  assert.equal(singleAttachmentMrtTarget.textures[0].source.data, singleAttachmentMrtTarget.textures[0].image)
})

test('renderToTarget and options.target populate depthTexture with normalized RGBA depth', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const near = new THREE.Mesh(
    new THREE.PlaneGeometry(0.9, 1.2),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  )
  near.position.set(-0.7, 0, 1)

  const far = new THREE.Mesh(
    new THREE.PlaneGeometry(0.9, 1.2),
    new THREE.MeshBasicMaterial({ color: 0x0000ff }),
  )
  far.position.set(0.7, 0, -3)
  scene.add(near, far)

  const camera = new THREE.OrthographicCamera(-2, 2, 2, -2, 0.1, 10)
  camera.position.set(0, 0, 5)
  camera.lookAt(0, 0, 0)

  const depthTexture = { source: { data: {} } }
  const target = { texture: {}, depthTexture }
  renderToTarget(scene, camera, target, { width: 64, height: 64 })

  assert.equal(target.data.length, 64 * 64 * 4)
  assert.equal(target.texture.image.data, target.data)
  assert.equal(depthTexture.image.data.length, 64 * 64 * 4)
  assert.notStrictEqual(depthTexture.image.data, target.data)
  assert.equal(depthTexture.source.data.data, depthTexture.image.data)
  assert.equal(depthTexture.source.data.width, 64)
  assert.equal(depthTexture.source.data.height, 64)

  const leftDepth = meanRegion(depthTexture.image.data, 64, 64, 18, 26, 26, 38)
  const rightDepth = meanRegion(depthTexture.image.data, 64, 64, 38, 26, 46, 38)
  assert.ok(
    leftDepth.r > rightDepth.r + 80,
    `near depth should be brighter than far depth (${leftDepth.r} vs ${rightDepth.r})`,
  )
  assert.ok(Math.abs(leftDepth.r - leftDepth.g) <= 1, 'depth red and green channels should match')
  assert.ok(Math.abs(leftDepth.r - leftDepth.b) <= 1, 'depth red and blue channels should match')

  const optionsDepthTexture = { source: { data: {} } }
  const optionsTarget = { texture: {}, depthTexture: optionsDepthTexture }
  const returned = renderRgba(scene, camera, { width: 64, height: 64, target: optionsTarget })
  assert.equal(returned, optionsTarget.data, 'options.target should return target.data')
  assert.equal(optionsTarget.texture.image.data, optionsTarget.data)
  assert.equal(optionsDepthTexture.image.data.length, 64 * 64 * 4)
  assert.notStrictEqual(optionsDepthTexture.image.data, optionsTarget.data)
  assert.equal(optionsDepthTexture.source.data.data, optionsDepthTexture.image.data)
  assert.equal(optionsDepthTexture.source.data.width, 64)
  assert.equal(optionsDepthTexture.source.data.height, 64)

  const optionsLeftDepth = meanRegion(optionsDepthTexture.image.data, 64, 64, 18, 26, 26, 38)
  const optionsRightDepth = meanRegion(optionsDepthTexture.image.data, 64, 64, 38, 26, 46, 38)
  assert.ok(
    optionsLeftDepth.r > optionsRightDepth.r + 80,
    `options.target near depth should be brighter than far depth (${optionsLeftDepth.r} vs ${optionsRightDepth.r})`,
  )
})

test('renderToTarget populates FloatType depthTexture with normalized scalar depth', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const near = new THREE.Mesh(
    new THREE.PlaneGeometry(0.9, 1.2),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  )
  near.position.set(-0.7, 0, 1)

  const far = new THREE.Mesh(
    new THREE.PlaneGeometry(0.9, 1.2),
    new THREE.MeshBasicMaterial({ color: 0x0000ff }),
  )
  far.position.set(0.7, 0, -3)
  scene.add(near, far)

  const camera = new THREE.OrthographicCamera(-2, 2, 2, -2, 0.1, 10)
  camera.position.set(0, 0, 5)
  camera.lookAt(0, 0, 0)

  const depthTexture = { type: THREE.FloatType, source: { data: {} } }
  renderToTarget(scene, camera, { texture: {}, depthTexture }, { width: 64, height: 64 })

  assert.ok(depthTexture.image.data instanceof Float32Array, 'FloatType depthTexture should receive Float32Array data')
  assert.equal(depthTexture.image.data.length, 64 * 64)
  assert.equal(depthTexture.source.data.data, depthTexture.image.data)
  assert.equal(depthTexture.source.data.width, 64)
  assert.equal(depthTexture.source.data.height, 64)

  const leftDepth = meanScalarRegion(depthTexture.image.data, 64, 64, 18, 26, 26, 38)
  const rightDepth = meanScalarRegion(depthTexture.image.data, 64, 64, 38, 26, 46, 38)
  assert.ok(leftDepth > rightDepth + 0.3, `near float depth should be greater than far depth (${leftDepth} vs ${rightDepth})`)
  assert.ok(leftDepth <= 1 && rightDepth >= 0, `float depth values should be normalized (${leftDepth}, ${rightDepth})`)
})

test('renderToTarget populates HalfFloatType depthTexture with normalized scalar depth', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const near = new THREE.Mesh(
    new THREE.PlaneGeometry(0.9, 1.2),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  )
  near.position.set(-0.7, 0, 1)

  const far = new THREE.Mesh(
    new THREE.PlaneGeometry(0.9, 1.2),
    new THREE.MeshBasicMaterial({ color: 0x0000ff }),
  )
  far.position.set(0.7, 0, -3)
  scene.add(near, far)

  const camera = new THREE.OrthographicCamera(-2, 2, 2, -2, 0.1, 10)
  camera.position.set(0, 0, 5)
  camera.lookAt(0, 0, 0)

  const depthTexture = { type: THREE.HalfFloatType, source: { data: {} } }
  renderToTarget(scene, camera, { texture: {}, depthTexture }, { width: 64, height: 64 })

  assert.ok(depthTexture.image.data instanceof Uint16Array, 'HalfFloatType depthTexture should receive Uint16Array half-float data')
  assert.equal(depthTexture.image.data.length, 64 * 64)
  assert.equal(depthTexture.source.data.data, depthTexture.image.data)
  assert.equal(depthTexture.source.data.width, 64)
  assert.equal(depthTexture.source.data.height, 64)

  const leftDepth = halfFloatToNumber(Math.round(meanScalarRegion(depthTexture.image.data, 64, 64, 18, 26, 26, 38)))
  const rightDepth = halfFloatToNumber(Math.round(meanScalarRegion(depthTexture.image.data, 64, 64, 38, 26, 46, 38)))
  assert.ok(leftDepth > rightDepth + 0.3, `near half-float depth should be greater than far depth (${leftDepth} vs ${rightDepth})`)
  assert.ok(leftDepth <= 1 && rightDepth >= 0, `half-float depth values should be normalized (${leftDepth}, ${rightDepth})`)
})

test('renderToTarget populates UnsignedByteType depthTexture with normalized scalar depth', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const near = new THREE.Mesh(
    new THREE.PlaneGeometry(0.9, 1.2),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  )
  near.position.set(-0.7, 0, 1)

  const far = new THREE.Mesh(
    new THREE.PlaneGeometry(0.9, 1.2),
    new THREE.MeshBasicMaterial({ color: 0x0000ff }),
  )
  far.position.set(0.7, 0, -3)
  scene.add(near, far)

  const camera = new THREE.OrthographicCamera(-2, 2, 2, -2, 0.1, 10)
  camera.position.set(0, 0, 5)
  camera.lookAt(0, 0, 0)

  const depthTexture = { type: THREE.UnsignedByteType, source: { data: {} } }
  renderToTarget(scene, camera, { texture: {}, depthTexture }, { width: 64, height: 64 })

  assert.ok(depthTexture.image.data instanceof Uint8Array, 'UnsignedByteType depthTexture should receive Uint8Array data')
  assert.equal(depthTexture.image.data.length, 64 * 64)
  assert.equal(depthTexture.source.data.data, depthTexture.image.data)
  assert.equal(depthTexture.source.data.width, 64)
  assert.equal(depthTexture.source.data.height, 64)

  const leftDepth = meanScalarRegion(depthTexture.image.data, 64, 64, 18, 26, 26, 38)
  const rightDepth = meanScalarRegion(depthTexture.image.data, 64, 64, 38, 26, 46, 38)
  assert.ok(leftDepth > rightDepth + 80, `near byte depth should be greater than far depth (${leftDepth} vs ${rightDepth})`)
})

test('renderToTarget populates UnsignedShortType depthTexture with normalized scalar depth', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const near = new THREE.Mesh(
    new THREE.PlaneGeometry(0.9, 1.2),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  )
  near.position.set(-0.7, 0, 1)

  const far = new THREE.Mesh(
    new THREE.PlaneGeometry(0.9, 1.2),
    new THREE.MeshBasicMaterial({ color: 0x0000ff }),
  )
  far.position.set(0.7, 0, -3)
  scene.add(near, far)

  const camera = new THREE.OrthographicCamera(-2, 2, 2, -2, 0.1, 10)
  camera.position.set(0, 0, 5)
  camera.lookAt(0, 0, 0)

  const depthTexture = { type: THREE.UnsignedShortType, source: { data: {} } }
  renderToTarget(scene, camera, { texture: {}, depthTexture }, { width: 64, height: 64 })

  assert.ok(depthTexture.image.data instanceof Uint16Array, 'UnsignedShortType depthTexture should receive Uint16Array data')
  assert.equal(depthTexture.image.data.length, 64 * 64)
  assert.equal(depthTexture.source.data.data, depthTexture.image.data)
  assert.equal(depthTexture.source.data.width, 64)
  assert.equal(depthTexture.source.data.height, 64)

  const leftDepth = meanScalarRegion(depthTexture.image.data, 64, 64, 18, 26, 26, 38)
  const rightDepth = meanScalarRegion(depthTexture.image.data, 64, 64, 38, 26, 46, 38)
  assert.ok(leftDepth > rightDepth + 20000, `near ushort depth should be greater than far depth (${leftDepth} vs ${rightDepth})`)
  assert.ok(leftDepth <= 0xffff && rightDepth >= 0, `ushort depth values should be normalized (${leftDepth}, ${rightDepth})`)
})

test('renderToTarget populates THREE.DepthTexture with unsigned scalar depth', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const near = new THREE.Mesh(
    new THREE.PlaneGeometry(0.9, 1.2),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  )
  near.position.set(-0.7, 0, 1)

  const far = new THREE.Mesh(
    new THREE.PlaneGeometry(0.9, 1.2),
    new THREE.MeshBasicMaterial({ color: 0x0000ff }),
  )
  far.position.set(0.7, 0, -3)
  scene.add(near, far)

  const camera = new THREE.OrthographicCamera(-2, 2, 2, -2, 0.1, 10)
  camera.position.set(0, 0, 5)
  camera.lookAt(0, 0, 0)

  const depthTexture = new THREE.DepthTexture(64, 64)
  renderToTarget(scene, camera, { texture: {}, depthTexture }, { width: 64, height: 64 })

  assert.ok(depthTexture.image.data instanceof Uint32Array, 'DepthTexture should receive Uint32Array data for UnsignedIntType')
  assert.equal(depthTexture.image.data.length, 64 * 64)
  assert.equal(depthTexture.source.data.data, depthTexture.image.data)

  const leftDepth = meanScalarRegion(depthTexture.image.data, 64, 64, 18, 26, 26, 38)
  const rightDepth = meanScalarRegion(depthTexture.image.data, 64, 64, 38, 26, 46, 38)
  assert.ok(leftDepth > rightDepth + 1_000_000_000, `near unsigned depth should be greater than far depth (${leftDepth} vs ${rightDepth})`)
})

test('renderToTarget populates UnsignedInt248Type depthTexture with depth24-stencil8-like data', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const near = new THREE.Mesh(
    new THREE.PlaneGeometry(0.9, 1.2),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  )
  near.position.set(-0.7, 0, 1)

  const far = new THREE.Mesh(
    new THREE.PlaneGeometry(0.9, 1.2),
    new THREE.MeshBasicMaterial({ color: 0x0000ff }),
  )
  far.position.set(0.7, 0, -3)
  scene.add(near, far)

  const camera = new THREE.OrthographicCamera(-2, 2, 2, -2, 0.1, 10)
  camera.position.set(0, 0, 5)
  camera.lookAt(0, 0, 0)

  const depthTexture = new THREE.DepthTexture(64, 64, THREE.UnsignedInt248Type)
  depthTexture.format = THREE.DepthStencilFormat
  renderToTarget(scene, camera, { texture: {}, depthTexture }, { width: 64, height: 64 })

  assert.ok(depthTexture.image.data instanceof Uint32Array, 'UnsignedInt248Type depthTexture should receive Uint32Array data')
  assert.equal(depthTexture.image.data.length, 64 * 64)
  assert.equal(depthTexture.source.data.data, depthTexture.image.data)
  assert.equal(depthTexture.source.data.width, 64)
  assert.equal(depthTexture.source.data.height, 64)

  for (let i = 0; i < depthTexture.image.data.length; i += 197) {
    assert.equal(depthTexture.image.data[i] & 0xff, 0, `stencil byte should be zero at ${i}`)
  }

  const leftDepth24 = meanScalarRegion(depthTexture.image.data, 64, 64, 18, 26, 26, 38) / 0x100
  const rightDepth24 = meanScalarRegion(depthTexture.image.data, 64, 64, 38, 26, 46, 38) / 0x100
  assert.ok(leftDepth24 > rightDepth24 + 1_000_000, `near depth24 should be greater than far depth (${leftDepth24} vs ${rightDepth24})`)
  assert.ok(leftDepth24 <= 0xffffff && rightDepth24 >= 0, `depth24 values should be normalized (${leftDepth24}, ${rightDepth24})`)
})

test('renderToTarget depthTexture honors scissor clipping', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(4, 4),
    new THREE.MeshBasicMaterial({ color: 0xffffff }),
  ))

  const camera = new THREE.OrthographicCamera(-2, 2, 2, -2, 0.1, 10)
  camera.position.set(0, 0, 5)
  camera.lookAt(0, 0, 0)

  const depthTexture = {}
  renderToTarget(scene, camera, { texture: {}, depthTexture }, {
    width: 64,
    height: 64,
    scissor: { x: 16, y: 16, width: 32, height: 32 },
  })
  assert.equal(depthTexture.source.data, depthTexture.image)
  assert.equal(depthTexture.source.data.data, depthTexture.image.data)
  assert.equal(depthTexture.source.data.width, 64)
  assert.equal(depthTexture.source.data.height, 64)

  const inside = meanRegion(depthTexture.image.data, 64, 64, 24, 24, 40, 40)
  const outsideLeft = meanRegion(depthTexture.image.data, 64, 64, 4, 24, 12, 40)
  const outsideTop = meanRegion(depthTexture.image.data, 64, 64, 24, 4, 40, 12)
  assert.ok(inside.r > 80, `scissored depth region should contain visible mesh depth (${inside.r})`)
  assert.ok(outsideLeft.r < 2, `left of scissor should keep background depth (${outsideLeft.r})`)
  assert.ok(outsideTop.r < 2, `above scissor should keep background depth (${outsideTop.r})`)
})

test('renderToTarget and options.target use target viewport and scissor fields', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 1)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(4, 4),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  function makeTarget() {
    const depthTexture = {}
    return {
      depthTexture,
      target: {
        texture: {},
        depthTexture,
        viewport: new THREE.Vector4(16, 16, 40, 32),
        scissor: new THREE.Vector4(24, 20, 24, 24),
        scissorTest: true,
      },
    }
  }

  function assertTargetClip(label, target, depthTexture) {
    const inside = meanRegion(target.data, 64, 64, 30, 26, 42, 38)
    const viewportOutside = meanRegion(target.data, 64, 64, 4, 26, 12, 38)
    const scissorOutside = meanRegion(target.data, 64, 64, 18, 26, 22, 38)
    assert.ok(inside.r > inside.b + 80, `${label} target viewport/scissor region should contain the red mesh (${inside.r} vs ${inside.b})`)
    assert.ok(viewportOutside.b > viewportOutside.r + 80, `${label} outside target viewport should retain blue background (${viewportOutside.b} vs ${viewportOutside.r})`)
    assert.ok(scissorOutside.b > scissorOutside.r + 80, `${label} outside target scissor should retain blue background (${scissorOutside.b} vs ${scissorOutside.r})`)

    const depthInside = meanRegion(depthTexture.image.data, 64, 64, 30, 26, 42, 38)
    const depthOutside = meanRegion(depthTexture.image.data, 64, 64, 18, 26, 22, 38)
    assert.ok(depthInside.r > 0, `${label} target viewport/scissor depth should contain visible geometry (${depthInside.r})`)
    assert.ok(depthOutside.r < 2, `${label} outside target scissor depth should keep background depth (${depthOutside.r})`)
  }

  const direct = makeTarget()
  renderToTarget(scene, camera, direct.target, { width: 64, height: 64 })
  assertTargetClip('renderToTarget', direct.target, direct.depthTexture)

  const options = makeTarget()
  const returned = renderRgba(scene, camera, { width: 64, height: 64, target: options.target })
  assert.equal(returned, options.target.data, 'options.target should return target.data')
  assertTargetClip('options.target', options.target, options.depthTexture)
})

test('renderToTarget depthTexture preserves alphaMap cutouts', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const alphaMap = rgbaTexture([
    255, 0, 255, 255,
    255, 255, 255, 255,
  ], 2, 1)
  alphaMap.magFilter = THREE.NearestFilter
  alphaMap.minFilter = THREE.NearestFilter
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(3.2, 2.4),
    new THREE.MeshBasicMaterial({
      color: 0xffffff,
      alphaMap,
      alphaTest: 0.5,
    }),
  ))

  const camera = new THREE.OrthographicCamera(-2, 2, 2, -2, 0.1, 10)
  camera.position.set(0, 0, 5)
  camera.lookAt(0, 0, 0)

  const depthTexture = {}
  renderToTarget(scene, camera, { texture: {}, depthTexture }, { width: 64, height: 64 })

  const discarded = meanRegion(depthTexture.image.data, 64, 64, 14, 26, 24, 38)
  const visible = meanRegion(depthTexture.image.data, 64, 64, 40, 26, 50, 38)
  assert.ok(discarded.r < 2, `alphaMap cutout should keep background depth (${discarded.r})`)
  assert.ok(visible.r > 80, `opaque alphaMap region should write visible mesh depth (${visible.r})`)
})

test('renderToTarget depthTexture preserves base texture alpha cutouts', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const map = rgbaTexture([
    255, 255, 255, 0,
    255, 255, 255, 255,
  ], 2, 1)
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(3.2, 2.4),
    new THREE.MeshBasicMaterial({
      color: 0xffffff,
      map,
      alphaTest: 0.5,
    }),
  ))

  const camera = new THREE.OrthographicCamera(-2, 2, 2, -2, 0.1, 10)
  camera.position.set(0, 0, 5)
  camera.lookAt(0, 0, 0)

  const depthTexture = {}
  renderToTarget(scene, camera, { texture: {}, depthTexture }, { width: 64, height: 64 })

  const discarded = meanRegion(depthTexture.image.data, 64, 64, 14, 26, 24, 38)
  const visible = meanRegion(depthTexture.image.data, 64, 64, 40, 26, 50, 38)
  assert.ok(discarded.r < 2, `base texture cutout should keep background depth (${discarded.r})`)
  assert.ok(visible.r > 80, `opaque base texture region should write visible mesh depth (${visible.r})`)
})

test('renderToTarget depthTexture preserves alphaHash cutouts', () => {
  function renderAlphaHashDepth(alphaHash) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({
        alphaHash,
        color: 0xffffff,
        opacity: alphaHash ? 0.35 : 1,
      }),
    ))

    const camera = new THREE.OrthographicCamera(-2, 2, 2, -2, 0.1, 10)
    camera.position.set(0, 0, 5)
    camera.lookAt(0, 0, 0)

    const depthTexture = {}
    renderToTarget(scene, camera, { texture: {}, depthTexture }, { width: 64, height: 64 })
    return depthTexture.image.data
  }

  const opaque = renderAlphaHashDepth(false)
  const hashed = renderAlphaHashDepth(true)
  const depthPixel = (r) => r > 2
  const opaquePixels = countRegionPixels(opaque, 64, 64, 20, 20, 44, 44, depthPixel)
  const hashedPixels = countRegionPixels(hashed, 64, 64, 20, 20, 44, 44, depthPixel)

  assert.ok(opaquePixels > 520, `opaque target depth should fill the sampled region (${opaquePixels})`)
  assert.ok(hashedPixels > 40, `alphaHash target depth should retain some depth pixels (${hashedPixels})`)
  assert.ok(hashedPixels < opaquePixels - 120, `alphaHash target depth should discard depth pixels (${hashedPixels} vs ${opaquePixels})`)
})

test('renderToTarget depthTexture honors transparent default depthWrite', () => {
  function renderTransparentDepth(depthWrite) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    const material = new THREE.MeshBasicMaterial({
      color: 0xff0000,
      opacity: 0.5,
      transparent: true,
    })
    if (depthWrite !== undefined) material.depthWrite = depthWrite
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    const depthTexture = {}
    renderToTarget(scene, camera, { texture: {}, depthTexture }, { width: 64, height: 64 })
    return meanRegion(depthTexture.image.data, 64, 64, 24, 24, 40, 40)
  }

  const defaultDepth = renderTransparentDepth(undefined)
  const disabledDepth = renderTransparentDepth(false)

  assert.ok(defaultDepth.r > disabledDepth.r + 0.5, `transparent default depthWrite should populate target depth (${defaultDepth.r} vs ${disabledDepth.r})`)
  assert.ok(disabledDepth.r < 0.5, `transparent depthWrite=false should leave target depth clear (${disabledDepth.r})`)
})

test('renderToTarget color textures honor typed readback requests', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const center = ((32 * 64) + 32) * 4
  const options = { width: 64, height: 64, outputColorSpace: THREE.LinearSRGBColorSpace }

  const redTarget = { texture: { format: THREE.RedFormat } }
  renderToTarget(scene, camera, redTarget, options)
  const redData = redTarget.texture.image.data
  const redCenter = (32 * 64) + 32
  assert.ok(redData instanceof Uint8Array, 'RedFormat color target should receive Uint8Array data')
  assert.equal(redData.length, 64 * 64, 'RedFormat color target should receive one channel per pixel')
  assert.ok(redData[redCenter] > 128, `RedFormat red channel should keep the source red (${redData[redCenter]})`)

  const alphaScene = new THREE.Scene()
  alphaScene.background = new THREE.Color(0, 0, 1)
  alphaScene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0x0000ff }),
  ))
  const alphaTarget = { texture: { format: THREE.AlphaFormat } }
  renderToTarget(alphaScene, camera, alphaTarget, options)
  const alphaData = alphaTarget.texture.image.data
  assert.ok(alphaData instanceof Uint8Array, 'AlphaFormat color target should receive Uint8Array data')
  assert.equal(alphaData.length, 64 * 64, 'AlphaFormat color target should receive one channel per pixel')
  assert.ok(alphaData[redCenter] > 250, `AlphaFormat should extract opaque alpha instead of red (${alphaData[redCenter]})`)

  const floatTarget = { texture: { type: THREE.FloatType } }
  renderToTarget(scene, camera, floatTarget, options)
  const floatData = floatTarget.texture.image.data
  assert.ok(floatData instanceof Float32Array, 'FloatType color target should receive Float32Array data')
  assert.ok(Buffer.isBuffer(floatTarget.data), 'target.data should remain raw RGBA8 for compatibility')
  assert.ok(floatData[center] > 0.5, `FloatType red channel should be normalized (${floatData[center]})`)
  assert.ok(floatData[center + 1] < 0.05, `FloatType green channel should stay near zero (${floatData[center + 1]})`)
  assert.ok(floatData[center + 3] > 0.99, `FloatType alpha channel should stay opaque (${floatData[center + 3]})`)

  const byteTarget = { texture: { type: THREE.ByteType } }
  renderToTarget(scene, camera, byteTarget, options)
  const byteData = byteTarget.texture.image.data
  assert.ok(byteData instanceof Int8Array, 'ByteType color target should receive Int8Array data')
  assert.ok(byteData[center] > 63, `ByteType red channel should be normalized (${byteData[center]})`)
  assert.ok(byteData[center + 1] < 8, `ByteType green channel should stay near zero (${byteData[center + 1]})`)
  assert.ok(byteData[center + 3] > 120, `ByteType alpha channel should stay opaque (${byteData[center + 3]})`)

  const rgFloatTarget = { texture: { format: THREE.RGFormat, type: THREE.FloatType } }
  renderToTarget(scene, camera, rgFloatTarget, options)
  const rgFloatData = rgFloatTarget.texture.image.data
  const rgCenter = ((32 * 64) + 32) * 2
  assert.ok(rgFloatData instanceof Float32Array, 'RGFormat FloatType color target should receive Float32Array data')
  assert.equal(rgFloatData.length, 64 * 64 * 2, 'RGFormat color target should receive two channels per pixel')
  assert.ok(rgFloatData[rgCenter] > 0.5, `RGFormat FloatType red channel should be normalized (${rgFloatData[rgCenter]})`)
  assert.ok(rgFloatData[rgCenter + 1] < 0.05, `RGFormat FloatType green channel should stay near zero (${rgFloatData[rgCenter + 1]})`)

  const ushortTarget = { texture: { type: THREE.UnsignedShortType } }
  renderToTarget(scene, camera, ushortTarget, options)
  const ushortData = ushortTarget.texture.image.data
  assert.ok(ushortData instanceof Uint16Array, 'UnsignedShortType color target should receive Uint16Array data')
  assert.ok(ushortData[center] > 0x8000, `UnsignedShortType red channel should be normalized (${ushortData[center]})`)
  assert.ok(ushortData[center + 1] < 0x1000, `UnsignedShortType green channel should stay near zero (${ushortData[center + 1]})`)
  assert.ok(ushortData[center + 3] > 0xff00, `UnsignedShortType alpha channel should stay opaque (${ushortData[center + 3]})`)

  const shortTarget = { texture: { type: THREE.ShortType } }
  renderToTarget(scene, camera, shortTarget, options)
  const shortData = shortTarget.texture.image.data
  assert.ok(shortData instanceof Int16Array, 'ShortType color target should receive Int16Array data')
  assert.ok(shortData[center] > 0x4000, `ShortType red channel should be normalized (${shortData[center]})`)
  assert.ok(shortData[center + 1] < 0x1000, `ShortType green channel should stay near zero (${shortData[center + 1]})`)
  assert.ok(shortData[center + 3] > 0x7f00, `ShortType alpha channel should stay opaque (${shortData[center + 3]})`)

  const rgbUshortTarget = { texture: { format: THREE.RGBFormat, type: THREE.UnsignedShortType } }
  renderToTarget(scene, camera, rgbUshortTarget, options)
  const rgbUshortData = rgbUshortTarget.texture.image.data
  const rgbCenter = ((32 * 64) + 32) * 3
  assert.ok(rgbUshortData instanceof Uint16Array, 'RGBFormat UnsignedShortType color target should receive Uint16Array data')
  assert.equal(rgbUshortData.length, 64 * 64 * 3, 'RGBFormat color target should receive three channels per pixel')
  assert.ok(rgbUshortData[rgbCenter] > 0x8000, `RGBFormat red channel should be normalized (${rgbUshortData[rgbCenter]})`)
  assert.ok(rgbUshortData[rgbCenter + 1] < 0x1000, `RGBFormat green channel should stay near zero (${rgbUshortData[rgbCenter + 1]})`)
  assert.ok(rgbUshortData[rgbCenter + 2] < 0x1000, `RGBFormat blue channel should stay near zero (${rgbUshortData[rgbCenter + 2]})`)

  const packed4444Target = { texture: { type: THREE.UnsignedShort4444Type } }
  renderToTarget(scene, camera, packed4444Target, options)
  const packed4444Data = packed4444Target.texture.image.data
  const packed4444 = packed4444Data[redCenter]
  assert.ok(packed4444Data instanceof Uint16Array, 'UnsignedShort4444Type color target should receive Uint16Array data')
  assert.ok(((packed4444 >> 12) & 0xf) > 7, `UnsignedShort4444Type red channel should be packed (${packed4444.toString(16)})`)
  assert.ok(((packed4444 >> 8) & 0xf) < 2, `UnsignedShort4444Type green channel should stay near zero (${packed4444.toString(16)})`)
  assert.ok(((packed4444 >> 4) & 0xf) < 2, `UnsignedShort4444Type blue channel should stay near zero (${packed4444.toString(16)})`)
  assert.equal(packed4444 & 0xf, 0xf, `UnsignedShort4444Type alpha channel should stay opaque (${packed4444.toString(16)})`)

  const packed5551Target = { texture: { type: THREE.UnsignedShort5551Type } }
  renderToTarget(scene, camera, packed5551Target, options)
  const packed5551Data = packed5551Target.texture.image.data
  const packed5551 = packed5551Data[redCenter]
  assert.ok(packed5551Data instanceof Uint16Array, 'UnsignedShort5551Type color target should receive Uint16Array data')
  assert.ok(((packed5551 >> 11) & 0x1f) > 15, `UnsignedShort5551Type red channel should be packed (${packed5551.toString(16)})`)
  assert.ok(((packed5551 >> 6) & 0x1f) < 2, `UnsignedShort5551Type green channel should stay near zero (${packed5551.toString(16)})`)
  assert.ok(((packed5551 >> 1) & 0x1f) < 2, `UnsignedShort5551Type blue channel should stay near zero (${packed5551.toString(16)})`)
  assert.equal(packed5551 & 0x1, 1, `UnsignedShort5551Type alpha channel should stay opaque (${packed5551.toString(16)})`)

  const rgb9e5Target = { texture: { type: THREE.UnsignedInt5999Type } }
  renderToTarget(scene, camera, rgb9e5Target, options)
  const rgb9e5Data = rgb9e5Target.texture.image.data
  const rgb9e5 = rgb9e5Data[redCenter]
  const rgb9e5Scale = 2 ** (((rgb9e5 >>> 27) & 0x1f) - 24)
  const rgb9e5Red = (rgb9e5 & 0x1ff) * rgb9e5Scale
  const rgb9e5Green = ((rgb9e5 >>> 9) & 0x1ff) * rgb9e5Scale
  const rgb9e5Blue = ((rgb9e5 >>> 18) & 0x1ff) * rgb9e5Scale
  assert.ok(rgb9e5Data instanceof Uint32Array, 'UnsignedInt5999Type color target should receive Uint32Array data')
  assert.ok(rgb9e5Red > 0.5, `UnsignedInt5999Type red channel should be packed (${rgb9e5Red})`)
  assert.ok(rgb9e5Green < 0.05, `UnsignedInt5999Type green channel should stay near zero (${rgb9e5Green})`)
  assert.ok(rgb9e5Blue < 0.05, `UnsignedInt5999Type blue channel should stay near zero (${rgb9e5Blue})`)

  const uintTarget = { texture: { type: THREE.UnsignedIntType } }
  renderToTarget(scene, camera, uintTarget, options)
  const uintData = uintTarget.texture.image.data
  assert.ok(uintData instanceof Uint32Array, 'UnsignedIntType color target should receive Uint32Array data')
  assert.ok(uintData[center] > 0x80000000, `UnsignedIntType red channel should be normalized (${uintData[center]})`)
  assert.ok(uintData[center + 1] < 0x10000000, `UnsignedIntType green channel should stay near zero (${uintData[center + 1]})`)
  assert.ok(uintData[center + 3] > 0xff000000, `UnsignedIntType alpha channel should stay opaque (${uintData[center + 3]})`)

  const intTarget = { texture: { type: THREE.IntType } }
  renderToTarget(scene, camera, intTarget, options)
  const intData = intTarget.texture.image.data
  assert.ok(intData instanceof Int32Array, 'IntType color target should receive Int32Array data')
  assert.ok(intData[center] > 0x40000000, `IntType red channel should be normalized (${intData[center]})`)
  assert.ok(intData[center + 1] < 0x10000000, `IntType green channel should stay near zero (${intData[center + 1]})`)
  assert.ok(intData[center + 3] > 0x7f000000, `IntType alpha channel should stay opaque (${intData[center + 3]})`)

  const halfTarget = { texture: { type: THREE.HalfFloatType } }
  renderToTarget(scene, camera, halfTarget, options)
  const halfData = halfTarget.texture.image.data
  assert.ok(halfData instanceof Uint16Array, 'HalfFloatType color target should receive Uint16Array half-float data')
  const halfRed = halfFloatToNumber(halfData[center])
  const halfGreen = halfFloatToNumber(halfData[center + 1])
  const halfAlpha = halfFloatToNumber(halfData[center + 3])
  assert.ok(halfRed > 0.5, `HalfFloatType red channel should be normalized (${halfRed})`)
  assert.ok(halfGreen < 0.05, `HalfFloatType green channel should stay near zero (${halfGreen})`)
  assert.ok(halfAlpha > 0.99, `HalfFloatType alpha channel should stay opaque (${halfAlpha})`)
})

test('single-attachment target array paths honor typed color readback requests', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const options = { width: 32, height: 32, outputColorSpace: THREE.LinearSRGBColorSpace }
  const center = ((16 * 32) + 16) * 2
  const cases = [
    ['target.texture array', { texture: [{ format: THREE.RGFormat, type: THREE.FloatType }] }, (target) => target.texture[0]],
    ['target.textures array', { textures: [{ format: THREE.RGFormat, type: THREE.FloatType }] }, (target) => target.textures[0]],
    ['single-attachment MRT target', { isWebGLMultipleRenderTargets: true, textures: [{ format: THREE.RGFormat, type: THREE.FloatType }] }, (target) => target.textures[0]],
  ]

  for (const [label, target, colorTexture] of cases) {
    renderToTarget(scene, camera, target, options)
    const data = colorTexture(target).image.data
    assert.ok(Buffer.isBuffer(target.data), `${label} top-level target.data should remain RGBA8`)
    assert.ok(data instanceof Float32Array, `${label} should receive Float32Array data`)
    assert.equal(data.length, 32 * 32 * 2, `${label} should receive two channels per pixel`)
    assert.equal(colorTexture(target).source.data.data, data, `${label} source should reference typed data`)
    assert.ok(data[center] > 0.5, `${label} red channel should be normalized (${data[center]})`)
    assert.ok(data[center + 1] < 0.05, `${label} green channel should stay near zero (${data[center + 1]})`)
  }

  const optionsTarget = { textures: [{ format: THREE.RGFormat, type: THREE.FloatType }] }
  const returned = renderRgba(scene, camera, { ...options, target: optionsTarget })
  const data = optionsTarget.textures[0].image.data
  assert.equal(optionsTarget.data, returned)
  assert.ok(data instanceof Float32Array, 'options.target target.textures[0] should receive Float32Array data')
  assert.equal(data.length, 32 * 32 * 2)
  assert.ok(data[center] > 0.5, `options.target red channel should be normalized (${data[center]})`)
  assert.ok(data[center + 1] < 0.05, `options.target green channel should stay near zero (${data[center + 1]})`)
})

test('MSAA sampleCount 4 resolves antialiased color output and render targets', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute([
    -1, -1, 0,
    1, -1, 0,
    -1, 1, 0,
  ], 3))
  geometry.setIndex([0, 1, 2])
  scene.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: 0xffffff })))

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const single = renderRgba(scene, camera, {
    width: 64,
    height: 64,
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })
  const msaa = renderRgba(scene, camera, {
    width: 64,
    height: 64,
    sampleCount: 4,
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })
  const msaaSamplesAlias = renderRgba(scene, camera, {
    width: 64,
    height: 64,
    samples: 4,
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  function intermediateCoverage(rgba) {
    return countRegionPixels(rgba, 64, 64, 0, 0, 64, 64, (r, g, b) => {
      return r > 20 && r < 180 && Math.abs(r - g) < 3 && Math.abs(r - b) < 3
    })
  }

  const singleCoverage = intermediateCoverage(single)
  const msaaCoverage = intermediateCoverage(msaa)
  const msaaAliasCoverage = intermediateCoverage(msaaSamplesAlias)
  assert.ok(msaaCoverage > singleCoverage + 20, `4x MSAA should add resolved edge coverage (${msaaCoverage} vs ${singleCoverage})`)
  assert.ok(msaaAliasCoverage > singleCoverage + 20, `4x MSAA samples alias should add resolved edge coverage (${msaaAliasCoverage} vs ${singleCoverage})`)

  for (const [label, target] of [
    ['target.samples', { texture: {}, samples: 4 }],
    ['target.sampleCount', { texture: {}, sampleCount: 4 }],
  ]) {
    renderToTarget(scene, camera, target, {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    })
    assert.equal(target.data.length, 64 * 64 * 4, `${label} target should receive RGBA8 data`)
    assert.equal(target.texture.image.data, target.data, `${label} texture should reference target data`)
    const targetCoverage = intermediateCoverage(target.data)
    assert.ok(
      targetCoverage > singleCoverage + 20,
      `${label} 4x MSAA should add resolved target edge coverage (${targetCoverage} vs ${singleCoverage})`,
    )
  }

  for (const [label, target] of [
    ['options.target.samples', { texture: {}, samples: 4 }],
    ['options.target.sampleCount', { texture: {}, sampleCount: 4 }],
  ]) {
    const returned = renderRgba(scene, camera, {
      width: 64,
      height: 64,
      target,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    })
    assert.equal(returned, target.data, `${label} should return target.data`)
    assert.equal(target.texture.image.data, target.data, `${label} texture should reference target data`)
    const targetCoverage = intermediateCoverage(target.data)
    assert.ok(
      targetCoverage > singleCoverage + 20,
      `${label} 4x MSAA should add resolved target edge coverage (${targetCoverage} vs ${singleCoverage})`,
    )
  }
})

test('unsupported render target MRT and invalid MSAA requests fail clearly', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.1, 0.1, 0.1)
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial({ color: 0x00ffaa })))
  const camera = makeCamera()

  assert.throws(
    () => renderToTarget(scene, camera, null, { width: 32, height: 32 }),
    /target must be a target-like object/i,
  )
  assert.throws(
    () => renderToTarget(scene, camera, [], { width: 32, height: 32 }),
    /target must be a target-like object/i,
  )
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32, target: 'bad' }),
    /options\.target must be a target-like object/i,
  )
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32, target: [] }),
    /options\.target must be a target-like object/i,
  )

  const targetCases = [
    [{ image: 'bad' }, /target\.image must be an image-like object/i, 'target image container'],
    [{ texture: 'bad' }, /target\.texture must be a texture-like object/i, 'color texture container'],
    [{ texture: [] }, /target\.texture must contain one texture-like object/i, 'empty texture array'],
    [{ texture: ['bad'] }, /target\.texture\[0\] must be a texture-like object/i, 'texture array element'],
    [{ textures: 'bad' }, /target\.textures must be an array of texture-like objects/i, 'textures container'],
    [{ textures: [] }, /target\.textures must contain one texture-like object/i, 'empty textures array'],
    [{ textures: ['bad'] }, /target\.textures\[0\] must be a texture-like object/i, 'textures array element'],
    [{ depthTexture: 'bad' }, /target\.depthTexture must be a texture-like object/i, 'depth texture container'],
    [{ texture: { image: 'bad' } }, /target\.texture\.image must be an image-like object/i, 'texture image container'],
    [{ texture: { mipmaps: ['bad'] } }, /target\.texture\.mipmaps\[0\] must be an image-like object/i, 'texture mipmap container'],
    [{ texture: { source: 'bad' } }, /target\.texture\.source must be a source-like object/i, 'texture source container'],
    [{ texture: { source: { data: 'bad' } } }, /target\.texture\.source\.data must be an image-like object/i, 'texture source data container'],
    [{ texture: [{}, {}] }, /Multiple render target color attachments.*not supported/i, 'texture array'],
    [{ textures: [{}, {}] }, /Multiple render target color attachments.*not supported/i, 'textures array'],
    [{ texture: new THREE.DataArrayTexture(new Uint8Array([255, 0, 0, 255]), 1, 1, 1) }, /target color texture uses an array or 3D texture/i, 'color array texture'],
    [{ depthTexture: new THREE.Data3DTexture(new Uint8Array([255, 0, 0, 255]), 1, 1, 1) }, /target\.depthTexture uses an array or 3D texture/i, 'depth 3D texture'],
    [{ texture: { isCubeTexture: true } }, /target color texture uses a cube texture.*THREE\.CubeCamera/i, 'regular camera cube color texture'],
    [{ depthTexture: { isCubeTexture: true } }, /target\.depthTexture uses a cube texture.*THREE\.CubeCamera/i, 'regular camera cube depth texture'],
    [{ samples: 2 }, /MSAA sample count 2.*not supported/i, 'target samples'],
    [{ sampleCount: 8 }, /MSAA sample count 8.*not supported/i, 'target sampleCount'],
    [{ texture: { format: THREE.DepthFormat } }, /target color texture format .*not supported.*AlphaFormat.*RedFormat.*RGFormat.*RGBFormat.*RGBAFormat/i, 'color texture format'],
    [{ texture: { type: THREE.UnsignedInt248Type } }, /target color texture type .*not supported.*UnsignedByteType.*ByteType.*ShortType.*UnsignedShortType.*IntType.*UnsignedIntType.*HalfFloatType.*FloatType.*UnsignedShort4444Type.*UnsignedShort5551Type.*UnsignedInt5999Type/i, 'color texture type'],
    [{ depthTexture: { type: THREE.ByteType } }, /target\.depthTexture\.type .*not supported/i, 'depth texture type'],
    [{ depthTexture: { format: THREE.RGBAFormat } }, /target\.depthTexture\.format .*not supported/i, 'depth texture format'],
    [{ depthTexture: { type: THREE.FloatType, format: THREE.DepthStencilFormat } }, /DepthStencilFormat.*UnsignedInt248Type/i, 'depth-stencil format with scalar type'],
    [{ depthTexture: { type: THREE.UnsignedInt248Type, format: THREE.DepthFormat } }, /DepthFormat.*UnsignedInt248Type/i, 'depth format with packed depth-stencil type'],
  ]

  for (const [target, pattern, label] of targetCases) {
    assert.throws(
      () => renderToTarget(scene, camera, target, { width: 32, height: 32 }),
      pattern,
      label,
    )
  }

  const optionsTargetCases = [
    [{ texture: [{}, {}] }, /Multiple render target color attachments.*not supported/i, 'options.target texture array'],
    [{ texture: new THREE.DataArrayTexture(new Uint8Array([255, 0, 0, 255]), 1, 1, 1) }, /target color texture uses an array or 3D texture/i, 'options.target color array texture'],
    [{ sampleCount: 8 }, /MSAA sample count 8.*not supported/i, 'options.target sampleCount'],
    [{ texture: { format: THREE.DepthFormat } }, /target color texture format .*not supported.*AlphaFormat.*RedFormat.*RGFormat.*RGBFormat.*RGBAFormat/i, 'options.target color texture format'],
    [{ depthTexture: { type: THREE.ByteType } }, /target\.depthTexture\.type .*not supported/i, 'options.target depth texture type'],
  ]

  for (const [target, pattern, label] of optionsTargetCases) {
    assert.throws(
      () => renderRgba(scene, camera, { width: 32, height: 32, target }),
      pattern,
      label,
    )
  }

  for (const options of [{ samples: 2 }, { sampleCount: 8 }]) {
    assert.throws(
      () => renderRgba(scene, camera, { width: 32, height: 32, ...options }),
      /MSAA sample count .*not supported/i,
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

test('post-processing exposure contrast and grayscale controls modify output', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.25, 0.5, 0.75)
  const camera = makeCamera()
  const options = { width: 64, height: 64, outputColorSpace: THREE.LinearSRGBColorSpace }

  const base = meanRgba(renderRgba(scene, camera, options))
  const exposed = meanRgba(renderRgba(scene, camera, {
    ...options,
    postProcessing: { exposure: 1 },
  }))
  const contrasted = meanRgba(renderRgba(scene, camera, {
    ...options,
    postProcessing: { contrast: 2 },
  }))
  const grayscale = meanRgba(renderRgba(scene, camera, {
    ...options,
    postProcessing: { grayscale: true },
  }))

  assert.ok(exposed.r > base.r + 20, `exposure should brighten red (${exposed.r} vs ${base.r})`)
  assert.ok(exposed.g > base.g + 20, `exposure should brighten green (${exposed.g} vs ${base.g})`)
  assert.ok(contrasted.r < base.r - 20, `contrast should darken below-mid red (${contrasted.r} vs ${base.r})`)
  assert.ok(contrasted.b > base.b + 20, `contrast should brighten above-mid blue (${contrasted.b} vs ${base.b})`)
  assert.ok(Math.max(grayscale.r, grayscale.g, grayscale.b) - Math.min(grayscale.r, grayscale.g, grayscale.b) < 3, `boolean grayscale should equalize color channels (${grayscale.r}, ${grayscale.g}, ${grayscale.b})`)
})

test('post-processing enabled false bypasses configured effects', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.25, 0.5, 0.75)
  const camera = makeCamera()
  const options = { width: 32, height: 32, outputColorSpace: THREE.LinearSRGBColorSpace }

  const base = renderRgba(scene, camera, options)
  const disabled = renderRgba(scene, camera, {
    ...options,
    postProcessing: {
      enabled: false,
      exposure: 4,
      contrast: 4,
      grayscale: true,
      invert: true,
      saturation: 0,
      vignette: 1,
    },
  })

  assert.deepEqual(disabled, base, 'postProcessing.enabled=false should ignore configured effects')
})

test('invalid post-processing option values fail clearly', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(1, 0, 0)
  const camera = makeCamera()

  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32, postProcessing: [] }),
    /options\.postProcessing must be an object/i,
  )
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32, postProcessing: { exposure: Number.NaN } }),
    /options\.postProcessing\.exposure must be a finite number/i,
  )
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32, postProcessing: { vignette: -0.1 } }),
    /options\.postProcessing\.vignette must be between 0 and 1/i,
  )
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32, postProcessing: { invert: 'yes' } }),
    /options\.postProcessing\.invert must be a finite number or boolean/i,
  )
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32, postProcessing: { invert: 1.5 } }),
    /options\.postProcessing\.invert must be between 0 and 1/i,
  )
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32, postProcessing: { grayscale: -0.1 } }),
    /options\.postProcessing\.grayscale must be between 0 and 1/i,
  )
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32, postProcessing: { enabled: 'yes' } }),
    /options\.postProcessing\.enabled must be a boolean/i,
  )
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

test('invalid environment intensity values fail clearly', () => {
  const camera = makeCamera()

  const scene = new THREE.Scene()
  scene.environment = makeEnvironmentTexture()
  scene.environmentIntensity = Number.NaN
  scene.add(new THREE.Mesh(
    new THREE.SphereGeometry(1, 16, 16),
    new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0.2 }),
  ))
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /scene\.environmentIntensity must be a finite number/i,
  )

  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32, environmentIntensity: Number.POSITIVE_INFINITY }),
    /options\.environmentIntensity must be a finite number/i,
  )

  const probeScene = new THREE.Scene()
  probeScene.userData.headlessThreeRenderer = {
    reflectionProbe: {
      texture: makeEnvironmentTexture(),
      intensity: 'bright',
    },
  }
  probeScene.add(new THREE.Mesh(
    new THREE.SphereGeometry(1, 16, 16),
    new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0.2 }),
  ))
  assert.throws(
    () => renderRgba(probeScene, camera, { width: 32, height: 32 }),
    /reflectionProbe\.intensity must be a finite number/i,
  )
})

test('options.environmentIntensity overrides scene and reflection-probe intensity', () => {
  function addReflectivePlane(scene) {
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0.15 }),
    ))
  }

  function sampledRed(scene, options = {}) {
    return meanRegion(renderRgba(scene, camera, {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
      ...options,
    }), 64, 64, 24, 24, 40, 40).r
  }

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const scene = new THREE.Scene()
  addReflectivePlane(scene)
  scene.environment = makeEnvironmentTexture()
  scene.environmentIntensity = 0.15

  const sceneIntensity = sampledRed(scene)
  const optionIntensity = sampledRed(scene, { environmentIntensity: 4 })

  const probeScene = new THREE.Scene()
  addReflectivePlane(probeScene)
  probeScene.userData.headlessThreeRenderer = {
    reflectionProbe: {
      texture: makeEnvironmentTexture(),
      intensity: 0.15,
    },
  }
  const probeIntensity = sampledRed(probeScene)
  const optionProbeIntensity = sampledRed(probeScene, { environmentIntensity: 4 })

  assert.ok(
    optionIntensity > sceneIntensity + 35,
    `options.environmentIntensity should brighten scene IBL (${optionIntensity} vs ${sceneIntensity})`,
  )
  assert.ok(
    optionProbeIntensity > probeIntensity + 35,
    `options.environmentIntensity should brighten reflection-probe IBL (${optionProbeIntensity} vs ${probeIntensity})`,
  )
})

test('scene.environment intensity takes precedence over reflection-probe intensity', () => {
  function sampledRed(scene) {
    return meanRegion(renderRgba(scene, camera, {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 64, 64, 24, 24, 40, 40).r
  }

  function makeScene(withProbe) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.environment = makeEnvironmentTexture()
    scene.environmentIntensity = 0.15
    if (withProbe) {
      scene.userData.headlessThreeRenderer = {
        reflectionProbe: {
          texture: makeEnvironmentTexture(),
          intensity: 4,
        },
      }
    }
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0.15 }),
    ))
    return scene
  }

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const sceneOnly = sampledRed(makeScene(false))
  const withProbe = sampledRed(makeScene(true))
  assert.ok(
    Math.abs(sceneOnly - withProbe) <= 3,
    `scene.environment should ignore reflection-probe intensity when both are present (${sceneOnly} vs ${withProbe})`,
  )
})

test('invalid environment and background rotation values fail clearly', () => {
  const camera = makeCamera()

  const backgroundScene = new THREE.Scene()
  backgroundScene.background = splitEnvironmentTexture()
  backgroundScene.backgroundRotation = { x: 'left', y: 0, z: 0 }
  assert.throws(
    () => renderRgba(backgroundScene, camera, { width: 32, height: 32 }),
    /scene\.backgroundRotation\.x must be a finite number/i,
  )

  const backgroundOrderScene = new THREE.Scene()
  backgroundOrderScene.background = splitEnvironmentTexture()
  backgroundOrderScene.backgroundRotation = [0, 0, 0, 'BAD']
  assert.throws(
    () => renderRgba(backgroundOrderScene, camera, { width: 32, height: 32 }),
    /scene\.backgroundRotation\[3\] must be one of XYZ, YXZ, ZXY, ZYX, YZX, or XZY/i,
  )

  const optionBackgroundScene = new THREE.Scene()
  optionBackgroundScene.background = splitEnvironmentTexture()
  assert.throws(
    () => renderRgba(optionBackgroundScene, camera, {
      width: 32,
      height: 32,
      backgroundRotation: { x: 'left', y: 0, z: 0 },
    }),
    /options\.backgroundRotation\.x must be a finite number/i,
  )
  assert.throws(
    () => renderRgba(new THREE.Scene(), camera, {
      width: 32,
      height: 32,
      renderMode: 'mask',
      backgroundRotation: { x: 'left', y: 0, z: 0 },
    }),
    /options\.backgroundRotation\.x must be a finite number/i,
  )

  const environmentScene = new THREE.Scene()
  environmentScene.background = new THREE.Color(0, 0, 0)
  environmentScene.environment = splitEnvironmentTexture()
  environmentScene.environmentRotation = [0, Number.NaN, 0]
  environmentScene.add(new THREE.Mesh(
    new THREE.SphereGeometry(1, 16, 16),
    new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0.2 }),
  ))
  assert.throws(
    () => renderRgba(environmentScene, camera, { width: 32, height: 32 }),
    /scene\.environmentRotation\[1\] must be a finite number/i,
  )

  const optionEnvironmentScene = new THREE.Scene()
  optionEnvironmentScene.background = new THREE.Color(0, 0, 0)
  optionEnvironmentScene.environment = splitEnvironmentTexture()
  optionEnvironmentScene.add(new THREE.Mesh(
    new THREE.SphereGeometry(1, 16, 16),
    new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0.2 }),
  ))
  assert.throws(
    () => renderRgba(optionEnvironmentScene, camera, {
      width: 32,
      height: 32,
      environmentRotation: [0, Number.NaN, 0],
    }),
    /options\.environmentRotation\[1\] must be a finite number/i,
  )
  assert.throws(
    () => renderRgba(new THREE.Scene(), camera, {
      width: 32,
      height: 32,
      renderMode: 'mask',
      environmentRotation: [0, Number.NaN, 0],
    }),
    /options\.environmentRotation\[1\] must be a finite number/i,
  )
  assert.throws(
    () => renderRgba(new THREE.Scene(), camera, {
      width: 32,
      height: 32,
      environmentRotation: 0,
    }),
    /options\.environmentRotation must be a rotation object or array/i,
  )
})

test('cube scene environments feed physical IBL', () => {
  function makeScene(environment) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.environment = environment
    scene.environmentIntensity = 4
    scene.add(new THREE.Mesh(
      new THREE.SphereGeometry(1, 32, 32),
      new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0.2 }),
    ))
    return scene
  }

  const camera = makeCamera()
  const noEnvironment = renderRgba(makeScene(null), camera)
  const rawCube = renderRgba(makeScene(cubeTexture([
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0],
  ])), camera)
  const encodedCube = renderRgba(makeScene(encodedCubeTexture()), camera)

  const rawDiff = meanAbsDiff(noEnvironment, rawCube)
  const encodedDiff = meanAbsDiff(noEnvironment, encodedCube)
  assert.ok(rawDiff > 0.5, `raw cube environment should affect metallic IBL, diff=${rawDiff.toFixed(3)}`)
  assert.ok(encodedDiff > 0.5, `encoded cube environment should affect metallic IBL, diff=${encodedDiff.toFixed(3)}`)
})

test('cube reflection probes feed physical IBL', () => {
  function makeScene(texture) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0.04, 0.04, 0.045)
    addLights(scene)
    scene.userData.headlessThreeRenderer = {
      reflectionProbe: {
        texture,
        intensity: 1.0,
      },
    }
    scene.add(new THREE.Mesh(
      new THREE.SphereGeometry(1, 32, 32),
      new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1.0, roughness: 0.2 }),
    ))
    return scene
  }

  const camera = makeCamera()
  const withoutProbe = new THREE.Scene()
  withoutProbe.background = new THREE.Color(0.04, 0.04, 0.045)
  addLights(withoutProbe)
  withoutProbe.add(new THREE.Mesh(
    new THREE.SphereGeometry(1, 32, 32),
    new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1.0, roughness: 0.2 }),
  ))

  const noProbe = renderRgba(withoutProbe, camera)
  const withCubeProbe = renderRgba(makeScene(encodedCubeTexture()), camera)
  const diff = meanAbsDiff(noProbe, withCubeProbe)
  assert.ok(diff > 0.5, `encoded cube reflection probe should affect metallic IBL, diff=${diff.toFixed(3)}`)
})

test('scene environmentRotation rotates equirectangular IBL', () => {
  function renderWithRotation(yRotation) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.environment = splitEnvironmentTexture()
    scene.environmentIntensity = 4
    scene.environmentRotation = new THREE.Euler(0, yRotation, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0 }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 64, 64, 24, 24, 40, 40)
  }

  const unrotated = renderWithRotation(0)
  const rotated = renderWithRotation(-Math.PI / 2)
  assert.ok(unrotated.r > unrotated.g + 15, `unrotated reflection should sample the red environment half (${unrotated.r} vs ${unrotated.g})`)
  assert.ok(rotated.g > rotated.r + 15, `rotated reflection should sample the green environment half (${rotated.g} vs ${rotated.r})`)
})

test('options.environmentRotation overrides scene environmentRotation', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.environment = splitEnvironmentTexture()
  scene.environmentIntensity = 4
  scene.environmentRotation = new THREE.Euler(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0 }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const sceneRotation = meanRegion(renderRgba(scene, camera, {
    width: 64,
    height: 64,
    outputColorSpace: THREE.LinearSRGBColorSpace,
  }), 64, 64, 24, 24, 40, 40)
  const optionRotation = meanRegion(renderRgba(scene, camera, {
    width: 64,
    height: 64,
    outputColorSpace: THREE.LinearSRGBColorSpace,
    environmentRotation: new THREE.Euler(0, -Math.PI / 2, 0),
  }), 64, 64, 24, 24, 40, 40)

  assert.ok(sceneRotation.r > sceneRotation.g + 15, `scene rotation should sample red environment half (${sceneRotation.r} vs ${sceneRotation.g})`)
  assert.ok(optionRotation.g > optionRotation.r + 15, `options.environmentRotation should override to green half (${optionRotation.g} vs ${optionRotation.r})`)
})

test('scene environmentRotation rotates cube IBL', () => {
  const environment = cubeTexture([
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
  ])

  function renderWithRotation(yRotation) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.environment = environment
    scene.environmentIntensity = 4
    scene.environmentRotation = new THREE.Euler(0, yRotation, 0)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0 }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return renderRgba(scene, camera, {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    })
  }

  const unrotated = renderWithRotation(0)
  const rotated = renderWithRotation(-Math.PI / 2)
  const diff = meanAbsDiff(unrotated, rotated)
  assert.ok(diff > 1.0, `rotated cube IBL should change the reflection, diff=${diff.toFixed(3)}`)
})

test('scene environment colorSpace controls RGBA8 IBL decode', () => {
  function renderColorSpace(colorSpace) {
    const data = new Uint8Array([
      128, 128, 128, 255,
      128, 128, 128, 255,
      128, 128, 128, 255,
      128, 128, 128, 255,
    ])
    const environment = new THREE.DataTexture(data, 2, 2, THREE.RGBAFormat)
    environment.colorSpace = colorSpace
    environment.mapping = THREE.EquirectangularReflectionMapping
    environment.needsUpdate = true

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.environment = environment
    scene.environmentIntensity = 1
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0.2 }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 64, 64, 24, 24, 40, 40)
  }

  const srgb = renderColorSpace(THREE.SRGBColorSpace)
  const linear = renderColorSpace(THREE.LinearSRGBColorSpace)
  assert.ok(linear.r > srgb.r + 20, `linear environment should precompute brighter IBL than decoded sRGB (${linear.r} vs ${srgb.r})`)
  assert.deepEqual(renderColorSpace('linear'), linear, 'linear alias should match THREE.LinearSRGBColorSpace for scene environment IBL')
})

test('cube environment and reflection probe colorSpace controls IBL decode', () => {
  function grayCube(colorSpace) {
    const environment = cubeTexture([
      [128, 128, 128],
      [128, 128, 128],
      [128, 128, 128],
      [128, 128, 128],
      [128, 128, 128],
      [128, 128, 128],
    ])
    environment.colorSpace = colorSpace
    return environment
  }

  function makeMetallicScene(setup) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    setup(scene)
    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0.2 }),
    ))
    return scene
  }

  function renderSceneEnvironment(colorSpace) {
    const scene = makeMetallicScene((target) => {
      target.environment = grayCube(colorSpace)
      target.environmentIntensity = 1
    })
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 64, 64, 24, 24, 40, 40)
  }

  function renderReflectionProbe(colorSpace) {
    const scene = makeMetallicScene((target) => {
      target.userData.headlessThreeRenderer = {
        reflectionProbe: {
          texture: grayCube(colorSpace),
          intensity: 1,
        },
      }
    })
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 64, 64, 24, 24, 40, 40)
  }

  const srgbEnvironment = renderSceneEnvironment(THREE.SRGBColorSpace)
  const linearEnvironment = renderSceneEnvironment(THREE.LinearSRGBColorSpace)
  assert.ok(
    linearEnvironment.r > srgbEnvironment.r + 20,
    `linear cube environment should precompute brighter IBL than decoded sRGB (${linearEnvironment.r} vs ${srgbEnvironment.r})`,
  )
  assert.deepEqual(
    renderSceneEnvironment('linear-srgb'),
    linearEnvironment,
    'linear-srgb alias should match THREE.LinearSRGBColorSpace for cube environment IBL',
  )

  const srgbProbe = renderReflectionProbe(THREE.SRGBColorSpace)
  const linearProbe = renderReflectionProbe(THREE.LinearSRGBColorSpace)
  assert.ok(
    linearProbe.r > srgbProbe.r + 20,
    `linear cube reflection probe should precompute brighter IBL than decoded sRGB (${linearProbe.r} vs ${srgbProbe.r})`,
  )
  assert.deepEqual(
    renderReflectionProbe('linearsrgb'),
    linearProbe,
    'linearsrgb alias should match THREE.LinearSRGBColorSpace for reflection probe IBL',
  )
})

test('unsupported environment and reflection probe mappings fail clearly', () => {
  const cases = [
    ['CubeUV scene environment', (scene) => {
      scene.environment = Object.assign(makeEnvironmentTexture(), { mapping: THREE.CubeUVReflectionMapping })
    }],
    ['refraction scene environment', (scene) => {
      scene.environment = Object.assign(makeEnvironmentTexture(), { mapping: THREE.EquirectangularRefractionMapping })
    }],
    ['CubeUV reflection probe', (scene) => {
      scene.userData.headlessThreeRenderer = {
        reflectionProbe: {
          texture: Object.assign(makeEnvironmentTexture(), { mapping: THREE.CubeUVReflectionMapping }),
        },
      }
    }],
    ['refraction reflection probe', (scene) => {
      scene.userData.headlessThreeRenderer = {
        reflectionProbe: {
          texture: Object.assign(makeEnvironmentTexture(), { mapping: THREE.EquirectangularRefractionMapping }),
        },
      }
    }],
  ]

  for (const [name, setup] of cases) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    setup(scene)

    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      /refraction or PMREM\/CubeUV environment mapping.*not supported/i,
      name,
    )
  }
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

test('physical transmission roughness softens scene-color refraction', () => {
  const width = 64
  const height = 64
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  function makeScene(roughness) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)

    const left = new THREE.Mesh(
      new THREE.PlaneGeometry(1.6, 3),
      new THREE.MeshBasicMaterial({ color: 0xff0000 }),
    )
    left.position.set(-0.8, 0, -0.1)
    scene.add(left)

    const right = new THREE.Mesh(
      new THREE.PlaneGeometry(1.6, 3),
      new THREE.MeshBasicMaterial({ color: 0x0000ff }),
    )
    right.position.set(0.8, 0, -0.1)
    scene.add(right)

    scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(3, 3),
      new THREE.MeshPhysicalMaterial({
        color: 0xffffff,
        metalness: 0,
        roughness,
        transmission: 1,
        thickness: 0,
        ior: 1.5,
      }),
    ))
    return scene
  }

  function centerEdgeContrast(rgba) {
    const left = meanRegion(rgba, width, height, 25, 20, 31, 44)
    const right = meanRegion(rgba, width, height, 33, 20, 39, 44)
    return Math.abs((left.r - left.b) - (right.r - right.b))
  }

  const smooth = renderRgba(makeScene(0.02), camera, { width, height })
  const rough = renderRgba(makeScene(0.95), camera, { width, height })
  const smoothContrast = centerEdgeContrast(smooth)
  const roughContrast = centerEdgeContrast(rough)

  assert.ok(smoothContrast > 80, `smooth transmission should preserve the sharp red/blue edge (${smoothContrast.toFixed(1)})`)
  assert.ok(
    roughContrast < smoothContrast - 20,
    `rough transmission should reduce scene-color edge contrast (${roughContrast.toFixed(1)} vs ${smoothContrast.toFixed(1)})`,
  )
})

test('physical transmission dispersion separates transmitted color channels', () => {
  const width = 64
  const height = 64
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  function makeScene(dispersion) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)

    const left = new THREE.Mesh(
      new THREE.PlaneGeometry(1.6, 3),
      new THREE.MeshBasicMaterial({ color: 0xff0000 }),
    )
    left.position.set(-0.8, 0, -0.2)
    scene.add(left)

    const right = new THREE.Mesh(
      new THREE.PlaneGeometry(1.6, 3),
      new THREE.MeshBasicMaterial({ color: 0x0000ff }),
    )
    right.position.set(0.8, 0, -0.2)
    scene.add(right)

    const glass = new THREE.Mesh(
      new THREE.SphereGeometry(0.95, 48, 24),
      new THREE.MeshPhysicalMaterial({
        color: 0xffffff,
        metalness: 0,
        roughness: 0.02,
        transmission: 1,
        thickness: 40,
        ior: 2.2,
        dispersion,
      }),
    )
    scene.add(glass)
    return scene
  }

  const normal = renderRgba(makeScene(0), camera, { width, height })
  const dispersed = renderRgba(makeScene(10), camera, { width, height })
  const diff = meanAbsDiff(normal, dispersed)
  const normalEdge = meanRegion(normal, width, height, 28, 22, 36, 42)
  const dispersedEdge = meanRegion(dispersed, width, height, 28, 22, 36, 42)
  const normalSeparation = Math.abs(normalEdge.r - normalEdge.b)
  const dispersedSeparation = Math.abs(dispersedEdge.r - dispersedEdge.b)

  assert.ok(diff > 10, `dispersion should affect transmitted color, diff=${diff.toFixed(2)}`)
  assert.ok(
    Math.abs(dispersedSeparation - normalSeparation) > 20,
    `dispersion should change edge channel separation (${dispersedSeparation.toFixed(1)} vs ${normalSeparation.toFixed(1)})`,
  )
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

test('directional shadow cascade hints over four valid cascades fail clearly', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshStandardMaterial({ color: 0xffffff }),
  ))

  const light = new THREE.DirectionalLight(0xffffff, 1)
  light.position.set(4, 6, 3)
  light.target.position.set(0, 0, 0)
  light.castShadow = true
  light.userData.headlessThreeRenderer = {
    shadowCascades: Array.from({ length: 5 }, (_, index) => ({
      left: -2 - index,
      right: 2 + index,
      top: 2 + index,
      bottom: -2 - index,
      near: 0.1,
      far: 12 + index,
      split: 2 + index,
    })),
  }
  scene.add(light)
  scene.add(light.target)

  assert.throws(
    () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
    /shadow cascade hints.*at most 4 valid cascades/i,
  )
})

test('invalid directional shadow cascade hints fail clearly', () => {
  const validCascade = (index = 0) => ({
    left: -2 - index,
    right: 2 + index,
    top: 2 + index,
    bottom: -2 - index,
    near: 0.1,
    far: 12 + index,
    split: 2 + index,
  })

  const containerCases = [
    ['light userData container', (light) => {
      light.userData = 'cascades'
    }, /light\.userData must be an object/i],
    ['modern hint container', (light) => {
      light.userData.headlessThreeRenderer = 'cascades'
    }, /light\.userData\.headlessThreeRenderer must be an object/i],
    ['modern shadowCascades', (light) => {
      light.userData.headlessThreeRenderer = { shadowCascades: 'cascades' }
    }, /light\.userData\.headlessThreeRenderer\.shadowCascades must be an array/i],
    ['legacy hint container', (light) => {
      light.userData.headlessRenderer = []
    }, /light\.userData\.headlessRenderer must be an object/i],
    ['legacy cascades', (light) => {
      light.userData.headlessRenderer = { cascades: 'cascades' }
    }, /light\.userData\.headlessRenderer\.cascades must be an array/i],
    ['shadow cascades', (light) => {
      light.shadow.cascades = 'cascades'
    }, /light\.shadow\.cascades must be an array/i],
  ]
  for (const [name, setup, pattern] of containerCases) {
    const scene = new THREE.Scene()
    const light = new THREE.DirectionalLight(0xffffff, 1)
    light.castShadow = true
    setup(light)
    scene.add(light)

    assert.throws(
      () => extractLights(scene),
      pattern,
      `${name} should fail clearly`,
    )
  }

  const cases = [
    ['non-object cascade', [validCascade(), null], /shadowCascades\[1\] must be an object/i],
    ['missing far bound', [{ ...validCascade(), far: undefined }, validCascade(1)], /shadowCascades\[0\]\.far must be a finite number/i],
    ['invalid split', [{ ...validCascade(), split: 'near' }, validCascade(1)], /shadowCascades\[0\]\.split must be a finite number/i],
    ['invalid distance alias', [{ ...validCascade(), split: undefined, distance: Number.NaN }, validCascade(1)], /shadowCascades\[0\]\.distance must be a finite number/i],
  ]

  for (const [name, shadowCascades, pattern] of cases) {
    const scene = new THREE.Scene()
    const light = new THREE.DirectionalLight(0xffffff, 1)
    light.castShadow = true
    light.userData.headlessThreeRenderer = { shadowCascades }
    scene.add(light)

    assert.throws(
      () => extractLights(scene),
      pattern,
      `${name} should fail clearly`,
    )
  }
})

test('multiple shadow-casting directional lights render separate shadow maps', () => {
  function renderDirectionalShadows(lightXs) {
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
      light.shadow.mapSize.set(512, 512)
      light.shadow.camera.left = -6
      light.shadow.camera.right = 6
      light.shadow.camera.top = 6
      light.shadow.camera.bottom = -6
      light.shadow.camera.near = 0.1
      light.shadow.camera.far = 12
      scene.add(light)
      scene.add(light.target)
    }

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 10, 0)
    camera.up.set(0, 0, -1)
    camera.lookAt(0, 0, 0)
    return renderRgba(scene, camera, { width: 128, height: 128 })
  }

  const firstOnly = renderDirectionalShadows([5])
  const secondOnly = renderDirectionalShadows([-5])
  const both = renderDirectionalShadows([5, -5])
  const luminance = (mean) => mean.r + mean.g + mean.b
  const left = [28, 54, 48, 74]
  const right = [80, 54, 100, 74]

  const firstLeft = luminance(meanRegion(firstOnly, 128, 128, ...left))
  const firstRight = luminance(meanRegion(firstOnly, 128, 128, ...right))
  const secondLeft = luminance(meanRegion(secondOnly, 128, 128, ...left))
  const secondRight = luminance(meanRegion(secondOnly, 128, 128, ...right))
  const bothLeft = luminance(meanRegion(both, 128, 128, ...left))
  const bothRight = luminance(meanRegion(both, 128, 128, ...right))

  assert.ok(firstLeft < firstRight - 30, `first light should cast the left shadow (${firstLeft} vs ${firstRight})`)
  assert.ok(secondRight < secondLeft - 30, `second light should cast the right shadow (${secondRight} vs ${secondLeft})`)
  assert.ok(bothLeft < secondLeft - 30, `dual shadow maps should keep the first light's left shadow (${bothLeft} vs ${secondLeft})`)
  assert.ok(bothRight < firstRight - 30, `dual shadow maps should add the second light's right shadow (${bothRight} vs ${firstRight})`)
})

test('point and directional shadow lights render within the expanded layer budget', () => {
  function renderMixedShadow(castShadow) {
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
      new THREE.BoxGeometry(2, 2, 2),
      new THREE.MeshBasicMaterial({
        color: 0xffffff,
        colorWrite: false,
        depthWrite: false,
      }),
    )
    caster.position.y = 1
    caster.castShadow = castShadow
    scene.add(caster)

    const point = new THREE.PointLight(0xffffff, 2.5, 16)
    point.position.set(0, 5, 4)
    point.castShadow = true
    point.shadow.mapSize.set(256, 256)
    point.shadow.camera.near = 0.1
    point.shadow.camera.far = 16
    scene.add(point)

    const directional = new THREE.DirectionalLight(0xffffff, 2)
    directional.position.set(5, 6, 0)
    directional.target.position.set(0, 0, 0)
    directional.castShadow = true
    directional.shadow.mapSize.set(256, 256)
    directional.shadow.camera.left = -7
    directional.shadow.camera.right = 7
    directional.shadow.camera.top = 7
    directional.shadow.camera.bottom = -7
    directional.shadow.camera.near = 0.1
    directional.shadow.camera.far = 16
    scene.add(directional)
    scene.add(directional.target)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 7, 8)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, { width: 96, height: 96 }), 96, 96, 28, 42, 68, 82)
  }

  const unshadowed = renderMixedShadow(false)
  const shadowed = renderMixedShadow(true)
  const unshadowedLum = unshadowed.r + unshadowed.g + unshadowed.b
  const shadowedLum = shadowed.r + shadowed.g + shadowed.b
  assert.ok(shadowedLum < unshadowedLum - 20, `mixed point/directional shadows should darken the receiver (${shadowedLum} vs ${unshadowedLum})`)
})

test('shadow lights over the native layer budget fail clearly', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshStandardMaterial({ color: 0xffffff }),
  ))

  for (const x of [-2, 2]) {
    const point = new THREE.PointLight(0xffffff, 1)
    point.position.set(x, 4, 0)
    point.castShadow = true
    scene.add(point)
  }

  const directional = new THREE.DirectionalLight(0xffffff, 1)
  directional.position.set(4, 6, 3)
  directional.target.position.set(0, 0, 0)
  directional.castShadow = true
  scene.add(directional)
  scene.add(directional.target)

  assert.throws(
    () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
    /more than 12 shadow map layers.*13 requested/i,
  )
})

test('camera-layer-filtered shadow lights are omitted before native shadow packing', () => {
  const scene = new THREE.Scene()
  const camera = makeCamera()
  camera.layers.set(1)

  const filtered = new THREE.DirectionalLight(0xff0000, 1)
  filtered.castShadow = true
  filtered.layers.set(0)
  scene.add(filtered)

  const visible = new THREE.DirectionalLight(0x00ff00, 1)
  visible.castShadow = true
  visible.layers.set(1)
  scene.add(visible)

  const lights = extractLights(scene, camera)
  assert.equal(lights?.length, 1)
  assert.equal(lights?.[0]?.castShadow, true)
  assert.deepEqual(lights?.[0]?.color, [0, 1, 0])
})

test('shadow bias options are extracted for native shadow lights', () => {
  const makeLightCases = [
    ['directional', () => {
      const light = new THREE.DirectionalLight(0xffffff, 1)
      light.position.set(4, 6, 3)
      light.target.position.set(0, 0, 0)
      return { light, extras: [light.target], mapSize: [320, 192] }
    }],
    ['spot', () => {
      const light = new THREE.SpotLight(0xffffff, 1)
      light.position.set(3, 5, 2)
      light.target.position.set(0, 0, 0)
      return { light, extras: [light.target], mapSize: [320, 192] }
    }],
    ['point', () => {
      const light = new THREE.PointLight(0xffffff, 1)
      light.position.set(2, 4, 2)
      return { light, extras: [], mapSize: [256, 256] }
    }],
  ]

  for (const [lightType, makeLight] of makeLightCases) {
    const scene = new THREE.Scene()
    const { light, extras, mapSize } = makeLight()
    light.castShadow = true
    light.shadow.mapSize.set(mapSize[0], mapSize[1])
    light.shadow.bias = -0.004
    light.shadow.normalBias = 0.125
    light.shadow.radius = 3.5
    light.shadow.camera.near = 0.2
    light.shadow.camera.far = 24
    if ('left' in light.shadow.camera) {
      light.shadow.camera.left = -5
      light.shadow.camera.right = 4
      light.shadow.camera.top = 6
      light.shadow.camera.bottom = -3
    }
    scene.add(light, ...extras)
    scene.updateMatrixWorld(true)

    const extracted = extractLights(scene, makeCamera()) ?? []
    assert.equal(extracted.length, 1, `${lightType} shadow light should be extracted`)
    const nativeLight = extracted[0]
    assert.equal(nativeLight.lightType, lightType)
    assert.equal(nativeLight.castShadow, true)
    assert.equal(nativeLight.shadowMapSize, Math.max(mapSize[0], mapSize[1]))
    assert.equal(nativeLight.shadowMapWidth, mapSize[0])
    assert.equal(nativeLight.shadowMapHeight, mapSize[1])
    assert.equal(nativeLight.shadowBias, -0.004)
    assert.equal(nativeLight.shadowNormalBias, 0.125)
    assert.equal(nativeLight.shadowRadius, 3.5)
    assert.equal(nativeLight.shadowCameraNear, 0.2)
    assert.equal(nativeLight.shadowCameraFar, 24)
    if (lightType === 'directional') {
      assert.equal(nativeLight.shadowCameraLeft, -5)
      assert.equal(nativeLight.shadowCameraRight, 4)
      assert.equal(nativeLight.shadowCameraTop, 6)
      assert.equal(nativeLight.shadowCameraBottom, -3)
    }
  }
})

test('rectangular directional shadow map sizes render shadows', () => {
  function renderRectShadow(castShadow) {
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
    light.shadow.mapSize.set(512, 256)
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

  const unshadowed = renderRectShadow(false)
  const shadowed = renderRectShadow(true)
  const unshadowedLum = unshadowed.r + unshadowed.g + unshadowed.b
  const shadowedLum = shadowed.r + shadowed.g + shadowed.b
  assert.ok(shadowedLum < unshadowedLum - 30, `rectangular shadow map should darken the receiver (${shadowedLum} vs ${unshadowedLum})`)
})

test('material shadowSide filters shadow caster faces', () => {
  function renderShadowSide(shadowSide) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const casterMaterial = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      colorWrite: false,
      depthWrite: false,
    })
    casterMaterial.shadowSide = shadowSide
    const caster = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), casterMaterial)
    caster.rotation.x = -Math.PI / 2
    caster.position.y = 2
    caster.castShadow = true
    scene.add(caster)

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(0, 6, 2)
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

  const front = renderShadowSide(THREE.FrontSide)
  const back = renderShadowSide(THREE.BackSide)
  const frontLum = front.r + front.g + front.b
  const backLum = back.r + back.g + back.b
  assert.ok(frontLum < backLum - 30, `front shadowSide should cast a darker shadow than back shadowSide (${frontLum} vs ${backLum})`)
})

test('unsupported material shadowSide values fail clearly', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.DirectionalLight(0xffffff, 1))
  const material = new THREE.MeshBasicMaterial({ color: 0xffffff })
  material.shadowSide = 999
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material)
  mesh.castShadow = true
  scene.add(mesh)

  assert.throws(
    () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
    /material\.shadowSide 999.*not supported/i,
  )
})

test('source material shadowSide applies to custom shadow material casters', () => {
  function renderCustomDepthShadowSide(shadowSide) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const sourceMaterial = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      colorWrite: false,
      depthWrite: false,
    })
    sourceMaterial.shadowSide = shadowSide
    const caster = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), sourceMaterial)
    caster.rotation.x = -Math.PI / 2
    caster.position.y = 2
    caster.castShadow = true
    caster.customDepthMaterial = new THREE.MeshDepthMaterial()
    scene.add(caster)

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(0, 6, 2)
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
    return meanRgba(renderRgbaIsolated(scene, camera, { width: 96, height: 96 }))
  }

  function renderCustomDistanceShadowSide(shadowSide) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const sourceMaterial = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      colorWrite: false,
      depthWrite: false,
    })
    sourceMaterial.shadowSide = shadowSide
    const caster = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), sourceMaterial)
    caster.rotation.x = -Math.PI / 2
    caster.position.y = 2
    caster.castShadow = true
    caster.customDistanceMaterial = new THREE.MeshDistanceMaterial()
    scene.add(caster)

    const light = new THREE.PointLight(0xffffff, 2)
    light.position.set(0, 6, 2)
    light.distance = 12
    light.castShadow = true
    light.shadow.mapSize.set(256, 256)
    light.shadow.camera.near = 0.1
    light.shadow.camera.far = 12
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    return meanRgba(renderRgbaIsolated(scene, camera, { width: 96, height: 96 }))
  }

  const front = renderCustomDepthShadowSide(THREE.FrontSide)
  const back = renderCustomDepthShadowSide(THREE.BackSide)
  const frontLum = front.r + front.g + front.b
  const backLum = back.r + back.g + back.b
  assert.ok(frontLum < backLum - 30, `source material shadowSide should affect customDepthMaterial shadows (${frontLum} vs ${backLum})`)

  const frontDistance = renderCustomDistanceShadowSide(THREE.FrontSide)
  const backDistance = renderCustomDistanceShadowSide(THREE.BackSide)
  const frontDistanceLum = frontDistance.r + frontDistance.g + frontDistance.b
  const backDistanceLum = backDistance.r + backDistance.g + backDistance.b
  assert.ok(
    frontDistanceLum < backDistanceLum - 10,
    `source material shadowSide should affect customDistanceMaterial shadows (${frontDistanceLum} vs ${backDistanceLum})`,
  )
})

test('material alphaToCoverage approximates shadow caster alpha cutouts', () => {
  function renderAlphaCoverageShadow(alphaToCoverage) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const alphaMap = rgbaTexture([
      255, 0, 0, 255,
      255, 255, 255, 255,
    ], 2, 1)
    const casterMaterial = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      alphaMap,
      alphaToCoverage,
      colorWrite: false,
      depthWrite: false,
    })
    const caster = new THREE.Mesh(new THREE.PlaneGeometry(5, 4), casterMaterial)
    caster.rotation.x = -Math.PI / 2
    caster.position.y = 2
    caster.castShadow = true
    scene.add(caster)

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(0, 6, 2)
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

  const fullShadow = renderAlphaCoverageShadow(false)
  const cutoutShadow = renderAlphaCoverageShadow(true)
  const fullLum = fullShadow.r + fullShadow.g + fullShadow.b
  const cutoutLum = cutoutShadow.r + cutoutShadow.g + cutoutShadow.b
  assert.ok(cutoutLum > fullLum + 15, `alphaToCoverage shadow cutoff should let more receiver light through (${cutoutLum} vs ${fullLum})`)
})

test('material alphaHash approximates shadow caster opacity cutouts', () => {
  function renderAlphaHashShadow(alphaHash) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const casterMaterial = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      opacity: alphaHash ? 0.35 : 1,
      alphaHash,
      colorWrite: false,
      depthWrite: false,
    })
    const caster = new THREE.Mesh(new THREE.BoxGeometry(3, 3, 3), casterMaterial)
    caster.position.y = 1.5
    caster.castShadow = true
    scene.add(caster)

    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(8, 6, 0)
    light.target.position.set(0, 0, 0)
    light.castShadow = true
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

  const fullShadow = renderAlphaHashShadow(false)
  const hashedShadow = renderAlphaHashShadow(true)
  const fullLum = fullShadow.r + fullShadow.g + fullShadow.b
  const hashedLum = hashedShadow.r + hashedShadow.g + hashedShadow.b
  assert.ok(hashedLum > fullLum + 20, `alphaHash shadow cutoff should let more receiver light through (${hashedLum} vs ${fullLum})`)
})

test('invalid visibility flag values fail clearly', () => {
  const camera = makeCamera()

  const objectScene = new THREE.Scene()
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial())
  mesh.visible = 'yes'
  objectScene.add(mesh)
  assert.throws(
    () => renderRgba(objectScene, camera, { width: 32, height: 32 }),
    /object\.visible must be a boolean/i,
  )

  const materialScene = new THREE.Scene()
  const material = new THREE.MeshBasicMaterial({ color: 0xffffff })
  material.visible = 'yes'
  materialScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))
  assert.throws(
    () => renderRgba(materialScene, camera, { width: 32, height: 32 }),
    /material\.visible must be a boolean/i,
  )

  const lightScene = new THREE.Scene()
  const light = new THREE.DirectionalLight(0xffffff, 1)
  light.visible = 'yes'
  lightScene.add(light)
  assert.throws(
    () => extractLights(lightScene),
    /object\.visible must be a boolean/i,
  )
})

test('non-square point-light shadow map sizes fail clearly', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const light = new THREE.PointLight(0xffffff, 1)
  light.position.set(2, 4, 2)
  light.castShadow = true
  light.shadow.mapSize.set(512, 256)
  scene.add(light)

  assert.throws(
    () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
    /non-square PointLight shadow map sizes.*not supported/i,
  )
})

test('invalid object and light shadow flag values fail clearly', () => {
  const camera = makeCamera()
  const objectCases = [
    ['mesh castShadow', () => {
      const scene = new THREE.Scene()
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial())
      mesh.castShadow = 'yes'
      scene.add(mesh)
      return scene
    }, /object\.castShadow must be a boolean/i],
    ['mesh receiveShadow', () => {
      const scene = new THREE.Scene()
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial())
      mesh.receiveShadow = 1
      scene.add(mesh)
      return scene
    }, /object\.receiveShadow must be a boolean/i],
    ['sprite castShadow', () => {
      const scene = new THREE.Scene()
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial())
      sprite.castShadow = 'yes'
      scene.add(sprite)
      return scene
    }, /object\.castShadow must be a boolean/i],
    ['sprite receiveShadow', () => {
      const scene = new THREE.Scene()
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial())
      sprite.receiveShadow = 'yes'
      scene.add(sprite)
      return scene
    }, /object\.receiveShadow must be a boolean/i],
    ['points castShadow', () => {
      const scene = new THREE.Scene()
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))
      const points = new THREE.Points(geometry, new THREE.PointsMaterial())
      points.castShadow = 1
      scene.add(points)
      return scene
    }, /object\.castShadow must be a boolean/i],
    ['line receiveShadow', () => {
      const scene = new THREE.Scene()
      const geometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-1, 0, 0),
        new THREE.Vector3(1, 0, 0),
      ])
      const line = new THREE.Line(geometry, new THREE.LineBasicMaterial())
      line.receiveShadow = 'yes'
      scene.add(line)
      return scene
    }, /object\.receiveShadow must be a boolean/i],
  ]

  for (const [name, makeScene, pattern] of objectCases) {
    assert.throws(
      () => renderRgba(makeScene(), camera, { width: 32, height: 32 }),
      pattern,
      name,
    )
  }

  const lightScene = new THREE.Scene()
  const light = new THREE.DirectionalLight(0xffffff, 1)
  light.castShadow = 'yes'
  lightScene.add(light)
  assert.throws(
    () => extractLights(lightScene),
    /light\.castShadow must be a boolean/i,
  )
})

test('invalid shadow numeric values fail clearly', () => {
  const cases = [
    ['shadow container', (light) => {
      light.shadow = 'shadow'
    }, /light\.shadow must be an object/i],
    ['mapSize container', (light) => {
      light.shadow.mapSize = [512, 512]
    }, /light\.shadow\.mapSize must be an object/i],
    ['mapSize.x', (light) => {
      light.shadow.mapSize.x = 'wide'
    }, /light\.shadow\.mapSize\.x must be a finite number/i],
    ['mapSize.y', (light) => {
      light.shadow.mapSize.y = Number.NaN
    }, /light\.shadow\.mapSize\.y must be a finite number/i],
    ['mapSize.x zero', (light) => {
      light.shadow.mapSize.x = 0
    }, /light\.shadow\.mapSize\.x must be positive/i],
    ['mapSize.width', (light) => {
      light.shadow.mapSize = { width: 'wide', height: 512 }
    }, /light\.shadow\.mapSize\.width must be a finite number/i],
    ['mapSize.height', (light) => {
      light.shadow.mapSize = { width: 512, height: Number.NaN }
    }, /light\.shadow\.mapSize\.height must be a finite number/i],
    ['mapSize.height zero', (light) => {
      light.shadow.mapSize = { width: 512, height: 0 }
    }, /light\.shadow\.mapSize\.height must be positive/i],
    ['bias', (light) => {
      light.shadow.bias = 'biased'
    }, /light\.shadow\.bias must be a finite number/i],
    ['normalBias', (light) => {
      light.shadow.normalBias = Number.POSITIVE_INFINITY
    }, /light\.shadow\.normalBias must be a finite number/i],
    ['radius', (light) => {
      light.shadow.radius = Number.NEGATIVE_INFINITY
    }, /light\.shadow\.radius must be a finite number/i],
    ['radius negative', (light) => {
      light.shadow.radius = -1
    }, /light\.shadow\.radius must be non-negative/i],
    ['blurSamples', (light) => {
      light.shadow.blurSamples = 'many'
    }, /light\.shadow\.blurSamples must be a finite number/i],
    ['blurSamples negative', (light) => {
      light.shadow.blurSamples = -1
    }, /light\.shadow\.blurSamples must be non-negative/i],
    ['camera.left', (light) => {
      light.shadow.camera.left = 'left'
    }, /light\.shadow\.camera\.left must be a finite number/i],
    ['camera.right before left', (light) => {
      light.shadow.camera.left = 4
      light.shadow.camera.right = 4
    }, /light\.shadow\.camera\.right must be greater than light\.shadow\.camera\.left/i],
    ['camera.left beyond default right', (light) => {
      light.shadow.camera.left = 10
      delete light.shadow.camera.right
    }, /light\.shadow\.camera\.left must be less than the effective light\.shadow\.camera\.right/i],
    ['camera.top below bottom', (light) => {
      light.shadow.camera.top = -6
      light.shadow.camera.bottom = -6
    }, /light\.shadow\.camera\.top must be greater than light\.shadow\.camera\.bottom/i],
    ['camera.bottom beyond default top', (light) => {
      light.shadow.camera.bottom = 10
      delete light.shadow.camera.top
    }, /light\.shadow\.camera\.bottom must be less than the effective light\.shadow\.camera\.top/i],
    ['camera container', (light) => {
      light.shadow.camera = 'camera'
    }, /light\.shadow\.camera must be an object/i],
    ['camera.near', (light) => {
      light.shadow.camera.near = Number.NaN
    }, /light\.shadow\.camera\.near must be a finite number/i],
    ['camera.far', (light) => {
      light.shadow.camera.far = 'far'
    }, /light\.shadow\.camera\.far must be a finite number/i],
    ['camera.near negative', (light) => {
      light.shadow.camera.near = -0.1
    }, /light\.shadow\.camera\.near must be non-negative/i],
    ['camera.far zero', (light) => {
      light.shadow.camera.far = 0
    }, /light\.shadow\.camera\.far must be positive/i],
    ['camera.far before near', (light) => {
      light.shadow.camera.near = 10
      light.shadow.camera.far = 1
    }, /light\.shadow\.camera\.far must be greater than light\.shadow\.camera\.near/i],
    ['camera.near beyond default far', (light) => {
      light.shadow.camera.near = 600
      delete light.shadow.camera.far
    }, /light\.shadow\.camera\.near must be less than the effective light\.shadow\.camera\.far/i],
  ]

  for (const [name, mutate, pattern] of cases) {
    const scene = new THREE.Scene()
    const light = new THREE.DirectionalLight(0xffffff, 1)
    light.castShadow = true
    mutate(light)
    scene.add(light)
    assert.throws(
      () => extractLights(scene),
      pattern,
      `${name} should fail clearly`,
    )
  }

  for (const [name, makeLight] of [
    ['point near zero', () => new THREE.PointLight(0xffffff, 1)],
    ['spot near zero', () => new THREE.SpotLight(0xffffff, 1)],
  ]) {
    const scene = new THREE.Scene()
    const light = makeLight()
    light.castShadow = true
    light.shadow.camera.near = 0
    scene.add(light)
    assert.throws(
      () => extractLights(scene),
      /light\.shadow\.camera\.near must be positive for point and spot shadows/i,
      `${name} should fail clearly`,
    )
  }

  const directionalScene = new THREE.Scene()
  const directionalLight = new THREE.DirectionalLight(0xffffff, 1)
  directionalLight.castShadow = true
  directionalLight.shadow.camera.near = 0
  directionalLight.shadow.camera.far = 24
  directionalScene.add(directionalLight)
  const [nativeDirectionalLight] = extractLights(directionalScene) ?? []
  assert.equal(nativeDirectionalLight.shadowCameraNear, 0)
})

test('shadow radius values render PCF shadows', () => {
  function renderRadiusShadow(castShadow) {
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
    light.shadow.radius = 4
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

  const unshadowed = renderRadiusShadow(false)
  const shadowed = renderRadiusShadow(true)
  const unshadowedLum = unshadowed.r + unshadowed.g + unshadowed.b
  const shadowedLum = shadowed.r + shadowed.g + shadowed.b
  assert.ok(shadowedLum < unshadowedLum - 20, `shadow radius should still render received shadows (${shadowedLum} vs ${unshadowedLum})`)
})

test('non-default shadow blurSamples are accepted for PCF shadows', () => {
  function renderBlurSamplesShadow(castShadow) {
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
    light.shadow.blurSamples = 4
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

  const unshadowed = renderBlurSamplesShadow(false)
  const shadowed = renderBlurSamplesShadow(true)
  const unshadowedLum = unshadowed.r + unshadowed.g + unshadowed.b
  const shadowedLum = shadowed.r + shadowed.g + shadowed.b
  assert.ok(shadowedLum < unshadowedLum - 20, `blurSamples should not disable PCF shadows (${shadowedLum} vs ${unshadowedLum})`)
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

test('ShadowMaterial opacity scales received shadow alpha', () => {
  function renderShadowMaterialOpacity(opacity) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity }),
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

  const opaque = renderShadowMaterialOpacity(1)
  const translucent = renderShadowMaterialOpacity(0.35)
  const opaqueLum = opaque.r + opaque.g + opaque.b
  const translucentLum = translucent.r + translucent.g + translucent.b
  assert.ok(
    translucentLum > opaqueLum + 30,
    `lower ShadowMaterial opacity should blend more background through received shadows (${translucentLum} vs ${opaqueLum})`,
  )
})

test('ShadowMaterial color tints received shadows', () => {
  function renderShadowMaterialTint(color) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ color, opacity: 1 }),
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
    return meanRegion(
      renderRgba(scene, camera, { width: 96, height: 96 }),
      96,
      96,
      4,
      40,
      20,
      54,
    )
  }

  const blue = renderShadowMaterialTint(0x2040ff)
  assert.ok(blue.b > blue.r + 30, `blue ShadowMaterial should tint received shadows blue (${blue.b} vs ${blue.r})`)
  assert.ok(blue.b > blue.g + 20, `blue ShadowMaterial should tint received shadows blue (${blue.b} vs ${blue.g})`)
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

test('ShadowMaterial shadow color honors outputColorSpace', () => {
  function renderShadowMaterialColor(outputColorSpace) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ color: 0x808080, opacity: 1 }),
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
    return meanRegion(
      renderRgba(scene, camera, { width: 96, height: 96, outputColorSpace }),
      96,
      96,
      32,
      32,
      64,
      64,
    )
  }

  const srgb = renderShadowMaterialColor(THREE.SRGBColorSpace)
  const linear = renderShadowMaterialColor(THREE.LinearSRGBColorSpace)
  assert.ok(
    srgb.r > linear.r + 15,
    `sRGB ShadowMaterial output should apply display conversion (${srgb.r} vs ${linear.r})`,
  )
  assert.ok(
    Math.abs(srgb.r - srgb.g) < 2,
    `ShadowMaterial gray color should stay neutral in sRGB output (${srgb.r} vs ${srgb.g})`,
  )
  assert.ok(
    Math.abs(linear.r - linear.g) < 2,
    `ShadowMaterial gray color should stay neutral in linear output (${linear.r} vs ${linear.g})`,
  )
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
  const buf = getRenderer().render(scene, camera, { width: SIZE, height: SIZE })
  assertValidPng(buf, { width: SIZE, height: SIZE })
})

test('LineLoop renders the implicit closing segment', () => {
  const geom = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-1, -0.8, 0),
    new THREE.Vector3(1, -0.8, 0),
    new THREE.Vector3(1, 0.8, 0),
  ])
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.LineLoop(geom, new THREE.LineBasicMaterial({ color: 0xffffff })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  const rgba = renderRgba(scene, camera, { width: 96, height: 96 })
  const closingPixels = countRegionPixels(
    rgba,
    96,
    96,
    20,
    28,
    36,
    68,
    (r, g, b) => r > 180 && g > 180 && b > 180,
  )
  assert.ok(closingPixels > 2, `LineLoop should render the closing segment (${closingPixels})`)
})

test('LineBasicMaterial opacity blends over the background', () => {
  const geom = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-1.5, 0, 0),
    new THREE.Vector3(1.5, 0, 0),
  ])
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 1)
  scene.add(new THREE.Line(
    geom,
    new THREE.LineBasicMaterial({
      color: 0xff0000,
      opacity: 0.5,
      transparent: true,
    }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  const rgba = renderRgba(scene, camera, { width: 96, height: 96 })
  const blendedPixels = countRegionPixels(
    rgba,
    96,
    96,
    8,
    44,
    88,
    52,
    (r, g, b) => r > 50 && b > 80 && g < 40,
  )
  assert.ok(blendedPixels > 2, `semi-transparent line should blend red over blue (${blendedPixels})`)
})

test('LineBasicMaterial and LineDashedMaterial ignore linecap and linejoin like WebGLRenderer', () => {
  function renderLine(kind, configure = () => {}) {
    const geom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-1.2, 0, 0),
      new THREE.Vector3(0, 0.5, 0),
      new THREE.Vector3(1.2, 0, 0),
    ])
    const material = kind === 'basic'
      ? new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 8 })
      : new THREE.LineDashedMaterial({
        color: 0xffffff,
        linewidth: 8,
        dashSize: 0.3,
        gapSize: 0.15,
        scale: 1,
      })
    configure(material)

    const line = new THREE.Line(geom, material)
    if (kind === 'dashed') line.computeLineDistances()

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(line)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return renderRgba(scene, camera, { width: 96, height: 96 })
  }

  for (const kind of ['basic', 'dashed']) {
    const baseline = renderLine(kind)
    const noOp = renderLine(kind, (material) => {
      material.linecap = 'butt'
      material.linejoin = 'bevel'
    })
    assert.deepEqual(noOp, baseline, `${kind} linecap/linejoin should be accepted as WebGL-compatible no-ops`)
  }
})

test('LineBasicMaterial and LineDashedMaterial receiveShadow is accepted as an unlit WebGL-compatible no-op', () => {
  function renderLine(kind, receiveShadow) {
    const geom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-1.2, 0, 0),
      new THREE.Vector3(1.2, 0, 0),
    ])
    const material = kind === 'basic'
      ? new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 8 })
      : new THREE.LineDashedMaterial({
        color: 0xffffff,
        linewidth: 8,
        dashSize: 0.3,
        gapSize: 0.15,
        scale: 1,
      })
    const line = new THREE.Line(geom, material)
    line.receiveShadow = receiveShadow
    if (kind === 'dashed') line.computeLineDistances()

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(line)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return renderRgba(scene, camera, { width: 96, height: 96 })
  }

  for (const kind of ['basic', 'dashed']) {
    assert.deepEqual(
      renderLine(kind, true),
      renderLine(kind, false),
      `${kind} line receiveShadow should not change unlit line output`,
    )
  }
})

test('LineBasicMaterial and LineDashedMaterial alphaHash produce main-pass stochastic coverage', () => {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  function renderLine(kind, alphaHash) {
    const geom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-1.2, 0, 0),
      new THREE.Vector3(1.2, 0, 0),
    ])
    const materialProps = {
      alphaHash,
      color: 0xffffff,
      linewidth: 16,
      opacity: alphaHash ? 0.35 : 1,
    }
    const material = kind === 'basic'
      ? new THREE.LineBasicMaterial(materialProps)
      : new THREE.LineDashedMaterial({
        ...materialProps,
        dashSize: 10,
        gapSize: 0,
        scale: 1,
      })
    const line = new THREE.Line(geom, material)
    if (kind === 'dashed') line.computeLineDistances()

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(line)
    return renderRgba(scene, camera, { width: 96, height: 96 })
  }

  for (const kind of ['basic', 'dashed']) {
    const opaque = renderLine(kind, false)
    const hashed = renderLine(kind, true)
    const visiblePixel = (r, g, b) => r > 20 || g > 20 || b > 20
    const opaquePixels = countRegionPixels(opaque, 96, 96, 16, 40, 80, 56, visiblePixel)
    const hashedPixels = countRegionPixels(hashed, 96, 96, 16, 40, 80, 56, visiblePixel)

    assert.ok(opaquePixels > 600, `${kind} opaque line should fill the sampled region (${opaquePixels})`)
    assert.ok(hashedPixels > 80, `${kind} alphaHash line should retain some visible pixels (${hashedPixels})`)
    assert.ok(hashedPixels < opaquePixels - 180, `${kind} alphaHash line should discard visible pixels (${hashedPixels} vs ${opaquePixels})`)
  }
})

test('LineBasicMaterial and LineDashedMaterial alphaToCoverage produce 4x-MSAA main-pass coverage', () => {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  function renderLine(kind, alphaToCoverage, sampleCount = 4) {
    const geom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-1.2, 0, 0),
      new THREE.Vector3(1.2, 0, 0),
    ])
    const materialProps = {
      alphaToCoverage,
      color: 0xffffff,
      linewidth: 16,
      opacity: 0.5,
      transparent: false,
    }
    const material = kind === 'basic'
      ? new THREE.LineBasicMaterial(materialProps)
      : new THREE.LineDashedMaterial({
        ...materialProps,
        dashSize: 10,
        gapSize: 0,
        scale: 1,
      })
    const line = new THREE.Line(geom, material)
    if (kind === 'dashed') line.computeLineDistances()

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(line)
    return meanRegion(renderRgba(scene, camera, {
      width: 96,
      height: 96,
      sampleCount,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 96, 96, 28, 44, 68, 52)
  }

  for (const kind of ['basic', 'dashed']) {
    const noCoverage = renderLine(kind, false)
    const coverage = renderLine(kind, true)
    const singleSample = renderLine(kind, true, 1)

    assert.ok(noCoverage.r > 170, `${kind} non-A2C line should keep bright RGB despite opacity alpha (${noCoverage.r})`)
    assert.ok(Math.abs(singleSample.r - noCoverage.r) < 5, `${kind} single-sample line alphaToCoverage should not alter RGB coverage (${singleSample.r} vs ${noCoverage.r})`)
    assert.ok(coverage.r > 30 && coverage.r < noCoverage.r - 80, `${kind} 4x line alphaToCoverage should resolve partial RGB coverage (${coverage.r} vs ${noCoverage.r})`)
  }
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

test('LineBasicMaterial alphaMap samples the selected secondary UV channel', () => {
  const alphaMap = rgbaTexture([
    255, 0, 255, 255,
    255, 255, 255, 255,
  ], 2, 1)

  function renderLine(channel) {
    alphaMap.channel = channel
    const geom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-1.5, 0, 0),
      new THREE.Vector3(1.5, 0, 0),
    ])
    geom.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([
      0.25, 0.5,
      0.25, 0.5,
    ]), 2))
    geom.setAttribute('uv1', new THREE.BufferAttribute(new Float32Array([
      0.75, 0.5,
      0.75, 0.5,
    ]), 2))

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0.1, 0.1, 0.1)
    const material = new THREE.LineBasicMaterial({ color: 0xffffff, alphaTest: 0.5 })
    material.alphaMap = alphaMap
    scene.add(new THREE.Line(
      geom,
      material,
    ))
    return nonBackgroundRatio(renderRgba(scene, makeCamera(), { width: 96, height: 96 }), BG)
  }

  const primary = renderLine(0)
  const secondary = renderLine(1)
  assert.ok(secondary > 0.001, `line alphaMap channel=1 should sample the opaque uv1 texel (${secondary})`)
  assert.ok(primary < secondary * 0.3, `line alphaMap channel=0 should sample the transparent primary UV texel (${primary} vs ${secondary})`)
})

test('LineBasicMaterial map and alphaMap can sample distinct non-primary UV channels', () => {
  const map = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter
  map.channel = 1
  const alphaMap = rgbaTexture([
    255, 255, 255, 255,
    255, 0, 255, 255,
  ], 2, 1)
  alphaMap.magFilter = THREE.NearestFilter
  alphaMap.minFilter = THREE.NearestFilter
  alphaMap.channel = 2

  const geometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-1.5, 0, 0),
    new THREE.Vector3(1.5, 0, 0),
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
    0.25, 0.5,
    0.25, 0.5,
  ]), 2))

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const material = new THREE.LineBasicMaterial({
    alphaTest: 0.5,
    color: 0xffffff,
    linewidth: 4,
    map,
  })
  material.alphaMap = alphaMap
  scene.add(new THREE.Line(
    geometry,
    material,
  ))

  const rgba = renderRgba(scene, makeCamera(), { width: 96, height: 96 })
  const greenPixels = countRegionPixels(
    rgba,
    96,
    96,
    8,
    42,
    88,
    54,
    (r, g, b) => g > 80 && g > r + 40 && g > b + 40,
  )
  assert.ok(greenPixels > 8, `line map channel=1 should render green while alphaMap channel=2 keeps it visible (${greenPixels})`)
})

test('LineBasicMaterial map RGB multiplies line color from UVs', () => {
  const map = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
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
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Line(
      geom,
      new THREE.LineBasicMaterial({ color: 0xffffff, map }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return renderRgba(scene, camera, { width: 96, height: 96 })
  }

  const red = renderLine(0.25)
  const green = renderLine(0.75)
  const redPixels = countRegionPixels(red, 96, 96, 0, 0, 96, 96, (r, g, b) => r > g + 40 && r > b + 40)
  const greenPixels = countRegionPixels(green, 96, 96, 0, 0, 96, 96, (r, g, b) => g > r + 40 && g > b + 40)
  assert.ok(redPixels > 2, `primary line map texel should tint line red (${redPixels})`)
  assert.ok(greenPixels > 2, `secondary line map texel should tint line green (${greenPixels})`)
})

test('LineBasicMaterial map applies texture UV transforms', () => {
  const map = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  map.offset.set(0.5, 0)
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter

  const geom = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-1.5, 0, 0),
    new THREE.Vector3(1.5, 0, 0),
  ])
  geom.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([
    0.25, 0.5,
    0.25, 0.5,
  ]), 2))

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Line(
    geom,
    new THREE.LineBasicMaterial({ color: 0xffffff, map }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const image = renderRgba(scene, camera, { width: 96, height: 96 })
  const greenPixels = countRegionPixels(image, 96, 96, 0, 0, 96, 96, (r, g, b) => g > r + 40 && g > b + 40)
  assert.ok(greenPixels > 2, `line map offset should shift line UVs from red to green (${greenPixels})`)
})

test('LineBasicMaterial map honors explicit texture matrices', () => {
  const map = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter
  setTextureMatrixOffset(map, 0.5)

  const geom = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-1.5, 0, 0),
    new THREE.Vector3(1.5, 0, 0),
  ])
  geom.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([
    0.25, 0.5,
    0.25, 0.5,
  ]), 2))

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Line(
    geom,
    new THREE.LineBasicMaterial({ color: 0xffffff, map }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const image = renderRgba(scene, camera, { width: 96, height: 96 })
  const greenPixels = countRegionPixels(image, 96, 96, 0, 0, 96, 96, (r, g, b) => g > r + 40 && g > b + 40)
  assert.ok(greenPixels > 2, `explicit line map matrix should shift line UVs from red to green (${greenPixels})`)
})

test('LineBasicMaterial map honors horizontal and vertical repeat wrapping', () => {
  const renderer = new Renderer()

  function renderWithWrapping({ wrapS = THREE.ClampToEdgeWrapping, wrapT = THREE.ClampToEdgeWrapping, vertical = false }) {
    const map = vertical
      ? rgbaTexture([
        0, 255, 0, 255,
        0, 255, 0, 255,
        255, 0, 0, 255,
        255, 0, 0, 255,
      ], 2, 2)
      : rgbaTexture([
        0, 255, 0, 255,
        255, 0, 0, 255,
        0, 255, 0, 255,
        255, 0, 0, 255,
      ], 2, 2)
    map.wrapS = wrapS
    map.wrapT = wrapT
    map.magFilter = THREE.NearestFilter
    map.minFilter = THREE.NearestFilter

    const geom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-1.5, 0, 0),
      new THREE.Vector3(1.5, 0, 0),
    ])
    geom.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([
      vertical ? 0.25 : 1.25, vertical ? 1.25 : 0.25,
      vertical ? 0.25 : 1.25, vertical ? 1.25 : 0.25,
    ]), 2))

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Line(
      geom,
      new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 8, map }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return renderer.render(scene, camera, { width: 96, height: 96, format: 'rgba' })
  }

  const clamped = renderWithWrapping({ wrapS: THREE.ClampToEdgeWrapping })
  const repeated = renderWithWrapping({ wrapS: THREE.RepeatWrapping })
  const mirrored = renderWithWrapping({ wrapS: THREE.MirroredRepeatWrapping })
  const clampedRed = countRegionPixels(clamped, 96, 96, 0, 0, 96, 96, (r, g, b) => r > g + 40 && r > b + 40)
  const repeatedGreen = countRegionPixels(repeated, 96, 96, 0, 0, 96, 96, (r, g, b) => g > r + 40 && g > b + 40)
  const mirroredRed = countRegionPixels(mirrored, 96, 96, 0, 0, 96, 96, (r, g, b) => r > g + 40 && r > b + 40)
  assert.ok(clampedRed > 600, `clamped line map U coordinates should sample the red edge texel (${clampedRed})`)
  assert.ok(repeatedGreen > 600, `repeated line map U coordinates should wrap to the green texel (${repeatedGreen})`)
  assert.ok(mirroredRed > 600, `mirrored line map U coordinates should reflect to the red texel (${mirroredRed})`)

  const clampedVertical = renderWithWrapping({ wrapT: THREE.ClampToEdgeWrapping, vertical: true })
  const repeatedVertical = renderWithWrapping({ wrapT: THREE.RepeatWrapping, vertical: true })
  const mirroredVertical = renderWithWrapping({ wrapT: THREE.MirroredRepeatWrapping, vertical: true })
  const clampedVerticalRed = countRegionPixels(clampedVertical, 96, 96, 0, 0, 96, 96, (r, g, b) => r > g + 40 && r > b + 40)
  const repeatedVerticalGreen = countRegionPixels(repeatedVertical, 96, 96, 0, 0, 96, 96, (r, g, b) => g > r + 40 && g > b + 40)
  const mirroredVerticalRed = countRegionPixels(mirroredVertical, 96, 96, 0, 0, 96, 96, (r, g, b) => r > g + 40 && r > b + 40)
  assert.ok(clampedVerticalRed > 600, `clamped line map V coordinates should sample the red edge texel (${clampedVerticalRed})`)
  assert.ok(repeatedVerticalGreen > 600, `repeated line map V coordinates should wrap to the green texel (${repeatedVerticalGreen})`)
  assert.ok(mirroredVerticalRed > 600, `mirrored line map V coordinates should reflect to the red texel (${mirroredVerticalRed})`)
})

test('LineBasicMaterial alphaMap honors horizontal and vertical repeat wrapping', () => {
  const renderer = new Renderer()

  function renderWithWrapping({ wrapS = THREE.ClampToEdgeWrapping, wrapT = THREE.ClampToEdgeWrapping, vertical = false }) {
    const alphaMap = vertical
      ? rgbaTexture([
        255, 255, 255, 255,
        255, 255, 255, 255,
        255, 0, 255, 255,
        255, 0, 255, 255,
      ], 2, 2)
      : rgbaTexture([
        255, 255, 255, 255,
        255, 0, 255, 255,
        255, 255, 255, 255,
        255, 0, 255, 255,
      ], 2, 2)
    alphaMap.wrapS = wrapS
    alphaMap.wrapT = wrapT
    alphaMap.magFilter = THREE.NearestFilter
    alphaMap.minFilter = THREE.NearestFilter

    const geom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-1.5, 0, 0),
      new THREE.Vector3(1.5, 0, 0),
    ])
    geom.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([
      vertical ? 0.25 : 1.25, vertical ? 1.25 : 0.25,
      vertical ? 0.25 : 1.25, vertical ? 1.25 : 0.25,
    ]), 2))

    const material = new THREE.LineBasicMaterial({
      alphaTest: 0.5,
      color: 0x00ff00,
      linewidth: 8,
    })
    material.alphaMap = alphaMap

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 1)
    scene.add(new THREE.Line(geom, material))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return renderer.render(scene, camera, { width: 96, height: 96, format: 'rgba' })
  }

  const clamped = renderWithWrapping({ wrapS: THREE.ClampToEdgeWrapping })
  const repeated = renderWithWrapping({ wrapS: THREE.RepeatWrapping })
  const mirrored = renderWithWrapping({ wrapS: THREE.MirroredRepeatWrapping })
  const clampedGreen = countRegionPixels(clamped, 96, 96, 0, 0, 96, 96, (r, g, b) => g > b + 40 && g > r + 40)
  const repeatedGreen = countRegionPixels(repeated, 96, 96, 0, 0, 96, 96, (r, g, b) => g > b + 40 && g > r + 40)
  const mirroredGreen = countRegionPixels(mirrored, 96, 96, 0, 0, 96, 96, (r, g, b) => g > b + 40 && g > r + 40)
  assert.ok(clampedGreen < 20, `clamped line alphaMap U coordinates should discard the line (${clampedGreen})`)
  assert.ok(repeatedGreen > 600, `repeated line alphaMap U coordinates should wrap to the opaque texel (${repeatedGreen})`)
  assert.ok(mirroredGreen < 20, `mirrored line alphaMap U coordinates should reflect to the transparent texel (${mirroredGreen})`)

  const clampedVertical = renderWithWrapping({ wrapT: THREE.ClampToEdgeWrapping, vertical: true })
  const repeatedVertical = renderWithWrapping({ wrapT: THREE.RepeatWrapping, vertical: true })
  const mirroredVertical = renderWithWrapping({ wrapT: THREE.MirroredRepeatWrapping, vertical: true })
  const clampedVerticalGreen = countRegionPixels(clampedVertical, 96, 96, 0, 0, 96, 96, (r, g, b) => g > b + 40 && g > r + 40)
  const repeatedVerticalGreen = countRegionPixels(repeatedVertical, 96, 96, 0, 0, 96, 96, (r, g, b) => g > b + 40 && g > r + 40)
  const mirroredVerticalGreen = countRegionPixels(mirroredVertical, 96, 96, 0, 0, 96, 96, (r, g, b) => g > b + 40 && g > r + 40)
  assert.ok(clampedVerticalGreen < 20, `clamped line alphaMap V coordinates should discard the line (${clampedVerticalGreen})`)
  assert.ok(repeatedVerticalGreen > 600, `repeated line alphaMap V coordinates should wrap to the opaque texel (${repeatedVerticalGreen})`)
  assert.ok(mirroredVerticalGreen < 20, `mirrored line alphaMap V coordinates should reflect to the transparent texel (${mirroredVerticalGreen})`)
})

test('LineBasicMaterial map decodes sRGB colorSpace before shading', () => {
  function renderColorSpace(colorSpace) {
    const map = solidTexture(128, 128, 128)
    map.colorSpace = colorSpace

    const geom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-1.5, 0, 0),
      new THREE.Vector3(1.5, 0, 0),
    ])
    geom.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([
      0.5, 0.5,
      0.5, 0.5,
    ]), 2))

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Line(
      geom,
      new THREE.LineBasicMaterial({ color: 0xffffff, map }),
    ))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, {
      width: 96,
      height: 96,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 96, 96, 0, 46, 96, 50)
  }

  const srgb = renderColorSpace(THREE.SRGBColorSpace)
  const linear = renderColorSpace(THREE.LinearSRGBColorSpace)
  assert.ok(linear.r > srgb.r + 5, `linear line map should render brighter than decoded sRGB texture (${linear.r} vs ${srgb.r})`)
})

test('LineDashedMaterial map decodes sRGB colorSpace before shading', () => {
  function renderColorSpace(colorSpace) {
    const map = solidTexture(128, 128, 128)
    map.colorSpace = colorSpace

    const geom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-1.5, 0, 0),
      new THREE.Vector3(1.5, 0, 0),
    ])
    geom.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([
      0.5, 0.5,
      0.5, 0.5,
    ]), 2))

    const line = new THREE.Line(
      geom,
      new THREE.LineDashedMaterial({
        color: 0xffffff,
        dashSize: 4,
        gapSize: 0,
        map,
        scale: 1,
      }),
    )
    line.computeLineDistances()

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(line)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, {
      width: 96,
      height: 96,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 96, 96, 0, 46, 96, 50)
  }

  const srgb = renderColorSpace(THREE.SRGBColorSpace)
  const linear = renderColorSpace(THREE.LinearSRGBColorSpace)
  assert.ok(linear.r > srgb.r + 5, `linear dashed-line map should render brighter than decoded sRGB texture (${linear.r} vs ${srgb.r})`)
})

test('LineBasicMaterial map samples the selected secondary UV channel', () => {
  const map = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  map.channel = 1

  const geom = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-1.5, 0, 0),
    new THREE.Vector3(1.5, 0, 0),
  ])
  geom.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([
    0.25, 0.5,
    0.25, 0.5,
  ]), 2))
  geom.setAttribute('uv1', new THREE.BufferAttribute(new Float32Array([
    0.75, 0.5,
    0.75, 0.5,
  ]), 2))

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Line(
    geom,
    new THREE.LineBasicMaterial({ color: 0xffffff, map }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 96, height: 96 })
  const greenPixels = countRegionPixels(rgba, 96, 96, 0, 0, 96, 96, (r, g, b) => g > r + 40 && g > b + 40)
  assert.ok(greenPixels > 2, `line map channel=1 should sample uv1 green texel (${greenPixels})`)
})

test('LineBasicMaterial map samples texture channel 2 from uv2 attributes', () => {
  const map = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  map.channel = 2

  const geom = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-1.5, 0, 0),
    new THREE.Vector3(1.5, 0, 0),
  ])
  geom.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([
    0.25, 0.5,
    0.25, 0.5,
  ]), 2))
  geom.setAttribute('uv1', new THREE.BufferAttribute(new Float32Array([
    0.25, 0.5,
    0.25, 0.5,
  ]), 2))
  geom.setAttribute('uv2', new THREE.BufferAttribute(new Float32Array([
    0.75, 0.5,
    0.75, 0.5,
  ]), 2))

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Line(
    geom,
    new THREE.LineBasicMaterial({ color: 0xffffff, map }),
  ))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 96, height: 96 })
  const greenPixels = countRegionPixels(rgba, 96, 96, 0, 0, 96, 96, (r, g, b) => g > r + 40 && g > b + 40)
  assert.ok(greenPixels > 2, `line map channel=2 should sample uv2 green texel (${greenPixels})`)
})

test('Line material arrays honor geometry groups', () => {
  const geom = new THREE.BufferGeometry()
  geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -1.5, 0, 0,
    -0.3, 0, 0,
    0.3, 0, 0,
    1.5, 0, 0,
  ]), 3))
  geom.addGroup(0, 2, 0)
  geom.addGroup(2, 2, 1)

  const line = new THREE.LineSegments(geom, [
    new THREE.LineBasicMaterial({ color: 0xff0000 }),
    new THREE.LineDashedMaterial({ color: 0x00ff00, dashSize: 10, gapSize: 0, scale: 1 }),
  ])

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(line)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 96, height: 96 })
  const redPixels = countRegionPixels(rgba, 96, 96, 8, 40, 44, 56, (r, g, b) => r > g + 40 && r > b + 40)
  const greenPixels = countRegionPixels(rgba, 96, 96, 52, 40, 88, 56, (r, g, b) => g > r + 40 && g > b + 40)
  assert.ok(redPixels > 2, `left line group should use the red material (${redPixels})`)
  assert.ok(greenPixels > 2, `right line group should use the green dashed material (${greenPixels})`)
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

test('LineDashedMaterial without lineDistance renders solid like WebGL', () => {
  function renderLine(computeDistances) {
    const geom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-1.5, 0, 0),
      new THREE.Vector3(1.5, 0, 0),
    ])
    const line = new THREE.Line(geom, new THREE.LineDashedMaterial({
      color: 0xffffff,
      dashSize: 0.15,
      gapSize: 0.15,
      scale: 1,
    }))
    if (computeDistances) line.computeLineDistances()

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0.1, 0.1, 0.1)
    scene.add(line)
    return nonBackgroundRatio(renderRgba(scene, makeCamera(), { width: 96, height: 96 }), BG)
  }

  const missingDistances = renderLine(false)
  const dashed = renderLine(true)

  assert.ok(missingDistances > 0.001, `missing lineDistance should still render visible line pixels (${missingDistances})`)
  assert.ok(dashed > 0.001, `computed lineDistance should render visible dash pixels (${dashed})`)
  assert.ok(dashed < missingDistances * 0.85, `computed lineDistance should dash; missing lineDistance should stay solid (${dashed} vs ${missingDistances})`)
})

test('LineDashedMaterial scale changes dash coverage', () => {
  function renderScale(scale) {
    const geom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-1.5, 0, 0),
      new THREE.Vector3(1.5, 0, 0),
    ])
    const line = new THREE.Line(geom, new THREE.LineDashedMaterial({
      color: 0xffffff,
      dashSize: 0.5,
      gapSize: 10,
      scale,
    }))
    line.computeLineDistances()

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0.1, 0.1, 0.1)
    scene.add(line)
    return nonBackgroundRatio(renderRgba(scene, makeCamera(), { width: 96, height: 96 }), BG)
  }

  const lowScale = renderScale(0.1)
  const highScale = renderScale(2)
  assert.ok(lowScale > 0.001, `low scale should keep the line visible (${lowScale})`)
  assert.ok(highScale < lowScale * 0.35, `higher scale should advance into the gap sooner (${highScale} vs ${lowScale})`)
})

test('LineDashedMaterial uses custom lineDistance attributes', () => {
  const geom = new THREE.BufferGeometry()
  geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -1.4, 0, 0,
    -0.2, 0, 0,
    0.2, 0, 0,
    1.4, 0, 0,
  ]), 3))
  geom.setAttribute('lineDistance', new THREE.BufferAttribute(new Float32Array([
    0, 0.4,
    0, 0.4,
  ]), 1))

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.LineSegments(geom, new THREE.LineDashedMaterial({
    color: 0xffffff,
    dashSize: 0.5,
    gapSize: 10,
    scale: 1,
  })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)
  const rgba = renderRgba(scene, camera, { width: 96, height: 96 })
  const leftPixels = countRegionPixels(
    rgba,
    96,
    96,
    10,
    44,
    44,
    52,
    (r, g, b) => r > 180 && g > 180 && b > 180,
  )
  const rightPixels = countRegionPixels(
    rgba,
    96,
    96,
    52,
    44,
    86,
    52,
    (r, g, b) => r > 180 && g > 180 && b > 180,
  )
  assert.ok(leftPixels > 2, `custom lineDistance should keep the left dashed segment visible (${leftPixels})`)
  assert.ok(rightPixels > 2, `custom lineDistance should reset and keep the right dashed segment visible (${rightPixels})`)
})

test('LineDashedMaterial handles descending custom lineDistance spans', () => {
  const geom = new THREE.BufferGeometry()
  geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.9, 0, 0,
    0.9, 0, 0,
  ]), 3))
  geom.setAttribute('lineDistance', new THREE.BufferAttribute(new Float32Array([
    2,
    0,
  ]), 1))

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Line(geom, new THREE.LineDashedMaterial({
    color: 0xffffff,
    dashSize: 0.6,
    gapSize: 10,
    scale: 1,
  })))

  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 1.2, -1.2, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 96, height: 96 })
  const leftGapPixels = countRegionPixels(
    rgba,
    96,
    96,
    14,
    44,
    30,
    52,
    (r, g, b) => r > 180 && g > 180 && b > 180,
  )
  const rightDashPixels = countRegionPixels(
    rgba,
    96,
    96,
    66,
    44,
    84,
    52,
    (r, g, b) => r > 180 && g > 180 && b > 180,
  )

  assert.equal(leftGapPixels, 0, `descending lineDistance should keep high-distance span in the gap (${leftGapPixels})`)
  assert.ok(rightDashPixels > 2, `descending lineDistance should render the low-distance dash span (${rightDashPixels})`)
})

test('LineDashedMaterial renders LineLoop closing dashes', () => {
  const geom = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-1, -0.8, 0),
    new THREE.Vector3(1, -0.8, 0),
    new THREE.Vector3(1, 0.8, 0),
  ])
  const line = new THREE.LineLoop(geom, new THREE.LineDashedMaterial({
    color: 0xffffff,
    dashSize: 0.25,
    gapSize: 0.15,
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
  const closingPixels = countRegionPixels(
    rgba,
    96,
    96,
    20,
    28,
    36,
    68,
    (r, g, b) => r > 180 && g > 180 && b > 180,
  )
  assert.ok(closingPixels > 2, `dashed LineLoop should render dashes on the closing segment (${closingPixels})`)
})

test('LineDashedMaterial LineLoop closing segment interpolates existing lineDistance values', () => {
  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 1.2, -1.2, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const geom = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-0.8, -0.8, 0),
    new THREE.Vector3(0.8, -0.8, 0),
    new THREE.Vector3(0.8, 0.8, 0),
    new THREE.Vector3(-0.8, 0.8, 0),
  ])
  const line = new THREE.LineLoop(geom, new THREE.LineDashedMaterial({
    color: 0xffffff,
    dashSize: 2,
    gapSize: 10,
    scale: 1,
  }))
  line.computeLineDistances()

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(line)

  const rgba = renderRgba(scene, camera, { width: 96, height: 96 })
  const closingPixels = countRegionPixels(
    rgba,
    96,
    96,
    12,
    48,
    20,
    76,
    (r, g, b) => r > 180 && g > 180 && b > 180,
  )
  const upperClosingPixels = countRegionPixels(
    rgba,
    96,
    96,
    12,
    18,
    20,
    34,
    (r, g, b) => r > 180 && g > 180 && b > 180,
  )

  assert.ok(closingPixels > 1, `closing LineLoop segment should use the first vertex lineDistance near the lower-left corner (${closingPixels})`)
  assert.equal(upperClosingPixels, 0, `closing LineLoop segment should keep the high-distance upper-left side in the dash gap (${upperClosingPixels})`)
})

test('LineBasicMaterial linewidth expands to thick camera-facing quads', () => {
  function renderLine(linewidth) {
    const geom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-0.8, 0, 0),
      new THREE.Vector3(0.8, 0, 0),
    ])
    const line = new THREE.Line(geom, new THREE.LineBasicMaterial({ color: 0xffffff, linewidth }))

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(line)

    const camera = new THREE.OrthographicCamera(-1.2, 1.2, 1.2, -1.2, 0.01, 10)
    camera.position.set(0, 0, 2)
    camera.lookAt(0, 0, 0)
    return renderRgba(scene, camera, { width: 96, height: 96 })
  }

  const white = (r, g, b) => r > 180 && g > 180 && b > 180
  const thinPixels = countRegionPixels(renderLine(1), 96, 96, 12, 32, 84, 64, white)
  const thickPixels = countRegionPixels(renderLine(10), 96, 96, 12, 32, 84, 64, white)

  assert.ok(thinPixels > 0, `default line width should render visible pixels (${thinPixels})`)
  assert.ok(thickPixels > thinPixels * 3, `wide linewidth should increase covered pixels (${thinPixels} -> ${thickPixels})`)
})

test('LineDashedMaterial linewidth expands dash segments to thick camera-facing quads', () => {
  function renderLine(linewidth) {
    const geom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-0.9, 0, 0),
      new THREE.Vector3(0.9, 0, 0),
    ])
    const line = new THREE.Line(geom, new THREE.LineDashedMaterial({
      color: 0xffffff,
      linewidth,
      dashSize: 0.25,
      gapSize: 0.2,
      scale: 1,
    }))
    line.computeLineDistances()

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(line)

    const camera = new THREE.OrthographicCamera(-1.2, 1.2, 1.2, -1.2, 0.01, 10)
    camera.position.set(0, 0, 2)
    camera.lookAt(0, 0, 0)
    return renderRgba(scene, camera, { width: 96, height: 96 })
  }

  const white = (r, g, b) => r > 180 && g > 180 && b > 180
  const thinPixels = countRegionPixels(renderLine(1), 96, 96, 10, 32, 86, 64, white)
  const thickPixels = countRegionPixels(renderLine(10), 96, 96, 10, 32, 86, 64, white)

  assert.ok(thinPixels > 0, `default dashed linewidth should render visible dash pixels (${thinPixels})`)
  assert.ok(thickPixels > thinPixels * 3, `wide dashed linewidth should increase covered pixels (${thinPixels} -> ${thickPixels})`)
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

  const ratio = nonBackgroundRatio(renderRgbaIsolated(scene, makeCamera(), { width: 96, height: 96 }), BG)
  assert.ok(ratio > 0.0005, `dashed line UVs should sample the opaque map region (${ratio})`)
})

test('LineDashedMaterial alphaMap samples reconstructed secondary UVs', () => {
  const alphaMap = rgbaTexture([
    255, 0, 255, 255,
    255, 255, 255, 255,
  ], 2, 1)

  function renderLine(channel) {
    alphaMap.channel = channel
    const geom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-1.5, 0, 0),
      new THREE.Vector3(1.5, 0, 0),
    ])
    geom.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([
      0.25, 0.5,
      0.25, 0.5,
    ]), 2))
    geom.setAttribute('uv1', new THREE.BufferAttribute(new Float32Array([
      0.75, 0.5,
      0.75, 0.5,
    ]), 2))

    const material = new THREE.LineDashedMaterial({
      color: 0xffffff,
      alphaTest: 0.5,
      dashSize: 0.5,
      gapSize: 0.2,
      scale: 1,
    })
    material.alphaMap = alphaMap
    const line = new THREE.Line(geom, material)
    line.computeLineDistances()

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0.1, 0.1, 0.1)
    scene.add(line)
    return nonBackgroundRatio(renderRgba(scene, makeCamera(), { width: 96, height: 96 }), BG)
  }

  const primary = renderLine(0)
  const secondary = renderLine(1)
  assert.ok(secondary > 0.0005, `dashed line alphaMap channel=1 should sample reconstructed uv1 (${secondary})`)
  assert.ok(primary < secondary * 0.35, `dashed line alphaMap channel=0 should sample the transparent primary UV (${primary} vs ${secondary})`)
})

test('LineDashedMaterial map and alphaMap can sample distinct non-primary UV channels', () => {
  const map = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  map.channel = 1
  const alphaMap = rgbaTexture([
    255, 255, 255, 255,
    255, 0, 255, 255,
  ], 2, 1)
  alphaMap.channel = 2

  const geom = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-1.5, 0, 0),
    new THREE.Vector3(1.5, 0, 0),
  ])
  geom.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([
    0.25, 0.5,
    0.25, 0.5,
  ]), 2))
  geom.setAttribute('uv1', new THREE.BufferAttribute(new Float32Array([
    0.75, 0.5,
    0.75, 0.5,
  ]), 2))
  geom.setAttribute('uv2', new THREE.BufferAttribute(new Float32Array([
    0.25, 0.5,
    0.25, 0.5,
  ]), 2))

  const material = new THREE.LineDashedMaterial({
    alphaMap,
    alphaTest: 0.5,
    color: 0xffffff,
    dashSize: 0.5,
    gapSize: 0.2,
    map,
    scale: 1,
  })
  const line = new THREE.Line(geom, material)
  line.computeLineDistances()

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(line)

  const rgba = renderRgba(scene, makeCamera(), { width: 96, height: 96 })
  const greenPixels = countRegionPixels(
    rgba,
    96,
    96,
    8,
    42,
    88,
    54,
    (r, g, b) => g > 80 && g > r + 40 && g > b + 40,
  )
  assert.ok(greenPixels > 4, `dashed line map channel=1 should render green while alphaMap channel=2 keeps it visible (${greenPixels})`)
})

test('LineDashedMaterial map applies texture UV transforms', () => {
  const map = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  map.offset.set(0.5, 0)
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter

  const geom = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-1.5, 0, 0),
    new THREE.Vector3(1.5, 0, 0),
  ])
  geom.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([
    0.25, 0.5,
    0.25, 0.5,
  ]), 2))

  const line = new THREE.Line(geom, new THREE.LineDashedMaterial({
    color: 0xffffff,
    map,
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

  const image = renderRgba(scene, camera, { width: 96, height: 96 })
  const greenPixels = countRegionPixels(image, 96, 96, 0, 0, 96, 96, (r, g, b) => g > r + 40 && g > b + 40)
  assert.ok(greenPixels > 2, `dashed line map offset should shift reconstructed UVs from red to green (${greenPixels})`)
})

test('LineDashedMaterial map honors explicit texture matrices', () => {
  const map = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter
  setTextureMatrixOffset(map, 0.5)

  const geom = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-1.5, 0, 0),
    new THREE.Vector3(1.5, 0, 0),
  ])
  geom.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([
    0.25, 0.5,
    0.25, 0.5,
  ]), 2))

  const line = new THREE.Line(geom, new THREE.LineDashedMaterial({
    color: 0xffffff,
    map,
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

  const image = renderRgba(scene, camera, { width: 96, height: 96 })
  const greenPixels = countRegionPixels(image, 96, 96, 0, 0, 96, 96, (r, g, b) => g > r + 40 && g > b + 40)
  assert.ok(greenPixels > 2, `explicit dashed line map matrix should shift reconstructed UVs from red to green (${greenPixels})`)
})

test('LineDashedMaterial map honors horizontal and vertical repeat wrapping', () => {
  const renderer = new Renderer()

  function renderWithWrapping({ wrapS = THREE.ClampToEdgeWrapping, wrapT = THREE.ClampToEdgeWrapping, vertical = false }) {
    const map = vertical
      ? rgbaTexture([
        0, 255, 0, 255,
        0, 255, 0, 255,
        255, 0, 0, 255,
        255, 0, 0, 255,
      ], 2, 2)
      : rgbaTexture([
        0, 255, 0, 255,
        255, 0, 0, 255,
        0, 255, 0, 255,
        255, 0, 0, 255,
      ], 2, 2)
    map.wrapS = wrapS
    map.wrapT = wrapT
    map.magFilter = THREE.NearestFilter
    map.minFilter = THREE.NearestFilter

    const geom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-1.5, 0, 0),
      new THREE.Vector3(1.5, 0, 0),
    ])
    geom.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([
      vertical ? 0.25 : 1.25, vertical ? 1.25 : 0.25,
      vertical ? 0.25 : 1.25, vertical ? 1.25 : 0.25,
    ]), 2))

    const line = new THREE.Line(geom, new THREE.LineDashedMaterial({
      color: 0xffffff,
      dashSize: 10,
      gapSize: 0,
      linewidth: 8,
      map,
      scale: 1,
    }))
    line.computeLineDistances()

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(line)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return renderer.render(scene, camera, { width: 96, height: 96, format: 'rgba' })
  }

  const clamped = renderWithWrapping({ wrapS: THREE.ClampToEdgeWrapping })
  const repeated = renderWithWrapping({ wrapS: THREE.RepeatWrapping })
  const mirrored = renderWithWrapping({ wrapS: THREE.MirroredRepeatWrapping })
  const clampedRed = countRegionPixels(clamped, 96, 96, 0, 0, 96, 96, (r, g, b) => r > g + 40 && r > b + 40)
  const repeatedGreen = countRegionPixels(repeated, 96, 96, 0, 0, 96, 96, (r, g, b) => g > r + 40 && g > b + 40)
  const mirroredRed = countRegionPixels(mirrored, 96, 96, 0, 0, 96, 96, (r, g, b) => r > g + 40 && r > b + 40)
  assert.ok(clampedRed > 600, `clamped dashed-line map U coordinates should sample the red edge texel (${clampedRed})`)
  assert.ok(repeatedGreen > 600, `repeated dashed-line map U coordinates should wrap to the green texel (${repeatedGreen})`)
  assert.ok(mirroredRed > 600, `mirrored dashed-line map U coordinates should reflect to the red texel (${mirroredRed})`)

  const clampedVertical = renderWithWrapping({ wrapT: THREE.ClampToEdgeWrapping, vertical: true })
  const repeatedVertical = renderWithWrapping({ wrapT: THREE.RepeatWrapping, vertical: true })
  const mirroredVertical = renderWithWrapping({ wrapT: THREE.MirroredRepeatWrapping, vertical: true })
  const clampedVerticalRed = countRegionPixels(clampedVertical, 96, 96, 0, 0, 96, 96, (r, g, b) => r > g + 40 && r > b + 40)
  const repeatedVerticalGreen = countRegionPixels(repeatedVertical, 96, 96, 0, 0, 96, 96, (r, g, b) => g > r + 40 && g > b + 40)
  const mirroredVerticalRed = countRegionPixels(mirroredVertical, 96, 96, 0, 0, 96, 96, (r, g, b) => r > g + 40 && r > b + 40)
  assert.ok(clampedVerticalRed > 600, `clamped dashed-line map V coordinates should sample the red edge texel (${clampedVerticalRed})`)
  assert.ok(repeatedVerticalGreen > 600, `repeated dashed-line map V coordinates should wrap to the green texel (${repeatedVerticalGreen})`)
  assert.ok(mirroredVerticalRed > 600, `mirrored dashed-line map V coordinates should reflect to the red texel (${mirroredVerticalRed})`)
})

test('LineDashedMaterial alphaMap honors horizontal and vertical repeat wrapping', () => {
  const renderer = new Renderer()

  function renderWithWrapping({ wrapS = THREE.ClampToEdgeWrapping, wrapT = THREE.ClampToEdgeWrapping, vertical = false }) {
    const alphaMap = vertical
      ? rgbaTexture([
        255, 255, 255, 255,
        255, 255, 255, 255,
        255, 0, 255, 255,
        255, 0, 255, 255,
      ], 2, 2)
      : rgbaTexture([
        255, 255, 255, 255,
        255, 0, 255, 255,
        255, 255, 255, 255,
        255, 0, 255, 255,
      ], 2, 2)
    alphaMap.wrapS = wrapS
    alphaMap.wrapT = wrapT
    alphaMap.magFilter = THREE.NearestFilter
    alphaMap.minFilter = THREE.NearestFilter

    const geom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-1.5, 0, 0),
      new THREE.Vector3(1.5, 0, 0),
    ])
    geom.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([
      vertical ? 0.25 : 1.25, vertical ? 1.25 : 0.25,
      vertical ? 0.25 : 1.25, vertical ? 1.25 : 0.25,
    ]), 2))

    const material = new THREE.LineDashedMaterial({
      alphaTest: 0.5,
      color: 0x00ff00,
      dashSize: 10,
      gapSize: 0,
      linewidth: 8,
      scale: 1,
    })
    material.alphaMap = alphaMap
    const line = new THREE.Line(geom, material)
    line.computeLineDistances()

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 1)
    scene.add(line)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return renderer.render(scene, camera, { width: 96, height: 96, format: 'rgba' })
  }

  const clamped = renderWithWrapping({ wrapS: THREE.ClampToEdgeWrapping })
  const repeated = renderWithWrapping({ wrapS: THREE.RepeatWrapping })
  const mirrored = renderWithWrapping({ wrapS: THREE.MirroredRepeatWrapping })
  const clampedGreen = countRegionPixels(clamped, 96, 96, 0, 0, 96, 96, (r, g, b) => g > b + 40 && g > r + 40)
  const repeatedGreen = countRegionPixels(repeated, 96, 96, 0, 0, 96, 96, (r, g, b) => g > b + 40 && g > r + 40)
  const mirroredGreen = countRegionPixels(mirrored, 96, 96, 0, 0, 96, 96, (r, g, b) => g > b + 40 && g > r + 40)
  assert.ok(clampedGreen < 20, `clamped dashed-line alphaMap U coordinates should discard the line (${clampedGreen})`)
  assert.ok(repeatedGreen > 600, `repeated dashed-line alphaMap U coordinates should wrap to the opaque texel (${repeatedGreen})`)
  assert.ok(mirroredGreen < 20, `mirrored dashed-line alphaMap U coordinates should reflect to the transparent texel (${mirroredGreen})`)

  const clampedVertical = renderWithWrapping({ wrapT: THREE.ClampToEdgeWrapping, vertical: true })
  const repeatedVertical = renderWithWrapping({ wrapT: THREE.RepeatWrapping, vertical: true })
  const mirroredVertical = renderWithWrapping({ wrapT: THREE.MirroredRepeatWrapping, vertical: true })
  const clampedVerticalGreen = countRegionPixels(clampedVertical, 96, 96, 0, 0, 96, 96, (r, g, b) => g > b + 40 && g > r + 40)
  const repeatedVerticalGreen = countRegionPixels(repeatedVertical, 96, 96, 0, 0, 96, 96, (r, g, b) => g > b + 40 && g > r + 40)
  const mirroredVerticalGreen = countRegionPixels(mirroredVertical, 96, 96, 0, 0, 96, 96, (r, g, b) => g > b + 40 && g > r + 40)
  assert.ok(clampedVerticalGreen < 20, `clamped dashed-line alphaMap V coordinates should discard the line (${clampedVerticalGreen})`)
  assert.ok(repeatedVerticalGreen > 600, `repeated dashed-line alphaMap V coordinates should wrap to the opaque texel (${repeatedVerticalGreen})`)
  assert.ok(mirroredVerticalGreen < 20, `mirrored dashed-line alphaMap V coordinates should reflect to the transparent texel (${mirroredVerticalGreen})`)
})

test('LineDashedMaterial map samples the selected secondary UV channel', () => {
  const map = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  map.channel = 1

  const geom = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-1.5, 0, 0),
    new THREE.Vector3(1.5, 0, 0),
  ])
  geom.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([
    0.25, 0.5,
    0.25, 0.5,
  ]), 2))
  geom.setAttribute('uv1', new THREE.BufferAttribute(new Float32Array([
    0.75, 0.5,
    0.75, 0.5,
  ]), 2))

  const line = new THREE.Line(geom, new THREE.LineDashedMaterial({
    color: 0xffffff,
    map,
    dashSize: 0.5,
    gapSize: 0.2,
    scale: 1,
  }))
  line.computeLineDistances()

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(line)

  const rgba = renderRgba(scene, makeCamera(), { width: 96, height: 96 })
  const greenPixels = countRegionPixels(rgba, 96, 96, 0, 0, 96, 96, (r, g, b) => g > r + 40 && g > b + 40)
  assert.ok(greenPixels > 2, `dashed line map channel=1 should sample uv1 green texel (${greenPixels})`)
})

test('line and point maps sample texture channel 3 from uv3 attributes', () => {
  const map = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter
  map.channel = 3

  function setUvChannels(geometry, count) {
    const redUvs = new Float32Array(count * 2)
    const greenUvs = new Float32Array(count * 2)
    for (let i = 0; i < count; i += 1) {
      redUvs[i * 2] = 0.25
      redUvs[i * 2 + 1] = 0.5
      greenUvs[i * 2] = 0.75
      greenUvs[i * 2 + 1] = 0.5
    }
    geometry.setAttribute('uv', new THREE.BufferAttribute(redUvs.slice(), 2))
    geometry.setAttribute('uv1', new THREE.BufferAttribute(redUvs.slice(), 2))
    geometry.setAttribute('uv2', new THREE.BufferAttribute(redUvs.slice(), 2))
    geometry.setAttribute('uv3', new THREE.BufferAttribute(greenUvs, 2))
  }

  function renderLine(material) {
    const geometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-1.5, 0, 0),
      new THREE.Vector3(1.5, 0, 0),
    ])
    setUvChannels(geometry, 2)
    const line = new THREE.Line(geometry, material)
    if (material.isLineDashedMaterial === true) line.computeLineDistances()

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(line)
    return renderRgba(scene, makeCamera(), { width: 96, height: 96 })
  }

  const basic = renderLine(new THREE.LineBasicMaterial({ color: 0xffffff, map }))
  const dashed = renderLine(new THREE.LineDashedMaterial({
    color: 0xffffff,
    dashSize: 4,
    gapSize: 0,
    map,
    scale: 1,
  }))
  const greenLinePixels = countRegionPixels(basic, 96, 96, 0, 42, 96, 54, (r, g, b) => g > r + 40 && g > b + 40)
  const greenDashedPixels = countRegionPixels(dashed, 96, 96, 0, 42, 96, 54, (r, g, b) => g > r + 40 && g > b + 40)
  assert.ok(greenLinePixels > 2, `line map channel=3 should sample uv3 green texel (${greenLinePixels})`)
  assert.ok(greenDashedPixels > 2, `dashed line map channel=3 should sample uv3 green texel (${greenDashedPixels})`)

  const pointGeometry = new THREE.BufferGeometry()
  pointGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))
  setUvChannels(pointGeometry, 1)
  const pointScene = new THREE.Scene()
  pointScene.background = new THREE.Color(0, 0, 0)
  pointScene.add(new THREE.Points(pointGeometry, new THREE.PointsMaterial({
    color: 0xffffff,
    map,
    size: 48,
    sizeAttenuation: false,
  })))
  const pointMean = meanRegion(renderRgba(pointScene, makeCamera(), { width: 96, height: 96 }), 96, 96, 40, 40, 56, 56)
  assert.ok(pointMean.g > pointMean.r + 60, `point map channel=3 should sample uv3 green texel (${pointMean.g} vs ${pointMean.r})`)
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

test('LineSegments with InstancedBufferGeometry default instanceCount expands offsets and colors', () => {
  const geometry = new THREE.InstancedBufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.25, 0, 0,
    0.25, 0, 0,
  ]), 3))
  geometry.setAttribute('instanceOffset', new THREE.InstancedBufferAttribute(new Float32Array([
    -0.45, 0, 0,
    0.45, 0, 0,
  ]), 3))
  geometry.setAttribute('color', new THREE.InstancedBufferAttribute(new Float32Array([
    1, 0, 0,
    0, 1, 0,
  ]), 3))

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({
    color: 0xffffff,
    vertexColors: true,
  })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 96, height: 96 })
  const redPixels = countRegionPixels(rgba, 96, 96, 14, 42, 44, 54, (r, g, b) => r > g + 30 && r > b + 30)
  const greenPixels = countRegionPixels(rgba, 96, 96, 52, 42, 82, 54, (r, g, b) => g > r + 30 && g > b + 30)
  assert.ok(redPixels > 2, `left instanced line should render red pixels (${redPixels})`)
  assert.ok(greenPixels > 2, `right instanced line should render green pixels (${greenPixels})`)
})

test('Line and LineLoop with InstancedBufferGeometry expand offsets and colors', () => {
  function renderLine(makeObject) {
    const geometry = new THREE.InstancedBufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
      -0.25, 0, 0,
      0.25, 0, 0,
    ]), 3))
    geometry.setAttribute('instanceOffset', new THREE.InstancedBufferAttribute(new Float32Array([
      -0.45, 0, 0,
      0.45, 0, 0,
    ]), 3))
    geometry.setAttribute('color', new THREE.InstancedBufferAttribute(new Float32Array([
      1, 0, 0,
      0, 1, 0,
    ]), 3))

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(makeObject(geometry, new THREE.LineBasicMaterial({
      color: 0xffffff,
      linewidth: 8,
      vertexColors: true,
    })))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return renderRgba(scene, camera, { width: 96, height: 96 })
  }

  for (const [label, makeObject] of [
    ['Line', (geometry, material) => new THREE.Line(geometry, material)],
    ['LineLoop', (geometry, material) => new THREE.LineLoop(geometry, material)],
  ]) {
    const rgba = renderLine(makeObject)
    const redPixels = countRegionPixels(rgba, 96, 96, 14, 40, 44, 56, (r, g, b) => r > g + 30 && r > b + 30)
    const greenPixels = countRegionPixels(rgba, 96, 96, 52, 40, 82, 56, (r, g, b) => g > r + 30 && g > b + 30)
    const bridgePixels = countRegionPixels(rgba, 96, 96, 43, 40, 53, 56, (r, g, b) => r > 30 || g > 30 || b > 30)
    assert.ok(redPixels > 6, `${label} should render red pixels from the first instanced color (${redPixels})`)
    assert.ok(greenPixels > 6, `${label} should render green pixels from the second instanced color (${greenPixels})`)
    assert.ok(bridgePixels < 4, `${label} should not connect separate instances across the center gap (${bridgePixels})`)
  }
})

test('LineLoop with InstancedBufferGeometry preserves per-instance closing segments', () => {
  const geometry = new THREE.InstancedBufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.25, -0.35, 0,
    0.25, -0.35, 0,
    0.25, 0.35, 0,
  ]), 3))
  geometry.setAttribute('instanceOffset', new THREE.InstancedBufferAttribute(new Float32Array([
    -0.45, 0, 0,
    0.45, 0, 0,
  ]), 3))
  geometry.setAttribute('color', new THREE.InstancedBufferAttribute(new Float32Array([
    1, 0, 0,
    0, 1, 0,
  ]), 3))

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.LineLoop(geometry, new THREE.LineBasicMaterial({
    color: 0xffffff,
    linewidth: 6,
    vertexColors: true,
  })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 96, height: 96 })
  const redClosingPixels = countRegionPixels(rgba, 96, 96, 22, 42, 36, 58, (r, g, b) => r > g + 30 && r > b + 30)
  const greenClosingPixels = countRegionPixels(rgba, 96, 96, 60, 42, 74, 58, (r, g, b) => g > r + 30 && g > b + 30)
  assert.ok(redClosingPixels > 6, `left instanced LineLoop should render its red closing segment (${redClosingPixels})`)
  assert.ok(greenClosingPixels > 6, `right instanced LineLoop should render its green closing segment (${greenClosingPixels})`)
})

test('LineSegments with InstancedBufferGeometry honor meshPerAttribute repeat values for colors', () => {
  const geometry = new THREE.InstancedBufferGeometry()
  geometry.instanceCount = 4
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.06, 0, 0,
    0.06, 0, 0,
  ]), 3))
  geometry.setAttribute('instanceOffset', new THREE.InstancedBufferAttribute(new Float32Array([
    -0.6, 0, 0,
    -0.2, 0, 0,
    0.2, 0, 0,
    0.6, 0, 0,
  ]), 3))

  const colors = new THREE.InstancedBufferAttribute(new Float32Array([
    1, 0, 0,
    0, 1, 0,
  ]), 3)
  colors.meshPerAttribute = 2
  geometry.setAttribute('color', colors)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({
    color: 0xffffff,
    linewidth: 8,
    vertexColors: true,
  })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 96, height: 96 })
  const firstRed = countRegionPixels(rgba, 96, 96, 18, 42, 32, 54, (r, g, b) => r > g + 30 && r > b + 30)
  const secondRed = countRegionPixels(rgba, 96, 96, 34, 42, 47, 54, (r, g, b) => r > g + 30 && r > b + 30)
  const firstGreen = countRegionPixels(rgba, 96, 96, 50, 42, 63, 54, (r, g, b) => g > r + 30 && g > b + 30)
  const secondGreen = countRegionPixels(rgba, 96, 96, 65, 42, 79, 54, (r, g, b) => g > r + 30 && g > b + 30)
  assert.ok(firstRed > 6, `first repeated line color should render red pixels (${firstRed})`)
  assert.ok(secondRed > 6, `second repeated line color should render red pixels (${secondRed})`)
  assert.ok(firstGreen > 6, `first repeated line color should render green pixels (${firstGreen})`)
  assert.ok(secondGreen > 6, `second repeated line color should render green pixels (${secondGreen})`)
})

test('LineSegments with InstancedBufferGeometry default instanceCount expands instanced map UV attributes', () => {
  const geometry = new THREE.InstancedBufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.25, 0, 0,
    0.25, 0, 0,
  ]), 3))
  geometry.setAttribute('instanceOffset', new THREE.InstancedBufferAttribute(new Float32Array([
    -0.45, 0, 0,
    0.45, 0, 0,
  ]), 3))
  geometry.setAttribute('uv', new THREE.InstancedBufferAttribute(new Float32Array([
    0.25, 0.5,
    0.75, 0.5,
  ]), 2))

  const map = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({
    color: 0xffffff,
    linewidth: 8,
    map,
  })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 96, height: 96 })
  const redPixels = countRegionPixels(rgba, 96, 96, 12, 40, 44, 56, (r, g, b) => r > g + 30 && r > b + 30)
  const greenPixels = countRegionPixels(rgba, 96, 96, 52, 40, 84, 56, (r, g, b) => g > r + 30 && g > b + 30)
  assert.ok(redPixels > 4, `left instanced line uv should sample red (${redPixels})`)
  assert.ok(greenPixels > 4, `right instanced line uv should sample green (${greenPixels})`)
})

test('Line and LineLoop with InstancedBufferGeometry expand instanced map UV attributes', () => {
  const map = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter

  function renderLine(makeObject) {
    const geometry = new THREE.InstancedBufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
      -0.25, 0, 0,
      0.25, 0, 0,
    ]), 3))
    geometry.setAttribute('instanceOffset', new THREE.InstancedBufferAttribute(new Float32Array([
      -0.45, 0, 0,
      0.45, 0, 0,
    ]), 3))
    geometry.setAttribute('uv', new THREE.InstancedBufferAttribute(new Float32Array([
      0.25, 0.5,
      0.75, 0.5,
    ]), 2))

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(makeObject(geometry, new THREE.LineBasicMaterial({
      color: 0xffffff,
      linewidth: 8,
      map,
    })))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return renderRgba(scene, camera, { width: 96, height: 96 })
  }

  for (const [label, makeObject] of [
    ['Line', (geometry, material) => new THREE.Line(geometry, material)],
    ['LineLoop', (geometry, material) => new THREE.LineLoop(geometry, material)],
  ]) {
    const rgba = renderLine(makeObject)
    const redPixels = countRegionPixels(rgba, 96, 96, 12, 40, 44, 56, (r, g, b) => r > g + 30 && r > b + 30)
    const greenPixels = countRegionPixels(rgba, 96, 96, 52, 40, 84, 56, (r, g, b) => g > r + 30 && g > b + 30)
    const bridgePixels = countRegionPixels(rgba, 96, 96, 43, 40, 53, 56, (r, g, b) => r > 30 || g > 30 || b > 30)
    assert.ok(redPixels > 4, `${label} instanced line uv should sample red (${redPixels})`)
    assert.ok(greenPixels > 4, `${label} instanced line uv should sample green (${greenPixels})`)
    assert.ok(bridgePixels < 4, `${label} instanced line uv should not connect separate instances (${bridgePixels})`)
  }
})

test('LineDashedMaterial with InstancedBufferGeometry default instanceCount expands instanced map UV attributes', () => {
  const geometry = new THREE.InstancedBufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.25, 0, 0,
    0.25, 0, 0,
  ]), 3))
  geometry.setAttribute('instanceOffset', new THREE.InstancedBufferAttribute(new Float32Array([
    -0.45, 0, 0,
    0.45, 0, 0,
  ]), 3))
  geometry.setAttribute('uv', new THREE.InstancedBufferAttribute(new Float32Array([
    0.25, 0.5,
    0.75, 0.5,
  ]), 2))
  geometry.setAttribute('lineDistance', new THREE.BufferAttribute(new Float32Array([0, 1]), 1))

  const map = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.LineSegments(geometry, new THREE.LineDashedMaterial({
    color: 0xffffff,
    dashSize: 4,
    gapSize: 0,
    linewidth: 8,
    map,
    scale: 1,
  })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 96, height: 96 })
  const redPixels = countRegionPixels(rgba, 96, 96, 12, 40, 44, 56, (r, g, b) => r > g + 30 && r > b + 30)
  const greenPixels = countRegionPixels(rgba, 96, 96, 52, 40, 84, 56, (r, g, b) => g > r + 30 && g > b + 30)
  assert.ok(redPixels > 4, `left instanced dashed line uv should sample red (${redPixels})`)
  assert.ok(greenPixels > 4, `right instanced dashed line uv should sample green (${greenPixels})`)
})

test('LineDashedMaterial with InstancedBufferGeometry default instanceCount expands offsets and colors', () => {
  const geometry = new THREE.InstancedBufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.25, 0, 0,
    0.25, 0, 0,
  ]), 3))
  geometry.setAttribute('instanceOffset', new THREE.InstancedBufferAttribute(new Float32Array([
    -0.45, 0, 0,
    0.45, 0, 0,
  ]), 3))
  geometry.setAttribute('color', new THREE.InstancedBufferAttribute(new Float32Array([
    1, 0, 0,
    0, 1, 0,
  ]), 3))
  geometry.setAttribute('lineDistance', new THREE.BufferAttribute(new Float32Array([0, 1]), 1))

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.LineSegments(geometry, new THREE.LineDashedMaterial({
    color: 0xffffff,
    vertexColors: true,
    dashSize: 0.2,
    gapSize: 0.1,
  })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 96, height: 96 })
  const redPixels = countRegionPixels(rgba, 96, 96, 10, 42, 42, 54, (r, g, b) => r > g + 30 && r > b + 30)
  const greenPixels = countRegionPixels(rgba, 96, 96, 54, 42, 86, 54, (r, g, b) => g > r + 30 && g > b + 30)
  assert.ok(redPixels > 2, `left instanced dashed line should render red pixels (${redPixels})`)
  assert.ok(greenPixels > 2, `right instanced dashed line should render green pixels (${greenPixels})`)
})

test('LineDashedMaterial with InstancedBufferGeometry honors meshPerAttribute repeat values for colors', () => {
  const geometry = new THREE.InstancedBufferGeometry()
  geometry.instanceCount = 4
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.06, 0, 0,
    0.06, 0, 0,
  ]), 3))
  geometry.setAttribute('instanceOffset', new THREE.InstancedBufferAttribute(new Float32Array([
    -0.6, 0, 0,
    -0.2, 0, 0,
    0.2, 0, 0,
    0.6, 0, 0,
  ]), 3))

  const colors = new THREE.InstancedBufferAttribute(new Float32Array([
    1, 0, 0,
    0, 1, 0,
  ]), 3)
  colors.meshPerAttribute = 2
  geometry.setAttribute('color', colors)
  geometry.setAttribute('lineDistance', new THREE.BufferAttribute(new Float32Array([0, 1]), 1))

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.LineSegments(geometry, new THREE.LineDashedMaterial({
    color: 0xffffff,
    dashSize: 4,
    gapSize: 0,
    linewidth: 8,
    scale: 1,
    vertexColors: true,
  })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 96, height: 96 })
  const firstRed = countRegionPixels(rgba, 96, 96, 18, 42, 32, 54, (r, g, b) => r > g + 30 && r > b + 30)
  const secondRed = countRegionPixels(rgba, 96, 96, 34, 42, 47, 54, (r, g, b) => r > g + 30 && r > b + 30)
  const firstGreen = countRegionPixels(rgba, 96, 96, 50, 42, 63, 54, (r, g, b) => g > r + 30 && g > b + 30)
  const secondGreen = countRegionPixels(rgba, 96, 96, 65, 42, 79, 54, (r, g, b) => g > r + 30 && g > b + 30)
  assert.ok(firstRed > 6, `first repeated dashed-line color should render red pixels (${firstRed})`)
  assert.ok(secondRed > 6, `second repeated dashed-line color should render red pixels (${secondRed})`)
  assert.ok(firstGreen > 6, `first repeated dashed-line color should render green pixels (${firstGreen})`)
  assert.ok(secondGreen > 6, `second repeated dashed-line color should render green pixels (${secondGreen})`)
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
  const buf = getRenderer().render(scene, camera, { width: SIZE, height: SIZE })
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

test('PointsMaterial untextured billboards keep square point-sprite corners', () => {
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({
    color: 0xffffff,
    size: 48,
    sizeAttenuation: false,
  })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 96, height: 96 })
  const center = meanRegion(rgba, 96, 96, 44, 44, 52, 52)
  const corner = meanRegion(rgba, 96, 96, 28, 28, 36, 36)

  assert.ok(center.r > 180 && center.g > 180 && center.b > 180, `point billboard center should be white (${center.r}, ${center.g}, ${center.b})`)
  assert.ok(corner.r > 180 && corner.g > 180 && corner.b > 180, `untextured point billboard corners should remain square/visible (${corner.r}, ${corner.g}, ${corner.b})`)
})

test('PointsMaterial clamps oversized billboard bounds to a WebGL-style point size cap', () => {
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
    return nonBackgroundBounds(renderRgba(scene, camera, { width: 128, height: 128 }), 128, 128, [0, 0, 0])
  }

  const capped = renderPoint(64)
  const oversized = renderPoint(200)
  assert.ok(capped.width >= 60 && capped.width <= 68, `64px point should render near the cap (${capped.width})`)
  assert.ok(capped.height >= 60 && capped.height <= 68, `64px point should render near the cap (${capped.height})`)
  assert.ok(oversized.width <= capped.width + 2, `oversized point width should clamp near cap (${oversized.width} vs ${capped.width})`)
  assert.ok(oversized.height <= capped.height + 2, `oversized point height should clamp near cap (${oversized.height} vs ${capped.height})`)
})

test('PointsMaterial orthographic size is depth independent', () => {
  function renderPoint(z) {
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, z]), 3))

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({
      color: 0xffffff,
      size: 20,
    })))

    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 20)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return nonBackgroundBounds(
      renderRgba(scene, camera, { width: 96, height: 96 }),
      96,
      96,
      [0, 0, 0],
    )
  }

  const near = renderPoint(0)
  const far = renderPoint(-3)
  assert.ok(
    near.width >= 16 && near.width <= 26,
    `orthographic point should render near its pixel size (${near.width})`,
  )
  assert.ok(
    far.width >= 16 && far.width <= 26,
    `far orthographic point should render near its pixel size (${far.width})`,
  )
  assert.ok(
    Math.abs(near.width - far.width) <= 2,
    `orthographic point size should not scale with depth (${near.width} vs ${far.width})`,
  )
})

test('PointsMaterial perspective size attenuation shrinks distant point billboards', () => {
  function renderPoint(z) {
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, z]), 3))

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({
      color: 0xffffff,
      size: 0.8,
    })))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return nonBackgroundBounds(renderRgba(scene, camera, { width: 96, height: 96 }), 96, 96, [0, 0, 0])
  }

  const near = renderPoint(0)
  const far = renderPoint(-3)
  assert.ok(
    near.width >= far.width * 1.7,
    `perspective point size should shrink with distance (${near.width} vs ${far.width})`,
  )
  assert.ok(
    near.height >= far.height * 1.7,
    `perspective point height should shrink with distance (${near.height} vs ${far.height})`,
  )
})

test('PointsMaterial opacity blends billboard color over the background', () => {
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 1)
  scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({
    color: 0xff0000,
    opacity: 0.5,
    transparent: true,
    size: 34,
    sizeAttenuation: false,
  })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRegion(renderRgba(scene, camera, { width: 64, height: 64 }), 64, 64, 24, 24, 40, 40)
  assert.ok(mean.r > 50, `semi-transparent point should contribute red over blue (${mean.r})`)
  assert.ok(mean.b > 80, `semi-transparent point should preserve blue background contribution (${mean.b})`)
  assert.ok(mean.g < 40, `semi-transparent point should not add green (${mean.g})`)
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

test('PointsMaterial map applies texture UV transforms', () => {
  const map = rgbaTexture([
    255, 0, 0, 255,
    255, 0, 0, 255,
    0, 255, 0, 255,
    0, 255, 0, 255,
  ], 4, 1)
  map.offset.set(0.5, 0)
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({
    color: 0xffffff,
    map,
    size: 48,
    sizeAttenuation: false,
  })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRegion(renderRgba(scene, camera, { width: 96, height: 96 }), 96, 96, 30, 44, 38, 52)
  assert.ok(mean.g > mean.r + 40, `point map offset should shift left point-sprite UVs from red to green (${mean.g} vs ${mean.r})`)
})

test('PointsMaterial map honors explicit texture matrices', () => {
  const map = rgbaTexture([
    255, 0, 0, 255,
    255, 0, 0, 255,
    0, 255, 0, 255,
    0, 255, 0, 255,
  ], 4, 1)
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter
  setTextureMatrixOffset(map, 0.5)

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({
    color: 0xffffff,
    map,
    size: 48,
    sizeAttenuation: false,
  })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRegion(renderRgba(scene, camera, { width: 96, height: 96 }), 96, 96, 30, 44, 38, 52)
  assert.ok(mean.g > mean.r + 40, `explicit point map matrix should shift left point-sprite UVs from red to green (${mean.g} vs ${mean.r})`)
})

test('PointsMaterial map honors horizontal and vertical repeat wrapping', () => {
  const renderer = new Renderer()

  function renderWithWrapping({ wrapS = THREE.ClampToEdgeWrapping, wrapT = THREE.ClampToEdgeWrapping, vertical = false }) {
    const map = vertical
      ? rgbaTexture([
        0, 255, 0, 255,
        0, 255, 0, 255,
        255, 0, 0, 255,
        255, 0, 0, 255,
      ], 2, 2)
      : rgbaTexture([
        0, 255, 0, 255,
        255, 0, 0, 255,
        0, 255, 0, 255,
        255, 0, 0, 255,
      ], 2, 2)
    map.wrapS = wrapS
    map.wrapT = wrapT
    map.offset.set(vertical ? 0 : 1, vertical ? 1 : 0)
    map.magFilter = THREE.NearestFilter
    map.minFilter = THREE.NearestFilter

    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({
      color: 0xffffff,
      map,
      size: 48,
      sizeAttenuation: false,
    })))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    const rgba = renderer.render(scene, camera, { width: 96, height: 96, format: 'rgba' })
    return vertical
      ? meanRegion(rgba, 96, 96, 44, 58, 52, 66)
      : meanRegion(rgba, 96, 96, 30, 44, 38, 52)
  }

  const clamped = renderWithWrapping({ wrapS: THREE.ClampToEdgeWrapping })
  const repeated = renderWithWrapping({ wrapS: THREE.RepeatWrapping })
  const mirrored = renderWithWrapping({ wrapS: THREE.MirroredRepeatWrapping })
  assert.ok(clamped.r > clamped.g + 100, `clamped point map U coordinates should sample the red edge texel (${clamped.r} vs ${clamped.g})`)
  assert.ok(repeated.g > repeated.r + 60, `repeated point map U coordinates should wrap to the green texel (${repeated.g} vs ${repeated.r})`)
  assert.ok(mirrored.r > mirrored.g + 100, `mirrored point map U coordinates should reflect to the red texel (${mirrored.r} vs ${mirrored.g})`)

  const clampedVertical = renderWithWrapping({ wrapT: THREE.ClampToEdgeWrapping, vertical: true })
  const repeatedVertical = renderWithWrapping({ wrapT: THREE.RepeatWrapping, vertical: true })
  const mirroredVertical = renderWithWrapping({ wrapT: THREE.MirroredRepeatWrapping, vertical: true })
  assert.ok(clampedVertical.r > clampedVertical.g + 100, `clamped point map V coordinates should sample the red edge texel (${clampedVertical.r} vs ${clampedVertical.g})`)
  assert.ok(repeatedVertical.g > repeatedVertical.r + 60, `repeated point map V coordinates should wrap to the green texel (${repeatedVertical.g} vs ${repeatedVertical.r})`)
  assert.ok(mirroredVertical.r > mirroredVertical.g + 100, `mirrored point map V coordinates should reflect to the red texel (${mirroredVertical.r} vs ${mirroredVertical.g})`)
})

test('PointsMaterial alphaMap applies texture UV transforms', () => {
  const alphaMap = rgbaTexture([
    255, 0, 255, 255,
    255, 255, 255, 255,
  ], 2, 1)
  alphaMap.offset.set(0.5, 0)
  alphaMap.magFilter = THREE.NearestFilter
  alphaMap.minFilter = THREE.NearestFilter

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 1)
  scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({
    color: 0x00ff00,
    alphaMap,
    alphaTest: 0.5,
    size: 48,
    sizeAttenuation: false,
  })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRegion(renderRgba(scene, camera, { width: 96, height: 96 }), 96, 96, 30, 44, 38, 52)
  assert.ok(mean.g > mean.b + 40, `point alphaMap offset should shift left point-sprite UVs into the opaque texel (${mean.g} vs ${mean.b})`)
})

test('PointsMaterial alphaMap honors explicit texture matrices', () => {
  const alphaMap = rgbaTexture([
    255, 0, 255, 255,
    255, 255, 255, 255,
  ], 2, 1)
  alphaMap.magFilter = THREE.NearestFilter
  alphaMap.minFilter = THREE.NearestFilter
  setTextureMatrixOffset(alphaMap, 0.5)

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 1)
  scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({
    color: 0x00ff00,
    alphaMap,
    alphaTest: 0.5,
    size: 48,
    sizeAttenuation: false,
  })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRegion(renderRgba(scene, camera, { width: 96, height: 96 }), 96, 96, 30, 44, 38, 52)
  assert.ok(mean.g > mean.b + 40, `explicit point alphaMap matrix should shift left point-sprite UVs into the opaque texel (${mean.g} vs ${mean.b})`)
})

test('PointsMaterial alphaMap honors horizontal and vertical repeat wrapping', () => {
  const renderer = new Renderer()

  function renderWithWrapping({ wrapS = THREE.ClampToEdgeWrapping, wrapT = THREE.ClampToEdgeWrapping, vertical = false }) {
    const alphaMap = vertical
      ? rgbaTexture([
        255, 255, 255, 255,
        255, 255, 255, 255,
        255, 0, 255, 255,
        255, 0, 255, 255,
      ], 2, 2)
      : rgbaTexture([
        255, 255, 255, 255,
        255, 0, 255, 255,
        255, 255, 255, 255,
        255, 0, 255, 255,
      ], 2, 2)
    alphaMap.wrapS = wrapS
    alphaMap.wrapT = wrapT
    alphaMap.offset.set(vertical ? 0 : 1, vertical ? 1 : 0)
    alphaMap.magFilter = THREE.NearestFilter
    alphaMap.minFilter = THREE.NearestFilter

    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 1)
    scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({
      color: 0x00ff00,
      alphaMap,
      alphaTest: 0.5,
      size: 48,
      sizeAttenuation: false,
    })))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)

    const rgba = renderer.render(scene, camera, { width: 96, height: 96, format: 'rgba' })
    return vertical
      ? meanRegion(rgba, 96, 96, 44, 58, 52, 66)
      : meanRegion(rgba, 96, 96, 30, 44, 38, 52)
  }

  const clamped = renderWithWrapping({ wrapS: THREE.ClampToEdgeWrapping })
  const repeated = renderWithWrapping({ wrapS: THREE.RepeatWrapping })
  const mirrored = renderWithWrapping({ wrapS: THREE.MirroredRepeatWrapping })
  assert.ok(clamped.b > clamped.g + 40, `clamped point alphaMap U coordinates should discard against the blue background (${clamped.b} vs ${clamped.g})`)
  assert.ok(repeated.g > repeated.b + 40, `repeated point alphaMap U coordinates should wrap to the opaque texel (${repeated.g} vs ${repeated.b})`)
  assert.ok(mirrored.b > mirrored.g + 40, `mirrored point alphaMap U coordinates should reflect to the transparent texel (${mirrored.b} vs ${mirrored.g})`)

  const clampedVertical = renderWithWrapping({ wrapT: THREE.ClampToEdgeWrapping, vertical: true })
  const repeatedVertical = renderWithWrapping({ wrapT: THREE.RepeatWrapping, vertical: true })
  const mirroredVertical = renderWithWrapping({ wrapT: THREE.MirroredRepeatWrapping, vertical: true })
  assert.ok(clampedVertical.b > clampedVertical.g + 40, `clamped point alphaMap V coordinates should discard against the blue background (${clampedVertical.b} vs ${clampedVertical.g})`)
  assert.ok(repeatedVertical.g > repeatedVertical.b + 40, `repeated point alphaMap V coordinates should wrap to the opaque texel (${repeatedVertical.g} vs ${repeatedVertical.b})`)
  assert.ok(mirroredVertical.b > mirroredVertical.g + 40, `mirrored point alphaMap V coordinates should reflect to the transparent texel (${mirroredVertical.b} vs ${mirroredVertical.g})`)
})

test('PointsMaterial map decodes sRGB colorSpace before shading', () => {
  function renderColorSpace(colorSpace) {
    const map = solidTexture(128, 128, 128)
    map.colorSpace = colorSpace

    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({
      color: 0xffffff,
      map,
      size: 48,
      sizeAttenuation: false,
    })))

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 3)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, {
      width: 96,
      height: 96,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 96, 96, 40, 40, 56, 56)
  }

  const srgb = renderColorSpace(THREE.SRGBColorSpace)
  const linear = renderColorSpace(THREE.LinearSRGBColorSpace)
  assert.ok(linear.r > srgb.r + 15, `linear point map should render brighter than decoded sRGB texture (${linear.r} vs ${srgb.r})`)
})

test('PointsMaterial maps use geometry UV attributes when present', () => {
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))
  geometry.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([0.75, 0.5]), 2))

  const map = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({
    color: 0xffffff,
    map,
    size: 48,
    sizeAttenuation: false,
  })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 96, height: 96 })
  const left = meanRegion(rgba, 96, 96, 28, 40, 42, 56)
  const right = meanRegion(rgba, 96, 96, 54, 40, 68, 56)
  assert.ok(left.g > left.r + 60, `geometry UV should sample green across the point left half (${left.g} vs ${left.r})`)
  assert.ok(right.g > right.r + 60, `geometry UV should sample green across the point right half (${right.g} vs ${right.r})`)
})

test('PointsMaterial maps honor selected geometry UV channels', () => {
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))
  geometry.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([0.25, 0.5]), 2))
  geometry.setAttribute('uv1', new THREE.BufferAttribute(new Float32Array([0.75, 0.5]), 2))

  const map = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter
  map.channel = 1

  const alphaMap = rgbaTexture([
    255, 255, 255, 255,
    255, 255, 255, 255,
  ], 2, 1)
  alphaMap.channel = 1

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({
    alphaMap,
    color: 0xffffff,
    map,
    size: 48,
    sizeAttenuation: false,
  })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRegion(renderRgba(scene, camera, { width: 96, height: 96 }), 96, 96, 40, 40, 56, 56)
  assert.ok(mean.g > mean.r + 60, `selected point uv1 should sample green instead of primary red (${mean.g} vs ${mean.r})`)
})

test('PointsMaterial map and alphaMap can sample distinct geometry UV channels', () => {
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))
  geometry.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([0.25, 0.5]), 2))
  geometry.setAttribute('uv1', new THREE.BufferAttribute(new Float32Array([0.75, 0.5]), 2))
  geometry.setAttribute('uv2', new THREE.BufferAttribute(new Float32Array([0.75, 0.5]), 2))

  const map = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter
  map.channel = 1

  const alphaMap = rgbaTexture([
    255, 0, 255, 255,
    255, 255, 255, 255,
  ], 2, 1)
  alphaMap.magFilter = THREE.NearestFilter
  alphaMap.minFilter = THREE.NearestFilter
  alphaMap.channel = 2

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({
    alphaMap,
    alphaTest: 0.5,
    color: 0xffffff,
    map,
    size: 48,
    sizeAttenuation: false,
  })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const mean = meanRegion(renderRgba(scene, camera, { width: 96, height: 96 }), 96, 96, 40, 40, 56, 56)
  assert.ok(mean.g > mean.r + 60, `point map channel=1 and alphaMap channel=2 should render green from uv1 while uv2 keeps it opaque (${mean.g} vs ${mean.r})`)
})

test('PointsMaterial maps use point-sprite UVs when geometry UVs are absent', () => {
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))

  const map = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter
  map.channel = 1

  const alphaMap = rgbaTexture([
    255, 255, 255, 255,
    255, 255, 255, 255,
  ], 2, 1)
  alphaMap.channel = 1

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({
    color: 0xffffff,
    map,
    alphaMap,
    size: 48,
    sizeAttenuation: false,
  })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 96, height: 96 })
  const left = meanRegion(rgba, 96, 96, 28, 40, 42, 56)
  const right = meanRegion(rgba, 96, 96, 54, 40, 68, 56)
  assert.ok(left.r > left.g + 60, `left point-sprite half should sample red (${left.r} vs ${left.g})`)
  assert.ok(right.g > right.r + 60, `right point-sprite half should sample green (${right.g} vs ${right.r})`)
})

test('Points with InstancedBufferGeometry default instanceCount expands offsets and colors', () => {
  const geometry = new THREE.InstancedBufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    0, 0, 0,
  ]), 3))
  geometry.setAttribute('instanceOffset', new THREE.InstancedBufferAttribute(new Float32Array([
    -0.35, 0, 0,
    0.35, 0, 0,
  ]), 3))
  geometry.setAttribute('color', new THREE.InstancedBufferAttribute(new Float32Array([
    1, 0, 0,
    0, 1, 0,
  ]), 3))

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({
    color: 0xffffff,
    vertexColors: true,
    size: 24,
    sizeAttenuation: false,
  })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 96, height: 96 })
  const redPixels = countRegionPixels(rgba, 96, 96, 20, 34, 44, 62, (r, g, b) => r > g + 40 && r > b + 40)
  const greenPixels = countRegionPixels(rgba, 96, 96, 52, 34, 76, 62, (r, g, b) => g > r + 40 && g > b + 40)
  assert.ok(redPixels > 20, `left instanced point should render red pixels (${redPixels})`)
  assert.ok(greenPixels > 20, `right instanced point should render green pixels (${greenPixels})`)
})

test('Points with InstancedBufferGeometry honor meshPerAttribute repeat values for colors', () => {
  const geometry = new THREE.InstancedBufferGeometry()
  geometry.instanceCount = 4
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    0, 0, 0,
  ]), 3))
  geometry.setAttribute('instanceOffset', new THREE.InstancedBufferAttribute(new Float32Array([
    -0.6, 0, 0,
    -0.2, 0, 0,
    0.2, 0, 0,
    0.6, 0, 0,
  ]), 3))

  const colors = new THREE.InstancedBufferAttribute(new Float32Array([
    1, 0, 0,
    0, 1, 0,
  ]), 3)
  colors.meshPerAttribute = 2
  geometry.setAttribute('color', colors)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({
    color: 0xffffff,
    vertexColors: true,
    size: 12,
    sizeAttenuation: false,
  })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 96, height: 96 })
  const firstRed = countRegionPixels(rgba, 96, 96, 18, 42, 32, 54, (r, g, b) => r > g + 40 && r > b + 40)
  const secondRed = countRegionPixels(rgba, 96, 96, 34, 42, 47, 54, (r, g, b) => r > g + 40 && r > b + 40)
  const firstGreen = countRegionPixels(rgba, 96, 96, 50, 42, 63, 54, (r, g, b) => g > r + 40 && g > b + 40)
  const secondGreen = countRegionPixels(rgba, 96, 96, 65, 42, 79, 54, (r, g, b) => g > r + 40 && g > b + 40)
  assert.ok(firstRed > 20, `first repeated point color should render red pixels (${firstRed})`)
  assert.ok(secondRed > 20, `second repeated point color should render red pixels (${secondRed})`)
  assert.ok(firstGreen > 20, `first repeated point color should render green pixels (${firstGreen})`)
  assert.ok(secondGreen > 20, `second repeated point color should render green pixels (${secondGreen})`)
})

test('Points with InstancedBufferGeometry default instanceCount expands instanced map UV attributes', () => {
  const geometry = new THREE.InstancedBufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    0, 0, 0,
  ]), 3))
  geometry.setAttribute('instanceOffset', new THREE.InstancedBufferAttribute(new Float32Array([
    -0.35, 0, 0,
    0.35, 0, 0,
  ]), 3))
  geometry.setAttribute('uv', new THREE.InstancedBufferAttribute(new Float32Array([
    0.25, 0.5,
    0.75, 0.5,
  ]), 2))

  const map = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({
    color: 0xffffff,
    map,
    size: 24,
    sizeAttenuation: false,
  })))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const rgba = renderRgba(scene, camera, { width: 96, height: 96 })
  const left = meanRegion(rgba, 96, 96, 24, 38, 42, 58)
  const right = meanRegion(rgba, 96, 96, 54, 38, 72, 58)
  assert.ok(left.r > left.g + 50, `left instanced point uv should sample red (${left.r} vs ${left.g})`)
  assert.ok(right.g > right.r + 50, `right instanced point uv should sample green (${right.g} vs ${right.r})`)
})

test('Points receiveShadow is accepted as an unlit WebGL-compatible no-op', () => {
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))

  const receiveScene = new THREE.Scene()
  receiveScene.background = new THREE.Color(0, 0, 0)
  const receiver = new THREE.Points(geometry, new THREE.PointsMaterial({ color: 0xffffff, size: 12 }))
  receiver.receiveShadow = true
  receiveScene.add(receiver)

  const mean = meanRegion(renderRgba(receiveScene, makeCamera(), { width: 64, height: 64 }), 64, 64, 28, 28, 36, 36)
  assert.ok(mean.r > 180 && mean.g > 180 && mean.b > 180, `points receiveShadow no-op should still render the unlit billboard (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('Points cast point-light shadows from expanded billboard quads', () => {
  function renderPointShadow(castShadow) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(1, 1, 1)

    const receiver = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ opacity: 1 }),
    )
    receiver.rotation.x = -Math.PI / 2
    receiver.receiveShadow = true
    scene.add(receiver)

    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 2.2, 1.8]), 3))
    const caster = new THREE.Points(geometry, new THREE.PointsMaterial({
      color: 0xffffff,
      size: 48,
      sizeAttenuation: false,
    }))
    caster.castShadow = castShadow
    scene.add(caster)

    const light = new THREE.PointLight(0xffffff, 2)
    light.position.set(0, 5, 4)
    light.distance = 12
    light.castShadow = true
    light.shadow.mapSize.set(256, 256)
    light.shadow.camera.near = 0.1
    light.shadow.camera.far = 12
    scene.add(light)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 6, 8)
    camera.lookAt(0, 0, 0)
    return meanRegion(renderRgba(scene, camera, { width: 96, height: 96 }), 96, 96, 28, 42, 68, 82)
  }

  const unshadowed = renderPointShadow(false)
  const shadowed = renderPointShadow(true)
  const unshadowedLum = unshadowed.r + unshadowed.g + unshadowed.b
  const shadowedLum = shadowed.r + shadowed.g + shadowed.b
  assert.ok(shadowedLum < unshadowedLum - 10, `point billboard point-light shadow should darken the receiver (${shadowedLum} vs ${unshadowedLum})`)
})

test('empty scene renders the background color', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(1, 0, 0)
  const camera = makeCamera()

  const rgba = getRenderer().render(scene, camera, { width: 64, height: 64, format: 'rgba' })
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

test('invalid background control values fail clearly', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(1, 0, 0)
  const camera = makeCamera()

  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32, backgroundIntensity: Number.POSITIVE_INFINITY }),
    /options\.backgroundIntensity must be a finite number/i,
  )
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32, backgroundIntensity: -0.1 }),
    /options\.backgroundIntensity must be non-negative/i,
  )

  scene.backgroundIntensity = 'bright'
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /scene\.backgroundIntensity must be a finite number/i,
  )
  scene.backgroundIntensity = -0.1
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /scene\.backgroundIntensity must be non-negative/i,
  )
  scene.backgroundIntensity = 1

  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32, backgroundBlurriness: 'strong' }),
    /options\.backgroundBlurriness must be a finite number/i,
  )
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32, backgroundBlurriness: 1.5 }),
    /options\.backgroundBlurriness must be between 0 and 1/i,
  )

  scene.background = solidTexture(0, 255, 0)
  scene.backgroundBlurriness = 'soft'
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /scene\.backgroundBlurriness must be a finite number/i,
  )
  scene.backgroundBlurriness = -0.1
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /scene\.backgroundBlurriness must be between 0 and 1/i,
  )
})

test('invalid background color values fail clearly', () => {
  const camera = makeCamera()

  assert.throws(
    () => renderRgba(new THREE.Scene(), camera, { width: 32, height: 32, background: [1, 'green', 0] }),
    /options\.background\[1\] must be a finite number/i,
  )

  assert.throws(
    () => renderRgba(new THREE.Scene(), camera, { width: 32, height: 32, background: [1, 0] }),
    /options\.background must be \[r, g, b\] or \[r, g, b, a\]/i,
  )
  assert.throws(
    () => renderRgba(new THREE.Scene(), camera, { width: 32, height: 32, background: 'red' }),
    /options\.background must be a color, texture, or null/i,
  )
  assert.throws(
    () => renderRgba(new THREE.Scene(), camera, { width: 32, height: 32, background: {} }),
    /options\.background must be a color, texture, or null/i,
  )

  const scene = new THREE.Scene()
  scene.background = { isColor: true, r: 0, g: Number.NaN, b: 1 }
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /scene\.background\.g must be a finite number/i,
  )
  scene.background = 'red'
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /scene\.background must be a color, texture, or null/i,
  )
  scene.background = {}
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32 }),
    /scene\.background must be a color, texture, or null/i,
  )

  scene.background = solidTexture(0, 255, 0)
  assert.throws(
    () => renderRgba(scene, camera, { width: 32, height: 32, background: { r: 1, g: 0 } }),
    /options\.background\.b must be a finite number/i,
  )
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

test('options.backgroundBlurriness overrides scene backgroundBlurriness', () => {
  const texture = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  texture.magFilter = THREE.NearestFilter
  texture.minFilter = THREE.NearestFilter

  const scene = new THREE.Scene()
  scene.background = texture
  scene.backgroundBlurriness = 0
  const sharp = meanRegion(renderRgba(scene, makeCamera(), { width: 64, height: 64 }), 64, 64, 28, 20, 31, 44)
  const blurred = meanRegion(renderRgba(scene, makeCamera(), {
    width: 64,
    height: 64,
    backgroundBlurriness: 1,
  }), 64, 64, 28, 20, 31, 44)

  assert.ok(sharp.r > sharp.g + 120, `scene blurriness 0 should keep the red texel sharp (${sharp.r} vs ${sharp.g})`)
  assert.ok(blurred.g > sharp.g + 80, `options.backgroundBlurriness should soften in the green texel (${blurred.g} vs ${sharp.g})`)
})

test('backgroundBlurriness softens equirectangular and cube texture backgrounds', () => {
  function renderBackground(background, blurriness) {
    const scene = new THREE.Scene()
    scene.background = background
    scene.backgroundBlurriness = blurriness

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 0)
    camera.lookAt(new THREE.Vector3(0, 0, -1))
    return meanRegion(renderRgba(scene, camera, {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 64, 64, 28, 28, 36, 36)
  }

  const equirect = rgbaTexture([
    255, 0, 0, 255,
    255, 0, 0, 255,
    255, 0, 0, 255,
    255, 0, 0, 255,
    0, 255, 0, 255,
    0, 255, 0, 255,
    0, 255, 0, 255,
    0, 255, 0, 255,
  ], 8, 1)
  equirect.mapping = THREE.EquirectangularReflectionMapping
  equirect.magFilter = THREE.NearestFilter
  equirect.minFilter = THREE.NearestFilter

  const cube = cubeTexture([
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0],
    [255, 0, 0],
  ])
  cube.magFilter = THREE.NearestFilter
  cube.minFilter = THREE.NearestFilter

  for (const [name, background] of [['equirect', equirect], ['cube', cube]]) {
    const sharp = renderBackground(background, 0)
    const blurred = renderBackground(background, 1)
    assert.ok(sharp.r > sharp.g + 80, `${name} sharp background should sample red (${sharp.r} vs ${sharp.g})`)
    assert.ok(blurred.g > sharp.g + 30, `${name} blurred background should mix in green (${blurred.g} vs ${sharp.g})`)
    assert.ok(sharp.r > blurred.r + 20, `${name} blurred background should soften red (${sharp.r} vs ${blurred.r})`)
  }
})

test('unsupported scene background rotations fail clearly', () => {
  const cases = [
    ['color backgroundRotation', (scene) => {
      scene.background = new THREE.Color(0, 0, 0)
      scene.backgroundRotation = new THREE.Euler(0, Math.PI / 4, 0)
    }, /scene\.backgroundRotation.*equirectangular or cube texture backgrounds/i],
    ['2D backgroundRotation', (scene) => {
      scene.background = solidTexture(0, 255, 0)
      scene.backgroundRotation = new THREE.Euler(0, Math.PI / 4, 0)
    }, /scene\.backgroundRotation.*equirectangular or cube texture backgrounds/i],
  ]

  for (const [name, setup, pattern] of cases) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)
    setup(scene)

    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
      pattern,
      name,
    )
  }
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

test('background textures honor explicit texture matrices', () => {
  const background = rgbaTexture([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ], 2, 1)
  background.magFilter = THREE.NearestFilter
  background.minFilter = THREE.NearestFilter
  background.matrixAutoUpdate = false
  background.matrix.set(
    0, 0, 0.25,
    0, 0, 0.5,
    0, 0, 1,
  )

  const scene = new THREE.Scene()
  scene.background = background
  const mean = meanRgba(renderRgba(scene, makeCamera(), { width: 64, height: 64 }))
  assert.ok(mean.r > mean.g + 80, `explicit background matrix should pin sampling to the red texel (${mean.r} vs ${mean.g})`)
})

test('background textures honor horizontal wrap modes', () => {
  function renderWrap(wrapS) {
    const background = rgbaTexture([
      255, 0, 0, 255,
      0, 255, 0, 255,
    ], 2, 1)
    background.magFilter = THREE.NearestFilter
    background.minFilter = THREE.NearestFilter
    background.offset.set(1, 0)
    if (wrapS != null) background.wrapS = wrapS

    const scene = new THREE.Scene()
    scene.background = background
    return meanRegion(renderRgba(scene, makeCamera(), { width: 64, height: 64 }), 64, 64, 8, 20, 24, 44)
  }

  const clamped = renderWrap(undefined)
  const repeated = renderWrap(THREE.RepeatWrapping)
  const mirrored = renderWrap(THREE.MirroredRepeatWrapping)
  assert.ok(clamped.g > clamped.r + 80, `clamped offset should hold the green edge texel (${clamped.g} vs ${clamped.r})`)
  assert.ok(repeated.r > repeated.g + 80, `repeated offset should wrap back to the red texel (${repeated.r} vs ${repeated.g})`)
  assert.ok(mirrored.g > mirrored.r + 80, `mirrored repeat should reflect the offset into the green texel (${mirrored.g} vs ${mirrored.r})`)
})

test('background textures honor vertical wrap modes', () => {
  function renderWrap(wrapT) {
    const background = rgbaTexture([
      255, 0, 0, 255,
      255, 0, 0, 255,
      0, 255, 0, 255,
      0, 255, 0, 255,
    ], 2, 2)
    background.magFilter = THREE.NearestFilter
    background.minFilter = THREE.NearestFilter
    background.offset.set(0, 0.5)
    if (wrapT != null) background.wrapT = wrapT

    const scene = new THREE.Scene()
    scene.background = background
    return meanRgba(renderRgba(scene, makeCamera(), { width: 64, height: 64 }))
  }

  const clamped = renderWrap(undefined)
  const repeated = renderWrap(THREE.RepeatWrapping)
  const mirrored = renderWrap(THREE.MirroredRepeatWrapping)
  assert.ok(clamped.g > clamped.r + 80, `clamped vertical offset should hold the green edge texel (${clamped.g} vs ${clamped.r})`)
  assert.ok(repeated.r > clamped.r + 80, `repeated vertical offset should wrap red texels back into view (${repeated.r} vs ${clamped.r})`)
  assert.ok(repeated.g < clamped.g - 80, `repeated vertical offset should no longer be fully clamped green (${repeated.g} vs ${clamped.g})`)
  assert.ok(mirrored.g > mirrored.r + 80, `mirrored repeat should reflect the vertical offset into green texels (${mirrored.g} vs ${mirrored.r})`)
})

test('background texture anisotropy renders with native sampler settings', () => {
  const background = solidTexture(32, 180, 64)
  background.anisotropy = 4

  const scene = new THREE.Scene()
  scene.background = background
  const mean = meanRgba(renderRgba(scene, makeCamera(), { width: 64, height: 64 }))
  assert.ok(mean.g > mean.r + 80 && mean.g > mean.b + 80, `anisotropic background texture should render green (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('background textures decode sRGB colorSpace before output conversion', () => {
  function renderColorSpace(colorSpace) {
    const background = solidTexture(128, 128, 128)
    background.colorSpace = colorSpace

    const scene = new THREE.Scene()
    scene.background = background
    return meanRgba(renderRgba(scene, makeCamera(), {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }))
  }

  const srgb = renderColorSpace(THREE.SRGBColorSpace)
  const linear = renderColorSpace(THREE.LinearSRGBColorSpace)
  assert.ok(linear.r > srgb.r + 50, `linear background texture should remain brighter than decoded sRGB (${linear.r} vs ${srgb.r})`)
})

test('equirect background textures sample from camera direction', () => {
  const background = rgbaTexture([
    255, 0, 0, 255,
    255, 0, 0, 255,
    255, 0, 0, 255,
    255, 0, 0, 255,
    0, 255, 0, 255,
    0, 255, 0, 255,
    0, 255, 0, 255,
    0, 255, 0, 255,
  ], 8, 1)
  background.mapping = THREE.EquirectangularReflectionMapping
  background.magFilter = THREE.NearestFilter
  background.minFilter = THREE.NearestFilter

  function renderFacing(target) {
    const scene = new THREE.Scene()
    scene.background = background
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 0)
    camera.lookAt(target)
    return meanRegion(renderRgba(scene, camera, {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 64, 64, 28, 28, 36, 36)
  }

  const negativeZ = renderFacing(new THREE.Vector3(0, 0, -1))
  const positiveZ = renderFacing(new THREE.Vector3(0, 0, 1))
  assert.ok(negativeZ.r > negativeZ.g + 80, `-Z view should sample the red equirect half (${negativeZ.r} vs ${negativeZ.g})`)
  assert.ok(positiveZ.g > positiveZ.r + 80, `+Z view should sample the green equirect half (${positiveZ.g} vs ${positiveZ.r})`)
})

test('equirect background textures honor scene backgroundRotation', () => {
  const background = rgbaTexture([
    255, 0, 0, 255,
    255, 0, 0, 255,
    255, 0, 0, 255,
    255, 0, 0, 255,
    0, 255, 0, 255,
    0, 255, 0, 255,
    0, 255, 0, 255,
    0, 255, 0, 255,
  ], 8, 1)
  background.mapping = THREE.EquirectangularReflectionMapping
  background.magFilter = THREE.NearestFilter
  background.minFilter = THREE.NearestFilter

  function renderWithRotation(yRotation) {
    const scene = new THREE.Scene()
    scene.background = background
    scene.backgroundRotation = new THREE.Euler(0, yRotation, 0)
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 0)
    camera.lookAt(new THREE.Vector3(0, 0, -1))
    return meanRegion(renderRgba(scene, camera, {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 64, 64, 28, 28, 36, 36)
  }

  const unrotated = renderWithRotation(0)
  const rotated = renderWithRotation(Math.PI)
  assert.ok(unrotated.r > unrotated.g + 80, `unrotated -Z view should sample red (${unrotated.r} vs ${unrotated.g})`)
  assert.ok(rotated.g > rotated.r + 80, `rotated -Z view should sample green (${rotated.g} vs ${rotated.r})`)
})

test('options.backgroundRotation overrides scene backgroundRotation', () => {
  const background = rgbaTexture([
    255, 0, 0, 255,
    255, 0, 0, 255,
    255, 0, 0, 255,
    255, 0, 0, 255,
    0, 255, 0, 255,
    0, 255, 0, 255,
    0, 255, 0, 255,
    0, 255, 0, 255,
  ], 8, 1)
  background.mapping = THREE.EquirectangularReflectionMapping
  background.magFilter = THREE.NearestFilter
  background.minFilter = THREE.NearestFilter

  const scene = new THREE.Scene()
  scene.background = background
  scene.backgroundRotation = new THREE.Euler(0, 0, 0)
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 0)
  camera.lookAt(new THREE.Vector3(0, 0, -1))

  const sceneRotation = meanRegion(renderRgba(scene, camera, {
    width: 64,
    height: 64,
    outputColorSpace: THREE.LinearSRGBColorSpace,
  }), 64, 64, 28, 28, 36, 36)
  const optionRotation = meanRegion(renderRgba(scene, camera, {
    width: 64,
    height: 64,
    outputColorSpace: THREE.LinearSRGBColorSpace,
    backgroundRotation: new THREE.Euler(0, Math.PI, 0),
  }), 64, 64, 28, 28, 36, 36)

  assert.ok(sceneRotation.r > sceneRotation.g + 80, `scene backgroundRotation should keep red -Z view (${sceneRotation.r} vs ${sceneRotation.g})`)
  assert.ok(optionRotation.g > optionRotation.r + 80, `options.backgroundRotation should override to green -Z view (${optionRotation.g} vs ${optionRotation.r})`)
})

test('cube DataTexture backgrounds sample from camera direction', () => {
  const background = cubeTexture([
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
    [0, 255, 0],
    [255, 0, 0],
  ])
  background.magFilter = THREE.NearestFilter
  background.minFilter = THREE.NearestFilter

  function renderFacing(target, yRotation = 0) {
    const scene = new THREE.Scene()
    scene.background = background
    scene.backgroundRotation = new THREE.Euler(0, yRotation, 0)
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 0)
    camera.lookAt(target)
    return meanRegion(renderRgba(scene, camera, {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 64, 64, 28, 28, 36, 36)
  }

  const negativeZ = renderFacing(new THREE.Vector3(0, 0, -1))
  const positiveZ = renderFacing(new THREE.Vector3(0, 0, 1))
  assert.ok(negativeZ.r > negativeZ.g + 80, `-Z cube face should render red (${negativeZ.r} vs ${negativeZ.g})`)
  assert.ok(positiveZ.g > positiveZ.r + 80, `+Z cube face should render green (${positiveZ.g} vs ${positiveZ.r})`)

  const rotatedNegativeZ = renderFacing(new THREE.Vector3(0, 0, -1), Math.PI)
  assert.ok(rotatedNegativeZ.g > rotatedNegativeZ.r + 80, `rotated -Z cube background should render +Z green (${rotatedNegativeZ.g} vs ${rotatedNegativeZ.r})`)
})

test('cube background textures decode sRGB colorSpace before output conversion', () => {
  function renderColorSpace(colorSpace) {
    const scene = new THREE.Scene()
    scene.background = cubeTexture([
      [128, 128, 128],
      [128, 128, 128],
      [128, 128, 128],
      [128, 128, 128],
      [128, 128, 128],
      [128, 128, 128],
    ])
    scene.background.colorSpace = colorSpace

    return meanRgba(renderRgba(scene, makeCamera(), {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }))
  }

  const srgb = renderColorSpace(THREE.SRGBColorSpace)
  const linear = renderColorSpace(THREE.LinearSRGBColorSpace)
  assert.ok(linear.r > srgb.r + 20, `linear cube background should render brighter than decoded sRGB cube texture (${linear.r} vs ${srgb.r})`)
})

test('encoded cube background textures decode face images', () => {
  const background = encodedCubeTexture()
  background.magFilter = THREE.NearestFilter
  background.minFilter = THREE.NearestFilter

  function renderFacing(target, yRotation = 0) {
    const scene = new THREE.Scene()
    scene.background = background
    scene.backgroundRotation = new THREE.Euler(0, yRotation, 0)
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(0, 0, 0)
    camera.lookAt(target)
    return meanRegion(renderRgbaIsolated(scene, camera, {
      width: 64,
      height: 64,
      outputColorSpace: THREE.LinearSRGBColorSpace,
    }), 64, 64, 28, 28, 36, 36)
  }

  const negativeZ = renderFacing(new THREE.Vector3(0, 0, -1))
  const positiveZ = renderFacing(new THREE.Vector3(0, 0, 1))
  assert.ok(negativeZ.r > negativeZ.g + 80, `encoded -Z cube face should render red (${negativeZ.r} vs ${negativeZ.g})`)
  assert.ok(positiveZ.g > positiveZ.r + 80, `encoded +Z cube face should render green (${positiveZ.g} vs ${positiveZ.r})`)

  const rotatedNegativeZ = renderFacing(new THREE.Vector3(0, 0, -1), Math.PI)
  assert.ok(rotatedNegativeZ.g > rotatedNegativeZ.r + 80, `rotated encoded cube background should render +Z green (${rotatedNegativeZ.g} vs ${rotatedNegativeZ.r})`)
})

test('render options accept cube DataTexture backgrounds', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(1, 0, 0)
  const camera = makeCamera()
  const background = cubeTexture([
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
    [0, 255, 0],
    [0, 0, 255],
  ])

  const mean = meanRegion(renderRgba(scene, camera, {
    width: 64,
    height: 64,
    background,
    outputColorSpace: THREE.LinearSRGBColorSpace,
  }), 64, 64, 28, 28, 36, 36)
  assert.ok(mean.b > mean.r + 80, `options.background cube texture should override scene background (${mean.b} vs ${mean.r})`)
})

test('cube background textures require six face images', () => {
  const scene = new THREE.Scene()
  scene.background = Object.assign(solidTexture(0, 255, 0), { mapping: THREE.CubeReflectionMapping })

  assert.throws(
    () => renderRgba(scene, makeCamera(), { width: 64, height: 64 }),
    /six raw or encoded face images/i,
  )
})

test('CubeUV background texture mappings fail clearly', () => {
  const cases = [
    ['scene.background', () => {
      const scene = new THREE.Scene()
      scene.background = Object.assign(solidTexture(0, 255, 0), { mapping: THREE.CubeUVReflectionMapping })
      return { scene, options: {} }
    }],
    ['options.background', () => {
      const scene = new THREE.Scene()
      scene.background = new THREE.Color(0, 0, 0)
      return {
        scene,
        options: {
          background: Object.assign(solidTexture(0, 255, 0), { mapping: THREE.CubeUVReflectionMapping }),
        },
      }
    }],
  ]

  for (const [name, makeCase] of cases) {
    const { scene, options } = makeCase()
    assert.throws(
      () => renderRgba(scene, makeCamera(), { width: 64, height: 64, ...options }),
      /PMREM\/CubeUV texture mapping.*not supported/i,
      name,
    )
  }
})

test('render options accept texture backgrounds', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(1, 0, 0)
  scene.backgroundIntensity = 0
  scene.backgroundRotation = new THREE.Euler(0, Math.PI / 4, 0)
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
  scene.backgroundIntensity = 0
  scene.backgroundRotation = new THREE.Euler(0, Math.PI / 4, 0)
  const camera = makeCamera()

  const mean = meanRgba(renderRgba(scene, camera, { width: 64, height: 64, background: [1, 0, 0] }))
  assert.ok(mean.r > 200, `options.background color should override scene texture background (${mean.r})`)
  assert.ok(mean.g < 30, `options.background color should suppress scene texture background (${mean.g})`)
})

test('render option null background clears scene backgrounds', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(1, 0, 0)
  const camera = makeCamera()

  const sceneBackground = meanRgba(renderRgba(scene, camera, { width: 64, height: 64 }))
  scene.backgroundIntensity = 'ignored'
  scene.backgroundRotation = new THREE.Euler(0, Math.PI / 4, 0)
  const cleared = meanRgba(renderRgba(scene, camera, { width: 64, height: 64, background: null }))

  assert.ok(sceneBackground.r > 200, `scene color background should render red (${sceneBackground.r})`)
  assert.ok(cleared.r < sceneBackground.r - 120, `options.background null should clear scene color background (${cleared.r} vs ${sceneBackground.r})`)
  assert.ok(cleared.g > 5 && cleared.b > 5, `cleared background should use renderer default color (${cleared.g}, ${cleared.b})`)

  const textureScene = new THREE.Scene()
  textureScene.background = solidTexture(0, 255, 0)
  const textureBackground = meanRgba(renderRgba(textureScene, camera, { width: 64, height: 64 }))
  textureScene.backgroundIntensity = 'ignored'
  textureScene.backgroundBlurriness = 'ignored'
  const textureCleared = meanRgba(renderRgba(textureScene, camera, { width: 64, height: 64, background: null }))

  assert.ok(textureBackground.g > 180, `scene texture background should render green (${textureBackground.g})`)
  assert.ok(
    textureCleared.g < textureBackground.g - 120,
    `options.background null should clear scene texture background (${textureCleared.g} vs ${textureBackground.g})`,
  )
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

test('render options accept Vector4 viewport and scissor rectangles', () => {
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
    viewport: new THREE.Vector4(16, 16, 40, 32),
    scissor: new THREE.Vector4(24, 20, 24, 24),
  })
  const inside = meanRegion(rgba, 64, 64, 30, 26, 42, 38)
  const viewportOutside = meanRegion(rgba, 64, 64, 4, 26, 12, 38)
  const scissorOutside = meanRegion(rgba, 64, 64, 18, 26, 22, 38)
  assert.ok(inside.r > inside.b + 80, `Vector4 viewport/scissor region should contain the red mesh (${inside.r} vs ${inside.b})`)
  assert.ok(viewportOutside.b > viewportOutside.r + 80, `outside Vector4 viewport should retain blue background (${viewportOutside.b} vs ${viewportOutside.r})`)
  assert.ok(scissorOutside.b > scissorOutside.r + 80, `outside Vector4 scissor should retain blue background (${scissorOutside.b} vs ${scissorOutside.r})`)
})

test('invalid viewport and scissor rectangles fail clearly', () => {
  const scene = new THREE.Scene()
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial({ color: 0xffffff })))
  const camera = makeCamera()

  assert.throws(
    () => getRenderer().render(scene, camera, { width: 32, height: 32, viewport: [0, 0, 0, 16] }),
    /options\.viewport width and height must be greater than 0/i,
  )
  assert.throws(
    () => getRenderer().render(scene, camera, { width: 32, height: 32, viewport: { x: '0', y: 0, width: 16, height: 16 } }),
    /options\.viewport must contain finite x, y, width, and height values/i,
  )
  assert.throws(
    () => getRenderer().render(scene, camera, { width: 32, height: 32, scissor: { x: 0, y: 0, width: 64, height: 16 } }),
    /options\.scissor must fit inside the render target/i,
  )
  assert.throws(
    () => renderToTarget(scene, camera, { viewport: { x: 0, y: 0, width: Number.NaN, height: 16 } }, { width: 32, height: 32 }),
    /target\.viewport must contain finite x, y, width, and height values/i,
  )
  assert.throws(
    () => renderToTarget(scene, camera, { scissorTest: true, scissor: [0, 0, 16, 0] }, { width: 32, height: 32 }),
    /target\.scissor width and height must be greater than 0/i,
  )
  assert.throws(
    () => renderToTarget(scene, camera, { scissorTest: 'yes', scissor: [0, 0, 16, 16] }, { width: 32, height: 32 }),
    /target\.scissorTest must be a boolean/i,
  )

  const cubeTarget = new THREE.WebGLCubeRenderTarget(32)
  const cubeCamera = new THREE.CubeCamera(0.01, 100, cubeTarget)
  cubeCamera.activeMipmapLevel = 1
  cubeTarget.viewport = { x: 0, y: 0, width: '32', height: 32 }
  assert.throws(
    () => renderToTarget(scene, cubeCamera, cubeTarget),
    /target\.viewport must contain finite x, y, width, and height values/i,
  )

  cubeTarget.viewport = undefined
  cubeTarget.scissorTest = true
  cubeTarget.scissor = { x: 0, y: 0, width: 64, height: 32 }
  assert.throws(
    () => renderToTarget(scene, cubeCamera, cubeTarget),
    /target\.scissor must fit inside the render target/i,
  )
})
