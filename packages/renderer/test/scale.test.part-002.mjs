import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { deflateSync } from 'node:zlib'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'
import { BACKGROUND, NESTED_GRAPH_COLUMNS, NESTED_GRAPH_DEPTH, NESTED_GRAPH_ROWS, SIZE, renderer } from './scale.test.part-001.mjs'
test('nested scene graph budget renders 2,048 transform groups with 256 meshes', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.02)

  const geometry = new THREE.PlaneGeometry(0.078, 0.078)
  const materials = [
    new THREE.MeshBasicMaterial({ color: 0xf25f5c }),
    new THREE.MeshBasicMaterial({ color: 0x247ba0 }),
    new THREE.MeshBasicMaterial({ color: 0xffe066 }),
    new THREE.MeshBasicMaterial({ color: 0x70c1b3 }),
  ]
  let groupCount = 0
  let meshCount = 0

  for (let row = 0; row < NESTED_GRAPH_ROWS; row += 1) {
    for (let col = 0; col < NESTED_GRAPH_COLUMNS; col += 1) {
      const root = new THREE.Object3D()
      root.position.set(
        (col - (NESTED_GRAPH_COLUMNS - 1) / 2) * 0.13,
        (row - (NESTED_GRAPH_ROWS - 1) / 2) * 0.13,
        0,
      )
      scene.add(root)

      let parent = root
      groupCount += 1
      for (let depth = 1; depth < NESTED_GRAPH_DEPTH; depth += 1) {
        const group = new THREE.Object3D()
        group.rotation.z = ((row + col + depth) % 5 - 2) * 0.006
        parent.add(group)
        parent = group
        groupCount += 1
      }

      const mesh = new THREE.Mesh(geometry, materials[(row + col) % materials.length])
      mesh.rotation.z = ((row * NESTED_GRAPH_COLUMNS + col) % 9) * 0.035
      parent.add(mesh)
      meshCount += 1
    }
  }

  assert.equal(groupCount, NESTED_GRAPH_ROWS * NESTED_GRAPH_COLUMNS * NESTED_GRAPH_DEPTH)
  assert.equal(meshCount, NESTED_GRAPH_ROWS * NESTED_GRAPH_COLUMNS)

  const camera = new THREE.OrthographicCamera(-1.08, 1.08, 1.08, -1.08, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)

  const rgba = renderer().render(scene, camera, { width: SIZE, height: SIZE, format: 'rgba' })
  assert.equal(rgba.length, SIZE * SIZE * 4)
  const ratio = nonBackgroundRatio(rgba, BACKGROUND, 6)
  assert.ok(ratio > 0.15, `nested scene graph should render broad visible coverage (${ratio})`)
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 25 && mean.g > 25 && mean.b > 20, `nested scene graph colors should survive traversal (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('instanced mesh budget renders 7,056 transformed colored instances', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.02)

  const columns = 84
  const rows = 84
  const count = columns * rows
  const mesh = new THREE.InstancedMesh(
    new THREE.PlaneGeometry(0.022, 0.022),
    new THREE.MeshBasicMaterial({ color: 0xffffff }),
    count,
  )

  const matrix = new THREE.Matrix4()
  const color = new THREE.Color()
  for (let i = 0; i < count; i += 1) {
    const col = i % columns
    const row = Math.floor(i / columns)
    matrix.makeTranslation((col - (columns - 1) / 2) * 0.027, (row - (rows - 1) / 2) * 0.027, 0)
    mesh.setMatrixAt(i, matrix)
    color.setRGB(
      0.25 + 0.75 * (col / (columns - 1)),
      0.25 + 0.75 * (row / (rows - 1)),
      0.25 + 0.75 * ((col + row) / (columns + rows - 2)),
    )
    mesh.setColorAt(i, color)
  }
  scene.add(mesh)

  const camera = new THREE.OrthographicCamera(-1.1, 1.1, 1.1, -1.1, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)

  const rgba = renderer().render(scene, camera, { width: SIZE, height: SIZE, format: 'rgba' })
  assert.equal(rgba.length, SIZE * SIZE * 4)
  const ratio = nonBackgroundRatio(rgba, BACKGROUND, 6)
  assert.ok(ratio > 0.4, `instanced scale scene should fill much of the frame (${ratio})`)
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 40 && mean.g > 40 && mean.b > 40, `instanced colors should survive expansion (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('InstancedBufferGeometry budget renders 4,096 mapped colored mesh instances', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.02)

  const columns = 64
  const rows = 64
  const count = columns * rows
  const base = new THREE.PlaneGeometry(0.026, 0.026)
  const geometry = new THREE.InstancedBufferGeometry()
  geometry.setAttribute('position', base.getAttribute('position'))
  geometry.setIndex(base.index)

  const offsets = new Float32Array(count * 3)
  const scales = new Float32Array(count)
  const colors = new Float32Array(count * 3)
  const normals = new Float32Array(count * 3)
  const uvs = new Float32Array(count * 2)
  for (let index = 0; index < count; index += 1) {
    const col = index % columns
    const row = Math.floor(index / columns)
    offsets[index * 3] = (col / (columns - 1) - 0.5) * 1.9
    offsets[index * 3 + 1] = (row / (rows - 1) - 0.5) * 1.9
    offsets[index * 3 + 2] = Math.sin(col * 0.19 + row * 0.13) * 0.01
    scales[index] = 0.75 + 0.5 * ((col + row) % 5) / 4
    colors[index * 3] = 0.25 + 0.75 * (col / (columns - 1))
    colors[index * 3 + 1] = 0.25 + 0.75 * (row / (rows - 1))
    colors[index * 3 + 2] = 0.35 + 0.65 * ((col + row) / (columns + rows - 2))
    normals[index * 3] = 0
    normals[index * 3 + 1] = 0
    normals[index * 3 + 2] = 1
    uvs[index * 2] = col < columns / 2 ? 0.25 : 0.75
    uvs[index * 2 + 1] = row < rows / 2 ? 0.25 : 0.75
  }

  geometry.setAttribute('instanceOffset', new THREE.InstancedBufferAttribute(offsets, 3))
  geometry.setAttribute('instanceScale', new THREE.InstancedBufferAttribute(scales, 1))
  geometry.setAttribute('color', new THREE.InstancedBufferAttribute(colors, 3))
  geometry.setAttribute('normal', new THREE.InstancedBufferAttribute(normals, 3))
  geometry.setAttribute('uv', new THREE.InstancedBufferAttribute(uvs, 2))

  const textureData = new Uint8Array([
    255, 255, 255, 255,
    96, 180, 255, 255,
    255, 160, 96, 255,
    180, 255, 160, 255,
  ])
  const texture = new THREE.DataTexture(textureData, 2, 2, THREE.RGBAFormat)
  texture.colorSpace = THREE.SRGBColorSpace
  texture.needsUpdate = true

  scene.add(new THREE.Mesh(
    geometry,
    new THREE.MeshBasicMaterial({ color: 0xffffff, vertexColors: true, map: texture }),
  ))

  const camera = new THREE.OrthographicCamera(-1.1, 1.1, 1.1, -1.1, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)

  const rgba = renderer().render(scene, camera, { width: SIZE, height: SIZE, format: 'rgba' })
  assert.equal(rgba.length, SIZE * SIZE * 4)
  const ratio = nonBackgroundRatio(rgba, BACKGROUND, 6)
  assert.ok(ratio > 0.35, `InstancedBufferGeometry scale scene should cover much of the frame (${ratio})`)
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 35 && mean.g > 35 && mean.b > 35, `InstancedBufferGeometry mapped colors should survive expansion (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('BatchedMesh budget renders 2,048 packed colored instances', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.02)

  const columns = 64
  const rows = 32
  const count = columns * rows
  const source = new THREE.PlaneGeometry(0.04, 0.04)
  const material = new THREE.MeshBasicMaterial({ color: 0xffffff })
  const batched = new THREE.BatchedMesh(
    count,
    source.getAttribute('position').count,
    source.index.count,
    material,
  )
  const geometryId = batched.addGeometry(source)
  const matrix = new THREE.Matrix4()
  const color = new THREE.Color()
  for (let index = 0; index < count; index += 1) {
    const col = index % columns
    const row = Math.floor(index / columns)
    const instanceId = batched.addInstance(geometryId)
    matrix.makeTranslation(
      (col / (columns - 1) - 0.5) * 1.9,
      (row / (rows - 1) - 0.5) * 1.9,
      Math.sin(col * 0.11 + row * 0.29) * 0.01,
    )
    batched.setMatrixAt(instanceId, matrix)
    color.setRGB(
      0.2 + 0.8 * (col / (columns - 1)),
      0.2 + 0.8 * (row / (rows - 1)),
      0.35 + 0.65 * ((col + row) / (columns + rows - 2)),
    )
    batched.setColorAt(instanceId, color)
  }
  batched.frustumCulled = false
  batched.perObjectFrustumCulled = false
  batched.sortObjects = false
  scene.add(batched)

  const camera = new THREE.OrthographicCamera(-1.08, 1.08, 1.08, -1.08, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)

  const rgba = renderer().render(scene, camera, { width: SIZE, height: SIZE, format: 'rgba' })
  assert.equal(rgba.length, SIZE * SIZE * 4)
  const ratio = nonBackgroundRatio(rgba, BACKGROUND, 6)
  assert.ok(ratio > 0.3, `BatchedMesh scale scene should cover much of the frame (${ratio})`)
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 35 && mean.g > 35 && mean.b > 35, `BatchedMesh instance colors should survive expansion (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('CPU deformation budget renders a 4,096-vertex morphed skinned mesh', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.02)

  const panelColumns = 32
  const rows = 64
  const panelCount = 2
  const vertexCount = panelColumns * rows * panelCount
  const positions = new Float32Array(vertexCount * 3)
  const normals = new Float32Array(vertexCount * 3)
  const colors = new Float32Array(vertexCount * 3)
  const skinIndices = new Uint16Array(vertexCount * 4)
  const skinWeights = new Float32Array(vertexCount * 4)
  const morphPositions = new Float32Array(vertexCount * 3)
  const morphNormals = new Float32Array(vertexCount * 3)
  const indices = []

  let vertex = 0
  for (let panel = 0; panel < panelCount; panel += 1) {
    const boneIndex = panel
    for (let row = 0; row < rows; row += 1) {
      const v = row / (rows - 1)
      for (let column = 0; column < panelColumns; column += 1) {
        const u = column / (panelColumns - 1)
        const baseX = panel === 0 ? -2.85 + u : 1.85 + u
        const baseY = -3.0 + v * 1.2
        const wave = Math.sin((u * 5 + v * 7 + panel) * Math.PI) * 0.06
        const offset = vertex * 3

        positions[offset] = baseX
        positions[offset + 1] = baseY
        positions[offset + 2] = 0
        normals[offset] = 0
        normals[offset + 1] = 0
        normals[offset + 2] = 1
        colors[offset] = panel === 0 ? 1 - v * 0.4 : 0.25 + u * 0.5
        colors[offset + 1] = 0.25 + v * 0.7
        colors[offset + 2] = panel === 0 ? 0.35 + u * 0.55 : 0.95 - v * 0.35
        skinIndices[vertex * 4] = boneIndex
        skinWeights[vertex * 4] = 1
        morphPositions[offset + 1] = 2.25 + wave
        morphPositions[offset + 2] = wave
        morphNormals[offset + 2] = 0.02
        vertex += 1
      }
    }
  }

  for (let panel = 0; panel < panelCount; panel += 1) {
    const panelStart = panel * panelColumns * rows
    for (let row = 0; row < rows - 1; row += 1) {
      for (let column = 0; column < panelColumns - 1; column += 1) {
        const a = panelStart + row * panelColumns + column
        const b = a + 1
        const c = a + panelColumns
        const d = c + 1
        indices.push(a, c, b, b, c, d)
      }
    }
  }

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3))
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  geometry.setAttribute('skinIndex', new THREE.BufferAttribute(skinIndices, 4))
  geometry.setAttribute('skinWeight', new THREE.BufferAttribute(skinWeights, 4))
  geometry.setIndex(indices)
  geometry.morphTargetsRelative = true
  geometry.morphAttributes.position = [new THREE.BufferAttribute(morphPositions, 3)]
  geometry.morphAttributes.normal = [new THREE.BufferAttribute(morphNormals, 3)]

  const mesh = new THREE.SkinnedMesh(
    geometry,
    new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }),
  )
  mesh.frustumCulled = false
  mesh.morphTargetInfluences = [1]

  const leftBone = new THREE.Bone()
  const rightBone = new THREE.Bone()
  mesh.add(leftBone)
  mesh.add(rightBone)
  mesh.bind(new THREE.Skeleton([leftBone, rightBone]))
  leftBone.position.x = 2.35
  rightBone.position.x = -2.35
  scene.add(mesh)

  const camera = new THREE.OrthographicCamera(-1.08, 1.08, 1.08, -1.08, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)

  assert.equal(vertexCount, 4096)
  assert.equal(indices.length, (panelColumns - 1) * (rows - 1) * panelCount * 6)

  const rgba = renderer().render(scene, camera, { width: SIZE, height: SIZE, format: 'rgba' })
  assert.equal(rgba.length, SIZE * SIZE * 4)
  const ratio = nonBackgroundRatio(rgba, BACKGROUND, 6)
  assert.ok(ratio > 0.22, `morphed skinned scale scene should render after CPU deformation (${ratio})`)
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 35 && mean.g > 35 && mean.b > 35, `deformed vertex colors should survive CPU baking (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('points billboard budget renders 4,096 colored points', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.02)

  const columns = 64
  const rows = 64
  const count = columns * rows
  const positions = new Float32Array(count * 3)
  const colors = new Float32Array(count * 3)
  for (let index = 0; index < count; index += 1) {
    const col = index % columns
    const row = Math.floor(index / columns)
    positions[index * 3] = (col / (columns - 1) - 0.5) * 1.9
    positions[index * 3 + 1] = (row / (rows - 1) - 0.5) * 1.9
    positions[index * 3 + 2] = Math.sin(col * 0.23 + row * 0.17) * 0.02
    colors[index * 3] = 0.2 + 0.8 * (col / (columns - 1))
    colors[index * 3 + 1] = 0.2 + 0.8 * (row / (rows - 1))
    colors[index * 3 + 2] = 0.35 + 0.65 * ((col + row) / (columns + rows - 2))
  }

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  scene.add(new THREE.Points(
    geometry,
    new THREE.PointsMaterial({ size: 2.2, sizeAttenuation: false, vertexColors: true }),
  ))

  const camera = new THREE.OrthographicCamera(-1.08, 1.08, 1.08, -1.08, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)

  const rgba = renderer().render(scene, camera, { width: SIZE, height: SIZE, format: 'rgba' })
  assert.equal(rgba.length, SIZE * SIZE * 4)
  const ratio = nonBackgroundRatio(rgba, BACKGROUND, 6)
  assert.ok(ratio > 0.55, `point billboard scale scene should cover much of the frame (${ratio})`)
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 35 && mean.g > 35 && mean.b > 35, `point colors should survive billboard expansion (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('point object budget renders 2,048 separate transformed Points objects', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.02)

  const columns = 64
  const rows = 32
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3))
  const materials = [
    new THREE.PointsMaterial({ color: 0xf25f5c, size: 2.4, sizeAttenuation: false }),
    new THREE.PointsMaterial({ color: 0x247ba0, size: 2.4, sizeAttenuation: false }),
    new THREE.PointsMaterial({ color: 0x70c1b3, size: 2.4, sizeAttenuation: false }),
    new THREE.PointsMaterial({ color: 0xffe066, size: 2.4, sizeAttenuation: false }),
    new THREE.PointsMaterial({ color: 0xc77dff, size: 2.4, sizeAttenuation: false }),
  ]

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < columns; col += 1) {
      const points = new THREE.Points(geometry, materials[(row + col) % materials.length])
      points.position.set(
        (col / (columns - 1) - 0.5) * 1.9,
        (row / (rows - 1) - 0.5) * 1.9,
        Math.sin(col * 0.21 + row * 0.13) * 0.02,
      )
      scene.add(points)
    }
  }

  assert.equal(scene.children.length, columns * rows)

  const camera = new THREE.OrthographicCamera(-1.08, 1.08, 1.08, -1.08, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)

  const rgba = renderer().render(scene, camera, { width: SIZE, height: SIZE, format: 'rgba' })
  assert.equal(rgba.length, SIZE * SIZE * 4)
  const ratio = nonBackgroundRatio(rgba, BACKGROUND, 6)
  assert.ok(ratio > 0.4, `separate Points object scale scene should cover much of the frame (${ratio})`)
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 30 && mean.g > 30 && mean.b > 30, `separate Points object colors should survive traversal (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('sprite billboard budget renders 2,048 colored sprites', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.02)

  const columns = 64
  const rows = 32
  const materialCount = 64
  const materials = Array.from({ length: materialCount }, (_, index) => {
    const t = index / (materialCount - 1)
    return new THREE.SpriteMaterial({
      color: new THREE.Color(0.25 + 0.75 * t, 0.25 + 0.65 * (1 - t), 0.45 + 0.45 * Math.sin(t * Math.PI)),
      transparent: false,
    })
  })

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < columns; col += 1) {
      const sprite = new THREE.Sprite(materials[(row * columns + col) % materialCount])
      sprite.position.set(
        (col / (columns - 1) - 0.5) * 1.9,
        (row / (rows - 1) - 0.5) * 1.9,
        Math.sin(col * 0.17 + row * 0.31) * 0.01,
      )
      sprite.scale.setScalar(0.045)
      scene.add(sprite)
    }
  }

  const camera = new THREE.OrthographicCamera(-1.08, 1.08, 1.08, -1.08, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)

  const rgba = renderer().render(scene, camera, { width: SIZE, height: SIZE, format: 'rgba' })
  assert.equal(rgba.length, SIZE * SIZE * 4)
  const ratio = nonBackgroundRatio(rgba, BACKGROUND, 6)
  assert.ok(ratio > 0.35, `sprite billboard scale scene should cover much of the frame (${ratio})`)
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 35 && mean.g > 35 && mean.b > 35, `sprite colors should survive billboard expansion (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('wide line budget renders 4,032 colored segments', () => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.02, 0.02, 0.02)

  const columns = 64
  const rows = 64
  const segments = rows * (columns - 1)
  const positions = new Float32Array(segments * 2 * 3)
  const colors = new Float32Array(segments * 2 * 3)
  for (let index = 0; index < segments; index += 1) {
    const row = Math.floor(index / (columns - 1))
    const col = index % (columns - 1)
    const x0 = (col / (columns - 1) - 0.5) * 1.9
    const x1 = ((col + 1) / (columns - 1) - 0.5) * 1.9
    const y = (row / (rows - 1) - 0.5) * 1.9
    const z = Math.sin(col * 0.19 + row * 0.13) * 0.01
    const offset = index * 6
    positions[offset] = x0
    positions[offset + 1] = y
    positions[offset + 2] = z
    positions[offset + 3] = x1
    positions[offset + 4] = y
    positions[offset + 5] = z

    const r0 = 0.2 + 0.8 * (col / (columns - 1))
    const r1 = 0.2 + 0.8 * ((col + 1) / (columns - 1))
    const g = 0.2 + 0.8 * (row / (rows - 1))
    const b0 = 0.35 + 0.65 * ((col + row) / (columns + rows - 2))
    const b1 = 0.35 + 0.65 * ((col + 1 + row) / (columns + rows - 2))
    colors[offset] = r0
    colors[offset + 1] = g
    colors[offset + 2] = b0
    colors[offset + 3] = r1
    colors[offset + 4] = g
    colors[offset + 5] = b1
  }

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  scene.add(new THREE.LineSegments(
    geometry,
    new THREE.LineBasicMaterial({ linewidth: 2.2, vertexColors: true }),
  ))

  const camera = new THREE.OrthographicCamera(-1.08, 1.08, 1.08, -1.08, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)

  const rgba = renderer().render(scene, camera, { width: SIZE, height: SIZE, format: 'rgba' })
  assert.equal(rgba.length, SIZE * SIZE * 4)
  const ratio = nonBackgroundRatio(rgba, BACKGROUND, 6)
  assert.ok(ratio > 0.5, `wide line scale scene should cover much of the frame (${ratio})`)
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 35 && mean.g > 35 && mean.b > 35, `line colors should survive wide-line expansion (${mean.r}, ${mean.g}, ${mean.b})`)
})
