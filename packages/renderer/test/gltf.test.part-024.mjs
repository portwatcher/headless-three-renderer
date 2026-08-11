import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'
import { Renderer, SAMPLE_ASSET_MORPH_PRIMITIVES_TEST, SAMPLE_ASSET_MULTIPLE_SCENES, SAMPLE_ASSET_NEGATIVE_SCALE_TEST, SAMPLE_ASSET_ORIENTATION_TEST, SAMPLE_ASSET_PRIMITIVE_MODE_NORMALS_TEST, SAMPLE_ASSET_SIMPLE_MATERIAL, SAMPLE_ASSET_SIMPLE_MESHES, SAMPLE_ASSET_TRIANGLE, SAMPLE_ASSET_UNLIT_TEST } from './gltf.test.part-001.mjs'
import { assertVectorClose, findFirst, loadGltfFixture, meanRegion, pngDimensions, renderSingleObjectRatio, worldDeterminant } from './gltf.test.part-028.mjs'
test('committed Khronos glTF Sample Assets PrimitiveModeNormalsTest fixture loads primitive modes with normals and colors', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_PRIMITIVE_MODE_NORMALS_TEST, 'utf8'))
  assert.deepEqual(source.buffers.map((buffer) => buffer.uri), [
    'Points.bin',
    'Lines.bin',
    'Triangles.bin',
    'Colors.bin',
    'Plane.bin',
  ])
  assert.deepEqual(source.buffers.map((buffer) => buffer.byteLength), [786432, 786432, 4380, 262144, 92])
  assert.deepEqual(source.images.map((image) => image.uri), ['Labels.png'])
  assert.equal(source.meshes.length, 25)
  assert.deepEqual(source.meshes.slice(0, 6).map((mesh) => ({
    mode: mesh.primitives[0].mode,
    attributes: Object.keys(mesh.primitives[0].attributes),
  })), [
    { mode: 0, attributes: ['POSITION'] },
    { mode: 3, attributes: ['POSITION'] },
    { mode: 4, attributes: ['POSITION'] },
    { mode: 0, attributes: ['POSITION', 'COLOR_0'] },
    { mode: 3, attributes: ['POSITION', 'COLOR_0'] },
    { mode: 4, attributes: ['POSITION', 'COLOR_0'] },
  ])
  assert.deepEqual(source.meshes.slice(12, 18).map((mesh) => Object.keys(mesh.primitives[0].attributes)), [
    ['POSITION', 'NORMAL'],
    ['POSITION', 'NORMAL'],
    ['POSITION', 'NORMAL'],
    ['POSITION', 'COLOR_0', 'NORMAL'],
    ['POSITION', 'COLOR_0', 'NORMAL'],
    ['POSITION', 'COLOR_0', 'NORMAL'],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_PRIMITIVE_MODE_NORMALS_TEST)
  const renderables = []
  gltf.scene.traverse((object) => {
    if (
      object.isMesh === true ||
      object.isLine === true ||
      object.isLineSegments === true ||
      object.isLineLoop === true ||
      object.isPoints === true
    ) {
      renderables.push(object)
    }
  })

  assert.equal(renderables.length, 25)
  assert.deepEqual(renderables.reduce((counts, object) => {
    counts[object.type] = (counts[object.type] ?? 0) + 1
    return counts
  }, {}), { Points: 8, Line: 8, Mesh: 9 })

  const points = renderables[0]
  const coloredPoints = renderables[3]
  const normalMesh = renderables[14]
  const coloredNormalMesh = renderables[17]
  const labelPlane = renderables[24]
  assert.equal(points.geometry.getAttribute('position')?.count, 65536)
  assert.equal(coloredPoints.geometry.getAttribute('color')?.count, 65536)
  assert.equal(coloredPoints.geometry.getAttribute('color')?.itemSize, 4)
  assert.equal(coloredPoints.geometry.getAttribute('color')?.normalized, true)
  assert.equal(normalMesh.geometry.getAttribute('normal')?.count, 205)
  assert.equal(coloredNormalMesh.geometry.getAttribute('normal')?.count, 205)
  assert.equal(coloredNormalMesh.geometry.getAttribute('color')?.normalized, true)
  assert.equal(labelPlane.material.map?.name, 'Labels.png')
  assert.deepEqual(pngDimensions(labelPlane.material.map.image), [1024, 1024])
  assert.equal(labelPlane.material.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(labelPlane.material.map.flipY, false)

  for (const object of renderables) {
    if (object.material?.color) object.material.color.set(0xffffff)
    if (object.isPoints === true) object.material.size = 2.5
    if (object.isLine === true || object.isLineSegments === true || object.isLineLoop === true) {
      object.material.linewidth = 3
    }
  }

  const camera = new THREE.OrthographicCamera(-7, 11, 8, -8, 0.01, 30)
  camera.position.set(0, 0, 12)
  camera.lookAt(0, 0, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1.0))
  const light = new THREE.DirectionalLight(0xffffff, 1.2)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 180,
    height: 180,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.45, 'PrimitiveModeNormalsTest should render visible primitive-mode grids')
  const topLeft = meanRegion(rgba, 180, 180, 20, 20, 50, 50)
  const center = meanRegion(rgba, 180, 180, 75, 75, 105, 105)
  assert.ok(topLeft.r > 150 && topLeft.g > 150 && topLeft.b > 150, `upper primitive samples should render bright points/lines (${topLeft.r}, ${topLeft.g}, ${topLeft.b})`)
  assert.ok(center.r > 50 && center.g > 45 && center.b < 20, `center primitive sample should include normalized color attributes (${center.r}, ${center.g}, ${center.b})`)
})

test('committed Khronos glTF Sample Assets MorphPrimitivesTest fixture preserves morph targets across split primitives', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_MORPH_PRIMITIVES_TEST, 'utf8'))
  assert.equal(source.buffers[0].uri, 'MorphPrimitivesTest.bin')
  assert.deepEqual(source.images.map((image) => image.uri), ['uv_texture.jpg'])
  assert.deepEqual(source.meshes[0].weights, [0.5])
  assert.deepEqual(source.meshes[0].primitives.map((primitive) => ({
    mode: primitive.mode,
    material: primitive.material,
    targetAttributes: Object.keys(primitive.targets[0]),
  })), [
    { mode: 4, material: 0, targetAttributes: ['POSITION'] },
    { mode: 4, material: 1, targetAttributes: ['POSITION'] },
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_MORPH_PRIMITIVES_TEST)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })

  assert.deepEqual(meshes.map((mesh) => ({
    name: mesh.name,
    material: mesh.material.name,
    positions: mesh.geometry.getAttribute('position')?.count,
    normals: mesh.geometry.getAttribute('normal')?.count,
    uvs: mesh.geometry.getAttribute('uv')?.count,
    index: mesh.geometry.index?.count,
    morphPositions: mesh.geometry.morphAttributes.position?.map((attribute) => attribute.count),
    influences: mesh.morphTargetInfluences,
    morphTargetsRelative: mesh.geometry.morphTargetsRelative,
  })), [
    {
      name: 'mesh_1',
      material: 'red',
      positions: 21,
      normals: 21,
      uvs: 21,
      index: 72,
      morphPositions: [21],
      influences: [0.5],
      morphTargetsRelative: true,
    },
    {
      name: 'mesh_2',
      material: 'green',
      positions: 9,
      normals: 9,
      uvs: 9,
      index: 24,
      morphPositions: [9],
      influences: [0.5],
      morphTargetsRelative: true,
    },
  ])

  assertVectorClose(meshes[0].material.color.toArray(), [1, 0, 0], 'MorphPrimitivesTest red material')
  assertVectorClose(meshes[1].material.color.toArray(), [0, 1, 0], 'MorphPrimitivesTest green material')
  for (const mesh of meshes) {
    assert.equal(mesh.material.isMeshStandardMaterial, true)
    assert.equal(mesh.material.map?.name, 'uv_texture.jpg')
    assert.equal(Buffer.isBuffer(mesh.material.map.image), true, `${mesh.name} should load the external JPEG as an encoded Buffer`)
    assert.equal(mesh.material.map.colorSpace, THREE.SRGBColorSpace)
    assert.equal(mesh.material.map.flipY, false)

    const position = mesh.geometry.getAttribute('position')
    const morphPosition = mesh.geometry.morphAttributes.position[0]
    let morphedMaxY = -Infinity
    for (let i = 0; i < position.count; i += 1) {
      morphedMaxY = Math.max(morphedMaxY, position.getY(i) + morphPosition.getY(i))
    }
    assert.ok(Math.abs(morphedMaxY - 0.20000000298023224) < 1e-8, `${mesh.name} should preserve its upward morph target`)
  }

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 20)
  camera.position.set(1.8, 1.4, 3.2)
  camera.lookAt(0, 0, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1.2))
  const light = new THREE.DirectionalLight(0xffffff, 1.2)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 128,
    height: 128,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.03, 'MorphPrimitivesTest should render visible morphed primitive meshes')
  const redRegion = meanRegion(rgba, 128, 128, 52, 52, 76, 76)
  const greenRegion = meanRegion(rgba, 128, 128, 76, 52, 100, 76)
  assert.ok(redRegion.r > redRegion.g + 70 && redRegion.r > redRegion.b + 90, `MorphPrimitivesTest should render the red primitive (${redRegion.r}, ${redRegion.g}, ${redRegion.b})`)
  assert.ok(greenRegion.g > greenRegion.r && greenRegion.g > greenRegion.b + 10, `MorphPrimitivesTest should render the green primitive (${greenRegion.r}, ${greenRegion.g}, ${greenRegion.b})`)
})

test('committed Khronos glTF Sample Assets NegativeScaleTest fixture preserves negative node determinants', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_NEGATIVE_SCALE_TEST)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })

  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'NegativeScaleBack',
    'BackgroundMesh',
    'Labels',
    'PositiveScaleTest',
    'NegativeScaleFront',
    'NotShiny1',
    'NotShinyMinus1',
    'Shiny1',
    'ShinyMinus1',
    'Dark1',
    'DarkMinus1',
  ])

  const positivePanel = gltf.scene.getObjectByName('PositiveScaleTest')
  const negativeFrontPanel = gltf.scene.getObjectByName('NegativeScaleFront')
  const labelPanel = gltf.scene.getObjectByName('Labels')
  const notShinyMinusOne = gltf.scene.getObjectByName('NotShinyMinus1')
  const shinyOne = gltf.scene.getObjectByName('Shiny1')
  const shinyMinusOne = gltf.scene.getObjectByName('ShinyMinus1')
  assert.ok(positivePanel?.isMesh, 'NegativeScaleTest should load the positive front-face panel')
  assert.ok(negativeFrontPanel?.isMesh, 'NegativeScaleTest should load the negative-scale front-face panel')
  assert.ok(labelPanel?.isMesh, 'NegativeScaleTest should load the external PNG label panel')
  assert.ok(notShinyMinusOne?.isMesh, 'NegativeScaleTest should load the negative-scale double-sided sphere')
  assert.ok(shinyOne?.isMesh, 'NegativeScaleTest should load a child under a negative-scale parent')
  assert.ok(shinyMinusOne?.isMesh, 'NegativeScaleTest should load a negative-scale child under a negative-scale parent')

  assert.equal(positivePanel.material.side, THREE.FrontSide)
  assert.equal(negativeFrontPanel.material.side, THREE.FrontSide)
  assert.equal(notShinyMinusOne.material.side, THREE.DoubleSide)
  assert.equal(Buffer.isBuffer(negativeFrontPanel.material.map?.image), true, 'NegativeScaleTest check/X PNG should load as an encoded Buffer')
  assert.equal(Buffer.isBuffer(labelPanel.material.map?.image), true, 'NegativeScaleTest label PNG should load as an encoded Buffer')

  assert.ok(worldDeterminant(positivePanel) > 0, 'positive-scale panel should keep positive world winding')
  assert.ok(worldDeterminant(negativeFrontPanel) < 0, 'negative-scale panel should expose negative world winding')
  assert.ok(worldDeterminant(shinyOne) < 0, 'child under a negative-scale parent should inherit negative world winding')
  assert.ok(worldDeterminant(shinyMinusOne) > 0, 'negative-scale child under a negative-scale parent should recover positive world winding')

  const renderer = new Renderer()
  assert.ok(
    renderSingleObjectRatio(renderer, positivePanel) > 0.3,
    'NegativeScaleTest positive-scale front-face panel should render visible pixels',
  )
  assert.ok(
    renderSingleObjectRatio(renderer, negativeFrontPanel) > 0.15,
    'NegativeScaleTest negative-scale front-face panel should render visible pixels',
  )
})

test('committed Khronos glTF Sample Assets OrientationTest fixture preserves quaternion and matrix rotations', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_ORIENTATION_TEST)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 13)

  const arrowX1 = gltf.scene.getObjectByName('ArrowX1')
  const arrowY1 = gltf.scene.getObjectByName('ArrowY1')
  const arrowZ1 = gltf.scene.getObjectByName('ArrowZ1')
  const arrowX2 = gltf.scene.getObjectByName('ArrowX2')
  const arrowY2 = gltf.scene.getObjectByName('ArrowY2')
  const arrowZ2 = gltf.scene.getObjectByName('ArrowZ2')
  for (const arrow of [arrowX1, arrowY1, arrowZ1, arrowX2, arrowY2, arrowZ2]) {
    assert.ok(arrow?.isMesh, 'OrientationTest should load all quaternion and matrix arrow meshes')
  }

  assertVectorClose(arrowX1.position.toArray(), [5, 0, 0], 'ArrowX1 quaternion translation')
  assertVectorClose(arrowY1.position.toArray(), [0, 5, 0], 'ArrowY1 quaternion translation')
  assertVectorClose(arrowZ1.position.toArray(), [0, 0, 5], 'ArrowZ1 quaternion translation')
  assertVectorClose(arrowX2.position.toArray(), [-5, 0, 0], 'ArrowX2 matrix translation')
  assertVectorClose(arrowY2.position.toArray(), [0, -5, 0], 'ArrowY2 matrix translation')
  assertVectorClose(arrowZ2.position.toArray(), [0, 0, -5], 'ArrowZ2 matrix translation')

  assert.ok(arrowX1.quaternion.x < -0.29 && Math.abs(arrowX1.quaternion.y) < 1e-6 && Math.abs(arrowX1.quaternion.z) < 1e-6, 'ArrowX1 should keep its X-axis quaternion rotation')
  assert.ok(arrowY1.quaternion.y < -0.57 && Math.abs(arrowY1.quaternion.x) < 1e-6 && Math.abs(arrowY1.quaternion.z) < 1e-6, 'ArrowY1 should keep its Y-axis quaternion rotation')
  assert.ok(arrowZ1.quaternion.z > 0.13 && Math.abs(arrowZ1.quaternion.x) < 1e-6 && Math.abs(arrowZ1.quaternion.y) < 1e-6, 'ArrowZ1 should keep its Z-axis quaternion rotation')
  assert.ok(arrowX2.quaternion.x > 0.04 && Math.abs(arrowX2.quaternion.y) < 1e-6 && Math.abs(arrowX2.quaternion.z) < 1e-6, 'ArrowX2 should decompose its matrix into an X-axis rotation')
  assert.ok(arrowY2.quaternion.y < -0.10 && Math.abs(arrowY2.quaternion.x) < 1e-6 && Math.abs(arrowY2.quaternion.z) < 1e-6, 'ArrowY2 should decompose its matrix into a Y-axis rotation')
  assert.ok(arrowZ2.quaternion.z < -0.14 && Math.abs(arrowZ2.quaternion.x) < 1e-6 && Math.abs(arrowZ2.quaternion.y) < 1e-6, 'ArrowZ2 should decompose its matrix into a Z-axis rotation')

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.75))
  const light = new THREE.DirectionalLight(0xffffff, 1.0)
  light.position.set(4, 5, 6)
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)
  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const halfExtent = Math.max(size.x, size.y, size.z) / 2 + 0.5
  const camera = new THREE.OrthographicCamera(-halfExtent, halfExtent, halfExtent, -halfExtent, 0.01, 40)
  camera.position.set(center.x + 6, center.y + 4, center.z + 6)
  camera.lookAt(center)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 160,
    height: 160,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.9, 'OrientationTest should render visible rotated arrows and targets')
})

test('committed Khronos glTF Sample Assets MultipleScenes fixture preserves default and alternate scenes', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_MULTIPLE_SCENES)
  assert.equal(gltf.scenes.length, 2)
  assert.equal(gltf.scene, gltf.scenes[1], 'MultipleScenes should select glTF scene index 1 as the default scene')

  const triangleMesh = findFirst(gltf.scenes[0], (object) => object.isMesh === true)
  const squareMesh = findFirst(gltf.scenes[1], (object) => object.isMesh === true)
  assert.ok(triangleMesh, 'MultipleScenes first scene should load a triangle mesh')
  assert.ok(squareMesh, 'MultipleScenes default scene should load a square mesh')
  assert.equal(triangleMesh.geometry.getAttribute('position')?.count, 3)
  assert.equal(triangleMesh.geometry.index?.count, 3)
  assert.equal(squareMesh.geometry.getAttribute('position')?.count, 4)
  assert.equal(squareMesh.geometry.index?.count, 6)

  triangleMesh.material = new THREE.MeshBasicMaterial({ color: 0xffffff })
  squareMesh.material = new THREE.MeshBasicMaterial({ color: 0xffffff })

  const camera = new THREE.OrthographicCamera(-0.6, 0.6, 0.6, -0.6, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  const renderer = new Renderer()
  const renderScene = (scene) => {
    scene.position.set(-0.5, -0.5, 0)
    scene.updateMatrixWorld(true)
    return renderer.render(scene, camera, {
      width: 96,
      height: 96,
      format: 'rgba',
      background: [0, 0, 0],
      outputColorSpace: THREE.LinearSRGBColorSpace,
    })
  }

  const triangleRatio = nonBackgroundRatio(renderScene(gltf.scenes[0]), [0, 0, 0], 3)
  const squareRatio = nonBackgroundRatio(renderScene(gltf.scene), [0, 0, 0], 3)
  assert.ok(triangleRatio > 0.25, `alternate triangle scene should render visible pixels (${triangleRatio})`)
  assert.ok(squareRatio > 0.6, `default square scene should render visible pixels (${squareRatio})`)
  assert.ok(squareRatio > triangleRatio + 0.25, `default square scene should cover more pixels than alternate triangle scene (${squareRatio} vs ${triangleRatio})`)
})

test('committed Khronos glTF Sample Assets SimpleMaterial fixture loads scalar PBR material factors', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_SIMPLE_MATERIAL)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'Khronos SimpleMaterial sample should load a mesh')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 3)
  assert.equal(mesh.geometry.index?.count, 3)
  assert.equal(mesh.material.isMeshStandardMaterial, true)
  assert.deepEqual(mesh.material.color.toArray(), [1, 0.766, 0.336])
  assert.equal(mesh.material.metalness, 0.5)
  assert.equal(mesh.material.roughness, 0.1)

  mesh.position.set(-0.5, -0.5, 0)
  const camera = new THREE.OrthographicCamera(-0.6, 0.6, 0.6, -0.6, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1.0))
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.25, 'SimpleMaterial sample should render visible PBR geometry')
  const center = meanRegion(rgba, 96, 96, 34, 34, 62, 62)
  assert.ok(center.r > center.b + 30 && center.g > center.b + 20, `SimpleMaterial sample should render warm base-color pixels (${center.r}, ${center.g}, ${center.b})`)
})

test('committed Khronos glTF Sample Assets SimpleMeshes fixture reuses a mesh across nodes', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_SIMPLE_MESHES)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), ['mesh_0_instance_0', 'mesh_0_instance_1'])
  assert.equal(meshes[0].geometry, meshes[1].geometry, 'SimpleMeshes nodes should share one loaded geometry')
  assert.deepEqual(meshes[0].position.toArray(), [0, 0, 0])
  assert.deepEqual(meshes[1].position.toArray(), [1, 0, 0])
  assert.equal(meshes[0].geometry.getAttribute('position')?.count, 3)
  assert.equal(meshes[0].geometry.getAttribute('normal')?.count, 3)
  assert.equal(meshes[0].geometry.index?.count, 3)

  for (const mesh of meshes) mesh.material = new THREE.MeshBasicMaterial({ color: 0xffffff })
  gltf.scene.position.set(-0.75, -0.5, 0)
  const camera = new THREE.OrthographicCamera(-0.85, 0.85, 0.65, -0.65, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 128,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.35, 'SimpleMeshes sample should render both shared-geometry mesh instances')
  const left = meanRegion(rgba, 128, 96, 20, 45, 48, 75)
  const right = meanRegion(rgba, 128, 96, 80, 45, 108, 75)
  assert.ok(left.r > 120 && left.g > 120 && left.b > 120, `first shared mesh instance should render visibly (${left.r}, ${left.g}, ${left.b})`)
  assert.ok(right.r > 120 && right.g > 120 && right.b > 120, `second shared mesh instance should render visibly (${right.r}, ${right.g}, ${right.b})`)
})

test('committed Khronos glTF Sample Assets UnlitTest fixture loads KHR_materials_unlit', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_UNLIT_TEST)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), ['Orange_Object', 'Blue_Object'])

  const [orange, blue] = meshes
  assert.equal(orange.material.isMeshBasicMaterial, true)
  assert.equal(blue.material.isMeshBasicMaterial, true)
  assert.equal(orange.geometry.getAttribute('position')?.count, 96)
  assert.equal(orange.geometry.getAttribute('normal')?.count, 96)
  assert.equal(orange.geometry.index?.count, 132)
  assert.deepEqual(orange.material.color.toArray(), [1, 0.217637640824031, 0])
  assert.deepEqual(blue.material.color.toArray(), [0, 0.217637640824031, 1])

  const camera = new THREE.PerspectiveCamera(45, 4 / 3, 0.01, 20)
  camera.position.set(0, 0, 6)
  camera.lookAt(0, 0, 0)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 128,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.3, 'UnlitTest should render visible objects without scene lights')
  const left = meanRegion(rgba, 128, 96, 24, 32, 54, 64)
  const right = meanRegion(rgba, 128, 96, 74, 32, 104, 64)
  assert.ok(left.r > left.g + 120 && left.r > left.b + 200, `unlit orange mesh should render orange without lights (${left.r}, ${left.g}, ${left.b})`)
  assert.ok(right.b > right.g + 110 && right.b > right.r + 180, `unlit blue mesh should render blue without lights (${right.r}, ${right.g}, ${right.b})`)
})

test('committed Khronos glTF Sample Assets Triangle fixture loads minimal indexed primitive', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_TRIANGLE, 'utf8'))
  assert.deepEqual(source.buffers, [{ uri: 'Triangle.bin', byteLength: 44 }])
  assert.deepEqual(source.meshes[0].primitives, [
    {
      attributes: { POSITION: 1 },
      indices: 0,
    },
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_TRIANGLE)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'Khronos Triangle sample should load a mesh')
  assert.equal(mesh.name, 'mesh_0')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 3)
  assert.equal(mesh.geometry.getAttribute('normal') ?? null, null)
  assert.equal(mesh.geometry.getAttribute('uv') ?? null, null)
  assert.equal(mesh.geometry.index?.count, 3)
  assert.equal(mesh.material.isMeshStandardMaterial, true)
  assert.deepEqual(mesh.material.color.toArray(), [1, 1, 1])
  assert.equal(mesh.material.metalness, 1)
  assert.equal(mesh.material.roughness, 1)

  const camera = new THREE.OrthographicCamera(-0.2, 1.2, 1.2, -0.2, 0.01, 10)
  camera.position.set(0.5, 0.5, 2)
  camera.lookAt(0.5, 0.5, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1))
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 64,
    height: 64,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.035, 'Triangle sample should render visible minimal indexed geometry')
})
