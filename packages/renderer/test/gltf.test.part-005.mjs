import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'
import { Renderer, SAMPLE_ASSET_COMPARE_IRIDESCENCE, SAMPLE_ASSET_IRIDESCENCE_ABALONE, SAMPLE_ASSET_IRIDESCENCE_DIELECTRIC_SPHERES, SAMPLE_ASSET_IRIDESCENCE_METALLIC_SPHERES, SAMPLE_ASSET_IRIDESCENT_DISH_WITH_OLIVES } from './gltf.test.part-001.mjs'
import { loadGltfFixture, pngDimensions } from './gltf.test.part-028.mjs'
test('committed Khronos glTF Sample Assets CompareIridescence fixture loads iridescence comparison variants', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_COMPARE_IRIDESCENCE, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_iridescence'])
  assert.equal(source.buffers[0].uri, 'CompareIridescence.bin')
  assert.deepEqual(source.images.map((image) => image.uri), [
    'Compare_Iridescence_img0.jpg',
    'Compare_Iridescence_img1.jpg',
    'Compare_Iridescence_img2.jpg',
  ])
  assert.deepEqual(source.materials.map((material) => [
    material.name,
    material.pbrMetallicRoughness?.baseColorTexture?.index ?? null,
    material.pbrMetallicRoughness?.metallicRoughnessTexture?.index ?? null,
    material.extensions?.KHR_materials_iridescence?.iridescenceFactor ?? null,
    material.extensions?.KHR_materials_iridescence?.iridescenceIor ?? null,
    material.extensions?.KHR_materials_iridescence?.iridescenceTexture?.index ?? null,
  ]), [
    ['glTF Logo', 0, 1, null, null, null],
    ['glTF Logo Iridescence', 0, 1, 1, 1.5, 2],
  ])

  const gltf = await loadGltfFixture(SAMPLE_ASSET_COMPARE_IRIDESCENCE)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), ['GeoSphere001', 'GeoSphere002'])
  assert.deepEqual(meshes.map((mesh) => mesh.material.name), ['glTF Logo', 'glTF Logo Iridescence'])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [673, 673])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [3840, 3840])

  const [baseline, iridescent] = meshes.map((mesh) => mesh.material)
  assert.equal(baseline.isMeshStandardMaterial, true)
  assert.equal(iridescent.isMeshPhysicalMaterial, true)
  assert.equal(baseline.metalness, 1)
  assert.equal(iridescent.metalness, 1)
  assert.equal(baseline.roughness, 0.69999)
  assert.equal(iridescent.roughness, 0.69999)
  assert.equal(iridescent.iridescence, 1)
  assert.equal(iridescent.iridescenceIOR, 1.5)
  assert.deepEqual(iridescent.iridescenceThicknessRange, [100, 400])
  assert.equal(baseline.iridescenceMap ?? null, null)
  assert.equal(Buffer.isBuffer(iridescent.iridescenceMap?.image), true, 'iridescence JPEG should load as an encoded Buffer')
  assert.equal(iridescent.iridescenceMap.name, 'Compare_Iridescence_img2.jpg')
  assert.equal(iridescent.iridescenceMap.colorSpace, THREE.NoColorSpace)
  assert.equal(iridescent.iridescenceMap.flipY, false)

  assert.equal(baseline.map, iridescent.map, 'iridescence comparison materials should share the base-color texture')
  assert.equal(Buffer.isBuffer(baseline.map?.image), true, 'iridescence base-color JPEG should load as an encoded Buffer')
  assert.equal(baseline.map.name, 'Compare_Iridescence_img0.jpg')
  assert.equal(baseline.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(baseline.map.flipY, false)

  assert.equal(baseline.roughnessMap, baseline.metalnessMap)
  assert.equal(iridescent.roughnessMap, iridescent.metalnessMap)
  assert.equal(baseline.roughnessMap, iridescent.roughnessMap)
  assert.equal(Buffer.isBuffer(baseline.roughnessMap?.image), true, 'iridescence metallic-roughness JPEG should load as an encoded Buffer')
  assert.equal(baseline.roughnessMap.name, 'Compare_Iridescence_img1.jpg')
  assert.equal(baseline.roughnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(baseline.roughnessMap.flipY, false)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1))
  const light = new THREE.DirectionalLight(0xffffff, 4)
  light.position.set(2, 4, 5)
  gltf.scene.add(light)
  const camera = new THREE.PerspectiveCamera(35, 1.5, 0.01, 20)
  camera.position.copy(center).add(new THREE.Vector3(0, -2.7, 1.4))
  camera.lookAt(center)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 144,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.SRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.02, 'CompareIridescence should render visible iridescence comparison geometry')
})

test('committed Khronos glTF Sample Assets IridescenceAbalone fixture loads real iridescence and thickness maps', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_IRIDESCENCE_ABALONE, 'utf8'))
  assert.deepEqual(source.extensionsUsed, ['KHR_materials_iridescence'])
  assert.deepEqual(source.buffers, [
    { uri: 'IridescenceAbalone.bin', byteLength: 162576 },
  ])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'IridescenceAbalone_BaseColor.jpg',
    'IridescenceAbalone_Iridescence.jpg',
    'IridescenceAbalone_NormalBump.png',
    'IridescenceAbalone_ORM.jpg',
  ])
  assert.deepEqual(source.materials[0].extensions?.KHR_materials_iridescence, {
    iridescenceFactor: 1,
    iridescenceTexture: { index: 1 },
    iridescenceThicknessTexture: { index: 1 },
    iridescenceThicknessMinimum: 600,
    iridescenceThicknessMaximum: 800,
  })

  const gltf = await loadGltfFixture(SAMPLE_ASSET_IRIDESCENCE_ABALONE)
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_iridescence'))
  const mesh = gltf.scene.getObjectByName('IridescenceAbalone')
  assert.ok(mesh?.isMesh, 'IridescenceAbalone should load a named mesh')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 3159)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 3159)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 3159)
  assert.equal(mesh.geometry.index?.count, 18108)

  const material = mesh.material
  assert.equal(material.name, 'IridescenceAbalone')
  assert.equal(material.isMeshPhysicalMaterial, true)
  assert.equal(material.metalness, 0.5)
  assert.equal(material.roughness, 1)
  assert.equal(material.iridescence, 1)
  assert.equal(material.iridescenceIOR, 1.3)
  assert.deepEqual(material.iridescenceThicknessRange, [600, 800])
  assert.equal(material.map.name, 'IridescenceAbalone_BaseColor.jpg')
  assert.equal(material.normalMap.name, 'IridescenceAbalone_NormalBump.png')
  assert.deepEqual(material.normalScale.toArray(), [1, -1])
  assert.equal(material.aoMap.name, 'IridescenceAbalone_ORM.jpg')
  assert.equal(material.roughnessMap.name, 'IridescenceAbalone_ORM.jpg')
  assert.equal(material.metalnessMap.name, 'IridescenceAbalone_ORM.jpg')
  assert.equal(material.iridescenceMap.name, 'IridescenceAbalone_Iridescence.jpg')
  assert.equal(material.iridescenceThicknessMap.name, 'IridescenceAbalone_Iridescence.jpg')
  assert.equal(material.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(material.normalMap.colorSpace, THREE.NoColorSpace)
  assert.equal(material.aoMap.colorSpace, THREE.NoColorSpace)
  assert.equal(material.roughnessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(material.iridescenceMap.colorSpace, THREE.NoColorSpace)
  assert.equal(material.iridescenceThicknessMap.colorSpace, THREE.NoColorSpace)
  assert.equal(material.aoMap.channel, 0)
  assert.equal(Buffer.isBuffer(material.normalMap.image), true, 'IridescenceAbalone normal PNG should load as an encoded Buffer')
  assert.deepEqual(pngDimensions(material.normalMap.image), [2048, 1024])
  for (const texture of [
    material.map,
    material.aoMap,
    material.roughnessMap,
    material.metalnessMap,
    material.iridescenceMap,
    material.iridescenceThicknessMap,
  ]) {
    assert.equal(Buffer.isBuffer(texture.image), true, `${texture.name} should load as an encoded Buffer`)
    assert.equal(texture.flipY, false)
  }

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 50)
  camera.position.copy(center).add(new THREE.Vector3(0, size.y * 0.2, Math.max(size.x, size.y, size.z) * 2.2))
  camera.lookAt(center)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.75))
  const light = new THREE.DirectionalLight(0xffffff, 2.6)
  light.position.copy(center).add(new THREE.Vector3(2, 3, 4))
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.2, 'IridescenceAbalone should render visible iridescent shell geometry')
  const mean = meanRgba(rgba)
  assert.ok(mean.r > 15 && mean.g > 15 && mean.b > 10, `IridescenceAbalone should render lit shell pixels (${mean.r}, ${mean.g}, ${mean.b})`)
})

test('committed Khronos glTF Sample Assets IridescentDishWithOlives fixture loads iridescent glass and textured food materials', async () => {
  const source = JSON.parse(await readFile(SAMPLE_ASSET_IRIDESCENT_DISH_WITH_OLIVES, 'utf8'))
  assert.deepEqual(source.extensionsUsed, [
    'KHR_materials_ior',
    'KHR_materials_iridescence',
    'KHR_materials_transmission',
    'KHR_materials_volume',
  ])
  assert.deepEqual(source.buffers, [
    { uri: 'IridescentDishWithOlives.bin', byteLength: 830680 },
  ])
  assert.deepEqual(source.images.map((image) => image.uri), [
    'glassdish_irid.png',
    'olives_nrm.png',
    'olives_orm.png',
    'olives_col.png',
    'glasscover_nrm.png',
    'glasscover_thick.png',
    'glasscover_irid.png',
    'goldleaf_nrm.png',
    'goldleaf_orm.png',
    'goldleaf_col.png',
    'glasscover_orm.png',
  ])
  assert.deepEqual(source.materials.map((material) => material.name), [
    'glassDish',
    'olives',
    'glassCover',
    'goldLeaf',
  ])
  assert.deepEqual(source.materials[0].extensions, {
    KHR_materials_transmission: { transmissionFactor: 1 },
    KHR_materials_iridescence: {
      iridescenceFactor: 1,
      iridescenceTexture: { index: 0 },
      iridescenceThicknessMaximum: 550,
      iridescenceThicknessMinimum: 500,
      iridescenceThicknessTexture: { index: 0 },
    },
    KHR_materials_volume: { thicknessFactor: 0.01 },
  })
  assert.deepEqual(source.materials[2].extensions, {
    KHR_materials_ior: { ior: 1.5 },
    KHR_materials_iridescence: {
      iridescenceFactor: 1,
      iridescenceTexture: { index: 6 },
      iridescenceThicknessMaximum: 550,
      iridescenceThicknessMinimum: 500,
      iridescenceThicknessTexture: { index: 6 },
    },
    KHR_materials_transmission: { transmissionFactor: 1 },
    KHR_materials_volume: { thicknessFactor: 0.1, thicknessTexture: { index: 5 } },
  })

  const gltf = await loadGltfFixture(SAMPLE_ASSET_IRIDESCENT_DISH_WITH_OLIVES)
  assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_iridescence'))
  assert.deepEqual(gltf.animations.map((animation) => [
    animation.name,
    animation.tracks.map((track) => track.name),
  ]), [
    ['glassCover rotation', ['glassCover_animation.quaternion']],
  ])

  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.deepEqual(meshes.map((mesh) => mesh.name), [
    'glassDish',
    'olives',
    'glassCover',
    'goldLeaf',
  ])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('position')?.count), [1090, 10992, 1857, 924])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('normal')?.count), [1090, 10992, 1857, 924])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.getAttribute('uv')?.count), [1090, 10992, 1857, 924])
  assert.deepEqual(meshes.map((mesh) => mesh.geometry.index?.count), [6144, 51840, 10752, 4608])

  const materials = new Map(meshes.map((mesh) => [mesh.material.name, mesh.material]))
  const dish = materials.get('glassDish')
  assert.equal(dish?.isMeshPhysicalMaterial, true)
  assert.equal(dish.transmission, 1)
  assert.equal(dish.thickness, 0.01)
  assert.equal(dish.ior, 1.5)
  assert.equal(dish.iridescence, 1)
  assert.deepEqual(dish.iridescenceThicknessRange, [500, 550])
  assert.equal(dish.iridescenceMap.name, 'glassdish_irid.png')
  assert.equal(dish.iridescenceThicknessMap.name, 'glassdish_irid.png')

  const olives = materials.get('olives')
  assert.equal(olives?.isMeshStandardMaterial, true)
  assert.equal(olives.map.name, 'olives_col.png')
  assert.equal(olives.normalMap.name, 'olives_nrm.png')
  assert.equal(olives.aoMap.name, 'olives_orm.png')
  assert.equal(olives.roughnessMap.name, 'olives_orm.png')
  assert.equal(olives.metalnessMap.name, 'olives_orm.png')
  assert.deepEqual(olives.normalScale.toArray(), [1, -1])

  const cover = materials.get('glassCover')
  assert.equal(cover?.isMeshPhysicalMaterial, true)
  assert.equal(cover.transmission, 1)
  assert.equal(cover.thickness, 0.1)
  assert.equal(cover.ior, 1.5)
  assert.equal(cover.iridescence, 1)
  assert.deepEqual(cover.iridescenceThicknessRange, [500, 550])
  assert.deepEqual(cover.normalScale.toArray(), [2, -2])
  assert.equal(cover.normalMap.name, 'glasscover_nrm.png')
  assert.equal(cover.roughnessMap.name, 'glasscover_orm.png')
  assert.equal(cover.metalnessMap.name, 'glasscover_orm.png')
  assert.equal(cover.thicknessMap.name, 'glasscover_thick.png')
  assert.equal(cover.iridescenceMap.name, 'glasscover_irid.png')
  assert.equal(cover.iridescenceThicknessMap.name, 'glasscover_irid.png')

  const goldLeaf = materials.get('goldLeaf')
  assert.equal(goldLeaf?.isMeshStandardMaterial, true)
  assert.equal(goldLeaf.alphaTest, 0.5)
  assert.equal(goldLeaf.map.name, 'goldleaf_col.png')
  assert.equal(goldLeaf.normalMap.name, 'goldleaf_nrm.png')
  assert.equal(goldLeaf.aoMap.name, 'goldleaf_orm.png')
  assert.equal(goldLeaf.roughnessMap.name, 'goldleaf_orm.png')
  assert.equal(goldLeaf.metalnessMap.name, 'goldleaf_orm.png')

  const smallPngTextures = [
    olives.map,
    olives.normalMap,
    olives.aoMap,
    olives.roughnessMap,
    olives.metalnessMap,
  ]
  for (const texture of smallPngTextures) {
    assert.equal(Buffer.isBuffer(texture.image), true, `${texture.name} should load as an encoded Buffer`)
    assert.deepEqual(pngDimensions(texture.image), [512, 512])
    assert.equal(texture.flipY, false)
  }
  assert.equal(olives.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(olives.normalMap.colorSpace, THREE.NoColorSpace)
  assert.equal(olives.aoMap.colorSpace, THREE.NoColorSpace)

  const widePngTextures = [
    dish.iridescenceMap,
    dish.iridescenceThicknessMap,
    cover.normalMap,
    cover.roughnessMap,
    cover.metalnessMap,
    cover.thicknessMap,
    cover.iridescenceMap,
    cover.iridescenceThicknessMap,
    goldLeaf.map,
    goldLeaf.normalMap,
    goldLeaf.aoMap,
    goldLeaf.roughnessMap,
    goldLeaf.metalnessMap,
  ]
  for (const texture of widePngTextures) {
    assert.equal(Buffer.isBuffer(texture.image), true, `${texture.name} should load as an encoded Buffer`)
    assert.deepEqual(pngDimensions(texture.image), [2048, 1024])
    assert.equal(texture.flipY, false)
  }
  assert.equal(goldLeaf.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(cover.normalMap.colorSpace, THREE.NoColorSpace)
  assert.equal(cover.thicknessMap.colorSpace, THREE.NoColorSpace)

  const bounds = new THREE.Box3().setFromObject(gltf.scene)
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100)
  camera.position.copy(center).add(new THREE.Vector3(0, size.y * 0.15, Math.max(size.x, size.y, size.z) * 2.4))
  camera.lookAt(center)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 2.8)
  light.position.copy(center).add(new THREE.Vector3(2, 3, 4))
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.04, 'IridescentDishWithOlives should render visible dish and food geometry')
})

test('committed Khronos glTF Sample Assets iridescence sphere-grid fixtures load dielectric and metallic variants', async () => {
  const cases = [
    {
      label: 'IridescenceDielectricSpheres',
      filePath: SAMPLE_ASSET_IRIDESCENCE_DIELECTRIC_SPHERES,
      bufferUri: 'IridescenceDielectricSpheres.bin',
      expectedBufferLength: 34068,
      expectedMaterialIors: [1, 1.17, 1.33, 1.5, 1.67, 1.83, 2],
      expectedMetalness: 0,
      minimumVisibleRatio: 0.2,
    },
    {
      label: 'IridescenceMetallicSpheres',
      filePath: SAMPLE_ASSET_IRIDESCENCE_METALLIC_SPHERES,
      bufferUri: 'IridescenceMetallicSpheres.bin',
      expectedBufferLength: 34068,
      expectedMaterialIors: [1.5],
      expectedMetalness: 1,
      minimumVisibleRatio: 0.15,
    },
  ]

  for (const fixture of cases) {
    const source = JSON.parse(await readFile(fixture.filePath, 'utf8'))
    assert.deepEqual(source.extensionsUsed, ['KHR_materials_ior', 'KHR_materials_iridescence'])
    assert.deepEqual(source.buffers, [{ uri: fixture.bufferUri, byteLength: fixture.expectedBufferLength }])
    assert.deepEqual(source.images, [{ name: 'guides', mimeType: 'image/png', uri: 'textures/guides.png' }])
    assert.equal(source.materials.length, 344)
    assert.equal(source.meshes.length, 346)
    assert.equal(source.nodes.length, 346)

    const iridescentSources = source.materials.filter((material) => material.extensions?.KHR_materials_iridescence)
    assert.equal(iridescentSources.length, 343)
    assert.deepEqual(
      [...new Set(iridescentSources.map((material) => Number((material.extensions.KHR_materials_ior?.ior ?? 1.5).toFixed(2))))].sort((a, b) => a - b),
      fixture.expectedMaterialIors,
    )
    assert.deepEqual(
      [...new Set(iridescentSources.map((material) => Number(material.extensions.KHR_materials_iridescence.iridescenceIor.toFixed(2))))].sort((a, b) => a - b),
      [1, 1.17, 1.33, 1.5, 1.67, 1.83, 2],
    )
    assert.deepEqual(
      [...new Set(iridescentSources.map((material) => material.extensions.KHR_materials_iridescence.iridescenceThicknessMaximum ?? 400))].sort((a, b) => a - b),
      [100, 200, 300, 400, 500, 600, 700],
    )

    const gltf = await loadGltfFixture(fixture.filePath)
    assert.ok(gltf.parser?.json?.extensionsUsed?.includes('KHR_materials_iridescence'))
    const meshes = []
    gltf.scene.traverse((object) => {
      if (object.isMesh === true) meshes.push(object)
    })
    assert.equal(meshes.length, 346)

    const sphereMeshes = meshes.filter((mesh) => /^Sphere\d+$/.test(mesh.name))
    assert.equal(sphereMeshes.length, 343)
    assert.ok(sphereMeshes.every((mesh) => mesh.material.isMeshPhysicalMaterial === true), `${fixture.label} spheres should use MeshPhysicalMaterial`)
    assert.ok(sphereMeshes.every((mesh) => mesh.geometry.getAttribute('position')?.count === 961))
    assert.ok(sphereMeshes.every((mesh) => mesh.geometry.index?.count === 5400))

    const sphereMaterials = sphereMeshes.map((mesh) => mesh.material)
    assert.deepEqual([...new Set(sphereMaterials.map((material) => Number(material.ior.toFixed(2))))].sort((a, b) => a - b), fixture.expectedMaterialIors)
    assert.deepEqual([...new Set(sphereMaterials.map((material) => material.metalness))].sort((a, b) => a - b), [fixture.expectedMetalness])
    assert.ok(sphereMaterials.every((material) => material.iridescence === 1))
    assert.deepEqual([...new Set(sphereMaterials.map((material) => Number(material.iridescenceIOR.toFixed(2))))].sort((a, b) => a - b), [1, 1.17, 1.33, 1.5, 1.67, 1.83, 2])
    assert.deepEqual([...new Set(sphereMaterials.map((material) => material.iridescenceThicknessRange[1]))].sort((a, b) => a - b), [100, 200, 300, 400, 500, 600, 700])

    const guides = meshes.filter((mesh) => ['ThicknessPlane', 'IorPlane', 'ThinFilmIorPlane'].includes(mesh.name))
    assert.equal(guides.length, 3)
    assert.ok(guides.every((mesh) => mesh.material.name === 'Guides Material'))
    assert.ok(guides.every((mesh) => mesh.geometry.getAttribute('position')?.count === 4))
    assert.ok(guides.every((mesh) => mesh.geometry.index?.count === 6))
    assert.equal(guides[0].material.map, guides[1].material.map)
    assert.equal(guides[0].material.map, guides[2].material.map)
    assert.equal(Buffer.isBuffer(guides[0].material.map?.image), true, `${fixture.label} guide PNG should load as an encoded Buffer`)
    assert.equal(guides[0].material.map.name, 'guides')
    assert.deepEqual(pngDimensions(guides[0].material.map.image), [2048, 2048])
    assert.equal(guides[0].material.map.colorSpace, THREE.SRGBColorSpace)

    const bounds = new THREE.Box3().setFromObject(gltf.scene)
    const center = bounds.getCenter(new THREE.Vector3())
    const size = bounds.getSize(new THREE.Vector3())
    gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(2, 3, 4)
    gltf.scene.add(light)
    const camera = new THREE.OrthographicCamera(
      -size.x / 2 - 0.25,
      size.x / 2 + 0.25,
      size.y / 2 + 0.25,
      -size.y / 2 - 0.25,
      0.01,
      20,
    )
    camera.position.copy(center).add(new THREE.Vector3(0, 0, 8))
    camera.lookAt(center)
    gltf.scene.updateMatrixWorld(true)
    camera.updateMatrixWorld(true)

    const rgba = new Renderer().render(gltf.scene, camera, {
      width: 128,
      height: 128,
      format: 'rgba',
      background: [0, 0, 0],
      outputColorSpace: THREE.LinearSRGBColorSpace,
    })

    assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > fixture.minimumVisibleRatio, `${fixture.label} should render visible iridescence sphere-grid geometry`)
  }
})
