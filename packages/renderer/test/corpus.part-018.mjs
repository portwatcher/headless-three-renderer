import * as THREE from 'three'
import { CORPUS_RENDER_SIZE, countRegionPixels, meanRegion } from './corpus.part-001.mjs'
export function batchedMeshMultiSourceGroupOffsetsCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const padding = new THREE.BufferGeometry()
  padding.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.1, -0.1, 0,
    0.1, -0.1, 0,
    0, 0.1, 0,
  ]), 3))
  padding.setIndex([0, 1, 2])

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
  source.addGroup(0, 6, 0)
  source.addGroup(6, 6, 1)

  const batch = new THREE.BatchedMesh(
    1,
    padding.getAttribute('position').count + source.getAttribute('position').count,
    padding.index.count + source.index.count,
    [
      new THREE.MeshBasicMaterial({ color: 0xff0000 }),
      new THREE.MeshBasicMaterial({ color: 0x00ff00 }),
    ],
  )
  batch.addGeometry(padding)
  const geometryId = batch.addGeometry(source)
  batch.addInstance(geometryId)
  batch.perObjectFrustumCulled = false

  const range = batch.getGeometryRangeAt(geometryId, {})
  batch.geometry.clearGroups()
  for (const group of source.groups) {
    batch.geometry.addGroup(range.start + group.start, group.count, group.materialIndex)
  }
  scene.add(batch)

  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 1.2, -1.2, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  return {
    name: 'batched-mesh-multi-source-group-offsets',
    scene,
    camera,
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    browserReference: false,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.08,
    validate(rgba, { width }) {
      const left = meanRegion(rgba, width, 20, 42, 30, 54)
      const right = meanRegion(rgba, width, 66, 42, 76, 54)
      if (!(left.r > left.g + 80 && left.r > left.b + 80 && right.g > right.r + 70 && right.g > right.b + 80)) {
        throw new Error(`BatchedMesh translated source groups should render left red and right green, got left=${JSON.stringify(left)} right=${JSON.stringify(right)}`)
      }
    },
  }
}

export function batchedMeshNonIndexedGroupsCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const source = new THREE.BufferGeometry()
  source.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.9, -0.45, 0,
    -0.25, -0.45, 0,
    -0.25, 0.45, 0,
    -0.9, -0.45, 0,
    -0.25, 0.45, 0,
    -0.9, 0.45, 0,
    0.25, -0.45, 0,
    0.9, -0.45, 0,
    0.9, 0.45, 0,
    0.25, -0.45, 0,
    0.9, 0.45, 0,
    0.25, 0.45, 0,
  ]), 3))

  const batch = new THREE.BatchedMesh(
    1,
    source.getAttribute('position').count,
    0,
    [
      new THREE.MeshBasicMaterial({ color: 0xff0000 }),
      new THREE.MeshBasicMaterial({ color: 0x00ff00 }),
    ],
  )
  const geometryId = batch.addGeometry(source)
  batch.addInstance(geometryId)
  batch.perObjectFrustumCulled = false

  const range = batch.getGeometryRangeAt(geometryId, {})
  batch.geometry.clearGroups()
  batch.geometry.addGroup(range.start, 6, 0)
  batch.geometry.addGroup(range.start + 6, 6, 1)
  scene.add(batch)

  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 1.2, -1.2, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  return {
    name: 'batched-mesh-non-indexed-groups',
    scene,
    camera,
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.08,
    validate(rgba, { width }) {
      const left = meanRegion(rgba, width, 20, 42, 30, 54)
      const right = meanRegion(rgba, width, 66, 42, 76, 54)
      if (!(left.r > left.g + 80 && left.r > left.b + 80 && right.g > right.r + 70 && right.g > right.b + 80)) {
        throw new Error(`BatchedMesh non-indexed material groups should render left red and right green, got left=${JSON.stringify(left)} right=${JSON.stringify(right)}`)
      }
    },
  }
}

export function batchedMeshDefaultGroupMaterialCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

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

  const batch = new THREE.BatchedMesh(
    1,
    source.getAttribute('position').count,
    source.index.count,
    [
      new THREE.MeshBasicMaterial({ color: 0xff0000 }),
      new THREE.MeshBasicMaterial({ color: 0x00ff00 }),
    ],
  )
  const geometryId = batch.addGeometry(source)
  batch.addInstance(geometryId)
  batch.perObjectFrustumCulled = false

  const range = batch.getGeometryRangeAt(geometryId, {})
  batch.geometry.clearGroups()
  batch.geometry.addGroup(range.start, 6)
  batch.geometry.addGroup(range.start + 6, 6, 1)
  batch._geometryInfo[geometryId].start = range.start + 3
  batch._geometryInfo[geometryId].count = 6
  scene.add(batch)

  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 1.2, -1.2, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  return {
    name: 'batched-mesh-default-group-material',
    scene,
    camera,
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    browserReference: false,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.015,
    validate(rgba, { width }) {
      const redPixels = countRegionPixels(rgba, width, 12, 21, 42, 51, (r, g, b) => r > 120 && r > g + 50 && r > b + 50)
      const greenPixels = countRegionPixels(rgba, width, 62, 45, 90, 78, (r, g, b) => g > 120 && g > r + 50 && g > b + 50)
      if (!(redPixels > 150 && greenPixels > 150)) {
        throw new Error(`BatchedMesh missing materialIndex group should default to red material zero while explicit group stays green, red=${redPixels} green=${greenPixels}`)
      }
    },
  }
}

export function batchedMeshPartialGroupRangeCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

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

  const batch = new THREE.BatchedMesh(
    1,
    source.getAttribute('position').count,
    source.index.count,
    [
      new THREE.MeshBasicMaterial({ color: 0xff0000 }),
      new THREE.MeshBasicMaterial({ color: 0x00ff00 }),
    ],
  )
  const geometryId = batch.addGeometry(source)
  batch.addInstance(geometryId)
  batch.perObjectFrustumCulled = false

  const range = batch.getGeometryRangeAt(geometryId, {})
  batch.geometry.clearGroups()
  batch.geometry.addGroup(range.start, 6, 0)
  batch.geometry.addGroup(range.start + 6, 6, 1)
  batch._geometryInfo[geometryId].start = range.start + 3
  batch._geometryInfo[geometryId].count = 6
  scene.add(batch)

  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 1.2, -1.2, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  return {
    name: 'batched-mesh-partial-group-range',
    scene,
    camera,
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.015,
    validate(rgba, { width }) {
      const redPixels = countRegionPixels(rgba, width, 12, 21, 42, 51, (r, g, b) => r > 120 && r > g + 50 && r > b + 50)
      const greenPixels = countRegionPixels(rgba, width, 62, 45, 90, 78, (r, g, b) => g > 120 && g > r + 50 && g > b + 50)
      const clippedLeft = meanRegion(rgba, width, 30, 54, 40, 69)
      const clippedRight = meanRegion(rgba, width, 56, 27, 66, 42)
      if (!(redPixels > 150 && greenPixels > 150 && clippedLeft.r < 5 && clippedLeft.g < 5 && clippedLeft.b < 5 && clippedRight.r < 5 && clippedRight.g < 5 && clippedRight.b < 5)) {
        throw new Error(`BatchedMesh partial group range should clip source groups while preserving material colors, red=${redPixels} green=${greenPixels} clippedLeft=${JSON.stringify(clippedLeft)} clippedRight=${JSON.stringify(clippedRight)}`)
      }
    },
  }
}

export function batchedMeshSparseMaterialGroupsCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const source = new THREE.BufferGeometry()
  source.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.95, -0.45, 0,
    -0.35, -0.45, 0,
    -0.35, 0.45, 0,
    -0.95, 0.45, 0,
    -0.2, -0.45, 0,
    0.2, -0.45, 0,
    0.2, 0.45, 0,
    -0.2, 0.45, 0,
    0.35, -0.45, 0,
    0.95, -0.45, 0,
    0.95, 0.45, 0,
    0.35, 0.45, 0,
  ]), 3))
  source.setIndex([
    0, 1, 2,
    0, 2, 3,
    4, 5, 6,
    4, 6, 7,
    8, 9, 10,
    8, 10, 11,
  ])

  const batch = new THREE.BatchedMesh(
    1,
    source.getAttribute('position').count,
    source.index.count,
    [
      new THREE.MeshBasicMaterial({ color: 0xff0000 }),
      new THREE.MeshBasicMaterial({ color: 0x00ff00 }),
    ],
  )
  const geometryId = batch.addGeometry(source)
  batch.addInstance(geometryId)
  batch.perObjectFrustumCulled = false

  const range = batch.getGeometryRangeAt(geometryId, {})
  batch.geometry.clearGroups()
  batch.geometry.addGroup(range.start, 6, 0)
  batch.geometry.addGroup(range.start + 12, 6, 1)
  batch._geometryInfo[geometryId].start = range.start + 3
  batch._geometryInfo[geometryId].count = 12
  scene.add(batch)

  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 1.2, -1.2, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  return {
    name: 'batched-mesh-sparse-material-groups',
    scene,
    camera,
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.015,
    validate(rgba, { width }) {
      const redPixels = countRegionPixels(rgba, width, 10, 16, 40, 80, (r, g, b) => r > 120 && r > g + 50 && r > b + 50)
      const greenPixels = countRegionPixels(rgba, width, 58, 16, 88, 80, (r, g, b) => g > 120 && g > r + 50 && g > b + 50)
      const gap = meanRegion(rgba, width, 42, 34, 54, 62)
      if (!(redPixels > 40 && greenPixels > 40 && gap.r < 5 && gap.g < 5 && gap.b < 5)) {
        throw new Error(`BatchedMesh sparse material groups should keep red/green clipped groups and skip the center gap, red=${redPixels} green=${greenPixels} gap=${JSON.stringify(gap)}`)
      }
    },
  }
}

export function batchedMeshCullingCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const visibleSource = new THREE.PlaneGeometry(0.6, 0.6)
  const culledSource = new THREE.PlaneGeometry(2.2, 2.2)
  culledSource.boundingSphere = new THREE.Sphere(new THREE.Vector3(5, 0, 0), 0.05)

  const batch = new THREE.BatchedMesh(
    2,
    visibleSource.getAttribute('position').count + culledSource.getAttribute('position').count,
    visibleSource.index.count + culledSource.index.count,
    new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false }),
  )
  const visibleGeometryId = batch.addGeometry(visibleSource)
  const culledGeometryId = batch.addGeometry(culledSource)
  const visible = batch.addInstance(visibleGeometryId)
  const culled = batch.addInstance(culledGeometryId)
  batch.setColorAt(visible, new THREE.Color(0.05, 0.95, 0.1))
  batch.setColorAt(culled, new THREE.Color(1, 0, 0))
  scene.add(batch)

  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 1.2, -1.2, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  return {
    name: 'batched-mesh-per-object-culling',
    scene,
    camera,
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.03,
    validate(rgba, { width, height }) {
      const x = Math.floor(width / 2)
      const y = Math.floor(height / 2)
      const offset = (y * width + x) * 4
      const r = rgba[offset]
      const g = rgba[offset + 1]
      const b = rgba[offset + 2]
      if (g <= r + 60 || g <= b + 80) {
        throw new Error(`batched culling should leave the center green, got rgb(${r}, ${g}, ${b})`)
      }
    },
  }
}

export function batchedMeshCullingOptOutCorpus() {
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  const options = { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' }
  const stats = {}

  function makeScene({ frustumCulled, perObjectFrustumCulled }) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0, 0, 0)

    const source = new THREE.PlaneGeometry(2, 2)
    source.boundingSphere = new THREE.Sphere(new THREE.Vector3(4, 0, 0), 0.1)
    const batch = new THREE.BatchedMesh(
      1,
      source.getAttribute('position').count,
      source.index.count,
      new THREE.MeshBasicMaterial({ color: 0xff0000 }),
    )
    const geometryId = batch.addGeometry(source)
    const instanceId = batch.addInstance(geometryId)
    batch.setMatrixAt(instanceId, new THREE.Matrix4())
    batch.frustumCulled = frustumCulled
    batch.perObjectFrustumCulled = perObjectFrustumCulled
    scene.add(batch)

    return scene
  }

  function centerMean(rgba) {
    return meanRegion(rgba, options.width, 32, 32, 64, 64)
  }

  return {
    name: 'batched-mesh-culling-opt-outs',
    scene: makeScene({ frustumCulled: false, perObjectFrustumCulled: false }),
    camera,
    options,
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.5,
    render(renderer) {
      const perObjectCulled = renderer.render(
        makeScene({ frustumCulled: false, perObjectFrustumCulled: true }),
        camera,
        options,
      )
      const perObjectOptOut = renderer.render(
        makeScene({ frustumCulled: false, perObjectFrustumCulled: false }),
        camera,
        options,
      )
      const aggregateCulled = renderer.render(
        makeScene({ frustumCulled: true, perObjectFrustumCulled: false }),
        camera,
        options,
      )
      stats.perObjectCulled = centerMean(perObjectCulled)
      stats.perObjectOptOut = centerMean(perObjectOptOut)
      stats.aggregateCulled = centerMean(aggregateCulled)
      return perObjectOptOut
    },
    validate() {
      if (!(stats.perObjectCulled.r < 5 && stats.perObjectCulled.g < 5 && stats.perObjectCulled.b < 5)) {
        throw new Error(`BatchedMesh per-object bounds should cull the off-frustum geometry, stats=${JSON.stringify(stats)}`)
      }
      if (!(stats.perObjectOptOut.r > 200 && stats.perObjectOptOut.g < 40 && stats.perObjectOptOut.b < 40)) {
        throw new Error(`BatchedMesh perObjectFrustumCulled=false should render the geometry, stats=${JSON.stringify(stats)}`)
      }
      if (!(stats.aggregateCulled.r < 5 && stats.aggregateCulled.g < 5 && stats.aggregateCulled.b < 5)) {
        throw new Error(`BatchedMesh aggregate frustum culling should still cull with object frustumCulled=true, stats=${JSON.stringify(stats)}`)
      }
    },
  }
}
