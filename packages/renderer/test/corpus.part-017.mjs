import * as THREE from 'three'
import { CORPUS_RENDER_SIZE, countRegionPixels, makeCamera, meanRegion, pixelAt, solidTexture } from './corpus.part-001.mjs'
export function instancedLinesPointsCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const pointGeometry = new THREE.InstancedBufferGeometry()
  pointGeometry.instanceCount = 3
  pointGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0.35, 0]), 3))
  pointGeometry.setAttribute('instanceOffset', new THREE.InstancedBufferAttribute(new Float32Array([
    -0.55, 0, 0,
    0, 0, 0,
    0.55, 0, 0,
  ]), 3))
  pointGeometry.setAttribute('color', new THREE.InstancedBufferAttribute(new Float32Array([
    1, 0, 0,
    0, 1, 0,
    0, 0.4, 1,
  ]), 3))
  scene.add(new THREE.Points(pointGeometry, new THREE.PointsMaterial({
    color: 0xffffff,
    vertexColors: true,
    size: 18,
    sizeAttenuation: false,
    map: solidTexture(255, 255, 255),
  })))

  const lineGeometry = new THREE.InstancedBufferGeometry()
  lineGeometry.instanceCount = 2
  lineGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.35, -0.35, 0,
    0.35, -0.35, 0,
  ]), 3))
  lineGeometry.setAttribute('instanceOffset', new THREE.InstancedBufferAttribute(new Float32Array([
    -0.35, 0, 0,
    0.35, 0, 0,
  ]), 3))
  lineGeometry.setAttribute('color', new THREE.InstancedBufferAttribute(new Float32Array([
    1, 1, 0,
    0, 1, 1,
  ]), 3))
  scene.add(new THREE.LineSegments(lineGeometry, new THREE.LineBasicMaterial({
    color: 0xffffff,
    vertexColors: true,
  })))

  return {
    name: 'instanced-lines-and-points',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    validate(rgba, { width, height }) {
      const redPixels = countRegionPixels(rgba, width, 0, 0, width, height, (r, g, b) => r > 120 && r > g + 60 && r > b + 60)
      const greenPixels = countRegionPixels(rgba, width, 0, 0, width, height, (r, g, b) => g > 120 && g > r + 60 && g > b + 60)
      const bluePixels = countRegionPixels(rgba, width, 0, 0, width, height, (r, g, b) => b > 120 && b > r + 35 && b > g + 35)
      const yellowPixels = countRegionPixels(rgba, width, 0, 0, width, height, (r, g, b) => r > 120 && g > 120 && b < 120)
      const cyanPixels = countRegionPixels(rgba, width, 0, 0, width, height, (r, g, b) => g > 120 && b > 120 && r < 120)
      if (!(redPixels > 250 && greenPixels > 250 && bluePixels > 250 && yellowPixels > 250 && cyanPixels > 250)) {
        throw new Error(`instanced line/point corpus should render all per-instance colors, got red=${redPixels} green=${greenPixels} blue=${bluePixels} yellow=${yellowPixels} cyan=${cyanPixels}`)
      }
    },
  }
}

export function instancedLineNoBridgeCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  function makeGeometry(y) {
    const geometry = new THREE.InstancedBufferGeometry()
    geometry.instanceCount = 2
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
      -0.25, y, 0,
      0.25, y, 0,
    ]), 3))
    geometry.setAttribute('instanceOffset', new THREE.InstancedBufferAttribute(new Float32Array([
      -0.5, 0, 0,
      0.5, 0, 0,
    ]), 3))
    geometry.setAttribute('color', new THREE.InstancedBufferAttribute(new Float32Array([
      1, 0, 0,
      0, 1, 0,
    ]), 3))
    return geometry
  }

  function makeMaterial() {
    return new THREE.LineBasicMaterial({
      color: 0xffffff,
      linewidth: 8,
      vertexColors: true,
    })
  }

  scene.add(new THREE.Line(makeGeometry(0.4), makeMaterial()))
  scene.add(new THREE.LineLoop(makeGeometry(-0.4), makeMaterial()))

  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 1.2, -1.2, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  return {
    name: 'instanced-line-no-bridge',
    scene,
    camera,
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.004,
    browserReference: false,
    validate(rgba, { width }) {
      function checkRow(label, minY, maxY) {
        const redPixels = countRegionPixels(rgba, width, 16, minY, 40, maxY, (r, g, b) => r > g + 30 && r > b + 30)
        const greenPixels = countRegionPixels(rgba, width, 56, minY, 80, maxY, (r, g, b) => g > r + 30 && g > b + 30)
        const bridgePixels = countRegionPixels(rgba, width, 42, minY, 54, maxY, (r, g, b) => r > 30 || g > 30 || b > 30)
        if (redPixels < 20 || greenPixels < 20 || bridgePixels > 1) {
          throw new Error(`${label} instanced line corpus should draw both instances without a center bridge, red=${redPixels} green=${greenPixels} bridge=${bridgePixels}`)
        }
      }

      checkRow('Line', 26, 38)
      checkRow('LineLoop', 58, 70)
    },
  }
}

export function instancedTextureUvCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const map = new THREE.DataTexture(new Uint8Array([
    255, 0, 0, 255,
    0, 255, 0, 255,
  ]), 2, 1, THREE.RGBAFormat)
  map.magFilter = THREE.NearestFilter
  map.minFilter = THREE.NearestFilter
  map.channel = 1
  map.needsUpdate = true

  const base = new THREE.PlaneGeometry(0.34, 0.34)
  const meshGeometry = new THREE.InstancedBufferGeometry()
  meshGeometry.index = base.index
  meshGeometry.setAttribute('position', base.getAttribute('position'))
  meshGeometry.setAttribute('uv', base.getAttribute('uv'))
  meshGeometry.setAttribute('instanceOffset', new THREE.InstancedBufferAttribute(new Float32Array([
    -0.52, 0.35, 0,
    0.52, 0.35, 0,
  ]), 3))
  meshGeometry.setAttribute('uv1', new THREE.InstancedBufferAttribute(new Float32Array([
    0.25, 0.5,
    0.75, 0.5,
  ]), 2))
  scene.add(new THREE.Mesh(
    meshGeometry,
    new THREE.MeshBasicMaterial({ color: 0xffffff, map }),
  ))

  const lineGeometry = new THREE.InstancedBufferGeometry()
  lineGeometry.instanceCount = 2
  lineGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.18, -0.08, 0,
    0.18, -0.08, 0,
  ]), 3))
  lineGeometry.setAttribute('instanceOffset', new THREE.InstancedBufferAttribute(new Float32Array([
    -0.52, 0, 0,
    0.52, 0, 0,
  ]), 3))
  lineGeometry.setAttribute('uv1', new THREE.InstancedBufferAttribute(new Float32Array([
    0.25, 0.5,
    0.75, 0.5,
  ]), 2))
  scene.add(new THREE.LineSegments(
    lineGeometry,
    new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 8, map }),
  ))

  const pointGeometry = new THREE.InstancedBufferGeometry()
  pointGeometry.instanceCount = 2
  pointGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, -0.48, 0]), 3))
  pointGeometry.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([0.5, 0.5]), 2))
  pointGeometry.setAttribute('instanceOffset', new THREE.InstancedBufferAttribute(new Float32Array([
    -0.52, 0, 0,
    0.52, 0, 0,
  ]), 3))
  pointGeometry.setAttribute('uv1', new THREE.InstancedBufferAttribute(new Float32Array([
    0.25, 0.5,
    0.75, 0.5,
  ]), 2))
  scene.add(new THREE.Points(pointGeometry, new THREE.PointsMaterial({
    color: 0xffffff,
    map,
    size: 18,
    sizeAttenuation: false,
  })))

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  return {
    name: 'instanced-texture-uv-streams',
    scene,
    camera,
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.02,
    validate(rgba, { width }) {
      const samples = [
        ['left mesh', meanRegion(rgba, width, 20, 28, 30, 38), 'red'],
        ['right mesh', meanRegion(rgba, width, 66, 28, 76, 38), 'green'],
        ['left line', meanRegion(rgba, width, 17, 48, 31, 57), 'red'],
        ['right line', meanRegion(rgba, width, 65, 48, 79, 57), 'green'],
        ['left point', meanRegion(rgba, width, 18, 64, 32, 78), 'red'],
        ['right point', meanRegion(rgba, width, 66, 64, 80, 78), 'green'],
      ]
      for (const [label, color, expected] of samples) {
        if (expected === 'red' && color.r <= color.g + 45) {
          throw new Error(`${label} should sample the red instanced UV texel, got rgb(${color.r}, ${color.g}, ${color.b})`)
        }
        if (expected === 'green' && color.g <= color.r + 45) {
          throw new Error(`${label} should sample the green instanced UV texel, got rgb(${color.r}, ${color.g}, ${color.b})`)
        }
      }
    },
  }
}

export function renderableFrustumCullingCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const culledGeometry = new THREE.PlaneGeometry(0.62, 0.62)
  culledGeometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(5, 0, 0), 0.05)
  const culled = new THREE.Mesh(
    culledGeometry,
    new THREE.MeshBasicMaterial({ color: 0xff0000 }),
  )
  culled.position.set(-0.42, 0, 0)
  scene.add(culled)

  const uncullableGeometry = new THREE.PlaneGeometry(0.62, 0.62)
  uncullableGeometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(5, 0, 0), 0.05)
  const uncullable = new THREE.Mesh(
    uncullableGeometry,
    new THREE.MeshBasicMaterial({ color: 0x00ff00 }),
  )
  uncullable.frustumCulled = false
  uncullable.position.set(0.42, 0, 0)
  scene.add(uncullable)

  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 1.2, -1.2, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  return {
    name: 'renderable-frustum-culling',
    scene,
    camera,
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.02,
    validate(rgba, { width, height }) {
      const redPixels = countRegionPixels(rgba, width, 0, 0, Math.floor(width / 2), height, (r, g, b) => r > 120 && r > g + 50 && r > b + 50)
      const greenPixels = countRegionPixels(rgba, width, Math.floor(width / 2), 0, width, height, (r, g, b) => g > 120 && g > r + 50 && g > b + 50)
      if (redPixels > 5 || greenPixels < 200) {
        throw new Error(`renderable frustum culling should skip the red object and keep frustumCulled=false green visible, got red=${redPixels} green=${greenPixels}`)
      }
    },
  }
}

export function batchedMeshCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const source = new THREE.PlaneGeometry(0.42, 0.42)
  const batch = new THREE.BatchedMesh(
    3,
    source.getAttribute('position').count,
    source.index.count,
    new THREE.MeshBasicMaterial({ color: 0xffffff }),
  )
  const geometryId = batch.addGeometry(source)
  const left = batch.addInstance(geometryId)
  const right = batch.addInstance(geometryId)
  const hidden = batch.addInstance(geometryId)
  batch.setMatrixAt(left, new THREE.Matrix4().makeTranslation(-0.52, 0, 0))
  batch.setMatrixAt(right, new THREE.Matrix4().makeTranslation(0.52, 0, 0))
  batch.setMatrixAt(hidden, new THREE.Matrix4().makeTranslation(0, 0, 0))
  batch.setColorAt(left, new THREE.Color(1, 0.15, 0.05))
  batch.setColorAt(right, new THREE.Color(0.05, 0.9, 0.25))
  batch.setColorAt(hidden, new THREE.Color(0.1, 0.2, 1))
  batch.setVisibleAt(hidden, false)
  scene.add(batch)

  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 1.2, -1.2, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  return {
    name: 'batched-mesh-instance-colors',
    scene,
    camera,
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.03,
    validate(rgba, { width, height }) {
      const redPixels = countRegionPixels(rgba, width, 0, 0, width, height, (r, g, b) => r > 120 && r > g + 50 && r > b + 50)
      const greenPixels = countRegionPixels(rgba, width, 0, 0, width, height, (r, g, b) => g > 120 && g > r + 50 && g > b + 50)
      const bluePixels = countRegionPixels(rgba, width, 0, 0, width, height, (r, g, b) => b > 120 && b > r + 50 && b > g + 50)
      const hidden = pixelAt(rgba, width, 48, 48)
      if (!(redPixels > 200 && greenPixels > 200 && bluePixels < 5 && hidden.r < 5 && hidden.g < 5 && hidden.b < 5)) {
        throw new Error(`BatchedMesh corpus should render red/green visible instances and hide the blue instance, got red=${redPixels} green=${greenPixels} blue=${bluePixels} hidden=${JSON.stringify(hidden)}`)
      }
    },
  }
}

export function batchedMeshInactiveGeometryCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const source = new THREE.PlaneGeometry(0.42, 0.42)
  const batch = new THREE.BatchedMesh(
    2,
    source.getAttribute('position').count * 2,
    source.index.count * 2,
    new THREE.MeshBasicMaterial({ color: 0xffffff }),
  )
  const activeGeometryId = batch.addGeometry(source)
  const deletedGeometryId = batch.addGeometry(source.clone())
  const left = batch.addInstance(activeGeometryId)
  const right = batch.addInstance(deletedGeometryId)
  batch.setMatrixAt(left, new THREE.Matrix4().makeTranslation(-0.52, 0, 0))
  batch.setMatrixAt(right, new THREE.Matrix4().makeTranslation(0.52, 0, 0))
  batch.setColorAt(left, new THREE.Color(1, 0.05, 0.05))
  batch.setColorAt(right, new THREE.Color(0.05, 1, 0.05))
  batch.deleteGeometry(deletedGeometryId)
  scene.add(batch)

  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 1.2, -1.2, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  return {
    name: 'batched-mesh-inactive-geometry',
    scene,
    camera,
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.01,
    validate(rgba, { width, height }) {
      const y = Math.floor(height / 2)
      const leftOffset = (y * width + Math.floor(width * 0.29)) * 4
      const rightOffset = (y * width + Math.floor(width * 0.71)) * 4
      const leftR = rgba[leftOffset]
      const leftG = rgba[leftOffset + 1]
      const rightR = rgba[rightOffset]
      const rightG = rgba[rightOffset + 1]
      const rightB = rgba[rightOffset + 2]
      if (leftR <= leftG + 80) {
        throw new Error(`active BatchedMesh geometry should render red, got red=${leftR} green=${leftG}`)
      }
      if (rightR > 8 || rightG > 8 || rightB > 8) {
        throw new Error(`deleted BatchedMesh geometry should remain black, got rgb(${rightR}, ${rightG}, ${rightB})`)
      }
    },
  }
}

export function batchedMeshOptimizedRangeCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const source = new THREE.PlaneGeometry(0.45, 0.8)
  const batch = new THREE.BatchedMesh(
    3,
    source.getAttribute('position').count * 3,
    source.index.count * 3,
    new THREE.MeshBasicMaterial({ color: 0xffffff }),
  )
  const leftGeometryId = batch.addGeometry(source)
  const middleGeometryId = batch.addGeometry(source.clone())
  const rightGeometryId = batch.addGeometry(source.clone())
  const left = batch.addInstance(leftGeometryId)
  const middle = batch.addInstance(middleGeometryId)
  const right = batch.addInstance(rightGeometryId)
  batch.setMatrixAt(left, new THREE.Matrix4().makeTranslation(-0.55, 0, 0))
  batch.setMatrixAt(middle, new THREE.Matrix4())
  batch.setMatrixAt(right, new THREE.Matrix4().makeTranslation(0.55, 0, 0))
  batch.setColorAt(left, new THREE.Color(1, 0, 0))
  batch.setColorAt(middle, new THREE.Color(0, 1, 0))
  batch.setColorAt(right, new THREE.Color(0, 0, 1))
  batch.frustumCulled = false
  batch.perObjectFrustumCulled = false
  batch.sortObjects = false

  const rightRangeBefore = batch.getGeometryRangeAt(rightGeometryId, {})
  batch.deleteGeometry(middleGeometryId)
  batch.optimize()
  const rightRangeAfter = batch.getGeometryRangeAt(rightGeometryId, {})
  scene.add(batch)

  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 1.2, -1.2, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  return {
    name: 'batched-mesh-optimized-ranges',
    scene,
    camera,
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.03,
    validate(rgba, { width }) {
      if (!(rightRangeAfter.start < rightRangeBefore.start && rightRangeAfter.count === rightRangeBefore.count)) {
        throw new Error(`BatchedMesh optimize should repack the right geometry into the deleted range, before=${JSON.stringify(rightRangeBefore)} after=${JSON.stringify(rightRangeAfter)}`)
      }

      const leftMean = meanRegion(rgba, width, 20, 42, 30, 54)
      const centerMean = meanRegion(rgba, width, 43, 42, 53, 54)
      const rightMean = meanRegion(rgba, width, 66, 42, 76, 54)
      if (!(leftMean.r > leftMean.g + 80 && leftMean.r > leftMean.b + 80)) {
        throw new Error(`left optimized BatchedMesh geometry should remain red, got ${JSON.stringify(leftMean)}`)
      }
      if (!(centerMean.r < 5 && centerMean.g < 5 && centerMean.b < 5)) {
        throw new Error(`deleted optimized BatchedMesh geometry should leave the center empty, got ${JSON.stringify(centerMean)}`)
      }
      if (!(rightMean.b > rightMean.r + 80 && rightMean.b > rightMean.g + 80)) {
        throw new Error(`repacked BatchedMesh geometry should render blue on the right, got ${JSON.stringify(rightMean)}`)
      }
    },
  }
}

export function batchedMeshIndexedGroupsCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const source = new THREE.PlaneGeometry(0.45, 0.45)
  const batch = new THREE.BatchedMesh(
    2,
    source.getAttribute('position').count * 2,
    source.index.count * 2,
    [
      new THREE.MeshBasicMaterial({ color: 0xff0000 }),
      new THREE.MeshBasicMaterial({ color: 0x00ff00 }),
    ],
  )
  const leftGeometryId = batch.addGeometry(source)
  const rightGeometryId = batch.addGeometry(source.clone())
  const left = batch.addInstance(leftGeometryId)
  const right = batch.addInstance(rightGeometryId)
  batch.setMatrixAt(left, new THREE.Matrix4().makeTranslation(-0.55, 0, 0))
  batch.setMatrixAt(right, new THREE.Matrix4().makeTranslation(0.55, 0, 0))
  batch.perObjectFrustumCulled = false

  const leftRange = batch.getGeometryRangeAt(leftGeometryId, {})
  const rightRange = batch.getGeometryRangeAt(rightGeometryId, {})
  batch.geometry.clearGroups()
  batch.geometry.addGroup(leftRange.start, leftRange.count, 0)
  batch.geometry.addGroup(rightRange.start, rightRange.count, 1)
  scene.add(batch)

  const camera = new THREE.OrthographicCamera(-1.2, 1.2, 1.2, -1.2, 0.01, 10)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  return {
    name: 'batched-mesh-indexed-groups',
    scene,
    camera,
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.025,
    validate(rgba, { width }) {
      const leftMean = meanRegion(rgba, width, 20, 42, 30, 54)
      const rightMean = meanRegion(rgba, width, 66, 42, 76, 54)
      if (!(leftMean.r > leftMean.g + 80 && leftMean.r > leftMean.b + 80)) {
        throw new Error(`indexed BatchedMesh left group should use the red material, got ${JSON.stringify(leftMean)}`)
      }
      if (!(rightMean.g > rightMean.r + 80 && rightMean.g > rightMean.b + 80)) {
        throw new Error(`indexed BatchedMesh right group should use the green material, got ${JSON.stringify(rightMean)}`)
      }
    },
  }
}
