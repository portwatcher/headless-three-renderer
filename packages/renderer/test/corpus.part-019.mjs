import * as THREE from 'three'
import { CORPUS_RENDER_SIZE, addBasicLights, countRegionPixels, makeCamera, meanRegion, pixelAt, solidTexture } from './corpus.part-001.mjs'
export function batchedMeshCustomSortCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const nearGeometry = new THREE.PlaneGeometry(1.5, 1.5)
  nearGeometry.translate(0, 0, 0.35)
  const farGeometry = new THREE.PlaneGeometry(1.5, 1.5)
  farGeometry.translate(0, 0, -0.35)

  const batch = new THREE.BatchedMesh(
    2,
    nearGeometry.getAttribute('position').count + farGeometry.getAttribute('position').count,
    nearGeometry.index.count + farGeometry.index.count,
    new THREE.MeshBasicMaterial({
      color: 0xffffff,
      depthWrite: false,
      transparent: true,
    }),
  )
  const nearGeometryId = batch.addGeometry(nearGeometry)
  const farGeometryId = batch.addGeometry(farGeometry)
  const near = batch.addInstance(nearGeometryId)
  const far = batch.addInstance(farGeometryId)
  batch.setColorAt(near, new THREE.Color(1, 0, 0))
  batch.setColorAt(far, new THREE.Color(0, 0, 1))
  batch.setCustomSort((list) => {
    list.sort((a, b) => a.index - b.index)
  })
  scene.add(batch)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(0, 0, 3)
  camera.lookAt(0, 0, 0)

  return {
    name: 'batched-mesh-custom-sort',
    scene,
    camera,
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.1,
    validate(rgba, { width, height }) {
      const x = Math.floor(width / 2)
      const y = Math.floor(height / 2)
      const offset = (y * width + x) * 4
      const r = rgba[offset]
      const g = rgba[offset + 1]
      const b = rgba[offset + 2]
      if (b <= r + 80 || b <= g + 80) {
        throw new Error(`batched customSort should draw the blue instance last, got rgb(${r}, ${g}, ${b})`)
      }
    },
  }
}

export function lodAndGroupsCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.08, 0.08, 0.08)
  addBasicLights(scene)

  const group = new THREE.Group()
  group.renderOrder = 2
  group.add(new THREE.Mesh(
    new THREE.BoxGeometry(0.65, 0.65, 0.65),
    [
      new THREE.MeshLambertMaterial({ color: 0xff4444 }),
      new THREE.MeshLambertMaterial({ color: 0x44ff44 }),
      new THREE.MeshLambertMaterial({ color: 0x4444ff }),
      new THREE.MeshLambertMaterial({ color: 0xffff44 }),
      new THREE.MeshLambertMaterial({ color: 0xff44ff }),
      new THREE.MeshLambertMaterial({ color: 0x44ffff }),
    ],
  ))
  group.position.x = -0.45
  scene.add(group)

  const lod = new THREE.LOD()
  lod.position.x = 0.65
  lod.addLevel(
    new THREE.Mesh(new THREE.SphereGeometry(0.32, 16, 12), new THREE.MeshBasicMaterial({ color: 0x00aaff })),
    0,
  )
  lod.addLevel(
    new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.5, 0.5), new THREE.MeshBasicMaterial({ color: 0xffaa00 })),
    4,
  )
  scene.add(lod)

  return {
    name: 'lod-groups-material-array',
    scene,
    camera: makeCamera([1.4, 1.2, 3.2]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [80, 80, 80],
    validate(rgba, { width }) {
      const group = meanRegion(rgba, width, 16, 36, 36, 60)
      const lod = meanRegion(rgba, width, 60, 36, 80, 60)
      if (!(group.r > 80 && group.b > 80 && group.r > group.g + 70 && group.b > group.g + 80 && lod.b > lod.r + 95 && lod.g > lod.r + 65)) {
        throw new Error(`LOD/groups corpus should render the material-array group and near LOD sphere, got group=${JSON.stringify(group)} lod=${JSON.stringify(lod)}`)
      }
    },
  }
}

export function lodZoomCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)

  const lod = new THREE.LOD()
  lod.addLevel(
    new THREE.Mesh(new THREE.SphereGeometry(0.48, 24, 16), new THREE.MeshBasicMaterial({ color: 0xff0000 })),
    0,
  )
  lod.addLevel(
    new THREE.Mesh(new THREE.BoxGeometry(0.75, 0.75, 0.75), new THREE.MeshBasicMaterial({ color: 0x0000ff })),
    4,
  )
  scene.add(lod)

  const camera = makeCamera([0, 0, 6])
  camera.zoom = 2
  camera.updateProjectionMatrix()

  return {
    name: 'lod-zoom-selection',
    scene,
    camera,
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [0, 0, 0],
    minNonBackgroundRatio: 0.02,
    validate(rgba, { width, height }) {
      const x = Math.floor(width / 2)
      const y = Math.floor(height / 2)
      const offset = (y * width + x) * 4
      const r = rgba[offset]
      const b = rgba[offset + 2]
      if (r <= b + 80) {
        throw new Error(`zoomed LOD corpus should render the red near level, got red=${r} blue=${b}`)
      }
    },
  }
}

export function pathologicalGeometryCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.05, 0.05, 0.05)

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.8, -0.55, 0,
    0.8, -0.55, 0,
    -0.7, 0.55, 0,
    0.65, 0.5, 0.25,
  ]), 3))
  geometry.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([
    0, 0,
    1, 0,
    0, 1,
    1, 1,
  ]), 2))
  geometry.setAttribute('normal', new THREE.BufferAttribute(new Float32Array([
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
  ]), 3))
  geometry.setIndex([0, 1, 2, 1, 3, 2, 3, 3, 3])

  scene.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
    color: 0xffffff,
    map: solidTexture(64, 180, 255),
    side: THREE.DoubleSide,
  })))

  return {
    name: 'pathological-degenerate-geometry',
    scene,
    camera: makeCamera([0, 0, 2.6]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [63, 63, 63],
    validate(rgba, { width, height }) {
      const center = pixelAt(rgba, width, 48, 48)
      const corner = pixelAt(rgba, width, 4, 4)
      const geometryPixels = countRegionPixels(rgba, width, 0, 0, width, height, (r, g, b) => b > 150 && g > 140 && r > 100)
      if (!(center.b > center.r + 40 && center.g > center.r + 20 && corner.r === 63 && corner.g === 63 && corner.b === 63 && geometryPixels > 2500)) {
        throw new Error(`pathological geometry corpus should render the non-degenerate cyan triangles over background, got center=${JSON.stringify(center)} corner=${JSON.stringify(corner)} geometry=${geometryPixels}`)
      }
    },
  }
}
