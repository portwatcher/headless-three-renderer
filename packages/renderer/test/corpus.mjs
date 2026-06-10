import * as THREE from 'three'

export const CORPUS_RENDER_SIZE = 96

export function createSceneCorpus() {
  return [
    transparentLayerCorpus(),
    skinnedMorphCorpus(),
    physicalIblShadowCorpus(),
    instancedLinesPointsCorpus(),
    lodAndGroupsCorpus(),
    pathologicalGeometryCorpus(),
  ]
}

function makeCamera(position = [2.2, 1.6, 3.1], target = [0, 0, 0]) {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(position[0], position[1], position[2])
  camera.lookAt(target[0], target[1], target[2])
  return camera
}

function addBasicLights(scene) {
  scene.add(new THREE.AmbientLight(0xffffff, 0.25))
  const dir = new THREE.DirectionalLight(0xffffff, 1.2)
  dir.position.set(3, 5, 2)
  dir.target.position.set(0, 0, 0)
  scene.add(dir)
  scene.add(dir.target)
}

function solidTexture(r, g, b, a = 255) {
  const texture = new THREE.DataTexture(new Uint8Array([r, g, b, a]), 1, 1, THREE.RGBAFormat)
  texture.needsUpdate = true
  return texture
}

function environmentTexture() {
  const data = new Uint8Array([
    255, 255, 255, 255,
    64, 128, 255, 255,
    255, 180, 96, 255,
    16, 24, 40, 255,
  ])
  const texture = new THREE.DataTexture(data, 2, 2, THREE.RGBAFormat)
  texture.needsUpdate = true
  return texture
}

function transparentLayerCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.08, 0.08, 0.1)

  const back = new THREE.Mesh(
    new THREE.PlaneGeometry(1.6, 1.6),
    new THREE.MeshBasicMaterial({ color: 0xff5522, transparent: true, opacity: 0.65 }),
  )
  back.position.z = -0.04
  back.renderOrder = 1

  const front = new THREE.Mesh(
    new THREE.PlaneGeometry(1.2, 1.2),
    new THREE.MeshBasicMaterial({ color: 0x2266ff, transparent: true, opacity: 0.55 }),
  )
  front.position.z = 0.04
  front.renderOrder = 2

  scene.add(back, front)
  return {
    name: 'transparent-layer-stack',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [20, 20, 26],
  }
}

function skinnedMorphCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.05, 0.06, 0.08)
  addBasicLights(scene)

  const geometry = new THREE.PlaneGeometry(1, 1, 1, 1)
  const count = geometry.getAttribute('position').count
  geometry.setAttribute('skinIndex', new THREE.BufferAttribute(new Uint16Array(count * 4), 4))
  const skinWeights = new Float32Array(count * 4)
  for (let i = 0; i < count; i += 1) {
    skinWeights[i * 4] = 1
  }
  geometry.setAttribute('skinWeight', new THREE.BufferAttribute(skinWeights, 4))
  geometry.morphTargetsRelative = true
  geometry.morphAttributes.position = [
    new THREE.BufferAttribute(new Float32Array([
      0, 0, 0,
      0.15, 0, 0,
      0, 0.2, 0,
      0.15, 0.2, 0,
    ]), 3),
  ]

  const material = new THREE.MeshStandardMaterial({ color: 0x77ccff, roughness: 0.55, metalness: 0.05 })
  const mesh = new THREE.SkinnedMesh(geometry, material)
  const bone = new THREE.Bone()
  mesh.add(bone)
  const skeleton = new THREE.Skeleton([bone])
  mesh.bind(skeleton)
  mesh.morphTargetInfluences = [0.6]
  bone.position.set(0.12, 0.05, 0)
  bone.updateMatrixWorld(true)
  mesh.rotation.y = -0.25
  scene.add(mesh)

  return {
    name: 'skinned-morphed-plane',
    scene,
    camera: makeCamera([0.2, 0.1, 2.5]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [13, 15, 20],
  }
}

function physicalIblShadowCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.04, 0.04, 0.05)
  scene.environment = environmentTexture()
  scene.environmentIntensity = 1.6
  scene.add(new THREE.AmbientLight(0xffffff, 0.15))

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(4, 4),
    new THREE.ShadowMaterial({ opacity: 0.65 }),
  )
  ground.rotation.x = -Math.PI / 2
  ground.position.y = -0.65
  ground.receiveShadow = true
  scene.add(ground)

  const sphere = new THREE.Mesh(
    new THREE.SphereGeometry(0.7, 24, 16),
    new THREE.MeshPhysicalMaterial({
      color: 0xffffff,
      metalness: 0.35,
      roughness: 0.22,
      clearcoat: 0.5,
      transmission: 0.15,
      thickness: 0.2,
      ior: 1.35,
    }),
  )
  sphere.castShadow = true
  sphere.receiveShadow = true
  scene.add(sphere)

  const light = new THREE.DirectionalLight(0xffffff, 2)
  light.position.set(3, 5, 2)
  light.target.position.set(0, 0, 0)
  light.castShadow = true
  light.shadow.mapSize.set(256, 256)
  light.shadow.camera.left = -3
  light.shadow.camera.right = 3
  light.shadow.camera.top = 3
  light.shadow.camera.bottom = -3
  light.shadow.camera.near = 0.1
  light.shadow.camera.far = 12
  scene.add(light, light.target)

  return {
    name: 'physical-ibl-shadow',
    scene,
    camera: makeCamera([2.2, 1.4, 3.2]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [10, 10, 13],
  }
}

function instancedLinesPointsCorpus() {
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
  }
}

function lodAndGroupsCorpus() {
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
    background: [20, 20, 20],
  }
}

function pathologicalGeometryCorpus() {
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
    background: [13, 13, 13],
  }
}
