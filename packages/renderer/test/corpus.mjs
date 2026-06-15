import * as THREE from 'three'

export const CORPUS_RENDER_SIZE = 96

export function createSceneCorpus() {
  return [
    transparentLayerCorpus(),
    customSortGroupCorpus(),
    skinnedMorphCorpus(),
    avatarLikeCorpus(),
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

function gradientTexture() {
  const texture = new THREE.DataTexture(new Uint8Array([
    88, 88, 120, 255,
    255, 226, 178, 255,
  ]), 2, 1, THREE.RGBAFormat)
  texture.magFilter = THREE.NearestFilter
  texture.minFilter = THREE.NearestFilter
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

function customSortGroupCorpus() {
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.9, -0.9, 0,
    0.9, -0.9, 0,
    0.9, 0.9, 0,
    -0.9, -0.9, 0,
    0.9, 0.9, 0,
    -0.9, 0.9, 0,
    -0.9, -0.9, 0,
    0.9, -0.9, 0,
    0.9, 0.9, 0,
    -0.9, -0.9, 0,
    0.9, 0.9, 0,
    -0.9, 0.9, 0,
  ]), 3))
  geometry.addGroup(0, 6, 0)
  geometry.addGroup(6, 6, 1)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  scene.add(new THREE.Mesh(geometry, [
    new THREE.MeshBasicMaterial({ color: 0xff3344, depthTest: false }),
    new THREE.MeshBasicMaterial({ color: 0x2266ff, depthTest: false }),
  ]))

  return {
    name: 'custom-opaque-sort-group-items',
    scene,
    camera: makeCamera([0, 0, 3]),
    options: {
      width: CORPUS_RENDER_SIZE,
      height: CORPUS_RENDER_SIZE,
      format: 'rgba',
      opaqueSort: (a, b) => b.group.materialIndex - a.group.materialIndex,
    },
    background: [0, 0, 0],
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

function avatarLikeCorpus() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.06, 0.07, 0.1)
  scene.environment = environmentTexture()
  scene.environmentIntensity = 0.8
  scene.fog = new THREE.Fog(0x111827, 3.5, 7)
  scene.add(new THREE.HemisphereLight(0xbfd7ff, 0x443322, 0.6))

  const key = new THREE.DirectionalLight(0xffffff, 1.1)
  key.position.set(2.5, 3.5, 2)
  key.target.position.set(0, 0.45, 0)
  scene.add(key, key.target)

  const bodyGeometry = new THREE.BoxGeometry(0.58, 1.24, 0.28, 1, 2, 1)
  const position = bodyGeometry.getAttribute('position')
  const vertexCount = position.count
  const skinIndex = new Uint16Array(vertexCount * 4)
  const skinWeight = new Float32Array(vertexCount * 4)
  const morph = new Float32Array(vertexCount * 3)
  for (let i = 0; i < vertexCount; i += 1) {
    const y = position.getY(i)
    const topWeight = Math.max(0, Math.min(1, (y + 0.15) / 0.9))
    skinIndex[i * 4] = 0
    skinIndex[i * 4 + 1] = 1
    skinWeight[i * 4] = 1 - topWeight
    skinWeight[i * 4 + 1] = topWeight
    if (y > 0.15) {
      morph[i * 3] = position.getX(i) * 0.08
      morph[i * 3 + 1] = 0.04
    }
  }
  bodyGeometry.setAttribute('skinIndex', new THREE.BufferAttribute(skinIndex, 4))
  bodyGeometry.setAttribute('skinWeight', new THREE.BufferAttribute(skinWeight, 4))
  bodyGeometry.morphTargetsRelative = true
  bodyGeometry.morphAttributes.position = [new THREE.BufferAttribute(morph, 3)]

  const body = new THREE.SkinnedMesh(bodyGeometry, new THREE.MeshToonMaterial({
    color: 0x8fc7ff,
    gradientMap: gradientTexture(),
  }))
  const hips = new THREE.Bone()
  hips.name = 'hips'
  hips.position.y = -0.55
  const chest = new THREE.Bone()
  chest.name = 'chest'
  chest.position.y = 0.85
  chest.rotation.z = -0.12
  hips.add(chest)
  body.add(hips)
  body.bind(new THREE.Skeleton([hips, chest]))
  body.morphTargetInfluences = [0.55]
  body.rotation.y = -0.25
  scene.add(body)

  const head = new THREE.Mesh(
    new THREE.SphereGeometry(0.34, 20, 12),
    new THREE.MeshPhongMaterial({
      color: 0xffd8b8,
      specular: 0x222222,
      shininess: 24,
    }),
  )
  head.position.set(0, 0.88, 0.02)
  head.rotation.y = -0.25
  scene.add(head)

  const hair = new THREE.Mesh(
    new THREE.SphereGeometry(0.38, 16, 10, 0, Math.PI * 2, 0, Math.PI * 0.62),
    new THREE.MeshBasicMaterial({
      color: 0x2f2448,
      transparent: true,
      opacity: 0.78,
      side: THREE.DoubleSide,
      alphaHash: true,
    }),
  )
  hair.position.set(0, 0.98, -0.02)
  hair.rotation.y = -0.25
  scene.add(hair)

  const eyeGeometry = new THREE.BufferGeometry()
  eyeGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -0.12, 0.92, 0.34,
    0.12, 0.92, 0.34,
  ]), 3))
  scene.add(new THREE.Points(eyeGeometry, new THREE.PointsMaterial({
    color: 0x102033,
    size: 5,
    sizeAttenuation: false,
  })))

  const outline = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(0.66, 1.32, 0.34)),
    new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.35 }),
  )
  outline.position.y = 0.02
  outline.rotation.y = -0.25
  scene.add(outline)

  return {
    name: 'avatar-like-skinned-toon',
    scene,
    camera: makeCamera([0.95, 0.75, 3.2], [0, 0.25, 0]),
    options: { width: CORPUS_RENDER_SIZE, height: CORPUS_RENDER_SIZE, format: 'rgba' },
    background: [15, 18, 26],
    backgroundTolerance: 8,
    minNonBackgroundRatio: 0.035,
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
