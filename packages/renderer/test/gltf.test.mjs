import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as THREE from 'three'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'

const {
  Renderer,
  loadGltfFromFile,
  loadVrmAnimationFromFile,
  loadVrmFromFile,
} = pkg

const FIXTURE_DIR = fileURLToPath(new URL('./fixtures/', import.meta.url))
const SIMPLE_TRIANGLE = path.join(FIXTURE_DIR, 'simple-triangle.gltf')
const TEXTURED_QUAD = path.join(FIXTURE_DIR, 'textured-quad.gltf')
const VERTEX_COLOR_QUAD = path.join(FIXTURE_DIR, 'vertex-color-quad.gltf')
const MORPHED_TRIANGLE = path.join(FIXTURE_DIR, 'morphed-triangle.gltf')
const SKINNED_QUAD = path.join(FIXTURE_DIR, 'skinned-quad.gltf')
const SYNTHETIC_VRM = path.join(FIXTURE_DIR, 'synthetic-avatar.vrm')
const SYNTHETIC_VRMA = path.join(FIXTURE_DIR, 'synthetic-animation.vrma')
const SAMPLE_ASSET_BOX_ANIMATED = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'BoxAnimated', 'glTF', 'BoxAnimated.gltf')
const SAMPLE_ASSET_BOX = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'Box', 'glTF', 'Box.gltf')
const SAMPLE_ASSET_BOX_INTERLEAVED = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'BoxInterleaved', 'glTF', 'BoxInterleaved.gltf')
const SAMPLE_ASSET_BOX_VERTEX_COLORS = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'BoxVertexColors', 'glTF', 'BoxVertexColors.gltf')
const SAMPLE_ASSET_CAMERAS = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'Cameras', 'glTF', 'Cameras.gltf')
const SAMPLE_ASSET_INTERPOLATION_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'InterpolationTest', 'glTF', 'InterpolationTest.gltf')
const SAMPLE_ASSET_MESH_PRIMITIVE_MODES = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'MeshPrimitiveModes', 'glTF', 'MeshPrimitiveModes.gltf')
const SAMPLE_ASSET_MULTIPLE_SCENES = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'MultipleScenes', 'glTF', 'MultipleScenes.gltf')
const SAMPLE_ASSET_SIMPLE_INSTANCING = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'SimpleInstancing', 'glTF', 'SimpleInstancing.gltf')
const SAMPLE_ASSET_SIMPLE_MORPH = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'SimpleMorph', 'glTF', 'SimpleMorph.gltf')
const SAMPLE_ASSET_SIMPLE_SKIN = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'SimpleSkin', 'glTF', 'SimpleSkin.gltf')
const SAMPLE_ASSET_SIMPLE_SPARSE_ACCESSOR = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'SimpleSparseAccessor', 'glTF', 'SimpleSparseAccessor.gltf')
const SAMPLE_ASSET_SIMPLE_TEXTURE = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'SimpleTexture', 'glTF', 'SimpleTexture.gltf')
const SAMPLE_ASSET_TEXTURE_COORDINATE_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'TextureCoordinateTest', 'glTF', 'TextureCoordinateTest.gltf')
const SAMPLE_ASSET_TEXTURE_TRANSFORM_TEST = path.join(FIXTURE_DIR, 'gltf-sample-assets', 'TextureTransformTest', 'glTF', 'TextureTransformTest.gltf')

test('committed glTF fixture loads through GLTFLoader and renders', async () => {
  let configured = false
  const gltf = await loadGltfFixture(SIMPLE_TRIANGLE, {
    configureLoader(loader) {
      configured = typeof loader.parse === 'function'
    },
  })
  assert.equal(configured, true, 'loadGltfFromFile should expose the loader before parsing')

  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'fixture should load a mesh')
  assert.equal(mesh.geometry.getAttribute('position').count, 3)
  assert.equal(mesh.material.isMeshStandardMaterial, true)

  const camera = gltf.cameras[0]
  assert.ok(camera, 'fixture should load a camera')
  camera.aspect = 1
  camera.updateProjectionMatrix()

  const scene = gltf.scene
  scene.add(new THREE.AmbientLight(0xffffff, 0.6))
  const light = new THREE.DirectionalLight(0xffffff, 1.4)
  light.position.set(2, 3, 4)
  scene.add(light)
  scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [0.02, 0.02, 0.03],
  })
  assert.equal(rgba.length, 96 * 96 * 4)
  assert.ok(nonBackgroundRatio(rgba, [5, 5, 8], 3) > 0.04, 'glTF triangle should render visible pixels')

  const mean = meanRgba(rgba)
  assert.ok(mean.b > mean.r, `loaded blue PBR material should contribute blue output (${mean.b} vs ${mean.r})`)
  assert.ok(mean.a > 240, `loaded glTF output should be opaque (${mean.a})`)
})

test('committed Khronos glTF Sample Assets Box fixture loads external buffer and renders', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_BOX)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'Khronos Box sample should load a mesh')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 24)
  assert.equal(mesh.geometry.index?.count, 36)
  assert.equal(mesh.material.name, 'Red')

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 10)
  camera.position.set(1.4, 1.1, 2.2)
  camera.lookAt(0, 0, 0)

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.55))
  const light = new THREE.DirectionalLight(0xffffff, 1.6)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.2, 'Khronos Box sample should render visible pixels')
  const center = meanRegion(rgba, 96, 96, 40, 40, 56, 56)
  assert.ok(center.r > center.b + 150 && center.r > center.g + 180, `Khronos Box sample should render a red cube (${center.r}, ${center.g}, ${center.b})`)
})

test('committed Khronos glTF Sample Assets BoxInterleaved fixture loads byteStride attributes', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_BOX_INTERLEAVED)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'Khronos BoxInterleaved sample should load a mesh')
  const position = mesh.geometry.getAttribute('position')
  const normal = mesh.geometry.getAttribute('normal')
  assert.equal(position?.count, 24)
  assert.equal(normal?.count, 24)
  assert.equal(mesh.geometry.index?.count, 36)
  assert.equal(position.isInterleavedBufferAttribute, true)
  assert.equal(normal.isInterleavedBufferAttribute, true)
  assert.equal(position.data.stride, 6)
  assert.deepEqual(vectorFromAttribute(position, 0), [-0.5, -0.5, 0.5])
  assert.deepEqual(vectorFromAttribute(normal, 0), [0, 0, 1])
  assert.equal(mesh.material.color.r, 0.800000011920929)
  assert.equal(mesh.material.color.g, 0)
  assert.equal(mesh.material.color.b, 0)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 10)
  camera.position.set(1.4, 1.1, 2.2)
  camera.lookAt(0, 0, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.55))
  const light = new THREE.DirectionalLight(0xffffff, 1.6)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.2, 'Khronos BoxInterleaved sample should render visible pixels')
  const center = meanRegion(rgba, 96, 96, 40, 40, 56, 56)
  assert.ok(center.r > center.b + 150 && center.r > center.g + 180, `BoxInterleaved sample should render a red cube (${center.r}, ${center.g}, ${center.b})`)
})

test('committed Khronos glTF Sample Assets Cameras fixture loads and renders imported cameras', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_CAMERAS)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'Khronos Cameras sample should load a mesh')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 4)
  assert.equal(mesh.geometry.index?.count, 6)
  assert.equal(gltf.cameras.length, 2)

  const [perspective, orthographic] = gltf.cameras
  assert.equal(perspective.isPerspectiveCamera, true)
  assert.equal(perspective.near, 0.01)
  assert.equal(perspective.far, 100)
  assert.equal(perspective.aspect, 1)
  assert.ok(Math.abs(perspective.fov - THREE.MathUtils.radToDeg(0.7)) < 1e-10, `perspective camera should preserve glTF yfov (${perspective.fov})`)
  assert.deepEqual(perspective.position.toArray(), [0.5, 0.5, 3])

  assert.equal(orthographic.isOrthographicCamera, true)
  assert.equal(orthographic.near, 0.01)
  assert.equal(orthographic.far, 100)
  assert.equal(orthographic.left, -1)
  assert.equal(orthographic.right, 1)
  assert.equal(orthographic.top, 1)
  assert.equal(orthographic.bottom, -1)
  assert.deepEqual(orthographic.position.toArray(), [0.5, 0.5, 3])

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1.0))
  gltf.scene.updateMatrixWorld(true)

  const renderer = new Renderer()
  const renderWithCamera = (camera) => {
    camera.updateMatrixWorld(true)
    return renderer.render(gltf.scene, camera, {
      width: 96,
      height: 96,
      format: 'rgba',
      background: [0, 0, 0],
      outputColorSpace: THREE.LinearSRGBColorSpace,
    })
  }

  const perspectiveRgba = renderWithCamera(perspective)
  const orthographicRgba = renderWithCamera(orthographic)

  assert.ok(nonBackgroundRatio(perspectiveRgba, [0, 0, 0], 3) > 0.1, 'Cameras sample should render through imported perspective camera')
  assert.ok(nonBackgroundRatio(orthographicRgba, [0, 0, 0], 3) > 0.15, 'Cameras sample should render through imported orthographic camera')
  const perspectiveCenter = meanRegion(perspectiveRgba, 96, 96, 24, 24, 72, 72)
  const orthographicCenter = meanRegion(orthographicRgba, 96, 96, 24, 24, 72, 72)
  assert.ok(perspectiveCenter.r > 80 && perspectiveCenter.g > 80 && perspectiveCenter.b > 80, `perspective camera should see the white mesh (${perspectiveCenter.r}, ${perspectiveCenter.g}, ${perspectiveCenter.b})`)
  assert.ok(orthographicCenter.r > 80 && orthographicCenter.g > 80 && orthographicCenter.b > 80, `orthographic camera should see the white mesh (${orthographicCenter.r}, ${orthographicCenter.g}, ${orthographicCenter.b})`)
})

test('committed Khronos glTF Sample Assets InterpolationTest fixture applies animation interpolation modes', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_INTERPOLATION_TEST)
  assert.deepEqual(gltf.animations.map((clip) => clip.name), [
    'Step Scale',
    'Linear Scale',
    'CubicSpline Scale',
    'Step Rotation',
    'CubicSpline Rotation',
    'Linear Rotation',
    'Step Translation',
    'CubicSpline Translation',
    'Linear Translation',
  ])

  const tracksByClip = new Map(gltf.animations.map((clip) => [clip.name, clip.tracks[0]]))
  assert.equal(tracksByClip.get('Step Scale')?.name, 'Cube.scale')
  assert.equal(tracksByClip.get('Step Scale')?.getInterpolation(), THREE.InterpolateDiscrete)
  assert.equal(tracksByClip.get('Linear Scale')?.getInterpolation(), THREE.InterpolateLinear)
  assert.equal(tracksByClip.get('CubicSpline Scale')?.getValueSize(), 9)
  assert.equal(tracksByClip.get('Step Rotation')?.name, 'Cube003.quaternion')
  assert.equal(tracksByClip.get('Linear Rotation')?.getInterpolation(), THREE.InterpolateLinear)
  assert.equal(tracksByClip.get('CubicSpline Rotation')?.getValueSize(), 12)
  assert.equal(tracksByClip.get('Step Translation')?.name, 'Cube006.position')
  assert.equal(tracksByClip.get('Linear Translation')?.getInterpolation(), THREE.InterpolateLinear)
  assert.equal(tracksByClip.get('CubicSpline Translation')?.getValueSize(), 9)

  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 10, 'InterpolationTest should load nine animated cubes plus one textured plane')
  const plane = gltf.scene.getObjectByName('Plane')
  assert.ok(Buffer.isBuffer(plane?.material?.map?.image), 'InterpolationTest external PNG should load as an encoded Buffer')

  const mixer = new THREE.AnimationMixer(gltf.scene)
  for (const clip of gltf.animations) mixer.clipAction(clip).play()
  mixer.setTime(0.25)
  gltf.scene.updateMatrixWorld(true)

  assert.ok(Math.abs(gltf.scene.getObjectByName('Cube').scale.x - 1) < 1e-6, 'STEP scale should hold the previous keyframe at t=0.25')
  assert.ok(Math.abs(gltf.scene.getObjectByName('Cube001').scale.x - 0.5) < 1e-6, 'LINEAR scale should interpolate halfway at t=0.25')
  assert.ok(Math.abs(gltf.scene.getObjectByName('Cube002').scale.x - 0.5) < 1e-6, 'CUBICSPLINE scale should interpolate halfway at t=0.25')
  assert.ok(Math.abs(gltf.scene.getObjectByName('Cube003').quaternion.z) < 1e-6, 'STEP rotation should hold the previous keyframe at t=0.25')
  assert.ok(Math.abs(gltf.scene.getObjectByName('Cube005').quaternion.z + 0.19509032) < 1e-5, 'LINEAR rotation should slerp at t=0.25')
  assert.ok(Math.abs(gltf.scene.getObjectByName('Cube006').position.y - 6.80000019) < 1e-5, 'STEP translation should hold the previous keyframe at t=0.25')
  assert.ok(Math.abs(gltf.scene.getObjectByName('Cube009').position.y - 8.80000019) < 1e-5, 'LINEAR translation should interpolate halfway at t=0.25')

  const camera = new THREE.OrthographicCamera(-6, 6, 10, -2.5, 0.01, 20)
  camera.position.set(0, 3.6, 10)
  camera.lookAt(0, 3.6, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1.0))
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 128,
    height: 128,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.1, 'InterpolationTest animated fixture should render visible geometry')
})

test('committed Khronos glTF Sample Assets BoxAnimated fixture applies transform animation', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_BOX_ANIMATED)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 2, 'Khronos BoxAnimated sample should load inner and outer meshes')
  assert.deepEqual(meshes.map((mesh) => mesh.material.name).sort(), ['inner', 'outer'])
  assert.equal(gltf.animations.length, 1)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 20)
  camera.position.set(1.7, 1.7, 4.4)
  camera.lookAt(0, 0.8, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.7))
  const light = new THREE.DirectionalLight(0xffffff, 1.2)
  light.position.set(3, 4, 5)
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const renderer = new Renderer()
  const renderBounds = () => nonBackgroundBounds(renderer.render(gltf.scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
  }), 96, 96, [0, 0, 0], 3)

  const base = renderBounds()
  const mixer = new THREE.AnimationMixer(gltf.scene)
  mixer.clipAction(gltf.animations[0]).play()
  mixer.setTime(1.25)
  gltf.scene.updateMatrixWorld(true)
  const animated = renderBounds()

  assert.ok(base.height > 25, `BoxAnimated base pose should render visible box bounds (${base.height})`)
  assert.ok(animated.height > base.height + 40, `BoxAnimated translation track should expand vertical bounds (${animated.height} vs ${base.height})`)
  assert.ok(animated.minY < base.minY - 40, `BoxAnimated translation track should move the animated mesh upward (${animated.minY} vs ${base.minY})`)
})

test('committed Khronos glTF Sample Assets BoxVertexColors fixture renders COLOR_0 gradients', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_BOX_VERTEX_COLORS)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'Khronos BoxVertexColors sample should load a mesh')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 24)
  assert.equal(mesh.geometry.getAttribute('color')?.count, 24)
  assert.equal(mesh.geometry.index?.count, 36)
  assert.equal(mesh.material.vertexColors, true)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 10)
  camera.position.set(1.4, 1.1, 2.2)
  camera.lookAt(0, 0, 0)

  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 1.0)
  light.position.set(2, 3, 4)
  gltf.scene.add(light)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.4, 'Khronos BoxVertexColors sample should render visible pixels')
  const topLeft = meanRegion(rgba, 96, 96, 24, 22, 36, 34)
  const bottomLeft = meanRegion(rgba, 96, 96, 24, 58, 36, 68)
  const bottomRight = meanRegion(rgba, 96, 96, 62, 58, 74, 68)
  assert.ok(topLeft.g > bottomLeft.g + 80, `vertex color gradient should make the upper-left face greener than lower-left (${topLeft.g} vs ${bottomLeft.g})`)
  assert.ok(bottomRight.r > bottomLeft.r + 80, `vertex color gradient should make the lower-right face redder than lower-left (${bottomRight.r} vs ${bottomLeft.r})`)
  assert.ok(bottomLeft.b > 170 && bottomRight.b > 170, `vertex color gradient should keep blue channel visible (${bottomLeft.b}, ${bottomRight.b})`)
})

test('committed Khronos glTF Sample Assets SimpleSkin fixture applies skin animation', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_SIMPLE_SKIN)
  const mesh = findFirst(gltf.scene, (object) => object.isSkinnedMesh === true)
  assert.ok(mesh, 'Khronos SimpleSkin sample should load a SkinnedMesh')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 10)
  assert.equal(mesh.geometry.getAttribute('skinIndex')?.count, 10)
  assert.equal(mesh.geometry.getAttribute('skinWeight')?.count, 10)
  assert.equal(mesh.geometry.index?.count, 24)
  assert.equal(mesh.skeleton.bones.length, 2)
  assert.equal(gltf.animations.length, 1)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 10)
  camera.position.set(0, 1, 4)
  camera.lookAt(0, 1, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1.0))
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const renderer = new Renderer()
  const renderBounds = () => nonBackgroundBounds(renderer.render(gltf.scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  }), 96, 96, [0, 0, 0], 3)

  const base = renderBounds()
  const mixer = new THREE.AnimationMixer(gltf.scene)
  mixer.clipAction(gltf.animations[0]).play()
  mixer.setTime(1)
  gltf.scene.updateMatrixWorld(true)
  const animated = renderBounds()

  assert.ok(base.height > 50, `SimpleSkin base pose should render a tall strip (${base.height})`)
  assert.ok(animated.width > base.width + 10, `SimpleSkin animation should widen the skinned mesh (${animated.width} vs ${base.width})`)
  assert.ok(animated.minY > base.minY + 10, `SimpleSkin animation should bend the top downward (${animated.minY} vs ${base.minY})`)
})

test('committed Khronos glTF Sample Assets SimpleMorph fixture applies morph weight animation', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_SIMPLE_MORPH)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'Khronos SimpleMorph sample should load a mesh')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 3)
  assert.equal(mesh.geometry.index?.count, 3)
  assert.equal(mesh.geometry.morphAttributes.position?.length, 2)
  assert.deepEqual(mesh.morphTargetInfluences, [0.5, 0.5])
  assert.equal(gltf.animations.length, 1)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 10)
  camera.position.set(0.5, 0.5, 3.2)
  camera.lookAt(0.45, 0.4, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1.0))
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const renderer = new Renderer()
  const renderBounds = () => nonBackgroundBounds(renderer.render(gltf.scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  }), 96, 96, [0, 0, 0], 3)

  mesh.morphTargetInfluences = [0, 0]
  gltf.scene.updateMatrixWorld(true)
  const base = renderBounds()

  const mixer = new THREE.AnimationMixer(gltf.scene)
  mixer.clipAction(gltf.animations[0]).play()
  mixer.setTime(2)
  gltf.scene.updateMatrixWorld(true)
  const animated = renderBounds()

  assert.ok(base.height > 10, `SimpleMorph base triangle should render visible bounds (${base.height})`)
  assert.ok(animated.height > base.height + 35, `SimpleMorph animation should expand rendered height (${animated.height} vs ${base.height})`)
  assert.ok(animated.minY < base.minY - 35, `SimpleMorph animation should lift the triangle top (${animated.minY} vs ${base.minY})`)
})

test('committed Khronos glTF Sample Assets SimpleSparseAccessor fixture applies sparse POSITION overrides', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_SIMPLE_SPARSE_ACCESSOR)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'Khronos SimpleSparseAccessor sample should load a mesh')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 14)
  assert.equal(mesh.geometry.index?.count, 36)

  const position = mesh.geometry.getAttribute('position')
  assert.deepEqual(vectorFromAttribute(position, 8), [1, 2, 0])
  assert.deepEqual(vectorFromAttribute(position, 10), [3, 3, 0])
  assert.deepEqual(vectorFromAttribute(position, 12), [5, 4, 0])

  const camera = new THREE.OrthographicCamera(-0.5, 6.5, 4.5, -0.5, 0.01, 10)
  camera.position.set(3, 2, 5)
  camera.lookAt(3, 2, 0)
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.05, 'SimpleSparseAccessor sample should render visible sparse geometry')
})

test('committed Khronos glTF Sample Assets SimpleInstancing fixture loads EXT_mesh_gpu_instancing', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_SIMPLE_INSTANCING)
  const mesh = findFirst(gltf.scene, (object) => object.isInstancedMesh === true)
  assert.ok(mesh, 'Khronos SimpleInstancing sample should load an InstancedMesh')
  assert.equal(mesh.count, 125)
  assert.equal(mesh.geometry.getAttribute('position')?.count, 24)
  assert.equal(mesh.geometry.getAttribute('normal')?.count, 24)
  assert.equal(mesh.geometry.index?.count, 36)
  assert.ok(mesh.instanceMatrix?.isInstancedBufferAttribute, 'EXT_mesh_gpu_instancing should populate instance matrices')
  assert.deepEqual(Array.from(mesh.instanceMatrix.array.slice(0, 16)), [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ])
  assert.deepEqual(Array.from(mesh.instanceMatrix.array.slice(124 * 16, 125 * 16)), [
    0, 2, 0, 0,
    0, 0, 2, 0,
    2, 0, 0, 0,
    10, 10, 10, 1,
  ])

  const camera = new THREE.OrthographicCamera(-1, 12, 12, -1, 0.01, 50)
  camera.position.set(6, 6, 20)
  camera.lookAt(6, 6, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const light = new THREE.DirectionalLight(0xffffff, 1.2)
  light.position.set(10, 12, 20)
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.2, 'SimpleInstancing sample should render visible instanced geometry')
})

test('committed Khronos glTF Sample Assets SimpleTexture fixture loads sampler state and renders mirrored texture repeats', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_SIMPLE_TEXTURE)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'Khronos SimpleTexture sample should load a mesh')
  assert.equal(mesh.geometry.getAttribute('position')?.count, 4)
  assert.equal(mesh.geometry.getAttribute('uv')?.count, 4)
  assert.equal(mesh.geometry.index?.count, 6)

  const texture = mesh.material.map
  assert.ok(texture?.isTexture, 'SimpleTexture sample should load a base color texture')
  assert.equal(Buffer.isBuffer(texture.image), true, 'SimpleTexture external PNG should load as an encoded Buffer')
  assert.equal(texture.wrapS, THREE.MirroredRepeatWrapping)
  assert.equal(texture.wrapT, THREE.MirroredRepeatWrapping)
  assert.equal(texture.magFilter, THREE.LinearFilter)
  assert.equal(texture.minFilter, THREE.LinearMipmapLinearFilter)
  assert.equal(texture.flipY, false)

  texture.repeat.set(2, 2)
  mesh.material = new THREE.MeshBasicMaterial({ map: texture })
  mesh.position.set(-0.5, -0.5, 0)

  const camera = new THREE.OrthographicCamera(-0.6, 0.6, 0.6, -0.6, 0.01, 10)
  camera.position.set(0, 0, 2)
  camera.lookAt(0, 0, 0)
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.6, 'SimpleTexture sample should render a visible repeated texture')
  const topLeft = meanRegion(rgba, 96, 96, 18, 18, 38, 38)
  const topRight = meanRegion(rgba, 96, 96, 58, 18, 78, 38)
  const bottomLeft = meanRegion(rgba, 96, 96, 18, 58, 38, 78)
  const bottomRight = meanRegion(rgba, 96, 96, 58, 58, 78, 78)

  for (const [label, sample] of [
    ['top-right', topRight],
    ['bottom-left', bottomLeft],
    ['bottom-right', bottomRight],
  ]) {
    assert.ok(
      Math.abs(sample.r - topLeft.r) < 8 &&
        Math.abs(sample.g - topLeft.g) < 8 &&
        Math.abs(sample.b - topLeft.b) < 8,
      `mirrored-repeat ${label} sample should match top-left (${topLeft.r}, ${topLeft.g}, ${topLeft.b}) vs (${sample.r}, ${sample.g}, ${sample.b})`,
    )
  }

  const center = meanRegion(rgba, 96, 96, 38, 38, 58, 58)
  assert.ok(center.r > topLeft.r + 80 && center.g > topLeft.g + 80, `repeated texture center should sample brighter texels (${center.r}, ${center.g}, ${center.b})`)
})

test('committed Khronos glTF Sample Assets TextureCoordinateTest fixture renders external PNG UV quadrants', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_TEXTURE_COORDINATE_TEST)
  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 5, 'Khronos TextureCoordinateTest sample should load four textured planes plus a back plane')
  assert.equal(meshes.filter((mesh) => mesh.material.map?.isTexture === true).length, 4)
  assert.ok(
    meshes.filter((mesh) => mesh.material.map?.isTexture === true).every((mesh) => Buffer.isBuffer(mesh.material.map.image)),
    'external PNG textures should be exposed as encoded Buffers',
  )

  const camera = new THREE.OrthographicCamera(-1.45, 1.45, 1.45, -1.45, 0.01, 10)
  camera.position.set(0, 0, 3)
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

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.4, 'TextureCoordinateTest sample should render visible textured planes')
  const topLeft = meanRegion(rgba, 96, 96, 18, 18, 38, 38)
  const topRight = meanRegion(rgba, 96, 96, 58, 18, 78, 38)
  const bottomLeft = meanRegion(rgba, 96, 96, 18, 58, 38, 78)
  const bottomRight = meanRegion(rgba, 96, 96, 58, 58, 78, 78)
  assert.ok(topLeft.r > 130 && topLeft.g > 120 && topLeft.b < 50, `top-left UV quadrant should sample yellow texels (${topLeft.r}, ${topLeft.g}, ${topLeft.b})`)
  assert.ok(topRight.r > topRight.g + 140 && topRight.r > topRight.b + 140, `top-right UV quadrant should sample red texels (${topRight.r}, ${topRight.g}, ${topRight.b})`)
  assert.ok(bottomLeft.b > bottomLeft.r + 120 && bottomLeft.b > bottomLeft.g + 120, `bottom-left UV quadrant should sample blue texels (${bottomLeft.r}, ${bottomLeft.g}, ${bottomLeft.b})`)
  assert.ok(bottomRight.g > bottomRight.r + 100 && bottomRight.g > bottomRight.b + 100, `bottom-right UV quadrant should sample green texels (${bottomRight.r}, ${bottomRight.g}, ${bottomRight.b})`)
})

test('committed Khronos glTF Sample Assets TextureTransformTest fixture loads KHR_texture_transform', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_TEXTURE_TRANSFORM_TEST)
  const offsetU = gltf.scene.getObjectByName('Offset_U')
  const offsetV = gltf.scene.getObjectByName('Offset_V')
  const offsetUv = gltf.scene.getObjectByName('Offset_UV')
  const rotation = gltf.scene.getObjectByName('Rotation')
  const scale = gltf.scene.getObjectByName('Scale')
  const all = gltf.scene.getObjectByName('All')
  assert.ok(offsetU?.isMesh, 'TextureTransformTest should load Offset_U mesh')
  assert.ok(offsetV?.isMesh, 'TextureTransformTest should load Offset_V mesh')
  assert.ok(offsetUv?.isMesh, 'TextureTransformTest should load Offset_UV mesh')
  assert.ok(rotation?.isMesh, 'TextureTransformTest should load Rotation mesh')
  assert.ok(scale?.isMesh, 'TextureTransformTest should load Scale mesh')
  assert.ok(all?.isMesh, 'TextureTransformTest should load All mesh')

  assert.equal(Buffer.isBuffer(offsetU.material.map.image), true)
  assert.deepEqual(offsetU.material.map.offset.toArray(), [0.5, 0])
  assert.deepEqual(offsetV.material.map.offset.toArray(), [0, 0.5])
  assert.deepEqual(offsetUv.material.map.offset.toArray(), [0.5, 0.5])
  assert.ok(Math.abs(rotation.material.map.rotation - 0.39269908169872414) < 1e-12)
  assert.deepEqual(scale.material.map.repeat.toArray(), [1.5, 1.5])
  assert.deepEqual(all.material.map.offset.toArray(), [-0.2, -0.1])
  assert.deepEqual(all.material.map.repeat.toArray(), [1.5, 1.5])
  assert.ok(Math.abs(all.material.map.rotation - 0.3) < 1e-12)

  const meshes = []
  gltf.scene.traverse((object) => {
    if (object.isMesh === true) meshes.push(object)
  })
  assert.equal(meshes.length, 12, 'TextureTransformTest should load transformed samples and reference badges')
  assert.ok(
    meshes.every((mesh) => Buffer.isBuffer(mesh.material.map?.image)),
    'TextureTransformTest external PNG textures should load as encoded Buffers',
  )

  const camera = new THREE.OrthographicCamera(-1.8, 1.8, 1.2, -1.2, 0.01, 20)
  camera.position.set(0, 0, 10)
  camera.lookAt(0, 0, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1.0))
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 144,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.6, 'TextureTransformTest should render visible transformed texture samples')
  const topLeft = meanRegion(rgba, 144, 96, 18, 16, 38, 36)
  const topCenter = meanRegion(rgba, 144, 96, 62, 16, 82, 36)
  const topRight = meanRegion(rgba, 144, 96, 106, 16, 126, 36)
  assert.ok(topLeft.g > topLeft.r + 60 && topLeft.g > topLeft.b + 60, `offset-U sample should expose green-dominant texels (${topLeft.r}, ${topLeft.g}, ${topLeft.b})`)
  assert.ok(topCenter.b > topCenter.r + 80 && topCenter.b > topCenter.g + 80, `offset-V sample should expose blue-dominant texels (${topCenter.r}, ${topCenter.g}, ${topCenter.b})`)
  assert.ok(topRight.g > topRight.r + 60 && topRight.b > topRight.r + 60, `offset-UV sample should expose cyan texels (${topRight.r}, ${topRight.g}, ${topRight.b})`)
})

test('committed Khronos glTF Sample Assets MeshPrimitiveModes fixture loads and renders primitive modes', async () => {
  const gltf = await loadGltfFixture(SAMPLE_ASSET_MESH_PRIMITIVE_MODES)
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

  assert.deepEqual(renderables.map((object) => ({
    name: object.name,
    type: object.type,
    index: object.geometry.index?.count,
    positions: object.geometry.getAttribute('position')?.count,
  })), [
    { name: 'mesh_with_POINTS', type: 'Points', index: 7, positions: 7 },
    { name: 'mesh_with_LINES', type: 'LineSegments', index: 12, positions: 7 },
    { name: 'mesh_with_LINE_LOOP', type: 'LineLoop', index: 7, positions: 7 },
    { name: 'mesh_with_LINE_STRIP', type: 'Line', index: 7, positions: 7 },
    { name: 'mesh_with_TRIANGLES', type: 'Mesh', index: 18, positions: 7 },
    { name: 'mesh_with_GL_TRIANGLE_STRIP', type: 'Mesh', index: 12, positions: 7 },
    { name: 'mesh_with_GL_TRIANGLE_FAN', type: 'Mesh', index: 18, positions: 7 },
  ])

  for (const object of renderables) {
    if (object.material?.color) object.material.color.set(0xffffff)
    if (object.isPoints === true) object.material.size = 10
    if (object.isLine === true || object.isLineSegments === true || object.isLineLoop === true) {
      object.material.linewidth = 4
    }
  }

  const camera = new THREE.OrthographicCamera(-4, 4, 4, -4, 0.01, 10)
  camera.position.set(0, 0, 4)
  camera.lookAt(0, 0, 0)
  gltf.scene.add(new THREE.AmbientLight(0xffffff, 1.0))
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 128,
    height: 128,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })

  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.2, 'MeshPrimitiveModes sample should render visible points, lines, and meshes')
  const points = meanRegion(rgba, 128, 128, 56, 8, 72, 24)
  const lineLoop = meanRegion(rgba, 128, 128, 56, 56, 72, 72)
  const triangleFan = meanRegion(rgba, 128, 128, 88, 104, 104, 120)
  assert.ok(points.r > 60 && points.g > 60 && points.b > 60, `POINTS primitive should render visible pixels (${points.r}, ${points.g}, ${points.b})`)
  assert.ok(lineLoop.r > 40 && lineLoop.g > 40 && lineLoop.b > 40, `LINE_LOOP primitive should render visible pixels (${lineLoop.r}, ${lineLoop.g}, ${lineLoop.b})`)
  assert.ok(triangleFan.r > 120 && triangleFan.g > 120 && triangleFan.b > 120, `TRIANGLE_FAN primitive should render visible pixels (${triangleFan.r}, ${triangleFan.g}, ${triangleFan.b})`)
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

test('committed textured glTF fixture loads data URI image and renders texture', async () => {
  const gltf = await loadGltfFixture(TEXTURED_QUAD)

  assertTexturedQuadLoadsEncodedMap(gltf, 'textured fixture')
  assertTexturedQuadRendersTexture(gltf, 'textured quad')
})

test('loadGltfFromFile loads helper-normalized GLB bufferView images', async () => {
  const source = JSON.parse(await readFile(TEXTURED_QUAD, 'utf8'))
  const glbBytes = buildTexturedQuadGlb(source)
  const tmp = await mkdtemp(path.join(os.tmpdir(), 'headless-three-glb-image-'))
  try {
    const modelPath = path.join(tmp, 'buffer-view-image.glb')
    await writeFile(modelPath, glbBytes)

    const gltf = await loadGltfFixture(modelPath)
    assertTexturedQuadLoadsEncodedMap(gltf, 'GLB bufferView-image fixture')
    assertTexturedQuadRendersTexture(gltf, 'GLB bufferView-image quad')
  } finally {
    await rm(tmp, { recursive: true, force: true })
  }
})

test('loadGltfFromFile rejects compressed GLB bufferView images with pre-decode guidance', async () => {
  const source = JSON.parse(await readFile(TEXTURED_QUAD, 'utf8'))
  source.images[0].mimeType = 'image/ktx2'
  const glbBytes = buildTexturedQuadGlb(source)
  const tmp = await mkdtemp(path.join(os.tmpdir(), 'headless-three-glb-compressed-image-'))
  try {
    const modelPath = path.join(tmp, 'compressed-buffer-view-image.glb')
    await writeFile(modelPath, glbBytes)

    await assert.rejects(
      () => loadGltfFixture(modelPath),
      /GLB bufferView image.*compressed texture.*KTX2.*Basis.*pre-decode/i,
    )
  } finally {
    await rm(tmp, { recursive: true, force: true })
  }
})

test('loadGltfFromFile rejects external compressed glTF image references with pre-decode guidance', async () => {
  const source = JSON.parse(await readFile(TEXTURED_QUAD, 'utf8'))
  source.images[0].uri = 'textures/albedo.ktx2'
  const tmp = await mkdtemp(path.join(os.tmpdir(), 'headless-three-gltf-compressed-image-'))
  try {
    const modelPath = path.join(tmp, 'compressed-image-reference.gltf')
    await writeFile(modelPath, JSON.stringify(source))

    await assert.rejects(
      () => loadGltfFixture(modelPath),
      /glTF image URI.*compressed texture.*KTX2.*Basis.*pre-decode/i,
    )
  } finally {
    await rm(tmp, { recursive: true, force: true })
  }
})

test('loadGltfFromFile rejects malformed glTF image metadata clearly', async () => {
  await assertRejectsMutatedGltfSource((source) => {
    source.images = 'images'
  }, /glTF\.images must be an array/i)

  await assertRejectsMutatedGltfSource((source) => {
    source.images[0] = 'image'
  }, /glTF\.images\[0\] must be an object/i)

  await assertRejectsMutatedGltfSource((source) => {
    source.images[0] = { bufferView: 0 }
  }, /glTF bufferView image is missing mimeType/i)

  await assertRejectsMutatedGltfSource((source) => {
    source.images[0] = { bufferView: 0, mimeType: 'image/ktx2' }
  }, /glTF bufferView image.*compressed texture.*KTX2.*Basis.*pre-decode/i)
})

test('committed vertex-color glTF fixture renders COLOR_0 attributes', async () => {
  const gltf = await loadGltfFixture(VERTEX_COLOR_QUAD)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'vertex-color fixture should load a mesh')
  assert.equal(mesh.geometry.getAttribute('color')?.count, 4)
  assert.equal(mesh.material.vertexColors, true)

  const camera = gltf.cameras[0]
  assert.ok(camera, 'vertex-color fixture should load a camera')
  camera.aspect = 1
  camera.updateProjectionMatrix()
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(gltf.scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })
  assert.equal(rgba.length, 96 * 96 * 4)
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.25, 'vertex-color quad should render visible pixels')

  const left = meanRegion(rgba, 96, 96, 24, 36, 42, 60)
  const right = meanRegion(rgba, 96, 96, 54, 36, 72, 60)
  assert.ok(left.r > left.g + 60, `left half should be dominated by COLOR_0 red (${left.r} vs ${left.g})`)
  assert.ok(right.g > right.r + 60, `right half should be dominated by COLOR_0 green (${right.g} vs ${right.r})`)
})

test('committed morph-target glTF fixture applies POSITION targets', async () => {
  const gltf = await loadGltfFixture(MORPHED_TRIANGLE)
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, 'morph fixture should load a mesh')
  assert.equal(mesh.geometry.morphAttributes.position?.length, 1)
  assert.equal(mesh.morphTargetInfluences?.length, 1)

  const camera = gltf.cameras[0]
  assert.ok(camera, 'morph fixture should load a camera')
  camera.aspect = 1
  camera.updateProjectionMatrix()
  gltf.scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  function renderBounds(influence) {
    mesh.morphTargetInfluences[0] = influence
    const rgba = new Renderer().render(gltf.scene, camera, {
      width: 96,
      height: 96,
      format: 'rgba',
      background: [0, 0, 0],
      outputColorSpace: THREE.LinearSRGBColorSpace,
    })
    return nonBackgroundBounds(rgba, 96, 96, [0, 0, 0], 3)
  }

  const flat = renderBounds(0)
  const morphed = renderBounds(1)
  assert.ok(flat.height > 10, `flat triangle should render visible bounds (${flat.height})`)
  assert.ok(morphed.minY < flat.minY - 12, `morph target should move the triangle top upward (${morphed.minY} vs ${flat.minY})`)
  assert.ok(morphed.height > flat.height + 10, `morph target should expand rendered height (${morphed.height} vs ${flat.height})`)
})

test('committed skinned glTF fixture applies JOINTS_0 and WEIGHTS_0 attributes', async () => {
  const gltf = await loadGltfFixture(SKINNED_QUAD)
  const mesh = findFirst(gltf.scene, (object) => object.isSkinnedMesh === true)
  assert.ok(mesh, 'skinned fixture should load a SkinnedMesh')
  assert.equal(mesh.geometry.getAttribute('skinIndex')?.count, 4)
  assert.equal(mesh.geometry.getAttribute('skinWeight')?.count, 4)
  assert.equal(mesh.skeleton.bones.length, 1)

  const camera = gltf.cameras[0]
  assert.ok(camera, 'skinned fixture should load a camera')
  camera.aspect = 1
  camera.updateProjectionMatrix()
  camera.updateMatrixWorld(true)

  function renderBounds(jointY) {
    mesh.skeleton.bones[0].position.y = jointY
    gltf.scene.updateMatrixWorld(true)
    const rgba = new Renderer().render(gltf.scene, camera, {
      width: 96,
      height: 96,
      format: 'rgba',
      background: [0, 0, 0],
      outputColorSpace: THREE.LinearSRGBColorSpace,
    })
    return nonBackgroundBounds(rgba, 96, 96, [0, 0, 0], 3)
  }

  const base = renderBounds(0)
  const moved = renderBounds(0.55)
  assert.ok(base.height > 20, `base skinned quad should render visible bounds (${base.height})`)
  assert.ok(moved.minY < base.minY - 12, `joint translation should move the skinned quad upward (${moved.minY} vs ${base.minY})`)
  assert.ok(Math.abs(moved.height - base.height) <= 4, `single-joint translation should preserve quad height (${moved.height} vs ${base.height})`)
})

test('VRM loader helpers register supplied Pixiv-style plugins', async () => {
  let vrmPluginParser = null
  let animationPluginParser = null
  let modelPluginParser = null

  class FakeVRMLoaderPlugin {
    constructor(parser) {
      this.name = 'FakeVRMLoaderPlugin'
      vrmPluginParser = parser
    }
  }

  class FakeModelLoaderPlugin {
    constructor(parser) {
      this.name = 'FakeModelLoaderPlugin'
      modelPluginParser = parser
    }
  }

  class FakeVRMAnimationLoaderPlugin {
    constructor(parser) {
      this.name = 'FakeVRMAnimationLoaderPlugin'
      animationPluginParser = parser
    }
  }

  const vrmGltf = await loadVrmFromFile(SYNTHETIC_VRM, {
    VRMLoaderPlugin: FakeVRMLoaderPlugin,
  })
  assert.ok(findFirst(vrmGltf.scene, (object) => object.isMesh === true), 'VRM helper should still parse glTF scenes')
  assert.ok(vrmPluginParser, 'VRM helper should install the supplied VRMLoaderPlugin')
  assert.ok(vrmPluginParser.json?.extensionsUsed?.includes('VRMC_vrm'), 'VRM fixture should expose VRMC_vrm metadata to the plugin')

  const animationGltf = await loadVrmAnimationFromFile(SYNTHETIC_VRMA, {
    VRMLoaderPlugin: FakeModelLoaderPlugin,
    VRMAnimationLoaderPlugin: FakeVRMAnimationLoaderPlugin,
  })
  assert.ok(findFirst(animationGltf.scene, (object) => object.isMesh === true), 'VRMA helper should still parse glTF scenes')
  assert.ok(modelPluginParser, 'VRMA helper should install the supplied VRMLoaderPlugin when provided')
  assert.ok(animationPluginParser, 'VRMA helper should install the supplied VRMAnimationLoaderPlugin')
  assert.ok(
    animationPluginParser.json?.extensionsUsed?.includes('VRMC_vrm_animation'),
    'VRMA fixture should expose VRMC_vrm_animation metadata to the plugin',
  )
})

test('loadGltfFromFile resolves external glTF image files from the model directory', async () => {
  const source = JSON.parse(await readFile(TEXTURED_QUAD, 'utf8'))
  const imageUri = source.images[0].uri
  const encodedImage = imageUri.slice(imageUri.indexOf(',') + 1)
  const tmp = await mkdtemp(path.join(os.tmpdir(), 'headless-three-gltf-image-'))
  try {
    const textureDir = path.join(tmp, 'textures')
    await mkdir(textureDir)
    await writeFile(path.join(textureDir, 'quad.png'), Buffer.from(encodedImage, 'base64'))
    source.images[0].uri = 'textures/quad.png'
    const modelPath = path.join(tmp, 'external-image.gltf')
    await writeFile(modelPath, JSON.stringify(source))

    const gltf = await loadGltfFixture(modelPath)
    assertTexturedQuadLoadsEncodedMap(gltf, 'external-image fixture')
    assertTexturedQuadRendersTexture(gltf, 'external-image quad')
  } finally {
    await rm(tmp, { recursive: true, force: true })
  }
})

test('loadGltfFromFile resolves external glTF buffers from the model directory', async () => {
  const source = JSON.parse(await readFile(SIMPLE_TRIANGLE, 'utf8'))
  const bufferUri = source.buffers[0].uri
  const encodedBuffer = bufferUri.slice(bufferUri.indexOf(',') + 1)
  const tmp = await mkdtemp(path.join(os.tmpdir(), 'headless-three-gltf-'))
  try {
    await writeFile(path.join(tmp, 'triangle.bin'), Buffer.from(encodedBuffer, 'base64'))
    source.buffers[0].uri = 'triangle.bin'
    const modelPath = path.join(tmp, 'external-buffer.gltf')
    await writeFile(modelPath, JSON.stringify(source))

    const gltf = await loadGltfFixture(modelPath)
    const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
    assert.ok(mesh, 'external-buffer fixture should load a mesh')

    const camera = gltf.cameras[0]
    assert.ok(camera, 'external-buffer fixture should load a camera')
    camera.aspect = 1
    camera.updateProjectionMatrix()
    gltf.scene.add(new THREE.AmbientLight(0xffffff, 1))
    gltf.scene.updateMatrixWorld(true)
    camera.updateMatrixWorld(true)

    const rgba = new Renderer().render(gltf.scene, camera, {
      width: 64,
      height: 64,
      format: 'rgba',
      background: [0, 0, 0],
    })
    assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.04, 'external buffer glTF should render visible pixels')
  } finally {
    await rm(tmp, { recursive: true, force: true })
  }
})

function assertTexturedQuadLoadsEncodedMap(gltf, label) {
  const mesh = findFirst(gltf.scene, (object) => object.isMesh === true)
  assert.ok(mesh, `${label} should load a mesh`)
  assert.ok(mesh.material.map?.isTexture, `${label} should load a base color texture`)
  assert.ok(Buffer.isBuffer(mesh.material.map.image), 'encoded image helper should expose the PNG as a Buffer')
}

function assertTexturedQuadRendersTexture(gltf, label) {
  const camera = gltf.cameras[0]
  assert.ok(camera, `${label} should load a camera`)
  camera.aspect = 1
  camera.updateProjectionMatrix()

  const scene = gltf.scene
  scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const rgba = new Renderer().render(scene, camera, {
    width: 96,
    height: 96,
    format: 'rgba',
    background: [0, 0, 0],
    outputColorSpace: THREE.LinearSRGBColorSpace,
  })
  assert.equal(rgba.length, 96 * 96 * 4)
  assert.ok(nonBackgroundRatio(rgba, [0, 0, 0], 3) > 0.25, `${label} should render visible pixels`)

  const left = meanRegion(rgba, 96, 96, 24, 36, 42, 60)
  const right = meanRegion(rgba, 96, 96, 54, 36, 72, 60)
  assert.ok(left.r > left.g + 80, `left half should sample the red texture texel (${left.r} vs ${left.g})`)
  assert.ok(right.g > right.r + 80, `right half should sample the green texture texel (${right.g} vs ${right.r})`)
}

async function loadGltfFixture(filePath, options) {
  return await loadGltfFromFile(filePath, options)
}

function vectorFromAttribute(attribute, index) {
  return [attribute.getX(index), attribute.getY(index), attribute.getZ(index)]
}

async function assertRejectsMutatedGltfSource(mutator, pattern) {
  const source = JSON.parse(await readFile(TEXTURED_QUAD, 'utf8'))
  mutator(source)
  const tmp = await mkdtemp(path.join(os.tmpdir(), 'headless-three-gltf-malformed-image-'))
  try {
    const modelPath = path.join(tmp, 'malformed-image-metadata.gltf')
    await writeFile(modelPath, JSON.stringify(source))
    await assert.rejects(
      () => loadGltfFixture(modelPath),
      pattern,
    )
  } finally {
    await rm(tmp, { recursive: true, force: true })
  }
}

function buildTexturedQuadGlb(source) {
  const geometryBytes = decodeDataUriBuffer(source.buffers[0].uri, 'textured fixture geometry buffer')
  const imageBytes = decodeDataUriBuffer(source.images[0].uri, 'textured fixture image')
  const imageOffset = alignedLength(geometryBytes.length)
  const binLength = imageOffset + imageBytes.length
  const bin = Buffer.alloc(alignedLength(binLength))
  geometryBytes.copy(bin, 0)
  imageBytes.copy(bin, imageOffset)

  const glb = structuredClone(source)
  delete glb.buffers[0].uri
  glb.buffers[0].byteLength = binLength
  glb.bufferViews.push({
    buffer: 0,
    byteOffset: imageOffset,
    byteLength: imageBytes.length,
  })
  glb.images[0] = {
    name: source.images[0].name,
    mimeType: source.images[0].mimeType,
    bufferView: glb.bufferViews.length - 1,
  }

  return encodeGlb(glb, bin)
}

function decodeDataUriBuffer(uri, label) {
  assert.equal(typeof uri, 'string', `${label} should be a data URI`)
  const comma = uri.indexOf(',')
  assert.notEqual(comma, -1, `${label} should contain a comma separator`)
  const metadata = uri.slice(5, comma)
  const payload = uri.slice(comma + 1)
  return /(?:^|;)base64(?:;|$)/i.test(metadata)
    ? Buffer.from(payload, 'base64')
    : Buffer.from(decodeURIComponent(payload), 'utf8')
}

function encodeGlb(json, bin) {
  const jsonChunk = paddedBuffer(Buffer.from(JSON.stringify(json), 'utf8'), 0x20)
  const binChunk = paddedBuffer(bin, 0x00)
  const totalLength = 12 + 8 + jsonChunk.length + 8 + binChunk.length
  const glb = Buffer.alloc(totalLength)
  let offset = 0
  offset = writeUint32(glb, offset, 0x46546c67)
  offset = writeUint32(glb, offset, 2)
  offset = writeUint32(glb, offset, totalLength)
  offset = writeUint32(glb, offset, jsonChunk.length)
  offset = writeUint32(glb, offset, 0x4e4f534a)
  jsonChunk.copy(glb, offset)
  offset += jsonChunk.length
  offset = writeUint32(glb, offset, binChunk.length)
  offset = writeUint32(glb, offset, 0x004e4942)
  binChunk.copy(glb, offset)
  return glb
}

function paddedBuffer(buffer, fill) {
  const padded = Buffer.alloc(alignedLength(buffer.length), fill)
  buffer.copy(padded)
  return padded
}

function alignedLength(length) {
  return (length + 3) & ~3
}

function writeUint32(buffer, offset, value) {
  buffer.writeUInt32LE(value, offset)
  return offset + 4
}

function findFirst(root, predicate) {
  let match = null
  root.traverse((object) => {
    if (!match && predicate(object)) match = object
  })
  return match
}

function meanRegion(rgba, width, _height, x0, y0, x1, y1) {
  let r = 0
  let g = 0
  let b = 0
  let a = 0
  let count = 0
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const i = (y * width + x) * 4
      r += rgba[i]
      g += rgba[i + 1]
      b += rgba[i + 2]
      a += rgba[i + 3]
      count++
    }
  }
  return { r: r / count, g: g / count, b: b / count, a: a / count }
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
