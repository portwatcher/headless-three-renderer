#!/usr/bin/env node
import fs from 'node:fs/promises'
import * as THREE from 'three'
import {
  loadGltfFromFile,
  render,
} from '../packages/renderer/dist/index.js'

const [modelPath, secondArg, thirdArg] = process.argv.slice(2)
const hasAnimation = typeof secondArg === 'string' && /\.vrma$/i.test(secondArg)
const animationPath = hasAnimation ? secondArg : null
const outputPath = hasAnimation ? (thirdArg ?? 'render.png') : (secondArg ?? 'render.png')

if (!modelPath) {
  console.error('Usage: node examples/render-vrm.mjs <avatar.vrm> [animation.vrma] [render.png]')
  process.exit(1)
}

const width = positiveInteger(process.env.WIDTH, 1024)
const height = positiveInteger(process.env.HEIGHT, width)
const animationTime = Number.isFinite(Number(process.env.TIME)) ? Number(process.env.TIME) : 1.5

try {
  const packages = await importVrmPackages(Boolean(animationPath))
  const configureLoader = (loader) => {
    loader.register((parser) => new packages.VRMLoaderPlugin(parser))
    if (packages.VRMAnimationLoaderPlugin) {
      loader.register((parser) => new packages.VRMAnimationLoaderPlugin(parser))
    }
  }

  const modelGltf = await loadGltfFromFile(modelPath, { configureLoader })
  const vrm = modelGltf.userData?.vrm
  if (!vrm?.scene) {
    throw new Error(`No VRM model was found in ${modelPath}. Confirm that the file is a VRM asset and @pixiv/three-vrm can parse it.`)
  }

  packages.VRMUtils?.removeUnnecessaryVertices?.(vrm.scene)
  packages.VRMUtils?.removeUnnecessaryJoints?.(vrm.scene)
  vrm.scene.rotation.y = Math.PI

  if (animationPath) {
    const animationGltf = await loadGltfFromFile(animationPath, { configureLoader })
    const vrmAnimation = animationGltf.userData?.vrmAnimations?.[0]
    if (!vrmAnimation) {
      throw new Error(`No VRMA animation was found in ${animationPath}. Confirm that the file is a VRM Animation asset.`)
    }
    const clip = packages.createVRMAnimationClip(vrmAnimation, vrm)
    const mixer = new THREE.AnimationMixer(vrm.scene)
    mixer.clipAction(clip).play()
    mixer.update(animationTime)
  }

  vrm.update?.(0)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.04, 0.045, 0.055)
  scene.add(vrm.scene)
  addPortraitLights(scene)

  const camera = findRenderableCamera(vrm.scene, width / height) ?? frameSceneCamera(vrm.scene, width / height)
  scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const image = render(scene, camera, { width, height })
  await fs.writeFile(outputPath, image)
  const animationLabel = animationPath ? ` with ${animationPath} at ${animationTime}s` : ''
  console.log(`Rendered ${modelPath}${animationLabel} to ${outputPath} (${width}x${height})`)
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error))
  process.exit(1)
}

async function importVrmPackages(needsAnimation) {
  let vrmModule
  try {
    vrmModule = await import('@pixiv/three-vrm')
  } catch {
    throw new Error('Missing optional dependency @pixiv/three-vrm. Install it in your project before running this VRM example.')
  }

  if (!needsAnimation) return vrmModule

  let animationModule
  try {
    animationModule = await import('@pixiv/three-vrm-animation')
  } catch {
    throw new Error('Missing optional dependency @pixiv/three-vrm-animation. Install it before rendering VRMA animation files.')
  }

  return {
    ...vrmModule,
    VRMAnimationLoaderPlugin: animationModule.VRMAnimationLoaderPlugin,
    createVRMAnimationClip: animationModule.createVRMAnimationClip,
  }
}

function addPortraitLights(scene) {
  scene.add(new THREE.HemisphereLight(0xbfd7ff, 0x332211, 0.7))
  const key = new THREE.DirectionalLight(0xffffff, 1.4)
  key.position.set(2.5, 4, 3)
  key.target.position.set(0, 1, 0)
  scene.add(key, key.target)
}

function findRenderableCamera(scene, aspect) {
  let camera = null
  scene.traverse((object) => {
    if (!camera && object.isCamera === true && object.isArrayCamera !== true && object.type !== 'CubeCamera') {
      camera = object
    }
  })
  if (!camera) return null
  if (camera.isPerspectiveCamera === true && Number.isFinite(aspect)) {
    camera.aspect = aspect
    camera.updateProjectionMatrix()
  }
  return camera
}

function frameSceneCamera(scene, aspect) {
  const box = new THREE.Box3().setFromObject(scene)
  const sphere = box.getBoundingSphere(new THREE.Sphere())
  const center = Number.isFinite(sphere.center.x) ? sphere.center : new THREE.Vector3(0, 1, 0)
  const radius = Number.isFinite(sphere.radius) && sphere.radius > 0 ? sphere.radius : 1

  const camera = new THREE.PerspectiveCamera(30, aspect, Math.max(0.01, radius / 1000), radius * 100)
  const fov = THREE.MathUtils.degToRad(camera.fov)
  const distance = radius / Math.sin(fov / 2)
  camera.position.set(
    center.x + distance * 0.35,
    center.y + distance * 0.1,
    center.z + distance,
  )
  camera.lookAt(center)
  camera.updateProjectionMatrix()
  return camera
}

function positiveInteger(value, fallback) {
  const parsed = Number.parseInt(value ?? '', 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback
}
