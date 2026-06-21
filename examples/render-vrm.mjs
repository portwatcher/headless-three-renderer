#!/usr/bin/env node
import fs from 'node:fs/promises'
import { createRequire } from 'node:module'
import path from 'node:path'
import { pathToFileURL } from 'node:url'
import {
  applyVrmAnimation,
  loadVrmAnimationFromFile,
  loadVrmFromFile,
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

const THREE = await importThree()
try {
  const animationIndex = nonNegativeInteger(process.env.ANIMATION_INDEX, 0, 'ANIMATION_INDEX')
  const packages = await importVrmPackages(Boolean(animationPath))
  const modelGltf = await loadVrmFromFile(modelPath, {
    VRMLoaderPlugin: packages.VRMLoaderPlugin,
  })
  const vrm = modelGltf.userData?.vrm
  if (!vrm?.scene) {
    throw new Error(`No VRM model was found in ${modelPath}. Confirm that the file is a VRM asset and @pixiv/three-vrm can parse it.`)
  }

  packages.VRMUtils?.removeUnnecessaryVertices?.(vrm.scene)
  packages.VRMUtils?.removeUnnecessaryJoints?.(vrm.scene)
  vrm.scene.rotation.y = Math.PI

  if (animationPath) {
    const animationGltf = await loadVrmAnimationFromFile(animationPath, {
      VRMLoaderPlugin: packages.VRMLoaderPlugin,
      VRMAnimationLoaderPlugin: packages.VRMAnimationLoaderPlugin,
    })
    const vrmAnimations = animationGltf.userData?.vrmAnimations
    const vrmAnimation = Array.isArray(vrmAnimations) ? vrmAnimations[animationIndex] : null
    if (!vrmAnimation) {
      throw new Error(`No VRMA animation was found at index ${animationIndex} in ${animationPath}. Confirm that the file is a VRM Animation asset.`)
    }
    await applyVrmAnimation(modelGltf, animationGltf, {
      animationIndex,
      createVRMAnimationClip: packages.createVRMAnimationClip,
      time: animationTime,
    })
  }

  if (!animationPath) {
    vrm.update?.(0)
  }

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0.04, 0.045, 0.055)
  scene.add(vrm.scene)
  addPortraitLights(scene)

  const camera = findRenderableCamera(vrm.scene, width / height) ?? frameSceneCamera(vrm.scene, width / height)
  scene.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const image = render(scene, camera, { width, height })
  await fs.writeFile(outputPath, image)
  const animationLabel = animationPath ? ` with ${animationPath} animation #${animationIndex} at ${animationTime}s` : ''
  console.log(`Rendered ${modelPath}${animationLabel} to ${outputPath} (${width}x${height})`)
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error))
  process.exit(1)
}

async function importVrmPackages(needsAnimation) {
  const vrmModule = await importOptionalPackage(
    '@pixiv/three-vrm',
    'Missing optional dependency @pixiv/three-vrm. Install it in your project before running this VRM example.',
  )

  if (!needsAnimation) return vrmModule

  const animationModule = await importOptionalPackage(
    '@pixiv/three-vrm-animation',
    'Missing optional dependency @pixiv/three-vrm-animation. Install it before rendering VRMA animation files.',
  )

  return {
    ...vrmModule,
    VRMAnimationLoaderPlugin: animationModule.VRMAnimationLoaderPlugin,
    createVRMAnimationClip: animationModule.createVRMAnimationClip,
  }
}

async function importOptionalPackage(specifier, missingMessage) {
  try {
    return normalizeModuleNamespace(await import(specifier))
  } catch (error) {
    if (error?.code !== 'ERR_MODULE_NOT_FOUND') throw error
  }

  try {
    const requireFromCaller = createRequire(path.join(process.cwd(), 'package.json'))
    return normalizeModuleNamespace(await import(pathToFileURL(requireFromCaller.resolve(specifier)).href))
  } catch (error) {
    if (error?.code !== 'ERR_MODULE_NOT_FOUND' && error?.code !== 'MODULE_NOT_FOUND') throw error
    throw new Error(missingMessage)
  }
}

function normalizeModuleNamespace(module) {
  if (module?.default && typeof module.default === 'object') {
    return { ...module.default, ...module }
  }
  return module
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

function nonNegativeInteger(value, fallback, label) {
  if (value == null || value === '') return fallback
  const parsed = Number(value)
  if (Number.isInteger(parsed) && parsed >= 0) return parsed
  throw new Error(`${label} must be a non-negative integer.`)
}

async function importThree() {
  try {
    return await import('three')
  } catch (error) {
    if (error?.code !== 'ERR_MODULE_NOT_FOUND') throw error
    const requireFromRenderer = createRequire(new URL('../packages/renderer/package.json', import.meta.url))
    return requireFromRenderer('three')
  }
}
