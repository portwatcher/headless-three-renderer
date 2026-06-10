import type { ThreeCameraLike, ThreeObject3DLike, NativeSceneLight } from './types'
import { colorLikeToArray } from './color'
import { objectLayersMatchCamera } from './layers'

type ShadowMapSizeLike = { x?: number; y?: number; width?: number; height?: number } | undefined
const MAX_NATIVE_LIGHTS = 16

export function extractLights(scene: ThreeObject3DLike, camera?: ThreeCameraLike): NativeSceneLight[] | undefined {
  const lights: NativeSceneLight[] = []
  visitLights(scene, camera, lights)
  assertSupportedLightCount(lights)
  assertSupportedShadowLightCount(lights)
  return lights.length > 0 ? lights : undefined
}

function visitLights(object: ThreeObject3DLike, camera: ThreeCameraLike | undefined, lights: NativeSceneLight[]): void {
  if (!object) return
  if (object.visible === false) return

  if (object.isLight === true && objectLayersMatchCamera(object, camera)) {
    const light = extractLight(object)
    if (light) lights.push(light)
  }

  const children = Array.isArray(object.children) ? object.children : []
  for (const child of children) {
    visitLights(child, camera, lights)
  }
}

function extractLight(light: ThreeObject3DLike): NativeSceneLight | null {
  const color = colorLikeToArray(light.color) ?? [1, 1, 1, 1]
  const intensity = Number.isFinite(light.intensity) ? light.intensity! : 1

  if (light.isDirectionalLight === true) {
    const pos = light.matrixWorld
      ? [light.matrixWorld.elements[12], light.matrixWorld.elements[13], light.matrixWorld.elements[14]]
      : [0, 10, 0]
    let targetPos = [0, 0, 0]
    if (light.target?.matrixWorld) {
      const te = light.target.matrixWorld.elements
      targetPos = [te[12], te[13], te[14]]
    }
    const direction = [
      targetPos[0] - pos[0],
      targetPos[1] - pos[1],
      targetPos[2] - pos[2],
    ]
    const len = Math.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
    if (len > 0) {
      direction[0] /= len
      direction[1] /= len
      direction[2] /= len
    }
    const out: NativeSceneLight = {
      lightType: 'directional',
      color: [color[0], color[1], color[2]],
      intensity,
      position: pos,
      direction,
    }
    if (light.castShadow === true) {
      out.castShadow = true
      applyShadowOptions(out, light)
    }
    return out
  }

  if (light.isPointLight === true) {
    const pos = light.matrixWorld
      ? [light.matrixWorld.elements[12], light.matrixWorld.elements[13], light.matrixWorld.elements[14]]
      : [0, 0, 0]
    const out: NativeSceneLight = {
      lightType: 'point',
      color: [color[0], color[1], color[2]],
      intensity,
      position: pos,
      distance: Number.isFinite(light.distance) ? light.distance! : 0,
      decay: Number.isFinite(light.decay) ? light.decay! : 2,
    }
    if (light.castShadow === true) {
      out.castShadow = true
      applyShadowOptions(out, light)
    }
    return out
  }

  if (light.isSpotLight === true) {
    const pos = light.matrixWorld
      ? [light.matrixWorld.elements[12], light.matrixWorld.elements[13], light.matrixWorld.elements[14]]
      : [0, 0, 0]
    let targetPos = [0, 0, 0]
    if (light.target?.matrixWorld) {
      const te = light.target.matrixWorld.elements
      targetPos = [te[12], te[13], te[14]]
    }
    const direction = [
      targetPos[0] - pos[0],
      targetPos[1] - pos[1],
      targetPos[2] - pos[2],
    ]
    const len = Math.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
    if (len > 0) {
      direction[0] /= len
      direction[1] /= len
      direction[2] /= len
    }
    const out: NativeSceneLight = {
      lightType: 'spot',
      color: [color[0], color[1], color[2]],
      intensity,
      position: pos,
      direction,
      distance: Number.isFinite(light.distance) ? light.distance! : 0,
      decay: Number.isFinite(light.decay) ? light.decay! : 2,
      angle: Number.isFinite(light.angle) ? light.angle! : Math.PI / 3,
      penumbra: Number.isFinite(light.penumbra) ? light.penumbra! : 0,
    }
    if (light.castShadow === true) {
      out.castShadow = true
      applyShadowOptions(out, light)
    }
    return out
  }

  if (light.isRectAreaLight === true) {
    const pos = light.matrixWorld
      ? [light.matrixWorld.elements[12], light.matrixWorld.elements[13], light.matrixWorld.elements[14]]
      : [0, 0, 0]
    let direction = [0, 0, -1]
    if (light.matrixWorld) {
      const e = light.matrixWorld.elements
      direction = [-e[8], -e[9], -e[10]]
    }
    const len = Math.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
    if (len > 0) {
      direction[0] /= len
      direction[1] /= len
      direction[2] /= len
    }
    return {
      lightType: 'rectArea',
      color: [color[0], color[1], color[2]],
      intensity,
      position: pos,
      direction,
      width: Number.isFinite(light.width) ? light.width! : 10,
      height: Number.isFinite(light.height) ? light.height! : 10,
    }
  }

  if (light.isHemisphereLight === true) {
    const groundColor = colorLikeToArray(light.groundColor) ?? [0.04, 0.02, 0.0, 1]
    let direction = [0, 1, 0]
    if (light.matrixWorld) {
      const e = light.matrixWorld.elements
      const ux = e[4], uy = e[5], uz = e[6]
      const ulen = Math.sqrt(ux * ux + uy * uy + uz * uz)
      if (ulen > 0) {
        direction = [ux / ulen, uy / ulen, uz / ulen]
      }
    }
    return {
      lightType: 'hemisphere',
      color: [color[0], color[1], color[2]],
      intensity,
      direction,
      groundColor: [groundColor[0], groundColor[1], groundColor[2]],
    }
  }

  // AmbientLight is handled separately
  return null
}

function applyShadowOptions(out: NativeSceneLight, light: ThreeObject3DLike): void {
  const shadow = light.shadow
  assertSupportedShadowBlurOptions(shadow)
  out.shadowMapSize = shadowMapSizeOrDefault(shadow?.mapSize)
  if (Number.isFinite(shadow?.bias)) out.shadowBias = shadow!.bias!
  if (Number.isFinite(shadow?.normalBias)) out.shadowNormalBias = shadow!.normalBias!

  const cam = shadow?.camera
  if (cam) {
    if (Number.isFinite(cam.left)) out.shadowCameraLeft = cam.left!
    if (Number.isFinite(cam.right)) out.shadowCameraRight = cam.right!
    if (Number.isFinite(cam.top)) out.shadowCameraTop = cam.top!
    if (Number.isFinite(cam.bottom)) out.shadowCameraBottom = cam.bottom!
    if (Number.isFinite(cam.near)) out.shadowCameraNear = cam.near!
    if (Number.isFinite(cam.far)) out.shadowCameraFar = cam.far!
  }

  applyShadowCascadeOptions(out, light)
}

function shadowMapSizeOrDefault(mapSize: ShadowMapSizeLike): number {
  const width = numberOrNull(mapSize?.x ?? mapSize?.width)
  const height = numberOrNull(mapSize?.y ?? mapSize?.height)
  if (width != null && height != null && Math.floor(width) !== Math.floor(height)) {
    throw new Error(
      `Non-square light.shadow.mapSize values are not supported by @headless-three/renderer yet (${Math.floor(width)}x${Math.floor(height)}). Use a square shadow map size until rectangular shadow maps are supported.`,
    )
  }
  return Math.max(32, Math.floor(width ?? height ?? 512))
}

function assertSupportedShadowBlurOptions(shadow: ThreeObject3DLike['shadow']): void {
  if (Number.isFinite(shadow?.radius) && Math.abs(shadow!.radius! - 1) > 1e-12) {
    throw new Error(
      `Non-default light.shadow.radius values are not supported by @headless-three/renderer yet (${shadow!.radius}). Use the default radius of 1 until configurable shadow blur/radius support lands.`,
    )
  }
  if (Number.isFinite(shadow?.blurSamples) && shadow!.blurSamples !== 8) {
    throw new Error(
      `Non-default light.shadow.blurSamples values are not supported by @headless-three/renderer yet (${shadow!.blurSamples}). Use the default blurSamples value of 8 until configurable shadow blur/radius support lands.`,
    )
  }
}

function applyShadowCascadeOptions(out: NativeSceneLight, light: ThreeObject3DLike): void {
  const hints = light.userData?.headlessThreeRenderer ?? light.userData?.headlessRenderer ?? {}
  const cascades = hints.shadowCascades ?? hints.cascades ?? (light.shadow as any)?.cascades
  if (!Array.isArray(cascades) || cascades.length < 2) return

  const splits: number[] = []
  const bounds: number[] = []

  for (const cascade of cascades.slice(0, 4)) {
    if (!cascade || typeof cascade !== 'object') continue
    const left = numberOrNull(cascade.left)
    const right = numberOrNull(cascade.right)
    const top = numberOrNull(cascade.top)
    const bottom = numberOrNull(cascade.bottom)
    const near = numberOrNull(cascade.near)
    const far = numberOrNull(cascade.far)
    if (left == null || right == null || top == null || bottom == null || near == null || far == null) {
      continue
    }
    bounds.push(left, right, top, bottom, near, far)
    const split = numberOrNull(cascade.split ?? cascade.distance ?? cascade.farDistance)
    if (split != null) splits.push(split)
  }

  const count = bounds.length / 6
  if (count >= 2) {
    out.shadowCascadeBounds = bounds
    out.shadowCascadeSplits = splits.slice(0, count - 1)
  }
}

function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function assertSupportedShadowLightCount(lights: NativeSceneLight[]): void {
  let shadowCasters = 0
  for (const light of lights) {
    if (light.castShadow === true) shadowCasters += 1
  }
  if (shadowCasters > 1) {
    throw new Error(
      'Multiple shadow-casting lights are not supported by @headless-three/renderer yet. Keep one visible directional, spot, or point light with castShadow enabled, or render separate passes until multiple shadow maps are supported.',
    )
  }
}

function assertSupportedLightCount(lights: NativeSceneLight[]): void {
  if (lights.length > MAX_NATIVE_LIGHTS) {
    throw new Error(
      `More than ${MAX_NATIVE_LIGHTS} visible non-ambient lights are not supported by @headless-three/renderer yet (${lights.length} found). Keep the closest/brightest ${MAX_NATIVE_LIGHTS} lights, bake lighting, or render separate passes until native light arrays are expanded.`,
    )
  }
}

export function extractAmbientLight(scene: ThreeObject3DLike, camera?: ThreeCameraLike): number[] | null {
  let color: number[] | null = null
  visitForAmbient(scene, camera, (light) => {
    const c = colorLikeToArray(light.color) ?? [1, 1, 1, 1]
    if (!color) {
      color = [c[0], c[1], c[2]]
    } else {
      color[0] = Math.min(1, color[0] + c[0])
      color[1] = Math.min(1, color[1] + c[1])
      color[2] = Math.min(1, color[2] + c[2])
    }
  })
  return color
}

export function extractAmbientIntensity(scene: ThreeObject3DLike, camera?: ThreeCameraLike): number | undefined {
  let intensity = 0
  visitForAmbient(scene, camera, (light) => {
    intensity += Number.isFinite(light.intensity) ? light.intensity! : 1
  })
  return intensity > 0 ? intensity : undefined
}

export function extractLightProbe(scene: ThreeObject3DLike, camera?: ThreeCameraLike): number[] | null {
  const coefficients = new Array<number>(27).fill(0)
  let found = false

  visitForLightProbe(scene, camera, (light) => {
    const source = light.sh?.coefficients
    if (!Array.isArray(source) || source.length < 9) return

    const intensity = Number.isFinite(light.intensity) ? light.intensity! : 1
    for (let i = 0; i < 9; i += 1) {
      const coefficient = coefficientToRgb(source[i])
      if (!coefficient) continue
      coefficients[i * 3] += coefficient[0] * intensity
      coefficients[i * 3 + 1] += coefficient[1] * intensity
      coefficients[i * 3 + 2] += coefficient[2] * intensity
    }
    found = true
  })

  return found ? coefficients : null
}

function visitForLightProbe(
  object: ThreeObject3DLike,
  camera: ThreeCameraLike | undefined,
  callback: (light: ThreeObject3DLike) => void,
): void {
  if (!object) return
  if (object.visible === false) return
  if (object.isLightProbe === true && objectLayersMatchCamera(object, camera)) callback(object)
  const children = Array.isArray(object.children) ? object.children : []
  for (const child of children) {
    visitForLightProbe(child, camera, callback)
  }
}

function coefficientToRgb(value: unknown): [number, number, number] | null {
  if (!value) return null
  if (Array.isArray(value) || ArrayBuffer.isView(value)) {
    const array = value as ArrayLike<number>
    return finiteRgb(array[0], array[1], array[2])
  }
  const v = value as { r?: number; g?: number; b?: number; x?: number; y?: number; z?: number }
  return finiteRgb(v.r ?? v.x, v.g ?? v.y, v.b ?? v.z)
}

function finiteRgb(r: unknown, g: unknown, b: unknown): [number, number, number] | null {
  if (typeof r !== 'number' || typeof g !== 'number' || typeof b !== 'number') return null
  if (!Number.isFinite(r) || !Number.isFinite(g) || !Number.isFinite(b)) return null
  return [r, g, b]
}

function visitForAmbient(
  object: ThreeObject3DLike,
  camera: ThreeCameraLike | undefined,
  callback: (light: ThreeObject3DLike) => void,
): void {
  if (!object) return
  if (object.visible === false) return
  if (object.isAmbientLight === true && objectLayersMatchCamera(object, camera)) callback(object)
  const children = Array.isArray(object.children) ? object.children : []
  for (const child of children) {
    visitForAmbient(child, camera, callback)
  }
}
