import type { ThreeCameraLike, ThreeObject3DLike, NativeSceneLight } from './types'
import { strictColorLikeToArray } from './color'
import { objectLayersMatchCamera } from './layers'

type ShadowMapSizeLike = { x?: number; y?: number; width?: number; height?: number } | undefined
const MAX_NATIVE_LIGHTS = 64
const MAX_SHADOW_CASCADES = 4

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
  const color = strictColorLikeToArray(light.color, 'light.color') ?? [1, 1, 1, 1]
  const intensity = finiteNumberOrDefault(light.intensity, 'light.intensity', 1)

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
      distance: finiteNumberOrDefault(light.distance, 'PointLight.distance', 0),
      decay: finiteNumberOrDefault(light.decay, 'PointLight.decay', 2),
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
      distance: finiteNumberOrDefault(light.distance, 'SpotLight.distance', 0),
      decay: finiteNumberOrDefault(light.decay, 'SpotLight.decay', 2),
      angle: finiteNumberOrDefault(light.angle, 'SpotLight.angle', Math.PI / 3),
      penumbra: finiteNumberOrDefault(light.penumbra, 'SpotLight.penumbra', 0),
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
      width: finiteNumberOrDefault(light.width, 'RectAreaLight.width', 10),
      height: finiteNumberOrDefault(light.height, 'RectAreaLight.height', 10),
    }
  }

  if (light.isHemisphereLight === true) {
    const groundColor = strictColorLikeToArray(light.groundColor, 'HemisphereLight.groundColor') ?? [0.04, 0.02, 0.0, 1]
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
  const mapSize = shadowMapSizeOrDefault(shadow?.mapSize, light)
  out.shadowMapSize = Math.max(mapSize.width, mapSize.height)
  out.shadowMapWidth = mapSize.width
  out.shadowMapHeight = mapSize.height
  const bias = optionalFiniteNumber(shadow?.bias, 'light.shadow.bias')
  const normalBias = optionalFiniteNumber(shadow?.normalBias, 'light.shadow.normalBias')
  const radius = optionalFiniteNumber(shadow?.radius, 'light.shadow.radius')
  if (bias !== undefined) out.shadowBias = bias
  if (normalBias !== undefined) out.shadowNormalBias = normalBias
  if (radius !== undefined) out.shadowRadius = radius

  const cam = shadow?.camera
  if (cam) {
    const left = optionalFiniteNumber(cam.left, 'light.shadow.camera.left')
    const right = optionalFiniteNumber(cam.right, 'light.shadow.camera.right')
    const top = optionalFiniteNumber(cam.top, 'light.shadow.camera.top')
    const bottom = optionalFiniteNumber(cam.bottom, 'light.shadow.camera.bottom')
    const near = optionalFiniteNumber(cam.near, 'light.shadow.camera.near')
    const far = optionalFiniteNumber(cam.far, 'light.shadow.camera.far')
    if (left !== undefined) out.shadowCameraLeft = left
    if (right !== undefined) out.shadowCameraRight = right
    if (top !== undefined) out.shadowCameraTop = top
    if (bottom !== undefined) out.shadowCameraBottom = bottom
    if (near !== undefined) out.shadowCameraNear = near
    if (far !== undefined) out.shadowCameraFar = far
  }

  applyShadowCascadeOptions(out, light)
}

function shadowMapSizeOrDefault(mapSize: ShadowMapSizeLike, light: ThreeObject3DLike): { width: number; height: number } {
  const width = shadowMapSizeComponent(mapSize, 'x', 'width')
  const height = shadowMapSizeComponent(mapSize, 'y', 'height')
  const resolvedWidth = Math.max(32, Math.floor(width ?? height ?? 512))
  const resolvedHeight = Math.max(32, Math.floor(height ?? width ?? 512))
  if (light.isPointLight === true && resolvedWidth !== resolvedHeight) {
    throw new Error(
      `Non-square PointLight shadow map sizes are not supported by @headless-three/renderer yet (${resolvedWidth}x${resolvedHeight}). Use square point-light shadow maps until rectangular cube-face shadows are supported.`,
    )
  }
  return { width: resolvedWidth, height: resolvedHeight }
}

function shadowMapSizeComponent(
  mapSize: ShadowMapSizeLike,
  vectorKey: 'x' | 'y',
  dimensionKey: 'width' | 'height',
): number | undefined {
  if (!mapSize) return undefined
  const vectorValue = mapSize[vectorKey]
  if (vectorValue != null) return optionalFiniteNumber(vectorValue, `light.shadow.mapSize.${vectorKey}`)
  return optionalFiniteNumber(mapSize[dimensionKey], `light.shadow.mapSize.${dimensionKey}`)
}

function applyShadowCascadeOptions(out: NativeSceneLight, light: ThreeObject3DLike): void {
  const hints = light.userData?.headlessThreeRenderer ?? light.userData?.headlessRenderer ?? {}
  const cascades = hints.shadowCascades ?? hints.cascades ?? (light.shadow as any)?.cascades
  if (!Array.isArray(cascades) || cascades.length < 2) return

  const splits: number[] = []
  const bounds: number[] = []

  for (let i = 0; i < cascades.length; i += 1) {
    const rawCascade = cascades[i]
    if (!rawCascade || typeof rawCascade !== 'object') {
      throw new TypeError(`shadowCascades[${i}] must be an object with finite left, right, top, bottom, near, and far values.`)
    }
    const cascade = rawCascade as Record<string, unknown>
    const label = `shadowCascades[${i}]`
    const left = requiredFiniteNumber(cascade.left, `${label}.left`)
    const right = requiredFiniteNumber(cascade.right, `${label}.right`)
    const top = requiredFiniteNumber(cascade.top, `${label}.top`)
    const bottom = requiredFiniteNumber(cascade.bottom, `${label}.bottom`)
    const near = requiredFiniteNumber(cascade.near, `${label}.near`)
    const far = requiredFiniteNumber(cascade.far, `${label}.far`)
    if (bounds.length / 6 >= MAX_SHADOW_CASCADES) {
      throw new Error(
        `Directional shadow cascade hints support at most ${MAX_SHADOW_CASCADES} valid cascades in @headless-three/renderer. Reduce light.userData.headlessThreeRenderer.shadowCascades or render separate shadow passes.`,
      )
    }
    bounds.push(left, right, top, bottom, near, far)
    const split = shadowCascadeSplit(cascade, label)
    if (split !== undefined) splits.push(split)
  }

  const count = bounds.length / 6
  if (count >= 2) {
    out.shadowCascadeBounds = bounds
    out.shadowCascadeSplits = splits.slice(0, count - 1)
  }
}

function shadowCascadeSplit(cascade: Record<string, unknown>, label: string): number | undefined {
  if (cascade.split != null) return optionalFiniteNumber(cascade.split, `${label}.split`)
  if (cascade.distance != null) return optionalFiniteNumber(cascade.distance, `${label}.distance`)
  return optionalFiniteNumber(cascade.farDistance, `${label}.farDistance`)
}

function finiteNumberOrDefault(value: unknown, label: string, fallback: number): number {
  if (value == null) return fallback
  if (typeof value === 'number' && Number.isFinite(value)) return value
  throw new TypeError(`${label} must be a finite number.`)
}

function optionalFiniteNumber(value: unknown, label: string): number | undefined {
  if (value == null) return undefined
  if (typeof value === 'number' && Number.isFinite(value)) return value
  throw new TypeError(`${label} must be a finite number.`)
}

function requiredFiniteNumber(value: unknown, label: string): number {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  throw new TypeError(`${label} must be a finite number.`)
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
    const c = strictColorLikeToArray(light.color, 'AmbientLight.color') ?? [1, 1, 1, 1]
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
    intensity += finiteNumberOrDefault(light.intensity, 'AmbientLight.intensity', 1)
  })
  return intensity > 0 ? intensity : undefined
}

export function extractLightProbe(scene: ThreeObject3DLike, camera?: ThreeCameraLike): number[] | null {
  const coefficients = new Array<number>(27).fill(0)
  let found = false

  visitForLightProbe(scene, camera, (light) => {
    const source = light.sh?.coefficients
    if (!Array.isArray(source) || source.length < 9) return

    const intensity = finiteNumberOrDefault(light.intensity, 'LightProbe.intensity', 1)
    for (let i = 0; i < 9; i += 1) {
      const coefficient = coefficientToRgb(source[i], `LightProbe.sh.coefficients[${i}]`)
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

function coefficientToRgb(value: unknown, label: string): [number, number, number] | null {
  if (!value) return null
  const maybeArrayLike = value as { length?: unknown }
  if (Array.isArray(value) || (ArrayBuffer.isView(value) && typeof maybeArrayLike.length === 'number')) {
    const array = value as unknown as ArrayLike<number>
    return [
      requiredFiniteRgbComponent(array[0], `${label}[0]`),
      requiredFiniteRgbComponent(array[1], `${label}[1]`),
      requiredFiniteRgbComponent(array[2], `${label}[2]`),
    ]
  }
  const v = value as { r?: number; g?: number; b?: number; x?: number; y?: number; z?: number }
  if ('r' in v || 'g' in v || 'b' in v) {
    return [
      requiredFiniteRgbComponent(v.r, `${label}.r`),
      requiredFiniteRgbComponent(v.g, `${label}.g`),
      requiredFiniteRgbComponent(v.b, `${label}.b`),
    ]
  }
  if ('x' in v || 'y' in v || 'z' in v) {
    return [
      requiredFiniteRgbComponent(v.x, `${label}.x`),
      requiredFiniteRgbComponent(v.y, `${label}.y`),
      requiredFiniteRgbComponent(v.z, `${label}.z`),
    ]
  }
  return null
}

function requiredFiniteRgbComponent(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a finite number.`)
  }
  return value
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
