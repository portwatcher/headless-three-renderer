import type { ThreeCameraLike, ThreeObject3DLike, NativeSceneLight } from './types'
import { validatedColorLikeToArray } from './color'
import { objectLayersMatchCamera } from './layers'
import { matrixElements } from './math'
import { objectChildren } from './objects'

type ShadowMapSizeLike = { x?: number; y?: number; width?: number; height?: number } | undefined
type ShadowCameraLike = NonNullable<NonNullable<ThreeObject3DLike['shadow']>['camera']>
const MAX_NATIVE_LIGHTS = 64
const MAX_SHADOW_CASCADES = 4

export function extractLights(scene: ThreeObject3DLike, camera?: ThreeCameraLike): NativeSceneLight[] | undefined {
  const lights: NativeSceneLight[] = []
  visitLights(scene, camera, lights)
  assertSupportedLightCount(lights)
  return lights.length > 0 ? lights : undefined
}

function visitLights(object: ThreeObject3DLike, camera: ThreeCameraLike | undefined, lights: NativeSceneLight[]): void {
  if (!object) return
  if (optionalBoolean(object.visible, 'object.visible') === false) return

  if (object.isLight === true && objectLayersMatchCamera(object, camera)) {
    const light = extractLight(object)
    if (light) lights.push(light)
  }

  for (const child of objectChildren(object)) {
    visitLights(child, camera, lights)
  }
}

function extractLight(light: ThreeObject3DLike): NativeSceneLight | null {
  const color = validatedColorLikeToArray(light.color, 'light.color') ?? [1, 1, 1, 1]
  const intensity = finiteNumberOrDefault(light.intensity, 'light.intensity', 1)

  if (light.isDirectionalLight === true) {
    const pos = positionFromMatrix(light.matrixWorld, 'DirectionalLight.matrixWorld', [0, 10, 0])
    const targetPos = positionFromMatrix(lightTargetMatrix(light, 'DirectionalLight.target'), 'DirectionalLight.target.matrixWorld', [0, 0, 0])
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
    if (optionalBoolean(light.castShadow, 'light.castShadow') === true) {
      out.castShadow = true
      applyShadowOptions(out, light)
    }
    return out
  }

  if (light.isPointLight === true) {
    const pos = positionFromMatrix(light.matrixWorld, 'PointLight.matrixWorld', [0, 0, 0])
    const out: NativeSceneLight = {
      lightType: 'point',
      color: [color[0], color[1], color[2]],
      intensity,
      position: pos,
      distance: nonNegativeFiniteNumberOrDefault(light.distance, 'PointLight.distance', 0),
      decay: nonNegativeFiniteNumberOrDefault(light.decay, 'PointLight.decay', 2),
    }
    if (optionalBoolean(light.castShadow, 'light.castShadow') === true) {
      out.castShadow = true
      applyShadowOptions(out, light)
    }
    return out
  }

  if (light.isSpotLight === true) {
    const pos = positionFromMatrix(light.matrixWorld, 'SpotLight.matrixWorld', [0, 0, 0])
    const targetPos = positionFromMatrix(lightTargetMatrix(light, 'SpotLight.target'), 'SpotLight.target.matrixWorld', [0, 0, 0])
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
      distance: nonNegativeFiniteNumberOrDefault(light.distance, 'SpotLight.distance', 0),
      decay: nonNegativeFiniteNumberOrDefault(light.decay, 'SpotLight.decay', 2),
      angle: spotAngleOrDefault(light.angle),
      penumbra: normalizedFiniteNumberOrDefault(light.penumbra, 'SpotLight.penumbra', 0),
    }
    if (optionalBoolean(light.castShadow, 'light.castShadow') === true) {
      out.castShadow = true
      applyShadowOptions(out, light)
    }
    return out
  }

  if (light.isRectAreaLight === true) {
    const matrix = matrixElementsOrUndefined(light.matrixWorld, 'RectAreaLight.matrixWorld')
    const pos = matrix ? [matrix[12], matrix[13], matrix[14]] : [0, 0, 0]
    let direction = [0, 0, -1]
    if (matrix) {
      direction = [-matrix[8], -matrix[9], -matrix[10]]
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
      width: positiveFiniteNumberOrDefault(light.width, 'RectAreaLight.width', 10),
      height: positiveFiniteNumberOrDefault(light.height, 'RectAreaLight.height', 10),
    }
  }

  if (light.isHemisphereLight === true) {
    const groundColor = validatedColorLikeToArray(light.groundColor, 'HemisphereLight.groundColor') ?? [0.04, 0.02, 0.0, 1]
    let direction = [0, 1, 0]
    const matrix = matrixElementsOrUndefined(light.matrixWorld, 'HemisphereLight.matrixWorld')
    if (matrix) {
      const ux = matrix[4], uy = matrix[5], uz = matrix[6]
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

function positionFromMatrix(
  matrix: ThreeObject3DLike['matrixWorld'] | undefined,
  label: string,
  fallback: number[],
): number[] {
  const elements = matrixElementsOrUndefined(matrix, label)
  return elements ? [elements[12], elements[13], elements[14]] : fallback
}

function matrixElementsOrUndefined(
  matrix: ThreeObject3DLike['matrixWorld'] | undefined,
  label: string,
) {
  return matrix ? matrixElements(matrix, label) : undefined
}

function lightTargetMatrix(light: ThreeObject3DLike, label: string): ThreeObject3DLike['matrixWorld'] | undefined {
  const target = light.target
  if (target == null) return undefined
  if (typeof target !== 'object' || Array.isArray(target)) {
    throw new TypeError(`${label} must be an object.`)
  }
  return target.matrixWorld
}

function applyShadowOptions(out: NativeSceneLight, light: ThreeObject3DLike): void {
  const shadow = light.shadow
  assertShadowContainerLike(shadow)
  const mapSize = shadowMapSizeOrDefault(shadow?.mapSize, light)
  out.shadowMapSize = Math.max(mapSize.width, mapSize.height)
  out.shadowMapWidth = mapSize.width
  out.shadowMapHeight = mapSize.height
  const bias = optionalFiniteNumber(shadow?.bias, 'light.shadow.bias')
  const normalBias = optionalFiniteNumber(shadow?.normalBias, 'light.shadow.normalBias')
  const radius = optionalNonNegativeFiniteNumber(shadow?.radius, 'light.shadow.radius')
  const blurSamples = optionalNonNegativeFiniteNumber(shadow?.blurSamples, 'light.shadow.blurSamples')
  if (bias !== undefined) out.shadowBias = bias
  if (normalBias !== undefined) out.shadowNormalBias = normalBias
  if (radius !== undefined) out.shadowRadius = radius
  if (blurSamples !== undefined) out.shadowBlurSamples = blurSamples

  const cam = shadow?.camera
  if (cam != null) {
    assertPlainObject(cam, 'light.shadow.camera')
    const bounds = shadowCameraOrthoBounds(cam, out)
    const clipDistances = shadowCameraClipDistances(cam, out)
    if (bounds.shadowCameraLeft !== undefined) out.shadowCameraLeft = bounds.shadowCameraLeft
    if (bounds.shadowCameraRight !== undefined) out.shadowCameraRight = bounds.shadowCameraRight
    if (bounds.shadowCameraTop !== undefined) out.shadowCameraTop = bounds.shadowCameraTop
    if (bounds.shadowCameraBottom !== undefined) out.shadowCameraBottom = bounds.shadowCameraBottom
    if (clipDistances.shadowCameraNear !== undefined) out.shadowCameraNear = clipDistances.shadowCameraNear
    if (clipDistances.shadowCameraFar !== undefined) out.shadowCameraFar = clipDistances.shadowCameraFar
  }

  applyShadowCascadeOptions(out, light)
}

function shadowCameraClipDistances(
  camera: ShadowCameraLike,
  light: NativeSceneLight,
): Pick<NativeSceneLight, 'shadowCameraNear' | 'shadowCameraFar'> {
  const near = optionalFiniteNumber(camera.near, 'light.shadow.camera.near')
  const far = optionalFiniteNumber(camera.far, 'light.shadow.camera.far')

  if (near !== undefined && near < 0) {
    throw new TypeError('light.shadow.camera.near must be non-negative.')
  }
  if (near === 0 && (light.lightType === 'point' || light.lightType === 'spot')) {
    throw new TypeError('light.shadow.camera.near must be positive for point and spot shadows.')
  }
  if (far !== undefined && far <= 0) {
    throw new TypeError('light.shadow.camera.far must be positive.')
  }

  const effectiveNear = near ?? 0.5
  const effectiveFar = far ?? defaultShadowCameraFar(light)
  if (effectiveFar <= effectiveNear) {
    if (far !== undefined) {
      throw new TypeError('light.shadow.camera.far must be greater than light.shadow.camera.near.')
    }
    throw new TypeError('light.shadow.camera.near must be less than the effective light.shadow.camera.far.')
  }

  return { shadowCameraNear: near, shadowCameraFar: far }
}

function defaultShadowCameraFar(light: NativeSceneLight): number {
  return light.lightType === 'point' && light.distance !== undefined && light.distance > 0
    ? light.distance
    : 500
}

function shadowCameraOrthoBounds(
  camera: ShadowCameraLike,
  light: NativeSceneLight,
): Pick<NativeSceneLight, 'shadowCameraLeft' | 'shadowCameraRight' | 'shadowCameraTop' | 'shadowCameraBottom'> {
  const left = optionalFiniteNumber(camera.left, 'light.shadow.camera.left')
  const right = optionalFiniteNumber(camera.right, 'light.shadow.camera.right')
  const top = optionalFiniteNumber(camera.top, 'light.shadow.camera.top')
  const bottom = optionalFiniteNumber(camera.bottom, 'light.shadow.camera.bottom')

  if (light.lightType === 'directional') {
    const effectiveLeft = left ?? -5
    const effectiveRight = right ?? 5
    const effectiveTop = top ?? 5
    const effectiveBottom = bottom ?? -5

    if (effectiveRight <= effectiveLeft) {
      if (right !== undefined) {
        throw new TypeError('light.shadow.camera.right must be greater than light.shadow.camera.left.')
      }
      throw new TypeError('light.shadow.camera.left must be less than the effective light.shadow.camera.right.')
    }
    if (effectiveTop <= effectiveBottom) {
      if (top !== undefined) {
        throw new TypeError('light.shadow.camera.top must be greater than light.shadow.camera.bottom.')
      }
      throw new TypeError('light.shadow.camera.bottom must be less than the effective light.shadow.camera.top.')
    }
  }

  return {
    shadowCameraLeft: left,
    shadowCameraRight: right,
    shadowCameraTop: top,
    shadowCameraBottom: bottom,
  }
}

function shadowMapSizeOrDefault(mapSize: ShadowMapSizeLike, light: ThreeObject3DLike): { width: number; height: number } {
  if (mapSize != null) {
    assertPlainObject(mapSize, 'light.shadow.mapSize')
  }
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

function assertShadowContainerLike(shadow: unknown): void {
  if (shadow == null) return
  assertPlainObject(shadow, 'light.shadow')
}

function assertPlainObject(value: unknown, label: string): void {
  if (typeof value === 'object' && !Array.isArray(value)) return
  throw new TypeError(`${label} must be an object.`)
}

function shadowMapSizeComponent(
  mapSize: ShadowMapSizeLike,
  vectorKey: 'x' | 'y',
  dimensionKey: 'width' | 'height',
): number | undefined {
  if (!mapSize) return undefined
  const vectorValue = mapSize[vectorKey]
  if (vectorValue != null) return optionalPositiveFiniteNumber(vectorValue, `light.shadow.mapSize.${vectorKey}`)
  return optionalPositiveFiniteNumber(mapSize[dimensionKey], `light.shadow.mapSize.${dimensionKey}`)
}

function applyShadowCascadeOptions(out: NativeSceneLight, light: ThreeObject3DLike): void {
  const cascadeHints = shadowCascadeHints(light)
  if (!cascadeHints) return
  const { value: cascades, label } = cascadeHints
  if (!Array.isArray(cascades)) {
    throw new TypeError(`${label} must be an array of shadow cascade hint objects.`)
  }
  if (cascades.length < 2) return

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

function shadowCascadeHints(light: ThreeObject3DLike): { value: unknown; label: string } | null {
  const userData = light.userData
  if (userData != null && (typeof userData !== 'object' || Array.isArray(userData))) {
    throw new TypeError('light.userData must be an object.')
  }

  const modernHints = userData?.headlessThreeRenderer
  if (modernHints != null) {
    assertPlainObject(modernHints, 'light.userData.headlessThreeRenderer')
    if (modernHints.shadowCascades != null) {
      return {
        value: modernHints.shadowCascades,
        label: 'light.userData.headlessThreeRenderer.shadowCascades',
      }
    }
    if (modernHints.cascades != null) {
      return {
        value: modernHints.cascades,
        label: 'light.userData.headlessThreeRenderer.cascades',
      }
    }
  }

  const legacyHints = userData?.headlessRenderer
  if (legacyHints != null) {
    assertPlainObject(legacyHints, 'light.userData.headlessRenderer')
    if (legacyHints.shadowCascades != null) {
      return {
        value: legacyHints.shadowCascades,
        label: 'light.userData.headlessRenderer.shadowCascades',
      }
    }
    if (legacyHints.cascades != null) {
      return {
        value: legacyHints.cascades,
        label: 'light.userData.headlessRenderer.cascades',
      }
    }
  }

  const shadowCascades = (light.shadow as any)?.cascades
  return shadowCascades != null
    ? { value: shadowCascades, label: 'light.shadow.cascades' }
    : null
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

function nonNegativeFiniteNumberOrDefault(value: unknown, label: string, fallback: number): number {
  const number = finiteNumberOrDefault(value, label, fallback)
  if (number < 0) {
    throw new TypeError(`${label} must be non-negative.`)
  }
  return number
}

function positiveFiniteNumberOrDefault(value: unknown, label: string, fallback: number): number {
  const number = finiteNumberOrDefault(value, label, fallback)
  if (number <= 0) {
    throw new TypeError(`${label} must be positive.`)
  }
  return number
}

function normalizedFiniteNumberOrDefault(value: unknown, label: string, fallback: number): number {
  const number = finiteNumberOrDefault(value, label, fallback)
  if (number < 0 || number > 1) {
    throw new TypeError(`${label} must be between 0 and 1.`)
  }
  return number
}

function spotAngleOrDefault(value: unknown): number {
  const angle = finiteNumberOrDefault(value, 'SpotLight.angle', Math.PI / 3)
  if (angle < 0 || angle > Math.PI / 2) {
    throw new TypeError('SpotLight.angle must be between 0 and Math.PI / 2.')
  }
  return angle
}

function optionalFiniteNumber(value: unknown, label: string): number | undefined {
  if (value == null) return undefined
  if (typeof value === 'number' && Number.isFinite(value)) return value
  throw new TypeError(`${label} must be a finite number.`)
}

function optionalNonNegativeFiniteNumber(value: unknown, label: string): number | undefined {
  const number = optionalFiniteNumber(value, label)
  if (number === undefined) return undefined
  if (number < 0) {
    throw new TypeError(`${label} must be non-negative.`)
  }
  return number
}

function optionalBoolean(value: unknown, label: string): boolean | undefined {
  if (value == null) return undefined
  if (typeof value === 'boolean') return value
  throw new TypeError(`${label} must be a boolean.`)
}

function optionalPositiveFiniteNumber(value: unknown, label: string): number | undefined {
  const number = optionalFiniteNumber(value, label)
  if (number === undefined) return undefined
  if (number <= 0) {
    throw new TypeError(`${label} must be positive.`)
  }
  return number
}

function requiredFiniteNumber(value: unknown, label: string): number {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  throw new TypeError(`${label} must be a finite number.`)
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
    const c = validatedColorLikeToArray(light.color, 'AmbientLight.color') ?? [1, 1, 1, 1]
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
    if (source == null) return
    if (!Array.isArray(source)) {
      throw new TypeError('LightProbe.sh.coefficients must be an array of 9 coefficients.')
    }
    if (source.length < 9) {
      throw new TypeError('LightProbe.sh.coefficients must contain 9 coefficients.')
    }

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
  if (optionalBoolean(object.visible, 'object.visible') === false) return
  if (object.isLightProbe === true && objectLayersMatchCamera(object, camera)) callback(object)
  for (const child of objectChildren(object)) {
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
  if (optionalBoolean(object.visible, 'object.visible') === false) return
  if (object.isAmbientLight === true && objectLayersMatchCamera(object, camera)) callback(object)
  for (const child of objectChildren(object)) {
    visitForAmbient(child, camera, callback)
  }
}
