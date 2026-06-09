import type { ThreeObject3DLike, NativeSceneLight } from './types'
import { colorLikeToArray } from './color'

export function extractLights(scene: ThreeObject3DLike): NativeSceneLight[] | undefined {
  const lights: NativeSceneLight[] = []
  visitLights(scene, lights)
  return lights.length > 0 ? lights : undefined
}

function visitLights(object: ThreeObject3DLike, lights: NativeSceneLight[]): void {
  if (!object) return
  if (object.visible === false) return

  if (object.isLight === true) {
    const light = extractLight(object)
    if (light) lights.push(light)
  }

  const children = Array.isArray(object.children) ? object.children : []
  for (const child of children) {
    visitLights(child, lights)
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
      const shadow = light.shadow
      const mapSize = shadow?.mapSize
      const size = Math.max(
        32,
        Math.floor(mapSize?.x ?? mapSize?.width ?? 512),
      )
      const cam = shadow?.camera
      out.castShadow = true
      out.shadowMapSize = size
      if (Number.isFinite(shadow?.bias)) out.shadowBias = shadow!.bias!
      if (Number.isFinite(shadow?.normalBias)) out.shadowNormalBias = shadow!.normalBias!
      if (cam) {
        if (Number.isFinite(cam.left)) out.shadowCameraLeft = cam.left!
        if (Number.isFinite(cam.right)) out.shadowCameraRight = cam.right!
        if (Number.isFinite(cam.top)) out.shadowCameraTop = cam.top!
        if (Number.isFinite(cam.bottom)) out.shadowCameraBottom = cam.bottom!
        if (Number.isFinite(cam.near)) out.shadowCameraNear = cam.near!
        if (Number.isFinite(cam.far)) out.shadowCameraFar = cam.far!
      }
    }
    return out
  }

  if (light.isPointLight === true) {
    const pos = light.matrixWorld
      ? [light.matrixWorld.elements[12], light.matrixWorld.elements[13], light.matrixWorld.elements[14]]
      : [0, 0, 0]
    return {
      lightType: 'point',
      color: [color[0], color[1], color[2]],
      intensity,
      position: pos,
      distance: Number.isFinite(light.distance) ? light.distance! : 0,
      decay: Number.isFinite(light.decay) ? light.decay! : 2,
    }
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
    return {
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

export function extractAmbientLight(scene: ThreeObject3DLike): number[] | null {
  let color: number[] | null = null
  visitForAmbient(scene, (light) => {
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

export function extractAmbientIntensity(scene: ThreeObject3DLike): number | undefined {
  let intensity = 0
  visitForAmbient(scene, (light) => {
    intensity += Number.isFinite(light.intensity) ? light.intensity! : 1
  })
  return intensity > 0 ? intensity : undefined
}

function visitForAmbient(object: ThreeObject3DLike, callback: (light: ThreeObject3DLike) => void): void {
  if (!object) return
  if (object.visible === false) return
  if (object.isAmbientLight === true) callback(object)
  const children = Array.isArray(object.children) ? object.children : []
  for (const child of children) {
    visitForAmbient(child, callback)
  }
}
