import type { ThreeCameraLike, ThreeLayersLike, ThreeObject3DLike } from './types'

const DEFAULT_LAYER_MASK = 1

export function objectLayersMatchCamera(
  object: ThreeObject3DLike,
  camera?: ThreeCameraLike,
): boolean {
  const cameraLayers = camera?.layers
  if (!cameraLayers) return true

  const objectLayers = object.layers
  layerMask(cameraLayers, 'camera.layers')
  layerMask(objectLayers, 'object.layers')
  if (typeof objectLayers?.test === 'function') {
    return objectLayers.test(cameraLayers)
  }

  return (layerMask(objectLayers, 'object.layers') & layerMask(cameraLayers, 'camera.layers')) !== 0
}

function layerMask(layers: ThreeLayersLike | undefined, label: string): number {
  if (layers == null) return DEFAULT_LAYER_MASK
  if (typeof layers !== 'object' || Array.isArray(layers)) {
    throw new TypeError(`${label} must be a layers-like object.`)
  }
  if (layers.mask == null) return DEFAULT_LAYER_MASK
  if (typeof layers.mask === 'number' && Number.isFinite(layers.mask)) return layers.mask
  throw new TypeError(`${label}.mask must be a finite number.`)
}
