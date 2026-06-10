import type { ThreeCameraLike, ThreeLayersLike, ThreeObject3DLike } from './types'

const DEFAULT_LAYER_MASK = 1

export function objectLayersMatchCamera(
  object: ThreeObject3DLike,
  camera?: ThreeCameraLike,
): boolean {
  const cameraLayers = camera?.layers
  if (!cameraLayers) return true

  const objectLayers = object.layers
  if (typeof objectLayers?.test === 'function') {
    return objectLayers.test(cameraLayers)
  }

  return (layerMask(objectLayers) & layerMask(cameraLayers)) !== 0
}

function layerMask(layers: ThreeLayersLike | undefined): number {
  return layers?.mask == null ? DEFAULT_LAYER_MASK : layers.mask
}
