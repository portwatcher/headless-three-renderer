import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { ConfigureGltfLoader, GltfLoaderModule, LoadGltfFromFileOptions, NodeGltfLoaderOptions, ThreeGltfLoaderLike, ThreeLoadingManagerLike, VrmAnimationClipFactory, VrmLoaderPluginConstructor } from './loaders.part-001'
export function optionalBoolean(value: unknown, label: string): boolean | undefined {
  if (value == null) return undefined
  if (typeof value === 'boolean') return value
  throw new TypeError(`${label} must be a boolean.`)
}

export function optionalNonNegativeNumber(value: unknown, label: string): number | undefined {
  if (value == null) return undefined
  if (typeof value === 'number' && Number.isFinite(value) && value >= 0) return value
  throw new TypeError(`${label} must be a finite non-negative number.`)
}

export function optionalNonNegativeInteger(value: unknown, label: string): number | undefined {
  if (value == null) return undefined
  if (typeof value === 'number' && Number.isInteger(value) && value >= 0) return value
  throw new TypeError(`${label} must be a non-negative integer.`)
}

export function validateNodeGltfLoaderOptions(options: NodeGltfLoaderOptions): {
  configureLoader?: ConfigureGltfLoader
  installFetch?: boolean
  manager?: ThreeLoadingManagerLike
  registerTextureHandlers?: boolean
} {
  return {
    configureLoader: optionalFunction(options.configureLoader, 'options.configureLoader'),
    installFetch: optionalBoolean(options.installFetch, 'options.installFetch'),
    manager: optionalLoadingManager(options.manager, 'options.manager'),
    registerTextureHandlers: optionalBoolean(options.registerTextureHandlers, 'options.registerTextureHandlers'),
  }
}

export function validateLoadGltfFromFileOptions(options: LoadGltfFromFileOptions): void {
  validateNodeGltfLoaderOptions(options)
  optionalString(options.rootDir, 'options.rootDir')
  optionalString(options.baseUrl, 'options.baseUrl')
}

export function requiredString(value: unknown, label: string): string {
  if (typeof value === 'string') return value
  throw new TypeError(`${label} must be a string.`)
}

export function optionalString(value: unknown, label: string): string | undefined {
  if (value == null) return undefined
  return requiredString(value, label)
}

export function objectOptions(value: unknown, label: string): Record<string, unknown> {
  if (value != null && typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>
  }
  throw new TypeError(`${label} must be an object.`)
}

export function optionalFunction<T>(value: T | null | undefined, label: string): T | undefined {
  if (value == null) return undefined
  if (typeof value === 'function') return value
  throw new TypeError(`${label} must be a function.`)
}

export function optionalLoadingManager(value: ThreeLoadingManagerLike | null | undefined, label: string): ThreeLoadingManagerLike | undefined {
  if (value == null) return undefined
  if (typeof (value as any).addHandler !== 'function') {
    throw new TypeError(`${label} must provide an addHandler() function.`)
  }
  for (const method of ['itemStart', 'itemEnd', 'itemError', 'resolveURL']) {
    if ((value as any)[method] != null && typeof (value as any)[method] !== 'function') {
      throw new TypeError(`${label}.${method} must be a function when provided.`)
    }
  }
  return value
}

export function registerLoaderPlugin(
  loader: ThreeGltfLoaderLike,
  callback: (parser: unknown) => unknown,
  label: string,
): void {
  if (typeof loader.register !== 'function') {
    throw new Error(`GLTFLoader.register is required to install ${label}.`)
  }
  loader.register(callback)
}

export const dynamicImport = new Function('specifier', 'return import(specifier)') as <T = unknown>(
  specifier: string,
) => Promise<T>

export function importGltfLoader(): Promise<GltfLoaderModule> {
  return dynamicImport<GltfLoaderModule>('three/examples/jsm/loaders/GLTFLoader.js')
}

export async function importVrmLoaderPlugin(): Promise<VrmLoaderPluginConstructor> {
  let module: any
  try {
    module = await dynamicImport('@pixiv/three-vrm')
  } catch {
    throw new Error('Missing optional dependency @pixiv/three-vrm. Install it in your project or pass VRMLoaderPlugin to loadVrmFromFile().')
  }

  if (typeof module.VRMLoaderPlugin !== 'function') {
    throw new Error('@pixiv/three-vrm did not export VRMLoaderPlugin.')
  }
  return module.VRMLoaderPlugin as VrmLoaderPluginConstructor
}

export async function importVrmAnimationLoaderPlugin(): Promise<VrmLoaderPluginConstructor> {
  let module: any
  try {
    module = await dynamicImport('@pixiv/three-vrm-animation')
  } catch {
    throw new Error('Missing optional dependency @pixiv/three-vrm-animation. Install it in your project or pass VRMAnimationLoaderPlugin to loadVrmAnimationFromFile().')
  }

  if (typeof module.VRMAnimationLoaderPlugin !== 'function') {
    throw new Error('@pixiv/three-vrm-animation did not export VRMAnimationLoaderPlugin.')
  }
  return module.VRMAnimationLoaderPlugin as VrmLoaderPluginConstructor
}

export async function importVrmAnimationClipFactory(): Promise<VrmAnimationClipFactory> {
  let module: any
  try {
    module = await dynamicImport('@pixiv/three-vrm-animation')
  } catch {
    throw new Error('Missing optional dependency @pixiv/three-vrm-animation. Install it in your project or pass createVRMAnimationClip to applyVrmAnimation().')
  }

  if (typeof module.createVRMAnimationClip !== 'function') {
    throw new Error('@pixiv/three-vrm-animation did not export createVRMAnimationClip.')
  }
  return module.createVRMAnimationClip as VrmAnimationClipFactory
}
