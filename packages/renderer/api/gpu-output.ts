export type GpuTextureHandleType = 'metal-texture' | 'vulkan-image' | 'd3d12-resource'

export interface GpuTextureOutputCapability {
  supported: boolean
  handleType?: GpuTextureHandleType
  synchronization: 'submission-complete'
  scope: 'same-renderer-device'
  format: 'rgba8unorm'
  usage: 'copy-dst|copy-src|texture-binding'
  layout: 'backend-managed-copy-dst'
  reason?: string
}

export interface DmaBufOutputCapability {
  supported: boolean
  reason?: string
}

export interface GpuOutputCapabilities {
  backend: 'metal' | 'vulkan' | 'dx12' | 'gles' | 'browser-webgpu' | 'noop'
  texture: GpuTextureOutputCapability
  dmaBuf: DmaBufOutputCapability
}

export interface DmaBufPlane {
  fd: number
  offset: number
  stride: number
}

export interface DmaBufFrameLease {
  width: number
  height: number
  fourcc: number
  modifier: bigint
  planes: DmaBufPlane[]
  released: boolean
  release(): void
}

interface NativeGpuFrameLeaseLike {
  readonly width: number
  readonly height: number
  readonly format: string
  readonly backend: string
  readonly handleType: string
  readonly released: boolean
  readonly completed: boolean
  readonly scope: string
  readonly usage: string
  readonly layout: string
  nativeHandle(): bigint
  exportDmaBuf(): unknown
  release(): void
}

/**
 * Owns one rendered GPU texture and controls the lifetime of its borrowed
 * native handle. Call `release()` in a `finally` block when interop completes.
 */
export class GpuFrameLease {
  private readonly nativeLease: NativeGpuFrameLeaseLike

  constructor(nativeLease: NativeGpuFrameLeaseLike) {
    this.nativeLease = nativeLease
  }

  get width(): number { return this.nativeLease.width }
  get height(): number { return this.nativeLease.height }
  get format(): 'rgba8unorm' { return this.nativeLease.format as 'rgba8unorm' }
  get backend(): GpuOutputCapabilities['backend'] {
    return this.nativeLease.backend as GpuOutputCapabilities['backend']
  }
  get handleType(): GpuTextureHandleType {
    return this.nativeLease.handleType as GpuTextureHandleType
  }
  get released(): boolean { return this.nativeLease.released }
  get completed(): true { return this.nativeLease.completed as true }
  get scope(): 'same-renderer-device' { return this.nativeLease.scope as 'same-renderer-device' }
  get usage(): 'copy-dst|copy-src|texture-binding' {
    return this.nativeLease.usage as 'copy-dst|copy-src|texture-binding'
  }
  get layout(): 'backend-managed-copy-dst' {
    return this.nativeLease.layout as 'backend-managed-copy-dst'
  }

  /**
   * Returns a borrowed MTLTexture*, VkImage, or ID3D12Resource* bigint.
   * It is valid only for integrations already using this renderer's native
   * device/context. Never destroy it, use it after release, or access/transition
   * it concurrently with renderer work.
   */
  nativeHandle(): bigint {
    return this.nativeLease.nativeHandle()
  }

  /** Export an owned DMA-BUF descriptor when `capabilities.dmaBuf.supported`. */
  exportDmaBuf(): DmaBufFrameLease {
    return this.nativeLease.exportDmaBuf() as DmaBufFrameLease
  }

  /** Idempotently invalidates the borrowed handle and drops the GPU texture. */
  release(): void {
    this.nativeLease.release()
  }
}

export function wrapGpuOutputCapabilities(value: unknown): GpuOutputCapabilities {
  return value as GpuOutputCapabilities
}
