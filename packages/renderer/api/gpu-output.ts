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
  encoderSurface: { supported: boolean; reason: string }
  mediaFormats: GpuMediaFormatCapability[]
}

export type GpuMediaOutputFormat = 'rgba8unorm' | 'nv12-planes' | 'p010-planes'
export type GpuFramePoolOverflowPolicy = 'error' | 'drop-newest'

export interface GpuMediaFormatCapability {
  format: GpuMediaOutputFormat
  supported: boolean
  storage: 'single-texture' | 'separate-textures'
  planeFormats: string[]
  reason?: string
  colorMatrix?: 'bt709'
  colorRange?: 'limited'
  chromaSiting?: 'centered-2x2-box'
}

export interface GpuFramePoolOptions {
  width: number
  height: number
  capacity?: number
  format?: GpuMediaOutputFormat
  overflow?: GpuFramePoolOverflowPolicy
}

export interface GpuFramePoolStats {
  capacity: number
  available: number
  inFlight: number
  peakInFlight: number
  submitted: number
  completed: number
  dropped: number
  rejected: number
  reused: number
  allocations: number
  retired: number
  closed: boolean
}

export interface GpuMediaPlaneInfo {
  index: number
  format: string
  width: number
  height: number
  bytesPerRow: number
  rowSemantics: string
  expectedStateBeforeUse: string
  requiredStateOnRelease: string
}

export interface GpuMediaPlaneData extends Omit<GpuMediaPlaneInfo, 'rowSemantics' | 'expectedStateBeforeUse' | 'requiredStateOnRelease'> {
  data: Buffer
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

interface NativeGpuMediaFrameLeaseLike {
  readonly width: number
  readonly height: number
  readonly format: string
  readonly backend: string
  readonly handleType: string
  readonly sequence: number
  readonly released: boolean
  readonly ready: boolean
  planeInfo(index: number): GpuMediaPlaneInfo
  planeHandle(index: number): bigint
  readPlanes(): Promise<GpuMediaPlaneData[]>
  exportDmaBuf(): unknown
  completeExternalUse(): void
  release(): void
}

interface NativeGpuFramePoolLike {
  reserve(): object | null
  renderAsync(reservation: object, scene: unknown, camera: unknown): Promise<NativeGpuMediaFrameLeaseLike>
  stats(): GpuFramePoolStats
  close(): void
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

/** A ready, pooled media frame. Release only after all external GPU use ends. */
export class GpuMediaFrameLease {
  constructor(private readonly nativeLease: NativeGpuMediaFrameLeaseLike) {}

  get width(): number { return this.nativeLease.width }
  get height(): number { return this.nativeLease.height }
  get format(): GpuMediaOutputFormat { return this.nativeLease.format as GpuMediaOutputFormat }
  get backend(): GpuOutputCapabilities['backend'] {
    return this.nativeLease.backend as GpuOutputCapabilities['backend']
  }
  get handleType(): GpuTextureHandleType | 'unsupported' {
    return this.nativeLease.handleType as GpuTextureHandleType | 'unsupported'
  }
  get sequence(): number { return this.nativeLease.sequence }
  get released(): boolean { return this.nativeLease.released }
  get ready(): boolean { return this.nativeLease.ready }

  get planes(): GpuMediaPlaneInfo[] {
    const count = this.format === 'rgba8unorm' ? 1 : 2
    return Array.from({ length: count }, (_, index) => this.nativeLease.planeInfo(index))
  }

  /** Borrowed same-device texture handle for one physical plane. */
  planeHandle(index: number): bigint {
    return this.nativeLease.planeHandle(index)
  }

  /** Confirm external work ended and every plane is back in requiredStateOnRelease. */
  completeExternalUse(): void {
    this.nativeLease.completeExternalUse()
  }

  /** Explicit diagnostic CPU readback; the normal media path performs no readback. */
  readPlanes(): Promise<GpuMediaPlaneData[]> {
    return this.nativeLease.readPlanes()
  }

  exportDmaBuf(): DmaBufFrameLease {
    return this.nativeLease.exportDmaBuf() as DmaBufFrameLease
  }

  release(): void { this.nativeLease.release() }
}

type PoolRender = (
  scene: ThreeSceneRootLike,
  camera: ThreeCameraLike,
  options: RenderOptions,
) => Promise<GpuMediaFrameLease | null>

/** Fixed-size reusable GPU output pool with explicit overflow and lease semantics. */
export class GpuFramePool {
  constructor(
    private readonly nativePool: NativeGpuFramePoolLike,
    private readonly renderFrame: PoolRender,
    readonly options: Required<GpuFramePoolOptions>,
  ) {}

  render(
    scene: ThreeSceneRootLike,
    camera: ThreeCameraLike,
    options: RenderOptions = {},
  ): Promise<GpuMediaFrameLease | null> {
    return this.renderFrame(scene, camera, options)
  }

  /** @internal */
  async renderNative(scene: unknown, camera: unknown): Promise<GpuMediaFrameLease | null> {
    const reservation = this.nativePool.reserve()
    if (!reservation) return null
    const lease = await this.nativePool.renderAsync(reservation, scene, camera)
    return new GpuMediaFrameLease(lease)
  }

  stats(): GpuFramePoolStats { return this.nativePool.stats() }
  close(): void { this.nativePool.close() }
}

export function wrapGpuOutputCapabilities(value: unknown): GpuOutputCapabilities {
  return value as GpuOutputCapabilities
}
import type { RenderOptions, ThreeCameraLike, ThreeSceneRootLike } from './types'
