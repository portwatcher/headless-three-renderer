# GPU media output

`Renderer.createGpuFramePool()` is the production-oriented GPU media path. It
preallocates a bounded set of output surfaces and reserves a slot before scene
extraction. Accepted work runs on one renderer-owned media thread, outside
Node's shared libuv worker pool. Render and conversion share one command buffer
and queue submission, and the worker waits once for that exact submission.
Node's main thread and libuv workers never call the blocking `Device::poll()` on
this path. Normal GPU-frame delivery performs no CPU readback.

```js
const pool = renderer.createGpuFramePool({
  width: 1920,
  height: 1080,
  capacity: 3,
  format: 'nv12-planes',
  overflow: 'drop-newest',
})

const frame = await pool.render(scene, camera)
if (frame) {
  try {
    consumeOnRendererDevice(frame)
    await externalGpuWork
    restoreEveryPlaneTo(frame.planes.map((plane) => plane.requiredStateOnRelease))
    frame.completeExternalUse()
  } finally {
    frame.release()
  }
}
```

## Formats and capability negotiation

Always inspect `renderer.getGpuOutputCapabilities().mediaFormats`.

| Format | Physical storage | Conversion | Encoder-native export |
| --- | --- | --- | --- |
| `rgba8unorm` | One reusable RGBA texture | Renderer output | No |
| `nv12-planes` | Separate `r8unorm-y` and `rg8unorm-uv` textures | GPU compute, BT.709 limited range | No |
| `p010-planes` | Separate `r16unorm-y10-msb` and `rg16unorm-uv10-msb` textures | GPU compute, BT.709 limited range | No |
| `i420-planes` | Separate `r8unorm-y`, `r8unorm-u`, and `r8unorm-v` textures plus a packed buffer | GPU compute, BT.601 limited range | CPU fallback only |

The NV12/P010 formats are deliberately named `*-planes`. They contain real Y and
interleaved UV samples, but they are two ordinary GPU textures—not one native
multi-planar NV12/P010 image and not directly importable by VAAPI, NVENC,
VideoToolbox, or Media Foundation. Chroma is 4:2:0, produced by a centered 2x2
box average. Width and height must be even.

P010 stores 10-bit limited-range codes in bits 15..6 of each little-endian
16-bit word; bits 5..0 are zero. The code ranges are Y 64..940 and UV 64..960
with neutral chroma 512. Support is enabled only when the adapter exposes
writable R16/RG16 normalized storage textures.

## Packed CPU I420 fallback

`i420-planes` exists for CPU-only consumers such as
`@roamhq/wrtc` 0.10.0 `RTCVideoSource.onFrame()`. `renderI420()` performs RGBA
to planar I420 conversion and tight Y/U/V packing on the GPU, then reads back
only the packed payload: 1.5 bytes per pixel for the required even dimensions.
It does not read back RGBA and does not perform CPU color conversion.

```js
const width = 1920
const height = 1080
const target = Buffer.allocUnsafeSlow(width * height * 1.5)
const pool = renderer.createGpuFramePool({
  width,
  height,
  capacity: 3,
  format: 'i420-planes',
  overflow: 'drop-newest',
})

const frame = await pool.renderI420(scene, camera, {}, target)
if (frame) {
  videoSource.onFrame(frame)
  // @roamhq/wrtc copies synchronously, so target can be reused after return.
}
```

The optional target avoids a renderer-side output allocation and is returned
with the same JS identity. It must be an exact standalone `Buffer`: byte offset
zero, backing `ArrayBuffer` length equal to view length, and exact I420 length.
This restriction avoids a `@roamhq/wrtc` binding behavior that ignores typed
array offsets and view lengths. The default renderer-created buffer also meets
those invariants. A CPU I420 result releases its pool slot before its Promise
resolves; there is no external GPU lease to release.
The same target cannot be submitted again until its current Promise resolves;
concurrent reuse rejects instead of allowing the media worker to overwrite a
frame before JavaScript consumes it.

The packed layout is tight Y, then U, then V with strides `[width, width / 2,
width / 2]`. Color is BT.601 limited range with centered 2x2 chroma, matching
the libyuv conversion convention used by `@roamhq/wrtc`. That consumer provides
no color-space metadata and still allocates and copies into its internal
WebRTC I420 buffer. Eliminating that final copy requires an upstream consumer
API. `gpuReadbackBytes` reports the at-most-three-byte tail padding needed for
the GPU buffer copy; `byteLength` and `data.length` remain exactly 1.5 B/pixel.

Each `frame.planes` entry reports its physical format, dimensions, tightly
packed logical `bytesPerRow`, and backend-managed native pitch semantics.
`frame.readPlanes()` is an explicit diagnostic readback API used for validation;
it is not part of the zero-readback delivery path.

## Bounded scheduling and lifetime

- Capacity defaults to three. All output surface sets are allocated when the
  pool is created; no output texture is allocated per frame after warm-up.
- A slot is reserved synchronously before scene extraction or async work. With
  `drop-newest`, an over-capacity submission immediately resolves to `null`.
  With `error`, it rejects immediately. Dropped work cannot later reappear as a
  stale frame.
- `stats()` reports configured capacity, available/in-flight/peak counts,
  submitted/completed/dropped/rejected frames, reuse, fixed surface allocation
  count, and safety-retired slots.
- `close()` is idempotent and rejects new reservations. Existing work and leases
  remain valid and release normally.
- A resolved lease is ready (`ready === true`). `release()` is idempotent unless
  raw external use is still pending; use-after-release operations throw.
- All pools owned by one renderer share one media thread. This bounds blocking
  OS threads and preserves safe reuse of renderer scratch resources. Each
  pool's own capacity remains the bound on its accepted queued work, and the
  renderer-wide dispatch channel rejects above 64 pending jobs rather than
  accumulating unbounded scene payloads.

## Raw handle and resource-state contract

`planeHandle(index)` returns a borrowed `bigint` for the physical plane: an
`MTLTexture*`, `VkImage`, or `ID3D12Resource*`. It is process-local and valid
only on the renderer's device/context. A Vulkan image by itself is not portable
and cannot be used with another device or queue.

Calling `planeHandle()` marks external use pending. The caller must:

1. start only after the render Promise resolves;
2. never destroy the resource;
3. serialize work with the renderer's device/queue;
4. finish all external work;
5. restore every plane to its exact `requiredStateOnRelease` value; and
6. call `completeExternalUse()` before `release()`.

For Vulkan, pooled RGBA ends in `VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL`, writable
Y/UV planes end in `VK_IMAGE_LAYOUT_GENERAL`, and diagnostic readback changes
the current state to `VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL`. D3D12 reports the
corresponding COPY_DEST, UNORDERED_ACCESS, or COPY_SOURCE state. Metal reports
its untracked resource contract and requires same-command-queue ordering.
`expectedStateBeforeUse` and `requiredStateOnRelease` always expose the current
contract.

Premature release throws and preserves the live lease. If a pending external
handle is garbage-collected without acknowledgement, its slot is permanently
retired rather than unsafely reused.

## Encoder-native export boundary

`encoderSurface.supported` and `dmaBuf.supported` remain false. No provisional
`VkImage` or fd is returned.

- wgpu 29 exposes NV12/P010 multi-planar formats for sampling, but its public
  format contract does not permit writing their planes; its core source also
  explicitly leaves multi-planar COPY_DST unsupported.
- Ordinary wgpu Vulkan textures are optimal-layout allocator suballocations
  created without external-image/allocation chains. wgpu's state tracker also
  cannot represent `VK_QUEUE_FAMILY_FOREIGN_EXT`; its barriers use ignored queue
  families. Exporting one of those images as DMA-BUF would be unsound.
- On Linux Vulkan, this release uses wgpu-hal's supported single-device escape
  hatch to enable external-memory-fd, DMA-BUF, DRM-modifier,
  queue-family-foreign, and external-semaphore-fd extensions when all are
  advertised. `encoderSurface.prerequisitesReady` reports that bootstrap and
  `encoderSurface.prerequisites` explains its status. Readiness is not surface
  support.
- wgpu-managed Metal textures are not IOSurface/CVPixelBuffer-backed resources.
- wgpu-managed D3D12 textures are not shared encoder resources paired with a
  shared fence.

The remaining Linux AMD path must allocate a real VAAPI NV12/P010 encoding
surface on the matching `/dev/dri` render node, constrained to Vulkan-importable
modifiers. It must export `DRM_PRIME_2`, import every object/modifier/plane/aux
layout exactly into a raw-only Vulkan image, acquire from and release to the
foreign queue family, bridge producer readiness through sync-file/DMA-BUF
implicit synchronization, and hold each pooled slot until `vaSyncSurface()` or
equivalent consumer completion. wgpu-hal exposes a next-submit signal semaphore
hook but no symmetric external wait-semaphore hook, so fully GPU-side consumer
to-renderer reuse remains an upstream boundary; bounded host/VA completion is
still sound.

Support must not flip until a matched AMD amdgpu/VCN system produces a valid
H.264/HEVC bitstream, software-decodes known colors, survives thousands of
pooled drop/reuse frames without early reuse, and keeps `/proc/self/fd` stable.
On the production AMD host, a production-equivalent container has already
confirmed the Vulkan prerequisite bootstrap, all GPU media tests, and the
packed-I420 performance/resource contract. That does not validate a VA-created
surface import: the end-to-end external-surface encode, decode, synchronization,
lifetime, and fd-leak test above remains the exact platform/upstream blocker.

Relevant primary contracts include wgpu-hal's
[`open_with_callback`](https://github.com/gfx-rs/wgpu/blob/v29.0.1/wgpu-hal/src/vulkan/adapter.rs),
the Linux kernel's [DMA-BUF sync-file bridge](https://docs.kernel.org/driver-api/dma-buf.html),
Vulkan [foreign queue ownership](https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_queue_family_foreign.html),
and libva's [DRM PRIME descriptor](https://github.com/intel/libva/blob/master/va/va_drmcommon.h).

`Renderer.renderGpuFrame()` remains as the 0.2 compatibility API for a dedicated
completed RGBA texture. It is synchronous and not the recommended real-time
path.

## Reproduce the media measurements

`pnpm -C packages/renderer benchmark:media` compares legacy RGBA readback with
packed I420 and includes a libuv PBKDF2 probe. With `@roamhq/wrtc` installed,
`pnpm -C packages/renderer benchmark:wrtc` compares the full legacy
RGBA/libyuv/`onFrame` path with packed I420/`onFrame`. `WRTC_MODULE_PATH` can
point at an unpacked module for the optional consumer benchmark and integration
test.
