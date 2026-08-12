# GPU media output

`Renderer.createGpuFramePool()` is the production-oriented GPU media path. It
preallocates a bounded set of output surfaces, renders on a libuv worker, and
resolves each Promise only after GPU completion. Node's main thread never calls
`Device::poll(wait_indefinitely)` on this path, and normal frame delivery does
not perform CPU readback.

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

The two YUV formats are deliberately named `*-planes`. They contain real Y and
interleaved UV samples, but they are two ordinary GPU textures—not one native
multi-planar NV12/P010 image and not directly importable by VAAPI, NVENC,
VideoToolbox, or Media Foundation. Chroma is 4:2:0, produced by a centered 2x2
box average. Width and height must be even.

P010 stores 10-bit limited-range codes in bits 15..6 of each little-endian
16-bit word; bits 5..0 are zero. The code ranges are Y 64..940 and UV 64..960
with neutral chroma 512. Support is enabled only when the adapter exposes
writable R16/RG16 normalized storage textures.

Each `frame.planes` entry reports its physical format, dimensions, tightly
packed logical `bytesPerRow`, and backend-managed native pitch semantics.
`frame.readPlanes()` is an explicit diagnostic readback API used for validation;
it is not part of the zero-readback delivery path.

## Bounded scheduling and lifetime

- Capacity defaults to three. All output surface sets are allocated when the
  pool is created; no output texture is allocated per frame after warm-up.
- A slot is reserved synchronously before async work enters libuv. With
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

## Encoder-native export blockers

`encoderSurface.supported` and `dmaBuf.supported` remain false:

- wgpu 29 exposes NV12/P010 multi-planar formats for sampling, but its public
  format contract does not permit writing their planes; its core source also
  explicitly leaves multi-planar COPY_DST unsupported.
- Vulkan DMA-BUF needs a DRM-modifier image, exportable dedicated allocation,
  plane offsets/strides/modifier, duplicated fd ownership, queue-family/layout
  transfer, and an exportable sync fd or semaphore. wgpu's safe API does not
  expose that complete allocation and state-tracker contract. Bypassing only
  part of it through raw hal handles would be unsound.
- wgpu-managed Metal textures are not IOSurface/CVPixelBuffer-backed resources.
- wgpu-managed D3D12 textures are not shared encoder resources paired with a
  shared fence.

Accordingly, `exportDmaBuf()` throws and no fd or portable encoder surface is
fabricated. Platform consumer smoke tests are not claimed; CI instead validates
real plane samples, native same-device handles, state/lifetime rules, and all
five backends' truthful capability shape.

`Renderer.renderGpuFrame()` remains as the 0.2 compatibility API for a dedicated
completed RGBA texture. It is synchronous and not the recommended real-time
path.

This work is hardware-encoding infrastructure related to profiling session
`019fec05-6b95-7fb0-863d-25033283b6a7`. It does not replace queue-capacity-one,
latest-state WebRTC scheduling and does not by itself fix that queueing delay.
