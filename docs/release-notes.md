# Release Notes

## 0.4.0

- Moved pooled render, conversion completion, and diagnostic readback waits off
  N-API/libuv async work onto one renderer-owned media thread. Pooled render and
  conversion now wait once for the exact final submission instead of creating
  two global completion bubbles. Synchronous reservation also occurs before
  JS scene extraction, preserving fixed-capacity `error`/`drop-newest`
  scheduling under overload.
- Added capability-gated `i420-planes` and `GpuFramePool.renderI420()`: GPU
  BT.601 limited-range conversion, tight Y/U/V packing, only 1.5 B/pixel CPU
  readback, preallocated per-slot GPU/readback resources, and reusable exact
  caller buffers. Added a real optional `@roamhq/wrtc` source/sink test.
- On Apple M4/Metal/Node 24 with `UV_THREADPOOL_SIZE=1`, the released 0.3.0
  1080p NV12 pool averaged 3.169 ms; 0.4.0 averaged 1.598 ms after removing the
  second completion bubble. An unrelated PBKDF2 probe changed from 0.535 ms
  idle / 5.212 ms during a 4K pooled frame to 0.409 ms / 0.038 ms. The new
  1080p packed-I420 benchmark read 3,110,400 bytes instead of 8,294,400 bytes.
- In the same environment, a 100-frame `@roamhq/wrtc` run averaged 2.986 ms for
  legacy RGBA readback + libyuv conversion + `onFrame`, versus 1.772 ms for GPU
  packed I420 + `onFrame` (1.69x throughput, with renderer caller-buffer reuse).
- Added a Linux Vulkan single-device prerequisite bootstrap that safely enables
  the external-memory, DMA-BUF, DRM-modifier, foreign-queue, and semaphore-fd
  extensions when available, and reports that state separately from support.
  DMA-BUF/encoder surfaces remain unavailable until a VA-created surface can be
  imported, synchronized, encoded, lifetime-tested, and fd-leak-tested on
  matched AMD amdgpu/VCN hardware; no unusable VkImage or fd is exposed.
- In a production-equivalent container on the matched AMD amdgpu/VCN host,
  Vulkan reported `encoderSurface.prerequisitesReady=true` while correctly
  keeping encoder-surface/DMA-BUF support false. All 10 GPU media tests passed.
  A 720x720, 120-frame run (after 20 warm-up frames) reduced mean/p95 output
  time from 1.702/2.302 ms for legacy RGBA to 0.996/1.034 ms for packed I420,
  and reduced readback from 2,073,600 to 777,600 bytes per frame with one fixed
  allocation reused for the remaining 139 submissions.

## 0.3.0

- Added a genuinely asynchronous, fixed-capacity `GpuFramePool` with default
  triple buffering, synchronous pre-libuv reservation, `error`/`drop-newest`
  overflow behavior, reuse statistics, deterministic close, and no per-frame
  output surface allocation after warm-up.
- Added real GPU compute conversion to truthful `nv12-planes` and optional
  `p010-planes` outputs. Both expose separate Y and interleaved UV textures,
  BT.709 limited-range/centered-chroma metadata, and validated plane content;
  P010 uses upper-10-bit word placement.
- Added explicit per-plane native handle, dimensions, logical row semantics,
  backend state restoration, external-use acknowledgement, and unsafe-slot
  retirement contracts.
- Kept DMA-BUF, IOSurface/CVPixelBuffer, shared D3D12, and encoder-native
  multi-planar surface capabilities false with wgpu 29-specific blockers rather
  than presenting separate textures as portable encoder surfaces.

## 0.2.0

- Added capability-gated `Renderer.renderGpuFrame()` output for a leased,
  submission-complete native GPU texture without CPU RGBA readback. Borrowed
  Metal, Vulkan, and D3D12 handles have explicit same-device lifetime rules.
- Added the future DMA-BUF lease surface with precise unsupported capability
  reporting. Current wgpu-managed Vulkan textures are not falsely advertised as
  exportable because their allocations lack external-memory flags.
- Split large Rust, TypeScript, JavaScript test, and WGSL sources into real
  modules and added a repository-wide 800-line source guard.

## 0.1.11

- Package metadata now targets `0.1.11` for a compatibility release that restores conformance coverage against Three.js `0.183.x`.
- The renderer backend and conformance suite now cover the CommonRenderer timestamp, bind group, DOM element, XR, and example module surface changes introduced by the Three.js `0.183.x` upgrade.

## 0.1.10

- Package metadata now targets `0.1.10` for a metadata-only npm publish that refreshes the npm README and keyword list from the GitHub package metadata.
- The renderer package README is now kept below npm registry README metadata limits and links to the full GitHub compatibility and loader documentation.

## 0.1.9

- Reusable `Renderer` instances now retain native mesh buffers after a seed render and send compact native mesh references for unchanged, cacheable geometry on later frames. This reduces repeated JS-to-native geometry payloads for transform-heavy animation with mostly static mesh attributes.
- The cache is conservative: meshes that need native vertex re-preparation for displacement, normal maps, bump maps, clearcoat normal maps, or anisotropy continue sending full geometry payloads.
- Package metadata now targets `0.1.9` for the root package and optional native binary packages.
