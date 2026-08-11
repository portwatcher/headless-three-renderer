# GPU-native frame output

`Renderer.renderGpuFrame()` renders to a leased GPU texture without copying RGBA
pixels into Node.js memory. Check `Renderer.getGpuOutputCapabilities()` first;
the ordinary `render()` API remains the compatible PNG/RGBA fallback.

```js
const capabilities = renderer.getGpuOutputCapabilities()
if (!capabilities.texture.supported) {
  return renderer.render(scene, camera, { width, height, format: 'rgba' })
}

const frame = renderer.renderGpuFrame(scene, camera, { width, height })
try {
  encoder.consumeBorrowedTexture({
    backend: frame.backend,
    handle: frame.nativeHandle(), // bigint
    width: frame.width,
    height: frame.height,
    format: frame.format,
  })
} finally {
  frame.release()
}
```

## Lease and synchronization contract

- The lease owns one dedicated `rgba8unorm` texture. Its native handle is a
  borrowed `MTLTexture*`, `VkImage`, or `ID3D12Resource*` represented as a
  JavaScript `bigint`; it is not ownership of the resource.
- `renderGpuFrame()` waits for the renderer submission to finish before it
  returns (`completed === true`, synchronization `submission-complete`). No
  CPU pixel readback occurs on this path.
- The texture was last used as a copy destination and was created with
  copy-destination, copy-source, and texture-binding usage. Backend layout/state
  transitions remain the native integrator's responsibility.
- A handle is valid only in this process and on the same native device/context
  as the renderer. In particular, a Vulkan `VkImage` alone is not portable and
  is unusable with another `VkDevice` or queue. This first API intentionally
  does not expose raw device/queue handles; it is for integrations already
  coordinated with the renderer's context.
- Do not destroy the resource. Do not transition or access it concurrently
  with renderer work. If a same-device integration transitions it after the
  completed submission, finish all external use before calling `release()`.
- `release()` is idempotent. Garbage collection also drops the lease, but code
  should release deterministically. After release, `nativeHandle()` and
  `exportDmaBuf()` throw.

## DMA-BUF

`capabilities.dmaBuf` is currently false. The renderer's Vulkan textures are
ordinary wgpu-managed allocations and were not created with exportable
external-memory flags, so manufacturing a DMA-BUF fd from them would be invalid.
`exportDmaBuf()` therefore throws a precise unsupported error.

DMA-BUF is the planned cross-component/cross-process path: it requires dedicated
exportable Vulkan allocation, plane/stride/modifier metadata, duplicated-fd
ownership, and explicit synchronization. The capability shape and lease method
reserve that contract without claiming support before those resources exist.

This API is future hardware-encoding infrastructure. It does not by itself fix
WebRTC queueing or backpressure latency.
