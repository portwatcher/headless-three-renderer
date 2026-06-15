# Scale Budget Notes

These budgets describe the minimum scene sizes covered by the current CI scale
tests. They are not renderer maximums; larger scenes should be measured on the
target platform and GPU backend.

## CI-Covered Budgets

| Area | Covered budget | Notes |
|---|---:|---|
| Mesh count | 80 meshes | `scale.test.mjs` renders an 8 x 10 grid of transformed box meshes. |
| Unique material textures | 64 maps | A texture-heavy scene renders an 8 x 8 grid where every plane has a unique `DataTexture`. |
| Unique encoded material textures | 36 maps | An encoded-texture scene renders a 6 x 6 grid where every plane has a unique PNG buffer texture. |
| Direct lights | 64 visible non-ambient lights | The supported native light-array budget is covered with 64 point lights plus ambient light. |
| Excess direct lights | 65 visible non-ambient lights | Scenes above the supported direct-light budget fail clearly instead of silently dropping lights. |
| Output size | 96 x 96 RGBA | The scale tests intentionally use a small output to keep CI focused on scene breadth and adapter/native resource handling. |

## Platform Expectations

The invariant scale tests are expected to pass on the same platforms as the
published native packages:

| Platform package | Expected backend notes |
|---|---|
| `@headless-three/renderer-linux-x64-gnu` | Vulkan-capable runners and software `wgpu` fallback paths should pass the invariant budget. |
| `@headless-three/renderer-linux-arm64-gnu` | Treat CI coverage as a functional floor; profile production scenes on the target ARM host. |
| `@headless-three/renderer-darwin-x64` | Metal backend output can differ slightly from other GPU backends, so scale tests use non-snapshot invariants. |
| `@headless-three/renderer-darwin-arm64` | Same budget expectations as Darwin x64; memory headroom depends on the host GPU and unified-memory pressure. |
| `@headless-three/renderer-win32-x64-msvc` | DX12 backend output can differ slightly from Metal/Vulkan, so avoid using these tests as pixel-golden references. |

## Interpreting The Numbers

- The mesh, texture, and light budgets above are regression floors. Passing them means the supported scene breadth remains covered, not that production scenes must stay under those counts.
- The direct light limit is a real current renderer limit: more than 64 visible non-ambient lights fail clearly until native light arrays are expanded.
- Texture memory depends on decoded image dimensions, not just texture count. The CI texture-heavy scenes use tiny 4 x 4 raw and encoded textures to exercise many unique bindings and native image decode paths without creating a large memory benchmark.
- For production workloads, benchmark representative scenes on each target OS, CPU architecture, GPU backend, and output size. Track render time, peak RSS, and whether textures are encoded buffers or already-decoded RGBA data.
- Pixel-accurate budget snapshots are intentionally avoided because Metal, Vulkan, DX12, and software fallback backends can differ. Use invariant statistics for cross-platform scale tests and reserve golden images for a chosen reference platform.
