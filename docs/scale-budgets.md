# Scale Budget Notes

These budgets describe the minimum scene sizes covered by the current CI scale
tests. They are not renderer maximums; larger scenes should be measured on the
target platform and GPU backend.

## CI-Covered Budgets

| Area | Covered budget | Notes |
|---|---:|---|
| Mesh count | 1,936 meshes | `scale.test.mjs` renders a 44 x 44 grid of separate mesh objects with shared lightweight geometry and a small material set. |
| Transparent sorting | 1,024 layered meshes | A 32 x 16 grid renders paired transparent planes with traversal order opposite `renderOrder` to exercise the transparent sort list at scale. |
| Nested scene graph traversal | 2,048 transform groups and 256 meshes | A 16 x 16 grid renders visible meshes below eight-level `Object3D` transform chains to exercise hierarchy traversal and world-matrix propagation. |
| Mixed mesh/texture/light scene | 144 meshes | A 12 x 12 grid of transformed box meshes renders with ten unique raw textures and supported punctual lights. |
| Instanced mesh expansion | 7,056 instances | A single `InstancedMesh` renders an 84 x 84 grid with per-instance matrices and colors. |
| InstancedBufferGeometry expansion | 4,096 instances | A single mapped mesh uses `InstancedBufferGeometry` with per-instance offset, scale, color, normal, and UV attributes. |
| BatchedMesh expansion | 2,048 packed instances | A single `BatchedMesh` renders a 64 x 32 grid using one packed plane geometry plus per-instance matrices and colors. |
| CPU deformation baking | 4,096 vertices | A morphed `SkinnedMesh` only enters the frame after both morph target deltas and skeletal transforms are baked on the CPU. |
| Points billboard expansion | 4,096 points | A single `Points` object renders a 64 x 64 grid with per-point colors through CPU billboard expansion. |
| Points object count | 2,048 `Points` objects | A 64 x 32 grid of independently transformed single-point objects renders with shared geometry and a small material set. |
| Sprite billboard expansion | 2,048 sprites | A 64 x 32 grid of `Sprite` objects renders with shared colored `SpriteMaterial` instances through CPU billboard expansion. |
| Wide line expansion | 4,032 line segments | A single `LineSegments` object renders a 64-row grid with per-vertex colors through wide-line quad expansion. |
| Line object count | 2,048 `LineSegments` objects | A 64 x 32 grid of independently transformed line objects renders with shared geometry and a small material set. |
| Unique material textures | 225 maps | A texture-heavy scene renders a 15 x 15 grid where every plane has a unique `DataTexture`. |
| Unique encoded material textures | 169 maps | An encoded-texture scene renders a 13 x 13 grid where every plane has a unique PNG buffer texture. |
| Raw material texture resolution | 512 x 512 RGBA | A single mapped plane renders a 512 x 512 `DataTexture` to exercise larger decoded texture upload, sampling, and native binding paths. |
| Output readback size | 512 x 512 RGBA | A focused output-size scene renders a 512 x 512 frame to exercise larger color textures and readback buffers. |
| NodePerformanceTest-shaped glTF graph | 10,000 nodes, meshes, materials, and texture definitions | A generated glTF loader stress fixture mirrors the upstream Khronos sample's scene-graph scale with 100 shared encoded PNG images, 40,000 bufferViews, and 40,000 accessors without rendering the full graph in CI. |
| Direct lights | 64 visible non-ambient lights | The supported native light-array budget is covered with 64 point lights plus ambient light. |
| Excess direct lights | 65 visible non-ambient lights | Scenes above the supported direct-light budget fail clearly instead of silently dropping lights. |
| Breadth-test output size | 96 x 96 RGBA | Most scale tests intentionally use a small output to keep CI focused on scene breadth and adapter/native resource handling. |

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

- The mesh, hierarchy, deformation, texture, and light budgets above are regression floors. Passing them means the supported scene breadth remains covered, not that production scenes must stay under those counts.
- The direct light limit is a real current renderer limit: more than 64 visible non-ambient lights fail clearly until native light arrays are expanded.
- Texture memory depends on decoded image dimensions, not just texture count. The CI texture-heavy scenes and generated NodePerformanceTest-shaped glTF graph use tiny 4 x 4 raw and encoded textures to exercise many unique bindings, loader objects, and native image decode paths without creating a large memory benchmark; a separate 512 x 512 raw texture test covers a larger single decoded upload.
- For production workloads, benchmark representative scenes on each target OS, CPU architecture, GPU backend, and output size. Track render time, peak RSS, and whether textures are encoded buffers or already-decoded RGBA data.
- Pixel-accurate budget snapshots are intentionally avoided because Metal, Vulkan, DX12, and software fallback backends can differ. Use invariant statistics for cross-platform scale tests and reserve golden images for a chosen reference platform.
