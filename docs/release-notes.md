# Release Notes

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
