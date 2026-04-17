| Priority | Feature | Effort | Impact | Rationale |
|---|---|---|---|---|
| 1 | ~~ACES Filmic tone mapping~~ ✅ done | ~10 lines WGSL | High | Narkowicz fit with three.js's 1/0.6 exposure pre-scale in `shader.rs::aces_filmic_tone_mapping`. |
| 2 | ~~Occlusion map (`aoMap`)~~ ✅ done | Low | Medium | Sampled red channel, applied to indirect (ambient/IBL) lighting per three.js convention. |
| 3 | ~~Double-sided / back-face culling~~ ✅ done | Low | Medium | `material.side` extracted in `api/materials.ts`; 3 pipeline variants (front/back/double) per opaque/transparent pass in `renderer.rs`. Shader flips normals on back-facing fragments for correct lighting on `BackSide`/`DoubleSi
de`. |
| 4 | `MeshBasicMaterial` / `MeshLambertMaterial` unlit path | Low | Medium | Currently all materials go through PBR. Basic/Lambert should skip specular for correct flat-shaded scenes. |
| 5 | Shadows (directional shadow map) | High | High | Requires depth-only pass from light POV, shadow sampler, PCF. Significant wgpu pipeline work. |
| 6 | Lines / Points | Medium | Low | Separate pipelines with different topologies. Niche use case. |