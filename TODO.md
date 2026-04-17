| Priority | Feature | Effort | Impact | Rationale |
|---|---|---|---|---|
| 1 | ACES Filmic tone mapping | ~10 lines WGSL | High | Color accuracy vs Three.js. Single biggest visual fidelity gap for users comparing output. |
| 2 | Occlusion map (`aoMap`) | Low | Medium | Already have the texture plumbing. Just another bind group + `* (1 - ao * aoIntensity)` in shader ambient term. |
| 3 | Double-sided / back-face culling | Low | Medium | Currently `cull_mode: None` always. Extract `material.side` from Three.js and use pipeline variants. |
| 4 | `MeshBasicMaterial` / `MeshLambertMaterial` unlit path | Low | Medium | Currently all materials go through PBR. Basic/Lambert should skip specular for correct flat-shaded scenes. |
| 5 | Shadows (directional shadow map) | High | High | Requires depth-only pass from light POV, shadow sampler, PCF. Significant wgpu pipeline work. |
| 6 | Lines / Points | Medium | Low | Separate pipelines with different topologies. Niche use case. |