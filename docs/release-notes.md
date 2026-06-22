# Release Notes

## 0.1.10

- Package metadata now targets `0.1.10` for a metadata-only npm publish that refreshes the npm README and keyword list from the GitHub package metadata.
- The renderer package README is now kept below npm registry README metadata limits and links to the full GitHub compatibility and loader documentation.

## 0.1.9

- Reusable `Renderer` instances now retain native mesh buffers after a seed render and send compact native mesh references for unchanged, cacheable geometry on later frames. This reduces repeated JS-to-native geometry payloads for transform-heavy animation with mostly static mesh attributes.
- The cache is conservative: meshes that need native vertex re-preparation for displacement, normal maps, bump maps, clearcoat normal maps, or anisotropy continue sending full geometry payloads.
- Package metadata now targets `0.1.9` for the root package and optional native binary packages.
