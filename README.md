# headless-three

Headless Three.js ecosystem for rendering Three.js scenes outside the browser.

## Packages

| Package | Description |
|---|---|
| [`@headless-three/renderer`](./packages/renderer) | Headless `wgpu` renderer for Three.js scenes in Node.js |

## Documentation

- [`@headless-three/renderer` README](./packages/renderer/README.md)
- [Three.js compatibility matrix](./docs/compatibility.md)
- [Renderer roadmap](./TODO.md)

## Development

```bash
pnpm install
pnpm -r build
pnpm -r test
```

Releases are tag-driven: push `v<semver>` to trigger the publish workflow.
