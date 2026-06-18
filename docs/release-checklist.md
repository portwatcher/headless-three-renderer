# Release Checklist

Use this checklist before pushing a `v<semver>` tag. The publish workflow is
tag-driven, so the release commit must already contain the compatibility,
fixture, and golden-reference state for that version.

## Documentation Gate

- Update the release notes with any supported, partial, unsupported, or planned
  renderer behavior changes.
- Update [Three.js compatibility matrix](./compatibility.md) in the same change
  set when behavior, platform support, loader coverage, or test coverage
  changes.
- Update [Khronos glTF Sample Asset Coverage](./gltf-sample-assets.md) when
  committed glTF Sample Asset fixtures change.
- Run `pnpm -C packages/renderer run test:docs` so public Markdown links and
  synchronized fixture coverage are checked before tagging.

## Golden-Reference Gate

- Regenerate browser references from
  `packages/renderer/test/browser-reference/` whenever the generated corpus,
  Three.js version, renderer output semantics, or reference tolerance changes.
- Commit platform-scoped references under
  `packages/renderer/test/browser-reference/references/<platform>-<arch>/`
  when that platform is expected to enforce golden parity.
- Run `pnpm -C packages/renderer run test:golden` for the default no-reference
  or auto-detected platform-reference path.
- Run
  `HEADLESS_THREE_REQUIRE_BROWSER_REFERENCES=1 HEADLESS_THREE_BROWSER_REFERENCE_DIR=/path/to/browser-references pnpm -C packages/renderer run test:golden`
  before release when the platform should require committed or externally
  supplied browser references.
- If a release intentionally ships without committed browser references, say so
  in the release notes and keep the compatibility matrix golden-image parity
  row marked `Partial`.

## Package Gate

- Run the full package test suite with `pnpm -C packages/renderer run test`
  after building the native artifact for the current platform.
- Confirm CI passes on Linux x64, Linux arm64, macOS x64, macOS arm64, and
  Windows x64 before publishing the tag.
- Push `v<semver>` only after the release notes, compatibility matrix, and
  golden-reference decision are part of the tagged commit.
