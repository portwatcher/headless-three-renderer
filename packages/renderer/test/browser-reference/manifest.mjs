import * as THREE from 'three'
import { createSceneCorpus } from '../corpus.mjs'

export const BROWSER_REFERENCE_MANIFEST_FILE = 'manifest.json'

export function createBrowserReferenceFixtures(fixtures = createSceneCorpus()) {
  return fixtures.filter((fixture) => fixture.browserReference !== false)
}

export function createBrowserReferenceManifest(fixtures = createSceneCorpus()) {
  const browserFixtures = createBrowserReferenceFixtures(fixtures)
  return {
    schemaVersion: 1,
    generator: '@headless-three/renderer/test/browser-reference',
    renderer: 'THREE.WebGLRenderer',
    threeRevision: THREE.REVISION,
    fixtures: browserFixtures.map((fixture) => ({
      file: `${fixture.name}.png`,
      name: fixture.name,
      width: fixture.options.width,
      height: fixture.options.height,
      outputColorSpace: normalizeBrowserReferenceOutputColorSpace(fixture.options.outputColorSpace),
    })),
  }
}

export function normalizeBrowserReferenceOutputColorSpace(value) {
  if (
    value === THREE.LinearSRGBColorSpace ||
    value === 'srgb-linear' ||
    value === 'linear-srgb' ||
    value === 'linearsrgb' ||
    value === 'linear'
  ) {
    return THREE.LinearSRGBColorSpace
  }
  return THREE.SRGBColorSpace
}
