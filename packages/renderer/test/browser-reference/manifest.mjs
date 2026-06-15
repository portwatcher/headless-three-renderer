import * as THREE from 'three'
import { createSceneCorpus } from '../corpus.mjs'

export const BROWSER_REFERENCE_MANIFEST_FILE = 'manifest.json'

export function createBrowserReferenceManifest(fixtures = createSceneCorpus()) {
  return {
    schemaVersion: 1,
    generator: '@headless-three/renderer/test/browser-reference',
    renderer: 'THREE.WebGLRenderer',
    threeRevision: THREE.REVISION,
    fixtures: fixtures.map((fixture) => ({
      file: `${fixture.name}.png`,
      name: fixture.name,
      width: fixture.options.width,
      height: fixture.options.height,
      outputColorSpace: fixture.options.outputColorSpace ?? THREE.SRGBColorSpace,
    })),
  }
}
