import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import * as THREE from 'three'
import { MToonMaterial } from '@pixiv/three-vrm'
import pkg from '../dist/index.js'
import { createParityCases } from './parity-cases.mjs'

// Native output against Chrome WebGLRenderer output of the same scenes (Three.js r180 and
// @pixiv/three-vrm 3.4.4; see browser-reference/generate-parity.mjs). The references match
// Chrome on an NVIDIA GPU (ANGLE D3D11) and on SwiftShader within 1 level. Each case allows
// 2 levels per channel, or 3 where specular highlights or derivative-based tangent frames
// meet the sample points, for rasterizer and float differences between GPUs.
const { Renderer } = pkg
const references = JSON.parse(readFileSync(new URL('./parity-references.json', import.meta.url), 'utf8'))
const cases = createParityCases({ THREE, MToonMaterial })

test('parity references cover every parity case', () => {
  assert.equal(references.generator.three, '180', 'parity references must come from Three.js r180')
  assert.deepEqual(Object.keys(references.cases).sort(), cases.map((entry) => entry.name).sort())
})

test('native output matches Three.js r180 WebGL output', async (t) => {
  const renderer = new Renderer()
  try {
    for (const entry of cases) {
      await t.test(entry.name, () => {
        const built = entry.build()
        renderer.shadowMap.enabled = false
        built.scene.updateMatrixWorld(true)
        const frame = renderer.render(built.scene, built.camera, {
          width: entry.width,
          height: entry.height,
          format: 'rgba',
          toneMapping: built.toneMapping ?? THREE.NoToneMapping,
          outputColorSpace: built.outputColorSpace ?? THREE.SRGBColorSpace,
        })
        const expected = references.cases[entry.name]
        const mismatches = []
        entry.samples.forEach(([x, y], index) => {
          const i = (y * entry.width + x) * 4
          const actual = [frame[i], frame[i + 1], frame[i + 2]]
          if (actual.some((value, channel) => Math.abs(value - expected[index][channel]) > entry.tolerance)) {
            mismatches.push(`(${x}, ${y}) WebGL ${expected[index]} native ${actual}`)
          }
        })
        assert.equal(mismatches.length, 0, `${entry.name} differs by more than ${entry.tolerance}:\n${mismatches.join('\n')}`)
      })
    }
  } finally {
    renderer.dispose()
  }
})
