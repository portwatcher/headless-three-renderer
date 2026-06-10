import test from 'node:test'
import assert from 'node:assert/strict'
import pkg from '../dist/index.js'
import { meanRgba, nonBackgroundRatio } from './helpers.mjs'
import { createSceneCorpus } from './corpus.mjs'

const { Renderer } = pkg

test('representative scene corpus renders without crashes', async (t) => {
  const renderer = new Renderer()
  const corpus = createSceneCorpus()
  assert.ok(corpus.length >= 6, 'expected a representative scene corpus')

  for (const fixture of corpus) {
    await t.test(fixture.name, () => {
      const rgba = renderer.render(fixture.scene, fixture.camera, fixture.options)
      const width = fixture.options.width
      const height = fixture.options.height
      assert.equal(rgba.length, width * height * 4)

      const ratio = nonBackgroundRatio(rgba, fixture.background, fixture.backgroundTolerance ?? 3)
      assert.ok(ratio > (fixture.minNonBackgroundRatio ?? 0.002), `${fixture.name} should render visible non-background pixels (${ratio})`)

      const mean = meanRgba(rgba)
      assert.ok(mean.a > 240, `${fixture.name} should produce opaque output alpha (${mean.a})`)
    })
  }
})
