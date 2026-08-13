import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { createRequire } from 'node:module'
import { promisify } from 'node:util'
import { fileURLToPath } from 'node:url'
import test from 'node:test'

const execFileAsync = promisify(execFile)
const require = createRequire(import.meta.url)

test('packed I420 is consumed directly by @roamhq/wrtc RTCVideoSource', async (t) => {
  let wrtcPath = process.env.WRTC_MODULE_PATH
  if (!wrtcPath) {
    try {
      wrtcPath = require.resolve('@roamhq/wrtc')
    } catch (error) {
      t.skip(`optional @roamhq/wrtc integration is unavailable: ${error.code ?? error.message}`)
      return
    }
  }
  const fixture = fileURLToPath(new URL('./fixtures/wrtc-i420-consumer.mjs', import.meta.url))
  const { stdout } = await execFileAsync(process.execPath, [fixture], {
    env: { ...process.env, WRTC_MODULE_PATH: wrtcPath },
    timeout: 5_000,
  })
  assert.match(stdout, /wrtc-i420-consumer: ok/)
})
