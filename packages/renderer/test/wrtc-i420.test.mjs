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
    // Windows CI can spend tens of seconds creating its first native GPU
    // device. The fixture keeps a separate two-second deadline after onFrame,
    // so this watchdog covers process/device startup without weakening the
    // actual RTCVideoSink delivery assertion.
    timeout: 60_000,
  })
  assert.match(stdout, /wrtc-i420-consumer: ok/)
})
