import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import * as THREE from 'three'
import { Renderer } from '../../dist/index.js'

const require = createRequire(import.meta.url)
const wrtc = require(process.env.WRTC_MODULE_PATH)
const width = 8
const height = 6
const target = Buffer.alloc(width * height * 1.5)
const renderer = new Renderer()
const pool = renderer.createGpuFramePool({ width, height, capacity: 1, format: 'i420-planes' })
const scene = new THREE.Scene()
scene.background = new THREE.Color(0.8, 0.25, 0.1)
const camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 100)
const output = await pool.renderI420(scene, camera, { toneMapping: THREE.NoToneMapping }, target)
assert.ok(output)
assert.strictEqual(output.data, target)
assert.equal(output.data.byteOffset, 0)
assert.equal(output.data.buffer.byteLength, output.data.byteLength)
// Cross the native-addon boundary through wrtc's documented plain frame
// shape. Perohub also constructs this object instead of passing renderer-only
// metadata into a second addon.
const consumerFrame = {
  width: output.width,
  height: output.height,
  data: output.data,
}
const expected = Buffer.from(consumerFrame.data)

const source = new wrtc.nonstandard.RTCVideoSource()
const track = source.createTrack()
const sink = new wrtc.nonstandard.RTCVideoSink(track)
const received = new Promise((resolve, reject) => {
  const timeout = setTimeout(() => reject(new Error('RTCVideoSink did not receive the I420 frame')), 2_000)
  sink.onframe = ({ frame }) => {
    clearTimeout(timeout)
    resolve(frame)
  }
})
source.onFrame(consumerFrame)
consumerFrame.data.fill(0)
const frame = await received
assert.equal(frame.width, width)
assert.equal(frame.height, height)
assert.deepEqual(Buffer.from(frame.data), expected)
assert.equal(pool.stats().available, 1)
sink.stop()
track.stop()
pool.close()
// @roamhq/wrtc 0.10.0 can abort during Node 24 environment teardown after
// otherwise-correct source/sink cleanup. Exit after the observable contract is
// verified so that upstream finalizer bug cannot make this integration flaky.
process.stdout.write('wrtc-i420-consumer: ok\n', () => process.exit(0))
