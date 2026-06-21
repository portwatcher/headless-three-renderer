import { spawnSync } from 'node:child_process'

const shardCount = Number(process.env.HEADLESS_THREE_CONFORMANCE_SHARDS ?? 4)

if (!Number.isSafeInteger(shardCount) || shardCount < 1) {
  throw new Error('HEADLESS_THREE_CONFORMANCE_SHARDS must be a positive integer.')
}

for (let shard = 1; shard <= shardCount; shard += 1) {
  const label = `${shard}/${shardCount}`
  console.log(`Running conformance shard ${label}`)

  const result = spawnSync(
    process.execPath,
    [
      '--test',
      '--test-reporter=spec',
      'test/scenes.test.mjs',
    ],
    {
      env: {
        ...process.env,
        HEADLESS_THREE_CONFORMANCE_SHARD: label,
      },
      stdio: 'inherit',
    },
  )

  if (result.signal) {
    console.error(`Conformance shard ${label} exited with signal ${result.signal}.`)
    process.exit(1)
  }

  if (result.status !== 0) {
    process.exit(result.status ?? 1)
  }
}
