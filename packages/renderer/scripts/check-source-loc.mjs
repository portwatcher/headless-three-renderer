import { readFile, readdir } from 'node:fs/promises'
import { basename, extname, join, relative } from 'node:path'
import { fileURLToPath } from 'node:url'

const repositoryRoot = fileURLToPath(new URL('../../../', import.meta.url))
const hardLimit = 800
const sourceExtensions = new Set(['.cjs', '.js', '.mjs', '.rs', '.ts', '.tsx', '.wgsl'])
const ignoredDirectories = new Set(['.git', 'dist', 'node_modules', 'target'])
// NAPI-RS rewrites these package build outputs from the Rust binding metadata.
const generatedNativeBindings = new Set(['native.d.ts', 'native.js'])
const violations = []

async function visit(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    if (ignoredDirectories.has(entry.name)) continue
    const path = join(directory, entry.name)
    if (entry.isDirectory()) {
      await visit(path)
      continue
    }
    if (!sourceExtensions.has(extname(entry.name))) continue
    if (generatedNativeBindings.has(basename(path))) continue
    const content = await readFile(path, 'utf8')
    const lines = content === '' ? 0 : content.split(/\r?\n/).length
    if (lines > hardLimit) violations.push(`${relative(repositoryRoot, path)}: ${lines} LOC`)
  }
}

await visit(repositoryRoot)
if (violations.length > 0) {
  throw new Error(`Source files exceed the ${hardLimit} LOC hard limit:\n${violations.join('\n')}`)
}
console.log(`All repository source files are at or below ${hardLimit} LOC`)
