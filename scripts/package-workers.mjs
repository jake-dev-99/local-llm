import { readFile } from 'node:fs/promises';
import path from 'node:path';

import { parseWorkerManifest, verifyPlatformBundles } from '../src/worker/workerManifest.ts';

const TARGETS = new Set(['darwin-arm64', 'win32-x64']);
const GENERAL_IGNORE_ENTRIES = [
  'build/**',
  '.git/**',
  '.gitignore',
  '.vscodeignore',
  '.github/**',
  '.agents/**',
  '.codex/**',
  '**/.DS_Store',
  'dist/vsix/**',
  'docs/**',
  'node_modules/**',
  'src/**',
  'scripts/**',
  'tsconfig.json',
  'esbuild.mjs',
  'package-lock.json',
  '*.vsix',
];

export async function prepareTargetWorkers(root, target) {
  const manifestPath = path.join(root, 'resources', 'workers', 'manifest.json');
  const manifest = parseWorkerManifest(JSON.parse(await readFile(manifestPath, 'utf8')));
  await verifyPlatformBundles(root, manifest, target);
}

export function targetIgnoreEntries(target) {
  if (!TARGETS.has(target)) {
    throw new Error('Worker target must be darwin-arm64 or win32-x64.');
  }
  const otherTarget = target === 'darwin-arm64' ? 'win32-x64' : 'darwin-arm64';
  return [...GENERAL_IGNORE_ENTRIES, `resources/workers/${otherTarget}/**`];
}
