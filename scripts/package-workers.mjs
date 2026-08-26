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
  if (!TARGETS.has(target)) {
    throw new Error('Worker target must be darwin-arm64 or win32-x64.');
  }
  const manifestPath = path.join(root, 'resources', 'workers', 'manifest.json');
  const rawManifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  if (rawManifest?.manifestVersion !== 2) {
    const buildCommand = target === 'win32-x64'
      ? 'npm run build:worker -- --target win32-x64 --backend all'
      : 'npm run build:worker -- --target darwin-arm64';
    throw new Error(
      `Worker manifest version 2 has not been generated. Run ${buildCommand} before packaging.`,
    );
  }
  const manifest = parseWorkerManifest(rawManifest);
  await verifyPlatformBundles(root, manifest, target);
}

export function targetIgnoreEntries(target) {
  if (!TARGETS.has(target)) {
    throw new Error('Worker target must be darwin-arm64 or win32-x64.');
  }
  const otherTarget = target === 'darwin-arm64' ? 'win32-x64' : 'darwin-arm64';
  return [...GENERAL_IGNORE_ENTRIES, `resources/workers/${otherTarget}/**`];
}
