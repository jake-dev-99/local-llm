import { readFile } from 'node:fs/promises';
import path from 'node:path';

import { parseWorkerManifest, verifyPlatformBundles } from '../src/worker/workerManifest.ts';
import { prepareWindowsWorkerArchive } from './windows-worker-archive.mjs';

const TARGETS = new Set(['darwin-arm64', 'win32-x64']);

export async function prepareTargetWorkers(root, target, dependencies = {}) {
  if (!TARGETS.has(target)) {
    throw new Error('Worker target must be darwin-arm64 or win32-x64.');
  }
  if (target === 'win32-x64') {
    const prepareWindowsArchive = dependencies.prepareWindowsArchive
      ?? prepareWindowsWorkerArchive;
    await prepareWindowsArchive(root);
  }
  const manifestPath = path.join(root, 'resources', 'workers', 'manifest.json');
  const rawManifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  if (rawManifest?.manifestVersion !== 2) {
    throw new Error(
      target === 'win32-x64'
        ? 'Official Windows worker archive preparation did not produce manifest version 2.'
        : 'Worker manifest version 2 has not been generated. Run npm run build:worker -- --target darwin-arm64 before packaging.',
    );
  }
  const manifest = parseWorkerManifest(rawManifest);
  await verifyPlatformBundles(root, manifest, target);
}

/**
 * The ignore rules for one target's VSIX: the repository's .vscodeignore, plus
 * the other platform's workers, which is the one rule that depends on the
 * target. vsce accepts a single ignore file, so the two are combined here
 * rather than kept as a second list.
 */
export async function targetIgnoreEntries(root, target) {
  if (!TARGETS.has(target)) {
    throw new Error('Worker target must be darwin-arm64 or win32-x64.');
  }
  const otherTarget = target === 'darwin-arm64' ? 'win32-x64' : 'darwin-arm64';
  const listed = (await readFile(path.join(root, '.vscodeignore'), 'utf8'))
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith('#'));
  return [...listed, `resources/workers/${otherTarget}/**`];
}
