import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { sha256File } from '../src/worker/workerManifest.ts';
import { prepareTargetWorkers, targetIgnoreEntries } from './package-workers.mjs';

test('Windows packaging requires and verifies both bundles', async (context) => {
  const fixture = await packageFixture(context, 'win32-x64');
  await prepareTargetWorkers(fixture.root, 'win32-x64');
  await rm(fixture.syclDll);
  await assert.rejects(
    prepareTargetWorkers(fixture.root, 'win32-x64'),
    /sycl8\.dll/,
  );
});

test('Darwin verification neither reads nor requires Windows files', async (context) => {
  const fixture = await packageFixture(context, 'darwin-arm64');
  await rm(fixture.windowsDirectory, { recursive: true, force: true });
  await prepareTargetWorkers(fixture.root, 'darwin-arm64');
});

test('ignore rules exclude the complete other-platform tree', () => {
  assert.equal(targetIgnoreEntries('darwin-arm64').includes('resources/workers/win32-x64/**'), true);
  assert.equal(targetIgnoreEntries('win32-x64').includes('resources/workers/darwin-arm64/**'), true);
});

async function packageFixture(context, target) {
  const root = await mkdtemp(path.join(tmpdir(), 'local-llm-package-workers-'));
  context.after(() => rm(root, { recursive: true, force: true }));

  const manifest = {
    manifestVersion: 2,
    llamaCppCommit: '60eeeb6082c1126bb8bc72902c83123cd056811b',
    llamaCppBuild: 'b10472',
    platforms: {
      'darwin-arm64': {
        modes: {
          auto: { bundle: 'default', backend: 'metal' },
          cpu: { bundle: 'default', backend: 'cpu' },
        },
        bundles: {
          default: await workerBundle(root, 'resources/workers/darwin-arm64/llama-server'),
        },
      },
      'win32-x64': {
        modes: {
          auto: { bundle: 'sycl', backend: 'sycl' },
          cpu: { bundle: 'cpu', backend: 'cpu' },
        },
        bundles: {
          sycl: await workerBundle(
            root,
            'resources/workers/win32-x64/sycl/llama-server.exe',
            ['resources/workers/win32-x64/sycl/sycl8.dll'],
          ),
          cpu: await workerBundle(root, 'resources/workers/win32-x64/cpu/llama-server.exe'),
        },
      },
    },
  };
  const manifestPath = path.join(root, 'resources', 'workers', 'manifest.json');
  await mkdir(path.dirname(manifestPath), { recursive: true });
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);

  return {
    root,
    target,
    syclDll: path.join(root, 'resources', 'workers', 'win32-x64', 'sycl', 'sycl8.dll'),
    windowsDirectory: path.join(root, 'resources', 'workers', 'win32-x64'),
  };
}

async function workerBundle(root, executable, additionalPaths = []) {
  const paths = [executable, ...additionalPaths];
  const files = [];
  for (const relativePath of paths) {
    const absolutePath = path.join(root, ...relativePath.split('/'));
    await mkdir(path.dirname(absolutePath), { recursive: true });
    await writeFile(absolutePath, relativePath);
    files.push({ path: relativePath, sha256: await sha256File(absolutePath) });
  }
  return { executable, files };
}
