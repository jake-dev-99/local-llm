import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { verifiedWorkerBundle } from './workerIntegrity.ts';
import { sha256File } from './workerManifest.ts';

test('verified Windows auto selects only the SYCL bundle', async (context) => {
  const root = await workerFixture(context);
  const bundle = await verifiedWorkerBundle(root, 'win32-x64', 'auto');

  assert.equal(bundle.backend, 'sycl');
  assert.equal(bundle.bundleName, 'sycl');
  assert.match(bundle.executablePath, /win32-x64[\\/]sycl[\\/]llama-server\.exe$/);
});

test('verified Windows cpu selects only the explicit CPU bundle', async (context) => {
  const root = await workerFixture(context);
  const bundle = await verifiedWorkerBundle(root, 'win32-x64', 'cpu');

  assert.equal(bundle.backend, 'cpu');
  assert.equal(bundle.bundleName, 'cpu');
  assert.match(bundle.executablePath, /win32-x64[\\/]cpu[\\/]llama-server\.exe$/);
});

async function workerFixture(context: test.TestContext): Promise<string> {
  const root = await mkdtemp(path.join(tmpdir(), 'worker-integrity-'));
  context.after(() => rm(root, { recursive: true, force: true }));
  const sycl = await bundle(root, 'resources/workers/win32-x64/sycl/llama-server.exe', [
    'resources/workers/win32-x64/sycl/sycl8.dll',
  ]);
  const cpu = await bundle(root, 'resources/workers/win32-x64/cpu/llama-server.exe');
  const manifest = {
    manifestVersion: 2,
    llamaCppCommit: '60eeeb6082c1126bb8bc72902c83123cd056811b',
    llamaCppBuild: 'b10472',
    platforms: {
      'win32-x64': {
        modes: {
          auto: { bundle: 'sycl', backend: 'sycl' },
          cpu: { bundle: 'cpu', backend: 'cpu' },
        },
        bundles: { sycl, cpu },
      },
    },
  };
  const manifestPath = path.join(root, 'resources', 'workers', 'manifest.json');
  await mkdir(path.dirname(manifestPath), { recursive: true });
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  return root;
}

async function bundle(root: string, executable: string, extra: string[] = []) {
  const files = [];
  for (const relativePath of [executable, ...extra]) {
    const absolutePath = path.join(root, ...relativePath.split('/'));
    await mkdir(path.dirname(absolutePath), { recursive: true });
    await writeFile(absolutePath, relativePath);
    files.push({ path: relativePath, sha256: await sha256File(absolutePath) });
  }
  return { executable, files };
}
