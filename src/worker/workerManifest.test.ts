import assert from 'node:assert/strict';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { tmpdir } from 'node:os';
import test, { type TestContext } from 'node:test';
import {
  parseWorkerManifest,
  resolveWorkerBundle,
  sha256File,
  verifyWorkerBundleFiles,
  type WorkerBundle,
} from './workerManifest.ts';

test('Windows auto selects only the SYCL bundle', () => {
  const manifest = parseWorkerManifest(validManifest());
  assert.deepEqual(resolveWorkerBundle(manifest, 'win32-x64', 'auto'), {
    target: 'win32-x64', bundleName: 'sycl', backend: 'sycl',
    executable: 'resources/workers/win32-x64/sycl/llama-server.exe',
    files: manifest.platforms['win32-x64']?.bundles.sycl?.files,
  });
});

test('Windows cpu selects only the CPU bundle', () => {
  const manifest = parseWorkerManifest(validManifest());
  assert.equal(resolveWorkerBundle(manifest, 'win32-x64', 'cpu').bundleName, 'cpu');
  assert.equal(resolveWorkerBundle(manifest, 'win32-x64', 'cpu').backend, 'cpu');
});

test('Darwin auto and cpu reuse one file bundle with different backends', () => {
  const manifest = parseWorkerManifest(validManifest());
  assert.equal(resolveWorkerBundle(manifest, 'darwin-arm64', 'auto').backend, 'metal');
  assert.equal(resolveWorkerBundle(manifest, 'darwin-arm64', 'cpu').backend, 'cpu');
  assert.equal(resolveWorkerBundle(manifest, 'darwin-arm64', 'auto').executable,
    resolveWorkerBundle(manifest, 'darwin-arm64', 'cpu').executable);
});

test('resolved bundle files are isolated from the manifest', () => {
  const manifest = parseWorkerManifest(validManifest());
  const resolved = resolveWorkerBundle(manifest, 'win32-x64', 'auto');
  resolved.files[0]!.path = 'resources/workers/win32-x64/sycl/changed.exe';
  assert.equal(
    resolveWorkerBundle(manifest, 'win32-x64', 'auto').files[0]!.path,
    'resources/workers/win32-x64/sycl/llama-server.exe',
  );
});

test('rejects paths outside the selected platform directory', () => {
  const value = validManifest();
  value.platforms['win32-x64'].bundles.sycl.files[0].path = '../sycl8.dll';
  assert.throws(() => parseWorkerManifest(value), /safe relative path/);
});

test('rejects duplicate paths and an executable omitted from files', () => {
  const value = validManifest();
  value.platforms['win32-x64'].bundles.sycl.files.push(value.platforms['win32-x64'].bundles.sycl.files[0]);
  assert.throws(() => parseWorkerManifest(value), /duplicate worker file/);
  const missing = validManifest();
  missing.platforms['win32-x64'].bundles.sycl.files = missing.platforms['win32-x64'].bundles.sycl.files.slice(1);
  assert.throws(() => parseWorkerManifest(missing), /executable.*files/i);
});

test('verifies every file hash in a bundle', async (context) => {
  const fixture = await createBundleFixture(context);
  await verifyWorkerBundleFiles(fixture.root, fixture.bundle);
  await writeFile(fixture.dllPath, 'tampered');
  await assert.rejects(verifyWorkerBundleFiles(fixture.root, fixture.bundle), /failed its SHA-256 integrity check/);
});

function validManifest(): Record<string, any> {
  const hash = 'a'.repeat(64);
  return { manifestVersion: 2, llamaCppCommit: '60eeeb6082c1126bb8bc72902c83123cd056811b', llamaCppBuild: 'b10472', platforms: {
    'darwin-arm64': { modes: { auto: { bundle: 'default', backend: 'metal' }, cpu: { bundle: 'default', backend: 'cpu' } }, bundles: { default: { executable: 'resources/workers/darwin-arm64/llama-server', files: [{ path: 'resources/workers/darwin-arm64/llama-server', sha256: hash }] } } },
    'win32-x64': { modes: { auto: { bundle: 'sycl', backend: 'sycl' }, cpu: { bundle: 'cpu', backend: 'cpu' } }, bundles: {
      sycl: { executable: 'resources/workers/win32-x64/sycl/llama-server.exe', files: [{ path: 'resources/workers/win32-x64/sycl/llama-server.exe', sha256: hash }, { path: 'resources/workers/win32-x64/sycl/sycl8.dll', sha256: hash }] },
      cpu: { executable: 'resources/workers/win32-x64/cpu/llama-server.exe', files: [{ path: 'resources/workers/win32-x64/cpu/llama-server.exe', sha256: hash }] },
    } },
  } };
}

async function createBundleFixture(context: TestContext): Promise<{ root: string; dllPath: string; bundle: WorkerBundle }> {
  const root = await mkdtemp(path.join(tmpdir(), 'worker-manifest-'));
  context.after(() => rm(root, { recursive: true, force: true }));
  const executablePath = path.join(root, 'resources/workers/win32-x64/sycl/llama-server.exe');
  const dllPath = path.join(root, 'resources/workers/win32-x64/sycl/sycl8.dll');
  await mkdir(path.dirname(executablePath), { recursive: true });
  await writeFile(executablePath, 'server'); await writeFile(dllPath, 'runtime');
  return { root, dllPath, bundle: { executable: 'resources/workers/win32-x64/sycl/llama-server.exe', files: [
    { path: 'resources/workers/win32-x64/sycl/llama-server.exe', sha256: await sha256File(executablePath) },
    { path: 'resources/workers/win32-x64/sycl/sycl8.dll', sha256: await sha256File(dllPath) },
  ] } };
}
