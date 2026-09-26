import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';

import {
  LINUX_WORKER_RELEASE,
  createLinuxWorkerManifest,
  describeLinuxBundle,
} from './linux-sycl-worker.mjs';
import { WINDOWS_WORKER_RELEASE } from './windows-worker-archive.mjs';

test('pins the same llama.cpp commit and build as the other platforms', () => {
  assert.equal(LINUX_WORKER_RELEASE.commit, WINDOWS_WORKER_RELEASE.commit);
  assert.equal(LINUX_WORKER_RELEASE.build, WINDOWS_WORKER_RELEASE.build);
  assert.equal(LINUX_WORKER_RELEASE.commit, '60eeeb6082c1126bb8bc72902c83123cd056811b');
  assert.equal(LINUX_WORKER_RELEASE.build, 'b10472');
});

test('creates one Linux bundle with auto->sycl and cpu->cpu mappings', () => {
  const hash = 'a'.repeat(64);
  const bundle = {
    executable: 'resources/workers/linux-x64/sycl/llama-server',
    files: [
      { path: 'resources/workers/linux-x64/sycl/llama-server', sha256: hash },
      { path: 'resources/workers/linux-x64/sycl/libsycl.so.8', sha256: 'b'.repeat(64) },
    ],
  };

  const manifest = createLinuxWorkerManifest(
    {
      manifestVersion: 2,
      llamaCppCommit: LINUX_WORKER_RELEASE.commit,
      llamaCppBuild: LINUX_WORKER_RELEASE.build,
      platforms: {
        'darwin-arm64': {
          modes: {
            auto: { bundle: 'default', backend: 'metal' },
            cpu: { bundle: 'default', backend: 'cpu' },
          },
          bundles: {
            default: {
              executable: 'resources/workers/darwin-arm64/llama-server',
              files: [{ path: 'resources/workers/darwin-arm64/llama-server', sha256: 'c'.repeat(64) }],
            },
          },
        },
      },
    },
    bundle,
  );

  assert.deepEqual(manifest.platforms['linux-x64'], {
    modes: {
      auto: { bundle: 'sycl', backend: 'sycl' },
      cpu: { bundle: 'sycl', backend: 'cpu' },
    },
    bundles: { sycl: bundle },
  });
  assert.equal(manifest.platforms['darwin-arm64'].modes.auto.backend, 'metal');
});

test('stages recursive dependencies, skips system libraries, and patches $ORIGIN', async (context) => {
  const root = await mkdtemp(path.join(tmpdir(), 'local-llm-linux-bundle-'));
  context.after(() => rm(root, { recursive: true, force: true }));
  const stagedBundle = path.join(root, 'sycl');
  await mkdir(stagedBundle, { recursive: true });
  await writeFile(path.join(stagedBundle, 'llama-server'), 'server');

  const fakeLibDir = path.join(root, 'fake-libs');
  await mkdir(fakeLibDir, { recursive: true });
  await writeFile(path.join(fakeLibDir, 'libsycl.so.8'), 'sycl-runtime');
  await writeFile(path.join(fakeLibDir, 'libur_loader.so.0'), 'ur-loader');

  const copied = [];
  const patched = [];
  const bundle = await describeLinuxBundle(root, stagedBundle, {
    listDependencies: async (binary) => {
      if (binary.endsWith('llama-server')) {
        return [
          { name: 'libsycl.so.8', path: path.join(fakeLibDir, 'libsycl.so.8') },
          { name: 'libc.so.6', path: '/lib/x86_64-linux-gnu/libc.so.6' },
          { name: 'libze_loader.so.1', path: '/usr/lib/libze_loader.so.1' },
        ];
      }
      if (binary.endsWith('libsycl.so.8')) {
        return [{ name: 'libur_loader.so.0', path: path.join(fakeLibDir, 'libur_loader.so.0') }];
      }
      return [];
    },
    copyDependency: async (from, to) => {
      copied.push([from, to]);
      await readFile(from).then((contents) => writeFile(to, contents));
    },
    patchRpath: async (binary) => {
      patched.push(binary);
    },
  });

  assert.equal(bundle.executable, 'resources/workers/linux-x64/sycl/llama-server');
  const paths = bundle.files.map(({ path: filePath }) => filePath);
  assert.ok(paths.includes('resources/workers/linux-x64/sycl/llama-server'));
  assert.ok(paths.includes('resources/workers/linux-x64/sycl/libsycl.so.8'));
  assert.ok(paths.includes('resources/workers/linux-x64/sycl/libur_loader.so.0'));
  assert.ok(!paths.some((filePath) => filePath.includes('libc.so')));
  assert.ok(!paths.some((filePath) => filePath.includes('libze_loader')));
  // Every staged ELF gets an $ORIGIN rpath (server + 2 bundled libs).
  assert.equal(patched.length, 3);
  assert.deepEqual([...patched].sort(), [
    path.join(stagedBundle, 'libur_loader.so.0'),
    path.join(stagedBundle, 'libsycl.so.8'),
    path.join(stagedBundle, 'llama-server'),
  ].sort());
  // Files are hashed in sorted order.
  assert.deepEqual(paths, [...paths].sort());
});

test('rejects a missing non-system dependency before publication', async (context) => {
  const root = await mkdtemp(path.join(tmpdir(), 'local-llm-linux-missing-'));
  context.after(() => rm(root, { recursive: true, force: true }));
  const stagedBundle = path.join(root, 'sycl');
  await mkdir(stagedBundle, { recursive: true });
  await writeFile(path.join(stagedBundle, 'llama-server'), 'server');

  await assert.rejects(
    describeLinuxBundle(root, stagedBundle, {
      listDependencies: async () => [{ name: 'libsycl.so.8' }],
      copyDependency: async () => {
        throw new Error('copy should not run for a missing dependency');
      },
      patchRpath: async () => undefined,
    }),
    /not available locally: libsycl\.so\.8/,
  );
});
