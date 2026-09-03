import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';

import {
  WINDOWS_WORKER_RELEASE,
  createWindowsWorkerManifest,
  extractZipArchive,
  prepareWindowsWorkerArchive,
} from './windows-worker-archive.mjs';

test('pins the official llama.cpp b10472 Windows SYCL asset', () => {
  assert.deepEqual(WINDOWS_WORKER_RELEASE, {
    commit: '60eeeb6082c1126bb8bc72902c83123cd056811b',
    build: 'b10472',
    assetName: 'llama-b10472-bin-win-sycl-x64.zip',
    url: 'https://github.com/ggml-org/llama.cpp/releases/download/b10472/llama-b10472-bin-win-sycl-x64.zip',
    size: 119700367,
    sha256: '0c4c50f1e9805933e043d4970f0c2050e4fb5343b8ac0244a49efaa474705830',
  });
});

test('downloads once, verifies the archive, and reuses the verified cache', async (context) => {
  const root = await fixtureRoot(context);
  const archive = storedZip([
    { name: 'llama-server.exe', contents: 'server' },
    { name: 'runtime.dll', contents: 'runtime' },
  ]);
  const release = fixtureRelease(archive);
  let requests = 0;
  const dependencies = {
    release,
    fetchFn: async (url) => {
      requests += 1;
      assert.equal(url, release.url);
      return new Response(archive, { status: 200 });
    },
    log: () => undefined,
  };

  await prepareWindowsWorkerArchive(root, dependencies);
  await prepareWindowsWorkerArchive(root, {
    ...dependencies,
    fetchFn: async () => {
      throw new Error('verified cache should avoid a second download');
    },
  });

  assert.equal(requests, 1);
  const cached = path.join(root, 'build', 'worker-downloads', release.assetName);
  assert.deepEqual(await readFile(cached), archive);
});

test('rejects an archive with the wrong byte count before publication', async (context) => {
  const root = await fixtureRoot(context);
  const archive = storedZip([{ name: 'llama-server.exe', contents: 'server' }]);
  const release = { ...fixtureRelease(archive), size: archive.length + 1 };

  await assert.rejects(
    prepareWindowsWorkerArchive(root, {
      release,
      fetchFn: async () => new Response(archive, { status: 200 }),
      log: () => undefined,
    }),
    /archive size.*expected.*actual/is,
  );
  await assert.rejects(stat(path.join(root, 'resources', 'workers', 'win32-x64')), /ENOENT/);
});

test('rejects an archive with the wrong SHA-256 before publication', async (context) => {
  const root = await fixtureRoot(context);
  const archive = storedZip([{ name: 'llama-server.exe', contents: 'server' }]);
  const release = { ...fixtureRelease(archive), sha256: '0'.repeat(64) };

  await assert.rejects(
    prepareWindowsWorkerArchive(root, {
      release,
      fetchFn: async () => new Response(archive, { status: 200 }),
      log: () => undefined,
    }),
    /archive SHA-256.*expected.*actual/is,
  );
  await assert.rejects(stat(path.join(root, 'resources', 'workers', 'win32-x64')), /ENOENT/);
});

test('extracts a flat regular-file payload', async (context) => {
  const root = await fixtureRoot(context);
  const archivePath = path.join(root, 'worker.zip');
  const destination = path.join(root, 'staging');
  await writeFile(archivePath, storedZip([
    { name: 'llama-server.exe', contents: 'server' },
    { name: 'nested/runtime.dll', contents: 'runtime' },
  ]));

  const files = await extractZipArchive(archivePath, destination);

  assert.deepEqual(files, ['llama-server.exe', 'nested/runtime.dll']);
  assert.equal(await readFile(path.join(destination, 'llama-server.exe'), 'utf8'), 'server');
  assert.equal(await readFile(path.join(destination, 'nested', 'runtime.dll'), 'utf8'), 'runtime');
});

for (const unsafe of [
  { label: 'parent traversal', entries: [{ name: 'llama-server.exe', contents: 'server' }, { name: '../escape.dll', contents: 'escape' }] },
  { label: 'absolute path', entries: [{ name: '/llama-server.exe', contents: 'server' }] },
  { label: 'symbolic link', entries: [{ name: 'llama-server.exe', contents: 'server' }, { name: 'link.dll', contents: 'target', mode: 0o120777 }] },
  { label: 'duplicate output', entries: [{ name: 'llama-server.exe', contents: 'one' }, { name: 'llama-server.exe', contents: 'two' }] },
]) {
  test(`rejects a ZIP ${unsafe.label}`, async (context) => {
    const root = await fixtureRoot(context);
    const archivePath = path.join(root, 'worker.zip');
    const destination = path.join(root, 'staging');
    await writeFile(archivePath, storedZip(unsafe.entries));

    await assert.rejects(
      extractZipArchive(archivePath, destination),
      /unsafe|symbolic|duplicate|absolute|parent/i,
    );
    await assert.rejects(stat(path.join(root, 'escape.dll')), /ENOENT/);
  });
}

test('rejects an archive without llama-server.exe', async (context) => {
  const root = await fixtureRoot(context);
  const archivePath = path.join(root, 'worker.zip');
  await writeFile(archivePath, storedZip([{ name: 'runtime.dll', contents: 'runtime' }]));

  await assert.rejects(
    extractZipArchive(archivePath, path.join(root, 'staging')),
    /llama-server\.exe.*missing/i,
  );
});

test('creates one Windows bundle while migrating the legacy Darwin worker', () => {
  const hash = 'a'.repeat(64);
  const bundle = {
    executable: 'resources/workers/win32-x64/sycl/llama-server.exe',
    files: [
      { path: 'resources/workers/win32-x64/sycl/llama-server.exe', sha256: hash },
      { path: 'resources/workers/win32-x64/sycl/runtime.dll', sha256: 'b'.repeat(64) },
    ],
  };

  const manifest = createWindowsWorkerManifest({
    llamaCppCommit: WINDOWS_WORKER_RELEASE.commit,
    llamaCppBuild: WINDOWS_WORKER_RELEASE.build,
    workers: {
      'darwin-arm64': {
        path: 'resources/workers/darwin-arm64/llama-server',
        sha256: 'c'.repeat(64),
      },
    },
  }, bundle);

  assert.deepEqual(manifest.platforms['win32-x64'], {
    modes: {
      auto: { bundle: 'sycl', backend: 'sycl' },
      cpu: { bundle: 'sycl', backend: 'cpu' },
    },
    bundles: { sycl: bundle },
  });
  assert.deepEqual(manifest.platforms['darwin-arm64'].modes, {
    auto: { bundle: 'default', backend: 'metal' },
    cpu: { bundle: 'default', backend: 'cpu' },
  });
});

test('hashes every extracted file in sorted order', async (context) => {
  const root = await fixtureRoot(context);
  const archive = storedZip([
    { name: 'z-runtime.dll', contents: 'z' },
    { name: 'llama-server.exe', contents: 'server' },
    { name: 'a-runtime.dll', contents: 'a' },
  ]);

  const manifest = await prepareWindowsWorkerArchive(root, {
    release: fixtureRelease(archive),
    fetchFn: async () => new Response(archive, { status: 200 }),
    log: () => undefined,
  });

  const bundle = manifest.platforms['win32-x64'].bundles.sycl;
  assert.deepEqual(bundle.files.map(({ path: filePath }) => filePath), [
    'resources/workers/win32-x64/sycl/a-runtime.dll',
    'resources/workers/win32-x64/sycl/llama-server.exe',
    'resources/workers/win32-x64/sycl/z-runtime.dll',
  ]);
  assert.equal(bundle.files[0].sha256, sha256(Buffer.from('a')));
  assert.equal(bundle.files[1].sha256, sha256(Buffer.from('server')));
  assert.equal(bundle.files[2].sha256, sha256(Buffer.from('z')));
});

test('publishes the extracted Windows tree when Windows rejects a directory rename', async (context) => {
  const root = await fixtureRoot(context);
  const windowsDirectory = path.join(root, 'resources', 'workers', 'win32-x64');
  const archive = storedZip([
    { name: 'llama-server.exe', contents: 'server' },
    { name: 'runtime.dll', contents: 'runtime' },
  ]);
  const { rename } = await import('node:fs/promises');

  const manifest = await prepareWindowsWorkerArchive(root, {
    release: fixtureRelease(archive),
    fetchFn: async () => new Response(archive, { status: 200 }),
    log: () => undefined,
    renameFile: async (source, destination) => {
      if (destination === windowsDirectory && source.includes('.win32-x64-stage-')) {
        const error = new Error('operation not permitted');
        error.code = 'EPERM';
        throw error;
      }
      await rename(source, destination);
    },
  });

  assert.equal(
    await readFile(path.join(windowsDirectory, 'sycl', 'llama-server.exe'), 'utf8'),
    'server',
  );
  assert.equal(manifest.platforms['win32-x64'].bundles.sycl.files.length, 2);
});

test('restores the previous Windows tree and manifest when publication fails', async (context) => {
  const root = await fixtureRoot(context);
  const windowsDirectory = path.join(root, 'resources', 'workers', 'win32-x64');
  const manifestPath = path.join(root, 'resources', 'workers', 'manifest.json');
  await mkdir(windowsDirectory, { recursive: true });
  await writeFile(path.join(windowsDirectory, 'old-worker.exe'), 'old worker');
  const oldManifest = await readFile(manifestPath, 'utf8');
  const archive = storedZip([{ name: 'llama-server.exe', contents: 'new server' }]);
  const { rename } = await import('node:fs/promises');
  let injected = false;

  await assert.rejects(
    prepareWindowsWorkerArchive(root, {
      release: fixtureRelease(archive),
      fetchFn: async () => new Response(archive, { status: 200 }),
      log: () => undefined,
      renameFile: async (source, destination) => {
        if (!injected && destination === manifestPath && source.includes('.manifest-stage-')) {
          injected = true;
          throw new Error('injected manifest publication failure');
        }
        await rename(source, destination);
      },
    }),
    /injected manifest publication failure/,
  );

  assert.equal(await readFile(path.join(windowsDirectory, 'old-worker.exe'), 'utf8'), 'old worker');
  assert.equal(await readFile(manifestPath, 'utf8'), oldManifest);
});

async function fixtureRoot(context) {
  const root = await mkdtemp(path.join(tmpdir(), 'local-llm-windows-archive-'));
  context.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(path.join(root, 'resources', 'workers'), { recursive: true });
  await writeFile(path.join(root, 'resources', 'workers', 'manifest.json'), `${JSON.stringify({
    llamaCppCommit: WINDOWS_WORKER_RELEASE.commit,
    llamaCppBuild: WINDOWS_WORKER_RELEASE.build,
    workers: {
      'darwin-arm64': {
        path: 'resources/workers/darwin-arm64/llama-server',
        sha256: 'c'.repeat(64),
      },
    },
  }, null, 2)}\n`);
  return root;
}

function fixtureRelease(archive) {
  return {
    ...WINDOWS_WORKER_RELEASE,
    assetName: 'fixture-win-sycl.zip',
    url: 'https://example.invalid/fixture-win-sycl.zip',
    size: archive.length,
    sha256: sha256(archive),
  };
}

function sha256(buffer) {
  return createHash('sha256').update(buffer).digest('hex');
}

function storedZip(entries) {
  const localParts = [];
  const centralParts = [];
  let offset = 0;
  for (const entry of entries) {
    const name = Buffer.from(entry.name, 'utf8');
    const contents = Buffer.from(entry.contents ?? '');
    const crc = crc32(contents);
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(20, 4);
    local.writeUInt32LE(crc, 14);
    local.writeUInt32LE(contents.length, 18);
    local.writeUInt32LE(contents.length, 22);
    local.writeUInt16LE(name.length, 26);
    localParts.push(local, name, contents);

    const central = Buffer.alloc(46);
    central.writeUInt32LE(0x02014b50, 0);
    central.writeUInt16LE(0x0314, 4);
    central.writeUInt16LE(20, 6);
    central.writeUInt32LE(crc, 16);
    central.writeUInt32LE(contents.length, 20);
    central.writeUInt32LE(contents.length, 24);
    central.writeUInt16LE(name.length, 28);
    central.writeUInt32LE(((entry.mode ?? 0o100644) << 16) >>> 0, 38);
    central.writeUInt32LE(offset, 42);
    centralParts.push(central, name);
    offset += local.length + name.length + contents.length;
  }
  const centralDirectory = Buffer.concat(centralParts);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(entries.length, 8);
  end.writeUInt16LE(entries.length, 10);
  end.writeUInt32LE(centralDirectory.length, 12);
  end.writeUInt32LE(offset, 16);
  return Buffer.concat([...localParts, centralDirectory, end]);
}

function crc32(buffer) {
  let crc = 0xffffffff;
  for (const byte of buffer) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1) {
      crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}
