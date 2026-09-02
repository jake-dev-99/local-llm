import assert from 'node:assert/strict';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';

import {
  assembleWindowsSyclBundle,
  collectDependencyClosure,
  parseDumpbinDependents,
  writeUpdatedManifest,
} from './windows-sycl-bundle.mjs';

test('parses only DLL names from dumpbin dependent output', () => {
  assert.deepEqual(parseDumpbinDependents(`
Dump of file llama-server.exe

  Image has the following dependencies:

    llama-server-impl.dll
    KERNEL32.DLL
    not-a-library.txt
  Summary
`), ['llama-server-impl.dll', 'KERNEL32.DLL']);
});

test('collects transitive non-system DLL dependencies once', async () => {
  const imports = new Map([
    ['llama-server.exe', ['llama-server-impl.dll', 'KERNEL32.dll']],
    ['llama-server-impl.dll', ['llama.dll', 'VCRUNTIME140.dll']],
    ['llama.dll', ['ggml.dll']],
    ['ggml.dll', []],
  ]);
  const files = await collectDependencyClosure(['llama-server.exe'], fakeOptions(imports));
  assert.deepEqual(files.sort(), [
    'ggml.dll',
    'llama-server-impl.dll',
    'llama-server.exe',
    'llama.dll',
    'VCRUNTIME140.dll',
  ].sort());
});

test('rejects distinct source files that flatten to the same DLL name', async () => {
  await assert.rejects(
    collectDependencyClosure(
      [path.join('/first', 'shared.dll'), path.join('/second', 'shared.dll')],
      fakeOptions(new Map()),
    ),
    /Dependency filename collision for shared\.dll/,
  );
});

test('copies dynamically loaded llama.cpp CPU and SYCL backend modules', async (context) => {
  const fixture = await syclFixture(context);
  const bundle = await assembleWindowsSyclBundle(fixture.options);
  for (const name of ['ggml-cpu.dll', 'ggml-sycl.dll']) {
    assert.equal(bundle.files.some((file) => file.path.endsWith(`/${name}`)), true);
  }
});

test('fails before publishing when a required llama.cpp backend module is missing', async (context) => {
  const fixture = await syclFixture(context);
  await rm(fixture.build('ggml-cpu.dll'));
  await assert.rejects(
    assembleWindowsSyclBundle(fixture.options),
    /Required llama\.cpp backend module ggml-cpu\.dll was not found/,
  );
});

test('copies the VC redistributable CRT instead of a PATH System32 copy', async (context) => {
  const fixture = await syclFixture(context);
  const vcRuntime = path.join(fixture.options.vcToolsRedistDir, 'x64', 'Microsoft.VC143.CRT');
  const systemRoot = path.join(fixture.options.root, 'Windows');
  const system32 = path.join(systemRoot, 'System32');
  await mkdir(vcRuntime, { recursive: true });
  await mkdir(system32, { recursive: true });
  await writeFile(path.join(vcRuntime, 'VCRUNTIME140.dll'), 'redistributable CRT');
  await writeFile(path.join(system32, 'VCRUNTIME140.dll'), 'System32 CRT');
  fixture.options.pathDirectories = [...fixture.options.pathDirectories, system32];
  fixture.options.systemRoot = systemRoot;
  fixture.options.runDumpbin = async (absoluteFile) => (
    path.basename(absoluteFile).toLowerCase() === 'ggml-cpu.dll'
      ? '    VCRUNTIME140.dll\n'
      : ''
  );

  await assembleWindowsSyclBundle(fixture.options);

  assert.equal(
    await readFile(path.join(fixture.options.destination, 'VCRUNTIME140.dll'), 'utf8'),
    'redistributable CRT',
  );
});

test('bundles renamed SYCL and MKL runtimes from actual dependency closure', async (context) => {
  const fixture = await syclFixture(context);
  const bundle = await assembleWindowsSyclBundle(fixture.options);
  for (const name of ['sycl42.dll', 'mkl_sycl_blas.42.dll', 'mkl_core.42.dll']) {
    assert.equal(bundle.files.some((file) => file.path.endsWith(`/${name}`)), true);
  }
  assert.equal(bundle.files.some((file) => file.path.endsWith('/sycl8.dll')), false);
});

test('reports the importing file for an unresolved non-system dependency', async (context) => {
  const fixture = await syclFixture(context);
  fixture.imports.set('ggml-sycl.dll', ['missing-runtime.dll']);
  await assert.rejects(
    assembleWindowsSyclBundle(fixture.options),
    /missing-runtime\.dll imported by .*ggml-sycl\.dll.*searched/is,
  );
});

test('collects licenses for Intel files found only through PE closure', async (context) => {
  const fixture = await syclFixture(context);
  const bundle = await assembleWindowsSyclBundle(fixture.options);
  assert.equal(bundle.files.some((file) => file.path.includes('/licenses/compiler/')), true);
  assert.equal(bundle.files.some((file) => file.path.includes('/licenses/mkl/')), true);
});

test('logs the active runtime scope, selected resources, resolved files, and system exclusions', async (context) => {
  const fixture = await syclFixture(context);
  const messages = [];
  fixture.options.log = (message) => messages.push(message);
  await assembleWindowsSyclBundle(fixture.options);
  const output = messages.join('\n');
  assert.match(output, /oneAPI root:.*oneapi/is);
  assert.match(output, /active oneAPI runtime directory:.*compiler.*2026\.1.*bin/is);
  assert.match(output, /dynamic SYCL resource:.*ur_adapter_level_zero\.dll/is);
  assert.match(output, /resolved bundle source:.*sycl42\.dll/is);
  assert.match(output, /excluded system dependency:.*KERNEL32\.dll/is);
});

test('does not run the PE dependency inspector on SPIR-V companion data', async (context) => {
  const fixture = await syclFixture(context);
  fixture.options.runDumpbin = async (absoluteFile) => {
    assert.match(absoluteFile, /\.(?:exe|dll)$/i);
    return '';
  };
  await assembleWindowsSyclBundle(fixture.options);
});

test('replaces only the exact sycl destination directory', async (context) => {
  const fixture = await syclFixture(context);
  await writeFile(path.join(fixture.cpuDirectory, 'llama-server.exe'), 'cpu');
  await assembleWindowsSyclBundle(fixture.options);
  assert.equal(await readFile(path.join(fixture.cpuDirectory, 'llama-server.exe'), 'utf8'), 'cpu');
});

test('copies controlling licenses and returns sorted hashes for every file', async (context) => {
  const fixture = await syclFixture(context);
  const bundle = await assembleWindowsSyclBundle(fixture.options);
  const paths = bundle.files.map((file) => file.path);
  assert.deepEqual(paths, [...paths].sort());
  assert.equal(paths.some((file) => file.includes('/licenses/compiler/license.txt')), true);
  assert.equal(paths.some((file) => file.includes('/licenses/mkl/license.txt')), true);
  assert.equal(bundle.files.every((file) => /^[0-9a-f]{64}$/.test(file.sha256)), true);
});

test('requires controlling license material for every copied oneAPI component', async (context) => {
  const fixture = await syclFixture(context);
  await rm(path.join(fixture.options.oneApiRoot, 'mkl/2026.1/licensing'), { recursive: true });
  await assert.rejects(
    assembleWindowsSyclBundle(fixture.options),
    /No controlling license material was found for oneAPI component mkl/,
  );
});

test('keeps the published bundle when staged verification reports a DLL-load failure', async (context) => {
  const fixture = await syclFixture(context);
  const existing = await existingBundleFile(fixture.options.destination);
  fixture.options.runProcess = async () => ({
    code: 3221225781,
    stdout: '',
    stderr: 'The code execution cannot proceed because sycl42.dll was not found.',
  });

  await assert.rejects(assembleWindowsSyclBundle(fixture.options), /clean-environment launch exited with code 3221225781/);
  assert.equal(await readFile(existing, 'utf8'), 'existing bundle');
});

test('keeps the published bundle when staged verification does not report SYCL0', async (context) => {
  const fixture = await syclFixture(context);
  const existing = await existingBundleFile(fixture.options.destination);
  fixture.options.runProcess = async () => ({ code: 0, stdout: 'CPU0: Generic CPU', stderr: '' });

  await assert.rejects(assembleWindowsSyclBundle(fixture.options), /device discovery did not report SYCL0/);
  assert.equal(await readFile(existing, 'utf8'), 'existing bundle');
});

test('updates only one version-2 manifest bundle with stable JSON formatting', async (context) => {
  const fixture = await syclFixture(context);
  const bundle = await assembleWindowsSyclBundle(fixture.options);
  const manifestPath = path.join(fixture.options.root, 'resources/workers/manifest.json');
  const hash = '0'.repeat(64);
  const cpu = {
    executable: 'resources/workers/win32-x64/cpu/llama-server.exe',
    files: [{ path: 'resources/workers/win32-x64/cpu/llama-server.exe', sha256: hash }],
  };
  await writeFile(manifestPath, `${JSON.stringify({
    manifestVersion: 2,
    llamaCppCommit: '60eeeb6082c1126bb8bc72902c83123cd056811b',
    llamaCppBuild: 'b10472',
    platforms: {
      'win32-x64': {
        modes: {
          auto: { bundle: 'sycl', backend: 'sycl' },
          cpu: { bundle: 'cpu', backend: 'cpu' },
        },
        bundles: { sycl: cpu, cpu },
      },
    },
  }, null, 2)}\n`);

  await writeUpdatedManifest(fixture.options.root, 'win32-x64', 'sycl', bundle);

  const serialized = await readFile(manifestPath, 'utf8');
  const updated = JSON.parse(serialized);
  assert.equal(serialized.endsWith('\n'), true);
  assert.deepEqual(updated.platforms['win32-x64'].bundles.cpu, cpu);
  assert.deepEqual(updated.platforms['win32-x64'].bundles.sycl, bundle);
});

test('migrates the legacy manifest after CPU and SYCL bundles are published', async (context) => {
  const root = await mkdtemp(path.join(tmpdir(), 'worker-manifest-migration-'));
  context.after(() => rm(root, { recursive: true, force: true }));
  const manifestPath = path.join(root, 'resources/workers/manifest.json');
  const legacyWindowsWorker = path.join(root, 'resources/workers/win32-x64/llama-server.exe');
  await mkdir(path.dirname(manifestPath), { recursive: true });
  await mkdir(path.dirname(legacyWindowsWorker), { recursive: true });
  await writeFile(legacyWindowsWorker, 'legacy worker');
  await writeFile(manifestPath, `${JSON.stringify({
    llamaCppCommit: '60eeeb6082c1126bb8bc72902c83123cd056811b',
    llamaCppBuild: 'b10472',
    workers: {
      'darwin-arm64': {
        path: 'resources/workers/darwin-arm64/llama-server',
        sha256: 'a'.repeat(64),
      },
      'win32-x64': {
        path: 'resources/workers/win32-x64/llama-server.exe',
        sha256: 'b'.repeat(64),
      },
    },
  }, null, 2)}\n`);
  const cpu = {
    executable: 'resources/workers/win32-x64/cpu/llama-server.exe',
    files: [{
      path: 'resources/workers/win32-x64/cpu/llama-server.exe',
      sha256: 'c'.repeat(64),
    }],
  };
  const sycl = {
    executable: 'resources/workers/win32-x64/sycl/llama-server.exe',
    files: [
      {
        path: 'resources/workers/win32-x64/sycl/llama-server.exe',
        sha256: 'd'.repeat(64),
      },
      {
        path: 'resources/workers/win32-x64/sycl/sycl8.dll',
        sha256: 'e'.repeat(64),
      },
    ],
  };

  await writeUpdatedManifest(root, 'win32-x64', 'cpu', cpu);
  const intermediate = JSON.parse(await readFile(manifestPath, 'utf8'));
  assert.equal(intermediate.manifestVersion, undefined);
  assert.deepEqual(intermediate.workers['win32-x64'], {
    path: cpu.executable,
    sha256: cpu.files[0].sha256,
  });

  await writeUpdatedManifest(root, 'win32-x64', 'sycl', sycl);

  const migrated = JSON.parse(await readFile(manifestPath, 'utf8'));
  assert.equal(migrated.manifestVersion, 2);
  assert.deepEqual(migrated.platforms['darwin-arm64'].modes, {
    auto: { bundle: 'default', backend: 'metal' },
    cpu: { bundle: 'default', backend: 'cpu' },
  });
  assert.deepEqual(migrated.platforms['win32-x64'].modes, {
    auto: { bundle: 'sycl', backend: 'sycl' },
    cpu: { bundle: 'cpu', backend: 'cpu' },
  });
  assert.deepEqual(migrated.platforms['win32-x64'].bundles, { sycl, cpu });
  await assert.rejects(readFile(legacyWindowsWorker), { code: 'ENOENT' });
});

test('rejects a legacy SYCL publish until the CPU bundle is isolated', async (context) => {
  const root = await mkdtemp(path.join(tmpdir(), 'worker-manifest-sycl-first-'));
  context.after(() => rm(root, { recursive: true, force: true }));
  const manifestPath = path.join(root, 'resources/workers/manifest.json');
  await mkdir(path.dirname(manifestPath), { recursive: true });
  await writeFile(manifestPath, `${JSON.stringify({
    llamaCppCommit: '60eeeb6082c1126bb8bc72902c83123cd056811b',
    llamaCppBuild: 'b10472',
    workers: {
      'darwin-arm64': {
        path: 'resources/workers/darwin-arm64/llama-server',
        sha256: 'a'.repeat(64),
      },
      'win32-x64': {
        path: 'resources/workers/win32-x64/llama-server.exe',
        sha256: 'b'.repeat(64),
      },
    },
  }, null, 2)}\n`);

  await assert.rejects(
    writeUpdatedManifest(root, 'win32-x64', 'sycl', {
      executable: 'resources/workers/win32-x64/sycl/llama-server.exe',
      files: [{
        path: 'resources/workers/win32-x64/sycl/llama-server.exe',
        sha256: 'd'.repeat(64),
      }],
    }),
    /Build with --backend all or publish cpu before sycl/,
  );
});

function fakeOptions(imports) {
  return {
    imports: async (file) => imports.get(path.basename(file)) ?? [],
    resolve: async (name) => name,
    isSystemDependency: (name) => /^KERNEL32\.dll$/i.test(name),
  };
}

async function syclFixture(context) {
  const root = await mkdtemp(path.join(tmpdir(), 'sycl-bundle-'));
  context.after(() => rm(root, { recursive: true, force: true }));
  const buildOutput = path.join(root, 'build/bin');
  const oneApiRoot = path.join(root, 'oneapi');
  const compilerBin = path.join(oneApiRoot, 'compiler/2026.1/bin');
  const mklBin = path.join(oneApiRoot, 'mkl/2026.1/bin');
  const destination = path.join(root, 'resources/workers/win32-x64/sycl');
  const cpuDirectory = path.join(root, 'resources/workers/win32-x64/cpu');
  await mkdir(buildOutput, { recursive: true });
  await mkdir(cpuDirectory, { recursive: true });
  for (const name of ['llama-server.exe', 'llama-server-impl.dll', 'ggml-cpu.dll', 'ggml-sycl.dll']) {
    await writeFile(path.join(buildOutput, name), name);
  }
  const runtimeFiles = [
    [compilerBin, 'sycl42.dll'],
    [compilerBin, 'ur_loader.dll'],
    [compilerBin, 'ur_adapter_level_zero.dll'],
    [compilerBin, 'libsycl-fallback-bfloat16.spv'],
    [compilerBin, 'libsycl-native-bfloat16.spv'],
    [mklBin, 'mkl_sycl_blas.42.dll'],
    [mklBin, 'mkl_core.42.dll'],
  ];
  for (const [directory, name] of runtimeFiles) {
    const absolute = path.join(directory, name);
    await mkdir(path.dirname(absolute), { recursive: true });
    await writeFile(absolute, name);
    const licensing = path.join(path.dirname(path.dirname(absolute)), 'licensing');
    await mkdir(licensing, { recursive: true });
    await writeFile(path.join(licensing, 'license.txt'), `license for ${directory}`);
  }
  const vcRuntime = path.join(root, 'vc-redist/x64/Microsoft.VC143.CRT');
  await mkdir(vcRuntime, { recursive: true });
  await writeFile(path.join(vcRuntime, 'VCRUNTIME140.dll'), 'redistributable CRT');
  const imports = new Map([
    ['llama-server.exe', ['llama-server-impl.dll']],
    ['ggml-sycl.dll', ['sycl42.dll', 'mkl_sycl_blas.42.dll']],
    ['sycl42.dll', ['VCRUNTIME140.dll', 'KERNEL32.dll']],
    ['mkl_sycl_blas.42.dll', ['mkl_core.42.dll']],
  ]);
  return {
    cpuDirectory,
    build: (name) => path.join(buildOutput, name),
    imports,
    options: {
      root,
      buildOutput,
      destination,
      oneApiRoot,
      vcToolsRedistDir: path.join(root, 'vc-redist'),
      environment: { PATH: [compilerBin, mklBin].join(';') },
      pathDirectories: [compilerBin, mklBin],
      runProcess: async () => ({ code: 0, stdout: 'SYCL0: Intel Arc Graphics', stderr: '' }),
      runDumpbin: async (absoluteFile) => (
        (imports.get(path.basename(absoluteFile).toLowerCase()) ?? []).map((name) => `    ${name}`).join('\n')
      ),
    },
  };
}

async function existingBundleFile(destination) {
  await mkdir(destination, { recursive: true });
  const existing = path.join(destination, 'existing.txt');
  await writeFile(existing, 'existing bundle');
  return existing;
}
