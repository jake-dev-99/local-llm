import assert from 'node:assert/strict';
import { mkdir, mkdtemp, readFile, readdir, rename as fsRename, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';

import {
  collectDependencyClosure,
  createUpdatedWorkerManifest,
  parseDumpbinDependents,
  prepareWindowsSyclBundle,
  publishWindowsWorkerBuilds,
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
  const { bundle } = await prepareWindowsSyclBundle(fixture.options);
  for (const name of ['ggml-cpu.dll', 'ggml-sycl.dll']) {
    assert.equal(bundle.files.some((file) => file.path.endsWith(`/${name}`)), true);
  }
});

test('fails before publishing when a required llama.cpp backend module is missing', async (context) => {
  const fixture = await syclFixture(context);
  await rm(fixture.build('ggml-cpu.dll'));
  await assert.rejects(
    prepareWindowsSyclBundle(fixture.options),
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

  const prepared = await prepareWindowsSyclBundle(fixture.options);

  assert.equal(
    await readFile(path.join(prepared.staging, 'VCRUNTIME140.dll'), 'utf8'),
    'redistributable CRT',
  );
});

test('bundles renamed SYCL and MKL runtimes from actual dependency closure', async (context) => {
  const fixture = await syclFixture(context);
  const { bundle } = await prepareWindowsSyclBundle(fixture.options);
  for (const name of ['sycl42.dll', 'mkl_sycl_blas.42.dll', 'mkl_core.42.dll']) {
    assert.equal(bundle.files.some((file) => file.path.endsWith(`/${name}`)), true);
  }
  assert.equal(bundle.files.some((file) => file.path.endsWith('/sycl8.dll')), false);
});

test('reports the importing file for an unresolved non-system dependency', async (context) => {
  const fixture = await syclFixture(context);
  fixture.imports.set('ggml-sycl.dll', ['missing-runtime.dll']);
  await assert.rejects(
    prepareWindowsSyclBundle(fixture.options),
    /missing-runtime\.dll imported by .*ggml-sycl\.dll.*searched/is,
  );
});

test('does not resolve a non-system dependency from an unrelated PATH directory', async (context) => {
  const fixture = await syclFixture(context);
  const unrelated = path.join(fixture.options.root, 'unrelated-sdk/bin');
  await mkdir(unrelated, { recursive: true });
  await writeFile(path.join(unrelated, 'decoy-runtime.dll'), 'unrelated PATH DLL');
  fixture.options.pathDirectories = [...fixture.options.pathDirectories, unrelated];
  fixture.imports.set('ggml-sycl.dll', ['decoy-runtime.dll']);

  await assert.rejects(
    prepareWindowsSyclBundle(fixture.options),
    (error) => {
      assert.match(error.message, /decoy-runtime\.dll imported by .*ggml-sycl\.dll.*could not be resolved/is);
      assert.doesNotMatch(error.message, /unrelated-sdk/);
      return true;
    },
  );
});

test('rejects distinct dependency candidates within the active oneAPI tier', async (context) => {
  const fixture = await syclFixture(context);
  const [compilerBin, mklBin] = fixture.options.pathDirectories;
  await writeFile(path.join(compilerBin, 'duplicate-runtime.dll'), 'compiler copy');
  await writeFile(path.join(mklBin, 'duplicate-runtime.dll'), 'mkl copy');
  fixture.imports.set('ggml-sycl.dll', ['duplicate-runtime.dll']);

  await assert.rejects(
    prepareWindowsSyclBundle(fixture.options),
    (error) => {
      assert.match(error.message, /Ambiguous active oneAPI dependency duplicate-runtime\.dll/i);
      assert.match(error.message, /compiler.*2026\.1.*bin.*duplicate-runtime\.dll/is);
      assert.match(error.message, /mkl.*2026\.1.*bin.*duplicate-runtime\.dll/is);
      assert.match(error.message, /imported by .*ggml-sycl\.dll/is);
      return true;
    },
  );
});

test('collapses byte-identical oneAPI component aliases to one payload', async (context) => {
  const fixture = await syclFixture(context);
  const compilerBin = path.join(fixture.options.oneApiRoot, 'compiler/latest/bin');
  const umfBin = path.join(fixture.options.oneApiRoot, 'umf/latest/bin');
  const identicalUmf = Buffer.from('identical oneAPI 2026 UMF runtime');
  await mkdir(compilerBin, { recursive: true });
  await writeFile(path.join(compilerBin, 'UMF.dll'), identicalUmf);
  await mkdir(umfBin, { recursive: true });
  await writeFile(path.join(umfBin, 'UMF.dll'), identicalUmf);
  fixture.options.pathDirectories.push(compilerBin, umfBin);
  fixture.imports.set('ur_adapter_level_zero.dll', ['UMF.dll']);
  const messages = [];
  fixture.options.log = (message) => messages.push(message);

  const { bundle } = await prepareWindowsSyclBundle(fixture.options);

  assert.equal(bundle.files.filter((file) => file.path.endsWith('/UMF.dll')).length, 1);
  assert.match(messages.join('\n'), /identical active oneAPI dependency UMF\.dll/i);
});

test('packages the complete oneAPI 2026 root licensing tree without component-local mappings', async (context) => {
  const fixture = await syclFixture(context);
  const rootLicensing = path.join(fixture.options.oneApiRoot, 'licensing/2026.1');
  await mkdir(path.join(rootLicensing, 'notices'), { recursive: true });
  await writeFile(path.join(rootLicensing, 'EULA.rtf'), 'oneAPI 2026 terms');
  await writeFile(path.join(rootLicensing, 'notices', 'dependencies.txt'), 'third-party terms');

  const { bundle } = await prepareWindowsSyclBundle(fixture.options);
  const paths = bundle.files.map((file) => file.path);

  assert.equal(paths.some((file) => file.endsWith('/licenses/oneapi/2026.1/EULA.rtf')), true);
  assert.equal(
    paths.some((file) => file.endsWith('/licenses/oneapi/2026.1/notices/dependencies.txt')),
    true,
  );
});

test('logs the active runtime scope, selected runtime DLLs, resolved files, and system exclusions', async (context) => {
  const fixture = await syclFixture(context);
  const messages = [];
  fixture.options.log = (message) => messages.push(message);
  await prepareWindowsSyclBundle(fixture.options);
  const output = messages.join('\n');
  assert.match(output, /oneAPI root:.*oneapi/is);
  assert.match(output, /oneAPI 2026 licensing directory:.*oneapi.*licensing/is);
  assert.match(output, /oneAPI license file:.*licensing.*license\.htm/is);
  assert.match(output, /active oneAPI runtime directory:.*compiler.*2026\.1.*bin/is);
  assert.match(output, /dynamic SYCL runtime DLL:.*ur_adapter_level_zero\.dll/is);
  assert.match(output, /resolved bundle source:.*sycl42\.dll/is);
  assert.match(output, /excluded system dependency:.*KERNEL32\.dll/is);
});

test('stages the SYCL bundle without changing the explicit CPU directory', async (context) => {
  const fixture = await syclFixture(context);
  await writeFile(path.join(fixture.cpuDirectory, 'llama-server.exe'), 'cpu');
  await prepareWindowsSyclBundle(fixture.options);
  assert.equal(await readFile(path.join(fixture.cpuDirectory, 'llama-server.exe'), 'utf8'), 'cpu');
});

test('copies the oneAPI root licensing tree and returns sorted hashes for every file', async (context) => {
  const fixture = await syclFixture(context);
  const { bundle } = await prepareWindowsSyclBundle(fixture.options);
  const paths = bundle.files.map((file) => file.path);
  assert.deepEqual(paths, [...paths].sort());
  assert.equal(paths.some((file) => file.includes('/licenses/oneapi/2026.1/license.htm')), true);
  assert.equal(paths.some((file) => file.includes('/licenses/oneapi/2026.1/third-party-programs.txt')), true);
  assert.equal(bundle.files.every((file) => /^[0-9a-f]{64}$/.test(file.sha256)), true);
});

test('requires the oneAPI 2026 root licensing directory', async (context) => {
  const fixture = await syclFixture(context);
  await rm(path.join(fixture.options.oneApiRoot, 'licensing'), { recursive: true });
  await assert.rejects(
    prepareWindowsSyclBundle(fixture.options),
    /oneAPI 2026 licensing directory was not found/,
  );
});

test('rejects an empty oneAPI 2026 root licensing directory', async (context) => {
  const fixture = await syclFixture(context);
  const licensing = path.join(fixture.options.oneApiRoot, 'licensing');
  await rm(licensing, { recursive: true });
  await mkdir(licensing);

  await assert.rejects(
    prepareWindowsSyclBundle(fixture.options),
    /oneAPI 2026 licensing directory contains no files/,
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

  await assert.rejects(
    prepareWindowsSyclBundle(fixture.options),
    /clean-environment launch exited with code 3221225781/,
  );
  assert.equal(await readFile(existing, 'utf8'), 'existing bundle');
});

test('keeps the published bundle when staged verification does not report SYCL0', async (context) => {
  const fixture = await syclFixture(context);
  const existing = await existingBundleFile(fixture.options.destination);
  fixture.options.runProcess = async () => ({ code: 0, stdout: 'CPU0: Generic CPU', stderr: '' });

  await assert.rejects(prepareWindowsSyclBundle(fixture.options), /device discovery did not report SYCL0/);
  assert.equal(await readFile(existing, 'utf8'), 'existing bundle');
});

test('backend all leaves CPU, SYCL, and manifest byte-identical when the clean SYCL gate fails', async (context) => {
  const fixture = await publicationFixture(context);
  const before = await publicationSnapshot(fixture);
  fixture.syclOptions.runProcess = async () => ({
    code: 3221225781,
    stdout: '',
    stderr: 'staged SYCL runtime could not load',
  });

  await assert.rejects(
    publishWindowsWorkerBuilds(fixture.options()),
    /clean-environment launch exited with code 3221225781/,
  );

  assert.deepEqual(await publicationSnapshot(fixture), before);
});

test('rolls back CPU, SYCL, and manifest when a destination rename fails', async (context) => {
  const fixture = await publicationFixture(context);
  const before = await publicationSnapshot(fixture);
  let injected = false;

  await assert.rejects(
    publishWindowsWorkerBuilds(fixture.options({
      fileOperations: {
        rename: async (source, destination) => {
          if (!injected && path.basename(source).startsWith('.sycl-stage-')
              && destination === fixture.syclDirectory) {
            injected = true;
            throw new Error('simulated destination rename failure');
          }
          await fsRename(source, destination);
        },
      },
    })),
    /simulated destination rename failure/,
  );

  assert.equal(injected, true);
  assert.deepEqual(await publicationSnapshot(fixture), before);
});

test('a SYCL-only atomic manifest write failure leaves CPU, SYCL, and manifest byte-identical', async (context) => {
  const fixture = await publicationFixture(context);
  const before = await publicationSnapshot(fixture);
  let injected = false;

  await assert.rejects(
    publishWindowsWorkerBuilds(fixture.options({
      backends: ['sycl'],
      fileOperations: {
        rename: async (source, destination) => {
          if (!injected && source.includes('.manifest-stage-') && destination === fixture.manifestPath) {
            injected = true;
            throw new Error('simulated manifest write failure');
          }
          await fsRename(source, destination);
        },
      },
    })),
    /simulated manifest write failure/,
  );

  assert.equal(injected, true);
  assert.deepEqual(await publicationSnapshot(fixture), before);
});

test('publishes CPU, SYCL, and their validated manifest together', async (context) => {
  const fixture = await publicationFixture(context);

  const published = await publishWindowsWorkerBuilds(fixture.options());

  assert.equal(await readFile(path.join(fixture.cpuDirectory, 'llama-server.exe'), 'utf8'), 'new CPU worker');
  assert.equal(await readFile(path.join(fixture.syclDirectory, 'llama-server.exe'), 'utf8'), 'llama-server.exe');
  const manifest = JSON.parse(await readFile(fixture.manifestPath, 'utf8'));
  assert.deepEqual(manifest.platforms['win32-x64'].bundles.cpu, published.cpu);
  assert.deepEqual(manifest.platforms['win32-x64'].bundles.sycl, published.sycl);
  await assert.rejects(readFile(path.join(fixture.cpuDirectory, 'cpu-resource.bin')), { code: 'ENOENT' });
  await assert.rejects(readFile(path.join(fixture.syclDirectory, 'old-runtime.dll')), { code: 'ENOENT' });
});

test('publishes only the oneAPI 2026 runtime contract after staged SYCL0 verification', async (context) => {
  const fixture = await publicationFixture(context);
  await writeFile(fixture.runtime('obsolete-device-library.spv'), 'not part of the oneAPI 2026 runtime contract');
  let verification;
  fixture.syclOptions.runProcess = async (command, args, options) => {
    verification = { command, args, options };
    return { code: 0, stdout: 'SYCL0: Intel Arc Graphics', stderr: '' };
  };

  const published = await publishWindowsWorkerBuilds(fixture.options());

  assert.deepEqual(verification.args, ['--list-devices']);
  const manifest = JSON.parse(await readFile(fixture.manifestPath, 'utf8'));
  const files = manifest.platforms['win32-x64'].bundles.sycl.files.map((file) => file.path);
  assert.deepEqual(manifest.platforms['win32-x64'].bundles.sycl, published.sycl);
  assert.equal(await readFile(path.join(fixture.syclDirectory, 'ur_loader.dll'), 'utf8'), 'ur_loader.dll');
  assert.equal(
    await readFile(path.join(fixture.syclDirectory, 'ur_adapter_level_zero.dll'), 'utf8'),
    'ur_adapter_level_zero.dll',
  );
  assert.equal(files.some((file) => file.endsWith('/ur_loader.dll')), true);
  assert.equal(files.some((file) => file.endsWith('/ur_adapter_level_zero.dll')), true);
  assert.equal(files.some((file) => /\.spv$/i.test(file)), false);
});

test('backend all transaction migrates the legacy manifest and retires its Windows worker', async (context) => {
  const fixture = await publicationFixture(context);
  const legacyWorker = path.join(path.dirname(fixture.cpuDirectory), 'llama-server.exe');
  await writeFile(fixture.manifestPath, `${JSON.stringify(legacyManifest(), null, 2)}\n`);
  await writeFile(legacyWorker, 'legacy Windows worker');

  const published = await publishWindowsWorkerBuilds(fixture.options());

  const manifest = JSON.parse(await readFile(fixture.manifestPath, 'utf8'));
  assert.equal(manifest.manifestVersion, 2);
  assert.deepEqual(manifest.platforms['win32-x64'].bundles, {
    sycl: published.sycl,
    cpu: published.cpu,
  });
  await assert.rejects(readFile(legacyWorker), { code: 'ENOENT' });
});

test('updates only one version-2 manifest bundle', async (context) => {
  const fixture = await syclFixture(context);
  const { bundle } = await prepareWindowsSyclBundle(fixture.options);
  const hash = '0'.repeat(64);
  const cpu = {
    executable: 'resources/workers/win32-x64/cpu/llama-server.exe',
    files: [{ path: 'resources/workers/win32-x64/cpu/llama-server.exe', sha256: hash }],
  };
  const current = {
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
  };

  const updated = createUpdatedWorkerManifest(current, 'win32-x64', { sycl: bundle });
  assert.deepEqual(updated.platforms['win32-x64'].bundles.cpu, cpu);
  assert.deepEqual(updated.platforms['win32-x64'].bundles.sycl, bundle);
  assert.deepEqual(current.platforms['win32-x64'].bundles.sycl, cpu);
});

test('migrates the legacy manifest after CPU and SYCL bundles are prepared', () => {
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
        path: 'resources/workers/win32-x64/sycl/sycl42.dll',
        sha256: 'e'.repeat(64),
      },
    ],
  };

  const intermediate = createUpdatedWorkerManifest(legacyManifest(), 'win32-x64', { cpu });
  assert.equal(intermediate.manifestVersion, undefined);
  assert.deepEqual(intermediate.workers['win32-x64'], {
    path: cpu.executable,
    sha256: cpu.files[0].sha256,
  });

  const migrated = createUpdatedWorkerManifest(intermediate, 'win32-x64', { sycl });
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
});

test('rejects a legacy SYCL publish until the CPU bundle is isolated', () => {
  assert.throws(
    () => createUpdatedWorkerManifest(legacyManifest(), 'win32-x64', { sycl: {
      executable: 'resources/workers/win32-x64/sycl/llama-server.exe',
      files: [{
        path: 'resources/workers/win32-x64/sycl/llama-server.exe',
        sha256: 'd'.repeat(64),
      }],
    } }),
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
    [mklBin, 'mkl_sycl_blas.42.dll'],
    [mklBin, 'mkl_core.42.dll'],
  ];
  for (const [directory, name] of runtimeFiles) {
    const absolute = path.join(directory, name);
    await mkdir(path.dirname(absolute), { recursive: true });
    await writeFile(absolute, name);
  }
  const licensing = path.join(oneApiRoot, 'licensing/2026.1');
  await mkdir(licensing, { recursive: true });
  await writeFile(path.join(licensing, 'license.htm'), 'oneAPI 2026 license');
  await writeFile(path.join(licensing, 'third-party-programs.txt'), 'oneAPI 2026 third-party terms');
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
    runtime: (name) => path.join(compilerBin, name),
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

async function publicationFixture(context) {
  const sycl = await syclFixture(context);
  const syclDirectory = sycl.options.destination;
  const cpuDirectory = sycl.cpuDirectory;
  const manifestPath = path.join(sycl.options.root, 'resources/workers/manifest.json');
  const cpuBinary = path.join(sycl.options.root, 'new-cpu/llama-server.exe');
  await mkdir(syclDirectory, { recursive: true });
  await mkdir(path.dirname(cpuBinary), { recursive: true });
  await writeFile(path.join(cpuDirectory, 'llama-server.exe'), Buffer.from([0, 1, 2, 3]));
  await writeFile(path.join(cpuDirectory, 'cpu-resource.bin'), Buffer.from([4, 5, 6]));
  await writeFile(path.join(syclDirectory, 'llama-server.exe'), Buffer.from([7, 8, 9]));
  await writeFile(path.join(syclDirectory, 'old-runtime.dll'), Buffer.from([10, 11, 12]));
  await writeFile(cpuBinary, 'new CPU worker');
  const oldCpu = manifestBundle('cpu', ['llama-server.exe', 'cpu-resource.bin'], 'a');
  const oldSycl = manifestBundle('sycl', ['llama-server.exe', 'old-runtime.dll'], 'b');
  await mkdir(path.dirname(manifestPath), { recursive: true });
  await writeFile(manifestPath, Buffer.from(`${JSON.stringify({
    manifestVersion: 2,
    llamaCppCommit: '60eeeb6082c1126bb8bc72902c83123cd056811b',
    llamaCppBuild: 'b10472',
    platforms: {
      'win32-x64': {
        modes: {
          auto: { bundle: 'sycl', backend: 'sycl' },
          cpu: { bundle: 'cpu', backend: 'cpu' },
        },
        bundles: { cpu: oldCpu, sycl: oldSycl },
      },
    },
  }, null, 4)}\n`, 'utf8'));

  return {
    cpuDirectory,
    manifestPath,
    runtime: sycl.runtime,
    syclDirectory,
    syclOptions: sycl.options,
    options({ backends = ['cpu', 'sycl'], fileOperations } = {}) {
      const builds = [];
      if (backends.includes('cpu')) {
        builds.push({ backend: 'cpu', binary: cpuBinary, destination: cpuDirectory });
      }
      if (backends.includes('sycl')) {
        builds.push({
          backend: 'sycl',
          binary: sycl.build('llama-server.exe'),
          destination: syclDirectory,
          bundleOptions: this.syclOptions,
        });
      }
      return {
        root: sycl.options.root,
        target: 'win32-x64',
        builds,
        fileOperations,
      };
    },
  };
}

function manifestBundle(name, files, hashCharacter) {
  const root = `resources/workers/win32-x64/${name}`;
  return {
    executable: `${root}/llama-server.exe`,
    files: files.map((file, index) => ({
      path: `${root}/${file}`,
      sha256: String.fromCharCode(hashCharacter.charCodeAt(0) + index).repeat(64),
    })),
  };
}

function legacyManifest() {
  return {
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
  };
}

async function publicationSnapshot(fixture) {
  return {
    cpu: await directorySnapshot(fixture.cpuDirectory),
    sycl: await directorySnapshot(fixture.syclDirectory),
    manifest: await readFile(fixture.manifestPath),
  };
}

async function directorySnapshot(root) {
  const files = [];
  async function visit(directory) {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const absolute = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        await visit(absolute);
      } else if (entry.isFile()) {
        files.push([
          path.relative(root, absolute).split(path.sep).join('/'),
          (await readFile(absolute)).toString('base64'),
        ]);
      }
    }
  }
  await visit(root);
  return files.sort(([left], [right]) => left.localeCompare(right));
}
