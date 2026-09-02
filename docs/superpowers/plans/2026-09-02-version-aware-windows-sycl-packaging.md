# Version-Aware Windows SYCL Packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package the Windows SYCL worker from its actual oneAPI runtime dependency closure without hard-coding oneAPI 2025.3.3 filenames, and reject any staged bundle that cannot discover `SYCL0` without the oneAPI development environment.

**Architecture:** Keep the existing atomic bundle assembler and version-2 worker manifest. Add a focused runtime-discovery module that finds semantic Unified Runtime resources beside the SYCL runtime selected by PE dependency closure, constructs a clean verification environment, and runs the staged worker before publication. The build entry point records the active compiler banner and supplies a captured process runner; the worker manifest continues to hash the arbitrary file set returned by the assembler.

**Tech Stack:** Node.js 22+, JavaScript ES modules, Node test runner, Windows PE `dumpbin`, CMake/Ninja, llama.cpp `60eeeb6082c1126bb8bc72902c83123cd056811b`, Intel oneAPI, Visual Studio 2022.

**Spec:** `docs/superpowers/specs/2026-09-02-version-aware-windows-sycl-packaging-design.md`

## Global Constraints

- Windows `auto` selects the SYCL bundle; `cpu` remains explicit; no SYCL failure launches the CPU worker.
- Do not maintain an exact-filename catalog for oneAPI releases and do not require `sycl8.dll` by name.
- Do not copy complete oneAPI component directories or non-Intel Unified Runtime adapters.
- All non-system PE imports must resolve recursively; dynamically loaded Level Zero resources must be added semantically.
- Every copied Intel component must contribute controlling license material.
- The previous bundle and manifest remain untouched until staging, licensing, clean launch, and `SYCL0` discovery all pass.
- Clean verification removes oneAPI development paths and variables and intentionally requires a compatible Intel GPU machine.
- The worker manifest schema and application runtime selection remain unchanged.
- macOS tests validate packaging logic only; they do not prove Windows or Intel Arc runtime success.

---

## File structure

- Create `scripts/windows-sycl-runtime.mjs`: version-neutral runtime-role discovery, clean environment construction, and staged device verification.
- Create `scripts/windows-sycl-runtime.test.mjs`: focused unit coverage for runtime roles and clean verification.
- Modify `scripts/windows-sycl-bundle.mjs`: two-pass dependency closure, all-Intel-file license classification, and pre-publication staged verification.
- Modify `scripts/windows-sycl-bundle.test.mjs`: version-neutral fixtures, dependency diagnostics, licensing, and atomic failure tests.
- Modify `scripts/windows-oneapi.mjs`: compiler-banner identification using an injected process runner.
- Modify `scripts/windows-oneapi.test.mjs`: compiler-identification success and failure coverage.
- Modify `scripts/build-worker.mjs`: captured child-process results, compiler logging, and bundler verification wiring.
- Modify `scripts/package-workers.test.mjs`: remove the 2025-specific representative filename.
- Modify `docs/WINDOWS_SYCL_HANDOFF.md`: describe version-aware packaging logs and the automatic clean-environment Arc gate.

### Task 1: Discover semantic SYCL runtime resources

**Files:**
- Create: `scripts/windows-sycl-runtime.mjs`
- Create: `scripts/windows-sycl-runtime.test.mjs`

**Interfaces:**
- Produces: `activeOneApiRuntimeDirectories(oneApiRoot, pathDirectories): string[]`
- Produces: `discoverSyclDynamicResources(activeDirectories): Promise<string[]>`
- Produces: `createSyclVerificationEnvironment(baseEnvironment, { staging, systemRoot }): Record<string, string>`
- Produces: `verifyStagedSyclBundle({ staging, systemRoot, baseEnvironment, runProcess }): Promise<void>`
- `runProcess(command, args, options)` returns `{ code: number | null, stdout: string, stderr: string }`.

- [ ] **Step 1: Write failing tests for active oneAPI directories and version-neutral runtime roles**

Create `scripts/windows-sycl-runtime.test.mjs` with temporary fixtures. Use an arbitrary renamed runtime (`sycl42.dll`) so the test cannot pass through the old `sycl8.dll` assumption:

```js
test('discovers dynamic Level Zero resources beside the active Unified Runtime loader', async (context) => {
  const fixture = await runtimeFixture(context, 'sycl42.dll');
  const files = await discoverSyclDynamicResources([fixture.compilerBin]);
  assert.deepEqual(files.map(path.basename).sort(), [
    'libsycl-fallback-bfloat16.spv',
    'libsycl-native-bfloat16.spv',
    'ur_adapter_level_zero.dll',
    'ur_adapter_level_zero_v2.dll',
    'ur_loader.dll',
    'ur_win_proxy_loader.dll',
  ].sort());
});

test('restricts runtime discovery to active PATH directories below ONEAPI_ROOT', () => {
  assert.deepEqual(activeOneApiRuntimeDirectories('/oneapi', [
    '/oneapi/compiler/2026.1/bin',
    '/unrelated/bin',
    '/oneapi/mkl/2026.1/bin',
  ]), [
    '/oneapi/compiler/2026.1/bin',
    '/oneapi/mkl/2026.1/bin',
  ]);
});

test('reports a missing semantic runtime role and searched directories', async (context) => {
  const fixture = await runtimeFixture(context, 'sycl42.dll');
  await rm(fixture.file('ur_loader.dll'));
  await assert.rejects(
    discoverSyclDynamicResources([fixture.compilerBin]),
    /Unified Runtime loader.*ur_loader\.dll.*compiler.*bin/is,
  );
});

test('rejects distinct active Unified Runtime loaders instead of choosing by PATH order', async (context) => {
  const first = await runtimeFixture(context, 'sycl42.dll', 'compiler/2026.1/bin');
  const second = await runtimeFixture(context, 'sycl42.dll', 'compiler/latest/bin');
  await assert.rejects(
    discoverSyclDynamicResources([first.compilerBin, second.compilerBin]),
    /exactly one active Unified Runtime loader.*found 2/is,
  );
});
```

`runtimeFixture(context, syclRuntimeName, compilerDirectory = 'compiler/2026.1/bin')`
creates both Level Zero adapter variants, two `libsycl-*.spv` files, the
loader, and optional proxy loader beneath the requested compiler directory.

- [ ] **Step 2: Run the focused test and confirm the module is missing**

Run: `node --test scripts/windows-sycl-runtime.test.mjs`

Expected: FAIL with `ERR_MODULE_NOT_FOUND` for `windows-sycl-runtime.mjs`.

- [ ] **Step 3: Implement active-directory selection and semantic discovery**

Create `scripts/windows-sycl-runtime.mjs` with these core rules:

```js
const LEVEL_ZERO_ADAPTER = /^ur_adapter_level_zero.*\.dll$/i;
const SYCL_DEVICE_RESOURCE = /^libsycl-.*\.spv$/i;

export function activeOneApiRuntimeDirectories(oneApiRoot, pathDirectories) {
  const active = uniquePaths(pathDirectories.filter((directory) => isAtOrBelow(oneApiRoot, directory)));
  if (active.length === 0) {
    throw new Error(`No active oneAPI runtime directories were found below ${oneApiRoot}.`);
  }
  return active;
}

export async function discoverSyclDynamicResources(activeDirectories) {
  const loaderFiles = await findMatchingFiles(activeDirectories, (name) => /^ur_loader\.dll$/i.test(name));
  if (loaderFiles.length !== 1) {
    throw new Error(`Expected exactly one active Unified Runtime loader (ur_loader.dll), found ${loaderFiles.length}. Searched: ${activeDirectories.join(', ')}.`);
  }
  const runtimeDirectory = path.dirname(loaderFiles[0]);
  const entries = (await readdir(runtimeDirectory, { withFileTypes: true }))
    .filter((entry) => entry.isFile())
    .map((entry) => entry.name);
  const adapters = entries.filter((name) => LEVEL_ZERO_ADAPTER.test(name));
  const resources = entries.filter((name) => SYCL_DEVICE_RESOURCE.test(name));
  requireRole('Level Zero Unified Runtime adapter', adapters, activeDirectories);
  requireRole('SYCL device resource (libsycl-*.spv)', resources, activeDirectories);
  const proxy = entries.filter((name) => /^ur_win_proxy_loader\.dll$/i.test(name));
  return [path.basename(loaderFiles[0]), ...adapters, ...resources, ...proxy]
    .sort()
    .map((name) => path.join(runtimeDirectory, name));
}
```

`uniquePaths` and `isAtOrBelow` compare normalized absolute paths
case-insensitively because the production host is Windows. `findMatchingFiles`
scans only the active directories and rejects distinct loader matches instead
of selecting by PATH order. `requireRole` includes the semantic role and all
searched directories in its error.

- [ ] **Step 4: Run the discovery tests**

Run: `node --test scripts/windows-sycl-runtime.test.mjs`

Expected: the discovery tests PASS.

- [ ] **Step 5: Write failing tests for clean environment construction and staged verification**

Add:

```js
test('staged verification removes oneAPI development paths and variables', async () => {
  let observed;
  await verifyStagedSyclBundle({
    staging: 'C:\\bundle',
    systemRoot: 'C:\\Windows',
    baseEnvironment: {
      PATH: 'C:\\Intel\\oneAPI\\compiler\\latest\\bin;C:\\other',
      ONEAPI_ROOT: 'C:\\Intel\\oneAPI',
      CMPLR_ROOT: 'C:\\Intel\\oneAPI\\compiler\\latest',
      LIB: 'C:\\Intel\\lib',
      INCLUDE: 'C:\\Intel\\include',
      SystemRoot: 'C:\\Windows',
      KEEP_ME: 'yes',
    },
    runProcess: async (command, args, options) => {
      observed = { command, args, options };
      return { code: 0, stdout: 'SYCL0: Intel Arc Graphics\r\n', stderr: '' };
    },
  });
  assert.equal(observed.command, 'C:\\bundle\\llama-server.exe');
  assert.deepEqual(observed.args, ['--list-devices']);
  assert.equal(observed.options.env.PATH, 'C:\\bundle;C:\\Windows\\System32;C:\\Windows');
  assert.equal(observed.options.env.ONEAPI_ROOT, undefined);
  assert.equal(observed.options.env.CMPLR_ROOT, undefined);
  assert.equal(observed.options.env.LIB, undefined);
  assert.equal(observed.options.env.KEEP_ME, 'yes');
});

test('staged verification preserves output when the worker cannot load', async () => {
  await assert.rejects(verifyStagedSyclBundle({
    staging: 'C:\\bundle',
    systemRoot: 'C:\\Windows',
    baseEnvironment: {},
    runProcess: async () => ({ code: 3221225781, stdout: '', stderr: 'missing runtime' }),
  }), /clean-environment launch.*3221225781.*missing runtime/is);
});

test('staged verification requires SYCL0', async () => {
  await assert.rejects(verifyStagedSyclBundle({
    staging: 'C:\\bundle',
    systemRoot: 'C:\\Windows',
    baseEnvironment: {},
    runProcess: async () => ({ code: 0, stdout: 'no devices', stderr: '' }),
  }), /did not report SYCL0.*no devices/is);
});
```

- [ ] **Step 6: Run the clean-verification tests and confirm they fail**

Run: `node --test scripts/windows-sycl-runtime.test.mjs`

Expected: FAIL because the verification exports are not implemented.

- [ ] **Step 7: Implement clean environment construction and staged verification**

Add:

```js
const DEVELOPMENT_KEYS = new Set([
  'CMPLR_ROOT', 'CPATH', 'CMAKE_PREFIX_PATH', 'DNNLROOT', 'INCLUDE', 'LIB',
  'LIBPATH', 'MKLROOT', 'TBBROOT', 'TCM_ROOT', 'UMF_ROOT',
]);

export function createSyclVerificationEnvironment(baseEnvironment, { staging, systemRoot }) {
  const clean = { ...baseEnvironment };
  for (const key of Object.keys(clean)) {
    const normalized = key.toUpperCase();
    if (normalized === 'PATH' || normalized.startsWith('ONEAPI_') || DEVELOPMENT_KEYS.has(normalized)) {
      delete clean[key];
    }
  }
  clean.PATH = [staging, path.win32.join(systemRoot, 'System32'), systemRoot].join(';');
  return clean;
}

export async function verifyStagedSyclBundle({
  staging, systemRoot, baseEnvironment, runProcess,
}) {
  const executable = path.win32.join(staging, 'llama-server.exe');
  const env = createSyclVerificationEnvironment(baseEnvironment, { staging, systemRoot });
  const result = await runProcess(executable, ['--list-devices'], { cwd: staging, env });
  const output = [result.stdout, result.stderr].filter(Boolean).join('\n').trim();
  if (result.code !== 0) {
    throw new Error(`SYCL staged clean-environment launch exited with code ${result.code ?? 'unknown'}${output ? `:\n${output}` : '.'}`);
  }
  if (!/(?:^|\s)SYCL0(?:\s|:|$)/m.test(output)) {
    throw new Error(`SYCL staged device discovery did not report SYCL0${output ? `:\n${output}` : '.'}`);
  }
}
```

- [ ] **Step 8: Run the complete new helper test file**

Run: `node --test scripts/windows-sycl-runtime.test.mjs`

Expected: PASS.

- [ ] **Step 9: Commit the isolated runtime helper**

```bash
git add scripts/windows-sycl-runtime.mjs scripts/windows-sycl-runtime.test.mjs
git commit -m "Add version-neutral SYCL runtime discovery"
```

### Task 2: Assemble and license the discovered runtime closure

**Files:**
- Modify: `scripts/windows-sycl-bundle.mjs`
- Modify: `scripts/windows-sycl-bundle.test.mjs`

**Interfaces:**
- Consumes: all four exports from `scripts/windows-sycl-runtime.mjs`.
- Preserves: `assembleWindowsSyclBundle(options): Promise<{ executable: string, files: Array<{ path: string, sha256: string }> }>`
- Changes required options: `environment: Record<string, string>` and `runProcess(command, args, options)`.
- Preserves: `collectDependencyClosure(roots, options): Promise<string[]>`, with improved importing-file diagnostics.

- [ ] **Step 1: Make the fixture version-neutral and express actual PE imports**

Remove the import and fixture use of `REQUIRED_SYCL_COMPANIONS`. Build the
fixture under `oneapi/compiler/2026.1/bin`, using `sycl42.dll` as the imported
runtime and `mkl_sycl_blas.42.dll` as a transitive import. Its default
`runDumpbin` map must establish:

```js
const imports = new Map([
  ['llama-server.exe', ['llama-server-impl.dll']],
  ['ggml-sycl.dll', ['sycl42.dll', 'mkl_sycl_blas.42.dll']],
  ['sycl42.dll', ['VCRUNTIME140.dll', 'KERNEL32.dll']],
  ['mkl_sycl_blas.42.dll', ['mkl_core.42.dll']],
]);
```

The fixture supplies `environment`, `pathDirectories`, and a default
`runProcess` returning `{ code: 0, stdout: 'SYCL0: Intel Arc Graphics', stderr: '' }`.

- [ ] **Step 2: Add failing tests for renamed runtimes, import provenance, and all-component licenses**

Replace the exact companion tests with:

```js
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
```

- [ ] **Step 3: Run the focused bundle tests and confirm the old list fails them**

Run: `node --test scripts/windows-sycl-bundle.test.mjs`

Expected: FAIL because `sycl8.dll` is still required and renamed imports are
not used as the source of truth.

- [ ] **Step 4: Track importing files during recursive dependency closure**

Change queue entries from bare strings to records. Preserve the returned
string array:

```js
const queue = roots.map((absoluteFile) => ({ absoluteFile, importedBy: undefined }));
// ...
const { absoluteFile } = queue.shift();
// ...
if (!resolved) {
  throw new Error(
    `Required dependency ${dllName} imported by ${absoluteFile} could not be resolved. `
    + `Searched: ${options.searchDirectories.join(', ')}.`,
  );
}
queue.push({ absoluteFile: resolved, importedBy: absoluteFile });
```

Pass `searchDirectories` in the collector options. The `importedBy` property is
retained for future chain reporting, while the immediate importer in the error
is `absoluteFile`.

- [ ] **Step 5: Replace fixed companions with two-pass dependency closure**

Delete `REQUIRED_SYCL_COMPANIONS`. In `assembleWindowsSyclBundle`:

```js
const activeDirectories = activeOneApiRuntimeDirectories(options.oneApiRoot, oneApiDirectories);
const closureOptions = {
  imports: async (absoluteFile) => (
    /\.(?:exe|dll)$/i.test(absoluteFile)
      ? parseDumpbinDependents(await options.runDumpbin(absoluteFile))
      : []
  ),
  resolve: resolveFrom(searchDirectories),
  isSystemDependency: (dllName, absoluteFile) => (
    isSystemDependencyName(dllName) || (absoluteFile ? isBelow(system32, absoluteFile) : false)
  ),
  searchDirectories,
};
const baseRoots = [...backendModules, executable];
const baseClosure = await collectDependencyClosure(baseRoots, closureOptions);
const dynamicResources = await discoverSyclDynamicResources(activeDirectories);
const dynamicDlls = dynamicResources.filter((file) => /\.dll$/i.test(file));
const closure = await collectDependencyClosure([...baseRoots, ...dynamicDlls], closureOptions);
const nonPeResources = dynamicResources.filter((file) => !/\.dll$/i.test(file));
const bundleSources = [...closure, ...nonPeResources];
```

Extract the current resolve loop into `resolveFrom(searchDirectories)` without
changing build-output, VC-redist, and system precedence.

Use `const log = options.log ?? console.log` and accumulate system exclusions
inside the `isSystemDependency` callback. After discovery and final closure,
emit stable, prefixed lines with the exact values required by the diagnostics
test:

```js
log(`[sycl-package] oneAPI root: ${options.oneApiRoot}`);
for (const directory of activeDirectories) {
  log(`[sycl-package] active oneAPI runtime directory: ${directory}`);
}
for (const file of dynamicResources) {
  log(`[sycl-package] dynamic SYCL resource: ${file}`);
}
for (const file of bundleSources) {
  log(`[sycl-package] resolved bundle source: ${file}`);
}
for (const dependency of [...excludedSystemDependencies].sort()) {
  log(`[sycl-package] excluded system dependency: ${dependency}`);
}
```

- [ ] **Step 6: Classify every copied Intel file for license collection**

Replace companion-only components with:

```js
function classifyOneApiFiles(oneApiRoot, files) {
  return files.flatMap((absolute) => {
    if (!isAtOrBelow(oneApiRoot, absolute)) return [];
    const relative = path.relative(oneApiRoot, absolute);
    const [component] = relative.split(path.sep);
    if (!component || component === '..') {
      throw new Error(`Could not identify the oneAPI component for ${absolute}.`);
    }
    return [{ absolute, component }];
  });
}
```

Pass `classifyOneApiFiles(options.oneApiRoot, bundleSources)` to
`collectControllingLicenses`. Continue failing when any selected component has
no license or notice file.

- [ ] **Step 7: Verify staging before bundle publication**

After copying dependencies and licenses but before describing, deleting, or
renaming the destination, call:

```js
await verifyStagedSyclBundle({
  staging,
  systemRoot,
  baseEnvironment: options.environment,
  runProcess: options.runProcess,
});
```

The `catch` block continues deleting only the temporary staging directory.

- [ ] **Step 8: Add and run atomic verification-failure tests**

Add one test where `runProcess` returns a DLL-load exit code and one where it
returns success without `SYCL0`. In both, create
`destination/existing.txt` first and assert its contents remain after the
rejection.

Run: `node --test scripts/windows-sycl-bundle.test.mjs`

Expected: PASS, including renamed-runtime, license, import-diagnostic, and
atomic-failure tests.

- [ ] **Step 9: Run both SYCL script test files together**

Run: `node --test scripts/windows-sycl-runtime.test.mjs scripts/windows-sycl-bundle.test.mjs`

Expected: PASS.

- [ ] **Step 10: Commit the assembler change**

```bash
git add scripts/windows-sycl-bundle.mjs scripts/windows-sycl-bundle.test.mjs
git commit -m "Package the discovered Windows SYCL runtime"
```

### Task 3: Identify the compiler and wire clean process capture

**Files:**
- Modify: `scripts/windows-oneapi.mjs`
- Modify: `scripts/windows-oneapi.test.mjs`
- Modify: `scripts/build-worker.mjs`

**Interfaces:**
- Produces: `identifyOneApiCompiler(environment, runProcess): Promise<string>`
- Produces internally: `captureProcess(command, args, options): Promise<{ code, stdout, stderr }>`
- Consumes: `assembleWindowsSyclBundle` options `environment` and `runProcess`.

- [ ] **Step 1: Write failing compiler-identification tests**

Add to `scripts/windows-oneapi.test.mjs`:

```js
test('returns the first non-empty Intel compiler banner line', async () => {
  const banner = await identifyOneApiCompiler({ KEEP_ME: 'yes' }, async (command, args, options) => {
    assert.equal(command, 'icx');
    assert.deepEqual(args, ['--version']);
    assert.equal(options.env.KEEP_ME, 'yes');
    return { code: 0, stdout: '\r\nIntel(R) oneAPI DPC++/C++ Compiler 2026.1.0\r\nBuild 1', stderr: '' };
  });
  assert.equal(banner, 'Intel(R) oneAPI DPC++/C++ Compiler 2026.1.0');
});

test('fails before CMake when the active Intel compiler cannot be identified', async () => {
  await assert.rejects(
    identifyOneApiCompiler({}, async () => ({ code: 1, stdout: '', stderr: 'icx failed' })),
    /identify Intel oneAPI compiler.*code 1.*icx failed/is,
  );
});
```

- [ ] **Step 2: Run the oneAPI tests and confirm the export is missing**

Run: `node --test scripts/windows-oneapi.test.mjs`

Expected: FAIL because `identifyOneApiCompiler` is not exported.

- [ ] **Step 3: Implement compiler identification**

Add the export:

```js
export async function identifyOneApiCompiler(environment, runProcess) {
  const result = await runProcess('icx', ['--version'], { env: environment });
  const output = [result.stdout, result.stderr].filter(Boolean).join('\n').trim();
  if (result.code !== 0) {
    throw new Error(`Could not identify Intel oneAPI compiler; icx exited with code ${result.code ?? 'unknown'}${output ? `:\n${output}` : '.'}`);
  }
  const banner = output.split(/\r?\n/).map((line) => line.trim()).find(Boolean);
  if (!banner || !/intel|oneapi|dpc\+\+/i.test(banner)) {
    throw new Error(`Could not identify Intel oneAPI compiler from icx --version output${output ? `:\n${output}` : '.'}`);
  }
  return banner;
}
```

- [ ] **Step 4: Run the oneAPI tests**

Run: `node --test scripts/windows-oneapi.test.mjs`

Expected: PASS.

- [ ] **Step 5: Add a captured process runner to the build entry point**

In `scripts/build-worker.mjs`, implement a runner that never discards stderr:

```js
async function captureProcess(command, args, options = {}) {
  return await new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: options.cwd ?? root,
      env: cleanBuildEnvironment(options.env ?? process.env),
      stdio: ['ignore', 'pipe', 'pipe'],
      shell: false,
    });
    let stdout = '';
    let stderr = '';
    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');
    child.stdout.on('data', (chunk) => { stdout += chunk; });
    child.stderr.on('data', (chunk) => { stderr += chunk; });
    child.once('error', reject);
    child.once('close', (code) => resolve({ code, stdout, stderr }));
  });
}
```

Refactor `capture` to call this runner and include captured stderr in nonzero
errors. Preserve direct inherited output for the long-running CMake commands.

- [ ] **Step 6: Log the compiler banner and pass verification dependencies**

After loading `syclEnvironment`:

```js
const oneApiCompiler = syclEnvironment
  ? await identifyOneApiCompiler(syclEnvironment, captureProcess)
  : undefined;
if (oneApiCompiler) {
  console.log(`[worker-build] Intel compiler: ${oneApiCompiler}`);
}
```

Pass these additional options when publishing SYCL:

```js
environment,
runProcess: captureProcess,
```

Keep `runDumpbin` as a small adapter around `capture`; the bundler owns the
clean verification environment.

- [ ] **Step 7: Run all Windows script tests**

Run:

```bash
node --test scripts/windows-oneapi.test.mjs scripts/windows-vs-tools.test.mjs scripts/worker-build-options.test.mjs scripts/windows-sycl-runtime.test.mjs scripts/windows-sycl-bundle.test.mjs
```

Expected: PASS.

- [ ] **Step 8: Commit compiler and process wiring**

```bash
git add scripts/windows-oneapi.mjs scripts/windows-oneapi.test.mjs scripts/build-worker.mjs
git commit -m "Verify staged SYCL workers outside oneAPI"
```

### Task 4: Remove the final 2025-specific packaging fixtures and update handoff instructions

**Files:**
- Modify: `scripts/package-workers.test.mjs`
- Modify: `docs/WINDOWS_SYCL_HANDOFF.md`

**Interfaces:**
- Consumes: existing version-2 worker manifest and automatic staged hardware gate.
- Produces: platform packaging regressions with no reference to a required `sycl8.dll`.

- [ ] **Step 1: Replace the representative manifest filename**

In `scripts/package-workers.test.mjs`, replace fixture-only `sycl8.dll` names
with `sycl-runtime.dll`. Keep the missing-file assertion semantic:

```js
await assert.rejects(
  prepareTargetWorkers(fixture.root, 'win32-x64'),
  /Worker bundle file is missing.*sycl-runtime\.dll/is,
);
```

- [ ] **Step 2: Run the packaging tests**

Run: `node --test scripts/package-workers.test.mjs`

Expected: PASS.

- [ ] **Step 3: Update the Windows handoff build and diagnostic contract**

Update `docs/WINDOWS_SYCL_HANDOFF.md` to state:

```markdown
The SYCL packager uses the active oneAPI environment that built the worker. It
does not require a version-specific runtime filename such as `sycl8.dll`.
Build output lists the Intel compiler banner, active oneAPI runtime directories,
resolved non-system dependencies, semantic Level Zero resources, and excluded
system dependencies.

Before publishing the bundle, the build runs the staged
`llama-server.exe --list-devices` with oneAPI development paths and variables
removed. The build fails unless that isolated process reports `SYCL0`. A
failure leaves the prior bundle and worker manifest unchanged; do not copy a
missing DLL manually or switch to the CPU bundle implicitly.
```

Retain the exact build command:

```bash
npm run build:worker -- --target win32-x64 --backend all
```

- [ ] **Step 4: Scan active code and handoff instructions for the obsolete assertion**

Run:

```bash
rg -n "REQUIRED_SYCL_COMPANIONS|Required SYCL runtime file sycl8|compiler/latest/bin/sycl8" scripts docs/WINDOWS_SYCL_HANDOFF.md
```

Expected: no matches. Historical approved specs and plans may retain the old
filename as provenance and must not be rewritten.

- [ ] **Step 5: Commit fixtures and documentation**

```bash
git add scripts/package-workers.test.mjs docs/WINDOWS_SYCL_HANDOFF.md
git commit -m "Document version-aware Windows SYCL packaging"
```

### Task 5: Complete repository verification and delivery

**Files:**
- Verify only unless a test exposes a defect in the files listed above.

**Interfaces:**
- Produces: a clean, pushed `codex/windows-sycl-worker` branch ready for the native Windows acceptance gate.

- [ ] **Step 1: Run focused packaging tests**

Run:

```bash
node --test scripts/windows-sycl-runtime.test.mjs scripts/windows-sycl-bundle.test.mjs scripts/windows-oneapi.test.mjs scripts/package-workers.test.mjs
```

Expected: PASS.

- [ ] **Step 2: Run the complete project test suite**

Run: `npm test`

Expected: PASS.

- [ ] **Step 3: Run type checking and extension build**

Run: `npm run typecheck`

Expected: PASS.

Run: `npm run build`

Expected: PASS.

- [ ] **Step 4: Run platform-neutral packaging checks**

Run: `npm run package -- darwin-arm64`

Do not claim this produces or validates a Windows worker.

Expected: the command succeeds without modifying committed native artifacts.

- [ ] **Step 5: Review the branch diff and confirm scope**

Run:

```bash
git status --short
git diff --check origin/codex/windows-sycl-worker...HEAD
git diff --stat origin/codex/windows-sycl-worker...HEAD
```

Expected: only the approved design/plan, runtime discovery, bundler, build
wiring, tests, and Windows handoff documentation have changed; `git diff
--check` emits no output.

- [ ] **Step 6: Commit any verification-only corrections**

If verification required a correction, commit only the affected approved files:

```bash
git add scripts/windows-sycl-runtime.mjs scripts/windows-sycl-runtime.test.mjs \
  scripts/windows-sycl-bundle.mjs scripts/windows-sycl-bundle.test.mjs \
  scripts/windows-oneapi.mjs scripts/windows-oneapi.test.mjs \
  scripts/build-worker.mjs scripts/package-workers.test.mjs \
  docs/WINDOWS_SYCL_HANDOFF.md
git commit -m "Fix Windows SYCL packaging verification"
```

If no correction was required, do not create an empty commit.

- [ ] **Step 7: Push the implementation branch directly**

Run: `git push origin codex/windows-sycl-worker`

Expected: `origin/codex/windows-sycl-worker` advances to local `HEAD`.

- [ ] **Step 8: Report the native Windows gate without overstating macOS verification**

Report the pushed commit, tests completed locally, and this exact Windows next
step:

```bash
git switch codex/windows-sycl-worker
git pull --ff-only origin codex/windows-sycl-worker
npm run build:worker -- --target win32-x64 --backend all
npm run package -- win32-x64
```

The first command now automatically verifies the staged bundle without oneAPI
development paths and must report `SYCL0`. The native build, VSIX installation,
real GGUF inference, and induced no-fallback failure remain the Windows Arc
acceptance evidence.
