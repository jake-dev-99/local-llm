# Self-Contained Windows SYCL Worker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a Windows VSIX containing a self-contained Intel SYCL worker and the existing CPU worker, with explicit backend selection and no automatic SYCL-to-CPU fallback.

**Architecture:** Add a versioned, bundle-oriented worker manifest shared by runtime and packaging. Build Windows CPU and SYCL workers into isolated directories, copy and hash the SYCL runtime closure, preflight `SYCL0`, and launch exactly one selected backend. Keep the production manifest and current runtime behavior unchanged until the native Windows artifacts exist, then switch the manifest, runtime, and packaging atomically.

**Tech Stack:** TypeScript 7, Node.js 22/26, Node test runner, esbuild, CMake, llama.cpp `b10472`, Visual Studio 2022, Intel oneAPI 2025.3.3, Intel Level Zero, VSCE.

**Spec:** `docs/superpowers/specs/2026-08-24-windows-sycl-worker-design.md`

## Global Constraints

- The SYCL worker must be built from llama.cpp commit `60eeeb6082c1126bb8bc72902c83123cd056811b` with `GGML_SYCL=ON`.
- The Windows VSIX must contain isolated `resources/workers/win32-x64/sycl/` and `resources/workers/win32-x64/cpu/` bundles.
- The installed extension must require no Intel oneAPI installation; a compatible Intel graphics driver is the only external GPU runtime prerequisite.
- `localLlm.acceleration: auto` means Metal on `darwin-arm64` and SYCL on `win32-x64`.
- `localLlm.acceleration: cpu` is the only way to select the CPU worker on Windows.
- A SYCL discovery, DLL, allocation, model-load, or startup failure must be reported without starting the CPU bundle.
- Existing macOS Metal fitting and macOS CPU behavior must remain unchanged.
- Packaging must hash and verify every declared worker executable, DLL, data file, and bundled license file.
- Windows SYCL runtime success is not complete until the Arc hardware smoke test has loaded a model and completed inference.
- Do not enable `GGML_SYCL_F16` in this change; establish the FP32 correctness baseline before performance tuning.

---

## File Structure

### New files

- `src/worker/workerManifest.ts`: version-2 schema parsing, safe path checks, mode-to-bundle resolution, hashing, bundle verification, and immutable manifest updates.
- `src/worker/workerManifest.test.ts`: schema, path safety, completeness, hash, and mode-selection tests.
- `scripts/worker-build-options.mjs`: pure CLI parsing and CMake-option construction for testable build orchestration.
- `scripts/worker-build-options.test.mjs`: CPU, SYCL, native/cross-build, and default-option tests.
- `scripts/windows-oneapi.mjs`: loads the oneAPI environment through `setvars.bat` when called from Git Bash/zsh and resolves Intel/MSVC redistributable paths.
- `scripts/windows-oneapi.test.mjs`: environment parsing and missing-toolchain diagnostics.
- `scripts/windows-sycl-bundle.mjs`: copies llama.cpp build outputs, Intel runtime companions, MSVC redistributables, SPIR-V data, and license files into the isolated SYCL bundle.
- `scripts/windows-sycl-bundle.test.mjs`: dependency-closure, companion-file, missing-file, cleanup-scope, and manifest-update tests.
- `src/worker/syclDevice.ts`: parses `--list-devices` output and runs the SYCL preflight.
- `src/worker/syclDevice.test.ts`: valid device, missing device, process failure, and diagnostic-preservation tests.
- `src/worker/workerLaunch.ts`: resolves one platform/mode launch plan and contains the no-fallback contract.
- `src/worker/workerLaunch.test.ts`: platform mapping and one-bundle-only failure tests.
- `scripts/package-workers.mjs`: target bundle verification and target-specific ignore rules used by VSIX packaging.
- `scripts/package-workers.test.mjs`: fixture-based Windows/Darwin package validation.
- `scripts/smoke-worker.mjs`: native worker preflight, health, and one-token inference smoke utility.

### Modified files

- `scripts/build-worker.mjs`: thin CLI orchestrator over the new build and bundle modules.
- `scripts/package-vsix.mjs`: verify all target bundles through the shared manifest module before invoking VSCE.
- `src/worker/workerIntegrity.ts`: return a verified bundle selection instead of one platform executable.
- `src/worker/workerManager.ts`: preflight SYCL, launch the resolved backend, preserve it across restarts, and report actionable failures.
- `src/worker/workerLaunchArguments.integration.test.ts`: cover SYCL arguments while retaining Metal and CPU assertions.
- `resources/workers/manifest.json`: migrate to manifest version 2 with Darwin, Windows SYCL, and Windows CPU bundles.
- `resources/workers/win32-x64/llama-server.exe`: move to `resources/workers/win32-x64/cpu/llama-server.exe`.
- `package.json`: update acceleration descriptions and add the worker smoke command.
- `README.md`: document Windows SYCL defaults, native build requirements, explicit CPU mode, profile-aware installation, and smoke commands.
- `THIRD_PARTY_NOTICES.md`: list the pinned Intel runtime components and the controlling license files copied from the oneAPI installation.

---

### Task 1: Versioned worker-manifest contract

**Files:**
- Create: `src/worker/workerManifest.ts`
- Create: `src/worker/workerManifest.test.ts`

**Interfaces:**
- Produces: `type AccelerationMode = 'auto' | 'cpu'`
- Produces: `type WorkerBackend = 'metal' | 'sycl' | 'cpu'`
- Produces: `parseWorkerManifest(value: unknown): WorkerManifestV2`
- Produces: `resolveWorkerBundle(manifest: WorkerManifestV2, target: string, mode: AccelerationMode): ResolvedWorkerBundle`
- Produces: `verifyWorkerBundleFiles(root: string, bundle: WorkerBundle): Promise<void>`
- Produces: `verifyPlatformBundles(root: string, manifest: WorkerManifestV2, target: string): Promise<void>`
- Produces: `replaceWorkerBundle(manifest: WorkerManifestV2, target: string, bundleName: string, bundle: WorkerBundle): WorkerManifestV2`
- Produces: `sha256File(filePath: string): Promise<string>`

- [ ] **Step 1: Write the failing schema and resolution tests**

```ts
test('Windows auto selects only the SYCL bundle', () => {
  const manifest = parseWorkerManifest(validManifest());
  assert.deepEqual(resolveWorkerBundle(manifest, 'win32-x64', 'auto'), {
    target: 'win32-x64',
    bundleName: 'sycl',
    backend: 'sycl',
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
  assert.equal(
    resolveWorkerBundle(manifest, 'darwin-arm64', 'auto').executable,
    resolveWorkerBundle(manifest, 'darwin-arm64', 'cpu').executable,
  );
});
```

- [ ] **Step 2: Add failing safety and integrity tests**

```ts
test('rejects paths outside the selected platform directory', () => {
  const value = validManifest();
  value.platforms['win32-x64'].bundles.sycl.files[0].path = '../sycl8.dll';
  assert.throws(() => parseWorkerManifest(value), /safe relative path/);
});

test('rejects duplicate paths and an executable omitted from files', () => {
  const value = validManifest();
  value.platforms['win32-x64'].bundles.sycl.files.push(
    value.platforms['win32-x64'].bundles.sycl.files[0],
  );
  assert.throws(() => parseWorkerManifest(value), /duplicate worker file/);
});

test('verifies every file hash in a bundle', async (context) => {
  const fixture = await createBundleFixture(context);
  await verifyWorkerBundleFiles(fixture.root, fixture.bundle);
  await writeFile(fixture.dllPath, 'tampered');
  await assert.rejects(
    verifyWorkerBundleFiles(fixture.root, fixture.bundle),
    /failed its SHA-256 integrity check/,
  );
});
```

Define the test fixtures in the same file so the tests are standalone:

```ts
function validManifest(): Record<string, any> {
  const hash = 'a'.repeat(64);
  return {
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
          default: {
            executable: 'resources/workers/darwin-arm64/llama-server',
            files: [{ path: 'resources/workers/darwin-arm64/llama-server', sha256: hash }],
          },
        },
      },
      'win32-x64': {
        modes: {
          auto: { bundle: 'sycl', backend: 'sycl' },
          cpu: { bundle: 'cpu', backend: 'cpu' },
        },
        bundles: {
          sycl: {
            executable: 'resources/workers/win32-x64/sycl/llama-server.exe',
            files: [
              { path: 'resources/workers/win32-x64/sycl/llama-server.exe', sha256: hash },
              { path: 'resources/workers/win32-x64/sycl/sycl8.dll', sha256: hash },
            ],
          },
          cpu: {
            executable: 'resources/workers/win32-x64/cpu/llama-server.exe',
            files: [{ path: 'resources/workers/win32-x64/cpu/llama-server.exe', sha256: hash }],
          },
        },
      },
    },
  };
}

async function createBundleFixture(context: TestContext) {
  const root = await mkdtemp(path.join(tmpdir(), 'worker-manifest-'));
  context.after(() => rm(root, { recursive: true, force: true }));
  const executablePath = path.join(root, 'resources/workers/win32-x64/sycl/llama-server.exe');
  const dllPath = path.join(root, 'resources/workers/win32-x64/sycl/sycl8.dll');
  await mkdir(path.dirname(executablePath), { recursive: true });
  await writeFile(executablePath, 'server');
  await writeFile(dllPath, 'runtime');
  return {
    root,
    dllPath,
    bundle: {
      executable: 'resources/workers/win32-x64/sycl/llama-server.exe',
      files: [
        { path: 'resources/workers/win32-x64/sycl/llama-server.exe', sha256: await sha256File(executablePath) },
        { path: 'resources/workers/win32-x64/sycl/sycl8.dll', sha256: await sha256File(dllPath) },
      ],
    },
  };
}
```

- [ ] **Step 3: Run the tests and confirm the contract is absent**

Run: `node --disable-warning=MODULE_TYPELESS_PACKAGE_JSON --test src/worker/workerManifest.test.ts`

Expected: FAIL because `workerManifest.ts` does not exist.

- [ ] **Step 4: Implement the schema and immutable selectors**

```ts
export interface WorkerFile {
  path: string;
  sha256: string;
}

export interface WorkerBundle {
  executable: string;
  files: WorkerFile[];
}

export interface WorkerMode {
  bundle: string;
  backend: WorkerBackend;
}

export interface WorkerPlatform {
  modes: Record<AccelerationMode, WorkerMode>;
  bundles: Record<string, WorkerBundle>;
}

export interface WorkerManifestV2 {
  manifestVersion: 2;
  llamaCppCommit: string;
  llamaCppBuild: string;
  platforms: Record<string, WorkerPlatform>;
}

export interface ResolvedWorkerBundle extends WorkerBundle {
  target: string;
  bundleName: string;
  backend: WorkerBackend;
}
```

Implement strict object/array/string checks, require 64 lowercase-or-uppercase hexadecimal SHA-256 values, normalize manifest paths with POSIX separators, require each path to remain below its declared `resources/workers/` platform and bundle directory, and require the executable to appear exactly once in `files`.

- [ ] **Step 5: Implement complete bundle hashing and verification**

```ts
export async function verifyWorkerBundleFiles(
  root: string,
  bundle: WorkerBundle,
): Promise<void> {
  for (const file of bundle.files) {
    const absolutePath = path.join(root, ...file.path.split('/'));
    const actual = await sha256File(absolutePath);
    if (actual.toLowerCase() !== file.sha256.toLowerCase()) {
      throw new Error(
        `Bundled worker file ${file.path} failed its SHA-256 integrity check.`,
      );
    }
  }
}
```

`verifyPlatformBundles` iterates every bundle declared under one target. It does not inspect or require another target's files.

- [ ] **Step 6: Run focused and full tests**

Run: `node --disable-warning=MODULE_TYPELESS_PACKAGE_JSON --test src/worker/workerManifest.test.ts`

Expected: PASS.

Run: `npm test`

Expected: all existing and new tests PASS.

- [ ] **Step 7: Commit the isolated contract**

```bash
git add src/worker/workerManifest.ts src/worker/workerManifest.test.ts
git commit -m "Add versioned worker bundle manifest"
```

### Task 2: Windows backend build orchestration from Git Bash/zsh

**Files:**
- Create: `scripts/worker-build-options.mjs`
- Create: `scripts/worker-build-options.test.mjs`
- Create: `scripts/windows-oneapi.mjs`
- Create: `scripts/windows-oneapi.test.mjs`
- Modify: `scripts/build-worker.mjs`

**Interfaces:**
- Consumes: llama.cpp commit constant already in `scripts/build-worker.mjs`
- Produces: `parseWorkerBuildOptions(argv: string[], hostTarget: string): WorkerBuildOptions`
- Produces: `cmakeOptionsForBuild(options: WorkerBuildOptions & ToolchainPaths): string[]`
- Produces: `parseWindowsEnvironment(stdout: string): NodeJS.ProcessEnv`
- Produces: `loadOneApiEnvironment(baseEnv: NodeJS.ProcessEnv): Promise<NodeJS.ProcessEnv>`
- Produces: `resolveOneApiFiles(environment: NodeJS.ProcessEnv): OneApiPaths`

`OneApiPaths` is:

```js
{
  oneApiRoot: String,
  setvarsPath: String,
  levelZeroSdkPath: String,
  vcToolsRedistDir: String,
}
```

Resolve `levelZeroSdkPath` from `LEVEL_ZERO_V1_SDK_PATH` and fail with a direct diagnostic when it is absent. Resolve `vcToolsRedistDir` from the environment emitted by the Visual Studio/oneAPI setup.

- [ ] **Step 1: Write failing option and CMake tests**

```js
test('native Windows defaults to building both backends', () => {
  assert.deepEqual(parseWorkerBuildOptions([], 'win32-x64'), {
    target: 'win32-x64',
    backend: 'all',
  });
});

test('SYCL is native-Windows-only and uses the pinned backend flags', () => {
  assert.throws(
    () => parseWorkerBuildOptions(['--target', 'win32-x64', '--backend', 'sycl'], 'darwin-arm64'),
    /SYCL worker must be built on native x64 Windows/,
  );
  const flags = cmakeOptionsForBuild({
    target: 'win32-x64',
    backend: 'sycl',
    hostTarget: 'win32-x64',
    oneApiEnvironment: {},
  });
  assert.equal(flags.includes('-DGGML_SYCL=ON'), true);
  assert.equal(flags.includes('-DGGML_SYCL_F16=ON'), false);
  assert.equal(flags.includes('-DBUILD_SHARED_LIBS=ON'), true);
  assert.equal(flags.includes('-DCMAKE_C_COMPILER=cl'), true);
  assert.equal(flags.includes('-DCMAKE_CXX_COMPILER=icx'), true);
});
```

- [ ] **Step 2: Write failing oneAPI environment tests**

```js
test('parses cmd set output without corrupting values containing equals signs', () => {
  assert.deepEqual(parseWindowsEnvironment('ONEAPI_ROOT=C:\\Intel\\oneAPI\r\nPATH=C:\\A;C:\\B=2\r\n'), {
    ONEAPI_ROOT: 'C:\\Intel\\oneAPI',
    PATH: 'C:\\A;C:\\B=2',
  });
});

test('reports the exact missing oneAPI bootstrap path', async () => {
  await assert.rejects(
    loadOneApiEnvironment({ ONEAPI_ROOT: 'C:\\missing' }),
    /C:\\missing\\setvars.bat/,
  );
});
```

- [ ] **Step 3: Run the new script tests and confirm they fail**

Run: `node --test scripts/worker-build-options.test.mjs scripts/windows-oneapi.test.mjs`

Expected: FAIL because both implementation modules are missing.

- [ ] **Step 4: Implement shell-independent oneAPI environment loading**

Use `execFile('cmd.exe', ['/d', '/s', '/c', command])` only to call the required batch file and print the resulting environment. Use this fixed command structure:

```js
const command = `call "${setvarsPath}" intel64 --force >nul && set`;
```

Resolve `setvars.bat` from `ONEAPI_ROOT`, falling back to `C:\\Program Files (x86)\\Intel\\oneAPI\\setvars.bat`. Parse each non-empty line at its first `=` and ignore cmd pseudo-variables whose names begin with `=`. All later CMake and copy processes use `shell: false` with the captured environment.

- [ ] **Step 5: Implement backend-specific CMake options**

For SYCL, append exactly:

```js
[
  '-G', 'Ninja',
  '-DBUILD_SHARED_LIBS=ON',
  '-DGGML_STATIC=OFF',
  '-DGGML_BACKEND_DL=ON',
  '-DGGML_NATIVE=OFF',
  '-DGGML_OPENMP=OFF',
  '-DGGML_METAL=OFF',
  '-DGGML_SYCL=ON',
  '-DGGML_SYCL_TARGET=INTEL',
  '-DCMAKE_C_COMPILER=cl',
  '-DCMAKE_CXX_COMPILER=icx',
]
```

Do not add `GGML_SYCL_F16`. Preserve the existing Darwin flags, native Windows CPU flags, and llvm-mingw CPU cross-build flags.

- [ ] **Step 6: Refactor the CLI to build isolated directories**

Use these build directories and destinations:

```text
build/llama.cpp-darwin-arm64
build/llama.cpp-win32-x64-cpu
build/llama.cpp-win32-x64-sycl
resources/workers/darwin-arm64/llama-server
resources/workers/win32-x64/cpu/llama-server.exe
resources/workers/win32-x64/sycl/llama-server.exe
```

`--backend all` invokes CPU and SYCL builds sequentially after fetching the pinned source once. A single-backend build never deletes the other bundle. Restrict recursive removal to the exact backend build directory.

- [ ] **Step 7: Run script tests and existing verification**

Run: `node --test scripts/worker-build-options.test.mjs scripts/windows-oneapi.test.mjs`

Expected: PASS.

Run: `npm test && npm run typecheck`

Expected: PASS.

- [ ] **Step 8: Commit build orchestration**

```bash
git add scripts/build-worker.mjs scripts/worker-build-options.mjs scripts/worker-build-options.test.mjs scripts/windows-oneapi.mjs scripts/windows-oneapi.test.mjs
git commit -m "Add Windows CPU and SYCL worker builds"
```

### Task 3: Self-contained Windows SYCL bundle assembly

**Files:**
- Create: `scripts/windows-sycl-bundle.mjs`
- Create: `scripts/windows-sycl-bundle.test.mjs`
- Modify: `scripts/build-worker.mjs`

**Interfaces:**
- Consumes: `sha256File` and `replaceWorkerBundle` from `src/worker/workerManifest.ts`
- Consumes: `resolveOneApiFiles` from `scripts/windows-oneapi.mjs`
- Produces: `parseDumpbinDependents(output: string): string[]`
- Produces: `collectDependencyClosure(roots: string[], options: DependencyOptions): Promise<string[]>`
- Produces: `assembleWindowsSyclBundle(options: SyclBundleOptions): Promise<WorkerBundle>`
- Produces: `writeUpdatedManifest(root: string, target: string, bundleName: string, bundle: WorkerBundle): Promise<void>`

The script-level option contracts are:

```js
// DependencyOptions
{
  imports: async (absoluteFile) => ['dependency.dll'],
  resolve: async (dllName) => 'C:/absolute/dependency.dll',
  isSystemDependency: (dllName, absoluteFile) => false,
}

// SyclBundleOptions
{
  root: 'C:/repo',
  buildOutput: 'C:/repo/build/llama.cpp-win32-x64-sycl/bin',
  destination: 'C:/repo/resources/workers/win32-x64/sycl',
  oneApiRoot: 'C:/Program Files (x86)/Intel/oneAPI',
  vcToolsRedistDir: 'C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Redist/MSVC/14.44.35211',
  levelZeroSdkPath: 'C:/level-zero-sdk',
  runDumpbin: async (absoluteFile) => '',
}
```

- [ ] **Step 1: Write failing dependency and companion-file tests**

```js
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

test('copies required SYCL companions that are loaded dynamically', async (context) => {
  const fixture = await syclFixture(context);
  const bundle = await assembleWindowsSyclBundle(fixture.options);
  for (const name of ['sycl8.dll', 'ur_loader.dll', 'ur_adapter_level_zero.dll',
    'libsycl-fallback-bfloat16.spv', 'libsycl-native-bfloat16.spv']) {
    assert.equal(bundle.files.some((file) => file.path.endsWith(`/${name}`)), true);
  }
});
```

- [ ] **Step 2: Add failing missing-runtime and cleanup-scope tests**

```js
test('fails before publishing a bundle when one required runtime is missing', async (context) => {
  const fixture = await syclFixture(context);
  await rm(fixture.runtime('sycl8.dll'));
  await assert.rejects(
    assembleWindowsSyclBundle(fixture.options),
    /Required SYCL runtime file sycl8.dll was not found/,
  );
});

test('replaces only the exact sycl destination directory', async (context) => {
  const fixture = await syclFixture(context);
  await writeFile(path.join(fixture.cpuDirectory, 'llama-server.exe'), 'cpu');
  await assembleWindowsSyclBundle(fixture.options);
  assert.equal(await readFile(path.join(fixture.cpuDirectory, 'llama-server.exe'), 'utf8'), 'cpu');
});
```

Implement the test helpers in `windows-sycl-bundle.test.mjs`:

```js
function fakeOptions(imports) {
  return {
    imports: async (file) => imports.get(path.basename(file)) ?? [],
    resolve: async (name) => name,
    isSystemDependency: (name) => /^KERNEL32\.dll$/i.test(name),
  };
}

function findFixtureRuntime(oneApiRoot, name) {
  const relative = REQUIRED_SYCL_COMPANIONS.find(
    (candidate) => path.basename(candidate).toLowerCase() === name.toLowerCase(),
  );
  if (!relative) {
    throw new Error(`Unknown fixture runtime ${name}`);
  }
  return path.join(oneApiRoot, ...relative.split('/'));
}

async function syclFixture(context) {
  const root = await mkdtemp(path.join(tmpdir(), 'sycl-bundle-'));
  context.after(() => rm(root, { recursive: true, force: true }));
  const buildOutput = path.join(root, 'build/bin');
  const oneApiRoot = path.join(root, 'oneapi');
  const destination = path.join(root, 'resources/workers/win32-x64/sycl');
  const cpuDirectory = path.join(root, 'resources/workers/win32-x64/cpu');
  await mkdir(buildOutput, { recursive: true });
  await mkdir(cpuDirectory, { recursive: true });
  for (const name of ['llama-server.exe', 'llama-server-impl.dll', 'ggml-sycl.dll']) {
    await writeFile(path.join(buildOutput, name), name);
  }
  for (const relative of REQUIRED_SYCL_COMPANIONS) {
    const absolute = path.join(oneApiRoot, ...relative.split('/'));
    await mkdir(path.dirname(absolute), { recursive: true });
    await writeFile(absolute, relative);
    const licensing = path.join(path.dirname(path.dirname(absolute)), 'licensing');
    await mkdir(licensing, { recursive: true });
    await writeFile(path.join(licensing, 'license.txt'), `license for ${relative}`);
  }
  return {
    cpuDirectory,
    runtime: (name) => findFixtureRuntime(oneApiRoot, name),
    options: {
      root,
      buildOutput,
      destination,
      oneApiRoot,
      vcToolsRedistDir: path.join(root, 'vc-redist'),
      levelZeroSdkPath: path.join(root, 'level-zero'),
      runDumpbin: async () => '',
    },
  };
}
```

`findFixtureRuntime` searches `REQUIRED_SYCL_COMPANIONS` by basename and returns its absolute path below `oneApiRoot`; it throws if the name is not in the constant.

- [ ] **Step 3: Run the bundler tests and confirm they fail**

Run: `node --test scripts/windows-sycl-bundle.test.mjs`

Expected: FAIL because `windows-sycl-bundle.mjs` does not exist.

- [ ] **Step 4: Implement PE dependency parsing and system classification**

Parse `dumpbin /dependents` lines matching `^\s+([^\s]+\.dll)\s*$` case-insensitively. Resolve dependencies in this order:

1. the llama.cpp build output directory;
2. the oneAPI environment `PATH` directories;
3. `VCToolsRedistDir/x64/Microsoft.VC143.CRT/`; and
4. `%SystemRoot%/System32`.

Dependencies resolved only from System32 are recorded as system-provided and are not copied. `ze_loader.dll` and the Windows `api-ms-win-*`, `KERNEL32.dll`, `ADVAPI32.dll`, `CRYPT32.dll`, `DBGHELP.dll`, `OpenCL.dll`, `SHELL32.dll`, `SHLWAPI.dll`, `WINTRUST.dll`, and `WS2_32.dll` families are system/driver prerequisites. Every other unresolved import fails the build.

- [ ] **Step 5: Add the pinned oneAPI companion set**

Copy these files from the oneAPI 2025.3.3 environment in addition to the PE closure because Unified Runtime loads adapters and SPIR-V helpers dynamically:

```js
export const REQUIRED_SYCL_COMPANIONS = [
  'compiler/latest/bin/sycl8.dll',
  'compiler/latest/bin/ur_adapter_level_zero.dll',
  'compiler/latest/bin/ur_adapter_level_zero_v2.dll',
  'compiler/latest/bin/ur_adapter_opencl.dll',
  'compiler/latest/bin/ur_loader.dll',
  'compiler/latest/bin/ur_win_proxy_loader.dll',
  'compiler/latest/bin/svml_dispmd.dll',
  'compiler/latest/bin/libmmd.dll',
  'compiler/latest/bin/libiomp5md.dll',
  'compiler/latest/bin/libsycl-fallback-bfloat16.spv',
  'compiler/latest/bin/libsycl-native-bfloat16.spv',
  'mkl/latest/bin/mkl_sycl_blas.5.dll',
  'mkl/latest/bin/mkl_core.2.dll',
  'mkl/latest/bin/mkl_tbb_thread.2.dll',
  'dnnl/latest/bin/dnnl.dll',
  'tbb/latest/bin/tbb12.dll',
  'tcm/latest/bin/tcm.dll',
  'tcm/latest/bin/libhwloc-15.dll',
  'umf/latest/bin/umf.dll',
];
```

Treat the complete companion list as mandatory even when a file does not appear in the PE import table: Unified Runtime, oneMKL, and oneDNN can load these files dynamically. Dependency pruning applies only to unrelated llama.cpp build outputs, not to this pinned oneAPI companion set.

- [ ] **Step 6: Collect controlling license files**

For each oneAPI component directory used above, locate its nearest `licensing` directory and copy its license and third-party notice text files into the matching `resources/workers/win32-x64/sycl/licenses/compiler`, `licenses/mkl`, `licenses/dnnl`, `licenses/tbb`, `licenses/tcm`, or `licenses/umf` directory. Deduplicate identical content by SHA-256. Fail bundle assembly when a copied runtime component has no controlling license material.

- [ ] **Step 7: Hash the complete bundle and update a version-2 manifest**

Sort file entries by POSIX path before serialization. Write JSON with two-space indentation and a trailing newline. The SYCL bundle's executable is:

```text
resources/workers/win32-x64/sycl/llama-server.exe
```

The build script updates only the produced bundle; `--backend all` writes both Windows bundles after both builds succeed.

- [ ] **Step 8: Run focused and full tests**

Run: `node --test scripts/windows-sycl-bundle.test.mjs`

Expected: PASS.

Run: `npm test && npm run typecheck`

Expected: PASS.

- [ ] **Step 9: Commit bundle assembly**

```bash
git add scripts/build-worker.mjs scripts/windows-sycl-bundle.mjs scripts/windows-sycl-bundle.test.mjs
git commit -m "Bundle Windows SYCL runtime dependencies"
```

### Task 4: SYCL device preflight and backend-specific arguments

**Files:**
- Create: `src/worker/syclDevice.ts`
- Create: `src/worker/syclDevice.test.ts`
- Create: `src/worker/workerLaunch.ts`
- Create: `src/worker/workerLaunch.test.ts`
- Modify: `src/worker/workerManager.ts`
- Modify: `src/worker/workerLaunchArguments.integration.test.ts`

**Interfaces:**
- Consumes: `ResolvedWorkerBundle` and `WorkerBackend` from `workerManifest.ts`
- Produces: `parseSyclDevices(output: string): SyclDevice[]`
- Produces: `discoverSycl0(executable: string, run?: DeviceRunner): Promise<SyclDevice>`
- Produces: `resolveExecutionBackend(target: string, mode: AccelerationMode): WorkerBackend`
- Produces: `prepareWorkerLaunch(input: PrepareWorkerLaunchInput): Promise<PreparedWorkerLaunch>`
- Changes: `buildWorkerArguments(..., backend: WorkerBackend, concurrentWorkerBytes?: number): string[]`

Define the new launch types as:

```ts
export interface SyclDevice {
  id: `SYCL${number}`;
  description: string;
}

export type DeviceRunner = (
  executable: string,
  args: string[],
) => Promise<{ stdout: string; stderr: string }>;

export interface PrepareWorkerLaunchInput {
  target: string;
  mode: AccelerationMode;
  resolveBundle: (target: string, mode: AccelerationMode) => Promise<LaunchableWorkerBundle>;
  discoverSycl: (executable: string) => Promise<SyclDevice>;
}

export interface PreparedWorkerLaunch {
  bundle: LaunchableWorkerBundle;
  backend: WorkerBackend;
  syclDevice?: SyclDevice;
}

export interface LaunchableWorkerBundle extends ResolvedWorkerBundle {
  executablePath: string;
}
```

- [ ] **Step 1: Write failing SYCL discovery tests**

```ts
test('parses the first explicit SYCL device', () => {
  assert.deepEqual(parseSyclDevices(
    'Available devices:\n  SYCL0: Intel(R) Arc(TM) Graphics (15473 MiB, 15000 MiB free)\n',
  ), [{ id: 'SYCL0', description: 'Intel(R) Arc(TM) Graphics (15473 MiB, 15000 MiB free)' }]);
});

test('fails loudly when the selected executable exposes no SYCL GPU', async () => {
  await assert.rejects(
    discoverSycl0('llama-server.exe', async () => ({ stdout: 'Available devices:\n', stderr: '' })),
    /No SYCL GPU was reported.*localLlm.acceleration.*cpu/s,
  );
});

test('preserves stderr from a DLL load failure', async () => {
  await assert.rejects(
    discoverSycl0('llama-server.exe', async () => {
      throw Object.assign(new Error('exit 3221225781'), { stderr: 'sycl8.dll was not found' });
    }),
    /sycl8.dll was not found/,
  );
});
```

- [ ] **Step 2: Add failing SYCL launch-argument assertions**

```ts
test('Windows SYCL explicitly selects SYCL0 and requests GPU layers', async () => {
  const args = buildArguments(
    config({ acceleration: 'auto' }),
    'sycl',
  );
  assert.deepEqual(valueFor(args, '--device'), 'SYCL0');
  assert.deepEqual(valueFor(args, '--n-gpu-layers'), '99');
  assert.deepEqual(valueFor(args, '--split-mode'), 'none');
  assert.deepEqual(valueFor(args, '--main-gpu'), '0');
  assert.equal(args.includes('--no-op-offload'), false);
});
```

- [ ] **Step 3: Add the failing no-fallback launch-plan test**

```ts
test('a SYCL preflight failure does not resolve or invoke the CPU bundle', async () => {
  const selected: string[] = [];
  await assert.rejects(prepareWorkerLaunch({
    target: 'win32-x64',
    mode: 'auto',
    resolveBundle: async (_target, mode) => {
      selected.push(mode);
      return {
        target: 'win32-x64',
        bundleName: 'sycl',
        backend: 'sycl',
        executable: 'resources/workers/win32-x64/sycl/llama-server.exe',
        executablePath: 'C:\\extension\\resources\\workers\\win32-x64\\sycl\\llama-server.exe',
        files: [],
      };
    },
    discoverSycl: async () => { throw new Error('device discovery failed'); },
  }), /device discovery failed/);
  assert.deepEqual(selected, ['auto']);
});
```

- [ ] **Step 4: Run focused tests and confirm failure**

Run: `node --disable-warning=MODULE_TYPELESS_PACKAGE_JSON --test src/worker/syclDevice.test.ts src/worker/workerLaunch.test.ts src/worker/workerLaunchArguments.integration.test.ts`

Expected: FAIL because the SYCL modules and backend argument parameter are absent.

- [ ] **Step 5: Implement discovery and launch preparation**

`discoverSycl0` runs:

```ts
execFile(executable, ['--list-devices'], {
  cwd: path.dirname(executable),
  windowsHide: true,
  maxBuffer: 4 * 1024 * 1024,
});
```

Parse both stdout and stderr because llama.cpp may emit device information on either stream. Return only the exact `SYCL0` record. Prefix failures with `Windows SYCL device discovery failed:` and append `Set localLlm.acceleration to cpu to select the CPU worker.`

- [ ] **Step 6: Implement backend-specific argument branches**

Use three explicit branches:

```ts
switch (backend) {
  case 'metal':
    args.push('--fit', 'on', '--fit-target', String(resolveFitTargetMiB(...)));
    break;
  case 'sycl':
    args.push(
      '--fit', 'off',
      '--n-gpu-layers', '99',
      '--device', 'SYCL0',
      '--split-mode', 'none',
      '--main-gpu', '0',
    );
    break;
  case 'cpu':
    args.push('--fit', 'off', '--n-gpu-layers', '0', '--device', 'none', '--no-op-offload');
    break;
}
```

During this task, keep `WorkerManager`'s production selection equivalent to current behavior: Darwin `auto` uses Metal and every other current path uses CPU. The final manifest migration wires Windows `auto` to SYCL atomically in Task 6.

- [ ] **Step 7: Run focused and full verification**

Run: `node --disable-warning=MODULE_TYPELESS_PACKAGE_JSON --test src/worker/syclDevice.test.ts src/worker/workerLaunch.test.ts src/worker/workerLaunchArguments.integration.test.ts`

Expected: PASS.

Run: `npm test && npm run typecheck && npm run build`

Expected: PASS.

- [ ] **Step 8: Commit launch policy without enabling it yet**

```bash
git add src/worker/syclDevice.ts src/worker/syclDevice.test.ts src/worker/workerLaunch.ts src/worker/workerLaunch.test.ts src/worker/workerManager.ts src/worker/workerLaunchArguments.integration.test.ts
git commit -m "Add strict Windows SYCL launch policy"
```

### Task 5: Target package verification and smoke tooling

**Files:**
- Create: `scripts/package-workers.mjs`
- Create: `scripts/package-workers.test.mjs`
- Create: `scripts/smoke-worker.mjs`
- Modify: `package.json`

**Interfaces:**
- Consumes: manifest parsing and `verifyPlatformBundles` from `src/worker/workerManifest.ts`
- Produces: `prepareTargetWorkers(root: string, target: string): Promise<void>`
- Produces: `targetIgnoreEntries(target: string): string[]`
- Produces: CLI `npm run smoke:worker -- --backend sycl|cpu --model "$LOCAL_LLM_SMOKE_MODEL"`

- [ ] **Step 1: Write failing package-fixture tests**

```js
test('Windows packaging requires and verifies both bundles', async (context) => {
  const fixture = await packageFixture(context, 'win32-x64');
  await prepareTargetWorkers(fixture.root, 'win32-x64');
  await rm(fixture.syclDll);
  await assert.rejects(
    prepareTargetWorkers(fixture.root, 'win32-x64'),
    /sycl8.dll/,
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
```

Keep `packageFixture` local to `package-workers.test.mjs`; it must not import another test file. Its manifest body is:

```js
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
        default: workerBundle('resources/workers/darwin-arm64/llama-server'),
      },
    },
    'win32-x64': {
      modes: {
        auto: { bundle: 'sycl', backend: 'sycl' },
        cpu: { bundle: 'cpu', backend: 'cpu' },
      },
      bundles: {
        sycl: workerBundle(
          'resources/workers/win32-x64/sycl/llama-server.exe',
          ['resources/workers/win32-x64/sycl/sycl8.dll'],
        ),
        cpu: workerBundle('resources/workers/win32-x64/cpu/llama-server.exe'),
      },
    },
  },
};
```

`workerBundle(executable, additionalPaths = [])` creates every listed file below the temporary root, computes each real digest with `sha256File`, and returns `{ executable, files }`. `packageFixture` writes the JSON manifest with two-space indentation and returns the absolute `syclDll` and `windowsDirectory` paths used by the tests.

- [ ] **Step 2: Run package tests and confirm they fail**

Run: `node --test scripts/package-workers.test.mjs`

Expected: FAIL because `package-workers.mjs` does not exist.

- [ ] **Step 3: Implement target preparation without invoking VSCE**

`prepareTargetWorkers` reads `resources/workers/manifest.json`, parses version 2, and verifies every bundle declared for only the requested target. `targetIgnoreEntries` returns the exact other-platform exclusion plus the existing general exclusions.

Keep `scripts/package-vsix.mjs` on the production version-1 path until Task 6 so source-only commits do not break current Windows packaging.

- [ ] **Step 4: Implement the native smoke CLI**

The CLI:

1. resolves and verifies the requested Windows bundle;
2. runs `--list-devices` before SYCL startup;
3. allocates a loopback port;
4. writes a random API key to a temporary `0o600` file;
5. starts the selected worker with the same backend arguments as the extension;
6. waits for `/health`;
7. sends a `/v1/chat/completions` request with `max_tokens: 1` and prompt `Reply with OK.`;
8. prints the selected backend, executable, detected device, health result, and response status; and
9. terminates the worker and removes the temporary key in a `finally` block.

The command rejects a relative model path and never downloads or modifies a model.

- [ ] **Step 5: Add the package script and run verification**

Add:

```json
"smoke:worker": "node scripts/smoke-worker.mjs"
```

Run: `node --test scripts/package-workers.test.mjs`

Expected: PASS.

Run: `npm test && npm run typecheck && npm run build`

Expected: PASS.

- [ ] **Step 6: Commit package helpers and smoke tooling**

```bash
git add scripts/package-workers.mjs scripts/package-workers.test.mjs scripts/smoke-worker.mjs package.json
git commit -m "Add worker package and smoke verification"
```

### Task 6: Native Windows build and atomic production migration

**Files:**
- Modify: `resources/workers/manifest.json`
- Move: `resources/workers/win32-x64/llama-server.exe` to `resources/workers/win32-x64/cpu/llama-server.exe`
- Create: `resources/workers/win32-x64/sycl/llama-server.exe`
- Create: `resources/workers/win32-x64/sycl/*.dll`
- Create: `resources/workers/win32-x64/sycl/*.spv`
- Create: `resources/workers/win32-x64/sycl/licenses/**`
- Modify: `src/worker/workerIntegrity.ts`
- Modify: `src/worker/workerManager.ts`
- Modify: `scripts/package-vsix.mjs`
- Modify: `scripts/package-workers.test.mjs`

**Interfaces:**
- Consumes: all modules produced by Tasks 1–5
- Changes: `verifiedWorkerPath(...)` to `verifiedWorkerBundle(extensionPath: string, target: string, mode: AccelerationMode): Promise<VerifiedWorkerBundle>`
- Wires: Windows `auto` to `{ bundleName: 'sycl', backend: 'sycl' }`
- Wires: Windows `cpu` to `{ bundleName: 'cpu', backend: 'cpu' }`

`VerifiedWorkerBundle` is:

```ts
export interface VerifiedWorkerBundle extends LaunchableWorkerBundle {
  executablePath: string;
}
```

- [ ] **Step 1: Verify native Windows prerequisites from Git Bash/zsh**

Run:

```bash
node --version
cmake --version
cmd.exe /d /s /c 'if exist "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" (exit /b 0) else (exit /b 1)'
cmd.exe /d /s /c 'call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 --force >nul && where cl && where icx && where ninja'
```

Expected: Node 22 or newer; CMake available; `cl`, `icx`, and `ninja` resolve after the build script loads oneAPI; `setvars.bat` exists. If the final command fails, install Intel Deep Learning Essentials 2025.3.3 and the Level Zero 1.28.2 SDK before continuing.

- [ ] **Step 2: Build both bundles and generate the version-2 manifest**

Run:

```bash
npm ci
npm run build:worker -- --target win32-x64 --backend all
```

Expected: both isolated executables exist, the SYCL companion and license files are copied, and `resources/workers/manifest.json` contains version 2 with complete CPU and SYCL hashes.

- [ ] **Step 3: Write failing production-integration tests**

Add assertions that:

```ts
const auto = await verifiedWorkerBundle(root, 'win32-x64', 'auto');
assert.equal(auto.backend, 'sycl');
assert.match(auto.executablePath, /win32-x64[\\/]sycl[\\/]llama-server\.exe$/);

const cpu = await verifiedWorkerBundle(root, 'win32-x64', 'cpu');
assert.equal(cpu.backend, 'cpu');
assert.match(cpu.executablePath, /win32-x64[\\/]cpu[\\/]llama-server\.exe$/);
```

Add a WorkerManager launch seam test whose SYCL discovery throws and whose spawn spy remains at zero calls. Add a package test that tampers with one SYCL DLL and observes packaging rejection before VSCE invocation.

- [ ] **Step 4: Run focused tests and confirm the production code is not wired**

Run: `node --disable-warning=MODULE_TYPELESS_PACKAGE_JSON --test --test-name-pattern='worker bundle|SYCL|packaging' src/worker/workerManifest.test.ts src/worker/workerLaunch.test.ts src/worker/syclDevice.test.ts src/worker/workerLaunchArguments.integration.test.ts scripts/package-workers.test.mjs`

Expected: FAIL because `workerIntegrity.ts`, `workerManager.ts`, and `package-vsix.mjs` still use version 1/current Windows CPU behavior.

- [ ] **Step 5: Wire runtime selection atomically**

`verifiedWorkerBundle` reads and parses version 2, resolves exactly one mode, verifies every file in that selected bundle, and returns its absolute executable path and backend. In `WorkerManager.start`:

```ts
const config = readConfig(this.context);
const launch = await prepareWorkerLaunch({
  target: `${process.platform}-${process.arch}`,
  mode: config.acceleration,
  resolveBundle: (target, mode) => verifiedWorkerBundle(
    this.context.extensionUri.fsPath,
    target,
    mode,
  ),
  discoverSycl: discoverSycl0,
});
```

Pass `launch.backend` into `buildWorkerArguments`. Store the backend with the active child so an unexpected-exit restart re-resolves the same configuration mode; the configuration change listener already stops the worker before a mode change takes effect. Do not add a catch branch that resolves `cpu` after a SYCL error.

- [ ] **Step 6: Wire target package verification atomically**

Replace the one-file hash block in `scripts/package-vsix.mjs` with:

```js
await prepareTargetWorkers(root, target);
```

Use `targetIgnoreEntries(target)` when writing the ignore file. Windows verification must traverse both `sycl` and `cpu`; Darwin verification must traverse only the Darwin bundle.

- [ ] **Step 7: Run all automated gates on Windows**

Run:

```bash
npm test
npm run typecheck
npm run build
npm run package -- win32-x64
```

Expected: all tests PASS and the target VSIX is created under `dist/vsix/win32-x64/`.

- [ ] **Step 8: Confirm VSIX contents**

Run:

```bash
unzip -l dist/vsix/win32-x64/local-llm-engine-0.3.4-win32-x64.vsix | rg 'resources/workers/(win32-x64/(sycl|cpu)|darwin-arm64)'
```

Expected: every manifest-declared Windows file is listed and no Darwin worker is listed.

- [ ] **Step 9: Commit the production migration and generated artifacts**

```bash
git add resources/workers src/worker/workerIntegrity.ts src/worker/workerManager.ts scripts/package-vsix.mjs scripts/package-workers.test.mjs
git commit -m "Enable self-contained Windows SYCL worker"
```

### Task 7: Documentation, notices, and Arc hardware release gate

**Files:**
- Modify: `package.json`
- Modify: `README.md`
- Modify: `THIRD_PARTY_NOTICES.md`

**Interfaces:**
- Consumes: the actual version-2 manifest and native Windows VSIX from Task 6
- Produces: documented build/install/smoke procedure and complete third-party attribution

- [ ] **Step 1: Update user-facing acceleration text**

Change `localLlm.acceleration`'s description to:

```text
Use Metal on Apple Silicon or Intel SYCL on x64 Windows automatically. Select cpu to force the bundled CPU worker. SYCL failures are reported and never fall back automatically.
```

Update README platform support to name Intel Arc/built-in Arc on Windows 11, state that the VSIX includes the oneAPI runtime DLLs, and retain the compatible-driver requirement.

- [ ] **Step 2: Document native Windows build commands for zsh/Git Bash**

Document:

```bash
npm run build:worker -- --target win32-x64 --backend all
npm run package -- win32-x64
```

Explain that the build machine needs Visual Studio 2022, CMake/Ninja, Intel oneAPI 2025.3.3, and the Level Zero 1.28.2 SDK; the installed VSIX does not.

- [ ] **Step 3: Update third-party notices from the actual bundle**

List the shipped Intel compiler runtime, Unified Runtime, oneMKL, oneDNN, TBB, TCM, UMF, and Microsoft VC runtime components that remain in the manifest after dependency pruning. Point each group to its copied controlling license path under `resources/workers/win32-x64/sycl/licenses/`. Update the worker digest list to refer readers to the per-file version-2 manifest instead of duplicating dozens of hashes in Markdown.

- [ ] **Step 4: Install into the intended VS Code profile**

Set `LOCAL_LLM_VSCODE_PROFILE` to the exact existing profile name and run:

```bash
code --profile "$LOCAL_LLM_VSCODE_PROFILE" --install-extension dist/vsix/win32-x64/local-llm-engine-0.3.4-win32-x64.vsix --force
```

Expected: VS Code reports a successful install into that profile.

- [ ] **Step 5: Run the SYCL and explicit CPU smoke utility**

Set `LOCAL_LLM_SMOKE_MODEL` to the absolute path of an installed GGUF that fits the reported SYCL memory, then run:

```bash
npm run smoke:worker -- --backend sycl --model "$LOCAL_LLM_SMOKE_MODEL"
npm run smoke:worker -- --backend cpu --model "$LOCAL_LLM_SMOKE_MODEL"
```

Expected for SYCL: `SYCL0` is reported, the log assigns model layers to `SYCL0`, `/health` succeeds, and one inference response completes.

Expected for CPU: the CPU bundle path is reported, no SYCL preflight runs, `/health` succeeds, and one inference response completes.

- [ ] **Step 6: Exercise the loud-failure boundary without changing bundle files**

Close VS Code, then launch it from Git Bash with an impossible Level Zero selector:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:999 code --profile "$LOCAL_LLM_VSCODE_PROFILE"
```

Start the local model from that window. Expected: startup fails with the SYCL discovery diagnostic and the instruction to select `cpu` manually.

Confirm no CPU bundle process was launched:

```bash
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" | Select-Object ExecutablePath,CommandLine | Format-List"
```

Expected: no process path under `resources\\workers\\win32-x64\\cpu\\` appears as a result of the failed SYCL start.

- [ ] **Step 7: Run final cross-platform gates**

On Windows:

```bash
npm test
npm run typecheck
npm run build
npm run package -- win32-x64
git diff --check
```

On Apple Silicon macOS:

```bash
npm test
npm run typecheck
npm run build
npm run package -- darwin-arm64
git diff --check
```

Expected: every command succeeds. Report Windows SYCL inference, Windows CPU inference, Darwin packaging, and automated tests as separate gates.

- [ ] **Step 8: Commit documentation and verified release metadata**

```bash
git add package.json README.md THIRD_PARTY_NOTICES.md resources/workers/manifest.json
git commit -m "Document Windows SYCL worker release"
```

---

## Reference Evidence

- The pinned llama.cpp SYCL documentation describes Windows source builds with `GGML_SYCL=ON`, `cl`/`icx`, shared libraries, `--list-devices`, and `--device SYCL0`: `https://github.com/ggml-org/llama.cpp/blob/60eeeb6082c1126bb8bc72902c83123cd056811b/docs/backend/SYCL.md`.
- The pinned upstream Windows release workflow records the oneAPI 2025.3.3 runtime companion set used above: `https://github.com/ggml-org/llama.cpp/blob/60eeeb6082c1126bb8bc72902c83123cd056811b/.github/workflows/release.yml`.
- Intel documents private deployment of required shared runtime libraries and requires reviewing the license files shipped with the installed components: `https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2023-0/redistribute-libraries-when-deploying-apps.html`.
