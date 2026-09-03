# Official Windows SYCL Archive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the custom Windows llama.cpp/oneAPI build and dependency-reconstruction pipeline with deterministic packaging of the pinned official Windows SYCL release archive.

**Architecture:** A single archive preparer owns download, checksum, safe extraction, transactional publication, and Windows manifest generation. Packaging calls that preparer automatically for `win32-x64`; runtime selection maps both acceleration modes to the same extracted executable and distinguishes them only by backend arguments.

**Tech Stack:** Node.js 24 ESM, `yauzl` 3.4.0, TypeScript 7, Node test runner, VSCE.

**Spec:** `docs/superpowers/specs/2026-09-03-official-windows-sycl-archive-design.md`

## Global Constraints

- Pin llama.cpp commit `60eeeb6082c1126bb8bc72902c83123cd056811b` and release `b10472`.
- Pin `llama-b10472-bin-win-sycl-x64.zip` at SHA-256 `0c4c50f1e9805933e043d4970f0c2050e4fb5343b8ac0244a49efaa474705830` and size `119700367`.
- Extract the complete official payload unchanged; do not prune files in this correctness pass.
- Keep the Windows archive and extracted payload out of Git while including the extracted payload in the Windows VSIX.
- Windows packaging must not execute worker or SYCL tools and must not require GPU hardware or developer toolchains.
- Windows `auto` and `cpu` use one `sycl` bundle and one `llama-server.exe`; no SYCL failure may fall back to CPU.
- Preserve the Darwin worker build and package behavior.
- Use failing tests before each production behavior change.

---

### Task 1: Archive preparation boundary

**Files:**
- Create: `scripts/windows-worker-archive.mjs`
- Create: `scripts/windows-worker-archive.test.mjs`
- Modify: `package.json`
- Modify: `package-lock.json`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `WINDOWS_WORKER_RELEASE` with `commit`, `build`, `assetName`, `url`, `size`, and `sha256`.
- Produces: `prepareWindowsWorkerArchive(root: string, dependencies?: object): Promise<WorkerManifestV2>`.
- Produces: `createWindowsWorkerManifest(existing: unknown, bundle: WorkerBundle): WorkerManifestV2`.

- [ ] **Step 1: Add direct ZIP-reader dependency**

Run:

```shell
npm install --save-dev --save-exact yauzl@3.4.0
```

Expected: `package.json` and `package-lock.json` declare `yauzl` directly.

- [ ] **Step 2: Write checksum/cache failing tests**

Add tests that inject a fetch response and assert that preparation downloads to
`build/worker-downloads/llama-b10472-bin-win-sycl-x64.zip`, rejects a byte count
other than `119700367`, rejects a SHA other than the pinned value, and never
publishes the destination on failure. Test exported lower-level helpers with a
fixture-specific expected size and SHA so the test does not allocate 120 MB.

Run:

```shell
node --disable-warning=MODULE_TYPELESS_PACKAGE_JSON --test scripts/windows-worker-archive.test.mjs
```

Expected: FAIL because `scripts/windows-worker-archive.mjs` does not exist.

- [ ] **Step 3: Implement verified atomic cache download**

Implement `ensureVerifiedArchive` so an existing cache is always size/hash
verified, a missing cache is streamed to a unique `.partial` file, and only a
verified partial is renamed to the cache path. A corrupt cache is rejected with
the expected and actual values; it is never extracted.

- [ ] **Step 4: Write safe-extraction failing tests**

Add fixture ZIP cases for a flat regular payload, `../escape.dll`, an absolute
entry, a symbolic-link entry, duplicate normalized paths, and an archive
without `llama-server.exe`. Assert that only the flat payload succeeds and no
entry is written outside staging.

Run the focused test and confirm the new cases fail for the missing extractor.

- [ ] **Step 5: Implement streaming safe extraction**

Use `yauzl.open(..., { lazyEntries: true })`, create parents explicitly, stream
each regular file with exclusive creation, reject unsafe or duplicate targets,
and require `llama-server.exe`. Return a bundle whose file list is sorted by
POSIX path and whose SHA-256 values are calculated from extracted bytes.

- [ ] **Step 6: Write manifest/publication failing tests**

Assert that both Windows modes reference bundle `sycl`, backend values remain
`sycl` and `cpu`, a legacy Darwin manifest migrates to v2, and an injected
publication failure restores the prior `win32-x64` directory and manifest bytes.

- [ ] **Step 7: Implement manifest creation and transactional publication**

Stage an entire replacement `win32-x64/sycl` tree, stage the new manifest,
rename the previous Windows tree and manifest to unique backups, copy the staged
Windows tree into place, publish the manifest, and restore both backups on any
error. Delete backups only after success.

- [ ] **Step 8: Ignore generated Windows payload and run focused tests**

Add:

```gitignore
resources/workers/win32-x64/sycl/
```

Run the archive tests. Expected: all archive tests pass.

### Task 2: Package and build orchestration

**Files:**
- Modify: `scripts/package-workers.mjs`
- Modify: `scripts/package-workers.test.mjs`
- Modify: `scripts/build-worker.mjs`
- Delete: `scripts/worker-build-options.mjs`
- Delete: `scripts/worker-build-options.test.mjs`

**Interfaces:**
- Consumes: `prepareWindowsWorkerArchive(root)` from Task 1.
- Produces: `prepareTargetWorkers(root, target, dependencies?)` that stages the official archive before Windows validation.

- [ ] **Step 1: Change packaging tests first**

Update the Windows fixture to one `sycl` bundle. Inject a `prepareWindowsArchive`
function and assert it is called once before validation for `win32-x64`, is not
called for Darwin, and makes a missing/mismatched extracted file fail existing
hash validation.

Run the focused package tests. Expected: FAIL because production packaging does
not call the injected preparer and still expects two bundles.

- [ ] **Step 2: Wire Windows preparation into packaging**

Call `prepareWindowsWorkerArchive(root)` before reading the manifest for
`win32-x64`. Retain `verifyPlatformBundles` afterward and remove the obsolete
`--backend all` recovery message.

- [ ] **Step 3: Simplify the worker build entrypoint**

For target `win32-x64`, call `prepareWindowsWorkerArchive(root)` and exit. For
target `darwin-arm64`, retain the existing pinned source clone, CMake options,
build, and copy path. Accept only `--target <target>`; remove backend,
Visual Studio, oneAPI, Ninja, `dumpbin`, and Windows compiler logic.

- [ ] **Step 4: Delete obsolete option parsing and verify**

Delete `worker-build-options` and its tests, run the archive and package tests,
then run `npm run build` to prove imports resolve.

### Task 3: Single-bundle runtime discovery

**Files:**
- Modify: `src/worker/syclDevice.ts`
- Modify: `src/worker/syclDevice.test.ts`
- Modify: `src/worker/workerLaunch.ts`
- Modify: `src/worker/workerLaunch.test.ts`
- Modify: `src/worker/workerManager.ts`
- Modify: `scripts/smoke-worker.mjs`
- Modify: `scripts/smoke-worker.test.mjs`
- Modify: `src/worker/workerManifest.test.ts`
- Modify: `src/worker/workerLaunchArguments.integration.test.ts`

**Interfaces:**
- Produces: `cleanSyclEnvironment(executable, baseEnvironment): NodeJS.ProcessEnv`.
- Changes: `discoverSycl0(...)` returns `{ id, description, environment }` with no adapter metadata.
- Changes: `PreparedWorkerLaunch.environment` carries the successful clean environment to the worker spawn.

- [ ] **Step 1: Write one-attempt discovery tests**

Replace adapter fallback expectations with assertions that discovery calls
`--list-devices` exactly once, removes all four selector/adapter variables
case-insensitively, and sets `PATH` to the executable directory plus
`System32` and the Windows root. Assert stdout and stderr are both parsed and a
missing `SYCL0` retains captured loader diagnostics.

Run `src/worker/syclDevice.test.ts`. Expected: FAIL because discovery still
forces and retries Level Zero/OpenCL.

- [ ] **Step 2: Implement clean default discovery**

Remove adapter candidates and run one process using `cleanSyclEnvironment`.
Return the found device and exact environment. Keep the direct CPU-selection
message in failure output and preserve stderr.

- [ ] **Step 3: Update launch tests and types first**

Expect `prepareWorkerLaunch` to expose `environment` rather than `syclRuntime`.
Keep the test proving a failed SYCL preflight never resolves CPU.

- [ ] **Step 4: Carry the clean environment into execution**

Update `workerLaunch.ts`, `workerManager.ts`, and `smoke-worker.mjs` to pass
`launch.environment`/`device.environment` into the SYCL spawn. Log only the
detected `SYCL0` description, not an inferred adapter.

- [ ] **Step 5: Update single-bundle fixtures**

Make every Windows manifest fixture use:

```js
modes: {
  auto: { bundle: 'sycl', backend: 'sycl' },
  cpu: { bundle: 'sycl', backend: 'cpu' },
}
```

Assert the smoke resolver returns the same executable for both modes while CPU
skips discovery and retains CPU launch arguments.

- [ ] **Step 6: Run focused runtime tests**

Run the SYCL device, worker launch, manifest, launch-argument, and smoke tests.
Expected: all focused runtime tests pass.

### Task 4: Remove Windows reconstruction implementation

**Files:**
- Delete: `scripts/windows-oneapi.mjs`
- Delete: `scripts/windows-oneapi.test.mjs`
- Delete: `scripts/windows-vs-tools.mjs`
- Delete: `scripts/windows-vs-tools.test.mjs`
- Delete: `scripts/windows-sycl-bundle.mjs`
- Delete: `scripts/windows-sycl-bundle.test.mjs`
- Delete: `scripts/windows-sycl-runtime.mjs`
- Delete: `scripts/windows-sycl-runtime.test.mjs`
- Modify: `src/worker/workerManifest.ts`

**Interfaces:**
- Removes: all local oneAPI/Visual Studio discovery, PE closure, runtime reconstruction, adapter forcing, and custom multi-bundle publication APIs.

- [ ] **Step 1: Prove no retained imports require deleted modules**

Run `rg` for the five deprecated production module names after Tasks 1–3.
Expected: only the modules themselves, their tests, and obsolete docs remain.

- [ ] **Step 2: Delete deprecated modules and tests**

Delete all eight oneAPI/Visual Studio/SYCL reconstruction files. Remove
`replaceWorkerBundle` from `workerManifest.ts` after confirming the archive
preparer constructs and validates the complete manifest without it.

- [ ] **Step 3: Run the complete test and type gates**

Run:

```shell
npm test
npm run typecheck
npm run build
```

Expected: zero failures and no deleted import resolution errors.

### Task 5: Native verification command and documentation

**Files:**
- Create: `scripts/verify-windows-worker.mjs`
- Create: `scripts/verify-windows-worker.test.mjs`
- Modify: `package.json`
- Modify: `README.md`
- Modify: `docs/WINDOWS_SYCL_HANDOFF.md`
- Modify: `docs/superpowers/specs/2026-08-24-windows-sycl-worker-design.md`
- Delete: `docs/superpowers/specs/2026-09-02-version-aware-windows-sycl-packaging-design.md`
- Modify: `THIRD_PARTY_NOTICES.md`

**Interfaces:**
- Produces: `npm run verify:windows-worker` as a native Windows clean-environment probe.

- [ ] **Step 1: Write probe reporting tests**

Inject process runners and assert three commands execute in order, every command
uses its executable directory as `cwd` and the clean environment, a nonzero
exit reports decimal and unsigned hexadecimal status plus stdout/stderr, and a
zero `--list-devices` result without `SYCL0` fails.

Run the focused test. Expected: FAIL because the verifier does not exist.

- [ ] **Step 2: Implement native probe**

Resolve and hash-check the manifest `auto` bundle, require x64 Windows in the
real entrypoint, then run `sycl-ls`, version, and device discovery without
forcing an adapter. Do not load a model in this command.

- [ ] **Step 3: Replace obsolete documentation**

Document the one-command package flow, cache/checksum behavior, absence of
developer-tool prerequisites, single-bundle modes, three-location diagnostic
comparison, native probe, smoke commands, installed-VSIX checks, and the
remaining Intel driver prerequisite. Delete the dependency-reconstruction spec
instead of leaving it as an apparent alternative.

- [ ] **Step 4: Update third-party notices**

Identify the official archive URL and SHA, state that its complete payload is
included unchanged, retain the llama.cpp MIT text, and identify the bundled
Intel runtime family and applicable Intel licensing source. Do not claim that
the upstream ZIP contains license files it does not contain.

- [ ] **Step 5: Run documentation and dead-code scans**

Search retained files for `setvars.bat`, `dumpbin`, `VsDevCmd`, `--backend all`,
separate `win32-x64/cpu`, and Level-Zero/OpenCL retry language. Matches may
remain only in historical Git history, not active source or current guidance.

### Task 6: Materialize, package, inspect, and publish

**Files:**
- Modify: `resources/workers/manifest.json`
- Delete: `resources/workers/win32-x64/llama-server.exe`
- Generated/ignored: `resources/workers/win32-x64/sycl/**`
- Generated: `dist/vsix/win32-x64/local-llm-engine-0.3.4-win32-x64.vsix`

**Interfaces:**
- Consumes: all previous tasks.
- Produces: deterministic checked-in manifest and a locally inspected Windows VSIX.

- [ ] **Step 1: Run archive preparation against the pinned real asset**

Run:

```shell
npm run build:worker -- --target win32-x64
```

Expected: the pinned cached ZIP is verified, its complete flat payload is
extracted, and the manifest records one sorted `sycl` bundle for both modes.

- [ ] **Step 2: Remove the legacy tracked executable**

Delete `resources/workers/win32-x64/llama-server.exe`. Confirm Git ignores the
generated `sycl` payload while tracking the deterministic manifest change.

- [ ] **Step 3: Run full verification fresh**

Run:

```shell
npm test
npm run typecheck
npm run build
npm run package -- win32-x64
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 4: Inspect VSIX contents and manifest parity**

List the VSIX and assert every manifest-declared Windows file appears exactly
once, no Darwin worker appears, no nested release ZIP appears, and no
`resources/workers/win32-x64/cpu/` path appears.

- [ ] **Step 5: Review scope and commit**

Inspect `git status`, `git diff --stat`, and the full diff. Confirm the unrelated
main-checkout plan remains untouched. Commit the surgical replacement on
`codex/windows-sycl-worker`.

- [ ] **Step 6: Push the branch exactly as requested**

Run:

```shell
git push origin codex/windows-sycl-worker
git ls-remote origin refs/heads/codex/windows-sycl-worker
```

Expected: the remote ref equals the committed local HEAD. Report native Windows
Arc probe/model smoke/installed-VSIX gates as still pending until they are run
on the Windows host; do not describe macOS packaging as Windows runtime proof.
