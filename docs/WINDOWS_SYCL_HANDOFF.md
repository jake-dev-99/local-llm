# Windows SYCL worker handoff

## Use this branch

Do not continue this work from `main`. The implementation is on:

```text
codex/windows-sycl-worker
```

The completed and reviewed Tasks 1 through 5 end at commit `3849eeb`. This handoff checkpoint also contains a regression-tested fix for the legacy-to-version-2 manifest transition that the native `--backend all` build requires. The branch contains:

- the version-2 multi-bundle worker manifest contract;
- isolated Windows CPU and SYCL build orchestration;
- self-contained SYCL dependency, license, and hash assembly;
- strict `SYCL0` discovery with no automatic CPU fallback;
- target package verification; and
- the explicit CPU/SYCL smoke command.

The handoff checkpoint is verified on macOS with the full source test suite, typecheck, and build. That is not a Windows artifact or hardware result.

Everything needed to continue is committed to this branch. Do not depend on `.superpowers/` or another local-only file. The full design and executable plan are:

- [`docs/superpowers/specs/2026-08-24-windows-sycl-worker-design.md`](superpowers/specs/2026-08-24-windows-sycl-worker-design.md)
- [`docs/superpowers/plans/2026-08-24-windows-sycl-worker.md`](superpowers/plans/2026-08-24-windows-sycl-worker.md)

Continue at **Task 6: Native Windows build and atomic production migration**, then complete Task 7 and the final whole-branch review.

## Check out the correct branch on Windows

From Git Bash or zsh in the Windows clone:

```bash
git fetch origin
git switch --track origin/codex/windows-sycl-worker
git rev-parse --short HEAD
```

If the local branch already exists, use:

```bash
git switch codex/windows-sycl-worker
git pull --ff-only origin codex/windows-sycl-worker
```

Confirm this document and the implementation helpers are present:

```bash
test -f docs/WINDOWS_SYCL_HANDOFF.md
test -f scripts/windows-oneapi.mjs
test -f scripts/windows-sycl-bundle.mjs
test -f scripts/package-workers.mjs
test -f scripts/smoke-worker.mjs
test -f src/worker/workerManifest.ts
test -f src/worker/workerLaunch.ts
test -f src/worker/syclDevice.ts
```

## Native Windows prerequisites

The build machine needs:

- x64 Windows 11;
- Node.js 22 or newer;
- Visual Studio 2022 C++ build tools;
- CMake and Ninja;
- Intel oneAPI 2025.3.3, including Deep Learning Essentials; and
- the Level Zero 1.28.2 SDK.

The installed VSIX must not require those developer tools. Required runtime DLLs, SPIR-V files, and controlling licenses are copied into the SYCL worker bundle.

Run these preflight checks from Git Bash or zsh:

```bash
node --version
cmake --version
cmd.exe /d /s /c 'if exist "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" (exit /b 0) else (exit /b 1)'
cmd.exe /d /s /c 'call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 --force >nul && where cl && where icx && where ninja'
```

Do not proceed until `setvars.bat`, `cl`, `icx`, and `ninja` all resolve.

## Build the real CPU and SYCL bundles

```bash
npm ci
npm run build:worker -- --target win32-x64 --backend all
```

This must build locally from the pinned llama.cpp source. Do not substitute an upstream release archive or fabricate the SYCL files.

The checked-in manifest is intentionally still version 1 before this command. The build publishes the isolated CPU entry first, then migrates to version 2 only when the SYCL bundle is assembled. If the command fails, do not hand-edit hashes or claim that migration completed.

Before changing production runtime selection, confirm the build produced:

```text
resources/workers/win32-x64/cpu/llama-server.exe
resources/workers/win32-x64/sycl/llama-server.exe
resources/workers/win32-x64/sycl/*.dll
resources/workers/win32-x64/sycl/*.spv
resources/workers/win32-x64/sycl/licenses/**
```

Also confirm `resources/workers/manifest.json` is version 2 and hashes every file in both Windows bundles.

## Complete the atomic production migration

Follow Task 6 Steps 3 through 9 in the committed implementation plan. The migration is one unit:

1. add failing production integration tests;
2. replace `verifiedWorkerPath` with bundle-aware verification;
3. make Windows `auto` select the SYCL bundle;
4. make Windows `cpu` select only the CPU bundle;
5. run `SYCL0` discovery before spawning the SYCL worker;
6. never catch a SYCL failure and retry with CPU;
7. make packaging verify both Windows bundles; and
8. commit the generated native artifacts and production wiring together.

The CPU worker is an explicit user-selected mode only. A missing DLL, missing `SYCL0`, or failed SYCL startup must fail loudly before any CPU worker is launched.

Run the Windows automated gates:

```bash
npm test
npm run typecheck
npm run build
npm run package -- win32-x64
git diff --check
```

Inspect the VSIX:

```bash
unzip -l dist/vsix/win32-x64/local-llm-engine-0.3.4-win32-x64.vsix | rg 'resources/workers/(win32-x64/(sycl|cpu)|darwin-arm64)'
```

Every manifest-declared Windows file must be present. No Darwin worker may be present.

## Hardware and failure-boundary gates

Task 7 is not optional. Use an absolute GGUF path that fits the Arc GPU and run both explicit smoke modes:

```bash
export LOCAL_LLM_SMOKE_MODEL='C:/absolute/path/to/model.gguf'
npm run smoke:worker -- --backend sycl --model "$LOCAL_LLM_SMOKE_MODEL"
npm run smoke:worker -- --backend cpu --model "$LOCAL_LLM_SMOKE_MODEL"
```

The SYCL run must report `SYCL0`, load model layers onto it, pass `/health`, and complete the one-token chat request. The CPU run must use the isolated CPU executable and perform no SYCL discovery.

Then follow Task 7 Step 6 to force an invalid Level Zero selector. The extension must report the SYCL failure and must not launch the CPU executable.

Install the VSIX into the exact intended existing VS Code profile, not the default profile, using Task 7 Step 4.

## Finish and publish

Only after Task 6, Task 7, the Windows hardware gates, the Darwin packaging gate, and the final whole-branch review are clean:

1. commit all remaining release changes on `codex/windows-sycl-worker`;
2. push that branch;
3. merge it into `main` without dropping any commits; and
4. push the updated `main` to `origin`.

The user has authorized that final merge and push after the release gates pass. Do not merge the incomplete production migration.
