# Windows SYCL worker handoff

## Use this branch

Do not continue this work from `main`. The implementation is on:

```text
codex/windows-sycl-worker
```

The completed and reviewed Tasks 1 through 5 end at commit `3849eeb`. The branch now also contains the source-side Task 6 production wiring and generation diagnostics. The native Windows artifacts still have to be built on Windows before the migration is complete. The branch contains:

- the version-2 multi-bundle worker manifest contract;
- isolated Windows CPU and SYCL build orchestration;
- self-contained SYCL dependency, license, and hash assembly;
- strict `SYCL0` discovery with no automatic CPU fallback;
- target package verification;
- visible request failures, 30-second stream-stall warnings, and 60-second long-generation warnings; and
- the explicit CPU/SYCL smoke command.

The handoff checkpoint is verified on macOS with the full source test suite, typecheck, and build. That is not a Windows artifact or hardware result.

Everything needed to continue is committed to this branch. Do not depend on `.superpowers/` or another local-only file. The full design and executable plan are:

- [`docs/superpowers/specs/2026-08-24-windows-sycl-worker-design.md`](superpowers/specs/2026-08-24-windows-sycl-worker-design.md)
- [`docs/superpowers/plans/2026-08-24-windows-sycl-worker.md`](superpowers/plans/2026-08-24-windows-sycl-worker.md)

Continue at the native Windows build below. Do not reimplement the Task 6 source wiring; it is already on this branch. The build must generate the real version-2 manifest and native bundles before packaging, then the Task 7 hardware gates and final whole-branch review remain.

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
- Visual Studio 2022 C++ build tools, including the **C++ CMake tools for Windows** component;
- Intel oneAPI, including Deep Learning Essentials; and
- a current Intel Arc Pro graphics driver providing the system Level Zero runtime.

The Visual Studio CMake component supplies both `cmake.exe` and `ninja.exe`. They do not need to
be on `PATH`: the worker build locates the Visual Studio installation with `vswhere.exe`, verifies
both executable paths, and passes the absolute paths directly. It also locates `VsDevCmd.bat`, uses
it to initialize the x64 MSVC and Windows SDK environment, then initializes oneAPI in the same
`cmd.exe` session. The build checks `VSCMD_VER`, `LIB`, and `INCLUDE`, then confirms that
`kernel32.lib` actually exists in the resolved library directories before cloning or compiling
llama.cpp. A missing Windows SDK environment therefore fails before CMake instead of at the linker.

The installed VSIX must not require those developer tools. Required redistributable runtime DLLs,
SPIR-V files, and controlling licenses are copied into the SYCL worker bundle. The Intel graphics
driver remains a host prerequisite because it supplies the Level Zero loader and GPU driver; a
standalone Level Zero SDK path is not used by either the build or the installed extension.

Run these preflight checks from Git Bash or zsh:

```bash
node --version
"/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe" -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.CMake.Project -find 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
"/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe" -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.CMake.Project -find 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
"/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe" -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.CMake.Project -find 'Common7\Tools\VsDevCmd.bat'
cmd.exe /d /s /c 'if exist "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" (exit /b 0) else (exit /b 1)'
```

Do not proceed until all three `vswhere.exe` searches print a path and the `setvars.bat` existence
check succeeds. If Visual Studio is installed somewhere else, the build still discovers its
installation path automatically; only the optional manual `vswhere.exe` checks above use the default
installer location. The build command itself performs the authoritative developer-environment check
and prints the exact CMake, Ninja, and `VsDevCmd.bat` paths it uses.

## Build the real CPU and SYCL bundles

```bash
npm ci
npm run build:worker -- --target win32-x64 --backend all
```

The SYCL packager uses the active oneAPI environment that built the worker. It
does not require a version-specific runtime filename such as `sycl8.dll`.
Build output lists the Intel compiler banner, active oneAPI runtime directories,
resolved non-system dependencies, semantic Level Zero resources, and excluded
system dependencies.

Before publishing the bundle, the build runs the staged
`llama-server.exe --list-devices` with oneAPI development paths and variables
removed. The build fails unless that isolated process reports `SYCL0`. A
failure leaves the prior CPU bundle, SYCL bundle, and worker manifest
unchanged; do not copy a missing DLL manually or switch to the CPU bundle
implicitly.

This must build locally from the pinned llama.cpp source. Do not substitute an upstream release archive or fabricate the SYCL files.

The checked-in manifest is intentionally still version 1 before this command.
The build completes CPU and SYCL staging plus the clean `SYCL0` gate first,
then publishes both directories and the version-2 manifest as one recoverable
transaction. If any publication step fails, it restores the prior directories
and manifest; do not hand-edit hashes or claim that migration completed.

Before changing production runtime selection, confirm the build produced:

```text
resources/workers/win32-x64/cpu/llama-server.exe
resources/workers/win32-x64/sycl/llama-server.exe
resources/workers/win32-x64/sycl/*.dll
resources/workers/win32-x64/sycl/*.spv
resources/workers/win32-x64/sycl/licenses/**
```

Also confirm `resources/workers/manifest.json` is version 2 and hashes every file in both Windows bundles.

## Complete the native half of the production migration

The source-side production wiring is already committed. It:

1. verifies the selected bundle and every declared file;
2. makes Windows `auto` select the SYCL bundle;
3. makes Windows `cpu` select only the CPU bundle;
4. runs `SYCL0` discovery before spawning the SYCL worker;
5. never catches a SYCL failure and retries with CPU; and
6. makes packaging verify both Windows bundles.

The checked-in version-1 manifest is intentionally incompatible with that production wiring. Run the `--backend all` build first; it atomically creates both native bundles and migrates the manifest to version 2. Do not run the package command against the pre-build checkout.

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
