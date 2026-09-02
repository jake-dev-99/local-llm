# oneAPI 2026+ Windows SYCL Packaging Design

Date: 2026-09-02
Status: Approved

## Summary

Package the Windows SYCL worker exclusively from a oneAPI 2026-or-newer
environment, using runtime discovery based on the worker that was actually
built and the active oneAPI environment. Preserve loud failures, per-file
hashes, license collection, and atomic publication.

The final gate runs the staged worker with the oneAPI development environment
removed. Packaging succeeds only when the staged worker loads independently
and discovers an Intel SYCL GPU. This proves that the bundle, rather than the
build machine's oneAPI installation, supplies its non-system runtime files.

This design extends the approved self-contained Windows worker design. It does
not change backend selection: Windows `auto` still selects SYCL, `cpu` remains
an explicit choice, and SYCL failures never trigger CPU fallback.

## Problem

The supported toolchain begins at oneAPI 2026. The build must reject an older
compiler before CMake, and the packager must derive versioned runtime DLL names
from the binaries produced by that supported compiler instead of maintaining a
filename catalog.

Removing validation is not acceptable. The installed VSIX must not depend on
oneAPI being present on the extension user's machine, and Unified Runtime
adapters are loaded dynamically rather than appearing in the executable's
static PE import table.

## Goals

- Package the runtime required by the SYCL worker built with the active oneAPI
  installation without assuming a versioned runtime filename.
- Continue resolving the complete non-system PE dependency closure.
- Include the dynamically loaded Unified Runtime loader and Level Zero adapter
  needed for Intel Arc execution.
- Prove that the staged worker does not load files from the installed oneAPI
  environment.
- Fail before replacing a previously valid bundle when dependency discovery,
  licensing, device discovery, or runtime loading is incomplete.
- Print enough diagnostic information to identify the failed phase, detected
  toolchain, searched directories, missing semantic role, or unresolved DLL.
- Keep the existing CPU bundle and no-fallback runtime behavior unchanged.

## Non-goals

- Supporting oneAPI releases older than 2026.
- Maintaining a catalog of exact DLL filenames for every oneAPI release.
- Copying entire oneAPI component directories into the VSIX.
- Bundling Intel graphics drivers or Windows system DLLs.
- Adding CUDA, HIP, or non-Intel Unified Runtime adapters.
- Silently publishing a bundle when the native Windows hardware gate cannot
  run.
- Changing llama.cpp, its pinned source commit, model handling, or inference
  behavior.

## Selected approach

The bundler will use three complementary sources of truth:

1. **Built artifacts.** `llama-server.exe`, `ggml-cpu.dll`, and
   `ggml-sycl.dll` are mandatory roots.
2. **Resolved PE dependencies.** `dumpbin /dependents` recursively discovers
   the actual versioned DLL names imported by those roots and by every copied
   non-system DLL.
3. **Semantic runtime DLLs.** The active oneAPI compiler directories are
   searched for the Unified Runtime loader. Level Zero adapter variants are
   selected beside that loader because they may not occur in PE imports.

The semantic dynamic resource rules express runtime roles rather than release
filenames:

- exactly one active Unified Runtime loader named `ur_loader.dll`;
- one or more Level Zero adapters matching `ur_adapter_level_zero*.dll`;
- the Windows Unified Runtime proxy loader when it is present in the active
  compiler runtime directory.

The SYCL runtime DLL itself and math/runtime libraries are obtained through PE
dependency closure. They are not required by a hard-coded basename. Optional
OpenCL or non-Intel adapters are not semantic roots; they are included only if
the selected roots actually depend on them.

Every discovered DLL root is also passed through recursive PE dependency
resolution. When active oneAPI component paths expose the same basename, the
bundler compares the candidates by SHA-256. Byte-identical candidates collapse
to one payload. Candidates with different contents fail as ambiguous instead of
being selected by search order. License discovery is independent of DLL source
paths.

## Runtime search scope

The bundler receives the environment produced by the combined Visual Studio
x64 and Intel oneAPI bootstrap. It constructs an ordered search scope from:

1. the SYCL build output directory;
2. PATH entries below the detected `ONEAPI_ROOT`;
3. the selected Visual C++ redistributable directory; and
4. Windows system directories, used only to classify system-provided imports.

Dynamic-resource discovery is restricted to active PATH entries below
`ONEAPI_ROOT`. It does not recursively scan every installed oneAPI version or
select files from inactive component versions. This ensures that the files
come from the same environment that built the worker.

The build log records:

- the compiler banner/version reported by `icx --version`;
- `ONEAPI_ROOT`;
- active oneAPI runtime directories;
- the selected source path for each semantic role;
- every PE root and resolved non-system dependency; and
- the oneAPI 2026 root licensing directory and every copied legal file; and
- every system dependency intentionally excluded from the bundle.

Paths are diagnostic only. The packaged manifest continues to contain safe
bundle-relative paths and SHA-256 hashes.

## Assembly and licensing flow

The bundle remains staged in a temporary sibling directory and is published by
rename only after all gates pass.

1. Verify the built executable and llama.cpp backend modules.
2. Discover semantic runtime DLLs from the active compiler runtime.
3. Compute recursive PE dependency closure from the executable, backend
   modules, and discovered dynamic DLL roots.
4. Reject unresolved non-system imports and content-distinct source filename
   collisions; collapse byte-identical active oneAPI aliases.
5. Copy the dependency closure into the staging directory.
6. Require the installer-provided `ONEAPI_ROOT/licensing` directory to contain
   files, then copy its complete tree without component mapping or filename
   filtering.
7. Generate the normal bundle description containing the executable and every
   staged file hash.
8. Run clean-environment verification from the staging directory.
9. Replace the destination directory atomically and update the worker manifest.

If the oneAPI 2026 root licensing directory is missing or empty, assembly fails.
Files from the worker build output are covered by the existing llama.cpp notice.
Microsoft VC runtime handling remains unchanged unless inspection shows a
current licensing gap; such a gap blocks publication rather than being ignored.

## Clean-environment verification

The verifier launches the staged `llama-server.exe --list-devices` with:

- the staging directory first on `PATH`;
- only the minimal Windows system directories retained after it;
- all oneAPI installation directories removed from `PATH`;
- oneAPI, compiler, include, and library environment variables removed; and
- the current Windows graphics-driver environment left available.

The command must exit successfully and report at least one SYCL GPU device.
For the current Windows target, the result must include the `SYCL0` device that
the runtime selects. Output and stderr are captured and included in a failure
without launching the CPU worker.

This gate intentionally requires packaging to run on a compatible Intel GPU
machine. A separate flag to bypass the hardware gate is not part of this
change, because it would allow an unverified bundle to be presented as
self-contained.

The clean environment proves DLL/resource independence for device discovery.
The existing manual release gate still loads a real GGUF and completes an
inference request from the installed VSIX, because `--list-devices` cannot prove
model-specific execution.

## Failure reporting

Errors identify one of these phases:

- toolchain identification;
- semantic runtime discovery;
- PE dependency resolution;
- license collection;
- staged clean-environment launch; or
- SYCL device discovery.

A semantic discovery failure names the missing role and lists the active
oneAPI directories searched. It does not tell the user to find a historical
filename such as `sycl8.dll`.

An unresolved import reports the importing file, requested DLL basename, and
ordered search directories. A clean-launch failure reports the command, exit
code, and captured output. The previous published bundle and manifest remain
untouched on every failure.

## Code boundaries

- `scripts/windows-sycl-bundle.mjs`
  - replace `REQUIRED_SYCL_COMPANIONS` with semantic discovery;
  - retain and improve recursive PE closure;
  - classify licenses for every copied Intel dependency; and
  - stage and verify before publication.
- `scripts/build-worker.mjs`
  - capture the compiler version;
  - pass active oneAPI directories and a process launcher to the bundler; and
  - print phase-oriented diagnostics.
- `scripts/windows-sycl-bundle.test.mjs`
  - use oneAPI 2026+ discovery fixtures;
  - cover renamed SYCL and MKL runtime DLLs; and
  - cover clean-environment verification and atomic failure behavior.
- `scripts/package-workers.test.mjs`
  - stop using `sycl8.dll` as the representative manifest file.
- `docs/WINDOWS_SYCL_HANDOFF.md`
  - document the oneAPI 2026+ environment and native clean-environment hardware
    gate.

The worker manifest schema and runtime selection code do not need to change.
They already hash arbitrary bundle file lists and enforce no automatic fallback.

## Automated verification

Platform-neutral tests use synthetic Windows directory fixtures and injected
process output. They must prove:

- the actual imported SYCL runtime basename is bundled even when it is not
  `sycl8.dll`;
- versioned MKL DLL basenames are followed through PE closure rather than
  enumerated in source;
- the Unified Runtime loader and all Level Zero adapter variants are included
  as semantic runtime DLLs;
- a compiler older than oneAPI 2026 is rejected before CMake;
- an absent required semantic role fails with the role and search scope;
- byte-identical duplicate basenames from separate active component paths
  collapse to one payload;
- duplicate basenames with different contents fail as ambiguous;
- unresolved non-system imports identify their importing file;
- system and driver DLLs are not copied;
- the complete oneAPI 2026 root licensing tree is copied without component
  mapping or filename filtering, and a missing or empty tree fails;
- clean verification receives no oneAPI development paths or variables;
- clean-launch and missing-SYCL0 failures leave the prior bundle untouched;
- a successful staged verification publishes atomically and hashes every file;
- Windows packaging still requires valid SYCL and CPU bundles; and
- SYCL failures still never launch the CPU worker.

Repository verification runs the focused script tests, the complete project
test suite, type checking, extension build, and packaging checks available on
macOS. These tests validate logic but do not satisfy the Windows hardware gate.

## Native Windows acceptance gate

On the Intel Arc Windows machine with the active oneAPI 2026.1 environment:

1. Build both workers with
   `npm run build:worker -- --target win32-x64 --backend all`.
2. Confirm logs identify oneAPI 2026.1 and never require `sycl8.dll` by name.
3. Confirm staged clean-environment discovery reports `SYCL0`.
4. Package the `win32-x64` VSIX and inspect that all manifest-declared files
   are present.
5. Install into the intended VS Code profile.
6. Run a real GGUF request in `auto` mode and confirm layers are assigned to
   `SYCL0`.
7. Induce a SYCL load/startup failure and confirm it is visible and does not
   start the CPU worker.
8. Select `cpu` explicitly and verify the CPU worker remains usable.

Completion requires the native build, clean-environment device gate, VSIX
installation, real inference, and no-fallback failure test. macOS results will
be reported separately and will not be described as Windows runtime proof.

## Rejected approaches

### Copy entire oneAPI runtime directories

This avoids identifying required files but materially inflates the VSIX,
increases license and security surface, can pull in unrelated adapters, and
still does not prove the staged worker is independent of the development
environment.

### Use only `dumpbin`

Static import closure alone cannot reliably identify dynamically loaded
Unified Runtime adapters. It can produce a bundle that passes packaging but
fails during device discovery.
