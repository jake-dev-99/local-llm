# Official Windows SYCL Archive Design

## Decision

The `win32-x64` VSIX consumes the official pinned `ggml-org/llama.cpp`
Windows SYCL release archive. This repository does not compile a Windows
worker, inspect PE dependencies, activate Intel oneAPI, discover Visual Studio,
or reconstruct an upstream binary distribution.

The pinned artifact is:

- llama.cpp commit: `60eeeb6082c1126bb8bc72902c83123cd056811b`
- llama.cpp release: `b10472`
- asset: `llama-b10472-bin-win-sycl-x64.zip`
- URL: `https://github.com/ggml-org/llama.cpp/releases/download/b10472/llama-b10472-bin-win-sycl-x64.zip`
- size published by GitHub: `119700367` bytes
- SHA-256 published by GitHub and independently verified:
  `0c4c50f1e9805933e043d4970f0c2050e4fb5343b8ac0244a49efaa474705830`

The official release workflow builds the SYCL payload and merges the matching
Windows CPU payload into the final SYCL ZIP. One extracted archive therefore
supplies both extension modes:

- `auto` selects bundle `sycl` with backend `sycl`;
- `cpu` selects the same bundle `sycl` with backend `cpu`.

The backend changes launch arguments. It does not select a second executable.
There is no automatic SYCL-to-CPU fallback.

## Build and package contract

The complete official archive payload is extracted unchanged into:

```text
resources/workers/win32-x64/sycl/
```

The archive itself is cached under `build/` and is not committed or embedded
as a nested ZIP in the VSIX. The extracted Windows payload is also generated
and ignored by Git. The checked-in manifest contains the deterministic file
list and hashes for the pinned artifact, allowing packaging and runtime startup
to verify every installed worker file.

`npm run package -- win32-x64` is the complete Windows assembly command. It:

1. downloads the archive when the cache is absent;
2. verifies the pinned archive size and SHA-256;
3. safely extracts every regular file without pruning;
4. rejects absolute paths, parent traversal, symbolic links, duplicate output
   paths, and an archive without `llama-server.exe`;
5. hashes every extracted file and creates the single Windows manifest bundle;
6. replaces the staged Windows directory and manifest transactionally;
7. verifies the selected platform bundle;
8. builds the extension JavaScript; and
9. packages the target VSIX.

Packaging does not execute `llama-server.exe` and does not require an Intel
GPU. It requires Node.js/npm plus network access on a cold cache. It does not
require Intel oneAPI, Visual Studio, CMake, Ninja, `icx`, `dumpbin`, MinGW, or a
manual file copy.

`npm run build:worker -- --target win32-x64` invokes the same archive
preparation operation for developers who want to stage the Windows payload
without creating a VSIX. The Darwin worker continues to use the existing
source-build path.

## Runtime contract

Windows `auto` verifies the manifest-selected executable and performs exactly
one preflight:

```text
llama-server.exe --list-devices
```

The child environment removes caller-supplied `ONEAPI_DEVICE_SELECTOR`,
`SYCL_DEVICE_FILTER`, `UR_ADAPTERS_FORCE_LOAD`, and
`UR_ADAPTERS_SEARCH_PATH` variables case-insensitively. `PATH` contains only
the worker directory and the Windows system directories. No Level Zero or
OpenCL selector is forced; the official bundled runtime chooses its usable
adapter. Preflight succeeds only when output contains `SYCL0`.

The same sanitized environment is passed to the real SYCL worker process.
Failure is surfaced with captured stderr and never resolves or starts a CPU
fallback. Explicit `cpu` mode uses the same executable with CPU launch
arguments and skips SYCL discovery.

## Deprecated implementation removed

The following modules and their tests are deleted because their responsibilities
belong to the official release producer:

- `scripts/windows-oneapi.mjs`
- `scripts/windows-oneapi.test.mjs`
- `scripts/windows-vs-tools.mjs`
- `scripts/windows-vs-tools.test.mjs`
- `scripts/windows-sycl-bundle.mjs`
- `scripts/windows-sycl-bundle.test.mjs`
- `scripts/windows-sycl-runtime.mjs`
- `scripts/windows-sycl-runtime.test.mjs`
- `scripts/worker-build-options.mjs`
- `scripts/worker-build-options.test.mjs`

The old root Windows executable and separate `win32-x64/cpu/` bundle contract
are also removed. Existing v2 manifest parsing, per-file hashing, target VSIX
packaging, smoke testing, backend-specific launch arguments, diagnostics, and
the no-fallback policy remain.

The version-aware dependency-reconstruction design is deleted rather than kept
as an alternative path. The original Windows SYCL design is updated to point
to this document for artifact assembly.

## Verification boundaries

### Platform-neutral packaging gate

Automated tests and packaging prove:

- the cached/downloaded archive must match the pinned size and SHA-256;
- unsafe ZIP entries cannot escape the staging directory;
- a failed download, hash, extraction, or publication leaves the previous
  Windows directory and manifest intact;
- the complete extracted payload is represented by sorted manifest hashes;
- both Windows modes resolve the same physical bundle with different backends;
- all manifest files are present in the VSIX;
- Darwin files are excluded from the Windows VSIX;
- Windows packaging does not execute an archive executable; and
- no deleted oneAPI, Visual Studio, or dependency-discovery module remains
  reachable.

### Native Windows probe

`npm run verify:windows-worker` runs on x64 Windows against the staged bundle
with the same clean environment used by the extension. It runs:

```text
sycl-ls.exe --verbose --ignore-device-selectors
llama-server.exe --version
llama-server.exe --list-devices
```

Every process must exit zero and `--list-devices` must report `SYCL0`. Failure
output includes the executable, working directory, decimal and hexadecimal exit
code, stdout, and stderr.

The identical probe may be run against three separately extracted locations to
classify a failure:

1. direct official archive extraction;
2. `resources/workers/win32-x64/sycl`; and
3. the extracted completed VSIX payload.

Interpretation is deterministic:

- all three fail: host driver/runtime or upstream artifact compatibility;
- official passes and staging fails: repository extraction/staging;
- official and staging pass and VSIX fails: VSIX omission or alteration;
- all three pass: artifact discovery and host runtime are valid.

### Native model smoke and installed extension

After discovery passes, the Windows host runs:

```shell
export LOCAL_LLM_SMOKE_MODEL='C:/absolute/path/model.gguf'
npm run smoke:worker -- --backend sycl --model "$LOCAL_LLM_SMOKE_MODEL"
npm run smoke:worker -- --backend cpu --model "$LOCAL_LLM_SMOKE_MODEL"
```

SYCL acceptance requires `SYCL0`, model load with GPU layers assigned, `/health`,
and one chat completion. CPU acceptance requires the same official executable,
offload disabled, `/health`, and one chat completion. The completed VSIX is then
installed into the intended VS Code profile and tested in `auto` and explicit
`cpu` modes. An induced SYCL failure must remain visible and must not start a
CPU fallback.

macOS tests can establish archive, manifest, extension, and VSIX assembly logic.
They cannot establish Intel Arc runtime compatibility; that claim requires the
native Windows probe and model smoke gates above.
