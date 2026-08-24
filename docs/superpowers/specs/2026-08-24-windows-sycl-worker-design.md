# Self-Contained Windows SYCL Worker Design

Date: 2026-08-24
Status: Approved

## Summary

Ship two isolated Windows worker bundles in the same `win32-x64` VSIX:

- a self-contained Intel SYCL worker built with `GGML_SYCL=ON`; and
- the existing portable CPU worker.

On Windows, `localLlm.acceleration: auto` selects the SYCL worker. Selecting
`cpu` explicitly selects the CPU worker. A SYCL failure must never launch the
CPU worker automatically: it is an actionable error that should be visible to
the user.

The macOS worker and its existing Metal/CPU argument selection remain
unchanged.

## Goals

- Use the Intel Arc GPU on supported Windows systems through llama.cpp's SYCL
  backend.
- Package all required non-system SYCL and Level Zero runtime DLLs in the VSIX,
  so the installed extension requires only a compatible Intel graphics driver.
- Preserve the current CPU worker as an explicit user-selected fallback.
- Verify the integrity of every executable and DLL in each worker bundle.
- Fail packaging when either Windows bundle is missing, incomplete, or does
  not match the manifest.
- Surface device discovery and worker startup failures without silently
  changing execution backends.

## Non-goals

- Automatically retrying a failed SYCL launch on the CPU worker.
- Requiring Intel oneAPI to be installed on the extension user's machine.
- Supporting non-Intel GPU backends in this change.
- Changing macOS Metal behavior.
- Claiming Windows runtime success based only on macOS tests or a successful
  cross-platform package build.

## Artifact layout

The Windows VSIX contains two dependency-isolated directories:

```text
resources/workers/win32-x64/
  sycl/
    llama-server.exe
    <required Intel SYCL and Level Zero DLLs>
  cpu/
    llama-server.exe
```

The bundles do not share DLLs. This prevents one worker's runtime dependencies
from changing how the other worker loads.

The worker manifest moves to a versioned bundle model. Each platform maps the
configuration modes it supports to a named bundle, and each bundle declares an
executable plus the complete list of packaged files and their SHA-256 hashes.
A representative shape is:

```json
{
  "manifestVersion": 2,
  "llamaCppCommit": "<pinned commit>",
  "llamaCppBuild": "<pinned build>",
  "platforms": {
    "darwin-arm64": {
      "modes": { "auto": "default", "cpu": "default" },
      "bundles": {
        "default": {
          "executable": "resources/workers/darwin-arm64/llama-server",
          "files": [
            { "path": "resources/workers/darwin-arm64/llama-server", "sha256": "..." }
          ]
        }
      }
    },
    "win32-x64": {
      "modes": { "auto": "sycl", "cpu": "cpu" },
      "bundles": {
        "sycl": {
          "executable": "resources/workers/win32-x64/sycl/llama-server.exe",
          "files": [
            { "path": "resources/workers/win32-x64/sycl/llama-server.exe", "sha256": "..." },
            { "path": "resources/workers/win32-x64/sycl/<runtime>.dll", "sha256": "..." }
          ]
        },
        "cpu": {
          "executable": "resources/workers/win32-x64/cpu/llama-server.exe",
          "files": [
            { "path": "resources/workers/win32-x64/cpu/llama-server.exe", "sha256": "..." }
          ]
        }
      }
    }
  }
}
```

Manifest parsing rejects absolute paths, path traversal, duplicate files,
undeclared bundle files, missing files, and mismatched hashes. Runtime and
packaging use the same manifest-validation module so their integrity rules
cannot drift.

## Build flow

`scripts/build-worker.mjs` gains a Windows backend selector:

```text
npm run build:worker -- --target win32-x64 --backend cpu
npm run build:worker -- --target win32-x64 --backend sycl
npm run build:worker -- --target win32-x64 --backend all
```

`all` is the default for a native Windows build. The CPU build preserves the
existing flags and static runtime behavior. The SYCL build:

- is supported only on native Windows;
- uses the same pinned llama.cpp commit as every other worker;
- requires an Intel oneAPI build toolchain, resolved from the build machine's
  environment with an actionable error when it is unavailable;
- sets `GGML_SYCL=ON` and targets Intel devices;
- writes into a backend-specific build directory; and
- copies the resulting server and its non-system runtime dependency closure
  into `resources/workers/win32-x64/sycl/`.

The dependency collector follows executable and DLL imports recursively. It
excludes Windows system libraries, includes only redistributable runtime files,
and fails when a non-system dependency cannot be resolved or is not approved
for redistribution. The repository's third-party notices record the shipped
Intel runtime components.

After a successful build, the script updates the hashes for the bundle it
produced. `--backend all` produces and records both bundles.

## Packaging flow

`npm run package -- win32-x64` remains an assembly step rather than invoking a
compiler toolchain. It validates both Windows bundles and every manifest file
before building the extension. It fails if either bundle is absent or invalid.

The Windows VSIX includes both Windows bundles and excludes the Darwin bundle.
The Darwin VSIX includes the Darwin bundle and excludes the entire Windows
worker tree. Existing target-specific output directories remain unchanged.

## Runtime selection

The runtime resolves the executable from the manifest using the current
platform and `localLlm.acceleration` mode:

| Platform | Mode | Bundle |
| --- | --- | --- |
| macOS ARM64 | `auto` | Existing Metal-capable worker |
| macOS ARM64 | `cpu` | Existing worker with CPU-only arguments |
| Windows x64 | `auto` | SYCL worker |
| Windows x64 | `cpu` | CPU worker |

Before starting a Windows SYCL model server, the manager executes the selected
SYCL server with `--list-devices`. Discovery must return at least one `SYCL`
GPU device. The initial implementation selects `SYCL0` explicitly.

The SYCL server arguments request maximum GPU layer offload and name `SYCL0`
as the device. They must not include the CPU-only `--device none`, zero GPU
layers, or `--no-op-offload` arguments. Device discovery or startup failure
ends the operation; the manager never changes the selected bundle.

The CPU worker retains the current CPU-only flags. macOS `auto` retains Metal
memory fitting and macOS `cpu` retains the current CPU-only behavior.

## Failure reporting

SYCL failures are grouped by phase:

1. bundle integrity or DLL loading;
2. SYCL device discovery;
3. model server startup or model loading; and
4. unexpected worker exit.

The user-facing error identifies the SYCL phase, preserves relevant worker
stderr in the output channel, and explains that setting
`localLlm.acceleration` to `cpu` will select the CPU worker. It does not change
the setting or launch the CPU worker itself.

Automatic restarts, where already supported, always restart the same selected
bundle. A restart cannot change SYCL to CPU.

## Verification

Automated coverage includes:

- manifest schema, safe-path, completeness, and hash validation;
- mapping Windows `auto` to SYCL and Windows `cpu` to CPU;
- mapping both macOS modes to the existing Darwin worker;
- SYCL launch arguments and absence of CPU-only flags;
- parsing a valid `SYCL0` discovery result;
- loud failures for missing devices, DLL/load errors, startup failures, and
  malformed discovery output;
- proof that no CPU process is spawned after any SYCL failure;
- Windows VSIX contents include both complete bundles;
- Darwin VSIX contents exclude all Windows worker files; and
- existing extension tests, type checking, build, and Darwin packaging.

The Windows Arc hardware smoke test is a separate release gate:

1. build both Windows bundles with the native Windows command;
2. package and install the resulting Windows VSIX into the intended VS Code
   profile;
3. verify `--list-devices` reports `SYCL0`;
4. load a GGUF and verify the worker log assigns layers to `SYCL0`;
5. complete one inference request in `auto` mode;
6. switch explicitly to `cpu` and complete one inference request; and
7. induce a SYCL startup failure and verify that it is reported without a CPU
   worker launch.

macOS test success does not satisfy this Windows hardware gate.

## Delivery boundary

The implementation can be developed and its platform-neutral behavior tested
on macOS. A self-contained Windows bundle is complete only after the native
Windows build has produced the executable and dependency set, the manifest has
been updated, Windows packaging has passed, and the Arc hardware smoke test has
completed. These outcomes must be reported separately.
