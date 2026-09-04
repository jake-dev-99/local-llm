# Windows SYCL worker handoff

## Branch and architecture

Use branch:

```text
codex/windows-sycl-worker
```

The authoritative design is
[`docs/superpowers/specs/2026-09-03-official-windows-sycl-archive-design.md`](superpowers/specs/2026-09-03-official-windows-sycl-archive-design.md).

Windows no longer compiles or reconstructs llama.cpp. The `win32-x64` VSIX
uses the complete official `ggml-org/llama.cpp` SYCL archive pinned to:

```text
commit: 60eeeb6082c1126bb8bc72902c83123cd056811b
release: b10472
asset: llama-b10472-bin-win-sycl-x64.zip
size: 119700367 bytes
sha256: 0c4c50f1e9805933e043d4970f0c2050e4fb5343b8ac0244a49efaa474705830
```

The upstream release workflow merges the matching Windows CPU payload into the
final SYCL ZIP. `auto` and explicit `cpu` therefore use the same
`resources/workers/win32-x64/sycl/llama-server.exe`. SYCL mode requests
`SYCL0`; CPU mode disables device offload. A SYCL failure never falls back to
CPU automatically.

## Prerequisites

Packaging needs:

- Node.js 22 or newer;
- npm; and
- network access to GitHub only when the verified archive cache is absent.

Packaging does not need Intel oneAPI, Visual Studio, CMake, Ninja, a C/C++
compiler, MinGW, a GPU, or a manual file copy. The installed extension needs a
compatible Intel graphics driver for SYCL acceleration. The Intel development
toolkit is not an installed-extension prerequisite.

## Build the Windows VSIX

From Git Bash, PowerShell, or another shell capable of running Node/npm:

```shell
git fetch origin
git switch codex/windows-sycl-worker
git pull --ff-only origin codex/windows-sycl-worker
npm ci
npm run package -- win32-x64
```

That one package command:

1. downloads the pinned ZIP to `build/worker-downloads/` when absent;
2. verifies exact byte count and SHA-256;
3. safely extracts the complete payload into
   `resources/workers/win32-x64/sycl/`;
4. generates and verifies the manifest file hashes;
5. builds the extension; and
6. writes
   `dist/vsix/win32-x64/local-llm-engine-0.3.4-win32-x64.vsix`.

The package phase never launches an executable from the archive. A failed
download, checksum, extraction, or publication does not replace the previous
Windows tree or manifest.

To stage the official payload without packaging:

```shell
npm run build:worker -- --target win32-x64
```

The extracted payload is generated and Git-ignored. The checked-in manifest is
the deterministic integrity declaration for the pinned official bytes.

## Inspect the package

Run the source gates:

```shell
npm test
npm run typecheck
npm run build
git diff --check
```

List the VSIX with a ZIP viewer and verify:

- every `resources/workers/win32-x64/sycl/` file declared by
  `resources/workers/manifest.json` is present;
- `resources/workers/darwin-arm64/` is absent;
- `resources/workers/win32-x64/cpu/` is absent; and
- `llama-b10472-bin-win-sycl-x64.zip` is not nested inside the VSIX.

## Native clean-environment probe

On the x64 Windows Intel Arc host, run:

```shell
npm run verify:windows-worker
```

The command verifies the staged manifest bundle and runs, in order:

```text
sycl-ls.exe --verbose --ignore-device-selectors
llama-server.exe --version
llama-server.exe --list-devices
```

It removes caller-supplied SYCL selector and Unified Runtime adapter variables.
Its `PATH` contains only the bundle, `System32`, and the Windows root. It does
not force Level Zero or OpenCL. Every command must exit zero and device output
must contain `SYCL0`.

Failures report the executable, working directory, stdout, stderr, and both
decimal and unsigned hexadecimal exit code. For example, Windows status
`3221225477` is reported as `0xC0000005`.

To apply the identical probe to a direct official extraction or an extracted
VSIX payload, provide its absolute directory:

```shell
npm run verify:windows-worker -- --bundle 'C:\absolute\path\to\sycl'
```

Compare these locations:

1. a direct extraction of the official archive;
2. `resources/workers/win32-x64/sycl`; and
3. the completed VSIX's extracted `resources/workers/win32-x64/sycl`.

Classify results as follows:

- all fail: host graphics driver/runtime or upstream artifact compatibility;
- official passes, staging fails: repository extraction/staging defect;
- official and staging pass, VSIX fails: VSIX omission or alteration;
- all pass: bundle discovery and host runtime are valid.

## Model smoke gates

Use an absolute GGUF path that fits the Arc GPU:

```shell
export LOCAL_LLM_SMOKE_MODEL='C:/absolute/path/model.gguf'
npm run smoke:worker -- --backend sycl --model "$LOCAL_LLM_SMOKE_MODEL"
npm run smoke:worker -- --backend cpu --model "$LOCAL_LLM_SMOKE_MODEL"
```

The SYCL run must report `SYCL0`, load model layers onto the GPU, pass
`/health`, and complete the chat request. The CPU run must use the same official
executable with offload disabled, skip SYCL discovery, pass `/health`, and
complete the chat request.

## Installed VSIX gates

Install the exact generated VSIX into the intended VS Code profile. Verify one
request in `auto` and one in explicit `cpu` mode. Then induce a SYCL preflight
failure in the disposable test installation and confirm the error remains
visible and no CPU fallback process starts.

macOS tests and packaging prove source logic and artifact assembly only. Do not
claim Intel Arc runtime acceptance until the native probe, model smoke, and
installed-VSIX gates pass on Windows.

## Final-answer cache and tool-boundary gate

With Qwen3.5-4B and the bundled Local Agent, repeat a coding request that reaches
the configured tool-call limit. Keep the model, context size, and tool set
unchanged during this check. The final request must:

- retain the same tool definitions and report `toolChoice=none`;
- log `final answer: tool execution disabled; preserving ...`;
- reuse most of the previous prompt, visible in `timings ... cached tokens`,
  processing only the new result/control tail instead of resetting to zero;
- emit no further executable tool call;
- produce normal final text, or an explicit final-only protocol error if the
  model still emits an unquoted tool call. Code examples remain intact, and
  previous edits are neither replayed nor rolled back by this check.

The runtime note is appended to the final tool result without modifying the
stored conversation. In Qwen's template, adding a new user message instead
changes which historical assistant reasoning blocks are rendered and loses
much of the otherwise reusable prefix.

A macOS cache-mechanics check with the pinned `b10472` worker, existing
Qwen3-4B-Instruct-2507-Q8_0 weights, and the official Qwen3.5-4B template showed
273 cached / 3,151 processed tokens with a new user instruction. The corrected
production client retained 3,418 cached tokens and processed 70. This is a
controlled template/cache check, **not** Qwen3.5-4B response-quality or native
Windows/SYCL acceptance; repeat the installed-VSIX gate above on the target host.
