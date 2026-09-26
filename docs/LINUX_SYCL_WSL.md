# Arch WSL (linux-x64) SYCL worker

## Architecture

The extension runs as a workspace extension inside WSL (`extensionKind:
["workspace"]`) so it executes beside remote files. Inline completions are
registered for both `file` and `vscode-remote` schemes. Activation logs
`vscode.env.remoteName`, platform, architecture, and worker target.

The `linux-x64` VSIX bundles a locally built llama.cpp SYCL worker pinned to
the same commit as the other platforms:

```text
commit: 60eeeb6082c1126bb8bc72902c83123cd056811b
release: b10472
```

`auto` and explicit `cpu` use the same `resources/workers/linux-x64/sycl/`
bundle. SYCL mode passes the discovered device (`--device <discovered>`,
defaulting to the first `--list-devices` result) with `--fit on`,
`--split-mode none`, and `--main-gpu 0`. `localLlm.syclDevice` overrides the
discovered device on multi-GPU systems. A SYCL failure never restarts in CPU
mode automatically.

`GGML_SYCL_F16=OFF` is kept until the Arc 140T BF16 correctness tests pass
(`ggml-org/llama.cpp#27771`).

## Prerequisites

Windows host:

- Current Intel Arc Pro graphics driver with WSL support.
- WSL2 with `/dev/dxg` present.

Arch WSL guest:

- Runtime: `intel-compute-runtime`, `level-zero-loader`.
- Build: `intel-oneapi-dpcpp-cpp`, `intel-oneapi-mkl-sycl`, `onedpl`, `cmake`,
  `ninja`, `git`, `patchelf`.

glibc, libstdc++, Level Zero, and the Intel GPU driver stay as Arch system
dependencies. The bundle stages `llama-server`, `libggml*.so`, SYCL, Unified
Runtime, oneMKL, oneDNN, OpenMP, and TBB files discovered recursively with
`readelf`/`ldd`, with staged ELF runtime paths set to `$ORIGIN` via
`patchelf`. Every staged file is hashed in `resources/workers/manifest.json`.

## Build, verify, smoke, package

On x64 Linux (Arch WSL):

```shell
npm ci
npm run build:worker -- --target linux-x64
npm run verify:linux-worker
npm run smoke:worker -- --target linux-x64 --backend sycl --model /absolute/model.gguf
npm run package -- linux-x64
```

Install into the WSL remote:

```shell
npm run package:install:wsl
```

Or explicitly:

```shell
code --install-extension dist/vsix/linux-x64/local-llm-engine-0.3.4-linux-x64.vsix --force
```

## Acceptance

- WSL:Arch reports the platform, architecture, and worker target.
- Activation runs on `linux-x64`.
- The Intel Arc Pro 140T is discovered through the clean default, Level Zero,
  then OpenCL probe sequence.
- CPU and SYCL deterministic runs agree.
- Chat, Agent, and inline completions work.
- No orphan worker remains.
- A real GGUF smoke run passes.
