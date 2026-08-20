# Local VS Code LLM Engine PoC Design

**Status:** Implemented prototype; native Intel Windows execution remains unverified

**Date:** 2026-08-17

**Target platforms:** macOS ARM64 and Windows x64

## 1. Goal

Build a self-contained VS Code extension that runs GGUF language models locally through a bundled `llama.cpp` worker.

The extension will expose local models through VS Code Chat, Agent mode, and basic inline completion.

Users will not install Ollama, LM Studio, Python, Docker, or another inference application.

The extension will download model data or import existing GGUF files.

Source code and inference requests handled by the local provider remain on the
machine. Because the stock VS Code Chat UI cannot be model-pinned by a provider,
the user must select Local Agent and a Local LLM model; enforceable no-egress is
an external policy boundary.

## 2. Locked Decisions

The PoC will use these decisions:

- GGUF is the only supported model format.
- `llama.cpp` is the only inference runtime.
- `llama-server` is bundled inside each platform-specific VSIX.
- The extension owns the worker lifecycle.
- The worker binds only to `127.0.0.1`.
- The extension uses an automatically selected local port.
- The extension authenticates inference requests with an ephemeral secret.
- macOS ARM64 uses Metal acceleration.
- Windows x64 guarantees CPU inference.
- Windows GPU acceleration is outside the PoC.
- Hugging Face is the first model download provider.
- Direct GGUF URL downloads are supported.
- Existing GGUF files can be imported.
- VS Code's `LanguageModelChatProvider` API powers Chat and Agent integration.
- An extension-contributed `Local Agent` bounds the stock Agent harness to local
  read/search/edit selectors.
- VS Code's `InlineCompletionItemProvider` API powers ghost-text completion.
- Models advertise Agent support only after a live structured-call and
  tool-result-continuation probe succeeds.
- Complete templated messages and effective tool contracts are counted before
  inference; the extension never invents context or silently trims tools.
- Local embeddings and workspace indexing are outside the PoC.
- No inference binary is downloaded during extension activation or model download.
- No telemetry is collected.
- The extension never calls GitHub's Copilot API for local inference.
- Local Chat and completion do not consume Copilot requests or require a Copilot plan.
- Any GGUF model supported by the pinned `llama.cpp` revision can be installed.
- Chat, tool-calling, and fill-in-the-middle capabilities remain model-dependent.

## 3. PoC Definition

The PoC must prove the complete product path on both target machines.

It must include:

1. Platform-specific VSIX installation.
2. Bundled worker startup and shutdown.
3. Local GGUF import.
4. Hugging Face GGUF download.
5. Direct GGUF URL download.
6. Basic model configuration.
7. Model registration in VS Code's native model picker.
8. Streaming Chat responses.
9. Request cancellation.
10. One Agent tool-call round trip.
11. Basic inline completion.
12. Worker crash detection and recovery.
13. Offline inference after model download.
14. Zero Copilot API calls during local inference.
15. Auditable worker manifest and package checksums. Release signing is deferred.

The PoC is complete only after passing on an M4 MacBook Pro and the target Intel Windows laptop.

## 4. Non-Goals

The PoC will not include:

- ONNX or SafeTensors inference.
- Local embeddings.
- Semantic workspace indexing.
- Linux packages.
- Windows ARM packages.
- VS Code for the Web.
- Remote SSH, WSL, Dev Containers, or Codespaces execution.
- Multiple simultaneously loaded models.
- Multiple simultaneous inference workers.
- Windows Vulkan, SYCL, CUDA, or DirectML acceleration.
- Model training or fine-tuning.
- Model conversion or quantization.
- Multimodal input.
- A full visual model marketplace.
- Automatic model recommendations.
- Production telemetry or analytics.
- Marketplace publication.
- Automatic extension updates.

These exclusions protect the two-to-three-week PoC target.

## 5. User Experience

### 5.1 First Run

The extension activates without starting inference.

The user opens `Local LLM: Manage Models`.

The management view offers three actions:

- Import GGUF from disk.
- Download GGUF from Hugging Face.
- Download GGUF from a direct URL.

The Hugging Face flow prompts for a repository identifier and lists its GGUF files.

The direct URL flow prompts for an HTTPS file URL and display name.

An optional Hugging Face token supports gated repositories.

The extension stores models in its configured model directory.

The default directory is the extension's global storage `models` directory.

### 5.2 Model Selection

Installed models appear under the `Local LLM` provider in VS Code's model picker.

Selecting an unloaded model starts the bundled worker with that model.

Selecting another model stops the current worker before loading the new model.

Only one model remains loaded at a time.

### 5.3 Chat and Agent

The user selects a local model in VS Code Chat.

The extension converts VS Code messages into `llama-server` chat-completion requests.

Text streams into VS Code as the worker emits tokens.

Cancellation aborts the HTTP request and stops generation.

Tool definitions pass through the provider only after the selected model passes
behavioral tool validation. Local Agent declares a bounded set; VS Code may
expand selectors into concrete contracts, which are exact-counted at runtime.

The PoC must demonstrate one successful tool call and tool-result continuation.

### 5.4 Inline Completion

Inline completion remains disabled until a compatible coding model is selected.

The provider debounces editor changes before requesting a completion.

New edits cancel stale completion requests.

Chat requests take priority over inline completions.

The PoC supports one configured fill-in-the-middle prompt preset.

## 6. Architecture

```text
VS Code
├── Native Chat and Agent UI
├── Editor inline completion UI
└── Local LLM extension
    ├── LanguageModelChatProvider
    ├── Local Agent contribution
    ├── InlineCompletionItemProvider
    ├── priority InferenceScheduler
    ├── ModelManager
    ├── ModelDownloader
    ├── WorkerManager
    └── platform worker
        ├── darwin-arm64/llama-server
        └── win32-x64/llama-server.exe
             │
             └── local GGUF model
```

The TypeScript extension is shared across platforms.

Each VSIX contains only its matching worker artifact.

VS Code already supports platform-specific extension packages for native dependencies.

## 7. Component Boundaries

### 7.1 Extension Core

The extension core registers commands, settings, providers, and lifecycle handlers.

It does not contain model-download logic or worker-process logic.

The extension declares `extensionKind: ["ui"]` so inference remains on the local machine.

### 7.2 Model Registry

The model registry persists installed-model metadata.

Each record contains:

- Stable model identifier.
- Display name.
- Absolute GGUF path.
- File size.
- Local SHA-256 digest.
- Source type.
- Source URL when applicable.
- Hugging Face repository and filename when applicable.
- Imported or downloaded timestamp.
- User configuration overrides.
- Capability flags.

The registry never stores prompts, generated content, or source-code context.

### 7.3 Model Downloader

The downloader handles Hugging Face files and direct HTTPS URLs.

Provider-specific behavior lives behind a model-source adapter interface.

The PoC contains `HuggingFaceSource` and `DirectUrlSource` adapters.

Provider credentials use VS Code `SecretStorage`.

Credentials never appear in the model registry, settings, URLs, or logs.

Downloads use a `.partial` file in the target model directory.

The downloader resumes when the remote server supports byte ranges.

Successful downloads receive a locally computed SHA-256 digest.

The final file appears through an atomic rename.

Cancellation preserves a resumable partial file.

The downloader accepts only HTTPS remote URLs.

It checks available disk space before starting a known-size download.

The downloader never executes repository code.

The downloader never honors `trust_remote_code` behavior.

### 7.4 Model Manager

The model manager coordinates the registry, downloader, and worker manager.

It validates the `.gguf` extension before registration.

The worker performs authoritative GGUF compatibility validation during loading.

Import copies the selected file into the configured model directory.

Removing a model stops the worker before deleting the managed file.

Deletion always requires explicit user confirmation.

### 7.5 Worker Manager

The worker manager is the only component allowed to spawn `llama-server`.

It starts the worker with `shell: false` and an explicit argument array.

It passes the selected model path, context size, thread settings, and acceleration settings.

It binds the worker to `127.0.0.1` on an available port.

It retries worker startup when another process claims the selected port.

It generates a new API secret for every worker process.

It polls `GET /health` until the model is ready.

The startup timeout is 120 seconds.

Graceful shutdown receives five seconds before forced termination.

Unexpected termination changes the worker state to `failed`.

An interrupted inference request fails without automatic replay.

The manager allows three automatic restarts within five minutes.

A fourth failure requires explicit user action.

The manager captures standard output and standard error in a dedicated Output channel.

Logs exclude prompt bodies and generated content by default.

### 7.6 Worker Client

The worker client owns authenticated loopback requests.

It supports:

- Health checks.
- Model metadata queries.
- Token counting.
- Streaming chat completions.
- Tool-call responses.
- Fill-in-the-middle completion requests.
- Cancellation through `AbortSignal`.

The client rejects non-loopback worker URLs.

### 7.7 Language Model Provider

The provider implements VS Code's `LanguageModelChatProvider` contract.

It reports installed models through `provideLanguageModelChatInformation`.

It streams responses through `provideLanguageModelChatResponse`.

It counts tokens through `provideTokenCount`.

It maps text, tool calls, and tool results between VS Code and the worker protocol.

It declares tool-calling capability only for validated model presets.

### 7.8 Inline Completion Provider

The completion provider implements `InlineCompletionItemProvider`.

It captures bounded prefix and suffix context from the active document.

It never reads the entire workspace for a completion request.

It debounces requests by 250 milliseconds.

It limits generated completion output to 64 tokens.

It cancels the previous request after any document change.

It suppresses requests while Chat generation is active.

## 8. Process and Network Boundaries

The extension spawns only the worker bundled inside its installation directory.

The extension never searches `PATH` for an inference executable.

The extension never invokes an installed Ollama or LM Studio instance.

Inference traffic uses authenticated loopback HTTP only.

The worker listens on `127.0.0.1`, never `0.0.0.0`.

External network traffic occurs only during user-initiated model downloads.

Offline inference requires no GitHub or model-provider connection.

The extension never sends prompts, source content, tokens, or model metadata to GitHub.

The PoC does not claim control over unrelated traffic from VS Code or other extensions.

## 9. Platform Strategy

### 9.1 macOS ARM64

The macOS package targets `darwin-arm64`.

Its worker is compiled from pinned `llama.cpp` source with Metal enabled.

The M4 MacBook Pro is the required macOS acceptance machine.

Metal is the default acceleration mode.

### 9.2 Windows x64

The Windows package targets `win32-x64`.

Its worker is compiled from the same pinned `llama.cpp` revision.

CPU inference is the guaranteed PoC backend.

The build must run without administrator elevation or developer tools.

The build must not download runtime DLLs after installation.

The worker uses a CPU-only x64 build with runtime hardware detection.

Unsupported CPU instructions must produce a clear startup error.

### 9.3 Build Outputs

Continuous integration produces:

- One `darwin-arm64` VSIX.
- One `win32-x64` VSIX.
- One SHA-256 checksum for each VSIX.
- One worker checksum inside each build manifest.
- The pinned `llama.cpp` commit identifier.
- Compiler and linker versions.
- Effective `llama.cpp` build flags.
- One detached signature or CI provenance attestation for each VSIX.

## 10. Configuration

The PoC exposes these settings:

| Setting | Default | Purpose |
|---|---:|---|
| `localLlm.modelDirectory` | Extension global storage | Managed GGUF location |
| `localLlm.defaultModelId` | Empty | Preferred installed model |
| `localLlm.contextSize` | `32768` | Physical llama.cpp context tokens |
| `localLlm.maxTools` | `8` | Maximum complete tool contracts accepted |
| `localLlm.maxOutputTokens` | `2048` | Chat output ceiling |
| `localLlm.startupTimeoutSeconds` | `600` | Slow model-load ceiling |
| `localLlm.cpuThreads` | `0` | Automatic thread selection |
| `localLlm.acceleration` | `auto` | Metal on M4 and CPU on Windows |
| `localLlm.temperature` | `0.2` | Chat sampling temperature |
| `localLlm.inline.enabled` | `true` | Enable compatible inline completion |
| `localLlm.inline.maxTokens` | `64` | Inline output ceiling |
| `localLlm.inline.debounceMilliseconds` | `250` | Stale-request debounce |
| `localLlm.logLevel` | `info` | Output-channel verbosity |

Model-specific overrides live in the model registry.

Invalid values fail before worker startup.

## 11. State Model

The worker uses these states:

```text
stopped
  → starting
  → ready
  → stopping
  → stopped

starting | ready
  → failed
  → starting
```

Only one state transition executes at a time.

Model switching completes shutdown before starting the replacement worker.

VS Code deactivation always requests worker shutdown.

## 12. Error Handling

User-facing errors must identify the failed operation and recovery action.

Required error classes include:

- Worker artifact missing.
- Worker launch denied.
- Worker startup timeout.
- Worker crashed.
- Port allocation failed.
- Model file missing.
- Model format unsupported.
- Model exceeds available memory.
- Download interrupted.
- Download URL rejected.
- Download integrity failed.
- Chat request rejected.
- Tool-calling unsupported.
- Inline completion unsupported.

Raw worker errors remain available in the Output channel.

Prompt or source content must not appear in normal logs.

## 13. Supply Chain and Auditability

The repository is open source from the first commit.

The project pins one exact upstream `llama.cpp` commit.

Continuous integration builds workers from pinned source.

The VSIX never downloads executable code at runtime.

Build manifests record source revisions, tools, flags, and checksums.

Third-party dependencies require explicit lockfile entries.

Runtime dependencies remain minimal.

Public distribution requires signed release artifacts or Marketplace signing.

PoC artifacts require published SHA-256 checksums and signed CI provenance.

## 14. Testing Strategy

### 14.1 Unit Tests

Unit tests cover:

- Model-registry serialization.
- Download URL validation.
- Resumable-download decisions.
- Atomic download finalization.
- Worker state transitions.
- Restart-limit enforcement.
- Worker argument construction.
- Loopback URL enforcement.
- VS Code message conversion.
- Streaming event conversion.
- Tool-call conversion.
- Inline-context bounds.
- Cancellation propagation.

### 14.2 Contract Tests

Contract tests run against a fake local HTTP worker.

They cover health, streaming, errors, cancellation, and tool-call payloads.

### 14.3 Worker Integration Tests

Integration tests run the packaged worker with a small approved GGUF fixture.

They verify startup, health, generation, cancellation, and shutdown.

### 14.4 VS Code Extension Tests

Extension-host tests verify activation, commands, provider registration, and settings.

### 14.5 Hardware Acceptance

The M4 MacBook Pro must pass:

- VSIX installation.
- Metal worker startup.
- Model download and import.
- Streaming Chat.
- Cancellation.
- One tool call.
- Inline completion.
- Offline inference.
- Worker restart recovery.
- No Copilot API requests from the extension.
- Signed CI provenance and matching checksums.

The Intel Windows laptop must pass the same scenarios using CPU inference.

## 15. Implementation Stages

### Stage 1: Extension and Worker Foundation

Create the TypeScript extension shell and platform build pipeline.

Bundle a pinned `llama-server` for each target.

Prove startup, health, shutdown, and crash detection.

### Stage 2: Local Model Inference

Add the registry and local GGUF import.

Start the worker with an imported model.

Stream one test prompt through a VS Code command.

### Stage 3: Native VS Code Chat

Register `LanguageModelChatProvider`.

Expose installed models in the model picker.

Implement token counting, streaming, cancellation, and model switching.

### Stage 4: Model Downloads and Configuration

Add Hugging Face and direct HTTPS downloads.

Add resume, progress, cancellation, hashing, and atomic finalization.

Expose the defined user settings.

### Stage 5: Agent and Inline Completion

Add one validated tool-calling preset.

Prove one tool-call round trip.

Add bounded, cancellable inline completion.

### Stage 6: Dual-Platform Acceptance

Package both VSIX targets.

Run the complete acceptance suite on both machines.

Record worker, extension, model, and hardware versions.

## 16. Exit Criteria

The PoC succeeds when every requirement in Section 3 passes on both target machines.

The PoC fails when either platform requires a separately installed inference runtime.

The PoC also fails when inference sends prompt or source content beyond loopback.

Performance measurements inform later work but do not block PoC completion.

## 17. Estimated Schedule

The expected AI-assisted schedule is:

| Stage | Duration |
|---|---:|
| Worker and extension foundation | 2–3 days |
| Local model inference | 1–2 days |
| Native VS Code Chat | 2–3 days |
| Downloads and configuration | 2–3 days |
| Agent and inline completion | 3–4 days |
| Dual-platform testing and fixes | 3–5 days |

The total expected duration is two to three working weeks.

Native packaging and hardware testing control the schedule.

## 18. Deferred Product Work

After the PoC, product work may add:

- Local embedding models.
- Semantic workspace indexing.
- Windows GPU acceleration.
- Linux packages.
- Multiple runtime families.
- Model discovery and recommendations.
- Model license presentation.
- Storage quotas and cleanup policies.
- Production artifact signing.
- Marketplace publication.
- Performance benchmarks and compatibility matrices.

## 19. Technical References

- [VS Code Language Model Chat Provider API](https://code.visualstudio.com/api/extension-guides/ai/language-model-chat-provider)
- [VS Code extension-contributed custom agents](https://code.visualstudio.com/api/references/contribution-points#contributes.chatAgents)
- [VS Code AI language models](https://code.visualstudio.com/docs/agent-customization/language-models)
- [VS Code platform-specific extensions](https://code.visualstudio.com/api/working-with-extensions/publishing-extension#_platformspecific-extensions)
- [`llama.cpp` repository](https://github.com/ggml-org/llama.cpp)
- [`llama-server` documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
