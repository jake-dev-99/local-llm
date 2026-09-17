# Safetensors support — scope (rescope v2)

Status: scope. Date: 2026-09-17. Supersedes the 2026-09-16 scope in this file.

This document is self-contained. It does not depend on the external
"Universal Local Safetensors LLM Runtime" spec: every section reference
(`§2`, `§4`, `§15`–`§36`) from v1 is replaced below by an inline definition
grounded in this repo. Where v1 cited an outside body, this version states
the rule directly and names the code that owns it. Inferred gaps are marked
`[inference]`; verified facts name the file.

## 1. Goal and positioning

Support Hugging Face Safetensors checkpoint directories alongside single-file
GGUF models in this VS Code extension.

- Safetensors = the *universal, day-one* path: run any architecture
  Transformers supports, including models with no GGUF conversion.
- GGUF via llama.cpp = the *fast, daily-driver* path. Both stay.
- Do not position Safetensors as a performance path. bf16 checkpoints are
  large and offload easily into unusable speed (see §7).

MVP: install / inspect / load / stream chat / cancel / unload / recover for a
local checkpoint directory. Local Agent (tool calling) and inline completion
(FIM) stay GGUF-only until their phases land, advertised through existing
capability flags.

## 2. Supported artifact

A Safetensors checkpoint is a **flat directory** containing:

- `config.json` (architecture, context, quantization — see §6),
- one or more `*.safetensors` files (case-insensitive match).

Rules (verified in `src/models/safetensorsDirectory.ts`):

- Detection is shallow: `config.json` present plus any `*.safetensors`.
  Transformers stays authoritative for whether it actually loads.
- Collection is flat and non-recursive. A nested directory is not part of
  the checkpoint.
- Each `.safetensors` file starts with an 8-byte little-endian header length
  followed by a JSON header naming every tensor with dtype, shape, and byte
  offsets. Only headers are ever read on the extension side; payloads are
  never touched.
- Sharding: multiple `*.safetensors` files are one checkpoint. The Python
  inspector additionally notes `model.safetensors.index.json` presence and
  multi-file layout as `sharded` for display; sharded verification is per-file
  (path + size + header), not via the index.
- An unreadable header (short read, non-positive or oversize length, parse
  failure) never blocks install. The file still contributes path + size to
  identity, with reduced precision plus an optional warning.

Header cap rationale: the header length is an untrusted u64 read from the
first 8 bytes of the file, and both sides allocate that many bytes before
validating. The cap bounds the allocation, so a corrupt or hostile file
claiming exabytes cannot OOM the extension host or worker. The cap is
necessary structure; only its value is judgment (legitimate headers are
KBs to low MBs, large vocabs tens of MB — both caps sit far above that).
Known defect to fix: the two sides disagree. TS rejects above
`MAX_SAFETENSORS_HEADER_BYTES = 100 MiB`; Python `inspector.py` rejects above
`MAX_HEADER_BYTES = 128 MiB`. Unify on one value — **decided and aligned: 128 MiB** both sides, with a
shared cross-side fixture test.

Out of scope for MVP: downloading Safetensors from Hugging Face
(`src/models/modelSources.ts:70` filters `.gguf` only and throws when none
match), copying checkpoints into extension storage, and any GGUF↔Safetensors
conversion.

## 3. Architecture

Two engines, one manager, one inference contract.

- `src/models/`: GGUF file path (`ggufMetadata.ts`) and Safetensors
  directory path (`safetensorsDirectory.ts` + `modelIdentity.ts`).
  `src/domain.ts` tags `InstalledModel` with `format` and
  `runtime: 'llama-cpp' | 'transformers'`, plus `managed: false` for
  in-place checkpoints.
- `resources/runtime/`: Python worker (`runtime.py`, `worker.py`,
  `protocol.py`, `models.py`, `errors.py`, `inspector.py`).
  Pinned baseline in `resources/runtime/requirements.txt`:
  `torch==2.14.0`, `transformers==5.17.0`, `accelerate==1.15.0`,
  `safetensors==0.8.0`. Optimisation backends (bitsandbytes, torchao, AWQ,
  GPTQ, flash-attn, optimum) are added only when a real model requires them.
- `src/worker/runtimeSession.ts`: the seam. A session is a started process +
  the client that talks to it + how it ends. `WorkerManager` holds **one**
  session at a time and dispatches on `runtimeForModel(model)`. Starting
  either runtime evicts the other, so two large models are never resident.
  No second scheduler: the single `InferenceScheduler` already serializes
  across both runtimes.
- `src/worker/inferenceClient.ts`: `InferenceClient`, what the extension asks
  of a loaded model whichever worker holds it. `llamaClient.ts` implements it;
  `transformersClient.ts` adapts the Python runtime to it.
- Naming: `WorkerBackend` (in `workerManifest.ts`) means metal/sycl/cpu for
  llama.cpp — a hardware axis. `ModelRuntime` means which engine holds the
  weights. Do not conflate them.
- Lifecycle asymmetry is contained in the session: llama-server stops by
  signal, the Python worker by closed stdin; unexpected-exit meaning and
  crash recovery stay in one `handleExit`. `exitNotifier` must fire
  immediately for handlers registered after death, or a crash between health
  check and watch leaves a phantom-live session.

## 4. Worker contract (protocol version 1)

`PROTOCOL_VERSION = 1` (`resources/runtime/runtime/models.py`) must equal
`SUPPORTED_PROTOCOL_VERSION = 1` (`src/worker/pythonWorkerTypes.ts`);
mismatch is a hard error telling the user to reinstall.

Transport: stdio JSONL. stdout is protocol-only; stderr is logs
(`protocol.py`). Every request carries an id for correlation; `runtime.info`
is the version/compat handshake.

Methods (verified in `resources/runtime/runtime/worker.py`):

| Method | Purpose |
|---|---|
| `runtime.info` | protocol version, worker state, diagnosability |
| `model.inspect` | pre-load inspection without weights (§6) |
| `model.load` / `model.unload` / `model.info` | lifecycle |
| `model.tokenize` | chat-template token counting (also backs `countChatInputTokens`) |
| `generate.chat` | streaming chat generation |
| `generate.complete` | plain completion (kept for symmetry; FIM is separate, §8) |
| `--probe` / `--inspect` / `--worker` | throwaway interpreter check, offline inspect, serve loop |

Streaming: token notifications stream, then the ordinary `{id, ok, result}`
response signals completion. There is deliberately **no** separate
`generation.complete` notification — two completion signals invite client /
worker disagreement. `GenerationResult` carries text plus the prompt token
count computed at encode time (no second round trip).

Cancellation names its `requestId`. The worker rejects a cancel for a request
not in flight, so a cancel racing a just-finished generation cannot kill its
successor.

Capabilities are declared, not inferred (`supports: { infill,
constrainedDecoding }` on `InferenceClient`; `supportsInfill()` requires both
declaration and implementation). The Transformers client declares neither
today. A chat request carrying tools is refused with a reason before reaching
the worker.

## 5. Error contract

Owned by `resources/runtime/runtime/errors.py`. Codes observed:

`unsupported_architecture`, `unsupported_quantization`,
`unsupported_grammar`, `grammar_compile_failed`, `custom_code_required`
(`auto_map` present), `missing_tokenizer`, `missing_weights`,
`device_unavailable`, `insufficient_memory`, `context_overflow`,
`model_load_failed`, `model_not_loaded`, `chat_not_supported`,
`generation_failed`, `generation_cancelled`, `generation_busy`,
`invalid_request`, `unknown_method`, `internal_error`.

`unsupported_grammar` is already a live refusal: `runtime.py` raises
`GrammarUnsupportedError` when tools/grammar are requested without a backend,
and `transformersClient.ts` maps it. `grammar_compile_failed` is defined in
`errors.py` for the future backend. Neither code path is reachable for
successful generation until §8 ships.

## 6. Inspection gate (`model.inspect`)

Inspection runs **before** load, without weights, and must keep working when
the ML stack is broken or unprovisioned. The Python inspector is standard
library only.

From `config.json` + headers it reports: architecture (`architectures[0]` /
`model_type`), trained context length (including `text_config` nesting),
quantization (`quantization_config`), weight bytes + dominant dtype summed
over shards, `sharded`, `customCodeRequired` (`auto_map` in config), and a
per-file list. The TS mirror (`readSafetensorsCheckpoint`) reports the same
subset so the UI works before Python is provisioned or where it is broken.

`customCodeRequired: true` refuses to load (no `trust_remote_code` path in
MVP). Quantized checkpoints (FP8, NVFP4, AWQ, GPTQ) report
`unsupported_quantization` up front where the kernel is CUDA-only (MPS/XPU
would fail mid-load or dequantize to bf16, doubling memory).

## 7. Memory and offload policy

Safetensors checkpoints are typically bf16: 27–30B ≈ 54–60 GB of weights.
`device_map="auto"` will not refuse — Accelerate offloads to CPU then disk
and yields a model that loads and generates unusably slowly. llama.cpp does
the opposite (shrink context or refuse).

Policy:

- `model.inspect` estimates weight bytes from headers before load and
  compares against device and host memory.
- Policy object is `allowCpuOffload`, optional on the TS side
  (`pythonWorkerTypes.ts`). **Decided and aligned: defaults to `false`.**
  The caller may opt in to CPU offload explicitly; silent offload is never
  the default.
- The gate is a tolerance band, not a binary. Estimates from headers carry
  error (framework overhead, fragmentation, resident-vs-weight deltas), so a
  checkpoint estimated within **5% over** calculated capacity is allowed
  (surfaced as a warning where the UI can show it); beyond 5% refuses with
  `insufficient_memory`. Exceeding host memory such that disk offload is
  unavoidable refuses regardless of the CPU-offload flag.
- On unified-memory hardware (Apple Silicon) the pools coincide, so CPU
  offload buys nothing — the host-memory check is the one that fires.
- Covered by a memory-gate test (e.g. 60 GB checkpoint on a 36 GB machine
  refuses).

## 8. Explicitly deferred: tool calling and FIM

Tool calling (Local Agent): Transformers `generate()` has no equivalent of
llama.cpp's `response_format` → GBNF constrained sampling, which per project
notes is the only working tool path on reference models. Fix is an upstream
grammar library (recommend `xgrammar` as a `LogitsProcessor` behind the
existing `response_format` contract so `toolProtocol.ts` is untouched), added
to the required baseline — not optional. Until then Safetensors models are
Chat-only: profile reports `supportsTools/supportsToolCalls: false`, the
agent will not select them, and tool-bearing chats are refused with advice.

FIM (inline completion): no Transformers analogue; FIM tokens live outside
the chat template (`tokenizer_config.json` / `generation_config.json`), so
template delegation does not reach them. MVP advertises
`fillInMiddle: 'unsupported'` for this runtime. `canRunInline` runs before
any client exists, so the guard lives in inline completion +
`validateFillInMiddle`, not in client construction. A future phase may
assemble prefix/suffix prompts from tokenizer config as a narrow exception.

## 9. Identity, verification, migration, ownership

- GGUF identity is SHA-256 over file bytes. Checkpoint identity is a manifest
  digest over per-file path + size + Safetensors header (tensor names,
  dtypes, shapes, offsets). Rationale: re-hashing tens of GB on every
  activation is not viable; header coverage keeps activation proportional to
  file count. Accepted reduction: a payload value flip preserving shape,
  dtype, and size is **not** detected. Document it; do not silently
  strengthen later without measuring activation cost.
- Change detection is cheaper still: one stat per file (size + mtime); headers
  re-read only after movement. mtimes are excluded from the digest so copying
  a checkpoint to a faster disk does not invalidate it; the digest still
  catches re-quantize, re-shard, truncate, and substitution.
- Ownership: checkpoints register **in place** (`managed: false`); `remove`
  unregisters and leaves files alone (a test asserts weights survive — deleting
  a user's 60 GB directory on list-removal is unacceptable). GGUF models
  remain copied into extension storage.
- Migration: `localLlm.installedModels.v1` keeps its key. Legacy records with
  no `format` are single GGUF files on llama.cpp — label on load, do not
  discard. Checkpoints with no fingerprints are backfilled, not invalidated.
  Tests assert legacy capabilities and runtime profiles survive.

## 10. Provisioning (largest open risk)

The worker stack cannot be assumed installed, and a VSIX cannot carry
the interpreter + torch. `localLlm.pythonPath`, when set, always wins and
is used directly. When empty, `ensureEnvironment()` provisions per the
manifest design below; on machines without a manifest entry (or until the
release manifest is generated) that path fails by naming the setting.
Do not auto-use `python3` from `PATH` — installing gigabytes into a system
interpreter on the user's behalf is unacceptable, and the dev machine itself
demonstrates the failure (torch import abort via duplicate OpenMP runtime,
no Transformers present).

- Probe with `python -m runtime.worker --probe` in a throwaway process before
  spawning a session. Rationale: torch can abort rather than raise, and a
  worker cannot report its own abort — a dead probe is cheaper than a dead
  session. Lazy detection (resolve on first use) keeps `model.inspect` usable
  while provisioning is pending.
- Download-on-first-use into global storage, hash-pinned, diagnosable via
  `runtime.info`. Trade-off: small VSIX, but a network + corporate-proxy
  failure mode. Keep the extension's existing per-byte integrity discipline
  rather than trusting pip resolution.
- Until provisioning ships, everything in this scope must degrade to a named
  error, never a hang.

### Env manifest design (in-scope targets: macOS arm64, Windows x64)

Layout under `globalStorageUri`:

```text
python-env/
  <target>/<flavor>/          # venv root, e.g. win32-x64/cuda
  <target>/<flavor>.json      # installed manifest (what + when)
```

A release-time generated manifest (checked in, e.g.
`resources/runtime/env-manifest.json`) pins per target:

```text
target:
  interpreter: { url, sha256, exe }   # python-build-standalone build
  wheels: [{ name, url, sha256 }]      # full closure, no PyPI at install time
  packs: { cuda: {...}, ipex: {...} } # Windows-only optional packs
```

Targets and flavors (see platform matrix for the full map):

- `darwin-arm64`: one flavor. Standalone CPython + PyPI CPU torch (MPS
  included), transformers/accelerate/safetensors per `requirements.txt`.
  ~0.5–1 GB installed.
- `win32-x64/cpu`: standalone CPython + PyPI CPU torch. ~0.5–1 GB.
- `win32-x64/cuda`: cpu base + CUDA torch wheel set (torch + `nvidia-*`
  closure). ~3–5 GB. Selected when `nvidia-smi` succeeds.
- `win32-x64/xpu`: cpu base with stock torch *replaced* by torch's own
  `+xpu` build from the PyTorch XPU channel (in-tree `torch.xpu`, no IPEX
  needed; required: Intel Arc Pro 140T work laptop). A pack replaces rather
  than adds: one torch per env, and pack wheels win incidental collisions.
  Selected when Intel graphics is detected; probe must confirm `torch.xpu`
  sees the GPU or the env is rejected as broken. The XPU channel lags stock
  torch slightly (2.12 vs 2.14 at manifest time) — see the platform matrix.
  Regenerate with `--xpu <torch-version>` as new builds appear.
  Flavor override setting `localLlm.pythonEnvFlavor: auto | cpu | cuda |
  xpu` always wins over detection.

### Windows GPU detection (decided)

No in-repo precedent (backend selection is by setting, not probing). New
`detectWindowsGpu()` in the environment module, result cached per session:

1. `nvidia-smi -L` exit 0 → `cuda`.
2. Else PowerShell `Get-CimInstance Win32_VideoController`, names matching
   `/intel.*arc|arc.*intel/i` (covers discrete Arc and 140T-class iGPUs
   reporting as e.g. `Intel(R) Arc(TM) 140T Graphics`) → `ipex`.
3. Else (or any spawn failure) → `cpu`. Detection never throws; failure
   degrades to CPU, and the setting always wins.

### Proxy posture (decided)

Env downloads reuse `downloadModel`, inheriting exactly the existing model
path's behavior: plain `fetch` honoring `HTTP(S)_PROXY` env, with no
VS Code `http.proxy`-setting integration. Full setting-aware proxy support
is backlog shared with model downloads, not a provisioning blocker; failure
degrades to naming `localLlm.pythonPath`.

Install flow (`ensureEnvironment()`, before first spawn):

1. Read installed manifest; match against release manifest (target, flavor,
   wheel SHAs). Match → use.
2. Else VS Code progress download: interpreter archive + each wheel via the
   existing `downloadModel` (resume, per-file SHA-256, single-flight).
3. Extract interpreter, create venv, `pip install --no-index` the local
   wheels (hermetic: PyPI is never consulted at install time).
4. `--probe` in a throwaway process; probe failure deletes nothing but
   refuses to write the installed manifest, so next use retries.
5. Write installed manifest. Env replacement is by new directory;
   stale dirs are removed lazily on next successful provision.

Fallbacks: `localLlm.pythonPath` remains the permanent manual escape hatch
(proxy, air-gap, custom CUDA). Proxy behavior of `fetch` download is
verified during implementation; failure degrades to naming the setting.

## 11. UI and acquisition gaps

- Nothing invokes `importSafetensorsDirectory` (defined in
  `modelManager.ts:160`, no non-test caller); the model picker shows no
  format/runtime; `quantization` / `customCodeRequired` are recorded but never
  surfaced pre-load. Partial progress exists: status readout names the engine
  via `runtimeDisplayName` (`modelCommands.ts`). Still GGUF-assumed: the
  remove dialog asks about the "GGUF file" unconditionally. Estimate ~2–3 days.
- HF acquisition is GGUF-only (see §2). Designing Safetensors selection
  (which files, what size, revision pinning) is unscheduled scope.

## 12. Acceptance (replaces v1 §34)

MVP is done when, on macOS arm64 + Windows x64 (Windows ARM64 unsupported —
no PyPI torch; Linux deferred per platform matrix; state both in requirements):

1. Local directory installs as `transformers` runtime, detected shallowly,
   headers-only.
2. `model.inspect` reports arch / context / quant / weight bytes / dtype /
   sharded / custom-code without loading weights, including when Python is
   broken (TS mirror).
3. Memory gate: estimate within 5% over capacity loads (with warning);
   beyond 5% refuses pre-load with `insufficient_memory`. CPU offload only
   when explicitly opted in (default `false`). Quantized-unsupported and
   custom-code checkpoints refuse with their codes.
4. Chat streams tokens, `GenerationResult` carries prompt count, cancel of an
   in-flight id stops it and a stale cancel is rejected.
5. Unload + reload, unexpected-exit recovery (no phantom-live session), and
   one-session eviction across runtimes.
6. Digest survives a directory move; shard substitution / truncation
   invalidates; `remove` leaves in-place files on disk.
7. Legacy GGUF records migrate with capabilities and runtime intact.
8. Tool-bearing chat on a Safetensors model refuses with runtime-specific
   advice; Local Agent never selects it; FIM reports `unsupported` without
   loading.
9. Protocol mismatch fails with reinstall guidance; probe failure fails by
   naming `localLlm.pythonPath`.
10. `npm test` (node + python), typecheck, and build green on a clean
    checkout.

## 13. Phases with done criteria [inference, recalibrated]

| Phase | Done when | Est. |
|---|---|---|
| 1. Provisioning + pinning | Clean machine installs pinned env into global storage, hash-verified, `runtime.info` diagnosable | 2–3 wks |
| 2. Worker protocol + lifecycle | Table in §4 + error codes in §5 over stdio JSONL, crash recovery tested | done (verify) |
| 3. Streaming + cancel via extension | §12 items 4–5 through `WorkerManager`, not just worker-direct tests | ~1 wk |
| 4. Dual-runtime integration | `runtimeForModel` routing, one-session eviction, refusals with advice | done (verify) |
| 5. Inspect + capabilities + memory gate | §12 items 2–3, unified header cap, host+device checks | ~1 wk + defect fix |
| 6. HF + UI acquisition | HF Safetensors select/download + invoke/surface import | 1–2 wks (new vs v1) |
| 7. Tool calling (grammar) | `xgrammar` baseline, agent selects Safetensors, error codes live | 1.5–2 wks, post-MVP |
| 8. FIM decision | Permanent `unsupported` or tokenizer-config assembly | 0.5–1 wk, post-MVP |
| 9. Test hardening | Per-module tests, cross-side header fixture, red-on-clean-checkout ban | 1–1.5 wks |

MVP (1–5 + partial 9): ~7–9 wks from v1 baseline, minus verified-done items 2/4
pending a green clean-checkout run. Full scope (+6–9): ~11–15 wks.
Estimates are judgment, not measurement — re-estimate after provisioning
spikes on all three OS targets.

## 14. Non-goals

No conversion as the primary path (llama.cpp cannot load Safetensors;
convert-then-run keeps the Python dependency while adding per-model time,
disk, fidelity loss, and architecture lag — rejected). No curated model list:
any Transformers-supported checkpoint must be loadable, not only approved
ones. No `trust_remote_code`. No second scheduler or per-runtime queues. No
bundled interpreter in the VSIX. No bundled optimisation backends until a real
model needs them.

## 15. Open questions

1. Decided and aligned: header cap 128 MiB both sides, shared fixture test each side.
2. HF Safetensors file-selection UX (full dir vs filtered subset, revision pin)?
3. Windows ARM64: CPU-only via unofficial torch vs explicit unsupported?
4. Permanent FIM `unsupported` vs tokenizer-config assembly?
5. Strengthen digest toward payload sampling later, and at what activation
   budget?
6. Decided and aligned: `allowCpuOffload` defaults to `false`; 5% tolerance
   band with warning in the `runtime.py` gate; usable = 90% of pool.
