# Safetensors support — scope against the Universal Local Safetensors LLM Runtime spec

Status: scope only. Date: 2026-09-16.
Scopes the Transformers-worker design into this extension as it exists today.

## Bottom line

**MVP (chat + streaming + cancel): 7–9 weeks. Full parity with the GGUF path:
11–15 weeks.**

The spec is sound and the delegate-to-upstream rule is the right one. The cost is
not in the runtime — the prototype already covers most of §35 CURRENT. It is in
three things the spec does not cover, because the spec describes a runtime and
this is a shipping extension:

1. Provisioning Python inside a VSIX (§4 assumes the stack is installed).
2. Tool calling — absent from all 36 sections, and it is this extension's
   headline feature.
3. Running two runtimes side by side in a `WorkerManager` built for exactly one.

Everything else in §16–§36 is ordinary work with a clear upstream answer.

## Verified facts

Checked 2026-09-16 against PyPI and llama.cpp at the pinned commit
`60eeeb6082c1126bb8bc72902c83123cd056811b` (b10472).

| Fact | Result |
|---|---|
| `llama-server` loading safetensors | Impossible. Zero `safetensor` references in any `.c/.cpp/.h/.hpp` at the pin. Conversion is the only llama.cpp path — which is what this design correctly bypasses. |
| torch wheel platforms | `macosx`, `manylinux`, `win_amd64`. **No `win_arm64` on PyPI.** |
| torch wheel size | 121.4 MB macOS arm64, 118.4 MB win_amd64 |
| `transformers` 5.17.0, `accelerate` 1.15.0 | Pure Python, small |
| `safetensors` 0.8.0 | Native wheels, small |
| Constrained decoding libs | `xgrammar` 0.2.7, `outlines` 1.3.3, `lm-format-enforcer` 0.11.3 — all current |

**§2 correction:** Windows ARM64 has no PyPI torch. The spec's hedge is accurate,
but in practice that target is CPU-only via unofficial builds, or unsupported.
Worth stating plainly in the extension's requirements rather than discovering it
at install time.

## Gap 1 — Tool calling (not in the spec)

This is the largest single finding.

`localLanguageModelProvider.ts` and the bundled Local Agent depend on
schema-constrained decoding. `llamaClient.ts:493` sends `response_format` to
llama.cpp, which compiles it to a GBNF grammar and constrains sampling. Per this
project's own operating notes, on Qwen2.5-Coder the schema-constrained fallback
is the **only** working tool path — the native pass produces nothing.

Transformers `generate()` has no equivalent. Without one, Local Agent does not
work on safetensors models at all.

The fix is consistent with §32 — it is an upstream library, not a custom
implementation — but it is a dependency §4 does not list:

```text
xgrammar          # fastest, used by vLLM/SGLang; LogitsProcessor integration
lm-format-enforcer # simplest to wire, broadest model support
outlines          # richest API, heaviest dependency
```

Recommend `xgrammar` as a `LogitsProcessor`, exposed as a new
`generate.chat` parameter mirroring the existing `response_format` contract, so
`toolProtocol.ts` and `localAgentToolChoice.ts` need no changes. Add to §4 as a
required baseline dependency, not an optional one.

Add to the §15 error contract: `unsupported_grammar`, `grammar_compile_failed`.

## Gap 2 — Fill-in-the-middle

`completion/localInlineCompletionProvider.ts` calls llama.cpp's `/infill`
endpoint, which knows each model family's FIM tokens. Transformers has no
analogue, and FIM tokens do not live in the chat template, so §9.3's
delegate-to-the-template rule does not reach them.

Two honest options:

- Read FIM special tokens from `tokenizer_config.json` / `generation_config.json`
  where present and assemble the prefix/suffix prompt in `runtime.py`. This is a
  narrow, defensible exception to §32.
- **Recommended for MVP:** scope inline completion to GGUF models only. Advertise
  `fillInMiddle: 'unsupported'` for safetensors models through the existing
  `ModelCapabilities` in `domain.ts`. The plumbing for that already exists.

## Gap 3 — Two runtimes in one WorkerManager

`worker/workerManager.ts` (772 lines) assumes one llama-server child, HTTP on an
allocated loopback port, health-polled, with `WorkerState` in `domain.ts` keyed
to a single `modelId` and `port`. The Python worker is a different shape: stdio,
JSONL, no port, different lifecycle (§24).

Required changes:

| File | Change |
|---|---|
| `src/domain.ts` | Tag `InstalledModel` with `runtime: 'llama-cpp' \| 'transformers'`; `WorkerState` gains the runtime kind. Persisted-schema migration on `localLlm.installedModels.v1`. |
| `src/worker/workerManager.ts` | Extract a `WorkerBackend` interface over start/stop/health; llama.cpp and Python become two implementations. |
| `src/worker/inferenceScheduler.ts` | Already serializes work; must now serialize across both runtimes so two 30B models never load at once. |
| `src/worker/pythonWorkerClient.ts` *(new)* | JSONL transport, request-ID correlation, §18 streaming notifications, §25 crash recovery. |
| `src/provider/localLanguageModelProvider.ts` | Route by `model.runtime`. `messageAdapter.ts` output already matches the spec's `messages` shape. |

The `ChatStreamEvent` union in `domain.ts` maps cleanly onto §18's
`generation.token` / `generation.complete`. No provider-layer redesign needed.

## Gap 4 — Python provisioning and integrity

The spec assumes the stack is installed. A VSIX cannot.

This extension currently SHA-256-pins every worker byte
(`resources/workers/manifest.json`, verified at launch by
`worker/workerIntegrity.ts`). A pip-resolved environment has no equivalent
guarantee, and the README's opening promise — "without installing Ollama,
Python, Docker, or llama.cpp separately" — is broken by this feature regardless
of how it is delivered.

Options:

- **Download on first use** into global storage, hash-pinned per wheel. Fits the
  existing integrity discipline, keeps the VSIX small, introduces a network
  dependency and a corporate-proxy failure mode.
- **Bundle** a standalone interpreter plus wheels. ~400–600 MB installed per
  platform, doubling VSIX size across two targets.

Recommend download-on-first-use with a pinned lockfile and per-file SHA-256,
surfaced through `runtime.info` (§28) so a mismatch is diagnosable.

## Gap 5 — Memory, and the honest performance story

§22 identifies this correctly; here is the concrete consequence.

Safetensors are typically bf16. A 27–30B model is ~54–60 GB of weights. It will
not fit in most Apple Silicon unified memory. `device_map="auto"` will not
refuse — Accelerate will offload to CPU and then disk and produce a model that
loads successfully and generates at unusable speed.

llama.cpp does the opposite: `--fit` reduces the context window or refuses,
and `memoryFit.ts` / `modelCapacity.ts` are built on parsing that decision.
There is no upstream equivalent to port.

So `model.inspect` (§16 P3) is not a nice-to-have — it is the gate that keeps
this feature from feeling broken. It must estimate weight bytes from the
safetensors headers before load, compare against available device memory, and
apply §23's `allow_disk_offload=False` default.

**Product consequence worth stating explicitly:** safetensors is the *universal,
day-one* path — run any architecture Transformers supports, including models
with no GGUF conversion. GGUF remains the *fast, daily-driver* path. Both stay.
Positioning safetensors as a performance path would set up users to be
disappointed.

### Pre-quantized safetensors caveat

§19's delegate-to-Transformers rule is right, with a platform catch: FP8, NVFP4,
AWQ, and GPTQ kernels are largely CUDA-only. On MPS and XPU those checkpoints
either fail to load or must dequantize to bf16, which *doubles* the memory a user
was trying to save. `model.inspect` should read `quantization_config` and report
`unsupported_quantization` up front rather than failing mid-load.

## Effort

| Phase | Work | Est. | MVP? |
|---|---|---|---|
| 1 | Python provisioning + integrity pinning | 2–3 wks | yes |
| 2 | Worker protocol, lifecycle, crash recovery (§24–26) | 1 wk | yes |
| 3 | Streaming + cancellation (§35 P1, P2) | 1–1.5 wks | yes |
| 4 | Dual-runtime integration (Gap 3) | 1.5–2 wks | yes |
| 5 | `model.inspect`, capabilities, context validation (§16, §17, §29) | 1 wk | yes |
| 6 | Tool calling via xgrammar (Gap 1) | 1.5–2 wks | no |
| 7 | FIM (Gap 2) — or advertise unsupported | 0.5–1 wk | no |
| 8 | Memory estimation + offload policy (§22–23) | 1 wk | no |
| 9 | Tests — this repo keeps a `.test.ts` per module | 1.5–2 wks | partial |
| | **MVP (1–5, partial 9)** | **7–9 wks** | |
| | **Full parity (1–9)** | **11–15 wks** | |

MVP delivers §34 criteria 1–15 except tool calling and FIM: load any
Transformers-supported safetensors model, stream chat, cancel, unload, recover.
Local Agent and inline completion stay GGUF-only until phases 6–7 land, which the
existing `ModelCapabilities` flags already express without UI work.

## Recommendation

Build it as specified. Three amendments to the spec:

1. **§4** — add a constrained-decoding library (`xgrammar`) to the required
   baseline. It is not optional for this extension; Local Agent depends on it.
2. **§2** — state that Windows ARM64 has no PyPI torch, so that target is CPU-only
   or unsupported.
3. **Add a section on environment provisioning.** It is the single largest risk
   and the only part with no upstream answer to delegate to.

The §36 design rule holds and is the reason to do this: new architecture appears,
upgrade Transformers, it loads. That is worth 7–9 weeks.

---

# Implementation status — 2026-09-16

Phase 2 of the table above (worker protocol, lifecycle, crash recovery) is built,
along with `model.inspect` from phase 5. Nothing is wired into the extension yet;
`WorkerManager` is untouched.

## Built

| Path | Purpose |
|---|---|
| `resources/runtime/runtime/errors.py` | The §15 error contract, with `unsupported_grammar` added |
| `resources/runtime/runtime/models.py` | Wire dataclasses, `PROTOCOL_VERSION`, §24 worker states |
| `resources/runtime/runtime/inspector.py` | §16 P3 pre-load inspection — **standard library only** |
| `resources/runtime/runtime/runtime.py` | `LocalLLM` lifecycle, generation, streaming, cancellation |
| `resources/runtime/runtime/protocol.py` | JSONL framing; stdout protocol-only, stderr logs (§26) |
| `resources/runtime/runtime/worker.py` | Reader loop, dispatch, `--worker` / `--inspect` / `--probe` |
| `resources/runtime/requirements.txt` | Pinned baseline (§4) |
| `src/worker/pythonWorkerTypes.ts` | Wire contracts mirroring the dataclasses |
| `src/worker/pythonWorkerClient.ts` | Correlation, streaming, crash recovery; transport injected |
| `src/worker/pythonWorkerProcess.ts` | Real spawn transport |

Also `resources/runtime/tests/test_runtime.py`, run by the new `test:python`
script. 29 tests added: 13 TypeScript unit, 3 integration against a spawned
worker, 13 Python. Node suite is 175 tests / 174 passing; Python is 13/13;
`npm run build` succeeds.

**`npm test` now chains `test:node && test:python`.** Because
`src/worker/toolProtocol.test.ts` fails on a clean checkout of HEAD, the chain
stops before the Python suite. Run `npm run test:python` directly until that
pre-existing failure is fixed.

## Deviations from the spec, and why

**§18 `generation.complete` is not emitted.** Token notifications stream, then
the ordinary `{id, ok, result}` response signals completion. Two completion
signals for one request is redundant framing that invites the client and worker
to disagree about which is authoritative.

**Runtime detection is lazy, not constructed eagerly.** Discovered while
testing: importing torch can *abort the interpreter* rather than raise — a
duplicate OpenMP runtime is the common cause, and it reproduces on this
development machine today. Eager detection therefore killed the worker before it
could answer anything. Detection now resolves on first use, so `model.inspect`
keeps working when the ML stack is broken or still being provisioned.

**A `--probe` mode was added.** Because that abort cannot be caught in-process,
the extension should run `python -m runtime.worker --probe` as a throwaway
process before trusting a worker. A torch that kills its process then costs a
probe rather than the session.

## Corrected after review

**The memory gate was inverted.** `_assert_fits` originally raised only when
`allowCpuOffload` was false, which is off by default — so a 60 GB checkpoint on
a 36 GB machine passed the very gate meant to stop it. It now checks in two
steps, because the two policy flags gate different fallbacks: exceeding device
memory is acceptable only when CPU offload is allowed, and exceeding *host*
memory means disk offload is unavoidable regardless. On unified-memory hardware
the two pools are the same, so CPU offload buys nothing and the second check is
the one that fires. Covered by `test_rejects_a_model_larger_than_host_memory`.

**Cancel now names its target.** It previously sent no `requestId`, so a cancel
racing a generation that had just finished would have stopped whichever one
started next. Both sides now follow §18, and the worker rejects a cancel naming
a request that is not in flight.

**`generate.complete` sits one letter from §18's `generation.complete`.** The
request name is kept for spec fidelity; the dispatch site now states explicitly
that the similarly-named notification is deliberately absent.

**`from_params` read `cls.__slots__`.** That happens to hold the field names
under `@dataclass(slots=True)`, but it is an implementation detail;
`dataclasses.fields()` is the contract.

## What this validated

Provisioning an isolated environment is not optional. The development machine
here has a torch whose import aborts and no Transformers at all — exactly the
condition a user's system Python will present. Relying on `python3` from PATH
would fail for reasons the extension cannot diagnose or repair.

## Next

Phase 3 remains streaming and cancellation *wired through the extension*; the
runtime side of both is built and the protocol carries them, but nothing calls
it yet. Then phase 1 (provisioning) and phase 4 (dual-runtime integration).

## Packaging

Verified with `vsce ls` that the VSIX carries exactly the eight runtime files and
excludes `__pycache__` and the test package. `npm run package` itself currently
fails, but only at the `vscode:prepublish` typecheck, on the uncommitted
`finalResponse.ts` edit below.

## Unrelated, noticed in passing

`src/worker/finalResponse.ts` has an uncommitted edit importing `'../domain'`
without the `.js` extension, which fails `npm run typecheck` under NodeNext
resolution. `src/worker/toolProtocol.test.ts` fails on a clean checkout of HEAD.
Neither is touched by this work.

---

# Implementation status — increment 2

Phase 5 (domain, inspection, capabilities) is built. A Safetensors checkpoint can
now be represented, installed and tracked. It still cannot be *run*: nothing
routes to the Python worker yet.

## Built

| Path | Purpose |
|---|---|
| `src/models/modelIdentity.ts` | Manifest digest, directory change detection, format→runtime, ownership |
| `src/models/safetensorsDirectory.ts` | The filesystem half: header reads, fingerprints, checkpoint description |
| `src/domain.ts` | `format`, `runtime`, `files`, `quantization`, `customCodeRequired`, `managed` |
| `src/models/modelRegistry.ts` | Migration and per-format verification, split out of `initialize` |
| `src/models/modelManager.ts` | `importSafetensorsDirectory`, and an ownership guard on `remove` |

Node suite 210 tests / 209 passing; Python 13/13; typecheck and build clean.

## Two decisions worth knowing about

**A checkpoint is registered where it already lives.** GGUF models are copied
into extension storage; Safetensors checkpoints are not. Copying tens of
gigabytes to duplicate what the user already has on disk cannot be justified, and
these models are routinely kept on an external volume.

The consequence is recorded as `managed: false`, and `remove` honours it: an
in-place checkpoint is unregistered and its files are left alone. A test asserts
the weights survive removal, because the failure mode — deleting a user's 60 GB
checkpoint because they took it out of a list — is unacceptable.

**Identity is a manifest digest, not a content hash.** A GGUF model is identified
by SHA-256 over its bytes. Re-hashing a directory that size on every activation
is not viable, so a checkpoint is identified by a digest over each file's path,
size and *Safetensors header* — which names every tensor with its dtype, shape
and offsets.

This detects a re-quantized, re-sharded, truncated or substituted checkpoint. It
does **not** detect a value flipped inside a tensor payload that leaves shape,
dtype and file size intact. That is a real reduction against the guarantee GGUF
models get, and it buys an activation that stays proportional to file count
rather than to bytes on disk.

Change *detection* is separate and cheaper still: one stat per file comparing
size and mtime. Headers are re-read only once something has actually moved.
Modification times are excluded from the digest deliberately, so that copying a
checkpoint does not discard capabilities it was already verified for.

## Migration

`localLlm.installedModels.v1` keeps its key. Records written before this work
carry no `format`, and every one of them is a single GGUF file served by
llama.cpp, so they are labelled on load rather than discarded. A checkpoint with
no recorded fingerprints is backfilled rather than invalidated, for the same
reason. Both are covered by tests that assert a legacy record keeps its verified
capabilities and its runtime profile across the migration.

## Next

Phase 4, the dual-runtime integration: `WorkerManager` still assumes one
llama-server child on a loopback port, and the provider does not yet route on
`model.runtime`. After that, phase 1 provisioning, which is what makes any of it
run on a machine other than a developer's.

---

# Implementation status — increment 3 (phase 4, in progress)

The runtime seam is built: both workers now satisfy one contract. What remains
in phase 4 is the plumbing behind it — `WorkerManager` still spawns only
llama-server, and the provider still does not route on `model.runtime`.

## Built

| Path | Purpose |
|---|---|
| `src/worker/inferenceClient.ts` | `InferenceClient` — what the extension asks of a loaded model, whichever worker holds it |
| `src/worker/transformersClient.ts` | Adapts the Python runtime to that contract |
| `src/worker/llamaClient.ts` | Now declares `implements InferenceClient` and its capabilities |
| `resources/runtime/runtime/runtime.py` | `count_tokens`, and `generate` returns a `GenerationResult` |
| `resources/runtime/runtime/worker.py` | `model.tokenize` |

Node suite 226 tests / 225 passing; Python 13/13; typecheck and build clean.

## Capabilities are declared, not inferred

`InferenceClient` carries a `supports` record rather than letting callers test
for a method. Inline completion asks `supportsInfill(client)`, which requires
*both* the declaration and the implementation — a client claiming a capability
it does not implement is rejected rather than called and crashed. llama.cpp
declares `infill` and `constrainedDecoding`; the Transformers client declares
neither, for now.

That is what makes the missing pieces safe to ship before they exist. A
Safetensors model reports `supportsTools: false` through the ordinary profile,
so the Local Agent will not select it, and a chat request carrying tools is
refused with a reason before anything reaches the worker rather than returning
prose the agent cannot parse.

## One thing that got faster

The prompt token count now travels with the generation response. The first cut
asked the worker for it separately, which cost a round trip per reply to report
a number the worker had already computed at encode time. `generate` returns a
`GenerationResult` carrying both.

Found because the first version of the adapter test hung: it answered the
token-count request before that request had been sent. The fix removed the
request rather than the race.

# Implementation status — increment 4 (phase 4, complete)

| Path | Purpose |
|---|---|
| `src/worker/runtimeSession.ts` | The seam: `RuntimeSession`, `runtimeForModel`, `exitNotifier` |
| `src/worker/transformersBackend.ts` | Verifies, probes, spawns and loads the Python runtime |
| `src/worker/workerManager.ts` | Routes on `model.runtime`; lifecycle is now session-shaped |
| `src/provider/localLanguageModelProvider.ts` | Runtime-specific advice on the two refusals |
| `src/completion/localInlineCompletionProvider.ts`, `src/ui/modelCommands.ts` | Branch on `supportsInfill` |

Node suite 236/236; Python 13/13; typecheck and build clean. `npm test` runs
end to end for the first time — see "A test that had been red" below.

## The seam is `RuntimeSession`, not `WorkerBackend`

The scope's original wording predates a collision: `workerManifest.ts` already
exports `WorkerBackend` for metal/sycl/cpu, and `workerManager.ts` uses
`backend` as a local throughout the llama.cpp start path. That is a different
axis entirely — a *backend* is how llama.cpp uses the hardware, a *runtime* is
which engine holds the weights. The discriminator already existed as
`ModelRuntime`.

A `RuntimeSession` is a started process, the client that talks to it, and how
it ends. `WorkerManager` holds one at a time and dispatches on
`runtimeForModel(model)`.

## No second scheduler, deliberately

The concern was two 30B models resident at once. It does not arise, and adding
queueing would not have been what prevented it: one manager owns one session,
and every request already funnels through its single `InferenceScheduler`.
Starting either runtime evicts the other for exactly the reason two GGUF models
could never coexist. Splitting ownership per runtime is the change that *would*
have created the risk.

What actually needed generalizing was teardown and crash recovery. How a
process is asked to stop differs — llama-server takes a signal, the Python
worker takes a closed stdin — but what an unexpected exit *means* is identical,
so `handleExit` moved onto the session and stayed in one place.

`exitNotifier` fires immediately for a handler registered after the process has
already died. Without that, a worker crashing between passing its health check
and being watched would leave a session that looks alive forever, and the
manager would keep handing callers a client to a process that is gone.

## Three call sites the contract had to grow for

Widening `run<T>` from `LlamaClient` to `InferenceClient` exposed methods the
first cut of the interface had missed: `countChatInputTokens`, and the native
tool-call cache (`get`/`setNativeToolCallSupport`). Both are now on the
contract. The Python adapter implements the token count through the worker's
`model.tokenize`, which already counts a conversation through the chat
template — the same question. Native tool-call support is reported as
structurally `unavailable` rather than cached, because there the answer does
not depend on the model.

`canRunInline` **cannot** consult `supports.infill`: it runs before any client
exists. The capability guard belongs where a live client does — inline
completion itself, and `validateFillInMiddle`, which now records `unsupported`
without spending a load to discover what the runtime already knows.

## Checkpoint verification is not the GGUF check

The llama.cpp path rejects a model whose size or mtime moved since it was
registered. Reusing that for a checkpoint would reject every one of them:
copying a checkpoint to a faster disk rewrites every mtime and not one byte,
which is exactly why `manifestDigest` excludes mtimes. The Safetensors path
compares the digest instead — paths, sizes and Safetensors headers — so it
survives a move and still catches a swap.

## `localLlm.pythonPath`, until provisioning ships

Phase 1 is still the open risk. Until it lands, the interpreter is a setting,
and an empty value fails by naming it. Discovering `python3` on `PATH` would
find the system interpreter, and several gigabytes of PyTorch is not something
to install there on a user's behalf.

The interpreter is probed with `--probe` in a throwaway process before the
worker is spawned. That is not defensive styling: importing torch can abort the
interpreter rather than raise — a duplicate OpenMP runtime does it on this very
machine — and a worker cannot report its own abort.

## A test that had been red

`toolProtocol.test.ts` failed on a clean checkout, and because `npm test`
chains `test:node && test:python`, the Python suite had never run from the
top-level command. The cause was one word: `toolProtocol.ts` imported
`ChatTool` as a value rather than `import type`, so Node's strip-only
TypeScript kept the import and then could not resolve `../domain.js`, which
does not exist in the source tree. Fixed.

## Still to do

- **UI commands** (~2–3 days). Nothing invokes `importSafetensorsDirectory`
  yet, the model list does not show format or runtime, and `quantization` /
  `customCodeRequired` are recorded but never surfaced before a load.
- **Phase 1, Python provisioning** (~2–3 weeks). The largest remaining risk
  and the only one with no upstream answer.
- **Phase 6, tool calling** via a grammar backend such as `xgrammar`. Until
  then Safetensors models are Chat-only, and say so.
- **Phase 7, FIM.** No Transformers analogue exists; FIM tokens live outside
  the chat template. The alternative is advertising `fillInMiddle:
  'unsupported'` permanently for this runtime, which is what happens today.
