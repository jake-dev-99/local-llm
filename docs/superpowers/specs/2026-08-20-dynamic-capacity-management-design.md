# Dynamic Capacity Management Design

**Status:** Approved architecture; implementation not started

**Date:** 2026-08-20

**Target platforms:** Apple Silicon macOS with Metal; Windows x64 with CPU inference after platform validation

## 1. Goal

Replace implicit, partly stale context sizing with an explicit capacity subsystem that:

- distinguishes automatic fitting from manual overrides;
- calculates automatic capacity from the model, llama.cpp's projected memory use, and current platform memory signals;
- advertises a truthful static input/output partition to VS Code;
- treats the running worker's `/props` response as authoritative;
- invalidates capacity when launch inputs or memory safety change;
- never interrupts an active request solely to resize context; and
- makes every estimate, override, fallback, and stale condition visible.

The design addresses the observed `34048` physical context and `32000` advertised input limit. The physical value was a llama.cpp fit produced with an explicit 8 GiB reserve; the advertised value was the physical window minus the configured 2,048-token output allowance. Neither number should appear unexplained or look hard-coded.

## 2. Locked Decisions

- Capacity management is a separate domain from model compatibility.
- The public capacity modes are `auto`, `manual-reserve`, and `manual-context`.
- Existing explicit numeric overrides keep their behavior during migration.
- Automatic fitting uses a platform memory probe plus the pinned llama.cpp fitter.
- The extension bundles the pinned `llama-fit-params` utility for dry-run planning.
- The extension bundles a small platform-specific capacity-probe executable rather than parsing shell-command output or maintaining a llama.cpp fork.
- A dry-run plan is provisional; `/props` from the loaded worker is authoritative.
- VS Code is always told a static `maxInputTokens` and `maxOutputTokens` whose sum does not exceed the physical or safely planned context.
- Required-tool turns may generate at most 256 tokens but do not reclaim the rest of the advertised output allowance after VS Code has packed the prompt.
- An active request is never cancelled solely because memory pressure changed.
- Unsafe capacity is sticky until the worker is refitted.
- Automatic shrink happens before the next request; automatic growth never interrupts work or forces an immediate restart.
- No request is silently trimmed to compensate for a stale advertisement.
- macOS Metal and Windows CPU fitting have separate platform validation gates.
- Local unit tests, packaged artifacts, installed runtime behavior, and platform validation are reported as separate evidence.

## 3. Non-Goals

This work does not include:

- resizing a loaded llama.cpp context in place;
- starting multiple inference workers intentionally;
- reclaiming VS Code's static output reservation on a per-request basis;
- automatic restarts merely to gain a larger context;
- Linux support;
- Windows GPU acceleration;
- changing model download, tool protocol, or Local Agent behavior except where capacity errors must fail closed;
- maintaining a patched llama.cpp fork;
- exact prediction of all future memory allocations by unrelated applications; or
- claiming that advisory operating-system memory signals are a hard memory guarantee.

## 4. Capacity Modes and Configuration

### 4.1 Public mode

Add `localLlm.capacityMode` with these values:

- `auto`: calculate a safe memory budget, then let the pinned llama.cpp fitter select context and offload parameters.
- `manual-reserve`: fit automatically while leaving the configured manual reserve unused.
- `manual-context`: disable context fitting and request the configured physical context directly.

The default is `auto`.

### 4.2 Numeric values

- `localLlm.contextSize` remains the value used by `manual-context`.
- `localLlm.metalMemoryReserveMiB` remains the value used by `manual-reserve` on Metal and remains supported for compatibility.
- `metalMemoryReserveMiB` is explicitly described as a Metal-specific legacy name. A platform-neutral reserve key is outside this change; Windows automatic fitting does not depend on this setting.
- Numeric values for inactive modes remain stored. Status identifies them as inactive rather than deleting them.

`manual-reserve` is supported only where reserve-based fitting has passed the platform validation matrix. In this delivery, that means macOS Metal. Selecting it on an unvalidated target fails configuration validation with a specific message.

### 4.3 Legacy resolution

When `capacityMode` has no explicit global, workspace, or workspace-folder value, resolve legacy settings in effective VS Code precedence order:

1. An explicitly stored `contextSize > 0` resolves to `manual-context`.
2. Otherwise, an explicitly stored `metalMemoryReserveMiB` resolves to `manual-reserve`.
3. Otherwise, resolve to `auto`.

The resolver uses `WorkspaceConfiguration.inspect()` so the extension default is not mistaken for a user override. It does not rewrite settings automatically.

An explicitly selected `capacityMode` is authoritative. For example, explicitly selecting `auto` leaves an existing reserve stored but inactive.

### 4.4 Validation

Validation runs before capacity planning or worker spawn.

- `manual-context` requires an integer context greater than zero.
- A manual context cannot exceed a known GGUF trained-context value.
- A manual reserve must be an integer from 1,024 through 32,768 MiB.
- Invalid values are rejected, not silently clamped.
- The final context must leave at least one input token and one output token.
- Automatic fitting must produce at least llama.cpp's 4,096-token fitted-context floor.
- Unsupported mode/platform combinations are errors rather than silent fallbacks.
- Every diagnostic identifies whether the effective value came from the default, global, workspace, or workspace-folder scope.

## 5. Capacity State and Provenance

### 5.1 State

Each installed model has a capacity state independent of its chat-template and tool-call capability:

- `unknown`: no usable plan or observation exists.
- `estimated`: a dry-run plan or conservative bootstrap value is being advertised.
- `current`: the worker reported `/props`, and the observation matches the active launch fingerprint.
- `stale`: a prior record exists, but a launch input or validation assumption changed.
- `unsafe`: current pressure invalidated the worker for future requests.

`unsafe` is stronger than `stale`. It is sticky: a later return to normal pressure does not make the existing worker safe again without a new successful plan/load cycle.

`current` always describes a live worker. A normal worker stop converts its persisted authoritative observation to `estimated`; a historical `/props` result never remains labeled current after its process exits.

An independent `expandable` flag means sustained additional memory may support a larger context. It never changes `current` to `stale` and never forces a restart.

### 5.2 Persisted provenance

Persist capacity under a separate versioned key, `localLlm.capacityProfiles.v1`, keyed by model SHA-256 and launch fingerprint. Each record contains:

- schema and capacity-policy versions;
- model identifier, SHA-256, file size, and trained context;
- resolved capacity mode and configuration source;
- base reserve, other-worker observation, and effective fit target;
- planned context and planned worker-memory requirement when available;
- authoritative loaded context and measured worker footprint when available;
- last advertised input and output limits;
- platform, architecture, acceleration, and device identity;
- worker manifest digest, worker build string, and pinned llama.cpp revision;
- launch fingerprint and advertisement fingerprint;
- preflight and post-load memory snapshots;
- planning, loading, and validation timestamps;
- freshness state, stale reason, and whether the record was persisted successfully.

Legacy `runtimeProfile.loadedContextSize` values are imported only as historical estimates. They cannot become `current` because they lack a launch fingerprint and memory provenance.

### 5.3 Fingerprints

The versioned `launchFingerprint` is the SHA-256 of canonical JSON containing:

- model identity;
- worker manifest digest and build identity;
- platform, architecture, acceleration, and device identity;
- capacity mode and active manual value;
- batch and micro-batch sizes;
- parallel slot count;
- GPU-layer, device, offload, KV, and context-related flags; and
- every other argument that changes the worker process, including CPU threads.

Any launch-argument change stops the idle worker or marks the active worker for stop after the request. It invalidates the prior launch profile.

The memory snapshot is provenance, not fingerprint input. Volatile memory fluctuations therefore do not make every stored record immediately stale.

The `advertisementFingerprint` covers the chosen physical context and configured maximum output. Changing `maxOutputTokens` refreshes VS Code information without restarting the worker. `maxTools` also refreshes model information but is not a capacity or launch input. Changing `startupTimeoutSeconds` affects the next startup and does not stop an already ready worker.

## 6. Native Platform Probe

### 6.1 Boundary

Bundle `local-llm-capacity-probe` beside `llama-server` in each platform VSIX. The helper has two operations:

- `snapshot --json`: print one versioned JSON snapshot and exit.
- `watch --jsonl --pid <workerPid>`: emit normalized pressure transitions and periodic snapshots until its parent closes the pipe or it is terminated.

The helper reports memory metadata only. It never receives model paths, prompts, document content, tool results, or worker credentials.

### 6.2 Normalized snapshot

The version-one schema contains:

- timestamp, platform, architecture, and probe build;
- total physical memory;
- available or reclaimable physical memory;
- available commit headroom when the platform exposes it;
- pressure state: `normal`, `warning`, `critical`, or `unknown`;
- Metal device name and recommended working-set size when applicable;
- observed worker resident footprint when a PID was supplied; and
- an optional diagnostic when a platform measurement is unavailable.

On macOS, the helper uses native Mach/Dispatch memory information and Metal device properties. It does not parse `vm_stat`, `memory_pressure`, or other command output. On Windows, it uses `GlobalMemoryStatusEx` and `GetPerformanceInfo` for physical and commit availability.

macOS native normal, warning, and critical events map directly to the normalized states. On Windows, the native low-memory resource notification maps to `warning`; the policy derives `critical` when available physical or commit headroom is at or below the 1 GiB safety floor, or when the safe budget cannot sustain the active worker's recorded minimum requirement. Failure to obtain either platform signal produces `unknown`, not `normal`.

Metal's process-local current allocation remains llama.cpp's responsibility during fitting. The external probe uses the operating-system snapshot and worker process footprint to account for system-wide conditions without pretending it can read another process's Metal allocator internals.

### 6.3 Probe failure

If the probe cannot produce a valid snapshot:

- `auto` may use llama.cpp's native device-only fitter;
- status and logs label the result `Auto — device-only fallback`;
- the profile records the probe failure and cannot claim system-aware fitting; and
- a previously received sticky critical-pressure event still blocks launch.

Manual modes do not change their configured values because the probe failed.

## 7. Automatic Fit Policy

### 7.1 Preflight budget

The platform adapter normalizes the safe budget immediately before planning or launching:

```text
system headroom = min(available physical memory, commit headroom when present)
system budget   = max(0, system headroom - 1 GiB)
device budget   = max(0, Metal recommended working-set headroom - 1 GiB)
safe worker budget = min(system budget, device budget when present)
```

The 1 GiB floor matches llama.cpp's default fit margin. It is a minimum margin, not the primary capacity setting. Other application use reduces `system headroom`, which dynamically increases the effective fit target.

For Metal, convert the safe worker budget into the margin llama.cpp expects:

```text
fit target = max(1 GiB, Metal fit headroom - safe worker budget)
```

The planner receives the resulting target and lets llama.cpp project model weights, context, and compute buffers. The policy never estimates token memory from GGUF file size alone.

Memory already held by another local worker is recorded. It is not added again when the system snapshot already accounts for it.

A critical preflight state blocks launch. A warning state may plan a smaller worker if the normalized budget still supports the model and 4,096-token minimum.

### 7.2 Dry-run planner

Bundle `llama-fit-params` from the same pinned commit and build configuration as `llama-server`.

`CapacityPlanner` runs one planner process at a time, default or active model first. It passes the same model, acceleration, batch, micro-batch, device, offload, and fit-target arguments intended for the worker. It parses the final fitted context and projected memory diagnostics from the pinned output contract.

A plan is valid only for its launch fingerprint and while the current safe worker budget is at least its planned requirement. Cancellation terminates the planner process. Planner output never enters the general model registry.

### 7.3 Closed-loop startup

Automatic startup follows this sequence:

1. Capture a fresh preflight snapshot.
2. Reject a critical-pressure launch.
3. Reuse or create a matching dry-run plan.
4. Launch `llama-server` with the resolved fit policy.
5. Wait for health and read `/props` before marking the worker ready.
6. Capture a post-load snapshot and measured worker footprint.
7. If loading itself caused unsafe pressure, stop and retry once with a freshly calculated tighter budget.
8. If the second load is unsafe, fail clearly and leave the worker stopped.

There are at most two load attempts for one request. Crash restart limits remain separate and do not reset the capacity retry count.

## 8. Pressure and Restart Policy

### 8.1 Automatic mode

- Warning or critical pressure during an active request marks capacity `unsafe` but does not cancel the request.
- When that request finishes, the coordinator stops the now-idle worker.
- Warning or critical pressure while idle stops the worker immediately.
- The next request performs fresh planning and fitting.
- Returning to normal pressure does not clear the sticky unsafe state.
- Sustained additional budget marks the profile `expandable` after the normalized safe budget remains at least 10 percent and 2 GiB above the budget used for 60 seconds.
- `expandable` produces status and a `Refit Model Capacity` action; it does not restart automatically.

### 8.2 Manual modes

The extension does not change a user's manual reserve or manual context automatically.

- Warning pressure is surfaced without resizing.
- Critical pressure follows the same no-interruption rule for an active request, then stops the idle worker.
- A new manual-mode launch remains blocked until pressure is no longer critical.

### 8.3 Other invalidation

Model changes, model-byte changes, worker changes, capacity-policy changes, and launch-setting changes always invalidate the active session. An active request finishes first unless the worker itself reports a fatal error.

## 9. Provider Bootstrap and VS Code Contract

### 9.1 Advertisement priority

`LanguageModelChatProvider` chooses the advertised physical context in this order:

1. Live authoritative `/props` for the active matching worker.
2. A dry-run plan valid for the current launch fingerprint and safe budget.
3. A persisted authoritative profile whose recorded required footprint still fits a fresh preflight budget; it is advertised as `estimated`, not `current`.
4. The 4,096-token conservative floor while planning is pending.

Model enumeration returns promptly. Planning runs sequentially in the background, then fires `onDidChangeLanguageModelChatInformation` as usable plans arrive.

### 9.2 Static partition

For every advertisement:

```text
maxOutputTokens = clamp(configured output, 1, physical context - 1)
maxInputTokens  = physical context - maxOutputTokens
```

The sum never exceeds the physical or safely planned context.

Required-tool discovery may set the worker request's output ceiling to 256 tokens. It does not increase `maxInputTokens`, because VS Code has already packed the prompt using the advertised static partition.

### 9.3 Generation race

Every custom model-information object carries a capacity generation identifier.

When a request reaches the provider:

- an equal generation proceeds;
- a newer capacity that is at least as large proceeds using the request's smaller advertised limit, then refreshes information;
- a smaller or unsafe generation fails before inference, fires a refresh, and asks the user to retry; and
- no message, tool definition, or history entry is silently removed.

After load, `/props` must return a positive integer context:

- loaded context equal to or larger than the advertised context is accepted;
- loaded context smaller than the advertised context rejects that request before sending inference; and
- missing or malformed `/props` prevents the worker from becoming ready.

## 10. Component Boundaries

```text
CapacityProbe
  Native snapshots and pressure events

CapacityPolicy
  Pure configuration resolution, validation, budget math, and fingerprints

CapacityPlanner
  Pinned llama-fit-params execution and provisional plans

CapacityStore
  Versioned capacity provenance, separate from model compatibility

CapacityCoordinator
  State machine, invalidation, planning, load validation, refitting, and refresh events

WorkerManager
  Starts and stops the exact launch plan supplied by the coordinator

LanguageModelProvider
  Advertises immutable capacity snapshots and accepts matching generations
```

The coordinator is the only component that combines platform memory, persisted capacity, and worker lifecycle. `WorkerManager` remains responsible for process mechanics, health, cancellation, and crash recovery. The provider no longer writes runtime capacity into the model registry.

Model compatibility remains responsible for chat-template, system-role, tool-call, and infill observations. A worker-build change can invalidate both domains, but each domain stores and explains its own state.

## 11. User Experience and Logging

The status-bar text stays compact:

```text
Qwen 32B · 62.2K ctx
```

The hover and `Local LLM: Show Capacity` command display:

- physical context and freshness state;
- automatic or manual mode;
- manual reserve/context and configuration scope when applicable;
- effective fit target;
- advertised input and output limits;
- live, planned, persisted, conservative, or device-only source;
- worker build and fitted timestamp;
- stale or unsafe reason;
- inactive stored overrides;
- persistence failure when applicable; and
- whether more capacity may be available.

Add `Local LLM: Refit Model Capacity`. If a model is loaded and idle, the command immediately stops, replans, and reloads that model. If a request is active, it queues the same operation for the moment the request finishes. If no model is loaded, it invalidates the selected/default model's plan and performs the fresh fit on its next use.

Representative logs are:

```text
Capacity policy: manual-reserve; reserve=1024 MiB; source=global
Capacity plan: context=62208; fitTarget=1024 MiB; source=system-aware
Capacity loaded: context=62208; input=60160; output=2048; source=/props
Capacity invalidated: pressure=warning; refit=before-next-request
```

Logs never include prompt bodies, generated text, tool results, source content, or API credentials.

## 12. Failure Handling

- Invalid configuration fails before planner or worker spawn and identifies the setting and scope.
- Unsupported automatic fitting is reported explicitly; it never becomes an unrelated llama.cpp default silently.
- Probe failure can use the labeled device-only fallback.
- Planner failure leaves the model at the conservative bootstrap limit and exposes the error. A later real load may still succeed.
- Critical preflight pressure blocks launch.
- Post-load unsafe pressure receives one bounded retry.
- A loaded context below the advertised generation fails before inference and requests a retry after refresh.
- `/props` failure prevents readiness.
- A capacity-store write failure leaves a verified live session usable, logs the failure, and marks the profile nonpersistent.
- Configuration-triggered worker stops are awaited through the lifecycle serializer and all failures are logged.
- Fatal worker errors retain the existing crash-recovery policy and are not relabeled as ordinary capacity changes.

## 13. Platform Validation Gates

### 13.1 macOS Metal

Validate automatic fitting on supported Apple Silicon hardware with the Qwen2.5-Coder-32B-Instruct Q5_K_M model and at least one smaller fallback model.

The matrix must cover:

- abundant normal-pressure startup;
- substantial unrelated application memory use before startup;
- pressure introduced during an active request;
- pressure introduced while idle;
- pressure returning to normal and producing `expandable` without restart;
- a model load that triggers the bounded tighter-fit retry; and
- manual 1 GiB and larger reserves with visible provenance.

### 13.2 Windows x64 CPU

Automatic CPU fitting remains disabled until native Windows execution demonstrates:

- context reduction based on available physical memory and commit headroom;
- peak worker footprint within the planned budget;
- no excessive paging during prompt processing and generation;
- predictable behavior under concurrent memory load;
- correct 4,096-token minimum handling; and
- clean refusal when the model cannot fit safely.

If this matrix fails, Windows exposes `auto` as unavailable and requires an explicit validated manual context. The extension does not claim Windows automatic fitting from cross-build or unit-test evidence alone.

## 14. Testing Strategy

### 14.1 Pure tests

- Legacy configuration resolution at every VS Code scope.
- Invalid mode/value/platform combinations.
- Canonical launch and advertisement fingerprints.
- Fit-target and safe-budget calculations.
- State transitions, sticky unsafe behavior, and expandable hysteresis.
- Invalidation classification for every relevant setting.
- Property tests proving:
  - input plus output never exceeds physical context;
  - no negative budget is produced;
  - no fitted context exceeds known trained context; and
  - unsafe capacity never becomes current without a new load.

### 14.2 Process and integration tests

- Capacity-probe JSON and JSONL schemas.
- Planner parsing with captured pinned llama.cpp fixtures.
- Planner cancellation and serialization.
- Worker launch argument parity with the dry-run plan.
- Valid, malformed, smaller, and larger `/props` results.
- Capacity-generation races.
- Pressure transitions during active and idle work.
- One-retry startup behavior.
- Store migration and persistence failures.
- Provider refresh after plan, load, invalidation, and output-limit changes.
- Required-tool 256-token ceiling with unchanged static advertisement.

### 14.3 Packaging tests

Each VSIX must contain only its platform's:

- `llama-server`;
- `llama-fit-params`;
- `local-llm-capacity-probe`; and
- manifest entries, hashes, and executable permissions.

The packaged executables must report the expected pinned worker and probe builds.

### 14.4 Installed-runtime tests

Install and reload the exact VSIX, then verify:

- the status and capacity command show the effective mode and source;
- the provider refreshes from planned to authoritative capacity;
- advertised input plus output equals or stays below `/props` context;
- induced pressure obeys the active-request and idle-worker policies;
- stale-generation requests fail closed without silent trimming;
- ordinary final responses retain the configured output allowance;
- required-tool turns retain the 256-token ceiling; and
- macOS and Windows results are reported separately.

Passing TypeScript tests, successful packaging, and an installed runtime smoke test are distinct release gates.

## 15. Documentation

Update the README and setting descriptions to cover:

- the three capacity modes;
- migration of explicit legacy settings;
- automatic versus device-only fallback status;
- supported and unvalidated platform combinations;
- the difference between physical context, input limit, and output allowance;
- why required-tool turns cannot reclaim the static output reservation;
- `Show Capacity` and `Refit Model Capacity`;
- stale, unsafe, and expandable states; and
- troubleshooting planner, probe, `/props`, pressure, and persistence failures.

Unsupported-platform messages must identify the detected platform and list the validated combinations.

## 16. Acceptance Criteria

The implementation is acceptable when:

1. An explicit 1 GiB or 8 GiB reserve is visibly manual in status and logs.
2. A user with no explicit legacy override resolves to `auto`.
3. Automatic Metal fitting incorporates current system headroom rather than only Metal's process-local working-set approximation.
4. A matching dry-run plan gives VS Code useful capacity before full model load.
5. `/props` immediately becomes authoritative and refreshes provider information.
6. No advertised input/output sum exceeds the current or safely planned physical context.
7. A capacity decrease between advertisement and request fails before inference without silent trimming.
8. Launch-affecting changes invalidate the worker and profile; output-only changes refresh without restart.
9. Unsafe pressure never interrupts an active request but forces refitting before later inference.
10. Increased capacity never forces an automatic restart.
11. Legacy profiles without fingerprints are never treated as current.
12. CPU/Windows auto fitting is enabled only after the native validation matrix passes.
13. Probe and planner failures remain visible and accurately labeled.
14. The exact packaged VSIX passes the installed-runtime checks on every platform claimed as validated.

## 17. Delivery Sequence

The later implementation plan should order work as follows:

1. Pure capacity types, configuration resolution, validation, fingerprints, and store migration.
2. Provider snapshot generation and truthful conservative fallback.
3. Pinned `llama-fit-params` build, planner, and provisional provider refresh.
4. Native probe snapshot support and automatic budget policy.
5. Capacity coordinator and authoritative `/props` integration.
6. Pressure watch, sticky invalidation, hybrid restart policy, and expandable status.
7. Commands, status details, logging, and documentation.
8. Packaging verification and macOS installed-runtime matrix.
9. Windows native CPU validation, followed by enablement only if its gate passes.

No implementation step may claim platform completion from source-level or cross-build checks alone.
