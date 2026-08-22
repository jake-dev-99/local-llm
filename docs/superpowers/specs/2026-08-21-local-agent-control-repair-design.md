# Local Agent Response Quality and Work Control Design

> **Superseded:** `docs/superpowers/specs/2026-08-22-thin-local-agent-provider-design.md` replaces this controller design.

**Status:** Implemented and source-validated; installed macOS and Windows runtime validation pending

**Date:** 2026-08-21

## 1. North Star

The project must deliver valid local-model results inside native VS Code workflows.

Result quality comes before response speed.

Local inference may be slow because models and hardware are limited.

A thirty-minute request is acceptable when it produces a strong, evidence-based result.

A five-second request is unacceptable when it blindly repeats an earlier answer.

Time becomes waste only when work is redundant, stalled, irrelevant, or unable to improve the result.

The extension must remain well-scoped, scalable, and useful across models, hardware, and VS Code experiences.

The current PoC validates two machines:

1. An Apple Silicon M4 Pro Mac using Metal.
2. A Windows x64 Intel Arc machine using the current CPU worker.

Future platforms must reuse the same provider contracts and pass separate runtime validation.

Prompts, source code, tool results, and generated text must remain local.

The model-download workflow is the only allowed remote exception.

## 2. Reported Quality Failure

The user sent the same audit prompt twice inside one Local Agent conversation.

The first request returned an answer.

The second request later returned the exact same answer.

That duplicated result is the failure.

The request duration increased the wasted computation, but did not define the failure.

The second response did not demonstrate a fresh evaluation of the active request.

### 2.1 Confirmed Trace Facts

The trace captures about twenty-seven minutes of the second request's worker processing.

It shows one discarded native attempt, eight required fallback decisions, and one automatic final decision.

The final response was freshly generated, not replayed from an extension cache.

The current `maxAgentToolRounds` behavior explains the eight forced decisions.

It does not explain the duplicated final wording by itself.

### 2.2 Diagnosis Boundary

The prior answer remained in conversation history.

Schema-constrained fallback generation used temperature zero.

Those facts may explain deterministic repetition, but the trace cannot prove the copying mechanism.

The repair must prevent duplicate output without pretending the exact cause is known.

## 3. Definition of a Valid Result

A valid Local Agent result must satisfy these requirements:

1. It directly addresses the active user request.
2. Workspace-dependent claims come from current tool evidence.
3. Relevant claims remain traceable to collected evidence.
4. The answer distinguishes completed work, unfinished work, and uncertainty.
5. The answer never claims an edit or check succeeded without supporting tool output.
6. A repeated request receives an independent evaluation.
7. An exact normalized duplicate final response fails repeated-request validation.

A repeated evaluation may reach the same conclusion.

It must still demonstrate fresh evidence and current reasoning.

Matching conclusions remain acceptable; matching normalized response text does not.

## 4. Quality Control Contract

Program checks enforce measurable facts:

1. At least one successful evidence result exists for workspace-dependent requests.
2. Tool calls and results remain novel enough to improve the answer.
3. Invocation and physical context limits remain truthful.
4. A repeated final response does not exactly match the prior final response.
5. Native capability records match the active runtime fingerprint.

The model judges semantic sufficiency:

1. Whether collected evidence answers the request.
2. Whether another tool call could materially improve the result.
3. Whether remaining uncertainty requires more evidence.
4. Whether the final answer accurately reflects completed work.

One evidence result satisfies minimum discovery.

It does not automatically mean the request is complete.

The model may continue gathering useful evidence until sufficient or constrained.

No separate classification generation is added.

The existing automatic fallback already chooses another tool or final text.

## 5. Minimal Architecture

The repair extends existing helpers instead of adding a general agent framework.

`localAgentToolChoice.ts` becomes a pure evaluator for the active Local Agent request.

It reconstructs state from messages already supplied by VS Code.

It stores no duplicate conversation history.

The evaluator returns three outcomes:

1. `requireEvidence` offers recognized read-only tools.
2. `allowToolOrFinal` lets the existing fallback choose the next useful action.
3. `forceFinal` requests the best supported answer from collected evidence.

The evaluator tracks only these facts:

1. The active user request.
2. Emitted invocation count.
3. Successful evidence presence.
4. Prior call signatures and result digests.
5. Repeated-request status.
6. Prior final-response digest.

`localAgentTools.ts` retains Local Agent filtering and edit-after-read behavior.

`localLanguageModelProvider.ts` applies evaluator decisions before shared worker execution.

The shared worker client retains generation, validation, streaming, and cancellation responsibilities.

Ordinary Chat and VS Code's built-in Agent retain their current orchestration.

## 6. Evidence and Novelty

A successful evidence result must satisfy these conditions:

1. It matches an emitted recognized discovery call.
2. Its content is nonempty.
3. It is not a known tool failure.
4. Its normalized digest is new during the active request.

Explicit no-match and no-error results count because they establish absence.

Failed, empty, malformed, or repeated results do not satisfy minimum evidence.

Every emitted invocation still counts toward the ceiling.

A call signature combines the tool name with canonical JSON arguments.

An exact duplicate call is blocked unless a successful edit followed the earlier call.

The model then chooses a different tool call or final text.

A second consecutive duplicate forces final generation from collected evidence.

This stops loops because work repeated, not because work took too long.

## 7. Repeated-Request Protection

Protection activates for an exact normalized request already followed by final text.

The provider omits only the matching prior assistant answer from model input.

Other history, tool calls, and tool results remain available.

The repeated request must obtain one fresh evidence-bearing result.

The model then answers independently from current evidence.

The final response is buffered before delivery for this repeated-request case.

Its normalized digest is compared with the matching prior response.

An exact duplicate is rejected once.

The model receives one revision request using fresh evidence and no prior answer text.

If the revision duplicates again, the extension does not present it as a valid result.

It returns a clear quality failure and names the evidence collected.

This bounded retry prevents another deterministic generation loop.

## 8. Work and Context Control

Local Agent permits at most eight tool invocations per active user request by default.

The existing setting remains configurable.

Eight is a maximum, never a required count.

Loaded tool definitions never count as invocations.

Failed, blocked, and repeated invocations count because they consume work.

Final generation never counts as an invocation.

The physical input budget remains authoritative.

Exact token counting continues before every inference request.

The model receives remaining input tokens, invocation count, and cumulative result size.

Token and result pressure instruct narrower work without removing raw history.

Only physical context and invocation limits are hard boundaries.

Elapsed time remains diagnostic and cancellation-aware.

Progressing requests receive no time-based cancellation.

User cancellation ends work immediately without automatic final text.

Worker transport failures remain visible errors.

## 9. Persistent Native Capability

The current worker client remembers native tool support for one process only.

That causes one discarded generation after every worker restart.

The persisted record uses this exact fingerprint:

1. Model file SHA-256.
2. Worker build identifier.
3. Chat-template fingerprint.
4. Platform and architecture.
5. Tool-protocol version.

An unknown fingerprint receives one native attempt.

An unavailable fingerprint uses the existing schema-constrained fallback immediately.

Any fingerprint change invalidates the record.

This cache benefits every tool-enabled request.

It prevents known-useless work without judging request duration.

## 10. Scalability and Platform Boundaries

Controller decisions depend on messages, tool evidence, configuration, and runtime capabilities.

They never depend on a specific model name or hardware speed.

The TypeScript behavior remains shared across platform packages.

Platform workers continue reporting their actual capabilities.

The current macOS package uses Metal on the M4 Pro acceptance machine.

The current Windows package guarantees CPU inference on the Intel Arc acceptance machine.

Intel Arc acceleration is not added by this repair.

No untested platform receives a support claim.

Future workers may add acceleration without changing the response-quality contract.

## 11. Absolute Locality

1. Worker traffic remains authenticated and bound to `127.0.0.1`.
2. The worker client rejects non-loopback runtime addresses.
3. Local Agent receives only local workspace tools.
4. Remote inference and remote fallback remain prohibited.
5. New state remains inside VS Code global storage.
6. Model downloads may contact HTTPS sources for model metadata and model bytes.
7. Download requests never include prompts, source, tool results, or generated text.
8. Logs never include prompt bodies, source bodies, or tool-result bodies.
9. This extension adds no telemetry.
10. This repair adds no network path.

The source audit found remote fetches only inside the model-download workflow.

Inference fetches use the loopback address created by `WorkerManager`.

Unrelated VS Code extensions remain outside this extension's enforcement boundary.

Documentation must state that boundary without weakening this extension's guarantee.

## 12. Error and Status Behavior

1. Missing required discovery tools fail before inference.
2. Invalid tool names or arguments never reach VS Code.
3. Malformed fallback decisions continue failing closed.
4. Duplicate final output never appears as successful completion.
5. Ceiling exhaustion produces the best supported partial answer.
6. Partial answers name unfinished work and uncertainty.
7. Capability persistence failure reverts to `unknown`.
8. Capability persistence failure never enables remote inference.

Debug logs record these measurements without content bodies:

1. Active-request invocation count.
2. Successful evidence count.
3. Duplicate call and result detection.
4. Repeated-response detection and revision.
5. Input tokens and cumulative result size.
6. Elapsed time and cancellation.
7. Native capability source and fingerprint status.

Elapsed time alone never produces a warning or failure.

## 13. Expected Changes

1. Replace mandatory-round logic in `src/provider/localAgentToolChoice.ts`.
2. Update policy handling in `src/provider/localAgentTools.ts`.
3. Integrate quality decisions in `src/provider/localLanguageModelProvider.ts`.
4. Add repeated-response comparison around repeated-request final generation.
5. Persist native capability through `src/models/modelRegistry.ts`.
6. Consume capability state in `src/worker/llamaClient.ts`.
7. Define the capability record in `src/domain.ts`.
8. Correct the setting description in `package.json`.
9. Replace tests that encode eight mandatory calls.
10. Update `README.md` behavior, platform, and privacy wording.

No unrelated provider, download, completion, or capacity refactor belongs here.

## 14. Validation

Source validation must prove these behaviors:

1. One evidence result permits completion when the model judges it sufficient.
2. Complex requests may gather additional useful evidence.
3. Eight emitted calls prevent a ninth call by default.
4. Failed and duplicate calls count toward the ceiling.
5. Long progressing requests receive no elapsed-time cancellation.
6. Repeated requests omit the matching prior answer.
7. Repeated requests require fresh evidence.
8. Exact duplicate final output triggers one revision.
9. A second duplicate becomes an explicit quality failure.
10. Same conclusions remain valid when fresh evidence supports them.
11. Persisted unavailable capability skips the discarded probe after restart.
12. Fingerprint changes invalidate persisted capability.
13. Non-loopback worker addresses fail.
14. Ordinary Chat remains unchanged.
15. Built-in Agent receives no Local Agent invocation ceiling.
16. Logs contain metrics without prompt or workspace content.

Installed runtime validation remains separate:

1. Install the exact package on the M4 Pro Mac.
2. Run the same evidence-dependent request twice.
3. Confirm the second response uses fresh evidence.
4. Confirm the second response never blindly duplicates the first.
5. Confirm a simple request may finish after one call.
6. Confirm a complex progressing request may run without a time deadline.
7. Restart the worker and confirm the discarded probe stays absent.
8. Confirm inference traffic remains loopback-only.
9. Repeat the same product path on the Windows Intel Arc machine.
10. Record that Windows validation used CPU inference.

Tests, packages, macOS runtime, and Windows runtime remain separate evidence claims.

## 15. Non-Goals and Delivery

This repair excludes these items:

1. A general autonomous-agent framework.
2. Response-speed targets or runtime service levels.
3. Chat-history compaction or retention redesign.
4. Fixed total runtime limits.
5. New Local Agent tools.
6. Local embeddings or workspace indexing.
7. Intel Arc acceleration.
8. Dynamic-capacity implementation inside this repair.
9. Remote inference, remote fallback, or telemetry.

Delivery order remains:

1. Implement this response-quality repair.
2. Validate source behavior.
3. Validate the installed M4 Pro package.
4. Validate the Windows Intel Arc machine using CPU inference.
5. Return immediately to dynamic-capacity implementation.

Success means valid local results improve through useful work, regardless of necessary runtime.
