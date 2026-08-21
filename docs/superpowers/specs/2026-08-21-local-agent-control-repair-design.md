# Local Agent Control Repair Design

**Status:** Draft for review; implementation not started

**Date:** 2026-08-21

## 1. North Star

The project must prove a dependable, self-contained local language-model experience inside native VS Code workflows.

The product path includes model installation, Chat, bounded agent tools, inline completion, and offline inference.

Local Agent is one proof point, not the product's center.

This repair must remain small and must not delay the approved capacity work.

Prompts, source code, tool results, and generated text must never leave the machine.

The model-download workflow is the only allowed remote exception.

## 2. Confirmed Problem

The current `maxAgentToolRounds` setting requires eight completed calls before Local Agent may answer.

It therefore acts as a minimum instead of a ceiling.

One repeated request caused ten generations and about twenty-seven minutes of inference.

The run included one discarded native attempt, eight forced decisions, and one final decision.

The repeated answer was freshly generated, not replayed from an extension cache.

## 3. Required Behavior

1. One successful, evidence-bearing read-only result satisfies minimum discovery.
2. Explicit no-match and no-error results count as evidence.
3. Failed, empty, malformed, or repeated results do not satisfy discovery.
4. Every emitted tool invocation counts toward the ceiling.
5. Local Agent permits at most eight invocations per active user request.
6. Loaded tool definitions never count as invocations.
7. Final generation never counts as an invocation.
8. After discovery, the model may choose another tool or answer.
9. Existing automatic fallback supplies semantic early stopping.
10. No separate classification generation is added.
11. Exact repeated tool choices are blocked.
12. No fixed runtime deadline stops progressing work.
13. Token and result pressure narrows future work without rewriting history.
14. Native tool capability persists across worker restarts.
15. Repeated requests omit the matching prior answer and require fresh evidence.
16. Ordinary Chat and VS Code's built-in Agent retain current orchestration.

## 4. Minimal Design

The repair extends existing helpers instead of adding a general agent framework.

`localAgentToolChoice.ts` becomes a pure evaluator for the active Local Agent request.

It reconstructs state from messages already supplied by VS Code.

It stores no duplicate conversation history.

The evaluator returns three outcomes:

1. `requireDiscovery` offers only recognized read-only tools.
2. `allowToolOrFinal` uses the existing automatic tool-or-final fallback.
3. `forceFinal` removes tools and requests an evidence-based answer.

The evaluator tracks the active request, invocation count, evidence, and prior call signatures.

`localAgentTools.ts` retains filtering and edit-after-read behavior.

`localLanguageModelProvider.ts` applies the evaluator before shared worker execution.

The shared worker client retains generation, validation, streaming, and cancellation responsibilities.

## 5. Request Flow

1. The first Local Agent workspace decision requires one recognized read-only tool.
2. A matching nonempty result satisfies discovery unless it is a known failure.
3. The next decision may call another tool or return final text.
4. A duplicate tool signature receives one chance to choose differently or answer.
5. Another duplicate forces final generation from collected evidence.
6. The eighth emitted call prevents any ninth call.
7. Forced final text names unfinished work and uncertainty.
8. User cancellation ends the request without automatic final text.

The call signature combines the tool name with canonical JSON arguments.

A successful edit permits a later repeat because workspace contents changed.

## 6. Runtime and Context

The physical input budget remains authoritative.

Exact token counting continues before every inference request.

Cumulative invocation count and tool-result size guide narrower future choices.

Before automatic decisions, the model receives remaining input tokens and cumulative result size.

Only the physical input budget and invocation ceiling are hard limits.

The controller never removes raw VS Code history or tool results.

Elapsed time remains diagnostic and cancellation-aware.

Progressing requests receive no time-based cancellation.

Worker transport failures remain visible errors.

## 7. Persistent Native Capability

The current worker client remembers native tool support for one process only.

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

## 8. Repeated Requests

Protection applies to an exact normalized Local Agent request already followed by final text.

The provider omits only that matching prior answer from model input.

Other conversation history, calls, and results remain available.

The repeated request must obtain fresh evidence before answering.

The design prevents copying through direct answer exposure.

It does not force cosmetic rewording when fresh evidence supports the same conclusion.

## 9. Absolute Locality

1. Worker traffic remains authenticated and bound to `127.0.0.1`.
2. The worker client rejects non-loopback runtime addresses.
3. Local Agent receives only local workspace tools.
4. Remote inference and remote fallback remain prohibited.
5. New state remains inside VS Code global storage.
6. Model downloads may contact HTTPS sources for model metadata and model bytes.
7. Download requests never include prompts, source, tool results, or generated text.
8. This extension adds no telemetry.
9. This repair adds no network path.

The source audit found remote fetches only inside the model-download workflow.

Inference fetches use the loopback address created by `WorkerManager`.

Unrelated VS Code extensions remain outside this extension's enforcement boundary.

Documentation must state that boundary without weakening this extension's guarantee.

## 10. Expected Changes

1. Replace mandatory-round logic in `src/provider/localAgentToolChoice.ts`.
2. Update policy handling in `src/provider/localAgentTools.ts`.
3. Integrate decisions in `src/provider/localLanguageModelProvider.ts`.
4. Persist native capability through `src/models/modelRegistry.ts`.
5. Consume capability state in `src/worker/llamaClient.ts`.
6. Define the record in `src/domain.ts`.
7. Correct the setting description in `package.json`.
8. Replace tests that encode eight mandatory calls.
9. Update `README.md` behavior and privacy wording.

No unrelated provider, download, completion, or capacity refactor belongs here.

## 11. Validation

1. One evidence result permits final text.
2. Useful additional evidence remains possible.
3. Eight emitted calls prevent a ninth.
4. Failed and duplicate calls count toward eight.
5. Repeated requests omit the matching prior answer.
6. Repeated requests require fresh evidence.
7. Persisted unavailable capability skips the discarded probe after restart.
8. Fingerprint changes invalidate persisted capability.
9. Non-loopback worker addresses fail.
10. Ordinary Chat remains unchanged.
11. Built-in Agent receives no Local Agent ceiling.
12. Progressing work receives no elapsed-time cancellation.

Installed runtime validation remains separate:

1. Install the exact macOS package.
2. Run the same Local Agent request twice.
3. Confirm both requests gather evidence independently.
4. Confirm a simple request may finish after one call.
5. Restart the worker and confirm the discarded probe stays absent.
6. Confirm inference traffic remains loopback-only.
7. Repeat the product path on the target Intel Windows machine.

## 12. Non-Goals and Delivery

This repair excludes these items:

1. A general autonomous-agent framework.
2. Chat-history compaction or retention redesign.
3. Fixed total runtime limits.
4. New Local Agent tools.
5. Local embeddings or workspace indexing.
6. Dynamic-capacity implementation inside this repair.
7. Remote inference, remote fallback, or telemetry.

Delivery order remains:

1. Implement this narrow repair.
2. Validate source behavior.
3. Validate the installed macOS package.
4. Validate the Intel Windows path.
5. Return immediately to dynamic-capacity implementation.

Success means Local Agent performs bounded useful work without dominating the product roadmap.
