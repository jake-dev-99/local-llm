# Thin Local Agent Provider Design

**Status:** Approved for implementation planning

**Date:** 2026-08-22

**Supersedes:** `docs/superpowers/specs/2026-08-21-local-agent-control-repair-design.md`

## 1. Goal

Local LLM Engine remains a thin, truthful adapter between VS Code and a local `llama.cpp` worker.

The extension must optimize local-model operation without replacing model judgment or VS Code orchestration.

Valid results matter more than response speed.

Long work remains acceptable when each step can improve the result.

Prompts, source, tool results, and generated text remain local.

Model downloads remain the only allowed remote operation.

## 2. Verified Failure

The original trace showed a repeated audit request lasting about twenty-seven minutes.

The second request returned the same answer as the first request.

The trace contained this provider decision:

> `Local Agent workspace turn needs discovery; requiring one of 5 read-only tools.`

That decision appeared before every tool round.

The trace grew from 31 adapted messages to 47 adapted messages.

It showed one discarded native generation and eight required structured decisions.

The last step used one automatic structured decision.

The worker logged this fallback repeatedly:

> `Native tool output was unavailable; using one schema-constrained required decision.`

The automatic fallback generated final answer text inside its response schema.

`src/worker/llamaClient.ts` set that generation to `temperature: 0`.

The prior answer remained in conversation history.

Those facts made exact repetition plausible.

The trace cannot prove the model's exact copying mechanism.

## 3. Root Cause

The extension enforced semantic decisions that only the model could judge.

It required evidence before understanding whether evidence was relevant.

It classified edit intent with regular expressions.

It hid tools based on that classification.

It rewrote history for repeated requests.

It rejected repeated tool calls without understanding their purpose.

It rejected repeated final text without understanding whether repetition was correct.

It injected corrective prompts and replacement answers outside model judgment.

The resulting controller duplicated part of VS Code's agent harness.

These controls created new failures instead of improving model outcomes.

## 4. Design Principle

Deterministic enforcement remains appropriate only for objectively measurable facts.

The extension may enforce:

1. Loopback-only inference.
2. Worker authentication.
3. User cancellation.
4. Physical context limits.
5. Loaded tool limits.
6. Configured invocation limits.
7. Tool-name and argument schemas.
8. Runtime capability fingerprints.
9. Worker transport failures.

The extension must not enforce:

1. Request relevance.
2. Evidence sufficiency.
3. Edit intent.
4. Tool-call novelty.
5. Final-answer novelty.
6. Semantic completion.
7. Final wording.

## 5. Ownership

### 5.1 VS Code

VS Code owns conversation history and the agent loop.

VS Code executes tools and returns their results.

VS Code owns edit previews, approvals, and reversions.

### 5.2 Local Agent Instructions

The agent instructions define the model's working behavior.

They tell the model when workspace evidence is needed.

They tell the model to read relevant content before editing.

They tell the model to respect explicit read-only requests.

They tell the model to continue only while another tool can improve the result.

They tell the model to state unfinished work and uncertainty.

### 5.3 Provider

The provider adapts VS Code messages and tools to the worker contract.

It preserves the complete history supplied by VS Code.

It preserves VS Code's automatic or required tool mode.

It limits Local Agent tools to the declared local workspace set.

It counts Local Agent invocations for the configured maximum.

It streams validated model output directly to VS Code.

### 5.4 Worker Client

The worker client performs inference, validation, token counting, and fallback adaptation.

It does not judge task quality.

## 6. Local Agent Tool Policy

Local Agent always receives every recognized local workspace tool supplied by VS Code.

Unknown, remote, terminal, memory, and subagent tools remain unavailable.

Review requests still receive edit tools as available capabilities.

Availability does not authorize an edit.

The model follows instructions, and VS Code still controls edit approval.

Empty tool lists remain valid for ordinary responses.

The provider never forces discovery solely because Local Agent is active.

The provider never forces mutation based on request wording.

## 7. Invocation Ceiling

`localLlm.maxAgentToolRounds` remains eight by default.

The value counts invoked tools, not loaded definitions.

Final generation never counts.

The ceiling applies only to the bundled Local Agent.

Before the ceiling, VS Code's requested mode remains authoritative.

At the ceiling, the provider sends no tools and requests one normal final response.

The ceiling protects resources without judging work quality.

No elapsed-time limit ends progressing work.

## 8. Tool Fallback

Native llama.cpp tool calls remain the preferred path.

Known unsupported fingerprints skip the discarded native attempt.

Known supported fingerprints accept native automatic final responses without fallback.

Schema-constrained fallback selects only the next action.

The action has two possible forms:

1. A schema-valid tool name and arguments.
2. A `final` decision without answer text.

Structured decisions use the configured tool-decision output limit.

Structured decisions use temperature zero for reliable JSON.

A tool decision returns one validated tool call to VS Code.

A final decision starts ordinary generation without tools or a response schema.

Ordinary final generation uses the configured chat temperature.

Ordinary final generation streams directly to VS Code.

This separation prevents tool-selection constraints from shaping final prose.

## 9. Repetition

The provider does not remove matching prior answers from history.

The provider does not reject repeated tool calls.

The provider does not reject repeated final text.

The agent instructions prioritize the active request and current workspace evidence.

Repeated requests may reach the same supported conclusion.

Exact wording may also repeat when that wording remains correct.

Behavioral validation checks whether workspace answers use relevant evidence.

Different wording alone never proves improvement.

## 10. Capability Persistence

The existing native-tool capability record remains.

Its fingerprint contains:

1. Model file SHA-256.
2. Worker build.
3. Chat-template fingerprint.
4. Platform and architecture.
5. Tool-protocol version.

Exact matches reuse the stored capability.

Fingerprint changes restore one native capability probe.

Persistence failures return capability state to unknown.

## 11. Errors and Logging

Invalid tool names and arguments remain visible errors.

Oversized prompts remain visible errors.

Missing chat templates remain visible errors.

Worker transport failures remain visible errors.

User cancellation remains immediate.

Logs record tool mode, supplied tool count, invocation count, token use, and capability source.

Logs never contain prompt bodies, source bodies, tool results, or generated text.

Logs never alter model output.

## 12. Expected Source Changes

1. Simplify `src/provider/localAgentToolChoice.ts` to detection and invocation counting.
2. Simplify `src/provider/localAgentTools.ts` to allowlisting and objective mode selection.
3. Remove `src/provider/localAgentResponse.ts` and its tests.
4. Remove provider history rewriting, response buffering, and controller messages.
5. Preserve direct provider streaming and native capability persistence.
6. Change `src/worker/toolProtocol.ts` so `final` carries no answer text.
7. Change `src/worker/llamaClient.ts` to run ordinary generation after `final` selection.
8. Cap automatic and required structured decisions with `maxToolCallTokens`.
9. Tighten `agents/local-agent.agent.md` instructions without adding workflows.
10. Correct `README.md`, settings descriptions, version, and package documentation.

No new dependency or agent framework belongs in this work.

## 13. Source Validation

Automated tests must prove:

1. Local Agent greetings do not require tools.
2. Local Agent receives all recognized local tools.
3. Unknown tools remain excluded.
4. VS Code's required mode remains required.
5. Automatic mode remains automatic before the ceiling.
6. Eight invoked tools prevent a ninth tool.
7. Loaded definitions do not consume the ceiling.
8. Final responses do not consume the ceiling.
9. Structured fallback returns validated tool calls.
10. Structured fallback `final` decisions contain no final text.
11. Final decisions trigger ordinary streamed generation.
12. Final generation uses configured temperature and no response schema.
13. Persisted unavailable capability skips native probing.
14. Ordinary Chat remains unchanged.
15. Non-loopback worker addresses remain rejected.

## 14. Installed Runtime Validation

The exact macOS package must pass these checks:

1. `Reply Hi` invokes no workspace tool.
2. Repeating `Reply Hi` may return `Hi` again without failure.
3. A read-only audit gathers model-chosen evidence.
4. Repeating the audit performs useful work without forced rounds.
5. An explicit edit can invoke an editor after model-chosen reading.
6. VS Code displays its normal edit approval and reversal experience.
7. Long progressing work receives no elapsed-time cancellation.
8. A worker restart reuses matching unavailable native-tool capability.
9. Inference traffic remains loopback-only.

The same product path must later pass on the Windows Intel Arc machine.

Windows validation continues using the current CPU worker.

## 15. Non-Goals

This work excludes:

1. A custom agent runtime.
2. Local embeddings.
3. Workspace indexing.
4. Chat-history compaction.
5. Fixed runtime limits.
6. New Local Agent tools.
7. Windows acceleration.
8. Dynamic-capacity implementation.
9. Remote inference.
10. Telemetry.

## 16. Success

The extension protects objective boundaries while leaving semantic decisions to the model.

Simple requests remain simple.

Workspace work remains evidence-driven through model instructions and tool access.

Structured fallback improves tool reliability without generating final prose.

The original forced-round and replacement-answer failures disappear.

Dynamic-capacity implementation resumes after this repair passes macOS runtime validation.
