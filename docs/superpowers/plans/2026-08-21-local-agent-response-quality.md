# Local Agent Response Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Do not delegate this implementation.

**Goal:** Prevent mandatory eight-call loops and exact repeated answers while preserving useful long-running local work.

**Architecture:** Extend the existing Local Agent policy helpers with one reconstructed turn state. Keep generation inside `LlamaClient`. Apply retries only around Local Agent responses. Persist native-tool support by an exact runtime fingerprint.

**Tech Stack:** TypeScript, Node test runner, VS Code language-model provider, llama.cpp OpenAI-compatible API, esbuild, VSIX packaging.

**Spec:** `docs/superpowers/specs/2026-08-21-local-agent-control-repair-design.md`

## Global Constraints

1. Keep ordinary Chat and VS Code built-in Agent behavior unchanged.
2. Add no elapsed-time cancellation or remote inference path.
3. Keep prompts, source, tool results, and generated text out of logs.
4. Treat eight Local Agent invocations as a ceiling, never a minimum.
5. Keep this repair separate from dynamic-capacity implementation.

## Task 1: Reconstruct Local Agent Work State

**Files:**

- Modify: `src/provider/localAgentToolChoice.test.ts`
- Modify: `src/provider/localAgentToolChoice.ts`
- Modify: `src/provider/localAgentTools.test.ts`
- Modify: `src/provider/localAgentTools.ts`
- Modify: `src/provider/localLanguageModelProvider.ts`
- Modify: `package.json`

1. Replace mandatory-round tests with failing tests for these outcomes:
   - no evidence requires one discovery call;
   - one successful discovery result permits tool-or-final judgment;
   - failed results do not satisfy evidence;
   - emitted calls reach the configured ceiling;
   - final generation is excluded;
   - ordinary Chat remains unaffected.
2. Run `npm test -- src/provider/localAgentToolChoice.test.ts src/provider/localAgentTools.test.ts`.
3. Confirm failures show the current mandatory-minimum behavior.
4. Add one pure evaluator returning `requireEvidence`, `allowToolOrFinal`, or `forceFinal`.
5. Count emitted assistant tool calls, not loaded definitions or final generations.
6. Recognize unique, nonempty, successful discovery results as evidence.
7. Update provider policy so one evidence result enables automatic tool-or-final choice.
8. Force a final answer when the ceiling is reached.
9. Update the setting description to describe a maximum.
10. Re-run the focused tests and confirm they pass.

## Task 2: Protect Fresh Repeated Requests

**Files:**

- Modify: `src/provider/localAgentToolChoice.test.ts`
- Modify: `src/provider/localAgentToolChoice.ts`
- Modify: `src/provider/localLanguageModelProvider.ts`

1. Add failing tests proving an exact repeated request finds its matching prior final.
2. Add a failing test proving only that matching final is omitted from fresh model input.
3. Add failing tests for normalized duplicate response detection.
4. Add a failing test proving a changed answer remains acceptable.
5. Add failing tests for canonical duplicate tool-call signatures.
6. Run the focused test file and confirm the new failures.
7. Implement request normalization, prior-final lookup, selective omission, and canonical call signatures.
8. Buffer Local Agent decisions that require duplicate validation.
9. Reject one exact duplicate final and request one evidence-grounded revision.
10. Return an explicit quality failure if the revision duplicates again.
11. Block one repeated tool call, request a novel decision, then force a supported final after another duplicate.
12. Re-run the focused tests and confirm they pass.

## Task 3: Persist Native Tool Capability Safely

**Files:**

- Modify: `src/domain.ts`
- Modify: `src/worker/runtimeProfile.test.ts`
- Modify: `src/worker/runtimeProfile.ts`
- Create: `src/worker/nativeToolCapability.test.ts`
- Create: `src/worker/nativeToolCapability.ts`
- Modify: `src/worker/llamaClient.integration.test.ts`
- Modify: `src/worker/llamaClient.ts`
- Modify: `src/models/modelRegistry.ts`
- Modify: `src/provider/localLanguageModelProvider.ts`

1. Add failing tests for chat-template fingerprinting and exact capability fingerprints.
2. Add a failing integration test rejecting a non-loopback worker address.
3. Add a failing integration test proving persisted `unavailable` skips native generation.
4. Run those focused tests and confirm the expected failures.
5. Add the persisted record containing fingerprint, support, and observation time.
6. Build the fingerprint from model SHA-256, worker build, chat-template hash, platform, and protocol version.
7. Add minimal `LlamaClient` support getters and setters.
8. Reject non-loopback base URLs during client construction.
9. Load matching capability before tool generation and persist observed changes afterward.
10. Treat persistence failures as unknown capability without changing locality.
11. Re-run the focused tests and confirm they pass.

## Task 4: Document, Validate, Package, and Commit

**Files:**

- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-21-local-agent-control-repair-design.md`

1. Update Local Agent behavior and privacy documentation.
2. Mark the design implemented only after source validation passes.
3. Run `npm test`.
4. Run `npm run typecheck`.
5. Run `npm run build`.
6. Run `git diff --check`.
7. Run `npm run package:vsix -- darwin-arm64`.
8. Record the exact VSIX path and checksum.
9. Commit the implementation without unrelated changes.
10. Report source validation, package creation, and installed-runtime validation separately.

