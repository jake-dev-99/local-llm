# Thin Local Agent Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove semantic provider controls while preserving local safety, bounded tools, reliable fallback, and normal final-answer generation.

**Architecture:** The provider preserves VS Code history and tool mode. Structured fallback selects only tool or final. Final answers use ordinary streamed generation.

**Tech Stack:** TypeScript, Node tests, VS Code language-model provider, llama.cpp HTTP API, Ajv, esbuild, and VSIX packaging.

**Spec:** `docs/superpowers/specs/2026-08-22-thin-local-agent-provider-design.md`

## Global Constraints

1. Inference remains authenticated and loopback-only.
2. Model downloads remain the only remote operation.
3. The model decides relevance, evidence sufficiency, edit intent, completion, and wording.
4. Local Agent permits eight invoked tools by default.
5. Loaded tools and final generation never consume that allowance.
6. No elapsed-time limit ends progressing work.
7. Ordinary Chat and built-in Agent behavior remain unchanged.
8. Native-tool capability persistence remains fingerprinted and local.
9. No dependency, agent framework, remote fallback, or telemetry is added.
10. Source, package, macOS runtime, and Windows runtime results remain separate.

---

### Task 1: Remove Semantic Provider Controls

**Files:**

1. Modify: `src/provider/localAgentToolChoice.test.ts`
2. Modify: `src/provider/localAgentToolChoice.ts`
3. Modify: `src/provider/localAgentTools.test.ts`
4. Modify: `src/provider/localAgentTools.ts`
5. Modify: `src/provider/localLanguageModelProvider.ts`
6. Delete: `src/provider/localAgentResponse.test.ts`
7. Delete: `src/provider/localAgentResponse.ts`
8. Modify: `agents/local-agent.agent.md`

**Interfaces:**

1. Produce `localAgentToolInvocationCount(messages): number`.
2. Produce `localAgentToolLimitReached(messages, maximum): boolean`.
3. Preserve `isLocalAgentRequest(messages): boolean`.
4. Keep `resolveLocalAgentToolPolicy()` limited to caller mode and the invocation ceiling.

1. [ ] **Step 1: Write failing objective-policy tests**

Replace evidence, mutation, and repetition tests with these cases:

```typescript
test('Local Agent exposes every recognized local tool', () => {
  const supplied = [
    tool('read_file'),
    tool('grep_search'),
    tool('replace_string_in_file'),
    tool('session_store_sql'),
  ];
  assert.deepEqual(
    localAgentAvailableTools(supplied).map((item) => item.function.name),
    ['read_file', 'grep_search', 'replace_string_in_file'],
  );
});

test('automatic mode remains automatic before the ceiling', () => {
  const tools = [tool('read_file'), tool('replace_string_in_file')];
  assert.deepEqual(
    resolveLocalAgentToolPolicy(tools, false, false),
    { tools, toolChoice: 'auto', source: 'caller-auto' },
  );
});

test('the ceiling disables tools but preserves final generation', () => {
  assert.deepEqual(
    resolveLocalAgentToolPolicy([tool('read_file')], true, true),
    { tools: [], toolChoice: 'none', source: 'local-agent-final' },
  );
});
```

Add one invocation-count test with eight assistant tool calls.

Add one final-text test proving the invocation count remains zero.

2. [ ] **Step 2: Verify the focused tests fail**

Run:

```bash
node --disable-warning=MODULE_TYPELESS_PACKAGE_JSON --test src/provider/localAgentToolChoice.test.ts src/provider/localAgentTools.test.ts
```

Expected: failures show forced evidence and request-wording filters remain active.

3. [ ] **Step 3: Reduce Local Agent state to invocation counting**

Keep marker detection and active-request indexing.

Replace semantic state with:

```typescript
export function localAgentToolInvocationCount(
  messages: readonly ChatMessage[],
): number {
  const requestIndex = activeLocalAgentRequestIndex(messages);
  if (!isLocalAgentRequest(messages) || requestIndex < 0) {
    return 0;
  }
  return messages.slice(requestIndex + 1).reduce(
    (count, message) => count + (
      message.role === 'assistant' ? message.tool_calls?.length ?? 0 : 0
    ),
    0,
  );
}

export function localAgentToolLimitReached(
  messages: readonly ChatMessage[],
  maximum: number = DEFAULT_MAX_AGENT_TOOL_ROUNDS,
): boolean {
  return isLocalAgentRequest(messages) &&
    localAgentToolInvocationCount(messages) >= Math.max(1, Math.floor(maximum));
}
```

Delete evidence phases, response comparison, call comparison, and history rewriting helpers.

4. [ ] **Step 4: Replace request parsing with a capability allowlist**

Keep only these Local Agent tools:

```typescript
const LOCAL_AGENT_TOOL_NAMES = new Set([
  'file_search',
  'grep_search',
  'get_errors',
  'insert_edit_into_file',
  'list_dir',
  'read_file',
  'replace_string_in_file',
]);

export function localAgentAvailableTools(tools: readonly ChatTool[]): ChatTool[] {
  return tools.filter((tool) => LOCAL_AGENT_TOOL_NAMES.has(tool.function.name));
}
```

Use this complete policy:

```typescript
export function resolveLocalAgentToolPolicy(
  tools: readonly ChatTool[],
  callerRequired: boolean,
  forceFinal: boolean,
): LocalAgentToolPolicy {
  if (forceFinal) {
    return { tools: [], toolChoice: 'none', source: 'local-agent-final' };
  }
  if (callerRequired) {
    return { tools: [...tools], toolChoice: 'required', source: 'caller-required' };
  }
  return { tools: [...tools], toolChoice: 'auto', source: 'caller-auto' };
}
```

Delete mutation regular expressions and edit-after-read enforcement.

5. [ ] **Step 5: Restore direct provider behavior**

Compute only the Local Agent ceiling and allowlisted tools:

```typescript
const localAgentRequest = isLocalAgentRequest(adaptedMessages);
const forceLocalAgentFinal = localAgentRequest && localAgentToolLimitReached(
  adaptedMessages,
  config.maxAgentToolRounds,
);
const availableTools = localAgentRequest
  ? localAgentAvailableTools(tools ?? [])
  : tools ?? [];
```

Build model messages directly from `adaptedMessages`.

Call `client.chat()` once and stream every event through `reportChatEvent()`.

Keep native capability loading and persistence unchanged.

Remove controller messages, history removal, buffering, retries, and replacement responses.

Delete `localAgentResponse.ts` and its test file.

6. [ ] **Step 6: Tighten Local Agent instructions**

Keep the existing frontmatter and tool selectors.

Replace the body with these rules:

```markdown
1. Answer directly when the request needs no workspace evidence.
2. Use current workspace evidence for workspace-dependent claims.
3. Read relevant content before editing.
4. Edit only when the user requests workspace changes.
5. Respect explicit read-only and no-change requests.
6. Continue only while another tool can improve the result.
7. Never print tool calls as ordinary text.
8. Never claim success without a supporting tool result.
9. State unfinished work and uncertainty plainly.
10. Never request remote, terminal, memory, subagent, or network tools.
```

7. [ ] **Step 7: Validate and commit**

Run:

```bash
node --disable-warning=MODULE_TYPELESS_PACKAGE_JSON --test src/provider/localAgentToolChoice.test.ts src/provider/localAgentTools.test.ts
npm run typecheck
git add agents src/provider
git commit -m "Simplify Local Agent provider control"
```

Expected: focused tests and type checking pass before the commit.

---

### Task 2: Separate Tool Decisions From Final Answers

**Files:**

1. Create: `src/worker/toolProtocol.test.ts`
2. Modify: `src/worker/toolProtocol.ts`
3. Modify: `src/worker/llamaClient.integration.test.ts`
4. Modify: `src/worker/llamaClient.schema.integration.test.ts`
5. Modify: `src/worker/llamaClient.ts`

**Interfaces:**

1. Change final decisions to `{ kind: 'final' }`.
2. Keep tool decisions schema-valid.
3. Keep `LlamaClient.chat()` and `ChatResult` signatures unchanged.
4. Start ordinary streamed generation after a final decision.

1. [ ] **Step 1: Write failing protocol tests**

Create tests proving:

```typescript
assert.deepEqual(
  parseToolDecision('{"kind":"final"}', tools, false),
  { kind: 'final' },
);

assert.throws(
  () => parseToolDecision(
    '{"kind":"final","text":"Schema-generated answer"}',
    tools,
    false,
  ),
  /violates the supplied tool schema/,
);

assert.throws(
  () => parseToolDecision('{"kind":"final"}', tools, true),
  /violates the supplied tool schema/,
);
```

2. [ ] **Step 2: Replace the automatic-final integration test**

Use the existing loopback test server.

Set native capability to `unavailable`.

Return `{"kind":"final"}` for the structured request.

Return `Current evidence supports the answer.` for the following ordinary request.

Assert:

```typescript
assert.equal(completionBodies.length, 2);
assert.equal(completionBodies[0]?.temperature, 0);
assert.equal(completionBodies[0]?.max_tokens, 64);
assert.ok(completionBodies[0]?.response_format);
assert.equal(completionBodies[1]?.temperature, 0.2);
assert.equal(completionBodies[1]?.max_tokens, 256);
assert.equal(completionBodies[1]?.response_format, undefined);
assert.equal(completionBodies[1]?.tools, undefined);
assert.deepEqual(events, [{
  kind: 'text',
  text: 'Current evidence supports the answer.',
}]);
```

3. [ ] **Step 3: Verify focused tests fail**

Run:

```bash
node --disable-warning=MODULE_TYPELESS_PACKAGE_JSON --test src/worker/toolProtocol.test.ts src/worker/llamaClient.integration.test.ts src/worker/llamaClient.schema.integration.test.ts
```

Expected: final decisions still require answer text, so ordinary final generation never starts.

4. [ ] **Step 4: Make structured fallback select actions only**

Change the decision type:

```typescript
export type ToolDecision =
  | { kind: 'tool'; name: string; arguments: Record<string, unknown> }
  | { kind: 'final' };
```

Change the final schema to require only `kind`.

Reject additional final-decision properties.

Change the automatic instruction to:

```text
Return one action matching the response schema. Choose kind tool when another tool is needed. Otherwise choose kind final.
```

5. [ ] **Step 5: Run ordinary generation after final selection**

Keep the existing validated tool-call branch.

Use this final branch:

```typescript
const {
  tools: _tools,
  workerToolChoice: _workerToolChoice,
  ...withoutTools
} = request;
return this.streamNativeChat(
  { ...withoutTools, toolChoice: 'none' },
  onEvent,
  signal,
);
```

Limit automatic and required structured decisions with `toolCallMaxTokens`.

Keep structured temperature zero.

Use configured chat temperature for ordinary final generation.

6. [ ] **Step 6: Accept valid native automatic finals**

When persisted native capability is `available`, accept a native automatic response without a tool call.

Use this condition after native generation:

```typescript
const nativeAutomaticFinal = toolChoice === 'auto' &&
  this.nativeToolCalls === 'available' &&
  nativeResult.toolCallCount === 0;
if (!toolProtocolEnabled || nativeResult.toolCallCount > 0 || nativeAutomaticFinal) {
  for (const event of nativeEvents) {
    onEvent(event);
  }
  return nativeResult;
}
```

Unknown capability still uses one structured action decision after a missing native call.

7. [ ] **Step 7: Validate and commit**

Run:

```bash
node --disable-warning=MODULE_TYPELESS_PACKAGE_JSON --test src/worker/toolProtocol.test.ts src/worker/llamaClient.integration.test.ts src/worker/llamaClient.schema.integration.test.ts src/worker/nativeToolCapability.test.ts
npm test
npm run typecheck
git add src/worker
git commit -m "Separate tool decisions from final answers"
```

Expected: focused tests, complete tests, and type checking pass before the commit.

---

### Task 3: Document, Build, and Package Version 0.3.4

**Files:**

1. Modify: `README.md`
2. Modify: `package.json`
3. Modify: `package-lock.json`
4. Modify: `docs/superpowers/specs/2026-08-22-thin-local-agent-provider-design.md`

1. [ ] **Step 1: Correct documentation and settings**

Document automatic tool selection, complete local tool availability, and the eight-invocation maximum.

Document structured action selection followed by ordinary final generation.

Remove forced-evidence, history-rewriting, and repeated-answer rejection claims.

Set version `0.3.4` in both package files.

Use these setting descriptions:

```json
"Maximum tool invocations per active bundled Local Agent request. Loaded definitions and final generation do not count."
```

```json
"Maximum output tokens for schema-constrained tool action decisions. Final answers use maxOutputTokens."
```

2. [ ] **Step 2: Run complete source validation**

Run:

```bash
npm test
npm run typecheck
npm run build
git diff --check
```

Expected: every command passes.

3. [ ] **Step 3: Package macOS and record its checksum**

Run:

```bash
npm run package:vsix -- darwin-arm64
shasum -a 256 local-llm-engine-0.3.4-darwin-arm64.vsix
```

Expected: the package exists with one recorded SHA-256 checksum.

4. [ ] **Step 4: Mark source status accurately and commit**

Set design status to implemented and source-validated.

Keep macOS and Windows runtime validation pending.

Run:

```bash
git add README.md package.json package-lock.json docs/superpowers
git commit -m "Document thin Local Agent provider"
```

Report tests, build, package, checksum, commits, and runtime status separately.

---

### Task 4: Validate the Installed Product

**Artifact:** `local-llm-engine-0.3.4-darwin-arm64.vsix`

1. [ ] **Step 1: Install and reload the exact macOS package**

```bash
code --install-extension local-llm-engine-0.3.4-darwin-arm64.vsix --force
```

2. [ ] **Step 2: Test direct conversation**

Send `Reply Hi` twice.

Expected: no workspace tools, two valid answers, and no replacement failure.

3. [ ] **Step 3: Test repeated workspace analysis**

Send this request twice inside one conversation:

```text
Audit extension.ts for remaining error-handling gaps. Do not change files.
```

Expected: model-chosen evidence, no forced rounds, no edits, and specific evidence-backed answers.

Matching supported findings remain acceptable.

4. [ ] **Step 4: Test explicit editing**

Use a disposable file and request one read followed by one edit.

Expected: edit tools remain available, the model reads first, and VS Code displays approval.

5. [ ] **Step 5: Test objective boundaries**

Confirm a ninth Local Agent invocation never occurs.

Confirm final generation still occurs after eight invocations.

Confirm progressing work has no elapsed-time cancellation.

Restart the worker and confirm matching unavailable capability skips native probing.

Confirm inference targets only authenticated `127.0.0.1` endpoints.

6. [ ] **Step 6: Validate Windows separately**

Package `win32-x64`, install it on the Intel Arc machine, and repeat Steps 2 through 5.

Record that Windows validation used the current CPU worker.

7. [ ] **Step 7: Resume dynamic-capacity implementation**

Return to `docs/superpowers/specs/2026-08-20-dynamic-capacity-management-design.md` after macOS runtime acceptance.
