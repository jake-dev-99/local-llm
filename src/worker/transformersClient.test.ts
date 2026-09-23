import assert from 'node:assert/strict';
import test from 'node:test';
import type { ChatRequest, ChatStreamEvent, ChatTool } from '../domain.ts';
import type { InferenceClient } from './inferenceClient.ts';
import { PythonWorkerClient, type WorkerTransport } from './pythonWorkerClient.ts';
import { PythonWorkerError, type PythonModelInfo } from './pythonWorkerTypes.ts';
import { toolDecisionJsonSchema } from './toolProtocol.ts';
import { TransformersClient } from './transformersClient.ts';

interface FakeTransport extends WorkerTransport {
  readonly sent: Array<Record<string, unknown>>;
  emit(message: unknown): void;
}

function fakeTransport(): FakeTransport {
  const sent: Array<Record<string, unknown>> = [];
  let lineHandler: (line: string) => void = () => undefined;
  return {
    sent,
    send: (line) => { sent.push(JSON.parse(line) as Record<string, unknown>); },
    onLine: (handler) => { lineHandler = handler; },
    onExit: () => undefined,
    close: () => undefined,
    emit: (message) => lineHandler(`${JSON.stringify(message)}\n`),
  };
}

function requestsOf(transport: FakeTransport, method: string) {
  return transport.sent.filter((message) => message['method'] === method);
}

const modelInfo: PythonModelInfo = {
  path: '/models/demo',
  modelType: 'qwen3_moe',
  architecture: 'Qwen3MoeForCausalLM',
  modelClass: 'Qwen3MoeForCausalLM',
  tokenizerClass: 'Qwen2Tokenizer',
  dtype: 'torch.bfloat16',
  isEncoderDecoder: false,
  supportsChat: true,
  contextLength: 262144,
  deviceMap: null,
  capabilities: {
    completion: true, chat: true, streaming: true, encoderDecoder: false,
    customCodeRequired: false, quantized: false, adapter: false,
    contextLength: 262144,
  },
  runtime: {
    protocolVersion: 1, platform: 'macOS', pythonVersion: '3.13.0',
    deviceType: 'mps', deviceName: 'Apple Metal', backend: 'mps',
    versions: { transformers: '5.17.0' },
  },
};

function harness(info: PythonModelInfo = modelInfo) {
  const transport = fakeTransport();
  const worker = new PythonWorkerClient({ transport });
  const client: InferenceClient = new TransformersClient(worker, info);
  return { transport, worker, client };
}

const grammarInfo: PythonModelInfo = {
  ...modelInfo,
  runtime: {
    ...modelInfo.runtime,
    versions: { ...modelInfo.runtime.versions, xgrammar: '0.2.7' },
  },
};

function grammarHarness() {
  return harness(grammarInfo);
}

const readFileTool: ChatTool = {
  type: 'function',
  function: {
    name: 'read_file',
    description: 'Read a file.',
    parameters: {
      type: 'object',
      properties: { path: { type: 'string' } },
      required: ['path'],
      additionalProperties: false,
    },
  },
};

function toolRequest(overrides: Partial<ChatRequest> = {}): ChatRequest {
  return chatRequest({
    tools: [readFileTool],
    toolChoice: 'required',
    toolCallMaxTokens: 64,
    ...overrides,
  });
}

function tick(): Promise<void> {
  return new Promise((resolve) => setImmediate(resolve));
}

/** Answers the decision's tokenize step, then yields until generate.chat is sent. */
async function answerTokenize(transport: FakeTransport, tokens: number): Promise<void> {
  const tokenize = requestsOf(transport, 'model.tokenize').at(-1);
  assert.ok(tokenize, 'decision input was counted');
  transport.emit({ id: tokenize['id'], ok: true, result: { tokens } });
  await tick();
  await tick();
}

function chatRequest(overrides: Partial<ChatRequest> = {}): ChatRequest {
  return {
    messages: [{ role: 'user', content: 'hello' }],
    inputTokenBudget: 1000,
    maxTokens: 256,
    temperature: 0.2,
    ...overrides,
  };
}

test('streams tokens through as text events', async () => {
  const { transport, client } = harness();
  const events: ChatStreamEvent[] = [];

  const pending = client.chat(chatRequest(), (event) => events.push(event));
  const generation = requestsOf(transport, 'generate.chat')[0];
  assert.ok(generation);
  const id = generation['id'];

  transport.emit({ method: 'generation.token', params: { requestId: id, text: 'he' } });
  transport.emit({ method: 'generation.token', params: { requestId: id, text: 'llo' } });
  transport.emit({ id, ok: true, result: { text: 'hello', inputTokens: 7 } });

  const result = await pending;
  assert.deepEqual(events, [
    { kind: 'text', text: 'he' },
    { kind: 'text', text: 'llo' },
  ]);
  assert.equal(result.textCharacters, 5);
  assert.equal(result.toolCallCount, 0);
  assert.equal(result.inputTokens, 7);
});

test('passes the request budget through as generation options', async () => {
  const { transport, client } = harness();

  void client.chat(chatRequest({ maxTokens: 64, temperature: 0 }), () => undefined);
  const params = requestsOf(transport, 'generate.chat')[0]?.['params'] as Record<string, unknown>;

  assert.deepEqual(params['options'], { maxNewTokens: 64, temperature: 0 });
  assert.deepEqual(params['messages'], [{ role: 'user', content: 'hello' }]);
  assert.equal(params['stream'], true);
});

test('strips fields the chat template does not take', async () => {
  const { transport, client } = harness();

  void client.chat(chatRequest({
    messages: [{ role: 'user', content: 'hi' }],
    // The worker's template owns formatting; anything extra is not its input.
    toolChoice: 'none',
  }), () => undefined);
  const params = requestsOf(transport, 'generate.chat')[0]?.['params'] as Record<string, unknown>;
  const messages = params['messages'] as Array<Record<string, unknown>>;

  assert.deepEqual(Object.keys(messages[0] ?? {}).sort(), ['content', 'role']);
});

test('refuses a request carrying tool definitions', async () => {
  const { client } = harness();

  await assert.rejects(
    client.chat(chatRequest({
      tools: [{ type: 'function', function: { name: 'readFile' } }],
    }), () => undefined),
    (error: unknown) => error instanceof PythonWorkerError &&
      error.code === 'unsupported_grammar',
  );
});

test('a history with a tool result answers with final text', async () => {
  const { transport, client } = grammarHarness();
  const events: ChatStreamEvent[] = [];

  const pending = client.chat(toolRequest({
    toolChoice: 'auto',
    messages: [
      { role: 'user', content: 'Get the value.' },
      {
        role: 'assistant',
        content: '',
        tool_calls: [{
          id: 'call_1',
          type: 'function',
          function: { name: 'read_file', arguments: '{"path":"a.txt"}' },
        }],
      },
      { role: 'tool', content: 'ok', tool_call_id: 'call_1' },
    ],
  }), (event) => events.push(event));
  const generation = requestsOf(transport, 'generate.chat')[0];
  assert.ok(generation);
  const sent = (generation['params'] as Record<string, unknown>)['messages'] as Array<Record<string, unknown>>;
  const options = (generation['params'] as Record<string, unknown>)['options'] as Record<string, unknown>;
  assert.ok(!('jsonSchema' in options), 'continuation is unconstrained');
  assert.equal(options['enableThinking'], true, 'continuation opens the reasoning channel');
  // The call record travels with the history: without it the tool result
  // arrives unattributed and the model cannot use it.
  const assistant = sent.find((message) => message['role'] === 'assistant');
  assert.ok(Array.isArray(assistant?.['tool_calls']), 'tool_calls reach the template');
  // Templates take the parsed mapping, not the OpenAI wire string.
  assert.deepEqual(
    (assistant?.['tool_calls'] as Array<Record<string, unknown>>)[0],
    {
      id: 'call_1',
      type: 'function',
      function: { name: 'read_file', arguments: { path: 'a.txt' } },
    },
  );
  const tool = sent.find((message) => message['role'] === 'tool');
  assert.equal(tool?.['tool_call_id'], 'call_1');

  transport.emit({ id: generation['id'], ok: true, result: { text: 'The value is ok.', inputTokens: 20 } });
  const result = await pending;

  assert.equal(result.toolCallCount, 0);
  assert.deepEqual(events, [{ kind: 'text', text: 'The value is ok.' }]);
});

test('a dangling tool call with no result is refused', async () => {
  const { client } = grammarHarness();

  // No tools on the request, so this is a plain turn: the half-turn history
  // (a call record with no result) is malformed input, not a decision.
  await assert.rejects(
    client.chat(chatRequest({
      messages: [
        { role: 'user', content: 'hi' },
        {
          role: 'assistant',
          content: '',
          tool_calls: [{
            id: 'call_1',
            type: 'function',
            function: { name: 'read_file', arguments: '{}' },
          }],
        },
      ],
    }), () => undefined),
    /cannot use tools yet/,
  );
});

test('sends nothing to the worker when a request is refused', async () => {
  const { transport, client } = harness();

  await client.chat(chatRequest({
    tools: [{ type: 'function', function: { name: 'readFile' } }],
  }), () => undefined).catch(() => undefined);

  assert.equal(requestsOf(transport, 'generate.chat').length, 0);
});

test('an abort asks the worker to cancel', async () => {
  const { transport, client } = harness();
  const controller = new AbortController();

  const pending = client.chat(chatRequest(), () => undefined, controller.signal);
  const id = requestsOf(transport, 'generate.chat')[0]?.['id'];
  controller.abort();

  const cancel = requestsOf(transport, 'generation.cancel')[0];
  assert.ok(cancel, 'a cancel was sent');
  assert.deepEqual(cancel['params'], { requestId: id });

  transport.emit({
    id, ok: false,
    error: { code: 'generation_cancelled', message: 'Generation was cancelled.' },
  });
  await assert.rejects(pending, (error: unknown) =>
    error instanceof PythonWorkerError && error.code === 'generation_cancelled');
});

test('the prompt token count rides with the response, costing no round trip', async () => {
  const { transport, client } = harness();

  const pending = client.chat(chatRequest(), () => undefined);
  const id = requestsOf(transport, 'generate.chat')[0]?.['id'];
  transport.emit({ id, ok: true, result: { text: 'done', inputTokens: 11 } });

  const result = await pending;
  assert.equal(result.inputTokens, 11);
  assert.equal(result.textCharacters, 4);
  // The worker already knew the count, so nothing was asked for separately.
  assert.equal(requestsOf(transport, 'model.tokenize').length, 0);
});

test('a worker that streamed nothing still delivers its completion', async () => {
  const { transport, client } = harness();
  const events: ChatStreamEvent[] = [];

  const pending = client.chat(chatRequest(), (event) => events.push(event));
  const id = requestsOf(transport, 'generate.chat')[0]?.['id'];
  transport.emit({ id, ok: true, result: { text: 'unstreamed answer', inputTokens: 3 } });

  await pending;
  assert.deepEqual(events, [{ kind: 'text', text: 'unstreamed answer' }]);
});

test('counts a bare string through the tokenizer', async () => {
  const { transport, client } = harness();

  const pending = client.tokenize('some text');
  const request = requestsOf(transport, 'model.tokenize')[0];
  assert.deepEqual(request?.['params'], { text: 'some text' });

  transport.emit({ id: request?.['id'], ok: true, result: { tokens: 2 } });
  assert.equal(await pending, 2);
});

test('the profile reports no tool support until a grammar backend ships', async () => {
  const { client } = harness();
  const profile = await client.getModelProfile();

  assert.equal(profile.supportsTools, false);
  assert.equal(profile.supportsToolCalls, false);
  assert.equal(profile.loadedContextSize, 262144);
  assert.equal(profile.hasChatTemplate, true);
  assert.match(profile.workerBuild ?? '', /transformers 5\.17\.0/);
});

test('declares the capabilities this runtime does not have', () => {
  const { client } = harness();
  assert.equal(client.supports.infill, false);
  assert.equal(client.supports.constrainedDecoding, false);
  assert.equal(client.infill, undefined);
});

test('the profile reports tool support when the grammar backend is present', async () => {
  const { client } = grammarHarness();
  assert.equal(client.supports.constrainedDecoding, true);
  const profile = await client.getModelProfile();

  assert.equal(profile.supportsTools, true);
  assert.equal(profile.supportsToolCalls, true);
});

test('a required decision emits one validated tool call', async () => {
  const { transport, client } = grammarHarness();
  const events: ChatStreamEvent[] = [];
  const decision = '{"kind":"tool","name":"read_file","arguments":{"path":"a.txt"}}';

  const pending = client.chat(toolRequest(), (event) => events.push(event));
  await answerTokenize(transport, 41);
  const generation = requestsOf(transport, 'generate.chat')[0];
  assert.ok(generation);
  const params = generation['params'] as Record<string, unknown>;
  const options = params['options'] as Record<string, unknown>;
  assert.equal(options['temperature'], 0);
  assert.equal(options['maxNewTokens'], 64);
  assert.deepEqual(options['jsonSchema'], toolDecisionJsonSchema([readFileTool], true));
  const messages = params['messages'] as Array<Record<string, unknown>>;
  assert.match(String(messages.at(-1)?.['content'] ?? ''), /Available tools/);

  transport.emit({ id: generation['id'], ok: true, result: { text: decision, inputTokens: 41 } });
  const result = await pending;

  assert.equal(result.toolCallCount, 1);
  assert.equal(result.inputTokens, 41);
  assert.deepEqual(events, [{
    kind: 'toolCall',
    id: (events[0] as { id: string }).id,
    name: 'read_file',
    input: { path: 'a.txt' },
  }]);
});

test('a decision over budget fails before generating', async () => {
  const { transport, client } = grammarHarness();

  const pending = client.chat(toolRequest({ inputTokenBudget: 10 }), () => undefined);
  // Attach before answering: the refusal fires while yielding, and an
  // unhandled rejection is fatal before the assertion would attach.
  const asserted = assert.rejects(pending, /input budget/);
  await answerTokenize(transport, 41);

  await asserted;
  assert.equal(requestsOf(transport, 'generate.chat').length, 0);
});

test('an empty decision is an error, not an empty answer', async () => {
  const { transport, client } = grammarHarness();

  const pending = client.chat(toolRequest(), () => undefined);
  await answerTokenize(transport, 41);
  const generation = requestsOf(transport, 'generate.chat')[0];
  assert.ok(generation);
  transport.emit({ id: generation['id'], ok: true, result: { text: '  ', inputTokens: 41 } });

  await assert.rejects(pending, /no decision/);
});

test('an automatic final answers with tools disabled', async () => {
  const { transport, client } = grammarHarness();
  const events: ChatStreamEvent[] = [];

  const pending = client.chat(toolRequest({ toolChoice: 'auto' }), (event) => events.push(event));
  await answerTokenize(transport, 41);
  const decision = requestsOf(transport, 'generate.chat')[0];
  assert.ok(decision);
  transport.emit({ id: decision['id'], ok: true, result: { text: '{"kind":"final"}', inputTokens: 41 } });
  await tick();
  await tick();
  const final = requestsOf(transport, 'generate.chat')[1];
  assert.ok(final, 'a final answer was requested');
  const params = final['params'] as Record<string, unknown>;
  assert.ok(!('jsonSchema' in (params['options'] as Record<string, unknown>)), 'final is unconstrained');

  transport.emit({ id: final['id'], ok: true, result: { text: 'done', inputTokens: 12 } });
  const result = await pending;

  assert.equal(result.toolCallCount, 0);
  assert.deepEqual(events, [{ kind: 'text', text: 'done' }]);
});
