import assert from 'node:assert/strict';
import test from 'node:test';
import type { ChatRequest, ChatStreamEvent } from '../domain.ts';
import type { InferenceClient } from './inferenceClient.ts';
import { PythonWorkerClient, type WorkerTransport } from './pythonWorkerClient.ts';
import { PythonWorkerError, type PythonModelInfo } from './pythonWorkerTypes.ts';
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

function harness() {
  const transport = fakeTransport();
  const worker = new PythonWorkerClient({ transport });
  const client: InferenceClient = new TransformersClient(worker, modelInfo);
  return { transport, worker, client };
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

test('refuses a history that already contains tool turns', async () => {
  const { client } = harness();

  await assert.rejects(
    client.chat(chatRequest({
      messages: [
        { role: 'user', content: 'hi' },
        { role: 'tool', content: '{}', tool_call_id: 'call_1' },
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
