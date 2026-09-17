import assert from 'node:assert/strict';
import test from 'node:test';
import { PythonWorkerClient, type WorkerTransport } from './pythonWorkerClient.ts';
import { PythonWorkerError } from './pythonWorkerTypes.ts';

interface FakeTransport extends WorkerTransport {
  readonly sent: string[];
  emit(message: unknown): void;
  emitRaw(line: string): void;
  exit(code: number | null, signal?: string | null): void;
}

function fakeTransport(): FakeTransport {
  const sent: string[] = [];
  let lineHandler: (line: string) => void = () => undefined;
  let exitHandler: (code: number | null, signal: string | null) => void = () => undefined;
  return {
    sent,
    send: (line) => { sent.push(line); },
    onLine: (handler) => { lineHandler = handler; },
    onExit: (handler) => { exitHandler = handler; },
    close: () => undefined,
    emit: (message) => lineHandler(`${JSON.stringify(message)}\n`),
    emitRaw: (line) => lineHandler(line),
    exit: (code, signal = null) => exitHandler(code, signal),
  };
}

function lastRequest(transport: FakeTransport): Record<string, unknown> {
  const line = transport.sent.at(-1);
  assert.ok(line, 'expected a request to have been sent');
  return JSON.parse(line) as Record<string, unknown>;
}

test('correlates a response with its own request', async () => {
  const transport = fakeTransport();
  const client = new PythonWorkerClient({ transport });

  const pending = client.inspect('/models/demo');
  const request = lastRequest(transport);
  assert.equal(request['method'], 'model.inspect');
  assert.deepEqual(request['params'], { path: '/models/demo' });

  transport.emit({ id: request['id'], ok: true, result: { modelType: 'qwen3' } });
  assert.deepEqual(await pending, { modelType: 'qwen3' });
});

test('delivers streamed tokens only to the request that asked for them', async () => {
  const transport = fakeTransport();
  const client = new PythonWorkerClient({ transport });
  const chunks: string[] = [];

  const pending = client.chat(
    [{ role: 'user', content: 'hi' }],
    { onToken: (text) => chunks.push(text) },
  );
  const id = lastRequest(transport)['id'];

  transport.emit({ method: 'generation.token', params: { requestId: id, text: 'he' } });
  transport.emit({ method: 'generation.token', params: { requestId: id, text: 'llo' } });
  // A token for an unrelated request must not reach this collector.
  transport.emit({ method: 'generation.token', params: { requestId: 999, text: 'no' } });
  transport.emit({ id, ok: true, result: { text: 'hello', inputTokens: 4 } });

  const result = await pending;
  assert.equal(result.text, 'hello');
  assert.equal(result.inputTokens, 4);
  assert.deepEqual(chunks, ['he', 'llo']);
});

test('requests streaming only when a token sink is supplied', async () => {
  const transport = fakeTransport();
  const client = new PythonWorkerClient({ transport });

  void client.complete('prompt');
  assert.equal(
    (lastRequest(transport)['params'] as Record<string, unknown>)['stream'],
    false,
  );

  void client.complete('prompt', { onToken: () => undefined });
  assert.equal(
    (lastRequest(transport)['params'] as Record<string, unknown>)['stream'],
    true,
  );
});

test('surfaces the worker error code rather than its message text', async () => {
  const transport = fakeTransport();
  const client = new PythonWorkerClient({ transport });

  const pending = client.load('/models/custom');
  const id = lastRequest(transport)['id'];
  transport.emit({
    id,
    ok: false,
    error: { code: 'custom_code_required', message: 'ships custom model code' },
  });

  const error = await pending.then(() => undefined, (reason: unknown) => reason);
  assert.ok(error instanceof PythonWorkerError);
  assert.equal(error.code, 'custom_code_required');
});

test('a crash rejects every in-flight request and reports STOPPED', async () => {
  const transport = fakeTransport();
  const states: string[] = [];
  const client = new PythonWorkerClient({
    transport,
    onState: (state) => states.push(state),
  });

  const first = client.inspect('/models/a');
  const second = client.chat([{ role: 'user', content: 'hi' }]);
  transport.exit(null, 'SIGKILL');

  for (const pending of [first, second]) {
    const error = await pending.then(() => undefined, (reason: unknown) => reason);
    assert.ok(error instanceof PythonWorkerError);
    assert.equal(error.code, 'worker_crashed');
    assert.match(error.message, /SIGKILL/);
  }
  assert.deepEqual(states, ['STOPPED']);
});

test('refuses to issue requests once the worker has exited', async () => {
  const transport = fakeTransport();
  const client = new PythonWorkerClient({ transport });
  transport.exit(1);

  await assert.rejects(
    client.inspect('/models/a'),
    (error: unknown) => error instanceof PythonWorkerError && error.code === 'worker_crashed',
  );
});

test('rejects a worker whose protocol version is not the supported one', async () => {
  const transport = fakeTransport();
  const client = new PythonWorkerClient({ transport });

  const pending = client.runtimeInfo();
  transport.emit({
    id: lastRequest(transport)['id'],
    ok: true,
    result: { protocolVersion: 99, deviceType: 'mps' },
  });

  await assert.rejects(pending, /protocol 99/);
});

test('survives non-protocol output on stdout without failing the request', async () => {
  const transport = fakeTransport();
  const logs: string[] = [];
  const client = new PythonWorkerClient({ transport, onLog: (m) => logs.push(m) });

  const pending = client.inspect('/models/a');
  const id = lastRequest(transport)['id'];
  transport.emitRaw('OMP: Error #15: libomp already initialized\n');
  transport.emit({ id, ok: true, result: { modelType: 'qwen3' } });

  assert.deepEqual(await pending, { modelType: 'qwen3' });
  assert.equal(logs.length, 1);
  assert.match(logs[0] ?? '', /Discarded unparseable/);
});

test('cancel is not sent to a worker that has already exited', () => {
  const transport = fakeTransport();
  const client = new PythonWorkerClient({ transport });
  transport.exit(0);
  const before = transport.sent.length;

  client.cancel();
  assert.equal(transport.sent.length, before);
});

test('cancel names the generation in flight', () => {
  const transport = fakeTransport();
  const client = new PythonWorkerClient({ transport });

  void client.chat([{ role: 'user', content: 'hi' }]);
  const generationId = lastRequest(transport)['id'];

  client.cancel();
  const cancel = lastRequest(transport);
  assert.equal(cancel['method'], 'generation.cancel');
  assert.deepEqual(cancel['params'], { requestId: generationId });
});

test('cancel does nothing when no generation is running', () => {
  const transport = fakeTransport();
  const client = new PythonWorkerClient({ transport });

  // An inspect is in flight, but it is not a generation and must not be
  // stopped by a Stop button aimed at one.
  void client.inspect('/models/a');
  const before = transport.sent.length;

  client.cancel();
  assert.equal(transport.sent.length, before);
});

test('a cancelled generation rejects with generation_cancelled', async () => {
  const transport = fakeTransport();
  const client = new PythonWorkerClient({ transport });
  const chunks: string[] = [];

  const pending = client.chat(
    [{ role: 'user', content: 'hi' }],
    { onToken: (text) => chunks.push(text) },
  );
  const id = lastRequest(transport)['id'];
  transport.emit({ method: 'generation.token', params: { requestId: id, text: 'par' } });
  client.cancel();
  transport.emit({
    id,
    ok: false,
    error: { code: 'generation_cancelled', message: 'Generation was cancelled.' },
  });

  const error = await pending.then(() => undefined, (reason: unknown) => reason);
  assert.ok(error instanceof PythonWorkerError);
  assert.equal(error.code, 'generation_cancelled');
  // Tokens produced before the cancel still reached the caller.
  assert.deepEqual(chunks, ['par']);
});

test('cancel stops naming a generation once it has finished', async () => {
  const transport = fakeTransport();
  const client = new PythonWorkerClient({ transport });

  const pending = client.chat([{ role: 'user', content: 'hi' }]);
  const id = lastRequest(transport)['id'];
  transport.emit({ id, ok: true, result: { text: 'done', inputTokens: 2 } });
  await pending;
  const before = transport.sent.length;

  client.cancel();
  assert.equal(transport.sent.length, before);
});
