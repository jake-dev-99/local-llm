import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import type { AddressInfo } from 'node:net';
import test from 'node:test';
import { build } from 'esbuild';
import type { ChatRequest, ChatStreamEvent } from '../domain';

interface TestLlamaClient {
  dispose(): Promise<void>;
  getNativeToolCallSupport(): 'unknown' | 'available' | 'unavailable';
  setNativeToolCallSupport(support: 'unknown' | 'available' | 'unavailable'): void;
  tokenize(content: string, signal?: AbortSignal): Promise<number>;
  chat(
    request: ChatRequest,
    onEvent: (event: ChatStreamEvent) => void,
    signal?: AbortSignal,
  ): Promise<{ toolCallCount: number }>;
}

type TestLlamaClientConstructor = new (
  baseUrl: string,
  apiKey: string,
  diagnostics?: {
    info(message: string): void;
    warn(message: string): void;
  },
) => TestLlamaClient;

test('the worker client rejects non-loopback addresses', async () => {
  const LlamaClient = await loadLlamaClient();
  assert.throws(
    () => new LlamaClient('http://192.0.2.1:8080', 'test-key'),
    /loopback/i,
  );
});

test('tokenization reuses one worker connection for an ordinary burst', async () => {
  let connectionCount = 0;
  const server = createServer((_request, response) => {
    response.writeHead(200, { 'content-type': 'application/json' });
    response.end('{"tokens":[1]}');
  });
  server.on('connection', () => {
    connectionCount += 1;
  });
  await listen(server);

  try {
    const client = await testClient(server);
    for (let index = 0; index < 50; index += 1) {
      assert.equal(await client.tokenize(`token-${index}`), 1);
    }
    assert.equal(connectionCount, 1);
  } finally {
    server.closeAllConnections();
    await close(server);
  }
});

test('tokenization caches repeated text for the loaded worker', async () => {
  let requestCount = 0;
  const server = createServer((_request, response) => {
    requestCount += 1;
    response.writeHead(200, { 'content-type': 'application/json' });
    response.end('{"tokens":[1,2]}');
  });
  await listen(server);

  try {
    const client = await testClient(server);
    for (let index = 0; index < 100; index += 1) {
      assert.equal(await client.tokenize('repeated tool schema text'), 2);
    }
    assert.equal(requestCount, 1);
  } finally {
    server.closeAllConnections();
    await close(server);
  }
});

test('disposing the client closes its worker connection', async () => {
  let openConnections = 0;
  let resolveConnectionClosed: (() => void) | undefined;
  const connectionClosed = new Promise<void>((resolve) => {
    resolveConnectionClosed = resolve;
  });
  const server = createServer((_request, response) => {
    response.writeHead(200, { 'content-type': 'application/json' });
    response.end('{"tokens":[1]}');
  });
  server.on('connection', (socket) => {
    openConnections += 1;
    socket.on('close', () => {
      openConnections -= 1;
      resolveConnectionClosed?.();
    });
  });
  await listen(server);

  try {
    const client = await testClient(server);
    assert.equal(await client.tokenize('hello'), 1);
    assert.equal(openConnections, 1);
    await client.dispose();
    await connectionClosed;
    assert.equal(openConnections, 0);
  } finally {
    server.closeAllConnections();
    await close(server);
  }
});

test('tokenization continues when the worker retires a keep-alive socket', async () => {
  let connectionCount = 0;
  const server = createServer((_request, response) => {
    response.writeHead(200, { 'content-type': 'application/json' });
    response.end('{"tokens":[1]}');
  });
  server.maxRequestsPerSocket = 5;
  server.on('connection', () => {
    connectionCount += 1;
  });
  await listen(server);

  try {
    const client = await testClient(server);
    for (let index = 0; index < 20; index += 1) {
      assert.equal(await client.tokenize(`token-${index}`), 1);
    }
    assert.equal(connectionCount, 4);
    await client.dispose();
  } finally {
    server.closeAllConnections();
    await close(server);
  }
});

test('chat survives a worker that resets connections at its hard request limit', async () => {
  let connectionCount = 0;
  const requestCounts = new WeakMap<object, number>();
  const server = createServer((request, response) => {
    const requestCount = (requestCounts.get(request.socket) ?? 0) + 1;
    requestCounts.set(request.socket, requestCount);
    if (requestCount >= 100) {
      request.socket.destroy();
      return;
    }
    if (request.url === '/tokenize') {
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end('{"tokens":[1]}');
      return;
    }
    if (request.url === '/v1/chat/completions/input_tokens') {
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end('{"input_tokens":10}');
      return;
    }
    if (request.url === '/v1/chat/completions') {
      response.writeHead(200, { 'content-type': 'text/event-stream' });
      response.end(
        'data: {"choices":[{"delta":{"content":"done"}}]}\n\n' +
          'data: [DONE]\n\n',
      );
      return;
    }
    response.writeHead(404);
    response.end();
  });
  server.on('connection', () => {
    connectionCount += 1;
  });
  await listen(server);

  try {
    const client = await testClient(server);
    try {
      for (let index = 0; index < 99; index += 1) {
        assert.equal(await client.tokenize(`token-${index}`), 1);
      }
      const events: ChatStreamEvent[] = [];
      await client.chat(
        {
          messages: [{ role: 'user', content: 'Finish.' }],
          inputTokenBudget: 100,
          maxTokens: 8,
          temperature: 0,
        },
        (event) => events.push(event),
      );

      assert.deepEqual(events, [{ kind: 'text', text: 'done' }]);
      assert.equal(connectionCount, 2);
    } finally {
      await client.dispose();
    }
  } finally {
    await close(server);
  }
});

test('chat rejects a llama-server error delivered after an SSE stream starts', async () => {
  const server = createServer((request, response) => {
    if (request.url === '/v1/chat/completions/input_tokens') {
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end('{"input_tokens":10}');
      return;
    }
    if (request.url === '/v1/chat/completions') {
      response.writeHead(200, { 'content-type': 'text/event-stream' });
      response.write('data: {"choices":[{"delta":{}}]}\n\n');
      response.end(
        'data: {"error":{"code":500,"message":"Compute error","type":"server_error"}}\n\n',
      );
      return;
    }
    response.writeHead(404);
    response.end();
  });
  await listen(server);

  try {
    const client = await testClient(server);
    await assert.rejects(
      client.chat(
        {
          messages: [{ role: 'user', content: 'Call the probe.' }],
          tools: [{
            type: 'function',
            function: { name: 'local_llm_probe', parameters: { type: 'object' } },
          }],
          toolChoice: 'required',
          inputTokenBudget: 100,
          maxTokens: 8,
          temperature: 0,
        },
        () => undefined,
      ),
      (error: unknown) => {
        assert.ok(error instanceof Error);
        assert.match(
          error.message,
          /Local worker stream \/v1\/chat\/completions failed: Compute error\./,
        );
        assert.equal(error.name, 'LocalWorkerFatalError');
        return true;
      },
    );
  } finally {
    await close(server);
  }
});

test('chat accepts a valid native tool call without running the fallback', async () => {
  let completionCount = 0;
  const server = createServer((request, response) => {
    if (request.url === '/v1/chat/completions/input_tokens') {
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end('{"input_tokens":10}');
      return;
    }
    if (request.url === '/v1/chat/completions') {
      completionCount += 1;
      response.writeHead(200, { 'content-type': 'text/event-stream' });
      response.end(
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-native","function":{"name":"read_file","arguments":"{\\"filePath\\":\\"/workspace/config.ts\\"}"}}]}}]}\n\n' +
          'data: [DONE]\n\n',
      );
      return;
    }
    response.writeHead(404);
    response.end();
  });
  await listen(server);

  try {
    const client = await testClient(server);
    const events: ChatStreamEvent[] = [];
    const result = await client.chat(
      {
        messages: [{ role: 'user', content: 'Review config.ts.' }],
        tools: [{
          type: 'function',
          function: {
            name: 'read_file',
            parameters: {
              type: 'object',
              properties: { filePath: { type: 'string' } },
              required: ['filePath'],
              additionalProperties: false,
            },
          },
        }],
        toolChoice: 'required',
        inputTokenBudget: 100,
        maxTokens: 64,
        temperature: 0,
      },
      (event) => events.push(event),
    );

    assert.equal(completionCount, 1);
    assert.equal(result.toolCallCount, 1);
    assert.deepEqual(events, [{
      kind: 'toolCall',
      id: 'call-native',
      name: 'read_file',
      input: { filePath: '/workspace/config.ts' },
    }]);
  } finally {
    await close(server);
  }
});

test('chat stops repeating the native pass once the worker proves it emits no tool calls', async () => {
  const nativeRequests: string[] = [];
  const fallbackRequests: string[] = [];
  const server = createServer((request, response) => {
    if (request.url === '/v1/chat/completions/input_tokens') {
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end('{"input_tokens":10}');
      return;
    }
    if (request.url === '/v1/chat/completions') {
      let body = '';
      request.on('data', (chunk) => {
        body += chunk;
      });
      request.on('end', () => {
        if ((JSON.parse(body) as { response_format?: unknown }).response_format) {
          fallbackRequests.push(body);
          response.writeHead(200, { 'content-type': 'text/event-stream' });
          response.end(
            `data: ${JSON.stringify({
              choices: [{
                delta: {
                  content: JSON.stringify({
                    kind: 'tool',
                    name: 'read_file',
                    arguments: { filePath: '/workspace/config.ts' },
                  }),
                },
              }],
            })}\n\ndata: [DONE]\n\n`,
          );
          return;
        }
        // The measured Qwen2.5-Coder behaviour: prose that prints the call instead
        // of emitting it on the native tool_calls channel.
        nativeRequests.push(body);
        response.writeHead(200, { 'content-type': 'text/event-stream' });
        response.end(
          'data: {"choices":[{"delta":{"content":"I will use the read_file tool."}}]}\n\n' +
            'data: [DONE]\n\n',
        );
      });
      return;
    }
    response.writeHead(404);
    response.end();
  });
  await listen(server);

  try {
    const client = await testClient(server);
    const request: ChatRequest = {
      messages: [{ role: 'user', content: 'Review config.ts.' }],
      tools: [{
        type: 'function',
        function: {
          name: 'read_file',
          parameters: {
            type: 'object',
            properties: { filePath: { type: 'string' } },
            required: ['filePath'],
            additionalProperties: false,
          },
        },
      }],
      toolChoice: 'required',
      inputTokenBudget: 100,
      maxTokens: 64,
      temperature: 0,
    };

    const first: ChatStreamEvent[] = [];
    await client.chat(request, (event) => first.push(event));
    const second: ChatStreamEvent[] = [];
    await client.chat(request, (event) => second.push(event));

    // The first turn may probe the native path. The second must not repeat a
    // generation the worker has already shown to be useless.
    assert.equal(nativeRequests.length, 1);
    assert.equal(fallbackRequests.length, 2);
    assert.deepEqual(second.map((event) => event.kind), ['toolCall']);
  } finally {
    await close(server);
  }
});

test('persisted unavailable support skips the discarded native probe after restart', async () => {
  let nativeRequests = 0;
  let fallbackRequests = 0;
  const server = createServer((request, response) => {
    if (request.url === '/v1/chat/completions/input_tokens') {
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end('{"input_tokens":10}');
      return;
    }
    if (request.url === '/v1/chat/completions') {
      let body = '';
      request.on('data', (chunk) => {
        body += chunk;
      });
      request.on('end', () => {
        const parsed = JSON.parse(body) as { response_format?: unknown };
        if (!parsed.response_format) {
          nativeRequests += 1;
          response.writeHead(200, { 'content-type': 'text/event-stream' });
          response.end('data: [DONE]\n\n');
          return;
        }
        fallbackRequests += 1;
        response.writeHead(200, { 'content-type': 'text/event-stream' });
        response.end(
          `data: ${JSON.stringify({
            choices: [{
              delta: {
                content: JSON.stringify({
                  kind: 'tool',
                  name: 'read_file',
                  arguments: { filePath: '/workspace/config.ts' },
                }),
              },
            }],
          })}\n\ndata: [DONE]\n\n`,
        );
      });
      return;
    }
    response.writeHead(404);
    response.end();
  });
  await listen(server);

  try {
    const client = await testClient(server);
    client.setNativeToolCallSupport('unavailable');
    await client.chat(
      {
        messages: [{ role: 'user', content: 'Review config.ts.' }],
        tools: [{
          type: 'function',
          function: {
            name: 'read_file',
            parameters: {
              type: 'object',
              properties: { filePath: { type: 'string' } },
              required: ['filePath'],
              additionalProperties: false,
            },
          },
        }],
        toolChoice: 'required',
        inputTokenBudget: 100,
        maxTokens: 64,
        temperature: 0,
      },
      () => undefined,
    );

    assert.equal(nativeRequests, 0);
    assert.equal(fallbackRequests, 1);
    assert.equal(client.getNativeToolCallSupport(), 'unavailable');
  } finally {
    await close(server);
  }
});

test('an automatic final does not prove native tool calls are unavailable', async () => {
  const server = createServer((request, response) => {
    if (request.url === '/v1/chat/completions/input_tokens') {
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end('{"input_tokens":10}');
      return;
    }
    if (request.url === '/v1/chat/completions') {
      let body = '';
      request.on('data', (chunk) => {
        body += chunk;
      });
      request.on('end', () => {
        const parsed = JSON.parse(body) as { response_format?: unknown };
        response.writeHead(200, { 'content-type': 'text/event-stream' });
        response.end(parsed.response_format
          ? `data: ${JSON.stringify({
            choices: [{
              delta: {
                content: JSON.stringify({ kind: 'final' }),
              },
            }],
          })}\n\ndata: [DONE]\n\n`
          : 'data: {"choices":[{"delta":{"content":"Enough evidence."}}]}\n\ndata: [DONE]\n\n');
      });
      return;
    }
    response.writeHead(404);
    response.end();
  });
  await listen(server);

  try {
    const client = await testClient(server);
    await client.chat(
      {
        messages: [{ role: 'user', content: 'Finish the review.' }],
        tools: [{
          type: 'function',
          function: { name: 'read_file', parameters: { type: 'object' } },
        }],
        toolChoice: 'auto',
        inputTokenBudget: 100,
        maxTokens: 64,
        temperature: 0,
      },
      () => undefined,
    );

    assert.equal(client.getNativeToolCallSupport(), 'unknown');
  } finally {
    await close(server);
  }
});

test('persisted available support accepts a native automatic final', async () => {
  let completionCount = 0;
  const server = createServer((request, response) => {
    if (request.url === '/v1/chat/completions/input_tokens') {
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end('{"input_tokens":10}');
      return;
    }
    if (request.url === '/v1/chat/completions') {
      completionCount += 1;
      response.writeHead(200, { 'content-type': 'text/event-stream' });
      response.end(
        'data: {"choices":[{"delta":{"content":"Current final."}}]}\n\n' +
          'data: [DONE]\n\n',
      );
      return;
    }
    response.writeHead(404);
    response.end();
  });
  await listen(server);

  try {
    const client = await testClient(server);
    client.setNativeToolCallSupport('available');
    const events: ChatStreamEvent[] = [];
    const result = await client.chat(
      {
        messages: [{ role: 'user', content: 'Finish the review.' }],
        tools: [{
          type: 'function',
          function: { name: 'read_file', parameters: { type: 'object' } },
        }],
        toolChoice: 'auto',
        inputTokenBudget: 100,
        maxTokens: 64,
        temperature: 0.2,
      },
      (event) => events.push(event),
    );

    assert.equal(completionCount, 1);
    assert.equal(result.toolCallCount, 0);
    assert.deepEqual(events, [{ kind: 'text', text: 'Current final.' }]);
  } finally {
    await close(server);
  }
});

test('the schema-constrained decision streams so it cannot hit a headers timeout', async () => {
  let fallbackBody = '';
  const server = createServer((request, response) => {
    if (request.url === '/v1/chat/completions/input_tokens') {
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end('{"input_tokens":10}');
      return;
    }
    if (request.url === '/v1/chat/completions') {
      let body = '';
      request.on('data', (chunk) => {
        body += chunk;
      });
      request.on('end', () => {
        const parsed = JSON.parse(body) as { response_format?: unknown };
        if (!parsed.response_format) {
          response.writeHead(200, { 'content-type': 'text/event-stream' });
          response.end(
            'data: {"choices":[{"delta":{"content":"I will read the file."}}]}\n\n'
              + 'data: [DONE]\n\n',
          );
          return;
        }
        fallbackBody = body;
        // A non-streaming worker sends no headers until generation ends. At the
        // 1.81 tokens per second measured on a loaded machine, a 2048 token answer
        // outlives Node's 300 second headers timeout.
        response.writeHead(200, { 'content-type': 'text/event-stream' });
        response.end(
          'data: {"choices":[{"delta":{"content":"{\\"kind\\":\\"tool\\","}}]}\n\n'
            + 'data: {"choices":[{"delta":{"content":"\\"name\\":\\"read_file\\","}}]}\n\n'
            + 'data: {"choices":[{"delta":{"content":"\\"arguments\\":{\\"filePath\\":\\"/a.ts\\"}}"}}]}\n\n'
            + 'data: [DONE]\n\n',
        );
      });
      return;
    }
    response.writeHead(404);
    response.end();
  });
  await listen(server);

  try {
    const client = await testClient(server);
    const events: ChatStreamEvent[] = [];
    const result = await client.chat(
      {
        messages: [{ role: 'user', content: 'Review config.ts.' }],
        tools: [{
          type: 'function',
          function: {
            name: 'read_file',
            parameters: {
              type: 'object',
              properties: { filePath: { type: 'string' } },
              required: ['filePath'],
              additionalProperties: false,
            },
          },
        }],
        toolChoice: 'required',
        inputTokenBudget: 100,
        maxTokens: 64,
        temperature: 0,
      },
      (event) => events.push(event),
    );

    assert.equal((JSON.parse(fallbackBody) as { stream?: boolean }).stream, true);
    assert.equal(result.toolCallCount, 1);
    assert.deepEqual(events, [{
      kind: 'toolCall',
      id: events[0]?.kind === 'toolCall' ? events[0].id : '',
      name: 'read_file',
      input: { filePath: '/a.ts' },
    }]);
  } finally {
    await close(server);
  }
});

test('a broken connection names the endpoint and the underlying cause', async () => {
  const server = createServer((request, response) => {
    if (request.url === '/tokenize') {
      request.socket.destroy();
      return;
    }
    response.writeHead(404);
    response.end();
  });
  await listen(server);

  try {
    const client = await testClient(server);
    await assert.rejects(
      client.tokenize('hello'),
      (error: unknown) => {
        assert.ok(error instanceof Error);
        // "fetch failed" alone is useless in a log. The endpoint and the cause
        // underneath it are what make the failure diagnosable.
        assert.match(error.message, /\/tokenize/);
        assert.match(error.message, /caused by/);
        return true;
      },
    );
  } finally {
    await close(server);
  }
});

test('delayed first stream data remains a normal request without a warning', async () => {
  const server = createServer((request, response) => {
    if (request.url === '/v1/chat/completions/input_tokens') {
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end('{"input_tokens":7}');
      return;
    }
    if (request.url === '/v1/chat/completions') {
      response.writeHead(200, { 'content-type': 'text/event-stream' });
      response.flushHeaders();
      setTimeout(() => {
        response.end(
          'data: {"choices":[{"delta":{"content":"OK"}}]}\n\n' +
          'data: [DONE]\n\n',
        );
      }, 40);
      return;
    }
    response.writeHead(404);
    response.end();
  });
  await listen(server);

  try {
    const address = server.address() as AddressInfo;
    const LlamaClient = await loadLlamaClient();
    const info: string[] = [];
    const warnings: string[] = [];
    const client = new LlamaClient(
      `http://127.0.0.1:${address.port}`,
      'test-key',
      {
        info: (message) => info.push(message),
        warn: (message) => warnings.push(message),
      },
    );
    const events: ChatStreamEvent[] = [];

    const result = await client.chat({
      messages: [{ role: 'user', content: 'SECRET PROMPT CONTENT' }],
      toolChoice: 'none',
      inputTokenBudget: 100,
      maxTokens: 16,
      temperature: 0,
    }, (event) => events.push(event));

    assert.equal(result.toolCallCount, 0);
    assert.deepEqual(events, [{ kind: 'text', text: 'OK' }]);
    assert.equal(info.some((message) => /chat-\d+ start.*messages=1.*tools=0.*maxOutputTokens=16/i.test(message)), true);
    assert.equal(info.some((message) => /chat-\d+ response headers.*elapsed=/i.test(message)), true);
    assert.equal(info.some((message) => /chat-\d+ first stream data.*elapsed=/i.test(message)), true);
    assert.equal(info.some((message) => /chat-\d+ complete.*outputCharacters=2.*toolCalls=0.*elapsed=/i.test(message)), true);
    assert.deepEqual(warnings, []);
    assert.equal([...info, ...warnings].some((message) => message.includes('SECRET PROMPT CONTENT')), false);
    await client.dispose();
  } finally {
    server.closeAllConnections();
    await close(server);
  }
});

test('a long generation remains normal while worker stream data is flowing', async () => {
  const server = createServer((request, response) => {
    if (request.url === '/v1/chat/completions/input_tokens') {
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end('{"input_tokens":7}');
      return;
    }
    if (request.url === '/v1/chat/completions') {
      response.writeHead(200, { 'content-type': 'text/event-stream' });
      response.flushHeaders();
      let emitted = 0;
      const interval = setInterval(() => {
        emitted += 1;
        response.write('data: {"choices":[{"delta":{"content":"x"}}]}\n\n');
        if (emitted === 6) {
          clearInterval(interval);
          response.end('data: [DONE]\n\n');
        }
      }, 5);
      return;
    }
    response.writeHead(404);
    response.end();
  });
  await listen(server);

  try {
    const address = server.address() as AddressInfo;
    const LlamaClient = await loadLlamaClient();
    const warnings: string[] = [];
    const client = new LlamaClient(
      `http://127.0.0.1:${address.port}`,
      'test-key',
      {
        info: () => undefined,
        warn: (message) => warnings.push(message),
      },
    );

    await client.chat({
      messages: [{ role: 'user', content: 'hello' }],
      toolChoice: 'none',
      inputTokenBudget: 100,
      maxTokens: 16,
      temperature: 0,
    }, () => undefined);

    assert.deepEqual(warnings, []);
    await client.dispose();
  } finally {
    server.closeAllConnections();
    await close(server);
  }
});

async function testClient(server: ReturnType<typeof createServer>): Promise<TestLlamaClient> {
  const address = server.address() as AddressInfo;
  const LlamaClient = await loadLlamaClient();
  return new LlamaClient(`http://127.0.0.1:${address.port}`, 'test-key');
}

async function listen(server: ReturnType<typeof createServer>): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
}

async function close(server: ReturnType<typeof createServer>): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    server.close((error) => (error ? reject(error) : resolve()));
  });
}

async function loadLlamaClient(): Promise<TestLlamaClientConstructor> {
  const bundled = await build({
    entryPoints: ['src/worker/llamaClient.ts'],
    absWorkingDir: process.cwd(),
    bundle: true,
    format: 'esm',
    platform: 'node',
    target: 'node26',
    banner: {
      js: "import { createRequire as __createRequire } from 'node:module'; const require = __createRequire(process.cwd() + '/');",
    },
    write: false,
  });
  const source = bundled.outputFiles[0]?.contents;
  assert.ok(source, 'esbuild returned the bundled LlamaClient');
  const url = `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`;
  const loaded = await import(url) as { LlamaClient?: TestLlamaClientConstructor };
  assert.ok(loaded.LlamaClient, 'bundled module exports LlamaClient');
  return loaded.LlamaClient;
}
