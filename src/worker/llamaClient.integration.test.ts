import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import type { AddressInfo } from 'node:net';
import test from 'node:test';
import { build } from 'esbuild';
import type { ChatRequest, ChatStreamEvent } from '../domain.js';

interface TestLlamaClient {
  dispose(): Promise<void>;
  getNativeToolCallSupport(): 'unknown' | 'available' | 'unavailable';
  setNativeToolCallSupport(support: 'unknown' | 'available' | 'unavailable'): void;
  tokenize(content: string, signal?: AbortSignal): Promise<number>;
  chat(
    request: ChatRequest,
    onEvent: (event: ChatStreamEvent) => void,
    signal?: AbortSignal,
  ): Promise<{ toolCallCount: number; tokensPerSecond?: number }>;
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

test('a thinking model cut off mid-reasoning is not recorded as lacking native tool calls', async () => {
  const thinking = (tokens: string[]) =>
    tokens.map((reasoning_content) => `data: ${JSON.stringify({ choices: [{ delta: { reasoning_content } }] })}\n\n`).join('') +
    `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: 'length' }] })}\n\ndata: [DONE]\n\n`;
  const server = createServer((request, response) => {
    request.resume();
    request.on('end', () => {
      if (request.url === '/v1/chat/completions/input_tokens') {
        response.writeHead(200, { 'content-type': 'application/json' });
        response.end('{"input_tokens":10}');
        return;
      }
      response.writeHead(200, { 'content-type': 'text/event-stream' });
      response.end(thinking(['The user wants', ' the probe called', ' with value ok.']));
    });
  });
  await listen(server);
  const client = await testClient(server);
  try {
    await assert.rejects(
      client.chat({
        messages: [{ role: 'user', content: 'Call local_llm_probe with value set to ok.' }],
        tools: [{
          type: 'function',
          function: { name: 'local_llm_probe', parameters: { type: 'object' } },
        }],
        toolChoice: 'required',
        inputTokenBudget: 1_000,
        maxTokens: 128,
        toolCallMaxTokens: 128,
        temperature: 0,
      }, () => undefined),
      // The native pass already showed reasoning, so the decision ran on the
      // chat output limit, and that is the setting to raise.
      /spent its whole tool-decision limit \(128 tokens\) reasoning.*localLlm\.maxOutputTokens/s,
    );
    assert.equal(
      client.getNativeToolCallSupport(),
      'unknown',
      'running out of tokens while thinking says nothing about the native tool-call channel',
    );
  } finally {
    await client.dispose();
    server.closeAllConnections();
    await close(server);
  }
});

test('once a model has shown it reasons, its tool decision gets the chat output budget', async () => {
  // Observed: Qwen3 4B Thinking reasoned through all 512 maxToolCallTokens of
  // the schema decision and never reached the JSON.
  const fallbackBudgets: number[] = [];
  const server = createServer((request, response) => {
    let raw = '';
    request.on('data', (chunk) => { raw += chunk; });
    request.on('end', () => {
      if (request.url === '/v1/chat/completions/input_tokens') {
        response.writeHead(200, { 'content-type': 'application/json' });
        response.end('{"input_tokens":10}');
        return;
      }
      const body = JSON.parse(raw) as { response_format?: unknown; max_tokens: number };
      response.writeHead(200, { 'content-type': 'text/event-stream' });
      const frame = (delta: object, finish_reason?: string) =>
        `data: ${JSON.stringify({ choices: [{ delta, ...(finish_reason ? { finish_reason } : {}) }] })}\n\n`;
      if (body.response_format) {
        fallbackBudgets.push(body.max_tokens);
        response.end(
          frame({ reasoning_content: 'I should read the file.' }) +
          frame({ content: JSON.stringify({ kind: 'tool', name: 'read_file', arguments: { filePath: 'a.ts' } }) }, 'stop') +
          'data: [DONE]\n\n',
        );
        return;
      }
      response.end(
        frame({ reasoning_content: 'Let me think about which file.' }) +
        frame({ content: 'I will look at a.ts.' }, 'stop') +
        'data: [DONE]\n\n',
      );
    });
  });
  await listen(server);
  const client = await testClient(server);
  try {
    const events: ChatStreamEvent[] = [];
    await client.chat({
      messages: [{ role: 'user', content: 'Review a.ts.' }],
      tools: [{
        type: 'function',
        function: {
          name: 'read_file',
          parameters: {
            type: 'object',
            properties: { filePath: { type: 'string' } },
            required: ['filePath'],
          },
        },
      }],
      toolChoice: 'auto',
      inputTokenBudget: 1_000,
      maxTokens: 16_384,
      toolCallMaxTokens: 512,
      temperature: 0,
    }, (event) => events.push(event));

    assert.deepEqual(fallbackBudgets, [16_384]);
    assert.deepEqual(events.map((event) => event.kind), ['toolCall']);
  } finally {
    await client.dispose();
    server.closeAllConnections();
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
          'data: {"choices":[],"timings":{"cache_n":3,"prompt_n":4,"prompt_ms":20,"prompt_per_second":200,"predicted_n":1,"predicted_ms":5,"predicted_per_second":200}}\n\n' +
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
    assert.equal(result.tokensPerSecond, 200);
    assert.deepEqual(events, [{ kind: 'text', text: 'OK' }]);
    assert.equal(info.some((message) => /\[Generating Response\] chat-\d+ start.*messages=1.*tools=0.*maxOutputTokens=16/i.test(message)), true);
    assert.equal(info.some((message) => /\[Generating Response\] chat-\d+ response headers.*elapsed=/i.test(message)), true);
    assert.equal(info.some((message) => /\[Generating Response\] chat-\d+ first stream data.*elapsed=/i.test(message)), true);
    assert.equal(info.some((message) => /\[Generating Response\] chat-\d+ timings \(native\).*prompt=4 processed \+ 3 cached tokens.*200\.00 tokens\/s.*output=1 token.*200\.00 tokens\/s/i.test(message)), true);
    assert.equal(info.some((message) => /\[Generating Response\] chat-\d+ complete.*outputCharacters=2.*toolCalls=0.*elapsed=/i.test(message)), true);
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

const finalRequest: ChatRequest = {
  messages: [
    { role: 'system', content: 'Use tools to make requested edits. Report verified results.' },
    { role: 'user', content: 'Update example.cls.' },
    { role: 'assistant', content: '', tool_calls: [{
      id: 'edit-1', type: 'function',
      function: { name: 'replace_string_in_file', arguments: '{"filePath":"example.cls"}' },
    }] },
    { role: 'tool', tool_call_id: 'edit-1', content: 'Edit applied.' },
  ],
  tools: [{ type: 'function', function: {
    name: 'replace_string_in_file',
    parameters: { type: 'object', properties: { filePath: { type: 'string' } }, required: ['filePath'] },
  } }],
  toolChoice: 'none', inputTokenBudget: 1000, maxTokens: 2048, temperature: 0,
};

test('final generation preserves the tool and history prefix and appends final-only guidance', async () => {
  await withFinalWorker([{ content: 'Updated example.cls.' }], async (client, bodies) => {
    const events: ChatStreamEvent[] = [];
    await client.chat(finalRequest, event => events.push(event));
    const body = bodies[0]!;
    assert.deepEqual(body.tools, finalRequest.tools);
    assert.equal(body.tool_choice, 'none');
    const messages = body.messages as ChatRequest['messages'];
    assert.deepEqual(messages.slice(0, -1), finalRequest.messages.slice(0, -1));
    assert.equal(messages.length, finalRequest.messages.length);
    assert.equal(messages.at(-1)?.role, 'tool');
    assert.equal(messages.at(-1)?.tool_call_id, 'edit-1');
    assert.ok(messages.at(-1)?.content.startsWith('Edit applied.\n\n'));
    assert.match(messages.at(-1)?.content ?? '', /do not call tools/i);
    assert.equal(finalRequest.messages.length, 4, 'caller history must not be mutated');
    assert.equal(finalRequest.messages.at(-1)?.content, 'Edit applied.');
    assert.deepEqual(events, [{ kind: 'text', text: 'Updated example.cls.' }]);
  });
});

test('final-only intent reaches the worker even with no tool definitions', async () => {
  await withFinalWorker([{ content: 'Done.' }], async (client, bodies) => {
    const { tools: _tools, ...request } = finalRequest;
    await client.chat(request, () => undefined);
    assert.equal(bodies[0]?.tool_choice, 'none');
  });
});

test('final generation never emits a structured tool call even if the worker ignores none', async () => {
  await withFinalWorker([
    { content: 'I will edit again.' },
    { tool_calls: [{ index: 0, id: 'unexpected', function: {
      name: 'replace_string_in_file', arguments: '{"filePath":"example.cls"}',
    } }] },
  ], async client => {
    const events: ChatStreamEvent[] = [];
    await assert.rejects(client.chat(finalRequest, event => events.push(event)), /tool.*disabled/i);
    assert.deepEqual(events, [], 'no text or executable call may escape the rejected response');
  });
});

for (const text of [
  '<tool_call>\n<function=replace_string_in_file>\n<parameter=filePath>example.cls</parameter>\n</function>\n</tool_call>',
  'One more change.\n<function=replace_string_in_file>\n<parameter=filePath>example.cls</parameter>',
]) {
  test(`final generation rejects leaked protocol split across SSE chunks: ${text.slice(0, 25)}`, async () => {
    await withFinalWorker([...text].map(content => ({ content })), async client => {
      const events: ChatStreamEvent[] = [];
      await assert.rejects(client.chat(finalRequest, event => events.push(event)), /tool.*disabled/i);
      assert.deepEqual(events, []);
    });
  });
}

test('final generation preserves legitimate fenced and inline tool-markup examples', async () => {
  const text = 'The parser handles `<tool_call>` and `<function=replace_string_in_file>`.\n' +
    '```xml\n<tool_call>\n<function=replace_string_in_file>\n</function>\n</tool_call>\n```\n' +
    '~~~xml\n<tool_call>\n</tool_call>\n~~~\nThe edit is complete.';
  await withFinalWorker([...text].map(content => ({ content })), async client => {
    const events: ChatStreamEvent[] = [];
    await client.chat(finalRequest, event => events.push(event));
    assert.equal(events.map(event => event.kind === 'text' ? event.text : '').join(''), text);
  });
});

test('final guidance is included in the measured prompt budget before generation', async () => {
  await withFinalWorker([{ content: 'Done.' }], async (client, bodies) => {
    await assert.rejects(client.chat({ ...finalRequest, inputTokenBudget: 100 }, () => undefined), /budget/i);
    assert.equal(bodies.length, 0, 'an oversized final prompt must never start generation');
  }, body => (body.messages as ChatRequest['messages']).at(-1)?.content === 'Edit applied.' ? 100 : 101);
});

const plainRequest: ChatRequest = {
  messages: [{ role: 'user', content: 'Say Hi' }],
  inputTokenBudget: 1_000,
  maxTokens: 64,
  temperature: 0,
};

test('reasoning is not the answer: the reply streams only content', async () => {
  await withFinalWorker([
    { reasoning_content: 'The user wants a greeting.' },
    { content: 'Hi!' },
  ], async client => {
    const events: ChatStreamEvent[] = [];
    await client.chat(plainRequest, event => events.push(event));
    assert.deepEqual(events, [{ kind: 'text', text: 'Hi!' }]);
  });
});

test('a reply that spends the output limit thinking fails instead of ending silently', async () => {
  await withFinalWorker([
    { reasoning_content: 'Thinking Process:' },
    { reasoning_content: ' 1. Analyze' },
  ], async client => {
    await assert.rejects(
      client.chat(plainRequest, () => undefined),
      /spent its whole output limit \(64 tokens\) reasoning.*localLlm\.maxOutputTokens/s,
    );
  }, undefined, 'length');
});

test('a final answer that spends the output limit thinking fails the same way', async () => {
  await withFinalWorker([{ reasoning_content: 'Checking the edit.' }], async client => {
    await assert.rejects(
      client.chat(finalRequest, () => undefined),
      /reasoning.*localLlm\.maxOutputTokens/s,
    );
  }, undefined, 'length');
});

test('a speed measurement may end inside the reasoning', async () => {
  await withFinalWorker([
    { reasoning_content: 'Counting from one.' },
  ], async client => {
    const result = await client.chat({ ...plainRequest, allowReasoningOnly: true }, () => undefined);
    assert.equal(result.toolCallCount, 0);
  }, undefined, 'length');
});

test('a speed measurement that generated nothing at all still fails', async () => {
  await withFinalWorker([{ content: '' }], async client => {
    await assert.rejects(
      client.chat({ ...plainRequest, allowReasoningOnly: true }, () => undefined),
      /returned an empty response/,
    );
  }, undefined, 'stop');
});

test('an empty reply fails instead of ending silently', async () => {
  await withFinalWorker([{ content: '' }], async client => {
    await assert.rejects(
      client.chat(plainRequest, () => undefined),
      /returned an empty response/,
    );
  }, undefined, 'stop');
});

test('a stream event that is not JSON fails instead of being dropped', async () => {
  await withRawStreamWorker(
    'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\ndata: {"choices":[{"delta":{"content":\n\ndata: [DONE]\n\n',
    async client => {
      await assert.rejects(
        client.chat(plainRequest, () => undefined),
        /stream event that is not valid JSON/,
      );
    },
  );
});

test('a stream cut off in the middle of an event fails', async () => {
  await withRawStreamWorker(
    'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\ndata: {"choices":[{"delta":{"con',
    async client => {
      await assert.rejects(
        client.chat(plainRequest, () => undefined),
        /ended in the middle of an event/,
      );
    },
  );
});

test('a malformed event on a stream that stays open does not hold the worker connection', async () => {
  const openStreams: Array<import('node:http').ServerResponse> = [];
  const server = createServer((request, response) => {
    request.resume();
    request.on('end', () => {
      if (request.url === '/v1/chat/completions/input_tokens') {
        response.writeHead(200, { 'content-type': 'application/json' });
        response.end('{"input_tokens":10}');
      } else if (request.url === '/tokenize') {
        response.writeHead(200, { 'content-type': 'application/json' });
        response.end('{"tokens":[1,2,3]}');
      } else {
        // The worker keeps generating after the client has given up on it.
        response.writeHead(200, { 'content-type': 'text/event-stream' });
        response.write('data: {"choices":[{"delta":{"content":\n\n');
        openStreams.push(response);
      }
    });
  });
  await listen(server);
  const client = await testClient(server);
  try {
    await assert.rejects(client.chat(plainRequest, () => undefined), /not valid JSON/);
    const next = client.tokenize('next request');
    const timeout = new Promise<never>((_resolve, reject) => {
      setTimeout(() => reject(new Error('the next request queued behind the abandoned stream')), 2_000).unref();
    });
    assert.equal(await Promise.race([next, timeout]), 3);
  } finally {
    for (const response of openStreams) {
      response.destroy();
    }
    await client.dispose();
    server.closeAllConnections();
    await close(server);
  }
});

async function withRawStreamWorker(
  stream: string,
  run: (client: TestLlamaClient) => Promise<void>,
): Promise<void> {
  const server = createServer((request, response) => {
    request.resume();
    request.on('end', () => {
      if (request.url === '/v1/chat/completions/input_tokens') {
        response.writeHead(200, { 'content-type': 'application/json' });
        response.end('{"input_tokens":10}');
      } else {
        response.writeHead(200, { 'content-type': 'text/event-stream' });
        response.end(stream);
      }
    });
  });
  await listen(server);
  const client = await testClient(server);
  try {
    await run(client);
  } finally {
    await client.dispose();
    server.closeAllConnections();
    await close(server);
  }
}

async function withFinalWorker(
  deltas: Array<Record<string, unknown>>,
  run: (client: TestLlamaClient, bodies: Array<Record<string, unknown>>) => Promise<void>,
  countTokens: (body: Record<string, unknown>) => number = () => 100,
  finishReason?: string,
): Promise<void> {
  const bodies: Array<Record<string, unknown>> = [];
  const server = createServer((request, response) => {
    let raw = '';
    request.on('data', chunk => { raw += chunk; });
    request.on('end', () => {
      const body = JSON.parse(raw) as Record<string, unknown>;
      if (request.url === '/v1/chat/completions/input_tokens') {
        response.writeHead(200, { 'content-type': 'application/json' });
        response.end(JSON.stringify({ input_tokens: countTokens(body) }));
      } else {
        bodies.push(body);
        response.writeHead(200, { 'content-type': 'text/event-stream' });
        const finish = finishReason
          ? `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: finishReason }] })}\n\n`
          : '';
        response.end(deltas.map(delta => `data: ${JSON.stringify({ choices: [{ delta }] })}\n\n`).join('') + finish + 'data: [DONE]\n\n');
      }
    });
  });
  await listen(server);
  const client = await testClient(server);
  try {
    await run(client, bodies);
  } finally {
    await client.dispose();
    server.closeAllConnections();
    await close(server);
  }
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
