import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import type { AddressInfo } from 'node:net';
import test from 'node:test';
import { build } from 'esbuild';
import type { ChatRequest, ChatStreamEvent } from '../domain';

interface TestLlamaClient {
  chat(
    request: ChatRequest,
    onEvent: (event: ChatStreamEvent) => void,
    signal?: AbortSignal,
  ): Promise<{ toolCallCount: number }>;
}

type TestLlamaClientConstructor = new (
  baseUrl: string,
  apiKey: string,
) => TestLlamaClient;

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
          response.writeHead(200, { 'content-type': 'application/json' });
          response.end(JSON.stringify({
            choices: [{
              message: {
                content: JSON.stringify({
                  kind: 'tool',
                  name: 'read_file',
                  arguments: { filePath: '/workspace/config.ts' },
                }),
              },
            }],
          }));
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
    write: false,
  });
  const source = bundled.outputFiles[0]?.contents;
  assert.ok(source, 'esbuild returned the bundled LlamaClient');
  const url = `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`;
  const loaded = await import(url) as { LlamaClient?: TestLlamaClientConstructor };
  assert.ok(loaded.LlamaClient, 'bundled module exports LlamaClient');
  return loaded.LlamaClient;
}
