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

test('required chat falls back to one schema-constrained tool decision', async () => {
  const completionBodies: Array<Record<string, unknown>> = [];
  const server = createServer((request, response) => {
    if (request.url === '/v1/chat/completions/input_tokens') {
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end('{"input_tokens":10}');
      return;
    }
    if (request.url === '/v1/chat/completions') {
      const chunks: Buffer[] = [];
      request.on('data', (chunk: Buffer) => chunks.push(chunk));
      request.on('end', () => {
        const body = JSON.parse(Buffer.concat(chunks).toString()) as Record<string, unknown>;
        completionBodies.push(body);
        if (completionBodies.length === 1) {
          response.writeHead(200, { 'content-type': 'text/event-stream' });
          response.end(
            'data: {"choices":[{"delta":{"content":"I should inspect the file."}}]}\n\n' +
              'data: [DONE]\n\n',
          );
          return;
        }
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
      });
      return;
    }
    response.writeHead(404);
    response.end();
  });
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });

  try {
    const address = server.address() as AddressInfo;
    const LlamaClient = await loadLlamaClient();
    const client = new LlamaClient(`http://127.0.0.1:${address.port}`, 'test-key');
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

    assert.equal(completionBodies.length, 2);
    assert.ok(Array.isArray(completionBodies[0]?.tools));
    assert.equal(completionBodies[1]?.stream, false);
    assert.equal(completionBodies[1]?.tools, undefined);
    assert.equal(
      (completionBodies[1]?.response_format as { type?: unknown } | undefined)?.type,
      'json_object',
    );
    assert.equal(result.toolCallCount, 1);
    assert.equal(events.length, 1);
    const event = events[0];
    assert.ok(event && event.kind === 'toolCall');
    assert.equal(event.name, 'read_file');
    assert.deepEqual(event.input, { filePath: '/workspace/config.ts' });
  } finally {
    await new Promise<void>((resolve, reject) => {
      server.close((error) => (error ? reject(error) : resolve()));
    });
  }
});

test('schema fallback rejects arguments that violate the supplied tool schema', async () => {
  let completionCount = 0;
  const server = createServer((request, response) => {
    if (request.url === '/v1/chat/completions/input_tokens') {
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end('{"input_tokens":10}');
      return;
    }
    if (request.url === '/v1/chat/completions') {
      completionCount += 1;
      if (completionCount === 1) {
        response.writeHead(200, { 'content-type': 'text/event-stream' });
        response.end(
          'data: {"choices":[{"delta":{"content":"I should inspect the file."}}]}\n\n' +
            'data: [DONE]\n\n',
        );
        return;
      }
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify({
        choices: [{
          message: {
            content: JSON.stringify({
              kind: 'tool',
              name: 'read_file',
              arguments: { filePath: 42 },
            }),
          },
        }],
      }));
      return;
    }
    response.writeHead(404);
    response.end();
  });
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });

  try {
    const address = server.address() as AddressInfo;
    const LlamaClient = await loadLlamaClient();
    const client = new LlamaClient(`http://127.0.0.1:${address.port}`, 'test-key');

    await assert.rejects(
      client.chat(
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
      ),
      /violates the supplied tool schema/,
    );
    assert.equal(completionCount, 2);
  } finally {
    await new Promise<void>((resolve, reject) => {
      server.close((error) => (error ? reject(error) : resolve()));
    });
  }
});

test('native tool calls also validate arguments against the supplied schema', async () => {
  const server = createServer((request, response) => {
    if (request.url === '/v1/chat/completions/input_tokens') {
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end('{"input_tokens":10}');
      return;
    }
    if (request.url === '/v1/chat/completions') {
      response.writeHead(200, { 'content-type': 'text/event-stream' });
      response.end(
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-native","function":{"name":"read_file","arguments":"{\\"filePath\\":42}"}}]}}]}\n\n' +
          'data: [DONE]\n\n',
      );
      return;
    }
    response.writeHead(404);
    response.end();
  });
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });

  try {
    const address = server.address() as AddressInfo;
    const LlamaClient = await loadLlamaClient();
    const client = new LlamaClient(`http://127.0.0.1:${address.port}`, 'test-key');

    await assert.rejects(
      client.chat(
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
      ),
      /violates the schema for tool read_file/,
    );
  } finally {
    await new Promise<void>((resolve, reject) => {
      server.close((error) => (error ? reject(error) : resolve()));
    });
  }
});

test('automatic chat uses the constrained fallback to choose a final answer', async () => {
  const completionBodies: Array<Record<string, unknown>> = [];
  const server = createServer((request, response) => {
    if (request.url === '/v1/chat/completions/input_tokens') {
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end('{"input_tokens":10}');
      return;
    }
    if (request.url === '/v1/chat/completions') {
      const chunks: Buffer[] = [];
      request.on('data', (chunk: Buffer) => chunks.push(chunk));
      request.on('end', () => {
        completionBodies.push(
          JSON.parse(Buffer.concat(chunks).toString()) as Record<string, unknown>,
        );
        if (completionBodies.length === 1) {
          response.writeHead(200, { 'content-type': 'text/event-stream' });
          response.end(
            'data: {"choices":[{"delta":{"content":"Untrusted first answer."}}]}\n\n' +
              'data: [DONE]\n\n',
          );
          return;
        }
        response.writeHead(200, { 'content-type': 'application/json' });
        response.end(JSON.stringify({
          choices: [{
            message: {
              content: JSON.stringify({ kind: 'final', text: 'Validated final answer.' }),
            },
          }],
        }));
      });
      return;
    }
    response.writeHead(404);
    response.end();
  });
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });

  try {
    const address = server.address() as AddressInfo;
    const LlamaClient = await loadLlamaClient();
    const client = new LlamaClient(`http://127.0.0.1:${address.port}`, 'test-key');
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
        temperature: 0,
      },
      (event) => events.push(event),
    );

    assert.equal(completionBodies.length, 2);
    assert.equal(result.toolCallCount, 0);
    assert.deepEqual(events, [{ kind: 'text', text: 'Validated final answer.' }]);
    assert.equal(JSON.stringify(events).includes('Untrusted first answer.'), false);
  } finally {
    await new Promise<void>((resolve, reject) => {
      server.close((error) => (error ? reject(error) : resolve()));
    });
  }
});

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
