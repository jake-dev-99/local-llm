import assert from 'node:assert/strict';
import test from 'node:test';
import type { ChatMessage, ChatRequest, ChatStreamEvent } from '../domain.ts';
import type { ChatResult } from '../worker/llamaClient.ts';
import { LOCAL_AGENT_PROTOCOL_MARKER } from './localAgentToolChoice.ts';
import {
  runLocalAgentResponse,
  type LocalAgentGeneration,
} from './localAgentResponse.ts';

const messages: ChatMessage[] = [
  { role: 'user', content: `Protocol marker: ${LOCAL_AGENT_PROTOCOL_MARKER}.` },
  { role: 'user', content: 'Review config.ts.' },
  {
    role: 'assistant',
    content: '',
    tool_calls: [{
      id: 'call-1',
      type: 'function',
      function: { name: 'read_file', arguments: '{"path":"config.ts"}' },
    }],
  },
  { role: 'tool', content: 'fresh contents', tool_call_id: 'call-1' },
];

const request: ChatRequest = {
  messages,
  tools: [{
    type: 'function',
    function: { name: 'read_file', parameters: { type: 'object' } },
  }],
  toolChoice: 'auto',
  inputTokenBudget: 10_000,
  maxTokens: 2_048,
  temperature: 0.2,
};

test('an exact repeated final receives one independent revision', async () => {
  const requests: ChatRequest[] = [];
  const generations = generator([
    textGeneration('Same answer.'),
    textGeneration('Fresh evidence supports the same conclusion.'),
  ], requests);

  const response = await runLocalAgentResponse({
    messages,
    request,
    previousFinal: 'Same answer.',
    maxToolInvocations: 8,
    generate: generations,
  });

  assert.deepEqual(response.events, [{
    kind: 'text',
    text: 'Fresh evidence supports the same conclusion.',
  }]);
  assert.equal(requests.length, 2);
  assert.equal(requests[1]?.toolChoice, 'none');
  assert.equal(requests[1]?.tools, undefined);
});

test('a second exact duplicate becomes an explicit quality failure', async () => {
  const response = await runLocalAgentResponse({
    messages,
    request,
    previousFinal: 'Same answer.',
    maxToolInvocations: 8,
    generate: generator([
      textGeneration('Same answer.'),
      textGeneration(' Same   answer. '),
    ]),
  });

  assert.equal(response.events.length, 1);
  const failure = response.events[0];
  assert.ok(failure?.kind === 'text');
  assert.match(failure.text, /repeated its previous answer/i);
  assert.match(failure.text, /read_file/);
});

test('one repeated tool call is replaced by a novel decision', async () => {
  const response = await runLocalAgentResponse({
    messages,
    request,
    maxToolInvocations: 8,
    generate: generator([
      toolGeneration('read_file', { path: 'config.ts' }),
      toolGeneration('read_file', { path: 'domain.ts' }),
    ]),
  });

  assert.deepEqual(response.events[0], {
    kind: 'toolCall',
    id: 'generated-call',
    name: 'read_file',
    input: { path: 'domain.ts' },
  });
  assert.equal(response.observedToolCall, true);
});

test('a second repeated tool call forces final generation', async () => {
  const requests: ChatRequest[] = [];
  const response = await runLocalAgentResponse({
    messages,
    request,
    maxToolInvocations: 8,
    generate: generator([
      toolGeneration('read_file', { path: 'config.ts' }),
      toolGeneration('read_file', { path: 'config.ts' }),
      textGeneration('I inspected config.ts. Further work remains uncertain.'),
    ], requests),
  });

  assert.equal(requests.length, 3);
  assert.equal(requests[2]?.toolChoice, 'none');
  assert.equal(requests[2]?.tools, undefined);
  assert.deepEqual(response.events, [{
    kind: 'text',
    text: 'I inspected config.ts. Further work remains uncertain.',
  }]);
});

test('text accompanying a novel tool call is not treated as a duplicate final', async () => {
  const requests: ChatRequest[] = [];
  const response = await runLocalAgentResponse({
    messages,
    request,
    previousFinal: 'Same answer.',
    maxToolInvocations: 8,
    generate: generator([generation([
      { kind: 'text', text: 'Same answer.' },
      {
        kind: 'toolCall',
        id: 'generated-call',
        name: 'read_file',
        input: { path: 'domain.ts' },
      },
    ])], requests),
  });

  assert.equal(requests.length, 1);
  assert.equal(response.events.some((event) => event.kind === 'toolCall'), true);
});

function generator(
  generations: LocalAgentGeneration[],
  requests: ChatRequest[] = [],
): (request: ChatRequest) => Promise<LocalAgentGeneration> {
  return async (nextRequest) => {
    requests.push(nextRequest);
    const generation = generations.shift();
    assert.ok(generation, 'test supplied enough generations');
    return generation;
  };
}

function textGeneration(text: string): LocalAgentGeneration {
  return generation([{ kind: 'text', text }]);
}

function toolGeneration(name: string, input: object): LocalAgentGeneration {
  return generation([{ kind: 'toolCall', id: 'generated-call', name, input }]);
}

function generation(events: ChatStreamEvent[]): LocalAgentGeneration {
  const result: ChatResult = {
    inputTokens: 100,
    textCharacters: events.reduce(
      (total, event) => total + (event.kind === 'text' ? event.text.length : 0),
      0,
    ),
    toolCallCount: events.filter((event) => event.kind === 'toolCall').length,
  };
  return { events, result };
}
