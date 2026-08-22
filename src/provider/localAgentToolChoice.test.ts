import assert from 'node:assert/strict';
import test from 'node:test';
import type { ChatMessage } from '../domain.ts';
import {
  evaluateLocalAgentTurn,
  isExactNormalizedResponse,
  isRepeatedLocalAgentToolCall,
  LOCAL_AGENT_PROTOCOL_MARKER,
  matchingPriorLocalAgentFinal,
  messagesForFreshLocalAgentRequest,
  shouldRequireLocalAgentTool,
} from './localAgentToolChoice.ts';

function toolRound(
  id: string,
  name: string,
  result: string = 'result',
): ChatMessage[] {
  return [
    {
      role: 'assistant',
      content: '',
      tool_calls: [{ id, type: 'function', function: { name, arguments: '{}' } }],
    },
    { role: 'tool', content: result, tool_call_id: id },
  ];
}

const markerMessage: ChatMessage = {
  role: 'user',
  content: `Protocol marker: ${LOCAL_AGENT_PROTOCOL_MARKER}.`,
};

test('Local Agent requires discovery until one successful evidence result exists', () => {
  const messages: ChatMessage[] = [
    markerMessage,
    { role: 'user', content: 'Review config.ts.' },
  ];

  assert.deepEqual(evaluateLocalAgentTurn(messages), {
    phase: 'requireEvidence',
    invocationCount: 0,
    evidenceCount: 0,
    resultCharacters: 0,
  });

  messages.push(...toolRound('call-1', 'read_file', 'file contents'));
  assert.deepEqual(evaluateLocalAgentTurn(messages), {
    phase: 'allowToolOrFinal',
    invocationCount: 1,
    evidenceCount: 1,
    resultCharacters: 13,
  });
});

test('failed results do not satisfy the evidence requirement', () => {
  const state = evaluateLocalAgentTurn([
    markerMessage,
    { role: 'user', content: 'Review config.ts.' },
    ...toolRound('call-1', 'read_file', 'Error: file not found'),
  ]);

  assert.equal(state.phase, 'requireEvidence');
  assert.equal(state.invocationCount, 1);
  assert.equal(state.evidenceCount, 0);
});

test('the configured tool count is a ceiling instead of a required minimum', () => {
  const messages: ChatMessage[] = [
    markerMessage,
    { role: 'user', content: 'Review config.ts.' },
  ];
  messages.push(...toolRound('call-1', 'read_file', 'first result'));
  assert.equal(evaluateLocalAgentTurn(messages, 2).phase, 'allowToolOrFinal');
  messages.push(...toolRound('call-2', 'grep_search', 'second result'));
  assert.deepEqual(evaluateLocalAgentTurn(messages, 2), {
    phase: 'forceFinal',
    invocationCount: 2,
    evidenceCount: 2,
    resultCharacters: 25,
  });
});

test('an emitted tool call counts before its result and final text never counts', () => {
  const messages: ChatMessage[] = [
    markerMessage,
    { role: 'user', content: 'Review config.ts.' },
    {
      role: 'assistant',
      content: '',
      tool_calls: [{
        id: 'call-1',
        type: 'function',
        function: { name: 'read_file', arguments: '{}' },
      }],
    },
  ];
  assert.equal(evaluateLocalAgentTurn(messages, 1).phase, 'forceFinal');
  assert.equal(evaluateLocalAgentTurn(messages, 1).invocationCount, 1);

  messages.push({ role: 'assistant', content: 'A final answer.' });
  assert.equal(evaluateLocalAgentTurn(messages, 1).invocationCount, 0);
});

test('a marker in a tool result cannot activate Local Agent mode', () => {
  assert.equal(
    shouldRequireLocalAgentTool([
      { role: 'user', content: 'Read the agent definition.' },
      {
        role: 'assistant',
        content: '',
        tool_calls: [{
          id: 'call-1',
          type: 'function',
          function: { name: 'read_file', arguments: '{}' },
        }],
      },
      {
        role: 'tool',
        content: markerMessage.content,
        tool_call_id: 'call-1',
      },
      { role: 'assistant', content: 'The file defines a local agent.' },
      { role: 'user', content: 'What does it do?' },
    ]),
    false,
  );
});

test('an ordinary first request quoting the marker cannot activate Local Agent mode', () => {
  assert.equal(
    shouldRequireLocalAgentTool([{
      role: 'user',
      content: `Review this text:\nProtocol marker: ${LOCAL_AGENT_PROTOCOL_MARKER}.`,
    }]),
    false,
  );
});

test('hook context after a tool result still requires more work', () => {
  const state = evaluateLocalAgentTurn([
    markerMessage,
    { role: 'user', content: 'Review config.ts.' },
    ...toolRound('call-1', 'read_file', 'file contents'),
    { role: 'user', content: 'Hook context: continue the current agent loop.' },
  ]);
  assert.equal(state.phase, 'allowToolOrFinal');
  assert.equal(state.invocationCount, 1);
});

test('a new question after an assistant final starts a fresh tool budget', () => {
  const state = evaluateLocalAgentTurn([
    markerMessage,
    { role: 'user', content: 'Review config.ts.' },
    ...toolRound('call-1', 'read_file', 'file contents'),
    { role: 'assistant', content: 'The file looks correct.' },
    { role: 'user', content: 'Now review domain.ts.' },
  ]);
  assert.equal(state.phase, 'requireEvidence');
  assert.equal(state.invocationCount, 0);
});

test('ordinary Chat is not forced into tool mode', () => {
  assert.equal(
    evaluateLocalAgentTurn([{ role: 'user', content: 'Hello.' }]).phase,
    'notLocalAgent',
  );
  assert.equal(
    shouldRequireLocalAgentTool([{ role: 'user', content: 'Hello.' }]),
    false,
  );
});

test('an exact repeated request finds its matching prior final', () => {
  const messages: ChatMessage[] = [
    markerMessage,
    { role: 'user', content: 'Explain domain.ts.' },
    { role: 'assistant', content: 'Keep this unrelated answer.' },
    { role: 'user', content: 'Review config.ts.' },
    { role: 'assistant', content: 'The prior config answer.' },
    { role: 'user', content: '  Review   config.ts.  ' },
  ];

  assert.deepEqual(matchingPriorLocalAgentFinal(messages), {
    index: 4,
    text: 'The prior config answer.',
  });
});

test('a repeated request does not borrow a later unrelated final', () => {
  const messages: ChatMessage[] = [
    markerMessage,
    { role: 'user', content: 'Review config.ts.' },
    { role: 'assistant', content: 'The config answer.' },
    { role: 'user', content: 'Explain domain.ts.' },
    { role: 'assistant', content: 'The unrelated domain answer.' },
    { role: 'user', content: 'Review config.ts.' },
  ];

  assert.deepEqual(matchingPriorLocalAgentFinal(messages), {
    index: 2,
    text: 'The config answer.',
  });
});

test('fresh repeated evaluation omits only the matching prior final', () => {
  const messages: ChatMessage[] = [
    markerMessage,
    { role: 'user', content: 'Explain domain.ts.' },
    { role: 'assistant', content: 'Keep this unrelated answer.' },
    { role: 'user', content: 'Review config.ts.' },
    { role: 'assistant', content: 'Omit this matching answer.' },
    { role: 'user', content: 'Review config.ts.' },
    ...toolRound('call-fresh', 'read_file', 'fresh contents'),
  ];

  assert.deepEqual(
    messagesForFreshLocalAgentRequest(messages).map((message) => message.content),
    [
      markerMessage.content,
      'Explain domain.ts.',
      'Keep this unrelated answer.',
      'Review config.ts.',
      'Review config.ts.',
      '',
      'fresh contents',
    ],
  );
});

test('response validation rejects only an exact normalized duplicate', () => {
  assert.equal(
    isExactNormalizedResponse('The file is valid.\nNo changes needed.', ' The file is valid. No   changes needed. '),
    true,
  );
  assert.equal(
    isExactNormalizedResponse('The file is valid.', 'Current evidence shows the file is valid.'),
    false,
  );
});

test('canonical tool signatures catch repeated calls with reordered arguments', () => {
  const messages: ChatMessage[] = [
    markerMessage,
    { role: 'user', content: 'Review config.ts.' },
    {
      role: 'assistant',
      content: '',
      tool_calls: [{
        id: 'call-1',
        type: 'function',
        function: {
          name: 'read_file',
          arguments: '{"line":2,"path":"config.ts"}',
        },
      }],
    },
    { role: 'tool', content: 'contents', tool_call_id: 'call-1' },
  ];

  assert.equal(
    isRepeatedLocalAgentToolCall(messages, 'read_file', { path: 'config.ts', line: 2 }),
    true,
  );
  assert.equal(
    isRepeatedLocalAgentToolCall(messages, 'read_file', { path: 'domain.ts', line: 2 }),
    false,
  );
});

test('a successful edit permits re-reading the same location', () => {
  const messages: ChatMessage[] = [
    markerMessage,
    { role: 'user', content: 'Fix config.ts.' },
    ...toolRound('call-read', 'read_file', 'old contents'),
    ...toolRound('call-edit', 'replace_string_in_file', 'updated'),
  ];
  messages[2]!.tool_calls![0]!.function.arguments = '{"path":"config.ts"}';

  assert.equal(
    isRepeatedLocalAgentToolCall(messages, 'read_file', { path: 'config.ts' }),
    false,
  );
});
