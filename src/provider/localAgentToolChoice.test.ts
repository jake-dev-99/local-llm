import assert from 'node:assert/strict';
import test from 'node:test';
import type { ChatMessage } from '../domain.ts';
import {
  completedToolRounds,
  LOCAL_AGENT_PROTOCOL_MARKER,
  shouldRequireLocalAgentTool,
} from './localAgentToolChoice.ts';

function toolRound(id: string, name: string): ChatMessage[] {
  return [
    {
      role: 'assistant',
      content: '',
      tool_calls: [{ id, type: 'function', function: { name, arguments: '{}' } }],
    },
    { role: 'tool', content: 'result', tool_call_id: id },
  ];
}

const markerMessage: ChatMessage = {
  role: 'user',
  content: `Protocol marker: ${LOCAL_AGENT_PROTOCOL_MARKER}.`,
};

test('Local Agent requires a tool at the start of each user turn', () => {
  assert.equal(
    shouldRequireLocalAgentTool([
      markerMessage,
      { role: 'user', content: 'Review config.ts.' },
    ]),
    true,
  );
  assert.equal(
    shouldRequireLocalAgentTool([
      markerMessage,
      { role: 'user', content: 'Review config.ts.' },
      { role: 'assistant', content: 'Done.' },
      { role: 'user', content: 'Now review domain.ts.' },
    ]),
    true,
  );
});

test('one completed tool result does not end the Local Agent turn', () => {
  assert.equal(
    shouldRequireLocalAgentTool([
      markerMessage,
      { role: 'user', content: 'Review config.ts.' },
      ...toolRound('call-1', 'read_file'),
    ]),
    true,
  );
});

test('Local Agent keeps requiring tools until the round budget is spent', () => {
  const messages: ChatMessage[] = [
    markerMessage,
    { role: 'user', content: 'Review config.ts.' },
  ];
  for (let round = 1; round <= 3; round += 1) {
    messages.push(...toolRound(`call-${round}`, 'read_file'));
    assert.equal(completedToolRounds(messages), round);
    assert.equal(shouldRequireLocalAgentTool(messages, 3), round < 3);
  }
});

test('a spent budget releases the model so it can answer', () => {
  const messages: ChatMessage[] = [
    markerMessage,
    { role: 'user', content: 'Review config.ts.' },
    ...toolRound('call-1', 'read_file'),
    ...toolRound('call-2', 'grep_search'),
  ];
  assert.equal(shouldRequireLocalAgentTool(messages, 2), false);
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
  assert.equal(
    shouldRequireLocalAgentTool([
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
      { role: 'tool', content: 'file contents', tool_call_id: 'call-1' },
      { role: 'user', content: 'Hook context: continue the current agent loop.' },
    ]),
    true,
  );
});

test('a new question after an assistant final starts a fresh tool budget', () => {
  assert.equal(
    shouldRequireLocalAgentTool([
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
      { role: 'tool', content: 'file contents', tool_call_id: 'call-1' },
      { role: 'assistant', content: 'The file looks correct.' },
      { role: 'user', content: 'Now review domain.ts.' },
    ]),
    true,
  );
});

test('ordinary Chat is not forced into tool mode', () => {
  assert.equal(
    shouldRequireLocalAgentTool([{ role: 'user', content: 'Hello.' }]),
    false,
  );
});
