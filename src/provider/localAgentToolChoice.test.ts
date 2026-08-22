import assert from 'node:assert/strict';
import test from 'node:test';
import type { ChatMessage } from '../domain.ts';
import {
  isLocalAgentRequest,
  localAgentToolInvocationCount,
  localAgentToolLimitReached,
  LOCAL_AGENT_PROTOCOL_MARKER,
} from './localAgentToolChoice.ts';

const markerMessage: ChatMessage = {
  role: 'user',
  content: `Protocol marker: ${LOCAL_AGENT_PROTOCOL_MARKER}.`,
};

function toolCall(id: string, name: string): ChatMessage {
  return {
    role: 'assistant',
    content: '',
    tool_calls: [{
      id,
      type: 'function',
      function: { name, arguments: '{}' },
    }],
  };
}

test('Local Agent detection requires its protocol instruction', () => {
  assert.equal(
    isLocalAgentRequest([
      markerMessage,
      { role: 'user', content: 'Review config.ts.' },
    ]),
    true,
  );
  assert.equal(
    isLocalAgentRequest([{ role: 'user', content: 'Review config.ts.' }]),
    false,
  );
});

test('only emitted tools in the active request consume invocation allowance', () => {
  const messages: ChatMessage[] = [
    markerMessage,
    { role: 'user', content: 'Review config.ts.' },
    toolCall('call-1', 'read_file'),
    { role: 'tool', content: 'contents', tool_call_id: 'call-1' },
    toolCall('call-2', 'get_errors'),
    { role: 'tool', content: 'none', tool_call_id: 'call-2' },
  ];
  assert.equal(localAgentToolInvocationCount(messages), 2);
  assert.equal(localAgentToolLimitReached(messages, 3), false);
  assert.equal(localAgentToolLimitReached(messages, 2), true);
});

test('an emitted tool counts before its result arrives', () => {
  const messages: ChatMessage[] = [
    markerMessage,
    { role: 'user', content: 'Review config.ts.' },
    toolCall('call-1', 'read_file'),
  ];
  assert.equal(localAgentToolInvocationCount(messages), 1);
  assert.equal(localAgentToolLimitReached(messages, 1), true);
});

test('assistant final text consumes no invocation allowance', () => {
  const messages: ChatMessage[] = [
    markerMessage,
    { role: 'user', content: 'Reply Hi.' },
    { role: 'assistant', content: 'Hi' },
  ];
  assert.equal(localAgentToolInvocationCount(messages), 0);
  assert.equal(localAgentToolLimitReached(messages, 8), false);
});

test('a new request starts a fresh invocation allowance', () => {
  const messages: ChatMessage[] = [
    markerMessage,
    { role: 'user', content: 'Review config.ts.' },
    toolCall('call-1', 'read_file'),
    { role: 'tool', content: 'contents', tool_call_id: 'call-1' },
    { role: 'assistant', content: 'The file is valid.' },
    { role: 'user', content: 'Now review domain.ts.' },
  ];
  assert.equal(localAgentToolInvocationCount(messages), 0);
  assert.equal(localAgentToolLimitReached(messages, 1), false);
});

test('hook context does not replace the active user request', () => {
  const messages: ChatMessage[] = [
    markerMessage,
    { role: 'user', content: 'Review config.ts.' },
    toolCall('call-1', 'read_file'),
    { role: 'tool', content: 'contents', tool_call_id: 'call-1' },
    { role: 'user', content: 'Hook context: continue the current agent loop.' },
  ];
  assert.equal(localAgentToolInvocationCount(messages), 1);
});

test('ordinary Chat receives no Local Agent invocation limit', () => {
  const messages: ChatMessage[] = [
    { role: 'user', content: 'Review config.ts.' },
    toolCall('call-1', 'read_file'),
  ];
  assert.equal(localAgentToolInvocationCount(messages), 0);
  assert.equal(localAgentToolLimitReached(messages, 1), false);
});
