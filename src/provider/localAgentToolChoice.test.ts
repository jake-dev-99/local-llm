import assert from 'node:assert/strict';
import test from 'node:test';
import type { ChatMessage } from '../domain.ts';
import {
  LOCAL_AGENT_PROTOCOL_MARKER,
  shouldRequireLocalAgentTool,
} from './localAgentToolChoice.ts';

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
    false,
  );
});

test('Local Agent returns to auto mode after a tool result', () => {
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
    ]),
    false,
  );
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

test('hook context after a tool result remains in auto mode', () => {
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
    false,
  );
});

test('a user turn after an assistant final does not require a new tool round', () => {
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
    false,
  );
});

test('ordinary Chat is not forced into tool mode', () => {
  assert.equal(
    shouldRequireLocalAgentTool([{ role: 'user', content: 'Hello.' }]),
    false,
  );
});
