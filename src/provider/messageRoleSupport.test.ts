import assert from 'node:assert/strict';
import test from 'node:test';
import type { ChatMessage } from '../domain.ts';
import { messagesForSystemRoleSupport } from './messageRoleSupport.ts';

const messages: ChatMessage[] = [
  { role: 'system', content: 'Follow local instructions.' },
  { role: 'user', content: 'Review config.ts.' },
];

test('system-capable models retain system messages', () => {
  assert.deepEqual(messagesForSystemRoleSupport(messages, true), messages);
});

test('models without a system role receive those instructions as user messages', () => {
  assert.deepEqual(messagesForSystemRoleSupport(messages, false), [
    { role: 'user', content: 'Follow local instructions.' },
    { role: 'user', content: 'Review config.ts.' },
  ]);
});
