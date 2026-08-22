import assert from 'node:assert/strict';
import test from 'node:test';
import type { ChatTool } from '../domain.ts';
import { parseToolDecision } from './toolProtocol.ts';

const tools: ChatTool[] = [{
  type: 'function',
  function: {
    name: 'read_file',
    parameters: { type: 'object' },
  },
}];

test('an automatic final decision selects ordinary final generation', () => {
  assert.deepEqual(
    parseToolDecision('{"kind":"final"}', tools, false),
    { kind: 'final' },
  );
});

test('a final decision cannot contain schema-generated answer text', () => {
  assert.throws(
    () => parseToolDecision(
      '{"kind":"final","text":"Schema-generated answer"}',
      tools,
      false,
    ),
    /violates the supplied tool schema/,
  );
});

test('required tool mode rejects a final decision', () => {
  assert.throws(
    () => parseToolDecision('{"kind":"final"}', tools, true),
    /violates the supplied tool schema/,
  );
});
