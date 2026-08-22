import assert from 'node:assert/strict';
import test from 'node:test';
import type { ChatTool } from '../domain.ts';
import {
  localAgentAvailableTools,
  resolveLocalAgentToolPolicy,
} from './localAgentTools.ts';

function tool(name: string): ChatTool {
  return { type: 'function', function: { name, parameters: { type: 'object' } } };
}

test('Local Agent exposes every recognized local tool and excludes unknown tools', () => {
  const supplied = [
    tool('read_file'),
    tool('grep_search'),
    tool('replace_string_in_file'),
    tool('insert_edit_into_file'),
    tool('list_dir'),
    tool('get_errors'),
    tool('file_search'),
    tool('session_store_sql'),
  ];
  assert.deepEqual(
    localAgentAvailableTools(supplied).map((item) => item.function.name),
    [
      'read_file',
      'grep_search',
      'replace_string_in_file',
      'insert_edit_into_file',
      'list_dir',
      'get_errors',
      'file_search',
    ],
  );
});

test('automatic mode remains automatic before the invocation ceiling', () => {
  const tools = [tool('read_file'), tool('replace_string_in_file')];
  assert.deepEqual(
    resolveLocalAgentToolPolicy(tools, false, false),
    { tools, toolChoice: 'auto', source: 'caller-auto' },
  );
  assert.equal(tools.length, 2);
});

test('VS Code required mode remains required before the invocation ceiling', () => {
  const tools = [tool('read_file')];
  assert.deepEqual(
    resolveLocalAgentToolPolicy(tools, true, false),
    { tools, toolChoice: 'required', source: 'caller-required' },
  );
});

test('the invocation ceiling disables tools and requests final generation', () => {
  assert.deepEqual(
    resolveLocalAgentToolPolicy([tool('read_file')], true, true),
    { tools: [], toolChoice: 'none', source: 'local-agent-final' },
  );
});

test('an empty automatic tool list remains a valid final-answer request', () => {
  assert.deepEqual(
    resolveLocalAgentToolPolicy([], false, false),
    { tools: [], toolChoice: 'auto', source: 'caller-auto' },
  );
});
