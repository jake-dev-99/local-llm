import assert from 'node:assert/strict';
import test from 'node:test';
import { assertFinalResponse } from './finalResponse.ts';

const call = '<tool_call>\n<function=read_file>\n</function>\n</tool_call>';

for (const text of [
  'An unmatched ` tick.\n' + call + '\nExample: ``code``',
  'An escaped \\` tick.\n' + call + '\nAnother escaped \\` tick.',
]) {
  test(`unquoted tool calls cannot hide behind invalid inline code: ${text.slice(0, 22)}`, () => {
    assert.throws(() => assertFinalResponse(text, false), /tool.*disabled/i);
  });
}

test('indented code examples are preserved as code, not mistaken for tool execution', () => {
  assert.doesNotThrow(() => assertFinalResponse('Example:\n\n' + call.split('\n').map(line => '    ' + line).join('\n'), false));
});

test('ordinary XML source remains valid final-response content', () => {
  assert.doesNotThrow(() => assertFinalResponse('<config><enabled>true</enabled></config>', false));
});
