import assert from 'node:assert/strict';
import test from 'node:test';
import { assertPromptFits } from './toolBudget.ts';

test('assertPromptFits keeps the complete tool contract when it fits', async () => {
  const tools = [
    { name: 'read_file' },
    { name: 'write_file' },
  ];
  const countInputTokens = async (candidate: typeof tools): Promise<number> =>
    12_000 + candidate.length * 1_000;

  const result = await assertPromptFits(tools, 27_000, countInputTokens);

  assert.deepEqual(result.tools, tools);
  assert.equal(result.inputTokens, 14_000);
});

test('assertPromptFits rejects an oversized prompt instead of silently deleting tools', async () => {
  const tools = [
    { name: 'read_file' },
    { name: 'write_file' },
    { name: 'search_files' },
  ];

  await assert.rejects(
    assertPromptFits(tools, 27_000, async () => 30_000),
    /complete prompt \(messages and 3 tool definitions\) requires 30000 input tokens.*only 27000 are available/i,
  );
});

test('oversized Local Agent prompts explain the response reserve without recommending Local Agent', async () => {
  const tools = Array.from({ length: 7 }, (_, index) => ({ name: `tool_${index}` }));
  await assert.rejects(
    assertPromptFits(tools, 2048, async () => 4677),
    (error: unknown) => {
      assert.ok(error instanceof Error);
      assert.match(error.message, /messages and 7 tool definitions.*4677 input tokens/);
      assert.match(error.message, /2048 are available after reserving response tokens/);
      assert.match(error.message, /Even Local Agent's bounded tool set/);
      assert.doesNotMatch(error.message, /use the Local Agent/i);
      assert.match(error.message, /only when the prompt itself fits within the loaded context/);
      assert.match(error.message, /No tools were removed/);
      return true;
    },
  );
});
