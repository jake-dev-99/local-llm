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
    /complete 3-tool contract requires 30000 input tokens.*budget is 27000/i,
  );
});
