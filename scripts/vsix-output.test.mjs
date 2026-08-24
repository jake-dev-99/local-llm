import assert from 'node:assert/strict';
import { mkdtemp, stat } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { prepareVsixOutput } from './vsix-output.mjs';

test('prepares a target-specific VSIX distribution path', async (context) => {
  const root = await mkdtemp(path.join(tmpdir(), 'local-llm-vsix-output-'));
  context.after(async () => {
    await import('node:fs/promises').then(({ rm }) => rm(root, { recursive: true, force: true }));
  });

  const output = await prepareVsixOutput(root, '0.3.4', 'darwin-arm64');

  assert.equal(
    output,
    path.join(
      root,
      'dist',
      'vsix',
      'darwin-arm64',
      'local-llm-engine-0.3.4-darwin-arm64.vsix',
    ),
  );
  assert.equal((await stat(path.dirname(output))).isDirectory(), true);
});
