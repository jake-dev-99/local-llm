import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { installVsix } from './install-vsix.mjs';

test('installs the versioned target VSIX through the VS Code CLI', async (context) => {
  const root = await mkdtemp(path.join(tmpdir(), 'local-llm-install-vsix-'));
  context.after(() => rm(root, { recursive: true, force: true }));
  await writeFile(path.join(root, 'package.json'), '{"version":"0.3.4"}');
  let accessedPath;
  let invocation;

  await installVsix('win32-x64', {
    root,
    platform: 'win32',
    commandProcessor: 'C:\\Windows\\System32\\cmd.exe',
    accessFile: async (filePath) => {
      accessedPath = filePath;
    },
    run: async (...args) => {
      invocation = args;
    },
    log: () => undefined,
  });

  const vsixPath = path.join(
    root,
    'dist',
    'vsix',
    'win32-x64',
    'local-llm-engine-0.3.4-win32-x64.vsix',
  );
  assert.equal(accessedPath, vsixPath);
  assert.deepEqual(invocation, [
    'C:\\Windows\\System32\\cmd.exe',
    ['/d', '/s', '/c', `code --install-extension "${vsixPath}" --force`],
    root,
    { windowsVerbatimArguments: true },
  ]);
});

test('rejects installation when the package output is absent', async (context) => {
  const root = await mkdtemp(path.join(tmpdir(), 'local-llm-install-vsix-'));
  context.after(() => rm(root, { recursive: true, force: true }));
  await writeFile(path.join(root, 'package.json'), '{"version":"0.3.4"}');

  await assert.rejects(
    installVsix('win32-x64', { root, log: () => undefined }),
    /VSIX package is missing: .*Run npm run package first/,
  );
});