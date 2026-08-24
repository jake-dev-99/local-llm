import { mkdir } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

export function prepareVsceInvocation(root, args) {
  return {
    command: process.execPath,
    args: [path.join(root, 'node_modules', '@vscode', 'vsce', 'vsce'), ...args],
  };
}

export async function prepareVsixOutput(root, version, target) {
  const outputDirectory = path.join(root, 'dist', 'vsix', target);
  await mkdir(outputDirectory, { recursive: true });
  return path.join(outputDirectory, `local-llm-engine-${version}-${target}.vsix`);
}
