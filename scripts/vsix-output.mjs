import { mkdir } from 'node:fs/promises';
import path from 'node:path';

export async function prepareVsixOutput(root, version, target) {
  const outputDirectory = path.join(root, 'dist', 'vsix', target);
  await mkdir(outputDirectory, { recursive: true });
  return path.join(outputDirectory, `local-llm-engine-${version}-${target}.vsix`);
}
