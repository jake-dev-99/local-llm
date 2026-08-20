import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { readFile } from 'node:fs/promises';
import * as path from 'node:path';

interface WorkerManifestEntry {
  path: string;
  sha256: string;
}

interface WorkerManifest {
  workers?: Record<string, WorkerManifestEntry>;
}

export async function verifiedWorkerPath(
  extensionPath: string,
  target: string,
): Promise<string> {
  const manifestPath = path.join(extensionPath, 'resources', 'workers', 'manifest.json');
  let manifest: WorkerManifest;
  try {
    manifest = JSON.parse(await readFile(manifestPath, 'utf8')) as WorkerManifest;
  } catch (error) {
    throw new Error(
      `Bundled worker manifest is missing or invalid: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  const entry = manifest.workers?.[target];
  const requiredPrefix = `resources/workers/${target}/`;
  if (
    !entry ||
    !entry.path.startsWith(requiredPrefix) ||
    !/^[a-f0-9]{64}$/i.test(entry.sha256)
  ) {
    throw new Error(`Bundled worker manifest has no valid ${target} entry.`);
  }
  const workerPath = path.join(extensionPath, ...entry.path.split('/'));
  const actualSha256 = await sha256File(workerPath);
  if (actualSha256.toLowerCase() !== entry.sha256.toLowerCase()) {
    throw new Error(
      `Bundled ${target} worker failed its SHA-256 integrity check. Reinstall the extension from a trusted VSIX.`,
    );
  }
  return workerPath;
}

export async function sha256File(filePath: string): Promise<string> {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(filePath)) {
    hash.update(chunk as Buffer);
  }
  return hash.digest('hex');
}
