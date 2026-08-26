import { readFile } from 'node:fs/promises';
import * as path from 'node:path';
import {
  parseWorkerManifest,
  resolveWorkerBundle,
  verifyWorkerBundleFiles,
  type AccelerationMode,
  type ResolvedWorkerBundle,
} from './workerManifest.ts';

export interface VerifiedWorkerBundle extends ResolvedWorkerBundle {
  executablePath: string;
}

export async function verifiedWorkerBundle(
  extensionPath: string,
  target: string,
  mode: AccelerationMode,
): Promise<VerifiedWorkerBundle> {
  const manifestPath = path.join(extensionPath, 'resources', 'workers', 'manifest.json');
  let manifest;
  try {
    manifest = parseWorkerManifest(JSON.parse(await readFile(manifestPath, 'utf8')));
  } catch (error) {
    throw new Error(
      `Bundled worker manifest is missing or invalid: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  const bundle = resolveWorkerBundle(manifest, target, mode);
  await verifyWorkerBundleFiles(extensionPath, bundle);
  return {
    ...bundle,
    executablePath: path.join(extensionPath, ...bundle.executable.split('/')),
  };
}
