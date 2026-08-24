import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import path from 'node:path';

export type AccelerationMode = 'auto' | 'cpu';
export type WorkerBackend = 'metal' | 'sycl' | 'cpu';

export interface WorkerFile { path: string; sha256: string }
export interface WorkerBundle { executable: string; files: WorkerFile[] }
export interface WorkerMode { bundle: string; backend: WorkerBackend }
export interface WorkerPlatform { modes: Record<AccelerationMode, WorkerMode>; bundles: Record<string, WorkerBundle> }
export interface WorkerManifestV2 { manifestVersion: 2; llamaCppCommit: string; llamaCppBuild: string; platforms: Record<string, WorkerPlatform> }
export interface ResolvedWorkerBundle extends WorkerBundle { target: string; bundleName: string; backend: WorkerBackend }

const BACKENDS = new Set<WorkerBackend>(['metal', 'sycl', 'cpu']);
const MODES = new Set<AccelerationMode>(['auto', 'cpu']);

function object(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) throw new Error(`${label} must be an object`);
  return value as Record<string, unknown>;
}
function string(value: unknown, label: string): string {
  if (typeof value !== 'string' || value.length === 0) throw new Error(`${label} must be a string`);
  return value;
}
function safePath(value: unknown, platform: string, bundle: string): string {
  const raw = string(value, 'worker file path').replaceAll('\\', '/');
  const normalized = path.posix.normalize(raw);
  const root = `resources/workers/${platform}/`;
  const bundleRoot = bundle === 'default' ? root : `${root}${bundle}/`;
  if (normalized.startsWith('/') || normalized === '.' || normalized.startsWith('../') || normalized.includes('/../') || !normalized.startsWith(bundleRoot)) {
    throw new Error(`worker file path must be a safe relative path below its platform and bundle directory`);
  }
  return normalized;
}
function parseBundle(value: unknown, target: string, bundleName: string): WorkerBundle {
  const raw = object(value, `bundle ${bundleName}`);
  const executable = safePath(raw.executable, target, bundleName);
  if (!Array.isArray(raw.files) || raw.files.length === 0) throw new Error(`bundle ${bundleName} files must be a non-empty array`);
  const files: WorkerFile[] = [];
  const seen = new Set<string>();
  for (const item of raw.files) {
    const file = object(item, 'worker file');
    const filePath = safePath(file.path, target, bundleName);
    if (seen.has(filePath)) throw new Error(`duplicate worker file path: ${filePath}`);
    seen.add(filePath);
    const sha256 = string(file.sha256, 'worker file sha256');
    if (!/^[0-9a-f]{64}$/i.test(sha256)) throw new Error(`worker file sha256 must be a 64-character hexadecimal value`);
    files.push({ path: filePath, sha256 });
  }
  if (files.filter((file) => file.path === executable).length !== 1) throw new Error(`bundle executable must appear exactly once in files`);
  return { executable, files };
}

export function parseWorkerManifest(value: unknown): WorkerManifestV2 {
  const raw = object(value, 'worker manifest');
  if (raw.manifestVersion !== 2) throw new Error('manifestVersion must be 2');
  const platformsRaw = object(raw.platforms, 'platforms');
  const platforms: Record<string, WorkerPlatform> = {};
  for (const [target, platformValue] of Object.entries(platformsRaw)) {
    const platform = object(platformValue, `platform ${target}`);
    const modesRaw = object(platform.modes, `platform ${target} modes`);
    const bundlesRaw = object(platform.bundles, `platform ${target} bundles`);
    const bundles: Record<string, WorkerBundle> = {};
    for (const [name, bundle] of Object.entries(bundlesRaw)) bundles[name] = parseBundle(bundle, target, name);
    const modes = {} as Record<AccelerationMode, WorkerMode>;
    for (const mode of MODES) {
      const modeRaw = object(modesRaw[mode], `${target} ${mode} mode`);
      const bundle = string(modeRaw.bundle, `${target} ${mode} bundle`);
      const backend = string(modeRaw.backend, `${target} ${mode} backend`) as WorkerBackend;
      if (!BACKENDS.has(backend) || !bundles[bundle]) throw new Error(`invalid ${target} ${mode} mode`);
      modes[mode] = { bundle, backend };
    }
    platforms[target] = { modes, bundles };
  }
  return { manifestVersion: 2, llamaCppCommit: string(raw.llamaCppCommit, 'llamaCppCommit'), llamaCppBuild: string(raw.llamaCppBuild, 'llamaCppBuild'), platforms };
}

export function resolveWorkerBundle(manifest: WorkerManifestV2, target: string, mode: AccelerationMode): ResolvedWorkerBundle {
  if (!MODES.has(mode)) throw new Error(`unsupported acceleration mode: ${mode}`);
  const platform = manifest.platforms[target];
  if (!platform) throw new Error(`worker target is not declared: ${target}`);
  const selection = platform.modes[mode];
  const bundle = platform.bundles[selection.bundle];
  if (!bundle) throw new Error(`worker bundle is not declared: ${selection.bundle}`);
  return { target, bundleName: selection.bundle, backend: selection.backend, executable: bundle.executable, files: bundle.files };
}

export async function sha256File(filePath: string): Promise<string> {
  return await new Promise((resolve, reject) => {
    const hash = createHash('sha256');
    const stream = createReadStream(filePath);
    stream.on('error', reject).on('data', (chunk: Buffer) => hash.update(chunk)).on('end', () => resolve(hash.digest('hex')));
  });
}

export async function verifyWorkerBundleFiles(root: string, bundle: WorkerBundle): Promise<void> {
  for (const file of bundle.files) {
    const actual = await sha256File(path.join(root, ...file.path.split('/')));
    if (actual.toLowerCase() !== file.sha256.toLowerCase()) throw new Error(`Bundled worker file ${file.path} failed its SHA-256 integrity check.`);
  }
}

export async function verifyPlatformBundles(root: string, manifest: WorkerManifestV2, target: string): Promise<void> {
  const platform = manifest.platforms[target];
  if (!platform) throw new Error(`worker target is not declared: ${target}`);
  for (const bundle of Object.values(platform.bundles)) await verifyWorkerBundleFiles(root, bundle);
}

export function replaceWorkerBundle(manifest: WorkerManifestV2, target: string, bundleName: string, bundle: WorkerBundle): WorkerManifestV2 {
  const raw = JSON.parse(JSON.stringify(manifest)) as Record<string, unknown>;
  const platforms = raw.platforms as Record<string, Record<string, unknown>>;
  const platform = platforms[target];
  if (!platform) throw new Error(`worker target is not declared: ${target}`);
  const bundles = platform.bundles as Record<string, unknown>;
  bundles[bundleName] = bundle;
  return parseWorkerManifest(raw);
}
