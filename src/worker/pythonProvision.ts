/**
 * The effectful half of provisioning: downloads, extraction, venv creation,
 * and hermetic wheel install. Pure rules (detection, flavor, manifest match)
 * live in `pythonEnvironment.ts` and are unit-tested there.
 *
 * The downloader and command runner are injectable so the flow is tested
 * without network or processes; production passes the real ones.
 */

import { mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import * as path from 'node:path';
import * as vscode from 'vscode';
import { downloadModel } from '../models/modelSources.ts';
import {
  envDirectory,
  envMatchesManifest,
  installedManifestPath,
  parseEnvManifest,
  resolveEnvFlavor,
  runCommand,
  wheelsForFlavor,
  type CommandRunner,
  type EnvManifestTarget,
  type PythonEnvFlavor,
  type PythonEnvFlavorSetting,
} from './pythonEnvironment.ts';

export interface EnsureEnvironmentOptions {
  storagePath: string;
  target: string;
  flavorSetting: PythonEnvFlavorSetting;
  releaseManifestPath: string;
  progress: vscode.Progress<{ message?: string; increment?: number }>;
  token: vscode.CancellationToken;
  onLog: (message: string) => void;
  /**
   * Blesses a fresh env before its record is written. A failed probe throws
   * and the record is never written, so the next use retries instead of
   * trusting a broken env. Omitted only in tests.
   */
  probe?: (pythonPath: string) => Promise<unknown>;
}

export interface EnsuredEnvironment {
  pythonPath: string;
  target: string;
  flavor: PythonEnvFlavor;
  fresh: boolean;
}

export interface ProvisionDeps {
  download: typeof downloadModel;
  run: CommandRunner;
  readReleaseManifest: (manifestPath: string, target: string) => Promise<EnvManifestTarget>;
}

async function readReleaseTarget(
  manifestPath: string,
  target: string,
): Promise<EnvManifestTarget> {
  let manifest;
  try {
    manifest = parseEnvManifest(JSON.parse(await readFile(manifestPath, 'utf8')));
  } catch (error) {
    throw new Error(
      `Python environment manifest is missing or invalid: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  const entry = manifest.targets[target];
  if (!entry) {
    throw new Error(`Python environment manifest has no entry for ${target}.`);
  }
  return entry;
}

/**
 * Returns a usable interpreter, provisioning the env on first use.
 *
 * Steps: resolve flavor → compare installed record → download (interpreter
 * archive + wheels, each resumed and checksummed) → extract → venv →
 * hermetic `pip install --no-index` → write record. The caller probes
 * afterwards with the existing `--probe`; a dead probe refuses to bless the
 * env, so the next use retries instead of trusting it.
 */
export async function ensureEnvironment(
  options: EnsureEnvironmentOptions,
  deps: ProvisionDeps = {
    download: downloadModel,
    run: runCommand,
    readReleaseManifest: readReleaseTarget,
  },
): Promise<EnsuredEnvironment> {
  const { storagePath, target } = options;
  const flavor = await resolveEnvFlavor(target, options.flavorSetting);
  const directory = envDirectory(storagePath, target, flavor);
  const release = await deps.readReleaseManifest(options.releaseManifestPath, target);
  const wheels = wheelsForFlavor(release, flavor);
  const releaseForFlavor: EnvManifestTarget = { ...release, wheels };
  if (await envMatchesManifest(storagePath, target, flavor, releaseForFlavor)) {
    return { pythonPath: venvPython(directory, target), target, flavor, fresh: false };
  }
  options.onLog(`[Provisioning] Installing Python environment (${target}/${flavor}).`);
  await mkdir(directory, { recursive: true });
  // python-build-standalone install_only archives are .tar.gz on every
  // platform; bsdtar extracts them on macOS and Windows alike.
  const archivePath = path.join(directory, 'interpreter.tar.gz');
  options.progress.report({ message: 'Downloading Python interpreter' });
  await deps.download(
    release.interpreter.url, archivePath, release.interpreter.sha256,
    undefined, options.progress, options.token,
  );
  const interpreterDir = path.join(directory, 'interpreter');
  await mkdir(interpreterDir, { recursive: true });
  // bsdtar ships with macOS and with Windows 10+, so no new dependency.
  await deps.run('tar', ['-xf', archivePath, '-C', interpreterDir]);
  const wheelPaths: string[] = [];
  for (const [index, wheel] of wheels.entries()) {
    options.progress.report({ message: `Downloading ${wheel.name} (${index + 1}/${wheels.length})` });
    const destination = path.join(directory, 'wheels', safeWheelFilename(wheel));
    await deps.download(wheel.url, destination, wheel.sha256, undefined, options.progress, options.token);
    wheelPaths.push(destination);
  }
  const interpreterExe = target.startsWith('win32')
    ? path.join(interpreterDir, 'python.exe')
    : path.join(interpreterDir, 'bin', 'python3');
  await deps.run(interpreterExe, ['-m', 'venv', path.join(directory, 'venv')]);
  const pythonPath = venvPython(directory, target);
  await deps.run(pythonPath, ['-m', 'pip', 'install', '--no-index', ...wheelPaths]);
  if (options.probe) {
    options.onLog('[Provisioning] Probing the fresh environment.');
    await options.probe(pythonPath);
  }
  const installed: EnvManifestTarget = { interpreter: release.interpreter, wheels };
  await writeFile(installedManifestPath(storagePath, target, flavor), JSON.stringify({ installed }, null, 2));
  return { pythonPath, target, flavor, fresh: true };
}

function venvPython(directory: string, target: string): string {
  return target.startsWith('win32')
    ? path.join(directory, 'venv', 'Scripts', 'python.exe')
    : path.join(directory, 'venv', 'bin', 'python3');
}

function safeWheelFilename(wheel: { url: string; name: string }): string {
  return wheel.url.split('/').pop()?.split('?')[0] || `${wheel.name}.whl`;
}

export async function removeStaleEnvironments(
  storagePath: string,
  keep: { target: string; flavor: PythonEnvFlavor },
): Promise<void> {
  // Best-effort hygiene after a successful provision; never throws.
  try {
    const root = path.join(storagePath, 'python-env');
    for (const target of await readdir(root).catch(() => [] as string[])) {
      for (const flavor of await readdir(path.join(root, target)).catch(() => [] as string[])) {
        if (flavor.endsWith('.json')) {
          continue;
        }
        if (target !== keep.target || flavor !== keep.flavor) {
          await rm(path.join(root, target, flavor), { recursive: true, force: true }).catch(() => undefined);
        }
      }
    }
  } catch {
    // Hygiene must never break a working env.
  }
}
