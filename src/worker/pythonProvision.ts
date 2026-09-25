/**
 * The effectful half of provisioning: downloads, extraction, venv creation,
 * and hermetic wheel install. Pure rules (detection, flavor, manifest match)
 * live in `pythonEnvironment.ts` and are unit-tested there.
 *
 * The downloader and command runner are injectable so the flow is tested
 * without network or processes; production passes the real ones.
 */

import { mkdir, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises';
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
import { describeError, isMissingFileError } from '../errorDetail.ts';

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
  // Provisioning output is logged as worker output; the prefix raises these
  // above debug so a fallback is visible at the default log level.
  const warn = (message: string): void => options.onLog(`[warning] ${message}`);
  const flavor = await resolveEnvFlavor(target, options.flavorSetting, undefined, warn);
  const directory = envDirectory(storagePath, target, flavor);
  const release = await deps.readReleaseManifest(options.releaseManifestPath, target);
  const wheels = wheelsForFlavor(release, flavor);
  const releaseForFlavor: EnvManifestTarget = { ...release, wheels };
  if (await envMatchesManifest(storagePath, target, flavor, releaseForFlavor, warn)) {
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
    undefined, options.progress, options.token, warn,
  );
  const interpreterDir = path.join(directory, 'interpreter');
  await mkdir(interpreterDir, { recursive: true });
  // bsdtar ships with macOS and with Windows 10+, so no new dependency.
  // Standalone archives nest everything under one top-level `python/`
  // directory; stripping it keeps interpreterExePath fixed per platform.
  await deps.run('tar', ['-xf', archivePath, '--strip-components', '1', '-C', interpreterDir]);
  const wheelPaths: string[] = [];
  for (const [index, wheel] of wheels.entries()) {
    options.progress.report({ message: `Downloading ${wheel.name} (${index + 1}/${wheels.length})` });
    const destination = path.join(directory, 'wheels', safeWheelFilename(wheel));
    await deps.download(wheel.url, destination, wheel.sha256, undefined, options.progress, options.token, warn);
    wheelPaths.push(destination);
  }
  const interpreterExe = interpreterExePath(interpreterDir, target);
  try {
    await stat(interpreterExe);
  } catch {
    // An archive layout change would otherwise surface pages later as a bare
    // ENOENT from the venv spawn. Fail here, naming the actual problem.
    throw new Error(
      `The downloaded Python interpreter has an unexpected layout: ${interpreterExe} ` +
      `is missing after extraction. Re-run provisioning; if it persists, the ` +
      `release manifest pins an incompatible interpreter build.`,
    );
  }
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

/**
 * The standalone interpreter inside an extracted archive. python-build-standalone
 * ships everything under one top-level `python/` directory, which extraction
 * strips (see extractArchive), so the layout is fixed per platform.
 */
export function interpreterExePath(interpreterDir: string, target: string): string {
  return target.startsWith('win32')
    ? path.join(interpreterDir, 'python.exe')
    : path.join(interpreterDir, 'bin', 'python3');
}

/**
 * The on-disk name for a downloaded wheel: the url's last path segment,
 * percent-decoded. PyTorch serves local version segments encoded
 * (`torch-2.14.0%2Bcpu-...`), and pip parses a wheel's filename before it opens
 * the archive, so writing the segment verbatim fails the install with
 * InvalidWheelFilename — after the whole multi-gigabyte download. A malformed
 * escape keeps the raw segment, which still names the wheel where the pin
 * fallback would not, and a decode that yields a path separator is discarded so
 * the file cannot land outside the wheels directory.
 */
function safeWheelFilename(wheel: { url: string; name: string }): string {
  const segment = wheel.url.split('/').pop()?.split('?')[0] || `${wheel.name}.whl`;
  try {
    const decoded = decodeURIComponent(segment);
    return /[/\\]/.test(decoded) ? segment : decoded;
  } catch {
    return segment;
  }
}

export async function removeStaleEnvironments(
  storagePath: string,
  keep: { target: string; flavor: PythonEnvFlavor },
  onWarning?: (message: string) => void,
): Promise<void> {
  // Best-effort hygiene after a successful provision: it never throws, because
  // it must never break a working env, but a stale env it cannot remove holds
  // gigabytes, so every failure is reported.
  const listed = async (directory: string): Promise<string[]> => {
    try {
      return await readdir(directory);
    } catch (error) {
      if (!isMissingFileError(error)) {
        onWarning?.(`Could not list ${directory} to remove stale Python environments: ${describeError(error)}`);
      }
      return [];
    }
  };
  const root = path.join(storagePath, 'python-env');
  for (const target of await listed(root)) {
    for (const flavor of await listed(path.join(root, target))) {
      if (flavor.endsWith('.json')) {
        continue;
      }
      if (target !== keep.target || flavor !== keep.flavor) {
        const stale = path.join(root, target, flavor);
        try {
          await rm(stale, { recursive: true, force: true });
        } catch (error) {
          onWarning?.(`Could not remove the stale Python environment ${stale}: ${describeError(error)}`);
        }
      }
    }
  }
}
