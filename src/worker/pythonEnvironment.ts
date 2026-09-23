/**
 * Provisioning the Safetensors runtime's Python environment: detection,
 * flavor resolution, manifest validation, and path layout.
 *
 * Pure by design — no `vscode`, no network, no process spawn outside the
 * injected `CommandRunner` — so every rule here is unit-tested without an
 * interpreter. The effectful install flow lives in `pythonProvision.ts`.
 *
 * OS probing follows the `syclDevice.ts` idiom: pure parsers over injected
 * command output.
 */

import { execFile } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import * as path from 'node:path';
import { describeError, isMissingFileError } from '../errorDetail.ts';

export type PythonEnvFlavor = 'cpu' | 'cuda' | 'xpu';
export type PythonEnvFlavorSetting = 'auto' | PythonEnvFlavor;

export interface EnvManifestWheel {
  name: string;
  url: string;
  sha256: string;
}

export interface EnvManifestPack {
  wheels: EnvManifestWheel[];
  /**
   * Base wheel names this pack supersedes. The XPU flavor ships torch's own
   * +xpu build, so the xpu pack replaces stock torch rather than adding to
   * it — two torch builds in one env would be undefined behavior.
   */
  replaces?: string[];
}

export interface EnvManifestTarget {
  interpreter: { url: string; sha256: string; exe: string };
  wheels: EnvManifestWheel[];
  packs?: Partial<Record<Exclude<PythonEnvFlavor, 'cpu'>, EnvManifestPack>>;
}

export interface EnvManifest {
  manifestVersion: 1;
  targets: Record<string, EnvManifestTarget>;
}

export type CommandRunner = (
  executable: string,
  args: string[],
) => Promise<{ stdout: string; stderr: string }>;

/** Matches `Intel(R) Arc(TM) 140T Graphics` and discrete Arc names alike. */
const INTEL_ARC_PATTERN = /intel.*arc|arc.*intel/i;

export function parseNvidiaSmi(exitOk: boolean): boolean {
  return exitOk;
}

export function parseVideoControllerNames(output: string): string[] {
  const names: string[] = [];
  for (const line of output.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (trimmed) {
      names.push(trimmed);
    }
  }
  return names;
}

export function hasIntelArc(names: readonly string[]): boolean {
  return names.some((name) => INTEL_ARC_PATTERN.test(name));
}

/**
 * Detects the Windows GPU flavor. Never throws: any probe failure degrades
 * to CPU, and the `pythonEnvFlavor` setting always wins over detection. Every
 * failed probe is reported, because a GPU machine quietly provisioned for CPU
 * looks healthy and is just slow.
 */
export async function detectWindowsGpuFlavor(
  run: CommandRunner = runCommand,
  onWarning?: (message: string) => void,
): Promise<PythonEnvFlavor> {
  try {
    await run('nvidia-smi', ['-L']);
    return 'cuda';
  } catch (error) {
    // No nvidia-smi means no NVIDIA driver, the normal case on Intel. One that
    // exists and fails is a broken driver, which must not pass for "no GPU".
    if (!isMissingFileError(error)) {
      onWarning?.(`nvidia-smi is installed but failed, so CUDA is not used: ${describeError(error)}`);
    }
  }
  try {
    const { stdout } = await run('powershell', [
      '-NoProfile',
      '-NonInteractive',
      '-Command',
      'Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name',
    ]);
    if (hasIntelArc(parseVideoControllerNames(stdout))) {
      return 'xpu';
    }
  } catch (error) {
    // CPU is the safe answer, but not a silent one.
    onWarning?.(`Could not list video controllers, so the Python environment falls back to CPU: ${describeError(error)}`);
  }
  return 'cpu';
}

/**
 * Maps a Node platform/arch pair onto a provisioned target. Anything else is
 * a named error, not a guess: installing the wrong interpreter wastes
 * gigabytes before failing.
 */
export function resolveEnvTarget(
  platform: string = process.platform,
  arch: string = process.arch,
): string {
  if (platform === 'darwin' && arch === 'arm64') {
    return 'darwin-arm64';
  }
  if (platform === 'win32' && arch === 'x64') {
    return 'win32-x64';
  }
  throw new Error(
    `Automatic Python provisioning supports macOS (arm64) and Windows (x64); ` +
    `this machine is ${platform}/${arch}. Set "localLlm.pythonPath" to an ` +
    `interpreter with the Safetensors requirements installed instead.`,
  );
}

export async function resolveEnvFlavor(
  target: string,
  setting: PythonEnvFlavorSetting,
  detect?: () => Promise<PythonEnvFlavor>,
  onWarning?: (message: string) => void,
): Promise<PythonEnvFlavor> {
  if (setting !== 'auto') {
    return setting;
  }
  if (target === 'darwin-arm64') {
    return 'cpu';
  }
  if (target === 'win32-x64') {
    return (detect ?? (() => detectWindowsGpuFlavor(runCommand, onWarning)))();
  }
  return 'cpu';
}

export function envDirectory(storagePath: string, target: string, flavor: PythonEnvFlavor): string {
  return path.join(storagePath, 'python-env', target, flavor);
}

export function installedManifestPath(storagePath: string, target: string, flavor: PythonEnvFlavor): string {
  return `${envDirectory(storagePath, target, flavor)}.json`;
}

export function parseEnvManifest(raw: unknown): EnvManifest {
  if (typeof raw !== 'object' || raw === null) {
    throw new Error('Python environment manifest is not an object.');
  }
  const manifest = raw as { manifestVersion?: unknown; targets?: unknown };
  if (manifest.manifestVersion !== 1 || typeof manifest.targets !== 'object' || manifest.targets === null) {
    throw new Error('Python environment manifest has an unsupported version or shape.');
  }
  return { manifestVersion: 1, targets: manifest.targets as Record<string, EnvManifestTarget> };
}

function wheelName(wheel: EnvManifestWheel): string {
  return (wheel.name.split('==')[0] ?? wheel.name).toLowerCase();
}

export function wheelsForFlavor(entry: EnvManifestTarget, flavor: PythonEnvFlavor): EnvManifestWheel[] {
  if (flavor === 'cpu') {
    return entry.wheels;
  }
  const pack = entry.packs?.[flavor];
  if (!pack) {
    return entry.wheels;
  }
  // Pack wins twice: declared replacements drop the base build, and any
  // incidental version collision (sympy 1.14 base vs 1.13 pack) resolves to
  // the pack — one version per package, or pip installs over itself in
  // directory order.
  const packNames = new Set(
    pack.wheels.map((wheel) => wheelName(wheel)),
  );
  const replaced = new Set((pack.replaces ?? []).map((name) => name.toLowerCase()));
  const base = entry.wheels.filter((wheel) => {
    const name = wheelName(wheel);
    return !replaced.has(name) && !packNames.has(name);
  });
  return [...base, ...pack.wheels];
}

/**
 * Whether the installed env matches the release manifest: same interpreter
 * and wheel hashes for the resolved flavor. Any mismatch (or unreadable
 * record) means re-provision.
 */
export async function envMatchesManifest(
  storagePath: string,
  target: string,
  flavor: PythonEnvFlavor,
  expected: EnvManifestTarget,
  onWarning?: (message: string) => void,
): Promise<boolean> {
  let recorded: EnvManifestTarget;
  try {
    recorded = parseInstalledRecord(
      JSON.parse(await readFile(installedManifestPath(storagePath, target, flavor), 'utf8')),
    );
  } catch (error) {
    // No record is a first install. An unreadable one forces a reinstall of
    // several gigabytes, which must say why.
    if (!isMissingFileError(error)) {
      onWarning?.(`The installed Python environment record is unreadable; reinstalling: ${describeError(error)}`);
    }
    return false;
  }
  const expectedWheels = wheelsForFlavor(expected, flavor);
  const recordedWheels = wheelsForFlavor(recorded, flavor);
  return (
    recorded.interpreter.sha256.toLowerCase() === expected.interpreter.sha256.toLowerCase() &&
    recordedWheels.length === expectedWheels.length &&
    recordedWheels.every((wheel, index) =>
      wheel.sha256.toLowerCase() === expectedWheels[index]?.sha256.toLowerCase() &&
      wheel.url === expectedWheels[index]?.url,
    )
  );
}

function parseInstalledRecord(raw: unknown): EnvManifestTarget {
  if (typeof raw !== 'object' || raw === null) {
    throw new Error('Installed environment record is not an object.');
  }
  const record = (raw as { installed?: unknown }).installed ?? raw;
  return record as EnvManifestTarget;
}

export function runCommand(
  executable: string,
  args: string[],
): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    execFile(executable, args, {
      windowsHide: true,
      timeout: 120_000,
      maxBuffer: 4 * 1024 * 1024,
    }, (error, stdout, stderr) => {
      if (error) {
        Object.assign(error, { stdout, stderr });
        reject(error);
        return;
      }
      resolve({ stdout, stderr });
    });
  });
}
