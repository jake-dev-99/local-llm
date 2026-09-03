import { execFile } from 'node:child_process';
import path from 'node:path';

export interface SyclDevice {
  id: `SYCL${number}`;
  description: string;
}

export interface DiscoveredSyclDevice extends SyclDevice {
  environment: NodeJS.ProcessEnv;
}

export type DeviceRunner = (
  executable: string,
  args: string[],
  options: { env: NodeJS.ProcessEnv },
) => Promise<{ stdout: string; stderr: string }>;

const CONFLICTING_SYCL_VARIABLES = new Set([
  'ONEAPI_DEVICE_SELECTOR',
  'SYCL_DEVICE_FILTER',
  'UR_ADAPTERS_FORCE_LOAD',
  'UR_ADAPTERS_SEARCH_PATH',
]);

export function parseSyclDevices(output: string): SyclDevice[] {
  const devices: SyclDevice[] = [];
  for (const line of output.split(/\r?\n/)) {
    const match = /^\s*(SYCL\d+)\s*:\s*(.+?)\s*$/.exec(line);
    if (match?.[1] && match[2]) {
      devices.push({ id: match[1] as SyclDevice['id'], description: match[2] });
    }
  }
  return devices;
}

export function cleanSyclEnvironment(
  executable: string,
  baseEnvironment: NodeJS.ProcessEnv = process.env,
): NodeJS.ProcessEnv {
  const clean: NodeJS.ProcessEnv = {};
  for (const [key, value] of Object.entries(baseEnvironment)) {
    const canonical = key.toUpperCase();
    if (canonical === 'PATH' || CONFLICTING_SYCL_VARIABLES.has(canonical)) continue;
    clean[key] = value;
  }
  const windowsRoot = environmentValue(baseEnvironment, 'SystemRoot')
    ?? environmentValue(baseEnvironment, 'WINDIR')
    ?? 'C:\\Windows';
  clean.PATH = [
    path.win32.dirname(executable),
    path.win32.join(windowsRoot, 'System32'),
    windowsRoot,
  ].join(';');
  return clean;
}

export async function discoverSycl0(
  executable: string,
  run: DeviceRunner = runDeviceDiscovery,
  baseEnvironment: NodeJS.ProcessEnv = process.env,
): Promise<DiscoveredSyclDevice> {
  const environment = cleanSyclEnvironment(executable, baseEnvironment);
  try {
    const { stdout, stderr } = await run(executable, ['--list-devices'], { env: environment });
    const device = parseSyclDevices(`${stdout}\n${stderr}`).find(({ id }) => id === 'SYCL0');
    if (!device) {
      throw new Error('No SYCL GPU was reported by the selected worker executable.');
    }
    return { ...device, environment };
  } catch (error) {
    throw new Error(
      `Windows SYCL device discovery failed: ${describeDiscoveryError(error)} `
      + 'Set localLlm.acceleration to cpu to disable GPU offload explicitly.',
    );
  }
}

function runDeviceDiscovery(
  executable: string,
  args: string[],
  options: { env: NodeJS.ProcessEnv },
): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    execFile(executable, args, {
      cwd: path.win32.dirname(executable),
      env: options.env,
      windowsHide: true,
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

function describeDiscoveryError(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  const stderr = typeof error === 'object' && error !== null && 'stderr' in error &&
    typeof error.stderr === 'string' ? error.stderr.trim() : '';
  return stderr && !message.includes(stderr) ? `${message}: ${stderr}` : message;
}

function environmentValue(environment: NodeJS.ProcessEnv, name: string): string | undefined {
  const key = Object.keys(environment).find(
    (candidate) => candidate.toUpperCase() === name.toUpperCase(),
  );
  return key ? environment[key] : undefined;
}
