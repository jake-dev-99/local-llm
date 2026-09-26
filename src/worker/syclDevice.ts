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

const POSIX_SYCL_PROBES: Array<{ label: string; selector?: string }> = [
  { label: 'default' },
  { label: 'level-zero', selector: 'level_zero:gpu' },
  { label: 'opencl', selector: 'opencl:gpu' },
];

export function isWindowsExecutable(executable: string): boolean {
  return (
    /\.exe$/i.test(executable) ||
    /^[A-Za-z]:[\\/]/.test(executable) ||
    executable.includes('\\')
  );
}

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
  return isWindowsExecutable(executable)
    ? cleanWindowsSyclEnvironment(executable, baseEnvironment)
    : cleanPosixSyclEnvironment(executable, baseEnvironment);
}

function cleanWindowsSyclEnvironment(
  executable: string,
  baseEnvironment: NodeJS.ProcessEnv,
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

function cleanPosixSyclEnvironment(
  executable: string,
  baseEnvironment: NodeJS.ProcessEnv,
): NodeJS.ProcessEnv {
  const clean: NodeJS.ProcessEnv = {};
  for (const [key, value] of Object.entries(baseEnvironment)) {
    const canonical = key.toUpperCase();
    if (
      canonical === 'PATH' ||
      canonical === 'LD_LIBRARY_PATH' ||
      CONFLICTING_SYCL_VARIABLES.has(canonical)
    ) {
      continue;
    }
    clean[key] = value;
  }
  const bundleDirectory = path.posix.dirname(toPosixPath(executable));
  const systemPath = environmentValue(baseEnvironment, 'PATH') ?? '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin';
  clean.PATH = [bundleDirectory, systemPath].filter(Boolean).join(':');
  const systemLibraryPath = environmentValue(baseEnvironment, 'LD_LIBRARY_PATH');
  clean.LD_LIBRARY_PATH = [bundleDirectory, systemLibraryPath].filter(Boolean).join(':');
  return clean;
}

function toPosixPath(executable: string): string {
  return executable.replaceAll('\\', '/');
}

export async function discoverSycl0(
  executable: string,
  run: DeviceRunner = runDeviceDiscovery,
  baseEnvironment: NodeJS.ProcessEnv = process.env,
): Promise<DiscoveredSyclDevice> {
  if (isWindowsExecutable(executable)) {
    return discoverWindowsSycl0(executable, run, baseEnvironment);
  }
  return discoverPosixSyclDevice(executable, run, baseEnvironment);
}

async function discoverWindowsSycl0(
  executable: string,
  run: DeviceRunner,
  baseEnvironment: NodeJS.ProcessEnv,
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
      `SYCL device discovery failed: ${describeDiscoveryError(error)} `
      + 'Set localLlm.acceleration to cpu to disable GPU offload explicitly.',
    );
  }
}

async function discoverPosixSyclDevice(
  executable: string,
  run: DeviceRunner,
  baseEnvironment: NodeJS.ProcessEnv,
): Promise<DiscoveredSyclDevice> {
  const environment = cleanSyclEnvironment(executable, baseEnvironment);
  const failures: string[] = [];
  for (const probe of POSIX_SYCL_PROBES) {
    const probeEnvironment = { ...environment };
    if (probe.selector) {
      probeEnvironment.ONEAPI_DEVICE_SELECTOR = probe.selector;
    }
    try {
      const { stdout, stderr } = await run(executable, ['--list-devices'], { env: probeEnvironment });
      const device = parseSyclDevices(`${stdout}\n${stderr}`)[0];
      if (!device) {
        failures.push(`${probe.label}: no SYCL GPU was reported by the selected worker executable.`);
        continue;
      }
      return { ...device, environment: probeEnvironment };
    } catch (error) {
      failures.push(`${probe.label}: ${describeDiscoveryError(error)}`);
    }
  }
  throw new Error(
    `SYCL device discovery failed (${failures.join('; ') || 'no probes ran'}). `
    + 'Set localLlm.acceleration to cpu to disable GPU offload explicitly.',
  );
}

function runDeviceDiscovery(
  executable: string,
  args: string[],
  options: { env: NodeJS.ProcessEnv },
): Promise<{ stdout: string; stderr: string }> {
  const windows = isWindowsExecutable(executable);
  return new Promise((resolve, reject) => {
    execFile(executable, args, {
      cwd: windows ? path.win32.dirname(executable) : path.posix.dirname(toPosixPath(executable)),
      env: options.env,
      ...(windows ? { windowsHide: true } : {}),
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
