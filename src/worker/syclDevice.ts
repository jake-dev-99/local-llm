import { execFile } from 'node:child_process';
import path from 'node:path';

export interface SyclDevice {
  id: `SYCL${number}`;
  description: string;
}

export interface DiscoveredSyclDevice extends SyclDevice {
  runtime: {
    adapter: 'level_zero' | 'opencl';
    environment: NodeJS.ProcessEnv;
  };
}

export type DeviceRunner = (
  executable: string,
  args: string[],
  options: { env: NodeJS.ProcessEnv },
) => Promise<{ stdout: string; stderr: string }>;

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

export async function discoverSycl0(
  executable: string,
  run: DeviceRunner = runDeviceDiscovery,
  baseEnvironment: NodeJS.ProcessEnv = process.env,
): Promise<DiscoveredSyclDevice> {
  const failures: string[] = [];
  for (const runtime of syclRuntimeCandidates(executable, baseEnvironment)) {
    try {
      const { stdout, stderr } = await run(
        executable,
        ['--list-devices'],
        { env: runtime.environment },
      );
      const device = parseSyclDevices(`${stdout}\n${stderr}`).find(({ id }) => id === 'SYCL0');
      if (!device) {
        throw new Error('No SYCL GPU was reported by the selected worker executable.');
      }
      return { ...device, runtime };
    } catch (error) {
      failures.push(`${runtime.adapter}: ${describeDiscoveryError(error)}`);
    }
  }
  throw new Error(
    `Windows SYCL device discovery failed for Level Zero and OpenCL: ${failures.join(' | ')} `
    + 'Set localLlm.acceleration to cpu to select the CPU worker.',
  );
}

function syclRuntimeCandidates(
  executable: string,
  baseEnvironment: NodeJS.ProcessEnv,
): DiscoveredSyclDevice['runtime'][] {
  const clean = { ...baseEnvironment };
  for (const key of Object.keys(clean)) {
    if (['ONEAPI_DEVICE_SELECTOR', 'SYCL_DEVICE_FILTER', 'UR_ADAPTERS_FORCE_LOAD', 'UR_ADAPTERS_SEARCH_PATH']
      .includes(key.toUpperCase())) {
      delete clean[key];
    }
  }
  return [
    {
      adapter: 'level_zero',
      environment: { ...clean, ONEAPI_DEVICE_SELECTOR: 'level_zero:gpu' },
    },
    {
      adapter: 'opencl',
      environment: {
        ...clean,
        ONEAPI_DEVICE_SELECTOR: 'opencl:gpu',
        UR_ADAPTERS_FORCE_LOAD: path.win32.join(
          path.win32.dirname(executable),
          'ur_adapter_opencl.dll',
        ),
      },
    },
  ];
}

function runDeviceDiscovery(
  executable: string,
  args: string[],
  options: { env: NodeJS.ProcessEnv },
): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    execFile(executable, args, {
      cwd: path.dirname(executable),
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
    typeof error.stderr === 'string' ? error.stderr : '';
  return stderr && !message.includes(stderr) ? `${message}: ${stderr}` : message;
}
