import { execFile } from 'node:child_process';
import path from 'node:path';

export interface SyclDevice {
  id: `SYCL${number}`;
  description: string;
}

export type DeviceRunner = (
  executable: string,
  args: string[],
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
): Promise<SyclDevice> {
  try {
    const { stdout, stderr } = await run(executable, ['--list-devices']);
    const device = parseSyclDevices(`${stdout}\n${stderr}`).find(({ id }) => id === 'SYCL0');
    if (!device) {
      throw new Error('No SYCL GPU was reported by the selected worker executable.');
    }
    return device;
  } catch (error) {
    throw new Error(
      `Windows SYCL device discovery failed: ${describeDiscoveryError(error)} Set localLlm.acceleration to cpu to select the CPU worker.`,
    );
  }
}

function runDeviceDiscovery(executable: string, args: string[]): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    execFile(executable, args, {
      cwd: path.dirname(executable),
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
