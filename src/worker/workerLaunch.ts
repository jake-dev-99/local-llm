import type {
  AccelerationMode,
  ResolvedWorkerBundle,
  WorkerBackend,
} from './workerManifest';
import type { SyclDevice } from './syclDevice';

export interface LaunchableWorkerBundle extends ResolvedWorkerBundle {
  executablePath: string;
}

export interface PrepareWorkerLaunchInput {
  target: string;
  mode: AccelerationMode;
  resolveBundle: (target: string, mode: AccelerationMode) => Promise<LaunchableWorkerBundle>;
  discoverSycl: (executable: string) => Promise<SyclDevice>;
}

export interface PreparedWorkerLaunch {
  bundle: LaunchableWorkerBundle;
  backend: WorkerBackend;
  syclDevice?: SyclDevice;
}

// The production manifest remains version 1 in this task. Keep its current
// platform policy until the version-2 bundle migration switches Windows auto.
export function resolveExecutionBackend(target: string, mode: AccelerationMode): WorkerBackend {
  if (target === 'darwin-arm64' && mode === 'auto') {
    return 'metal';
  }
  if (target === 'win32-x64' && mode === 'auto') {
    throw new Error(
      'Windows SYCL acceleration is not enabled in this worker manifest. This incomplete build refuses to fall back to CPU automatically. Build and install the self-contained Windows SYCL VSIX, or set localLlm.acceleration to cpu explicitly.',
    );
  }
  return 'cpu';
}

export async function prepareWorkerLaunch(
  input: PrepareWorkerLaunchInput,
): Promise<PreparedWorkerLaunch> {
  const bundle = await input.resolveBundle(input.target, input.mode);
  if (bundle.backend !== 'sycl') {
    return { bundle, backend: bundle.backend };
  }
  const syclDevice = await input.discoverSycl(bundle.executablePath);
  return { bundle, backend: bundle.backend, syclDevice };
}
