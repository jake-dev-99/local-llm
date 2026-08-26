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
