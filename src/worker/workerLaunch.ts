import type {
  AccelerationMode,
  ResolvedWorkerBundle,
  WorkerBackend,
} from './workerManifest.js';
import type { DiscoveredSyclDevice, SyclDevice } from './syclDevice.js';

export interface LaunchableWorkerBundle extends ResolvedWorkerBundle {
  executablePath: string;
}

export interface PrepareWorkerLaunchInput {
  target: string;
  mode: AccelerationMode;
  syclDeviceOverride?: string;
  resolveBundle: (target: string, mode: AccelerationMode) => Promise<LaunchableWorkerBundle>;
  discoverSycl: (executable: string) => Promise<DiscoveredSyclDevice>;
}

export interface PreparedWorkerLaunch {
  bundle: LaunchableWorkerBundle;
  backend: WorkerBackend;
  syclDevice?: SyclDevice;
  environment?: NodeJS.ProcessEnv;
}

export async function prepareWorkerLaunch(
  input: PrepareWorkerLaunchInput,
): Promise<PreparedWorkerLaunch> {
  const bundle = await input.resolveBundle(input.target, input.mode);
  if (bundle.backend !== 'sycl') {
    return { bundle, backend: bundle.backend };
  }
  const discovered = await input.discoverSycl(bundle.executablePath);
  const { environment, ...syclDevice } = discovered;
  const override = input.syclDeviceOverride?.trim();
  const selected: SyclDevice = override
    ? { id: override as SyclDevice['id'], description: `${syclDevice.description} (override ${override})` }
    : syclDevice;
  return { bundle, backend: bundle.backend, syclDevice: selected, environment };
}

/**
 * The settings a running worker was started with. Changing any of them only
 * takes effect by starting a new worker; everything else is read per request.
 */
export interface LaunchSettings {
  contextSize: number;
  cpuThreads: number;
  acceleration: string;
  batchSize: number;
  microBatchSize: number;
  metalMemoryReserveMiB: number;
  syclDevice: string;
  pythonPath: string;
  pythonEnvFlavor: string;
}

/** Picks the launch settings out of a full configuration. */
export function launchSettings<T extends LaunchSettings>(config: T): LaunchSettings {
  return {
    contextSize: config.contextSize,
    cpuThreads: config.cpuThreads,
    acceleration: config.acceleration,
    batchSize: config.batchSize,
    microBatchSize: config.microBatchSize,
    metalMemoryReserveMiB: config.metalMemoryReserveMiB,
    syclDevice: config.syclDevice,
    pythonPath: config.pythonPath,
    pythonEnvFlavor: config.pythonEnvFlavor,
  };
}

/** Each launch setting that differs, as `name old → new`. */
export function changedLaunchSettings(running: LaunchSettings, current: LaunchSettings): string[] {
  return (Object.keys(running) as Array<keyof LaunchSettings>)
    .filter((name) => running[name] !== current[name])
    .map((name) => `${name} ${String(running[name]) || '(empty)'} → ${String(current[name]) || '(empty)'}`);
}
