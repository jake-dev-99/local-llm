import assert from 'node:assert/strict';
import test from 'node:test';
import {
  changedLaunchSettings,
  launchSettings,
  prepareWorkerLaunch,
  type LaunchableWorkerBundle,
} from './workerLaunch.ts';

test('prepares a SYCL bundle only after it reports SYCL0', async () => {
  const bundle = syclBundle();
  const environment = {
    KEEP_ME: 'yes',
    PATH: 'C:\\extension\\resources\\workers\\win32-x64\\sycl;C:\\Windows\\System32;C:\\Windows',
  };
  const prepared = await prepareWorkerLaunch({
    target: 'win32-x64',
    mode: 'auto',
    resolveBundle: async () => bundle,
    discoverSycl: async () => ({
      id: 'SYCL0',
      description: 'Intel Arc',
      environment,
    }),
  });

  assert.deepEqual(prepared, {
    bundle,
    backend: 'sycl',
    syclDevice: { id: 'SYCL0', description: 'Intel Arc' },
    environment,
  });
});

test('a SYCL preflight failure does not resolve or invoke the CPU bundle', async () => {
  const selected: string[] = [];
  await assert.rejects(prepareWorkerLaunch({
    target: 'win32-x64',
    mode: 'auto',
    resolveBundle: async (_target, mode) => {
      selected.push(mode);
      return syclBundle();
    },
    discoverSycl: async () => { throw new Error('device discovery failed'); },
  }), /device discovery failed/);
  assert.deepEqual(selected, ['auto']);
});

function syclBundle(): LaunchableWorkerBundle {
  return {
    target: 'win32-x64',
    bundleName: 'sycl',
    backend: 'sycl',
    executable: 'resources/workers/win32-x64/sycl/llama-server.exe',
    executablePath: 'C:\\extension\\resources\\workers\\win32-x64\\sycl\\llama-server.exe',
    files: [],
  };
}

const baseConfig = {
  contextSize: 32768,
  cpuThreads: 0,
  acceleration: 'auto' as const,
  batchSize: 256,
  microBatchSize: 64,
  metalMemoryReserveMiB: 1024,
  pythonPath: '',
  pythonEnvFlavor: 'auto' as const,
  maxOutputTokens: 8192,
  temperature: 0.2,
};

test('a worker launch setting change is named, so the next request reloads with it', () => {
  const running = launchSettings(baseConfig);
  assert.deepEqual(
    changedLaunchSettings(running, launchSettings({ ...baseConfig, batchSize: 1024, microBatchSize: 256 })),
    ['batchSize 256 → 1024', 'microBatchSize 64 → 256'],
  );
  assert.deepEqual(
    changedLaunchSettings(running, launchSettings({ ...baseConfig, pythonEnvFlavor: 'cpu' })),
    ['pythonEnvFlavor auto → cpu'],
  );
});

test('per-request settings never reload the worker', () => {
  const running = launchSettings(baseConfig);
  assert.deepEqual(
    changedLaunchSettings(running, launchSettings({ ...baseConfig, maxOutputTokens: 16384, temperature: 0.7 })),
    [],
  );
});
