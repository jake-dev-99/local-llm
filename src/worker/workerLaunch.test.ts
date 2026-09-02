import assert from 'node:assert/strict';
import test from 'node:test';
import {
  prepareWorkerLaunch,
  type LaunchableWorkerBundle,
} from './workerLaunch.ts';

test('prepares a SYCL bundle only after it reports SYCL0', async () => {
  const bundle = syclBundle();
  const environment = {
    KEEP_ME: 'yes',
    ONEAPI_DEVICE_SELECTOR: 'opencl:gpu',
    UR_ADAPTERS_FORCE_LOAD: 'C:\\extension\\resources\\workers\\win32-x64\\sycl\\ur_adapter_opencl.dll',
  };
  const prepared = await prepareWorkerLaunch({
    target: 'win32-x64',
    mode: 'auto',
    resolveBundle: async () => bundle,
    discoverSycl: async () => ({
      id: 'SYCL0',
      description: 'Intel Arc',
      runtime: { adapter: 'opencl', environment },
    }),
  });

  assert.deepEqual(prepared, {
    bundle,
    backend: 'sycl',
    syclDevice: { id: 'SYCL0', description: 'Intel Arc' },
    syclRuntime: { adapter: 'opencl', environment },
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
