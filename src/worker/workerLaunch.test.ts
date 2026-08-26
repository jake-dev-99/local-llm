import assert from 'node:assert/strict';
import test from 'node:test';
import {
  prepareWorkerLaunch,
  resolveExecutionBackend,
  type LaunchableWorkerBundle,
} from './workerLaunch.ts';

test('an incomplete Windows auto build fails instead of silently running CPU', () => {
  assert.equal(resolveExecutionBackend('darwin-arm64', 'auto'), 'metal');
  assert.equal(resolveExecutionBackend('darwin-arm64', 'cpu'), 'cpu');
  assert.throws(
    () => resolveExecutionBackend('win32-x64', 'auto'),
    /Windows SYCL.*not enabled.*refuses to fall back.*localLlm\.acceleration.*cpu/is,
  );
  assert.equal(resolveExecutionBackend('win32-x64', 'cpu'), 'cpu');
});

test('prepares a SYCL bundle only after it reports SYCL0', async () => {
  const bundle = syclBundle();
  const prepared = await prepareWorkerLaunch({
    target: 'win32-x64',
    mode: 'auto',
    resolveBundle: async () => bundle,
    discoverSycl: async () => ({ id: 'SYCL0', description: 'Intel Arc' }),
  });

  assert.deepEqual(prepared, {
    bundle,
    backend: 'sycl',
    syclDevice: { id: 'SYCL0', description: 'Intel Arc' },
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
