import assert from 'node:assert/strict';
import test from 'node:test';
import { discoverSycl0, parseSyclDevices } from './syclDevice.ts';

test('parses the first explicit SYCL device', () => {
  assert.deepEqual(parseSyclDevices(
    'Available devices:\n  SYCL0: Intel(R) Arc(TM) Graphics (15473 MiB, 15000 MiB free)\n',
  ), [{ id: 'SYCL0', description: 'Intel(R) Arc(TM) Graphics (15473 MiB, 15000 MiB free)' }]);
});

test('fails loudly when the selected executable exposes no SYCL GPU', async () => {
  await assert.rejects(
    discoverSycl0('llama-server.exe', async () => ({ stdout: 'Available devices:\n', stderr: '' })),
    /No SYCL GPU was reported.*localLlm\.acceleration.*cpu/s,
  );
});

test('preserves stderr from a DLL load failure', async () => {
  await assert.rejects(
    discoverSycl0('llama-server.exe', async () => {
      throw Object.assign(new Error('exit 3221225781'), { stderr: 'sycl8.dll was not found' });
    }),
    /sycl8\.dll was not found/,
  );
});

test('discovers SYCL0 when llama.cpp writes devices to stderr', async () => {
  const device = await discoverSycl0(
    'llama-server.exe',
    async () => ({ stdout: '', stderr: 'SYCL0: Intel(R) Arc(TM) Graphics\n' }),
    { KEEP_ME: 'yes' },
  );
  assert.deepEqual(device, {
    id: 'SYCL0',
    description: 'Intel(R) Arc(TM) Graphics',
    runtime: {
      adapter: 'level_zero',
      environment: {
        KEEP_ME: 'yes',
        ONEAPI_DEVICE_SELECTOR: 'level_zero:gpu',
      },
    },
  });
});

test('retries with the bundled OpenCL adapter when Level Zero discovery crashes', async () => {
  const attempts: NodeJS.ProcessEnv[] = [];
  const device = await discoverSycl0(
    'C:\\bundle\\llama-server.exe',
    async (_executable, _args, options) => {
      attempts.push(options?.env ?? {});
      if (attempts.length === 1) {
        throw new Error('exit 3221225477');
      }
      return { stdout: 'SYCL0: Intel(R) Arc(TM) Graphics\n', stderr: '' };
    },
    {
      KEEP_ME: 'yes',
      oneapi_device_selector: 'caller-value',
      Ur_Adapters_Force_Load: 'C:\\outside\\ur_adapter_level_zero.dll',
    },
  );

  assert.deepEqual(attempts, [
    {
      KEEP_ME: 'yes',
      ONEAPI_DEVICE_SELECTOR: 'level_zero:gpu',
    },
    {
      KEEP_ME: 'yes',
      ONEAPI_DEVICE_SELECTOR: 'opencl:gpu',
      UR_ADAPTERS_FORCE_LOAD: 'C:\\bundle\\ur_adapter_opencl.dll',
    },
  ]);
  assert.deepEqual(device, {
    id: 'SYCL0',
    description: 'Intel(R) Arc(TM) Graphics',
    runtime: {
      adapter: 'opencl',
      environment: attempts[1],
    },
  });
});
