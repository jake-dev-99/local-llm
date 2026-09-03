import assert from 'node:assert/strict';
import test from 'node:test';
import { cleanSyclEnvironment, discoverSycl0, parseSyclDevices } from './syclDevice.ts';

test('parses the first explicit SYCL device', () => {
  assert.deepEqual(parseSyclDevices(
    'Available devices:\n  SYCL0: Intel(R) Arc(TM) Graphics (15473 MiB, 15000 MiB free)\n',
  ), [{ id: 'SYCL0', description: 'Intel(R) Arc(TM) Graphics (15473 MiB, 15000 MiB free)' }]);
});

test('builds a selector-free environment limited to the bundle and Windows', () => {
  assert.deepEqual(cleanSyclEnvironment(
    'C:\\extension\\workers\\sycl\\llama-server.exe',
    {
      KEEP_ME: 'yes',
      SystemRoot: 'C:\\Windows',
      Path: 'C:\\Intel\\oneAPI;C:\\other',
      oneapi_device_selector: 'level_zero:gpu',
      Sycl_Device_Filter: 'gpu',
      Ur_Adapters_Force_Load: 'C:\\outside\\adapter.dll',
      ur_adapters_search_path: 'C:\\outside',
    },
  ), {
    KEEP_ME: 'yes',
    SystemRoot: 'C:\\Windows',
    PATH: 'C:\\extension\\workers\\sycl;C:\\Windows\\System32;C:\\Windows',
  });
});

test('fails loudly after one attempt when the selected executable exposes no SYCL GPU', async () => {
  let attempts = 0;
  await assert.rejects(
    discoverSycl0('llama-server.exe', async () => {
      attempts += 1;
      return { stdout: 'Available devices:\n', stderr: '' };
    }),
    /No SYCL GPU was reported.*localLlm\.acceleration.*cpu/s,
  );
  assert.equal(attempts, 1);
});

test('preserves stderr from a DLL load failure without retrying an adapter', async () => {
  let attempts = 0;
  await assert.rejects(
    discoverSycl0('llama-server.exe', async () => {
      attempts += 1;
      throw Object.assign(new Error('exit 3221225781'), { stderr: 'sycl8.dll was not found' });
    }),
    /sycl8\.dll was not found/,
  );
  assert.equal(attempts, 1);
});

test('discovers SYCL0 from stderr and returns the exact clean environment', async () => {
  const observed: Array<{
    executable: string;
    args: string[];
    options: { env: NodeJS.ProcessEnv };
  }> = [];
  const device = await discoverSycl0(
    'C:\\bundle\\llama-server.exe',
    async (executable, args, options) => {
      observed.push({ executable, args, options });
      return { stdout: '', stderr: 'SYCL0: Intel(R) Arc(TM) Graphics\n' };
    },
    {
      KEEP_ME: 'yes',
      SystemRoot: 'C:\\Windows',
      ONEAPI_DEVICE_SELECTOR: 'caller-value',
      UR_ADAPTERS_FORCE_LOAD: 'C:\\outside\\ur_adapter_opencl.dll',
    },
  );
  const environment = {
    KEEP_ME: 'yes',
    SystemRoot: 'C:\\Windows',
    PATH: 'C:\\bundle;C:\\Windows\\System32;C:\\Windows',
  };
  assert.deepEqual(observed, [{
    executable: 'C:\\bundle\\llama-server.exe',
    args: ['--list-devices'],
    options: { env: environment },
  }]);
  assert.deepEqual(device, {
    id: 'SYCL0',
    description: 'Intel(R) Arc(TM) Graphics',
    environment,
  });
});
