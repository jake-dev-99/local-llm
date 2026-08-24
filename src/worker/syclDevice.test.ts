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
  );
  assert.deepEqual(device, { id: 'SYCL0', description: 'Intel(R) Arc(TM) Graphics' });
});
