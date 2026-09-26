import assert from 'node:assert/strict';
import test from 'node:test';

import {
  parseLinuxVerificationArguments,
  runLinuxWorkerVerification,
} from './verify-linux-worker.mjs';

test('runs version then list-devices probes in order with one clean environment', async () => {
  const calls = [];
  const logs = [];
  const bundleDirectory = '/extension/resources/workers/linux-x64/sycl';
  await runLinuxWorkerVerification({
    bundleDirectory,
    baseEnvironment: {
      PATH: '/opt/intel/oneapi/bin:/usr/bin',
      LD_LIBRARY_PATH: '/opt/intel/oneapi/lib',
      ONEAPI_DEVICE_SELECTOR: 'level_zero:gpu',
    },
    runProcess: async (executable, args, options) => {
      calls.push({ executable, args, options });
      return {
        code: 0,
        stdout: args.includes('--list-devices') ? 'SYCL0: Intel Arc Pro 140T\n' : 'ok\n',
        stderr: '',
      };
    },
    log: (message) => logs.push(message),
  });

  const environment = calls[0].options.env;
  assert.equal(calls.length, 2);
  assert.deepEqual([calls[0].executable, calls[0].args], [
    '/extension/resources/workers/linux-x64/sycl/llama-server',
    ['--version'],
  ]);
  assert.deepEqual([calls[1].executable, calls[1].args], [
    '/extension/resources/workers/linux-x64/sycl/llama-server',
    ['--list-devices'],
  ]);
  // Both probes share one clean environment: bundle-first PATH/LD_LIBRARY_PATH,
  // caller selectors stripped.
  assert.deepEqual(calls[1].options.env, environment);
  assert.match(environment.PATH, /^\/extension\/resources\/workers\/linux-x64\/sycl:/);
  assert.match(environment.LD_LIBRARY_PATH, /^\/extension\/resources\/workers\/linux-x64\/sycl:/);
  assert.equal(environment.ONEAPI_DEVICE_SELECTOR, undefined);
  assert.match(logs.join('\n'), /SYCL0: Intel Arc Pro 140T/);
});

test('reports exit code with complete output', async () => {
  await assert.rejects(
    runLinuxWorkerVerification({
      bundleDirectory: '/bundle',
      baseEnvironment: {},
      runProcess: async (executable, args) => ({
        code: args.includes('--version') ? 1 : 0,
        stdout: 'loader stdout',
        stderr: 'missing library stderr',
      }),
      log: () => undefined,
    }),
    (error) => {
      assert.match(error.message, /llama-server --version failed with exit code 1/);
      assert.match(error.message, /Executable: \/bundle\/llama-server/);
      assert.match(error.message, /Working directory: \/bundle/);
      assert.match(error.message, /loader stdout/);
      assert.match(error.message, /missing library stderr/);
      return true;
    },
  );
});

test('requires a SYCL device in llama-server device output', async () => {
  await assert.rejects(
    runLinuxWorkerVerification({
      bundleDirectory: '/bundle',
      baseEnvironment: {},
      runProcess: async () => ({ code: 0, stdout: 'Available devices:\n', stderr: '' }),
      log: () => undefined,
    }),
    /llama-server --list-devices.*did not report a SYCL device/is,
  );
});

test('accepts only an absolute optional POSIX bundle directory', () => {
  assert.deepEqual(parseLinuxVerificationArguments([]), {});
  assert.deepEqual(
    parseLinuxVerificationArguments(['--bundle', '/official/sycl']),
    { bundleDirectory: '/official/sycl' },
  );
  assert.throws(
    () => parseLinuxVerificationArguments(['--bundle', 'relative/sycl']),
    /absolute POSIX path/,
  );
  assert.throws(() => parseLinuxVerificationArguments(['--unknown', 'value']), /Usage:/);
});
