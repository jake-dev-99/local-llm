import assert from 'node:assert/strict';
import path from 'node:path';
import test from 'node:test';

import {
  parseVerificationArguments,
  runWindowsWorkerVerification,
} from './verify-windows-worker.mjs';

test('runs the three native probes in order with one clean environment', async () => {
  const calls = [];
  const logs = [];
  const bundleDirectory = 'C:\\extension\\resources\\workers\\win32-x64\\sycl';
  await runWindowsWorkerVerification({
    bundleDirectory,
    baseEnvironment: {
      SystemRoot: 'C:\\Windows',
      Path: 'C:\\Intel\\oneAPI;C:\\other',
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

  const environment = {
    SystemRoot: 'C:\\Windows',
    PATH: `${bundleDirectory};C:\\Windows\\System32;C:\\Windows`,
  };
  assert.deepEqual(calls, [
    {
      executable: path.win32.join(bundleDirectory, 'sycl-ls.exe'),
      args: ['--verbose', '--ignore-device-selectors'],
      options: { cwd: bundleDirectory, env: environment },
    },
    {
      executable: path.win32.join(bundleDirectory, 'llama-server.exe'),
      args: ['--version'],
      options: { cwd: bundleDirectory, env: environment },
    },
    {
      executable: path.win32.join(bundleDirectory, 'llama-server.exe'),
      args: ['--list-devices'],
      options: { cwd: bundleDirectory, env: environment },
    },
  ]);
  assert.match(logs.join('\n'), /SYCL0: Intel Arc Pro 140T/);
});

test('reports decimal and unsigned hexadecimal exit code with complete output', async () => {
  const bundleDirectory = 'C:\\bundle';
  await assert.rejects(
    runWindowsWorkerVerification({
      bundleDirectory,
      baseEnvironment: { SystemRoot: 'C:\\Windows' },
      runProcess: async (executable) => ({
        code: executable.endsWith('sycl-ls.exe') ? 3221225477 : 0,
        stdout: 'loader stdout',
        stderr: 'access violation stderr',
      }),
      log: () => undefined,
    }),
    (error) => {
      assert.match(error.message, /3221225477 \(0xC0000005\)/);
      assert.match(error.message, /C:\\bundle\\sycl-ls\.exe/);
      assert.match(error.message, /Working directory: C:\\bundle/);
      assert.match(error.message, /loader stdout/);
      assert.match(error.message, /access violation stderr/);
      return true;
    },
  );
});

test('requires SYCL0 in llama-server device output', async () => {
  await assert.rejects(
    runWindowsWorkerVerification({
      bundleDirectory: 'C:\\bundle',
      baseEnvironment: { SystemRoot: 'C:\\Windows' },
      runProcess: async () => ({ code: 0, stdout: 'Available devices:\n', stderr: '' }),
      log: () => undefined,
    }),
    /llama-server\.exe --list-devices.*did not report SYCL0/is,
  );
});

test('accepts only an absolute optional bundle directory', () => {
  assert.deepEqual(parseVerificationArguments([]), {});
  assert.deepEqual(
    parseVerificationArguments(['--bundle', 'C:\\official\\sycl']),
    { bundleDirectory: 'C:\\official\\sycl' },
  );
  assert.throws(
    () => parseVerificationArguments(['--bundle', 'relative\\sycl']),
    /absolute Windows path/,
  );
  assert.throws(() => parseVerificationArguments(['--unknown', 'value']), /Usage:/);
});
