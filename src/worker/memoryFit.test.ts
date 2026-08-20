import assert from 'node:assert/strict';
import test from 'node:test';
import {
  LLAMA_CPP_DEFAULT_FIT_TARGET_MIB,
  parseFittedContext,
  parseFreeDeviceMemoryMiB,
  resolveFitTargetMiB,
} from './memoryFit.ts';

test('automatic sizing is never more aggressive than plain llama-server', () => {
  assert.equal(resolveFitTargetMiB({ reserveMiB: 0 }), LLAMA_CPP_DEFAULT_FIT_TARGET_MIB);
  assert.equal(resolveFitTargetMiB({ reserveMiB: -5000 }), LLAMA_CPP_DEFAULT_FIT_TARGET_MIB);
});

test('the configured reserve is passed through once it exceeds the llama.cpp default', () => {
  assert.equal(resolveFitTargetMiB({ reserveMiB: 2048 }), 2048);
});

test('a second worker this extension owns is added to the margin', () => {
  // llama.cpp reports free device memory as its own Metal budget minus its own
  // allocation, so a concurrent worker is invisible to it and must be added here.
  const sixGiB = 6 * 1024 * 1024 * 1024;

  assert.equal(
    resolveFitTargetMiB({ reserveMiB: 2048, concurrentWorkerBytes: sixGiB }),
    2048 + 6144,
  );
});

test('a missing or nonsensical concurrent worker size changes nothing', () => {
  assert.equal(resolveFitTargetMiB({ reserveMiB: 2048 }), 2048);
  assert.equal(resolveFitTargetMiB({ reserveMiB: 2048, concurrentWorkerBytes: -1 }), 2048);
  assert.equal(
    resolveFitTargetMiB({ reserveMiB: 2048, concurrentWorkerBytes: Number.NaN }),
    2048,
  );
});

test('reads the context reduction llama.cpp reports', () => {
  const line = 'common_params_fit_impl: context size reduced from 131072 to 49920 '
    + '-> need 20297 MiB less memory in total';

  assert.deepEqual(parseFittedContext(line), {
    trainedContextSize: 131072,
    fittedContextSize: 49920,
  });
});

test('no reduction line means no reduction happened', () => {
  assert.equal(
    parseFittedContext('common_params_fit_impl: no changes needed'),
    undefined,
  );
});

test('reads the device memory budget llama.cpp believes it has', () => {
  const line = 'common_params_fit_impl: projected to use 54478 MiB of device memory '
    + 'vs. 38338 MiB of free device memory';

  assert.equal(parseFreeDeviceMemoryMiB(line), 38338);
  assert.equal(parseFreeDeviceMemoryMiB('nothing to see'), undefined);
});
