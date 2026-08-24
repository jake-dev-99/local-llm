import assert from 'node:assert/strict';
import test from 'node:test';

import { cmakeOptionsForBuild, parseWorkerBuildOptions } from './worker-build-options.mjs';

test('native Windows defaults to building both backends', () => {
  assert.deepEqual(parseWorkerBuildOptions([], 'win32-x64'), {
    target: 'win32-x64',
    backend: 'all',
  });
});

test('SYCL is native-Windows-only and uses the pinned backend flags', () => {
  assert.throws(
    () => parseWorkerBuildOptions(['--target', 'win32-x64', '--backend', 'sycl'], 'darwin-arm64'),
    /SYCL worker must be built on native x64 Windows/,
  );
  const flags = cmakeOptionsForBuild({
    target: 'win32-x64',
    backend: 'sycl',
    hostTarget: 'win32-x64',
    oneApiEnvironment: {},
  });
  assert.equal(flags.includes('-DGGML_SYCL=ON'), true);
  assert.equal(flags.includes('-DGGML_SYCL_F16=ON'), false);
  assert.equal(flags.includes('-DBUILD_SHARED_LIBS=ON'), true);
  assert.equal(flags.includes('-DCMAKE_C_COMPILER=cl'), true);
  assert.equal(flags.includes('-DCMAKE_CXX_COMPILER=icx'), true);
});

test('Windows CPU cross-build keeps the llvm-mingw toolchain flags', () => {
  const flags = cmakeOptionsForBuild({
    target: 'win32-x64',
    backend: 'cpu',
    hostTarget: 'darwin-arm64',
    llvmMingwRoot: '/toolchains/llvm-mingw',
  });

  assert.equal(flags.includes('-DCMAKE_SYSTEM_NAME=Windows'), true);
  assert.equal(
    flags.includes('-DCMAKE_C_COMPILER=/toolchains/llvm-mingw/bin/x86_64-w64-mingw32-clang'),
    true,
  );
  assert.equal(flags.includes('-DGGML_SYCL=ON'), false);
});
