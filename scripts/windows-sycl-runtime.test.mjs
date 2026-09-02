import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';

import {
  activeOneApiRuntimeDirectories,
  discoverSyclDynamicResources,
  verifyStagedSyclBundle,
} from './windows-sycl-runtime.mjs';

test('discovers dynamic Level Zero resources beside the active Unified Runtime loader', async (context) => {
  const fixture = await runtimeFixture(context, 'sycl42.dll');
  const files = await discoverSyclDynamicResources([fixture.compilerBin]);
  assert.deepEqual(files.map((file) => path.basename(file)).sort(), [
    'libsycl-fallback-bfloat16.spv',
    'libsycl-native-bfloat16.spv',
    'ur_adapter_level_zero.dll',
    'ur_adapter_level_zero_v2.dll',
    'ur_loader.dll',
    'ur_win_proxy_loader.dll',
  ].sort());
});

test('restricts runtime discovery to active PATH directories below ONEAPI_ROOT', () => {
  assert.deepEqual(activeOneApiRuntimeDirectories('/oneapi', [
    '/oneapi/compiler/2026.1/bin',
    '/unrelated/bin',
    '/oneapi/mkl/2026.1/bin',
  ]), [
    '/oneapi/compiler/2026.1/bin',
    '/oneapi/mkl/2026.1/bin',
  ]);
});

test('reports a missing semantic runtime role and searched directories', async (context) => {
  const fixture = await runtimeFixture(context, 'sycl42.dll');
  await rm(fixture.file('ur_loader.dll'));
  await assert.rejects(
    discoverSyclDynamicResources([fixture.compilerBin]),
    /Unified Runtime loader.*ur_loader\.dll.*compiler.*bin/is,
  );
});

test('rejects distinct active Unified Runtime loaders instead of choosing by PATH order', async (context) => {
  const first = await runtimeFixture(context, 'sycl42.dll', 'compiler/2026.1/bin');
  const second = await runtimeFixture(context, 'sycl42.dll', 'compiler/latest/bin');
  await assert.rejects(
    discoverSyclDynamicResources([first.compilerBin, second.compilerBin]),
    /exactly one active Unified Runtime loader.*found 2/is,
  );
});

test('staged verification removes oneAPI development paths and variables', async () => {
  let observed;
  await verifyStagedSyclBundle({
    staging: 'C:\\bundle',
    systemRoot: 'C:\\Windows',
    oneApiRoot: 'C:\\Intel\\oneAPI',
    baseEnvironment: {
      PATH: 'C:\\Intel\\oneAPI\\compiler\\latest\\bin;C:\\other',
      ONEAPI_ROOT: 'C:\\Intel\\oneAPI',
      CMPLR_ROOT: 'C:\\Intel\\oneAPI\\compiler\\latest',
      LIB: 'C:\\Intel\\lib',
      INCLUDE: 'C:\\Intel\\include',
      LIBRARY_PATH: 'C:\\Intel\\oneAPI\\compiler\\latest\\lib',
      nlspath: 'C:\\Intel\\oneAPI\\compiler\\latest\\share\\locale',
      Pkg_Config_Path: 'C:\\Intel\\oneAPI\\compiler\\latest\\lib\\pkgconfig',
      ur_adapters_force_load: 'C:\\Intel\\oneAPI\\compiler\\latest\\bin\\ur_adapter_level_zero.dll',
      Ur_Adapters_Search_Path: 'C:\\Intel\\oneAPI\\compiler\\latest\\bin',
      CUSTOM_TOOL_ROOT: 'C:\\Intel\\oneAPI\\compiler\\2026.1',
      MIXED_RUNTIME_PATHS: 'C:\\unrelated;C:\\Intel\\oneAPI\\mkl\\2026.1\\bin',
      UNRELATED_PATH: 'C:\\unrelated',
      SystemRoot: 'C:\\Windows',
      KEEP_ME: 'yes',
    },
    runProcess: async (command, args, options) => {
      observed = { command, args, options };
      return { code: 0, stdout: 'SYCL0: Intel Arc Graphics\r\n', stderr: '' };
    },
  });
  assert.equal(observed.command, 'C:\\bundle\\llama-server.exe');
  assert.deepEqual(observed.args, ['--list-devices']);
  assert.equal(observed.options.env.PATH, 'C:\\bundle;C:\\Windows\\System32;C:\\Windows');
  assert.equal(observed.options.env.ONEAPI_ROOT, undefined);
  assert.equal(observed.options.env.CMPLR_ROOT, undefined);
  assert.equal(observed.options.env.LIB, undefined);
  assert.equal(observed.options.env.LIBRARY_PATH, undefined);
  assert.equal(observed.options.env.nlspath, undefined);
  assert.equal(observed.options.env.Pkg_Config_Path, undefined);
  assert.equal(observed.options.env.ur_adapters_force_load, undefined);
  assert.equal(observed.options.env.Ur_Adapters_Search_Path, undefined);
  assert.equal(observed.options.env.CUSTOM_TOOL_ROOT, undefined);
  assert.equal(observed.options.env.MIXED_RUNTIME_PATHS, undefined);
  assert.equal(observed.options.env.UNRELATED_PATH, 'C:\\unrelated');
  assert.equal(observed.options.env.KEEP_ME, 'yes');
});

test('staged verification preserves output when the worker cannot load', async () => {
  await assert.rejects(verifyStagedSyclBundle({
    staging: 'C:\\bundle',
    systemRoot: 'C:\\Windows',
    oneApiRoot: 'C:\\Intel\\oneAPI',
    baseEnvironment: {},
    runProcess: async () => ({ code: 3221225781, stdout: '', stderr: 'missing runtime' }),
  }), /clean-environment launch.*3221225781.*missing runtime/is);
});

test('staged verification requires SYCL0', async () => {
  await assert.rejects(verifyStagedSyclBundle({
    staging: 'C:\\bundle',
    systemRoot: 'C:\\Windows',
    oneApiRoot: 'C:\\Intel\\oneAPI',
    baseEnvironment: {},
    runProcess: async () => ({ code: 0, stdout: 'no devices', stderr: '' }),
  }), /did not report SYCL0.*no devices/is);
});

test('staged verification requires the oneAPI root used to build the worker', async () => {
  await assert.rejects(verifyStagedSyclBundle({
    staging: 'C:\\bundle',
    systemRoot: 'C:\\Windows',
    baseEnvironment: {},
    runProcess: async () => ({ code: 0, stdout: 'SYCL0: Intel Arc Graphics', stderr: '' }),
  }), /requires the active oneAPI root/i);
});

async function runtimeFixture(context, syclRuntimeName, compilerDirectory = 'compiler/2026.1/bin') {
  const root = await mkdtemp(path.join(tmpdir(), 'windows-sycl-runtime-'));
  context.after(() => rm(root, { recursive: true, force: true }));
  const compilerBin = path.join(root, compilerDirectory);
  await mkdir(compilerBin, { recursive: true });
  for (const name of [
    syclRuntimeName,
    'ur_adapter_level_zero.dll',
    'ur_adapter_level_zero_v2.dll',
    'libsycl-fallback-bfloat16.spv',
    'libsycl-native-bfloat16.spv',
    'ur_loader.dll',
    'ur_win_proxy_loader.dll',
  ]) {
    await writeFile(path.join(compilerBin, name), name);
  }
  return {
    compilerBin,
    file: (name) => path.join(compilerBin, name),
  };
}
