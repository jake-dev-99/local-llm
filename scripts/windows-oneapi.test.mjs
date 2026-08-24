import assert from 'node:assert/strict';
import test from 'node:test';

import { loadOneApiEnvironment, parseWindowsEnvironment, resolveOneApiFiles } from './windows-oneapi.mjs';

test('parses cmd set output without corrupting values containing equals signs', () => {
  assert.deepEqual(parseWindowsEnvironment('ONEAPI_ROOT=C:\\Intel\\oneAPI\r\nPATH=C:\\A;C:\\B=2\r\n'), {
    ONEAPI_ROOT: 'C:\\Intel\\oneAPI',
    PATH: 'C:\\A;C:\\B=2',
  });
});

test('reports the exact missing oneAPI bootstrap path', async () => {
  await assert.rejects(
    loadOneApiEnvironment({ ONEAPI_ROOT: 'C:\\missing' }),
    /C:\\missing\\setvars.bat/,
  );
});

test('ignores cmd pseudo-variables and resolves emitted toolchain paths', () => {
  assert.deepEqual(parseWindowsEnvironment('=C:=C:\\repo\r\nPATH=C:\\bin\r\n'), {
    PATH: 'C:\\bin',
  });
  assert.deepEqual(resolveOneApiFiles({
    ONEAPI_ROOT: 'C:\\Intel\\oneAPI',
    LEVEL_ZERO_V1_SDK_PATH: 'C:\\level-zero',
    VCToolsRedistDir: 'C:\\VS\\VC\\Redist',
  }), {
    oneApiRoot: 'C:\\Intel\\oneAPI',
    setvarsPath: 'C:\\Intel\\oneAPI\\setvars.bat',
    levelZeroSdkPath: 'C:\\level-zero',
    vcToolsRedistDir: 'C:\\VS\\VC\\Redist',
  });
});

test('requires the Level Zero SDK from the bootstrapped environment', () => {
  assert.throws(
    () => resolveOneApiFiles({ ONEAPI_ROOT: 'C:\\Intel\\oneAPI', VCToolsRedistDir: 'C:\\VS\\VC\\Redist' }),
    /LEVEL_ZERO_V1_SDK_PATH/,
  );
});
