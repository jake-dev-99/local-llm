import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import test from 'node:test';

import {
  loadOneApiEnvironment,
  parseWindowsEnvironment,
  resolveOneApiFiles,
  runWindowsCmdScript,
} from './windows-oneapi.mjs';

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

test('bootstraps a oneAPI batch path containing spaces through cmd stdin', async () => {
  const baseEnvironment = {
    ONEAPI_ROOT: 'C:\\Program Files (x86)\\Intel\\oneAPI',
    KEEP_ME: 'base',
  };
  let observedScript = '';
  const environment = await loadOneApiEnvironment(baseEnvironment, {
    accessFile: async () => undefined,
    runCmdScript: async (script, receivedEnvironment) => {
      observedScript = script;
      assert.equal(receivedEnvironment, baseEnvironment);
      return {
        stdout: 'SETVARS_COMPLETED=1\r\nPATH=C:\\Intel\\bin;C:\\Windows\r\n',
        stderr: '',
      };
    },
  });

  assert.match(
    observedScript,
    /^@call "C:\\Program Files \(x86\)\\Intel\\oneAPI\\setvars\.bat" intel64 --force >nul\r\n/,
  );
  assert.match(observedScript, /\r\n@set\r\n@exit \/b 0$/);
  assert.deepEqual(environment, {
    ONEAPI_ROOT: 'C:\\Program Files (x86)\\Intel\\oneAPI',
    KEEP_ME: 'base',
    SETVARS_COMPLETED: '1',
    PATH: 'C:\\Intel\\bin;C:\\Windows',
  });
});

test('cmd transport writes the batch script to stdin instead of a slash-c argument', async () => {
  const child = new EventEmitter();
  child.stdin = new PassThrough();
  child.stdout = new PassThrough();
  child.stderr = new PassThrough();
  child.kill = () => true;
  let input = '';
  child.stdin.setEncoding('utf8');
  child.stdin.on('data', (chunk) => {
    input += chunk;
  });

  const script = '@call "C:\\Program Files (x86)\\Intel\\oneAPI\\setvars.bat" intel64 --force >nul';
  const resultPromise = runWindowsCmdScript(script, { KEEP_ME: 'yes' }, (command, args, options) => {
    assert.equal(command, 'cmd.exe');
    assert.deepEqual(args, ['/d', '/q']);
    assert.equal(args.includes('/c'), false);
    assert.equal(options.shell, false);
    assert.deepEqual(options.env, { KEEP_ME: 'yes' });
    queueMicrotask(() => {
      child.stdout.end('SETVARS_COMPLETED=1\r\n');
      child.stderr.end();
      child.emit('close', 0);
    });
    return child;
  });

  assert.deepEqual(await resultPromise, {
    stdout: 'SETVARS_COMPLETED=1\r\n',
    stderr: '',
  });
  assert.equal(input, `${script}\r\n`);
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
