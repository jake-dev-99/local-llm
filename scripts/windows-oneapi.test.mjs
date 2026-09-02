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
    loadOneApiEnvironment(
      { ONEAPI_ROOT: 'C:\\missing' },
      { visualStudioDeveloperCommand: 'C:\\Visual Studio\\Common7\\Tools\\VsDevCmd.bat' },
    ),
    /C:\\missing\\setvars.bat/,
  );
});

test('bootstraps Visual Studio x64 before oneAPI through cmd stdin', async () => {
  const baseEnvironment = {
    ONEAPI_ROOT: 'C:\\Program Files (x86)\\Intel\\oneAPI',
    KEEP_ME: 'base',
  };
  const visualStudioDeveloperCommand = 'C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\Common7\\Tools\\VsDevCmd.bat';
  let observedScript = '';
  const environment = await loadOneApiEnvironment(baseEnvironment, {
    visualStudioDeveloperCommand,
    accessFile: async () => undefined,
    runCmdScript: async (script, receivedEnvironment) => {
      observedScript = script;
      assert.equal(receivedEnvironment, baseEnvironment);
      return {
        stdout: [
          'VSCMD_VER=17.14.0',
          'LIB=C:\\Windows Kits\\10\\Lib\\um\\x64',
          'INCLUDE=C:\\Windows Kits\\10\\Include',
          'SETVARS_COMPLETED=1',
          'PATH=C:\\Intel\\bin;C:\\Windows',
          '',
        ].join('\r\n'),
        stderr: '',
      };
    },
  });

  assert.match(
    observedScript,
    /^@call "C:\\Program Files \(x86\)\\Microsoft Visual Studio\\2022\\BuildTools\\Common7\\Tools\\VsDevCmd\.bat" -arch=amd64 -host_arch=amd64 >nul\r\n/,
  );
  assert.match(
    observedScript,
    /\r\n@if errorlevel 1 exit \/b %errorlevel%\r\n@call "C:\\Program Files \(x86\)\\Intel\\oneAPI\\setvars\.bat" intel64 --force >nul\r\n/,
  );
  assert.match(observedScript, /\r\n@set\r\n@exit \/b 0$/);
  assert.deepEqual(environment, {
    ONEAPI_ROOT: 'C:\\Program Files (x86)\\Intel\\oneAPI',
    KEEP_ME: 'base',
    VSCMD_VER: '17.14.0',
    LIB: 'C:\\Windows Kits\\10\\Lib\\um\\x64',
    INCLUDE: 'C:\\Windows Kits\\10\\Include',
    SETVARS_COMPLETED: '1',
    PATH: 'C:\\Intel\\bin;C:\\Windows',
  });
});

test('fails before CMake when the bootstrapped environment has no Windows SDK libraries', async () => {
  await assert.rejects(
    loadOneApiEnvironment(
      { ONEAPI_ROOT: 'C:\\Intel\\oneAPI' },
      {
        visualStudioDeveloperCommand: 'C:\\Visual Studio\\Common7\\Tools\\VsDevCmd.bat',
        accessFile: async () => undefined,
        runCmdScript: async () => ({
          stdout: 'VSCMD_VER=17.14.0\r\nINCLUDE=C:\\Windows Kits\\10\\Include\r\n',
          stderr: '',
        }),
      },
    ),
    /Visual Studio developer environment is missing LIB.*kernel32\.lib/,
  );
});

test('fails before CMake when kernel32.lib is absent from the Visual Studio LIB directories', async () => {
  await assert.rejects(
    loadOneApiEnvironment(
      { ONEAPI_ROOT: 'C:\\Intel\\oneAPI' },
      {
        visualStudioDeveloperCommand: 'C:\\Visual Studio\\Common7\\Tools\\VsDevCmd.bat',
        accessFile: async (file) => {
          if (file.endsWith('kernel32.lib')) {
            throw new Error('missing');
          }
        },
        runCmdScript: async () => ({
          stdout: [
            'VSCMD_VER=17.14.0',
            'LIB=C:\\VC\\lib;C:\\Windows Kits\\10\\Lib\\um\\x64',
            'INCLUDE=C:\\Windows Kits\\10\\Include',
            '',
          ].join('\r\n'),
          stderr: '',
        }),
      },
    ),
    /kernel32\.lib was not found in the Visual Studio LIB directories/,
  );
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

test('ignores cmd pseudo-variables', () => {
  assert.deepEqual(parseWindowsEnvironment('=C:=C:\\repo\r\nPATH=C:\\bin\r\n'), {
    PATH: 'C:\\bin',
  });
});

test('does not require an unused standalone Level Zero SDK path', () => {
  assert.deepEqual(
    resolveOneApiFiles({
      ONEAPI_ROOT: 'C:\\Intel\\oneAPI',
      VCToolsRedistDir: 'C:\\VS\\VC\\Redist',
    }),
    {
      oneApiRoot: 'C:\\Intel\\oneAPI',
      setvarsPath: 'C:\\Intel\\oneAPI\\setvars.bat',
      vcToolsRedistDir: 'C:\\VS\\VC\\Redist',
    },
  );
});
