import assert from 'node:assert/strict';
import path from 'node:path';
import test from 'node:test';

import { resolveVisualStudioBuildTools } from './windows-vs-tools.mjs';

const installationPath = 'C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools';
const vswherePath = 'C:\\Program Files (x86)\\Microsoft Visual Studio\\Installer\\vswhere.exe';

test('finds Visual Studio bundled CMake and Ninja without relying on PATH', async () => {
  const accessed = [];
  let invocation;
  const tools = await resolveVisualStudioBuildTools(
    { 'ProgramFiles(x86)': 'C:\\Program Files (x86)' },
    {
      accessFile: async (file) => accessed.push(file),
      captureFile: async (command, args) => {
        invocation = { command, args };
        return `${installationPath}\r\n`;
      },
    },
  );

  assert.deepEqual(invocation, {
    command: vswherePath,
    args: [
      '-latest',
      '-products', '*',
      '-requires', 'Microsoft.VisualStudio.Component.VC.CMake.Project',
      '-property', 'installationPath',
    ],
  });
  assert.deepEqual(tools, {
    cmake: path.win32.join(
      installationPath,
      'Common7', 'IDE', 'CommonExtensions', 'Microsoft', 'CMake', 'CMake', 'bin', 'cmake.exe',
    ),
    ninja: path.win32.join(
      installationPath,
      'Common7', 'IDE', 'CommonExtensions', 'Microsoft', 'CMake', 'Ninja', 'ninja.exe',
    ),
  });
  assert.deepEqual(accessed, [vswherePath, tools.cmake, tools.ninja]);
});

test('fails before CMake with an actionable message when the CMake component is absent', async () => {
  await assert.rejects(
    resolveVisualStudioBuildTools(
      { 'ProgramFiles(x86)': 'C:\\Program Files (x86)' },
      {
        accessFile: async () => undefined,
        captureFile: async () => '\r\n',
      },
    ),
    /Visual Studio installation with C\+\+ CMake tools for Windows was not found/,
  );
});

test('reports the exact missing bundled executable', async () => {
  await assert.rejects(
    resolveVisualStudioBuildTools(
      { 'ProgramFiles(x86)': 'C:\\Program Files (x86)' },
      {
        accessFile: async (file) => {
          if (file.endsWith('ninja.exe')) {
            throw new Error('missing');
          }
        },
        captureFile: async () => `${installationPath}\r\n`,
      },
    ),
    new RegExp(`Ninja executable was not found: ${escapeRegExp(path.win32.join(
      installationPath,
      'Common7', 'IDE', 'CommonExtensions', 'Microsoft', 'CMake', 'Ninja', 'ninja.exe',
    ))}`),
  );
});

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
