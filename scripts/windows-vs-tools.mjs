import { spawn } from 'node:child_process';
import { access } from 'node:fs/promises';
import path from 'node:path';

const cmakeComponent = 'Microsoft.VisualStudio.Component.VC.CMake.Project';

export async function resolveVisualStudioBuildTools(environment, dependencies = {}) {
  const accessFile = dependencies.accessFile ?? access;
  const captureFile = dependencies.captureFile ?? captureExecutable;
  const programFilesX86 = environmentValue(environment, 'ProgramFiles(x86)')
    || 'C:\\Program Files (x86)';
  const vswhere = environment.LOCAL_LLM_VSWHERE
    || path.win32.join(programFilesX86, 'Microsoft Visual Studio', 'Installer', 'vswhere.exe');

  try {
    await accessFile(vswhere);
  } catch {
    throw new Error(
      `Visual Studio Installer locator was not found: ${vswhere}. `
      + 'Install Visual Studio Build Tools and the C++ CMake tools for Windows component.',
    );
  }

  const installationOutput = await captureFile(vswhere, [
    '-latest',
    '-products', '*',
    '-requires', cmakeComponent,
    '-property', 'installationPath',
  ], environment);
  const installationPath = installationOutput.split(/\r?\n/).find((line) => line.trim())?.trim();
  if (!installationPath) {
    throw new Error(
      'Visual Studio installation with C++ CMake tools for Windows was not found. '
      + 'In Visual Studio Installer, modify Build Tools and select that individual component.',
    );
  }

  const cmake = path.win32.join(
    installationPath,
    'Common7', 'IDE', 'CommonExtensions', 'Microsoft', 'CMake', 'CMake', 'bin', 'cmake.exe',
  );
  const ninja = path.win32.join(
    installationPath,
    'Common7', 'IDE', 'CommonExtensions', 'Microsoft', 'CMake', 'Ninja', 'ninja.exe',
  );
  await requireExecutable(accessFile, cmake, 'CMake');
  await requireExecutable(accessFile, ninja, 'Ninja');
  return { cmake, ninja };
}

async function requireExecutable(accessFile, executable, name) {
  try {
    await accessFile(executable);
  } catch {
    throw new Error(
      `${name} executable was not found: ${executable}. `
      + 'Repair the C++ CMake tools for Windows component in Visual Studio Installer.',
    );
  }
}

async function captureExecutable(command, args, environment) {
  return await new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      env: environment,
      windowsHide: true,
      shell: false,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');
    child.stdout.on('data', (chunk) => {
      stdout += chunk;
    });
    child.stderr.on('data', (chunk) => {
      stderr += chunk;
    });
    child.once('error', reject);
    child.once('close', (code) => {
      if (code === 0) {
        resolve(stdout);
      } else {
        const detail = stderr.trim();
        reject(new Error(
          `vswhere.exe exited with code ${code ?? 'unknown'}${detail ? `: ${detail}` : '.'}`,
        ));
      }
    });
  });
}

function environmentValue(environment, name) {
  const key = Object.keys(environment).find((candidate) => candidate.toLowerCase() === name.toLowerCase());
  return key ? environment[key] : undefined;
}
