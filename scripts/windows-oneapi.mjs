import { execFile } from 'node:child_process';
import { access } from 'node:fs/promises';
import { promisify } from 'node:util';
import path from 'node:path';

const execFileAsync = promisify(execFile);
const defaultOneApiRoot = 'C:\\Program Files (x86)\\Intel\\oneAPI';

export function parseWindowsEnvironment(stdout) {
  const environment = {};
  for (const line of stdout.split(/\r?\n/)) {
    if (!line || line.startsWith('=')) {
      continue;
    }
    const separator = line.indexOf('=');
    if (separator <= 0) {
      continue;
    }
    environment[line.slice(0, separator)] = line.slice(separator + 1);
  }
  return environment;
}

export async function loadOneApiEnvironment(baseEnv) {
  const oneApiRoot = baseEnv.ONEAPI_ROOT || defaultOneApiRoot;
  const setvarsPath = path.win32.join(oneApiRoot, 'setvars.bat');
  try {
    await access(setvarsPath);
  } catch {
    throw new Error(`oneAPI bootstrap batch file was not found: ${setvarsPath}`);
  }

  const command = `call "${setvarsPath}" intel64 --force >nul && set`;
  const { stdout } = await execFileAsync('cmd.exe', ['/d', '/s', '/c', command], {
    env: baseEnv,
    windowsHide: true,
    maxBuffer: 10 * 1024 * 1024,
  });
  return { ...baseEnv, ...parseWindowsEnvironment(stdout) };
}

export function resolveOneApiFiles(environment) {
  const oneApiRoot = environment.ONEAPI_ROOT || defaultOneApiRoot;
  const setvarsPath = path.win32.join(oneApiRoot, 'setvars.bat');
  const levelZeroSdkPath = environment.LEVEL_ZERO_V1_SDK_PATH;
  if (!levelZeroSdkPath) {
    throw new Error('oneAPI environment is missing LEVEL_ZERO_V1_SDK_PATH.');
  }
  const vcToolsRedistDir = environment.VCToolsRedistDir;
  if (!vcToolsRedistDir) {
    throw new Error('oneAPI environment is missing VCToolsRedistDir.');
  }
  return { oneApiRoot, setvarsPath, levelZeroSdkPath, vcToolsRedistDir };
}
