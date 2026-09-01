import { spawn } from 'node:child_process';
import { access } from 'node:fs/promises';
import path from 'node:path';

const defaultOneApiRoot = 'C:\\Program Files (x86)\\Intel\\oneAPI';
const maxEnvironmentOutputBytes = 10 * 1024 * 1024;

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

export async function loadOneApiEnvironment(baseEnv, dependencies = {}) {
  const accessFile = dependencies.accessFile ?? access;
  const runCmdScript = dependencies.runCmdScript ?? runWindowsCmdScript;
  const oneApiRoot = baseEnv.ONEAPI_ROOT || defaultOneApiRoot;
  const setvarsPath = path.win32.join(oneApiRoot, 'setvars.bat');
  try {
    await accessFile(setvarsPath);
  } catch {
    throw new Error(`oneAPI bootstrap batch file was not found: ${setvarsPath}`);
  }

  const script = [
    `@call "${setvarsPath}" intel64 --force >nul`,
    '@if errorlevel 1 exit /b %errorlevel%',
    '@set',
    '@exit /b 0',
  ].join('\r\n');
  const { stdout } = await runCmdScript(script, baseEnv);
  return { ...baseEnv, ...parseWindowsEnvironment(stdout) };
}

export async function runWindowsCmdScript(script, environment, spawnProcess = spawn) {
  return await new Promise((resolve, reject) => {
    const child = spawnProcess('cmd.exe', ['/d', '/q'], {
      env: environment,
      windowsHide: true,
      shell: false,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    let outputBytes = 0;
    let outputError;
    const capture = (stream, chunk) => {
      outputBytes += Buffer.byteLength(chunk);
      if (outputBytes > maxEnvironmentOutputBytes) {
        outputError ??= new Error('oneAPI environment output exceeded 10 MiB.');
        child.kill();
        return stream;
      }
      return stream + chunk;
    };
    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');
    child.stdout.on('data', (chunk) => {
      stdout = capture(stdout, chunk);
    });
    child.stderr.on('data', (chunk) => {
      stderr = capture(stderr, chunk);
    });
    child.once('error', reject);
    child.once('close', (code) => {
      if (outputError) {
        reject(outputError);
      } else if (code === 0) {
        resolve({ stdout, stderr });
      } else {
        const detail = stderr.trim();
        reject(new Error(
          `oneAPI bootstrap cmd.exe exited with code ${code ?? 'unknown'}${detail ? `: ${detail}` : '.'}`,
        ));
      }
    });
    child.stdin.end(`${script}\r\n`);
  });
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
