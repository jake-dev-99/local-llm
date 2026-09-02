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

export async function identifyOneApiCompiler(environment, runProcess) {
  const result = await runProcess('icx', ['--version'], { env: environment });
  const output = [result.stdout, result.stderr].filter(Boolean).join('\n').trim();
  if (result.code !== 0) {
    throw new Error(`Could not identify Intel oneAPI compiler; icx exited with code ${result.code ?? 'unknown'}${output ? `:\n${output}` : '.'}`);
  }
  const banner = output.split(/\r?\n/).map((line) => line.trim()).find(Boolean);
  if (!banner || !/intel|oneapi|dpc\+\+/i.test(banner)) {
    throw new Error(`Could not identify Intel oneAPI compiler from icx --version output${output ? `:\n${output}` : '.'}`);
  }
  const version = banner.match(/\b(\d{4})(?:\.\d+)+\b/);
  if (!version || Number(version[1]) < 2026) {
    throw new Error(
      `Intel oneAPI 2026 or newer is required for the Windows SYCL worker; detected: ${banner}.`,
    );
  }
  return banner;
}

export async function loadOneApiEnvironment(baseEnv, dependencies = {}) {
  const accessFile = dependencies.accessFile ?? access;
  const runCmdScript = dependencies.runCmdScript ?? runWindowsCmdScript;
  const visualStudioDeveloperCommand = dependencies.visualStudioDeveloperCommand;
  const oneApiRoot = baseEnv.ONEAPI_ROOT || defaultOneApiRoot;
  const setvarsPath = path.win32.join(oneApiRoot, 'setvars.bat');
  try {
    await accessFile(setvarsPath);
  } catch {
    throw new Error(`oneAPI bootstrap batch file was not found: ${setvarsPath}`);
  }
  if (!visualStudioDeveloperCommand) {
    throw new Error('Visual Studio developer environment bootstrap path is required for oneAPI.');
  }
  try {
    await accessFile(visualStudioDeveloperCommand);
  } catch {
    throw new Error(
      `Visual Studio developer environment bootstrap was not found: ${visualStudioDeveloperCommand}`,
    );
  }

  const script = [
    `@call "${visualStudioDeveloperCommand}" -arch=amd64 -host_arch=amd64 >nul`,
    '@if errorlevel 1 exit /b %errorlevel%',
    `@call "${setvarsPath}" intel64 --force >nul`,
    '@if errorlevel 1 exit /b %errorlevel%',
    '@set',
    '@exit /b 0',
  ].join('\r\n');
  const { stdout } = await runCmdScript(script, baseEnv);
  const environment = { ...baseEnv, ...parseWindowsEnvironment(stdout) };
  requireEnvironmentValue(
    environment,
    'VSCMD_VER',
    'Visual Studio developer environment did not initialize (VSCMD_VER is missing).',
  );
  requireEnvironmentValue(
    environment,
    'LIB',
    'Visual Studio developer environment is missing LIB, so Windows SDK libraries such as kernel32.lib cannot be linked.',
  );
  requireEnvironmentValue(
    environment,
    'INCLUDE',
    'Visual Studio developer environment is missing INCLUDE, so Windows SDK headers cannot be compiled.',
  );
  await requireLibrary(environment, 'kernel32.lib', accessFile);
  return environment;
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
  const vcToolsRedistDir = environment.VCToolsRedistDir;
  if (!vcToolsRedistDir) {
    throw new Error('oneAPI environment is missing VCToolsRedistDir.');
  }
  return { oneApiRoot, setvarsPath, vcToolsRedistDir };
}

function requireEnvironmentValue(environment, name, message) {
  const key = Object.keys(environment).find((candidate) => candidate.toLowerCase() === name.toLowerCase());
  if (!key || !environment[key]) {
    throw new Error(message);
  }
}

async function requireLibrary(environment, library, accessFile) {
  const key = Object.keys(environment).find((candidate) => candidate.toLowerCase() === 'lib');
  const directories = environment[key].split(';')
    .map((directory) => directory.trim().replace(/^"(.*)"$/, '$1'))
    .filter(Boolean);
  for (const directory of directories) {
    try {
      await accessFile(path.win32.join(directory, library));
      return;
    } catch {
      // Continue through every library directory before reporting the broken SDK environment.
    }
  }
  throw new Error(
    `${library} was not found in the Visual Studio LIB directories. `
    + 'Repair or install the Windows 10/11 SDK through Visual Studio Installer.',
  );
}
