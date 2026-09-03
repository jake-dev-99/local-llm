import { spawn } from 'node:child_process';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

import { verifiedWorkerBundle } from '../src/worker/workerIntegrity.ts';
import { cleanSyclEnvironment, parseSyclDevices } from '../src/worker/syclDevice.ts';
import { prepareWindowsWorkerArchive } from './windows-worker-archive.mjs';

const root = path.resolve(import.meta.dirname, '..');

export function parseVerificationArguments(args) {
  if (args.length === 0) return {};
  if (
    args.length !== 2 ||
    args[0] !== '--bundle' ||
    !args[1] ||
    !path.win32.isAbsolute(args[1])
  ) {
    throw new Error(
      'Usage: npm run verify:windows-worker -- [--bundle C:\\absolute\\path\\to\\bundle] '
      + '(bundle must be an absolute Windows path).',
    );
  }
  return { bundleDirectory: path.win32.normalize(args[1]) };
}

export async function runWindowsWorkerVerification(options) {
  const runProcess = options.runProcess ?? captureProcess;
  const log = options.log ?? console.log;
  const bundleDirectory = path.win32.normalize(options.bundleDirectory);
  const server = path.win32.join(bundleDirectory, 'llama-server.exe');
  const environment = cleanSyclEnvironment(server, options.baseEnvironment ?? process.env);
  const probes = [
    {
      label: 'sycl-ls.exe --verbose --ignore-device-selectors',
      executable: path.win32.join(bundleDirectory, 'sycl-ls.exe'),
      args: ['--verbose', '--ignore-device-selectors'],
    },
    {
      label: 'llama-server.exe --version',
      executable: server,
      args: ['--version'],
    },
    {
      label: 'llama-server.exe --list-devices',
      executable: server,
      args: ['--list-devices'],
      requireSycl0: true,
    },
  ];

  for (const probe of probes) {
    let result;
    try {
      result = await runProcess(probe.executable, probe.args, {
        cwd: bundleDirectory,
        env: environment,
      });
    } catch (error) {
      throw new Error(
        `${probe.label} could not start.\nExecutable: ${probe.executable}\n`
        + `Working directory: ${bundleDirectory}\n${describeError(error)}`,
      );
    }
    if (result.code !== 0) {
      throw new Error(formatProcessFailure(probe, result, bundleDirectory));
    }
    const output = `${result.stdout}\n${result.stderr}`.trim();
    if (probe.requireSycl0 && !parseSyclDevices(output).some(({ id }) => id === 'SYCL0')) {
      throw new Error(
        `${probe.label} exited successfully but did not report SYCL0.\n`
        + `Executable: ${probe.executable}\nWorking directory: ${bundleDirectory}\n`
        + `stdout:\n${result.stdout || '<empty>'}\nstderr:\n${result.stderr || '<empty>'}`,
      );
    }
    log(`[windows-worker-verify] ${probe.label}: PASS${output ? `\n${output}` : ''}`);
  }
}

function formatProcessFailure(probe, result, bundleDirectory) {
  const unsigned = Number(result.code) >>> 0;
  const hexadecimal = `0x${unsigned.toString(16).toUpperCase().padStart(8, '0')}`;
  return `${probe.label} failed with exit code ${result.code} (${hexadecimal}).\n`
    + `Executable: ${probe.executable}\nWorking directory: ${bundleDirectory}\n`
    + `stdout:\n${result.stdout || '<empty>'}\nstderr:\n${result.stderr || '<empty>'}`;
}

function captureProcess(executable, args, options) {
  return new Promise((resolve, reject) => {
    const child = spawn(executable, args, {
      cwd: options.cwd,
      env: options.env,
      shell: false,
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');
    child.stdout.on('data', (chunk) => { stdout += chunk; });
    child.stderr.on('data', (chunk) => { stderr += chunk; });
    child.once('error', reject);
    child.once('close', (code) => resolve({ code, stdout, stderr }));
  });
}

function describeError(error) {
  return error instanceof Error ? error.message : String(error);
}

async function main() {
  if (process.platform !== 'win32' || process.arch !== 'x64') {
    throw new Error('Windows worker verification must run on x64 Windows.');
  }
  const options = parseVerificationArguments(process.argv.slice(2));
  let bundleDirectory = options.bundleDirectory;
  if (!bundleDirectory) {
    await prepareWindowsWorkerArchive(root);
    const bundle = await verifiedWorkerBundle(root, 'win32-x64', 'auto');
    bundleDirectory = path.win32.dirname(bundle.executablePath);
  }
  await runWindowsWorkerVerification({ bundleDirectory });
}

const entry = process.argv[1] ? path.resolve(process.argv[1]) : undefined;
if (entry && entry === fileURLToPath(import.meta.url)) {
  await main();
}
