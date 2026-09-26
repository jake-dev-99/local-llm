import { spawn } from 'node:child_process';
import path from 'node:path';
import process from 'node:process';

import { cleanSyclEnvironment, parseSyclDevices } from '../src/worker/syclDevice.ts';
import { prepareLinuxWorkerArchive } from './linux-sycl-worker.mjs';

const root = path.resolve(import.meta.dirname, '..');

export function parseLinuxVerificationArguments(args) {
  if (args.length === 0) return {};
  if (
    args.length !== 2 ||
    args[0] !== '--bundle' ||
    !args[1] ||
    !path.posix.isAbsolute(args[1])
  ) {
    throw new Error(
      'Usage: npm run verify:linux-worker -- [--bundle /absolute/path/to/bundle] '
      + '(bundle must be an absolute POSIX path).',
    );
  }
  return { bundleDirectory: path.posix.normalize(args[1]) };
}

export async function runLinuxWorkerVerification(options) {
  const runProcess = options.runProcess ?? captureProcess;
  const log = options.log ?? console.log;
  const bundleDirectory = path.posix.normalize(options.bundleDirectory);
  const server = path.posix.join(bundleDirectory, 'llama-server');
  const environment = cleanSyclEnvironment(server, options.baseEnvironment ?? process.env);
  const probes = [
    {
      label: 'llama-server --version',
      executable: server,
      args: ['--version'],
    },
    {
      label: 'llama-server --list-devices',
      executable: server,
      args: ['--list-devices'],
      requireSyclDevice: true,
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
      throw new Error(
        `${probe.label} failed with exit code ${result.code}.\n`
        + `Executable: ${probe.executable}\nWorking directory: ${bundleDirectory}\n`
        + `stdout:\n${result.stdout || '<empty>'}\nstderr:\n${result.stderr || '<empty>'}`,
      );
    }
    const output = `${result.stdout}\n${result.stderr}`.trim();
    if (probe.requireSyclDevice && parseSyclDevices(output).length === 0) {
      throw new Error(
        `${probe.label} exited successfully but did not report a SYCL device.\n`
        + `Executable: ${probe.executable}\nWorking directory: ${bundleDirectory}\n`
        + `stdout:\n${result.stdout || '<empty>'}\nstderr:\n${result.stderr || '<empty>'}`,
      );
    }
    log(`[linux-worker-verify] ${probe.label}: PASS${output ? `\n${output}` : ''}`);
  }
}

function captureProcess(executable, args, options) {
  return new Promise((resolve, reject) => {
    const child = spawn(executable, args, {
      cwd: options.cwd,
      env: options.env,
      shell: false,
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
  if (process.platform !== 'linux' || process.arch !== 'x64') {
    throw new Error('Linux worker verification must run on x64 Linux.');
  }
  const options = parseLinuxVerificationArguments(process.argv.slice(2));
  let bundleDirectory = options.bundleDirectory;
  if (!bundleDirectory) {
    await prepareLinuxWorkerArchive(root);
    bundleDirectory = path.join(root, 'resources', 'workers', 'linux-x64', 'sycl');
  }
  await runLinuxWorkerVerification({ bundleDirectory });
  console.log('[linux-worker-verify] Linux SYCL worker verification passed.');
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}

function fileURLToPath(url) {
  return new URL(url).pathname;
}
