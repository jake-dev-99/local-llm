import { randomBytes } from 'node:crypto';
import { spawn } from 'node:child_process';
import { readFile, mkdtemp, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import process from 'node:process';
import * as net from 'node:net';

import {
  parseWorkerManifest,
  resolveWorkerBundle,
  verifyWorkerBundleFiles,
} from '../src/worker/workerManifest.ts';
import { discoverSycl0 } from '../src/worker/syclDevice.ts';

const ROOT = path.resolve(import.meta.dirname, '..');
const STARTUP_TIMEOUT_MS = 10 * 60 * 1_000;
const HEALTH_INTERVAL_MS = 500;

async function main() {
  const { backend, modelPath } = await parseArguments(process.argv.slice(2));
  const { bundle, executable } = await resolveRequestedBundle(ROOT, backend);
  const device = backend === 'sycl' ? await discoverSycl0(executable) : undefined;
  const port = await allocateLoopbackPort();
  const keyDirectory = await mkdtemp(path.join(tmpdir(), 'local-llm-worker-smoke-'));
  const apiKey = randomBytes(32).toString('hex');
  const apiKeyFile = path.join(keyDirectory, 'api-key');
  let child;

  try {
    await writeFile(apiKeyFile, `${apiKey}\n`, { mode: 0o600 });
    const args = buildExtensionEquivalentArguments(modelPath, port, apiKeyFile, backend);
    child = spawn(executable, args, {
      cwd: path.dirname(executable),
      shell: false,
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    child.stdout.pipe(process.stderr);
    child.stderr.pipe(process.stderr);

    await waitUntilHealthy(`http://127.0.0.1:${port}`, child);
    const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'local',
        messages: [{ role: 'user', content: 'Reply with OK.' }],
        stream: false,
        max_tokens: 1,
      }),
    });
    const responseBody = await response.text();
    if (!response.ok) {
      throw new Error(`Chat completion failed with HTTP ${response.status}: ${responseBody}`);
    }

    console.log(`Selected backend: ${bundle.backend}`);
    console.log(`Executable: ${executable}`);
    console.log(`Detected device: ${device ? `${device.id}: ${device.description}` : 'CPU (not applicable)'}`);
    console.log('Health: OK');
    console.log(`Chat response: HTTP ${response.status}`);
  } finally {
    await terminate(child);
    await rm(keyDirectory, { recursive: true, force: true });
  }
}

async function parseArguments(args) {
  let backend;
  let modelPath;
  for (let index = 0; index < args.length; index += 2) {
    const flag = args[index];
    const value = args[index + 1];
    if (!value || !['--backend', '--model'].includes(flag)) {
      throw new Error('Usage: npm run smoke:worker -- --backend sycl|cpu --model /absolute/path/to/model.gguf');
    }
    if (flag === '--backend') backend = value;
    if (flag === '--model') modelPath = value;
  }
  if (!['sycl', 'cpu'].includes(backend) || !modelPath || !path.isAbsolute(modelPath)) {
    throw new Error('Usage: npm run smoke:worker -- --backend sycl|cpu --model /absolute/path/to/model.gguf');
  }
  const model = await stat(modelPath);
  if (!model.isFile()) {
    throw new Error(`Smoke model must be an existing file: ${modelPath}`);
  }
  return { backend, modelPath };
}

async function resolveRequestedBundle(root, backend) {
  const manifestPath = path.join(root, 'resources', 'workers', 'manifest.json');
  const manifest = parseWorkerManifest(JSON.parse(await readFile(manifestPath, 'utf8')));
  const bundle = resolveWorkerBundle(manifest, 'win32-x64', backend === 'sycl' ? 'auto' : 'cpu');
  await verifyWorkerBundleFiles(root, bundle);
  return {
    bundle,
    executable: path.join(root, ...bundle.executable.split('/')),
  };
}

function buildExtensionEquivalentArguments(modelPath, port, apiKeyFile, backend) {
  const args = [
    '--model', modelPath,
    '--host', '127.0.0.1',
    '--port', String(port),
    '--api-key-file', apiKeyFile,
    '--parallel', '1',
    '--batch-size', '256',
    '--ubatch-size', '64',
    '--jinja',
    '--no-webui',
  ];
  if (backend === 'sycl') {
    args.push('--fit', 'off', '--n-gpu-layers', '99', '--device', 'SYCL0', '--split-mode', 'none', '--main-gpu', '0');
  } else {
    args.push('--fit', 'off', '--n-gpu-layers', '0', '--device', 'none', '--no-op-offload');
  }
  return args;
}

async function allocateLoopbackPort() {
  return await new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      if (!address || typeof address === 'string') {
        server.close();
        reject(new Error('Unable to allocate a loopback port.'));
        return;
      }
      server.close((error) => (error ? reject(error) : resolve(address.port)));
    });
  });
}

async function waitUntilHealthy(baseUrl, child) {
  const deadline = Date.now() + STARTUP_TIMEOUT_MS;
  while (Date.now() < deadline) {
    if (child.exitCode !== null || child.killed) {
      throw new Error(`Worker exited while loading the model (code ${child.exitCode ?? 'unknown'}).`);
    }
    try {
      if ((await fetch(`${baseUrl}/health`)).ok) return;
    } catch {
      // The worker has not opened the loopback port yet.
    }
    await new Promise((resolve) => setTimeout(resolve, HEALTH_INTERVAL_MS));
  }
  throw new Error(`Timed out after ${STARTUP_TIMEOUT_MS / 1_000} seconds while loading the model.`);
}

async function terminate(child) {
  if (!child || child.exitCode !== null) return;
  child.kill('SIGTERM');
  const exited = await new Promise((resolve) => {
    const timer = setTimeout(() => resolve(false), 5_000);
    child.once('exit', () => {
      clearTimeout(timer);
      resolve(true);
    });
  });
  if (!exited && child.exitCode === null) child.kill('SIGKILL');
}

await main();
