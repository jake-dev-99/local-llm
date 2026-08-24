import { randomBytes as createRandomBytes } from 'node:crypto';
import { spawn as spawnProcess } from 'node:child_process';
import { readFile, mkdtemp, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import process from 'node:process';
import * as net from 'node:net';
import { fileURLToPath } from 'node:url';

import {
  parseWorkerManifest,
  resolveWorkerBundle,
  verifyWorkerBundleFiles,
} from '../src/worker/workerManifest.ts';
import { discoverSycl0 } from '../src/worker/syclDevice.ts';

const ROOT = path.resolve(import.meta.dirname, '..');
const STARTUP_TIMEOUT_MS = 10 * 60 * 1_000;
const HEALTH_INTERVAL_MS = 500;
const CHAT_TIMEOUT_MS = 60 * 1_000;

export async function runSmokeWorker(args, dependencies = {}) {
  const options = {
    root: ROOT,
    statFile: stat,
    resolveBundle: resolveRequestedWorkerBundle,
    discoverSycl: discoverSycl0,
    allocatePort: allocateLoopbackPort,
    makeTempDirectory: mkdtemp,
    randomBytes: createRandomBytes,
    writeFile,
    removeDirectory: rm,
    spawnWorker: spawnProcess,
    fetchFn: fetch,
    log: console.log,
    stderr: process.stderr,
    startupTimeoutMs: STARTUP_TIMEOUT_MS,
    healthIntervalMs: HEALTH_INTERVAL_MS,
    chatTimeoutMs: CHAT_TIMEOUT_MS,
    ...dependencies,
  };
  const { backend, modelPath } = await parseSmokeWorkerArguments(args, options);
  const { bundle, executable } = await options.resolveBundle(options.root, backend);
  const device = backend === 'sycl' ? await options.discoverSycl(executable) : undefined;
  const port = await options.allocatePort();
  const keyDirectory = await options.makeTempDirectory(path.join(tmpdir(), 'local-llm-worker-smoke-'));
  const apiKey = options.randomBytes(32).toString('hex');
  const apiKeyFile = path.join(keyDirectory, 'api-key');
  const spawnAbort = new AbortController();
  let child;
  let removeSpawnErrorListener = () => undefined;

  try {
    await options.writeFile(apiKeyFile, `${apiKey}\n`, { mode: 0o600 });
    const workerArgs = buildExtensionEquivalentArguments(modelPath, port, apiKeyFile, backend);
    child = options.spawnWorker(executable, workerArgs, {
      cwd: path.dirname(executable),
      shell: false,
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    const spawnFailure = observeSpawnError(child, backend, executable);
    removeSpawnErrorListener = spawnFailure.removeListener;
    void spawnFailure.promise.catch((error) => {
      if (!spawnAbort.signal.aborted) spawnAbort.abort(error);
    });
    child.stdout?.pipe(options.stderr);
    child.stderr?.pipe(options.stderr);

    const baseUrl = `http://127.0.0.1:${port}`;
    await waitUntilHealthy(baseUrl, child, apiKey, {
      fetchFn: options.fetchFn,
      startupTimeoutMs: options.startupTimeoutMs,
      healthIntervalMs: options.healthIntervalMs,
      signal: spawnAbort.signal,
    });
    throwIfAborted(spawnAbort.signal);
    const { response, body } = await fetchTextWithTimeout(
      options.fetchFn,
      `${baseUrl}/v1/chat/completions`,
      {
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
      },
      options.chatTimeoutMs,
      spawnAbort.signal,
      'Chat completion',
    );
    if (!response.ok) {
      throw new Error(`Chat completion failed with HTTP ${response.status}: ${body}`);
    }

    options.log(`Selected backend: ${bundle.backend}`);
    options.log(`Executable: ${executable}`);
    options.log(`Detected device: ${device ? `${device.id}: ${device.description}` : 'CPU (not applicable)'}`);
    options.log('Health: OK');
    options.log(`Chat response: HTTP ${response.status}`);
  } finally {
    try {
      await terminateWorkerProcess(child);
    } finally {
      try {
        await options.removeDirectory(keyDirectory, { recursive: true, force: true });
      } finally {
        removeSpawnErrorListener();
      }
    }
  }
}

export async function parseSmokeWorkerArguments(args, { statFile = stat } = {}) {
  let backend;
  let modelPath;
  for (let index = 0; index < args.length; index += 2) {
    const flag = args[index];
    const value = args[index + 1];
    if (!value || !['--backend', '--model'].includes(flag)) {
      throw usageError();
    }
    if (flag === '--backend') backend = value;
    if (flag === '--model') modelPath = value;
  }
  if (!['sycl', 'cpu'].includes(backend) || !modelPath || !path.isAbsolute(modelPath)) {
    throw usageError();
  }
  const model = await statFile(modelPath);
  if (!model.isFile()) {
    throw new Error(`Smoke model must be an existing file: ${modelPath}`);
  }
  return { backend, modelPath };
}

export async function resolveRequestedWorkerBundle(root, backend, { readManifest = readFile } = {}) {
  const manifestPath = path.join(root, 'resources', 'workers', 'manifest.json');
  const manifest = parseWorkerManifest(JSON.parse(await readManifest(manifestPath, 'utf8')));
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

async function waitUntilHealthy(baseUrl, child, apiKey, options) {
  const deadline = Date.now() + options.startupTimeoutMs;
  while (Date.now() < deadline) {
    throwIfAborted(options.signal);
    if (child.exitCode !== null || child.killed) {
      throw new Error(`Worker exited while loading the model (code ${child.exitCode ?? 'unknown'}).`);
    }
    const remainingMs = Math.max(1, deadline - Date.now());
    try {
      const response = await fetchWithTimeout(
        options.fetchFn,
        `${baseUrl}/health`,
        { headers: { Authorization: `Bearer ${apiKey}` } },
        remainingMs,
        options.signal,
        'Health request',
      );
      if (response.ok) return;
    } catch (error) {
      if (options.signal?.aborted) throw options.signal.reason ?? error;
    }
    const delayMs = Math.min(options.healthIntervalMs, Math.max(0, deadline - Date.now()));
    if (delayMs > 0) await delay(delayMs, options.signal);
  }
  throw new Error(`Timed out after ${options.startupTimeoutMs / 1_000} seconds while loading the model.`);
}

async function fetchTextWithTimeout(fetchFn, url, init, timeoutMs, signal, label) {
  return await withRequestTimeout(async (requestSignal) => {
    const response = await fetchFn(url, { ...init, signal: requestSignal });
    return { response, body: await response.text() };
  }, timeoutMs, signal, label);
}

async function fetchWithTimeout(fetchFn, url, init, timeoutMs, signal, label) {
  return await withRequestTimeout(
    async (requestSignal) => await fetchFn(url, { ...init, signal: requestSignal }),
    timeoutMs,
    signal,
    label,
  );
}

async function withRequestTimeout(operation, timeoutMs, signal, label) {
  const controller = new AbortController();
  let timedOut = false;
  const timeoutError = new Error(`${label} timed out after ${timeoutMs / 1_000} seconds.`);
  const abortFromParent = () => controller.abort(signal.reason);
  if (signal?.aborted) abortFromParent();
  else signal?.addEventListener('abort', abortFromParent, { once: true });
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort(timeoutError);
  }, timeoutMs);
  try {
    const result = await operation(controller.signal);
    throwIfAborted(signal);
    return result;
  } catch (error) {
    if (signal?.aborted) throw signal.reason ?? error;
    if (timedOut) throw timeoutError;
    throw error;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', abortFromParent);
  }
}

export async function terminateWorkerProcess(child, { termTimeoutMs = 5_000, killTimeoutMs = 5_000 } = {}) {
  if (!child || child.exitCode !== null) return;
  child.kill('SIGTERM');
  if (await waitForWorkerExit(child, termTimeoutMs)) return;
  if (child.exitCode !== null) return;
  child.kill('SIGKILL');
  if (await waitForWorkerExit(child, killTimeoutMs)) return;
  throw new Error('Worker did not exit after SIGKILL.');
}

function observeSpawnError(child, backend, executable) {
  let rejectSpawnError;
  const promise = new Promise((_, reject) => {
    rejectSpawnError = reject;
  });
  const onError = (error) => {
    rejectSpawnError(new Error(`Failed to start ${backend} worker ${executable}: ${describeError(error)}`));
  };
  child.on('error', onError);
  return { promise, removeListener: () => child.off('error', onError) };
}

function waitForWorkerExit(child, timeoutMs) {
  if (child.exitCode !== null) return Promise.resolve(true);
  return new Promise((resolve) => {
    const onExit = () => finish(true);
    const timer = setTimeout(() => finish(false), timeoutMs);
    const finish = (exited) => {
      clearTimeout(timer);
      child.off('exit', onExit);
      resolve(exited);
    };
    child.once('exit', onExit);
  });
}

function delay(milliseconds, signal) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, milliseconds);
    const onAbort = () => {
      clearTimeout(timer);
      signal?.removeEventListener('abort', onAbort);
      reject(signal.reason ?? new Error('Worker startup was aborted.'));
    };
    if (signal?.aborted) onAbort();
    else signal?.addEventListener('abort', onAbort, { once: true });
  });
}

function throwIfAborted(signal) {
  if (signal?.aborted) throw signal.reason ?? new Error('Worker startup was aborted.');
}

function describeError(error) {
  return error instanceof Error ? error.message : String(error);
}

function usageError() {
  return new Error('Usage: npm run smoke:worker -- --backend sycl|cpu --model /absolute/path/to/model.gguf');
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await runSmokeWorker(process.argv.slice(2));
}
