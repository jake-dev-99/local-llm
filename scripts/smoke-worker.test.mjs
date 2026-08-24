import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { sha256File } from '../src/worker/workerManifest.ts';
import {
  parseSmokeWorkerArguments,
  resolveRequestedWorkerBundle,
  runSmokeWorker,
  terminateWorkerProcess,
} from './smoke-worker.mjs';

test('smoke options reject a relative model path before inspecting it', async () => {
  await assert.rejects(
    parseSmokeWorkerArguments(['--backend', 'cpu', '--model', 'model.gguf']),
    /model \/absolute\/path\/to\/model\.gguf/,
  );
});

test('Windows smoke maps sycl and cpu to their isolated bundles', async (context) => {
  const fixture = await workerFixture(context);
  const sycl = await resolveRequestedWorkerBundle(fixture.root, 'sycl');
  const cpu = await resolveRequestedWorkerBundle(fixture.root, 'cpu');

  assert.deepEqual(
    { bundleName: sycl.bundle.bundleName, backend: sycl.bundle.backend, executable: sycl.bundle.executable },
    {
      bundleName: 'sycl',
      backend: 'sycl',
      executable: 'resources/workers/win32-x64/sycl/llama-server.exe',
    },
  );
  assert.equal(cpu.bundle.bundleName, 'cpu');
  assert.equal(cpu.bundle.backend, 'cpu');
});

test('SYCL discovery completes before the worker is spawned', async () => {
  const events = [];
  const child = fakeChild({ exitOn: 'SIGTERM' });
  await runSmokeWorker(['--backend', 'sycl', '--model', '/models/test.gguf'], smokeDependencies({
    discoverSycl: async () => {
      events.push('discover');
      return { id: 'SYCL0', description: 'Intel Arc' };
    },
    spawnWorker: () => {
      events.push('spawn');
      return child;
    },
  }));

  assert.deepEqual(events, ['discover', 'spawn']);
});

test('health and chat requests both carry the temporary bearer credential', async () => {
  const requests = [];
  await runSmokeWorker(['--backend', 'cpu', '--model', '/models/test.gguf'], smokeDependencies({
    fetchFn: async (url, init = {}) => {
      requests.push({ url, init });
      return response(200);
    },
  }));

  assert.equal(requests.length, 2);
  assert.equal(requests[0].init.headers.Authorization, 'Bearer 746573742d6b6579');
  assert.equal(requests[1].init.headers.Authorization, 'Bearer 746573742d6b6579');
});

test('a stalled health request aborts by the startup deadline and cleans up', async () => {
  const cleanup = [];
  const child = fakeChild({ exitOn: 'SIGTERM' });
  await assert.rejects(
    runSmokeWorker(['--backend', 'cpu', '--model', '/models/test.gguf'], smokeDependencies({
      child,
      startupTimeoutMs: 20,
      healthIntervalMs: 1,
      fetchFn: hangingFetch,
      removeDirectory: async (directory) => cleanup.push(directory),
    })),
    /Timed out after 0\.02 seconds while loading the model/,
  );

  assert.deepEqual(child.kills, ['SIGTERM']);
  assert.deepEqual(cleanup, ['/tmp/local-llm-smoke-test']);
});

test('a stalled chat request aborts by its finite timeout and cleans up', async () => {
  const cleanup = [];
  const child = fakeChild({ exitOn: 'SIGTERM' });
  let call = 0;
  await assert.rejects(
    runSmokeWorker(['--backend', 'cpu', '--model', '/models/test.gguf'], smokeDependencies({
      child,
      chatTimeoutMs: 20,
      fetchFn: async (...args) => {
        call += 1;
        return call === 1 ? response(200) : hangingFetch(...args);
      },
      removeDirectory: async (directory) => cleanup.push(directory),
    })),
    /Chat completion timed out after 0\.02 seconds/,
  );

  assert.deepEqual(child.kills, ['SIGTERM']);
  assert.deepEqual(cleanup, ['/tmp/local-llm-smoke-test']);
});

test('a worker spawn error includes backend context and still removes the key directory', async () => {
  const cleanup = [];
  const child = fakeChild({ exitOn: 'SIGTERM' });
  await assert.rejects(
    runSmokeWorker(['--backend', 'sycl', '--model', '/models/test.gguf'], smokeDependencies({
      child,
      spawnWorker: () => {
        queueMicrotask(() => child.emit('error', new Error('missing runtime DLL')));
        return child;
      },
      fetchFn: hangingFetch,
      removeDirectory: async (directory) => cleanup.push(directory),
    })),
    /Failed to start sycl worker .*missing runtime DLL/,
  );

  assert.deepEqual(child.kills, ['SIGTERM']);
  assert.deepEqual(cleanup, ['/tmp/local-llm-smoke-test']);
});

test('forced worker termination waits for the SIGKILL exit event', async () => {
  const child = fakeChild({ exitOn: 'SIGKILL' });
  await terminateWorkerProcess(child, { termTimeoutMs: 1, killTimeoutMs: 20 });
  assert.deepEqual(child.kills, ['SIGTERM', 'SIGKILL']);
  assert.equal(child.exitCode, 0);
});

function smokeDependencies(overrides = {}) {
  const child = overrides.child ?? fakeChild({ exitOn: 'SIGTERM' });
  return {
    statFile: async () => ({ isFile: () => true }),
    resolveBundle: async (_root, backend) => ({
      bundle: {
        target: 'win32-x64',
        bundleName: backend === 'sycl' ? 'sycl' : 'cpu',
        backend,
        executable: `resources/workers/win32-x64/${backend === 'sycl' ? 'sycl' : 'cpu'}/llama-server.exe`,
        files: [],
      },
      executable: `C:/workers/${backend}/llama-server.exe`,
    }),
    discoverSycl: async () => ({ id: 'SYCL0', description: 'Intel Arc' }),
    allocatePort: async () => 43123,
    makeTempDirectory: async () => '/tmp/local-llm-smoke-test',
    randomBytes: () => Buffer.from('test-key'),
    writeFile: async () => undefined,
    removeDirectory: async () => undefined,
    spawnWorker: () => child,
    fetchFn: async () => response(200),
    log: () => undefined,
    startupTimeoutMs: 100,
    healthIntervalMs: 1,
    chatTimeoutMs: 100,
    ...overrides,
  };
}

function fakeChild({ exitOn } = {}) {
  const child = new EventEmitter();
  child.exitCode = null;
  child.killed = false;
  child.kills = [];
  child.stdout = { pipe() {} };
  child.stderr = { pipe() {} };
  child.kill = (signal) => {
    child.kills.push(signal);
    if (exitOn === signal) {
      setTimeout(() => {
        child.exitCode = 0;
        child.emit('exit', 0, signal);
      }, 0);
    }
    return true;
  };
  return child;
}

function response(status) {
  return { ok: status >= 200 && status < 300, status, text: async () => '' };
}

function hangingFetch(_url, init = {}) {
  return new Promise((_resolve, reject) => {
    init.signal.addEventListener('abort', () => reject(init.signal.reason), { once: true });
  });
}

async function workerFixture(context) {
  const root = await mkdtemp(path.join(tmpdir(), 'local-llm-smoke-worker-'));
  context.after(() => rm(root, { recursive: true, force: true }));
  const manifest = {
    manifestVersion: 2,
    llamaCppCommit: '60eeeb6082c1126bb8bc72902c83123cd056811b',
    llamaCppBuild: 'b10472',
    platforms: {
      'win32-x64': {
        modes: {
          auto: { bundle: 'sycl', backend: 'sycl' },
          cpu: { bundle: 'cpu', backend: 'cpu' },
        },
        bundles: {
          sycl: await workerBundle(root, 'resources/workers/win32-x64/sycl/llama-server.exe'),
          cpu: await workerBundle(root, 'resources/workers/win32-x64/cpu/llama-server.exe'),
        },
      },
    },
  };
  const manifestPath = path.join(root, 'resources', 'workers', 'manifest.json');
  await mkdir(path.dirname(manifestPath), { recursive: true });
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  return { root };
}

async function workerBundle(root, executable) {
  const absolutePath = path.join(root, ...executable.split('/'));
  await mkdir(path.dirname(absolutePath), { recursive: true });
  await writeFile(absolutePath, executable);
  return {
    executable,
    files: [{ path: executable, sha256: await sha256File(absolutePath) }],
  };
}
