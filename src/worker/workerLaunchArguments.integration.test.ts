import assert from 'node:assert/strict';
import test from 'node:test';
import { build, type Plugin } from 'esbuild';
import type { LocalLlmConfig } from '../domain';

type BuildWorkerArguments = (
  modelPath: string,
  port: number,
  apiKeyFile: string,
  config: LocalLlmConfig,
  backend: 'metal' | 'sycl' | 'cpu',
  concurrentWorkerBytes?: number,
) => string[];

test('an automatic context window omits --ctx-size so llama.cpp can fit the model', async () => {
  const buildArguments = await loadBuildWorkerArguments();
  const args = buildArguments(
    '/models/qwen.gguf', 60000, '/keys/worker.key', config({ contextSize: 0 }), 'metal',
  );

  // Passing --ctx-size at all pins the window. Passing --ctx-size 0 is worse: it sets
  // fit_params_min_ctx to UINT32_MAX inside llama.cpp and disables context reduction.
  assert.equal(args.includes('--ctx-size'), false);
  assert.equal(args.includes('-c'), false);
});

test('an explicit context window is still passed through unchanged', async () => {
  const buildArguments = await loadBuildWorkerArguments();
  const args = buildArguments(
    '/models/qwen.gguf', 60000, '/keys/worker.key', config({ contextSize: 32768 }), 'metal',
  );

  assert.deepEqual(valueFor(args, '--ctx-size'), '32768');
});

test('the memory margin never drops below the llama.cpp default of 1024 MiB', async () => {
  const buildArguments = await loadBuildWorkerArguments();
  const args = buildArguments(
    '/models/qwen.gguf', 60000, '/keys/worker.key',
    config({ metalMemoryReserveMiB: 256 }), 'metal',
  );

  assert.deepEqual(valueFor(args, '--fit-target'), '1024');
});

test('a second worker this extension owns is added to the memory margin', async () => {
  const buildArguments = await loadBuildWorkerArguments();
  const sixGiB = 6 * 1024 * 1024 * 1024;
  const args = buildArguments(
    '/models/qwen.gguf', 60000, '/keys/worker.key',
    config({ metalMemoryReserveMiB: 2048 }), 'metal', sixGiB,
  );

  // llama.cpp measures free device memory as its own Metal budget minus its own
  // allocation, so it cannot see the other worker.
  assert.deepEqual(valueFor(args, '--fit-target'), String(2048 + 6144));
});

test('Apple auto acceleration uses conservative batches and llama.cpp memory fitting', async () => {
  const buildArguments = await loadBuildWorkerArguments();
  const args = buildArguments('/models/qwen.gguf', 60000, '/keys/worker.key', config(), 'metal');

  assert.deepEqual(valueFor(args, '--batch-size'), '256');
  assert.deepEqual(valueFor(args, '--ubatch-size'), '64');
  assert.deepEqual(valueFor(args, '--fit'), 'on');
  assert.deepEqual(valueFor(args, '--fit-target'), '4096');
  assert.equal(args.includes('--n-gpu-layers'), false);
  assert.equal(args.includes('--device'), false);
});

test('CPU execution explicitly disables device offload while retaining bounded batches', async () => {
  const buildArguments = await loadBuildWorkerArguments();
  const args = buildArguments(
    '/models/qwen.gguf',
    60000,
    '/keys/worker.key',
    config({ acceleration: 'cpu' }),
    'cpu',
  );

  assert.deepEqual(valueFor(args, '--batch-size'), '256');
  assert.deepEqual(valueFor(args, '--ubatch-size'), '64');
  assert.deepEqual(valueFor(args, '--fit'), 'off');
  assert.deepEqual(valueFor(args, '--n-gpu-layers'), '0');
  assert.deepEqual(valueFor(args, '--device'), 'none');
  assert.equal(args.includes('--no-op-offload'), true);
});

test('Windows SYCL explicitly selects SYCL0 and requests GPU layers', async () => {
  const buildArguments = await loadBuildWorkerArguments();
  const args = buildArguments(
    '/models/qwen.gguf', 60000, '/keys/worker.key', config({ acceleration: 'auto' }), 'sycl',
  );

  assert.deepEqual(valueFor(args, '--fit'), 'off');
  assert.deepEqual(valueFor(args, '--device'), 'SYCL0');
  assert.deepEqual(valueFor(args, '--n-gpu-layers'), '99');
  assert.deepEqual(valueFor(args, '--split-mode'), 'none');
  assert.deepEqual(valueFor(args, '--main-gpu'), '0');
  assert.equal(args.includes('--no-op-offload'), false);
});

function config(overrides: Partial<LocalLlmConfig> = {}): LocalLlmConfig {
  return {
    modelDirectory: '/models',
    defaultModelId: '',
    contextSize: 32768,
    maxTools: 8,
    maxAgentToolRounds: 8,
    maxOutputTokens: 2048,
    maxToolCallTokens: 512,
    startupTimeoutMilliseconds: 600_000,
    cpuThreads: 0,
    acceleration: 'auto',
    batchSize: 256,
    microBatchSize: 64,
    metalMemoryReserveMiB: 4096,
    temperature: 0.2,
    inlineEnabled: true,
    inlineMaxTokens: 64,
    inlineDebounceMilliseconds: 250,
    logLevel: 'info',
    ...overrides,
  };
}

function valueFor(args: readonly string[], flag: string): string | undefined {
  const index = args.indexOf(flag);
  return index >= 0 ? args[index + 1] : undefined;
}

async function loadBuildWorkerArguments(): Promise<BuildWorkerArguments> {
  const vscodeStub: Plugin = {
    name: 'vscode-stub',
    setup(buildContext) {
      buildContext.onResolve({ filter: /^vscode$/ }, () => ({
        path: 'vscode',
        namespace: 'test-stub',
      }));
      buildContext.onLoad({ filter: /.*/, namespace: 'test-stub' }, () => ({
        contents: `
          export class EventEmitter {}
          export class CancellationError extends Error {}
          export const workspace = { getConfiguration() { return {}; } };
          export const ConfigurationTarget = { Global: 1 };
        `,
        loader: 'js',
      }));
    },
  };
  const bundled = await build({
    entryPoints: ['src/worker/workerManager.ts'],
    absWorkingDir: process.cwd(),
    bundle: true,
    format: 'esm',
    platform: 'node',
    target: 'node26',
    banner: {
      js: "import { createRequire as __createRequire } from 'node:module'; const require = __createRequire(process.cwd() + '/');",
    },
    plugins: [vscodeStub],
    write: false,
  });
  const source = bundled.outputFiles[0]?.contents;
  assert.ok(source, 'esbuild returned the bundled WorkerManager');
  const url = `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`;
  const loaded = await import(url) as { buildWorkerArguments?: BuildWorkerArguments };
  const buildWorkerArguments = loaded.buildWorkerArguments;
  assert.equal(
    typeof buildWorkerArguments,
    'function',
    'WorkerManager must export buildWorkerArguments',
  );
  return buildWorkerArguments as BuildWorkerArguments;
}
