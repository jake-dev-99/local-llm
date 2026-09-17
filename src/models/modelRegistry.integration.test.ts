import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, utimesSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import * as path from 'node:path';
import test from 'node:test';
import { build } from 'esbuild';
import type { InstalledModel } from '../domain.ts';

/**
 * The registry imports `vscode`, so it is bundled here with that module stubbed
 * rather than imported directly. The alternative — threading a fake through the
 * constructor — would change production code to suit the test.
 */
interface TestRegistry {
  initialize(): Promise<void>;
  list(): readonly InstalledModel[];
}

type TestRegistryConstructor = new (
  state: { get: <T>(key: string, fallback: T) => T; update: (key: string, value: unknown) => Promise<void> },
  logger: { info(message: string): void; error(message: string, error?: unknown): void },
) => TestRegistry;

const VSCODE_STUB = `
  export class EventEmitter {
    event = () => ({ dispose() {} });
    fire() {}
    dispose() {}
  }
  export class CancellationError extends Error {}
`;

let cached: TestRegistryConstructor | undefined;

async function loadRegistry(): Promise<TestRegistryConstructor> {
  if (cached) {
    return cached;
  }
  const bundled = await build({
    entryPoints: ['src/models/modelRegistry.ts'],
    absWorkingDir: process.cwd(),
    bundle: true,
    format: 'esm',
    platform: 'node',
    target: 'node26',
    write: false,
    plugins: [{
      name: 'stub-vscode',
      setup(pluginBuild) {
        pluginBuild.onResolve({ filter: /^vscode$/ }, () => ({
          path: 'vscode', namespace: 'stub',
        }));
        pluginBuild.onLoad({ filter: /.*/, namespace: 'stub' }, () => ({
          contents: VSCODE_STUB, loader: 'js',
        }));
      },
    }],
  });
  const source = bundled.outputFiles[0]?.contents;
  assert.ok(source, 'esbuild returned the bundled ModelRegistry');
  const url = `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`;
  const loaded = await import(url) as { ModelRegistry?: TestRegistryConstructor };
  assert.ok(loaded.ModelRegistry, 'bundled module exports ModelRegistry');
  cached = loaded.ModelRegistry;
  return cached;
}

function memento(stored: unknown[]) {
  const state = { value: stored };
  return {
    state,
    get: <T,>(_key: string, fallback: T): T => (state.value as unknown as T) ?? fallback,
    update: async (_key: string, value: unknown) => { state.value = value as unknown[]; },
  };
}

const logger = { info: () => undefined, error: () => undefined };

function shard(header: string, payloadBytes: number): Buffer {
  const body = Buffer.from(header);
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(body.length));
  return Buffer.concat([prefix, body, Buffer.alloc(payloadBytes)]);
}

function checkpointDirectory(payloadBytes = 64): string {
  const directory = mkdtempSync(path.join(tmpdir(), 'local-llm-registry-'));
  writeFileSync(path.join(directory, 'config.json'), JSON.stringify({
    architectures: ['Qwen3MoeForCausalLM'],
    max_position_embeddings: 8192,
  }));
  writeFileSync(
    path.join(directory, 'model.safetensors'),
    shard('{"w":{"dtype":"BF16","shape":[8],"data_offsets":[0,16]}}', payloadBytes),
  );
  return directory;
}

/** A record as written by a build that predates Safetensors support. */
function legacyGgufRecord(filePath: string): Record<string, unknown> {
  return {
    id: 'legacy-model',
    name: 'Legacy Model',
    filePath,
    fileSize: 32,
    fileModifiedAt: 1_700_000_000_000,
    sha256: 'abc',
    source: 'import',
    filename: 'legacy.gguf',
    installedAt: '2026-01-01T00:00:00.000Z',
    capabilities: { toolCalling: 'supported', fillInMiddle: 'supported' },
    runtimeProfile: {
      validatedAt: '2026-01-01T00:00:00.000Z',
      loadedContextSize: 4096,
      hasChatTemplate: true,
      supportsTools: true,
      supportsToolCalls: true,
      supportsSystemRole: true,
    },
  };
}

test('a record written before formats existed is labelled, not discarded', async () => {
  const ModelRegistry = await loadRegistry();
  const directory = mkdtempSync(path.join(tmpdir(), 'local-llm-gguf-'));
  const filePath = path.join(directory, 'legacy.gguf');
  writeFileSync(filePath, Buffer.alloc(32));
  const record = legacyGgufRecord(filePath);
  // Match what is on disk so nothing is treated as changed.
  record['fileModifiedAt'] = undefined;

  const store = memento([record]);
  const registry = new ModelRegistry(store, logger);
  try {
    await registry.initialize();
    const [model] = registry.list();
    assert.ok(model, 'the legacy model survived migration');
    assert.equal(model.format, 'gguf');
    assert.equal(model.runtime, 'llama-cpp');
    // Its verified capabilities are preserved: nothing about it changed.
    assert.equal(model.capabilities.toolCalling, 'supported');
    assert.ok(model.runtimeProfile);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('a Safetensors model is kept and its fingerprints backfilled', async () => {
  const ModelRegistry = await loadRegistry();
  const directory = checkpointDirectory();
  const store = memento([{
    ...legacyGgufRecord(directory),
    id: 'safetensors-model',
    filename: 'checkpoint',
    format: 'safetensors',
    runtime: 'transformers',
  }]);
  const registry = new ModelRegistry(store, logger);
  try {
    await registry.initialize();
    const [model] = registry.list();
    assert.ok(model, 'the checkpoint survived');
    assert.equal(model.format, 'safetensors');
    assert.equal(model.runtime, 'transformers');
    // Backfilled without discarding capabilities it was already verified for.
    assert.equal(model.files?.length, 2);
    assert.equal(model.capabilities.toolCalling, 'supported');
    assert.ok(model.runtimeProfile);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('a changed shard invalidates the checkpoint and re-digests it', async () => {
  const ModelRegistry = await loadRegistry();
  const directory = checkpointDirectory();
  const store = memento([{
    ...legacyGgufRecord(directory),
    id: 'safetensors-model',
    filename: 'checkpoint',
    format: 'safetensors',
    runtime: 'transformers',
    files: [
      { path: 'config.json', size: 1, modifiedAt: 1 },
      { path: 'model.safetensors', size: 1, modifiedAt: 1 },
    ],
  }]);
  const registry = new ModelRegistry(store, logger);
  try {
    await registry.initialize();
    const [model] = registry.list();
    assert.ok(model);
    assert.equal(model.capabilities.toolCalling, 'unverified');
    assert.equal(model.capabilities.fillInMiddle, 'unverified');
    assert.equal(model.runtimeProfile, undefined);
    assert.match(model.sha256, /^[0-9a-f]{64}$/);
    assert.equal(model.trainedContextLength, 8192);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('an untouched checkpoint keeps its verified capabilities across restarts', async () => {
  const ModelRegistry = await loadRegistry();
  const directory = checkpointDirectory();
  const store = memento([{
    ...legacyGgufRecord(directory),
    id: 'safetensors-model',
    filename: 'checkpoint',
    format: 'safetensors',
    runtime: 'transformers',
  }]);
  try {
    await new ModelRegistry(store, logger).initialize();
    // Second activation reads back what the first one persisted.
    const second = new ModelRegistry(store, logger);
    await second.initialize();
    const [model] = second.list();
    assert.ok(model);
    assert.equal(model.capabilities.toolCalling, 'supported');
    assert.ok(model.runtimeProfile, 'the runtime profile survived a restart');
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('touching a shard between activations invalidates it', async () => {
  const ModelRegistry = await loadRegistry();
  const directory = checkpointDirectory();
  const store = memento([{
    ...legacyGgufRecord(directory),
    id: 'safetensors-model',
    filename: 'checkpoint',
    format: 'safetensors',
    runtime: 'transformers',
  }]);
  try {
    await new ModelRegistry(store, logger).initialize();

    const shardPath = path.join(directory, 'model.safetensors');
    const future = new Date(Date.now() + 60_000);
    utimesSync(shardPath, future, future);

    const second = new ModelRegistry(store, logger);
    await second.initialize();
    const [model] = second.list();
    assert.ok(model);
    assert.equal(model.capabilities.toolCalling, 'unverified');
    assert.equal(model.runtimeProfile, undefined);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('a checkpoint directory that has vanished is dropped', async () => {
  const ModelRegistry = await loadRegistry();
  const directory = checkpointDirectory();
  rmSync(directory, { recursive: true, force: true });
  const store = memento([{
    ...legacyGgufRecord(directory),
    format: 'safetensors',
    runtime: 'transformers',
  }]);
  const registry = new ModelRegistry(store, logger);
  await registry.initialize();
  assert.deepEqual(registry.list(), []);
});

test('a checkpoint whose path became a file is dropped', async () => {
  const ModelRegistry = await loadRegistry();
  const directory = mkdtempSync(path.join(tmpdir(), 'local-llm-swap-'));
  const swapped = path.join(directory, 'checkpoint');
  writeFileSync(swapped, Buffer.alloc(8));
  const store = memento([{
    ...legacyGgufRecord(swapped),
    format: 'safetensors',
    runtime: 'transformers',
  }]);
  try {
    const registry = new ModelRegistry(store, logger);
    await registry.initialize();
    assert.deepEqual(registry.list(), []);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
