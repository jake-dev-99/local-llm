import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import * as path from 'node:path';
import test from 'node:test';
import { build } from 'esbuild';
import type { InstalledModel } from '../domain.ts';

/**
 * Covers the two paths that touch a user's own files: registering a checkpoint
 * in place, and removal. Removal deletes data, so it is verified against a real
 * directory rather than by reading the code.
 */
interface TestModelManager {
  importSafetensorsDirectory(uri: { fsPath: string; toString(): string }): Promise<InstalledModel>;
  remove(model: InstalledModel): Promise<void>;
}

type TestModelManagerConstructor = new (
  context: unknown,
  registry: unknown,
  logger: { info(message: string): void; error(message: string, error?: unknown): void },
) => TestModelManager;

const VSCODE_STUB = `
  export class EventEmitter {
    event = () => ({ dispose() {} });
    fire() {}
    dispose() {}
  }
  export class CancellationError extends Error {}
  export const ProgressLocation = { Notification: 15 };
  export const ConfigurationTarget = { Global: 1 };
  export const window = {
    withProgress: async (_options, task) => task(
      { report() {} },
      { isCancellationRequested: false, onCancellationRequested: () => ({ dispose() {} }) },
    ),
    showQuickPick: async () => undefined,
  };
  export const workspace = {
    getConfiguration: () => ({ get: (_key, fallback) => fallback }),
  };
`;

let cached: TestModelManagerConstructor | undefined;

async function loadModelManager(): Promise<TestModelManagerConstructor> {
  if (cached) {
    return cached;
  }
  const bundled = await build({
    entryPoints: ['src/models/modelManager.ts'],
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
  assert.ok(source, 'esbuild returned the bundled ModelManager');
  const url = `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`;
  const loaded = await import(url) as { ModelManager?: TestModelManagerConstructor };
  assert.ok(loaded.ModelManager, 'bundled module exports ModelManager');
  cached = loaded.ModelManager;
  return cached;
}

function fakeRegistry() {
  const models: InstalledModel[] = [];
  return {
    models,
    list: () => models,
    get: (id: string) => models.find((model) => model.id === id),
    upsert: async (model: InstalledModel) => { models.push(model); },
    remove: async (id: string) => {
      const index = models.findIndex((model) => model.id === id);
      return index < 0 ? undefined : models.splice(index, 1)[0];
    },
  };
}

const logger = { info: () => undefined, error: () => undefined };
const context = { secrets: { get: async () => undefined } };

function uriFor(fsPath: string) {
  return { fsPath, toString: () => `file://${fsPath}` };
}

function shard(header: string, payloadBytes: number): Buffer {
  const body = Buffer.from(header);
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(body.length));
  return Buffer.concat([prefix, body, Buffer.alloc(payloadBytes)]);
}

function checkpointDirectory(config: object = {}): string {
  const directory = mkdtempSync(path.join(tmpdir(), 'local-llm-mgr-'));
  writeFileSync(path.join(directory, 'config.json'), JSON.stringify({
    architectures: ['Qwen3MoeForCausalLM'],
    max_position_embeddings: 262144,
    ...config,
  }));
  writeFileSync(
    path.join(directory, 'model.safetensors'),
    shard('{"w":{"dtype":"BF16","shape":[8],"data_offsets":[0,16]}}', 128),
  );
  return directory;
}

test('registers a checkpoint in place without copying it', async () => {
  const ModelManager = await loadModelManager();
  const registry = fakeRegistry();
  const directory = checkpointDirectory({ quantization_config: { quant_method: 'fp8' } });
  try {
    const manager = new ModelManager(context, registry, logger);
    const model = await manager.importSafetensorsDirectory(uriFor(directory));

    assert.equal(model.format, 'safetensors');
    assert.equal(model.runtime, 'transformers');
    assert.equal(model.managed, false, 'the extension does not own these bytes');
    // filePath points at the user's own directory; nothing was duplicated.
    assert.equal(model.filePath, directory);
    assert.equal(model.files?.length, 2);
    assert.equal(model.trainedContextLength, 262144);
    assert.equal(model.quantization, 'fp8');
    assert.match(model.sha256, /^[0-9a-f]{64}$/);
    assert.equal(registry.models.length, 1);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('flags a checkpoint that would execute its own Python', async () => {
  const ModelManager = await loadModelManager();
  const directory = checkpointDirectory({
    auto_map: { AutoModelForCausalLM: 'modeling_custom.CustomModel' },
  });
  try {
    const manager = new ModelManager(context, fakeRegistry(), logger);
    const model = await manager.importSafetensorsDirectory(uriFor(directory));
    assert.equal(model.customCodeRequired, true);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('refuses a folder that is not a checkpoint', async () => {
  const ModelManager = await loadModelManager();
  const directory = mkdtempSync(path.join(tmpdir(), 'local-llm-notckpt-'));
  writeFileSync(path.join(directory, 'README.md'), 'nothing here');
  try {
    const manager = new ModelManager(context, fakeRegistry(), logger);
    await assert.rejects(
      manager.importSafetensorsDirectory(uriFor(directory)),
      /not a Safetensors checkpoint/,
    );
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('removing a checkpoint registered in place never deletes the user\'s files', async () => {
  const ModelManager = await loadModelManager();
  const registry = fakeRegistry();
  const directory = checkpointDirectory();
  try {
    const manager = new ModelManager(context, registry, logger);
    const model = await manager.importSafetensorsDirectory(uriFor(directory));

    await manager.remove(model);

    assert.equal(registry.models.length, 0, 'it was unregistered');
    assert.ok(existsSync(directory), 'the directory survived');
    assert.ok(
      existsSync(path.join(directory, 'model.safetensors')),
      'the weights survived',
    );
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('removing a model the extension copied does delete it', async () => {
  const ModelManager = await loadModelManager();
  const registry = fakeRegistry();
  const directory = mkdtempSync(path.join(tmpdir(), 'local-llm-owned-'));
  const filePath = path.join(directory, 'owned.gguf');
  writeFileSync(filePath, Buffer.alloc(16));
  try {
    const manager = new ModelManager(context, registry, logger);
    const owned = {
      id: 'owned', name: 'Owned', filePath, fileSize: 16, sha256: 'x',
      source: 'import', filename: 'owned.gguf',
      installedAt: '2026-01-01T00:00:00.000Z',
      format: 'gguf', runtime: 'llama-cpp',
      capabilities: { toolCalling: 'unverified', fillInMiddle: 'unverified' },
    } as unknown as InstalledModel;
    registry.models.push(owned);

    await manager.remove(owned);

    assert.equal(existsSync(filePath), false, 'the copied file was deleted');
    assert.equal(registry.models.length, 0);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
