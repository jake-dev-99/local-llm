<<<<<<< Updated upstream
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import {
  existsSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs';
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
  importLocal(uri: { fsPath: string; toString(): string }, mode?: 'copy' | 'link'): Promise<InstalledModel>;
  downloadFromHuggingFace(repository: string): Promise<InstalledModel | undefined>;
  importSafetensorsDirectory(uri: { fsPath: string; toString(): string }): Promise<InstalledModel>;
  stageSafetensorsFile(
    uri: { fsPath: string; toString(): string },
    options?: { repository?: string; configFile?: string; configJson?: string },
    mode?: 'copy' | 'link',
  ): Promise<{ directory: string; checkpoint: { architecture?: string } }>;
  registerStagedSafetensorsDirectory(directory: string): Promise<InstalledModel>;
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

const logger = { info: () => undefined, warn: () => undefined, error: () => undefined };
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

function responseAt(url: string, body: string | Buffer, init?: ResponseInit): Response {
  const response = new Response(body as unknown as BodyInit, init);
  Object.defineProperty(response, 'url', { value: url });
  return response;
}

test('downloads every file in the Qwen Safetensors repository and reuses them', async () => {
  const ModelManager = await loadModelManager();
  const storage = mkdtempSync(path.join(tmpdir(), 'local-llm-hf-safetensors-'));
  const movedStorage = mkdtempSync(path.join(tmpdir(), 'local-llm-hf-moved-'));
  const repository = 'unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit';
  const revision = '6700c3e5bdeb050a379c8d2a4133f43f3647f20f';
  const shardNames = Array.from(
    { length: 5 },
    (_, index) => `model-${String(index + 1).padStart(5, '0')}-of-00005.safetensors`,
  );
  const bodies = new Map<string, Buffer>([
    ['chat_template.jinja', Buffer.from('{{ messages }}')],
    ['config.json', Buffer.from(JSON.stringify({
      architectures: ['Qwen3_5MoeForConditionalGeneration'],
      model_type: 'qwen3_5_moe',
    }))],
    ...shardNames.map((filename, index) => [
      filename,
      shard(`{"w${index}":{"dtype":"BF16","shape":[8],"data_offsets":[0,16]}}`, 16),
    ] as const),
    ['model.safetensors.index.json', Buffer.from('{"weight_map":{}}')],
    ['processor_config.json', Buffer.from('{}')],
    ['tokenizer.json', Buffer.from('{}')],
    ['tokenizer_config.json', Buffer.from('{}')],
  ]);
  let fileDownloads = 0;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input) => {
    const url = String(input);
    if (url.includes('/api/models/')) {
      const siblings = [
        { rfilename: 'README.md', size: 10 },
        ...[...bodies].map(([rfilename, body]) => ({
          rfilename,
          lfs: {
            size: body.length,
            sha256: createHash('sha256').update(body).digest('hex'),
          },
        })),
      ];
      return responseAt(url, JSON.stringify({ id: repository, sha: revision, siblings }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      });
    }
    const marker = `/resolve/${revision}/`;
    const filename = decodeURIComponent(url.slice(url.indexOf(marker) + marker.length));
    const body = bodies.get(filename);
    if (!body) {
      return responseAt(url, 'missing', { status: 404 });
    }
    fileDownloads += 1;
    return responseAt(url, body, {
      status: 200,
      headers: {
        'content-length': String(body.length),
        etag: `"${filename}"`,
      },
    });
  };

  try {
    const managedContext = { ...context, globalStorageUri: uriFor(storage) };
    const firstRegistry = fakeRegistry();
    const first = await new ModelManager(managedContext, firstRegistry, logger)
      .downloadFromHuggingFace(repository);
    assert.ok(first);
    assert.equal(first.format, 'safetensors');
    assert.equal(first.runtime, 'transformers');
    assert.equal(first.managed, true);
    assert.equal(first.name, 'Qwen3.6 35B A3B UD MLX 4bit');
    assert.equal(first.repository, repository);
    assert.equal(first.revision, revision);
    assert.deepEqual(readdirSync(first.filePath).sort(), [...bodies.keys()].sort());
    assert.equal(fileDownloads, bodies.size);

    const secondRegistry = fakeRegistry();
    const second = await new ModelManager(managedContext, secondRegistry, logger)
      .downloadFromHuggingFace(repository);
    assert.ok(second);
    assert.equal(second.id, first.id);
    assert.equal(fileDownloads, bodies.size, 'completed immutable files were reused');

    const afterDirectoryChange = await new ModelManager(
      { ...context, globalStorageUri: uriFor(movedStorage) },
      firstRegistry,
      logger,
    ).downloadFromHuggingFace(repository);
    assert.ok(afterDirectoryChange);
    assert.equal(afterDirectoryChange.filePath, first.filePath);
    assert.equal(fileDownloads, bodies.size, 'the registered managed directory was retained');

    const damagedName = shardNames[0];
    assert.ok(damagedName);
    const expectedShard = bodies.get(damagedName);
    assert.ok(expectedShard);
    writeFileSync(path.join(first.filePath, damagedName), Buffer.alloc(expectedShard.length));
    const repaired = await new ModelManager(managedContext, firstRegistry, logger)
      .downloadFromHuggingFace(repository);
    assert.ok(repaired);
    assert.equal(fileDownloads, bodies.size + 1, 'the damaged registered shard was fetched again');
    assert.deepEqual(readFileSync(path.join(first.filePath, damagedName)), expectedShard);
  } finally {
    globalThis.fetch = originalFetch;
    rmSync(storage, { recursive: true, force: true });
    rmSync(movedStorage, { recursive: true, force: true });
  }
});

test('does not register a downloaded checkpoint that fails static validation', async () => {
  const ModelManager = await loadModelManager();
  const storage = mkdtempSync(path.join(tmpdir(), 'local-llm-hf-invalid-'));
  const repository = 'example/invalid-safetensors';
  const revision = '1111111111111111111111111111111111111111';
  const weights = shard('{"w":{"dtype":"BF16","shape":[8],"data_offsets":[0,16]}}', 16);
  const bodies = new Map<string, Buffer>([
    ['config.json', Buffer.from('{invalid')],
    ['model.safetensors', weights],
    ['tokenizer.json', Buffer.from('{}')],
  ]);
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input) => {
    const url = String(input);
    if (url.includes('/api/models/')) {
      return responseAt(url, JSON.stringify({
        id: repository,
        sha: revision,
        siblings: [...bodies].map(([rfilename, body]) => ({
          rfilename,
          lfs: {
            size: body.length,
            sha256: createHash('sha256').update(body).digest('hex'),
          },
        })),
      }), { status: 200, headers: { 'content-type': 'application/json' } });
    }
    const marker = `/resolve/${revision}/`;
    const filename = decodeURIComponent(url.slice(url.indexOf(marker) + marker.length));
    const body = bodies.get(filename);
    return body
      ? responseAt(url, body, {
        status: 200,
        headers: { 'content-length': String(body.length), etag: `"${filename}"` },
      })
      : responseAt(url, 'missing', { status: 404 });
  };

  try {
    const registry = fakeRegistry();
    const manager = new ModelManager(
      { ...context, globalStorageUri: uriFor(storage) },
      registry,
      logger,
    );
    await assert.rejects(
      manager.downloadFromHuggingFace(repository),
      /validation failed.*config\.json is missing or unreadable/i,
    );
    assert.equal(registry.models.length, 0);
  } finally {
    globalThis.fetch = originalFetch;
    rmSync(storage, { recursive: true, force: true });
  }
});

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

test('stages a lone weights file with sibling sidecars, then registers it owned', async () => {
  const ModelManager = await loadModelManager();
  const registry = fakeRegistry();
  const source = mkdtempSync(path.join(tmpdir(), 'local-llm-lone-'));
  const storage = mkdtempSync(path.join(tmpdir(), 'local-llm-storage-'));
  writeFileSync(
    path.join(source, 'gemma4.safetensors'),
    shard('{"w":{"dtype":"BF16","shape":[8],"data_offsets":[0,16]}}', 128),
  );
  writeFileSync(path.join(source, 'config.json'), JSON.stringify({
    architectures: ['GemmaForCausalLM'],
    max_position_embeddings: 32768,
  }));
  writeFileSync(path.join(source, 'tokenizer.json'), '{}');
  try {
    const manager = new ModelManager(
      { ...context, globalStorageUri: uriFor(storage) },
      registry,
      logger,
    );
    const staged = await manager.stageSafetensorsFile(uriFor(path.join(source, 'gemma4.safetensors')));
    assert.ok(staged.directory.startsWith(path.join(storage, 'models')), 'staged into managed storage');
    assert.equal(staged.checkpoint.architecture, 'GemmaForCausalLM');
    assert.ok(existsSync(path.join(staged.directory, 'config.json')), 'sibling config came along');

    const model = await manager.registerStagedSafetensorsDirectory(staged.directory);
    assert.equal(model.format, 'safetensors');
    assert.equal(model.managed, true, 'staged bytes are extension-owned');
    assert.equal(model.filePath, staged.directory);

    await manager.remove(model);
    assert.ok(!existsSync(staged.directory), 'owned bytes are deleted');
    assert.ok(existsSync(path.join(source, 'gemma4.safetensors')), 'the original is untouched');
  } finally {
    rmSync(source, { recursive: true, force: true });
    rmSync(storage, { recursive: true, force: true });
  }
});

test('a sibling chat_template.jinja stages with the weights file', async () => {
  // Checkpoints increasingly keep the template standalone rather than inside
  // tokenizer_config.json; without it the loaded tokenizer reports no chat
  // template and validation refuses the model.
  const ModelManager = await loadModelManager();
  const registry = fakeRegistry();
  const source = mkdtempSync(path.join(tmpdir(), 'local-llm-lone-'));
  const storage = mkdtempSync(path.join(tmpdir(), 'local-llm-storage-'));
  writeFileSync(
    path.join(source, 'gemma4.safetensors'),
    shard('{"w":{"dtype":"BF16","shape":[8],"data_offsets":[0,16]}}', 128),
  );
  writeFileSync(path.join(source, 'config.json'), JSON.stringify({
    architectures: ['GemmaForCausalLM'],
    max_position_embeddings: 32768,
  }));
  writeFileSync(path.join(source, 'chat_template.jinja'), '{{ messages }}');
  try {
    const manager = new ModelManager(
      { ...context, globalStorageUri: uriFor(storage) },
      registry,
      logger,
    );
    const staged = await manager.stageSafetensorsFile(uriFor(path.join(source, 'gemma4.safetensors')));
    assert.ok(existsSync(path.join(staged.directory, 'chat_template.jinja')), 'standalone template came along');
  } finally {
    rmSync(source, { recursive: true, force: true });
    rmSync(storage, { recursive: true, force: true });
  }
});

test('a lone file with no config and no repository fails without littering', async () => {
  const ModelManager = await loadModelManager();
  const source = mkdtempSync(path.join(tmpdir(), 'local-llm-bare-'));
  const storage = mkdtempSync(path.join(tmpdir(), 'local-llm-storage-'));
  writeFileSync(
    path.join(source, 'bare.safetensors'),
    shard('{"w":{"dtype":"BF16","shape":[8],"data_offsets":[0,16]}}', 128),
  );
  try {
    const manager = new ModelManager(
      { ...context, globalStorageUri: uriFor(storage) },
      fakeRegistry(),
      logger,
    );
    await assert.rejects(
      manager.stageSafetensorsFile(uriFor(path.join(source, 'bare.safetensors'))),
      /config\.json is missing/,
    );
    assert.deepEqual(readdirSync(path.join(storage, 'models')), [], 'no orphaned stage directory');
  } finally {
    rmSync(source, { recursive: true, force: true });
    rmSync(storage, { recursive: true, force: true });
  }
});

test('a lone file takes an explicit config file or pasted JSON', async () => {
  const ModelManager = await loadModelManager();
  const config = JSON.stringify({ architectures: ['GemmaForCausalLM'] });
  for (const variant of ['file', 'pasted'] as const) {
    const source = mkdtempSync(path.join(tmpdir(), 'local-llm-lone-'));
    const storage = mkdtempSync(path.join(tmpdir(), 'local-llm-storage-'));
    writeFileSync(
      path.join(source, 'bare.safetensors'),
      shard('{"w":{"dtype":"BF16","shape":[8],"data_offsets":[0,16]}}', 128),
    );
    try {
      const manager = new ModelManager(
        { ...context, globalStorageUri: uriFor(storage) },
        fakeRegistry(),
        logger,
      );
      const options = variant === 'file'
        ? (() => {
          writeFileSync(path.join(source, 'mine.json'), config);
          return { configFile: path.join(source, 'mine.json') };
        })()
        : { configJson: config };
      const staged = await manager.stageSafetensorsFile(
        uriFor(path.join(source, 'bare.safetensors')),
        options,
      );
      assert.equal(staged.checkpoint.architecture, 'GemmaForCausalLM');
      assert.ok(existsSync(path.join(staged.directory, 'config.json')));
    } finally {
      rmSync(source, { recursive: true, force: true });
      rmSync(storage, { recursive: true, force: true });
    }
  }
});

test('pasted garbage is refused without littering', async () => {
  const ModelManager = await loadModelManager();
  const source = mkdtempSync(path.join(tmpdir(), 'local-llm-lone-'));
  const storage = mkdtempSync(path.join(tmpdir(), 'local-llm-storage-'));
  writeFileSync(
    path.join(source, 'bare.safetensors'),
    shard('{"w":{"dtype":"BF16","shape":[8],"data_offsets":[0,16]}}', 128),
  );
  try {
    const manager = new ModelManager(
      { ...context, globalStorageUri: uriFor(storage) },
      fakeRegistry(),
      logger,
    );
    await assert.rejects(
      manager.stageSafetensorsFile(
        uriFor(path.join(source, 'bare.safetensors')),
        { configJson: 'not json at all' },
      ),
      /not valid JSON/,
    );
    assert.deepEqual(readdirSync(path.join(storage, 'models')), [], 'no orphaned stage directory');
  } finally {
    rmSync(source, { recursive: true, force: true });
    rmSync(storage, { recursive: true, force: true });
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

test('a successful Hugging Face response that is not JSON fails instead of reading as an empty repository', async () => {
  const ModelManager = await loadModelManager();
  const storage = mkdtempSync(path.join(tmpdir(), 'local-llm-hf-garbage-'));
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response('<html>Service temporarily unavailable</html>', { status: 200 });
  try {
    const managedContext = { ...context, globalStorageUri: uriFor(storage) };
    await assert.rejects(
      new ModelManager(managedContext, fakeRegistry(), logger).downloadFromHuggingFace('owner/repo'),
      /not JSON: <html>Service temporarily unavailable/,
    );
  } finally {
    globalThis.fetch = originalFetch;
    rmSync(storage, { recursive: true, force: true });
  }
});

/**
 * Runs `downloadFromHuggingFace` against a repository listing `filenames`, with
 * the stub's picker dismissed. Returns the settled outcome and every file URL
 * fetched.
 */
async function downloadWithDismissedPicker(filenames: string[]): Promise<{
  outcome: { value: InstalledModel | undefined } | { error: unknown };
  fileRequests: string[];
}> {
  const ModelManager = await loadModelManager();
  const storage = mkdtempSync(path.join(tmpdir(), 'local-llm-hf-dismissed-'));
  const repository = 'owner/repo';
  const fileRequests: string[] = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input) => {
    const url = String(input);
    if (url.includes('/api/models/')) {
      return responseAt(url, JSON.stringify({
        id: repository,
        sha: '2222222222222222222222222222222222222222',
        siblings: filenames.map((rfilename) => ({ rfilename, lfs: { size: 1024 } })),
      }), { status: 200, headers: { 'content-type': 'application/json' } });
    }
    fileRequests.push(url);
    return responseAt(url, 'unexpected', { status: 404 });
  };
  try {
    const manager = new ModelManager(
      { ...context, globalStorageUri: uriFor(storage) },
      fakeRegistry(),
      logger,
    );
    const outcome = await manager.downloadFromHuggingFace(repository).then(
      (value) => ({ value }),
      (error: unknown) => ({ error }),
    );
    return { outcome, fileRequests };
  } finally {
    globalThis.fetch = originalFetch;
    rmSync(storage, { recursive: true, force: true });
  }
}

test('dismissing the GGUF picker cancels quietly instead of reporting the repository unsupported', async () => {
  const { outcome, fileRequests } = await downloadWithDismissedPicker([
    'Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf',
    'Devstral-Small-2-24B-Instruct-2512-Q8_0.gguf',
    'README.md',
    'config.json',
  ]);
  assert.deepEqual(outcome, { value: undefined });
  assert.deepEqual(fileRequests, [], 'nothing was downloaded');
});

test('dismissing the format picker in a repository with GGUF and Safetensors cancels quietly', async () => {
  const { outcome, fileRequests } = await downloadWithDismissedPicker([
    'config.json',
    'model.Q4_K_M.gguf',
    'model.safetensors',
    'tokenizer.json',
  ]);
  assert.deepEqual(outcome, { value: undefined });
  assert.deepEqual(fileRequests, [], 'nothing was downloaded');
});

test('a repository of only sharded GGUF parts is still reported as unsupported', async () => {
  const { outcome } = await downloadWithDismissedPicker([
    'model-Q4_K_M-00001-of-00002.gguf',
    'model-Q4_K_M-00002-of-00002.gguf',
  ]);
  assert.ok('error' in outcome, 'the download was rejected');
  assert.match(
    String(outcome.error),
    /contains GGUF files, but none are supported single-file downloads/,
  );
});

/** The smallest file assertGguf accepts: magic, version 3, no tensors, no metadata. */
function ggufBytes(): Buffer {
  const bytes = Buffer.alloc(24);
  bytes.write('GGUF', 0, 'ascii');
  bytes.writeUInt32LE(3, 4);
  return bytes;
}

test('a linked GGUF is registered where it is, never copied, and never deleted', async () => {
  const ModelManager = await loadModelManager();
  const registry = fakeRegistry();
  const source = mkdtempSync(path.join(tmpdir(), 'local-llm-link-src-'));
  const storage = mkdtempSync(path.join(tmpdir(), 'local-llm-link-storage-'));
  const original = path.join(source, 'Qwen3-4B.Q8_0.gguf');
  writeFileSync(original, ggufBytes());
  try {
    const manager = new ModelManager({ ...context, globalStorageUri: uriFor(storage) }, registry, logger);
    const model = await manager.importLocal(uriFor(original), 'link');

    assert.equal(model.filePath, original);
    assert.equal(model.managed, false, 'the extension does not own a linked file');
    assert.equal(model.sha256, createHash('sha256').update(ggufBytes()).digest('hex'));
    assert.deepEqual(readdirSync(storage), [], 'linking writes nothing to the models folder');

    await manager.remove(model);
    assert.ok(existsSync(original), 'removing a linked model leaves the user\'s file alone');
  } finally {
    rmSync(source, { recursive: true, force: true });
    rmSync(storage, { recursive: true, force: true });
  }
});

test('copying a model that was linked does not delete the user\'s original', async () => {
  const ModelManager = await loadModelManager();
  const registry = fakeRegistry();
  const source = mkdtempSync(path.join(tmpdir(), 'local-llm-relink-src-'));
  const storage = mkdtempSync(path.join(tmpdir(), 'local-llm-relink-storage-'));
  const original = path.join(source, 'model.gguf');
  writeFileSync(original, ggufBytes());
  try {
    const manager = new ModelManager({ ...context, globalStorageUri: uriFor(storage) }, registry, logger);
    await manager.importLocal(uriFor(original), 'link');
    const copied = await manager.importLocal(uriFor(original), 'copy');

    assert.notEqual(copied.filePath, original);
    assert.ok(existsSync(original), 'replacing a linked registration must not delete the file it linked');
  } finally {
    rmSync(source, { recursive: true, force: true });
    rmSync(storage, { recursive: true, force: true });
  }
});

test('a linked single Safetensors file is staged as a hard link, not a second copy', async () => {
  const ModelManager = await loadModelManager();
  const source = mkdtempSync(path.join(tmpdir(), 'local-llm-hardlink-src-'));
  const storage = mkdtempSync(path.join(tmpdir(), 'local-llm-hardlink-storage-'));
  const weights = path.join(source, 'gemma4.safetensors');
  writeFileSync(weights, shard('{"w":{"dtype":"BF16","shape":[8],"data_offsets":[0,16]}}', 128));
  writeFileSync(path.join(source, 'config.json'), JSON.stringify({ architectures: ['GemmaForCausalLM'] }));
  try {
    const manager = new ModelManager({ ...context, globalStorageUri: uriFor(storage) }, fakeRegistry(), logger);
    const staged = await manager.stageSafetensorsFile(uriFor(weights), {}, 'link');

    const stagedWeights = statSync(path.join(staged.directory, 'gemma4.safetensors'));
    assert.equal(stagedWeights.ino, statSync(weights).ino, 'the staged weights are the same file on disk');
    assert.equal(statSync(weights).nlink, 2);
  } finally {
    rmSync(source, { recursive: true, force: true });
    rmSync(storage, { recursive: true, force: true });
  }
});
=======
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import {
  existsSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs';
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
  importLocal(uri: { fsPath: string; toString(): string }, mode?: 'copy' | 'link'): Promise<InstalledModel>;
  downloadFromHuggingFace(repository: string): Promise<InstalledModel | undefined>;
  importSafetensorsDirectory(uri: { fsPath: string; toString(): string }): Promise<InstalledModel>;
  stageSafetensorsFile(
    uri: { fsPath: string; toString(): string },
    options?: { repository?: string; configFile?: string; configJson?: string },
    mode?: 'copy' | 'link',
  ): Promise<{ directory: string; checkpoint: { architecture?: string } }>;
  registerStagedSafetensorsDirectory(directory: string): Promise<InstalledModel>;
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

const logger = { info: () => undefined, warn: () => undefined, error: () => undefined };
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

function responseAt(url: string, body: string | Buffer, init?: ResponseInit): Response {
  const response = new Response(body as unknown as BodyInit, init);
  Object.defineProperty(response, 'url', { value: url });
  return response;
}

test('downloads every file in the Qwen Safetensors repository and reuses them', async () => {
  const ModelManager = await loadModelManager();
  const storage = mkdtempSync(path.join(tmpdir(), 'local-llm-hf-safetensors-'));
  const movedStorage = mkdtempSync(path.join(tmpdir(), 'local-llm-hf-moved-'));
  const repository = 'unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit';
  const revision = '6700c3e5bdeb050a379c8d2a4133f43f3647f20f';
  const shardNames = Array.from(
    { length: 5 },
    (_, index) => `model-${String(index + 1).padStart(5, '0')}-of-00005.safetensors`,
  );
  const bodies = new Map<string, Buffer>([
    ['chat_template.jinja', Buffer.from('{{ messages }}')],
    ['config.json', Buffer.from(JSON.stringify({
      architectures: ['Qwen3_5MoeForConditionalGeneration'],
      model_type: 'qwen3_5_moe',
    }))],
    ...shardNames.map((filename, index) => [
      filename,
      shard(`{"w${index}":{"dtype":"BF16","shape":[8],"data_offsets":[0,16]}}`, 16),
    ] as const),
    ['model.safetensors.index.json', Buffer.from('{"weight_map":{}}')],
    ['processor_config.json', Buffer.from('{}')],
    ['tokenizer.json', Buffer.from('{}')],
    ['tokenizer_config.json', Buffer.from('{}')],
  ]);
  let fileDownloads = 0;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input) => {
    const url = String(input);
    if (url.includes('/api/models/')) {
      const siblings = [
        { rfilename: 'README.md', size: 10 },
        ...[...bodies].map(([rfilename, body]) => ({
          rfilename,
          lfs: {
            size: body.length,
            sha256: createHash('sha256').update(body).digest('hex'),
          },
        })),
      ];
      return responseAt(url, JSON.stringify({ id: repository, sha: revision, siblings }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      });
    }
    const marker = `/resolve/${revision}/`;
    const filename = decodeURIComponent(url.slice(url.indexOf(marker) + marker.length));
    const body = bodies.get(filename);
    if (!body) {
      return responseAt(url, 'missing', { status: 404 });
    }
    fileDownloads += 1;
    return responseAt(url, body, {
      status: 200,
      headers: {
        'content-length': String(body.length),
        etag: `"${filename}"`,
      },
    });
  };

  try {
    const managedContext = { ...context, globalStorageUri: uriFor(storage) };
    const firstRegistry = fakeRegistry();
    const first = await new ModelManager(managedContext, firstRegistry, logger)
      .downloadFromHuggingFace(repository);
    assert.ok(first);
    assert.equal(first.format, 'safetensors');
    assert.equal(first.runtime, 'transformers');
    assert.equal(first.managed, true);
    assert.equal(first.name, 'Qwen3.6 35B A3B UD MLX 4bit');
    assert.equal(first.repository, repository);
    assert.equal(first.revision, revision);
    assert.deepEqual(readdirSync(first.filePath).sort(), [...bodies.keys()].sort());
    assert.equal(fileDownloads, bodies.size);

    const secondRegistry = fakeRegistry();
    const second = await new ModelManager(managedContext, secondRegistry, logger)
      .downloadFromHuggingFace(repository);
    assert.ok(second);
    assert.equal(second.id, first.id);
    assert.equal(fileDownloads, bodies.size, 'completed immutable files were reused');

    const afterDirectoryChange = await new ModelManager(
      { ...context, globalStorageUri: uriFor(movedStorage) },
      firstRegistry,
      logger,
    ).downloadFromHuggingFace(repository);
    assert.ok(afterDirectoryChange);
    assert.equal(afterDirectoryChange.filePath, first.filePath);
    assert.equal(fileDownloads, bodies.size, 'the registered managed directory was retained');

    const damagedName = shardNames[0];
    assert.ok(damagedName);
    const expectedShard = bodies.get(damagedName);
    assert.ok(expectedShard);
    writeFileSync(path.join(first.filePath, damagedName), Buffer.alloc(expectedShard.length));
    const repaired = await new ModelManager(managedContext, firstRegistry, logger)
      .downloadFromHuggingFace(repository);
    assert.ok(repaired);
    assert.equal(fileDownloads, bodies.size + 1, 'the damaged registered shard was fetched again');
    assert.deepEqual(readFileSync(path.join(first.filePath, damagedName)), expectedShard);
  } finally {
    globalThis.fetch = originalFetch;
    rmSync(storage, { recursive: true, force: true });
    rmSync(movedStorage, { recursive: true, force: true });
  }
});

test('does not register a downloaded checkpoint that fails static validation', async () => {
  const ModelManager = await loadModelManager();
  const storage = mkdtempSync(path.join(tmpdir(), 'local-llm-hf-invalid-'));
  const repository = 'example/invalid-safetensors';
  const revision = '1111111111111111111111111111111111111111';
  const weights = shard('{"w":{"dtype":"BF16","shape":[8],"data_offsets":[0,16]}}', 16);
  const bodies = new Map<string, Buffer>([
    ['config.json', Buffer.from('{invalid')],
    ['model.safetensors', weights],
    ['tokenizer.json', Buffer.from('{}')],
  ]);
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input) => {
    const url = String(input);
    if (url.includes('/api/models/')) {
      return responseAt(url, JSON.stringify({
        id: repository,
        sha: revision,
        siblings: [...bodies].map(([rfilename, body]) => ({
          rfilename,
          lfs: {
            size: body.length,
            sha256: createHash('sha256').update(body).digest('hex'),
          },
        })),
      }), { status: 200, headers: { 'content-type': 'application/json' } });
    }
    const marker = `/resolve/${revision}/`;
    const filename = decodeURIComponent(url.slice(url.indexOf(marker) + marker.length));
    const body = bodies.get(filename);
    return body
      ? responseAt(url, body, {
        status: 200,
        headers: { 'content-length': String(body.length), etag: `"${filename}"` },
      })
      : responseAt(url, 'missing', { status: 404 });
  };

  try {
    const registry = fakeRegistry();
    const manager = new ModelManager(
      { ...context, globalStorageUri: uriFor(storage) },
      registry,
      logger,
    );
    await assert.rejects(
      manager.downloadFromHuggingFace(repository),
      /validation failed.*config\.json is missing or unreadable/i,
    );
    assert.equal(registry.models.length, 0);
  } finally {
    globalThis.fetch = originalFetch;
    rmSync(storage, { recursive: true, force: true });
  }
});

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

test('stages a lone weights file with sibling sidecars, then registers it owned', async () => {
  const ModelManager = await loadModelManager();
  const registry = fakeRegistry();
  const source = mkdtempSync(path.join(tmpdir(), 'local-llm-lone-'));
  const storage = mkdtempSync(path.join(tmpdir(), 'local-llm-storage-'));
  writeFileSync(
    path.join(source, 'gemma4.safetensors'),
    shard('{"w":{"dtype":"BF16","shape":[8],"data_offsets":[0,16]}}', 128),
  );
  writeFileSync(path.join(source, 'config.json'), JSON.stringify({
    architectures: ['GemmaForCausalLM'],
    max_position_embeddings: 32768,
  }));
  writeFileSync(path.join(source, 'tokenizer.json'), '{}');
  try {
    const manager = new ModelManager(
      { ...context, globalStorageUri: uriFor(storage) },
      registry,
      logger,
    );
    const staged = await manager.stageSafetensorsFile(uriFor(path.join(source, 'gemma4.safetensors')));
    assert.ok(staged.directory.startsWith(path.join(storage, 'models')), 'staged into managed storage');
    assert.equal(staged.checkpoint.architecture, 'GemmaForCausalLM');
    assert.ok(existsSync(path.join(staged.directory, 'config.json')), 'sibling config came along');

    const model = await manager.registerStagedSafetensorsDirectory(staged.directory);
    assert.equal(model.format, 'safetensors');
    assert.equal(model.managed, true, 'staged bytes are extension-owned');
    assert.equal(model.filePath, staged.directory);

    await manager.remove(model);
    assert.ok(!existsSync(staged.directory), 'owned bytes are deleted');
    assert.ok(existsSync(path.join(source, 'gemma4.safetensors')), 'the original is untouched');
  } finally {
    rmSync(source, { recursive: true, force: true });
    rmSync(storage, { recursive: true, force: true });
  }
});

test('a sibling chat_template.jinja stages with the weights file', async () => {
  // Checkpoints increasingly keep the template standalone rather than inside
  // tokenizer_config.json; without it the loaded tokenizer reports no chat
  // template and validation refuses the model.
  const ModelManager = await loadModelManager();
  const registry = fakeRegistry();
  const source = mkdtempSync(path.join(tmpdir(), 'local-llm-lone-'));
  const storage = mkdtempSync(path.join(tmpdir(), 'local-llm-storage-'));
  writeFileSync(
    path.join(source, 'gemma4.safetensors'),
    shard('{"w":{"dtype":"BF16","shape":[8],"data_offsets":[0,16]}}', 128),
  );
  writeFileSync(path.join(source, 'config.json'), JSON.stringify({
    architectures: ['GemmaForCausalLM'],
    max_position_embeddings: 32768,
  }));
  writeFileSync(path.join(source, 'chat_template.jinja'), '{{ messages }}');
  try {
    const manager = new ModelManager(
      { ...context, globalStorageUri: uriFor(storage) },
      registry,
      logger,
    );
    const staged = await manager.stageSafetensorsFile(uriFor(path.join(source, 'gemma4.safetensors')));
    assert.ok(existsSync(path.join(staged.directory, 'chat_template.jinja')), 'standalone template came along');
  } finally {
    rmSync(source, { recursive: true, force: true });
    rmSync(storage, { recursive: true, force: true });
  }
});

test('a lone file with no config and no repository fails without littering', async () => {
  const ModelManager = await loadModelManager();
  const source = mkdtempSync(path.join(tmpdir(), 'local-llm-bare-'));
  const storage = mkdtempSync(path.join(tmpdir(), 'local-llm-storage-'));
  writeFileSync(
    path.join(source, 'bare.safetensors'),
    shard('{"w":{"dtype":"BF16","shape":[8],"data_offsets":[0,16]}}', 128),
  );
  try {
    const manager = new ModelManager(
      { ...context, globalStorageUri: uriFor(storage) },
      fakeRegistry(),
      logger,
    );
    await assert.rejects(
      manager.stageSafetensorsFile(uriFor(path.join(source, 'bare.safetensors'))),
      /config\.json is missing/,
    );
    assert.deepEqual(readdirSync(path.join(storage, 'models')), [], 'no orphaned stage directory');
  } finally {
    rmSync(source, { recursive: true, force: true });
    rmSync(storage, { recursive: true, force: true });
  }
});

test('a lone file takes an explicit config file or pasted JSON', async () => {
  const ModelManager = await loadModelManager();
  const config = JSON.stringify({ architectures: ['GemmaForCausalLM'] });
  for (const variant of ['file', 'pasted'] as const) {
    const source = mkdtempSync(path.join(tmpdir(), 'local-llm-lone-'));
    const storage = mkdtempSync(path.join(tmpdir(), 'local-llm-storage-'));
    writeFileSync(
      path.join(source, 'bare.safetensors'),
      shard('{"w":{"dtype":"BF16","shape":[8],"data_offsets":[0,16]}}', 128),
    );
    try {
      const manager = new ModelManager(
        { ...context, globalStorageUri: uriFor(storage) },
        fakeRegistry(),
        logger,
      );
      const options = variant === 'file'
        ? (() => {
          writeFileSync(path.join(source, 'mine.json'), config);
          return { configFile: path.join(source, 'mine.json') };
        })()
        : { configJson: config };
      const staged = await manager.stageSafetensorsFile(
        uriFor(path.join(source, 'bare.safetensors')),
        options,
      );
      assert.equal(staged.checkpoint.architecture, 'GemmaForCausalLM');
      assert.ok(existsSync(path.join(staged.directory, 'config.json')));
    } finally {
      rmSync(source, { recursive: true, force: true });
      rmSync(storage, { recursive: true, force: true });
    }
  }
});

test('pasted garbage is refused without littering', async () => {
  const ModelManager = await loadModelManager();
  const source = mkdtempSync(path.join(tmpdir(), 'local-llm-lone-'));
  const storage = mkdtempSync(path.join(tmpdir(), 'local-llm-storage-'));
  writeFileSync(
    path.join(source, 'bare.safetensors'),
    shard('{"w":{"dtype":"BF16","shape":[8],"data_offsets":[0,16]}}', 128),
  );
  try {
    const manager = new ModelManager(
      { ...context, globalStorageUri: uriFor(storage) },
      fakeRegistry(),
      logger,
    );
    await assert.rejects(
      manager.stageSafetensorsFile(
        uriFor(path.join(source, 'bare.safetensors')),
        { configJson: 'not json at all' },
      ),
      /not valid JSON/,
    );
    assert.deepEqual(readdirSync(path.join(storage, 'models')), [], 'no orphaned stage directory');
  } finally {
    rmSync(source, { recursive: true, force: true });
    rmSync(storage, { recursive: true, force: true });
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

test('a successful Hugging Face response that is not JSON fails instead of reading as an empty repository', async () => {
  const ModelManager = await loadModelManager();
  const storage = mkdtempSync(path.join(tmpdir(), 'local-llm-hf-garbage-'));
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response('<html>Service temporarily unavailable</html>', { status: 200 });
  try {
    const managedContext = { ...context, globalStorageUri: uriFor(storage) };
    await assert.rejects(
      new ModelManager(managedContext, fakeRegistry(), logger).downloadFromHuggingFace('owner/repo'),
      /not JSON: <html>Service temporarily unavailable/,
    );
  } finally {
    globalThis.fetch = originalFetch;
    rmSync(storage, { recursive: true, force: true });
  }
});

/** The smallest file assertGguf accepts: magic, version 3, no tensors, no metadata. */
function ggufBytes(): Buffer {
  const bytes = Buffer.alloc(24);
  bytes.write('GGUF', 0, 'ascii');
  bytes.writeUInt32LE(3, 4);
  return bytes;
}

test('a linked GGUF is registered where it is, never copied, and never deleted', async () => {
  const ModelManager = await loadModelManager();
  const registry = fakeRegistry();
  const source = mkdtempSync(path.join(tmpdir(), 'local-llm-link-src-'));
  const storage = mkdtempSync(path.join(tmpdir(), 'local-llm-link-storage-'));
  const original = path.join(source, 'Qwen3-4B.Q8_0.gguf');
  writeFileSync(original, ggufBytes());
  try {
    const manager = new ModelManager({ ...context, globalStorageUri: uriFor(storage) }, registry, logger);
    const model = await manager.importLocal(uriFor(original), 'link');

    assert.equal(model.filePath, original);
    assert.equal(model.managed, false, 'the extension does not own a linked file');
    assert.equal(model.sha256, createHash('sha256').update(ggufBytes()).digest('hex'));
    assert.deepEqual(readdirSync(storage), [], 'linking writes nothing to the models folder');

    await manager.remove(model);
    assert.ok(existsSync(original), 'removing a linked model leaves the user\'s file alone');
  } finally {
    rmSync(source, { recursive: true, force: true });
    rmSync(storage, { recursive: true, force: true });
  }
});

test('copying a model that was linked does not delete the user\'s original', async () => {
  const ModelManager = await loadModelManager();
  const registry = fakeRegistry();
  const source = mkdtempSync(path.join(tmpdir(), 'local-llm-relink-src-'));
  const storage = mkdtempSync(path.join(tmpdir(), 'local-llm-relink-storage-'));
  const original = path.join(source, 'model.gguf');
  writeFileSync(original, ggufBytes());
  try {
    const manager = new ModelManager({ ...context, globalStorageUri: uriFor(storage) }, registry, logger);
    await manager.importLocal(uriFor(original), 'link');
    const copied = await manager.importLocal(uriFor(original), 'copy');

    assert.notEqual(copied.filePath, original);
    assert.ok(existsSync(original), 'replacing a linked registration must not delete the file it linked');
  } finally {
    rmSync(source, { recursive: true, force: true });
    rmSync(storage, { recursive: true, force: true });
  }
});

test('a linked single Safetensors file is staged as a hard link, not a second copy', async () => {
  const ModelManager = await loadModelManager();
  const source = mkdtempSync(path.join(tmpdir(), 'local-llm-hardlink-src-'));
  const storage = mkdtempSync(path.join(tmpdir(), 'local-llm-hardlink-storage-'));
  const weights = path.join(source, 'gemma4.safetensors');
  writeFileSync(weights, shard('{"w":{"dtype":"BF16","shape":[8],"data_offsets":[0,16]}}', 128));
  writeFileSync(path.join(source, 'config.json'), JSON.stringify({ architectures: ['GemmaForCausalLM'] }));
  try {
    const manager = new ModelManager({ ...context, globalStorageUri: uriFor(storage) }, fakeRegistry(), logger);
    const staged = await manager.stageSafetensorsFile(uriFor(weights), {}, 'link');

    const stagedWeights = statSync(path.join(staged.directory, 'gemma4.safetensors'));
    assert.equal(stagedWeights.ino, statSync(weights).ino, 'the staged weights are the same file on disk');
    assert.equal(statSync(weights).nlink, 2);
  } finally {
    rmSync(source, { recursive: true, force: true });
    rmSync(storage, { recursive: true, force: true });
  }
});
>>>>>>> Stashed changes
