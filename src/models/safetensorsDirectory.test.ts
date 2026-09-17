import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import * as path from 'node:path';
import test from 'node:test';
import {
  collectCheckpointFiles,
  fingerprintCheckpoint,
  isSafetensorsDirectory,
  MAX_SAFETENSORS_HEADER_BYTES,
  readSafetensorsCheckpoint,
  readSafetensorsHeader,
  validateCheckpointStatic,
} from './safetensorsDirectory.ts';

function shard(header: string, payloadBytes = 32): Buffer {
  const body = Buffer.from(header);
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(body.length));
  return Buffer.concat([prefix, body, Buffer.alloc(payloadBytes)]);
}

function checkpoint(config: object = {}): string {
  const directory = mkdtempSync(path.join(tmpdir(), 'local-llm-ckpt-'));
  writeFileSync(path.join(directory, 'config.json'), JSON.stringify(config));
  writeFileSync(
    path.join(directory, 'model.safetensors'),
    shard('{"w":{"dtype":"BF16","shape":[16],"data_offsets":[0,32]}}'),
  );
  return directory;
}

test('recognises a checkpoint directory, and rejects what is not one', async () => {
  const valid = checkpoint();
  const weightless = mkdtempSync(path.join(tmpdir(), 'local-llm-empty-'));
  writeFileSync(path.join(weightless, 'config.json'), '{}');
  try {
    assert.equal(await isSafetensorsDirectory(valid), true);
    assert.equal(await isSafetensorsDirectory(weightless), false);
    assert.equal(await isSafetensorsDirectory('/definitely/not/here'), false);
  } finally {
    rmSync(valid, { recursive: true, force: true });
    rmSync(weightless, { recursive: true, force: true });
  }
});

test('reads a header without reading the payload', async () => {
  const directory = checkpoint();
  try {
    const header = await readSafetensorsHeader(path.join(directory, 'model.safetensors'));
    assert.ok(header);
    assert.match(header.toString('utf8'), /"dtype":"BF16"/);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('a truncated or absurd header is reported as absent, not thrown', async () => {
  const directory = mkdtempSync(path.join(tmpdir(), 'local-llm-bad-'));
  try {
    writeFileSync(path.join(directory, 'short.safetensors'), Buffer.alloc(4));
    const absurd = Buffer.alloc(8);
    absurd.writeBigUInt64LE(BigInt(Number.MAX_SAFE_INTEGER));
    writeFileSync(path.join(directory, 'absurd.safetensors'), absurd);

    assert.equal(await readSafetensorsHeader(path.join(directory, 'short.safetensors')), undefined);
    assert.equal(await readSafetensorsHeader(path.join(directory, 'absurd.safetensors')), undefined);
    assert.equal(await readSafetensorsHeader(path.join(directory, 'missing.safetensors')), undefined);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('static validation passes a complete checkpoint', async () => {
  const directory = checkpoint({ architectures: ['LlamaForCausalLM'] });
  writeFileSync(path.join(directory, 'tokenizer.json'), '{}');
  try {
    const result = await validateCheckpointStatic(directory);
    assert.equal(result.ok, true);
    assert.deepEqual(result.errors, []);
    assert.deepEqual(result.warnings, []);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('static validation warns on missing tokenizer and unknown arch', async () => {
  const directory = checkpoint();
  try {
    const result = await validateCheckpointStatic(directory);
    assert.equal(result.ok, true);
    assert.match(result.warnings.join(' '), /tokenizer/);
    assert.match(result.warnings.join(' '), /architecture/);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('static validation fails corrupt weights and missing config', async () => {
  const corrupt = checkpoint();
  writeFileSync(path.join(corrupt, 'model.safetensors'), Buffer.from('garbage'));
  const noconfig = mkdtempSync(path.join(tmpdir(), 'local-llm-nocfg-'));
  writeFileSync(
    path.join(noconfig, 'model.safetensors'),
    shard('{"w":{"dtype":"BF16","shape":[16],"data_offsets":[0,32]}}'),
  );
  try {
    const badHeaders = await validateCheckpointStatic(corrupt);
    assert.equal(badHeaders.ok, false);
    assert.match(badHeaders.errors.join(' '), /header/);
    const missing = await validateCheckpointStatic(noconfig);
    assert.equal(missing.ok, false);
    assert.match(missing.errors.join(' '), /config\.json/);
    const gone = await validateCheckpointStatic('/definitely/not/here');
    assert.equal(gone.ok, false);
  } finally {
    rmSync(corrupt, { recursive: true, force: true });
    rmSync(noconfig, { recursive: true, force: true });
  }
});

test('the header cap matches the Python inspector (128 MiB)', async () => {
  // Parity contract with resources/runtime/runtime/inspector.py
  // MAX_HEADER_BYTES. Both sides must agree or one side inspects what the
  // other refuses to load.
  assert.equal(MAX_SAFETENSORS_HEADER_BYTES, 128 * 1024 * 1024);
});

test('a header claim above the shared cap is refused without allocating', async () => {
  // Shared fixture with resources/runtime/tests/test_runtime.py: a 14-byte
  // file claiming a 200 MiB header. Must return undefined, not throw or
  // attempt a 200 MiB allocation.
  const fixture = path.join(
    import.meta.dirname, '..', '..', 'resources', 'runtime', 'tests',
    'fixtures', 'oversize-header-claim.safetensors',
  );
  assert.equal(await readSafetensorsHeader(fixture), undefined);
});

test('collects every regular file and skips nested directories', async () => {
  const directory = checkpoint();
  mkdirSync(path.join(directory, 'nested'));
  writeFileSync(path.join(directory, 'nested', 'ignored.safetensors'), shard('{}'));
  writeFileSync(path.join(directory, 'tokenizer.json'), '{}');
  try {
    const files = await collectCheckpointFiles(directory);
    assert.deepEqual(files.map((file) => file.path).sort(), [
      'config.json', 'model.safetensors', 'tokenizer.json',
    ]);
    // Headers are read only for shards.
    assert.ok(files.find((file) => file.path === 'model.safetensors')?.header);
    assert.equal(files.find((file) => file.path === 'tokenizer.json')?.header, undefined);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('fingerprinting reads no headers and stays sorted', async () => {
  const directory = checkpoint();
  try {
    const files = await fingerprintCheckpoint(directory);
    assert.deepEqual(files.map((file) => file.path), ['config.json', 'model.safetensors']);
    assert.deepEqual(Object.keys(files[0] ?? {}).sort(), ['modifiedAt', 'path', 'size']);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('describes a checkpoint from its config', async () => {
  const directory = checkpoint({
    architectures: ['Qwen3MoeForCausalLM'],
    max_position_embeddings: 262144,
    quantization_config: { quant_method: 'fp8' },
  });
  try {
    const described = await readSafetensorsCheckpoint(directory);
    assert.equal(described.architecture, 'Qwen3MoeForCausalLM');
    assert.equal(described.trainedContextLength, 262144);
    assert.equal(described.quantization, 'fp8');
    assert.equal(described.customCodeRequired, false);
    assert.equal(described.identity.files.length, 2);
    assert.match(described.identity.digest, /^[0-9a-f]{64}$/);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('reads a compressed-tensors quantization format from config_groups', async () => {
  const directory = checkpoint({
    quantization_config: { config_groups: { group_0: { format: 'nvfp4-pack-quantized' } } },
  });
  try {
    assert.equal((await readSafetensorsCheckpoint(directory)).quantization, 'nvfp4-pack-quantized');
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('flags a checkpoint that would execute its own Python', async () => {
  const directory = checkpoint({
    auto_map: { AutoModelForCausalLM: 'modeling_custom.CustomModel' },
  });
  try {
    assert.equal((await readSafetensorsCheckpoint(directory)).customCodeRequired, true);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('an unreadable config warns but still yields an identity', async () => {
  const directory = checkpoint();
  writeFileSync(path.join(directory, 'config.json'), 'not json');
  const warnings: string[] = [];
  try {
    const described = await readSafetensorsCheckpoint(directory, (w) => warnings.push(w));
    assert.equal(described.architecture, undefined);
    assert.equal(described.customCodeRequired, false);
    assert.match(described.identity.digest, /^[0-9a-f]{64}$/);
    assert.equal(warnings.length, 1);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('nested text_config supplies the context length', async () => {
  const directory = checkpoint({ text_config: { max_position_embeddings: 32768 } });
  try {
    assert.equal((await readSafetensorsCheckpoint(directory)).trainedContextLength, 32768);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
