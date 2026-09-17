import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import * as path from 'node:path';
import test from 'node:test';
import { PythonWorkerClient } from './pythonWorkerClient.ts';
import { spawnPythonWorkerTransport } from './pythonWorkerProcess.ts';
import { PythonWorkerError } from './pythonWorkerTypes.ts';

const RUNTIME_DIRECTORY = path.resolve(import.meta.dirname, '../../resources/runtime');
const PYTHON = process.env['LOCAL_LLM_TEST_PYTHON'] ?? 'python3';

/**
 * These drive the real worker process. They exercise only the standard-library
 * paths — inspection, framing, error mapping, crash recovery — so they run
 * without torch or Transformers installed.
 */
const pythonAvailable = (() => {
  try {
    execFileSync(PYTHON, ['-c', 'import sys; sys.exit(0)'], { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
})();

/** A checkpoint that is structurally real: valid Safetensors headers, two shards. */
function writeCheckpoint(directory: string): void {
  writeFileSync(path.join(directory, 'config.json'), JSON.stringify({
    architectures: ['Qwen3MoeForCausalLM'],
    model_type: 'qwen3_moe',
    max_position_embeddings: 262144,
    quantization_config: { quant_method: 'fp8' },
  }));
  writeFileSync(path.join(directory, 'chat_template.jinja'), '{{ messages }}');
  for (const [index, bytes] of [[1, 4096], [2, 2048]] as const) {
    const header = Buffer.from(JSON.stringify({
      [`model.layers.${index}.weight`]: {
        dtype: 'BF16', shape: [bytes / 2], data_offsets: [0, bytes],
      },
    }));
    const prefix = Buffer.alloc(8);
    prefix.writeBigUInt64LE(BigInt(header.length));
    writeFileSync(
      path.join(directory, `model-0000${index}-of-00002.safetensors`),
      Buffer.concat([prefix, header, Buffer.alloc(bytes)]),
    );
  }
}

function withWorker<T>(run: (client: PythonWorkerClient, modelPath: string) => Promise<T>) {
  return async () => {
    const directory = mkdtempSync(path.join(tmpdir(), 'local-llm-safetensors-'));
    writeCheckpoint(directory);
    const transport = spawnPythonWorkerTransport({
      pythonPath: PYTHON,
      runtimeDirectory: RUNTIME_DIRECTORY,
    });
    const client = new PythonWorkerClient({ transport });
    try {
      return await run(client, directory);
    } finally {
      client.dispose();
      rmSync(directory, { recursive: true, force: true });
    }
  };
}

test('inspects a sharded checkpoint without loading any weights', {
  skip: pythonAvailable ? false : `${PYTHON} is not available`,
}, withWorker(async (client, modelPath) => {
  const inspection = await client.inspect(modelPath);

  assert.equal(inspection.architecture, 'Qwen3MoeForCausalLM');
  assert.equal(inspection.modelType, 'qwen3_moe');
  assert.equal(inspection.contextLength, 262144);
  assert.equal(inspection.sharded, true);
  assert.equal(inspection.fileCount, 2);
  // Read from the Safetensors headers, not from the file sizes.
  assert.equal(inspection.weightBytes, 6144);
  assert.equal(inspection.dtype, 'BF16');
  assert.equal(inspection.quantization, 'fp8');
  assert.equal(inspection.supportsChat, true);
  assert.equal(inspection.customCodeRequired, false);
}));

test('maps worker failures onto the error contract', {
  skip: pythonAvailable ? false : `${PYTHON} is not available`,
}, withWorker(async (client) => {
  await assert.rejects(
    client.inspect('/definitely/not/a/model'),
    (error: unknown) => error instanceof PythonWorkerError &&
      error.code === 'invalid_model',
  );
  await assert.rejects(
    client.unload().then(() => client.chat([{ role: 'user', content: 'hi' }])),
    (error: unknown) => error instanceof PythonWorkerError &&
      error.code === 'model_not_loaded',
  );
}));

test('reports worker lifecycle state over the protocol', {
  skip: pythonAvailable ? false : `${PYTHON} is not available`,
}, async () => {
  const states: string[] = [];
  const transport = spawnPythonWorkerTransport({
    pythonPath: PYTHON,
    runtimeDirectory: RUNTIME_DIRECTORY,
  });
  const client = new PythonWorkerClient({
    transport,
    onState: (state) => states.push(state),
  });
  try {
    await client.inspect(mkdtempSync(path.join(tmpdir(), 'empty-'))).catch(() => undefined);
    assert.ok(states.includes('READY'), `expected READY in ${states.join(',')}`);
  } finally {
    client.dispose();
  }
});
