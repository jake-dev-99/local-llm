import assert from 'node:assert/strict';
import test from 'node:test';
import { selectableHuggingFaceFiles } from './huggingFaceFileSelection.ts';

test('selectableHuggingFaceFiles excludes numbered GGUF shard parts', () => {
  const files = [
    { filename: 'model-q4_k_m.gguf', size: 20_000 },
    { filename: 'model-q4_k_m-00001-of-00003.gguf', size: 10_000 },
    { filename: 'nested/model-q4_k_m-00002-of-00003.GGUF', size: 10_000 },
  ];

  assert.deepEqual(selectableHuggingFaceFiles(files), [files[0]]);
});
