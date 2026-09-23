import assert from 'node:assert/strict';
import test from 'node:test';
import {
  safetensorsHuggingFaceFiles,
  selectableHuggingFaceFiles,
} from './huggingFaceFileSelection.ts';

test('selectableHuggingFaceFiles excludes numbered GGUF shard parts', () => {
  const files = [
    { filename: 'model-q4_k_m.gguf', size: 20_000 },
    { filename: 'model-q4_k_m-00001-of-00003.gguf', size: 10_000 },
    { filename: 'nested/model-q4_k_m-00002-of-00003.GGUF', size: 10_000 },
  ];

  assert.deepEqual(selectableHuggingFaceFiles(files), [files[0]]);
});

test('safetensorsHuggingFaceFiles selects the complete Qwen MLX checkpoint', () => {
  const files = [
    { filename: '.gitattributes', size: 1_500 },
    { filename: 'README.md', size: 13_000 },
    { filename: 'chat_template.jinja', size: 8_057 },
    { filename: 'config.json', size: 116_599 },
    { filename: 'model-00001-of-00005.safetensors', size: 5_365_814_198 },
    { filename: 'model-00002-of-00005.safetensors', size: 5_257_566_213 },
    { filename: 'model-00003-of-00005.safetensors', size: 5_254_090_144 },
    { filename: 'model-00004-of-00005.safetensors', size: 5_217_177_892 },
    { filename: 'model-00005-of-00005.safetensors', size: 540_344_667 },
    { filename: 'model.safetensors.index.json', size: 199_635 },
    { filename: 'processor_config.json', size: 1_191 },
    { filename: 'tokenizer.json', size: 19_989_343 },
    { filename: 'tokenizer_config.json', size: 1_482 },
    { filename: 'onnx/model.safetensors', size: 100 },
    { filename: 'pytorch_model.bin', size: 100 },
  ];

  assert.deepEqual(
    safetensorsHuggingFaceFiles(files).map((file) => file.filename),
    [
      'chat_template.jinja',
      'config.json',
      'model-00001-of-00005.safetensors',
      'model-00002-of-00005.safetensors',
      'model-00003-of-00005.safetensors',
      'model-00004-of-00005.safetensors',
      'model-00005-of-00005.safetensors',
      'model.safetensors.index.json',
      'processor_config.json',
      'tokenizer.json',
      'tokenizer_config.json',
    ],
  );
});

test('safetensorsHuggingFaceFiles requires root config and weights', () => {
  assert.deepEqual(safetensorsHuggingFaceFiles([
    { filename: 'model.safetensors', size: 100 },
  ]), []);
  assert.deepEqual(safetensorsHuggingFaceFiles([
    { filename: 'config.json', size: 100 },
  ]), []);
});
