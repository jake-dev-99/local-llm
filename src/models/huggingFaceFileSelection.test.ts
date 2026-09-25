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

test('selectableHuggingFaceFiles excludes vision projectors wherever mmproj appears in the name', () => {
  const models = [
    { filename: 'Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf', size: 14_334_446_752 },
    { filename: 'Qwen2.5-VL-7B-Instruct.Q4_K_M.gguf', size: 4_683_073_184 },
  ];
  const projectors = [
    { filename: 'mmproj-BF16.gguf', size: 879_258_272 },
    { filename: 'MMPROJ-F32.gguf', size: 1_755_867_808 },
    { filename: 'mmproj-google_gemma-3-4b-it-f16.gguf', size: 851_251_104 },
    { filename: 'llava-v1.5-7b-mmproj-model-f16.gguf', size: 624_434_336 },
    { filename: 'Qwen2.5-VL-7B-Instruct.mmproj-Q8_0.gguf', size: 853_119_328 },
    { filename: 'nested/mmproj-F16.gguf', size: 878_054_048 },
  ];

  assert.deepEqual(selectableHuggingFaceFiles([...models, ...projectors]), models);
});

test('selectableHuggingFaceFiles excludes importance matrices saved as GGUF', () => {
  const models = [
    { filename: 'mistralai_Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf', size: 14_334_446_752 },
    { filename: 'Qwen2-7B-Instruct.i1-Q4_K_M.gguf', size: 4_683_073_184 },
    { filename: 'Llama-3-8B-Instruct-imat-Q4_K_M.gguf', size: 4_920_734_016 },
  ];
  const matrices = [
    { filename: 'mistralai_Devstral-Small-2-24B-Instruct-2512-imatrix.gguf', size: 10_037_344 },
    { filename: 'imatrix.gguf', size: 5_018_672 },
  ];

  assert.deepEqual(selectableHuggingFaceFiles([...models, ...matrices]), models);
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
