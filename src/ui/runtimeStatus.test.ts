import assert from 'node:assert/strict';
import test from 'node:test';
import { runtimeStatusPresentation } from './runtimeStatus.ts';

test('starting presents model loading as a visible phase', () => {
  assert.deepEqual(
    runtimeStatusPresentation({ kind: 'starting', runtime: 'llama-cpp', modelId: 'qwen' }, 1),
    {
      text: '$(loading~spin) Local LLM: Model Loading',
      tooltip: 'Loading model into memory',
      error: false,
    },
  );
});

test('ready chat activity presents response generation as a visible phase', () => {
  assert.deepEqual(
    runtimeStatusPresentation({
      kind: 'ready',
      runtime: 'llama-cpp',
      modelId: 'qwen',
      port: 8080,
      activity: 'generating-response',
    }, 1),
    {
      text: '$(loading~spin) Local LLM: Generating Response',
      tooltip: 'Generating response',
      error: false,
    },
  );
});

test('ready without chat activity remains ready', () => {
  assert.deepEqual(
    runtimeStatusPresentation({ kind: 'ready', runtime: 'llama-cpp', modelId: 'qwen', port: 8080 }, 1),
    {
      text: '$(sparkle) Local LLM',
      tooltip: 'Local model ready',
      error: false,
    },
  );
});
