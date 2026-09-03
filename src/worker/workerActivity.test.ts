import assert from 'node:assert/strict';
import test from 'node:test';
import {
  beginWorkerActivity,
  finishWorkerActivity,
} from './workerActivity.ts';

test('chat work marks a ready worker as generating a response', () => {
  assert.deepEqual(
    beginWorkerActivity({ kind: 'ready', modelId: 'qwen', port: 8080 }, 'chat'),
    {
      kind: 'ready',
      modelId: 'qwen',
      port: 8080,
      activity: 'generating-response',
    },
  );
});

test('utility work does not present itself as response generation', () => {
  assert.deepEqual(
    beginWorkerActivity({ kind: 'ready', modelId: 'qwen', port: 8080 }, 'utility'),
    { kind: 'ready', modelId: 'qwen', port: 8080 },
  );
});

test('finishing response generation returns the worker to ready', () => {
  assert.deepEqual(
    finishWorkerActivity({
      kind: 'ready',
      modelId: 'qwen',
      port: 8080,
      activity: 'generating-response',
    }),
    { kind: 'ready', modelId: 'qwen', port: 8080 },
  );
});
