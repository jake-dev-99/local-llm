import assert from 'node:assert/strict';
import test from 'node:test';
import { modelTokenLimits } from './modelCapacity.ts';

test('modelTokenLimits never advertises more than the physical context window', () => {
  assert.deepEqual(modelTokenLimits(32_768, 2_048), {
    maxInputTokens: 30_720,
    maxOutputTokens: 2_048,
  });
});

test('modelTokenLimits clamps output while preserving at least one input token', () => {
  assert.deepEqual(modelTokenLimits(512, 2_048), {
    maxInputTokens: 1,
    maxOutputTokens: 511,
  });
});
