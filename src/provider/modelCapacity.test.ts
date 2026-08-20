import assert from 'node:assert/strict';
import test from 'node:test';
import { modelTokenLimits, resolveAdvertisedContextSize } from './modelCapacity.ts';

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

test('an explicit context setting always wins', () => {
  assert.equal(
    resolveAdvertisedContextSize({
      configuredContextSize: 16384,
      loadedContextSize: 49920,
      trainedContextLength: 131072,
    }),
    16384,
  );
});

test('automatic sizing advertises the window the worker actually loaded', () => {
  assert.equal(
    resolveAdvertisedContextSize({
      configuredContextSize: 0,
      loadedContextSize: 49920,
      trainedContextLength: 131072,
    }),
    49920,
  );
});

test('before any load, automatic sizing falls back to the trained window', () => {
  assert.equal(
    resolveAdvertisedContextSize({
      configuredContextSize: 0,
      trainedContextLength: 131072,
    }),
    131072,
  );
});

test('with nothing known, automatic sizing uses the llama.cpp reduction floor', () => {
  assert.equal(resolveAdvertisedContextSize({ configuredContextSize: 0 }), 4096);
});
