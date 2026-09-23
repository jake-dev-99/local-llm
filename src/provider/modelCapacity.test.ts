import assert from 'node:assert/strict';
import test from 'node:test';
import { modelTokenLimits, resolveAdvertisedContextSize } from './modelCapacity.ts';

test('modelTokenLimits never advertises more than the physical context window', () => {
  assert.deepEqual(modelTokenLimits(32_768, 2_048), {
    maxInputTokens: 30_720,
    maxOutputTokens: 2_048,
  });
});

test('output never takes more than half the window, so the prompt keeps the rest', () => {
  // A 27B model fitted to 4096 tokens with maxOutputTokens 8192 used to
  // advertise an input limit of 1. VS Code cannot fit any prompt in that and
  // abandons the request without calling the provider.
  assert.deepEqual(modelTokenLimits(4_096, 8_192), {
    maxInputTokens: 2_048,
    maxOutputTokens: 2_048,
  });
  assert.deepEqual(modelTokenLimits(512, 2_048), {
    maxInputTokens: 256,
    maxOutputTokens: 256,
  });
  assert.deepEqual(modelTokenLimits(4_097, 8_192), {
    maxInputTokens: 2_049,
    maxOutputTokens: 2_048,
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
