import assert from 'node:assert/strict';
import test from 'node:test';
import type { ChatRequest } from '../domain';
import {
  benchmarkModel,
  isModelValidated,
  MODEL_BENCHMARK_SAMPLE_COUNT,
  summarizeTokenRates,
} from './modelBenchmark.ts';

test('only models with completed compatibility checks are benchmarkable', () => {
  const validated = {
    id: 'model',
    name: 'Model',
    filePath: 'model.gguf',
    fileSize: 1,
    sha256: 'sha',
    source: 'import' as const,
    filename: 'model.gguf',
    installedAt: '2026-09-15T00:00:00.000Z',
    capabilities: { toolCalling: 'supported' as const, fillInMiddle: 'unsupported' as const },
    runtimeProfile: {
      validatedAt: '2026-09-15T00:00:00.000Z',
      loadedContextSize: 4_096,
      hasChatTemplate: true,
      supportsTools: true,
      supportsToolCalls: true,
      supportsSystemRole: true,
    },
  };

  assert.equal(isModelValidated(validated), true);
  assert.equal(isModelValidated({
    ...validated,
    capabilities: { ...validated.capabilities, toolCalling: 'unverified' },
  }), false);
  assert.equal(isModelValidated({ ...validated, runtimeProfile: undefined }), false);
});

test('token rates are summarized as min, average, and max', () => {
  assert.deepEqual(summarizeTokenRates([12, 18, 15]), {
    minTokensPerSecond: 12,
    averageTokensPerSecond: 15,
    maxTokensPerSecond: 18,
  });
});

test('benchmark runs three bounded deterministic output samples', async () => {
  const requests: ChatRequest[] = [];
  const completed: number[] = [];
  const rates = [12, 18, 15];
  const result = await benchmarkModel(
    {
      async getModelProfile() {
        return { loadedContextSize: 4_096 };
      },
      async chat(request) {
        requests.push(request);
        return {
          inputTokens: 20,
          textCharacters: 100,
          toolCallCount: 0,
          tokensPerSecond: rates[requests.length - 1],
        };
      },
    },
    new AbortController().signal,
    (sample) => completed.push(sample),
  );

  assert.equal(requests.length, MODEL_BENCHMARK_SAMPLE_COUNT);
  assert.deepEqual(completed, [1, 2, 3]);
  assert.equal(requests.every((request) => request.maxTokens === 64), true);
  assert.equal(requests.every((request) => request.temperature === 0), true);
  assert.deepEqual(result, {
    minTokensPerSecond: 12,
    averageTokensPerSecond: 15,
    maxTokensPerSecond: 18,
  });
});

test('benchmark rejects a worker response without output timing data', async () => {
  await assert.rejects(
    benchmarkModel(
      {
        async getModelProfile() {
          return { loadedContextSize: 4_096 };
        },
        async chat() {
          return { inputTokens: 20, textCharacters: 100, toolCallCount: 0 };
        },
      },
      new AbortController().signal,
    ),
    /did not report output token timing data/i,
  );
});