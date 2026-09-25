import type { ChatRequest, ChatStreamEvent, InstalledModel } from '../domain.js';
import type { ChatResult } from '../worker/llamaClient.js';

export const MODEL_BENCHMARK_SAMPLE_COUNT = 3;
// Long enough for a stable decode rate; the prompt asks for far more, so a
// sample never ends early.
const MODEL_BENCHMARK_MAX_TOKENS = 256;

interface BenchmarkClient {
  getModelProfile(signal?: AbortSignal): Promise<{ loadedContextSize: number }>;
  chat(
    request: ChatRequest,
    onEvent: (event: ChatStreamEvent) => void,
    signal?: AbortSignal,
  ): Promise<ChatResult>;
}

export interface ModelBenchmarkResult {
  minTokensPerSecond: number;
  averageTokensPerSecond: number;
  maxTokensPerSecond: number;
}

export function isModelValidated(model: InstalledModel): boolean {
  return model.runtimeProfile !== undefined &&
    model.capabilities.toolCalling !== 'unverified' &&
    model.capabilities.fillInMiddle !== 'unverified';
}

export async function benchmarkModel(
  client: BenchmarkClient,
  signal: AbortSignal,
  onSampleComplete?: (completed: number) => void,
): Promise<ModelBenchmarkResult> {
  const profile = await client.getModelProfile(signal);
  const maxTokens = Math.min(
    MODEL_BENCHMARK_MAX_TOKENS,
    Math.max(1, profile.loadedContextSize - 1),
  );
  const rates: number[] = [];
  for (let sample = 0; sample < MODEL_BENCHMARK_SAMPLE_COUNT; sample += 1) {
    const result = await client.chat(
      {
        messages: [
          {
            role: 'system',
            content: 'Follow the user instruction exactly and emit only the requested list.',
          },
          {
            role: 'user',
            content: 'Write a numbered list from 1 through 100. Put the word benchmark after each number.',
          },
        ],
        toolChoice: 'none',
        inputTokenBudget: Math.max(1, profile.loadedContextSize - maxTokens),
        maxTokens,
        temperature: 0,
        // A thinking model may spend the whole sample reasoning; the rate is
        // what is measured, not the answer.
        allowReasoningOnly: true,
      },
      () => undefined,
      signal,
    );
    const rate = result.tokensPerSecond;
    if (rate === undefined || !Number.isFinite(rate) || rate <= 0) {
      throw new Error('The local worker did not report output token timing data.');
    }
    rates.push(rate);
    onSampleComplete?.(sample + 1);
  }
  return summarizeTokenRates(rates);
}

export function summarizeTokenRates(rates: readonly number[]): ModelBenchmarkResult {
  if (!rates.length || rates.some((rate) => !Number.isFinite(rate) || rate <= 0)) {
    throw new Error('Benchmark token rates must be positive finite numbers.');
  }
  return {
    minTokensPerSecond: Math.min(...rates),
    averageTokensPerSecond: rates.reduce((total, rate) => total + rate, 0) / rates.length,
    maxTokensPerSecond: Math.max(...rates),
  };
}