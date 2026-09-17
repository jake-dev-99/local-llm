/**
 * Adapts the Python Safetensors runtime to the extension's inference contract.
 *
 * The mapping is deliberately thin. Role formatting, special tokens and
 * generation markers belong to the model's own chat template, which the worker
 * applies through the tokenizer; reproducing any of that here would drift per
 * model family.
 */

import type { ChatMessage, ChatRequest, ChatStreamEvent, ChatTool } from '../domain.ts';
import type { InferenceCapabilities, InferenceClient } from './inferenceClient.ts';
import type { ChatResult, NativeToolCallSupport } from './llamaClient.ts';
import type { PythonWorkerClient } from './pythonWorkerClient.ts';
import { PythonWorkerError, type PythonModelInfo } from './pythonWorkerTypes.ts';
import type { WorkerModelProfile } from './runtimeProfile.ts';

export class TransformersClient implements InferenceClient {
  /**
   * Neither capability is available yet.
   *
   * Transformers has no `/infill` equivalent, and FIM tokens live outside the
   * chat template. Constrained decoding needs a grammar backend that the
   * baseline dependency set does not yet carry.
   */
  readonly supports: InferenceCapabilities = {
    infill: false,
    constrainedDecoding: false,
  };

  private readonly worker: PythonWorkerClient;
  private readonly modelInfo: PythonModelInfo;

  constructor(worker: PythonWorkerClient, modelInfo: PythonModelInfo) {
    this.worker = worker;
    this.modelInfo = modelInfo;
  }

  async chat(
    request: ChatRequest,
    onEvent: (event: ChatStreamEvent) => void,
    signal?: AbortSignal,
  ): Promise<ChatResult> {
    assertNoTools(request);
    const messages = request.messages.map(({ role, content }) => ({ role, content }));
    const startedAt = Date.now();
    let textCharacters = 0;

    const release = this.forwardCancellation(signal);
    try {
      const generated = await this.worker.chat(messages, {
        options: {
          maxNewTokens: request.maxTokens,
          temperature: request.temperature,
        },
        onToken: (chunk) => {
          textCharacters += chunk.length;
          onEvent({ kind: 'text', text: chunk });
        },
      });
      // A worker that streamed nothing still returns the whole completion, so
      // the caller is not left with an empty response.
      if (textCharacters === 0 && generated.text) {
        textCharacters = generated.text.length;
        onEvent({ kind: 'text', text: generated.text });
      }
      const elapsedSeconds = (Date.now() - startedAt) / 1000;
      return {
        inputTokens: generated.inputTokens,
        textCharacters,
        toolCallCount: 0,
        ...(elapsedSeconds > 0
          ? { tokensPerSecond: estimateTokensPerSecond(generated.text, elapsedSeconds) }
          : {}),
      };
    } finally {
      release();
    }
  }

  async tokenize(content: string, _signal?: AbortSignal): Promise<number> {
    return await this.worker.tokenize({ text: content });
  }

  /**
   * Counted by applying the model's own chat template, the same way a
   * generation would be, so the role markers are included.
   *
   * Tools are ignored rather than counted: `chat` refuses a request carrying
   * them, so a conversation that reaches generation never has any.
   */
  async countChatInputTokens(
    messages: ChatMessage[],
    _tools?: ChatTool[],
    _toolChoice?: ChatRequest['toolChoice'],
    _signal?: AbortSignal,
  ): Promise<number> {
    return await this.worker.tokenize({
      messages: messages.map(({ role, content }) => ({ role, content })),
    });
  }

  /**
   * Structurally unavailable, so there is nothing to discover or cache.
   *
   * llama.cpp learns this from a model's first reply because its server may
   * or may not parse tool calls out of one. Here the answer does not depend
   * on the model: nothing in this runtime emits a tool call, so the setter
   * has no observation worth recording.
   */
  getNativeToolCallSupport(): NativeToolCallSupport {
    return 'unavailable';
  }

  setNativeToolCallSupport(_support: NativeToolCallSupport): void {
    // Intentionally empty; see getNativeToolCallSupport.
  }

  /**
   * Fails every in-flight request and closes the pipe.
   *
   * Stopping the process itself belongs to the session that started it.
   */
  async dispose(): Promise<void> {
    this.worker.dispose();
  }

  /**
   * The llama.cpp-shaped profile, from what Transformers reports.
   *
   * Tool support is reported as false until a grammar backend ships: claiming
   * it would let the Local Agent select this model and then fail mid-request.
   */
  async getModelProfile(_signal?: AbortSignal): Promise<WorkerModelProfile> {
    const info = this.modelInfo;
    return {
      loadedContextSize: info.contextLength ?? 0,
      hasChatTemplate: info.supportsChat,
      supportsTools: this.supports.constrainedDecoding,
      supportsToolCalls: this.supports.constrainedDecoding,
      // Every Transformers chat template this runtime can use accepts a system
      // turn; templates that do not are rejected by the tokenizer itself.
      supportsSystemRole: info.supportsChat,
      workerBuild: `transformers ${info.runtime.versions['transformers'] ?? 'unknown'}`,
    };
  }

  /** Turns an abort into the worker's own cancellation, and unhooks after. */
  private forwardCancellation(signal: AbortSignal | undefined): () => void {
    if (!signal) {
      return () => undefined;
    }
    const cancel = () => this.worker.cancel();
    if (signal.aborted) {
      cancel();
      return () => undefined;
    }
    signal.addEventListener('abort', cancel, { once: true });
    return () => signal.removeEventListener('abort', cancel);
  }
}

/**
 * Rejects a request carrying tools.
 *
 * Transformers `generate` has no schema-constrained decoding, so a tool request
 * would return prose the agent cannot parse. Failing with a clear reason beats
 * producing an answer that silently ignored every tool.
 */
function assertNoTools(request: ChatRequest): void {
  const hasTools = (request.tools?.length ?? 0) > 0;
  const hasToolTurns = request.messages.some(
    (message) => message.role === 'tool' || (message.tool_calls?.length ?? 0) > 0,
  );
  if (hasTools || hasToolTurns) {
    throw new PythonWorkerError(
      'unsupported_grammar',
      'Safetensors models cannot use tools yet: this runtime has no ' +
      'schema-constrained decoding. Use ordinary Chat, or select a GGUF model ' +
      'for Local Agent.',
    );
  }
}

/**
 * Throughput, estimated from characters.
 *
 * The worker reports no token count for generated text, and asking for one
 * would cost another round trip per response to refine a number shown only as
 * a diagnostic. Four characters per token is the usual English approximation.
 */
function estimateTokensPerSecond(text: string, elapsedSeconds: number): number {
  return Math.round((text.length / 4) / elapsedSeconds);
}
