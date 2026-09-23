/**
 * Adapts the Python Safetensors runtime to the extension's inference contract.
 *
 * The mapping is deliberately thin. Role formatting, special tokens and
 * generation markers belong to the model's own chat template, which the worker
 * applies through the tokenizer; reproducing any of that here would drift per
 * model family.
 */

import { randomUUID } from 'node:crypto';
import type { ChatMessage, ChatRequest, ChatStreamEvent, ChatTool } from '../domain.ts';
import { finalResponseMessages } from './finalResponse.ts';
import type { InferenceCapabilities, InferenceClient } from './inferenceClient.ts';
import type { ChatResult, NativeToolCallSupport } from './llamaClient.ts';
import type { PythonWorkerClient } from './pythonWorkerClient.ts';
import { PythonWorkerError, type PythonChatMessage, type PythonModelInfo } from './pythonWorkerTypes.ts';
import type { WorkerModelProfile } from './runtimeProfile.ts';
import { parseToolDecision, toolDecisionJsonSchema } from './toolProtocol.ts';

export class TransformersClient implements InferenceClient {
  /**
   * Transformers has no `/infill` equivalent, and FIM tokens live outside the
   * chat template. Constrained decoding is available when the provisioned
   * environment carries the grammar backend; the worker advertises it through
   * its reported package versions.
   */
  readonly supports: InferenceCapabilities;

  private readonly worker: PythonWorkerClient;
  private readonly modelInfo: PythonModelInfo;

  constructor(worker: PythonWorkerClient, modelInfo: PythonModelInfo) {
    this.worker = worker;
    this.modelInfo = modelInfo;
    this.supports = {
      infill: false,
      constrainedDecoding: hasGrammarBackend(modelInfo),
    };
  }

  async chat(
    request: ChatRequest,
    onEvent: (event: ChatStreamEvent) => void,
    signal?: AbortSignal,
  ): Promise<ChatResult> {
    const tools = request.tools ?? [];
    const toolChoice = request.toolChoice ?? 'auto';
    if (toolChoice === 'required' && tools.length === 0) {
      throw new Error('The caller required a tool call but supplied no tool definitions.');
    }
    // No native tool-call channel here: unlike llama.cpp there is no server
    // to parse model-emitted calls, so a fresh tool turn goes straight to a
    // schema-constrained decision. A turn that already carries a tool result
    // is the counterpart of llama's native final: the call happened, so an
    // automatic turn answers with text. Only `required` decides again.
    if (isDecisionTurn(request.messages, tools, toolChoice)) {
      if (!this.supports.constrainedDecoding) {
        assertGrammarAvailable();
      }
      return this.schemaConstrainedDecision(request, tools, toolChoice === 'required', onEvent, signal);
    }
    assertNoDanglingToolCall(request);
    const messages = toWorkerMessages(request.messages);
    // A turn continuing after a tool result opens the reasoning channel:
    // thinking templates (Gemma, Qwen3, …) answer post-result turns through
    // it, and the worker strips the trace back to the final text. Templates
    // without the kwarg render exactly as before.
    const continued = request.messages.some((message) => message.role === 'tool');
    const startedAt = Date.now();
    let textCharacters = 0;

    const release = this.forwardCancellation(signal);
    try {
      const generated = await this.worker.chat(messages, {
        options: {
          maxNewTokens: request.maxTokens,
          temperature: request.temperature,
          ...(continued ? { enableThinking: true } : {}),
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
   * A tool turn is counted exactly as it would generate: catalogue and
   * instruction included, so the budget check sees the real prompt.
   */
  async countChatInputTokens(
    messages: ChatMessage[],
    tools?: ChatTool[],
    toolChoice?: ChatRequest['toolChoice'],
    _signal?: AbortSignal,
  ): Promise<number> {
    const counted = isDecisionTurn(messages, tools ?? [], toolChoice)
      ? decisionMessages(messages, tools ?? [], toolChoice === 'required')
      : toWorkerMessages(messages);
    return await this.worker.tokenize({ messages: counted });
  }

  /**
   * One grammar-constrained tool decision, mirroring llama.cpp's
   * schema-constrained fallback: the worker enforces the decision schema
   * token-by-token, and the result is validated before anyone acts on it.
   */
  private async schemaConstrainedDecision(
    request: ChatRequest,
    tools: readonly ChatTool[],
    toolRequired: boolean,
    onEvent: (event: ChatStreamEvent) => void,
    signal?: AbortSignal,
  ): Promise<ChatResult> {
    const maxTokens = Math.min(
      request.maxTokens,
      Math.max(1, Math.floor(request.toolCallMaxTokens ?? request.maxTokens)),
    );
    const messages = decisionMessages(request.messages, tools, toolRequired);
    const inputTokens = await this.worker.tokenize({ messages });
    if (inputTokens > request.inputTokenBudget) {
      throw new Error(
        `The schema-constrained fallback requires ${inputTokens} input tokens, but the local model input budget is ${request.inputTokenBudget}.`,
      );
    }
    let textCharacters = 0;
    const release = this.forwardCancellation(signal);
    try {
      const generated = await this.worker.chat(messages, {
        options: {
          maxNewTokens: maxTokens,
          temperature: 0,
          jsonSchema: toolDecisionJsonSchema(tools, toolRequired),
        },
        onToken: (chunk) => {
          textCharacters += chunk.length;
        },
      });
      const text = generated.text;
      if (!text.trim()) {
        throw new Error(
          'The schema-constrained fallback returned no decision. The worker streamed an empty response.',
        );
      }
      const decision = parseToolDecision(text, tools, toolRequired);
      if (decision.kind === 'tool') {
        onEvent({
          kind: 'toolCall',
          id: `call-${randomUUID()}`,
          name: decision.name,
          input: decision.arguments,
        });
        return {
          inputTokens: generated.inputTokens,
          textCharacters,
          toolCallCount: 1,
        };
      }
      return this.chat(
        {
          ...request,
          tools: [],
          toolChoice: 'none',
          messages: finalResponseMessages(request.messages),
        },
        onEvent,
        signal,
      );
    } finally {
      release();
    }
  }

  /**
   * Structurally unavailable, so there is nothing to discover or cache.
   *
   * llama.cpp learns this from a model's first reply because its server may
   * or may not parse tool calls out of one. Here there is no native channel
   * to learn about: tool turns are answered by schema-constrained decisions
   * (which still surface as `toolCall` events), so the setter has no
   * observation worth recording.
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
   * Tool support follows the grammar backend: claiming it without xgrammar
   * would let the Local Agent select this model and then fail mid-request.
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
 * Whether the worker that produced this model info carries the grammar
 * backend. Old provisioned environments predate it, so the capability is read
 * off the reported versions rather than assumed from the extension side.
 */
function hasGrammarBackend(modelInfo: PythonModelInfo): boolean {
  const version = modelInfo.runtime.versions['xgrammar'];
  return typeof version === 'string' && version !== '' && version !== 'not installed';
}

/**
 * Refuses a tool turn when the grammar backend is missing.
 *
 * Without constrained decoding a tool request would return prose the agent
 * cannot parse. Failing with a clear reason beats producing an answer that
 * silently ignored every tool.
 */
function assertGrammarAvailable(): void {
  throw new PythonWorkerError(
    'unsupported_grammar',
    'Safetensors models cannot use tools yet: this runtime has no ' +
    'schema-constrained decoding. Use ordinary Chat, or select a GGUF model ' +
    'for Local Agent.',
  );
}

/**
 * Whether this turn makes a tool decision: tools supplied, execution enabled,
 * and — for automatic turns — no tool result in history yet. A result already
 * present means the call happened and the turn answers with text.
 */
function isDecisionTurn(
  messages: readonly ChatMessage[],
  tools: readonly ChatTool[],
  toolChoice: ChatRequest['toolChoice'],
): boolean {
  if (tools.length === 0 || (toolChoice ?? 'auto') === 'none') {
    return false;
  }
  if (toolChoice === 'required') {
    return true;
  }
  return !messages.some((message) => message.role === 'tool');
}

/**
 * An assistant tool-call record with no later tool result is a half-turn no
 * template can continue: the call went out but its result never came back.
 */
function assertNoDanglingToolCall(request: ChatRequest): void {
  let dangling = false;
  for (const message of request.messages) {
    if ((message.tool_calls?.length ?? 0) > 0) {
      dangling = true;
    } else if (message.role === 'tool') {
      dangling = false;
    }
  }
  if (dangling) {
    throw new PythonWorkerError(
      'unsupported_grammar',
      'Safetensors models cannot use tools yet: this runtime has no ' +
      'schema-constrained decoding. Use ordinary Chat, or select a GGUF model ' +
      'for Local Agent.',
    );
  }
}

/**
 * What the worker's template receives. Role formatting stays with the
 * template, but the tool-call record travels with the message: without it a
 * tool result arrives unattributed and the model cannot use it. Fields the
 * template does not take are still stripped.
 */
function toWorkerMessages(
  messages: readonly ChatMessage[],
): PythonChatMessage[] {
  return messages.map((message) => ({
    role: message.role,
    content: message.content,
    ...(message.tool_call_id ? { tool_call_id: message.tool_call_id } : {}),
    ...(message.tool_calls?.length ? { tool_calls: message.tool_calls.map(toWorkerTemplateCall) } : {}),
  }));
}

/**
 * The extension carries call arguments as a JSON string (the OpenAI shape);
 * chat templates take the parsed mapping. A string that does not parse is
 * passed through untouched so the template — which knows its own format —
 * reports the problem.
 */
function toWorkerTemplateCall(
  call: NonNullable<ChatMessage['tool_calls']>[number],
): NonNullable<PythonChatMessage['tool_calls']>[number] {
  let args: unknown = call.function.arguments;
  if (typeof args === 'string') {
    try {
      args = JSON.parse(args) as unknown;
    } catch {
      // Leave it; the template reports what it cannot render.
    }
  }
  return {
    id: call.id,
    type: call.type,
    function: { name: call.function.name, arguments: args },
  };
}

/**
 * The decision prompt: the conversation plus a tool catalogue and a closing
 * instruction. The catalogue is part of the counted and generated prompt, so
 * the model sees the schemas its arguments must satisfy and the token budget
 * sees the same text the model will.
 */
function decisionMessages(
  messages: readonly ChatMessage[],
  tools: readonly ChatTool[],
  toolRequired: boolean,
): Array<{ role: string; content: string }> {
  const catalogue = tools.map((tool) => ({
    name: tool.function.name,
    ...(tool.function.description ? { description: tool.function.description } : {}),
    parameters: tool.function.parameters ?? { type: 'object' },
  }));
  return [
    ...toWorkerMessages(messages),
    {
      role: 'user',
      content: (toolRequired
        ? 'Return exactly one tool decision matching the response schema. Choose the supplied tool needed for the current task.'
        : 'Return one action matching the response schema. Choose kind tool when another tool is needed. Otherwise choose kind final.') +
        ` Available tools: ${JSON.stringify(catalogue)}`,
    },
  ];
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
