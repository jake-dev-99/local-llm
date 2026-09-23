import { randomUUID } from 'node:crypto';
import { fetch as pooledFetch, Pool } from 'undici';
import type {
  ChatMessage,
  ChatRequest,
  ChatStreamEvent,
  ChatTool,
  InfillRequest,
} from '../domain.js';
import type { InferenceCapabilities, InferenceClient } from './inferenceClient.ts';
import { parseWorkerModelProfile, type WorkerModelProfile } from './runtimeProfile.js';
import { assertPromptFits } from './toolBudget.js';
import { assertFinalResponse, finalResponseMessages } from './finalResponse.js';
import {
  parseToolDecision,
  toolDecisionResponseFormat,
  type ToolDecision,
  validateToolCallInput,
} from './toolProtocol.js';
import { workerRequestError, workerStreamError, workerTransportError } from './workerError.js';

type WorkerResponse = Awaited<ReturnType<typeof pooledFetch>>;

interface WorkerRequestInit {
  method: 'GET' | 'POST';
  body?: string;
  signal?: AbortSignal;
}

interface OpenAiChunk {
  error?: unknown;
  timings?: {
    cache_n?: number;
    prompt_n?: number;
    prompt_ms?: number;
    prompt_per_second?: number;
    predicted_n?: number;
    predicted_ms?: number;
    predicted_per_second?: number;
  };
  choices?: Array<{
    finish_reason?: string | null;
    delta?: {
      content?: string;
      reasoning_content?: string;
      tool_calls?: Array<{
        index: number;
        id?: string;
        function?: { name?: string; arguments?: string };
      }>;
    };
  }>;
}

interface PendingToolCall {
  id: string;
  name: string;
  arguments: string;
}

interface ConsumedSseFrame {
  text: string;
  reasoning: string;
  finishReason?: string;
  timings?: NonNullable<OpenAiChunk['timings']>;
}

/**
 * What one streamed generation left behind beyond its visible answer. A
 * thinking template sends its reasoning on a separate channel, and a reply
 * can end with no answer text at all.
 */
interface StreamedChatResult extends ChatResult {
  reasoningCharacters?: number;
  finishReason?: string;
  outputTokenLimit?: number;
}

const TOKEN_COUNT_CACHE_MAX_ENTRIES = 4_096;
const TOKEN_COUNT_CACHE_MAX_TEXT_LENGTH = 32 * 1_024;
const WORKER_CONNECTION_MAX_REQUESTS = 64;
const GENERATION_PHASE = '[Generating Response]';

export interface ChatResult {
  inputTokens: number;
  textCharacters: number;
  toolCallCount: number;
  tokensPerSecond?: number;
}

/**
 * Whether the loaded model emits tool calls on llama.cpp's native tool_calls
 * channel. Measured once per worker process, then reused.
 */
export type NativeToolCallSupport = 'unknown' | 'available' | 'unavailable';

export interface LlamaClientDiagnostics {
  info(message: string): void;
}

interface ChatTrace {
  id: string;
  startedAt: number;
}

export class LlamaClient implements InferenceClient {
  /**
   * llama.cpp provides both: `/infill` for fill-in-the-middle, and GBNF
   * grammars compiled from a response schema for constrained tool decisions.
   */
  readonly supports: InferenceCapabilities = {
    infill: true,
    constrainedDecoding: true,
  };

  private modelProfile: WorkerModelProfile | undefined;
  private nativeToolCalls: NativeToolCallSupport = 'unknown';
  private readonly pool: Pool;
  private readonly tokenCounts = new Map<string, number>();
  private chatSequence = 0;

  constructor(
    readonly baseUrl: string,
    private readonly apiKey: string,
    private readonly diagnostics?: LlamaClientDiagnostics,
  ) {
    assertLoopbackWorkerUrl(baseUrl);
    this.pool = new Pool(baseUrl, {
      connections: 1,
      pipelining: 1,
      // llama.cpp retires keep-alive connections after 100 requests. Rotate
      // first so an immediate follow-up cannot race the server's final close.
      maxRequestsPerClient: WORKER_CONNECTION_MAX_REQUESTS,
    });
  }

  getNativeToolCallSupport(): NativeToolCallSupport {
    return this.nativeToolCalls;
  }

  setNativeToolCallSupport(support: NativeToolCallSupport): void {
    this.nativeToolCalls = support;
  }

  async dispose(): Promise<void> {
    this.tokenCounts.clear();
    await this.pool.destroy();
  }

  async health(signal?: AbortSignal): Promise<boolean> {
    try {
      const response = await pooledFetch(`${this.baseUrl}/health`, {
        ...requestSignal(signal),
        dispatcher: this.pool,
      });
      return response.ok;
    } catch (error) {
      if (signal?.aborted) {
        throw error;
      }
      return false;
    }
  }

  async tokenize(content: string, signal?: AbortSignal): Promise<number> {
    const cached = this.tokenCounts.get(content);
    if (cached !== undefined) {
      this.tokenCounts.delete(content);
      this.tokenCounts.set(content, cached);
      return cached;
    }
    const response = await this.request('/tokenize', {
      method: 'POST',
      body: JSON.stringify({
        content,
        add_special: false,
        with_pieces: false,
      }),
      ...requestSignal(signal),
    });
    const payload = (await response.json()) as { tokens?: unknown[] };
    const count = Array.isArray(payload.tokens) ? payload.tokens.length : 0;
    if (content.length <= TOKEN_COUNT_CACHE_MAX_TEXT_LENGTH) {
      while (this.tokenCounts.size >= TOKEN_COUNT_CACHE_MAX_ENTRIES) {
        const oldest = this.tokenCounts.keys().next().value;
        if (oldest === undefined) {
          break;
        }
        this.tokenCounts.delete(oldest);
      }
      this.tokenCounts.set(content, count);
    }
    return count;
  }

  async getModelProfile(signal?: AbortSignal): Promise<WorkerModelProfile> {
    if (this.modelProfile) {
      return this.modelProfile;
    }
    const response = await this.request('/props', {
      method: 'GET',
      ...requestSignal(signal),
    });
    const profile = parseWorkerModelProfile(await response.json());
    this.modelProfile = profile;
    return profile;
  }

  async countChatInputTokens(
    messages: ChatMessage[],
    tools?: ChatTool[],
    toolChoice: ChatRequest['toolChoice'] = 'auto',
    signal?: AbortSignal,
  ): Promise<number> {
    const body: Record<string, unknown> = { model: 'local', messages };
    Object.assign(body, withTools({}, tools, toolChoice));
    return this.countChatBodyInputTokens(body, signal);
  }

  async chat(
    request: ChatRequest,
    onEvent: (event: ChatStreamEvent) => void,
    signal?: AbortSignal,
  ): Promise<ChatResult> {
    const tools = request.tools ?? [];
    const toolChoice = request.toolChoice ?? 'auto';
    const trace: ChatTrace = {
      id: `chat-${++this.chatSequence}`,
      startedAt: Date.now(),
    };
    this.diagnostics?.info(
      `${GENERATION_PHASE} ${trace.id} start: messages=${request.messages.length} tools=${tools.length} ` +
      `toolChoice=${toolChoice} maxOutputTokens=${request.maxTokens}.`,
    );

    const { reasoningCharacters, finishReason, outputTokenLimit, ...result } =
      await this.executeChat(request, onEvent, trace, signal);
    this.diagnostics?.info(
      `${GENERATION_PHASE} ${trace.id} complete: inputTokens=${result.inputTokens} ` +
      `outputCharacters=${result.textCharacters} reasoningCharacters=${reasoningCharacters ?? 0} ` +
      `toolCalls=${result.toolCallCount} finishReason=${finishReason ?? 'unknown'} ` +
      `elapsed=${Date.now() - trace.startedAt} ms.`,
    );
    assertAnswered(result, reasoningCharacters ?? 0, finishReason, outputTokenLimit ?? request.maxTokens);
    return result;
  }

  private async executeChat(
    request: ChatRequest,
    onEvent: (event: ChatStreamEvent) => void,
    trace: ChatTrace,
    signal?: AbortSignal,
  ): Promise<StreamedChatResult> {
    const tools = request.tools ?? [];
    const toolChoice = request.toolChoice ?? 'auto';
    if (toolChoice === 'required' && tools.length === 0) {
      throw new Error('The caller required a tool call but supplied no tool definitions.');
    }

    const toolProtocolEnabled = tools.length > 0 && toolChoice !== 'none';

    // Some instruct models describe a tool call in prose and never populate the
    // native tool_calls channel. Once a model has demonstrated that, its native
    // generation is discarded every turn, so stop paying for it.
    if (toolProtocolEnabled && this.nativeToolCalls === 'unavailable') {
      return this.schemaConstrainedDecision(request, tools, toolChoice, 0, onEvent, trace, signal);
    }

    const nativeEvents: ChatStreamEvent[] = [];
    const nativeResult = await this.streamNativeChat(
      request,
      (event) => nativeEvents.push(event),
      trace,
      'native',
      signal,
    );
    const nativeAutomaticFinal = toolChoice === 'auto' &&
      this.nativeToolCalls === 'available' &&
      nativeResult.toolCallCount === 0;
    if (!toolProtocolEnabled || nativeResult.toolCallCount > 0 || nativeAutomaticFinal) {
      if (toolProtocolEnabled && nativeResult.toolCallCount > 0) {
        this.nativeToolCalls = 'available';
      }
      for (const event of nativeEvents) {
        onEvent(event);
      }
      return nativeResult;
    }

    if (toolChoice === 'required') {
      this.nativeToolCalls = 'unavailable';
      this.diagnostics?.info(
        `${GENERATION_PHASE} This model returned no required native tool call; using schema-constrained decisions for this runtime fingerprint.`,
      );
    }
    return this.schemaConstrainedDecision(
      request,
      tools,
      toolChoice,
      nativeResult.textCharacters,
      onEvent,
      trace,
      signal,
    );
  }

  private async schemaConstrainedDecision(
    request: ChatRequest,
    tools: readonly ChatTool[],
    toolChoice: NonNullable<ChatRequest['toolChoice']>,
    textCharacters: number,
    onEvent: (event: ChatStreamEvent) => void,
    trace: ChatTrace,
    signal?: AbortSignal,
  ): Promise<StreamedChatResult> {
    const fallback = await this.schemaConstrainedFallback(
      request,
      tools,
      toolChoice === 'required',
      trace,
      signal,
    );
    if (fallback.decision.kind === 'tool') {
      onEvent({
        kind: 'toolCall',
        id: `call-${randomUUID()}`,
        name: fallback.decision.name,
        input: fallback.decision.arguments,
      });
      return {
        inputTokens: fallback.inputTokens,
        textCharacters,
        toolCallCount: 1,
      };
    }
    return this.streamNativeChat(
      { ...request, toolChoice: 'none', workerToolChoice: 'none' },
      onEvent,
      trace,
      'final',
      signal,
    );
  }

  private async streamNativeChat(
    request: ChatRequest,
    onEvent: (event: ChatStreamEvent) => void,
    trace: ChatTrace,
    stage: string,
    signal?: AbortSignal,
  ): Promise<StreamedChatResult> {
    const tools = request.tools ?? [];
    const toolChoice = request.toolChoice ?? 'auto';
    const finalOnly = toolChoice === 'none';
    const messages = finalOnly && tools.length > 0
      ? finalResponseMessages(request.messages)
      : request.messages;
    if (finalOnly && tools.length > 0) {
      this.diagnostics?.info(
        `${GENERATION_PHASE} ${trace.id} final answer: tool execution disabled; preserving ${tools.length} tool definitions and the conversation prefix.`,
      );
    }
    const outputTokenLimit = toolChoice === 'required'
      ? Math.min(
        request.maxTokens,
        Math.max(1, Math.floor(request.toolCallMaxTokens ?? request.maxTokens)),
      )
      : request.maxTokens;
    if (toolChoice === 'required' && outputTokenLimit < request.maxTokens) {
      this.diagnostics?.info(
        `${GENERATION_PHASE} Required tool generation output limit: ${outputTokenLimit} tokens (chat limit ${request.maxTokens}).`,
      );
    }
    const body: Record<string, unknown> = {
      model: 'local',
      messages,
      stream: true,
      max_tokens: outputTokenLimit,
      temperature: request.temperature,
    };
    const workerToolChoice = finalOnly ? 'none' : request.workerToolChoice ?? toolChoice;
    const measured = await assertPromptFits(
      tools,
      request.inputTokenBudget,
      (candidateTools) => this.countChatBodyInputTokens(
        withTools(body, candidateTools, workerToolChoice),
        signal,
      ),
    );
    Object.assign(body, withTools({}, measured.tools, workerToolChoice));
    this.diagnostics?.info(
      `${GENERATION_PHASE} ${trace.id} prompt budget (${stage}): ${measured.inputTokens}/${request.inputTokenBudget} ` +
      `input tokens with ${tools.length} tool${tools.length === 1 ? '' : 's'}.`,
    );

    const response = await this.request('/v1/chat/completions', {
      method: 'POST',
      body: JSON.stringify(body),
      ...requestSignal(signal),
    });
    this.diagnostics?.info(
      `${GENERATION_PHASE} ${trace.id} response headers (${stage}): elapsed=${Date.now() - trace.startedAt} ms.`,
    );
    if (!response.body) {
      throw new Error('The local worker returned an empty streaming response.');
    }

    const pendingTools = new Map<number, PendingToolCall>();
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    const bufferText = tools.length > 0 || finalOnly;
    let buffer = '';
    let bufferedText = '';
    let textCharacters = 0;
    let reasoningCharacters = 0;
    let finishReason: string | undefined;
    let tokensPerSecond: number | undefined;
    let firstStreamData = true;
    while (true) {
      const { done, value } = await reader.read();
      if (firstStreamData && value && value.byteLength > 0) {
        firstStreamData = false;
        this.diagnostics?.info(
          `${GENERATION_PHASE} ${trace.id} first stream data (${stage}): elapsed=${Date.now() - trace.startedAt} ms.`,
        );
      }
      buffer += decoder.decode(value, { stream: !done });
      const frames = buffer.split(/\r?\n\r?\n/);
      buffer = frames.pop() ?? '';
      for (const frame of frames) {
        const consumed = this.consumeSseFrame(frame, pendingTools);
        if (consumed.timings) {
          this.logTimings(trace, stage, consumed.timings);
          tokensPerSecond = finiteNumber(consumed.timings.predicted_per_second)
            ?? tokensPerSecond;
        }
        const text = consumed.text;
        textCharacters += text.length;
        reasoningCharacters += consumed.reasoning.length;
        finishReason = consumed.finishReason ?? finishReason;
        if (bufferText) {
          bufferedText += text;
        } else if (text) {
          onEvent({ kind: 'text', text });
        }
      }
      if (done) {
        break;
      }
    }
    if (buffer.trim()) {
      const consumed = this.consumeSseFrame(buffer, pendingTools);
      if (consumed.timings) {
        this.logTimings(trace, stage, consumed.timings);
        tokensPerSecond = finiteNumber(consumed.timings.predicted_per_second)
          ?? tokensPerSecond;
      }
      const text = consumed.text;
      textCharacters += text.length;
      reasoningCharacters += consumed.reasoning.length;
      finishReason = consumed.finishReason ?? finishReason;
      if (bufferText) {
        bufferedText += text;
      } else if (text) {
        onEvent({ kind: 'text', text });
      }
    }

    let toolCallCount = 0;
    if (finalOnly) {
      assertFinalResponse(bufferedText, pendingTools.size > 0);
    }
    if (pendingTools.size > 0 && bufferedText) {
      onEvent({ kind: 'text', text: bufferedText });
    }
    for (const tool of pendingTools.values()) {
      if (!tool.name) {
        throw new Error('The local model returned a tool call without a function name.');
      }
      let input: unknown;
      try {
        input = JSON.parse(tool.arguments || '{}') as unknown;
      } catch {
        throw new Error(`Local model produced invalid JSON for tool ${tool.name}.`);
      }
      validateToolCallInput(tool.name, input, tools);
      onEvent({ kind: 'toolCall', id: tool.id, name: tool.name, input });
      toolCallCount += 1;
    }
    if (toolCallCount === 0 && bufferedText) {
      onEvent({ kind: 'text', text: bufferedText });
    }
    return {
      inputTokens: measured.inputTokens,
      textCharacters,
      toolCallCount,
      ...(tokensPerSecond !== undefined ? { tokensPerSecond } : {}),
      reasoningCharacters,
      ...(finishReason !== undefined ? { finishReason } : {}),
      outputTokenLimit,
    };
  }

  private async schemaConstrainedFallback(
    request: ChatRequest,
    tools: readonly ChatTool[],
    toolRequired: boolean,
    trace: ChatTrace,
    signal?: AbortSignal,
  ): Promise<{ decision: ToolDecision; inputTokens: number }> {
    const maxTokens = Math.min(
      request.maxTokens,
      Math.max(1, Math.floor(request.toolCallMaxTokens ?? request.maxTokens)),
    );
    const messages: ChatMessage[] = [
      ...request.messages,
      {
        role: 'user',
        content: toolRequired
          ? 'Return exactly one tool decision matching the response schema. Choose the supplied tool needed for the current task.'
          : 'Return one action matching the response schema. Choose kind tool when another tool is needed. Otherwise choose kind final.',
      },
    ];
    const body: Record<string, unknown> = {
      model: 'local',
      messages,
      stream: true,
      max_tokens: maxTokens,
      temperature: 0,
      response_format: toolDecisionResponseFormat(tools, toolRequired),
    };
    const inputTokens = await this.countChatBodyInputTokens(body, signal);
    if (inputTokens > request.inputTokenBudget) {
      throw new Error(
        `The schema-constrained fallback requires ${inputTokens} input tokens, but the local model input budget is ${request.inputTokenBudget}.`,
      );
    }
    this.diagnostics?.info(
      `${GENERATION_PHASE} Native tool output was unavailable; using one schema-constrained ${toolRequired ? 'required' : 'automatic'} decision.`,
    );
    const response = await this.request('/v1/chat/completions', {
      method: 'POST',
      body: JSON.stringify(body),
      ...requestSignal(signal),
    });
    this.diagnostics?.info(
      `${GENERATION_PHASE} ${trace.id} response headers (schema decision): elapsed=${Date.now() - trace.startedAt} ms.`,
    );
    const content = await this.readStreamedContent(
      response,
      '/v1/chat/completions',
      trace,
      'schema decision',
    );
    if (!content.trim()) {
      throw new Error(
        'The schema-constrained fallback returned no decision. The worker streamed an empty response.',
      );
    }
    return {
      decision: parseToolDecision(content, tools, toolRequired),
      inputTokens,
    };
  }

  async infill(request: InfillRequest, signal?: AbortSignal): Promise<string> {
    const body: Record<string, unknown> = {
      input_prefix: request.prefix,
      input_suffix: request.suffix,
      n_predict: request.maxTokens,
      temperature: request.temperature,
      stream: false,
    };
    if (request.stop?.length) {
      body.stop = request.stop;
    }
    const response = await this.request('/infill', {
      method: 'POST',
      body: JSON.stringify(body),
      ...requestSignal(signal),
    });
    const payload = (await response.json()) as { content?: unknown };
    return typeof payload.content === 'string' ? payload.content : '';
  }

  private async countChatBodyInputTokens(
    body: Record<string, unknown>,
    signal?: AbortSignal,
  ): Promise<number> {
    const response = await this.request('/v1/chat/completions/input_tokens', {
      method: 'POST',
      body: JSON.stringify(body),
      ...requestSignal(signal),
    });
    const payload = (await response.json()) as { input_tokens?: unknown };
    if (typeof payload.input_tokens !== 'number' || !Number.isFinite(payload.input_tokens)) {
      throw new Error('The local worker returned an invalid chat token count.');
    }
    return payload.input_tokens;
  }

  private consumeSseFrame(
    frame: string,
    pendingTools: Map<number, PendingToolCall>,
  ): ConsumedSseFrame {
    const data = frame
      .split(/\r?\n/)
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).trim())
      .join('');
    if (!data || data === '[DONE]') {
      return { text: '', reasoning: '' };
    }

    let chunk: OpenAiChunk;
    try {
      chunk = JSON.parse(data) as OpenAiChunk;
    } catch {
      return { text: '', reasoning: '' };
    }
    if (chunk.error !== undefined) {
      throw workerStreamError('/v1/chat/completions', chunk.error);
    }
    const choice = chunk.choices?.[0];
    const delta = choice?.delta;
    const toolCalls = delta?.tool_calls ?? [];
    for (const part of toolCalls) {
      const existing = pendingTools.get(part.index) ?? {
        id: part.id ?? `call-${part.index}`,
        name: '',
        arguments: '',
      };
      if (part.id) {
        existing.id = part.id;
      }
      if (part.function?.name) {
        existing.name += part.function.name;
      }
      if (part.function?.arguments) {
        existing.arguments += part.function.arguments;
      }
      pendingTools.set(part.index, existing);
    }
    const text = delta?.content ?? '';
    return {
      text,
      reasoning: delta?.reasoning_content ?? '',
      ...(choice?.finish_reason ? { finishReason: choice.finish_reason } : {}),
      ...(chunk.timings ? { timings: chunk.timings } : {}),
    };
  }

  private logTimings(
    trace: ChatTrace,
    stage: string,
    timings: NonNullable<OpenAiChunk['timings']>,
  ): void {
    const promptTokens = finiteNumber(timings.prompt_n);
    const cachedTokens = finiteNumber(timings.cache_n);
    const promptMilliseconds = finiteNumber(timings.prompt_ms);
    const promptPerSecond = finiteNumber(timings.prompt_per_second);
    const predictedTokens = finiteNumber(timings.predicted_n);
    const predictedMilliseconds = finiteNumber(timings.predicted_ms);
    const predictedPerSecond = finiteNumber(timings.predicted_per_second);
    if (
      promptTokens === undefined &&
      cachedTokens === undefined &&
      predictedTokens === undefined
    ) {
      return;
    }
    this.diagnostics?.info(
      `${GENERATION_PHASE} ${trace.id} timings (${stage}): ` +
      `prompt=${promptTokens ?? 'unknown'} processed + ${cachedTokens ?? 0} cached tokens` +
      formatDurationAndRate(promptMilliseconds, promptPerSecond) +
      `; output=${predictedTokens ?? 'unknown'} ${predictedTokens === 1 ? 'token' : 'tokens'}` +
      formatDurationAndRate(predictedMilliseconds, predictedPerSecond) + '.',
    );
  }

  private async request(path: string, init: WorkerRequestInit): Promise<WorkerResponse> {
    let response: WorkerResponse;
    try {
      response = await pooledFetch(`${this.baseUrl}${path}`, {
        ...init,
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json',
        },
        dispatcher: this.pool,
      });
    } catch (error) {
      // A caller-driven abort is expected control flow, not a fault to describe.
      if (init.signal?.aborted) {
        throw error;
      }
      throw workerTransportError(path, error);
    }
    if (!response.ok) {
      throw workerRequestError(path, response.status, await response.text());
    }
    return response;
  }

  /**
   * Concatenates the text of a streamed completion.
   *
   * Every generating request streams. A non-streaming request sends no response
   * headers until the whole answer exists, and slow local generation outlives the
   * 300 second headers timeout built into Node's fetch.
   */
  private async readStreamedContent(
    response: WorkerResponse,
    path: string,
    trace: ChatTrace,
    stage: string,
  ): Promise<string> {
    if (!response.body) {
      throw new Error(`Local worker request ${path} returned an empty streaming response.`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let content = '';
    let firstStreamData = true;
    const pendingTools = new Map<number, PendingToolCall>();
    const consume = (frame: string): void => {
      const consumed = this.consumeSseFrame(frame, pendingTools);
      if (consumed.timings) {
        this.logTimings(trace, stage, consumed.timings);
      }
      content += consumed.text;
    };
    while (true) {
      const { done, value } = await reader.read();
      if (firstStreamData && value && value.byteLength > 0) {
        firstStreamData = false;
        this.diagnostics?.info(
          `${GENERATION_PHASE} ${trace.id} first stream data (${stage}): elapsed=${Date.now() - trace.startedAt} ms.`,
        );
      }
      buffer += decoder.decode(value, { stream: !done });
      const frames = buffer.split(/\r?\n\r?\n/);
      buffer = frames.pop() ?? '';
      for (const frame of frames) {
        consume(frame);
      }
      if (done) {
        break;
      }
    }
    if (buffer.trim()) {
      consume(buffer);
    }
    return content;
  }

}

/**
 * A thinking template streams its reasoning on `reasoning_content`, which is
 * not the answer. A reply that ends with no answer and no tool call would
 * otherwise leave VS Code Chat blank with nothing to explain why.
 */
function assertAnswered(
  result: ChatResult,
  reasoningCharacters: number,
  finishReason: string | undefined,
  outputTokenLimit: number,
): void {
  if (result.textCharacters > 0 || result.toolCallCount > 0) {
    return;
  }
  if (reasoningCharacters > 0 && finishReason === 'length') {
    throw new Error(
      `The model spent its whole output limit (${outputTokenLimit} tokens) reasoning and never started its answer. ` +
      'Raise localLlm.maxOutputTokens, or use a model or chat template that thinks less.',
    );
  }
  throw new Error(
    reasoningCharacters > 0
      ? 'The model finished reasoning but returned an empty response.'
      : `The model returned an empty response (finish reason: ${finishReason ?? 'unknown'}).`,
  );
}

function finiteNumber(value: number | undefined): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function formatDurationAndRate(
  milliseconds: number | undefined,
  perSecond: number | undefined,
): string {
  const duration = milliseconds === undefined ? '' : ` in ${milliseconds.toFixed(2)} ms`;
  const rate = perSecond === undefined ? '' : ` (${perSecond.toFixed(2)} tokens/s)`;
  return `${duration}${rate}`;
}

function requestSignal(signal?: AbortSignal): Pick<WorkerRequestInit, 'signal'> {
  return signal ? { signal } : {};
}

function assertLoopbackWorkerUrl(baseUrl: string): void {
  let parsed: URL;
  try {
    parsed = new URL(baseUrl);
  } catch {
    throw new Error('The local worker address is invalid.');
  }
  if (parsed.protocol !== 'http:' || parsed.hostname !== '127.0.0.1') {
    throw new Error('The local worker client accepts only HTTP loopback addresses on 127.0.0.1.');
  }
}

function withTools(
  body: Record<string, unknown>,
  tools: ChatRequest['tools'],
  toolChoice: NonNullable<ChatRequest['toolChoice']>,
): Record<string, unknown> {
  if (!tools?.length) {
    return { ...body, ...(toolChoice === 'none' ? { tool_choice: 'none' } : {}) };
  }
  return {
    ...body,
    tools,
    tool_choice: toolChoice,
    parallel_tool_calls: false,
  };
}
