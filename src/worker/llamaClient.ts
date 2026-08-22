import { randomUUID } from 'node:crypto';
import { fetch as pooledFetch, Pool } from 'undici';
import type {
  ChatMessage,
  ChatRequest,
  ChatStreamEvent,
  ChatTool,
  InfillRequest,
} from '../domain';
import { parseWorkerModelProfile, type WorkerModelProfile } from './runtimeProfile';
import { assertPromptFits } from './toolBudget';
import {
  parseToolDecision,
  toolDecisionResponseFormat,
  type ToolDecision,
  validateToolCallInput,
} from './toolProtocol';
import { workerRequestError, workerStreamError, workerTransportError } from './workerError';

type WorkerResponse = Awaited<ReturnType<typeof pooledFetch>>;

interface WorkerRequestInit {
  method: 'GET' | 'POST';
  body?: string;
  signal?: AbortSignal;
}

interface OpenAiChunk {
  error?: unknown;
  choices?: Array<{
    delta?: {
      content?: string;
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

const TOKEN_COUNT_CACHE_MAX_ENTRIES = 4_096;
const TOKEN_COUNT_CACHE_MAX_TEXT_LENGTH = 32 * 1_024;

export interface ChatResult {
  inputTokens: number;
  textCharacters: number;
  toolCallCount: number;
}

/**
 * Whether the loaded model emits tool calls on llama.cpp's native tool_calls
 * channel. Measured once per worker process, then reused.
 */
export type NativeToolCallSupport = 'unknown' | 'available' | 'unavailable';

export class LlamaClient {
  private modelProfile: WorkerModelProfile | undefined;
  private nativeToolCalls: NativeToolCallSupport = 'unknown';
  private readonly pool: Pool;
  private readonly tokenCounts = new Map<string, number>();

  constructor(
    readonly baseUrl: string,
    private readonly apiKey: string,
    private readonly log?: (message: string) => void,
  ) {
    assertLoopbackWorkerUrl(baseUrl);
    this.pool = new Pool(baseUrl, {
      connections: 1,
      pipelining: 1,
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
    if (toolChoice === 'required' && tools.length === 0) {
      throw new Error('The caller required a tool call but supplied no tool definitions.');
    }

    const toolProtocolEnabled = tools.length > 0 && toolChoice !== 'none';

    // Some instruct models describe a tool call in prose and never populate the
    // native tool_calls channel. Once a model has demonstrated that, its native
    // generation is discarded every turn, so stop paying for it.
    if (toolProtocolEnabled && this.nativeToolCalls === 'unavailable') {
      return this.schemaConstrainedDecision(request, tools, toolChoice, 0, onEvent, signal);
    }

    const nativeEvents: ChatStreamEvent[] = [];
    const nativeResult = await this.streamNativeChat(
      request,
      (event) => nativeEvents.push(event),
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
      this.log?.(
        'This model returned no required native tool call; using schema-constrained decisions for this runtime fingerprint.',
      );
    }
    return this.schemaConstrainedDecision(
      request,
      tools,
      toolChoice,
      nativeResult.textCharacters,
      onEvent,
      signal,
    );
  }

  private async schemaConstrainedDecision(
    request: ChatRequest,
    tools: readonly ChatTool[],
    toolChoice: NonNullable<ChatRequest['toolChoice']>,
    textCharacters: number,
    onEvent: (event: ChatStreamEvent) => void,
    signal?: AbortSignal,
  ): Promise<ChatResult> {
    const fallback = await this.schemaConstrainedFallback(
      request,
      tools,
      toolChoice === 'required',
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
    const {
      tools: _tools,
      workerToolChoice: _workerToolChoice,
      ...withoutTools
    } = request;
    return this.streamNativeChat(
      { ...withoutTools, toolChoice: 'none' },
      onEvent,
      signal,
    );
  }

  private async streamNativeChat(
    request: ChatRequest,
    onEvent: (event: ChatStreamEvent) => void,
    signal?: AbortSignal,
  ): Promise<ChatResult> {
    const tools = request.tools ?? [];
    const toolChoice = request.toolChoice ?? 'auto';
    const outputTokenLimit = toolChoice === 'required'
      ? Math.min(
        request.maxTokens,
        Math.max(1, Math.floor(request.toolCallMaxTokens ?? request.maxTokens)),
      )
      : request.maxTokens;
    if (toolChoice === 'required' && outputTokenLimit < request.maxTokens) {
      this.log?.(
        `Required tool generation output limit: ${outputTokenLimit} tokens (chat limit ${request.maxTokens}).`,
      );
    }
    const body: Record<string, unknown> = {
      model: 'local',
      messages: request.messages,
      stream: true,
      max_tokens: outputTokenLimit,
      temperature: request.temperature,
    };
    const workerToolChoice = request.workerToolChoice ?? toolChoice;
    const measured = await assertPromptFits(
      tools,
      request.inputTokenBudget,
      (candidateTools) => this.countChatBodyInputTokens(
        withTools(body, candidateTools, workerToolChoice),
        signal,
      ),
    );
    Object.assign(body, withTools({}, measured.tools, workerToolChoice));
    this.log?.(
      `Prompt budget: ${measured.inputTokens}/${request.inputTokenBudget} input tokens with ${tools.length} tool${tools.length === 1 ? '' : 's'}.`,
    );

    const response = await this.request('/v1/chat/completions', {
      method: 'POST',
      body: JSON.stringify(body),
      ...requestSignal(signal),
    });
    if (!response.body) {
      throw new Error('The local worker returned an empty streaming response.');
    }

    const pendingTools = new Map<number, PendingToolCall>();
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    const bufferText = tools.length > 0 && toolChoice !== 'none';
    let buffer = '';
    let bufferedText = '';
    let textCharacters = 0;
    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      const frames = buffer.split(/\r?\n\r?\n/);
      buffer = frames.pop() ?? '';
      for (const frame of frames) {
        const text = this.consumeSseFrame(frame, pendingTools);
        textCharacters += text.length;
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
      const text = this.consumeSseFrame(buffer, pendingTools);
      textCharacters += text.length;
      if (bufferText) {
        bufferedText += text;
      } else if (text) {
        onEvent({ kind: 'text', text });
      }
    }

    let toolCallCount = 0;
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
    return { inputTokens: measured.inputTokens, textCharacters, toolCallCount };
  }

  private async schemaConstrainedFallback(
    request: ChatRequest,
    tools: readonly ChatTool[],
    toolRequired: boolean,
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
    this.log?.(
      `Native tool output was unavailable; using one schema-constrained ${toolRequired ? 'required' : 'automatic'} decision.`,
    );
    const response = await this.request('/v1/chat/completions', {
      method: 'POST',
      body: JSON.stringify(body),
      ...requestSignal(signal),
    });
    const content = await this.readStreamedContent(response, '/v1/chat/completions');
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
  ): string {
    const data = frame
      .split(/\r?\n/)
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).trim())
      .join('');
    if (!data || data === '[DONE]') {
      return '';
    }

    let chunk: OpenAiChunk;
    try {
      chunk = JSON.parse(data) as OpenAiChunk;
    } catch {
      return '';
    }
    if (chunk.error !== undefined) {
      throw workerStreamError('/v1/chat/completions', chunk.error);
    }
    const delta = chunk.choices?.[0]?.delta;
    for (const part of delta?.tool_calls ?? []) {
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
    return delta?.content ?? '';
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
  private async readStreamedContent(response: WorkerResponse, path: string): Promise<string> {
    if (!response.body) {
      throw new Error(`Local worker request ${path} returned an empty streaming response.`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let content = '';
    const consume = (frame: string): void => {
      const data = frame
        .split(/\r?\n/)
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice(5).trim())
        .join('');
      if (!data || data === '[DONE]') {
        return;
      }
      let chunk: OpenAiChunk;
      try {
        chunk = JSON.parse(data) as OpenAiChunk;
      } catch {
        return;
      }
      if (chunk.error !== undefined) {
        throw workerStreamError(path, chunk.error);
      }
      content += chunk.choices?.[0]?.delta?.content ?? '';
    };
    while (true) {
      const { done, value } = await reader.read();
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
    return { ...body };
  }
  return {
    ...body,
    tools,
    tool_choice: toolChoice,
    parallel_tool_calls: false,
  };
}
