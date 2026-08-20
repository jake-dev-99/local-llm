import { randomUUID } from 'node:crypto';
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
import { workerRequestError, workerStreamError } from './workerError';

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

export interface ChatResult {
  inputTokens: number;
  textCharacters: number;
  toolCallCount: number;
}

/**
 * Whether the loaded model emits tool calls on llama.cpp's native tool_calls
 * channel. Measured once per worker process, then reused.
 */
type NativeToolCallSupport = 'unknown' | 'available' | 'unavailable';

export class LlamaClient {
  private modelProfile: WorkerModelProfile | undefined;
  private nativeToolCalls: NativeToolCallSupport = 'unknown';

  constructor(
    readonly baseUrl: string,
    private readonly apiKey: string,
    private readonly log?: (message: string) => void,
  ) {}

  async health(signal?: AbortSignal): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/health`, requestSignal(signal));
      return response.ok;
    } catch (error) {
      if (signal?.aborted) {
        throw error;
      }
      return false;
    }
  }

  async tokenize(content: string, signal?: AbortSignal): Promise<number> {
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
    return Array.isArray(payload.tokens) ? payload.tokens.length : 0;
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
    if (!toolProtocolEnabled || nativeResult.toolCallCount > 0) {
      if (toolProtocolEnabled) {
        this.nativeToolCalls = 'available';
      }
      for (const event of nativeEvents) {
        onEvent(event);
      }
      return nativeResult;
    }

    this.nativeToolCalls = 'unavailable';
    this.log?.(
      'This model returned no native tool call; using schema-constrained decisions for the rest of this worker session.',
    );
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
    onEvent({ kind: 'text', text: fallback.decision.text });
    return {
      inputTokens: fallback.inputTokens,
      textCharacters: textCharacters + fallback.decision.text.length,
      toolCallCount: 0,
    };
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
    const maxTokens = toolRequired
      ? Math.min(
        request.maxTokens,
        Math.max(1, Math.floor(request.toolCallMaxTokens ?? request.maxTokens)),
      )
      : request.maxTokens;
    const messages: ChatMessage[] = [
      ...request.messages,
      {
        role: 'user',
        content: toolRequired
          ? 'Return exactly one tool decision matching the response schema. Choose the supplied tool needed for the current task.'
          : 'Return one decision matching the response schema. Choose kind tool when another tool is needed. Otherwise choose kind final with the final answer.',
      },
    ];
    const body: Record<string, unknown> = {
      model: 'local',
      messages,
      stream: false,
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
    const payload = (await response.json()) as {
      choices?: Array<{ message?: { content?: unknown } }>;
    };
    const content = payload.choices?.[0]?.message?.content;
    if (typeof content !== 'string' || !content.trim()) {
      throw new Error('The schema-constrained fallback returned no decision.');
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

  private async request(path: string, init: RequestInit): Promise<Response> {
    const headers = new Headers(init.headers);
    headers.set('Authorization', `Bearer ${this.apiKey}`);
    headers.set('Content-Type', 'application/json');
    const response = await fetch(`${this.baseUrl}${path}`, { ...init, headers });
    if (!response.ok) {
      throw workerRequestError(path, response.status, await response.text());
    }
    return response;
  }
}

function requestSignal(signal?: AbortSignal): RequestInit {
  return signal ? { signal } : {};
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
