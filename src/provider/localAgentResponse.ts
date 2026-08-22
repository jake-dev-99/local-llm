import type { ChatMessage, ChatRequest, ChatStreamEvent } from '../domain';
import type { ChatResult } from '../worker/llamaClient';
import {
  activeLocalAgentRequestIndex,
  evaluateLocalAgentTurn,
  isExactNormalizedResponse,
  isRepeatedLocalAgentToolCall,
  LOCAL_AGENT_DISCOVERY_TOOL_NAMES,
} from './localAgentToolChoice.ts';

export interface LocalAgentGeneration {
  events: ChatStreamEvent[];
  result: ChatResult;
}

export interface LocalAgentResponse {
  events: ChatStreamEvent[];
  result: ChatResult;
  observedToolCall: boolean;
}

interface RunLocalAgentResponseOptions {
  messages: readonly ChatMessage[];
  request: ChatRequest;
  previousFinal?: string;
  maxToolInvocations: number;
  generate(request: ChatRequest): Promise<LocalAgentGeneration>;
  log?(message: string): void;
}

export async function runLocalAgentResponse(
  options: RunLocalAgentResponseOptions,
): Promise<LocalAgentResponse> {
  let generation = await options.generate(options.request);
  let observedToolCall = generation.result.toolCallCount > 0;
  const repeatedCall = repeatedToolCall(options.messages, generation.events);
  if (repeatedCall) {
    options.log?.('Local Agent blocked a repeated tool call and requested one novel decision.');
    const state = evaluateLocalAgentTurn(options.messages, options.maxToolInvocations);
    const correction = state.invocationCount + 1 >= options.maxToolInvocations
      ? forceFinalRequest(options.request, duplicateCallFinalInstruction)
      : correctedRequest(options.request, duplicateCallCorrectionInstruction);
    generation = await options.generate(correction);
    observedToolCall ||= generation.result.toolCallCount > 0;

    if (repeatedToolCall(options.messages, generation.events)) {
      options.log?.('Local Agent received a second repeated tool call and forced final generation.');
      generation = await options.generate(
        forceFinalRequest(options.request, duplicateCallFinalInstruction),
      );
      observedToolCall ||= generation.result.toolCallCount > 0;
    }
  }

  if (
    options.previousFinal &&
    generation.result.toolCallCount === 0 &&
    isExactNormalizedResponse(options.previousFinal, responseText(generation.events))
  ) {
    options.log?.('Local Agent rejected an exact repeated final response and requested one revision.');
    const revision = await options.generate(
      forceFinalRequest(options.request, duplicateResponseRevisionInstruction),
    );
    observedToolCall ||= revision.result.toolCallCount > 0;
    if (isExactNormalizedResponse(options.previousFinal, responseText(revision.events))) {
      options.log?.('Local Agent revision repeated exactly; returning an explicit quality failure.');
      return {
        events: [{ kind: 'text', text: qualityFailureMessage(options.messages) }],
        result: revision.result,
        observedToolCall,
      };
    }
    generation = revision;
  }

  return { ...generation, observedToolCall };
}

function repeatedToolCall(
  messages: readonly ChatMessage[],
  events: readonly ChatStreamEvent[],
): boolean {
  return events.some(
    (event) => event.kind === 'toolCall' &&
      isRepeatedLocalAgentToolCall(messages, event.name, event.input),
  );
}

function correctedRequest(request: ChatRequest, instruction: string): ChatRequest {
  return {
    ...request,
    messages: [...request.messages, { role: 'user', content: instruction }],
  };
}

function forceFinalRequest(request: ChatRequest, instruction: string): ChatRequest {
  const {
    tools: _tools,
    workerToolChoice: _workerToolChoice,
    ...withoutTools
  } = correctedRequest(request, instruction);
  return {
    ...withoutTools,
    toolChoice: 'none',
    workerToolChoice: 'none',
  };
}

function responseText(events: readonly ChatStreamEvent[]): string {
  return events
    .filter((event): event is Extract<ChatStreamEvent, { kind: 'text' }> => event.kind === 'text')
    .map((event) => event.text)
    .join('');
}

function qualityFailureMessage(messages: readonly ChatMessage[]): string {
  const evidence = evidenceToolNames(messages);
  return 'The local model repeated its previous answer after a fresh evidence pass. ' +
    'I cannot verify an independent result. ' +
    `Evidence collected: ${evidence.length ? evidence.join(', ') : 'none'}.`;
}

function evidenceToolNames(messages: readonly ChatMessage[]): string[] {
  const requestIndex = activeLocalAgentRequestIndex(messages);
  const calls = new Map<string, string>();
  const names = new Set<string>();
  for (let index = requestIndex + 1; index < messages.length; index += 1) {
    const message = messages[index];
    if (message?.role === 'assistant') {
      for (const call of message.tool_calls ?? []) {
        calls.set(call.id, call.function.name);
      }
    } else if (message?.role === 'tool' && message.tool_call_id) {
      const name = calls.get(message.tool_call_id);
      if (name && LOCAL_AGENT_DISCOVERY_TOOL_NAMES.has(name)) {
        names.add(name);
      }
    }
  }
  return [...names];
}

const duplicateCallCorrectionInstruction =
  'The proposed tool call exactly repeats completed work. Choose a different useful tool call, or answer from current evidence.';

const duplicateCallFinalInstruction =
  'Repeated tool work was blocked. Answer from collected evidence now. Clearly name unfinished work and uncertainty.';

const duplicateResponseRevisionInstruction =
  'Your proposed answer exactly matched the earlier answer. Re-evaluate the fresh tool evidence and answer independently. Preserve supported conclusions, but explain them from current evidence.';
