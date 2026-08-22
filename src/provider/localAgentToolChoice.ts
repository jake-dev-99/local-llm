import type { ChatMessage } from '../domain';

export const LOCAL_AGENT_PROTOCOL_MARKER = 'LOCAL_LLM_WORKSPACE_AGENT_PROTOCOL_9E218F31_V1';

export const DEFAULT_MAX_AGENT_TOOL_ROUNDS = 8;

export function localAgentToolInvocationCount(
  messages: readonly ChatMessage[],
): number {
  if (!isLocalAgentRequest(messages)) {
    return 0;
  }
  const requestIndex = activeLocalAgentRequestIndex(messages);
  if (requestIndex < 0) {
    return 0;
  }
  let invocationCount = 0;
  for (let index = requestIndex + 1; index < messages.length; index += 1) {
    const message = messages[index];
    if (message?.role === 'assistant') {
      invocationCount += message.tool_calls?.length ?? 0;
    }
  }
  return invocationCount;
}

export function localAgentToolLimitReached(
  messages: readonly ChatMessage[],
  maxToolInvocations: number = DEFAULT_MAX_AGENT_TOOL_ROUNDS,
): boolean {
  const ceiling = Math.max(1, Math.floor(maxToolInvocations));
  return isLocalAgentRequest(messages) &&
    localAgentToolInvocationCount(messages) >= ceiling;
}

export function activeLocalAgentRequestIndex(messages: readonly ChatMessage[]): number {
  const lastFinalIndex = messages.findLastIndex(
    (message) => message.role === 'assistant' && !message.tool_calls?.length,
  );
  let requestIndex = -1;
  for (let index = lastFinalIndex + 1; index < messages.length; index += 1) {
    const message = messages[index];
    if (
      message?.role === 'user' &&
      !hasProtocolMarker(message.content) &&
      !/^\s*(?:additional\s+)?hook context\s*:/i.test(message.content)
    ) {
      requestIndex = index;
    }
  }
  return requestIndex;
}

export function isLocalAgentRequest(messages: readonly ChatMessage[]): boolean {
  let foundMarkerInstruction = false;
  for (const message of messages) {
    if (message.role === 'assistant' || message.role === 'tool') {
      return false;
    }
    if (hasProtocolMarker(message.content)) {
      foundMarkerInstruction = true;
      continue;
    }
    if (foundMarkerInstruction && message.role === 'user') {
      return true;
    }
  }
  return false;
}

function hasProtocolMarker(content: string): boolean {
  return content.split(/\r?\n/).some(
    (line) => line.trim() === `Protocol marker: ${LOCAL_AGENT_PROTOCOL_MARKER}.`,
  );
}
