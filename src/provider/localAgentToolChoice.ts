import type { ChatMessage } from '../domain';

export const LOCAL_AGENT_PROTOCOL_MARKER = 'LOCAL_LLM_WORKSPACE_AGENT_PROTOCOL_9E218F31_V1';

export function shouldRequireLocalAgentTool(messages: readonly ChatMessage[]): boolean {
  if (!isLocalAgentRequest(messages)) {
    return false;
  }

  const lastAssistantIndex = messages.findLastIndex(
    (message) => message.role === 'assistant',
  );

  // First Local Agent request in a conversation has no assistant history yet.
  // In that case, require a tool on the first user turn so the task can
  // start with workspace context discovery.
  if (lastAssistantIndex < 0) {
    return true;
  }

  const lastAssistant = messages[lastAssistantIndex];
  const callIds = new Set(lastAssistant?.tool_calls?.map((call) => call.id) ?? []);
  if (callIds.size === 0) {
    return false;
  }

  const resultIndex = messages.findLastIndex(
    (message, index) =>
      index > lastAssistantIndex &&
      message.role === 'tool' &&
      Boolean(message.tool_call_id && callIds.has(message.tool_call_id)),
  );
  if (resultIndex < 0) {
    return true;
  }

  return false;
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
