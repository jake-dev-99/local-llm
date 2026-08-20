import type { ChatMessage } from '../domain';

export const LOCAL_AGENT_PROTOCOL_MARKER = 'LOCAL_LLM_WORKSPACE_AGENT_PROTOCOL_9E218F31_V1';

export const DEFAULT_MAX_AGENT_TOOL_ROUNDS = 8;

/**
 * Completed tool rounds belonging to the request currently being worked on.
 *
 * A round is one assistant tool call whose result has come back. Counting restarts
 * after the last assistant message that answered in plain text, so a follow-up
 * question gets a fresh budget rather than inheriting a spent one.
 */
export function completedToolRounds(messages: readonly ChatMessage[]): number {
  const lastFinalIndex = messages.findLastIndex(
    (message) => message.role === 'assistant' && !message.tool_calls?.length,
  );
  const callIds = new Set<string>();
  let rounds = 0;
  for (let index = lastFinalIndex + 1; index < messages.length; index += 1) {
    const message = messages[index];
    if (message?.role === 'assistant') {
      for (const call of message.tool_calls ?? []) {
        callIds.add(call.id);
      }
    } else if (
      message?.role === 'tool' &&
      message.tool_call_id &&
      callIds.has(message.tool_call_id)
    ) {
      rounds += 1;
    }
  }
  return rounds;
}

/**
 * Whether this Local Agent turn must call a tool.
 *
 * A single completed tool round used to end the agent turn, so one fruitless
 * search finished the task and the model answered from the prompt alone. The
 * turn now stays forced until the round budget is spent.
 */
export function shouldRequireLocalAgentTool(
  messages: readonly ChatMessage[],
  maxToolRounds: number = DEFAULT_MAX_AGENT_TOOL_ROUNDS,
): boolean {
  if (!isLocalAgentRequest(messages)) {
    return false;
  }
  return completedToolRounds(messages) < Math.max(1, Math.floor(maxToolRounds));
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
