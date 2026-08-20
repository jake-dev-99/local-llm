import type { ChatMessage, ChatTool } from '../domain';
import {
  isLocalAgentRequest,
  shouldRequireLocalAgentTool,
} from './localAgentToolChoice.ts';

const LOCAL_AGENT_PROTOCOL_LINE =
  'Protocol marker: LOCAL_LLM_WORKSPACE_AGENT_PROTOCOL_9E218F31_V1.';

const LOCAL_AGENT_DISCOVERY_TOOL_NAMES = new Set([
  'file_search',
  'grep_search',
  'get_errors',
  'list_dir',
  'read_file',
]);

const LOCAL_AGENT_EDIT_TOOL_NAMES = new Set([
  'insert_edit_into_file',
  'replace_string_in_file',
]);

const MUTATION_VERB =
  '(?:fix|repair|correct|resolve|patch|edit|modify|update|rename|refactor|implement|create|add|remove|delete|replace|apply|write|change|set|enable|disable|make)';

const DIRECT_MUTATION = new RegExp(
  '(?:^|[.!?]\\s+)' +
  '(?:(?:please|now)\\s+)*' +
  '(?:(?:(?:can|could|would)\\s+you|' +
  'i\\s+(?:want|need)\\s+you\\s+to|' +
  'i\\s+would\\s+like\\s+you\\s+to|' +
  'go\\s+ahead\\s+(?:and|to))\\s+)?' +
  '(?:(?:please|now)\\s+)*' +
  `${MUTATION_VERB}\\b`,
  'i',
);

const EXPLICIT_MUTATION_SEQUENCE = new RegExp(
  `\\b(?:and|then)\\s+(?:please\\s+)?${MUTATION_VERB}\\b`,
  'i',
);

const MUTATION_PROHIBITIONS = [
  /\b(?:do not|don't|never)\s+(?:make|apply|perform|write|change|edit|modify|fix|update|rename|remove|delete|replace)/i,
  /\bwithout\s+(?:(?:making|applying|writing|changing|editing|modifying)\s+)?(?:any\s+)?(?:changes?|edits?|modifications?|writes?|files?)\b/i,
  /\b(?:review|read|analysis|audit)[ -]?only\b/i,
  /\bno\s+(?:workspace\s+)?(?:changes?|edits?|modifications?|writes?)\b/i,
];

export function localAgentDiscoveryTools(tools: readonly ChatTool[]): ChatTool[] {
  return tools.filter((tool) => LOCAL_AGENT_DISCOVERY_TOOL_NAMES.has(tool.function.name));
}

export function localAgentMutationTools(tools: readonly ChatTool[]): ChatTool[] {
  return tools.filter((tool) => LOCAL_AGENT_EDIT_TOOL_NAMES.has(tool.function.name));
}

export function localAgentAvailableTools(
  tools: readonly ChatTool[],
  messages: readonly ChatMessage[],
): ChatTool[] {
  const mutationRequested = requestAllowsMutation(messages);
  const completedTools = completedToolNames(messages);
  const allowEdits = mutationRequested && completedTools.has('read_file');
  if (mutationRequested && !allowEdits && completedTools.size > 0) {
    return tools.filter((tool) => tool.function.name === 'read_file');
  }
  return tools.filter((tool) =>
    LOCAL_AGENT_DISCOVERY_TOOL_NAMES.has(tool.function.name) ||
    (allowEdits && LOCAL_AGENT_EDIT_TOOL_NAMES.has(tool.function.name)),
  );
}

/**
 * Whether a Local Agent turn must produce a tool call.
 *
 * This deliberately ignores how many tools the host supplied. An empty tool list
 * is the case that most needs catching: without a tool the model answers from the
 * prompt alone and prints the call it wanted as ordinary text.
 */
export function requiresLocalAgentTool(
  messages: readonly ChatMessage[],
  maxToolRounds: number,
): boolean {
  if (!isLocalAgentRequest(messages)) {
    return false;
  }
  return shouldRequireLocalAgentTool(messages, maxToolRounds) ||
    localAgentNeedsReadForMutation(messages);
}

export function localAgentNeedsReadForMutation(messages: readonly ChatMessage[]): boolean {
  return requestAllowsMutation(messages) && !hasCompletedFileRead(messages);
}

export function localAgentNeedsMutationTool(messages: readonly ChatMessage[]): boolean {
  if (!requestAllowsMutation(messages)) {
    return false;
  }
  const completed = completedToolNames(messages);
  return completed.has('read_file') &&
    ![...LOCAL_AGENT_EDIT_TOOL_NAMES].some((name) => completed.has(name));
}

export interface LocalAgentToolPolicy {
  tools: ChatTool[];
  toolChoice: 'auto' | 'required';
  source:
    | 'caller-required'
    | 'caller-auto'
    | 'local-agent-discovery'
    | 'local-agent-mutation';
}

export function resolveLocalAgentToolPolicy(
  tools: readonly ChatTool[],
  callerRequired: boolean,
  localAgentNeedsInitialTool: boolean,
  localAgentNeedsMutation: boolean = false,
): LocalAgentToolPolicy {
  if (callerRequired) {
    return { tools: [...tools], toolChoice: 'required', source: 'caller-required' };
  }
  if (localAgentNeedsMutation) {
    const mutationTools = localAgentMutationTools(tools);
    if (mutationTools.length === 0) {
      const supplied = tools.map((tool) => tool.function.name).join(', ') || 'none';
      throw new Error(
        `The bundled Local Agent cannot complete the requested workspace change because it received no recognized editor. Supplied tools: ${supplied}.`,
      );
    }
    return {
      tools: mutationTools,
      toolChoice: 'required',
      source: 'local-agent-mutation',
    };
  }
  if (!localAgentNeedsInitialTool) {
    return { tools: [...tools], toolChoice: 'auto', source: 'caller-auto' };
  }
  const discoveryTools = localAgentDiscoveryTools(tools);
  if (discoveryTools.length === 0) {
    const supplied = tools.map((tool) => tool.function.name).join(', ') || 'none';
    throw new Error(
      `The bundled Local Agent did not receive a recognized read-only discovery tool. Supplied tools: ${supplied}. Check the VS Code/Copilot tool contract before using Agent mode.`,
    );
  }
  return {
    tools: discoveryTools,
    toolChoice: 'required',
    source: 'local-agent-discovery',
  };
}

function requestAllowsMutation(messages: readonly ChatMessage[]): boolean {
  const request = activeUserRequest(messages);
  if (!request || MUTATION_PROHIBITIONS.some((pattern) => pattern.test(request))) {
    return false;
  }
  return DIRECT_MUTATION.test(request) || EXPLICIT_MUTATION_SEQUENCE.test(request);
}

function hasCompletedFileRead(messages: readonly ChatMessage[]): boolean {
  return completedToolNames(messages).has('read_file');
}

function completedToolNames(messages: readonly ChatMessage[]): Set<string> {
  const requestIndex = activeUserRequestIndex(messages);
  const completed = new Set<string>();
  if (requestIndex < 0) {
    return completed;
  }
  const calls = new Map<string, string>();
  for (let index = requestIndex + 1; index < messages.length; index += 1) {
    const message = messages[index];
    if (message?.role === 'assistant') {
      for (const call of message.tool_calls ?? []) {
        calls.set(call.id, call.function.name);
      }
    } else if (
      message?.role === 'tool' &&
      message.tool_call_id
    ) {
      const name = calls.get(message.tool_call_id);
      if (name) {
        completed.add(name);
      }
    }
  }
  return completed;
}

function activeUserRequest(messages: readonly ChatMessage[]): string | undefined {
  const index = activeUserRequestIndex(messages);
  return index >= 0 ? messages[index]?.content : undefined;
}

function activeUserRequestIndex(messages: readonly ChatMessage[]): number {
  const lastFinalAssistantIndex = messages.findLastIndex(
    (message) => message.role === 'assistant' && !message.tool_calls?.length,
  );
  const firstToolCallIndex = messages.findIndex(
    (message, index) =>
      index > lastFinalAssistantIndex &&
      message.role === 'assistant' &&
      Boolean(message.tool_calls?.length),
  );
  if (firstToolCallIndex >= 0) {
    const laterUserIndex = messages.findLastIndex(
      (message, index) =>
        index > firstToolCallIndex &&
        message.role === 'user' &&
        !isProtocolInstruction(message.content) &&
        !isLikelyHookContext(message.content),
    );
    if (laterUserIndex >= 0) {
      return laterUserIndex;
    }
  }
  const searchEnd = firstToolCallIndex >= 0 ? firstToolCallIndex : messages.length;
  for (let index = searchEnd - 1; index > lastFinalAssistantIndex; index -= 1) {
    const message = messages[index];
    if (message?.role === 'user' && !isProtocolInstruction(message.content)) {
      return index;
    }
  }
  return -1;
}

function isLikelyHookContext(content: string): boolean {
  return /^\s*(?:additional\s+)?hook context\s*:/i.test(content);
}

function isProtocolInstruction(content: string): boolean {
  return content.split(/\r?\n/).some(
    (line) => line.trim() === LOCAL_AGENT_PROTOCOL_LINE,
  );
}
