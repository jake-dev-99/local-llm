import type { ChatMessage } from '../domain';

export const LOCAL_AGENT_PROTOCOL_MARKER = 'LOCAL_LLM_WORKSPACE_AGENT_PROTOCOL_9E218F31_V1';

export const DEFAULT_MAX_AGENT_TOOL_ROUNDS = 8;

export const LOCAL_AGENT_DISCOVERY_TOOL_NAMES = new Set([
  'file_search',
  'grep_search',
  'get_errors',
  'list_dir',
  'read_file',
]);

export type LocalAgentTurnPhase =
  | 'notLocalAgent'
  | 'requireEvidence'
  | 'allowToolOrFinal'
  | 'forceFinal';

export interface LocalAgentTurnState {
  phase: LocalAgentTurnPhase;
  invocationCount: number;
  evidenceCount: number;
  resultCharacters: number;
}

export interface MatchingPriorFinal {
  index: number;
  text: string;
}

const FAILED_TOOL_RESULT = /^(?:error|failed|failure|cancelled|canceled|invalid)\b/i;

export function evaluateLocalAgentTurn(
  messages: readonly ChatMessage[],
  maxToolInvocations: number = DEFAULT_MAX_AGENT_TOOL_ROUNDS,
): LocalAgentTurnState {
  const requestIndex = activeLocalAgentRequestIndex(messages);
  if (!isLocalAgentRequest(messages) || requestIndex < 0) {
    return emptyTurnState('notLocalAgent');
  }

  const calls = new Map<string, string>();
  const evidenceDigests = new Set<string>();
  let invocationCount = 0;
  let resultCharacters = 0;
  for (let index = requestIndex + 1; index < messages.length; index += 1) {
    const message = messages[index];
    if (message?.role === 'assistant') {
      for (const call of message.tool_calls ?? []) {
        invocationCount += 1;
        calls.set(call.id, call.function.name);
      }
      continue;
    }
    if (message?.role !== 'tool' || !message.tool_call_id) {
      continue;
    }
    resultCharacters += message.content.length;
    const toolName = calls.get(message.tool_call_id);
    const normalizedResult = normalizeText(message.content);
    if (
      toolName &&
      LOCAL_AGENT_DISCOVERY_TOOL_NAMES.has(toolName) &&
      normalizedResult &&
      !FAILED_TOOL_RESULT.test(normalizedResult)
    ) {
      evidenceDigests.add(normalizedResult);
    }
  }

  const ceiling = Math.max(1, Math.floor(maxToolInvocations));
  const phase: LocalAgentTurnPhase = invocationCount >= ceiling
    ? 'forceFinal'
    : evidenceDigests.size > 0
      ? 'allowToolOrFinal'
      : 'requireEvidence';
  return {
    phase,
    invocationCount,
    evidenceCount: evidenceDigests.size,
    resultCharacters,
  };
}

/** Whether this Local Agent turn still needs its first successful evidence result. */
export function shouldRequireLocalAgentTool(
  messages: readonly ChatMessage[],
  maxToolRounds: number = DEFAULT_MAX_AGENT_TOOL_ROUNDS,
): boolean {
  return evaluateLocalAgentTurn(messages, maxToolRounds).phase === 'requireEvidence';
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

export function matchingPriorLocalAgentFinal(
  messages: readonly ChatMessage[],
): MatchingPriorFinal | undefined {
  const requestIndex = activeLocalAgentRequestIndex(messages);
  const request = messages[requestIndex];
  if (requestIndex < 0 || request?.role !== 'user') {
    return undefined;
  }
  const normalizedRequest = normalizeText(request.content);
  for (let index = requestIndex - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message?.role !== 'user' || normalizeText(message.content) !== normalizedRequest) {
      continue;
    }
    for (let candidate = index + 1; candidate < requestIndex; candidate += 1) {
      const response = messages[candidate];
      if (
        response?.role === 'user' &&
        !hasProtocolMarker(response.content) &&
        !/^\s*(?:additional\s+)?hook context\s*:/i.test(response.content)
      ) {
        break;
      }
      if (
        response?.role === 'assistant' &&
        !response.tool_calls?.length &&
        response.content.trim()
      ) {
        return { index: candidate, text: response.content };
      }
    }
  }
  return undefined;
}

export function messagesForFreshLocalAgentRequest(
  messages: readonly ChatMessage[],
): ChatMessage[] {
  const prior = matchingPriorLocalAgentFinal(messages);
  return prior
    ? messages.filter((_message, index) => index !== prior.index)
    : [...messages];
}

export function isExactNormalizedResponse(previous: string, candidate: string): boolean {
  const normalized = normalizeText(candidate);
  return normalized.length > 0 && normalized === normalizeText(previous);
}

export function isRepeatedLocalAgentToolCall(
  messages: readonly ChatMessage[],
  name: string,
  input: object,
): boolean {
  const requestIndex = activeLocalAgentRequestIndex(messages);
  if (requestIndex < 0) {
    return false;
  }
  const candidateSignature = toolCallSignature(name, input);
  const calls = new Map<string, { index: number; name: string }>();
  let matchingCallIndex = -1;
  let successfulEditAfterMatch = false;
  for (let index = requestIndex + 1; index < messages.length; index += 1) {
    const message = messages[index];
    if (message?.role === 'assistant') {
      for (const call of message.tool_calls ?? []) {
        calls.set(call.id, { index, name: call.function.name });
        if (
          toolCallSignatureFromJson(call.function.name, call.function.arguments) ===
          candidateSignature
        ) {
          matchingCallIndex = index;
          successfulEditAfterMatch = false;
        }
      }
      continue;
    }
    if (message?.role !== 'tool' || !message.tool_call_id || matchingCallIndex < 0) {
      continue;
    }
    const call = calls.get(message.tool_call_id);
    if (
      call &&
      call.index > matchingCallIndex &&
      LOCAL_AGENT_EDIT_TOOL_NAMES.has(call.name) &&
      isSuccessfulToolResult(message.content)
    ) {
      successfulEditAfterMatch = true;
    }
  }
  return matchingCallIndex >= 0 && !successfulEditAfterMatch;
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

function emptyTurnState(phase: LocalAgentTurnPhase): LocalAgentTurnState {
  return { phase, invocationCount: 0, evidenceCount: 0, resultCharacters: 0 };
}

function normalizeText(value: string): string {
  return value.normalize('NFC').trim().replace(/\s+/g, ' ');
}

const LOCAL_AGENT_EDIT_TOOL_NAMES = new Set([
  'insert_edit_into_file',
  'replace_string_in_file',
]);

function isSuccessfulToolResult(content: string): boolean {
  const normalized = normalizeText(content);
  return Boolean(normalized) && !FAILED_TOOL_RESULT.test(normalized);
}

function toolCallSignature(name: string, input: object): string {
  return `${name}:${canonicalJson(input)}`;
}

function toolCallSignatureFromJson(name: string, input: string): string {
  try {
    const parsed = JSON.parse(input) as unknown;
    return `${name}:${canonicalJson(parsed)}`;
  } catch {
    return `${name}:${input.trim()}`;
  }
}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map((item) => canonicalJson(item)).join(',')}]`;
  }
  if (value && typeof value === 'object') {
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record).sort().map(
      (key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`,
    ).join(',')}}`;
  }
  return JSON.stringify(value) ?? 'null';
}
