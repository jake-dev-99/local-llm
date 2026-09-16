import type { ChatTool } from '../domain.js';

const LOCAL_AGENT_TOOL_NAMES = new Set([
  'file_search',
  'grep_search',
  'get_errors',
  'insert_edit_into_file',
  'list_dir',
  'read_file',
  'replace_string_in_file',
]);

export function localAgentAvailableTools(
  tools: readonly ChatTool[],
): ChatTool[] {
  return tools.filter((tool) => LOCAL_AGENT_TOOL_NAMES.has(tool.function.name));
}

export interface LocalAgentToolPolicy {
  tools: ChatTool[];
  toolChoice: 'auto' | 'required' | 'none';
  source: 'caller-required' | 'caller-auto' | 'local-agent-final';
}

export function resolveLocalAgentToolPolicy(
  tools: readonly ChatTool[],
  callerRequired: boolean,
  localAgentForceFinal: boolean,
): LocalAgentToolPolicy {
  if (localAgentForceFinal) {
    // Definitions are part of the cached prompt, not permission to execute.
    return { tools: [...tools], toolChoice: 'none', source: 'local-agent-final' };
  }
  if (callerRequired) {
    return { tools: [...tools], toolChoice: 'required', source: 'caller-required' };
  }
  return { tools: [...tools], toolChoice: 'auto', source: 'caller-auto' };
}
