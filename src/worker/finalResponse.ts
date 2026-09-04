import type { ChatMessage } from '../domain';
import MarkdownIt from 'markdown-it';

const markdown = new MarkdownIt('commonmark');
const finalInstruction = '[Local LLM runtime instruction — not part of the tool output]\n' +
  'The tool phase is over. Do not call tools or print tool-call markup. ' +
  'Provide a concise final answer using only the existing conversation and tool results. ' +
  'State unfinished work plainly. Do not claim actions or verification without a supporting tool result.';

export function finalResponseMessages(messages: readonly ChatMessage[]): ChatMessage[] {
  // A new user turn changes Qwen's last_query_index and re-renders earlier
  // assistant reasoning blocks. Extend only the last result's tail instead.
  const last = messages.at(-1);
  if (last?.role === 'tool') {
    return [...messages.slice(0, -1), {
      ...last,
      content: `${last.content}\n\n${finalInstruction}`,
    }];
  }
  return [...messages, { role: 'user', content: finalInstruction }];
}

export function assertFinalResponse(text: string, hasToolCalls: boolean): void {
  if (hasToolCalls || containsUnquotedToolCall(text)) {
    throw new Error(
      'The model returned tool-call output while tool execution was disabled for the final answer. ' +
      'No additional tool was executed. Review the existing edits and tool results; ' +
      'start a follow-up request if more work is needed.',
    );
  }
}

function containsUnquotedToolCall(text: string): boolean {
  // Inspect prose only. Use Markdown's code classification so escaped/mismatched
  // backticks cannot hide a call, and real fenced/indented/inline code is intact.
  return markdown.parse(text, {}).some(block => {
    const prose = block.type === 'html_block' ? block.content :
      (block.children ?? [])
        .filter(token => token.type === 'text' || token.type === 'html_inline')
        .map(token => token.content).join('');
    return /<tool_call>|<function=[^>\r\n]+>/.test(prose);
  });
}
