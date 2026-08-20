import * as vscode from 'vscode';
import type { ChatMessage, ChatTool } from '../domain';

export function adaptMessages(
  messages: readonly vscode.LanguageModelChatRequestMessage[],
): ChatMessage[] {
  const result: ChatMessage[] = [];
  for (const message of messages) {
    if (message.role === vscode.LanguageModelChatMessageRole.Assistant) {
      result.push(adaptAssistantMessage(message));
      continue;
    }
    if (message.role === vscode.LanguageModelChatMessageRole.User) {
      result.push(...adaptUserMessage(message));
      continue;
    }
    result.push({
      role: 'system',
      content: message.content.map(partToText).join('\n'),
    });
  }
  return result;
}

export function adaptTools(
  tools: readonly vscode.LanguageModelChatTool[] | undefined,
): ChatTool[] | undefined {
  if (!tools?.length) {
    return undefined;
  }
  return tools.map((tool) => ({
    type: 'function',
    function: {
      name: tool.name,
      ...(tool.description ? { description: tool.description } : {}),
      ...(tool.inputSchema ? { parameters: tool.inputSchema } : {}),
    },
  }));
}

export function serializeMessageForTokenCount(
  message: vscode.LanguageModelChatRequestMessage,
): string {
  return message.content.map(partToText).join('\n');
}

function adaptAssistantMessage(
  message: vscode.LanguageModelChatRequestMessage,
): ChatMessage {
  const text: string[] = [];
  const toolCalls: NonNullable<ChatMessage['tool_calls']> = [];
  for (const part of message.content) {
    if (part instanceof vscode.LanguageModelTextPart) {
      text.push(part.value);
    } else if (part instanceof vscode.LanguageModelToolCallPart) {
      toolCalls.push({
        id: part.callId,
        type: 'function',
        function: { name: part.name, arguments: JSON.stringify(part.input) },
      });
    }
  }
  return {
    role: 'assistant',
    content: text.join(''),
    ...(toolCalls.length ? { tool_calls: toolCalls } : {}),
  };
}

function adaptUserMessage(
  message: vscode.LanguageModelChatRequestMessage,
): ChatMessage[] {
  const result: ChatMessage[] = [];
  const text: string[] = [];
  for (const part of message.content) {
    if (part instanceof vscode.LanguageModelTextPart) {
      text.push(part.value);
    } else if (part instanceof vscode.LanguageModelToolResultPart) {
      if (text.length) {
        result.push({ role: 'user', content: text.splice(0).join('') });
      }
      result.push({
        role: 'tool',
        content: part.content.map(partToText).join('\n'),
        tool_call_id: part.callId,
      });
    } else {
      text.push(partToText(part));
    }
  }
  if (text.length || result.length === 0) {
    result.push({ role: 'user', content: text.join('') });
  }
  return result;
}

function partToText(part: unknown): string {
  if (part instanceof vscode.LanguageModelTextPart) {
    return part.value;
  }
  if (part instanceof vscode.LanguageModelToolCallPart) {
    return JSON.stringify({ tool: part.name, input: part.input });
  }
  if (part instanceof vscode.LanguageModelToolResultPart) {
    return part.content.map(partToText).join('\n');
  }
  if (part instanceof vscode.LanguageModelDataPart) {
    return `[binary data: ${part.mimeType}]`;
  }
  if (typeof part === 'string') {
    return part;
  }
  try {
    return JSON.stringify(part);
  } catch {
    return String(part);
  }
}
