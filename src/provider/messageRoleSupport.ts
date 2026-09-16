import type { ChatMessage } from '../domain.js';

export function messagesForSystemRoleSupport(
  messages: readonly ChatMessage[],
  supportsSystemRole: boolean,
): ChatMessage[] {
  if (supportsSystemRole) {
    return [...messages];
  }
  return messages.map((message) =>
    message.role === 'system'
      ? { ...message, role: 'user' }
      : message,
  );
}
