import type { ChatMessage } from '../domain';

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
