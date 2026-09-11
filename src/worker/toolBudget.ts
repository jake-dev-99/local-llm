export interface ToolBudgetResult<T> {
  tools: T[];
  inputTokens: number;
}

export async function assertPromptFits<T>(
  tools: T[],
  budget: number,
  countInputTokens: (tools: T[]) => Promise<number>,
): Promise<ToolBudgetResult<T>> {
  const inputTokens = await countInputTokens(tools);
  if (inputTokens > budget) {
    throw new Error(
      `The complete prompt (messages and ${tools.length} tool definitions) requires ${inputTokens} input tokens, but only ${budget} are available after reserving response tokens. Even Local Agent's bounded tool set may exceed a small loaded context window. Start a new chat or reduce attached context, or increase localLlm.contextSize if memory allows. Lowering localLlm.maxOutputTokens frees input space only when the prompt itself fits within the loaded context. No tools were removed.`,
    );
  }
  return { tools, inputTokens };
}
