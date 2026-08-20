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
      `The complete ${tools.length}-tool contract requires ${inputTokens} input tokens, but the local model input budget is ${budget}. Start a new chat, use the Local Agent with its bounded tool set, reduce attached context, or increase localLlm.contextSize.`,
    );
  }
  return { tools, inputTokens };
}
