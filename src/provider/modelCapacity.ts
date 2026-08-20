export interface ModelTokenLimits {
  maxInputTokens: number;
  maxOutputTokens: number;
}

export function modelTokenLimits(
  contextSize: number,
  requestedMaxOutput: number,
): ModelTokenLimits {
  const physicalContext = Math.max(2, Math.floor(contextSize));
  const maxOutputTokens = Math.min(
    Math.max(1, Math.floor(requestedMaxOutput)),
    physicalContext - 1,
  );
  return {
    maxInputTokens: physicalContext - maxOutputTokens,
    maxOutputTokens,
  };
}
