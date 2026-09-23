export interface ModelTokenLimits {
  maxInputTokens: number;
  maxOutputTokens: number;
}

/**
 * Splits one context window between the prompt and the reply.
 *
 * The reply may take at most half. VS Code budgets its whole prompt against
 * `maxInputTokens` before it sends a request, and when the prompt cannot fit it
 * abandons the request without ever calling the provider. Letting a large
 * `maxOutputTokens` take everything but one token of a small fitted window
 * made a model impossible to chat with, silently.
 */
export function modelTokenLimits(
  contextSize: number,
  requestedMaxOutput: number,
): ModelTokenLimits {
  const physicalContext = Math.max(2, Math.floor(contextSize));
  const maxOutputTokens = Math.min(
    Math.max(1, Math.floor(requestedMaxOutput)),
    Math.floor(physicalContext / 2),
  );
  return {
    maxInputTokens: physicalContext - maxOutputTokens,
    maxOutputTokens,
  };
}

/**
 * Context window to advertise to VS Code before a request runs.
 *
 * llama.cpp fits the window to this machine at load time, so the only truthful
 * source is what the worker actually loaded. Everything else is a bootstrap
 * estimate for a model that has never been loaded here.
 */
export interface AdvertisedContextInput {
  /** localLlm.contextSize. Zero means automatic. */
  configuredContextSize: number;
  /** Window the worker reported through /props on its last load. */
  loadedContextSize?: number;
  /** Trained window from the GGUF header, used only before a first load. */
  trainedContextLength?: number;
}

/**
 * llama.cpp's own floor when reducing a context window to fit, from
 * `fit_params_min_ctx` in common/common.h at the pinned revision.
 */
export const LLAMA_CPP_MIN_FITTED_CONTEXT = 4096;

export function resolveAdvertisedContextSize(input: AdvertisedContextInput): number {
  if (isPositive(input.configuredContextSize)) {
    return Math.floor(input.configuredContextSize);
  }
  if (isPositive(input.loadedContextSize)) {
    return Math.floor(input.loadedContextSize);
  }
  if (isPositive(input.trainedContextLength)) {
    return Math.floor(input.trainedContextLength);
  }
  return LLAMA_CPP_MIN_FITTED_CONTEXT;
}

function isPositive(value: number | undefined): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0;
}
