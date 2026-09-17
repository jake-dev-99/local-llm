/**
 * What the extension asks of a loaded model, independent of which worker holds
 * it.
 *
 * `LlamaClient` speaks HTTP to llama-server; `TransformersClient` speaks JSON
 * Lines to the Python runtime. Callers that only chat, count tokens, or read a
 * profile work against either. Everything a single runtime provides — infill
 * today, grammars later — is declared optional and guarded by `supports`, so a
 * missing capability is a branch rather than a runtime failure.
 */

import type {
  ChatRequest,
  ChatStreamEvent,
  InfillRequest,
} from '../domain.ts';
import type { ChatResult } from './llamaClient.ts';
import type { WorkerModelProfile } from './runtimeProfile.ts';

export interface InferenceCapabilities {
  /** Fill-in-the-middle for inline completion. llama.cpp's `/infill` only. */
  infill: boolean;
  /**
   * Schema-constrained decoding.
   *
   * The Local Agent depends on it: on some models the constrained fallback is
   * the only tool path that produces anything at all.
   */
  constrainedDecoding: boolean;
}

export interface InferenceClient {
  readonly supports: InferenceCapabilities;

  chat(
    request: ChatRequest,
    onEvent: (event: ChatStreamEvent) => void,
    signal?: AbortSignal,
  ): Promise<ChatResult>;

  /** Token count from the model's own tokenizer, never an estimate. */
  tokenize(content: string, signal?: AbortSignal): Promise<number>;

  getModelProfile(signal?: AbortSignal): Promise<WorkerModelProfile>;

  /** Present only when `supports.infill` is true. */
  infill?(request: InfillRequest, signal?: AbortSignal): Promise<string>;
}

/**
 * Narrows a client to one that can serve inline completion.
 *
 * Inline completion is offered per model rather than per runtime, so the check
 * belongs at the call site instead of in a cast.
 */
export function supportsInfill(
  client: InferenceClient,
): client is InferenceClient & Required<Pick<InferenceClient, 'infill'>> {
  return client.supports.infill && typeof client.infill === 'function';
}
