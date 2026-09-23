/**
 * Detects a chat prompt VS Code measured and then never sent.
 *
 * VS Code counts a prompt's tokens against the advertised input limit before
 * it sends a request, and when the prompt cannot be made to fit it abandons
 * the request without calling the provider again or telling the user. The
 * provider has no call left to fail, so the drop is only visible as token
 * counting that no chat request follows. This turns that absence into a
 * report.
 */
export interface DroppedPrompt {
  modelName: string;
  /** How many token counts VS Code requested while assembling the prompt. */
  counts: number;
  /**
   * The sum of those counts. VS Code recounts pieces while it prunes, so this
   * overstates the prompt; a sum within the limit still proves it fit.
   */
  countedTokens: number;
  largestCount: number;
  inputLimit: number;
}

export interface DetectorTimers {
  set(callback: () => void, milliseconds: number): unknown;
  clear(handle: unknown): void;
}

interface Assembly {
  modelName: string;
  inputLimit: number;
  counts: number;
  countedTokens: number;
  largestCount: number;
  timer: unknown;
}

/** VS Code sends a request within milliseconds of its last count; seconds of silence means it will not. */
export const PROMPT_SEND_GRACE_MS = 3_000;

const defaultTimers: DetectorTimers = {
  set: (callback, milliseconds) => setTimeout(callback, milliseconds),
  clear: (handle) => clearTimeout(handle as ReturnType<typeof setTimeout>),
};

export class PromptDropDetector {
  private readonly assemblies = new Map<string, Assembly>();
  private readonly onDropped: (drop: DroppedPrompt) => void;
  private readonly graceMs: number;
  private readonly timers: DetectorTimers;

  constructor(
    onDropped: (drop: DroppedPrompt) => void,
    graceMs = PROMPT_SEND_GRACE_MS,
    timers: DetectorTimers = defaultTimers,
  ) {
    this.onDropped = onDropped;
    this.graceMs = graceMs;
    this.timers = timers;
  }

  /** Records one token count VS Code requested for a model. */
  counted(modelId: string, modelName: string, tokens: number, inputLimit: number): void {
    const current = this.assemblies.get(modelId);
    if (current) {
      this.timers.clear(current.timer);
    }
    const assembly: Assembly = {
      modelName,
      inputLimit,
      counts: (current?.counts ?? 0) + 1,
      countedTokens: (current?.countedTokens ?? 0) + tokens,
      largestCount: Math.max(current?.largestCount ?? 0, tokens),
      timer: undefined,
    };
    assembly.timer = this.timers.set(() => {
      if (this.assemblies.get(modelId) !== assembly) {
        return;
      }
      this.assemblies.delete(modelId);
      this.onDropped({
        modelName: assembly.modelName,
        counts: assembly.counts,
        countedTokens: assembly.countedTokens,
        largestCount: assembly.largestCount,
        inputLimit: assembly.inputLimit,
      });
    }, this.graceMs);
    this.assemblies.set(modelId, assembly);
  }

  /** A chat request arrived, so the prompt VS Code was measuring was sent. */
  sent(modelId: string): void {
    const assembly = this.assemblies.get(modelId);
    if (assembly) {
      this.timers.clear(assembly.timer);
      this.assemblies.delete(modelId);
    }
  }

  dispose(): void {
    for (const assembly of this.assemblies.values()) {
      this.timers.clear(assembly.timer);
    }
    this.assemblies.clear();
  }
}
