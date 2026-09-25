/**
 * Detects a chat prompt VS Code measured and then never sent.
 *
 * VS Code counts a prompt's tokens against the advertised input limit before
 * it sends a request, and when the prompt cannot be made to fit it abandons
 * the request without calling the provider again or telling the user. The
 * provider has no call left to fail, so the drop is only visible as token
 * counting that no chat request follows. This turns that absence into a
 * report.
 *
 * VS Code also counts tokens right after a reply, to update its context
 * display, and sends nothing then because nothing is pending. Counting that
 * starts within `settleMs` of a finished request is therefore not watched.
 */
export interface DroppedPrompt {
  modelName: string;
  /** How many token counts VS Code requested while assembling the prompt. */
  counts: number;
  /**
   * The largest single count. VS Code recounts the same pieces while it
   * prunes, so a sum of counts means nothing; one piece larger than the whole
   * input limit is the only proof that the prompt could not fit.
   */
  largestCount: number;
  inputLimit: number;
}

export interface DetectorTimers {
  set(callback: () => void, milliseconds: number): unknown;
  clear(handle: unknown): void;
}

export interface DetectorOptions {
  /** Silence after the last count before a missing request is reported. */
  graceMs?: number;
  /** How long after a finished request counting is treated as VS Code's own bookkeeping. */
  settleMs?: number;
  timers?: DetectorTimers;
  now?: () => number;
}

interface Assembly {
  modelName: string;
  inputLimit: number;
  counts: number;
  largestCount: number;
  /** Counting that began right after a reply; tracked only to be ignored as one burst. */
  afterReply: boolean;
  timer: unknown;
}

/** VS Code sends a request within milliseconds of its last count; seconds of silence means it will not. */
export const PROMPT_SEND_GRACE_MS = 3_000;
export const REPLY_BOOKKEEPING_MS = 5_000;

const defaultTimers: DetectorTimers = {
  set: (callback, milliseconds) => setTimeout(callback, milliseconds),
  clear: (handle) => clearTimeout(handle as ReturnType<typeof setTimeout>),
};

export class PromptDropDetector {
  private readonly assemblies = new Map<string, Assembly>();
  private readonly lastFinished = new Map<string, number>();
  private readonly onDropped: (drop: DroppedPrompt) => void;
  private readonly graceMs: number;
  private readonly settleMs: number;
  private readonly timers: DetectorTimers;
  private readonly now: () => number;

  constructor(onDropped: (drop: DroppedPrompt) => void, options: DetectorOptions = {}) {
    this.onDropped = onDropped;
    this.graceMs = options.graceMs ?? PROMPT_SEND_GRACE_MS;
    this.settleMs = options.settleMs ?? REPLY_BOOKKEEPING_MS;
    this.timers = options.timers ?? defaultTimers;
    this.now = options.now ?? Date.now;
  }

  /** Records one token count VS Code requested for a model. */
  counted(modelId: string, modelName: string, tokens: number, inputLimit: number): void {
    const current = this.assemblies.get(modelId);
    if (current) {
      this.timers.clear(current.timer);
    }
    const finishedAt = this.lastFinished.get(modelId);
    const assembly: Assembly = {
      modelName,
      inputLimit,
      counts: (current?.counts ?? 0) + 1,
      largestCount: Math.max(current?.largestCount ?? 0, tokens),
      afterReply: current?.afterReply ??
        (finishedAt !== undefined && this.now() - finishedAt < this.settleMs),
      timer: undefined,
    };
    assembly.timer = this.timers.set(() => {
      if (this.assemblies.get(modelId) !== assembly) {
        return;
      }
      this.assemblies.delete(modelId);
      if (assembly.afterReply) {
        return;
      }
      this.onDropped({
        modelName: assembly.modelName,
        counts: assembly.counts,
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

  /** A chat request ended, however it ended. */
  finished(modelId: string): void {
    this.lastFinished.set(modelId, this.now());
  }

  dispose(): void {
    for (const assembly of this.assemblies.values()) {
      this.timers.clear(assembly.timer);
    }
    this.assemblies.clear();
  }
}
