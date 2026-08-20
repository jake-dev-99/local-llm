export type InferenceKind = 'chat' | 'utility' | 'inline';

interface QueueItem<T> {
  readonly kind: InferenceKind;
  readonly sequence: number;
  readonly controller: AbortController;
  readonly task: (signal: AbortSignal) => Promise<T>;
  readonly resolve: (value: T | PromiseLike<T>) => void;
  readonly reject: (reason?: unknown) => void;
  readonly disposeExternalCancellation: () => void;
  settled: boolean;
}

export class InferenceScheduler {
  private readonly queue: Array<QueueItem<unknown>> = [];
  private active: QueueItem<unknown> | undefined;
  private sequence = 0;
  private draining = false;

  get activeKind(): InferenceKind | undefined {
    return this.active?.kind;
  }

  get hasQueuedChat(): boolean {
    return this.queue.some((item) => item.kind === 'chat' && !item.settled);
  }

  run<T>(
    kind: InferenceKind,
    task: (signal: AbortSignal) => Promise<T>,
    externalSignal?: AbortSignal,
  ): Promise<T> {
    if (externalSignal?.aborted) {
      return Promise.reject(abortError());
    }

    const controller = new AbortController();
    return new Promise<T>((resolve, reject) => {
      let item: QueueItem<T>;
      const cancel = (): void => {
        controller.abort();
        if (this.active !== item && !item.settled) {
          item.settled = true;
          item.disposeExternalCancellation();
          reject(abortError());
        }
      };
      if (externalSignal) {
        externalSignal.addEventListener('abort', cancel, { once: true });
      }
      item = {
        kind,
        sequence: this.sequence++,
        controller,
        task,
        resolve,
        reject,
        settled: false,
        disposeExternalCancellation: () => {
          externalSignal?.removeEventListener('abort', cancel);
        },
      };
      this.queue.push(item as QueueItem<unknown>);
      if (kind === 'chat' && this.active?.kind === 'inline') {
        this.active.controller.abort();
      }
      void this.drain();
    });
  }

  cancelAll(): void {
    this.active?.controller.abort();
    for (const item of this.queue.splice(0)) {
      if (!item.settled) {
        item.settled = true;
        item.controller.abort();
        item.disposeExternalCancellation();
        item.reject(abortError());
      }
    }
  }

  private async drain(): Promise<void> {
    if (this.draining) {
      return;
    }
    this.draining = true;
    try {
      while (!this.active) {
        const item = this.takeNext();
        if (!item) {
          return;
        }
        if (item.settled || item.controller.signal.aborted) {
          if (!item.settled) {
            item.settled = true;
            item.disposeExternalCancellation();
            item.reject(abortError());
          }
          continue;
        }
        this.active = item;
        try {
          const value = await item.task(item.controller.signal);
          if (!item.settled) {
            item.settled = true;
            if (item.controller.signal.aborted) {
              item.reject(abortError());
            } else {
              item.resolve(value);
            }
          }
        } catch (error) {
          if (!item.settled) {
            item.settled = true;
            item.reject(error);
          }
        } finally {
          item.disposeExternalCancellation();
          if (this.active === item) {
            this.active = undefined;
          }
        }
      }
    } finally {
      this.draining = false;
      if (!this.active && this.queue.some((item) => !item.settled)) {
        void this.drain();
      }
    }
  }

  private takeNext(): QueueItem<unknown> | undefined {
    const candidates = this.queue.filter((item) => !item.settled);
    if (!candidates.length) {
      this.queue.length = 0;
      return undefined;
    }
    candidates.sort((left, right) =>
      priority(left.kind) - priority(right.kind) || left.sequence - right.sequence,
    );
    const next = candidates[0];
    if (!next) {
      return undefined;
    }
    const index = this.queue.indexOf(next);
    if (index >= 0) {
      this.queue.splice(index, 1);
    }
    return next;
  }
}

function priority(kind: InferenceKind): number {
  switch (kind) {
    case 'chat':
      return 0;
    case 'utility':
      return 1;
    case 'inline':
      return 2;
  }
}

export function abortError(): Error {
  const error = new Error('Local inference was cancelled.');
  error.name = 'AbortError';
  return error;
}
