/**
 * A started runtime: the process, the client that talks to it, and how it ends.
 *
 * `WorkerManager` holds exactly one of these at a time, whichever engine is
 * holding the model. That single slot is what keeps two large checkpoints from
 * ever being resident at once, and it is why adding a second runtime needed no
 * second scheduler: every request already funnels through one manager.
 *
 * llama.cpp is an HTTP server on a loopback port; the Safetensors runtime is a
 * Python process on a pipe. Above this line the difference does not exist.
 */

import type { ModelFormat, ModelRuntime } from '../models/modelIdentity.ts';
import type { InferenceClient } from './inferenceClient.ts';

export interface RuntimeExit {
  readonly code: number | null;
  readonly signal: string | null;
}

export interface RuntimeSession {
  readonly runtime: ModelRuntime;
  readonly client: InferenceClient;
  /** The loopback port, for runtimes that serve HTTP. */
  readonly port: number | undefined;
  /** True once the process is gone, however it went. */
  readonly exited: boolean;
  /** Called when the process goes away, including when `stop` ended it. */
  onExit(handler: (exit: RuntimeExit) => void): void;
  /** Ends the process and releases the client. Safe to call more than once. */
  stop(): Promise<void>;
}

/**
 * Which engine loads this model.
 *
 * Records written before Safetensors support carry no runtime, and every one
 * of them is GGUF, so an absent value resolves to llama.cpp instead of
 * failing. The format is the fallback rather than the primary answer because a
 * format can outlive its default runtime: a GGUF file read by something other
 * than llama.cpp would still be GGUF.
 */
export function runtimeForModel(
  model: { runtime?: ModelRuntime; format?: ModelFormat },
): ModelRuntime {
  if (model.runtime) {
    return model.runtime;
  }
  return model.format === 'safetensors' ? 'transformers' : 'llama-cpp';
}

/** The engine's name, for status text a user reads. */
export function runtimeDisplayName(runtime: ModelRuntime): string {
  return runtime === 'transformers' ? 'Transformers' : 'llama.cpp';
}

export interface ExitNotifier {
  readonly exited: boolean;
  notify(exit: RuntimeExit): void;
  onExit(handler: (exit: RuntimeExit) => void): void;
}

/**
 * One-shot exit notification that cannot be missed.
 *
 * A handler registered after the process has already gone is called straight
 * away. Without that, a process dying in the window between becoming healthy
 * and being watched would leave a session that looks alive forever, and the
 * manager would keep handing out a client to a worker that is not there.
 */
export function exitNotifier(): ExitNotifier {
  let exit: RuntimeExit | undefined;
  const handlers: Array<(exit: RuntimeExit) => void> = [];
  return {
    get exited() {
      return exit !== undefined;
    },
    notify(next) {
      if (exit) {
        return;
      }
      exit = next;
      for (const handler of handlers.splice(0)) {
        handler(next);
      }
    },
    onExit(handler) {
      if (exit) {
        handler(exit);
        return;
      }
      handlers.push(handler);
    },
  };
}
