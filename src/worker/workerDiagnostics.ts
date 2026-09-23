import type { LlamaClientDiagnostics } from './llamaClient.js';

interface WorkerDiagnosticLogger {
  info(message: string): void;
}

export function createWorkerDiagnostics(
  logger: WorkerDiagnosticLogger,
): LlamaClientDiagnostics {
  return {
    info: (message) => logger.info(message),
  };
}

export type WorkerLineLevel = 'error' | 'warn' | 'debug';

/**
 * How loudly to log one line of worker output.
 *
 * Worker output is mostly progress chatter, but both runtimes mark their own
 * failures: llama.cpp prefixes each line with a level letter after its
 * timestamp, and the Python worker writes `[error]` and `[warning]`. Those
 * must reach the log at the default level, not only under debug.
 */
export function workerLineLevel(line: string): WorkerLineLevel {
  const llama = /^\s*\d+\.\d+\.\d+\.\d+ ([EW]) /.exec(line);
  if (llama) {
    return llama[1] === 'E' ? 'error' : 'warn';
  }
  const python = /^\[(error|warning)\] /.exec(line);
  if (python) {
    return python[1] === 'error' ? 'error' : 'warn';
  }
  return 'debug';
}
