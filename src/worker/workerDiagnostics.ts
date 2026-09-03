import type { LlamaClientDiagnostics } from './llamaClient';

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
