import type { LlamaClientDiagnostics } from './llamaClient';

interface WorkerDiagnosticLogger {
  info(message: string): void;
  warn(message: string, visible?: boolean): void;
}

const GENERATION_STALL_WARNING_MS = 30_000;
const LONG_GENERATION_WARNING_MS = 60_000;

export function createWorkerDiagnostics(
  logger: WorkerDiagnosticLogger,
): LlamaClientDiagnostics {
  return {
    info: (message) => logger.info(message),
    warn: (message) => logger.warn(message, true),
    stallWarningMilliseconds: GENERATION_STALL_WARNING_MS,
    longGenerationWarningMilliseconds: LONG_GENERATION_WARNING_MS,
  };
}
