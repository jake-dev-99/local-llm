import type { LlamaClientDiagnostics } from './llamaClient';

interface WorkerDiagnosticLogger {
  info(message: string): void;
}

export class WorkerStartupDiagnostics {
  private output = '';
  private finished = false;
  private readonly logger: { error(message: string): void };

  constructor(logger: { error(message: string): void }) {
    this.logger = logger;
  }

  append(chunk: string): void {
    if (!this.finished) {
      this.output = (this.output + chunk).slice(-16 * 1024);
    }
  }

  complete(): void {
    this.finished = true;
    this.output = '';
  }

  reportFailure(): void {
    if (this.finished) {
      return;
    }
    const output = this.output.trim();
    this.complete();
    this.logger.error('[Model Loading] Recent worker startup output:');
    for (const line of (output || '<no startup output captured>').split(/\r?\n/)) {
      if (line.trim()) {
        this.logger.error(`worker: ${line.trim()}`);
      }
    }
  }
}

export function createWorkerDiagnostics(
  logger: WorkerDiagnosticLogger,
): LlamaClientDiagnostics {
  return {
    info: (message) => logger.info(message),
  };
}
