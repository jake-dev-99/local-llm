/**
 * JSON Lines client for the Safetensors runtime worker.
 *
 * The transport is injected rather than spawned here so the protocol — request
 * correlation, streaming, crash recovery — can be tested without a Python
 * interpreter. `spawnPythonWorkerTransport` supplies the real one.
 */

import {
  PythonWorkerError,
  SUPPORTED_PROTOCOL_VERSION,
  type ModelInspection,
  type PythonChatMessage,
  type PythonErrorCode,
  type PythonGenerationOptions,
  type PythonGenerationResult,
  type PythonModelInfo,
  type PythonRuntimeInfo,
  type PythonRuntimePolicy,
  type WorkerLifecycleState,
} from './pythonWorkerTypes.ts';

export interface WorkerTransport {
  send(line: string): void;
  onLine(handler: (line: string) => void): void;
  onExit(handler: (code: number | null, signal: string | null) => void): void;
  close(): void;
}

export interface PythonWorkerClientOptions {
  transport: WorkerTransport;
  onState?: (state: WorkerLifecycleState, detail: string | null) => void;
  /** Protocol-level diagnostics. Never carries prompt or source content. */
  onLog?: (message: string) => void;
}

interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  onToken?: (text: string) => void;
}

export interface GenerateRequest {
  options?: PythonGenerationOptions;
  onToken?: (text: string) => void;
}

/**
 * Drives one worker process.
 *
 * A single worker holds at most one model and runs at most one generation, so
 * the extension's existing scheduler remains the place where concurrency is
 * decided. This class only refuses overlap; it does not queue.
 */
export class PythonWorkerClient {
  private readonly transport: WorkerTransport;
  private readonly pending = new Map<number, PendingRequest>();
  private readonly onState: PythonWorkerClientOptions['onState'];
  private readonly onLog: PythonWorkerClientOptions['onLog'];
  private nextId = 1;
  private exited = false;
  private activeGenerationId: number | undefined;

  constructor(options: PythonWorkerClientOptions) {
    this.transport = options.transport;
    this.onState = options.onState;
    this.onLog = options.onLog;
    this.transport.onLine((line) => this.receive(line));
    this.transport.onExit((code, signal) => this.handleExit(code, signal));
  }

  // ------------------------------------------------------------------ requests

  async runtimeInfo(): Promise<PythonRuntimeInfo> {
    const info = await this.request<PythonRuntimeInfo>('runtime.info');
    if (info.protocolVersion !== SUPPORTED_PROTOCOL_VERSION) {
      throw new PythonWorkerError(
        'internal_error',
        `Worker speaks protocol ${info.protocolVersion}; this extension supports ` +
        `${SUPPORTED_PROTOCOL_VERSION}. Reinstall so both sides match.`,
      );
    }
    return info;
  }

  async inspect(path: string): Promise<ModelInspection> {
    return await this.request<ModelInspection>('model.inspect', { path });
  }

  async load(path: string, policy?: PythonRuntimePolicy): Promise<PythonModelInfo> {
    return await this.request<PythonModelInfo>('model.load', {
      path,
      ...(policy ? { policy } : {}),
    });
  }

  async unload(): Promise<void> {
    await this.request<null>('model.unload');
  }

  /**
   * Token count from the loaded model's own tokenizer.
   *
   * Messages are counted through the chat template rather than concatenated,
   * so the count includes the role markers the model actually receives.
   */
  async tokenize(
    input: { text: string } | { messages: ReadonlyArray<PythonChatMessage> },
  ): Promise<number> {
    const result = await this.request<{ tokens: number }>('model.tokenize', { ...input });
    return result.tokens;
  }

  async chat(
    messages: ReadonlyArray<PythonChatMessage>,
    request: GenerateRequest = {},
  ): Promise<PythonGenerationResult> {
    return await this.generate('generate.chat', { messages }, request);
  }

  async complete(
    prompt: string,
    request: GenerateRequest = {},
  ): Promise<PythonGenerationResult> {
    return await this.generate('generate.complete', { prompt }, request);
  }

  /**
   * Asks the worker to stop the current generation.
   *
   * `generate` blocks inside PyTorch, so the worker can only stop at a token
   * boundary. The in-flight promise then rejects with `generation_cancelled`.
   *
   * The cancel names the generation it means. Without that, a cancel racing a
   * generation that has already finished would stop whichever one started next.
   */
  cancel(): void {
    if (this.exited || this.activeGenerationId === undefined) {
      return;
    }
    this.write({
      id: this.nextId++,
      method: 'generation.cancel',
      params: { requestId: this.activeGenerationId },
    });
  }

  dispose(): void {
    this.rejectAll(new PythonWorkerError('worker_crashed', 'Worker was shut down.'));
    this.transport.close();
  }

  // ------------------------------------------------------------------ internals

  private async generate(
    method: string,
    params: Record<string, unknown>,
    request: GenerateRequest,
  ): Promise<PythonGenerationResult> {
    const id = this.nextId;
    this.activeGenerationId = id;
    try {
      return await this.request<PythonGenerationResult>(
        method,
        {
          ...params,
          stream: Boolean(request.onToken),
          ...(request.options ? { options: request.options } : {}),
        },
        request.onToken,
      );
    } finally {
      if (this.activeGenerationId === id) {
        this.activeGenerationId = undefined;
      }
    }
  }

  private request<T>(
    method: string,
    params: Record<string, unknown> = {},
    onToken?: (text: string) => void,
  ): Promise<T> {
    if (this.exited) {
      return Promise.reject(
        new PythonWorkerError('worker_crashed', 'Worker is not running.'),
      );
    }
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, {
        resolve: resolve as (value: unknown) => void,
        reject,
        ...(onToken ? { onToken } : {}),
      });
      this.write({ id, method, params });
    });
  }

  private write(message: unknown): void {
    this.transport.send(`${JSON.stringify(message)}\n`);
  }

  private receive(line: string): void {
    const trimmed = line.trim();
    if (!trimmed) {
      return;
    }
    let message: Record<string, unknown>;
    try {
      message = JSON.parse(trimmed) as Record<string, unknown>;
    } catch {
      // A worker that writes to stdout outside the protocol desynchronises the
      // stream. Log and skip rather than tear the session down for one line.
      this.onLog?.(`Discarded unparseable worker output: ${trimmed.slice(0, 200)}`);
      return;
    }
    if (typeof message['method'] === 'string') {
      this.handleNotification(message['method'], message['params']);
      return;
    }
    this.handleResponse(message);
  }

  private handleNotification(method: string, rawParams: unknown): void {
    const params = (rawParams ?? {}) as Record<string, unknown>;
    if (method === 'generation.token') {
      const id = params['requestId'];
      const text = params['text'];
      if (typeof id === 'number' && typeof text === 'string') {
        this.pending.get(id)?.onToken?.(text);
      }
      return;
    }
    if (method === 'worker.state') {
      const state = params['state'];
      const detail = params['detail'];
      if (typeof state === 'string') {
        this.onState?.(
          state as WorkerLifecycleState,
          typeof detail === 'string' ? detail : null,
        );
      }
      return;
    }
    this.onLog?.(`Ignored unknown worker notification: ${method}`);
  }

  private handleResponse(message: Record<string, unknown>): void {
    const id = message['id'];
    if (typeof id !== 'number') {
      // A failure the worker could not attribute to a request, such as a
      // malformed line. Nothing is pending on it; record it and move on.
      this.onLog?.(`Worker reported an unattributed error: ${JSON.stringify(message)}`);
      return;
    }
    const pending = this.pending.get(id);
    if (!pending) {
      return;
    }
    this.pending.delete(id);
    if (message['ok'] === true) {
      pending.resolve(message['result']);
      return;
    }
    const error = (message['error'] ?? {}) as Record<string, unknown>;
    const code = typeof error['code'] === 'string'
      ? (error['code'] as PythonErrorCode)
      : 'internal_error';
    const text = typeof error['message'] === 'string'
      ? error['message']
      : 'The worker reported an error with no message.';
    pending.reject(new PythonWorkerError(code, text));
  }

  private handleExit(code: number | null, signal: string | null): void {
    this.exited = true;
    const detail = signal ? `signal ${signal}` : `exit code ${code ?? 'unknown'}`;
    this.rejectAll(new PythonWorkerError(
      'worker_crashed',
      `The local model worker stopped (${detail}). It can be restarted.`,
    ));
    this.onState?.('STOPPED', detail);
  }

  /** Fails every in-flight request so no caller waits on a dead worker. */
  private rejectAll(error: Error): void {
    for (const pending of [...this.pending.values()]) {
      pending.reject(error);
    }
    this.pending.clear();
  }
}
