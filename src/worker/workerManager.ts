import { randomBytes } from 'node:crypto';
import { execFile, spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { constants as fsConstants } from 'node:fs';
import { access, mkdir, rm, stat, writeFile } from 'node:fs/promises';
import * as net from 'node:net';
import * as vscode from 'vscode';
import { readConfig } from '../config.js';
import type { InstalledModel, WorkerState } from '../domain.js';
import type { LocalLlmLogger } from '../logging.js';
import { abortError, InferenceScheduler, type InferenceKind } from './inferenceScheduler.js';
import type { InferenceClient } from './inferenceClient.js';
import { LlamaClient } from './llamaClient.js';
import {
  exitNotifier,
  runtimeDisplayName,
  runtimeForModel,
  type ExitNotifier,
  type RuntimeExit,
  type RuntimeSession,
} from './runtimeSession.js';
import { probePythonRuntime, startTransformersSession } from './transformersBackend.js';
import { ensureEnvironment, removeStaleEnvironments } from './pythonProvision.js';
import { resolveEnvTarget } from './pythonEnvironment.js';
import { describeError } from '../errorDetail.js';
import { SYCL_INITIAL_FIT_TARGET_MIB, isSyclDeviceOutOfMemory, nextSyclFitTargetMiB, parseFittedContext, parseFreeDeviceMemoryMiB, resolveFitTargetMiB } from './memoryFit.js';
import { isFatalWorkerError } from './workerError.js';
import { verifiedWorkerBundle } from './workerIntegrity.js';
import { prepareWorkerLaunch } from './workerLaunch.js';
import type { WorkerBackend } from './workerManifest.js';
import { createWorkerDiagnostics } from './workerDiagnostics.js';
import { beginWorkerActivity, finishWorkerActivity } from './workerActivity.js';
import { discoverSycl0 } from './syclDevice.js';

const STOP_TIMEOUT_MS = 5_000;
const HEALTH_INTERVAL_MS = 500;
const MAX_RESTARTS = 3;
const RESTART_WINDOW_MS = 5 * 60_000;

interface MemoryAttempt {
  outOfMemory: boolean;
  canRetry: boolean;
}

class RetrySyclMemoryError extends Error {}

export class WorkerManager implements vscode.Disposable {
  private readonly stateEmitter = new vscode.EventEmitter<WorkerState>();
  /**
   * The one runtime that may be resident.
   *
   * A single slot rather than one per engine: a 30B checkpoint and a 30B GGUF
   * would not fit in memory together, so starting either has to evict the
   * other. That is also why no second scheduler was needed.
   */
  private session: RuntimeSession | undefined;
  private currentModel: InstalledModel | undefined;
  private apiKeyFile: string | undefined;
  private startPromise: Promise<RuntimeSession> | undefined;
  private requestedStop = false;
  private restartTimes: number[] = [];
  private disposed = false;
  private generation = 0;
  private lifecycleTail: Promise<void> = Promise.resolve();
  private workerState: WorkerState = { kind: 'stopped' };
  private readonly scheduler = new InferenceScheduler();
  private readonly syclFitTargets = new Map<string, number>();

  readonly onDidChangeState = this.stateEmitter.event;

  constructor(
    private readonly context: vscode.ExtensionContext,
    private readonly logger: LocalLlmLogger,
  ) {}

  get state(): WorkerState {
    return this.workerState;
  }

  get currentModelId(): string | undefined {
    return this.currentModel?.id;
  }

  canRunInline(modelId: string): boolean {
    if (this.scheduler.activeKind === 'chat' || this.scheduler.hasQueuedChat) {
      return false;
    }
    if (
      (this.workerState.kind === 'ready' || this.workerState.kind === 'starting') &&
      this.currentModel?.id !== modelId
    ) {
      return false;
    }
    return true;
  }

  async run<T>(
    model: InstalledModel,
    kind: InferenceKind,
    operation: (client: InferenceClient, signal: AbortSignal) => Promise<T>,
    signal?: AbortSignal,
  ): Promise<T> {
    return this.scheduler.run(
      kind,
      async (scheduledSignal) => {
        const session = await this.ensureReady(model, scheduledSignal);
        const client = session.client;
        const activeState = beginWorkerActivity(this.workerState, kind);
        const generatingResponse = activeState !== this.workerState;
        if (generatingResponse) {
          this.setState(activeState);
        }
        try {
          return await operation(client, scheduledSignal);
        } catch (error) {
          if (isFatalWorkerError(error)) {
            this.logger.error(
              'Local worker reported a fatal streaming error; restarting it before the next request.',
              error,
            );
            try {
              await this.withLifecycle(async () => {
                if (this.session === session) {
                  await this.stopProcess();
                }
              });
            } catch (stopError) {
              this.logger.error('Failed to stop the unusable local worker', stopError);
            }
          }
          throw error;
        } finally {
          if (generatingResponse && this.session === session) {
            const readyState = finishWorkerActivity(this.workerState);
            if (readyState !== this.workerState) {
              this.setState(readyState);
            }
          }
        }
      },
      signal,
    );
  }

  private async ensureReady(
    model: InstalledModel,
    signal?: AbortSignal,
  ): Promise<RuntimeSession> {
    return this.withLifecycle(async () => {
      if (
        this.session &&
        !this.session.exited &&
        this.currentModel?.id === model.id &&
        this.workerState.kind === 'ready'
      ) {
        return this.session;
      }
      if (this.startPromise && this.currentModel?.id === model.id) {
        return this.startPromise;
      }
      const switchingModel = this.currentModel?.id !== model.id;
      await this.stopProcess();
      if (switchingModel) {
        this.restartTimes = [];
      }
      this.currentModel = model;
      const promise = this.start(model, signal);
      this.startPromise = promise;
      try {
        return await promise;
      } finally {
        if (this.startPromise === promise) {
          this.startPromise = undefined;
        }
      }
    });
  }

  async stop(): Promise<void> {
    // Invalidate delayed crash restarts immediately, before waiting for lifecycle cleanup.
    this.generation += 1;
    this.scheduler.cancelAll();
    await this.withLifecycle(() => this.stopProcess());
  }

  /**
   * Ends whichever runtime is resident.
   *
   * How a process is asked to stop belongs to its session: llama.cpp takes a
   * signal, the Safetensors worker takes a closed stdin. Everything around
   * that — the state transitions, the API key file, the generation counter —
   * is the same either way and stays here.
   */
  private async stopProcess(): Promise<void> {
    this.generation += 1;
    const session = this.session;
    if (!session) {
      await this.removeApiKeyFile();
      if (this.workerState.kind !== 'stopped') {
        this.setState({ kind: 'stopped' });
      }
      return;
    }

    this.requestedStop = true;
    this.session = undefined;
    const modelId = this.currentModel?.id ?? 'unknown';
    this.setState({ kind: 'stopping', modelId, runtime: session.runtime });
    try {
      await session.stop();
    } catch (error) {
      this.logger.error('Failed to stop the local worker cleanly', error);
    }
    await session.client.dispose();
    await this.removeApiKeyFile();
    this.requestedStop = false;
    this.setState({ kind: 'stopped' });
  }

  dispose(): void {
    this.disposed = true;
    void this.stop();
    this.stateEmitter.dispose();
  }

  /**
   * Starts the engine this model needs.
   *
   * The routing decision is the model's own: a GGUF file goes to llama.cpp, a
   * Safetensors checkpoint to the Python runtime. Nothing above this method
   * knows which one ran.
   */
  private async start(
    model: InstalledModel,
    signal?: AbortSignal,
  ): Promise<RuntimeSession> {
    return runtimeForModel(model) === 'transformers'
      ? await this.startTransformers(model, signal)
      : await this.startLlama(model, signal);
  }

  private async startLlama(
    model: InstalledModel,
    signal?: AbortSignal,
  ): Promise<RuntimeSession> {
    const generation = this.generation;
    for (;;) {
      throwIfAborted(signal);
      if (this.disposed || generation !== this.generation) {
        throw new vscode.CancellationError();
      }
      try {
        return await this.startAttempt(model, signal);
      } catch (error) {
        if (!(error instanceof RetrySyclMemoryError)) {
          throw error;
        }
      }
    }
  }

  /**
   * Starts the Safetensors runtime.
   *
   * Short next to the llama.cpp path because the work it would otherwise do
   * is not applicable: there is no bundle to verify, no loopback port to
   * allocate, no API key to write, and no health poll, because the worker
   * answers `model.load` only once the weights are actually resident.
   */
  /**
   * Provisions the Python env behind a cancellable notification, then blesses
   * it with the throwaway probe before any worker spawns from it.
   */
  private async provisionPythonEnv(
    runtimeDirectory: string,
    signal?: AbortSignal,
  ): Promise<string> {
    throwIfAborted(signal);
    const config = readConfig(this.context);
    const target = resolveEnvTarget();
    const storagePath = this.context.globalStorageUri.fsPath;
    const env = await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: 'Installing Python environment for Safetensors models',
        cancellable: true,
      },
      async (progress, token) =>
        ensureEnvironment({
          storagePath,
          target,
          flavorSetting: config.pythonEnvFlavor,
          releaseManifestPath: `${runtimeDirectory}/env-manifest.json`,
          progress,
          token,
          onLog: (message) => this.logWorkerOutput(message),
          probe: (pythonPath) => probePythonRuntime(pythonPath, runtimeDirectory),
        }),
    );
    throwIfAborted(signal);
    await removeStaleEnvironments(storagePath, { target: env.target, flavor: env.flavor });
    return env.pythonPath;
  }

  private async startTransformers(
    model: InstalledModel,
    signal?: AbortSignal,
  ): Promise<RuntimeSession> {
    const startGeneration = this.generation;
    const startedAt = Date.now();
    this.requestedStop = false;
    this.setState({ kind: 'starting', modelId: model.id, runtime: 'transformers' });
    try {
      throwIfAborted(signal);
      const config = readConfig(this.context);
      const runtimeDirectory = safetensorsRuntimeDirectory(this.context);
      // An explicit interpreter always wins; otherwise provision on first use.
      const pythonPath = config.pythonPath.trim() ||
        (await this.provisionPythonEnv(runtimeDirectory, signal));
      const session = await startTransformersSession({
        model,
        pythonPath,
        runtimeDirectory,
        onLog: (message) => this.logWorkerOutput(message),
        ...(signal ? { signal } : {}),
      });
      throwIfAborted(signal);
      if (startGeneration !== this.generation || this.disposed) {
        await session.stop();
        await session.client.dispose();
        throw new vscode.CancellationError();
      }
      this.adopt(session, model, { outOfMemory: false, canRetry: false });
      this.logger.info(
        `[Model Loading] Complete: ${model.name}; runtime=transformers; ` +
        `startupElapsed=${Date.now() - startedAt} ms.`,
      );
      return session;
    } catch (error) {
      const cancelled = signal?.aborted || startGeneration !== this.generation || this.disposed;
      if (cancelled) {
        if (startGeneration === this.generation && !this.disposed) {
          this.setState({ kind: 'stopped' });
        }
        throw new vscode.CancellationError();
      }
      const message = error instanceof Error ? error.message : String(error);
      this.setState({ kind: 'failed', modelId: model.id, runtime: 'transformers', message });
      throw error;
    }
  }

  /**
   * Takes ownership of a started session and announces it as ready.
   *
   * Registering the exit handler here rather than in each backend keeps the
   * crash-and-restart policy in one place, and `onExit` fires immediately for
   * a process that died before this ran, so the window between starting and
   * being watched cannot swallow a crash.
   */
  private adopt(
    session: RuntimeSession,
    model: InstalledModel,
    memoryAttempt: MemoryAttempt,
  ): void {
    this.session = session;
    this.setState({
      kind: 'ready',
      modelId: model.id,
      runtime: session.runtime,
      ...(session.port === undefined ? {} : { port: session.port }),
    });
    session.onExit((exit) => void this.handleExit(session, exit, memoryAttempt));
  }

  private async startAttempt(
    model: InstalledModel,
    signal?: AbortSignal,
  ): Promise<RuntimeSession> {
    const startGeneration = this.generation;
    const startedAt = Date.now();
    let child: ChildProcessWithoutNullStreams | undefined;
    let client: LlamaClient | undefined;
    let closed: Promise<void> | undefined;
    const memoryAttempt: MemoryAttempt = { outOfMemory: false, canRetry: false };
    this.requestedStop = false;
    this.setState({ kind: 'starting', modelId: model.id, runtime: 'llama-cpp' });
    try {
      throwIfAborted(signal);
      const config = readConfig(this.context);
      const target = `${process.platform}-${process.arch}`;
      const launch = await prepareWorkerLaunch({
        target,
        mode: config.acceleration,
        resolveBundle: (workerTarget, mode) => verifiedWorkerBundle(
          this.context.extensionUri.fsPath,
          workerTarget,
          mode,
        ),
        discoverSycl: discoverSycl0,
      });
      const executable = launch.bundle.executablePath;
      const backend = launch.backend;
      const fitKey = JSON.stringify([
        model.id, model.filePath, model.fileSize, model.fileModifiedAt,
        config.contextSize, config.batchSize, config.microBatchSize,
      ]);
      const syclFitTargetMiB = this.syclFitTargets.get(fitKey) ?? SYCL_INITIAL_FIT_TARGET_MIB;
      if (launch.syclDevice) {
        this.logger.info(
          `[Model Loading] Windows SYCL preflight passed: ${launch.syclDevice.id} `
          + `(${launch.syclDevice.description}).`,
        );
      }
      await access(executable, fsConstants.X_OK).catch(() => {
        throw new Error(
          `Bundled local worker is missing or not executable: ${executable}. Install the VSIX for this operating system and architecture.`,
        );
      });
      await access(model.filePath, fsConstants.R_OK).catch(() => {
        throw new Error(`Model file is missing or unreadable: ${model.filePath}`);
      });
      const modelMetadata = await stat(model.filePath);
      if (
        modelMetadata.size !== model.fileSize ||
        (model.fileModifiedAt !== undefined && modelMetadata.mtimeMs !== model.fileModifiedAt)
      ) {
        throw new Error(
          'The model file changed after it was registered. Reload VS Code to reverify it before local execution.',
        );
      }
      throwIfAborted(signal);

      const port = await allocateLoopbackPort();
      throwIfAborted(signal);
      const apiKey = randomBytes(32).toString('hex');
      const apiKeyFile = await this.createApiKeyFile(apiKey);
      throwIfAborted(signal);
      const orphanBytes = await orphanWorkerMemoryBytes(
        executable,
        (warning) => this.logger.error(warning),
      );
      if (orphanBytes) {
        this.logger.info(
          `Another local worker still holds ${Math.round(orphanBytes / (1024 * 1024))} MiB; reserving that memory as well.`,
        );
      }
      const args = buildWorkerArguments(
        model.filePath,
        port,
        apiKeyFile,
        config,
        backend,
        orphanBytes,
        syclFitTargetMiB,
      );
      this.logger.info(
        `[Model Loading] Starting local worker: target=${target} bundle=${launch.bundle.bundleName} ` +
        `backend=${backend} model=${model.name} context=${config.contextSize || 'auto'} ` +
        `batch=${config.batchSize}/${config.microBatchSize} threads=${config.cpuThreads || 'auto'}.`,
      );
      if (backend === 'sycl') {
        this.logger.info(
          `[Model Loading] SYCL automatic GPU layers; execution reserve=${syclFitTargetMiB} MiB; warmup enabled.`,
        );
      }
      child = spawn(executable, args, {
        cwd: pathDirectory(executable),
        env: launch.environment,
        shell: false,
        windowsHide: true,
        stdio: ['pipe', 'pipe', 'pipe'],
      });
      child.stdin.end();
      this.logger.info(
        `[Model Loading] Local worker spawned: pid=${child.pid ?? 'unknown'} backend=${backend}. Waiting for health.`,
      );
      child.stdout.setEncoding('utf8');
      child.stderr.setEncoding('utf8');
      const observeOutput = (): ((data: string) => void) => {
        let tail = '';
        return (data) => {
          this.logWorkerOutput(data);
          if (backend !== 'sycl' || memoryAttempt.outOfMemory) {
            return;
          }
          tail = (tail + data).slice(-8192);
          if (isSyclDeviceOutOfMemory(tail)) {
            memoryAttempt.outOfMemory = true;
            const next = nextSyclFitTargetMiB(syclFitTargetMiB);
            if (next !== undefined) {
              memoryAttempt.canRetry = true;
              this.syclFitTargets.set(fitKey, next);
              this.logger.info(
                `[Model Loading] SYCL device memory exhausted; next load reserves ${next} MiB `
                + 'and refits GPU layers. This is a resource failure, not an invalid model.',
              );
            }
          }
        };
      };
      child.stdout.on('data', observeOutput());
      child.stderr.on('data', observeOutput());
      const spawnError = new Promise<never>((_resolve, reject) => {
        child?.once('error', (error) => {
          this.logger.error('Local worker process error', error);
          reject(error);
        });
      });
      const exits = exitNotifier();
      closed = new Promise<void>((resolve) => {
        child?.once('close', (code, exitSignal) => {
          resolve();
          exits.notify({ code, signal: exitSignal });
        });
      });

      client = new LlamaClient(
        `http://127.0.0.1:${port}`,
        apiKey,
        createWorkerDiagnostics(this.logger),
      );
      await Promise.race([
        this.waitUntilHealthy(client, child, signal),
        spawnError,
      ]);
      throwIfAborted(signal);
      if (startGeneration !== this.generation || this.disposed) {
        throw new vscode.CancellationError();
      }

      const session = llamaSession(
        child, client, port, exits, (message: string) => this.logger.info(message),
      );
      this.adopt(session, model, memoryAttempt);
      this.logger.info(
        `[Model Loading] Complete: ${model.name}; backend=${backend}; ` +
        `startupElapsed=${Date.now() - startedAt} ms.`,
      );
      return session;
    } catch (error) {
      if (child && this.session?.client !== client) {
        child.kill('SIGKILL');
      }
      const processClosed = closed ? await waitForClose(closed, STOP_TIMEOUT_MS) : true;
      await client?.dispose();
      await this.removeApiKeyFile();
      const cancelled = signal?.aborted || startGeneration !== this.generation || this.disposed;
      if (cancelled) {
        if (startGeneration === this.generation && !this.disposed) {
          this.setState({ kind: 'stopped' });
        }
        throw new vscode.CancellationError();
      }
      if (memoryAttempt.outOfMemory) {
        if (memoryAttempt.canRetry && processClosed) {
          throw new RetrySyclMemoryError('Retrying SYCL loading with more execution headroom.');
        }
        const message = processClosed
          ? 'SYCL device memory exhausted after bounded memory fitting. The model is not marked invalid. '
            + 'Reduce context or batch size, free memory, or explicitly select CPU acceleration.'
          : 'SYCL device memory exhausted and worker termination was not confirmed; automatic retry stopped.';
        this.setState({ kind: 'failed', modelId: model.id, runtime: 'llama-cpp', message });
        throw new Error(message, { cause: error });
      }
      const message = error instanceof Error ? error.message : String(error);
      this.setState({ kind: 'failed', modelId: model.id, runtime: 'llama-cpp', message });
      throw error;
    }
  }

  private async waitUntilHealthy(
    client: LlamaClient,
    child: ChildProcessWithoutNullStreams,
    signal?: AbortSignal,
  ): Promise<void> {
    const timeout = readConfig(this.context).startupTimeoutMilliseconds;
    const deadline = Date.now() + timeout;
    while (Date.now() < deadline) {
      if (signal?.aborted) {
        throw abortError();
      }
      if (child.exitCode !== null || child.signalCode !== null || child.killed) {
        throw new Error(`Local worker exited while loading the model (code ${child.exitCode ?? 'unknown'}).`);
      }
      if (await client.health(signal)) {
        return;
      }
      await delay(HEALTH_INTERVAL_MS, signal);
    }
    throw new Error(
      `Timed out after ${Math.round(timeout / 1_000)} seconds while loading the local model.`,
    );
  }

  /**
   * Recovers from a runtime that went away on its own.
   *
   * Identical for both engines, which is why it takes a session rather than a
   * child process: what differs between them is how a process is started and
   * stopped, not what an unexpected exit means.
   */
  private async handleExit(
    session: RuntimeSession,
    exit: RuntimeExit,
    memoryAttempt: MemoryAttempt,
  ): Promise<void> {
    if (this.session !== session || this.workerState.kind === 'starting') {
      return;
    }
    const exitGeneration = this.generation;
    const wasReady = this.workerState.kind === 'ready';
    const wasRequestedStop = this.requestedStop;
    const runtime = session.runtime;
    this.session = undefined;
    await session.client.dispose();
    await this.removeApiKeyFile();
    if (wasRequestedStop || this.disposed) {
      return;
    }

    const model = this.currentModel;
    const message = memoryAttempt.outOfMemory
      ? 'SYCL device memory exhausted. The interrupted request was not replayed. '
        + (memoryAttempt.canRetry
          ? 'The next load will refit GPU layers with more execution headroom.'
          : 'Automatic memory retries exhausted; reduce context or batch size, free memory, or select CPU acceleration.')
      : `The ${runtimeDisplayName(runtime)} worker exited unexpectedly `
        + `(code ${exit.code ?? 'none'}, signal ${exit.signal ?? 'none'}).`;
    this.logger.error(message, undefined, true);
    if (!model) {
      this.setState({ kind: 'failed', runtime, message });
      return;
    }
    this.setState({ kind: 'failed', modelId: model.id, runtime, message });
    if (!wasReady || (memoryAttempt.outOfMemory && !memoryAttempt.canRetry)) {
      return;
    }
    // Only llama.cpp is restarted eagerly. Its worker starts in seconds and a
    // crash is often transient, so keeping it warm is worth the attempt. A
    // Safetensors worker that died while ready has almost always exhausted
    // memory, and reloading it costs a digest pass, an interpreter probe and
    // tens of gigabytes of weights — three times over five minutes, arriving
    // at the same failure. The model is not stranded: the next request starts
    // a fresh session through `ensureReady`.
    if (runtime === 'transformers') {
      return;
    }

    const cutoff = Date.now() - RESTART_WINDOW_MS;
    this.restartTimes = this.restartTimes.filter((time) => time >= cutoff);
    if (this.restartTimes.length >= MAX_RESTARTS) {
      this.setState({
        kind: 'failed', modelId: model.id, runtime,
        message: `${message} Restart limit reached.`,
      });
      return;
    }
    this.restartTimes.push(Date.now());
    try {
      await delay(500);
      if (this.generation !== exitGeneration) {
        return;
      }
      if (this.startPromise) {
        try {
          await this.startPromise;
        } catch {
          // The replacement starts below after the failed start has settled.
        }
      }
      if (
        this.generation !== exitGeneration ||
        this.currentModel?.id !== model.id ||
        this.disposed ||
        this.requestedStop
      ) {
        return;
      }
      await this.run(model, 'utility', async () => undefined);
    } catch (error) {
      this.logger.error('Automatic local worker restart failed', error, true);
    }
  }

  private logWorkerOutput(data: string): void {
    const fitted = parseFittedContext(data);
    if (fitted) {
      const budget = parseFreeDeviceMemoryMiB(data);
      this.logger.info(
        `[Model Loading] llama.cpp reduced the context window from ${fitted.trainedContextSize} to ${fitted.fittedContextSize} tokens to fit this computer${budget ? ` (it sees ${budget} MiB of device memory)` : ''}.`,
      );
    }
    for (const line of data.split(/\r?\n/)) {
      if (line.trim()) {
        this.logger.debug(`worker: ${line.trim()}`);
      }
    }
  }

  private setState(state: WorkerState): void {
    this.workerState = state;
    this.stateEmitter.fire(state);
  }

  private async createApiKeyFile(apiKey: string): Promise<string> {
    await mkdir(this.context.globalStorageUri.fsPath, { recursive: true });
    await this.removeApiKeyFile();
    const filePath = `${this.context.globalStorageUri.fsPath}/worker-${randomBytes(12).toString('hex')}.key`;
    await writeFile(filePath, `${apiKey}\n`, { encoding: 'utf8', mode: 0o600 });
    this.apiKeyFile = filePath;
    return filePath;
  }

  private async removeApiKeyFile(): Promise<void> {
    const filePath = this.apiKeyFile;
    this.apiKeyFile = undefined;
    if (filePath) {
      // A worker API key left on disk is a real problem, not a detail to swallow.
      await rm(filePath, { force: true }).catch((error: unknown) => {
        this.logger.error(`Failed to delete the worker API key file ${filePath}`, error);
      });
    }
  }

  private async withLifecycle<T>(operation: () => Promise<T>): Promise<T> {
    const previous = this.lifecycleTail;
    let release: (() => void) | undefined;
    this.lifecycleTail = new Promise<void>((resolve) => {
      release = resolve;
    });
    await previous.catch(() => undefined);
    try {
      return await operation();
    } finally {
      release?.();
    }
  }
}

/**
 * Wraps a healthy llama-server child as a session.
 *
 * Stopping is a signal escalation rather than a protocol message: llama-server
 * has no shutdown endpoint, so SIGTERM with a bounded SIGKILL behind it is the
 * only way to end it.
 */
function llamaSession(
  child: ChildProcessWithoutNullStreams,
  client: LlamaClient,
  port: number,
  exits: ExitNotifier,
  onInfo: (message: string) => void,
): RuntimeSession {
  return {
    runtime: 'llama-cpp',
    client,
    port,
    get exited() {
      return exits.exited;
    },
    onExit: exits.onExit,
    stop: async () => {
      if (child.exitCode !== null || child.signalCode !== null) {
        return;
      }
      child.kill('SIGTERM');
      if (!await waitForExit(child, STOP_TIMEOUT_MS)) {
        onInfo('Local worker did not stop gracefully; forcing termination.');
        child.kill('SIGKILL');
        await waitForExit(child, 2_000);
      }
    },
  };
}

/**
 * Where the Safetensors runtime package lives inside the installed extension.
 *
 * Under `resources/` because `.vscodeignore` excludes `src/` and `scripts/`
 * from the VSIX; anything the runtime needs at execution time has to ship
 * from a directory that survives packaging.
 */
function safetensorsRuntimeDirectory(context: vscode.ExtensionContext): string {
  return `${context.extensionUri.fsPath}/resources/runtime`;
}

export function buildWorkerArguments(
  modelPath: string,
  port: number,
  apiKeyFile: string,
  config: import('../domain.js').WorkerConfig,
  backend: WorkerBackend,
  concurrentWorkerBytes?: number,
  syclFitTargetMiB = SYCL_INITIAL_FIT_TARGET_MIB,
): string[] {
  const batchSize = Math.max(32, Math.floor(config.batchSize));
  const microBatchSize = Math.max(
    32,
    Math.min(batchSize, Math.floor(config.microBatchSize)),
  );
  const args = [
    '--model',
    modelPath,
    '--host',
    '127.0.0.1',
    '--port',
    String(port),
    '--api-key-file',
    apiKeyFile,
    '--parallel',
    '1',
    '--batch-size',
    String(batchSize),
    '--ubatch-size',
    String(microBatchSize),
    '--jinja',
    '--no-webui',
  ];
  // Omitting --ctx-size entirely lets llama.cpp start from the model's trained
  // window and reduce it to fit this machine. Passing --ctx-size 0 would instead
  // set fit_params_min_ctx to UINT32_MAX and disable that reduction.
  if (config.contextSize > 0) {
    args.push('--ctx-size', String(config.contextSize));
  }
  switch (backend) {
    case 'metal':
      args.push(
        '--fit',
        'on',
        '--fit-target',
        String(resolveFitTargetMiB({
          reserveMiB: config.metalMemoryReserveMiB,
          ...(concurrentWorkerBytes === undefined ? {} : { concurrentWorkerBytes }),
        })),
      );
      break;
    case 'sycl':
      args.push(
        '--fit',
        'on',
        '--fit-target',
        String(syclFitTargetMiB),
        '--device',
        'SYCL0',
        '--split-mode',
        'none',
        '--main-gpu',
        '0',
      );
      break;
    case 'cpu':
      args.push(
        '--fit',
        'off',
        '--n-gpu-layers',
        '0',
        '--device',
        'none',
        '--no-op-offload',
      );
      break;
  }
  if (config.cpuThreads > 0) {
    args.push('--threads', String(config.cpuThreads));
  }
  return args;
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw abortError();
  }
}

async function waitForClose(closed: Promise<void>, timeoutMs: number): Promise<boolean> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      closed.then(() => true),
      new Promise<false>((resolve) => {
        timer = setTimeout(() => resolve(false), timeoutMs);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

async function allocateLoopbackPort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      if (!address || typeof address === 'string') {
        server.close();
        reject(new Error('Unable to allocate a loopback port.'));
        return;
      }
      const { port } = address;
      server.close((error) => (error ? reject(error) : resolve(port)));
    });
  });
}

async function waitForExit(
  child: ChildProcessWithoutNullStreams,
  timeoutMs: number,
): Promise<boolean> {
  if (child.exitCode !== null) {
    return true;
  }
  return new Promise((resolve) => {
    const timer = setTimeout(() => resolve(false), timeoutMs);
    child.once('exit', () => {
      clearTimeout(timer);
      resolve(true);
    });
  });
}

async function delay(milliseconds: number, signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) {
    throw abortError();
  }
  await new Promise<void>((resolve, reject) => {
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', cancel);
      resolve();
    }, milliseconds);
    const cancel = (): void => {
      clearTimeout(timer);
      reject(abortError());
    };
    signal?.addEventListener('abort', cancel, { once: true });
  });
}

function pathDirectory(filePath: string): string {
  const separator = Math.max(filePath.lastIndexOf('/'), filePath.lastIndexOf('\\'));
  return separator >= 0 ? filePath.slice(0, separator) : '.';
}


/**
 * Resident memory held by any llama.cpp server already running on this computer.
 *
 * A VS Code reload can leave a previous worker alive, and an earlier version of
 * this extension installs to a different path. llama.cpp measures free device
 * memory as its own Metal budget minus its own allocation, so every one of those
 * processes is invisible to it and their memory has to be reserved explicitly.
 *
 * Returns undefined when the platform offers no cheap way to ask.
 */
async function orphanWorkerMemoryBytes(
  executable: string,
  onWarning?: (message: string) => void,
): Promise<number | undefined> {
  const workerName = executable.slice(
    Math.max(executable.lastIndexOf('/'), executable.lastIndexOf('\\')) + 1,
  ) || executable;
  if (process.platform === 'win32') {
    return undefined;
  }
  try {
    const listing = await new Promise<string>((resolve, reject) => {
      execFile('ps', ['-axo', 'rss=,command='], { maxBuffer: 8 * 1024 * 1024 }, (error, stdout) => {
        if (error) {
          reject(error);
          return;
        }
        resolve(stdout);
      });
    });
    let bytes = 0;
    for (const line of listing.split(/\r?\n/)) {
      const match = /^\s*(\d+)\s+(.*)$/.exec(line);
      if (match?.[2]?.includes(workerName) && match[1]) {
        bytes += Number(match[1]) * 1024;
      }
    }
    return bytes > 0 ? bytes : undefined;
  } catch (error) {
    onWarning?.(`Could not measure memory held by other local workers: ${describeError(error)}`);
    return undefined;
  }
}
