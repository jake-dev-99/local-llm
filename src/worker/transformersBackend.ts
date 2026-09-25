/**
 * Starting the Safetensors runtime as a session `WorkerManager` can hold.
 *
 * Three steps, in an order that matters: verify the checkpoint is still the
 * one that was registered, probe the interpreter in a process whose death
 * costs nothing, then spawn the worker and load the model into it.
 */

import { execFile } from 'node:child_process';
import { access, stat } from 'node:fs/promises';
import { constants as fsConstants } from 'node:fs';
import { describeError } from '../errorDetail.ts';
import type { InstalledModel } from '../domain.ts';
import { readSafetensorsCheckpoint } from '../models/safetensorsDirectory.ts';
import { PythonWorkerClient } from './pythonWorkerClient.ts';
import { spawnPythonWorkerTransport } from './pythonWorkerProcess.ts';
import type { PythonRuntimeInfo, PythonRuntimePolicy } from './pythonWorkerTypes.ts';
import { exitNotifier, type RuntimeSession } from './runtimeSession.ts';
import { TransformersClient } from './transformersClient.ts';

/** Long enough for a cold torch import on a slow disk, short enough to fail. */
const PROBE_TIMEOUT_MS = 120_000;

export interface TransformersLaunchOptions {
  model: InstalledModel;
  /** Interpreter with the runtime's dependencies installed. */
  pythonPath: string;
  /** Directory holding the `runtime` package. */
  runtimeDirectory: string;
  policy?: PythonRuntimePolicy;
  onLog: (message: string) => void;
  signal?: AbortSignal;
}

export async function startTransformersSession(
  options: TransformersLaunchOptions,
): Promise<RuntimeSession> {
  const { model, pythonPath, runtimeDirectory, onLog } = options;
  throwIfAborted(options.signal);
  await assertCheckpointUnchanged(model, onLog);

  throwIfAborted(options.signal);
  const info = await probePythonRuntime(pythonPath, runtimeDirectory);
  onLog(
    `[Model Loading] Safetensors runtime ready: python ${info.pythonVersion} on ` +
    `${info.deviceType}${info.deviceName ? ` (${info.deviceName})` : ''}; ` +
    `transformers ${info.versions['transformers'] ?? 'unknown'}.`,
  );

  throwIfAborted(options.signal);
  const transport = spawnPythonWorkerTransport({
    pythonPath,
    runtimeDirectory,
    onLog,
  });
  const worker = new PythonWorkerClient({ transport, onLog });

  const exits = exitNotifier();
  transport.onExit((code, signal) => exits.notify({ code, signal }));

  let modelInfo;
  try {
    modelInfo = await worker.load(model.filePath, options.policy);
  } catch (error) {
    worker.dispose();
    throw error;
  }
  onLog(
    `[Model Loading] Complete: ${model.name}; ${modelInfo.architecture ?? 'unknown architecture'}; ` +
    `dtype=${modelInfo.dtype}; context=${modelInfo.contextLength ?? 'unknown'}.`,
  );

  const client = new TransformersClient(worker, modelInfo);
  return {
    runtime: 'transformers',
    client,
    // Nothing listens on a socket: the protocol runs over this process's pipes.
    port: undefined,
    get exited() {
      return exits.exited;
    },
    onExit: exits.onExit,
    stop: async () => {
      // Ends the process only. Closing stdin is the worker's shutdown signal:
      // it unloads the model and exits on its own, with a SIGKILL behind it if
      // it does not. Failing the in-flight requests belongs to the client's
      // own dispose, which the manager calls next.
      transport.close();
    },
  };
}

/**
 * Refuses to load a checkpoint whose manifest no longer matches.
 *
 * Compares the digest rather than modification times. Copying a checkpoint to
 * a faster disk rewrites every mtime and not one byte of the model, and an
 * mtime check would reject the result; the digest covers paths, sizes and
 * Safetensors headers, so it survives the move and still catches a swap.
 */
async function assertCheckpointUnchanged(
  model: InstalledModel,
  onLog: (message: string) => void,
): Promise<void> {
  await access(model.filePath, fsConstants.R_OK).catch(() => {
    throw new Error(`Model directory is missing or unreadable: ${model.filePath}`);
  });
  const metadata = await stat(model.filePath);
  if (!metadata.isDirectory()) {
    throw new Error(
      `${model.filePath} is no longer a directory. Reinstall this model to reverify it.`,
    );
  }
  const checkpoint = await readSafetensorsCheckpoint(model.filePath, onLog);
  if (checkpoint.identity.digest !== model.sha256) {
    throw new Error(
      'The checkpoint changed after it was registered. Reload VS Code to reverify it before local execution.',
    );
  }
}

/**
 * Asks a throwaway process whether this interpreter can run the runtime.
 *
 * Importing torch can abort the interpreter instead of raising — a duplicate
 * OpenMP runtime does exactly that, and no `except` clause catches it. Finding
 * that out in the worker would kill the worker before it could explain itself,
 * so the question is asked in a process that is allowed to die.
 */
export async function probePythonRuntime(
  pythonPath: string,
  runtimeDirectory: string,
): Promise<PythonRuntimeInfo> {
  if (!pythonPath.trim()) {
    throw new Error(
      'No Python interpreter is configured for Safetensors models. Set "localLlm.pythonPath" ' +
      'to an interpreter with torch, transformers, accelerate, safetensors and xgrammar installed ' +
      `(see ${runtimeDirectory}/requirements.txt).`,
    );
  }
  const stdout = await new Promise<string>((resolve, reject) => {
    execFile(
      pythonPath,
      ['-m', 'runtime.worker', '--probe'],
      {
        cwd: runtimeDirectory,
        timeout: PROBE_TIMEOUT_MS,
        env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONPATH: runtimeDirectory },
      },
      (error, out, errorOutput) => {
        if (error) {
          reject(new Error(
            `The configured Python interpreter cannot run the Safetensors runtime: ` +
            `${describeError(error)}${errorOutput ? `\n${errorOutput.trim().slice(-2000)}` : ''}`,
          ));
          return;
        }
        resolve(out);
      },
    );
  });
  return parseProbe(stdout);
}

function parseProbe(stdout: string): PythonRuntimeInfo {
  let parsed: unknown;
  try {
    parsed = JSON.parse(stdout.trim());
  } catch {
    throw new Error(
      `The Python runtime probe returned something that is not JSON: ${stdout.trim().slice(0, 500)}`,
    );
  }
  if (typeof parsed !== 'object' || parsed === null) {
    throw new Error('The Python runtime probe returned no runtime information.');
  }
  const info = parsed as PythonRuntimeInfo;
  if (!info.versions?.['transformers']) {
    throw new Error(
      'Transformers is not installed in the configured Python interpreter. Install the ' +
      'runtime requirements before loading a Safetensors model.',
    );
  }
  return info;
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    const error = new Error('Local inference was cancelled.');
    error.name = 'AbortError';
    throw error;
  }
}
