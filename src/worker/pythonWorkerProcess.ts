/**
 * Spawns the Safetensors runtime worker and adapts its pipes to a transport.
 *
 * The worker's contract is that stdout carries protocol traffic only and every
 * diagnostic goes to stderr, so the two streams are handled separately here:
 * stdout is framed into lines, stderr is forwarded to the logger untouched.
 */

import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import * as readline from 'node:readline';
import type { WorkerTransport } from './pythonWorkerClient.ts';

export interface SpawnPythonWorkerOptions {
  /** Interpreter from the extension's provisioned environment, never bare `python`. */
  pythonPath: string;
  /** Directory holding the `runtime` package. */
  runtimeDirectory: string;
  onLog?: (message: string) => void;
  env?: NodeJS.ProcessEnv;
}

export interface SpawnedPythonWorker extends WorkerTransport {
  readonly pid: number | undefined;
}

export function spawnPythonWorkerTransport(
  options: SpawnPythonWorkerOptions,
): SpawnedPythonWorker {
  const child: ChildProcessWithoutNullStreams = spawn(
    options.pythonPath,
    ['-m', 'runtime.worker', '--worker'],
    {
      cwd: options.runtimeDirectory,
      stdio: ['pipe', 'pipe', 'pipe'],
      env: {
        ...(options.env ?? process.env),
        // Unbuffered stdio, so a response is never held back in a pipe buffer
        // while the extension waits on it.
        PYTHONUNBUFFERED: '1',
        // The worker package is imported from its own directory rather than
        // from whatever the interpreter's site-packages happens to contain.
        PYTHONPATH: options.runtimeDirectory,
      },
    },
  );

  const stdout = readline.createInterface({ input: child.stdout });
  const stderr = readline.createInterface({ input: child.stderr });
  stderr.on('line', (line) => options.onLog?.(`[worker] ${line}`));

  let exited = false;
  const emitExit = (code: number | null, signal: string | null) => {
    if (exited) {
      return;
    }
    exited = true;
    stdout.close();
    stderr.close();
    exitHandlers.forEach((handler) => handler(code, signal));
  };
  const exitHandlers: Array<(code: number | null, signal: string | null) => void> = [];

  child.on('exit', emitExit);
  // A spawn failure never produces 'exit', so pending requests would hang
  // without this. It is reported as an exit so one recovery path covers both.
  child.on('error', (error) => {
    options.onLog?.(`[worker] failed to start: ${error.message}`);
    emitExit(null, null);
  });

  return {
    pid: child.pid,
    send: (line) => {
      if (!exited && child.stdin.writable) {
        child.stdin.write(line);
      }
    },
    onLine: (handler) => { stdout.on('line', handler); },
    onExit: (handler) => { exitHandlers.push(handler); },
    close: () => {
      if (exited) {
        return;
      }
      // Closing stdin is the worker's documented shutdown signal; it unloads
      // the model and exits on its own. SIGKILL is the fallback if it does not.
      child.stdin.end();
      const killTimer = setTimeout(() => child.kill('SIGKILL'), 5_000);
      killTimer.unref?.();
      child.once('exit', () => clearTimeout(killTimer));
    },
  };
}
