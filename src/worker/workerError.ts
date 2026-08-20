import { describeError } from '../errorDetail.ts';

export class LocalWorkerFatalError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'LocalWorkerFatalError';
  }
}

/**
 * A request that never reached a response.
 *
 * Node reports these as a bare "fetch failed", which says nothing about which
 * endpoint died or why. The endpoint and the full cause chain go in the message
 * so the Output channel is enough to diagnose it.
 */
export function workerTransportError(path: string, error: unknown): Error {
  return new Error(
    `Local worker request ${path} never completed: ${describeError(error)}.`,
    { cause: error },
  );
}

export function workerRequestError(path: string, status: number, body: string): Error {
  const base = `Local worker request ${path} failed with HTTP ${status}`;
  const detail = errorDetail(body);
  if (!detail) {
    return new Error(`${base}.`);
  }
  return classifiedWorkerError(
    `${base}: ${detail}${/[.!?]$/.test(detail) ? '' : '.'}`,
    detail,
  );
}

export function workerStreamError(path: string, payload: unknown): Error {
  const base = `Local worker stream ${path} failed`;
  const detail = errorDetail(JSON.stringify(payload));
  if (!detail) {
    return new Error(`${base}.`);
  }
  return classifiedWorkerError(
    `${base}: ${detail}${/[.!?]$/.test(detail) ? '' : '.'}`,
    detail,
  );
}

export function isFatalWorkerError(error: unknown): error is LocalWorkerFatalError {
  return error instanceof LocalWorkerFatalError;
}

function classifiedWorkerError(message: string, detail: string): Error {
  return /(?:compute error|failed to compute|failed to decode|backend is in error state|out of memory|failed to allocate)/i.test(detail)
    ? new LocalWorkerFatalError(message)
    : new Error(message);
}

function errorDetail(body: string): string {
  let detail = '';
  try {
    const payload = JSON.parse(body) as unknown;
    if (isRecord(payload)) {
      if (isRecord(payload.error) && typeof payload.error.message === 'string') {
        detail = payload.error.message;
      } else if (typeof payload.error === 'string') {
        detail = payload.error;
      } else if (typeof payload.message === 'string') {
        detail = payload.message;
      }
    }
  } catch {
    detail = body;
  }
  return detail.replace(/\s+/g, ' ').trim().slice(0, 500);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
