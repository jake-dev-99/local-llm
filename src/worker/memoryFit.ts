/**
 * llama.cpp's own default for `--fit-target`, the memory it leaves free per device.
 * Verified against common/common.h at the pinned llama.cpp revision, where
 * `fit_params_target` is initialised to 1024 * 1024*1024 bytes.
 */
export const LLAMA_CPP_DEFAULT_FIT_TARGET_MIB = 1024;

const MIB = 1024 * 1024;

export interface FitTargetInput {
  /**
   * Memory to leave free for the rest of the system, from
   * localLlm.metalMemoryReserveMiB.
   */
  reserveMiB: number;
  /**
   * Resident bytes held by another worker this extension still owns.
   *
   * llama.cpp measures free device memory as its Metal working-set budget minus
   * its own current allocation, so it cannot see a second worker at all. Adding
   * that worker's memory to the margin is the one system-level correction this
   * extension can make from a measurement rather than a guess.
   */
  concurrentWorkerBytes?: number;
}

/**
 * The `--fit-target` value to pass to llama-server.
 *
 * Never returns less than llama.cpp's own default, so automatic sizing is never
 * more aggressive than running llama-server with no flags at all.
 */
export function resolveFitTargetMiB(input: FitTargetInput): number {
  const reserve = Number.isFinite(input.reserveMiB) ? Math.floor(input.reserveMiB) : 0;
  const concurrent = Number.isFinite(input.concurrentWorkerBytes ?? 0)
    ? Math.ceil(Math.max(0, input.concurrentWorkerBytes ?? 0) / MIB)
    : 0;
  return Math.max(LLAMA_CPP_DEFAULT_FIT_TARGET_MIB, reserve + concurrent);
}

export interface FittedContext {
  trainedContextSize: number;
  fittedContextSize: number;
}

/**
 * Reads llama.cpp's own report of how it resized the context window.
 *
 * The worker prints this only at high verbosity. Returning undefined simply means
 * the line was absent, which is the normal case when no resizing was needed.
 */
export function parseFittedContext(text: string): FittedContext | undefined {
  const match = /context size reduced from (\d+) to (\d+)/.exec(text);
  const trained = Number(match?.[1]);
  const fitted = Number(match?.[2]);
  if (!isPositiveInteger(trained) || !isPositiveInteger(fitted)) {
    return undefined;
  }
  return { trainedContextSize: trained, fittedContextSize: fitted };
}

/**
 * Reads the device memory budget llama.cpp believes it has.
 *
 * This is the Metal working-set budget minus llama-server's own allocation, so it
 * ignores every other process on the machine. Captured for diagnostics only.
 */
export function parseFreeDeviceMemoryMiB(text: string): number | undefined {
  const match = /vs\. (\d+) MiB of free device memory/.exec(text);
  const value = Number(match?.[1]);
  return isPositiveInteger(value) ? value : undefined;
}

function isPositiveInteger(value: number): boolean {
  return Number.isFinite(value) && value > 0;
}
