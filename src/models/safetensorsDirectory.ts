/**
 * Reading a Safetensors checkpoint directory from disk.
 *
 * The filesystem half of `modelIdentity.ts`, kept separate so the identity and
 * change-detection rules stay pure and directly testable.
 *
 * Only headers are ever read. A Safetensors file begins with an 8-byte
 * little-endian header length followed by a JSON header naming every tensor
 * with its dtype, shape and byte offsets; the payload after it is never
 * touched here.
 */

import { open, readdir, stat } from 'node:fs/promises';
import * as path from 'node:path';
import { describeError } from '../errorDetail.ts';
import { directoryIdentity, type DirectoryIdentity, type ModelFileInput } from './modelIdentity.ts';

/**
 * Upper bound on a Safetensors header.
 *
 * A large vocabulary pushes headers well past a megabyte, but a length beyond
 * this means the file is not what it claims to be and is not worth allocating
 * for.
 */
export const MAX_SAFETENSORS_HEADER_BYTES = 128 * 1024 * 1024;

export interface SafetensorsCheckpoint {
  identity: DirectoryIdentity;
  /** Architecture from `config.json`, when it declares one. */
  architecture?: string;
  /** Trained context window, used as a bootstrap estimate only. */
  trainedContextLength?: number;
  /** Quantization scheme declared by the checkpoint, when there is one. */
  quantization?: string;
  /** True when loading would execute Python shipped inside the checkpoint. */
  customCodeRequired: boolean;
}

/**
 * Whether a directory looks like a Safetensors checkpoint.
 *
 * Deliberately shallow. Transformers stays authoritative for whether a
 * checkpoint actually loads; this only decides which install path to take.
 */
export async function isSafetensorsDirectory(directory: string): Promise<boolean> {
  try {
    const entries = await readdir(directory);
    return entries.includes('config.json') &&
      entries.some((entry) => entry.toLowerCase().endsWith('.safetensors'));
  } catch {
    return false;
  }
}

/**
 * Reads one Safetensors header, or undefined when it cannot be read.
 *
 * An unreadable header must never block an install: the digest still covers the
 * file's path and size, so the model remains identifiable, just less precisely.
 */
export async function readSafetensorsHeader(filePath: string): Promise<Buffer | undefined> {
  let handle;
  try {
    handle = await open(filePath, 'r');
    const prefix = Buffer.alloc(8);
    const { bytesRead } = await handle.read(prefix, 0, 8, 0);
    if (bytesRead < 8) {
      return undefined;
    }
    const length = Number(prefix.readBigUInt64LE(0));
    if (!Number.isSafeInteger(length) || length <= 0 ||
        length > MAX_SAFETENSORS_HEADER_BYTES) {
      return undefined;
    }
    const header = Buffer.alloc(length);
    const read = await handle.read(header, 0, length, 8);
    return read.bytesRead === length ? header : undefined;
  } catch {
    return undefined;
  } finally {
    await handle?.close().catch(() => undefined);
  }
}

/**
 * Every regular file in a checkpoint directory, with Safetensors headers read.
 *
 * Checkpoints are flat, so this does not recurse. A nested directory would not
 * be part of the checkpoint and including it would only make the digest drift.
 */
export async function collectCheckpointFiles(directory: string): Promise<ModelFileInput[]> {
  const entries = await readdir(directory, { withFileTypes: true });
  const files: ModelFileInput[] = [];
  for (const entry of entries) {
    if (!entry.isFile()) {
      continue;
    }
    const absolute = path.join(directory, entry.name);
    const metadata = await stat(absolute);
    const header = entry.name.toLowerCase().endsWith('.safetensors')
      ? await readSafetensorsHeader(absolute)
      : undefined;
    files.push({
      path: entry.name,
      size: metadata.size,
      modifiedAt: metadata.mtimeMs,
      ...(header ? { header } : {}),
    });
  }
  return files;
}

/** Re-reads only what change detection needs: paths, sizes and mtimes. */
export async function fingerprintCheckpoint(directory: string) {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    if (!entry.isFile()) {
      continue;
    }
    const metadata = await stat(path.join(directory, entry.name));
    files.push({ path: entry.name, size: metadata.size, modifiedAt: metadata.mtimeMs });
  }
  return files.sort((a, b) => a.path.localeCompare(b.path));
}

/**
 * Describes a checkpoint without loading any weights.
 *
 * Mirrors what `resources/runtime/runtime/inspector.py` reports, so the
 * extension can show a model's shape before the Python runtime is provisioned
 * or on a machine where it is broken.
 */
export interface CheckpointValidation {
  ok: boolean;
  /** Fatal problems; the checkpoint cannot load as it stands. */
  errors: string[];
  /** Non-fatal concerns worth showing before first use. */
  warnings: string[];
}

/**
 * Static validation: everything checkable without the Python runtime.
 *
 * Runs automatically after import. Loading itself stays the explicit
 * Validate command — it provisions gigabytes on first use, so it must
 * never run uninvited.
 */
export async function validateCheckpointStatic(directory: string): Promise<CheckpointValidation> {
  const errors: string[] = [];
  const warnings: string[] = [];
  let entries: string[];
  try {
    entries = await readdir(directory);
  } catch (error) {
    return { ok: false, errors: [`Cannot read ${directory}: ${describeError(error)}.`], warnings };
  }
  const names = new Set(entries.map((entry) => entry.toLowerCase()));
  const config = await readJsonFile(path.join(directory, 'config.json'));
  if (!names.has('config.json') || config === undefined) {
    errors.push('config.json is missing or unreadable — Transformers cannot load these weights without it.');
  }
  const weights = entries.filter((entry) => entry.toLowerCase().endsWith('.safetensors'));
  if (weights.length === 0) {
    errors.push('No .safetensors weights files found.');
  } else {
    let readable = 0;
    for (const file of weights) {
      if ((await readSafetensorsHeader(path.join(directory, file))) !== undefined) {
        readable += 1;
      }
    }
    if (readable === 0) {
      errors.push('No weights file has a readable header — the files may be truncated or corrupt.');
    } else if (readable < weights.length) {
      warnings.push(`${weights.length - readable} of ${weights.length} weights files have unreadable headers and contribute size only.`);
    }
  }
  if (!names.has('tokenizer.json') && !names.has('tokenizer_config.json')) {
    warnings.push('No tokenizer files — chat will fail at load with missing_tokenizer until they are added.');
  }
  if (config !== undefined && architectureOf(config) === undefined) {
    warnings.push('config.json names no recognized architecture — Transformers decides at load whether it runs.');
  }
  return { ok: errors.length === 0, errors, warnings };
}

export async function readSafetensorsCheckpoint(
  directory: string,
  onWarning?: (message: string) => void,
): Promise<SafetensorsCheckpoint> {
  const files = await collectCheckpointFiles(directory);
  const config = await readJsonFile(path.join(directory, 'config.json'), onWarning);
  const architecture = architectureOf(config);
  const trainedContextLength = contextLengthOf(config);
  const quantization = quantizationOf(config);
  return {
    identity: directoryIdentity(files),
    ...(architecture ? { architecture } : {}),
    ...(trainedContextLength ? { trainedContextLength } : {}),
    ...(quantization ? { quantization } : {}),
    customCodeRequired: config !== undefined && 'auto_map' in config,
  };
}

async function readJsonFile(
  filePath: string,
  onWarning?: (message: string) => void,
): Promise<Record<string, unknown> | undefined> {
  let handle;
  try {
    handle = await open(filePath, 'r');
    const parsed: unknown = JSON.parse(await handle.readFile('utf8'));
    return typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)
      ? parsed as Record<string, unknown>
      : undefined;
  } catch (error) {
    onWarning?.(`Could not read ${path.basename(filePath)}: ${describeError(error)}`);
    return undefined;
  } finally {
    await handle?.close().catch(() => undefined);
  }
}

function architectureOf(config: Record<string, unknown> | undefined): string | undefined {
  const architectures = config?.['architectures'];
  const first = Array.isArray(architectures) ? architectures[0] : undefined;
  return typeof first === 'string' ? first : undefined;
}

function contextLengthOf(config: Record<string, unknown> | undefined): number | undefined {
  for (const key of ['max_position_embeddings', 'n_positions', 'max_sequence_length']) {
    const value = config?.[key];
    if (typeof value === 'number' && Number.isInteger(value) && value > 0) {
      return value;
    }
  }
  const nested = config?.['text_config'];
  return typeof nested === 'object' && nested !== null
    ? contextLengthOf(nested as Record<string, unknown>)
    : undefined;
}

/**
 * The declared quantization scheme.
 *
 * Reported so a user can be warned before a load: pre-quantized checkpoints
 * mostly depend on CUDA-only kernels, so on Metal and Intel XPU they either
 * fail or dequantize back to bf16 and use more memory than they saved.
 */
function quantizationOf(config: Record<string, unknown> | undefined): string | undefined {
  const quantization = config?.['quantization_config'];
  if (typeof quantization !== 'object' || quantization === null) {
    return undefined;
  }
  const record = quantization as Record<string, unknown>;
  for (const key of ['quant_method', 'format']) {
    const value = record[key];
    if (typeof value === 'string') {
      return value;
    }
  }
  const groups = record['config_groups'];
  if (typeof groups === 'object' && groups !== null) {
    for (const group of Object.values(groups as Record<string, unknown>)) {
      const format = (group as Record<string, unknown> | null)?.['format'];
      if (typeof format === 'string') {
        return format;
      }
    }
  }
  return 'unknown';
}
