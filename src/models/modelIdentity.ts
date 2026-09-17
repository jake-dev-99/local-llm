/**
 * Identity and change detection for models that are directories.
 *
 * A GGUF model is one file, so its identity is that file's SHA-256 and its
 * change detection is a size and mtime comparison. A Safetensors model is a
 * directory of shards, which breaks both halves of that.
 *
 * Re-hashing the contents is not an option: these checkpoints reach tens of
 * gigabytes and the registry verifies every model on activation. So a directory
 * model is identified by a *manifest digest* — see `manifestDigest` for exactly
 * what that covers and what it does not.
 */

import { createHash } from 'node:crypto';

export type ModelFormat = 'gguf' | 'safetensors';

/** Which worker can load a model. Recorded per model, not inferred at use. */
export type ModelRuntime = 'llama-cpp' | 'transformers';

export interface ModelFileFingerprint {
  /** POSIX-separated path relative to the model directory. */
  path: string;
  size: number;
  modifiedAt: number;
}

export interface DirectoryIdentity {
  digest: string;
  totalBytes: number;
  files: ModelFileFingerprint[];
}

/**
 * Input for one file in the directory.
 *
 * `header` is the raw Safetensors header bytes where the file has one. It is
 * what lifts the digest above a mere size comparison: the header names every
 * tensor with its dtype, shape and byte offsets, so any change to the model's
 * structure changes the digest even when the total size happens to match.
 */
export interface ModelFileInput extends ModelFileFingerprint {
  header?: Buffer;
}

/**
 * A stable digest over a directory model's manifest.
 *
 * Covers every file's relative path and size, plus the Safetensors header of
 * each shard. It does **not** cover tensor payload bytes, so a value flipped
 * inside a tensor without changing its shape, dtype or file size will not be
 * detected. That is a deliberate trade: the alternative is hashing tens of
 * gigabytes every time the extension activates.
 *
 * Modification times are excluded on purpose. Copying a checkpoint changes them
 * without changing the model, and a digest that moved on every copy would
 * discard verified capabilities for no reason. Change *detection* still uses
 * mtimes — see `directoryHasChanged`.
 */
export function manifestDigest(files: readonly ModelFileInput[]): string {
  const hash = createHash('sha256');
  for (const file of [...files].sort((a, b) => a.path.localeCompare(b.path))) {
    // JSON encoding keeps the path and size unambiguous, so a path containing
    // a separator character cannot be made to collide with another manifest.
    hash.update(JSON.stringify([file.path, file.size, file.header?.length ?? 0]));
    if (file.header) {
      hash.update(file.header);
    }
  }
  return hash.digest('hex');
}

export function directoryIdentity(files: readonly ModelFileInput[]): DirectoryIdentity {
  const fingerprints = files
    .map(({ path, size, modifiedAt }) => ({ path, size, modifiedAt }))
    .sort((a, b) => a.path.localeCompare(b.path));
  return {
    digest: manifestDigest(files),
    totalBytes: fingerprints.reduce((total, file) => total + file.size, 0),
    files: fingerprints,
  };
}

/**
 * Whether a directory model differs from what was recorded.
 *
 * Deliberately cheap: a stat per file and no reads. This runs for every
 * installed model on activation, so it has to stay proportional to the file
 * count rather than to the bytes on disk.
 *
 * A model with no recorded fingerprints predates this field and is treated as
 * unchanged, matching how the registry backfills rather than invalidates.
 */
export function directoryHasChanged(
  recorded: readonly ModelFileFingerprint[] | undefined,
  current: readonly ModelFileFingerprint[],
): boolean {
  if (!recorded || recorded.length === 0) {
    return false;
  }
  if (recorded.length !== current.length) {
    return true;
  }
  const byPath = new Map(current.map((file) => [file.path, file]));
  return recorded.some((file) => {
    const match = byPath.get(file.path);
    return !match || match.size !== file.size || match.modifiedAt !== file.modifiedAt;
  });
}

/** The runtime that can load a given format. */
export function runtimeForFormat(format: ModelFormat): ModelRuntime {
  return format === 'safetensors' ? 'transformers' : 'llama-cpp';
}

/**
 * Whether removing a model should delete its bytes.
 *
 * GGUF models are copied into extension storage, so the extension owns them and
 * deletes them on removal. Safetensors checkpoints are registered where they
 * already live — copying tens of gigabytes to duplicate what the user already
 * has on disk would be indefensible — so removing one must unregister it and
 * leave the directory untouched.
 *
 * A record with no flag predates this distinction. Every such model is a copied
 * GGUF, so absence means owned.
 */
export function isExtensionOwned(model: { managed?: boolean }): boolean {
  return model.managed !== false;
}
