import assert from 'node:assert/strict';
import test from 'node:test';
import {
  directoryHasChanged,
  directoryIdentity,
  isExtensionOwned,
  manifestDigest,
  runtimeForFormat,
  type ModelFileInput,
} from './modelIdentity.ts';

function file(overrides: Partial<ModelFileInput> = {}): ModelFileInput {
  return {
    path: 'model-00001-of-00002.safetensors',
    size: 4096,
    modifiedAt: 1_700_000_000_000,
    ...overrides,
  };
}

test('the digest ignores the order files are listed in', () => {
  const a = file({ path: 'a.safetensors' });
  const b = file({ path: 'b.safetensors', size: 2048 });
  assert.equal(manifestDigest([a, b]), manifestDigest([b, a]));
});

test('the digest ignores modification times', () => {
  // Copying a checkpoint moves mtimes without changing the model; a digest that
  // moved with them would discard verified capabilities for nothing.
  assert.equal(
    manifestDigest([file()]),
    manifestDigest([file({ modifiedAt: 1_800_000_000_000 })]),
  );
});

test('the digest changes when a shard changes size', () => {
  assert.notEqual(manifestDigest([file()]), manifestDigest([file({ size: 8192 })]));
});

test('the digest changes when a tensor header changes but the size does not', () => {
  // The case a size comparison cannot catch: same bytes on disk, different
  // tensors described inside them.
  const before = file({ header: Buffer.from('{"w":{"dtype":"BF16"}}') });
  const after = file({ header: Buffer.from('{"w":{"dtype":"F16"}}') });
  assert.notEqual(manifestDigest([before]), manifestDigest([after]));
});

test('a renamed shard changes the digest', () => {
  const before = [file({ path: 'a.safetensors' }), file({ path: 'b.safetensors' })];
  const after = [file({ path: 'a.safetensors' }), file({ path: 'c.safetensors' })];
  assert.notEqual(manifestDigest(before), manifestDigest(after));
});

test('a path cannot be crafted to collide with another manifest', () => {
  // Encoding is unambiguous, so a separator inside a filename stays inert.
  const crafted = [file({ path: 'a.safetensors", 4096], ["b.safetensors' })];
  const plain = [file({ path: 'a.safetensors' }), file({ path: 'b.safetensors' })];
  assert.notEqual(manifestDigest(crafted), manifestDigest(plain));
});

test('identity totals the bytes and sorts the fingerprints', () => {
  const identity = directoryIdentity([
    file({ path: 'b.safetensors', size: 2048 }),
    file({ path: 'a.safetensors', size: 4096 }),
  ]);
  assert.equal(identity.totalBytes, 6144);
  assert.deepEqual(identity.files.map((entry) => entry.path), [
    'a.safetensors',
    'b.safetensors',
  ]);
});

test('headers never leak into the persisted fingerprints', () => {
  const identity = directoryIdentity([file({ header: Buffer.alloc(64) })]);
  assert.deepEqual(Object.keys(identity.files[0] ?? {}).sort(), [
    'modifiedAt',
    'path',
    'size',
  ]);
});

test('an unchanged directory reports no change', () => {
  const recorded = directoryIdentity([file()]).files;
  assert.equal(directoryHasChanged(recorded, recorded), false);
});

test('a resized, re-dated, renamed, removed or added file all count as changes', () => {
  const recorded = directoryIdentity([file(), file({ path: 'b.safetensors' })]).files;
  const current = (files: ModelFileInput[]) => directoryIdentity(files).files;
  const other = file({ path: 'b.safetensors' });

  assert.equal(
    directoryHasChanged(recorded, current([file({ size: 9999 }), other])),
    true, 'resized',
  );
  assert.equal(
    directoryHasChanged(recorded, current([file({ modifiedAt: 1 }), other])),
    true, 're-dated',
  );
  assert.equal(
    directoryHasChanged(recorded, current([file({ path: 'renamed.safetensors' }), other])),
    true, 'renamed',
  );
  assert.equal(directoryHasChanged(recorded, current([file()])), true, 'removed');
  assert.equal(
    directoryHasChanged(recorded, current([file(), other, file({ path: 'c.safetensors' })])),
    true, 'added',
  );
});

test('a model recorded before fingerprints existed is left alone', () => {
  // The registry backfills such models rather than discarding their verified
  // capabilities, so absent fingerprints must not read as a change.
  const current = directoryIdentity([file()]).files;
  assert.equal(directoryHasChanged(undefined, current), false);
  assert.equal(directoryHasChanged([], current), false);
});

test('format maps to the runtime that can load it', () => {
  assert.equal(runtimeForFormat('safetensors'), 'transformers');
  assert.equal(runtimeForFormat('gguf'), 'llama-cpp');
});

test('a model with no ownership flag is treated as extension-owned', () => {
  // Every record written before checkpoints could be registered in place is a
  // GGUF copied into extension storage, so absence must not read as borrowed.
  assert.equal(isExtensionOwned({}), true);
  assert.equal(isExtensionOwned({ managed: true }), true);
  assert.equal(isExtensionOwned({ managed: false }), false);
});
