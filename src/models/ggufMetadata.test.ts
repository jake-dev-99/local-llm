import assert from 'node:assert/strict';
import test from 'node:test';
import { parseGgufHeader } from './ggufMetadata.ts';

const UINT32 = 4;
const UINT64 = 10;
const STRING = 8;
const ARRAY = 9;

function buildHeader(
  pairs: Array<[string, number, Buffer]>,
  overrides: { magic?: string; pairCount?: number } = {},
): Buffer {
  const parts: Buffer[] = [];
  parts.push(Buffer.from(overrides.magic ?? 'GGUF', 'ascii'));
  const preamble = Buffer.alloc(20);
  preamble.writeUInt32LE(3, 0); // format version
  preamble.writeBigUInt64LE(0n, 4); // tensor count
  preamble.writeBigUInt64LE(BigInt(overrides.pairCount ?? pairs.length), 12);
  parts.push(preamble);
  for (const [key, type, value] of pairs) {
    const keyBytes = Buffer.from(key, 'utf8');
    const header = Buffer.alloc(8 + keyBytes.length + 4);
    header.writeBigUInt64LE(BigInt(keyBytes.length), 0);
    keyBytes.copy(header, 8);
    header.writeUInt32LE(type, 8 + keyBytes.length);
    parts.push(header, value);
  }
  return Buffer.concat(parts);
}

function stringValue(value: string): Buffer {
  const bytes = Buffer.from(value, 'utf8');
  const out = Buffer.alloc(8 + bytes.length);
  out.writeBigUInt64LE(BigInt(bytes.length), 0);
  bytes.copy(out, 8);
  return out;
}

function uint32Value(value: number): Buffer {
  const out = Buffer.alloc(4);
  out.writeUInt32LE(value, 0);
  return out;
}

function uint64Value(value: number): Buffer {
  const out = Buffer.alloc(8);
  out.writeBigUInt64LE(BigInt(value), 0);
  return out;
}

function uint32ArrayValue(values: number[]): Buffer {
  const out = Buffer.alloc(12 + values.length * 4);
  out.writeUInt32LE(UINT32, 0);
  out.writeBigUInt64LE(BigInt(values.length), 4);
  values.forEach((value, index) => out.writeUInt32LE(value, 12 + index * 4));
  return out;
}

test('reads architecture and trained context length from a GGUF header', () => {
  const header = buildHeader([
    ['general.architecture', STRING, stringValue('qwen2')],
    ['qwen2.block_count', UINT32, uint32Value(64)],
    ['qwen2.context_length', UINT32, uint32Value(131072)],
  ]);

  assert.deepEqual(parseGgufHeader(header), {
    architecture: 'qwen2',
    trainedContextLength: 131072,
  });
});

test('skips over array values to reach later keys', () => {
  // Real models put a large tokenizer array before some metadata; the parser has
  // to walk past it rather than stopping.
  const header = buildHeader([
    ['general.architecture', STRING, stringValue('llama')],
    ['tokenizer.ggml.token_type', ARRAY, uint32ArrayValue([1, 2, 3, 4, 5])],
    ['llama.context_length', UINT64, uint64Value(8192)],
  ]);

  assert.deepEqual(parseGgufHeader(header), {
    architecture: 'llama',
    trainedContextLength: 8192,
  });
});

test('returns architecture alone when no context length is present', () => {
  const header = buildHeader([['general.architecture', STRING, stringValue('phi3')]]);

  assert.deepEqual(parseGgufHeader(header), { architecture: 'phi3' });
});

test('rejects a file that is not GGUF', () => {
  const header = buildHeader(
    [['general.architecture', STRING, stringValue('qwen2')]],
    { magic: 'ZZZZ' },
  );

  assert.equal(parseGgufHeader(header), undefined);
});

test('a truncated header yields no metadata instead of throwing', () => {
  const header = buildHeader(
    [['general.architecture', STRING, stringValue('qwen2')]],
    { pairCount: 50 },
  );

  assert.equal(parseGgufHeader(header.subarray(0, header.length - 4)), undefined);
});
