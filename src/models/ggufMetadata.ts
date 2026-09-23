import { open } from 'node:fs/promises';
import { describeError } from '../errorDetail.ts';

/**
 * Metadata read straight from a GGUF file header, without loading the model.
 *
 * Used only as a bootstrap hint before a model has ever been loaded. The
 * authoritative context window is whatever the worker reports through /props
 * after llama.cpp has fitted the model to this machine.
 */
export interface GgufModelMetadata {
  architecture: string;
  /** Context length the model was trained with, from `<architecture>.context_length`. */
  trainedContextLength?: number;
}

const GGUF_MAGIC = 'GGUF';

// Header scan limit. A 151k-token vocabulary pushes the tokenizer arrays past
// 4 MiB, so the window has to clear them to reach later keys.
const MAX_HEADER_BYTES = 128 * 1024 * 1024;

const GgufType = {
  Uint8: 0,
  Int8: 1,
  Uint16: 2,
  Int16: 3,
  Uint32: 4,
  Int32: 5,
  Float32: 6,
  Bool: 7,
  String: 8,
  Array: 9,
  Uint64: 10,
  Int64: 11,
  Float64: 12,
} as const;

/**
 * Reads architecture and trained context length from a GGUF header.
 *
 * Returns undefined for anything unreadable. Callers treat this as a hint, never
 * as a requirement, so a malformed or unusual file must not block model import.
 */
export async function readGgufMetadata(
  filePath: string,
  onWarning?: (message: string) => void,
): Promise<GgufModelMetadata | undefined> {
  let handle;
  try {
    handle = await open(filePath, 'r');
    const { size } = await handle.stat();
    const length = Math.min(size, MAX_HEADER_BYTES);
    const buffer = Buffer.alloc(length);
    await handle.read(buffer, 0, length, 0);
    const metadata = parseGgufHeader(buffer, onWarning);
    if (!metadata) {
      onWarning?.(`No GGUF metadata could be read from ${filePath}.`);
    }
    return metadata;
  } catch (error) {
    onWarning?.(`Could not read GGUF metadata from ${filePath}: ${describeError(error)}`);
    return undefined;
  } finally {
    await handle?.close().catch((error: unknown) => {
      onWarning?.(`Could not close ${filePath}: ${describeError(error)}`);
    });
  }
}

export function parseGgufHeader(
  buffer: Buffer,
  onWarning?: (message: string) => void,
): GgufModelMetadata | undefined {
  if (buffer.length < 24 || buffer.toString('ascii', 0, 4) !== GGUF_MAGIC) {
    return undefined;
  }
  const reader = new HeaderReader(buffer);
  try {
    reader.skip(4); // magic
    reader.readUint32(); // format version
    reader.readUint64(); // tensor count
    const pairCount = reader.readUint64();

    let architecture: string | undefined;
    let trainedContextLength: number | undefined;
    for (let index = 0; index < pairCount; index += 1) {
      const key = reader.readString();
      const value = reader.readValue(reader.readUint32());
      if (key === 'general.architecture' && typeof value === 'string') {
        architecture = value;
      } else if (key.endsWith('.context_length') && typeof value === 'number') {
        trainedContextLength = value;
      }
      if (architecture && trainedContextLength) {
        break;
      }
    }
    if (!architecture) {
      onWarning?.('The GGUF header declared no general.architecture key.');
      return undefined;
    }
    return {
      architecture,
      ...(isPositiveInteger(trainedContextLength) ? { trainedContextLength } : {}),
    };
  } catch (error) {
    onWarning?.(`The GGUF header could not be parsed: ${describeError(error)}`);
    return undefined;
  }
}

class HeaderReader {
  private offset = 0;
  private readonly buffer: Buffer;

  constructor(buffer: Buffer) {
    this.buffer = buffer;
  }

  skip(bytes: number): void {
    this.require(bytes);
    this.offset += bytes;
  }

  readUint32(): number {
    this.require(4);
    const value = this.buffer.readUInt32LE(this.offset);
    this.offset += 4;
    return value;
  }

  readUint64(): number {
    this.require(8);
    const value = this.buffer.readBigUInt64LE(this.offset);
    this.offset += 8;
    return Number(value);
  }

  readString(): string {
    const length = this.readUint64();
    this.require(length);
    const value = this.buffer.toString('utf8', this.offset, this.offset + length);
    this.offset += length;
    return value;
  }

  readValue(type: number): string | number | boolean | undefined {
    switch (type) {
      case GgufType.Uint8:
      case GgufType.Int8:
      case GgufType.Bool: {
        this.require(1);
        const value = type === GgufType.Int8
          ? this.buffer.readInt8(this.offset)
          : this.buffer.readUInt8(this.offset);
        this.offset += 1;
        return type === GgufType.Bool ? value !== 0 : value;
      }
      case GgufType.Uint16:
      case GgufType.Int16: {
        this.require(2);
        const value = type === GgufType.Int16
          ? this.buffer.readInt16LE(this.offset)
          : this.buffer.readUInt16LE(this.offset);
        this.offset += 2;
        return value;
      }
      case GgufType.Uint32:
      case GgufType.Int32:
      case GgufType.Float32: {
        this.require(4);
        const value = type === GgufType.Float32
          ? this.buffer.readFloatLE(this.offset)
          : type === GgufType.Int32
            ? this.buffer.readInt32LE(this.offset)
            : this.buffer.readUInt32LE(this.offset);
        this.offset += 4;
        return value;
      }
      case GgufType.Uint64:
      case GgufType.Int64:
      case GgufType.Float64: {
        this.require(8);
        const value = type === GgufType.Float64
          ? this.buffer.readDoubleLE(this.offset)
          : type === GgufType.Int64
            ? Number(this.buffer.readBigInt64LE(this.offset))
            : Number(this.buffer.readBigUInt64LE(this.offset));
        this.offset += 8;
        return value;
      }
      case GgufType.String:
        return this.readString();
      case GgufType.Array: {
        const elementType = this.readUint32();
        const count = this.readUint64();
        for (let index = 0; index < count; index += 1) {
          this.readValue(elementType);
        }
        return undefined;
      }
      default:
        throw new Error(`Unsupported GGUF value type ${type}.`);
    }
  }

  private require(bytes: number): void {
    if (bytes < 0 || this.offset + bytes > this.buffer.length) {
      throw new Error('GGUF header ended before the requested metadata.');
    }
  }
}

function isPositiveInteger(value: number | undefined): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0;
}
