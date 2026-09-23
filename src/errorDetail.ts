/**
 * Full, human-readable description of any thrown value.
 *
 * Failures in this extension must be loud. `TypeError: fetch failed` on its own
 * says nothing; the cause underneath it names the actual fault, such as a
 * headers timeout or a refused connection. This walks the whole chain and keeps
 * the diagnostic fields Node attaches to system errors.
 */
export function describeError(error: unknown): string {
  const parts: string[] = [];
  let current: unknown = error;
  for (let depth = 0; depth < MAX_CAUSE_DEPTH && current !== undefined && current !== null; depth += 1) {
    if (current instanceof Error) {
      parts.push(`${current.name}: ${current.message}${diagnosticFields(current)}`);
      current = (current as { cause?: unknown }).cause;
      continue;
    }
    parts.push(describeValue(current));
    break;
  }
  return parts.join(' <- caused by ') || 'no error detail available';
}

/**
 * Stack of the outermost error, for the log line that follows the description.
 */
export function errorStack(error: unknown): string | undefined {
  return error instanceof Error && typeof error.stack === 'string' && error.stack
    ? error.stack
    : undefined;
}

const MAX_CAUSE_DEPTH = 8;

// Fields Node sets on transport and file system failures. Each one turns a vague
// message into an actionable one.
const DIAGNOSTIC_KEYS = [
  'code',
  'errno',
  'syscall',
  'address',
  'port',
  'path',
  'status',
  'statusCode',
] as const;

function diagnosticFields(error: Error): string {
  const record = error as unknown as Record<string, unknown>;
  const fields: string[] = [];
  for (const key of DIAGNOSTIC_KEYS) {
    const value = record[key];
    if (value !== undefined && value !== null && typeof value !== 'object') {
      fields.push(`${key}=${String(value)}`);
    }
  }
  return fields.length ? ` (${fields.join(', ')})` : '';
}

function describeValue(value: unknown): string {
  if (typeof value === 'string') {
    return value;
  }
  try {
    return JSON.stringify(value) ?? String(value);
  } catch {
    return String(value);
  }
}

/** Whether a filesystem error means the path does not exist, the one case a caller may treat as an answer rather than a failure. */
export function isMissingFileError(error: unknown): boolean {
  const code = typeof error === 'object' && error !== null
    ? (error as NodeJS.ErrnoException).code
    : undefined;
  return code === 'ENOENT' || code === 'ENOTDIR';
}
