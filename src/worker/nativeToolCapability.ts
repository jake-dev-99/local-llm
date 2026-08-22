import { createHash } from 'node:crypto';
import type {
  NativeToolCallSupport,
  NativeToolCapabilityRecord,
} from '../domain';

export const TOOL_PROTOCOL_VERSION = 'local-llm-tool-protocol-v1';

export interface NativeToolCapabilityFingerprintInput {
  modelSha256: string;
  workerBuild: string;
  chatTemplateFingerprint: string;
  platform: string;
  toolProtocolVersion: string;
}

export function nativeToolCapabilityFingerprint(
  input: NativeToolCapabilityFingerprintInput,
): string {
  return createHash('sha256').update(JSON.stringify([
    input.modelSha256,
    input.workerBuild,
    input.chatTemplateFingerprint,
    input.platform,
    input.toolProtocolVersion,
  ])).digest('hex');
}

export function matchingNativeToolSupport(
  record: NativeToolCapabilityRecord | undefined,
  fingerprint: string,
): NativeToolCallSupport | 'unknown' {
  return record?.fingerprint === fingerprint &&
    (record.support === 'available' || record.support === 'unavailable')
    ? record.support
    : 'unknown';
}
