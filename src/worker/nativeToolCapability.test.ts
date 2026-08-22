import assert from 'node:assert/strict';
import test from 'node:test';
import {
  matchingNativeToolSupport,
  nativeToolCapabilityFingerprint,
  TOOL_PROTOCOL_VERSION,
} from './nativeToolCapability.ts';

const fingerprintInput = {
  modelSha256: 'model-sha256',
  workerBuild: 'b10472-60eeeb608',
  chatTemplateFingerprint: 'template-sha256',
  platform: 'darwin-arm64',
  toolProtocolVersion: TOOL_PROTOCOL_VERSION,
};

test('native capability fingerprint includes every runtime compatibility field', () => {
  const fingerprint = nativeToolCapabilityFingerprint(fingerprintInput);
  assert.match(fingerprint, /^[a-f0-9]{64}$/);

  for (const [field, value] of [
    ['modelSha256', 'other-model'],
    ['workerBuild', 'other-worker'],
    ['chatTemplateFingerprint', 'other-template'],
    ['platform', 'win32-x64'],
    ['toolProtocolVersion', 'other-protocol'],
  ] as const) {
    assert.notEqual(
      nativeToolCapabilityFingerprint({ ...fingerprintInput, [field]: value }),
      fingerprint,
      field,
    );
  }
});

test('persisted support is reused only for an exact fingerprint match', () => {
  const fingerprint = nativeToolCapabilityFingerprint(fingerprintInput);
  const record = {
    fingerprint,
    support: 'unavailable' as const,
    observedAt: '2026-08-21T00:00:00.000Z',
  };

  assert.equal(matchingNativeToolSupport(record, fingerprint), 'unavailable');
  assert.equal(matchingNativeToolSupport(record, 'different'), 'unknown');
  assert.equal(matchingNativeToolSupport(undefined, fingerprint), 'unknown');
  assert.equal(
    matchingNativeToolSupport({ ...record, support: 'corrupt' } as never, fingerprint),
    'unknown',
  );
});
