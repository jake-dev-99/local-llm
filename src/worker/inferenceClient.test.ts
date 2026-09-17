import assert from 'node:assert/strict';
import test from 'node:test';
import { supportsInfill, type InferenceClient } from './inferenceClient.ts';

/**
 * Assignability of the two real clients is enforced by `implements` on each
 * class, so it fails the build rather than a test. What is worth asserting here
 * is the capability narrowing, which decides at runtime whether inline
 * completion may call a client at all.
 */
function clientWith(overrides: Partial<InferenceClient>): InferenceClient {
  return {
    supports: { infill: false, constrainedDecoding: false },
    chat: async () => ({ inputTokens: 0, textCharacters: 0, toolCallCount: 0 }),
    tokenize: async () => 0,
    getModelProfile: async () => ({
      loadedContextSize: 4096,
      hasChatTemplate: true,
      supportsTools: false,
      supportsToolCalls: false,
      supportsSystemRole: true,
    }),
    ...overrides,
  };
}

test('a client that declares and implements infill is narrowed to one', () => {
  const client = clientWith({
    supports: { infill: true, constrainedDecoding: true },
    infill: async () => 'completion',
  });

  assert.equal(supportsInfill(client), true);
  if (supportsInfill(client)) {
    // The narrowing is what lets the caller invoke it without a cast.
    assert.equal(typeof client.infill, 'function');
  }
});

test('a client without infill is not narrowed', () => {
  assert.equal(supportsInfill(clientWith({})), false);
});

test('a client claiming infill without implementing it is rejected', () => {
  // The flag alone is not enough: trusting it would call a missing method.
  const lying = clientWith({ supports: { infill: true, constrainedDecoding: false } });
  assert.equal(supportsInfill(lying), false);
});

test('a client that implements infill without declaring it stays unused', () => {
  // Capability is declared, not inferred from the shape of the object.
  const undeclared = clientWith({ infill: async () => 'completion' });
  assert.equal(supportsInfill(undeclared), false);
});
