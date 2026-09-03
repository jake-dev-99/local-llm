import assert from 'node:assert/strict';
import test from 'node:test';
import { modelLoadProgress } from './modelLoadProgress.ts';

test('a cold response-only request reports both model loading milestones', () => {
  assert.deepEqual(
    modelLoadProgress({
      modelResident: false,
      toolCallsPossible: false,
      agentRequest: false,
    }),
    {
      loading: 'Loading Model into Memory',
      loaded: 'Model loaded successfully - Response Processing',
    },
  );
});

test('a resident model does not add loading text to the response', () => {
  assert.equal(
    modelLoadProgress({
      modelResident: true,
      toolCallsPossible: false,
      agentRequest: false,
    }),
    undefined,
  );
});

test('a tool-capable request does not add loading text to the response', () => {
  assert.equal(
    modelLoadProgress({
      modelResident: false,
      toolCallsPossible: true,
      agentRequest: false,
    }),
    undefined,
  );
});

test('an agent request does not add loading text even without tools', () => {
  assert.equal(
    modelLoadProgress({
      modelResident: false,
      toolCallsPossible: false,
      agentRequest: true,
    }),
    undefined,
  );
});
