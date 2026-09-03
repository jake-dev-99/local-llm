import assert from 'node:assert/strict';
import test from 'node:test';
import { createWorkerDiagnostics } from './workerDiagnostics.ts';

test('worker diagnostics forward request milestones', () => {
  const info: string[] = [];
  const diagnostics = createWorkerDiagnostics({
    info: (message) => info.push(message),
  });

  diagnostics.info('chat-1 start');

  assert.deepEqual(info, ['chat-1 start']);
});
