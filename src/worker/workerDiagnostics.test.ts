import assert from 'node:assert/strict';
import test from 'node:test';
import { createWorkerDiagnostics } from './workerDiagnostics.ts';

test('worker stalls are logged and surfaced as visible warnings', () => {
  const info: string[] = [];
  const warnings: Array<{ message: string; visible: boolean }> = [];
  const diagnostics = createWorkerDiagnostics({
    info: (message) => info.push(message),
    warn: (message, visible) => warnings.push({ message, visible: visible === true }),
  });

  diagnostics.info('chat-1 start');
  diagnostics.warn('chat-1 stalled');

  assert.deepEqual(info, ['chat-1 start']);
  assert.deepEqual(warnings, [{ message: 'chat-1 stalled', visible: true }]);
  assert.equal(diagnostics.stallWarningMilliseconds, 30_000);
  assert.equal(diagnostics.longGenerationWarningMilliseconds, 60_000);
});
