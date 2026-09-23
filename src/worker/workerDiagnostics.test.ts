import assert from 'node:assert/strict';
import test from 'node:test';
import { createLineBuffer, createWorkerDiagnostics, workerLineLevel } from './workerDiagnostics.ts';

test('worker diagnostics forward request milestones', () => {
  const info: string[] = [];
  const diagnostics = createWorkerDiagnostics({
    info: (message) => info.push(message),
  });

  diagnostics.info('chat-1 start');

  assert.deepEqual(info, ['chat-1 start']);
});

test('worker lines that report failures are not left at debug level', () => {
  assert.equal(workerLineLevel('0.07.060.888 W model has unused tensor blk.64.attn_norm.weight'), 'warn');
  assert.equal(workerLineLevel('1.02.000.001 E srv  send_error: task id = 0, error: failed'), 'error');
  assert.equal(workerLineLevel('0.01.608.035 I srv          init: The UI is disabled'), 'debug');
  assert.equal(workerLineLevel('[error] chat.stream failed: RuntimeError()'), 'error');
  assert.equal(workerLineLevel('[warning] generation config unreadable'), 'warn');
  assert.equal(workerLineLevel('Traceback (most recent call last):'), 'debug');
});

test('a line split across output chunks is reassembled before it is classified', () => {
  const buffer = createLineBuffer();
  assert.deepEqual(buffer.push('0.06.049.407 W'), []);
  assert.deepEqual(
    buffer.push(' model has unused tensor blk.64.ffn_down.weight -- ignoring\r\n0.06.049.412 W next'),
    ['0.06.049.407 W model has unused tensor blk.64.ffn_down.weight -- ignoring'],
  );
  assert.deepEqual(buffer.flush(), ['0.06.049.412 W next']);
  assert.deepEqual(buffer.flush(), []);
});
