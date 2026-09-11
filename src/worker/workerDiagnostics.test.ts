import assert from 'node:assert/strict';
import test from 'node:test';
import { createWorkerDiagnostics, WorkerStartupDiagnostics } from './workerDiagnostics.ts';

test('startup failures report chunked native output at error level once', () => {
  const errors: string[] = [];
  const diagnostics = new WorkerStartupDiagnostics({ error: (message) => errors.push(message) });
  diagnostics.append('SYCL failed to allo');
  diagnostics.append('cate memory\r\nmodel load failed');
  diagnostics.reportFailure();
  diagnostics.reportFailure();
  assert.deepEqual(errors, [
    '[Model Loading] Recent worker startup output:',
    'worker: SYCL failed to allocate memory',
    'worker: model load failed',
  ]);
});

test('startup capture keeps only a bounded tail', () => {
  const errors: string[] = [];
  const diagnostics = new WorkerStartupDiagnostics({ error: (message) => errors.push(message) });
  diagnostics.append('old output\n' + 'x'.repeat(32 * 1024));
  diagnostics.append('\nfatal error');
  diagnostics.reportFailure();
  assert.equal(errors.join('\n').includes('old output'), false);
  assert.ok(errors.join('\n').length < 17 * 1024);
  assert.equal(errors.at(-1), 'worker: fatal error');
});

test('ready workers discard startup output and never capture chat output', () => {
  const errors: string[] = [];
  const diagnostics = new WorkerStartupDiagnostics({ error: (message) => errors.push(message) });
  diagnostics.append('startup');
  diagnostics.complete();
  diagnostics.append('private chat output');
  diagnostics.reportFailure();
  assert.deepEqual(errors, []);
});

test('silent startup failures explicitly report missing native output', () => {
  const errors: string[] = [];
  const diagnostics = new WorkerStartupDiagnostics({ error: (message) => errors.push(message) });
  diagnostics.reportFailure();
  assert.equal(errors.at(-1), 'worker: <no startup output captured>');
});

test('worker diagnostics forward request milestones', () => {
  const info: string[] = [];
  const diagnostics = createWorkerDiagnostics({
    info: (message) => info.push(message),
  });

  diagnostics.info('chat-1 start');

  assert.deepEqual(info, ['chat-1 start']);
});
