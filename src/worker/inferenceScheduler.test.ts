import assert from 'node:assert/strict';
import test from 'node:test';
import { InferenceScheduler } from './inferenceScheduler.ts';

test('a queued chat cancels active inline inference and runs before later inline work', async () => {
  const scheduler = new InferenceScheduler();
  const events: string[] = [];
  let releaseInline: (() => void) | undefined;

  const firstInline = scheduler.run('inline', async (signal) => {
    events.push('inline:start');
    await new Promise<void>((resolve) => {
      releaseInline = resolve;
      signal.addEventListener('abort', () => resolve(), { once: true });
    });
    events.push(signal.aborted ? 'inline:aborted' : 'inline:done');
  });
  await new Promise<void>((resolve) => setImmediate(resolve));

  const chat = scheduler.run('chat', async () => {
    events.push('chat');
  });
  const secondInline = scheduler.run('inline', async () => {
    events.push('inline:second');
  });

  releaseInline?.();
  await Promise.all([
    assert.rejects(firstInline, (error: unknown) =>
      error instanceof Error && error.name === 'AbortError',
    ),
    chat,
    secondInline,
  ]);
  assert.deepEqual(events, [
    'inline:start',
    'inline:aborted',
    'chat',
    'inline:second',
  ]);
});

test('chat work is selected ahead of queued utility and inline work', async () => {
  const scheduler = new InferenceScheduler();
  const events: string[] = [];
  let releaseActive: (() => void) | undefined;

  const active = scheduler.run('utility', async () => {
    events.push('active');
    await new Promise<void>((resolve) => {
      releaseActive = resolve;
    });
  });
  await new Promise<void>((resolve) => setImmediate(resolve));

  const inline = scheduler.run('inline', async () => events.push('inline'));
  const utility = scheduler.run('utility', async () => events.push('utility'));
  const chat = scheduler.run('chat', async () => events.push('chat'));
  releaseActive?.();

  await Promise.all([active, inline, utility, chat]);
  assert.deepEqual(events, ['active', 'chat', 'utility', 'inline']);
});

test('active work rejects when cancelled even if the task ignores its signal', async () => {
  const scheduler = new InferenceScheduler();
  const controller = new AbortController();
  let release: (() => void) | undefined;

  const work = scheduler.run(
    'utility',
    async () => {
      await new Promise<void>((resolve) => {
        release = resolve;
      });
      return 42;
    },
    controller.signal,
  );
  await new Promise<void>((resolve) => setImmediate(resolve));

  controller.abort();
  release?.();

  await assert.rejects(work, (error: unknown) =>
    error instanceof Error && error.name === 'AbortError',
  );
});
