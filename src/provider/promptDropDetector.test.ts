import assert from 'node:assert/strict';
import test from 'node:test';
import { type DetectorTimers, type DroppedPrompt, PromptDropDetector } from './promptDropDetector.ts';

function manualTimers(): DetectorTimers & { fire(): void; pending(): number } {
  const scheduled = new Map<number, () => void>();
  let next = 0;
  return {
    set(callback) {
      next += 1;
      scheduled.set(next, callback);
      return next;
    },
    clear(handle) {
      scheduled.delete(handle as number);
    },
    fire() {
      const callbacks = [...scheduled.values()];
      scheduled.clear();
      for (const callback of callbacks) {
        callback();
      }
    },
    pending: () => scheduled.size,
  };
}

test('token counting that no chat request follows is reported as a dropped prompt', () => {
  const timers = manualTimers();
  const drops: DroppedPrompt[] = [];
  const detector = new PromptDropDetector((drop) => drops.push(drop), 3_000, timers);

  detector.counted('qwen38', 'Qwen3.8 27B', 45, 1);
  detector.counted('qwen38', 'Qwen3.8 27B', 1, 1);
  detector.counted('qwen38', 'Qwen3.8 27B', 9, 1);
  assert.equal(timers.pending(), 1, 'each count restarts one grace period rather than stacking');
  timers.fire();

  assert.deepEqual(drops, [{
    modelName: 'Qwen3.8 27B',
    counts: 3,
    countedTokens: 55,
    largestCount: 45,
    inputLimit: 1,
  }]);
});

test('a chat request after counting means the prompt was sent', () => {
  const timers = manualTimers();
  const drops: DroppedPrompt[] = [];
  const detector = new PromptDropDetector((drop) => drops.push(drop), 3_000, timers);

  detector.counted('qwen35-9b', 'Qwen3.5 9B', 1_200, 253_952);
  detector.sent('qwen35-9b');
  timers.fire();

  assert.deepEqual(drops, []);
});

test('models are tracked separately', () => {
  const timers = manualTimers();
  const drops: DroppedPrompt[] = [];
  const detector = new PromptDropDetector((drop) => drops.push(drop), 3_000, timers);

  detector.counted('a', 'Model A', 10, 100);
  detector.counted('b', 'Model B', 20, 100);
  detector.sent('a');
  timers.fire();

  assert.deepEqual(drops.map((drop) => drop.modelName), ['Model B']);
});

test('disposing cancels pending reports', () => {
  const timers = manualTimers();
  const drops: DroppedPrompt[] = [];
  const detector = new PromptDropDetector((drop) => drops.push(drop), 3_000, timers);

  detector.counted('a', 'Model A', 10, 100);
  detector.dispose();
  timers.fire();

  assert.deepEqual(drops, []);
  assert.equal(timers.pending(), 0);
});
