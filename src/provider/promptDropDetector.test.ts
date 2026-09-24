import assert from 'node:assert/strict';
import test from 'node:test';
import { type DetectorTimers, type DroppedPrompt, PromptDropDetector } from './promptDropDetector.ts';

let clock = 0;

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
  const detector = new PromptDropDetector((drop) => drops.push(drop), { timers, now: () => clock });

  detector.counted('qwen38', 'Qwen3.8 27B', 45, 1);
  detector.counted('qwen38', 'Qwen3.8 27B', 1, 1);
  detector.counted('qwen38', 'Qwen3.8 27B', 9, 1);
  assert.equal(timers.pending(), 1, 'each count restarts one grace period rather than stacking');
  timers.fire();

  assert.deepEqual(drops, [{
    modelName: 'Qwen3.8 27B',
    counts: 3,
    largestCount: 45,
    inputLimit: 1,
  }]);
});

test('a chat request after counting means the prompt was sent', () => {
  const timers = manualTimers();
  const drops: DroppedPrompt[] = [];
  const detector = new PromptDropDetector((drop) => drops.push(drop), { timers, now: () => clock });

  detector.counted('qwen35-9b', 'Qwen3.5 9B', 1_200, 253_952);
  detector.sent('qwen35-9b');
  timers.fire();

  assert.deepEqual(drops, []);
});

test('models are tracked separately', () => {
  const timers = manualTimers();
  const drops: DroppedPrompt[] = [];
  const detector = new PromptDropDetector((drop) => drops.push(drop), { timers, now: () => clock });

  detector.counted('a', 'Model A', 10, 100);
  detector.counted('b', 'Model B', 20, 100);
  detector.sent('a');
  timers.fire();

  assert.deepEqual(drops.map((drop) => drop.modelName), ['Model B']);
});

test('disposing cancels pending reports', () => {
  const timers = manualTimers();
  const drops: DroppedPrompt[] = [];
  const detector = new PromptDropDetector((drop) => drops.push(drop), { timers, now: () => clock });

  detector.counted('a', 'Model A', 10, 100);
  detector.dispose();
  timers.fire();

  assert.deepEqual(drops, []);
  assert.equal(timers.pending(), 0);
});

test('counting right after a reply is VS Code bookkeeping, not a dropped prompt', () => {
  // Observed: a Qwen3.5 reply completed, then VS Code counted tokens 486 times
  // for its context display and sent nothing, because nothing was pending.
  const timers = manualTimers();
  const drops: DroppedPrompt[] = [];
  const detector = new PromptDropDetector((drop) => drops.push(drop), { timers, now: () => clock });

  clock = 100_000;
  detector.sent('qwen35-4b');
  clock = 112_000;
  detector.finished('qwen35-4b');
  clock = 112_007;
  detector.counted('qwen35-4b', 'Qwen3.5 4B', 2_809, 16_384);
  clock = 112_500;
  detector.counted('qwen35-4b', 'Qwen3.5 4B', 1, 16_384);
  timers.fire();

  assert.deepEqual(drops, []);
});

test('a prompt counted well after the last reply is watched again', () => {
  const timers = manualTimers();
  const drops: DroppedPrompt[] = [];
  const detector = new PromptDropDetector((drop) => drops.push(drop), { timers, now: () => clock });

  clock = 200_000;
  detector.finished('qwen38');
  clock = 260_000;
  detector.counted('qwen38', 'Qwen3.8 27B', 45, 1);
  timers.fire();

  assert.equal(drops.length, 1);
});
