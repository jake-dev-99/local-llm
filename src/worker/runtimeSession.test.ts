import assert from 'node:assert/strict';
import test from 'node:test';
import {
  exitNotifier,
  runtimeDisplayName,
  runtimeForModel,
  type RuntimeExit,
} from './runtimeSession.ts';

test('a model carrying a runtime is routed to it', () => {
  assert.equal(runtimeForModel({ runtime: 'transformers', format: 'safetensors' }), 'transformers');
  assert.equal(runtimeForModel({ runtime: 'llama-cpp', format: 'gguf' }), 'llama-cpp');
});

test('the recorded runtime wins over what the format would imply', () => {
  // A format can outlive its default engine. The record is the decision; the
  // format is only the fallback for records written before there was one.
  assert.equal(runtimeForModel({ runtime: 'llama-cpp', format: 'safetensors' }), 'llama-cpp');
});

test('a record from before Safetensors support goes to llama.cpp', () => {
  // Every model installed then was GGUF, so an absent runtime is an answer
  // rather than a missing field to fail on.
  assert.equal(runtimeForModel({}), 'llama-cpp');
  assert.equal(runtimeForModel({ format: 'gguf' }), 'llama-cpp');
});

test('a format-only Safetensors record still finds its runtime', () => {
  assert.equal(runtimeForModel({ format: 'safetensors' }), 'transformers');
});

test('each runtime has a name a user would recognise', () => {
  assert.equal(runtimeDisplayName('llama-cpp'), 'llama.cpp');
  assert.equal(runtimeDisplayName('transformers'), 'Transformers');
});

test('an exit reaches every handler registered before it', () => {
  const exits = exitNotifier();
  const seen: RuntimeExit[] = [];
  exits.onExit((exit) => seen.push(exit));
  exits.onExit((exit) => seen.push(exit));

  assert.equal(exits.exited, false);
  exits.notify({ code: 1, signal: null });

  assert.equal(exits.exited, true);
  assert.deepEqual(seen, [{ code: 1, signal: null }, { code: 1, signal: null }]);
});

test('a handler registered after the exit is called immediately', () => {
  // This is the race the notifier exists for: a process that dies between
  // becoming healthy and being watched would otherwise leave the manager
  // holding a session that looks alive forever.
  const exits = exitNotifier();
  exits.notify({ code: null, signal: 'SIGKILL' });

  const seen: RuntimeExit[] = [];
  exits.onExit((exit) => seen.push(exit));

  assert.deepEqual(seen, [{ code: null, signal: 'SIGKILL' }]);
});

test('only the first exit counts', () => {
  const exits = exitNotifier();
  const seen: RuntimeExit[] = [];
  exits.onExit((exit) => seen.push(exit));

  exits.notify({ code: 0, signal: null });
  exits.notify({ code: 9, signal: 'SIGKILL' });

  assert.deepEqual(seen, [{ code: 0, signal: null }]);
});
