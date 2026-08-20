import assert from 'node:assert/strict';
import test from 'node:test';
import { describeError, errorStack } from './errorDetail.ts';

test('a bare fetch failure reveals the cause underneath it', () => {
  // The exact shape Node produces when a request outlives its headers timeout.
  const cause = Object.assign(new Error('Headers Timeout Error'), {
    name: 'HeadersTimeoutError',
    code: 'UND_ERR_HEADERS_TIMEOUT',
  });
  const outer = Object.assign(new TypeError('fetch failed'), { cause });

  assert.equal(
    describeError(outer),
    'TypeError: fetch failed <- caused by HeadersTimeoutError: Headers Timeout Error '
      + '(code=UND_ERR_HEADERS_TIMEOUT)',
  );
});

test('system error fields survive into the description', () => {
  const error = Object.assign(new Error('connect ECONNREFUSED 127.0.0.1:51514'), {
    code: 'ECONNREFUSED',
    errno: -61,
    syscall: 'connect',
    address: '127.0.0.1',
    port: 51514,
  });

  assert.equal(
    describeError(error),
    'Error: connect ECONNREFUSED 127.0.0.1:51514 '
      + '(code=ECONNREFUSED, errno=-61, syscall=connect, address=127.0.0.1, port=51514)',
  );
});

test('a chain of causes is walked all the way down', () => {
  const root = new Error('socket hang up');
  const middle = Object.assign(new Error('request to worker failed'), { cause: root });
  const outer = Object.assign(new Error('chat request failed'), { cause: middle });

  assert.equal(
    describeError(outer),
    'Error: chat request failed <- caused by Error: request to worker failed '
      + '<- caused by Error: socket hang up',
  );
});

test('values thrown that are not errors are still reported', () => {
  assert.equal(describeError('worker exploded'), 'worker exploded');
  assert.equal(describeError({ reason: 'oom' }), '{"reason":"oom"}');
  assert.equal(describeError(undefined), 'no error detail available');
});

test('a cause cycle terminates instead of looping forever', () => {
  const first = new Error('first');
  const second = Object.assign(new Error('second'), { cause: first });
  (first as { cause?: unknown }).cause = second;

  const described = describeError(first);
  assert.ok(described.startsWith('Error: first <- caused by Error: second'));
  assert.ok(described.split('<- caused by').length <= 9);
});

test('the stack is available for the log line that follows', () => {
  assert.ok(errorStack(new Error('boom'))?.includes('Error: boom'));
  assert.equal(errorStack('not an error'), undefined);
});
