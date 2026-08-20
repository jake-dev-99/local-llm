import assert from 'node:assert/strict';
import test from 'node:test';
import { workerRequestError } from './workerError.ts';

test('workerRequestError includes a structured llama-server error message', () => {
  const body = JSON.stringify({
    error: {
      code: 400,
      message: 'request (8779 tokens) exceeds the available context size (8192 tokens)',
      type: 'exceed_context_size_error',
    },
  });

  const error = workerRequestError('/v1/chat/completions', 400, body);
  assert.equal(
    error.message,
    'Local worker request /v1/chat/completions failed with HTTP 400: request (8779 tokens) exceeds the available context size (8192 tokens).',
  );
  assert.equal(error.name, 'Error');
});

test('workerRequestError classifies an HTTP compute failure as fatal', () => {
  const error = workerRequestError(
    '/v1/chat/completions',
    500,
    JSON.stringify({
      error: { code: 500, message: 'Compute error', type: 'server_error' },
    }),
  );

  assert.equal(error.name, 'LocalWorkerFatalError');
});

test('workerStreamError leaves a streamed invalid request nonfatal', async () => {
  const workerErrors = await import('./workerError.ts');
  const error = workerErrors.workerStreamError('/v1/chat/completions', {
    code: 400,
    message: 'Invalid tool_choice value',
    type: 'invalid_request_error',
  });

  assert.equal(error.name, 'Error');
});
