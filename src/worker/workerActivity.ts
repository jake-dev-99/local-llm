import type { WorkerState } from '../domain.js';
import type { InferenceKind } from './inferenceScheduler.js';

export function beginWorkerActivity(
  state: WorkerState,
  kind: InferenceKind,
): WorkerState {
  if (kind !== 'chat' || state.kind !== 'ready') {
    return state;
  }
  return { ...state, activity: 'generating-response' };
}

export function finishWorkerActivity(state: WorkerState): WorkerState {
  if (state.kind !== 'ready' || state.activity !== 'generating-response') {
    return state;
  }
  const { activity: _activity, ...ready } = state;
  return ready;
}
