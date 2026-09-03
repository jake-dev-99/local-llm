import type { WorkerState } from '../domain';

export interface RuntimeStatusPresentation {
  text: string;
  tooltip: string;
  error: boolean;
}

export function runtimeStatusPresentation(
  state: WorkerState,
  modelCount: number,
): RuntimeStatusPresentation {
  switch (state.kind) {
    case 'ready':
      if (state.activity === 'generating-response') {
        return {
          text: '$(loading~spin) Local LLM: Generating Response',
          tooltip: 'Generating response',
          error: false,
        };
      }
      return {
        text: '$(sparkle) Local LLM',
        tooltip: 'Local model ready',
        error: false,
      };
    case 'starting':
      return {
        text: '$(loading~spin) Local LLM: Model Loading',
        tooltip: 'Loading model into memory',
        error: false,
      };
    case 'failed':
      return {
        text: '$(error) Local LLM',
        tooltip: state.message,
        error: true,
      };
    case 'stopping':
      return {
        text: '$(loading~spin) Local LLM',
        tooltip: 'Stopping local model',
        error: false,
      };
    case 'stopped':
      return {
        text: modelCount ? '$(circle-outline) Local LLM' : '$(add) Local LLM',
        tooltip: modelCount ? 'Local worker stopped' : 'Install a local GGUF model',
        error: false,
      };
  }
}
