/**
 * Wire contracts for the Safetensors runtime worker.
 *
 * These mirror the dataclasses in `resources/runtime/runtime/models.py`. The
 * worker owns the format; anything unrecognised here is dropped rather than
 * rejected, so a newer worker never breaks an older extension.
 */

/** Bumped by the worker whenever a request or response shape changes. */
export const SUPPORTED_PROTOCOL_VERSION = 1;

export type WorkerLifecycleState =
  | 'STARTING'
  | 'READY'
  | 'LOADING'
  | 'LOADED'
  | 'GENERATING'
  | 'UNLOADING'
  | 'FAILED'
  | 'STOPPED';

export interface PythonRuntimeInfo {
  protocolVersion: number;
  platform: string;
  pythonVersion: string;
  deviceType: string;
  deviceName: string;
  backend: string;
  versions: Record<string, string>;
}

export interface ModelInspection {
  path: string;
  modelType: string;
  architecture: string | null;
  isEncoderDecoder: boolean;
  supportsChat: boolean;
  contextLength: number | null;
  weightFormat: string;
  sharded: boolean;
  fileCount: number;
  weightBytes: number;
  dtype: string | null;
  quantization: string | null;
  customCodeRequired: boolean;
  adapter: boolean;
}

export interface PythonModelCapabilities {
  completion: boolean;
  chat: boolean;
  streaming: boolean;
  encoderDecoder: boolean;
  customCodeRequired: boolean;
  quantized: boolean;
  adapter: boolean;
  contextLength: number | null;
}

export interface PythonModelInfo {
  path: string;
  modelType: string;
  architecture: string | null;
  modelClass: string | null;
  tokenizerClass: string | null;
  dtype: string | null;
  isEncoderDecoder: boolean;
  supportsChat: boolean;
  contextLength: number | null;
  deviceMap: unknown;
  capabilities: PythonModelCapabilities;
  runtime: PythonRuntimeInfo;
}

/** One completion, with the prompt size the model actually saw. */
export interface PythonGenerationResult {
  text: string;
  inputTokens: number;
}

/**
 * A chat turn as the worker's template receives it. Tool records travel with
 * the message so a result arrives attributed to its call; HF templates render
 * the OpenAI shape natively.
 */
export interface PythonChatMessage {
  role: string;
  content: string;
  tool_call_id?: string;
  tool_calls?: Array<{
    id: string;
    type: 'function';
    function: { name: string; arguments: unknown };
  }>;
}

export interface PythonGenerationOptions {
  maxNewTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  repetitionPenalty?: number;
  /** JSON Schema constraining the output; the worker compiles it to a grammar. */
  jsonSchema?: unknown;
  /**
   * Reasoning-channel switch for the chat template. Absent leaves the
   * template default; true opens the thinking channel where supported.
   */
  enableThinking?: boolean;
}

export interface PythonRuntimePolicy {
  device?: string;
  allowCpuOffload?: boolean;
  allowDiskOffload?: boolean;
  trustRemoteCode?: boolean;
}

/**
 * Error codes the worker promises to use.
 *
 * Callers branch on the code, never on the message: upstream exception text
 * changes between Transformers releases and is only ever fit for logs.
 */
export type PythonErrorCode =
  | 'invalid_model'
  | 'unsupported_architecture'
  | 'missing_tokenizer'
  | 'missing_weights'
  | 'custom_code_required'
  | 'unsupported_quantization'
  | 'unsupported_grammar'
  | 'grammar_compile_failed'
  | 'device_unavailable'
  | 'insufficient_memory'
  | 'context_overflow'
  | 'model_load_failed'
  | 'model_not_loaded'
  | 'chat_not_supported'
  | 'generation_failed'
  | 'generation_cancelled'
  | 'generation_busy'
  | 'unknown_method'
  | 'invalid_request'
  | 'worker_crashed'
  | 'internal_error';

export class PythonWorkerError extends Error {
  readonly code: PythonErrorCode;

  constructor(code: PythonErrorCode, message: string) {
    super(message);
    this.name = 'PythonWorkerError';
    this.code = code;
  }
}
