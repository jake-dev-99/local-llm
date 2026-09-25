import type {
  ModelFileFingerprint,
  ModelFormat,
  ModelRuntime,
} from './models/modelIdentity.ts';

export type { ModelFileFingerprint, ModelFormat, ModelRuntime };

export type ModelSource = 'import' | 'huggingface' | 'url';

export type CapabilitySupport = 'unverified' | 'supported' | 'unsupported';

export interface ModelCapabilities {
  toolCalling: CapabilitySupport;
  fillInMiddle: CapabilitySupport;
}

export interface ModelRuntimeProfile {
  validatedAt: string;
  loadedContextSize: number;
  hasChatTemplate: boolean;
  supportsTools: boolean;
  supportsToolCalls: boolean;
  supportsSystemRole: boolean;
  workerBuild?: string;
  chatTemplateFingerprint?: string;
}

export type NativeToolCallSupport = 'available' | 'unavailable';

export interface NativeToolCapabilityRecord {
  fingerprint: string;
  support: NativeToolCallSupport;
  observedAt: string;
}

export interface InstalledModel {
  id: string;
  name: string;
  filePath: string;
  fileSize: number;
  fileModifiedAt?: number;
  sha256: string;
  source: ModelSource;
  sourceIdentity?: string;
  sourceUrl?: string;
  repository?: string;
  revision?: string;
  filename: string;
  installedAt: string;
  /**
   * On-disk shape. A `gguf` model is the single file at `filePath`; a
   * `safetensors` model is the directory at `filePath`.
   */
  format: ModelFormat;
  /** The worker that can load this model. Recorded at install, never guessed. */
  runtime: ModelRuntime;
  /**
   * Per-file fingerprints, for directory models only.
   *
   * Change detection compares these rather than re-hashing: a Safetensors
   * checkpoint reaches tens of gigabytes and every model is verified on
   * activation. For a directory model `sha256` holds a manifest digest rather
   * than a content hash — see `models/modelIdentity.ts`.
   */
  files?: ModelFileFingerprint[];
  /** Quantization declared by a Safetensors checkpoint, when it declares one. */
  quantization?: string;
  /** True when loading this model would execute Python shipped inside it. */
  customCodeRequired?: boolean;
  /**
   * Whether the extension owns the bytes at `filePath` and may delete them.
   *
   * Absent means owned, which is correct for every record written before
   * checkpoints could be registered in place.
   */
  managed?: boolean;
  capabilities: ModelCapabilities;
  runtimeProfile?: ModelRuntimeProfile;
  nativeToolCapability?: NativeToolCapabilityRecord;
  /**
   * Context length from the GGUF header, or `config.json` for a Safetensors
   * checkpoint. A bootstrap estimate only; the window actually used comes from
   * the worker once the model has been fitted to this machine.
   */
  trainedContextLength?: number;
}

/**
 * The runtime is part of the state because two engines can now hold a model,
 * and "which one is loaded" is the first thing a status readout has to answer.
 * `port` is absent for a runtime that does not serve HTTP.
 */
export type WorkerState =
  | { kind: 'stopped' }
  | { kind: 'starting'; modelId: string; runtime: ModelRuntime }
  | {
      kind: 'ready';
      modelId: string;
      runtime: ModelRuntime;
      port?: number;
      activity?: 'generating-response';
    }
  | { kind: 'stopping'; modelId: string; runtime: ModelRuntime }
  | { kind: 'failed'; modelId?: string; runtime?: ModelRuntime; message: string };

export interface WorkerConfig {
  contextSize: number;
  cpuThreads: number;
  acceleration: 'auto' | 'cpu';
  batchSize: number;
  microBatchSize: number;
  metalMemoryReserveMiB: number;
}

export interface LocalLlmConfig extends WorkerConfig {
  modelDirectory: string;
  defaultModelId: string;
  maxTools: number;
  maxAgentToolRounds: number;
  maxOutputTokens: number;
  maxToolCallTokens: number;
  startupTimeoutMilliseconds: number;
  temperature: number;
  inlineEnabled: boolean;
  inlineMaxTokens: number;
  inlineDebounceMilliseconds: number;
  /**
   * Interpreter for the Safetensors runtime, or empty when none is set.
   *
   * Explicit rather than discovered: `python3` on PATH is usually the system
   * interpreter, and installing several gigabytes of PyTorch into it is not a
   * thing to do to a user's machine by default.
   */
  pythonPath: string;
  /**
   * Provisioned Python env flavor on Windows x64. `auto` detects: CUDA when
   * `nvidia-smi` succeeds, XPU when an Intel Arc GPU is reported, else CPU.
   * macOS has a single flavor; the setting is accepted and ignored there.
   */
  pythonEnvFlavor: 'auto' | 'cpu' | 'cuda' | 'xpu';
  logLevel: 'error' | 'info' | 'debug';
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  tool_call_id?: string;
  tool_calls?: Array<{
    id: string;
    type: 'function';
    function: { name: string; arguments: string };
  }>;
}

export interface ChatTool {
  type: 'function';
  function: {
    name: string;
    description?: string;
    parameters?: object;
  };
}

export interface ChatRequest {
  messages: ChatMessage[];
  tools?: ChatTool[];
  toolChoice?: 'auto' | 'required' | 'none';
  /** Internal llama.cpp compatibility override; the caller-facing contract remains toolChoice. */
  workerToolChoice?: 'auto' | 'required' | 'none';
  inputTokenBudget: number;
  maxTokens: number;
  toolCallMaxTokens?: number;
  temperature: number;
  /**
   * Accept a reply that ends inside the model's reasoning. Only a measurement
   * of generation speed wants this: reasoning tokens are generated tokens.
   * An empty generation is still a failure.
   */
  allowReasoningOnly?: boolean;
}

export type ChatStreamEvent =
  | { kind: 'text'; text: string }
  | { kind: 'toolCall'; id: string; name: string; input: object };

export interface InfillRequest {
  prefix: string;
  suffix: string;
  maxTokens: number;
  temperature: number;
  stop?: string[];
}
