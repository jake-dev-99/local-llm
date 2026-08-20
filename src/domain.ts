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
  capabilities: ModelCapabilities;
  runtimeProfile?: ModelRuntimeProfile;
}

export type WorkerState =
  | { kind: 'stopped' }
  | { kind: 'starting'; modelId: string }
  | { kind: 'ready'; modelId: string; port: number }
  | { kind: 'stopping'; modelId: string }
  | { kind: 'failed'; modelId?: string; message: string };

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
  maxOutputTokens: number;
  maxToolCallTokens: number;
  startupTimeoutMilliseconds: number;
  temperature: number;
  inlineEnabled: boolean;
  inlineMaxTokens: number;
  inlineDebounceMilliseconds: number;
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
