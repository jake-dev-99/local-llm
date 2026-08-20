import * as vscode from 'vscode';
import { readConfig } from '../config';
import type { InstalledModel, ModelRuntimeProfile } from '../domain';
import type { LocalLlmLogger } from '../logging';
import type { ModelRegistry } from '../models/modelRegistry';
import type { WorkerManager } from '../worker/workerManager';
import { isFatalWorkerError } from '../worker/workerError';
import { modelTokenLimits } from './modelCapacity';
import {
  adaptMessages,
  adaptTools,
  serializeMessageForTokenCount,
} from './messageAdapter';
import { isLocalAgentRequest } from './localAgentToolChoice';
import {
  localAgentAvailableTools,
  localAgentDiscoveryTools,
  localAgentNeedsMutationTool,
  requiresLocalAgentTool,
  resolveLocalAgentToolPolicy,
} from './localAgentTools';
import { messagesForSystemRoleSupport } from './messageRoleSupport';

export interface LocalLanguageModelInformation extends vscode.LanguageModelChatInformation {
  readonly installedModelId: string;
}

export class LocalLanguageModelProvider
implements vscode.LanguageModelChatProvider<LocalLanguageModelInformation>, vscode.Disposable {
  private readonly changeEmitter = new vscode.EventEmitter<void>();
  private readonly registrySubscription: vscode.Disposable;

  readonly onDidChangeLanguageModelChatInformation = this.changeEmitter.event;

  constructor(
    private readonly context: vscode.ExtensionContext,
    private readonly registry: ModelRegistry,
    private readonly worker: WorkerManager,
    private readonly logger: LocalLlmLogger,
  ) {
    this.registrySubscription = registry.onDidChange(() => this.changeEmitter.fire());
  }

  provideLanguageModelChatInformation(
    _options: vscode.PrepareLanguageModelChatModelOptions,
    _token: vscode.CancellationToken,
  ): LocalLanguageModelInformation[] {
    const config = readConfig(this.context);
    return this.registry.list()
      .filter((model) => model.runtimeProfile?.hasChatTemplate !== false)
      .map((model) => this.modelInformation(
        model,
        config.contextSize,
        config.maxOutputTokens,
        config.maxTools,
      ));
  }

  async provideLanguageModelChatResponse(
    model: LocalLanguageModelInformation,
    messages: readonly vscode.LanguageModelChatRequestMessage[],
    options: vscode.ProvideLanguageModelChatResponseOptions,
    progress: vscode.Progress<vscode.LanguageModelResponsePart>,
    token: vscode.CancellationToken,
  ): Promise<void> {
    const installed = this.requireModel(model.installedModelId);
    const config = readConfig(this.context);
    const cancellation = toAbortSignal(token);
    const tools = adaptTools(options.tools);
    const adaptedMessages = adaptMessages(messages);
    const localAgentRequest = isLocalAgentRequest(adaptedMessages);
    const availableTools = localAgentRequest
      ? localAgentAvailableTools(tools ?? [], adaptedMessages)
      : tools ?? [];
    const needsMutationTool = localAgentRequest &&
      localAgentNeedsMutationTool(adaptedMessages);
    // Never gate this on how many tools arrived. An empty tool list is exactly the
    // case that must reach resolveLocalAgentToolPolicy so it can refuse the turn.
    const requireLocalAgentTool = requiresLocalAgentTool(adaptedMessages);
    const discoveryTools = localAgentRequest
      ? localAgentDiscoveryTools(availableTools)
      : [];
    this.logger.debug(
      `Chat request adapted ${adaptedMessages.length} message${adaptedMessages.length === 1 ? '' : 's'} with roles: ${adaptedMessages.map((message) => message.role).join(', ')}.`,
    );
    this.logger.debug(
      tools?.length
        ? `Chat request supplied ${tools.length} tool contracts: ${tools.map((tool) => tool.function.name).join(', ')}.`
        : 'Chat request supplied no tool contracts.',
    );
    let toolPolicy: ReturnType<typeof resolveLocalAgentToolPolicy>;
    try {
      toolPolicy = resolveLocalAgentToolPolicy(
        availableTools,
        options.toolMode === vscode.LanguageModelChatToolMode.Required,
        requireLocalAgentTool,
        needsMutationTool,
      );
    } catch (error) {
      cancellation.dispose();
      throw error;
    }
    if (toolPolicy.tools.length > config.maxTools) {
      cancellation.dispose();
      throw new Error(
        `This request supplied ${toolPolicy.tools.length} authorized tools, but the local model is configured for at most ${config.maxTools}. Select the bundled Local Agent or increase localLlm.maxTools deliberately.`,
      );
    }
    if (
      toolPolicy.source === 'local-agent-discovery'
    ) {
      this.logger.debug(
        `Local Agent workspace turn needs discovery; requiring one of ${discoveryTools.length} read-only tools.`,
      );
    }
    if (toolPolicy.source === 'local-agent-mutation') {
      this.logger.debug(
        `Local Agent workspace turn requires one authorized editor: ${toolPolicy.tools.map((tool) => tool.function.name).join(', ')}.`,
      );
    }
    const temperature = clamp(
      numericOption(options.modelOptions, 'temperature', config.temperature),
      0,
      2,
    );
    try {
      await this.worker.run(
        installed,
        'chat',
        async (client, signal) => {
          const profile = await client.getModelProfile(signal);
          const profileMatchesValidatedWorker = Boolean(
            installed.runtimeProfile?.workerBuild &&
            profile.workerBuild &&
            installed.runtimeProfile.workerBuild === profile.workerBuild,
          );
          await this.recordRuntimeProfile(installed.id, profile);
          if (
            installed.capabilities.toolCalling === 'supported' &&
            !profileMatchesValidatedWorker
          ) {
            await this.registry.markToolCalling(installed.id, 'unverified');
          }
          if (!profile.hasChatTemplate) {
            throw new Error(
              'This GGUF does not expose a llama.cpp chat template. Install an instruct/chat GGUF for VS Code Chat.',
            );
          }
          if (toolPolicy.tools.length && (!profile.supportsTools || !profile.supportsToolCalls)) {
            await this.registry.markToolCalling(installed.id, 'unsupported');
            throw new Error(
              'This GGUF chat template does not support structured llama.cpp tool calls. Use Chat without tools or install a tool-use instruct model.',
            );
          }
          if (
            toolPolicy.tools.length &&
            (
              installed.capabilities.toolCalling !== 'supported' ||
              !profileMatchesValidatedWorker
            )
          ) {
            throw new Error(
              'Agent tool calling is disabled for this model until it passes Local LLM: Validate Model Compatibility.',
            );
          }
          const physicalContext = Math.max(
            2,
            Math.min(config.contextSize, profile.loadedContextSize),
          );
          const maxTokens = clamp(
            numericOption(options.modelOptions, 'maxTokens', config.maxOutputTokens),
            1,
            physicalContext - 1,
          );
          const modelMessages = messagesForSystemRoleSupport(
            adaptedMessages,
            profile.supportsSystemRole,
          );
          const result = await client.chat(
            {
              messages: modelMessages,
              ...(toolPolicy.tools.length ? {
                tools: toolPolicy.tools,
              } : {}),
              toolChoice: toolPolicy.toolChoice,
              ...(toolPolicy.source === 'local-agent-discovery'
                ? { workerToolChoice: 'auto' as const }
                : {}),
              inputTokenBudget: physicalContext - maxTokens,
              maxTokens,
              toolCallMaxTokens: toolPolicy.source === 'local-agent-discovery'
                ? Math.min(config.maxToolCallTokens, 256)
                : config.maxToolCallTokens,
              temperature,
            },
            (event) => {
              if (event.kind === 'text') {
                progress.report(new vscode.LanguageModelTextPart(event.text));
              } else {
                progress.report(
                  new vscode.LanguageModelToolCallPart(event.id, event.name, event.input),
                );
              }
            },
            signal,
          );
          if (result.toolCallCount > 0) {
            await this.registry.markToolCalling(installed.id, 'supported');
          }
        },
        cancellation.signal,
      );
    } catch (error) {
      if (token.isCancellationRequested || isAbortError(error)) {
        throw new vscode.CancellationError();
      }
      this.logger.error(`Local chat request failed for ${installed.name}`, error);
      throw error;
    } finally {
      cancellation.dispose();
    }
  }

  async provideTokenCount(
    model: LocalLanguageModelInformation,
    text: string | vscode.LanguageModelChatRequestMessage,
    token: vscode.CancellationToken,
  ): Promise<number> {
    const installed = this.requireModel(model.installedModelId);
    const cancellation = toAbortSignal(token);
    try {
      return await this.worker.run(
        installed,
        'utility',
        async (client, signal) => {
          if (typeof text === 'string') {
            return client.tokenize(text, signal);
          }
          try {
            const profile = await client.getModelProfile(signal);
            const modelMessages = messagesForSystemRoleSupport(
              adaptMessages([text]),
              profile.supportsSystemRole,
            );
            return await client.countChatInputTokens(modelMessages, undefined, 'auto', signal);
          } catch (error) {
            if (isFatalWorkerError(error)) {
              throw error;
            }
            this.logger.debug(
              `Exact message token count unavailable; using tokenizer fallback: ${error instanceof Error ? error.message : String(error)}`,
            );
            return client.tokenize(serializeMessageForTokenCount(text), signal);
          }
        },
        cancellation.signal,
      );
    } finally {
      cancellation.dispose();
    }
  }

  dispose(): void {
    this.registrySubscription.dispose();
    this.changeEmitter.dispose();
  }

  refresh(): void {
    this.changeEmitter.fire();
  }

  private modelInformation(
    model: InstalledModel,
    contextSize: number,
    maxOutputTokens: number,
    maxTools: number,
  ): LocalLanguageModelInformation {
    const limits = modelTokenLimits(contextSize, maxOutputTokens);
    const toolStatus = model.capabilities.toolCalling === 'supported'
      ? 'tool calls validated'
      : model.capabilities.toolCalling === 'unsupported'
        ? 'tool calls unavailable'
        : 'run model validation to enable Agent';
    return {
      id: model.id,
      installedModelId: model.id,
      name: model.name,
      family: 'gguf',
      version: model.sha256.slice(0, 12),
      tooltip: `Local GGUF model · ${model.filename} · ${toolStatus}`,
      detail: 'Runs entirely on this computer',
      maxInputTokens: limits.maxInputTokens,
      maxOutputTokens: limits.maxOutputTokens,
      capabilities: {
        imageInput: false,
        toolCalling: model.capabilities.toolCalling === 'supported' ? maxTools : false,
      },
    };
  }

  private requireModel(id: string): InstalledModel {
    const model = this.registry.get(id);
    if (!model) {
      throw new Error(`Local model ${id} is no longer installed.`);
    }
    return model;
  }

  private async recordRuntimeProfile(
    modelId: string,
    profile: import('../worker/runtimeProfile').WorkerModelProfile,
  ): Promise<void> {
    const current = this.registry.get(modelId)?.runtimeProfile;
    if (
      current &&
      current.loadedContextSize === profile.loadedContextSize &&
      current.hasChatTemplate === profile.hasChatTemplate &&
      current.supportsTools === profile.supportsTools &&
      current.supportsToolCalls === profile.supportsToolCalls &&
      current.supportsSystemRole === profile.supportsSystemRole &&
      current.workerBuild === profile.workerBuild
    ) {
      return;
    }
    const stored: ModelRuntimeProfile = {
      ...profile,
      validatedAt: new Date().toISOString(),
    };
    await this.registry.updateRuntimeProfile(modelId, stored);
  }
}

function numericOption(
  options: { readonly [name: string]: unknown } | undefined,
  name: string,
  fallback: number,
): number {
  const value = options?.[name];
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError';
}

export function toAbortSignal(
  token: vscode.CancellationToken,
): AbortController & { dispose(): void } {
  const controller = new AbortController() as AbortController & { dispose(): void };
  if (token.isCancellationRequested) {
    controller.abort();
  }
  const subscription = token.onCancellationRequested(() => controller.abort());
  controller.dispose = () => subscription.dispose();
  return controller;
}
