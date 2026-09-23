import * as vscode from 'vscode';
import { readConfig } from '../config.js';
import type { InstalledModel, ModelRuntimeProfile } from '../domain.js';
import type { LocalLlmLogger } from '../logging.js';
import type { ModelRegistry } from '../models/modelRegistry.js';
import type { WorkerManager } from '../worker/workerManager.js';
import {
  matchingNativeToolSupport,
  nativeToolCapabilityFingerprint,
  TOOL_PROTOCOL_VERSION,
} from '../worker/nativeToolCapability.js';
import { runtimeForModel } from '../worker/runtimeSession.js';
import { isFatalWorkerError } from '../worker/workerError.js';
import { modelTokenLimits, resolveAdvertisedContextSize } from './modelCapacity.js';
import {
  adaptMessages,
  adaptTools,
  serializeMessageForTokenCount,
} from './messageAdapter.js';
import {
  isLocalAgentRequest,
  localAgentToolInvocationCount,
  localAgentToolLimitReached,
} from './localAgentToolChoice.js';
import {
  localAgentAvailableTools,
  resolveLocalAgentToolPolicy,
} from './localAgentTools.js';
import { modelLoadProgress } from './modelLoadProgress.js';
import { type DroppedPrompt, PromptDropDetector } from './promptDropDetector.js';
import { messagesForSystemRoleSupport } from './messageRoleSupport.js';

export interface LocalLanguageModelInformation extends vscode.LanguageModelChatInformation {
  readonly installedModelId: string;
}

export class LocalLanguageModelProvider
implements vscode.LanguageModelChatProvider<LocalLanguageModelInformation>, vscode.Disposable {
  private readonly changeEmitter = new vscode.EventEmitter<void>();
  private readonly registrySubscription: vscode.Disposable;

  readonly onDidChangeLanguageModelChatInformation = this.changeEmitter.event;
  private lastAdvertised: string | undefined;
  private readonly promptDrops: PromptDropDetector;

  constructor(
    private readonly context: vscode.ExtensionContext,
    private readonly registry: ModelRegistry,
    private readonly worker: WorkerManager,
    private readonly logger: LocalLlmLogger,
  ) {
    this.registrySubscription = registry.onDidChange(() => this.changeEmitter.fire());
    this.promptDrops = new PromptDropDetector((drop) => this.reportDroppedPrompt(drop));
  }

  provideLanguageModelChatInformation(
    _options: vscode.PrepareLanguageModelChatModelOptions,
    _token: vscode.CancellationToken,
  ): LocalLanguageModelInformation[] {
    const config = readConfig(this.context);
    const installed = this.registry.list();
    const hidden = installed.filter((model) => model.runtimeProfile?.hasChatTemplate === false);
    const models = installed
      .filter((model) => model.runtimeProfile?.hasChatTemplate !== false)
      .map((model) => this.modelInformation(
        model,
        config.contextSize,
        config.maxOutputTokens,
        config.maxTools,
      ));
    // VS Code budgets a chat prompt against these limits before it sends the
    // request, so a prompt that does not fit never reaches this provider.
    const advertised = models.map((model) =>
      `${model.name}: input ${model.maxInputTokens}, output ${model.maxOutputTokens}, ` +
      `tools ${model.capabilities.toolCalling === false ? 'off' : 'on'}`,
    ).join('; ') + (hidden.length
      ? ` | hidden from the model picker (no chat template): ${hidden.map((model) => model.name).join(', ')}`
      : '');
    if (advertised !== this.lastAdvertised) {
      this.lastAdvertised = advertised;
      this.logger.info(`Advertised to VS Code: ${advertised || 'no models'}.`);
      for (const model of models) {
        if (model.maxOutputTokens < config.maxOutputTokens) {
          this.logger.warn(
            `${model.name} has a ${model.maxInputTokens + model.maxOutputTokens}-token context window on this computer, ` +
            `so replies are limited to ${model.maxOutputTokens} tokens (localLlm.maxOutputTokens is ${config.maxOutputTokens}) ` +
            `and prompts to ${model.maxInputTokens}. VS Code drops any chat prompt larger than that without sending it; ` +
            'agent prompts with tools often are.',
          );
        }
      }
    }
    return models;
  }

  /**
   * Every chat submission is logged on arrival and every way it ends is logged,
   * including a failure while adapting the request before any generation.
   */
  async provideLanguageModelChatResponse(
    model: LocalLanguageModelInformation,
    messages: readonly vscode.LanguageModelChatRequestMessage[],
    options: vscode.ProvideLanguageModelChatResponseOptions,
    progress: vscode.Progress<vscode.LanguageModelResponsePart>,
    token: vscode.CancellationToken,
  ): Promise<void> {
    this.promptDrops.sent(model.installedModelId);
    this.logger.info(
      `Chat request received for ${model.name}: ${messages.length} message${messages.length === 1 ? '' : 's'}, ` +
      `${options.tools?.length ?? 0} tool${options.tools?.length === 1 ? '' : 's'}.`,
    );
    try {
      await this.respond(model, messages, options, progress, token);
      this.logger.info(`Chat request for ${model.name} completed.`);
    } catch (error) {
      if (error instanceof vscode.CancellationError) {
        this.logger.info(`Chat request for ${model.name} ended: cancelled.`);
      } else {
        this.logger.error(`Local chat request failed for ${model.name}`, error, true);
      }
      throw error;
    }
  }

  private async respond(
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
      ? localAgentAvailableTools(tools ?? [])
      : tools ?? [];
    const invocationCount = localAgentToolInvocationCount(adaptedMessages);
    const forceLocalAgentFinal = localAgentToolLimitReached(
      adaptedMessages,
      config.maxAgentToolRounds,
    );
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
        forceLocalAgentFinal,
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
    if (localAgentRequest) {
      this.logger.debug(
        `Local Agent tool allowance: ${invocationCount}/${config.maxAgentToolRounds} invoked; final generation ${forceLocalAgentFinal ? 'required' : 'model-selected'}.`,
      );
    }
    const temperature = clamp(
      numericOption(options.modelOptions, 'temperature', config.temperature),
      0,
      2,
    );
    const loadProgress = modelLoadProgress({
      modelResident:
        this.worker.currentModelId === installed.id && this.worker.state.kind === 'ready',
      toolCallsPossible: toolPolicy.tools.length > 0,
      agentRequest: localAgentRequest,
    });
    if (loadProgress) {
      progress.report(new vscode.LanguageModelTextPart(`${loadProgress.loading}\n\n`));
    }
    try {
      await this.worker.run(
        installed,
        'chat',
        async (client, signal) => {
          if (loadProgress) {
            progress.report(new vscode.LanguageModelTextPart(`${loadProgress.loaded}\n\n`));
          }
          const profile = await client.getModelProfile(signal);
          const nativeCapabilityFingerprint = toolPolicy.tools.length > 0 &&
            profile.workerBuild && profile.chatTemplateFingerprint
            ? nativeToolCapabilityFingerprint({
              modelSha256: installed.sha256,
              workerBuild: profile.workerBuild,
              chatTemplateFingerprint: profile.chatTemplateFingerprint,
              platform: `${process.platform}-${process.arch}`,
              toolProtocolVersion: TOOL_PROTOCOL_VERSION,
            })
            : undefined;
          if (nativeCapabilityFingerprint) {
            const persisted = this.registry.get(installed.id)?.nativeToolCapability;
            const support = matchingNativeToolSupport(
              persisted,
              nativeCapabilityFingerprint,
            );
            client.setNativeToolCallSupport(support);
            this.logger.debug(
              `Native tool capability ${support} for fingerprint ${nativeCapabilityFingerprint.slice(0, 12)}.`,
            );
          }
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
          // These two refusals are the same check for both runtimes but not
          // the same advice: telling someone holding a Safetensors checkpoint
          // to install a tool-use GGUF would send them after the wrong thing.
          const runtime = runtimeForModel(installed);
          if (!profile.hasChatTemplate) {
            throw new Error(runtime === 'transformers'
              ? 'This checkpoint ships no chat template, so its tokenizer cannot format a conversation. Install an instruct or chat variant for VS Code Chat.'
              : 'This GGUF does not expose a llama.cpp chat template. Install an instruct/chat GGUF for VS Code Chat.');
          }
          if (toolPolicy.tools.length && (!profile.supportsTools || !profile.supportsToolCalls)) {
            await this.registry.markToolCalling(installed.id, 'unsupported');
            throw new Error(runtime === 'transformers'
              ? 'Safetensors models cannot use tools yet: this runtime has no schema-constrained decoding, so a tool call would come back as unparseable prose. Use Chat without tools, or select a GGUF model for Local Agent.'
              : 'This GGUF chat template does not support structured llama.cpp tool calls. Use Chat without tools or install a tool-use instruct model.');
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
          // In automatic mode localLlm.contextSize is zero, so clamping against it
          // would yield a two token window. The worker's loaded window is the truth.
          const physicalContext = config.contextSize > 0
            ? Math.max(2, Math.min(config.contextSize, profile.loadedContextSize))
            : Math.max(2, profile.loadedContextSize);
          // The same split VS Code was told about, so a request it sends fits.
          const { maxOutputTokens: maxTokens } = modelTokenLimits(
            physicalContext,
            numericOption(options.modelOptions, 'maxTokens', config.maxOutputTokens),
          );
          const modelMessages = messagesForSystemRoleSupport(
            adaptedMessages,
            profile.supportsSystemRole,
          );
          const chatRequest = {
            messages: modelMessages,
            ...(toolPolicy.tools.length ? { tools: toolPolicy.tools } : {}),
            toolChoice: toolPolicy.toolChoice,
            inputTokenBudget: physicalContext - maxTokens,
            maxTokens,
            toolCallMaxTokens: config.maxToolCallTokens,
            temperature,
          };
          let observedToolCall = false;
          try {
            const result = await client.chat(
              chatRequest,
              (event) => reportChatEvent(progress, event),
              signal,
            );
            observedToolCall = result.toolCallCount > 0;
          } finally {
            if (nativeCapabilityFingerprint) {
              const support = client.getNativeToolCallSupport();
              const persisted = this.registry.get(installed.id)?.nativeToolCapability;
              if (
                support !== 'unknown' &&
                (
                  persisted?.fingerprint !== nativeCapabilityFingerprint ||
                  persisted.support !== support
                )
              ) {
                try {
                  await this.registry.updateNativeToolCapability(installed.id, {
                    fingerprint: nativeCapabilityFingerprint,
                    support,
                    observedAt: new Date().toISOString(),
                  });
                } catch (error) {
                  client.setNativeToolCallSupport('unknown');
                  this.logger.error('Could not persist native tool capability.', error);
                }
              }
            }
          }
          if (observedToolCall) {
            await this.registry.markToolCalling(installed.id, 'supported');
          }
        },
        cancellation.signal,
      );
    } catch (error) {
      if (token.isCancellationRequested || isAbortError(error)) {
        this.logger.info(
          `Chat request for ${installed.name} cancelled ` +
          `(${token.isCancellationRequested ? 'by VS Code' : 'inside the local runtime'}).`,
        );
        throw new vscode.CancellationError();
      }
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
      const count = await this.worker.run(
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
      this.promptDrops.counted(installed.id, installed.name, count, model.maxInputTokens);
      this.logger.debug(
        `Token count for ${installed.name}: ${count} tokens ` +
        `(${typeof text === 'string' ? 'text' : `${roleName(text.role)} message`}; ` +
        `advertised input limit ${model.maxInputTokens}).`,
      );
      return count;
    } catch (error) {
      if (token.isCancellationRequested || isAbortError(error)) {
        this.logger.info(
          `Token count for ${installed.name} cancelled ` +
          `(${token.isCancellationRequested ? 'by VS Code' : 'inside the local runtime'}).`,
        );
      } else {
        this.logger.error(`Token count failed for ${installed.name}`, error);
      }
      throw error;
    } finally {
      cancellation.dispose();
    }
  }

  dispose(): void {
    this.promptDrops.dispose();
    this.registrySubscription.dispose();
    this.changeEmitter.dispose();
  }

  refresh(): void {
    this.changeEmitter.fire();
  }

  /**
   * VS Code measured a prompt and then never sent it. It says nothing when it
   * does this, so the provider has to.
   */
  private reportDroppedPrompt(drop: DroppedPrompt): void {
    const measured = `VS Code counted tokens ${drop.counts} time${drop.counts === 1 ? '' : 's'} ` +
      `for a ${drop.modelName} prompt (largest piece ${drop.largestCount} tokens) against its ` +
      `${drop.inputLimit}-token input limit, then did not send the request.`;
    if (drop.countedTokens > drop.inputLimit) {
      this.logger.error(
        `Chat request to ${drop.modelName} was dropped by VS Code: the prompt does not fit. ${measured} ` +
        'Use a model with a larger context window on this computer, or a smaller prompt ' +
        '(plain Chat instead of Agent, fewer attachments).',
        undefined,
        true,
      );
      return;
    }
    this.logger.warn(
      `Chat request to ${drop.modelName} was not sent by VS Code, although the prompt fits. ${measured} ` +
      'The reason is inside VS Code: see Output > GitHub Copilot Chat.',
      true,
    );
  }

  private modelInformation(
    model: InstalledModel,
    configuredContextSize: number,
    maxOutputTokens: number,
    maxTools: number,
  ): LocalLanguageModelInformation {
    const contextSize = resolveAdvertisedContextSize({
      configuredContextSize,
      ...(model.runtimeProfile?.loadedContextSize
        ? { loadedContextSize: model.runtimeProfile.loadedContextSize }
        : {}),
      ...(model.trainedContextLength
        ? { trainedContextLength: model.trainedContextLength }
        : {}),
    });
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
    profile: import('../worker/runtimeProfile.js').WorkerModelProfile,
  ): Promise<void> {
    const current = this.registry.get(modelId)?.runtimeProfile;
    if (
      current &&
      current.loadedContextSize === profile.loadedContextSize &&
      current.hasChatTemplate === profile.hasChatTemplate &&
      current.supportsTools === profile.supportsTools &&
      current.supportsToolCalls === profile.supportsToolCalls &&
      current.supportsSystemRole === profile.supportsSystemRole &&
      current.workerBuild === profile.workerBuild &&
      current.chatTemplateFingerprint === profile.chatTemplateFingerprint
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

function reportChatEvent(
  progress: vscode.Progress<vscode.LanguageModelResponsePart>,
  event: import('../domain.js').ChatStreamEvent,
): void {
  if (event.kind === 'text') {
    progress.report(new vscode.LanguageModelTextPart(event.text));
  } else {
    progress.report(new vscode.LanguageModelToolCallPart(event.id, event.name, event.input));
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

function roleName(role: vscode.LanguageModelChatMessageRole): string {
  switch (role) {
    case vscode.LanguageModelChatMessageRole.User:
      return 'user';
    case vscode.LanguageModelChatMessageRole.Assistant:
      return 'assistant';
    default:
      return 'system';
  }
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
