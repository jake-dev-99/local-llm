import * as vscode from 'vscode';
import { readConfig } from '../config';
import type { InstalledModel, ModelRuntimeProfile } from '../domain';
import type { LocalLlmLogger } from '../logging';
import type { ModelRegistry } from '../models/modelRegistry';
import type { WorkerManager } from '../worker/workerManager';
import {
  matchingNativeToolSupport,
  nativeToolCapabilityFingerprint,
  TOOL_PROTOCOL_VERSION,
} from '../worker/nativeToolCapability';
import { isFatalWorkerError } from '../worker/workerError';
import { modelTokenLimits, resolveAdvertisedContextSize } from './modelCapacity';
import {
  adaptMessages,
  adaptTools,
  serializeMessageForTokenCount,
} from './messageAdapter';
import {
  evaluateLocalAgentTurn,
  isLocalAgentRequest,
  matchingPriorLocalAgentFinal,
  messagesForFreshLocalAgentRequest,
} from './localAgentToolChoice';
import { runLocalAgentResponse } from './localAgentResponse';
import {
  localAgentAvailableTools,
  localAgentDiscoveryTools,
  localAgentNeedsMutationTool,
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
    const localAgentState = evaluateLocalAgentTurn(
      adaptedMessages,
      config.maxAgentToolRounds,
    );
    const availableTools = localAgentRequest
      ? localAgentAvailableTools(tools ?? [], adaptedMessages)
      : tools ?? [];
    const forceLocalAgentFinal = localAgentState.phase === 'forceFinal';
    const needsMutationTool = localAgentRequest && !forceLocalAgentFinal &&
      localAgentNeedsMutationTool(adaptedMessages);
    // Never gate this on how many tools arrived. An empty tool list is exactly the
    // case that must reach resolveLocalAgentToolPolicy so it can refuse the turn.
    const requireLocalAgentTool = localAgentState.phase === 'requireEvidence';
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
    if (localAgentRequest) {
      this.logger.debug(
        `Local Agent work state: ${localAgentState.invocationCount}/${config.maxAgentToolRounds} invoked, ${localAgentState.evidenceCount} evidence result(s), ${localAgentState.resultCharacters} result characters, phase ${localAgentState.phase}.`,
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
          // In automatic mode localLlm.contextSize is zero, so clamping against it
          // would yield a two token window. The worker's loaded window is the truth.
          const physicalContext = config.contextSize > 0
            ? Math.max(2, Math.min(config.contextSize, profile.loadedContextSize))
            : Math.max(2, profile.loadedContextSize);
          const maxTokens = clamp(
            numericOption(options.modelOptions, 'maxTokens', config.maxOutputTokens),
            1,
            physicalContext - 1,
          );
          const freshMessages = localAgentRequest
            ? messagesForFreshLocalAgentRequest(adaptedMessages)
            : adaptedMessages;
          const controllerMessages = localAgentRequest
            ? [...freshMessages, {
              role: 'user' as const,
              content: localAgentControllerStatus(
                localAgentState,
                config.maxAgentToolRounds,
                physicalContext - maxTokens,
              ),
            }]
            : freshMessages;
          const modelMessages = messagesForSystemRoleSupport(
            controllerMessages,
            profile.supportsSystemRole,
          );
          const chatRequest = {
            messages: modelMessages,
            ...(toolPolicy.tools.length ? { tools: toolPolicy.tools } : {}),
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
          };
          let observedToolCall = false;
          try {
            if (localAgentRequest) {
              const priorFinal = matchingPriorLocalAgentFinal(adaptedMessages);
              const response = await runLocalAgentResponse({
                messages: adaptedMessages,
                request: chatRequest,
                ...(priorFinal ? { previousFinal: priorFinal.text } : {}),
                maxToolInvocations: config.maxAgentToolRounds,
                log: (message) => this.logger.debug(message),
                generate: async (request) => {
                  const events: import('../domain').ChatStreamEvent[] = [];
                  const result = await client.chat(request, (event) => events.push(event), signal);
                  return { events, result };
                },
              });
              observedToolCall = response.observedToolCall;
              for (const event of response.events) {
                reportChatEvent(progress, event);
              }
            } else {
              const result = await client.chat(
                chatRequest,
                (event) => reportChatEvent(progress, event),
                signal,
              );
              observedToolCall = result.toolCallCount > 0;
            }
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

function localAgentControllerStatus(
  state: ReturnType<typeof evaluateLocalAgentTurn>,
  maxToolInvocations: number,
  inputTokenBudget: number,
): string {
  const remaining = Math.max(0, maxToolInvocations - state.invocationCount);
  const nextStep = state.phase === 'forceFinal'
    ? 'The invocation ceiling is reached. Answer from collected evidence and name unfinished work or uncertainty.'
    : 'Answer when evidence is sufficient. Otherwise choose only a tool that materially improves the result.';
  return `Local Agent controller: ${state.invocationCount}/${maxToolInvocations} tool invocations used; ${remaining} remain; ${state.evidenceCount} successful evidence result(s); ${state.resultCharacters} result characters; ${inputTokenBudget} input-token budget. ${nextStep}`;
}

function reportChatEvent(
  progress: vscode.Progress<vscode.LanguageModelResponsePart>,
  event: import('../domain').ChatStreamEvent,
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
