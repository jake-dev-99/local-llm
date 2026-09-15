import * as vscode from 'vscode';
import { readConfig, setDefaultModelId } from '../config';
import type {
  ChatStreamEvent,
  ChatTool,
  InstalledModel,
  ModelRuntimeProfile,
} from '../domain';
import type { LocalLlmLogger } from '../logging';
import type { ModelManager } from '../models/modelManager';
import { formatBytes } from '../models/modelSources';
import {
  benchmarkModel,
  isModelValidated,
  MODEL_BENCHMARK_SAMPLE_COUNT,
} from './modelBenchmark';
import { isFatalWorkerError } from '../worker/workerError';
import type { WorkerManager } from '../worker/workerManager';

interface CommandServices {
  context: vscode.ExtensionContext;
  models: ModelManager;
  worker: WorkerManager;
  logger: LocalLlmLogger;
}

export function registerModelCommands(services: CommandServices): vscode.Disposable[] {
  const command = (
    id: string,
    handler: (...args: unknown[]) => Promise<void>,
  ): vscode.Disposable =>
    vscode.commands.registerCommand(id, (...args: unknown[]) =>
      handler(...args).catch((error: unknown) => handleError(services.logger, error)),
    );

  return [
    command('localLlm.manageModels', () => manageModels(services)),
    command('localLlm.importModel', () => importModel(services)),
    command('localLlm.downloadHuggingFace', () => downloadHuggingFace(services)),
    command('localLlm.downloadUrl', () => downloadUrl(services)),
    command('localLlm.removeModel', () => removeModel(services)),
    command('localLlm.selectDefaultModel', () => selectDefaultModel(services)),
    command('localLlm.stopWorker', async () => {
      await services.worker.stop();
      void vscode.window.showInformationMessage('Local LLM worker stopped.');
    }),
    command('localLlm.showStatus', () => showStatus(services)),
    command('localLlm.setHuggingFaceToken', () => setHuggingFaceToken(services)),
    command('localLlm.validateModel', () => validateModel(services)),
    command('localLlm.benchmarkModel', () => benchmarkSelectedModel(services)),
    command('localLlm.configureStrictLocal', () => configureStrictLocal(services)),
    command('localLlm.checkPrivacyDefaults', () => checkPrivacyDefaults()),
  ];
}

async function manageModels(services: CommandServices): Promise<void> {
  const installed = services.models.registry.list();
  const choice = await vscode.window.showQuickPick(
    [
      {
        label: '$(cloud-download) Download from Hugging Face',
        description: 'Choose a GGUF file from a model repository',
        command: 'localLlm.downloadHuggingFace',
      },
      {
        label: '$(link-external) Download from HTTPS URL',
        description: 'Download a direct .gguf URL',
        command: 'localLlm.downloadUrl',
      },
      {
        label: '$(file-add) Import local GGUF',
        description: 'Copy an existing model into managed storage',
        command: 'localLlm.importModel',
      },
      ...(installed.length
        ? [
            {
              label: '$(star) Select default model',
              description: 'Choose the model used for inline completion',
              command: 'localLlm.selectDefaultModel',
            },
            {
              label: '$(trash) Remove a model',
              description: `${installed.length} model${installed.length === 1 ? '' : 's'} installed`,
              command: 'localLlm.removeModel',
            },
          ]
        : []),
      {
        label: '$(shield) Configure local privacy defaults',
        description: 'Prefer local Chat paths and disable known remote editor suggestions',
        command: 'localLlm.configureStrictLocal',
      },
      {
        label: '$(checklist) Check local privacy defaults',
        description: 'Verify the relevant global VS Code and Copilot settings',
        command: 'localLlm.checkPrivacyDefaults',
      },
      {
        label: '$(beaker) Validate model compatibility',
        description: 'Load a model and verify its structured tool-call path',
        command: 'localLlm.validateModel',
      },
      {
        label: '$(dashboard) Benchmark model',
        description: 'Measure min, average, and max output tok/s on this machine',
        command: 'localLlm.benchmarkModel',
      },
      {
        label: '$(key) Set Hugging Face token',
        description: 'Optional; required only for gated or private repositories',
        command: 'localLlm.setHuggingFaceToken',
      },
      {
        label: '$(pulse) Show runtime status',
        description: 'Inspect the active local worker',
        command: 'localLlm.showStatus',
      },
    ],
    {
      title: 'Local LLM Model Manager',
      placeHolder: installed.length
        ? `${installed.length} local model${installed.length === 1 ? '' : 's'} installed`
        : 'Install your first GGUF model',
    },
  );
  if (choice) {
    await vscode.commands.executeCommand(choice.command);
  }
}

async function importModel(services: CommandServices): Promise<void> {
  const selection = await vscode.window.showOpenDialog({
    title: 'Import GGUF Model',
    canSelectMany: false,
    canSelectFiles: true,
    canSelectFolders: false,
    filters: { 'GGUF models': ['gguf'] },
  });
  const uri = selection?.[0];
  if (!uri) {
    return;
  }
  const model = await services.models.importLocal(uri);
  await finishInstall(services, model);
}

async function downloadHuggingFace(services: CommandServices): Promise<void> {
  const repository = await vscode.window.showInputBox({
    title: 'Download GGUF from Hugging Face',
    prompt: 'Repository',
    placeHolder: 'owner/model-GGUF',
    ignoreFocusOut: true,
    validateInput: (value) =>
      /^[\w.-]+\/[\w.-]+$/.test(value.trim())
        ? undefined
        : 'Enter a repository as owner/name.',
  });
  if (!repository) {
    return;
  }
  const model = await services.models.downloadFromHuggingFace(repository);
  if (model) {
    await finishInstall(services, model);
  }
}

async function downloadUrl(services: CommandServices): Promise<void> {
  const url = await vscode.window.showInputBox({
    title: 'Download GGUF from HTTPS URL',
    prompt: 'Direct model URL',
    placeHolder: 'https://example.com/model.Q4_K_M.gguf',
    ignoreFocusOut: true,
    validateInput: validateModelUrl,
  });
  if (!url) {
    return;
  }
  const model = await services.models.downloadFromUrl(url);
  await finishInstall(services, model);
}

async function removeModel(services: CommandServices): Promise<void> {
  const selected = await chooseModel(services, 'Remove Local Model');
  if (!selected) {
    return;
  }
  const confirmation = await vscode.window.showWarningMessage(
    `Delete ${selected.name} and its ${formatBytes(selected.fileSize)} GGUF file?`,
    { modal: true },
    'Delete Model',
  );
  if (confirmation !== 'Delete Model') {
    return;
  }
  if (
    services.worker.state.kind !== 'stopped' &&
    'modelId' in services.worker.state &&
    services.worker.state.modelId === selected.id
  ) {
    await services.worker.stop();
  }
  await services.models.remove(selected);
  if (readConfig(services.context).defaultModelId === selected.id) {
    await setDefaultModelId(services.models.registry.list()[0]?.id ?? '');
  }
  void vscode.window.showInformationMessage(`Removed local model ${selected.name}.`);
}

async function selectDefaultModel(services: CommandServices): Promise<void> {
  const selected = await chooseModel(services, 'Select Default Local Model');
  if (!selected) {
    return;
  }
  await setDefaultModelId(selected.id);
  void vscode.window.showInformationMessage(`Default local model: ${selected.name}.`);
}

async function showStatus(services: CommandServices): Promise<void> {
  const state = services.worker.state;
  const installed = services.models.registry.list();
  let runtime: string;
  switch (state.kind) {
    case 'ready': {
      runtime = `Ready on loopback port ${state.port} with ${modelName(installed, state.modelId)}`;
      break;
    }
    case 'starting':
      runtime = `Loading ${modelName(installed, state.modelId)}`;
      break;
    case 'stopping':
      runtime = `Stopping ${modelName(installed, state.modelId)}`;
      break;
    case 'failed':
      runtime = `Failed: ${state.message}`;
      break;
    case 'stopped':
      runtime = 'Stopped; a worker starts automatically when a local model is used';
  }
  const action = await vscode.window.showInformationMessage(
    `Local LLM: ${runtime}. ${installed.length} model${installed.length === 1 ? '' : 's'} installed.`,
    'Show Logs',
    ...(state.kind === 'ready' ? ['Stop Worker'] : []),
  );
  if (action === 'Show Logs') {
    services.logger.show();
  } else if (action === 'Stop Worker') {
    await services.worker.stop();
  }
}

async function setHuggingFaceToken(services: CommandServices): Promise<void> {
  const token = await vscode.window.showInputBox({
    title: 'Set Hugging Face Token',
    prompt: 'Stored securely in VS Code SecretStorage. Leave blank to remove the saved token.',
    password: true,
    ignoreFocusOut: true,
  });
  if (token === undefined) {
    return;
  }
  await services.models.setHuggingFaceToken(token.trim() || undefined);
  void vscode.window.showInformationMessage(
    token.trim() ? 'Hugging Face token saved securely.' : 'Hugging Face token removed.',
  );
}

async function validateModel(
  services: CommandServices,
  selectedModel?: InstalledModel,
): Promise<void> {
  const model = selectedModel ?? await chooseModel(services, 'Validate Local Model');
  if (!model) {
    return;
  }
  const result = await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: `Validating ${model.name}`,
      cancellable: true,
    },
    async (_progress, token) => {
      const cancellation = abortOnCancellation(token);
      try {
        return await services.worker.run(
          model,
          'utility',
          async (client, signal) => {
            const profile = await client.getModelProfile(signal);
            const storedProfile: ModelRuntimeProfile = {
              ...profile,
              validatedAt: new Date().toISOString(),
            };
            await services.models.registry.updateRuntimeProfile(model.id, storedProfile);
            const fillInMiddle = await validateFillInMiddle(
              services,
              model,
              client,
              signal,
            );
            if (!profile.hasChatTemplate) {
              throw new Error(
                'The GGUF loaded, but it does not expose a llama.cpp chat template and cannot be used in VS Code Chat.',
              );
            }
            if (!profile.supportsTools || !profile.supportsToolCalls) {
              await services.models.registry.markToolCalling(model.id, 'unsupported');
              return {
                profile,
                toolCalls: 0,
                continuationCharacters: 0,
                agentValidated: false,
                fillInMiddle,
              };
            }
            const probeTool: ChatTool = {
              type: 'function',
              function: {
                name: 'local_llm_probe',
                description: 'Returns the supplied validation value.',
                parameters: {
                  type: 'object',
                  properties: { value: { type: 'string' } },
                  required: ['value'],
                  additionalProperties: false,
                },
              },
            };
            const probeCalls: Array<Extract<ChatStreamEvent, { kind: 'toolCall' }>> = [];
            let probe;
            probe = await client.chat(
              {
                messages: [
                  {
                    role: 'system',
                    content: 'Call the supplied function exactly once. Do not answer with prose.',
                  },
                  { role: 'user', content: 'Call local_llm_probe with value set to ok.' },
                ],
                tools: [probeTool],
                toolChoice: 'required',
                inputTokenBudget: Math.max(1, profile.loadedContextSize - 128),
                maxTokens: Math.min(128, Math.max(1, profile.loadedContextSize - 1)),
                temperature: 0,
              },
              (event) => {
                if (event.kind === 'toolCall') {
                  probeCalls.push(event);
                }
              },
              signal,
            );
            const probeCall = probeCalls[0];
            if (
              probe.toolCallCount !== 1 ||
              probeCalls.length !== 1 ||
              !probeCall ||
              (probeCall.input as Record<string, unknown>).value !== 'ok'
            ) {
              throw new Error(
                'The model did not produce exactly one valid local_llm_probe call with value set to ok.',
              );
            }

            let continuationText = '';
            const continuation = await client.chat(
              {
                messages: [
                  {
                    role: 'system',
                    content: 'Call the supplied function when needed. After receiving its result, answer with that result and do not call again.',
                  },
                  { role: 'user', content: 'Get the validation value.' },
                  {
                    role: 'assistant',
                    content: '',
                    tool_calls: [
                      {
                        id: probeCall.id,
                        type: 'function',
                        function: {
                          name: probeCall.name,
                          arguments: JSON.stringify(probeCall.input),
                        },
                      },
                    ],
                  },
                  {
                    role: 'tool',
                    content: 'ok',
                    tool_call_id: probeCall.id,
                  },
                ],
                tools: [probeTool],
                toolChoice: 'auto',
                inputTokenBudget: Math.max(1, profile.loadedContextSize - 128),
                maxTokens: Math.min(128, Math.max(1, profile.loadedContextSize - 1)),
                temperature: 0,
              },
              (event) => {
                if (event.kind === 'text') {
                  continuationText += event.text;
                }
              },
              signal,
            );
            if (
              continuation.toolCallCount !== 0 ||
              !isValidProbeContinuation(continuationText)
            ) {
              throw new Error(
                'The model produced a tool call but could not continue with a final text response after the tool result.',
              );
            }
            await services.models.registry.markToolCalling(model.id, 'supported');
            return {
              profile,
              toolCalls: probe.toolCallCount,
              continuationCharacters: continuationText.length,
              agentValidated: true,
              fillInMiddle,
            };
          },
          cancellation.signal,
        );
      } finally {
        cancellation.dispose();
      }
    },
  );
  const inlineStatus = result.fillInMiddle ? 'inline FIM available' : 'inline FIM unavailable';
  if (result.agentValidated) {
    void vscode.window.showInformationMessage(
      `${model.name} passed local runtime validation: ${result.profile.loadedContextSize} token context, ${result.toolCalls} structured tool call, tool-result continuation, and ${inlineStatus}.`,
    );
  } else {
    void vscode.window.showWarningMessage(
      `${model.name} passed local Chat loading validation (${result.profile.loadedContextSize} token context; ${inlineStatus}), but its template does not support Agent tool calls.`,
    );
  }
}

async function benchmarkSelectedModel(services: CommandServices): Promise<void> {
  const model = await chooseModel(
    services,
    'Benchmark Local Model',
    services.models.registry.list().filter(isModelValidated),
    'No validated local models are available. Run Local LLM: Validate Model Compatibility first.',
  );
  if (!model) {
    return;
  }
  const result = await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: `Benchmarking ${model.name}`,
      cancellable: true,
    },
    async (progress, token) => {
      const cancellation = abortOnCancellation(token);
      try {
        return await services.worker.run(
          model,
          'utility',
          (client, signal) => benchmarkModel(client, signal, (completed) => {
            progress.report({
              increment: 100 / MODEL_BENCHMARK_SAMPLE_COUNT,
              message: `Sample ${completed} of ${MODEL_BENCHMARK_SAMPLE_COUNT}`,
            });
          }),
          cancellation.signal,
        );
      } finally {
        cancellation.dispose();
      }
    },
  );
  void vscode.window.showInformationMessage(
    `${model.name}: min ${formatTokenRate(result.minTokensPerSecond)}, ` +
    `avg ${formatTokenRate(result.averageTokensPerSecond)}, ` +
    `max ${formatTokenRate(result.maxTokensPerSecond)}.`,
  );
}

function formatTokenRate(rate: number): string {
  return `${rate.toFixed(2)} tok/s`;
}

function isValidProbeContinuation(value: string): boolean {
  const normalized = value.trim().toLowerCase();
  return /\bok\b/.test(normalized) &&
    !/local_llm_probe|["']?arguments["']?|<\/?tools?>|<\/?tool_call>/.test(normalized);
}

async function validateFillInMiddle(
  services: CommandServices,
  model: InstalledModel,
  client: import('../worker/llamaClient').LlamaClient,
  signal: AbortSignal,
): Promise<boolean> {
  try {
    const completion = await client.infill(
      {
        prefix: 'function add(a, b) {\n  return ',
        suffix: ';\n}\n',
        maxTokens: 16,
        temperature: 0,
      },
      signal,
    );
    const supported = completion.trim().length > 0;
    await services.models.registry.markFillInMiddle(
      model.id,
      supported ? 'supported' : 'unsupported',
    );
    return supported;
  } catch (error) {
    if (signal.aborted || (error instanceof Error && error.name === 'AbortError')) {
      throw error;
    }
    if (isFatalWorkerError(error)) {
      throw error;
    }
    await services.models.registry.markFillInMiddle(model.id, 'unsupported');
    services.logger.debug(
      `FIM validation unavailable for ${model.name}: ${error instanceof Error ? error.message : String(error)}`,
    );
    return false;
  }
}

async function configureStrictLocal(services: CommandServices): Promise<void> {
  const selected = await chooseModel(services, 'Configure Local Privacy Defaults');
  if (!selected) {
    return;
  }
  const confirmation = await vscode.window.showWarningMessage(
    `Use ${selected.name} for VS Code Chat utility requests, disable Copilot remote semantic search, and disable Copilot completions and next-edit suggestions? This updates global VS Code settings.`,
    { modal: true },
    'Configure',
  );
  if (confirmation !== 'Configure') {
    return;
  }
  await Promise.all([applyStrictLocalSettings(selected), setDefaultModelId(selected.id)]);
  void vscode.window.showInformationMessage(
    `Local privacy defaults configured for ${selected.name}. Before sending source, select Local Agent and this model in Chat.`,
  );
}

async function checkPrivacyDefaults(): Promise<void> {
  const chat = vscode.workspace.getConfiguration('chat');
  const copilot = vscode.workspace.getConfiguration('github.copilot');
  const semanticSearch = vscode.workspace.getConfiguration('github.copilot.chat');
  const nextEdits = vscode.workspace.getConfiguration('github.copilot.nextEditSuggestions');
  const utilityModel = chat.get<string>('utilityModel', '');
  const utilitySmallModel = chat.get<string>('utilitySmallModel', '');
  const completions = copilot.get<Record<string, boolean>>('enable', {});
  const failures: string[] = [];
  if (!utilityModel.startsWith('local-llm-engine/')) {
    failures.push('chat.utilityModel is not local');
  }
  if (!utilitySmallModel.startsWith('local-llm-engine/')) {
    failures.push('chat.utilitySmallModel is not local');
  }
  if (chat.get<string>('byokUtilityModelDefault', 'copilot') !== 'none') {
    failures.push('the BYOK utility fallback is enabled');
  }
  if (semanticSearch.get<string>('semanticSearchTool.mode', 'enabled') !== 'disabled') {
    failures.push('remote semantic search is enabled');
  }
  if (completions['*'] !== false || Object.values(completions).some(Boolean)) {
    failures.push('Copilot completions are enabled for at least one language');
  }
  if (nextEdits.get<boolean>('enabled', true)) {
    failures.push('Copilot next-edit suggestions are enabled');
  }
  if (failures.length) {
    void vscode.window.showWarningMessage(
      `Local privacy defaults are incomplete: ${failures.join('; ')}. Run Local LLM: Configure Local Privacy Defaults.`,
    );
    return;
  }
  void vscode.window.showInformationMessage(
    'Local privacy defaults pass. VS Code cannot pin the stock Chat model through this extension, so verify Local Agent and a Local LLM model are selected before sending source.',
  );
}

async function finishInstall(
  services: CommandServices,
  model: InstalledModel,
): Promise<void> {
  if (!readConfig(services.context).defaultModelId) {
    await setDefaultModelId(model.id);
  }
  const selection = await vscode.window.showInformationMessage(
    `Installed ${model.name} (${formatBytes(model.fileSize)}). It is now available in the Chat model picker.`,
    'Validate Model',
    'Configure Privacy Defaults',
    'Select as Default',
  );
  if (selection === 'Validate Model') {
    await validateModel(services, model);
  } else if (selection === 'Configure Privacy Defaults') {
    await setDefaultModelId(model.id);
    await configureStrictLocalForModel(model);
  } else if (selection === 'Select as Default') {
    await setDefaultModelId(model.id);
  }
}

async function configureStrictLocalForModel(
  model: InstalledModel,
): Promise<void> {
  await applyStrictLocalSettings(model);
  void vscode.window.showInformationMessage(
    `Local privacy defaults configured for ${model.name}. Select Local Agent and this model in Chat.`,
  );
}

async function applyStrictLocalSettings(model: InstalledModel): Promise<void> {
  const modelReference = `local-llm-engine/${model.id}`;
  await Promise.all([
    vscode.workspace.getConfiguration('chat').update(
      'utilityModel', modelReference, vscode.ConfigurationTarget.Global,
    ),
    vscode.workspace.getConfiguration('chat').update(
      'utilitySmallModel', modelReference, vscode.ConfigurationTarget.Global,
    ),
    vscode.workspace.getConfiguration('chat').update(
      'byokUtilityModelDefault', 'none', vscode.ConfigurationTarget.Global,
    ),
    vscode.workspace.getConfiguration('github.copilot.chat').update(
      'semanticSearchTool.mode', 'disabled', vscode.ConfigurationTarget.Global,
    ),
    vscode.workspace.getConfiguration('github.copilot').update(
      'enable', { '*': false }, vscode.ConfigurationTarget.Global,
    ),
    vscode.workspace.getConfiguration('github.copilot.nextEditSuggestions').update(
      'enabled', false, vscode.ConfigurationTarget.Global,
    ),
  ]);
}

async function chooseModel(
  services: CommandServices,
  title: string,
  models: readonly InstalledModel[] = services.models.registry.list(),
  emptyMessage = 'No local GGUF models are installed.',
): Promise<InstalledModel | undefined> {
  const config = readConfig(services.context);
  const items = models.map((model) => ({
    label: model.name,
    description: `${formatBytes(model.fileSize)} · ${model.source}`,
    detail: `${model.filename}${model.id === config.defaultModelId ? ' · default' : ''}`,
    model,
  }));
  if (!items.length) {
    void vscode.window.showInformationMessage(emptyMessage);
    return undefined;
  }
  return (await vscode.window.showQuickPick(items, { title, matchOnDetail: true }))?.model;
}

function validateModelUrl(value: string): string | undefined {
  try {
    const url = new URL(value);
    if (url.protocol !== 'https:') {
      return 'The URL must use HTTPS.';
    }
    if (!decodeURIComponent(url.pathname).toLowerCase().endsWith('.gguf')) {
      return 'The URL path must end with .gguf.';
    }
    return undefined;
  } catch {
    return 'Enter a valid HTTPS URL.';
  }
}

function modelName(models: readonly InstalledModel[], id: string): string {
  return models.find((model) => model.id === id)?.name ?? id;
}

function handleError(logger: LocalLlmLogger, error: unknown): void {
  if (error instanceof vscode.CancellationError) {
    return;
  }
  const message = error instanceof Error ? error.message : String(error);
  logger.error('Command failed', error);
  void vscode.window.showErrorMessage(`Local LLM: ${message}`, 'Show Logs').then((action) => {
    if (action === 'Show Logs') {
      logger.show();
    }
  });
}

function abortOnCancellation(
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
