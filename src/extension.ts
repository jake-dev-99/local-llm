import * as vscode from 'vscode';
import { LocalInlineCompletionProvider } from './completion/localInlineCompletionProvider.js';
import { readConfig } from './config.js';
import { LocalLlmLogger } from './logging.js';
import { ModelManager } from './models/modelManager.js';
import { ModelRegistry } from './models/modelRegistry.js';
import { LocalLanguageModelProvider } from './provider/localLanguageModelProvider.js';
import { registerModelCommands } from './ui/modelCommands.js';
import { runtimeStatusPresentation } from './ui/runtimeStatus.js';
import { WorkerManager } from './worker/workerManager.js';

let activeWorker: WorkerManager | undefined;

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  const logger = new LocalLlmLogger(readConfig(context).logLevel);
  const registry = new ModelRegistry(context.globalState, logger);
  await registry.initialize();
  const worker = new WorkerManager(context, logger);
  activeWorker = worker;
  const models = new ModelManager(context, registry, logger);
  const languageModels = new LocalLanguageModelProvider(context, registry, worker, logger);
  const inlineCompletions = new LocalInlineCompletionProvider(context, registry, worker, logger);

  const status = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 20);
  status.command = 'localLlm.manageModels';
  status.name = 'Local LLM';
  updateStatus(status, worker.state, registry.list().length);
  status.show();

  context.subscriptions.push(
    logger,
    registry,
    worker,
    languageModels,
    inlineCompletions,
    status,
    vscode.lm.registerLanguageModelChatProvider('local-llm-engine', languageModels),
    vscode.languages.registerInlineCompletionItemProvider({ scheme: 'file' }, inlineCompletions),
    ...registerModelCommands({ context, models, worker, logger }),
    registry.onDidChange(() => updateStatus(status, worker.state, registry.list().length)),
    worker.onDidChangeState((state) => updateStatus(status, state, registry.list().length)),
    vscode.workspace.onDidChangeConfiguration((event) => {
      if (!event.affectsConfiguration('localLlm')) {
        return;
      }
      logger.setLevel(readConfig(context).logLevel);
      if (
        event.affectsConfiguration('localLlm.contextSize') ||
        event.affectsConfiguration('localLlm.maxOutputTokens') ||
        event.affectsConfiguration('localLlm.maxTools')
      ) {
        languageModels.refresh();
      }
      // Worker launch settings (context, batch sizes, acceleration, threads,
      // Python runtime) are compared against the running worker at the next
      // request, which reloads it then. Stopping here would kill a response
      // in flight on every keystroke in the settings editor.
    }),
  );

  logger.info(
    `Local LLM Engine activated on ${process.platform}-${process.arch}; ${registry.list().length} model(s) installed.`,
  );
  if (!isSupportedPlatform()) {
    void vscode.window.showWarningMessage(
      `Local LLM Engine does not include a worker for ${process.platform}-${process.arch}. Supported platforms are Apple Silicon macOS and x64 Windows.`,
    );
  }
}

export async function deactivate(): Promise<void> {
  const worker = activeWorker;
  activeWorker = undefined;
  await worker?.stop();
}

function isSupportedPlatform(): boolean {
  return (
    (process.platform === 'darwin' && process.arch === 'arm64') ||
    (process.platform === 'win32' && process.arch === 'x64')
  );
}

function updateStatus(
  item: vscode.StatusBarItem,
  state: import('./domain.js').WorkerState,
  modelCount: number,
): void {
  const presentation = runtimeStatusPresentation(state, modelCount);
  item.text = presentation.text;
  item.tooltip = presentation.tooltip;
  item.backgroundColor = presentation.error
    ? new vscode.ThemeColor('statusBarItem.errorBackground')
    : undefined;
}
