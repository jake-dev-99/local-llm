import * as vscode from 'vscode';
import { readConfig } from '../config.js';
import type { LocalLlmLogger } from '../logging.js';
import type { ModelRegistry } from '../models/modelRegistry.js';
import { toAbortSignal } from '../provider/localLanguageModelProvider.js';
import type { WorkerManager } from '../worker/workerManager.js';

const MAX_PREFIX_CHARS = 24_000;
const MAX_SUFFIX_CHARS = 8_000;

export class LocalInlineCompletionProvider
implements vscode.InlineCompletionItemProvider, vscode.Disposable {
  private pendingController: AbortController | undefined;

  constructor(
    private readonly extensionContext: vscode.ExtensionContext,
    private readonly registry: ModelRegistry,
    private readonly worker: WorkerManager,
    private readonly logger: LocalLlmLogger,
  ) {}

  async provideInlineCompletionItems(
    document: vscode.TextDocument,
    position: vscode.Position,
    _context: vscode.InlineCompletionContext,
    token: vscode.CancellationToken,
  ): Promise<vscode.InlineCompletionItem[]> {
    const config = readConfig(this.extensionContext);
    if (!config.inlineEnabled || document.uri.scheme !== 'file' || token.isCancellationRequested) {
      return [];
    }
    this.pendingController?.abort();
    const requestController = new AbortController();
    this.pendingController = requestController;
    const cancellation = toAbortSignal(token);
    const signal = linkedAbortSignal(cancellation.signal, requestController.signal);
    const documentVersion = document.version;

    try {
      await cancellableDelay(config.inlineDebounceMilliseconds, signal);
      if (document.version !== documentVersion || signal.aborted) {
        return [];
      }
      const models = this.registry.list();
      const model = this.registry.get(config.defaultModelId) ?? models[0];
      if (
        !model ||
        model.capabilities.fillInMiddle !== 'supported' ||
        !this.worker.canRunInline(model.id)
      ) {
        return [];
      }

      const offset = document.offsetAt(position);
      const prefix = document.getText(new vscode.Range(
        document.positionAt(Math.max(0, offset - MAX_PREFIX_CHARS)),
        position,
      ));
      const suffix = document.getText(new vscode.Range(
        position,
        document.positionAt(offset + MAX_SUFFIX_CHARS),
      ));
      if (!prefix.trim()) {
        return [];
      }

      const completion = await this.worker.run(
        model,
        'inline',
        (client, scheduledSignal) => client.infill(
          {
            prefix,
            suffix,
            maxTokens: config.inlineMaxTokens,
            temperature: Math.min(config.temperature, 0.4),
            stop: [
              '<|endoftext|>',
              '<|eot_id|>',
              '<|end|>',
              ...(suffix ? [suffix.slice(0, 128)] : []),
            ],
          },
          scheduledSignal,
        ),
        signal,
      );
      const insertText = cleanCompletion(completion, suffix);
      if (document.version !== documentVersion || signal.aborted) {
        return [];
      }
      return insertText ? [new vscode.InlineCompletionItem(insertText, new vscode.Range(position, position))] : [];
    } catch (error) {
      if (!signal.aborted) {
        this.logger.debug(
          `Inline completion skipped: ${error instanceof Error ? error.message : String(error)}`,
        );
      }
      return [];
    } finally {
      cancellation.dispose();
      if (this.pendingController === requestController) {
        this.pendingController = undefined;
      }
    }
  }

  dispose(): void {
    this.pendingController?.abort();
    this.pendingController = undefined;
  }
}

async function cancellableDelay(milliseconds: number, signal: AbortSignal): Promise<void> {
  if (milliseconds <= 0 || signal.aborted) {
    return;
  }
  await new Promise<void>((resolve, reject) => {
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', cancel);
      resolve();
    }, milliseconds);
    const cancel = (): void => {
      clearTimeout(timer);
      const error = new Error('Inline completion cancelled.');
      error.name = 'AbortError';
      reject(error);
    };
    signal.addEventListener('abort', cancel, { once: true });
  });
}

function linkedAbortSignal(...signals: AbortSignal[]): AbortSignal {
  return AbortSignal.any(signals);
}

function cleanCompletion(value: string, suffix: string): string {
  let result = value.replace(/<\|[^|]+\|>/g, '');
  const suffixMarker = suffix.slice(0, 128);
  const suffixIndex = suffixMarker ? result.indexOf(suffixMarker) : -1;
  if (suffixIndex >= 0) {
    result = result.slice(0, suffixIndex);
  }
  const overlap = suffixOverlapLength(result, suffix);
  if (overlap >= 4) {
    result = result.slice(0, -overlap);
  }
  return result.replace(/[\t ]+$/gm, '').slice(0, 8_000);
}

function suffixOverlapLength(completion: string, suffix: string): number {
  const limit = Math.min(completion.length, suffix.length);
  for (let length = limit; length >= 1; length -= 1) {
    if (completion.endsWith(suffix.slice(0, length))) {
      return length;
    }
  }
  return 0;
}
