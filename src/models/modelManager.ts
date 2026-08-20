
import { createHash, randomUUID } from 'node:crypto';
import { open, rm, stat } from 'node:fs/promises';
import * as path from 'node:path';
import * as vscode from 'vscode';
import { readConfig } from '../config';
import type { InstalledModel, ModelSource } from '../domain';
import type { LocalLlmLogger } from '../logging';
import { readGgufMetadata } from './ggufMetadata';
import { isSingleFileGguf, selectableHuggingFaceFiles } from './huggingFaceFileSelection';
import { ModelRegistry } from './modelRegistry';
import {
  copyLocalModel,
  downloadModel,
  huggingFaceDownloadUrl,
  inspectHuggingFaceRepository,
  safeModelFilename,
  sourceUrlForRegistry,
  type HuggingFaceFile,
} from './modelSources';

const HF_TOKEN_SECRET = 'localLlm.huggingFaceToken';

export class ModelManager {
  constructor(
    private readonly context: vscode.ExtensionContext,
    readonly registry: ModelRegistry,
    private readonly logger: LocalLlmLogger,
  ) {}

  async importLocal(uri: vscode.Uri): Promise<InstalledModel> {
    const sourcePath = uri.fsPath;
    const filename = safeModelFilename(sourcePath);
    await assertGguf(sourcePath);
    const destination = this.destination(filename);
    const result = await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: `Importing ${filename}`,
        cancellable: true,
      },
      (progress, token) => copyLocalModel(sourcePath, destination, progress, token),
    );
    try {
      await assertGguf(destination);
    } catch (error) {
      await rm(destination, { force: true });
      throw error;
    }
    return this.register({
      filename,
      filePath: destination,
      fileSize: result.size,
      sha256: result.sha256,
      source: 'import',
      sourceUrl: uri.toString(),
    });
  }

  async downloadFromHuggingFace(repository: string): Promise<InstalledModel | undefined> {
    const token = await this.context.secrets.get(HF_TOKEN_SECRET);
    const metadata = await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: `Inspecting ${repository}`,
        cancellable: true,
      },
      async (_progress, cancellationToken) => {
        const controller = abortOnCancellation(cancellationToken);
        try {
          return await inspectHuggingFaceRepository(repository, token, controller.signal);
        } finally {
          controller.dispose();
        }
      },
    );
    const selectableFiles = selectableHuggingFaceFiles(metadata.files);
    if (selectableFiles.length === 0) {
      throw new Error(
        `Repository ${metadata.id} contains GGUF files, but none are supported single-file downloads.`,
      );
    }
    const chosen = await chooseHuggingFaceFile(selectableFiles);
    if (!chosen) {
      return undefined;
    }
    if (!isSingleFileGguf(chosen.filename)) {
      throw new Error(
        'This PoC installs single-file GGUF models. Select a non-sharded GGUF from the repository.',
      );
    }
    const filename = safeModelFilename(chosen.filename);
    const url = huggingFaceDownloadUrl(metadata.id, metadata.revision, chosen.filename);
    const existing = this.registry.list().find((model) =>
      model.source === 'huggingface' &&
      model.repository === metadata.id &&
      model.revision === metadata.revision &&
      model.filename === filename,
    );
    if (existing) {
      return existing;
    }
    const destination = this.destination(
      filename,
      `huggingface:${metadata.id}:${metadata.revision}:${chosen.filename}`,
    );
    const result = await this.download(url, destination, chosen.sha256, token, filename);
    return this.register({
      filename,
      filePath: destination,
      fileSize: result.size,
      sha256: result.sha256,
      source: 'huggingface',
      sourceUrl: sourceUrlForRegistry(url),
      repository: metadata.id,
      revision: metadata.revision,
    });
  }

  async downloadFromUrl(url: string): Promise<InstalledModel> {
    const parsed = new URL(url);
    const filename = safeModelFilename(decodeURIComponent(parsed.pathname));
    const sourceUrl = sourceUrlForRegistry(url);
    const sourceIdentity = createHash('sha256').update(url).digest('hex');
    const existing = this.registry.list().find((model) =>
      model.source === 'url' &&
      model.sourceIdentity === sourceIdentity &&
      model.filename === filename,
    );
    if (existing) {
      return existing;
    }
    const destination = this.destination(filename, `url:${sourceIdentity}`);
    const result = await this.download(url, destination, undefined, undefined, filename);
    return this.register({
      filename,
      filePath: destination,
      fileSize: result.size,
      sha256: result.sha256,
      source: 'url',
      sourceIdentity,
      sourceUrl,
    });
  }

  async remove(model: InstalledModel): Promise<void> {
    await rm(model.filePath, { force: true });
    await this.registry.remove(model.id);
    this.logger.info(`Removed local model: ${model.name}.`);
  }

  async setHuggingFaceToken(token: string | undefined): Promise<void> {
    if (token) {
      await this.context.secrets.store(HF_TOKEN_SECRET, token);
    } else {
      await this.context.secrets.delete(HF_TOKEN_SECRET);
    }
  }

  private destination(filename: string, stableIdentity?: string): string {
    const prefix = stableIdentity
      ? createHash('sha256').update(stableIdentity).digest('hex').slice(0, 24)
      : randomUUID();
    return path.join(readConfig(this.context).modelDirectory, `${prefix}-${filename}`);
  }

  private async download(
    url: string,
    destination: string,
    expectedSha256: string | undefined,
    bearerToken: string | undefined,
    filename: string,
  ): Promise<{ sha256: string; size: number }> {
    const result = await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: `Downloading ${filename}`,
        cancellable: true,
      },
      (progress, token) =>
        downloadModel(url, destination, expectedSha256, bearerToken, progress, token),
    );
    try {
      await assertGguf(destination);
    } catch (error) {
      await rm(destination, { force: true });
      throw error;
    }
    return result;
  }

  private async register(input: {
    filename: string;
    filePath: string;
    fileSize: number;
    sha256: string;
    source: ModelSource;
    sourceIdentity?: string;
    sourceUrl?: string;
    repository?: string;
    revision?: string;
  }): Promise<InstalledModel> {
    const name = friendlyName(input.filename);
    const fileMetadata = await stat(input.filePath);
    const ggufMetadata = await readGgufMetadata(
      input.filePath,
      (warning) => this.logger.info(warning),
    );
    const model: InstalledModel = {
      id: `${slug(name)}-${input.sha256.slice(0, 12)}`,
      name,
      filePath: input.filePath,
      fileSize: input.fileSize,
      fileModifiedAt: fileMetadata.mtimeMs,
      sha256: input.sha256,
      source: input.source,
      filename: input.filename,
      installedAt: new Date().toISOString(),
      capabilities: { toolCalling: 'unverified', fillInMiddle: 'unverified' },
      ...(input.sourceIdentity ? { sourceIdentity: input.sourceIdentity } : {}),
      ...(input.sourceUrl ? { sourceUrl: input.sourceUrl } : {}),
      ...(input.repository ? { repository: input.repository } : {}),
      ...(input.revision ? { revision: input.revision } : {}),
      ...(ggufMetadata?.trainedContextLength
        ? { trainedContextLength: ggufMetadata.trainedContextLength }
        : {}),
    };
    const existing = this.registry.get(model.id);
    await this.registry.upsert(model);
    if (existing && existing.filePath !== model.filePath) {
      await rm(existing.filePath, { force: true });
    }
    this.logger.info(`Installed local model: ${model.name} (${model.fileSize} bytes).`);
    return model;
  }
}

async function assertGguf(filePath: string): Promise<void> {
  const handle = await open(filePath, 'r');
  try {
    const header = Buffer.alloc(8);
    const { bytesRead } = await handle.read(header, 0, header.length, 0);
    const version = bytesRead >= 8 ? header.readUInt32LE(4) : 0;
    if (
      bytesRead !== header.length ||
      header.subarray(0, 4).toString('ascii') !== 'GGUF' ||
      (version !== 2 && version !== 3)
    ) {
      throw new Error('The selected file is not a valid GGUF model.');
    }
  } finally {
    await handle.close();
  }
}

async function chooseHuggingFaceFile(
  files: HuggingFaceFile[],
): Promise<HuggingFaceFile | undefined> {
  const items: Array<vscode.QuickPickItem & { file: HuggingFaceFile }> = files.map((file) => ({
    label: path.basename(file.filename),
    ...(file.size > 0 ? { description: formatSize(file.size) } : {}),
    detail: file.filename,
    file,
  }));
  const selected = await vscode.window.showQuickPick(items, {
    title: 'Select a GGUF model file',
    placeHolder: 'Q4_K_M is usually a good balance of speed and quality',
    matchOnDescription: true,
    matchOnDetail: true,
  });
  return selected?.file;
}

function friendlyName(filename: string): string {
  return path.basename(filename, path.extname(filename)).replace(/[-_]+/g, ' ').trim();
}

function slug(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 64) || 'model';
}

function formatSize(bytes: number): string {
  return `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
}

function abortOnCancellation(token: vscode.CancellationToken): AbortController & { dispose(): void } {
  const controller = new AbortController() as AbortController & { dispose(): void };
  const subscription = token.onCancellationRequested(() => controller.abort());
  controller.dispose = () => subscription.dispose();
  return controller;
}
