
import { createHash, randomUUID } from 'node:crypto';
import { copyFile, mkdir, open, readFile, rm, stat, writeFile } from 'node:fs/promises';
import * as path from 'node:path';
import * as vscode from 'vscode';
import { readConfig } from '../config.js';
import type { InstalledModel, ModelSource } from '../domain.js';
import type { LocalLlmLogger } from '../logging.js';
import { readGgufMetadata } from './ggufMetadata.js';
import {
  isSingleFileGguf,
  safetensorsHuggingFaceFiles,
  selectableHuggingFaceFiles,
} from './huggingFaceFileSelection.js';
import { isExtensionOwned } from './modelIdentity.ts';
import {
  isSafetensorsDirectory,
  readSafetensorsCheckpoint,
  type SafetensorsCheckpoint,
} from './safetensorsDirectory.ts';
import { ModelRegistry } from './modelRegistry.js';
import {
  copyLocalModel,
  downloadModel,
  huggingFaceDownloadUrl,
  inspectHuggingFaceRepository,
  safeModelFilename,
  sha256File,
  sourceUrlForRegistry,
  type HuggingFaceFile,
} from './modelSources.js';

const HF_TOKEN_SECRET = 'localLlm.huggingFaceToken';

/**
 * Files that turn bare weights into a loadable checkpoint, in fetch order.
 * config.json is required; the rest are best-effort.
 */
const SIDECAR_FILES = [
  'config.json',
  'tokenizer.json',
  'tokenizer_config.json',
  // Recent checkpoints keep the template in a standalone jinja file rather
  // than inside tokenizer_config.json; without it the loaded tokenizer
  // reports no chat template and validation refuses the model.
  'chat_template.jinja',
  'generation_config.json',
];

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
    const artifact = await chooseHuggingFaceArtifact(
      selectableHuggingFaceFiles(metadata.files),
      safetensorsHuggingFaceFiles(metadata.files),
    );
    if (!artifact) {
      if (metadata.files.some((file) => file.filename.toLowerCase().endsWith('.safetensors'))) {
        throw new Error(
          `Repository ${metadata.id} has Safetensors weights, but no complete root checkpoint with config.json.`,
        );
      }
      if (metadata.files.some((file) => file.filename.toLowerCase().endsWith('.gguf'))) {
        throw new Error(
          `Repository ${metadata.id} contains GGUF files, but none are supported single-file downloads.`,
        );
      }
      throw new Error(`Repository ${metadata.id} contains no supported GGUF or Safetensors model.`);
    }
    if (artifact.kind === 'safetensors') {
      return this.downloadSafetensorsRepository(metadata, artifact.files, token);
    }
    const chosen = artifact.file;
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

  private async downloadSafetensorsRepository(
    metadata: { id: string; revision: string },
    files: HuggingFaceFile[],
    token: string | undefined,
  ): Promise<InstalledModel> {
    const existing = this.registry.list().find((model) =>
      model.source === 'huggingface' &&
      model.format === 'safetensors' &&
      model.repository === metadata.id &&
      model.revision === metadata.revision,
    );
    if (existing) {
      return existing;
    }
    const repositoryName = safeDirectoryName(path.basename(metadata.id));
    const directory = this.destination(
      repositoryName,
      `huggingface:${metadata.id}:${metadata.revision}:safetensors`,
    );
    await mkdir(directory, { recursive: true });
    for (const [index, file] of files.entries()) {
      const destination = path.join(directory, file.filename);
      if (await completedRepositoryFileMatches(destination, file)) {
        continue;
      }
      await rm(destination, { force: true });
      const url = huggingFaceDownloadUrl(metadata.id, metadata.revision, file.filename);
      await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: `Downloading ${file.filename} (${index + 1}/${files.length})`,
          cancellable: true,
        },
        (progress, cancellation) => downloadModel(
          url,
          destination,
          file.sha256,
          token,
          progress,
          cancellation,
        ),
      );
    }
    if (!(await isSafetensorsDirectory(directory))) {
      throw new Error(
        `Downloaded repository ${metadata.id} is not a complete Safetensors checkpoint.`,
      );
    }
    const checkpoint = await readSafetensorsCheckpoint(
      directory,
      (warning) => this.logger.info(warning),
    );
    return this.registerCheckpointDirectory(directory, checkpoint, {
      managed: true,
      source: 'huggingface',
      sourceUrl: sourceUrlForRegistry(
        `https://huggingface.co/${metadata.id}/tree/${metadata.revision}`,
      ),
      repository: metadata.id,
      revision: metadata.revision,
      filename: repositoryName,
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

  /**
   * Registers a Safetensors checkpoint where it already lives.
   *
   * Unlike a GGUF import this copies nothing. These checkpoints run to tens of
   * gigabytes and the user already has them on disk, so duplicating them into
   * extension storage would waste the space twice over. The consequence is
   * recorded on the model: the extension does not own these bytes and never
   * deletes them.
   */
  async importSafetensorsDirectory(uri: vscode.Uri): Promise<InstalledModel> {
    const directory = uri.fsPath;
    if (!(await isSafetensorsDirectory(directory))) {
      throw new Error(
        'That folder is not a Safetensors checkpoint. It needs a config.json and at least one .safetensors file.',
      );
    }
    const checkpoint = await readSafetensorsCheckpoint(
      directory,
      (warning) => this.logger.info(warning),
    );
    return this.registerCheckpointDirectory(directory, checkpoint, {
      managed: false,
      source: 'import',
      sourceUrl: uri.toString(),
    });
  }

  /**
   * Stages a single `.safetensors` file as a managed checkpoint directory.
   *
   * A bare weights file cannot load: Transformers needs its config and
   * tokenizer. Siblings next to the file win; otherwise they are fetched
   * from the given Hugging Face repository (config required, tokenizer
   * best-effort). Nothing is registered — the caller inspects the returned
   * checkpoint (warnings, consent) and registers explicitly.
   */
  async stageSafetensorsFile(
    fileUri: vscode.Uri,
    options: { repository?: string; configFile?: string; configJson?: string } = {},
  ): Promise<{ directory: string; checkpoint: SafetensorsCheckpoint }> {
    const sourcePath = fileUri.fsPath;
    if (!sourcePath.toLowerCase().endsWith('.safetensors')) {
      throw new Error('Select a .safetensors weights file.');
    }
    const filename = safeModelFilename(path.basename(sourcePath), ['.safetensors']);
    const stem = filename.replace(/\.safetensors$/i, '');
    const directory = path.join(readConfig(this.context).modelDirectory, `${randomUUID()}-${stem}`);
    await mkdir(directory, { recursive: true });
    try {
      await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: `Importing ${filename}`,
          cancellable: true,
        },
        (progress, token) => copyLocalModel(sourcePath, path.join(directory, filename), progress, token),
      );
      const sourceDir = path.dirname(sourcePath);
      for (const sidecar of SIDECAR_FILES) {
        try {
          await stat(path.join(sourceDir, sidecar));
          await copyFile(path.join(sourceDir, sidecar), path.join(directory, sidecar));
        } catch {
          // Absent siblings are normal for a lone download; HF or error below.
        }
      }
      try {
        await stat(path.join(directory, 'config.json'));
      } catch {
        await this.supplyStagedConfig(directory, options);
      }
      if (!(await isSafetensorsDirectory(directory))) {
        throw new Error(
          'That file cannot become a checkpoint here: config.json is missing and ' +
          'no Hugging Face repository was given to fetch it from.',
        );
      }
      const checkpoint = await readSafetensorsCheckpoint(
        directory,
        (warning) => this.logger.info(warning),
      );
      return { directory, checkpoint };
    } catch (error) {
      await rm(directory, { recursive: true, force: true });
      throw error;
    }
  }

  /**
   * Registers a staged directory as extension-owned. Staged bytes were put
   * on disk by `stageSafetensorsFile`, so unlike an in-place import the
   * extension owns and may delete them.
   */
  async registerStagedSafetensorsDirectory(
    directory: string,
    source: ModelSource = 'import',
  ): Promise<InstalledModel> {
    const checkpoint = await readSafetensorsCheckpoint(
      directory,
      (warning) => this.logger.info(warning),
    );
    return this.registerCheckpointDirectory(directory, checkpoint, {
      managed: true,
      source,
    });
  }

  private async registerCheckpointDirectory(
    directory: string,
    checkpoint: SafetensorsCheckpoint,
    options: {
      managed: boolean;
      source: ModelSource;
      sourceUrl?: string;
      repository?: string;
      revision?: string;
      filename?: string;
    },
  ): Promise<InstalledModel> {
    const filename = options.filename ?? path.basename(directory);
    const name = friendlyDirectoryName(filename);
    const model: InstalledModel = {
      id: `${slug(name)}-${checkpoint.identity.digest.slice(0, 12)}`,
      name,
      filePath: directory,
      fileSize: checkpoint.identity.totalBytes,
      sha256: checkpoint.identity.digest,
      source: options.source,
      ...(options.sourceUrl ? { sourceUrl: options.sourceUrl } : {}),
      ...(options.repository ? { repository: options.repository } : {}),
      ...(options.revision ? { revision: options.revision } : {}),
      filename,
      installedAt: new Date().toISOString(),
      format: 'safetensors',
      runtime: 'transformers',
      managed: options.managed,
      files: checkpoint.identity.files,
      capabilities: { toolCalling: 'unverified', fillInMiddle: 'unsupported' },
      ...(checkpoint.quantization ? { quantization: checkpoint.quantization } : {}),
      ...(checkpoint.customCodeRequired ? { customCodeRequired: true } : {}),
      ...(checkpoint.trainedContextLength
        ? { trainedContextLength: checkpoint.trainedContextLength }
        : {}),
    };
    await this.registry.upsert(model);
    this.logger.info(
      `Registered Safetensors checkpoint: ${model.name} ` +
      `(${checkpoint.identity.files.length} files, ${model.fileSize} bytes` +
      `${options.managed ? '' : ', in place'}).`,
    );
    return model;
  }

  /**
   * Supplies a missing config.json for a staged file, first source wins:
   * an explicit file, pasted JSON, then the Hugging Face repository.
   */
  private async supplyStagedConfig(
    directory: string,
    options: { repository?: string; configFile?: string; configJson?: string },
  ): Promise<void> {
    if (options.configFile) {
      if (!options.configFile.toLowerCase().endsWith('.json')) {
        throw new Error('The selected config must be a .json file.');
      }
      await copyFile(options.configFile, path.join(directory, 'config.json'));
      assertValidConfigJson(directory);
      return;
    }
    if (options.configJson) {
      await writeStagedConfig(directory, options.configJson);
      return;
    }
    await this.fetchCheckpointSidecars(directory, options.repository);
  }

  /** Fetches a checkpoint's sidecars; config.json is required, the rest best-effort. */
  private async fetchCheckpointSidecars(directory: string, repository: string | undefined): Promise<void> {
    if (!repository) {
      return;
    }
    const token = await this.context.secrets.get(HF_TOKEN_SECRET);
    const revision = 'main';
    for (const sidecar of SIDECAR_FILES) {
      const url = huggingFaceDownloadUrl(repository, revision, sidecar);
      try {
        await vscode.window.withProgress(
          {
            location: vscode.ProgressLocation.Notification,
            title: `Downloading ${sidecar}`,
            cancellable: true,
          },
          (progress, cancellation) => downloadModel(
            url, path.join(directory, sidecar), undefined, token, progress, cancellation,
          ),
        );
      } catch (error) {
        if (sidecar === 'config.json') {
          throw new Error(
            `Could not fetch config.json from ${repository}: ` +
            `${error instanceof Error ? error.message : String(error)}`,
          );
        }
        this.logger.info(`Optional ${sidecar} not fetched from ${repository}; continuing.`);
      }
    }
  }

  async remove(model: InstalledModel): Promise<void> {
    // Only bytes the extension put on disk are the extension's to delete. A
    // checkpoint registered in place is unregistered and left alone.
    if (isExtensionOwned(model)) {
      await rm(model.filePath, {
        force: true,
        recursive: model.format === 'safetensors',
      });
      this.logger.info(`Removed local model: ${model.name}.`);
    } else {
      this.logger.info(
        `Unregistered ${model.name}. Its files were left at ${model.filePath}.`,
      );
    }
    await this.registry.remove(model.id);
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
      format: 'gguf',
      runtime: 'llama-cpp',
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

type HuggingFaceArtifact =
  | { kind: 'gguf'; file: HuggingFaceFile }
  | { kind: 'safetensors'; files: HuggingFaceFile[] };

async function chooseHuggingFaceArtifact(
  ggufFiles: HuggingFaceFile[],
  safetensorsFiles: HuggingFaceFile[],
): Promise<HuggingFaceArtifact | undefined> {
  if (safetensorsFiles.length === 0 && ggufFiles.length === 0) {
    return undefined;
  }
  if (safetensorsFiles.length === 0) {
    const file = await chooseHuggingFaceFile(ggufFiles);
    return file ? { kind: 'gguf', file } : undefined;
  }
  if (ggufFiles.length === 0) {
    return { kind: 'safetensors', files: safetensorsFiles };
  }
  const safetensorsBytes = safetensorsFiles.reduce((total, file) => total + file.size, 0);
  const items: Array<vscode.QuickPickItem & { artifact: HuggingFaceArtifact }> = [
    {
      label: 'Safetensors checkpoint',
      description: `${formatSize(safetensorsBytes)} · ${safetensorsFiles.length} files`,
      detail: 'All root checkpoint shards and sidecars',
      artifact: { kind: 'safetensors', files: safetensorsFiles },
    },
    ...ggufFiles.map((file) => ({
      label: path.basename(file.filename),
      ...(file.size > 0 ? { description: formatSize(file.size) } : {}),
      detail: file.filename,
      artifact: { kind: 'gguf', file } as HuggingFaceArtifact,
    })),
  ];
  const selected = await vscode.window.showQuickPick(items, {
    title: 'Select a model download',
    placeHolder: 'Choose the checkpoint format or GGUF quantization',
    matchOnDescription: true,
    matchOnDetail: true,
  });
  return selected?.artifact;
}

async function completedRepositoryFileMatches(
  destination: string,
  file: HuggingFaceFile,
): Promise<boolean> {
  try {
    const metadata = await stat(destination);
    if (!metadata.isFile() || file.size <= 0 || metadata.size !== file.size) {
      return false;
    }
    return file.sha256
      ? (await sha256File(destination)).toLowerCase() === file.sha256.toLowerCase()
      : true;
  } catch {
    return false;
  }
}

function safeDirectoryName(value: string): string {
  return value.replace(/[^a-zA-Z0-9._-]+/g, '-').replace(/^-+|-+$/g, '') || 'model';
}

function friendlyName(filename: string): string {
  return path.basename(filename, path.extname(filename)).replace(/[-_]+/g, ' ').trim();
}

function friendlyDirectoryName(filename: string): string {
  return path.basename(filename).replace(/[-_]+/g, ' ').trim();
}

function slug(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 64) || 'model';
}

/** A staged config must at least be a JSON object; Transformers judges the rest at load. */
async function assertValidConfigJson(directory: string): Promise<void> {
  let parsed: unknown;
  try {
    parsed = JSON.parse(await readFile(path.join(directory, 'config.json'), 'utf8'));
  } catch {
    throw new Error('That config.json is not valid JSON.');
  }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    throw new Error('That config.json must be a JSON object.');
  }
}

async function writeStagedConfig(directory: string, raw: string): Promise<void> {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new Error('That text is not valid JSON — paste the contents of a config.json file.');
  }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    throw new Error('That text must be a JSON object — paste the contents of a config.json file.');
  }
  await writeFile(path.join(directory, 'config.json'), JSON.stringify(parsed, null, 2));
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
