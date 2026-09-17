import { access, stat } from 'node:fs/promises';
import * as vscode from 'vscode';
import type {
  CapabilitySupport,
  InstalledModel,
  ModelRuntimeProfile,
  NativeToolCapabilityRecord,
} from '../domain.js';
import type { LocalLlmLogger } from '../logging.js';
import { readGgufMetadata } from './ggufMetadata.js';
import { directoryHasChanged, runtimeForFormat } from './modelIdentity.ts';
import { sha256File } from './modelSources.js';
import {
  fingerprintCheckpoint,
  readSafetensorsCheckpoint,
} from './safetensorsDirectory.ts';

const REGISTRY_KEY = 'localLlm.installedModels.v1';

interface VerifiedModel {
  model: InstalledModel;
  /** True when the stored record differs from what was read back. */
  rewritten: boolean;
}

/**
 * Drops everything that was only true of the bytes that used to be on disk.
 *
 * Capabilities are re-verified against the model actually present rather than
 * inherited from the one it replaced.
 */
function withoutVerifiedCapabilities(model: InstalledModel): InstalledModel {
  const {
    runtimeProfile: _runtimeProfile,
    nativeToolCapability: _nativeToolCapability,
    ...rest
  } = model;
  return {
    ...rest,
    capabilities: { toolCalling: 'unverified', fillInMiddle: 'unverified' },
  };
}

export class ModelRegistry {
  private readonly changeEmitter = new vscode.EventEmitter<void>();
  private models: InstalledModel[] = [];

  readonly onDidChange = this.changeEmitter.event;

  constructor(
    private readonly state: vscode.Memento,
    private readonly logger: LocalLlmLogger,
  ) {}

  async initialize(): Promise<void> {
    const stored = this.state.get<InstalledModel[]>(REGISTRY_KEY, []);
    const available: InstalledModel[] = [];
    let migrated = false;
    for (const storedModel of stored) {
      const model = normalizeStoredModel(storedModel);
      migrated ||= model !== storedModel;
      try {
        await access(model.filePath);
        const verified = model.format === 'safetensors'
          ? await this.verifyDirectoryModel(model)
          : await this.verifyFileModel(model);
        if (verified) {
          migrated ||= verified.rewritten;
          available.push(verified.model);
        }
      } catch (error) {
        this.logger.error(
          `Removed unavailable model from registry: ${model.name} (${model.filePath})`,
          error,
        );
      }
    }
    this.models = available;
    if (available.length !== stored.length || migrated) {
      await this.persist();
    }
  }

  /**
   * Verifies a single-file model: GGUF, loaded by llama.cpp.
   *
   * Returns undefined when the path is no longer a file, which drops it from
   * the registry the same way a missing path does.
   */
  private async verifyFileModel(model: InstalledModel): Promise<VerifiedModel | undefined> {
    const metadata = await stat(model.filePath);
    if (!metadata.isFile()) {
      return undefined;
    }
    const identityChanged = model.fileModifiedAt !== undefined && (
      model.fileSize !== metadata.size || model.fileModifiedAt !== metadata.mtimeMs
    );
    if (identityChanged) {
      this.logger.info(
        `Model bytes changed on disk; invalidated compatibility: ${model.name}.`,
      );
      return {
        rewritten: true,
        model: {
          ...withoutVerifiedCapabilities(model),
          fileSize: metadata.size,
          fileModifiedAt: metadata.mtimeMs,
          sha256: await sha256File(model.filePath),
        },
      };
    }
    let rewritten = model.fileSize !== metadata.size ||
      model.fileModifiedAt !== metadata.mtimeMs;
    // Models installed before the GGUF header was read have no trained window
    // recorded. Backfill it rather than requiring a reinstall.
    let trainedContextLength = model.trainedContextLength;
    if (trainedContextLength === undefined) {
      trainedContextLength = (await readGgufMetadata(
        model.filePath,
        (warning) => this.logger.info(warning),
      ))?.trainedContextLength;
      rewritten ||= trainedContextLength !== undefined;
    }
    return {
      rewritten,
      model: {
        ...model,
        fileSize: metadata.size,
        fileModifiedAt: metadata.mtimeMs,
        ...(trainedContextLength ? { trainedContextLength } : {}),
      },
    };
  }

  /**
   * Verifies a directory model: a Safetensors checkpoint, loaded by the Python
   * runtime.
   *
   * Compares per-file fingerprints rather than re-hashing. These checkpoints
   * reach tens of gigabytes and this runs for every model on activation, so the
   * cost has to track the file count, not the bytes on disk. Headers are re-read
   * only once a fingerprint has actually moved.
   */
  private async verifyDirectoryModel(
    model: InstalledModel,
  ): Promise<VerifiedModel | undefined> {
    const metadata = await stat(model.filePath);
    if (!metadata.isDirectory()) {
      return undefined;
    }
    const current = await fingerprintCheckpoint(model.filePath);
    if (current.length === 0) {
      // An emptied directory is gone as far as the registry is concerned.
      return undefined;
    }
    if (directoryHasChanged(model.files, current)) {
      this.logger.info(
        `Checkpoint files changed on disk; invalidated compatibility: ${model.name}.`,
      );
      const checkpoint = await readSafetensorsCheckpoint(
        model.filePath,
        (warning) => this.logger.info(warning),
      );
      return {
        rewritten: true,
        model: {
          ...withoutVerifiedCapabilities(model),
          files: checkpoint.identity.files,
          fileSize: checkpoint.identity.totalBytes,
          sha256: checkpoint.identity.digest,
          ...(checkpoint.trainedContextLength
            ? { trainedContextLength: checkpoint.trainedContextLength }
            : {}),
        },
      };
    }
    // Installed before fingerprints were recorded. Backfill rather than
    // discarding capabilities the model has already been verified for.
    if (!model.files || model.files.length === 0) {
      return {
        rewritten: true,
        model: { ...model, files: current },
      };
    }
    return { rewritten: false, model };
  }

  list(): readonly InstalledModel[] {
    return [...this.models].sort((a, b) => a.name.localeCompare(b.name));
  }

  get(id: string): InstalledModel | undefined {
    return this.models.find((model) => model.id === id);
  }

  async upsert(model: InstalledModel): Promise<void> {
    const index = this.models.findIndex((item) => item.id === model.id);
    if (index >= 0) {
      this.models[index] = model;
    } else {
      this.models.push(model);
    }
    await this.persist();
    this.changeEmitter.fire();
  }

  async remove(id: string): Promise<InstalledModel | undefined> {
    const index = this.models.findIndex((item) => item.id === id);
    if (index < 0) {
      return undefined;
    }
    const [removed] = this.models.splice(index, 1);
    await this.persist();
    this.changeEmitter.fire();
    return removed;
  }

  async updateRuntimeProfile(id: string, profile: ModelRuntimeProfile): Promise<void> {
    const model = this.get(id);
    if (!model) {
      return;
    }
    const toolCalling: CapabilitySupport = profile.supportsTools && profile.supportsToolCalls
      ? model.capabilities.toolCalling === 'supported' ? 'supported' : 'unverified'
      : 'unsupported';
    const updated: InstalledModel = {
      ...model,
      capabilities: { ...model.capabilities, toolCalling },
      runtimeProfile: profile,
    };
    if (JSON.stringify(updated) !== JSON.stringify(model)) {
      await this.upsert(updated);
    }
  }

  async markToolCalling(id: string, support: CapabilitySupport): Promise<void> {
    await this.markCapability(id, 'toolCalling', support);
  }

  async updateNativeToolCapability(
    id: string,
    record: NativeToolCapabilityRecord,
  ): Promise<void> {
    const model = this.get(id);
    if (!model || JSON.stringify(model.nativeToolCapability) === JSON.stringify(record)) {
      return;
    }
    await this.upsert({ ...model, nativeToolCapability: record });
  }

  async markFillInMiddle(id: string, support: CapabilitySupport): Promise<void> {
    await this.markCapability(id, 'fillInMiddle', support);
  }

  dispose(): void {
    this.changeEmitter.dispose();
  }

  private async persist(): Promise<void> {
    await this.state.update(REGISTRY_KEY, this.models);
  }

  private async markCapability(
    id: string,
    capability: keyof InstalledModel['capabilities'],
    support: CapabilitySupport,
  ): Promise<void> {
    const model = this.get(id);
    if (!model || model.capabilities[capability] === support) {
      return;
    }
    await this.upsert({
      ...model,
      capabilities: { ...model.capabilities, [capability]: support },
    });
  }
}

function normalizeStoredModel(model: InstalledModel): InstalledModel {
  const toolCalling = normalizeSupport(
    (model.capabilities as { toolCalling?: unknown } | undefined)?.toolCalling,
  );
  const fillInMiddle = normalizeSupport(
    (model.capabilities as { fillInMiddle?: unknown } | undefined)?.fillInMiddle,
  );
  // Records written before Safetensors support existed carry no format. Every
  // one of them is a single GGUF file served by llama.cpp, so they are labelled
  // rather than discarded.
  const format = model.format === 'safetensors' ? 'safetensors' : 'gguf';
  const runtime = runtimeForFormat(format);
  if (
    model.capabilities?.toolCalling === toolCalling &&
    model.capabilities.fillInMiddle === fillInMiddle &&
    model.format === format &&
    model.runtime === runtime
  ) {
    return model;
  }
  return {
    ...model,
    format,
    runtime,
    capabilities: { toolCalling, fillInMiddle },
  };
}

function normalizeSupport(value: unknown): CapabilitySupport {
  return value === 'supported' || value === 'unsupported' || value === 'unverified'
    ? value
    : 'unverified';
}
