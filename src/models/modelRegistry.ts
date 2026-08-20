import { access, stat } from 'node:fs/promises';
import * as vscode from 'vscode';
import type {
  CapabilitySupport,
  InstalledModel,
  ModelRuntimeProfile,
} from '../domain';
import type { LocalLlmLogger } from '../logging';
import { readGgufMetadata } from './ggufMetadata';
import { sha256File } from './modelSources';

const REGISTRY_KEY = 'localLlm.installedModels.v1';

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
        const metadata = await stat(model.filePath);
        if (metadata.isFile()) {
          const identityChanged = model.fileModifiedAt !== undefined && (
            model.fileSize !== metadata.size || model.fileModifiedAt !== metadata.mtimeMs
          );
          if (identityChanged) {
            migrated = true;
            this.logger.info(`Model bytes changed on disk; invalidated compatibility: ${model.name}.`);
            const { runtimeProfile: _runtimeProfile, ...modelWithoutRuntimeProfile } = model;
            available.push({
              ...modelWithoutRuntimeProfile,
              fileSize: metadata.size,
              fileModifiedAt: metadata.mtimeMs,
              sha256: await sha256File(model.filePath),
              capabilities: { toolCalling: 'unverified', fillInMiddle: 'unverified' },
            });
          } else {
            migrated ||= model.fileSize !== metadata.size || model.fileModifiedAt !== metadata.mtimeMs;
            // Models installed before the GGUF header was read have no trained
            // window recorded. Backfill it rather than requiring a reinstall.
            let trainedContextLength = model.trainedContextLength;
            if (trainedContextLength === undefined) {
              trainedContextLength = (await readGgufMetadata(
                model.filePath,
                (warning) => this.logger.info(warning),
              ))?.trainedContextLength;
              migrated ||= trainedContextLength !== undefined;
            }
            available.push({
              ...model,
              fileSize: metadata.size,
              fileModifiedAt: metadata.mtimeMs,
              ...(trainedContextLength ? { trainedContextLength } : {}),
            });
          }
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
  if (
    model.capabilities?.toolCalling === toolCalling &&
    model.capabilities.fillInMiddle === fillInMiddle
  ) {
    return model;
  }
  return { ...model, capabilities: { toolCalling, fillInMiddle } };
}

function normalizeSupport(value: unknown): CapabilitySupport {
  return value === 'supported' || value === 'unsupported' || value === 'unverified'
    ? value
    : 'unverified';
}
