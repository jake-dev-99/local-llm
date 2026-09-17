import * as os from 'node:os';
import * as path from 'node:path';
import * as vscode from 'vscode';
import type { LocalLlmConfig } from './domain.js';

const SECTION = 'localLlm';

export function defaultModelDirectory(context: vscode.ExtensionContext): string {
  return path.join(context.globalStorageUri.fsPath, 'models');
}

export function readConfig(context: vscode.ExtensionContext): LocalLlmConfig {
  const config = vscode.workspace.getConfiguration(SECTION);
  const configuredDirectory = config.get<string>('modelDirectory', '').trim();
  const batchSize = config.get<number>('batchSize', 256);
  const microBatchSize = Math.min(
    batchSize,
    config.get<number>('microBatchSize', 64),
  );

  return {
    modelDirectory: configuredDirectory
      ? expandHome(configuredDirectory)
      : defaultModelDirectory(context),
    defaultModelId: config.get<string>('defaultModelId', ''),
    contextSize: config.get<number>('contextSize', 0),
    maxTools: config.get<number>('maxTools', 128),
    maxAgentToolRounds: config.get<number>('maxAgentToolRounds', 8),
    maxOutputTokens: config.get<number>('maxOutputTokens', 2048),
    maxToolCallTokens: config.get<number>('maxToolCallTokens', 512),
    startupTimeoutMilliseconds:
      config.get<number>('startupTimeoutSeconds', 600) * 1_000,
    cpuThreads: config.get<number>('cpuThreads', 0),
    acceleration: config.get<'auto' | 'cpu'>('acceleration', 'auto'),
    batchSize,
    microBatchSize,


    metalMemoryReserveMiB: config.get<number>('metalMemoryReserveMiB', 1024),
    temperature: config.get<number>('temperature', 0.2),
    inlineEnabled: config.get<boolean>('inline.enabled', true),
    inlineMaxTokens: config.get<number>('inline.maxTokens', 64),
    inlineDebounceMilliseconds: config.get<number>('inline.debounceMilliseconds', 250),
    pythonPath: config.get<string>('pythonPath', '').trim(),
    logLevel: config.get<'error' | 'info' | 'debug'>('logLevel', 'info'),
  };
}

export async function setDefaultModelId(modelId: string): Promise<void> {
  await vscode.workspace
    .getConfiguration(SECTION)
    .update('defaultModelId', modelId, vscode.ConfigurationTarget.Global);
}

function expandHome(value: string): string {
  if (value === '~') {
    return os.homedir();
  }
  if (value.startsWith(`~${path.sep}`)) {
    return path.join(os.homedir(), value.slice(2));
  }
  return path.resolve(value);
}
