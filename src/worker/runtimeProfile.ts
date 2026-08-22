import { createHash } from 'node:crypto';

export interface WorkerModelProfile {
  loadedContextSize: number;
  hasChatTemplate: boolean;
  supportsTools: boolean;
  supportsToolCalls: boolean;
  supportsSystemRole: boolean;
  workerBuild?: string;
  chatTemplateFingerprint?: string;
}

export function parseWorkerModelProfile(value: unknown): WorkerModelProfile {
  const payload = record(value);
  const settings = record(payload.default_generation_settings);
  const capabilities = record(payload.chat_template_caps);
  const loadedContextSize = finitePositiveInteger(settings.n_ctx);
  if (!loadedContextSize) {
    throw new Error('The local worker did not report a valid loaded context size.');
  }
  const workerBuild = typeof payload.build_info === 'string' && payload.build_info
    ? payload.build_info
    : undefined;
  const chatTemplate = typeof payload.chat_template === 'string' && payload.chat_template
    ? payload.chat_template
    : undefined;
  const chatTemplateFingerprint = chatTemplate
    ? createHash('sha256').update(chatTemplate).digest('hex')
    : undefined;

  return {
    loadedContextSize,
    hasChatTemplate: Boolean(chatTemplate),
    supportsTools: capabilities.supports_tools === true,
    supportsToolCalls: capabilities.supports_tool_calls === true,
    supportsSystemRole: capabilities.supports_system_role === true,
    ...(workerBuild ? { workerBuild } : {}),
    ...(chatTemplateFingerprint ? { chatTemplateFingerprint } : {}),
  };
}

function record(value: unknown): Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function finitePositiveInteger(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) && value > 0
    ? Math.floor(value)
    : undefined;
}
