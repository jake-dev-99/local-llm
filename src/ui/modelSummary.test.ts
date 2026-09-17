import assert from 'node:assert/strict';
import test from 'node:test';
import { build } from 'esbuild';
import type { InstalledModel } from '../domain.ts';

/**
 * `modelSummary.ts` formats through `modelSources.ts`, which imports `vscode`
 * at runtime, so the module is bundled with the standard stub like
 * `modelManager.integration.test.ts`.
 */

const VSCODE_STUB = `
  export class CancellationError extends Error {}
  export const ProgressLocation = { Notification: 15 };
`;

let cached: {
  describeModel: Function;
  describeRemoval: Function;
  checkpointWarnings: Function;
} | undefined;

async function loadSummary() {
  if (cached) {
    return cached;
  }
  const bundled = await build({
    entryPoints: ['src/ui/modelSummary.ts'],
    absWorkingDir: process.cwd(),
    bundle: true,
    format: 'esm',
    platform: 'node',
    target: 'node26',
    write: false,
    plugins: [{
      name: 'stub-vscode',
      setup(pluginBuild) {
        pluginBuild.onResolve({ filter: /^vscode$/ }, () => ({
          path: 'vscode', namespace: 'stub',
        }));
        pluginBuild.onLoad({ filter: /.*/, namespace: 'stub' }, () => ({
          contents: VSCODE_STUB, loader: 'js',
        }));
      },
    }],
  });
  const source = bundled.outputFiles[0]?.contents;
  assert.ok(source, 'esbuild returned the bundled summary module');
  const url = `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`;
  const loaded = await import(url) as {
    describeModel: Function;
    describeRemoval: Function;
    checkpointWarnings: Function;
  };
  assert.ok(loaded.describeModel, 'bundled module exports describeModel');
  cached = loaded;
  return cached;
}

function model(overrides: Partial<InstalledModel> = {}): InstalledModel {
  return {
    id: 'm1',
    name: 'Qwen3-32B',
    filePath: '/models/qwen',
    fileSize: 62 * 1024 ** 3,
    sha256: 'x',
    source: 'import',
    filename: 'model.safetensors',
    installedAt: '2026-09-17',
    format: 'safetensors',
    runtime: 'transformers',
    capabilities: { toolCalling: 'unsupported', fillInMiddle: 'unsupported' },
    ...overrides,
  } as InstalledModel;
}

test('picker rows name the runtime, quant, and in-place state', async () => {
  const { describeModel } = await loadSummary();
  const summary = describeModel(
    model({ quantization: 'awq', managed: false, customCodeRequired: true }),
    { isDefault: true },
  ) as { description: string; detail: string };
  assert.match(summary.description, /Transformers/);
  assert.match(summary.description, /awq/);
  assert.match(summary.detail, /in place/);
  assert.match(summary.detail, /runs its own code/);
  assert.match(summary.detail, /default/);
});

test('gguf rows stay plain', async () => {
  const { describeModel } = await loadSummary();
  const summary = describeModel(
    model({ format: 'gguf', runtime: 'llama-cpp', filename: 'qwen.gguf' }),
  ) as { description: string; detail: string };
  assert.match(summary.description, /llama\.cpp/);
  assert.doesNotMatch(summary.detail, /in place/);
});

test('removal wording matches ownership', async () => {
  const { describeRemoval } = await loadSummary();
  const inplace = describeRemoval(model({ managed: false })) as { message: string; confirmLabel: string };
  assert.match(inplace.message, /stay.*where it is/);
  assert.equal(inplace.confirmLabel, 'Remove from List');
  const owned = describeRemoval(model({ managed: true })) as { message: string; confirmLabel: string };
  assert.match(owned.message, /Delete/);
  assert.equal(owned.confirmLabel, 'Delete Model');
});

test('custom code always earns consent; quant earns an advisory off CUDA', async () => {
  const { checkpointWarnings } = await loadSummary();
  const both = checkpointWarnings({ quantization: 'awq', customCodeRequired: true }) as {
    consent?: string; advisory?: string;
  };
  assert.ok(both.consent);
  assert.ok(both.advisory);
  assert.deepEqual(checkpointWarnings({ quantization: 'awq' }, { cudaAvailable: true }), {});
  assert.deepEqual(checkpointWarnings({}), {});
});
