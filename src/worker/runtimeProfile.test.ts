import assert from 'node:assert/strict';
import test from 'node:test';
import { parseWorkerModelProfile } from './runtimeProfile.ts';

test('parseWorkerModelProfile reads runtime context and tool-template support', () => {
  assert.deepEqual(
    parseWorkerModelProfile({
      default_generation_settings: { n_ctx: 32_768 },
      chat_template: '{{ tools }}',
      chat_template_caps: {
        supports_tools: true,
        supports_tool_calls: true,
        supports_system_role: true,
      },
      build_info: 'b10472-60eeeb608',
    }),
    {
      loadedContextSize: 32_768,
      hasChatTemplate: true,
      supportsTools: true,
      supportsToolCalls: true,
      supportsSystemRole: true,
      workerBuild: 'b10472-60eeeb608',
      chatTemplateFingerprint: 'e45627f95325271d40ad1ed82e196e096212abb84cce29ccb7f72733291af356',
    },
  );
});

test('parseWorkerModelProfile treats missing tool capabilities as unsupported', () => {
  assert.deepEqual(
    parseWorkerModelProfile({ default_generation_settings: { n_ctx: 8_192 } }),
    {
      loadedContextSize: 8_192,
      hasChatTemplate: false,
      supportsTools: false,
      supportsToolCalls: false,
      supportsSystemRole: false,
    },
  );
});

test('parseWorkerModelProfile rejects a missing loaded context size', () => {
  assert.throws(
    () => parseWorkerModelProfile({ chat_template: '{{ messages }}' }),
    /loaded context size/i,
  );
});
