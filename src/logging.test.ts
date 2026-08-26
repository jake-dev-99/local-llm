import assert from 'node:assert/strict';
import test from 'node:test';
import { build, type Plugin } from 'esbuild';

interface LoggerTestState {
  lines: string[];
  warnings: Array<{ message: string; actions: string[] }>;
  errors: Array<{ message: string; actions: string[] }>;
  showCount: number;
}

interface TestLogger {
  warn?(message: string, visible?: boolean): void;
  error(message: string, error?: unknown, visible?: boolean): void;
}

interface TestLoggerConstructor {
  new(level: 'error' | 'info' | 'debug'): TestLogger;
}

let loggerModuleSequence = 0;

test('a visible warning writes to the output channel and offers Show Logs', async () => {
  const loaded = await loadLogger();
  const logger = new loaded.LocalLlmLogger('info');

  logger.warn?.('Chat generation has produced no data for 30 seconds.', true);
  await new Promise((resolve) => setImmediate(resolve));

  assert.equal(
    loaded.state.lines.some((line) => (
      line.includes('[WARN] Chat generation has produced no data for 30 seconds.')
    )),
    true,
  );
  assert.deepEqual(loaded.state.warnings, [{
    message: 'Local LLM: Chat generation has produced no data for 30 seconds.',
    actions: ['Show Logs'],
  }]);
  assert.equal(loaded.state.showCount, 1);
});

test('a visible error preserves its cause and offers Show Logs', async () => {
  const loaded = await loadLogger();
  const logger = new loaded.LocalLlmLogger('error');

  logger.error('Local chat request failed', new Error('worker socket reset'), true);
  await new Promise((resolve) => setImmediate(resolve));

  assert.equal(
    loaded.state.lines.some((line) => (
      line.includes('[ERROR] Local chat request failed: Error: worker socket reset')
    )),
    true,
  );
  assert.deepEqual(loaded.state.errors, [{
    message: 'Local LLM: Local chat request failed: Error: worker socket reset',
    actions: ['Show Logs'],
  }]);
  assert.equal(loaded.state.showCount, 1);
});

async function loadLogger(): Promise<{
  LocalLlmLogger: TestLoggerConstructor;
  state: LoggerTestState;
}> {
  const vscodeStub: Plugin = {
    name: 'vscode-stub',
    setup(buildContext) {
      buildContext.onResolve({ filter: /^vscode$/ }, () => ({
        path: 'vscode',
        namespace: 'test-stub',
      }));
      buildContext.onLoad({ filter: /.*/, namespace: 'test-stub' }, () => ({
        contents: `
          export const __testState = {
            lines: [],
            warnings: [],
            errors: [],
            showCount: 0,
          };
          export const window = {
            createOutputChannel() {
              return {
                appendLine(line) { __testState.lines.push(line); },
                show() { __testState.showCount += 1; },
                dispose() {},
              };
            },
            showWarningMessage(message, ...actions) {
              __testState.warnings.push({ message, actions });
              return Promise.resolve('Show Logs');
            },
            showErrorMessage(message, ...actions) {
              __testState.errors.push({ message, actions });
              return Promise.resolve('Show Logs');
            },
          };
        `,
        loader: 'js',
      }));
    },
  };
  const bundled = await build({
    stdin: {
      contents: `
        export { LocalLlmLogger } from './src/logging.ts';
        export { __testState as state } from 'vscode';
      `,
      resolveDir: process.cwd(),
      sourcefile: 'logging-test-entry.ts',
      loader: 'ts',
    },
    bundle: true,
    format: 'esm',
    platform: 'node',
    target: 'node26',
    plugins: [vscodeStub],
    write: false,
  });
  const source = bundled.outputFiles[0]?.contents;
  assert.ok(source, 'esbuild returned the bundled logger');
  loggerModuleSequence += 1;
  return await import(
    `data:text/javascript;base64,${Buffer.from(source).toString('base64')}#${loggerModuleSequence}`
  ) as {
    LocalLlmLogger: TestLoggerConstructor;
    state: LoggerTestState;
  };
}
