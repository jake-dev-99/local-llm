import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import * as path from 'node:path';
import test from 'node:test';
import { build } from 'esbuild';
import { envMatchesManifest } from './pythonEnvironment.ts';

/**
 * The install flow with every effect faked: downloads write marker files,
 * commands are recorded, nothing touches the network or spawns. Proves the
 * order (flavor → match check → download → extract → venv → pip → record)
 * and the fresh/cached distinction.
 */

const VSCODE_STUB = `
  export class CancellationError extends Error {}
  export const ProgressLocation = { Notification: 15 };
`;

let cached: {
  ensureEnvironment: Function;
  envMatchesManifest: Function;
} | undefined;

async function loadProvision(): Promise<{
  ensureEnvironment: Function;
  envMatchesManifest: Function;
}> {
  if (cached) {
    return cached;
  }
  const bundled = await build({
    entryPoints: ['src/worker/pythonProvision.ts'],
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
  assert.ok(source, 'esbuild returned the bundled provision module');
  const url = `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`;
  const loaded = await import(url) as {
    ensureEnvironment: Function;
    envMatchesManifest: Function;
  };
  assert.ok(loaded.ensureEnvironment, 'bundled module exports ensureEnvironment');
  cached = loaded;
  return cached;
}

const RELEASE = {
  manifestVersion: 1,
  targets: {
    'darwin-arm64': {
      interpreter: { url: 'https://example.invalid/py.tar.gz', sha256: 'aa', exe: 'bin/python3' },
      wheels: [{ name: 'torch==2.14.0', url: 'https://example.invalid/torch.whl', sha256: 'bb' }],
    },
  },
};

function harness(storage: string) {
  const commands: Array<{ exe: string; args: string[] }> = [];
  const downloads: string[] = [];
  return {
    commands,
    downloads,
    deps: {
      download: async (_url: string, destination: string) => {
        mkdirSync(path.dirname(destination), { recursive: true });
        writeFileSync(destination, 'bytes');
        downloads.push(destination);
        return { sha256: 'x', size: 5 };
      },
      run: async (exe: string, args: string[]) => {
        commands.push({ exe, args });
        if (exe === 'tar') {
          // Simulate the flattened layout: standalone archives nest under
          // one top-level dir, stripped at extract time.
          const dest = args[args.indexOf('-C') + 1] as string;
          mkdirSync(dest, { recursive: true });
          mkdirSync(path.join(dest, 'bin'), { recursive: true });
          writeFileSync(path.join(dest, 'bin', 'python3'), 'fake');
        }
        return { stdout: '', stderr: '' };
      },
      readReleaseManifest: async () => RELEASE.targets['darwin-arm64'],
    },
    options: {
      storagePath: storage,
      target: 'darwin-arm64',
      flavorSetting: 'auto' as const,
      releaseManifestPath: path.join(storage, 'release.json'),
      progress: { report() {} },
      token: { isCancellationRequested: false, onCancellationRequested: () => ({ dispose() {} }) },
      onLog() {},
    },
  };
}

test('first use provisions in order, second use is cached', async () => {
  const provision = await loadProvision();
  const storage = mkdtempSync(path.join(tmpdir(), 'local-llm-prov-'));
  try {
    const first = harness(storage);
    const installed = await provision.ensureEnvironment(first.options, first.deps) as {
      pythonPath: string; flavor: string; fresh: boolean;
    };
    assert.equal(installed.flavor, 'cpu');
    assert.equal(installed.fresh, true);
    assert.match(installed.pythonPath, /venv/);
    assert.equal(first.downloads.length, 2);
    const verbs = first.commands.map((command: { exe: string; args: string[] }) =>
      command.args.includes('-m') ? command.args[command.args.indexOf('-m') + 1] : command.exe);
    assert.deepEqual(verbs, ['tar', 'venv', 'pip']);
    assert.ok(first.commands.some((command: { exe: string; args: string[] }) =>
      command.args.includes('--no-index')));

    const second = harness(storage);
    const cachedResult = await provision.ensureEnvironment(second.options, second.deps) as { fresh: boolean };
    assert.equal(cachedResult.fresh, false);
    assert.equal(second.downloads.length, 0);
    assert.equal(second.commands.length, 0);
  } finally {
    rmSync(storage, { recursive: true, force: true });
  }
});

test('extraction passes --strip-components for the nested python dir', async () => {
  const provision = await loadProvision();
  const storage = mkdtempSync(path.join(tmpdir(), 'local-llm-prov-'));
  try {
    const attempt = harness(storage);
    await provision.ensureEnvironment(
      { ...attempt.options, probe: async () => undefined },
      attempt.deps,
    );
    const tar = attempt.commands.find((command: { exe: string }) => command.exe === 'tar');
    assert.ok(tar, 'extraction ran');
    assert.ok((tar as { args: string[] }).args.includes('--strip-components'));
  } finally {
    rmSync(storage, { recursive: true, force: true });
  }
});

test('a missing interpreter exe fails named, not as a bare ENOENT', async () => {
  const provision = await loadProvision();
  const storage = mkdtempSync(path.join(tmpdir(), 'local-llm-prov-'));
  try {
    const attempt = harness(storage);
    const bare = {
      ...attempt.deps,
      run: async () => ({ stdout: '', stderr: '' }),
    };
    await assert.rejects(
      provision.ensureEnvironment(
        { ...attempt.options, probe: async () => undefined },
        bare,
      ),
      /unexpected layout/,
    );
  } finally {
    rmSync(storage, { recursive: true, force: true });
  }
});

test('a failed probe blocks the record so the next use retries', async () => {
  const provision = await loadProvision();
  const storage = mkdtempSync(path.join(tmpdir(), 'local-llm-prov-'));
  try {
    const attempt = harness(storage);
    await assert.rejects(
      provision.ensureEnvironment(
        { ...attempt.options, probe: async () => { throw new Error('torch aborts here'); } },
        attempt.deps,
      ),
      /torch aborts here/,
    );
    assert.equal(
      await envMatchesManifest(storage, 'darwin-arm64', 'cpu', RELEASE.targets['darwin-arm64']),
      false,
    );
  } finally {
    rmSync(storage, { recursive: true, force: true });
  }
});
