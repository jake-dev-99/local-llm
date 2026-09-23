import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import * as path from 'node:path';
import test from 'node:test';
import { writeFileSync } from 'node:fs';
import {
  detectWindowsGpuFlavor,
  envDirectory,
  envMatchesManifest,
  hasIntelArc,
  installedManifestPath,
  parseEnvManifest,
  parseNvidiaSmi,
  parseVideoControllerNames,
  resolveEnvFlavor,
  resolveEnvTarget,
  wheelsForFlavor,
  type EnvManifestTarget,
} from './pythonEnvironment.ts';

test('nvidia-smi exit decides CUDA', () => {
  assert.equal(parseNvidiaSmi(true), true);
  assert.equal(parseNvidiaSmi(false), false);
});

test('video controller names split on newlines', () => {
  assert.deepEqual(
    parseVideoControllerNames('Intel(R) Arc(TM) 140T Graphics\r\nNVIDIA GeForce RTX 4090\r\n'),
    ['Intel(R) Arc(TM) 140T Graphics', 'NVIDIA GeForce RTX 4090'],
  );
});

test('Intel Arc detection covers discrete and integrated names', () => {
  assert.equal(hasIntelArc(['Intel(R) Arc(TM) 140T Graphics']), true);
  assert.equal(hasIntelArc(['Intel(R) Arc(TM) A770 Graphics']), true);
  assert.equal(hasIntelArc(['NVIDIA GeForce RTX 4090']), false);
  assert.equal(hasIntelArc([]), false);
});

test('detection prefers CUDA, then Arc, then CPU, never throws', async () => {
  assert.equal(
    await detectWindowsGpuFlavor(async () => ({ stdout: 'GPU 0: NVIDIA', stderr: '' })),
    'cuda',
  );
  assert.equal(
    await detectWindowsGpuFlavor(async (exe) => {
      if (exe === 'nvidia-smi') {
        throw new Error('not found');
      }
      return { stdout: 'Intel(R) Arc(TM) 140T Graphics', stderr: '' };
    }),
    'xpu',
  );
  assert.equal(
    await detectWindowsGpuFlavor(async () => {
      throw new Error('no shell at all');
    }),
    'cpu',
  );
});

test('a GPU probe that fails for a reason other than absence is reported', async () => {
  const missing = Object.assign(new Error('spawn nvidia-smi ENOENT'), { code: 'ENOENT' });
  const quiet: string[] = [];
  assert.equal(
    await detectWindowsGpuFlavor(async (exe) => {
      if (exe === 'nvidia-smi') {
        throw missing;
      }
      return { stdout: 'Intel(R) Arc(TM) 140T Graphics', stderr: '' };
    }, (warning) => quiet.push(warning)),
    'xpu',
  );
  assert.deepEqual(quiet, [], 'no NVIDIA driver is the normal case on Intel, not a warning');

  const warnings: string[] = [];
  assert.equal(
    await detectWindowsGpuFlavor(async (exe) => {
      throw exe === 'nvidia-smi'
        ? Object.assign(new Error('NVIDIA-SMI has failed'), { code: 9 })
        : new Error('Get-CimInstance : Access denied');
    }, (warning) => warnings.push(warning)),
    'cpu',
  );
  assert.match(warnings[0] ?? '', /nvidia-smi is installed but failed/);
  assert.match(warnings[1] ?? '', /falls back to CPU: .*Access denied/);
});

test('target resolution covers macOS arm64 and Windows x64 only', () => {
  assert.equal(resolveEnvTarget('darwin', 'arm64'), 'darwin-arm64');
  assert.equal(resolveEnvTarget('win32', 'x64'), 'win32-x64');
  assert.throws(() => resolveEnvTarget('linux', 'x64'), /pythonPath/);
  assert.throws(() => resolveEnvTarget('win32', 'arm64'), /pythonPath/);
  assert.throws(() => resolveEnvTarget('darwin', 'x64'), /pythonPath/);
});

test('flavor resolution honors the setting over detection', async () => {
  assert.equal(await resolveEnvFlavor('win32-x64', 'cuda', async () => 'xpu'), 'cuda');
  assert.equal(await resolveEnvFlavor('darwin-arm64', 'auto'), 'cpu');
  assert.equal(await resolveEnvFlavor('win32-x64', 'auto', async () => 'xpu'), 'xpu');
  assert.equal(await resolveEnvFlavor('linux-x64', 'auto', async () => 'cuda'), 'cpu');
});

test('env paths nest under global storage', () => {
  assert.equal(
    envDirectory('/storage', 'win32-x64', 'cuda'),
    path.join('/storage', 'python-env', 'win32-x64', 'cuda'),
  );
  assert.equal(
    installedManifestPath('/storage', 'win32-x64', 'cuda'),
    path.join('/storage', 'python-env', 'win32-x64', 'cuda.json'),
  );
});

test('release manifest parsing rejects the wrong shape', () => {
  assert.throws(() => parseEnvManifest(null), /not an object/);
  assert.throws(() => parseEnvManifest({ manifestVersion: 2, targets: {} }), /unsupported/);
  const valid = parseEnvManifest({ manifestVersion: 1, targets: {} });
  assert.deepEqual(valid.targets, {});
});

const targetEntry: EnvManifestTarget = {
  interpreter: { url: 'https://example.invalid/py.tar.gz', sha256: 'aa', exe: 'bin/python3' },
  wheels: [{ name: 'torch==2.14.0', url: 'https://example.invalid/torch.whl', sha256: 'bb' }],
};

test('a pack can replace base wheels instead of adding', async () => {
  const entry: EnvManifestTarget = {
    interpreter: targetEntry.interpreter,
    wheels: [
      { name: 'torch==2.14.0', url: 'https://example.invalid/torch.whl', sha256: 'bb' },
      { name: 'transformers==5.17.0', url: 'https://example.invalid/tr.whl', sha256: 'cc' },
    ],
    packs: {
      xpu: {
        replaces: ['torch'],
        wheels: [{ name: 'torch==2.5.1+cxx11.abi', url: 'https://example.invalid/torch-xpu.whl', sha256: 'dd' }],
      },
    },
  };
  const flavored = wheelsForFlavor(entry, 'xpu').map((wheel) => wheel.name).sort();
  assert.deepEqual(flavored, ['torch==2.5.1+cxx11.abi', 'transformers==5.17.0']);
  assert.deepEqual(wheelsForFlavor(entry, 'cuda').map((wheel) => wheel.name).sort(), [
    'torch==2.14.0',
    'transformers==5.17.0',
  ]);
});

test('pack wheels win incidental version collisions', () => {
  const entry: EnvManifestTarget = {
    interpreter: targetEntry.interpreter,
    wheels: [
      { name: 'sympy==1.14.0', url: 'https://example.invalid/s1.whl', sha256: 'bb' },
    ],
    packs: {
      cuda: {
        wheels: [{ name: 'sympy==1.13.1', url: 'https://example.invalid/s2.whl', sha256: 'cc' }],
      },
    },
  };
  assert.deepEqual(wheelsForFlavor(entry, 'cuda').map((wheel) => wheel.name), ['sympy==1.13.1']);
});

test('manifest match requires identical hashes', async () => {
  const storage = mkdtempSync(path.join(tmpdir(), 'local-llm-env-'));
  try {
    assert.equal(await envMatchesManifest(storage, 'darwin-arm64', 'cpu', targetEntry), false);
    const recordPath = installedManifestPath(storage, 'darwin-arm64', 'cpu');
    mkdirSync(path.dirname(recordPath), { recursive: true });
    writeFileSync(recordPath, JSON.stringify({ installed: targetEntry }));
    assert.equal(await envMatchesManifest(storage, 'darwin-arm64', 'cpu', targetEntry), true);
    const drifted: EnvManifestTarget = {
      ...targetEntry,
      wheels: [{ ...targetEntry.wheels[0]!, sha256: 'cc' }],
    };
    assert.equal(await envMatchesManifest(storage, 'darwin-arm64', 'cpu', drifted), false);
  } finally {
    rmSync(storage, { recursive: true, force: true });
  }
});
