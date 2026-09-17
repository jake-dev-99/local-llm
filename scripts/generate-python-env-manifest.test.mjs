import assert from 'node:assert/strict';
import test from 'node:test';

import { writeFile } from 'node:fs/promises';

import {
  filenameVersion,
  mapFilenamesToPypi,
  parseRequirements,
  pickWheel,
  requiresPythonOk,
  resolveBaseClosure,
  toManifestWheel,
  wheelPlatformTags,
  wheelPythonTag,
} from './generate-python-env-manifest.mjs';

test('requirements pins parse, comments and flags ignored', () => {
  const pins = parseRequirements('# Baseline runtime.\ntorch==2.14.0\ntransformers==5.17.0  # pinned\n--extra-index-url https://example.invalid\n');
  assert.deepEqual([...pins], [['torch', '2.14.0'], ['transformers', '5.17.0']]);
});

test('wheel tags split correctly', () => {
  assert.equal(wheelPythonTag('torch-2.14.0-cp312-cp312-macosx_11_0_arm64.whl'), 'cp312');
  assert.deepEqual(wheelPlatformTags('torch-2.14.0-cp312-cp312-macosx_11_0_arm64.whl'), ['macosx_11_0_arm64']);
  assert.deepEqual(
    wheelPlatformTags('transformers-5.17.0-py3-none-any.whl'),
    ['any'],
  );
  assert.equal(filenameVersion('nvidia_cublas_cu12-12.1.3.1-py3-none-win_amd64.whl'), '12.1.3.1');
});

const entry = (filename) => ({
  filename,
  url: `https://files.example.invalid/${filename}`,
  digests: { sha256: '0'.repeat(64) },
});

test('pure packages resolve to py3-none-any regardless of platform', () => {
  const picked = pickWheel(
    [entry('transformers-5.17.0-py3-none-any.whl')],
    { pythonTag: 'cp312', platformTags: ['win_amd64', 'any'], packageName: 'transformers' },
  );
  assert.equal(picked.filename, 'transformers-5.17.0-py3-none-any.whl');
});

test('platform packages prefer the first listed platform tag', () => {
  const picked = pickWheel(
    [
      entry('torch-2.14.0-cp312-cp312-win_amd64.whl'),
      entry('torch-2.14.0-cp312-cp312-macosx_11_0_arm64.whl'),
    ],
    { pythonTag: 'cp312', platformTags: ['macosx_11_0_arm64', 'win_amd64'], packageName: 'torch' },
  );
  assert.equal(picked.filename, 'torch-2.14.0-cp312-cp312-macosx_11_0_arm64.whl');
});

test('macOS matching tolerates the publisher version prefix', () => {
  const picked = pickWheel(
    [entry('torch-2.14.0-cp312-cp312-macosx_14_0_arm64.whl')],
    { pythonTag: 'cp312', platformTags: ['arm64', 'universal2', 'any'], packageName: 'torch' },
  );
  assert.equal(picked.filename, 'torch-2.14.0-cp312-cp312-macosx_14_0_arm64.whl');
});

test('wrong python tag never matches', () => {
  assert.equal(
    pickWheel(
      [entry('torch-2.14.0-cp311-cp311-win_amd64.whl')],
      { pythonTag: 'cp312', platformTags: ['win_amd64', 'any'], packageName: 'torch' },
    ),
    undefined,
  );
});

test('python floors gate pure wheels', () => {
  assert.equal(requiresPythonOk('>=3.10', '3.10'), true);
  assert.equal(requiresPythonOk('>=3.10', '3.12'), true);
  assert.equal(requiresPythonOk('>=3.11', '3.10'), false);
  assert.equal(requiresPythonOk(undefined, '3.10'), true);
});

test('manifest wheels carry name, url, and hash', () => {
  assert.deepEqual(toManifestWheel(entry('torch-2.14.0-cp312-cp312-win_amd64.whl'), 'torch', '2.14.0'), {
    name: 'torch==2.14.0',
    url: 'https://files.example.invalid/torch-2.14.0-cp312-cp312-win_amd64.whl',
    sha256: '0'.repeat(64),
  });
});

test('filename mapping hits the resolved version, not latest', async () => {
  const calls = [];
  const fetchJson = async (url) => {
    calls.push(url);
    return {
      info: { version: '12.1.3.1' },
      urls: [entry('nvidia_cublas_cu12-12.1.3.1-py3-none-win_amd64.whl')],
    };
  };
  const wheels = await mapFilenamesToPypi(
    ['nvidia_cublas_cu12-12.1.3.1-py3-none-win_amd64.whl'],
    new Map(),
    fetchJson,
  );
  assert.deepEqual(calls, ['https://pypi.org/pypi/nvidia-cublas-cu12/12.1.3.1/json']);
  assert.equal(wheels[0].name, 'nvidia-cublas-cu12==12.1.3.1');
});

test('filename mapping rejects a version that contradicts the pins', async () => {
  await assert.rejects(
    mapFilenamesToPypi(
      ['torch-2.15.0-cp312-cp312-win_amd64.whl'],
      new Map([['torch', '2.14.0']]),
      async () => ({ info: { version: '2.15.0' }, urls: [] }),
    ),
    /but pins say/,
  );
});

test('base closure keeps transitive deps, not just the pins', async () => {
  // Regression: per-pin PyPI lookups shipped only torch/transformers/
  // accelerate/safetensors, so the hermetic `pip install --no-index` failed
  // with "Could not find a version that satisfies the requirement
  // huggingface-hub (from transformers) (from versions: none)".
  const pins = new Map([['transformers', '5.17.0'], ['huggingface-hub', '1.32.0']]);
  const dumped = ['transformers-5.17.0-py3-none-any.whl', 'huggingface_hub-1.32.0-py3-none-any.whl'];
  const download = async (specs, { dest }) => {
    assert.deepEqual([...specs].sort(), ['huggingface-hub==1.32.0', 'transformers==5.17.0']);
    for (const filename of dumped) {
      await writeFile(`${dest}/${filename}`, 'bytes');
    }
  };
  const fetchJson = async (url) => {
    const packageName = url.split('/pypi/')[1].split('/')[0];
    const filename = dumped.find((name) => name.toLowerCase().startsWith(packageName.split('-')[0]));
    return {
      info: { version: pins.get(packageName) },
      urls: [{ filename, url: `https://files.example.invalid/${filename}`, digests: { sha256: '0'.repeat(64) } }],
    };
  };
  const wheels = await resolveBaseClosure(
    pins,
    { platform: 'win_amd64', pythonVersion: '3.10' },
    { fetchJson, download },
  );
  assert.deepEqual(wheels.map((wheel) => wheel.name), ['huggingface-hub==1.32.0', 'transformers==5.17.0']);
});
