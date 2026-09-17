import assert from 'node:assert/strict';
import test from 'node:test';

import {
  baseVersion,
  collectUrls,
  interpreterPythonTag,
  mergeWheels,
  normalizeName,
  parseArgs,
  parsePin,
  parseRequirementPins,
  parseWheelFilename,
  platformTagAdmits,
  pythonTagAdmits,
  safeWheelFilename,
  validateManifest,
} from './validate-python-env-manifest.mjs';

const SHA = (seed) => String(seed).repeat(64).slice(0, 64);

const wheel = (name, filename, sha = SHA('a')) => ({
  name,
  url: `https://files.pythonhosted.org/packages/ab/cd/${filename}`,
  sha256: sha,
});

const darwin = (overrides = {}) => ({
  interpreter: {
    url: 'https://example.invalid/cpython-3.10.21%2B20260901-aarch64-apple-darwin-install_only.tar.gz',
    sha256: SHA('b'),
    exe: 'bin/python3',
  },
  wheels: [wheel('transformers==5.17.0', 'transformers-5.17.0-py3-none-any.whl')],
  ...overrides,
});

const windows = (overrides = {}) => ({
  interpreter: {
    url: 'https://example.invalid/cpython-3.10.21%2B20260901-x86_64-pc-windows-msvc-install_only.tar.gz',
    sha256: SHA('c'),
    exe: 'python.exe',
  },
  wheels: [wheel('transformers==5.17.0', 'transformers-5.17.0-py3-none-any.whl')],
  ...overrides,
});

const manifestOf = (targets) => ({ manifestVersion: 1, targets });
const both = (overrides = {}) => manifestOf({
  'darwin-arm64': darwin(overrides['darwin-arm64']),
  'win32-x64': windows(overrides['win32-x64']),
});
const codes = (findings, level) => findings
  .filter((item) => level === undefined || item.level === level)
  .map((item) => item.code);

test('name normalization follows PEP 503 on both sides of the hyphen/underscore split', () => {
  assert.equal(normalizeName('Huggingface_Hub'), 'huggingface-hub');
  assert.equal(normalizeName('annotated.doc'), 'annotated-doc');
  assert.equal(normalizeName('markdown-it-py'), 'markdown-it-py');
});

test('pins accept local version segments', () => {
  assert.deepEqual(parsePin('torch==2.14.0+cpu'), { name: 'torch', version: '2.14.0+cpu' });
  assert.deepEqual(parsePin('typing-extensions==4.16.0'), { name: 'typing-extensions', version: '4.16.0' });
  assert.equal(parsePin('torch>=2.14.0'), undefined);
  assert.equal(baseVersion('2.12.0+xpu'), '2.12.0');
});

test('wheel filenames parse, with and without a build tag', () => {
  assert.deepEqual(parseWheelFilename('hf_xet-1.6.0-cp38-abi3-macosx_11_0_arm64.whl'), {
    name: 'hf_xet', version: '1.6.0', pythonTag: 'cp38', abiTag: 'abi3', platformTag: 'macosx_11_0_arm64',
  });
  assert.equal(parseWheelFilename('colorama-0.4.6-py2.py3-none-any.whl').pythonTag, 'py2.py3');
  assert.equal(parseWheelFilename('foo-1.0-1-py3-none-any.whl').version, '1.0');
  assert.equal(parseWheelFilename('torch-2.14.0+cpu-cp310-cp310-win_amd64.whl').version, '2.14.0+cpu');
});

test('a percent-encoded local version is not a wheel filename pip accepts', () => {
  // pip 26 raises InvalidWheelFilename on this exact name; safeWheelFilename
  // writes it to disk verbatim, so the install dies after the download.
  assert.equal(parseWheelFilename('torch-2.14.0%2Bcpu-cp310-cp310-win_amd64.whl'), undefined);
  assert.equal(parseWheelFilename('transformers-5.17.0.tar.gz'), undefined);
});

test('safeWheelFilename mirrors the provisioner: last url segment, no decoding', () => {
  assert.equal(
    safeWheelFilename({ name: 'torch==2.14.0+cpu', url: 'https://h/whl/cpu/torch-2.14.0%2Bcpu-cp310-cp310-win_amd64.whl' }),
    'torch-2.14.0%2Bcpu-cp310-cp310-win_amd64.whl',
  );
  assert.equal(
    safeWheelFilename({ name: 'x==1', url: 'https://h/x-1-py3-none-any.whl?token=abc' }),
    'x-1-py3-none-any.whl',
  );
});

test('interpreter version reads through percent-encoding', () => {
  assert.equal(interpreterPythonTag('https://h/cpython-3.10.21%2B20260901-aarch64-apple-darwin-install_only.tar.gz'), 'cp310');
  assert.equal(interpreterPythonTag('https://h/not-a-python.tar.gz'), undefined);
});

test('abi3 wheels declare a floor, plain cp tags do not', () => {
  assert.ok(pythonTagAdmits('cp38', 'abi3', 'cp310'));
  assert.ok(pythonTagAdmits('cp310', 'cp310', 'cp310'));
  assert.ok(pythonTagAdmits('py3', 'none', 'cp310'));
  assert.ok(pythonTagAdmits('py2.py3', 'none', 'cp310'));
  assert.ok(!pythonTagAdmits('cp38', 'cp38', 'cp310'));
  assert.ok(!pythonTagAdmits('cp312', 'cp312', 'cp310'));
  assert.ok(!pythonTagAdmits('py2', 'none', 'cp310'));
});

test('platform tags match their target only', () => {
  assert.ok(platformTagAdmits('macosx_11_0_arm64', 'darwin-arm64'));
  assert.ok(platformTagAdmits('macosx_10_9_universal2.macosx_11_0_arm64', 'darwin-arm64'));
  assert.ok(platformTagAdmits('any', 'win32-x64'));
  assert.ok(platformTagAdmits('win_amd64', 'win32-x64'));
  assert.ok(!platformTagAdmits('win_amd64', 'darwin-arm64'));
  assert.ok(!platformTagAdmits('macosx_11_0_x86_64', 'darwin-arm64'));
  assert.ok(!platformTagAdmits('win32', 'win32-x64'));
});

test('a well-formed manifest produces no findings', () => {
  assert.deepEqual(validateManifest(both()), []);
});

test('unsupported version and missing targets are refused', () => {
  assert.equal(validateManifest({ manifestVersion: 2, targets: {} })[0].code, 'VERSION');
  const findings = validateManifest(manifestOf({ 'darwin-arm64': darwin(), 'linux-x64': darwin() }));
  assert.ok(codes(findings).includes('UNKNOWN_TARGET'));
  assert.ok(codes(findings).includes('MISSING_TARGET'));
});

test('a wheel for the wrong platform or interpreter is an error', () => {
  const findings = validateManifest(both({
    'darwin-arm64': {
      wheels: [
        wheel('numpy==2.2.6', 'numpy-2.2.6-cp310-cp310-win_amd64.whl', SHA('d')),
        wheel('regex==2026.9.10', 'regex-2026.9.10-cp312-cp312-macosx_11_0_arm64.whl', SHA('e')),
      ],
    },
  }));
  assert.ok(codes(findings, 'error').includes('PLATFORM_TAG'));
  assert.ok(codes(findings, 'error').includes('PYTHON_TAG'));
});

test('filename and pin must agree on name and version', () => {
  const findings = validateManifest(both({
    'darwin-arm64': {
      wheels: [
        wheel('tqdm==4.70.1', 'tqdm-4.69.0-py3-none-any.whl', SHA('d')),
        wheel('idna==3.20', 'certifi-3.20-py3-none-any.whl', SHA('e')),
      ],
    },
  }));
  assert.ok(codes(findings, 'error').includes('FILENAME_VERSION_MISMATCH'));
  assert.ok(codes(findings, 'error').includes('FILENAME_NAME_MISMATCH'));
});

test('underscored wheel filenames match hyphenated pins', () => {
  const findings = validateManifest(both({
    'darwin-arm64': { wheels: [wheel('huggingface-hub==1.32.0', 'huggingface_hub-1.32.0-py3-none-any.whl')] },
  }));
  assert.deepEqual(findings, []);
});

test('duplicates are detected after the pack merge, not before', () => {
  const entry = windows({
    wheels: [
      wheel('torch==2.14.0+cpu', 'torch-2.14.0+cpu-cp310-cp310-win_amd64.whl', SHA('d')),
      wheel('setuptools==84.0.0', 'setuptools-84.0.0-py3-none-any.whl', SHA('e')),
    ],
    packs: {
      // Only torch is declared replaced; setuptools collides incidentally and
      // wheelsForFlavor resolves it to the pack. That is by design, not a bug.
      xpu: {
        wheels: [
          wheel('torch==2.12.0+xpu', 'torch-2.12.0+xpu-cp310-cp310-win_amd64.whl', SHA('f')),
          wheel('setuptools==81.0.0', 'setuptools-81.0.0-py3-none-any.whl', SHA('1')),
        ],
        replaces: ['torch'],
      },
    },
  });
  const merged = mergeWheels(entry, 'xpu').map((item) => item.name);
  assert.deepEqual(merged, ['torch==2.12.0+xpu', 'setuptools==81.0.0']);
  assert.deepEqual(validateManifest(manifestOf({ 'darwin-arm64': darwin(), 'win32-x64': entry })), []);
});

test('two versions of one package in a single list is an error', () => {
  const findings = validateManifest(both({
    'darwin-arm64': {
      wheels: [
        wheel('sympy==1.14.0', 'sympy-1.14.0-py3-none-any.whl', SHA('d')),
        wheel('sympy==1.13.0', 'sympy-1.13.0-py3-none-any.whl', SHA('e')),
      ],
    },
  }));
  assert.deepEqual(codes(findings, 'error'), ['DUPLICATE_PACKAGE']);
});

test('a replaces entry matching no base wheel warns', () => {
  const findings = validateManifest(both({
    'win32-x64': {
      packs: { cuda: { wheels: [wheel('triton==3.7.1', 'triton-3.7.1-cp310-cp310-win_amd64.whl', SHA('d'))], replaces: ['torch'] } },
    },
  }));
  assert.deepEqual(codes(findings, 'warn'), ['DANGLING_REPLACES']);
});

test('distinct wheels that download to one filename collide on disk', () => {
  const findings = validateManifest(both({
    'darwin-arm64': {
      wheels: [
        { name: 'a==1.0', url: 'https://h/one/pkg-1.0-py3-none-any.whl', sha256: SHA('d') },
        { name: 'b==1.0', url: 'https://h/two/pkg-1.0-py3-none-any.whl', sha256: SHA('e') },
      ],
    },
  }));
  assert.ok(codes(findings, 'error').includes('FILENAME_COLLISION'));
});

test('one url may not carry two hashes, but one file may have two hosts', () => {
  const conflict = validateManifest(both({
    'darwin-arm64': { wheels: [{ name: 'jinja2==3.1.6', url: 'https://h/jinja2-3.1.6-py3-none-any.whl', sha256: SHA('d') }] },
    'win32-x64': { wheels: [{ name: 'jinja2==3.1.6', url: 'https://h/jinja2-3.1.6-py3-none-any.whl', sha256: SHA('e') }] },
  }));
  assert.ok(codes(conflict, 'error').includes('HASH_CONFLICT'));

  const mirrored = validateManifest(both({
    'darwin-arm64': { wheels: [{ name: 'jinja2==3.1.6', url: 'https://pypi/jinja2-3.1.6-py3-none-any.whl', sha256: SHA('d') }] },
    'win32-x64': { wheels: [{ name: 'jinja2==3.1.6', url: 'https://pytorch/jinja2-3.1.6-py3-none-any.whl', sha256: SHA('d') }] },
  }));
  assert.deepEqual(mirrored, []);
});

test('requirements pins must be vendored in every target', () => {
  const pins = parseRequirementPins('torch==2.14.0\n# comment\nexceptiongroup==1.3.1\n--extra-index-url https://x\n');
  assert.deepEqual([...pins], [['torch', '2.14.0'], ['exceptiongroup', '1.3.1']]);

  const findings = validateManifest(both({
    'darwin-arm64': { wheels: [wheel('torch==2.14.0', 'torch-2.14.0-cp310-cp310-macosx_14_0_arm64.whl', SHA('d'))] },
    'win32-x64': { wheels: [wheel('torch==2.13.0+cpu', 'torch-2.13.0+cpu-cp310-cp310-win_amd64.whl', SHA('e'))] },
  }), { requirementPins: pins });
  // exceptiongroup is absent from both targets; torch drifts on Windows only.
  assert.equal(codes(findings, 'error').filter((code) => code === 'MISSING_PIN').length, 2);
  assert.ok(codes(findings, 'error').includes('PIN_MISMATCH'));
});

test('a local segment alone is not pin drift', () => {
  const findings = validateManifest(both({
    'win32-x64': { wheels: [wheel('torch==2.14.0+cpu', 'torch-2.14.0+cpu-cp310-cp310-win_amd64.whl', SHA('d'))] },
    'darwin-arm64': { wheels: [wheel('torch==2.14.0', 'torch-2.14.0-cp310-cp310-macosx_14_0_arm64.whl', SHA('e'))] },
  }), { requirementPins: new Map([['torch', '2.14.0']]) });
  assert.deepEqual(codes(findings, 'error'), []);
});

test('a pack trailing the pinned version is a note, not a failure', () => {
  const findings = validateManifest(both({
    'win32-x64': {
      wheels: [wheel('torch==2.14.0+cpu', 'torch-2.14.0+cpu-cp310-cp310-win_amd64.whl', SHA('d'))],
      packs: {
        xpu: { wheels: [wheel('torch==2.12.0+xpu', 'torch-2.12.0+xpu-cp310-cp310-win_amd64.whl', SHA('e'))], replaces: ['torch'] },
      },
    },
    'darwin-arm64': { wheels: [wheel('torch==2.14.0', 'torch-2.14.0-cp310-cp310-macosx_14_0_arm64.whl', SHA('f'))] },
  }), { requirementPins: new Map([['torch', '2.14.0']]) });
  assert.deepEqual(codes(findings, 'error'), []);
  assert.deepEqual(codes(findings, 'info'), ['PACK_VERSION_DRIFT']);
});

test('interpreter shape is checked, exe mismatch only warns', () => {
  const findings = validateManifest(both({
    'darwin-arm64': { interpreter: { url: 'https://h/cpython-3.10.21-aarch64-apple-darwin.tar.gz', sha256: 'nope', exe: 'bin/python' } },
  }));
  assert.ok(codes(findings, 'error').includes('INTERPRETER_SHA'));
  assert.deepEqual(codes(findings, 'warn'), ['INTERPRETER_EXE']);
});

test('url collection dedupes across targets and flavors', () => {
  const urls = collectUrls(both({
    'win32-x64': {
      packs: { cuda: { wheels: [wheel('torch==2.14.0', 'torch-2.14.0-cp310-cp310-win_amd64.whl', SHA('d'))], replaces: ['torch'] } },
    },
  }));
  // Both interpreters, the shared transformers wheel once, and the cuda torch.
  assert.equal(urls.length, 4);
  assert.equal(new Set(urls.map((entry) => entry.url)).size, 4);
});

test('argument parsing', () => {
  const options = parseArgs(['--target', 'win32-x64', '--verify-hashes', '--concurrency', '4', '--strict']);
  assert.deepEqual(options.targets, ['win32-x64']);
  assert.ok(options.verifyHashes && options.checkUrls && options.strict);
  assert.equal(options.concurrency, 4);
  assert.throws(() => parseArgs(['--nope']), /Unknown argument/);
  assert.throws(() => parseArgs(['--concurrency', '0']), /positive integer/);
  assert.throws(() => parseArgs(['--target']), /needs a value/);
});
