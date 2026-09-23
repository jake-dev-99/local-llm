/**
 * Generates `resources/runtime/env-manifest.json`: the hash-pinned
 * interpreter + wheel closure the provisioner installs.
 *
 * Release-time only. PyPI is consulted here, never at install time — the
 * extension installs hermetically with `pip install --no-index`.
 *
 * Usage:
 *   node scripts/generate-python-env-manifest.mjs [--out <path>] [--python 3.10] [--xpu <torch-version> | --no-xpu]
 *
 * Prerequisites: network, `pip` (used as the dependency resolver for the
 * CUDA pack closure). Wheel matching is unit-tested in
 * `generate-python-env-manifest.test.mjs`; thefetch paths run here.
 */

import { execFile } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';

const REQUIREMENTS = 'resources/runtime/requirements.txt';
const DEFAULT_OUT = 'resources/runtime/env-manifest.json';
const PYTORCH_CPU_INDEX = 'https://download.pytorch.org/whl/cpu/';
const STANDALONE_RELEASES = 'https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest';

/** Packages with platform wheels; everything else must resolve pure. */
const PLATFORM_PACKAGES = new Set(['torch', 'safetensors', 'intel-extension-for-pytorch']);

export function parseRequirements(text) {
  const pins = new Map();
  for (const line of text.split('\n')) {
    const cleaned = line.split('#')[0].trim();
    if (!cleaned || cleaned.startsWith('-')) {
      continue;
    }
    const match = /^([A-Za-z0-9_.-]+)==([A-Za-z0-9_.-]+)$/.exec(cleaned);
    if (match) {
      pins.set(match[1].toLowerCase().replace(/_/g, '-'), match[2]);
    }
  }
  return pins;
}

export function wheelPythonTag(filename) {
  // torch-2.14.0-cp312-cp312-macosx_11_0_arm64.whl -> cp312
  const parts = filename.split('-');
  return parts.length >= 3 ? parts[2] : undefined;
}

export function wheelPlatformTags(filename) {
  const parts = filename.split('-');
  const platform = (parts[4] ?? '').replace(/\.whl$/, '');
  return platform.split('.');
}

/**
 * Picks one wheel from candidate `{ filename, url, digests }` entries for a
 * cpXY + platform set. Pure `py3-none-any` wins for non-platform packages;
 * otherwise the first listed platform tag wins, in listed order.
 */
export function pickWheel(candidates, { pythonTag, platformTags, packageName }) {
  const wheels = candidates.filter((entry) => entry.filename.endsWith('.whl'));
  const forPython = wheels.filter((entry) => {
    const tag = wheelPythonTag(entry.filename);
    return tag === pythonTag || tag === 'py3' || tag === 'py2.py3';
  });
  if (!PLATFORM_PACKAGES.has(packageName)) {
    const pure = forPython.find((entry) => wheelPlatformTags(entry.filename).includes('any'));
    if (pure) {
      return pure;
    }
  }
  // Platform tokens match as substrings: macOS publishers bump the version
  // prefix (macosx_11_0, macosx_14_0) while the arch suffix is the stable
  // contract, so specs list arch fragments like 'arm64', not full tags.
  const matches = (tags, token) => token === 'any'
    ? tags.includes('any')
    : tags.some((tag) => tag.includes(token));
  for (const platform of platformTags) {
    const match = forPython.find((entry) => matches(wheelPlatformTags(entry.filename), platform));
    if (match) {
      return match;
    }
  }
  return undefined;
}

export function toManifestWheel(entry, packageName, version) {
  return {
    name: `${packageName}==${version}`,
    url: entry.url,
    sha256: entry.digests.sha256,
  };
}

/**
 * Maps downloaded filenames back to PyPI JSON entries (pip resolves, PyPI
 * provides the otherwise-unguessable file URLs). `knownVersions` pins
 * versions; unknown packages resolve their latest metadata version.
 */
export function filenameVersion(filename) {
  // nvidia_cublas_cu12-12.1.3.1-py3-none-win_amd64.whl -> 12.1.3.1
  return filename.split('-')[1];
}

export async function mapFilenamesToPypi(filenames, knownVersions, fetchJson) {
  const byPackage = new Map();
  for (const filename of filenames) {
    const packageName = filename.split('-')[0].toLowerCase().replace(/_/g, '-');
    if (!byPackage.has(packageName)) {
      byPackage.set(packageName, []);
    }
    byPackage.get(packageName).push(filename);
  }
  const wheels = [];
  for (const [packageName, names] of byPackage) {
    // pip resolved the version; read it off the filename so the mapping hits
    // the exact release even when it is older than latest.
    const version = filenameVersion(names[0]);
    if (knownVersions.has(packageName) && knownVersions.get(packageName) !== version) {
      throw new Error(`Resolver picked ${packageName}==${version} but pins say ${knownVersions.get(packageName)}.`);
    }
    const data = await fetchJson(`https://pypi.org/pypi/${packageName}/${version}/json`);
    const resolvedVersion = data.info.version;
    for (const filename of names) {
      const entry = (data.urls ?? []).find((url) => url.filename === filename);
      if (!entry) {
        throw new Error(`PyPI has no file ${filename} for ${packageName}==${resolvedVersion}.`);
      }
      wheels.push(toManifestWheel(entry, packageName, resolvedVersion));
    }
  }
  return wheels;
}

async function fetchJson(url) {
  const response = await fetch(url, { headers: { Accept: 'application/json' } });
  if (!response.ok) {
    throw new Error(`GET ${url} failed with HTTP ${response.status}.`);
  }
  return response.json();
}

export function requiresPythonOk(requiresPython, pythonMinor) {
  // Enforces floors like ">=3.10" for pure wheels, whose tags match any
  // interpreter. Returns true when there is no floor to check.
  if (!requiresPython) {
    return true;
  }
  const floor = />=\s*(\d+)\.(\d+)/.exec(requiresPython);
  if (!floor) {
    return true;
  }
  const [major, minor] = pythonMinor.split('.').map(Number);
  return major > Number(floor[1]) || (major === Number(floor[1]) && minor >= Number(floor[2]));
}

async function resolvePypiWheel(packageName, version, spec) {
  const data = await fetchJson(`https://pypi.org/pypi/${packageName}/${version}/json`);
  if (!requiresPythonOk(data.info?.requires_python, spec.pythonMinor)) {
    throw new Error(
      `${packageName}==${version} needs Python ${data.info.requires_python}; env is ${spec.pythonMinor}.`,
    );
  }
  const picked = pickWheel(
    (data.urls ?? []).map((url) => ({ ...url })),
    { ...spec, packageName },
  );
  if (!picked) {
    throw new Error(`No ${spec.pythonTag}/${spec.platformTags.join('|')} wheel for ${packageName}==${version}.`);
  }
  return toManifestWheel(picked, packageName, version);
}

function pipDownload(specs, { platform, pythonVersion, dest, extraIndexUrls = [] }) {
  return new Promise((resolve, reject) => {
    execFile(
      'python3',
      [
        '-m', 'pip',
        'download', ...specs,
        '--platform', platform,
        '--python-version', pythonVersion,
        '--implementation', 'cp',
        '--abi', `cp${pythonVersion.replace('.', '')}`,
        '--only-binary', ':all:',
        ...extraIndexUrls.flatMap((url) => ['--extra-index-url', url]),
        '--dest', dest,
      ],
      { timeout: 600_000, maxBuffer: 16 * 1024 * 1024 },
      (error, stdout, stderr) => (error ? reject(Object.assign(error, { stdout, stderr })) : resolve(stdout)),
    );
  });
}

/**
 * Uses pip as the resolver for a pack closure (CUDA torch + nvidia-*, XPU torch),
 * then maps the downloaded filenames back to pinned PyPI entries. Base
 * packages are excluded so packs stay additive.
 */
export async function resolvePackWithPip(
  specs,
  { platform, pythonVersion, baseNames, extraIndexUrls = [], replaces = [] },
  { fetchJson: get, download },
) {
  // A pack may supersede a base wheel with a different build of the same
  // package (CUDA torch over CPU torch): replaced names are kept even though
  // they collide with the base closure.
  const replaced = new Set(replaces.map((name) => name.toLowerCase()));
  const dir = await mkdtemp(path.join(tmpdir(), 'local-llm-pack-'));
  try {
    await download(specs, { platform, pythonVersion, dest: dir, extraIndexUrls });
    const { readdir } = await import('node:fs/promises');
    const filenames = (await readdir(dir)).filter((name) => name.endsWith('.whl'));
    const wheels = await mapFilenamesToPypi(filenames, new Map(), get);
    const kept = wheels.filter((wheel) => {
      const name = wheel.name.split('==')[0].toLowerCase();
      return replaced.has(name) || !baseNames.has(name);
    });
    return { wheels: kept, replaces: [...replaced] };
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

const PYTORCH_XPU_INDEX = 'https://download.pytorch.org/whl/xpu/';


/**
 * Resolves a base closure: every pin plus its transitive dependencies, via
 * pip as the resolver. Per-pin PyPI lookups only fetch the top-level wheels
 * and miss what transformers/accelerate actually import (huggingface-hub,
 * tokenizers, …), which then fails the hermetic `pip install --no-index`
 * with "from versions: none". Downloading the pins together and mapping the
 * result back to pinned URLs keeps the manifest hermetic and complete.
 */
export async function resolveBaseClosure(
  pins,
  { platform, pythonVersion, extraIndexUrls = [], cpuTorch = false },
  { fetchJson: get, download },
) {
  const specs = [...pins.entries()].map(([name, version]) => `${name}==${version}`);
  const dir = await mkdtemp(path.join(tmpdir(), 'local-llm-base-'));
  try {
    await download(specs, { platform, pythonVersion, dest: dir, extraIndexUrls });
    const { readdir, readFile } = await import('node:fs/promises');
    const { createHash } = await import('node:crypto');
    const filenames = (await readdir(dir)).filter((name) => name.endsWith('.whl')).sort();
    const wheels = [];
    const pypiFilenames = [];
    for (const filename of filenames) {
      // The +cpu torch build lives on the PyTorch channel, not PyPI, so it
      // is located there with a content hash like the XPU pack.
      if (cpuTorch && filename.startsWith('torch-') && filename.includes('+cpu')) {
        const bytes = await readFile(path.join(dir, filename));
        const sha256 = createHash('sha256').update(bytes).digest('hex');
        const url = await locateChannelFile(filename, PYTORCH_CPU_INDEX, get);
        const stem = filename.split('-');
        wheels.push({ name: `${stem[0].toLowerCase()}==${stem[1]}`, url, sha256 });
      } else {
        pypiFilenames.push(filename);
      }
    }
    wheels.push(...await mapFilenamesToPypi(pypiFilenames, pins, get));
    wheels.sort((a, b) => a.name.localeCompare(b.name));
    return wheels;
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

/**
 * Resolves the XPU pack: torch's own +xpu build (in-tree torch.xpu, no IPEX
 * needed) plus whatever it drags in, from the PyTorch XPU channel. Returns
 * { wheels, replaces } — the pack supersedes stock torch, so the manifest
 * records the replacement instead of shipping two torch builds.
 */
export async function resolveXpuPack({ torchVersion, pythonVersion }, { fetchJson: get, download }) {
  const dir = await mkdtemp(path.join(tmpdir(), 'local-llm-xpu-'));
  try {
    await download(
      [`torch==${torchVersion}`],
      {
        platform: 'win_amd64',
        pythonVersion,
        dest: dir,
        extraIndexUrls: [PYTORCH_XPU_INDEX],
      },
    );
    const { readdir, readFile } = await import('node:fs/promises');
    const { createHash } = await import('node:crypto');
    const filenames = (await readdir(dir)).filter((name) => name.endsWith('.whl'));
    const wheels = [];
    for (const filename of filenames) {
      const bytes = await readFile(path.join(dir, filename));
      const sha256 = createHash('sha256').update(bytes).digest('hex');
      const url = await locateChannelFile(filename, PYTORCH_XPU_INDEX, get);
      const stem = filename.split('-');
      wheels.push({ name: `${stem[0].toLowerCase().replace(/_/g, '-')}==${stem[1]}`, url, sha256 });
    }
    return { wheels, replaces: ['torch'] };
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

async function locateChannelFile(filename, channelBase, get) {
  // Prefer the channel index href (comparing decoded, since indexes encode
  // '+' as '%2B'); fall back to PyPI for shared deps.
  const stem = filename.split('-')[0].toLowerCase().replace(/_/g, '-');
  for (const project of [stem, stem.replace(/-/g, '_'), stem.replace(/_/g, '-')]) {
    try {
      const response = await fetch(`${channelBase}${project}/`);
      if (!response.ok) {
        continue;
      }
      const html = await response.text();
      const hrefs = [...html.matchAll(/href="([^"]+)"/gi)].map((m) => m[1]);
      const hit = hrefs.find((candidate) => {
        try {
          return decodeURIComponent(candidate.split('#')[0]).endsWith(`/${filename}`);
        } catch {
          return false;
        }
      });
      if (hit) {
        return new URL(hit.split('#')[0], channelBase).href;
      }
    } catch {
      // Fall through to PyPI.
    }
  }
  const data = await get(`https://pypi.org/pypi/${stem}/${filename.split('-')[1]}/json`);
  const entry = (data.urls ?? []).find((url) => url.filename === filename);
  if (!entry) {
    throw new Error(`Neither the channel index nor PyPI lists ${filename}.`);
  }
  return entry.url;
}

async function resolveStandaloneInterpreter(pythonMinor, triple) {
  const release = await fetchJson(STANDALONE_RELEASES);
  const prefix = `cpython-${pythonMinor}.`;
  const asset = (release.assets ?? []).find((entry) =>
    entry.name.startsWith(prefix) &&
    entry.name.includes(triple) &&
    entry.name.endsWith('-install_only.tar.gz') &&
    !entry.name.includes('stripped') &&
    !entry.name.includes('freethreaded'),
  );
  if (!asset) {
    throw new Error(`No install_only interpreter for cpython ${pythonMinor} / ${triple}.`);
  }
  const sumsEntry = (release.assets ?? []).find((entry) => /^sha256sums|SHASUMS/i.test(entry.name));
  let sha256 = asset.digest?.replace(/^sha256:/, '');
  if (!sha256 && sumsEntry) {
    const text = await (await fetch(sumsEntry.browser_download_url)).text();
    sha256 = text.split('\n').find((line) => line.includes(asset.name))?.split(/\s+/)[0];
  }
  if (!sha256) {
    throw new Error(`No SHA-256 for interpreter asset ${asset.name}.`);
  }
  return { url: asset.browser_download_url, sha256, exe: '' };
}

function exeFor(triple) {
  return triple.includes('windows') ? 'python.exe' : 'bin/python3';
}

export async function generateManifest({ pythonMinor = '3.10', requirementsPath = REQUIREMENTS, xpu = undefined } = {}) {
  const pins = parseRequirements(await readFile(requirementsPath, 'utf8'));
  const pip = { fetchJson, download: (specs, options) => pipDownload(specs, options) };

  // Base closures resolve every pin together so transitive dependencies
  // (huggingface-hub, tokenizers, …) ship in the hermetic install.
  const darwinWheels = await resolveBaseClosure(
    pins,
    { platform: 'macosx_14_0_arm64', pythonVersion: pythonMinor },
    pip,
  );

  // Windows base is the CPU torch build plus the shared closure.
  const winWheels = await resolveBaseClosure(
    pins,
    { platform: 'win_amd64', pythonVersion: pythonMinor, extraIndexUrls: [PYTORCH_CPU_INDEX], cpuTorch: true },
    pip,
  );
  const baseNames = new Set(winWheels.map((wheel) => wheel.name.split('==')[0].toLowerCase()));

  const torchPin = pins.get('torch');
  const cudaPack = await resolvePackWithPip(
    [`torch==${torchPin}`],
    { platform: 'win_amd64', pythonVersion: pythonMinor, baseNames, replaces: ['torch'] },
    pip,
  );

  // The XPU pack is torch's own +xpu build from the PyTorch XPU channel
  // (in-tree torch.xpu, no IPEX needed): pass --xpu <torch-version>
  // (e.g. 2.12.0+xpu) or --no-xpu explicitly. There is no silent default.
  let xpuPack = null;
  if (xpu && typeof xpu === 'string') {
    const resolved = await resolveXpuPack(
      { torchVersion: xpu, pythonVersion: pythonMinor },
      pip,
    );
    xpuPack = { wheels: resolved.wheels, replaces: resolved.replaces };
  } else if (xpu !== false) {
    throw new Error('Pass --xpu <torch-version> or --no-xpu explicitly.');
  }

  const darwinInterpreter = await resolveStandaloneInterpreter(pythonMinor, 'aarch64-apple-darwin');
  darwinInterpreter.exe = exeFor('aarch64-apple-darwin');
  const winInterpreter = await resolveStandaloneInterpreter(pythonMinor, 'x86_64-pc-windows-msvc');
  winInterpreter.exe = exeFor('x86_64-pc-windows-msvc');

  return {
    manifestVersion: 1,
    targets: {
      'darwin-arm64': { interpreter: darwinInterpreter, wheels: darwinWheels },
      'win32-x64': {
        interpreter: winInterpreter,
        wheels: winWheels,
        packs: {
          cuda: cudaPack,
          ...(xpuPack ? { xpu: xpuPack } : {}),
        },
      },
    },
  };
}

async function main() {
  const args = process.argv.slice(2);
  const at = (flag) => {
    const index = args.indexOf(flag);
    return index >= 0 ? args[index + 1] : undefined;
  };
  const manifest = await generateManifest({
    pythonMinor: at('--python') ?? '3.10',
    xpu: args.includes('--no-xpu') ? false : (at('--xpu') ?? undefined),
  });
  if (at('--out')) {
    const destination = path.resolve(at('--out'));
    await writeFile(destination, `${JSON.stringify(manifest, null, 2)}\n`);
    console.log(`Wrote ${destination}`);
  } else {
    console.log(JSON.stringify(manifest, null, 2));
  }
}

const invokedAsScript = process.argv[1] !== undefined &&
  import.meta.url === new URL(`file://${process.argv[1]}`).href;
if (invokedAsScript) {
  await main();
}
