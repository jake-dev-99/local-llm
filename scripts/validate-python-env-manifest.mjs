/**
 * Validates `resources/runtime/env-manifest.json` against the rules the
 * provisioner actually enforces at install time.
 *
 * The manifest is the only thing standing between a release and a user
 * watching a multi-gigabyte download fail on `pip install`. Provisioning is
 * hermetic (`pip install --no-index <every wheel path>`), so a bad manifest
 * cannot self-correct from the network: whatever is pinned here is what runs.
 *
 * Checks mirror consumer code rather than restating PyPI rules in the
 * abstract — `wheelsForFlavor` and `safeWheelFilename` are reimplemented here
 * so a divergence in either shows up as a failing check, not a silent drift.
 *
 * Usage:
 *   node scripts/validate-python-env-manifest.mjs [options]
 *
 *     --manifest <path>      default resources/runtime/env-manifest.json
 *     --requirements <path>  default resources/runtime/requirements.txt
 *                            ('none' skips the pin cross-check)
 *     --target <name>        only this target (repeatable)
 *     --check-urls           HEAD every URL; catches 404s and dead mirrors
 *     --verify-hashes        download every wheel and verify sha256 (slow,
 *                            multi-GB per flavor; implies --check-urls)
 *     --concurrency <n>      parallel network requests (default 8)
 *     --strict               treat warnings as failures
 *     --json                 machine-readable findings on stdout
 *
 * Offline by default: the structural checks are the ones worth running in CI
 * on every change, and they need no network. Exits non-zero on any error.
 */

import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

const DEFAULT_MANIFEST = 'resources/runtime/env-manifest.json';
const DEFAULT_REQUIREMENTS = 'resources/runtime/requirements.txt';

/**
 * Targets the extension knows how to provision. Kept in step with
 * `resolveEnvTarget` in `src/worker/pythonEnvironment.ts`: a target here that
 * the extension cannot resolve is dead weight, and the reverse is a gap.
 */
export const KNOWN_TARGETS = {
  'darwin-arm64': {
    interpreterExe: 'bin/python3',
    platformOk: (tag) => tag === 'any'
      || (tag.startsWith('macosx_') && (tag.endsWith('arm64') || tag.endsWith('universal2'))),
    describePlatform: 'macosx_*_arm64 (or universal2)',
  },
  'win32-x64': {
    interpreterExe: 'python.exe',
    platformOk: (tag) => tag === 'any' || tag === 'win_amd64',
    describePlatform: 'win_amd64',
  },
};

/** Flavors resolvable per target; `cpu` is the base wheel list itself. */
export const FLAVORS = ['cpu', 'cuda', 'xpu'];

const SHA256_PATTERN = /^[0-9a-f]{64}$/i;

/** PEP 503 name normalization. Manifest pins hyphenate, wheels underscore. */
export function normalizeName(name) {
  return String(name).trim().toLowerCase().replace(/[-_.]+/g, '-');
}

/**
 * Splits a `pkg==version` pin. Local version segments are legal and load
 * bearing here (`torch==2.14.0+cpu`), so this is deliberately looser than the
 * generator's requirements parser.
 */
export function parsePin(name) {
  const match = /^([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9._!+]+)$/.exec(String(name ?? '').trim());
  return match ? { name: match[1], version: match[2] } : undefined;
}

/** `2.14.0+cpu` -> `2.14.0`. Local segments are build flavor, not version. */
export function baseVersion(version) {
  return String(version).split('+')[0];
}

/**
 * Parses a wheel filename into its tags, returning undefined for anything pip
 * would reject. The version charset matters: a percent-encoded local segment
 * (`2.14.0%2Bcpu`) parses as a filename but fails pip's PEP 440 version check,
 * which is exactly the failure this validator exists to catch early.
 */
export function parseWheelFilename(filename) {
  if (!filename.endsWith('.whl')) {
    return undefined;
  }
  const parts = filename.slice(0, -4).split('-');
  // name-version[-build]-python-abi-platform
  if (parts.length < 5 || parts.length > 6) {
    return undefined;
  }
  const [name, version] = parts;
  const [pythonTag, abiTag, platformTag] = parts.slice(-3);
  if (!name || !version || !/^[A-Za-z0-9._!+_]+$/.test(version)) {
    return undefined;
  }
  return { name, version, pythonTag, abiTag, platformTag };
}

export function parseCpTag(tag) {
  const match = /^cp(\d)(\d+)$/.exec(String(tag));
  return match ? [Number(match[1]), Number(match[2])] : undefined;
}

/** `cpython-3.10.21+20260901-...` -> `cp310`. */
export function interpreterPythonTag(url) {
  const match = /cpython-(\d+)\.(\d+)\./.exec(decodeURIComponent(String(url)));
  return match ? `cp${match[1]}${match[2]}` : undefined;
}

/**
 * Whether a wheel's python/abi tags admit the target interpreter. `abi3`
 * wheels declare a floor rather than an exact version, so `cp38-abi3` installs
 * on cp310; a plain `cp38` does not.
 */
export function pythonTagAdmits(pythonTag, abiTag, targetTag) {
  const target = parseCpTag(targetTag);
  if (!target) {
    return false;
  }
  const stableAbi = String(abiTag).split('.').includes('abi3');
  for (const tag of String(pythonTag).split('.')) {
    const generic = /^py(\d)(\d*)$/.exec(tag);
    if (generic && Number(generic[1]) === target[0]) {
      if ((generic[2] === '' ? 0 : Number(generic[2])) <= target[1]) {
        return true;
      }
      continue;
    }
    const cp = parseCpTag(tag);
    if (cp && cp[0] === target[0] && (cp[1] === target[1] || (stableAbi && cp[1] <= target[1]))) {
      return true;
    }
  }
  return false;
}

export function platformTagAdmits(platformTag, target) {
  const spec = KNOWN_TARGETS[target];
  return spec ? String(platformTag).split('.').some((tag) => spec.platformOk(tag)) : false;
}

/**
 * The on-disk name the provisioner gives a downloaded wheel. Mirrors
 * `safeWheelFilename` in `src/worker/pythonProvision.ts` — pip is handed this
 * path, so this is the name that has to parse, not the manifest's `name`.
 * Including the percent-decode: an encoded local version segment is a legal url
 * but not a legal wheel filename, so the decode is what keeps the two apart.
 */
export function safeWheelFilename(wheel) {
  const segment = String(wheel.url).split('/').pop()?.split('?')[0] || `${wheel.name}.whl`;
  try {
    const decoded = decodeURIComponent(segment);
    return /[/\\]/.test(decoded) ? segment : decoded;
  } catch {
    return segment;
  }
}

/**
 * The wheel list a flavor installs. Mirrors `wheelsForFlavor` in
 * `src/worker/pythonEnvironment.ts`, including its lowercase-only name
 * comparison: duplicates are detected afterwards with full normalization, so a
 * name the merge fails to collapse surfaces as a duplicate rather than hiding.
 */
export function mergeWheels(entry, flavor) {
  const base = entry.wheels ?? [];
  if (flavor === 'cpu') {
    return base;
  }
  const pack = entry.packs?.[flavor];
  if (!pack) {
    return base;
  }
  const lower = (wheel) => (String(wheel.name).split('==')[0] ?? wheel.name).toLowerCase();
  const packNames = new Set((pack.wheels ?? []).map(lower));
  const replaced = new Set((pack.replaces ?? []).map((name) => String(name).toLowerCase()));
  const kept = base.filter((wheel) => !replaced.has(lower(wheel)) && !packNames.has(lower(wheel)));
  return [...kept, ...(pack.wheels ?? [])];
}

export function parseRequirementPins(text) {
  const pins = new Map();
  for (const line of String(text).split('\n')) {
    const cleaned = line.split('#')[0].trim();
    if (!cleaned || cleaned.startsWith('-')) {
      continue;
    }
    const pin = parsePin(cleaned);
    if (pin) {
      pins.set(normalizeName(pin.name), pin.version);
    }
  }
  return pins;
}

function finding(level, where, code, message) {
  return { level, where, code, message };
}

/**
 * Every structural rule, as a flat list of findings. Pure: no fs, no network,
 * so the whole rule set is unit-testable against hand-built manifests.
 */
export function validateManifest(manifest, options = {}) {
  const findings = [];
  const requirementPins = options.requirementPins;
  const onlyTargets = options.targets;

  if (typeof manifest !== 'object' || manifest === null) {
    return [finding('error', 'manifest', 'SHAPE', 'Manifest is not an object.')];
  }
  if (manifest.manifestVersion !== 1) {
    findings.push(finding('error', 'manifest', 'VERSION',
      `manifestVersion must be 1, found ${JSON.stringify(manifest.manifestVersion)}. ` +
      'parseEnvManifest rejects anything else at install time.'));
  }
  if (typeof manifest.targets !== 'object' || manifest.targets === null) {
    return [...findings, finding('error', 'manifest', 'SHAPE', 'Manifest has no `targets` object.')];
  }

  const targetNames = Object.keys(manifest.targets);
  for (const name of targetNames) {
    if (!KNOWN_TARGETS[name]) {
      findings.push(finding('error', name, 'UNKNOWN_TARGET',
        `Target is not one resolveEnvTarget can produce (${Object.keys(KNOWN_TARGETS).join(', ')}); ` +
        'nothing will ever install it.'));
    }
  }
  for (const name of Object.keys(KNOWN_TARGETS)) {
    if (!targetNames.includes(name)) {
      findings.push(finding('error', name, 'MISSING_TARGET',
        'resolveEnvTarget resolves this platform but the manifest has no entry for it.'));
    }
  }

  // Same URL must always carry the same hash. The converse is legal: jinja2
  // ships byte-identical from PyPI and download.pytorch.org.
  const hashByUrl = new Map();

  for (const [target, entry] of Object.entries(manifest.targets)) {
    if (onlyTargets && !onlyTargets.includes(target)) {
      continue;
    }
    const spec = KNOWN_TARGETS[target];
    if (!spec || typeof entry !== 'object' || entry === null) {
      continue;
    }
    const pythonTag = validateInterpreter(entry, target, spec, findings, hashByUrl);

    if (!Array.isArray(entry.wheels)) {
      findings.push(finding('error', target, 'SHAPE', 'Target has no `wheels` array.'));
      continue;
    }

    validatePacks(entry, target, findings);

    for (const flavor of FLAVORS) {
      if (flavor !== 'cpu' && !entry.packs?.[flavor]) {
        continue;
      }
      const where = `${target}/${flavor}`;
      const wheels = mergeWheels(entry, flavor);
      validateWheelList(wheels, where, target, pythonTag, findings, hashByUrl);
    }

    if (requirementPins) {
      validatePins(entry, target, requirementPins, findings);
    }
  }

  return findings;
}

function validateInterpreter(entry, target, spec, findings, hashByUrl) {
  const interpreter = entry.interpreter;
  if (typeof interpreter !== 'object' || interpreter === null) {
    findings.push(finding('error', target, 'SHAPE', 'Target has no `interpreter` object.'));
    return undefined;
  }
  if (typeof interpreter.url !== 'string' || !interpreter.url) {
    findings.push(finding('error', target, 'INTERPRETER_URL', 'Interpreter has no url.'));
  }
  if (!SHA256_PATTERN.test(String(interpreter.sha256))) {
    findings.push(finding('error', target, 'INTERPRETER_SHA',
      `Interpreter sha256 is not 64 hex characters: ${JSON.stringify(interpreter.sha256)}.`));
  } else if (interpreter.url) {
    recordHash(interpreter.url, interpreter.sha256, target, findings, hashByUrl);
  }
  // The provisioner hardcodes the exe path via interpreterExePath, so a wrong
  // value here misleads readers rather than breaking installs.
  if (interpreter.exe !== spec.interpreterExe) {
    findings.push(finding('warn', target, 'INTERPRETER_EXE',
      `interpreter.exe is ${JSON.stringify(interpreter.exe)} but interpreterExePath uses ` +
      `${JSON.stringify(spec.interpreterExe)} for this target.`));
  }
  const pythonTag = interpreterPythonTag(interpreter.url ?? '');
  if (!pythonTag) {
    findings.push(finding('error', target, 'INTERPRETER_VERSION',
      `Cannot read a CPython version out of the interpreter url, so wheel ` +
      `compatibility cannot be checked: ${interpreter.url}`));
  }
  return pythonTag;
}

function validatePacks(entry, target, findings) {
  const packs = entry.packs;
  if (packs === undefined) {
    return;
  }
  if (typeof packs !== 'object' || packs === null) {
    findings.push(finding('error', target, 'SHAPE', '`packs` is not an object.'));
    return;
  }
  const baseNames = new Set((entry.wheels ?? []).map((wheel) => normalizeName(String(wheel.name).split('==')[0])));
  for (const [flavor, pack] of Object.entries(packs)) {
    const where = `${target}/${flavor}`;
    if (!FLAVORS.includes(flavor) || flavor === 'cpu') {
      findings.push(finding('error', where, 'UNKNOWN_FLAVOR',
        `Pack flavor must be one of ${FLAVORS.filter((name) => name !== 'cpu').join(', ')}.`));
      continue;
    }
    if (!Array.isArray(pack?.wheels)) {
      findings.push(finding('error', where, 'SHAPE', 'Pack has no `wheels` array.'));
      continue;
    }
    for (const name of pack.replaces ?? []) {
      if (!baseNames.has(normalizeName(name))) {
        findings.push(finding('warn', where, 'DANGLING_REPLACES',
          `Pack replaces "${name}", which is not in the base wheel list; the entry does nothing.`));
      }
    }
  }
}

function validateWheelList(wheels, where, target, pythonTag, findings, hashByUrl) {
  const byName = new Map();
  const byDiskName = new Map();

  for (const wheel of wheels) {
    if (typeof wheel !== 'object' || wheel === null) {
      findings.push(finding('error', where, 'SHAPE', 'Wheel entry is not an object.'));
      continue;
    }
    const pin = parsePin(wheel.name);
    if (!pin) {
      findings.push(finding('error', where, 'WHEEL_NAME',
        `Wheel name is not a pkg==version pin: ${JSON.stringify(wheel.name)}.`));
      continue;
    }
    const label = `${pin.name}==${pin.version}`;
    const normalized = normalizeName(pin.name);

    if (!SHA256_PATTERN.test(String(wheel.sha256))) {
      findings.push(finding('error', where, 'WHEEL_SHA',
        `${label}: sha256 is not 64 hex characters: ${JSON.stringify(wheel.sha256)}.`));
    }
    if (typeof wheel.url !== 'string' || !wheel.url) {
      findings.push(finding('error', where, 'WHEEL_URL', `${label}: missing url.`));
      continue;
    }
    recordHash(wheel.url, wheel.sha256, where, findings, hashByUrl);

    // Duplicates are the documented install hazard: every path goes into one
    // `pip install --no-index` call, so two versions of a package install over
    // each other in argument order.
    if (byName.has(normalized)) {
      findings.push(finding('error', where, 'DUPLICATE_PACKAGE',
        `${pin.name} appears twice after the pack merge (${byName.get(normalized)} and ${pin.version}); ` +
        'pip installs both in order and the winner is whichever lands last.'));
    } else {
      byName.set(normalized, pin.version);
    }

    const diskName = safeWheelFilename(wheel);
    const previous = byDiskName.get(diskName);
    if (previous && previous.sha256 !== wheel.sha256) {
      findings.push(finding('error', where, 'FILENAME_COLLISION',
        `${label} and ${previous.name} both download to "${diskName}"; the second overwrites the first.`));
    } else {
      byDiskName.set(diskName, { name: label, sha256: wheel.sha256 });
    }

    const parsed = parseWheelFilename(diskName);
    if (!parsed) {
      findings.push(finding('error', where, 'BAD_WHEEL_FILENAME',
        `${label}: the url's last path segment lands on disk as "${diskName}", which is not a ` +
        `filename pip accepts (pip 26.2.1 raises InvalidWheelFilename on it; older pip derives a ` +
        `version that disagrees with the archive's METADATA). safeWheelFilename() percent-decodes ` +
        `the segment but changes nothing else, and pip parses the name before reading the archive, ` +
        `so this fails the install after the whole download.`));
      continue;
    }
    if (normalizeName(parsed.name) !== normalized) {
      findings.push(finding('error', where, 'FILENAME_NAME_MISMATCH',
        `${label} points at a wheel for "${parsed.name}".`));
    }
    if (parsed.version !== pin.version) {
      findings.push(finding('error', where, 'FILENAME_VERSION_MISMATCH',
        `${label} points at version ${parsed.version}.`));
    }
    if (pythonTag && !pythonTagAdmits(parsed.pythonTag, parsed.abiTag, pythonTag)) {
      findings.push(finding('error', where, 'PYTHON_TAG',
        `${label}: wheel is tagged ${parsed.pythonTag}-${parsed.abiTag}, which does not install on ` +
        `the pinned interpreter (${pythonTag}).`));
    }
    if (!platformTagAdmits(parsed.platformTag, target)) {
      findings.push(finding('error', where, 'PLATFORM_TAG',
        `${label}: wheel is tagged ${parsed.platformTag}, but ${target} needs ` +
        `${KNOWN_TARGETS[target].describePlatform} or any.`));
    }
  }
}

function validatePins(entry, target, requirementPins, findings) {
  const base = new Map();
  for (const wheel of entry.wheels ?? []) {
    const pin = parsePin(wheel.name);
    if (pin) {
      base.set(normalizeName(pin.name), pin.version);
    }
  }
  for (const [name, version] of requirementPins) {
    const found = base.get(name);
    if (found === undefined) {
      findings.push(finding('error', target, 'MISSING_PIN',
        `requirements.txt pins ${name}==${version} but no wheel for it is vendored. ` +
        'Marker-gated deps drop silently when the generator runs on a newer host interpreter.'));
    } else if (baseVersion(found) !== baseVersion(version)) {
      findings.push(finding('error', target, 'PIN_MISMATCH',
        `requirements.txt pins ${name}==${version}, manifest vendors ${found}.`));
    }
  }
  // A pack shipping a different base version than the pin is a deliberate
  // choice (xpu torch trails stock torch), but it should never be a surprise.
  for (const [flavor, pack] of Object.entries(entry.packs ?? {})) {
    for (const wheel of pack?.wheels ?? []) {
      const pin = parsePin(wheel.name);
      const pinned = pin && requirementPins.get(normalizeName(pin.name));
      if (pinned && baseVersion(pinned) !== baseVersion(pin.version)) {
        findings.push(finding('info', `${target}/${flavor}`, 'PACK_VERSION_DRIFT',
          `Pack ships ${pin.name}==${pin.version} against the pinned ${pinned}.`));
      }
    }
  }
}

function recordHash(url, sha256, where, findings, hashByUrl) {
  const known = hashByUrl.get(url);
  if (known && String(known.sha256).toLowerCase() !== String(sha256).toLowerCase()) {
    findings.push(finding('error', where, 'HASH_CONFLICT',
      `${url} is pinned to ${sha256} here and ${known.sha256} in ${known.where}.`));
    return;
  }
  if (!known) {
    hashByUrl.set(url, { sha256, where });
  }
}

/** Every distinct URL the manifest can download, for the network passes. */
export function collectUrls(manifest, targets) {
  const urls = new Map();
  const add = (url, label, where) => {
    if (typeof url === 'string' && url && !urls.has(url)) {
      urls.set(url, { url, label, where });
    }
  };
  for (const [target, entry] of Object.entries(manifest?.targets ?? {})) {
    if (targets && !targets.includes(target)) {
      continue;
    }
    add(entry?.interpreter?.url, 'interpreter', target);
    for (const flavor of FLAVORS) {
      if (flavor !== 'cpu' && !entry?.packs?.[flavor]) {
        continue;
      }
      for (const wheel of mergeWheels(entry ?? {}, flavor)) {
        add(wheel?.url, wheel?.name, `${target}/${flavor}`);
      }
    }
  }
  return [...urls.values()];
}

async function mapWithConcurrency(items, limit, worker) {
  const results = [];
  let cursor = 0;
  const runners = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (cursor < items.length) {
      const index = cursor++;
      results[index] = await worker(items[index], index);
    }
  });
  await Promise.all(runners);
  return results;
}

async function checkUrl(entry) {
  try {
    let response = await fetch(entry.url, { method: 'HEAD', redirect: 'follow' });
    if (response.status === 405 || response.status === 501) {
      // Some CDNs refuse HEAD; a one-byte range is the cheap fallback.
      response = await fetch(entry.url, { headers: { Range: 'bytes=0-0' }, redirect: 'follow' });
      await response.body?.cancel();
    }
    return response.ok
      ? undefined
      : finding('error', entry.where, 'URL_UNREACHABLE', `${entry.label}: HTTP ${response.status} for ${entry.url}`);
  } catch (error) {
    return finding('error', entry.where, 'URL_UNREACHABLE', `${entry.label}: ${error.message} (${entry.url})`);
  }
}

/** Finds the hash the manifest pins for a url, whichever list it came from. */
function expectedHash(manifest, url) {
  for (const entry of Object.values(manifest?.targets ?? {})) {
    if (entry?.interpreter?.url === url) {
      return entry.interpreter.sha256;
    }
    for (const flavor of FLAVORS) {
      for (const wheel of mergeWheels(entry ?? {}, flavor)) {
        if (wheel?.url === url) {
          return wheel.sha256;
        }
      }
    }
  }
  return undefined;
}

async function verifyHash(entry, manifest) {
  const expected = expectedHash(manifest, entry.url);
  try {
    const response = await fetch(entry.url, { redirect: 'follow' });
    if (!response.ok) {
      return finding('error', entry.where, 'URL_UNREACHABLE', `${entry.label}: HTTP ${response.status} for ${entry.url}`);
    }
    const hash = createHash('sha256');
    for await (const chunk of response.body) {
      hash.update(chunk);
    }
    const actual = hash.digest('hex');
    return actual.toLowerCase() === String(expected).toLowerCase()
      ? undefined
      : finding('error', entry.where, 'HASH_MISMATCH',
        `${entry.label}: manifest pins ${expected}, download hashes to ${actual}.`);
  } catch (error) {
    return finding('error', entry.where, 'HASH_MISMATCH', `${entry.label}: ${error.message} (${entry.url})`);
  }
}

export function parseArgs(argv) {
  const options = {
    manifest: DEFAULT_MANIFEST,
    requirements: DEFAULT_REQUIREMENTS,
    targets: undefined,
    checkUrls: false,
    verifyHashes: false,
    concurrency: 8,
    strict: false,
    json: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = () => {
      const value = argv[index + 1];
      if (value === undefined) {
        throw new Error(`${arg} needs a value.`);
      }
      index += 1;
      return value;
    };
    switch (arg) {
      case '--manifest': options.manifest = next(); break;
      case '--requirements': options.requirements = next(); break;
      case '--target': (options.targets ??= []).push(next()); break;
      case '--check-urls': options.checkUrls = true; break;
      case '--verify-hashes': options.verifyHashes = true; options.checkUrls = true; break;
      case '--concurrency': options.concurrency = Number(next()); break;
      case '--strict': options.strict = true; break;
      case '--json': options.json = true; break;
      case '--help': case '-h': options.help = true; break;
      default: throw new Error(`Unknown argument: ${arg}`);
    }
  }
  if (!Number.isInteger(options.concurrency) || options.concurrency < 1) {
    throw new Error('--concurrency must be a positive integer.');
  }
  return options;
}

const HELP = `Validate the hash-pinned Python environment manifest.

  node scripts/validate-python-env-manifest.mjs [options]

  --manifest <path>      manifest to validate (default ${DEFAULT_MANIFEST})
  --requirements <path>  pins to cross-check ('none' to skip)
  --target <name>        restrict to one target (repeatable)
  --check-urls           HEAD every url
  --verify-hashes        download every file and verify sha256 (slow)
  --concurrency <n>      parallel requests (default 8)
  --strict               treat warnings as failures
  --json                 findings as JSON
`;

const LEVEL_ORDER = { error: 0, warn: 1, info: 2 };
const LEVEL_LABEL = { error: 'ERROR', warn: 'WARN ', info: 'INFO ' };

function report(findings, options) {
  if (options.json) {
    process.stdout.write(`${JSON.stringify(findings, null, 2)}\n`);
    return;
  }
  const sorted = [...findings].sort((a, b) =>
    LEVEL_ORDER[a.level] - LEVEL_ORDER[b.level] || a.where.localeCompare(b.where));
  for (const item of sorted) {
    process.stdout.write(`${LEVEL_LABEL[item.level]} [${item.where}] ${item.code}: ${item.message}\n`);
  }
  const counts = { error: 0, warn: 0, info: 0 };
  for (const item of findings) {
    counts[item.level] += 1;
  }
  process.stdout.write(
    `\n${counts.error} error(s), ${counts.warn} warning(s), ${counts.info} note(s).\n`,
  );
}

export async function main(argv = process.argv.slice(2)) {
  let options;
  try {
    options = parseArgs(argv);
  } catch (error) {
    process.stderr.write(`${error.message}\n\n${HELP}`);
    return 2;
  }
  if (options.help) {
    process.stdout.write(HELP);
    return 0;
  }

  const manifestPath = path.resolve(options.manifest);
  let manifest;
  try {
    manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  } catch (error) {
    process.stderr.write(`Cannot read manifest ${manifestPath}: ${error.message}\n`);
    return 2;
  }

  let requirementPins;
  if (options.requirements !== 'none') {
    try {
      requirementPins = parseRequirementPins(await readFile(path.resolve(options.requirements), 'utf8'));
    } catch (error) {
      process.stderr.write(`Cannot read requirements ${options.requirements}: ${error.message}\n`);
      return 2;
    }
  }

  const findings = validateManifest(manifest, { requirementPins, targets: options.targets });

  if (options.checkUrls || options.verifyHashes) {
    const urls = collectUrls(manifest, options.targets);
    const action = options.verifyHashes ? 'Verifying' : 'Checking';
    process.stderr.write(`${action} ${urls.length} url(s) with concurrency ${options.concurrency}...\n`);
    const results = await mapWithConcurrency(urls, options.concurrency, (entry) =>
      options.verifyHashes ? verifyHash(entry, manifest) : checkUrl(entry));
    findings.push(...results.filter(Boolean));
  }

  report(findings, options);

  const failed = findings.some((item) =>
    item.level === 'error' || (options.strict && item.level === 'warn'));
  return failed ? 1 : 0;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  process.exitCode = await main();
}
