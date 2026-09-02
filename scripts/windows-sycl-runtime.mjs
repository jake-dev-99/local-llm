import { readdir } from 'node:fs/promises';
import path from 'node:path';

const LEVEL_ZERO_ADAPTER = /^ur_adapter_level_zero.*\.dll$/i;
const OPENCL_ADAPTER = /^ur_adapter_opencl.*\.dll$/i;
const DEVELOPMENT_KEYS = new Set([
  'CMPLR_ROOT', 'CPATH', 'CMAKE_PREFIX_PATH', 'DNNLROOT', 'INCLUDE', 'LIB',
  'LIBPATH', 'LIBRARY_PATH', 'MKLROOT', 'NLSPATH', 'PKG_CONFIG_PATH', 'TBBROOT',
  'TCM_ROOT', 'UMF_ROOT', 'UR_ADAPTERS_FORCE_LOAD', 'UR_ADAPTERS_SEARCH_PATH',
]);

export function activeOneApiRuntimeDirectories(oneApiRoot, pathDirectories) {
  const active = uniquePaths(pathDirectories.filter((directory) => isAtOrBelow(oneApiRoot, directory)));
  if (active.length === 0) {
    throw new Error(`No active oneAPI runtime directories were found below ${oneApiRoot}.`);
  }
  return active;
}

export async function discoverSyclDynamicResources(activeDirectories, {
  log = () => {},
  readDirectory = readdir,
} = {}) {
  const loaderFiles = await findMatchingFiles(
    activeDirectories,
    (name) => /^ur_loader\.dll$/i.test(name),
    { log, readDirectory },
  );
  if (loaderFiles.length !== 1) {
    throw new Error(
      `Expected exactly one active Unified Runtime loader (ur_loader.dll), found ${loaderFiles.length}. `
      + `Searched: ${activeDirectories.join(', ')}.`,
    );
  }
  const runtimeDirectory = path.dirname(loaderFiles[0]);
  const entries = (await readDirectoryOrSkip(runtimeDirectory, { log, readDirectory }))
    .filter((entry) => entry.isFile())
    .map((entry) => entry.name);
  const adapters = entries.filter((name) => LEVEL_ZERO_ADAPTER.test(name));
  requireRole('Level Zero Unified Runtime adapter', adapters, activeDirectories);
  const openclAdapters = entries.filter((name) => OPENCL_ADAPTER.test(name));
  const proxy = entries.filter((name) => /^ur_win_proxy_loader\.dll$/i.test(name));
  return [path.basename(loaderFiles[0]), ...adapters, ...openclAdapters, ...proxy]
    .sort()
    .map((name) => path.join(runtimeDirectory, name));
}

export function createSyclVerificationEnvironment(baseEnvironment, { staging, systemRoot, oneApiRoot }) {
  if (typeof oneApiRoot !== 'string' || oneApiRoot.length === 0) {
    throw new Error('SYCL staged verification requires the active oneAPI root.');
  }
  const clean = { ...baseEnvironment };
  for (const key of Object.keys(clean)) {
    const normalized = key.toUpperCase();
    if (normalized === 'PATH'
        || normalized.startsWith('ONEAPI_')
        || DEVELOPMENT_KEYS.has(normalized)
        || valueReferencesRoot(clean[key], oneApiRoot)) {
      delete clean[key];
    }
  }
  clean.PATH = [staging, path.win32.join(systemRoot, 'System32'), systemRoot].join(';');
  return clean;
}

export async function verifyStagedSyclBundle({
  staging, systemRoot, oneApiRoot, baseEnvironment, runProcess,
}) {
  const executable = path.win32.join(staging, 'llama-server.exe');
  const env = createSyclVerificationEnvironment(baseEnvironment, { staging, systemRoot, oneApiRoot });
  const result = await runProcess(executable, ['--list-devices'], { cwd: staging, env });
  const output = [result.stdout, result.stderr].filter(Boolean).join('\n').trim();
  if (result.code !== 0) {
    throw new Error(
      `SYCL staged clean-environment launch exited with code ${result.code ?? 'unknown'}`
      + `${output ? `:\n${output}` : '.'}`,
    );
  }
  if (!/(?:^|\s)SYCL0(?:\s|:|$)/m.test(output)) {
    throw new Error(
      `SYCL staged device discovery did not report SYCL0${output ? `:\n${output}` : '.'}`,
    );
  }
}

async function findMatchingFiles(directories, predicate, { log, readDirectory }) {
  const matches = [];
  for (const directory of directories) {
    const entries = await readDirectoryOrSkip(directory, { log, readDirectory });
    for (const entry of entries) {
      if (entry.isFile() && predicate(entry.name)) {
        matches.push(path.join(directory, entry.name));
      }
    }
  }
  return matches;
}

async function readDirectoryOrSkip(directory, { log, readDirectory }) {
  try {
    return await readDirectory(directory, { withFileTypes: true });
  } catch (error) {
    if (error?.code !== 'ENOENT' && error?.code !== 'ENOTDIR') {
      throw error;
    }
    log(`[sycl-package] skipped missing active oneAPI runtime directory: ${directory}`);
    return [];
  }
}

function requireRole(role, matches, searchedDirectories) {
  if (matches.length === 0) {
    throw new Error(`${role} was not found. Searched: ${searchedDirectories.join(', ')}.`);
  }
}

function uniquePaths(paths) {
  const unique = [];
  const seen = new Set();
  for (const candidate of paths) {
    const normalized = normalizeAbsolutePath(candidate);
    if (!seen.has(normalized)) {
      seen.add(normalized);
      unique.push(candidate);
    }
  }
  return unique;
}

function isAtOrBelow(root, candidate) {
  const normalizedRoot = normalizeAbsolutePath(root).replace(/\/+$/, '') || '/';
  const normalizedCandidate = normalizeAbsolutePath(candidate);
  return normalizedCandidate === normalizedRoot
    || normalizedCandidate.startsWith(`${normalizedRoot}/`);
}

function valueReferencesRoot(value, root) {
  if (!root || typeof value !== 'string') {
    return false;
  }
  const entries = value.includes(';')
    ? value.split(';')
    : (/^[A-Za-z]:[\\/]/.test(value) ? [value] : value.split(':'));
  return entries.some((entry) => {
    const candidate = entry.trim().replace(/^(['"])(.*)\1$/, '$2');
    return (path.win32.isAbsolute(candidate) || path.posix.isAbsolute(candidate))
      && isAtOrBelow(root, candidate);
  });
}

function normalizeAbsolutePath(value) {
  const normalized = /^[A-Za-z]:[\\/]/.test(value)
    ? path.win32.normalize(value).replaceAll('\\', '/')
    : path.resolve(value).replaceAll('\\', '/');
  return normalized.length > 1 ? normalized.replace(/\/+$/, '').toLowerCase() : normalized.toLowerCase();
}
