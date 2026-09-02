import { randomUUID } from 'node:crypto';
import { cp, mkdir, mkdtemp, readFile, readdir, rename, rm, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';

import { parseWorkerManifest, replaceWorkerBundle, sha256File } from '../src/worker/workerManifest.ts';
import {
  activeOneApiRuntimeDirectories,
  discoverSyclDynamicResources,
  verifyStagedSyclBundle,
} from './windows-sycl-runtime.mjs';

export const REQUIRED_LLAMA_BACKEND_MODULES = [
  'ggml-cpu.dll',
  'ggml-sycl.dll',
];

const SYSTEM_DEPENDENCIES = new Set([
  'advapi32.dll',
  'crypt32.dll',
  'dbghelp.dll',
  'kernel32.dll',
  'opencl.dll',
  'shell32.dll',
  'shlwapi.dll',
  'wintrust.dll',
  'ws2_32.dll',
  'ze_loader.dll',
]);

export function parseDumpbinDependents(output) {
  const dependencies = [];
  for (const line of output.split(/\r?\n/)) {
    const match = /^\s+([^\s]+\.dll)\s*$/i.exec(line);
    if (match) {
      dependencies.push(match[1]);
    }
  }
  return dependencies;
}

export async function collectDependencyClosure(roots, options) {
  const queue = roots.map((absoluteFile) => ({ absoluteFile, importedBy: undefined }));
  const files = [];
  const seen = new Map();

  while (queue.length > 0) {
    const { absoluteFile } = queue.shift();
    const key = path.basename(absoluteFile).toLowerCase();
    const previous = seen.get(key);
    if (previous && normalizedSourcePath(previous) !== normalizedSourcePath(absoluteFile)) {
      throw new Error(
        `Dependency filename collision for ${path.basename(absoluteFile)}: ${previous} and ${absoluteFile}.`,
      );
    }
    if (previous) {
      continue;
    }
    seen.set(key, absoluteFile);
    files.push(absoluteFile);

    for (const dllName of await options.imports(absoluteFile)) {
      let resolved;
      try {
        resolved = await options.resolve(dllName);
      } catch (error) {
        throw new Error(`${error.message} Imported by ${absoluteFile}.`, { cause: error });
      }
      if (options.isSystemDependency(dllName, resolved)) {
        continue;
      }
      if (!resolved) {
        throw new Error(
          `Required dependency ${dllName} imported by ${absoluteFile} could not be resolved. `
          + `Searched: ${options.searchDirectories.join(', ')}.`,
        );
      }
      queue.push({ absoluteFile: resolved, importedBy: absoluteFile });
    }
  }

  return files;
}

export async function prepareWindowsSyclBundle(options) {
  if (!options.environment || typeof options.environment !== 'object' || Array.isArray(options.environment)) {
    throw new Error('Windows SYCL bundle assembly requires an environment record.');
  }
  if (typeof options.runProcess !== 'function') {
    throw new Error('Windows SYCL bundle assembly requires a runProcess function.');
  }

  const executable = path.join(options.buildOutput, 'llama-server.exe');
  await requireFile(executable, 'Built SYCL executable llama-server.exe was not found');

  const backendModules = [];
  for (const name of REQUIRED_LLAMA_BACKEND_MODULES) {
    const absolute = await findNamedFile(options.buildOutput, name);
    if (!absolute) {
      throw new Error(`Required llama.cpp backend module ${name} was not found in ${options.buildOutput}.`);
    }
    backendModules.push(absolute);
  }

  const pathDirectories = options.pathDirectories ?? splitSearchPath(options.oneApiPath ?? process.env.PATH);
  const systemRoot = options.systemRoot ?? process.env.SystemRoot ?? 'C:\\Windows';
  const system32 = path.join(systemRoot, 'System32');
  const vcRuntime = path.join(options.vcToolsRedistDir, 'x64', 'Microsoft.VC143.CRT');
  const pathSystemDirectories = pathDirectories.filter((directory) => isAtOrBelow(systemRoot, directory));
  const activeDirectories = activeOneApiRuntimeDirectories(options.oneApiRoot, pathDirectories);
  const systemDirectories = uniqueSourcePaths([...pathSystemDirectories, system32]);
  const searchDirectories = [
    options.buildOutput,
    ...activeDirectories,
    vcRuntime,
    ...systemDirectories,
  ];
  const excludedSystemDependencies = new Set();
  const closureOptions = {
    imports: async (absoluteFile) => (
      /\.(?:exe|dll)$/i.test(absoluteFile)
        ? parseDumpbinDependents(await options.runDumpbin(absoluteFile))
        : []
    ),
    resolve: resolveFromTiers([
      { label: 'build output', directories: [options.buildOutput] },
      { label: 'active oneAPI', directories: activeDirectories, rejectAmbiguous: true },
      { label: 'VC redistributable', directories: [vcRuntime] },
      { label: 'Windows system', directories: systemDirectories },
    ]),
    isSystemDependency: (dllName, absoluteFile) => {
      const excluded = isSystemDependencyName(dllName)
        || (absoluteFile ? systemDirectories.some((directory) => isAtOrBelow(directory, absoluteFile)) : false);
      if (excluded) {
        excludedSystemDependencies.add(dllName);
      }
      return excluded;
    },
    searchDirectories,
  };
  const baseRoots = [...backendModules, executable];
  await collectDependencyClosure(baseRoots, closureOptions);
  const log = options.log ?? console.log;
  const dynamicRuntimeDlls = await discoverSyclDynamicResources(activeDirectories, { log });
  const bundleSources = await collectDependencyClosure(
    [...baseRoots, ...dynamicRuntimeDlls],
    closureOptions,
  );
  const licenses = await collectControllingLicenses(
    options.oneApiRoot,
    classifyOneApiFiles(options.oneApiRoot, bundleSources),
  );
  log(`[sycl-package] oneAPI root: ${options.oneApiRoot}`);
  for (const directory of activeDirectories) {
    log(`[sycl-package] active oneAPI runtime directory: ${directory}`);
  }
  for (const file of dynamicRuntimeDlls) {
    log(`[sycl-package] dynamic SYCL runtime DLL: ${file}`);
  }
  for (const file of bundleSources) {
    log(`[sycl-package] resolved bundle source: ${file}`);
  }
  for (const dependency of [...excludedSystemDependencies].sort()) {
    log(`[sycl-package] excluded system dependency: ${dependency}`);
  }

  const parent = path.dirname(options.destination);
  await mkdir(parent, { recursive: true });
  const staging = await mkdtemp(path.join(parent, '.sycl-stage-'));
  try {
    const copiedNames = new Map();
    for (const source of bundleSources) {
      const name = path.basename(source);
      const key = name.toLowerCase();
      const previous = copiedNames.get(key);
      if (previous && path.resolve(previous) !== path.resolve(source)) {
        throw new Error(`Two SYCL dependency files resolve to the same bundle name ${name}.`);
      }
      if (!previous) {
        copiedNames.set(key, source);
        await cp(source, path.join(staging, name));
      }
    }
    for (const license of licenses) {
      const output = path.join(staging, 'licenses', license.component, ...license.relative.split('/'));
      await mkdir(path.dirname(output), { recursive: true });
      await cp(license.absolute, output);
    }

    await verifyStagedSyclBundle({
      staging,
      systemRoot,
      oneApiRoot: options.oneApiRoot,
      baseEnvironment: options.environment,
      runProcess: options.runProcess,
    });
    const bundle = await describeStagedBundle(options.root, options.destination, staging);
    return {
      bundleName: 'sycl',
      bundle,
      destination: options.destination,
      staging,
    };
  } catch (error) {
    await rm(staging, { recursive: true, force: true });
    throw error;
  }
}

export async function publishWindowsWorkerBuilds({
  root, target, builds, fileOperations, log = console.log,
}) {
  if (!Array.isArray(builds) || builds.length === 0) {
    throw new Error('Windows worker publication requires at least one completed build.');
  }
  const prepared = [];
  try {
    for (const build of builds) {
      if (build.backend === 'cpu') {
        prepared.push(await prepareWindowsCpuBundle({
          root,
          binary: build.binary,
          destination: build.destination,
        }));
      } else if (build.backend === 'sycl') {
        prepared.push(await prepareWindowsSyclBundle({
          ...build.bundleOptions,
          root,
          buildOutput: path.dirname(build.binary),
          destination: build.destination,
        }));
      } else {
        throw new Error(`Unsupported Windows worker backend for publication: ${build.backend}.`);
      }
    }
    await publishPreparedWindowsBundles({
      root,
      target,
      prepared,
      fileOperations,
      log,
    });
    return Object.fromEntries(prepared.map(({ bundleName, bundle }) => [bundleName, bundle]));
  } finally {
    await Promise.all(prepared.map(({ staging }) => rm(staging, { recursive: true, force: true })));
  }
}

async function prepareWindowsCpuBundle({ root, binary, destination }) {
  await requireFile(binary, 'Built CPU executable llama-server.exe was not found');
  await mkdir(path.dirname(destination), { recursive: true });
  const staging = await mkdtemp(path.join(path.dirname(destination), '.cpu-stage-'));
  try {
    await cp(binary, path.join(staging, 'llama-server.exe'));
    return {
      bundleName: 'cpu',
      bundle: await describeStagedBundle(root, destination, staging),
      destination,
      staging,
    };
  } catch (error) {
    await rm(staging, { recursive: true, force: true });
    throw error;
  }
}

async function publishPreparedWindowsBundles({
  root, target, prepared, fileOperations, log,
}) {
  const fs = {
    mkdir,
    mkdtemp,
    readFile,
    rename,
    rm,
    stat,
    writeFile,
    ...fileOperations,
  };
  const manifestPath = path.join(root, 'resources', 'workers', 'manifest.json');
  const current = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
  const prospective = createUpdatedWorkerManifest(
    current,
    target,
    Object.fromEntries(prepared.map(({ bundleName, bundle }) => [bundleName, bundle])),
  );
  const serialized = `${JSON.stringify(prospective, null, 2)}\n`;
  validateProspectiveManifest(JSON.parse(serialized));

  const manifestParent = path.dirname(manifestPath);
  await fs.mkdir(manifestParent, { recursive: true });
  const manifestStageDirectory = await fs.mkdtemp(path.join(manifestParent, '.manifest-stage-'));
  const stagedManifest = path.join(manifestStageDirectory, 'manifest.json');
  const directoryStates = prepared.map((publication) => ({
    ...publication,
    backup: uniqueBackupPath(publication.destination),
    backedUp: false,
    published: false,
  }));
  const manifestState = {
    backup: uniqueBackupPath(manifestPath),
    backedUp: false,
    published: false,
  };
  const legacyWorker = path.join(root, 'resources', 'workers', 'win32-x64', 'llama-server.exe');
  const legacyState = {
    backup: uniqueBackupPath(legacyWorker),
    retired: false,
  };
  let committed = false;
  try {
    await fs.writeFile(stagedManifest, serialized);
    const stagedContents = await fs.readFile(stagedManifest, 'utf8');
    if (stagedContents !== serialized) {
      throw new Error('Staged worker manifest did not preserve the prospective manifest bytes.');
    }
    validateProspectiveManifest(JSON.parse(stagedContents));

    try {
      for (const state of directoryStates) {
        if (await pathExists(state.destination, fs)) {
          await fs.rename(state.destination, state.backup);
          state.backedUp = true;
        }
        await fs.rename(state.staging, state.destination);
        state.published = true;
      }

      if (current.manifestVersion !== 2 && prospective.manifestVersion === 2
          && await pathExists(legacyWorker, fs)) {
        await fs.rename(legacyWorker, legacyState.backup);
        legacyState.retired = true;
      }

      await fs.rename(manifestPath, manifestState.backup);
      manifestState.backedUp = true;
      await fs.rename(stagedManifest, manifestPath);
      manifestState.published = true;
      committed = true;
    } catch (error) {
      const rollbackErrors = await rollbackPublication({
        fs,
        directoryStates,
        manifestPath,
        manifestState,
        legacyWorker,
        legacyState,
      });
      if (rollbackErrors.length > 0) {
        throw new Error(
          `${error.message} Rollback also failed: ${rollbackErrors.map((failure) => failure.message).join('; ')}`,
          { cause: error },
        );
      }
      throw error;
    }
  } finally {
    await fs.rm(manifestStageDirectory, { recursive: true, force: true });
    if (committed) {
      const cleanup = [
        ...directoryStates.filter((state) => state.backedUp).map((state) => state.backup),
        ...(manifestState.backedUp ? [manifestState.backup] : []),
        ...(legacyState.retired ? [legacyState.backup] : []),
      ];
      const results = await Promise.allSettled(
        cleanup.map((backup) => fs.rm(backup, { recursive: true, force: true })),
      );
      for (const result of results) {
        if (result.status === 'rejected') {
          log(`[worker-publish] Backup cleanup failed after successful publication: ${result.reason}`);
        }
      }
    }
  }
}

async function rollbackPublication({
  fs, directoryStates, manifestPath, manifestState, legacyWorker, legacyState,
}) {
  const errors = [];
  await attemptRollback(errors, async () => {
    if (manifestState.published) {
      await fs.rm(manifestPath, { force: true });
    }
    if (manifestState.backedUp) {
      await fs.rename(manifestState.backup, manifestPath);
    }
  });
  await attemptRollback(errors, async () => {
    if (legacyState.retired) {
      await fs.rename(legacyState.backup, legacyWorker);
    }
  });
  for (const state of [...directoryStates].reverse()) {
    await attemptRollback(errors, async () => {
      if (state.published) {
        await fs.rm(state.destination, { recursive: true, force: true });
      }
      if (state.backedUp) {
        await fs.rename(state.backup, state.destination);
      }
    });
  }
  return errors;
}

async function attemptRollback(errors, operation) {
  try {
    await operation();
  } catch (error) {
    errors.push(error);
  }
}

function updatedManifestForBundles(current, target, prepared) {
  let updated = JSON.parse(JSON.stringify(current));
  const ordered = [...prepared].sort((left, right) => (
    left.bundleName === right.bundleName ? 0 : left.bundleName === 'cpu' ? -1 : 1
  ));
  for (const { bundleName, bundle } of ordered) {
    updated = updated.manifestVersion === 2
      ? replaceWorkerBundle(updated, target, bundleName, bundle)
      : updateLegacyManifest(updated, target, bundleName, bundle);
  }
  validateProspectiveManifest(updated);
  return updated;
}

export function createUpdatedWorkerManifest(current, target, bundles) {
  return updatedManifestForBundles(
    current,
    target,
    Object.entries(bundles).map(([bundleName, bundle]) => ({ bundleName, bundle })),
  );
}

function validateProspectiveManifest(manifest) {
  if (manifest.manifestVersion === 2) {
    parseWorkerManifest(manifest);
    return;
  }
  const workers = legacyWorkers(manifest);
  for (const target of Object.keys(workers)) {
    legacyWorkerBundle(workers, target);
  }
}

function uniqueBackupPath(target) {
  return path.join(path.dirname(target), `.${path.basename(target)}.backup-${randomUUID()}`);
}

async function pathExists(value, fs) {
  try {
    await fs.stat(value);
    return true;
  } catch (error) {
    if (error?.code === 'ENOENT') {
      return false;
    }
    throw error;
  }
}

function updateLegacyManifest(current, target, bundleName, bundle) {
  const workers = legacyWorkers(current);
  if (bundleName === 'cpu') {
    workers[target] = legacyEntryFromBundle(bundle);
    return current;
  }
  if (target !== 'win32-x64' || bundleName !== 'sycl') {
    throw new Error(`Cannot publish ${target} ${bundleName} into the legacy worker manifest.`);
  }

  const darwin = legacyWorkerBundle(workers, 'darwin-arm64');
  const cpu = legacyWorkerBundle(workers, 'win32-x64');
  if (!cpu.executable.startsWith('resources/workers/win32-x64/cpu/')) {
    throw new Error(
      'The legacy Windows CPU worker has not been published into its isolated bundle. Build with --backend all or publish cpu before sycl.',
    );
  }
  return parseWorkerManifest({
    manifestVersion: 2,
    llamaCppCommit: current.llamaCppCommit,
    llamaCppBuild: current.llamaCppBuild,
    platforms: {
      'darwin-arm64': {
        modes: {
          auto: { bundle: 'default', backend: 'metal' },
          cpu: { bundle: 'default', backend: 'cpu' },
        },
        bundles: { default: darwin },
      },
      'win32-x64': {
        modes: {
          auto: { bundle: 'sycl', backend: 'sycl' },
          cpu: { bundle: 'cpu', backend: 'cpu' },
        },
        bundles: { sycl: bundle, cpu },
      },
    },
  });
}

function legacyWorkers(manifest) {
  if (!manifest || typeof manifest !== 'object' || Array.isArray(manifest) ||
      !manifest.workers || typeof manifest.workers !== 'object' || Array.isArray(manifest.workers)) {
    throw new Error('Legacy worker manifest must contain a workers object.');
  }
  return manifest.workers;
}

function legacyEntryFromBundle(bundle) {
  const executable = bundle?.executable;
  const file = Array.isArray(bundle?.files)
    ? bundle.files.find((candidate) => candidate?.path === executable)
    : undefined;
  if (typeof executable !== 'string' || !file || typeof file.sha256 !== 'string') {
    throw new Error('Published worker bundle must hash its executable.');
  }
  return { path: executable, sha256: file.sha256 };
}

function legacyWorkerBundle(workers, target) {
  const entry = workers[target];
  if (!entry || typeof entry.path !== 'string' || typeof entry.sha256 !== 'string') {
    throw new Error(`Legacy worker manifest has no valid ${target} entry.`);
  }
  return {
    executable: entry.path,
    files: [{ path: entry.path, sha256: entry.sha256 }],
  };
}

async function collectControllingLicenses(oneApiRoot, companionFiles) {
  const byComponent = new Map();
  for (const { component } of companionFiles) {
    if (!byComponent.has(component)) {
      byComponent.set(component, []);
    }
  }

  for (const component of byComponent.keys()) {
    const componentRoot = path.join(oneApiRoot, component);
    const licensing = await nearestLicensingDirectory(
      path.dirname(companionFiles.find((file) => file.component === component).absolute),
      componentRoot,
    );
    if (!licensing) {
      throw new Error(`No controlling license material was found for oneAPI component ${component}.`);
    }
    const candidates = (await listFiles(licensing)).filter((file) => (
      /(license|notice|third[-_ ]?party)/i.test(path.basename(file))
    ));
    if (candidates.length === 0) {
      throw new Error(`No controlling license material was found for oneAPI component ${component}.`);
    }

    const seenHashes = new Set();
    const seenOutputs = new Map();
    for (const absolute of candidates.sort()) {
      const hash = await sha256File(absolute);
      if (seenHashes.has(hash)) {
        continue;
      }
      seenHashes.add(hash);
      const relative = toPosix(path.relative(licensing, absolute));
      const outputKey = relative.toLowerCase();
      if (seenOutputs.has(outputKey)) {
        throw new Error(`Controlling license files for oneAPI component ${component} collide at ${relative}.`);
      }
      seenOutputs.set(outputKey, absolute);
      byComponent.get(component).push({ absolute, component, relative });
    }
  }

  return [...byComponent.values()].flat();
}

async function nearestLicensingDirectory(start, boundary) {
  let current = path.resolve(start);
  const resolvedBoundary = path.resolve(boundary);
  while (current === resolvedBoundary || isBelow(resolvedBoundary, current)) {
    const candidate = path.join(current, 'licensing');
    if (await isDirectory(candidate)) {
      return candidate;
    }
    if (current === resolvedBoundary) {
      break;
    }
    current = path.dirname(current);
  }
  return undefined;
}

async function describeStagedBundle(root, destination, staging) {
  const destinationRelative = toPosix(path.relative(root, destination));
  if (!destinationRelative || destinationRelative.startsWith('../') || path.isAbsolute(destinationRelative)) {
    throw new Error('SYCL bundle destination must be below the repository root.');
  }
  const files = [];
  for (const absolute of await listFiles(staging)) {
    const relative = toPosix(path.relative(staging, absolute));
    files.push({
      path: path.posix.join(destinationRelative, relative),
      sha256: await sha256File(absolute),
    });
  }
  files.sort((left, right) => (left.path < right.path ? -1 : left.path > right.path ? 1 : 0));
  const executable = path.posix.join(destinationRelative, 'llama-server.exe');
  if (!files.some((file) => file.path === executable)) {
    throw new Error('The assembled SYCL bundle does not contain llama-server.exe.');
  }
  return { executable, files };
}

function classifyOneApiFiles(oneApiRoot, files) {
  return files.flatMap((absolute) => {
    if (!isAtOrBelow(oneApiRoot, absolute)) return [];
    const relative = path.relative(oneApiRoot, absolute);
    const [component] = relative.split(path.sep);
    if (!component || component === '..') {
      throw new Error(`Could not identify the oneAPI component for ${absolute}.`);
    }
    return [{ absolute, component }];
  });
}

function resolveFromTiers(tiers) {
  return async (dllName) => {
    for (const tier of tiers) {
      const candidates = [];
      for (const directory of tier.directories) {
        const found = await findNamedFile(directory, dllName);
        if (found && !candidates.some((candidate) => normalizedSourcePath(candidate) === normalizedSourcePath(found))) {
          candidates.push(found);
        }
      }
      if (tier.rejectAmbiguous && candidates.length > 1) {
        throw new Error(
          `Ambiguous ${tier.label} dependency ${dllName}; distinct candidates: ${candidates.join(', ')}.`,
        );
      }
      if (candidates.length > 0) {
        return candidates[0];
      }
    }
    return undefined;
  };
}

function uniqueSourcePaths(values) {
  const unique = [];
  const seen = new Set();
  for (const value of values) {
    const normalized = normalizedSourcePath(value);
    if (!seen.has(normalized)) {
      seen.add(normalized);
      unique.push(value);
    }
  }
  return unique;
}

async function findNamedFile(directory, name) {
  if (!directory) {
    return undefined;
  }
  const exact = path.join(directory, name);
  if (await isFile(exact)) {
    return exact;
  }
  try {
    const entries = await readdir(directory, { withFileTypes: true });
    const entry = entries.find((candidate) => candidate.isFile() && candidate.name.toLowerCase() === name.toLowerCase());
    return entry ? path.join(directory, entry.name) : undefined;
  } catch {
    return undefined;
  }
}

function isSystemDependencyName(name) {
  const normalized = name.toLowerCase();
  return normalized.startsWith('api-ms-win-') || SYSTEM_DEPENDENCIES.has(normalized);
}

function splitSearchPath(value) {
  if (!value) {
    return [];
  }
  const delimiter = value.includes(';') ? ';' : path.delimiter;
  return value.split(delimiter).filter(Boolean);
}

function isBelow(parent, child) {
  const relative = path.relative(path.resolve(parent), path.resolve(child));
  return relative !== '' && !relative.startsWith('..') && !path.isAbsolute(relative);
}

function isAtOrBelow(parent, child) {
  return path.resolve(parent) === path.resolve(child) || isBelow(parent, child);
}

function normalizedSourcePath(value) {
  return path.resolve(value).toLowerCase();
}

function toPosix(value) {
  return value.split(path.sep).join('/');
}

async function listFiles(root) {
  const files = [];
  for (const entry of await readdir(root, { withFileTypes: true })) {
    const absolute = path.join(root, entry.name);
    if (entry.isDirectory()) {
      files.push(...await listFiles(absolute));
    } else if (entry.isFile()) {
      files.push(absolute);
    }
  }
  return files;
}

async function requireFile(file, message) {
  if (!(await isFile(file))) {
    throw new Error(`${message}: ${file}.`);
  }
}

async function isFile(value) {
  try {
    return (await stat(value)).isFile();
  } catch {
    return false;
  }
}

async function isDirectory(value) {
  try {
    return (await stat(value)).isDirectory();
  } catch {
    return false;
  }
}
