import { cp, mkdir, mkdtemp, readFile, readdir, rename, rm, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';

import { parseWorkerManifest, replaceWorkerBundle, sha256File } from '../src/worker/workerManifest.ts';

export const REQUIRED_SYCL_COMPANIONS = [
  'compiler/latest/bin/sycl8.dll',
  'compiler/latest/bin/ur_adapter_level_zero.dll',
  'compiler/latest/bin/ur_adapter_level_zero_v2.dll',
  'compiler/latest/bin/ur_adapter_opencl.dll',
  'compiler/latest/bin/ur_loader.dll',
  'compiler/latest/bin/ur_win_proxy_loader.dll',
  'compiler/latest/bin/svml_dispmd.dll',
  'compiler/latest/bin/libmmd.dll',
  'compiler/latest/bin/libiomp5md.dll',
  'compiler/latest/bin/libsycl-fallback-bfloat16.spv',
  'compiler/latest/bin/libsycl-native-bfloat16.spv',
  'mkl/latest/bin/mkl_sycl_blas.5.dll',
  'mkl/latest/bin/mkl_core.2.dll',
  'mkl/latest/bin/mkl_tbb_thread.2.dll',
  'dnnl/latest/bin/dnnl.dll',
  'tbb/latest/bin/tbb12.dll',
  'tcm/latest/bin/tcm.dll',
  'tcm/latest/bin/libhwloc-15.dll',
  'umf/latest/bin/umf.dll',
];

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
  const queue = [...roots];
  const files = [];
  const seen = new Map();

  while (queue.length > 0) {
    const absoluteFile = queue.shift();
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
      const resolved = await options.resolve(dllName);
      if (options.isSystemDependency(dllName, resolved)) {
        continue;
      }
      if (!resolved) {
        throw new Error(`Required dependency ${dllName} could not be resolved.`);
      }
      queue.push(resolved);
    }
  }

  return files;
}

export async function assembleWindowsSyclBundle(options) {
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

  const companionFiles = [];
  for (const relative of REQUIRED_SYCL_COMPANIONS) {
    const absolute = path.join(options.oneApiRoot, ...relative.split('/'));
    if (!(await isFile(absolute))) {
      throw new Error(`Required SYCL runtime file ${path.basename(relative)} was not found at ${absolute}.`);
    }
    companionFiles.push({ absolute, component: relative.split('/')[0] });
  }

  const licenses = await collectControllingLicenses(options.oneApiRoot, companionFiles);
  const pathDirectories = options.pathDirectories ?? splitSearchPath(options.oneApiPath ?? process.env.PATH);
  const systemRoot = options.systemRoot ?? process.env.SystemRoot ?? 'C:\\Windows';
  const system32 = path.join(systemRoot, 'System32');
  const vcRuntime = path.join(options.vcToolsRedistDir, 'x64', 'Microsoft.VC143.CRT');
  const oneApiDirectories = pathDirectories.filter((directory) => !isAtOrBelow(systemRoot, directory));
  const pathSystemDirectories = pathDirectories.filter((directory) => isAtOrBelow(systemRoot, directory));
  const searchDirectories = [
    options.buildOutput,
    ...oneApiDirectories,
    vcRuntime,
    ...pathSystemDirectories,
    system32,
  ];
  const closure = await collectDependencyClosure(
    [...backendModules, ...companionFiles.map((file) => file.absolute), executable],
    {
      imports: async (absoluteFile) => (
        /\.(?:exe|dll)$/i.test(absoluteFile)
          ? parseDumpbinDependents(await options.runDumpbin(absoluteFile))
          : []
      ),
      resolve: async (dllName) => {
        for (const directory of searchDirectories) {
          const found = await findNamedFile(directory, dllName);
          if (found) {
            return found;
          }
        }
        return undefined;
      },
      isSystemDependency: (dllName, absoluteFile) => (
        isSystemDependencyName(dllName) || (absoluteFile ? isBelow(system32, absoluteFile) : false)
      ),
    },
  );

  const parent = path.dirname(options.destination);
  await mkdir(parent, { recursive: true });
  const staging = await mkdtemp(path.join(parent, '.sycl-stage-'));
  try {
    const copiedNames = new Map();
    for (const source of closure) {
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

    const bundle = await describeStagedBundle(options.root, options.destination, staging);
    await rm(options.destination, { recursive: true, force: true });
    await rename(staging, options.destination);
    return bundle;
  } catch (error) {
    await rm(staging, { recursive: true, force: true });
    throw error;
  }
}

export async function writeUpdatedManifest(root, target, bundleName, bundle) {
  const manifestPath = path.join(root, 'resources', 'workers', 'manifest.json');
  const current = JSON.parse(await readFile(manifestPath, 'utf8'));
  const updated = current.manifestVersion === 2
    ? replaceWorkerBundle(current, target, bundleName, bundle)
    : updateLegacyManifest(current, target, bundleName, bundle);
  await writeFile(manifestPath, `${JSON.stringify(updated, null, 2)}\n`);
  if (target === 'win32-x64' && bundleName === 'sycl') {
    await rm(path.join(root, 'resources', 'workers', 'win32-x64', 'llama-server.exe'), { force: true });
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
