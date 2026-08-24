import { cp, chmod, mkdir, rm, stat } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import path from 'node:path';
import process from 'node:process';

import { sha256File } from '../src/worker/workerManifest.ts';
import { cmakeOptionsForBuild, parseWorkerBuildOptions } from './worker-build-options.mjs';
import { assembleWindowsSyclBundle, writeUpdatedManifest } from './windows-sycl-bundle.mjs';
import { loadOneApiEnvironment, resolveOneApiFiles } from './windows-oneapi.mjs';

const LLAMA_CPP_COMMIT = '60eeeb6082c1126bb8bc72902c83123cd056811b';
const root = path.resolve(import.meta.dirname, '..');
const hostTarget = `${process.platform}-${process.arch}`;
const options = parseWorkerBuildOptions(process.argv.slice(2), hostTarget);
const source = path.join(root, 'build', 'llama.cpp');

if (!(await exists(path.join(source, '.git')))) {
  await run('git', ['clone', '--filter=blob:none', 'https://github.com/ggml-org/llama.cpp.git', source]);
}
await run('git', ['fetch', '--depth=1', 'origin', LLAMA_CPP_COMMIT], source);
await run('git', ['checkout', '--detach', LLAMA_CPP_COMMIT], source);

const backends = options.backend === 'all' ? ['cpu', 'sycl'] : [options.backend];
const builds = [];
for (const backend of backends) {
  const environment = backend === 'sycl'
    ? await loadOneApiEnvironment(process.env)
    : process.env;
  builds.push(await buildWorker({ ...options, backend, hostTarget, environment }));
}
for (const build of builds) {
  await publishWorker(build);
}

async function buildWorker({ target, backend, hostTarget: buildHostTarget, environment }) {
  const buildDirectory = workerBuildDirectory(target, backend);
  const cmakeOptions = cmakeOptionsForBuild({
    target,
    backend,
    hostTarget: buildHostTarget,
    llvmMingwRoot: environment.LOCAL_LLM_LLVM_MINGW_ROOT,
    oneApiEnvironment: environment,
  });

  await rm(buildDirectory, { recursive: true, force: true });
  await run('cmake', ['-S', source, '-B', buildDirectory, ...cmakeOptions], root, environment);
  await run(
    'cmake',
    ['--build', buildDirectory, '--config', 'Release', '--target', 'llama-server', '--parallel'],
    root,
    environment,
  );

  const binary = await firstExisting(workerCandidates(buildDirectory, target));
  return { target, backend, environment, buildDirectory, binary };
}

async function publishWorker({ target, backend, environment, binary }) {
  const destination = workerDestination(target, backend);
  if (target === 'win32-x64' && backend === 'sycl') {
    const oneApi = resolveOneApiFiles(environment);
    const bundle = await assembleWindowsSyclBundle({
      root,
      buildOutput: path.dirname(binary),
      destination: path.dirname(destination),
      oneApiRoot: oneApi.oneApiRoot,
      vcToolsRedistDir: oneApi.vcToolsRedistDir,
      levelZeroSdkPath: oneApi.levelZeroSdkPath,
      oneApiPath: environmentValue(environment, 'PATH'),
      systemRoot: environmentValue(environment, 'SystemRoot'),
      runDumpbin: async (absoluteFile) => await capture(
        'dumpbin',
        ['/nologo', '/dependents', absoluteFile],
        root,
        environment,
      ),
    });
    await writeUpdatedManifest(root, target, backend, bundle);
  } else {
    if (target === 'win32-x64') {
      await rm(path.dirname(destination), { recursive: true, force: true });
    }
    await mkdir(path.dirname(destination), { recursive: true });
    await cp(binary, destination);
    if (target === 'win32-x64') {
      const relative = toPosix(path.relative(root, destination));
      await writeUpdatedManifest(root, target, backend, {
        executable: relative,
        files: [{ path: relative, sha256: await sha256File(destination) }],
      });
    }
  }
  if (target === 'darwin-arm64') {
    await chmod(destination, 0o755);
  }
  console.log(`Bundled llama.cpp ${LLAMA_CPP_COMMIT} ${backend} worker at ${destination}`);
}

function workerBuildDirectory(target, backend) {
  return path.join(root, 'build', `llama.cpp-${target}${target === 'win32-x64' ? `-${backend}` : ''}`);
}

function workerDestination(target, backend) {
  if (target === 'darwin-arm64') {
    return path.join(root, 'resources', 'workers', target, 'llama-server');
  }
  return path.join(root, 'resources', 'workers', target, backend, 'llama-server.exe');
}

function workerCandidates(buildDirectory, target) {
  return target === 'win32-x64'
    ? [
        path.join(buildDirectory, 'bin', 'Release', 'llama-server.exe'),
        path.join(buildDirectory, 'bin', 'llama-server.exe'),
      ]
    : [path.join(buildDirectory, 'bin', 'llama-server')];
}

async function run(command, args, cwd = root, environment = process.env) {
  await new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd,
      env: cleanBuildEnvironment(environment),
      stdio: 'inherit',
      shell: false,
    });
    child.once('error', reject);
    child.once('exit', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`${command} exited with code ${code ?? 'unknown'}.`));
      }
    });
  });
}

async function capture(command, args, cwd = root, environment = process.env) {
  return await new Promise((resolve, reject) => {
    const cleanEnvironment = cleanBuildEnvironment(environment);
    const child = spawn(command, args, {
      cwd,
      env: cleanEnvironment,
      stdio: ['ignore', 'pipe', 'inherit'],
      shell: false,
    });
    let stdout = '';
    child.stdout.setEncoding('utf8');
    child.stdout.on('data', (chunk) => {
      stdout += chunk;
    });
    child.once('error', reject);
    child.once('exit', (code) => {
      if (code === 0) {
        resolve(stdout);
      } else {
        reject(new Error(`${command} exited with code ${code ?? 'unknown'}.`));
      }
    });
  });
}

function cleanBuildEnvironment(environment) {
  const cleanEnvironment = { ...environment };
  delete cleanEnvironment.CFLAGS;
  delete cleanEnvironment.CXXFLAGS;
  delete cleanEnvironment.CPPFLAGS;
  delete cleanEnvironment.LDFLAGS;
  return cleanEnvironment;
}

function environmentValue(environment, name) {
  const key = Object.keys(environment).find((candidate) => candidate.toLowerCase() === name.toLowerCase());
  return key ? environment[key] : undefined;
}

function toPosix(value) {
  return value.split(path.sep).join('/');
}

async function exists(value) {
  try {
    await stat(value);
    return true;
  } catch {
    return false;
  }
}

async function firstExisting(candidates) {
  for (const candidate of candidates) {
    if (await exists(candidate)) {
      return candidate;
    }
  }
  throw new Error(`Could not find llama-server in ${candidates.join(', ')}.`);
}
