import { cp, mkdir, rm, stat } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import path from 'node:path';
import process from 'node:process';

import { cmakeOptionsForBuild, parseWorkerBuildOptions } from './worker-build-options.mjs';
import { loadOneApiEnvironment } from './windows-oneapi.mjs';

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
for (const backend of backends) {
  const environment = backend === 'sycl'
    ? await loadOneApiEnvironment(process.env)
    : process.env;
  await buildWorker({ ...options, backend, hostTarget, environment });
}

async function buildWorker({ target, backend, hostTarget: buildHostTarget, environment }) {
  const buildDirectory = workerBuildDirectory(target, backend);
  const destination = workerDestination(target, backend);
  const cmakeOptions = cmakeOptionsForBuild({
    target,
    backend,
    hostTarget: buildHostTarget,
    llvmMingwRoot: environment.LOCAL_LLM_LLVM_MINGW_ROOT,
    oneApiEnvironment: environment,
  });

  await mkdir(path.dirname(destination), { recursive: true });
  await rm(buildDirectory, { recursive: true, force: true });
  await run('cmake', ['-S', source, '-B', buildDirectory, ...cmakeOptions], root, environment);
  await run(
    'cmake',
    ['--build', buildDirectory, '--config', 'Release', '--target', 'llama-server', '--parallel'],
    root,
    environment,
  );

  const binary = await firstExisting(workerCandidates(buildDirectory, target));
  await cp(binary, destination);
  if (target === 'darwin-arm64') {
    await import('node:fs/promises').then(({ chmod }) => chmod(destination, 0o755));
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
    const cleanEnvironment = { ...environment };
    delete cleanEnvironment.CFLAGS;
    delete cleanEnvironment.CXXFLAGS;
    delete cleanEnvironment.CPPFLAGS;
    delete cleanEnvironment.LDFLAGS;
    const child = spawn(command, args, {
      cwd,
      env: cleanEnvironment,
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
