import { cp, chmod, mkdir, rm, stat } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import path from 'node:path';
import process from 'node:process';

import {
  WINDOWS_WORKER_RELEASE,
  prepareWindowsWorkerArchive,
} from './windows-worker-archive.mjs';

const root = path.resolve(import.meta.dirname, '..');
const hostTarget = `${process.platform}-${process.arch}`;
const target = parseTarget(process.argv.slice(2), hostTarget);

if (target === 'win32-x64') {
  await prepareWindowsWorkerArchive(root);
} else {
  await buildDarwinWorker();
}

function parseTarget(args, defaultTarget) {
  if (args.length === 0) return defaultTarget;
  if (args.length !== 2 || args[0] !== '--target' || !args[1]) {
    throw new Error('Usage: npm run build:worker -- [--target darwin-arm64|win32-x64]');
  }
  const requested = args[1];
  if (!['darwin-arm64', 'win32-x64'].includes(requested)) {
    throw new Error(`Worker builds support darwin-arm64 and win32-x64, not ${requested}.`);
  }
  return requested;
}

async function buildDarwinWorker() {
  if (hostTarget !== 'darwin-arm64') {
    throw new Error('The darwin-arm64 worker must be built on Apple Silicon macOS.');
  }

  const source = path.join(root, 'build', 'llama.cpp');
  const buildDirectory = path.join(root, 'build', 'llama.cpp-darwin-arm64');
  const destination = path.join(root, 'resources', 'workers', 'darwin-arm64', 'llama-server');
  await mkdir(path.dirname(destination), { recursive: true });
  if (!(await exists(path.join(source, '.git')))) {
    await run('git', ['clone', '--filter=blob:none', 'https://github.com/ggml-org/llama.cpp.git', source]);
  }
  await run('git', ['fetch', '--depth=1', 'origin', WINDOWS_WORKER_RELEASE.commit], source);
  await run('git', ['checkout', '--detach', WINDOWS_WORKER_RELEASE.commit], source);
  await rm(buildDirectory, { recursive: true, force: true });
  await run('cmake', [
    '-S', source,
    '-B', buildDirectory,
    '-DCMAKE_BUILD_TYPE=Release',
    '-DBUILD_SHARED_LIBS=OFF',
    '-DGGML_STATIC=ON',
    '-DGGML_NATIVE=OFF',
    '-DGGML_OPENMP=OFF',
    '-DLLAMA_BUILD_TESTS=OFF',
    '-DLLAMA_BUILD_EXAMPLES=OFF',
    '-DLLAMA_BUILD_APP=OFF',
    '-DLLAMA_BUILD_SERVER=ON',
    '-DLLAMA_BUILD_UI=OFF',
    '-DLLAMA_USE_PREBUILT_UI=OFF',
    '-DLLAMA_OPENSSL=OFF',
    '-DLLAMA_LLGUIDANCE=OFF',
    '-DLLAMA_SUBPROCESS=OFF',
    '-DGGML_METAL=ON',
    '-DGGML_METAL_EMBED_LIBRARY=ON',
    '-DCMAKE_OSX_ARCHITECTURES=arm64',
    '-DCMAKE_OSX_DEPLOYMENT_TARGET=13.3',
  ]);
  await run(
    'cmake',
    ['--build', buildDirectory, '--config', 'Release', '--target', 'llama-server', '--parallel'],
  );
  const binary = path.join(buildDirectory, 'bin', 'llama-server');
  if (!(await exists(binary))) {
    throw new Error(`Could not find llama-server at ${binary}.`);
  }
  await cp(binary, destination);
  await chmod(destination, 0o755);
  console.log(
    `Bundled llama.cpp ${WINDOWS_WORKER_RELEASE.commit} Darwin worker at ${destination}`,
  );
}

async function run(command, args, cwd = root) {
  await new Promise((resolve, reject) => {
    const environment = { ...process.env };
    delete environment.CFLAGS;
    delete environment.CXXFLAGS;
    delete environment.CPPFLAGS;
    delete environment.LDFLAGS;
    const child = spawn(command, args, {
      cwd,
      env: environment,
      stdio: 'inherit',
      shell: false,
    });
    child.once('error', reject);
    child.once('exit', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`${command} exited with code ${code ?? 'unknown'}.`));
    });
  });
}

async function exists(value) {
  try {
    await stat(value);
    return true;
  } catch (error) {
    if (error?.code === 'ENOENT') return false;
    throw error;
  }
}
