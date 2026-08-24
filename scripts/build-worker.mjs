import { cp, mkdir, rm, stat } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import path from 'node:path';
import process from 'node:process';

const LLAMA_CPP_COMMIT = '60eeeb6082c1126bb8bc72902c83123cd056811b';
const root = path.resolve(import.meta.dirname, '..');
const hostPlatform = `${process.platform}-${process.arch}`;
const targetArgument = process.argv.indexOf('--target');
const platform = targetArgument >= 0 ? process.argv[targetArgument + 1] : hostPlatform;

if (!['darwin-arm64', 'win32-x64'].includes(platform)) {
  throw new Error(`Worker builds support darwin-arm64 and win32-x64, not ${platform}.`);
}
if (platform === 'darwin-arm64' && hostPlatform !== 'darwin-arm64') {
  throw new Error('The darwin-arm64 worker must be built on Apple Silicon macOS.');
}

const crossCompilingWindows = platform === 'win32-x64' && process.platform !== 'win32';
const llvmMingwRoot = process.env.LOCAL_LLM_LLVM_MINGW_ROOT;
if (crossCompilingWindows && !llvmMingwRoot) {
  throw new Error(
    'Cross-building win32-x64 requires LOCAL_LLM_LLVM_MINGW_ROOT to point to an llvm-mingw toolchain.',
  );
}

const source = path.join(root, 'build', 'llama.cpp');
const buildDirectory = path.join(root, 'build', `llama.cpp-${platform}`);
const destination = path.join(
  root,
  'resources',
  'workers',
  platform,
  platform === 'win32-x64' ? 'llama-server.exe' : 'llama-server',
);

await mkdir(path.dirname(destination), { recursive: true });
if (!(await exists(path.join(source, '.git')))) {
  await run('git', ['clone', '--filter=blob:none', 'https://github.com/ggml-org/llama.cpp.git', source]);
}
await run('git', ['fetch', '--depth=1', 'origin', LLAMA_CPP_COMMIT], source);
await run('git', ['checkout', '--detach', LLAMA_CPP_COMMIT], source);
await rm(buildDirectory, { recursive: true, force: true });

const cmakeOptions = [
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
];
if (platform === 'darwin-arm64') {
  cmakeOptions.push(
    '-DGGML_METAL=ON',
    '-DGGML_METAL_EMBED_LIBRARY=ON',
    '-DCMAKE_OSX_ARCHITECTURES=arm64',
    '-DCMAKE_OSX_DEPLOYMENT_TARGET=13.3',
  );
} else if (crossCompilingWindows && llvmMingwRoot) {
  cmakeOptions.push(
    '-DGGML_METAL=OFF',
    '-DGGML_BLAS=OFF',
    '-DCMAKE_SYSTEM_NAME=Windows',
    '-DCMAKE_SYSTEM_PROCESSOR=x86_64',
    `-DCMAKE_C_COMPILER=${path.join(llvmMingwRoot, 'bin', 'x86_64-w64-mingw32-clang')}`,
    `-DCMAKE_CXX_COMPILER=${path.join(llvmMingwRoot, 'bin', 'x86_64-w64-mingw32-clang++')}`,
    `-DCMAKE_RC_COMPILER=${path.join(llvmMingwRoot, 'bin', 'x86_64-w64-mingw32-windres')}`,
    '-DCMAKE_C_FLAGS=-D_WIN32_WINNT=0x0A00',
    '-DCMAKE_CXX_FLAGS=-D_WIN32_WINNT=0x0A00',
    '-DCMAKE_EXE_LINKER_FLAGS=-static',
  );
} else {
  cmakeOptions.push(
    '-DGGML_METAL=OFF',
    '-DGGML_BLAS=OFF',
    '-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded',
  );
}

await run('cmake', cmakeOptions);
await run('cmake', ['--build', buildDirectory, '--config', 'Release', '--target', 'llama-server', '--parallel']);

const candidates = platform === 'win32-x64'
  ? [
      path.join(buildDirectory, 'bin', 'Release', 'llama-server.exe'),
      path.join(buildDirectory, 'bin', 'llama-server.exe'),
    ]
  : [path.join(buildDirectory, 'bin', 'llama-server')];
const binary = await firstExisting(candidates);
await cp(binary, destination);
if (platform === 'darwin-arm64') {
  await import('node:fs/promises').then(({ chmod }) => chmod(destination, 0o755));
}
console.log(`Bundled llama.cpp ${LLAMA_CPP_COMMIT} worker at ${destination}`);

async function run(command, args, cwd = root) {
  await new Promise((resolve, reject) => {
    const cleanEnvironment = { ...process.env };
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
