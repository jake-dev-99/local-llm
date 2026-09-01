import path from 'node:path';

const supportedTargets = new Set(['darwin-arm64', 'win32-x64']);
const supportedBackends = new Set(['cpu', 'sycl', 'all']);

const commonCmakeOptions = [
  '-DCMAKE_BUILD_TYPE=Release',
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

export function parseWorkerBuildOptions(argv, hostTarget) {
  let target = hostTarget;
  let backend;

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument !== '--target' && argument !== '--backend') {
      throw new Error(`Unknown worker build option ${argument}.`);
    }
    const value = argv[index + 1];
    if (!value || value.startsWith('--')) {
      throw new Error(`Worker build option ${argument} requires a value.`);
    }
    if (argument === '--target') {
      target = value;
    } else {
      backend = value;
    }
    index += 1;
  }

  if (!supportedTargets.has(target)) {
    throw new Error(`Worker builds support darwin-arm64 and win32-x64, not ${target}.`);
  }
  if (target === 'darwin-arm64' && hostTarget !== 'darwin-arm64') {
    throw new Error('The darwin-arm64 worker must be built on Apple Silicon macOS.');
  }

  backend ??= target === 'win32-x64' && hostTarget === 'win32-x64' ? 'all' : 'cpu';
  if (!supportedBackends.has(backend)) {
    throw new Error(`Worker builds support cpu, sycl, and all backends, not ${backend}.`);
  }
  if (target !== 'win32-x64' && backend !== 'cpu') {
    throw new Error(`The ${backend} backend is supported only for win32-x64 workers.`);
  }
  if ((backend === 'sycl' || backend === 'all') && hostTarget !== 'win32-x64') {
    throw new Error('SYCL worker must be built on native x64 Windows.');
  }

  return { target, backend };
}

export function cmakeOptionsForBuild({ target, backend, hostTarget, llvmMingwRoot, ninjaPath }) {
  if (backend === 'sycl') {
    if (!ninjaPath) {
      throw new Error('Visual Studio Ninja path is required for the Windows SYCL worker build.');
    }
    return [
      ...commonCmakeOptions,
      '-G', 'Ninja',
      `-DCMAKE_MAKE_PROGRAM=${ninjaPath}`,
      '-DBUILD_SHARED_LIBS=ON',
      '-DGGML_STATIC=OFF',
      '-DGGML_BACKEND_DL=ON',
      '-DGGML_NATIVE=OFF',
      '-DGGML_OPENMP=OFF',
      '-DGGML_METAL=OFF',
      '-DGGML_SYCL=ON',
      '-DGGML_SYCL_TARGET=INTEL',
      '-DCMAKE_C_COMPILER=cl',
      '-DCMAKE_CXX_COMPILER=icx',
    ];
  }

  const options = [
    ...commonCmakeOptions,
    '-DBUILD_SHARED_LIBS=OFF',
    '-DGGML_STATIC=ON',
    '-DGGML_NATIVE=OFF',
    '-DGGML_OPENMP=OFF',
  ];
  if (target === 'darwin-arm64') {
    return options.concat(
      '-DGGML_METAL=ON',
      '-DGGML_METAL_EMBED_LIBRARY=ON',
      '-DCMAKE_OSX_ARCHITECTURES=arm64',
      '-DCMAKE_OSX_DEPLOYMENT_TARGET=13.3',
    );
  }
  if (target === 'win32-x64' && hostTarget !== 'win32-x64') {
    if (!llvmMingwRoot) {
      throw new Error(
        'Cross-building win32-x64 requires LOCAL_LLM_LLVM_MINGW_ROOT to point to an llvm-mingw toolchain.',
      );
    }
    return options.concat(
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
  }
  return options.concat(
    '-DGGML_METAL=OFF',
    '-DGGML_BLAS=OFF',
    '-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded',
  );
}
