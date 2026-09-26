import { randomUUID } from 'node:crypto';
import { spawn } from 'node:child_process';
import {
  chmod,
  cp,
  mkdir,
  readFile,
  readdir,
  rename,
  rm,
  stat,
  writeFile,
} from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

import { parseWorkerManifest, sha256File } from '../src/worker/workerManifest.ts';
import { WINDOWS_WORKER_RELEASE } from './windows-worker-archive.mjs';

export const LINUX_WORKER_RELEASE = Object.freeze({
  commit: WINDOWS_WORKER_RELEASE.commit,
  build: WINDOWS_WORKER_RELEASE.build,
});

const SYSTEM_LIBRARY_PREFIXES = [
  'linux-vdso.so',
  'ld-linux-x86-64.so',
  'libc.so',
  'libm.so',
  'libpthread.so',
  'libdl.so',
  'librt.so',
  'libresolv.so',
  'libstdc++.so',
  'libgcc_s.so',
  'libze_loader.so',
  'libze_intel_gpu.so',
  'libigdrcl.so',
  'libigdfcl.so',
];

export async function prepareLinuxWorkerArchive(root, dependencies = {}) {
  const release = dependencies.release ?? LINUX_WORKER_RELEASE;
  const runCommand = dependencies.runCommand ?? run;
  const log = dependencies.log ?? console.log;
  const workersDirectory = path.join(root, 'resources', 'workers');
  const manifestPath = path.join(workersDirectory, 'manifest.json');
  const token = `${process.pid}-${randomUUID()}`;
  const stagedLinux = path.join(workersDirectory, `.linux-x64-stage-${token}`);
  const stagedBundle = path.join(stagedLinux, 'sycl');
  const stagedManifest = path.join(workersDirectory, `.manifest-stage-${token}.json`);
  const linuxDestination = path.join(workersDirectory, 'linux-x64');
  const linuxBackup = path.join(workersDirectory, `.linux-x64-backup-${token}`);
  const manifestBackup = path.join(workersDirectory, `.manifest-backup-${token}.json`);
  let published = false;

  await mkdir(workersDirectory, { recursive: true });
  try {
    await buildLinuxSyclServer(root, stagedBundle, release, runCommand, log);
    const bundle = await describeLinuxBundle(root, stagedBundle, dependencies);
    const existingManifest = JSON.parse(await readFile(manifestPath, 'utf8'));
    const manifest = createLinuxWorkerManifest(existingManifest, bundle, release);
    await writeFile(stagedManifest, `${JSON.stringify(manifest, null, 2)}\n`, { flag: 'wx' });
    await publishTransaction({
      manifestPath,
      manifestBackup,
      stagedManifest,
      stagedLinux,
      linuxBackup,
      linuxDestination,
      copyDirectory: dependencies.copyDirectory ?? cp,
      renameFile: dependencies.renameFile ?? rename,
    });
    published = true;
    log(`[worker-archive] Prepared ${release.build} Linux SYCL worker from ${release.commit}.`);
    return manifest;
  } finally {
    await Promise.all([
      rm(stagedLinux, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 }),
      rm(stagedManifest, { force: true }),
      ...(published ? [
        rm(linuxBackup, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 }),
        rm(manifestBackup, { force: true }),
      ] : []),
    ]);
  }
}

export function createLinuxWorkerManifest(existingManifest, bundle, release = LINUX_WORKER_RELEASE) {
  const existing = structuredClone(existingManifest);
  const platforms = existing?.manifestVersion === 2
    ? existing.platforms
    : legacyPlatforms(existing);
  platforms['linux-x64'] = {
    modes: {
      auto: { bundle: 'sycl', backend: 'sycl' },
      cpu: { bundle: 'sycl', backend: 'cpu' },
    },
    bundles: { sycl: bundle },
  };
  return parseWorkerManifest({
    manifestVersion: 2,
    llamaCppCommit: release.commit,
    llamaCppBuild: release.build,
    platforms,
  });
}

function legacyPlatforms(existing) {
  const platforms = {};
  for (const [target, platform] of Object.entries(existing?.platforms ?? {})) {
    platforms[target] = platform;
  }
  return platforms;
}

async function buildLinuxSyclServer(root, stagedBundle, release, runCommand, log) {
  if (process.platform !== 'linux' || process.arch !== 'x64') {
    throw new Error('The linux-x64 worker must be built on x64 Linux.');
  }
  const source = path.join(root, 'build', 'llama.cpp');
  const buildDirectory = path.join(root, 'build', 'llama.cpp-linux-x64');
  await mkdir(path.dirname(stagedBundle), { recursive: true });
  if (!(await exists(path.join(source, '.git')))) {
    await runCommand('git', ['clone', '--filter=blob:none', 'https://github.com/ggml-org/llama.cpp.git', source], root);
  }
  await runCommand('git', ['fetch', '--depth=1', 'origin', release.commit], source);
  await runCommand('git', ['checkout', '--detach', release.commit], source);
  await rm(buildDirectory, { recursive: true, force: true });
  // Keep GGML_SYCL_F16=OFF until the Arc 140T BF16 correctness tests pass
  // (ggml-org/llama.cpp#27771).
  await runCommand('cmake', [
    '-S', source,
    '-B', buildDirectory,
    '-DCMAKE_BUILD_TYPE=Release',
    '-DCMAKE_C_COMPILER=icx',
    '-DCMAKE_CXX_COMPILER=icpx',
    '-DGGML_SYCL=ON',
    '-DGGML_SYCL_F16=OFF',
    '-DGGML_CUDA=OFF',
    '-DGGML_NATIVE=OFF',
    '-DLLAMA_BUILD_TESTS=OFF',
    '-DLLAMA_BUILD_EXAMPLES=OFF',
    '-DLLAMA_BUILD_APP=OFF',
    '-DLLAMA_BUILD_SERVER=ON',
    '-DLLAMA_BUILD_UI=OFF',
    '-DLLAMA_USE_PREBUILT_UI=OFF',
    '-DLLAMA_OPENSSL=OFF',
    '-DLLAMA_LLGUIDANCE=OFF',
    '-DLLAMA_SUBPROCESS=OFF',
  ], root);
  await runCommand('cmake', ['--build', buildDirectory, '--config', 'Release', '--target', 'llama-server', '--parallel'], root);
  const binary = path.join(buildDirectory, 'bin', 'llama-server');
  if (!(await exists(binary))) {
    throw new Error(`Could not find llama-server at ${binary}.`);
  }
  await mkdir(stagedBundle, { recursive: true });
  await cp(binary, path.join(stagedBundle, 'llama-server'));
  await chmod(path.join(stagedBundle, 'llama-server'), 0o755);
  log(`[worker-archive] Built llama.cpp ${release.commit} Linux SYCL server.`);
}

export async function describeLinuxBundle(root, stagedBundle, dependencies = {}) {
  const listDependencies = dependencies.listDependencies ?? listSharedDependencies;
  const copyDependency = dependencies.copyDependency ?? cp;
  const patchRpath = dependencies.patchRpath ?? setOriginRpath;
  const stagedFiles = new Map();
  stagedFiles.set('llama-server', path.join(stagedBundle, 'llama-server'));
  const queue = [path.join(stagedBundle, 'llama-server')];
  const seen = new Set(queue);
  while (queue.length > 0) {
    const binary = queue.pop();
    const needed = await listDependencies(binary, dependencies);
    for (const dependency of needed) {
      if (isSystemLibrary(dependency.name)) {
        continue;
      }
      if (!dependency.path) {
        throw new Error(`Linux worker dependency is not available locally: ${dependency.name} (required by ${binary}).`);
      }
      const stagedName = path.basename(dependency.path);
      if (!stagedFiles.has(stagedName)) {
        const destination = path.join(stagedBundle, stagedName);
        await copyDependency(dependency.path, destination);
        await chmod(destination, 0o755).catch(() => undefined);
        stagedFiles.set(stagedName, destination);
      }
      const resolved = stagedFiles.get(stagedName);
      if (!seen.has(resolved)) {
        seen.add(resolved);
        queue.push(resolved);
      }
    }
  }
  for (const staged of stagedFiles.values()) {
    await patchRpath(staged, dependencies);
  }
  const prefix = 'resources/workers/linux-x64/sycl';
  const files = [];
  for (const name of [...stagedFiles.keys()].sort((left, right) => left.localeCompare(right))) {
    files.push({
      path: `${prefix}/${name}`,
      sha256: await sha256File(stagedFiles.get(name)),
    });
  }
  await copyLicenses(root, stagedBundle, dependencies);
  const licenseFiles = await readdir(stagedBundle).then(
    (entries) => entries.filter((entry) => /^LICENSE|NOTICE|THIRD[-_]PARTY/i.test(entry)).sort(),
  );
  for (const license of licenseFiles) {
    files.push({
      path: `${prefix}/${license}`,
      sha256: await sha256File(path.join(stagedBundle, license)),
    });
  }
  files.sort((left, right) => left.path.localeCompare(right.path));
  return { executable: `${prefix}/llama-server`, files };
}

function isSystemLibrary(name) {
  return SYSTEM_LIBRARY_PREFIXES.some((prefix) => name === prefix || name.startsWith(`${prefix}.`));
}

async function listSharedDependencies(binary, dependencies = {}) {
  const runCommand = dependencies.runCommand ?? runCapture;
  const ldd = await runCommand('ldd', [binary], process.cwd());
  const results = [];
  for (const line of ldd.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('linux-vdso')) {
      continue;
    }
    const arrow = trimmed.match(/^(\S+)\s+=>\s+(\S+)(?:\s+\(.*\))?$/);
    if (arrow) {
      if (arrow[2] === 'not') {
        results.push({ name: arrow[1] });
      } else {
        results.push({ name: arrow[1], path: arrow[2] });
      }
      continue;
    }
    const direct = trimmed.match(/^(\/\S+)(?:\s+\(.*\))?$/);
    if (direct) {
      results.push({ name: path.basename(direct[1]), path: direct[1] });
    }
  }
  return results;
}

async function setOriginRpath(binary, dependencies = {}) {
  const runCommand = dependencies.runCommand ?? run;
  await runCommand('patchelf', ['--set-rpath', '$ORIGIN', binary], process.cwd());
}

async function copyLicenses(root, stagedBundle, dependencies = {}) {
  const copyFile = dependencies.copyDependency ?? cp;
  const source = path.join(root, 'build', 'llama.cpp');
  const candidates = ['LICENSE', 'LICENSE.md', 'NOTICE', 'NOTICE.md', 'THIRD_PARTY_NOTICES.md'];
  for (const candidate of candidates) {
    const from = path.join(source, candidate);
    if (await exists(from)) {
      await copyFile(from, path.join(stagedBundle, candidate));
    }
  }
}

async function publishTransaction(options) {
  let linuxBackedUp = false;
  let manifestBackedUp = false;
  const rollback = [];
  try {
    if (await exists(options.linuxDestination)) {
      await options.renameFile(options.linuxDestination, options.linuxBackup);
      linuxBackedUp = true;
      rollback.push(async () => {
        await rm(options.linuxDestination, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
        await options.renameFile(options.linuxBackup, options.linuxDestination);
      });
    }
    await options.copyDirectory(options.stagedLinux, options.linuxDestination, { recursive: true });
    if (await exists(options.manifestPath)) {
      await options.renameFile(options.manifestPath, options.manifestBackup);
      manifestBackedUp = true;
      rollback.push(async () => {
        await options.renameFile(options.manifestBackup, options.manifestPath);
      });
    }
    await options.renameFile(options.stagedManifest, options.manifestPath);
    return;
  } catch (error) {
    for (const undo of rollback.reverse()) {
      try {
        await undo();
      } catch {
        // Keep the original error; rollback failures leave backups for recovery.
      }
    }
    if (!linuxBackedUp) {
      await rm(options.linuxDestination, { recursive: true, force: true }).catch(() => undefined);
    }
    if (!manifestBackedUp) {
      await rm(options.manifestPath, { force: true }).catch(() => undefined);
    }
    throw error;
  }
}

async function run(command, args, cwd = process.cwd()) {
  await new Promise((resolve, reject) => {
    const environment = { ...process.env };
    delete environment.CFLAGS;
    delete environment.CXXFLAGS;
    delete environment.CPPFLAGS;
    delete environment.LDFLAGS;
    const child = spawn(command, args, { cwd, env: environment, stdio: 'inherit', shell: false });
    child.once('error', reject);
    child.once('exit', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`${command} exited with code ${code ?? 'unknown'}.`));
    });
  });
}

async function runCapture(command, args, cwd = process.cwd()) {
  return await new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd, stdio: ['ignore', 'pipe', 'pipe'], shell: false });
    let stdout = '';
    let stderr = '';
    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');
    child.stdout.on('data', (chunk) => { stdout += chunk; });
    child.stderr.on('data', (chunk) => { stderr += chunk; });
    child.once('error', reject);
    child.once('close', (code) => {
      if (code === 0) resolve(stdout);
      else reject(new Error(`${command} exited with code ${code ?? 'unknown'}: ${stderr || stdout}`));
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
