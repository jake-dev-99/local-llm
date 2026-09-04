import { randomUUID } from 'node:crypto';
import { createWriteStream } from 'node:fs';
import {
  cp,
  mkdir,
  readFile,
  rename,
  rm,
  stat,
  writeFile,
} from 'node:fs/promises';
import path from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';

import yauzl from 'yauzl';

import { parseWorkerManifest, sha256File } from '../src/worker/workerManifest.ts';

export const WINDOWS_WORKER_RELEASE = Object.freeze({
  commit: '60eeeb6082c1126bb8bc72902c83123cd056811b',
  build: 'b10472',
  assetName: 'llama-b10472-bin-win-sycl-x64.zip',
  url: 'https://github.com/ggml-org/llama.cpp/releases/download/b10472/llama-b10472-bin-win-sycl-x64.zip',
  size: 119700367,
  sha256: '0c4c50f1e9805933e043d4970f0c2050e4fb5343b8ac0244a49efaa474705830',
});

export async function prepareWindowsWorkerArchive(root, dependencies = {}) {
  const release = dependencies.release ?? WINDOWS_WORKER_RELEASE;
  const fetchFn = dependencies.fetchFn ?? fetch;
  const log = dependencies.log ?? console.log;
  const copyDirectory = dependencies.copyDirectory ?? cp;
  const renameFile = dependencies.renameFile ?? rename;
  const archivePath = await ensureVerifiedArchive(root, release, fetchFn, log);
  const workersDirectory = path.join(root, 'resources', 'workers');
  const manifestPath = path.join(workersDirectory, 'manifest.json');
  const token = `${process.pid}-${randomUUID()}`;
  const stagedWindows = path.join(workersDirectory, `.win32-x64-stage-${token}`);
  const stagedBundle = path.join(stagedWindows, 'sycl');
  const stagedManifest = path.join(workersDirectory, `.manifest-stage-${token}.json`);
  const windowsDestination = path.join(workersDirectory, 'win32-x64');
  const windowsBackup = path.join(workersDirectory, `.win32-x64-backup-${token}`);
  const manifestBackup = path.join(workersDirectory, `.manifest-backup-${token}.json`);
  let published = false;

  await mkdir(workersDirectory, { recursive: true });
  try {
    const extractedFiles = await extractZipArchive(archivePath, stagedBundle);
    const bundle = await describeWindowsBundle(stagedBundle, extractedFiles);
    const existingManifest = JSON.parse(await readFile(manifestPath, 'utf8'));
    const manifest = createWindowsWorkerManifest(existingManifest, bundle, release);
    await writeFile(stagedManifest, `${JSON.stringify(manifest, null, 2)}\n`, { flag: 'wx' });
    await publishTransaction({
      manifestPath,
      manifestBackup,
      stagedManifest,
      stagedWindows,
      windowsBackup,
      windowsDestination,
      copyDirectory,
      renameFile,
    });
    published = true;
    log(`[worker-archive] Prepared ${release.build} Windows SYCL worker from ${release.assetName}.`);
    return manifest;
  } finally {
    await Promise.all([
      rm(stagedWindows, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 }),
      rm(stagedManifest, { force: true }),
      // If rollback itself failed, these may be the only original files left.
      ...(published ? [
        rm(windowsBackup, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 }),
        rm(manifestBackup, { force: true }),
      ] : []),
    ]);
  }
}

export function createWindowsWorkerManifest(existingManifest, bundle, release = WINDOWS_WORKER_RELEASE) {
  const existing = structuredClone(existingManifest);
  const platforms = existing?.manifestVersion === 2
    ? existing.platforms
    : legacyPlatforms(existing);
  platforms['win32-x64'] = {
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

export async function extractZipArchive(archivePath, destination) {
  await mkdir(destination, { recursive: true });
  const destinationRoot = path.resolve(destination);
  const files = [];
  const outputs = new Set();
  const zip = await openZip(archivePath);

  try {
    await new Promise((resolve, reject) => {
      let settled = false;
      const fail = (error) => {
        if (settled) return;
        settled = true;
        zip.close();
        reject(error);
      };
      zip.once('error', (error) => fail(new Error(`Unsafe or invalid ZIP archive: ${error.message}`)));
      zip.once('end', () => {
        if (settled) return;
        settled = true;
        resolve();
      });
      zip.on('entry', (entry) => {
        void extractEntry(zip, entry, destinationRoot, outputs, files)
          .then(() => zip.readEntry(), fail);
      });
      zip.readEntry();
    });
  } finally {
    zip.close();
  }

  if (!files.includes('llama-server.exe')) {
    throw new Error('Official Windows worker archive is invalid: llama-server.exe is missing.');
  }
  return files.sort((left, right) => left.localeCompare(right));
}

async function ensureVerifiedArchive(root, release, fetchFn, log) {
  const cacheDirectory = path.join(root, 'build', 'worker-downloads');
  const archivePath = path.join(cacheDirectory, release.assetName);
  await mkdir(cacheDirectory, { recursive: true });
  if (await exists(archivePath)) {
    await verifyArchive(archivePath, release);
    log(`[worker-archive] Using verified cache ${archivePath}.`);
    return archivePath;
  }

  const partial = `${archivePath}.partial-${process.pid}-${randomUUID()}`;
  try {
    log(`[worker-archive] Downloading ${release.url}.`);
    const response = await fetchFn(release.url);
    if (!response.ok) {
      throw new Error(`Windows worker download failed with HTTP ${response.status}.`);
    }
    if (!response.body) {
      throw new Error('Windows worker download returned an empty response body.');
    }
    await pipeline(
      Readable.fromWeb(response.body),
      createWriteStream(partial, { flags: 'wx' }),
    );
    await verifyArchive(partial, release);
    await rename(partial, archivePath);
    return archivePath;
  } finally {
    await rm(partial, { force: true });
  }
}

async function verifyArchive(archivePath, release) {
  const metadata = await stat(archivePath);
  if (metadata.size !== release.size) {
    throw new Error(
      `Windows worker archive size mismatch: expected ${release.size} bytes, actual ${metadata.size} bytes.`,
    );
  }
  const actualHash = await sha256File(archivePath);
  if (actualHash.toLowerCase() !== release.sha256.toLowerCase()) {
    throw new Error(
      `Windows worker archive SHA-256 mismatch: expected ${release.sha256}, actual ${actualHash}.`,
    );
  }
}

async function describeWindowsBundle(stagedBundle, extractedFiles) {
  const prefix = 'resources/workers/win32-x64/sycl';
  const files = [];
  for (const relativePath of extractedFiles) {
    files.push({
      path: `${prefix}/${relativePath}`,
      sha256: await sha256File(path.join(stagedBundle, ...relativePath.split('/'))),
    });
  }
  const executable = `${prefix}/llama-server.exe`;
  return { executable, files };
}

async function extractEntry(zip, entry, destinationRoot, outputs, files) {
  const relativePath = safeEntryPath(entry.fileName);
  const outputKey = relativePath.toLowerCase();
  if (outputs.has(outputKey)) {
    throw new Error(`Duplicate ZIP output path: ${relativePath}.`);
  }
  outputs.add(outputKey);

  const unixType = (entry.externalFileAttributes >>> 16) & 0o170000;
  if (unixType === 0o120000) {
    throw new Error(`Symbolic links are not allowed in the Windows worker archive: ${relativePath}.`);
  }
  const directory = entry.fileName.endsWith('/');
  if (unixType !== 0 && unixType !== 0o100000 && unixType !== 0o040000) {
    throw new Error(`Unsafe ZIP entry type: ${relativePath}.`);
  }

  const output = path.resolve(destinationRoot, ...relativePath.split('/'));
  if (output !== destinationRoot && !output.startsWith(`${destinationRoot}${path.sep}`)) {
    throw new Error(`Unsafe ZIP entry escaped its destination: ${entry.fileName}.`);
  }
  if (directory) {
    await mkdir(output, { recursive: true });
    return;
  }
  await mkdir(path.dirname(output), { recursive: true });
  const input = await openEntryStream(zip, entry);
  await pipeline(input, createWriteStream(output, { flags: 'wx' }));
  files.push(relativePath);
}

function safeEntryPath(fileName) {
  if (
    fileName.includes('\\') ||
    fileName.includes('\0') ||
    fileName.startsWith('/') ||
    /^[A-Za-z]:/.test(fileName)
  ) {
    throw new Error(`Unsafe absolute or platform-specific ZIP entry: ${fileName}.`);
  }
  const segments = fileName.split('/');
  if (segments.includes('..')) {
    throw new Error(`Unsafe parent traversal ZIP entry: ${fileName}.`);
  }
  const normalized = path.posix.normalize(fileName);
  if (!normalized || normalized === '.' || normalized.startsWith('../')) {
    throw new Error(`Unsafe ZIP entry: ${fileName}.`);
  }
  return normalized.replace(/\/$/, '');
}

async function publishTransaction(options) {
  let windowsBackedUp = false;
  let manifestBackedUp = false;
  let windowsCopyStarted = false;
  let manifestPublished = false;
  try {
    if (await exists(options.windowsDestination)) {
      await options.renameFile(options.windowsDestination, options.windowsBackup);
      windowsBackedUp = true;
    }
    if (await exists(options.manifestPath)) {
      await options.renameFile(options.manifestPath, options.manifestBackup);
      manifestBackedUp = true;
    }
    windowsCopyStarted = true;
    await options.copyDirectory(options.stagedWindows, options.windowsDestination, {
      recursive: true,
      force: false,
      errorOnExist: true,
    });
    await options.renameFile(options.stagedManifest, options.manifestPath);
    manifestPublished = true;
  } catch (error) {
    try {
      if (manifestPublished) await rm(options.manifestPath, { force: true });
      if (manifestBackedUp) await options.renameFile(options.manifestBackup, options.manifestPath);
      if (windowsCopyStarted) {
        await rm(options.windowsDestination, {
          recursive: true,
          force: true,
          maxRetries: 10,
          retryDelay: 100,
        });
      }
      if (windowsBackedUp) await options.renameFile(options.windowsBackup, options.windowsDestination);
    } catch (rollbackError) {
      throw new AggregateError([error, rollbackError],
        `Windows bundle publication and rollback failed. Any remaining originals are preserved at ${options.windowsBackup} and ${options.manifestBackup}.`);
    }
    throw error;
  }
}

function legacyPlatforms(manifest) {
  const worker = manifest?.workers?.['darwin-arm64'];
  if (!worker) return {};
  return {
    'darwin-arm64': {
      modes: {
        auto: { bundle: 'default', backend: 'metal' },
        cpu: { bundle: 'default', backend: 'cpu' },
      },
      bundles: {
        default: {
          executable: worker.path,
          files: [{ path: worker.path, sha256: worker.sha256 }],
        },
      },
    },
  };
}

function openZip(archivePath) {
  return new Promise((resolve, reject) => {
    yauzl.open(archivePath, {
      autoClose: false,
      lazyEntries: true,
      strictFileNames: true,
      validateEntrySizes: true,
    }, (error, zip) => {
      if (error) reject(error);
      else resolve(zip);
    });
  });
}

function openEntryStream(zip, entry) {
  return new Promise((resolve, reject) => {
    zip.openReadStream(entry, (error, stream) => {
      if (error) reject(error);
      else resolve(stream);
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
