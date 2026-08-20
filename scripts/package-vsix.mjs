import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import path from 'node:path';
import process from 'node:process';

const root = path.resolve(import.meta.dirname, '..');
const target = process.argv[2] ?? `${process.platform}-${process.arch}`;
if (!['darwin-arm64', 'win32-x64'].includes(target)) {
  throw new Error('Usage: npm run package:vsix -- [darwin-arm64|win32-x64]');
}
const workerManifest = JSON.parse(
  await readFile(path.join(root, 'resources', 'workers', 'manifest.json'), 'utf8'),
);
const workerEntry = workerManifest.workers?.[target];
if (!workerEntry || !workerEntry.path?.startsWith(`resources/workers/${target}/`)) {
  throw new Error(`Worker manifest has no valid ${target} entry.`);
}
const worker = path.join(root, workerEntry.path);
try {
  await stat(worker);
} catch {
  throw new Error(`Missing ${target} worker. Build it on that platform with npm run build:worker.`);
}
const workerSha256 = await sha256File(worker);
if (workerSha256.toLowerCase() !== workerEntry.sha256?.toLowerCase()) {
  throw new Error(
    `${target} worker SHA-256 ${workerSha256} does not match resources/workers/manifest.json.`,
  );
}

await run(process.execPath, ['esbuild.mjs']);
const packageDirectory = path.join(root, '.build');
await mkdir(packageDirectory, { recursive: true });
const ignoreFile = path.join(packageDirectory, `.vscodeignore-${target}`);
const otherTarget = target === 'darwin-arm64' ? 'win32-x64' : 'darwin-arm64';
await writeFile(
  ignoreFile,
  [
    '.build/**',
    '.git/**',
    '.gitignore',
    '.vscodeignore',
    '.github/**',
    '.agents/**',
    '.codex/**',
    '**/.DS_Store',
    'docs/**',
    'node_modules/**',
    'src/**',
    'scripts/**',
    'tsconfig.json',
    'esbuild.mjs',
    'package-lock.json',
    `resources/workers/${otherTarget}/**`,
    '*.vsix',
  ].join('\n'),
);
const packageManifest = JSON.parse(await readFile(path.join(root, 'package.json'), 'utf8'));
const output = path.join(root, `local-llm-engine-${packageManifest.version}-${target}.vsix`);
await run(path.join(root, 'node_modules', '.bin', 'vsce'), [
  'package',
  '--target', target,
  '--ignoreFile', ignoreFile,
  '--no-dependencies',
  '--allow-missing-repository',
  '--out', output,
]);
console.log(`Created ${output}`);

async function run(command, args) {
  await new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd: root, stdio: 'inherit', shell: false });
    child.once('error', reject);
    child.once('exit', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`${path.basename(command)} exited with code ${code ?? 'unknown'}.`));
      }
    });
  });
}

async function sha256File(filePath) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(filePath)) {
    hash.update(chunk);
  }
  return hash.digest('hex');
}
