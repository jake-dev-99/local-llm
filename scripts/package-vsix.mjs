import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import path from 'node:path';
import process from 'node:process';

import { prepareVsceInvocation, prepareVsixOutput } from './vsix-output.mjs';
import { prepareTargetWorkers, targetIgnoreEntries } from './package-workers.mjs';

const root = path.resolve(import.meta.dirname, '..');
const target = process.argv[2] ?? `${process.platform}-${process.arch}`;
if (!['darwin-arm64', 'win32-x64'].includes(target)) {
  throw new Error('Usage: npm run package:vsix -- [darwin-arm64|win32-x64]');
}
await prepareTargetWorkers(root, target);

await run(process.execPath, ['esbuild.mjs']);
const packageDirectory = path.join(root, 'build');
await mkdir(packageDirectory, { recursive: true });
const ignoreFile = path.join(packageDirectory, `.vscodeignore-${target}`);
await writeFile(
  ignoreFile,
  (await targetIgnoreEntries(root, target)).join('\n'),
);
const packageManifest = JSON.parse(await readFile(path.join(root, 'package.json'), 'utf8'));
const output = await prepareVsixOutput(root, packageManifest.version, target);
const vsce = prepareVsceInvocation(root, [
  'package',
  '--target', target,
  '--ignoreFile', ignoreFile,
  '--no-dependencies',
  '--allow-missing-repository',
  '--out', output,
]);
await run(vsce.command, vsce.args);
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
