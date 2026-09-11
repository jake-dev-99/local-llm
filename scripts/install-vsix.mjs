import { spawn } from 'node:child_process';
import { access, readFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

import { prepareVsixOutput } from './vsix-output.mjs';

const root = path.resolve(import.meta.dirname, '..');
const supportedTargets = new Set(['darwin-arm64', 'win32-x64']);

export async function installVsix(
  target = `${process.platform}-${process.arch}`,
  dependencies = {},
) {
  if (!supportedTargets.has(target)) {
    throw new Error('Usage: node scripts/install-vsix.mjs [darwin-arm64|win32-x64]');
  }
  const options = {
    root,
    readPackage: readFile,
    accessFile: access,
    run: runCommand,
    log: console.log,
    platform: process.platform,
    commandProcessor: process.env.ComSpec ?? 'cmd.exe',
    ...dependencies,
  };
  const packageManifest = JSON.parse(
    await options.readPackage(path.join(options.root, 'package.json'), 'utf8'),
  );
  const vsixPath = await prepareVsixOutput(options.root, packageManifest.version, target);
  await options.accessFile(vsixPath).catch(() => {
    throw new Error(`VSIX package is missing: ${vsixPath}. Run npm run package first.`);
  });
  const invocation = prepareCodeInvocation(vsixPath, options.platform, options.commandProcessor);
  await options.run(invocation.command, invocation.args, options.root, invocation.spawnOptions);
  options.log(`Installed ${vsixPath}. Reload VS Code to activate this version.`);
}

function prepareCodeInvocation(vsixPath, platform, commandProcessor) {
  if (platform === 'win32') {
    return {
      command: commandProcessor,
      args: ['/d', '/s', '/c', `code --install-extension "${vsixPath}" --force`],
      spawnOptions: { windowsVerbatimArguments: true },
    };
  }
  return {
    command: 'code',
    args: ['--install-extension', vsixPath, '--force'],
    spawnOptions: undefined,
  };
}

async function runCommand(command, args, cwd, spawnOptions) {
  await new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd,
      stdio: 'inherit',
      shell: false,
      ...spawnOptions,
    });
    child.once('error', (error) => {
      reject(new Error(`Failed to invoke the VS Code CLI: ${error.message}. Ensure 'code' is on PATH.`));
    });
    child.once('exit', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`VS Code CLI exited with code ${code ?? 'unknown'}.`));
      }
    });
  });
}

const entry = process.argv[1] ? path.resolve(process.argv[1]) : undefined;
if (entry === fileURLToPath(import.meta.url)) {
  await installVsix(process.argv[2]);
}