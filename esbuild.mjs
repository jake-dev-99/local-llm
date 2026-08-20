import { build, context } from 'esbuild';

const options = {
  entryPoints: ['src/extension.ts'],
  bundle: true,
  platform: 'node',
  format: 'cjs',
  target: 'node22',
  outfile: 'dist/extension.js',
  external: ['vscode'],
  sourcemap: true,
  sourcesContent: true,
};

if (process.argv.includes('--watch')) {
  const buildContext = await context(options);
  await buildContext.watch();
  console.log('Watching Local LLM Engine sources...');
} else {
  await build(options);
}
