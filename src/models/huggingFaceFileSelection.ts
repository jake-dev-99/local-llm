export function selectableHuggingFaceFiles<T extends { filename: string }>(files: T[]): T[] {
  return files.filter((file) =>
    isSingleFileGguf(file.filename) && !isGgufSidecar(file.filename),
  );
}

/**
 * Selects one flat Safetensors checkpoint from repository metadata.
 *
 * The local checkpoint reader is intentionally shallow, so nested variants
 * are not mixed into the root model. Broad sidecar extensions keep tokenizer
 * and processor formats forward-compatible without downloading alternate
 * weight formats or repository documentation.
 */
export function safetensorsHuggingFaceFiles<T extends { filename: string }>(files: T[]): T[] {
  const rootFiles = files.filter((file) =>
    !file.filename.includes('/') &&
    !file.filename.includes('\\') &&
    file.filename !== '.' &&
    file.filename !== '..',
  );
  const hasConfig = rootFiles.some((file) => file.filename === 'config.json');
  const hasWeights = rootFiles.some((file) =>
    file.filename.toLowerCase().endsWith('.safetensors'),
  );
  if (!hasConfig || !hasWeights) {
    return [];
  }
  return rootFiles.filter((file) => isSafetensorsCheckpointFile(file.filename));
}

export function isSingleFileGguf(filename: string): boolean {
  return filename.toLowerCase().endsWith('.gguf') &&
    !/-\d{5}-of-\d{5}\.gguf$/i.test(filename);
}

/**
 * GGUF files that repositories ship beside the models but that do not load as
 * one: vision projectors (`mmproj`) and importance matrices (`imatrix`).
 * Publishers put the token at the start (`mmproj-F16.gguf`), after a dash
 * (`llava-v1.5-7b-mmproj-model-f16.gguf`, `...-imatrix.gguf`) or after a dot
 * (`Qwen2.5-VL-7B-Instruct.mmproj-f16.gguf`).
 */
function isGgufSidecar(filename: string): boolean {
  const basename = filename.slice(filename.lastIndexOf('/') + 1);
  return /(?:^|[-_.])(?:mmproj|imatrix)(?:[-_.]|$)/i.test(basename);
}

function isSafetensorsCheckpointFile(filename: string): boolean {
  const lower = filename.toLowerCase();
  return [
    '.safetensors',
    '.json',
    '.jinja',
    '.model',
    '.txt',
    '.tiktoken',
    '.vocab',
  ].some((extension) => lower.endsWith(extension));
}
