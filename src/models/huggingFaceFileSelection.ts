export function selectableHuggingFaceFiles<T extends { filename: string }>(files: T[]): T[] {
  return files.filter((file) => isSingleFileGguf(file.filename));
}

export function isSingleFileGguf(filename: string): boolean {
  return filename.toLowerCase().endsWith('.gguf') &&
    !/-\d{5}-of-\d{5}\.gguf$/i.test(filename);
}
