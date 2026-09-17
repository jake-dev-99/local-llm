import { createHash } from 'node:crypto';
import { createReadStream, createWriteStream } from 'node:fs';
import { mkdir, readFile, rename, rm, stat, statfs, writeFile } from 'node:fs/promises';
import * as path from 'node:path';
import { pipeline } from 'node:stream/promises';
import { Transform } from 'node:stream';
import * as vscode from 'vscode';

export interface HuggingFaceFile {
  filename: string;
  size: number;
  sha256?: string;
}

export interface HuggingFaceRepository {
  id: string;
  revision: string;
  files: HuggingFaceFile[];
}

interface HuggingFaceApiModel {
  id?: string;
  sha?: string;
  siblings?: Array<{
    rfilename?: string;
    size?: number;
    lfs?: { sha256?: string; size?: number };
  }>;
  error?: string;
}

interface PartialDownloadMetadata {
  etag?: string;
  lastModified?: string;
}

const activeDownloadDestinations = new Set<string>();

export async function inspectHuggingFaceRepository(
  repository: string,
  token: string | undefined,
  signal?: AbortSignal,
): Promise<HuggingFaceRepository> {
  const normalized = normalizeRepository(repository);
  const response = await fetch(
    `https://huggingface.co/api/models/${encodeRepository(normalized)}?blobs=true`,
    {
      headers: authorizationHeaders(token),
      ...(signal ? { signal } : {}),
    },
  );
  const responseText = await response.text();
  let payload: HuggingFaceApiModel;
  try {
    payload = JSON.parse(responseText) as HuggingFaceApiModel;
  } catch {
    payload = {};
  }
  if (!response.ok) {
    throw new Error(
      payload.error ||
      responseText.trim().slice(0, 300) ||
      `Hugging Face returned HTTP ${response.status}.`,
    );
  }
  if (!payload.sha) {
    throw new Error('Hugging Face metadata did not include a repository revision.');
  }
  const files = (payload.siblings ?? [])
    .filter((item) => item.rfilename?.toLowerCase().endsWith('.gguf'))
    .map((item) => ({
      filename: item.rfilename as string,
      size: item.lfs?.size ?? item.size ?? 0,
      ...(item.lfs?.sha256 ? { sha256: item.lfs.sha256 } : {}),
    }))
    .sort((a, b) => a.filename.localeCompare(b.filename));
  if (files.length === 0) {
    throw new Error(`Repository ${normalized} contains no GGUF files.`);
  }
  return { id: payload.id ?? normalized, revision: payload.sha, files };
}

export function huggingFaceDownloadUrl(
  repository: string,
  revision: string,
  filename: string,
): string {
  return `https://huggingface.co/${encodeRepository(normalizeRepository(repository))}/resolve/${encodeURIComponent(revision)}/${filename
    .split('/')
    .map(encodeURIComponent)
    .join('/')}`;
}

export async function copyLocalModel(
  sourcePath: string,
  destinationPath: string,
  progress: vscode.Progress<{ message?: string; increment?: number }>,
  token: vscode.CancellationToken,
): Promise<{ sha256: string; size: number }> {
  const source = await stat(sourcePath);
  const hash = createHash('sha256');
  let copied = 0;
  let lastPercent = 0;
  await mkdir(path.dirname(destinationPath), { recursive: true });
  const temporaryPath = `${destinationPath}.partial`;
  try {
    const hasher = new Transform({
      transform(chunk: Buffer, _encoding, callback) {
        if (token.isCancellationRequested) {
          callback(new vscode.CancellationError());
          return;
        }
        hash.update(chunk);
        copied += chunk.length;
        const percent = source.size > 0 ? (copied / source.size) * 100 : 100;
        progress.report({
          message: `${formatBytes(copied)} of ${formatBytes(source.size)}`,
          increment: Math.max(0, percent - lastPercent),
        });
        lastPercent = percent;
        callback(null, chunk);
      },
    });
    await pipeline(createReadStream(sourcePath), hasher, createWriteStream(temporaryPath));
    await rename(temporaryPath, destinationPath);
    return { sha256: hash.digest('hex'), size: copied };
  } catch (error) {
    await rm(temporaryPath, { force: true });
    throw error;
  }
}

export async function downloadModel(
  url: string,
  destinationPath: string,
  expectedSha256: string | undefined,
  bearerToken: string | undefined,
  progress: vscode.Progress<{ message?: string; increment?: number }>,
  token: vscode.CancellationToken,
): Promise<{ sha256: string; size: number }> {
  if (activeDownloadDestinations.has(destinationPath)) {
    throw new Error('A download for this model is already in progress.');
  }
  activeDownloadDestinations.add(destinationPath);
  try {
    return await downloadModelExclusive(
      url,
      destinationPath,
      expectedSha256,
      bearerToken,
      progress,
      token,
    );
  } finally {
    activeDownloadDestinations.delete(destinationPath);
  }
}

async function downloadModelExclusive(
  url: string,
  destinationPath: string,
  expectedSha256: string | undefined,
  bearerToken: string | undefined,
  progress: vscode.Progress<{ message?: string; increment?: number }>,
  token: vscode.CancellationToken,
): Promise<{ sha256: string; size: number }> {
  const parsed = new URL(url);
  if (parsed.protocol !== 'https:') {
    throw new Error('Model downloads require an HTTPS URL.');
  }
  const controller = new AbortController();
  const cancellation = token.onCancellationRequested(() => controller.abort());
  const temporaryPath = `${destinationPath}.partial`;
  const metadataPath = `${temporaryPath}.json`;
  await mkdir(path.dirname(destinationPath), { recursive: true });
  try {
    let existingSize = await stat(temporaryPath).then((value) => value.size).catch(() => 0);
    let partialMetadata = existingSize > 0
      ? await readPartialMetadata(metadataPath)
      : undefined;
    let validator = resumeValidator(partialMetadata);
    if (existingSize > 0 && !validator) {
      await Promise.all([
        rm(temporaryPath, { force: true }),
        rm(metadataPath, { force: true }),
      ]);
      existingSize = 0;
      partialMetadata = undefined;
    }
    let response = await fetchDownload(
      parsed,
      bearerToken,
      existingSize,
      validator,
      controller.signal,
    );
    if (existingSize > 0 && response.status !== 206) {
      await response.body?.cancel();
      await Promise.all([
        rm(temporaryPath, { force: true }),
        rm(metadataPath, { force: true }),
      ]);
      existingSize = 0;
      partialMetadata = undefined;
      validator = undefined;
      response = await fetchDownload(parsed, bearerToken, 0, undefined, controller.signal);
    }
    if ((response.status !== 200 && response.status !== 206) || !response.body) {
      throw new Error(`Model download failed with HTTP ${response.status}.`);
    }
    if (response.status === 206) {
      const rangeStart = contentRangeStart(response.headers.get('content-range'));
      if (rangeStart !== existingSize || !responseMatchesMetadata(response, partialMetadata)) {
        await response.body.cancel();
        await Promise.all([
          rm(temporaryPath, { force: true }),
          rm(metadataPath, { force: true }),
        ]);
        throw new Error(
          'The remote model changed or returned an invalid resume range. Run the download again to restart safely.',
        );
      }
    }
    partialMetadata = metadataFromResponse(response, partialMetadata);
    await writeFile(metadataPath, JSON.stringify(partialMetadata), { encoding: 'utf8', mode: 0o600 });
    const responseBytes = finiteContentLength(response.headers.get('content-length'));
    const total = contentRangeTotal(response.headers.get('content-range')) ??
      (responseBytes > 0 ? existingSize + responseBytes : 0);
    await assertDiskSpace(path.dirname(destinationPath), responseBytes || undefined);
    let downloaded = existingSize;
    let lastPercent = total > 0 ? (downloaded / total) * 100 : 0;
    const counter = new Transform({
      transform(chunk: Buffer, _encoding, callback) {
        downloaded += chunk.length;
        const percent = total > 0 ? (downloaded / total) * 100 : 0;
        progress.report({
          message: total > 0
            ? `${formatBytes(downloaded)} of ${formatBytes(total)}`
            : formatBytes(downloaded),
          increment: Math.max(0, percent - lastPercent),
        });
        lastPercent = percent;
        callback(null, chunk);
      },
    });
    await pipeline(
      // Node's bundled fetch exposes a Web ReadableStream compatible with this cast.
      response.body as unknown as NodeJS.ReadableStream,
      counter,
      createWriteStream(temporaryPath, { flags: existingSize > 0 ? 'a' : 'w' }),
      { signal: controller.signal },
    );
    const completed = await stat(temporaryPath);
    if (total > 0 && completed.size !== total) {
      throw new Error(
        `Model download ended at ${formatBytes(completed.size)}, expected ${formatBytes(total)}. Run the download again to resume.`,
      );
    }
    const sha256 = await sha256File(temporaryPath);
    if (expectedSha256 && sha256.toLowerCase() !== expectedSha256.toLowerCase()) {
      await Promise.all([
        rm(temporaryPath, { force: true }),
        rm(metadataPath, { force: true }),
      ]);
      throw new Error(
        `Downloaded model checksum mismatch. Expected ${expectedSha256}, received ${sha256}.`,
      );
    }
    await rename(temporaryPath, destinationPath);
    await rm(metadataPath, { force: true });
    return { sha256, size: completed.size };
  } catch (error) {
    if (controller.signal.aborted) {
      throw new vscode.CancellationError();
    }
    throw error;
  } finally {
    cancellation.dispose();
  }
}

export function sourceUrlForRegistry(value: string): string {
  try {
    const url = new URL(value);
    url.username = '';
    url.password = '';
    url.search = '';
    url.hash = '';
    return url.toString();
  } catch {
    return value;
  }
}

export function safeModelFilename(value: string, allowedExtensions: readonly string[] = ['.gguf']): string {
  const filename = path.basename(value).replace(/[^a-zA-Z0-9._-]+/g, '-');
  if (!allowedExtensions.some((extension) => filename.toLowerCase().endsWith(extension))) {
    throw new Error(`The selected model must be a ${allowedExtensions.join(' or ')} file.`);
  }
  return filename;
}

export function formatBytes(value: number): string {
  if (!Number.isFinite(value) || value <= 0) {
    return '0 B';
  }
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  const order = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
  return `${(value / 1024 ** order).toFixed(order === 0 ? 0 : 1)} ${units[order]}`;
}

function normalizeRepository(value: string): string {
  const trimmed = value.trim().replace(/^https:\/\/huggingface\.co\//i, '').replace(/\/$/, '');
  if (!/^[\w.-]+\/[\w.-]+$/.test(trimmed)) {
    throw new Error('Enter a Hugging Face repository as owner/name.');
  }
  return trimmed;
}

function encodeRepository(repository: string): string {
  return repository.split('/').map(encodeURIComponent).join('/');
}

function authorizationHeaders(token: string | undefined): HeadersInit {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function fetchDownload(
  url: URL,
  bearerToken: string | undefined,
  start: number,
  validator: string | undefined,
  signal: AbortSignal,
): Promise<Response> {
  const headers = new Headers(authorizationHeaders(bearerToken));
  if (start > 0) {
    headers.set('Range', `bytes=${start}-`);
    if (validator) {
      headers.set('If-Range', validator);
    }
  }
  const response = await fetch(url, { headers, signal, redirect: 'follow' });
  if (new URL(response.url).protocol !== 'https:') {
    await response.body?.cancel();
    throw new Error('Model download redirected to a non-HTTPS URL.');
  }
  return response;
}

async function readPartialMetadata(
  metadataPath: string,
): Promise<PartialDownloadMetadata | undefined> {
  try {
    const parsed = JSON.parse(await readFile(metadataPath, 'utf8')) as unknown;
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
      return undefined;
    }
    const record = parsed as Record<string, unknown>;
    return {
      ...(typeof record.etag === 'string' ? { etag: record.etag } : {}),
      ...(typeof record.lastModified === 'string'
        ? { lastModified: record.lastModified }
        : {}),
    };
  } catch {
    return undefined;
  }
}

function metadataFromResponse(
  response: Response,
  previous: PartialDownloadMetadata | undefined,
): PartialDownloadMetadata {
  const etag = strongEtag(response.headers.get('etag')) ?? previous?.etag;
  const lastModified = response.headers.get('last-modified') ?? previous?.lastModified;
  return {
    ...(etag ? { etag } : {}),
    ...(lastModified ? { lastModified } : {}),
  };
}

function resumeValidator(metadata: PartialDownloadMetadata | undefined): string | undefined {
  return metadata?.etag ?? metadata?.lastModified;
}

function responseMatchesMetadata(
  response: Response,
  metadata: PartialDownloadMetadata | undefined,
): boolean {
  const responseEtag = strongEtag(response.headers.get('etag'));
  if (metadata?.etag) {
    return responseEtag === metadata.etag;
  }
  const responseModified = response.headers.get('last-modified');
  return Boolean(
    metadata?.lastModified &&
    responseModified &&
    metadata.lastModified === responseModified,
  );
}

function strongEtag(value: string | null): string | undefined {
  return value && !value.startsWith('W/') ? value : undefined;
}

function finiteContentLength(value: string | null): number {
  const parsed = Number(value ?? 0);
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : 0;
}

function contentRangeStart(value: string | null): number | undefined {
  const match = /^bytes\s+(\d+)-\d+\/(?:\d+|\*)$/i.exec(value ?? '');
  return match?.[1] ? Number(match[1]) : undefined;
}

function contentRangeTotal(value: string | null): number | undefined {
  const match = /^bytes\s+\d+-\d+\/(\d+)$/i.exec(value ?? '');
  return match?.[1] ? Number(match[1]) : undefined;
}

async function assertDiskSpace(directory: string, requiredBytes: number | undefined): Promise<void> {
  if (!requiredBytes || requiredBytes <= 0) {
    return;
  }
  try {
    const available = await statfs(directory);
    const availableBytes = Number(available.bavail) * Number(available.bsize);
    const reserve = 64 * 1024 ** 2;
    if (Number.isFinite(availableBytes) && availableBytes < requiredBytes + reserve) {
      throw new Error(
        `Not enough free disk space for this model. ${formatBytes(requiredBytes + reserve)} is required and ${formatBytes(availableBytes)} is available.`,
      );
    }
  } catch (error) {
    if (error instanceof Error && error.message.startsWith('Not enough free disk space')) {
      throw error;
    }
    // Some remote filesystems do not support statfs. The streamed download still
    // fails safely without replacing a completed model.
  }
}

export async function sha256File(filePath: string): Promise<string> {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(filePath)) {
    hash.update(chunk as Buffer);
  }
  return hash.digest('hex');
}
