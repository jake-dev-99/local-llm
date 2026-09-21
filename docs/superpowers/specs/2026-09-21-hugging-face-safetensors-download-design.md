# Hugging Face Safetensors Download Design

**Status:** Approved on 2026-09-21.

## Goal

Allow **Local LLM: Download from Hugging Face** to install complete, flat
Safetensors checkpoint repositories. Keep the existing GGUF flow and existing
runtime selection unchanged.

The motivating repository is
`unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit`. It contains five Safetensors shards,
`config.json`, a shard index, tokenizer files, a chat template, and processor
configuration. It contains no GGUF file, so the current metadata filter rejects
it before any download starts.

## Repository inspection

The Hugging Face metadata request keeps every repository file and pins the
returned immutable revision. Model selection classifies those files afterward.

A downloadable Safetensors checkpoint requires root-level `config.json` and at
least one root-level `.safetensors` file. The downloaded set contains every
root-level Safetensors shard and model sidecar ending in `.json`, `.jinja`,
`.model`, `.txt`, `.tiktoken`, or `.vocab`. Documentation, Git metadata,
alternate weight formats, and nested artifacts are excluded.

Pure Safetensors repositories start the checkpoint download directly. Pure
GGUF repositories retain the existing file picker. Mixed repositories show a
single picker containing the Safetensors checkpoint and every supported GGUF
file.

## Download and registration

The destination directory is stable for repository, revision, and format.
Each file uses the existing resumable HTTPS downloader and expected Hugging
Face SHA-256 when available. A completed file is reused when its expected hash
matches, or when its immutable revision and reported size match.

Cancellation and network failures preserve completed files and `.partial`
state. A retry resumes the same revision. The extension registers nothing
until every selected file exists and the directory passes existing static
Safetensors inspection.

The registered model records its repository, immutable revision, managed
ownership, source URL, and complete directory fingerprint. Removal deletes the
extension-owned directory through the existing Safetensors removal path.

## Runtime boundary

This change adds downloading and registration only. It does not add MLX,
`mlx-vlm`, architecture support, or a new worker. Registered checkpoints keep
the current `transformers` runtime assignment and existing validation behavior.

## Verification

Tests cover file classification, the exact motivating repository layout,
revision-pinned URLs, directory registration, unrelated-file exclusion, and
reuse of completed files. Existing GGUF selection tests remain unchanged.

