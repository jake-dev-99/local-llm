# Hugging Face Safetensors Download Implementation Plan

**Goal:** Download complete flat Safetensors repositories from Hugging Face.

**Spec:** `docs/superpowers/specs/2026-09-21-hugging-face-safetensors-download-design.md`

## Task 1: Classify repository files

- Add failing tests for the motivating repository layout.
- Select root Safetensors shards and supported sidecars.
- Preserve existing single-file GGUF selection.

## Task 2: Download and register checkpoint directories

- Add a failing manager test with mocked Hugging Face metadata and files.
- Keep all repository files during metadata inspection.
- Download the selected checkpoint into revision-stable managed storage.
- Reuse verified completed files and preserve resumable partial files.
- Register repository and revision metadata after static inspection succeeds.

## Task 3: Update user-facing text and documentation

- Remove GGUF-only wording from the Hugging Face command.
- Document both GGUF and Safetensors repository downloads.

## Task 4: Verify

- Run focused classification and manager tests.
- Run the complete Node test suite, typecheck, and build.
- Package the macOS extension and inspect its bundled download logic.

