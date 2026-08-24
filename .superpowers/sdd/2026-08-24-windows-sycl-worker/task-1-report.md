# Task 1 report: versioned worker-manifest contract

## Implementation summary

Added the version-2 worker manifest contract and its standalone tests. The implementation validates manifest structure, acceleration/backend selections, SHA-256 values, executable membership, duplicate files, and safe POSIX-normalized platform/bundle paths. It provides immutable bundle resolution/replacement, file hashing, bundle verification, and target-scoped platform verification. Existing production runtime wiring remains unchanged.

## Files changed

- `src/worker/workerManifest.ts` — manifest types, parser, selectors, replacement, hashing, and verification.
- `src/worker/workerManifest.test.ts` — resolution, safety, and integrity tests with standalone fixtures.

## RED

Command:

```text
node --disable-warning=MODULE_TYPELESS_PACKAGE_JSON --test src/worker/workerManifest.test.ts
```

Output: failed as expected with `ERR_MODULE_NOT_FOUND` because `src/worker/workerManifest.ts` did not exist.

## GREEN

Focused command:

```text
node --disable-warning=MODULE_TYPELESS_PACKAGE_JSON --test src/worker/workerManifest.test.ts
```

Result: 6 tests passed, 0 failed.

Full-suite command:

```text
npm test
```

Result: 81 tests passed, 0 failed.

Additional type validation: `npx tsc --noEmit` passed; `git diff --check` passed.

## Self-review

Reviewed the implementation against the brief: exported names and literal unions match; selectors return the selected bundle/backend; Darwin CPU and auto share the bundle; path normalization prevents traversal and cross-bundle paths; executable inclusion and duplicate detection are enforced; hashes are compared case-insensitively; platform verification only walks the requested target; replacement clones before parsing. No unrelated files were changed.

## Concerns

No blocking concerns. The shell emits an existing `/Users/jake/.zshenv:7: unmatched \`` startup warning during commands; it did not affect test or typecheck results.

## Fix Round 1

Changed `resolveWorkerBundle` to clone each `WorkerFile` record before returning the resolved bundle, preventing callers from mutating manifest state.

Covering test: `resolved bundle files are isolated from the manifest` in `src/worker/workerManifest.test.ts`.

RED evidence (before the fix):

```text
node --disable-warning=MODULE_TYPELESS_PACKAGE_JSON --test src/worker/workerManifest.test.ts
```

Result: 6 passed, 1 failed. The regression observed `changed.exe` in the subsequent manifest resolution instead of `llama-server.exe`.

GREEN evidence (after the fix):

```text
node --disable-warning=MODULE_TYPELESS_PACKAGE_JSON --test src/worker/workerManifest.test.ts
npx tsc --noEmit
```

Result: 7 tests passed, 0 failed; TypeScript check passed.
