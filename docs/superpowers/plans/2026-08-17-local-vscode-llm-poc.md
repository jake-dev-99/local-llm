# Local VS Code LLM Engine PoC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a usable VS Code extension that downloads or imports GGUF models, runs them through a bundled `llama-server`, and provides local Chat, Agent, and inline completion.

**Architecture:** One shared TypeScript extension manages model storage and a single platform-specific `llama-server` child process. The extension connects only through authenticated loopback HTTP and maps the worker protocol into stable VS Code language-model APIs.

**Tech Stack:** TypeScript 7.0.2, Node.js APIs, VS Code API types 1.125.0, esbuild 0.28.2, `llama.cpp` commit `60eeeb6082c1126bb8bc72902c83123cd056811b` (release `b10472`), CMake, GGUF.

**Spec:** `docs/superpowers/specs/2026-08-17-local-vscode-llm-poc-design.md`

**Implementation status (2026-08-17):** Version 0.2.0 is implemented and packaged
for both target platforms. Sixteen focused tests, TypeScript typechecking,
extension bundling, VSIX content inspection, worker-integrity verification, and
live Qwen 7B Chat/FIM/structured-tool/tool-result inference pass on macOS. The
Windows executable was cross-built and structurally audited, but execution on
Intel Windows hardware remains an explicit platform-validation step. Release
signing remains deferred by operator direction.

## Global Constraints

- Support `darwin-arm64` with Metal and `win32-x64` with CPU inference.
- Bundle `llama-server`; never discover or download an inference binary at runtime.
- Bind only to `127.0.0.1` with an ephemeral API key.
- Support GGUF models only.
- Support local import, Hugging Face download, and direct HTTPS download.
- Never send prompts or source code beyond loopback.
- Collect no telemetry.
- Use no runtime dependencies beyond Node.js and VS Code APIs.
- Keep tests focused on prototype-critical budgeting, scheduling, parsing, and
  runtime contracts; release automation remains out of scope.
- Typecheck, bundle, build the current-platform worker, and package the current-platform VSIX.

---

## Planned File Structure

```text
.
├── .gitignore
├── .vscodeignore
├── LICENSE
├── README.md
├── package.json
├── package-lock.json
├── tsconfig.json
├── esbuild.mjs
├── scripts/
│   └── build-worker.mjs
├── resources/
│   └── workers/
│       ├── darwin-arm64/llama-server
│       └── win32-x64/llama-server.exe
└── src/
    ├── extension.ts
    ├── config.ts
    ├── domain.ts
    ├── logging.ts
    ├── models/
    │   ├── modelManager.ts
    │   ├── modelRegistry.ts
    │   └── modelSources.ts
    ├── worker/
    │   ├── llamaClient.ts
    │   └── workerManager.ts
    ├── provider/
    │   ├── messageAdapter.ts
    │   └── localLanguageModelProvider.ts
    ├── completion/
    │   └── localInlineCompletionProvider.ts
    └── ui/
        └── modelCommands.ts
```

### Task 1: Project Foundation and Shared Domain

**Files:**
- Create: `package.json`
- Create: `package-lock.json`
- Create: `tsconfig.json`
- Create: `esbuild.mjs`
- Create: `.gitignore`
- Create: `.vscodeignore`
- Create: `LICENSE`
- Create: `src/domain.ts`
- Create: `src/config.ts`
- Create: `src/logging.ts`

**Interfaces:**
- Produces: `InstalledModel`, `ModelCapabilities`, `WorkerState`, `WorkerConfig`, `LocalLlmConfig`, `readConfig()`, and `LocalLlmLogger`.
- Consumes: VS Code configuration and output-channel APIs.

- [ ] **Step 1: Create the extension manifest**

Use these exact package values:

```json
{
  "name": "local-llm-engine",
  "displayName": "Local LLM Engine",
  "version": "0.2.0",
  "publisher": "local-llm",
  "license": "MIT",
  "engines": { "vscode": "^1.125.0" },
  "main": "./dist/extension.js",
  "extensionKind": ["ui"],
  "activationEvents": ["onStartupFinished"],
  "contributes": {
    "languageModelChatProviders": [{
      "vendor": "local-llm-engine",
      "displayName": "Local LLM",
      "managementCommand": "localLlm.manageModels"
    }]
  }
}
```

Add commands, settings, scripts, and development dependencies without changing these identifiers.

- [ ] **Step 2: Configure strict TypeScript and esbuild**

Use CommonJS output because VS Code loads `dist/extension.js` through `main`.

```ts
// esbuild.mjs core settings
await build({
  entryPoints: ['src/extension.ts'],
  bundle: true,
  platform: 'node',
  format: 'cjs',
  target: 'node22',
  outfile: 'dist/extension.js',
  external: ['vscode'],
  sourcemap: true,
});
```

- [ ] **Step 3: Define shared domain types**

```ts
export interface InstalledModel {
  id: string;
  name: string;
  filePath: string;
  fileSize: number;
  sha256: string;
  source: 'import' | 'huggingface' | 'url';
  sourceUrl?: string;
  repository?: string;
  revision?: string;
  filename: string;
  installedAt: string;
  capabilities: { toolCalling: boolean; fillInMiddle: boolean };
}

export type WorkerState =
  | { kind: 'stopped' }
  | { kind: 'starting'; modelId: string }
  | { kind: 'ready'; modelId: string; port: number }
  | { kind: 'stopping'; modelId: string }
  | { kind: 'failed'; modelId?: string; message: string };
```

- [ ] **Step 4: Add configuration and logging**

Implement the exact settings defined in the spec.

Use one `Local LLM` Output channel.

Never log prompt bodies, tool results, document contents, or API secrets.

- [ ] **Step 5: Install dependencies and verify the foundation**

```bash
npm install --save-dev --save-exact @types/node@26.2.0 @types/vscode@1.125.0 @vscode/vsce@3.9.2 esbuild@0.28.2 typescript@7.0.2
npm run typecheck
npm run build
```

Expected: both commands exit successfully.

### Task 2: Managed Worker and HTTP Client

**Files:**
- Create: `src/worker/workerManager.ts`
- Create: `src/worker/llamaClient.ts`

**Interfaces:**
- Consumes: `InstalledModel`, `WorkerState`, `LocalLlmConfig`, and `LocalLlmLogger`.
- Produces: `WorkerManager.ensureReady(model)`, `WorkerManager.stop()`, `LlamaClient.chat()`, `LlamaClient.tokenize()`, and `LlamaClient.infill()`.

- [ ] **Step 1: Implement worker lifecycle**

```ts
export class WorkerManager implements vscode.Disposable {
  readonly onDidChangeState: vscode.Event<WorkerState>;
  get state(): WorkerState;
  ensureReady(model: InstalledModel): Promise<LlamaClient>;
  stop(): Promise<void>;
  dispose(): void;
}
```

Resolve the worker exclusively from:

```ts
context.asAbsolutePath(
  process.platform === 'win32'
    ? 'resources/workers/win32-x64/llama-server.exe'
    : 'resources/workers/darwin-arm64/llama-server'
)
```

Spawn with `shell: false` and these arguments:

```text
--model <absolute GGUF path>
--host 127.0.0.1
--port <allocated port>
--api-key <random 32-byte hex secret>
--ctx-size <configured context>
--parallel 1
--jinja
--no-webui
--fit off
--threads <configured value when positive>
--n-gpu-layers 99 on macOS
--n-gpu-layers 0 on Windows
--device none and --no-op-offload when CPU inference is selected
```

Poll `GET /health` every 500 milliseconds for 120 seconds.

Stop gracefully for five seconds before forcing termination.

Retry three unexpected exits within five minutes.

- [ ] **Step 2: Implement authenticated worker requests**

```ts
export class LlamaClient {
  constructor(readonly baseUrl: string, private readonly apiKey: string);
  health(signal?: AbortSignal): Promise<boolean>;
  tokenize(content: string, signal?: AbortSignal): Promise<number>;
  chat(request: ChatRequest, onEvent: (event: ChatStreamEvent) => void, signal?: AbortSignal): Promise<void>;
  infill(request: InfillRequest, signal?: AbortSignal): Promise<string>;
}
```

Use `Authorization: Bearer <secret>` for every non-health request.

Use `/tokenize` with `{ content, add_special: false, with_pieces: false }`.

Use `/v1/chat/completions` with SSE streaming.

Use `/infill` with `input_prefix`, `input_suffix`, `n_predict`, and `stream: false`.

- [ ] **Step 3: Parse SSE and tool calls**

```ts
export type ChatStreamEvent =
  | { kind: 'text'; text: string }
  | { kind: 'toolCall'; id: string; name: string; input: object };
```

Accumulate streamed `tool_calls[].function.arguments` by index.

Emit each tool call after parsing its complete JSON arguments.

- [ ] **Step 4: Verify worker modules compile**

```bash
npm run typecheck
npm run build
```

Expected: both commands exit successfully.

### Task 3: Model Registry, Import, and Downloads

**Files:**
- Create: `src/models/modelRegistry.ts`
- Create: `src/models/modelSources.ts`
- Create: `src/models/modelManager.ts`

**Interfaces:**
- Consumes: extension global storage, `SecretStorage`, `WorkerManager`, and logging.
- Produces: persistent model listing, import, Hugging Face download, direct URL download, removal, and change events.

- [ ] **Step 1: Implement persistent registry**

```ts
export class ModelRegistry {
  readonly onDidChange: vscode.Event<void>;
  list(): readonly InstalledModel[];
  get(id: string): InstalledModel | undefined;
  upsert(model: InstalledModel): Promise<void>;
  remove(id: string): Promise<void>;
}
```

Persist under global-state key `localLlm.installedModels.v1`.

- [ ] **Step 2: Implement model-source adapters**

```ts
export interface ModelDownloadDescriptor {
  name: string;
  filename: string;
  url: string;
  expectedSize?: number;
  expectedSha256?: string;
  headers?: Record<string, string>;
  source: InstalledModel['source'];
  repository?: string;
  revision?: string;
}

export class HuggingFaceSource {
  listGgufFiles(repository: string, token?: string): Promise<ModelDownloadDescriptor[]>;
}

export class DirectUrlSource {
  describe(url: string): ModelDownloadDescriptor;
}
```

Resolve Hugging Face metadata through:

```text
GET https://huggingface.co/api/models/<repository>?blobs=true
```

Read `sha`, `siblings[].rfilename`, `siblings[].size`, and `siblings[].lfs.sha256`.

Download through:

```text
https://huggingface.co/<repository>/resolve/<sha>/<filename>
```

- [ ] **Step 3: Implement safe resumable download**

Download into `<filename>.partial`.

Send `Range: bytes=<partial-size>-` when resuming.

Append only after a `206` response.

Restart from zero after a `200` response.

Compute SHA-256 while reading the final file.

Reject mismatched Hugging Face LFS digests.

Rename atomically only after validation.

- [ ] **Step 4: Implement local import and model removal**

Copy imported files into the managed model directory.

Reject non-GGUF filenames.

Hash each imported model before registration.

Stop an active worker before removing its model.

- [ ] **Step 5: Verify model modules compile**

```bash
npm run typecheck
npm run build
```

Expected: both commands exit successfully.

### Task 4: Native VS Code Chat and Agent Provider

**Files:**
- Create: `src/provider/messageAdapter.ts`
- Create: `src/provider/localLanguageModelProvider.ts`

**Interfaces:**
- Consumes: `ModelRegistry`, `WorkerManager`, and `LlamaClient`.
- Produces: a registered `LanguageModelChatProvider` with text, token counting, and tool calls.

- [ ] **Step 1: Implement message conversion**

```ts
export function toOpenAiMessages(
  messages: readonly vscode.LanguageModelChatRequestMessage[]
): OpenAiMessage[];

export function toOpenAiTools(
  tools: readonly vscode.LanguageModelChatTool[] | undefined
): OpenAiTool[] | undefined;
```

Map `LanguageModelTextPart` into message content.

Map `LanguageModelToolCallPart` into assistant `tool_calls`.

Map `LanguageModelToolResultPart` into `tool` messages.

- [ ] **Step 2: Implement provider information**

```ts
export class LocalLanguageModelProvider implements vscode.LanguageModelChatProvider<InstalledModelInformation> {
  readonly onDidChangeLanguageModelChatInformation: vscode.Event<void>;
  provideLanguageModelChatInformation(
    options: vscode.PrepareLanguageModelChatModelOptions,
    token: vscode.CancellationToken
  ): vscode.ProviderResult<InstalledModelInformation[]>;
  provideLanguageModelChatResponse(
    model: InstalledModelInformation,
    messages: readonly vscode.LanguageModelChatRequestMessage[],
    options: vscode.ProvideLanguageModelChatResponseOptions,
    progress: vscode.Progress<vscode.LanguageModelResponsePart>,
    token: vscode.CancellationToken
  ): Thenable<void>;
  provideTokenCount(
    model: InstalledModelInformation,
    text: string | vscode.LanguageModelChatRequestMessage,
    token: vscode.CancellationToken
  ): Thenable<number>;
}
```

Report each installed model under vendor `local-llm-engine`.

Report `toolCalling: true` when the model registry enables it.

- [ ] **Step 3: Stream text and tool calls**

Emit text with:

```ts
progress.report(new vscode.LanguageModelTextPart(text));
```

Emit tool calls with:

```ts
progress.report(new vscode.LanguageModelToolCallPart(id, name, input));
```

Map `LanguageModelChatToolMode.Required` to `tool_choice: 'required'`.

Map all other modes to `tool_choice: 'auto'`.

- [ ] **Step 4: Verify provider modules compile**

```bash
npm run typecheck
npm run build
```

Expected: both commands exit successfully.

### Task 5: Inline Completion

**Files:**
- Create: `src/completion/localInlineCompletionProvider.ts`

**Interfaces:**
- Consumes: active model metadata and `WorkerManager`.
- Produces: cancellable fill-in-the-middle ghost text.

- [ ] **Step 1: Implement bounded context extraction**

Capture at most 8,000 characters before the cursor.

Capture at most 4,000 characters after the cursor.

Do not read other workspace files.

- [ ] **Step 2: Implement completion requests**

```ts
export class LocalInlineCompletionProvider implements vscode.InlineCompletionItemProvider {
  provideInlineCompletionItems(
    document: vscode.TextDocument,
    position: vscode.Position,
    context: vscode.InlineCompletionContext,
    token: vscode.CancellationToken
  ): vscode.ProviderResult<vscode.InlineCompletionItem[]>;
}
```

Return no completion when inline completion is disabled.

Return no completion when the selected model lacks fill-in-the-middle capability.

Debounce automatic requests for 250 milliseconds.

Request at most 64 output tokens.

Cancel stale requests after document changes.

- [ ] **Step 3: Verify completion compiles**

```bash
npm run typecheck
npm run build
```

Expected: both commands exit successfully.

### Task 6: Commands and Extension Wiring

**Files:**
- Create: `src/ui/modelCommands.ts`
- Create: `src/extension.ts`

**Interfaces:**
- Consumes: every prior component.
- Produces: complete user workflows and clean activation/deactivation.

- [ ] **Step 1: Implement model-management commands**

Register:

```text
localLlm.manageModels
localLlm.importModel
localLlm.downloadFromHuggingFace
localLlm.downloadFromUrl
localLlm.removeModel
localLlm.selectDefaultModel
localLlm.stopWorker
localLlm.showStatus
localLlm.setHuggingFaceToken
```

Use `window.withProgress` for import, hashing, and downloads.

Store the Hugging Face token only through `context.secrets`.

- [ ] **Step 2: Compose the extension**

```ts
export async function activate(context: vscode.ExtensionContext): Promise<void>;
export async function deactivate(): Promise<void>;
```

Create each singleton once.

Register the language provider with `vscode.lm.registerLanguageModelChatProvider`.

Register inline completion with `vscode.languages.registerInlineCompletionItemProvider`.

Stop the worker during deactivation.

- [ ] **Step 3: Add status-bar state**

Display `Local LLM: stopped`, `Local LLM: loading`, or the active model name.

Clicking the status item opens model management.

- [ ] **Step 4: Verify the complete extension compiles**

```bash
npm run typecheck
npm run build
```

Expected: both commands exit successfully.

### Task 7: Reproducible Worker Build

**Files:**
- Create: `scripts/build-worker.mjs`
- Create when building: `resources/workers/darwin-arm64/llama-server`
- Create when building: `resources/workers/win32-x64/llama-server.exe`

**Interfaces:**
- Consumes: Git, CMake, a C++ compiler, and pinned `llama.cpp` source.
- Produces: one static worker for the current platform.

- [ ] **Step 1: Implement pinned source acquisition**

Use this exact revision:

```text
60eeeb6082c1126bb8bc72902c83123cd056811b
```

Clone into `.build/llama.cpp` and checkout detached at that revision.

- [ ] **Step 2: Configure the worker build**

Use these shared CMake options:

```text
-DCMAKE_BUILD_TYPE=Release
-DBUILD_SHARED_LIBS=OFF
-DGGML_STATIC=ON
-DGGML_NATIVE=OFF
-DGGML_OPENMP=OFF
-DGGML_CCACHE=OFF
-DLLAMA_BUILD_TESTS=OFF
-DLLAMA_BUILD_EXAMPLES=OFF
-DLLAMA_BUILD_APP=OFF
-DLLAMA_BUILD_SERVER=ON
-DLLAMA_BUILD_UI=OFF
-DLLAMA_USE_PREBUILT_UI=OFF
-DLLAMA_OPENSSL=OFF
-DLLAMA_LLGUIDANCE=OFF
-DLLAMA_SUBPROCESS=OFF
```

Add these macOS options:

```text
-DGGML_METAL=ON
-DGGML_METAL_EMBED_LIBRARY=ON
```

Add this Windows option:

```text
-DGGML_METAL=OFF
```

- [ ] **Step 3: Build and copy the current worker**

```bash
npm run worker:build
```

Expected on M4: `resources/workers/darwin-arm64/llama-server` exists and is executable.

Expected on Windows: `resources/workers/win32-x64/llama-server.exe` exists.

### Task 8: Documentation and Usable Package

**Files:**
- Create: `README.md`
- Modify: `.vscodeignore`
- Modify: `package.json`

**Interfaces:**
- Consumes: the finished extension and worker.
- Produces: installation and usage instructions plus a current-platform VSIX.

- [ ] **Step 1: Document the product workflow**

Document:

```text
1. Install the platform-specific VSIX.
2. Run Local LLM: Manage Models.
3. Import or download a GGUF.
4. Select the model in VS Code Chat.
5. Use local Chat, Agent, and compatible inline completion.
```

Document model-size and hardware limits.

Document that Windows GPU acceleration and workspace indexing remain excluded.

- [ ] **Step 2: Package only runtime files**

Include `dist/extension.js`, its source map, `package.json`, `README.md`, `LICENSE`, and the current worker.

Exclude TypeScript source, build source, `.build`, docs, and unrelated platform workers.

- [ ] **Step 3: Build the usable artifact**

```bash
npm run typecheck
npm run build
npm run worker:build
npm run package:vsix
```

Expected on M4: a `local-llm-engine-0.2.0-darwin-arm64.vsix` package.

Expected on Windows: a `local-llm-engine-0.2.0-win32-x64.vsix` package.

### Task 9: Manual End-to-End Proof

**Files:**
- No source changes unless an objective failure is discovered.

**Interfaces:**
- Consumes: current-platform VSIX and one compatible GGUF.
- Produces: an observed working product.

- [ ] **Step 1: Install and activate**

```bash
code --install-extension local-llm-engine-0.2.0-darwin-arm64.vsix --force
```

Expected: VS Code activates the extension without another application installation.

- [ ] **Step 2: Import or download a model**

Use `Local LLM: Manage Models`.

Expected: the model appears under the `Local LLM` provider.

- [ ] **Step 3: Exercise local inference**

Verify Chat streaming, cancellation, one tool call, and compatible inline completion.

Expected: the worker listens only on loopback and stops with VS Code.
