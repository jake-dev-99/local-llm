# Local LLM Engine

Local LLM Engine is a self-contained VS Code extension for running GGUF models
without installing Ollama, Python, Docker, or llama.cpp separately. Each VSIX
contains a platform-specific `llama-server` worker that the extension starts and
manages. A bundled manifest pins the worker commit and SHA-256; packaging and
runtime startup both reject worker bytes that do not match it.

Managed model files are recorded with their size, modification time, and
SHA-256. If a model changes on disk, its cached Chat/Agent/FIM compatibility is
invalidated before it can be used again.

The prototype supports:

- Apple Silicon macOS (`darwin-arm64`) with Metal acceleration
- Intel/AMD 64-bit Windows 10 or 11 (`win32-x64`) with CPU inference
- GGUF import, resumable Hugging Face downloads, and direct HTTPS downloads
- installed models in VS Code's native Chat model picker
- ordinary streaming Chat
- a bundled **Local Agent** that preserves VS Code's native agent loop, edit
  previews, approvals, and revert UX with a focused local workspace tool set
- native llama.cpp tool calls with schema-constrained action selection fallback
- local inline fill-in-the-middle completion with Chat priority

## Install

Install the VSIX that matches the computer:

```shell
code --install-extension dist/vsix/darwin-arm64/local-llm-engine-0.3.4-darwin-arm64.vsix --force
```

or:

```shell
code --install-extension dist/vsix/win32-x64/local-llm-engine-0.3.4-win32-x64.vsix --force
```

Reload VS Code after upgrading from an earlier prototype. Installed models live
in VS Code global storage and remain available across extension upgrades.

## First use

1. Run **Local LLM: Manage Models** from the Command Palette.
2. Import a `.gguf`, enter a Hugging Face GGUF repository, or provide a direct
   HTTPS `.gguf` URL. The Hugging Face picker only shows complete single-file
   GGUF downloads; numbered shard parts are excluded.
3. Run **Local LLM: Validate Model Compatibility**. This loads the model,
   reads the actual llama.cpp runtime capabilities, probes inline FIM, performs
   one structured tool call, and verifies the final response after its result.
4. Run **Local LLM: Configure Local Privacy Defaults**. This selects the local model
   for VS Code Chat utility requests and disables Copilot remote semantic
   workspace search, completions, and next-edit suggestions.
5. Open Chat, select **Local Agent** from the agent picker, and select the
   installed model under **Local LLM** in the model picker.

Use ordinary Chat when no workspace tools are needed. Use Local Agent for a
bounded read/search/edit flow. The general-purpose built-in Agent can expose
dozens of tools and is intentionally rejected when it exceeds
`localLlm.maxTools`; the extension never silently deletes tool definitions.

For a private or gated Hugging Face repository, run **Local LLM: Set Hugging
Face Token**. The token is stored in VS Code SecretStorage. Interrupted model
downloads preserve their `.partial` file and resume when the same source is
requested again. Resume uses the server's ETag or Last-Modified validator; if
the remote artifact changed or provides no safe validator, the partial is
discarded instead of spliced into new bytes.

## Why Local Agent exists

VS Code constructs Agent history, instructions, and tool schemas before a
`LanguageModelChatProvider` receives the request. A provider cannot remove
upstream history or control Copilot compaction. Advertising imaginary context
capacity or trimming an arbitrary prefix of tools only moves the failure.

This extension instead contributes a supported native custom agent with these
recognized local workspace tools:

- `read_file`, `get_errors`, and `list_dir`
- `file_search` and `grep_search`
- `insert_edit_into_file` and `replace_string_in_file`

VS Code can expand a selector such as edit into a small number of concrete tool
contracts. The provider logs and exact-counts the effective contracts it
actually receives, then rejects the request if that complete set exceeds
`localLlm.maxTools`. This keeps the stock VS Code agent experience while
bounding schema overhead for common 7B models. Remote semantic search, web,
GitHub repository search, MCP, terminal execution, memory, and subagents are not
available to Local Agent.

Local Agent receives every recognized local tool supplied by VS Code.
The model decides whether tools are relevant and which tool to invoke.
VS Code still owns edit previews, approvals, and reversions.

The configured Local Agent ceiling defaults to eight invoked tools per active
request. Loaded definitions and final generation do not count. At the ceiling,
the provider disables tools and requests normal final generation. No elapsed-time
deadline ends progressing work.

The provider preserves the history supplied by VS Code. It never rewrites
history or rejects repeated calls and answers based on semantic guesses.

## Privacy boundary

Prompts, source code, tool results, and generated text handled by this provider
are sent only to an authenticated worker bound to `127.0.0.1`. The worker API
key is written to a private temporary file rather than exposed in its process
arguments. The worker client rejects non-loopback runtime addresses. The
extension has no telemetry.

**Configure Local Privacy Defaults** sets `chat.utilityModel` and
`chat.utilitySmallModel` to the selected local model, sets
`chat.byokUtilityModelDefault` to `none`, and disables
`github.copilot.chat.semanticSearchTool.mode`, `github.copilot.enable`, and
`github.copilot.nextEditSuggestions.enabled`. This prevents the native Chat
flow from silently using a Copilot model for utility requests covered by those
settings and turns off the two remote editor-suggestion paths. The command
updates global user settings and does not automatically restore their previous
values. **Check Local Privacy Defaults** verifies those settings.

This is a configuration guardrail, not a firewall: VS Code does not let this
provider pin the model selected in the stock Chat UI, and other extensions run
independently. Before sending workspace content, confirm that **Local Agent** and
a model under **Local LLM** are selected. An enforceable process-wide no-egress
boundary requires an external network policy or a separate VS Code profile with
remote AI features disabled.

Model installation is the only network function in this extension. VS Code,
GitHub Copilot, and unrelated extensions may still make their own background
network requests; this extension cannot impose a process-wide network policy.

## Model compatibility

Use an instruct/chat GGUF with a llama.cpp-compatible Jinja chat template for
Chat and Agent use. The extension does not guess capabilities from repository
or file names. On first use it reads `/props` from the running worker and stores
the observed context and chat-template capabilities. Unvalidated models remain
available for ordinary Chat but are not advertised to Agent mode. The validation
command enables Agent only after the model produces one schema-valid structured
tool call and continues with a final text response after receiving its result.

The client accepts llama.cpp-native `tool_calls` first. Every native function
name must match a supplied tool. Its arguments must validate against that
tool's supplied JSON Schema before VS Code receives the call.

If a tool-enabled response contains no usable native call, the client makes one
schema-constrained action request. Required turns select one supplied tool.
Automatic turns select one tool or a text-free `final` action. A `final` action
starts ordinary streamed generation without tools or a response schema.
Structured actions use temperature zero. Final generation uses the configured
chat temperature and output limit. Invalid names, arguments, JSON, or decisions
fail closed.

A failed required native-tool probe is persisted by model hash, worker build,
chat-template hash, platform, and tool-protocol version. Matching restarts skip
that known-useless generation. Any fingerprint change restores one native probe.

Inline completion is enabled only after the validation command gets a nonempty
result from the default model's `/infill` endpoint. It reads a bounded range
around the cursor, debounces typing, cancels stale work, and never switches away
from a different model already active for Chat.

## Settings

- `localLlm.modelDirectory`: managed GGUF directory; empty uses extension global storage
- `localLlm.defaultModelId`: preferred model for inline completion
- `localLlm.contextSize`: physical llama.cpp context window; default `0` lets llama.cpp fit it
- `localLlm.maxOutputTokens`: maximum Chat output, default `2048`
- `localLlm.maxToolCallTokens`: schema-constrained tool-action ceiling, default `512`
- `localLlm.maxTools`: maximum loaded tool definitions per request, default `128`
- `localLlm.maxAgentToolRounds`: maximum invoked tools per active Local Agent request, default `8`; loaded definitions and final generation do not count
- `localLlm.startupTimeoutSeconds`: model-load timeout, default `600`
- `localLlm.cpuThreads`: zero lets llama.cpp choose
- `localLlm.acceleration`: `auto` fits Metal offload to available memory on macOS; `cpu` disables it
- `localLlm.batchSize`: logical prompt batch, default `256`
- `localLlm.microBatchSize`: physical compute batch, default `64`; lower values reduce peak memory
- `localLlm.metalMemoryReserveMiB`: Metal fitting reserve, default `1024`
- `localLlm.temperature`: generation temperature, default `0.2`
- `localLlm.inline.enabled`: enables inline completion
- `localLlm.inline.maxTokens`: inline output ceiling, default `64`
- `localLlm.inline.debounceMilliseconds`: typing debounce, default `250`
- `localLlm.logLevel`: `error`, `info`, or `debug`

The advertised maximum input plus maximum output never exceeds the physical
context. Before inference, llama.cpp templates the complete message and tool
contract and counts it through `/v1/chat/completions/input_tokens`. Oversized
requests fail with a direct explanation instead of triggering hidden tool loss.

## Build from source

Requirements are Node.js, npm, Git, CMake, and a native C/C++ toolchain. Build
each worker on its target platform so the VSIX has no separately installed
runtime dependency.

```shell
npm install
npm test
npm run typecheck
npm run build
npm run build:worker
npm run package -- darwin-arm64
npm run package -- win32-x64
```

Each packaged extension is written to `dist/vsix/<target>/`.

The worker is pinned to llama.cpp commit
`60eeeb6082c1126bb8bc72902c83123cd056811b` (build `b10472`). The checked-in
Windows worker is a portable CPU build. Runtime speed is not a prototype
acceptance gate; actual execution on an Intel Windows machine remains a
platform validation step.

## Prototype boundaries

This version does not provide local embeddings or plug into Copilot's native
semantic `#codebase` index. It uses lexical file/text search and language-service
problems through Local Agent. It also does not support multimodal input,
sharded GGUF installation, remote inference fallback, release signing, or a
custom Chat Participant. A custom participant is only necessary if a future VS
Code Agent harness adds unavoidable fixed context that the bounded native agent
cannot fit.

VS Code references:

- [Language model chat providers](https://code.visualstudio.com/api/extension-guides/ai/language-model-chat-provider)
- [Extension-contributed custom agents](https://code.visualstudio.com/api/references/contribution-points#contributes.chatAgents)
- [Built-in Chat tools](https://code.visualstudio.com/docs/agents/reference/ai-features-cheat-sheet#_chat-tools)
