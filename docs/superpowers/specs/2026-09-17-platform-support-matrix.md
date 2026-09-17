# Platform support matrix — local inference runtimes

Date: 2026-09-17. Tracks every consumer platform, what runs on it, and the
specific blocker where something does not. Update this file when a blocker
clears or a new target appears.

Current focus (2026-09-17): macOS arm64 and Windows x64 (NVIDIA, Arc, CPU).
Linux rows are deferred packaging work; Win-ARM64 stays unsupported.

"GGUF" = llama.cpp path (shipped). "Safetensors" = Transformers worker
(env provisioned per the Safetensors scope). Desktop-only throughout:
VS Code Web cannot spawn processes, so neither runtime can run there.

| Platform | GGUF | Safetensors | Blocker / note |
|---|---|---|---|
| macOS arm64 | Yes | Yes, MPS | None; primary target |
| Windows x64 + NVIDIA | Yes | Yes, CUDA | None; ~GB env download |
| Windows x64 + Arc | Yes, SYCL | Works, probe-gated | torch 2.12+xpu, no IPEX; confirmed on Arc 140T |
| Windows x64 CPU-only | Yes | Yes, CPU | Slow; refusal gate applies |
| Windows ARM64 | Yes | No | No PyPI torch |
| Linux x64 + NVIDIA | No | Yes, CUDA | New packaged platform |
| Linux x64 CPU-only | No | Yes, CPU | New packaged platform |
| Linux arm64 | No | CPU only | No PyPI CUDA; Jetson out of scope |
| Intel Mac | Yes | CPU, verify | Confirm torch wheel at manifest build |
| VS Code Web | No | No | No process spawn; desktop-only product |

## Blocker detail

- **XPU on Windows x64 + Arc (required — Intel Arc Pro 140T work laptop).**
  Pinned in `env-manifest.json` and confirmed on hardware (Arc Pro 140T,
  driver 32.0.101.8805, `torch.xpu.is_available() == True` with the XPU
  build): the flavor is torch's own `+xpu` build (2.12.0, in-tree
  `torch.xpu`, 30 wheels, replaces stock torch 2.14) from PyTorch's XPU
  channel — no IPEX anywhere (Intel's index lags at 2.5/2.6-era and its
  2.5.10 file 403s; IPEX itself was archived 2026-03-30, upstreamed).
  The channel lags stock slightly, so re-resolve with `--xpu` as new
  builds appear. Stock (non-`+xpu`) torch answers `False` unconditionally —
  the single most likely cause of any future "XPU unavailable" report.
  The 140T is integrated graphics on shared RAM: device and host are one
  pool, so the host-memory gate check governs (same as Apple Silicon
  unified memory). CPU fallback remains automatic where XPU is absent or
  rejected.
- **Windows ARM64.** No PyPI torch exists, so no env can be provisioned
  regardless of interpreter availability. Revisit only if upstream ships
  wheels; until then the requirement states CPU-only/unsupported.
- **Linux x64 packaging.** Design-clean (standalone + PyPI torch) but the
  worker manifest and CI currently ship darwin-arm64 + win32-x64 only.
  Adding Linux means new VSIX targets, manifest entries, and per-release
  wheel pinning.
- **Linux arm64 CUDA (Jetson).** Needs NVIDIA's channel instead of PyPI.
  Not scheduled; CPU path suffices for the target.
- **Intel Mac torch.** Verify the CPU wheel still publishes when generating
  the release manifest; drop the row to unsupported if it is gone.
