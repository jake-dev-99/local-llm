# Windows SYCL Worker Design (Superseded Assembly)

The original August 24 design established the version-2 worker manifest,
explicit Windows SYCL selection, visible failure behavior, per-file integrity,
and the prohibition on automatic CPU fallback. Those runtime contracts remain.

Its separate native CPU/SYCL build and bundle-assembly design is superseded by
[`2026-09-03-official-windows-sycl-archive-design.md`](./2026-09-03-official-windows-sycl-archive-design.md).
The authoritative Windows artifact is now the pinned official llama.cpp SYCL
release archive. Both `auto` and explicit `cpu` modes use that one physical
distribution with different launch arguments.

Do not reintroduce separate Windows executables, local oneAPI or Visual Studio
activation, PE dependency discovery, runtime DLL reconstruction, or a
packaging-time GPU gate. Native Intel Arc verification remains a release gate
performed after deterministic VSIX assembly.
