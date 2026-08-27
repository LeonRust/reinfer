# reinfer

[English](README.md) | [简体中文](README.zh-CN.md)

**reinfer** is a memory-safe, high-throughput LLM inference engine written in **Rust**, targeting **NVIDIA CUDA** and **Ascend CANN** (Huawei NPU) with single-binary delivery.

> ⚠️ Status: early development. **CUDA**: runtime base (L1) + JIT kernel pipeline (L2) implemented and machine-verified (RTX 5090, 6/6 smoke). **Ascend**: L0 consumer mirror implemented and NPU-verified (5/5 smoke). Serving (P1) lands next.

## Highlights

- **Memory-safe engine core** — `#![forbid(unsafe_code)]` in scheduler, radix cache and memory management; all `unsafe` confined to narrow vendor-FFI crates
- **Three-tier kernel architecture** — vendor prebuilt kernels (FlashInfer cubin / CUTLASS / cuBLAS / CANN ACLNN), JIT kernels (own CUDA C++ / AscendC compiled at runtime, the numeric main path), and Rust-native kernels (cudarc / CubeCL, reserved)
- **JIT kernel pipeline** — kernel source → `nvcc -cubin` → cross-process disk cache (sha256 content key, meta-as-commit-point, flock + double-check, compile-once across processes) → `cuLibraryLoadData` launch; offline prebake with `REINFER_CUDA_ARCH`; **device-adaptive**: arch from measured compute capability and toolchain auto-selected from installed candidates (no hardware-specific defaults)
- **High-throughput serving** — continuous batching, chunked prefill, token-budget admission, deterministic decode batching
- **Radix prefix caching** — token-level prefix reuse across requests (RadixAttention lineage)
- **Structured generation** — llguidance-backed grammar / JSON / FSM constraints with zero C FFI
- **Quantization** — GGUF-compatible Q4_0 / Q8_0 / K-quants / IQ family, FP8 / NVFP4 paths
- **Single binary** — `server` / `cli` / `bench` in one executable; GPU backends selected via cargo features
- **Dual hardware** — NVIDIA and Ascend (ACLNN + AscendC kernels); the JIT cache layer is shared, platform-neutral, zero-unsafe

## Quick start

```bash
rustup toolchain install   # stable (see rust-toolchain.toml)
cargo build --release --features cpu   # or: --features cuda / --features ascend
cargo run                 # prints "reinfer 0.1.0" (scaffold)
```

## Repository layout

```
bin/reinfer     single binary (server | cli | bench)
crates/         workspace crates: core, gguf, arch, memory, cache, scheduler,
                kernels, samplers, grammar, ipc, cpu, jit, cuda, ascend, server
docs/design/    design documents (analysis, engine design, deep dives, machine notes)
docs/rfcs/      RFCs (required for constitution-level changes)
specs/          SDD specs (spec/plan/tasks per feature; see docs/sdd/README.md)
```

## Documentation

- **Specs (SDD)** — [`specs/`](specs/) (MVP, GGUF loader, CUDA runtime base, JIT L2, Ascend mirror; see [Spec-Driven Development](docs/sdd/README.md))
- **Feature list** — [`docs/design/feature-list.md`](docs/design/feature-list.md) (implementation roadmap with traceability)
- **Machine verification** — [`docs/design/notes-jit-l2-2026-08-27.md`](docs/design/notes-jit-l2-2026-08-27.md) (CUDA L2 manual review checklist; Ascend: `specs/011-ascend-l0-mirror/npu-test-checklist.md`)
- **Project constitution** — [`CONSTITUTION.md`](CONSTITUTION.md) (read before contributing)
- **Agent rules** — [`AGENTS.md`](AGENTS.md) / [`CLAUDE.md`](CLAUDE.md)
- **Contributing** — [`CONTRIBUTING.md`](CONTRIBUTING.md)

## License

[Apache-2.0](LICENSE)
