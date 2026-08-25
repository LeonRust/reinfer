# reinfer

[English](README.md) | [简体中文](README.zh-CN.md)

**reinfer** is a memory-safe, high-throughput LLM inference engine written in **Rust**, targeting **NVIDIA CUDA** and **Ascend CANN** (Huawei NPU) with single-binary delivery.

> ⚠️ Status: early development — workspace scaffold (P0). Serving support lands with P1.

## Highlights

- **Memory-safe engine core** — `#![forbid(unsafe_code)]` in scheduler, radix cache and memory management; all `unsafe` confined to narrow vendor-FFI crates
- **Three-tier kernel architecture** — vendor prebuilt kernels (FlashInfer cubin / CUTLASS / cuBLAS / CANN ACLNN), Rust-native kernels (cudarc / CubeCL), and JIT/DSL bridges (Triton / TileLang / AscendC)
- **High-throughput serving** — continuous batching, chunked prefill, token-budget admission, deterministic decode batching
- **Radix prefix caching** — token-level prefix reuse across requests (RadixAttention lineage)
- **Structured generation** — llguidance-backed grammar / JSON / FSM constraints with zero C FFI
- **Quantization** — GGUF-compatible Q4_0 / Q8_0 / K-quants / IQ family, FP8 / NVFP4 paths
- **Single binary** — `server` / `cli` / `bench` in one executable; GPU backends selected via cargo features
- **Dual hardware** — NVIDIA (sm90/100+ via FlashInfer/CUTLASS) and Ascend (ACLNN + AscendC kernels)

## Quick start

```bash
rustup toolchain install   # stable (see rust-toolchain.toml)
cargo build --release --features cpu   # or: --features cuda / --features can
cargo run                 # prints "reinfer 0.1.0" (scaffold)
```

## Repository layout

```
bin/reinfer     single binary (server | cli | bench)
crates/         workspace crates: core, gguf, arch, memory, cache, scheduler,
                kernels, samplers, grammar, ipc, cpu, jit, cuda, can, server
docs/design/    design documents (analysis, engine design, deep dives)
docs/rfcs/      RFCs (required for constitution-level changes)
```

## Documentation

- **Project constitution** — [`CONSTITUTION.md`](CONSTITUTION.md) (read before contributing)
- **Agent rules** — [`AGENTS.md`](AGENTS.md) / [`CLAUDE.md`](CLAUDE.md)
- **Contributing** — [`CONTRIBUTING.md`](CONTRIBUTING.md)

## License

[Apache-2.0](LICENSE)
