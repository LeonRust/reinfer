# reinfer

[English](README.md) | [简体中文](README.zh-CN.md)

**reinfer** is a memory-safe, high-throughput LLM inference engine written in **Rust**, targeting **NVIDIA CUDA** and **Ascend CANN** (Huawei NPU) with single-binary delivery.

> ⚠️ Status: early development. **CUDA**: runtime base (L1) + JIT kernel pipeline (L2) implemented and machine-verified (RTX 5090, 6/6 smoke); L3 single-request pipeline in flight. **Model fetch** (pure-Rust, ModelScope-first) implemented and verified against the real repo. **Ascend**: L0 consumer mirror implemented and NPU-verified (5/5 smoke). Serving (P1) lands next.

## Highlights

- **Memory-safe engine core** — `#![forbid(unsafe_code)]` in scheduler, radix cache and memory management; all `unsafe` confined to narrow vendor-FFI crates
- **Three-tier kernel architecture** — vendor prebuilt kernels (FlashInfer cubin / CUTLASS / cuBLAS / CANN ACLNN), JIT kernels (own CUDA C++ / AscendC compiled at runtime, the numeric main path), and Rust-native kernels (cudarc / CubeCL, reserved)
- **JIT kernel pipeline** — kernel source → `nvcc -cubin` → cross-process disk cache (sha256 content key, meta-as-commit-point, flock + double-check, compile-once across processes) → `cuLibraryLoadData` launch; offline prebake with `REINFER_CUDA_ARCH`; **device-adaptive**: arch from measured compute capability and toolchain auto-selected from installed candidates (no hardware-specific defaults)
- **High-throughput serving** — continuous batching, chunked prefill, token-budget admission, deterministic decode batching
- **Radix prefix caching** — token-level prefix reuse across requests (RadixAttention lineage)
- **Structured generation** — llguidance-backed grammar / JSON / FSM constraints with zero C FFI
- **Quantization** — GGUF-compatible Q4_0 / Q8_0 / K-quants / IQ family, FP8 / NVFP4 paths
- **Single binary** — `server` / `cli` / `bench` in one executable; GPU backends selected via cargo features
- **Model fetch** — pure-Rust ModelScope client (no Python); `reinfer model list/get` with sha256-verified atomic downloads + runtime auto-download (`ModelResolver`), ModelScope-first with optional HuggingFace fallback
- **Dual hardware** — NVIDIA and Ascend (ACLNN + AscendC kernels); the JIT cache layer is shared, platform-neutral, zero-unsafe

## Model fetch

`reinfer model` downloads GGUF models with pure Rust — no Python, no pip, no external CLI
(`crates/models`, spec [`specs/013-model-fetch`](specs/013-model-fetch/spec.md)):

```bash
# list GGUF files of a repo (name / size / sha256)
reinfer model list Qwen/Qwen2.5-0.5B-Instruct-GGUF

# download a quantized GGUF (resolves quant tag → file name, verifies size + sha256)
reinfer model get Qwen/Qwen2.5-0.5B-Instruct-GGUF --quant q8_0

# exact file / every GGUF / custom dir
reinfer model get Qwen/Qwen2.5-0.5B-Instruct-GGUF --file qwen2.5-0.5b-instruct-q8_0.gguf
reinfer model get Qwen/Qwen2.5-0.5B-Instruct-GGUF --all
reinfer model get Qwen/Qwen2.5-0.5B-Instruct-GGUF --quant q8_0 --to ~/models/reinfer
```

Source priority and download policy come from env (CLI args win):

| Var | Values | Default | Meaning |
|---|---|---|---|
| `REINFER_MODEL_SOURCE` | `modelscope`/`huggingface`/`auto` | `auto` | `auto` = ModelScope first, falls back to HuggingFace on miss |
| `REINFER_MODEL_DIR` | path | `~/models/reinfer` | download/search root (`~` is expanded) |
| `REINFER_MODEL_VERIFY` | `sha256`/`size`/`none` | `sha256` | verify depth; HF source degrades sha256 → ETag+size (no sha field upstream) |
| `REINFER_MODEL_AUTODOWNLOAD` | `on`/`off` | `on` | `off` = never dials out (missing model → error) |
| `REINFER_MODEL_REPO`/`QUANT`/`FILE` | repo, quant tag, exact name | — | convenience injection (CLI takes precedence) |
| `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY` | standard | — | egress, e.g. `http://192.168.0.1:7890`; `NO_PROXY=...,modelscope.cn,huggingface.co` for direct |

Downloads stream to a temp file, verify against the ModelScope files API sha256 (ETag+size
for HuggingFace), rename atomically and record a `manifest.json` entry; a failed verification
retries once and fails loudly — no half-written leftovers. `AUTODOWNLOAD=off` keeps the
runtime fully offline. Model identifiers are never hardcoded in the engine: repo/file names
always come from CLI/env.

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
