# reinfer Implementation Feature List

> Living document · Last updated: 2026-08-27
> Rules: constitution §6.4 (SDD) — every feature must have a spec anchor; a feature without one needs a new `specs/NNN-*` first.
> Upstream facts as of 2026-08-25: CANN 8.5.0 symbols verified (cann-rs `docs/cann-850-catalog.md`); cann-rs L0 **implemented** (cann-sys `5c17e9e`, cann `825a792`); L1 (aclnn/GE) SDD docs landed (`88a3cf4`), code pending.

## Track 1 — Ascend L0 (spec 002) — **unblocked, start here**

| ID | Feature | Crate(s) | Spec anchor | Upstream dep | Gate | Status |
|---|---|---|---|---|---|---|
| ASC-01 | `cann` wiring: workspace dep + `ascend` feature forwarding (`bin/reinfer`) | `crates/ascend`, `bin/reinfer` | 002/plan §Decision | cann-rs L0 ✅ | `cargo check --workspace` (no SDK) green | ✅ ready |
| ASC-02 | `error.rs`: `cann::Error → LaunchError` (whitelist mapping: OOM/Driver/Fatal) | `crates/ascend` | 002/plan §Error mapping | cann::Error (is_oom/is_recoverable) ✅ | unit tests | ✅ ready |
| ASC-03 | `diag()` + `reinfer diag` subcommand (CANN version str/num, device count, readable errors) | `crates/ascend`, `bin/reinfer` | 002/spec §AC | cann::Version/Context ✅ | NPU-runner smoke | ✅ ready |
| ASC-04 | Contract conformance tests (signatures vs `cann` HEAD) | `crates/ascend` | 002/tasks T4 | — | CI | ↔ later (needs SDK) |
| ASC-05 | CI three-tier: lint (no SDK) / build (`--features ascend`, SDK) / NPU smoke | infra | specs/008 (can-gpu runner 预留) | — | CI | ✅ specs/008 定义 |

## Track 2 — CPU / P0 (specs 000 + 001) — **unblocked**

| ID | Feature | Crate(s) | Spec anchor | Gate | Status |
|---|---|---|---|---|---|
| P0-01 | Core types: `DType`, `TensorId`, `DeviceId`, `OpConfig`, `ReqId`, `Error` | `crates/core` | 001/tasks T1 | `cargo test -p reinfer-core`, forbid unsafe | ✅ ready |
| P0-02 | GGUF reader (header/meta/tensor table + mmap views) | `crates/gguf` | 001/tasks T2–T3 | golden-file + proptest | ✅ ready |
| P0-03 | Quant codecs naive (Q8_0 / Q4_0 / F16 / FP32) + proptest | `crates/gguf` | 001/tasks T4 | ≤1 ULP golden blocks | ✅ ready |
| P0-04 | Arch config loader (Llama metadata → typed config; Qwen2 parse) | `crates/arch` | 001/plan §Modules | parse tests | ✅ ready |
| P0-05 | GGUF tokenizer (SPM/BPE encode + increment-decode from GGUF tokenizer models) | `crates/tokenizer` | ✅ `specs/004-tokenizer` | golden vs llama.cpp tokens | ✅ spec ready |
| P0-06 | CPU inference loop: RMSNorm/RoPE/softmax/MHA naive + contiguous KV (GQA), streamed decode | `crates/cpu`, `crates/arch` | 🔒 spec **007-core-inference** (CPU 全链路，为 005 `--backend cpu` 与无 GPU CI 提供载体) | 000/spec parity criteria | 🔒 spec first |
| P0-07 | `info` + `cli` subcommands + differential harness vs llama.cpp | `bin/reinfer`, `bench/` | 001/tasks T5–T6; 000 | PNG: token 100% on 20 prompts; ≥60% llama.cpp decode | ↔ after P0-05/06 |

## Track 3 — P1+ (design report §7) — specs needed before code

| ID | Feature | Notes |
|---|---|---|
| P1-01 | CUDA backend (cudarc + JitCache + kernels + cuBLAS) | ✅ `specs/003-cuda-l0` (single-request GPU loop)；L1 wrappers 展开锚 `specs/009-cuda-runtime-base` — spec ready |
| P1-02 | Paged KV pool (refcount + free list) in `crates/memory` | included in 003 T9; policy-side is ours (boundary §4) |
| P1-03 | Scheduler: continuous batching, chunked prefill, token-budget admission, `req_id` determinism | ✅ `specs/005-scheduler-serving` |
| P1-04 | CUDA graph bucket capture + stream overlap | ✅ `specs/006-cuda-perf` (FA3 vendor cubin optional) |
| P1-05 | OpenAI-compatible HTTP server (axum) + sampler chain (greedy/top-p/top-k) | ✅ specs/005 (P1-03 same spec) |
| P2-01 | Ascend full: Vendor-tier aclnn ops (L1, via cann safe wrappers), GE graph (aclgrph*) session pattern, AscendC pipeline (`crates/jit`), HCCL | depends cann-rs 0002 code; graph note: 8.5 = GE engine `aclgrph*`, not legacy `aclrtGraph*` |
| P3-01 | RadixCache · speculative decode · llguidance grammar · TP/PP/CP | semantics from design report §2 |
| P4-01 | MoE/MLA/FP8 · autotune TuneDb · KV offload · PD separation (lightllm protocol) · plugins | |

## Recommended execution order (backlog)

1. **ASC-01/02/03** — quick win: version/device diagnostics closed loop (cann-rs L0 already verified + implemented)
2. **P0-01/02/03/04** — pure-Rust data path (001 tasks T1–T4); no GPU, no upstream deps
3. **Write specs 007-core-inference (CPU 全链路) + 008-ci-infra (CI 工件)** — 005 的 `--backend cpu` 一致性套件依赖前者，全部 GPU 硬门禁依赖后者；then P0-07 → P0 gate: llama.cpp token parity + 60% decode (CPU 档)
4. 下一步并行：specs/005（serving，已修订）+ 006（vendor/graph，已修订）实施；Ascend 轨道按 002 契约随 cann-rs L1/L2 节奏推进
