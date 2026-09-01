# reinfer Implementation Feature List

> Living document · Last updated: 2026-08-27
> Rules: constitution §6.4 (SDD) — every feature must have a spec anchor; a feature without one needs a new `specs/NNN-*` first.
> Upstream facts as of 2026-08-25: CANN 8.5.0 symbols verified (cann-rs `docs/cann-850-catalog.md`); cann-rs L0 **implemented** (cann-sys `5c17e9e`, cann `825a792`); L1 (aclnn/GE) SDD docs landed (`88a3cf4`), code pending.

## Track 1 — Ascend L0 (spec 002) — **unblocked, start here**

| ID | Feature | Crate(s) | Spec anchor | Upstream dep | Gate | Status |
|---|---|---|---|---|---|---|
| ASC-01 | `cann` wiring: workspace dep + `ascend` feature forwarding (`bin/reinfer`) | `crates/ascend`, `bin/reinfer` | 002/plan §Decision | cann-rs L0 ✅ | `cargo check --workspace --no-default-features` (no SDK) green；ascend/cuda crate default=能力开（2026-08-27 定） | ✅ ready |
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
| P0-08 | Model fetch: pure-Rust ModelScope/HF client (`reinfer model list/get` + runtime `ModelResolver` env policy: SOURCE/DIR/VERIFY/AUTODOWNLOAD, ModelScope-first, HF fallback) | `crates/models`, `bin/reinfer` | ✅ specs/013-model-fetch | stub network tests + end-to-end (q8_0 675710816 B, sha256 matches repo value; manifest left) | ✅ done (f35205b + T3 CLI) |
| P0-05 | GGUF tokenizer (SPM/BPE encode + increment-decode from GGUF tokenizer models) | `crates/tokenizer` | ✅ `specs/004-tokenizer` | golden vs llama.cpp tokens | ✅ spec ready |
| P0-06 | CPU inference loop: RMSNorm/RoPE/softmax/MHA naive + contiguous KV (GQA), streamed decode | `crates/cpu`, `crates/arch` | ✅ specs/007-core-inference（2026-08-28 r2 四代理评审）——「无加速卡也能推理」兜底端 + 无卡 CI 载体 | 000/spec parity criteria + 007 r2 判据（F16 token 100% / Q8_0 ≥99.9% 硬；吞吐记录不设档） | ✅ spec r2 提案（待实施） |
| P0-07 | `info` + `run` subcommands + differential harness vs llama.cpp | `bin/reinfer`, `bench/` | 001/tasks T5–T6; 000 | **r2 修订（007 评审：naive 单线程 ≥60% 结构不可达——带宽分析 8-25× 差距）**：F16 token 100% + Q8_0 ≥99.9%（20 prompts）硬；吞吐记录（notes 四元组，无 % 档） | ↔ after P0-05/06（007 r2 已定） |

## Track 3 — P1+ (design report §7) — specs needed before code

| ID | Feature | Notes |
|---|---|---|
| P1-01 | CUDA backend (cudarc + JitCache + kernels + cuBLAS) | **L1 ✅**（009）；**L2 ✅**（012：真机 6/6 smoke、命中 2.7ms；见 notes-jit-l2-2026-08-27.md）；**剩余 = specs/014（r2，2026-08-28 四代理评审修订）T0-T11**：dequant/GEMM/attention/run 闭环 + parity 四层 + 3×（1.5B Q8_0）；llama.cpp referee = 014 T0（CPU 档）；实施顺序 **014 → 015 → 007**（用户 2026-08-28 定） |
| P1-02 | Paged KV pool (refcount + free list) in `crates/memory` | included in 003 T9; policy-side is ours (boundary §4) |
| P1-03 | Scheduler: continuous batching, chunked prefill, token-budget admission, `req_id` determinism | ✅ `specs/005-scheduler-serving` |
| P1-04 | CUDA graph bucket capture + stream overlap | ✅ `specs/006-cuda-perf` (FA3 vendor cubin optional) |
| P1-06 | Decode-side kernel performance（本轮：GPU sampler 契约——基准 G4；挂起至 006-2b：decode-attn 性能档 G3 / 融合核组 G5；warp specialization 不计划） | ✅ `specs/006-2-decoding-kernel-performance`（approved r2 2026-08-29 四代理评审） |
| P1-05 | OpenAI-compatible HTTP server (axum) + sampler chain (greedy/top-p/top-k) | ✅ specs/005 (P1-03 same spec) |
| P2-01 | Ascend full: Vendor-tier aclnn ops (L1, via cann safe wrappers), GE graph (aclgrph*) session pattern, AscendC pipeline (`crates/jit`), HCCL | depends cann-rs 0002 code; graph note: 8.5 = GE engine `aclgrph*`, not legacy `aclrtGraph*`；**进度：L1 代码在仓（0002 proposal，未真机验证）——015（r2 提案）是「昇腾第一次跑出 token」的最小闭环（依赖 014 M4/M5 + 0002 aclnn 面），实施列 014 之后 |
| P3-01 | RadixCache · speculative decode · llguidance grammar · TP/PP/CP | semantics from design report §2 |
| P4-01 | MoE/MLA/FP8 · autotune TuneDb · KV offload · PD separation (lightllm protocol) · plugins | |

## Recommended execution order (backlog)

1. **ASC-01/02/03** — quick win: version/device diagnostics closed loop (cann-rs L0 already verified + implemented)
2. **P0-01/02/03/04** — pure-Rust data path (001 tasks T1–T4); no GPU, no upstream deps
3. ~~Write specs 007-core-inference …~~ ✅ 007 r2 已写（2026-08-28 四代理评审）；P0 gate 修订 = F16 token 100% + Q8_0 ≥99.9% + 吞吐记录（「60% decode」经带宽分析不可达，已撤销）
4. **运行模型实施顺序（用户 2026-08-28 定）**：① **014**（CUDA L3：referee T0 → 数据管道 → 真机内核 → run 闭环 → parity/3×）② **015**（Ascend L3：依赖 014 M4/M5 + cann-rs 0002 aclnn 面 + T0（cann-rs 真机 smoke）→ 权重组装 → α 实测定档 → 层循环 → run 闭环 → 记录档报告）③ **007**（CPU：兜底端，T1 可与 014 并行，T2-T4 依赖 014 T0/T10）；specs/005（serving）+ 006（vendor/graph）实施不受阻（后续并行）
5. **性能差距基线（2026-08-29 实测）**：`docs/design/benchmark-gap-2026-08-29.md`（中文并存 `-zh-CN`）—— vs vLLM 0.28（5090/fp16/Qwen3-0.6B）：TTFT ≈550× / decode ≈33× / prefill ≈7 万× / c4 并发 ≈60× / T1 0%（正确性未闭合）。**推进顺序追记**：014 的 D8「EOS 命中即停 + parity 四层」作为波 0 先行（先正确后性能）；005/006 实施时以该基线文档为性能参照（006 基准协议与 P1 门禁不变）。
6. **达到 vLLM 同量级路线图（2026-08-31）**：`docs/design/roadmap-to-vllm-parity-2026-08-31.md`（中文并存 `-zh-CN`）—— Stage 0（正确性基座：B2/EOS/parity）→ Stage 1（单流：profile/lm_head/graph-replay/fusion/prefill 深度/门禁）→ Stage 2（服务化：005 scheduler/KV 池/prefix cache）→ Stage 3（API 对齐；关键项 n>1、stop）。**实施波次与并行分组见 roadmap 依赖图**；每波结束跑 perf_c1+perf_prefill 增量。**新功能 spec 规则**：S1-2/S1-3 属 006 增量（T 卡扩展），S1-4 属 006-2b（挂起条款触发），S2-* 属 005；涉及新 spec 先走 SDD 再实施。
