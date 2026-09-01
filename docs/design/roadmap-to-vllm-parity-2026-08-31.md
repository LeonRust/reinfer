# Roadmap to vLLM parity — full feature list (2026-08-31)

> Chinese companion: `roadmap-to-vllm-parity-2026-08-31-zh-CN.md`.
> Evidence base: `docs/design/benchmark-gap-2026-08-29.md` (G1-G10),
> `bench/notes.md` (measurements/blockers), specs/005/006/006-2 (approved).
> Goal anchors (measured, RTX 5090 Laptop / Qwen3-0.6B fp16):
> decode 363 tok/s · long-prefill TTFT 13 ms · TTFT(c1) 9 ms · c4(20 conc) 1.8 s ·
> VRAM 23 GB · T1 = 100 %. Gate = 0.85× llama.cpp CUDA (299.8) decode /
> 0.7× (238.8) prefill.

## Feature list (id · owner spec · dependency · acceptance)

### Stage 0 — correctness base (≈1 week; nothing is comparable before this)

| ID | Feature | Spec anchor | Dep | Acceptance |
|---|---|---|---|---|
| S0-1 | Fix B2: 2048-token FMHA stall (blockage between `FmhaKernels::new` return and batched-prefill loop) | 006 T1 | — | 2048-word prompt prefill via FMHA completes <60 s |
| S0-2 | EOS stop semantics (`<|im_end|>` = 151643 from generation/tokenizer config, not null config key) | 014 D8 | — | T2/eos_short == stop; T1 prompts stop at natural EOS |
| S0-3 | 014 parity four layers vs llama.cpp (tokenizer 100 % / F16 100 % / Q8_0 ≥99.9 % / logits drift ≤1e-2) | 014 D8 | (referee CPU tier ready) | T1 gate 10/10 = 100 % |

### Stage 1 — single-stream throughput (≈2-4 weeks; decode → 299.8, prefill → 238.8)

| ID | Feature | Spec anchor | Dep | Acceptance |
|---|---|---|---|---|
| S1-1 | Decode per-step profile (6 segments cudaEvent: norm/QKV/attn/o/MLP/lm_head) | 006 (record) | S0 | attribution table in notes |
| S1-2 | lm_head optimization (m=1, n=151936 GEMM + cast + tied-embedding layout; split kernel) | 006 increment | S1-1 | decode ×2-4 (measured) |
| S1-3 | Graph replay wiring (BLOCKER-A 3 steps: gemm param-shape refactor → cudarc by cuda-13020 node param read-back → per-launch KernelSpec + PtrUpdate registry) | 006 T4 | S0 | replay == eager bit-identical; per-step kernel launch amortized |
| S1-4 | G5 fusion ① fused MLP-SiLU ② fused norm+add (006-2b ①②, ≥5 % rule) | 006-2 T4 | S1-1/S1-3 (post graph) | ≤4 kernel/layer; ≥5 % vs baseline else record-skip |
| S1-5 | (conditional) decode-attn FMHA tier (G3; only if profile shows attn >40 %) | 006-2 T2 | S1-1 | D7 + 4K text 100 % |
| S1-6 | Dual-stream mode ① (event nodes in graph) | 006 T5 | S1-3 | both modes identical |
| S1-7 | prefill depth: QKV fused kernel + FMHA heuristics tuning (+ conditional vendor tier) | 006 T1 | S0-1 | prefill ≈238.8+ tok/s |
| S1-8 | Benchmark regression gate (baseline.json 5-median + CI red δ≤0.9× + harness diff runs) | 006 T7 | S1-2/3/4/7 | CI red on 10 % regression |

### Stage 2 — serving concurrency (≈4-8 weeks; c4 → ≈2 s)

| ID | Feature | Spec anchor | Dep | Acceptance |
|---|---|---|---|---|
| S2-1 | 005 Scheduler state machine (Waiting→Prefill→Chunked→Decode→Done/Aborted/Preempted; req 2-cursor; token-budget admission; abort tombstone; preempt=recompute) | 005 | S1 base | c4 TTFT ≈2 s; bit-identical rerun |
| S2-2 | Continuous batching + chunked prefill with token budget (scheduler core) | 005 | S2-1 | batch decode GPU-utilized |
| S2-3 | KV pool budget (90 % VMM) + `max-num-seqs` semantics | 005 D2 | S2-1 | VRAM resident 20 GB+, flat trend |
| S2-4 | Prefix-cache interface implementation (D9 lookup/refill → P3-01 RadixCache) | 005 D9 / P3-01 | S2-3 | 2-10× on shared prefixes; bit-identical |

### Stage 3 — API/feature parity & maturity (long tail; two critical)

| ID | Feature | Spec anchor | Dep | Acceptance |
|---|---|---|---|---|
| S3-1 | `n>1` multiple candidates + `stop` honoured (T7 gaps) | 005 (serving) | S2 base | T7 8/8 |
| S3-2 | penalties/logit_bias serve surface (005 D5 chain to GPU sampler) | 005 | S0 | API compat + numeric record |
| S3-3 | speculative decode / grammar (llguidance) | P3 | — | 95 % tests |
| S3-4 | FP8 / KV offload / PD separation (lightllm protocol) | P4 | — | recorded |
| S3-5 | multi-model validation (GLM/other; harness is ready) | 013/bench | — | matrix green per model |

## Dependency graph (parallel waves)

```
Wave 0 (parallel ×3)      S0-1 ‖ S0-2 ‖ S0-3
Wave 1 (parallel ×2)      A=[S1-1→S1-2] ‖ B=[S1-3 graph-side (graph.rs+cudarc+kernels)]
Wave 2                     C=[S1-3 engine wiring] ‖ (S1-6 after)
Wave 3 (parallel ×2)      D=[S1-4] ‖ E=[S1-7]        (engine.rs disjoint regions; hunk-report)
Wave 4                     S1-6 ‖ S1-8 gate
Wave 5 (serial mini)      S2-1 → S2-2 → S3-1/S3-2 (serving surface)
Wave 6 (parallel)         S2-3 ‖ S2-4
Wave 7                     S3-3/S3-4/S3-5 (long tail)
Wave 8                     Full `run_all.py --engine both` acceptance suite + report refresh
```

## Completion gates (re-test checklist)

T1 = 100 % · T2 eos stop · T7 = 8/8 · decode ≥299.8 tok/s · prefill ≥238.8 ·
c4 ≈2 s · steady 20 GB+ flat · multi-model green. Every wave ends with a
`run_all.py --engine both --suite perf_c1,perf_prefill` delta run (5 min).
