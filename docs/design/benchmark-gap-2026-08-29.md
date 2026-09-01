# Benchmark Gap vs vLLM — measured baseline (2026-08-29)

> Companion Chinese version: `benchmark-gap-2026-08-29-zh-CN.md`.
> Evidence source: `bench-vs-vllm/` harness (`report.md`, `results/{engine}/*`, `probes/`),
> commit `cd4620995b6bde104a614aa806cd6ebb7b477b71`; run via one-shot matrix
> (`python run_all.py --engine both`). This document records *what the current
> binary delivers* and *what is missing*; it is the baseline reference for
> specs/005 and specs/006.

## 1. Setup (protocol, locked)

| Item | Value |
|---|---|
| GPU | RTX 5090 (sm_120a), driver driver-side CUDA 13.2 JIT (nvcc) |
| Model | Qwen/Qwen3-0.6B fp16, same directory `~/.reinfer/models/Qwen/Qwen3-0.6B` for both engines |
| Reference (vLLM) | vLLM 0.28.0 official wheel; `serve --dtype float16 --max-model-len 4096 --max-num-seqs 1` (F1 fair mode) / `8` (F2) |
| Subject (reinfer) | `reinfer serve` release binary, serial engine queue (V1), fp16, `max-model-len 4096` |
| Semantics | `chat_template_kwargs.enable_thinking=false` on **both** sides (Qwen3 thinking off) |
| Fixed params | seed=42, temperature=0 for gates; temperature=1.0 + top_p=1.0 for record suites; logprobs top-5 for R1 |
| Metrics | TTFT / TPOT / ITL / E2EL (client monotonic, per-SSE-frame); steady-state via pynvml |

## 2. Headline gap (measured)

| Metric | vLLM (gold) | reinfer (current) | Ratio |
|---|---|---|---|
| TTFT, c1 (conc=1) p50 | 9 ms | 4 926 ms | **≈550×** |
| Decode throughput, single stream (tpot p50) | 363 tok/s (2.7 ms/tok) | 11.2 tok/s (89.6 ms/tok) | **≈33×** |
| Long-prefill TTFT (in=2048 words, out=16) p50 | 13 ms | 944 235 ms (~15.7 min) | **≈70 000×** |
| c4 (conc=20) TTFT p50 / p95 | 1 822 / 3 594 ms | 106 199 / 207 460 ms | **≈60× and diverging** |
| GPU memory steady-state | 23 336.9 MB | 3 102.9 MB | KV pool not used |
| T1 greedy token identity (temperature=0) | baseline | **0 %** (first token differs on all 10 prompts; EOS never fires) | correctness not closed |
| T7 API compatibility | 8/8 | 7/8 (`n>1` single-candidate; `stop` parsed-but-ignored) | near parity |

Additional gates observed: `t2/eos_short` — vLLM `finish=stop` (natural EOS after 10 tok),
reinfer `finish=length` (runs full 64); R1 distribution Jaccard 0.296 / TV 0.245
(only 3 aligned steps — sequence diverges at token 0); R5 steady-state both `trend=flat`.

## 3. Gap → missing piece mapping

Each gap carries a G-tag (used as a stable reference by specs/005/006/006-2/014).

| G-tag | Measured gap | Missing piece | Spec owner |
|---|---|---|---|
| G1 | prefill ≈70 000× | FMHA prefill (current path = two-GEMM + fp32-internal buffer, 014 D7 naive structure); no fused pretokenize/attention | specs/006 (D2) |
| G2 | decode ≈33× (generic part) | decode fused Q8_0 dequant-dot kernel; CUDA graph decode buckets; dual-stream overlap | specs/006 (D3–D5) |
| G3 | decode ≈33× (attn structure part) | decode-attn performance tier (no flash-style paged decode kernel beyond 003 naive) | specs/006-2 (G1; deferred to 006-2b) |
| G4 | decode ≈33× (sampling part) | sampling/penalty chain on CPU (llm-samplers crates.io 0.0.7 + rand StdRng); no GPU sampler / logits-host round-trip | specs/006-2 (G2, this round) |
| G5 | decode ≈33× (launch part) | decode kernel fusion group (fused MLP-SiLU, fused norm+add, …; graph already zeroes launch cost but not kernel-internal data movement) | specs/006-2 (G4; deferred to 006-2b) |
| G6 | concurrency ≈60×, diverging | serve currently `engine.lock()` serial queue; scheduler crate is a 2-line stub | specs/005 (chat/serving) |
| G7 | GPU mem 3.1 GB vs 23.3 GB | token-budget admission + KV pool budget (90% of VMM) + `max-num-seqs` semantics | specs/005 (D2) + P1-02 |
| G8 | (no direct metric yet) | prefix cache (RadixCache/vLLM semantics) — see 005 D9 interface commitment | specs/005 (D9) + P3-01 |
| G9 | T1 0 %, EOS never fires | EOS stop semantics (see §5, open item O1); (014 D8 requires "EOS hit ⇒ stop") | specs/014 (D8) |
| G10 | t3 abort still computes server-side (~100 s tail) | no client-disconnect cancellation; abort/tombstone isolation | specs/005 (isolation) |

Bandwidth sanity — derivation (no hardcoded model constants; everything is
derived at runtime from the model `config.json` / harness measurement):

- Model per-step weight bytes = derived from config fields: `vocab_size` ×
  `hidden_size` × 2B (embeddings; tied iff `tie_word_embeddings` — the tied
  matrix doubles as the LM head, counted once) + Σ layers [(qkv: 3×
  `hidden×hidden`-ish + o projection) + (gate/up/down: 3× `hidden×intermediate`)] ×
  2B. Qwen3-0.6B instance (values read from its config.json): ≈1.50 GB/step.
- Machine bandwidth = device-reported (nvidia-smi memory speed × bus width);
  **this machine is an RTX 5090 Laptop** (256-bit GDDR7 ≈896 GB/s — not the
  desktop 1.79 TB/s) → ceiling ≈596 tok/s for the instance above.
- **Measured reference** (llama.cpp f280b2698 + nvcc 13.2 + sm120, llama-bench
  `-b 1 -n 512 -fa 1 -ngl 99`, 5-run median, model sha `d04bceb6…` — see
  `bench/baseline-llamacpp.json`): **352.70 tok/s** (59 % bandwidth efficiency,
  bs=1 small-GEMM launch overhead) → 0.85× gate target ≈ **299.8 tok/s**;
  KV read at 4 K ctx adds ≈19 %.

## 4. Recommended order (three waves)

1. **Wave 0 — correctness closure**: EOS `(<|im_end|>)` stop semantics; 014 parity
   four layers (tokenizer 100 % / F16 100 % / Q8_0 ≥99.9 % / logits drift ≤1e-2);
   temp=0 argmax short-circuit. No speed gain — but numbers are only comparable
   once this closes.
2. **Wave 1 — single-request throughput (specs/006 + 006-2 G2 contract)**:
   FMHA prefill (biggest lever) → CUDA graph decode buckets → fused Q8_0
   dequant-dot → dual-stream overlap → vendor fallback chain with TuneDb.
   **006-2 (this round)**: GPU sampler contract (G4: determinism anchor +
   LogitsView + function-level bit-identical). **G3/G5 deferred**: profile-gated
   re-open after 006 lands. Gate: decode ≥0.85× llama.cpp CUDA
   (reference measured first — see specs/006-2 T0/T6).
3. **Wave 2 — serving concurrency (specs/005)**: scheduler state machine
   (Waiting→Prefill→Chunked→Decode→Done/Aborted/Preempted), continuous batching,
   chunked prefill with token budget, token-budget admission, abort isolation
   (tombstone, exactly-once release), preemption = recompute (vLLM semantics),
   `req_id` determinism, KV pool budget. Gate: P1 (decode ≥85 % SGLang).
4. **Wave 3 — maturity (P3/P4, optional for "same order of magnitude")**:
   RadixCache/prefix cache (real benefit on shared-prefix workloads),
   speculative decode, grammar, TP/PP/CP, MoE/MLA/FP8.

Expected acceptance ladder — **expectation track only, NOT a second gate**;
the sole performance gate for Wave 1 is decode ≥0.85× llama.cpp CUDA (measured
reference first, see specs/006-2 T0). Measure via
`bench-vs-vllm: run_all.py --engine both --suite perf_c1,perf_prefill`:

| Milestone | single-stream decode | long-prefill TTFT | c4 TTFT p50 | vs vLLM |
|---|---|---|---|---|
| now | 11 tok/s | 944 s | 106 s | 33× / 70 000× / 60× |
| after Wave 0 | 11 (trustworthy) | 944 (correct) | — | comparable |
| after Wave 1 | ~150–250* | ~3–8 s | ~50 s | ~1.5–2×* / ~100× / ~25× |
| after Wave 2 | ~150–250 | ~2–4 s | ~2–4 s | ~1.5× / ~20–50× / ~1–2× |

\* Note: decode ~150-250 tok/s and "1.5-2× vs vLLM" (=545-726 tok/s) were **not
consistent** — resolved 2026-08-29 after the reference measurement: llama.cpp CUDA
= 352.70, 0.85× gate = 299.8 → that is ≈0.83× vs the vLLM anchor (363), so the
"after Wave 1" vs-vLLM column is corrected to **~0.83×** and the ladder's decode
row stays as the 006+006-2b projection.
**Intermediate anchor (2026-08-29, GPU sampler landed before 006)**: measured
single-stream decode **12.53 tok/s** (tpot 79.8 ms) — G4 contract delivered
(+12 % vs 11.16); confirms the decode-step body (79.8 ms) is dense-loop
GEMM/attn/launch, i.e. G3/G5 remain mandatory but gated on 006 profile
(006-2b trigger "measurement needed" now satisfied by this row).

## 5. Open items (unverified hypotheses — do not treat as findings)

- **O1 — EOS id source**: `pipeline.rs:144` compares `Some(next) == eos_id`; the
  parameter comes from `serve.rs` ← `AppState.eos` ← model `config.json`
  `eos_token_id`. Qwen3's real EOS (`<|im_end|>` = 151643) lives in
  `generation_config.json`/`tokenizer_config.json`; `config.json` key may be
  `null` → `eos_id=None` → EOS never fires. **Unverified**: read the value before
  opening a spec task.
- **O2 — t3 tail**: client disconnect does not cancel the server-side request
  (behavior confirmed: ~100 s+ long-prefill completed after client timeout);
  root cause in serve/pipeline not yet traced.
- **O3 — SIGTERM**: `serve` does not exit within 30 s of SIGTERM (needs SIGKILL
  fallback in `stop_servers.sh`); shutdown path incomplete.

## 6. Related artifacts

- `../bench-vs-vllm/report.md` — full bilingual report (gates, records, perf tables)
- `../bench-vs-vllm/results/` — raw jsonl/csv per engine per suite
- `../bench-vs-vllm/README.md` — harness usage + findings checklist
- specs/005-scheduler-serving, specs/006-cuda-perf, specs/014-cuda-l3-single-request,
  specs/007-core-inference — implementation anchors referenced above
