# Spec: decode stage pipeline — cross-stage/layer overlap (S1-12)

> Status: draft (2026-09-02) · Parent: specs/017 (block-width wave, closed) ·
> 017 series endpoint: micro-kernel efficiency is saturated (p1_gu 650 GB/s
> = sector-limit, flash at machine optimum, barrier/backoff zero-gain); the
> remaining single-step time (4.00 ms vs the 3.335 ms gate; 16.7 % gap) is
> sequential latency across stages that overlap nothing — every stage runs
> on ONE grid with full barriers. This wave overlaps them without touching
> any arithmetic.

## Problem Statement

Per-layer stage budget (017-c/d probes, W=2, µs/层): flash ~15-19 (latency-
bound, "machine optimum"), p2_qkv ~15 (head-norm sync-tree latency ~40× its
IO floor), p1_o ~10, p2_gu_d ~9, p1_gu ~19 (DRAM-saturated), lm_head 0.57 ms.
The layer chain is strictly sequential: qkv → flash → o → (add+rms) → gu →
down → next layer. **Nothing of layer i+1 starts before layer i ends** —
yet the data dependencies are narrow: the only cross-stage input edges are
(x_prev, qkv) for the next layer. p2_qkv's sync-tree latency and p1_o's
low-occupancy row could run concurrently with other stages' memory streams.

**This spec: explicit pipelining of the decode step — two (or more)
concurrent execution streams over the per-layer stage DAG, with the
arithmetic sequence, tile order and reduction topology untouched.**

## Success Metrics

- **性能（硬门 S1-8）**：gpu busy 4.00 → **≤3.335 ms**（≥299.8 tok/s）；
  中间目标 ≤3.60 ms（≥278 tok/s）。
- **位级（硬门）**：pipline 变体与 017 终态（W=2, WC=1）逐字节一致
  （现有 5 项 bit 门全保持——**不改任何运行顺序中的数值，只并发化**）。
- **回退（软门）**：`REINFER_FUSED_PIPE` off（缺省 off——先实验后默认）= 
  S1-11 终态字节一致；每种 overlap 变体带开关位。
- **段表（硬门）**：REINFER_DECODE_PROFILE——与串行基线比，p2_qkv/flash
  行的"汇合等待"归零幅度可见；层均值 ≥5% 下降（目标 ≥10%）。

## Approach (staged, each a small experiment on top of the closed 017 state)

- **P1** — *poll-away overlap*: split the layer grid into two cooperating
  groups (A: qkv/flash; B: o/gu/down) with **two barrier trees** (one per
  group) instead of one; the groups synchronize only on the data edge
  (o-out, down-out). Cost: an extra buffer page per layer for the o/gu row
  (2 MB × 2). Bit-level: same scans, same trees, no arithmetic change.
- **P2** — *cycle amortization by unrolled double staging*: perform the
  017 layer body for layer i and the (small) p2_qkv head-norm part of
  layer i+1 in one launch — i.e. software-interleave the serial stages
  across the grid.
- **P3** — if barriers are the residual: producer-consumer arrival (no
  sense-reversal broadcast; consumer polls a produce-count) for the
  off-critical bar-only edges.

Each step: measure, gate on bit-exact, revert if ≤0.

## Non-Goals

- Multi-stream CUDA concurrency (different kernels per stream; the single-
  kernel grid design stays — separate streams break the 31-launch
  amortization).
- Changing arithmetic/split order (D7) — bit-level hard gate.
- flash redesign (repeatedly proven at machine optimum).
- Power-limit changes (user-approved, separate; this spec is software).

## Constraints

- Same scan/tile/reduction code paths — only participation windows and
  buffer indices change.
- Occupancy limits (W=2 = 64 regs, occ 2 blocks/SM; extra buffers must fit
  the max_tiles budget and smem/reg constraints).
- 模型无关；不触 bin/reinfer/ 接口（env 级开关）。

## References

- specs/017-*（closed — S1-11 records in bench/notes.md §S1-11）
- bench/notes.md 017-c/d (probability tables, zero-gain evidence)
- roadmap S1-12 (Stage 1 follow-up)
