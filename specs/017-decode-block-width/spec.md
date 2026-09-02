# Spec: decode block-width wave — per-layer single-block stages widened (S1-11)

> Status: draft (2026-09-01) · Parent: specs/006-2 (decoding kernel performance)
> · Requires: S1-10 layer-fused (31 launches/step, grid barrier) · 017 continues
> the S1-10c record — the leftover "single-block serial" hotspots are the last
> software lever after the power-limit raise (user-approved, 110 W).

## Problem Statement

`REINFER_DECODE_PROFILE` (2026-09-01, window 21-40): gpu busy **4.34 ms/step**
(31 launches, host 0.06 ms). Per-layer stage distribution (µs/层, normalized
to the 135 µs/层 layer budget) shows the biggest rows are **single-block,
serial-compute stages**: p2_o ≈ 0.83 ms/step, p1_gu ≈ 0.74, gather ≈ 0.69,
p2_qkv ≈ 0.40, flash ≈ 0.40 (latency-bound, already proven at machine
optimum), p2_gu_d ≈ 0.26. The S1-10 barrier/cost model was falsified twice
(S1-10b: partial-participant barrier and nanosleep both zero-gain) — the
remaining time really is the arithmetic/latency of the per-stage work, which
runs on one grid of 82 blocks with each stage's work swept by a single block
group. **This wave widens the block coverage per stage without changing any
scan order or reduction structure — bit-level strict.**

Target: gate 0.85× = **299.8 tok/s (tpot ≤ 3.335 ms)** — the approved
"持平" criterion (perf-gate.sh). With the 110 W power raise (~+10-20% on
this machine) the engine is expected to land ~3.4-3.9 ms; the block-width
wave targets the remaining ~0.5-1.0 ms of single-block time.

## Success Metrics

- **性能（硬门 S1-8）**：`bench/perf-gate.sh` verdict PASS（median tpot ≤
  3.335 ms→≥299.8 tok/s；CI red 317.4 为记录线）。
- **位级（硬门）**：layer-fused block-width 变体与 S1-10 基线逐段、逐字节
  位级一致（现有 `layer_fused_li1_bit_exact_vs_split` + D7 聚合序检查 +
  `layer_fused_determinism_double_run` 全部保持）；**列序/归约树/聚合
  顺序零改动**——只改"每块覆盖的列集合"。
- **段表（硬门）**：REINFER_DECODE_PROFILE：block-width on vs off 的
  p2_o/gather/p1_gu/p2_qkv add_rms 段均值下降 ≥25%（合计 ≥0.4ms/step），
  其余段（flash/p2_gu_d）±2% 内。
- **回退（软门）**：REINFER_FUSED=off / REINFER_LAYER_FUSED=off 仍位级
  与 split 路径一致（现有回退面不破）。

## User Stories

1. 作为性能研究者：block-width 变体经 env 开关（REINFER_FUSED_BW=on 默认），
   段表可观测（现有 REINFER_DECODE_PROFILE）。
2. 作为确定性依赖者：开/关位级相同（D7 聚合序不动）。

## Acceptance Criteria

T1 审计——用现有 REINFER_DECODE_PROFILE 给出每段的"单 block 覆盖率"（当前
  每段实际使用 block 数）与带宽/处理器占用模型，定候选段（p2_o、gather、
  p1_gu、p2_qkv、add_rms(o/down)）每条目标宽度。
T2 块宽化内核——设置每层 grid 更宽（如 82→2-4×），把每段的列空间切块分给
  更多 block（**同一 scan 顺序/列序/归约前缀序，仅切块**）：
  - [ ] p2_o（2 MB 读 + 2 MB 写/层，当前单块）→ 目标 ≥2×
  - [ ] gather/rms0（嵌入 8 MB/层首层 + 后续层 gather）→ ≥2×
  - [ ] p1_gu / p2_qkv / add_rms(o)/add_rms(down) → ≥2×
  - [ ] 位级测试（li1 位比对 + D7 + 双跑确定性）
  - [ ] 段表按 T1 判据下降
T3 门禁与记录——perf-gate.sh PASS 或 FAIL（记录）；notes.md S1-11 节；
  roadmap S1-11 行；提交。
T4（条件）若 T3 未达标——退化表：记录"需要下一阶（multi-stream/warp-spec）
  还是回退"。

## Non-Goals

- warp-specialization / multi-stream 跨阶段重叠（下一阶，若 017 不够）。
- flash 内核改动（S1-10c 判机器最优；未证明无用不改）。
- 改变聚合序/D7（位级硬门）。
- 功耗抬升本身（用户已批准,作为环境前提）。

## Constraints

- 层内 grid barrier 的参与者集合随 block 数变化——复用 S1-10b 的
  partial-participant DAG（位级透明、已保）。
- 聚合序/归约树/扫描方向零改动（位级判据）。
- smem 上限（sm_120 opt-in 101376 B；kernel 用 16 KB @4096 窗）——宽块
  不得超。
- 模型无关（无模型常量）。
