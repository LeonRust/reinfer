# Spec: prefix cache — shared-prompt KV re-use (P3-01 v1, D9 concrete)

> Status: draft-r2 (2026-09-01, adversarial review applied) · Parent: specs/005
> (D9 interface draft) · Requires: S2 wave landed.

## Problem Statement

Every request re-computes its full prompt KV. Serving loads with a shared
system prompt / multi-turn history pay O(prompt) attention+MLP for identical
tokens on every request. 005 D9 committed to an **interface-compatible,
structure-preserving** shape; this spec delivers P3-01 **v1**: a page-run
prefix cache that cuts prefill compute on cache hits while keeping the
engine's identity-page-table, contiguous-segment design **unchanged in the
compute path**.

## Success Metrics (口径锁定; r2 修订经 016 adversarial review)

- **性能（硬门，P3-01 标的）**：同 prompt 串发（prompt ≥ 512 token，页对齐前缀），
  温缓存请求 TTFT p50 ≤ **0.5×** 冷缓存（首请求）。场景=整 prompt 复用的
  共享系统提示（v1 键语义=D9 补记 + 评审 #5：**整 prompt 的页对齐前缀**；
  部分前缀/非整 prompt 复用为 v2）。
- **一致性（硬门，D9 前提按项目判据体系解读）**：
  - **同路径 bit-identical**：温缓存双跑（同 seed、同 launch 配置）逐 token
    bit 级一致；
  - **温 vs 冷**：按 014 T1 的 F16 判据档（greedy token 100% 一致、logits
    drift ≤ 1e-2）——**非逐位**（跨内核 FMHA/per-token 数值不承诺逐位，与
    vLLM 前缀缓存的现实口径一致）。D9 原文"bit-identical（仅算后缀 vs 算
    整段）"在 r2 明确为同 launch 配置限定，跨路径用 F16 档；
  - 判定档写进验收表（T1 口径相同表头）。
- **池守恒（硬门）**：refill/驱逐全程 `assert_conserved`；**无泄漏**（同键
  连发不得积累额外引用——r2 修复；评审 #3）；abort/抢占直 free 不进缓存
  （评审 #4）。
- **回归（软门，记录项）**：`REINFER_PREFIX_CACHE=off` 与 S2 wave 行为位级
  相同（同一代码路径、no-op 检查点）；c1/c4/确定性验收复跑绿。
- **工程**：`REINFER_PREFIX_CACHE=on/off`（缺省 on）；`REINFER_PREFIX_CACHE_PAGES`
  预算（per-layer 页；缺省 = kv_pages × 10%，≥ 1）。

## User Stories

1. 作为服务者：同一系统提示的多个串发请求，TTFT 显著下降（≥2×），无需配置。
2. 作为研究者：LRU 驱逐预算可调、命中/驱逐计数可观测。
3. 作为确定性依赖者：on/off、同路径重跑，greedy 输出恒一致（F16 判据档）。

## Acceptance Criteria

**T0 缓存前端（纯 Rust，无 CUDA 依赖）**
- [ ] `TokenRadixCache`（crates/scheduler/src/radix.rs）：token 前缀树匹配
      （页对齐粒度），`lookup` 返回**最长页对齐命中**；无部分页（v1
      Non-Goal——split 树 v2）。
- [ ] entry = (前缀 token 键, `base_page`, `L`, LRU 序)。前缀 run 地址表示：
      `entry.base_page` + 层步长 pp（层 li 的 run = `[base + li*pp, +L)`；
      这是 n_layer 个不连续 run——见 plan D1）。
- [ ] `REINFER_PREFIX_CACHE_PAGES` 预算（缺省 = 池总页 × 10%，≥ 1）+ LRU
      驱逐（evict 回调→调用方 per-entry 逐层 `pool.unref`）。
- [ ] 单线程契约：无 RNG/哈希序；两次同序列操作 entry 检查表一致。
- [ ] 测试 ≥ 12：串行链/分支/复用/预算精确驱逐/空键与 L=0 拒绝/LRU 次序/
      **同键重插入不复制 ref（自身不新增引用计数）**/确定性双跑。

**T0b 池反射测试**
- [ ] mock pool（Vec<u32> refcount，模拟 ref_/unref/free 语义：
      ref_=每页+1；free=每页-1、归 0 入 free 列表；unref=相同）注入 radix：
      两个主要场景（评审 #3/#4 修正后）：
      (a) **首次 refill**：`ref_(prefix 每层 run)` → `free(seg)` → 前缀页 ref
      =1（缓存独有）、后缀页=0 归池；
      (b) **同键后 refill**：裸 `free(seg)`（无 ref_）→ 全部页=0 归池，无残留；
      (c) eject（LRU）后前缀页 0 归池、free 列表无相邻 run。

**T1a 命中计算路径（无新引擎 API——评审 #1 修正：FMHA 前缀盲）**
- [ ] v1 命中后的剩余计算**不用 FMHA batch prefill**（`launch_batched_prefill`
      只读本 chunk KV，不感知池前缀——fmha.rs:711 起签名无反 Pool；引擎
      注释 engine.rs:5233 自认"注意力段不感知 pos"）。命中路径 =
      **逐 token decode 步**：`copy_prefix_to_engine` 后对后缀 token 依次
      `engine.step(tok, pos, pos+1)`（各步注意力读全窗 KV `[0, pos+1]`，
      含复制前缀——与 vLLM "miss 点后即 decode" 行为同构）。
- [ ] 后缀为 1..(≤1 block)token 时成本≈ 单 decode 步（v1 场景=整 prompt
      复用 → 后缀极小）；长后缀性能退化记录为非目标（v2 池读 FMHA 后再议）。
- [ ] 验证测试：`engine.step(tok, pos, pos+1)` 全窗读写位级 == 既有行为
      （pos 参数语义已由 S1-10/s2 使用；补一个"跨边界 pos 写读+注意力覆盖
      [0,pos]"的显式测试）。

**T1b Executor 钩子（评审 #2/#6 修正）**
- [ ] **单个 executor 方法** `prefill_prefix_hit(entry, ids_suffix)`：
      flush_singleton → copy_prefix_to_engine（逐层页精确 D2D K+V，
      CtxGuard）→ 后缀逐 token step（一条方法保证次序：不会发生
      "copy 污染旧 singleton 后再 flush" 的次序陷阱）。
- [ ] `refill_hook(seg, prompt_bytes, prompt_len)`（**仅 Done 释放守卫**，
      abort/抢占直 free）：先 **flush_singleton（若 singleton 是该请求）**
      ——否则 B=1 世界段未写、refill 收垃圾（评审 #2 P0）；然后 D2 序列：
      `L<2` → 直 free；**同键 → 裸 `pool.free(seg)`（评审 #3；旧 entry 已
      服务该键，不做 ref_）**；新键 → `ref_(逐层 run)` → free(seg) → insert。
- [ ] 释放守卫只扩展一处（D8 恰一次），无第二释放点。

**T2 调度器/服务接线**
- [ ] prefill 前 `lookup`；命中 → `prefill_prefix_hit`；未命中 → 现状
      `prefill()`（FMHA 路径不变——**命中路径不得触碰 TuneDb/采样/图状态**）。
- [ ] refill 时机=terminal（done）释放守卫（单线程）；键=整 prompt
      （页对齐后）；off → 双 no-op。
- [ ] 观测：命中/未命中/驱逐 stderr 行 + 池统计字段。

**T3 验收（真机；判据按 r2 口径）**
- [ ] warm/cold：同 prompt（≥512 token）× nreq=10 **串发**（首请求=冷，
      2..10=温——perf.py p 档逐请求 await；harness 加首请求标记或 off/on
      两次运行对比单元——评审 #7）；温 p50 ≤ 0.5× 冷 p50。
- [ ] 温双跑 bit-identical；温 vs 冷 014 F16 判据档（greedy token 100% +
      drift ≤1e-2）。
- [ ] 守恒+泄漏：同键 10 连发后 assert_conserved；含 abort 全程守恒；
      小预算驱逐压力测试。
- [ ] 回归：off 位级 == S2 wave；c1/c4 复跑记录。

## Non-Goals (v1)

- 部分前缀命中 / 页内分裂 / 跨请求链式匹配（v2：显式页表两径内核）。
- 长后缀的 batched 计算（FMHA 池前缀读——内核级 v2 候选；
  v1 长后缀性能退化记录）。
- **并发共享计数**（v1 refill 只在请求终结后；同键并发直接 miss——评审 #5
  明确无并发 ref 需求；多个请求同 prompt 并行时依然全部 miss，第 2 个
  完成者可取第 1 个的缓存——串发才是收益场景）。
- 显存收益承诺（命中=计算省 + 复制成本 ~14 MB/512-token 前缀（评审 #8
  实算：#每层 16 页×28 层×32 KB/页 = 14.3 MB，非 4-8 MB））。
- swap/E2E、跨实例。

## Constraints

- 005 D1 单线程 + D8 恰一次释放（refill 并入释放守卫）。
- 005 D9：页表 v1 形态不动（identity 恒等）；**D9 的 bit-identical 前提按
  r2 解读**（同 launch 配置限定；跨路径 F16 档）——D9 修订记录引用本文档。
- 模型无关（geometry 走 config 既有接口）。
- 命中路径不得改变采样/TuneDb/JitGemm/graph 状态（D3）。

## References

- specs/005-scheduler-serving/plan.md D9 + 2026-08-29 补记
- docs/design/benchmark-gap-2026-08-29.md G8
- roadmap Stage 2 S2-4 / bench/notes.md S2-D 节
- 016 adversarial review 2026-09-01（8 条，全部应用到 r2）
