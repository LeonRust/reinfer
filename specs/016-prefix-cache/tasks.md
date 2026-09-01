# Tasks: prefix cache v1 (specs/016 — r2, review applied)

> Status: open 2026-09-01 · Wave A parallel: T0a ‖ T1a-test → B: T1b → C: T2 → D: T3
> Registration: roadmap S2-4；005 D9 具体化（r2 引用口径修订）。
> 变更（vs r1，评审 8 条全部应用）：无 prefill_batch_from（FMHA 前缀盲——
> 命中剩余走逐 token step）；refill 只在 Done、逐层 ref_、同键裸 free（防
> 泄漏）；键=整 prompt 页对齐前缀；验收判据方向修正（温 ≤ 0.5× 冷，非
> 温/冷 ≥2×）；复制成本 ≤14.3 MB 口径（512 前缀）。

## T0a — TokenRadixCache（纯 CPU 前端）

- [ ] `crates/scheduler/src/radix.rs`：`TokenRadixCache`（键=token 前缀切片
      页对齐；entry=`(key_span, base_page, L, lru)`；LRU 链表；页预算数
      per-layer 页）
- [ ] `lookup(ids) -> Option<Hit{ base_page, pages, key_len }>`（最长页对齐命中）
- [ ] `insert(key, base, pages) -> Result<Vec<Evicted>, Error>`：**同键重插入
      返回 Ok(无驱逐) 且不新增引用（树不复制 ref；调用方同键裸 free）**;
      超预算先驱逐最老（evict 回调返回列表）
- [ ] 单线程契约：无 RNG/哈希序；两次同序列 ops entry 检查表一致
- [ ] 测试 ≥ 12：串行链/分支/复用/预算精确驱逐/空键与 L=0 拒绝/LRU 次序/
      同键重插入/确定性双跑/分叉链（两键共享前缀，lookup 各自正确）

## T0b — 池反射测试（scheduler crate 内 mock）

- [ ] mock pool（Vec<u32> refcount；ref_=+1；free=每页-1 归 0 入 free
      列表；unref=free 语义；free 列表无相邻 run 不变式）：
      (a) 首次 refill（ref_ 逐层 → free(seg))：前缀页 ref=1、后缀页=0、
          free 列表无相邻; (b) **同键后 refill：裸 free(seg) → 全部归 0，
          无残留引用（泄漏回归）**; (c) eject 后全部归 0。
- [ ] 不变式检查器 `assert_conserved` 的等价检查全绿。

## T1a — step 全窗语义验证（无新引擎 API —— r2 修订）

- [ ] 验收 `engine.step(tok, pos, pos+1)` 的 KV 写读全窗语义（已运行于
      S2 引擎,补显式测试固化）：(i) pos 跨页边界（pos=31/32/63 …) 写入
      物理页 `li*pp + pos/32` 正确; (ii) 解码步注意力读 `[0, pos+1]`
      （logits 与"逐步从 0 写"路径的该步结果一致——同内核同输入）。
- [ ] 测试位置：`crates/cuda/tests/` 新增 `prefill_step_offsets.rs`
      （真机档 RTX 5090；接入现有测试模式）或并入 `batch_decode.rs`。
- [ ] 回归：cargo test -p reinfer-cuda --features cuda 全绿（现有套件不动,
      只加）。

## T1b — Executor 单方法命中 + refill_hook（依赖 A 结果）

- [ ] `prefill_prefix_hit(hit, ids_suffix)`：flush_singleton →
      `copy_prefix_to_engine`（逐层 K/V 页精确 D2D,层步长 pp,同步
      CtxGuard）→ 后缀逐 token `engine.step(tok, pos, pos+1)`（一次调用
      保证顺序——评审 #6 次序陷阱修复）
- [ ] `refill_hook(seg, prompt_len)`：仅 Done 守卫接入点;内部：
      flush_singleton(若 singleton 是该请求 id) → D2 序列（L<2 直 free /
      同键裸 free / 新键逐层 ref_ + free + insert）+ 预算驱逐
- [ ] 位级集成测试（真机档）：首次 refill 后二次同 prompt 请求输出 ==
      **copy 后再 step 序列** == 温双跑 bit-identical；放"无缓存"冷对照
      （F16 档：greedy token 100% + drift ≤ 1e-2）。
- [ ] abort 路径不 refill（直 free）守恒。

## T2 — 调度器/服务接线

- [ ] sched_loop：prefill 前 lookup（on 时）→ 命中走 `prefill_prefix_hit` /
      未命中走现状; terminal 释放守卫内调 `refill_hook`（恰一次）
- [ ] `REINFER_PREFIX_CACHE`（缺省 on;off 双 no-op 且位级 == S2 wave 路径）;
      `REINFER_PREFIX_CACHE_PAGES`（缺省 kv_pages×10%，≥1;入口
      成本 L×n_layer 页）
- [ ] 观测 stderr + 池统计; 文档: 005 D9 补订记录（r2 口径引用）、
      roadmap S2-4、feature-list（P1-06 项）
- [ ] 回归：off 时 c1/c4 与 S2 wave 记录比对（上次表格）

## T3 — 真机验收（RTX 5090 / Qwen3-0.6B;判据按 r2 口径）

- [ ] warm/cold TTFT：同 prompt（≥512 token）串发 nreq=10（perf.py p 档
      或脚本化 openai 串发;首请求=冷、2..10=温）;**判据：温 p50 ≤ 0.5×
      冷 p50**
- [ ] 一致性：温双跑 bit-identical；温 vs 冷 = greedy token 100% +
      logits drift ≤1e-2（014 F16 判据档）
- [ ] 守恒/泄漏：同键 10 连发 assert_conserved 全程；abort 中断场景;小预算
      驱逐压力（cached_pages 归零）
- [ ] 回归表：off == S2 wave（c1/c4/确定性）;记录表格
- [ ] notes.md "P3-01 验收" 节 + roadmap S2-4 标注 ✓
