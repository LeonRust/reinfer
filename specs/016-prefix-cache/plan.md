# Plan: prefix cache v1 — architecture decisions (r2, review applied)

> Derived from specs/016-prefix-cache/spec.md (r2) · Parent 005 D9 ·
> Substrate: S2 wave. r2 applies the 2026-09-01 adversarial review (8 items).

## Architecture Decision Record

### D1 缓存粒度与物理地址（页对齐 run，每层一个 run）

设备布局（已核）：请求段 = **全窗**页数（`n_layer × ceil(max_len/32)`，
`alloc_segment` 以 `window_pages()` 调用——sched_loop.rs:780），层步长 =
**全窗 pp**（`copy_engine_to_pool`/batch identity 表均按 `li*pp + j` 编址）。
前缀 run 因此是**每层一个** L 页 run：

```text
prefix_run(entry) = { base: u32, L: u32 }                  // L = floor(prefix_len/block_len)
物理地址(li, j)    = (base + li*pp + j)   × 页大小         // li=n_layer 个不连续 run
```

`pool.ref_`/`unref`/`free` 是**单连续 run API**——对前缀 run 的调用一律
**逐层 28 次循环**（memory crate 只认识 (base, n_pages) 单 run；不扩 API，
T1b 在 executor 侧循环——评审 #8）。

### D2 entry 生命周期（refill 只在 Done 守卫；逐层 ref_；无同键泄漏）

v1 允许进入缓存的**只有正常 Done 释放**（abort/抢占直 `pool.free(seg)`——
评审 #4；这些路径无 tombstone 且段内容不可靠）。释放守卫扩展为：

```text
refill_hook(seg, prompt_len):
  1. flush_singleton(该请求)                      // B=1 世界段未写——P0（评审 #2）
  2. L = floor(prompt_len/32)                     // 页对齐
  3. L < 2                        → pool.free(seg)
  4. 树上已有同键 entry           → pool.free(seg)      // 裸放——旧 entry 已服务该键
                                                        // （评审 #3：不做 ref_，无泄漏）
  5. 新键                       → for li in 0..n_layer: pool.ref_(run(li, L))
                                  pool.free(seg)        // 前缀页 ref 2→1=缓存独有
                                  tree.insert(key, entry)
```

- 顺序 = **先 ref_ 后 free**（页不瞬时归零）；前缀页最终 ref=1（缓存独有）、
  后缀页 0 → 归池。
- 驱逐：`evict()` 在预算不足/insert 时拔最老 entry → 逐层 `pool.unref` →
  页归池。
- 保守不变量：**缓存页永远 refcount=1（池持有）**,除瞬态 2。
- 消费**不再计算前缀页的 KV 写入**（命中后从 L 起步进）——省的是
  FMHA 计算 + KV 写。

### D3 命中计算路径与确定性（评审 #1 修正——FMHA 前缀盲）

`launch_batched_prefill` 无池/前缀参数，且引擎 prefill 注意力"不感知 pos"。
**v1 命中后的剩余部分走逐 token decode 步**（`engine.step(tok, pos, pos+1)`，
各步注意力窗口 = `[0, pos+1]` 全窗 KV——含复制的前缀），与 vLLM "miss 点后
即 decode" 的行为同构；命中路径**不触碰 TuneDb / FMHA 选择 / 采样 / graph**。

确定性断言（两组，判据不同——引用 spec r2 口径）：
- 同路径（温双跑）：同一 launch 序列 ⇒ bit-identical（既有确定性命门）。
- **温 vs 冷**：跨内核（FMHA batch prefill vs per-token），按 014 T1 的
  **F16 判据档**（greedy token 100%、logits drift ≤1e-2）——与 vLLM 前缀
  缓存现实口径一致；**D9 原"bit-identical"按"同 launch 配置"解读**，
  跨路径档位记录为 D9 修订。
  - 若后续 v2 为 FMHA 加池读参数（内核级）并保持同扫描顺序，
    则跨路径可升级回逐位（候选，不在 v1 承诺）。

### D4 树结构（v1：整 prompt 页对齐前缀，单链匹配）

键 = 请求**整 prompt**（渲染后 id 序列）的**页对齐前缀**（L×32 的倍数段）。
lookup(prompt) = 树上最长页对齐匹配（L = floor(len/32)，最多命中 len-32+1
内的整页）；**部分前缀不匹配**（v1）。**明确**:两个不同 prompt 仅在尾段
不同的场景 v1 不命中（这是 v2 树的价值）——v1 验收场景=**整 prompt 相同**
的共享系统提示（G8 的串发基准）。树形状：前缀链 + parent 共享（避免重复
存储）,insert/lookup 纯函数。

### D5 预算与开关

- `REINFER_PREFIX_CACHE_PAGES`：per-layer 页数预算（**口径钉死**：executor
  kv_pages（per-layer）,入口成本 = L × n_layer 页——评审 #8),缺省 =
  kv_pages × 10%（向下取整≥1）。超预算 refill → 先驱逐最老。
- `REINFER_PREFIX_CACHE=off` → `lookup`=None + `refill_hook`=裸 free
  （同 S2 wave 路径;回归基线）。

### D6 观测

stderr 行：`prefix-cache: hit(L=..)/miss/evict(n)` + 池统计
`cached_pages`（记录项）。

## Module/File Plan

| 单元 | 文件 | 内容 |
|---|---|---|
| T0a | `crates/scheduler/src/radix.rs` | 树/lookup/insert/evict/预算/LRU+测试（无 CUDA） |
| T0b | 同上 (+lib.rs) | mock 池反射（D2 三序列） |
| T1a | `crates/cuda/tests/*`(新增测试) + 可选：`bin/...` 验证 | **无新引擎 API**；验证 `step(pos)` 全窗语义（pos 跨页边界、[0,pos] 注意力覆盖） |
| T1b | `bin/reinfer/src/sched_loop.rs` | 单方法 `prefill_prefix_hit` + `refill_hook`（逐层 D2D/ref_）+ 顺序保证 |
| T2 | `bin/reinfer/src/{sched_loop,serve}.rs` | 释放守卫接线/开关/预算/env/日志 |
| T3 | `bench/notes.md` + bench-vs-vllm | warm/cold 对比单元 + 验收表 |

## Wave plan (parallel)

```
Wave A (parallel ×2)  T0a ‖ T1a-test（纯前端 ‖ step 全窗验证）
Wave B (serial)       T1b（单方法命中 + refill_hook）
Wave C (serial)       T2（接线/开关/预算）
Wave D                T3 真机验收 + 记录
```

> Note: T1a 原 "prefill_batch_from" 已废除（FMHA 前缀盲——评审 #1），
> 无引擎 API 改动；engine.rs 只加测试不改签名。
