# Plan: scheduler + OpenAI-compatible serving

> Derived from specs/005-scheduler-serving/spec.md · 设计报告 D2 为进程模型依据（review 修正引用）
> Performance baseline: docs/design/benchmark-gap-2026-08-29.md（Wave-2 targets；zh-CN 并存）
> Note: scheduler crate 当前为空壳（2 行）；本 spec 的 continuous batching/chunked prefill/
> token-budget admission/req_id 确定性即为 vLLM 对标件（对比门禁 P1: decode ≥85% SGLang）。

## Architecture Decisions

- **D1 调度线程模型与并发安全（评审 #5：参考实现的双 free/池竞态教训）**：单线程事件循环；**所有 CPU 可见状态（free_slots、前缀写回、页释放、表池、计数）只由调度线程突变**——结果处理（`process_last`）与调度决策（`schedule_next`）同线程串行，保持"先分配、后释放"顺序（mini-sglang 安全不变量）；worker 线程仅报告"批完成（event 同步后）"消息，不做任何池操作。真并发方案（vLLM async-scheduling 节拍门控）留 RFC。
- **D2 准入估值公式（lightllm 口径，完整定义）**：
  - 每请求估算二元组 (a, b)：`a = max(input_len + has_out_len + 1, shm_kv_len + 1)`；b：busy 时取 `max_new_tokens`（悲观），非 busy 取 `min(max_new_tokens, max(1.1×has_out_len, ema_req_out_len))`；chunked 请求另加生命周期延长项 `ceil(剩余prefill/chunk_size)×(max_waiting_token+1)` 与 `ADDED_OUTPUT_LEN=16` slack；
  - busy 判定：KV 占用 / max_total_token_num ≥ router_token_ratio；
  - EMA：请求完成时更新（初始 2048、下限 64、自适应 α）；
  - 峰值：按 b 降序后 `need_max = max_k(left_out_len[k]×k + cum_run_len[k])`；准入 = 峰值 < 预算 且 请求数 ≤ running_max_req_size 且 首 chunk token 和 ≤ batch_max_tokens（chunked 模式预算翻倍）；
  - **换算**：全部按页大小向上取整（ceil）；每运行请求预留 page_size-1 的页内 slack；
  - 章节项声明：本项目 chunked 延迟策略按 lightllm（允许生命周期延长），公式保留该项；若后续改为"绝不延迟 chunk"则将该项=1（须 RFC）。
  - vLLM 对拍范围：仅"每步 token 预算 + prefill 切块"语义（vLLM 无估计式准入——勿对拍）。
- **D3 策略族**：`SchedulePolicy`（FCFS/LPM）+ `PriorityPolicy`（恢复请求 > 新进，避免 swap 恢复饿死——虽 P1 为重算，回收路径同理）。
- **D4 流协议（backpressure 边界）**：tokio channel（上限 + 有界窗口）；丢发仅影响该请求（token 位置由采样决定，SSE 不往回补）；连续丢 `DROP_LIMIT=64` token 或写失败 → 该请求 abort；断连清场 = per-req channel close + per-uid Abort msg；SSE 事件 = OpenAI 语义（delta/usage）。
- **D5 确定性 RNG（数学定义，评审 #4）**：
  ```
  seed_i   = SplitMix64(base_seed ⊕ req_id)          // base_seed: CLI/--seed 或环境 REINFER_SEED
  rng(i,p,v) = SplitMix64(hash(seed_i, pos=p, vocab=v)) → u64→[0,1) 映射（CPU/GPU 逐位一致约定）
  greedy    = argmax(logits)、跳过 RNG；temp==0 → argmax
  ```
  采样顺序（vLLM 语义）：logit bias → penalties → bad words → temperature → min_p（对数域，`max_val+ln(min_p)`）→ top_k（第 k 大阈值）→ top_p（softmax 后 cumsum，尾项强制保留）→ gumbel 噪声（纯函数域）→ argmax；关闭语义：top_k ≤0 或 >vocab → 关闭；TopK/P 开关系数化= vLLM；`top_k==1 且 top_p<1` → 走 top_p 采样（与 vLLM 一致）。
- **D6 观测**：tick 采样 + metrics（OTel）+ tracing + collect_env。
- **D7 抢占（重算语义）**：victim 选择 = 最新/最低优先级；释放全部块（refcount-1，共享块保留）→ 状态 `Preempted`（仅记录，无独立资源）→ 回 waiting 队首；恢复时 Prefill 从头（cached=0）；swap 换页 = RFC（event 纪律 + refcount==1 + H2D 同步要求见 RFC 初案）。
- **D8 状态机唯一账目**：双游标（cached_len/device_len）派生 chunk；`Preempted` 为状态标记；stop 匹配在调度层做（每请求增量缓冲 + 部分匹配状态，无歧义延迟 ≤1 步）。
- **D9 前缀缓存（prefix cache）接口边界（追加决策，2026-08-29 gap 审计——基准 G8）**：本 spec 级承诺 = **页表设计不得做成破坏前缀复用结构的形态**（防 P3-01 返工的核心约束）+ 方法签名**草稿**（非"定死"）：`lookup_prefix(ids) -> Option<Vec<PageRef>>`（键=请求 token 前缀，规范化哈希）与 `refill_prefix(ids, pages)`（页表段登记/引用）。命中后的可见性、refcount、释放沿用 D1 单线程突变 + D8 恰一次释放/共用守卫；不破坏"先分配、后释放"与 D7 重算语义（victim 长驻前缀页的 budget/refcount 口径具体化留至 P3-01——**本决策不提前承诺**）。确定性前提（补记）：「前缀命中（仅算后缀）与全量重跑（算整段）bit-identical」要求**计算部分 launch 配置一致**，且确定性测试须覆盖 warm-cache 重跑。数据/策略（Radix 树、LRU/自驱逐）= P3-01；接口经过期若 P3-01 评估不符合（如页表 v2 换型）允许以新决策修订 D9——**不做"此刻定死"承诺**。005 spec Non-Goals 的 "RadixCache(P3)" 表述与本决策兼容（接口留缝，非实现前置）。

## Module Breakdown

| 模块 | 内容 |
|---|---|
| `crates/scheduler/src/{req,scheduler,prefill,decode}.rs` | 状态机/主循环/准入与切块（含 D2 公式）/确定性组批 |
| `crates/scheduler/src/policy/` | Policy 族（D3） |
| `crates/samplers/src/lib.rs` | 采样链（D5），GPU 实现由 003 T5 sampler 核提供 |
| `crates/server/src/{api,stream,metrics}.rs` | axum 路由/SSE/backpressure（D4）/观测 |
| `crates/engine/src/` | Engine + ModelRunner（本切片正式建立，forbid unsafe；003 T12 为最小宿主） |

## Interface Contracts（slice-local）

```rust
pub enum ReqState { Waiting, Prefill, Decode, Done, Aborted, Preempted }
pub struct Req { id: ReqId, state: ReqState, seed_i: u64, cached_len: usize, device_len: usize, stop_state: StopMatcher, ... }
pub trait SchedulePolicy { fn rank(&mut self, queue: &[PendingReq], pool: &KvPoolStats) -> Vec<usize>; }
pub fn rng(seed_i: u64, pos: usize, vocab: u32) -> f32;   // 纯函数，CPU/GPU 一致
pub struct Sampler;   // 链式配置 + 每步（分片）执行；Sampler 归属 crates/samplers
pub fn serve(engine: Engine, addr: SocketAddr) -> Result<(), ServerError>;  // engine 归属 crates/engine
```

## Reference assets（增量，全量见 深入补充 §3）

- mini-sglang `scheduler/*`、`core.py`（状态机移植大纲；**其 abort 路径缺陷不得照搬**——见 D1/D8）
- lightllm `req_queue/impl.py`、`req.py`（b/a 估算、EMA、峰值公式）；vLLM `scheduler.py`（仅重算语义/预算子集对拍）
- llama.cpp `tools/server`（SSE/usage 语义）；vLLM `gumbel.py`（纯函数 RNG 范式）

## Risk Assessment

| Risk | Mitigation |
|---|---|
| 调度顺序漂移 → bit-identical 破 | D5/D8 单一账目 + 到达序为输入 + 确定性测试（120× 随机种子 × 2 runs） |
| 准入估算偏差 → OOM | 全局水位 + 重算抢占（D7）为硬保险；swap 仅 RFC |
| backpressure 误杀（实测丢 token/误清场） | D4 精确边界；abort 延迟上界文档化 |
| 参考实现移植引入双 free/池竞态 | D1 串行化 + 恰一次守卫 + in-flight abort 测试 (Task1 矩阵) |
