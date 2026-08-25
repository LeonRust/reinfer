# Plan: scheduler + OpenAI-compatible serving

> Derived from specs/005-scheduler-serving/spec.md

## Architecture Decisions

- **D1 调度线程模型**：单线程事件循环（mini-sglang 蓝图）——`schedule()` 仅操作整数索引/句柄队列；`Batch`/`BatchHeap` 由 worker 线程走 `&batch` 借用；tokenizer 前端独立预算化。
- **D2 预算准入**：以"预计峰值 token 数"(queue + decode 数组) 计算内存预留给新请求，参考 lightllm `BaseQueue` 的 `estimated_token` 准入与 `ChunkedPrefillQueue`。
- **D3 调度策略插件族**：`SchedulePolicy` trait（FCFS / LPM / batched），Registry 注册（对应 SGLang LPM/DFS 的工程化子集）。
- **D4 流协议**：token-buffer 到 HTTP 双向解耦（tokio channel + backpressure）；SSE 事件用 OpenAI 语义按 `delta`块 传递；end 事件带 usage。
- **D5 采样链**：`Sampler` 由不可变配置 + 状态分离；弃用 `rand`（确定性）——用 `SplitMix64` 固定种子；采样器在 GPU 端走 003 的 `kernels/samplers`（可行时），末解禁 fallback CPU。
- **D6 观测**：tokio 计时器 tick 采样 + `metrics` (OTel) + `tracing`；`collect_env` 复用 flashinfer 思路。
- **D7 优先级与抢占**：`PriorityPolicy`（默认 FIFO + 偷跑优先）；swap 路径先把活跃页写宿主池再回收（`crates/memory` 已备），暂停路径先把该 req 从 batch 摘除但不释放页。

## Module Breakdown

| 模块 | 内容 |
|---|---|
| `crates/scheduler/src/scheduler.rs` | 主循环（prepare/decision/build/deliver） |
| `crates/scheduler/src/req.rs` | `Req` 状态机 + 双游标 |
| `crates/scheduler/src/prefill.rs` | `PrefillManager` + `ChunkedReq` |
| `crates/scheduler/src/decode.rs` | `DecodeManager`（uid 排序防跨 rank） |
| `crates/scheduler/src/policy/` | `SchedulePolicy` + FCFS/LPM |
| `crates/server/src/api.rs` | `POST /v1/*` + SSE + 错误映射 |
| `crates/server/src/stream.rs` | 上半程 backpressure（channel, window） |
| `crates/server/src/metrics.rs` | OTel 指标 + tracing |

## Interface Contracts (slice-local)

```rust
pub enum ReqState { Waiting, Prefill { chunk: usize }, Decode { gpu_len: usize }, Done, Aborted }
pub struct Req { id: ReqId, state: ReqState, seed: u64, tokens: Vec<u32>, ... }
pub trait SchedulePolicy { fn rank(&mut self, queue: &[PendingReq], pool: &KvPoolStats) -> Vec<usize>; }
pub struct Sampler;             // greedy/topk/topp/minp + SplitMix64 链
pub fn serve(engine: Engine, addr: SocketAddr) -> Result<(), ServerError>;   // axum 装配
```

## Reference assets

- mini-sglang `scheduler/*`（整体移植大纲）、`core.py` 状态机
- lightllm `BaseQueue/ChunkedPrefillQueue`（准入/切块）
- vLLM `v1/core/sched/scheduler.py`（预算、抢占 swap/暂停策略）
- llama.cpp `tools/server`（流式/usage 语义、错误码）

## Risk Assessment

| Risk | Impact | Mitigation |
|---|---|---|
| 并发一致性被破坏（调度顺序漂移） | High | 唯一确定性来源=req_id 排序；`Cargo test --features cpu` 并发乱序集 diff 断言 |
| 准入估偏差导致 OOM | High | 每 req 预算上限 + 全局水位；swap 作为硬保险（D7） |
| SSE backpressure 断电 → worker 阻塞 | Medium | channel 带限 + 超时丢发（前端断连清场） |
| 采样器与 vLLM 语义差异 | Medium | 语义单测（分布与截断规则，用固定种子对照 vLLM 输出） |
