# Tasks: scheduler + OpenAI-compatible serving

> Derived from specs/005-scheduler-serving/plan.md

## Task 1: `Req`/`Batch`/`SamplingParams` 核心类型 + 状态机

- 双游标 `cached_len/device_len`、`ReqState` enum、`SamplingParams`（temperature/top_p/top_k/max_tokens/stop）
- Verification: 状态机转移单测（各分支 + abort at every state）

## Task 2: PrefillManager + ChunkedReq

- 预算准入（峰值 token 估算）+ 1024 切块 + 请求队列（Backpressure 上限）
- Verification: 排队/插入/切块顺序测试；与 vLLM SchedulingBudget 语义对拍

## Task 3: DecodeManager + 确定性排序

- 按 `req_id` 排序组批；`uid`/池分配稳定；与 Task1/2 构成完整 `ScheduleStep`
- Verification: 乱序输入同结果；`req_id` 排序不变量测试（随机 64 请求 × 100 轮）

## Task 4: Scheduler 主循环（线程 + 通道）

- `overlap_loop/normal_loop`（mini-sglang 移植）；状态清账：批上/下移交
- Verification: CPU 上 `--backend cpu` 全链路跑通 `SchedulerSmoke`（10 请求混合）

## Task 5: 服务层 API + SSE

- axum：`/v1/chat/completions`、`/v1/completions`、`/v1/models`；SSE 流；usage 字段收敛
- Verification: 与 llama.cpp server 的 OpenAI 兼容 diff（tools 测试）；断连/超时路径注入

## Task 6: 采样链

- `Sampler` 链（greedy/temperature/top-p/top-k/min-p）+ `SplitMix64` 固定种子 + CPU fallback
- Verification: 语义单测（固定种子对照 vLLM 输出序列 100% 一致（greedy））；分布采样统计检查（温和）

## Task 7: 端到端并发一致性 + 泄漏压测

- `${loadgen: 64 并发}`；与单请求逐 token 比对；10k 请求后池恰回到基线（无泄漏）
- Verification: 一致性 100%；`tok/s ≥ 1.5×` 003 单请求；P95 TTFT < 500ms 记录

## Task 8: 观测与文档

- OTel metrics + tracing + `collect_env`；README/`docs/` 补 serving 用法与指标表
- Verification: `curl /metrics` 可见；`tracing` 链路（RUST_LOG）可用

---

Completion gate: Tasks 1–8 accepted; 并发一致性 100%; 吞吐/时延记录备案；评审通过。下一片：P3 规格（Radix/投机/grammar）。
