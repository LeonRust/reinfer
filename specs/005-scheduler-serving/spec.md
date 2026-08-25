# Spec: scheduler + OpenAI-compatible serving (multi-request)

> Status: proposal · Owner: maintainers · Created: 2026-08-25 · Parent: specs/003 (single-request base)

## Problem Statement

003 证明了单请求闭环。005 把 reinfer 变成服务：**多请求并发**（continuous batching、chunked prefill、token 预算准入）与 **OpenAI 兼容 HTTP**（流式、采样链）。正确性以"并发结果 == 单请求结果"为锚（确定性铁律 req_id 排序，宪法 §2.3）。

## Success Metrics

- **确定性**: 同一 prompts 集（64 并发混入）产出与逐条单请求**逐 token 完全一致**（种子相同）；2× 重复运行 bit-identical
- **吞吐**: 单卡 8B Q8_0 解码吞吐 ≥ 003 单请求吞吐 × 1.5（并发收益）；**P95 TTFT < 500ms**（8B/单卡/并发 64）
- **正确性污染**: 并发下任意两请求的输出不互相泄漏（取消/错误请求不影响他人——abort 路径注入测试）
- **工程**: 无 GPU 环境可用 `--backend cpu` 跑通全链路（101 提示词一致性套件，无 GPU CI 档）

## User Stories

1. 作为 API 用户：`POST /v1/chat/completions`（含 stream SSE、`max_tokens`、`temperature/top_p/top_k`、`stop`）OpenAI 兼容。
2. 作为服务者：`reinfer serve --model model.gguf` 即可启动；指标（tok/s、队列长度、KV 占用）经 OTel 导出。
3. 作为研究者：DP/抢占策略可插拔（schedule_policy 走 Registry）。

## Acceptance Criteria

- [ ] `Scheduler`（单线程事件循环）：token 预算准入（预计峰值 token）+ chunked prefill（1024 切块）+ decode 组批按 `req_id` 排序 + 退出阶段（abort/error 结果隔离）
- [ ] `Req` 状态机（cached_len/device_len 双游标）：Waiting→Prefill→(Chunked)→Decode→Done/Aborted；内存池页引用计数随状态释放（无泄漏：10k 请求压测后池规模回到基线）
- [ ] 服务层：`/v1/chat/completions`、`/v1/completions`、`/v1/models` + SSE 流式 + `usage` 字段与 llama.cpp server / vLLM 输出的语义一致；错误透传：`timeout/abort/invalid params` → 4xx（RFC 7807 风格可选）
- [ ] 采样链：greedy / temperature / top-p / top-k / min-p，单测对齐 vLLM 语义；任意请求失败不拖垮服务进程
- [ ] 指标：`OTel metrics`（token 吞吐/s、调度时延、KV 页占用、队列深度）; `collect_env` 诊断可用
- [ ] 端到端：`reinfer serve` + `bench/loadgen` 跑满 100 并发；结果与 vLLM/SGLang 同参 diff ≤ 仅舍入差

## Non-Goals

- RadixCache / 投机解码 / grammar（P3）；PD 分离与跨机 KV（P4）；多模态与多模型热切换；TLS/鉴权（建议网关层，P4 插件）

## Constraints

- 单进程：API（axum/tokio）+ 调度线程 + per-GPU worker 线程；无 GIL 故不用多进程（宪法 D2 思路）
- 拒绝服务/资源拮抗必须存在：优先级（新进 vs 续跑）可配置；抢占先 swap 后暂停（页式 KV 已备）
- 无 `torch`；量化/神经内核全部经 003 的 KernelProvider 路径
