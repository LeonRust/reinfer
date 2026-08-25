# Spec: scheduler + OpenAI-compatible serving (multi-request)

> Status: approved (review 2026-08-25) · Parent: specs/003 (single-request base) · **Dependency: specs/004 (tokenizer)** · Feeds on 003 (kernels incl. sampler)
> 修订记录：确定性锚数学化；抢占改为重算（swap→RFC）；abort 恰一次释放；指标/并发定义；准入公式完整化；stop/`n>1` 决策；spec 层去厂商化。

## Problem Statement

003 交付单请求闭环。005 交付服务化：多请求并发（continuous batching、chunked prefill、预算准入）与 OpenAI 兼容 HTTP。核心锚：**确定性**（同输入顺序重跑 bit-identical；乱序并发结果与基线对齐）与**隔离**（abort/错误不污染他人）。

## Success Metrics（口径全部锁定）

- **确定性（硬门）**：同输入顺序（loadgen 到达序为输入的一部分）2× 重跑 bit-identical；`--seed <base>` 显式可控
- **批 vs 单（软门，记录项）**：64 并发 vs 单请求 logits 相对漂移 ≤1e-3 + greedy token 一致率 ≥99.9%
- **采样语义**：greedy = argmax(logits)（无 RNG），与参考实现逐 token 一致；**非 greedy 仅承诺与 vLLM 分布统计对齐**（频次/熵），不承诺逐 token
- **吞吐（口径）** = 调度层采样确认的生成 token 数 / 墙钟（不含 prompt token、不含 SSE 投递抖动）；单卡 8B Q8_0 并发 64 下 ≥ 003 单请求 × 1.5
- **TTFT** = HTTP 到达 → 首个 SSE `delta` 字节；闭环 64 in-flight、prompt 512 token、样本 ≥10k、预热 1k、loadgen 同机、机型 = gpu-runner 固定；P95 < 500ms
- **池基线**：10k 请求后"在用页数（refcount>0）== 0 且空闲页链表长度 == 预热后长度"（不按 slab 容量判定）
- **Abort 隔离**：注入点 Waiting/Prefill(含 chunk 中)/Decode 各若干；断言=其余请求输出与无 abort 基线运行逐 token 一致 + 被 abort 请求 KV 引用归零
- **工程**：`--backend cpu` 全链路一致性套件（无 GPU CI 档）；metrics（OTel）指标可见

## User Stories

1. 作为 API 用户：`POST /v1/chat/completions`（stream SSE、max_tokens、temperature/top_p/top_k/min_p、stop、seed）OpenAI 兼容；`n>1` 暂返回 4xx 并文档声明。
2. 作为服务者：`reinfer serve --model model.gguf` 启动；OTel 指标与 `collect_env` 可用。
3. 作为研究者：`SchedulePolicy`/`PriorityPolicy` 经 Registry 可插拔。

## Acceptance Criteria

- [ ] 调度线程模型：单线程事件循环；**结果处理（采样确认、cache 回填、页释放、计数）与调度同线程串行化**，并保持"先分配、后释放"顺序（见 plan D1/D8）
- [ ] `Req` 状态机：Waiting→Prefill→(Chunked)→Decode→Done/Aborted/**Preempted**；双游标为唯一账目来源（无独立 chunk 计数器）；abort-after-done 幂等 no-op；abort 用 tombstone、资源在批完成后**恰一次**释放（与 finish 共用守卫）
- [ ] 准入估值公式按 lightllm 口径完整实现（见 plan §预算公式）——a/b 二元组、busy 阈值、EMA、峰值排序公式、token→页向上取整 + 每运行请求 page_size-1 slack、chunk 生命周期项声明
- [ ] 抢占语义 = **重算**（vLLM 方式：释放块、num_computed_tokens 归零、回 waiting 队首；victim=最新/最低优先级）；swap 转 RFC
- [ ] Scheduler 策略可插拔（FCFS/LPM）+ 优先级：恢复请求先于新进
- [ ] 采样链（契约见 plan D5）：greedy/temperature/top-p/top-k/min-p，顺序与"至少保留 1 token"规则按 vLLM 语义；RNG 纯函数
- [ ] `stop` 字符串：**调度层**增量缓冲 + 部分匹配状态（停止延迟 ≤1 步）；tokenizer 只做字节解码
- [ ] HTTP：`/v1/chat/completions`、`/v1/completions`、`/v1/models`；SSE 流（delta+usage）；错误 4xx；`n>1` → 4xx + 文档声明
- [ ] SSE/backpressure：采样与投递解耦；丢发仅限该请求；连续丢 N 或写失败 → 该请求 abort；断连清场 = per-req channel 关闭 + per-uid abort（不触共享池）
- [ ] abort 延迟上界文档化：常态 ≤1 调度迭代；in-flight chunk 中期 → ≤1 chunk 时长（约百 ms 级）+1 迭代
- [ ] metrics：吞吐/token 计数（调度层口径）、队列长度、KV 页占用；`collect_env` 诊断

## Non-Goals

- RadixCache / 投机 / grammar（P3）；PD 分离（P4）；多模态；TLS/鉴权（网关层）；swap 换页（RFC）；多 API 进程集群

## Constraints

- 经 003 KernelProvider 路径；无 torch；数值裁判与管理语义参照 vLLM（子集，见 plan 对拍清单）
- 并发正确性铁律（§2.3）：req_id 排序 + 种子显式；确定性输入 = 到达序
- CI 硬门禁工件见 specs/008-ci-infra（本 spec 引用，不自行定义 job 细节）
