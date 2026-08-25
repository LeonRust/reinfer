# Tasks: scheduler + OpenAI-compatible serving

> Derived from specs/005-scheduler-serving/plan.md

## T1: Req/Batch/SamplingParams + 状态机（含 abort 恰一次）

- 双游标唯一账目；`ReqState`+`Preempted`；tombstone abort（批完成后与 finish 共用一次释放守卫）；abort-after-done 幂等
- Verification: 状态机单测矩阵 —— 每状态 × {finish, abort(含 in-flight), preempt, resume, abort-after-done}；`abort 于批完成前 → 无二次释放` 专项断言（池计数）

## T2: 准入估值（D2 公式）

- 完整 a/b/EMA/busy/峰值 公式；token→页 ceil + page_size-1 slack；chunk 生命周期延长项（可配置=1 以备 RFC）
- Verification: 公式单测（对拍 lightllm 算例 5 组）；vLLM 对拍仅限"每步预算/切块"子集

## T3: DecodeManager + 确定性组批

- req_id 排序；`--seed`/`REINFER_SEED` 贯通；采样 RNG 纯函数（与 003 T5 一致）
- Verification: 乱序输入同结果；同种子 2×bit-identical；不同种子结果不同（抽样）

## T4: 主循环（含并发安全不变量）

- 「结果处理与调度同线程串行 + 先分配后释放」；worker 仅报"批完成(事件同步后)"
- Verification: 与 mini-sglang 对拍并发安全点（abort 注入 → 无双 free）；CPU 全链路 10 请求混合

## T5: HTTP + SSE + backpressure 边界

- `/v1/chat/completions|completions|models`；`stop` 调度层匹配（≤1 步）；`n>1`→4xx；丢发/清场语义（D4）
- Verification: 与 llama.cpp server OpenAI 兼容 diff；断连注入 → 仅该请求 abort；TTFT 计数点=首个 delta 字节

## T6: 采样链

- greedy/temperature/top-p/top-k/min_p 顺序与关闭语义（vLLM 矩阵）；greedy 无 RNG；temp==0→argmax
- Verification: 参数组合断言矩阵（greedy 100% 对拍；非 greedy 分布统计对齐）；top_k==1 & top_p<1 路径专项

## T7: 端到端（口径按 spec）

- 闭环 64 in-flight、prompt 512、样本 ≥10k、预热 1k；loadgen 同机；吞吐=调度层口径；TTFT P95 记录；池基线（在用页==0 + 空闲表==预热长度）
- Verification: 确定性硬门（2×bit-identical）+ 软门（logits ≤1e-3 / token ≥99.9%）+ 吞吐 ≥1.5× + P95<500ms + 池基线；abort oracle（三类注入点）

## T8: 观测与文档（Exemption lane 落地）

- OTel metrics/tracing/collect_env；docs/sdd Exemption lane 入册（判据+PR 记录）；README/API 文档（stop/seed/n>1 语义，parity 矩阵路径）
- Verification: `/metrics` 可见；文档与 008-ci-infra 矩阵条目一致

---

Completion gate: T1–T8 accepted；硬门+软门记录；评审通过。下一片：specs/006（vendor+graph）与 008-ci-infra。
