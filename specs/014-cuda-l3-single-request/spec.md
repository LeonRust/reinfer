# Spec: CUDA L3 — single-request full loop（第一次跑出 token）

> Status: proposal · Owner: maintainers · Created: 2026-08-27
> Parent: phase-plan §L3（003 T9-T12 + 004）· 实施锚：001 T1-T4 / 004（spec-ready，无代码）/ 003 T8-T12（评审过）/ 000 parity.md / phase-plan 决策记录（模型一律 ModelScope：**经 specs/013 resolver 获取，仅文档示例可含真名——代码零硬编码**）
> 前序：009（L1 运行时基座）/ 012（L2 JitCache + 首批内核 + 选择链）已交付并真机验证

## Problem Statement

L1/L2 证明了"能安全消费运行时"与"能编译并跑我们自己的 kernel"。L3 的跳跃是：**一次真实请求在 GPU 上端到端流出 token**——GGUF 模型文件 → 加载/解析 → 量化解码 → 计算（GEMM/attention）→ 采样 → 流式输出；验收判据与 llama.cpp 对拍（token 一致性 + 吞吐 ≥3× CPU）。先正确、后性能。

## 范围与验收（逐块的 gate）

| 块 | 功能 | 实施锚 | 验收 gate |
|---|---|---|---|
| D1 GGUF 读取 | header/meta/tensor 表 + mmap 视图 | 001 T2–T3 | golden-file + proptest；元数据→typed 配置（Qwen2 解析） |
| D2 量化 codec | Q8_0（block 256）/F16/FP32 解量化 + proptest | 001 T4 | ≤1 ULP golden（金块 = llama.cpp `llama-quant` 产物） |
| D3 arch 配置 | Llama/Qwen2 元数据 → typed config | 001/plan §Modules | parse 单测（与 llama.cpp metadata 转储对拍） |
| D4 tokenizer | SPM/BPE encode + 增量 decode（GGUF tokenizer.model） | 004 | golden vs llama.cpp tokens（20 prompts 100%） |
| D5 dequant 内核 | Q8_0 解量化核（并行化）+ F16 视图 | 003 T8 | 与 CPU 参考差分 ≤1 ulp（金块） |
| D6 GEMM | cuBLAS wrapper（f16 16F-acc，默认对 llama.cpp） | 003 T9 | vs CPU matmul：f16 rel 1e-4 / f32 rel 1e-5（100 形状）+ perf sanity |
| D7 attention | Prefill：两段 GEMM（QK^T+softmax+PV，NHD）；Decode：paged GQA（块 16/32，smem staging）+ MemOps | 003 T10/T11 | seq 1k fp16 diff（≤1 ulp vs 参考声明）；随机页表 batch 1..64 diff；泄漏运行（1M 页 alloc/free 后"在用==0"） |
| D8 闭环 | 最小 Runner（bin/cli 模块）+ `reinfer cli --backend cuda --model <gguf> "prompt"` 流式 + sampler（复用 012 管线）| 003 T12 | parity.md 三层（F16 三层 / Q8_0 ≥99.9%）+ decode ≥3× llama.cpp CPU（008 D5 协议）+ 确定性（固定 seed 同输出） |

模型：**Qwen2.5-0.5B-Instruct（调试）+ Qwen2.5-1.5B-Instruct（验收）**，GGUF（fp16/q8_0），从 ModelScope 经 `reinfer model get` 获取（013）。

## Success Metrics

- **第一次流式 token**：`reinfer cli` 200 token 稳出（无 NaN/锯；seed 固定复现）
- **数值对拍**：parity 三层门禁全绿（F16 三层 / Q8_0 ≥99.9%）；D7 参考差分按各自判据
- **性能判据**：decode ≥3× llama.cpp CPU（同机、同参数、008 D5 协议、notes 留痕）
- **复现包**：模型 sha256 + CLI 参数 + seed 记录（换机可复制）

## User Stories

1. 作为作者：断点"能编译内核"→"能跑通一个小模型"——真机证据是唯一可信。
2. 作为验证者：与 llama.cpp 同机对拍 token 与吞吐——数值/速度两判据可判定。
3. 作为维护者：单小提交推进（每块带 gate）；失败定位到具体块。
4. 作为未来服务化读者：L3 的单请求路径即 005（serving）的解码内核（decode attention + sampler 管线）的真实载体。

## Acceptance Criteria

- [ ] D1–D4：纯 Rust 数据管道全绿（golden + proptest + token 对拍 100%）
- [ ] D5–D7：真机差分全绿（各块 gate 判据；`--test-threads=1`；确定性 bit-exact 范围沿用 012 声明）
- [ ] D8：`reinfer cli --backend cuda --model <通过 013 获取的 gguf> "prompt"` 流式输出；parity 三层 + 3× CPU 判据通过；notes 记录（模型 sha/命令/结果/硬件四元组）
- [ ] 模型标识零硬编码（013 铁律在 L3 全链路生效：cli 参数/调用方输入；测试用 0.5B 文件路径由 env 提供，不写死在单测内）
- [ ] feature-list/phase-plan 勾选 L3；changelog 归档

## Non-Goals

- 006：CUDA graph 桶化/stream overlap/FA3 vendor cubin（性能编排；L3 只做"正确优先"）
- 005：serving 面（批处理/调度/HTTP）、多请求并发、engine crate 正式建立（A-M4：归 005；L3 用 bin 最小 Runner）
- Radix 缓存/投机解码/grammar 集成（P3）
- 昇腾对应路径（AscendC 后端在 002 复活；共享层 jit/models 无 CUDA 知识已就位）
- prewarm（仍延至 005 引擎首启）
- 跨多卡/TP/PP、MoE/MLA

## Constraints

- **先正确后性能**、fail-closed 不变（LaunchError 三分类；数据管道纯 Rust `forbid(unsafe_code)`）
- 模型经 013 resolver；**标识零硬编码**（代码路径无模型常量；D8 测试的模型参数经 env/CLI 注入）
- llama.cpp 对拍：同机、同量化、同参数（top_p/temperature=0 档 或固定 seed）；基准协议 = 008 D5 `gate_throughput.sh`（中位数、warmup）
- 真机纪律沿用：`--test-threads=1`、差分容差表（003 D7）、确定性声明范围、notes 留痕
- 提交：英文 Conventional Commits 小提交；无 AI trailer（宪法）

## Changelog

- （评审后回填）
