# Spec: CUDA L3 — single-request full loop（第一次跑出 token）

> Status: proposal · r2（2026-08-28 四代理评审修订：命令面/金块实体/判据 atol/trait 单点/EOS 防呆）· Owner: maintainers
> Parent: phase-plan §L3（003 T8-T12 + 004）· 实施锚：001 Task2–4 / 004 / 003 T8-T12 / 000 parity.md
> 模型：经 **specs/013 resolver** 获取（**ModelScope 优先，`auto` 可回退 HuggingFace**——013 r2 铁律修订版；代码路径零模型标识，另见 Constraints）
> 前序：009（L1）/ 012（L2 JitCache + 首批内核 + 选择链）

## Problem Statement

L1/L2 证明了"能安全消费运行时"与"能编译并跑我们自己的 kernel"。L3 的跳跃：**一次真实请求在 GPU 上端到端流出 token**——GGUF 模型 → 加载/解析 → 量化解码 → 计算（GEMM/attention）→ 采样 → 流式输出；判据与 llama.cpp 对拍（token 一致性 + decode ≥3× llama.cpp CPU）。先正确、后性能。

## 范围与验收（判据为 r1 修订版）

| 块 | 功能 | 实施锚 | 验收 gate（r1） |
|---|---|---|---|
| D1 GGUF 读取 | header/meta/tensor 表 + pread views | 001 Task2–3 | golden = **013 真模型存档**（0.5B q8_0，env 注入）reader+arch 全链 + llama.cpp `gguf --dump` 逐键/类型比对（r2：真实 tiny GGUF 无生成器——convert 需真实模型为输入，存档对拍承担 golden，`tests/data/tiny.gguf` 不再作为独立 gate）；自生成 fixture 仅作 proptest |
| D2 量化 codec | Q8_0（**block 32 = QK8_0，34 B/block；非 256**）/F16/FP32 | 001 Task4 | 金块（`llama-quantize`，pinned commit f280b2698 + 明确 flags）**位精确（0 ulp：`y=f32(q)*f32(f16(d))` 单乘语义）**；proptest 仅作 quant∘dequant 有界性 |
| D3 arch config | Llama/Qwen2 元数据 → typed config | 001 plan | parse 单测 + 真模型存档对拍 |
| D4 tokenizer | GGUF BPE（qwen2 pre 特例）/SPM 容器 | 004 | **004 原文判据**：encode 逐 token ids & piece texts 100%（`llama-tokenize --ids`）+ 增量 decode 分块自洽；**不与 llama.cpp 流式 detokenizer 逐 token 对拍**；SPM 路径依赖 004 M4 金块（014 内仅 gate BPE） |
| D5 dequant 内核 | Q8_0 并行核（QK8_0=32 语义）+ F16 直读 | 003 T8 | 与 T2 CPU 参考**位精确**；内核约束：直读存储 fp16 scale、禁 FMA 化写法（`q*d+0.0f` 禁写） |
| D6 GEMM | Vendor 档 GEMM：**cuBLAS 封装**（f16 16F-acc，CUBLAS_COMPUTE_16F 显式；llama.cpp `GGML_CUDA_CUBLAS_COMPUTE_TYPE=16F` 对齐）| 003 T9 | 数值门禁用 **CUBLAS_COMPUTE_32F** 档：**rtol 1e-4 + atol 1e-6**（r2：rel-only 在近零元素爆表、K=1536 顺序归约差与容差同阶——双条款同 012 D6）；f16-out 档=双侧 fp16 舍入 ≤1 ulp；**16F-acc 档为记录项**（rel ≤1e-1 声明，parity 兜底）；形状含模型真实 K（896/1536）、K∈1..4096；perf sanity（notes） |
| D7 attention | Prefill：两段 GEMM（NHD，**fp32 中间 buffer**）；Decode：paged GQA（块 16/32，smem staged）+ MemOps | 003 T10/T11 | 参考=**CPU naive**（`prefill_attn_ref`：012 ref 语义 + fp32 累加 matmul）；两段 GEMM 32F + fp32 中间 → 输出舍入 fp16 **|out|≥2^-14 元素 ≤1 ulp（近零按 atol 1e-6，r2 条款）**；16F+fp16 中间为记录项；GQA 映射 pin（见 plan）；随机页表 diff + 毒化测试（**含 unmasked 位置 NaN 注入：仅全 masked 行允许输出 0，r2 反例修正**）；泄漏三合一断言 |
| D8 闭环 | 最小 Runner（bin）+ `reinfer run <model> --backend cuda "<prompt>"` 流式（r2：命令面契约 v2.15——`run <model>` 位置参数 + `--backend`；`cli` 子命令与 `--model` 旗撤销） | 003 T12 | parity 四层（见 success metrics）；**decode ≥3× llama.cpp CPU——硬闸仅 1.5B Q8_0 档**（0.5B/F16 仅记录）；确定性（temp=0 档）；**生成语义必含（r2）：EOS 命中即停（模型 self EOS id）· `-n` 硬限（缺省=模型上下文上限）· logits 全 NaN → 显式错误不继续走号 · embedding 越界 id → LaunchError · `-t 0` 短路 argmax 链（005 D5 语义）** |

模型：Qwen2.5-0.5B-Instruct（调试）+ 1.5B（验收），from 013（q8_0 锚点 675,710,816 B/sha256 已钉；fp16 与 1.5B 的 4 份文件 sha 由 013 M3 端到端时补钉）。

## Success Metrics

- 第一次流式 token：200 token 稳出（无 NaN；temp=0 档 seed 固定复现）
- 数值对拍（**parity 四层，r1 显式枚举**）：① tokenizer 100%（004）；② F16 同 compute type 逐 token 100%（回退档:累积 drift ≤1e-4；notes 记 attention 算法差异实际一致率）；③ Q8_0 greedy ≥99.9%；④ logits 相对漂移 ≤1e-2（记录项）——**r2 加固：harness（落点 `tests/parity.rs`，allowlist `l3-parity`）先做 logits 全量 finite 硬断言（防「全 NaN 双方 argmax 一致」假通过），drift 检查与 token 比对覆盖同一数据全量；Q8_0 档补序列级用例（整 tensor 解码 vs llama.cpp 张量输出；金块单块判据不足以测 block 跨页/boundary）**
- decode ≥3× llama.cpp CPU（**仅 1.5B Q8_0；同机同参，008 D5 协议**，notes 四元组+CPU 身份）
- 复现包：模型 sha（013 manifest）+ 参数 + seed + sampler 语义锚点

## User Stories

1. 作为作者：断点从"能编译内核"到"能跑通小模型"——真机证据唯一可信。
2. 作为验证者：与 llama.cpp 同机对拍 token 与吞吐——两判据可判定。
3. 作为维护者：小提交逐块 gate；判据不可执行时（如 16F-acc diff）降级记录并有 parity 兜底。
4. 作为未来服务化读者：L3 解码路径即 005 的真实载体（decode attention + 采样管线）。

## Acceptance Criteria

- [ ] D1–D4 数据管道全绿（golden（真实 tiny GGUF + 真模型存档）+ proptest + 004 判据）
- [ ] D5–D7 真机差分全绿（r1 判据：位精确 / 32F 门禁 + 16F 记录 / CPU naive 参考）
- [ ] D8：cli 流式 200 token；parity 四层（F16≥1e-4 回退记录 + Q8_0 ≥99.9%）；3× CPU（1.5B Q8_0）通过；notes 记录
- [ ] 008 接线表新增 `l3-*` 行 + allowlist 登记（T5-T9 真机用例）；checked-ignores 通过
- [ ] `scripts/ci/gate_throughput.sh` 由 T10 **创建**（008 纸面协议落地）+ bench/prompts + gen_tokens.sh + goldens
- [ ] 模型标识零硬编码（013 铁律；grep 命令见 tasks T11）
- [ ] feature-list/phase-plan 勾选 L3；changelog 归档

## Non-Goals（r1 保留）

- 006：graph 桶化/stream overlap/FA3 vendor cubin；attention 用 FA 算法对拍（**fa2 依赖 torch——宪法禁 torch；llama.cpp fp16 prefill 走 flash-attn 的事实作为差异记录**，不执行 FA diff）
- 005：serving/批处理/HTTP；engine crate 正式建（A-M4；L3 用 bin 最小 Runner）
- Radix/投机/grammar（P3）；昇腾本体（002）；prewarm（延至 005）；多卡/多用户/MoE
- **HuggingFace 强校验等价物**（无 sha 字段——013 降级链）；私有仓库

## Constraints

- **判据可执行性（r1 铁律）**：凡"参考"必须是**本机可执行**对象（CPU naive/参考函数）；凡"对拍"必须有可执行命令与存在实体（`gate_throughput.sh` 若缺→先建）；凡数值门禁必须是**实测可过**判据（16F-acc 对 fp32 朴素 1e-4 经实测不可行——按 r1 判据分布）。
- **模型标识零硬编码**（013）：repo/file/quant 由 CLI/env 注入；代码无模型常量；测试用虚构数据（golden 真模型存档除外——路径经 env）。
- llama.cpp 对拍：**temperature=0 档**（双方 argmax，tie-break=取首个最大）；temp>0 仅自确定性（RNG 算法不同 mt19937 vs SplitMix64——**同 seed 不可对拍**）；输入隔离=golden token ids（非文本）；`--special`/chatml 模板固定；referee 构建 pinned f280b2698。
- 真机纪律：`--test-threads=1`、`CUDA_VISIBLE_DEVICES=0`、D7 容差、确定性声明范围（同机同产物同 launch 配置 bit-exact）、notes 留痕。
- 资源纪律：权重路径 f16 直读（无 fp32 staging）；计算中间按 D6/D7 判据（32F diff 路径用 fp32 中间）；KV 按 max_seq（4096）预算一次性分配。
