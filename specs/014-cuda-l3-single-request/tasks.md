# Tasks: CUDA L3 — single-request full loop

> Derived from specs/014-cuda-l3-single-request/plan.md · r1 修订
> 外部依赖（非本切片）：**013**（T9←013 模型获取；未达→env 路径直传兜底）、**004**（T4←004 golden/判据；未达→014 顺带生成 golden 并声明来源）。
> 提交类型映射：T1-T9→`feat(...)`；T10→`ci:`+`perf:`；T11→`docs:`。模型标识零硬编码（013 铁律；T11 grep 命令为 gate）。

## T1: GGUF 读取器（承接 001 Task2-3 + 014 增量）

- **承接**：001 已交付 `GgufReader`（mmap/元数据/tensor 表，若未达 gate 则本任务阻塞并向 001 反馈）；**014 增量**：Qwen 元数据全键探针（`qwen2.rope.*` nested arrays 等）、QK8_0/被忽略键行为
- **真模型存档测试**：经 013 获取 `qwen2.5-0.5b-instruct-q8_0.gguf`（路径经 env `REINFER_MODEL_DIR` 注入）→ reader+arch 全链 + 与 llama.cpp `gguf --dump` 逐键/类型比对；`#[ignore]`-风格独立脚本（note：非 PR 主档）
- Verification: golden（真实 tiny GGUF `tests/data/`，001 原案——自生成仅作 proptest）+ proptest + 真模型存档（env 注入）

## T2: 量化 codec（Q8_0 **QK8_0=32** / F16 / FP32）

- 承接 001 codec（若未达 → 阻塞）；增量：位精确判据路径（`y=f32(q)*f32(f16(d))` 单乘语义）；金块 = `llama-quantize`（pinned f280b2698 + `--tensor-type ".*=q8_0"` + `--leave-output-tensor`）→ `tests/golden/q8_0_*.json` + 生成脚本入仓
- Verification: 金块**位精确（0 ulp）**；proptest 仅 quant∘dequant 有界性（与金块门禁分离）

## T3: arch typed config（Llama/Qwen2）

- 承接 001 + 增量（Qwen 键集）；真模型存档对拍（T1 存档复用）
- Verification: parse 单测（**≥5 具名用例**：缺 vocab/缺 head_dim/非法 rope_theta/非法 kv heads/未知 arch）+ 存档对拍

## T4: tokenizer（004 判据原文；BPE 主档）

- `crates/tokenizer`（**新建 crate**；workspace + bin 注册）：GGUF tokenizer 解析（键名 **`tokenizer.ggml.model`**）+ BPE（qwen2 pre 空格特例、byte-level UTF-8 切断 read/surr 语义）+ SPM 容器读入（byte-fallback `<0xXX>`）
- Verification: **004 判据**：encode ids & piece texts 100%（`llama-tokenize --ids` 金块）+ 增量 decode 分块自洽（`decode_all==decode_one_by_one`）+ 边界 golden（中文/emoji/换行/**token 边界切断多字节字符**）；SPM 档依赖 004 M4 金块（未达→显式 skip 注记）

## T5: dequant 内核（003 T8 + r1 判据）

- QK8_0=32 并行核；内核约束：直读存储 fp16 scale、符号扩展、**禁 `q*d+0.0f` FMA 化写法**；JitCache 管线（012）
- Verification: 真机 diff（随机块）**位精确（0 ulp）**；确定性；`#[ignore]` + allowlist `l3-kernels` 行

## T6: GEMM（003 T9；r1 判据化；Vendor 首件）

- workspace `cudarc` 增 `cublas` feature；`gemm_f32acc`（diff/门禁档）与 **`gemm_f16_16acc`（sys::cublasGemmEx 直调：CUBLAS_COMPUTE_16F + half 标量 + 009 流句柄（禁 default stream 0））**
- Verification（判据矩阵）：f16-in/f32-out rel 1e-4（atol 1e-6，K=896/1536、K∈1..4096）/ f32-in/f32-out rel 1e-5；f16-out 档双侧 fp16 舍入 ≤1 ulp；**16F-acc 档=记录项**（rel ≤1e-1 声明 + parity 兜底）；perf sanity（CPU 时间/GEMM 时间入 notes）；runner-info `cublas` 回填

## T7: Prefill attention（003 T10 + r1 判据）

- `prefill_attn_ref`（crates/kernels：012 ref 语义 + fp32 累加 matmul）；`prefill_attention`：两段 GEMM（**32F**）+ **fp32 中间 buffer** + fp32 softmax → fp16 输出
- Verification: seq 1k fp16 差分（输出舍入 fp16 ≤1 ulp，与 CPU naive）；全行 sum≈1/掩码行 0；16F+fp16 中间档=记录项（notes）；`#[ignore]`+`l3-attn` allowlist

## T8: Paged decode attention GQA + MemOps（003 T11 + r1 强化）

- `crates/memory`：页池（16/32，**LIFO 头插**、refcount、守恒断言）；`crates/cuda`：decode_step_gqa（**kv_head=q_head/kv_ratio 整数除法连续分组**；非整除三例 14/2、12/2、5/2 核验）+ MemOps
- Verification: 随机页表 diff（跨 2-3 页/首尾部分页/乱序物理页/batch 1..64/kv_len 1..1k；**物理页快照 fixture**，固定 seed）；**毒化测试**（0xFF/NaN 未初始化页；mask 位置输出恰为 0）；**泄漏三合一**（在用==0 + 空闲==预热 + 守恒式/分配对称/pool 不变）；decode_step 确定性（无 atomicAdd/固定归约树/每 (batch,head) 一 CTA）；`#[ignore]`+`l3-attn` allowlist

## T9: 最小 Runner + cli 流式闭环（003 T12）

- bin/cli：`reinfer cli --backend cuda --model <path> "<prompt>" [--seed]`；加载（T1/T3）→ prefill（T7）→ decode 循环（T5/T6/T8 + 012 sampler：**temp=0=argmax（tie-break 首个最大）；种子注入点=Runner 构造一次 SplitMix64 顺序消费**；GEMM 后流同步点）
- Verification: 0.5B F16/Q8_0 各 200 token 稳出（无 NaN；temp=0 复现）；`#[ignore]`/脚本（allowlist `l3-e2e`）或显式手工命令

## T10: judge 构建 + gate_throughput.sh 创建 + parity 四层 + 3×（r1 落地）

- **创建**（非扩展）：`scripts/ci/gate_throughput.sh`（llama-bench `-ngl 0 -t <ncores> -b 1 -n 512 -r 5` ×3 取中位数 + reinfer `cli --seed 0` gen tok/s + loadavg<1 采信 + ≥3× 断言；参数与 llama.cpp 侧同机同参同量化）；`bench/prompts/` + `gen_tokens.sh` + goldens（004 依赖）；parity harness=**纯 Rust**（argmax 逐 token 比对 + 抽样 drift≤1e-4；F16 logits 全量=一次性人工，notes）
- Verification: **3× 硬闸=1.5B Q8_0**（0.5B/F16 仅记录）；parity 四层（tokenizer 100% / F16 100% / Q8_0 ≥99.9% / logits drift ≤1e-2 记录）；runner-info（四元组+CPU 身份+cuBLAS）更新；llama.cpp judge 构建（f280b2698 + 命令见 plan D7）先于本条

## T11: 文档与状态 + 零模型名 gate

- feature-list/phase-plan 勾选 L3；notes 追加节（或 `bench/notes.md`；模型 sha/参数/seed/sampler 锚点/硬件）；README 模型段
- Verification: **零模型名 grep**：`rg -n '"(qwen|llama|q8_0\\.gguf|f16\\.gguf)' crates bin --glob '*.rs'` 必须零命中（例外仅注释/doc 与 judge 引用语义）；>=1 条 grep 非契约文件（docs）允许

---

Completion gate: T1–T11 accepted；数据管道全绿 + 真机 diff（r1 判据）+ cli 流式 + parity 四层 + 3×（1.5B Q8_0）记录；评审通过；008 接线无悬挂。

依赖表（r1 修订）：T1；T2←T1；T3←T1；T4←T1；T5←T2；T6←（003 参考+012 管线）；T7←T3,T6；T8←T6；T9←T1,T3,T4,T5,T6,T7,T8（**含 T4/T1 直连**：tokenizer encode 与 reader 加载）；T10←T9 + **013/004 外部交付**；T11←T10。并行点：T2/T3/T4 可并行（文件独立）；M2 起 llama.cpp judge 构建可并行。
