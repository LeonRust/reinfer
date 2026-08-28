# Tasks: CUDA L3 — single-request full loop

> Derived from specs/014-cuda-l3-single-request/plan.md · r2（2026-08-28 四代理评审修订）
> 外部依赖（非本切片）：**013**（模型获取 0.5B/1.5B q8_0 存档——sha 锚已钉；未达→env 路径直传兜底）、**004**（交付面=scaffold/BPE encode/decode ✅；**SPM encode 未交付**——已核实 `SpmEncodeUnimplemented`，等 004 M4 golden；本切片 T4 仅 BPE 档 gate，SPM 显式 skip）、**T0**（referee 构建——本切片首任务，全部金块/对拍流依赖）。
> 提交类型映射：T0→`build(referee)`；T1-T9→`feat(...)`；T10→`ci:`+`perf:`；T11→`docs:`。模型标识零硬编码（013 铁律；T11 grep 命令为 gate）。

## T0: llama.cpp referee 构建（f280b2698，CPU 档）

- 构建：`cmake -B build -DCMAKE_BUILD_TYPE=Release --target llama-cli llama-bench llama-tokenize llama-quantize llama-gguf`（**CPU 档；r2 修正**——D7 原 CUDA 构建在对拍中不使用且 ARCH=120 在 nvcc 12.6 判定机不可构建）；产物路径记录（`bench/notes.md` 或 runner-info.json）
- Verification: 4 工具存在可跑（`llama-tokenize --version` + `llama-bench --help` exit 0）；commit 钉 f280b2698（`git rev-parse HEAD` 记录）

## T1: GGUF 读取器（承接 001 Task2-3 + 014 增量）

- **承接**：001 已交付 `GgufReader`（pread views/元数据/tensor 表）；**014 增量**：Qwen 元数据全键探针（`qwen2.rope.*` nested arrays 等）、QK8_0/被忽略键行为
- **真模型存档测试**（r2：**golden 角色**）：经 013 获取 `qwen2.5-0.5b-instruct-q8_0.gguf`（路径经 env `REINFER_MODEL_DIR` 注入）→ reader+arch 全链 + 与 llama.cpp `gguf --dump` 逐键/类型比对；**独立脚本（非 `#[ignore]` 测试，免 allowlist——r2 明示）**；note：真实 tiny GGUF（`tests/data/tiny.gguf`）无生成器（convert 需真实模型为输入），不再作为独立 gate——自生成 fixture 仅作 proptest（r2：消除悬空实体）
- Verification: 自生成 fixture proptest + 真模型存档逐键比对（与 llama.cpp `gguf --dump`）；档案路径 env 注入

## T2: 量化 codec（Q8_0 **QK8_0=32** / F16 / FP32）

- 承接 001 codec（单乘语义已落地——**kernels 不另建 dequant_ref，本切片判据直接消费 codes::dequantize_q8_0**，r2 单源化）；增量：判据路径（`y=f32(q)*f32(f16(d))` 单乘语义 + **f32→f16 转换 RNE 单次舍入**——r2 钉死）
- 金块 = `llama-quantize`（f280b2698 = **T0** + `--tensor-type ".*=q8_0"` + `--leave-output-tensor`）→ **`tests/golden/q8_0_*.json`**；生成器 `scripts/golden/gen_q8_0_golden.sh`（已在仓，skeleton 实测化；数据落盘 `tests/golden/`）——**依赖显式行：T2 ← T0 + 013 0.5B 存档**；存档未达 → env 直传兜底；金块未达 → 该 gate 显式跳过 + 仅 proptest 有界性+notes 记录（r2 兜底条款）
- Verification: 金块**位精确（0 ulp）**（块级）；**序列级：整 tensor 解码 vs llama.cpp 张量输出对拍（复用 T1 存档；r2 加固——块级金块测不到 block 跨页/boundary 错误）**；proptest 仅 quant∘dequant 有界性

## T3: arch typed config（Llama/Qwen2）

- 承接 001 + 增量（Qwen 键集）；真模型存档对拍（T1 存档复用）
- Verification: parse 单测（**≥5 具名用例**：缺 vocab/缺 head_dim/非法 rope_theta/非法 kv heads/未知 arch）+ 存档对拍

## T4: tokenizer（004 判据原文；BPE 主档）

- `crates/tokenizer`（已建）：GGUF tokenizer 解析 + BPE（qwen2 pre 特例、byte-level UTF-8 切断 read/surr 语义；encode 已实现 bpe.rs）+ SPM **容器读入**（byte-fallback `<0xXX>`；**SPM encode 未交付（004）→ 显式 skip 注记**）
- **BPE 金块首建（r2 显式任务）**：llama.cpp `llama-tokenize --ids`（T0）f280b2698 → `tests/golden/tokenizer-*.json`；生成/重建脚本 `bench/golden/gen_tokens.sh`（**路径以 parity.md 为准**；prompts 文件 `bench/prompts/`）——未达 → 与 SPM 同款「显式 skip + 过渡声明」（r2 补 BPE 档兜底——原只有 SPM 档有）
- Verification: **004 判据**：encode ids & piece texts 100%（金块）+ 增量 decode 分块自洽（`decode_all==decode_one_by_one`）+ 边界 golden（中文/emoji/换行/**token 边界切断多字节字符**）

## T5: dequant 内核（003 T8 + r1 判据）

- QK8_0=32 并行核；内核约束：直读存储 fp16 scale、符号扩展、**禁 `q*d+0.0f` FMA 化写法**；JitCache 管线（012）；输出经 `__float2half`（**RNE**——r2）
- Verification: 真机 diff（随机块）**位精确（0 ulp）**（与 001 codes::dequantize_q8_0）；确定性；`#[ignore]` + allowlist `l3-kernels`；命令模板按 012 T8 先例（`CUDA_VISIBLE_DEVICES=0 cargo test -p reinfer-cuda --features cuda -- --ignored --test-threads=1`）

## T6: GEMM（003 T9；r1 判据化；Vendor 首件）

- workspace `cudarc` 增 `cublas` feature；`gemm_f32acc`（diff/门禁档）与 **`gemm_f16_16acc`（sys::cublasGemmEx 直调：CUBLAS_COMPUTE_16F + half 标量 + 009 流句柄（禁 default stream 0））**；**新增 `kernels::matmul_ref`（fp32 累加 CPU naive——r2：matmul_ref 唯一源，T7 共用）**
- Verification（判据矩阵）：f16-in/f32-out **rtol 1e-4 + atol 1e-6**（r2：K=896/1536、K∈1..4096）/ f32-in/f32-out **rtol 1e-4 + atol 1e-6**（r2 修正：rel-only 1e-5 在近零元素爆表、K=1536 顺序归约差与容差同阶）；f16-out 档双侧 fp16 舍入 ≤1 ulp；**16F-acc 档=记录项**（rel ≤1e-1 声明 + parity 兜底）；perf sanity（CPU 时间/GEMM 时间入 notes）；runner-info `cublas` 回填

## T7: Prefill attention（003 T10 + r1 判据）

- 新增 `prefill_attn_ref`（crates/kernels：012 ref 语义 + fp32 累加 matmul）；`prefill_attention`：两段 GEMM（**32F**）+ **fp32 中间 buffer** + fp32 softmax → fp16 输出（RNE）
- Verification: seq 1k fp16 差分（**|out|≥2^-14 元素 ≤1 ulp；近零元素 atol 1e-6**——r2 条款，防 seed 敏感翻转，与 012 掩码位跳过规则同风格）；全行 sum≈1/掩码行 0；16F+fp16 中间档=记录项（notes）；**r2 反例用例：unmasked 位置注入 NaN——输出必须与 ref 逐位一致（NaN 传播），仅全 masked 行允许输出 0**；`#[ignore]`+`l3-attn` allowlist；命令模板 T5 同款

## T8: Paged decode attention GQA + MemOps（003 T11 + r1 强化）

- `crates/memory`：页池（16/32，**LIFO 头插**、refcount、守恒断言；**`PageTable` 类型与 fixture 公开工具落本 crate——r2：015 跨后端复用的前提**）；`crates/cuda`：decode_step_gqa（**kv_head=q_head/kv_ratio 整数除法连续分组**；非整除三例 14/2、12/2、5/2 核验）+ 页数据设备内存由 crates/cuda 分配（r2 归属注记）
- Verification: 随机页表 diff（跨 2-3 页/首尾部分页/乱序物理页/batch 1..64/kv_len 1..1k；**物理页快照 fixture**，固定 seed）+ **毒化测试（0xFF/NaN 未初始化页；mask 位置输出恰为 0；r2 补 unmasked NaN 注入——仅全 masked 行允许 0）**；**泄漏三合一**（在用==0 + 空闲==预热 + 守恒式/分配对称/pool 不变）；decode_step 确定性（无 atomicAdd/固定归约树/每 (batch,head) 一 CTA）；`#[ignore]`+`l3-attn` allowlist

## T9: 最小 Runner + run 闭环（003 T12；r2：trait 单点 + 生成语义）

- bin：`run <model> [--backend cuda] "[prompt...]"`（**r2 命令面=契约 v2.15**：`cli` 子命令与 `--model` 旗撤销；`-n/-t/--seed` 契约表）；**Backend trait 首次确立（r2 单点契约：load_weights/prefill(ids)/decode_step()/logits()——015/007 只实现不定义）**；加载（T1/T3）→ prefill（T7）→ decode 循环（T5/T6/T8 + 012 sampler；**层循环含 SiLU（gate 激活）与 embedding 查表——r2 归属明示：本任务内的 host 元素算子**；GEMM 后流同步点）
- **生成语义（r2 必备块）**：EOS 命中即停（模型 self-eos id）· `-n` 硬限（缺省=模型上下文上限）· logits 全 NaN → `LaunchError` 显式错误（不许 argmax 任意走号）· embedding 越界 id → 错误 · `-t 0` 短路 argmax（tie-break 首个最大；**`-t 0` 与 `-t 1e-9` 边界单测——r2**）· seed 注入点=Runner 构造一次 SplitMix64（temp=0 不消费；temp>0 顺序流=012 语义——005 差异化随其立项记录）
- Verification: 0.5B F16/Q8_0 各 200 token 稳出（无 NaN；temp=0 固定 seed 复现）；EOS 单测（构造 EOS 序列）与 `-n` 硬限断言；`#[ignore]`/脚本（allowlist `l3-e2e`）；**014 自身回归（r2：015/007 改造后跑）**

## T10: gate_throughput.sh 创建 + parity 四层 + 3×（r1 落地；r2 加固）

- **创建**（非扩展）：`scripts/ci/gate_throughput.sh`（**r2：backend/阈值参数化——007/015 只传参不改造**；llama-bench `-ngl 0 -t <ncores> -b 1 -n 512 -r 5` ×3 取中位数 + reinfer `run <model> --backend cuda --seed 0 -n 512` 计时 gen tok/s + loadavg<1 采信 + ≥3× 断言）；`bench/prompts/` + `bench/golden/gen_tokens.sh`（**路径以 parity.md 为准**）
- **parity harness 落点（r2 显式实体）**：`tests/parity.rs`（`#[ignore]`；allowlist `l3-parity`）；**硬断言 = logits 全量 finite（先于任何比对——防「全 NaN 双方 argmax 一致」假通过）**；drift 检查与 token 比对覆盖同一数据全量：argmax 逐 token 比对 + 抽样 drift≤1e-4（F16 logits 全量比对=一次性人工，notes）
- **008 接线（r2 显式任务）**：008 plan D5 新增 `l3-kernels/l3-attn/l3-e2e/l3-parity` 行（命令模板照 012 T8 先例）+ `gpu.yml` job 定义（008 T2 交付核对——无 job 则接线承诺悬空，`checked-ignores.sh` 不校验 job 名）
- Verification: **3× 硬闸=1.5B Q8_0**（0.5B/F16 仅记录）；parity 四层（tokenizer 100% / F16 100% / Q8_0 ≥99.9% / logits drift ≤1e-2 记录）+ **Q8_0 序列级用例（T2）**；runner-info（四元组+CPU 身份+cuBLAS）更新；**T0 referee 先于本条**

## T11: 文档与状态 + 零模型名 gate

- feature-list/phase-plan 勾选 L3；notes 追加节（模型 sha/参数/seed/sampler 锚点/硬件）；README 模型段；partity.md 路径互引（`scripts/golden/` + `bench/golden/` 各归其主——r2 唯一化）
- Verification: **零模型名 grep（r2 去扩展名——`.gguf` 特判与「文件格式无关」铁律在审查工具层冲突）**：`rg -n '"(qwen|llama|mistral|vicuna)' crates bin --glob '*.rs'` 必须零命中（例外仅注释/doc 与 judge 引用语义）；>=1 条 grep 非契约文件（docs）允许

---

Completion gate: T0–T11 accepted；数据管道全绿 + 真机 diff（r1/r2 判据）+ run 流式 + parity 四层 + 3×（1.5B Q8_0）记录；评审通过；008 接线无悬挂。

依赖表（r2 修订）：**T0（referee 构建，全局前置）**；T1←013 存档；T2←**T0**+T1+013 存档；T3←T1；T4←**T0**+004 交付面（BPE/decode）+parity.md prompts；T5←T2；T6←T0（matmul_ref 判据用）+012 管线；T7←T3,T6；T8←T6；T9←T1,T3,T4,T5,T6,T7,T8（**含 T4/T1 直连**：tokenizer encode 与 reader 加载；**trait 单点契约**）；T10←T9+**T0**+013/004 外部交付；T11←T10。并行点：T2/T3/T4 可并行（文件独立）；T0 在一切之前（外部构建时间）。
