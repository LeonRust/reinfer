# Plan: CUDA L3 — single-request full loop

> Derived from specs/014-cuda-l3-single-request/spec.md · r1（四代理评审修订）

## Architecture Decisions

- **D1 归属与顺序**：数据管道 `crates/gguf`（reader+codes，**001 已交付/承接面见差异注记**）、`crates/arch`、`crates/tokenizer`（**新建 crate**，004）、`crates/models`（013 resolver，L3 消费）；数值核 `crates/cuda` + `crates/kernels`（参考/选择链）；**cuBLAS 挂 `ProviderTier::Vendor`**（workspace F10 注释"L2/L3 再开 cublas feature"落点）；最小 Runner 在 **bin/cli**（engine crate 正式建归 005/A-M4）；MemOps/页池 `crates/memory`（003 T11）。
- **D2 数值一致性链**（r1 判据化）：CPU 参考（新增 `dequant_ref`（单乘语义）、`matmul_ref`（fp32 累加）、`prefill_attn_ref`（012 ref 语义 + fp32 matmul））；差分判据矩阵：**Q8_0→位精确**；GEMM→32F 档门禁 + 16F 档记录；attention→32F+fp32 中间 → ≤1 ulp（fp16 舍入）；确定性声明沿用 012（同机同产物同 launch 配置）。
- **D3 GQA/分页/泄漏**：GQA 映射 **`kv_head = q_head / kv_ratio`**（整数除法，连续分组，与 HF/llama.cpp 一致；非整除取整方向实现注释固定，14/2、12/2、5/2 三例核验）；页块 16/32；**未初始化页毒化测试**（0xFF/NaN 填充；被 mask 位置输出必须恰为 0）；页表 fixture 覆盖（跨 2-3 页/首尾部分页/乱序物理页/batch 1..64/kv_len 1..1k）+ **分配器物理页快照**入 fixture；free-list **LIFO 头插**（vLLM 语义）；泄漏断言**三合一**：在用==0 + 空闲==预热长度 + 守恒式（在用+空闲==预热；总分配==释放；pool 大小不变）；decode_step 确定性：无 atomicAdd、固定归约树、每 (batch,head) 一 CTA。
- **D4 对齐 llama.cpp 的实现细节**：Q8_0=QK8_0=32（34B/block）；F16 权重 mmap 直读直拷（**无 fp32 权重 staging**）；Q8_0 **按层一次性解量化到 fp16 device buffer**（decode 循环复用；与 llama.cpp convert 路径差异记 notes）；GEMM 16F-acc 直调 `cublas::sys::cublasGemmEx`（**cudarc safe 层硬编码 CUBLAS_COMPUTE_32F——必须绕过**；compute type=16F；alpha/beta half 标量；用 009 `CudaStream` 句柄、**禁 default stream 0**；GEMM 后事件/流同步点写入 T9 契约）；tokenizer 增量 decode = 004 判据（自洽；不对拍流式 detokenizer）。
- **D5 基准协议（落地版）**：T10 **创建** `scripts/ci/gate_throughput.sh`（llama-bench `-ngl 0 -t 24 -b 1 -n 512 -r 5` ×3 独立调用取中位数 + reinfer 侧 `cli --seed 0 --n 512` 计时 gen tok/s + **loadavg <1 才采信** + 两侧同机同参同量化 KV=f16 + 断言 ≥3×）；runner-info.json 扩展 `cpu:{model,ncores,threads}` + T6 回填 `cublas` 版本；**3× 硬闸仅 1.5B Q8_0**（0.5B CPU 太快噪声翻转；F16 CPU 太慢失真——notes 记录理由）。
- **D6 验收递减**：① 数据管道 CPU-only → ② dequant/GEMM（32F 门禁档）真机 → ③ attention 真机 → ④ 装配（先 F16 后 Q8_0）→ ⑤ parity/3×（1.5B）。
- **D7 Judge/Referee 构建**：llama.cpp pinned f280b2698 CUDA 构建（`-DLLAMA_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release --target llama-cli llama-bench llama-tokenize`）——**T10 前置任务**（现判定机有源码无构建产物）。

## Module Breakdown

| 模块 | 内容 |
|---|---|
| `crates/gguf/src/{reader,schema,codes}.rs` | **承接 001 交付** + 014 增量（Qwen 元数据探针/真模型存档测试）；memmap2/byteorder/bytemuck（001 原案显式化） |
| `crates/arch/src/llama.rs` | typed config（Qwen2/Llama） |
| `crates/tokenizer/src/{spm,bpe}.rs` | 004 判据实现（BPE 主档 + SPM 容器读入（判据依赖 004 M4 金块）） |
| `crates/cuda/src/{dequant,gemm,attention}.rs` | dequant 核（QK8_0）、`gemm_f16_16acc`（sys 直调 16F）、`gemm_f32acc`（数值门禁档）、prefill/`decode_step_gqa` |
| `crates/memory/src/pool.rs` | 页块池（LIFO/refcount/守恒断言/毒化测试） |
| `bin/reinfer/src/cli.rs` | Runner 最小组装（seed 注入点=构造一次 SplitMix64，temp=0 不消费 RNG） |
| `scripts/ci/gate_throughput.sh` + `bench/prompts/` + `scripts/golden/` | **T10 创建**（D5 落地）+ gen_tokens.sh + goldens（004 依赖） |

## Interface Contracts（关键签名，r1 修订）

```rust
// crates/gguf（承接 001：假设 GgufReader/dequantize_q8_0 已存在；本切片仅增量）
pub fn dequantize_q8_0(blob: &[u8], out: &mut [f32]) -> Result<(), ...>;   // QK8_0=32，单乘
// crates/cuda（新）
pub struct Gemm;                                               // 三层明示：
pub fn gemm_f32acc(a: &[f16], ...) -> Result<...>;             //   CUBLAS_COMPUTE_32F（diff/门禁档）
pub fn gemm_f16_16acc(a: &[f16], ...) -> Result<...>;          //   16F（parity 记录档；sys 直调）
pub fn prefill_attention(...) ;                                //   32F GEMM + fp32 中间（判据档）
pub fn decode_step_gqa(...) ;                                  //   smem staged + GQA 映射（D3）
// bin cli
pub fn run_cli(backend, model_path, prompt, seed) -> ...;      // 模型路径=CLI/env（零硬编码）
```

## 差异注记（001/003/004 原文 vs 本切片，r1）

| 项 | 原文 | 本切片 | 原因 |
|---|---|---|---|
| Q8_0 块宽 | 001/003 "block 256" | **QK8_0 = 32**（34 B/block；256=QK_K 为 K-quant） | 事实修正（ggml-common.h）；K-quant 与 Q8_0 不混 |
| Q8_0 判据 | "≤1 ULP" | **位精确（0 ulp）** | 单乘语义可达成；1 ulp 表述过弱 |
| GEMM 判据 | 003 D7：f16 rel 1e-4（未分 compute type） | diff 门禁=**32F-acc**（1e-4/1e-5）；**16F-acc=记录项**（rel≤1e-1） | 实测 16F-acc vs fp32 参考在真实 K 92–98% 失败；parity 兜底——003 D7 表同步回写 |
| attention 参考 | "flash-attn 声明 ±1ulp" | **CPU naive（prefill_attn_ref）**；32F+fp32 中间→≤1 ulp | flash-attn 依赖 torch（宪法禁）；16F+fp16 中间误差量级差 1-2 个数量级 |
| tokenizer 对拍 | "与 llama.cpp 逐条对拍" | **004 原文判据**（encode 100% + 分块自洽；不对拍流式 detokenizer） | 004 判据为自洽性；流式逐 token 对齐是更强且不可承诺判据 |
| 3× 判据档 | "decode ≥3×"（无档位） | **仅 1.5B Q8_0 硬闸**；0.5B/F16 记录 | 0.5B CPU ~100-250 tok/s 噪声翻转；F16 CPU 慢到无意义 |
| gate_throughput | 008 纸面声明 | **014 T10 创建**（含测量协议项） | 脚本/bench assets 从未落地 |
| cuBLAS compute | "cuBLAS F16 默认" | **显式 CUBLAS_COMPUTE_16F（sys 直调）** | cudarc safe 层硬编码 32F——默认表述为事实错误 |
| parity 层数 | "F16 三层 / Q8_0 ≥99.9%" | **四层显式枚举**（tokenizer/F16/Q8_0/logits 记录） | parity.md 未约定"三层"简称 |
| 模型铁律 | "一律 ModelScope" | **经 013 resolver（ModelScope 优先，auto 回退 HF）** | 013 r2 修订 |
| mmap | 001 原案 memmap2 | **pread（FileExt::read_at）替代** | 实现决策（2026-08-27 T1 交付）：memmap2 map 0.7+ 为 unsafe 且并发截断即 UB（与 forbid(unsafe_code) 冲突）；OS 页缓存语义等价；非 unix 退路=seek+read；正式 golden 为真实 tiny GGUF（001 原案，随 001/013 交付补水） |

## Risk Assessment

| Risk | Mitigation |
|---|---|
| 16F-acc 数值漂移被误当 bug | D6 判据档位化（记录不代表错误）；parity 兜底；notes 记录 |
| llama.cpp 对拍输入变量（模板/BOS/special） | 输入=golden ids；标志固定；模板差异归 tokenizer 层 |
| CPU 基准抢占（桌面会话/并行编译） | loadavg <1 采信 + 3 次中位数 + 同脚本进程内顺序执行 |
| F16 token 100% 因 attention 算法差异（两段 GEMM vs FA）未达 | 回退档（accum drift ≤1e-4）+ notes 记录实际一致率 |
| 页表/分配器耦合输入漂移 | 物理页快照入 fixture；固定 seed |
| 013/004 交付不足 | 外部依赖行（T9←013；T4←004/顺带 golden）；plan M1/M4 标注；env 直传路径兜底 |
| CUDA 头文件/workspace 占用（cuBLAS workspace） | 1.5B 最重档 <5GB/24GB（资源评审）；单请求 200 token KV 极小 |

## 里程碑（r1）

- M0：001/004/013 交付状态核对（外部依赖行）
- M1（D1-D4）：数据管道（CPU-only，golden+真模型存档+004 判据）
- M2（D5-D6）：dequant 位精确 + GEMM 32F 门禁/16F 记录（**judge 构建随 T10；但 llama.cpp referee 构建可作为 M2 起并行任务**）
- M3（D7）：attention 真机（32F 判据 + 泄漏三合一 + 毒化）
- M4（D8）：cli 流式（F16→Q8_0）+ parity 四层
- M5（T10）：gate_throughput.sh 创建 + 3×（1.5B）判据 + notes
- M6：文档/feature-list/phase-plan
