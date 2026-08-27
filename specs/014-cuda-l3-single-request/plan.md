# Plan: CUDA L3 — single-request full loop

> Derived from specs/014-cuda-l3-single-request/spec.md

## Architecture Decisions

- **D1 归属与顺序**（无新 crate 爆炸；沿用既有锚）：
  - 数据管道：`crates/gguf`（读取+codes）、`crates/arch`（typed config）、`crates/tokenizer`（**新建 crate**——004 定义）、`crates/models`（013 resolver，L3 消费）；
  - 数值核：`crates/cuda`（dequant/attention/launch）+ `crates/kernels`（CPU 参考/选择链）；**cuBLAS 挂 `ProviderTier::Vendor`**（`cudarc` 增 `cublas` feature——009 时的"L2/L3 再开"落点）；
  - 最小 Runner：**bin/cli 模块内组装**（engine crate 正式建归 005/A-M4，不提前设计）；
  - MemOps/页池：`crates/memory`（001/T11 的分配器 + refcount），CUDA 侧 MemOps 在 `crates/cuda`。
- **D2 数值一致性链**：每块"CPU 参考 ↔ GPU 实现"差分（012 的 refs 已覆盖 rms/rope/softmax——L3 新参考：dequant、matmul（CPU 朴素）、attention（逐 token naive）、paged 索引）；容差一律 003 D7 表；确定性：同 seed 同输入 → 仅在"同机同产物同 launch 配置"声明 bit-exact（012 R 声明沿用）。
- **D3 GQA/分页**：attention decode 先按"正确语义"实现（每 token 一次 smem staged 累加，paged index 查页）——先正确，FA3/图编排留 006；block 16/32 由 `crates/memory` 分配器参数化；MemOps 记录 alloc/free refcount，泄漏运行断言。
- **D4 对齐 llama.cpp 的实现细节**（判据权威）：Q8_0 内核=block 256（llama.cpp 算法），F16 存储直读；GEMM 16F-acc（`cuBLAS F16` 默认）；tokenizer 增量 decode 语义（SPM/BPE 序列化与 llama.cpp 逐条对拍）；logits 采样链路复用 012 sampler（SplitMix64 锚）。
- **D5 基准协议**：3× llama.cpp CPU（同机同参同量化；warmup 5 轮、中位数 3 轮；`bench/runner-info.json` 硬件四元组）；记入 008 D5 协议（脚本按 003 T12 notes 规则扩展，不新造协议）。
- **D6 验收递减拆解**：① 数据管道（CPU-only，全绿即可开始）→ ② 单 kernel diff（dequant → GEMM → attention）→ ③ 装配解码循环（先 fp16 模型整列正确，再 Q8_0 校验）→ ④ parity/性能判据。

## Module Breakdown

| 模块 | 内容 | 锚 |
|---|---|---|
| `crates/gguf/src/{reader,schema,codes}.rs` | header/meta/tensor 表 + mmap；Q8_0/F16/FP32 codec（+proptest） | 001 T2-T4 |
| `crates/arch/src/llama.rs` | Qwen2/Llama 元数据 → typed config | 001 plan |
| `crates/tokenizer/src/{spm,bpe}.rs` | GGUF tokenizer 模型解析 + encode/decode-step | 004 |
| `crates/cuda/src/{dequant,gemm,attention}.rs` | dequant 核；cuBLAS wrapper；prefill/decode attention | 003 T8-T11 |
| `crates/memory/src/pool.rs` | 页块 alloc/free/refcount（+泄漏断言） | 003 T11 |
| `bin/reinfer/src/cli.rs` | `cli --backend cuda --model <gguf> "<prompt>"` 流式 loop（Runner 最小组装） | 003 T12 |
| `crates/bench*/verify` | parity harness（对拍 llama.cpp）、gate_throughput 扩展 | 000 parity.md / 008 D5 |

## Interface Contracts（关键签名）

```rust
// crates/gguf
pub struct GgufReader<'a> { mmap: Mmap }          // header/meta/tensor 表
impl<'a> GgufReader<'a> { pub fn open(p: &Path) -> Result<Self, ...>; pub fn tensor(&self, name: &str) -> Result<GgufTensor<'a>, ...>; pub fn metadata(&self) -> &ModelMeta; }
pub fn dequantize_q8_0(blob: &[u8], out: &mut [f32]) -> Result<(), ...>;   // CPU 参考 + GPU 差分 anchor
// crates/arch
pub struct LlamaConfig { vocab, n_layer, hidden, q_per_head, kv_per_head, head_dim, rope_theta, ... }
pub fn from_gguf_meta(meta: &ModelMeta) -> Result<LlamaConfig, ...>;
// crates/cuda
pub struct Gemm;   // cuBUBlas f16-acc wrapper；launch 走 vendor tier 语义
pub fn prefill_attention(...); pub fn decode_step_gqa(...);   // 语义见 D2/D3
// bin cli
pub fn run_cli(backend: Backend, model: &Path, prompt: &str, seed: u64) -> Result<(), ...>;  // 流式（stdout 逐 token）
```

## Risk Assessment

| Risk | Mitigation |
|---|---|
| 模型元数据与 llama.cpp 转储差（不同量化的 variants） | typed config 对拍 llama.cpp dump；0.5B 两份（F16/Q8_0）先行 |
| Q8_0 解码语义偏移（块反量化±scale 截断） | CPU 参考 + ≤1 ulp 金块（001 golden）双保险；parity Q8_0≥99.9% 兜底 |
| cuBLAS 与 CUTLASS 别名差异（16F-acc 兼容性） | 参考矩阵 CPU 朴素（fp32 累加）与 16F 差 1e-4 判据先行；对齐 llama.cpp 惯用 gemm 参数 |
| 长 prompt/长 seq 的数值积累漂移 | fp16 预填充逐层 diff（参考 flash-attn 声明 ±1ulp）；记录噪声声明 |
| 判别 3× CPU 判据受硬件差异干扰 | 同机同参基准协议（D5）+ runner-info 四元组 |
| tokenizer 边界（BPE/SPM 增量 decode 尾 token 不同） | 004 golden 20 prompts 100% + 常规边界（中文/emoji/换行）用例 |
| 依赖顺序倒置（数据管道未完成即 GPU 侧开工） | D6 递减拆分：①② 串行依赖明确，数据管道块内部可并行 |

## 里程碑

- M0：本 spec 评审（同 012，4 代理）
- M1（D1-D4）：数据管道全绿（gguf/codes/arch/tokenizer）——**纯 CPU，先行**
- M2（D5-D6）：dequant + GEMM 真机 diff
- M3（D7）：prefill/decode attention 真机 diff + 泄漏
- M4（D8）：cli 流式闭环 + parity + 3× CPU（notes）
- M5：文档/feature-list/phase-plan 更新
