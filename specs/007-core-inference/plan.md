# Plan: Core inference — CPU 全链路

> Derived from specs/007-core-inference/spec.md · r2（2026-08-28 四代理评审修订）

## Architecture Decisions

- **D1 模块归属**：执行器 `crates/cpu`（空壳落地）；复用面：001 `codes.rs`（q8_0/f16 解量化——**单源：即 CPU 参考本身，kernels 不另建 dequant_ref——r2**）、`arch/llama.rs`、004 tokenizer（BPE 主档；SPM 不依赖）、012 sampler+`refs.rs`、gguf reader（pread views——`&[u8]` 直用）。
- **D2 数值设计**：全链 **fp32 累加**（matmul naive `acc += a*f32(b)`；权重：Q8_0 dequant→**fp16 stash（f32→f16 RNE——r2 同 014/015 钉死）** → matmul 时升 f32；F16 直读）；GQA 与 014 一致（`kv_head=q_head/kv_ratio` 整除连续分组、**非整除三例 14/2、12/2、5/2 核验——r2 恢复三例**）；KV = 连续矩形 buffer（`[layers][heads][max_seq][head_dim]`；CPU 不做 paged）；RoPE=012 `rope_ref`；softmax=012 `masked_softmax_ref`；RMSNorm=012 `rms_norm_ref`。
- **D3 分层解耦**：`crates/cpu` 只做计算；bin 侧**实现 014 T9 的 Backend trait（单点契约——r2：删除「007 与 A5 一致」自证，007 与 015 只实现不定义）**；`run` 无 backend 分支差异（trait 分派）。
- **D4 对拍协议**：沿用 014（golden ids 注入、temp=0、referee f280b2698 = **014 T0**——r2 明确前置）；20 prompts（中文/英文/多行/stop hits）— `bench/prompts/`（**014 T10 同址分享**）。
- **D5 性能记录**：**记录不设档（r2 定案；spec C4 带宽分析）**；采集：`gate_throughput.sh`（**014 T10 参数化产物**）`--backend cpu` 传参；loadavg<1；notes 四元组。

## Module Breakdown

| 模块 | 内容 |
|---|---|
| `crates/cpu/src/{model,ops}.rs` | `Model`（layers + 权重视图：Q8_0 经 001 dequant 按需解量化——layer 计算时机，内存 <1GB 目标 0.5B）；`linear`（fp32 累加）、`attention`（GQA）、`mlp`（silu/gelu gate 按 arch）、`embed`（OOV 防呆——r2） |
| `bin/reinfer/src/backend_cpu.rs` | 014 T9 Backend trait 的 CPU 实现（C2 `run` 接入） |
| `bench/prompts/` + harness | 014 T10 产物（`bench/prompts/`、`tests/parity.rs`）传参指向 `--backend cpu` |

## Interface Contracts（r2：与 014 T9 单点契约对齐）

```rust
// crates/cpu（计算层内部形态——非 trait）
pub struct Model;
pub fn load(reader: &GgufReader, config: &ArchConfig) -> Result<Model, LaunchError>;
pub fn prefill(&mut self, ids: &[u32]) -> Result<Vec<f32>, LaunchError>;   // 一次性 seq → 最后 logits
pub fn decode_step(&mut self, last: u32) -> Result<Vec<f32>, LaunchError>; // 增量 → 最后 logits
pub fn reset(&mut self);                                                   // KV 清空（chat 复用）
// bin：impl 014 T9 Backend（load_weights/prefill(ids)/decode_step()/logits()）——只实现不定义（r2）
```

## Risk Assessment

| Risk | Mitigation |
|---|---|
| CPU naive 内存/速度 | Q8_0 按需解量化（layer 计算时）；0.5B 权重 <1GB 目标；速度=记录档（spec C4/r2） |
| F16 token 100% 判据 | 双方 fp32 累加同构（llama.cpp CPU=fp16 权重+fp32 acc）；golden 相同输入；运行 logits 差分漂移记录 |
| Q8_0 ≥99.9% 偶发翻转 | 回退档（一致率+drift）+top-2 logit 间距记录；失败样本量统计入 notes |
| 014 产物未就绪（referee/harness/prompts） | **T3/T4 前置声明 014 T10/T0（r2：阻止先引用后创建死锁）** |
| 014 runner 组装点移动（trait） | 014 T9 单点契约；007 只实现；014 回归兜底 |

## 里程碑（r2）

- M0: 依赖核对（001/004(交付面)/012/013 + **014 M4/M5——r2 显式**）
- M1 (C1): layer loop + 0.5B 存档 golden 逐 token 对拍
- M2 (C1/C2): decode_step + run 流式（temp=0 复现；ModelRef 三态）
- M3 (C3): F16 100% + Q8_0 ≥99.9%（20 prompts）
- M4 (C4): 吞吐记录（notes 四元组；无 % 闸）
