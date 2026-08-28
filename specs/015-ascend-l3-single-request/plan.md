# Plan: Ascend L3 — single-request full loop

> Derived from specs/015-ascend-l3-single-request/spec.md · r2（2026-08-28 四代理评审修订）

## Architecture Decisions

- **A1 算子来源（与 CUDA 路径的根本差异）**：数值计算消费 `cann` safe 层（0002 封装：`Tensor`/`Matmul`/`Softmax`/`RmsNorm`，两段式 `new`(GetWorkspaceSize)+`launch(&Stream)`）；**RoPE/SiLU/embedding = host 侧 per-token（r2 定案；见 A4-L）**；不做 GE 图（`cann::graph` 由 cann-rs 自验——015 非目标，**依赖切割：0002 GE 部分失败不阻塞 015**）。错误走 `cann::Error → LaunchError` 白名单（002 表；aclnnStatus 非 0 → Fatal；码族真机核定后回填）。
- **A2 权重就位=host 侧解量化**：q8_0 用 001 CPU codec 一次性解量化（单乘语义；恒等于参考——位精确无竞态）→ fp16 host（**f32→f16 = RNE 单次舍入——r2 钉死：与 014 核内 `__float2half` 同语义，跨端字节一致前提**）→ `cann::DeviceBuffer` H2D；F16 直拷。布局：per-tensor 整块 buffer（一次性预处理，不进 decode 热路径——差异注记：014 为 device 侧 kernel 解量化）。
- **A3 GEMM 判据=候选矩阵（α 实测定稿；r2 修正）**：

  | 档 | 输入/输出 | cubeMathType | 候选判据 |
  |---|---|---|---|
  | G1 | f16-in / f32-out | 高精度（0，待 α 复核） | **rtol 1e-4 + atol 1e-6——门禁候选**（r2：r1 的「G-A 若有」删除——档位明确化） |
  | G2 | f32-in / f32-out | 高精度（0） | **rtol 1e-4 + atol 1e-6——门禁候选**（r2：原 rel 1e-5 缺 atol——近零爆表；双条款） |
  | G3 | f16-in / f16-out | 低精度（1） | 记录档（与 014 16F-acc 同款：声明 + parity 兜底） |

  **cubeMathType → i8 映射（r2 钉入契约）：0=HIGH_PRECISION（fp32 累积）、1=LOW_PRECISION（fp16）——α 复核**；α 实测（T2）：每档形状 K=896/1536、K∈1..4096、M∈1..256 × `matmul_ref`（014 T6 交付）diff；**α 报告每档记录实际执行单元（cube/AIV——fp32-in 是否真走 cube 为未知假设，r2）**；输出：档位终版写回 spec/plan（r1→r2 修订条款）；**α 扩展至 Softmax 与整链 prefill（r2：K=1536 顺序归约差与容差同阶教训吸取——Softmax 误差矩阵同样入表，不可过→记录档 + parity 兜底）**。
- **A4 attention 与层循环（r2 增补组件映射）**：
  - prefill = 两段 GEMM（A3 档）+ fp32 中间 + `cann::Softmax(dim=-1)`（mask 由 host 构造上传）→ fp16 输出（RNE）。
  - decode = **row-gather 连续化 + E3 GEMM + Softmax**（gather 用片内 d2d；strided view 零拷贝（aclTensor stride 语义）为记录项——T2 时评估）。
  - **A4-L 层循环组件（r2 定案表）**：

    | 组件 | 实现 | 判据 |
    |---|---|---|
    | RMSNorm ×2/层 + final | `cann::RmsNorm`（**y + rstd 双输出：rstd 分配空白 tensor 仅消费 y —— r2 注记**） | 与 012 `rms_norm_ref` 差分（fp32 out：rtol 1e-4 + atol 1e-6）——α 项 |
    | RoPE | host per-token（012 `rope_ref` fp32 语义）→ H2D 小拷贝 | 与 012 恒等（公式/位旋转一致） |
    | SiLU | host 元素算子 → H2D | 与 CPU naive 恒等 |
    | embedding | host 查表（GGUF）→ H2D 一次性 | 越界 id → LaunchError |
    | vocab GEMM | E3 gemm（G 档） | fp32 logits；012 host sampler 消费 |

    性能注记：每 decode 步 host→device 小拷贝（RoPE/SiLU 输出，K 级字节）——记录档可接受，notes 留痕。
- **A5 Runner 边界（014 T9 单点契约；r2 修订）**：**Backend trait 以 014 T9 首次确立的签名为唯一契约（`load_weights / prefill(ids) / decode_step() / logits()`）——015 只实现不定义**（r2：删除 015 自定签名漂移；005 A-M4 吸收点不变）。seed 注入点=Runner 构造一次 SplitMix64（temp=0 不消费 RNG——014 r2 声明；temp>0 顺序流=012 语义，与 005 纯函数差异随 005 立项记录）。
- **A6 性能记录协议**：T6 复用 014 T10 `gate_throughput.sh`（**014 T10 创建时已参数化 backend/阈值——015 只传参**）：`run <model> --backend ascend --seed 0 -n 512` vs llama-bench（同机同参同量化）；**不设 ≥3× 断言**（015 记录档；notes 四元组）。
- **A7 判据继承表**（015 不重复实现）：E1=014 D1-D4（真模型存档/004 golden/位精确金块）；refs = 012 refs + 014 `matmul_ref`（T6）/`prefill_attn_ref`（T7）/001 `codes::dequantize_q8_0`（单源——r2）；页池 = 014 T8 `crates/memory`（**PageTable + fixture 公开工具归该 crate——r2 跨端复用前提**）；referee = 014 T0；harness = 014 T10 `tests/parity.rs`。**只消费不自产**。

## Module Breakdown

| 模块 | 内容 |
|---|---|
| `crates/ascend/src/{tensor_view,weights}.rs`（新） | `TensorView`：**持有 `cann::DeviceBuffer + dims + dtype`；每次 op 调用经 `view()` 重建 `cann::Tensor` 句柄（r2：对齐 op.rs 所有权消费——owned Tensor 由重建过渡，TensorView 不持有句柄）**；`weights`：q8_0 host-dequant（RNE）→ op buf H2D / f16 直拷 |
| `crates/ascend/src/ops.rs`（新） | `Gemm(cube)`（cann::Matmul 两段式）、`prefill_attention`、`decode_step_gqa`、**`rms_norm`（rstd 分配注记）** |
| `crates/ascend/src/layer.rs`（新，r2） | **层循环组装**（RMSNorm×3/RoPE/SiLU/embedding/vocab GEMM 编排 + host 元素算子） |
| `crates/memory` | 承接 014 T8（页池只消费） |
| `bin/reinfer` | `--backend ascend` 分支 + **实现 014 T9 的 Backend trait**（r2） |
| `scripts/ci/gate_throughput.sh` | 014 T10 参数化后传参复用（015 T6） |
| notes（`bench/notes.md` 增节「Ascend L3 真机证据」） | α 报告（**固定路径——r2：只此一处**）、cube 档实测、gather vs strided、吞吐四元组、确定性范围（含 CANN 版本） |

## Interface Contracts（r2 与 cann-rs op.rs 实际签名对齐）

```rust
// crates/ascend（新；无一 unsafe；默认特性 stub 可编译——r2 约束）
pub struct TensorView { buf: DeviceBuffer, dims: Vec<u64>, dtype: DataType }
impl TensorView {
    fn view(&self) -> Result<cann::Tensor, LaunchError>;   // 重建句柄（op.rs ffi_smoke 模式；owned 消费过渡）
    pub fn from_host(&self, data: &[f32], dims) -> Result<Self, LaunchError>;   // h2d
    pub fn to_host_f32(&self) -> Result<Vec<f32>, LaunchError>;                 // d2h（验证）
}
pub enum CubeMath { HighPrecision, LowPrecision }   // i8: 0 / 1（A3 映射表；α 复核）
pub fn gemm(a: &TensorView, b: &TensorView, out: &TensorView, cube: CubeMath, stream: &Stream) -> Result<(), LaunchError>;
    // 内部：view() 重建 → cann::Matmul::new(a,b,out,cube) → launch
pub fn rms_norm(x: &TensorView, gamma: &TensorView, eps: f64, stream: &Stream) -> Result<TensorView, LaunchError>;
    // rstd 临时 TensorView（空白分配）——r2 注记
pub fn softmax(x: &TensorView, dim: i64, stream: &Stream) -> Result<TensorView, LaunchError>;
pub fn prefill_attention(q,k,v,mask,stream) -> Result<TensorView, ...>;
pub fn decode_step_gqa(q, kv_pages: &PageTable, kv_tensor: &KvBuf, stream) -> Result<TensorView, ...>;
    // PageTable/KvBuf = 014 T8 共享类型（r2 锚定）；GQA 映射 014 D3 公式 + 三例核验
// 层循环（layer.rs）
pub fn forward_layers(&mut self, ids: &[u32] | last: u32) -> Result<Vec<f32>, LaunchError>;  // fp32 logits
// bin：实现 014 T9 Backend（load_weights/prefill/decode_step/logits）——只实现不定义（r2）
```

## Risk Assessment

| Risk | Mitigation |
|---|---|
| aclnn 算子真机失败（0002 未真机验证） | T0 前移最小真机 smoke（Matmul 往返/RmsNorm/Softmax 各 1）——0002 验收面；失败回报 cann-rs（边界条约 §4）；**0002 GE 部分失败不阻塞 015（依赖切割——r2）** |
| F32 cube 档不可达 / 判据过强 | A3 α 实测最高优先级；**α 覆盖 Softmax/整链 prefill（r2）**；判据按实测回写；parity 兜底 |
| host 解量化/每步拷贝开销 | 预处理一次性 + 每步 K 级字节（记录档）；notes 留痕；不做 NPU 自定义核（P2） |
| gather 正确性/D2D 传输误写 | mem_check 复用 + 毒化测试（含 unmasked NaN）+ bound check |
| 014 未完成（依赖面） | 启动条件硬约束：014 达 M4/M5（**r2：引用修正——原文"E7/E8"系 015 自身块号，错引已改**） |
| trait 迁移破坏 014 | 014 T9 单点确立；015/007 只实现（r2）；014 测试回归兜底 |
| 无 SDK/NPU 机编译 | **新模块默认特性 stub 可编译（r2 约束）：不得 build.rs 探测、不得直连 cann-sys**；本地 --exclude reinfer-ascend 纪律 |

## 里程碑（r2）

- M0：依赖核对——011 消费层、cann-rs 0002 状态、**014 交付（数据管道/refs/页池/gate 脚本/parity.rs/referee T0）**、环境前提（CANN 8.5+设备）
- M1（E2）：权重就位（q8_0 host-dequant RNE + H2D）真机 smoke
- M2（E3）：GEMM **+Softmax+整链 prefill** α 实测 → 判据终版（回写 spec/plan）
- M3（E4-E5）：prefill/decode attention 真机 diff + 泄漏/毒化（含 unmasked NaN；GQA 三例）
- M4（E6）：层循环组装 + cli 闭环（200 token；F16/Q8_0；生成语义）
- M5（E7）：记录档报告（一致率/drift/吞吐）+ notes
- M6：文档/状态/feature-list/changelog（002 复活）
