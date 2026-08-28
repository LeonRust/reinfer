# Spec: Ascend L3 — single-request full loop（昇腾第一次跑出 token）

> Status: proposal · r2（2026-08-28 四代理评审修订：层循环组件映射/E7 判据重写/trait 单点/命令面）· Owner: maintainers · Parent: specs/002-ascend-backend
> 锚：014（CUDA L3）的同构镜像——判据分层/对拍协议/数据管道按 014 r1/r2 继承；本 spec 定义差异面。
> 前序：011（L0 mirror ✅）+ cann-rs 0002（L1 aclnn 代码在仓、**未真机验证**）+ 014（数据管道/refs/页池/referee 的交付母体）。
> 模型：经 specs/013 resolver（ModelScope 优先，auto 回退 HuggingFace；零模型标识铁律沿用）。
> 环境前提：**CANN 8.5 + NPU 设备**（cann-rs docs/cann-850-catalog.md 锚定）；设备长期缺位时真机 gate 挂「记录档不卡」（011 先例条款）。

## Problem Statement

014 证明 NVIDIA 路径可以端到端；昇腾达成同样的跳跃：**一次真实请求在 NPU 上端到端流出 token**。差异本质：CUDA 侧数值内核自写自验；**昇腾侧不写内核**——GEMM/Softmax/RMSNorm 用 CANN aclnn 算子（cann-rs 0002 已含封装），**RoPE/SiLU/embedding 走 host 侧 per-token 计算**（r2 定案——aclnn 首批仅 3 算子，自写 AscendC 归 P2）。因此 015 是「组装与验证」spec。

## 范围与验收（判据档位=Constraints α 条款：真机实测后定稿）

| 块 | 功能 | 锚 | 验收 gate（r2 提案档） |
|---|---|---|---|
| E1 数据管道 | GGUF 读取/arch/tokenizer/resolver | **继承 014（无重复实现）** | 014 门禁即本块门禁（共享产物） |
| E2 权重就位 | q8_0 **host 侧解量化**（001 codec，单乘 + **f32→f16 RNE 单次舍入**）→ H2D；F16 直拷 | 001 codec + 014 D4 | 解量化=codec 恒等；H2D/D2H 逐字节往返 diff |
| E3 GEMM | `cann::Matmul`（两段式；**不碰 cann-sys**） | cann-rs 0002 | 判据矩阵 r2 档：G1 f16-in/f32-out、G2 f32-in/f32-out=**门禁候选（rtol 1e-4 + atol 1e-6）**；G3 f16-in/f16-out=记录档；**α 实测定稿（含实际执行单元 cube/AIV 记录——r2）** |
| E4 prefill attention | 两段 GEMM（E3）+ fp32 中间 + `cann::Softmax` → fp16 输出 | 014 D7 | 与 `prefill_attn_ref`（014 T7）差分 **|out|≥2^-14 ≤1 ulp（近零 atol 1e-6）**；全行 sum≈1/掩码行 0；**unmasked NaN 用例（仅全 masked 行允许 0——r2）** |
| E5 decode GQA | 页池（`crates/memory`，014 T8 共享）+ **row-gather 连续化**（strided view=记录项）→ E3 GEMM + Softmax | 014 D7/D8 | 随机页表 diff（fixture 复用 014 T8）+ 毒化（**含 unmasked NaN**）+ 泄漏三合一；**GQA 映射 = 014 D3 公式 + 非整除三例 14/2/12/2/5/2 核验（r2 显式继承）**；确定性=同机同产物双跑逐位一致（r2 适配——「无 atomicAdd/固定归约」为 CUDA 语义须删） |
| E6 闭环 | `reinfer run <model> --backend ascend "<prompt>"`（r2 命令面=契约 v2.15；`cli`/`--model` 撤销）；**全层前向组装**（r2：RMSNorm×3/RoPE/SiLU/embedding/vocab GEMM 有实现锚——见「层循环组件映射」） | 014 D8 | 200 token 稳出（无 NaN；temp=0 双跑复现）；**生成语义必备（r2）= 014 T9 同款：EOS 停/`-n` 硬限/logits 全 NaN 显式错误/embedding OOV 防呆/`-t 0` 短路** |
| E7 parity 与记录 | 复用 014 T10 harness（`tests/parity.rs`） | 014 D8 | **r2 判据重写**：硬闸 = ① 组件级（E3/E4/E5 真机 diffs，前表）② 行为级（E6 生成语义）→ **端到端 parity = 记录档**（F16/Q8_0 token 一致率 + logits drift 报告入 notes；**理由：referee=llama.cpp CUDA 16F+flash-attn 与 aclnn 跨厂商实现——「token 100%/≥99.9%/drift ≤1e-4」均不可过（014 实测：16F-acc vs fp32 参考 92-98% 失败），立闸即违反判据可执行性铁律**）；性能=记录档（gen tok/s + ≥1× llama.cpp CPU 记录；**3× 不设闸**——310B 跨架构，理由 014 先例） |

### 层循环组件映射（r2 新增——评审 ①② S1 修正表）

| 组件 | 实现 | 判据 |
|---|---|---|
| 每层 RMSNorm ×2 + final norm | `cann::RmsNorm`（0002 封装；**rstd 为必需输出参——分配空白 tensor，仅消费 y 输出**） | 与 `rms_norm_ref`（012）机器差（fp32 out：rtol 1e-4 + atol 1e-6）——真机 α 项 |
| RoPE | **host per-token（012 `rope_ref` 语义，fp32）→ H2D 小拷贝**（r2 定案：CANN 无 rope 算子；每 decode 步 1×2·head_dim·n_heads 元素，性能记录档可接受） | 与 012 rope_ref 恒等（同位旋转公式一致） |
| SiLU（MLP gate） | **host 元素算子（`x*sigmoid(x)`）→ H2D** | 与 CPU naive 恒等 |
| embedding | host 查表（GGUF vocab blob）→ H2D 一次性 | 越界 id → LaunchError（E6 防呆） |
| vocab GEMM（logits） | E3 gemm（G 档），隐藏→词表维度 | 输出 fp32 logits；softmax 前 sample（012 host 管线） |

## Success Metrics

- 昇腾第一次流式 token：200 token 稳出（无 NaN；temp=0 双跑复现；EOS 停住）
- 数值对拍（**记录档口径，r2**）：F16/Q8_0 token 一致率 + logits 相对漂移（报告，%值入 notes）——**不作通过/不通过的 gate**；gate = E2-E6 判据 + 行为语义
- 性能记录：gen tok/s 与 llama.cpp CPU 同机对拍（≥1× 记录 + notes 四元组）
- 复现包：模型 sha（013 manifest）+ cube 档 + seed + sampler 语义锚点（同 014）

## User Stories

1. 作为作者：断路「昇腾能跑吗」一步判定——真机证据唯一可信（011 同款纪律）。
2. 作为验证者：与 014 共用参考函数体系——后端切换的差异只在记录口径。
3. 作为维护者：3 个首批 aclnn 算子 + host 元素计算满足全闭环——**不新增自写内核面**（AscendC 归 P2）。
4. 作为服务化读者：E6 是 005 昇腾侧载体；Backend trait（014 T9 单点）自此存在。

## Acceptance Criteria

- [ ] E2-E6 真机全绿（α 修订后的终版 + 行为语义）；E7 记录档报告产出
- [ ] **cann-rs 边界**：仅消费 `cann` 安全层（无 cann-sys 引用、无 `unsafe` 增量、**新模块默认特性 stub 可编译**——ci.yml 无 exclude 依赖此路径）
- [ ] 008 接线表新增 `l3-ascend-*` 行（行名分解：T1/T5 `l3-ascend-e2e`、T3/T4 `l3-ascend-attn`、T2 `l3-ascend-alpha`——r2）+ allowlist 登记
- [ ] 模型标识零硬编码（013 铁律；T7 grep gate）
- [ ] feature-list/phase-plan 勾选；changelog 归档（002 复活行）

## Non-Goals（r2 保留）

- **自写内核**：AscendC/TBE 自定义算子（含 RoPE/SiLU 的 NPU 核——host 侧已定案）、GE 图（`cann::graph` 不消费）、FA 类融合
- **006 类性能工程**：graph 桶化/stream overlap；NPU 性能硬闸（记录档）
- **005/003 类服务面**：批处理/HTTP/engine crate；多卡/HCCL（P2）；prewarm
- decode 性能档同位化（row-gather vs 014 smem staged——记录项）
- CANN >8.5 适配；量化变体扩展（仅 Q8_0/F16/FP32——与 014 D2 同集合）
- **端到端 parity 硬闸（r2 定案：跨厂商 16F 数值路径不可承诺——记录档 + 组件/行为双硬闸替代）**

## Constraints

- **判据可执行性（014 r1 铁律继承并强化）**：凡"参考"本机可执行（CPU naive/refs）；GEMM 档位初版=候选矩阵（plan A3），以 α 真机实测输出定稿（不可过即降级记录档——**含 Softmax/整链 prefill——r2：α 不只测 GEMM**）；「3× 不设闸」为铁律直接推论。
- **单向依赖**：crates/ascend → cann（safe 层）；unsafe 增量零；新模块默认特性 stub 可编译（r2）。
- **模型标识零硬编码**（013）；代码无模型常量；测试用虚构/存档（env 注入）。
- **llama.cpp 对拍协议 = 014 全文继承**（temp=0、golden ids、f280b2698 referee=T0、禁 torch）。
- **真机纪律**：--test-threads=1、DEVICE_ID=0、同机同产物同 launch 配置；**跨 CANN 版本/驱动不承诺（r2：确定性维度加入版本号）**；设备缺位 → 记录档不卡（011 先例）。
- **环境前提**：CANN 8.5 + NPU 设备（缺失 = 真机项全部转记录档，可执行判据项照跑）。
