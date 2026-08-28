# Tasks: Ascend L3 — single-request full loop

> Derived from specs/015-ascend-l3-single-request/plan.md · r2（2026-08-28 四代理评审修订）
> **实施启动条件（硬依赖）**：014 达 M4/M5（run 流式 + parity harness/tests/parity.rs + gate_throughput.sh + referee T0）；**Backend trait 以 014 T9 单点契约为准（015 只实现不定义）**；cann-rs 0002 aclnn 面交付（GE 面失败不阻塞——依赖切割）。
> **本地编译纪律（r2）**：无 SDK/NPU 机一律 `--exclude reinfer-ascend`（或 `--no-default-features` 桩编译）；**新模块不得直连 cann-sys、不得 build.rs 探测（保持默认特性 stub 可编译——ci.yml 无 exclude 依赖此路径）**。
> 设备缺位：真机项转记录档不卡（011 先例）。提交类型：T0 无 reinfer 提交（仅 notes 记录）；T1-T5→`feat(ascend)`；T6→`ci:`；T7→`docs:`。模型标识零硬编码（013 铁律）。

## T0: cann-rs 算子真机 smoke（0002 验收面，015 前置）

- **命令锚定（r2）**：cann-rs 仓 `cd ../cann-rs && cargo test -p cann --features ffi --test aclnn_smoke -- --ignored --test-threads=1`；**用例文件 `cann/tests/aclnn_smoke.rs` 由 cann-rs 0002 任务落地**（reinfer 不建、不提交——边界条约 §4）；用例= Tensor 创建/销毁 + `Matmul::new+launch`（随机小矩阵近似）+ `RmsNorm::new+launch` + `Softmax::new+launch` 往返
- reinfer 侧仅记录：结果 + 命令/环境（CANN 版本/驱动/板卡）→ **`bench/notes.md`「Ascend L3 真机证据」节（r2：固定路径）**
- **依赖切割注记（r2）**：GE/graph 部分验收失败**不阻塞** 015（015 只消费 aclnn 面）
- Verification: 4/4 通过（notes 记录）；任何失败 → 回报 cann-rs，015 阻塞至 aclnn 面修复

## T1: ascend 权重就位（plan A2）

- `crates/ascend/src/tensor_view.rs`（`TensorView`：DeviceBuffer + dims + dtype；`view()` 重建句柄——**r2 对齐 op.rs owned 消费**）+ `weights.rs`：q8_0 host-dequant（001 codec 单乘 + **f32→f16 RNE**）→ H2D；F16 直拷；`from_host/from_host_f16/to_host_f32`（mem_check 消费：方向/对齐/边界）
- Verification: 单测（h2d/d2h 逐字节往返 diff；对齐断言）+ 真机 smoke（0.5B F16 权重上传+sanity probe；Q8_0 上传后 host-dequant diff 0 ulp；**RNE：0xFFFF 相邻值转换与 `half` crate 一致性单测——r2**）；`#[ignore]`，allowlist `l3-ascend-e2e`

## T2: GEMM/Softmax/整链 α 实测 → 终版判据（plan A3；最高优先级）

- `crates/ascend/src/ops.rs`：`Gemm(cube)` + 3 档参数化 + α 小工具（bin 侧或测试）：**r2 范围扩大 = GEMM G1/G2/G3 × 形状（K=896/1536、K∈1..4096、M∈1..256）+ Softmax + 整链 prefill 差分矩阵** vs `matmul_ref`/`prefill_attn_ref`（014 T6/T7）；**逐档记录：误差表 + 实际执行单元（cube/AIV——r2）**
- 按误差表定终版判据：可立门禁 → 写回 spec/plan A3（r2→r3 修订条款）；不可 → 记录档 + notes 理由
- Verification: α 报告（**固定路径 `bench/notes.md`「Ascend L3 真机证据」节——r2 取消二选一**）+ 档位回写；`#[ignore]` 或脚本，allowlist `l3-ascend-alpha`

## T3: prefill attention（plan A4）

- 两段 GEMM（T2 终版档）+ fp32 中间 + `cann::Softmax(dim=-1)` → fp16 输出（RNE）；掩码 host 构造上传
- Verification: 与 `prefill_attn_ref`（014 T7）seq=1k 随机差分（**|out|≥2^-14 ≤1 ulp；近零 atol 1e-6——r2 条款**）；全行 sum≈1/掩码行 0；**unmasked NaN 注入（仅全 masked 行允许 0——r2）**；`#[ignore]`，allowlist `l3-ascend-attn`

## T4: decode_step_gqa（plan A4；页池共享）

- row-gather（片内 d2d；strided view 评估=T2 阶段）+ GEMM + Softmax；页池 = `crates/memory`（014 T8，不重复实现；**PageTable/fixture 复用之**）
- Verification：随机页表 diff（fixture 复用 014：跨页/首尾部分页/乱序/batch 1..64/kv_len 1..1k）+ 毒化（0xFF/NaN，**含 unmasked NaN 注入**）+ 泄漏三合一（在用==0/空闲==预热/守恒式）+ **GQA 映射 = 014 D3 公式 + 三例核验（14/2、12/2、5/2——r2 显式）** + **确定性（r2 适配）：同机同产物双跑逐位一致（删除「无原子、固定归约」CUDA 语义）；跨 CANN 版本不承诺**；`#[ignore]`，allowlist `l3-ascend-attn`

## T5: 层循环组装 + cli 闭环（plan A4-L/A5；r2 新拆分）

- `crates/ascend/src/layer.rs`：**RMSNorm×3（cann::RmsNorm；rstd 空白分配注记）/RoPE（host per-token 012 语义→H2D）/SiLU（host）/embedding（host 查表）/vocab GEMM（G 档）编排**；bin：`run <model> --backend ascend`（r2 命令面）分支 + **实现 014 T9 的 Backend trait（只实现不定义）**；seed 注入点=Runner 构造（temp=0 不消费）
- **生成语义（r2 必备块，与 014 T9 同款）**：EOS 停/`-n` 硬限/logits 全 NaN 显式错误/embedding 越界 id → 错误/`-t 0` 短路（+`-t 1e-9` 边界）
- Verification: 0.5B F16/Q8_0 各 200 token 稳出（无 NaN；temp=0 双跑复现）；EOS/`-n`/OOV 单测；**014 回归全绿**；`#[ignore]`/脚本，allowlist `l3-ascend-e2e`

## T6: parity 记录档 + 性能记录（plan A6；014 T10 harness 复用）

- parity harness（014 `tests/parity.rs`）指向 `--backend ascend`：F16/Q8_0 token 一致率 + logits drift 报告（**记录档——r2 定位；无通过/不通过 gate**）+ **logits 全量 finite 硬断言（复用 014 harness 先验）**
- 性能：`gate_throughput.sh`（014 T10 参数化产物）传参调用 → gen tok/s + llama.cpp CPU 对拍（同机同参）→ notes 四元组（≥1× 记录；3× 不设闸）
- Verification: 报告产出（一致率%/drift/吞吐/立方档/硬件身份）；notes「Ascend L3 真机证据」归档

## T7: 文档与状态（r2）

- feature-list（002 复活行 + 015 勾选）/phase-plan/CLI 契约 §6.2 后端注记；**008 接线新增 `l3-ascend-e2e/l3-ascend-attn/l3-ascend-alpha` 行（与 job 定义——r2 显式）**；notes 归档
- Verification: **零模型名 grep（014 T11 r2 命令复用）**；评审通过

---

依赖表（r2）：T0（前置；与 014 M5 无依赖——014 实施期间可并行）→ T1→T2→T3→T4（T3/T4 可并行）→ T5←T1,T2,T3,T4（含 T2 判据终版）→ T6←T5+014 T10 产物→ T7←T6。全局前置：014 M4/M5 + 014 T9 trait 单点。
