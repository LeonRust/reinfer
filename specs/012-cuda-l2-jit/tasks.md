# Tasks: CUDA L2 — JitCache v1 & Jit-tier kernels

> Derived from specs/012-cuda-l2-jit/plan.md · 评审修订 r1（4 代理评审 2026-08-27）
> 依赖：T2←T1；T3←T1；T4←T1,T2,T3；T5←T4；T6←T5；T7←T4；T8←T4,T5,T6,T7；T9←T8。T1/T2/T3 可并行。按编号顺序推进。
> 提交拆分（宪法小提交，英文 + 无 AI trailer）：T1/T3→`feat(jit): ...`；T2/T4/T5/T6→`feat(cuda): ...`；T7→`feat(kernels): ...`；T8/T9→`ci: ...`/`docs: ...`。

## T1: `crates/jit` — JitKey/JitCache 实体（平台无关；评审 r1）

- `crates/jit/src/lib.rs` 翻转为 `#![forbid(unsafe_code)]`（现为 allow 残留；锁用 safe wrapper：rustix 或 fs2，决策记录依赖选择）
- `JitKey`（r1 编码：前缀连接 + sha256；source ⊕ headers 内容哈希列表（按路径排序后取哈希再排序，路径不入键；不可读条目→报错不占位）⊕ flags 原始顺序 ⊕ toolchain 版本行 + realpath + `-ccbin` ⊕ capability ⊕ triple）；`JitCache::{open,key,try_load,lock,store,remove,build_once}`；提交顺序 .cubin 先/meta 后（meta 为提交点）；temp 同目录 + 失败清理 + open 清扫；锁目录 `<cache>/locks/` + NB + 超时（300s 可配）；`JLibMeta` sha256 校验
- Verification: `cargo test -p reinfer-jit`；**计数闸 ≥8（009 模式：具名清单 + `cargo test -p reinfer-jit -- --list | grep -c ': test$' ≥ 8`，禁止"空跑绿"措辞）**。具名清单：键注入任一元素变更即失配 / 键与头路径无关（同内容不同路径同键）/ flags 顺序敏感（交换 `-I` 顺序键不同）/ 原子写（提交点后产物与 meta 一致；失败无残留）/ 锁互斥（两线程并发 store 串行）/ 锁超时含 key 错误 / build_once 删重建恰一次 ×2 失败上抛不循环 / meta 损坏或 .cubin 坏字节 → try_load 判 miss → 重建一次 / open 清扫残留 temp

## T2: `crates/jit` — nvcc 工具链（解析/梯度/编译子进程）

- nvcc 解析链 `REINFER_CUDA_NVCC` → `CUDA_HOME` → `CUDA_PATH` → `PATH`；版本行提取；`ToolchainId{ver_line, realpath, ccbin}`；梯度检查（sm_90a≥12.3 / sm_100a≥12.8 / **sm_120a≥12.8**）；`nvcc -cubin` 编译子进程（flags 最终展开、`-gencode` 全数组）；nvcc 缺失 → `LaunchError::Fatal` 专用消息
- Verification（CPU 侧单测，判定机）：nvcc 缺失 → `Fatal`（专用消息）；版本过旧（构造 <=12.6 检出）→ `Fatal`（专用消息）；版本满足 → 编译分支进入；解析链 env 优先于 PATH；`-ccbin` 参与 ToolchainId。**工具链探测与编译链不依赖 GPU**（launch 才属真机档）；三轴版本表（nvcc 判 arch / 驱动判加载 API / cudarc 检测面）写入 plan 注释或 notes

## T3: 离线预烘焙路径

- `REINFER_CUDA_ARCH` **显式指定**（无默认：本 crate 零 CUDA 知识不设设备检测——未设置时 prebake 打印 skip）→ 无 GPU 完整走编译→store→try_load；`REINFER_JIT_CACHE` 目录覆盖；`tests/prebake.rs`
- Verification: `cargo test -p reinfer-jit --test prebake`（需本机 nvcc 且有 toolkit；入口：`REINFER_CUDA_ARCH=sm_120a`）——产物生成 + 同链二次命中 <50ms + key 一致；**跨机命中不承诺**（系统头漂移；如同 arch 无 GPU 机器可用并带 toolkit，可做搬运实验并记录 notes）

## T4: vec_add 链路最小闭环（r1 产物形态）

- `crates/cuda/kernels/vec_add.cu`（`extern "C" __global__` 导出）；`crates/cuda`：`JLib`（cuLibraryLoadData+cuLibraryGetKernel+CUkernel→CUfunction cast，启动/卸载/生命周期契约：持 `Arc<CudaContext>` 或声明仅 context 存活期；禁止新建 safety-layer context）、Jit provider `matches/base_priority/launch`；差分 harness 雏形（固定 seed）
- Verification: 真机 `CUDA_VISIBLE_DEVICES=0 cargo test -p reinfer-cuda --features cuda --test jit_smoke -- --ignored --test-threads=1` 的 vec_add 用例绿（D7 容差 + bit-exact 同机两次）；编译→再命中 <50ms（记录含或不含 `-M` 校验）；改源码 → key 失配 → 重建；`-M` 构建期漂移校验触发（模拟漏列头）→ 报错有提示

## T5: diff 内核：rms_norm / rope / masked_softmax

- 依 T5-D7 算法（rms_norm/rope f32 累积；masked_softmax online-max；禁 `-use_fast_math`）；各自 CPU 参考（`crates/kernels`）；rms_norm eps=1e-5 与参考同语义；全 masked 行 NaN 防护语义一致
- Verification: 无 GPU 档 CPU 参考单测先行（含零行、全 masked、非 2 次幂长度 96/160）；真机差分 per 核（`head_dim` 64/128、行数随机 1..64；D7：fp32 rtol 1e-5 / atol 1e-7 逐项；**掩码位：掩码一致即匹配，不比较 −inf**）；bit-exact 仅"同机同产物同 grid/block 配置"GPU-vs-GPU

## T6: sampler host 管线（r1 定型）

- `crates/kernels`：SplitMix64 纯函数（`next_u64`/均匀分布）、温度语义（temp=0 → argmax 决定论；temp>0 → logits+噪声采样）、mask 边界（含全 masked → 错误而非 NaN）；确定性序列（同 seed 同输入 → 同序列）
- Verification: 纯函数单测（同 seed 同序列；temp=0 决定论；mask 边界）；**组合差分**：GPU masked_softmax 输出（T5 内核）→ host sampler（同 seed）→ 序列 vs CPU 参考管线（T5 参考 + kernels sampler）同 seed 同序列（真机 ignore 用例）；无新 GPU 采样核（差异注记）

## T7: KernelProvider 选择链最小落地（r1 并入 M2）

- `crates/kernels`：`ProviderTier`（显式 discriminant；CpuRef 不注册运行时链）、`KernelProvider` trait、`OpConfig` 最小型、`select`（tier+priority；全不匹配或仅 CpuRef → **明确 `LaunchError`**，非 panic）、`TuneEntry` 最小结构（op/arch/形状哈希 + 耗时字段；正式 TuneDb 归 006）
- Verification: `cargo test -p reinfer-kernels`：tier 决定顺序（Vendor>Jit>Native 排序正确）；不匹配拒绝；无 provider → Err（消息明确）；CpuRef 调用 `select` 不返回；（T4 内）vec_add 经 select 选择 Jit provider

## T8: 真机验证包 + 008 接线

- `crates/cuda/tests/jit_smoke.rs`（差分/确定性/命中/跨进程锁并发首发/产物损坏重建，`#[ignore]`）+ allowlist `l2-jit` 行 + 008-ci-infra 唯一接线表新增行
- Verification: 判定机 `CUDA_VISIBLE_DEVICES=0 cargo test -p reinfer-cuda --features cuda --test jit_smoke -- --ignored --test-threads=1` 全绿；`scripts/ci/checked-ignores.sh` 通过；并发首发（两进程同时 get_or_build 同 key）只编译一次且结果一致；bench/notes.md 留痕（命令/产物/命中耗时）

## T9: 文档/状态回写（r1 裁决闭环）

- 回写清单：深入设计补充 §1.1/§1.4 档位序（Vendor>Jit>Native，Jit=自有核；Native=CubeCL 保留档）与"Jit=外部 DSL"表述；003 plan D2"与 §1 对齐"失实表述；009 spec/plan changelog（编排归属 r2 澄清：编译子进程在 jit、加载/launch 在 cuda）；feature-list（P1-01/L2 行 + 012 锚）；cuda-phase-plan（L2 功能列表补 prewarm 延后注）；008 接线表
- Verification: 文档一致性检查（评审裁决全部有落点；无残留"对齐 §1.4"式旧表述）；feature-list/phase-plan 勾选 L2 状态

---

Completion gate: T1–T9 accepted；真机全绿 + 差分 D7 判据记录于 notes；评审裁决回写完成。下一片按 phase-plan L3 推进（003 T8–T12 + 004 tokenizer 前置；模型一律 ModelScope 下载）。
