# Spec: CUDA L2 — JitCache v1 & Jit-tier kernels（003 T4/T5 落地）

> Status: proposal · Owner: maintainers · Created: 2026-08-27 · 评审修订：2026-08-27（4 代理评审 r1，见本文件 changelog）
> Parent/锚：specs/009（继承 runtime 基座）· 003 T4/T5-D7 原文 · docs/design/cuda-phase-plan §L2 · review-2026-08-25（A-H1/A-M1/A-M2）

## Problem Statement

L1（009）交付了可安全消费的运行时基座，但还没有任何"自己的计算"。L2 的目标是：把 003 T4（JitCache v1 加固版）与 T5-D7（第一批 diff 内核 + CPU 参考）落地，并打开 KernelProvider 三档选择链中的 **Jit 档第一格**——首次走通"内核源码 → 编译 → 跨进程缓存 → 加载 → launch → 差分验证"全闭环。JitCache 实体是平台无关纯 Rust（`crates/jit`）：AscendC（002 边界条约：AscendC 编译流水线 = reinfer jit）共用同一缓存层；本切片只落共享层 + CUDA 侧接线。

档位语义（本切片内确立，见 R1 裁决）：**Vendor > Jit > Native > CPU 参考**；其中 Jit = 引擎自有 CUDA C++ 经 nvcc 编译（主路径档）；Native（CubeCL）为 CUDA 侧保留档位；CPU 参考仅供差分/显式 opt-in，不注册进运行时选择链。

## Success Metrics

- **JitCache v1 全契约**（003 T4 原文，见关键契约）：键组成、原子写、失败重建一次、跨进程锁 + 双检、`REINFER_CUDA_ARCH` 离线预烘焙、工具链梯度检查、nvcc 缺失专用 Fatal 消息——**契约表是 003 T4 的修订继承（changelog 列明字面差异，含 R4 键组成 / R5 产物形态）**
- **第一次真实计算**：vec_add 链路最小闭环真机差分通过；再叠加 T5-D7 四件套（rms_norm / rope / masked_softmax + sampler host 管线）
- **双档可测**：无 GPU 档（纯 CPU 单测，计数闸 ≥8——009 模式：具名清单 + `--list` 计数，009 为 ≥7）+ 真机档（差分/确定性/命中）；预烘焙路径依赖 nvcc 环境前提（见约束）
- **选择链最小落地**：`KernelProvider` trait + `select` 落到 `crates/kernels`（评审 A-H1）；编译子进程在 `crates/jit`（009 约束）、加载/launch 在 `crates/cuda`（A-M2）

## 关键契约（003 T4 修订继承；字面差异一律见 changelog）

| 项 | 契约 |
|---|---|
| JitKey | 前缀编码连接：sha256(source ⊕ headers 内容哈希列表 ⊕ flags 原始顺序 ⊕ toolchain 版本行 + realpath ⊕ `-ccbin` 编译器 realpath/版本 ⊕ capability 规范串 ⊕ target triple)。headers 以内容哈希列表编码、路径不入键（换机可命中）；flags **保持原始顺序**（`-I`/`-L`/`-include`/`-Xcompiler` 顺序敏感）；元素间长度前缀。`-M` 头闭包为构建期漂移校验（R4） |
| 写盘 | 产物 `.cubin` 先行 temp+rename，**meta.json 为提交点**（最后 rename）；meta 含产物 sha256、key 全字段、toolchain realpath、gencode 全量、created_at；`try_load` 校验 `.cubin` 存在且哈希一致，不符 = miss → 重建；temp 与目标同目录，store 失败即清理，open 时清扫残留 temp |
| 并发 | 按 key 跨进程文件锁：flock `LOCK_NB` 轮询 + 等待上限（`REINFER_JIT_LOCK_TIMEOUT`，默认 300s 可配），超时返回含 key 的明确错误；锁目录默认 `<cache>/locks/`（同一命名空间），`REINFER_JIT_LOCK_DIR` 可覆盖；**store 必须持锁**（签名强制）；build_once = 锁 + 双检 + 至多一次"删 + 重建" |
| 预烘焙 | `REINFER_CUDA_ARCH`（如 `sm_120a`）离线指定目标架构，无 GPU 但需本机有 nvcc；验收目标 = **同工具链同 arch 命中**（跨机受系统头漂移制约，notes 记录，R3）；capability 规范串归一（sm_120），`-a` 后缀仅在内核声明且工具链/设备支持时入键 |
| 工具链梯度 | sm_90a ≥ 12.3 / sm_100a ≥ 12.8 / **sm_120a ≥ 12.8（实测基线）**；nvcc 解析链 `REINFER_CUDA_NVCC` → `CUDA_HOME` → `CUDA_PATH` → `PATH` |
| nvcc 缺失 | `LaunchError::Fatal` 专用消息（**不静默降级 CPU**） |
| 产物形态 | `nvcc -cubin`（非 `-shared`，R5）；内核一律 `extern "C" __global__` 导出（符号名即 KernelSource.name）；加载/卸载在平台侧（加载 API 见 plan，需驱动 ≥ 支持库加载语义，判据见 plan 三轴表） |
| 内核算法 | rms_norm（f32 累积，eps 语义与 CPU 参考一致：llama.cpp 惯例 1e-5）、rope（f32 累积）、masked_softmax（online-max；全 masked 行 = NaN 防护语义，与 CPU 参考一致）、sampler = **host 管线**（SplitMix64 纯函数 + 温度语义 + argmax 决定论；GPU 侧仅产出 masked_softmax logits）；每核在 `crates/kernels` 有 CPU 参考 |
| 错误面 | 磁盘 IO/权限/锁超时 → `Fatal`（fail-closed；重试语义归上层）；磁盘满 → `Oom`；编译失败 → `Fatal` 且消息附 nvcc stderr 尾部；驱动加载失败 → 00 9 白名单分类 |

## User Stories

1. 作为引擎作者：`get_or_build(key, src)` 一次构建，后续所有进程命中同一缓存——冷启动不重复编译。
2. 作为维护者：改内核源码/头内容（`KernelSource.headers` 携带新内容）→ 键失效 → 自动重建；构建时 `-M` 漂移校验拦截"漏列头"。
3. 作为 Ascend 作者：JitCache 直接复用（`toolchain_ver`/`arch` 字段为平台无关命名）；AscendC 提供自己的 KernelSource 与编译后端。
4. 作为 CI 作者：无 GPU 但装 CUDA toolchain 的环境依赖 `REINFER_CUDA_ARCH` 预烘焙路径与 CPU 参考差分，不依赖设备。

## Acceptance Criteria

- [ ] JitCache v1 契约表逐项有实现锚点与单测（含锁互斥、失败重建一次、原子写提交点、产物哈希校验、键稳定性、headers 路径无关）
- [ ] `REINFER_CUDA_ARCH` 预烘焙在带 nvcc 的无 GPU 机器上完整走通（产物生成 + 同链二次命中 <50ms）
- [ ] vec_add 真机差分通过（D7 容差；固定 seed、同机同产物 bit-exact）
- [ ] T5-D7 真机差分通过（rms_norm/rope/masked_softmax GPU 核 + sampler host 管线组合；head_dim 64/128、随机行数 1..64、零行、全 masked 行、非 2 次幂长度）
- [ ] nvcc 缺失 → `LaunchError::Fatal` 专用消息有单测；梯度检查按 12.8 实测基线裁剪
- [ ] KernelProvider/select 单测绿（tier 顺序、不匹配拒绝、无 provider → 明确错误而非 panic；CpuRef 不注册运行时选择链）
- [ ] 008 接线表新增 `l2-jit` 行；真机用例走 `#[ignore]` + allowlist 登记
- [ ] feature-list / phase-plan 勾选 L2 状态；changelog 回写完成（R1-R5 涉及的 009/003/深入设计/边界文）

## Non-Goals

- Vendor 档（FlashInfer cubin / cuDNN）与 cuBLAS GEMM（003 T9）
- Attention（T10/T11，prefill/decode）与 Paged KV
- TuneDb 正式持久化与 autotune（006；本次只留 TuneEntry 最小结构）
- **启动阻塞式 prewarm（003 T4 原文）延至 L3 引擎启动切片（005/引擎首启）——本切片仅离线预烘焙 + 首次懒构建**
- 昇腾 AscendC 本体（共享 JitCache 层除外，验证走 002 节奏）
- 性能基准与 3× CPU 判据（L3/006）
- 过期产物自动清理策略（归 006/运维侧；本切片仅 temp 残留清扫）

## Constraints

- `crates/jit` 零 unsafe（锁用 safe wrapper 如 rustix/fs2）、零 CUDA 依赖；**编译子进程（nvcc/-M/工具链探测）在此**（009 约束；`std::process` 非 CUDA 依赖），编译产物加载/launch 在 `crates/cuda`
- `unsafe`（cudarc 驱动、cuLibraryLoad*/cuLaunchKernel）收敛在 `crates/cuda` 的 Jit provider launch 内部
- 内核源码资产仍在 `crates/cuda/kernels/`（003 约束）；经 `include_str!`/读取进入 KernelSource
- 模型下载纪律（phase-plan 决策）：L3 起一律 **ModelScope（魔搭）**，本切片不涉及模型
- 真机判据：判定机 RTX 5090 sm_120（PATH 上 nvcc 当前为 12.6——见 plan 三轴表与 nvcc 解析链）；核内不写死 SM，能力由 key/检查驱动
- 跟随 009 实测纪律：先真机测再定契约；评审修改留 changelog

## Changelog

- r1（2026-08-27，4 代理评审后）：
  - R1 档位裁决：`Vendor > Jit > Native > CPU 参考`；与深入设计 §1.1/§1.4 冲突（其 Jit=Triton/AscendC 桥接为末档）——以本 spec 为准并回写（T9）
  - R2 边界澄清：crates/jit 含编译子进程（009 约束原样）；crates/cuda 仅加载/launch — 原文 D1 措辞修正
  - R3 预烘焙验收降级"同工具链同 arch 命中"（系统头漂移制约跨机）
  - R4 键组成修订：嵌入内容哈希 + flags 保序 + realpath 链 + 长度前缀；`-M` 降为构建期漂移校验；新增 triple
  - R5 产物形态修订：`nvcc -cubin` + `extern "C" __global__`（实测 `-shared` 管线在 sm_120 判定机不可用）
  - R6 梯度基线实测：sm_120a ≥ 12.8（原文 ≥13.0 为事实错误）
  - R7 sampler 定型为 host 管线（无独立 GPU 采样核；差分对象 pin 组合管线）
  - R8 计数闸操作化（≥8 具名清单）；prewarm 延后声明（Non-Goals）
