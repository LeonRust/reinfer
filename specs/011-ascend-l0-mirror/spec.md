# Spec: Ascend L0 mirror — same capability set as CUDA L1 (specs/009)

> Status: proposal · Owner: maintainers · Created: 2026-08-27 · Parent: specs/002-ascend-backend (replaces its deferred status with a mirror acceptance)
> 依据：边界条约（boundary.md §2/§4）+ 011 的对称性矩阵；cann-rs 0001 L0 已实现大部分原语。

## Problem Statement

CUDA L1 验证了"可安全消费的运行时基座"模式。昇腾后端必须提供**同构功能集合**（跨端对称既是工程一致性要求，也让引擎层无需分支）；按边界条约，SDK 表面原语归 cann-rs（0001 已交付），**策略/校验/消费语义归 reinfer 共享层**。本 spec 固化对称矩阵与缺口清单，把 002 从"契约文档"升级为"镜像验收"。

## 对称矩阵（CUDA L1 → Ascend）

| CUDA L1 能力 | Ascend 等价物 | 归属 | 状态 |
|---|---|---|---|
| `core::DeviceId` | 同（复用） | reinfer | ✅ 已共享 |
| `CudaContext/device_count/current` | `cann::Context` + device count/set（0001） | cann-rs | ✅ |
| `DeviceInfo`（name/cc/mem/UUID） | `DeviceProps`（SoC/显存/算力；UUID via SoC 信息） | cann-rs | ⚠️ 0001 仅有 SocName——**需补 DeviceProps 泛化** |
| `CudaStream` | `cann::Stream`（0001） | cann-rs | ✅ |
| `CudaEvent`（BlockingSync/同步语义） | `cann::Event`（0001）——**ACL 事件同步天然阻塞 CPU**，无 flag 等价物，语义天然同构 | cann-rs | ✅（语义注记写入 plan） |
| `DeviceBuffer(Send)/HostBuffer` | `cann::DeviceBuffer/HostBuffer`（0001，Send 决策已拍板） | cann-rs | ✅ |
| **`validate_memref`（方向/边界/归属）** | 同（**已上移 `crates/kernels::mem_check`，本 spec 的直接受益者**） | reinfer 共享 | ✅ |
| MemRef + copy/copy_async 安全面 | `crates/ascend` 消费层（consumes cann 原语 + 共享校验） | reinfer | 🔒 待建 |
| 跨设备 D2D | **语义差异**：CANN 用 `aclrtMemcpyPeer`（无"can_access"探测 API）→ 运行时错误码分类（fail-closed 不变） | cann-rs（绑定）+ reinfer（语义） | ⚠️ spec 定义 |
| 错误白名单 | `LaunchError` 共享 + `cann::Error::is_oom/is_recoverable`（0001 已实现，码段表 002/plan） | 共享 + cann-rs | ✅ |
| 测试策略（泄漏/注入/事件状态/真机 smoke） | cann-rs 侧自查（其仓库职责）+ reinfer 跨端集成镜像（008 `can-gpu` 预留） | 双方 | 🔒 002 复活时 |

## Success Metrics（镜像验收）

- **对称性 100%**：上表每一行都有实现锚点；任一"reinfer 消费层"API 形状与 011 plan 契约一致
- **共享校验复用**：`crates/ascend` 的 copy 路径直接调用 `kernels::mem_check`（无 fork）；对应的方向/边界/归属单测与 CUDA 侧共用同一测试用例集（`mem_check` 已是单源）
- **真机镜像 smoke**：昇腾设备上跑与 `tests/smoke.rs` 同构的 5 类用例（设备/往返/事件状态/泄漏/注入）→ 008 `can-gpu`（002 复活时接线）
- **无 NPU 档**：校验/DeviceProps 纯函数测试集 ≥ 与 CUDA 侧同规模（计数闸 ≥7 沿用 009）

## User Stories

1. 作为引擎作者：切后端不切换心智模型——`copy(dst, src, bytes, stream)` 在两端语义一致。
2. 作为 cann-rs 作者：按 0001 已交付的原语接着补 `DeviceProps`；reinfer 侧不动原语。
3. 作为维护者：跨端 diff（CUDA vs Ascend smoke 报告）只需对齐"行为同构"，不核对实现细节。

## Acceptance Criteria

- [ ] 011 plan 契约表（Ascend 消费层 API）经评审（对比 009：同构签名校验）
- [ ] `crates/ascend` 消费层最小实现：`Context/Stream/Event/Buffer/MemRef/copy`（cann-safe 调用）+ `kernels::mem_check` 复用
- [ ] `kernels::mem_check` 有真实调用方（CUDA ✅ + Ascend ✅）——`#[allow(dead_code)]` 全部移除（当前已满足 CUDA 侧）
- [ ] 真机镜像 smoke（昇腾 5 类用例）→ 008 接线表 `can-gpu` 行生效
- [ ] 跨端差异注记：Event 同步（无 flag 差异）、peer 探测语义（ACL 无 CanAccess，错误分类代替）进 002/plan 或本 plan 的"差异注记"节

## Non-Goals

- aclnn 算子 / GE / AscendC（L1/L2 per 0002 与边界条约）；昇腾侧性能 benchmark（reinfer 负责，等 002 复活的阶段）；双卡 peer 真机矩阵（设备供给依赖）

## Constraints

- 单向依赖不变：`crates/ascend` → `cann`（绝不碰 cann-sys）；无效 unsafe 增量
- 自 009 学到的实测纪律：Event 语义/错误路径等**先真机测再定契约**（评审批改留 changelog）
- 昇腾 smoke 前置依赖：NPU 设备/驱动（环境缺失时标记"记录档"不卡）
